"""Name-conflict repair merges two clusters that one odd mention keeps apart."""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import s2and.model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, _RustFeaturizerPredictDataset
from s2and.model_pairwise import FastCluster
from s2and.name_conflict_repair import (
    NameConflictRepair,
    has_full_first_name,
    quantize_distances,
    repair_name_conflicts_in_block,
)
from s2and.prediction_state import PredictionState
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset
from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.model_helpers import ConstantDistanceClassifier

# Mentions a0..a5 and b0..b4 are one person. a5 carries a wrong name that the
# name rules call impossible against every b mention. c0..c4 are someone else.
SIGNATURES = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
A, B, C = range(0, 6), range(6, 11), range(11, 16)
ODD = 5
FULL = frozenset(SIGNATURES)


def _square(cross: float = 0.1) -> np.ndarray:
    square = np.full((16, 16), 0.9)
    for group in (A, B, C):
        for i in group:
            for j in group:
                square[i, j] = 0.05
    for i in A:
        for j in B:
            square[i, j] = square[j, i] = cross
    for j in B:
        square[ODD, j] = square[j, ODD] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    return square


def _labels(square: np.ndarray) -> list[int]:
    labels = FastCluster(eps=0.5).fit(squareform(square)).labels_
    assert labels is not None
    return [int(label) for label in labels]


def _repair(square: np.ndarray, repair: NameConflictRepair, **kwargs: Any) -> list[int]:
    codes = quantize_distances(squareform(square))
    assert codes is not None
    return [int(x) for x in repair_name_conflicts_in_block(SIGNATURES, _labels(square), codes, repair, **kwargs)]


def _partition(labels: list[int], signatures: list[str] = SIGNATURES) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for signature, label in zip(signatures, labels, strict=True):
        groups.setdefault(label, set()).add(signature)
    return {frozenset(group) for group in groups.values()}


def test_one_odd_mention_splits_the_person_without_repair() -> None:
    assert _partition(_labels(_square())) == {
        frozenset(SIGNATURES[:6]),
        frozenset(SIGNATURES[6:11]),
        frozenset(SIGNATURES[11:]),
    }


def test_repair_merges_the_two_clusters_and_keeps_the_odd_mention() -> None:
    labels = _repair(_square(), NameConflictRepair(0.2, FULL))
    assert _partition(labels) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}


def test_repair_needs_a_low_average_distance() -> None:
    square = _square(cross=0.3)
    assert _repair(square, NameConflictRepair(0.2, FULL)) == _labels(square)


def test_repair_needs_one_mention_behind_every_impossible_pair() -> None:
    square = _square()
    square[0, 7] = square[7, 0] = LARGE_DISTANCE
    assert _repair(square, NameConflictRepair(0.2, FULL)) == _labels(square)


def test_repair_needs_five_mentions_on_each_side() -> None:
    # Without a4 the first cluster keeps four mentions besides the odd one.
    keep = [i for i in range(16) if i != 4]
    square = _square()[np.ix_(keep, keep)]
    signatures = [SIGNATURES[i] for i in keep]
    labels = _labels(square)
    codes = quantize_distances(squareform(square))
    assert codes is not None
    repaired = repair_name_conflicts_in_block(signatures, labels, codes, NameConflictRepair(0.2, FULL))
    assert list(repaired) == labels


def test_repair_needs_a_full_first_name_on_each_side() -> None:
    square = _square()
    only_b_and_odd = frozenset(SIGNATURES[5:11])
    assert _repair(square, NameConflictRepair(0.2, only_b_and_odd)) == _labels(square)


def test_repair_skips_sides_with_different_orcids() -> None:
    square = _square()
    orcids = {"a0": "0000-0000-0000-0001", "b0": "0000-0000-0000-0002"}
    assert _repair(square, NameConflictRepair(0.2, FULL, orcids)) == _labels(square)
    same = {"a0": "0000-0000-0000-0001", "b0": "0000-0000-0000-0001"}
    assert len(_partition(_repair(square, NameConflictRepair(0.2, FULL, same)))) == 2


def test_repair_counts_the_odd_mention_orcid() -> None:
    # The odd mention stays in the merged cluster, so its ORCID may not differ from the other side's.
    square = _square()
    orcids = {"a5": "0000-0000-0000-0001", "b0": "0000-0000-0000-0002"}
    assert _repair(square, NameConflictRepair(0.2, FULL, orcids)) == _labels(square)


def test_repair_does_not_join_two_required_seed_groups() -> None:
    # Impossible pairs between two claimed profiles look like name conflicts to the kernel.
    square = _square()
    repair = NameConflictRepair(0.2, FULL)
    assert _repair(square, repair, seed_groups={"a5": "P1", "b0": "P2"}) == _labels(square)
    # One group on both sides, or a group on one side only, does not stop the merge.
    assert len(_partition(_repair(square, repair, seed_groups={"a5": "P1", "b0": "P1"}))) == 2
    assert len(_partition(_repair(square, repair, seed_groups={"a0": "P1"}))) == 2


def test_repair_skips_a_forbidden_pair() -> None:
    square = _square()
    assert _repair(square, NameConflictRepair(0.2, FULL), forbidden={"a1": {"b3"}}) == _labels(square)
    assert _repair(square, NameConflictRepair(0.2, FULL), forbidden={"b3": {"a1"}}) == _labels(square)


def test_repair_needs_every_impossible_pair_to_be_a_name_rule_conflict() -> None:
    asked: list[list[tuple[str, str]]] = []

    def rules(answer: bool):
        def forbid(pairs):
            asked.append(sorted(tuple(sorted(pair)) for pair in pairs))
            # One pair is impossible for another reason, such as a seed.
            return [answer or pair != pairs[0] for pair in pairs]

        return forbid

    merged = _repair(_square(), NameConflictRepair(0.2, FULL, name_rules_alone_forbid=rules(True)))
    assert len(set(merged)) == 2
    # The test reads exactly the impossible pairs between the two clusters: a5 against every b mention.
    assert asked == [[("a5", f"b{i}") for i in range(5)]]
    vetoed = _repair(_square(), NameConflictRepair(0.2, FULL, name_rules_alone_forbid=rules(False)))
    assert vetoed == _labels(_square())


@pytest.mark.parametrize(
    ("values", "merges"),
    [
        # True average 0.20074: above the limit, though the coded average reads 0.199.
        ([0.2039] * 15 + [0.1960] * 10, False),
        # True average 0.1996: under the limit, but not certainly so after coding.
        ([0.200001] * 15 + [0.1990] * 10, False),
        ([0.19] * 25, True),
    ],
)
def test_repair_merges_only_when_the_average_is_certainly_under_the_limit(values: list[float], merges: bool) -> None:
    square = _square()
    kept_a = [i for i in A if i != ODD]
    for (i, j), value in zip(itertools.product(kept_a, B), values, strict=True):
        square[i, j] = square[j, i] = value

    assert (len(set(_repair(square, NameConflictRepair(0.2, FULL)))) == 2) is merges


@pytest.mark.parametrize(
    ("first", "full"),
    [("ada", True), ("jean pierre", True), ("j", False), ("j p", False), ("", False), (None, False)],
)
def test_two_initials_are_not_a_full_first_name(first: str | None, full: bool) -> None:
    assert has_full_first_name(first) is full


def test_dataset_facts_parse_raw_first_names_as_the_name_rules_do() -> None:
    # An Arrow-backed dataset leaves normalized names unset; "J. D." and "J.-P." are initials.
    raw_first = {"0": "J. D.", "1": "J.-P.", "2": "Jean-Pierre", "3": "Ada"}
    dataset = SimpleNamespace(
        signatures={
            signature_id: SimpleNamespace(
                author_info_first_normalized=None,
                author_info_first=first,
                author_info_middle="",
                author_info_last="Smith",
                author_info_orcid=None,
            )
            for signature_id, first in raw_first.items()
        }
    )
    model = _model()
    model.name_conflict_repair_max_distance = 0.2

    repair = model_module._name_conflict_repair_from_dataset(model, {"b": list(raw_first) * 3}, cast(Any, dataset))

    assert repair is not None
    assert repair.full_first_names == {"2", "3"}


def test_one_thread_quantization_matches_the_pooled_one() -> None:
    condensed = squareform(_square())
    serial = quantize_distances(condensed, num_threads=1)
    pooled = quantize_distances(condensed, num_threads=2)
    assert serial is not None and pooled is not None
    np.testing.assert_array_equal(serial, pooled)


def test_quantize_returns_none_without_impossible_pairs() -> None:
    assert quantize_distances(np.array([0.1, 0.2, 0.3])) is None
    codes = quantize_distances(np.array([0.0, 0.5, 1.0, LARGE_DISTANCE]))
    assert codes is not None
    assert codes.tolist() == [0, 127, 254, 255]


@pytest.mark.parametrize("other", [1.0001, 5.0, 10 * LARGE_DISTANCE, np.nan, np.inf])
def test_quantize_returns_none_for_a_value_that_is_neither_model_nor_impossible(other: float) -> None:
    assert quantize_distances(np.array([0.1, other, LARGE_DISTANCE])) is None


def test_block_clustering_does_not_repair_across_a_soft_distance_above_one() -> None:
    # Linkage treats 5.0 as a soft "keep apart" distance, so the repair may not read it as a name conflict.
    square = _square()
    for j in B:
        square[ODD, j] = square[j, ODD] = 5.0
    square[0, 11] = square[11, 0] = LARGE_DISTANCE
    labels = _model()._cluster_one_block(
        SIGNATURES,
        squareform(square),
        None,
        build_dummy_dataset("name_conflict_repair"),
        {},
        name_conflict_repair=NameConflictRepair(0.2, FULL),
    )
    assert len(_partition([int(label) for label in labels])) == 3


def test_repair_accepts_two_odd_mentions_but_not_three() -> None:
    # Seven and seven mentions of one person, close to each other.
    signatures = [f"a{i}" for i in range(7)] + [f"b{i}" for i in range(7)]
    labels = [0] * 7 + [1] * 7
    repair = NameConflictRepair(0.2, frozenset(signatures))

    def run(impossible: list[tuple[int, int]]) -> list[int]:
        square = np.full((14, 14), 0.1)
        for i, j in impossible:
            square[i, j] = square[j, i] = LARGE_DISTANCE
        np.fill_diagonal(square, 0.0)
        codes = quantize_distances(squareform(square))
        assert codes is not None
        return [int(x) for x in repair_name_conflicts_in_block(signatures, labels, codes, repair)]

    merged = [0] * 14
    # Two odd mentions in one cluster, then one in each cluster.
    assert run([(i, j) for i in (5, 6) for j in range(7, 14)]) == merged
    assert run([(6, j) for j in range(7, 14)] + [(i, 7) for i in range(7)]) == merged
    assert run([(i, j) for i in (4, 5, 6) for j in range(7, 14)]) == labels


@pytest.mark.parametrize("seed", range(8))
def test_repair_preserves_direct_pair_search_and_bounds_extensions(seed: int) -> None:
    # Random clusters and impossible pairs, checked against a plain search over cluster pairs.
    rng = np.random.default_rng(seed)
    n = 72
    labels = rng.integers(0, 6, size=n)
    square = rng.uniform(0.0, 0.3, size=(n, n))
    square = (square + square.T) / 2
    for odd in rng.choice(n, size=6, replace=False):
        other = rng.choice([c for c in range(6) if c != labels[odd]])
        for j in np.flatnonzero(labels == other):
            square[odd, j] = square[j, odd] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    codes = quantize_distances(squareform(square))
    assert codes is not None
    signatures = [str(i) for i in range(n)]
    repair = NameConflictRepair(0.2, frozenset(signatures))

    expected: list[tuple[float, int, int]] = []
    hard = square >= LARGE_DISTANCE
    for a in range(6):
        for b in range(a + 1, 6):
            members_a, members_b = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
            edges = [(i, j) for i in members_a for j in members_b if hard[i, j]]
            if not edges:
                continue
            nodes = sorted({node for edge in edges for node in edge})
            # The smallest sets of one or two mentions that are in every impossible pair.
            covers = [{x} for x in nodes if all(x in edge for edge in edges)]
            if not covers:
                covers = [{x, y} for x, y in itertools.combinations(nodes, 2) if all(x in e or y in e for e in edges)]
            for cover in covers:
                kept_a = [i for i in members_a if i not in cover]
                kept_b = [j for j in members_b if j not in cover]
                if len(kept_a) < 5 or len(kept_b) < 5:
                    continue
                coded = np.floor(square[np.ix_(kept_a, kept_b)] * 254.999).mean()
                if (coded + 1) / 254.999 <= 0.2:
                    expected.append((float((coded + 0.5) / 254.999), a, b))
    want = labels.copy()
    used: set[int] = set()
    for _average, a, b in sorted(expected):
        if a in used or b in used:
            continue
        used.update((a, b))
        want[labels == b] = a

    got = repair_name_conflicts_in_block(signatures, labels, codes, repair)
    for group in _partition(want.tolist(), signatures):
        assert len({int(got[int(signature)]) for signature in group}) == 1
    baseline_groups = _partition(want.tolist(), signatures)
    for group in _partition(list(got), signatures):
        original = sorted({int(labels[int(signature)]) for signature in group})
        assert len(original) <= 3
        if len(original) <= 2:
            assert group in baseline_groups
            continue
        # Check new triples directly against the full matrix, without the kernel's sums.
        sides = [np.flatnonzero(labels == label).tolist() for label in original]
        edges = [(i, j) for a, b in itertools.combinations(sides, 2) for i in a for j in b if hard[i, j]]
        nodes = sorted({i for edge in edges for i in edge})
        covers = [set(cover) for size in (1, 2) for cover in itertools.combinations(nodes, size)]
        assert any(
            all(i in cover or j in cover for i, j in edges)
            and all(len(set(side) - cover) >= 5 for side in sides)
            and all(
                (
                    np.floor(
                        square[np.ix_([i for i in a if i not in cover], [j for j in b if j not in cover])] * 254.999
                    ).mean()
                    + 1
                )
                / 254.999
                <= 0.2
                for a, b in itertools.combinations(sides, 2)
            )
            for cover in covers
        )


def _three_clusters() -> tuple[np.ndarray, list[int], list[str]]:
    """Keep three close groups apart with the two Vallati-style name errors."""
    square = np.full((18, 18), 0.01)
    for a, b, distance in ((0, 1, 0.15), (0, 2, 0.19), (1, 2, 0.09)):
        square[a * 6 : (a + 1) * 6, b * 6 : (b + 1) * 6] = distance
        square[b * 6 : (b + 1) * 6, a * 6 : (a + 1) * 6] = distance
    square[0, 6:] = square[6:, 0] = LARGE_DISTANCE
    square[6:12, 12] = square[12, 6:12] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    return square, [0] * 6 + [1] * 6 + [2] * 6, [str(i) for i in range(18)]


def _repair_three(square: np.ndarray, **kwargs: Any) -> list[int]:
    _, labels, signatures = _three_clusters()
    codes = quantize_distances(squareform(square), num_threads=1)
    assert codes is not None
    repair = kwargs.pop("repair", NameConflictRepair(0.2, frozenset(signatures)))
    return [int(label) for label in repair_name_conflicts_in_block(signatures, labels, codes, repair, **kwargs)]


def test_repaired_pair_absorbs_a_third_cluster_when_every_original_pair_passes() -> None:
    square, _, _ = _three_clusters()
    assert _repair_three(square) == [1] * 18


def test_extension_considers_other_accepted_covers_of_the_selected_pair() -> None:
    square = np.full((18, 18), 0.01)
    square[:6, 6:12] = square[6:12, :6] = 0.05
    square[:6, 12:] = square[12:, :6] = 0.10
    square[6:12, 12:] = square[12:, 6:12] = 0.15
    # The pair merge scores without 0, but the shared cover is 6 and 12.
    square[0, 6] = square[6, 0] = LARGE_DISTANCE
    square[:12, 12] = square[12, :12] = LARGE_DISTANCE
    square[6, 12:] = square[12:, 6] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    assert _repair_three(square) == [0] * 18


def test_extension_requires_a_direct_eligible_pair_between_every_original_cluster() -> None:
    square, _, _ = _three_clusters()
    square[:6, 12:] = square[12:, :6] = 0.8
    assert _repair_three(square) == [0] * 6 + [1] * 12


def test_extension_limits_the_shared_exceptions_to_two_mentions() -> None:
    square, _, _ = _three_clusters()
    square[0, 12:] = square[12:, 0] = 0.19
    square[1, 12:] = square[12:, 1] = LARGE_DISTANCE
    assert _repair_three(square) == [0] * 6 + [1] * 12


@pytest.mark.parametrize("clean_distance", [0.21, 0.1999], ids=["above-limit", "quantization-uncertainty"])
def test_extension_rechecks_each_pair_after_removing_the_same_exceptions(clean_distance: float) -> None:
    square, _, _ = _three_clusters()
    # Removing the second exception loses a cheap column and raises the first/third mean.
    square[1:6, 12] = square[12, 1:6] = 0.0
    square[1:6, 13:] = square[13:, 1:6] = clean_distance
    assert _repair_three(square) == [0] * 6 + [1] * 12


@pytest.mark.parametrize("veto", ["orcid", "seed", "exclusion", "name-rule-provenance"])
def test_extension_keeps_every_original_pair_veto(veto: str) -> None:
    square, _, signatures = _three_clusters()
    kwargs: dict[str, Any] = {}
    repair = NameConflictRepair(
        0.2,
        frozenset(signatures),
        {"1": "X", "13": "Y"} if veto == "orcid" else {},
        (lambda pairs: [not (a == "0" and int(b) >= 12) for a, b in pairs]) if veto == "name-rule-provenance" else None,
    )
    if veto == "seed":
        kwargs["seed_groups"] = {"1": "X", "13": "Y"}
    if veto == "exclusion":
        kwargs["forbidden"] = {"13": {"1"}}
    assert _repair_three(square, repair=repair, **kwargs) == [0] * 6 + [1] * 12


def test_extension_preserves_another_selected_pair() -> None:
    square = np.full((24, 24), 0.1)
    square[:6, 6:12] = square[6:12, :6] = 0.05
    square[12:18, 18:] = square[18:, 12:18] = 0.06
    # Every original pair qualifies, but the two existing pair merges retain their members.
    for start in (0, 6, 12):
        square[start, start + 6 :] = square[start + 6 :, start] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    codes = quantize_distances(squareform(square), num_threads=1)
    assert codes is not None
    signatures = [str(i) for i in range(24)]
    labels = repair_name_conflicts_in_block(
        signatures, np.repeat(np.arange(4), 6), codes, NameConflictRepair(0.2, frozenset(signatures))
    )
    assert list(labels) == [0] * 12 + [2] * 12


def test_extension_stops_at_three_original_clusters() -> None:
    square, _, _ = _three_clusters()
    four = np.full((24, 24), 0.8)
    four[:18, :18] = square
    four[18:, 18:] = 0.01
    four[6:12, 18:] = four[18:, 6:12] = 0.19
    four[12:18, 18:] = four[18:, 12:18] = 0.195
    four[18, 6:18] = four[6:18, 18] = LARGE_DISTANCE
    np.fill_diagonal(four, 0.0)
    codes = quantize_distances(squareform(four), num_threads=1)
    assert codes is not None
    signatures = [str(i) for i in range(24)]
    labels = repair_name_conflicts_in_block(
        signatures, np.repeat(np.arange(4), 6), codes, NameConflictRepair(0.2, frozenset(signatures))
    )
    assert list(labels) == [1] * 18 + [3] * 6


def _model() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.4),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )


def test_block_clustering_applies_the_repair_and_respects_exclusions() -> None:
    dataset = build_dummy_dataset("name_conflict_repair")
    repair = NameConflictRepair(0.2, FULL)

    def run(**kwargs: Any) -> set[frozenset[str]]:
        # Linkage may overwrite the matrix; each run gets its own.
        labels = _model()._cluster_one_block(SIGNATURES, squareform(_square()), None, dataset, {}, **kwargs)
        return _partition([int(label) for label in labels])

    assert len(run()) == 3
    assert run(name_conflict_repair=repair) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}
    # b3 is excluded from the profile of a1 and sits in another cluster. The repair may not bring it back.
    excluded = run(name_conflict_repair=repair, profile_exclusions={"a1": frozenset({"b3"})})
    assert len(excluded) == 3


@pytest.mark.parametrize(("dont_merge_cluster_seeds", "clusters"), [(True, 3), (False, 2)])
def test_block_clustering_keeps_two_claimed_profiles_apart(dont_merge_cluster_seeds: bool, clusters: int) -> None:
    # a5 belongs to profile P1 and b0 to P2: the impossible pairs between them are seed constraints.
    dataset = build_dummy_dataset("name_conflict_repair_seeds")
    model = _model()
    model.use_default_constraints_as_supervision = True
    model.dont_merge_cluster_seeds = dont_merge_cluster_seeds
    state = PredictionState(cluster_seeds_require={"a5": "P1", "b0": "P2"})

    labels = model._cluster_one_block(
        SIGNATURES,
        squareform(_square()),
        None,
        dataset,
        {},
        prediction_state=state,
        name_conflict_repair=NameConflictRepair(0.2, FULL),
    )

    assert len(_partition([int(label) for label in labels])) == clusters


def _facts_featurizer(name_rule_conflicts: bool = True) -> SimpleNamespace:
    """A featurizer stand-in that reports full first names and one ORCID.

    Its constraint call answers every pair alike: a conflict of the name rules alone, or a seed constraint.
    """
    constraint = model_module._NAME_RULE_CONFLICT if name_rule_conflicts else LARGE_DISTANCE
    return SimpleNamespace(
        name_conflict_facts=lambda: (list(SIGNATURES), {"a0": "0000000218250097"}),
        cluster_seeds_require=lambda: [],
        signature_ids=lambda: list(SIGNATURES),
        get_constraints_matrix_indexed=lambda pairs, *_args, **_kwargs: [constraint] * len(pairs),
    )


@pytest.mark.parametrize("suppress_orcid", [False, True])
def test_rust_featurizer_facts_honor_orcid_suppression(suppress_orcid: bool) -> None:
    model = _model()
    model.suppress_orcid = suppress_orcid

    repair = model_module._name_conflict_repair_from_rust_featurizer(model, {"b": SIGNATURES}, _facts_featurizer())

    assert repair is not None
    assert repair.full_first_names == FULL
    assert dict(repair.orcids) == ({} if suppress_orcid else {"a0": "0000000218250097"})


def test_a_stand_in_dataset_reads_repair_facts_from_its_featurizer() -> None:
    # The Rust routes cluster through a stand-in dataset that has no signatures to read facts from.
    stand_in = _RustFeaturizerPredictDataset(
        cluster_seeds_require={}, cluster_seeds_disallow=set(), signatures={}, rust_featurizer=_facts_featurizer()
    )
    repair = model_module._name_conflict_repair_from_dataset(_model(), {"b": SIGNATURES}, stand_in)
    assert repair is not None and repair.full_first_names == FULL


def test_prediction_from_precomputed_distances_skips_the_repair() -> None:
    # A caller's matrix can hold impossible pairs of its own, which nothing can tell from name conflicts.
    clusters, _ = _model().predict_from_rust_featurizer(
        {"b": SIGNATURES}, _facts_featurizer(), dists={"b": squareform(_square())}
    )
    assert len(clusters) == 3

    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 10, {})
    blocks = {"a smith": [str(i) for i in range(16)]}
    python_clusters, _ = _model().predict(blocks, dataset, dists={"a smith": squareform(_square())})
    assert len(python_clusters) == 3


def test_predict_eps_grid_cuts_as_predict_does_and_repairs() -> None:
    # Signature 5 is the odd mention: its name conflicts with the others under the name rules.
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 10, {})
    blocks = {"a smith": [str(i) for i in range(16)]}
    dists = {"a smith": squareform(_square())}
    model = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.4),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )

    def grid() -> dict[float, set[frozenset[str]]]:
        signatures, ids_by_eps = model.predict_eps_grid(blocks, dataset, dists, [0.3, 0.5])
        partitions: dict[float, set[frozenset[str]]] = {}
        for eps, ids in ids_by_eps.items():
            groups: dict[int, set[str]] = {}
            for signature, cluster in zip(signatures, ids.tolist(), strict=True):
                groups.setdefault(cluster, set()).add(signature)
            partitions[eps] = {frozenset(group) for group in groups.values()}
        return partitions

    # At 0.5 the repair joins the two clusters that the odd mention keeps apart.
    assert len(grid()[0.5]) == 2
    # Without the repair, the grid gives predict's clusters from the same matrices at each EPS.
    model.name_conflict_repair_max_distance = None
    unrepaired = grid()
    for eps in (0.3, 0.5):
        model.cluster_model.eps = eps
        expected, _ = model.predict(blocks, dataset, dists=dists)
        assert unrepaired[eps] == {frozenset(group) for group in expected.values()}
    assert len(unrepaired[0.5]) == 3


def test_repair_does_not_merge_across_pairs_that_the_name_rules_do_not_explain() -> None:
    # The featurizer answers that the impossible pairs are seed constraints, not name conflicts.
    _, ids_by_eps = _model().predict_eps_grid_from_rust_featurizer(
        {"b": SIGNATURES}, _facts_featurizer(name_rule_conflicts=False), {"b": squareform(_square())}, [0.5]
    )
    assert len(set(ids_by_eps[0.5].tolist())) == 3


def _eleven_ada_smiths(tmp_path: Path) -> tuple[ANDData, ArrowDataset]:
    dataset = _dataset([("Ada", "", "Smith")] * 11, {})
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    return dataset, arrow_dataset


def _near_model() -> Clusterer:
    """Every pair scores 0.1, so only impossible pairs keep signatures apart."""
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=ConstantDistanceClassifier(0.1),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )


def test_repair_keeps_disallows_that_only_the_featurizer_holds(tmp_path: Path) -> None:
    _, arrow_dataset = _eleven_ada_smiths(tmp_path)
    ids = [str(i) for i in range(11)]
    disallows = {("5", str(i)) for i in range(6, 11)}
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    ).with_cluster_seeds({}, disallows)
    model = _near_model()
    distances = model.make_distance_matrices_from_rust_featurizer({"block": ids}, featurizer)
    assert int(np.count_nonzero(distances["block"] == LARGE_DISTANCE)) == 5

    def violations(clusters: dict[str, list[str]]) -> int:
        return sum(
            any(left in members and right in members for members in clusters.values()) for left, right in disallows
        )

    # The disallows are not repeated as arguments; the repair must still see them.
    assert violations(model.predict_from_rust_featurizer({"block": ids}, featurizer)[0]) == 0
    assert violations(model.predict_from_rust_featurizer({"block": ids}, featurizer, dists=distances)[0]) == 0
    signatures, ids_by_eps = model.predict_eps_grid_from_rust_featurizer({"block": ids}, featurizer, distances, [0.5])
    grid: dict[str, list[str]] = {}
    for signature, cluster in zip(signatures, ids_by_eps[0.5].tolist(), strict=True):
        grid.setdefault(str(cluster), []).append(signature)
    assert violations(grid) == 0


def test_repair_keeps_explicit_seed_groups_apart_without_default_constraints(tmp_path: Path) -> None:
    _, arrow_dataset = _eleven_ada_smiths(tmp_path)
    ids = [str(i) for i in range(11)]
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    )
    seeds = {"5": "A", **{str(i): "B" for i in range(6, 11)}}
    model = _near_model()
    # Explicit seeds stay in the distances when default constraints are off.
    model.use_default_constraints_as_supervision = False

    clusters, _ = model.predict_from_rust_featurizer({"block": ids}, featurizer, cluster_seeds_require=seeds)

    assert all(not ({"5", "6"} <= set(members)) for members in clusters.values())


def test_python_route_repair_asks_the_dataset_about_name_rules() -> None:
    # Signature 5 carries a conflicting name; signature 6 is kept from it by a seed disallow as well.
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 10, {})
    repair = model_module._name_conflict_repair_from_dataset(
        _model(), {"a smith": [str(i) for i in range(16)]}, dataset
    )
    assert repair is not None and repair.name_rules_alone_forbid is not None
    assert list(repair.name_rules_alone_forbid([("5", "6"), ("0", "6")])) == [True, False]
    dataset.cluster_seeds_disallow = {("5", "6")}
    assert list(repair.name_rules_alone_forbid([("5", "6"), ("5", "7")])) == [False, True]
    # One required group never keeps its own signatures apart; two different groups do.
    dataset.cluster_seeds_require = {"5": "P", "7": "P", "8": "Q"}
    assert list(repair.name_rules_alone_forbid([("5", "7"), ("5", "8"), ("5", "9")])) == [True, False, True]


def test_rust_route_name_rule_test_reads_the_featurizer_seeds(tmp_path: Path) -> None:
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 5, {})
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    ids = [str(i) for i in range(11)]
    plain = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    )

    def answers(featurizer: Any, pairs: list[tuple[str, str]]) -> list[bool]:
        repair = model_module._name_conflict_repair_from_rust_featurizer(_model(), {"block": ids}, featurizer)
        assert repair is not None and repair.name_rules_alone_forbid is not None
        return list(repair.name_rules_alone_forbid(pairs))

    # Signature 5 is "Alan": its name conflicts with every "Ada"; two "Ada" signatures do not conflict.
    assert answers(plain, [("5", "6"), ("0", "6")]) == [True, False]
    assert answers(plain.with_cluster_seeds({}, {("5", "6")}), [("5", "6"), ("5", "7")]) == [False, True]
    seeded = plain.with_cluster_seeds({"5": "P", "7": "P", "8": "Q"}, set())
    assert answers(seeded, [("5", "7"), ("5", "8"), ("5", "9")]) == [True, False, True]


def test_eps_grid_from_a_rust_featurizer_repairs_as_its_predict_does() -> None:
    blocks = {"b": SIGNATURES}
    model = _model()

    signatures, ids_by_eps = model.predict_eps_grid_from_rust_featurizer(
        blocks, _facts_featurizer(), {"b": squareform(_square())}, [0.3, 0.5]
    )

    assert signatures == SIGNATURES
    # At 0.5 the repair joins the two clusters that the odd mention keeps apart.
    assert len(set(ids_by_eps[0.5].tolist())) == 2
    # Without the repair, the grid gives the clusters of prediction from the same matrices.
    model.name_conflict_repair_max_distance = None
    _, unrepaired = model.predict_eps_grid_from_rust_featurizer(
        blocks, _facts_featurizer(), {"b": squareform(_square())}, [0.3, 0.5]
    )
    for eps in (0.3, 0.5):
        model.cluster_model.eps = eps
        expected, _ = model.predict_from_rust_featurizer(
            blocks, _facts_featurizer(), dists={"b": squareform(_square())}
        )
        assert _partition(unrepaired[eps].tolist()) == {frozenset(group) for group in expected.values()}


def test_eps_grid_from_a_rust_featurizer_refuses_required_seeds() -> None:
    seeded = SimpleNamespace(name_conflict_facts=lambda: ([], {}), cluster_seeds_require=lambda: [("a0", "p1")])

    with pytest.raises(ValueError, match="cluster_seeds_require cannot be used with precomputed dists"):
        _model().predict_eps_grid_from_rust_featurizer({"b": SIGNATURES}, seeded, {"b": squareform(_square())}, [0.5])


def test_eps_sweep_applies_the_repair_to_cached_distances(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.eps_sweep import sweep_eps_on_linking_gold

    monkeypatch.setattr(sweep_eps_on_linking_gold, "_build_arrow_featurizer", lambda *_args: _facts_featurizer())
    monkeypatch.setattr(sweep_eps_on_linking_gold, "_cache_metadata", lambda *_args: {})
    monkeypatch.setattr(sweep_eps_on_linking_gold, "_load_cached_distance", lambda *_args: squareform(_square()))

    blocks, _ = sweep_eps_on_linking_gold._cluster_cached_blocks(  # noqa: SLF001
        args=cast(Any, SimpleNamespace(dataset="d")),
        clusterer=_model(),
        selected_subblocks={"b": SIGNATURES},
        cache_dir=Path("unused"),
        arrow_dataset=cast(Any, SimpleNamespace(generation_id="g")),
        eps_values=[0.5],
    )

    # Linkage alone leaves three clusters; production block clustering repairs them to two.
    assert _partition(blocks["b"].labels_by_eps[0.5]) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}


def test_every_bulk_route_hands_the_same_facts_to_block_clustering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dummy blocks are smaller than a repairable block.
    monkeypatch.setattr(model_module, "MIN_BLOCK_SIZE", 2)
    seen: list[NameConflictRepair | None] = []
    original = Clusterer._cluster_one_block

    def spy(self: Clusterer, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("name_conflict_repair"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Clusterer, "_cluster_one_block", spy)
    dataset = build_dummy_dataset("name_conflict_repair_route")
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    blocks = {"a": ["0", "1", "2", "3"]}

    off = _model()
    off.name_conflict_repair_max_distance = None
    plain, _ = off.predict_from_arrow(blocks, arrow_dataset, name_tuples=frozenset())
    assert seen == [None]

    # The repair is on by default.
    model = _model()
    from_arrow, _ = model.predict_from_arrow(blocks, arrow_dataset, name_tuples=frozenset())
    from_python, _ = model.predict(blocks, dataset)
    assert from_arrow == plain
    assert sorted(map(sorted, from_python.values())) == sorted(map(sorted, plain.values()))
    arrow_facts, python_facts = seen[1:]
    assert arrow_facts is not None and python_facts is not None
    assert arrow_facts.max_average_distance == python_facts.max_average_distance == 0.2
    # The Rust featurizer also holds the signatures of the block's other papers' authors.
    assert arrow_facts.full_first_names & set(blocks["a"]) == python_facts.full_first_names
    assert python_facts.full_first_names
    assert {k: v for k, v in arrow_facts.orcids.items() if k in blocks["a"]} == dict(python_facts.orcids)


def test_incremental_prediction_repairs_when_it_clusters_new_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_module, "MIN_BLOCK_SIZE", 2)
    seen: list[NameConflictRepair | None] = []
    original = Clusterer._cluster_one_block

    def spy(self: Clusterer, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("name_conflict_repair"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Clusterer, "_cluster_one_block", spy)
    dataset = _dataset([("Ada", "", "Smith"), ("Ada", "", "Smith"), ("Ada", "", "Smith")], {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    model = _model()
    # A linker that links nothing leaves both new mentions for block clustering.
    attach_tiny_incremental_linker(model, tmp_path / "linker", link=False)
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        model.predict_incremental_from_arrow(
            ["1", "2"], arrow_dataset, cluster_seeds_require={"0": "seed"}, name_tuples=frozenset()
        )
    assert len(seen) == 1 and seen[0] is not None
    assert seen[0].full_first_names >= {"1", "2"}
