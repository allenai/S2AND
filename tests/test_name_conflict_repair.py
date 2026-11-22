"""Name-conflict repair merges two clusters that one odd mention keeps apart."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import s2and.block_corrections as block_corrections
import s2and.name_conflict_repair as name_conflict_repair_module
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, _RustFeaturizerPredictDataset
from s2and.model_pairwise import FastCluster
from s2and.name_conflict_repair import (
    REPAIR_ROUNDS,
    NameConflictRepair,
    has_full_first_name,
    repair_name_conflicts_in_block,
)
from s2and.pair_codes import quantize_distances
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


def test_repair_merges_the_two_clusters_and_keeps_the_odd_mention() -> None:
    square = _square()
    labels = _labels(square)
    assert _partition(labels) == {
        frozenset(SIGNATURES[:6]),
        frozenset(SIGNATURES[6:11]),
        frozenset(SIGNATURES[11:]),
    }

    codes = quantize_distances(squareform(square))
    assert codes is not None
    repaired = repair_name_conflicts_in_block(SIGNATURES, labels, codes, NameConflictRepair(0.2, FULL))
    assert _partition(list(repaired)) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}


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


def _half_close_square() -> np.ndarray:
    """A third of the clean cross pairs are close (0.01), the rest 0.8: a plain average of 0.516.

    Each side is tight (0.0), so linkage forms both sides before any cross pair, and the odd
    mention then keeps them apart.
    """
    square = _square()
    for group in (A, B):
        for i in group:
            for j in group:
                if i != j:
                    square[i, j] = 0.0
    kept_a = [i for i in A if i != ODD]
    for k, (i, j) in enumerate(itertools.product(kept_a, B)):
        square[i, j] = square[j, i] = 0.01 if k % 3 == 0 else 0.8
    return square


def test_close_pair_height_merges_what_the_plain_average_keeps_apart() -> None:
    square = _half_close_square()
    assert _repair(square, NameConflictRepair(0.2, FULL)) == _labels(square)
    # 9 of 25 pairs are close: the height is about 0.52 - 8 * 0.36, far under the limit.
    repair = NameConflictRepair(0.2, FULL, close_pair_bonus=8.0, close_pair_distance=0.05)
    assert _partition(_repair(square, repair)) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}


def test_close_pair_height_counts_only_codes_certainly_under_the_close_distance() -> None:
    square = _half_close_square()
    kept_a = [i for i in A if i != ODD]
    for k, (i, j) in enumerate(itertools.product(kept_a, B)):
        if k % 3 == 0:
            # Code 12 holds [0.0471, 0.0510): it may lie above 0.05, so it does not count.
            square[i, j] = square[j, i] = 0.0499
    repair = NameConflictRepair(0.2, FULL, close_pair_bonus=8.0, close_pair_distance=0.05)
    assert _repair(square, repair) == _labels(square)


def _bounded(floor: float | None, cap: float | None, min_side: int | None = None) -> NameConflictRepair:
    return NameConflictRepair(
        0.2,
        FULL,
        close_pair_bonus=8.0,
        close_pair_distance=0.05,
        bonus_min_close_share=floor,
        bonus_max_plain_average=cap,
        bonus_min_side_mentions=min_side,
    )


def _code_sum(plain_bound: float, count: int) -> float:
    """Code sum whose plain-average upper bound is ``plain_bound``."""
    return count * (plain_bound * 254.999 - 1)


@pytest.mark.parametrize(
    ("plain", "close", "passes"),
    [
        # The bonus carries a pair over the limit only with a 10% close share ...
        (0.30, 10, True),
        (0.30, 9, False),
        # ... and a plain average under the cap, however many pairs are close.
        (0.36, 50, False),
        (0.35, 50, True),
        # A pair under the limit on its plain average needs no close pair.
        (0.15, 0, True),
    ],
)
def test_bonus_bound_limits_what_the_bonus_can_carry(plain: float, close: int, passes: bool) -> None:
    bound = name_conflict_repair_module._height_bound(_code_sum(plain, 100), close, 100, _bounded(0.10, 0.35))
    assert (bound <= 0.2) is passes


def test_bonus_bound_off_keeps_the_close_pair_height() -> None:
    repair = _bounded(None, None)
    assert name_conflict_repair_module._height_bound(_code_sum(0.30, 100), 2, 100, repair) == pytest.approx(
        0.30 - 8.0 * 0.02
    )


def test_bonus_bounds_apply_in_block_repair() -> None:
    # Nine of 25 clean pairs are close and the plain average is about 0.52.
    square = _half_close_square()
    labels = _labels(square)
    codes = quantize_distances(squareform(square))
    assert codes is not None
    for repair, merges in (
        (_bounded(0.10, None), True),
        (_bounded(0.40, None), False),
        (_bounded(None, 0.35), False),
        (_bounded(0.10, None, 6), False),
    ):
        repaired = repair_name_conflicts_in_block(SIGNATURES, labels, codes, repair)
        expected = {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])} if merges else _partition(labels)
        assert _partition(list(repaired)) == expected, repair


@pytest.mark.parametrize(
    ("floor", "cap", "min_side"),
    [(-0.1, None, None), (None, 0.0, None), (None, None, 0)],
)
def test_repair_settings_reject_an_invalid_bonus_bound(
    floor: float | None, cap: float | None, min_side: int | None
) -> None:
    with pytest.raises(ValueError, match="bonus_m"):
        _bounded(floor, cap, min_side)


@pytest.mark.parametrize(("smaller_side", "passes"), [(9, False), (10, True)])
def test_bonus_bound_needs_enough_mentions_on_each_side(smaller_side: int | None, passes: bool) -> None:
    repair = _bounded(0.10, 0.35, min_side=10)
    bound = name_conflict_repair_module._height_bound(_code_sum(0.30, 100), 20, 100, repair, smaller_side)
    assert (bound <= 0.2) is passes
    # A pair under the limit on its plain average keeps the repair's own minimum.
    assert name_conflict_repair_module._height_bound(_code_sum(0.15, 100), 0, 100, repair, 5) <= 0.2


def test_repair_settings_follow_the_cluster_model_and_the_module_constants() -> None:
    cluster_model = SimpleNamespace(close_pair_bonus=8.0, close_pair_distance=0.05)
    # An object without the repair attribute gets the same settings as the class default.
    bare = SimpleNamespace(cluster_model=cluster_model, name_conflict_repair_max_distance=0.2)
    with_class_defaults = type("WithClassDefaults", (Clusterer,), {"cluster_model": cluster_model})
    for source in (bare, with_class_defaults):
        settings = block_corrections._repair_settings(source)
        assert settings is not None
        assert settings["max_average_distance"] == 0.2
        assert settings["close_pair_bonus"] == 8.0
        assert settings["rounds"] == REPAIR_ROUNDS == 5
        assert (
            settings["bonus_min_close_share"],
            settings["bonus_max_plain_average"],
            settings["bonus_min_side_mentions"],
        ) == (0.10, 0.40, 10)
    # A cluster model without a close-pair bonus keeps the plain average.
    plain = SimpleNamespace(
        cluster_model=SimpleNamespace(close_pair_bonus=0.0, close_pair_distance=0.0),
        name_conflict_repair_max_distance=0.2,
    )
    settings = block_corrections._repair_settings(plain)
    assert settings is not None and "close_pair_bonus" not in settings
    plain.name_conflict_repair_max_distance = None
    assert block_corrections._repair_settings(plain) is None


@pytest.mark.parametrize(("bonus", "distance"), [(float("nan"), 0.05), (8.0, 0.0)])
def test_repair_settings_reject_an_invalid_close_pair_rule(bonus: float, distance: float) -> None:
    with pytest.raises(ValueError, match="close_pair"):
        NameConflictRepair(0.2, FULL, close_pair_bonus=bonus, close_pair_distance=distance)


def test_repair_orcid_tie_does_not_depend_on_mention_order() -> None:
    square = _square()
    split = _labels(square)
    assert len(set(_repair(square, NameConflictRepair(0.2, FULL, {"a0": "x", "a1": "y", "b0": "x"})))) == 2
    assert len(set(_repair(square, NameConflictRepair(0.2, FULL, {"a0": "y", "a1": "x", "b0": "x"})))) == 2
    assert _repair(square, NameConflictRepair(0.2, FULL, {"a0": "x", "a1": "y", "b0": "z"})) == split


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
    [("jean pierre", True), ("j p", False), (None, False)],
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

    repair = _repair_from_dataset(model, {"b": list(raw_first) * 3}, cast(Any, dataset))

    assert repair is not None
    assert repair.full_first_names == {"2", "3"}


def test_one_thread_quantization_matches_the_pooled_one() -> None:
    condensed = squareform(_square())
    serial = quantize_distances(condensed, num_threads=1)
    pooled = quantize_distances(condensed, num_threads=2)
    assert serial is not None and pooled is not None
    np.testing.assert_array_equal(serial, pooled)


def test_quantize_can_require_an_impossible_pair() -> None:
    condensed = np.array([0.0, 0.5, 1.0])
    assert quantize_distances(condensed) is None
    np.testing.assert_array_equal(quantize_distances(condensed, require_hard=False), [0, 127, 254])
    np.testing.assert_array_equal(quantize_distances(np.append(condensed, LARGE_DISTANCE)), [0, 127, 254, 255])


def test_quantize_reads_float32_and_strided_matrices_as_float64() -> None:
    # Linkage accepts these layouts, so the repair and the year-gap split must see them too.
    condensed = np.array([0.0, 0.5, 1.0, LARGE_DISTANCE])
    for layout in (condensed.astype(np.float32), np.repeat(condensed, 2)[::2]):
        codes = quantize_distances(layout)
        assert codes is not None
        assert codes.tolist() == [0, 127, 254, 255]


def test_quantize_reads_integer_distances() -> None:
    codes = quantize_distances(np.array([0, 1, int(LARGE_DISTANCE)], dtype=np.int64))
    assert codes is not None
    assert codes.tolist() == [0, 254, 255]


@pytest.mark.parametrize(("other", "require_hard"), [(5.0, True), (5.0, False), (np.nan, True)])
def test_quantize_returns_none_for_a_value_that_is_neither_model_nor_impossible(
    other: float, require_hard: bool
) -> None:
    assert quantize_distances(np.array([0.1, other, LARGE_DISTANCE]), require_hard=require_hard) is None


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
        corrections=_corrections(NameConflictRepair(0.2, FULL)),
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


@pytest.mark.parametrize(
    ("seed", "odd_count", "planted"),
    [(0, 6, False), (1, 1, True)],
)
def test_one_repair_round_matches_a_direct_pair_search(seed: int, odd_count: int, planted: bool) -> None:
    # Random clusters and impossible pairs, checked against a plain search over cluster pairs.
    rng = np.random.default_rng(seed)
    n = 72
    labels = rng.integers(0, 6, size=n)
    square = rng.uniform(0.0, 0.3, size=(n, n))
    square = (square + square.T) / 2

    def forbid(odd: int, clusters: list[int]) -> None:
        for j in np.flatnonzero(np.isin(labels, clusters)):
            square[odd, j] = square[j, odd] = LARGE_DISTANCE

    for odd in rng.choice(n, size=odd_count, replace=False):
        forbid(odd, [rng.choice([c for c in range(6) if c != labels[odd]])])
    if planted:
        # Two odd mentions hold three clusters apart, as in `_three_clusters`.
        a, b, c = rng.permutation(6)[:3]
        forbid(rng.choice(np.flatnonzero(labels == a)), [b, c])
        forbid(rng.choice(np.flatnonzero(labels == b)), [c])
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
    assert _partition(list(got), signatures) == _partition(want.tolist(), signatures)


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
    repair = kwargs.pop("repair", NameConflictRepair(0.2, frozenset(signatures), rounds=REPAIR_ROUNDS))
    return [int(label) for label in repair_name_conflicts_in_block(signatures, labels, codes, repair, **kwargs)]


def test_a_later_round_limits_the_odd_mentions_to_two() -> None:
    square, _, _ = _three_clusters()
    square[0, 12:] = square[12:, 0] = 0.19
    square[1, 12:] = square[12:, 1] = LARGE_DISTANCE
    assert _repair_three(square) == [0] * 6 + [1] * 12


@pytest.mark.parametrize(("third_distance", "merges"), [(0.21, True), (0.35, False)])
def test_a_later_round_tests_the_merged_cluster_average(third_distance: float, merges: bool) -> None:
    # Round one joins the second and third clusters (0.09). Round two reads the first cluster
    # against both, without the odd mention 0: 30 pairs at 0.15, 5 at 0.0 and 25 at the third
    # distance. That is 0.1625 at 0.21 and 0.221 at 0.35, as average linkage would see it.
    square, _, _ = _three_clusters()
    square[1:6, 12] = square[12, 1:6] = 0.0
    square[1:6, 13:] = square[13:, 1:6] = third_distance
    assert (len(set(_repair_three(square))) == 1) is merges


@pytest.mark.parametrize("veto", ["orcid", "seed", "exclusion", "name-rule-provenance"])
def test_a_later_round_keeps_every_veto(veto: str) -> None:
    square, _, signatures = _three_clusters()
    kwargs: dict[str, Any] = {}
    repair = NameConflictRepair(
        0.2,
        frozenset(signatures),
        {"1": "X", "13": "Y"} if veto == "orcid" else {},
        (lambda pairs: [not (a == "0" and int(b) >= 12) for a, b in pairs]) if veto == "name-rule-provenance" else None,
        rounds=REPAIR_ROUNDS,
    )
    if veto == "seed":
        kwargs["seed_groups"] = {"1": "X", "13": "Y"}
    if veto == "exclusion":
        kwargs["forbidden"] = {"13": {"1"}}
    assert _repair_three(square, repair=repair, **kwargs) == [0] * 6 + [1] * 12


def _corrections(repair: NameConflictRepair) -> block_corrections.BlockCorrections:
    """Corrections that only repair, with the given settings."""
    return block_corrections.BlockCorrections(repair, None, repair.orcids)


def _repair_from_dataset(model: Any, block_dict: Any, dataset: Any) -> NameConflictRepair | None:
    corrections = block_corrections.corrections_from_dataset(model, block_dict, dataset)
    return None if corrections is None else corrections.repair


def _repair_from_rust_featurizer(model: Any, block_dict: Any, featurizer: Any) -> NameConflictRepair | None:
    corrections = block_corrections.corrections_from_rust_featurizer(model, block_dict, featurizer)
    return None if corrections is None else corrections.repair


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
    assert run(corrections=_corrections(repair)) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}
    # b3 is excluded from the profile of a1 and sits in another cluster. The repair may not bring it back.
    excluded = run(corrections=_corrections(repair), profile_exclusions={"a1": frozenset({"b3"})})
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
        corrections=_corrections(NameConflictRepair(0.2, FULL)),
    )

    assert len(_partition([int(label) for label in labels])) == clusters


def _facts_featurizer(name_rule_conflicts: bool = True) -> SimpleNamespace:
    """A featurizer stand-in that reports full first names and one ORCID.

    Its constraint call answers every pair alike: a conflict of the name rules alone, or a seed constraint.
    """
    constraint = block_corrections._NAME_RULE_CONFLICT if name_rule_conflicts else LARGE_DISTANCE
    return SimpleNamespace(
        name_conflict_facts=lambda: (list(SIGNATURES), {"a0": "0000000218250097"}),
        joined_reading_facts=lambda: ([], []),
        cluster_seeds_require=lambda: [],
        signature_ids=lambda: list(SIGNATURES),
        get_constraints_matrix_indexed=lambda pairs, *_args, **_kwargs: [constraint] * len(pairs),
    )


def test_rust_featurizer_facts_honor_orcid_suppression() -> None:
    model = _model()
    model.suppress_orcid = True

    repair = _repair_from_rust_featurizer(model, {"b": SIGNATURES}, _facts_featurizer())

    assert repair is not None
    assert repair.full_first_names == FULL
    assert not repair.orcids


def test_a_stand_in_dataset_reads_repair_facts_from_its_featurizer() -> None:
    # The Rust routes cluster through a stand-in dataset that has no signatures to read facts from.
    stand_in = _RustFeaturizerPredictDataset(
        cluster_seeds_require={}, cluster_seeds_disallow=set(), signatures={}, rust_featurizer=_facts_featurizer()
    )
    repair = _repair_from_dataset(_model(), {"b": SIGNATURES}, stand_in)
    assert repair is not None and repair.full_first_names == FULL
    assert dict(repair.orcids) == {"a0": "0000000218250097"}


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
    # A caller's precomputed matrix can carry impossible pairs unrelated to names.
    clusters, _ = model.predict(blocks, dataset, dists=dists)
    assert len(clusters) == 3
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


def test_rust_route_repair_keeps_required_seeds_and_disallows_apart(tmp_path: Path) -> None:
    _, arrow_dataset = _eleven_ada_smiths(tmp_path)
    ids = [str(i) for i in range(11)]
    disallows = {("5", str(i)) for i in range(6, 11)}
    plain = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    )
    featurizer = plain.with_cluster_seeds({}, disallows)
    model = _near_model()
    distances = model.make_distance_matrices_from_rust_featurizer({"block": ids}, featurizer)
    assert int(np.count_nonzero(distances["block"] == LARGE_DISTANCE)) == 5

    def violations(clusters: dict[str, list[str]]) -> int:
        return sum(
            any(left in members and right in members for members in clusters.values()) for left, right in disallows
        )

    # The disallows are not repeated as arguments; the repair must still see them.
    assert violations(model.predict_from_rust_featurizer({"block": ids}, featurizer)[0]) == 0
    signatures, ids_by_eps = model.predict_eps_grid_from_rust_featurizer({"block": ids}, featurizer, distances, [0.5])
    grid: dict[str, list[str]] = {}
    for signature, cluster in zip(signatures, ids_by_eps[0.5].tolist(), strict=True):
        grid.setdefault(str(cluster), []).append(signature)
    assert violations(grid) == 0

    seeds = {"5": "A", **{str(i): "B" for i in range(6, 11)}}
    # Explicit seeds stay in the distances when default constraints are off.
    model.use_default_constraints_as_supervision = False

    clusters, _ = model.predict_from_rust_featurizer({"block": ids}, plain, cluster_seeds_require=seeds)

    assert all(not ({"5", "6"} <= set(members)) for members in clusters.values())


def test_python_route_repair_asks_the_dataset_about_name_rules() -> None:
    # Signature 5 carries a conflicting name; signature 6 is kept from it by a seed disallow as well.
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 10, {})
    repair = _repair_from_dataset(_model(), {"a smith": [str(i) for i in range(16)]}, dataset)
    assert repair is not None and repair.name_rules_alone_forbid is not None
    assert list(repair.name_rules_alone_forbid([("5", "6"), ("0", "6")])) == [True, False]
    dataset.cluster_seeds_disallow = {("5", "6")}
    assert list(repair.name_rules_alone_forbid([("5", "6"), ("5", "7")])) == [False, True]
    # One required group never keeps its own signatures apart; two different groups do.
    dataset.cluster_seeds_require = {"5": "P", "7": "P", "8": "Q"}
    assert list(repair.name_rules_alone_forbid([("5", "7"), ("5", "8"), ("5", "9")])) == [True, False, True]


def test_arrow_backed_dataset_reads_facts_from_its_rust_featurizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 10
    block = {"a smith": [str(i) for i in range(16)]}
    arrow_backed = build_arrow_training_dataset(_dataset(names, {}), tmp_path / "arrow")
    arrow_backed.cluster_seeds_disallow = {("5", "6")}
    featurizers: list[Any] = []
    original = block_corrections.corrections_from_rust_featurizer

    def spy(*args: Any, **kwargs: Any) -> Any:
        featurizers.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(block_corrections, "corrections_from_rust_featurizer", spy)
    repair = _repair_from_dataset(_model(), block, arrow_backed)
    assert len(featurizers) == 1 and repair is not None and repair.name_rules_alone_forbid is not None
    # The featurizer carries the dataset's seeds: the disallowed pair is no name-rule conflict.
    assert list(repair.name_rules_alone_forbid([("5", "6"), ("5", "7"), ("0", "6")])) == [False, True, False]
    classic = _repair_from_dataset(_model(), block, _dataset(names, {}))
    assert classic is not None
    assert repair.full_first_names & set(block["a smith"]) == classic.full_first_names


def test_rust_route_name_rule_test_reads_the_featurizer_seeds(tmp_path: Path) -> None:
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Alan", "", "Smith")] + [("Ada", "", "Smith")] * 5, {})
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    ids = [str(i) for i in range(11)]
    plain = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    )

    def answers(featurizer: Any, pairs: list[tuple[str, str]]) -> list[bool]:
        repair = _repair_from_rust_featurizer(_model(), {"block": ids}, featurizer)
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
    clusters, _ = model.predict_from_rust_featurizer(blocks, _facts_featurizer(), dists={"b": squareform(_square())})
    assert len(clusters) == 3
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
    seeded = SimpleNamespace(
        name_conflict_facts=lambda: ([], {}),
        joined_reading_facts=lambda: ([], []),
        cluster_seeds_require=lambda: [("a0", "p1")],
    )

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
    monkeypatch.setattr(block_corrections, "MIN_BLOCK_SIZE", 2)
    seen: list[NameConflictRepair | None] = []
    original = Clusterer._cluster_one_block

    def spy(self: Clusterer, *args: Any, **kwargs: Any) -> Any:
        corrections = kwargs.get("corrections")
        seen.append(None if corrections is None else corrections.repair)
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
    monkeypatch.setattr(block_corrections, "MIN_BLOCK_SIZE", 2)
    seen: list[NameConflictRepair | None] = []
    original = Clusterer._cluster_one_block

    def spy(self: Clusterer, *args: Any, **kwargs: Any) -> Any:
        corrections = kwargs.get("corrections")
        seen.append(None if corrections is None else corrections.repair)
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


def _four_way_split() -> tuple[np.ndarray, list[str]]:
    """Four groups of six mentions of one person; each group's last mention conflicts with the others'."""
    signatures = [f"{group}{i}" for group in "wxyz" for i in range(6)]
    square = np.full((24, 24), 0.1)
    for g in range(4):
        square[6 * g : 6 * g + 6, 6 * g : 6 * g + 6] = 0.05
    odd = [6 * g + 5 for g in range(4)]
    for i in odd:
        for j in odd:
            if i != j:
                square[i, j] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    return square, signatures


@pytest.mark.parametrize(("rounds", "clusters"), [(1, 2), (2, 1)])
def test_later_rounds_rejoin_a_person_split_four_ways(rounds: int, clusters: int) -> None:
    square, signatures = _four_way_split()
    labels = _labels(square)
    assert len(set(labels)) == 4
    codes = quantize_distances(squareform(square))
    assert codes is not None
    repair = NameConflictRepair(0.2, frozenset(signatures), rounds=rounds)
    repaired = repair_name_conflicts_in_block(signatures, labels, codes, repair)
    assert len(set(np.asarray(repaired).tolist())) == clusters


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("rounds", 0),
        ("rounds", True),
        ("bonus_min_side_mentions", 1.5),
    ],
)
def test_repair_settings_reject_a_count_that_is_not_a_positive_integer(setting: str, value: Any) -> None:
    with pytest.raises(ValueError, match=setting):
        NameConflictRepair(0.2, FULL, **{setting: value})


def test_bonus_floor_admits_a_share_exactly_at_the_floor() -> None:
    # 0.07 * 100 is 7.000000000000001 in floating point; the share itself compares exactly.
    bound = name_conflict_repair_module._height_bound(_code_sum(0.30, 100), 7, 100, _bounded(0.07, 0.40))
    assert np.isfinite(bound)


# Reading step: a5 is a joined-initials reading ("Ec"), b0 spells it ("Edward C."), and the
# rest of the b mentions are bare names that the reading conflicts with.
def _reading_repair(spellers: frozenset[str] = frozenset({"b0"}), **kwargs: Any) -> NameConflictRepair:
    readings = {"a5": "Jec", **kwargs.pop("readings", {})}
    return NameConflictRepair(0.2, FULL, joined_readings=readings, reading_spellers={"Jec": set(spellers)}, **kwargs)


def test_reading_step_merges_under_the_linkage_cut_when_the_other_cluster_spells_the_reading() -> None:
    # 0.3 fails the odd-mention test (0.2) but is under the 0.5 cut, as linkage would merge it.
    square = _square(cross=0.3)
    assert _repair(square, _reading_repair()) == _labels(square)
    merged = _repair(square, _reading_repair(), cut_height=0.5)
    assert _partition(merged) == {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}


@pytest.mark.parametrize(
    ("cross", "spellers"),
    [
        (0.3, frozenset()),  # no mention in the other cluster spells the reading
        (0.6, frozenset({"b0"})),  # over the linkage cut
    ],
)
def test_reading_step_needs_a_speller_and_a_height_under_the_cut(cross: float, spellers: frozenset[str]) -> None:
    square = _square(cross=cross)
    assert _repair(square, _reading_repair(spellers), cut_height=0.5) == _labels(square)


def test_reading_step_hands_other_conflicts_to_the_odd_mention_tests() -> None:
    # Readings a4 and a5 plus a plain conflict a0-b1: three mentions, over today's two, but
    # once the readings are set aside one odd mention (a0 or b1) explains the rest.
    square = _square(cross=0.1)
    for j in B:
        square[4, j] = square[j, 4] = LARGE_DISTANCE
    square[0, 7] = square[7, 0] = LARGE_DISTANCE
    labels = _labels(square)
    assert _repair(square, NameConflictRepair(0.5, FULL)) == labels
    # A limit of 0.5: the ten set-aside pairs count as far apart (1.0), and 15 pairs at 0.1
    # bring the average after leaving out a0 to 0.46.
    repair = NameConflictRepair(0.5, FULL, joined_readings={"a4": "Jec", "a5": "Jec"}, reading_spellers={"Jec": {"b0"}})
    assert _partition(_repair(square, repair, cut_height=0.5)) == {
        frozenset(SIGNATURES[:11]),
        frozenset(SIGNATURES[11:]),
    }
    # At 0.3 the odd-mention distance test fails: (10 * 1.0 + 15 * 0.3) / 25 is 0.58.
    square_far = square.copy()
    for i in A:
        for j in B:
            if square_far[i, j] != LARGE_DISTANCE:
                square_far[i, j] = square_far[j, i] = 0.3
    assert _repair(square_far, repair, cut_height=0.5) == _labels(square_far)


def test_reading_step_keeps_pairs_the_name_rules_do_not_explain() -> None:
    square = _square(cross=0.3)
    repair = _reading_repair(name_rules_alone_forbid=lambda pairs: [False for _ in pairs])
    assert _repair(square, repair, cut_height=0.5) == _labels(square)


def test_reading_step_counts_set_aside_pairs_as_far_apart() -> None:
    # Every a mention reads "Jec"; b0 spells it and b1..b4 are bare names the readings conflict
    # with. Only the six pairs to b0 carry a distance (0.35, under the cut). Counting the 24
    # set-aside pairs as far apart keeps one speller from joining the two clusters.
    square = _square(cross=0.35)
    for i in A:
        for j in range(7, 11):
            square[i, j] = square[j, i] = LARGE_DISTANCE
    readings = {f"a{i}": "Jec" for i in range(6)}
    repair = NameConflictRepair(0.2, FULL, joined_readings=readings, reading_spellers={"Jec": {"b0"}})
    assert _repair(square, repair, cut_height=0.5) == _labels(square)


@pytest.mark.parametrize("b_label_first", [False, True])
def test_reading_step_takes_the_lowest_height_cover(b_label_first: bool, caplog: pytest.LogCaptureFixture) -> None:
    # Six a and six b mentions. Readings a4 and a5 ("Jec", spelled by b0) conflict with every
    # b mention, and a plain conflict a0-b1 remains. Leaving out a0 gives height 0.48 and
    # leaving out b1 gives 0.40, whichever cluster comes first.
    signatures = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(6)] + [f"c{i}" for i in range(5)]
    a, b, c = range(0, 6), range(6, 12), range(12, 17)
    square = np.full((17, 17), 0.9)
    for group in (a, b, c):
        for i in group:
            for j in group:
                square[i, j] = 0.05
    for i in a:
        for j in b:
            square[i, j] = square[j, i] = 0.3 if j == 7 else 0.1
    for i in (4, 5):
        for j in b:
            square[i, j] = square[j, i] = LARGE_DISTANCE
    square[0, 7] = square[7, 0] = LARGE_DISTANCE
    np.fill_diagonal(square, 0.0)
    labels = [1 if b_label_first else 0] * 6 + [0 if b_label_first else 1] * 6 + [2] * 5
    codes = quantize_distances(squareform(square))
    assert codes is not None
    repair = NameConflictRepair(
        0.5, frozenset(signatures), joined_readings={"a4": "Jec", "a5": "Jec"}, reading_spellers={"Jec": {"b0"}}
    )
    with caplog.at_level(logging.INFO, logger="s2and"):
        merged = repair_name_conflicts_in_block(signatures, labels, codes, repair, cut_height=0.5)
    assert _partition([int(x) for x in merged], signatures) == {frozenset(signatures[:12]), frozenset(signatures[12:])}
    assert "joined-initials readings, height 0.40" in caplog.text
    assert "odd mentions b1" in caplog.text


@pytest.mark.parametrize(("spellers", "merges"), [({"a0"}, True), ({"c0"}, False)])
def test_reading_step_accepts_a_speller_in_the_reading_mentions_own_cluster(spellers: set[str], merges: bool) -> None:
    # a5 reads "Jec" and its own cluster spells it ("Edward C." beside "Ec"); the b cluster holds
    # bare names. A speller in a third cluster does not count.
    square = _square(cross=0.3)
    merged = _repair(square, _reading_repair(frozenset(spellers)), cut_height=0.5)
    expected = {frozenset(SIGNATURES[:11]), frozenset(SIGNATURES[11:])}
    assert (_partition(merged) == expected) is merges


def test_reading_step_skips_a_partner_every_own_cluster_speller_conflicts_with() -> None:
    # a5 reads "Jec" and only a0 spells it, but a0 also conflicts with every b mention. The
    # reading is then not a5's only veto, so the clusters stay apart.
    square = _square(cross=0.3)
    for j in B:
        square[0, j] = square[j, 0] = LARGE_DISTANCE
    assert _repair(square, _reading_repair(frozenset({"a0"})), cut_height=0.5) == _labels(square)


def test_reading_step_keeps_conflicts_the_own_cluster_speller_shares() -> None:
    # a4 and a5 read "Jec" and only a0 spells it ("Ec" beside "Edward C."). b0 and b1 are bare
    # names a0 agrees with; b2..b4 are "Edward D.", which a0 and the readings conflict with.
    # Those conflicts are not the readings' alone. Leaving out a0 would explain the rest only
    # if they were set aside, and no cover of two mentions explains them all.
    square = _square(cross=0.1)
    for i in (0, 4):
        for j in B if i == 4 else B[2:]:
            square[i, j] = square[j, i] = LARGE_DISTANCE
    for j in B[2:]:
        square[ODD, j] = square[j, ODD] = LARGE_DISTANCE
    repair = NameConflictRepair(0.5, FULL, joined_readings={"a4": "Jec", "a5": "Jec"}, reading_spellers={"Jec": {"a0"}})
    assert _repair(square, repair, cut_height=0.5) == _labels(square)
    # With b2..b4 bare names too, the readings are a4 and a5's only veto, and the clusters join.
    for j in B[2:]:
        square[0, j] = square[j, 0] = 0.1
    assert _partition(_repair(square, repair, cut_height=0.5)) == {
        frozenset(SIGNATURES[:11]),
        frozenset(SIGNATURES[11:]),
    }


@pytest.mark.parametrize("setting", ["name_conflict_repair_rounds", "name_conflict_repair_uses_close_pair_bonus"])
def test_a_removed_repair_setting_is_refused(setting: str) -> None:
    # Setting a removed attribute would silently change nothing, so prediction refuses it.
    model = _model()
    setattr(model, setting, 1)
    with pytest.raises(ValueError, match=setting):
        model.predict({"b": SIGNATURES}, build_dummy_dataset("name_conflict_repair"))
