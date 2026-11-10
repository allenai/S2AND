"""Name-conflict repair merges two clusters that one odd mention keeps apart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import s2and.model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.model_pairwise import FastCluster
from s2and.name_conflict_repair import (
    NameConflictRepair,
    quantize_distances,
    repair_name_conflicts_in_block,
)
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


def _partition(labels: list[int]) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for signature, label in zip(SIGNATURES, labels, strict=True):
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


def test_repair_skips_a_forbidden_pair() -> None:
    square = _square()
    assert _repair(square, NameConflictRepair(0.2, FULL), forbidden={"a1": {"b3"}}) == _labels(square)
    assert _repair(square, NameConflictRepair(0.2, FULL), forbidden={"b3": {"a1"}}) == _labels(square)


def test_quantize_returns_none_without_impossible_pairs() -> None:
    assert quantize_distances(np.array([0.1, 0.2, 0.3])) is None
    codes = quantize_distances(np.array([0.0, 0.5, 1.0, LARGE_DISTANCE]))
    assert codes is not None
    assert codes.tolist() == [0, 127, 254, 255]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_repair_matches_a_direct_search(seed: int) -> None:
    # Random clusters and impossible pairs, checked against a plain search over cluster pairs.
    rng = np.random.default_rng(seed)
    n = 60
    labels = rng.integers(0, 6, size=n)
    square = rng.uniform(0.0, 0.3, size=(n, n))
    square = (square + square.T) / 2
    for odd in rng.choice(n, size=4, replace=False):
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
        for b in range(6):
            members_a, members_b = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
            block = hard[np.ix_(members_a, members_b)]
            if a == b or not block.any() or block.any(axis=1).sum() != 1:
                continue
            kept = members_a[~block.any(axis=1)]
            if len(kept) < 5 or len(members_b) < 5:
                continue
            average = (np.floor(square[np.ix_(kept, members_b)] * 254.999).mean() + 0.5) / 254.999
            if average < 0.2:
                expected.append((float(average), a, b))
    want = labels.copy()
    used: set[int] = set()
    for _average, a, b in sorted(expected):
        if a in used or b in used:
            continue
        used.update((a, b))
        want[labels == b] = a

    got = repair_name_conflicts_in_block(signatures, labels, codes, repair)
    assert list(got) == list(want)


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
