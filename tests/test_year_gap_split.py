"""Tests for the year-gap split after name-conflict repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import s2and.model as model_module
from s2and import year_gap_split
from s2and.consts import YEAR_DIFF_CAP
from s2and.data import ANDData
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.model_pairwise import FastCluster
from s2and.name_conflict_repair import quantize_distances
from s2and.year_gap_split import split_year_gaps, split_years
from tests.helpers import build_arrow_training_dataset
from tests.incremental_test_helpers import _dataset
from tests.model_helpers import ConstantDistanceClassifier

# Mentions o0 and o1 are old papers; n0..n9 are one modern career. x is an undated stranger
# whose name conflicts with o0, which keeps the codes non-empty, as in any block the repair reads.
SIGNATURES = ["o0", "o1"] + [f"n{i}" for i in range(10)] + ["x"]
OLD, NEW, X = [0, 1], list(range(2, 12)), 12
# The stranger is its own cluster from the start.
LABELS = [0] * 12 + [1]
# Close-pair distance 0.05: codes up to 11 are certainly close.
CLOSE_CODE = 11


def _square(old_new: float = 0.3) -> np.ndarray:
    square = np.full((13, 13), 0.2)
    for i in OLD:
        for j in NEW:
            square[i, j] = square[j, i] = old_new
    square[X, :] = square[:, X] = 0.9
    square[0, X] = square[X, 0] = 1e4
    np.fill_diagonal(square, 0.0)
    return square


def _codes(square: np.ndarray) -> np.ndarray:
    codes = quantize_distances(squareform(square))
    assert codes is not None
    return codes


def _years(old: int = 1928, new: tuple[int, int] = (2000, 2023)) -> dict[str, int]:
    years = {"o0": old, "o1": old + 2}
    for k, signature_id in enumerate(SIGNATURES[2:]):
        years[signature_id] = new[0] + (new[1] - new[0]) * k // 9
    return years


def _split(square: np.ndarray, years: dict[str, int], **kwargs: Any) -> set[frozenset[str]]:
    return _partition(split_year_gaps(SIGNATURES, LABELS, _codes(square), years, CLOSE_CODE, **kwargs))


def _partition(labels: Any) -> set[frozenset[str]]:
    """Return the clusters other than the stranger's."""
    groups: dict[int, set[str]] = {}
    for signature_id, label in zip(SIGNATURES, np.asarray(labels).tolist(), strict=True):
        groups.setdefault(label, set()).add(signature_id)
    assert {"x"} in groups.values()
    return {frozenset(group) for group in groups.values() if group != {"x"}}


WHOLE = {frozenset(SIGNATURES[:12])}
SPLIT = {frozenset(SIGNATURES[:2]), frozenset(SIGNATURES[2:12])}


def test_split_separates_papers_across_an_empty_stretch_beyond_the_cap() -> None:
    assert _split(_square(), _years()) == SPLIT


def test_split_keeps_the_largest_part_on_the_original_label() -> None:
    labels = split_year_gaps(SIGNATURES, [7] * 12 + [1], _codes(_square()), _years(), CLOSE_CODE)
    assert np.asarray(labels).tolist() == [8, 8] + [7] * 10 + [1]


@pytest.mark.parametrize(("old", "splits"), [(2000 - YEAR_DIFF_CAP, False), (2000 - YEAR_DIFF_CAP - 1, True)])
def test_split_needs_a_gap_longer_than_the_cap(old: int, splits: bool) -> None:
    # o1 is two years after o0, so the gap that counts is from o1 to the first new paper.
    assert (_split(_square(), _years(old - 2)) == SPLIT) is splits


def test_split_leaves_a_continuous_long_career_whole() -> None:
    # 1900 to 2023 with no empty stretch over the cap: a long career, not a gap.
    years = {signature_id: 1900 + 12 * k for k, signature_id in enumerate(SIGNATURES)}
    assert _split(_square(), years) == WHOLE


def test_split_keeps_parts_that_a_close_pair_bridges() -> None:
    # A reprint: the model is certain of one pair across the gap.
    square = _square()
    square[0, 5] = square[5, 0] = 0.01
    assert _split(square, _years()) == WHOLE


def test_split_does_not_count_a_pair_at_the_close_distance_as_close() -> None:
    square = _square()
    square[0, 5] = square[5, 0] = 0.05
    assert _split(square, _years()) == SPLIT


def test_split_keeps_parts_that_share_an_orcid() -> None:
    assert _split(_square(), _years(), orcids={"o0": "x", "n4": "x"}) == WHOLE


def test_split_keeps_parts_that_share_a_required_seed_group() -> None:
    assert _split(_square(), _years(), seed_groups={"o1": "P", "n9": "P"}) == WHOLE


def test_split_keeps_parts_that_share_a_first_name() -> None:
    assert _split(_square(), _years(), first_names={"o1": "ada", "n7": "ada"}) == WHOLE
    assert _split(_square(), _years(), first_names={"o1": "ada", "n7": "alan"}) == SPLIT


def test_split_years_drops_placeholder_and_missing_years() -> None:
    pairs = [("a", 1601), ("b", 1800), ("c", 0), ("d", -5), ("e", 1799), ("f", 1801), ("g", 2020)]
    assert split_years(pairs) == {"e": 1799, "f": 1801, "g": 2020}


def test_undated_mentions_follow_the_closer_part() -> None:
    square = _square()
    years = _years()
    del years["n0"]
    square[2, OLD] = square[OLD, 2] = 0.1
    assert _split(square, years) == {frozenset({"o0", "o1", "n0"}), frozenset(SIGNATURES[3:12])}


def test_undated_mentions_follow_their_seed_group_first() -> None:
    square = _square()
    years = _years()
    del years["n0"]
    square[2, OLD] = square[OLD, 2] = 0.1
    assert _split(square, years, seed_groups={"n0": "P", "n5": "P"}) == SPLIT


def test_undated_mentions_of_one_seed_group_move_together() -> None:
    # Alone, n0 would follow the old part and n1 the new one.
    square = _square()
    years = _years()
    del years["n0"], years["n1"]
    square[2, OLD] = square[OLD, 2] = 0.15
    assert _split(square, years) == {frozenset({"o0", "o1", "n0"}), frozenset(SIGNATURES[3:12])}
    assert _split(square, years, seed_groups={"n0": "G", "n1": "G"}) == SPLIT
    assert _split(square, years, orcids={"n0": "O", "n1": "O"}) == SPLIT


def test_an_undated_mention_tied_to_both_parts_keeps_them_together() -> None:
    # n0 shares an ORCID with the old part and a seed group with the new one.
    years = _years()
    del years["n0"]
    seeds = {"n0": "G", "n5": "G"}
    assert _split(_square(), years, orcids={"o0": "O", "n0": "O"}, seed_groups=seeds) == WHOLE


def test_split_leaves_labels_alone_when_nothing_splits() -> None:
    assert split_year_gaps(SIGNATURES, LABELS, _codes(_square()), {}, CLOSE_CODE) is LABELS


def _model(close_pair_bonus: float) -> Clusterer:
    cluster_model = FastCluster(eps=0.5)
    if close_pair_bonus:
        cluster_model.close_pair_bonus = close_pair_bonus
        cluster_model.close_pair_distance = 0.05
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.4),
        cluster_model=cluster_model,
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )


def _dated_dataset() -> ANDData:
    """Thirteen "Ada Smith" mentions with SIGNATURES' years; signature i is SIGNATURES[i]."""
    dataset = _dataset([("Ada", "", "Smith")] * 13, {})
    for index, signature_id in enumerate(SIGNATURES):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=_years().get(signature_id))
    return dataset


def test_both_routes_read_the_same_years(tmp_path: Path) -> None:
    dataset = _dated_dataset()
    ids = [str(i) for i in range(13)]
    model = _model(8.0)
    python_repair = model_module._name_conflict_repair_from_dataset(model, {"b": ids}, dataset)
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=ids, name_tuples=frozenset(), num_threads=1
    )
    rust_repair = model_module._name_conflict_repair_from_rust_featurizer(model, {"b": ids}, featurizer)

    assert python_repair is not None and rust_repair is not None
    assert (
        dict(python_repair.years) == dict(rust_repair.years) == {str(i): _years()[s] for i, s in enumerate(SIGNATURES)}
    )
    assert dict(python_repair.first_names) == dict(rust_repair.first_names) == dict.fromkeys(ids, "ada")


@pytest.mark.parametrize(("close_pair_bonus", "enabled", "parts"), [(8.0, True, 2), (8.0, False, 1), (0.0, True, 1)])
def test_block_clustering_splits_year_gaps_after_the_repair(close_pair_bonus: float, enabled: bool, parts: int) -> None:
    dataset = _dated_dataset()
    ids = [str(i) for i in range(13)]
    model = _model(close_pair_bonus)
    model.year_gap_split = enabled
    repair = model_module._name_conflict_repair_from_dataset(model, {"b": ids}, dataset)

    labels = model._cluster_one_block(ids, squareform(_square(0.25)), None, dataset, {}, name_conflict_repair=repair)

    assert len(_partition(labels)) == parts


def test_eps_grid_splits_as_block_clustering_does() -> None:
    dataset = _dated_dataset()
    ids = [str(i) for i in range(13)]
    signatures, ids_by_eps = _model(8.0).predict_eps_grid({"b": ids}, dataset, {"b": squareform(_square(0.25))}, [0.5])
    assert signatures == ids
    assert _partition(ids_by_eps[0.5]) == SPLIT


def _cluster(model: Clusterer, dataset: ANDData, square: np.ndarray, **kwargs: Any) -> set[frozenset[str]]:
    ids = [str(i) for i in range(13)]
    repair = model_module._name_conflict_repair_from_dataset(model, {"b": ids}, dataset)
    labels = model._cluster_one_block(ids, squareform(square), None, dataset, {}, name_conflict_repair=repair, **kwargs)
    return _partition(labels)


def test_block_clustering_splits_a_block_without_an_impossible_pair() -> None:
    # No name conflict anywhere, so no repair applies; the split still runs.
    square = _square(0.25)
    square[0, X] = square[X, 0] = 0.9
    assert _cluster(_model(8.0), _dated_dataset(), square) == SPLIT


def test_block_clustering_splits_when_the_repair_uses_the_plain_average() -> None:
    # The 1.0.0b11 repair rule; the split reads the cluster model's close-pair distance.
    model = _model(8.0)
    model.name_conflict_repair_uses_close_pair_bonus = False
    assert _cluster(model, _dated_dataset(), _square(0.25)) == SPLIT


@pytest.mark.parametrize(("seeds_off", "expected"), [(False, WHOLE), (True, SPLIT)])
def test_block_clustering_ignores_seeds_the_caller_turned_off(seeds_off: bool, expected: Any) -> None:
    # The altered-profile pre-split turns seeds off: a seed group o0 + n5 may then split.
    dataset = _dated_dataset()
    dataset.cluster_seeds_require = {"0": "P", "7": "P"}
    labels = _cluster(_model(8.0), dataset, _square(0.25), incremental_dont_use_cluster_seeds=seeds_off)
    assert labels == expected


def test_block_clustering_splits_when_no_code_is_certainly_close() -> None:
    # A close-pair distance under every code leaves the split on, with no pair to bridge a gap.
    model = _model(8.0)
    model.cluster_model.close_pair_distance = 0.001
    assert _cluster(model, _dated_dataset(), _square(0.25)) == SPLIT


def test_mean_cross_code_reads_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    codes = quantize_distances(squareform(_square(0.25)), require_hard=False)
    assert codes is not None
    rows, cols = np.arange(0, 5), np.arange(5, 13)
    expected = float(year_gap_split._cross_codes(codes, 13, rows, cols).mean())
    monkeypatch.setattr(year_gap_split, "_BATCH_CELLS", 9)
    assert year_gap_split._mean_cross_code(codes, 13, rows, cols) == expected


def _small_block_labels(old_first: str, years: list[int]) -> list[int]:
    """Cluster five Smith mentions, two old ones named ``old_first`` and three "Ada", with ``years``."""
    dataset = _dataset([(old_first, "", "Smith")] * 2 + [("Ada", "", "Smith")] * 3, {})
    for index, year in enumerate(years):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=year)
    ids = [str(i) for i in range(5)]
    model = _model(8.0)
    repair = model_module._name_conflict_repair_from_dataset(model, {"b": ids}, dataset)
    square = np.full((5, 5), 0.25)
    np.fill_diagonal(square, 0.0)
    labels = model._cluster_one_block(ids, squareform(square), None, dataset, {}, name_conflict_repair=repair)
    return np.asarray(labels).tolist()


def test_block_clustering_splits_a_block_under_the_repair_size() -> None:
    # Five mentions: too few for the repair, enough for the split.
    labels = _small_block_labels("A.", [1928, 1930, 2001, 2010, 2020])
    assert labels[0] == labels[1] != labels[2] == labels[3] == labels[4]


def test_small_block_keeps_parts_that_share_a_first_name() -> None:
    assert len(set(_small_block_labels("Ada", [1928, 1930, 2001, 2010, 2020]))) == 1


def test_large_block_splits_parts_that_share_a_first_name() -> None:
    # Thirteen "Ada Smith" mentions: in a large block a first name is too common to bridge a gap.
    assert _cluster(_model(8.0), _dated_dataset(), _square(0.25)) == SPLIT


@pytest.mark.parametrize(("old_years", "parts"), [((1601, 1800), 1), ((1799, 1790), 2)])
def test_block_clustering_treats_placeholder_years_as_undated(old_years: tuple[int, int], parts: int) -> None:
    assert len(set(_small_block_labels("A.", [*old_years, 2001, 2010, 2020]))) == parts


def test_short_spans_skip_the_split() -> None:
    years = {"a": 2010 - YEAR_DIFF_CAP - 1, "b": 2010}
    assert model_module._spans_a_year_gap(["a", "b"], years, 11)
    assert not model_module._spans_a_year_gap(["a", "b"], {"a": 2010 - YEAR_DIFF_CAP, "b": 2010}, 11)
    assert not model_module._spans_a_year_gap(["a", "b"], years, None)
