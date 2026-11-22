"""Tests for the year-gap split after name-conflict repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.distance import squareform

import s2and.block_corrections as block_corrections
from s2and import pair_codes, year_gap_split
from s2and.consts import YEAR_DIFF_CAP
from s2and.data import ANDData, NameCounts
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.model_pairwise import FastCluster
from s2and.pair_codes import quantize_distances
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
    years = {"a": 1601, "b": 1800, "c": 0, "d": -5, "e": 1799, "f": 1801, "g": 2020}
    assert split_years(years) == {"e": 1799, "f": 1801, "g": 2020}


def _stamped_years(stamp: int) -> dict[str, int]:
    """Old part 1790-1817, new part 2000-2023, and n0 stamped ``stamp``."""
    years = _years()
    years["o0"], years["o1"], years["n0"] = 1790, 1817, stamp
    return years


@pytest.mark.parametrize(("stamp", "joins_old"), [(1800, True), (1601, False)])
def test_a_placeholder_stamp_follows_the_part_whose_years_span_it(stamp: int, joins_old: bool) -> None:
    # n0 is closer to the new part, so the average code alone would put it there.
    old_with_n0 = {frozenset({"o0", "o1", "n0"}), frozenset(SIGNATURES[3:12])}
    assert (_split(_square(), _stamped_years(stamp)) == old_with_n0) is joins_old


def test_an_undated_mention_bridges_parts_by_first_name() -> None:
    # n0 is undated; its ORCID ties it to the old part and its first name to the new one.
    years = _years()
    del years["n0"]
    orcids = {"n0": "O", "o0": "O"}
    assert _split(_square(), years, orcids=orcids) == {frozenset({"o0", "o1", "n0"}), frozenset(SIGNATURES[3:12])}
    assert _split(_square(), years, orcids=orcids, first_names={"n0": "ada", "n5": "ada"}) == WHOLE


def test_undated_mentions_follow_seed_groups_before_distance() -> None:
    square = _square()
    years = _years()
    del years["n0"]
    square[2, OLD] = square[OLD, 2] = 0.1
    assert _split(square, years) == {frozenset({"o0", "o1", "n0"}), frozenset(SIGNATURES[3:12])}
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


def _dated_dataset(extra_names: tuple[str, ...] = ()) -> ANDData:
    """Thirteen "Ada Smith" mentions with SIGNATURES' years; signature i is SIGNATURES[i]."""
    dataset = _dataset([("Ada", "", "Smith")] * 13 + [(name, "", "Jones") for name in extra_names], {})
    for index, signature_id in enumerate(SIGNATURES):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=_years().get(signature_id))
    return dataset


def test_both_routes_read_the_same_years_and_small_block_names(tmp_path: Path) -> None:
    dataset = _dated_dataset(("Ada", "A.", "Jean-Pierre", "Ömer", "J. Robert", "Li"))
    smiths = [str(i) for i in range(13)]
    joneses = [str(i) for i in range(13, len(dataset.signatures))]
    blocks = {"smith": smiths, "jones": joneses}
    model = _model(8.0)
    python_repair = block_corrections.corrections_from_dataset(model, blocks, dataset)
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset, signature_ids=smiths + joneses, name_tuples=frozenset(), num_threads=1
    )
    rust_repair = block_corrections.corrections_from_rust_featurizer(model, blocks, featurizer)

    assert python_repair is not None and rust_repair is not None
    years = _years()
    expected_years = {str(i): years[s] for i, s in enumerate(SIGNATURES)}
    assert dict(python_repair.years) == dict(rust_repair.years) == expected_years | dict.fromkeys(joneses, 2000)
    assert dict(python_repair.first_names) == dict(rust_repair.first_names)
    assert python_repair.first_names["13"] == "ada"
    assert "14" not in python_repair.first_names
    assert set(python_repair.first_names).isdisjoint(smiths)

    # Count the original surname block when a claimed profile includes only part of it.
    part = {"profile": smiths[:5], "jones": joneses}
    assert block_corrections.dataset_small_block_signatures(part, dataset) == frozenset(joneses)
    with arrow_dataset.use() as lease:
        assert block_corrections.arrow_small_block_signatures(lease, part) == frozenset(joneses)


def test_python_route_reads_a_nan_year_as_undated() -> None:
    dataset = _dated_dataset()
    dataset.papers["100"] = dataset.papers["100"]._replace(year=float("nan"))
    ids = [str(i) for i in range(13)]
    repair = block_corrections.corrections_from_dataset(_model(8.0), {"b": ids}, dataset)
    assert repair is not None
    assert "0" not in repair.years


def test_block_clustering_splits_year_gaps_after_the_repair() -> None:
    dataset = _dated_dataset()
    ids = [str(i) for i in range(13)]
    square = _square(0.25)
    model = _model(8.0)
    assert _cluster(model, dataset, square) == SPLIT
    signatures, labels = model.predict_eps_grid({"b": ids}, dataset, {"b": squareform(square)}, [0.5])
    assert signatures == ids
    assert _partition(labels[0.5]) == SPLIT

    model.year_gap_split = False
    assert _cluster(model, dataset, square) == WHOLE
    assert _cluster(_model(0.0), dataset, square) == WHOLE


def _cluster(model: Clusterer, dataset: ANDData, square: np.ndarray, **kwargs: Any) -> set[frozenset[str]]:
    ids = [str(i) for i in range(13)]
    repair = block_corrections.corrections_from_dataset(model, {"b": ids}, dataset)
    labels = model._cluster_one_block(ids, squareform(square), None, dataset, {}, corrections=repair, **kwargs)
    return _partition(labels)


def test_block_clustering_splits_a_block_without_an_impossible_pair() -> None:
    # No name conflict anywhere, so no repair applies; the split still runs.
    square = _square(0.25)
    square[0, X] = square[X, 0] = 0.9
    assert _cluster(_model(8.0), _dated_dataset(), square) == SPLIT


def test_block_clustering_splits_with_the_repair_off() -> None:
    # The two switches are independent; the split reads the cluster model's close-pair distance.
    model = _model(8.0)
    model.name_conflict_repair_max_distance = None
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
    expected = float(codes[pair_codes.condensed_index(13, rows[:, None], cols[None, :])].mean())
    monkeypatch.setattr(pair_codes, "TILE_CELLS", 9)
    assert year_gap_split._mean_cross_code(codes, 13, rows, cols) == expected


def _small_block_labels(
    old_first: str, years: list[int], other_smiths: int = 0, name_count: float | None = None
) -> list[int]:
    """Cluster five Smith mentions, two old ones named ``old_first`` and three "Ada", with ``years``.

    ``other_smiths`` more Smith mentions sit in the dataset but not in the clustered block.
    ``name_count`` is every mention's last-name-and-first-initial name count.
    """
    names = [(old_first, "", "Smith")] * 2 + [("Ada", "", "Smith")] * 3 + [("Bo", "", "Smith")] * other_smiths
    dataset = _dataset(names, {})
    if name_count is not None:
        counts = NameCounts(first=None, last=None, first_last=None, last_first_initial=name_count)
        for signature_id, signature in dataset.signatures.items():
            dataset.signatures[signature_id] = signature._replace(author_info_name_counts=counts)
    for index, year in enumerate(years):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=year)
    ids = [str(i) for i in range(5)]
    model = _model(8.0)
    repair = block_corrections.corrections_from_dataset(model, {"b": ids}, dataset)
    square = np.full((5, 5), 0.25)
    np.fill_diagonal(square, 0.0)
    labels = model._cluster_one_block(ids, squareform(square), None, dataset, {}, corrections=repair)
    return np.asarray(labels).tolist()


def test_block_clustering_splits_a_block_under_the_repair_size() -> None:
    # Five mentions: too few for the repair, enough for the split.
    labels = _small_block_labels("A.", [1928, 1930, 2001, 2010, 2020])
    assert labels[0] == labels[1] != labels[2] == labels[3] == labels[4]


@pytest.mark.parametrize(("name_count", "parts"), [(2194.0, 2), (3.0, 1), (float("nan"), 1)])
def test_a_common_name_count_rules_out_the_bridge(name_count: float, parts: int) -> None:
    # Scholar reclusters excluded mentions alone, so the request may hold only part of the block.
    assert len(set(_small_block_labels("Ada", [1928, 1930, 2001, 2010, 2020], name_count=name_count))) == parts


def test_a_small_part_of_a_large_surname_block_splits_parts_that_share_a_first_name() -> None:
    # A claimed profile, residual or subblock of five mentions from a block of eleven.
    labels = _small_block_labels("Ada", [1928, 1930, 2001, 2010, 2020], other_smiths=6)
    assert labels[0] == labels[1] != labels[2] == labels[3] == labels[4]


def test_block_clustering_treats_placeholder_years_as_undated() -> None:
    assert len(set(_small_block_labels("A.", [1601, 1800, 2001, 2010, 2020]))) == 1


def test_short_spans_skip_the_split() -> None:
    years = {"a": 2010 - YEAR_DIFF_CAP - 1, "b": 2010}
    assert block_corrections._spans_a_year_gap(["a", "b"], years)
    assert not block_corrections._spans_a_year_gap(["a", "b"], {"a": 2010 - YEAR_DIFF_CAP, "b": 2010})


@pytest.mark.parametrize("other_smiths", [0, 6])
def test_bulk_routes_bridge_only_in_a_small_surname_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, other_smiths: int
) -> None:
    # Five mentions requested; with six more Smiths in the dataset, they are part of a larger block.
    seen: list[dict[str, str]] = []
    original = Clusterer._cluster_one_block

    def spy(self: Clusterer, *args: Any, **kwargs: Any) -> Any:
        corrections = kwargs.get("corrections")
        seen.append(dict(corrections.first_names) if corrections is not None else {})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Clusterer, "_cluster_one_block", spy)
    dataset = _dataset([("Ada", "", "Smith")] * 5 + [("Bo", "", "Smith")] * other_smiths, {})
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow").arrow_dataset
    assert arrow_dataset is not None
    blocks = {"profile": [str(i) for i in range(5)]}
    model = _model(8.0)
    model.predict_from_arrow(blocks, arrow_dataset, name_tuples=frozenset())
    model.predict(blocks, dataset)
    expected = {} if other_smiths else dict.fromkeys(blocks["profile"], "ada")
    assert seen == [expected, expected]


@pytest.mark.parametrize("others", [0, 6])
def test_cached_grid_takes_small_surname_blocks_from_the_original_block(tmp_path: Path, others: int) -> None:
    # Five Ada Smiths across a 71-year gap. With six Bo Smiths the surname block has 11 mentions,
    # so the shared first name must not bridge the gap, though the five arrive as one subblock.
    from s2and.arrow_inputs import ArrowDataset
    from s2and.block_corrections import arrow_small_block_signatures
    from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
    from tests.incremental_test_helpers import _dataset, prepare_seed_arrow_bundle

    data = _dataset([("Ada", "", "Smith")] * 5 + [("Bo", "", "Smith")] * others, {})
    ids = [str(i) for i in range(5)]
    for sid, year in zip(ids, [1928, 1930, 2001, 2010, 2020], strict=True):
        key = str(data.signatures[sid].paper_id)
        data.papers[key] = data.papers[key]._replace(year=year)
    model = _model(8.0)
    model.orcid_join = False
    model.n_jobs = 1
    blocks = {"subblock": ids}
    prepare_seed_arrow_bundle(data, tmp_path)
    with ArrowDataset.open(tmp_path) as arrow:
        rust = build_rust_featurizer_from_arrow_dataset(
            arrow, signature_ids=ids, name_tuples=frozenset(), num_threads=1
        )
        dists = model.make_distance_matrices_from_rust_featurizer(blocks, rust)
        predicted, _ = model.predict_from_arrow(blocks, arrow, name_tuples=frozenset())
        with arrow.use() as lease:
            small = arrow_small_block_signatures(lease, {"request": list(data.signatures)})
        eps = float(model.cluster_model.eps)
        signatures, labels = model.predict_eps_grid_from_rust_featurizer(
            blocks, rust, dists, [eps], small_block_signatures=small
        )
    grid = {
        frozenset(s for s, label in zip(signatures, labels[eps], strict=True) if label == value)
        for value in set(labels[eps])
    }
    assert grid == {frozenset(members) for members in predicted.values()}
    expected = {frozenset(ids)} if others == 0 else {frozenset(("0", "1")), frozenset(("2", "3", "4"))}
    assert grid == expected
