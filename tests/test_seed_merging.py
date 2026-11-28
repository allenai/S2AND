"""Focused normal-case reproduction for the clustering bug scan."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from s2and.runtime import build_runtime_context
from tests.helpers import tiny_name_counts_index
from tests.model_helpers import ConstantDistanceClassifier


def test_seed_packing_prefers_full_names_and_keeps_native_representatives() -> None:
    from s2and.seed_merging import pack_seed_components

    subblocks = {"a": ["i0", "i1"], "ada": ["f0", "f1"], "amy": ["g0"]}
    components = {"first": ["i0", "f0"], "second": ["i1", "g0"]}

    packed = pack_seed_components(subblocks, components, 2, full_name_subblocks={"ada", "amy"})

    assert packed == {"a": [], "ada": [["f0", "i0"], ["f1"]], "amy": [["g0", "i1"]]}
    assert subblocks == {"a": ["i0", "i1"], "ada": ["f0", "f1"], "amy": ["g0"]}
    assert components == {"first": ["i0", "f0"], "second": ["i1", "g0"]}


def test_combining_block_local_predictions_preserves_reserved_cluster_keys() -> None:
    from s2and.model import _combine_subblocked_predictions

    result = _combine_subblocked_predictions(
        {
            "first": {"0": ["a"]},
            "second": {"0": ["b"]},
            "third": {"0|collision=0001": ["c"], "0": ["d"]},
        }
    )
    assert result == {
        "0": ["a"],
        "0|collision=0002": ["b"],
        "0|collision=0001": ["c"],
        "0|collision=0003": ["d"],
    }


@pytest.mark.parametrize(("backend", "reverse_block"), [("python", False), ("rust", True)])
def test_seed_restoration_accepts_equal_reverse_supervision(tmp_path: Path, backend: str, reverse_block: bool) -> None:
    """Seed restoration is independent of pair orientation and dictionary order."""
    from s2and.feature_port import _get_rust_featurizer
    from tests.helpers import build_arrow_training_dataset

    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="seed_directional_precedence",
        mode="inference",
        cluster_seeds={"0": {"1": "require"}},
        name_counts_index=tiny_name_counts_index(),
    )
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=FastCluster(eps=0.3),
        n_jobs=1,
    )
    blocks = {"block": ["2", "1", "0"] if reverse_block else ["2", "0", "1"]}
    soft_distance = 0.2
    reference = {("2", "0"): 0.0, ("2", "1"): 1.0, ("0", "1"): soft_distance}
    duplicate = {**reference, ("1", "0"): soft_distance}
    if backend == "rust":
        arrow_dataset = build_arrow_training_dataset(dataset, tmp_path)
        featurizer = _get_rust_featurizer(arrow_dataset)
        matrices = [
            clusterer.make_distance_matrices_from_rust_featurizer(blocks, featurizer, partial_supervision=supervision)
            for supervision in (reference, duplicate)
        ]
        outputs = [
            clusterer.predict_from_rust_featurizer(
                blocks,
                featurizer,
                partial_supervision=supervision,
                cluster_seeds_require=dataset.cluster_seeds_require,
            )[0]
            for supervision in (reference, duplicate)
        ]
    else:
        matrices = [
            clusterer.make_distance_matrices(blocks, dataset, partial_supervision=supervision)
            for supervision in (reference, duplicate)
        ]
        outputs = [
            clusterer.predict(blocks, dataset, partial_supervision=supervision)[0]
            for supervision in (reference, duplicate)
        ]
    np.testing.assert_array_equal(matrices[0]["block"], matrices[1]["block"])
    expected = {frozenset({"0", "1", "2"})}
    for output in outputs:
        assert {frozenset(group) for group in output.values()} == expected


@pytest.mark.parametrize("override", ["none", "ignore_seeds", "partial_disallow", "dataset_disallow"])
def test_batched_prediction_preserves_initial_only_seed_member(
    tmp_path, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    """Subblocking honors original seeds and explicit separation in combination."""
    monkeypatch.setattr("s2and.subblocking._resolved_orcid_prefix_counts", lambda counts: {})
    raw = json.loads(Path("tests/dummy/signatures.json").read_text())
    raw = {key: raw[key] for key in ("0", "1", "2")}
    for key, first in [("0", "Abdul"), ("1", "A"), ("2", "Ahmed")]:
        raw[key]["author_info"]["first"] = first
        raw[key]["author_info"]["middle"] = ""
    runtime = build_runtime_context("bug_scan", backend="python")
    dataset = ANDData(
        raw,
        "tests/dummy/papers.json",
        name="bug_scan",
        mode="inference",
        cluster_seeds={"0": {"1": "require"}},
        name_counts_index=tiny_name_counts_index(),
    )
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]), ConstantDistanceClassifier(0.9), n_jobs=1
    )
    blocks = {"a sattar": ["0", "1", "2"]}
    if override == "dataset_disallow":
        dataset.cluster_seeds_disallow = {("0", "1")}
    partial: dict[tuple[str, str], int | float] = {("0", "1"): 10000.0} if override == "partial_disallow" else {}
    normal, _ = clusterer.predict(
        blocks,
        dataset,
        runtime_context=runtime,
        incremental_dont_use_cluster_seeds=override == "ignore_seeds",
        partial_supervision=partial,
    )
    from s2and.arrow_inputs import ArrowDataset
    from tests.incremental_test_helpers import attach_tiny_incremental_linker, prepare_seed_arrow_bundle

    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker", link=False)
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        batched, _ = clusterer.predict_from_arrow(
            blocks,
            arrow_dataset,
            batching_threshold=2,
            cluster_seeds_require=dataset.cluster_seeds_require,
            cluster_seeds_disallow=dataset.cluster_seeds_disallow,
            incremental_dont_use_cluster_seeds=override == "ignore_seeds",
            partial_supervision=partial,
        )
    if override == "partial_disallow":
        distances = clusterer.make_distance_matrices(blocks, dataset, partial_supervision=partial)
        assert distances["a sattar"][0] == 10000.0
    assert any({"0", "1"} <= set(members) for members in normal.values()) == (override == "none")
    assert any({"0", "1"} <= set(members) for members in batched.values()) == (override == "none")
    assert sorted(sig for members in batched.values() for sig in members) == ["0", "1", "2"]


def test_cannot_link_survives_indirect_seed_unions_and_keeps_other_requires():
    """Indirect unions carry prohibitions while unrelated requires still merge."""
    from s2and.seed_merging import merge_seed_labels, seed_disallow_adjacency

    ids = ["a", "b", "c", "d", "x", "y", "e", "f", "g", "h", "i"]
    # Block an indirect union through ordinary occupants x and y.
    labels = np.array([0, 1, 1, 2, 0, 2, 3, 4, 5, 6, 7])
    groups = {"a": "A", "b": "A", "c": "B", "d": "B", "e": "C", "f": "C", "g": "D", "h": "D", "i": "D"}
    adjacency = seed_disallow_adjacency({("x", "y")}, {("g", "h"): 10000.0, ("g", "i"): 10000.0})
    actual = dict(zip(ids, merge_seed_labels(ids, labels, groups, adjacency), strict=True))
    assert actual["a"] == actual["b"] == actual["c"]
    assert actual["x"] != actual["y"]
    assert actual["c"] != actual["d"]
    assert actual["e"] == actual["f"]
    assert actual["g"] != actual["h"] == actual["i"]


def test_no_disallows_merge_transitive_seed_components():
    """Unconstrained transitive unions retain the smallest original label."""
    from s2and.seed_merging import merge_seed_labels

    ids = ["a", "b", "c", "d", "e", "f", "g"]
    labels = np.array([1, 2, 0, 1, 2, 2, 3])
    groups = {"a": "A", "b": "A", "c": "B", "d": "B", "e": "B", "f": "C", "g": "C"}
    assert merge_seed_labels(ids, labels, groups, {}) == [0] * len(ids)


def test_directional_precedence_is_scoped_to_evaluated_blocks() -> None:
    """Do not impose a matrix orientation on cross-block or unscored pairs."""
    from s2and.seed_merging import seed_disallow_adjacency

    partial = {("a", "b"): 0.2, ("b", "a"): 10000.0}
    hard = {"a": {"b"}, "b": {"a"}}
    assert seed_disallow_adjacency({("a", "b")}, partial, blocks=[["a", "b"]]) == {}
    assert seed_disallow_adjacency(set(), partial, blocks=[["b", "a"]]) == hard
    assert seed_disallow_adjacency(set(), partial, blocks=[["a"], ["b"]]) == hard
    assert seed_disallow_adjacency(set(), partial) == hard
    assert seed_disallow_adjacency(set(), {("b", "a"): 10000.0}, blocks=[["a", "b"]]) == hard
    assert seed_disallow_adjacency({("a", "b")}, {("b", "a"): 0.0}) == {}


def test_missing_seed_restoration_respects_ordinary_occupants_and_reserved_ids() -> None:
    from s2and.seed_merging import restore_seed_membership, seed_disallow_adjacency

    clusters = {"existing": ["a", "occupant"], "collision": ["other"], "allowed": ["d"]}
    seeds = {
        "a": "profile",
        "b": "profile",
        "c": "profile",
        "x": "collision",
        "y": "collision",
        "d": "another_profile",
        "e": "another_profile",
    }
    adjacency = seed_disallow_adjacency({("b", "occupant"), ("x", "y")}, {})
    restored = restore_seed_membership(clusters, seeds, adjacency)

    assert restored["a"] == restored["occupant"] == restored["c"] == "existing"
    assert restored["b"] != restored["a"]
    assert restored["x"] != restored["y"]
    assert restored["other"] == "collision"
    assert "collision" not in {restored["x"], restored["y"]}
    assert restored["d"] == restored["e"] == "allowed"
    assert clusters == {"existing": ["a", "occupant"], "collision": ["other"], "allowed": ["d"]}


@pytest.mark.parametrize("close_pair", [(0.0, 0.0), (2.0, 0.3)], ids=["plain", "close-pair-bonus"])
def test_predict_eps_grid_matches_predict_at_each_eps(close_pair: tuple[float, float]) -> None:
    """One linkage per block, cut at each EPS, gives predict's clusters, seed joins included."""

    raw = json.loads(Path("tests/dummy/signatures.json").read_text())
    dataset = ANDData(
        raw,
        "tests/dummy/papers.json",
        name="eps_grid",
        mode="inference",
        cluster_seeds={"3": {"4": "require", "5": "disallow"}, "6": {"7": "require"}, "1": {"2": "require"}},
        name_counts_index=tiny_name_counts_index(),
    )
    blocks = {"single": ["0"], "pair": ["1", "2"], "six": ["3", "4", "5", "6", "7", "8"]}
    rng = np.random.default_rng(0)
    dists = {key: rng.uniform(0.0, 1.0, len(members) * (len(members) - 1) // 2) for key, members in blocks.items()}
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(linkage="average", close_pair_bonus=close_pair[0], close_pair_distance=close_pair[1]),
        n_jobs=1,
    )
    # predict(..., dists=...) skips the repair and the year-gap split; the grid runs both, and
    # the split now reaches the six-mention block.
    clusterer.year_gap_split = False
    grid = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]

    signatures, ids_by_eps = clusterer.predict_eps_grid(blocks, dataset, dists, grid)

    assert signatures == [signature for members in blocks.values() for signature in members]
    for eps in grid:
        clusterer.cluster_model.eps = eps
        expected, _ = clusterer.predict(blocks, dataset, dists=dists)
        observed: dict[int, list[str]] = defaultdict(list)
        for signature, cluster in zip(signatures, ids_by_eps[eps].tolist(), strict=True):
            observed[cluster].append(signature)
        assert sorted(map(sorted, observed.values())) == sorted(map(sorted, expected.values()))
