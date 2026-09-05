"""Focused normal-case reproduction for the clustering bug scan."""

import json
from pathlib import Path

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from s2and.runtime import build_runtime_context
from tests.helpers import tiny_name_counts_index
from tests.model_helpers import ConstantDistanceClassifier


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("reverse_block", [False, True])
@pytest.mark.parametrize("reverse_entries", [False, True])
@pytest.mark.parametrize("soft_distance", [0.0, 0.2])
def test_seed_restoration_matches_effective_directional_supervision(
    tmp_path: Path, backend: str, reverse_block: bool, reverse_entries: bool, soft_distance: float
) -> None:
    """Restoration uses the same direct-before-reverse rule as scored distances."""
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
    reference = {("2", "0"): 0.0, ("2", "1"): 1.0, ("0", "1"): soft_distance}
    contradictory = {**reference, ("1", "0"): 10000.0}
    if reverse_entries:
        contradictory = dict(reversed(list(contradictory.items())))
    if reverse_block:
        reference[("0", "1")] = 10000.0

    if backend == "rust":
        arrow_dataset = build_arrow_training_dataset(dataset, tmp_path)
        featurizer = _get_rust_featurizer(arrow_dataset)
        matrices = [
            clusterer.make_distance_matrices_from_rust_featurizer(blocks, featurizer, partial_supervision=supervision)
            for supervision in (reference, contradictory)
        ]
        outputs = [
            clusterer.predict_from_rust_featurizer(
                blocks,
                featurizer,
                partial_supervision=supervision,
                cluster_seeds_require=dataset.cluster_seeds_require,
            )[0]
            for supervision in (reference, contradictory)
        ]
    else:
        matrices = [
            clusterer.make_distance_matrices(blocks, dataset, partial_supervision=supervision)
            for supervision in (reference, contradictory)
        ]
        outputs = [
            clusterer.predict(blocks, dataset, partial_supervision=supervision)[0]
            for supervision in (reference, contradictory)
        ]
    np.testing.assert_array_equal(matrices[0]["block"], matrices[1]["block"])
    expected = {frozenset({"0", "2"}), frozenset({"1"})} if reverse_block else {frozenset({"0", "1", "2"})}
    for output in outputs:
        assert {frozenset(group) for group in output.values()} == expected


@pytest.mark.parametrize("override", ["none", "ignore_seeds", "partial_disallow", "dataset_disallow"])
def test_batched_prediction_preserves_initial_only_seed_member(monkeypatch: pytest.MonkeyPatch, override: str) -> None:
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
    batched, _ = clusterer.predict(
        blocks,
        dataset,
        batching_threshold=2,
        runtime_context=runtime,
        incremental_dont_use_cluster_seeds=override == "ignore_seeds",
        partial_supervision=partial,
    )
    assert any({"0", "1"} <= set(members) for members in normal.values()) == (override == "none")
    assert any({"0", "1"} <= set(members) for members in batched.values()) == (override == "none")
    assert sorted(sig for members in batched.values() for sig in members) == ["0", "1", "2"]


def test_explicit_partial_disallow_overrides_original_seed():
    """The final clustering must retain a caller-supplied hard negative."""
    raw = json.loads(Path("tests/dummy/signatures.json").read_text())
    raw = {key: raw[key] for key in ("0", "1")}
    dataset = ANDData(
        raw,
        "tests/dummy/papers.json",
        name="bug_scan_override",
        mode="inference",
        cluster_seeds={"0": {"1": "require"}},
        name_counts_index=tiny_name_counts_index(),
    )
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]), ConstantDistanceClassifier(0.9), n_jobs=1
    )
    clusters, _ = clusterer.predict({"a sattar": ["0", "1"]}, dataset, partial_supervision={("0", "1"): 10000.0})
    distances = clusterer.make_distance_matrices(
        {"a sattar": ["0", "1"]}, dataset, partial_supervision={("0", "1"): 10000.0}
    )
    assert distances["a sattar"][0] == 10000.0
    assert len(clusters) == 2


def test_cannot_link_survives_indirect_seed_unions_and_keeps_other_requires():
    """Indirect unions carry prohibitions while unrelated requires still merge."""
    from s2and.seed_merging import merge_seed_labels, seed_disallow_adjacency

    ids = ["a", "b", "c", "d", "x", "y", "e", "f"]
    # x/y are unseeded occupants of clusters being indirectly merged by A/B.
    labels = np.array([0, 1, 1, 2, 0, 2, 3, 4])
    groups = {"a": "A", "b": "A", "c": "B", "d": "B", "e": "C", "f": "C"}
    adjacency = seed_disallow_adjacency({("x", "y")}, {})
    actual = dict(zip(ids, merge_seed_labels(ids, labels, groups, adjacency), strict=True))
    assert actual["a"] == actual["b"] == actual["c"]
    assert actual["x"] != actual["y"]
    assert actual["c"] != actual["d"]
    assert actual["e"] == actual["f"]


def test_cannot_link_endpoint_can_still_merge_with_compatible_seed_members():
    """A forbidden pair must not exclude its endpoints from every seed union."""
    from s2and.seed_merging import merge_seed_labels, seed_disallow_adjacency

    ids = ["a", "b", "c", "d"]
    labels = np.arange(4)
    groups = dict.fromkeys(ids, "same")
    adjacency = seed_disallow_adjacency(set(), {("a", "b"): 10000.0, ("a", "c"): 10000.0})
    actual = dict(zip(ids, merge_seed_labels(ids, labels, groups, adjacency), strict=True))
    assert actual["a"] != actual["b"]
    assert actual["a"] != actual["c"]
    assert actual["b"] == actual["c"]
    assert len(set(actual.values())) == 2


def test_no_disallows_merge_transitive_seed_components():
    """Unconstrained transitive unions retain the smallest original label."""
    from s2and.seed_merging import merge_seed_labels

    ids = ["a", "b", "c", "d", "e", "f"]
    labels = np.array([1, 2, 0, 1, 2, 3])
    groups = {"a": "A", "b": "A", "c": "B", "d": "B", "e": "C", "f": "C"}
    assert merge_seed_labels(ids, labels, groups, {}) == [0] * len(ids)


def test_soft_supervision_overrides_dataset_cannot_link():
    """Sparse edges follow established explicit-supervision priority."""
    from s2and.seed_merging import seed_disallow_adjacency

    assert seed_disallow_adjacency({("a", "b")}, {("b", "a"): 0.0}) == {}
    assert seed_disallow_adjacency(set(), {("a", "b"): 0.9}) == {}


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


@pytest.mark.parametrize("reverse_block", [False, True])
def test_subblocked_restoration_retains_prior_matrix_order(
    monkeypatch: pytest.MonkeyPatch, reverse_block: bool
) -> None:
    """Cluster iteration order must not redefine prior supervision precedence."""
    from s2and.prediction_state import PredictionState

    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="subblock_directional_precedence",
        mode="inference",
        cluster_seeds={"0": {"1": "require"}},
        name_counts_index=tiny_name_counts_index(),
    )
    clusterer = Clusterer(FeaturizationInfo(features_to_use=["year_diff"]), classifier=None, n_jobs=1)
    observed_seeds: list[dict[str, int | str]] = []

    def incremental(
        _signatures: list[str], _dataset: ANDData, *, prediction_state: PredictionState, **_kwargs: object
    ) -> dict[str, object]:
        observed_seeds.append(prediction_state.cluster_seeds_require)
        return {"clusters": {"done": ["0", "1", "2", "3"]}}

    monkeypatch.setattr(clusterer, "_predict_incremental_python", incremental)
    clusterer._predict_subblocked_single_letter_incremental_groups(
        {"initial": ["3"]},
        # Deliberately put 1 before 0 in the cluster iteration order.
        pred_clusters={"first": ["1"], "second": ["2", "0"]},
        dataset=dataset,
        partial_supervision={("0", "1"): 0.0, ("1", "0"): 10000.0},
        runtime_context=build_runtime_context("subblock_directional_precedence", backend="python"),
        prior_blocks=[["2", "1", "0"] if reverse_block else ["2", "0", "1"]],
    )
    assert len(observed_seeds) == 1
    assert (observed_seeds[0]["0"] == observed_seeds[0]["1"]) is (not reverse_block)


def test_initial_only_seed_partitions_receive_distinct_synthetic_ids():
    """Do not reunite a forbidden initial-only pair through synthetic IDs."""
    from s2and.seed_merging import restore_seed_membership, seed_disallow_adjacency

    restored = restore_seed_membership(
        {"0": ["unrelated"]},
        {"a": "0", "b": "0", "c": "0"},
        seed_disallow_adjacency({("a", "b")}, {}),
    )
    assert restored["a"] != restored["b"]
    assert restored["unrelated"] == "0"
    assert restored["a"] != "0"
    assert restored["b"] != "0"
    assert len({restored[sig] for sig in ("a", "b", "c")}) == 2


@pytest.mark.parametrize("disallowed", [False, True])
def test_missing_seed_restoration_checks_unseeded_cluster_occupants(disallowed: bool) -> None:
    """A sparse edge to an ordinary occupant can prohibit a new seed merge."""
    from s2and.seed_merging import restore_seed_membership, seed_disallow_adjacency

    clusters = {"existing": ["a", "occupant"], "collision": ["other"]}
    seeds = {"a": "profile", "b": "profile", "c": "collision", "d": "collision"}
    adjacency = seed_disallow_adjacency({("b", "occupant")} if disallowed else set(), {})
    restored = restore_seed_membership(clusters, seeds, adjacency)
    assert restored["a"] == restored["occupant"] == "existing"
    assert (restored["b"] == restored["a"]) is not disallowed
    assert restored["c"] == restored["d"] != restored["other"]
    assert list(restored) == ["a", "occupant", "other", "b", "c", "d"]
    assert clusters == {"existing": ["a", "occupant"], "collision": ["other"]}
