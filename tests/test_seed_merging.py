"""Focused normal-case reproduction for the clustering bug scan."""

import json
from pathlib import Path

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.runtime import build_runtime_context
from tests.helpers import tiny_name_counts_index


class DifferentPersonClassifier:
    """Keep unconstrained pairs separate without fitting a stochastic model."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return a fixed non-match probability for each feature row."""
        return np.tile([0.9, 0.1], (len(features), 1))


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
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]), DifferentPersonClassifier(), n_jobs=1
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
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]), DifferentPersonClassifier(), n_jobs=1
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
