"""Bulk seed policy is independent of altered markers and execution route."""

import pytest

from tests.incremental_test_helpers import _dataset, _predict
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("threshold", [None, 2], ids=["unbatched", "subblocked"])
@pytest.mark.parametrize("case", ["claimed", "seed_disallow", "query_disallow", "cross_block", "ignored_seeds"])
def test_bulk_altered_markers_preserve_ordinary_seed_policy(tmp_path, route, threshold, case):
    """Real public routes preserve claims without absorbing unrelated papers."""
    count = 6 if case == "cross_block" else 3
    seeds = {"0": "claimed", "1": "claimed"}
    if case == "cross_block":
        seeds["3"] = "claimed"
    dataset = _dataset([("Ada", "", "Smith")] * count, seeds)
    dataset.altered_cluster_signatures = ["0"]
    blocks = {"one": [str(index) for index in range(count)]}
    if case == "cross_block":
        blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    if case == "seed_disallow":
        dataset.cluster_seeds_disallow = {("0", "1")}
    elif case == "query_disallow":
        dataset.cluster_seeds_disallow = {("0", "2")}
    original_disallows = dataset.cluster_seeds_disallow.copy()
    kwargs = {"altered_cluster_signatures": ["0"]} if route == "arrow" else {}
    if case == "ignored_seeds":
        kwargs["incremental_dont_use_cluster_seeds"] = True

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        route,
        threshold=threshold,
        distance=0.1 if case == "query_disallow" else 0.9,
        **kwargs,
    )

    expected = {frozenset({str(index)}) for index in range(count)}
    if case not in {"seed_disallow", "ignored_seeds"}:
        expected -= {frozenset({"0"}), frozenset({"1"})}
        expected.add(frozenset({"0", "1"}))
    assert {frozenset(members) for members in clusters.values()} == expected
    _assert_partition(clusters, blocks)
    assert dataset.cluster_seeds_require == seeds
    assert dataset.cluster_seeds_disallow == original_disallows
    assert dataset.altered_cluster_signatures == ["0"]


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("threshold", [None, 2], ids=["unbatched", "subblocked"])
def test_bulk_altered_marker_does_not_bypass_required_component_size_limit(tmp_path, route, threshold):
    dataset = _dataset([("Ada", "", "Smith")] * 4, dict.fromkeys(["0", "1", "2"], "claimed"))
    dataset.altered_cluster_signatures = ["0"]
    kwargs = {"altered_cluster_signatures": ["0"]} if route == "arrow" else {}
    blocks = {"one": list(dataset.signatures)}
    if threshold is not None:
        with pytest.raises(ValueError, match="component exceeds batching_threshold"):
            _predict(tmp_path, dataset, blocks, route, threshold=threshold, distance=0.9, **kwargs)
    else:
        clusters = _predict(tmp_path, dataset, blocks, route, threshold=threshold, distance=0.9, **kwargs)
        assert {frozenset(members) for members in clusters.values()} == {frozenset({"0", "1", "2"}), frozenset({"3"})}


def _assert_partition(clusters, blocks):
    ids = [sid for members in clusters.values() for sid in members]
    assert sorted(ids) == sorted(sid for members in blocks.values() for sid in members)
    assert all(any(set(members) <= set(block) for block in blocks.values()) for members in clusters.values())
