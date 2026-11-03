from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from s2and import model_pairwise
from s2and.consts import LARGE_DISTANCE
from s2and.model_pairwise import FastCluster, constrained_linkage
from scripts.eps_sweep import sweep_eps_on_linking_gold

SIZE = 160
EPS = 0.55


def _two_clusters(hard_distance: float = LARGE_DISTANCE) -> np.ndarray:
    """Two tight clusters, close to each other, with one sentinel pair between them."""
    matrix = np.full((2 * SIZE, 2 * SIZE), 0.1)
    matrix[:SIZE, :SIZE] = 0.05
    matrix[SIZE:, SIZE:] = 0.05
    matrix[0, SIZE] = matrix[SIZE, 0] = hard_distance
    np.fill_diagonal(matrix, 0.0)
    return squareform(matrix, checks=False)


def _partition(labels: np.ndarray) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), set()).add(index)
    return {frozenset(group) for group in groups.values()}


@pytest.mark.parametrize("preserve_input", [True, False])
def test_average_linkage_keeps_hard_negative_apart_in_large_merge(preserve_input: bool) -> None:
    condensed = _two_clusters()
    original = condensed.copy()
    # 160 x 160 cross pairs average the finite sentinel to about 0.49, below EPS.
    raw = fcluster(linkage(condensed, "average", preserve_input=True), EPS, "distance")
    assert raw[0] == raw[SIZE]

    labels = FastCluster(linkage="average", eps=EPS, preserve_input=preserve_input).fit_transform(condensed)

    assert _partition(labels) == {frozenset(range(SIZE)), frozenset(range(SIZE, 2 * SIZE))}
    if preserve_input:
        np.testing.assert_array_equal(condensed, original)


def test_read_only_input_is_copied_even_when_released() -> None:
    condensed = _two_clusters()
    condensed.flags.writeable = False

    labels = FastCluster(linkage="average", eps=EPS, preserve_input=False).fit_transform(condensed)

    assert labels[0] != labels[SIZE]
    assert condensed[SIZE - 1] == LARGE_DISTANCE  # condensed offset of pair (0, SIZE)


def test_chunked_remap_matches_whole_array_remap(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    condensed = rng.uniform(0.0, 1.0, 50 * 49 // 2)
    condensed[rng.choice(len(condensed), 300, replace=False)] = LARGE_DISTANCE
    expected = linkage(np.where(condensed == LARGE_DISTANCE, 1e12, condensed), "average")

    monkeypatch.setattr(model_pairwise, "_HARD_NEGATIVE_SCAN_CHUNK", 7)

    np.testing.assert_array_equal(constrained_linkage(condensed, "average", preserve_input=True), expected)


@pytest.mark.parametrize("distance", [LARGE_DISTANCE - 1, LARGE_DISTANCE + 1])
def test_only_the_exact_sentinel_is_hard(distance: float) -> None:
    labels = FastCluster(linkage="average", eps=EPS).fit_transform(_two_clusters(distance))

    assert _partition(labels) == {frozenset(range(2 * SIZE))}


@pytest.mark.parametrize("method", ["average", "weighted", "complete", "single", "ward", "centroid", "median"])
def test_linkage_without_sentinels_is_bit_identical(method: str) -> None:
    rng = np.random.default_rng(0)
    condensed = rng.uniform(0.0, 1.0, 40 * 39 // 2)

    np.testing.assert_array_equal(
        constrained_linkage(condensed, method, preserve_input=True),
        linkage(condensed, method, preserve_input=True),
    )


def test_eps_sweep_cached_linkage_matches_fastcluster() -> None:
    condensed = _two_clusters()
    clusterer = SimpleNamespace(cluster_model=FastCluster(linkage="average", eps=EPS))
    signature_ids = [str(index) for index in range(2 * SIZE)]

    tree = sweep_eps_on_linking_gold._linkage_from_cached_distance(  # noqa: SLF001
        clusterer=cast(Any, clusterer),
        block_key="block",
        signature_ids=signature_ids,
        dist=condensed,
    )
    sweep_labels = sweep_eps_on_linking_gold._labels_from_cached_linkage(  # noqa: SLF001
        linkage_matrix=tree,
        signature_ids=signature_ids,
        eps=EPS,
    )

    assert _partition(np.asarray(sweep_labels)) == _partition(clusterer.cluster_model.fit_transform(condensed))
    assert sweep_labels[0] != sweep_labels[SIZE]
