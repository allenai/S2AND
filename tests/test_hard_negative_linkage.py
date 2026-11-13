import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from s2and import model_pairwise
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.model_pairwise import FastCluster, constrained_linkage
from scripts.eps_sweep import sweep_eps_on_linking_gold
from tests.model_helpers import ConstantDistanceClassifier

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


@pytest.mark.parametrize("method", ["average", "weighted", "complete", "single", "ward", "centroid", "median"])
@pytest.mark.parametrize("with_sentinel", [False, True])
def test_linkage_uses_the_callers_matrix_only_when_given_up(
    method: str, with_sentinel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(3)
    original = rng.uniform(0.0, 1.0, 30 * 29 // 2)
    if with_sentinel:
        original[5] = LARGE_DISTANCE
    expected = linkage(np.where(original == LARGE_DISTANCE, 1e12, original), method)
    worked_in: list[np.ndarray] = []
    real = model_pairwise._linkage_in_place

    def spy(condensed: np.ndarray, linkage_method: str) -> np.ndarray:
        worked_in.append(condensed)
        return real(condensed, linkage_method)

    monkeypatch.setattr(model_pairwise, "_linkage_in_place", spy)

    kept = original.copy()
    np.testing.assert_array_equal(constrained_linkage(kept, method, preserve_input=True), expected)
    np.testing.assert_array_equal(kept, original)
    assert not np.shares_memory(worked_in[-1], kept)

    # A matrix the caller gives up is the working memory: no second matrix is allocated.
    given_up = original.copy()
    np.testing.assert_array_equal(constrained_linkage(given_up, method, preserve_input=False), expected)
    assert worked_in[-1] is given_up


def test_linkage_copies_inputs_it_cannot_work_in(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    original = rng.uniform(0.0, 1.0, 20 * 19 // 2)
    expected = linkage(original, "average")
    read_only = original.copy()
    read_only.setflags(write=False)
    # Every second value of a longer array: not contiguous, so the C routine cannot read it.
    strided = np.repeat(original, 2)[::2]
    assert not strided.flags.c_contiguous
    narrow = original.astype(np.float32)

    for candidate in (read_only, strided):
        before = candidate.copy()
        np.testing.assert_array_equal(constrained_linkage(candidate, "average", preserve_input=False), expected)
        np.testing.assert_array_equal(candidate, before)
    before = narrow.copy()
    np.testing.assert_array_equal(
        constrained_linkage(narrow, "average", preserve_input=False), linkage(narrow.astype(np.float64), "average")
    )
    np.testing.assert_array_equal(narrow, before)


@pytest.mark.parametrize("method", ["average", "weighted", "complete", "single", "ward", "centroid", "median"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_observation_matrix_linkage_matches_fastcluster(method: str, dtype: type) -> None:
    observations = np.random.default_rng(5).normal(size=(40, 6)).astype(dtype)
    before = observations.copy()

    tree = FastCluster(linkage=method, input_as_observation_matrix=True).linkage_matrix(observations)

    np.testing.assert_array_equal(tree, linkage(observations, method))
    np.testing.assert_array_equal(observations, before)


def test_no_library_code_calls_the_copying_linkage_wrapper() -> None:
    # fastcluster.linkage copies its input under NumPy 2, which doubles a block's peak memory.
    # Every linkage in the library and its scripts goes through _linkage_in_place instead.
    assert not hasattr(model_pairwise, "linkage")
    root = Path(model_pairwise.__file__).parents[1]
    wrapper = re.compile(
        r"from fastcluster import .*\blinkage\b|fastcluster\.linkage\("
        r"|from scipy\.cluster\.hierarchy import .*\blinkage\b|hierarchy\.linkage\("
    )
    offenders = [
        str(path.relative_to(root))
        for folder in ("s2and", "scripts")
        for path in sorted((root / folder).rglob("*.py"))
        if wrapper.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_linkage_edge_sizes_behave_as_fastcluster() -> None:
    np.testing.assert_array_equal(
        constrained_linkage(np.array([0.3]), "average", preserve_input=False), linkage(np.array([0.3]), "average")
    )
    with pytest.raises(ValueError):
        linkage(np.zeros(4), "average")
    with pytest.raises(ValueError, match="condensed distance matrix"):
        constrained_linkage(np.zeros(4), "average", preserve_input=False)
    with pytest.raises(ValueError):
        linkage(np.zeros(0), "average")
    with pytest.raises(ValueError):
        constrained_linkage(np.zeros(0), "average", preserve_input=False)


def test_eps_sweep_clusters_cached_distances_as_fastcluster_does(monkeypatch: pytest.MonkeyPatch) -> None:
    condensed = _two_clusters()
    signature_ids = [str(index) for index in range(2 * SIZE)]
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.4),
        cluster_model=FastCluster(linkage="average", eps=EPS),
        n_jobs=1,
    )
    # No signature has a full first name, so name-conflict repair leaves the linkage clusters alone.
    featurizer = SimpleNamespace(name_conflict_facts=lambda: ([], {}), cluster_seeds_require=lambda: [])
    monkeypatch.setattr(sweep_eps_on_linking_gold, "_build_arrow_featurizer", lambda *_args: featurizer)
    monkeypatch.setattr(sweep_eps_on_linking_gold, "_cache_metadata", lambda *_args: {})
    monkeypatch.setattr(sweep_eps_on_linking_gold, "_load_cached_distance", lambda *_args: condensed)

    blocks, rows = sweep_eps_on_linking_gold._cluster_cached_blocks(  # noqa: SLF001
        args=cast(Any, SimpleNamespace(dataset="d")),
        clusterer=clusterer,
        selected_subblocks={"block": signature_ids, "solo": ["x"]},
        cache_dir=cast(Any, Path("unused")),
        arrow_dataset=cast(Any, SimpleNamespace(generation_id="g")),
        eps_values=[EPS],
    )

    sweep_labels = blocks["block"].labels_by_eps[EPS]
    assert _partition(np.asarray(sweep_labels)) == _partition(clusterer.cluster_model.fit_transform(condensed))
    assert sweep_labels[0] != sweep_labels[SIZE]
    assert blocks["solo"].labels_by_eps[EPS] == [0]
    assert [(row["block_key"], row["clustered"]) for row in rows] == [("block", True), ("solo", False)]
