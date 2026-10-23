"""Platform policy for Python pair featurization worker pools."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pytest

import s2and.featurizer as featurizer_module
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import tiny_name_counts_index


def _dataset() -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="platform_policy",
        name_counts_index=tiny_name_counts_index(),
    )


def _all_pairs(dataset: ANDData) -> list[tuple[str, str, int]]:
    return [(left, right, 0) for left, right in combinations(sorted(dataset.signatures), 2)]


def _featurize(dataset: ANDData, pairs: list[tuple[str, str, int]], n_jobs: int) -> tuple[Any, ...]:
    return many_pairs_featurize(
        pairs,
        dataset,
        FeaturizationInfo(),
        n_jobs=n_jobs,
        chunk_size=3,
        nameless_featurizer_info=FeaturizationInfo(),
        nan_value=np.nan,
    )


class _SerialFakePool:
    """Record pool construction and run ``imap`` in the calling thread."""

    constructed: list[bool] = []

    def __init__(self, *, processes: int, use_threads: bool, initializer: Any, initargs: tuple[Any, ...]) -> None:
        type(self).constructed.append(use_threads)
        initializer(*initargs)

    def __enter__(self) -> _SerialFakePool:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def imap(self, func: Any, iterable: Any, _chunksize: int) -> Any:
        return map(func, iterable)


def test_windows_featurizes_serially_with_identical_output(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset()
    pairs = _all_pairs(dataset)
    serial = _featurize(dataset, pairs, n_jobs=1)

    def refuse_pool(**_kwargs: Any) -> None:
        raise AssertionError("Windows Python featurization must not construct a worker pool")

    monkeypatch.setattr(featurizer_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(featurizer_module, "UniversalPool", refuse_pool)
    windows_parallel_request = _featurize(dataset, pairs, n_jobs=6)

    for expected, observed in zip(serial, windows_parallel_request, strict=True):
        np.testing.assert_array_equal(np.asarray(observed).view(np.uint64), np.asarray(expected).view(np.uint64))


@pytest.mark.parametrize(("system", "expected_use_threads"), [("Darwin", True), ("Linux", False)])
def test_other_platforms_keep_their_worker_pool(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    expected_use_threads: bool,
) -> None:
    dataset = _dataset()
    pairs = _all_pairs(dataset)
    serial = _featurize(dataset, pairs, n_jobs=1)

    _SerialFakePool.constructed = []
    monkeypatch.setattr(featurizer_module.platform, "system", lambda: system)
    monkeypatch.setattr(featurizer_module, "UniversalPool", _SerialFakePool)
    pooled = _featurize(dataset, pairs, n_jobs=6)

    assert _SerialFakePool.constructed == [expected_use_threads]
    for expected, observed in zip(serial, pooled, strict=True):
        np.testing.assert_array_equal(np.asarray(observed).view(np.uint64), np.asarray(expected).view(np.uint64))
