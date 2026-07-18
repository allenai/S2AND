from __future__ import annotations

import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import numpy as np
import pytest

from s2and.featurizer import NUM_FEATURES, FeaturizationInfo


def _write_incremental_features_worker(
    cache_db_path: str,
    start_event,
    worker_index: int,
    writes_per_worker: int,
) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_dir = Path(cache_db_path).parent
    featurizer_info.cache_directory = lambda _dataset_name: str(cache_dir)  # type: ignore[method-assign]
    featurizer_info.cache_db_path = lambda _dataset_name: cache_db_path  # type: ignore[method-assign]

    start_event.wait(timeout=5.0)

    for write_index in range(writes_per_worker):
        cache_key = f"worker{worker_index}_write{write_index}"
        feature_vector = np.full(NUM_FEATURES, float(write_index), dtype=np.float64)
        featurizer_info.write_cache({cache_key: feature_vector}, dataset_name="shared_dataset")


def test_feature_cache_concurrent_process_writes_preserve_all_keys(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db_path = str(cache_dir / "pair_features.sqlite3")

    worker_count = 4
    writes_per_worker = 25
    ctx = mp.get_context("spawn")
    start_event = ctx.Event()

    processes = [
        ctx.Process(
            target=_write_incremental_features_worker,
            args=(cache_db_path, start_event, worker_index, writes_per_worker),
        )
        for worker_index in range(worker_count)
    ]

    for process in processes:
        process.start()

    start_event.set()

    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0, f"worker failed pid={process.pid} exitcode={process.exitcode}"

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    featurizer_info.cache_directory = lambda _dataset_name: str(cache_dir)  # type: ignore[method-assign]
    featurizer_info.cache_db_path = lambda _dataset_name: cache_db_path  # type: ignore[method-assign]
    expected_keys = {
        f"worker{worker_index}_write{write_index}"
        for worker_index in range(worker_count)
        for write_index in range(writes_per_worker)
    }
    cached = featurizer_info.load_cache("shared_dataset", expected_keys)

    assert set(cached) == expected_keys


def test_feature_cache_reader_waits_for_complete_first_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir(parents=True)
    cache_db_path = str(cache_dir / "pair_features.sqlite3")
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    featurizer_info.cache_directory = lambda _dataset_name: str(cache_dir)  # type: ignore[method-assign]
    featurizer_info.cache_db_path = lambda _dataset_name: cache_db_path  # type: ignore[method-assign]

    schema_published = threading.Event()
    allow_writer_to_finish = threading.Event()
    initialize_call_lock = threading.Lock()
    initialize_call_count = 0
    real_initialize = FeaturizationInfo._initialize_pair_feature_cache_schema

    def pause_after_first_schema_publication(cls, connection) -> None:
        del cls
        nonlocal initialize_call_count
        real_initialize(connection)
        with initialize_call_lock:
            initialize_call_count += 1
            is_first_call = initialize_call_count == 1
        if is_first_call:
            schema_published.set()
            assert allow_writer_to_finish.wait(timeout=5.0)

    monkeypatch.setattr(
        FeaturizationInfo,
        "_initialize_pair_feature_cache_schema",
        classmethod(pause_after_first_schema_publication),
    )
    feature_vector = np.full(NUM_FEATURES, 1.0, dtype=np.float64)

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(
            featurizer_info.write_cache,
            {"pair": feature_vector},
            "shared_dataset",
        )
        assert schema_published.wait(timeout=5.0)
        reader = executor.submit(featurizer_info.load_cache, "shared_dataset", {"pair"})
        try:
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
        finally:
            allow_writer_to_finish.set()

        writer.result(timeout=5.0)
        cached = reader.result(timeout=5.0)

    np.testing.assert_array_equal(cached["pair"], feature_vector)
