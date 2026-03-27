from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

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
    featurizer_info.cache_storage_key = lambda _dataset_name: cache_db_path  # type: ignore[method-assign]
    featurizer_info.cache_file_path = lambda _dataset_name: str(cache_dir / "all_features.json")  # type: ignore[method-assign]

    start_event.wait(timeout=5.0)

    for write_index in range(writes_per_worker):
        cache_key = f"worker{worker_index}_write{write_index}"
        feature_vector = np.full(NUM_FEATURES, float(write_index), dtype=np.float64)
        cached_features = {
            "features": {},
            "features_to_use": featurizer_info.features_to_use,
            "__new_features__": {cache_key: feature_vector},
        }
        featurizer_info.write_cache(cached_features, dataset_name="shared_dataset", incremental=True)


def test_incremental_feature_cache_concurrent_process_writes_preserve_all_keys(tmp_path: Path):
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
    featurizer_info.cache_storage_key = lambda _dataset_name: cache_db_path  # type: ignore[method-assign]
    featurizer_info.cache_file_path = lambda _dataset_name: str(cache_dir / "all_features.json")  # type: ignore[method-assign]
    cached = featurizer_info.load_cache("shared_dataset")

    feature_keys = set(cached["features"].keys())
    expected_keys = {
        f"worker{worker_index}_write{write_index}"
        for worker_index in range(worker_count)
        for write_index in range(writes_per_worker)
    }
    assert expected_keys.issubset(feature_keys)
