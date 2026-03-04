from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import orjson

from s2and.featurizer import FeaturizationInfo


def _write_incremental_features_worker(
    cache_path: str,
    start_event,
    worker_index: int,
    writes_per_worker: int,
) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    featurizer_info.cache_file_path = lambda _dataset_name: cache_path  # type: ignore[method-assign]

    start_event.wait(timeout=5.0)

    for write_index in range(writes_per_worker):
        cache_key = f"worker{worker_index}_write{write_index}"
        cached_features = {
            "features": {},
            "features_to_use": featurizer_info.features_to_use,
            "__new_features__": {cache_key: [float(write_index)]},
        }
        featurizer_info.write_cache(cached_features, dataset_name="shared_dataset", incremental=True)


def test_incremental_feature_cache_concurrent_process_writes_preserve_all_keys(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir / "all_features.json")

    worker_count = 4
    writes_per_worker = 25
    ctx = mp.get_context("spawn")
    start_event = ctx.Event()

    processes = [
        ctx.Process(
            target=_write_incremental_features_worker,
            args=(cache_path, start_event, worker_index, writes_per_worker),
        )
        for worker_index in range(worker_count)
    ]

    for process in processes:
        process.start()

    start_event.set()

    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0, f"worker failed pid={process.pid} exitcode={process.exitcode}"

    with open(cache_path, "rb") as cache_file:
        cached = orjson.loads(cache_file.read())

    feature_keys = set(cached["features"].keys())
    expected_keys = {
        f"worker{worker_index}_write{write_index}"
        for worker_index in range(worker_count)
        for write_index in range(writes_per_worker)
    }
    assert expected_keys.issubset(feature_keys)
