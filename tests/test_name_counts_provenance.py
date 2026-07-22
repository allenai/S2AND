from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import threading
import time
import weakref
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import s2and.name_counts_index as name_counts_index_module
import s2and.name_counts_manifest as name_counts_manifest_module
import s2and.runtime as runtime_module
from s2and.data import ANDData
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import (
    NameCountsIndex,
    clear_name_counts_index_cache,
    evict_name_counts_index,
)
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _write_index(root: Path, *, generation_id: str, first_count: int = 10) -> str:
    mappings = tiny_name_counts_tuple()
    mappings[0]["abdul"] = first_count
    provenance = {**tiny_name_counts_provenance(), "generation_id": generation_id}
    path, _metrics = write_name_counts_index(root, mappings, provenance, overwrite=True)
    return path


def _fake_native_index(path: str) -> SimpleNamespace:
    manifest_bytes = (Path(path) / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    provenance = manifest["source_provenance"]
    return SimpleNamespace(
        normalization_version=manifest["normalization_version"],
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        name_counts_provenance_binding=(
            provenance["generation_id"],
            provenance["pickle_sha256"],
            provenance["source_snapshot_id"],
            provenance["selected_rows_sha256"],
        ),
    )


@pytest.fixture(autouse=True)
def _clear_shared_name_counts_index_cache() -> Iterator[None]:
    clear_name_counts_index_cache()
    yield
    clear_name_counts_index_cache()


def test_name_counts_index_open_is_shared_for_one_manifest_generation(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second
    assert dict(first.source_provenance) == {
        **tiny_name_counts_provenance(),
        "generation_id": "generation-one",
    }


def test_name_counts_index_open_does_not_repeat_native_material_validation_in_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")

    def unexpected_python_hash(*args, **kwargs):
        pytest.fail("normal NameCountsIndex.open must use the native validation result")

    monkeypatch.setattr(name_counts_manifest_module, "_sha256_file", unexpected_python_hash)

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second


def test_name_counts_index_concurrent_open_shares_published_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    real_extension = runtime_module.load_s2and_rust_extension()
    native_open_started = threading.Event()
    release_native_open = threading.Event()
    call_lock = threading.Lock()
    native_open_calls = 0

    class BlockingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            with call_lock:
                native_open_calls += 1
            native_open_started.set()
            if not release_native_open.wait(timeout=10):
                raise TimeoutError("test did not release native name-count open")
            return real_extension.NameCountsIndex.open(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=BlockingNativeNameCountsIndex),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(NameCountsIndex.open, path)
        assert native_open_started.wait(timeout=10)
        second_future = pool.submit(NameCountsIndex.open, path)
        try:
            deadline = time.monotonic() + 10
            while not second_future.running() and time.monotonic() < deadline:
                time.sleep(0.001)
            assert second_future.running()
        finally:
            release_native_open.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first is second
    assert native_open_calls == 1


def test_name_counts_index_remains_cached_after_callers_release_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    native_open_calls = 0

    class CountingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            return _fake_native_index(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=CountingNativeNameCountsIndex),
    )

    first = NameCountsIndex.open(path)
    retained = weakref.ref(first)
    del first
    gc.collect()
    second = NameCountsIndex.open(path)

    assert retained() is second
    assert native_open_calls == 1


def test_name_counts_index_strong_cache_is_bounded_and_explicitly_evictable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(name_counts_index_module, "_LATEST_INDEX_CACHE_MAX_PATHS", 2)
    paths = [_write_index(tmp_path / f"index-{index}", generation_id=f"generation-{index}") for index in range(3)]
    native_open_calls = 0

    class CountingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            return _fake_native_index(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=CountingNativeNameCountsIndex),
    )

    for path in paths:
        opened = NameCountsIndex.open(path)
        del opened
        gc.collect()
    assert native_open_calls == 3

    latest = NameCountsIndex.open(paths[-1])
    del latest
    gc.collect()
    assert native_open_calls == 3

    oldest = NameCountsIndex.open(paths[0])
    del oldest
    gc.collect()
    assert native_open_calls == 4

    assert evict_name_counts_index(paths[0]) is True
    evicted = NameCountsIndex.open(paths[0])
    del evicted
    gc.collect()
    assert native_open_calls == 5

    clear_name_counts_index_cache()
    cleared = NameCountsIndex.open(paths[0])
    del cleared
    assert native_open_calls == 6


def test_name_counts_index_interrupted_owner_wakes_waiter_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    owner_started = threading.Event()
    release_owner = threading.Event()
    call_lock = threading.Lock()
    native_open_calls = 0

    class InterruptOnceNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            with call_lock:
                native_open_calls += 1
                call_number = native_open_calls
            if call_number == 1:
                owner_started.set()
                if not release_owner.wait(timeout=10):
                    raise TimeoutError("test did not release interrupted name-count open")
                raise KeyboardInterrupt
            return _fake_native_index(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=InterruptOnceNativeNameCountsIndex),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner_future = pool.submit(NameCountsIndex.open, path)
        assert owner_started.wait(timeout=10)
        waiter_future = pool.submit(NameCountsIndex.open, path)
        try:
            deadline = time.monotonic() + 10
            while not waiter_future.running() and time.monotonic() < deadline:
                time.sleep(0.001)
            assert waiter_future.running()
        finally:
            release_owner.set()
        with pytest.raises(KeyboardInterrupt):
            owner_future.result(timeout=10)
        opened = waiter_future.result(timeout=10)

    assert opened.source_provenance["generation_id"] == "generation-one"
    assert native_open_calls == 2
    assert name_counts_index_module._INDEX_INFLIGHT == {}


def test_name_counts_index_success_publication_interrupt_wakes_waiter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    owner_started = threading.Event()
    release_owner = threading.Event()
    native_open_calls = 0

    class BlockingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            owner_started.set()
            if not release_owner.wait(timeout=10):
                raise TimeoutError("test did not release interrupted name-count publication")
            return _fake_native_index(path_arg)

    real_future_type = name_counts_index_module.Future

    class InterruptOnceFuture(real_future_type):
        interrupt_next_result = True

        def set_result(self, result: NameCountsIndex) -> None:
            if self.interrupt_next_result:
                self.interrupt_next_result = False
                raise KeyboardInterrupt("injected during name-count success publication")
            super().set_result(result)

    monkeypatch.setattr(name_counts_index_module, "Future", InterruptOnceFuture)
    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=BlockingNativeNameCountsIndex),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner_future = pool.submit(NameCountsIndex.open, path)
        assert owner_started.wait(timeout=10)
        waiter_future = pool.submit(NameCountsIndex.open, path)
        try:
            deadline = time.monotonic() + 10
            while not waiter_future.running() and time.monotonic() < deadline:
                time.sleep(0.001)
            assert waiter_future.running()
        finally:
            release_owner.set()
        with pytest.raises(KeyboardInterrupt, match="success publication"):
            owner_future.result(timeout=10)
        opened = waiter_future.result(timeout=10)

    assert opened.source_provenance["generation_id"] == "generation-one"
    assert native_open_calls == 1
    assert name_counts_index_module._INDEX_INFLIGHT == {}


def test_name_counts_index_open_retries_manifest_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    manifest_path = (Path(path) / "manifest.json").resolve()
    real_read_bytes = Path.read_bytes
    replaced = False

    def replace_after_read(target: Path) -> bytes:
        nonlocal replaced
        payload = real_read_bytes(target)
        if target.resolve() == manifest_path and not replaced:
            replaced = True
            _write_index(tmp_path, generation_id="generation-two", first_count=99)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    with caplog.at_level(logging.WARNING, logger="s2and.name_counts_index"):
        opened = NameCountsIndex.open(path)

    assert replaced is True
    assert opened.source_provenance["generation_id"] == "generation-two"
    assert opened.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [99.0]
    assert "retrying attempt=1 max_attempts=3" in caplog.text


def test_name_counts_index_open_retries_when_parsed_generation_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    old_generation_dir = Path(path) / Path(manifest["files"]["first"]["path"]).parent
    real_extension = runtime_module.load_s2and_rust_extension()
    native_open_calls = 0

    class CleanupRacingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            if native_open_calls == 1:
                _write_index(tmp_path, generation_id="generation-two", first_count=99)
                shutil.rmtree(old_generation_dir)
                raise FileNotFoundError("generation-one disappeared before mmap")
            return real_extension.NameCountsIndex.open(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=CleanupRacingNativeNameCountsIndex),
    )

    with caplog.at_level(logging.WARNING, logger="s2and.name_counts_index"):
        opened = NameCountsIndex.open(path)

    assert native_open_calls == 2
    assert opened.source_provenance["generation_id"] == "generation-two"
    assert opened.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [99.0]
    assert "retrying attempt=1 max_attempts=3" in caplog.text


def test_name_counts_index_open_bounds_and_instruments_manifest_race_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    native_open_calls = 0

    class AlwaysMismatchedNativeNameCountsIndex:
        @staticmethod
        def open(_path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            return SimpleNamespace(manifest_sha256="0" * 64)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(NameCountsIndex=AlwaysMismatchedNativeNameCountsIndex),
    )

    with caplog.at_level(logging.WARNING, logger="s2and.name_counts_index"):
        with pytest.raises(RuntimeError, match="changed during all 3 open attempts"):
            NameCountsIndex.open(path)

    assert native_open_calls == 3
    assert caplog.text.count("retrying attempt=") == 2
    assert "final_failure attempt=3 max_attempts=3" in caplog.text


def test_name_counts_index_manifest_replacement_opens_new_generation(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    first = NameCountsIndex.open(path)
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]

    _write_index(tmp_path, generation_id="generation-two", first_count=99)
    second = NameCountsIndex.open(path)

    assert second is not first
    assert second.source_provenance["generation_id"] == "generation-two"
    assert second.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [99.0]
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]


def test_name_counts_index_revalidates_replaced_generation_files(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    first = NameCountsIndex.open(path)

    _write_index(tmp_path, generation_id="generation-two", first_count=99)
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_path = Path(path) / manifest["files"]["first"]["path"]
    payload = bytearray(first_path.read_bytes())
    payload[-1] ^= 1
    first_path.write_bytes(payload)
    manifest["test_generation_nonce"] = "corrupt-replacement"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        NameCountsIndex.open(path)
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]


def test_name_counts_index_constructor_revalidates_native_binding() -> None:
    provenance = tiny_name_counts_provenance()
    native = SimpleNamespace(
        normalization_version="canonical_v2",
        manifest_sha256="2" * 64,
        name_counts_provenance_binding=("wrong-generation", "0" * 64, "snapshot", "1" * 64),
    )

    with pytest.raises(ValueError, match="native provenance mismatch"):
        NameCountsIndex(
            native=native,
            path="unused",
            manifest_sha256="2" * 64,
            normalization_version="canonical_v2",
            source_provenance=provenance,
        )


def test_anddata_exposes_read_only_index_provenance(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="index-provenance",
        mode="inference",
        name_counts_index=path,
        preprocess=False,
    )

    assert dataset.name_counts_provenance is not None
    with pytest.raises(TypeError):
        dataset.name_counts_provenance["generation_id"] = "mutated"  # type: ignore[index]


def test_index_writer_rejects_incomplete_provenance(tmp_path: Path) -> None:
    provenance = tiny_name_counts_provenance()
    provenance.pop("selected_rows_sha256")

    with pytest.raises(ValueError, match="selected_rows_sha256"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple(), provenance)
