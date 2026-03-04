from __future__ import annotations

import threading
import time

import pytest

from s2and.mp import UniversalPool


def test_streaming_imap_cancels_pending_futures_on_exception():
    started: list[int] = []
    lock = threading.Lock()

    def _task(item: int) -> int:
        with lock:
            started.append(item)
        if item == 0:
            # Keep this task briefly active so remaining futures are already queued
            # when the exception is raised.
            time.sleep(0.05)
            raise ValueError("boom")
        time.sleep(0.2)
        return item

    with pytest.raises(RuntimeError, match="imap item 0 raised") as exc_info:
        with UniversalPool(processes=1, use_threads=True) as pool:
            list(pool.imap(_task, range(6), chunksize=1, max_prefetch=6))

    # With best-effort cancellation enabled, the queued tail should not start.
    assert 0 in started
    assert all(item in {0, 1} for item in started)
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("best-effort cancelled" in note for note in notes)


def test_universal_pool_rejects_zero_processes():
    with pytest.raises(ValueError, match="processes must be a positive integer"):
        UniversalPool(processes=0, use_threads=True)


def test_universal_pool_defaults_to_one_when_cpu_count_unknown(monkeypatch):
    import s2and.mp as mp_module

    monkeypatch.setattr(mp_module.os, "cpu_count", lambda: None)
    with UniversalPool(processes=None, use_threads=True) as pool:
        assert pool.processes == 1


def test_streaming_imap_rejects_non_positive_chunksize():
    with UniversalPool(processes=1, use_threads=True) as pool:
        with pytest.raises(ValueError, match="chunksize must be >= 1"):
            list(pool.imap(lambda item: item, [1, 2, 3], chunksize=0))


def test_streaming_imap_rejects_non_positive_max_prefetch():
    with UniversalPool(processes=1, use_threads=True) as pool:
        with pytest.raises(ValueError, match="max_prefetch must be >= 1"):
            list(pool.imap(lambda item: item, [1, 2, 3], chunksize=1, max_prefetch=0))
