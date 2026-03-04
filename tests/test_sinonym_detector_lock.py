from __future__ import annotations

import sys
import threading
import time
from types import ModuleType

import s2and.data as data_module


def test_sinonym_detector_lazy_init_is_thread_safe(monkeypatch):
    created = {"count": 0}
    created_lock = threading.Lock()
    start_event = threading.Event()
    results: list[object] = []

    class _FakeDetector:
        def __init__(self):
            with created_lock:
                created["count"] += 1
            time.sleep(0.05)

    fake_detector_module = ModuleType("sinonym.detector")
    fake_detector_module.ChineseNameDetector = _FakeDetector  # type: ignore[attr-defined]
    fake_sinonym_package = ModuleType("sinonym")

    monkeypatch.setitem(sys.modules, "sinonym", fake_sinonym_package)
    monkeypatch.setitem(sys.modules, "sinonym.detector", fake_detector_module)
    monkeypatch.setattr(data_module, "_SINONYM_DETECTOR", None)

    def _worker() -> None:
        start_event.wait(timeout=2.0)
        results.append(data_module._ensure_sinonym_detector())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join(timeout=3.0)

    assert created["count"] == 1
    assert len(results) == 8
    first = results[0]
    assert all(result is first for result in results)
