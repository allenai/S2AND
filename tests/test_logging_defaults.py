from __future__ import annotations

import logging

from s2and import memory_budget
from scripts import rust_suite


def test_rust_suite_main_removes_file_handler_after_run(monkeypatch, tmp_path) -> None:
    logger = logging.getLogger("s2and")
    log_path = tmp_path / "rust-suite.log"

    monkeypatch.setattr(rust_suite, "_dispatch", lambda *_args, **_kwargs: 0)

    assert rust_suite.main(["--log-file", str(log_path), "compare"]) == 0
    assert not any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        for handler in logger.handlers
    )


def test_rust_suite_main_configures_memory_telemetry_path(monkeypatch, tmp_path) -> None:
    telemetry_path = tmp_path / "memory.jsonl"
    previous_path = memory_budget.memory_telemetry_jsonl_path()
    seen: dict[str, object] = {}

    def fake_dispatch(*_args, **_kwargs) -> int:
        seen["path"] = memory_budget.memory_telemetry_jsonl_path()
        return 0

    monkeypatch.setattr(rust_suite, "_dispatch", fake_dispatch)

    assert rust_suite.main(["--memory-telemetry-jsonl", str(telemetry_path), "compare"]) == 0
    assert seen["path"] == telemetry_path
    assert memory_budget.memory_telemetry_jsonl_path() == previous_path
