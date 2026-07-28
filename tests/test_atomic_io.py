"""Tests for cross-platform durable publication primitives."""

from __future__ import annotations

import errno
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest

import s2and._atomic_io as atomic_io_module
from s2and._atomic_io import exclusive_file_lock


def _hold_lock_until_released(path: Path, ready: Any, release: Any) -> None:
    """Hold a lock until the parent has observed a bounded timeout."""

    with exclusive_file_lock(path):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("contending test process did not release holder")


@pytest.mark.skipif(os.name != "nt", reason="Windows CRT locking regression")
def test_exclusive_file_lock_retries_past_windows_crt_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    clock = 0.0
    acquisition_attempts = 0

    def fake_monotonic() -> float:
        return clock

    def advance_clock(seconds: float) -> None:
        nonlocal clock
        clock += seconds

    def contend_past_former_window(_descriptor: int, mode: int, _length: int) -> None:
        nonlocal acquisition_attempts
        if mode != msvcrt.LK_NBLCK:
            return
        acquisition_attempts += 1
        if clock <= 10.5:
            raise OSError(errno.EACCES, "simulated contention")

    monkeypatch.setattr(atomic_io_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atomic_io_module.time, "sleep", advance_clock)
    monkeypatch.setattr(msvcrt, "locking", contend_past_former_window)

    with exclusive_file_lock(
        tmp_path / "publication.lock",
        timeout_seconds=15.0,
        poll_interval_seconds=0.1,
    ):
        pass

    assert clock > 10.5
    assert acquisition_attempts > 100


def test_exclusive_file_lock_times_out_on_permanent_contention(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    lock_path = tmp_path / "publication.lock"
    holder = context.Process(target=_hold_lock_until_released, args=(lock_path, ready, release))
    holder.start()
    try:
        assert ready.wait(timeout=5)
        started_at = time.monotonic()
        with (
            caplog.at_level(logging.ERROR, logger="s2and._atomic_io"),
            pytest.raises(TimeoutError, match="timed out acquiring exclusive file lock") as exc_info,
        ):
            with exclusive_file_lock(
                lock_path,
                timeout_seconds=0.2,
                poll_interval_seconds=0.02,
            ):
                pass
        elapsed_seconds = time.monotonic() - started_at
    finally:
        release.set()
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)

    assert 0.18 <= elapsed_seconds < 2.0
    assert str(lock_path) in str(exc_info.value)
    assert "attempts" in str(exc_info.value)
    assert str(exc_info.value) in caplog.text
    assert holder.exitcode == 0


def test_exclusive_file_lock_does_not_retry_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 0.0
    acquisition_attempts = 0

    def fake_monotonic() -> float:
        return clock

    def oversleep_deadline(_seconds: float) -> None:
        nonlocal clock
        clock = 2.0

    monkeypatch.setattr(atomic_io_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atomic_io_module.time, "sleep", oversleep_deadline)

    if os.name == "nt":
        import msvcrt

        def contend(_descriptor: int, mode: int, _length: int) -> None:
            nonlocal acquisition_attempts
            if mode != msvcrt.LK_NBLCK:
                return
            acquisition_attempts += 1
            raise OSError(errno.EACCES, "simulated contention")

        monkeypatch.setattr(msvcrt, "locking", contend)
    else:
        import fcntl

        def contend(_descriptor: int, _operation: int) -> None:
            nonlocal acquisition_attempts
            acquisition_attempts += 1
            raise OSError(errno.EAGAIN, "simulated contention")

        monkeypatch.setattr(fcntl, "flock", contend)

    with pytest.raises(TimeoutError, match="after 2.000 seconds and 1 attempts"):
        with exclusive_file_lock(
            tmp_path / "publication.lock",
            timeout_seconds=1.0,
            poll_interval_seconds=0.1,
        ):
            pass

    assert acquisition_attempts == 1


def test_exclusive_file_lock_zero_timeout_still_attempts_available_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "publication.lock"

    with exclusive_file_lock(lock_path, timeout_seconds=0.0):
        assert lock_path.exists()


def test_exclusive_file_lock_rejects_unbounded_timing_values(tmp_path: Path) -> None:
    cases = (
        ("negative-timeout", -1.0, 0.05),
        ("infinite-timeout", float("inf"), 0.05),
        ("zero-poll", 1.0, 0.0),
        ("nan-poll", 1.0, float("nan")),
    )
    for _case_id, timeout_seconds, poll_interval_seconds in cases:
        with pytest.raises(ValueError, match="must be finite"):
            with exclusive_file_lock(
                tmp_path / "publication.lock",
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            ):
                pass


@pytest.mark.skipif(os.name != "nt", reason="Windows CRT locking regression")
def test_exclusive_file_lock_does_not_retry_unexpected_windows_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    calls = 0

    def fail_lock(_descriptor: int, _mode: int, _length: int) -> None:
        nonlocal calls
        calls += 1
        raise OSError(errno.EIO, "simulated IO failure")

    monkeypatch.setattr(msvcrt, "locking", fail_lock)

    with pytest.raises(OSError, match="simulated IO failure"):
        with exclusive_file_lock(tmp_path / "publication.lock"):
            pass

    assert calls == 1
