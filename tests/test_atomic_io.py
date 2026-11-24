"""Tests for cross-platform durable publication primitives, SHA-256 helpers, and lazy path config."""

from __future__ import annotations

import errno
import importlib
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import pytest

import s2and._atomic_io as atomic_io_module
import s2and.consts as consts_module
from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import is_lowercase_sha256, sha256_file


def _hold_lock_until_released(path: Path, ready: Any, release: Any, retry: Any, reacquired: Any) -> None:
    """Hold a lock until the parent has observed a bounded timeout."""
    with exclusive_file_lock(path):
        ready.set()
        if not release.wait(timeout=10):
            raise RuntimeError("contending test process did not release holder")

    if not retry.wait(timeout=10):
        raise RuntimeError("contending test process did not signal retry")
    with exclusive_file_lock(path, timeout_seconds=0.0):
        reacquired.set()


def test_exclusive_file_lock_real_contention_and_error_release(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready, release, retry, reacquired = (context.Event() for _ in range(4))
    lock_path = tmp_path / "publication.lock"
    holder = context.Process(target=_hold_lock_until_released, args=(lock_path, ready, release, retry, reacquired))
    holder.start()
    try:
        assert ready.wait(timeout=5)
        started_at = time.monotonic()
        with pytest.raises(TimeoutError, match="timed out acquiring exclusive file lock"):
            with exclusive_file_lock(lock_path, timeout_seconds=0.12, poll_interval_seconds=0.02):
                pytest.fail("acquired a held lock")
        assert 0.10 <= time.monotonic() - started_at < 2.0

        release.set()
        with pytest.raises(RuntimeError, match="publication failure"):
            with exclusive_file_lock(lock_path, timeout_seconds=2.0):
                raise RuntimeError("publication failure")
        retry.set()
        assert reacquired.wait(timeout=5)
    finally:
        release.set()
        retry.set()
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_exclusive_file_lock_deadlines_validation_and_windows_crt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subtests: pytest.Subtests
) -> None:
    with subtests.test("unbounded-timing-rejected"):
        for timeout_seconds, poll_interval_seconds in (
            (-1.0, 0.05),
            (float("inf"), 0.05),
            (1.0, 0.0),
            (1.0, float("nan")),
        ):
            with pytest.raises(ValueError, match="must be finite"):
                with exclusive_file_lock(
                    tmp_path / "invalid.lock",
                    timeout_seconds=timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                ):
                    pass

    with subtests.test("no-retry-after-deadline"):
        clock = 0.0
        attempts = 0
        monkeypatch.setattr(atomic_io_module.time, "monotonic", lambda: clock)

        def oversleep(_seconds: float) -> None:
            nonlocal clock
            clock = 2.0

        monkeypatch.setattr(atomic_io_module.time, "sleep", oversleep)
        if os.name == "nt":
            import msvcrt

            def contend(_fd: int, mode: int, _len: int) -> None:
                nonlocal attempts
                if mode == msvcrt.LK_NBLCK:
                    attempts += 1
                    raise OSError(errno.EACCES, "simulated contention")

            monkeypatch.setattr(msvcrt, "locking", contend)
        else:
            import fcntl

            def contend_posix(_fd: int, _op: int) -> None:
                nonlocal attempts
                attempts += 1
                raise OSError(errno.EAGAIN, "simulated contention")

            monkeypatch.setattr(fcntl, "flock", contend_posix)

        with pytest.raises(TimeoutError, match="after 2.000 seconds and 1 attempts"):
            with exclusive_file_lock(tmp_path / "deadline.lock", timeout_seconds=1.0, poll_interval_seconds=0.1):
                pass
        assert attempts == 1

    if os.name == "nt":
        import msvcrt

        with subtests.test("windows-crt-retries-past-10s-window"):
            win_clock = 0.0
            win_attempts = 0
            monkeypatch.setattr(atomic_io_module.time, "monotonic", lambda: win_clock)

            def advance(sec: float) -> None:
                nonlocal win_clock
                win_clock += sec

            def contend_window(_fd: int, mode: int, _len: int) -> None:
                nonlocal win_attempts
                if mode == msvcrt.LK_NBLCK:
                    win_attempts += 1
                    if win_clock <= 10.5:
                        raise OSError(errno.EACCES, "simulated contention")

            monkeypatch.setattr(atomic_io_module.time, "sleep", advance)
            monkeypatch.setattr(msvcrt, "locking", contend_window)
            with exclusive_file_lock(tmp_path / "win.lock", timeout_seconds=15.0, poll_interval_seconds=0.1):
                pass
            assert win_clock > 10.5 and win_attempts > 100

        with subtests.test("windows-unexpected-io-error-not-retried"):
            io_calls = 0

            def fail_lock(_fd: int, _mode: int, _len: int) -> None:
                nonlocal io_calls
                io_calls += 1
                raise OSError(errno.EIO, "simulated IO failure")

            monkeypatch.setattr(msvcrt, "locking", fail_lock)
            with pytest.raises(OSError, match="simulated IO failure"):
                with exclusive_file_lock(tmp_path / "eio.lock"):
                    pass
            assert io_calls == 1


def test_sha256_and_deferred_consts_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abc")
    assert sha256_file(payload) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    for case_id, value, expected in (
        ("mixed-hex-digest", "0123456789abcdef" * 4, True),
        ("uppercase", "A" * 64, False),
        ("non-hex", "g" * 64, False),
        ("short", "0" * 63, False),
        ("none", None, False),
    ):
        assert is_lowercase_sha256(value) is expected, case_id

    module = importlib.reload(consts_module)
    monkeypatch.delenv(module.CONFIG_LOCATION_ENV, raising=False)
    monkeypatch.setattr(module, "CONFIG_LOCATION", "/definitely/missing/path_config.json")
    module.__dict__["_CONFIG"] = None
    with pytest.raises(FileNotFoundError, match="path config"):
        _ = module.CONFIG["main_data_dir"]
