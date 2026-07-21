"""Cross-platform primitives for durable artifact publication."""

from __future__ import annotations

import errno
import logging
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Never

logger = logging.getLogger(__name__)

_DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
_DEFAULT_LOCK_POLL_SECONDS = 0.05


def _lock_contention_errno() -> tuple[int, ...]:
    """Return error numbers that mean another process owns the lock."""

    if os.name == "nt":
        return (errno.EACCES,)
    return (errno.EACCES, errno.EAGAIN)


@contextmanager
def exclusive_file_lock(
    path: str | Path,
    *,
    timeout_seconds: float = _DEFAULT_LOCK_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_LOCK_POLL_SECONDS,
) -> Iterator[None]:
    """Hold a bounded exclusive process lock on one persistent lock file.

    Args:
        path: Persistent file whose first byte is used for the process lock.
        timeout_seconds: Maximum time to wait for another process to release the
            lock. A zero timeout performs one non-blocking acquisition attempt.
        poll_interval_seconds: Delay between acquisition attempts.

    Raises:
        ValueError: If either timing argument is outside its supported range.
        TimeoutError: If the lock remains held for ``timeout_seconds``.
    """

    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError(f"timeout_seconds must be finite and non-negative, got {timeout_seconds}")
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise ValueError(f"poll_interval_seconds must be finite and positive, got {poll_interval_seconds}")

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0)
        started_at = time.monotonic()
        deadline = started_at + timeout_seconds
        attempts = 0
        if os.name == "nt":
            import msvcrt

            while True:
                attempts += 1
                lock_file.seek(0)
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in _lock_contention_errno():
                        raise
                    _wait_for_lock_retry(
                        lock_path,
                        started_at=started_at,
                        deadline=deadline,
                        attempts=attempts,
                        poll_interval_seconds=poll_interval_seconds,
                        cause=exc,
                    )
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            while True:
                attempts += 1
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in _lock_contention_errno():
                        raise
                    _wait_for_lock_retry(
                        lock_path,
                        started_at=started_at,
                        deadline=deadline,
                        attempts=attempts,
                        poll_interval_seconds=poll_interval_seconds,
                        cause=exc,
                    )
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _wait_for_lock_retry(
    lock_path: Path,
    *,
    started_at: float,
    deadline: float,
    attempts: int,
    poll_interval_seconds: float,
    cause: OSError,
) -> None:
    """Wait for another lock attempt or raise the final timeout error."""

    now = time.monotonic()
    remaining_seconds = deadline - now
    if remaining_seconds <= 0:
        _raise_lock_timeout(
            lock_path,
            started_at=started_at,
            now=now,
            attempts=attempts,
            cause=cause,
        )
    time.sleep(min(poll_interval_seconds, remaining_seconds))
    now = time.monotonic()
    if now >= deadline:
        _raise_lock_timeout(
            lock_path,
            started_at=started_at,
            now=now,
            attempts=attempts,
            cause=cause,
        )


def _raise_lock_timeout(
    lock_path: Path,
    *,
    started_at: float,
    now: float,
    attempts: int,
    cause: OSError,
) -> Never:
    """Raise and log one contextual lock-acquisition timeout."""

    elapsed_seconds = now - started_at
    message = (
        f"timed out acquiring exclusive file lock {lock_path} after "
        f"{elapsed_seconds:.3f} seconds and {attempts} attempts"
    )
    logger.error(message)
    raise TimeoutError(message) from cause


def fsync_directory(path: str | Path) -> None:
    """Durably flush directory-entry changes where the platform supports it."""

    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
