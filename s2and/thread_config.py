"""Thread-count normalization helpers."""

from __future__ import annotations

import os


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Return a positive worker count, honoring sklearn negative `n_jobs` semantics.

    `n_jobs=-1` means all available CPUs; `-2` means all but one, and so on. A
    value of `None` means one worker. Zero and non-integer values are invalid.
    """

    if n_jobs is None:
        return 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise TypeError(f"n_jobs must be an int or None, got {type(n_jobs).__name__}")
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")
    if n_jobs < 0:
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count + 1 + n_jobs)
    return n_jobs
