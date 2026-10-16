"""Common runtime isolation for the Python orchestration test suite."""

import os

# Set these before a library starts its thread pool. Each xdist worker otherwise
# starts one OpenMP thread for each core, and the workers compete for the cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
# A known count stops the joblib physical-core probe, which costs about 2 s in each worker.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import random  # noqa: E402
from collections.abc import Iterator  # noqa: E402

import numpy as np  # noqa: E402
import pytest

from tests.rust_build_freshness import stale_rust_extension_message


def pytest_sessionstart(session: pytest.Session) -> None:
    """Stop before collection when the loaded native extension is stale."""
    message = stale_rust_extension_message()
    if message is not None:
        pytest.exit(message, returncode=pytest.ExitCode.USAGE_ERROR)


@pytest.fixture(autouse=True)
def restore_global_rng_state() -> Iterator[None]:
    """Restore global Python/NumPy RNG state after each test.

    Local RNG instances are preferred; legacy tests still seed global RNGs.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
