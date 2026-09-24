"""Common runtime isolation for the Python orchestration test suite."""

import random
from collections.abc import Iterator

import numpy as np
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
