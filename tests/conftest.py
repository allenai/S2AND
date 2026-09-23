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
def isolate_runtime_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Select Python by default and restore global RNG state after each test.

    Native tests route explicitly or override the backend with monkeypatch.
    Tests of default backend resolution may remove the environment variable.
    Local RNG instances are preferred; legacy tests still seed global RNGs.
    """
    monkeypatch.setenv("S2AND_BACKEND", "python")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
