from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
