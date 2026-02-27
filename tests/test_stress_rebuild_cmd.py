from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH


def _load_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rss_growth_fraction_edge_cases():
    module = _load_module()
    assert module._rss_growth_fraction([]) is None
    assert module._rss_growth_fraction([1.0]) is None
    assert module._rss_growth_fraction([0.0, 1.0]) is None
    assert module._rss_growth_fraction([2.0, 1.0]) == pytest.approx(-0.5)
    assert module._rss_growth_fraction([1.0, 1.2]) == pytest.approx(0.2)
