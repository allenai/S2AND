from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

import s2and.plotting_utils as plotting_utils


def test_plotting_utils_import_does_not_read_path_config(monkeypatch):
    real_open = builtins.open
    config_suffix = str(Path("data") / "path_config.json")

    def _guarded_open(path, *args, **kwargs):
        path_str = str(path)
        if path_str.endswith(config_suffix):
            raise AssertionError("path_config.json should not be opened at import time")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)
    module = importlib.reload(plotting_utils)
    assert hasattr(module, "_experiment_dir")


def test_plotting_utils_deferred_config_load_raises_when_missing(monkeypatch):
    monkeypatch.setattr(plotting_utils, "CONFIG_LOCATION", "/definitely/missing/path_config.json")
    plotting_utils._PLOTTING_CONFIG = None
    with pytest.raises(FileNotFoundError):
        plotting_utils._experiment_dir()
