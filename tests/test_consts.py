from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

import s2and.consts as consts_module


def test_consts_import_does_not_read_path_config(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = builtins.open
    config_suffix = str(Path("data") / "path_config.json")

    def _guarded_open(path, *args, **kwargs):
        if str(path).endswith(config_suffix):
            raise AssertionError("path_config.json should not be opened at import time")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _guarded_open)
    module = importlib.reload(consts_module)
    assert module.__dict__["_CONFIG"] is None


def test_consts_deferred_config_load_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.reload(consts_module)
    monkeypatch.delenv(module.CONFIG_LOCATION_ENV, raising=False)
    monkeypatch.setattr(module, "CONFIG_LOCATION", "/definitely/missing/path_config.json")
    module.__dict__["_CONFIG"] = None

    with pytest.raises(FileNotFoundError, match="path config"):
        _ = module.CONFIG["main_data_dir"]
