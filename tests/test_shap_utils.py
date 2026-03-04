from __future__ import annotations

import importlib

import numpy as np

import s2and.shap_utils as shap_utils


def test_import_shap_utils_does_not_mutate_numpy_legacy_aliases():
    legacy_aliases = ("bool", "int", "float", "object")
    present_before = {alias: (alias in np.__dict__) for alias in legacy_aliases}

    module = importlib.reload(shap_utils)
    assert module is shap_utils

    for alias in legacy_aliases:
        if present_before[alias]:
            continue
        assert alias not in np.__dict__
