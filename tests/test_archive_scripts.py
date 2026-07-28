"""Static contract checks for retained archive provenance recipes."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from s2and.data import ANDData

REPO_ROOT = Path(__file__).resolve().parents[1]
AUGMENTATION_RECIPES = (
    "scripts/archive/make_augmentation_dataset_a.py",
    "scripts/archive/make_augmentation_dataset_b.py",
)


@pytest.mark.parametrize("relative_path", AUGMENTATION_RECIPES)
def test_augmentation_recipe_uses_supported_anddata_kwargs(relative_path: str) -> None:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ANDData"
    ]
    assert calls

    supported_kwargs = set(inspect.signature(ANDData).parameters)
    for call in calls:
        supplied_kwargs = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        assert supplied_kwargs <= supported_kwargs

        name_counts_index = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "name_counts_index"),
            None,
        )
        assert isinstance(name_counts_index, ast.Constant)
        assert name_counts_index.value is None
