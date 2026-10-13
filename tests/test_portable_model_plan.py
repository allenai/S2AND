"""Verify model plans after corrected dataset relocation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from s2and.production_training_contract import load_model_plan


def _write_plan(root: Path) -> Path:
    source = root / "corrected_sources" / "inventors_s2and"
    source.mkdir(parents=True)
    files = {}
    for role in ("signatures", "papers", "clusters", "specter_embeddings"):
        content = f"{role}\n".encode()
        path = source / role
        path.write_bytes(content)
        files[role] = {
            "path": f"../corrected_sources/inventors_s2and/{role}",
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    plan = root / "pairwise" / "model_plan.json"
    plan.parent.mkdir()
    plan.write_text(
        json.dumps(
            {
                "release_version": "1.3",
                "datasets": {"inventors_s2and": files},
                "eps": {
                    "grid": [0.0, 0.5, 1.0],
                    "minimum_dataset_f1": 0.0,
                    "minimum_signature_weighted_f1": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_relative_model_plan_survives_source_removal(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _write_plan(original)
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    shutil.rmtree(original)
    plan = load_model_plan(relocated / "pairwise" / "model_plan.json")
    assert plan.datasets["inventors_s2and"].split_mode == "random_blocks"
    assert all(path.is_relative_to(relocated) for path in plan.datasets["inventors_s2and"].files.values())


def test_relative_model_plan_rejects_changed_source(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path)
    (tmp_path / "corrected_sources" / "inventors_s2and" / "signatures").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_model_plan(plan)


@pytest.mark.parametrize("name", ("../outside", "a/b", "a\\b", "C:patents"))
def test_model_dataset_name_cannot_escape_its_directory(tmp_path: Path, name: str) -> None:
    plan = _write_plan(tmp_path)
    contents = json.loads(plan.read_text(encoding="utf-8"))
    contents["datasets"] = {name: next(iter(contents["datasets"].values()))}
    plan.write_text(json.dumps(contents), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid model dataset"):
        load_model_plan(plan)
