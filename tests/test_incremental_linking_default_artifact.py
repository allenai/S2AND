from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.incremental_linking.artifact import load_incremental_linking_artifact
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.model import DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR
from tests.helpers import import_s2and_rust

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust(
    required_module_attrs=("RustLightGBMBooster",),
    prefer_site_packages=True,
)
if not _HAS_RUST_LIGHTGBM:
    raise pytest.skip.Exception(
        f"default incremental linker artifact requires RustLightGBMBooster: {_RUST_LIGHTGBM_PAYLOAD!r}",
        allow_module_level=True,
    )


def test_default_incremental_linker_artifact_loads_with_current_schema() -> None:
    artifact_dir = Path(DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR)
    if not artifact_dir.exists():
        raise pytest.skip.Exception(f"default incremental linker artifact is not present: {artifact_dir}")
    target_path = artifact_dir / "training_target.json"
    if target_path.exists():
        target = json.loads(target_path.read_text(encoding="utf-8"))
        if str(target.get("status", "")).endswith("pending_retrain"):
            raise pytest.skip.Exception("default artifact is intentionally pending retrain for the promoted schema")

    artifact = load_incremental_linking_artifact(artifact_dir)

    assert artifact.metadata.feature_columns == promoted_linker_feature_columns()
    assert len(artifact.metadata.feature_columns) == 53
    assert artifact.metadata.retrieval_top_k == 25
