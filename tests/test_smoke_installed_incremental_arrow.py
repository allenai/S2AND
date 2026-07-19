from __future__ import annotations

from pathlib import Path

import pytest

from s2and import production_model as production_model_module
from s2and.production_model import load_production_model
from scripts.verification import smoke_installed_incremental_arrow as smoke_module


def test_synthetic_bundle_uses_distinct_staging_and_final_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_hashes = {
        "name_tuples_data_sha256": "1" * 64,
        "orcid_prefix_counts_data_sha256": "2" * 64,
    }
    monkeypatch.setattr(smoke_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    bundle_dir = smoke_module._write_synthetic_bundle(tmp_path)

    pairwise_bundle_dir = tmp_path / "pairwise_stage" / "production_model_v0.0"
    assert pairwise_bundle_dir.is_dir()
    assert bundle_dir == tmp_path / "production_model_v0.0"
    assert bundle_dir.is_dir()
    assert bundle_dir != pairwise_bundle_dir

    clusterer = load_production_model(bundle_dir)
    assert clusterer.production_model_bundle_status == "complete"
    assert clusterer.production_model_bundle_version == "0.0"
