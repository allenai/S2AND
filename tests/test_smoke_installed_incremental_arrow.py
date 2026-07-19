from __future__ import annotations

from pathlib import Path

from s2and.production_model import load_production_model
from scripts.verification.smoke_installed_incremental_arrow import _write_synthetic_bundle


def test_synthetic_bundle_uses_distinct_staging_and_final_directories(tmp_path: Path) -> None:
    bundle_dir = _write_synthetic_bundle(tmp_path)

    pairwise_bundle_dir = tmp_path / "pairwise_stage" / "production_model_v0.0"
    assert pairwise_bundle_dir.is_dir()
    assert bundle_dir == tmp_path / "production_model_v0.0"
    assert bundle_dir.is_dir()
    assert bundle_dir != pairwise_bundle_dir

    clusterer = load_production_model(bundle_dir)
    assert clusterer.production_model_bundle_status == "complete"
    assert clusterer.production_model_bundle_version == "0.0"
