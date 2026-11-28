"""Training/replay data loaders for promoted incremental-linker artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from s2and.model import _ensure_lightgbm_fitted
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION
from s2and.production_model import _load_pairwise_staging_model


def load_clusterer(
    model_path: Path,
    *,
    n_jobs: int,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> Any:
    """Load the production clusterer and prepare it for inference."""

    clusterer = _load_pairwise_staging_model(
        model_path,
        expected_eps_calibration=CALIBRATED_EPS_CALIBRATION,
        expected_artifact_hashes=expected_artifact_hashes,
    )
    _ensure_lightgbm_fitted(clusterer.classifier)
    _ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.n_jobs = int(n_jobs)
    return clusterer


__all__ = ["load_clusterer"]
