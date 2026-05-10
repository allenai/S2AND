"""Freeze and load LightGBM artifacts for the private incremental linker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from s2and.incremental_linking.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_RETRIEVAL_TOP_K,
    GATE_SURFACE_PROMOTED_STRATIFIED,
    MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER,
    production_contract_digest,
    promoted_linker_feature_schema_digest,
    retrieval_stack_contract_digest,
    validate_artifact_contract_metadata,
    validate_required_rust_capabilities,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns

BOOSTER_FILENAME = "booster.lgb"
METADATA_FILENAME = "metadata.json"
PREDICTION_FIXTURE_ATOL = 1e-10
PREDICTION_FIXTURE_RTOL = 1e-10


@dataclass(frozen=True)
class IncrementalLinkingArtifactMetadata:
    """Self-contained metadata needed to validate a linker artifact at load time."""

    schema_version: str
    model_family: str
    feature_columns: tuple[str, ...]
    feature_schema_digest: str
    production_contract_digest: str
    retrieval_stack_digest: str
    retrieval_top_k: int
    gate_surface: str
    gate_config: dict[str, Any]
    prediction_fixture_matrix: tuple[tuple[float, ...], ...]
    prediction_fixture_expected_probabilities: tuple[float, ...]
    required_rust_capabilities: tuple[str, ...]
    booster_sha256: str
    lightgbm_version: str
    audit_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        feature_columns: Sequence[str] | None = None,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        gate_config: Mapping[str, Any] | None = None,
        prediction_fixture_matrix: Sequence[Sequence[float]],
        prediction_fixture_expected_probabilities: Sequence[float],
        required_rust_capabilities: Sequence[str] = (),
        booster_sha256: str,
        lightgbm_version: str,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> IncrementalLinkingArtifactMetadata:
        """Build validated metadata for a promoted linker artifact."""

        columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
        fixture_matrix = tuple(tuple(float(value) for value in row) for row in prediction_fixture_matrix)
        fixture_probabilities = tuple(float(value) for value in prediction_fixture_expected_probabilities)
        if len(fixture_matrix) == 0:
            raise ValueError("prediction_fixture_matrix must contain at least one row")
        if any(len(row) != len(columns) for row in fixture_matrix):
            raise ValueError("prediction_fixture_matrix row width must match feature_columns")
        if len(fixture_probabilities) != len(fixture_matrix):
            raise ValueError("prediction_fixture_expected_probabilities length must match fixture rows")
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            model_family=MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER,
            feature_columns=columns,
            feature_schema_digest=promoted_linker_feature_schema_digest(columns),
            production_contract_digest=production_contract_digest(columns),
            retrieval_stack_digest=retrieval_stack_contract_digest(retrieval_top_k=int(retrieval_top_k)),
            retrieval_top_k=int(retrieval_top_k),
            gate_surface=GATE_SURFACE_PROMOTED_STRATIFIED,
            gate_config=dict(gate_config or {}),
            prediction_fixture_matrix=fixture_matrix,
            prediction_fixture_expected_probabilities=fixture_probabilities,
            required_rust_capabilities=tuple(str(value) for value in required_rust_capabilities),
            booster_sha256=str(booster_sha256),
            lightgbm_version=str(lightgbm_version),
            audit_metadata=dict(audit_metadata or {}),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IncrementalLinkingArtifactMetadata:
        """Load metadata from a JSON mapping."""

        metadata = cls(
            schema_version=str(payload["schema_version"]),
            model_family=str(payload["model_family"]),
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            feature_schema_digest=str(payload["feature_schema_digest"]),
            production_contract_digest=str(payload["production_contract_digest"]),
            retrieval_stack_digest=str(payload["retrieval_stack_digest"]),
            retrieval_top_k=int(payload["retrieval_top_k"]),
            gate_surface=str(payload["gate_surface"]),
            gate_config=dict(payload.get("gate_config", {})),
            prediction_fixture_matrix=tuple(
                tuple(float(value) for value in row) for row in payload["prediction_fixture_matrix"]
            ),
            prediction_fixture_expected_probabilities=tuple(
                float(value) for value in payload["prediction_fixture_expected_probabilities"]
            ),
            required_rust_capabilities=tuple(str(value) for value in payload.get("required_rust_capabilities", ())),
            booster_sha256=str(payload.get("booster_sha256", "")),
            lightgbm_version=str(payload.get("lightgbm_version", "")),
            audit_metadata=dict(payload.get("audit_metadata", {})),
        )
        validate_artifact_contract_metadata(metadata.to_json_dict())
        validate_required_rust_capabilities(metadata.required_rust_capabilities)
        return metadata

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""

        payload = asdict(self)
        payload["feature_columns"] = list(self.feature_columns)
        payload["prediction_fixture_matrix"] = [list(row) for row in self.prediction_fixture_matrix]
        payload["prediction_fixture_expected_probabilities"] = list(self.prediction_fixture_expected_probabilities)
        payload["required_rust_capabilities"] = list(self.required_rust_capabilities)
        return payload


@dataclass(frozen=True)
class IncrementalLinkingArtifact:
    """Loaded LightGBM booster plus validated linker metadata."""

    booster: lgb.Booster
    metadata: IncrementalLinkingArtifactMetadata
    artifact_dir: Path

    def predict_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for an artifact-ordered matrix."""

        features = np.asarray(matrix, dtype=np.float32, order="C")
        if features.ndim != 2:
            raise ValueError(f"feature matrix must be 2D, got shape={features.shape}")
        expected_cols = len(self.metadata.feature_columns)
        if features.shape[1] != expected_cols:
            raise ValueError(f"feature matrix width must be {expected_cols}, got {features.shape[1]}")
        probabilities = np.asarray(self.booster.predict(features), dtype=np.float64)
        if probabilities.ndim == 2:
            if probabilities.shape[1] < 2:
                raise ValueError(f"booster returned unsupported probability shape={probabilities.shape}")
            probabilities = probabilities[:, 1]
        return probabilities.reshape(-1)


def _booster_from_model(model: Any) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    inner = getattr(model, "classifier", None)
    if inner is not None and inner is not model:
        return _booster_from_model(inner)
    booster = getattr(model, "booster_", None)
    if isinstance(booster, lgb.Booster):
        return booster
    booster = getattr(model, "_Booster", None)
    if isinstance(booster, lgb.Booster):
        return booster
    raise TypeError(f"Expected a LightGBM Booster or fitted LightGBM estimator, got {type(model)!r}")


def _positive_probabilities_from_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(matrix, dtype=np.float32, order="C")
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    booster = _booster_from_model(model)
    probabilities = np.asarray(booster.predict(features), dtype=np.float64)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_incremental_linking_artifact(
    model: Any,
    artifact_dir: Path,
    *,
    feature_columns: Sequence[str] | None = None,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    gate_config: Mapping[str, Any] | None = None,
    prediction_fixture_matrix: Sequence[Sequence[float]] | np.ndarray,
    required_rust_capabilities: Sequence[str] = (),
    audit_metadata: Mapping[str, Any] | None = None,
) -> IncrementalLinkingArtifactMetadata:
    """Write `booster.lgb` and `metadata.json` for a fitted linker model."""

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    fixture = np.asarray(prediction_fixture_matrix, dtype=np.float32)
    if fixture.ndim != 2:
        raise ValueError(f"prediction_fixture_matrix must be 2D, got shape={fixture.shape}")
    if fixture.shape[1] != len(columns):
        raise ValueError(f"prediction_fixture_matrix width must be {len(columns)}, got {fixture.shape[1]}")
    expected_probabilities = _positive_probabilities_from_model(model, fixture)
    if len(expected_probabilities) != fixture.shape[0]:
        raise ValueError("prediction fixture probability count does not match fixture rows")
    fixture_rows = tuple(tuple(float(value) for value in row) for row in fixture.tolist())
    expected_probability_values = tuple(float(value) for value in expected_probabilities.tolist())

    booster = _booster_from_model(model)
    booster_path = artifact_dir / BOOSTER_FILENAME
    booster.save_model(str(booster_path))
    metadata = IncrementalLinkingArtifactMetadata.build(
        feature_columns=columns,
        retrieval_top_k=int(retrieval_top_k),
        gate_config=gate_config,
        prediction_fixture_matrix=fixture_rows,
        prediction_fixture_expected_probabilities=expected_probability_values,
        required_rust_capabilities=required_rust_capabilities,
        booster_sha256=_sha256_file(booster_path),
        lightgbm_version=str(lgb.__version__),
        audit_metadata=audit_metadata,
    )
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_incremental_linking_artifact(artifact_dir: Path) -> IncrementalLinkingArtifact:
    """Load and validate an incremental linker artifact."""

    artifact_dir = Path(artifact_dir)
    metadata_payload = json.loads((artifact_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    metadata = IncrementalLinkingArtifactMetadata.from_mapping(metadata_payload)
    booster_path = artifact_dir / BOOSTER_FILENAME
    if metadata.booster_sha256:
        observed_booster_sha256 = _sha256_file(booster_path)
        if observed_booster_sha256 != metadata.booster_sha256:
            raise ValueError("Incremental linker artifact booster_sha256 mismatch")
    booster = lgb.Booster(model_file=str(booster_path))
    artifact = IncrementalLinkingArtifact(
        booster=booster,
        metadata=metadata,
        artifact_dir=artifact_dir,
    )
    observed = artifact.predict_probabilities(np.asarray(metadata.prediction_fixture_matrix, dtype=np.float32))
    expected = np.asarray(metadata.prediction_fixture_expected_probabilities, dtype=np.float64)
    if not np.allclose(observed, expected, rtol=PREDICTION_FIXTURE_RTOL, atol=PREDICTION_FIXTURE_ATOL):
        raise ValueError("Incremental linker artifact prediction fixture mismatch")
    return artifact
