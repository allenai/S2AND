"""Freeze and load LightGBM artifacts for the private incremental linker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import lightgbm as lgb
import numpy as np

from s2and.incremental_linking.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_RETRIEVAL_TOP_K,
    GATE_SURFACE_PROMOTED_LOGISTIC,
    MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER,
    canonical_json_digest,
    production_contract_digest,
    promoted_linker_feature_schema_digest,
    retrieval_constraint_decision_policy_payload,
    retrieval_stack_contract_digest,
    validate_artifact_contract_metadata,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import NumpyLogisticGate, load_logistic_gate_config
from s2and.model_pairwise import lightgbm_booster

BOOSTER_FILENAME = "booster.lgb"
METADATA_FILENAME = "metadata.json"
PREDICTION_FIXTURE_ATOL = 1e-10
PREDICTION_FIXTURE_RTOL = 1e-10
_METADATA_FIELDS = frozenset(
    {
        "audit_metadata",
        "booster_sha256",
        "feature_columns",
        "feature_schema_digest",
        "gate_config",
        "gate_surface",
        "lightgbm_version",
        "model_family",
        "pairwise_bundle_binding",
        "prediction_fixture_expected_probabilities",
        "prediction_fixture_matrix",
        "production_contract_digest",
        "retrieval_stack_digest",
        "retrieval_top_k",
        "schema_version",
        "target_spec_digest",
    }
)


def _freeze_json_value(value: Any) -> Any:
    """Return an immutable representation of one JSON-compatible value."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"artifact metadata mapping keys must be strings, got {key!r}")
            frozen[key] = _freeze_json_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact metadata numbers must be finite")
        return value
    raise ValueError(f"artifact metadata contains a non-JSON value: {type(value)!r}")


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_json_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - helper invariant
        raise TypeError("frozen artifact metadata must remain a mapping")
    return frozen


def _json_compatible_value(value: Any) -> Any:
    """Return mutable JSON containers without exposing stored metadata state."""

    if isinstance(value, Mapping):
        return {str(key): _json_compatible_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible_value(item) for item in value]
    return value


def _require_metadata_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Incremental linker artifact {field_name} must contain numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Incremental linker artifact {field_name} must contain finite numbers")
    return number


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
    gate_config: Mapping[str, Any]
    prediction_fixture_matrix: tuple[tuple[float, ...], ...]
    prediction_fixture_expected_probabilities: tuple[float, ...]
    booster_sha256: str
    lightgbm_version: str
    target_spec_digest: str
    pairwise_bundle_binding: Mapping[str, Any]
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the v4 contract and make all stored containers immutable."""

        string_fields = (
            "schema_version",
            "model_family",
            "feature_schema_digest",
            "production_contract_digest",
            "retrieval_stack_digest",
            "gate_surface",
            "booster_sha256",
            "lightgbm_version",
            "target_spec_digest",
        )
        for field_name in string_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Incremental linker artifact {field_name} must be a nonempty string")

        if isinstance(self.retrieval_top_k, bool) or not isinstance(self.retrieval_top_k, int):
            raise ValueError("Incremental linker artifact retrieval_top_k must be a positive integer")
        if self.retrieval_top_k <= 0:
            raise ValueError("Incremental linker artifact retrieval_top_k must be a positive integer")

        if isinstance(self.feature_columns, str | bytes) or not isinstance(self.feature_columns, Sequence):
            raise ValueError("Incremental linker artifact feature_columns must be a sequence of strings")
        columns = tuple(self.feature_columns)
        if not columns or not all(isinstance(column, str) and column.strip() for column in columns):
            raise ValueError("Incremental linker artifact feature_columns must be nonempty strings")
        object.__setattr__(self, "feature_columns", columns)

        if isinstance(self.prediction_fixture_matrix, str | bytes) or not isinstance(
            self.prediction_fixture_matrix, Sequence
        ):
            raise ValueError("prediction_fixture_matrix must be a sequence of rows")
        matrix: list[tuple[float, ...]] = []
        for row in self.prediction_fixture_matrix:
            if isinstance(row, str | bytes) or not isinstance(row, Sequence):
                raise ValueError("prediction_fixture_matrix rows must be sequences")
            resolved_row = tuple(
                _require_metadata_number(value, field_name="prediction_fixture_matrix") for value in row
            )
            matrix.append(resolved_row)
        fixture_matrix = tuple(matrix)
        if not fixture_matrix:
            raise ValueError("prediction_fixture_matrix must contain at least one row")
        if any(len(row) != len(columns) for row in fixture_matrix):
            raise ValueError("prediction_fixture_matrix row width must match feature_columns")
        object.__setattr__(self, "prediction_fixture_matrix", fixture_matrix)

        if isinstance(self.prediction_fixture_expected_probabilities, str | bytes) or not isinstance(
            self.prediction_fixture_expected_probabilities, Sequence
        ):
            raise ValueError("prediction_fixture_expected_probabilities must be a sequence")
        fixture_probabilities = tuple(
            _require_metadata_number(
                value,
                field_name="prediction_fixture_expected_probabilities",
            )
            for value in self.prediction_fixture_expected_probabilities
        )
        if len(fixture_probabilities) != len(fixture_matrix):
            raise ValueError("prediction_fixture_expected_probabilities length must match fixture rows")
        if any(probability < 0.0 or probability > 1.0 for probability in fixture_probabilities):
            raise ValueError("prediction_fixture_expected_probabilities must be between 0 and 1")
        object.__setattr__(self, "prediction_fixture_expected_probabilities", fixture_probabilities)

        if len(self.booster_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.booster_sha256):
            raise ValueError("Incremental linker artifact booster_sha256 is not a SHA-256")

        if not isinstance(self.gate_config, Mapping) or not self.gate_config:
            raise ValueError("Incremental linker artifact gate_config must be a nonempty object")
        if not isinstance(self.pairwise_bundle_binding, Mapping) or not self.pairwise_bundle_binding:
            raise ValueError("Incremental linker artifact pairwise_bundle_binding must be a nonempty object")
        if not isinstance(self.audit_metadata, Mapping):
            raise ValueError("Incremental linker artifact audit_metadata must be an object")
        if "pairwise_bundle_binding" in self.audit_metadata:
            raise ValueError(
                "audit_metadata key 'pairwise_bundle_binding' is reserved; "
                "use the top-level pairwise_bundle_binding field"
            )

        frozen_gate_config = _freeze_json_mapping(self.gate_config)
        frozen_pairwise_binding = _freeze_json_mapping(self.pairwise_bundle_binding)
        frozen_audit_metadata = _freeze_json_mapping(self.audit_metadata)
        object.__setattr__(self, "gate_config", frozen_gate_config)
        object.__setattr__(self, "pairwise_bundle_binding", frozen_pairwise_binding)
        object.__setattr__(self, "audit_metadata", frozen_audit_metadata)

        validate_artifact_contract_metadata(self.to_json_dict())

    def __deepcopy__(self, memo: dict[int, Any]) -> IncrementalLinkingArtifactMetadata:
        """Share metadata because every reachable container is immutable."""

        memo[id(self)] = self
        return self

    @classmethod
    def build(
        cls,
        *,
        feature_columns: Sequence[str] | None = None,
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        gate_config: Mapping[str, Any] | None = None,
        prediction_fixture_matrix: Sequence[Sequence[float]],
        prediction_fixture_expected_probabilities: Sequence[float],
        booster_sha256: str,
        lightgbm_version: str,
        target_spec_digest: str,
        pairwise_bundle_binding: Mapping[str, Any],
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> IncrementalLinkingArtifactMetadata:
        """Build validated metadata for a promoted linker artifact."""

        columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
        fixture_matrix = tuple(tuple(float(value) for value in row) for row in prediction_fixture_matrix)
        fixture_probabilities = tuple(float(value) for value in prediction_fixture_expected_probabilities)
        resolved_audit_metadata = dict(audit_metadata or {})
        resolved_audit_metadata.setdefault(
            "runtime_decision_policy",
            retrieval_constraint_decision_policy_payload(),
        )
        resolved_gate_config = dict(gate_config or {})
        load_logistic_gate_config(resolved_gate_config)
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            model_family=MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER,
            feature_columns=columns,
            feature_schema_digest=promoted_linker_feature_schema_digest(columns),
            production_contract_digest=production_contract_digest(columns),
            retrieval_stack_digest=retrieval_stack_contract_digest(retrieval_top_k=int(retrieval_top_k)),
            retrieval_top_k=int(retrieval_top_k),
            gate_surface=GATE_SURFACE_PROMOTED_LOGISTIC,
            gate_config=resolved_gate_config,
            prediction_fixture_matrix=fixture_matrix,
            prediction_fixture_expected_probabilities=fixture_probabilities,
            booster_sha256=str(booster_sha256),
            lightgbm_version=str(lightgbm_version),
            target_spec_digest=str(target_spec_digest),
            pairwise_bundle_binding=dict(pairwise_bundle_binding),
            audit_metadata=resolved_audit_metadata,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> IncrementalLinkingArtifactMetadata:
        """Load an exact, strictly typed v4 metadata mapping."""

        if not isinstance(payload, Mapping):
            raise ValueError("Incremental linker artifact metadata must be a JSON object")
        if any(not isinstance(key, str) for key in payload):
            raise ValueError("Incremental linker artifact metadata field names must be strings")
        observed_fields = set(payload)
        if observed_fields != _METADATA_FIELDS:
            missing = sorted(_METADATA_FIELDS - observed_fields)
            unknown = sorted(observed_fields - _METADATA_FIELDS)
            raise ValueError(
                "Incremental linker artifact metadata fields do not match the v4 schema: "
                f"missing={missing} unknown={unknown}"
            )

        return cls(
            schema_version=payload["schema_version"],
            model_family=payload["model_family"],
            feature_columns=payload["feature_columns"],
            feature_schema_digest=payload["feature_schema_digest"],
            production_contract_digest=payload["production_contract_digest"],
            retrieval_stack_digest=payload["retrieval_stack_digest"],
            retrieval_top_k=payload["retrieval_top_k"],
            gate_surface=payload["gate_surface"],
            gate_config=payload["gate_config"],
            prediction_fixture_matrix=payload["prediction_fixture_matrix"],
            prediction_fixture_expected_probabilities=payload["prediction_fixture_expected_probabilities"],
            booster_sha256=payload["booster_sha256"],
            lightgbm_version=payload["lightgbm_version"],
            target_spec_digest=payload["target_spec_digest"],
            pairwise_bundle_binding=payload["pairwise_bundle_binding"],
            audit_metadata=payload["audit_metadata"],
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""

        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "feature_columns": list(self.feature_columns),
            "feature_schema_digest": self.feature_schema_digest,
            "production_contract_digest": self.production_contract_digest,
            "retrieval_stack_digest": self.retrieval_stack_digest,
            "retrieval_top_k": self.retrieval_top_k,
            "gate_surface": self.gate_surface,
            "gate_config": _json_compatible_value(self.gate_config),
            "prediction_fixture_matrix": [list(row) for row in self.prediction_fixture_matrix],
            "prediction_fixture_expected_probabilities": list(self.prediction_fixture_expected_probabilities),
            "booster_sha256": self.booster_sha256,
            "lightgbm_version": self.lightgbm_version,
            "target_spec_digest": self.target_spec_digest,
            "pairwise_bundle_binding": _json_compatible_value(self.pairwise_bundle_binding),
            "audit_metadata": _json_compatible_value(self.audit_metadata),
        }


def _load_rust_lightgbm_booster(booster_path: Path) -> Any:
    from s2and.runtime import load_s2and_rust_extension

    return load_s2and_rust_extension().RustLightGBMBooster(str(booster_path))


@dataclass(frozen=True)
class IncrementalLinkingArtifact:
    """Loaded linker booster (scored by the Rust evaluator) plus validated metadata."""

    booster: Any
    metadata: IncrementalLinkingArtifactMetadata
    artifact_dir: Path
    gate_model: NumpyLogisticGate

    def __deepcopy__(self, memo: dict[int, Any]) -> IncrementalLinkingArtifact:
        """Share this immutable, thread-safe loaded artifact across clusterer copies."""

        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[Any, tuple[Path]]:
        """Revalidate and reload the native scorer instead of pickling a PyO3 object."""

        return load_incremental_linking_artifact, (self.artifact_dir,)

    def predict_probabilities(
        self,
        matrix: np.ndarray,
        *,
        num_threads: int | None = None,
        max_rows_per_chunk: int | None = None,
    ) -> np.ndarray:
        """Predict positive-class probabilities for an artifact-ordered matrix."""

        features = np.asarray(matrix)
        if features.ndim != 2:
            raise ValueError(f"feature matrix must be 2D, got shape={features.shape}")
        if features.dtype != np.float32 or not features.flags.c_contiguous:
            raise ValueError("incremental linker feature matrix must be C-contiguous float32")
        expected_cols = len(self.metadata.feature_columns)
        if features.shape[1] != expected_cols:
            raise ValueError(f"feature matrix width must be {expected_cols}, got {features.shape[1]}")
        row_count = int(features.shape[0])
        if max_rows_per_chunk is None:
            chunk_rows = max(1, row_count)
        else:
            chunk_rows = int(max_rows_per_chunk)
            if chunk_rows <= 0:
                raise ValueError(f"max_rows_per_chunk must be positive, got {max_rows_per_chunk}")
        if row_count == 0 or chunk_rows >= row_count:
            probabilities = np.asarray(
                self.booster.predict_proba_positive_f32(
                    features,
                    num_threads=num_threads,
                ),
                dtype=np.float64,
            )
            return probabilities.reshape(-1)

        probabilities = np.empty(row_count, dtype=np.float64)
        for start in range(0, row_count, chunk_rows):
            stop = min(row_count, start + chunk_rows)
            probabilities[start:stop] = np.asarray(
                self.booster.predict_proba_positive_f32(
                    features[start:stop],
                    num_threads=num_threads,
                ),
                dtype=np.float64,
            ).reshape(-1)
        return probabilities


def _positive_probabilities_from_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(matrix, dtype=np.float32, order="C")
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(features), dtype=np.float64)
        if probabilities.ndim == 2:
            return probabilities[:, 1]
        return probabilities.reshape(-1)
    booster = lightgbm_booster(model)
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


def _publish_immutable_artifact(staging_dir: Path, artifact_dir: Path) -> None:
    if artifact_dir.exists():
        raise FileExistsError(f"Incremental linker artifact output already exists: {artifact_dir}")
    try:
        os.replace(staging_dir, artifact_dir)
    except OSError:
        if artifact_dir.exists():
            raise FileExistsError(f"Incremental linker artifact output already exists: {artifact_dir}") from None
        raise


def _required_lightgbm_version() -> str:
    version = getattr(lgb, "__version__", None)
    if version is None:
        raise RuntimeError("lightgbm.__version__ is required when writing incremental linker artifact metadata")
    return str(version)


def save_incremental_linking_artifact(
    model: Any,
    artifact_dir: Path,
    *,
    feature_columns: Sequence[str] | None = None,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    gate_config: Mapping[str, Any] | None = None,
    prediction_fixture_matrix: Sequence[Sequence[float]] | np.ndarray,
    target_spec: Mapping[str, Any],
    pairwise_bundle_binding: Mapping[str, Any],
    audit_metadata: Mapping[str, Any] | None = None,
) -> IncrementalLinkingArtifactMetadata:
    """Write `booster.lgb` and `metadata.json` for a fitted linker model."""

    artifact_dir = Path(artifact_dir).resolve()
    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    fixture = np.asarray(prediction_fixture_matrix, dtype=np.float32)
    if fixture.ndim != 2:
        raise ValueError(f"prediction_fixture_matrix must be 2D, got shape={fixture.shape}")
    if fixture.shape[1] != len(columns):
        raise ValueError(f"prediction_fixture_matrix width must be {len(columns)}, got {fixture.shape[1]}")
    if gate_config is None:
        raise ValueError("gate_config is required and must contain a logistic gate model")
    expected_probabilities = _positive_probabilities_from_model(model, fixture)
    if len(expected_probabilities) != fixture.shape[0]:
        raise ValueError("prediction fixture probability count does not match fixture rows")
    fixture_rows = tuple(tuple(float(value) for value in row) for row in fixture.tolist())
    expected_probability_values = tuple(float(value) for value in expected_probabilities.tolist())
    lightgbm_version = _required_lightgbm_version()
    pairwise_bundle_binding = dict(pairwise_bundle_binding)
    if not pairwise_bundle_binding:
        raise ValueError("pairwise_bundle_binding is required and must be non-empty")

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}.staging-", dir=artifact_dir.parent))
    try:
        booster = lightgbm_booster(model)
        booster_path = staging_dir / BOOSTER_FILENAME
        booster.save_model(str(booster_path))
        metadata = IncrementalLinkingArtifactMetadata.build(
            feature_columns=columns,
            retrieval_top_k=int(retrieval_top_k),
            gate_config=gate_config,
            prediction_fixture_matrix=fixture_rows,
            prediction_fixture_expected_probabilities=expected_probability_values,
            booster_sha256=_sha256_file(booster_path),
            lightgbm_version=lightgbm_version,
            target_spec_digest=canonical_json_digest(dict(target_spec)),
            pairwise_bundle_binding=pairwise_bundle_binding,
            audit_metadata=audit_metadata,
        )
        (staging_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_incremental_linking_artifact(staging_dir)
        _publish_immutable_artifact(staging_dir, artifact_dir)
        return metadata
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def _load_incremental_linking_artifact(
    artifact_dir: Path,
    *,
    verified_booster_sha256: str | None,
) -> IncrementalLinkingArtifact:
    artifact_dir = Path(artifact_dir).resolve(strict=True)
    metadata_payload = json.loads((artifact_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    metadata = IncrementalLinkingArtifactMetadata.from_mapping(metadata_payload)
    booster_path = artifact_dir / BOOSTER_FILENAME
    observed_booster_sha256 = _sha256_file(booster_path) if verified_booster_sha256 is None else verified_booster_sha256
    if observed_booster_sha256 != metadata.booster_sha256:
        raise ValueError("Incremental linker artifact booster_sha256 mismatch")
    booster = _load_rust_lightgbm_booster(booster_path)
    artifact = IncrementalLinkingArtifact(
        booster=booster,
        metadata=metadata,
        artifact_dir=artifact_dir,
        gate_model=load_logistic_gate_config(metadata.gate_config),
    )
    observed = artifact.predict_probabilities(np.asarray(metadata.prediction_fixture_matrix, dtype=np.float32))
    expected = np.asarray(metadata.prediction_fixture_expected_probabilities, dtype=np.float64)
    if not np.allclose(observed, expected, rtol=PREDICTION_FIXTURE_RTOL, atol=PREDICTION_FIXTURE_ATOL):
        raise ValueError("Incremental linker artifact prediction fixture mismatch")
    return artifact


def _load_incremental_linking_artifact_from_verified_booster(
    artifact_dir: Path,
    *,
    booster_sha256: str,
) -> IncrementalLinkingArtifact:
    """Load an artifact whose booster was hashed by an enclosing manifest."""

    return _load_incremental_linking_artifact(
        artifact_dir,
        verified_booster_sha256=booster_sha256,
    )


def load_incremental_linking_artifact(artifact_dir: Path) -> IncrementalLinkingArtifact:
    """Load and validate an incremental linker artifact.

    Booster scoring always runs through the pinned Rust evaluator.
    """

    return _load_incremental_linking_artifact(
        artifact_dir,
        verified_booster_sha256=None,
    )
