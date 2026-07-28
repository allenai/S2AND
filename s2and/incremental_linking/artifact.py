"""Save and load the current incremental-linker runtime artifact."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s2and._sha256 import is_lowercase_sha256
from s2and._sha256 import sha256_file as _sha256_file
from s2and.incremental_linking.contracts import (
    DEFAULT_RETRIEVAL_TOP_K,
    canonical_json_digest,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import NumpyLogisticGate, load_logistic_gate_config
from s2and.model_pairwise import lightgbm_booster

BOOSTER_FILENAME = "booster.lgb"
METADATA_FILENAME = "metadata.json"
ARTIFACT_SCHEMA_VERSION = "incremental_linking_artifact_v5"
_METADATA_FIELDS = frozenset(
    {
        "booster_sha256",
        "gate_config",
        "pairwise_bundle_binding_digest",
        "retrieval_top_k",
        "schema_version",
        "target_spec_digest",
    }
)


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not is_lowercase_sha256(value):
        raise ValueError(f"Incremental linker artifact {field_name} is not a SHA-256")
    return value


def _validated_metadata(payload: Any) -> tuple[dict[str, Any], NumpyLogisticGate]:
    if not isinstance(payload, Mapping):
        raise ValueError("Incremental linker artifact metadata must be a JSON object")
    observed_fields = set(payload)
    if observed_fields != _METADATA_FIELDS:
        missing = sorted(_METADATA_FIELDS - observed_fields)
        unknown = sorted(observed_fields - _METADATA_FIELDS)
        raise ValueError(
            "Incremental linker artifact metadata fields do not match the v5 schema: "
            f"missing={missing} unknown={unknown}"
        )
    if payload["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported incremental linker artifact schema_version: {payload['schema_version']!r}")

    retrieval_top_k = payload["retrieval_top_k"]
    if isinstance(retrieval_top_k, bool) or not isinstance(retrieval_top_k, int) or retrieval_top_k <= 0:
        raise ValueError("Incremental linker artifact retrieval_top_k must be a positive integer")

    for field_name in (
        "booster_sha256",
        "pairwise_bundle_binding_digest",
        "target_spec_digest",
    ):
        _require_sha256(payload[field_name], field_name=field_name)

    gate_config = payload["gate_config"]
    if not isinstance(gate_config, Mapping) or not gate_config:
        raise ValueError("Incremental linker artifact gate_config must be a nonempty object")
    gate_model = load_logistic_gate_config(gate_config)
    return dict(payload), gate_model


def _load_rust_lightgbm_booster(booster_path: Path) -> Any:
    from s2and.runtime import load_s2and_rust_extension

    return load_s2and_rust_extension().RustLightGBMBooster(str(booster_path))


@dataclass(frozen=True)
class IncrementalLinkingArtifact:
    """Loaded linker booster and the runtime state that affects its outputs."""

    booster: Any
    artifact_dir: Path
    gate_model: NumpyLogisticGate
    retrieval_top_k: int
    pairwise_bundle_binding_digest: str
    target_spec_digest: str

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the one feature order implemented by the current runtime."""

        return promoted_linker_feature_columns()

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
        """Predict positive-class probabilities for a canonical feature matrix."""

        features = np.asarray(matrix)
        if features.ndim != 2:
            raise ValueError(f"feature matrix must be 2D, got shape={features.shape}")
        if features.dtype != np.float32 or not features.flags.c_contiguous:
            raise ValueError("incremental linker feature matrix must be C-contiguous float32")
        expected_cols = len(self.feature_columns)
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


def _publish_immutable_artifact(staging_dir: Path, artifact_dir: Path) -> None:
    if artifact_dir.exists():
        raise FileExistsError(f"Incremental linker artifact output already exists: {artifact_dir}")
    try:
        os.replace(staging_dir, artifact_dir)
    except OSError:
        if artifact_dir.exists():
            raise FileExistsError(f"Incremental linker artifact output already exists: {artifact_dir}") from None
        raise


def save_incremental_linking_artifact(
    model: Any,
    artifact_dir: Path,
    *,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    gate_config: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    pairwise_bundle_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write `booster.lgb` and the minimal current runtime metadata."""

    artifact_dir = Path(artifact_dir).resolve()
    pairwise_binding = dict(pairwise_bundle_binding)
    if not pairwise_binding:
        raise ValueError("pairwise_bundle_binding is required and must be non-empty")

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}.staging-", dir=artifact_dir.parent))
    try:
        booster = lightgbm_booster(model)
        booster_path = staging_dir / BOOSTER_FILENAME
        booster.save_model(str(booster_path))
        metadata: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "booster_sha256": _sha256_file(booster_path),
            "retrieval_top_k": retrieval_top_k,
            "gate_config": dict(gate_config),
            "pairwise_bundle_binding_digest": canonical_json_digest(pairwise_binding),
            "target_spec_digest": canonical_json_digest(dict(target_spec)),
        }
        _validated_metadata(metadata)
        (staging_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
    metadata, gate_model = _validated_metadata(metadata_payload)
    booster_path = artifact_dir / BOOSTER_FILENAME
    observed_booster_sha256 = _sha256_file(booster_path) if verified_booster_sha256 is None else verified_booster_sha256
    if observed_booster_sha256 != metadata["booster_sha256"]:
        raise ValueError("Incremental linker artifact booster_sha256 mismatch")
    booster = _load_rust_lightgbm_booster(booster_path)
    expected_feature_count = len(promoted_linker_feature_columns())
    observed_feature_count = int(booster.num_features())
    if observed_feature_count != expected_feature_count:
        raise ValueError(
            "Incremental linker booster feature count mismatch: "
            f"expected={expected_feature_count} observed={observed_feature_count}"
        )
    return IncrementalLinkingArtifact(
        booster=booster,
        artifact_dir=artifact_dir,
        gate_model=gate_model,
        retrieval_top_k=int(metadata["retrieval_top_k"]),
        pairwise_bundle_binding_digest=str(metadata["pairwise_bundle_binding_digest"]),
        target_spec_digest=str(metadata["target_spec_digest"]),
    )


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
    """Load and validate an incremental linker artifact."""

    return _load_incremental_linking_artifact(
        artifact_dir,
        verified_booster_sha256=None,
    )
