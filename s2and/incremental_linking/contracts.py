"""Explicit contracts for incremental link-or-abstain artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS, promoted_linker_feature_columns
from s2and.incremental_linking.linker_pairwise import (
    promoted_pairwise_aggregate_columns,
    promoted_pairwise_coverage_columns,
)
from s2and.runtime import (
    RUST_CAPABILITY_HYBRID_CENTROID_RETRIEVER_V1,
    RUST_CAPABILITY_INCREMENTAL_LINKING_CONSTRAINT_ARRAYS_V1,
    RUST_CAPABILITY_INCREMENTAL_LINKING_PAIR_PLAN_V1,
    RUST_CAPABILITY_INDEXED_PAIR_ARRAY_FEATURIZATION_V1,
    detect_rust_runtime_capabilities,
)

ARTIFACT_SCHEMA_VERSION = "incremental_linking_artifact_v1"
CONTRACT_SCHEMA_VERSION = "incremental_linking_contract_v1"
MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER = "classic_lightgbm_linker"
GATE_SURFACE_PROMOTED_STRATIFIED = "promoted_stratified_gate"
DEFAULT_RETRIEVAL_TOP_K = 25

INCREMENTAL_LINKING_RUST_CAPABILITIES: tuple[str, ...] = (
    RUST_CAPABILITY_HYBRID_CENTROID_RETRIEVER_V1,
    RUST_CAPABILITY_INDEXED_PAIR_ARRAY_FEATURIZATION_V1,
    RUST_CAPABILITY_INCREMENTAL_LINKING_PAIR_PLAN_V1,
    RUST_CAPABILITY_INCREMENTAL_LINKING_CONSTRAINT_ARRAYS_V1,
)


def canonical_json_digest(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def promoted_linker_feature_schema_payload(feature_columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the feature-schema contract payload for the promoted linker."""

    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema": "promoted_70_redundant_drop_weighted_pairwise_agg",
        "feature_count": len(columns),
        "feature_columns": list(columns),
    }


def promoted_linker_feature_schema_digest(feature_columns: Sequence[str] | None = None) -> str:
    """Return the stable feature-schema digest."""

    return canonical_json_digest(promoted_linker_feature_schema_payload(feature_columns))


def promoted_feature_production_manifest(feature_columns: Sequence[str] | None = None) -> dict[str, str]:
    """Return the compact producer ownership manifest for each feature column."""

    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    non_pairwise = set(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
    pairwise = set(promoted_pairwise_aggregate_columns())
    coverage = set(promoted_pairwise_coverage_columns())
    manifest: dict[str, str] = {}
    for column in columns:
        if column in non_pairwise:
            manifest[column] = "compact_non_pairwise_row_formula"
        elif column in pairwise:
            manifest[column] = "rust_pairwise_aggregate"
        elif column in coverage:
            manifest[column] = "rust_pairwise_aggregate_coverage"
        else:
            manifest[column] = "external_or_unknown"
    return manifest


def production_contract_payload(feature_columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the production feature contract covered by artifact metadata."""

    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema_digest": promoted_linker_feature_schema_digest(columns),
        "feature_production_manifest": promoted_feature_production_manifest(columns),
        "missing_value_policy": {
            "pairwise_model_nan_value": "preserve_nan",
            "pairwise_aggregate_nan_value": 0.0,
            "matrix_nan_allowed": False,
        },
        "rounding_policy": {
            "compact_non_pairwise": "round_to_6_decimal_places_where_formula_requires",
            "pairwise_aggregates": "rust_f64_then_matrix_float32",
        },
    }


def production_contract_digest(feature_columns: Sequence[str] | None = None) -> str:
    """Return the stable production contract digest."""

    return canonical_json_digest(production_contract_payload(feature_columns))


def retrieval_stack_contract_payload(*, retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> dict[str, Any]:
    """Return the retrieval-stack contract covered by artifact metadata."""

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "retrieval_method": "global_hybrid_centroid",
        "retrieval_top_k": int(retrieval_top_k),
        "query_view_policy": "caller_supplied_initial_only_or_full",
        "candidate_filter_policy": "label_free_runtime_filters_only",
        "tie_break": "score_descending_component_key_ascending",
    }


def retrieval_stack_contract_digest(*, retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> str:
    """Return the stable retrieval-stack contract digest."""

    return canonical_json_digest(retrieval_stack_contract_payload(retrieval_top_k=retrieval_top_k))


def validate_promoted_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Validate and return promoted linker feature columns."""

    columns = tuple(str(column) for column in feature_columns)
    expected = promoted_linker_feature_columns()
    if columns != expected:
        raise ValueError(
            "Incremental linker feature columns do not match the promoted schema: "
            f"expected_count={len(expected)} observed_count={len(columns)}"
        )
    return columns


def available_incremental_linking_rust_capabilities(extension_module: Any | None = None) -> tuple[str, ...]:
    """Return named Rust capabilities currently available to incremental linking."""

    return detect_rust_runtime_capabilities(extension_module=extension_module).named_capabilities


def validate_required_rust_capabilities(
    required: Iterable[str],
    *,
    available: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate that required named Rust capabilities are available."""

    required_tuple = tuple(str(capability) for capability in required)
    available_set = set(available_incremental_linking_rust_capabilities() if available is None else available)
    missing = tuple(capability for capability in required_tuple if capability not in available_set)
    if missing:
        raise RuntimeError(f"Missing required Rust capabilities for incremental linker artifact: {missing}")
    return required_tuple


def validate_artifact_contract_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate artifact metadata fields that are independent of LightGBM loading."""

    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported incremental linker artifact schema_version: " f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("model_family") != MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER:
        raise ValueError(f"Unsupported incremental linker model_family: {metadata.get('model_family')!r}")
    feature_columns = validate_promoted_feature_columns(tuple(metadata.get("feature_columns", ())))
    expected_feature_digest = promoted_linker_feature_schema_digest(feature_columns)
    if metadata.get("feature_schema_digest") != expected_feature_digest:
        raise ValueError("Incremental linker artifact feature_schema_digest mismatch")
    expected_production_digest = production_contract_digest(feature_columns)
    if metadata.get("production_contract_digest") != expected_production_digest:
        raise ValueError("Incremental linker artifact production_contract_digest mismatch")
    retrieval_top_k = int(metadata.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    expected_retrieval_digest = retrieval_stack_contract_digest(retrieval_top_k=retrieval_top_k)
    if metadata.get("retrieval_stack_digest") != expected_retrieval_digest:
        raise ValueError("Incremental linker artifact retrieval_stack_digest mismatch")
    if metadata.get("gate_surface") != GATE_SURFACE_PROMOTED_STRATIFIED:
        raise ValueError(f"Unsupported incremental linker gate_surface: {metadata.get('gate_surface')!r}")
