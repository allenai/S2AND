"""Explicit contracts for incremental link-or-abstain artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS, promoted_linker_feature_columns
from s2and.incremental_linking.linker_pairwise import promoted_pairwise_aggregate_columns

ARTIFACT_SCHEMA_VERSION = "incremental_linking_artifact_v3"
CONTRACT_SCHEMA_VERSION = "incremental_linking_contract_v1"
MODEL_FAMILY_CLASSIC_LIGHTGBM_LINKER = "classic_lightgbm_linker"
GATE_SURFACE_PROMOTED_LOGISTIC = "promoted_numpy_logistic_gate"
DEFAULT_RETRIEVAL_TOP_K = 25
CONSTRAINT_DECISION_POLICY: dict[str, Any] = {
    "same_orcid": {
        "action": "force_link",
        "return_all_matching_components": True,
        "exempt_from_disallow_veto": True,
        "beats_non_orcid_candidates": True,
    },
    "require_constraint": {
        "action": "force_link",
        "exempt_from_disallow_veto": True,
    },
    "disallow_constraint": {
        "action": "veto_candidate_row",
        "single_member_candidate": True,
        "all_pairs_disallow": True,
        "mostly_disallow_min_pair_count": 3,
        "mostly_disallow_fraction": 0.8,
    },
    "top_row_veto": {
        "action": "recompute_gate_over_eligible_rows",
        "all_rows_vetoed_action": "abstain",
    },
    "query_query_disallow": {
        "action": "request_global_finalize",
        "priority": ["require_forced", "initial_score_descending", "signature_id_ascending"],
        "conflict_action": "rescore_query_with_finalized_partner_components_excluded",
        "batch_and_input_order_invariant": True,
    },
}


def canonical_json_digest(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def promoted_linker_feature_schema_payload(feature_columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the feature-schema contract payload for the promoted linker."""

    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema": "promoted_53_round6_backward_shap_corr_weighted_pairwise_agg",
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
    manifest: dict[str, str] = {}
    for column in columns:
        if column in non_pairwise:
            manifest[column] = "compact_non_pairwise_row_formula"
        elif column in pairwise:
            manifest[column] = "rust_pairwise_aggregate"
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
        "query_author_policy": "canonical_v2_fields_plus_normalized_suffix",
        "title_normalization_policy": "transliterated_lower_alphanumeric_preserve_digits",
        "rounding_policy": {
            "compact_non_pairwise": "round_to_6_decimal_places_where_formula_requires",
            "incremental_six_decimal": "ties_to_even_before_float32",
            "pairwise_aggregates": "rust_f64_then_matrix_float32",
        },
    }


def production_contract_digest(feature_columns: Sequence[str] | None = None) -> str:
    """Return the stable production contract digest."""

    return canonical_json_digest(production_contract_payload(feature_columns))


def retrieval_constraint_decision_policy_payload() -> dict[str, Any]:
    """Return the post-retrieval constraint policy used by promoted linking."""

    return json.loads(json.dumps(CONSTRAINT_DECISION_POLICY, sort_keys=True, ensure_ascii=True))


def retrieval_stack_contract_payload(*, retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> dict[str, Any]:
    """Return the retrieval-stack contract covered by artifact metadata."""

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "retrieval_method": "global_hybrid_centroid",
        "retrieval_top_k": int(retrieval_top_k),
        "query_view_policy": "caller_supplied_initial_only_or_full_or_production_auto",
        "orcid_policy": "return_all_matches_force_link_exempt_from_disallow_veto",
        "candidate_filter_policy": "post_retrieval_constraint_row_policy",
        "constraint_decision_policy": retrieval_constraint_decision_policy_payload(),
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
    if metadata.get("gate_surface") != GATE_SURFACE_PROMOTED_LOGISTIC:
        raise ValueError(f"Unsupported incremental linker gate_surface: {metadata.get('gate_surface')!r}")
    binding = metadata.get("pairwise_bundle_binding")
    required_binding_fields = {
        "normalization_version",
        "featurizer_version",
        "ordered_feature_contract_digest",
        "main_booster_sha256",
        "nameless_booster_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding_fields:
        raise ValueError("Incremental linker artifact pairwise_bundle_binding is missing or malformed")
    if binding["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("Incremental linker artifact normalization_version mismatch")
    binding_featurizer_version = binding["featurizer_version"]
    if not isinstance(binding_featurizer_version, int) or isinstance(binding_featurizer_version, bool):
        raise ValueError("Incremental linker artifact featurizer_version must be an integer")
    if binding_featurizer_version != FEATURIZER_VERSION:
        raise ValueError("Incremental linker artifact featurizer_version mismatch")
    for field in ("ordered_feature_contract_digest", "main_booster_sha256", "nameless_booster_sha256"):
        digest = binding[field]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"Incremental linker artifact pairwise_bundle_binding {field} is not a SHA-256")
