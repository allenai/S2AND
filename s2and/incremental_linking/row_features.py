"""Compact non-pairwise linker feature formulas."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.runtime import load_s2and_rust_extension

s2and_rust = load_s2and_rust_extension()

logger = logging.getLogger("s2and")

GENERIC_FAMILY_MIN_COUNT = 3
GENERIC_FAMILY_MIN_RATIO = 0.6

_REQUIRED_BASE_SIGNALS: frozenset[str] = frozenset(
    {
        "retrieval_score",
        "retrieval_rank",
        "candidate_component_key",
        "cluster_size",
        "named_signature_count",
        "dominant_first_name",
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_first_token",
        "query_year",
        "query_year_missing",
        "query_has_affiliations",
        "affiliation_overlap",
        "coauthor_overlap",
        "year_compatibility",
        "specter_exemplar_similarity",
        "min_distance",
        "top5_mean_distance",
        "last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
        "candidate_cluster_max_paper_author_count",
        "paper_author_list_max_jaccard",
        "paper_author_list_max_containment",
        "paper_author_list_max_overlap_count",
        "local_author_window10_jaccard_max",
        "local_author_window10_overlap_count_max",
        "best_author_count_log_absdiff",
    }
)


def _float_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing compact linker row signal: {name}")
    values = np.asarray(row_signals[name], dtype=np.float32)
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError(f"Signal {name!r} must be 1D with row_count={row_count}, got shape={values.shape}")
    if np.isnan(values).any():
        raise ValueError(f"Signal {name!r} contains NaN values")
    return values


def _object_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing compact linker row signal: {name}")
    values = np.asarray(row_signals[name], dtype=object)
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError(f"Signal {name!r} must be 1D with row_count={row_count}, got shape={values.shape}")
    return values


def _rust_string_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> list[str]:
    return [str(value or "") for value in _object_signal(row_signals, name, row_count)]


def _rust_float_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    return np.ascontiguousarray(_float_signal(row_signals, name, row_count), dtype=np.float32)


def _rust_payload(candidate_batch: LinkerCandidateBatch, row_signals: Mapping[str, Any]) -> dict[str, Any]:
    row_count = candidate_batch.row_count
    missing = sorted(signal for signal in _REQUIRED_BASE_SIGNALS if signal not in row_signals)
    if missing:
        raise KeyError(f"Missing compact linker row signals: {missing}")
    if candidate_batch.row_query_signature_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for group-derived features")

    payload: dict[str, Any] = {
        "row_query_signature_indices": np.ascontiguousarray(
            candidate_batch.row_query_signature_indices,
            dtype=np.uint32,
        ),
        "candidate_component_key": _rust_string_signal(row_signals, "candidate_component_key", row_count),
        "dominant_first_name": _rust_string_signal(row_signals, "dominant_first_name", row_count),
        "query_first_token": _rust_string_signal(row_signals, "query_first_token", row_count),
    }
    if row_signals.get("family_id") is not None:
        payload["family_id"] = _rust_string_signal(row_signals, "family_id", row_count)
    for signal in (
        "retrieval_score",
        "retrieval_rank",
        "cluster_size",
        "named_signature_count",
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
        "query_has_affiliations",
        "affiliation_overlap",
        "coauthor_overlap",
        "year_compatibility",
        "specter_exemplar_similarity",
        "min_distance",
        "top5_mean_distance",
        "last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
        "candidate_cluster_max_paper_author_count",
        "paper_author_list_max_jaccard",
        "paper_author_list_max_containment",
        "paper_author_list_max_overlap_count",
        "local_author_window10_jaccard_max",
        "local_author_window10_overlap_count_max",
        "best_author_count_log_absdiff",
    ):
        payload[signal] = _rust_float_signal(row_signals, signal, row_count)
    for signal in ("mean_distance", "top3_mean_distance"):
        if signal in row_signals:
            payload[signal] = _rust_float_signal(row_signals, signal, row_count)
    if "candidate_last_name_count_min_rarity" in row_signals:
        payload["candidate_last_name_count_min_rarity"] = _rust_float_signal(
            row_signals,
            "candidate_last_name_count_min_rarity",
            row_count,
        )
    return payload


def _coerce_promoted_row_feature_telemetry(result: Mapping[str, Any]) -> dict[str, int]:
    raw_telemetry = result.get("telemetry")
    if not isinstance(raw_telemetry, Mapping):
        raise RuntimeError("Rust promoted row-feature result must include telemetry mapping")
    expected_keys = {"generated_family_id_count", "generic_family_override_count"}
    actual_keys = {str(key) for key in raw_telemetry}
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Rust promoted row-feature telemetry schema mismatch: "
            f"expected={sorted(expected_keys)}, got={sorted(actual_keys)}"
        )
    return {key: int(raw_telemetry[key]) for key in sorted(expected_keys)}


def build_promoted_non_pairwise_row_features_with_telemetry(
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build promoted non-`pw_*` linker features and return Rust row-formula telemetry."""

    result = s2and_rust.promoted_linker_non_pairwise_features(_rust_payload(candidate_batch, row_signals))
    if not isinstance(result, Mapping):
        raise RuntimeError("Rust promoted row-feature result must be a mapping")
    telemetry = _coerce_promoted_row_feature_telemetry(result)
    logger.info(
        "Telemetry: promoted_linker_non_pairwise_features generated_family_id_count=%d "
        "generic_family_override_count=%d",
        telemetry["generated_family_id_count"],
        telemetry["generic_family_override_count"],
    )
    result_payload = dict(result)
    missing_columns = sorted(column for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS if column not in result_payload)
    if missing_columns:
        raise RuntimeError(f"Rust promoted row-feature result is missing columns: {missing_columns}")
    features = {
        column: np.asarray(result_payload[column], dtype=np.float32) for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
    }
    return features, telemetry


def build_promoted_non_pairwise_row_features(
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build promoted non-`pw_*` linker features with the Rust row-formula kernel."""

    features, _telemetry = build_promoted_non_pairwise_row_features_with_telemetry(candidate_batch, row_signals)
    return features
