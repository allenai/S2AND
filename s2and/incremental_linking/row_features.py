"""Compact non-pairwise linker feature formulas."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.text import normalize_text

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


def _groups(candidate_batch: LinkerCandidateBatch) -> list[np.ndarray]:
    row_count = candidate_batch.row_count
    if candidate_batch.row_query_signature_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for group-derived features")
    query_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)
    order = np.argsort(query_indices, kind="stable")
    if len(order) == 0:
        return []
    ordered_queries = query_indices[order]
    starts = np.flatnonzero(ordered_queries[1:] != ordered_queries[:-1]) + 1
    return [chunk for chunk in np.split(order, starts) if len(chunk) and int(chunk.max()) < row_count]


def _retrieval_ordered_groups(
    groups: Sequence[np.ndarray],
    retrieval_score: np.ndarray,
    retrieval_rank: np.ndarray,
    component_keys: np.ndarray,
) -> list[list[int]]:
    return [
        sorted(
            (int(index) for index in group),
            key=lambda idx: (-float(retrieval_score[idx]), int(retrieval_rank[idx]), str(component_keys[idx])),
        )
        for group in groups
    ]


def _normalize_alpha(value: Any) -> str:
    normalized = normalize_text(str(value or ""))
    return "".join(character for character in normalized if character.isalpha())


def _normalized_alpha_array(values: np.ndarray) -> np.ndarray:
    cache: dict[str, str] = {}
    out: list[str] = []
    for value in values:
        key = str(value or "")
        normalized = cache.get(key)
        if normalized is None:
            normalized = _normalize_alpha(key)
            cache[key] = normalized
        out.append(normalized)
    return np.asarray(out, dtype=object)


def _cluster_size_log(cluster_size: np.ndarray) -> np.ndarray:
    values = np.maximum(cluster_size.astype(np.float32), np.float32(0.0))
    return np.log1p(values).astype(np.float32)


def _family_ids(
    component_keys: np.ndarray,
    dominant_first_names: np.ndarray,
    named_signature_count: np.ndarray,
    cluster_size: np.ndarray,
) -> np.ndarray:
    out = np.asarray([str(value) for value in component_keys], dtype=object)
    for index, dominant_first in enumerate(dominant_first_names):
        dominant = str(dominant_first or "")
        named_count = float(named_signature_count[index])
        dominance_ratio = float(named_count / max(1.0, float(cluster_size[index])))
        if dominant and named_count >= GENERIC_FAMILY_MIN_COUNT and dominance_ratio >= GENERIC_FAMILY_MIN_RATIO:
            out[index] = dominant
    return out


def _year_gap_to_candidate_range(
    query_year: np.ndarray,
    query_year_missing: np.ndarray,
    candidate_year_min: np.ndarray,
    candidate_year_max: np.ndarray,
    candidate_year_range_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gap = np.zeros(len(query_year), dtype=np.float32)
    signed_gap = np.zeros(len(query_year), dtype=np.float32)
    span = np.zeros(len(query_year), dtype=np.float32)
    candidate_observed = candidate_year_range_missing == 0
    span[candidate_observed] = np.maximum(
        np.float32(0.0),
        candidate_year_max[candidate_observed] - candidate_year_min[candidate_observed],
    )
    observed = (query_year_missing == 0) & candidate_observed
    lower = observed & (query_year < candidate_year_min)
    upper = observed & (query_year > candidate_year_max)
    lower_gap = (candidate_year_min[lower] - query_year[lower]).astype(np.float32)
    upper_gap = (query_year[upper] - candidate_year_max[upper]).astype(np.float32)
    gap[lower] = lower_gap
    gap[upper] = upper_gap
    signed_gap[lower] = -lower_gap
    signed_gap[upper] = upper_gap
    return np.round(gap, 6), np.round(signed_gap, 6), np.round(span, 6)


def _normalized_alpha_lengths(values: np.ndarray) -> np.ndarray:
    return np.asarray([float(len(str(value or ""))) for value in values], dtype=np.float32)


def _derive_group_features(
    *,
    ordered_groups: Sequence[Sequence[int]],
    retrieval_score: np.ndarray,
    retrieval_rank: np.ndarray,
    component_keys: np.ndarray,
    family_ids: np.ndarray,
    dominant_first_alpha: np.ndarray,
    top5_mean_distance: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(retrieval_score)
    retrieval_score_gap_vs_best_competitor = np.zeros(row_count, dtype=np.float32)
    same_family_as_top1 = np.zeros(row_count, dtype=np.float32)
    same_family_as_best_top5 = np.zeros(row_count, dtype=np.float32)
    same_family_as_heuristic_choice = np.zeros(row_count, dtype=np.float32)
    dominant_first_top1_match = np.zeros(row_count, dtype=np.float32)
    same_dominant_first_as_best_top5 = np.zeros(row_count, dtype=np.float32)
    current_retrieval_rank = np.zeros(row_count, dtype=np.float32)

    for ordered in ordered_groups:
        top1 = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
        best_top5 = min(
            ordered,
            key=lambda idx: (float(top5_mean_distance[idx]), int(retrieval_rank[idx]), str(component_keys[idx])),
        )
        for current_rank, index in enumerate(ordered, start=1):
            competitor = runner_up if int(index) == top1 else top1
            current_retrieval_rank[index] = float(current_rank)
            retrieval_score_gap_vs_best_competitor[index] = float(retrieval_score[index] - retrieval_score[competitor])
            same_family_as_top1[index] = float(
                bool(family_ids[index]) and str(family_ids[index]) == str(family_ids[top1])
            )
            same_family_as_best_top5[index] = float(
                bool(family_ids[index]) and str(family_ids[index]) == str(family_ids[best_top5])
            )
            dominant_first_top1_match[index] = float(
                bool(dominant_first_alpha[index])
                and str(dominant_first_alpha[index]) == str(dominant_first_alpha[top1])
            )
            same_dominant_first_as_best_top5[index] = float(
                bool(dominant_first_alpha[index])
                and str(dominant_first_alpha[index]) == str(dominant_first_alpha[best_top5])
            )
            same_family_as_heuristic_choice[index] = float(
                dominant_first_top1_match[index] * retrieval_score[index]
                + same_dominant_first_as_best_top5[index] * (1.0 - top5_mean_distance[index])
            )

    return {
        "retrieval_score_gap_vs_best_competitor": np.round(retrieval_score_gap_vs_best_competitor, 6),
        "same_family_as_top1": same_family_as_top1,
        "same_family_as_heuristic_choice": np.round(same_family_as_heuristic_choice, 6),
        "same_dominant_first_as_best_top5": same_dominant_first_as_best_top5,
        "current_retrieval_rank": current_retrieval_rank,
    }


def _build_promoted_non_pairwise_row_features_python_reference(
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Python parity-test oracle for the promoted non-`pw_*` linker features."""

    row_count = candidate_batch.row_count
    missing = sorted(signal for signal in _REQUIRED_BASE_SIGNALS if signal not in row_signals)
    if missing:
        raise KeyError(f"Missing compact linker row signals: {missing}")

    retrieval_score = _float_signal(row_signals, "retrieval_score", row_count)
    retrieval_rank = _float_signal(row_signals, "retrieval_rank", row_count)
    component_keys = _object_signal(row_signals, "candidate_component_key", row_count)
    cluster_size = _float_signal(row_signals, "cluster_size", row_count)
    named_signature_count = _float_signal(row_signals, "named_signature_count", row_count)
    dominant_first_name = _object_signal(row_signals, "dominant_first_name", row_count)
    candidate_year_min = _float_signal(row_signals, "candidate_year_min", row_count)
    candidate_year_max = _float_signal(row_signals, "candidate_year_max", row_count)
    candidate_year_range_missing = _float_signal(row_signals, "candidate_year_range_missing", row_count)
    query_first_token = _object_signal(row_signals, "query_first_token", row_count)
    query_year = _float_signal(row_signals, "query_year", row_count)
    query_year_missing = _float_signal(row_signals, "query_year_missing", row_count)
    query_has_affiliations = _float_signal(row_signals, "query_has_affiliations", row_count)
    affiliation_overlap = _float_signal(row_signals, "affiliation_overlap", row_count)
    coauthor_overlap = _float_signal(row_signals, "coauthor_overlap", row_count)
    year_compatibility = _float_signal(row_signals, "year_compatibility", row_count)
    specter_exemplar_similarity = _float_signal(row_signals, "specter_exemplar_similarity", row_count)
    min_distance = _float_signal(row_signals, "min_distance", row_count)
    top5_mean_distance = _float_signal(row_signals, "top5_mean_distance", row_count)
    candidate_cluster_max_paper_author_count = _float_signal(
        row_signals,
        "candidate_cluster_max_paper_author_count",
        row_count,
    )
    paper_author_list_max_jaccard = _float_signal(row_signals, "paper_author_list_max_jaccard", row_count)
    paper_author_list_max_containment = _float_signal(row_signals, "paper_author_list_max_containment", row_count)
    paper_author_list_max_overlap_count = _float_signal(row_signals, "paper_author_list_max_overlap_count", row_count)
    local_author_window10_jaccard_max = _float_signal(row_signals, "local_author_window10_jaccard_max", row_count)
    local_author_window10_overlap_count_max = _float_signal(
        row_signals,
        "local_author_window10_overlap_count_max",
        row_count,
    )
    best_author_count_log_absdiff = _float_signal(row_signals, "best_author_count_log_absdiff", row_count)

    groups = _groups(candidate_batch)
    ordered_groups = _retrieval_ordered_groups(groups, retrieval_score, retrieval_rank, component_keys)
    family_ids = (
        _object_signal(row_signals, "family_id", row_count)
        if "family_id" in row_signals
        else _family_ids(component_keys, dominant_first_name, named_signature_count, cluster_size)
    )
    query_first_alpha = _normalized_alpha_array(query_first_token)
    dominant_first_alpha = _normalized_alpha_array(dominant_first_name)
    group_features = _derive_group_features(
        ordered_groups=ordered_groups,
        retrieval_score=retrieval_score,
        retrieval_rank=retrieval_rank,
        component_keys=component_keys,
        family_ids=family_ids,
        dominant_first_alpha=dominant_first_alpha,
        top5_mean_distance=top5_mean_distance,
    )
    year_gap_to_candidate_range, year_gap_signed_to_candidate_range, candidate_year_span = _year_gap_to_candidate_range(
        query_year,
        query_year_missing,
        candidate_year_min,
        candidate_year_max,
        candidate_year_range_missing,
    )
    affiliation_contradiction_severity = np.where(
        query_has_affiliations > 0.0,
        np.maximum(0.0, 1.0 - affiliation_overlap),
        0.0,
    ).astype(np.float32)

    same_top1 = group_features["same_family_as_top1"].astype(np.float32)
    retrieval_gap = group_features["retrieval_score_gap_vs_best_competitor"].astype(np.float32)
    anchor_evidence_count = (min_distance <= 0.15).astype(np.float32) + (retrieval_gap >= 0.02).astype(np.float32)
    distance_signal = 1.0 - np.clip(min_distance, 0.0, 1.0)
    support_strength = 0.20 * distance_signal
    strong_positive_anchor_score = np.clip(support_strength, 0.0, 1.0) * (0.5 + 0.5 * np.clip(same_top1, 0.0, 1.0))
    retrieval_gap_scaled = np.clip((np.clip(retrieval_gap, -0.2, 0.3) + 0.2) / 0.5, 0.0, 1.0)
    residual_support = 0.28 * distance_signal + 0.08 * retrieval_gap_scaled
    weak_residual_anchor_score = same_top1 * np.clip(residual_support, 0.0, 1.0)
    sparse_relative_winner_score = (
        (group_features["current_retrieval_rank"] <= 1.0).astype(np.float32)
        * same_top1
        * np.clip(np.clip(retrieval_gap, 0.0, 0.3) / 0.3, 0.0, 1.0)
        * np.clip(residual_support, 0.0, 1.0)
    )
    query_first_prefix_match_any_length = np.asarray(
        [
            1.0
            if query_first and dominant and (query_first.startswith(dominant) or dominant.startswith(query_first))
            else 0.0
            for query_first, dominant in zip(query_first_alpha, dominant_first_alpha, strict=True)
        ],
        dtype=np.float32,
    )

    out: dict[str, np.ndarray] = {
        "min_distance": min_distance,
        "affiliation_contradiction_severity": np.round(affiliation_contradiction_severity, 6),
        "same_family_as_heuristic_choice": group_features["same_family_as_heuristic_choice"],
        "same_dominant_first_as_best_top5": group_features["same_dominant_first_as_best_top5"],
        "specter_exemplar_similarity": specter_exemplar_similarity,
        "coauthor_overlap": coauthor_overlap,
        "affiliation_overlap": affiliation_overlap,
        "year_compatibility": year_compatibility,
        "retrieval_rank": group_features["current_retrieval_rank"],
        "retrieval_reciprocal_rank": np.round(
            1.0 / np.maximum(group_features["current_retrieval_rank"].astype(np.float32), np.float32(1.0)),
            6,
        ),
        "cluster_size_log": _cluster_size_log(cluster_size),
        "candidate_year_span": candidate_year_span,
        "year_gap_to_candidate_range": year_gap_to_candidate_range,
        "year_gap_signed_to_candidate_range": year_gap_signed_to_candidate_range,
        "candidate_dominant_first_name_length": _normalized_alpha_lengths(dominant_first_alpha),
        "query_first_prefix_match_any_length": query_first_prefix_match_any_length,
        "candidate_cluster_max_paper_author_count": candidate_cluster_max_paper_author_count,
        "paper_author_list_max_jaccard": paper_author_list_max_jaccard,
        "paper_author_list_max_containment": paper_author_list_max_containment,
        "paper_author_list_max_overlap_count": paper_author_list_max_overlap_count,
        "local_author_window10_jaccard_max": local_author_window10_jaccard_max,
        "local_author_window10_overlap_count_max": local_author_window10_overlap_count_max,
        "best_author_count_log_absdiff": best_author_count_log_absdiff,
        "anchor_evidence_count": anchor_evidence_count.astype(np.float32),
        "strong_positive_anchor_score": np.round(strong_positive_anchor_score, 6).astype(np.float32),
        "weak_residual_anchor_score": np.round(weak_residual_anchor_score, 6).astype(np.float32),
        "sparse_relative_winner_score": np.round(sparse_relative_winner_score, 6).astype(np.float32),
        "last_name_count_min_rarity": _float_signal(row_signals, "last_name_count_min_rarity", row_count),
        "last_first_name_count_min_rarity": _float_signal(row_signals, "last_first_name_count_min_rarity", row_count),
        "top5_mean_distance": top5_mean_distance,
    }
    missing_output = sorted(set(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS) - set(out))
    if missing_output:
        raise RuntimeError(f"Promoted row feature builder did not produce columns: {missing_output}")
    return {column: np.asarray(out[column], dtype=np.float32) for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS}


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
    if "family_id" in row_signals:
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
        return {}
    telemetry: dict[str, int] = {}
    for key in ("generated_family_id_count", "generic_family_override_count"):
        if key in raw_telemetry:
            telemetry[key] = int(raw_telemetry[key])
    return telemetry


def build_promoted_non_pairwise_row_features_with_telemetry(
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build promoted non-`pw_*` linker features and return Rust row-formula telemetry."""

    try:
        import s2and_rust
    except ImportError as exc:  # pragma: no cover - production requires the Rust runtime
        raise RuntimeError("s2and_rust is required for promoted linker row feature generation") from exc
    method = getattr(s2and_rust, "promoted_linker_non_pairwise_features", None)
    if method is None:
        raise RuntimeError("s2and_rust.promoted_linker_non_pairwise_features is unavailable")
    result = method(_rust_payload(candidate_batch, row_signals))
    telemetry = _coerce_promoted_row_feature_telemetry(result)
    if telemetry:
        logger.info(
            "Telemetry: promoted_linker_non_pairwise_features generated_family_id_count=%d "
            "generic_family_override_count=%d",
            telemetry.get("generated_family_id_count", 0),
            telemetry.get("generic_family_override_count", 0),
        )
    result_payload = dict(result)
    for passthrough_column in (
        "candidate_cluster_max_paper_author_count",
        "paper_author_list_max_jaccard",
        "paper_author_list_max_containment",
        "paper_author_list_max_overlap_count",
        "local_author_window10_jaccard_max",
        "local_author_window10_overlap_count_max",
        "best_author_count_log_absdiff",
    ):
        if passthrough_column not in result_payload and passthrough_column in row_signals:
            result_payload[passthrough_column] = _float_signal(
                row_signals,
                passthrough_column,
                candidate_batch.row_count,
            )
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
