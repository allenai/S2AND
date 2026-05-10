"""Compact non-pairwise linker feature formulas."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.text import normalize_text

logger = logging.getLogger("s2and")

CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE = 192.0
GENERIC_HEURISTIC_OVERRIDE_MARGIN = 0.01
GENERIC_CROSS_FAMILY_EXTRA_MARGIN = 0.04
GENERIC_FAMILY_MIN_COUNT = 3
GENERIC_FAMILY_MIN_RATIO = 0.6
EXACT_TITLE_ANCHOR_THRESHOLD = 0.95
ANCHOR_SUPPORT_OVERLAP_THRESHOLD = 0.05
ANCHOR_YEAR_COMPATIBILITY_THRESHOLD = 0.85
SEVERE_CONTRADICTION_THRESHOLD = 0.75

_REQUIRED_BASE_SIGNALS: frozenset[str] = frozenset(
    {
        "retrieval_score",
        "retrieval_rank",
        "candidate_component_key",
        "query_view",
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
        "query_has_coauthors",
        "middle_initial_compatibility",
        "affiliation_overlap",
        "coauthor_overlap",
        "venue_overlap",
        "year_compatibility",
        "title_overlap",
        "specter_exemplar_similarity",
        "min_distance",
        "top5_mean_distance",
        "pair_count",
        "last_name_count_min_rarity",
        "candidate_last_first_name_count_min_rarity",
        "last_first_name_count_min_rarity",
        "first_prefix_x_last_first_name_count_min_rarity",
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


def _best_index_by_group(
    primary: np.ndarray,
    retrieval_ranks: np.ndarray,
    component_keys: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    higher_is_better: bool,
) -> np.ndarray:
    out = np.zeros(len(primary), dtype=np.uint32)
    for group in groups:
        best = min(
            (int(index) for index in group),
            key=lambda index: (
                -float(primary[index]) if higher_is_better else float(primary[index]),
                int(retrieval_ranks[index]),
                str(component_keys[index]),
            ),
        )
        out[group] = best
    return out


def _retrieval_ordered_groups(
    groups: Sequence[np.ndarray],
    retrieval_rank: np.ndarray,
    component_keys: np.ndarray,
) -> list[list[int]]:
    return [
        sorted(
            (int(index) for index in group),
            key=lambda idx: (int(retrieval_rank[idx]), str(component_keys[idx])),
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


def _cluster_size_log_capped(cluster_size: np.ndarray) -> np.ndarray:
    values = np.maximum(cluster_size.astype(np.float32), np.float32(0.0))
    out = np.zeros_like(values, dtype=np.float32)
    observed = values > 0.0
    out[observed] = np.minimum(
        np.float32(1.0),
        np.log1p(values[observed]) / np.float32(math.log1p(CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE)),
    )
    return out


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


def _confident_family_mask(family_ids: np.ndarray, component_keys: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            bool(str(family)) and str(family) != str(component)
            for family, component in zip(family_ids, component_keys, strict=True)
        ],
        dtype=bool,
    )


def _cross_family(left: int, right: int, family_ids: np.ndarray, confident_family: np.ndarray) -> bool:
    return bool(confident_family[left] and confident_family[right] and str(family_ids[left]) != str(family_ids[right]))


def _coarse_family_keys(
    dominant_first_alpha: np.ndarray,
    family_alpha: np.ndarray,
    component_alpha: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(dominant_first_alpha), dtype=object)
    for index, (dominant, family, component) in enumerate(
        zip(dominant_first_alpha, family_alpha, component_alpha, strict=True)
    ):
        out[index] = str(dominant or family or component)[:3]
    return out


def _first_name_compatibility(query_first: Any, dominant_first: Any) -> float:
    query = _normalize_alpha(query_first)
    dominant = _normalize_alpha(dominant_first)
    if not dominant:
        return 0.0
    if query and (query == dominant or query.startswith(dominant) or dominant.startswith(query)):
        return 1.0
    if query and dominant.startswith(query[0]):
        return 1.0
    return 0.0


def _year_mismatch_severity(
    query_year: np.ndarray,
    query_year_missing: np.ndarray,
    candidate_year_min: np.ndarray,
    candidate_year_max: np.ndarray,
    candidate_year_range_missing: np.ndarray,
) -> np.ndarray:
    out = np.zeros(len(query_year), dtype=np.float32)
    observed = (query_year_missing == 0) & (candidate_year_range_missing == 0)
    lower = observed & (query_year < candidate_year_min)
    upper = observed & (query_year > candidate_year_max)
    out[lower] = np.minimum(1.0, (candidate_year_min[lower] - query_year[lower]).astype(np.float32) / 10.0)
    out[upper] = np.minimum(1.0, (query_year[upper] - candidate_year_max[upper]).astype(np.float32) / 10.0)
    return np.round(out, 6)


def _derive_group_features(
    *,
    ordered_groups: Sequence[Sequence[int]],
    retrieval_score: np.ndarray,
    retrieval_rank: np.ndarray,
    component_keys: np.ndarray,
    family_ids: np.ndarray,
    confident_family: np.ndarray,
    coarse_family_keys: np.ndarray,
    pair_count: np.ndarray,
    top5_mean_distance: np.ndarray,
) -> dict[str, np.ndarray]:
    row_count = len(retrieval_score)
    retrieval_score_gap_vs_best_competitor = np.zeros(row_count, dtype=np.float32)
    retrieval_score_best_gap = np.zeros(row_count, dtype=np.float32)
    same_family_as_top1 = np.zeros(row_count, dtype=np.float32)
    same_family_as_best_top5 = np.zeros(row_count, dtype=np.float32)
    same_family_as_heuristic_choice = np.zeros(row_count, dtype=np.float32)
    coarse_family_top5_best_gap = np.zeros(row_count, dtype=np.float32)
    candidate_pair_share = np.ones(row_count, dtype=np.float32)

    for ordered in ordered_groups:
        group = np.asarray(ordered, dtype=np.uint32)
        top1 = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else ordered[0]
        best_top5 = min(
            ordered,
            key=lambda idx: (float(top5_mean_distance[idx]), int(retrieval_rank[idx]), str(component_keys[idx])),
        )
        heuristic_choice = best_top5
        if best_top5 != top1:
            effective_margin = GENERIC_HEURISTIC_OVERRIDE_MARGIN
            if _cross_family(top1, best_top5, family_ids, confident_family):
                effective_margin += GENERIC_CROSS_FAMILY_EXTRA_MARGIN
            if float(top5_mean_distance[best_top5]) + effective_margin >= float(top5_mean_distance[top1]):
                heuristic_choice = top1
        best_score = float(np.max(retrieval_score[group]))
        for index in group:
            competitor = runner_up if int(index) == top1 else top1
            retrieval_score_gap_vs_best_competitor[index] = float(retrieval_score[index] - retrieval_score[competitor])
            retrieval_score_best_gap[index] = float(best_score - retrieval_score[index])
            same_family_as_top1[index] = float(
                bool(family_ids[index]) and str(family_ids[index]) == str(family_ids[top1])
            )
            same_family_as_best_top5[index] = float(
                bool(family_ids[index]) and str(family_ids[index]) == str(family_ids[best_top5])
            )
            same_family_as_heuristic_choice[index] = float(
                bool(family_ids[index]) and str(family_ids[index]) == str(family_ids[heuristic_choice])
            )

        coarse_to_rows: dict[str, list[int]] = {}
        for index in group:
            coarse = str(coarse_family_keys[index])
            coarse_to_rows.setdefault(coarse, []).append(int(index))
        for coarse_rows in coarse_to_rows.values():
            total_pairs = max(1.0, float(np.sum(pair_count[coarse_rows])))
            best = min(
                coarse_rows,
                key=lambda idx: (float(top5_mean_distance[idx]), int(retrieval_rank[idx]), str(component_keys[idx])),
            )
            for index in coarse_rows:
                coarse_family_top5_best_gap[index] = float(top5_mean_distance[index] - top5_mean_distance[best])
                candidate_pair_share[index] = float(pair_count[index] / total_pairs)

    return {
        "retrieval_score_gap_vs_best_competitor": np.round(retrieval_score_gap_vs_best_competitor, 6),
        "retrieval_score_best_gap": np.round(retrieval_score_best_gap, 6),
        "same_family_as_top1": same_family_as_top1,
        "same_family_as_best_top5": same_family_as_best_top5,
        "same_family_as_heuristic_choice": same_family_as_heuristic_choice,
        "coarse_family_top5_best_gap": np.round(coarse_family_top5_best_gap, 6),
        "candidate_pair_share_within_coarse_family": np.round(candidate_pair_share, 6),
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
    query_view = _object_signal(row_signals, "query_view", row_count)
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
    query_has_coauthors = _float_signal(row_signals, "query_has_coauthors", row_count)
    affiliation_overlap = _float_signal(row_signals, "affiliation_overlap", row_count)
    coauthor_overlap = _float_signal(row_signals, "coauthor_overlap", row_count)
    venue_overlap = _float_signal(row_signals, "venue_overlap", row_count)
    year_compatibility = _float_signal(row_signals, "year_compatibility", row_count)
    title_overlap = _float_signal(row_signals, "title_overlap", row_count)
    specter_exemplar_similarity = _float_signal(row_signals, "specter_exemplar_similarity", row_count)
    min_distance = _float_signal(row_signals, "min_distance", row_count)
    top5_mean_distance = _float_signal(row_signals, "top5_mean_distance", row_count)
    pair_count = _float_signal(row_signals, "pair_count", row_count)

    groups = _groups(candidate_batch)
    ordered_groups = _retrieval_ordered_groups(groups, retrieval_rank, component_keys)
    family_ids = (
        _object_signal(row_signals, "family_id", row_count)
        if "family_id" in row_signals
        else _family_ids(component_keys, dominant_first_name, named_signature_count, cluster_size)
    )
    confident_family = _confident_family_mask(family_ids, component_keys)
    query_first_alpha = _normalized_alpha_array(query_first_token)
    dominant_first_alpha = _normalized_alpha_array(dominant_first_name)
    family_alpha = _normalized_alpha_array(family_ids)
    component_alpha = _normalized_alpha_array(component_keys)
    coarse_family_keys = _coarse_family_keys(dominant_first_alpha, family_alpha, component_alpha)
    best_top5_indices = _best_index_by_group(
        top5_mean_distance,
        retrieval_rank,
        component_keys,
        groups,
        higher_is_better=False,
    )
    group_features = _derive_group_features(
        ordered_groups=ordered_groups,
        retrieval_score=retrieval_score,
        retrieval_rank=retrieval_rank,
        component_keys=component_keys,
        family_ids=family_ids,
        confident_family=confident_family,
        coarse_family_keys=coarse_family_keys,
        pair_count=pair_count,
        top5_mean_distance=top5_mean_distance,
    )
    year_mismatch_severity = _year_mismatch_severity(
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
    first_name_compatibility = np.asarray(
        [
            1.0
            if dominant
            and (
                (query and (query == dominant or query.startswith(dominant) or dominant.startswith(query)))
                or (query and dominant.startswith(query[0]))
            )
            else 0.0
            for query, dominant in zip(query_first_alpha, dominant_first_alpha, strict=True)
        ],
        dtype=np.float32,
    )
    coauthor_contradiction = np.where(
        query_has_coauthors > 0.0,
        np.maximum(0.0, 1.0 - coauthor_overlap),
        0.0,
    ).astype(np.float32)
    title_anchor = title_overlap >= EXACT_TITLE_ANCHOR_THRESHOLD
    contradiction = np.maximum.reduce(
        [
            year_mismatch_severity,
            affiliation_contradiction_severity,
            np.where(title_anchor, coauthor_contradiction, 0.0),
            np.where(title_anchor & (first_name_compatibility <= 0.0), 1.0, 0.0),
        ]
    ).astype(np.float32)
    exact_anchor_evidence_flag = (
        (title_overlap >= EXACT_TITLE_ANCHOR_THRESHOLD)
        & (
            (coauthor_overlap >= ANCHOR_SUPPORT_OVERLAP_THRESHOLD)
            | (affiliation_overlap >= ANCHOR_SUPPORT_OVERLAP_THRESHOLD)
            | (year_compatibility >= ANCHOR_YEAR_COMPATIBILITY_THRESHOLD)
        )
    ).astype(np.float32)

    same_top1 = group_features["same_family_as_top1"].astype(np.float32)
    candidate_pair_share = group_features["candidate_pair_share_within_coarse_family"].astype(np.float32)
    retrieval_gap = group_features["retrieval_score_gap_vs_best_competitor"].astype(np.float32)
    anchor_evidence_count = (
        (min_distance <= 0.15).astype(np.float32)
        + (specter_exemplar_similarity >= 0.70).astype(np.float32)
        + (title_overlap >= 0.20).astype(np.float32)
        + (coauthor_overlap >= 0.25).astype(np.float32)
        + (affiliation_overlap >= 0.25).astype(np.float32)
        + (venue_overlap >= 0.20).astype(np.float32)
        + (year_compatibility >= 0.90).astype(np.float32)
        + (retrieval_gap >= 0.02).astype(np.float32)
    )
    support_strength = (
        0.20 * (1.0 - np.clip(min_distance, 0.0, 1.0))
        + 0.20 * np.clip(specter_exemplar_similarity, 0.0, 1.0)
        + 0.18 * np.clip(title_overlap, 0.0, 1.0)
        + 0.18 * np.clip(coauthor_overlap, 0.0, 1.0)
        + 0.12 * np.clip(affiliation_overlap, 0.0, 1.0)
        + 0.06 * np.clip(venue_overlap, 0.0, 1.0)
        + 0.06 * np.clip(year_compatibility, 0.0, 1.0)
    )
    strong_positive_anchor_score = (
        np.clip(support_strength, 0.0, 1.0)
        * (0.5 + 0.5 * np.clip(same_top1, 0.0, 1.0))
        * (0.35 + 0.65 * np.clip(1.0 - contradiction, 0.0, 1.0))
    )
    retrieval_gap_scaled = np.clip((np.clip(retrieval_gap, -0.2, 0.3) + 0.2) / 0.5, 0.0, 1.0)
    residual_support = (
        0.28 * (1.0 - np.clip(min_distance, 0.0, 1.0))
        + 0.20 * np.clip(specter_exemplar_similarity, 0.0, 1.0)
        + 0.20 * np.clip(coauthor_overlap, 0.0, 1.0)
        + 0.14 * np.clip(title_overlap, 0.0, 1.0)
        + 0.10 * np.clip(year_compatibility, 0.0, 1.0)
        + 0.08 * retrieval_gap_scaled
    )
    tiny_candidate = ((cluster_size <= 2.0) | (named_signature_count <= 2.0)).astype(np.float32)
    weak_residual_anchor_score = tiny_candidate * same_top1 * np.clip(residual_support, 0.0, 1.0)
    sparse_relative_winner_score = (
        (retrieval_rank <= 1.0).astype(np.float32)
        * same_top1
        * np.clip(np.clip(retrieval_gap, 0.0, 0.3) / 0.3, 0.0, 1.0)
        * (1.0 - np.clip(candidate_pair_share, 0.0, 1.0))
        * np.clip(residual_support, 0.0, 1.0)
    )
    query_first_prefix_match = np.asarray(
        [
            1.0
            if query_first
            and len(query_first) > 1
            and dominant
            and (query_first.startswith(dominant) or dominant.startswith(query_first))
            else 0.0
            for query_first, dominant in zip(query_first_alpha, dominant_first_alpha, strict=True)
        ],
        dtype=np.float32,
    )

    out: dict[str, np.ndarray] = {
        "min_distance": min_distance,
        "retrieval_score_gap_vs_best_competitor": group_features["retrieval_score_gap_vs_best_competitor"],
        "top5_distance_best_gap": np.round(top5_mean_distance - top5_mean_distance[best_top5_indices], 6),
        "retrieval_score": retrieval_score,
        "affiliation_contradiction_severity": np.round(affiliation_contradiction_severity, 6),
        "coarse_family_top5_best_gap": group_features["coarse_family_top5_best_gap"],
        "same_family_as_best_top5": group_features["same_family_as_best_top5"],
        "same_family_as_heuristic_choice": group_features["same_family_as_heuristic_choice"],
        "same_family_as_top1": same_top1,
        "query_first_prefix_match": query_first_prefix_match,
        "retrieval_score_best_gap": group_features["retrieval_score_best_gap"],
        "cluster_size_log_capped": _cluster_size_log_capped(cluster_size),
        "anchor_evidence_count": anchor_evidence_count.astype(np.float32),
        "strong_positive_anchor_score": np.round(strong_positive_anchor_score, 6).astype(np.float32),
        "weak_residual_anchor_score": np.round(weak_residual_anchor_score, 6).astype(np.float32),
        "sparse_relative_winner_score": np.round(sparse_relative_winner_score, 6).astype(np.float32),
        "query_view__initial_only": np.asarray(
            [float(str(value) == "initial_only") for value in query_view],
            dtype=np.float32,
        ),
        "last_name_count_min_rarity": _float_signal(row_signals, "last_name_count_min_rarity", row_count),
        "candidate_last_first_name_count_min_rarity": _float_signal(
            row_signals,
            "candidate_last_first_name_count_min_rarity",
            row_count,
        ),
        "last_first_name_count_min_rarity": _float_signal(row_signals, "last_first_name_count_min_rarity", row_count),
        "first_prefix_x_last_first_name_count_min_rarity": _float_signal(
            row_signals,
            "first_prefix_x_last_first_name_count_min_rarity",
            row_count,
        ),
        "exact_anchor_evidence_flag": exact_anchor_evidence_flag,
        "year_mismatch_severity": year_mismatch_severity,
        "top5_mean_distance": top5_mean_distance,
        "distance_spread_top5_minus_min": np.round(top5_mean_distance - min_distance, 6),
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
        "query_view": _rust_string_signal(row_signals, "query_view", row_count),
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
        "query_has_coauthors",
        "middle_initial_compatibility",
        "affiliation_overlap",
        "coauthor_overlap",
        "venue_overlap",
        "year_compatibility",
        "title_overlap",
        "specter_exemplar_similarity",
        "min_distance",
        "top5_mean_distance",
        "pair_count",
        "last_name_count_min_rarity",
        "candidate_last_first_name_count_min_rarity",
        "last_first_name_count_min_rarity",
        "first_prefix_x_last_first_name_count_min_rarity",
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
    features = {
        column: np.asarray(result[column], dtype=np.float32) for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
    }
    return features, telemetry


def build_promoted_non_pairwise_row_features(
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Build promoted non-`pw_*` linker features with the Rust row-formula kernel."""

    features, _telemetry = build_promoted_non_pairwise_row_features_with_telemetry(candidate_batch, row_signals)
    return features
