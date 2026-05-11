"""Artifact-ordered feature assembly for incremental linker candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and.data import ANDData
from s2and.incremental_linking.linker_pairwise import (
    LinkerCandidateBatch,
    PairwiseAggregateStats,
    compute_candidate_batch_pairwise_aggregate_stats_rust,
    promoted_pairwise_aggregate_columns,
    promoted_pairwise_coverage_columns,
)

PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS: tuple[str, ...] = (
    "min_distance",
    "retrieval_score_gap_vs_best_competitor",
    "top5_distance_best_gap",
    "retrieval_score",
    "affiliation_contradiction_severity",
    "coarse_family_top5_best_gap",
    "same_family_as_best_top5",
    "same_family_as_heuristic_choice",
    "same_family_as_top1",
    "query_first_prefix_match",
    "retrieval_score_best_gap",
    "cluster_size_log_capped",
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
    "query_view__initial_only",
    "last_name_count_min_rarity",
    "candidate_last_first_name_count_min_rarity",
    "last_first_name_count_min_rarity",
    "first_prefix_x_last_first_name_count_min_rarity",
    "exact_anchor_evidence_flag",
    "year_mismatch_severity",
    "top5_mean_distance",
    "distance_spread_top5_minus_min",
)
PROMOTED_LINKER_RANK_FRACTION_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "affiliation_overlap_rank_fraction",
        "coauthor_overlap_rank_fraction",
        "mean_distance_rank_fraction",
        "min_distance_rank_fraction",
        "retrieval_rank_fraction",
        "retrieval_score_rank_fraction",
        "title_overlap_rank_fraction",
        "top3_distance_rank_fraction",
        "top5_distance_rank_fraction",
        "venue_overlap_rank_fraction",
        "year_compatibility_rank_fraction",
    }
)
PROMOTED_LINKER_DROPPED_FEATURE_COLUMNS: frozenset[str] = (
    frozenset(
        {
            "candidate_last_name_count_min_rarity",
            "coarse_family_top5_best_gap",
            "exact_anchor_evidence_flag",
            "last_first_name_count_min_rarity",
            "pw_max_email_prefix_equal",
            "pw_max_email_suffix_equal",
            "pw_max_abstract_count",
            "pw_max_coauthor_match",
            "pw_max_english_count",
            "pw_max_first_names_equal",
            "pw_max_first_name_count_max",
            "pw_max_last_first_initial_count_min",
            "pw_max_last_first_name_count_max",
            "pw_max_last_first_name_count_min",
            "pw_max_last_name_count_min",
            "pw_max_lcs",
            "pw_max_levenshtein",
            "pw_max_middle_names_equal",
            "pw_max_middle_one_missing",
            "pw_max_language_reliability_count",
            "pw_max_same_language",
            "pw_max_single_char_first",
            "pw_max_single_char_middle",
            "pw_mean_jaro",
            "pw_mean_last_first_initial_count_min",
            "pw_mean_last_first_name_count_max",
            "pw_mean_last_name_count_min",
            "pw_mean_lcs",
            "pw_mean_levenshtein",
            "pw_mean_middle_initials_overlap",
            "pw_mean_prefix",
            "pw_mean_first_name_count_max",
            "pw_mean_first_name_count_min",
            "pw_min_english_count",
            "pw_min_first_names_equal",
            "pw_min_middle_names_equal",
            "pw_min_middle_one_missing",
            "pw_min_email_prefix_equal",
            "pw_min_email_suffix_equal",
            "pw_min_language_reliability_count",
            "pw_min_last_name_count_min",
            "pw_min_lcs",
            "pw_min_same_language",
            "pw_min_single_char_middle",
            "pw_min_single_char_first",
            "query_view__full",
            "query_view__initial_only",
            "retrieval_score_best_gap",
            "same_family_as_best_top5",
            "same_family_as_top1",
        }
    )
    | PROMOTED_LINKER_RANK_FRACTION_FEATURE_COLUMNS
)
PROMOTED_LINKER_FEATURE_COLUMNS: tuple[str, ...] = (
    *(
        column
        for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
        if column not in PROMOTED_LINKER_DROPPED_FEATURE_COLUMNS
    ),
    *(
        column
        for column in promoted_pairwise_aggregate_columns()
        if column not in PROMOTED_LINKER_DROPPED_FEATURE_COLUMNS
    ),
)
if len(PROMOTED_LINKER_FEATURE_COLUMNS) != 70:
    raise RuntimeError(
        f"Promoted linker feature schema must have 70 columns, got {len(PROMOTED_LINKER_FEATURE_COLUMNS)}"
    )
if PROMOTED_LINKER_DROPPED_FEATURE_COLUMNS.intersection(PROMOTED_LINKER_FEATURE_COLUMNS):
    raise RuntimeError("Promoted linker feature schema still contains dropped columns")


@dataclass(frozen=True)
class LinkerFeatureMatrix:
    """Artifact-ordered feature matrix plus production/training metadata."""

    matrix: np.ndarray
    feature_columns: tuple[str, ...]
    candidate_batch: LinkerCandidateBatch
    pairwise_stats: PairwiseAggregateStats | None = None


def promoted_linker_feature_columns() -> tuple[str, ...]:
    """Return the promoted 70-feature linker/reranker schema in artifact order."""

    return PROMOTED_LINKER_FEATURE_COLUMNS


def _has_column(row_features: Any, column: str) -> bool:
    if isinstance(row_features, Mapping):
        return column in row_features
    columns = getattr(row_features, "columns", None)
    if columns is not None:
        return column in columns
    try:
        row_features[column]
    except (KeyError, TypeError):
        return False
    return True


def _coerce_row_feature_column(row_features: Any, column: str, row_count: int) -> np.ndarray:
    if not _has_column(row_features, column):
        raise KeyError(f"Missing row-level linker feature column: {column}")
    values = np.asarray(row_features[column], dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Feature column {column!r} must be 1D, got shape={values.shape}")
    if len(values) != row_count:
        raise ValueError(f"Feature column {column!r} must have length row_count: {len(values)} != {row_count}")
    if np.isnan(values).any():
        raise ValueError(f"Feature column {column!r} contains NaN values")
    return values


def _pairwise_feature_columns_to_matrix(
    pairwise_stats: PairwiseAggregateStats,
    row_count: int,
) -> dict[str, np.ndarray]:
    pairwise_matrix = np.asarray(pairwise_stats.feature_matrix(), dtype=np.float32)
    if pairwise_matrix.ndim != 2:
        raise ValueError(f"pairwise feature matrix must be 2D, got shape={pairwise_matrix.shape}")
    if pairwise_matrix.shape != (row_count, len(pairwise_stats.aggregate_feature_columns)):
        raise ValueError(
            "pairwise feature matrix shape must match row_count and aggregate columns: "
            f"{pairwise_matrix.shape} != ({row_count}, {len(pairwise_stats.aggregate_feature_columns)})"
        )
    if np.isinf(pairwise_matrix).any():
        raise ValueError("pairwise feature matrix contains infinite values")
    if np.isnan(pairwise_matrix).any():
        raise ValueError("pairwise feature matrix contains NaN values")
    pairwise_columns = {
        feature_column: pairwise_matrix[:, column_index]
        for column_index, feature_column in enumerate(pairwise_stats.aggregate_feature_columns)
    }
    if hasattr(pairwise_stats, "coverage_feature_matrix"):
        coverage_matrix = np.asarray(pairwise_stats.coverage_feature_matrix(), dtype=np.float32)
        coverage_columns = promoted_pairwise_coverage_columns()
        if coverage_matrix.shape != (row_count, len(coverage_columns)):
            raise ValueError(
                "pairwise coverage feature matrix shape must match row_count and coverage columns: "
                f"{coverage_matrix.shape} != ({row_count}, {len(coverage_columns)})"
            )
        pairwise_columns.update(
            {
                feature_column: coverage_matrix[:, column_index]
                for column_index, feature_column in enumerate(coverage_columns)
            }
        )
    return pairwise_columns


def assemble_linker_feature_matrix(
    candidate_batch: LinkerCandidateBatch,
    row_features: Any,
    *,
    pairwise_stats: PairwiseAggregateStats,
    feature_columns: Sequence[str] = PROMOTED_LINKER_FEATURE_COLUMNS,
) -> LinkerFeatureMatrix:
    """Assemble an artifact-ordered linker/reranker matrix from compact candidate inputs."""

    resolved_columns = tuple(str(column) for column in feature_columns)
    pairwise_columns = _pairwise_feature_columns_to_matrix(pairwise_stats, candidate_batch.row_count)
    matrix = np.empty((candidate_batch.row_count, len(resolved_columns)), dtype=np.float32)
    for column_index, column in enumerate(resolved_columns):
        if column.startswith("pw_"):
            if column not in pairwise_columns:
                raise KeyError(f"Missing pairwise aggregate feature column: {column}")
            matrix[:, column_index] = pairwise_columns[column]
        else:
            matrix[:, column_index] = _coerce_row_feature_column(row_features, column, candidate_batch.row_count)
    return LinkerFeatureMatrix(
        matrix=matrix,
        feature_columns=resolved_columns,
        candidate_batch=candidate_batch,
        pairwise_stats=pairwise_stats,
    )


def assemble_linker_feature_matrix_rust(
    dataset: ANDData,
    candidate_batch: LinkerCandidateBatch,
    row_features: Any,
    *,
    feature_columns: Sequence[str] = PROMOTED_LINKER_FEATURE_COLUMNS,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = 0.0,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> LinkerFeatureMatrix:
    """Compute promoted pairwise aggregates in Rust and assemble the full feature matrix."""

    pairwise_stats = compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
    )
    return assemble_linker_feature_matrix(
        candidate_batch,
        row_features,
        pairwise_stats=pairwise_stats,
        feature_columns=feature_columns,
    )
