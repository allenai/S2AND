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
)

PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS: tuple[str, ...] = (
    "min_distance",
    "affiliation_contradiction_severity",
    "specter_exemplar_similarity",
    "coauthor_overlap",
    "affiliation_overlap",
    "year_compatibility",
    "retrieval_rank",
    "retrieval_reciprocal_rank",
    "cluster_size_log",
    "candidate_year_span",
    "year_gap_to_candidate_range",
    "year_gap_signed_to_candidate_range",
    "candidate_dominant_first_name_length",
    "query_first_prefix_match_any_length",
    "candidate_cluster_max_paper_author_count",
    "paper_author_list_max_jaccard",
    "paper_author_list_max_containment",
    "paper_author_list_max_overlap_count",
    "local_author_window10_jaccard_max",
    "local_author_window10_overlap_count_max",
    "best_author_count_log_absdiff",
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
    "same_dominant_first_as_best_top5",
    "same_family_as_heuristic_choice",
    "top5_mean_distance",
    "last_name_count_min_rarity",
    "last_first_name_count_min_rarity",
)
PROMOTED_LINKER_FEATURE_COLUMNS: tuple[str, ...] = (
    "min_distance",
    "pw_max_affiliation_overlap",
    "strong_positive_anchor_score",
    "pw_max_middle_initials_overlap",
    "top5_mean_distance",
    "retrieval_reciprocal_rank",
    "anchor_evidence_count",
    "retrieval_rank",
    "specter_exemplar_similarity",
    "paper_author_list_max_jaccard",
    "best_author_count_log_absdiff",
    "affiliation_overlap",
    "query_first_prefix_match_any_length",
    "pw_mean_email_prefix_equal",
    "year_gap_signed_to_candidate_range",
    "pw_mean_first_names_equal",
    "pw_min_middle_initials_overlap",
    "pw_max_title_overlap_words",
    "affiliation_contradiction_severity",
    "paper_author_list_max_containment",
    "pw_max_journal_overlap",
    "candidate_cluster_max_paper_author_count",
    "pw_mean_middle_names_equal",
    "pw_min_last_first_name_count_max",
    "pw_mean_coauthor_match",
    "pw_mean_coauthor_overlap",
    "weak_residual_anchor_score",
    "candidate_year_span",
    "pw_mean_title_overlap_words",
    "pw_max_venue_overlap",
    "pw_mean_journal_overlap",
    "same_dominant_first_as_best_top5",
    "year_compatibility",
    "last_first_name_count_min_rarity",
    "pw_min_specter_cosine_sim",
    "last_name_count_min_rarity",
    "cluster_size_log",
    "pw_min_first_name_count_max",
    "pw_max_coauthor_overlap",
    "candidate_dominant_first_name_length",
    "paper_author_list_max_overlap_count",
    "local_author_window10_jaccard_max",
    "local_author_window10_overlap_count_max",
    "pw_max_jaro",
    "same_family_as_heuristic_choice",
    "sparse_relative_winner_score",
    "pw_min_first_name_count_min",
    "pw_min_levenshtein",
    "coauthor_overlap",
    "pw_mean_english_count",
    "year_gap_to_candidate_range",
    "pw_mean_middle_one_missing",
    "pw_mean_specter_cosine_sim",
)
_AVAILABLE_PROMOTED_LINKER_FEATURE_COLUMNS = frozenset(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS) | frozenset(
    promoted_pairwise_aggregate_columns()
)
_UNKNOWN_PROMOTED_LINKER_FEATURE_COLUMNS = frozenset(PROMOTED_LINKER_FEATURE_COLUMNS) - (
    _AVAILABLE_PROMOTED_LINKER_FEATURE_COLUMNS
)
if _UNKNOWN_PROMOTED_LINKER_FEATURE_COLUMNS:
    raise RuntimeError(
        "Promoted linker feature schema contains columns without a production source: "
        f"{sorted(_UNKNOWN_PROMOTED_LINKER_FEATURE_COLUMNS)}"
    )
if len(PROMOTED_LINKER_FEATURE_COLUMNS) != 53:
    raise RuntimeError(
        f"Promoted linker feature schema must have 53 columns, got {len(PROMOTED_LINKER_FEATURE_COLUMNS)}"
    )


@dataclass(frozen=True)
class LinkerFeatureMatrix:
    """Artifact-ordered feature matrix plus production/training metadata."""

    matrix: np.ndarray
    feature_columns: tuple[str, ...]
    candidate_batch: LinkerCandidateBatch
    pairwise_stats: PairwiseAggregateStats | None = None


def promoted_linker_feature_columns() -> tuple[str, ...]:
    """Return the promoted 53-feature linker/reranker schema in artifact order."""

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
