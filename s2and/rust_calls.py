"""Thin Rust operation wrappers around the per-dataset Rust featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.incremental_linking.array_validation import as_uint32_1d
from s2and.thread_config import resolve_n_jobs


def _six_array_result(result: Any, method_name: str) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Validate the exact native six-array result contract."""

    if not isinstance(result, tuple) or len(result) != 6:
        raise RuntimeError(f"RustFeaturizer.{method_name} violated its six-array result contract")
    return result


def get_constraints_matrix_indexed_rust(
    pairs: list[tuple[int, int]],
    *,
    featurizer: Any,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    suppress_orcid: bool = False,
) -> list[float | None]:
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    return list(
        featurizer.get_constraints_matrix_indexed(
            pairs,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            resolved_num_threads,
            suppress_orcid=suppress_orcid,
        )
    )


def get_constraint_labels_index_arrays_rust(
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    *,
    featurizer: Any,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    suppress_orcid: bool = False,
    large_integer: float = LARGE_INTEGER,
) -> np.ndarray:
    """Resolve constraint labels for numeric pair-index arrays in Rust.

    Returned values use the existing pairwise-label convention:
    ``NaN`` means unconstrained, otherwise ``constraint_distance - LARGE_INTEGER``.
    """

    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    return np.asarray(
        featurizer.linker_pair_index_arrays_constraint_labels(
            as_uint32_1d("left_signature_indices", left_signature_indices),
            as_uint32_1d("right_signature_indices", right_signature_indices),
            float(low_value),
            float(high_value),
            bool(dont_merge_cluster_seeds),
            bool(incremental_dont_use_cluster_seeds),
            resolved_num_threads,
            bool(suppress_orcid),
            float(large_integer),
        ),
        dtype=np.float64,
    )


def build_linker_pair_distance_accumulators_rust(
    row_indices: np.ndarray,
    row_count: int,
    pair_distances: np.ndarray,
    *,
    featurizer: Any,
    pair_labels: np.ndarray | None = None,
    num_threads: int | None = None,
    large_integer: float = LARGE_INTEGER,
    hard_disallow_distance: float = LARGE_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate candidate pair distances into row-level accumulators in Rust."""

    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    labels_arg = None if pair_labels is None else np.ascontiguousarray(pair_labels, dtype=np.float64)
    counts, sums, mins, top_distances, hard_disallow_pair_count = featurizer.linker_pair_distance_accumulators(
        as_uint32_1d("row_indices", row_indices),
        int(row_count),
        np.ascontiguousarray(pair_distances, dtype=np.float64),
        labels_arg,
        resolved_num_threads,
        float(large_integer),
        float(hard_disallow_distance),
    )
    return (
        np.asarray(counts, dtype=np.uint32),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(top_distances, dtype=np.float64),
        int(hard_disallow_pair_count),
    )


def get_constraints_block_upper_triangle_indexed_rust(
    block_signature_indices: list[int],
    *,
    featurizer: Any,
    start_offset: int = 0,
    max_pairs: int | None = None,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    suppress_orcid: bool = False,
) -> tuple[list[int], list[int], list[float | None]]:
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    left_indices, right_indices, values = featurizer.get_constraints_block_upper_triangle_indexed(
        block_signature_indices,
        start_offset,
        max_pairs,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
        resolved_num_threads,
        suppress_orcid=suppress_orcid,
    )
    # PyO3 converts the native Vec<u32> and Vec<Option<f64>> directly to
    # list[int] and list[float | None]; no per-pair Python coercion is needed.
    return left_indices, right_indices, values


def _get_constraints_block_upper_triangle_values_indexed_rust(
    block_signature_indices: list[int],
    *,
    featurizer: Any,
    start_offset: int = 0,
    max_pairs: int | None = None,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    suppress_orcid: bool = False,
) -> list[float | None]:
    """Get ordered bulk constraints without unused square-matrix coordinates."""
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    return featurizer._get_constraints_block_upper_triangle_values_indexed(
        block_signature_indices,
        start_offset,
        max_pairs,
        low_value,
        high_value,
        dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds,
        resolved_num_threads,
        suppress_orcid=suppress_orcid,
    )


def build_linker_pair_features_and_aggregate_stats_arrays_rust(
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    *,
    featurizer: Any,
    matrix_indices: list[int] | None = None,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    aggregate_nan_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pair features and row-level aggregate stats from numeric index arrays.

    ``nan_value`` controls the pair-feature matrix returned for model prediction.
    ``aggregate_nan_value`` can differ when callers need separate missing-value
    policies for the pairwise model matrix and promoted ``pw_*`` aggregates.
    """

    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    resolved_aggregate_nan_value = nan_value if aggregate_nan_value is None else float(aggregate_nan_value)
    result = featurizer.linker_pair_index_arrays_and_aggregate_stats(
        as_uint32_1d("left_signature_indices", left_signature_indices),
        as_uint32_1d("right_signature_indices", right_signature_indices),
        as_uint32_1d("row_indices", row_indices),
        int(row_count),
        matrix_indices,
        aggregate_indices,
        resolved_num_threads,
        nan_value,
        resolved_aggregate_nan_value,
    )
    matrix, counts, valid_counts, sums, mins, maxs = _six_array_result(
        result,
        "linker_pair_index_arrays_and_aggregate_stats",
    )
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(counts, dtype=np.uint32),
        np.asarray(valid_counts, dtype=np.uint64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_linker_pair_aggregate_stats_arrays_rust(
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    *,
    featurizer: Any,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    aggregate_nan_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build row-level aggregate stats from numeric pair index arrays without returning pair features."""

    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    resolved_aggregate_nan_value = nan_value if aggregate_nan_value is None else float(aggregate_nan_value)
    result = featurizer.linker_pair_index_arrays_and_aggregate_stats(
        as_uint32_1d("left_signature_indices", left_signature_indices),
        as_uint32_1d("right_signature_indices", right_signature_indices),
        as_uint32_1d("row_indices", row_indices),
        int(row_count),
        None,
        aggregate_indices,
        resolved_num_threads,
        nan_value,
        resolved_aggregate_nan_value,
        False,
    )
    _matrix, counts, valid_counts, sums, mins, maxs = _six_array_result(
        result,
        "linker_pair_index_arrays_and_aggregate_stats",
    )
    return (
        np.asarray(counts, dtype=np.uint32),
        np.asarray(valid_counts, dtype=np.uint64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_block_upper_triangle_feature_matrix_indexed_rust(
    block_signature_indices: list[int],
    *,
    featurizer: Any,
    start_offset: int = 0,
    max_pairs: int | None = None,
    selected_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
) -> np.ndarray:
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    matrix = featurizer.featurize_block_upper_triangle_matrix_indexed(
        block_signature_indices,
        start_offset,
        max_pairs,
        selected_indices,
        resolved_num_threads,
        nan_value,
    )
    return np.asarray(matrix, dtype=np.float64)
