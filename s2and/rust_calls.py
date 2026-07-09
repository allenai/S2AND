"""Thin Rust operation wrappers around the per-dataset Rust featurizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.incremental_linking.array_validation import as_uint32_1d
from s2and.thread_config import resolve_n_jobs


def _get_rust_featurizer(*args: Any, **kwargs: Any) -> Any:
    from s2and import feature_port

    return feature_port._get_rust_featurizer(*args, **kwargs)  # noqa: SLF001


def _get_rust_featurizer_for_cluster_seed_update(
    dataset: ANDData,
    runtime_context: Any | None,
) -> Any:
    from s2and import feature_port

    return feature_port._get_cached_rust_featurizer_for_cluster_seed_update(  # noqa: SLF001
        dataset,
        runtime_context=runtime_context,
    )


def _promote_rust_featurizer_cluster_seed_version(
    dataset: ANDData,
    featurizer: Any,
    *,
    target_seed_version: int,
) -> None:
    from s2and import feature_port

    feature_port._promote_cached_rust_featurizer_cluster_seed_version(  # noqa: SLF001
        dataset,
        featurizer,
        target_seed_version=target_seed_version,
    )


def _rust_cluster_seed_update_lock(dataset: ANDData) -> Any:
    from s2and import feature_port

    return feature_port._rust_cluster_seed_update_lock(dataset)  # noqa: SLF001


def _resolve_featurizer(
    dataset: ANDData | None,
    featurizer: Any | None,
    runtime_context: Any | None,
) -> Any:
    if featurizer is not None:
        return featurizer
    if dataset is None:
        raise ValueError("dataset is required when featurizer is not provided")
    return _get_rust_featurizer(dataset, runtime_context=runtime_context)


def update_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: Any | None = None,
    *,
    bump_version: bool = False,
) -> None:
    """Sync current Python cluster seeds into the cached Rust featurizer.

    ``bump_version`` is opt-in because the model seed containers bump the
    dataset version when seed material changes.
    """

    with _rust_cluster_seed_update_lock(dataset):
        featurizer = _get_rust_featurizer_for_cluster_seed_update(dataset, runtime_context=runtime_context)
        current_seed_version = int(getattr(dataset, "_cluster_seeds_version", 0))
        target_seed_version = current_seed_version + 1 if bump_version else current_seed_version
        previous_require = None
        previous_disallow = None
        cluster_seeds_require = getattr(featurizer, "cluster_seeds_require", None)
        cluster_seeds_disallow = getattr(featurizer, "cluster_seeds_disallow", None)
        if callable(cluster_seeds_require) and callable(cluster_seeds_disallow):
            previous_require = dict(cluster_seeds_require())
            previous_disallow = set(tuple(pair) for pair in cluster_seeds_disallow())
        featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
        try:
            _promote_rust_featurizer_cluster_seed_version(
                dataset,
                featurizer,
                target_seed_version=target_seed_version,
            )
        except Exception:
            if previous_require is not None and previous_disallow is not None:
                featurizer.update_cluster_seeds(previous_require, previous_disallow)
            raise
        if bump_version:
            dataset._cluster_seeds_version = target_seed_version


def get_constraints_matrix_indexed_rust(
    dataset: ANDData,
    pairs: list[tuple[int, int]],
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
) -> list[float | None]:
    if featurizer is None:
        featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context)
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
    dataset: ANDData | None,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
    large_integer: float = LARGE_INTEGER,
) -> np.ndarray:
    """Resolve constraint labels for numeric pair-index arrays in Rust.

    Returned values use the existing pairwise-label convention:
    ``NaN`` means unconstrained, otherwise ``constraint_distance - LARGE_INTEGER``.
    """

    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    method = getattr(featurizer, "linker_pair_index_arrays_constraint_labels", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_constraint_labels is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    return np.asarray(
        method(
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
    dataset: ANDData | None,
    row_indices: np.ndarray,
    row_count: int,
    pair_distances: np.ndarray,
    pair_labels: np.ndarray | None = None,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    large_integer: float = LARGE_INTEGER,
    hard_disallow_distance: float = LARGE_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate candidate pair distances into row-level accumulators in Rust."""

    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)

    method = getattr(featurizer, "linker_pair_distance_accumulators", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_distance_accumulators is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    labels_arg = None if pair_labels is None else np.ascontiguousarray(pair_labels, dtype=np.float64)
    counts, sums, mins, top_distances, hard_disallow_pair_count = method(
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
    dataset: ANDData | None,
    block_signature_indices: list[int],
    start_offset: int = 0,
    max_pairs: int | None = None,
    low_value: float = 0.0,
    high_value: float = LARGE_DISTANCE,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    num_threads: int | None = None,
    featurizer: Any | None = None,
    runtime_context: Any | None = None,
    suppress_orcid: bool = False,
) -> tuple[list[int], list[int], list[float | None]]:
    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)

    method = getattr(featurizer, "get_constraints_block_upper_triangle_indexed", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.get_constraints_block_upper_triangle_indexed is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)

    left_indices, right_indices, values = method(
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
    return (
        [int(value) for value in left_indices],
        [int(value) for value in right_indices],
        list(values),
    )


def build_linker_pair_features_and_aggregate_stats_arrays_rust(
    dataset: ANDData | None,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    matrix_indices: list[int] | None = None,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    aggregate_nan_value: float | None = None,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build pair features and row-level aggregate stats from numeric index arrays.

    ``nan_value`` controls the pair-feature matrix returned for model prediction.
    ``aggregate_nan_value`` can differ when callers need separate missing-value
    policies for the pairwise model matrix and promoted ``pw_*`` aggregates.
    """

    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)
    method = getattr(featurizer, "linker_pair_index_arrays_and_aggregate_stats", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    resolved_aggregate_nan_value = nan_value if aggregate_nan_value is None else float(aggregate_nan_value)
    result = method(
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
    try:
        result_len = len(result)
    except TypeError as exc:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        ) from exc
    if result_len != 6:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        )
    matrix, counts, valid_counts, sums, mins, maxs = result
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(counts, dtype=np.uint32),
        np.asarray(valid_counts, dtype=np.uint64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_linker_pair_aggregate_stats_arrays_rust(
    dataset: ANDData | None,
    left_signature_indices: np.ndarray,
    right_signature_indices: np.ndarray,
    row_indices: np.ndarray,
    row_count: int,
    aggregate_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    aggregate_nan_value: float | None = None,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build row-level aggregate stats from numeric pair index arrays without returning pair features."""

    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)
    method = getattr(featurizer, "linker_pair_index_arrays_and_aggregate_stats", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    resolved_aggregate_nan_value = nan_value if aggregate_nan_value is None else float(aggregate_nan_value)
    result = method(
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
    try:
        result_len = len(result)
    except TypeError as exc:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        ) from exc
    if result_len != 6:
        raise RuntimeError(
            "RustFeaturizer.linker_pair_index_arrays_and_aggregate_stats returned an outdated "
            "aggregate contract; rebuild/install a newer s2and-rust extension."
        )
    _matrix, counts, valid_counts, sums, mins, maxs = result
    return (
        np.asarray(counts, dtype=np.uint32),
        np.asarray(valid_counts, dtype=np.uint64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(mins, dtype=np.float64),
        np.asarray(maxs, dtype=np.float64),
    )


def build_block_upper_triangle_feature_matrix_indexed_rust(
    dataset: ANDData | None,
    block_signature_indices: list[int],
    start_offset: int = 0,
    max_pairs: int | None = None,
    selected_indices: list[int] | None = None,
    num_threads: int | None = None,
    nan_value: float = np.nan,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> np.ndarray:
    featurizer = _resolve_featurizer(dataset, featurizer, runtime_context)
    method = getattr(featurizer, "featurize_block_upper_triangle_matrix_indexed", None)
    if not callable(method):
        raise RuntimeError(
            "RustFeaturizer.featurize_block_upper_triangle_matrix_indexed is unavailable; "
            "rebuild/install a newer s2and-rust extension."
        )
    resolved_num_threads = None if num_threads is None else resolve_n_jobs(num_threads)
    matrix = method(
        block_signature_indices,
        start_offset,
        max_pairs,
        selected_indices,
        resolved_num_threads,
        nan_value,
    )
    return np.asarray(matrix, dtype=np.float64)
