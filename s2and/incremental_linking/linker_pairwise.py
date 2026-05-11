"""Rust-backed pairwise aggregate features for linker candidate rows."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and import feature_port, memory_budget
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo

PROD_PAIRWISE_FEATURE_GROUPS: tuple[str, ...] = (
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
)
PROMOTED_DROPPED_PAIRWISE_BASE_FEATURES: frozenset[str] = frozenset({"year_diff", "position_diff"})
PAIRWISE_INFO = FeaturizationInfo(features_to_use=list(PROD_PAIRWISE_FEATURE_GROUPS))
PROD_PAIRWISE_FEATURE_NAMES: tuple[str, ...] = tuple(PAIRWISE_INFO.get_feature_names())
PROD_PAIRWISE_FEATURE_INDICES: tuple[int, ...] = tuple(
    feature_index
    for feature_group in PROD_PAIRWISE_FEATURE_GROUPS
    for feature_index in PAIRWISE_INFO.feature_group_to_index[feature_group]
)
PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature_name
    for feature_name in PROD_PAIRWISE_FEATURE_NAMES
    if feature_name not in PROMOTED_DROPPED_PAIRWISE_BASE_FEATURES
)
PROMOTED_PAIRWISE_AGG_FEATURE_INDICES: tuple[int, ...] = tuple(
    PROD_PAIRWISE_FEATURE_INDICES[PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
    for feature_name in PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES
)
PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"pw_{stat}_{feature_name}"
    for stat in ("min", "mean", "max")
    for feature_name in PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES
)
PROMOTED_PAIRWISE_COVERAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "middle_name": ("middle_initials_overlap", "middle_names_equal"),
    "email": ("email_prefix_equal", "email_suffix_equal"),
    "title": ("title_overlap_words", "title_overlap_chars"),
    "coauthor_overlap": ("coauthor_overlap", "coauthor_match"),
    "coauthor_similarity": ("coauthor_similarity",),
    "affiliation": ("affiliation_overlap",),
    "specter": ("specter_cosine_sim",),
    "venue": ("venue_overlap",),
    "journal": ("journal_overlap",),
}
PROMOTED_PAIRWISE_COVERAGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "pw_pair_count_log_capped",
    *(f"pw_valid_fraction_{group_name}" for group_name in PROMOTED_PAIRWISE_COVERAGE_GROUPS),
)
PAIRWISE_COUNT_LOG_CAPPED_REFERENCE_SIZE = 192.0


@dataclass(frozen=True)
class LinkerPairFeatureChunk:
    """One memory-bounded Rust pair-feature chunk plus aggregate stats."""

    start: int
    stop: int
    pair_features: np.ndarray
    matrix_indices: tuple[int, ...]
    aggregate_indices: tuple[int, ...]
    global_row_indices: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    valid_counts: np.ndarray | None = None


@dataclass(frozen=True)
class PairwiseAggregateStats:
    """Accumulated pairwise aggregate features for linker candidate rows."""

    counts: np.ndarray
    sums: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    base_feature_names: tuple[str, ...]
    aggregate_feature_columns: tuple[str, ...]
    chunk_plan: memory_budget.RustBatchChunkPlan
    chunk_count: int
    matrix_indices: tuple[int, ...]
    aggregate_indices: tuple[int, ...]
    valid_counts: np.ndarray | None = None

    def mean_matrix(self) -> np.ndarray:
        """Return row-wise aggregate means, ignoring missing per-pair values."""

        means = np.full_like(self.sums, np.nan, dtype=np.float64)
        if self.valid_counts is None:
            observed = self.counts > 0
            if np.any(observed):
                means[observed] = self.sums[observed] / self.counts[observed, None]
        else:
            observed = self.valid_counts > 0
            if np.any(observed):
                means[observed] = self.sums[observed] / self.valid_counts[observed]
        means[~np.isfinite(means)] = np.nan
        return means

    def feature_matrix(self) -> np.ndarray:
        """Return columns in promoted `pw_min`, `pw_mean`, `pw_max` order."""

        mins = self.mins.copy()
        maxs = self.maxs.copy()
        observed = (
            np.broadcast_to(self.counts[:, None] > 0, mins.shape)
            if self.valid_counts is None
            else self.valid_counts > 0
        )
        mins[~observed] = np.nan
        maxs[~observed] = np.nan
        mins[~np.isfinite(mins)] = np.nan
        maxs[~np.isfinite(maxs)] = np.nan
        return np.concatenate((mins, self.mean_matrix(), maxs), axis=1)

    def coverage_feature_matrix(self) -> np.ndarray:
        """Return compact denominator/validity coverage features for pairwise aggregates."""

        row_count = len(self.counts)
        coverage = np.full((row_count, len(PROMOTED_PAIRWISE_COVERAGE_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        counts = np.asarray(self.counts, dtype=np.float64)
        coverage[:, 0] = np.log1p(np.minimum(counts, PAIRWISE_COUNT_LOG_CAPPED_REFERENCE_SIZE))
        if self.valid_counts is None:
            return coverage

        position_by_name = {feature_name: index for index, feature_name in enumerate(self.base_feature_names)}
        observed = counts > 0.0
        for output_index, feature_names in enumerate(PROMOTED_PAIRWISE_COVERAGE_GROUPS.values(), start=1):
            if not all(feature_name in position_by_name for feature_name in feature_names):
                continue
            positions = [position_by_name[feature_name] for feature_name in feature_names]
            valid_count = np.asarray(self.valid_counts[:, positions], dtype=np.float64).min(axis=1)
            coverage[observed, output_index] = valid_count[observed] / counts[observed]
        coverage[~np.isfinite(coverage)] = np.nan
        return coverage


@dataclass(frozen=True)
class LinkerCandidateBatch:
    """Retrieval-optional candidate batch for training and production featureization.

    The hot-path contract is numeric arrays: one row per candidate component and
    one flat pair plan from query signature index to candidate member signature
    index. Production retrieval can construct these arrays directly; training
    scripts can load persisted candidates and labels into the same shape.
    """

    row_count: int
    left_signature_indices: np.ndarray
    right_signature_indices: np.ndarray
    pair_row_indices: np.ndarray
    row_query_signature_indices: np.ndarray | None = None
    row_component_keys: tuple[Any, ...] | None = None
    labels: np.ndarray | None = None
    retrieval_scores: np.ndarray | None = None
    retrieval_ranks: np.ndarray | None = None

    def __post_init__(self) -> None:
        row_count = int(self.row_count)
        if row_count < 0:
            raise ValueError(f"row_count must be non-negative, got {row_count}")
        left = _as_uint32_1d("left_signature_indices", self.left_signature_indices)
        right = _as_uint32_1d("right_signature_indices", self.right_signature_indices)
        rows = _as_uint32_1d("pair_row_indices", self.pair_row_indices)
        if not (len(left) == len(right) == len(rows)):
            raise ValueError(
                "left_signature_indices, right_signature_indices, and pair_row_indices must have equal length: "
                f"left={len(left)} right={len(right)} rows={len(rows)}"
            )
        if len(rows) and int(rows.max()) >= row_count:
            raise IndexError(f"pair_row_indices contains row >= row_count={row_count}")
        if self.row_query_signature_indices is not None and len(self.row_query_signature_indices) != row_count:
            raise ValueError(
                "row_query_signature_indices must have length row_count: "
                f"{len(self.row_query_signature_indices)} != {row_count}"
            )
        if self.row_component_keys is not None and len(self.row_component_keys) != row_count:
            raise ValueError(
                f"row_component_keys must have length row_count: {len(self.row_component_keys)} != {row_count}"
            )
        if self.labels is not None and len(self.labels) != row_count:
            raise ValueError(f"labels must have length row_count: {len(self.labels)} != {row_count}")
        if self.retrieval_scores is not None and len(self.retrieval_scores) != row_count:
            raise ValueError(
                f"retrieval_scores must have length row_count: {len(self.retrieval_scores)} != {row_count}"
            )
        if self.retrieval_ranks is not None and len(self.retrieval_ranks) != row_count:
            raise ValueError(f"retrieval_ranks must have length row_count: {len(self.retrieval_ranks)} != {row_count}")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "left_signature_indices", left)
        object.__setattr__(self, "right_signature_indices", right)
        object.__setattr__(self, "pair_row_indices", rows)
        if self.row_query_signature_indices is not None:
            object.__setattr__(
                self,
                "row_query_signature_indices",
                _as_uint32_1d("row_query_signature_indices", self.row_query_signature_indices),
            )

    @property
    def pair_count(self) -> int:
        """Return the number of query-to-member pairs in this batch."""

        return int(len(self.left_signature_indices))


def promoted_pairwise_aggregate_columns() -> tuple[str, ...]:
    """Return the promoted pairwise aggregate feature columns in model order."""

    return PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS


def promoted_pairwise_coverage_columns() -> tuple[str, ...]:
    """Return compact pairwise coverage feature columns in model order."""

    return PROMOTED_PAIRWISE_COVERAGE_FEATURE_COLUMNS


def _as_uint32_1d(name: str, values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.uint32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape={array.shape}")
    return array


def build_candidate_batch_from_members(
    query_signature_indices: Sequence[int] | np.ndarray,
    candidate_member_signature_indices: Sequence[Sequence[int] | np.ndarray],
    *,
    row_component_keys: Sequence[Any] | None = None,
    labels: Sequence[Any] | np.ndarray | None = None,
    retrieval_scores: Sequence[float] | np.ndarray | None = None,
    retrieval_ranks: Sequence[int] | np.ndarray | None = None,
) -> LinkerCandidateBatch:
    """Build a `LinkerCandidateBatch` from per-candidate member signature indices."""

    row_query_indices = _as_uint32_1d("query_signature_indices", query_signature_indices)
    row_count = int(len(row_query_indices))
    if len(candidate_member_signature_indices) != row_count:
        raise ValueError(
            "candidate_member_signature_indices must have one member-list per row: "
            f"{len(candidate_member_signature_indices)} != {row_count}"
        )
    member_arrays = [
        _as_uint32_1d(f"candidate_member_signature_indices[{row_index}]", members)
        for row_index, members in enumerate(candidate_member_signature_indices)
    ]
    pair_count = sum(len(members) for members in member_arrays)
    left = np.empty(pair_count, dtype=np.uint32)
    right = np.empty(pair_count, dtype=np.uint32)
    rows = np.empty(pair_count, dtype=np.uint32)
    offset = 0
    for row_index, members in enumerate(member_arrays):
        stop = offset + len(members)
        left[offset:stop] = row_query_indices[row_index]
        right[offset:stop] = members
        rows[offset:stop] = row_index
        offset = stop
    return LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left,
        right_signature_indices=right,
        pair_row_indices=rows,
        row_query_signature_indices=row_query_indices,
        row_component_keys=None if row_component_keys is None else tuple(row_component_keys),
        labels=None if labels is None else np.asarray(labels),
        retrieval_scores=None if retrieval_scores is None else np.asarray(retrieval_scores, dtype=np.float32),
        retrieval_ranks=None if retrieval_ranks is None else np.asarray(retrieval_ranks, dtype=np.uint16),
    )


def _signature_id_to_index_or_raise(signature_id_to_index: dict[Any, int], signature_id: Any) -> int:
    if signature_id in signature_id_to_index:
        return int(signature_id_to_index[signature_id])
    string_id = str(signature_id)
    if string_id in signature_id_to_index:
        return int(signature_id_to_index[string_id])
    raise KeyError(f"signature_id not present in Rust featurizer signature_ids: {signature_id!r}")


def _rust_signature_id_to_index(featurizer: Any) -> dict[Any, int]:
    signature_ids = featurizer.signature_ids()
    signature_id_to_index: dict[Any, int] = {}
    for index, signature_id in enumerate(signature_ids):
        signature_id_to_index[signature_id] = int(index)
        signature_id_to_index[str(signature_id)] = int(index)
    return signature_id_to_index


def _ordered_union(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys([int(value) for value in (*left, *right)]))


def _localize_row_indices(row_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return global rows plus chunk-local row indices for a pair-owner row slice."""

    if len(row_chunk) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint32)
    if len(row_chunk) == 1 or bool(np.all(row_chunk[1:] >= row_chunk[:-1])):
        first_row = int(row_chunk[0])
        last_row = int(row_chunk[-1])
        global_rows = np.arange(first_row, last_row + 1, dtype=np.int64)
        local_row_indices = np.ascontiguousarray(row_chunk - np.uint32(first_row), dtype=np.uint32)
        return global_rows, local_row_indices
    global_rows, local_row_indices = np.unique(row_chunk, return_inverse=True)
    return np.ascontiguousarray(global_rows, dtype=np.int64), np.ascontiguousarray(local_row_indices, dtype=np.uint32)


def _matrix_positions(matrix_indices: Sequence[int], selected_indices: Sequence[int]) -> tuple[int, ...]:
    position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
    return tuple(position_by_index[int(index)] for index in selected_indices)


def aggregate_pair_feature_chunk_nan_aware(
    *,
    pair_features: np.ndarray,
    local_row_indices: np.ndarray,
    row_count: int,
    matrix_indices: Sequence[int],
    aggregate_indices: Sequence[int],
    nan_value: float = math.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate pair features with `nan*` semantics per row and base feature.

    If ``nan_value`` is NaN, missing per-pair values are skipped and each
    feature is NaN only when all contributing pair values are missing. Otherwise
    NaNs are first replaced with ``nan_value`` and counted as ordinary values.
    """

    aggregate_positions = _matrix_positions(matrix_indices, aggregate_indices)
    aggregate_values = np.asarray(pair_features[:, aggregate_positions], dtype=np.float64)
    row_count = int(row_count)
    aggregate_count = len(aggregate_positions)
    counts = np.zeros(row_count, dtype=np.uint32)
    valid_counts = np.zeros((row_count, aggregate_count), dtype=np.uint64)
    sums = np.zeros((row_count, aggregate_count), dtype=np.float64)
    mins = np.full((row_count, aggregate_count), np.inf, dtype=np.float64)
    maxs = np.full((row_count, aggregate_count), -np.inf, dtype=np.float64)
    if len(aggregate_values) == 0 or aggregate_count == 0:
        return counts, valid_counts, sums, mins, maxs

    local_rows = np.asarray(local_row_indices, dtype=np.uint32)
    if local_rows.shape != (len(aggregate_values),):
        raise ValueError(f"local_row_indices must have shape ({len(aggregate_values)},), got {local_rows.shape}")
    if len(local_rows) and int(local_rows.max()) >= row_count:
        raise IndexError(f"local_row_indices contains row >= row_count={row_count}")

    counts += np.bincount(local_rows, minlength=row_count).astype(np.uint32, copy=False)
    replace_missing = not math.isnan(float(nan_value))
    if replace_missing:
        aggregate_values = np.where(np.isnan(aggregate_values), float(nan_value), aggregate_values)
        valid = np.ones(aggregate_values.shape, dtype=bool)
    else:
        valid = ~np.isnan(aggregate_values)
    for column_index in range(aggregate_count):
        column_valid = valid[:, column_index]
        if not np.any(column_valid):
            continue
        rows = local_rows[column_valid]
        values = aggregate_values[column_valid, column_index]
        valid_counts[:, column_index] += np.bincount(rows, minlength=row_count).astype(np.uint64, copy=False)
        np.add.at(sums[:, column_index], rows, values)
        np.minimum.at(mins[:, column_index], rows, values)
        np.maximum.at(maxs[:, column_index], rows, values)
    return counts, valid_counts, sums, mins, maxs


def compute_linker_pair_chunk_plan(
    *,
    total_pairs: int,
    row_count: int,
    matrix_feature_count: int,
    aggregate_feature_count: int,
    total_ram_bytes: int | None = None,
) -> memory_budget.RustBatchChunkPlan:
    """Return the shared Rust batch plan used for linker pair chunks."""

    memory_feature_count = max(1, int(matrix_feature_count) + int(aggregate_feature_count) * 3 + 1)
    return memory_budget.compute_rust_batch_chunk_plan(
        num_features=memory_feature_count,
        total_pairs=int(total_pairs),
        total_rows=int(row_count),
        total_ram_bytes=total_ram_bytes,
    )


def iter_linker_pair_feature_chunks_rust(
    dataset: ANDData,
    pairs: Sequence[tuple[Any, Any]],
    row_indices: Sequence[int],
    *,
    row_count: int,
    matrix_indices: Sequence[int] | None = None,
    aggregate_indices: Sequence[int] = PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
    chunk_plan: memory_budget.RustBatchChunkPlan | None = None,
    pairs_are_indices: bool = False,
) -> Iterator[LinkerPairFeatureChunk]:
    """Yield pair-feature chunks while aggregating selected pairwise features in Rust."""

    if len(pairs) != len(row_indices):
        raise ValueError(f"pairs and row_indices must have equal length: {len(pairs)} != {len(row_indices)}")
    if int(row_count) < 0:
        raise ValueError(f"row_count must be non-negative, got {row_count}")
    if not pairs:
        return

    resolved_aggregate_indices = tuple(int(index) for index in aggregate_indices)
    if matrix_indices is None:
        requested_matrix_indices = tuple(PROD_PAIRWISE_FEATURE_INDICES)
    else:
        requested_matrix_indices = tuple(int(index) for index in matrix_indices)
    resolved_matrix_indices = _ordered_union(requested_matrix_indices, resolved_aggregate_indices)
    plan = chunk_plan
    if plan is None:
        plan = compute_linker_pair_chunk_plan(
            total_pairs=len(pairs),
            row_count=int(row_count),
            matrix_feature_count=len(resolved_matrix_indices),
            aggregate_feature_count=len(resolved_aggregate_indices),
            total_ram_bytes=total_ram_bytes,
        )
    chunk_pairs = int(plan["chunk_pairs"])
    if featurizer is None:
        featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
            dataset,
            runtime_context=runtime_context,
            use_cache=use_cache,
        )
    signature_id_to_index = None
    if not pairs_are_indices:
        signature_id_to_index = _rust_signature_id_to_index(featurizer)

    for start in range(0, len(pairs), chunk_pairs):
        stop = min(len(pairs), start + chunk_pairs)
        pair_chunk = pairs[start:stop]
        row_chunk = [int(value) for value in row_indices[start:stop]]
        invalid_rows = [value for value in row_chunk if value < 0 or value >= int(row_count)]
        if invalid_rows:
            raise IndexError(f"row_indices contains out-of-range values for row_count={row_count}: {invalid_rows[:5]}")
        global_rows = np.asarray(tuple(dict.fromkeys(row_chunk)), dtype=np.int64)
        local_row_by_global = {int(global_row): int(local_index) for local_index, global_row in enumerate(global_rows)}
        local_row_indices = [local_row_by_global[int(global_row)] for global_row in row_chunk]
        if pairs_are_indices:
            indexed_pairs = [(int(left), int(right)) for left, right in pair_chunk]
        else:
            if signature_id_to_index is None:
                raise RuntimeError("signature_id_to_index was not initialized")
            indexed_pairs = [
                (
                    _signature_id_to_index_or_raise(signature_id_to_index, left),
                    _signature_id_to_index_or_raise(signature_id_to_index, right),
                )
                for left, right in pair_chunk
            ]
        pair_features, counts, sums, mins, maxs = (
            feature_port.build_linker_pair_features_and_aggregate_stats_indexed_rust(
                dataset,
                indexed_pairs,
                local_row_indices,
                len(global_rows),
                matrix_indices=list(resolved_matrix_indices),
                aggregate_indices=list(resolved_aggregate_indices),
                num_threads=max(1, int(n_jobs)),
                nan_value=float(nan_value),
                runtime_context=runtime_context,
                use_cache=use_cache,
                featurizer=featurizer,
            )
        )
        counts, valid_counts, sums, mins, maxs = aggregate_pair_feature_chunk_nan_aware(
            pair_features=pair_features,
            local_row_indices=np.asarray(local_row_indices, dtype=np.uint32),
            row_count=len(global_rows),
            matrix_indices=resolved_matrix_indices,
            aggregate_indices=resolved_aggregate_indices,
            nan_value=float(nan_value),
        )
        yield LinkerPairFeatureChunk(
            start=int(start),
            stop=int(stop),
            pair_features=pair_features,
            matrix_indices=resolved_matrix_indices,
            aggregate_indices=resolved_aggregate_indices,
            global_row_indices=global_rows,
            counts=counts,
            sums=sums,
            mins=mins,
            maxs=maxs,
            valid_counts=valid_counts,
        )


def iter_candidate_batch_pair_feature_chunks_rust(
    dataset: ANDData,
    candidate_batch: LinkerCandidateBatch,
    *,
    matrix_indices: Sequence[int] | None = None,
    aggregate_indices: Sequence[int] = PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
    chunk_plan: memory_budget.RustBatchChunkPlan | None = None,
) -> Iterator[LinkerPairFeatureChunk]:
    """Yield feature chunks from a numeric candidate batch without tuple materialization."""

    if candidate_batch.pair_count == 0:
        return
    resolved_aggregate_indices = tuple(int(index) for index in aggregate_indices)
    if matrix_indices is None:
        requested_matrix_indices = tuple(PROD_PAIRWISE_FEATURE_INDICES)
    else:
        requested_matrix_indices = tuple(int(index) for index in matrix_indices)
    resolved_matrix_indices = _ordered_union(requested_matrix_indices, resolved_aggregate_indices)
    plan = chunk_plan
    if plan is None:
        plan = compute_linker_pair_chunk_plan(
            total_pairs=candidate_batch.pair_count,
            row_count=candidate_batch.row_count,
            matrix_feature_count=len(resolved_matrix_indices),
            aggregate_feature_count=len(resolved_aggregate_indices),
            total_ram_bytes=total_ram_bytes,
        )
    chunk_pairs = int(plan["chunk_pairs"])
    if featurizer is None:
        featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
            dataset,
            runtime_context=runtime_context,
            use_cache=use_cache,
        )

    for start in range(0, candidate_batch.pair_count, chunk_pairs):
        stop = min(candidate_batch.pair_count, start + chunk_pairs)
        row_chunk = candidate_batch.pair_row_indices[start:stop]
        global_rows, local_row_indices = _localize_row_indices(row_chunk)
        pair_features, counts, sums, mins, maxs = (
            feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
                dataset,
                candidate_batch.left_signature_indices[start:stop],
                candidate_batch.right_signature_indices[start:stop],
                local_row_indices,
                len(global_rows),
                matrix_indices=list(resolved_matrix_indices),
                aggregate_indices=list(resolved_aggregate_indices),
                num_threads=max(1, int(n_jobs)),
                nan_value=float(nan_value),
                runtime_context=runtime_context,
                use_cache=use_cache,
                featurizer=featurizer,
            )
        )
        counts, valid_counts, sums, mins, maxs = aggregate_pair_feature_chunk_nan_aware(
            pair_features=pair_features,
            local_row_indices=local_row_indices,
            row_count=len(global_rows),
            matrix_indices=resolved_matrix_indices,
            aggregate_indices=resolved_aggregate_indices,
            nan_value=float(nan_value),
        )
        yield LinkerPairFeatureChunk(
            start=int(start),
            stop=int(stop),
            pair_features=pair_features,
            matrix_indices=resolved_matrix_indices,
            aggregate_indices=resolved_aggregate_indices,
            global_row_indices=global_rows,
            counts=counts,
            sums=sums,
            mins=mins,
            maxs=maxs,
            valid_counts=valid_counts,
        )


def compute_candidate_batch_pairwise_aggregate_stats_rust(
    dataset: ANDData,
    candidate_batch: LinkerCandidateBatch,
    *,
    aggregate_feature_names: Sequence[str] = PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> PairwiseAggregateStats:
    """Compute promoted pairwise aggregates for training or production candidate batches."""

    aggregate_feature_names = tuple(str(feature_name) for feature_name in aggregate_feature_names)
    aggregate_indices = tuple(
        PROD_PAIRWISE_FEATURE_INDICES[PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in aggregate_feature_names
    )
    aggregate_columns = tuple(
        f"pw_{stat}_{feature_name}" for stat in ("min", "mean", "max") for feature_name in aggregate_feature_names
    )
    plan = compute_linker_pair_chunk_plan(
        total_pairs=candidate_batch.pair_count,
        row_count=candidate_batch.row_count,
        matrix_feature_count=len(aggregate_indices),
        aggregate_feature_count=len(aggregate_indices),
        total_ram_bytes=total_ram_bytes,
    )
    counts = np.zeros(candidate_batch.row_count, dtype=np.uint64)
    valid_counts = np.zeros((candidate_batch.row_count, len(aggregate_indices)), dtype=np.uint64)
    sums = np.zeros((candidate_batch.row_count, len(aggregate_indices)), dtype=np.float64)
    mins = np.full((candidate_batch.row_count, len(aggregate_indices)), np.inf, dtype=np.float64)
    maxs = np.full((candidate_batch.row_count, len(aggregate_indices)), -np.inf, dtype=np.float64)
    chunk_count = 0
    for chunk in iter_candidate_batch_pair_feature_chunks_rust(
        dataset,
        candidate_batch,
        matrix_indices=aggregate_indices,
        aggregate_indices=aggregate_indices,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
        chunk_plan=plan,
    ):
        chunk_count += 1
        observed = chunk.counts > 0
        if not np.any(observed):
            continue
        rows = chunk.global_row_indices[observed]
        counts[rows] += chunk.counts[observed].astype(np.uint64, copy=False)
        if chunk.valid_counts is None:
            raise RuntimeError("nan-aware pairwise aggregate chunks must include valid_counts")
        valid_counts[rows] += chunk.valid_counts[observed]
        sums[rows] += chunk.sums[observed]
        mins[rows] = np.minimum(mins[rows], chunk.mins[observed])
        maxs[rows] = np.maximum(maxs[rows], chunk.maxs[observed])
    return PairwiseAggregateStats(
        counts=counts,
        sums=sums,
        mins=mins,
        maxs=maxs,
        base_feature_names=aggregate_feature_names,
        aggregate_feature_columns=aggregate_columns,
        chunk_plan=plan,
        chunk_count=int(chunk_count),
        matrix_indices=tuple(aggregate_indices),
        aggregate_indices=tuple(aggregate_indices),
        valid_counts=valid_counts,
    )


def compute_pairwise_aggregate_stats_rust(
    dataset: ANDData,
    pairs: Sequence[tuple[Any, Any]],
    row_indices: Sequence[int],
    *,
    row_count: int,
    aggregate_feature_names: Sequence[str] = PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
    pairs_are_indices: bool = False,
) -> PairwiseAggregateStats:
    """Compute promoted pairwise aggregate stats with memory-bounded Rust chunks."""

    aggregate_feature_names = tuple(str(feature_name) for feature_name in aggregate_feature_names)
    aggregate_indices = tuple(
        PROD_PAIRWISE_FEATURE_INDICES[PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in aggregate_feature_names
    )
    aggregate_columns = tuple(
        f"pw_{stat}_{feature_name}" for stat in ("min", "mean", "max") for feature_name in aggregate_feature_names
    )
    plan = compute_linker_pair_chunk_plan(
        total_pairs=len(pairs),
        row_count=int(row_count),
        matrix_feature_count=len(aggregate_indices),
        aggregate_feature_count=len(aggregate_indices),
        total_ram_bytes=total_ram_bytes,
    )
    counts = np.zeros(int(row_count), dtype=np.uint64)
    valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
    sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
    mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
    maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
    chunk_count = 0
    for chunk in iter_linker_pair_feature_chunks_rust(
        dataset,
        pairs,
        row_indices,
        row_count=int(row_count),
        matrix_indices=aggregate_indices,
        aggregate_indices=aggregate_indices,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
        chunk_plan=plan,
        pairs_are_indices=pairs_are_indices,
    ):
        chunk_count += 1
        observed = chunk.counts > 0
        if not np.any(observed):
            continue
        rows = chunk.global_row_indices[observed]
        counts[rows] += chunk.counts[observed].astype(np.uint64, copy=False)
        if chunk.valid_counts is None:
            raise RuntimeError("nan-aware pairwise aggregate chunks must include valid_counts")
        valid_counts[rows] += chunk.valid_counts[observed]
        sums[rows] += chunk.sums[observed]
        mins[rows] = np.minimum(mins[rows], chunk.mins[observed])
        maxs[rows] = np.maximum(maxs[rows], chunk.maxs[observed])
    return PairwiseAggregateStats(
        counts=counts,
        sums=sums,
        mins=mins,
        maxs=maxs,
        base_feature_names=aggregate_feature_names,
        aggregate_feature_columns=aggregate_columns,
        chunk_plan=plan,
        chunk_count=int(chunk_count),
        matrix_indices=tuple(aggregate_indices),
        aggregate_indices=tuple(aggregate_indices),
        valid_counts=valid_counts,
    )
