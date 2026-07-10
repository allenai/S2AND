"""Rust-backed pairwise aggregate features for linker candidate rows."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from s2and import feature_port, memory_budget
from s2and.data import ANDData
from s2and.featurizer import DEFAULT_FEATURE_GROUPS, FeaturizationInfo
from s2and.incremental_linking.array_validation import as_retrieval_rank_uint16_1d, as_uint32_1d
from s2and.thread_config import resolve_n_jobs

PAIRWISE_INFO = FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS))
PROD_PAIRWISE_FEATURE_NAMES: tuple[str, ...] = tuple(PAIRWISE_INFO.get_feature_names())
PROD_PAIRWISE_FEATURE_INDICES: tuple[int, ...] = tuple(
    feature_index
    for feature_group in DEFAULT_FEATURE_GROUPS
    for feature_index in PAIRWISE_INFO.feature_group_to_index[feature_group]
)
PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    (
        "pw_max_affiliation_overlap",
        "pw_max_middle_initials_overlap",
        "pw_mean_email_prefix_equal",
        "pw_mean_first_names_equal",
        "pw_min_middle_initials_overlap",
        "pw_max_title_overlap_words",
        "pw_max_journal_overlap",
        "pw_mean_middle_names_equal",
        "pw_min_last_first_name_count_max",
        "pw_mean_coauthor_match",
        "pw_mean_coauthor_overlap",
        "pw_mean_title_overlap_words",
        "pw_max_venue_overlap",
        "pw_mean_journal_overlap",
        "pw_min_specter_cosine_sim",
        "pw_min_first_name_count_max",
        "pw_max_coauthor_overlap",
        "pw_max_jaro",
        "pw_min_first_name_count_min",
        "pw_min_levenshtein",
        "pw_mean_english_count",
        "pw_mean_middle_one_missing",
        "pw_mean_specter_cosine_sim",
    )
)


def _pairwise_aggregate_column_parts(column: str) -> tuple[str, str]:
    for stat in ("min", "mean", "max"):
        prefix = f"pw_{stat}_"
        if column.startswith(prefix):
            return stat, column[len(prefix) :]
    raise ValueError(f"Unsupported pairwise aggregate column name: {column}")


PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES: tuple[str, ...] = tuple(
    dict.fromkeys(_pairwise_aggregate_column_parts(column)[1] for column in PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS)
)
PROMOTED_PAIRWISE_AGG_FEATURE_INDICES: tuple[int, ...] = tuple(
    PROD_PAIRWISE_FEATURE_INDICES[PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
    for feature_name in PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES
)


@dataclass(frozen=True)
class LinkerPairFeatureChunk:
    """One memory-bounded Rust pair-feature chunk plus aggregate stats."""

    start: int
    stop: int
    pair_features: np.ndarray
    matrix_indices: tuple[int, ...]
    aggregate_indices: tuple[int, ...]
    global_row_indices: np.ndarray
    local_row_indices: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray
    valid_counts: np.ndarray | None = None
    feature_seconds: float = 0.0


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
        """Return promoted pairwise aggregate columns in artifact order."""

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
        means = self.mean_matrix()
        position_by_name = {feature_name: index for index, feature_name in enumerate(self.base_feature_names)}
        columns: list[np.ndarray] = []
        for column in self.aggregate_feature_columns:
            stat, feature_name = _pairwise_aggregate_column_parts(column)
            position = position_by_name[feature_name]
            if stat == "min":
                columns.append(mins[:, position])
            elif stat == "mean":
                columns.append(means[:, position])
            elif stat == "max":
                columns.append(maxs[:, position])
            else:  # pragma: no cover - parser above constrains the domain
                raise ValueError(f"Unsupported pairwise aggregate stat: {stat}")
        return np.column_stack(columns) if columns else np.empty((len(self.counts), 0), dtype=np.float64)


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
        ranks = (
            None
            if self.retrieval_ranks is None
            else as_retrieval_rank_uint16_1d("retrieval_ranks", self.retrieval_ranks)
        )
        if ranks is not None and len(ranks) != row_count:
            raise ValueError(f"retrieval_ranks must have length row_count: {len(ranks)} != {row_count}")
        object.__setattr__(self, "row_count", row_count)
        object.__setattr__(self, "left_signature_indices", left)
        object.__setattr__(self, "right_signature_indices", right)
        object.__setattr__(self, "pair_row_indices", rows)
        object.__setattr__(self, "retrieval_ranks", ranks)
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


def _as_uint32_1d(name: str, values: Sequence[Any] | np.ndarray) -> np.ndarray:
    return as_uint32_1d(name, values)


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
        retrieval_ranks=None
        if retrieval_ranks is None
        else as_retrieval_rank_uint16_1d("retrieval_ranks", retrieval_ranks),
    )


def _ordered_union(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(dict.fromkeys([int(value) for value in (*left, *right)]))


def _localize_row_indices(row_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return global rows plus chunk-local row indices for a pair-owner row slice."""

    if len(row_chunk) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint32)
    if len(row_chunk) == 1 or bool(np.all(row_chunk[1:] >= row_chunk[:-1])):
        starts = np.empty(len(row_chunk), dtype=bool)
        starts[0] = True
        starts[1:] = row_chunk[1:] != row_chunk[:-1]
        global_rows = np.ascontiguousarray(row_chunk[starts], dtype=np.int64)
        local_row_indices = np.ascontiguousarray(
            np.cumsum(starts, dtype=np.uint32) - np.uint32(1),
            dtype=np.uint32,
        )
        return global_rows, local_row_indices
    global_rows, local_row_indices = np.unique(row_chunk, return_inverse=True)
    return np.ascontiguousarray(global_rows, dtype=np.int64), np.ascontiguousarray(local_row_indices, dtype=np.uint32)


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
        index_remap_bytes_per_pair=memory_budget.BORROWED_SIGNATURE_INDEX_REMAP_BYTES_PER_PAIR,
    )


def _chunk_plan_pairs(plan: memory_budget.RustBatchChunkPlan) -> int:
    return int(plan.chunk_pairs)


def resolve_linker_pairwise_featurizer(
    dataset: ANDData | None,
    featurizer: Any | None,
    *,
    runtime_context: Any | None = None,
) -> Any:
    """Return the Rust featurizer used for linker pairwise array APIs."""

    if featurizer is not None:
        return featurizer
    if dataset is None:
        raise ValueError("dataset is required when featurizer is not provided")
    return feature_port._get_rust_featurizer(dataset, runtime_context=runtime_context)  # noqa: SLF001


def iter_candidate_batch_pair_feature_chunks_rust(
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    *,
    matrix_indices: Sequence[int] | None = None,
    aggregate_indices: Sequence[int] = PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    aggregate_nan_value: float | None = None,
    runtime_context: Any | None = None,
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
    chunk_pairs = _chunk_plan_pairs(plan)
    featurizer = resolve_linker_pairwise_featurizer(dataset, featurizer, runtime_context=runtime_context)

    for start in range(0, candidate_batch.pair_count, chunk_pairs):
        stop = min(candidate_batch.pair_count, start + chunk_pairs)
        row_chunk = candidate_batch.pair_row_indices[start:stop]
        global_rows, local_row_indices = _localize_row_indices(row_chunk)
        feature_start = time.perf_counter()
        pair_features, counts, valid_counts, sums, mins, maxs = (
            feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
                candidate_batch.left_signature_indices[start:stop],
                candidate_batch.right_signature_indices[start:stop],
                local_row_indices,
                len(global_rows),
                featurizer=featurizer,
                matrix_indices=list(resolved_matrix_indices),
                aggregate_indices=list(resolved_aggregate_indices),
                num_threads=resolve_n_jobs(n_jobs),
                nan_value=float(nan_value),
                aggregate_nan_value=aggregate_nan_value,
            )
        )
        feature_seconds = time.perf_counter() - feature_start
        yield LinkerPairFeatureChunk(
            start=int(start),
            stop=int(stop),
            pair_features=pair_features,
            matrix_indices=resolved_matrix_indices,
            aggregate_indices=resolved_aggregate_indices,
            global_row_indices=global_rows,
            local_row_indices=local_row_indices,
            counts=counts,
            sums=sums,
            mins=mins,
            maxs=maxs,
            valid_counts=valid_counts,
            feature_seconds=float(feature_seconds),
        )


def compute_candidate_batch_pairwise_aggregate_stats_rust(
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    *,
    aggregate_feature_names: Sequence[str] = PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = math.nan,
    aggregate_nan_value: float | None = None,
    runtime_context: Any | None = None,
    featurizer: Any | None = None,
) -> PairwiseAggregateStats:
    """Compute promoted pairwise aggregates for training or production candidate batches."""

    aggregate_feature_names = tuple(str(feature_name) for feature_name in aggregate_feature_names)
    aggregate_indices = tuple(
        PROD_PAIRWISE_FEATURE_INDICES[PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in aggregate_feature_names
    )
    aggregate_columns = (
        PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS
        if aggregate_feature_names == PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES
        else tuple(
            f"pw_{stat}_{feature_name}" for stat in ("min", "mean", "max") for feature_name in aggregate_feature_names
        )
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
    chunk_pairs = _chunk_plan_pairs(plan)
    featurizer = resolve_linker_pairwise_featurizer(dataset, featurizer, runtime_context=runtime_context)
    for start in range(0, candidate_batch.pair_count, chunk_pairs):
        stop = min(candidate_batch.pair_count, start + chunk_pairs)
        row_chunk = candidate_batch.pair_row_indices[start:stop]
        global_rows, local_row_indices = _localize_row_indices(row_chunk)
        chunk_counts, chunk_valid_counts, chunk_sums, chunk_mins, chunk_maxs = (
            feature_port.build_linker_pair_aggregate_stats_arrays_rust(
                candidate_batch.left_signature_indices[start:stop],
                candidate_batch.right_signature_indices[start:stop],
                local_row_indices,
                len(global_rows),
                featurizer=featurizer,
                aggregate_indices=list(aggregate_indices),
                num_threads=resolve_n_jobs(n_jobs),
                nan_value=float(nan_value),
                aggregate_nan_value=aggregate_nan_value,
            )
        )
        chunk_count += 1
        observed = chunk_counts > 0
        if not np.any(observed):
            continue
        rows = global_rows[observed]
        counts[rows] += chunk_counts[observed].astype(np.uint64, copy=False)
        valid_counts[rows] += chunk_valid_counts[observed]
        sums[rows] += chunk_sums[observed]
        mins[rows] = np.minimum(mins[rows], chunk_mins[observed])
        maxs[rows] = np.maximum(maxs[rows], chunk_maxs[observed])
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
