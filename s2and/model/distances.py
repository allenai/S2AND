"""Score pairs and build distance matrices for clustering."""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tqdm import tqdm

from s2and import memory_budget
from s2and.block_corrections import BlockCorrections, corrections_copy_codes
from s2and.consts import DEFAULT_CHUNK_SIZE, LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import _get_rust_feature_data
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer as _require_arrow_name_counts_index_for_clusterer,
)
from s2and.incremental_linking.policy import (
    require_dataset_name_counts_binding_for_clusterer,
    require_rust_featurizer_name_counts_binding_for_clusterer,
)
from s2and.model_pairwise import FastCluster, native_pairwise_scorers, predict_pairwise_class0
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context, dataset_stage_uses_rust, stage_uses_rust
from s2and.rust_calls import (
    _get_constraint_labels_block_upper_triangle_indexed_rust,
    build_block_upper_triangle_feature_matrix_indexed_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    take_feature_columns_rust,
)
from s2and.supervision import normalize_partial_supervision
from s2and.thread_config import resolve_n_jobs

from .constraints import (
    _accumulate_constraint_telemetry,
    _build_incremental_constraint_backend,
    _build_signature_index_by_id,
    _ConstraintTelemetryAccumulator,
    _IncrementalConstraintBackend,
    _orcid_name_waiver_setting,
    _profile_exclusion_kwargs,
)
from .incremental import logger

if TYPE_CHECKING:
    from .clusterer import Clusterer

_PREDICT_FEATURE_COPY_MAX_BYTES = 128 * (1 << 20)


# Blocks up to this size fill their distance matrices concurrently; larger
# ones have enough tiles to use the pool alone.
_CONCURRENT_DISTANCE_BLOCK_MAX_SIZE = 512


_CONCURRENT_DISTANCE_BLOCKS = 8


def _count_selected_features(featurizer_info: FeaturizationInfo) -> int:
    """Count the number of feature indices selected by features_to_use."""
    return len(_selected_feature_indices(featurizer_info))


def _selected_feature_indices(featurizer_info: FeaturizationInfo) -> list[int]:
    return featurizer_info.selected_feature_indices()


def _build_block_feature_matrices_indexed_rust(
    block_signature_indices: list[int],
    *,
    featurizer: Any,
    start_offset: int,
    max_pairs: int,
    main_indices: list[int],
    nameless_indices: list[int] | None,
    num_threads: int,
    scored_rows: np.ndarray | None = None,
    compact: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build shared pair features, optionally omitting rows fixed by supervision.

    Args:
        scored_rows: Chunk-relative rows consumed by the classifiers. Other
            rows retain NaN placeholders and must be masked before scoring.
        compact: Return only scored rows when scored_rows is supplied.

    Returns:
        Main and optional nameless matrices, compact only when requested.
    """
    union_indices = list(dict.fromkeys(main_indices + (nameless_indices or [])))
    if scored_rows is None:
        union_features = build_block_upper_triangle_feature_matrix_indexed_rust(
            block_signature_indices,
            start_offset=start_offset,
            max_pairs=max_pairs,
            selected_indices=union_indices,
            num_threads=num_threads,
            nan_value=np.nan,
            featurizer=featurizer,
        )
    else:
        # Rust derives the scored rows' pairs itself; Python tuples for millions of
        # pairs cost more than featurizing them.
        sparse_features = featurizer.featurize_block_upper_triangle_rows_indexed(
            block_signature_indices,
            start_offset,
            max_pairs,
            np.ascontiguousarray(scored_rows, dtype=np.int64),
            union_indices,
            num_threads,
            np.nan,
        )
        # The bulk consumer can keep scored rows compact. Other callers retain
        # full matrix shapes and mask constrained rows before classification.
        if compact:
            union_features = sparse_features
        else:
            union_features = np.full((max_pairs, len(union_indices)), np.nan, dtype=np.float64)
            union_features[scored_rows] = sparse_features
        del sparse_features
    position_by_index = {index: position for position, index in enumerate(union_indices)}

    def project(indices: list[int]) -> np.ndarray:
        if indices == union_indices:
            return union_features
        return take_feature_columns_rust(
            union_features,
            [position_by_index[index] for index in indices],
            num_threads=num_threads,
        )

    return project(main_indices), None if nameless_indices is None else project(nameless_indices)


def _condensed_pair_index(block_size: int, left: int, right: int) -> int:
    if left >= right:
        raise ValueError(f"Expected left < right; got left={left} right={right}")
    return int(block_size * left - (left * (left + 1) // 2) + (right - left - 1))


def _build_partial_supervision_offset_maps_for_block(
    signatures: list[str],
    partial_supervision: dict[tuple[str, str], int | float],
) -> tuple[dict[int, float], dict[int, float]]:
    if not partial_supervision:
        return {}, {}
    signature_to_local_idx = {signature: idx for idx, signature in enumerate(signatures)}
    block_size = len(signatures)
    direct_overrides: dict[int, float] = {}
    reverse_overrides: dict[int, float] = {}
    for (sig_id_1, sig_id_2), value in partial_supervision.items():
        left = signature_to_local_idx.get(sig_id_1)
        right = signature_to_local_idx.get(sig_id_2)
        if left is None or right is None or left == right:
            continue
        adjusted = float(value - LARGE_INTEGER)
        if left < right:
            offset = _condensed_pair_index(block_size, left, right)
            direct_overrides[offset] = adjusted
        else:
            offset = _condensed_pair_index(block_size, right, left)
            reverse_overrides[offset] = adjusted
    return direct_overrides, reverse_overrides


def _partial_supervision_label_overrides_for_block(
    signatures: list[str],
    partial_supervision: dict[tuple[str, str], int | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted condensed pair offsets and their supervision labels for one block.

    A pair given in block order wins over the same pair given in reverse order.
    """
    direct_overrides, reverse_overrides = _build_partial_supervision_offset_maps_for_block(
        signatures,
        partial_supervision,
    )
    overrides = {**reverse_overrides, **direct_overrides}
    offsets = sorted(overrides)
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray([overrides[pair_offset] for pair_offset in offsets], dtype=np.float64),
    )


def _compute_predict_batch_chunk_plan(
    num_features: int,
    *,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_pairs: int,
    total_ram_bytes: int | None = None,
) -> memory_budget.RustBatchChunkPlan | None:
    """Compute a bounded pair-batch plan for exact/non-incremental prediction."""
    if total_ram_bytes is None:
        return None
    return memory_budget.compute_rust_batch_chunk_plan(
        num_features=num_features,
        total_pairs=max(1, int(total_pairs)),
        total_rows=max(1, int(total_pairs)),
        selected_feature_count=selected_feature_count,
        nameless_feature_count=nameless_feature_count,
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )


def _predict_distance_matrix_bytes(
    block_size: int,
    *,
    uses_fastcluster: bool,
    fastcluster_fused_dtype: Any,
) -> int:
    """Estimate bytes needed for the per-block distance matrix allocation."""
    if block_size <= 1:
        return 0
    if uses_fastcluster:
        itemsize = int(np.dtype(fastcluster_fused_dtype).itemsize)
        return int(block_size * (block_size - 1) // 2 * itemsize)
    return int(block_size * block_size * np.dtype(np.float16).itemsize)


def _guard_predict_block_matrix_allocation(
    *,
    block_key: str,
    block_size: int,
    uses_fastcluster: bool,
    fastcluster_fused_dtype: Any,
    total_ram_bytes: int | None,
    copies_codes: bool = False,
) -> None:
    """Fail fast before exact prediction allocates a block matrix above the RAM budget.

    Linkage works in the matrix, so the matrix is the block's peak. The post-linkage
    corrections add their one-byte-per-pair copy. A block with profile exclusions also keeps a
    second matrix for relinking, which this guard does not count.
    """
    if total_ram_bytes is None or block_size <= 1:
        return
    snapshot = memory_budget.memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )
    matrix_bytes = _predict_distance_matrix_bytes(
        block_size,
        uses_fastcluster=uses_fastcluster,
        fastcluster_fused_dtype=fastcluster_fused_dtype,
    )
    if copies_codes and uses_fastcluster:
        matrix_bytes += int(block_size * (block_size - 1) // 2)
    if matrix_bytes <= int(snapshot.available_bytes):
        return
    raise MemoryError(
        "Predict exact block exceeds memory budget before matrix allocation: "
        f"block={block_key} block_size={block_size} matrix_bytes={matrix_bytes} "
        f"available_bytes={int(snapshot.available_bytes)} total_ram_bytes={int(snapshot.total_ram_bytes)} "
        f"current_rss_bytes={int(snapshot.current_rss_bytes)} total_ram_source={snapshot.total_ram_source} "
        f"current_rss_source={snapshot.current_rss_source}"
    )


def _ensure_lightgbm_fitted(clf: Any) -> None:
    if clf is None:
        return
    inner = getattr(clf, "classifier", None)
    if inner is not None and inner is not clf:
        _ensure_lightgbm_fitted(inner)
    # A LightGBM estimator exists only after LightGBM was imported, so prediction never imports it here.
    lgb = sys.modules.get("lightgbm")
    if lgb is None or not hasattr(lgb, "LGBMModel") or not isinstance(clf, lgb.LGBMModel):
        return
    booster = getattr(clf, "_Booster", None)
    if booster is None:
        raise RuntimeError(
            "LightGBM estimator has no fitted booster (_Booster is None); fit the estimator before prediction."
        )
    if not getattr(clf, "fitted_", False):
        logger.debug("Patching missing LightGBM fitted_ flag for estimator=%s", type(clf).__name__)
        clf.fitted_ = True
    if not hasattr(clf, "n_features_in_"):
        n_feat = getattr(clf, "_n_features", None)
        if n_feat is not None:
            logger.debug(
                "Patching missing LightGBM n_features_in_ from _n_features=%d for estimator=%s",
                int(n_feat),
                type(clf).__name__,
            )
            clf.n_features_in_ = n_feat


def _predict_class0_with_runtime(
    classifier: Any,
    features: np.ndarray,
    *,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, float, str]:
    raw_features = np.asarray(features)
    native_backend = str(getattr(classifier, "prediction_backend", "python")) == "rust_lightgbm"
    features_2d = raw_features if native_backend else np.asarray(raw_features, dtype=np.float64, order="C")
    if features_2d.size == 0:
        return np.asarray([], dtype=np.float64), 0.0, "none"

    if native_backend:
        if features_2d.ndim != 2:
            raise ValueError(f"native scorer features must be 2D, got shape={features_2d.shape}")
        if features_2d.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError(f"native scorer features must be float32 or float64, got {features_2d.dtype}")
        if not features_2d.flags.c_contiguous:
            raise ValueError("native scorer features must be C-contiguous")
        scorer_plan = memory_budget.compute_native_scorer_chunk_plan(
            row_count=int(features_2d.shape[0]),
            feature_count=int(features_2d.shape[1]),
            input_itemsize=int(features_2d.dtype.itemsize),
            total_ram_bytes=total_ram_bytes,
        )
        started = time.perf_counter()
        positive = classifier.predict_proba_positive(
            features_2d,
            max_rows_per_chunk=max(1, scorer_plan.chunk_rows),
        )
        logger.info(
            "Telemetry: native_scorer rows=%d features=%d dtype=%s chunk_rows=%d chunks=%d "
            "predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d",
            int(features_2d.shape[0]),
            int(features_2d.shape[1]),
            str(features_2d.dtype),
            int(scorer_plan.chunk_rows),
            int(scorer_plan.chunk_count),
            int(scorer_plan.predicted_peak_delta_bytes),
            int(scorer_plan.predicted_peak_rss_bytes),
        )
        backend = "rust_lightgbm_chunked" if scorer_plan.chunk_count > 1 else "rust_lightgbm"
        return 1.0 - np.asarray(positive, dtype=np.float64), time.perf_counter() - started, backend

    python_start = time.perf_counter()
    predictions = predict_pairwise_class0(classifier, features_2d)
    backend = str(getattr(classifier, "prediction_backend", "python"))
    return predictions, time.perf_counter() - python_start, backend


def _predict_and_combine(
    classifier: Any,
    nameless_classifier: Any | None,
    features: np.ndarray,
    labels: np.ndarray,
    nameless_features: np.ndarray | None,
    batch_label: int | str,
    *,
    total_ram_bytes: int | None = None,
    compact_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Predict with main (and optional nameless) classifier, log telemetry, return (predictions, seconds)."""
    row_count = int(features.shape[0] if compact_rows is None else len(labels))
    if row_count <= 0:
        return np.asarray([], dtype=np.float64), 0.0

    predict_flag = np.isnan(labels)
    not_predict_flag = ~predict_flag
    predicted_rows = int(np.count_nonzero(predict_flag))
    predictions = np.zeros(row_count)
    seconds = 0.0

    def _predict_rows(
        main_matrix: np.ndarray,
        nameless_matrix: np.ndarray | None,
        *,
        row_total: int,
    ) -> tuple[np.ndarray, float]:
        main_pred, main_sec, main_backend = _predict_class0_with_runtime(
            classifier,
            main_matrix,
            total_ram_bytes=total_ram_bytes,
        )
        if nameless_classifier is not None:
            if nameless_matrix is None:
                raise RuntimeError("nameless_classifier is configured but nameless feature matrix is missing")
            nl_pred, nl_sec, nl_backend = _predict_class0_with_runtime(
                nameless_classifier,
                nameless_matrix,
                total_ram_bytes=total_ram_bytes,
            )
            logger.info(
                "Telemetry: model_predict batch=%s main=%s nameless=%s main_s=%.3f nl_s=%.3f rows=%d",
                batch_label,
                main_backend,
                nl_backend,
                main_sec,
                nl_sec,
                row_total,
            )
            return (main_pred + nl_pred) / 2, main_sec + nl_sec

        logger.info(
            "Telemetry: model_predict batch=%s main=%s main_s=%.3f rows=%d",
            batch_label,
            main_backend,
            main_sec,
            row_total,
        )
        return main_pred, main_sec

    # Boolean indexing creates a large temporary copy (often comparable to the full
    # feature matrix) when most rows are predicted. Keep each copy bounded without
    # exposing constrained sentinel rows to the classifiers.
    # Compact inputs retain the same call boundaries and independent writable
    # copies: arbitrary classifiers may depend on batching or mutate their input.
    would_copy_bytes = 0
    if predicted_rows > 0 and predicted_rows < row_count:
        would_copy_bytes += int(predicted_rows) * int(features.shape[1]) * int(features.dtype.itemsize)
        if nameless_classifier is not None and nameless_features is not None:
            would_copy_bytes += (
                int(predicted_rows) * int(nameless_features.shape[1]) * int(nameless_features.dtype.itemsize)
            )

    # Compact rows are the predicted rows in order, so each batch is a contiguous
    # range. The native scorer copies its input and scores rows independently,
    # so it takes those ranges as views; the copies cost seconds per giant block.
    compact_views = compact_rows is not None and all(
        model is None or str(getattr(model, "prediction_backend", "python")) == "rust_lightgbm"
        for model in (classifier, nameless_classifier)
    )
    if predicted_rows > 0:
        if predicted_rows == row_count:
            combined_predictions, batch_seconds = _predict_rows(
                features,
                nameless_features,
                row_total=predicted_rows,
            )
            predictions[:] = combined_predictions
            seconds += batch_seconds
        elif would_copy_bytes >= _PREDICT_FEATURE_COPY_MAX_BYTES:
            predicted_indices = np.flatnonzero(predict_flag)
            bytes_per_row = max(1, would_copy_bytes // predicted_rows)
            rows_per_chunk = max(1, _PREDICT_FEATURE_COPY_MAX_BYTES // bytes_per_row)
            for start in range(0, predicted_rows, rows_per_chunk):
                row_indices = predicted_indices[start : start + rows_per_chunk]
                if compact_views:
                    feature_rows: Any = slice(start, start + len(row_indices))
                elif compact_rows is None:
                    feature_rows = row_indices
                else:
                    feature_rows = np.arange(start, start + len(row_indices))
                predict_features = features[feature_rows, :]
                predict_nameless_features = (
                    nameless_features[feature_rows, :] if nameless_features is not None else None
                )
                combined_predictions, batch_seconds = _predict_rows(
                    predict_features,
                    predict_nameless_features,
                    row_total=len(row_indices),
                )
                predictions[row_indices] = combined_predictions
                seconds += batch_seconds
        else:
            if compact_views:
                feature_rows = slice(0, predicted_rows)
            elif compact_rows is None:
                feature_rows = predict_flag
            else:
                feature_rows = np.arange(predicted_rows)
            predict_features = features[feature_rows, :]
            predict_nameless_features = nameless_features[feature_rows, :] if nameless_features is not None else None
            combined_predictions, batch_seconds = _predict_rows(
                predict_features,
                predict_nameless_features,
                row_total=predicted_rows,
            )
            predictions[predict_flag] = combined_predictions
            seconds += batch_seconds

    if np.any(not_predict_flag):
        # Fill rows where we already had constraints/partial supervision.
        # Undo the LARGE_INTEGER offset that was applied when labels were staged.
        # For classifier outputs, index 0 corresponds to p(not the same).
        predictions[not_predict_flag] = labels[not_predict_flag] + LARGE_INTEGER
    return predictions, seconds


def _native_block_scorers(classifier: Any, nameless_classifier: Any | None) -> tuple[Any, Any | None] | None:
    """Return the Rust boosters of native classifiers, for fused block scoring.

    Fused scoring reproduces `_predict_and_combine` for native boosters only;
    other classifiers keep the feature-matrix path. Returns None when either
    classifier is not native, or when fused scoring is switched off.
    """

    if not _FUSED_BLOCK_SCORING:
        return None
    return native_pairwise_scorers(classifier, nameless_classifier)


# Tests compare the fused block scorer against the feature-matrix path.
_FUSED_BLOCK_SCORING = True


# Tests compare the one-pass block fill against the chunked fused loop.
_ONE_PASS_BLOCK_FILL = True


# Tests shrink the one-pass tiles to cross tile edges in small blocks.
_one_pass_tile_overrides: dict[str, int] = {}


@dataclass(frozen=True)
class _DistanceMatrixChunk:
    block_key: str
    block_size: int
    start_offset: int
    index_i: np.ndarray
    index_j: np.ndarray
    pair_ids: list[tuple[str, str]] | None
    labels: np.ndarray
    block_signature_indices: list[int] | None = None

    def signature_pairs(self) -> list[tuple[str, str, float]]:
        if self.pair_ids is None:
            raise RuntimeError("signature_pairs requested for fused Rust chunk without explicit pair ids")
        return [
            (sig_id_1, sig_id_2, float(label))
            for (sig_id_1, sig_id_2), label in zip(self.pair_ids, self.labels, strict=True)
        ]


@dataclass(frozen=True)
class _PredictedDistanceMatrixChunk:
    chunk: _DistanceMatrixChunk
    predictions: np.ndarray
    batch_seconds: float


@dataclass(frozen=True)
class _PredictedDistanceMatrixBatch:
    batch_num: int
    blocks: list[str]
    indices: list[tuple[int, int]]
    predictions: np.ndarray
    batch_seconds: float


def _upper_triangle_indices_for_range(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = int(block_size * (block_size - 1) // 2)
    start = int(start_offset)
    if start < 0 or start > total_pairs:
        raise ValueError(f"start_offset must be in [0, {total_pairs}], got {start}")
    count = total_pairs - start if max_pairs is None else min(int(max_pairs), total_pairs - start)
    if count <= 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)

    linear = np.arange(start, start + count, dtype=np.int64)
    block_size_int = int(block_size)
    discriminant = (2 * block_size_int - 1) ** 2 - 8 * linear
    left = np.floor(((2 * block_size_int - 1) - np.sqrt(discriminant)) / 2).astype(np.int64)
    row_start = left * (2 * block_size_int - left - 1) // 2

    too_high = row_start > linear
    if np.any(too_high):
        left[too_high] -= 1
        row_start = left * (2 * block_size_int - left - 1) // 2
    next_row_start = (left + 1) * (2 * block_size_int - left - 2) // 2
    too_low = next_row_start <= linear
    if np.any(too_low):
        left[too_low] += 1
        row_start = left * (2 * block_size_int - left - 1) // 2

    right = left + 1 + (linear - row_start)
    return left.astype(np.intp, copy=False), right.astype(np.intp, copy=False)


def _flush_completed_block(
    self: Clusterer,
    *,
    block_key: str,
    pairwise_proba: np.ndarray | None,
    block_dict: dict[str, list[str]],
    effective_cluster_model_params: dict[str, Any] | None,
    dataset: ANDData,
    disallow_adjacency: Mapping[str, set[str]],
    pred_clusters: defaultdict[str, list[str]],
    incremental_dont_use_cluster_seeds: bool,
    prediction_state: PredictionState | None = None,
    corrections: BlockCorrections | None = None,
) -> None:
    if block_key == "" or pairwise_proba is None:
        return
    if not isinstance(self.cluster_model, FastCluster):
        pairwise_proba += pairwise_proba.T
        np.fill_diagonal(pairwise_proba, 0)
    labels = self._cluster_one_block_with_logging(
        block_dict[block_key],
        pairwise_proba,
        effective_cluster_model_params,
        dataset,
        disallow_adjacency,
        block_key=block_key,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        prediction_state=prediction_state,
        corrections=corrections,
        **_profile_exclusion_kwargs(prediction_state),
    )
    for signature, label in zip(block_dict[block_key], labels, strict=True):
        pred_clusters[block_key + "_" + str(label)].append(signature)


def distance_matrix_helper(
    self: Clusterer,
    block_dict: dict,
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    pair_chunk_size: int | None = None,
    prediction_state: PredictionState | None = None,
):
    """
    Helper generator function to yield one pair for batch featurization on the fly

    Parameters
    ----------
    block_dict: Dict
        the block dictionary
    dataset: ANDData
        the dataset
    partial_supervision: Dict
        the dictionary of partial supervision provided with this dataset/these blocks
    incremental_dont_use_cluster_seeds: bool
        whether to ignore dataset cluster seeds while resolving constraints in incremental flows

    Returns
    -------
    yields pairs of ((sig id 1, sig id 2, label), index pair into the distance matrix, block key)
    """
    if runtime_context is None:
        runtime_context = dataset.runtime_context
    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        suppress_orcid=getattr(self, "suppress_orcid", False),
        prediction_state=prediction_state,
    )

    telemetry = _ConstraintTelemetryAccumulator()
    pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))

    for block_key, signatures in block_dict.items():
        pair_batch_ids: list[tuple[str, str]] = []
        index_batch: list[tuple[int, int]] = []
        for i, j in zip(*np.triu_indices(len(signatures), k=1), strict=True):
            pair_batch_ids.append((signatures[i], signatures[j]))
            index_batch.append((i, j))
            if len(pair_batch_ids) >= pair_chunk_size:
                labels, batch_telemetry = self._resolve_constraint_batch(
                    dataset,
                    pair_batch_ids,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    constraint_backend=constraint_backend,
                )
                _accumulate_constraint_telemetry(telemetry, batch_telemetry)
                for (sig_id_1, sig_id_2), label, (left, right) in zip(pair_batch_ids, labels, index_batch, strict=True):
                    yield ((sig_id_1, sig_id_2, label), (left, right), block_key)
                pair_batch_ids = []
                index_batch = []

        if pair_batch_ids:
            labels, batch_telemetry = self._resolve_constraint_batch(
                dataset,
                pair_batch_ids,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                constraint_backend=constraint_backend,
            )
            _accumulate_constraint_telemetry(telemetry, batch_telemetry)
            for (sig_id_1, sig_id_2), label, (left, right) in zip(pair_batch_ids, labels, index_batch, strict=True):
                yield ((sig_id_1, sig_id_2, label), (left, right), block_key)

    logger.info(
        "Telemetry: constraint_batch stage=distance_matrix total_pairs=%d partial_supervision_hits=%d "
        "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
        telemetry.total_pairs,
        telemetry.partial_supervision_hits,
        telemetry.unresolved_pairs,
        telemetry.rust_batch_call_count,
        telemetry.api_mode_summary,
        telemetry.elapsed_seconds,
        runtime_context.run_id,
    )


def _yield_non_fused_chunks(
    self: Clusterer,
    *,
    block_key: str,
    signatures: list[str],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    incremental_dont_use_cluster_seeds: bool,
    constraint_backend: _IncrementalConstraintBackend,
    telemetry: _ConstraintTelemetryAccumulator,
    pair_chunk_size: int | None = None,
    start_offset: int = 0,
):
    block_size = len(signatures)
    if block_size <= 1:
        return
    block_pair_count = int(block_size * (block_size - 1) / 2)
    pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
    tri_i, tri_j = np.triu_indices(block_size, k=1)
    offset = int(start_offset)
    if offset < 0 or offset > block_pair_count:
        raise ValueError(f"start_offset must be between 0 and {block_pair_count}, got {start_offset}")
    while offset < block_pair_count:
        end = min(offset + pair_chunk_size, block_pair_count)
        i_chunk = tri_i[offset:end]
        j_chunk = tri_j[offset:end]
        pair_batch_ids = [
            (signatures[int(left)], signatures[int(right)]) for left, right in zip(i_chunk, j_chunk, strict=True)
        ]
        labels, batch_telemetry = self._resolve_constraint_batch(
            dataset,
            pair_batch_ids,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            constraint_backend=constraint_backend,
        )
        _accumulate_constraint_telemetry(telemetry, batch_telemetry)
        yield _DistanceMatrixChunk(
            block_key=block_key,
            block_size=block_size,
            start_offset=offset,
            index_i=i_chunk,
            index_j=j_chunk,
            pair_ids=pair_batch_ids,
            labels=np.asarray(labels, dtype=np.float64),
        )
        offset = end


def _distance_matrix_chunk_helper_rust(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    pair_chunk_size: int | None = None,
    prediction_state: PredictionState | None = None,
):
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints", backend="rust")
    if not stage_uses_rust(runtime_context):
        raise ValueError("Rust chunk helper is only valid when runtime_context resolves to rust backend")

    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        suppress_orcid=getattr(self, "suppress_orcid", False),
        prediction_state=prediction_state,
    )
    rust_featurizer = constraint_backend.rust_featurizer
    constraint_api_mode = constraint_backend.constraint_api_mode
    signature_index_by_id = constraint_backend.signature_index_by_id

    telemetry = _ConstraintTelemetryAccumulator()
    pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
    used_fused_path = False
    use_fused_block_api = bool(
        self.use_default_constraints_as_supervision
        and constraint_api_mode == "indexed"
        and rust_featurizer is not None
        and signature_index_by_id is not None
    )

    for block_key, signatures in block_dict.items():
        block_size = len(signatures)
        if block_size <= 1:
            continue
        block_pair_count = int(block_size * (block_size - 1) / 2)
        if use_fused_block_api and signature_index_by_id is not None and rust_featurizer is not None:
            block_signature_indices = [int(signature_index_by_id[signature]) for signature in signatures]
            direct_overrides, reverse_overrides = _build_partial_supervision_offset_maps_for_block(
                signatures,
                partial_supervision,
            )
            offset = 0
            while offset < block_pair_count:
                chunk_pair_count = int(min(pair_chunk_size, block_pair_count - offset))
                constraint_start = time.perf_counter()
                try:
                    local_i, local_j, values = get_constraints_block_upper_triangle_indexed_rust(
                        block_signature_indices,
                        start_offset=offset,
                        max_pairs=chunk_pair_count,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        num_threads=self.n_jobs,
                        featurizer=rust_featurizer,
                        suppress_orcid=constraint_backend.suppress_orcid,
                        orcid_name_waiver=_orcid_name_waiver_setting(self),
                    )
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        "Rust fused block constraint evaluation failed in strict rust backend "
                        f"(block={block_key} start_offset={offset} pairs={chunk_pair_count} "
                        f"run_id={runtime_context.run_id} error={exc})"
                    ) from exc
                constraint_elapsed = float(time.perf_counter() - constraint_start)
                if len(local_i) != chunk_pair_count or len(local_j) != chunk_pair_count:
                    raise RuntimeError(
                        "Rust fused block constraint API returned mismatched index lengths: "
                        f"expected={chunk_pair_count} left={len(local_i)} right={len(local_j)}"
                    )
                if len(values) != chunk_pair_count:
                    raise RuntimeError(
                        "Rust fused block constraint API returned mismatched constraint length: "
                        f"expected={chunk_pair_count} got={len(values)}"
                    )

                labels = np.full(chunk_pair_count, np.nan, dtype=np.float64)
                partial_hits_chunk = 0
                unresolved_chunk = 0
                for row_offset in range(chunk_pair_count):
                    pair_offset = offset + row_offset
                    override = direct_overrides.get(pair_offset)
                    if override is None:
                        override = reverse_overrides.get(pair_offset)
                    if override is not None:
                        labels[row_offset] = float(override)
                        partial_hits_chunk += 1
                        continue
                    unresolved_chunk += 1
                    value = values[row_offset]
                    if value is not None:
                        labels[row_offset] = float(value - LARGE_INTEGER)

                telemetry.total_pairs += int(chunk_pair_count)
                telemetry.partial_supervision_hits += int(partial_hits_chunk)
                telemetry.unresolved_pairs += int(unresolved_chunk)
                telemetry.elapsed_seconds += float(constraint_elapsed)
                telemetry.api_modes.add("indexed_fused")
                if unresolved_chunk > 0:
                    telemetry.rust_batch_call_count += 1
                used_fused_path = True

                yield _DistanceMatrixChunk(
                    block_key=block_key,
                    block_size=block_size,
                    start_offset=offset,
                    index_i=np.asarray(local_i, dtype=np.intp),
                    index_j=np.asarray(local_j, dtype=np.intp),
                    pair_ids=None,
                    labels=labels,
                    block_signature_indices=block_signature_indices,
                )
                offset += chunk_pair_count
        else:
            yield from self._yield_non_fused_chunks(
                block_key=block_key,
                signatures=signatures,
                dataset=dataset,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                constraint_backend=constraint_backend,
                telemetry=telemetry,
                pair_chunk_size=pair_chunk_size,
            )

    logger.info(
        "Telemetry: constraint_batch stage=distance_matrix total_pairs=%d partial_supervision_hits=%d "
        "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f path=%s run_id=%s",
        telemetry.total_pairs,
        telemetry.partial_supervision_hits,
        telemetry.unresolved_pairs,
        telemetry.rust_batch_call_count,
        telemetry.api_mode_summary,
        telemetry.elapsed_seconds,
        "chunked_rust_fused" if used_fused_path else "chunked_rust",
        runtime_context.run_id,
    )


def _predict_distance_matrix_chunk(
    self: Clusterer,
    chunk: _DistanceMatrixChunk,
    dataset: ANDData,
    runtime_context: RuntimeContext,
    batch_label: int | str,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, float]:
    if chunk.block_signature_indices is not None and chunk.pair_ids is None:
        try:
            rust_featurizer = _get_rust_feature_data(dataset)
            if self.featurizer_info.legacy_paper_years:
                rust_featurizer = rust_featurizer.with_legacy_paper_years()
            selected_indices = _selected_feature_indices(self.featurizer_info)
            nameless_selected_indices = (
                _selected_feature_indices(self.nameless_featurizer_info)
                if self.nameless_classifier is not None and self.nameless_featurizer_info is not None
                else None
            )
            batch_features, batch_nameless_features = _build_block_feature_matrices_indexed_rust(
                chunk.block_signature_indices,
                featurizer=rust_featurizer,
                start_offset=int(chunk.start_offset),
                max_pairs=int(len(chunk.labels)),
                main_indices=selected_indices,
                nameless_indices=nameless_selected_indices,
                num_threads=self.n_jobs,
            )
            batch_labels = np.asarray(chunk.labels, dtype=np.float64)
        except Exception as exc:
            if stage_uses_rust(runtime_context):
                raise RuntimeError(
                    "Rust fused block featurization failed in strict rust backend "
                    f"(block={chunk.block_key} start_offset={chunk.start_offset} pairs={len(chunk.labels)} "
                    f"run_id={runtime_context.run_id} error={exc})"
                ) from exc
            raise
        expected_rows = int(len(chunk.labels))
    else:
        signature_pairs = chunk.signature_pairs()
        batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
            signature_pairs,
            dataset,
            self.featurizer_info,
            n_jobs=self.n_jobs,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        expected_rows = int(len(signature_pairs))
    batch_predictions, batch_seconds = _predict_and_combine(
        self.classifier,
        self.nameless_classifier,
        batch_features,
        batch_labels,
        batch_nameless_features,
        batch_label,
        total_ram_bytes=total_ram_bytes,
    )
    if int(batch_predictions.shape[0]) != expected_rows:
        raise RuntimeError(
            f"Distance-matrix chunk prediction size mismatch: expected={expected_rows} got={batch_predictions.shape[0]}"
        )
    return np.asarray(batch_predictions, dtype=np.float64), float(batch_seconds)


def _iter_rust_predicted_distance_matrix_chunks(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    *,
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    pair_chunk_size: int | None = None,
    total_ram_bytes: int | None = None,
    prediction_state: PredictionState | None = None,
):
    chunk_count = 0
    helper_output = self._distance_matrix_chunk_helper_rust(
        block_dict,
        dataset,
        partial_supervision,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        pair_chunk_size=pair_chunk_size,
        prediction_state=prediction_state,
    )
    for chunk in helper_output:
        batch_predictions, batch_seconds = self._predict_distance_matrix_chunk(
            chunk,
            dataset,
            runtime_context,
            batch_label=f"chunk_{chunk_count}",
            total_ram_bytes=total_ram_bytes,
        )
        expected = int(len(chunk.labels))
        if int(batch_predictions.shape[0]) != expected:
            raise RuntimeError(
                f"Distance-matrix batch prediction count mismatch: expected={expected} got={batch_predictions.shape[0]}"
            )
        yield _PredictedDistanceMatrixChunk(
            chunk=chunk,
            predictions=np.asarray(batch_predictions, dtype=np.float64),
            batch_seconds=float(batch_seconds),
        )
        chunk_count += 1
    logger.info(
        "Telemetry: distance_matrix_chunking backend=rust chunks=%d run_id=%s",
        chunk_count,
        runtime_context.run_id,
    )


def _iter_python_predicted_distance_matrix_batches(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    *,
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    num_pairs: int,
    pair_chunk_size: int | None = None,
    total_ram_bytes: int | None = None,
    prediction_state: PredictionState | None = None,
):
    helper_output = self.distance_matrix_helper(
        block_dict,
        dataset,
        partial_supervision,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        pair_chunk_size=pair_chunk_size,
        prediction_state=prediction_state,
    )
    batch_num = 0
    effective_pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
    num_batches = math.ceil(num_pairs / effective_pair_chunk_size) if num_pairs > 0 else 0
    while True:
        logger.info(f"Featurizing batch {batch_num}/{num_batches}")
        count = 0
        pairs: list[tuple[str, str, float]] = []
        indices: list[tuple[int, int]] = []
        blocks: list[str] = []
        for item in helper_output:
            pairs.append(item[0])
            indices.append(item[1])
            blocks.append(item[2])
            count += 1
            if count == effective_pair_chunk_size:
                break

        if len(pairs) == 0:
            break

        batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
            pairs,
            dataset,
            self.featurizer_info,
            n_jobs=self.n_jobs,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        batch_predictions, batch_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            batch_features,
            batch_labels,
            batch_nameless_features,
            batch_num,
            total_ram_bytes=total_ram_bytes,
        )
        yield _PredictedDistanceMatrixBatch(
            batch_num=int(batch_num),
            blocks=blocks,
            indices=indices,
            predictions=np.asarray(batch_predictions, dtype=np.float64),
            batch_seconds=float(batch_seconds),
        )

        if count < effective_pair_chunk_size:
            break
        batch_num += 1


def _featurize_predict_write_batches(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    *,
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    num_pairs: int,
    write_prediction: Callable[[str, tuple[int, int], float], None],
    on_block_start: Callable[[str], None] | None = None,
    post_block_callback: Callable[[str], None] | None = None,
    disable_tqdm: bool = True,
    tqdm_desc: str = "Writing matrices",
    pair_chunk_size: int | None = None,
    total_ram_bytes: int | None = None,
    prediction_state: PredictionState | None = None,
) -> float:
    model_predict_seconds = 0.0
    prev_block_key = ""
    for batch in self._iter_python_predicted_distance_matrix_batches(
        block_dict,
        dataset,
        partial_supervision,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        num_pairs=num_pairs,
        pair_chunk_size=pair_chunk_size,
        total_ram_bytes=total_ram_bytes,
        prediction_state=prediction_state,
    ):
        model_predict_seconds += batch.batch_seconds
        batch_iter = tqdm(
            enumerate(batch.predictions),
            total=len(batch.predictions),
            desc=tqdm_desc,
            disable=disable_tqdm,
        )
        for within_batch_index, prediction in batch_iter:
            block_key = batch.blocks[within_batch_index]
            if block_key != prev_block_key:
                if prev_block_key != "" and post_block_callback is not None:
                    post_block_callback(prev_block_key)
                if on_block_start is not None:
                    on_block_start(block_key)
            write_prediction(block_key, batch.indices[within_batch_index], float(prediction))
            prev_block_key = block_key

    if prev_block_key != "" and post_block_callback is not None:
        post_block_callback(prev_block_key)
    return float(model_predict_seconds)


def make_distance_matrices(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    disable_tqdm: bool = False,
    incremental_dont_use_cluster_seeds: bool = False,
    total_ram_bytes: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Creates the distance matrices for the input blocks.
    Note: This function is much more complicated than it needs to be in an
    effort to reduce its memory footprint

    Parameters
    ----------
    block_dict: Dict
        the block dictionary to make distances for
    dataset: ANDData
        the dataset
    partial_supervision: Dict
        the dictionary of partial supervision provided with this dataset/these blocks
    disable_tqdm: bool
        whether to turn off the tqdm progress bars in this function
    incremental_dont_use_cluster_seeds: bool
        whether to ignore dataset cluster seeds while resolving constraints in incremental flows
    total_ram_bytes: int | None
        Optional RAM allowance for matrix allocation, pair-feature batches, and native scoring.

    Returns
    -------
    Dict: the distance matrix dictionary, keyed by block key. FastCluster
        condensed matrices use float64, matching streaming prediction.
    """
    partial_supervision = normalize_partial_supervision(partial_supervision)
    require_dataset_name_counts_binding_for_clusterer(
        self,
        dataset,
        context="Clusterer.make_distance_matrices",
    )
    runtime_context = dataset.runtime_context
    use_rust_blockwise = dataset_stage_uses_rust(runtime_context, dataset)
    if use_rust_blockwise:
        arrow_dataset = getattr(dataset, "arrow_dataset", None)
        if arrow_dataset is not None:
            _require_arrow_name_counts_index_for_clusterer(
                self,
                arrow_dataset,
                context="Clusterer.make_distance_matrices Arrow featurizer",
            )
    _ensure_lightgbm_fitted(self.classifier)
    _ensure_lightgbm_fitted(self.nameless_classifier)
    logger.info(f"Making {len(block_dict)} distance matrices")
    logger.info("Initializing pairwise_probas")
    pairwise_probas = {}
    num_pairs = 0
    # Calibration and cached prediction must retain the same precision as
    # streaming inference; float16 rounding can change threshold decisions.
    fastcluster_dtype = np.float64
    for block_key, signatures in block_dict.items():
        block_size = len(signatures)
        num_pairs += int(block_size * (block_size - 1) / 2)
        _guard_predict_block_matrix_allocation(
            block_key=block_key,
            block_size=int(block_size),
            uses_fastcluster=isinstance(self.cluster_model, FastCluster),
            fastcluster_fused_dtype=fastcluster_dtype,
            total_ram_bytes=total_ram_bytes,
            copies_codes=corrections_copy_codes(self),
        )
        if isinstance(self.cluster_model, FastCluster):
            # flattened pdist style
            pairwise_proba = np.zeros(int(block_size * (block_size - 1) / 2), dtype=fastcluster_dtype)
        else:
            pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
        pairwise_probas[block_key] = pairwise_proba

    logger.info(f"Pairwise probas initialized with {num_pairs} elements, starting making all pairs")

    model_predict_seconds = 0.0
    selected_count = _count_selected_features(self.featurizer_info)
    nameless_count = (
        _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
    )
    batch_chunk_plan = _compute_predict_batch_chunk_plan(
        self.featurizer_info.number_of_features,
        selected_feature_count=selected_count,
        nameless_feature_count=nameless_count,
        total_pairs=num_pairs,
        total_ram_bytes=total_ram_bytes,
    )
    pair_chunk_size = max(
        1,
        min(
            int(self.batch_size),
            int(batch_chunk_plan.chunk_pairs) if batch_chunk_plan is not None else int(self.batch_size),
        ),
    )
    if use_rust_blockwise:
        for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
        ):
            chunk = prediction_chunk.chunk
            batch_predictions = prediction_chunk.predictions
            model_predict_seconds += prediction_chunk.batch_seconds
            expected = int(len(chunk.labels))
            pairwise_proba = pairwise_probas[chunk.block_key]
            if isinstance(self.cluster_model, FastCluster):
                start = int(chunk.start_offset)
                end = start + expected
                pairwise_proba[start:end] = np.asarray(batch_predictions, dtype=np.float64)
            else:
                pairwise_proba[chunk.index_i, chunk.index_j] = np.asarray(
                    batch_predictions,
                    dtype=pairwise_proba.dtype,
                )
    else:
        fastcluster_write_indices: dict[str, int] = defaultdict(int)

        def _write_prediction(
            block_key: str,
            index_pair: tuple[int, int],
            prediction: float,
        ) -> None:
            pairwise_proba = pairwise_probas[block_key]
            if isinstance(self.cluster_model, FastCluster):
                write_index = fastcluster_write_indices[block_key]
                if write_index >= len(pairwise_proba):
                    raise RuntimeError(
                        "FastCluster pairwise probability write overflow: "
                        f"block={block_key} index={write_index} capacity={len(pairwise_proba)}"
                    )
                pairwise_proba[write_index] = prediction
                fastcluster_write_indices[block_key] = write_index + 1
            else:
                i, j = index_pair
                pairwise_proba[i, j] = prediction

        model_predict_seconds += self._featurize_predict_write_batches(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            num_pairs=num_pairs,
            write_prediction=_write_prediction,
            disable_tqdm=disable_tqdm,
            tqdm_desc="Writing matrices",
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
        )

        if isinstance(self.cluster_model, FastCluster):
            for block_key, pairwise_proba in pairwise_probas.items():
                expected_pairs = int(len(pairwise_proba))
                observed_pairs = int(fastcluster_write_indices.get(block_key, 0))
                if observed_pairs != expected_pairs:
                    raise RuntimeError(
                        "FastCluster pairwise probability fill mismatch: "
                        f"block={block_key} expected_pairs={expected_pairs} observed_pairs={observed_pairs}"
                    )

    if not isinstance(self.cluster_model, FastCluster):
        for pairwise_proba in pairwise_probas.values():
            pairwise_proba += pairwise_proba.T
            np.fill_diagonal(pairwise_proba, 0)

    logger.info(
        "Telemetry stage: stage=model_predict_total seconds=%.3f blocks=%d",
        model_predict_seconds,
        len(block_dict),
    )
    logger.info(f"{len(block_dict)} distance matrices made")
    return pairwise_probas


def make_distance_matrices_from_rust_featurizer(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    rust_featurizer: object,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    pair_chunk_size: int | None = None,
    total_ram_bytes: int | None = None,
) -> dict[str, np.ndarray]:
    """Build full-predict distance matrices from an existing Rust featurizer.

    This is the ANDData-free full-predict bridge for Arrow/native Rust
    inputs. It reuses the same Rust block upper-triangle feature and
    constraint APIs as `make_distance_matrices`, but takes the already-built
    featurizer directly instead of looking it up through an `ANDData`.
    """

    partial_supervision = normalize_partial_supervision(partial_supervision)
    require_rust_featurizer_name_counts_binding_for_clusterer(
        self,
        rust_featurizer,
        context="Clusterer.make_distance_matrices_from_rust_featurizer",
    )
    if self.featurizer_info.legacy_paper_years:
        rust_featurizer = cast(Any, rust_featurizer).with_legacy_paper_years()
    return self._make_distance_matrices_from_verified_rust_featurizer(
        block_dict,
        rust_featurizer,
        partial_supervision=partial_supervision,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        pair_chunk_size=pair_chunk_size,
        total_ram_bytes=total_ram_bytes,
    )


def _make_distance_matrices_from_verified_rust_featurizer(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    rust_featurizer: object,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    pair_chunk_size: int | None = None,
    total_ram_bytes: int | None = None,
    signature_index_by_id: Mapping[str, int] | None = None,
    prediction_state: PredictionState | None = None,
) -> dict[str, np.ndarray]:
    """Build matrices after the request boundary verified the featurizer binding."""

    if prediction_state is None:
        prediction_state = PredictionState()

    if runtime_context is None:
        runtime_context = build_runtime_context("cluster_predict_rust_featurizer", backend="rust")
    if partial_supervision is None:
        partial_supervision = {}
    _ensure_lightgbm_fitted(self.classifier)
    _ensure_lightgbm_fitted(self.nameless_classifier)

    if signature_index_by_id is None:
        signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
    selected_indices = _selected_feature_indices(self.featurizer_info)
    nameless_selected_indices = (
        _selected_feature_indices(self.nameless_featurizer_info)
        if self.nameless_classifier is not None and self.nameless_featurizer_info is not None
        else None
    )
    total_pairs = sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())
    selected_count = _count_selected_features(self.featurizer_info)
    nameless_count = (
        _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
    )
    batch_chunk_plan = _compute_predict_batch_chunk_plan(
        self.featurizer_info.number_of_features,
        selected_feature_count=selected_count,
        nameless_feature_count=nameless_count,
        total_pairs=total_pairs,
        total_ram_bytes=total_ram_bytes,
    )
    resolved_pair_chunk_size = max(
        1,
        min(
            int(self.batch_size),
            int(batch_chunk_plan.chunk_pairs) if batch_chunk_plan is not None else int(self.batch_size),
        ),
    )
    if pair_chunk_size is not None:
        resolved_pair_chunk_size = min(resolved_pair_chunk_size, max(1, int(pair_chunk_size)))

    uses_fastcluster = isinstance(self.cluster_model, FastCluster)
    fastcluster_dtype = np.float64 if uses_fastcluster else np.float16
    block_scorers = _native_block_scorers(self.classifier, self.nameless_classifier) if uses_fastcluster else None
    pairwise_probas: dict[str, np.ndarray] = {}
    model_predict_seconds = 0.0
    constraint_seconds = 0.0
    upper_triangle_index_seconds = 0.0
    label_build_seconds = 0.0
    feature_matrix_seconds = 0.0
    nameless_feature_matrix_seconds = 0.0
    matrix_write_seconds = 0.0
    chunk_count = 0
    total_start = time.perf_counter()
    for block_key, signatures in block_dict.items():
        block_size = int(len(signatures))
        block_pair_count = int(block_size * (block_size - 1) // 2)
        _guard_predict_block_matrix_allocation(
            block_key=block_key,
            block_size=block_size,
            uses_fastcluster=uses_fastcluster,
            fastcluster_fused_dtype=fastcluster_dtype,
            total_ram_bytes=total_ram_bytes,
            copies_codes=corrections_copy_codes(self),
        )
        if uses_fastcluster:
            pairwise_proba = np.zeros(block_pair_count, dtype=fastcluster_dtype)
        else:
            pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
        pairwise_probas[block_key] = pairwise_proba
        if block_size <= 1:
            continue

        block_signature_indices = [int(signature_index_by_id[str(signature_id)]) for signature_id in signatures]
        override_offsets, override_labels = _partial_supervision_label_overrides_for_block(
            signatures,
            partial_supervision,
        )
        if block_scorers is not None and uses_fastcluster and _ONE_PASS_BLOCK_FILL:
            # One native pass labels, scores and writes every pair, with the
            # chunked loop's values; it holds no chunk-sized buffers.
            stage_start = time.perf_counter()
            main_booster, nameless_booster = block_scorers
            cast(Any, rust_featurizer).fill_block_distances_indexed(
                block_signature_indices,
                pairwise_proba,
                LARGE_INTEGER,
                override_offsets,
                override_labels,
                selected_indices,
                main_booster,
                nameless_selected_indices,
                nameless_booster,
                use_constraints=bool(self.use_default_constraints_as_supervision),
                low_value=0.0,
                high_value=LARGE_DISTANCE,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                suppress_orcid=bool(getattr(self, "suppress_orcid", False)),
                num_threads=resolve_n_jobs(self.n_jobs),
                orcid_name_waiver=_orcid_name_waiver_setting(self),
                **_one_pass_tile_overrides,
            )
            model_predict_seconds += time.perf_counter() - stage_start
            chunk_count += 1
            continue
        offset = 0
        while offset < block_pair_count:
            chunk_pair_count = int(min(resolved_pair_chunk_size, block_pair_count - offset))
            chunk_count += 1
            local_i_array: np.ndarray | None = None
            local_j_array: np.ndarray | None = None
            constraint_labels: np.ndarray | None = None
            if self.use_default_constraints_as_supervision:
                stage_start = time.perf_counter()
                suppress_orcid = bool(getattr(self, "suppress_orcid", False))
                if uses_fastcluster:
                    constraint_labels = _get_constraint_labels_block_upper_triangle_indexed_rust(
                        block_signature_indices,
                        large_integer=LARGE_INTEGER,
                        start_offset=offset,
                        max_pairs=chunk_pair_count,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        num_threads=self.n_jobs,
                        featurizer=rust_featurizer,
                        suppress_orcid=suppress_orcid,
                        orcid_name_waiver=_orcid_name_waiver_setting(self),
                    )
                    constraint_seconds += time.perf_counter() - stage_start
                else:
                    local_i, local_j, constraint_values = get_constraints_block_upper_triangle_indexed_rust(
                        block_signature_indices,
                        start_offset=offset,
                        max_pairs=chunk_pair_count,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        num_threads=self.n_jobs,
                        featurizer=rust_featurizer,
                        suppress_orcid=suppress_orcid,
                        orcid_name_waiver=_orcid_name_waiver_setting(self),
                    )
                    constraint_seconds += time.perf_counter() - stage_start
                    stage_start = time.perf_counter()
                    local_i_array = np.asarray(local_i, dtype=np.intp)
                    local_j_array = np.asarray(local_j, dtype=np.intp)
                    upper_triangle_index_seconds += time.perf_counter() - stage_start
                    # NumPy reads None as NaN, which marks a pair without a constraint.
                    constraint_labels = np.asarray(constraint_values, dtype=np.float64) - LARGE_INTEGER
                    del local_i, local_j, constraint_values
                label_count = int(len(constraint_labels))
                if label_count != chunk_pair_count:
                    raise RuntimeError(
                        "Rust constraint row count mismatch for vector write: "
                        f"block={block_key} start_offset={offset} "
                        f"expected={chunk_pair_count} got={label_count}"
                    )
            elif uses_fastcluster:
                label_count = chunk_pair_count
            else:
                stage_start = time.perf_counter()
                local_i_array, local_j_array = _upper_triangle_indices_for_range(
                    block_size,
                    offset,
                    chunk_pair_count,
                )
                upper_triangle_index_seconds += time.perf_counter() - stage_start
                label_count = int(len(local_i_array))

            stage_start = time.perf_counter()
            if constraint_labels is None:
                labels = np.full(label_count, np.nan, dtype=np.float64)
            else:
                labels = constraint_labels
            # Partial supervision overrides the constraint label of its pairs.
            first, stop = np.searchsorted(override_offsets, (offset, offset + label_count))
            if first < stop:
                labels[override_offsets[first:stop] - offset] = override_labels[first:stop]
            label_build_seconds += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            scored_rows = np.flatnonzero(np.isnan(labels))
            if block_scorers is not None:
                # Rust featurizes and scores each tile of pairs in one pass,
                # so no chunk-sized feature matrix exists.
                main_booster, nameless_booster = block_scorers
                scored_probabilities = cast(Any, rust_featurizer).score_block_upper_triangle_rows_indexed(
                    block_signature_indices,
                    offset,
                    chunk_pair_count,
                    None if len(scored_rows) == label_count else scored_rows.astype(np.int64, copy=False),
                    selected_indices,
                    main_booster,
                    nameless_selected_indices,
                    nameless_booster,
                    resolve_n_jobs(self.n_jobs),
                )
                batch_predictions = np.zeros(label_count, dtype=np.float64)
                batch_predictions[scored_rows] = scored_probabilities
                not_scored = ~np.isnan(labels)
                # The LARGE_INTEGER offset of staged labels is undone, as in
                # _predict_and_combine.
                batch_predictions[not_scored] = labels[not_scored] + LARGE_INTEGER
                model_predict_seconds += time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                end = offset + int(len(batch_predictions))
                pairwise_proba[offset:end] = batch_predictions
                matrix_write_seconds += time.perf_counter() - stage_start
                offset += chunk_pair_count
                del batch_predictions, scored_probabilities, not_scored, constraint_labels, labels, scored_rows
                continue
            # Use indexed pairs only when supervision resolves a majority;
            # otherwise retain the cheaper contiguous triangle traversal.
            if len(scored_rows) * 2 >= label_count:
                scored_rows = None
            batch_features, batch_nameless_features = _build_block_feature_matrices_indexed_rust(
                block_signature_indices,
                featurizer=rust_featurizer,
                start_offset=offset,
                max_pairs=chunk_pair_count,
                main_indices=selected_indices,
                nameless_indices=nameless_selected_indices,
                num_threads=self.n_jobs,
                scored_rows=scored_rows,
                compact=True,
            )
            feature_matrix_seconds += time.perf_counter() - stage_start
            batch_predictions, batch_seconds = _predict_and_combine(
                self.classifier,
                self.nameless_classifier,
                batch_features,
                labels,
                batch_nameless_features,
                f"{block_key}:{offset}",
                total_ram_bytes=total_ram_bytes,
                compact_rows=scored_rows,
            )
            model_predict_seconds += float(batch_seconds)
            stage_start = time.perf_counter()
            if uses_fastcluster:
                end = offset + int(len(batch_predictions))
                pairwise_proba[offset:end] = np.asarray(batch_predictions, dtype=pairwise_proba.dtype)
            else:
                if local_i_array is None or local_j_array is None:
                    raise RuntimeError("Non-FastCluster matrix write requires upper-triangle index arrays")
                pairwise_proba[local_i_array, local_j_array] = np.asarray(
                    batch_predictions,
                    dtype=pairwise_proba.dtype,
                )
            matrix_write_seconds += time.perf_counter() - stage_start
            offset += chunk_pair_count
            # Chunk sizing assumes one feature payload at a time. Release
            # every consumed native/Python buffer before the next builder
            # allocates its right-hand side.
            del (
                batch_features,
                batch_nameless_features,
                batch_predictions,
                constraint_labels,
                labels,
                scored_rows,
                local_i_array,
                local_j_array,
            )

        if not uses_fastcluster:
            pairwise_proba += pairwise_proba.T
            np.fill_diagonal(pairwise_proba, 0)

    logger.info(
        "Telemetry stage: stage=model_predict_rust_featurizer_total seconds=%.3f blocks=%d",
        model_predict_seconds,
        len(block_dict),
    )
    telemetry = {
        "total_seconds": float(time.perf_counter() - total_start),
        "constraint_seconds": float(constraint_seconds),
        "upper_triangle_index_seconds": float(upper_triangle_index_seconds),
        "label_build_seconds": float(label_build_seconds),
        "feature_matrix_seconds": float(feature_matrix_seconds),
        "nameless_feature_matrix_seconds": float(nameless_feature_matrix_seconds),
        "model_predict_seconds": float(model_predict_seconds),
        "matrix_write_seconds": float(matrix_write_seconds),
        "block_count": int(len(block_dict)),
        "pair_count": int(total_pairs),
        "chunk_count": int(chunk_count),
        "resolved_pair_chunk_size": int(resolved_pair_chunk_size),
    }
    prediction_state.telemetry["rust_featurizer_make_dists"] = telemetry
    logger.info(
        "Telemetry stage: stage=rust_featurizer_make_dists total_seconds=%.3f "
        "constraint_seconds=%.3f upper_triangle_index_seconds=%.3f "
        "label_build_seconds=%.3f feature_matrix_seconds=%.3f "
        "nameless_feature_matrix_seconds=%.3f model_predict_seconds=%.3f "
        "matrix_write_seconds=%.3f blocks=%d pairs=%d chunks=%d pair_chunk_size=%d",
        telemetry["total_seconds"],
        telemetry["constraint_seconds"],
        telemetry["upper_triangle_index_seconds"],
        telemetry["label_build_seconds"],
        telemetry["feature_matrix_seconds"],
        telemetry["nameless_feature_matrix_seconds"],
        telemetry["model_predict_seconds"],
        telemetry["matrix_write_seconds"],
        telemetry["block_count"],
        telemetry["pair_count"],
        telemetry["chunk_count"],
        telemetry["resolved_pair_chunk_size"],
    )
    return pairwise_probas
