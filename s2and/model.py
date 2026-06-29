from __future__ import annotations

import copy
import logging
import math
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self, TypeVar

import lightgbm as lgb
import numpy as np
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.base import clone
from sklearn.exceptions import EfficiencyWarning
from tqdm import tqdm

from s2and import memory_budget
from s2and.consts import _PACKAGE_DATA_DIR, DEFAULT_CHUNK_SIZE, LARGE_INTEGER
from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
    ANDData,
)
from s2and.eval import b3_precision_recall_fscore
from s2and.feature_port import _get_rust_featurizer, evict_rust_featurizer
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize, resolve_cache_policy
from s2and.incremental_linking.production import predict_incremental_promoted_linker
from s2and.model_pairwise import FastCluster, PairwiseModeler, VotingClassifier, intify
from s2and.runtime import RuntimeContext, build_runtime_context, stage_uses_rust
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust,
    get_constraint_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    get_constraints_matrix_indexed_rust,
    update_rust_cluster_seeds,
)
from s2and.subblocking import make_subblocks
from s2and.text import first_names_name_compatible
from s2and.thread_config import resolve_n_jobs
from s2and.warnings_utils import suppress_sklearn_feature_name_warnings

logger = logging.getLogger("s2and")
IncrementalPhaseBMode = Literal["exact"]
IncrementalBroadcastMode = Literal["always", "never", "top1_consensus"]
IncrementalSeedScoreMode = Literal["mean", "min", "mean_min_hybrid"]
IncrementalDistStats = tuple[float, int, float]
_TReturn = TypeVar("_TReturn")
_MISSING = object()
DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR = Path(_PACKAGE_DATA_DIR) / "production_model_v1.21" / "incremental_linker"

# Keep canonical pickle import paths stable after splitting module internals.
for _export in (FastCluster, PairwiseModeler, VotingClassifier, intify):
    _export.__module__ = __name__


def _build_incremental_result(
    clusters: dict[str, list[str]],
    *,
    phase_b_mode: IncrementalPhaseBMode,
    phase_b_budget_bytes: int,
    phase_b_required_bytes: int,
    phase_b_residual_count: int | None = None,
) -> dict[str, Any]:
    result = {
        "clusters": clusters,
        "phase_b_mode": phase_b_mode,
        "phase_b_budget_bytes": int(phase_b_budget_bytes),
        "phase_b_required_bytes": int(phase_b_required_bytes),
    }
    if phase_b_residual_count is not None:
        result["phase_b_residual_count"] = int(phase_b_residual_count)
    return result


def _resolve_total_ram_bytes_for_incremental(total_ram_bytes: int | None = None) -> tuple[int, str]:
    return memory_budget.resolve_total_ram_bytes(
        total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
    )


def _count_selected_features(featurizer_info: FeaturizationInfo) -> int:
    """Count the number of feature indices selected by features_to_use."""
    return len(_selected_feature_indices(featurizer_info))


def _selected_feature_indices(featurizer_info: FeaturizationInfo) -> list[int]:
    indices: set[int] = set()
    for feature_name in featurizer_info.features_to_use:
        indices.update(featurizer_info.feature_group_to_index[feature_name])
    return sorted(indices)


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
) -> None:
    """Fail fast before exact prediction allocates a block matrix above the RAM budget."""
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
    if matrix_bytes <= int(snapshot.available_bytes):
        return
    raise MemoryError(
        "Predict exact block exceeds memory budget before matrix allocation: "
        f"block={block_key} block_size={block_size} matrix_bytes={matrix_bytes} "
        f"available_bytes={int(snapshot.available_bytes)} total_ram_bytes={int(snapshot.total_ram_bytes)} "
        f"current_rss_bytes={int(snapshot.current_rss_bytes)} total_ram_source={snapshot.total_ram_source} "
        f"current_rss_source={snapshot.current_rss_source}"
    )


def _signature_first_for_rules(signature: Any) -> str:
    return signature.author_info_first_normalized_without_apostrophe or signature.author_info_first or ""


def _next_unused_cluster_id(pred_clusters: dict[str, Any], start: int) -> int:
    cluster_id = int(start)
    while str(cluster_id) in pred_clusters:
        cluster_id += 1
    return cluster_id


def _ensure_lightgbm_fitted(clf: Any) -> None:
    if clf is None:
        return
    inner = getattr(clf, "classifier", None)
    if inner is not None and inner is not clf:
        _ensure_lightgbm_fitted(inner)
    if not hasattr(lgb, "LGBMModel") or not isinstance(clf, lgb.LGBMModel):
        return
    booster = getattr(clf, "_Booster", None)
    if booster is None:
        raise RuntimeError(
            "LightGBM estimator has no fitted booster (_Booster is None); " "fit the estimator before prediction."
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


def _propagate_n_jobs(estimator: Any, n_jobs: int) -> None:
    """Best-effort propagation of `n_jobs` into estimators/wrappers.

    Keeps S2AND's `Clusterer.n_jobs` as the single knob for both Rust `num_threads` and
    Python model inference thread pools (LightGBM/OpenMP, sklearn estimators, etc.).
    """
    if estimator is None:
        return

    inner = getattr(estimator, "classifier", None)
    if inner is not None and inner is not estimator:
        _propagate_n_jobs(inner, n_jobs)

    for attr in ("estimators", "estimators_"):
        children = getattr(estimator, attr, None)
        if not isinstance(children, list | tuple):
            continue
        for child in children:
            if isinstance(child, tuple) and len(child) == 2:
                _propagate_n_jobs(child[1], n_jobs)
            else:
                _propagate_n_jobs(child, n_jobs)

    resolved_n_jobs = resolve_n_jobs(n_jobs)
    set_params = getattr(estimator, "set_params", None)
    if callable(set_params):
        try:
            set_params(n_jobs=resolved_n_jobs)
        except TypeError as exc:
            logger.debug(
                "Skipping set_params n_jobs propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via set_params for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise

    if hasattr(estimator, "n_jobs"):
        try:
            estimator.n_jobs = resolved_n_jobs
        except (AttributeError, TypeError) as exc:
            logger.debug(
                "Skipping n_jobs attribute propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via attribute assignment for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise


def _name_count_semantics_from_featurizer_version(
    featurizer_version: int | None,
) -> str | None:
    if not isinstance(featurizer_version, int):
        return None
    return NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR


def _resolve_clusterer_name_count_semantics(
    clusterer: Any,
) -> str:
    contract = getattr(clusterer, "feature_contract", None)
    if isinstance(contract, dict):
        contract_value = contract.get("name_counts_last_first_initial_semantics")
        if contract_value in {
            NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
            NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
        }:
            return str(contract_value)
        if contract_value is not None:
            raise ValueError(
                "Invalid clusterer feature_contract['name_counts_last_first_initial_semantics'] "
                f"value: {contract_value!r}"
            )

    featurizer_info = getattr(clusterer, "featurizer_info", None)
    featurizer_version = getattr(featurizer_info, "featurizer_version", None)
    inferred = _name_count_semantics_from_featurizer_version(featurizer_version)
    if inferred is not None:
        return inferred

    raise ValueError(
        "Unable to resolve model name-count semantics from feature_contract or featurizer_version. "
        "Inference requires explicit semantics metadata."
    )


def _apply_dataset_name_count_semantics_for_prediction(
    clusterer: Any,
    dataset: ANDData,
) -> None:
    desired = _resolve_clusterer_name_count_semantics(clusterer)
    dataset.set_name_counts_last_first_initial_semantics(desired)


def _predict_class0_with_runtime(
    classifier: Any,
    features: np.ndarray,
    *,
    num_threads: int | None = None,
) -> tuple[np.ndarray, float, str]:
    features_2d = np.asarray(features, dtype=np.float64, order="C")
    if features_2d.size == 0:
        return np.asarray([], dtype=np.float64), 0.0, "none"

    # Estimator threading is configured through propagated n_jobs; predict_proba(num_threads=...)
    # is LightGBM-specific and breaks sklearn-compatible wrappers.
    del num_threads

    python_start = time.perf_counter()
    with warnings.catch_warnings():
        suppress_sklearn_feature_name_warnings()
        predictions = classifier.predict_proba(features_2d)[:, 0]
    return predictions, time.perf_counter() - python_start, "python"


def _predict_and_combine(
    classifier: Any,
    nameless_classifier: Any | None,
    features: np.ndarray,
    labels: np.ndarray,
    nameless_features: np.ndarray | None,
    batch_label: int | str,
    *,
    num_threads: int | None = None,
    runtime_context: RuntimeContext | None = None,
) -> tuple[np.ndarray, float]:
    """Predict with main (and optional nameless) classifier, log telemetry, return (predictions, seconds)."""
    row_count = int(features.shape[0])
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
            classifier, main_matrix, num_threads=num_threads
        )
        if nameless_classifier is not None:
            if nameless_matrix is None:
                raise RuntimeError("nameless_classifier is configured but nameless feature matrix is missing")
            nl_pred, nl_sec, nl_backend = _predict_class0_with_runtime(
                nameless_classifier, nameless_matrix, num_threads=num_threads
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
    # features matrix) when most rows are predicted. Avoid that peak by predicting
    # on the full matrix in that case and overriding constrained rows afterwards.
    copy_avoid_threshold_bytes = 128 * (1 << 20)
    would_copy_bytes = 0
    if predicted_rows > 0 and predicted_rows < row_count:
        would_copy_bytes += int(predicted_rows) * int(features.shape[1]) * int(features.dtype.itemsize)
        if nameless_classifier is not None and nameless_features is not None:
            would_copy_bytes += (
                int(predicted_rows) * int(nameless_features.shape[1]) * int(nameless_features.dtype.itemsize)
            )

    predict_on_full_matrix = predicted_rows > 0 and (
        predicted_rows == row_count or would_copy_bytes >= int(copy_avoid_threshold_bytes)
    )

    if predicted_rows > 0:
        if predict_on_full_matrix:
            combined_predictions, batch_seconds = _predict_rows(
                features,
                nameless_features,
                row_total=predicted_rows,
            )
            predictions[:] = combined_predictions
            seconds += batch_seconds
        else:
            predict_features = features[predict_flag, :]
            predict_nameless_features = nameless_features[predict_flag, :] if nameless_features is not None else None
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


def _use_rust_constraints(runtime_context: RuntimeContext | None = None) -> bool:
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints")
    return stage_uses_rust(runtime_context)


def _handle_optional_rust_exception(
    runtime_context: RuntimeContext,
    *,
    strict_message: str,
    exc: Exception,
    python_path_warning: str,
    context_fields: tuple[str, ...] = (),
) -> None:
    details = " ".join((*context_fields, f"run_id={runtime_context.run_id}", f"error={exc}"))
    if stage_uses_rust(runtime_context):
        raise RuntimeError(f"{strict_message} ({details})") from exc
    logger.warning("%s: %s", python_path_warning, exc)


def _optional_rust_or_python_path(
    fn: Callable[[], _TReturn],
    python_fn: Callable[[], _TReturn],
    *,
    runtime_context: RuntimeContext,
    label: str,
    context_fields: tuple[str, ...] = (),
    strict_message: str | None = None,
    python_path_warning: str | None = None,
) -> _TReturn:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - native extension optional
        _handle_optional_rust_exception(
            runtime_context,
            strict_message=(strict_message or f"Rust {label} failed in strict rust backend"),
            exc=exc,
            python_path_warning=(
                python_path_warning
                or f"Optional Rust {label} failed while runtime backend is Python; using Python path"
            ),
            context_fields=context_fields,
        )
        return python_fn()


def _cluster_seeds_version(dataset: ANDData) -> int:
    return int(getattr(dataset, "_cluster_seeds_version", 0))


def _bump_cluster_seeds_version(dataset: ANDData) -> int:
    next_version = _cluster_seeds_version(dataset) + 1
    dataset._cluster_seeds_version = next_version
    return next_version


class _VersionedClusterSeedDict(dict[Any, Any]):
    def __init__(self, *args: Any, on_mutation: Callable[[], None] | None = None, **kwargs: Any) -> None:
        self._on_mutation = on_mutation
        super().__init__(*args, **kwargs)

    def set_on_mutation(self, on_mutation: Callable[[], None] | None) -> None:
        self._on_mutation = on_mutation

    def _mark_mutated(self) -> None:
        callback = self._on_mutation
        if callback is not None:
            callback()

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, value)
        self._mark_mutated()

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(key)
        self._mark_mutated()

    def clear(self) -> None:
        if self:
            super().clear()
            self._mark_mutated()

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if key in self:
            value = super().pop(key)
            self._mark_mutated()
            return value
        if default is not _MISSING:
            return default
        raise KeyError(key)

    def popitem(self) -> tuple[Any, Any]:
        value = super().popitem()
        self._mark_mutated()
        return value

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return self[key]
        super().__setitem__(key, default)
        self._mark_mutated()
        return default

    def update(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            super().update(*args, **kwargs)
            self._mark_mutated()

    def __ior__(self, value: Any) -> Self:
        super().__ior__(value)
        self._mark_mutated()
        return self


class _VersionedClusterSeedSet(set[tuple[Any, Any]]):
    def __init__(self, *args: Any, on_mutation: Callable[[], None] | None = None) -> None:
        self._on_mutation = on_mutation
        super().__init__(*args)

    def set_on_mutation(self, on_mutation: Callable[[], None] | None) -> None:
        self._on_mutation = on_mutation

    def _mark_mutated(self) -> None:
        callback = self._on_mutation
        if callback is not None:
            callback()

    def add(self, element: tuple[Any, Any]) -> None:
        super().add(element)
        self._mark_mutated()

    def remove(self, element: tuple[Any, Any]) -> None:
        super().remove(element)
        self._mark_mutated()

    def discard(self, element: tuple[Any, Any]) -> None:
        if element in self:
            super().discard(element)
            self._mark_mutated()

    def pop(self) -> tuple[Any, Any]:
        value = super().pop()
        self._mark_mutated()
        return value

    def clear(self) -> None:
        if self:
            super().clear()
            self._mark_mutated()

    def update(self, *others: Any) -> None:
        if others:
            super().update(*others)
            self._mark_mutated()

    def difference_update(self, *others: Any) -> None:
        if others:
            super().difference_update(*others)
            self._mark_mutated()

    def intersection_update(self, *others: Any) -> None:
        if others:
            super().intersection_update(*others)
            self._mark_mutated()

    def symmetric_difference_update(self, other: Any) -> None:
        super().symmetric_difference_update(other)
        self._mark_mutated()

    def __ior__(self, value: Any) -> Self:
        super().__ior__(value)
        self._mark_mutated()
        return self

    def __iand__(self, value: Any) -> Self:
        super().__iand__(value)
        self._mark_mutated()
        return self

    def __isub__(self, value: Any) -> Self:
        super().__isub__(value)
        self._mark_mutated()
        return self

    def __ixor__(self, value: Any) -> Self:
        super().__ixor__(value)
        self._mark_mutated()
        return self


def _ensure_cluster_seed_version_tracking(dataset: ANDData) -> None:
    def _mark_mutated() -> None:
        _bump_cluster_seeds_version(dataset)

    require = getattr(dataset, "cluster_seeds_require", {})
    if require is None:
        require = {}
    if isinstance(require, _VersionedClusterSeedDict):
        require.set_on_mutation(_mark_mutated)
    else:
        dataset.cluster_seeds_require = _VersionedClusterSeedDict(require, on_mutation=_mark_mutated)

    disallow = getattr(dataset, "cluster_seeds_disallow", set())
    if disallow is None:
        disallow = set()
    if isinstance(disallow, _VersionedClusterSeedSet):
        disallow.set_on_mutation(_mark_mutated)
    else:
        dataset.cluster_seeds_disallow = _VersionedClusterSeedSet(disallow, on_mutation=_mark_mutated)


@dataclass(frozen=True)
class _ConstraintPolicy:
    """Resolved hard-constraint behavior for one constraint evaluation boundary."""

    dont_merge_cluster_seeds: bool = True
    incremental_dont_use_cluster_seeds: bool = False
    suppress_orcid: bool = False


def _get_constraint_value(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    rust_featurizer: object | None = None,
    use_rust_constraints: bool | None = None,
    runtime_context: RuntimeContext | None = None,
    suppress_orcid: bool = False,
):
    policy = _ConstraintPolicy(
        dont_merge_cluster_seeds=dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        suppress_orcid=suppress_orcid,
    )
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints")
    if use_rust_constraints is None:
        use_rust_constraints = _use_rust_constraints(runtime_context)
    if use_rust_constraints:
        return _optional_rust_or_python_path(
            fn=lambda: get_constraint_rust(
                dataset,
                sig_id_1,
                sig_id_2,
                dont_merge_cluster_seeds=policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=policy.incremental_dont_use_cluster_seeds,
                featurizer=rust_featurizer,
                runtime_context=runtime_context,
                suppress_orcid=policy.suppress_orcid,
            ),
            python_fn=lambda: dataset.get_constraint(
                sig_id_1,
                sig_id_2,
                dont_merge_cluster_seeds=policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=policy.incremental_dont_use_cluster_seeds,
                suppress_orcid=policy.suppress_orcid,
            ),
            runtime_context=runtime_context,
            label="constraint evaluation",
            strict_message="Rust constraint evaluation failed in strict rust backend",
            python_path_warning=(
                "Optional Rust get_constraint failed while runtime backend is Python; using Python constraint path"
            ),
            context_fields=(f"pair=({sig_id_1}, {sig_id_2})",),
        )
    return dataset.get_constraint(
        sig_id_1,
        sig_id_2,
        dont_merge_cluster_seeds=policy.dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds=policy.incremental_dont_use_cluster_seeds,
        suppress_orcid=policy.suppress_orcid,
    )


def _sync_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: RuntimeContext | None = None,
) -> None:
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints")
    if _use_rust_constraints(runtime_context):
        # Best-effort instrumentation for subblocking lifecycle overhead.
        # Stored on the dataset to avoid changing return payloads on hot paths.
        dataset._rust_cluster_seeds_sync_calls = int(getattr(dataset, "_rust_cluster_seeds_sync_calls", 0)) + 1

        _ensure_cluster_seed_version_tracking(dataset)
        seed_version = _cluster_seeds_version(dataset)
        require = getattr(dataset, "cluster_seeds_require", {})
        disallow = getattr(dataset, "cluster_seeds_disallow", set())
        require_id = int(id(require))
        disallow_id = int(id(disallow))
        require_len = int(len(require))
        disallow_len = int(len(disallow))

        last_synced = getattr(dataset, "_rust_cluster_seeds_synced_version", None)
        last_require_id = getattr(dataset, "_rust_cluster_seeds_require_id", None)
        last_disallow_id = getattr(dataset, "_rust_cluster_seeds_disallow_id", None)
        last_require_len = getattr(dataset, "_rust_cluster_seeds_require_len", None)
        last_disallow_len = getattr(dataset, "_rust_cluster_seeds_disallow_len", None)
        if (
            last_synced == seed_version
            and last_require_id == require_id
            and last_require_len == require_len
            and last_disallow_id == disallow_id
            and last_disallow_len == disallow_len
        ):
            dataset._rust_cluster_seeds_sync_skipped_unchanged = (
                int(getattr(dataset, "_rust_cluster_seeds_sync_skipped_unchanged", 0)) + 1
            )
            return

        dataset._rust_cluster_seeds_sync_attempted = int(getattr(dataset, "_rust_cluster_seeds_sync_attempted", 0)) + 1

        def _sync() -> None:
            sync_start = time.perf_counter()
            update_rust_cluster_seeds(dataset, runtime_context=runtime_context)
            sync_seconds = float(time.perf_counter() - sync_start)
            dataset._rust_cluster_seeds_sync_succeeded = (
                int(getattr(dataset, "_rust_cluster_seeds_sync_succeeded", 0)) + 1
            )
            dataset._rust_cluster_seeds_sync_seconds_total = (
                float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_total", 0.0)) + sync_seconds
            )
            dataset._rust_cluster_seeds_sync_seconds_max = max(
                float(getattr(dataset, "_rust_cluster_seeds_sync_seconds_max", 0.0)),
                sync_seconds,
            )
            dataset._rust_cluster_seeds_synced_version = seed_version
            dataset._rust_cluster_seeds_require_id = require_id
            dataset._rust_cluster_seeds_require_len = require_len
            dataset._rust_cluster_seeds_disallow_id = disallow_id
            dataset._rust_cluster_seeds_disallow_len = disallow_len

        _optional_rust_or_python_path(
            fn=_sync,
            python_fn=lambda: None,
            runtime_context=runtime_context,
            label="cluster seed sync",
            strict_message="Rust cluster seed sync failed in strict rust backend",
            python_path_warning=(
                "Optional Rust cluster seed sync failed while runtime backend is Python; using Python seed state"
            ),
        )


def _initialize_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
) -> tuple[object | None, bool | None]:
    if not use_default_constraints_as_supervision:
        return None, None

    use_rust_constraints = _use_rust_constraints(runtime_context)
    if not use_rust_constraints:
        return None, False

    rust_featurizer = _optional_rust_or_python_path(
        fn=lambda: _get_rust_featurizer(dataset, runtime_context=runtime_context),
        python_fn=lambda: None,
        runtime_context=runtime_context,
        label="constraint featurizer init",
        strict_message="Rust constraint stage requested but Rust featurizer init failed",
        python_path_warning=(
            "Optional Rust featurizer init failed while runtime backend is Python; using Python constraints"
        ),
    )
    if rust_featurizer is None:
        return None, False

    return rust_featurizer, True


def _resolve_constraint_api_mode(
    rust_featurizer: object | None,
    use_rust_constraints: bool | None,
) -> str:
    if not use_rust_constraints or rust_featurizer is None:
        return "python"
    return "indexed"


def _build_signature_index_by_id(rust_featurizer: object) -> dict[str, int]:
    signature_ids = rust_featurizer.signature_ids()  # type: ignore[attr-defined]
    return {str(sig_id): idx for idx, sig_id in enumerate(signature_ids)}


def _build_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
    suppress_orcid: bool = False,
) -> _IncrementalConstraintBackend:
    """Build Phase A constraint backend state once for reuse across subblocks."""
    rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        runtime_context=runtime_context,
    )
    constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    signature_index_by_id: dict[str, int] | None = None
    if constraint_api_mode == "indexed" and rust_featurizer is not None:
        signature_index_by_id = _optional_rust_or_python_path(
            fn=lambda: _build_signature_index_by_id(rust_featurizer),
            python_fn=lambda: None,
            runtime_context=runtime_context,
            label="indexed constraint setup",
            strict_message="Rust indexed constraint setup failed in strict rust backend",
            python_path_warning=(
                "Optional Rust indexed constraint setup failed while runtime backend is Python; "
                "using Python constraints"
            ),
        )
        if signature_index_by_id is None:
            use_rust_constraints = False
            rust_featurizer = None
            constraint_api_mode = "python"
    return _IncrementalConstraintBackend(
        rust_featurizer=rust_featurizer,
        use_rust_constraints=use_rust_constraints,
        constraint_api_mode=constraint_api_mode,
        signature_index_by_id=signature_index_by_id,
        suppress_orcid=suppress_orcid,
    )


def _resolve_constraint_labels_batch(
    dataset: ANDData,
    pair_ids: list[tuple[str, str]],
    *,
    constraint_backend: _IncrementalConstraintBackend | None = None,
    partial_supervision: dict[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    constraint_policy: _ConstraintPolicy,
    rust_featurizer: object | None = None,
    use_rust_constraints: bool | None = None,
    runtime_context: RuntimeContext,
    num_threads: int | None = None,
    constraint_api_mode: str | None = None,
    signature_index_by_id: dict[str, int] | None = None,
) -> tuple[list[float], _ConstraintBatchTelemetry]:
    if constraint_backend is not None:
        rust_featurizer = constraint_backend.rust_featurizer
        use_rust_constraints = constraint_backend.use_rust_constraints
        constraint_policy = _ConstraintPolicy(
            dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
            suppress_orcid=constraint_backend.suppress_orcid,
        )
        if constraint_api_mode is None:
            constraint_api_mode = constraint_backend.constraint_api_mode
        if signature_index_by_id is None:
            signature_index_by_id = constraint_backend.signature_index_by_id

    labels: list[float] = [float(np.nan)] * len(pair_ids)
    unresolved_pairs: list[tuple[str, str]] = []
    unresolved_indices: list[int] = []
    partial_hits = 0
    for idx, (sig_id_1, sig_id_2) in enumerate(pair_ids):
        if (sig_id_1, sig_id_2) in partial_supervision:
            # Subtract LARGE_INTEGER so downstream featurization knows not to recompute these constraints.
            labels[idx] = float(partial_supervision[(sig_id_1, sig_id_2)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        if (sig_id_2, sig_id_1) in partial_supervision:
            # Subtract LARGE_INTEGER so downstream featurization knows not to recompute these constraints.
            labels[idx] = float(partial_supervision[(sig_id_2, sig_id_1)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        unresolved_pairs.append((sig_id_1, sig_id_2))
        unresolved_indices.append(idx)

    mode = constraint_api_mode or _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    telemetry = _ConstraintBatchTelemetry(
        total_pairs=int(len(pair_ids)),
        partial_supervision_hits=int(partial_hits),
        unresolved_pairs=int(len(unresolved_pairs)),
        rust_batch_call_count=0,
        api_mode=mode,
        elapsed_seconds=0.0,
    )
    if not unresolved_pairs or not use_default_constraints_as_supervision:
        if not use_default_constraints_as_supervision:
            telemetry.api_mode = "partial_only"
        return labels, telemetry

    start = time.perf_counter()
    values: list[float | None]

    def _resolve_values_python() -> list[float | None]:
        return [
            dataset.get_constraint(
                s1,
                s2,
                dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
                suppress_orcid=constraint_policy.suppress_orcid,
            )
            for s1, s2 in unresolved_pairs
        ]

    if use_rust_constraints and rust_featurizer is not None and mode == "indexed":

        def _resolve_values_rust() -> list[float | None]:
            if signature_index_by_id is None:
                raise RuntimeError("Indexed constraint API requested without signature index lookup")
            indexed_pairs = [(signature_index_by_id[s1], signature_index_by_id[s2]) for s1, s2 in unresolved_pairs]
            return get_constraints_matrix_indexed_rust(
                dataset,
                indexed_pairs,
                dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
                num_threads=num_threads,
                featurizer=rust_featurizer,
                runtime_context=runtime_context,
                suppress_orcid=constraint_policy.suppress_orcid,
            )

        used_python_path_after_optional_rust_failure = False

        def _resolve_values_python_after_optional_rust_failure() -> list[float | None]:
            nonlocal used_python_path_after_optional_rust_failure
            used_python_path_after_optional_rust_failure = True
            return _resolve_values_python()

        values = _optional_rust_or_python_path(
            fn=_resolve_values_rust,
            python_fn=_resolve_values_python_after_optional_rust_failure,
            runtime_context=runtime_context,
            label="batch constraint evaluation",
            strict_message="Rust batch constraint evaluation failed in strict rust backend",
            python_path_warning=(
                "Optional Rust batch constraint evaluation failed while runtime backend is Python; "
                "using Python constraints"
            ),
            context_fields=(f"pairs={len(unresolved_pairs)}",),
        )
        if used_python_path_after_optional_rust_failure:
            telemetry.api_mode = "optional_rust_failed_python"
            telemetry.rust_batch_call_count = 0
        else:
            telemetry.rust_batch_call_count = 1
    else:
        values = _resolve_values_python()
        telemetry.api_mode = "python"

    telemetry.elapsed_seconds = float(time.perf_counter() - start)
    for idx, value in zip(unresolved_indices, values, strict=True):
        if value is None:
            labels[idx] = float(np.nan)
        else:
            # Keep partial/constraint labels in the LARGE_INTEGER-offset convention.
            labels[idx] = float(value - LARGE_INTEGER)
    return labels, telemetry


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


@dataclass
class _ConstraintTelemetryAccumulator:
    total_pairs: int = 0
    partial_supervision_hits: int = 0
    unresolved_pairs: int = 0
    rust_batch_call_count: int = 0
    elapsed_seconds: float = 0.0
    api_modes: set[str] = field(default_factory=set)

    @property
    def api_mode_summary(self) -> str:
        return ",".join(sorted(self.api_modes)) if self.api_modes else "none"


@dataclass
class _ConstraintBatchTelemetry:
    total_pairs: int
    partial_supervision_hits: int
    unresolved_pairs: int
    rust_batch_call_count: int
    api_mode: str
    elapsed_seconds: float


def _accumulate_constraint_telemetry(
    accumulator: _ConstraintTelemetryAccumulator,
    batch_telemetry: _ConstraintBatchTelemetry,
) -> None:
    accumulator.total_pairs += int(batch_telemetry.total_pairs)
    accumulator.partial_supervision_hits += int(batch_telemetry.partial_supervision_hits)
    accumulator.unresolved_pairs += int(batch_telemetry.unresolved_pairs)
    accumulator.rust_batch_call_count += int(batch_telemetry.rust_batch_call_count)
    accumulator.elapsed_seconds += float(batch_telemetry.elapsed_seconds)
    accumulator.api_modes.add(str(batch_telemetry.api_mode))


@dataclass(frozen=True)
class _IncrementalConstraintBackend:
    """Pre-computed Phase A constraint backend state, invariant across subblocks."""

    rust_featurizer: object | None
    use_rust_constraints: bool | None
    constraint_api_mode: str
    signature_index_by_id: dict[str, int] | None
    suppress_orcid: bool = False


@dataclass(frozen=True)
class _IncrementalExperimentConfig:
    """Experiment-only controls for incremental single-letter assignment."""

    precluster_broadcast_mode: IncrementalBroadcastMode
    seed_score_mode: IncrementalSeedScoreMode
    mean_min_hybrid_weight: float


def _incremental_cluster_score(
    stats: IncrementalDistStats,
    *,
    config: _IncrementalExperimentConfig,
) -> float:
    """Return the assignment score for one query/seed cluster pair."""

    mean_dist, _count, min_dist = stats
    if config.seed_score_mode == "mean":
        return float(mean_dist)
    if config.seed_score_mode == "min":
        return float(min_dist)
    return float(
        (1.0 - float(config.mean_min_hybrid_weight)) * float(mean_dist)
        + float(config.mean_min_hybrid_weight) * float(min_dist)
    )


class Clusterer:
    """
    A wrapper for learning a clusterer

    Args:
        featurizer_info: FeaturizationInfo
            Featurization information
        classifier: sklearn compatible model
            Classifier which uses pairwise features to make a distance matrix
        val_blocks_size: int
            How many blocks to use during hyperparam optimization.
            Defaults to None, which uses all of them.
        cluster_model: sklearn compatible model
            Clusterer model
            Defaults to None, which uses FastCluster with average linking.
        search_space: Dict
            Search space for the hyperpamater optimization.
            Defaults to None, which uses a space appropriate to FastCluster.
        n_iter: int
            Number of hyperparameter evaluations
        n_jobs: int
            Parallelize each clusterer this many ways
        use_cache: bool
            Whether to use the cache when making distance matrices
        use_default_constraints_as_supervision: bool
            Whether to use the default constraints when constructing the distance matrices.
            These are high precision and can save a lot of compute/time.
        random_state: int
            Random state
        nameless_classifier: sklearn compatible model
            A second classifier which uses pairwise features excluding all name information, and
            whose predictions are averaged with the main classifier. Won't be used if None
        nameless_featurizer_info: FeaturizationInfo
            The FeaturizationInfo for the second classifier. Won't be used if None
        dont_merge_cluster_seeds: bool
            whether to enforce "disallow" constraints for signatures in different required seed clusters
        batch_size: int
            batch size for featurization, lower means less memory, but slower
        suppress_orcid: bool
            Whether default constraint resolution should ignore same-ORCID must-link constraints.
    """

    def __init__(
        self,
        featurizer_info: FeaturizationInfo,
        classifier: Any,
        val_blocks_size: int | None = None,
        cluster_model: Any | None = None,
        search_space: dict[str, Any] | None = None,
        n_iter: int = 25,
        n_jobs: int = 16,
        use_cache: bool = False,
        use_default_constraints_as_supervision: bool = True,
        random_state: int = 42,
        nameless_classifier: Any | None = None,
        nameless_featurizer_info: FeaturizationInfo | None = None,
        dont_merge_cluster_seeds: bool = True,
        batch_size: int = 1000000,
        suppress_orcid: bool = False,
    ):
        self.featurizer_info = featurizer_info
        self.nameless_featurizer_info = nameless_featurizer_info
        self.classifier = copy.deepcopy(classifier)
        self.nameless_classifier = copy.deepcopy(nameless_classifier)
        self.val_blocks_size = val_blocks_size
        self.n_iter = n_iter
        self._n_jobs = 1
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_cache = use_cache
        self.use_default_constraints_as_supervision = use_default_constraints_as_supervision
        self.dont_merge_cluster_seeds = dont_merge_cluster_seeds
        self.suppress_orcid = suppress_orcid
        if cluster_model is None:
            self.cluster_model = FastCluster(linkage="average")
        else:
            self.cluster_model = copy.deepcopy(cluster_model)

        if search_space is None:
            self.search_space = {"eps": hp.uniform("eps", 0, 1)}
        else:
            self.search_space = search_space

        default_name_count_semantics = _name_count_semantics_from_featurizer_version(
            getattr(self.featurizer_info, "featurizer_version", None)
        )
        self.feature_contract = {
            "name_counts_last_first_initial_semantics": (
                default_name_count_semantics or NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR
            ),
        }
        self.hyperopt_trials_store: Trials | list[Trials] | None = None
        self.best_params: dict[Any, Any] | None = None
        self.batch_size = batch_size
        self.incremental_precluster_broadcast_mode: IncrementalBroadcastMode = "always"
        self.incremental_seed_score_mode: IncrementalSeedScoreMode = "mean"
        self.incremental_mean_min_hybrid_weight: float = 0.5
        self.incremental_linker_artifact_dir: Path | None = None
        self.production_model_bundle_dir: Path | None = None
        self.production_model_bundle_version: str | None = None
        self.production_model_bundle_status: str | None = None
        # Post-clustering many-way topic split (see s2and/topic_split.py). Over-segments
        # each cluster's embeddings and re-merges sub-groups that share coauthors, to
        # separate distinct same-name authors that were over-merged. Never merges across
        # clusters, so it cannot lower recall on already-separated clusters. Read via
        # getattr at call sites so models unpickled from older bundles fall back to defaults.
        self.topic_split_enabled: bool = True
        self.topic_split_min_cluster: int = 8
        self.topic_split_min_subgroup: int = 2
        self.topic_split_embed_distance_threshold: float = 0.17
        self.topic_split_remerge_min_shared_coauthors: int = 2

    @property
    def n_jobs(self) -> int:
        return int(getattr(self, "_n_jobs", 1))

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        n_jobs = resolve_n_jobs(value)
        self._n_jobs = n_jobs
        _propagate_n_jobs(getattr(self, "classifier", None), n_jobs)
        _propagate_n_jobs(getattr(self, "nameless_classifier", None), n_jobs)

    @staticmethod
    def filter_blocks(block_dict: dict[str, list[str]], num_to_keep: int | None = None) -> dict[str, list[str]]:
        """
        Filter out blocks of size 1, as they are not useful or train/val

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        num_to_keep: int
            the number of blocks to keep, keeps all if None

        Returns
        -------
        either the loaded json, or the passed in object
        """
        # blocks with only 1 element are useless for train/val
        # and we can only keep as many as is specified
        out_dict = {}
        count = 0
        for block_key, signatures in block_dict.items():
            if len(signatures) > 1:
                out_dict[block_key] = signatures
                count += 1
                # early stopping if we have enough
                if num_to_keep is not None and count == num_to_keep:
                    return out_dict
        return out_dict

    def _incremental_experiment_config(self) -> _IncrementalExperimentConfig:
        """Return validated experiment controls for incremental assignment."""

        raw_broadcast_mode = str(getattr(self, "incremental_precluster_broadcast_mode", "always"))
        raw_seed_score_mode = str(getattr(self, "incremental_seed_score_mode", "mean"))
        raw_hybrid_weight = float(getattr(self, "incremental_mean_min_hybrid_weight", 0.5))

        valid_broadcast_modes: set[str] = {"always", "never", "top1_consensus"}
        if raw_broadcast_mode not in valid_broadcast_modes:
            raise ValueError(
                "Unsupported incremental_precluster_broadcast_mode="
                f"{raw_broadcast_mode!r}; expected one of {sorted(valid_broadcast_modes)}"
            )
        valid_seed_score_modes: set[str] = {"mean", "min", "mean_min_hybrid"}
        if raw_seed_score_mode not in valid_seed_score_modes:
            raise ValueError(
                "Unsupported incremental_seed_score_mode="
                f"{raw_seed_score_mode!r}; expected one of {sorted(valid_seed_score_modes)}"
            )
        if not 0.0 <= raw_hybrid_weight <= 1.0:
            raise ValueError("incremental_mean_min_hybrid_weight must be in [0, 1]; " f"got {raw_hybrid_weight!r}")
        return _IncrementalExperimentConfig(
            precluster_broadcast_mode=raw_broadcast_mode,  # type: ignore[arg-type]
            seed_score_mode=raw_seed_score_mode,  # type: ignore[arg-type]
            mean_min_hybrid_weight=float(raw_hybrid_weight),
        )

    def _best_incremental_cluster(
        self,
        cluster_dists: Mapping[int | str, IncrementalDistStats],
        *,
        config: _IncrementalExperimentConfig,
    ) -> tuple[int | str | None, float, float]:
        """Return the best and second-best seed-cluster scores for one query."""

        best_cluster_id: int | str | None = None
        best_score = float("inf")
        second_best_score = float("inf")
        for cluster_id, stats in cluster_dists.items():
            score = _incremental_cluster_score(stats, config=config)
            if score < best_score:
                second_best_score = best_score
                best_score = score
                best_cluster_id = cluster_id
            elif score < second_best_score:
                second_best_score = score
        return best_cluster_id, float(best_score), float(second_best_score)

    def _resolve_constraint_batch(
        self,
        dataset: ANDData,
        pair_ids: list[tuple[str, str]],
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        *,
        incremental_dont_use_cluster_seeds: bool,
        constraint_backend: _IncrementalConstraintBackend,
    ) -> tuple[list[float], _ConstraintBatchTelemetry]:
        return _resolve_constraint_labels_batch(
            dataset,
            pair_ids,
            constraint_backend=constraint_backend,
            partial_supervision=partial_supervision,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            constraint_policy=_ConstraintPolicy(
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                suppress_orcid=constraint_backend.suppress_orcid,
            ),
            runtime_context=runtime_context,
            num_threads=self.n_jobs,
        )

    def _flush_completed_block(
        self,
        *,
        block_key: str,
        pairwise_proba: np.ndarray | None,
        block_dict: dict[str, list[str]],
        effective_cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
        pred_clusters: defaultdict[str, list[str]],
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
            all_disallow_signature_ids,
            block_key=block_key,
        )
        for signature, label in zip(block_dict[block_key], labels, strict=True):
            pred_clusters[block_key + "_" + str(label)].append(signature)

    def distance_matrix_helper(
        self,
        block_dict: dict,
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
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
            runtime_context = build_runtime_context("constraints")
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
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
                    for (sig_id_1, sig_id_2), label, (left, right) in zip(
                        pair_batch_ids, labels, index_batch, strict=True
                    ):
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
        self,
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
    ):
        block_size = len(signatures)
        if block_size <= 1:
            return
        block_pair_count = int(block_size * (block_size - 1) / 2)
        pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
        tri_i, tri_j = np.triu_indices(block_size, k=1)
        offset = 0
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
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        pair_chunk_size: int | None = None,
    ):
        if runtime_context is None:
            runtime_context = build_runtime_context("constraints")
        if not stage_uses_rust(runtime_context):
            raise ValueError("Rust chunk helper is only valid when runtime_context resolves to rust backend")

        cache_policy = resolve_cache_policy(self.use_cache)
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
        )
        rust_featurizer = constraint_backend.rust_featurizer
        constraint_api_mode = constraint_backend.constraint_api_mode
        signature_index_by_id = constraint_backend.signature_index_by_id

        telemetry = _ConstraintTelemetryAccumulator()
        pair_chunk_size = max(1, int(pair_chunk_size if pair_chunk_size is not None else self.batch_size))
        used_fused_path = False
        use_fused_block_api = bool(
            self.use_default_constraints_as_supervision
            and not cache_policy.pair_feature_cache_enabled
            and constraint_api_mode == "indexed"
            and rust_featurizer is not None
            and signature_index_by_id is not None
            and hasattr(rust_featurizer, "get_constraints_block_upper_triangle_indexed")
            and hasattr(rust_featurizer, "featurize_block_upper_triangle_matrix_indexed")
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
                            dataset,
                            block_signature_indices,
                            start_offset=offset,
                            max_pairs=chunk_pair_count,
                            dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                            num_threads=self.n_jobs,
                            featurizer=rust_featurizer,
                            runtime_context=runtime_context,
                            suppress_orcid=constraint_backend.suppress_orcid,
                        )
                    except Exception as exc:
                        _handle_optional_rust_exception(
                            runtime_context,
                            strict_message="Rust fused block constraint evaluation failed in strict rust backend",
                            exc=exc,
                            python_path_warning=(
                                "Optional Rust fused block constraint evaluation failed while runtime backend "
                                "is Python; "
                                "using non-fused chunk path"
                            ),
                            context_fields=(
                                f"block={block_key}",
                                f"start_offset={offset}",
                                f"pairs={chunk_pair_count}",
                            ),
                        )
                        use_fused_block_api = False
                        break
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
                if not use_fused_block_api:
                    # Fused path disabled after optional-Rust failure; continue with non-fused chunks.
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
        self,
        chunk: _DistanceMatrixChunk,
        dataset: ANDData,
        runtime_context: RuntimeContext,
        batch_label: int | str,
        total_ram_bytes: int | None = None,
    ) -> tuple[np.ndarray, float]:
        cache_policy = resolve_cache_policy(self.use_cache)
        if chunk.block_signature_indices is not None and chunk.pair_ids is None:
            if cache_policy.pair_feature_cache_enabled:
                raise RuntimeError("Fused Rust chunk path does not support use_cache=True")
            try:
                rust_featurizer = _get_rust_featurizer(
                    dataset,
                    runtime_context=runtime_context,
                )
                selected_indices = _selected_feature_indices(self.featurizer_info)
                batch_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                    dataset,
                    chunk.block_signature_indices,
                    start_offset=int(chunk.start_offset),
                    max_pairs=int(len(chunk.labels)),
                    selected_indices=selected_indices,
                    num_threads=self.n_jobs,
                    nan_value=np.nan,
                    runtime_context=runtime_context,
                    featurizer=rust_featurizer,
                )
                batch_labels = np.asarray(chunk.labels, dtype=np.float64)
                batch_nameless_features: np.ndarray | None = None
                if self.nameless_classifier is not None and self.nameless_featurizer_info is not None:
                    nameless_selected_indices = _selected_feature_indices(self.nameless_featurizer_info)
                    batch_nameless_features = build_block_upper_triangle_feature_matrix_indexed_rust(
                        dataset,
                        chunk.block_signature_indices,
                        start_offset=int(chunk.start_offset),
                        max_pairs=int(len(chunk.labels)),
                        selected_indices=nameless_selected_indices,
                        num_threads=self.n_jobs,
                        nan_value=np.nan,
                        runtime_context=runtime_context,
                        featurizer=rust_featurizer,
                    )
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
                self.n_jobs,
                use_cache=cache_policy.pair_feature_cache_enabled,
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
            num_threads=self.n_jobs,
            runtime_context=runtime_context,
        )
        if int(batch_predictions.shape[0]) != expected_rows:
            raise RuntimeError(
                "Distance-matrix chunk prediction size mismatch: "
                f"expected={expected_rows} got={batch_predictions.shape[0]}"
            )
        return np.asarray(batch_predictions, dtype=np.float64), float(batch_seconds)

    def _iter_rust_predicted_distance_matrix_chunks(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        *,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ):
        chunk_count = 0
        helper_output = self._distance_matrix_chunk_helper_rust(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
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
                    "Distance-matrix batch prediction count mismatch: "
                    f"expected={expected} got={batch_predictions.shape[0]}"
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
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        *,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        num_pairs: int,
        pair_chunk_size: int | None = None,
        total_ram_bytes: int | None = None,
    ):
        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
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
                self.n_jobs,
                use_cache=self.use_cache,
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
                num_threads=self.n_jobs,
                runtime_context=runtime_context,
            )
            yield _PredictedDistanceMatrixBatch(
                batch_num=int(batch_num),
                blocks=blocks,
                indices=indices,
                predictions=np.asarray(batch_predictions, dtype=np.float64),
                batch_seconds=float(batch_seconds),
            )

            if count < self.batch_size:
                break
            batch_num += 1

    def _featurize_predict_write_batches(
        self,
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
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        disable_tqdm: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
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

        Returns
        -------
        Dict: the distance matrix dictionary, keyed by block key
        """
        runtime_context = build_runtime_context("model_predict")
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)
        if partial_supervision is None:
            partial_supervision = {}
        logger.info(f"Making {len(block_dict)} distance matrices")
        logger.info("Initializing pairwise_probas")
        # initialize pairwise_probas with correctly size arrays
        pairwise_probas = {}
        num_pairs = 0
        use_rust_blockwise = stage_uses_rust(runtime_context)
        fastcluster_dtype = np.float64 if use_rust_blockwise else np.float16
        for block_key, signatures in block_dict.items():
            block_size = len(signatures)
            num_pairs += int(block_size * (block_size - 1) / 2)
            if isinstance(self.cluster_model, FastCluster):
                # flattened pdist style
                pairwise_proba = np.zeros(int(block_size * (block_size - 1) / 2), dtype=fastcluster_dtype)
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            pairwise_probas[block_key] = pairwise_proba

        logger.info(f"Pairwise probas initialized with {num_pairs} elements, starting making all pairs")

        model_predict_seconds = 0.0
        if use_rust_blockwise:
            for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
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

    def fit(
        self,
        datasets: ANDData | list[ANDData],
        val_dists_precomputed: dict[str, dict[str, np.ndarray]] | None = None,
        metric_for_hyperopt: str = "b3",
    ) -> Clusterer:
        """
        Fits the clusterer

        Parameters
        ----------
        datasets: List[ANDData]
            the list of datasets to use for validations
        val_dists_precomputed: Dict
            precomputed distance matrices
        metric_for_hyperopt: string
            the metric to use for hyperparamter optimization

        Returns
        -------
        Clusterer: a fit clusterer, also sets the best params
        """
        assert metric_for_hyperopt in {"b3", "ratio"}
        logger.info("Fitting clusterer")
        if isinstance(datasets, ANDData):
            datasets = [datasets]
        if len(datasets) == 0:
            raise ValueError("Clusterer.fit requires at least one dataset")

        dataset_semantics = {getattr(dataset, "name_counts_last_first_initial_semantics", None) for dataset in datasets}
        if len(dataset_semantics) != 1:
            raise ValueError(
                "Clusterer.fit requires consistent name-count semantics across datasets; "
                f"observed={sorted(repr(value) for value in dataset_semantics)}"
            )
        training_semantics = next(iter(dataset_semantics))
        if training_semantics not in {
            NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
            NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
        }:
            raise ValueError(
                "Clusterer.fit could not determine valid dataset name-count semantics; "
                f"observed={training_semantics!r}"
            )
        contract = getattr(self, "feature_contract", None)
        if not isinstance(contract, dict):
            contract = {}
        contract["name_counts_last_first_initial_semantics"] = training_semantics
        self.feature_contract = contract

        val_block_dict_list = []
        val_cluster_to_signatures_list = []
        val_dists_list = []
        val_datasets_list: list[ANDData] = []
        weights: list[float] = []
        for dataset in datasets:
            # blocks
            train_block_dict, val_block_dict, _ = dataset.split_cluster_signatures()
            # incremental setting uses all the signatures in train and val
            # block-wise split uses only validation set for building the clustering model
            if dataset.unit_of_data_split == "time" or dataset.unit_of_data_split == "signatures":
                for block_key, signatures in train_block_dict.items():
                    if block_key in val_block_dict:
                        val_block_dict[block_key].extend(signatures)

            # we don't need val blocks with only a single element
            val_block_dict = self.filter_blocks(val_block_dict, self.val_blocks_size)
            val_block_dict_list.append(val_block_dict)

            # block ground truth labels: cluster_to_signatures
            val_cluster_to_signatures = dataset.construct_cluster_to_signatures(val_block_dict)
            val_cluster_to_signatures_list.append(val_cluster_to_signatures)

            # distance matrix
            if val_dists_precomputed is None:
                val_dists = self.make_distance_matrices(val_block_dict, dataset)
            else:
                val_dists = val_dists_precomputed[dataset.name]
            val_dists_list.append(val_dists)
            val_datasets_list.append(dataset)

            # weights for weighted F1 average: total # of signatures in dataset
            weights.append(np.sum([len(i) for i in val_block_dict.values()]))

        def obj(params):
            self.set_params(params)
            f1s = []
            ratios = []
            for val_dataset, val_block_dict, val_cluster_to_signatures, val_dists in zip(
                val_datasets_list, val_block_dict_list, val_cluster_to_signatures_list, val_dists_list, strict=True
            ):
                pred_clusters, _ = self.predict(
                    val_block_dict,
                    dataset=val_dataset,
                    dists=val_dists,
                )
                (
                    _,
                    _,
                    f1,
                    _,
                    pred_bigger_ratios,
                    true_bigger_ratios,
                ) = b3_precision_recall_fscore(val_cluster_to_signatures, pred_clusters)
                ratios.append(np.mean(pred_bigger_ratios + true_bigger_ratios))
                f1s.append(f1)
            if metric_for_hyperopt == "ratio":
                return np.average(ratios, weights=weights)
            elif metric_for_hyperopt == "b3":
                # minimize means we need to negate
                return -np.average(f1s, weights=weights)

        self.hyperopt_trials_store = Trials()
        _ = fmin(
            fn=obj,
            space=self.search_space,
            algo=partial(tpe.suggest, n_startup_jobs=5),
            max_evals=self.n_iter,
            trials=self.hyperopt_trials_store,
            rstate=np.random.default_rng(self.random_state),
        )
        # hyperopt has some problems with hp.choice so we need to do this:
        assert isinstance(self.hyperopt_trials_store, Trials)
        best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
        self.best_params = {k: intify(v) for k, v in best_params.items()}
        self.set_params(self.best_params)

        logger.info("Clusterer fit")
        return self

    def set_params(self, params: dict[str, Any] | None, clone_flag: bool = False):
        """
        Sets params on the cluster model

        Parameters
        ----------
        params: Dict
            the params to set
        clone_flag: bool
            whether to return a clone of the cluster model
        """
        if params is None:
            params = {}
        else:
            params = {k: intify(v) for k, v in params.items()}
        if clone_flag:
            cluster_model = clone(self.cluster_model)
            cluster_model.set_params(**params)
            return cluster_model
        else:
            self.cluster_model.set_params(**params)

    def _build_subblocked_block_dict(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        *,
        batching_threshold: int,
    ) -> dict[str, list[str]]:
        block_dict_subblocked: dict[str, list[str]] = {}
        for block_key in sorted(block_dict):
            block_signatures = block_dict[block_key]
            if len(block_signatures) > batching_threshold:
                subblocks = make_subblocks(block_signatures, dataset, maximum_size=batching_threshold)
                for subblock_key in sorted(subblocks):
                    subblock_signatures = subblocks[subblock_key]
                    block_dict_subblocked[f"{block_key}|subblock={subblock_key}"] = subblock_signatures
                    assert len(subblock_signatures) <= batching_threshold, "Subblock is too big for some reason!"
            else:
                block_dict_subblocked[block_key] = block_signatures
        return block_dict_subblocked

    def _partition_subblocked_first_name_groups(
        self,
        block_dict_subblocked: dict[str, list[str]],
        dataset: ANDData,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], bool]:
        single_letter = {
            block_key: block_signatures
            for block_key, block_signatures in block_dict_subblocked.items()
            if len(_signature_first_for_rules(dataset.signatures[block_signatures[0]])) <= 1
        }
        multiple_letter = {
            block_key: block_signatures
            for block_key, block_signatures in block_dict_subblocked.items()
            if block_key not in single_letter
        }
        if len(multiple_letter) == 0:
            return single_letter, {}, True
        return multiple_letter, single_letter, False

    def _predict_subblocked_multiple_letter_groups(
        self,
        block_dict_multiple_letter: dict[str, list[str]],
        *,
        alert_flag: bool,
        dataset: ANDData,
        cluster_model_params: dict[str, Any] | None,
        partial_supervision: dict[tuple[str, str], int | float],
        use_s2_clusters: bool,
        incremental_dont_use_cluster_seeds: bool,
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None,
    ) -> dict[str, list[str]]:
        pred_clusters: dict[str, list[str]] = {}
        if len(block_dict_multiple_letter) == 0:
            return pred_clusters

        if alert_flag:
            logger.info("Note! There are no subblocks with multiple letter first names")
            logger.info("Running predict on subblocks with single letter first names")
        else:
            logger.info("Running predict on subblocks with multiple letter first names")

        predict_times: dict[str, float] = {}
        for block_key in sorted(block_dict_multiple_letter):
            block_signatures = block_dict_multiple_letter[block_key]
            logger.info(f"Working on subblock {block_key}")
            start = time.time()
            pred_clusters_intermediate, _ = self.predict_helper(
                {block_key: block_signatures},
                dataset,
                dists=None,
                cluster_model_params=cluster_model_params,
                partial_supervision=partial_supervision,
                use_s2_clusters=use_s2_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            end = time.time()
            predict_times[block_key] = end - start
            pred_clusters.update(pred_clusters_intermediate)
        logger.info(f"Finished, here's how long each took: {predict_times}")
        return pred_clusters

    def _predict_subblocked_single_letter_incremental_groups(
        self,
        block_dict_single_letter: dict[str, list[str]],
        *,
        pred_clusters: dict[str, list[str]],
        desired_memory_use: int,
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        restore_rust_cluster_seeds_on_exit: bool,
        total_ram_bytes: int | None = None,
    ) -> dict[str, list[str]]:
        if len(block_dict_single_letter) == 0:
            return pred_clusters

        logger.info("Running predict incremental on subblocks with single letter first names")
        cluster_seeds_require_original = copy.deepcopy(dataset.cluster_seeds_require)
        dataset.cluster_seeds_require = {}
        for cluster_id, signatures in pred_clusters.items():
            for signature in signatures:
                dataset.cluster_seeds_require[signature] = cluster_id
        _bump_cluster_seeds_version(dataset)
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)

        predict_times: dict[str, float] = {}
        pred_clusters_intermediate: dict[str, list[str]] = pred_clusters
        for block_key in sorted(block_dict_single_letter.keys()):
            block_signatures = block_dict_single_letter[block_key]
            n_assigned = len(dataset.cluster_seeds_require)
            actual_memory_usage = len(block_signatures) * n_assigned
            logger.debug(
                "Incremental batching memory probe: "
                "n_seeds=%d n_signatures=%d desired_memory_use=%d actual_memory_usage=%d",
                n_assigned,
                len(block_signatures),
                int(desired_memory_use),
                int(actual_memory_usage),
            )
            if n_assigned <= 0:
                loop_batching_threshold = None
            elif actual_memory_usage > desired_memory_use:
                loop_batching_threshold = max(1, int(desired_memory_use / n_assigned))
            else:
                loop_batching_threshold = None
            logger.info(f"Working on subblock {block_key} with computed batching threshold {loop_batching_threshold}")
            start_predict_time = time.time()
            incremental_result = self.predict_incremental(
                block_signatures,
                dataset,
                prevent_new_incompatibilities=True,
                batching_threshold=loop_batching_threshold,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            clusters_payload = incremental_result.get("clusters")
            if not isinstance(clusters_payload, dict):
                raise RuntimeError(
                    "predict_incremental returned invalid clusters payload; expected dict "
                    f"got {type(clusters_payload).__name__}"
                )
            pred_clusters_intermediate = {}
            for cluster_id, signatures in clusters_payload.items():
                if not isinstance(signatures, list):
                    raise RuntimeError(
                        "predict_incremental returned invalid cluster member payload; expected list "
                        f"for cluster_id={cluster_id!r}, got {type(signatures).__name__}"
                    )
                pred_clusters_intermediate[str(cluster_id)] = [str(signature) for signature in signatures]
            end_predict_time = time.time()
            predict_times[block_key] = end_predict_time - start_predict_time

            dataset.cluster_seeds_require = {}
            for cluster_id, signatures in pred_clusters_intermediate.items():
                for signature in signatures:
                    dataset.cluster_seeds_require[signature] = cluster_id
            _bump_cluster_seeds_version(dataset)
            _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)

        logger.info(f"Finished subblocked predict incremental. Here's how long each subblock took: {predict_times}")
        dataset.cluster_seeds_require = cluster_seeds_require_original
        _bump_cluster_seeds_version(dataset)
        _ensure_cluster_seed_version_tracking(dataset)
        if restore_rust_cluster_seeds_on_exit:
            _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
        else:
            logger.info("Skipping final Rust cluster seed restore sync; evicting cached featurizer for dataset")
            evict_rust_featurizer(dataset)
        return pred_clusters_intermediate

    def _predict_subblocked(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        *,
        cluster_model_params: dict[str, Any] | None,
        partial_supervision: dict[tuple[str, str], int | float],
        use_s2_clusters: bool,
        incremental_dont_use_cluster_seeds: bool,
        batching_threshold: int,
        desired_memory_use: int | None,
        runtime_context: RuntimeContext,
        dists: dict[str, np.ndarray] | None,
        total_ram_bytes: int | None,
        restore_rust_cluster_seeds_on_exit: bool,
    ) -> tuple[dict[str, list[str]], None]:
        assert batching_threshold > 0, "Batching threshold must be positive"
        assert dists is None, "If batching_threshold is not None, then can't use precomputed dists"
        effective_desired_memory_use = (
            int(desired_memory_use) if desired_memory_use is not None else batching_threshold * batching_threshold
        )

        block_dict_subblocked = self._build_subblocked_block_dict(
            block_dict,
            dataset,
            batching_threshold=batching_threshold,
        )
        (
            block_dict_multiple_letter_first_names,
            block_dict_single_letter_first_names,
            alert_flag,
        ) = self._partition_subblocked_first_name_groups(block_dict_subblocked, dataset)

        pred_clusters = self._predict_subblocked_multiple_letter_groups(
            block_dict_multiple_letter_first_names,
            alert_flag=alert_flag,
            dataset=dataset,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            use_s2_clusters=use_s2_clusters,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        pred_clusters = self._predict_subblocked_single_letter_incremental_groups(
            block_dict_single_letter_first_names,
            pred_clusters=pred_clusters,
            desired_memory_use=effective_desired_memory_use,
            dataset=dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            restore_rust_cluster_seeds_on_exit=restore_rust_cluster_seeds_on_exit,
            total_ram_bytes=total_ram_bytes,
        )
        return dict(pred_clusters), None

    def predict(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
        batching_threshold: int | None = None,
        desired_memory_use: int | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        restore_rust_cluster_seeds_on_exit: bool = True,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """
        Predicts clusters

        Parameters
        ----------
        block_dict: Dict
            the block dict to predict clusters from
        dataset: ANDData
            the dataset
        dists: Dict
            (optional) precomputed distance matrices
        cluster_model_params: Dict
            params to set on the cluster model
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        use_s2_clusters: bool
            whether to "predict" using the clusters from Semantic Scholar's old system
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows
        batching_threshold: int | None
            If the number of signatures in a block is above this number, we will use subblocking on the block.
            This means that the single-letter first names will be sent through via predict_incremental.
            Defaults to None, which means no batching occurs
        desired_memory_use: int
            If batching_threshold is not None, then this is the desired memory use for predict_incremental.
            The units of this are the same as the units of batching_threshold -> number of signatures.
            If None, then using batching_threshold * batching_threshold as the desired memory use.
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for exact/non-incremental predict paths. When set, predict_helper
            uses it for pair-batch sizing and fails fast before allocating a block distance matrix that
            would exceed the usable budget.
        restore_rust_cluster_seeds_on_exit: bool
            If False, restore Python-side cluster seed state after the subblocked incremental path without
            issuing the final Rust seed sync. Intended for request-scoped datasets that are discarded after
            predict() returns.


        Note: batching_threshold is a hack to get around OOM issues. We will assume that it implies
        that we don't want to ever take up more memory than (batching_threshold ** 2)

        Returns
        -------
        Dict: the predicted clusters
        Optional[Dict]: the predicted distance matrices. This is None when
        distances are built and clustered in the fused one-block-at-a-time path.
        """

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict")

        if partial_supervision is None:
            partial_supervision = {}

        if batching_threshold is not None:
            pred_clusters, dists = self._predict_subblocked(
                block_dict,
                dataset,
                cluster_model_params=cluster_model_params,
                partial_supervision=partial_supervision,
                use_s2_clusters=use_s2_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                batching_threshold=int(batching_threshold),
                desired_memory_use=desired_memory_use,
                runtime_context=runtime_context,
                dists=dists,
                total_ram_bytes=total_ram_bytes,
                restore_rust_cluster_seeds_on_exit=restore_rust_cluster_seeds_on_exit,
            )

        else:
            # normal mode - everything goes through full block clustering
            logger.info("Running predict on full blocks - no subblocking")
            start = time.time()
            pred_clusters, dists = self.predict_helper(
                block_dict,
                dataset,
                dists=dists,
                cluster_model_params=cluster_model_params,
                partial_supervision=partial_supervision,
                use_s2_clusters=use_s2_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            end = time.time()
            total_predict_time = end - start
            logger.info(f"Finished predict on full blocks. Time taken: {total_predict_time}")

        pred_clusters = self._maybe_coauthor_corroborated_split(pred_clusters, dataset)

        return dict(pred_clusters), dists

    def _maybe_coauthor_corroborated_split(
        self, pred_clusters: dict[str, list[str]], dataset: ANDData
    ) -> dict[str, list[str]]:
        """Optionally split over-merged same-name clusters using topic + coauthor signals.

        Guarded by ``topic_split_enabled``; read via getattr so clusterers unpickled
        from older bundles (which lack these attributes) fall back to the defaults.
        """
        if not getattr(self, "topic_split_enabled", False):
            return pred_clusters
        if getattr(dataset, "specter_embeddings", None) is None:
            return pred_clusters
        try:
            from s2and.topic_split import coauthor_corroborated_split

            return coauthor_corroborated_split(
                pred_clusters,
                dataset,
                min_cluster=int(getattr(self, "topic_split_min_cluster", 8)),
                min_subgroup=int(getattr(self, "topic_split_min_subgroup", 2)),
                embed_distance_threshold=float(getattr(self, "topic_split_embed_distance_threshold", 0.17)),
                remerge_min_shared_coauthors=int(getattr(self, "topic_split_remerge_min_shared_coauthors", 2)),
            )
        except Exception as exc:  # never let the optional refinement break prediction
            logger.warning("coauthor_corroborated_split skipped due to error: %s", exc)
            return pred_clusters

    def _cluster_one_block(
        self,
        block_signatures: list[str],
        dist_matrix: np.ndarray | None,
        cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
    ) -> list:
        """Cluster one block from a distance matrix and return labels."""
        if len(block_signatures) <= 1:
            return [0]

        if dist_matrix is None:
            raise ValueError("Distance matrix is required for blocks with more than one signature.")

        cluster_model = clone(self.cluster_model)
        params = {k: intify(v) for k, v in (cluster_model_params or {}).items()}
        cluster_model.set_params(**params)
        with warnings.catch_warnings():
            # annoying sparse matrix not sorted warning
            warnings.simplefilter("ignore", category=EfficiencyWarning)
            cluster_model.fit(dist_matrix)
        labels = cluster_model.labels_
        max_label = labels.max()
        # In HDBSCAN, label -1 denotes outliers.
        # Give each outlier its own unique label starting at max_label + 1.
        negative_one_label_locations = np.where(labels == -1)[0]
        for i, loc in enumerate(negative_one_label_locations):
            labels[loc] = max_label + 1 + i
        if self.use_default_constraints_as_supervision:
            disallow_signature_ids = all_disallow_signature_ids
            inverse_id_map = defaultdict(set)
            for signature_id, label in zip(block_signatures, labels, strict=True):
                if signature_id in dataset.cluster_seeds_require and signature_id not in disallow_signature_ids:
                    inverse_id_map[dataset.cluster_seeds_require[signature_id]].add(label)
            # Clusters that should merge can still remain split after distance-based clustering.
            # This happens when required-pair zero distances are outweighed by many large distances
            # in average-linkage behavior. Post-hoc, merge label sets that overlap according to
            # cluster_seeds_require (excluding signatures that appear in disallow constraints).
            to_join_sets = [sorted(join_set) for join_set in inverse_id_map.values() if len(join_set) > 1]
            mapped_labels = {label: label for label in labels}
            labels = np.array(labels)
            for join_set in to_join_sets:
                for other_label in join_set[1:]:
                    labels[labels == mapped_labels[other_label]] = mapped_labels[join_set[0]]
                    mapped_labels[other_label] = mapped_labels[join_set[0]]
            labels = list(labels)
        return labels

    def _cluster_one_block_with_logging(
        self,
        block_signatures: list[str],
        dist_matrix: np.ndarray | None,
        cluster_model_params: dict[str, Any] | None,
        dataset: ANDData,
        all_disallow_signature_ids: set[str],
        *,
        block_key: str,
    ) -> list:
        """Cluster one block and emit explicit entry/exit logs around the cluster-model fit."""
        cluster_model_name = type(self.cluster_model).__name__
        logger.info(
            "Starting cluster_model.fit for block %s using %s (signatures=%d)",
            block_key,
            cluster_model_name,
            len(block_signatures),
        )
        cluster_start = time.perf_counter()
        labels = self._cluster_one_block(
            block_signatures,
            dist_matrix,
            cluster_model_params,
            dataset,
            all_disallow_signature_ids,
        )
        logger.info(
            "Finished cluster_model.fit for block %s using %s in %.3fs (clusters=%d)",
            block_key,
            cluster_model_name,
            time.perf_counter() - cluster_start,
            len(set(labels)),
        )
        return labels

    def predict_helper(
        self,
        block_dict: dict[str, list[str]],
        dataset: ANDData,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """
        Predicts clusters

        Parameters
        ----------
        block_dict: Dict
            the block dict to predict clusters from
        dataset: ANDData
            the dataset
        dists: Dict
            (optional) precomputed distance matrices
        cluster_model_params: Dict
            params to set on the cluster model
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        use_s2_clusters: bool
            whether to "predict" using the clusters from Semantic Scholar's old system
        incremental_dont_use_cluster_seeds: bool
            whether to ignore dataset cluster seeds while resolving constraints in incremental flows
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for pair-batch sizing and exact block-matrix allocation checks.

        Returns
        -------
        Dict: the predicted clusters
        Optional[Dict]: the predicted distance matrices. This is None when
        distances are built and clustered in the fused one-block-at-a-time path.
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict")

        if partial_supervision is None:
            partial_supervision = {}
        _apply_dataset_name_count_semantics_for_prediction(self, dataset)

        pred_clusters = defaultdict(list)

        if use_s2_clusters:
            for _, signature_list in block_dict.items():
                for _signature in signature_list:
                    s2_cluster_key = dataset.signatures[_signature].author_id
                    pred_clusters[s2_cluster_key].append(_signature)

            return dict(pred_clusters), dists

        # we may need this set later for post-hoc merging
        # pre-compute disallow set for post-hoc constraint merging
        all_disallow_signature_ids: set[str] = set()
        if self.use_default_constraints_as_supervision:
            for sig_id_a, sig_id_b in dataset.cluster_seeds_disallow:
                all_disallow_signature_ids.add(sig_id_a)
                all_disallow_signature_ids.add(sig_id_b)

        effective_cluster_model_params = cluster_model_params
        fastcluster_fused_dtype = np.float16
        if isinstance(self.cluster_model, FastCluster):
            fastcluster_params: dict[str, Any] = dict(cluster_model_params or {})
            # Reused matrices should stay immutable by default; single-use fused path favors lower peak memory.
            if "preserve_input" not in fastcluster_params:
                fastcluster_params["preserve_input"] = bool(dists is not None)
            effective_cluster_model_params = fastcluster_params
            if dists is None:
                fastcluster_fused_dtype = np.float64

        if dists is not None:
            # precomputed dists (hyperopt path) — cluster from existing matrices
            for block_key in block_dict.keys():
                if block_key not in dists:
                    raise KeyError(f"Missing precomputed distance matrix for block '{block_key}'")
                labels = self._cluster_one_block_with_logging(
                    block_dict[block_key],
                    dists[block_key],
                    effective_cluster_model_params,
                    dataset,
                    all_disallow_signature_ids,
                    block_key=block_key,
                )
                for signature, label in zip(block_dict[block_key], labels, strict=True):
                    pred_clusters[block_key + "_" + str(label)].append(signature)
            return dict(pred_clusters), dists

        # fused path: build one block's matrix, cluster it, free it, repeat
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)

        prev_block_key = ""
        pairwise_proba: np.ndarray | None = None
        seen_block_keys: set[str] = set()
        num_pairs = sum(len(sigs) * (len(sigs) - 1) // 2 for sigs in block_dict.values())
        logger.info("Predict helper: total_pairs=%d", num_pairs)
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
        if batch_chunk_plan is not None:
            logger.info(
                "Predict memory chunking: total_pairs=%d chunk_pairs=%d total_ram=%d total_ram_source=%s "
                "rss=%d rss_source=%s available=%d effective_available_fraction=%.3f stage_budget_bytes=%d",
                num_pairs,
                pair_chunk_size,
                int(batch_chunk_plan.total_ram_bytes),
                str(batch_chunk_plan.total_ram_source),
                int(batch_chunk_plan.current_rss_bytes),
                str(batch_chunk_plan.current_rss_source),
                int(batch_chunk_plan.available_bytes),
                float(batch_chunk_plan.effective_available_fraction),
                int(batch_chunk_plan.stage_budget_bytes),
            )
        use_rust_blockwise = stage_uses_rust(runtime_context)
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
                block_key = chunk.block_key
                if block_key != prev_block_key:
                    # cluster the completed block
                    self._flush_completed_block(
                        block_key=prev_block_key,
                        pairwise_proba=pairwise_proba,
                        block_dict=block_dict,
                        effective_cluster_model_params=effective_cluster_model_params,
                        dataset=dataset,
                        all_disallow_signature_ids=all_disallow_signature_ids,
                        pred_clusters=pred_clusters,
                    )
                    pairwise_proba = None

                    # allocate new block's matrix
                    seen_block_keys.add(block_key)
                    _guard_predict_block_matrix_allocation(
                        block_key=block_key,
                        block_size=int(chunk.block_size),
                        uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                        fastcluster_fused_dtype=fastcluster_fused_dtype,
                        total_ram_bytes=total_ram_bytes,
                    )
                    if isinstance(self.cluster_model, FastCluster):
                        pairwise_proba = np.zeros(
                            chunk.block_size * (chunk.block_size - 1) // 2,
                            dtype=fastcluster_fused_dtype,
                        )
                    else:
                        pairwise_proba = np.zeros((chunk.block_size, chunk.block_size), dtype=np.float16)

                batch_predictions = prediction_chunk.predictions
                model_predict_seconds += prediction_chunk.batch_seconds
                assert pairwise_proba is not None
                if isinstance(self.cluster_model, FastCluster):
                    start = int(chunk.start_offset)
                    end = start + int(len(chunk.labels))
                    pairwise_proba[start:end] = np.asarray(batch_predictions, dtype=pairwise_proba.dtype)
                else:
                    pairwise_proba[chunk.index_i, chunk.index_j] = np.asarray(
                        batch_predictions,
                        dtype=pairwise_proba.dtype,
                    )
                prev_block_key = block_key

            # cluster the final block
            self._flush_completed_block(
                block_key=prev_block_key,
                pairwise_proba=pairwise_proba,
                block_dict=block_dict,
                effective_cluster_model_params=effective_cluster_model_params,
                dataset=dataset,
                all_disallow_signature_ids=all_disallow_signature_ids,
                pred_clusters=pred_clusters,
            )
        else:
            block_pair_index = 0

            def _on_block_start(block_key: str) -> None:
                nonlocal pairwise_proba, block_pair_index
                pairwise_proba = None
                seen_block_keys.add(block_key)
                block_size = len(block_dict[block_key])
                _guard_predict_block_matrix_allocation(
                    block_key=block_key,
                    block_size=block_size,
                    uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                    fastcluster_fused_dtype=fastcluster_fused_dtype,
                    total_ram_bytes=total_ram_bytes,
                )
                if isinstance(self.cluster_model, FastCluster):
                    pairwise_proba = np.zeros(
                        block_size * (block_size - 1) // 2,
                        dtype=fastcluster_fused_dtype,
                    )
                else:
                    pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
                block_pair_index = 0

            def _write_prediction(
                _block_key: str,
                index_pair: tuple[int, int],
                prediction: float,
            ) -> None:
                nonlocal block_pair_index
                assert pairwise_proba is not None
                if isinstance(self.cluster_model, FastCluster):
                    pairwise_proba[block_pair_index] = prediction
                else:
                    i, j = index_pair
                    pairwise_proba[i, j] = prediction
                block_pair_index += 1

            def _post_block_callback(block_key: str) -> None:
                self._flush_completed_block(
                    block_key=block_key,
                    pairwise_proba=pairwise_proba,
                    block_dict=block_dict,
                    effective_cluster_model_params=effective_cluster_model_params,
                    dataset=dataset,
                    all_disallow_signature_ids=all_disallow_signature_ids,
                    pred_clusters=pred_clusters,
                )

            model_predict_seconds += self._featurize_predict_write_batches(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                num_pairs=num_pairs,
                write_prediction=_write_prediction,
                on_block_start=_on_block_start,
                post_block_callback=_post_block_callback,
                pair_chunk_size=pair_chunk_size,
                total_ram_bytes=total_ram_bytes,
            )

        # handle singleton blocks (0 or 1 signature — never appeared in generator)
        for block_key in block_dict.keys():
            if block_key not in seen_block_keys:
                for signature in block_dict[block_key]:
                    pred_clusters[block_key + "_0"].append(signature)

        logger.info(
            "Telemetry stage: stage=model_predict_total seconds=%.3f blocks=%d",
            model_predict_seconds,
            len(block_dict),
        )
        return dict(pred_clusters), None

    def _build_incremental_seed_setup(
        self,
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> tuple[
        dict[str, int | str],
        dict[int | str, int | str],
        dict[int | str, list[str]],
    ]:
        recluster_map: dict[int | str, int | str] = {}
        cluster_seeds_require = copy.deepcopy(dataset.cluster_seeds_require)
        cluster_seeds_require_inverse: dict[int | str, list[str]] = defaultdict(list)
        for signature_id, cluster_num in dataset.cluster_seeds_require.items():
            cluster_seeds_require_inverse[cluster_num].append(signature_id)

        # Split altered claimed profiles once so incremental assignment can map back to original cluster IDs.
        # Claimed profiles from production can be "unnatural" with respect to S2AND constraints;
        # this pre-split step aligns them to natural-looking clusters before adding new signatures.
        if dataset.altered_cluster_signatures is not None and len(dataset.altered_cluster_signatures) > 0:
            logger.info("Dealing with altered cluster signatures")
            altered_cluster_nums = set(
                dataset.cluster_seeds_require[altered_signature_id]
                for altered_signature_id in dataset.altered_cluster_signatures
                if altered_signature_id in dataset.cluster_seeds_require
            )
            # It's possible an altered signature is not in cluster_seeds_require when
            # a custom seed map is passed; skip those safely here.
            for altered_cluster_num in altered_cluster_nums:
                signature_ids_for_cluster_num = cluster_seeds_require_inverse.get(altered_cluster_num, [])
                if len(signature_ids_for_cluster_num) == 0:
                    continue

                # During this pre-split, do not apply incoming cluster seeds as constraints.
                # At this stage we are splitting claimed profiles to match S2AND predictions,
                # so claimed-profile seeds should not bias the split.
                reclustered_output, _ = self.predict_helper(
                    {"block": signature_ids_for_cluster_num},
                    dataset,
                    incremental_dont_use_cluster_seeds=True,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
                if len(reclustered_output) <= 1:
                    continue
                for i, new_cluster_of_signatures in enumerate(reclustered_output.values()):
                    new_cluster_num = str(altered_cluster_num) + f"_{i}"
                    recluster_map[new_cluster_num] = altered_cluster_num
                    for reclustered_signature_id in new_cluster_of_signatures:
                        cluster_seeds_require[reclustered_signature_id] = new_cluster_num

        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse

    def _convert_sum_count_to_average_distances(
        self,
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
    ) -> dict[str, dict[int | str, IncrementalDistStats]]:
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]] = defaultdict(dict)
        for signature_id, cluster_sum_count in signature_to_cluster_sum_count.items():
            for cluster_id, sum_count in cluster_sum_count.items():
                total = float(sum_count[0])
                count = int(sum_count[1])
                if count <= 0:
                    continue
                signature_to_cluster_to_average_dist[signature_id][cluster_id] = (
                    total / float(count),
                    count,
                    float(sum_count[2]),
                )
        return signature_to_cluster_to_average_dist

    def _finish_incremental_with_seed_links(
        self,
        unassigned_signature_ids: list[str],
        dataset: ANDData,
        linked_signature_to_cluster: Mapping[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> dict[str, list[str]]:
        """Apply supplied seed-link decisions, then recluster abstained signatures."""

        logger.info("Assigning unassigned signatures for incremental clustering")
        pred_clusters = defaultdict(list)
        singleton_signatures = []
        for signature_id, cluster_id in dataset.cluster_seeds_require.items():
            pred_clusters[f"{cluster_id}"].append(signature_id)
        for unassigned_signature in unassigned_signature_ids:
            if unassigned_signature not in linked_signature_to_cluster:
                singleton_signatures.append(unassigned_signature)
                continue

            best_cluster_id = linked_signature_to_cluster[unassigned_signature]
            # undo the altered-cluster split if applicable
            new_name_disallowed = False
            if best_cluster_id in recluster_map:
                best_cluster_id = recluster_map[best_cluster_id]

                if prevent_new_incompatibilities:
                    # restrict reclusterings that would add a new name incompatibility to the main cluster
                    main_cluster_signatures = cluster_seeds_require_inverse[best_cluster_id]
                    all_firsts = set(
                        _signature_first_for_rules(dataset.signatures[signature_id])
                        for signature_id in main_cluster_signatures
                    )
                    all_firsts = {first for first in all_firsts if len(first) > 1}

                    # if all existing first names are single characters, there is nothing else to check
                    if len(all_firsts) > 0:
                        first_unassigned = _signature_first_for_rules(dataset.signatures[unassigned_signature])
                        match_found = False
                        for first_assigned in all_firsts:
                            if first_names_name_compatible(first_assigned, first_unassigned, dataset.name_tuples):
                                match_found = True
                                break
                        # if the candidate name is a prefix or a name alias for any existing name,
                        # we allow it to cluster. Otherwise, it was clustered with a single-character
                        # name and we don't want to allow that merge.
                        if not match_found:
                            signature = dataset.signatures[unassigned_signature]
                            first = signature.author_info_first
                            last = signature.author_info_last
                            paper_id = signature.paper_id
                            logger.info(
                                "Incremental clustering prevented a name compatibility issue from being "
                                f"added while clustering {first} {last} on {paper_id}"
                            )
                            new_name_disallowed = True

            if new_name_disallowed:
                singleton_signatures.append(unassigned_signature)
            else:
                pred_clusters[f"{best_cluster_id}"].append(unassigned_signature)

        # all remaining singletons are reclustered together
        if len(singleton_signatures) > 0:
            logger.info("Clustering together the still unassigned signatures")
            reclustered_output, _ = self.predict_helper(
                {"block": singleton_signatures},
                dataset,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            new_cluster_id = _next_unused_cluster_id(pred_clusters, int(dataset.max_seed_cluster_id or 0))
            for new_cluster in reclustered_output.values():
                new_cluster_id = _next_unused_cluster_id(pred_clusters, new_cluster_id)
                pred_clusters[str(new_cluster_id)] = new_cluster
                new_cluster_id += 1
        logger.info("Done. Returning incrementally predicted clusters")
        return dict(pred_clusters)

    def _run_incremental_phases_bcd(
        self,
        unassigned_signature_ids: list[str],
        dataset: ANDData,
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]],
        cluster_seeds_require: dict[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> dict[str, list[str]]:
        config = self._incremental_experiment_config()
        # NEW!
        # First cluster the unassigned signatures, then decide which resulting unassigned
        # clusters should merge with existing seeded clusters.
        logger.info("Batch clustering the unassigned signatures")
        incremental_only_clusters, _ = self.predict_helper(
            {"incremental_unassigned": unassigned_signature_ids},
            dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info(
            "Made %d clusters out of %d unassigned signatures",
            len(incremental_only_clusters),
            len(unassigned_signature_ids),
        )

        if config.precluster_broadcast_mode != "never":
            # Average over Phase A signature-to-seed distances at the pre-cluster level.
            # This is equivalent to computing average distance between each unassigned cluster
            # and each assigned cluster, then broadcasting that score back to member signatures.
            cluster_ids = sorted(set(cluster_seeds_require.values()), key=lambda cluster_id: str(cluster_id))
            for incremental_cluster_signature_ids in incremental_only_clusters.values():
                if config.precluster_broadcast_mode == "top1_consensus":
                    top1_cluster_id: int | str | None = None
                    should_broadcast = True
                    for signature in incremental_cluster_signature_ids:
                        best_cluster_id, best_score, _second_best_score = self._best_incremental_cluster(
                            signature_to_cluster_to_average_dist.get(signature, {}),
                            config=config,
                        )
                        if best_cluster_id is None or not math.isfinite(best_score):
                            should_broadcast = False
                            break
                        if top1_cluster_id is None:
                            top1_cluster_id = best_cluster_id
                        elif top1_cluster_id != best_cluster_id:
                            should_broadcast = False
                            break
                    if not should_broadcast:
                        continue
                for cluster_id in cluster_ids:
                    mean_dists = []
                    min_dists = []
                    support_count = 0
                    for signature in incremental_cluster_signature_ids:
                        cluster_entry = signature_to_cluster_to_average_dist.get(signature, {}).get(cluster_id)
                        if cluster_entry is None:
                            continue
                        if int(cluster_entry[1]) <= 0:
                            continue
                        mean_dists.append(float(cluster_entry[0]))
                        min_dists.append(float(cluster_entry[2]))
                        support_count += int(cluster_entry[1])
                    if len(mean_dists) == 0:
                        continue
                    out = (
                        float(np.mean(mean_dists)),
                        int(support_count),
                        float(min(min_dists)),
                    )
                    for signature in incremental_cluster_signature_ids:
                        signature_to_cluster_to_average_dist.setdefault(signature, {})[cluster_id] = out

        linked_signature_to_cluster: dict[str, int | str] = {}
        for unassigned_signature in unassigned_signature_ids:
            cluster_dists = signature_to_cluster_to_average_dist.get(unassigned_signature, {})
            best_cluster_id, best_dist, _second_best_dist = self._best_incremental_cluster(
                cluster_dists,
                config=config,
            )
            if best_cluster_id is not None and best_dist < self.cluster_model.eps:
                linked_signature_to_cluster[unassigned_signature] = best_cluster_id

        return self._finish_incremental_with_seed_links(
            unassigned_signature_ids,
            dataset,
            linked_signature_to_cluster,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

    def _predict_incremental_promoted_linker(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        *,
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None,
        batching_threshold: int | None,
    ) -> dict[str, Any]:
        artifact_dir = Path(
            getattr(self, "incremental_linker_artifact_dir", None) or DEFAULT_INCREMENTAL_LINKER_ARTIFACT_DIR
        )
        return predict_incremental_promoted_linker(
            self,
            block_signatures,
            dataset,
            artifact_dir=artifact_dir,
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            batching_threshold=batching_threshold,
            resolve_total_ram_bytes=_resolve_total_ram_bytes_for_incremental,
            build_incremental_result=_build_incremental_result,
            get_rust_featurizer=_get_rust_featurizer,
            build_incremental_constraint_backend=_build_incremental_constraint_backend,
        )

    def predict_incremental(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool = True,
        batching_threshold: int | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        return_clusters_only: bool = False,
    ) -> dict[str, Any] | dict[str, list[str]]:
        """
        Predict clustering in incremental mode. This assumes that the majority of the labels are passed
        in using the cluster_seeds_require parameter of the dataset class, and skips work by simply assigning each
        unassigned signature to the closest cluster if distance is less than eps, and then separately clusters all
        the unassigned signatures that are not within eps of any existing cluster.

        Corrected, claimed profiles should be noted via the altered_cluster_signatures parameter (in ANDData).
        Then predict_incremental performs a pre-clustering step on each altered cluster to determine how
        S2AND would divide it into clusters. Mentions are incrementally added to these new subclusters,
        then reassembled to restore the complete claimed profile when S2AND returns results.

        Currently this would be useful in the following situation. We have a massive block, for which we want
        to cluster a small number of new signatures into (block size * number of new signatures should be less
        than the normal batch size).

        Note: this function was designed to work on a single block at a time.

        Parameters
        ----------
        block_signatures: List[str]
            the signature ids in the block to predict from
        dataset: ANDData
            the dataset
        prevent_new_incompatibilities: bool
            if True, prevents the addition to a cluster of new first names that are not prefix match
            or in the name pairs list, for at least one existing name in the cluster. This can happen
            if a claimed cluster has D Jones and David Jones, s2and would have split that cluster into two,
            and then s2and might add Donald Jones to the D Jones cluster, and once remerged, the resulting
            final cluster would have D Jones, David Jones, and Donald Jones.
        batching_threshold: int
            Optional promoted Rust query batch limit. This is only supported when the runtime backend resolves to
            Rust and cluster seeds are available. Python incremental fallback raises if this is provided because it
            does not implement incremental batching.
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for promoted incremental query batching.
        return_clusters_only: bool
            If True, return only the historical clusters dict shape instead of the full
            telemetry payload.
        Returns
        -------
        Dict: incremental clustering payload (default) or clusters-only dict when
        return_clusters_only=True
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental")
        _apply_dataset_name_count_semantics_for_prediction(self, dataset)
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
        if partial_supervision is None:
            partial_supervision = {}
        use_rust_backend = stage_uses_rust(runtime_context)
        if batching_threshold is not None and (not use_rust_backend or len(dataset.cluster_seeds_require) == 0):
            raise ValueError(
                "batching_threshold is only supported for promoted Rust incremental prediction with cluster seeds; "
                "Python incremental fallback does not implement batched incremental routing. "
                "Use the Rust backend with cluster seeds or pass batching_threshold=None."
            )
        if use_rust_backend:
            if len(dataset.cluster_seeds_require) == 0:
                logger.info(
                    "No cluster seeds provided for promoted incremental; "
                    "falling back to Python incremental helper for partition coverage."
                )
                incremental_result = self._predict_incremental_helper(
                    block_signatures,
                    dataset,
                    prevent_new_incompatibilities=prevent_new_incompatibilities,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                )
                return dict(incremental_result["clusters"]) if return_clusters_only else incremental_result
            incremental_result = self._predict_incremental_promoted_linker(
                block_signatures,
                dataset,
                prevent_new_incompatibilities=prevent_new_incompatibilities,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                batching_threshold=batching_threshold,
            )
            return dict(incremental_result["clusters"]) if return_clusters_only else incremental_result
        incremental_result = self._predict_incremental_helper(
            block_signatures,
            dataset,
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        return dict(incremental_result["clusters"]) if return_clusters_only else incremental_result

    def _predict_incremental_helper(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool = True,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Internal incremental execution path used by `predict_incremental`.

        For behavior/parameters, refer to `predict_incremental`.
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental")
        _apply_dataset_name_count_semantics_for_prediction(self, dataset)
        if partial_supervision is None:
            partial_supervision = {}
        logger.info(f"Beginning incremental clustering for {len(block_signatures)} signatures...")
        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = self._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info("Getting name constraints")
        all_pairs: list[tuple[str, str, float]] = []
        unassigned_signature_ids: list[str] = []
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            suppress_orcid=getattr(self, "suppress_orcid", False),
        )
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, IncrementalDistStats]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0, float("inf")))
        )
        assigned_signature_ids: list[str] = list(cluster_seeds_require.keys())
        pair_chunk_size = max(1, int(self.batch_size))
        constraint_telemetry = _ConstraintTelemetryAccumulator()

        def _update_signature_cluster_average(unassigned_signature: str, cluster_id: int | str, dist: float) -> None:
            previous_average, previous_count, previous_min = signature_to_cluster_to_average_dist[unassigned_signature][
                cluster_id
            ]
            signature_to_cluster_to_average_dist[unassigned_signature][cluster_id] = (
                (previous_average * previous_count + float(dist)) / (previous_count + 1),
                previous_count + 1,
                min(float(previous_min), float(dist)) if previous_count > 0 else float(dist),
            )

        for possibly_unassigned_signature in block_signatures:
            if possibly_unassigned_signature in cluster_seeds_require:
                continue
            unassigned_signature_ids.append(possibly_unassigned_signature)

        pair_id_batch: list[tuple[str, str]] = []
        for unassigned_signature in unassigned_signature_ids:
            for assigned_signature in assigned_signature_ids:
                pair_id_batch.append((unassigned_signature, assigned_signature))
                if len(pair_id_batch) >= pair_chunk_size:
                    labels, batch_telemetry = self._resolve_constraint_batch(
                        dataset,
                        pair_id_batch,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=False,
                        constraint_backend=constraint_backend,
                    )
                    _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
                    all_pairs.extend(
                        (sig_id_1, sig_id_2, label)
                        for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=True)
                    )
                    pair_id_batch = []

        if pair_id_batch:
            labels, batch_telemetry = self._resolve_constraint_batch(
                dataset,
                pair_id_batch,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=False,
                constraint_backend=constraint_backend,
            )
            _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
            all_pairs.extend(
                (sig_id_1, sig_id_2, label) for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=True)
            )

        logger.info(
            "Telemetry: constraint_batch stage=_predict_incremental_helper total_pairs=%d "
            "partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            constraint_telemetry.total_pairs,
            constraint_telemetry.partial_supervision_hits,
            constraint_telemetry.unresolved_pairs,
            constraint_telemetry.rust_batch_call_count,
            constraint_telemetry.api_mode_summary,
            constraint_telemetry.elapsed_seconds,
            runtime_context.run_id,
        )

        logger.info("Featurizing pairs")
        batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
            all_pairs,
            dataset,
            self.featurizer_info,
            self.n_jobs,
            use_cache=self.use_cache,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )

        logger.info("Performing pairwise classification")
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)
        # Get predictions where there isn't partial supervision,
        # and fill the rest from partial supervision labels.
        batch_predictions, model_predict_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            batch_features,
            batch_labels,
            batch_nameless_features,
            "incremental",
            num_threads=self.n_jobs,
            runtime_context=runtime_context,
        )
        logger.info("Telemetry: model_predict_total seconds=%.3f blocks=1", model_predict_seconds)

        logger.info("Computing average distances for unassigned signatures")
        for signature_pair, dist in zip(all_pairs, batch_predictions, strict=True):
            unassigned_signature, assigned_signature, _ = signature_pair
            if assigned_signature not in cluster_seeds_require:
                continue
            cluster_id = cluster_seeds_require[assigned_signature]
            _update_signature_cluster_average(unassigned_signature, cluster_id, float(dist))

        predicted_clusters = self._run_incremental_phases_bcd(
            unassigned_signature_ids,
            dataset,
            signature_to_cluster_to_average_dist,
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        phase_b_required_bytes = len(unassigned_signature_ids) * (len(unassigned_signature_ids) - 1) // 2 * 8
        return _build_incremental_result(
            predicted_clusters,
            phase_b_mode="exact",
            phase_b_budget_bytes=phase_b_required_bytes,
            phase_b_required_bytes=phase_b_required_bytes,
        )

    def predict_incremental_helper(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool = True,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        """Deprecated shim for `predict_incremental`; use the public method instead."""
        warnings.warn(
            "Clusterer.predict_incremental_helper is deprecated; use predict_incremental instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._predict_incremental_helper(
            block_signatures,
            dataset,
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
