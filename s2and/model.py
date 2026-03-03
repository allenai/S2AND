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
from typing import Any, Literal, Self, TypeVar

import lightgbm as lgb
import numpy as np
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.base import clone
from sklearn.exceptions import EfficiencyWarning
from tqdm import tqdm

from s2and import memory_budget
from s2and.consts import DEFAULT_CHUNK_SIZE, LARGE_INTEGER
from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
    ANDData,
)
from s2and.eval import b3_precision_recall_fscore
from s2and.feature_port import (
    _get_rust_featurizer,
    build_block_upper_triangle_feature_matrix_indexed_rust,
    get_constraint_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    get_constraints_matrix_indexed_rust,
    update_rust_cluster_seeds,
)
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.model_pairwise import FastCluster, PairwiseModeler, VotingClassifier, intify
from s2and.runtime import RuntimeContext, build_runtime_context, stage_uses_rust
from s2and.subblocking import make_subblocks
from s2and.text import same_prefix_tokens
from s2and.warnings_utils import suppress_sklearn_feature_name_warnings

logger = logging.getLogger("s2and")
IncrementalChunkLimits = Mapping[str, Any]
IncrementalPhaseBMode = Literal["exact", "subblock_local"]
_TReturn = TypeVar("_TReturn")
_MISSING = object()

# Keep canonical pickle import paths stable after splitting module internals.
for _export in (FastCluster, PairwiseModeler, VotingClassifier, intify):
    _export.__module__ = __name__


def _build_incremental_result(
    clusters: dict[str, list[str]],
    *,
    phase_b_mode: IncrementalPhaseBMode,
    phase_b_budget_bytes: int,
    phase_b_required_bytes: int,
    phase_a_accumulator_overflow_early_stop: bool = False,
    phase_a_adaptive_halvings_max: int = 0,
) -> dict[str, Any]:
    return {
        "clusters": clusters,
        "phase_b_mode": phase_b_mode,
        "phase_b_budget_bytes": int(phase_b_budget_bytes),
        "phase_b_required_bytes": int(phase_b_required_bytes),
        "phase_a_accumulator_overflow_early_stop": bool(phase_a_accumulator_overflow_early_stop),
        "phase_a_adaptive_halvings_max": int(phase_a_adaptive_halvings_max),
    }


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


def _compute_incremental_memory_limits(
    num_features: int,
    *,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_ram_bytes: int | None = None,
) -> memory_budget.IncrementalPhaseSplitLimits:
    return memory_budget.compute_incremental_phase_split_limits(
        num_features,
        selected_feature_count=selected_feature_count,
        nameless_feature_count=nameless_feature_count,
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
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

    set_params = getattr(estimator, "set_params", None)
    if callable(set_params):
        try:
            set_params(n_jobs=int(n_jobs))
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
            estimator.n_jobs = int(n_jobs)
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
    if featurizer_version <= 2:
        return NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY
    return NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR


def _resolve_clusterer_name_count_semantics(
    clusterer: Any,
    *,
    strict: bool,
) -> str:
    contract = getattr(clusterer, "feature_contract", None)
    if isinstance(contract, dict):
        contract_value = contract.get("name_counts_last_first_initial_semantics")
        if contract_value in {
            NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
            NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
        }:
            return str(contract_value)
        if contract_value is not None and strict:
            raise ValueError(
                "Invalid clusterer feature_contract['name_counts_last_first_initial_semantics'] "
                f"value: {contract_value!r}"
            )

    featurizer_info = getattr(clusterer, "featurizer_info", None)
    featurizer_version = getattr(featurizer_info, "featurizer_version", None)
    inferred = _name_count_semantics_from_featurizer_version(featurizer_version)
    if inferred is not None:
        return inferred

    if strict:
        raise ValueError(
            "Unable to resolve model name-count semantics from feature_contract or featurizer_version. "
            "Inference requires explicit semantics metadata."
        )
    return NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR


def _apply_dataset_name_count_semantics_for_prediction(
    clusterer: Any,
    dataset: ANDData,
) -> None:
    dataset_mode = str(getattr(dataset, "mode", "")).strip().lower()
    if dataset_mode == "inference":
        desired = _resolve_clusterer_name_count_semantics(clusterer, strict=True)
    else:
        desired = NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR
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

    python_start = time.perf_counter()
    with warnings.catch_warnings():
        suppress_sklearn_feature_name_warnings()
        if num_threads is not None:
            try:
                predictions = classifier.predict_proba(features_2d, num_threads=int(num_threads))[:, 0]
            except TypeError as exc:
                message = str(exc).lower()
                if "unexpected keyword" not in message and "unexpected argument" not in message:
                    raise
                predictions = classifier.predict_proba(features_2d)[:, 0]
        else:
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


def _handle_rust_backend_exception(
    runtime_context: RuntimeContext,
    *,
    strict_message: str,
    exc: Exception,
    fallback_warning: str,
    context_fields: tuple[str, ...] = (),
) -> None:
    details = " ".join((*context_fields, f"run_id={runtime_context.run_id}", f"error={exc}"))
    if stage_uses_rust(runtime_context):
        raise RuntimeError(f"{strict_message} ({details})") from exc
    logger.warning("%s: %s", fallback_warning, exc)


def _rust_with_fallback(
    fn: Callable[[], _TReturn],
    fallback_fn: Callable[[], _TReturn],
    *,
    runtime_context: RuntimeContext,
    label: str,
    context_fields: tuple[str, ...] = (),
    strict_message: str | None = None,
    fallback_warning: str | None = None,
) -> _TReturn:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - native extension optional
        _handle_rust_backend_exception(
            runtime_context,
            strict_message=(strict_message or f"Rust {label} failed in strict rust backend"),
            exc=exc,
            fallback_warning=(fallback_warning or f"Rust {label} failed, falling back to Python"),
            context_fields=context_fields,
        )
        return fallback_fn()


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


def _get_constraint_value(
    dataset: ANDData,
    sig_id_1: str,
    sig_id_2: str,
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
    rust_featurizer: object | None = None,
    use_rust_constraints: bool | None = None,
    runtime_context: RuntimeContext | None = None,
    use_cache: bool = False,
):
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints")
    if use_rust_constraints is None:
        use_rust_constraints = _use_rust_constraints(runtime_context)
    if use_rust_constraints:
        return _rust_with_fallback(
            fn=lambda: get_constraint_rust(
                dataset,
                sig_id_1,
                sig_id_2,
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                featurizer=rust_featurizer,
                runtime_context=runtime_context,
                use_cache=use_cache,
            ),
            fallback_fn=lambda: dataset.get_constraint(
                sig_id_1,
                sig_id_2,
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            ),
            runtime_context=runtime_context,
            label="constraint evaluation",
            strict_message="Rust constraint evaluation failed in strict rust backend",
            fallback_warning="Rust get_constraint failed, falling back to Python",
            context_fields=(f"pair=({sig_id_1}, {sig_id_2})",),
        )
    return dataset.get_constraint(
        sig_id_1,
        sig_id_2,
        dont_merge_cluster_seeds=dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
    )


def _sync_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: RuntimeContext | None = None,
    use_cache: bool = False,
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
            update_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=use_cache)
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

        _rust_with_fallback(
            fn=_sync,
            fallback_fn=lambda: None,
            runtime_context=runtime_context,
            label="cluster seed sync",
            strict_message="Rust cluster seed sync failed in strict rust backend",
            fallback_warning="Rust cluster seed sync failed, falling back to Python",
        )


def _initialize_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
    use_cache: bool = False,
) -> tuple[object | None, bool | None]:
    if not use_default_constraints_as_supervision:
        return None, None

    use_rust_constraints = _use_rust_constraints(runtime_context)
    if not use_rust_constraints:
        return None, False

    rust_featurizer = _rust_with_fallback(
        fn=lambda: _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache),
        fallback_fn=lambda: None,
        runtime_context=runtime_context,
        label="constraint featurizer init",
        strict_message="Rust constraint stage requested but Rust featurizer init failed",
        fallback_warning="Rust featurizer init failed, falling back to Python constraints",
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
    use_cache: bool = False,
) -> _IncrementalConstraintBackend:
    """Build Phase A constraint backend state once for reuse across subblocks."""
    rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        use_cache=use_cache,
    )
    constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    signature_index_by_id: dict[str, int] | None = None
    if constraint_api_mode == "indexed" and rust_featurizer is not None:
        signature_index_by_id = _rust_with_fallback(
            fn=lambda: _build_signature_index_by_id(rust_featurizer),
            fallback_fn=lambda: None,
            runtime_context=runtime_context,
            label="indexed constraint setup",
            strict_message="Rust indexed constraint setup failed in strict rust backend",
            fallback_warning=(
                "Rust indexed constraint setup failed in phase A; disabling Rust constraints and falling back "
                "to Python"
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
    )


def _resolve_constraint_labels_batch(
    dataset: ANDData,
    pair_ids: list[tuple[str, str]],
    *,
    constraint_backend: _IncrementalConstraintBackend | None = None,
    partial_supervision: dict[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    dont_merge_cluster_seeds: bool,
    incremental_dont_use_cluster_seeds: bool,
    rust_featurizer: object | None = None,
    use_rust_constraints: bool | None = None,
    runtime_context: RuntimeContext,
    use_cache: bool = False,
    num_threads: int | None = None,
    constraint_api_mode: str | None = None,
    signature_index_by_id: dict[str, int] | None = None,
) -> tuple[list[float], _ConstraintBatchTelemetry]:
    if constraint_backend is not None:
        rust_featurizer = constraint_backend.rust_featurizer
        use_rust_constraints = constraint_backend.use_rust_constraints
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
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            )
            for s1, s2 in unresolved_pairs
        ]

    if use_rust_constraints and rust_featurizer is not None and mode == "indexed":
        used_python_fallback = False

        def _resolve_values_rust() -> list[float | None]:
            if signature_index_by_id is None:
                raise RuntimeError("Indexed constraint API requested without signature index lookup")
            indexed_pairs = [(signature_index_by_id[s1], signature_index_by_id[s2]) for s1, s2 in unresolved_pairs]
            return get_constraints_matrix_indexed_rust(
                dataset,
                indexed_pairs,
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                num_threads=num_threads,
                featurizer=rust_featurizer,
                runtime_context=runtime_context,
                use_cache=use_cache,
            )

        def _resolve_values_python_fallback() -> list[float | None]:
            nonlocal used_python_fallback
            used_python_fallback = True
            return _resolve_values_python()

        values = _rust_with_fallback(
            fn=_resolve_values_rust,
            fallback_fn=_resolve_values_python_fallback,
            runtime_context=runtime_context,
            label="batch constraint evaluation",
            strict_message="Rust batch constraint evaluation failed in strict rust backend",
            fallback_warning="Rust batch constraint evaluation failed, falling back to Python constraints",
            context_fields=(f"pairs={len(unresolved_pairs)}",),
        )
        if used_python_fallback:
            telemetry.api_mode = "python_fallback"
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


@dataclass(frozen=True)
class _ConstraintSummary:
    pairs_total: int
    chunks_total: int
    partial_supervision_hits: int
    unresolved_pairs: int
    rust_batch_calls: int
    api_mode: str
    elapsed_seconds: float


@dataclass(frozen=True)
class _AccumulatorSummary:
    entries_peak: int
    chunk_features_peak_bytes: int
    pair_buffer_peak_bytes: int
    accumulator_entry_bytes: int
    pair_buffer_entry_bytes: int
    fixed_overhead_bytes: int
    adaptive_halvings: int
    overflow_early_stop: bool


@dataclass(frozen=True)
class _MemoryPredictionSummary:
    contract_version: str
    predicted_peak_delta_bytes: int
    predicted_peak_rss_bytes: int
    predicted_bytes: int
    rss_before_bytes: int
    rss_peak_bytes: int
    rss_after_bytes: int
    observed_peak_delta_bytes: int
    prediction_error_ratio: float
    underpredicted: bool
    rss_source: str


@dataclass(frozen=True)
class _PhaseASeedTelemetry:
    constraints: _ConstraintSummary
    accumulator: _AccumulatorSummary
    memory: _MemoryPredictionSummary
    model_predict_seconds: float


@dataclass
class _PhaseAChunkState:
    constraint_pairs_total: int = 0
    constraint_chunks_total: int = 0
    model_predict_total_seconds: float = 0.0
    accumulator_warned: bool = False
    accumulator_overflow_early_stop: bool = False
    accumulator_entries_peak: int = 0
    accumulator_entries: int = 0
    chunk_features_peak_bytes: int = 0
    chunk_pairs_peak: int = 0
    chunk_pairs: int = 0
    adaptive_halvings: int = 0
    rss_peak_bytes: int = 0


@dataclass
class _PhaseASummaryAccumulator:
    pairs_total: int = 0
    chunks_total: int = 0
    accumulator_peak: int = 0
    accumulator_peak_sample: int = 0
    model_predict_seconds: float = 0.0
    prediction_contract_version: str = "unknown"
    predicted_peak_delta_bytes: int = 0
    predicted_peak_rss_bytes: int = 0
    predicted_bytes: int = 0
    rss_before_bytes: int = 0
    rss_peak_bytes: int = 0
    rss_after_bytes: int = 0
    rss_source: str = "unavailable"
    observed_peak_delta_bytes: int = 0
    prediction_error_ratio: float = 0.0
    underpredicted: bool = False
    chunk_features_peak_bytes: int = 0
    pair_buffer_peak_bytes: int = 0
    accumulator_entry_bytes: int = 0
    pair_buffer_entry_bytes: int = 0
    fixed_overhead_bytes: int = 0
    worst_sample: _PhaseASeedTelemetry | None = None
    accumulator_runtime_calibrated: bool = False
    accumulator_overflow_early_stop: bool = False
    overflow_subblocks: int = 0
    adaptive_halvings_max: int = 0

    def observe(self, sample: _PhaseASeedTelemetry) -> None:
        if sample.accumulator.overflow_early_stop:
            self.accumulator_overflow_early_stop = True
            self.overflow_subblocks += 1
        self.adaptive_halvings_max = max(self.adaptive_halvings_max, sample.accumulator.adaptive_halvings)
        self.pairs_total += sample.constraints.pairs_total
        self.chunks_total += sample.constraints.chunks_total
        self.accumulator_peak = max(self.accumulator_peak, sample.accumulator.entries_peak)
        self.model_predict_seconds += sample.model_predict_seconds
        if self.worst_sample is None:
            self.worst_sample = sample
        else:
            candidate_ratio = sample.memory.prediction_error_ratio
            candidate_observed_delta = sample.memory.observed_peak_delta_bytes
            current_ratio = self.worst_sample.memory.prediction_error_ratio
            current_observed_delta = self.worst_sample.memory.observed_peak_delta_bytes
            if candidate_ratio > current_ratio or (
                candidate_ratio == current_ratio and candidate_observed_delta > current_observed_delta
            ):
                self.worst_sample = sample
        self.underpredicted = self.underpredicted or sample.memory.underpredicted

    def finalize_from_worst_sample(self) -> None:
        if self.worst_sample is None:
            return
        worst = self.worst_sample
        self.prediction_contract_version = worst.memory.contract_version
        self.predicted_peak_delta_bytes = worst.memory.predicted_peak_delta_bytes
        self.predicted_peak_rss_bytes = worst.memory.predicted_peak_rss_bytes
        self.predicted_bytes = worst.memory.predicted_bytes
        self.rss_before_bytes = worst.memory.rss_before_bytes
        self.rss_peak_bytes = worst.memory.rss_peak_bytes
        self.rss_after_bytes = worst.memory.rss_after_bytes
        self.rss_source = worst.memory.rss_source
        self.observed_peak_delta_bytes = worst.memory.observed_peak_delta_bytes
        self.prediction_error_ratio = worst.memory.prediction_error_ratio
        self.accumulator_peak_sample = worst.accumulator.entries_peak
        self.chunk_features_peak_bytes = worst.accumulator.chunk_features_peak_bytes
        self.pair_buffer_peak_bytes = worst.accumulator.pair_buffer_peak_bytes
        self.accumulator_entry_bytes = worst.accumulator.accumulator_entry_bytes
        self.pair_buffer_entry_bytes = worst.accumulator.pair_buffer_entry_bytes
        self.fixed_overhead_bytes = worst.accumulator.fixed_overhead_bytes


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

    @property
    def n_jobs(self) -> int:
        return int(getattr(self, "_n_jobs", 1))

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        n_jobs = max(1, int(value))
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
            dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
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
        labels = self._cluster_one_block(
            block_dict[block_key],
            pairwise_proba,
            effective_cluster_model_params,
            dataset,
            all_disallow_signature_ids,
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
            use_cache=self.use_cache,
        )

        telemetry = _ConstraintTelemetryAccumulator()
        pair_chunk_size = max(1, int(self.batch_size))

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
    ):
        block_size = len(signatures)
        if block_size <= 1:
            return
        block_pair_count = int(block_size * (block_size - 1) / 2)
        pair_chunk_size = max(1, int(self.batch_size))
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
    ):
        if runtime_context is None:
            runtime_context = build_runtime_context("constraints")
        if not stage_uses_rust(runtime_context):
            raise ValueError("Rust chunk helper is only valid when runtime_context resolves to rust backend")

        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
        )
        rust_featurizer = constraint_backend.rust_featurizer
        constraint_api_mode = constraint_backend.constraint_api_mode
        signature_index_by_id = constraint_backend.signature_index_by_id

        telemetry = _ConstraintTelemetryAccumulator()
        pair_chunk_size = max(1, int(self.batch_size))
        used_fused_path = False
        use_fused_block_api = bool(
            self.use_default_constraints_as_supervision
            and not self.use_cache
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
                            use_cache=self.use_cache,
                        )
                    except Exception as exc:
                        _handle_rust_backend_exception(
                            runtime_context,
                            strict_message="Rust fused block constraint evaluation failed in strict rust backend",
                            exc=exc,
                            fallback_warning=(
                                "Rust fused block constraint evaluation failed; falling back to non-fused chunk path"
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
                    # Fused path disabled after runtime failure; continue with fallback for this and later blocks.
                    yield from self._yield_non_fused_chunks(
                        block_key=block_key,
                        signatures=signatures,
                        dataset=dataset,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        constraint_backend=constraint_backend,
                        telemetry=telemetry,
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
    ) -> tuple[np.ndarray, float]:
        if chunk.block_signature_indices is not None and chunk.pair_ids is None:
            if self.use_cache:
                raise RuntimeError("Fused Rust chunk path does not support use_cache=True")
            try:
                rust_featurizer = _get_rust_featurizer(
                    dataset,
                    runtime_context=runtime_context,
                    use_cache=self.use_cache,
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
                    use_cache=self.use_cache,
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
                        use_cache=self.use_cache,
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
                use_cache=self.use_cache,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=self.nameless_featurizer_info,
                runtime_context=runtime_context,
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
    ):
        chunk_count = 0
        helper_output = self._distance_matrix_chunk_helper_rust(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
        )
        for chunk in helper_output:
            batch_predictions, batch_seconds = self._predict_distance_matrix_chunk(
                chunk,
                dataset,
                runtime_context,
                batch_label=f"chunk_{chunk_count}",
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
    ):
        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
        )
        batch_num = 0
        num_batches = math.ceil(num_pairs / self.batch_size) if num_pairs > 0 else 0
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
                if count == self.batch_size:
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
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
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
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)

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
            _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)

        logger.info(f"Finished subblocked predict incremental. Here's how long each subblock took: {predict_times}")
        dataset.cluster_seeds_require = cluster_seeds_require_original
        _bump_cluster_seeds_version(dataset)
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
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
        )
        pred_clusters = self._predict_subblocked_single_letter_incremental_groups(
            block_dict_single_letter_first_names,
            pred_clusters=pred_clusters,
            desired_memory_use=effective_desired_memory_use,
            dataset=dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
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
        batching_threshold: int
            If the number of signatures in a block is above this number, we will use subblocking on the block.
            This means that the single-letter first names will be sent through via predict_incremental.
            Defaults to None, which means no batching occurs
        desired_memory_use: int
            If batching_threshold is not None, then this is the desired memory use for predict_incremental.
            The units of this are the same as the units of batching_threshold -> number of signatures.
            If None, then using batching_threshold * batching_threshold as the desired memory use.


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
            )
            end = time.time()
            total_predict_time = end - start
            logger.info(f"Finished predict on full blocks. Time taken: {total_predict_time}")

        return dict(pred_clusters), dists

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

        cluster_model = self.set_params(cluster_model_params, clone_flag=True)
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
                labels = self._cluster_one_block(
                    block_dict[block_key],
                    dists[block_key],
                    effective_cluster_model_params,
                    dataset,
                    all_disallow_signature_ids,
                )
                for signature, label in zip(block_dict[block_key], labels, strict=True):
                    pred_clusters[block_key + "_" + str(label)].append(signature)
            return dict(pred_clusters), dists

        # fused path: build one block's matrix, cluster it, free it, repeat
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)

        prev_block_key = ""
        pairwise_proba: np.ndarray | None = None
        seen_block_keys: set[str] = set()
        num_pairs = sum(len(sigs) * (len(sigs) - 1) // 2 for sigs in block_dict.values())
        model_predict_seconds = 0.0
        use_rust_blockwise = stage_uses_rust(runtime_context)
        if use_rust_blockwise:
            for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
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
                )
                if len(reclustered_output) <= 1:
                    continue
                for i, new_cluster_of_signatures in enumerate(reclustered_output.values()):
                    new_cluster_num = str(altered_cluster_num) + f"_{i}"
                    recluster_map[new_cluster_num] = altered_cluster_num
                    for reclustered_signature_id in new_cluster_of_signatures:
                        cluster_seeds_require[reclustered_signature_id] = new_cluster_num

        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse

    def _process_phase_a_chunk(
        self,
        chunk_pairs_buffer: list[tuple[str, str, float]],
        *,
        dataset: ANDData,
        cluster_seeds_require: dict[str, int | str],
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
        chunk_state: _PhaseAChunkState,
        chunk_limits: IncrementalChunkLimits | None,
        runtime_context: RuntimeContext,
        total_ram_for_phase: int | None,
        rss_before_bytes: int,
    ) -> None:
        if len(chunk_pairs_buffer) == 0:
            return

        chunk_state.chunk_pairs_peak = max(chunk_state.chunk_pairs_peak, len(chunk_pairs_buffer))
        chunk_features, chunk_labels, chunk_nameless_features = many_pairs_featurize(
            chunk_pairs_buffer,
            dataset,
            self.featurizer_info,
            self.n_jobs,
            use_cache=self.use_cache,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_for_phase,
        )
        chunk_feature_bytes = int(getattr(chunk_features, "nbytes", 0))
        if chunk_nameless_features is not None:
            chunk_feature_bytes += int(getattr(chunk_nameless_features, "nbytes", 0))
        chunk_state.chunk_features_peak_bytes = max(chunk_state.chunk_features_peak_bytes, chunk_feature_bytes)
        chunk_predictions, chunk_model_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            chunk_features,
            chunk_labels,
            chunk_nameless_features,
            f"phase_a_chunk_{chunk_state.constraint_chunks_total + 1}",
            num_threads=self.n_jobs,
            runtime_context=runtime_context,
        )
        chunk_state.model_predict_total_seconds += chunk_model_seconds

        for signature_pair, dist in zip(chunk_pairs_buffer, chunk_predictions, strict=True):
            unassigned_signature, assigned_signature, _ = signature_pair
            if assigned_signature not in cluster_seeds_require:
                continue
            cluster_id = cluster_seeds_require[assigned_signature]
            cluster_sum_count = signature_to_cluster_sum_count.setdefault(unassigned_signature, {})
            if cluster_id not in cluster_sum_count:
                cluster_sum_count[cluster_id] = [0.0, 0]
                chunk_state.accumulator_entries += 1
            total_count = cluster_sum_count[cluster_id]
            total_count[0] = float(total_count[0]) + float(dist)
            total_count[1] = int(total_count[1]) + 1

        chunk_state.constraint_pairs_total += len(chunk_pairs_buffer)
        chunk_state.constraint_chunks_total += 1
        chunk_state.accumulator_entries_peak = max(
            chunk_state.accumulator_entries_peak,
            int(chunk_state.accumulator_entries),
        )

        if (
            not chunk_state.accumulator_warned
            and chunk_limits is not None
            and chunk_state.accumulator_entries >= int(chunk_limits["accumulator_warn"])
        ):
            chunk_state.accumulator_warned = True
            logger.warning(
                "Phase A accumulator approaching limit: entries=%d warn=%d max=%d run_id=%s",
                chunk_state.accumulator_entries,
                int(chunk_limits["accumulator_warn"]),
                int(chunk_limits["accumulator_max"]),
                runtime_context.run_id,
            )

        if chunk_limits is not None and chunk_state.accumulator_entries > int(chunk_limits["accumulator_max"]):
            clusters_touched_avg = float(chunk_state.accumulator_entries) / float(
                max(1, len(signature_to_cluster_sum_count))
            )
            logger.warning(
                "Phase A accumulator exceeded limit: "
                "entries=%d max=%d warn=%d "
                "chunk_pairs=%d available_bytes=%d current_rss_bytes=%d "
                "unassigned_signatures=%d clusters_touched_avg=%.3f run_id=%s. "
                "Stopping Phase A early; remaining unassigned signatures will proceed "
                "with partial seed distances.",
                chunk_state.accumulator_entries,
                int(chunk_limits["accumulator_max"]),
                int(chunk_limits["accumulator_warn"]),
                int(chunk_limits["chunk_pairs"]),
                int(chunk_limits["available_bytes"]),
                int(chunk_limits["current_rss_bytes"]),
                len(signature_to_cluster_sum_count),
                clusters_touched_avg,
                runtime_context.run_id,
            )
            chunk_state.accumulator_overflow_early_stop = True
            return

        # Fallback accumulator bound when chunk_limits is None.
        if chunk_limits is None and chunk_state.accumulator_entries > memory_budget.FALLBACK_ACCUMULATOR_MAX_ENTRIES:
            logger.warning(
                "Phase A accumulator exceeded fallback limit: "
                "entries=%d max=%d run_id=%s. "
                "Stopping Phase A early; remaining unassigned signatures will proceed "
                "with partial seed distances. "
                "Pass total_ram_bytes to enable proper memory budgeting.",
                chunk_state.accumulator_entries,
                memory_budget.FALLBACK_ACCUMULATOR_MAX_ENTRIES,
                runtime_context.run_id,
            )
            chunk_state.accumulator_overflow_early_stop = True
            return

        if total_ram_for_phase is not None:
            rss_now, _ = memory_budget.current_rss_bytes_best_effort(total_ram_for_phase)
            chunk_state.rss_peak_bytes = max(chunk_state.rss_peak_bytes, rss_now)

        # Fix #3: adaptive chunking — halve chunk_pairs if observed RSS delta exceeds prediction.
        if total_ram_for_phase is not None and chunk_state.adaptive_halvings < 3:
            acc_entry_bytes = int(memory_budget.INCREMENTAL_ACCUMULATOR_ENTRY_BYTES)
            if chunk_limits is not None and "accumulator_entry_bytes" in chunk_limits:
                acc_entry_bytes = int(chunk_limits["accumulator_entry_bytes"])
            current_predicted_delta = (
                chunk_state.chunk_features_peak_bytes
                + chunk_state.accumulator_entries_peak * acc_entry_bytes
                + chunk_state.chunk_pairs_peak * int(memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES)
                + int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES)
            )
            current_observed_delta = max(0, chunk_state.rss_peak_bytes - rss_before_bytes)
            if current_predicted_delta > 0 and current_observed_delta > current_predicted_delta * 1.2:
                chunk_state.chunk_pairs = max(1, chunk_state.chunk_pairs // 2)
                chunk_state.adaptive_halvings += 1
                logger.warning(
                    "Phase A adaptive chunking: observed_delta=%d > predicted_delta=%d * 1.2; "
                    "halving chunk_pairs to %d (halving %d/3) run_id=%s",
                    current_observed_delta,
                    current_predicted_delta,
                    chunk_state.chunk_pairs,
                    chunk_state.adaptive_halvings,
                    runtime_context.run_id,
                )

        del chunk_features
        del chunk_nameless_features
        del chunk_predictions
        del chunk_labels

    def _phase_a_seed_distances(
        self,
        unassigned_signature_ids: list[str],
        cluster_seeds_require: dict[str, int | str],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
        chunk_pairs: int,
        chunk_limits: IncrementalChunkLimits | None,
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
        constraint_backend: _IncrementalConstraintBackend | None = None,
    ) -> _PhaseASeedTelemetry:
        if chunk_pairs <= 0:
            raise ValueError("chunk_pairs must be positive")

        pair_buffer: list[tuple[str, str, float]] = []
        pair_id_buffer: list[tuple[str, str]] = []
        if constraint_backend is None:
            constraint_backend = _build_incremental_constraint_backend(
                dataset,
                use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                runtime_context=runtime_context,
                use_cache=self.use_cache,
            )
        constraint_telemetry = _ConstraintTelemetryAccumulator()
        total_ram_for_phase = total_ram_bytes
        rss_source = "unavailable"
        rss_before_bytes = 0
        rss_peak_bytes = 0

        if total_ram_for_phase is None and chunk_limits is not None:
            total_ram_for_phase = int(chunk_limits["total_ram_bytes"])

        if total_ram_for_phase is not None:
            rss_before_bytes, rss_source = memory_budget.current_rss_bytes_best_effort(total_ram_for_phase)
            rss_peak_bytes = rss_before_bytes

        chunk_state = _PhaseAChunkState(
            chunk_pairs=int(chunk_pairs),
            rss_peak_bytes=int(rss_peak_bytes),
        )

        for unassigned_signature in unassigned_signature_ids:
            if chunk_state.accumulator_overflow_early_stop:
                break
            for signature in cluster_seeds_require.keys():
                pair_id_buffer.append((unassigned_signature, signature))
                if len(pair_id_buffer) >= chunk_state.chunk_pairs:
                    labels, batch_telemetry = self._resolve_constraint_batch(
                        dataset,
                        pair_id_buffer,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                        incremental_dont_use_cluster_seeds=False,
                        constraint_backend=constraint_backend,
                    )
                    _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
                    pair_buffer = [
                        (sig_id_1, sig_id_2, label)
                        for (sig_id_1, sig_id_2), label in zip(pair_id_buffer, labels, strict=True)
                    ]
                    self._process_phase_a_chunk(
                        pair_buffer,
                        dataset=dataset,
                        cluster_seeds_require=cluster_seeds_require,
                        signature_to_cluster_sum_count=signature_to_cluster_sum_count,
                        chunk_state=chunk_state,
                        chunk_limits=chunk_limits,
                        runtime_context=runtime_context,
                        total_ram_for_phase=total_ram_for_phase,
                        rss_before_bytes=rss_before_bytes,
                    )
                    pair_buffer = []
                    pair_id_buffer = []
                    if chunk_state.accumulator_overflow_early_stop:
                        break

        if len(pair_id_buffer) > 0 and not chunk_state.accumulator_overflow_early_stop:
            labels, batch_telemetry = self._resolve_constraint_batch(
                dataset,
                pair_id_buffer,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                incremental_dont_use_cluster_seeds=False,
                constraint_backend=constraint_backend,
            )
            _accumulate_constraint_telemetry(constraint_telemetry, batch_telemetry)
            pair_buffer = [
                (sig_id_1, sig_id_2, label) for (sig_id_1, sig_id_2), label in zip(pair_id_buffer, labels, strict=True)
            ]
            self._process_phase_a_chunk(
                pair_buffer,
                dataset=dataset,
                cluster_seeds_require=cluster_seeds_require,
                signature_to_cluster_sum_count=signature_to_cluster_sum_count,
                chunk_state=chunk_state,
                chunk_limits=chunk_limits,
                runtime_context=runtime_context,
                total_ram_for_phase=total_ram_for_phase,
                rss_before_bytes=rss_before_bytes,
            )

        rss_after_bytes = rss_before_bytes
        if total_ram_for_phase is not None:
            rss_after_bytes, _ = memory_budget.current_rss_bytes_best_effort(total_ram_for_phase)
        accumulator_entry_bytes = int(memory_budget.INCREMENTAL_ACCUMULATOR_ENTRY_BYTES)
        if chunk_limits is not None and "accumulator_entry_bytes" in chunk_limits:
            accumulator_entry_bytes = int(chunk_limits["accumulator_entry_bytes"])
        phase_a_pair_buffer_peak_bytes = int(chunk_state.chunk_pairs_peak) * int(
            memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES
        )
        phase_a_predicted_peak_delta_bytes = (
            int(chunk_state.chunk_features_peak_bytes)
            + int(chunk_state.accumulator_entries_peak) * accumulator_entry_bytes
            + int(phase_a_pair_buffer_peak_bytes)
            + int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES)
        )
        phase_a_prediction = memory_budget.summarize_prediction_accuracy(
            stage_name="phase_a_seed_distances",
            predicted_peak_delta_bytes=phase_a_predicted_peak_delta_bytes,
            rss_before_bytes=rss_before_bytes,
            rss_peak_bytes=chunk_state.rss_peak_bytes,
            rss_after_bytes=rss_after_bytes,
        )
        logger.info(
            "Telemetry: constraint_batch stage=phase_a_seed_distances total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            int(chunk_state.constraint_pairs_total),
            int(constraint_telemetry.partial_supervision_hits),
            int(constraint_telemetry.unresolved_pairs),
            int(constraint_telemetry.rust_batch_call_count),
            constraint_telemetry.api_mode_summary,
            float(constraint_telemetry.elapsed_seconds),
            runtime_context.run_id,
        )

        return _PhaseASeedTelemetry(
            constraints=_ConstraintSummary(
                pairs_total=chunk_state.constraint_pairs_total,
                chunks_total=chunk_state.constraint_chunks_total,
                partial_supervision_hits=constraint_telemetry.partial_supervision_hits,
                unresolved_pairs=constraint_telemetry.unresolved_pairs,
                rust_batch_calls=constraint_telemetry.rust_batch_call_count,
                api_mode=constraint_telemetry.api_mode_summary,
                elapsed_seconds=constraint_telemetry.elapsed_seconds,
            ),
            accumulator=_AccumulatorSummary(
                entries_peak=chunk_state.accumulator_entries_peak,
                chunk_features_peak_bytes=chunk_state.chunk_features_peak_bytes,
                pair_buffer_peak_bytes=phase_a_pair_buffer_peak_bytes,
                accumulator_entry_bytes=accumulator_entry_bytes,
                pair_buffer_entry_bytes=int(memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES),
                fixed_overhead_bytes=int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES),
                adaptive_halvings=chunk_state.adaptive_halvings,
                overflow_early_stop=chunk_state.accumulator_overflow_early_stop,
            ),
            memory=_MemoryPredictionSummary(
                contract_version=str(phase_a_prediction["prediction_contract_version"]),
                predicted_peak_delta_bytes=int(phase_a_prediction["predicted_peak_delta_bytes"]),
                predicted_peak_rss_bytes=int(phase_a_prediction["predicted_peak_rss_bytes"]),
                # Backward-compatible alias; keep this while telemetry logs still emit predicted_bytes.
                predicted_bytes=int(phase_a_prediction["predicted_bytes"]),
                rss_before_bytes=int(phase_a_prediction["rss_before_bytes"]),
                rss_peak_bytes=int(phase_a_prediction["rss_peak_bytes"]),
                rss_after_bytes=int(phase_a_prediction["rss_after_bytes"]),
                observed_peak_delta_bytes=int(phase_a_prediction["observed_peak_delta_bytes"]),
                prediction_error_ratio=float(phase_a_prediction["prediction_error_ratio"]),
                underpredicted=bool(phase_a_prediction["underpredicted"]),
                rss_source=str(rss_source),
            ),
            model_predict_seconds=chunk_state.model_predict_total_seconds,
        )

    def _convert_sum_count_to_average_distances(
        self,
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
    ) -> dict[str, dict[int | str, tuple[float, int]]]:
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]] = defaultdict(dict)
        for signature_id, cluster_sum_count in signature_to_cluster_sum_count.items():
            for cluster_id, sum_count in cluster_sum_count.items():
                total = float(sum_count[0])
                count = int(sum_count[1])
                if count <= 0:
                    continue
                signature_to_cluster_to_average_dist[signature_id][cluster_id] = (total / float(count), count)
        return signature_to_cluster_to_average_dist

    def _run_incremental_phases_bcd(
        self,
        unassigned_signature_ids: list[str],
        dataset: ANDData,
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]],
        cluster_seeds_require: dict[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
    ) -> dict[str, list[str]]:
        # NEW!
        # First cluster the unassigned signatures, then decide which resulting unassigned
        # clusters should merge with existing seeded clusters.
        logger.info("Batch clustering the unassigned signatures")
        incremental_only_clusters, _ = self.predict_helper(
            {"incremental_unassigned": unassigned_signature_ids},
            dataset,
            partial_supervision=partial_supervision,
            runtime_context=runtime_context,
        )

        logger.info(
            "Made %d clusters out of %d unassigned signatures",
            len(incremental_only_clusters),
            len(unassigned_signature_ids),
        )

        # Average over Phase A signature-to-seed distances at the pre-cluster level.
        # This is equivalent to computing average distance between each unassigned cluster
        # and each assigned cluster, then broadcasting that score back to member signatures.
        cluster_ids = sorted(set(cluster_seeds_require.values()), key=lambda cluster_id: str(cluster_id))
        for incremental_cluster_signature_ids in incremental_only_clusters.values():
            for cluster_id in cluster_ids:
                dists = []
                for signature in incremental_cluster_signature_ids:
                    cluster_entry = signature_to_cluster_to_average_dist.get(signature, {}).get(cluster_id)
                    if cluster_entry is None:
                        continue
                    if int(cluster_entry[1]) <= 0:
                        continue
                    dists.append(float(cluster_entry[0]))
                if len(dists) == 0:
                    continue
                out = (float(np.mean(dists)), len(dists))
                for signature in incremental_cluster_signature_ids:
                    signature_to_cluster_to_average_dist.setdefault(signature, {})[cluster_id] = out

        logger.info("Assigning unassigned signatures for incremental clustering")
        pred_clusters = defaultdict(list)
        singleton_signatures = []
        for signature_id, cluster_id in dataset.cluster_seeds_require.items():
            pred_clusters[f"{cluster_id}"].append(signature_id)
        for unassigned_signature in unassigned_signature_ids:
            cluster_dists = signature_to_cluster_to_average_dist.get(unassigned_signature, {})
            best_cluster_id = None
            best_dist = float("inf")
            for cluster_id, (average_dist, _) in cluster_dists.items():
                if average_dist < best_dist and average_dist < self.cluster_model.eps:
                    best_cluster_id = cluster_id
                    best_dist = average_dist
            if best_cluster_id is not None:
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
                                prefix = same_prefix_tokens(first_assigned, first_unassigned)
                                known_alias = (first_assigned, first_unassigned) in dataset.name_tuples

                                if prefix or known_alias:
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
            else:
                singleton_signatures.append(unassigned_signature)

        # all remaining singletons are reclustered together
        if len(singleton_signatures) > 0:
            logger.info("Clustering together the still unassigned signatures")
            reclustered_output, _ = self.predict_helper(
                {"block": singleton_signatures},
                dataset,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
            )
            new_cluster_id = _next_unused_cluster_id(pred_clusters, int(dataset.max_seed_cluster_id or 0))
            for new_cluster in reclustered_output.values():
                new_cluster_id = _next_unused_cluster_id(pred_clusters, new_cluster_id)
                pred_clusters[str(new_cluster_id)] = new_cluster
                new_cluster_id += 1
        logger.info("Done. Returning incrementally predicted clusters")
        # end NEW!
        return dict(pred_clusters)

    def _phase_split_subblock_fallback(
        self,
        block_signatures: list[str],
        subblocks: dict[str, list[str]],
        dataset: ANDData,
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]],
        cluster_seeds_require: dict[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
    ) -> dict[str, list[str]]:
        original_seed_sigs = set(dataset.cluster_seeds_require.keys())
        original_cluster_ids = set(str(cid) for cid in dataset.cluster_seeds_require.values())
        predict_times: dict[str, float] = {}
        merged_sig_to_cluster: dict[str, int | str] = {}
        next_new_cluster_id = _next_unused_cluster_id({}, int(dataset.max_seed_cluster_id or 0))

        for subblock_key, subblock_signatures in sorted(subblocks.items()):
            subblock_unassigned = [s for s in subblock_signatures if s not in cluster_seeds_require]
            if len(subblock_unassigned) == 0:
                continue
            start_predict_time = time.time()
            subblock_distances = {
                signature_id: dict(signature_to_cluster_to_average_dist.get(signature_id, {}))
                for signature_id in subblock_unassigned
            }
            subblock_result = self._run_incremental_phases_bcd(
                subblock_unassigned,
                dataset,
                subblock_distances,
                cluster_seeds_require,
                recluster_map,
                cluster_seeds_require_inverse,
                prevent_new_incompatibilities,
                partial_supervision,
                runtime_context,
            )
            predict_times[subblock_key] = time.time() - start_predict_time

            subblock_non_seed = set(subblock_signatures) - original_seed_sigs
            local_new_id_remap: dict[str, str] = {}
            for cluster_id_str, cluster_sigs in subblock_result.items():
                for sig in cluster_sigs:
                    if sig not in subblock_non_seed:
                        continue
                    if cluster_id_str in original_cluster_ids:
                        merged_sig_to_cluster[sig] = cluster_id_str
                    else:
                        if cluster_id_str not in local_new_id_remap:
                            local_new_id_remap[cluster_id_str] = str(next_new_cluster_id)
                            next_new_cluster_id += 1
                        merged_sig_to_cluster[sig] = local_new_id_remap[cluster_id_str]

        pred_clusters_final: dict[str, list[str]] = defaultdict(list)
        for sig_id, cluster_id in dataset.cluster_seeds_require.items():
            pred_clusters_final[str(cluster_id)].append(sig_id)
        for sig_id, cluster_id in merged_sig_to_cluster.items():
            pred_clusters_final[str(cluster_id)].append(sig_id)

        predicted_signatures = {sig_id for sigs in pred_clusters_final.values() for sig_id in sigs}
        missing_signatures = [sig_id for sig_id in block_signatures if sig_id not in predicted_signatures]
        if len(missing_signatures) > 0:
            logger.error(
                "Subblock-local phase-split fallback produced incomplete signature coverage; "
                "adding singleton fallbacks. "
                "missing=%d total_block_signatures=%d",
                len(missing_signatures),
                len(block_signatures),
            )
            for missing_signature in missing_signatures:
                next_new_cluster_id = _next_unused_cluster_id(pred_clusters_final, next_new_cluster_id)
                pred_clusters_final[str(next_new_cluster_id)] = [missing_signature]
                next_new_cluster_id += 1

        logger.warning(
            "Phase-split fallback completed with subblock-local Phase B/C/D; "
            "results may diverge from monolithic. subblocks=%d predict_times=%s",
            len(subblocks),
            predict_times,
        )
        return dict(pred_clusters_final)

    def _predict_incremental_phase_split(
        self,
        block_signatures: list[str],
        dataset: ANDData,
        prevent_new_incompatibilities: bool,
        batching_threshold: int,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        logger.info("Phase-split incremental enabled")
        all_unassigned = [
            signature_id for signature_id in block_signatures if signature_id not in dataset.cluster_seeds_require
        ]

        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = self._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
        )
        subblocks = make_subblocks(block_signatures, dataset, maximum_size=batching_threshold)

        selected_count = _count_selected_features(self.featurizer_info)
        nameless_count = (
            _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
        )
        chunk_limits = _compute_incremental_memory_limits(
            self.featurizer_info.number_of_features,
            selected_feature_count=selected_count,
            nameless_feature_count=nameless_count,
            total_ram_bytes=total_ram_bytes,
        )
        chunk_pairs = int(chunk_limits["chunk_pairs"])
        logger.info(
            "Phase-split Phase A chunking: chunk_pairs=%d total_ram=%d total_ram_source=%s "
            "rss=%d rss_source=%s available=%d effective_available_fraction=%.3f "
            "accumulator_warn=%d accumulator_max=%d",
            chunk_pairs,
            int(chunk_limits["total_ram_bytes"]),
            str(chunk_limits["total_ram_source"]),
            int(chunk_limits["current_rss_bytes"]),
            str(chunk_limits["current_rss_source"]),
            int(chunk_limits["available_bytes"]),
            float(chunk_limits.get("effective_available_fraction", 0.0)),
            int(chunk_limits["accumulator_warn"]),
            int(chunk_limits["accumulator_max"]),
        )

        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]] = defaultdict(dict)
        phase_a = _PhaseASummaryAccumulator()

        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
        )

        for subblock_signatures in subblocks.values():
            subblock_unassigned = [
                signature_id for signature_id in subblock_signatures if signature_id not in cluster_seeds_require
            ]
            if len(subblock_unassigned) == 0:
                continue
            phase_a_telemetry = self._phase_a_seed_distances(
                subblock_unassigned,
                cluster_seeds_require,
                dataset,
                partial_supervision,
                signature_to_cluster_sum_count,
                chunk_pairs,
                chunk_limits,
                runtime_context,
                total_ram_bytes=int(chunk_limits["total_ram_bytes"]),
                constraint_backend=constraint_backend,
            )
            phase_a.observe(phase_a_telemetry)

            # Runtime accumulator calibration: after the first subblock with meaningful data,
            # derive the effective bytes/entry from the observed telemetry and recalibrate
            # the accumulator budget for subsequent subblocks.
            if (
                not phase_a.accumulator_runtime_calibrated
                and phase_a_telemetry.accumulator.entries_peak > 100
                and phase_a_telemetry.memory.observed_peak_delta_bytes > 0
            ):
                observed_delta = phase_a_telemetry.memory.observed_peak_delta_bytes
                modeled_non_accum = (
                    phase_a_telemetry.accumulator.chunk_features_peak_bytes
                    + phase_a_telemetry.accumulator.pair_buffer_peak_bytes
                    + phase_a_telemetry.accumulator.fixed_overhead_bytes
                )
                residual = observed_delta - modeled_non_accum
                entries = phase_a_telemetry.accumulator.entries_peak
                if residual > 0 and entries > 0:
                    effective_entry_bytes = float(residual) / float(entries)
                    configured_entry_bytes = int(chunk_limits.get("accumulator_entry_bytes", 200))
                    # Only recalibrate if the observed value is significantly higher (>20%).
                    if effective_entry_bytes > configured_entry_bytes * 1.2:
                        new_entry_bytes = int(effective_entry_bytes * 1.1)  # 10% safety margin
                        chunk_limits = dict(chunk_limits)
                        chunk_limits["accumulator_entry_bytes"] = new_entry_bytes
                        accum_budget = int(chunk_limits["accumulator_budget_bytes"])
                        chunk_limits["accumulator_max"] = max(1, accum_budget // new_entry_bytes)
                        chunk_limits["accumulator_warn"] = max(1, chunk_limits["accumulator_max"] // 5)
                        logger.info(
                            "Runtime accumulator calibration: effective_entry_bytes=%.1f "
                            "configured=%d -> recalibrated=%d accumulator_max=%d run_id=%s",
                            effective_entry_bytes,
                            configured_entry_bytes,
                            new_entry_bytes,
                            chunk_limits["accumulator_max"],
                            runtime_context.run_id,
                        )
                phase_a.accumulator_runtime_calibrated = True

        phase_a.finalize_from_worst_sample()

        logger.info(
            "Telemetry: phase_split_phase_a_overflow overflow_early_stop=%s overflow_subblocks=%d "
            "accumulator_entries_peak=%d accumulator_max=%d run_id=%s",
            phase_a.accumulator_overflow_early_stop,
            phase_a.overflow_subblocks,
            phase_a.accumulator_peak,
            int(chunk_limits["accumulator_max"]),
            runtime_context.run_id,
        )
        logger.info(
            "Telemetry: phase_split_phase_a constraints_pairs_total=%d constraint_chunks_total=%d "
            "accumulator_entries_peak=%d accumulator_entries_peak_sample=%d "
            "chunk_features_peak_bytes=%d phase_a_pair_buffer_peak_bytes=%d "
            "accumulator_entry_bytes=%d phase_a_pair_buffer_entry_bytes=%d "
            "phase_a_fixed_overhead_bytes=%d model_predict_seconds=%.3f "
            "prediction_contract_version=%s predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d "
            "predicted_bytes=%d rss_before_bytes=%d rss_peak_bytes=%d rss_after_bytes=%d "
            "observed_peak_delta_bytes=%d prediction_error_ratio=%.3f underpredicted=%s rss_source=%s",
            phase_a.pairs_total,
            phase_a.chunks_total,
            phase_a.accumulator_peak,
            phase_a.accumulator_peak_sample,
            phase_a.chunk_features_peak_bytes,
            phase_a.pair_buffer_peak_bytes,
            phase_a.accumulator_entry_bytes,
            phase_a.pair_buffer_entry_bytes,
            phase_a.fixed_overhead_bytes,
            phase_a.model_predict_seconds,
            phase_a.prediction_contract_version,
            phase_a.predicted_peak_delta_bytes,
            phase_a.predicted_peak_rss_bytes,
            phase_a.predicted_bytes,
            phase_a.rss_before_bytes,
            phase_a.rss_peak_bytes,
            phase_a.rss_after_bytes,
            phase_a.observed_peak_delta_bytes,
            phase_a.prediction_error_ratio,
            phase_a.underpredicted,
            phase_a.rss_source,
        )

        signature_to_cluster_to_average_dist = self._convert_sum_count_to_average_distances(
            signature_to_cluster_sum_count
        )
        del signature_to_cluster_sum_count
        memory_budget.gc_collect_and_log("phase_a")

        unassigned_count = len(all_unassigned)
        recluster_bytes = unassigned_count * (unassigned_count - 1) // 2 * 8
        # Phase B budget: by this point Phase A chunk features are freed, and the original
        # accumulator has been replaced by the converted distance dict. Available memory is
        # the freed chunk budget + freed accumulator budget - the still-live converted dict.
        accumulator_entry_bytes_for_budget = int(chunk_limits.get("accumulator_entry_bytes", 200))
        estimated_converted_dict_bytes = phase_a.accumulator_peak * accumulator_entry_bytes_for_budget
        phase_b_budget_bytes = max(
            1,
            int(chunk_limits["chunk_budget_bytes"])
            + int(chunk_limits["accumulator_budget_bytes"])
            - estimated_converted_dict_bytes,
        )
        if recluster_bytes > phase_b_budget_bytes:
            logger.warning(
                "Phase B over budget (%d bytes > %d); using subblock-local Phase B/C/D fallback.",
                recluster_bytes,
                phase_b_budget_bytes,
            )
            fallback_clusters = self._phase_split_subblock_fallback(
                block_signatures,
                subblocks,
                dataset,
                signature_to_cluster_to_average_dist,
                cluster_seeds_require,
                recluster_map,
                cluster_seeds_require_inverse,
                prevent_new_incompatibilities,
                partial_supervision,
                runtime_context,
            )
            logger.info(
                "Telemetry: phase_split_phase_b mode=subblock_local required_bytes=%d budget_bytes=%d",
                recluster_bytes,
                phase_b_budget_bytes,
            )
            return _build_incremental_result(
                fallback_clusters,
                phase_b_mode="subblock_local",
                phase_b_budget_bytes=phase_b_budget_bytes,
                phase_b_required_bytes=recluster_bytes,
                phase_a_accumulator_overflow_early_stop=phase_a.accumulator_overflow_early_stop,
                phase_a_adaptive_halvings_max=phase_a.adaptive_halvings_max,
            )

        exact_clusters = self._run_incremental_phases_bcd(
            all_unassigned,
            dataset,
            signature_to_cluster_to_average_dist,
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
        )
        logger.info(
            "Telemetry: phase_split_phase_b mode=exact required_bytes=%d budget_bytes=%d",
            recluster_bytes,
            phase_b_budget_bytes,
        )
        return _build_incremental_result(
            exact_clusters,
            phase_b_mode="exact",
            phase_b_budget_bytes=phase_b_budget_bytes,
            phase_b_required_bytes=recluster_bytes,
            phase_a_accumulator_overflow_early_stop=phase_a.accumulator_overflow_early_stop,
            phase_a_adaptive_halvings_max=phase_a.adaptive_halvings_max,
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
            If there are more unassigned signatures than this number,
            they will be predicted in batches of this size. This is to prevent OOM errors.
            Defaults to None, which means no batching occurs
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        total_ram_bytes: Optional[int]
            Optional explicit RAM budget for incremental phase-split memory-limit derivation.
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
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
        if partial_supervision is None:
            partial_supervision = {}
        if batching_threshold is None or len(block_signatures) <= batching_threshold:
            incremental_result = self._predict_incremental_helper(
                block_signatures,
                dataset,
                prevent_new_incompatibilities=prevent_new_incompatibilities,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )
            return dict(incremental_result["clusters"]) if return_clusters_only else incremental_result

        assert batching_threshold > 0, "Batching threshold must be positive"
        if len(dataset.cluster_seeds_require) == 0:
            logger.info(
                "No cluster seeds provided for subblocked incremental; "
                "falling back to monolithic incremental helper for partition parity."
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

        incremental_result = self._predict_incremental_phase_split(
            block_signatures,
            dataset,
            prevent_new_incompatibilities,
            batching_threshold,
            partial_supervision,
            runtime_context,
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
        )

        logger.info("Getting name constraints")
        all_pairs: list[tuple[str, str, float]] = []
        unassigned_signature_ids: list[str] = []
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
        )
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0))
        )
        assigned_signature_ids: list[str] = list(cluster_seeds_require.keys())
        pair_chunk_size = max(1, int(self.batch_size))
        constraint_telemetry = _ConstraintTelemetryAccumulator()

        def _update_signature_cluster_average(unassigned_signature: str, cluster_id: int | str, dist: float) -> None:
            previous_average, previous_count = signature_to_cluster_to_average_dist[unassigned_signature][cluster_id]
            signature_to_cluster_to_average_dist[unassigned_signature][cluster_id] = (
                (previous_average * previous_count + float(dist)) / (previous_count + 1),
                previous_count + 1,
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
