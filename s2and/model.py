from __future__ import annotations

import copy
import inspect
import logging
import math
import time
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
from fastcluster import linkage
from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import EfficiencyWarning
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from s2and import memory_budget
from s2and.consts import DEFAULT_CHUNK_SIZE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.eval import b3_precision_recall_fscore
from s2and.feature_port import (
    _get_rust_featurizer,
    get_constraint_rust,
    get_constraints_matrix_indexed_rust,
    update_rust_cluster_seeds,
)
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.runtime import RuntimeContext, build_runtime_context, stage_uses_rust
from s2and.subblocking import make_subblocks
from s2and.text import same_prefix_tokens

logger = logging.getLogger("s2and")
IncrementalPhaseBMode = Literal["exact", "subblock_local"]


def _build_incremental_result(
    clusters: dict[str, list[str]],
    *,
    phase_b_mode: IncrementalPhaseBMode,
    phase_b_budget_bytes: int,
    phase_b_required_bytes: int,
    phase_a_accumulator_overflow_early_stop: bool = False,
) -> dict[str, Any]:
    return {
        "clusters": clusters,
        "phase_b_mode": phase_b_mode,
        "phase_b_budget_bytes": int(phase_b_budget_bytes),
        "phase_b_required_bytes": int(phase_b_required_bytes),
        "phase_a_accumulator_overflow_early_stop": bool(phase_a_accumulator_overflow_early_stop),
    }


def _validate_positive_total_ram_bytes(total_ram_bytes: int, *, source: str) -> int:
    return memory_budget.validate_positive_total_ram_bytes(total_ram_bytes, source=source)


def _detect_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    return memory_budget.detect_total_ram_bytes_best_effort()


def _detect_cgroup_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    return memory_budget.detect_cgroup_total_ram_bytes_best_effort()


def _current_rss_bytes_best_effort(total_ram_bytes: int) -> tuple[int, str]:
    return memory_budget.current_rss_bytes_best_effort(total_ram_bytes)


def _resolve_total_ram_bytes_for_incremental(total_ram_bytes: int | None = None) -> tuple[int, str]:
    return memory_budget.resolve_total_ram_bytes(
        total_ram_bytes,
        detect_cgroup_fn=_detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=_detect_total_ram_bytes_best_effort,
    )


def _count_selected_features(featurizer_info: FeaturizationInfo) -> int:
    """Count the number of feature indices selected by features_to_use."""
    indices: set[int] = set()
    for feature_name in featurizer_info.features_to_use:
        indices.update(featurizer_info.feature_group_to_index[feature_name])
    return len(indices)


def _compute_incremental_memory_limits(
    num_features: int,
    *,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_ram_bytes: int | None = None,
) -> dict[str, int | str | float]:
    return memory_budget.compute_incremental_phase_split_limits(
        num_features,
        selected_feature_count=selected_feature_count,
        nameless_feature_count=nameless_feature_count,
        total_ram_bytes=total_ram_bytes,
        detect_cgroup_fn=_detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=_detect_total_ram_bytes_best_effort,
        current_rss_fn=_current_rss_bytes_best_effort,
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
        return
    if not getattr(clf, "fitted_", False):
        clf.fitted_ = True
    if not hasattr(clf, "n_features_in_"):
        n_feat = getattr(clf, "_n_features", None)
        if n_feat is not None:
            clf.n_features_in_ = n_feat


def _predict_class0_with_runtime(
    classifier: Any,
    features: np.ndarray,
    model_role: str,
    rust_failure_counts: dict[str, int],
    runtime_context: RuntimeContext | None = None,
) -> tuple[np.ndarray, float, str]:
    features_2d = np.asarray(features, dtype=np.float64, order="C")
    if features_2d.size == 0:
        return np.asarray([], dtype=np.float64), 0.0, "none"

    python_start = time.perf_counter()
    predictions = classifier.predict_proba(features_2d)[:, 0]
    return predictions, time.perf_counter() - python_start, "python"


def _predict_and_combine(
    classifier: Any,
    nameless_classifier: Any | None,
    features: np.ndarray,
    labels: np.ndarray,
    nameless_features: np.ndarray | None,
    batch_label: int | str,
    rust_failure_counts: dict[str, int],
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
            main_pred, main_sec, main_be = _predict_class0_with_runtime(
                classifier,
                features,
                "main",
                rust_failure_counts,
                runtime_context=runtime_context,
            )
            if nameless_classifier is not None:
                nl_pred, nl_sec, nl_be = _predict_class0_with_runtime(
                    nameless_classifier,
                    nameless_features,  # type: ignore[arg-type]
                    "nameless",
                    rust_failure_counts,
                    runtime_context=runtime_context,
                )
                seconds += main_sec + nl_sec
                logger.info(
                    "Telemetry: model_predict batch=%s main=%s nameless=%s main_s=%.3f nl_s=%.3f rows=%d",
                    batch_label,
                    main_be,
                    nl_be,
                    main_sec,
                    nl_sec,
                    predicted_rows,
                )
                predictions[:] = (main_pred + nl_pred) / 2
            else:
                seconds += main_sec
                logger.info(
                    "Telemetry: model_predict batch=%s main=%s main_s=%.3f rows=%d",
                    batch_label,
                    main_be,
                    main_sec,
                    predicted_rows,
                )
                predictions[:] = main_pred
        else:
            predict_features = features[predict_flag, :]
            main_pred, main_sec, main_be = _predict_class0_with_runtime(
                classifier,
                predict_features,
                "main",
                rust_failure_counts,
                runtime_context=runtime_context,
            )
            if nameless_classifier is not None:
                nl_pred, nl_sec, nl_be = _predict_class0_with_runtime(
                    nameless_classifier,
                    nameless_features[predict_flag, :],  # type: ignore[index]
                    "nameless",
                    rust_failure_counts,
                    runtime_context=runtime_context,
                )
                seconds += main_sec + nl_sec
                logger.info(
                    "Telemetry: model_predict batch=%s main=%s nameless=%s main_s=%.3f nl_s=%.3f rows=%d",
                    batch_label,
                    main_be,
                    nl_be,
                    main_sec,
                    nl_sec,
                    predicted_rows,
                )
                predictions[predict_flag] = (main_pred + nl_pred) / 2
            else:
                seconds += main_sec
                logger.info(
                    "Telemetry: model_predict batch=%s main=%s main_s=%.3f rows=%d",
                    batch_label,
                    main_be,
                    main_sec,
                    predicted_rows,
                )
                predictions[predict_flag] = main_pred

    if np.any(not_predict_flag):
        predictions[not_predict_flag] = labels[not_predict_flag] + LARGE_INTEGER
    return predictions, seconds


def _use_rust_constraints(runtime_context: RuntimeContext | None = None) -> bool:
    if runtime_context is None:
        runtime_context = build_runtime_context("constraints")
    return stage_uses_rust(runtime_context, "constraints")


def _cluster_seeds_version(dataset: ANDData) -> int:
    return int(getattr(dataset, "_cluster_seeds_version", 0))


def _bump_cluster_seeds_version(dataset: ANDData) -> int:
    next_version = _cluster_seeds_version(dataset) + 1
    dataset._cluster_seeds_version = next_version
    return next_version


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
        try:
            return get_constraint_rust(
                dataset,
                sig_id_1,
                sig_id_2,
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                featurizer=rust_featurizer,
                runtime_context=runtime_context,
                use_cache=use_cache,
            )
        except Exception as exc:  # pragma: no cover - native extension optional
            if stage_uses_rust(runtime_context, "constraints"):
                raise RuntimeError(
                    "Rust constraint evaluation failed in strict rust backend "
                    f"(pair=({sig_id_1}, {sig_id_2}) run_id={runtime_context.run_id} error={exc})"
                ) from exc
            logger.warning(f"Rust get_constraint failed, falling back to Python: {exc}")
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
            return
        try:
            update_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=use_cache)
            dataset._rust_cluster_seeds_synced_version = seed_version
            dataset._rust_cluster_seeds_require_id = require_id
            dataset._rust_cluster_seeds_require_len = require_len
            dataset._rust_cluster_seeds_disallow_id = disallow_id
            dataset._rust_cluster_seeds_disallow_len = disallow_len
        except Exception as exc:  # pragma: no cover - native extension optional
            if stage_uses_rust(runtime_context, "constraints"):
                raise RuntimeError(
                    "Rust cluster seed sync failed in strict rust backend "
                    f"(run_id={runtime_context.run_id} error={exc})"
                ) from exc
            logger.warning(f"Rust cluster seed sync failed, falling back to Python: {exc}")


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

    try:
        rust_featurizer = _get_rust_featurizer(dataset, runtime_context=runtime_context, use_cache=use_cache)
    except Exception as exc:  # pragma: no cover - native extension optional
        if stage_uses_rust(runtime_context, "constraints"):
            raise RuntimeError(
                "Rust constraint stage requested but Rust featurizer init failed "
                f"(run_id={runtime_context.run_id} error={exc})"
            ) from exc
        logger.warning(f"Rust featurizer init failed, falling back to Python constraints: {exc}")
        return None, False

    return rust_featurizer, True


def _resolve_incremental_pair_label(
    dataset: ANDData,
    unassigned_signature: str,
    assigned_signature: str,
    *,
    partial_supervision: dict[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    dont_merge_cluster_seeds: bool,
    incremental_dont_use_cluster_seeds: bool,
    rust_featurizer: object | None,
    use_rust_constraints: bool | None,
    runtime_context: RuntimeContext,
    use_cache: bool = False,
) -> float:
    if (unassigned_signature, assigned_signature) in partial_supervision:
        return partial_supervision[(unassigned_signature, assigned_signature)] - LARGE_INTEGER
    if (assigned_signature, unassigned_signature) in partial_supervision:
        return partial_supervision[(assigned_signature, unassigned_signature)] - LARGE_INTEGER
    if not use_default_constraints_as_supervision:
        return np.nan

    value = _get_constraint_value(
        dataset,
        unassigned_signature,
        assigned_signature,
        dont_merge_cluster_seeds=dont_merge_cluster_seeds,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        rust_featurizer=rust_featurizer,
        use_rust_constraints=use_rust_constraints,
        runtime_context=runtime_context,
        use_cache=use_cache,
    )
    if value is not None:
        return value - LARGE_INTEGER
    return np.nan


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


def _resolve_constraint_labels_batch(
    dataset: ANDData,
    pair_ids: list[tuple[str, str]],
    *,
    partial_supervision: dict[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    dont_merge_cluster_seeds: bool,
    incremental_dont_use_cluster_seeds: bool,
    rust_featurizer: object | None,
    use_rust_constraints: bool | None,
    runtime_context: RuntimeContext,
    use_cache: bool = False,
    num_threads: int | None = None,
    constraint_api_mode: str | None = None,
    signature_index_by_id: dict[str, int] | None = None,
) -> tuple[list[float], dict[str, int | float | str]]:
    labels: list[float] = [float(np.nan)] * len(pair_ids)
    unresolved_pairs: list[tuple[str, str]] = []
    unresolved_indices: list[int] = []
    partial_hits = 0
    for idx, (sig_id_1, sig_id_2) in enumerate(pair_ids):
        if (sig_id_1, sig_id_2) in partial_supervision:
            labels[idx] = float(partial_supervision[(sig_id_1, sig_id_2)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        if (sig_id_2, sig_id_1) in partial_supervision:
            labels[idx] = float(partial_supervision[(sig_id_2, sig_id_1)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        unresolved_pairs.append((sig_id_1, sig_id_2))
        unresolved_indices.append(idx)

    mode = constraint_api_mode or _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    telemetry: dict[str, int | float | str] = {
        "total_pairs": int(len(pair_ids)),
        "partial_supervision_hits": int(partial_hits),
        "unresolved_pairs": int(len(unresolved_pairs)),
        "rust_batch_call_count": 0,
        "api_mode": mode,
        "elapsed_seconds": 0.0,
    }
    if not unresolved_pairs or not use_default_constraints_as_supervision:
        if not use_default_constraints_as_supervision:
            telemetry["api_mode"] = "partial_only"
        return labels, telemetry

    start = time.perf_counter()
    values: list[float | None]
    if use_rust_constraints and rust_featurizer is not None:
        try:
            if mode == "indexed":
                if signature_index_by_id is None:
                    raise RuntimeError("Indexed constraint API requested without signature index lookup")
                indexed_pairs = [(signature_index_by_id[s1], signature_index_by_id[s2]) for s1, s2 in unresolved_pairs]
                values = get_constraints_matrix_indexed_rust(
                    dataset,
                    indexed_pairs,
                    dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    num_threads=num_threads,
                    featurizer=rust_featurizer,
                    runtime_context=runtime_context,
                    use_cache=use_cache,
                )
                telemetry["rust_batch_call_count"] = 1
            else:
                values = [
                    get_constraint_rust(
                        dataset,
                        s1,
                        s2,
                        dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        featurizer=rust_featurizer,
                        runtime_context=runtime_context,
                        use_cache=use_cache,
                    )
                    for s1, s2 in unresolved_pairs
                ]
                telemetry["rust_batch_call_count"] = int(len(unresolved_pairs))
        except Exception as exc:  # pragma: no cover - native extension optional
            if stage_uses_rust(runtime_context, "constraints"):
                raise RuntimeError(
                    "Rust batch constraint evaluation failed in strict rust backend "
                    f"(pairs={len(unresolved_pairs)} run_id={runtime_context.run_id} error={exc})"
                ) from exc
            logger.warning("Rust batch constraint evaluation failed, falling back to Python constraints: %s", exc)
            values = [
                dataset.get_constraint(
                    s1,
                    s2,
                    dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                )
                for s1, s2 in unresolved_pairs
            ]
            telemetry["api_mode"] = "python_fallback"
            telemetry["rust_batch_call_count"] = 0
    else:
        values = [
            dataset.get_constraint(
                s1,
                s2,
                dont_merge_cluster_seeds=dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            )
            for s1, s2 in unresolved_pairs
        ]
        telemetry["api_mode"] = "python"

    telemetry["elapsed_seconds"] = float(time.perf_counter() - start)
    for idx, value in zip(unresolved_indices, values, strict=False):
        if value is None:
            labels[idx] = float(np.nan)
        else:
            labels[idx] = float(value - LARGE_INTEGER)
    return labels, telemetry


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
            this flag controls whether to use cluster seeds to enforce "dont merge"
            as well as "must merge" constraints
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

        self.hyperopt_trials_store: Trials | list[Trials] | None = None
        self.best_params: dict[Any, Any] | None = None
        self.batch_size = batch_size

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
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        yields pairs of ((sig id 1, sig id 2, label), index pair into the distance matrix, block key)
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("constraints")
        rust_featurizer: object | None = None
        use_rust_constraints: bool | None = None
        if self.use_default_constraints_as_supervision:
            use_rust_constraints = _use_rust_constraints(runtime_context)
            if use_rust_constraints:
                try:
                    rust_featurizer = _get_rust_featurizer(
                        dataset,
                        runtime_context=runtime_context,
                        use_cache=self.use_cache,
                    )
                except Exception as exc:  # pragma: no cover - native extension optional
                    if stage_uses_rust(runtime_context, "constraints"):
                        raise RuntimeError(
                            "Rust constraint stage requested but Rust featurizer init failed "
                            f"(run_id={runtime_context.run_id} error={exc})"
                        ) from exc
                    use_rust_constraints = False
                    logger.warning(f"Rust featurizer init failed, falling back to Python constraints: {exc}")
        constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
        signature_index_by_id: dict[str, int] | None = None
        if constraint_api_mode == "indexed" and rust_featurizer is not None:
            try:
                signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
            except Exception as exc:  # pragma: no cover - native extension optional
                if stage_uses_rust(runtime_context, "constraints"):
                    raise RuntimeError(
                        "Rust indexed constraint setup failed in strict rust backend "
                        f"(run_id={runtime_context.run_id} error={exc})"
                    ) from exc
                logger.warning(
                    "Rust indexed constraint setup failed; disabling Rust constraints and falling back to Python: %s",
                    exc,
                )
                use_rust_constraints = False
                rust_featurizer = None
                constraint_api_mode = "python"

        telemetry_total_pairs = 0
        telemetry_partial_hits = 0
        telemetry_unresolved_pairs = 0
        telemetry_rust_batch_calls = 0
        telemetry_elapsed_seconds = 0.0
        telemetry_api_modes: set[str] = set()
        pair_chunk_size = max(1, int(self.batch_size))

        for block_key, signatures in block_dict.items():
            pair_batch_ids: list[tuple[str, str]] = []
            index_batch: list[tuple[int, int]] = []
            for i, j in zip(*np.triu_indices(len(signatures), k=1), strict=False):
                pair_batch_ids.append((signatures[i], signatures[j]))
                index_batch.append((i, j))
                if len(pair_batch_ids) >= pair_chunk_size:
                    labels, batch_telemetry = _resolve_constraint_labels_batch(
                        dataset,
                        pair_batch_ids,
                        partial_supervision=partial_supervision,
                        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                        rust_featurizer=rust_featurizer,
                        use_rust_constraints=use_rust_constraints,
                        runtime_context=runtime_context,
                        use_cache=self.use_cache,
                        num_threads=self.n_jobs,
                        constraint_api_mode=constraint_api_mode,
                        signature_index_by_id=signature_index_by_id,
                    )
                    telemetry_total_pairs += int(batch_telemetry["total_pairs"])
                    telemetry_partial_hits += int(batch_telemetry["partial_supervision_hits"])
                    telemetry_unresolved_pairs += int(batch_telemetry["unresolved_pairs"])
                    telemetry_rust_batch_calls += int(batch_telemetry["rust_batch_call_count"])
                    telemetry_elapsed_seconds += float(batch_telemetry["elapsed_seconds"])
                    telemetry_api_modes.add(str(batch_telemetry["api_mode"]))
                    for (sig_id_1, sig_id_2), label, (left, right) in zip(
                        pair_batch_ids,
                        labels,
                        index_batch,
                        strict=False,
                    ):
                        yield ((sig_id_1, sig_id_2, label), (left, right), block_key)
                    pair_batch_ids = []
                    index_batch = []

            if pair_batch_ids:
                labels, batch_telemetry = _resolve_constraint_labels_batch(
                    dataset,
                    pair_batch_ids,
                    partial_supervision=partial_supervision,
                    use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                    dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    rust_featurizer=rust_featurizer,
                    use_rust_constraints=use_rust_constraints,
                    runtime_context=runtime_context,
                    use_cache=self.use_cache,
                    num_threads=self.n_jobs,
                    constraint_api_mode=constraint_api_mode,
                    signature_index_by_id=signature_index_by_id,
                )
                telemetry_total_pairs += int(batch_telemetry["total_pairs"])
                telemetry_partial_hits += int(batch_telemetry["partial_supervision_hits"])
                telemetry_unresolved_pairs += int(batch_telemetry["unresolved_pairs"])
                telemetry_rust_batch_calls += int(batch_telemetry["rust_batch_call_count"])
                telemetry_elapsed_seconds += float(batch_telemetry["elapsed_seconds"])
                telemetry_api_modes.add(str(batch_telemetry["api_mode"]))
                for (sig_id_1, sig_id_2), label, (left, right) in zip(
                    pair_batch_ids,
                    labels,
                    index_batch,
                    strict=False,
                ):
                    yield ((sig_id_1, sig_id_2, label), (left, right), block_key)

        logger.info(
            "Telemetry: constraint_batch stage=distance_matrix total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            telemetry_total_pairs,
            telemetry_partial_hits,
            telemetry_unresolved_pairs,
            telemetry_rust_batch_calls,
            ",".join(sorted(telemetry_api_modes)) if telemetry_api_modes else "none",
            telemetry_elapsed_seconds,
            runtime_context.run_id,
        )

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
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

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
        for block_key, signatures in block_dict.items():
            block_size = len(signatures)
            num_pairs += int(block_size * (block_size - 1) / 2)
            if isinstance(self.cluster_model, FastCluster):
                # flattened pdist style
                pairwise_proba = np.zeros(int(block_size * (block_size - 1) / 2), dtype=np.float16)
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            pairwise_probas[block_key] = pairwise_proba

        logger.info(f"Pairwise probas initialized with {num_pairs} elements, starting making all pairs")

        # featurize and predict in batches
        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
        )

        prev_block_key = ""
        batch_num = 0
        num_batches = math.ceil(num_pairs / self.batch_size)
        model_predict_seconds = 0.0
        rust_failure_counts: dict[str, int] = {"main": 0, "nameless": 0}
        while True:
            logger.info(f"Featurizing batch {batch_num}/{num_batches}")
            count = 0
            pairs = []
            indices = []
            blocks = []
            # iterate over a batch_size number of pairs
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
                rust_failure_counts,
                runtime_context=runtime_context,
            )
            model_predict_seconds += batch_seconds

            for within_batch_index, prediction in tqdm(
                enumerate(batch_predictions),
                total=len(batch_predictions),
                desc="Writing matrices",
                disable=disable_tqdm,
            ):
                block_key = blocks[within_batch_index]
                if block_key != prev_block_key:
                    block_key_start_index = blocks.index(block_key) + (batch_num * self.batch_size)
                    pairwise_proba = pairwise_probas[block_key]

                if isinstance(self.cluster_model, FastCluster):
                    index = (batch_num * self.batch_size + within_batch_index) - block_key_start_index

                    pairwise_proba[index] = prediction
                else:
                    i, j = indices[within_batch_index]
                    pairwise_proba[i, j] = prediction

                prev_block_key = block_key

            if count < self.batch_size:
                break

            batch_num += 1

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
                val_datasets_list, val_block_dict_list, val_cluster_to_signatures_list, val_dists_list, strict=False
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
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset
            Don't use if you don't know what this is
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

        # The approach will be to (1) take every block, apply subblocking function to it.
        # (2) Then run the clusterer on the subblocked blocks, taking care to remove subblocks that are
        # single-letter first names.
        # (3) Then run predict incremental on the single-letter first names.
        if batching_threshold is not None:
            assert batching_threshold > 0, "Batching threshold must be positive"
            assert dists is None, "If batching_threshold is not None, then can't use precomputed dists"

            if desired_memory_use is None:
                desired_memory_use = batching_threshold * batching_threshold

            # run subblocking on each block in the block_dict
            block_dict_subblocked: dict[str, list[str]] = {}
            for block_key in sorted(block_dict):
                block_signatures = block_dict[block_key]
                if len(block_signatures) > batching_threshold:
                    # run subblocking on this block
                    subblocks = make_subblocks(block_signatures, dataset, maximum_size=batching_threshold)
                    # add these subblocks to the block_dict
                    for subblock_key in sorted(subblocks):
                        subblock_signatures = subblocks[subblock_key]
                        block_dict_subblocked[f"{block_key}|subblock={subblock_key}"] = subblock_signatures
                        assert len(subblock_signatures) <= batching_threshold, "Subblock is too big for some reason!"
                else:
                    # add this block to the block_dict_subblocked
                    block_dict_subblocked[block_key] = block_signatures

            # now run predict_helper on the blocks in block_dict_subblocked
            # pull out all of the ones that are single-letter first names
            block_dict_subblocked_single_letter_first_names = {
                block_key: block_signatures
                for block_key, block_signatures in block_dict_subblocked.items()
                if len(_signature_first_for_rules(dataset.signatures[block_signatures[0]])) <= 1
            }
            block_dict_subblocked_multiple_letter_first_names = {
                block_key: block_signatures
                for block_key, block_signatures in block_dict_subblocked.items()
                if block_key not in block_dict_subblocked_single_letter_first_names
            }

            # edge case: where there are no block_dict_subblocked_multiple_letter_first_names
            # so then it makes no sense to (1) run predict on multiple letters and (2) incremental on single.
            # the only thing we can do is run predict on the multi.
            if len(block_dict_subblocked_multiple_letter_first_names) == 0:
                # not really true, but it makes the code much easier below
                alert_flag = True
                block_dict_subblocked_multiple_letter_first_names = block_dict_subblocked_single_letter_first_names
                block_dict_subblocked_single_letter_first_names = {}
            else:
                alert_flag = False

            pred_clusters = {}
            # ideally we would batch the subblocks for predictions
            # but it's hard to know how to batch since this can be called
            # from inside of predict_incremental, which has different OOM behavior.
            # so just doing it one at a time here
            if len(block_dict_subblocked_multiple_letter_first_names) > 0:
                if alert_flag:
                    logger.info("Note! There are no subblocks with multiple letter first names")
                    logger.info("Running predict on subblocks with single letter first names")
                else:
                    logger.info("Running predict on subblocks with multiple letter first names")
                predict_times = {}
                for block_key in sorted(block_dict_subblocked_multiple_letter_first_names):
                    block_signatures = block_dict_subblocked_multiple_letter_first_names[block_key]
                    logger.info(f"Working on subblock {block_key}")
                    start = time.time()
                    pred_clusters_intermediate, _ = self.predict_helper(
                        {block_key: block_signatures},
                        dataset,
                        None,  # precomputed dists is too hard to do here
                        cluster_model_params,
                        partial_supervision,
                        use_s2_clusters,
                        incremental_dont_use_cluster_seeds,
                        runtime_context=runtime_context,
                    )
                    end = time.time()
                    total_predict_time = end - start
                    predict_times[block_key] = total_predict_time
                    pred_clusters.update(pred_clusters_intermediate)
                logger.info(f"Finished, here's how long each took: {predict_times}")
            # now we run predict_incremental on the single-letter first name blocks, one block at a time
            # and we will be using the pred_clusters as cluster_seeds_require because
            # that's how predict_incremental works: cluster_seeds_require is what is already clustered
            # and the input to predict_incremental will be assigned into those seed clusters
            # note: storing the original cluster_seeds_require so we can restore it later
            if len(block_dict_subblocked_single_letter_first_names) > 0:
                logger.info("Running predict incremental on subblocks with single letter first names")
                cluster_seeds_require_original = copy.deepcopy(dataset.cluster_seeds_require)
                dataset.cluster_seeds_require = {}
                for cluster_id, signatures in pred_clusters.items():
                    for signature in signatures:
                        dataset.cluster_seeds_require[signature] = cluster_id  # type: ignore
                _bump_cluster_seeds_version(dataset)
                _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)

                predict_times = {}
                for block_key in sorted(block_dict_subblocked_single_letter_first_names, key=str):
                    block_signatures = block_dict_subblocked_single_letter_first_names[block_key]
                    # we have to be super careful here and adjust the batching threshold take into account
                    # the implied requirement of passing batching_threshold into batch predict:
                    # it essentially assumes that max memory is batching_threshold ** 2,
                    # but it could be MUCH bigger here since predict incremental memory use is up to
                    # (batching_threshold * (total_block_size - batching_threshold))
                    # so we need a special batching_threshold just for this operation

                    # this is the number of signatures already assigned
                    N = len(dataset.cluster_seeds_require)
                    actual_memory_usage = len(block_signatures) * N
                    print(
                        f"N_seeds = {N}",
                        f"N_signatures = {len(block_signatures)}",
                        f"desired_memory_use: {desired_memory_use}",
                        f"actual_memory_usage: {actual_memory_usage}",
                    )
                    if N <= 0:
                        loop_batching_threshold = None  # type: ignore
                    elif actual_memory_usage > desired_memory_use:
                        # we need to have a loop_batching_threshold such that
                        # loop_batching_threshold * N = desired_memory_use
                        loop_batching_threshold = max(1, int(desired_memory_use / N))
                    else:
                        # already within memory limits using no batching
                        loop_batching_threshold = None  # type: ignore
                    logger.info(
                        f"Working on subblock {block_key} with computed batching threshold {loop_batching_threshold}"
                    )
                    start_predict_time = time.time()
                    incremental_result = self.predict_incremental(
                        block_signatures,
                        dataset,
                        prevent_new_incompatibilities=True,
                        batching_threshold=loop_batching_threshold,
                        partial_supervision=partial_supervision,
                        runtime_context=runtime_context,
                    )
                    pred_clusters_intermediate = incremental_result["clusters"]
                    end_predict_time = time.time()
                    total_predict_time = end_predict_time - start_predict_time
                    predict_times[block_key] = total_predict_time
                    # again, make cluster seeds require
                    dataset.cluster_seeds_require = {}
                    for cluster_id, signatures in pred_clusters_intermediate.items():
                        for signature in signatures:
                            dataset.cluster_seeds_require[signature] = cluster_id  # type: ignore
                    _bump_cluster_seeds_version(dataset)
                    _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)

                # undoing the damage
                logger.info(
                    f"Finished subblocked predict incremental. Here's how long each subblock took: {predict_times}"
                )
                dataset.cluster_seeds_require = cluster_seeds_require_original
                _bump_cluster_seeds_version(dataset)
                _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
                # the output of predict_incremental_helper has the ENTIRE clustering, not just the new stuff
                pred_clusters = pred_clusters_intermediate
            dists = None

        else:
            # normal mode - everything goes through full block clustering
            logger.info("Running predict on full blocks - no subblocking")
            start = time.time()
            pred_clusters, dists = self.predict_helper(
                block_dict,
                dataset,
                dists,
                cluster_model_params,
                partial_supervision,
                use_s2_clusters,
                incremental_dont_use_cluster_seeds,
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
            warnings.simplefilter("ignore", category=EfficiencyWarning)
            cluster_model.fit(dist_matrix)
        labels = cluster_model.labels_
        max_label = labels.max()
        negative_one_label_locations = np.where(labels == -1)[0]
        for i, loc in enumerate(negative_one_label_locations):
            labels[loc] = max_label + 1 + i
        if self.use_default_constraints_as_supervision:
            disallow_signature_ids = all_disallow_signature_ids
            inverse_id_map = defaultdict(set)
            for signature_id, label in zip(block_signatures, labels, strict=False):
                if signature_id in dataset.cluster_seeds_require and signature_id not in disallow_signature_ids:
                    inverse_id_map[dataset.cluster_seeds_require[signature_id]].add(label)
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
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

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

        pred_clusters = defaultdict(list)

        if use_s2_clusters:
            for _, signature_list in block_dict.items():
                for _signature in signature_list:
                    s2_cluster_key = dataset.signatures[_signature].author_id
                    pred_clusters[s2_cluster_key].append(_signature)

            return dict(pred_clusters), dists

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
                for signature, label in zip(block_dict[block_key], labels, strict=False):
                    pred_clusters[block_key + "_" + str(label)].append(signature)
            return dict(pred_clusters), dists

        # fused path: build one block's matrix, cluster it, free it, repeat
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
        _ensure_lightgbm_fitted(self.classifier)
        _ensure_lightgbm_fitted(self.nameless_classifier)

        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
        )

        prev_block_key = ""
        pairwise_proba: np.ndarray | None = None
        block_pair_index = 0
        seen_block_keys: set = set()
        batch_num = 0
        num_pairs = sum(len(sigs) * (len(sigs) - 1) // 2 for sigs in block_dict.values())
        num_batches = math.ceil(num_pairs / self.batch_size) if num_pairs > 0 else 0
        model_predict_seconds = 0.0
        rust_failure_counts: dict[str, int] = {"main": 0, "nameless": 0}

        while True:
            logger.info(f"Featurizing batch {batch_num}/{num_batches}")
            count = 0
            pairs: list = []
            indices: list = []
            blocks: list = []
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
                rust_failure_counts,
                runtime_context=runtime_context,
            )
            model_predict_seconds += batch_seconds

            for within_batch_index, prediction in enumerate(batch_predictions):
                block_key = blocks[within_batch_index]
                if block_key != prev_block_key:
                    # cluster the completed block
                    if prev_block_key != "" and pairwise_proba is not None:
                        if not isinstance(self.cluster_model, FastCluster):
                            pairwise_proba += pairwise_proba.T
                            np.fill_diagonal(pairwise_proba, 0)
                        labels = self._cluster_one_block(
                            block_dict[prev_block_key],
                            pairwise_proba,
                            effective_cluster_model_params,
                            dataset,
                            all_disallow_signature_ids,
                        )
                        for signature, label in zip(block_dict[prev_block_key], labels, strict=False):
                            pred_clusters[prev_block_key + "_" + str(label)].append(signature)
                        del pairwise_proba

                    # allocate new block's matrix
                    seen_block_keys.add(block_key)
                    block_size = len(block_dict[block_key])
                    if isinstance(self.cluster_model, FastCluster):
                        pairwise_proba = np.zeros(block_size * (block_size - 1) // 2, dtype=fastcluster_fused_dtype)
                    else:
                        pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
                    block_pair_index = 0

                if isinstance(self.cluster_model, FastCluster):
                    assert pairwise_proba is not None
                    pairwise_proba[block_pair_index] = prediction
                else:
                    assert pairwise_proba is not None
                    i, j = indices[within_batch_index]
                    pairwise_proba[i, j] = prediction
                block_pair_index += 1
                prev_block_key = block_key

            if count < self.batch_size:
                break
            batch_num += 1

        # cluster the final block
        if prev_block_key != "" and pairwise_proba is not None:
            if not isinstance(self.cluster_model, FastCluster):
                pairwise_proba += pairwise_proba.T
                np.fill_diagonal(pairwise_proba, 0)
            labels = self._cluster_one_block(
                block_dict[prev_block_key],
                pairwise_proba,
                effective_cluster_model_params,
                dataset,
                all_disallow_signature_ids,
            )
            for signature, label in zip(block_dict[prev_block_key], labels, strict=False):
                pred_clusters[prev_block_key + "_" + str(label)].append(signature)
            del pairwise_proba

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
        if dataset.altered_cluster_signatures is not None and len(dataset.altered_cluster_signatures) > 0:
            logger.info("Dealing with altered cluster signatures")
            altered_cluster_nums = set(
                dataset.cluster_seeds_require[altered_signature_id]
                for altered_signature_id in dataset.altered_cluster_signatures
                if altered_signature_id in dataset.cluster_seeds_require
            )
            for altered_cluster_num in altered_cluster_nums:
                signature_ids_for_cluster_num = cluster_seeds_require_inverse.get(altered_cluster_num, [])
                if len(signature_ids_for_cluster_num) == 0:
                    continue

                # During this pre-split, do not apply incoming cluster seeds as constraints.
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
                        cluster_seeds_require[reclustered_signature_id] = new_cluster_num  # type: ignore

        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse

    def _phase_a_seed_distances(
        self,
        unassigned_signature_ids: list[str],
        cluster_seeds_require: dict[str, int | str],
        dataset: ANDData,
        partial_supervision: dict[tuple[str, str], int | float],
        signature_to_cluster_sum_count: dict[str, dict[int | str, list[float | int]]],
        chunk_pairs: int,
        chunk_limits: dict[str, int | str | float] | None,
        runtime_context: RuntimeContext,
        total_ram_bytes: int | None = None,
    ) -> dict[str, Any]:
        if chunk_pairs <= 0:
            raise ValueError("chunk_pairs must be positive")

        pair_buffer: list[tuple[str, str, float]] = []
        pair_id_buffer: list[tuple[str, str]] = []
        rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
        )
        constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
        signature_index_by_id: dict[str, int] | None = None
        if constraint_api_mode == "indexed" and rust_featurizer is not None:
            try:
                signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
            except Exception as exc:  # pragma: no cover - native extension optional
                if stage_uses_rust(runtime_context, "constraints"):
                    raise RuntimeError(
                        "Rust indexed constraint setup failed in strict rust backend "
                        f"(run_id={runtime_context.run_id} error={exc})"
                    ) from exc
                logger.warning(
                    "Rust indexed constraint setup failed in phase A; falling back to per-pair constraints: %s",
                    exc,
                )
                constraint_api_mode = "fallback"
        constraint_pairs_total = 0
        constraint_chunks_total = 0
        constraint_partial_hits_total = 0
        constraint_unresolved_total = 0
        constraint_rust_batch_calls_total = 0
        constraint_batch_seconds_total = 0.0
        constraint_api_modes: set[str] = set()
        model_predict_total_seconds = 0.0
        accumulator_warned = False
        accumulator_overflow_early_stop = False
        accumulator_entries_peak = 0
        chunk_features_peak_bytes = 0
        chunk_pairs_peak = 0
        adaptive_halvings = 0
        total_ram_for_phase = total_ram_bytes
        rss_source = "unavailable"
        rss_before_bytes = 0
        rss_peak_bytes = 0

        if total_ram_for_phase is None and chunk_limits is not None:
            total_ram_for_phase = int(chunk_limits["total_ram_bytes"])

        if total_ram_for_phase is not None:
            rss_before_bytes, rss_source = _current_rss_bytes_best_effort(total_ram_for_phase)
            rss_peak_bytes = rss_before_bytes

        def _accumulator_entry_count() -> int:
            return int(sum(len(cluster_map) for cluster_map in signature_to_cluster_sum_count.values()))

        def _process_incremental_chunk(chunk_pairs_buffer: list[tuple[str, str, float]]) -> None:
            nonlocal accumulator_entries_peak
            nonlocal accumulator_warned
            nonlocal accumulator_overflow_early_stop
            nonlocal constraint_pairs_total
            nonlocal constraint_chunks_total
            nonlocal model_predict_total_seconds
            nonlocal rss_peak_bytes
            nonlocal chunk_features_peak_bytes
            nonlocal chunk_pairs_peak
            nonlocal chunk_pairs
            nonlocal adaptive_halvings

            if len(chunk_pairs_buffer) == 0:
                return

            chunk_pairs_peak = max(chunk_pairs_peak, len(chunk_pairs_buffer))
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
            chunk_features_peak_bytes = max(chunk_features_peak_bytes, chunk_feature_bytes)
            rust_failure_counts: dict[str, int] = {"main": 0, "nameless": 0}
            chunk_predictions, chunk_model_seconds = _predict_and_combine(
                self.classifier,
                self.nameless_classifier,
                chunk_features,
                chunk_labels,
                chunk_nameless_features,
                f"phase_a_chunk_{constraint_chunks_total + 1}",
                rust_failure_counts,
                runtime_context=runtime_context,
            )
            model_predict_total_seconds += chunk_model_seconds

            for signature_pair, dist in zip(chunk_pairs_buffer, chunk_predictions, strict=False):
                unassigned_signature, assigned_signature, _ = signature_pair
                if assigned_signature not in cluster_seeds_require:
                    continue
                cluster_id = cluster_seeds_require[assigned_signature]
                cluster_sum_count = signature_to_cluster_sum_count.setdefault(unassigned_signature, {})
                total_count = cluster_sum_count.setdefault(cluster_id, [0.0, 0])
                total_count[0] = float(total_count[0]) + float(dist)
                total_count[1] = int(total_count[1]) + 1

            constraint_pairs_total += len(chunk_pairs_buffer)
            constraint_chunks_total += 1
            accumulator_entries = _accumulator_entry_count()
            accumulator_entries_peak = max(accumulator_entries_peak, accumulator_entries)

            if (
                not accumulator_warned
                and chunk_limits is not None
                and accumulator_entries >= int(chunk_limits["accumulator_warn"])
            ):
                accumulator_warned = True
                logger.warning(
                    "Phase A accumulator approaching limit: entries=%d warn=%d max=%d run_id=%s",
                    accumulator_entries,
                    int(chunk_limits["accumulator_warn"]),
                    int(chunk_limits["accumulator_max"]),
                    runtime_context.run_id,
                )

            if chunk_limits is not None and accumulator_entries > int(chunk_limits["accumulator_max"]):
                clusters_touched_avg = float(accumulator_entries) / float(max(1, len(signature_to_cluster_sum_count)))
                logger.warning(
                    "Phase A accumulator exceeded limit: "
                    "entries=%d max=%d warn=%d "
                    "chunk_pairs=%d available_bytes=%d current_rss_bytes=%d "
                    "unassigned_signatures=%d clusters_touched_avg=%.3f run_id=%s. "
                    "Stopping Phase A early; remaining unassigned signatures will proceed "
                    "with partial seed distances.",
                    accumulator_entries,
                    int(chunk_limits["accumulator_max"]),
                    int(chunk_limits["accumulator_warn"]),
                    int(chunk_limits["chunk_pairs"]),
                    int(chunk_limits["available_bytes"]),
                    int(chunk_limits["current_rss_bytes"]),
                    len(signature_to_cluster_sum_count),
                    clusters_touched_avg,
                    runtime_context.run_id,
                )
                accumulator_overflow_early_stop = True
                return

            # Fallback accumulator bound when chunk_limits is None.
            if chunk_limits is None and accumulator_entries > memory_budget.FALLBACK_ACCUMULATOR_MAX_ENTRIES:
                logger.warning(
                    "Phase A accumulator exceeded fallback limit: "
                    "entries=%d max=%d run_id=%s. "
                    "Stopping Phase A early; remaining unassigned signatures will proceed "
                    "with partial seed distances. "
                    "Pass total_ram_bytes to enable proper memory budgeting.",
                    accumulator_entries,
                    memory_budget.FALLBACK_ACCUMULATOR_MAX_ENTRIES,
                    runtime_context.run_id,
                )
                accumulator_overflow_early_stop = True
                return

            if total_ram_for_phase is not None:
                rss_now, _ = _current_rss_bytes_best_effort(total_ram_for_phase)
                rss_peak_bytes = max(rss_peak_bytes, rss_now)

            # Fix #3: adaptive chunking — halve chunk_pairs if observed RSS delta exceeds prediction.
            if total_ram_for_phase is not None and adaptive_halvings < 3:
                acc_entry_bytes = int(memory_budget.INCREMENTAL_ACCUMULATOR_ENTRY_BYTES)
                if chunk_limits is not None and "accumulator_entry_bytes" in chunk_limits:
                    acc_entry_bytes = int(chunk_limits["accumulator_entry_bytes"])
                current_predicted_delta = (
                    chunk_features_peak_bytes
                    + accumulator_entries_peak * acc_entry_bytes
                    + chunk_pairs_peak * int(memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES)
                    + int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES)
                )
                current_observed_delta = max(0, rss_peak_bytes - rss_before_bytes)
                if current_predicted_delta > 0 and current_observed_delta > current_predicted_delta * 1.2:
                    chunk_pairs = max(1, chunk_pairs // 2)
                    adaptive_halvings += 1
                    logger.warning(
                        "Phase A adaptive chunking: observed_delta=%d > predicted_delta=%d * 1.2; "
                        "halving chunk_pairs to %d (halving %d/3) run_id=%s",
                        current_observed_delta,
                        current_predicted_delta,
                        chunk_pairs,
                        adaptive_halvings,
                        runtime_context.run_id,
                    )

            del chunk_features
            del chunk_nameless_features
            del chunk_predictions
            del chunk_labels

        for unassigned_signature in unassigned_signature_ids:
            if accumulator_overflow_early_stop:
                break
            for signature in cluster_seeds_require.keys():
                pair_id_buffer.append((unassigned_signature, signature))
                if len(pair_id_buffer) >= chunk_pairs:
                    labels, batch_telemetry = _resolve_constraint_labels_batch(
                        dataset,
                        pair_id_buffer,
                        partial_supervision=partial_supervision,
                        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=False,
                        rust_featurizer=rust_featurizer,
                        use_rust_constraints=use_rust_constraints,
                        runtime_context=runtime_context,
                        use_cache=self.use_cache,
                        num_threads=self.n_jobs,
                        constraint_api_mode=constraint_api_mode,
                        signature_index_by_id=signature_index_by_id,
                    )
                    constraint_partial_hits_total += int(batch_telemetry["partial_supervision_hits"])
                    constraint_unresolved_total += int(batch_telemetry["unresolved_pairs"])
                    constraint_rust_batch_calls_total += int(batch_telemetry["rust_batch_call_count"])
                    constraint_batch_seconds_total += float(batch_telemetry["elapsed_seconds"])
                    constraint_api_modes.add(str(batch_telemetry["api_mode"]))
                    pair_buffer = [
                        (sig_id_1, sig_id_2, label)
                        for (sig_id_1, sig_id_2), label in zip(pair_id_buffer, labels, strict=False)
                    ]
                    _process_incremental_chunk(pair_buffer)
                    pair_buffer = []
                    pair_id_buffer = []
                    if accumulator_overflow_early_stop:
                        break

        if len(pair_id_buffer) > 0 and not accumulator_overflow_early_stop:
            labels, batch_telemetry = _resolve_constraint_labels_batch(
                dataset,
                pair_id_buffer,
                partial_supervision=partial_supervision,
                use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=False,
                rust_featurizer=rust_featurizer,
                use_rust_constraints=use_rust_constraints,
                runtime_context=runtime_context,
                use_cache=self.use_cache,
                num_threads=self.n_jobs,
                constraint_api_mode=constraint_api_mode,
                signature_index_by_id=signature_index_by_id,
            )
            constraint_partial_hits_total += int(batch_telemetry["partial_supervision_hits"])
            constraint_unresolved_total += int(batch_telemetry["unresolved_pairs"])
            constraint_rust_batch_calls_total += int(batch_telemetry["rust_batch_call_count"])
            constraint_batch_seconds_total += float(batch_telemetry["elapsed_seconds"])
            constraint_api_modes.add(str(batch_telemetry["api_mode"]))
            pair_buffer = [
                (sig_id_1, sig_id_2, label) for (sig_id_1, sig_id_2), label in zip(pair_id_buffer, labels, strict=False)
            ]
            _process_incremental_chunk(pair_buffer)

        rss_after_bytes = rss_before_bytes
        if total_ram_for_phase is not None:
            rss_after_bytes, _ = _current_rss_bytes_best_effort(total_ram_for_phase)
        accumulator_entry_bytes = int(memory_budget.INCREMENTAL_ACCUMULATOR_ENTRY_BYTES)
        if chunk_limits is not None and "accumulator_entry_bytes" in chunk_limits:
            accumulator_entry_bytes = int(chunk_limits["accumulator_entry_bytes"])
        phase_a_pair_buffer_peak_bytes = int(chunk_pairs_peak) * int(memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES)
        phase_a_predicted_peak_delta_bytes = (
            int(chunk_features_peak_bytes)
            + int(accumulator_entries_peak) * accumulator_entry_bytes
            + int(phase_a_pair_buffer_peak_bytes)
            + int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES)
        )
        phase_a_prediction = memory_budget.summarize_prediction_accuracy(
            stage_name="phase_a_seed_distances",
            predicted_peak_delta_bytes=phase_a_predicted_peak_delta_bytes,
            rss_before_bytes=rss_before_bytes,
            rss_peak_bytes=rss_peak_bytes,
            rss_after_bytes=rss_after_bytes,
        )
        logger.info(
            "Telemetry: constraint_batch stage=phase_a_seed_distances total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            int(constraint_pairs_total),
            int(constraint_partial_hits_total),
            int(constraint_unresolved_total),
            int(constraint_rust_batch_calls_total),
            ",".join(sorted(constraint_api_modes)) if constraint_api_modes else "none",
            float(constraint_batch_seconds_total),
            runtime_context.run_id,
        )

        return {
            "constraint_pairs_total": int(constraint_pairs_total),
            "constraint_chunks_total": int(constraint_chunks_total),
            "constraint_partial_supervision_hits": int(constraint_partial_hits_total),
            "constraint_unresolved_pairs": int(constraint_unresolved_total),
            "constraint_rust_batch_calls": int(constraint_rust_batch_calls_total),
            "constraint_batch_api_mode": ",".join(sorted(constraint_api_modes)) if constraint_api_modes else "none",
            "constraint_batch_seconds": float(constraint_batch_seconds_total),
            "accumulator_entries_peak": int(accumulator_entries_peak),
            "model_predict_seconds": float(model_predict_total_seconds),
            "phase_a_prediction_contract_version": str(phase_a_prediction["prediction_contract_version"]),
            "phase_a_predicted_peak_delta_bytes": int(phase_a_prediction["predicted_peak_delta_bytes"]),
            "phase_a_predicted_peak_rss_bytes": int(phase_a_prediction["predicted_peak_rss_bytes"]),
            # Backward-compatible alias; prefer phase_a_predicted_peak_delta_bytes.
            "phase_a_predicted_bytes": int(phase_a_prediction["predicted_bytes"]),
            "phase_a_rss_before_bytes": int(phase_a_prediction["rss_before_bytes"]),
            "phase_a_rss_peak_bytes": int(phase_a_prediction["rss_peak_bytes"]),
            "phase_a_rss_after_bytes": int(phase_a_prediction["rss_after_bytes"]),
            "phase_a_observed_peak_delta_bytes": int(phase_a_prediction["observed_peak_delta_bytes"]),
            "phase_a_prediction_error_ratio": float(phase_a_prediction["prediction_error_ratio"]),
            "phase_a_underpredicted": bool(phase_a_prediction["underpredicted"]),
            "phase_a_rss_source": str(rss_source),
            "phase_a_chunk_features_peak_bytes": int(chunk_features_peak_bytes),
            "phase_a_pair_buffer_peak_bytes": int(phase_a_pair_buffer_peak_bytes),
            "phase_a_accumulator_entry_bytes": int(accumulator_entry_bytes),
            "phase_a_pair_buffer_entry_bytes": int(memory_budget.PHASE_A_PAIR_BUFFER_ENTRY_BYTES),
            "phase_a_fixed_overhead_bytes": int(memory_budget.PHASE_A_FIXED_OVERHEAD_BYTES),
            "phase_a_adaptive_halvings": int(adaptive_halvings),
            "phase_a_accumulator_overflow_early_stop": bool(accumulator_overflow_early_stop),
        }

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
                            # if not a prefix/alias match to any existing name, disallow this merge
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
        return dict(pred_clusters)

    def _phase_split_subblock_fallback(
        self,
        block_signatures: list[str],
        subblocks: dict[str, list[str]],
        dataset: ANDData,
        all_unassigned: list[str],
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]],
        cluster_seeds_require: dict[str, int | str],
        recluster_map: dict[int | str, int | str],
        cluster_seeds_require_inverse: dict[int | str, list[str]],
        prevent_new_incompatibilities: bool,
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: RuntimeContext,
    ) -> dict[str, list[str]]:
        del all_unassigned  # fallback is intentionally subblock-local in Phase B/C/D
        cluster_seeds_require_original = copy.deepcopy(dataset.cluster_seeds_require)
        original_seed_sigs = set(cluster_seeds_require_original.keys())
        original_cluster_ids = set(str(cid) for cid in cluster_seeds_require_original.values())
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
        for sig_id, cluster_id in cluster_seeds_require_original.items():
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
        cluster_seeds_require_original = copy.deepcopy(dataset.cluster_seeds_require)
        all_unassigned = [
            signature_id for signature_id in block_signatures if signature_id not in cluster_seeds_require_original
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
        phase_a_pairs_total = 0
        phase_a_chunks_total = 0
        phase_a_accumulator_peak = 0
        phase_a_accumulator_peak_sample = 0
        phase_a_model_predict_seconds = 0.0
        phase_a_prediction_contract_version = "unknown"
        phase_a_predicted_peak_delta_bytes = 0
        phase_a_predicted_peak_rss_bytes = 0
        phase_a_predicted_bytes = 0
        phase_a_rss_before_bytes = 0
        phase_a_rss_peak_bytes = 0
        phase_a_rss_after_bytes = 0
        phase_a_rss_source = "unavailable"
        phase_a_observed_peak_delta_bytes = 0
        phase_a_prediction_error_ratio = 0.0
        phase_a_underpredicted = False
        phase_a_chunk_features_peak_bytes = 0
        phase_a_pair_buffer_peak_bytes = 0
        phase_a_accumulator_entry_bytes = 0
        phase_a_pair_buffer_entry_bytes = 0
        phase_a_fixed_overhead_bytes = 0
        phase_a_worst_sample: dict[str, Any] | None = None
        accumulator_runtime_calibrated = False
        phase_a_accumulator_overflow_early_stop = False
        phase_a_overflow_subblocks = 0

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
            )
            if bool(phase_a_telemetry.get("phase_a_accumulator_overflow_early_stop", False)):
                phase_a_accumulator_overflow_early_stop = True
                phase_a_overflow_subblocks += 1
            phase_a_pairs_total += int(phase_a_telemetry["constraint_pairs_total"])
            phase_a_chunks_total += int(phase_a_telemetry["constraint_chunks_total"])
            phase_a_accumulator_peak = max(phase_a_accumulator_peak, int(phase_a_telemetry["accumulator_entries_peak"]))
            phase_a_model_predict_seconds += float(phase_a_telemetry["model_predict_seconds"])
            candidate_ratio = float(phase_a_telemetry["phase_a_prediction_error_ratio"])
            candidate_observed_delta = int(phase_a_telemetry["phase_a_observed_peak_delta_bytes"])
            if phase_a_worst_sample is None:
                phase_a_worst_sample = phase_a_telemetry
            else:
                current_ratio = float(phase_a_worst_sample["phase_a_prediction_error_ratio"])
                current_observed_delta = int(phase_a_worst_sample["phase_a_observed_peak_delta_bytes"])
                if candidate_ratio > current_ratio or (
                    candidate_ratio == current_ratio and candidate_observed_delta > current_observed_delta
                ):
                    phase_a_worst_sample = phase_a_telemetry
            phase_a_underpredicted = phase_a_underpredicted or bool(phase_a_telemetry["phase_a_underpredicted"])

            # Runtime accumulator calibration: after the first subblock with meaningful data,
            # derive the effective bytes/entry from the observed telemetry and recalibrate
            # the accumulator budget for subsequent subblocks.
            if (
                not accumulator_runtime_calibrated
                and int(phase_a_telemetry["accumulator_entries_peak"]) > 100
                and int(phase_a_telemetry["phase_a_observed_peak_delta_bytes"]) > 0
            ):
                observed_delta = int(phase_a_telemetry["phase_a_observed_peak_delta_bytes"])
                modeled_non_accum = (
                    int(phase_a_telemetry["phase_a_chunk_features_peak_bytes"])
                    + int(phase_a_telemetry["phase_a_pair_buffer_peak_bytes"])
                    + int(phase_a_telemetry.get("phase_a_fixed_overhead_bytes", 0))
                )
                residual = observed_delta - modeled_non_accum
                entries = int(phase_a_telemetry["accumulator_entries_peak"])
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
                    accumulator_runtime_calibrated = True

        if phase_a_worst_sample is not None:
            phase_a_prediction_contract_version = str(phase_a_worst_sample["phase_a_prediction_contract_version"])
            phase_a_predicted_peak_delta_bytes = int(phase_a_worst_sample["phase_a_predicted_peak_delta_bytes"])
            phase_a_predicted_peak_rss_bytes = int(phase_a_worst_sample["phase_a_predicted_peak_rss_bytes"])
            phase_a_predicted_bytes = int(phase_a_worst_sample["phase_a_predicted_bytes"])
            phase_a_rss_before_bytes = int(phase_a_worst_sample["phase_a_rss_before_bytes"])
            phase_a_rss_peak_bytes = int(phase_a_worst_sample["phase_a_rss_peak_bytes"])
            phase_a_rss_after_bytes = int(phase_a_worst_sample["phase_a_rss_after_bytes"])
            phase_a_rss_source = str(phase_a_worst_sample["phase_a_rss_source"])
            phase_a_observed_peak_delta_bytes = int(phase_a_worst_sample["phase_a_observed_peak_delta_bytes"])
            phase_a_prediction_error_ratio = float(phase_a_worst_sample["phase_a_prediction_error_ratio"])
            phase_a_accumulator_peak_sample = int(phase_a_worst_sample.get("accumulator_entries_peak", 0))
            phase_a_chunk_features_peak_bytes = int(phase_a_worst_sample.get("phase_a_chunk_features_peak_bytes", 0))
            phase_a_pair_buffer_peak_bytes = int(phase_a_worst_sample.get("phase_a_pair_buffer_peak_bytes", 0))
            phase_a_accumulator_entry_bytes = int(phase_a_worst_sample.get("phase_a_accumulator_entry_bytes", 0))
            phase_a_pair_buffer_entry_bytes = int(phase_a_worst_sample.get("phase_a_pair_buffer_entry_bytes", 0))
            phase_a_fixed_overhead_bytes = int(phase_a_worst_sample.get("phase_a_fixed_overhead_bytes", 0))

        logger.info(
            "Telemetry: phase_split_phase_a_overflow overflow_early_stop=%s overflow_subblocks=%d "
            "accumulator_entries_peak=%d accumulator_max=%d run_id=%s",
            phase_a_accumulator_overflow_early_stop,
            phase_a_overflow_subblocks,
            phase_a_accumulator_peak,
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
            phase_a_pairs_total,
            phase_a_chunks_total,
            phase_a_accumulator_peak,
            phase_a_accumulator_peak_sample,
            phase_a_chunk_features_peak_bytes,
            phase_a_pair_buffer_peak_bytes,
            phase_a_accumulator_entry_bytes,
            phase_a_pair_buffer_entry_bytes,
            phase_a_fixed_overhead_bytes,
            phase_a_model_predict_seconds,
            phase_a_prediction_contract_version,
            phase_a_predicted_peak_delta_bytes,
            phase_a_predicted_peak_rss_bytes,
            phase_a_predicted_bytes,
            phase_a_rss_before_bytes,
            phase_a_rss_peak_bytes,
            phase_a_rss_after_bytes,
            phase_a_observed_peak_delta_bytes,
            phase_a_prediction_error_ratio,
            phase_a_underpredicted,
            phase_a_rss_source,
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
        estimated_converted_dict_bytes = phase_a_accumulator_peak * accumulator_entry_bytes_for_budget
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
                all_unassigned,
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
                phase_a_accumulator_overflow_early_stop=phase_a_accumulator_overflow_early_stop,
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
            phase_a_accumulator_overflow_early_stop=phase_a_accumulator_overflow_early_stop,
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
    ) -> dict[str, Any]:
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
        Returns
        -------
        Dict: incremental clustering result payload
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental")
        _sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=self.use_cache)
        if partial_supervision is None:
            partial_supervision = {}
        if batching_threshold is None or len(block_signatures) <= batching_threshold:
            return self.predict_incremental_helper(
                block_signatures,
                dataset,
                prevent_new_incompatibilities=prevent_new_incompatibilities,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )

        assert batching_threshold > 0, "Batching threshold must be positive"
        if len(dataset.cluster_seeds_require) == 0:
            logger.info(
                "No cluster seeds provided for subblocked incremental; "
                "falling back to monolithic incremental helper for partition parity."
            )
            return self.predict_incremental_helper(
                block_signatures,
                dataset,
                prevent_new_incompatibilities=prevent_new_incompatibilities,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
            )

        return self._predict_incremental_phase_split(
            block_signatures,
            dataset,
            prevent_new_incompatibilities,
            batching_threshold,
            partial_supervision,
            runtime_context,
            total_ram_bytes=total_ram_bytes,
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

        Notes:
        -This function was designed to work on a single block at a time.
        -This function should not be called directly. Use predict_incremental instead.
        -Constraint resolution is chunked by self.batch_size; featurization remains monolithic in this helper.

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
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        total_ram_bytes: Optional[int]
            Optional explicit RAM input used to derive featurization chunk budgets.
        Returns
        -------
        Dict: incremental clustering result payload
        """
        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_incremental")
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
        rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.use_cache,
        )
        constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
        signature_index_by_id: dict[str, int] | None = None
        if constraint_api_mode == "indexed" and rust_featurizer is not None:
            try:
                signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
            except Exception as exc:  # pragma: no cover - native extension optional
                if stage_uses_rust(runtime_context, "constraints"):
                    raise RuntimeError(
                        "Rust indexed constraint setup failed in strict rust backend "
                        f"(run_id={runtime_context.run_id} error={exc})"
                    ) from exc
                logger.warning(
                    "Rust indexed constraint setup failed in incremental helper; falling back to per-pair "
                    "constraints: %s",
                    exc,
                )
                constraint_api_mode = "fallback"
        signature_to_cluster_to_average_dist: dict[str, dict[int | str, tuple[float, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0))
        )
        assigned_signature_ids: list[str] = list(cluster_seeds_require.keys())
        pair_chunk_size = max(1, int(self.batch_size))
        constraint_partial_hits_total = 0
        constraint_unresolved_total = 0
        constraint_rust_batch_calls_total = 0
        constraint_batch_seconds_total = 0.0
        constraint_api_modes: set[str] = set()

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
                    labels, batch_telemetry = _resolve_constraint_labels_batch(
                        dataset,
                        pair_id_batch,
                        partial_supervision=partial_supervision,
                        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=False,
                        rust_featurizer=rust_featurizer,
                        use_rust_constraints=use_rust_constraints,
                        runtime_context=runtime_context,
                        use_cache=self.use_cache,
                        num_threads=self.n_jobs,
                        constraint_api_mode=constraint_api_mode,
                        signature_index_by_id=signature_index_by_id,
                    )
                    constraint_partial_hits_total += int(batch_telemetry["partial_supervision_hits"])
                    constraint_unresolved_total += int(batch_telemetry["unresolved_pairs"])
                    constraint_rust_batch_calls_total += int(batch_telemetry["rust_batch_call_count"])
                    constraint_batch_seconds_total += float(batch_telemetry["elapsed_seconds"])
                    constraint_api_modes.add(str(batch_telemetry["api_mode"]))
                    all_pairs.extend(
                        (sig_id_1, sig_id_2, label)
                        for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=False)
                    )
                    pair_id_batch = []

        if pair_id_batch:
            labels, batch_telemetry = _resolve_constraint_labels_batch(
                dataset,
                pair_id_batch,
                partial_supervision=partial_supervision,
                use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=False,
                rust_featurizer=rust_featurizer,
                use_rust_constraints=use_rust_constraints,
                runtime_context=runtime_context,
                use_cache=self.use_cache,
                num_threads=self.n_jobs,
                constraint_api_mode=constraint_api_mode,
                signature_index_by_id=signature_index_by_id,
            )
            constraint_partial_hits_total += int(batch_telemetry["partial_supervision_hits"])
            constraint_unresolved_total += int(batch_telemetry["unresolved_pairs"])
            constraint_rust_batch_calls_total += int(batch_telemetry["rust_batch_call_count"])
            constraint_batch_seconds_total += float(batch_telemetry["elapsed_seconds"])
            constraint_api_modes.add(str(batch_telemetry["api_mode"]))
            all_pairs.extend(
                (sig_id_1, sig_id_2, label) for (sig_id_1, sig_id_2), label in zip(pair_id_batch, labels, strict=False)
            )

        logger.info(
            "Telemetry: constraint_batch stage=predict_incremental_helper total_pairs=%d partial_supervision_hits=%d "
            "unresolved_pairs=%d rust_batch_calls=%d api_mode=%s seconds=%.3f run_id=%s",
            len(all_pairs),
            constraint_partial_hits_total,
            constraint_unresolved_total,
            constraint_rust_batch_calls_total,
            ",".join(sorted(constraint_api_modes)) if constraint_api_modes else "none",
            constraint_batch_seconds_total,
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
        rust_failure_counts: dict[str, int] = {"main": 0, "nameless": 0}
        batch_predictions, model_predict_seconds = _predict_and_combine(
            self.classifier,
            self.nameless_classifier,
            batch_features,
            batch_labels,
            batch_nameless_features,
            "incremental",
            rust_failure_counts,
            runtime_context=runtime_context,
        )
        logger.info("Telemetry: model_predict_total seconds=%.3f blocks=1", model_predict_seconds)

        logger.info("Computing average distances for unassigned signatures")
        for signature_pair, dist in zip(all_pairs, batch_predictions, strict=False):
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


class PairwiseModeler:
    """
    Wrapper to learn the pairwise model + hyperparameter optimization

    Parameters
    ----------
    estimator: sklearn compatible classifier
        A binary classifier with fit/predict interface.
        Defaults to LGBMClassifier if not specified. Will be cloned.
    search_space: Dict:
            A hyperopt search space for hyperparam optimization.
            Defaults to an appropriate LGBMClassifier space if not specified.
    monotone_constraints: string
            Monotonic constraints for lightbm only.
            Defaults to None and is not used.
    n_iter: int
        Number of iterations for hyperparam optimization.
    n_jobs: int
        Parallelization for the classifier.
        Note: the hyperopt is serial, but can be made semi-parallel with batch search.
    random_state: int
        Random state for classifier and hyperopt.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        search_space: dict[str, Any] | None = None,
        monotone_constraints: str | None = None,
        n_iter: int = 50,
        n_jobs: int = 16,  # for the model, not the hyperopt
        random_state: int = 42,
    ):
        if estimator is None:
            self.estimator = lgb.LGBMClassifier(
                objective="binary",
                metric="auc",  # lightgbm doesn't do F1 directly
                n_jobs=n_jobs,
                verbose=-1,
                tree_learner="data",
                random_state=random_state,
            )
        else:
            self.estimator = clone(estimator)

        if search_space is None:
            self.search_space = {
                "learning_rate": hp.loguniform("learning_rate", -7, 0),
                "num_leaves": scope.int(hp.qloguniform("num_leaves", 2, 7, 1)),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                "subsample": hp.uniform("subsample", 0.5, 1),
                "min_child_samples": scope.int(hp.qloguniform("min_child_samples", 3, 9, 1)),
                "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
                "reg_alpha": hp.loguniform("reg_alpha", -16, 2),
                "reg_lambda": hp.loguniform("reg_lambda", -16, 2),
                "n_estimators": scope.int(hp.quniform("n_estimators", 1000, 2500, 1)),
                "max_depth": scope.int(hp.quniform("max_depth", 1, 100, 1)),
                "min_split_gain": hp.uniform("min_split_gain", 0, 2),
            }
        else:
            self.search_space = search_space

        self.monotone_constraints = monotone_constraints
        if self.monotone_constraints is not None and isinstance(self.estimator, lgb.LGBMClassifier):
            self.estimator.set_params(monotone_constraints=self.monotone_constraints)
            self.estimator.set_params(monotone_constraints_method="advanced")
            self.search_space["monotone_penalty"] = hp.uniform("monotone_penalty", 0, 5)

        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params: dict | None = None
        self.hyperopt_trials_store: Trials | dict[Any, Any] | None = None
        self.classifier: Any | None = None

    def fit(
        self,
        X_train: np.ndarray[Any, Any] | None | Any,
        y_train: np.ndarray[Any, Any] | None | Any,
        X_val: np.ndarray[Any, Any] | None | Any,
        y_val: np.ndarray[Any, Any] | None | Any,
    ) -> Trials | dict[Any, Any]:
        """
        Fits the classifier

        Parameters
        ----------
        X_train: np.ndarray
            feature matrix for the training set
        y_train: np.ndarray
            labels for the training set
        X_val: np.ndarray
            feature matrix for the validation set
        y_val: np.ndarray
            labels for the validation set

        Returns
        -------
        Trials: the Trials object from hyperparameter optimization
        """
        if len(self.search_space) > 0:

            def obj(params):
                params = {k: intify(v) for k, v in params.items()}
                self.estimator.set_params(**params)
                self.estimator.fit(X_train, y_train)
                y_pred_proba = self.estimator.predict_proba(X_val)[:, 1]
                return -roc_auc_score(y_val, y_pred_proba)

            self.hyperopt_trials_store = Trials()
            _ = fmin(
                fn=obj,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.n_iter,
                trials=self.hyperopt_trials_store,
                rstate=np.random.default_rng(self.random_state),
            )
            assert isinstance(self.hyperopt_trials_store, Trials)
            best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
            self.best_params = {k: intify(v) for k, v in best_params.items()}
            self.estimator.set_params(**self.best_params)
        else:
            self.best_params = {}
            self.hyperopt_trials_store = {}

        # refitting but only on training data so as not to leak anything
        self.classifier = self.estimator.fit(X_train, y_train)

        assert self.hyperopt_trials_store is not None
        return self.hyperopt_trials_store

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classifier is not None, "You need to call fit first"
        return self.classifier.predict_proba(X)


def intify(x):
    """Hyperopt is bad at ints..."""
    if hasattr(x, "is_integer") and x.is_integer():
        return int(x)
    else:
        return x


class FastCluster(TransformerMixin, BaseEstimator):
    """
    A scikit-learn wrapper for fastcluster.
    Inputs:
        linkage: string (default="average")
            Agglomerative linkage method. Defaults to "average".
            Must be one of "'complete', 'average', 'single,
            'weighted', 'ward', 'centroid', 'median'."
        eps: float (default=0.5)
            Cutoff used to determine number of clusters.
        preserve_input: bool (default=True)
            Whether to preserve the X input or modify in place.
            Defaults to False, which modifies in place.
        input_as_observation_matrix: bool (default=False)
            If True, the input to fit/transform must be a 2-D array
            of observation vectors (N by d). If False input to fit/transform
            must be a 1-D condensed distance matrix, then it must be a
            (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix, and
            d is the dimensionality of the vector space.

    Note: FastCluster does *not* support two-dimensional distance matrices
    as input. They *must* be flattened. For more details, please see:
    https://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf
    """

    def __init__(
        self,
        linkage: str = "average",
        eps: float = 0.5,
        preserve_input: bool = True,
        input_as_observation_matrix: bool = False,
    ):
        if linkage not in {
            "complete",
            "average",
            "weighted",
            "ward",
            "centroid",
            "median",
            "single",
        }:
            raise Exception(
                "The 'linkage' parameter has to be one of: "
                + "'single', complete', 'average', 'weighted', 'ward', 'centroid', 'median'."
            )

        self.linkage = linkage
        self.eps = eps
        self.preserve_input = preserve_input
        self.input_as_observation_matrix = input_as_observation_matrix
        self.labels_ = None

    # ---- new: robust get_params ----
    def get_params(self, deep=True):
        """
        Return params but gracefully handle the case where an instance
        (e.g., loaded from an old pickle) is missing attributes.
        """
        params = {}
        sig = inspect.signature(self.__class__.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # prefer the runtime attribute if present, otherwise the __init__ default
            if hasattr(self, name):
                params[name] = getattr(self, name)
            else:
                params[name] = param.default if param.default is not inspect._empty else None

        if deep:
            # sklearn convention: include nested estimator params with __ separator
            for key, val in list(params.items()):
                if hasattr(val, "get_params"):
                    for subk, subv in val.get_params(deep=True).items():
                        params[f"{key}__{subk}"] = subv
        return params

    # ---- new: ensure defaults after unpickling ----
    def __setstate__(self, state):
        """
        Called on unpickle. Populate any missing ctor attrs with their defaults.
        """
        self.__dict__.update(state)
        sig = inspect.signature(self.__class__.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if not hasattr(self, name):
                default = param.default if param.default is not inspect._empty else None
                setattr(self, name, default)

    def fit(self, X: np.ndarray) -> FastCluster:
        """
        Fit the estimator on input data. The results are stored in self.labels_.
        Parameters
        ----------
        X: np.array
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix. If X is 2-D
            then the flag `input_as_observation_matrix` must be set to True in init.
        Returns
        -------
        self
        """
        X = np.asarray(X)
        if len(X.shape) == 1 and self.input_as_observation_matrix:
            raise Exception(
                "Input to fit is one-dimensional, but input_as_observation_matrix flag is set to True. "
                "If you intended to pass in an observation matrix, it must be 2-D (N x feature_dimension)."
            )
        elif len(X.shape) == 2 and not self.input_as_observation_matrix:
            raise Exception(
                "Input to fit is two-dimensional, but input_as_observation_matrix flag is set to False. "
                "If you intended to pass in a distance matrix, it must be flattened (1-D)."
            )
        elif len(X.shape) > 2:
            raise Exception("The input to fit can only be one-dimensional or two-dimensional.")
        Z = linkage(X, self.linkage, preserve_input=self.preserve_input)
        self.labels_ = fcluster(Z, t=self.eps, criterion="distance")
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """
        Fit the estimator on input data, and returns results.
        Parameters
        ----------
        X: np.array
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix.
        Returns
        -------
        np.array: A N-length array of clustering labels.
        """
        del y, fit_params
        self.fit(X)
        return self.labels_  # type: ignore

    def transform(self, X: np.ndarray):
        raise Exception("FastCluster has no inductive mode. Use 'fit' or 'fit_transform' instead.")


class VotingClassifier:
    """
    Stripped-down version of VotingClassifier that uses prefit estimators

    Parameters
    ----------
    estimators: List[sklearn classifier]
        A list of sklearn classifiers that support predict_proba.
    voting: string
        Type of voting.
        Defaults to "hard", can also be "soft".
        "soft" means "take the highest average probability class" and
        "hard" means "take the class that the plurality of the models pick"
    weights: List or np.array
        Weights for each estimator.
    """

    def __init__(self, estimators, voting="soft", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        predictions : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == "soft":
            predictions = np.argmax(self.predict_proba(X), axis=1)
        elif self.voting == "hard":
            predictions = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=self._predict(X).astype("int"),
            )
        else:
            raise Exception("Voting type must be one of 'soft' or 'hard'")
        return predictions

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def predict_proba(self, X):
        """
        Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        if self.voting == "hard":
            raise AttributeError(f"predict_proba is not available when voting={self.voting!r}")
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """
        Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        if self.voting == "soft":
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict(X) for clf in self.estimators]).T
