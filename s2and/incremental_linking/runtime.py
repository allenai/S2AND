"""Private link-or-abstain orchestration helpers for incremental linking."""

from __future__ import annotations

import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from s2and import feature_port
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import IncrementalLinkingArtifact
from s2and.incremental_linking.features import LinkerFeatureMatrix, assemble_linker_feature_matrix
from s2and.incremental_linking.linker_pairwise import (
    PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES,
    PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    LinkerCandidateBatch,
    PairwiseAggregateStats,
    _localize_row_indices,
    aggregate_pair_feature_chunk_nan_aware,
    compute_candidate_batch_pairwise_aggregate_stats_rust,
    compute_linker_pair_chunk_plan,
)
from s2and.incremental_linking.retrieval import LinkerRetrievalBatch, build_linker_retrieval_batch_rust
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features_with_telemetry
from s2and.runtime import build_runtime_context

LinkAction = Literal["link", "abstain"]
_LETTERS_RE = re.compile(r"[A-Za-z]+")

# Production 1.2 dense output semantics. The pairwise distance model preserves
# NaNs internally; only the exported pw_* aggregate features are zero-filled.
_PAIRWISE_MODEL_NAN_VALUE: float = float("nan")
_PAIRWISE_AGGREGATE_NAN_VALUE: float = 0.0


@dataclass(frozen=True)
class CandidateBatchPairwiseModelResult:
    """Fused candidate-batch pairwise model outputs and promoted aggregates."""

    row_signals: dict[str, np.ndarray]
    pairwise_stats: PairwiseAggregateStats
    telemetry: dict[str, int | float]


def signature_id_to_index_map(featurizer: Any) -> dict[str, int]:
    """Return the Rust signature index map for candidate-batch construction."""

    return {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}


@dataclass(frozen=True)
class LinkOrAbstainDecision:
    """One private compact decision for a query signature."""

    query_signature_index: int
    action: LinkAction
    row_index: int | None
    component_key: str | None
    score: float | None
    runner_up_score: float | None
    score_margin: float | None


@dataclass(frozen=True)
class LinkOrAbstainCompactResult:
    """Private compact result for artifact-scored candidate rows."""

    probabilities: np.ndarray
    decisions: tuple[LinkOrAbstainDecision, ...]


@dataclass(frozen=True)
class LinkOrAbstainPrivateResult:
    """Private end-to-end result for retrieved candidates and the M3a production slice.

    The retrieved-candidate slice populates `feature_matrix`, `compact_result`, and
    `telemetry`. The M3a production slice additionally populates `retrieval_batch`,
    `pairwise_model_result`, and `linked_signature_clusters`.
    """

    feature_matrix: LinkerFeatureMatrix
    compact_result: LinkOrAbstainCompactResult
    telemetry: dict[str, int | float | str]
    retrieval_batch: LinkerRetrievalBatch | None = None
    pairwise_model_result: CandidateBatchPairwiseModelResult | None = None
    linked_signature_clusters: dict[str, Any] | None = None


def _ordered_group_indices(query_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    groups: list[np.ndarray] = []
    for query_index in tuple(dict.fromkeys(int(value) for value in query_indices)):
        groups.append(np.flatnonzero(query_indices == np.uint32(query_index)))
    return tuple(groups)


def _best_row_for_group(
    group: np.ndarray,
    *,
    probabilities: np.ndarray,
    retrieval_ranks: np.ndarray | None,
    component_keys: tuple[object, ...] | None,
) -> int:
    def sort_key(row_index: int) -> tuple[float, int, str]:
        rank = 0 if retrieval_ranks is None else int(retrieval_ranks[row_index])
        component_key = "" if component_keys is None else str(component_keys[row_index])
        return (-float(probabilities[row_index]), rank, component_key)

    return min((int(row_index) for row_index in group), key=sort_key)


def _normalize_letters(value: Any) -> str:
    if value is None:
        return ""
    return "".join(_LETTERS_RE.findall(str(value))).lower()


def _row_signal_string(row_signals: Mapping[str, Any] | None, name: str, row_index: int) -> str:
    if row_signals is None or name not in row_signals:
        return ""
    values = np.asarray(row_signals[name], dtype=object)
    if values.ndim != 1 or row_index >= len(values):
        return ""
    value = values[row_index]
    if value is None:
        return ""
    return str(value)


def _first_name_bucket(row_signals: Mapping[str, Any] | None, row_index: int) -> str:
    token = _normalize_letters(_row_signal_string(row_signals, "query_first_token", row_index))
    if not token and _row_signal_string(row_signals, "query_view", row_index) == "initial_only":
        return "single_letter_first"
    return "single_letter_first" if len(token) <= 1 else "multi_letter_first"


def _artifact_gate_thresholds(
    artifact: IncrementalLinkingArtifact,
    *,
    row_index: int,
    has_runner_up: bool,
    row_signals: Mapping[str, Any] | None,
) -> tuple[float | None, float | None]:
    gate_config = artifact.metadata.gate_config
    if "score_threshold" not in gate_config:
        raise ValueError("artifact gate_config must contain score_threshold")
    score_threshold = float(gate_config["score_threshold"])
    margin_threshold = gate_config.get("margin_threshold")
    single_candidate_score_threshold = gate_config.get("single_candidate_score_threshold")
    bucketed_score_thresholds = gate_config.get("bucketed_score_thresholds")
    single_candidate = not has_runner_up
    if bucketed_score_thresholds is not None:
        bucket = (
            ("single_candidate" if single_candidate else "multi_candidate")
            + "|"
            + _first_name_bucket(row_signals, row_index)
        )
        fallback_score = (
            float(single_candidate_score_threshold)
            if single_candidate and single_candidate_score_threshold is not None
            else score_threshold
        )
        resolved_score = float(dict(bucketed_score_thresholds).get(bucket, fallback_score))
        if single_candidate:
            return resolved_score, None
        bucketed_margin_thresholds = gate_config.get("bucketed_margin_thresholds")
        if bucketed_margin_thresholds is not None:
            margin_value = dict(bucketed_margin_thresholds).get(bucket)
            return resolved_score, None if margin_value is None else float(margin_value)
        bucketed_margin_threshold = gate_config.get("bucketed_margin_threshold")
        if bucketed_margin_threshold is not None:
            return resolved_score, float(bucketed_margin_threshold)
        return resolved_score, None
    if single_candidate:
        if single_candidate_score_threshold is None:
            return None, None
        return float(single_candidate_score_threshold), None
    return score_threshold, None if margin_threshold is None else float(margin_threshold)


def _passes_artifact_gate(
    artifact: IncrementalLinkingArtifact,
    *,
    row_index: int,
    score: float,
    margin: float | None,
    has_runner_up: bool,
    row_signals: Mapping[str, Any] | None,
) -> bool:
    score_threshold, margin_threshold = _artifact_gate_thresholds(
        artifact,
        row_index=row_index,
        has_runner_up=has_runner_up,
        row_signals=row_signals,
    )
    passes_score = score_threshold is None or score >= score_threshold
    if margin_threshold is None:
        return passes_score
    return passes_score or (margin is not None and margin >= margin_threshold)


def _predict_incremental_link_or_abstain_compact(
    artifact: IncrementalLinkingArtifact,
    feature_matrix: LinkerFeatureMatrix,
    *,
    row_signals: Mapping[str, Any] | None = None,
) -> LinkOrAbstainCompactResult:
    """Score artifact-ordered rows and apply the artifact's bucketed gate.

    This is intentionally not a public API. It exists to keep the first vertical
    slice concrete while retrieval policy, constraint handling, and telemetry are
    still private implementation details.
    """

    candidate_batch = feature_matrix.candidate_batch
    if candidate_batch.row_query_signature_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for compact decisions")
    probabilities = artifact.predict_probabilities(feature_matrix.matrix)
    if len(probabilities) != candidate_batch.row_count:
        raise ValueError("artifact probability count must match candidate row_count")
    query_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)
    retrieval_ranks = (
        None
        if candidate_batch.retrieval_ranks is None
        else np.asarray(candidate_batch.retrieval_ranks, dtype=np.uint16)
    )
    component_keys = candidate_batch.row_component_keys
    decisions: list[LinkOrAbstainDecision] = []
    for group in _ordered_group_indices(query_indices):
        if len(group) == 0:
            continue
        best_row = _best_row_for_group(
            group,
            probabilities=probabilities,
            retrieval_ranks=retrieval_ranks,
            component_keys=component_keys,
        )
        runner_ups = [int(row_index) for row_index in group if int(row_index) != best_row]
        runner_up_score = max((float(probabilities[row_index]) for row_index in runner_ups), default=np.nan)
        margin = None if np.isnan(runner_up_score) else float(probabilities[best_row] - runner_up_score)
        has_runner_up = margin is not None
        passes_gate = _passes_artifact_gate(
            artifact,
            row_index=best_row,
            score=float(probabilities[best_row]),
            margin=margin,
            has_runner_up=has_runner_up,
            row_signals=row_signals,
        )
        action: LinkAction = "link" if passes_gate else "abstain"
        component_key = None
        if action == "link" and component_keys is not None:
            component_key = str(component_keys[best_row])
        decisions.append(
            LinkOrAbstainDecision(
                query_signature_index=int(query_indices[best_row]),
                action=action,
                row_index=best_row if action == "link" else None,
                component_key=component_key,
                score=float(probabilities[best_row]),
                runner_up_score=None if np.isnan(runner_up_score) else float(runner_up_score),
                score_margin=margin,
            )
        )
    return LinkOrAbstainCompactResult(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        decisions=tuple(decisions),
    )


def _pairwise_model_feature_indices(featurizer_info: FeaturizationInfo) -> tuple[int, ...]:
    selected: set[int] = set()
    for feature_group in featurizer_info.features_to_use:
        selected.update(featurizer_info.feature_group_to_index[str(feature_group)])
    return tuple(sorted(selected))


def _matrix_positions(matrix_indices: Sequence[int], selected_indices: Sequence[int]) -> tuple[int, ...]:
    position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
    missing = [int(index) for index in selected_indices if int(index) not in position_by_index]
    if missing:
        raise ValueError(f"selected pairwise model feature indices are missing from matrix_indices: {missing[:5]}")
    return tuple(position_by_index[int(index)] for index in selected_indices)


def _predict_pairwise_class0(classifier: Any, features: np.ndarray, *, num_threads: int) -> np.ndarray:
    # Estimator threading is configured through propagated n_jobs; predict_proba(num_threads=...)
    # is LightGBM-specific and breaks sklearn-compatible wrappers.
    del num_threads

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        probabilities = classifier.predict_proba(features)
    return np.asarray(probabilities, dtype=np.float64)[:, 0]


def _predict_pairwise_model_distances(
    *,
    classifier: Any,
    features: np.ndarray,
    labels: np.ndarray,
    num_threads: int,
    nameless_classifier: Any | None = None,
    nameless_features: np.ndarray | None = None,
) -> np.ndarray:
    predictions = np.zeros(len(labels), dtype=np.float64)
    predict = np.isnan(labels)
    if np.any(predict):
        predicted = _predict_pairwise_class0(classifier, features[predict], num_threads=num_threads)
        if nameless_classifier is not None and nameless_features is not None:
            nameless_predicted = _predict_pairwise_class0(
                nameless_classifier,
                nameless_features[predict],
                num_threads=num_threads,
            )
            predicted = (predicted + nameless_predicted) / 2.0
        predictions[predict] = predicted
    return predictions


def _predict_pairwise_distances(
    *,
    classifier: Any,
    features: np.ndarray,
    labels: np.ndarray,
    num_threads: int,
    nameless_classifier: Any | None = None,
    nameless_features: np.ndarray | None = None,
) -> np.ndarray:
    predictions = _predict_pairwise_model_distances(
        classifier=classifier,
        features=features,
        labels=labels,
        num_threads=num_threads,
        nameless_classifier=nameless_classifier,
        nameless_features=nameless_features,
    )
    not_predict = ~np.isnan(labels)
    if np.any(not_predict):
        predictions[not_predict] = labels[not_predict] + LARGE_INTEGER
    return predictions


def _update_top_distances(top_distances: np.ndarray, row_index: int, distance: float) -> None:
    row = top_distances[row_index]
    if distance >= row[-1]:
        return
    row[-1] = distance
    row.sort()


def _distance_row_signals(
    *,
    counts: np.ndarray,
    sums: np.ndarray,
    mins: np.ndarray,
    top_distances: np.ndarray,
    empty_distance_value: float = 1.0,
) -> dict[str, np.ndarray]:
    row_count = len(counts)
    observed = counts > 0
    min_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    top3_mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    top5_mean_distance = np.full(row_count, float(empty_distance_value), dtype=np.float32)
    pair_count = counts.astype(np.float32, copy=False)
    if np.any(observed):
        min_distance[observed] = mins[observed].astype(np.float32, copy=False)
        mean_distance[observed] = (sums[observed] / counts[observed]).astype(np.float32, copy=False)
        for row_index in np.flatnonzero(observed):
            finite = top_distances[row_index][np.isfinite(top_distances[row_index])]
            if len(finite) == 0:
                continue
            top3_mean_distance[row_index] = float(np.mean(finite[:3]))
            top5_mean_distance[row_index] = float(np.mean(finite[:5]))
    return {
        "min_distance": min_distance,
        "mean_distance": mean_distance,
        "top3_mean_distance": top3_mean_distance,
        "top5_mean_distance": top5_mean_distance,
        "pair_count": pair_count,
    }


def _accumulate_pairwise_distance_chunk(
    *,
    dataset: ANDData,
    row_indices: np.ndarray,
    row_count: int,
    model_distances: np.ndarray,
    labels: np.ndarray,
    n_jobs: int,
    runtime_context: Any | None,
    use_cache: bool,
    featurizer: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if featurizer is not None and not callable(getattr(featurizer, "linker_pair_distance_accumulators", None)):
        raise RuntimeError(
            "RustFeaturizer.linker_pair_distance_accumulators is required for promoted linker distance "
            "aggregation; rebuild/install the current s2and-rust extension."
        )
    return feature_port.build_linker_pair_distance_accumulators_rust(
        dataset,
        row_indices,
        int(row_count),
        model_distances,
        pair_labels=labels,
        num_threads=max(1, int(n_jobs)),
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
    )


def compute_candidate_batch_pairwise_model_and_aggregate_stats(
    dataset: ANDData,
    candidate_batch: LinkerCandidateBatch,
    *,
    classifier: Any,
    featurizer_info: FeaturizationInfo,
    nameless_classifier: Any | None = None,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    pair_labels: np.ndarray | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    pairwise_model_nan_value: float = _PAIRWISE_MODEL_NAN_VALUE,
    pairwise_aggregate_nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> CandidateBatchPairwiseModelResult:
    """Score candidate pairs and compute promoted pairwise aggregates in one Rust feature pass.

    Production defaults reproduce the dense production 1.2 matrix by preserving
    NaNs for pairwise distance model inputs and zero-filling the exported
    promoted pairwise aggregate values.
    """

    start_seconds = time.perf_counter()
    row_count = int(candidate_batch.row_count)
    pair_count = int(candidate_batch.pair_count)
    labels = (
        np.full(pair_count, np.nan, dtype=np.float64)
        if pair_labels is None
        else np.asarray(pair_labels, dtype=np.float64)
    )
    if labels.shape != (pair_count,):
        raise ValueError(f"pair_labels must have shape ({pair_count},), got {labels.shape}")

    main_indices = _pairwise_model_feature_indices(featurizer_info)
    if not main_indices:
        raise ValueError("featurizer_info selects no pairwise model features")
    nameless_indices = (
        ()
        if nameless_classifier is None or nameless_featurizer_info is None
        else _pairwise_model_feature_indices(nameless_featurizer_info)
    )
    aggregate_indices = tuple(int(index) for index in PROMOTED_PAIRWISE_AGG_FEATURE_INDICES)
    matrix_indices = tuple(dict.fromkeys((*main_indices, *nameless_indices, *aggregate_indices)))
    main_positions = _matrix_positions(matrix_indices, main_indices)
    nameless_positions = _matrix_positions(matrix_indices, nameless_indices) if nameless_indices else ()
    aggregate_feature_names = tuple(PROMOTED_PAIRWISE_AGG_BASE_FEATURE_NAMES)
    aggregate_columns = tuple(
        f"pw_{stat}_{feature_name}" for stat in ("min", "mean", "max") for feature_name in aggregate_feature_names
    )
    plan = compute_linker_pair_chunk_plan(
        total_pairs=pair_count,
        row_count=row_count,
        matrix_feature_count=len(matrix_indices),
        aggregate_feature_count=len(aggregate_indices),
        total_ram_bytes=total_ram_bytes,
    )
    aggregate_counts = np.zeros(row_count, dtype=np.uint64)
    aggregate_valid_counts = np.zeros((row_count, len(aggregate_indices)), dtype=np.uint64)
    aggregate_sums = np.zeros((row_count, len(aggregate_indices)), dtype=np.float64)
    aggregate_mins = np.full((row_count, len(aggregate_indices)), np.inf, dtype=np.float64)
    aggregate_maxs = np.full((row_count, len(aggregate_indices)), -np.inf, dtype=np.float64)
    distance_counts = np.zeros(row_count, dtype=np.uint64)
    distance_sums = np.zeros(row_count, dtype=np.float64)
    distance_mins = np.full(row_count, np.inf, dtype=np.float64)
    top_distances = np.full((row_count, 5), np.inf, dtype=np.float64)
    hard_disallow_distance_pair_count = 0
    if featurizer is None:
        featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
            dataset,
            runtime_context=runtime_context,
            use_cache=use_cache,
        )

    chunk_pairs = int(plan["chunk_pairs"])
    chunk_count = 0
    feature_seconds = 0.0
    predict_seconds = 0.0
    for chunk_start in range(0, pair_count, chunk_pairs):
        chunk_stop = min(pair_count, chunk_start + chunk_pairs)
        row_chunk = candidate_batch.pair_row_indices[chunk_start:chunk_stop]
        global_rows, local_row_indices = _localize_row_indices(row_chunk)
        feature_start = time.perf_counter()
        pair_features, _rust_counts, _rust_sums, _rust_mins, _rust_maxs = (
            feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
                dataset,
                candidate_batch.left_signature_indices[chunk_start:chunk_stop],
                candidate_batch.right_signature_indices[chunk_start:chunk_stop],
                local_row_indices,
                len(global_rows),
                matrix_indices=list(matrix_indices),
                aggregate_indices=list(aggregate_indices),
                num_threads=max(1, int(n_jobs)),
                nan_value=float(pairwise_model_nan_value),
                aggregate_nan_value=float(pairwise_aggregate_nan_value),
                runtime_context=runtime_context,
                use_cache=use_cache,
                featurizer=featurizer,
            )
        )
        feature_seconds += time.perf_counter() - feature_start
        chunk_count += 1

        counts, valid_counts, sums, mins, maxs = aggregate_pair_feature_chunk_nan_aware(
            pair_features=pair_features,
            local_row_indices=local_row_indices,
            row_count=len(global_rows),
            matrix_indices=matrix_indices,
            aggregate_indices=aggregate_indices,
            nan_value=float(pairwise_aggregate_nan_value),
        )
        observed = counts > 0
        if np.any(observed):
            rows = global_rows[observed]
            aggregate_counts[rows] += counts[observed].astype(np.uint64, copy=False)
            aggregate_valid_counts[rows] += valid_counts[observed]
            aggregate_sums[rows] += sums[observed]
            aggregate_mins[rows] = np.minimum(aggregate_mins[rows], mins[observed])
            aggregate_maxs[rows] = np.maximum(aggregate_maxs[rows], maxs[observed])

        predict_start = time.perf_counter()
        labels_chunk = labels[chunk_start:chunk_stop]
        model_pair_features = pair_features
        if not np.isnan(float(pairwise_model_nan_value)):
            model_pair_features = pair_features.copy()
            model_pair_features[np.isnan(model_pair_features)] = float(pairwise_model_nan_value)
        model_distances = _predict_pairwise_model_distances(
            classifier=classifier,
            features=model_pair_features[:, main_positions],
            labels=labels_chunk,
            num_threads=max(1, int(n_jobs)),
            nameless_classifier=nameless_classifier,
            nameless_features=model_pair_features[:, nameless_positions] if nameless_positions else None,
        )
        predict_seconds += time.perf_counter() - predict_start
        distance_accumulators = _accumulate_pairwise_distance_chunk(
            dataset=dataset,
            row_indices=local_row_indices,
            row_count=len(global_rows),
            model_distances=model_distances,
            labels=labels_chunk,
            n_jobs=n_jobs,
            runtime_context=runtime_context,
            use_cache=use_cache,
            featurizer=featurizer,
        )
        chunk_counts, chunk_sums, chunk_mins, chunk_top_distances, chunk_hard_disallow_count = distance_accumulators
        observed_distance_rows = chunk_counts > 0
        if np.any(observed_distance_rows):
            rows = global_rows[observed_distance_rows]
            distance_counts[rows] += chunk_counts[observed_distance_rows].astype(np.uint64, copy=False)
            distance_sums[rows] += chunk_sums[observed_distance_rows]
            distance_mins[rows] = np.minimum(distance_mins[rows], chunk_mins[observed_distance_rows])
            for local_row_index, global_row_index in zip(
                np.flatnonzero(observed_distance_rows),
                rows,
                strict=True,
            ):
                row_top_distances = chunk_top_distances[int(local_row_index)]
                finite = row_top_distances[np.isfinite(row_top_distances)]
                for value in finite:
                    _update_top_distances(top_distances, int(global_row_index), float(value))
        hard_disallow_distance_pair_count += int(chunk_hard_disallow_count)

    pairwise_stats = PairwiseAggregateStats(
        counts=aggregate_counts,
        sums=aggregate_sums,
        mins=aggregate_mins,
        maxs=aggregate_maxs,
        base_feature_names=aggregate_feature_names,
        aggregate_feature_columns=aggregate_columns,
        chunk_plan=plan,
        chunk_count=int(chunk_count),
        matrix_indices=matrix_indices,
        aggregate_indices=aggregate_indices,
        valid_counts=aggregate_valid_counts,
    )
    telemetry: dict[str, int | float] = {
        "candidate_row_count": row_count,
        "pair_count": pair_count,
        "chunk_count": int(chunk_count),
        "matrix_feature_count": int(len(matrix_indices)),
        "aggregate_feature_count": int(len(aggregate_indices)),
        "feature_seconds": float(feature_seconds),
        "predict_seconds": float(predict_seconds),
        "total_seconds": float(time.perf_counter() - start_seconds),
        "hard_disallow_distance_pair_count": int(hard_disallow_distance_pair_count),
    }
    return CandidateBatchPairwiseModelResult(
        row_signals=_distance_row_signals(
            counts=distance_counts,
            sums=distance_sums,
            mins=distance_mins,
            top_distances=top_distances,
        ),
        pairwise_stats=pairwise_stats,
        telemetry=telemetry,
    )


def _merge_extra_row_signals(
    base_row_signals: Mapping[str, Any],
    extra_row_signals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row_signals = dict(base_row_signals)
    if extra_row_signals is None:
        return row_signals
    overlap = sorted(set(row_signals) & set(extra_row_signals))
    if overlap:
        raise ValueError(f"extra_row_signals may not override existing row signals: {overlap}")
    row_signals.update(extra_row_signals)
    return row_signals


def _featureize_linker_candidates_with_telemetry(
    *,
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
    feature_columns: Sequence[str],
    pairwise_stats: PairwiseAggregateStats | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> tuple[LinkerFeatureMatrix, dict[str, int]]:
    """Private featureizer from compact production-shaped candidate inputs."""

    resolved_feature_columns = tuple(str(column) for column in feature_columns)
    if candidate_batch.row_count == 0:
        return (
            LinkerFeatureMatrix(
                matrix=np.empty((0, len(resolved_feature_columns)), dtype=np.float32),
                feature_columns=resolved_feature_columns,
                candidate_batch=candidate_batch,
                pairwise_stats=pairwise_stats,
            ),
            {"generated_family_id_count": 0, "generic_family_override_count": 0},
        )
    if pairwise_stats is None:
        if dataset is None:
            raise ValueError("dataset is required when pairwise_stats is not provided")
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
    row_features, row_feature_telemetry = build_promoted_non_pairwise_row_features_with_telemetry(
        candidate_batch,
        row_signals,
    )
    return (
        assemble_linker_feature_matrix(
            candidate_batch,
            row_features,
            pairwise_stats=pairwise_stats,
            feature_columns=resolved_feature_columns,
        ),
        row_feature_telemetry,
    )


def _featureize_linker_candidates(
    *,
    dataset: ANDData | None,
    candidate_batch: LinkerCandidateBatch,
    row_signals: Mapping[str, Any],
    feature_columns: Sequence[str],
    pairwise_stats: PairwiseAggregateStats | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> LinkerFeatureMatrix:
    feature_matrix, _row_feature_telemetry = _featureize_linker_candidates_with_telemetry(
        dataset=dataset,
        candidate_batch=candidate_batch,
        row_signals=row_signals,
        feature_columns=feature_columns,
        pairwise_stats=pairwise_stats,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
    )
    return feature_matrix


def _no_candidate_abstain_decisions(
    query_signature_indices: Sequence[int] | np.ndarray,
) -> tuple[LinkOrAbstainDecision, ...]:
    return tuple(
        LinkOrAbstainDecision(
            query_signature_index=int(query_index),
            action="abstain",
            row_index=None,
            component_key=None,
            score=None,
            runner_up_score=None,
            score_margin=None,
        )
        for query_index in query_signature_indices
    )


def _signature_id_to_index(signature_id_to_index: Mapping[str, int], signature_id: Any) -> int:
    key = str(signature_id)
    if key not in signature_id_to_index:
        raise KeyError(f"signature_id not present in linker runtime signature_ids: {key!r}")
    return int(signature_id_to_index[key])


def _build_component_member_indices_by_key(
    cluster_seeds_require: Mapping[Any, Any],
    signature_id_to_index: Mapping[str, int],
) -> dict[str, np.ndarray]:
    component_member_indices: dict[str, list[int]] = {}
    for signature_id, component_key in cluster_seeds_require.items():
        component_member_indices.setdefault(str(component_key), []).append(
            _signature_id_to_index(signature_id_to_index, signature_id)
        )
    return {
        component_key: np.asarray(member_indices, dtype=np.uint32)
        for component_key, member_indices in component_member_indices.items()
        if member_indices
    }


def _empty_retrieval_batch() -> LinkerRetrievalBatch:
    candidate_batch = LinkerCandidateBatch(
        row_count=0,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.zeros(0, dtype=np.uint32),
        row_component_keys=(),
        retrieval_scores=np.zeros(0, dtype=np.float32),
        retrieval_ranks=np.zeros(0, dtype=np.uint16),
    )
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals={})


def _candidate_pair_ids(
    signature_ids_by_index: Sequence[Any],
    candidate_batch: LinkerCandidateBatch,
) -> list[tuple[str, str]]:
    signature_count = len(signature_ids_by_index)
    pair_ids: list[tuple[str, str]] = []
    for left_index, right_index in zip(
        candidate_batch.left_signature_indices,
        candidate_batch.right_signature_indices,
        strict=True,
    ):
        left = int(left_index)
        right = int(right_index)
        if left >= signature_count or right >= signature_count:
            raise IndexError(
                "candidate batch pair index out of range for linker runtime signature_ids: "
                f"left={left} right={right} signature_count={signature_count}"
            )
        pair_ids.append((str(signature_ids_by_index[left]), str(signature_ids_by_index[right])))
    return pair_ids


def _resolve_candidate_batch_pair_labels_rust(
    *,
    dataset: ANDData,
    candidate_batch: LinkerCandidateBatch,
    signature_ids_by_index: Sequence[Any],
    partial_supervision: Mapping[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    dont_merge_cluster_seeds: bool,
    suppress_orcid: bool,
    n_jobs: int,
    runtime_context: Any | None,
    use_cache: bool,
    featurizer: Any | None,
) -> tuple[np.ndarray, Any]:
    pair_count = int(candidate_batch.pair_count)
    start_seconds = time.perf_counter()
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    if use_default_constraints_as_supervision:
        method = None if featurizer is None else getattr(featurizer, "linker_pair_index_arrays_constraint_labels", None)
        if featurizer is not None and not callable(method):
            raise RuntimeError(
                "RustFeaturizer.linker_pair_index_arrays_constraint_labels is required for promoted linker "
                "constraint resolution; rebuild/install the current s2and-rust extension."
            )
        labels = feature_port.get_constraint_labels_index_arrays_rust(
            dataset,
            candidate_batch.left_signature_indices,
            candidate_batch.right_signature_indices,
            dont_merge_cluster_seeds=dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=False,
            num_threads=max(1, int(n_jobs)),
            runtime_context=runtime_context,
            use_cache=use_cache,
            featurizer=featurizer,
            suppress_orcid=suppress_orcid,
        )

    partial_hits = 0
    if partial_supervision:
        signature_count = len(signature_ids_by_index)
        for pair_offset, (left_index, right_index) in enumerate(
            zip(candidate_batch.left_signature_indices, candidate_batch.right_signature_indices, strict=True)
        ):
            left = int(left_index)
            right = int(right_index)
            if left >= signature_count or right >= signature_count:
                raise IndexError(
                    "candidate batch pair index out of range for linker runtime signature_ids: "
                    f"left={left} right={right} signature_count={signature_count}"
                )
            left_id = str(signature_ids_by_index[left])
            right_id = str(signature_ids_by_index[right])
            if (left_id, right_id) in partial_supervision:
                labels[pair_offset] = float(partial_supervision[(left_id, right_id)] - LARGE_INTEGER)
                partial_hits += 1
            elif (right_id, left_id) in partial_supervision:
                labels[pair_offset] = float(partial_supervision[(right_id, left_id)] - LARGE_INTEGER)
                partial_hits += 1

    api_mode = "rust_index_arrays" if use_default_constraints_as_supervision else "partial_only"
    telemetry = SimpleNamespace(
        total_pairs=pair_count,
        partial_supervision_hits=int(partial_hits),
        unresolved_pairs=int(pair_count - partial_hits),
        rust_batch_call_count=int(use_default_constraints_as_supervision),
        api_mode=api_mode,
        elapsed_seconds=float(time.perf_counter() - start_seconds),
    )
    return labels, telemetry


def _partial_supervision_kind(value: int | float) -> str:
    value_float = float(value)
    if value_float == 0.0:
        return "require"
    if value_float == float(LARGE_DISTANCE):
        return "disallow"
    return "other"


def _validate_partial_supervision_window(
    *,
    partial_supervision: Mapping[tuple[str, str], int | float],
    query_signature_ids: set[str],
    seed_signature_to_component: Mapping[str, Any],
    candidate_pair_ids: Sequence[tuple[str, str]],
) -> dict[str, int]:
    telemetry = {
        "partial_supervision_pair_count": int(len(partial_supervision)),
        "partial_supervision_disallow_outside_retrieval_window": 0,
        "partial_supervision_disallow_between_residual_queries": 0,
        "partial_supervision_ignored_outside_window": 0,
    }
    inside_window_pairs: set[tuple[str, str]] = set()
    for left, right in candidate_pair_ids:
        inside_window_pairs.add((left, right))
        inside_window_pairs.add((right, left))

    for (left_raw, right_raw), value in partial_supervision.items():
        left = str(left_raw)
        right = str(right_raw)
        kind = _partial_supervision_kind(value)
        left_is_query = left in query_signature_ids
        right_is_query = right in query_signature_ids
        if left_is_query and right_is_query:
            if kind == "require":
                raise ValueError(
                    "partial_supervision_require_between_residual_queries: "
                    f"query_signature_id_1={left!r} query_signature_id_2={right!r}"
                )
            if kind == "disallow":
                telemetry["partial_supervision_disallow_between_residual_queries"] += 1
            else:
                telemetry["partial_supervision_ignored_outside_window"] += 1
            continue

        query_signature_id: str | None = None
        seed_signature_id: str | None = None
        if left_is_query and right in seed_signature_to_component:
            query_signature_id = left
            seed_signature_id = right
        elif right_is_query and left in seed_signature_to_component:
            query_signature_id = right
            seed_signature_id = left

        if query_signature_id is None or seed_signature_id is None:
            telemetry["partial_supervision_ignored_outside_window"] += 1
            continue
        if (query_signature_id, seed_signature_id) in inside_window_pairs:
            continue
        if kind == "require":
            seed_component = seed_signature_to_component[seed_signature_id]
            raise ValueError(
                "partial_supervision_require_outside_retrieval_window: "
                f"query_signature_id={query_signature_id!r} seed_signature_id={seed_signature_id!r} "
                f"seed_component={seed_component!r}"
            )
        if kind == "disallow":
            telemetry["partial_supervision_disallow_outside_retrieval_window"] += 1
        else:
            telemetry["partial_supervision_ignored_outside_window"] += 1
    return telemetry


def _constraint_telemetry_dict(telemetry: Any) -> dict[str, int | float | str]:
    out: dict[str, int | float | str] = {}
    for name in (
        "total_pairs",
        "partial_supervision_hits",
        "unresolved_pairs",
        "rust_batch_call_count",
        "api_mode",
        "elapsed_seconds",
    ):
        value = getattr(telemetry, name, None)
        if value is not None:
            out[f"constraint_{name}"] = value
    return out


def _predict_incremental_link_or_abstain_retrieved_candidates(
    artifact: IncrementalLinkingArtifact,
    retrieval_batch: LinkerRetrievalBatch,
    *,
    dataset: ANDData | None = None,
    extra_row_signals: Mapping[str, Any] | None = None,
    pairwise_stats: PairwiseAggregateStats | None = None,
    no_candidate_query_signature_indices: Sequence[int] | np.ndarray = (),
    partial_supervision: Mapping[Any, Any] | None = None,
    n_jobs: int = 1,
    total_ram_bytes: int | None = None,
    nan_value: float = _PAIRWISE_AGGREGATE_NAN_VALUE,
    runtime_context: Any | None = None,
    use_cache: bool = False,
    featurizer: Any | None = None,
) -> LinkOrAbstainPrivateResult:
    """Private vertical slice over retrieved candidates.

    This intentionally remains private while retrieval parity, partial
    supervision, constraints, and telemetry are still under M2/M3 validation.
    """

    if partial_supervision:
        raise NotImplementedError("partial supervision is not yet wired into the compact linker runtime")
    candidate_batch = retrieval_batch.candidate_batch
    row_signals = _merge_extra_row_signals(retrieval_batch.row_signals, extra_row_signals)
    feature_matrix, row_feature_telemetry = _featureize_linker_candidates_with_telemetry(
        dataset=dataset,
        candidate_batch=candidate_batch,
        row_signals=row_signals,
        feature_columns=artifact.metadata.feature_columns,
        pairwise_stats=pairwise_stats,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
        nan_value=nan_value,
        runtime_context=runtime_context,
        use_cache=use_cache,
        featurizer=featurizer,
    )
    compact_result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals=row_signals,
    )
    no_candidate_decisions = _no_candidate_abstain_decisions(no_candidate_query_signature_indices)
    if no_candidate_decisions:
        compact_result = LinkOrAbstainCompactResult(
            probabilities=compact_result.probabilities,
            decisions=(*compact_result.decisions, *no_candidate_decisions),
        )
    link_count = sum(1 for decision in compact_result.decisions if decision.action == "link")
    abstain_count = sum(1 for decision in compact_result.decisions if decision.action == "abstain")
    return LinkOrAbstainPrivateResult(
        feature_matrix=feature_matrix,
        compact_result=compact_result,
        telemetry={
            "candidate_row_count": int(candidate_batch.row_count),
            "pair_count": int(candidate_batch.pair_count),
            "no_candidate_query_count": int(len(no_candidate_decisions)),
            "decision_count": int(len(compact_result.decisions)),
            "link_count": int(link_count),
            "abstain_count": int(abstain_count),
            **{f"row_feature_{key}": int(value) for key, value in row_feature_telemetry.items()},
        },
    )


def _predict_incremental_link_or_abstain_production_private(
    clusterer: Any,
    artifact: IncrementalLinkingArtifact,
    *,
    dataset: ANDData,
    featurizer: Any,
    retriever: Any,
    queries: Sequence[Any],
    query_signature_ids: Sequence[Any],
    query_view: str | Sequence[str] = "initial_only",
    top_k: int | None = None,
    partial_supervision: Mapping[tuple[Any, Any], int | float] | None = None,
    constraint_backend: Any | None = None,
    extra_row_signals: Mapping[str, Any] | None = None,
    extra_row_signal_builder: Callable[[LinkerRetrievalBatch, Mapping[int, str]], Mapping[str, Any]] | None = None,
    seed_setup: tuple[
        Mapping[str, int | str],
        Mapping[int | str, int | str],
        Mapping[int | str, Sequence[str]],
    ]
    | None = None,
    runtime_context: Any | None = None,
    n_jobs: int | None = None,
    total_ram_bytes: int | None = None,
) -> LinkOrAbstainPrivateResult:
    """Run the private M3a production-shaped link-or-abstain slice.

    The caller still owns production summary/query construction and the
    constraint backend so this runtime package stays free of `scripts.*` and
    `s2and.model` imports. This helper wires the pieces that are already runtime
    surfaces: seed setup, Rust retrieval into `LinkerCandidateBatch`, existing
    constraint-label resolution, fused pairwise scoring/aggregation, gate
    application, no-candidate abstains, and altered-cluster naturalization.
    """

    if len(queries) != len(query_signature_ids):
        raise ValueError(
            "queries and query_signature_ids must have equal length: " f"{len(queries)} != {len(query_signature_ids)}"
        )
    resolved_runtime_context = runtime_context or build_runtime_context("incremental_link_or_abstain_private")
    partial_supervision_dict = {
        (str(left), str(right)): value for (left, right), value in (partial_supervision or {}).items()
    }
    n_jobs_resolved = max(1, int(getattr(clusterer, "n_jobs", 1) if n_jobs is None else n_jobs))
    retrieval_top_k = int(artifact.metadata.retrieval_top_k if top_k is None else top_k)

    if seed_setup is None:
        build_seed_setup = getattr(clusterer, "_build_incremental_seed_setup", None)
        if not callable(build_seed_setup):
            raise TypeError("clusterer must expose _build_incremental_seed_setup for the private M3a slice")
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = build_seed_setup(
            dataset,
            partial_supervision_dict,
            resolved_runtime_context,
        )
    else:
        cluster_seeds_require, recluster_map, _cluster_seeds_require_inverse = seed_setup
    cluster_seeds_require = dict(cluster_seeds_require)
    recluster_map = dict(recluster_map)

    signature_id_to_index = signature_id_to_index_map(featurizer)
    signature_ids_by_index = tuple(str(signature_id) for signature_id in featurizer.signature_ids())
    query_signature_id_strings = tuple(str(signature_id) for signature_id in query_signature_ids)
    query_signature_indices = np.asarray(
        [_signature_id_to_index(signature_id_to_index, signature_id) for signature_id in query_signature_id_strings],
        dtype=np.uint32,
    )
    query_signature_id_by_index = {
        int(query_index): query_signature_id
        for query_index, query_signature_id in zip(query_signature_indices, query_signature_id_strings, strict=True)
    }
    component_member_indices_by_key = _build_component_member_indices_by_key(
        cluster_seeds_require,
        signature_id_to_index,
    )
    if len(queries) == 0 or len(component_member_indices_by_key) == 0:
        retrieval_batch = _empty_retrieval_batch()
    else:
        retrieval_batch = build_linker_retrieval_batch_rust(
            retriever=retriever,
            queries=queries,
            query_signature_indices=query_signature_indices,
            component_member_indices_by_key=component_member_indices_by_key,
            top_k=retrieval_top_k,
            query_view=query_view,
            n_jobs=n_jobs_resolved,
        )

    candidate_batch = retrieval_batch.candidate_batch
    retrieved_query_indices = (
        set()
        if candidate_batch.row_query_signature_indices is None
        else {int(value) for value in np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)}
    )
    no_candidate_query_signature_indices = np.asarray(
        [
            int(query_index)
            for query_index in query_signature_indices
            if int(query_index) not in retrieved_query_indices
        ],
        dtype=np.uint32,
    )
    pair_ids = _candidate_pair_ids(signature_ids_by_index, candidate_batch)
    partial_telemetry = _validate_partial_supervision_window(
        partial_supervision=partial_supervision_dict,
        query_signature_ids=set(query_signature_id_strings),
        seed_signature_to_component={
            str(signature_id): component for signature_id, component in cluster_seeds_require.items()
        },
        candidate_pair_ids=pair_ids,
    )

    constraint_featurizer = getattr(constraint_backend, "rust_featurizer", None) or featurizer
    pair_labels, constraint_telemetry = _resolve_candidate_batch_pair_labels_rust(
        dataset=dataset,
        candidate_batch=candidate_batch,
        signature_ids_by_index=signature_ids_by_index,
        partial_supervision=partial_supervision_dict,
        use_default_constraints_as_supervision=bool(getattr(clusterer, "use_default_constraints_as_supervision", True)),
        dont_merge_cluster_seeds=bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
        suppress_orcid=bool(getattr(clusterer, "suppress_orcid", False)),
        n_jobs=n_jobs_resolved,
        runtime_context=resolved_runtime_context,
        use_cache=bool(getattr(clusterer, "use_cache", False)),
        featurizer=constraint_featurizer,
    )
    if pair_labels.shape != (candidate_batch.pair_count,):
        raise ValueError(
            "constraint label count must match pair_count: " f"{pair_labels.shape} != ({candidate_batch.pair_count},)"
        )

    pairwise_model_result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
        candidate_batch,
        classifier=clusterer.classifier,
        featurizer_info=clusterer.featurizer_info,
        nameless_classifier=getattr(clusterer, "nameless_classifier", None),
        nameless_featurizer_info=getattr(clusterer, "nameless_featurizer_info", None),
        pair_labels=pair_labels,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        runtime_context=resolved_runtime_context,
        use_cache=bool(getattr(clusterer, "use_cache", False)),
        featurizer=featurizer,
    )
    built_extra_row_signals = (
        {}
        if extra_row_signal_builder is None
        else dict(extra_row_signal_builder(retrieval_batch, query_signature_id_by_index))
    )
    merged_extra_row_signals = _merge_extra_row_signals(built_extra_row_signals, extra_row_signals)
    private_result = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,
        retrieval_batch,
        dataset=dataset,
        extra_row_signals=_merge_extra_row_signals(
            pairwise_model_result.row_signals,
            merged_extra_row_signals,
        ),
        pairwise_stats=pairwise_model_result.pairwise_stats,
        no_candidate_query_signature_indices=no_candidate_query_signature_indices,
        n_jobs=n_jobs_resolved,
        total_ram_bytes=total_ram_bytes,
        nan_value=_PAIRWISE_AGGREGATE_NAN_VALUE,
        runtime_context=resolved_runtime_context,
        use_cache=bool(getattr(clusterer, "use_cache", False)),
        featurizer=featurizer,
    )

    raw_linked_clusters = {
        query_signature_id_by_index[decision.query_signature_index]: decision.component_key
        for decision in private_result.compact_result.decisions
        if decision.action == "link"
        and decision.component_key is not None
        and decision.query_signature_index in query_signature_id_by_index
    }
    linked_signature_clusters = naturalize_incremental_clusters(raw_linked_clusters, recluster_map)
    telemetry: dict[str, int | float | str] = {
        **private_result.telemetry,
        **{f"pairwise_{key}": value for key, value in pairwise_model_result.telemetry.items()},
        **_constraint_telemetry_dict(constraint_telemetry),
        **partial_telemetry,
        "query_count": int(len(query_signature_id_strings)),
        "seed_signature_count": int(len(cluster_seeds_require)),
        "seed_component_count": int(len(component_member_indices_by_key)),
        "retrieval_top_k": int(retrieval_top_k),
    }
    return LinkOrAbstainPrivateResult(
        feature_matrix=private_result.feature_matrix,
        compact_result=private_result.compact_result,
        telemetry=telemetry,
        retrieval_batch=retrieval_batch,
        pairwise_model_result=pairwise_model_result,
        linked_signature_clusters=linked_signature_clusters,
    )


def naturalize_incremental_clusters(
    predicted_clusters: Mapping[str, Any],
    split_cluster_to_natural_cluster: Mapping[Any, Any],
) -> dict[str, Any]:
    """Naturalize split incremental cluster IDs back to caller-visible IDs."""

    return {
        str(signature_id): split_cluster_to_natural_cluster.get(cluster_id, cluster_id)
        for signature_id, cluster_id in predicted_clusters.items()
    }
