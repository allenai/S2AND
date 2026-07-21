"""NumPy-only logistic link-or-abstain gate for incremental linking."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.gate_buckets import first_name_bucket_array
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch

LOGISTIC_GATE_MODEL_TYPE = "multiclass_logistic_numpy_v1"
LOGISTIC_GATE_SELECTION_FEATURE_COUNT = 240
LOGISTIC_GATE_SELECTION_C = 3.0
LOGISTIC_GATE_FINAL_C = 0.03
LOGISTIC_GATE_ERROR_WEIGHTS = {
    "false_abstain": 0.25,
    "false_link": 1.0,
    "wrong_candidate_link": 1.5,
}
LOGISTIC_GATE_REQUIRED_ERROR_WEIGHTS = frozenset(LOGISTIC_GATE_ERROR_WEIGHTS)
LOGISTIC_GATE_CLASSES = (0, 1, 2)

_BASE_QUERY_FEATURES = {
    "chosen_probability",
    "second_probability",
    "score_margin",
    "has_runner_up",
    "raw_candidate_count",
}
_PROBABILITY_FEATURES = (
    "prob_top",
    "prob_second",
    "prob_third",
    "prob_top2_gap",
    "prob_second_third_gap",
    "prob_top_share",
    "prob_top2_share",
    "prob_sum",
    "prob_mean",
    "prob_std",
    "prob_p90",
    "prob_p75",
    "prob_entropy",
    "prob_effective_candidates",
    "prob_count_ge_0_25",
    "prob_count_ge_0_50",
    "prob_count_ge_0_75",
    "prob_count_within_0_01",
    "prob_count_within_0_05",
    "prob_count_within_0_10",
)
_BASE_EVIDENCE_FEATURES = {
    "retrieval_score",
    "min_distance",
    "top5_mean_distance",
    "retrieval_reciprocal_rank",
    "retrieval_rank",
    "cluster_size_log",
    "candidate_cluster_max_paper_author_count",
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
    "specter_exemplar_similarity",
    "pw_mean_specter_cosine_sim",
    "pw_min_specter_cosine_sim",
    "pw_max_jaro",
    "pw_min_levenshtein",
    "coauthor_overlap",
    "pw_max_coauthor_overlap",
    "pw_mean_coauthor_overlap",
    "pw_mean_coauthor_match",
    "paper_author_list_max_jaccard",
    "paper_author_list_max_containment",
    "paper_author_list_max_overlap_count",
    "local_author_window10_jaccard_max",
    "local_author_window10_overlap_count_max",
    "pw_max_title_overlap_words",
    "pw_mean_title_overlap_words",
    "pw_max_journal_overlap",
    "pw_mean_journal_overlap",
    "pw_max_venue_overlap",
    "query_first_prefix_match_any_length",
    "same_dominant_first_as_best_top5",
    "same_family_as_heuristic_choice",
    "candidate_dominant_first_name_length",
    "last_first_name_count_min_rarity",
    "last_name_count_min_rarity",
    "pw_min_first_name_count_min",
    "pw_min_first_name_count_max",
    "pw_min_last_first_name_count_max",
    "year_compatibility",
    "year_gap_to_candidate_range",
    "year_gap_signed_to_candidate_range",
    "affiliation_overlap",
    "pw_max_affiliation_overlap",
    "affiliation_contradiction_severity",
}
_CATEGORICAL_FEATURES = (
    "gate_bucket_multi_candidate|multi_letter_first",
    "gate_bucket_multi_candidate|single_letter_first",
    "gate_bucket_single_candidate|multi_letter_first",
    "gate_bucket_single_candidate|single_letter_first",
    "first_name_bucket_multi_letter_first",
    "first_name_bucket_single_letter_first",
    "candidate_kind_multi_candidate",
    "candidate_kind_single_candidate",
    "query_view_full",
    "query_view_initial_only",
)


def _immutable_float64_array(value: Any) -> np.ndarray:
    """Copy an array into a read-only bytes-backed NumPy view."""

    array = np.asarray(value, dtype=np.float64, order="C")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    frozen.shape = array.shape
    return frozen


def _validate_unique_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Return feature names after rejecting duplicate columns."""

    resolved = tuple(str(feature_name) for feature_name in feature_names)
    seen: set[str] = set()
    duplicates: list[str] = []
    for feature_name in resolved:
        if feature_name in seen and feature_name not in duplicates:
            duplicates.append(feature_name)
        seen.add(feature_name)
    if duplicates:
        raise ValueError(f"logistic gate feature_names must be unique; duplicates={duplicates[:5]}")
    return resolved


def default_logistic_gate_feature_names() -> tuple[str, ...]:
    """Return the full prod-safe logistic gate feature universe."""

    evidence = tuple(sorted(_BASE_EVIDENCE_FEATURES))
    return (
        "chosen_probability",
        "second_probability",
        "score_margin",
        "has_runner_up",
        "raw_candidate_count",
        *_PROBABILITY_FEATURES,
        *(f"top_raw_{feature}" for feature in evidence),
        *(f"delta_top_second_{feature}" for feature in evidence),
        "top_meta_retrieval_rank",
        "top_meta_query_first_token_len",
        "top_meta_query_author_len",
        *(f"list_mean_{feature}" for feature in evidence),
        *(f"list_std_{feature}" for feature in evidence),
        *(f"list_min_{feature}" for feature in evidence),
        *(f"list_max_{feature}" for feature in evidence),
        *_CATEGORICAL_FEATURES,
    )


@dataclass(frozen=True)
class LogisticGateQueryRows:
    """Per-query candidate ranking state used to build gate features."""

    groups: tuple[np.ndarray, ...]
    ranked_groups: tuple[np.ndarray, ...]
    best_rows: np.ndarray
    runner_up_scores: np.ndarray
    score_margins: np.ndarray
    has_runner_up: np.ndarray


@dataclass(frozen=True)
class NumpyLogisticGate:
    """Loaded logistic gate that predicts with only NumPy arrays."""

    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: np.ndarray
    missing_values: np.ndarray
    classes: tuple[int, ...]
    error_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        feature_names = _validate_unique_feature_names(self.feature_names)
        classes = tuple(self.classes)
        weights = _immutable_float64_array(self.weights)
        bias = _immutable_float64_array(self.bias)
        missing_values = _immutable_float64_array(self.missing_values)
        error_weights = MappingProxyType({key: float(value) for key, value in self.error_weights.items()})
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "missing_values", missing_values)
        object.__setattr__(self, "error_weights", error_weights)

        n_features = len(feature_names)
        if weights.shape != (n_features, len(classes)):
            raise ValueError(f"logistic gate weights shape must be ({n_features}, {len(classes)}), got {weights.shape}")
        if bias.shape != (len(classes),):
            raise ValueError(f"logistic gate bias shape must be ({len(classes)},), got {bias.shape}")
        if missing_values.shape != (n_features,):
            raise ValueError(f"logistic gate missing_values shape must be ({n_features},), got {missing_values.shape}")
        if classes != LOGISTIC_GATE_CLASSES:
            raise ValueError(f"logistic gate classes must be {LOGISTIC_GATE_CLASSES}, got {classes}")
        missing_error_weights = sorted(LOGISTIC_GATE_REQUIRED_ERROR_WEIGHTS - set(error_weights))
        if missing_error_weights:
            raise ValueError(f"logistic gate error_weights missing required keys: {missing_error_weights}")

    def fill_missing(self, matrix: np.ndarray) -> np.ndarray:
        """Return a float64 feature matrix with training medians applied."""

        features = np.asarray(matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"logistic gate matrix must have shape (n, {len(self.feature_names)}), got {features.shape}"
            )
        invalid = ~np.isfinite(features)
        if np.any(invalid):
            features = features.copy()
            row_indices, col_indices = np.nonzero(invalid)
            features[row_indices, col_indices] = self.missing_values[col_indices]
        return features

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Predict calibrated outcome probabilities with the stored NumPy weights."""

        features = self.fill_missing(matrix)
        logits = features @ self.weights + self.bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_link(self, matrix: np.ndarray) -> np.ndarray:
        """Return True for link decisions and False for abstentions."""

        proba = self.predict_proba(matrix)
        false_abstain = float(self.error_weights["false_abstain"])
        false_link = float(self.error_weights["false_link"])
        wrong = float(self.error_weights["wrong_candidate_link"])
        link_cost = false_link * proba[:, 0] + wrong * proba[:, 1]
        abstain_cost = false_abstain * (proba[:, 1] + proba[:, 2])
        return link_cost < abstain_cost


def _ordered_group_indices(query_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    groups_by_query: dict[Any, list[int]] = {}
    for row_index, query_index in enumerate(query_indices.tolist()):
        groups_by_query.setdefault(query_index, []).append(row_index)
    return tuple(np.asarray(indices, dtype=np.int64) for indices in groups_by_query.values())


def _sort_group(
    group: np.ndarray,
    *,
    probabilities: np.ndarray,
    retrieval_ranks: np.ndarray | None,
    component_keys: Sequence[object] | None,
) -> np.ndarray:
    def sort_key(row_index: int) -> tuple[float, int, str]:
        rank = 0 if retrieval_ranks is None else int(retrieval_ranks[row_index])
        component_key = "" if component_keys is None else str(component_keys[row_index])
        return (-float(probabilities[row_index]), rank, component_key)

    return np.asarray(sorted((int(row_index) for row_index in group), key=sort_key), dtype=np.int64)


def ranked_query_rows(
    query_indices: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    retrieval_ranks: Sequence[Any] | np.ndarray | None = None,
    component_keys: Sequence[object] | None = None,
) -> LogisticGateQueryRows:
    """Return stable per-query candidate rankings from scored candidate rows."""

    query_array = np.asarray(query_indices)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if query_array.ndim != 1 or probability_array.ndim != 1 or len(query_array) != len(probability_array):
        raise ValueError(
            "query_indices and probabilities must be 1D arrays with the same length: "
            f"{query_array.shape} != {probability_array.shape}"
        )
    rank_array = None if retrieval_ranks is None else np.asarray(retrieval_ranks)
    if rank_array is not None and (rank_array.ndim != 1 or len(rank_array) != len(probability_array)):
        raise ValueError(f"retrieval_ranks must be 1D with row_count={len(probability_array)}, got {rank_array.shape}")
    if component_keys is not None and len(component_keys) != len(probability_array):
        raise ValueError(f"component_keys must have row_count={len(probability_array)}, got {len(component_keys)}")

    groups = _ordered_group_indices(query_array)
    ranked_groups = tuple(
        _sort_group(
            group,
            probabilities=probability_array,
            retrieval_ranks=rank_array,
            component_keys=component_keys,
        )
        for group in groups
    )
    best_rows = np.asarray([group[0] for group in ranked_groups], dtype=np.int64)
    runner_up_scores = np.full(len(ranked_groups), np.nan, dtype=np.float64)
    for query_pos, group in enumerate(ranked_groups):
        if len(group) > 1:
            runner_up_scores[query_pos] = float(probability_array[group[1]])
    has_runner_up = np.asarray([len(group) > 1 for group in ranked_groups], dtype=bool)
    score_margins = probability_array[best_rows] - runner_up_scores
    score_margins[~has_runner_up] = np.nan
    return LogisticGateQueryRows(
        groups=groups,
        ranked_groups=ranked_groups,
        best_rows=best_rows,
        runner_up_scores=runner_up_scores,
        score_margins=score_margins,
        has_runner_up=has_runner_up,
    )


def outcome_labels(query_safe_target: Sequence[Any], chosen_candidate_target: Sequence[Any]) -> np.ndarray:
    """Return logistic gate class labels: 0=no safe, 1=top wrong, 2=top correct."""

    safe = np.asarray(query_safe_target, dtype=np.int8)
    chosen = np.asarray(chosen_candidate_target, dtype=np.int8)
    return np.select([safe == 0, chosen == 0], [0, 1], default=2).astype(np.int8)


def logistic_gate_config(
    *,
    feature_names: Sequence[str],
    weights: Sequence[Sequence[float]] | np.ndarray,
    bias: Sequence[float] | np.ndarray,
    missing_values: Sequence[float] | np.ndarray,
    calibration_mode: str,
    error_weights: Mapping[str, float] | None = None,
    training_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-compatible logistic gate config."""

    resolved_weights = np.asarray(weights, dtype=np.float64)
    resolved_bias = np.asarray(bias, dtype=np.float64)
    resolved_missing = np.asarray(missing_values, dtype=np.float64)
    gate = NumpyLogisticGate(
        feature_names=tuple(str(feature) for feature in feature_names),
        weights=resolved_weights,
        bias=resolved_bias,
        missing_values=resolved_missing,
        classes=LOGISTIC_GATE_CLASSES,
        error_weights={key: float(value) for key, value in dict(error_weights or LOGISTIC_GATE_ERROR_WEIGHTS).items()},
    )
    return {
        "model_type": LOGISTIC_GATE_MODEL_TYPE,
        "calibration_mode": str(calibration_mode),
        "feature_names": list(gate.feature_names),
        "weights": gate.weights.tolist(),
        "bias": gate.bias.tolist(),
        "classes": list(gate.classes),
        "missing_values": gate.missing_values.tolist(),
        "error_weights": dict(gate.error_weights),
        "training_summary": dict(training_summary or {}),
    }


def load_logistic_gate_config(gate_config: Mapping[str, Any]) -> NumpyLogisticGate:
    """Load and validate a logistic gate config from artifact metadata."""

    if str(gate_config.get("model_type", "")) != LOGISTIC_GATE_MODEL_TYPE:
        raise ValueError(f"Unsupported logistic gate model_type: {gate_config.get('model_type')!r}")
    feature_names = tuple(str(feature) for feature in gate_config["feature_names"])
    classes = tuple(int(value) for value in gate_config.get("classes", LOGISTIC_GATE_CLASSES))
    weights = np.asarray(gate_config["weights"], dtype=np.float64)
    bias = np.asarray(gate_config["bias"], dtype=np.float64)
    missing_values = np.asarray(gate_config["missing_values"], dtype=np.float64)
    error_weights = {key: float(value) for key, value in dict(gate_config["error_weights"]).items()}
    return NumpyLogisticGate(
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        missing_values=missing_values,
        classes=classes,
        error_weights=error_weights,
    )


def _feature_array(
    feature_values: Mapping[str, Any],
    name: str,
    row_count: int,
    *,
    default: float = np.nan,
) -> np.ndarray:
    if name not in feature_values:
        return np.full(row_count, float(default), dtype=np.float64)
    values = np.asarray(feature_values[name], dtype=np.float64)
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError(f"gate feature source {name!r} must be 1D with row_count={row_count}, got {values.shape}")
    return values


def _object_array(feature_values: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    if name not in feature_values:
        return np.asarray([""] * row_count, dtype=object)
    values = np.asarray(feature_values[name], dtype=object)
    if values.ndim != 1 or len(values) != row_count:
        raise ValueError(f"gate feature source {name!r} must be 1D with row_count={row_count}, got {values.shape}")
    return values


def _optional_text_length(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float | np.floating) and np.isnan(value):
        return 0.0
    return float(len(str(value)))


def _normalized_entropy(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values / total
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))))
    return entropy / math.log(len(probabilities))


def build_logistic_gate_matrix(
    feature_names: Sequence[str],
    *,
    query_indices: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    feature_values: Mapping[str, Any],
    retrieval_ranks: Sequence[Any] | np.ndarray | None = None,
    component_keys: Sequence[object] | None = None,
) -> tuple[np.ndarray, LogisticGateQueryRows]:
    """Build per-query logistic gate features from scored candidate rows."""

    resolved_feature_names = _validate_unique_feature_names(feature_names)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    row_count = len(probability_array)
    if retrieval_ranks is None and "retrieval_rank" in feature_values:
        retrieval_ranks = np.asarray(feature_values["retrieval_rank"])
    query_rows = ranked_query_rows(
        query_indices,
        probability_array,
        retrieval_ranks=retrieval_ranks,
        component_keys=component_keys,
    )
    query_view = _object_array(feature_values, "query_view", row_count)
    if "first_name_bucket" in feature_values:
        first_name_bucket = _object_array(feature_values, "first_name_bucket", row_count)
    elif "query_first_token" in feature_values and "query_view" in feature_values:
        first_name_bucket = first_name_bucket_array(feature_values["query_first_token"], feature_values["query_view"])
    else:
        if any(name.startswith("first_name_bucket_") for name in resolved_feature_names):
            raise ValueError(
                "logistic gate first_name_bucket_* features require 'first_name_bucket' or both"
                " 'query_first_token' and 'query_view' in feature_values"
            )
        first_name_bucket = _object_array(feature_values, "first_name_bucket", row_count)
    has_query_first_token = "query_first_token" in feature_values
    has_query_author = "query_author" in feature_values
    query_first_token = _object_array(feature_values, "query_first_token", row_count)
    query_author = _object_array(feature_values, "query_author", row_count)

    matrix = np.empty((len(query_rows.groups), len(resolved_feature_names)), dtype=np.float64)
    feature_positions = {feature_name: index for index, feature_name in enumerate(resolved_feature_names)}
    handled: set[str] = set()

    def set_column(name: str, values: np.ndarray) -> None:
        position = feature_positions.get(name)
        if position is None:
            return
        matrix[:, position] = values
        handled.add(name)

    best_rows = query_rows.best_rows
    group_sizes = np.asarray([len(group) for group in query_rows.ranked_groups], dtype=np.float64)
    set_column("chosen_probability", probability_array[best_rows])
    set_column("second_probability", query_rows.runner_up_scores)
    set_column("score_margin", query_rows.score_margins)
    set_column("has_runner_up", query_rows.has_runner_up.astype(np.float64))
    set_column("raw_candidate_count", group_sizes)

    probability_feature_names = [name for name in resolved_feature_names if name.startswith("prob_")]
    if probability_feature_names:
        for query_pos, ranked in enumerate(query_rows.ranked_groups):
            probs = probability_array[ranked]
            n_probs = len(probs)
            top = float(probs[0]) if n_probs else 0.0
            second = float(probs[1]) if n_probs > 1 else 0.0
            third = float(probs[2]) if n_probs > 2 else 0.0
            prob_sum = float(np.sum(probs))
            for feature_name in probability_feature_names:
                if feature_name == "prob_top":
                    value = top
                elif feature_name == "prob_second":
                    value = second
                elif feature_name == "prob_third":
                    value = third
                elif feature_name == "prob_top2_gap":
                    value = top - second
                elif feature_name == "prob_second_third_gap":
                    value = second - third
                elif feature_name == "prob_top_share":
                    value = top / prob_sum if prob_sum > 0 else 0.0
                elif feature_name == "prob_top2_share":
                    value = float(np.sum(probs[:2])) / prob_sum if prob_sum > 0 else 0.0
                elif feature_name == "prob_sum":
                    value = prob_sum
                elif feature_name == "prob_mean":
                    value = float(np.mean(probs)) if n_probs else 0.0
                elif feature_name == "prob_std":
                    value = float(np.std(probs)) if n_probs else 0.0
                elif feature_name == "prob_p90":
                    value = float(np.quantile(probs, 0.9)) if n_probs else 0.0
                elif feature_name == "prob_p75":
                    value = float(np.quantile(probs, 0.75)) if n_probs else 0.0
                elif feature_name == "prob_entropy":
                    value = _normalized_entropy(probs)
                elif feature_name == "prob_effective_candidates":
                    if prob_sum <= 0.0 or not n_probs:
                        value = 0.0
                    else:
                        shares = probs / prob_sum
                        value = math.exp(-float(np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)))))
                elif feature_name == "prob_count_ge_0_25":
                    value = float((probs >= 0.25).sum())
                elif feature_name == "prob_count_ge_0_50":
                    value = float((probs >= 0.50).sum())
                elif feature_name == "prob_count_ge_0_75":
                    value = float((probs >= 0.75).sum())
                elif feature_name == "prob_count_within_0_01":
                    value = float((probs >= top - 0.01).sum()) if n_probs else 0.0
                elif feature_name == "prob_count_within_0_05":
                    value = float((probs >= top - 0.05).sum()) if n_probs else 0.0
                elif feature_name == "prob_count_within_0_10":
                    value = float((probs >= top - 0.10).sum()) if n_probs else 0.0
                else:
                    raise KeyError(f"Unsupported logistic gate probability feature: {feature_name}")
                matrix[query_pos, feature_positions[feature_name]] = value
        handled.update(probability_feature_names)

    candidate_kind = np.where(query_rows.has_runner_up, "multi_candidate", "single_candidate")
    best_first_name_bucket = first_name_bucket[best_rows].astype(str)
    best_query_view = query_view[best_rows].astype(str)
    gate_bucket = np.asarray(
        [f"{kind}|{bucket}" for kind, bucket in zip(candidate_kind, best_first_name_bucket, strict=True)],
        dtype=object,
    )
    for feature_name in resolved_feature_names:
        if feature_name.startswith("candidate_kind_"):
            set_column(feature_name, (candidate_kind == feature_name.removeprefix("candidate_kind_")).astype(float))
        elif feature_name.startswith("first_name_bucket_"):
            set_column(
                feature_name,
                (best_first_name_bucket == feature_name.removeprefix("first_name_bucket_")).astype(float),
            )
        elif feature_name.startswith("query_view_"):
            set_column(feature_name, (best_query_view == feature_name.removeprefix("query_view_")).astype(float))
        elif feature_name.startswith("gate_bucket_"):
            set_column(feature_name, (gate_bucket == feature_name.removeprefix("gate_bucket_")).astype(float))

    source_ops: dict[str, set[str]] = {}
    prefix_to_op = {
        "top_raw_": "top",
        "delta_top_second_": "delta",
        "list_mean_": "mean",
        "list_std_": "std",
        "list_min_": "min",
        "list_max_": "max",
    }
    for feature_name in resolved_feature_names:
        for prefix, op in prefix_to_op.items():
            if feature_name.startswith(prefix):
                source_ops.setdefault(feature_name.removeprefix(prefix), set()).add(op)
                break

    cache: dict[str, np.ndarray] = {}
    for source_name, ops in source_ops.items():
        values = cache.setdefault(source_name, _feature_array(feature_values, source_name, row_count))
        if "top" in ops:
            set_column(f"top_raw_{source_name}", values[best_rows])
        if "delta" in ops:
            deltas = np.full(len(query_rows.ranked_groups), np.nan, dtype=np.float64)
            for query_pos, ranked in enumerate(query_rows.ranked_groups):
                if len(ranked) > 1:
                    deltas[query_pos] = float(values[ranked[0]] - values[ranked[1]])
            set_column(f"delta_top_second_{source_name}", deltas)
        list_ops = ops & {"mean", "std", "min", "max"}
        if list_ops:
            aggregates = {op: np.full(len(query_rows.groups), np.nan, dtype=np.float64) for op in list_ops}
            for query_pos, group in enumerate(query_rows.groups):
                finite = values[group]
                finite = finite[np.isfinite(finite)]
                if len(finite) == 0:
                    continue
                if "mean" in list_ops:
                    aggregates["mean"][query_pos] = float(np.mean(finite))
                if "std" in list_ops:
                    aggregates["std"][query_pos] = float(np.std(finite, ddof=0))
                if "min" in list_ops:
                    aggregates["min"][query_pos] = float(np.min(finite))
                if "max" in list_ops:
                    aggregates["max"][query_pos] = float(np.max(finite))
            for op, values_for_op in aggregates.items():
                set_column(f"list_{op}_{source_name}", values_for_op)

    if "top_meta_retrieval_rank" in feature_positions:
        values = cache.setdefault("retrieval_rank", _feature_array(feature_values, "retrieval_rank", row_count))
        set_column("top_meta_retrieval_rank", values[best_rows])
    if "top_meta_query_first_token_len" in feature_positions:
        token_lengths = (
            np.fromiter((_optional_text_length(query_first_token[row]) for row in best_rows), dtype=np.float64)
            if has_query_first_token
            else np.full(len(best_rows), np.nan, dtype=np.float64)
        )
        set_column("top_meta_query_first_token_len", token_lengths)
    if "top_meta_query_author_len" in feature_positions:
        author_lengths = (
            np.fromiter((_optional_text_length(query_author[row]) for row in best_rows), dtype=np.float64)
            if has_query_author
            else np.full(len(best_rows), np.nan, dtype=np.float64)
        )
        set_column("top_meta_query_author_len", author_lengths)

    unhandled = [feature_name for feature_name in resolved_feature_names if feature_name not in handled]
    if unhandled:
        raise KeyError(f"Unsupported logistic gate feature: {unhandled[0]}")
    return matrix, query_rows


def feature_values_from_candidate_frame(rows: Any) -> dict[str, Any]:
    """Return gate feature source arrays from a pandas-like candidate row table."""

    return {str(column): rows[column].to_numpy() for column in rows.columns}


def feature_values_from_runtime(
    feature_matrix: LinkerFeatureMatrix,
    row_signals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return gate feature source arrays from runtime feature and signal objects."""

    values: dict[str, Any] = {
        column: np.asarray(feature_matrix.matrix[:, index], dtype=np.float64)
        for index, column in enumerate(feature_matrix.feature_columns)
    }
    candidate_batch = feature_matrix.candidate_batch
    if candidate_batch.retrieval_scores is not None:
        values["retrieval_score"] = np.asarray(candidate_batch.retrieval_scores, dtype=np.float64)
    if candidate_batch.retrieval_ranks is not None:
        values["retrieval_rank"] = np.asarray(candidate_batch.retrieval_ranks, dtype=np.float64)
    pairwise_column_set = (
        {str(column) for column in feature_matrix.pairwise_stats.aggregate_feature_columns}
        if feature_matrix.pairwise_stats is not None
        else set()
    )
    if row_signals is not None:
        for key, value in row_signals.items():
            key = str(key)
            if key in pairwise_column_set and key in values:
                continue
            values[key] = value
    if feature_matrix.pairwise_stats is not None:
        missing_pairwise_columns = [
            str(column)
            for column in feature_matrix.pairwise_stats.aggregate_feature_columns
            if str(column) not in values
        ]
        if missing_pairwise_columns:
            missing_pairwise_column_set = set(missing_pairwise_columns)
            pairwise_values = np.asarray(feature_matrix.pairwise_stats.feature_matrix(), dtype=np.float64)
            for index, column in enumerate(feature_matrix.pairwise_stats.aggregate_feature_columns):
                column = str(column)
                if column in missing_pairwise_column_set:
                    values[column] = pairwise_values[:, index]
    return values


def build_runtime_logistic_gate_matrix(
    gate: NumpyLogisticGate,
    feature_matrix: LinkerFeatureMatrix,
    probabilities: np.ndarray,
    *,
    row_signals: Mapping[str, Any] | None,
) -> tuple[np.ndarray, LogisticGateQueryRows]:
    """Build artifact-selected gate features for a scored runtime feature matrix."""

    candidate_batch: LinkerCandidateBatch = feature_matrix.candidate_batch
    if candidate_batch.row_query_signature_indices is None:
        raise ValueError("candidate_batch.row_query_signature_indices is required for logistic gate decisions")
    return build_logistic_gate_matrix(
        gate.feature_names,
        query_indices=candidate_batch.row_query_signature_indices,
        probabilities=probabilities,
        feature_values=feature_values_from_runtime(feature_matrix, row_signals),
        retrieval_ranks=candidate_batch.retrieval_ranks,
        component_keys=candidate_batch.row_component_keys,
    )
