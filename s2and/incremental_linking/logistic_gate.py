"""NumPy-only logistic link-or-abstain gate for incremental linking."""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator, Mapping, Sequence
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
_EVIDENCE_AGGREGATE_PREFIX_TO_OP = {
    "top_raw_": "top",
    "delta_top_second_": "delta",
    "list_mean_": "mean",
    "list_std_": "std",
    "list_min_": "min",
    "list_max_": "max",
}
_DIRECT_FEATURE_SOURCES = {
    "top_meta_retrieval_rank": "retrieval_rank",
    "top_meta_query_first_token_len": "query_first_token",
    "top_meta_query_author_len": "query_author",
}
_LIST_OPS = frozenset({"mean", "std", "min", "max"})
_CATEGORICAL_PREFIXES = ("candidate_kind_", "first_name_bucket_", "query_view_", "gate_bucket_")
_BUCKET_PREFIXES = ("first_name_bucket_", "gate_bucket_")


@dataclass(frozen=True)
class _GateFeaturePlan:
    """Parsed gate feature names. Incremental linking re-runs the gate once per query, so parse once."""

    feature_names: tuple[str, ...]
    positions: Mapping[str, int]
    probability_names: tuple[str, ...]
    unsupported_probability_name: str | None
    categorical: tuple[tuple[str, str, str], ...]
    bucket_features: tuple[str, ...]
    source_ops: tuple[tuple[str, frozenset[str]], ...]


@functools.lru_cache(maxsize=32)
def _gate_feature_plan(feature_names: tuple[str, ...]) -> _GateFeaturePlan:
    """Parse unique gate feature names into the columns each feature family fills."""

    resolved = _validate_unique_feature_names(feature_names)
    probability_names = tuple(name for name in resolved if name.startswith("prob_"))
    unsupported = [name for name in probability_names if name not in _PROBABILITY_FEATURES]
    categorical = []
    for name in resolved:
        for prefix in _CATEGORICAL_PREFIXES:
            if name.startswith(prefix):
                categorical.append((name, prefix, name.removeprefix(prefix)))
                break
    source_ops: dict[str, set[str]] = {}
    for name in resolved:
        for prefix, op in _EVIDENCE_AGGREGATE_PREFIX_TO_OP.items():
            if name.startswith(prefix):
                source_ops.setdefault(name.removeprefix(prefix), set()).add(op)
                break
    return _GateFeaturePlan(
        feature_names=resolved,
        positions=MappingProxyType({name: index for index, name in enumerate(resolved)}),
        probability_names=probability_names,
        unsupported_probability_name=unsupported[0] if unsupported else None,
        categorical=tuple(categorical),
        bucket_features=tuple(name for name in resolved if name.startswith(_BUCKET_PREFIXES)),
        source_ops=tuple((source, frozenset(ops)) for source, ops in source_ops.items()),
    )


@functools.lru_cache(maxsize=32)
def _feature_source_requirements(
    feature_names: tuple[str, ...],
) -> tuple[tuple[tuple[str, str | None, bool], ...], tuple[str, ...]]:
    """Return each feature's runtime source and whether it needs query_view, then the bucket features."""

    requirements = []
    for feature_name in feature_names:
        source_name = _DIRECT_FEATURE_SOURCES.get(feature_name)
        if source_name is None:
            for prefix in _EVIDENCE_AGGREGATE_PREFIX_TO_OP:
                if feature_name.startswith(prefix):
                    source_name = feature_name.removeprefix(prefix)
                    break
        needs_query_view = feature_name.startswith("query_view_")
        if source_name is not None or needs_query_view:
            requirements.append((feature_name, source_name, needs_query_view))
    bucket_features = tuple(feature_name for feature_name in feature_names if feature_name.startswith(_BUCKET_PREFIXES))
    return tuple(requirements), bucket_features


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


def _validate_runtime_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Return artifact feature names after enforcing the runtime-safe universe."""

    resolved = _validate_unique_feature_names(feature_names)
    unsupported = sorted(set(resolved).difference(default_logistic_gate_feature_names()))
    if unsupported:
        raise ValueError(f"logistic gate feature_names contain unsupported runtime features: {unsupported[:5]}")
    return resolved


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


def _array_ranked_groups(
    query_array: np.ndarray,
    probability_array: np.ndarray,
    rank_array: np.ndarray | None,
    component_keys: Sequence[object] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None:
    """The groups and ranked groups the loops below build, from array sorts.

    Groups follow the first appearance of each query, and a group's rows sort
    by probability (descending), retrieval rank and component key; every key
    ties only for equal values, as the tuple comparison does. Returns None for
    inputs the loops must handle: non-integer query ids, or probabilities or
    ranks that are not finite, whose comparisons a sort key cannot reproduce.
    A gate ranks one row per candidate, so the loops took a second on large
    requests.
    """

    row_count = len(query_array)
    if row_count == 0 or query_array.dtype.kind not in "iu" or not np.isfinite(probability_array).all():
        return None
    if rank_array is None:
        ranks = np.zeros(row_count, dtype=np.int64)
    elif rank_array.dtype.kind in "iu" or (rank_array.dtype.kind == "f" and np.isfinite(rank_array).all()):
        # int() truncates toward zero, as the cast does.
        ranks = rank_array.astype(np.int64)
    else:
        return None
    if component_keys is None:
        key_ranks = np.zeros(row_count, dtype=np.int64)
    else:
        keys = [str(key) for key in component_keys]
        key_rank = {key: position for position, key in enumerate(sorted(set(keys)))}
        key_ranks = np.fromiter((key_rank[key] for key in keys), dtype=np.int64, count=row_count)
    _, first_rows, inverse = np.unique(query_array, return_index=True, return_inverse=True)
    group_of_query = np.empty(len(first_rows), dtype=np.int64)
    group_of_query[np.argsort(first_rows, kind="stable")] = np.arange(len(first_rows))
    group_of_row = group_of_query[inverse.reshape(-1)]
    boundaries = np.cumsum(np.bincount(group_of_row, minlength=len(first_rows)))[:-1]
    grouped_rows = np.argsort(group_of_row, kind="stable").astype(np.int64)
    ranked_rows = np.lexsort((key_ranks, ranks, -probability_array, group_of_row)).astype(np.int64)
    return tuple(np.split(grouped_rows, boundaries)), tuple(np.split(ranked_rows, boundaries))


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

    array_groups = _array_ranked_groups(query_array, probability_array, rank_array, component_keys)
    if array_groups is not None:
        groups, ranked_groups = array_groups
    else:
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
    resolved_feature_names = _validate_runtime_feature_names(feature_names)
    gate = NumpyLogisticGate(
        feature_names=resolved_feature_names,
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
    feature_names = _validate_runtime_feature_names(gate_config["feature_names"])
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


def _concatenated_rows(groups: Sequence[np.ndarray]) -> np.ndarray:
    """Return the rows of every group, group after group."""

    if not groups:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(groups)


def _equal_length_blocks(
    grouped_values: np.ndarray,
    group_of_value: np.ndarray,
    group_count: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield group positions and one 2-D value block per group length.

    ``grouped_values`` holds each group's values in order, group after group, and
    ``group_of_value`` gives the group of each value. Each block row holds one group's
    values in their original order. NumPy reduces each row of the C-contiguous block
    along axis 1 as it reduces that group alone, so the results match a per-group loop.
    One reduction per group length replaces one NumPy call per group and feature.
    """

    lengths = np.bincount(group_of_value, minlength=group_count)
    for positions, index in _length_buckets(lengths):
        yield positions, grouped_values[index]


def _length_buckets(lengths: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the positions of the groups of each nonzero length and their value indices.

    The value indices assume the groups' values lie group after group in one array.
    """

    starts = np.cumsum(lengths) - lengths
    buckets = []
    for length in np.unique(lengths[lengths > 0]).tolist():
        positions = np.flatnonzero(lengths == length)
        buckets.append((positions, starts[positions, None] + np.arange(length)))
    return buckets


def _divide_where(numerator: np.ndarray, denominator: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Divide where ``where`` holds and return 0.0 elsewhere."""

    result = np.zeros(len(numerator), dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=where)
    return result


def _probability_feature_columns(probs: np.ndarray, feature_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Compute probability features for equal-length rows of ranked candidate probabilities.

    Each row holds one query's candidate probabilities, highest first. Each feature uses
    the same element operations as its per-query definition, including how a NaN sum
    meets each guard.
    """

    row_count, n_probs = probs.shape
    zeros = np.zeros(row_count, dtype=np.float64)
    top = probs[:, 0]
    second = probs[:, 1] if n_probs > 1 else zeros
    third = probs[:, 2] if n_probs > 2 else zeros
    prob_sum = np.sum(probs, axis=1)
    count_thresholds = {
        "prob_count_ge_0_25": 0.25,
        "prob_count_ge_0_50": 0.50,
        "prob_count_ge_0_75": 0.75,
    }
    top_margins = {
        "prob_count_within_0_01": 0.01,
        "prob_count_within_0_05": 0.05,
        "prob_count_within_0_10": 0.10,
    }
    columns: dict[str, np.ndarray] = {}
    for feature_name in feature_names:
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
            value = _divide_where(top, prob_sum, prob_sum > 0)
        elif feature_name == "prob_top2_share":
            value = _divide_where(np.sum(probs[:, :2], axis=1), prob_sum, prob_sum > 0)
        elif feature_name == "prob_sum":
            value = prob_sum
        elif feature_name == "prob_mean":
            value = np.mean(probs, axis=1)
        elif feature_name == "prob_std":
            value = np.std(probs, axis=1)
        elif feature_name == "prob_p90":
            value = np.quantile(probs, 0.9, axis=1)
        elif feature_name == "prob_p75":
            value = np.quantile(probs, 0.75, axis=1)
        elif feature_name == "prob_entropy":
            value = zeros.copy()
            if n_probs > 1:
                # A NaN sum fails the "<= 0" guard, so its row gets NaN, not 0.0.
                rows = ~(prob_sum <= 0.0)
                shares = probs[rows] / prob_sum[rows, None]
                entropy = -np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)), axis=1)
                value[rows] = entropy / math.log(n_probs)
        elif feature_name == "prob_effective_candidates":
            value = zeros.copy()
            rows = ~(prob_sum <= 0.0)
            shares = probs[rows] / prob_sum[rows, None]
            sums = np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)), axis=1)
            # math.exp, not np.exp, keeps the per-query results bit for bit.
            value[rows] = [math.exp(-total) for total in sums.tolist()]
        elif feature_name in count_thresholds:
            value = (probs >= count_thresholds[feature_name]).sum(axis=1).astype(np.float64)
        elif feature_name in top_margins:
            value = (probs >= (top - top_margins[feature_name])[:, None]).sum(axis=1).astype(np.float64)
        else:
            raise KeyError(f"Unsupported logistic gate probability feature: {feature_name}")
        columns[feature_name] = value
    return columns


def _validate_selected_feature_sources(
    feature_names: Sequence[str],
    feature_values: Mapping[str, Any],
) -> None:
    """Reject selected gate features whose runtime source column is absent."""

    requirements, bucket_features = _feature_source_requirements(tuple(str(name) for name in feature_names))
    missing: list[tuple[str, str]] = []
    for feature_name, source_name, needs_query_view in requirements:
        if source_name is not None and source_name not in feature_values:
            missing.append((feature_name, source_name))
        elif needs_query_view and "query_view" not in feature_values:
            missing.append((feature_name, "query_view"))

    has_bucket_source = "first_name_bucket" in feature_values or (
        "query_first_token" in feature_values and "query_view" in feature_values
    )
    if bucket_features and not has_bucket_source:
        missing.extend(
            (feature_name, "first_name_bucket or (query_first_token and query_view)")
            for feature_name in bucket_features
        )

    if missing:
        details = ", ".join(f"{feature!r} requires {source!r}" for feature, source in missing[:5])
        raise ValueError(f"logistic gate selected feature sources are missing: {details}")


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

    plan = _gate_feature_plan(tuple(str(feature_name) for feature_name in feature_names))
    resolved_feature_names = plan.feature_names
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
        if plan.bucket_features:
            raise ValueError(
                "logistic gate first_name_bucket_* and gate_bucket_* features require 'first_name_bucket' or both"
                " 'query_first_token' and 'query_view' in feature_values"
            )
        first_name_bucket = _object_array(feature_values, "first_name_bucket", row_count)
    has_query_first_token = "query_first_token" in feature_values
    has_query_author = "query_author" in feature_values
    query_first_token = _object_array(feature_values, "query_first_token", row_count)
    query_author = _object_array(feature_values, "query_author", row_count)

    matrix = np.empty((len(query_rows.groups), len(resolved_feature_names)), dtype=np.float64)
    feature_positions = plan.positions
    handled: set[str] = set()

    def set_column(name: str, values: np.ndarray) -> None:
        position = feature_positions.get(name)
        if position is None:
            return
        matrix[:, position] = values
        handled.add(name)

    best_rows = query_rows.best_rows
    group_count = len(query_rows.groups)
    group_lengths = np.asarray([len(group) for group in query_rows.ranked_groups], dtype=np.int64)
    group_starts = np.cumsum(group_lengths) - group_lengths
    length_buckets = _length_buckets(group_lengths)
    set_column("chosen_probability", probability_array[best_rows])
    set_column("second_probability", query_rows.runner_up_scores)
    set_column("score_margin", query_rows.score_margins)
    set_column("has_runner_up", query_rows.has_runner_up.astype(np.float64))
    set_column("raw_candidate_count", group_lengths.astype(np.float64))

    ranked_rows = _concatenated_rows(query_rows.ranked_groups)
    if plan.probability_names:
        if plan.unsupported_probability_name is not None:
            raise KeyError(f"Unsupported logistic gate probability feature: {plan.unsupported_probability_name}")
        ranked_probabilities = probability_array[ranked_rows]
        for positions, index in length_buckets:
            columns = _probability_feature_columns(ranked_probabilities[index], plan.probability_names)
            for feature_name, values in columns.items():
                matrix[positions, feature_positions[feature_name]] = values
        handled.update(plan.probability_names)

    candidate_kind = np.where(query_rows.has_runner_up, "multi_candidate", "single_candidate")
    best_first_name_bucket = first_name_bucket[best_rows].astype(str)
    best_query_view = query_view[best_rows].astype(str)
    gate_bucket = np.asarray(
        [f"{kind}|{bucket}" for kind, bucket in zip(candidate_kind, best_first_name_bucket, strict=True)],
        dtype=object,
    )
    categorical_values = {
        "candidate_kind_": candidate_kind,
        "first_name_bucket_": best_first_name_bucket,
        "query_view_": best_query_view,
        "gate_bucket_": gate_bucket,
    }
    for feature_name, prefix, value in plan.categorical:
        set_column(feature_name, (categorical_values[prefix] == value).astype(float))

    has_second = group_lengths > 1
    second_rows = ranked_rows[group_starts[has_second] + 1]
    group_rows = _concatenated_rows(query_rows.groups)
    group_of_row = np.repeat(np.arange(group_count), group_lengths)
    cache: dict[str, np.ndarray] = {}
    finite_list_sources: list[tuple[str, frozenset[str], np.ndarray]] = []
    for source_name, ops in plan.source_ops:
        values = cache.setdefault(source_name, _feature_array(feature_values, source_name, row_count))
        if "top" in ops:
            set_column(f"top_raw_{source_name}", values[best_rows])
        if "delta" in ops:
            deltas = np.full(group_count, np.nan, dtype=np.float64)
            deltas[has_second] = values[best_rows[has_second]] - values[second_rows]
            set_column(f"delta_top_second_{source_name}", deltas)
        list_ops = ops & _LIST_OPS
        if list_ops:
            grouped_values = values[group_rows]
            finite = np.isfinite(grouped_values)
            if finite.all():
                finite_list_sources.append((source_name, list_ops, grouped_values))
                continue
            aggregates = {op: np.full(group_count, np.nan, dtype=np.float64) for op in list_ops}
            # A query whose values are all missing gets no block, so its aggregates stay NaN.
            for positions, block in _equal_length_blocks(grouped_values[finite], group_of_row[finite], group_count):
                if "mean" in list_ops:
                    aggregates["mean"][positions] = np.mean(block, axis=1)
                if "std" in list_ops:
                    aggregates["std"][positions] = np.std(block, axis=1, ddof=0)
                if "min" in list_ops:
                    aggregates["min"][positions] = np.min(block, axis=1)
                if "max" in list_ops:
                    aggregates["max"][positions] = np.max(block, axis=1)
            for op, values_for_op in aggregates.items():
                set_column(f"list_{op}_{source_name}", values_for_op)
    if finite_list_sources:
        # Reduce all sources without missing values at once. Each row of the 2-D block holds one
        # (source, query) pair in the per-source order. NumPy sums a contiguous row pairwise, as in the
        # per-source path; a strided row is summed in order and can round differently.
        stacked = np.stack([grouped_values for _, _, grouped_values in finite_list_sources])
        source_count = len(finite_list_sources)
        needed = frozenset().union(*(list_ops for _, list_ops, _ in finite_list_sources))
        batched = {op: np.full((source_count, group_count), np.nan, dtype=np.float64) for op in needed}
        for positions, index in length_buckets:
            block = np.ascontiguousarray(stacked[:, index]).reshape(source_count * len(positions), index.shape[1])
            if "mean" in needed:
                batched["mean"][:, positions] = np.mean(block, axis=1).reshape(source_count, -1)
            if "std" in needed:
                batched["std"][:, positions] = np.std(block, axis=1, ddof=0).reshape(source_count, -1)
            if "min" in needed:
                batched["min"][:, positions] = np.min(block, axis=1).reshape(source_count, -1)
            if "max" in needed:
                batched["max"][:, positions] = np.max(block, axis=1).reshape(source_count, -1)
        for source_position, (source_name, list_ops, _) in enumerate(finite_list_sources):
            for op in list_ops:
                set_column(f"list_{op}_{source_name}", batched[op][source_position])

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
    feature_values = feature_values_from_runtime(feature_matrix, row_signals)
    _validate_selected_feature_sources(gate.feature_names, feature_values)
    return build_logistic_gate_matrix(
        gate.feature_names,
        query_indices=candidate_batch.row_query_signature_indices,
        probabilities=probabilities,
        feature_values=feature_values,
        retrieval_ranks=candidate_batch.retrieval_ranks,
        component_keys=candidate_batch.row_component_keys,
    )
