"""Build cutoff gates and score linker outcomes."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view, normalize_bucket_letters
from s2and.incremental_linking.logistic_gate import (
    LOGISTIC_GATE_ERROR_WEIGHTS,
    NumpyLogisticGate,
    build_logistic_gate_matrix,
    feature_values_from_candidate_frame,
    logistic_gate_config,
)

from .source_bundle import _is_missing_scalar

_GATE_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
    "single_candidate|multi_letter_first",
    "single_candidate|single_letter_first",
)


_PROMOTED_LOGISTIC_GATE_MODE = "promoted_logistic_topk_multiclass_l2"


_UNSUPPORTED_STRATIFIED_THRESHOLD_GATE_KEYS = frozenset(
    {
        "fixed_grid_step",
        "selection_metric",
        "score_thresholds",
        "margin_thresholds",
        "bucketed_score_thresholds",
        "bucketed_margin_thresholds",
    }
)


def cutoff_gate_config(t_full: float, t_init: float) -> dict[str, Any]:
    """Encode the two ranker-probability cutoffs in the logistic gate format.

    Args:
        t_full: Link above this probability for full-name queries.
        t_init: Link above this probability for initial-only queries.

    Returns:
        A JSON-compatible gate configuration.
    """
    if not (0 <= t_full <= 1 and 0 <= t_init <= 1):
        raise ValueError("Gate cutoffs must be probabilities in [0, 1]")

    slope = 16.0
    view_offset = t_full - t_init
    # Center the safe logit at zero to preserve precision near the cutoff.
    # The cost comparison at equality is exp(safe) = 4*exp(no_safe) + 5*exp(wrong).
    no_safe = -math.log(4.0 + 5.0 * math.exp(-30.0))
    # Round toward abstention at a tie. Cancel rounding from the initial-only offset.
    no_safe += 2 * math.ulp(no_safe)
    initial_roundoff = slope * ((t_init + view_offset) - t_full)
    return logistic_gate_config(
        feature_names=("chosen_probability", "query_view_initial_only"),
        weights=((0.0, 0.0, slope), (initial_roundoff, initial_roundoff, slope * view_offset)),
        bias=(no_safe, no_safe - 30.0, -slope * t_full),
        missing_values=(0.0, 0.0),
        calibration_mode="ranker_probability_cutoffs",
        training_summary={"t_full": t_full, "t_init": t_init},
    )


def _query_first_token(author: Any) -> str:
    """Return the first alphabetic token from a query author string."""

    if _is_missing_scalar(author):
        return ""
    tokens = re.findall(r"[A-Za-z]+", str(author))
    return tokens[0].lower() if tokens else ""


def _classic_gate_first_name_bucket(row: pd.Series | dict[str, Any]) -> str:
    """Classify a query into the gate's first-name-length bucket."""

    token = normalize_bucket_letters(row.get("query_first_token", ""))
    query_author = row.get("query_author")
    if not token and query_author is not None and pd.notna(query_author) and str(query_author).strip():
        token = _query_first_token(row.get("query_author"))
    return first_name_bucket_from_token_view(token, row.get("query_view", ""))


def _promoted_stratified_gate_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Read the historical source bundle’s calibration and test split names."""

    configured = spec.get("promoted_stratified_gate")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError("classic.promoted_stratified_gate must be a mapping when provided")
    out = dict(configured)
    mode = str(out.get("mode", _PROMOTED_LOGISTIC_GATE_MODE))
    if mode != _PROMOTED_LOGISTIC_GATE_MODE:
        raise ValueError(f"classic.promoted_stratified_gate.mode must be {_PROMOTED_LOGISTIC_GATE_MODE!r}")
    unsupported_keys = sorted(_UNSUPPORTED_STRATIFIED_THRESHOLD_GATE_KEYS.intersection(out))
    if unsupported_keys:
        raise ValueError(
            f"classic.promoted_stratified_gate no longer supports threshold calibration keys: {unsupported_keys}"
        )
    out["mode"] = _PROMOTED_LOGISTIC_GATE_MODE
    split_spec = spec.get("stratified_eval_test_split")
    split_spec = split_spec if isinstance(split_spec, dict) else {}
    calibration_splits = out.get("calibration_splits")
    if calibration_splits is None:
        calibration_splits = [
            split_spec.get("calibration_fit_split", "calibration_fit"),
            split_spec.get("calibration_check_split", "calibration_check"),
        ]
    if isinstance(calibration_splits, str) or not isinstance(calibration_splits, Sequence):
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must be a sequence of split names")
    resolved_calibration_splits = [str(split).strip() for split in calibration_splits]
    if not resolved_calibration_splits or any(not split for split in resolved_calibration_splits):
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must contain nonempty names")
    if len(set(resolved_calibration_splits)) != len(resolved_calibration_splits):
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must not contain duplicates")
    test_split = str(out.get("test_split", split_spec.get("test_split", "test"))).strip()
    if not test_split:
        raise ValueError("classic.promoted_stratified_gate.test_split must be nonempty")
    if test_split in resolved_calibration_splits:
        raise ValueError("classic.promoted_stratified_gate calibration_splits must not include test_split")
    out["calibration_splits"] = resolved_calibration_splits
    out["test_split"] = test_split

    return out


def _weighted_error_metrics(
    *,
    n_queries: int,
    false_abstain: int,
    false_link: int,
    wrong_candidate_link: int,
    error_weights: Mapping[str, float] = LOGISTIC_GATE_ERROR_WEIGHTS,
) -> dict[str, Any]:
    false_abstain_error_rate = float(false_abstain) / float(n_queries) if n_queries else 0.0
    false_link_error_rate = float(false_link) / float(n_queries) if n_queries else 0.0
    wrong_link_error_rate = float(wrong_candidate_link) / float(n_queries) if n_queries else 0.0
    weight_total = float(sum(float(value) for value in error_weights.values()))
    weighted_average_error = (
        (
            float(error_weights["false_abstain"]) * false_abstain_error_rate
            + float(error_weights["false_link"]) * false_link_error_rate
            + float(error_weights["wrong_candidate_link"]) * wrong_link_error_rate
        )
        / weight_total
        if weight_total
        else 0.0
    )
    return {
        "false_abstain_error_rate": false_abstain_error_rate,
        "false_link_error_rate": false_link_error_rate,
        "wrong_link_error_rate": wrong_link_error_rate,
        "weighted_average_error": weighted_average_error,
        "weighted_average_error_weights": {
            "false_abstain_error_rate": float(error_weights["false_abstain"]),
            "false_link_error_rate": float(error_weights["false_link"]),
            "wrong_link_error_rate": float(error_weights["wrong_candidate_link"]),
        },
    }


def _summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    if "chosen_candidate_gold_scored" in predictions:
        predictions = predictions.loc[predictions["chosen_candidate_gold_scored"]]
    positives = predictions[predictions["query_safe_target"] == 1]
    negatives = predictions[predictions["query_safe_target"] == 0]
    accepted = predictions["predicted_action"] == "link_candidate"
    correct_link = accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 1)
    false_abstain = (~accepted) & (predictions["query_safe_target"] == 1)
    false_link = accepted & (predictions["query_safe_target"] == 0)
    wrong_candidate_link = (
        accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 0)
    )
    tp = int(correct_link.sum())
    # A wrong candidate link is both an accepted incorrect link (precision error)
    # and a missed correct positive link (recall error), so these diagnostic
    # counts are intentionally not a mutually exclusive confusion matrix.
    fp = int((accepted & ~correct_link).sum())
    tn = int(((predictions["predicted_action"] == "abstain") & (predictions["query_safe_target"] == 0)).sum())
    fn = int(
        (
            ((predictions["predicted_action"] == "abstain") & (predictions["query_safe_target"] == 1))
            | (accepted & (predictions["query_safe_target"] == 1) & (predictions["chosen_candidate_target"] == 0))
        ).sum()
    )
    positive_recall = float(positives["correct"].mean()) if len(positives) else 0.0
    negative_recall = float(negatives["correct"].mean()) if len(negatives) else 0.0
    link_precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    link_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    errors = int((predictions["correct"] == 0).sum())
    summary = {
        "target_semantics": "query_safe_target_with_explicit_source",
        "n_queries": int(len(predictions)),
        "n_positive_queries": int(len(positives)),
        "n_negative_queries": int(len(negatives)),
        "accuracy": float(predictions["correct"].mean()) if len(predictions) else 0.0,
        "errors": errors,
        "error_rate": float(errors / len(predictions)) if len(predictions) else 0.0,
        "balanced_accuracy": (positive_recall + negative_recall) / 2.0,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "link_precision": link_precision,
        "link_recall": link_recall,
        "abstain_rate": float((predictions["predicted_action"] == "abstain").mean()) if len(predictions) else 0.0,
        "positive_forced_choice_accuracy": (
            float(positives["chosen_candidate_target"].mean()) if len(positives) else 0.0
        ),
        "false_abstain": int(false_abstain.sum()),
        "false_link": int(false_link.sum()),
        "wrong_candidate_link": int(wrong_candidate_link.sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    summary.update(
        _weighted_error_metrics(
            n_queries=int(len(predictions)),
            false_abstain=int(false_abstain.sum()),
            false_link=int(false_link.sum()),
            wrong_candidate_link=int(wrong_candidate_link.sum()),
        )
    )
    return summary


def _query_gate_bucket(rows: pd.DataFrame) -> pd.Series:
    """Return candidate-kind/first-name gate buckets for scored query choices."""

    if "candidate_kind" in rows.columns:
        candidate_kind = rows["candidate_kind"].astype(str)
    else:
        has_runner_up = pd.to_numeric(rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 1
        score_margin = pd.to_numeric(rows["score_margin"], errors="coerce")
        candidate_kind = pd.Series(
            np.where(has_runner_up & score_margin.notna(), "multi_candidate", "single_candidate"),
            index=rows.index,
        )
    if "first_name_bucket" in rows.columns:
        first_name_bucket = rows["first_name_bucket"].astype(str)
    else:
        first_name_bucket = rows.apply(_classic_gate_first_name_bucket, axis=1)
    return candidate_kind + "|" + first_name_bucket


def _summarize_training_gate_buckets(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count post-filter training rows and queries by promoted gate bucket."""

    query_ids = train_df["query_group_id"].astype(str)
    row_counts = query_ids.value_counts(sort=False)
    query_representatives = train_df.assign(_query_group_id=query_ids).groupby("_query_group_id", sort=False).head(1)
    candidate_kind = pd.Series(
        np.where(
            query_representatives["_query_group_id"].map(row_counts).astype(int).gt(1),
            "multi_candidate",
            "single_candidate",
        ),
        index=query_representatives.index,
    )
    bucket_frame = pd.DataFrame(
        {
            "bucket": candidate_kind + "|" + query_representatives.apply(_classic_gate_first_name_bucket, axis=1),
            "row_count": query_representatives["_query_group_id"].map(row_counts).astype(int),
        }
    )
    query_counts = bucket_frame["bucket"].value_counts(sort=False).to_dict()
    training_row_counts = bucket_frame.groupby("bucket", sort=False)["row_count"].sum().to_dict()
    return {
        "query_counts": {bucket: int(query_counts.get(bucket, 0)) for bucket in _GATE_BUCKETS},
        "row_counts": {bucket: int(training_row_counts.get(bucket, 0)) for bucket in _GATE_BUCKETS},
    }


def _gate_bucket_split_counts(predictions: pd.DataFrame, split_order: Sequence[str]) -> dict[str, dict[str, int]]:
    """Count scored query choices by promoted gate bucket and split."""

    if predictions.empty:
        return {bucket: {str(split): 0 for split in split_order} for bucket in _GATE_BUCKETS}
    counts = predictions.groupby(["gate_bucket", "split"], dropna=False, sort=False).size()
    return {
        bucket: {str(split): int(counts.get((bucket, str(split)), 0)) for split in split_order}
        for bucket in _GATE_BUCKETS
    }


def _apply_logistic_gate_predictions(choices: pd.DataFrame, link: np.ndarray) -> pd.DataFrame:
    """Apply query-level logistic gate decisions to scored query choices."""

    pred = choices.copy()
    if len(pred) != len(link):
        raise ValueError(f"logistic gate link count must match choices: {len(link)} != {len(pred)}")
    pred["predicted_action"] = np.where(np.asarray(link, dtype=bool), "link_candidate", "abstain")
    pred["correct"] = (
        ((pred["predicted_action"] == "abstain") & (pred["query_safe_target"] == 0))
        | (
            (pred["predicted_action"] == "link_candidate")
            & (pred["query_safe_target"] == 1)
            & (pred["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return pred


def _apply_promoted_logistic_gate_to_candidate_rows(
    candidate_rows: pd.DataFrame,
    choices: pd.DataFrame,
    probabilities: np.ndarray,
    gate: NumpyLogisticGate,
) -> pd.DataFrame:
    """Apply a logistic gate to raw candidate rows and scored choices."""

    matrix, query_rows = build_logistic_gate_matrix(
        gate.feature_names,
        query_indices=candidate_rows["query_group_id"].astype(str).to_numpy(),
        probabilities=probabilities,
        feature_values=feature_values_from_candidate_frame(candidate_rows),
        retrieval_ranks=candidate_rows["retrieval_rank"].to_numpy() if "retrieval_rank" in candidate_rows else None,
        component_keys=(
            candidate_rows["candidate_component_key"].astype(str).to_numpy()
            if "candidate_component_key" in candidate_rows
            else None
        ),
    )
    if len(matrix) != len(choices):
        raise ValueError(f"logistic gate query matrix mismatch: {len(matrix)} != {len(choices)}")
    matrix_query_ids = candidate_rows["query_group_id"].astype(str).to_numpy()[query_rows.best_rows]
    choice_query_ids = choices["query_case_id"].astype(str).to_numpy()
    if not np.array_equal(matrix_query_ids, choice_query_ids):
        mismatch_positions = np.flatnonzero(matrix_query_ids != choice_query_ids)[:5]
        sample = [
            {
                "position": int(position),
                "matrix_query_id": str(matrix_query_ids[position]),
                "choice_query_id": str(choice_query_ids[position]),
            }
            for position in mismatch_positions
        ]
        raise ValueError(f"logistic gate query order mismatch between candidate rows and choices: sample={sample}")
    return _apply_logistic_gate_predictions(choices, gate.predict_link(matrix))


WEIGHTED_ERROR_WEIGHTS = {
    "false_abstain_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["false_abstain"]),
    "false_link_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["false_link"]),
    "wrong_link_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["wrong_candidate_link"]),
}
