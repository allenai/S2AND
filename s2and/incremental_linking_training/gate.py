"""Fit and apply the linker gate."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view, normalize_bucket_letters
from s2and.incremental_linking.logistic_gate import (
    LOGISTIC_GATE_ERROR_WEIGHTS,
    LOGISTIC_GATE_FINAL_C,
    LOGISTIC_GATE_SELECTION_C,
    LOGISTIC_GATE_SELECTION_FEATURE_COUNT,
    NumpyLogisticGate,
    build_logistic_gate_matrix,
    default_logistic_gate_feature_names,
    feature_values_from_candidate_frame,
    load_logistic_gate_config,
    logistic_gate_config,
    outcome_labels,
)

from .source_bundle import _is_missing_scalar

_GATE_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
    "single_candidate|multi_letter_first",
    "single_candidate|single_letter_first",
)


_MISSING_LOGISTIC_CLASS_LOGIT = -30.0
_SAFE_LINK_CLASS = 2
_INITIAL_ONLY_FEATURE = "query_view_initial_only"
_GATE_CV_FOLDS = 4
_GATE_CV_SEED = 20261125
_GATE_CV_C = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
_GATE_CV_K = (60, 120, 240, "all")


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
    """Return configured stratified split names for logistic gate calibration."""

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

    fit_splits_raw = out.get("fit_splits")
    validation_splits_raw = out.get("validation_splits")
    if fit_splits_raw is not None:
        if isinstance(fit_splits_raw, str) or not isinstance(fit_splits_raw, Sequence):
            raise ValueError("classic.promoted_stratified_gate.fit_splits must be a sequence of split names")
        fit_splits = [str(s).strip() for s in fit_splits_raw]
        if not fit_splits or any(not s for s in fit_splits) or len(set(fit_splits)) != len(fit_splits):
            raise ValueError("classic.promoted_stratified_gate.fit_splits must contain unique nonempty names")
        if not set(fit_splits).issubset(resolved_calibration_splits):
            raise ValueError("classic.promoted_stratified_gate.fit_splits must be a subset of calibration_splits")
        out["fit_splits"] = fit_splits
    if validation_splits_raw is not None:
        if fit_splits_raw is None:
            raise ValueError("classic.promoted_stratified_gate.validation_splits requires explicit fit_splits")
        if isinstance(validation_splits_raw, str) or not isinstance(validation_splits_raw, Sequence):
            raise ValueError("classic.promoted_stratified_gate.validation_splits must be a sequence of split names")
        val_splits = [str(s).strip() for s in validation_splits_raw]
        if not val_splits or any(not s for s in val_splits) or len(set(val_splits)) != len(val_splits):
            raise ValueError("classic.promoted_stratified_gate.validation_splits must contain unique nonempty names")
        if not set(val_splits).issubset(resolved_calibration_splits):
            raise ValueError(
                "classic.promoted_stratified_gate.validation_splits must be a subset of calibration_splits"
            )
        if set(out["fit_splits"]).intersection(val_splits):
            raise ValueError("classic.promoted_stratified_gate fit_splits and validation_splits must be disjoint")
        out["validation_splits"] = val_splits
    tune_weights = out.get("tune_false_abstain_weights")
    if tune_weights is not None:
        if validation_splits_raw is None:
            raise ValueError("classic.promoted_stratified_gate.tune_false_abstain_weights requires validation_splits")
        if isinstance(tune_weights, str) or not isinstance(tune_weights, Sequence) or not tune_weights:
            raise ValueError(
                "classic.promoted_stratified_gate.tune_false_abstain_weights must be a nonempty sequence of floats"
            )
        out["tune_false_abstain_weights"] = [float(w) for w in tune_weights]
    if "false_abstain_weight" in out:
        weight = float(out["false_abstain_weight"])
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("classic.promoted_stratified_gate.false_abstain_weight must be finite and positive")
        out["false_abstain_weight"] = weight
    if "initial_only_offset" in out:
        offset = float(out["initial_only_offset"])
        if not np.isfinite(offset):
            raise ValueError("classic.promoted_stratified_gate.initial_only_offset must be finite")
        out["initial_only_offset"] = offset
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


def _fold_standard_scaler_into_logistic(
    scaler: StandardScaler,
    model: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw-space weights and bias for scaler -> logistic gate."""

    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    classes = tuple(int(value) for value in model.classes_)
    if coef.shape[0] == 1 and len(classes) == 2:
        binary_weights = coef[0] / scale
        binary_bias = float(intercept[0] - mean @ binary_weights)
        weights = np.zeros((len(scale), 3), dtype=np.float64)
        # LogisticRegression omits absent classes; keep their softmax mass negligible.
        bias = np.full(3, _MISSING_LOGISTIC_CLASS_LOGIT, dtype=np.float64)
        # Binary sklearn stores positive-class log odds; softmax([0, z]) reproduces predict_proba.
        negative_class, positive_class = classes
        weights[:, negative_class] = 0.0
        bias[negative_class] = 0.0
        weights[:, positive_class] = binary_weights
        bias[positive_class] = binary_bias
        return weights, bias

    raw_weights = (coef / scale[None, :]).T
    raw_bias = intercept - mean @ raw_weights
    weights = np.zeros((len(scale), 3), dtype=np.float64)
    bias = np.full(3, _MISSING_LOGISTIC_CLASS_LOGIT, dtype=np.float64)
    for source_index, class_value in enumerate(classes):
        weights[:, class_value] = raw_weights[:, source_index]
        bias[class_value] = raw_bias[source_index]
    return weights, bias


def _fill_gate_matrix(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Replace nonfinite gate features with the fitted column medians."""
    filled = np.array(matrix, dtype=np.float64, copy=True)
    rows, columns = np.nonzero(~np.isfinite(filled))
    filled[rows, columns] = medians[columns]
    return filled


def _fit_multiclass_logistic_gate(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """Fit the gate and return the fitted scaler and imputation medians."""

    raw_matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(raw_matrix)
    medians = np.zeros(raw_matrix.shape[1], dtype=np.float64)
    for column_index in np.flatnonzero(finite.any(axis=0)):
        medians[column_index] = np.median(raw_matrix[finite[:, column_index], column_index])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(_fill_gate_matrix(raw_matrix, medians))
    model = LogisticRegression(
        C=float(c_value),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=4000,
        random_state=20260517,
    )
    model.fit(scaled, labels)
    return model, scaler, medians


def _rank_gate_features(
    matrix: np.ndarray, labels: np.ndarray, names: tuple[str, ...], selection_c: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rank features by their largest absolute standardized class coefficient."""
    model, _, _ = _fit_multiclass_logistic_gate(matrix, labels, c_value=selection_c)
    importance = np.max(np.abs(np.asarray(model.coef_, dtype=np.float64)), axis=0)
    return np.lexsort((np.asarray(names, dtype=object), -importance)), importance


def _select_gate_columns(ranked: np.ndarray, count: int, initial_only_index: int) -> np.ndarray:
    """Keep the initial-only feature within the selected feature budget."""
    selected = ranked[:count].copy()
    if initial_only_index not in selected:
        selected[-1] = initial_only_index
    return selected


def _choose_gate_hyperparameters(
    matrix: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    names: tuple[str, ...],
    selection_c: float,
) -> tuple[int, float, dict[str, Any]]:
    """Choose k and final C by query-weighted binary safe-link log loss across block folds."""
    block_names, counts = np.unique(blocks, return_counts=True)
    if len(block_names) < _GATE_CV_FOLDS:
        raise ValueError(
            f"Gate cross-validation requires at least {_GATE_CV_FOLDS} fit_pool blocks with scored queries, "
            f"found {len(block_names)}"
        )
    folds: list[list[str]] = [[] for _ in range(_GATE_CV_FOLDS)]
    loads = [0] * _GATE_CV_FOLDS
    # Balance query counts and break equal-count ties with a fixed block permutation.
    shuffled = np.random.default_rng(_GATE_CV_SEED).permutation(len(block_names))
    for index in shuffled[np.argsort(-counts[shuffled], kind="stable")]:
        slot = int(np.argmin(loads))
        folds[slot].append(str(block_names[index]))
        loads[slot] += int(counts[index])

    initial_only_index = names.index(_INITIAL_ONLY_FEATURE)
    feature_counts = [k if isinstance(k, int) else len(names) for k in _GATE_CV_K]
    losses = np.empty((_GATE_CV_FOLDS, len(feature_counts), len(_GATE_CV_C)))
    for fold_index, held_out in enumerate(folds):
        test = np.isin(blocks, held_out)
        train_matrix, train_labels = matrix[~test], labels[~test]
        test_matrix, safe = matrix[test], labels[test] == _SAFE_LINK_CLASS
        if np.unique(train_labels).size < 2:
            raise ValueError(f"Gate cross-validation needs two outcome classes outside held-out blocks {held_out}")
        ranked, _ = _rank_gate_features(train_matrix, train_labels, names, selection_c)
        for ki, count in enumerate(feature_counts):
            columns = _select_gate_columns(ranked, count, initial_only_index)
            train_features, test_features = train_matrix[:, columns], test_matrix[:, columns]
            for ci, c in enumerate(_GATE_CV_C):
                model, scaler, medians = _fit_multiclass_logistic_gate(train_features, train_labels, c_value=c)
                proba = model.predict_proba(scaler.transform(_fill_gate_matrix(test_features, medians)))
                # An absent safe class contributes zero predicted probability before clipping.
                p_safe = np.clip(proba[:, model.classes_ == _SAFE_LINK_CLASS].sum(axis=1), 1e-9, 1 - 1e-9)
                losses[fold_index, ki, ci] = -np.mean(np.where(safe, np.log(p_safe), np.log1p(-p_safe)))

    means = np.average(losses, weights=loads, axis=0)
    ki, ci = np.unravel_index(np.argmin(means), means.shape)
    return (
        feature_counts[ki],
        _GATE_CV_C[ci],
        {
            "seed": _GATE_CV_SEED,
            "k_grid": list(_GATE_CV_K),
            "c_grid": list(_GATE_CV_C),
            "folds": [
                {"blocks": held_out, "queries": loads[i], "log_loss": losses[i].tolist()}
                for i, held_out in enumerate(folds)
            ],
            "mean_log_loss": means.tolist(),
            "selected_k": _GATE_CV_K[ki],
            "selected_c": _GATE_CV_C[ci],
        },
    )


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


def _fit_promoted_logistic_gate(
    candidate_rows: pd.DataFrame,
    choices: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    calibration_splits: Sequence[str],
    fit_splits: Sequence[str] | None = None,
    validation_splits: Sequence[str] | None = None,
    error_weights: Mapping[str, float] | None = None,
    selection_feature_count: int = LOGISTIC_GATE_SELECTION_FEATURE_COUNT,
    selection_c: float = LOGISTIC_GATE_SELECTION_C,
    final_c: float = LOGISTIC_GATE_FINAL_C,
    grouped_cv: bool = False,
    initial_only_offset: float = 0.0,
) -> dict[str, Any]:
    """Fit the prod-safe NumPy logistic gate on configured calibration/fit splits.

    When ``fit_splits`` and ``validation_splits`` are separated, feature selection and
    model weights are fit strictly on ``fit_splits`` and out-of-sample ``validation_metrics``
    are computed on ``validation_splits``. Grouped CV selects the feature count and
    final regularization within the fit splits. The initial-only offset adjusts
    the fitted safe-link weight in raw feature space.
    """

    all_feature_names = default_logistic_gate_feature_names()
    full_matrix, query_rows = build_logistic_gate_matrix(
        all_feature_names,
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
    if len(full_matrix) != len(choices):
        raise ValueError(f"logistic gate query matrix mismatch: {len(full_matrix)} != {len(choices)}")
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
    split_values = choices["split"].astype(str)
    active_fit_splits = tuple(str(split) for split in (calibration_splits if fit_splits is None else fit_splits))
    active_val_splits = tuple(str(split) for split in validation_splits) if validation_splits is not None else ()
    if active_val_splits and set(active_fit_splits).intersection(active_val_splits):
        raise ValueError(
            f"logistic gate fit_splits and validation_splits must be disjoint: "
            f"fit={list(active_fit_splits)}, validation={list(active_val_splits)}"
        )
    calibration_mask = split_values.isin(tuple(str(split) for split in calibration_splits)).to_numpy(dtype=bool)
    fit_mask = split_values.isin(active_fit_splits).to_numpy(dtype=bool)
    val_mask = split_values.isin(active_val_splits).to_numpy(dtype=bool) if active_val_splits else None
    if not np.any(fit_mask):
        raise ValueError(f"logistic gate requires non-empty fit splits: {list(active_fit_splits)}")
    if val_mask is not None and not np.any(val_mask):
        raise ValueError(f"logistic gate requires non-empty validation splits: {list(active_val_splits)}")
    labels = outcome_labels(choices["query_safe_target"], choices["chosen_candidate_target"])

    fit_matrix, fit_labels = full_matrix[fit_mask], labels[fit_mask]
    cv_summary = None
    if grouped_cv:
        selection_feature_count, final_c, cv_summary = _choose_gate_hyperparameters(
            fit_matrix,
            fit_labels,
            choices.loc[fit_mask, "dataset"].astype(str).to_numpy(),
            all_feature_names,
            selection_c,
        )
    ranked_indices, importance = _rank_gate_features(fit_matrix, fit_labels, all_feature_names, selection_c)
    selected_indices = _select_gate_columns(
        ranked_indices, selection_feature_count, all_feature_names.index(_INITIAL_ONLY_FEATURE)
    )
    selected_feature_names = tuple(all_feature_names[int(index)] for index in selected_indices)
    selected_matrix = full_matrix[:, selected_indices]

    final_model, final_scaler, final_medians = _fit_multiclass_logistic_gate(
        selected_matrix[fit_mask],
        labels[fit_mask],
        c_value=float(final_c),
    )
    weights, bias = _fold_standard_scaler_into_logistic(final_scaler, final_model)
    if initial_only_offset:
        weights[selected_feature_names.index(_INITIAL_ONLY_FEATURE), _SAFE_LINK_CLASS] += initial_only_offset
    resolved_error_weights = dict(LOGISTIC_GATE_ERROR_WEIGHTS if error_weights is None else error_weights)
    training_summary: dict[str, Any] = {
        "calibration_splits": [str(split) for split in calibration_splits],
        "selection_feature_count": int(selection_feature_count),
        "selection_c": float(selection_c),
        "final_c": float(final_c),
        "initial_only_offset": float(initial_only_offset),
        "all_feature_count": int(len(all_feature_names)),
        "selected_feature_count": int(len(selected_feature_names)),
        "calibration_queries": int(calibration_mask.sum()),
    }
    if cv_summary is not None:
        training_summary["grouped_cv"] = cv_summary
    if fit_splits is not None or validation_splits is not None:
        training_summary.update(
            {
                "fit_splits": list(active_fit_splits),
                "validation_splits": list(active_val_splits),
                "fit_queries": int(fit_mask.sum()),
                "validation_queries": int(val_mask.sum()) if val_mask is not None else 0,
            }
        )
    gate_config = logistic_gate_config(
        feature_names=selected_feature_names,
        weights=weights,
        bias=bias,
        missing_values=final_medians,
        calibration_mode=_PROMOTED_LOGISTIC_GATE_MODE,
        error_weights=resolved_error_weights,
        training_summary=training_summary,
    )
    gate = load_logistic_gate_config(gate_config)
    eval_cal_mask = fit_mask if val_mask is not None else calibration_mask
    calibration_predictions = _apply_logistic_gate_predictions(
        choices[eval_cal_mask].copy(),
        gate.predict_link(selected_matrix[eval_cal_mask]),
    )
    validation_metrics = None
    if val_mask is not None:
        validation_predictions = _apply_logistic_gate_predictions(
            choices[val_mask].copy(),
            gate.predict_link(selected_matrix[val_mask]),
        )
        validation_metrics = _summarize_predictions(validation_predictions)
    split_predictions = {
        str(split): _apply_logistic_gate_predictions(
            choices[split_values.eq(str(split)).to_numpy(dtype=bool)].copy(),
            gate.predict_link(selected_matrix[split_values.eq(str(split)).to_numpy(dtype=bool)]),
        )
        for split in tuple(dict.fromkeys(split_values.tolist()))
    }
    return {
        "gate_config": gate_config,
        "selected_feature_importance": [
            {
                "feature": str(all_feature_names[int(index)]),
                "importance": float(importance[int(index)]),
            }
            for index in selected_indices
        ],
        "calibration_metrics": _summarize_predictions(calibration_predictions),
        "validation_metrics": validation_metrics,
        "split_metrics": {
            split: _summarize_predictions(predictions) for split, predictions in split_predictions.items()
        },
        "predictions": _apply_logistic_gate_predictions(choices.copy(), gate.predict_link(selected_matrix)),
    }


_FALSE_ABSTAIN_REPORT_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)


def report_logistic_gate_error_weights(
    base_gate_config: Mapping[str, Any],
    candidate_rows: pd.DataFrame,
    choices: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    false_abstain_weights: Sequence[float] = _FALSE_ABSTAIN_REPORT_WEIGHTS,
) -> dict[str, Any]:
    """Report gold link metrics and decision changes on held-out gate choices."""
    if choices.empty or candidate_rows.empty:
        raise ValueError("The gate weight report requires non-empty validation choices and candidate rows")
    if not false_abstain_weights:
        raise ValueError("false_abstain_weights must be nonempty")
    base_weights = dict(base_gate_config.get("error_weights", LOGISTIC_GATE_ERROR_WEIGHTS))
    base_fa = float(base_weights.get("false_abstain", LOGISTIC_GATE_ERROR_WEIGHTS["false_abstain"]))
    candidate_grid: list[float] = []
    for w_fa in (*false_abstain_weights, base_fa):
        w_val = float(w_fa)
        if not np.isfinite(w_val) or w_val <= 0:
            raise ValueError(f"false_abstain weight must be finite and positive, got {w_fa!r}")
        if not any(np.isclose(w_val, existing, rtol=0.0, atol=1e-12) for existing in candidate_grid):
            candidate_grid.append(w_val)
    candidate_grid.sort()

    base_gate = load_logistic_gate_config(base_gate_config)
    matrix, query_rows = build_logistic_gate_matrix(
        base_gate.feature_names,
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
        raise ValueError("logistic gate query order mismatch between candidate rows and choices")

    # Compute class probabilities once; mirror NumpyLogisticGate.predict_link exactly.
    class_proba = base_gate.predict_proba(matrix)
    w_fl = float(base_weights["false_link"])
    w_wc = float(base_weights["wrong_candidate_link"])
    link_cost = w_fl * class_proba[:, 0] + w_wc * class_proba[:, 1]
    safe_prob = class_proba[:, 1] + class_proba[:, 2]

    link_columns = ["query_case_id", "dataset", "chosen_candidate_component_key"]
    if "query_signature_id" in choices:
        link_columns.append("query_signature_id")
    links = choices[link_columns].rename(columns={"chosen_candidate_component_key": "candidate_component_key"})
    seed_targets = choices.get("gold_safe_target", choices["query_safe_target"]).astype(bool).to_numpy()
    gold_scored = (
        choices["chosen_candidate_gold_scored"].to_numpy(dtype=bool)
        if "chosen_candidate_gold_scored" in choices
        else np.ones(len(choices), dtype=bool)
    )
    correct = choices["chosen_candidate_target"].eq(1).to_numpy() & seed_targets & gold_scored
    gold_linkable = int(np.count_nonzero(seed_targets))
    previous = link_cost < base_fa * safe_prob
    previous_weight = base_fa
    sweep: list[dict[str, Any]] = []
    for w_val in candidate_grid:
        link = link_cost < w_val * safe_prob
        link_count = int(np.count_nonzero(link))
        scored_links = int(np.count_nonzero(link & gold_scored))
        correct_count = int(np.count_nonzero(link & correct))
        sweep.append(
            {
                "false_abstain_weight": w_val,
                "previous_weight": previous_weight,
                "link_count": link_count,
                "gold_scored_links": scored_links,
                "gold_correct_links": correct_count,
                "gold_precision": correct_count / scored_links if scored_links else None,
                "gold_recall": correct_count / gold_linkable if gold_linkable else None,
                "added_links": links.loc[link & ~previous].to_dict(orient="records"),
                "removed_links": links.loc[previous & ~link].to_dict(orient="records"),
            }
        )
        previous, previous_weight = link, w_val
    return {
        "configured_false_abstain_weight": base_fa,
        "gold_linkable_queries": gold_linkable,
        "n_queries": int(len(choices)),
        "sweep": sweep,
    }


def _apply_promoted_logistic_gate_to_candidate_rows(
    candidate_rows: pd.DataFrame,
    choices: pd.DataFrame,
    probabilities: np.ndarray,
    gate: NumpyLogisticGate,
) -> pd.DataFrame:
    """Apply a fitted logistic gate to raw candidate rows and scored choices."""

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
