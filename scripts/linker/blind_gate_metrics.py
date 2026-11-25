"""Score query outcomes on fully judged blind gate windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import Booster

from s2and.incremental_linking.logistic_gate import load_logistic_gate_config
from s2and.incremental_linking_training.gate import (
    _apply_promoted_logistic_gate_to_candidate_rows,
    _summarize_predictions,
)
from s2and.incremental_linking_training.ranker import (
    CalibratedClassicModel,
    _classic_feature_matrix,
    _score_query_choices,
)
from s2and.thread_config import resolve_n_jobs


def with_query_gold_scored(choices: pd.DataFrame, candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Carry window-level eligibility alongside the chosen candidate's judgment."""
    eligibility = candidate_rows.drop_duplicates("query_group_id").set_index("query_group_id")["query_gold_scored"]
    return choices.assign(query_gold_scored=choices["query_case_id"].map(eligibility))


def summarize_blind_gate_predictions(predictions: pd.DataFrame) -> dict:
    """Exclude incomplete windows from query metrics."""
    resolved = predictions.loc[predictions["query_gold_scored"]]
    return {**_summarize_predictions(resolved), "n_unscored_queries": len(predictions) - len(resolved)}


def update_blind_gate_metrics(
    calibrated: CalibratedClassicModel,
    rows: pd.DataFrame,
    feature_columns: tuple[str, ...],
    n_jobs: int,
) -> None:
    """Report eligible held-out queries while retaining candidate-level link metrics."""
    matrix = _classic_feature_matrix(rows, feature_columns)
    model = calibrated.model
    if isinstance(model, Booster):
        probabilities = np.asarray(model.predict(matrix, num_threads=resolve_n_jobs(n_jobs)), dtype=np.float64)
    else:
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)[:, 1]
    choices = _score_query_choices(rows, probabilities, query_id_column="query_group_id", include_margin=True)
    choices = with_query_gold_scored(choices, rows)
    gate = load_logistic_gate_config(calibrated.gate_config)
    predictions = _apply_promoted_logistic_gate_to_candidate_rows(rows, choices, probabilities, gate)
    metrics = summarize_blind_gate_predictions(predictions)

    # The training split and the candidate-level weight sweep use their original judgments.
    summary = calibrated.abstain_rule_summary
    promoted = summary["promoted_logistic_gate"]
    promoted["validation_metrics"] = metrics
    promoted["calibration_split_metrics"]["gate_thresh"] = metrics
    summary["internal_eval_metrics"] = metrics
