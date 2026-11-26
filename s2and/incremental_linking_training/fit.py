"""Fit the linker ranker and gate, then score the held-out windows.

This module sits above `ranker` and `evaluate` so neither imports the other back.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier

from s2and.incremental_linking.contracts import DEFAULT_RETRIEVAL_TOP_K
from s2and.incremental_linking.logistic_gate import (
    LOGISTIC_GATE_ERROR_WEIGHTS,
    load_logistic_gate_config,
)
from s2and.thread_config import resolve_n_jobs

from .evaluate import _classic_internal_eval, _score_classic_stratified_rows, _score_frozen_gate_rows
from .gate import (
    _FALSE_ABSTAIN_REPORT_WEIGHTS,
    _PROMOTED_LOGISTIC_GATE_MODE,
    _fit_promoted_logistic_gate,
    _promoted_stratified_gate_spec,
    _summarize_predictions,
    _summarize_training_gate_buckets,
    report_logistic_gate_error_weights,
)
from .ranker import (
    DEFAULT_CLASSIC_N_JOBS,
    CalibratedClassicModel,
    _apply_classic_train_row_cap,
    _build_classic_classifier,
    _classic_feature_matrix,
    _drop_unlabeled_singleton_orcid_rows,
    _filter_candidate_rows_to_retrieval_top_k,
    _resolve_classic_monotone_constraints,
)
from .source_bundle import (
    ClassicHoldoutIdentities,
    OfficialBundle,
    _apply_classic_train_holdout_filter,
    _read_csv,
    _resolve_path,
    read_classic_holdout_identities,
)


def fit_linker(
    bundle: OfficialBundle,
    *,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
    holdout_identities: ClassicHoldoutIdentities | None = None,
    pre_materialization_holdout_summary: Mapping[str, Any] | None = None,
    frozen_task_rows: Mapping[str, pd.DataFrame] | None = None,
    gate_training_blocks: Collection[str] = (),
    fixed_ranker: Booster | None = None,
) -> CalibratedClassicModel:
    """Fit the booster and gate without reading frozen test labels/features.

    Args:
        bundle: Training and calibration feature tables.
        n_jobs: Number of classifier workers.
        holdout_identities: Complete source identity authority. Direct callers
            may omit this when the bundle retains complete holdout tables.
        pre_materialization_holdout_summary: Earlier exclusion counts to report.
            When supplied, any remaining overlap is a materialization failure.
        frozen_task_rows: Optional featureized fit_pool and gate_thresh windows.
        gate_training_blocks: Frozen fit_pool blocks marked for gate training.
        fixed_ranker: Verified source booster to reuse without fitting.

    Returns:
        Fitted booster, calibrated gate, and training diagnostics.
    """

    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    retrieval_top_k = int(spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    if retrieval_top_k <= 0:
        raise ValueError("classic.retrieval_top_k must be positive")
    configured_calibration_limit = int(spec.get("classic_gate_calibration_retrieval_limit", retrieval_top_k))
    if configured_calibration_limit != retrieval_top_k:
        raise ValueError(
            "classic_gate_calibration_retrieval_limit must match classic.retrieval_top_k: "
            f"{configured_calibration_limit} != {retrieval_top_k}"
        )
    calibration_limit = retrieval_top_k
    if frozen_task_rows is not None and not gate_training_blocks:
        raise ValueError("Frozen linker tasks require marked gate training blocks in fit_pool")
    model: LGBMClassifier | Booster
    if fixed_ranker is not None:
        model = fixed_ranker
        training_summary = {"ranker_refit": False, "rows": 0, "elapsed_seconds": 0.0}
    else:
        monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
        if frozen_task_rows is None:
            train_df = _read_csv(_resolve_path(bundle, spec["train_path"]), compression="gzip")
        else:
            frozen_train = frozen_task_rows["fit_pool"]
            train_df = frozen_train.loc[frozen_train["candidate_gold_scored"]].reset_index(drop=True)
        train_df, unlabeled_singleton_filter_summary = _drop_unlabeled_singleton_orcid_rows(
            train_df,
            context="classic_train",
        )
        train_rows_before_retrieval_window = int(len(train_df))
        train_df = _filter_candidate_rows_to_retrieval_top_k(
            train_df,
            retrieval_top_k=retrieval_top_k,
            context="classic_train",
        )
        train_rows_after_retrieval_window = int(len(train_df))
        train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(np.int8)
        if pre_materialization_holdout_summary is not None and holdout_identities is None:
            raise ValueError("Pre-materialization holdout summary requires complete holdout identities")
        identities = read_classic_holdout_identities(bundle) if holdout_identities is None else holdout_identities
        train_df, train_holdout_filter_summary = _apply_classic_train_holdout_filter(
            train_df,
            holdout_query_group_ids=set(identities.query_group_ids),
            holdout_base_group_ids=set(identities.base_group_ids),
            holdout_sources=list(identities.sources),
        )
        if pre_materialization_holdout_summary is not None:
            if train_holdout_filter_summary["rows_removed"]:
                raise ValueError("Materialized training rows overlap complete holdout identities")
            train_holdout_filter_summary = dict(pre_materialization_holdout_summary)
        if train_df.empty:
            raise ValueError("No training rows remain after holdout exclusion")
        train_df, train_filter_summary = _apply_classic_train_row_cap(
            train_df,
            rule_name=spec.get("train_row_cap_rule"),
            min_train_limit=(int(spec["train_row_cap_min_limit"]) if "train_row_cap_min_limit" in spec else None),
        )
        training_gate_bucket_summary = _summarize_training_gate_buckets(train_df)
        train_matrix = _classic_feature_matrix(train_df, feature_columns).to_numpy(dtype=np.float32)
        train_labels = train_df["label"].to_numpy(dtype=np.int8, copy=False)
        group_sizes = train_df["query_group_id"].astype(str).value_counts(sort=False)
        sample_weight = (1.0 / train_df["query_group_id"].astype(str).map(group_sizes).astype(float)).to_numpy(
            dtype=np.float32
        )
        model = _build_classic_classifier(
            spec["best_params"],
            monotone_constraints=monotone_constraints,
            n_jobs=n_jobs,
        )
        started = perf_counter()
        if train_df["query_group_id"].astype(str).isin(identities.query_group_ids).any() or (
            "base_group_id" in train_df and train_df["base_group_id"].astype(str).isin(identities.base_group_ids).any()
        ):
            raise ValueError("Training rows overlap complete holdout identities before fitting")
        model.fit(train_matrix, train_labels, sample_weight=sample_weight)
        train_seconds = float(perf_counter() - started)
        training_summary = {
            "rows": int(len(train_df)),
            "queries": int(train_df["query_group_id"].astype(str).nunique()),
            "positive_rows": int(train_df["label"].sum()),
            "gate_bucket_query_counts": training_gate_bucket_summary["query_counts"],
            "gate_bucket_row_counts": training_gate_bucket_summary["row_counts"],
            "elapsed_seconds": train_seconds,
            "retrieval_top_k": retrieval_top_k,
            "rows_removed_above_retrieval_top_k": (
                train_rows_before_retrieval_window - train_rows_after_retrieval_window
            ),
            "unlabeled_singleton_orcid_filter_summary": unlabeled_singleton_filter_summary,
            "train_holdout_filter_summary": train_holdout_filter_summary,
            "train_filter_summary": train_filter_summary,
        }

        training_summary["ranker_refit"] = True

    def predict(matrix: Any) -> np.ndarray:
        if isinstance(model, Booster):
            return np.asarray(model.predict(matrix, num_threads=resolve_n_jobs(n_jobs)), dtype=np.float64)
        return np.asarray(model.predict_proba(matrix), dtype=np.float64)[:, 1]

    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    promoted_gate_summary: dict[str, Any] | None = None
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    if spec.get("stratified_eval_test_split") is None:
        raise ValueError("classic.promoted_stratified_gate requires classic.stratified_eval_test_split")

    if frozen_task_rows is None:
        calibration_splits = tuple(str(split) for split in promoted_gate_config["calibration_splits"])
        fit_splits = promoted_gate_config.get("fit_splits")
        validation_splits = promoted_gate_config.get("validation_splits")
        stratified_scores = _score_classic_stratified_rows(
            bundle,
            spec,
            dict(spec["stratified_eval_test_split"]),
            predict,
            feature_columns,
            splits=calibration_splits,
            include_calibration_source=True,
            retrieval_top_k=retrieval_top_k,
        )
    else:
        calibration_splits = ("fit_pool", "gate_thresh")
        fit_splits = ("fit_pool",)
        validation_splits = ("gate_thresh",)
        stratified_scores = _score_frozen_gate_rows(frozen_task_rows, gate_training_blocks, predict, feature_columns)
    calibration_split_label = "+".join(calibration_splits)
    selected_gate_result = _fit_promoted_logistic_gate(
        stratified_scores.rows,
        stratified_scores.choices,
        stratified_scores.probabilities,
        calibration_splits=calibration_splits,
        fit_splits=fit_splits,
        validation_splits=validation_splits,
        error_weights={
            **LOGISTIC_GATE_ERROR_WEIGHTS,
            "false_abstain": promoted_gate_config["false_abstain_weight"],
        }
        if "false_abstain_weight" in promoted_gate_config
        else None,
    )
    logistic_gate_config = dict(selected_gate_result["gate_config"])
    fa_grid = promoted_gate_config.get("tune_false_abstain_weights")
    if frozen_task_rows is not None and fa_grid is None:
        fa_grid = _FALSE_ABSTAIN_REPORT_WEIGHTS
    weight_report: dict[str, Any] | None = None
    if fa_grid is not None and validation_splits is not None:
        val_splits = tuple(str(s) for s in validation_splits)
        val_choice_mask = stratified_scores.choices["split"].astype(str).isin(val_splits).to_numpy(dtype=bool)
        val_qids = set(stratified_scores.choices.loc[val_choice_mask, "query_case_id"].astype(str))
        val_row_mask = stratified_scores.rows["query_group_id"].astype(str).isin(val_qids).to_numpy(dtype=bool)
        weight_report = report_logistic_gate_error_weights(
            logistic_gate_config,
            stratified_scores.rows.loc[val_row_mask].reset_index(drop=True),
            stratified_scores.choices.loc[val_choice_mask].reset_index(drop=True),
            stratified_scores.probabilities[val_row_mask],
            false_abstain_weights=fa_grid,
        )
    calibration_metrics = {
        "split": calibration_split_label,
        **dict(selected_gate_result["calibration_metrics"]),
    }
    calibration_predictions = selected_gate_result["predictions"][
        stratified_scores.choices["split"].astype(str).isin(calibration_splits)
    ].copy()
    single_candidate_predictions = calibration_predictions[
        (pd.to_numeric(calibration_predictions["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
        | calibration_predictions["score_margin"].isna()
    ].copy()
    single_candidate_calibration_metrics = {
        **_summarize_predictions(single_candidate_predictions),
    }
    promoted_gate_summary = {
        "mode": _PROMOTED_LOGISTIC_GATE_MODE,
        "calibration_splits": list(calibration_splits),
        "test_split": str(promoted_gate_config["test_split"]),
        "error_weights": dict(logistic_gate_config["error_weights"]),
        "selected_gate": {
            "model_type": str(logistic_gate_config["model_type"]),
            "feature_count": int(len(logistic_gate_config["feature_names"])),
            "calibration_mode": str(logistic_gate_config["calibration_mode"]),
            "training_summary": dict(logistic_gate_config.get("training_summary", {})),
        },
        "calibration_metrics": dict(selected_gate_result["calibration_metrics"]),
        "validation_metrics": (
            dict(selected_gate_result["validation_metrics"])
            if selected_gate_result.get("validation_metrics") is not None
            else None
        ),
        "false_abstain_weight_report": weight_report,
        "calibration_split_metrics": {
            split: metrics
            for split, metrics in dict(selected_gate_result["split_metrics"]).items()
            if split in set(calibration_splits)
        },
        "selected_feature_importance": list(selected_gate_result["selected_feature_importance"]),
    }

    internal_eval_summary = (
        _classic_internal_eval(
            bundle, spec, predict, feature_columns, retrieval_top_k, load_logistic_gate_config(logistic_gate_config)
        )
        if frozen_task_rows is None
        else dict(selected_gate_result["validation_metrics"])
    )
    return CalibratedClassicModel(
        model=model,
        gate_config=logistic_gate_config,
        retrieval_top_k=retrieval_top_k,
        training_summary=training_summary,
        abstain_rule_summary={
            "calibration_mode": str(logistic_gate_config["calibration_mode"]),
            "promoted_logistic_gate": promoted_gate_summary,
            "logistic_gate_config": logistic_gate_config,
            "logistic_gate_feature_count": int(len(logistic_gate_config["feature_names"])),
            "calibration_retrieval_limit": calibration_limit,
            "calibration_metrics": calibration_metrics,
            "single_candidate_calibration_metrics": single_candidate_calibration_metrics,
            "internal_eval_metrics": internal_eval_summary,
        },
    )
