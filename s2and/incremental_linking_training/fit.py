"""Fit or reuse the linker ranker and attach its cutoff gate."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier

from s2and.incremental_linking.contracts import DEFAULT_RETRIEVAL_TOP_K

from .gate import _summarize_training_gate_buckets
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
    gate_config: dict[str, Any],
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
    holdout_identities: ClassicHoldoutIdentities | None = None,
    pre_materialization_holdout_summary: Mapping[str, Any] | None = None,
    frozen_task_rows: Mapping[str, pd.DataFrame] | None = None,
    fixed_ranker: Booster | None = None,
) -> CalibratedClassicModel:
    """Fit or reuse the booster without reading frozen test labels or features.

    Args:
        bundle: Training and calibration feature tables.
        gate_config: Cutoff gate to package with the ranker.
        n_jobs: Number of classifier workers.
        holdout_identities: Complete source identity authority. Direct callers
            may omit this when the bundle retains complete holdout tables.
        pre_materialization_holdout_summary: Earlier exclusion counts to report.
            When supplied, any remaining overlap is a materialization failure.
        frozen_task_rows: Optional featureized fit_pool windows.
        fixed_ranker: Verified source booster to reuse without fitting.

    Returns:
        Booster, cutoff gate, and training diagnostics.
    """

    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    retrieval_top_k = int(spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    if retrieval_top_k <= 0:
        raise ValueError("classic.retrieval_top_k must be positive")
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
        identities = holdout_identities
        if identities is None:
            identities = (
                read_classic_holdout_identities(bundle)
                if frozen_task_rows is None
                else ClassicHoldoutIdentities(frozenset(), frozenset(), ())
            )
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

    return CalibratedClassicModel(
        model=model,
        gate_config=gate_config,
        retrieval_top_k=retrieval_top_k,
        training_summary=training_summary,
        abstain_rule_summary={"logistic_gate_config": gate_config},
    )
