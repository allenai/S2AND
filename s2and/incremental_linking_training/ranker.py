"""Build the linker ranker and score its candidate choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import Booster, LGBMClassifier

from s2and.consts import MAX_FAST_PLAN_LEAVES
from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.linker_pairwise import promoted_pairwise_aggregate_columns
from s2and.thread_config import resolve_n_jobs

from .gate import (
    _classic_gate_first_name_bucket,
)

DEFAULT_CLASSIC_N_JOBS = 20


UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE = "unlabeled_singleton_orcid"


@dataclass(frozen=True)
class CalibratedClassicModel:
    """Ranker, calibrated gate, and pre-evaluation summaries."""

    model: LGBMClassifier | Booster
    gate_config: dict[str, Any]
    retrieval_top_k: int
    training_summary: dict[str, Any]
    abstain_rule_summary: dict[str, Any]


CalibratedClassicModel.__module__ = "s2and.incremental_linking_training.classic"


def _drop_unlabeled_singleton_orcid_rows(
    rows: pd.DataFrame,
    *,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop weak singleton-ORCID supervision rows from linker label tables."""

    query_count = int(rows["query_group_id"].astype(str).nunique()) if "query_group_id" in rows.columns else None
    summary: dict[str, Any] = {
        "context": context,
        "policy": f"drop_supervision_type:{UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE}",
        "rows_before": int(len(rows)),
        "rows_removed": 0,
        "rows_after": int(len(rows)),
        "queries_before": query_count,
        "queries_removed": 0,
        "queries_after": query_count,
        "positive_rows_removed": 0,
        "negative_rows_removed": 0,
    }
    if "supervision_type" not in rows.columns:
        return rows.copy(), summary

    drop_mask = rows["supervision_type"].astype(str).eq(UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE)
    if not bool(drop_mask.any()):
        return rows.copy(), summary

    labels = (
        pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(np.int8) if "label" in rows.columns else None
    )
    query_ids_before = set(rows["query_group_id"].astype(str)) if "query_group_id" in rows.columns else set()
    cleaned = rows.loc[~drop_mask].reset_index(drop=True)
    query_ids_after = set(cleaned["query_group_id"].astype(str)) if "query_group_id" in cleaned.columns else set()
    summary["rows_removed"] = int(drop_mask.sum())
    summary["rows_after"] = int(len(cleaned))
    summary["queries_removed"] = int(len(query_ids_before - query_ids_after))
    summary["queries_after"] = int(len(query_ids_after)) if "query_group_id" in rows.columns else None
    if labels is not None:
        summary["positive_rows_removed"] = int((drop_mask & (labels == 1)).sum())
        summary["negative_rows_removed"] = int((drop_mask & (labels == 0)).sum())
    return cleaned, summary


def _filter_candidate_rows_to_retrieval_top_k(
    rows: pd.DataFrame,
    *,
    retrieval_top_k: int,
    context: str,
) -> pd.DataFrame:
    """Return the exact candidate window available to the runtime linker."""

    if retrieval_top_k <= 0:
        raise ValueError(f"{context} retrieval_top_k must be positive")
    if "retrieval_rank" not in rows.columns:
        raise ValueError(f"{context} rows must contain retrieval_rank")
    ranks = pd.to_numeric(rows["retrieval_rank"], errors="coerce")
    invalid = ranks.isna() | ~np.isfinite(ranks) | ranks.lt(1) | ranks.mod(1).ne(0)
    if bool(invalid.any()):
        sample = rows.loc[invalid, "retrieval_rank"].head(5).tolist()
        raise ValueError(
            f"{context} retrieval_rank must contain positive integers: "
            f"invalid_rows={int(invalid.sum())}, sample={sample}"
        )
    selected = rows.loc[ranks.le(retrieval_top_k)].copy()
    selected["retrieval_rank"] = ranks.loc[selected.index].astype(np.int64)
    return selected.reset_index(drop=True)


def _validate_classic_feature_inputs(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    """Require active linker features to already be materialized."""

    missing_required = [str(column) for column in feature_columns if str(column) not in df.columns]
    if missing_required:
        raise ValueError(
            f"Classic feature matrix is missing required feature inputs: missing_features={missing_required}"
        )


def _coerce_classic_feature_matrix(features: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Coerce the final classic feature matrix while preserving numeric NaNs."""

    out = features.loc[:, list(feature_columns)].copy()
    non_numeric_cells: dict[str, int] = {}
    for column in feature_columns:
        raw_values = out[column]
        coerced = pd.to_numeric(raw_values, errors="coerce")
        non_numeric = coerced.isna() & raw_values.notna()
        if non_numeric.any():
            non_numeric_cells[str(column)] = int(non_numeric.sum())
        out[column] = coerced
    if non_numeric_cells:
        raise ValueError(f"Classic feature matrix contains non-numeric feature values: {non_numeric_cells}")
    infinite_cells = {
        str(column): int(np.isinf(out[column].to_numpy(dtype=np.float64, copy=False)).sum())
        for column in feature_columns
        if np.isinf(out[column].to_numpy(dtype=np.float64, copy=False)).any()
    }
    if infinite_cells:
        raise ValueError(f"Classic feature matrix contains infinite feature values: {infinite_cells}")
    return out.astype(np.float32)


def _resolve_classic_monotone_constraints(
    spec: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> list[int] | None:
    """Resolve the active classic monotone constraints from explicit spec config."""

    configured = spec.get("monotone_constraints")
    if configured is None:
        return None
    if not isinstance(configured, list):
        raise ValueError("classic.monotone_constraints must be a list when provided")
    if len(configured) != len(feature_columns):
        raise ValueError(
            "classic.monotone_constraints length must match classic.feature_columns "
            f"({len(configured)} != {len(feature_columns)})"
        )
    constraints = [int(value) for value in configured]
    invalid = [value for value in constraints if value not in {-1, 0, 1}]
    if invalid:
        raise ValueError(f"classic.monotone_constraints values must be in {{-1, 0, 1}}: {invalid}")
    return constraints if any(value != 0 for value in constraints) else None


def _build_classic_classifier(
    params: dict[str, Any],
    *,
    monotone_constraints: list[int] | None = None,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
) -> LGBMClassifier:
    classifier_params = {key: value for key, value in params.items()}
    num_leaves = classifier_params.get("num_leaves", 31)
    # Larger trees lose the native fast scoring plan, which makes every linker request slow.
    if isinstance(num_leaves, bool) or not isinstance(num_leaves, int) or not 2 <= num_leaves <= MAX_FAST_PLAN_LEAVES:
        raise ValueError(
            f"Classic linker num_leaves must be an integer from 2 to {MAX_FAST_PLAN_LEAVES}, got {num_leaves!r}"
        )
    if monotone_constraints is not None:
        classifier_params["monotone_constraints"] = list(monotone_constraints)
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        random_state=13,
        data_random_seed=13,
        feature_fraction_seed=13,
        verbosity=-1,
        n_jobs=resolve_n_jobs(n_jobs),
        class_weight=None,
        **classifier_params,
    )


def _classic_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Build a classic feature frame from already materialized promoted columns."""

    _validate_classic_feature_inputs(df, feature_columns)
    return _coerce_classic_feature_matrix(df, feature_columns)


def _apply_classic_train_row_cap(
    train_df: pd.DataFrame,
    *,
    rule_name: str | None,
    min_train_limit: int | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply an optional per-query classic training row cap."""

    if rule_name is None:
        return train_df.copy(), None
    if rule_name != "max_of_min_limit_and_first_positive_rank":
        raise ValueError(f"Unsupported classic train row cap rule: {rule_name}")
    if min_train_limit is None:
        raise ValueError("classic train row cap rule requires train_row_cap_min_limit")

    train_df = train_df.copy()
    train_df["_row_cap_query_group_key"] = train_df["query_group_id"].astype(str)
    positive_ranks = (
        train_df.loc[train_df["label"] == 1]
        .groupby("_row_cap_query_group_key", sort=False)["retrieval_rank"]
        .min()
        .rename("first_positive_rank")
    )
    query_caps = train_df[["query_group_id", "_row_cap_query_group_key"]].drop_duplicates().copy()
    query_caps["first_positive_rank"] = (
        query_caps["_row_cap_query_group_key"].map(positive_ranks.to_dict()).astype("float64")
    )
    query_caps["row_cap"] = (
        query_caps["first_positive_rank"].fillna(float(min_train_limit)).clip(lower=float(min_train_limit))
    )
    cap_map = query_caps.set_index("_row_cap_query_group_key")["row_cap"].to_dict()
    selected = train_df[
        train_df["retrieval_rank"] <= train_df["_row_cap_query_group_key"].map(cap_map).astype(float)
    ].copy()
    selected = selected.drop(columns=["_row_cap_query_group_key"])

    positive_queries_before = set(train_df.loc[train_df["label"] == 1, "query_group_id"].astype(str))
    positive_queries_after = set(selected.loc[selected["label"] == 1, "query_group_id"].astype(str))
    retained_beyond_min = int((query_caps["row_cap"] > float(min_train_limit)).sum())
    positive_rows_before = int(train_df["label"].sum())
    positive_rows_after = int(selected["label"].sum())
    return selected, {
        "rule_name": rule_name,
        "min_train_limit": int(min_train_limit),
        "train_rows_before": int(len(train_df)),
        "train_rows_after": int(len(selected)),
        "positive_rows_before": positive_rows_before,
        "positive_rows_after": positive_rows_after,
        "positive_rows_retained_pct": (
            float(positive_rows_after / positive_rows_before) if positive_rows_before else None
        ),
        "queries_before": int(train_df["query_group_id"].astype(str).nunique()),
        "queries_after": int(selected["query_group_id"].astype(str).nunique()),
        "positive_queries_before": int(len(positive_queries_before)),
        "positive_queries_after": int(len(positive_queries_after)),
        "lost_positive_queries": int(len(positive_queries_before - positive_queries_after)),
        "queries_with_row_cap_above_min": retained_beyond_min,
        "queries_with_row_cap_equal_min": int(len(query_caps) - retained_beyond_min),
    }


def _score_query_choices(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    query_id_column: str,
    include_margin: bool,
    bucket_column: str | None = None,
) -> pd.DataFrame:
    keep_columns = [query_id_column, "dataset", "query_view", "candidate_component_key", "retrieval_rank", "label"]
    for optional_column in ("query_author", "query_first_token"):
        if optional_column in df.columns:
            keep_columns.append(optional_column)
    if "supervision_type" in df.columns:
        keep_columns.append("supervision_type")
    if "base_group_id" in df.columns:
        keep_columns.append("base_group_id")
    frozen_columns = ("candidate_gold_scored", "gold_safe_target", "query_signature_id")
    keep_columns.extend(column for column in frozen_columns if column in df)
    if bucket_column:
        keep_columns.append(bucket_column)
    output_columns = [
        "query_case_id",
        "dataset",
        "query_view",
        "query_safe_target",
        "retrieved_window_safe_target",
        "query_safe_target_source",
        "chosen_candidate_target",
        "chosen_probability",
        "chosen_candidate_component_key",
        "predicted_action",
        "correct",
        "first_name_bucket",
    ]
    if "supervision_type" in df.columns:
        output_columns.append("supervision_type")
    if "base_group_id" in df.columns:
        output_columns.append("base_group_id")
    output_columns.extend(
        "chosen_candidate_gold_scored" if column == "candidate_gold_scored" else column
        for column in frozen_columns
        if column in df
    )
    if bucket_column:
        output_columns.append("review_bucket")
    if include_margin:
        output_columns.extend(
            [
                "second_probability",
                "score_margin",
                "has_runner_up",
                "candidate_kind",
                "top1_correct",
            ]
        )
    scored = df[keep_columns].copy()
    scored["candidate_probability"] = np.asarray(probabilities, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for query_id, group in scored.groupby(query_id_column, sort=False):
        ranked = group.sort_values(
            by=["candidate_probability", "retrieval_rank", "candidate_component_key"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        chosen = ranked.iloc[0]
        second_probability = float(ranked.iloc[1]["candidate_probability"]) if len(ranked) > 1 else np.nan
        retrieved_window_safe_target = int(group["label"].max())
        row = {
            "query_case_id": str(query_id),
            "dataset": str(chosen["dataset"]),
            "query_view": str(chosen["query_view"]),
            "query_safe_target": retrieved_window_safe_target,
            "retrieved_window_safe_target": retrieved_window_safe_target,
            "query_safe_target_source": "retrieved_window",
            "chosen_candidate_target": int(chosen["label"]),
            "chosen_probability": float(chosen["candidate_probability"]),
            "chosen_candidate_component_key": str(chosen["candidate_component_key"]),
            "predicted_action": "abstain",
            "correct": 0,
            "first_name_bucket": _classic_gate_first_name_bucket(chosen),
        }
        if "supervision_type" in group.columns:
            row["supervision_type"] = str(chosen["supervision_type"])
        if "base_group_id" in group.columns:
            row["base_group_id"] = str(chosen["base_group_id"])
        for column in frozen_columns:
            if column in group:
                key = "chosen_candidate_gold_scored" if column == "candidate_gold_scored" else column
                row[key] = chosen[column]
        if bucket_column:
            row["review_bucket"] = str(chosen[bucket_column])
        if include_margin:
            row["second_probability"] = second_probability if pd.notna(second_probability) else None
            row["score_margin"] = (
                float(chosen["candidate_probability"]) - float(second_probability)
                if pd.notna(second_probability)
                else None
            )
            row["has_runner_up"] = int(len(ranked) > 1)
            row["candidate_kind"] = "multi_candidate" if len(ranked) > 1 else "single_candidate"
            row["top1_correct"] = int(chosen["label"])
        rows.append(row)
    output = pd.DataFrame(rows, columns=pd.Index(output_columns))
    if include_margin:
        for column in ("second_probability", "score_margin"):
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(np.float64)
    return output


PROMOTED_PAIRWISE_COLUMNS = promoted_pairwise_aggregate_columns()


PROMOTED_NON_PAIRWISE_COLUMNS = tuple(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)


SUPPORTED_PROMOTED_FEATURE_COLUMNS = frozenset(PROMOTED_NON_PAIRWISE_COLUMNS) | frozenset(PROMOTED_PAIRWISE_COLUMNS)
