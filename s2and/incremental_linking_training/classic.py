"""Classic train/calibrate/eval helpers for promoted incremental linker training."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from s2and.incremental_linking.contracts import DEFAULT_RETRIEVAL_TOP_K
from s2and.incremental_linking.features import PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view, normalize_bucket_letters
from s2and.incremental_linking.linker_pairwise import promoted_pairwise_aggregate_columns
from s2and.incremental_linking.logistic_gate import (
    LOGISTIC_GATE_ERROR_WEIGHTS,
    LOGISTIC_GATE_FINAL_C,
    LOGISTIC_GATE_SELECTION_C,
    LOGISTIC_GATE_SELECTION_FEATURE_COUNT,
    build_logistic_gate_matrix,
    default_logistic_gate_feature_names,
    feature_values_from_candidate_frame,
    load_logistic_gate_config,
    logistic_gate_config,
    outcome_labels,
)
from s2and.thread_config import resolve_n_jobs

DEFAULT_CLASSIC_N_JOBS = 20
UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE = "unlabeled_singleton_orcid"

# Shared training, calibration, and evaluation helpers for the official replay
# target. CLI entrypoints should import this module instead of owning copies.


@dataclass(frozen=True)
class OfficialBundle:
    """Single-file metadata and paths for the official stack."""

    root: Path
    bundle_name: str
    assets: dict[str, Any]
    models: dict[str, Any]
    expected_metrics: dict[str, Any]


@dataclass(frozen=True)
class ClassicStratifiedEvalScores:
    """Loaded stratified eval rows, model probabilities, and scored choices."""

    rows: pd.DataFrame
    probabilities: np.ndarray
    choices: pd.DataFrame
    assignments: pd.DataFrame


@dataclass(frozen=True)
class FittedClassicRun:
    """Exact booster and gate evaluated by one classic pipeline run."""

    summary: dict[str, Any]
    model: LGBMClassifier
    gate_config: dict[str, Any]
    retrieval_top_k: int
    query_predictions: dict[str, Any]


_GATE_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
    "single_candidate|multi_letter_first",
    "single_candidate|single_letter_first",
)
_MISSING_LOGISTIC_CLASS_LOGIT = -30.0
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
CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET = {
    "a_khan": "a_khan_eval",
    "a_silva": "a_silva_eval",
    "h_wang": "hwang_eval",
    "j_smith": "j_smith_eval",
    "s_gupta": "s_gupta_eval",
    "s_lee": "s_lee_eval",
    "s_park": "s_park_eval",
}


def load_bundle(root: Path) -> OfficialBundle:
    """Load bundle metadata from an explicit bundle root."""

    root = root.resolve()
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    return OfficialBundle(
        root=root,
        bundle_name=str(payload["bundle_name"]),
        assets=dict(payload["assets"]),
        models=dict(payload["models"]),
        expected_metrics=dict(payload["expected_metrics"]),
    )


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if path.suffix == ".parquet":
        parquet_kwargs: dict[str, Any] = {}
        if "usecols" in kwargs:
            parquet_kwargs["columns"] = kwargs["usecols"]
        return pd.read_parquet(path, **parquet_kwargs)
    defaults = {"low_memory": False}
    defaults.update(kwargs)
    read_csv = cast(Any, pd.read_csv)
    return read_csv(path, **defaults)


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


def _resolve_path(bundle: OfficialBundle, path_like: str | Path) -> Path:
    """Resolve a stored bundle path relative to the bundle root."""

    path = Path(path_like)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            resolved.relative_to(bundle.root.resolve())
        except ValueError as exc:
            raise ValueError(f"Bundle asset path escapes bundle root: {path_like}") from exc
    else:
        resolved = (bundle.root / path).resolve()
        try:
            resolved.relative_to(bundle.root.resolve())
        except ValueError as exc:
            raise ValueError(f"Bundle asset path escapes bundle root: {path_like}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"Bundle asset does not exist: {resolved}")
    return resolved


def _normalize_dataset_slug(value: Any) -> str:
    """Normalize a dataset-like identifier into a stable lowercase slug."""

    normalized = re.sub(r"[^0-9a-z_]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError(f"Cannot derive dataset slug from {value!r}")
    return normalized


def _summary_key_for_eval_dataset(dataset_name: Any) -> str:
    """Return the runtime summary key for one eval dataset."""

    return f"overall_{_normalize_dataset_slug(dataset_name)}_eval"


def _iter_extra_eval_paths(spec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return all optional eval datasets configured for the classic bundle."""

    eval_paths: list[tuple[str, str]] = []
    seen_datasets: set[str] = set()
    for path_key, dataset_name in (
        ("s_park_eval_path", "s_park"),
        ("s_lee_eval_path", "s_lee"),
    ):
        if path_key not in spec:
            continue
        normalized_dataset = _normalize_dataset_slug(dataset_name)
        eval_paths.append((normalized_dataset, str(spec[path_key])))
        seen_datasets.add(normalized_dataset)

    extra_eval_paths = spec.get("extra_eval_paths", {})
    if extra_eval_paths is None:
        return tuple(eval_paths)
    if not isinstance(extra_eval_paths, dict):
        raise ValueError("classic.extra_eval_paths must be a mapping of dataset slug to relative path")
    for dataset_name, path_like in extra_eval_paths.items():
        normalized_dataset = _normalize_dataset_slug(dataset_name)
        if normalized_dataset in seen_datasets:
            raise ValueError(f"Duplicate extra eval dataset configured: {normalized_dataset}")
        eval_paths.append((normalized_dataset, str(path_like)))
        seen_datasets.add(normalized_dataset)
    return tuple(eval_paths)


def _iter_classic_train_holdout_paths(spec: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return calibration/eval row files whose identities must stay out of classic training."""

    holdout_paths: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for path_key, source_name in (
        ("classic_gate_source_path", "classic_gate_source"),
        ("s2and_eval_path", "s2and"),
        ("hwang_eval_path", "hwang"),
    ):
        path_like = spec.get(path_key)
        if path_like is None:
            continue
        normalized_path = str(path_like)
        if normalized_path in seen_paths:
            continue
        holdout_paths.append((source_name, normalized_path))
        seen_paths.add(normalized_path)

    for source_name, path_like in _iter_extra_eval_paths(spec):
        normalized_path = str(path_like)
        if normalized_path in seen_paths:
            continue
        holdout_paths.append((source_name, normalized_path))
        seen_paths.add(normalized_path)
    return tuple(holdout_paths)


def _nonempty_string_values(series: pd.Series) -> set[str]:
    """Return nonempty string values from one identity column."""

    values: set[str] = set()
    for value in series.dropna():
        text = str(value).strip()
        if text:
            values.add(text)
    return values


def _read_classic_holdout_identity_sets(
    bundle: OfficialBundle,
    spec: dict[str, Any],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Load query/base identity sets from calibration and eval row files."""

    query_group_ids: set[str] = set()
    base_group_ids: set[str] = set()
    source_summaries: list[dict[str, Any]] = []
    for source_name, path_like in _iter_classic_train_holdout_paths(spec):
        path = _resolve_path(bundle, path_like)
        header = _read_csv(path, nrows=0).columns
        identity_columns = [column for column in ("query_group_id", "base_group_id") if column in header]
        if not identity_columns:
            source_summaries.append(
                {
                    "source": source_name,
                    "path": str(path.relative_to(bundle.root)),
                    "query_groups": 0,
                    "base_groups": 0,
                }
            )
            continue
        rows = _read_csv(path, usecols=identity_columns)
        source_query_group_ids = (
            _nonempty_string_values(rows["query_group_id"]) if "query_group_id" in rows.columns else set()
        )
        source_base_group_ids = (
            _nonempty_string_values(rows["base_group_id"]) if "base_group_id" in rows.columns else set()
        )
        query_group_ids.update(source_query_group_ids)
        base_group_ids.update(source_base_group_ids)
        source_summaries.append(
            {
                "source": source_name,
                "path": str(path.relative_to(bundle.root)),
                "query_groups": int(len(source_query_group_ids)),
                "base_groups": int(len(source_base_group_ids)),
            }
        )
    return query_group_ids, base_group_ids, source_summaries


def _apply_classic_train_holdout_filter(
    train_df: pd.DataFrame,
    *,
    holdout_query_group_ids: set[str],
    holdout_base_group_ids: set[str],
    holdout_sources: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop training rows whose query/base identities appear in calibration or eval."""

    rows_before = int(len(train_df))
    queries_before = int(train_df["query_group_id"].astype(str).nunique()) if "query_group_id" in train_df else 0
    labels = pd.to_numeric(train_df["label"], errors="coerce").fillna(0) if "label" in train_df else pd.Series()
    positive_rows_before = int(labels.sum()) if not labels.empty else 0
    positive_queries_before = (
        int(train_df.loc[labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in train_df and not labels.empty
        else 0
    )
    query_overlap_mask = (
        train_df["query_group_id"].astype(str).isin(holdout_query_group_ids)
        if "query_group_id" in train_df and holdout_query_group_ids
        else pd.Series(False, index=train_df.index)
    )
    base_overlap_mask = (
        train_df["base_group_id"].astype(str).isin(holdout_base_group_ids)
        if "base_group_id" in train_df and holdout_base_group_ids
        else pd.Series(False, index=train_df.index)
    )
    remove_mask = query_overlap_mask | base_overlap_mask
    removed = train_df[remove_mask].copy()
    filtered = train_df[~remove_mask].copy()

    removed_labels = pd.to_numeric(removed["label"], errors="coerce").fillna(0) if "label" in removed else pd.Series()
    filtered_labels = (
        pd.to_numeric(filtered["label"], errors="coerce").fillna(0) if "label" in filtered else pd.Series()
    )
    summary = {
        "rows_before": rows_before,
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(removed)),
        "queries_before": queries_before,
        "queries_after": int(filtered["query_group_id"].astype(str).nunique()) if "query_group_id" in filtered else 0,
        "queries_removed": int(removed["query_group_id"].astype(str).nunique()) if "query_group_id" in removed else 0,
        "positive_rows_before": positive_rows_before,
        "positive_rows_after": int(filtered_labels.sum()) if not filtered_labels.empty else 0,
        "positive_rows_removed": int(removed_labels.sum()) if not removed_labels.empty else 0,
        "positive_queries_before": positive_queries_before,
        "positive_queries_after": int(filtered.loc[filtered_labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in filtered and not filtered_labels.empty
        else 0,
        "positive_queries_removed": int(removed.loc[removed_labels == 1, "query_group_id"].astype(str).nunique())
        if "query_group_id" in removed and not removed_labels.empty
        else 0,
        "overlapping_query_groups": int(train_df.loc[query_overlap_mask, "query_group_id"].astype(str).nunique())
        if "query_group_id" in train_df
        else 0,
        "overlapping_base_groups": int(train_df.loc[base_overlap_mask, "base_group_id"].astype(str).nunique())
        if "base_group_id" in train_df
        else 0,
        "holdout_query_groups": int(len(holdout_query_group_ids)),
        "holdout_base_groups": int(len(holdout_base_group_ids)),
        "holdout_sources": list(holdout_sources or []),
    }
    return filtered, summary


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a scalar-like value is pandas-missing."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool | np.bool_):
        return bool(missing)
    return False


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


def _fit_multiclass_logistic_gate(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """Fit the sklearn calibration model and return the filled raw matrix."""

    raw_matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(raw_matrix)
    medians = np.zeros(raw_matrix.shape[1], dtype=np.float64)
    for column_index in np.flatnonzero(finite.any(axis=0)):
        medians[column_index] = np.median(raw_matrix[finite[:, column_index], column_index])
    filled = raw_matrix.copy()
    invalid = ~np.isfinite(filled)
    if np.any(invalid):
        row_indices, col_indices = np.nonzero(invalid)
        filled[row_indices, col_indices] = medians[col_indices]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filled)
    model = LogisticRegression(
        C=float(c_value),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=4000,
        random_state=20260517,
    )
    model.fit(scaled, labels)
    return model, scaler, medians


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
    selection_feature_count: int = LOGISTIC_GATE_SELECTION_FEATURE_COUNT,
    selection_c: float = LOGISTIC_GATE_SELECTION_C,
    final_c: float = LOGISTIC_GATE_FINAL_C,
) -> dict[str, Any]:
    """Fit the prod-safe NumPy logistic gate on configured calibration splits."""

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
    calibration_mask = split_values.isin(tuple(str(split) for split in calibration_splits)).to_numpy(dtype=bool)
    if not np.any(calibration_mask):
        raise ValueError(f"logistic gate requires non-empty calibration splits: {list(calibration_splits)}")
    labels = outcome_labels(choices["query_safe_target"], choices["chosen_candidate_target"])

    selection_model, _selection_scaler, _selection_medians = _fit_multiclass_logistic_gate(
        full_matrix[calibration_mask],
        labels[calibration_mask],
        c_value=float(selection_c),
    )
    selection_coef = np.asarray(selection_model.coef_, dtype=np.float64)
    importance = np.max(np.abs(selection_coef), axis=0)
    ranked_indices = np.lexsort((np.asarray(all_feature_names, dtype=object), -importance))
    selected_indices = ranked_indices[: int(selection_feature_count)]
    selected_feature_names = tuple(all_feature_names[int(index)] for index in selected_indices)
    selected_matrix = full_matrix[:, selected_indices]

    final_model, final_scaler, final_medians = _fit_multiclass_logistic_gate(
        selected_matrix[calibration_mask],
        labels[calibration_mask],
        c_value=float(final_c),
    )
    weights, bias = _fold_standard_scaler_into_logistic(final_scaler, final_model)
    gate_config = logistic_gate_config(
        feature_names=selected_feature_names,
        weights=weights,
        bias=bias,
        missing_values=final_medians,
        calibration_mode=_PROMOTED_LOGISTIC_GATE_MODE,
        error_weights=LOGISTIC_GATE_ERROR_WEIGHTS,
        training_summary={
            "calibration_splits": [str(split) for split in calibration_splits],
            "selection_feature_count": int(selection_feature_count),
            "selection_c": float(selection_c),
            "final_c": float(final_c),
            "all_feature_count": int(len(all_feature_names)),
            "selected_feature_count": int(len(selected_feature_names)),
            "calibration_queries": int(calibration_mask.sum()),
        },
    )
    gate = load_logistic_gate_config(gate_config)
    calibration_predictions = _apply_logistic_gate_predictions(
        choices[calibration_mask].copy(),
        gate.predict_link(selected_matrix[calibration_mask]),
    )
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
        "split_metrics": {
            split: _summarize_predictions(predictions) for split, predictions in split_predictions.items()
        },
        "predictions": _apply_logistic_gate_predictions(choices.copy(), gate.predict_link(selected_matrix)),
    }


def _apply_promoted_logistic_gate_to_candidate_rows(
    candidate_rows: pd.DataFrame,
    choices: pd.DataFrame,
    probabilities: np.ndarray,
    gate_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply a fitted logistic gate to raw candidate rows and scored choices."""

    gate = load_logistic_gate_config(gate_config)
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


def _apply_clean_hwang_overrides(predictions: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        override_df[["query_group_id", "manual_safe_target"]].rename(columns={"query_group_id": "query_case_id"}),
        on="query_case_id",
        how="left",
        validate="one_to_one",
    )
    manual_override = merged["manual_safe_target"].notna()
    if "query_safe_target_source" not in merged.columns:
        merged["query_safe_target_source"] = "retrieved_window"
    merged.loc[manual_override, "query_safe_target_source"] = "manual_safe_target"
    merged["query_safe_target"] = merged["manual_safe_target"].fillna(merged["query_safe_target"]).astype(int)
    merged = merged.drop(columns=["manual_safe_target"])
    merged["correct"] = (
        ((merged["predicted_action"] == "abstain") & (merged["query_safe_target"] == 0))
        | (
            (merged["predicted_action"] == "link_candidate")
            & (merged["query_safe_target"] == 1)
            & (merged["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return merged


def _evaluate_logistic_scored_windows(
    candidate_rows: pd.DataFrame,
    probabilities: np.ndarray,
    gate_config: Mapping[str, Any],
    *,
    override_df: pd.DataFrame | None = None,
    limits: tuple[int, ...] = (5, 25),
) -> dict[str, dict[str, Any]]:
    """Score and summarize retrieval windows with the fitted logistic gate."""

    results: dict[str, dict[str, Any]] = {}
    if len(probabilities) != len(candidate_rows):
        raise ValueError(f"probabilities length must match rows: {len(probabilities)} != {len(candidate_rows)}")
    for limit in sorted(set(int(limit) for limit in limits)):
        mask = (candidate_rows["retrieval_rank"] <= limit).to_numpy(dtype=bool)
        limited = candidate_rows.loc[mask].copy()
        limited_probabilities = probabilities[np.flatnonzero(mask)]
        choices = _score_query_choices(
            limited.rename(columns={"query_group_id": "query_case_id"}),
            limited_probabilities,
            query_id_column="query_case_id",
            include_margin=True,
        )
        predictions = _apply_promoted_logistic_gate_to_candidate_rows(
            limited,
            choices,
            limited_probabilities,
            gate_config,
        )
        if override_df is not None:
            predictions = _apply_clean_hwang_overrides(predictions, override_df)
        results[str(limit)] = {"overall": _summarize_predictions(predictions)}
    return results


def _evaluate_logistic_manual_holdout(
    manual_holdout: pd.DataFrame,
    probabilities: np.ndarray,
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Score and summarize manual holdout candidates with the logistic gate."""

    candidate_rows = manual_holdout.rename(columns={"binary_safe_link_target": "label"}).copy()
    if "query_group_id" not in candidate_rows and "query_case_id" in candidate_rows:
        candidate_rows["query_group_id"] = candidate_rows["query_case_id"].astype(str)
    choices = _score_query_choices(
        candidate_rows,
        probabilities,
        query_id_column="query_case_id",
        include_margin=True,
        bucket_column="review_bucket",
    )
    predictions = _apply_promoted_logistic_gate_to_candidate_rows(
        candidate_rows,
        choices,
        probabilities,
        gate_config,
    )
    return {
        "overall": _summarize_predictions(predictions),
        "by_bucket": {
            str(bucket): _summarize_predictions(group.copy())
            for bucket, group in predictions.groupby("review_bucket", sort=False)
        },
    }


def _classic_stratified_eval_source_specs(spec: dict[str, Any]) -> tuple[dict[str, str], ...]:
    """Return source files used by the promoted eval/test stratified split."""

    sources = [
        {
            "source_key": "calibration_source",
            "path": str(spec["classic_gate_source_path"]),
            "source_kind": "calibration_source",
        },
        {"source_key": "s2and_eval", "path": str(spec["s2and_eval_path"]), "source_kind": "public_test"},
        {"source_key": "hwang_eval", "path": str(spec["hwang_eval_path"]), "source_kind": "public_test"},
    ]
    for path_key, source_key in (
        ("s_park_eval_path", "s_park_eval"),
        ("s_lee_eval_path", "s_lee_eval"),
    ):
        if path_key in spec:
            sources.append(
                {
                    "source_key": source_key,
                    "path": str(spec[path_key]),
                    "source_kind": "public_test",
                }
            )
    for dataset_name, path_like in sorted(dict(spec.get("extra_eval_paths") or {}).items()):
        sources.append(
            {
                "source_key": f"{_normalize_dataset_slug(dataset_name)}_eval",
                "path": str(path_like),
                "source_kind": "public_test",
            }
        )
    return tuple(sources)


def _drop_shadowed_calibration_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop calibration rows when the active public source has the same query."""

    query_source_key = ["query_group_id", "source_key"]
    public_query_sources = rows.loc[
        ~rows["source_kind"].astype(str).eq("calibration_source"),
        query_source_key,
    ].drop_duplicates()
    if public_query_sources.empty:
        return rows

    marked = rows.merge(
        public_query_sources.assign(_has_public_source_rows=True),
        on=query_source_key,
        how="left",
    )
    shadowed = marked["source_kind"].astype(str).eq("calibration_source") & marked["_has_public_source_rows"].eq(True)
    return marked.loc[~shadowed].drop(columns=["_has_public_source_rows"]).copy()


def _validate_unique_stratified_candidate_rows(rows: pd.DataFrame) -> None:
    """Fail if one selected query/source/candidate has multiple active rows."""

    key_columns = ["query_group_id", "source_key", "candidate_component_key"]
    duplicate_mask = rows.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = rows.loc[duplicate_mask, key_columns + ["label"]].copy()
    duplicate_summary = (
        duplicate_rows.groupby(key_columns, dropna=False)
        .agg(row_count=("label", "size"), labels=("label", lambda values: sorted({str(value) for value in values})))
        .reset_index()
    )
    conflict_count = int(duplicate_summary["labels"].map(len).gt(1).sum())
    sample = duplicate_summary.head(5).to_dict(orient="records")
    raise ValueError(
        "Promoted stratified eval rows contain duplicate query/source/candidate rows: "
        f"duplicate_pairs={len(duplicate_summary)}, conflicting_pairs={conflict_count}, sample={sample}"
    )


def _validate_stratified_base_identity_split_disjointness(assignments: pd.DataFrame) -> None:
    """Fail if one base identity is assigned to more than one split.

    Query group ids are masked views (``full``, ``initial_only``, ...) of one
    base query; splitting views of the same ``base_group_id`` across splits
    leaks the same author between calibration and test. Split assignment must
    group by ``base_group_id``, mirroring the identity notion the classic
    train/holdout filter enforces.
    """

    base_group_ids = assignments["base_group_id"]
    invalid_base_group_ids = base_group_ids.isna() | base_group_ids.astype(str).str.strip().eq("")
    if bool(invalid_base_group_ids.any()):
        raise ValueError(
            "Stratified split assignments require a nonempty base_group_id for every row: "
            f"invalid_rows={int(invalid_base_group_ids.sum())}"
        )
    split_values = assignments["split"]
    invalid_splits = split_values.isna() | split_values.astype(str).str.strip().eq("")
    if bool(invalid_splits.any()):
        raise ValueError(
            "Stratified split assignments require a nonempty split for every row: "
            f"invalid_rows={int(invalid_splits.sum())}"
        )

    splits_per_base_identity = assignments.groupby("base_group_id", dropna=False)["split"].nunique()
    leaking_base_identities = splits_per_base_identity[splits_per_base_identity > 1]
    if leaking_base_identities.empty:
        return

    raise ValueError(
        "Stratified split assignments place base_group_id values in multiple splits; "
        "masked views of one base query would leak between calibration and test. "
        "Regenerate the assignments with splits assigned per base_group_id: "
        f"count={len(leaking_base_identities)}, sample={leaking_base_identities.head(5).to_dict()}"
    )


def _validate_stratified_split_assignments(assignments: pd.DataFrame) -> None:
    """Validate the identity columns needed for leakage-safe split joins."""

    required = {"query_group_id", "source_key", "split", "base_group_id"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Stratified split assignments missing required columns: {missing}")
    _validate_stratified_base_identity_split_disjointness(assignments)


def _active_stratified_label_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Return query/source metadata recomputed from active candidate labels."""

    metadata_input = rows[["query_group_id", "source_key", "candidate_component_key", "retrieval_rank", "label"]].copy()
    metadata_input["_label"] = pd.to_numeric(metadata_input["label"], errors="coerce").fillna(0).astype(int)
    metadata_input["_retrieval_rank"] = pd.to_numeric(metadata_input["retrieval_rank"], errors="coerce").fillna(
        np.iinfo(np.int32).max
    )
    grouped = metadata_input.groupby(["query_group_id", "source_key"], sort=False)
    metadata = grouped.agg(
        candidate_count=("candidate_component_key", "nunique"),
        min_retrieval_rank=("_retrieval_rank", "min"),
        max_retrieval_rank=("_retrieval_rank", "max"),
        positive_candidate_rows=("_label", "sum"),
    ).reset_index()
    min_positive_rank = (
        metadata_input.loc[metadata_input["_label"].eq(1)]
        .groupby(["query_group_id", "source_key"], sort=False)["_retrieval_rank"]
        .min()
    )
    positive_rank_frame = min_positive_rank.rename("min_positive_rank").reset_index()
    metadata = metadata.merge(positive_rank_frame, on=["query_group_id", "source_key"], how="left")
    has_positive = metadata["positive_candidate_rows"].astype(int).gt(0)
    positive_first = has_positive & metadata["min_positive_rank"].eq(metadata["min_retrieval_rank"])
    metadata["has_positive_candidate"] = has_positive
    metadata["positive_first"] = positive_first
    metadata["positive_rank_bucket"] = np.select(
        [~has_positive, positive_first],
        ["no_positive", "positive_first"],
        default="positive_not_first",
    )
    metadata["raw_has_positive_candidate"] = metadata["has_positive_candidate"]
    metadata["raw_positive_first"] = metadata["positive_first"]
    metadata["manual_safe_target"] = has_positive.astype(int)
    metadata["multiple_candidates"] = metadata["candidate_count"].astype(int).gt(1)
    metadata["min_positive_rank"] = metadata["min_positive_rank"].astype(object)
    metadata.loc[~has_positive, "min_positive_rank"] = ""
    for column in ("candidate_count", "min_retrieval_rank", "max_retrieval_rank", "positive_candidate_rows"):
        metadata[column] = metadata[column].astype(int)
    return metadata


def _refresh_stratified_metadata_from_active_labels(
    rows: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh split metadata from the selected active candidate rows."""

    metadata = _active_stratified_label_metadata(rows)
    metadata_columns = [
        "has_positive_candidate",
        "positive_first",
        "positive_rank_bucket",
        "raw_has_positive_candidate",
        "raw_positive_first",
        "manual_safe_target",
        "multiple_candidates",
        "candidate_count",
        "min_positive_rank",
        "min_retrieval_rank",
        "max_retrieval_rank",
        "positive_candidate_rows",
    ]
    refreshed_rows = rows.drop(columns=[column for column in metadata_columns if column in rows.columns]).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    refreshed_assignments = assignments.drop(
        columns=[column for column in metadata_columns if column in assignments.columns]
    ).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    if "source_stratum" in refreshed_assignments.columns and "first_name_bucket" in refreshed_assignments.columns:
        refreshed_assignments["stratum_key"] = refreshed_assignments.apply(
            lambda row: (
                f"{row['source_stratum']}|has_pos={int(bool(row['has_positive_candidate']))}|"
                f"{row['positive_rank_bucket']}|{row['first_name_bucket']}|"
                f"multi_cand={int(bool(row['multiple_candidates']))}"
            ),
            axis=1,
        )
    if "stratum_key" in refreshed_assignments.columns:
        refreshed_rows = refreshed_rows.drop(columns=["stratum_key"], errors="ignore").merge(
            refreshed_assignments[["query_group_id", "source_key", "stratum_key"]],
            on=["query_group_id", "source_key"],
            how="left",
        )
    return refreshed_rows, refreshed_assignments


def _load_classic_stratified_eval_rows(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load source rows selected by the promoted query-level stratified split."""

    assignments = _read_csv(_resolve_path(bundle, str(split_spec["assignments_path"])))
    _validate_stratified_split_assignments(assignments)
    source_frames: list[pd.DataFrame] = []
    for source_spec in _classic_stratified_eval_source_specs(spec):
        rows = _read_csv(_resolve_path(bundle, source_spec["path"]), compression="gzip")
        if str(source_spec["source_key"]) == "calibration_source":
            rows["source_key"] = (
                rows["dataset"].astype(str).map(CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET).fillna("s2and_eval")
            )
        else:
            rows["source_key"] = str(source_spec["source_key"])
        rows["source_kind"] = str(source_spec["source_kind"])
        rows, _filter_summary = _drop_unlabeled_singleton_orcid_rows(
            rows,
            context=f"stratified_eval:{source_spec['source_key']}",
        )
        source_frames.append(rows)
    all_rows = _drop_shadowed_calibration_source_rows(pd.concat(source_frames, ignore_index=True))
    if "base_group_id" not in all_rows.columns:
        raise ValueError("Stratified eval source rows must contain base_group_id")
    assignment_columns = [
        "query_group_id",
        "source_key",
        "split",
        "base_group_id",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    selected_assignment_columns = [column for column in assignment_columns if column in assignments.columns]
    rows = all_rows.merge(
        assignments[selected_assignment_columns],
        on=["query_group_id", "source_key"],
        how="inner",
        suffixes=("", "_split"),
    )
    assignment_base_column = "base_group_id_split"
    if assignment_base_column not in rows.columns:
        raise ValueError("Stratified eval join did not retain assignment base_group_id")
    source_base_ids = rows["base_group_id"]
    assignment_base_ids = rows[assignment_base_column]
    invalid_source_base_ids = source_base_ids.isna() | source_base_ids.astype(str).str.strip().eq("")
    if bool(invalid_source_base_ids.any()):
        raise ValueError(
            "Stratified eval source rows require nonempty base_group_id values: "
            f"invalid_rows={int(invalid_source_base_ids.sum())}"
        )
    mismatched_base_ids = source_base_ids.astype(str) != assignment_base_ids.astype(str)
    if bool(mismatched_base_ids.any()):
        sample = rows.loc[
            mismatched_base_ids,
            ["query_group_id", "source_key", "base_group_id", assignment_base_column],
        ].head(5)
        raise ValueError(
            "Stratified split assignment base_group_id does not match source rows: "
            f"mismatched_rows={int(mismatched_base_ids.sum())}, sample={sample.to_dict(orient='records')}"
        )
    rows = rows.drop(columns=[assignment_base_column])
    for column in selected_assignment_columns:
        if column in {"query_group_id", "source_key"}:
            continue
        assignment_column = f"{column}_split"
        if assignment_column in rows.columns:
            rows[column] = rows[assignment_column]
            rows = rows.drop(columns=[assignment_column])
    matched_assignments = rows[["query_group_id", "source_key"]].drop_duplicates()
    if len(matched_assignments) != len(assignments):
        raise ValueError(
            "Stratified split assignments did not all match source rows: "
            f"matched={len(matched_assignments)}, expected={len(assignments)}"
        )
    _validate_unique_stratified_candidate_rows(rows)
    source_keys_per_query = rows.groupby("query_group_id")["source_key"].nunique()
    cross_source_queries = source_keys_per_query[source_keys_per_query > 1]
    if not cross_source_queries.empty:
        raise ValueError(
            "Stratified eval rows contain query_group_id values present under multiple source_keys; "
            "downstream scoring groups by query_group_id only and would silently collapse them: "
            f"count={len(cross_source_queries)}, sample={cross_source_queries.head(5).to_dict()}"
        )
    rows, assignments = _refresh_stratified_metadata_from_active_labels(rows, assignments)
    return rows, assignments


def _breakdown_predictions(predictions: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Summarize predictions by one column for JSON reports."""

    return {
        str(value): _summarize_predictions(group.copy())
        for value, group in predictions.groupby(column, dropna=False, sort=True)
    }


def _score_classic_stratified_eval_test(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    model: LGBMClassifier,
    feature_columns: tuple[str, ...],
) -> ClassicStratifiedEvalScores:
    """Load and score the promoted stratified calibration/test split once."""

    split_rows, assignments = _load_classic_stratified_eval_rows(bundle, spec, split_spec)
    probabilities = np.asarray(
        model.predict_proba(_classic_feature_matrix(split_rows, feature_columns)),
        dtype=np.float64,
    )[:, 1]
    choices = _score_query_choices(
        split_rows,
        probabilities,
        query_id_column="query_group_id",
        include_margin=True,
    )
    metadata_columns = [
        "query_group_id",
        "source_key",
        "source_kind",
        "split",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    metadata = split_rows[[column for column in metadata_columns if column in split_rows.columns]].drop_duplicates(
        "query_group_id"
    )
    metadata = metadata.rename(columns={"query_group_id": "query_case_id"})
    choices = choices.merge(metadata, on="query_case_id", how="left", suffixes=("", "_split"))
    if "first_name_bucket_split" in choices.columns:
        choices["first_name_bucket"] = choices["first_name_bucket_split"].fillna(choices["first_name_bucket"])
        choices = choices.drop(columns=["first_name_bucket_split"])
    if "query_safe_target_source" not in choices.columns:
        choices["query_safe_target_source"] = "retrieved_window"
    if "manual_safe_target" in choices.columns:
        manual_target = pd.to_numeric(choices["manual_safe_target"], errors="coerce")
        choices["manual_safe_target_matches_active_label"] = manual_target.isna() | manual_target.astype("Int64").eq(
            choices["query_safe_target"].astype("Int64")
        )
    return ClassicStratifiedEvalScores(
        rows=split_rows,
        probabilities=probabilities,
        choices=choices,
        assignments=assignments,
    )


def _summarize_classic_stratified_predictions(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    split_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build promoted stratified split summary and test breakdowns from predictions."""

    predictions = predictions.copy()
    predictions["gate_bucket"] = _query_gate_bucket(predictions) if not predictions.empty else pd.Series(dtype="string")
    split_order = tuple(
        str(value)
        for value in split_spec.get(
            "split_order",
            ("calibration_fit", "calibration_check", "test"),
        )
    )
    overall_by_split = {
        split: _summarize_predictions(predictions[predictions["split"] == split].copy()) for split in split_order
    }
    test_predictions = predictions[predictions["split"] == str(split_spec.get("test_split", "test"))].copy()
    factor_columns = [
        "gate_bucket",
        "source_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
    ]
    return {
        "assignment_query_counts": {
            str(split): int(count) for split, count in assignments["split"].value_counts().sort_index().items()
        },
        "scored_query_counts": {
            str(split): int(count) for split, count in predictions["split"].value_counts().sort_index().items()
        },
        "overall": overall_by_split,
        "gate_bucket_split_counts": _gate_bucket_split_counts(predictions, split_order),
        "test_breakdowns": {
            column: _breakdown_predictions(test_predictions, column)
            for column in factor_columns
            if column in test_predictions.columns
        },
    }


def _write_query_predictions(predictions: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Persist the exact query-level decisions in a deterministic CSV."""

    required = {
        "base_group_id",
        "chosen_candidate_component_key",
        "chosen_probability",
        "correct",
        "predicted_action",
        "query_case_id",
        "query_safe_target",
        "source_key",
        "split",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Linker query predictions are missing required columns: {missing}")
    frame = predictions.copy()
    identity_columns = ["source_key", "query_case_id"]
    if frame.duplicated(identity_columns).any():
        raise ValueError("Linker query predictions contain duplicate source/query identities")
    frame["label"] = pd.to_numeric(frame["query_safe_target"], errors="raise").astype(int)
    leading_columns = [*identity_columns, "base_group_id", "split", "label"]
    remaining_columns = sorted(set(frame.columns) - set(leading_columns))
    frame = frame[[*leading_columns, *remaining_columns]].sort_values(
        ["split", "source_key", "base_group_id", "query_case_id"],
        kind="mergesort",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")
    with path.open("rb") as input_file:
        digest = hashlib.file_digest(input_file, "sha256").hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def _format_metric_float(value: Any) -> str:
    """Format a metric value compactly for markdown tables."""

    return f"{float(value):.4f}"


def _metric_balanced_accuracy_cell(metrics: dict[str, Any]) -> str:
    """Return a balanced-accuracy table cell, suppressing single-class slices."""

    positive_queries = metrics.get("n_positive_queries")
    negative_queries = metrics.get("n_negative_queries")
    if positive_queries is not None and negative_queries is not None:
        if int(positive_queries) == 0 or int(negative_queries) == 0:
            return "n/a"
    return _format_metric_float(metrics["balanced_accuracy"])


def _metric_count_cell(metrics: dict[str, Any], key: str) -> str:
    """Return a query-count cell for optional metric count fields."""

    value = metrics.get(key)
    if value is None:
        return "n/a"
    return str(int(value))


def _metric_breakdown_row(label: str, metrics: dict[str, Any]) -> list[str]:
    """Return one selected-gate breakdown table row."""

    return [
        str(label),
        str(int(metrics.get("n_queries", metrics.get("queries", 0)))),
        _metric_count_cell(metrics, "n_positive_queries"),
        _metric_count_cell(metrics, "n_negative_queries"),
        _metric_balanced_accuracy_cell(metrics),
        _format_metric_float(metrics["error_rate"]),
        str(int(metrics.get("false_abstain", 0))),
        str(int(metrics.get("false_link", 0))),
        str(int(metrics.get("wrong_candidate_link", 0))),
    ]


def _metric_factor_row(factor: str, group: str, metrics: dict[str, Any]) -> list[str]:
    """Return one requested factor breakdown table row."""

    return [
        str(factor),
        str(group),
        str(int(metrics.get("n_queries", metrics.get("queries", 0)))),
        _metric_count_cell(metrics, "n_positive_queries"),
        _metric_count_cell(metrics, "n_negative_queries"),
        _format_metric_float(metrics["error_rate"]),
        str(int(metrics.get("false_abstain", 0))),
        str(int(metrics.get("false_link", 0))),
        str(int(metrics.get("wrong_candidate_link", 0))),
    ]


def _validated_metric_dict(metrics: Any, *, context: str) -> dict[str, Any]:
    """Return one metric breakdown mapping or reject a malformed summary payload."""

    if not isinstance(metrics, dict):
        raise TypeError(f"{context} metrics must be a dict, got {type(metrics).__name__}")
    return cast(dict[str, Any], metrics)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render a markdown table from string cells."""

    def cell(value: str) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return lines


def format_classic_selected_gate_tables(summary: dict[str, Any]) -> str:
    """Return the selected-gate stratified-test breakdown tables."""

    split_summary = summary.get("stratified_eval_test_split")
    if not isinstance(split_summary, dict):
        return ""
    breakdowns = split_summary.get("test_breakdowns")
    if not isinstance(breakdowns, dict):
        return ""

    lines: list[str] = []
    lines.extend(["## By Dataset Slice, Selected Gate", ""])
    source_breakdown = dict(breakdowns.get("source_key", {}))
    source_rows = [
        _metric_breakdown_row(
            str(slice_name),
            _validated_metric_dict(metrics, context=f"source_key[{slice_name!r}]"),
        )
        for slice_name, metrics in sorted(source_breakdown.items(), key=lambda item: str(item[0]))
    ]
    lines.extend(
        _markdown_table(
            [
                "slice",
                "queries",
                "positive queries",
                "negative queries",
                "BA",
                "error rate",
                "false abstain",
                "false link",
                "wrong link",
            ],
            source_rows,
        )
    )
    lines.extend(["", "BA is n/a for single-class slices.", ""])

    lines.extend(["## Requested Factor Breakdowns", ""])
    factor_rows: list[list[str]] = []
    for factor in (
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "source_stratum",
    ):
        factor_breakdown = breakdowns.get(factor)
        if not isinstance(factor_breakdown, dict):
            continue
        for group_name, metrics in sorted(factor_breakdown.items(), key=lambda item: str(item[0])):
            factor_rows.append(
                _metric_factor_row(
                    factor,
                    str(group_name),
                    _validated_metric_dict(metrics, context=f"{factor}[{group_name!r}]"),
                )
            )
    lines.extend(
        _markdown_table(
            [
                "factor",
                "group",
                "queries",
                "positive queries",
                "negative queries",
                "error rate",
                "false abstain",
                "false link",
                "wrong link",
            ],
            factor_rows,
        )
    )
    return "\n".join(lines) + "\n"


def _score_eval_candidate_rows(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_margin: bool,
    limits: tuple[int, ...] = (5, 25),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if len(probabilities) != len(df):
        raise ValueError(f"probabilities length must match df rows: {len(probabilities)} != {len(df)}")
    for limit in sorted(set(int(limit) for limit in limits)):
        mask = (df["retrieval_rank"] <= limit).to_numpy(dtype=bool)
        limited = df.loc[mask].copy()
        limited_probabilities = probabilities[np.flatnonzero(mask)]
        choices = _score_query_choices(
            limited.rename(columns={"query_group_id": "query_case_id"}),
            limited_probabilities,
            query_id_column="query_case_id",
            include_margin=include_margin,
        )
        choices["retrieval_rank_limit"] = limit
        frames.append(choices)
    return pd.concat(frames, ignore_index=True)


def run_classic(
    bundle: OfficialBundle,
    output_dir: Path,
    *,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
) -> FittedClassicRun:
    """Fit, calibrate, and evaluate the official classic pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    retrieval_top_k = int(spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    if retrieval_top_k <= 0:
        raise ValueError("classic.retrieval_top_k must be positive")
    monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
    train_df = _read_csv(_resolve_path(bundle, spec["train_path"]), compression="gzip")
    train_df, unlabeled_singleton_filter_summary = _drop_unlabeled_singleton_orcid_rows(
        train_df,
        context="classic_train",
    )
    train_df["retrieval_rank"] = pd.to_numeric(train_df["retrieval_rank"], errors="coerce")
    train_df = train_df[train_df["retrieval_rank"] <= retrieval_top_k].copy()
    train_df["label"] = pd.to_numeric(train_df["label"], errors="coerce").fillna(0).astype(np.int8)
    holdout_query_group_ids, holdout_base_group_ids, holdout_sources = _read_classic_holdout_identity_sets(
        bundle,
        spec,
    )
    train_df, train_holdout_filter_summary = _apply_classic_train_holdout_filter(
        train_df,
        holdout_query_group_ids=holdout_query_group_ids,
        holdout_base_group_ids=holdout_base_group_ids,
        holdout_sources=holdout_sources,
    )
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
    model.fit(train_matrix, train_labels, sample_weight=sample_weight)
    train_seconds = float(perf_counter() - started)

    gate_source_path = _resolve_path(bundle, spec.get("classic_gate_source_path", spec["hwang_eval_path"]))
    gate_source_eval = _read_csv(gate_source_path, compression="gzip")
    gate_source_eval, _gate_source_filter_summary = _drop_unlabeled_singleton_orcid_rows(
        gate_source_eval,
        context="classic_gate_source",
    )
    gate_source_probabilities = np.asarray(
        model.predict_proba(_classic_feature_matrix(gate_source_eval, feature_columns)),
        dtype=np.float64,
    )[:, 1]
    calibration_limit = int(spec.get("classic_gate_calibration_retrieval_limit", 50))
    gate_source_query_choices = _score_eval_candidate_rows(
        gate_source_eval,
        gate_source_probabilities,
        include_margin=True,
        limits=(calibration_limit,),
    )

    internal_eval_groups = set(
        _read_csv(_resolve_path(bundle, spec["classic_gate_internal_eval_base_groups_path"]))["base_group_id"].astype(
            str
        )
    )
    internal_eval_rows = gate_source_query_choices[
        (gate_source_query_choices["retrieval_rank_limit"] == calibration_limit)
        & (gate_source_query_choices["base_group_id"].isin(internal_eval_groups))
    ].copy()
    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    promoted_gate_summary: dict[str, Any] | None = None
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    if spec.get("stratified_eval_test_split") is None:
        raise ValueError("classic.promoted_stratified_gate requires classic.stratified_eval_test_split")

    split_spec = dict(spec["stratified_eval_test_split"])
    stratified_scores = _score_classic_stratified_eval_test(
        bundle,
        spec,
        split_spec,
        model,
        feature_columns,
    )
    calibration_splits = tuple(str(split) for split in promoted_gate_config["calibration_splits"])
    calibration_split_label = "+".join(calibration_splits)
    selected_gate_result = _fit_promoted_logistic_gate(
        stratified_scores.rows,
        stratified_scores.choices,
        stratified_scores.probabilities,
        calibration_splits=calibration_splits,
    )
    logistic_gate_config = dict(selected_gate_result["gate_config"])
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
        "error_weights": dict(LOGISTIC_GATE_ERROR_WEIGHTS),
        "selected_gate": {
            "model_type": str(logistic_gate_config["model_type"]),
            "feature_count": int(len(logistic_gate_config["feature_names"])),
            "calibration_mode": str(logistic_gate_config["calibration_mode"]),
            "training_summary": dict(logistic_gate_config.get("training_summary", {})),
        },
        "calibration_metrics": dict(selected_gate_result["calibration_metrics"]),
        "calibration_split_metrics": {
            split: metrics
            for split, metrics in dict(selected_gate_result["split_metrics"]).items()
            if split in set(calibration_splits)
        },
        "selected_feature_importance": list(selected_gate_result["selected_feature_importance"]),
    }

    s2and_eval = _read_csv(_resolve_path(bundle, spec["s2and_eval_path"]), compression="gzip")
    s2and_probabilities = np.asarray(
        model.predict_proba(_classic_feature_matrix(s2and_eval, feature_columns)),
        dtype=np.float64,
    )[:, 1]
    s2and_eval_summary = _evaluate_logistic_scored_windows(
        s2and_eval,
        s2and_probabilities,
        logistic_gate_config,
    )
    hwang_eval = _read_csv(_resolve_path(bundle, spec["hwang_eval_path"]), compression="gzip")
    hwang_probabilities = np.asarray(
        model.predict_proba(_classic_feature_matrix(hwang_eval, feature_columns)),
        dtype=np.float64,
    )[:, 1]
    optional_eval_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, path_like in _iter_extra_eval_paths(spec):
        eval_df = _read_csv(_resolve_path(bundle, path_like), compression="gzip")
        eval_probabilities = np.asarray(
            model.predict_proba(_classic_feature_matrix(eval_df, feature_columns)),
            dtype=np.float64,
        )[:, 1]
        optional_eval_summaries[_summary_key_for_eval_dataset(dataset_name)] = _evaluate_logistic_scored_windows(
            eval_df,
            eval_probabilities,
            logistic_gate_config,
        )

    override_path = spec.get("hwang_clean_override_path")
    override_df = _read_csv(_resolve_path(bundle, str(override_path))) if override_path else None
    hwang_eval_summary = _evaluate_logistic_scored_windows(
        hwang_eval,
        hwang_probabilities,
        logistic_gate_config,
        override_df=override_df,
    )
    internal_eval_source_rows = gate_source_eval[
        gate_source_eval["base_group_id"].astype(str).isin(internal_eval_groups)
        & (pd.to_numeric(gate_source_eval["retrieval_rank"], errors="coerce") <= calibration_limit)
    ].copy()
    internal_eval_source_probabilities = gate_source_probabilities[
        internal_eval_source_rows.index.to_numpy(dtype=np.int64)
    ]
    internal_eval_choices = internal_eval_rows.drop(columns=["retrieval_rank_limit"], errors="ignore")
    internal_eval_summary = _summarize_predictions(
        _apply_promoted_logistic_gate_to_candidate_rows(
            internal_eval_source_rows,
            internal_eval_choices,
            internal_eval_source_probabilities,
            logistic_gate_config,
        )
    )
    stratified_eval_test_summary = _summarize_classic_stratified_predictions(
        selected_gate_result["predictions"],
        stratified_scores.assignments,
        split_spec,
    )

    hwang_cleaned_eval = {
        f"w{limit}": {
            "cleaned_balanced_accuracy": window_summary["overall"]["balanced_accuracy"],
            "cleaned_positive_recall": window_summary["overall"]["positive_recall"],
            "cleaned_negative_recall": window_summary["overall"]["negative_recall"],
        }
        for limit, window_summary in hwang_eval_summary.items()
    }

    summary = {
        "model": "classic",
        "training_summary": {
            "rows": int(len(train_df)),
            "queries": int(train_df["query_group_id"].astype(str).nunique()),
            "positive_rows": int(train_df["label"].sum()),
            "gate_bucket_query_counts": training_gate_bucket_summary["query_counts"],
            "gate_bucket_row_counts": training_gate_bucket_summary["row_counts"],
            "elapsed_seconds": train_seconds,
            "unlabeled_singleton_orcid_filter_summary": unlabeled_singleton_filter_summary,
            "train_holdout_filter_summary": train_holdout_filter_summary,
            "train_filter_summary": train_filter_summary,
        },
        "abstain_rule": {
            "calibration_mode": str(logistic_gate_config["calibration_mode"]),
            "promoted_logistic_gate": promoted_gate_summary,
            "logistic_gate_config": logistic_gate_config,
            "logistic_gate_feature_count": int(len(logistic_gate_config["feature_names"])),
            "calibration_retrieval_limit": calibration_limit,
            "calibration_metrics": calibration_metrics,
            "single_candidate_calibration_metrics": single_candidate_calibration_metrics,
            "internal_eval_metrics": internal_eval_summary,
        },
        "overall_s2and_eval": s2and_eval_summary,
        "hwang_cleaned_eval": hwang_cleaned_eval,
    }
    if stratified_eval_test_summary is not None:
        summary["stratified_eval_test_split"] = stratified_eval_test_summary
    manual_holdout_path = spec.get("manual_holdout_candidates_path")
    if manual_holdout_path:
        manual_holdout = _read_csv(_resolve_path(bundle, manual_holdout_path), low_memory=False)
        manual_probabilities = np.asarray(
            model.predict_proba(_classic_feature_matrix(manual_holdout, feature_columns)),
            dtype=np.float64,
        )[:, 1]
        summary["manual_holdout"] = _evaluate_logistic_manual_holdout(
            manual_holdout,
            manual_probabilities,
            logistic_gate_config,
        )
    for summary_key, eval_summary in optional_eval_summaries.items():
        summary[summary_key] = eval_summary
    selected_gate_tables = format_classic_selected_gate_tables(summary)
    if selected_gate_tables:
        selected_gate_tables_path = output_dir / "selected_gate_tables.md"
        selected_gate_tables_path.write_text(selected_gate_tables, encoding="utf-8")
        summary["selected_gate_tables_path"] = str(selected_gate_tables_path.relative_to(output_dir))
    query_predictions = _write_query_predictions(
        selected_gate_result["predictions"],
        output_dir / "query_predictions.csv",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return FittedClassicRun(
        summary=summary,
        model=model,
        gate_config=logistic_gate_config,
        retrieval_top_k=retrieval_top_k,
        query_predictions=query_predictions,
    )


PROMOTED_PAIRWISE_COLUMNS = promoted_pairwise_aggregate_columns()
PROMOTED_NON_PAIRWISE_COLUMNS = tuple(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
SUPPORTED_PROMOTED_FEATURE_COLUMNS = frozenset(PROMOTED_NON_PAIRWISE_COLUMNS) | frozenset(PROMOTED_PAIRWISE_COLUMNS)
WEIGHTED_ERROR_WEIGHTS = {
    "false_abstain_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["false_abstain"]),
    "false_link_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["false_link"]),
    "wrong_link_error_rate": float(LOGISTIC_GATE_ERROR_WEIGHTS["wrong_candidate_link"]),
}
NAN_POLICY_CHOICES = ("preserve", "zero")
ROW_NAN_POLICY_CHOICES = ("finite", "semantic")
