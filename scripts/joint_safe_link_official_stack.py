"""Shared loaders, trainers, calibrators, and evaluators for the official stack."""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.contracts import INCREMENTAL_LINKING_RUST_CAPABILITIES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = REPO_ROOT / "s2and" / "data" / "joint_safe_link_featureless_self_contained_20260506a"
CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE = 192.0
_ANCHOR_EVIDENCE_FEATURE_COLUMNS = (
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
)
_ANCHOR_EVIDENCE_PREREQUISITES = (
    "min_distance",
    "specter_exemplar_similarity",
    "title_overlap",
    "coauthor_overlap",
    "affiliation_overlap",
    "venue_overlap",
    "year_compatibility",
    "retrieval_score_gap_vs_best_competitor",
    "candidate_contradiction_score",
    "same_family_as_top1",
    "candidate_pair_share_within_coarse_family",
    "cluster_size",
    "named_signature_count",
    "retrieval_rank",
)
_CLASSIC_DERIVABLE_FEATURE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "cluster_size_log_capped": ("cluster_size",),
    "query_first_prefix_match": ("dominant_first_name",),
    **{feature: _ANCHOR_EVIDENCE_PREREQUISITES for feature in _ANCHOR_EVIDENCE_FEATURE_COLUMNS},
    "query_view__full": ("query_view",),
    "query_view__initial_only": ("query_view",),
}

for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


@dataclass(frozen=True)
class OfficialBundle:
    """Single-file metadata and paths for the official stack."""

    root: Path
    bundle_name: str
    assets: dict[str, Any]
    models: dict[str, Any]
    expected_metrics: dict[str, Any]


@dataclass(frozen=True)
class TotalErrorGateSpec:
    """Bucketed score/margin gate selected by total query errors."""

    name: str
    score_thresholds: dict[str, float]
    margin_thresholds: dict[str, float]
    lambda_penalty: float | None


_TOTAL_ERROR_SCORE_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
    "single_candidate|multi_letter_first",
    "single_candidate|single_letter_first",
)
_TOTAL_ERROR_MARGIN_BUCKETS = (
    "multi_candidate|multi_letter_first",
    "multi_candidate|single_letter_first",
)
_DEFAULT_TOTAL_ERROR_LAMBDA_GRID = (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS = {
    "false_abstain": 0.25,
    "false_link": 1.0,
    "wrong_candidate_link": 1.5,
}
CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET = {
    "a_khan": "a_khan_eval",
    "a_silva": "a_silva_eval",
    "h_wang": "hwang_eval",
    "j_smith": "j_smith_eval",
    "s_gupta": "s_gupta_eval",
    "s_lee": "s_lee_eval",
    "s_park": "s_park_eval",
}


def load_bundle(root: Path = DEFAULT_PACKAGE_DIR) -> OfficialBundle:
    """Load the official bundle metadata from the single bundle file."""

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


def _bounded_threshold_grid(values: np.ndarray, grid_size: int) -> np.ndarray:
    """Build a bounded quantile grid with inclusive edge thresholds."""

    cleaned = np.asarray(values, dtype=np.float64)
    if cleaned.size == 0:
        return np.array([0.0], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, num=max(int(grid_size), 2), dtype=np.float64)
    thresholds = np.unique(np.quantile(cleaned, quantiles))
    epsilon = 1e-12
    return np.unique(
        np.concatenate(
            (
                np.array([float(cleaned.min()) - epsilon], dtype=np.float64),
                thresholds.astype(np.float64, copy=False),
                np.array([float(cleaned.max()) + epsilon], dtype=np.float64),
            )
        )
    )


def _normalize_letters(value: Any) -> str:
    """Normalize a name-like token down to lowercase letters."""

    return re.sub(r"[^a-z]", "", str(value).lower())


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


def _normalize_optional_letters(value: Any) -> str:
    """Normalize a name token, treating missing values as absent."""

    if _is_missing_scalar(value):
        return ""
    return _normalize_letters(value)


def _cluster_size_log_capped(cluster_size: Any) -> float:
    """Return a capped log-size prior anchored to the train p95 component size."""

    size = max(0.0, float(cluster_size or 0.0))
    if size <= 0.0:
        return 0.0
    reference = float(CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE)
    return float(min(1.0, math.log1p(size) / math.log1p(reference)))


def _numeric_feature_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    """Return one numeric feature column with a deterministic default for formula derivation."""

    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(np.float32)


def _derive_anchor_evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive anchor-evidence formulas from existing candidate-row evidence columns."""

    out = df.copy()
    min_distance = _numeric_feature_series(out, "min_distance", default=10000.0)
    specter = _numeric_feature_series(out, "specter_exemplar_similarity")
    title = _numeric_feature_series(out, "title_overlap")
    coauthor = _numeric_feature_series(out, "coauthor_overlap")
    affiliation = _numeric_feature_series(out, "affiliation_overlap")
    venue = _numeric_feature_series(out, "venue_overlap")
    year = _numeric_feature_series(out, "year_compatibility")
    retrieval_gap = _numeric_feature_series(out, "retrieval_score_gap_vs_best_competitor")
    contradiction = _numeric_feature_series(out, "candidate_contradiction_score")
    same_top1 = _numeric_feature_series(out, "same_family_as_top1")
    candidate_pair_share = _numeric_feature_series(out, "candidate_pair_share_within_coarse_family")
    cluster_size = _numeric_feature_series(out, "cluster_size")
    named_signature_count = _numeric_feature_series(out, "named_signature_count")
    retrieval_rank = _numeric_feature_series(out, "retrieval_rank", default=99.0)

    min_distance_values = min_distance.to_numpy(dtype=np.float32, copy=False)
    specter_values = specter.to_numpy(dtype=np.float32, copy=False)
    title_values = title.to_numpy(dtype=np.float32, copy=False)
    coauthor_values = coauthor.to_numpy(dtype=np.float32, copy=False)
    affiliation_values = affiliation.to_numpy(dtype=np.float32, copy=False)
    venue_values = venue.to_numpy(dtype=np.float32, copy=False)
    year_values = year.to_numpy(dtype=np.float32, copy=False)
    retrieval_gap_values = retrieval_gap.to_numpy(dtype=np.float32, copy=False)
    contradiction_values = contradiction.to_numpy(dtype=np.float32, copy=False)
    same_top1_values = same_top1.to_numpy(dtype=np.float32, copy=False)
    candidate_pair_share_values = candidate_pair_share.to_numpy(dtype=np.float32, copy=False)
    cluster_size_values = cluster_size.to_numpy(dtype=np.float32, copy=False)
    named_signature_count_values = named_signature_count.to_numpy(dtype=np.float32, copy=False)
    retrieval_rank_values = retrieval_rank.to_numpy(dtype=np.float32, copy=False)

    min_distance_clip = np.clip(min_distance_values, 0.0, 1.0)
    specter_clip = np.clip(specter_values, 0.0, 1.0)
    title_clip = np.clip(title_values, 0.0, 1.0)
    coauthor_clip = np.clip(coauthor_values, 0.0, 1.0)
    affiliation_clip = np.clip(affiliation_values, 0.0, 1.0)
    venue_clip = np.clip(venue_values, 0.0, 1.0)
    year_clip = np.clip(year_values, 0.0, 1.0)
    same_top1_clip = np.clip(same_top1_values, 0.0, 1.0)
    retrieval_gap_positive = np.clip(retrieval_gap_values, 0.0, 0.3) / 0.3
    retrieval_gap_normalized = np.clip((np.clip(retrieval_gap_values, -0.2, 0.3) + 0.2) / 0.5, 0.0, 1.0)
    candidate_pair_share_clip = np.clip(candidate_pair_share_values, 0.0, 1.0)

    out["anchor_evidence_count"] = (
        (min_distance_values <= 0.15).astype(np.float32)
        + (specter_values >= 0.70).astype(np.float32)
        + (title_values >= 0.20).astype(np.float32)
        + (coauthor_values >= 0.25).astype(np.float32)
        + (affiliation_values >= 0.25).astype(np.float32)
        + (venue_values >= 0.20).astype(np.float32)
        + (year_values >= 0.90).astype(np.float32)
        + (retrieval_gap_values >= 0.02).astype(np.float32)
    ).astype(np.float32)

    support_strength = (
        0.20 * (1.0 - min_distance_clip)
        + 0.20 * specter_clip
        + 0.18 * title_clip
        + 0.18 * coauthor_clip
        + 0.12 * affiliation_clip
        + 0.06 * venue_clip
        + 0.06 * year_clip
    )
    low_contradiction_multiplier = np.clip(1.0 - np.clip(contradiction_values, 0.0, 1.0), 0.0, 1.0)
    out["strong_positive_anchor_score"] = (
        np.clip(support_strength, 0.0, 1.0)
        * (0.5 + 0.5 * same_top1_clip)
        * (0.35 + 0.65 * low_contradiction_multiplier)
    ).astype(np.float32)

    tiny_candidate = ((cluster_size_values <= 2.0) | (named_signature_count_values <= 2.0)).astype(np.float32)
    residual_support = (
        0.28 * (1.0 - min_distance_clip)
        + 0.20 * specter_clip
        + 0.20 * coauthor_clip
        + 0.14 * title_clip
        + 0.10 * year_clip
        + 0.08 * retrieval_gap_normalized
    )
    out["weak_residual_anchor_score"] = (tiny_candidate * same_top1_clip * np.clip(residual_support, 0.0, 1.0)).astype(
        np.float32
    )

    out["sparse_relative_winner_score"] = (
        (retrieval_rank_values <= 1.0).astype(np.float32)
        * same_top1_clip
        * np.clip(retrieval_gap_positive, 0.0, 1.0)
        * (1.0 - candidate_pair_share_clip)
        * np.clip(residual_support, 0.0, 1.0)
    ).astype(np.float32)
    return out


def _query_first_token(author: Any) -> str:
    """Return the first alphabetic token from a query author string."""

    if _is_missing_scalar(author):
        return ""
    tokens = re.findall(r"[A-Za-z]+", str(author))
    return tokens[0].lower() if tokens else ""


def _classic_gate_first_name_bucket(row: pd.Series | dict[str, Any]) -> str:
    """Classify a query into the gate's first-name-length bucket."""

    token = _normalize_optional_letters(row.get("query_first_token", ""))
    query_author = row.get("query_author")
    if not token and query_author is not None and pd.notna(query_author) and str(query_author).strip():
        token = _query_first_token(row.get("query_author"))
    if not token and str(row.get("query_view", "")) == "initial_only":
        return "single_letter_first"
    return "single_letter_first" if len(token) <= 1 else "multi_letter_first"


def _fixed_bucketed_gate_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the configured fixed bucketed gate, if active."""

    configured = spec.get("fixed_bucketed_gate")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError("classic.fixed_bucketed_gate must be a mapping when provided")
    thresholds = configured.get("score_thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("classic.fixed_bucketed_gate.score_thresholds must be a non-empty mapping")
    normalized = {str(key): float(value) for key, value in thresholds.items()}
    required = {
        "multi_candidate|multi_letter_first",
        "multi_candidate|single_letter_first",
        "single_candidate|multi_letter_first",
        "single_candidate|single_letter_first",
    }
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"classic.fixed_bucketed_gate.score_thresholds missing buckets: {missing}")
    out = dict(configured)
    out["score_thresholds"] = normalized
    margin_thresholds = out.get("margin_thresholds")
    if margin_thresholds is not None:
        if not isinstance(margin_thresholds, dict):
            raise ValueError("classic.fixed_bucketed_gate.margin_thresholds must be a mapping when provided")
        normalized_margins = {str(key): float(value) for key, value in margin_thresholds.items()}
        valid_margin_buckets = {
            "multi_candidate|multi_letter_first",
            "multi_candidate|single_letter_first",
        }
        invalid_margin_buckets = sorted(set(normalized_margins) - valid_margin_buckets)
        if invalid_margin_buckets:
            raise ValueError(
                "classic.fixed_bucketed_gate.margin_thresholds contains unsupported buckets: "
                f"{invalid_margin_buckets}"
            )
        missing_margin_buckets = sorted(valid_margin_buckets - set(normalized_margins))
        if missing_margin_buckets:
            raise ValueError(
                "classic.fixed_bucketed_gate.margin_thresholds missing buckets: " f"{missing_margin_buckets}"
            )
        out["margin_thresholds"] = normalized_margins
    if "margin_threshold" in out and out["margin_threshold"] is not None:
        out["margin_threshold"] = float(out["margin_threshold"])
    return out


def _promoted_stratified_gate_spec(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Return the configured promoted stratified gate calibration, if active."""

    configured = spec.get("promoted_stratified_gate")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ValueError("classic.promoted_stratified_gate must be a mapping when provided")
    if str(configured.get("mode")) != "total_error_4score_2margin":
        raise ValueError("classic.promoted_stratified_gate.mode must be total_error_4score_2margin")
    score_thresholds = configured.get("reference_score_thresholds")
    margin_thresholds = configured.get("reference_margin_thresholds")
    if not isinstance(score_thresholds, dict) or not isinstance(margin_thresholds, dict):
        raise ValueError(
            "classic.promoted_stratified_gate requires reference_score_thresholds and reference_margin_thresholds"
        )
    missing_score_buckets = sorted(set(_TOTAL_ERROR_SCORE_BUCKETS) - set(score_thresholds))
    missing_margin_buckets = sorted(set(_TOTAL_ERROR_MARGIN_BUCKETS) - set(margin_thresholds))
    if missing_score_buckets or missing_margin_buckets:
        raise ValueError(
            "classic.promoted_stratified_gate has missing reference buckets: "
            f"score={missing_score_buckets}, margin={missing_margin_buckets}"
        )
    out = dict(configured)
    out["reference_score_thresholds"] = {
        bucket: float(score_thresholds[bucket]) for bucket in _TOTAL_ERROR_SCORE_BUCKETS
    }
    out["reference_margin_thresholds"] = {
        bucket: float(margin_thresholds[bucket]) for bucket in _TOTAL_ERROR_MARGIN_BUCKETS
    }
    out["lambda_grid"] = [float(value) for value in out.get("lambda_grid", list(_DEFAULT_TOTAL_ERROR_LAMBDA_GRID))]
    out["score_grid_size"] = int(out.get("score_grid_size", spec.get("score_grid_size", 101)))
    out["margin_grid_size"] = int(out.get("margin_grid_size", spec.get("margin_grid_size", 101)))
    out["fit_split"] = str(out.get("fit_split", "calibration_fit"))
    out["selection_split"] = str(out.get("selection_split", "calibration_check"))
    out["test_split"] = str(out.get("test_split", "test"))
    out["selection_metric"] = str(out.get("selection_metric", "weighted_average_error"))
    if out["selection_metric"] != "weighted_average_error":
        raise ValueError("classic.promoted_stratified_gate.selection_metric must be weighted_average_error")
    out["error_weights"] = dict(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS)
    return out


def _weighted_error_metrics(
    *,
    n_queries: int,
    false_abstain: int,
    false_link: int,
    wrong_candidate_link: int,
    error_weights: Mapping[str, float] = _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS,
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


def _normalize_augmented_feature_frame(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    requested_anchor_features = set(feature_columns) & set(_ANCHOR_EVIDENCE_FEATURE_COLUMNS)
    if requested_anchor_features:
        out = _derive_anchor_evidence_features(out)
    if "query_first_prefix_match" in feature_columns:
        if "dominant_first_name" in out.columns:
            if "query_author" in out.columns:
                query_first_from_author = out["query_author"].map(_query_first_token)
            else:
                query_first_from_author = pd.Series([""] * len(out), index=out.index, dtype="string")
            if "query_first_token" in out.columns:
                query_first_from_token = out["query_first_token"].map(_normalize_optional_letters)
            else:
                query_first_from_token = pd.Series([""] * len(out), index=out.index, dtype="string")
            query_first = [
                author_token if author_token else token
                for author_token, token in zip(query_first_from_author, query_first_from_token, strict=True)
            ]
            dominant_first = out["dominant_first_name"].map(_normalize_optional_letters)
            out["query_first_prefix_match"] = [
                1.0 if q and len(q) > 1 and d and (q.startswith(d) or d.startswith(q)) else 0.0
                for q, d in zip(query_first, dominant_first, strict=True)
            ]
    if "cluster_size_log_capped" in feature_columns and "cluster_size" in out.columns:
        cluster_size = pd.to_numeric(out["cluster_size"], errors="coerce")
        missing_cluster_size = int(cluster_size.isna().sum())
        if missing_cluster_size:
            raise ValueError(
                "Cannot derive cluster_size_log_capped because cluster_size has "
                f"{missing_cluster_size} missing/non-numeric rows"
            )
        out["cluster_size_log_capped"] = cluster_size.map(_cluster_size_log_capped)
    elif "cluster_size_log_capped" in feature_columns and "cluster_size_log_capped" not in out.columns:
        if "cluster_size" in out.columns:
            cluster_size = pd.to_numeric(out["cluster_size"], errors="coerce")
            missing_cluster_size = int(cluster_size.isna().sum())
            if missing_cluster_size:
                raise ValueError(
                    "Cannot derive cluster_size_log_capped because cluster_size has "
                    f"{missing_cluster_size} missing/non-numeric rows"
                )
        else:
            raise ValueError("Cannot derive cluster_size_log_capped without cluster_size")
        out["cluster_size_log_capped"] = cluster_size.map(_cluster_size_log_capped)
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    for column in feature_columns:
        if column == "query_view":
            out[column] = out[column].astype("string").fillna("missing")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _augmented_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    prepared = _normalize_augmented_feature_frame(df, feature_columns)
    numeric_columns = [column for column in feature_columns if column != "query_view"]
    features = prepared.loc[:, numeric_columns].copy().astype(np.float32)
    if "query_view" in prepared.columns:
        query_view = prepared["query_view"].astype("string").fillna("missing")
    else:
        query_view = pd.Series(["missing"] * len(prepared), index=prepared.index, dtype="string")
    features["query_view__full"] = (query_view == "full").astype(np.float32)
    features["query_view__initial_only"] = (query_view == "initial_only").astype(np.float32)
    return features


def _validate_classic_feature_inputs(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> None:
    """Require active features to be present or explicitly derivable."""

    missing_required: list[str] = []
    missing_prerequisites: dict[str, list[str]] = {}
    for column in feature_columns:
        if column in df.columns:
            continue
        prerequisites = _CLASSIC_DERIVABLE_FEATURE_PREREQUISITES.get(str(column))
        if prerequisites is None:
            missing_required.append(str(column))
            continue
        missing_for_column = [required for required in prerequisites if required not in df.columns]
        if missing_for_column:
            missing_prerequisites[str(column)] = missing_for_column
    if missing_required or missing_prerequisites:
        raise ValueError(
            "Classic feature matrix is missing required feature inputs: "
            f"missing_features={missing_required}, missing_prerequisites={missing_prerequisites}"
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


_CLASSIC_MONOTONE_CONSTRAINT_BY_FEATURE: dict[str, int] = {
    "min_distance_rank_fraction": -1,
    "affiliation_overlap": 1,
    "affiliation_contradiction_severity": -1,
    "count_normalized_confidence": 1,
    "venue_overlap": 1,
    "venue_overlap_rank_fraction": -1,
    "coauthor_overlap": 1,
    "same_family_as_top1": 0,
    "top5_gap_to_retrieval_top1": -1,
    "heuristic_prefers_top1": 0,
    "affiliation_overlap_rank_fraction": -1,
    "specter_exemplar_rank_fraction": -1,
    "exact_anchor_evidence_flag": 1,
    "year_compatibility": 1,
    "year_mismatch_severity": -1,
    "title_overlap_rank_fraction": -1,
    "middle_initial_compatibility": 1,
    "title_overlap": 1,
    "coauthor_overlap_rank_fraction": -1,
    "cluster_size": 0,
    "cluster_size_log_capped": 0,
    "min_distance": -1,
    "specter_exemplar_similarity": 1,
    "specter_centroid_similarity": 1,
    "raw_max_affiliation_jaccard": 1,
    "raw_max_coauthor_jaccard": 1,
    "raw_max_title_jaccard": 1,
    "raw_max_text_jaccard": 1,
    "top5_mean_distance": -1,
    "distance_spread_top5_minus_min": 0,
    "year_compatibility_rank_fraction": -1,
    "query_view__full": 0,
    "query_view__initial_only": 0,
    "anchor_evidence_count": 1,
    "strong_positive_anchor_score": 1,
    "weak_residual_anchor_score": 1,
    "sparse_relative_winner_score": 1,
    "first_name_count_min_rarity": 1,
    "last_first_name_count_min_rarity": 1,
    "last_name_count_min_rarity": 1,
    "last_first_initial_count_min_rarity": 1,
    "first_name_count_max_rarity": 1,
    "last_first_name_count_max_rarity": 1,
    "first_prefix_x_last_first_name_count_min_rarity": 1,
}


def _classic_monotone_constraints_for_features(feature_columns: tuple[str, ...] | list[str]) -> list[int]:
    """Return the default classic monotone constraints in feature order."""

    return [int(_CLASSIC_MONOTONE_CONSTRAINT_BY_FEATURE.get(str(column), 0)) for column in feature_columns]


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
        n_jobs=20,
        class_weight=None,
        **classifier_params,
    )


def _classic_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    """Build a classic feature frame, allowing union-style augmented features when requested."""

    _validate_classic_feature_inputs(df, feature_columns)
    needs_augmented_matrix = any(
        column in _CLASSIC_DERIVABLE_FEATURE_PREREQUISITES for column in feature_columns
    ) or any(column not in df.columns for column in feature_columns)
    if needs_augmented_matrix:
        features = _augmented_feature_matrix(df, feature_columns)
        return _coerce_classic_feature_matrix(features, feature_columns)
    out = df.copy()
    return _coerce_classic_feature_matrix(out, feature_columns)


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

    positive_ranks = (
        train_df.loc[train_df["label"] == 1]
        .groupby("query_group_id", sort=False)["retrieval_rank"]
        .min()
        .rename("first_positive_rank")
    )
    query_caps = train_df[["query_group_id"]].drop_duplicates().copy()
    query_caps["first_positive_rank"] = (
        query_caps["query_group_id"].astype(str).map(positive_ranks.to_dict()).astype("float64")
    )
    query_caps["row_cap"] = (
        query_caps["first_positive_rank"].fillna(float(min_train_limit)).clip(lower=float(min_train_limit))
    )
    cap_map = query_caps.set_index("query_group_id")["row_cap"].to_dict()
    selected = train_df[train_df["retrieval_rank"] <= train_df["query_group_id"].map(cap_map).astype(float)].copy()

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


def _select_negative_training_groups(
    top1_negative_rows: pd.DataFrame,
    *,
    filter_name: str,
) -> set[str]:
    """Select negative training groups using a named conservative filter."""

    top1 = top1_negative_rows.copy()
    for column in ["title_overlap", "coauthor_overlap", "affiliation_overlap", "count_normalized_confidence"]:
        top1[column] = pd.to_numeric(top1[column], errors="coerce").fillna(0.0)
    rules = {
        "better": (
            (top1["coauthor_overlap"] <= 0.0)
            & (top1["affiliation_overlap"] <= 0.0)
            & (top1["count_normalized_confidence"] < 0.4)
        ),
        "strict": (
            (top1["title_overlap"] <= 0.0)
            & (top1["coauthor_overlap"] <= 0.0)
            & (top1["affiliation_overlap"] <= 0.0)
            & (top1["count_normalized_confidence"] < 0.4)
        ),
        "medium": (
            (top1["title_overlap"] <= 0.05)
            & (top1["coauthor_overlap"] <= 0.0)
            & (top1["affiliation_overlap"] <= 0.1)
            & (top1["count_normalized_confidence"] < 0.5)
        ),
    }
    if filter_name not in rules:
        raise ValueError(f"Unknown negative filter: {filter_name}")
    return set(top1.loc[rules[filter_name], "train_group_id"].astype(str))


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
    scored = df[keep_columns].copy()
    scored["candidate_probability"] = probabilities.astype(np.float32)
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
    return pd.DataFrame(rows)


def _fit_score_margin_gate(
    query_choices: pd.DataFrame,
    *,
    reference_score_threshold: float,
    reference_margin_threshold: float,
    score_grid_size: int = 101,
    margin_grid_size: int = 101,
) -> dict[str, Any]:
    """Fit a classic-style 2D gate on chosen score and runner-up margin."""

    eligible = query_choices[query_choices["score_margin"].notna()].copy()
    if eligible.empty:
        raise ValueError("Cannot fit score+margin gate without runner-up margins")
    score_values = eligible["chosen_probability"].to_numpy(dtype=np.float64, copy=False)
    margin_values = eligible["score_margin"].to_numpy(dtype=np.float64, copy=False)
    score_thresholds = _bounded_threshold_grid(score_values, score_grid_size)
    margin_thresholds = _bounded_threshold_grid(margin_values, margin_grid_size)

    score_below = score_values[:, None] < score_thresholds[None, :]
    margin_below = margin_values[:, None] < margin_thresholds[None, :]
    abstain = score_below[:, :, None] & margin_below[:, None, :]
    link = ~abstain

    query_safe_target = eligible["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_candidate_target = eligible["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    positive_queries = query_safe_target == 1
    negative_queries = query_safe_target == 0
    correct_links = positive_queries & (chosen_candidate_target == 1)

    tp = np.count_nonzero(link & correct_links[:, None, None], axis=0).astype(np.int32, copy=False)
    fp = np.count_nonzero(link & ~correct_links[:, None, None], axis=0).astype(np.int32, copy=False)
    tn = np.count_nonzero(abstain & negative_queries[:, None, None], axis=0).astype(np.int32, copy=False)
    positive_errors = (abstain & positive_queries[:, None, None]) | (
        link & positive_queries[:, None, None] & (chosen_candidate_target[:, None, None] == 0)
    )
    fn = np.count_nonzero(positive_errors, axis=0).astype(np.int32, copy=False)

    n_queries = int(len(eligible))
    n_positive_queries = int(positive_queries.sum())
    n_negative_queries = int(negative_queries.sum())
    positive_recall = (
        tp.astype(np.float64) / float(n_positive_queries) if n_positive_queries else np.zeros_like(tp, dtype=np.float64)
    )
    negative_recall = (
        tn.astype(np.float64) / float(n_negative_queries) if n_negative_queries else np.zeros_like(tn, dtype=np.float64)
    )
    balanced_accuracy = (positive_recall + negative_recall) / 2.0
    link_denominator = tp + fp
    link_precision = np.divide(
        tp.astype(np.float64),
        link_denominator.astype(np.float64),
        out=np.zeros_like(tp, dtype=np.float64),
        where=link_denominator > 0,
    )
    recall_denominator = tp + fn
    link_recall = np.divide(
        tp.astype(np.float64),
        recall_denominator.astype(np.float64),
        out=np.zeros_like(tp, dtype=np.float64),
        where=recall_denominator > 0,
    )
    abstain_rate = abstain.mean(axis=0, dtype=np.float64)

    score_grid, margin_grid = np.meshgrid(score_thresholds, margin_thresholds, indexing="ij")
    ranking = np.lexsort(
        (
            np.abs(margin_grid.ravel() - float(reference_margin_threshold)),
            np.abs(score_grid.ravel() - float(reference_score_threshold)),
            -negative_recall.ravel(),
            -link_precision.ravel(),
            -balanced_accuracy.ravel(),
        )
    )
    best_index = np.unravel_index(int(ranking[0]), balanced_accuracy.shape)
    best_score_threshold = float(score_thresholds[best_index[0]])
    best_margin_threshold = float(margin_thresholds[best_index[1]])
    positive_forced_choice_accuracy = (
        float(chosen_candidate_target[positive_queries].mean()) if n_positive_queries else 0.0
    )
    metrics = {
        "n_queries": n_queries,
        "n_positive_queries": n_positive_queries,
        "n_negative_queries": n_negative_queries,
        "balanced_accuracy": float(balanced_accuracy[best_index]),
        "positive_recall": float(positive_recall[best_index]),
        "negative_recall": float(negative_recall[best_index]),
        "link_precision": float(link_precision[best_index]),
        "link_recall": float(link_recall[best_index]),
        "abstain_rate": float(abstain_rate[best_index]),
        "positive_forced_choice_accuracy": positive_forced_choice_accuracy,
        "tp": int(tp[best_index]),
        "fp": int(fp[best_index]),
        "tn": int(tn[best_index]),
        "fn": int(fn[best_index]),
    }
    return {
        "score_threshold": best_score_threshold,
        "margin_threshold": best_margin_threshold,
        "metrics": metrics,
        "sort_key": (
            metrics["balanced_accuracy"],
            metrics["link_precision"],
            metrics["negative_recall"],
            -abs(best_score_threshold - float(reference_score_threshold)),
            -abs(best_margin_threshold - float(reference_margin_threshold)),
        ),
    }


def _fit_single_candidate_score_gate(
    query_choices: pd.DataFrame,
    *,
    reference_score_threshold: float,
    score_grid_size: int = 101,
) -> dict[str, Any]:
    """Fit a score-only abstain gate for query windows without a runner-up."""

    if "has_runner_up" in query_choices.columns:
        no_runner_up = pd.to_numeric(query_choices["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0
    else:
        no_runner_up = pd.Series(False, index=query_choices.index)
    if "score_margin" in query_choices.columns:
        score_margin = pd.to_numeric(query_choices["score_margin"], errors="coerce")
    else:
        score_margin = pd.Series(np.nan, index=query_choices.index, dtype=np.float64)
    eligible = query_choices[no_runner_up | score_margin.isna()].copy()
    if eligible.empty:
        return {
            "single_candidate_score_threshold": float(reference_score_threshold),
            "metrics": {
                "n_queries": 0,
                "n_positive_queries": 0,
                "n_negative_queries": 0,
                "balanced_accuracy": 0.0,
                "positive_recall": 0.0,
                "negative_recall": 0.0,
                "link_precision": 0.0,
                "link_recall": 0.0,
                "abstain_rate": 0.0,
                "positive_forced_choice_accuracy": 0.0,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            },
        }

    score_values = eligible["chosen_probability"].to_numpy(dtype=np.float64, copy=False)
    score_thresholds = _bounded_threshold_grid(score_values, score_grid_size)
    link = score_values[:, None] >= score_thresholds[None, :]
    abstain = ~link

    query_safe_target = eligible["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_candidate_target = eligible["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    positive_queries = query_safe_target == 1
    negative_queries = query_safe_target == 0
    correct_links = positive_queries & (chosen_candidate_target == 1)

    tp = np.count_nonzero(link & correct_links[:, None], axis=0).astype(np.int32, copy=False)
    fp = np.count_nonzero(link & ~correct_links[:, None], axis=0).astype(np.int32, copy=False)
    tn = np.count_nonzero(abstain & negative_queries[:, None], axis=0).astype(np.int32, copy=False)
    positive_errors = (abstain & positive_queries[:, None]) | (
        link & positive_queries[:, None] & (chosen_candidate_target[:, None] == 0)
    )
    fn = np.count_nonzero(positive_errors, axis=0).astype(np.int32, copy=False)

    n_queries = int(len(eligible))
    n_positive_queries = int(positive_queries.sum())
    n_negative_queries = int(negative_queries.sum())
    positive_recall = (
        tp.astype(np.float64) / float(n_positive_queries) if n_positive_queries else np.zeros_like(tp, dtype=np.float64)
    )
    negative_recall = (
        tn.astype(np.float64) / float(n_negative_queries) if n_negative_queries else np.zeros_like(tn, dtype=np.float64)
    )
    balanced_accuracy = (positive_recall + negative_recall) / 2.0
    link_denominator = tp + fp
    link_precision = np.divide(
        tp.astype(np.float64),
        link_denominator.astype(np.float64),
        out=np.zeros_like(tp, dtype=np.float64),
        where=link_denominator > 0,
    )
    recall_denominator = tp + fn
    link_recall = np.divide(
        tp.astype(np.float64),
        recall_denominator.astype(np.float64),
        out=np.zeros_like(tp, dtype=np.float64),
        where=recall_denominator > 0,
    )
    abstain_rate = abstain.mean(axis=0, dtype=np.float64)
    ranking = np.lexsort(
        (
            np.abs(score_thresholds - float(reference_score_threshold)),
            -negative_recall,
            -link_precision,
            -balanced_accuracy,
        )
    )
    best_index = int(ranking[0])
    positive_forced_choice_accuracy = (
        float(chosen_candidate_target[positive_queries].mean()) if n_positive_queries else 0.0
    )
    metrics = {
        "n_queries": n_queries,
        "n_positive_queries": n_positive_queries,
        "n_negative_queries": n_negative_queries,
        "balanced_accuracy": float(balanced_accuracy[best_index]),
        "positive_recall": float(positive_recall[best_index]),
        "negative_recall": float(negative_recall[best_index]),
        "link_precision": float(link_precision[best_index]),
        "link_recall": float(link_recall[best_index]),
        "abstain_rate": float(abstain_rate[best_index]),
        "positive_forced_choice_accuracy": positive_forced_choice_accuracy,
        "tp": int(tp[best_index]),
        "fp": int(fp[best_index]),
        "tn": int(tn[best_index]),
        "fn": int(fn[best_index]),
    }
    return {
        "single_candidate_score_threshold": float(score_thresholds[best_index]),
        "metrics": metrics,
        "sort_key": (
            metrics["balanced_accuracy"],
            metrics["link_precision"],
            metrics["negative_recall"],
            -abs(float(score_thresholds[best_index]) - float(reference_score_threshold)),
        ),
    }


def _total_error_gate_bucket(rows: pd.DataFrame) -> pd.Series:
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
        "query_counts": {bucket: int(query_counts.get(bucket, 0)) for bucket in _TOTAL_ERROR_SCORE_BUCKETS},
        "row_counts": {bucket: int(training_row_counts.get(bucket, 0)) for bucket in _TOTAL_ERROR_SCORE_BUCKETS},
    }


def _gate_bucket_split_counts(predictions: pd.DataFrame, split_order: Sequence[str]) -> dict[str, dict[str, int]]:
    """Count scored query choices by promoted gate bucket and split."""

    if predictions.empty:
        return {bucket: {str(split): 0 for split in split_order} for bucket in _TOTAL_ERROR_SCORE_BUCKETS}
    counts = predictions.groupby(["gate_bucket", "split"], dropna=False, sort=False).size()
    return {
        bucket: {str(split): int(counts.get((bucket, str(split)), 0)) for split in split_order}
        for bucket in _TOTAL_ERROR_SCORE_BUCKETS
    }


def _total_error_threshold_grid(values: pd.Series, reference: float, grid_size: int) -> np.ndarray:
    """Build a quantile threshold grid that always includes the reference value."""

    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if numeric.size == 0:
        return np.array([float(reference)], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(int(grid_size), 2), dtype=np.float64)
    epsilon = 1e-12
    return np.unique(
        np.concatenate(
            (
                np.array([float(reference), float(numeric.min()) - epsilon], dtype=np.float64),
                np.quantile(numeric, quantiles).astype(np.float64, copy=False),
                np.array([float(numeric.max()) + epsilon], dtype=np.float64),
            )
        )
    )


def _total_error_count(rows: pd.DataFrame, link: np.ndarray) -> np.ndarray:
    """Count query-level errors for one or more link/abstain decisions."""

    query_target = rows["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_target = rows["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    matrix = np.asarray(link, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    correct = ((~matrix) & (query_target[:, None] == 0)) | (
        matrix & (query_target[:, None] == 1) & (chosen_target[:, None] == 1)
    )
    return (~correct).sum(axis=0)


def _weighted_error_count(
    rows: pd.DataFrame,
    link: np.ndarray,
    *,
    error_weights: Mapping[str, float] = _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS,
) -> np.ndarray:
    """Return weighted error counts for one or more link/abstain decisions."""

    query_target = rows["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_target = rows["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    matrix = np.asarray(link, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    false_abstain = ((~matrix) & (query_target[:, None] == 1)).sum(axis=0).astype(np.float64)
    false_link = (matrix & (query_target[:, None] == 0)).sum(axis=0).astype(np.float64)
    wrong_candidate_link = (
        (matrix & (query_target[:, None] == 1) & (chosen_target[:, None] == 0)).sum(axis=0).astype(np.float64)
    )
    return (
        float(error_weights["false_abstain"]) * false_abstain
        + float(error_weights["false_link"]) * false_link
        + float(error_weights["wrong_candidate_link"]) * wrong_candidate_link
    )


def _fit_total_error_single_score(
    rows: pd.DataFrame,
    *,
    reference_score: float,
    lambda_penalty: float,
    score_grid_size: int,
) -> float:
    """Fit a score-only threshold by weighted errors plus optional drift penalty."""

    if rows.empty:
        return float(reference_score)
    thresholds = _total_error_threshold_grid(rows["chosen_probability"], reference_score, score_grid_size)
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    links = score[:, None] >= thresholds[None, :]
    errors = _weighted_error_count(rows, links).astype(np.float64)
    drift = np.abs(thresholds - float(reference_score))
    objective = errors + float(lambda_penalty) * drift
    ranking = np.lexsort((drift, objective))
    return float(thresholds[int(ranking[0])])


def _fit_total_error_score_margin(
    rows: pd.DataFrame,
    *,
    reference_score: float,
    reference_margin: float,
    lambda_penalty: float,
    score_grid_size: int,
    margin_grid_size: int,
) -> tuple[float, float]:
    """Fit a score-or-margin gate by weighted errors plus optional drift penalty."""

    if rows.empty:
        return float(reference_score), float(reference_margin)
    score_grid = _total_error_threshold_grid(rows["chosen_probability"], reference_score, score_grid_size)
    margin_grid = _total_error_threshold_grid(rows["score_margin"], reference_margin, margin_grid_size)
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    margin = pd.to_numeric(rows["score_margin"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    best_key: tuple[float, float] | None = None
    best = (float(reference_score), float(reference_margin))
    for score_threshold in score_grid:
        links = (score[:, None] >= float(score_threshold)) | (margin[:, None] >= margin_grid[None, :])
        errors = _weighted_error_count(rows, links).astype(np.float64)
        drift = np.abs(float(score_threshold) - float(reference_score)) + np.abs(margin_grid - float(reference_margin))
        objective = errors + float(lambda_penalty) * drift
        best_index = int(np.lexsort((drift, objective))[0])
        key = (float(objective[best_index]), float(drift[best_index]))
        if best_key is None or key < best_key:
            best_key = key
            best = (float(score_threshold), float(margin_grid[best_index]))
    return best


def _fit_total_error_gate_candidate(
    fit_rows: pd.DataFrame,
    *,
    reference_score_thresholds: dict[str, float],
    reference_margin_thresholds: dict[str, float],
    lambda_penalty: float,
    score_grid_size: int,
    margin_grid_size: int,
) -> TotalErrorGateSpec:
    """Fit one 4-score/2-margin weighted-error gate candidate."""

    labels = _total_error_gate_bucket(fit_rows)
    score_thresholds: dict[str, float] = {}
    margin_thresholds: dict[str, float] = {}
    for bucket in _TOTAL_ERROR_SCORE_BUCKETS:
        rows = fit_rows[labels == bucket].copy()
        if bucket in _TOTAL_ERROR_MARGIN_BUCKETS:
            score_threshold, margin_threshold = _fit_total_error_score_margin(
                rows,
                reference_score=reference_score_thresholds[bucket],
                reference_margin=reference_margin_thresholds[bucket],
                lambda_penalty=lambda_penalty,
                score_grid_size=score_grid_size,
                margin_grid_size=margin_grid_size,
            )
            score_thresholds[bucket] = score_threshold
            margin_thresholds[bucket] = margin_threshold
        else:
            score_thresholds[bucket] = _fit_total_error_single_score(
                rows,
                reference_score=reference_score_thresholds[bucket],
                lambda_penalty=lambda_penalty,
                score_grid_size=score_grid_size,
            )
    return TotalErrorGateSpec(
        name=f"total_error_4score_2margin_lambda_{lambda_penalty:g}",
        score_thresholds=score_thresholds,
        margin_thresholds=margin_thresholds,
        lambda_penalty=float(lambda_penalty),
    )


def _apply_total_error_gate(rows: pd.DataFrame, gate: TotalErrorGateSpec) -> pd.DataFrame:
    """Apply a total-error 4-score/2-margin gate to scored query choices."""

    predictions = _apply_classic_gate(
        rows,
        score_threshold=float(gate.score_thresholds["multi_candidate|multi_letter_first"]),
        margin_threshold=float(gate.margin_thresholds["multi_candidate|multi_letter_first"]),
        single_candidate_score_threshold=float(gate.score_thresholds["single_candidate|single_letter_first"]),
        bucketed_score_thresholds=gate.score_thresholds,
        bucketed_margin_thresholds=gate.margin_thresholds,
    )
    predictions["gate_bucket"] = _total_error_gate_bucket(predictions)
    return predictions


def _total_error_gate_drift(
    gate: TotalErrorGateSpec,
    *,
    reference_score_thresholds: dict[str, float],
    reference_margin_thresholds: dict[str, float],
) -> float:
    """Return L1 threshold drift from the configured reference gate."""

    score_drift = sum(
        abs(float(gate.score_thresholds[bucket]) - float(reference_score_thresholds[bucket]))
        for bucket in _TOTAL_ERROR_SCORE_BUCKETS
    )
    margin_drift = sum(
        abs(float(gate.margin_thresholds[bucket]) - float(reference_margin_thresholds[bucket]))
        for bucket in _TOTAL_ERROR_MARGIN_BUCKETS
    )
    return float(score_drift + margin_drift)


def _total_error_gate_candidate_rows(
    gates: list[TotalErrorGateSpec],
    *,
    fit_rows: pd.DataFrame,
    check_rows: pd.DataFrame,
    reference_score_thresholds: dict[str, float],
    reference_margin_thresholds: dict[str, float],
) -> pd.DataFrame:
    """Return fit/check metrics for candidate weighted-error gates."""

    rows: list[dict[str, Any]] = []
    for gate in gates:
        fit_metrics = _summarize_predictions(_apply_total_error_gate(fit_rows, gate))
        check_metrics = _summarize_predictions(_apply_total_error_gate(check_rows, gate))
        rows.append(
            {
                "name": gate.name,
                "lambda_penalty": gate.lambda_penalty,
                "score_thresholds_json": json.dumps(gate.score_thresholds, sort_keys=True),
                "margin_thresholds_json": json.dumps(gate.margin_thresholds, sort_keys=True),
                "total_threshold_drift": _total_error_gate_drift(
                    gate,
                    reference_score_thresholds=reference_score_thresholds,
                    reference_margin_thresholds=reference_margin_thresholds,
                ),
                **{f"fit_{key}": value for key, value in fit_metrics.items()},
                **{f"check_{key}": value for key, value in check_metrics.items()},
            }
        )
    return pd.DataFrame(rows)


def _fit_promoted_stratified_total_error_gate(
    choices: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit and select the promoted non-fixed bucketed gate on weighted stratified errors."""

    fit_rows = choices[choices["split"] == str(gate_config["fit_split"])].copy()
    check_rows = choices[choices["split"] == str(gate_config["selection_split"])].copy()
    if fit_rows.empty or check_rows.empty:
        raise ValueError(
            "Promoted stratified gate requires non-empty fit/check splits: "
            f"fit={len(fit_rows)}, check={len(check_rows)}"
        )
    reference_score_thresholds = dict(gate_config["reference_score_thresholds"])
    reference_margin_thresholds = dict(gate_config["reference_margin_thresholds"])
    error_weights = dict(gate_config.get("error_weights", _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS))
    gates = [
        _fit_total_error_gate_candidate(
            fit_rows,
            reference_score_thresholds=reference_score_thresholds,
            reference_margin_thresholds=reference_margin_thresholds,
            lambda_penalty=float(lambda_penalty),
            score_grid_size=int(gate_config["score_grid_size"]),
            margin_grid_size=int(gate_config["margin_grid_size"]),
        )
        for lambda_penalty in gate_config["lambda_grid"]
    ]
    candidates = _total_error_gate_candidate_rows(
        gates,
        fit_rows=fit_rows,
        check_rows=check_rows,
        reference_score_thresholds=reference_score_thresholds,
        reference_margin_thresholds=reference_margin_thresholds,
    )
    ranked = candidates.sort_values(
        by=[
            "check_weighted_average_error",
            "check_wrong_candidate_link",
            "check_false_link",
            "check_false_abstain",
            "total_threshold_drift",
        ],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )
    selected_name = str(ranked.iloc[0]["name"])
    selected_gate = next(gate for gate in gates if gate.name == selected_name)
    fit_metrics = _summarize_predictions(_apply_total_error_gate(fit_rows, selected_gate))
    check_metrics = _summarize_predictions(_apply_total_error_gate(check_rows, selected_gate))
    return {
        "gate": selected_gate,
        "fit_metrics": fit_metrics,
        "check_metrics": check_metrics,
        "candidate_metrics": candidates.to_dict(orient="records"),
        "selection_key": {
            "check_weighted_average_error": float(check_metrics["weighted_average_error"]),
            "check_false_abstain_error_rate": float(check_metrics["false_abstain_error_rate"]),
            "check_false_link_error_rate": float(check_metrics["false_link_error_rate"]),
            "check_wrong_link_error_rate": float(check_metrics["wrong_link_error_rate"]),
            "error_weights": error_weights,
            "check_errors": int(check_metrics["errors"]),
            "check_wrong_candidate_link": int(check_metrics["wrong_candidate_link"]),
            "check_false_link": int(check_metrics["false_link"]),
            "check_false_abstain": int(check_metrics["false_abstain"]),
            "total_threshold_drift": _total_error_gate_drift(
                selected_gate,
                reference_score_thresholds=reference_score_thresholds,
                reference_margin_thresholds=reference_margin_thresholds,
            ),
        },
    }


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


def _score_abstain_rule(
    rows: pd.DataFrame,
    score_threshold: float,
    margin_threshold: float,
    *,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    scored_rows = _apply_classic_gate(
        rows,
        score_threshold=float(score_threshold),
        margin_threshold=float(margin_threshold),
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    scored_rows["accepted"] = scored_rows["predicted_action"] == "link_candidate"
    positives = scored_rows[scored_rows["query_safe_target"] == 1]
    negatives = scored_rows[scored_rows["query_safe_target"] == 0]
    positive_correct = int(
        (
            (scored_rows["accepted"])
            & (scored_rows["chosen_candidate_target"] == 1)
            & (scored_rows["query_safe_target"] == 1)
        ).sum()
    )
    positive_accuracy = float(positive_correct / len(positives)) if len(positives) else None
    negative_reject = int((~scored_rows["accepted"] & (scored_rows["query_safe_target"] == 0)).sum())
    negative_reject_accuracy = float(negative_reject / len(negatives)) if len(negatives) else None
    balanced_accuracy = (
        float(positive_accuracy if positive_accuracy is not None else 0.0)
        + float(negative_reject_accuracy if negative_reject_accuracy is not None else 0.0)
    ) / 2.0
    return {
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
        "single_candidate_score_threshold": (
            float(single_candidate_score_threshold) if single_candidate_score_threshold is not None else None
        ),
        "queries": int(len(rows)),
        "eligible_queries": int(len(scored_rows)),
        "runner_up_queries": int(
            (
                (pd.to_numeric(scored_rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 1)
                & scored_rows["score_margin"].notna()
            ).sum()
        ),
        "single_candidate_queries": int(
            (
                (pd.to_numeric(scored_rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
                | scored_rows["score_margin"].isna()
            ).sum()
        ),
        "positive_queries": int(len(positives)),
        "negative_queries": int(len(negatives)),
        "balanced_accuracy": float(balanced_accuracy),
        "positive_accuracy": positive_accuracy,
        "negative_reject_accuracy": negative_reject_accuracy,
        "positive_accept_rate": (
            float(scored_rows.loc[scored_rows["query_safe_target"] == 1, "accepted"].mean()) if len(positives) else None
        ),
        "rejection_rate": float((~scored_rows["accepted"]).mean()) if len(scored_rows) else 0.0,
        "false_positive_links": int(
            (
                scored_rows["accepted"]
                & ~((scored_rows["query_safe_target"] == 1) & (scored_rows["chosen_candidate_target"] == 1))
            ).sum()
        ),
    }


def _tune_classic_abstain_rule(rows: pd.DataFrame, score_grid_size: int, margin_grid_size: int) -> dict[str, Any]:
    eligible_rows = rows[(rows["has_runner_up"] == 1) & rows["score_margin"].notna()].copy()
    if eligible_rows.empty:
        return {"score_threshold": 0.0, "margin_threshold": 0.0, "balanced_accuracy": 0.0}
    score_values = eligible_rows["chosen_probability"].to_numpy(dtype=np.float64)
    margin_values = eligible_rows["score_margin"].to_numpy(dtype=np.float64)
    score_thresholds = np.unique(np.quantile(score_values, np.linspace(0.0, 1.0, int(score_grid_size))))
    margin_thresholds = np.unique(np.quantile(margin_values, np.linspace(0.0, 1.0, int(margin_grid_size))))
    epsilon = 1e-6
    score_thresholds = np.unique(
        np.concatenate(([float(score_values.min()) - epsilon], score_thresholds, [float(score_values.max()) + epsilon]))
    )
    margin_thresholds = np.unique(
        np.concatenate(
            (
                [float(margin_values.min()) - epsilon],
                margin_thresholds,
                [float(margin_values.max()) + epsilon],
            )
        )
    )
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            metrics = _score_abstain_rule(
                eligible_rows,
                score_threshold=float(score_threshold),
                margin_threshold=float(margin_threshold),
            )
            ranking_key = (
                float(metrics["balanced_accuracy"]),
                float(metrics["positive_accuracy"] if metrics["positive_accuracy"] is not None else -1.0),
                -float(metrics["rejection_rate"]),
                -float(metrics["score_threshold"]),
                -float(metrics["margin_threshold"]),
            )
            if best_key is None or ranking_key > best_key:
                best_key = ranking_key
                best_metrics = metrics
    assert best_metrics is not None
    return best_metrics


def _apply_classic_gate(
    query_choices: pd.DataFrame,
    score_threshold: float,
    margin_threshold: float,
    *,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    pred = query_choices.copy()
    score_margin = pd.to_numeric(pred["score_margin"], errors="coerce")
    if "has_runner_up" in pred.columns:
        no_runner_up = pd.to_numeric(pred["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0
    else:
        no_runner_up = pd.Series(False, index=pred.index)
    single_candidate = no_runner_up | score_margin.isna()
    if bucketed_score_thresholds is not None:
        normalized_thresholds = {str(key): float(value) for key, value in bucketed_score_thresholds.items()}
        if "first_name_bucket" in pred.columns:
            first_name_bucket = pred["first_name_bucket"].astype(str)
        else:
            first_name_bucket = pred.apply(_classic_gate_first_name_bucket, axis=1)
        candidate_kind = pd.Series(
            np.where(single_candidate, "single_candidate", "multi_candidate"),
            index=pred.index,
        )
        bucket_keys = candidate_kind + "|" + first_name_bucket
        single_threshold = (
            float(single_candidate_score_threshold)
            if single_candidate_score_threshold is not None
            else float(score_threshold)
        )
        fallback_thresholds = pd.Series(
            np.where(
                single_candidate,
                single_threshold,
                float(score_threshold),
            ),
            index=pred.index,
        )
        threshold_values = bucket_keys.map(normalized_thresholds).fillna(fallback_thresholds).astype(float)
        score_link = pd.to_numeric(pred["chosen_probability"], errors="coerce") >= threshold_values
        if bucketed_margin_thresholds is not None:
            normalized_margin_thresholds = {str(key): float(value) for key, value in bucketed_margin_thresholds.items()}
            margin_threshold_values = bucket_keys.map(normalized_margin_thresholds)
            margin_link = (
                (~single_candidate)
                & score_margin.notna()
                & margin_threshold_values.notna()
                & (score_margin >= margin_threshold_values.astype(float))
            )
        elif bucketed_margin_threshold is None:
            margin_link = pd.Series(False, index=pred.index)
        else:
            margin_link = (
                (~single_candidate) & score_margin.notna() & (score_margin >= float(bucketed_margin_threshold))
            )
        abstain = ~(score_link | margin_link)
    else:
        runner_up_abstain = (
            ~single_candidate
            & (pred["chosen_probability"] < float(score_threshold))
            & (score_margin < float(margin_threshold))
        )
        if single_candidate_score_threshold is None:
            single_candidate_abstain = pd.Series(False, index=pred.index)
        else:
            single_candidate_abstain = single_candidate & (
                pred["chosen_probability"] < float(single_candidate_score_threshold)
            )
        abstain = runner_up_abstain | single_candidate_abstain
    pred["predicted_action"] = np.where(abstain, "abstain", "link_candidate")
    pred["correct"] = (
        ((pred["predicted_action"] == "abstain") & (pred["query_safe_target"] == 0))
        | (
            (pred["predicted_action"] == "link_candidate")
            & (pred["query_safe_target"] == 1)
            & (pred["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return pred


def _evaluate_classic_manual_holdout(
    manual_holdout: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score and summarize the manual holdout for the freshly fit classic model."""

    query_choices = _score_query_choices(
        manual_holdout.rename(columns={"binary_safe_link_target": "label"}),
        probabilities,
        query_id_column="query_case_id",
        include_margin=True,
        bucket_column="review_bucket",
    )
    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=float(score_threshold),
        margin_threshold=float(margin_threshold),
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    return {
        "overall": _summarize_predictions(predictions),
        "by_bucket": {
            str(bucket): _summarize_predictions(group.copy())
            for bucket, group in predictions.groupby("review_bucket", sort=False)
        },
    }


def _evaluate_scored_windows(
    query_choices: pd.DataFrame,
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    override_df: pd.DataFrame | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for limit in sorted(query_choices["retrieval_rank_limit"].dropna().astype(int).unique()):
        limited = query_choices[query_choices["retrieval_rank_limit"] == limit].copy()
        predictions = _apply_classic_gate(
            limited,
            float(score_threshold),
            float(margin_threshold),
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )
        if override_df is not None:
            predictions = _apply_clean_hwang_overrides(predictions, override_df)
        results[str(limit)] = {"overall": _summarize_predictions(predictions)}
    return results


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
    shadowed = marked["source_kind"].astype(str).eq("calibration_source") & marked["_has_public_source_rows"].fillna(
        False
    )
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
    required_assignment_columns = {"query_group_id", "source_key", "split"}
    missing_assignment_columns = sorted(required_assignment_columns - set(assignments.columns))
    if missing_assignment_columns:
        raise ValueError(f"Stratified split assignments missing required columns: {missing_assignment_columns}")
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
        source_frames.append(rows)
    all_rows = _drop_shadowed_calibration_source_rows(pd.concat(source_frames, ignore_index=True))
    assignment_columns = [
        "query_group_id",
        "source_key",
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
    selected_assignment_columns = [column for column in assignment_columns if column in assignments.columns]
    rows = all_rows.merge(
        assignments[selected_assignment_columns],
        on=["query_group_id", "source_key"],
        how="inner",
        suffixes=("", "_split"),
    )
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
    rows, assignments = _refresh_stratified_metadata_from_active_labels(rows, assignments)
    return rows, assignments


def _breakdown_predictions(predictions: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Summarize predictions by one column for JSON reports."""

    return {
        str(value): _summarize_predictions(group.copy())
        for value, group in predictions.groupby(column, dropna=False, sort=True)
    }


def _score_classic_stratified_eval_test_choices(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    model: LGBMClassifier,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score query choices for the promoted stratified calibration/test split."""

    split_rows, assignments = _load_classic_stratified_eval_rows(bundle, spec, split_spec)
    probabilities = model.predict_proba(_classic_feature_matrix(split_rows, feature_columns))[:, 1]
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
    return choices, assignments


def _summarize_classic_stratified_predictions(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    split_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build promoted stratified split summary and test breakdowns from predictions."""

    predictions = predictions.copy()
    predictions["gate_bucket"] = (
        _total_error_gate_bucket(predictions) if not predictions.empty else pd.Series(dtype="string")
    )
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


def _optional_metric_float_cell(value: Any) -> str:
    """Return a formatted float cell, preserving missing values as n/a."""

    if value is None:
        return "n/a"
    return _format_metric_float(value)


def _count_from_mapping(values: Mapping[str, Any], key: str) -> int:
    """Return an integer count from a JSON-style mapping."""

    value = values.get(key, 0)
    if value is None:
        return 0
    return int(value)


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


def _classic_gate_bucket_table_rows(summary: dict[str, Any], breakdowns: dict[str, Any]) -> list[list[str]]:
    """Return selected-gate calibration bucket rows for the promoted bucketed gate."""

    abstain_rule = summary.get("abstain_rule")
    split_summary = summary.get("stratified_eval_test_split")
    if not isinstance(abstain_rule, dict) or not isinstance(split_summary, dict):
        return []
    score_thresholds = abstain_rule.get("bucketed_score_thresholds")
    if not isinstance(score_thresholds, dict):
        return []
    margin_thresholds = abstain_rule.get("bucketed_margin_thresholds")
    if not isinstance(margin_thresholds, dict):
        margin_thresholds = {}

    promoted_gate = abstain_rule.get("promoted_stratified_gate")
    if isinstance(promoted_gate, dict):
        fit_split = str(promoted_gate.get("fit_split", "calibration_fit"))
        check_split = str(promoted_gate.get("selection_split", "calibration_check"))
        test_split = str(promoted_gate.get("test_split", "test"))
    else:
        fit_split = "calibration_fit"
        check_split = "calibration_check"
        test_split = "test"

    training_summary = summary.get("training_summary")
    if not isinstance(training_summary, dict):
        training_summary = {}
    train_query_counts = training_summary.get("gate_bucket_query_counts")
    if not isinstance(train_query_counts, dict):
        train_query_counts = {}
    train_row_counts = training_summary.get("gate_bucket_row_counts")
    if not isinstance(train_row_counts, dict):
        train_row_counts = {}

    split_counts = split_summary.get("gate_bucket_split_counts")
    if not isinstance(split_counts, dict):
        split_counts = {}
    test_breakdowns = breakdowns.get("gate_bucket")
    if not isinstance(test_breakdowns, dict):
        test_breakdowns = {}

    rows: list[list[str]] = []
    for bucket in _TOTAL_ERROR_SCORE_BUCKETS:
        bucket_split_counts = split_counts.get(bucket)
        if not isinstance(bucket_split_counts, dict):
            bucket_split_counts = {}
        fit_count = _count_from_mapping(bucket_split_counts, fit_split)
        check_count = _count_from_mapping(bucket_split_counts, check_split)
        test_metrics = test_breakdowns.get(bucket)
        if not isinstance(test_metrics, dict):
            test_metrics = {}
        test_count = _count_from_mapping(test_metrics, "n_queries") or _count_from_mapping(
            bucket_split_counts,
            test_split,
        )
        calibration_count = fit_count + check_count
        margin_threshold = margin_thresholds.get(bucket) if bucket in _TOTAL_ERROR_MARGIN_BUCKETS else None
        rows.append(
            [
                bucket,
                _optional_metric_float_cell(score_thresholds.get(bucket)),
                _optional_metric_float_cell(margin_threshold),
                str(_count_from_mapping(train_query_counts, bucket)),
                str(_count_from_mapping(train_row_counts, bucket)),
                str(fit_count),
                str(check_count),
                str(calibration_count),
                str(test_count),
                str(_count_from_mapping(test_metrics, "n_positive_queries")),
                str(_count_from_mapping(test_metrics, "n_negative_queries")),
                str(_count_from_mapping(test_metrics, "errors")),
                "n/a" if test_count == 0 else _format_metric_float(test_metrics.get("error_rate", 0.0)),
                str(_count_from_mapping(test_metrics, "false_abstain")),
                str(_count_from_mapping(test_metrics, "false_link")),
                str(_count_from_mapping(test_metrics, "wrong_candidate_link")),
            ]
        )
    return rows


def format_classic_selected_gate_tables(summary: dict[str, Any]) -> str:
    """Return the selected-gate stratified-test breakdown tables."""

    split_summary = summary.get("stratified_eval_test_split")
    if not isinstance(split_summary, dict):
        return ""
    breakdowns = split_summary.get("test_breakdowns")
    if not isinstance(breakdowns, dict):
        return ""

    lines: list[str] = []
    bucket_rows = _classic_gate_bucket_table_rows(summary, breakdowns)
    if bucket_rows:
        lines.extend(["## By Calibration Bucket, Selected Gate", ""])
        lines.extend(
            _markdown_table(
                [
                    "bucket",
                    "score threshold",
                    "margin threshold",
                    "train queries",
                    "train rows",
                    "calibration fit",
                    "calibration check",
                    "calibration total",
                    "test queries",
                    "test positive queries",
                    "test negative queries",
                    "test errors",
                    "test error rate",
                    "false abstain",
                    "false link",
                    "wrong link",
                ],
                bucket_rows,
            )
        )
        lines.append("")

    lines.extend(["## By Dataset Slice, Selected Gate", ""])
    source_breakdown = dict(breakdowns.get("source_key", {}))
    source_rows = [
        _metric_breakdown_row(str(slice_name), dict(metrics))
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
            factor_rows.append(_metric_factor_row(factor, str(group_name), dict(metrics)))
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


def _evaluate_classic_stratified_eval_test_split(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    model: LGBMClassifier,
    feature_columns: tuple[str, ...],
    *,
    score_threshold: float,
    margin_threshold: float,
    single_candidate_score_threshold: float | None = None,
    bucketed_score_thresholds: dict[str, float] | None = None,
    bucketed_margin_threshold: float | None = None,
    bucketed_margin_thresholds: dict[str, float] | None = None,
    scored_choices: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Score the promoted stratified calibration/test split with the active gate."""

    if scored_choices is None:
        choices, assignments = _score_classic_stratified_eval_test_choices(
            bundle,
            spec,
            split_spec,
            model,
            feature_columns,
        )
    else:
        choices, assignments = scored_choices

    predictions = _apply_classic_gate(
        choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    return _summarize_classic_stratified_predictions(predictions, assignments, split_spec)


def _score_eval_candidate_rows(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_margin: bool,
    limits: tuple[int, ...] = (5, 25),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for limit in sorted(set(int(limit) for limit in limits)):
        limited = df[df["retrieval_rank"] <= limit].copy()
        limited_probabilities = probabilities[limited.index.to_numpy()]
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
    save_artifact_to: Path | None = None,
    artifact_audit_metadata: Mapping[str, Any] | None = None,
    required_rust_capabilities: Sequence[str] = INCREMENTAL_LINKING_RUST_CAPABILITIES,
) -> dict[str, Any]:
    """Fit, calibrate, and evaluate the official classic pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
    train_df = _read_csv(_resolve_path(bundle, spec["train_path"]), compression="gzip")
    train_df["retrieval_rank"] = pd.to_numeric(train_df["retrieval_rank"], errors="coerce")
    train_df = train_df[train_df["retrieval_rank"] <= 25].copy()
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
    model = _build_classic_classifier(spec["best_params"], monotone_constraints=monotone_constraints)
    started = perf_counter()
    model.fit(train_matrix, train_labels, sample_weight=sample_weight)
    train_seconds = float(perf_counter() - started)

    gate_source_path = _resolve_path(bundle, spec.get("classic_gate_source_path", spec["hwang_eval_path"]))
    gate_source_eval = _read_csv(gate_source_path, compression="gzip")
    gate_source_probabilities = model.predict_proba(_classic_feature_matrix(gate_source_eval, feature_columns))[:, 1]
    calibration_limit = int(spec.get("classic_gate_calibration_retrieval_limit", 50))
    gate_source_query_choices = _score_eval_candidate_rows(
        gate_source_eval,
        gate_source_probabilities,
        include_margin=True,
        limits=(calibration_limit,),
    )

    calibration_groups = set(
        _read_csv(_resolve_path(bundle, spec["classic_gate_calibration_base_groups_path"]))["base_group_id"].astype(str)
    )
    internal_eval_groups = set(
        _read_csv(_resolve_path(bundle, spec["classic_gate_internal_eval_base_groups_path"]))["base_group_id"].astype(
            str
        )
    )
    calibration_rows = gate_source_query_choices[
        (gate_source_query_choices["retrieval_rank_limit"] == calibration_limit)
        & (gate_source_query_choices["base_group_id"].isin(calibration_groups))
    ].copy()
    internal_eval_rows = gate_source_query_choices[
        (gate_source_query_choices["retrieval_rank_limit"] == calibration_limit)
        & (gate_source_query_choices["base_group_id"].isin(internal_eval_groups))
    ].copy()
    frozen_expected = dict(bundle.expected_metrics.get("classic", {}))
    fixed_bucketed_gate = _fixed_bucketed_gate_spec(spec)
    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    stratified_scored_choices: tuple[pd.DataFrame, pd.DataFrame] | None = None
    promoted_gate_summary: dict[str, Any] | None = None
    if promoted_gate_config is not None:
        if spec.get("stratified_eval_test_split") is None:
            raise ValueError("classic.promoted_stratified_gate requires classic.stratified_eval_test_split")
        split_spec = dict(spec["stratified_eval_test_split"])
        stratified_scored_choices = _score_classic_stratified_eval_test_choices(
            bundle,
            spec,
            split_spec,
            model,
            feature_columns,
        )
        selected_gate_result = _fit_promoted_stratified_total_error_gate(
            stratified_scored_choices[0],
            promoted_gate_config,
        )
        selected_gate = selected_gate_result["gate"]
        bucketed_score_thresholds = dict(selected_gate.score_thresholds)
        bucketed_margin_thresholds = dict(selected_gate.margin_thresholds)
        bucketed_margin_threshold = None
        score_threshold = float(bucketed_score_thresholds["multi_candidate|multi_letter_first"])
        margin_threshold = float(bucketed_margin_thresholds["multi_candidate|multi_letter_first"])
        single_candidate_score_threshold = float(bucketed_score_thresholds["single_candidate|single_letter_first"])
        calibration_metrics = {
            "split": str(promoted_gate_config["fit_split"]),
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            **dict(selected_gate_result["fit_metrics"]),
        }
        fit_predictions = _apply_total_error_gate(
            stratified_scored_choices[0][
                stratified_scored_choices[0]["split"] == str(promoted_gate_config["fit_split"])
            ].copy(),
            selected_gate,
        )
        single_candidate_predictions = fit_predictions[
            (pd.to_numeric(fit_predictions["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
            | fit_predictions["score_margin"].isna()
        ].copy()
        single_candidate_calibration_metrics = {
            "single_candidate_score_threshold": single_candidate_score_threshold,
            **_summarize_predictions(single_candidate_predictions),
        }
        promoted_gate_summary = {
            "mode": str(promoted_gate_config["mode"]),
            "fit_split": str(promoted_gate_config["fit_split"]),
            "selection_split": str(promoted_gate_config["selection_split"]),
            "test_split": str(promoted_gate_config["test_split"]),
            "score_grid_size": int(promoted_gate_config["score_grid_size"]),
            "margin_grid_size": int(promoted_gate_config["margin_grid_size"]),
            "lambda_grid": list(promoted_gate_config["lambda_grid"]),
            "selection_metric": str(promoted_gate_config["selection_metric"]),
            "error_weights": dict(promoted_gate_config["error_weights"]),
            "selected_gate_name": selected_gate.name,
            "selected_gate": {
                "name": selected_gate.name,
                "score_thresholds": dict(selected_gate.score_thresholds),
                "margin_thresholds": dict(selected_gate.margin_thresholds),
                "lambda_penalty": selected_gate.lambda_penalty,
            },
            "selection_key": dict(selected_gate_result["selection_key"]),
            "fit_metrics": dict(selected_gate_result["fit_metrics"]),
            "check_metrics": dict(selected_gate_result["check_metrics"]),
            "candidate_metrics": selected_gate_result["candidate_metrics"],
        }
    elif fixed_bucketed_gate is None:
        fitted_rule = _fit_score_margin_gate(
            calibration_rows,
            reference_score_threshold=float(frozen_expected["score_threshold"]),
            reference_margin_threshold=float(frozen_expected["margin_threshold"]),
        )
        single_candidate_rule = _fit_single_candidate_score_gate(
            calibration_rows,
            reference_score_threshold=float(
                frozen_expected.get("single_candidate_score_threshold", frozen_expected["score_threshold"])
            ),
            score_grid_size=int(spec.get("single_candidate_score_grid_size", spec.get("score_grid_size", 101))),
        )
        score_threshold = float(fitted_rule["score_threshold"])
        margin_threshold = float(fitted_rule["margin_threshold"])
        single_candidate_score_threshold = float(single_candidate_rule["single_candidate_score_threshold"])
        bucketed_score_thresholds = None
        bucketed_margin_threshold = None
        bucketed_margin_thresholds = None
        calibration_metrics = {
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            **fitted_rule["metrics"],
        }
        single_candidate_calibration_metrics = {
            "single_candidate_score_threshold": single_candidate_score_threshold,
            **single_candidate_rule["metrics"],
        }
    else:
        bucketed_score_thresholds = dict(fixed_bucketed_gate["score_thresholds"])
        bucketed_margin_thresholds = (
            dict(fixed_bucketed_gate["margin_thresholds"])
            if fixed_bucketed_gate.get("margin_thresholds") is not None
            else None
        )
        bucketed_margin_threshold = (
            float(fixed_bucketed_gate["margin_threshold"])
            if fixed_bucketed_gate.get("margin_threshold") is not None
            else None
        )
        score_threshold = float(
            fixed_bucketed_gate.get(
                "score_threshold",
                bucketed_score_thresholds["multi_candidate|multi_letter_first"],
            )
        )
        margin_threshold = float(
            fixed_bucketed_gate.get(
                "margin_threshold",
                frozen_expected.get("margin_threshold", 0.0),
            )
        )
        single_candidate_score_threshold = float(
            fixed_bucketed_gate.get(
                "single_candidate_score_threshold",
                bucketed_score_thresholds["single_candidate|single_letter_first"],
            )
        )
        calibration_metrics = _score_abstain_rule(
            calibration_rows,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )
        single_candidate_rows = calibration_rows[
            (pd.to_numeric(calibration_rows["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
            | calibration_rows["score_margin"].isna()
        ].copy()
        single_candidate_calibration_metrics = _score_abstain_rule(
            single_candidate_rows,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )

    s2and_eval = _read_csv(_resolve_path(bundle, spec["s2and_eval_path"]), compression="gzip")
    s2and_probabilities = model.predict_proba(_classic_feature_matrix(s2and_eval, feature_columns))[:, 1]
    s2and_query_choices = _score_eval_candidate_rows(s2and_eval, s2and_probabilities, include_margin=True)
    s2and_eval_summary = _evaluate_scored_windows(
        s2and_query_choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    hwang_eval = _read_csv(_resolve_path(bundle, spec["hwang_eval_path"]), compression="gzip")
    hwang_probabilities = model.predict_proba(_classic_feature_matrix(hwang_eval, feature_columns))[:, 1]
    hwang_query_choices = _score_eval_candidate_rows(hwang_eval, hwang_probabilities, include_margin=True)
    optional_eval_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, path_like in _iter_extra_eval_paths(spec):
        eval_df = _read_csv(_resolve_path(bundle, path_like), compression="gzip")
        eval_probabilities = model.predict_proba(_classic_feature_matrix(eval_df, feature_columns))[:, 1]
        eval_query_choices = _score_eval_candidate_rows(eval_df, eval_probabilities, include_margin=True)
        optional_eval_summaries[_summary_key_for_eval_dataset(dataset_name)] = _evaluate_scored_windows(
            eval_query_choices,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )

    override_path = spec.get("hwang_clean_override_path")
    override_df = _read_csv(_resolve_path(bundle, str(override_path))) if override_path else None
    hwang_eval_summary = _evaluate_scored_windows(
        hwang_query_choices,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        override_df=override_df,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    internal_eval_summary = _score_abstain_rule(
        internal_eval_rows,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        single_candidate_score_threshold=single_candidate_score_threshold,
        bucketed_score_thresholds=bucketed_score_thresholds,
        bucketed_margin_threshold=bucketed_margin_threshold,
        bucketed_margin_thresholds=bucketed_margin_thresholds,
    )
    stratified_eval_test_summary = None
    if spec.get("stratified_eval_test_split") is not None:
        stratified_eval_test_summary = _evaluate_classic_stratified_eval_test_split(
            bundle,
            spec,
            dict(spec["stratified_eval_test_split"]),
            model,
            feature_columns,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
            scored_choices=stratified_scored_choices,
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
            "train_holdout_filter_summary": train_holdout_filter_summary,
            "train_filter_summary": train_filter_summary,
        },
        "abstain_rule": {
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            "single_candidate_score_threshold": single_candidate_score_threshold,
            "calibration_mode": (
                "promoted_stratified_weighted_average_error_4score_2margin"
                if promoted_gate_config is not None
                else ("fixed_bucketed_gate" if fixed_bucketed_gate is not None else "legacy_score_margin")
            ),
            "fixed_bucketed_gate": promoted_gate_config is None and fixed_bucketed_gate is not None,
            "promoted_stratified_gate": promoted_gate_summary,
            "bucketed_score_thresholds": bucketed_score_thresholds,
            "bucketed_margin_threshold": bucketed_margin_threshold,
            "bucketed_margin_thresholds": bucketed_margin_thresholds,
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
        manual_probabilities = model.predict_proba(_classic_feature_matrix(manual_holdout, feature_columns))[:, 1]
        summary["manual_holdout"] = _evaluate_classic_manual_holdout(
            manual_holdout,
            manual_probabilities,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            single_candidate_score_threshold=single_candidate_score_threshold,
            bucketed_score_thresholds=bucketed_score_thresholds,
            bucketed_margin_threshold=bucketed_margin_threshold,
            bucketed_margin_thresholds=bucketed_margin_thresholds,
        )
    for summary_key, eval_summary in optional_eval_summaries.items():
        summary[summary_key] = eval_summary
    selected_gate_tables = format_classic_selected_gate_tables(summary)
    if selected_gate_tables:
        selected_gate_tables_path = output_dir / "selected_gate_tables.md"
        selected_gate_tables_path.write_text(selected_gate_tables, encoding="utf-8")
        summary["selected_gate_tables_path"] = str(selected_gate_tables_path.relative_to(output_dir))
    if save_artifact_to is not None:
        fixture_source = gate_source_eval.head(5)
        if len(fixture_source) == 0:
            fixture_matrix = train_matrix[:5]
        else:
            fixture_matrix = _classic_feature_matrix(fixture_source, feature_columns).to_numpy(dtype=np.float32)
        artifact_metadata = save_incremental_linking_artifact(
            model,
            Path(save_artifact_to),
            feature_columns=feature_columns,
            retrieval_top_k=25,
            gate_config={
                "score_threshold": score_threshold,
                "margin_threshold": margin_threshold,
                "single_candidate_score_threshold": single_candidate_score_threshold,
                "bucketed_score_thresholds": bucketed_score_thresholds,
                "bucketed_margin_threshold": bucketed_margin_threshold,
                "bucketed_margin_thresholds": bucketed_margin_thresholds,
                "calibration_mode": summary["abstain_rule"]["calibration_mode"],
            },
            prediction_fixture_matrix=fixture_matrix,
            required_rust_capabilities=required_rust_capabilities,
            audit_metadata=artifact_audit_metadata,
        )
        summary["artifact"] = {
            "path": str(Path(save_artifact_to)),
            "schema_version": artifact_metadata.schema_version,
            "feature_schema_digest": artifact_metadata.feature_schema_digest,
            "production_contract_digest": artifact_metadata.production_contract_digest,
            "retrieval_stack_digest": artifact_metadata.retrieval_stack_digest,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _expected_metric_actual_value(summary: dict[str, Any], *, key: str) -> float:
    """Resolve a frozen metric key against a runtime summary."""

    if key == "manual_holdout_overall_balanced_accuracy":
        return float(summary["manual_holdout"]["overall"]["balanced_accuracy"])
    if key == "score_threshold":
        return float(summary["abstain_rule"]["score_threshold"])
    if key == "margin_threshold":
        return float(summary["abstain_rule"]["margin_threshold"])
    if key == "single_candidate_score_threshold":
        return float(summary["abstain_rule"]["single_candidate_score_threshold"])
    if key == "multi_candidate_multi_letter_score_threshold":
        return float(summary["abstain_rule"]["bucketed_score_thresholds"]["multi_candidate|multi_letter_first"])
    if key == "multi_candidate_single_letter_score_threshold":
        return float(summary["abstain_rule"]["bucketed_score_thresholds"]["multi_candidate|single_letter_first"])
    if key == "single_candidate_multi_letter_score_threshold":
        return float(summary["abstain_rule"]["bucketed_score_thresholds"]["single_candidate|multi_letter_first"])
    if key == "single_candidate_single_letter_score_threshold":
        return float(summary["abstain_rule"]["bucketed_score_thresholds"]["single_candidate|single_letter_first"])
    if key == "multi_candidate_multi_letter_margin_threshold":
        margin_thresholds = summary["abstain_rule"].get("bucketed_margin_thresholds")
        if margin_thresholds is not None:
            return float(margin_thresholds["multi_candidate|multi_letter_first"])
        return float(summary["abstain_rule"]["bucketed_margin_threshold"])
    if key == "multi_candidate_single_letter_margin_threshold":
        margin_thresholds = summary["abstain_rule"].get("bucketed_margin_thresholds")
        if margin_thresholds is not None:
            return float(margin_thresholds["multi_candidate|single_letter_first"])
        return float(summary["abstain_rule"]["bucketed_margin_threshold"])
    if key == "stratified_test_balanced_accuracy":
        return float(summary["stratified_eval_test_split"]["overall"]["test"]["balanced_accuracy"])
    if key == "stratified_test_accuracy":
        return float(summary["stratified_eval_test_split"]["overall"]["test"]["accuracy"])
    if key == "stratified_test_error_rate":
        return float(summary["stratified_eval_test_split"]["overall"]["test"]["error_rate"])

    window_match = re.fullmatch(r"(.+)_w(\d+)_balanced_accuracy", key)
    if window_match is None:
        raise KeyError(f"Unsupported expected metric key: {key}")
    dataset_name, window = window_match.groups()
    if dataset_name == "hwang_clean":
        if f"w{window}" in summary["hwang_cleaned_eval"]:
            return float(summary["hwang_cleaned_eval"][f"w{window}"]["cleaned_balanced_accuracy"])
        return float(summary["hwang_cleaned_eval"][window]["overall"]["balanced_accuracy"])
    summary_key = _summary_key_for_eval_dataset(dataset_name)
    if summary_key not in summary:
        raise KeyError(f"Eval summary missing for expected metric key {key!r}: {summary_key}")
    return float(summary[summary_key][window]["overall"]["balanced_accuracy"])


def expected_metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Build the frozen expected-metrics payload from one runtime summary."""

    expected: dict[str, float] = {}
    if "manual_holdout" in summary:
        expected["manual_holdout_overall_balanced_accuracy"] = float(
            summary["manual_holdout"]["overall"]["balanced_accuracy"]
        )

    overall_eval_keys = sorted(key for key in summary if str(key).startswith("overall_") and str(key).endswith("_eval"))
    for summary_key in overall_eval_keys:
        dataset_name = str(summary_key)[len("overall_") : -len("_eval")]
        window_payload = dict(summary[summary_key])
        for window in sorted(window_payload, key=lambda value: int(value)):
            expected[f"{dataset_name}_w{int(window)}_balanced_accuracy"] = float(
                window_payload[str(window)]["overall"]["balanced_accuracy"]
            )

    if "hwang_cleaned_eval" in summary:
        cleaned_payload = dict(summary["hwang_cleaned_eval"])
        for window_key in sorted(cleaned_payload, key=lambda value: int(str(value).lstrip("w"))):
            normalized_window = int(str(window_key).lstrip("w"))
            expected[f"hwang_clean_w{normalized_window}_balanced_accuracy"] = float(
                cleaned_payload[str(window_key)]["cleaned_balanced_accuracy"]
            )

    expected["score_threshold"] = float(summary["abstain_rule"]["score_threshold"])
    expected["margin_threshold"] = float(summary["abstain_rule"]["margin_threshold"])
    if "single_candidate_score_threshold" in summary["abstain_rule"]:
        expected["single_candidate_score_threshold"] = float(
            summary["abstain_rule"]["single_candidate_score_threshold"]
        )
    bucketed_score_thresholds = summary["abstain_rule"].get("bucketed_score_thresholds")
    if bucketed_score_thresholds is not None:
        expected["multi_candidate_multi_letter_score_threshold"] = float(
            bucketed_score_thresholds["multi_candidate|multi_letter_first"]
        )
        expected["multi_candidate_single_letter_score_threshold"] = float(
            bucketed_score_thresholds["multi_candidate|single_letter_first"]
        )
        expected["single_candidate_multi_letter_score_threshold"] = float(
            bucketed_score_thresholds["single_candidate|multi_letter_first"]
        )
        expected["single_candidate_single_letter_score_threshold"] = float(
            bucketed_score_thresholds["single_candidate|single_letter_first"]
        )
    bucketed_margin_thresholds = summary["abstain_rule"].get("bucketed_margin_thresholds")
    if bucketed_margin_thresholds is not None:
        expected["multi_candidate_multi_letter_margin_threshold"] = float(
            bucketed_margin_thresholds["multi_candidate|multi_letter_first"]
        )
        expected["multi_candidate_single_letter_margin_threshold"] = float(
            bucketed_margin_thresholds["multi_candidate|single_letter_first"]
        )
    if "stratified_eval_test_split" in summary:
        stratified_test = summary["stratified_eval_test_split"]["overall"]["test"]
        expected["stratified_test_balanced_accuracy"] = float(stratified_test["balanced_accuracy"])
        expected["stratified_test_accuracy"] = float(stratified_test["accuracy"])
        expected["stratified_test_error_rate"] = float(stratified_test["error_rate"])
    return expected


def compare_to_expected(summary: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    """Return headline metric deltas relative to the frozen expectations."""

    return {key: _expected_metric_actual_value(summary, key=key) - float(expected[key]) for key in expected}
