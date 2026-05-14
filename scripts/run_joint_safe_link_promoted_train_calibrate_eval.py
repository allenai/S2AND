"""Train, calibrate, and evaluate the promoted joint-safe linker target.

This is the official replay entrypoint for the promoted LightGBM
linker/reranker target. It intentionally pins the promoted target JSON instead
of trusting bundle manifests whose classic model specs predate the promotion.

The default `minimal-raw-rust` mode starts from the self-contained
raw+SPECTER2+labels bundle, rebuilds the promoted feature tables through the
Rust-backed pairwise and row-formula paths, then runs the train/calibrate/eval
stack. The active candidate-member contract is block-local for retrieval,
pairwise distance summaries, and appended `pw_*` aggregates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import s2and.incremental_linking.query_adapter as retrieval  # noqa: E402
from s2and import feature_port  # noqa: E402
from s2and import text as s2and_text  # noqa: E402
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.incremental_linking.artifact import save_incremental_linking_artifact  # noqa: E402
from s2and.incremental_linking.contracts import (  # noqa: E402
    INCREMENTAL_LINKING_RUST_CAPABILITIES,
    canonical_json_digest,
    promoted_linker_feature_schema_digest,
)
from s2and.incremental_linking.features import (  # noqa: E402
    PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS,
    promoted_linker_feature_columns,
)
from s2and.incremental_linking.linker_pairwise import (  # noqa: E402
    LinkerCandidateBatch,
    compute_candidate_batch_pairwise_aggregate_stats_rust,
    promoted_pairwise_aggregate_columns,
)
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features  # noqa: E402
from s2and.incremental_linking.runtime import compute_candidate_batch_pairwise_model_and_aggregate_stats  # noqa: E402
from s2and.incremental_linking_training import (  # noqa: E402
    DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY,
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME,
    build_cluster_profile,
    build_labeled_retrieval_subblock_index,
    build_rust_hybrid_centroid_retriever,
    counter_query_overlap,
    load_clusterer,
    load_giant_block_dataset,
    load_labeled_dataset,
    name_count_rarity_features,
    rank_top_summaries_rust_hybrid_centroid,
    specter_exemplar_similarity,
    year_compatibility,
)
from s2and.model import (  # noqa: E402
    _apply_dataset_name_count_semantics_for_prediction,
    _build_incremental_constraint_backend,
)
from s2and.runtime import build_runtime_context  # noqa: E402

os.environ.setdefault("S2AND_BACKEND", "rust")
os.environ.setdefault("S2AND_RUST_FEATURIZER_MAX_INMEM", "1")

PACKAGE_DATA_ROOT = REPO_ROOT / "s2and" / "data"
DEFAULT_SOURCE_BUNDLE_ROOT = PACKAGE_DATA_ROOT / "joint_safe_link_minimal_raw_specter_20260507a"
DEFAULT_TARGET_JSON = PACKAGE_DATA_ROOT / "production_incremental_linker_v1.2" / "training_target.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "joint_safe_link_promoted_official_20260507"
DEFAULT_PAIRWISE_MODEL_PATH = PACKAGE_DATA_ROOT / "production_model_v1.2.pickle"
DEFAULT_LINKER_ARTIFACT_DIR = PACKAGE_DATA_ROOT / "production_incremental_linker_v1.2"
DEFAULT_GIANT_DATASET_ROOT = Path(r"D:\data")
DEFAULT_TOTAL_RAM_BYTES = 48 * 1024**3
GIANT_DATASETS = frozenset({"a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park"})
REQUIRED_TABLE_KEYS = (
    "train_path",
    "classic_gate_source_path",
    "s2and_eval_path",
    "hwang_eval_path",
)
PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION = "precomputed_promoted_feature_bundle_v1"


# Shared training, calibration, and evaluation helpers live in this entrypoint
# so production training cannot silently default to historical feature tables.
_ANCHOR_EVIDENCE_FEATURE_COLUMNS = (
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
)
_DERIVED_PROMOTED_FEATURE_COLUMNS = (
    "retrieval_reciprocal_rank",
    "cluster_size_log",
    "candidate_year_span",
    "year_gap_to_candidate_range",
    "year_gap_signed_to_candidate_range",
    "candidate_dominant_first_name_length",
    "query_first_prefix_match_any_length",
    "same_dominant_first_as_best_top5",
    "same_family_as_heuristic_choice",
)
_ANCHOR_EVIDENCE_PREREQUISITES = (
    "min_distance",
    "retrieval_score_gap_vs_best_competitor",
    "same_family_as_top1",
    "retrieval_rank",
)
_CLASSIC_DERIVABLE_FEATURE_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "retrieval_reciprocal_rank": ("retrieval_rank",),
    "cluster_size_log": ("cluster_size",),
    "candidate_year_span": ("candidate_year_min", "candidate_year_max", "candidate_year_range_missing"),
    "year_gap_to_candidate_range": (
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
    ),
    "year_gap_signed_to_candidate_range": (
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
    ),
    "candidate_dominant_first_name_length": ("dominant_first_name",),
    "query_first_prefix_match_any_length": ("dominant_first_name",),
    "same_dominant_first_as_best_top5": (
        "query_group_id",
        "dominant_first_name",
        "retrieval_rank",
        "top5_mean_distance",
        "candidate_component_key",
    ),
    "same_family_as_heuristic_choice": (
        "query_group_id",
        "dominant_first_name",
        "retrieval_rank",
        "top5_mean_distance",
        "candidate_component_key",
        "retrieval_score",
    ),
    **{feature: _ANCHOR_EVIDENCE_PREREQUISITES for feature in _ANCHOR_EVIDENCE_FEATURE_COLUMNS},
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
_DEFAULT_PROMOTED_GATE_FIXED_GRID_STEP = 0.01
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


def _cluster_size_log(cluster_size: Any) -> float:
    """Return an uncapped log-size primitive."""

    return float(math.log1p(max(0.0, float(cluster_size or 0.0))))


def _numeric_feature_series(df: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    """Return one numeric feature column with a deterministic default for formula derivation."""

    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(np.float32)


def _query_first_series_for_prefix(out: pd.DataFrame) -> list[str]:
    if "query_author" in out.columns:
        query_first_from_author = out["query_author"].map(_query_first_token)
    else:
        query_first_from_author = pd.Series([""] * len(out), index=out.index, dtype="string")
    if "query_first_token" in out.columns:
        query_first_from_token = out["query_first_token"].map(_normalize_optional_letters)
    else:
        query_first_from_token = pd.Series([""] * len(out), index=out.index, dtype="string")
    return [
        author_token if author_token else token
        for author_token, token in zip(query_first_from_author, query_first_from_token, strict=True)
    ]


def _derive_promoted_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive promoted row features from portable candidate-row primitives."""

    out = df.copy()
    if "retrieval_rank" in out.columns:
        retrieval_rank = pd.to_numeric(out["retrieval_rank"], errors="coerce").fillna(0.0).astype(np.float32)
        out["retrieval_reciprocal_rank"] = 1.0 / np.maximum(retrieval_rank.to_numpy(dtype=np.float32), 1.0)
    if "cluster_size" in out.columns:
        cluster_size = pd.to_numeric(out["cluster_size"], errors="coerce").fillna(0.0).astype(np.float32)
        out["cluster_size_log"] = cluster_size.map(_cluster_size_log)
    if {"candidate_year_min", "candidate_year_max", "candidate_year_range_missing"}.issubset(out.columns):
        candidate_year_min = pd.to_numeric(out["candidate_year_min"], errors="coerce").fillna(0.0).astype(np.float32)
        candidate_year_max = pd.to_numeric(out["candidate_year_max"], errors="coerce").fillna(0.0).astype(np.float32)
        candidate_missing = (
            pd.to_numeric(out["candidate_year_range_missing"], errors="coerce").fillna(1.0).astype(np.float32) > 0.0
        )
        out["candidate_year_span"] = np.where(
            candidate_missing,
            0.0,
            np.maximum(
                candidate_year_max.to_numpy(dtype=np.float32) - candidate_year_min.to_numpy(dtype=np.float32),
                0.0,
            ),
        )
        if {"query_year", "query_year_missing"}.issubset(out.columns):
            query_year = pd.to_numeric(out["query_year"], errors="coerce").fillna(0.0).astype(np.float32)
            query_missing = (
                pd.to_numeric(out["query_year_missing"], errors="coerce").fillna(1.0).astype(np.float32) > 0.0
            )
            observed = ~(candidate_missing | query_missing)
            lower = observed & (query_year < candidate_year_min)
            upper = observed & (query_year > candidate_year_max)
            gap = np.zeros(len(out), dtype=np.float32)
            signed_gap = np.zeros(len(out), dtype=np.float32)
            lower_gap = (candidate_year_min[lower] - query_year[lower]).to_numpy(dtype=np.float32)
            upper_gap = (query_year[upper] - candidate_year_max[upper]).to_numpy(dtype=np.float32)
            gap[lower.to_numpy(dtype=bool)] = lower_gap
            gap[upper.to_numpy(dtype=bool)] = upper_gap
            signed_gap[lower.to_numpy(dtype=bool)] = -lower_gap
            signed_gap[upper.to_numpy(dtype=bool)] = upper_gap
            out["year_gap_to_candidate_range"] = gap
            out["year_gap_signed_to_candidate_range"] = signed_gap
    if "dominant_first_name" in out.columns:
        query_first = _query_first_series_for_prefix(out)
        dominant_first = out["dominant_first_name"].map(_normalize_optional_letters)
        out["candidate_dominant_first_name_length"] = [float(len(value)) for value in dominant_first]
        out["query_first_prefix_match_any_length"] = [
            1.0 if query and dominant and (query.startswith(dominant) or dominant.startswith(query)) else 0.0
            for query, dominant in zip(query_first, dominant_first, strict=True)
        ]
        if {
            "query_group_id",
            "retrieval_rank",
            "top5_mean_distance",
            "candidate_component_key",
        }.issubset(out.columns):
            group_key = out["query_group_id"].astype(str)
            retrieval_rank_numeric = pd.to_numeric(out["retrieval_rank"], errors="coerce")
            retrieval_score_sort = (
                -pd.to_numeric(out["retrieval_score"], errors="coerce")
                if "retrieval_score" in out.columns
                else retrieval_rank_numeric
            )
            grouping_frame = out.assign(
                _query_group_key=group_key,
                _dominant_first_alpha=dominant_first,
                _retrieval_rank_numeric=retrieval_rank_numeric,
                _retrieval_score_sort=retrieval_score_sort,
                _top5_mean_distance_numeric=pd.to_numeric(out["top5_mean_distance"], errors="coerce"),
                _row_order=np.arange(len(out)),
            )
            top1_rows = grouping_frame.sort_values(
                [
                    "_query_group_key",
                    "_retrieval_score_sort",
                    "_retrieval_rank_numeric",
                    "candidate_component_key",
                    "_row_order",
                ],
                kind="stable",
            )
            top1_by_group = top1_rows.drop_duplicates("_query_group_key").set_index("_query_group_key")
            top1_dominant = group_key.map(top1_by_group["_dominant_first_alpha"])
            best_top5_rows = grouping_frame.sort_values(
                [
                    "_query_group_key",
                    "_top5_mean_distance_numeric",
                    "_retrieval_score_sort",
                    "_retrieval_rank_numeric",
                    "candidate_component_key",
                    "_row_order",
                ],
                kind="stable",
            )
            best_top5_by_group = best_top5_rows.drop_duplicates("_query_group_key").set_index("_query_group_key")
            best_top5_dominant = group_key.map(best_top5_by_group["_dominant_first_alpha"])
            dominant_first_top1_match = np.asarray(
                [
                    1.0 if dominant and top1 and dominant == top1 else 0.0
                    for dominant, top1 in zip(dominant_first, top1_dominant, strict=True)
                ],
                dtype=np.float32,
            )
            same_dominant_first_as_best_top5 = np.asarray(
                [
                    1.0 if dominant and best and dominant == best else 0.0
                    for dominant, best in zip(dominant_first, best_top5_dominant, strict=True)
                ],
                dtype=np.float32,
            )
            out["same_dominant_first_as_best_top5"] = same_dominant_first_as_best_top5
            out["same_family_as_heuristic_choice"] = (
                dominant_first_top1_match
                * pd.to_numeric(out["retrieval_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                + same_dominant_first_as_best_top5
                * (
                    1.0
                    - pd.to_numeric(out["top5_mean_distance"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                )
            ).astype(np.float32)
    return out


def _derive_anchor_evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive anchor-evidence formulas from existing candidate-row evidence columns."""

    out = df.copy()
    if {
        "query_group_id",
        "retrieval_score",
        "retrieval_rank",
        "candidate_component_key",
    }.issubset(out.columns):
        retrieval_score_values = pd.to_numeric(out["retrieval_score"], errors="coerce").astype(np.float32)
        stored_rank_values = pd.to_numeric(out["retrieval_rank"], errors="coerce").fillna(99.0).astype(np.float32)
        current_rank = np.zeros(len(out), dtype=np.float32)
        current_gap = np.zeros(len(out), dtype=np.float32)
        best_gap = np.zeros(len(out), dtype=np.float32)
        ordering_frame = pd.DataFrame(
            {
                "_query_group_key": out["query_group_id"].astype(str),
                "_score": retrieval_score_values,
                "_stored_rank": stored_rank_values,
                "_component_key": out["candidate_component_key"].astype(str),
                "_row_index": np.arange(len(out)),
            }
        )
        for _group_key, group in ordering_frame.groupby("_query_group_key", sort=False):
            ordered = group.sort_values(
                ["_score", "_stored_rank", "_component_key", "_row_index"],
                ascending=[False, True, True, True],
                kind="stable",
            )
            indices = ordered["_row_index"].to_numpy(dtype=np.int64)
            if len(indices) == 0:
                continue
            scores = retrieval_score_values.iloc[indices].to_numpy(dtype=np.float32)
            top1 = int(indices[0])
            runner_up = int(indices[1]) if len(indices) > 1 else top1
            best_score = float(np.max(scores))
            for rank, row_index in enumerate(indices, start=1):
                competitor = runner_up if int(row_index) == top1 else top1
                current_rank[int(row_index)] = float(rank)
                current_gap[int(row_index)] = float(
                    retrieval_score_values.iloc[int(row_index)] - retrieval_score_values.iloc[int(competitor)]
                )
                best_gap[int(row_index)] = float(best_score - retrieval_score_values.iloc[int(row_index)])
        out["retrieval_rank"] = current_rank
        out["retrieval_score_gap_vs_best_competitor"] = np.round(current_gap, 6).astype(np.float32)
        out["retrieval_score_best_gap"] = np.round(best_gap, 6).astype(np.float32)

    min_distance = _numeric_feature_series(out, "min_distance", default=10000.0)
    retrieval_gap = _numeric_feature_series(out, "retrieval_score_gap_vs_best_competitor")
    same_top1 = _numeric_feature_series(out, "same_family_as_top1")
    retrieval_rank = _numeric_feature_series(out, "retrieval_rank", default=99.0)

    min_distance_values = min_distance.to_numpy(dtype=np.float32, copy=False)
    retrieval_gap_values = retrieval_gap.to_numpy(dtype=np.float32, copy=False)
    same_top1_values = same_top1.to_numpy(dtype=np.float32, copy=False)
    retrieval_rank_values = retrieval_rank.to_numpy(dtype=np.float32, copy=False)

    min_distance_clip = np.clip(min_distance_values, 0.0, 1.0)
    same_top1_clip = np.clip(same_top1_values, 0.0, 1.0)
    retrieval_gap_positive = np.clip(retrieval_gap_values, 0.0, 0.3) / 0.3
    retrieval_gap_normalized = np.clip((np.clip(retrieval_gap_values, -0.2, 0.3) + 0.2) / 0.5, 0.0, 1.0)

    out["anchor_evidence_count"] = (
        (min_distance_values <= 0.15).astype(np.float32) + (retrieval_gap_values >= 0.02).astype(np.float32)
    ).astype(np.float32)

    distance_signal = 1.0 - min_distance_clip
    support_strength = 0.20 * distance_signal
    out["strong_positive_anchor_score"] = (np.clip(support_strength, 0.0, 1.0) * (0.5 + 0.5 * same_top1_clip)).astype(
        np.float32
    )

    residual_support = 0.28 * distance_signal + 0.08 * retrieval_gap_normalized
    out["weak_residual_anchor_score"] = (same_top1_clip * np.clip(residual_support, 0.0, 1.0)).astype(np.float32)

    out["sparse_relative_winner_score"] = (
        (retrieval_rank_values <= 1.0).astype(np.float32)
        * same_top1_clip
        * np.clip(retrieval_gap_positive, 0.0, 1.0)
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
    if str(configured.get("mode")) != "full_calibration_fixed_grid_4score_2margin":
        raise ValueError("classic.promoted_stratified_gate.mode must be full_calibration_fixed_grid_4score_2margin")
    out = dict(configured)
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
    out["calibration_splits"] = [str(split) for split in calibration_splits]
    if not out["calibration_splits"]:
        raise ValueError("classic.promoted_stratified_gate.calibration_splits must be non-empty")
    out["test_split"] = str(out.get("test_split", split_spec.get("test_split", "test")))
    out["fixed_grid_step"] = float(out.get("fixed_grid_step", _DEFAULT_PROMOTED_GATE_FIXED_GRID_STEP))
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
    requested_derived_features = set(feature_columns) & set(_DERIVED_PROMOTED_FEATURE_COLUMNS)
    if requested_derived_features:
        out = _derive_promoted_features(out)
    for column in feature_columns:
        if column not in out.columns:
            out[column] = np.nan
    for column in feature_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _augmented_feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    prepared = _normalize_augmented_feature_frame(df, feature_columns)
    return prepared.loc[:, list(feature_columns)].copy().astype(np.float32)


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
    "affiliation_contradiction_severity": -1,
    "coauthor_overlap": 1,
    "cluster_size_log": 0,
    "min_distance": -1,
    "specter_exemplar_similarity": 1,
    "affiliation_overlap": 1,
    "year_compatibility": 1,
    "paper_author_list_max_jaccard": 1,
    "paper_author_list_max_containment": 1,
    "paper_author_list_max_overlap_count": 1,
    "local_author_window10_jaccard_max": 1,
    "local_author_window10_overlap_count_max": 1,
    "best_author_count_log_absdiff": -1,
    "top5_mean_distance": -1,
    "retrieval_rank": -1,
    "retrieval_reciprocal_rank": 1,
    "candidate_year_span": 0,
    "year_gap_to_candidate_range": -1,
    "year_gap_signed_to_candidate_range": 0,
    "candidate_dominant_first_name_length": 0,
    "query_first_prefix_match_any_length": 1,
    "same_dominant_first_as_best_top5": 1,
    "same_family_as_heuristic_choice": 1,
    "candidate_cluster_max_paper_author_count": 0,
    "anchor_evidence_count": 1,
    "strong_positive_anchor_score": 1,
    "weak_residual_anchor_score": 1,
    "sparse_relative_winner_score": 1,
    "last_first_name_count_min_rarity": 1,
    "last_name_count_min_rarity": 1,
    "pw_max_affiliation_overlap": 1,
    "pw_max_middle_initials_overlap": 1,
    "pw_mean_email_prefix_equal": 1,
    "pw_mean_first_names_equal": 1,
    "pw_min_middle_initials_overlap": 1,
    "pw_max_title_overlap_words": 1,
    "pw_max_journal_overlap": 1,
    "pw_mean_middle_names_equal": 1,
    "pw_min_last_first_name_count_max": 0,
    "pw_mean_coauthor_match": 1,
    "pw_mean_coauthor_overlap": 1,
    "pw_mean_title_overlap_words": 1,
    "pw_max_venue_overlap": 1,
    "pw_mean_journal_overlap": 1,
    "pw_min_specter_cosine_sim": 1,
    "pw_min_first_name_count_max": 0,
    "pw_max_coauthor_overlap": 1,
    "pw_max_jaro": 1,
    "pw_min_first_name_count_min": 0,
    "pw_min_levenshtein": -1,
    "pw_mean_english_count": 0,
    "pw_mean_middle_one_missing": 0,
    "pw_mean_specter_cosine_sim": 1,
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
    missing_feature_columns = [column for column in feature_columns if column not in df.columns]
    if missing_feature_columns:
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


def _fixed_probability_threshold_grid(step: float) -> np.ndarray:
    """Return an inclusive fixed probability grid from 0.0 to 1.0."""

    step = float(step)
    if not 0.0 < step <= 1.0:
        raise ValueError("classic.promoted_stratified_gate.fixed_grid_step must be in (0, 1]")
    interval_count = int(round(1.0 / step))
    if not np.isclose(float(interval_count) * step, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("classic.promoted_stratified_gate.fixed_grid_step must evenly divide 1.0")
    return np.round(np.linspace(0.0, 1.0, interval_count + 1, dtype=np.float64), 6)


def _total_error_components(rows: pd.DataFrame, link: np.ndarray) -> dict[str, np.ndarray]:
    """Count error components for one or more link/abstain decisions."""

    query_target = rows["query_safe_target"].to_numpy(dtype=np.int8, copy=False)
    chosen_target = rows["chosen_candidate_target"].to_numpy(dtype=np.int8, copy=False)
    matrix = np.asarray(link, dtype=bool)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    false_abstain = ((~matrix) & (query_target[:, None] == 1)).sum(axis=0).astype(np.int64)
    false_link = (matrix & (query_target[:, None] == 0)).sum(axis=0).astype(np.int64)
    wrong_candidate_link = (
        (matrix & (query_target[:, None] == 1) & (chosen_target[:, None] == 0)).sum(axis=0).astype(np.int64)
    )
    return {
        "false_abstain": false_abstain,
        "false_link": false_link,
        "wrong_candidate_link": wrong_candidate_link,
        "errors": false_abstain + false_link + wrong_candidate_link,
    }


def _weighted_error_counts_from_components(
    components: Mapping[str, np.ndarray],
    *,
    error_weights: Mapping[str, float] = _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS,
) -> np.ndarray:
    """Return weighted error counts from vectorized error components."""

    return (
        float(error_weights["false_abstain"]) * np.asarray(components["false_abstain"], dtype=np.float64)
        + float(error_weights["false_link"]) * np.asarray(components["false_link"], dtype=np.float64)
        + float(error_weights["wrong_candidate_link"])
        * np.asarray(components["wrong_candidate_link"], dtype=np.float64)
    )


def _best_total_error_threshold_index(
    components: Mapping[str, np.ndarray],
    *,
    score_thresholds: np.ndarray,
    error_weights: Mapping[str, float],
    margin_thresholds: np.ndarray | None = None,
) -> int:
    """Select the best fixed-grid threshold by weighted error with deterministic ties."""

    weighted_errors = _weighted_error_counts_from_components(components, error_weights=error_weights)
    tie_keys: list[np.ndarray] = [np.asarray(score_thresholds, dtype=np.float64)]
    if margin_thresholds is not None:
        tie_keys.append(np.asarray(margin_thresholds, dtype=np.float64))
    ranking = np.lexsort(
        tuple(
            [
                *tie_keys,
                np.asarray(components["false_abstain"], dtype=np.int64),
                np.asarray(components["false_link"], dtype=np.int64),
                np.asarray(components["wrong_candidate_link"], dtype=np.int64),
                weighted_errors,
            ]
        )
    )
    return int(ranking[0])


def _total_error_fit_metrics(
    rows: pd.DataFrame,
    components: Mapping[str, np.ndarray],
    best_index: int,
    *,
    error_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Return scalar fit metrics for the selected threshold candidate."""

    false_abstain = int(np.asarray(components["false_abstain"])[best_index])
    false_link = int(np.asarray(components["false_link"])[best_index])
    wrong_candidate_link = int(np.asarray(components["wrong_candidate_link"])[best_index])
    weighted_error_count = float(
        _weighted_error_counts_from_components(components, error_weights=error_weights)[best_index]
    )
    return {
        "n_queries": int(len(rows)),
        "errors": int(np.asarray(components["errors"])[best_index]),
        "false_abstain": false_abstain,
        "false_link": false_link,
        "wrong_candidate_link": wrong_candidate_link,
        "weighted_error_count": weighted_error_count,
        **_weighted_error_metrics(
            n_queries=int(len(rows)),
            false_abstain=false_abstain,
            false_link=false_link,
            wrong_candidate_link=wrong_candidate_link,
            error_weights=error_weights,
        ),
    }


def _fit_total_error_single_score(
    rows: pd.DataFrame,
    *,
    threshold_grid: np.ndarray,
    error_weights: Mapping[str, float],
) -> tuple[float, dict[str, Any]]:
    """Fit a score-only threshold on a fixed probability grid."""

    if rows.empty:
        return float(threshold_grid[-1]), {
            "n_queries": 0,
            "errors": 0,
            "false_abstain": 0,
            "false_link": 0,
            "wrong_candidate_link": 0,
            "weighted_error_count": 0.0,
            **_weighted_error_metrics(
                n_queries=0,
                false_abstain=0,
                false_link=0,
                wrong_candidate_link=0,
                error_weights=error_weights,
            ),
        }
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    links = score[:, None] >= threshold_grid[None, :]
    components = _total_error_components(rows, links)
    best_index = _best_total_error_threshold_index(
        components,
        score_thresholds=threshold_grid,
        error_weights=error_weights,
    )
    return float(threshold_grid[best_index]), _total_error_fit_metrics(
        rows,
        components,
        best_index,
        error_weights=error_weights,
    )


def _fit_total_error_score_margin(
    rows: pd.DataFrame,
    *,
    threshold_grid: np.ndarray,
    error_weights: Mapping[str, float],
) -> tuple[float, float, dict[str, Any]]:
    """Fit a score-or-margin threshold pair on a fixed probability grid."""

    if rows.empty:
        empty_metrics = {
            "n_queries": 0,
            "errors": 0,
            "false_abstain": 0,
            "false_link": 0,
            "wrong_candidate_link": 0,
            "weighted_error_count": 0.0,
            **_weighted_error_metrics(
                n_queries=0,
                false_abstain=0,
                false_link=0,
                wrong_candidate_link=0,
                error_weights=error_weights,
            ),
        }
        return float(threshold_grid[-1]), float(threshold_grid[-1]), empty_metrics
    score = pd.to_numeric(rows["chosen_probability"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    margin = pd.to_numeric(rows["score_margin"], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    best_key: tuple[float, int, int, int, float, float] | None = None
    best_score_threshold = float(threshold_grid[-1])
    best_margin_threshold = float(threshold_grid[-1])
    best_metrics: dict[str, Any] | None = None
    for score_threshold in threshold_grid:
        links = (score[:, None] >= float(score_threshold)) | (margin[:, None] >= threshold_grid[None, :])
        components = _total_error_components(rows, links)
        score_thresholds = np.repeat(float(score_threshold), len(threshold_grid))
        best_index = _best_total_error_threshold_index(
            components,
            score_thresholds=score_thresholds,
            margin_thresholds=threshold_grid,
            error_weights=error_weights,
        )
        metrics = _total_error_fit_metrics(rows, components, best_index, error_weights=error_weights)
        key = (
            float(metrics["weighted_error_count"]),
            int(metrics["wrong_candidate_link"]),
            int(metrics["false_link"]),
            int(metrics["false_abstain"]),
            float(score_threshold),
            float(threshold_grid[best_index]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_score_threshold = float(score_threshold)
            best_margin_threshold = float(threshold_grid[best_index])
            best_metrics = metrics
    if best_metrics is None:
        raise ValueError("Unable to fit promoted score/margin gate on non-empty calibration rows")
    return best_score_threshold, best_margin_threshold, best_metrics


def _fit_total_error_gate(
    calibration_rows: pd.DataFrame,
    *,
    fixed_grid_step: float,
    error_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Fit the promoted 4-score/2-margin gate on all calibration rows."""

    threshold_grid = _fixed_probability_threshold_grid(fixed_grid_step)
    labels = _total_error_gate_bucket(calibration_rows)
    score_thresholds: dict[str, float] = {}
    margin_thresholds: dict[str, float] = {}
    bucket_metrics: dict[str, dict[str, Any]] = {}
    for bucket in _TOTAL_ERROR_SCORE_BUCKETS:
        rows = calibration_rows[labels == bucket].copy()
        if bucket in _TOTAL_ERROR_MARGIN_BUCKETS:
            score_threshold, margin_threshold, metrics = _fit_total_error_score_margin(
                rows,
                threshold_grid=threshold_grid,
                error_weights=error_weights,
            )
            score_thresholds[bucket] = score_threshold
            margin_thresholds[bucket] = margin_threshold
        else:
            score_threshold, metrics = _fit_total_error_single_score(
                rows,
                threshold_grid=threshold_grid,
                error_weights=error_weights,
            )
            score_thresholds[bucket] = score_threshold
        bucket_metrics[bucket] = {
            "score_threshold": float(score_thresholds[bucket]),
            "margin_threshold": float(margin_thresholds[bucket]) if bucket in margin_thresholds else None,
            **metrics,
        }
    return {
        "gate": TotalErrorGateSpec(
            name=f"full_calibration_fixed_grid_{float(fixed_grid_step):g}",
            score_thresholds=score_thresholds,
            margin_thresholds=margin_thresholds,
        ),
        "bucket_metrics": bucket_metrics,
        "threshold_grid_points": int(len(threshold_grid)),
    }


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


def _fit_promoted_stratified_total_error_gate(
    choices: pd.DataFrame,
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    """Fit the promoted bucketed gate on all configured calibration splits."""

    calibration_splits = tuple(str(split) for split in gate_config["calibration_splits"])
    calibration_rows = choices[choices["split"].astype(str).isin(calibration_splits)].copy()
    if calibration_rows.empty:
        raise ValueError(
            "Promoted stratified gate requires non-empty calibration splits: " f"splits={list(calibration_splits)}"
        )
    error_weights = dict(gate_config.get("error_weights", _DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS))
    fit_result = _fit_total_error_gate(
        calibration_rows,
        fixed_grid_step=float(gate_config["fixed_grid_step"]),
        error_weights=error_weights,
    )
    selected_gate = fit_result["gate"]
    calibration_predictions = _apply_total_error_gate(calibration_rows, selected_gate)
    calibration_metrics = _summarize_predictions(calibration_predictions)
    split_series = choices["split"].astype(str)
    split_metrics = {
        split: _summarize_predictions(_apply_total_error_gate(choices[split_series == split].copy(), selected_gate))
        for split in calibration_splits
    }
    return {
        "gate": selected_gate,
        "calibration_metrics": calibration_metrics,
        "calibration_split_metrics": split_metrics,
        "bucket_metrics": dict(fit_result["bucket_metrics"]),
        "threshold_grid_points": int(fit_result["threshold_grid_points"]),
        "selection_key": {
            "calibration_weighted_average_error": float(calibration_metrics["weighted_average_error"]),
            "calibration_false_abstain_error_rate": float(calibration_metrics["false_abstain_error_rate"]),
            "calibration_false_link_error_rate": float(calibration_metrics["false_link_error_rate"]),
            "calibration_wrong_link_error_rate": float(calibration_metrics["wrong_link_error_rate"]),
            "error_weights": error_weights,
            "calibration_errors": int(calibration_metrics["errors"]),
            "calibration_wrong_candidate_link": int(calibration_metrics["wrong_candidate_link"]),
            "calibration_false_link": int(calibration_metrics["false_link"]),
            "calibration_false_abstain": int(calibration_metrics["false_abstain"]),
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
        calibration_splits_value = promoted_gate.get("calibration_splits", ["calibration_fit", "calibration_check"])
        if isinstance(calibration_splits_value, str) or not isinstance(calibration_splits_value, Sequence):
            calibration_splits = ["calibration_fit", "calibration_check"]
        else:
            calibration_splits = [str(split) for split in calibration_splits_value]
        fit_split = calibration_splits[0] if calibration_splits else "calibration_fit"
        check_split = calibration_splits[1] if len(calibration_splits) > 1 else ""
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
        calibration_splits = tuple(str(split) for split in promoted_gate_config["calibration_splits"])
        calibration_split_label = "+".join(calibration_splits)
        calibration_metrics = {
            "split": calibration_split_label,
            "score_threshold": score_threshold,
            "margin_threshold": margin_threshold,
            **dict(selected_gate_result["calibration_metrics"]),
        }
        calibration_predictions = _apply_total_error_gate(
            stratified_scored_choices[0][
                stratified_scored_choices[0]["split"].astype(str).isin(calibration_splits)
            ].copy(),
            selected_gate,
        )
        single_candidate_predictions = calibration_predictions[
            (pd.to_numeric(calibration_predictions["has_runner_up"], errors="coerce").fillna(0).astype(int) == 0)
            | calibration_predictions["score_margin"].isna()
        ].copy()
        single_candidate_calibration_metrics = {
            "single_candidate_score_threshold": single_candidate_score_threshold,
            **_summarize_predictions(single_candidate_predictions),
        }
        promoted_gate_summary = {
            "mode": str(promoted_gate_config["mode"]),
            "calibration_splits": list(calibration_splits),
            "test_split": str(promoted_gate_config["test_split"]),
            "fixed_grid_step": float(promoted_gate_config["fixed_grid_step"]),
            "threshold_grid_points": int(selected_gate_result["threshold_grid_points"]),
            "selection_metric": str(promoted_gate_config["selection_metric"]),
            "error_weights": dict(promoted_gate_config["error_weights"]),
            "selected_gate": {
                "name": selected_gate.name,
                "score_thresholds": dict(selected_gate.score_thresholds),
                "margin_thresholds": dict(selected_gate.margin_thresholds),
            },
            "selection_key": dict(selected_gate_result["selection_key"]),
            "calibration_metrics": dict(selected_gate_result["calibration_metrics"]),
            "calibration_split_metrics": dict(selected_gate_result["calibration_split_metrics"]),
            "bucket_metrics": dict(selected_gate_result["bucket_metrics"]),
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
                "promoted_stratified_full_calibration_fixed_grid_4score_2margin"
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


PROMOTED_PAIRWISE_COLUMNS = promoted_pairwise_aggregate_columns()
PROMOTED_NON_PAIRWISE_COLUMNS = tuple(PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
PROMOTED_FEATURE_COLUMNS = promoted_linker_feature_columns()
SUPPORTED_PROMOTED_FEATURE_COLUMNS = frozenset(PROMOTED_NON_PAIRWISE_COLUMNS) | frozenset(PROMOTED_PAIRWISE_COLUMNS)
FROZEN_RETRIEVAL_POLICY = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY
FROZEN_RETRIEVAL_POLICY_NAME = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME
WEIGHTED_ERROR_WEIGHTS = {
    "false_abstain_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["false_abstain"]),
    "false_link_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["false_link"]),
    "wrong_link_error_rate": float(_DEFAULT_PROMOTED_GATE_ERROR_WEIGHTS["wrong_candidate_link"]),
}
NAN_POLICY_CHOICES = ("preserve", "zero")
ROW_NAN_POLICY_CHOICES = ("finite", "semantic")


@dataclass(frozen=True)
class ComponentMembers:
    """Candidate component members in both raw-id and Rust-index forms."""

    signature_ids: tuple[str, ...]
    signature_id_set: frozenset[str]
    signature_indices: np.ndarray


@dataclass
class MinimalRawDatasetContext:
    """Loaded raw dataset state shared across all linker row tables for one dataset."""

    dataset_name: str
    row_component_scope: str
    pairwise_component_scope: str
    dataset: ANDData
    runtime_context: Any
    constraint_backend: Any
    featurizer: Any
    signature_id_to_index: dict[str, int]
    component_details: dict[str, ComponentMembers]
    component_indices: dict[str, np.ndarray]
    pairwise_component_details: dict[str, ComponentMembers]
    pairwise_component_indices: dict[str, np.ndarray]
    component_keys_by_block: dict[str, tuple[str, ...]]
    feature_cache: dict[str, retrieval.QueryFeatures]
    paper_author_name_cache: dict[str, frozenset[str]]
    full_summary_cache: dict[str, retrieval.ClusterSummary]
    residual_summary_cache: dict[tuple[str, str], retrieval.ClusterSummary]
    rust_hybrid_centroid_retriever: Any
    retrieval_subblock_index: dict[str, Any]
    max_block_component_size: int


@dataclass
class MinimalRawPendingShard:
    """One table/dataset slice that still needs feature materialization."""

    table_key: str
    dataset_name: str
    rows: pd.DataFrame
    row_positions: np.ndarray
    partial_path: Path


@dataclass
class MinimalRawTablePlan:
    """Materialization state for one output feature table."""

    table_key: str
    labels_path: Path
    output_path: Path
    labels: pd.DataFrame
    required_output_columns: list[str]
    partial_dir: Path
    partial_paths: list[Path]
    dataset_summaries: list[dict[str, Any]]
    structural_cleaning_summary: dict[str, Any]
    started: float


@dataclass(frozen=True)
class FusedDistanceStats:
    """Distance-summary adapter for `_fill_row_signal`."""

    count: int
    min_distance: float
    mean_distance: float
    top3_mean_distance: float
    top5_mean_distance: float

    def topk_mean_distance(self, top_k: int) -> float:
        if int(top_k) <= 3:
            return float(self.top3_mean_distance)
        return float(self.top5_mean_distance)


@dataclass(frozen=True)
class ProdTrainingData:
    """Final production fit rows and per-row weights."""

    rows: pd.DataFrame
    sample_weight: np.ndarray
    source_summaries: list[dict[str, Any]]
    train_holdout_filter_summary: dict[str, Any]
    train_filter_summary: dict[str, Any] | None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_target(path: Path) -> dict[str, Any]:
    target = json.loads(path.read_text(encoding="utf-8"))
    features = tuple(str(feature) for feature in target["features"])
    if len(features) != int(target["feature_count"]):
        raise ValueError(f"Promoted target feature_count mismatch in {path}")
    unknown_pw = sorted(
        feature for feature in features if feature.startswith("pw_") and feature not in PROMOTED_PAIRWISE_COLUMNS
    )
    unknown_non_pw = sorted(
        feature
        for feature in features
        if not feature.startswith("pw_") and feature not in PROMOTED_NON_PAIRWISE_COLUMNS
    )
    if unknown_pw or unknown_non_pw:
        raise ValueError(f"Promoted target contains unknown features: {unknown_pw[:5] + unknown_non_pw[:5]}")
    unsupported = sorted(set(features) - SUPPORTED_PROMOTED_FEATURE_COLUMNS)
    if unsupported:
        raise ValueError(f"Promoted target contains unsupported features: {unsupported[:5]}")
    return target


def _target_expected_metrics(target: Mapping[str, Any]) -> dict[str, float]:
    metrics = dict(target.get("metrics", {}))
    return {
        key: float(metrics[key])
        for key in (
            "stratified_test_accuracy",
            "stratified_test_balanced_accuracy",
            "stratified_test_error_rate",
        )
        if key in metrics
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_production_model_path(path: Path) -> str | None:
    name = path.name
    prefix = "production_model_v"
    suffix = ".pickle"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return None


def _portable_repo_path(path: Path) -> str:
    raw_path = Path(path)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def _strip_local_paths(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): _strip_local_paths(value)
            for key, value in payload.items()
            if "path" not in str(key).lower() and "root" not in str(key).lower()
        }
    if isinstance(payload, list):
        return [_strip_local_paths(value) for value in payload]
    return payload


def _linker_artifact_audit_metadata(
    *,
    args: argparse.Namespace,
    target: Mapping[str, Any],
    feature_bundle: OfficialBundle,
    featureization_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairwise_model_path = Path(args.pairwise_model_path)
    pairwise_model = {
        "path": _portable_repo_path(pairwise_model_path),
        "filename": pairwise_model_path.name,
        "version": _version_from_production_model_path(pairwise_model_path),
    }
    if pairwise_model_path.exists():
        pairwise_model["sha256"] = _sha256_file(pairwise_model_path)
    return {
        "artifact_name": "production_incremental_linker",
        "artifact_version": str(args.linker_artifact_version),
        "target_variant": str(target.get("variant", "")),
        "target_status": str(target.get("status", "")),
        "target_metrics": dict(target.get("metrics", {})),
        "pairwise_model": pairwise_model,
        "training_source_bundle": _portable_repo_path(Path(args.source_bundle_root)),
        "training_feature_mode": str(args.feature_mode),
        "precomputed_feature_bundle": (
            _portable_repo_path(Path(args.precomputed_feature_bundle_root))
            if args.precomputed_feature_bundle_root is not None
            else None
        ),
        "training_feature_bundle_name": str(feature_bundle.bundle_name),
        "feature_nan_policy": _feature_nan_policy_summary(args),
        "featureization": [_strip_local_paths(dict(summary)) for summary in featureization_summaries],
        "target_spec": _portable_repo_path(Path(args.target_json)),
    }


def _bundle_with_promoted_target(bundle: OfficialBundle, target: Mapping[str, Any]) -> OfficialBundle:
    models = copy.deepcopy(bundle.models)
    classic = dict(models["classic"])
    classic["feature_columns"] = list(target["features"])
    classic["best_params"] = dict(target["params"])
    models["classic"] = classic
    feature_count = int(target["feature_count"])
    tree_count = int(target["params"]["n_estimators"])
    return OfficialBundle(
        root=bundle.root,
        bundle_name=f"{bundle.bundle_name}_promoted_{feature_count}_{tree_count}trees",
        assets=copy.deepcopy(bundle.assets),
        models=models,
        expected_metrics={"classic": _target_expected_metrics(target)},
    )


def _bundle_with_classic_params(bundle: OfficialBundle, params: Mapping[str, Any]) -> OfficialBundle:
    models = copy.deepcopy(bundle.models)
    classic = dict(models["classic"])
    classic["best_params"] = dict(params)
    models["classic"] = classic
    return OfficialBundle(
        root=bundle.root,
        bundle_name=bundle.bundle_name,
        assets=copy.deepcopy(bundle.assets),
        models=models,
        expected_metrics=copy.deepcopy(bundle.expected_metrics),
    )


def _intify_hyperopt_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return int(value) if float(value).is_integer() else float(value)
    if hasattr(value, "is_integer") and value.is_integer():
        return int(value)
    return value


def _normalize_hyperopt_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _intify_hyperopt_value(value) for key, value in params.items()}


def _classic_hyperopt_search_space(base_params: Mapping[str, Any]) -> dict[str, Any]:
    from hyperopt import hp
    from hyperopt.pyll import scope

    space = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "learning_rate": hp.loguniform("learning_rate", math.log(0.005), math.log(0.08)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 16, 1)),
        "min_child_samples": scope.int(hp.qloguniform("min_child_samples", math.log(50), math.log(2000), 1)),
        "min_child_weight": hp.loguniform("min_child_weight", math.log(1e-3), math.log(10.0)),
        "min_split_gain": hp.uniform("min_split_gain", 0.0, 1.0),
        "n_estimators": scope.int(hp.quniform("n_estimators", 300, 1200, 25)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", math.log(31), math.log(512), 1)),
        "reg_alpha": hp.loguniform("reg_alpha", math.log(1e-4), math.log(32.0)),
        "reg_lambda": hp.loguniform("reg_lambda", math.log(1e-4), math.log(64.0)),
        "subsample": hp.uniform("subsample", 0.7, 1.0),
        "subsample_freq": scope.int(hp.quniform("subsample_freq", 0, 3, 1)),
    }
    return {key: value for key, value in space.items() if key in base_params}


def _hyperopt_loss(summary: Mapping[str, Any], metric: str) -> float:
    observed = _observed_official_metrics(summary)
    if metric == "weighted_average_error":
        return float(observed["weighted_average_error"])
    if metric == "stratified_test_errors":
        return float(observed["stratified_test_errors"])
    if metric == "stratified_test_error_rate":
        return float(observed["stratified_test_error_rate"])
    if metric == "stratified_test_balanced_accuracy":
        return -float(observed["stratified_test_balanced_accuracy"])
    raise ValueError(f"Unsupported hyperopt metric: {metric}")


def _run_classic_hyperopt(
    *,
    feature_bundle: OfficialBundle,
    output_dir: Path,
    base_params: Mapping[str, Any],
    hyperopt_evals: int,
    metric: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune classic LightGBM params by running the full train/calibrate/eval stack."""

    from hyperopt import STATUS_OK, Trials, fmin, tpe

    if int(hyperopt_evals) <= 0:
        raise ValueError("hyperopt_evals must be positive when hyperopt is enabled")
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = _classic_hyperopt_search_space(base_params)
    if not search_space:
        raise ValueError("No tunable LightGBM parameters were present in base_params")
    trial_records: list[dict[str, Any]] = []

    def evaluate_params(resolved_params: Mapping[str, Any], *, source: str) -> float:
        trial_index = len(trial_records)
        trial_output_dir = output_dir / f"trial_{trial_index:03d}"
        print(
            json.dumps(
                {
                    "event": "classic_hyperopt_trial_start",
                    "trial": trial_index,
                    "source": source,
                    "output_dir": str(trial_output_dir),
                    "metric": metric,
                    "params": dict(resolved_params),
                }
            ),
            flush=True,
        )
        trial_summary = run_classic(
            _bundle_with_classic_params(feature_bundle, dict(resolved_params)),
            trial_output_dir,
        )
        observed = _observed_official_metrics(trial_summary)
        loss = _hyperopt_loss(trial_summary, metric)
        record = {
            "trial": trial_index,
            "source": source,
            "loss": float(loss),
            "metric": metric,
            "params": dict(resolved_params),
            "observed_metrics": observed,
            "classic_summary_path": str(trial_output_dir / "summary.json"),
        }
        trial_records.append(record)
        _write_json(output_dir / "trials.json", trial_records)
        print(json.dumps({"event": "classic_hyperopt_trial_done", **record}), flush=True)
        return float(loss)

    baseline_loss = evaluate_params(dict(base_params), source="base_params")

    def objective(params: Mapping[str, Any]) -> dict[str, Any]:
        resolved_params = dict(base_params)
        resolved_params.update(_normalize_hyperopt_params(params))
        loss = evaluate_params(resolved_params, source="tpe")
        return {"loss": float(loss), "status": STATUS_OK}

    trials = Trials()
    search_evals = max(0, int(hyperopt_evals) - 1)
    if search_evals:
        _ = fmin(
            fn=objective,
            space=search_space,
            algo=partial(tpe.suggest, n_startup_jobs=min(5, int(search_evals))),
            max_evals=int(search_evals),
            trials=trials,
            rstate=np.random.default_rng(int(seed)),
        )
    best_record = min(trial_records, key=lambda record: float(record["loss"]))
    best_params = dict(best_record["params"])
    summary = {
        "enabled": True,
        "hyperopt_evals": int(hyperopt_evals),
        "hyperopt_search_evals": int(search_evals),
        "hyperopt_trials_ran": int(len(trial_records)),
        "metric": metric,
        "seed": int(seed),
        "base_loss": float(baseline_loss),
        "best_loss": float(best_record["loss"]),
        "best_trial": int(best_record["trial"]),
        "best_source": str(best_record["source"]),
        "best_params": best_params,
        "trials_path": str(output_dir / "trials.json"),
    }
    _write_json(output_dir / "summary.json", summary)
    return best_params, summary


def _classic_table_keys(spec: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = [key for key in REQUIRED_TABLE_KEYS if key in spec]
    for optional_key in ("s_park_eval_path", "s_lee_eval_path"):
        if optional_key in spec:
            keys.append(optional_key)
    extra_eval_paths = spec.get("extra_eval_paths", {})
    if extra_eval_paths is not None:
        if not isinstance(extra_eval_paths, Mapping):
            raise ValueError("classic.extra_eval_paths must be a mapping")
        for dataset_name in extra_eval_paths:
            keys.append(f"extra_eval_paths.{dataset_name}")
    return tuple(dict.fromkeys(keys))


def _asset_file(bundle: OfficialBundle, asset_group: str, table_key: str) -> Path:
    files = dict(bundle.assets[asset_group]["files"])
    if table_key not in files:
        raise KeyError(f"Bundle asset group {asset_group!r} has no file for {table_key!r}")
    return _resolve_path(bundle, str(files[table_key]))


def _output_table_relpath(table_key: str, labels_path: Path) -> Path:
    if table_key.startswith("extra_eval_paths."):
        return Path("features_corrected") / labels_path.name
    return Path("features_corrected") / labels_path.name


def _selected_row_positions(labels: pd.DataFrame, datasets: set[str] | None, limit_rows: int | None) -> np.ndarray:
    mask = np.ones(len(labels), dtype=bool)
    if datasets is not None:
        mask &= labels["dataset"].astype(str).isin(datasets).to_numpy()
    positions = np.flatnonzero(mask)
    if limit_rows is not None:
        positions = positions[: int(limit_rows)]
    return positions.astype(np.int64, copy=False)


def _read_selected_rows(
    *,
    labels_path: Path,
    corrected_path: Path,
    datasets: set[str] | None,
    limit_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(labels_path)
    corrected = pd.read_parquet(corrected_path)
    if len(labels) != len(corrected):
        raise ValueError(f"labels/corrected row count mismatch: {labels_path} {len(labels)} != {len(corrected)}")
    positions = _selected_row_positions(labels, datasets, limit_rows)
    labels = labels.iloc[positions].reset_index(drop=True)
    corrected = corrected.iloc[positions].reset_index(drop=True)
    for column in ("dataset", "query_group_id", "candidate_component_key", "retrieval_rank", "label"):
        if column not in corrected.columns:
            continue
        left = labels[column].astype(str).to_numpy()
        right = corrected[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"labels/corrected identity mismatch for {column!r} in {labels_path.name}")
    return labels, corrected


def _load_raw_signature_blocks(bundle: OfficialBundle, dataset_name: str) -> dict[str, str]:
    raw_datasets = dict(bundle.assets["raw_metadata"]["datasets"])
    if dataset_name not in raw_datasets:
        raise KeyError(f"Minimal raw metadata is missing dataset {dataset_name!r}")
    raw_spec = dict(raw_datasets[dataset_name])
    signatures_path = _resolve_path(bundle, str(raw_spec["signatures_path"]))
    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    return {
        str(signature_id): str((signature.get("author_info") or {}).get("block", ""))
        for signature_id, signature in signatures.items()
    }


def _minimal_raw_component_membership_summary(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if dataset_name in cache:
        return cache[dataset_name]
    member_datasets = dict(bundle.assets["candidate_members"]["datasets"])
    if dataset_name not in member_datasets:
        raise KeyError(f"Candidate member metadata is missing dataset {dataset_name!r}")
    path = _resolve_path(bundle, str(member_datasets[dataset_name]))
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    component_keys = members["candidate_component_key"].astype(str)
    signature_to_block: dict[str, str] = {}
    if component_keys.str.contains("::", regex=False).any():
        signature_to_block = _load_raw_signature_blocks(bundle, dataset_name)

    rows: list[dict[str, Any]] = []
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        member_ids = tuple(str(value) for value in group.sort_values("member_index")["signature_id"].astype(str))
        member_ids = _block_local_member_ids_from_signature_blocks(str(component_key), member_ids, signature_to_block)
        rows.append(
            {
                "candidate_component_key": str(component_key),
                "_component_member_count": int(len(member_ids)),
                "_component_single_member_signature_id": member_ids[0] if len(member_ids) == 1 else None,
            }
        )
    summary = pd.DataFrame(rows)
    cache[dataset_name] = summary
    return summary


def _clean_minimal_raw_structural_rows(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    rows: pd.DataFrame,
    component_membership_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove candidate rows with no non-query member under the block-local contract."""

    required = {"dataset", "query_group_id", "query_signature_id", "candidate_component_key", "label"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{table_key}: minimal raw structural cleaning missing columns: {missing}")
    started = time.perf_counter()
    keep_mask = np.ones(len(rows), dtype=bool)
    labels = pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(np.int8)
    query_ids_before = set(rows["query_group_id"].astype(str))
    positive_query_ids_before = set(rows.loc[labels == 1, "query_group_id"].astype(str))
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_name, dataset_rows in rows.groupby(rows["dataset"].astype(str), sort=False):
        membership = _minimal_raw_component_membership_summary(
            source_bundle,
            str(dataset_name),
            cache=component_membership_cache,
        )
        local = dataset_rows[["candidate_component_key", "query_signature_id", "label"]].copy()
        local["candidate_component_key"] = local["candidate_component_key"].astype(str)
        local["query_signature_id"] = local["query_signature_id"].astype(str)
        local["_global_index"] = dataset_rows.index.to_numpy(dtype=np.int64)
        local = local.merge(membership, on="candidate_component_key", how="left", validate="many_to_one")
        if local["_component_member_count"].isna().any():
            missing_keys = sorted(
                set(local.loc[local["_component_member_count"].isna(), "candidate_component_key"].astype(str))
            )
            raise KeyError(
                f"{table_key} {dataset_name}: candidate components missing member metadata: {missing_keys[:10]}"
            )
        local_label = pd.to_numeric(local["label"], errors="coerce").fillna(0).astype(np.int8)
        drop = (local["_component_member_count"].astype(np.int64) == 1) & local[
            "_component_single_member_signature_id"
        ].astype(str).eq(local["query_signature_id"].astype(str))
        drop_indices = local.loc[drop, "_global_index"].to_numpy(dtype=np.int64, copy=False)
        keep_mask[drop_indices] = False
        dataset_summaries.append(
            {
                "dataset": str(dataset_name),
                "rows_before": int(len(dataset_rows)),
                "rows_removed": int(drop.sum()),
                "positive_rows_removed": int((drop & (local_label == 1)).sum()),
                "negative_rows_removed": int((drop & (local_label == 0)).sum()),
            }
        )

    cleaned = rows.loc[keep_mask].reset_index(drop=True)
    cleaned_labels = pd.to_numeric(cleaned["label"], errors="coerce").fillna(0).astype(np.int8)
    query_ids_after = set(cleaned["query_group_id"].astype(str))
    positive_query_ids_after = set(cleaned.loc[cleaned_labels == 1, "query_group_id"].astype(str))
    summary = {
        "table_key": table_key,
        "policy": "drop_candidate_rows_with_no_non_query_block_local_members",
        "rows_before": int(len(rows)),
        "rows_after": int(len(cleaned)),
        "rows_removed": int(len(rows) - len(cleaned)),
        "positive_rows_removed": int((labels[~keep_mask] == 1).sum()),
        "negative_rows_removed": int((labels[~keep_mask] == 0).sum()),
        "queries_before": int(len(query_ids_before)),
        "queries_after": int(len(query_ids_after)),
        "queries_removed": int(len(query_ids_before - query_ids_after)),
        "positive_queries_before": int(len(positive_query_ids_before)),
        "positive_queries_after": int(len(positive_query_ids_after)),
        "positive_queries_changed_or_removed": int(len(positive_query_ids_before - positive_query_ids_after)),
        "datasets": dataset_summaries,
        "seconds": round(float(time.perf_counter() - started), 3),
    }
    return cleaned, summary


def _component_members_by_key(path: Path, signature_id_to_index: Mapping[str, int]) -> dict[str, np.ndarray]:
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    out: dict[str, np.ndarray] = {}
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        member_ids = group.sort_values("member_index")["signature_id"].astype(str)
        member_indices: list[int] = []
        for signature_id in member_ids:
            try:
                member_indices.append(int(signature_id_to_index[str(signature_id)]))
            except KeyError as exc:
                raise KeyError(f"component member signature_id missing from Rust featurizer: {signature_id}") from exc
        out[str(component_key)] = np.asarray(member_indices, dtype=np.uint32)
    return out


def _block_local_member_ids(
    dataset: ANDData,
    component_key: str,
    member_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return _block_local_member_ids_from_signature_blocks(
        component_key,
        member_ids,
        getattr(dataset, "signature_to_block", {}) or {},
    )


def _block_local_member_ids_from_signature_blocks(
    component_key: str,
    member_ids: tuple[str, ...],
    signature_to_block: Mapping[str, str],
) -> tuple[str, ...]:
    if "::" not in str(component_key):
        return member_ids
    block_key, _cluster_id = str(component_key).split("::", 1)
    filtered = tuple(
        signature_id for signature_id in member_ids if str(signature_to_block.get(str(signature_id), "")) == block_key
    )
    return filtered or member_ids


def _component_member_details_by_key(
    path: Path,
    signature_id_to_index: Mapping[str, int],
    *,
    dataset: ANDData,
    component_scope: str = "block-local",
) -> dict[str, ComponentMembers]:
    if component_scope not in {"frozen", "block-local"}:
        raise ValueError(f"unknown component_scope={component_scope!r}")
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    out: dict[str, ComponentMembers] = {}
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        member_ids = tuple(str(value) for value in group.sort_values("member_index")["signature_id"].astype(str))
        if component_scope == "block-local":
            member_ids = _block_local_member_ids(dataset, str(component_key), member_ids)
        member_indices: list[int] = []
        for signature_id in member_ids:
            try:
                member_indices.append(int(signature_id_to_index[str(signature_id)]))
            except KeyError as exc:
                raise KeyError(f"component member signature_id missing from Rust featurizer: {signature_id}") from exc
        out[str(component_key)] = ComponentMembers(
            signature_ids=member_ids,
            signature_id_set=frozenset(member_ids),
            signature_indices=np.asarray(member_indices, dtype=np.uint32),
        )
    return out


def _enable_fasttext_language_detection() -> None:
    os.environ["S2AND_SKIP_FASTTEXT"] = "0"
    s2and_text._FASTTEXT_MODEL = None  # noqa: SLF001
    s2and_text._FASTTEXT_MODEL_INITIALIZED = False  # noqa: SLF001


def _signature_id_to_index(featurizer: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, signature_id in enumerate(featurizer.signature_ids()):
        out[str(signature_id)] = int(index)
    return out


def _candidate_batch_from_rows(
    rows: pd.DataFrame,
    component_members: Mapping[str, np.ndarray],
    signature_id_to_index: Mapping[str, int],
    *,
    row_group_ids: Sequence[int] | None = None,
) -> LinkerCandidateBatch:
    query_indices = np.empty(len(rows), dtype=np.uint32)
    member_arrays: list[np.ndarray] = []
    for row_offset, row in enumerate(rows.itertuples(index=False)):
        query_signature_id = str(row.query_signature_id)
        component_key = str(row.candidate_component_key)
        try:
            query_index = int(signature_id_to_index[query_signature_id])
        except KeyError as exc:
            raise KeyError(f"query_signature_id missing from Rust featurizer: {query_signature_id}") from exc
        try:
            members = component_members[component_key]
        except KeyError as exc:
            raise KeyError(f"candidate_component_key missing from members table: {component_key}") from exc
        query_indices[row_offset] = query_index
        active_members = members[members != query_index]
        member_arrays.append(np.ascontiguousarray(active_members, dtype=np.uint32))

    pair_count = int(sum(len(members) for members in member_arrays))
    left = np.empty(pair_count, dtype=np.uint32)
    right = np.empty(pair_count, dtype=np.uint32)
    owner_rows = np.empty(pair_count, dtype=np.uint32)
    offset = 0
    for row_offset, members in enumerate(member_arrays):
        stop = offset + len(members)
        left[offset:stop] = query_indices[row_offset]
        right[offset:stop] = members
        owner_rows[offset:stop] = row_offset
        offset = stop

    return LinkerCandidateBatch(
        row_count=len(rows),
        left_signature_indices=left,
        right_signature_indices=right,
        pair_row_indices=owner_rows,
        row_query_signature_indices=(
            np.asarray(row_group_ids, dtype=np.uint32) if row_group_ids is not None else query_indices
        ),
        row_component_keys=tuple(rows["candidate_component_key"].astype(str).tolist()),
        labels=rows["label"].to_numpy(dtype=np.int8, copy=False) if "label" in rows.columns else None,
        retrieval_scores=(
            rows["retrieval_score"].to_numpy(dtype=np.float32, copy=False)
            if "retrieval_score" in rows.columns
            else None
        ),
        retrieval_ranks=(
            rows["retrieval_rank"].to_numpy(dtype=np.uint16, copy=False) if "retrieval_rank" in rows.columns else None
        ),
    )


def _load_original_pairwise_dataset(dataset_name: str, *, n_jobs: int, giant_dataset_root: Path) -> Any:
    if dataset_name in GIANT_DATASETS:
        dataset_dir = giant_dataset_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Original giant dataset directory is required for exact recompute: {dataset_dir}")
        dataset, _load_info = load_giant_block_dataset(
            dataset_dir,
            block_key=None,
            n_jobs=int(n_jobs),
            load_name_counts=True,
        )
        return dataset
    return load_labeled_dataset(
        REPO_ROOT / "data",
        dataset_name,
        n_jobs=int(n_jobs),
        load_name_counts=True,
    )


def _load_featureless_raw_dataset(bundle: OfficialBundle, dataset_name: str, *, n_jobs: int) -> ANDData:
    raw_datasets = dict(bundle.assets["raw_metadata"]["datasets"])
    if dataset_name not in raw_datasets:
        raise KeyError(f"Featureless raw metadata is missing dataset {dataset_name!r}")
    raw_spec = dict(raw_datasets[dataset_name])
    signatures_path = _resolve_path(bundle, str(raw_spec["signatures_path"]))
    papers_path = _resolve_path(bundle, str(raw_spec["papers_path"]))
    return ANDData(
        str(signatures_path),
        str(papers_path),
        name=f"joint_safe_link_featureless_{dataset_name}",
        mode="inference",
        load_name_counts=True,
        preprocess=True,
        n_jobs=max(1, int(n_jobs)),
        compute_reference_features=False,
    )


def _load_minimal_raw_specter_dataset(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    clusterer: Any,
    n_jobs: int,
    rust_build_path: str | None,
) -> ANDData:
    raw_datasets = dict(bundle.assets["raw_metadata"]["datasets"])
    embedding_datasets = dict(bundle.assets.get("embeddings", {}).get("datasets", {}))
    if dataset_name not in raw_datasets:
        raise KeyError(f"Minimal raw metadata is missing dataset {dataset_name!r}")
    if dataset_name not in embedding_datasets:
        raise KeyError(f"Minimal SPECTER2 embeddings are missing dataset {dataset_name!r}")
    _enable_fasttext_language_detection()
    raw_spec = dict(raw_datasets[dataset_name])
    signatures_path = _resolve_path(bundle, str(raw_spec["signatures_path"]))
    papers_path = _resolve_path(bundle, str(raw_spec["papers_path"]))
    specter_path = _resolve_path(bundle, str(embedding_datasets[dataset_name]))
    dataset = ANDData(
        str(signatures_path),
        str(papers_path),
        name=f"joint_safe_link_minimal_raw_specter_{dataset_name}",
        mode="inference",
        specter_embeddings=str(specter_path),
        load_name_counts=True,
        preprocess=True,
        n_jobs=max(1, int(n_jobs)),
        compute_reference_features=False,
        use_orcid_id=False,
        use_sinonym_overwrite=False,
        name_tuples="filtered",
    )
    if rust_build_path is not None:
        dataset.rust_lifecycle_policy = replace(
            dataset.rust_lifecycle_policy,
            rust_build_path=rust_build_path,
        )
    _apply_dataset_name_count_semantics_for_prediction(clusterer, dataset)
    return dataset


def _build_full_retrieval_summary_cache(
    *,
    dataset: ANDData,
    component_details: Mapping[str, ComponentMembers],
    feature_cache: dict[str, retrieval.QueryFeatures],
    paper_author_name_cache: dict[str, frozenset[str]],
    max_exemplars: int,
) -> dict[str, retrieval.ClusterSummary]:
    summaries: dict[str, retrieval.ClusterSummary] = {}
    for component_key, details in component_details.items():
        summaries[str(component_key)] = _build_summary_for_members(
            dataset=dataset,
            component_key=str(component_key),
            candidate_cluster_id=None,
            signature_ids=details.signature_ids,
            feature_cache=feature_cache,
            paper_author_name_cache=paper_author_name_cache,
            max_exemplars=max_exemplars,
        )
    return summaries


def _component_block_key(
    dataset: ANDData,
    component_key: str,
    details: ComponentMembers,
) -> str:
    if "::" in str(component_key):
        return str(component_key).split("::", 1)[0]
    signature_to_block = getattr(dataset, "signature_to_block", {}) or {}
    for signature_id in details.signature_ids:
        block_key = str(signature_to_block.get(str(signature_id), ""))
        if block_key:
            return block_key
    return str(component_key)


def _build_retrieval_subblock_index_for_components(
    *,
    dataset: ANDData,
    component_details: Mapping[str, ComponentMembers],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[str, ...]]]:
    block_to_component_keys: dict[str, list[str]] = {}
    component_signatures: dict[str, list[str]] = {}
    for component_key, details in component_details.items():
        key = str(component_key)
        block_key = _component_block_key(dataset, key, details)
        block_to_component_keys.setdefault(block_key, []).append(key)
        component_signatures[key] = list(details.signature_ids)
    index, diagnostics = build_labeled_retrieval_subblock_index(
        dataset=dataset,
        block_to_component_keys=block_to_component_keys,
        component_signatures=component_signatures,
    )
    component_keys_by_block = {
        block_key: tuple(component_keys) for block_key, component_keys in block_to_component_keys.items()
    }
    return index, diagnostics, component_keys_by_block


def _build_minimal_raw_dataset_context(
    *,
    source_bundle: OfficialBundle,
    dataset_name: str,
    clusterer: Any,
    n_jobs: int,
    rust_build_path: str | None,
    max_exemplars: int,
) -> MinimalRawDatasetContext:
    started = time.perf_counter()
    dataset = _load_minimal_raw_specter_dataset(
        source_bundle,
        dataset_name,
        clusterer=clusterer,
        n_jobs=n_jobs,
        rust_build_path=rust_build_path,
    )
    runtime_context = build_runtime_context(
        "joint_safe_link_minimal_raw_featureization",
        emit_startup_warning=False,
    )
    featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
        dataset,
        runtime_context=runtime_context,
        use_cache=False,
        rust_build_path=cast(Any, rust_build_path),
    )
    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=bool(clusterer.use_default_constraints_as_supervision),
        runtime_context=runtime_context,
        use_cache=bool(clusterer.use_cache),
        suppress_orcid=True,
    )
    signature_id_to_index = _signature_id_to_index(featurizer)
    member_path = _resolve_path(
        source_bundle,
        str(source_bundle.assets["candidate_members"]["datasets"][dataset_name]),
    )
    row_component_scope = "block-local"
    pairwise_component_scope = "block-local"
    component_details = _component_member_details_by_key(
        member_path,
        signature_id_to_index,
        dataset=dataset,
        component_scope=row_component_scope,
    )
    component_indices = {
        component_key: details.signature_indices for component_key, details in component_details.items()
    }
    pairwise_component_details = component_details
    pairwise_component_indices = component_indices
    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    paper_author_name_cache: dict[str, frozenset[str]] = {}
    retrieval_subblock_index, retrieval_subblock_index_diagnostics, component_keys_by_block = (
        _build_retrieval_subblock_index_for_components(
            dataset=dataset,
            component_details=component_details,
        )
    )
    summary_started = time.perf_counter()
    full_summary_cache = _build_full_retrieval_summary_cache(
        dataset=dataset,
        component_details=component_details,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        max_exemplars=max_exemplars,
    )
    rust_hybrid_centroid_retriever = build_rust_hybrid_centroid_retriever(
        list(full_summary_cache.values()),
        include_exemplars=FROZEN_RETRIEVAL_POLICY.uses_exemplar_scoring(),
    )
    max_block_component_size = max((summary.size for summary in full_summary_cache.values()), default=0)
    summary_seconds = float(time.perf_counter() - summary_started)
    print(
        json.dumps(
            {
                "event": "minimal_raw_dataset_context_ready",
                "dataset": dataset_name,
                "components": int(len(component_details)),
                "specter_embeddings": int(len(dataset.specter_embeddings or {})),
                "retrieval_policy": FROZEN_RETRIEVAL_POLICY_NAME,
                "component_scope": "block-local",
                "row_component_scope": row_component_scope,
                "pairwise_component_scope": pairwise_component_scope,
                "retrieval_subblock_index": retrieval_subblock_index_diagnostics,
                "retrieval_summary_build_seconds": round(summary_seconds, 3),
                "seconds": round(float(time.perf_counter() - started), 3),
            }
        ),
        flush=True,
    )
    return MinimalRawDatasetContext(
        dataset_name=dataset_name,
        row_component_scope=row_component_scope,
        pairwise_component_scope=pairwise_component_scope,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        featurizer=featurizer,
        signature_id_to_index=signature_id_to_index,
        component_details=component_details,
        component_indices=component_indices,
        pairwise_component_details=pairwise_component_details,
        pairwise_component_indices=pairwise_component_indices,
        component_keys_by_block=component_keys_by_block,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        full_summary_cache=full_summary_cache,
        residual_summary_cache={},
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        retrieval_subblock_index=retrieval_subblock_index,
        max_block_component_size=int(max_block_component_size),
    )


def _release_minimal_raw_dataset_context(context: MinimalRawDatasetContext) -> None:
    context.feature_cache.clear()
    context.paper_author_name_cache.clear()
    context.full_summary_cache.clear()
    context.residual_summary_cache.clear()
    context.retrieval_subblock_index.clear()
    context.component_details.clear()
    context.component_indices.clear()
    context.component_keys_by_block.clear()
    context.pairwise_component_details.clear()
    context.pairwise_component_indices.clear()
    context.signature_id_to_index.clear()
    feature_port.clear_rust_featurizer_cache()
    gc.collect()


def _load_pairwise_dataset(
    *,
    source_bundle: OfficialBundle,
    dataset_name: str,
    pairwise_source: str,
    n_jobs: int,
    giant_dataset_root: Path,
) -> Any:
    if pairwise_source == "official-original":
        return _load_original_pairwise_dataset(
            dataset_name,
            n_jobs=n_jobs,
            giant_dataset_root=giant_dataset_root,
        )
    if pairwise_source == "featureless-raw":
        return _load_featureless_raw_dataset(source_bundle, dataset_name, n_jobs=n_jobs)
    raise ValueError(f"Unknown pairwise source: {pairwise_source}")


def _assert_pairwise_model_is_raw_bundle_compatible(clusterer: Any, model_path: Path) -> None:
    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = tuple(str(value) for value in getattr(featurizer_info, "features_to_use", ()) or ())
        if "reference_features" in features_to_use:
            raise ValueError(
                f"Pairwise model {model_path} uses reference_features in {attr_name}; "
                "the minimal raw bundle intentionally does not store reference papers."
            )


def _component_id_parts(component_key: str, candidate_cluster_id: str | None = None) -> tuple[str, str]:
    if "::" in component_key:
        block_key, cluster_id = component_key.split("::", 1)
        return block_key, cluster_id
    cluster_id = str(candidate_cluster_id or component_key)
    if "_" in component_key:
        return component_key.split("_", 1)[0], cluster_id
    return component_key, cluster_id


def _finite_distance(value: float, *, empty_value: float = 1.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(empty_value)


def _nan_value_from_policy(policy: str) -> float:
    if policy == "preserve":
        return float("nan")
    if policy == "zero":
        return 0.0
    raise ValueError(f"Unsupported NaN policy: {policy}")


def _feature_nan_policy_summary(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pairwise_model_nan_policy": str(args.pairwise_model_nan_policy),
        "pairwise_aggregate_nan_policy": str(args.pairwise_aggregate_nan_policy),
        "row_nan_policy": str(args.row_nan_policy),
    }


def _score_candidate_summaries_with_frozen_rust_policy(
    *,
    query: retrieval.QueryFeatures,
    summaries: Mapping[str, retrieval.ClusterSummary],
    retriever: Any,
    max_block_component_size: int,
    n_jobs: int,
) -> dict[str, float]:
    """Score one query's candidate rows with the frozen Rust retrieval policy."""

    component_keys = [str(component_key) for component_key in summaries]
    if not component_keys:
        return {}
    override_summary: retrieval.ClusterSummary | None = None
    overridden_component_keys: list[str] = []
    summary_by_component = getattr(retriever, "summary_by_component", {})
    for component_key in component_keys:
        base_summary = summary_by_component.get(component_key)
        if base_summary is None:
            raise KeyError(f"Unknown component_key for frozen Rust retrieval: {component_key}")
        current_summary = summaries[component_key]
        if current_summary is not base_summary:
            overridden_component_keys.append(component_key)
            override_summary = current_summary
    if len(overridden_component_keys) > 1:
        raise ValueError(
            "Frozen Rust retrieval scoring supports at most one residual summary per query group; "
            f"got {overridden_component_keys}"
        )
    ranked = rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=len(component_keys),
        retriever=retriever,
        component_keys=component_keys,
        max_block_component_size=max(1, int(max_block_component_size)),
        override_summary=override_summary,
        num_threads=max(1, int(n_jobs)),
        weights=FROZEN_RETRIEVAL_POLICY.weights_for_query(query),
        scoring_config=FROZEN_RETRIEVAL_POLICY.scoring_config_for_query(query),
    )
    return {str(summary.component_key): round(float(score), 6) for score, summary in ranked}


def _current_retrieval_ranks_from_scores(
    retrieval_scores: Mapping[str, float],
    stored_retrieval_ranks: Mapping[str, int],
) -> dict[str, int]:
    """Return rank order induced by recomputed retrieval scores over the frozen candidate set."""

    ordered = sorted(
        retrieval_scores,
        key=lambda component_key: (
            -float(retrieval_scores[str(component_key)]),
            int(stored_retrieval_ranks[str(component_key)]),
            str(component_key),
        ),
    )
    return {str(component_key): rank for rank, component_key in enumerate(ordered, start=1)}


def _truthy_row_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value) != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _row_text_value(row: Any, field_name: str) -> str:
    value = getattr(row, field_name, "")
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).lower()


def _row_label_is_positive(row: Any) -> bool:
    value = getattr(row, "label", 0)
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return str(value).strip() == "1"


def _dataset_has_cluster_seed_constraints(dataset: ANDData) -> bool:
    return bool(getattr(dataset, "cluster_seeds_require", None)) or bool(
        getattr(dataset, "cluster_seeds_disallow", None)
    )


def _signature_has_seed_constraint(dataset: ANDData, signature_id: str) -> bool:
    signature_id = str(signature_id)
    require = getattr(dataset, "cluster_seeds_require", {}) or {}
    if signature_id in require:
        return True
    disallow = getattr(dataset, "cluster_seeds_disallow", set()) or set()
    return any(str(left) == signature_id or str(right) == signature_id for left, right in disallow)


def _seed_constrained_signature_ids(dataset: ANDData) -> frozenset[str]:
    signature_ids = {str(signature_id) for signature_id in (getattr(dataset, "cluster_seeds_require", {}) or {})}
    for left, right in getattr(dataset, "cluster_seeds_disallow", set()) or set():
        signature_ids.add(str(left))
        signature_ids.add(str(right))
    return frozenset(signature_ids)


def _row_allows_seed_constraint_bypass(
    dataset: ANDData,
    row: Any,
    *,
    seed_constraint_signature_ids: frozenset[str] | None = None,
) -> bool:
    if _truthy_row_value(getattr(row, "query_in_seed_before_holdout", None)):
        return True
    query_signature_id = getattr(row, "query_signature_id", None)
    if query_signature_id is not None:
        if seed_constraint_signature_ids is None:
            has_seed_constraint = _signature_has_seed_constraint(dataset, str(query_signature_id))
        else:
            has_seed_constraint = str(query_signature_id) in seed_constraint_signature_ids
        if has_seed_constraint:
            return True
    split = _row_text_value(row, "split")
    source = _row_text_value(row, "source")
    source_key = _row_text_value(row, "source_key")
    support_type = _row_text_value(row, "support_type")
    source_kind = _row_text_value(row, "source_kind")
    supervision_type = _row_text_value(row, "supervision_type")
    return (
        "loo" in split
        or "loo" in source
        or "loo" in source_key
        or "loo" in support_type
        or "loo" in source_kind
        or "loo" in supervision_type
        or "self" in support_type
        or "self" in source_kind
        or "self" in supervision_type
    )


def _has_query_seed_connection(
    dataset: ANDData,
    *,
    query_signature_id: str,
    candidate_signature_ids: Sequence[str],
) -> bool:
    query_signature_id = str(query_signature_id)
    require = getattr(dataset, "cluster_seeds_require", {}) or {}
    disallow = getattr(dataset, "cluster_seeds_disallow", set()) or set()
    query_required_cluster = require.get(query_signature_id)
    for candidate_signature_id in candidate_signature_ids:
        candidate_signature_id = str(candidate_signature_id)
        if (query_signature_id, candidate_signature_id) in disallow or (
            candidate_signature_id,
            query_signature_id,
        ) in disallow:
            return True
        if query_required_cluster is not None and require.get(candidate_signature_id) == query_required_cluster:
            return True
    return False


def _constraint_label_is_disallow(label: float) -> bool:
    if math.isnan(float(label)):
        return False
    return float(label) + float(LARGE_INTEGER) >= float(LARGE_DISTANCE)


def _resolve_candidate_batch_pair_labels(
    *,
    clusterer: Any,
    dataset: ANDData,
    batch: LinkerCandidateBatch,
    index_to_signature_id: Mapping[int, str],
    runtime_context: Any,
    constraint_backend: Any,
    chunk_size: int,
    pair_seed_bypass: np.ndarray | None = None,
    pair_ignore_disallow: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    pair_count = int(batch.pair_count)
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    chunk_size = max(1, int(chunk_size))
    started = time.perf_counter()
    batch_calls = 0

    if pair_seed_bypass is None:
        pair_seed_bypass = np.zeros(pair_count, dtype=bool)
    else:
        pair_seed_bypass = np.asarray(pair_seed_bypass, dtype=bool)
    if pair_ignore_disallow is None:
        pair_ignore_disallow = np.zeros(pair_count, dtype=bool)
    else:
        pair_ignore_disallow = np.asarray(pair_ignore_disallow, dtype=bool)
    if len(pair_seed_bypass) != pair_count:
        raise ValueError(f"pair_seed_bypass length {len(pair_seed_bypass)} != pair_count {pair_count}")
    if len(pair_ignore_disallow) != pair_count:
        raise ValueError(f"pair_ignore_disallow length {len(pair_ignore_disallow)} != pair_count {pair_count}")

    for start in range(0, pair_count, chunk_size):
        stop = min(pair_count, start + chunk_size)
        pairs = [
            (
                index_to_signature_id[int(left_index)],
                index_to_signature_id[int(right_index)],
            )
            for left_index, right_index in zip(
                batch.left_signature_indices[start:stop],
                batch.right_signature_indices[start:stop],
                strict=True,
            )
        ]
        chunk_labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
            dataset,
            pairs,
            partial_supervision={},
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=False,
            constraint_backend=constraint_backend,
        )
        labels[start:stop] = np.asarray(chunk_labels, dtype=np.float64)
        batch_calls += 1

    seed_bypass_indices = np.flatnonzero(pair_seed_bypass)
    seed_bypass_batch_calls = 0
    for start in range(0, len(seed_bypass_indices), chunk_size):
        stop = min(len(seed_bypass_indices), start + chunk_size)
        chunk_indices = seed_bypass_indices[start:stop]
        pairs = [
            (
                index_to_signature_id[int(batch.left_signature_indices[int(index)])],
                index_to_signature_id[int(batch.right_signature_indices[int(index)])],
            )
            for index in chunk_indices
        ]
        chunk_labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
            dataset,
            pairs,
            partial_supervision={},
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=True,
            constraint_backend=constraint_backend,
        )
        labels[chunk_indices] = np.asarray(chunk_labels, dtype=np.float64)
        seed_bypass_batch_calls += 1

    disallow_ignored = np.zeros(pair_count, dtype=bool)
    if np.any(pair_ignore_disallow):
        disallow_ignored = pair_ignore_disallow & np.asarray(
            [_constraint_label_is_disallow(float(label)) for label in labels],
            dtype=bool,
        )
        labels[disallow_ignored] = np.nan
    return labels, {
        "constraint_pair_count": pair_count,
        "constraint_batch_calls": int(batch_calls),
        "constraint_seed_bypass_pair_count": int(len(seed_bypass_indices)),
        "constraint_seed_bypass_batch_calls": int(seed_bypass_batch_calls),
        "constraint_disallow_ignored_pair_count": int(disallow_ignored.sum()),
        "constraint_seconds": round(float(time.perf_counter() - started), 3),
    }


def _materialize_promoted_table(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    output_path: Path,
    target_features: Sequence[str],
    pairwise_source: str,
    n_jobs: int,
    total_ram_bytes: int,
    giant_dataset_root: Path,
    datasets: set[str] | None,
    limit_rows: int | None,
    pairwise_aggregate_nan_value: float,
) -> dict[str, Any]:
    labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
    corrected_path = _asset_file(source_bundle, "corrected_feature_rows", table_key)
    labels, corrected = _read_selected_rows(
        labels_path=labels_path,
        corrected_path=corrected_path,
        datasets=datasets,
        limit_rows=limit_rows,
    )
    non_pairwise = tuple(feature for feature in target_features if not str(feature).startswith("pw_"))
    missing_non_pairwise = sorted(feature for feature in non_pairwise if feature not in corrected.columns)
    if missing_non_pairwise:
        raise ValueError(f"{corrected_path.name} is missing promoted non-pw columns: {missing_non_pairwise}")

    output = corrected.copy()
    for column in labels.columns:
        if column not in output.columns:
            output[column] = labels[column]

    pairwise_columns = tuple(feature for feature in target_features if str(feature).startswith("pw_"))
    pairwise_matrix = np.full((len(labels), len(pairwise_columns)), np.nan, dtype=np.float32)

    started = time.perf_counter()
    pair_count = 0
    dataset_summaries: list[dict[str, Any]] = []
    for dataset_name, dataset_rows in labels.groupby(labels["dataset"].astype(str), sort=False):
        dataset_name = str(dataset_name)
        row_positions = dataset_rows.index.to_numpy(dtype=np.int64)
        dataset_started = time.perf_counter()
        dataset = _load_pairwise_dataset(
            source_bundle=source_bundle,
            dataset_name=dataset_name,
            pairwise_source=pairwise_source,
            n_jobs=n_jobs,
            giant_dataset_root=giant_dataset_root,
        )
        featurizer = feature_port._get_rust_featurizer(dataset, use_cache=False)  # noqa: SLF001
        signature_id_to_index = _signature_id_to_index(featurizer)
        member_path = _resolve_path(
            source_bundle,
            str(source_bundle.assets["candidate_members"]["datasets"][dataset_name]),
        )
        component_members = _component_members_by_key(member_path, signature_id_to_index)
        batch = _candidate_batch_from_rows(dataset_rows, component_members, signature_id_to_index)
        stats = compute_candidate_batch_pairwise_aggregate_stats_rust(
            dataset,
            batch,
            n_jobs=max(1, int(n_jobs)),
            total_ram_bytes=int(total_ram_bytes),
            nan_value=float(pairwise_aggregate_nan_value),
            use_cache=False,
            featurizer=featurizer,
        )
        pairwise_values = _pairwise_feature_values(stats)
        for column_index, column in enumerate(pairwise_columns):
            pairwise_matrix[row_positions, column_index] = pairwise_values[str(column)]
        pair_count += int(batch.pair_count)
        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "rows": int(len(dataset_rows)),
                "pairs": int(batch.pair_count),
                "seconds": round(float(time.perf_counter() - dataset_started), 3),
            }
        )
        del dataset, featurizer, component_members, batch, stats
        feature_port.clear_rust_featurizer_cache()
        gc.collect()

    pairwise_frame = pd.DataFrame(pairwise_matrix, columns=list(pairwise_columns))
    for column in pairwise_columns:
        output[column] = pairwise_frame[column].to_numpy(dtype=np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return {
        "table_key": table_key,
        "labels_path": str(labels_path.relative_to(source_bundle.root)),
        "corrected_path": str(corrected_path.relative_to(source_bundle.root)),
        "output_path": str(output_path),
        "rows": int(len(output)),
        "pairs": int(pair_count),
        "datasets": dataset_summaries,
        "seconds": round(float(time.perf_counter() - started), 3),
        "pairwise_source": pairwise_source,
        "pairwise_aggregate_nan_value": (
            "nan" if math.isnan(float(pairwise_aggregate_nan_value)) else float(pairwise_aggregate_nan_value)
        ),
    }


def _parquet_row_count_and_columns(path: Path) -> tuple[int, set[str]]:
    parquet_file = pq.ParquetFile(path)
    return int(parquet_file.metadata.num_rows), set(parquet_file.schema_arrow.names)


def _validate_reusable_parquet(
    path: Path,
    *,
    expected_rows: int,
    required_columns: Iterable[str],
    context: str,
) -> int:
    row_count, columns = _parquet_row_count_and_columns(path)
    if row_count != int(expected_rows):
        raise ValueError(f"{context}: reusable parquet row count mismatch: {row_count} != {expected_rows} ({path})")
    missing_columns = sorted(set(required_columns) - columns)
    if missing_columns:
        raise ValueError(f"{context}: reusable parquet is missing columns: {missing_columns[:10]} ({path})")
    return row_count


def _relative_bundle_asset_path(bundle: OfficialBundle, path: Path) -> str:
    """Return a portable bundle-relative path for a resolved asset path."""

    try:
        return str(path.resolve().relative_to(bundle.root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Precomputed feature path escapes bundle root: {path}") from exc


def _target_spec_digest(target: Mapping[str, Any]) -> str:
    """Return the stable digest for a promoted training target spec."""

    return canonical_json_digest(dict(target))


def _precomputed_table_metadata(bundle: OfficialBundle, target_features: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return validated table metadata for a portable precomputed feature bundle."""

    spec = dict(bundle.models["classic"])
    table_metadata: dict[str, dict[str, Any]] = {}
    for table_key in _classic_table_keys(spec):
        path = _asset_file(bundle, "corrected_feature_rows", table_key)
        row_count, columns = _parquet_row_count_and_columns(path)
        missing_features = sorted(set(str(feature) for feature in target_features) - columns)
        if missing_features:
            raise ValueError(f"{table_key}: precomputed table is missing target features: {missing_features[:10]}")
        table_metadata[table_key] = {
            "path": _relative_bundle_asset_path(bundle, path),
            "rows": int(row_count),
            "feature_count": int(len(target_features)),
        }
    return table_metadata


def _precomputed_promoted_bundle_metadata(
    *,
    bundle: OfficialBundle,
    target: Mapping[str, Any],
    source_mode: str,
) -> dict[str, Any]:
    """Build portable metadata for a validated precomputed promoted feature bundle."""

    target_features = tuple(str(feature) for feature in target["features"])
    return {
        "schema_version": PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION,
        "source_mode": str(source_mode),
        "target_spec_digest": _target_spec_digest(target),
        "feature_schema_digest": promoted_linker_feature_schema_digest(target_features),
        "feature_count": int(target["feature_count"]),
        "feature_columns": list(target_features),
        "tables": _precomputed_table_metadata(bundle, target_features),
    }


def _stamp_precomputed_promoted_bundle_metadata(
    *,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    source_mode: str,
) -> None:
    """Persist portable precomputed-feature metadata into `bundle.json`."""

    bundle = load_bundle(output_bundle_root)
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    payload["precomputed_promoted_feature_bundle"] = _precomputed_promoted_bundle_metadata(
        bundle=bundle,
        target=target,
        source_mode=source_mode,
    )
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_precomputed_promoted_feature_bundle(
    *,
    bundle_root: Path,
    target: Mapping[str, Any],
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    """Load and validate a portable precomputed promoted feature bundle."""

    root = bundle_root.resolve()
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    metadata = payload.get("precomputed_promoted_feature_bundle")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "precomputed-promoted bundles must include precomputed_promoted_feature_bundle metadata; "
            "rerun materialization with --reuse-existing-features to stamp it"
        )
    if metadata.get("schema_version") != PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported precomputed promoted bundle schema_version: " f"{metadata.get('schema_version')!r}"
        )
    target_features = tuple(str(feature) for feature in target["features"])
    if tuple(str(feature) for feature in metadata.get("feature_columns", ())) != target_features:
        raise ValueError("Precomputed promoted bundle feature columns do not match target_json")
    if int(metadata.get("feature_count", -1)) != int(target["feature_count"]):
        raise ValueError("Precomputed promoted bundle feature_count does not match target_json")
    expected_target_digest = _target_spec_digest(target)
    if metadata.get("target_spec_digest") != expected_target_digest:
        raise ValueError("Precomputed promoted bundle target_spec_digest does not match target_json")
    expected_schema_digest = promoted_linker_feature_schema_digest(target_features)
    if metadata.get("feature_schema_digest") != expected_schema_digest:
        raise ValueError("Precomputed promoted bundle feature_schema_digest does not match target_json")

    raw_files = dict(payload.get("assets", {}).get("corrected_feature_rows", {}).get("files", {}))
    absolute_paths = sorted(str(path) for path in raw_files.values() if Path(str(path)).is_absolute())
    if absolute_paths:
        raise ValueError(f"Precomputed promoted bundle contains absolute feature paths: {absolute_paths[:5]}")

    bundle = _bundle_with_promoted_target(load_bundle(root), target)
    if tuple(str(feature) for feature in bundle.models["classic"]["feature_columns"]) != target_features:
        raise ValueError("Precomputed promoted bundle classic feature_columns do not match target_json")
    table_metadata = metadata.get("tables", {})
    if not isinstance(table_metadata, Mapping):
        raise ValueError("Precomputed promoted bundle metadata must include table row counts")
    featureization_summaries: list[dict[str, Any]] = []
    for table_key in _classic_table_keys(bundle.models["classic"]):
        if table_key not in table_metadata:
            raise ValueError(f"Precomputed promoted bundle metadata is missing table {table_key!r}")
        table_payload = dict(cast(Mapping[str, Any], table_metadata[table_key]))
        table_path = _asset_file(bundle, "corrected_feature_rows", table_key)
        if Path(str(table_payload.get("path", ""))).is_absolute():
            raise ValueError(f"{table_key}: precomputed table metadata path must be bundle-relative")
        if str(table_payload.get("path", "")) != _relative_bundle_asset_path(bundle, table_path):
            raise ValueError(f"{table_key}: precomputed table metadata path does not match bundle asset path")
        expected_rows = int(table_payload["rows"])
        row_count = _validate_reusable_parquet(
            table_path,
            expected_rows=expected_rows,
            required_columns=target_features,
            context=f"{table_key} precomputed promoted feature table",
        )
        featureization_summaries.append(
            {
                "table_key": table_key,
                "output_path": str(table_path.relative_to(bundle.root)),
                "rows": int(row_count),
                "mode": "precomputed-promoted",
                "reused": True,
            }
        )
    return bundle, featureization_summaries


def _validate_materialized_target_features(
    frame: pd.DataFrame,
    target_features: Sequence[str],
    *,
    context: str,
) -> None:
    """Validate materialized model features while preserving numeric NaNs."""

    infinite_features: dict[str, int] = {}
    for column in target_features:
        try:
            values = pd.to_numeric(frame[str(column)], errors="raise").to_numpy(dtype=np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: materialized feature {column!r} contains non-numeric values") from exc
        infinite_count = int(np.isinf(values).sum())
        if infinite_count:
            infinite_features[str(column)] = infinite_count
    if infinite_features:
        raise ValueError(f"{context}: materialized features contain infinite values: {infinite_features}")


def _required_materialized_output_columns(labels: pd.DataFrame, target_features: Sequence[str]) -> list[str]:
    """Return output columns, treating label columns as reusable feature columns."""

    columns = [str(column) for column in labels.columns]
    seen = set(columns)
    for feature in target_features:
        feature = str(feature)
        if feature not in seen:
            columns.append(feature)
            seen.add(feature)
    return columns


def _target_feature_frame_to_append(
    rows: pd.DataFrame,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> pd.DataFrame:
    """Return materialized target features that are not already present in row labels."""

    existing = {str(column) for column in rows.columns}
    return pd.DataFrame(
        {str(column): dataset_features[str(column)] for column in target_features if str(column) not in existing}
    )


def _copy_bundle_support_files(
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    *,
    reuse_existing_features: bool = False,
) -> dict[str, Any]:
    if output_bundle_root.exists() and not reuse_existing_features:
        shutil.rmtree(output_bundle_root)
    output_bundle_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bundle.root / "splits", output_bundle_root / "splits", dirs_exist_ok=True)
    shutil.copy2(source_bundle.root / "bundle.json", output_bundle_root / "bundle.json")
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    payload["bundle_name"] = f"{payload['bundle_name']}_promoted_rust_recomputed_pw"
    payload["expected_metrics"] = {}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _materialize_promoted_feature_bundle(
    *,
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    pairwise_source: str,
    n_jobs: int,
    total_ram_bytes: int,
    giant_dataset_root: Path,
    table_keys: Sequence[str] | None,
    datasets: set[str] | None,
    limit_rows: int | None,
    pairwise_aggregate_nan_value: float,
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    _copy_bundle_support_files(source_bundle, output_bundle_root)
    table_key_set = set(table_keys) if table_keys is not None else None
    source_spec = dict(source_bundle.models["classic"])
    selected_keys = [
        table_key
        for table_key in _classic_table_keys(source_spec)
        if table_key_set is None or table_key in table_key_set
    ]
    summaries: list[dict[str, Any]] = []
    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        output_relpath = _output_table_relpath(table_key, labels_path)
        output_path = output_bundle_root / output_relpath
        print(
            json.dumps(
                {
                    "event": "materialize_promoted_table_start",
                    "table_key": table_key,
                    "output_path": str(output_path),
                }
            ),
            flush=True,
        )
        summary = _materialize_promoted_table(
            source_bundle=source_bundle,
            table_key=table_key,
            output_path=output_path,
            target_features=tuple(str(feature) for feature in target["features"]),
            pairwise_source=pairwise_source,
            n_jobs=n_jobs,
            total_ram_bytes=total_ram_bytes,
            giant_dataset_root=giant_dataset_root,
            datasets=datasets,
            limit_rows=limit_rows,
            pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
        )
        summaries.append(summary)
        print(json.dumps({"event": "materialize_promoted_table_done", **summary}), flush=True)

    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        relpath = str(_output_table_relpath(table_key, labels_path))
        payload["assets"]["corrected_feature_rows"]["files"][table_key] = relpath
        if table_key.startswith("extra_eval_paths."):
            dataset_name = table_key.split(".", 1)[1]
            payload["models"]["classic"]["extra_eval_paths"][dataset_name] = relpath
        else:
            payload["models"]["classic"][table_key] = relpath
    payload["models"]["classic"]["feature_columns"] = list(target["features"])
    payload["models"]["classic"]["best_params"] = dict(target["params"])
    payload["expected_metrics"] = {"classic": _target_expected_metrics(target)}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_json(output_bundle_root / "featureization_summary.json", summaries)
    if table_keys is None and datasets is None and limit_rows is None:
        _stamp_precomputed_promoted_bundle_metadata(
            output_bundle_root=output_bundle_root,
            target=target,
            source_mode="rust-recompute-pw",
        )
    return _bundle_with_promoted_target(load_bundle(output_bundle_root), target), summaries


def _build_summary_for_members(
    *,
    dataset: ANDData,
    component_key: str,
    candidate_cluster_id: str | None,
    signature_ids: Sequence[str],
    feature_cache: dict[str, retrieval.QueryFeatures],
    paper_author_name_cache: dict[str, frozenset[str]] | None,
    max_exemplars: int,
) -> retrieval.ClusterSummary:
    block_key, cluster_id = _component_id_parts(component_key, candidate_cluster_id)
    return retrieval.build_cluster_summary(
        dataset=dataset,
        block_key=block_key,
        cluster_id=cluster_id,
        component_key=component_key,
        signature_ids=[str(signature_id) for signature_id in signature_ids],
        max_exemplars=max_exemplars,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        orcid_enabled=False,
    )


def _initialize_row_signal_arrays(row_count: int, rows: pd.DataFrame) -> dict[str, Any]:
    float_signal_names = (
        "retrieval_score",
        "retrieval_rank",
        "cluster_size",
        "named_signature_count",
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
        "query_has_affiliations",
        "query_has_coauthors",
        "query_has_specter",
        "query_has_name_counts",
        "candidate_has_affiliations",
        "candidate_has_coauthors",
        "candidate_has_specter_exemplars",
        "candidate_has_name_counts",
        "candidate_cluster_max_paper_author_count",
        "paper_author_list_max_jaccard",
        "paper_author_list_max_containment",
        "paper_author_list_max_overlap_count",
        "local_author_window10_jaccard_max",
        "local_author_window10_overlap_count_max",
        "best_author_count_log_absdiff",
        "affiliation_overlap",
        "coauthor_overlap",
        "year_compatibility",
        "specter_exemplar_similarity",
        "min_distance",
        "mean_distance",
        "top3_mean_distance",
        "top5_mean_distance",
        "pair_count",
        "last_name_count_min_rarity",
        "candidate_last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
    )
    signals: dict[str, Any] = {name: np.full(row_count, np.nan, dtype=np.float32) for name in float_signal_names}
    signals["candidate_component_key"] = rows["candidate_component_key"].astype(str).to_numpy(dtype=object)
    signals["query_view"] = rows["query_view"].astype(str).to_numpy(dtype=object)
    signals["dominant_first_name"] = np.empty(row_count, dtype=object)
    signals["query_first_token"] = np.empty(row_count, dtype=object)
    signals["family_id"] = np.empty(row_count, dtype=object)
    return signals


def _fill_row_signal(
    *,
    row_signals: dict[str, Any],
    local_index: int,
    component_key: str,
    query: retrieval.QueryFeatures,
    summary: retrieval.ClusterSummary,
    stats: Any,
    retrieval_rank: int,
    retrieval_score: float,
    query_first_token_for_prefix: str,
) -> None:
    profile = build_cluster_profile(summary)
    rarity = name_count_rarity_features(query, summary)
    query_year_missing = query.year is None
    candidate_year_missing = summary.year_min is None or summary.year_max is None

    row_signals["retrieval_score"][local_index] = float(retrieval_score)
    row_signals["retrieval_rank"][local_index] = float(retrieval_rank)
    row_signals["cluster_size"][local_index] = float(summary.size)
    row_signals["named_signature_count"][local_index] = float(profile.family_named_count)
    row_signals["dominant_first_name"][local_index] = str(profile.dominant_first_name or "")
    row_signals["family_id"][local_index] = str(profile.family_id or component_key)
    row_signals["candidate_year_min"][local_index] = float(summary.year_min or 0)
    row_signals["candidate_year_max"][local_index] = float(summary.year_max or 0)
    row_signals["candidate_year_range_missing"][local_index] = float(candidate_year_missing)
    row_signals["query_first_token"][local_index] = str(query_first_token_for_prefix or query.first or "")
    row_signals["query_year"][local_index] = float(query.year or 0)
    row_signals["query_year_missing"][local_index] = float(query_year_missing)
    row_signals["query_has_affiliations"][local_index] = float(query.has_affiliations)
    row_signals["query_has_coauthors"][local_index] = float(query.has_coauthors)
    row_signals["query_has_specter"][local_index] = float(
        bool(getattr(query, "has_specter", False)) and getattr(query, "specter", None) is not None
    )
    row_signals["query_has_name_counts"][local_index] = float(getattr(query, "name_counts", None) is not None)
    row_signals["candidate_has_affiliations"][local_index] = float(
        bool(summary.affiliation_counts) and summary.size > 0
    )
    row_signals["candidate_has_coauthors"][local_index] = float(bool(summary.coauthor_counts) and summary.size > 0)
    row_signals["candidate_has_specter_exemplars"][local_index] = float(
        bool(getattr(summary, "exemplar_vectors", ()) or ())
    )
    row_signals["candidate_has_name_counts"][local_index] = float(
        bool(getattr(summary, "name_counts_values", ()) or ())
    )
    row_signals["candidate_cluster_max_paper_author_count"][local_index] = float(
        getattr(summary, "max_paper_author_count", 0)
    )
    for signal_name, value in retrieval.raw_paper_evidence_features(query, summary).items():
        row_signals[signal_name][local_index] = float(value)
    row_signals["affiliation_overlap"][local_index] = round(
        float(counter_query_overlap(query.affiliation_terms, summary.affiliation_counts, summary.size)),
        6,
    )
    row_signals["coauthor_overlap"][local_index] = round(
        float(counter_query_overlap(query.coauthor_blocks, summary.coauthor_counts, summary.size)),
        6,
    )
    row_signals["year_compatibility"][local_index] = round(float(year_compatibility(query.year, summary)), 6)
    row_signals["specter_exemplar_similarity"][local_index] = round(
        float(specter_exemplar_similarity(query, summary)),
        6,
    )
    row_signals["min_distance"][local_index] = round(_finite_distance(stats.min_distance), 6)
    row_signals["mean_distance"][local_index] = round(_finite_distance(stats.mean_distance), 6)
    row_signals["top3_mean_distance"][local_index] = round(_finite_distance(stats.topk_mean_distance(3)), 6)
    row_signals["top5_mean_distance"][local_index] = round(_finite_distance(stats.topk_mean_distance(5)), 6)
    row_signals["pair_count"][local_index] = float(stats.count)
    for signal_name in (
        "last_name_count_min_rarity",
        "candidate_last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
    ):
        row_signals[signal_name][local_index] = float(rarity.get(signal_name, 0.0) or 0.0)


def _validate_row_signals(row_signals: Mapping[str, Any]) -> None:
    missing: dict[str, int] = {}
    for name, values in row_signals.items():
        if np.asarray(values).dtype == object:
            continue
        array = np.asarray(values, dtype=np.float32)
        missing_count = int(np.isnan(array).sum())
        if missing_count:
            missing[name] = missing_count
    if missing:
        raise ValueError(f"Raw feature materialization left unfilled row signals: {missing}")


def _bool_row_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing row signal required for semantic NaN policy: {name}")
    values = np.asarray(row_signals[name], dtype=np.float32)
    if values.shape != (row_count,):
        raise ValueError(f"Row signal {name!r} must have shape ({row_count},), got {values.shape}")
    return values > 0.0


def _normalized_alpha_present_signal(
    row_signals: Mapping[str, Any],
    name: str,
    row_count: int,
    *,
    min_length: int = 1,
) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing row signal required for semantic NaN policy: {name}")
    values = np.asarray(row_signals[name], dtype=object)
    if values.shape != (row_count,):
        raise ValueError(f"Row signal {name!r} must have shape ({row_count},), got {values.shape}")
    present = np.zeros(row_count, dtype=bool)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        normalized = s2and_text.normalize_text(str(value), special_case_apostrophes=True)
        present[index] = len(normalized) >= min_length
    return present


def _singleton_query_group_mask(candidate_batch: LinkerCandidateBatch) -> np.ndarray:
    row_count = int(candidate_batch.row_count)
    if candidate_batch.row_query_signature_indices is None:
        return np.zeros(row_count, dtype=bool)
    query_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)
    if query_indices.shape != (row_count,):
        raise ValueError(f"row_query_signature_indices must have shape ({row_count},), got {query_indices.shape}")
    _unique, inverse, counts = np.unique(query_indices, return_inverse=True, return_counts=True)
    return counts[inverse] <= 1


def _semantic_row_nan_masks(
    row_signals: Mapping[str, Any],
    candidate_batch: LinkerCandidateBatch,
) -> dict[str, np.ndarray]:
    row_count = int(candidate_batch.row_count)
    pair_count = np.asarray(row_signals["pair_count"], dtype=np.float32)
    if pair_count.shape != (row_count,):
        raise ValueError(f"pair_count must have shape ({row_count},), got {pair_count.shape}")

    distance_missing = pair_count <= 0.0
    competitor_missing = _singleton_query_group_mask(candidate_batch)
    query_year_missing = np.asarray(row_signals["query_year_missing"], dtype=np.float32) > 0.0
    candidate_year_range_missing = np.asarray(row_signals["candidate_year_range_missing"], dtype=np.float32) > 0.0
    query_has_affiliations = _bool_row_signal(row_signals, "query_has_affiliations", row_count)
    query_has_coauthors = _bool_row_signal(row_signals, "query_has_coauthors", row_count)
    query_has_specter = _bool_row_signal(row_signals, "query_has_specter", row_count)
    query_has_name_counts = _bool_row_signal(row_signals, "query_has_name_counts", row_count)
    candidate_has_affiliations = _bool_row_signal(row_signals, "candidate_has_affiliations", row_count)
    candidate_has_coauthors = _bool_row_signal(row_signals, "candidate_has_coauthors", row_count)
    candidate_has_specter_exemplars = _bool_row_signal(row_signals, "candidate_has_specter_exemplars", row_count)
    candidate_has_name_counts = _bool_row_signal(row_signals, "candidate_has_name_counts", row_count)
    candidate_dominant_first_available = _normalized_alpha_present_signal(
        row_signals,
        "dominant_first_name",
        row_count,
    )
    query_name_count_missing = ~query_has_name_counts
    candidate_name_count_missing = ~candidate_has_name_counts
    name_count_missing = query_name_count_missing | candidate_name_count_missing
    query_first_any_available = _normalized_alpha_present_signal(
        row_signals,
        "query_first_token",
        row_count,
        min_length=1,
    )
    first_name_comparison_missing = ~query_first_any_available | ~candidate_dominant_first_available

    distance_available = ~distance_missing
    competitor_available = ~competitor_missing
    year_comparison_missing = query_year_missing | candidate_year_range_missing
    affiliation_comparison_missing = ~(query_has_affiliations & candidate_has_affiliations)
    coauthor_comparison_missing = ~(query_has_coauthors & candidate_has_coauthors)
    specter_comparison_missing = ~(query_has_specter & candidate_has_specter_exemplars)
    anchor_support_missing = ~(distance_available | competitor_available)
    strong_support_missing = distance_missing
    residual_support_missing = ~(distance_available | competitor_available)
    return {
        "min_distance": distance_missing,
        "retrieval_reciprocal_rank": np.zeros(row_count, dtype=bool),
        "specter_exemplar_similarity": specter_comparison_missing,
        "coauthor_overlap": coauthor_comparison_missing,
        "affiliation_overlap": affiliation_comparison_missing,
        "year_compatibility": year_comparison_missing,
        "candidate_year_span": candidate_year_range_missing,
        "year_gap_to_candidate_range": year_comparison_missing,
        "year_gap_signed_to_candidate_range": year_comparison_missing,
        "affiliation_contradiction_severity": ~query_has_affiliations,
        "same_dominant_first_as_best_top5": first_name_comparison_missing,
        "same_family_as_heuristic_choice": first_name_comparison_missing | distance_missing,
        "query_first_prefix_match_any_length": first_name_comparison_missing,
        "anchor_evidence_count": anchor_support_missing,
        "strong_positive_anchor_score": strong_support_missing,
        "weak_residual_anchor_score": residual_support_missing,
        "sparse_relative_winner_score": residual_support_missing,
        "last_name_count_min_rarity": name_count_missing,
        "last_first_name_count_min_rarity": name_count_missing,
        "top5_mean_distance": distance_missing,
        "cluster_size_log": np.zeros(row_count, dtype=bool),
        "candidate_dominant_first_name_length": ~candidate_dominant_first_available,
        "paper_author_list_max_jaccard": np.zeros(row_count, dtype=bool),
        "paper_author_list_max_containment": np.zeros(row_count, dtype=bool),
        "paper_author_list_max_overlap_count": np.zeros(row_count, dtype=bool),
        "local_author_window10_jaccard_max": np.zeros(row_count, dtype=bool),
        "local_author_window10_overlap_count_max": np.zeros(row_count, dtype=bool),
        "best_author_count_log_absdiff": np.zeros(row_count, dtype=bool),
        "candidate_cluster_max_paper_author_count": np.zeros(row_count, dtype=bool),
    }


def _apply_row_nan_policy(
    features: Mapping[str, np.ndarray],
    row_signals: Mapping[str, Any],
    candidate_batch: LinkerCandidateBatch,
    *,
    row_nan_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if row_nan_policy == "finite":
        return {str(column): np.asarray(values, dtype=np.float32) for column, values in features.items()}, {
            "row_nan_policy": "finite",
            "semantic_nan_counts": {},
            "semantic_nan_total": 0,
        }
    if row_nan_policy != "semantic":
        raise ValueError(f"Unsupported row_nan_policy: {row_nan_policy}")

    adjusted = {str(column): np.asarray(values, dtype=np.float32).copy() for column, values in features.items()}
    masks = _semantic_row_nan_masks(row_signals, candidate_batch)
    nan_counts: dict[str, int] = {}
    for column, mask in masks.items():
        if column not in adjusted:
            continue
        adjusted[column][mask] = np.nan
        nan_counts[column] = int(np.isnan(adjusted[column]).sum())
    return adjusted, {
        "row_nan_policy": "semantic",
        "semantic_nan_counts": nan_counts,
        "semantic_nan_total": int(sum(nan_counts.values())),
        "semantic_nan_feature_count": int(sum(count > 0 for count in nan_counts.values())),
    }


def _query_first_token_for_prefix(group: pd.DataFrame, base_query: retrieval.QueryFeatures) -> str:
    if "query_author" in group.columns:
        for value in group["query_author"].tolist():
            token = _query_first_token(value)
            if token:
                return token
    first = getattr(base_query, "first", None)
    if first:
        return str(first)
    if "query_first_token" in group.columns:
        for value in group["query_first_token"].tolist():
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                token = str(value).strip()
                if token:
                    return token
    return ""


def _pairwise_feature_values(pairwise_stats: Any) -> dict[str, np.ndarray]:
    pairwise_columns = tuple(pairwise_stats.aggregate_feature_columns)
    if pairwise_columns != PROMOTED_PAIRWISE_COLUMNS:
        raise ValueError("Rust pairwise aggregate column order mismatch in minimal raw materialization")
    pairwise_matrix = pairwise_stats.feature_matrix().astype(np.float32, copy=False)
    return {
        column: np.asarray(pairwise_matrix[:, column_index], dtype=np.float32)
        for column_index, column in enumerate(pairwise_columns)
    }


def _assemble_minimal_raw_feature_values(
    *,
    target_features: Sequence[str],
    non_pairwise_features: Mapping[str, Any],
    pairwise_stats: Any,
) -> dict[str, np.ndarray]:
    pairwise_values = _pairwise_feature_values(pairwise_stats)
    feature_values: dict[str, np.ndarray] = {}
    for column in target_features:
        column = str(column)
        if column.startswith("pw_"):
            feature_values[column] = pairwise_values[column]
        else:
            feature_values[column] = np.asarray(non_pairwise_features[column], dtype=np.float32)
    return feature_values


def _materialize_minimal_raw_dataset_rows(
    *,
    context: MinimalRawDatasetContext,
    rows: pd.DataFrame,
    target_features: Sequence[str],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_exemplars: int,
    max_top_k: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    row_nan_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    dataset_name = context.dataset_name
    dataset = context.dataset
    runtime_context = context.runtime_context
    constraint_backend = context.constraint_backend
    featurizer = context.featurizer
    signature_id_to_index = context.signature_id_to_index
    component_details = context.component_details
    component_indices = context.component_indices
    feature_cache = context.feature_cache
    full_summary_cache = context.full_summary_cache
    residual_summary_cache = context.residual_summary_cache

    dataset_rows = rows.reset_index(drop=True).copy()
    row_count = len(dataset_rows)
    row_signals = _initialize_row_signal_arrays(row_count, dataset_rows)
    group_codes = tuple(
        int(value) for value in pd.factorize(dataset_rows["query_group_id"].astype(str), sort=False)[0].tolist()
    )
    row_seed_bypass = np.zeros(row_count, dtype=bool)
    row_ignore_disallow = np.zeros(row_count, dtype=bool)
    seed_constraint_signature_ids = (
        _seed_constrained_signature_ids(dataset) if _dataset_has_cluster_seed_constraints(dataset) else frozenset()
    )
    batch = _candidate_batch_from_rows(
        dataset_rows,
        component_indices,
        signature_id_to_index,
        row_group_ids=group_codes,
    )

    component_cluster_ids = (
        dataset_rows[["candidate_component_key", "candidate_cluster_id"]]
        .drop_duplicates("candidate_component_key")
        .set_index("candidate_component_key")["candidate_cluster_id"]
        .astype(str)
        .to_dict()
    )

    def summary_for(component_key: str, query_signature_id: str | None) -> tuple[retrieval.ClusterSummary, list[str]]:
        details = component_details[str(component_key)]
        member_ids = list(details.signature_ids)
        if query_signature_id is not None and str(query_signature_id) in details.signature_id_set:
            cache_key = (str(component_key), str(query_signature_id))
            summary = residual_summary_cache.get(cache_key)
            active_member_ids = [signature_id for signature_id in member_ids if signature_id != str(query_signature_id)]
            if summary is None:
                summary = _build_summary_for_members(
                    dataset=dataset,
                    component_key=str(component_key),
                    candidate_cluster_id=component_cluster_ids.get(str(component_key)),
                    signature_ids=active_member_ids,
                    feature_cache=feature_cache,
                    paper_author_name_cache=context.paper_author_name_cache,
                    max_exemplars=max_exemplars,
                )
                residual_summary_cache[cache_key] = summary
            return summary, active_member_ids

        summary = full_summary_cache.get(str(component_key))
        if summary is None:
            summary = _build_summary_for_members(
                dataset=dataset,
                component_key=str(component_key),
                candidate_cluster_id=component_cluster_ids.get(str(component_key)),
                signature_ids=member_ids,
                feature_cache=feature_cache,
                paper_author_name_cache=context.paper_author_name_cache,
                max_exemplars=max_exemplars,
            )
            full_summary_cache[str(component_key)] = summary
        return summary, member_ids

    contexts: list[dict[str, Any]] = []

    for query_group_id, group in dataset_rows.groupby(dataset_rows["query_group_id"].astype(str), sort=False):
        query_signature_ids = set(group["query_signature_id"].astype(str))
        query_views = set(group["query_view"].astype(str))
        if len(query_signature_ids) != 1 or len(query_views) != 1:
            raise ValueError(f"{dataset_name}: query group {query_group_id!r} is not a single query/view")
        query_signature_id = next(iter(query_signature_ids))
        query_view = next(iter(query_views))
        base_query = retrieval.extract_query_features(
            dataset,
            query_signature_id,
            feature_cache=feature_cache,
            paper_author_name_cache=context.paper_author_name_cache,
            orcid_enabled=False,
        )
        query_first_token_for_prefix = _query_first_token_for_prefix(group, base_query)
        query = retrieval.mask_query_features(base_query, query_view, orcid_enabled=False)
        sorted_group = group.sort_values(["retrieval_rank", "candidate_component_key"], kind="stable")
        summaries: dict[str, retrieval.ClusterSummary] = {}
        retrieval_ranks: dict[str, int] = {}
        rows_by_component: dict[str, list[int]] = {}
        for row in sorted_group.itertuples():
            component_key = str(row.candidate_component_key)
            summary, _active_member_ids = summary_for(component_key, query_signature_id)
            summaries[component_key] = summary
            retrieval_ranks[component_key] = int(row.retrieval_rank)
            rows_by_component.setdefault(component_key, []).append(int(row.Index))
            if _row_label_is_positive(row):
                row_ignore_disallow[int(row.Index)] = True
            if (
                seed_constraint_signature_ids
                and _row_allows_seed_constraint_bypass(
                    dataset,
                    row,
                    seed_constraint_signature_ids=seed_constraint_signature_ids,
                )
                and _has_query_seed_connection(
                    dataset,
                    query_signature_id=str(query_signature_id),
                    candidate_signature_ids=_active_member_ids,
                )
            ):
                row_seed_bypass[int(row.Index)] = True
        retrieval_scores = _score_candidate_summaries_with_frozen_rust_policy(
            query=query,
            summaries=summaries,
            retriever=context.rust_hybrid_centroid_retriever,
            max_block_component_size=context.max_block_component_size,
            n_jobs=n_jobs,
        )
        current_retrieval_ranks = _current_retrieval_ranks_from_scores(retrieval_scores, retrieval_ranks)
        contexts.append(
            {
                "query": query,
                "query_first_token_for_prefix": query_first_token_for_prefix,
                "retrieval_scores": retrieval_scores,
                "retrieval_ranks": current_retrieval_ranks,
                "summaries": summaries,
                "rows_by_component": rows_by_component,
            }
        )

    index_to_signature_id = {int(index): str(signature_id) for signature_id, index in signature_id_to_index.items()}
    constraint_chunk_size = max(1, min(int(pair_batch_size), int(query_batch_pair_limit)))
    pair_labels, constraint_summary = _resolve_candidate_batch_pair_labels(
        clusterer=clusterer,
        dataset=dataset,
        batch=batch,
        index_to_signature_id=index_to_signature_id,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        chunk_size=constraint_chunk_size,
        pair_seed_bypass=row_seed_bypass[batch.pair_row_indices],
        pair_ignore_disallow=row_seed_bypass[batch.pair_row_indices] | row_ignore_disallow[batch.pair_row_indices],
    )
    fused_pairwise_started = time.perf_counter()
    fused_pairwise = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
        batch,
        classifier=clusterer.classifier,
        featurizer_info=clusterer.featurizer_info,
        nameless_classifier=clusterer.nameless_classifier,
        nameless_featurizer_info=clusterer.nameless_featurizer_info,
        pair_labels=pair_labels,
        n_jobs=max(1, int(n_jobs)),
        total_ram_bytes=int(total_ram_bytes),
        pairwise_model_nan_value=float(pairwise_model_nan_value),
        pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
        runtime_context=runtime_context,
        use_cache=False,
        featurizer=featurizer,
    )
    fused_pairwise_seconds = float(time.perf_counter() - fused_pairwise_started)
    for query_context in contexts:
        query = query_context["query"]
        query_first_token_for_prefix = query_context["query_first_token_for_prefix"]
        retrieval_scores = query_context["retrieval_scores"]
        retrieval_ranks = query_context["retrieval_ranks"]
        summaries = query_context["summaries"]
        rows_by_component = query_context["rows_by_component"]
        for component_key, local_indices in rows_by_component.items():
            for local_index in local_indices:
                stats = FusedDistanceStats(
                    count=int(fused_pairwise.row_signals["pair_count"][local_index]),
                    min_distance=float(fused_pairwise.row_signals["min_distance"][local_index]),
                    mean_distance=float(fused_pairwise.row_signals["mean_distance"][local_index]),
                    top3_mean_distance=float(fused_pairwise.row_signals["top3_mean_distance"][local_index]),
                    top5_mean_distance=float(fused_pairwise.row_signals["top5_mean_distance"][local_index]),
                )
                _fill_row_signal(
                    row_signals=row_signals,
                    local_index=int(local_index),
                    component_key=str(component_key),
                    query=query,
                    summary=summaries[str(component_key)],
                    stats=stats,
                    retrieval_rank=int(retrieval_ranks[str(component_key)]),
                    retrieval_score=float(retrieval_scores[str(component_key)]),
                    query_first_token_for_prefix=str(query_first_token_for_prefix),
                )
    _validate_row_signals(row_signals)

    non_pairwise_started = time.perf_counter()
    non_pairwise_features = build_promoted_non_pairwise_row_features(batch, row_signals)
    non_pairwise_features, row_nan_summary = _apply_row_nan_policy(
        non_pairwise_features,
        row_signals,
        batch,
        row_nan_policy=str(row_nan_policy),
    )
    non_pairwise_seconds = float(time.perf_counter() - non_pairwise_started)
    feature_values = _assemble_minimal_raw_feature_values(
        target_features=target_features,
        non_pairwise_features=non_pairwise_features,
        pairwise_stats=fused_pairwise.pairwise_stats,
    )
    summary = {
        "dataset": dataset_name,
        "rows": int(row_count),
        "rust_pairwise_aggregate_pairs": int(batch.pair_count),
        "separate_rust_pairwise_aggregate_pairs": 0,
        "fused_pairwise_pairs": int(batch.pair_count),
        "pair_operation_count": int(batch.pair_count),
        "pairwise_model_pairs": int(batch.pair_count),
        "component_count": int(dataset_rows["candidate_component_key"].astype(str).nunique()),
        "query_group_count": int(dataset_rows["query_group_id"].astype(str).nunique()),
        "component_scope": "block-local",
        "row_component_scope": context.row_component_scope,
        "pairwise_component_scope": context.pairwise_component_scope,
        "full_summary_cache_size": int(len(full_summary_cache)),
        "residual_summary_cache_size": int(len(residual_summary_cache)),
        "retrieval_policy": FROZEN_RETRIEVAL_POLICY_NAME,
        "retrieval_max_block_component_size": int(context.max_block_component_size),
        "specter_embeddings": int(len(dataset.specter_embeddings or {})),
        "pairwise_model_nan_value": "nan"
        if math.isnan(float(pairwise_model_nan_value))
        else float(pairwise_model_nan_value),
        "pairwise_aggregate_nan_value": (
            "nan" if math.isnan(float(pairwise_aggregate_nan_value)) else float(pairwise_aggregate_nan_value)
        ),
        **row_nan_summary,
        **constraint_summary,
        "fused_pairwise_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_featurize_seconds": round(float(fused_pairwise.telemetry["feature_seconds"]), 3),
        "pairwise_model_predict_seconds": round(float(fused_pairwise.telemetry["predict_seconds"]), 3),
        "non_pairwise_formula_seconds": round(non_pairwise_seconds, 3),
        "rust_pairwise_aggregate_seconds": 0.0,
        "seconds": round(float(time.perf_counter() - started), 3),
    }
    del pair_labels, fused_pairwise
    gc.collect()
    return feature_values, summary


def _materialize_minimal_raw_table(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    output_path: Path,
    target_features: Sequence[str],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    datasets: set[str] | None,
    limit_rows: int | None,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_exemplars: int,
    max_top_k: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    row_nan_policy: str,
    reuse_existing_features: bool,
    rust_build_path: str | None,
) -> dict[str, Any]:
    labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
    labels = pd.read_parquet(labels_path)
    positions = _selected_row_positions(labels, datasets, limit_rows)
    labels = labels.iloc[positions].reset_index(drop=True)
    required_output_columns = _required_materialized_output_columns(labels, target_features)
    if reuse_existing_features and output_path.exists():
        row_count = _validate_reusable_parquet(
            output_path,
            expected_rows=len(labels),
            required_columns=required_output_columns,
            context=f"{table_key} existing output",
        )
        return {
            "table_key": table_key,
            "labels_path": str(labels_path.relative_to(source_bundle.root)),
            "output_path": str(output_path),
            "rows": int(row_count),
            "datasets": [],
            "seconds": 0.0,
            "mode": "minimal-raw-rust",
            "reused": True,
        }

    started = time.perf_counter()
    dataset_summaries: list[dict[str, Any]] = []
    partial_dir = output_path.parent / "_partial" / output_path.stem
    if partial_dir.exists() and not reuse_existing_features:
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_paths: list[Path] = []
    for dataset_name, dataset_rows in labels.groupby(labels["dataset"].astype(str), sort=False):
        dataset_name = str(dataset_name)
        row_positions = dataset_rows.index.to_numpy(dtype=np.int64)
        safe_dataset_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in dataset_name)
        partial_path = partial_dir / f"{safe_dataset_name}.parquet"
        if reuse_existing_features and partial_path.exists():
            row_count = _validate_reusable_parquet(
                partial_path,
                expected_rows=len(dataset_rows),
                required_columns=["_row_position", *required_output_columns],
                context=f"{table_key} {dataset_name} partial",
            )
            dataset_summaries.append(
                {
                    "dataset": dataset_name,
                    "rows": int(row_count),
                    "seconds": 0.0,
                    "mode": "minimal-raw-rust",
                    "reused": True,
                }
            )
            partial_paths.append(partial_path)
            print(
                json.dumps(
                    {
                        "event": "minimal_raw_dataset_featureization_reused",
                        "table_key": table_key,
                        "dataset": dataset_name,
                        "rows": int(row_count),
                        "partial_path": str(partial_path),
                    }
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "minimal_raw_dataset_featureization_start",
                    "table_key": table_key,
                    "dataset": dataset_name,
                    "rows": int(len(dataset_rows)),
                }
            ),
            flush=True,
        )
        context = _build_minimal_raw_dataset_context(
            source_bundle=source_bundle,
            dataset_name=dataset_name,
            clusterer=clusterer,
            n_jobs=n_jobs,
            rust_build_path=rust_build_path,
            max_exemplars=max_exemplars,
        )
        try:
            dataset_features, dataset_summary = _materialize_minimal_raw_dataset_rows(
                context=context,
                rows=dataset_rows.reset_index(drop=True),
                target_features=target_features,
                clusterer=clusterer,
                n_jobs=n_jobs,
                total_ram_bytes=total_ram_bytes,
                pair_batch_size=pair_batch_size,
                query_batch_pair_limit=query_batch_pair_limit,
                max_exemplars=max_exemplars,
                max_top_k=max_top_k,
                pairwise_model_nan_value=float(pairwise_model_nan_value),
                pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                row_nan_policy=str(row_nan_policy),
            )
        finally:
            _release_minimal_raw_dataset_context(context)
            del context
        feature_frame = _target_feature_frame_to_append(dataset_rows, dataset_features, target_features)
        partial_output = pd.concat([dataset_rows.reset_index(drop=True), feature_frame], axis=1)
        partial_output.insert(0, "_row_position", row_positions)
        partial_output.to_parquet(partial_path, index=False)
        partial_paths.append(partial_path)
        dataset_summaries.append(dataset_summary)
        print(
            json.dumps(
                {
                    "event": "minimal_raw_dataset_featureization_done",
                    "partial_path": str(partial_path),
                    **dataset_summary,
                }
            ),
            flush=True,
        )
        del dataset_features, feature_frame, partial_output
        gc.collect()

    parts = [pd.read_parquet(path) for path in partial_paths]
    output = pd.concat(parts, axis=0, ignore_index=True)
    output = output.sort_values("_row_position", kind="stable").drop(columns=["_row_position"]).reset_index(drop=True)
    if len(output) != len(labels):
        raise ValueError(f"{table_key}: materialized row count mismatch: {len(output)} != {len(labels)}")
    _validate_materialized_target_features(output, target_features, context=table_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    return {
        "table_key": table_key,
        "labels_path": str(labels_path.relative_to(source_bundle.root)),
        "output_path": str(output_path),
        "rows": int(len(output)),
        "datasets": dataset_summaries,
        "seconds": round(float(time.perf_counter() - started), 3),
        "mode": "minimal-raw-rust",
    }


def _safe_dataset_filename(dataset_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(dataset_name))


def _write_minimal_raw_partial(
    *,
    shard: MinimalRawPendingShard,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> None:
    _write_minimal_raw_partial_frame(
        rows=shard.rows,
        row_positions=shard.row_positions,
        partial_path=shard.partial_path,
        dataset_features=dataset_features,
        target_features=target_features,
    )


def _write_minimal_raw_partial_frame(
    *,
    rows: pd.DataFrame,
    row_positions: np.ndarray,
    partial_path: Path,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> None:
    feature_frame = _target_feature_frame_to_append(rows, dataset_features, target_features)
    partial_output = pd.concat([rows.reset_index(drop=True), feature_frame], axis=1)
    partial_output.insert(0, "_row_position", row_positions)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output.to_parquet(partial_path, index=False)
    del feature_frame, partial_output


def _finalize_minimal_raw_table_plan(
    *,
    plan: MinimalRawTablePlan,
    target_features: Sequence[str],
    source_bundle: OfficialBundle,
) -> dict[str, Any]:
    parts = [pd.read_parquet(path) for path in plan.partial_paths]
    output = pd.concat(parts, axis=0, ignore_index=True)
    output = output.sort_values("_row_position", kind="stable").drop(columns=["_row_position"]).reset_index(drop=True)
    if len(output) != len(plan.labels):
        raise ValueError(f"{plan.table_key}: materialized row count mismatch: {len(output)} != {len(plan.labels)}")
    _validate_materialized_target_features(output, target_features, context=plan.table_key)
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(plan.output_path, index=False)
    del parts, output
    gc.collect()
    return {
        "table_key": plan.table_key,
        "labels_path": str(plan.labels_path.relative_to(source_bundle.root)),
        "output_path": str(plan.output_path),
        "rows": int(len(plan.labels)),
        "datasets": plan.dataset_summaries,
        "structural_cleaning": plan.structural_cleaning_summary,
        "seconds": round(float(time.perf_counter() - plan.started), 3),
        "mode": "minimal-raw-rust",
    }


def _finalize_minimal_raw_bundle_metadata(
    *,
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    selected_keys: Sequence[str],
    stamp_precomputed_metadata: bool,
) -> OfficialBundle:
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    feature_count = int(target["feature_count"])
    tree_count = int(target["params"]["n_estimators"])
    payload["bundle_name"] = (
        f"{payload['bundle_name']}_minimal_raw_block_local_promoted_{feature_count}_{tree_count}trees"
    )
    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        relpath = str(_output_table_relpath(table_key, labels_path))
        payload["assets"]["corrected_feature_rows"]["files"][table_key] = relpath
        if table_key.startswith("extra_eval_paths."):
            dataset_name = table_key.split(".", 1)[1]
            payload["models"]["classic"]["extra_eval_paths"][dataset_name] = relpath
        else:
            payload["models"]["classic"][table_key] = relpath
    payload["models"]["classic"]["feature_columns"] = list(target["features"])
    payload["models"]["classic"]["best_params"] = dict(target["params"])
    payload["expected_metrics"] = {"classic": _target_expected_metrics(target)}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if stamp_precomputed_metadata:
        _stamp_precomputed_promoted_bundle_metadata(
            output_bundle_root=output_bundle_root,
            target=target,
            source_mode="minimal-raw-rust",
        )
    return _bundle_with_promoted_target(load_bundle(output_bundle_root), target)


def _materialize_minimal_raw_feature_bundle(
    *,
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    table_keys: Sequence[str] | None,
    datasets: set[str] | None,
    limit_rows: int | None,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_exemplars: int,
    max_top_k: int,
    reuse_existing_features: bool,
    rust_build_path: str | None,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    row_nan_policy: str,
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    _copy_bundle_support_files(
        source_bundle,
        output_bundle_root,
        reuse_existing_features=reuse_existing_features,
    )
    table_key_set = set(table_keys) if table_keys is not None else None
    source_spec = dict(source_bundle.models["classic"])
    selected_keys = [
        table_key
        for table_key in _classic_table_keys(source_spec)
        if table_key_set is None or table_key in table_key_set
    ]
    summaries: list[dict[str, Any]] = []
    target_features = tuple(str(feature) for feature in target["features"])
    table_plans: dict[str, MinimalRawTablePlan] = {}
    table_plan_order: list[str] = []
    pending_by_dataset: dict[str, list[MinimalRawPendingShard]] = {}
    component_membership_cache: dict[str, pd.DataFrame] = {}
    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        output_relpath = _output_table_relpath(table_key, labels_path)
        output_path = output_bundle_root / output_relpath
        print(
            json.dumps(
                {
                    "event": "minimal_raw_table_featureization_start",
                    "table_key": table_key,
                    "output_path": str(output_path),
                }
            ),
            flush=True,
        )
        labels = pd.read_parquet(labels_path)
        positions = _selected_row_positions(labels, datasets, limit_rows)
        labels = labels.iloc[positions].reset_index(drop=True)
        labels, structural_cleaning_summary = _clean_minimal_raw_structural_rows(
            source_bundle=source_bundle,
            table_key=table_key,
            rows=labels,
            component_membership_cache=component_membership_cache,
        )
        required_output_columns = _required_materialized_output_columns(labels, target_features)
        if reuse_existing_features and output_path.exists():
            row_count = _validate_reusable_parquet(
                output_path,
                expected_rows=len(labels),
                required_columns=required_output_columns,
                context=f"{table_key} existing output",
            )
            summary = {
                "table_key": table_key,
                "labels_path": str(labels_path.relative_to(source_bundle.root)),
                "output_path": str(output_path),
                "rows": int(row_count),
                "datasets": [],
                "seconds": 0.0,
                "mode": "minimal-raw-rust",
                "reused": True,
                "structural_cleaning": structural_cleaning_summary,
            }
            summaries.append(summary)
            print(json.dumps({"event": "minimal_raw_table_featureization_done", **summary}), flush=True)
            continue

        partial_dir = output_path.parent / "_partial" / output_path.stem
        if partial_dir.exists() and not reuse_existing_features:
            shutil.rmtree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        plan = MinimalRawTablePlan(
            table_key=table_key,
            labels_path=labels_path,
            output_path=output_path,
            labels=labels,
            required_output_columns=required_output_columns,
            partial_dir=partial_dir,
            partial_paths=[],
            dataset_summaries=[],
            structural_cleaning_summary=structural_cleaning_summary,
            started=time.perf_counter(),
        )
        table_plans[table_key] = plan
        table_plan_order.append(table_key)
        for dataset_name, dataset_rows in labels.groupby(labels["dataset"].astype(str), sort=False):
            dataset_name = str(dataset_name)
            row_positions = dataset_rows.index.to_numpy(dtype=np.int64)
            partial_path = partial_dir / f"{_safe_dataset_filename(dataset_name)}.parquet"
            if reuse_existing_features and partial_path.exists():
                row_count = _validate_reusable_parquet(
                    partial_path,
                    expected_rows=len(dataset_rows),
                    required_columns=["_row_position", *required_output_columns],
                    context=f"{table_key} {dataset_name} partial",
                )
                plan.dataset_summaries.append(
                    {
                        "dataset": dataset_name,
                        "rows": int(row_count),
                        "seconds": 0.0,
                        "mode": "minimal-raw-rust",
                        "reused": True,
                    }
                )
                plan.partial_paths.append(partial_path)
                print(
                    json.dumps(
                        {
                            "event": "minimal_raw_dataset_featureization_reused",
                            "table_key": table_key,
                            "dataset": dataset_name,
                            "rows": int(row_count),
                            "partial_path": str(partial_path),
                        }
                    ),
                    flush=True,
                )
                continue
            pending_by_dataset.setdefault(dataset_name, []).append(
                MinimalRawPendingShard(
                    table_key=table_key,
                    dataset_name=dataset_name,
                    rows=dataset_rows.reset_index(drop=True),
                    row_positions=row_positions,
                    partial_path=partial_path,
                )
            )

    for dataset_name, shards in pending_by_dataset.items():
        print(
            json.dumps(
                {
                    "event": "minimal_raw_dataset_context_start",
                    "dataset": dataset_name,
                    "shards": len(shards),
                    "rows": int(sum(len(shard.rows) for shard in shards)),
                    "tables": sorted({shard.table_key for shard in shards}),
                }
            ),
            flush=True,
        )
        context = _build_minimal_raw_dataset_context(
            source_bundle=source_bundle,
            dataset_name=dataset_name,
            clusterer=clusterer,
            n_jobs=n_jobs,
            rust_build_path=rust_build_path,
            max_exemplars=max_exemplars,
        )
        try:
            for shard in shards:
                print(
                    json.dumps(
                        {
                            "event": "minimal_raw_dataset_featureization_start",
                            "table_key": shard.table_key,
                            "dataset": shard.dataset_name,
                            "rows": int(len(shard.rows)),
                        }
                    ),
                    flush=True,
                )
                dataset_features, dataset_summary = _materialize_minimal_raw_dataset_rows(
                    context=context,
                    rows=shard.rows,
                    target_features=target_features,
                    clusterer=clusterer,
                    n_jobs=n_jobs,
                    total_ram_bytes=total_ram_bytes,
                    pair_batch_size=pair_batch_size,
                    query_batch_pair_limit=query_batch_pair_limit,
                    max_exemplars=max_exemplars,
                    max_top_k=max_top_k,
                    pairwise_model_nan_value=float(pairwise_model_nan_value),
                    pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                    row_nan_policy=str(row_nan_policy),
                )
                _write_minimal_raw_partial(
                    shard=shard,
                    dataset_features=dataset_features,
                    target_features=target_features,
                )
                table_plan = table_plans[shard.table_key]
                table_plan.partial_paths.append(shard.partial_path)
                table_plan.dataset_summaries.append(dataset_summary)
                print(
                    json.dumps(
                        {
                            "event": "minimal_raw_dataset_featureization_done",
                            "table_key": shard.table_key,
                            "partial_path": str(shard.partial_path),
                            **dataset_summary,
                        }
                    ),
                    flush=True,
                )
                del dataset_features
                gc.collect()
        finally:
            _release_minimal_raw_dataset_context(context)
            del context

    for table_key in table_plan_order:
        summary = _finalize_minimal_raw_table_plan(
            plan=table_plans[table_key],
            target_features=target_features,
            source_bundle=source_bundle,
        )
        summaries.append(summary)
        print(json.dumps({"event": "minimal_raw_table_featureization_done", **summary}), flush=True)

    _write_json(output_bundle_root / "featureization_summary.json", summaries)
    return (
        _finalize_minimal_raw_bundle_metadata(
            source_bundle=source_bundle,
            output_bundle_root=output_bundle_root,
            target=target,
            selected_keys=selected_keys,
            stamp_precomputed_metadata=table_keys is None and datasets is None and limit_rows is None,
        ),
        summaries,
    )


def _classic_candidate_training_rows(rows: pd.DataFrame, *, retrieval_rank_limit: int) -> pd.DataFrame:
    required = {"query_group_id", "retrieval_rank", "label"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Classic training rows are missing required columns: {missing}")
    out = rows.copy()
    out["retrieval_rank"] = pd.to_numeric(out["retrieval_rank"], errors="coerce")
    out = out[out["retrieval_rank"] <= int(retrieval_rank_limit)].copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(np.int8)
    return out


def _prepare_prod_training_data(
    bundle: OfficialBundle,
    *,
    holdout_importance_weight: float,
    retrieval_rank_limit: int = 25,
) -> ProdTrainingData:
    """Build final production rows: train plus calibration/eval rows with extra importance."""

    if float(holdout_importance_weight) <= 0.0:
        raise ValueError("holdout_importance_weight must be positive")
    spec = dict(bundle.models["classic"])
    train_path = _resolve_path(bundle, spec["train_path"])
    train_df = _classic_candidate_training_rows(
        _read_csv(train_path, compression="gzip"),
        retrieval_rank_limit=int(retrieval_rank_limit),
    )
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

    train_df = train_df.copy()
    train_df["_prod_source_kind"] = "train"
    train_df["_prod_importance_weight"] = 1.0
    train_df["_prod_source_path"] = str(train_path.relative_to(bundle.root))
    frames = [train_df]

    for source_name, path_like in _iter_classic_train_holdout_paths(spec):
        path = _resolve_path(bundle, path_like)
        source_df = _classic_candidate_training_rows(
            _read_csv(path, compression="gzip"),
            retrieval_rank_limit=int(retrieval_rank_limit),
        )
        source_df["_prod_source_kind"] = str(source_name)
        source_df["_prod_importance_weight"] = float(holdout_importance_weight)
        source_df["_prod_source_path"] = str(path.relative_to(bundle.root))
        frames.append(source_df)

    combined = pd.concat(frames, ignore_index=True)
    source_kinds = combined["_prod_source_kind"].astype(str)
    importance_weights = pd.to_numeric(combined["_prod_importance_weight"], errors="raise").to_numpy(dtype=np.float32)
    query_group_ids = combined["query_group_id"].astype(str)
    group_sizes = query_group_ids.value_counts(sort=False)
    base_weights = (1.0 / query_group_ids.map(group_sizes).astype(float)).to_numpy(dtype=np.float32)
    sample_weight = (base_weights * importance_weights).astype(np.float32)

    source_summaries: list[dict[str, Any]] = []
    for source_name, source_rows in combined.groupby("_prod_source_kind", sort=False):
        source_indices = source_rows.index.to_numpy(dtype=np.int64)
        labels = pd.to_numeric(source_rows["label"], errors="coerce").fillna(0).astype(np.int8)
        paths = sorted(set(source_rows["_prod_source_path"].astype(str)))
        weight_values = sorted(float(value) for value in set(source_rows["_prod_importance_weight"].astype(float)))
        source_summaries.append(
            {
                "source": str(source_name),
                "paths": paths,
                "rows": int(len(source_rows)),
                "queries": int(source_rows["query_group_id"].astype(str).nunique()),
                "positive_rows": int(labels.sum()),
                "importance_weights": weight_values,
                "sample_weight_sum": round(float(sample_weight[source_indices].sum()), 6),
            }
        )

    model_rows = combined.drop(columns=["_prod_source_kind", "_prod_importance_weight", "_prod_source_path"])
    if len(model_rows) != len(source_kinds):
        raise RuntimeError("Production training row metadata length mismatch")
    return ProdTrainingData(
        rows=model_rows,
        sample_weight=sample_weight,
        source_summaries=source_summaries,
        train_holdout_filter_summary=train_holdout_filter_summary,
        train_filter_summary=train_filter_summary,
    )


def _artifact_gate_config_from_classic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    rule = dict(summary["abstain_rule"])
    return {
        "score_threshold": rule["score_threshold"],
        "margin_threshold": rule["margin_threshold"],
        "single_candidate_score_threshold": rule["single_candidate_score_threshold"],
        "bucketed_score_thresholds": rule.get("bucketed_score_thresholds"),
        "bucketed_margin_threshold": rule.get("bucketed_margin_threshold"),
        "bucketed_margin_thresholds": rule.get("bucketed_margin_thresholds"),
        "calibration_mode": rule["calibration_mode"],
    }


def _train_and_save_prod_artifact(
    *,
    feature_bundle: OfficialBundle,
    classic_summary: Mapping[str, Any],
    output_dir: Path,
    save_artifact_to: Path,
    artifact_audit_metadata: Mapping[str, Any] | None,
    holdout_importance_weight: float,
    required_rust_capabilities: Sequence[str] = INCREMENTAL_LINKING_RUST_CAPABILITIES,
) -> dict[str, Any]:
    spec = dict(feature_bundle.models["classic"])
    feature_columns = tuple(str(feature) for feature in spec["feature_columns"])
    monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
    prod_training_data = _prepare_prod_training_data(
        feature_bundle,
        holdout_importance_weight=float(holdout_importance_weight),
        retrieval_rank_limit=25,
    )
    train_matrix = _classic_feature_matrix(prod_training_data.rows, feature_columns).to_numpy(dtype=np.float32)
    train_labels = prod_training_data.rows["label"].to_numpy(dtype=np.int8, copy=False)
    model = _build_classic_classifier(spec["best_params"], monotone_constraints=monotone_constraints)
    started = time.perf_counter()
    model.fit(train_matrix, train_labels, sample_weight=prod_training_data.sample_weight)
    train_seconds = float(time.perf_counter() - started)

    audit_metadata = dict(artifact_audit_metadata or {})
    audit_metadata["prod_training"] = {
        "policy": "train_plus_calibration_eval_weighted",
        "holdout_importance_weight": float(holdout_importance_weight),
        "retrieval_rank_limit": 25,
        "rows": int(len(prod_training_data.rows)),
        "positive_rows": int(train_labels.sum()),
        "sample_weight_sum": round(float(prod_training_data.sample_weight.sum()), 6),
        "sources": prod_training_data.source_summaries,
        "train_holdout_filter_summary": prod_training_data.train_holdout_filter_summary,
        "train_filter_summary": prod_training_data.train_filter_summary,
        "params": dict(spec["best_params"]),
    }
    artifact_metadata = save_incremental_linking_artifact(
        model,
        Path(save_artifact_to),
        feature_columns=feature_columns,
        retrieval_top_k=25,
        gate_config=_artifact_gate_config_from_classic_summary(classic_summary),
        prediction_fixture_matrix=train_matrix[:5],
        required_rust_capabilities=required_rust_capabilities,
        audit_metadata=audit_metadata,
    )
    summary = {
        "path": str(Path(save_artifact_to)),
        "schema_version": artifact_metadata.schema_version,
        "feature_schema_digest": artifact_metadata.feature_schema_digest,
        "production_contract_digest": artifact_metadata.production_contract_digest,
        "retrieval_stack_digest": artifact_metadata.retrieval_stack_digest,
        "training_summary": {
            "rows": int(len(prod_training_data.rows)),
            "queries": int(prod_training_data.rows["query_group_id"].astype(str).nunique()),
            "positive_rows": int(train_labels.sum()),
            "sample_weight_sum": round(float(prod_training_data.sample_weight.sum()), 6),
            "holdout_importance_weight": float(holdout_importance_weight),
            "elapsed_seconds": round(train_seconds, 6),
            "sources": prod_training_data.source_summaries,
            "train_holdout_filter_summary": prod_training_data.train_holdout_filter_summary,
            "train_filter_summary": prod_training_data.train_filter_summary,
        },
    }
    _write_json(output_dir / "prod_artifact_summary.json", summary)
    return summary


def _observed_official_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    train = dict(summary["training_summary"])
    stratified_test = dict(summary["stratified_eval_test_split"]["overall"]["test"])
    n_queries = int(stratified_test["n_queries"])
    false_abstain_error_rate = float(stratified_test["false_abstain"]) / float(n_queries) if n_queries else 0.0
    false_link_error_rate = float(stratified_test["false_link"]) / float(n_queries) if n_queries else 0.0
    wrong_link_error_rate = float(stratified_test["wrong_candidate_link"]) / float(n_queries) if n_queries else 0.0
    weighted_average_error = (
        (
            WEIGHTED_ERROR_WEIGHTS["false_abstain_error_rate"] * false_abstain_error_rate
            + WEIGHTED_ERROR_WEIGHTS["false_link_error_rate"] * false_link_error_rate
            + WEIGHTED_ERROR_WEIGHTS["wrong_link_error_rate"] * wrong_link_error_rate
        )
        / sum(WEIGHTED_ERROR_WEIGHTS.values())
        if WEIGHTED_ERROR_WEIGHTS
        else 0.0
    )
    return {
        "training_rows": int(train["rows"]),
        "training_positive_rows": int(train["positive_rows"]),
        "stratified_test_queries": n_queries,
        "stratified_test_accuracy": float(stratified_test["accuracy"]),
        "stratified_test_balanced_accuracy": float(stratified_test["balanced_accuracy"]),
        "stratified_test_error_rate": float(stratified_test["error_rate"]),
        "stratified_test_errors": int(stratified_test["errors"]),
        "stratified_test_false_abstain": int(stratified_test["false_abstain"]),
        "stratified_test_false_link": int(stratified_test["false_link"]),
        "stratified_test_wrong_candidate_link": int(stratified_test["wrong_candidate_link"]),
        "false_abstain_error_rate": false_abstain_error_rate,
        "false_link_error_rate": false_link_error_rate,
        "wrong_link_error_rate": wrong_link_error_rate,
        "weighted_average_error": weighted_average_error,
        "weighted_average_error_weights": dict(WEIGHTED_ERROR_WEIGHTS),
    }


def _metric_deltas(observed: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    target_metrics = dict(target.get("metrics", {}))
    deltas: dict[str, Any] = {}
    for key, observed_value in observed.items():
        if key not in target_metrics:
            continue
        expected_value = target_metrics[key]
        if isinstance(observed_value, str):
            deltas[key] = observed_value == str(expected_value)
        elif isinstance(observed_value, Mapping) or isinstance(expected_value, Mapping):
            deltas[key] = dict(observed_value) == dict(expected_value)
        elif isinstance(observed_value, int):
            deltas[key] = int(observed_value) - int(expected_value)
        else:
            deltas[key] = float(observed_value) - float(expected_value)
    return deltas


def _assert_no_metric_drift(observed: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    deltas = _metric_deltas(observed, target)
    bad: dict[str, Any] = {}
    for key, delta in deltas.items():
        if isinstance(delta, bool):
            if not delta:
                bad[key] = {"observed": observed[key], "expected": target["metrics"][key]}
        elif isinstance(delta, int):
            if delta != 0:
                bad[key] = delta
        elif abs(float(delta)) > 1e-12:
            bad[key] = delta
    if bad:
        raise RuntimeError(f"Official promoted run drifted from target metrics: {bad}")


def _parse_tables(values: Sequence[str] | None) -> tuple[str, ...] | None:
    if not values:
        return None
    return tuple(dict.fromkeys(str(value) for value in values))


def _parse_datasets(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value) for value in values}


def _resolve_hyperopt_evals(args: argparse.Namespace) -> int:
    if args.hyperopt_evals is not None:
        resolved = int(args.hyperopt_evals)
    elif bool(args.hyperopt):
        resolved = 25
    else:
        resolved = 0
    if resolved < 0:
        raise SystemExit("--hyperopt-evals must be non-negative")
    if bool(args.hyperopt) and resolved == 0:
        raise SystemExit("--hyperopt requires --hyperopt-evals > 0")
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    target = _load_target(args.target_json)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise_model_nan_value = _nan_value_from_policy(str(args.pairwise_model_nan_policy))
    pairwise_aggregate_nan_value = _nan_value_from_policy(str(args.pairwise_aggregate_nan_policy))
    feature_nan_policy = _feature_nan_policy_summary(args)
    if args.feature_mode == "minimal-raw-rust":
        if args.limit_rows is None and not args.run_full:
            raise SystemExit("unbounded minimal raw feature materialization requires --run-full")
        if not args.materialize_only and (args.limit_rows is not None or args.tables or args.datasets):
            raise SystemExit(
                "limited/table-filtered minimal raw materialization is smoke-only; pass --materialize-only"
            )
        source_bundle = load_bundle(args.source_bundle_root)
        clusterer = load_clusterer(args.pairwise_model_path, n_jobs=int(args.n_jobs))
        clusterer.use_cache = False
        _assert_pairwise_model_is_raw_bundle_compatible(clusterer, args.pairwise_model_path)
        pair_batch_size = int(args.pair_batch_size) if args.pair_batch_size is not None else int(clusterer.batch_size)
        feature_bundle_root = output_dir / "minimal_raw_feature_bundle"
        feature_bundle, featureization_summaries = _materialize_minimal_raw_feature_bundle(
            source_bundle=source_bundle,
            output_bundle_root=feature_bundle_root,
            target=target,
            clusterer=clusterer,
            n_jobs=int(args.n_jobs),
            total_ram_bytes=int(args.total_ram_bytes),
            table_keys=_parse_tables(args.tables),
            datasets=_parse_datasets(args.datasets),
            limit_rows=args.limit_rows,
            pair_batch_size=pair_batch_size,
            query_batch_pair_limit=int(args.query_batch_pair_limit),
            max_exemplars=int(args.max_exemplars),
            max_top_k=int(args.max_top_k),
            reuse_existing_features=bool(args.reuse_existing_features),
            rust_build_path=args.minimal_raw_rust_build_path,
            pairwise_model_nan_value=pairwise_model_nan_value,
            pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
            row_nan_policy=str(args.row_nan_policy),
        )
        if args.materialize_only:
            result = {
                "mode": args.feature_mode,
                "source_bundle_root": str(source_bundle.root),
                "feature_bundle_root": str(feature_bundle.root),
                "pairwise_model_path": str(args.pairwise_model_path),
                "feature_count": int(target["feature_count"]),
                "minimal_raw_component_scope": "block-local",
                "feature_nan_policy": feature_nan_policy,
                "featureization": featureization_summaries,
            }
            _write_json(output_dir / "run_summary.json", result)
            return result
    elif args.feature_mode == "rust-recompute-pw":
        if args.limit_rows is None and not args.run_full:
            raise SystemExit("unbounded Rust feature rematerialization requires --run-full")
        if not args.materialize_only and (args.limit_rows is not None or args.tables or args.datasets):
            raise SystemExit("limited/table-filtered Rust rematerialization is smoke-only; pass --materialize-only")
        source_bundle = load_bundle(args.source_bundle_root)
        feature_bundle_root = output_dir / "recomputed_feature_bundle"
        feature_bundle, featureization_summaries = _materialize_promoted_feature_bundle(
            source_bundle=source_bundle,
            output_bundle_root=feature_bundle_root,
            target=target,
            pairwise_source=str(args.pairwise_source),
            n_jobs=int(args.n_jobs),
            total_ram_bytes=int(args.total_ram_bytes),
            giant_dataset_root=args.giant_dataset_root,
            table_keys=_parse_tables(args.tables),
            datasets=_parse_datasets(args.datasets),
            limit_rows=args.limit_rows,
            pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
        )
        if args.materialize_only:
            result = {
                "mode": args.feature_mode,
                "feature_bundle_root": str(feature_bundle.root),
                "feature_count": int(target["feature_count"]),
                "feature_nan_policy": feature_nan_policy,
                "featureization": featureization_summaries,
            }
            _write_json(output_dir / "run_summary.json", result)
            return result
    elif args.feature_mode == "precomputed-promoted":
        if args.precomputed_feature_bundle_root is None:
            raise SystemExit("--feature-mode precomputed-promoted requires --precomputed-feature-bundle-root")
        if not args.run_full:
            raise SystemExit("precomputed-promoted train/calibrate/eval requires --run-full")
        if args.materialize_only:
            raise SystemExit("precomputed-promoted does not materialize features")
        if args.limit_rows is not None or args.tables or args.datasets:
            raise SystemExit("precomputed-promoted requires a complete validated feature bundle")
        feature_bundle, featureization_summaries = _load_precomputed_promoted_feature_bundle(
            bundle_root=args.precomputed_feature_bundle_root,
            target=target,
        )
    else:
        raise ValueError(f"Unknown feature mode: {args.feature_mode}")

    if args.materialize_only:
        raise SystemExit("materialize-only is only valid with a materializing feature mode")

    run_output_dir = output_dir / "classic"
    started = time.perf_counter()
    active_params = dict(feature_bundle.models["classic"]["best_params"])
    hyperopt_evals = _resolve_hyperopt_evals(args)
    hyperopt_summary: dict[str, Any] | None = None
    if hyperopt_evals > 0:
        active_params, hyperopt_summary = _run_classic_hyperopt(
            feature_bundle=feature_bundle,
            output_dir=output_dir / "classic_hyperopt",
            base_params=active_params,
            hyperopt_evals=int(hyperopt_evals),
            metric=str(args.hyperopt_metric),
            seed=int(args.hyperopt_seed),
        )
        feature_bundle = _bundle_with_classic_params(feature_bundle, active_params)

    save_artifact_to = args.save_artifact_to.resolve() if args.save_artifact_to is not None else None
    artifact_audit_metadata = (
        _linker_artifact_audit_metadata(
            args=args,
            target=target,
            feature_bundle=feature_bundle,
            featureization_summaries=featureization_summaries,
        )
        if save_artifact_to is not None
        else None
    )
    summary = run_classic(
        feature_bundle,
        run_output_dir,
    )
    observed = _observed_official_metrics(summary)
    deltas = _metric_deltas(observed, target)
    if artifact_audit_metadata is not None:
        artifact_audit_metadata = {
            **artifact_audit_metadata,
            "classic_train_calibrate_eval": {
                "summary_artifact": "not bundled; observed metrics are embedded in this metadata",
                "observed_metrics": observed,
                "metric_deltas": deltas,
            },
            "hyperopt": hyperopt_summary or {"enabled": False},
        }
    prod_artifact_summary = None
    if save_artifact_to is not None:
        prod_artifact_summary = _train_and_save_prod_artifact(
            feature_bundle=feature_bundle,
            classic_summary=summary,
            output_dir=output_dir,
            save_artifact_to=save_artifact_to,
            artifact_audit_metadata=artifact_audit_metadata,
            holdout_importance_weight=float(args.prod_holdout_importance_weight),
        )
    result = {
        "mode": args.feature_mode,
        "feature_bundle_root": str(feature_bundle.root),
        "target_json": str(args.target_json),
        "feature_count": int(target["feature_count"]),
        "n_estimators": int(active_params["n_estimators"]),
        "target_n_estimators": int(target["params"]["n_estimators"]),
        "model_params": dict(active_params),
        "target_params": dict(target["params"]),
        "elapsed_seconds": round(float(time.perf_counter() - started), 3),
        "featureization": featureization_summaries,
        "observed_metrics": observed,
        "target_metrics": dict(target["metrics"]),
        "metric_deltas": deltas,
        "classic_summary_path": str(run_output_dir / "summary.json"),
        "hyperopt": hyperopt_summary or {"enabled": False},
        "feature_nan_policy": feature_nan_policy,
    }
    if save_artifact_to is not None:
        result["artifact_dir"] = str(save_artifact_to)
        result["artifact_summary"] = dict(prod_artifact_summary or {})
    if args.feature_mode == "minimal-raw-rust":
        result["minimal_raw_component_scope"] = "block-local"
    if hyperopt_summary is not None and not args.allow_metric_drift:
        result["metric_drift_check"] = "skipped_after_hyperopt_param_search"
    _write_json(output_dir / "run_summary.json", result)
    if not args.allow_metric_drift and hyperopt_summary is None:
        _assert_no_metric_drift(observed, target)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--target-json", type=Path, default=DEFAULT_TARGET_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pairwise-model-path", type=Path, default=DEFAULT_PAIRWISE_MODEL_PATH)
    parser.add_argument("--save-artifact-to", type=Path, default=None)
    parser.add_argument("--linker-artifact-version", default="v1.2")
    parser.add_argument(
        "--prod-holdout-importance-weight",
        type=float,
        default=10.0,
        help="Final production fit multiplier for calibration/eval candidate rows.",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperopt over the initial train/calibrate/eval stack before the final production fit.",
    )
    parser.add_argument(
        "--hyperopt-evals",
        type=int,
        default=None,
        help="Number of train/calibrate/eval hyperopt trials. Passing a value enables hyperopt.",
    )
    parser.add_argument(
        "--hyperopt-metric",
        choices=(
            "weighted_average_error",
            "stratified_test_errors",
            "stratified_test_error_rate",
            "stratified_test_balanced_accuracy",
        ),
        default="weighted_average_error",
        help=(
            "Metric optimized by hyperopt. weighted_average_error uses "
            "0.25*false_abstain_error_rate, 1.0*false_link_error_rate, "
            "and 1.5*wrong_link_error_rate, divided by the total weight."
        ),
    )
    parser.add_argument("--hyperopt-seed", type=int, default=13)
    parser.add_argument(
        "--feature-mode",
        choices=("minimal-raw-rust", "rust-recompute-pw", "precomputed-promoted"),
        default="minimal-raw-rust",
        help="Feature source for the official train/calibrate/eval run.",
    )
    parser.add_argument(
        "--precomputed-feature-bundle-root",
        type=Path,
        default=None,
        help=(
            "Explicit portable precomputed promoted feature bundle root. Required only with "
            "--feature-mode precomputed-promoted."
        ),
    )
    parser.add_argument(
        "--pairwise-source",
        choices=("official-original", "featureless-raw"),
        default="official-original",
        help="Dataset source for Rust pairwise aggregate recomputation.",
    )
    parser.add_argument(
        "--pairwise-model-nan-policy",
        choices=NAN_POLICY_CHOICES,
        default="preserve",
        help=(
            "Missing-value policy for pairwise model feature matrices in minimal raw materialization. "
            "The production default preserves NaNs for the pairwise distance model internals."
        ),
    )
    parser.add_argument(
        "--pairwise-aggregate-nan-policy",
        choices=NAN_POLICY_CHOICES,
        default="zero",
        help=(
            "Missing-value policy for promoted pw_* aggregates. The production default reproduces "
            "prod12 dense zero-filled pairwise semantics; preserve uses nan-aware aggregation."
        ),
    )
    parser.add_argument(
        "--row-nan-policy",
        choices=ROW_NAN_POLICY_CHOICES,
        default="finite",
        help="Missing-value policy for promoted non-pw row features.",
    )
    parser.add_argument("--giant-dataset-root", type=Path, default=DEFAULT_GIANT_DATASET_ROOT)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-bytes", type=int, default=DEFAULT_TOTAL_RAM_BYTES)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--query-batch-pair-limit", type=int, default=200_000)
    parser.add_argument("--max-exemplars", type=int, default=4)
    parser.add_argument("--max-top-k", type=int, default=DEFAULT_CHOOSER_CACHE_MAX_TOP_K)
    parser.add_argument(
        "--tables", nargs="*", help="Optional table keys to materialize in feature rematerialization modes."
    )
    parser.add_argument("--datasets", nargs="*", help="Optional dataset slugs to keep when materializing smoke checks.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional per-table row limit for smoke checks.")
    parser.add_argument("--materialize-only", action="store_true", help="Stop after Rust feature materialization.")
    parser.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help="Reuse already materialized output tables and dataset partials in the output directory.",
    )
    parser.add_argument(
        "--minimal-raw-rust-build-path",
        choices=("from_json_paths", "from_dataset"),
        default=None,
        help=(
            "Optional RustFeaturizer constructor override for minimal-raw-rust materialization. "
            "Defaults to the normal dataset lifecycle policy."
        ),
    )
    parser.add_argument("--run-full", action="store_true", help="Explicitly allow an unbounded official run.")
    parser.add_argument("--allow-metric-drift", action="store_true", help="Do not fail if final metrics differ.")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
