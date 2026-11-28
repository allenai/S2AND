"""Load linker training sources and holdout identities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class OfficialBundle:
    """Single-file metadata and paths for the official stack."""

    root: Path
    bundle_name: str
    assets: dict[str, Any]
    models: dict[str, Any]
    expected_metrics: dict[str, Any]


OfficialBundle.__module__ = "s2and.incremental_linking_training.classic"


@dataclass(frozen=True)
class ClassicHoldoutIdentities:
    """Complete query/base identity authority shared by training stages."""

    query_group_ids: frozenset[str]
    base_group_ids: frozenset[str]
    sources: tuple[dict[str, Any], ...]


ClassicHoldoutIdentities.__module__ = "s2and.incremental_linking_training.classic"


def read_classic_holdout_identities(bundle: OfficialBundle) -> ClassicHoldoutIdentities:
    """Read only identity columns from complete source holdout tables.

    Featureless assets are authoritative for release bundles. Direct training
    callers use their complete configured calibration/evaluation tables.
    """
    spec = dict(bundle.models["classic"])
    source_files = bundle.assets.get("featureless_rows", {}).get("files")
    if source_files is not None:
        spec = {"extra_eval_paths": {}}
        for key, path in source_files.items():
            if key.startswith("extra_eval_paths."):
                spec["extra_eval_paths"][key.split(".", 1)[1]] = path
            else:
                spec[key] = path
    query_ids, base_ids, sources = _read_classic_holdout_identity_sets(bundle, spec)
    return ClassicHoldoutIdentities(frozenset(query_ids), frozenset(base_ids), tuple(sources))


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
        header = pq.read_schema(path).names if path.suffix == ".parquet" else _read_csv(path, nrows=0).columns
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
    # Candidate rows form one query decision; never change its candidate window
    # by excluding only some rows when a base identity matches a holdout.
    if "query_group_id" in train_df:
        excluded_queries = set(train_df.loc[remove_mask, "query_group_id"].astype(str))
        remove_mask = train_df["query_group_id"].astype(str).isin(excluded_queries)
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
    return filtered.reset_index(drop=True), summary


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
