"""Promote strict Rust name-compatible eval rows into the official bundle."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd

from s2and.data import NameCounts, _canonicalize_last_for_counts, _load_name_counts_cached
from s2and.text import name_counts as pairwise_name_counts
from s2and.text import normalize_text, same_prefix_tokens, split_first_middle_hyphen_aware

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from scripts.single_letter_reranker_utils import (  # noqa: E402
    NAME_COUNT_RARITY_FEATURE_COLUMNS,
    PAIRWISE_NAME_COUNT_FEATURE_NAMES,
    _first_name_candidate_compatibility,
    _name_count_rarity,
    materialize_derived_rows,
)

DEFAULT_BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "scratch" / "name_compat_all_slices_20260429" / "official_row_promotion_summary.json"
NAME_COMPAT_CORRECTIONS_PATH = Path("dataset_contract") / "name_compat_manual_positive_corrections.csv"
HWANG_ROW_RELATIVE_PATH = Path("test") / "hwang_eval_rows.csv.gz"
HWANG_CLEAN_OVERRIDES_RELATIVE_PATH = Path("test") / "hwang_cleaned_eval_overrides.csv"
HWANG_CANDIDATE_LEVEL_MANIFEST_RELATIVE_PATH = Path("test") / "hwang_candidate_level_label_overrides.csv"
HWANG_CANDIDATE_LEVEL_SUMMARY_RELATIVE_PATH = Path("test") / "hwang_candidate_level_label_overrides_summary.json"


@dataclass(frozen=True)
class SlicePromotionConfig:
    """Static file inputs for one promoted official eval slice."""

    dataset: str
    slice_key: str
    row_relative_path: Path
    generated_rows_path: Path
    predicted_clusters_path: Path
    signatures_path: Path


SLICE_CONFIGS = (
    SlicePromotionConfig(
        dataset="h_wang",
        slice_key="hwang_eval",
        row_relative_path=Path("test") / "hwang_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "h_wang"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "h_wang_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/h_wang/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="s_park",
        slice_key="s_park_eval",
        row_relative_path=Path("test") / "s_park_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "s_park"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "s_park_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/s_park/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="s_lee",
        slice_key="s_lee_eval",
        row_relative_path=Path("test") / "s_lee_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT / "scratch" / "name_compat_slee_full_20260429" / "name_compat" / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "s_lee_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/s_lee/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="j_smith",
        slice_key="j_smith_eval",
        row_relative_path=Path("test") / "j_smith_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "j_smith"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "j_smith_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/j_smith/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="a_khan",
        slice_key="a_khan_eval",
        row_relative_path=Path("test") / "a_khan_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "a_khan"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "a_khan_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/a_khan/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="a_silva",
        slice_key="a_silva_eval",
        row_relative_path=Path("test") / "a_silva_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "a_silva"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "a_silva_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/a_silva/signatures.json"),
    ),
    SlicePromotionConfig(
        dataset="s_gupta",
        slice_key="s_gupta_eval",
        row_relative_path=Path("test") / "s_gupta_eval_rows.csv.gz",
        generated_rows_path=REPO_ROOT
        / "scratch"
        / "name_compat_all_slices_20260429"
        / "s_gupta"
        / "name_compat"
        / "rows.csv",
        predicted_clusters_path=REPO_ROOT / "scratch" / "s_gupta_multi_letter_v12_15000" / "predicted_clusters.json",
        signatures_path=Path("D:/data/s_gupta/signatures.json"),
    ),
)


def _read_rows(path: Path) -> pd.DataFrame:
    """Read a CSV or gzipped CSV row file with empty strings preserved."""

    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, compression=compression, keep_default_na=False, low_memory=False)


def _read_fieldnames(path: Path) -> list[str]:
    """Read CSV fieldnames without loading the full row file."""

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    if fieldnames is None:
        raise ValueError(f"Could not read CSV header from {path}")
    return [str(field) for field in fieldnames]


def _component_subblock_key(component_key: str) -> str:
    """Return the predicted-clusters subblock key for a component key."""

    return str(component_key).rsplit("_", 1)[0]


def _load_component_signature_ids(
    predicted_clusters_path: Path,
    *,
    component_keys: set[str],
) -> dict[str, tuple[str, ...]]:
    """Load signature ids for the requested predicted cluster components."""

    predicted_clusters = json.loads(predicted_clusters_path.read_text(encoding="utf-8"))
    component_signature_ids: dict[str, tuple[str, ...]] = {}
    for component_key in sorted(component_keys):
        if not component_key:
            continue
        subblock_key = _component_subblock_key(component_key)
        try:
            signature_ids = predicted_clusters[subblock_key][component_key]
        except KeyError as exc:
            raise KeyError(
                f"Missing component {component_key!r} under subblock {subblock_key!r} in {predicted_clusters_path}"
            ) from exc
        component_signature_ids[str(component_key)] = tuple(str(signature_id) for signature_id in signature_ids)
    return component_signature_ids


def _read_json_entries(path: Path, target_keys: set[str]) -> dict[str, Any]:
    """Read selected entries from the newline-oriented S2AND JSON object files."""

    found: dict[str, Any] = {}
    remaining = set(str(key) for key in target_keys)
    with path.open("rb") as handle:
        for raw_line in handle:
            if not remaining:
                break
            line = raw_line.strip()
            if not line or line in {b"{", b"}"}:
                continue
            line = line.rstrip(b",")
            if not line.startswith(b'"'):
                continue
            key = line.split(b'":', 1)[0][1:].decode("utf-8")
            if key not in remaining:
                continue
            found[key] = json.loads(b"{" + line + b"}")[key]
            remaining.remove(key)
    if remaining:
        raise ValueError(f"Missing {len(remaining)} signatures from {path}: examples={sorted(remaining)[:5]}")
    return found


def _signature_name_counts(
    signature: Mapping[str, Any],
    *,
    first_dict: Mapping[str, int],
    last_dict: Mapping[str, int],
    first_last_dict: Mapping[str, int],
    last_first_initial_dict: Mapping[str, int],
) -> tuple[NameCounts, str]:
    """Compute legacy prediction-time name counts for one raw signature."""

    author_info = dict(signature.get("author_info") or {})
    first_raw = str(author_info.get("first") or "")
    middle_raw = str(author_info.get("middle") or "")
    last_raw = str(author_info.get("last") or "")
    first_without_apostrophe, _middle_without_apostrophe = split_first_middle_hyphen_aware(first_raw, middle_raw)
    last_normalized = normalize_text(last_raw)
    first_normalized_token_for_counts = first_without_apostrophe.split(" ")[0] if first_without_apostrophe else ""
    first_for_counts = first_normalized_token_for_counts
    if "-" in first_raw:
        joined = (first_without_apostrophe or "").replace(" ", "")
        if joined:
            first_for_counts = joined
    last_for_counts = _canonicalize_last_for_counts(last_raw, last_normalized)
    first_last_for_count = (first_for_counts + " " + last_for_counts).strip()
    last_first_initial_for_count = (last_for_counts + " " + first_for_counts).strip()
    first_count = first_dict.get(first_for_counts, 1) if len(first_for_counts) > 1 else None
    first_last_count = first_last_dict.get(first_last_for_count, 1) if len(first_for_counts) > 1 else None
    first_name_for_summary = normalize_text(first_raw).split(" ")[0] if normalize_text(first_raw) else ""
    return (
        NameCounts(
            first=float(first_count) if first_count is not None else None,
            last=float(last_dict.get(last_for_counts, 1)),
            first_last=float(first_last_count) if first_last_count is not None else None,
            last_first_initial=float(last_first_initial_dict.get(last_first_initial_for_count, 1)),
        ),
        first_name_for_summary,
    )


def _candidate_name_count_rarity_features(candidate_name_counts_values: Sequence[NameCounts]) -> dict[str, float]:
    """Return visible candidate-component rarity from candidate signature counts."""

    minima: dict[str, float] = {}
    for candidate_name_counts in candidate_name_counts_values:
        for field_name in ("first", "first_last", "last", "last_first_initial"):
            raw_value = getattr(candidate_name_counts, field_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not isfinite(value) or value <= 0.0:
                continue
            minima[field_name] = min(value, minima.get(field_name, value))
    return {
        "candidate_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first")), 6),
        "candidate_last_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first_last")), 6),
        "candidate_last_name_count_min_rarity": round(_name_count_rarity(minima.get("last")), 6),
        "candidate_last_first_initial_count_min_rarity": round(
            _name_count_rarity(minima.get("last_first_initial")),
            6,
        ),
    }


def _build_name_count_features_by_pair(
    rows: pd.DataFrame,
    *,
    component_signature_ids: Mapping[str, Sequence[str]],
    signatures_path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute name-count rarity features for generated rows missing those columns."""

    needed_signature_ids = set(rows["query_signature_id"].astype(str))
    for component_key in rows["candidate_component_key"].astype(str).unique():
        needed_signature_ids.update(str(signature_id) for signature_id in component_signature_ids[str(component_key)])
    raw_signatures = _read_json_entries(signatures_path, needed_signature_ids)
    first_dict, last_dict, first_last_dict, last_first_initial_dict = _load_name_counts_cached()
    counts_by_signature: dict[str, NameCounts] = {}
    first_by_signature: dict[str, str] = {}
    for signature_id, signature in raw_signatures.items():
        counts, first_name = _signature_name_counts(
            signature,
            first_dict=first_dict,
            last_dict=last_dict,
            first_last_dict=first_last_dict,
            last_first_initial_dict=last_first_initial_dict,
        )
        counts_by_signature[str(signature_id)] = counts
        first_by_signature[str(signature_id)] = first_name

    features_by_pair: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows.to_dict(orient="records"):
        query_signature_id = str(row["query_signature_id"])
        component_key = str(row["candidate_component_key"])
        candidate_signature_ids = [
            str(signature_id)
            for signature_id in component_signature_ids[component_key]
            if str(signature_id) != query_signature_id
        ]
        candidate_name_counts = [counts_by_signature[signature_id] for signature_id in candidate_signature_ids]
        candidate_features = _candidate_name_count_rarity_features(candidate_name_counts)
        query_name_counts = counts_by_signature[query_signature_id]
        observed_minima: dict[str, float] = {}
        for candidate_counts in candidate_name_counts:
            values = pairwise_name_counts(query_name_counts, candidate_counts)
            for feature_name, raw_value in zip(PAIRWISE_NAME_COUNT_FEATURE_NAMES, values, strict=True):
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not isfinite(value) or value <= 0.0:
                    continue
                observed_minima[feature_name] = min(value, observed_minima.get(feature_name, value))
        features = {
            f"{feature_name}_rarity": round(_name_count_rarity(observed_minima.get(feature_name)), 6)
            for feature_name in PAIRWISE_NAME_COUNT_FEATURE_NAMES
        }
        if int(float(row.get("query_has_full_first") or 0)) == 0:
            for column in (
                "first_name_count_min_rarity",
                "last_first_name_count_min_rarity",
                "last_first_initial_count_min_rarity",
                "first_name_count_max_rarity",
                "last_first_name_count_max_rarity",
            ):
                features[column] = 0.0
        first_prefix_match = 0.0
        query_first = str(row.get("query_first_token") or "")
        if len(query_first) > 1 and candidate_signature_ids:
            first_counts = Counter(first_by_signature[signature_id] for signature_id in candidate_signature_ids)
            first_counts.pop("", None)
            for candidate_first, count in first_counts.items():
                if len(candidate_first) > 1 and same_prefix_tokens(query_first, candidate_first):
                    first_prefix_match = max(first_prefix_match, float(count) / float(len(candidate_signature_ids)))
        features["first_prefix_x_last_first_name_count_min_rarity"] = round(
            float(first_prefix_match) * float(features["last_first_name_count_min_rarity"]),
            6,
        )
        features_by_pair[(str(row["query_group_id"]), component_key)] = {**features, **candidate_features}
    return features_by_pair


def _write_rows(path: Path, rows: pd.DataFrame, *, fieldnames: Sequence[str]) -> None:
    """Write promoted rows with the official field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[str(field) for field in fieldnames])
        writer.writeheader()
        for row in rows.to_dict(orient="records"):
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp_path.replace(path)


def _write_plain_rows(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    """Write a plain CSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[str(field) for field in fieldnames])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    temp_path.replace(path)


def _pair_set(rows: pd.DataFrame) -> set[tuple[str, str]]:
    """Return nonblank query-candidate pairs from a row frame."""

    if rows.empty:
        return set()
    return {
        (str(query_id), str(candidate_key))
        for query_id, candidate_key in zip(
            rows["query_group_id"].astype(str),
            rows["candidate_component_key"].astype(str),
            strict=True,
        )
        if str(candidate_key)
    }


def _positive_pairs_from_rows(rows: pd.DataFrame) -> set[tuple[str, str]]:
    """Return active nonblank positive candidate pairs from existing official rows."""

    labels = pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(int)
    return _pair_set(rows.loc[labels == 1])


def _correction_pairs(corrections: pd.DataFrame, *, slice_key: str) -> set[tuple[str, str]]:
    """Return nonblank manual name-compat correction pairs for one slice."""

    if corrections.empty:
        return set()
    slice_rows = corrections[
        (corrections["slice_key"].astype(str) == str(slice_key))
        & (corrections["action"].astype(str) == "candidate_positive")
        & (corrections["target_label"].astype(str) == "1")
    ]
    return _pair_set(slice_rows)


def _base_group_id_by_query(official_rows: pd.DataFrame) -> dict[str, str]:
    """Return the first nonblank base group id observed for each query."""

    if "base_group_id" not in official_rows.columns:
        return {}
    mapping: dict[str, str] = {}
    for row in official_rows[["query_group_id", "base_group_id"]].itertuples(index=False):
        query_group_id = str(row.query_group_id)
        base_group_id = str(row.base_group_id)
        if query_group_id not in mapping and base_group_id:
            mapping[query_group_id] = base_group_id
    return mapping


def _append_missing_official_positive_rows(
    promoted_rows: pd.DataFrame,
    official_rows: pd.DataFrame,
    *,
    required_positive_pairs: set[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append official positive rows that the promoted retrieval surface missed."""

    generated_pairs = _pair_set(promoted_rows)
    missing_pairs = required_positive_pairs - generated_pairs
    if not missing_pairs:
        return promoted_rows, {
            "required_positive_pairs_missing_generated": 0,
            "official_positive_rows_appended": 0,
        }

    official_positive_rows = official_rows[
        pd.to_numeric(official_rows["label"], errors="coerce").fillna(0).astype(int) == 1
    ].copy()
    official_positive_rows["_pair"] = list(
        zip(
            official_positive_rows["query_group_id"].astype(str),
            official_positive_rows["candidate_component_key"].astype(str),
            strict=True,
        )
    )
    append_rows = official_positive_rows[official_positive_rows["_pair"].isin(missing_pairs)].copy()
    found_pairs = set(append_rows["_pair"].tolist())
    unresolved_pairs = sorted(missing_pairs - found_pairs)
    if unresolved_pairs:
        raise ValueError(
            "Promoted rows are missing required positive pairs that are not present in official positives: "
            f"{unresolved_pairs[:5]}"
        )
    append_rows = append_rows.drop(columns=["_pair"])
    return pd.concat([promoted_rows, append_rows], ignore_index=True, sort=False), {
        "required_positive_pairs_missing_generated": int(len(missing_pairs)),
        "official_positive_rows_appended": int(len(append_rows)),
    }


def _enrich_promoted_prerequisites(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill legacy official columns needed by classic feature derivation."""

    enriched = rows.copy()
    for column in ("query_year", "candidate_year_min", "candidate_year_max"):
        if column not in enriched.columns:
            continue
        enriched[column] = enriched[column].astype(object)
        numeric = pd.to_numeric(enriched[column], errors="coerce")
        integer_mask = numeric.notna() & ((numeric % 1.0) == 0.0)
        if integer_mask.any():
            enriched.loc[integer_mask, column] = numeric.loc[integer_mask].astype(int).astype(str)

    if "named_signature_count" not in enriched.columns:
        enriched["named_signature_count"] = ""
    missing_named = enriched["named_signature_count"].astype(str).str.strip() == ""
    if "cluster_size" in enriched.columns:
        enriched.loc[missing_named, "named_signature_count"] = enriched.loc[missing_named, "cluster_size"]

    if "confident_family_flag" not in enriched.columns:
        enriched["confident_family_flag"] = ""
    missing_confident = enriched["confident_family_flag"].astype(str).str.strip() == ""
    if {"family_id", "candidate_component_key"}.issubset(enriched.columns):
        enriched.loc[missing_confident, "confident_family_flag"] = (
            enriched.loc[missing_confident, "family_id"].astype(str)
            != enriched.loc[missing_confident, "candidate_component_key"].astype(str)
        ).astype(int)

    if "dominant_name_ratio" not in enriched.columns:
        enriched["dominant_name_ratio"] = ""
    missing_ratio = enriched["dominant_name_ratio"].astype(str).str.strip() == ""
    enriched.loc[missing_ratio, "dominant_name_ratio"] = 1.0

    if "first_name_expansion_compatibility" not in enriched.columns:
        enriched["first_name_expansion_compatibility"] = ""
    missing_compat = enriched["first_name_expansion_compatibility"].astype(str).str.strip() == ""
    if missing_compat.any():
        compat_values = [
            _first_name_candidate_compatibility(row) for row in enriched.loc[missing_compat].to_dict(orient="records")
        ]
        enriched.loc[missing_compat, "first_name_expansion_compatibility"] = compat_values
    return enriched


def _drop_self_containing_rows(
    rows: pd.DataFrame,
    *,
    component_signature_ids: Mapping[str, Sequence[str]] | None,
    required_positive_pairs: set[tuple[str, str]],
    preserve_required_self_containing: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop candidates whose predicted component still contains the query signature."""

    if component_signature_ids is None or "query_signature_id" not in rows.columns:
        return rows, {
            "self_containing_rows_dropped": 0,
            "self_containing_residual_rows_preserved": 0,
            "self_containing_required_rows_preserved": 0,
        }
    component_signature_id_sets = {
        str(key): {str(value) for value in values} for key, values in component_signature_ids.items()
    }

    def truthy(value: Any) -> bool:
        text = str(value).strip().lower()
        if text in {"true", "t", "yes"}:
            return True
        try:
            return float(text) == 1.0
        except ValueError:
            return False

    def int_or_none(value: Any) -> int | None:
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if not isfinite(numeric):
            return None
        if numeric % 1.0 != 0.0:
            return None
        return int(numeric)

    columns = ["query_group_id", "query_signature_id", "candidate_component_key"]
    if "query_in_seed_before_holdout" in rows.columns:
        columns.append("query_in_seed_before_holdout")
    if "cluster_size" in rows.columns:
        columns.append("cluster_size")
    keep_mask: list[bool] = []
    dropped_pairs: set[tuple[str, str]] = set()
    residual_rows_preserved = 0
    required_rows_preserved = 0
    for row in rows[columns].itertuples(index=False):
        row_values = row._asdict()
        pair = (str(row_values["query_group_id"]), str(row_values["candidate_component_key"]))
        component_key = str(row_values["candidate_component_key"])
        query_signature_id = str(row_values["query_signature_id"])
        component_signature_set = component_signature_id_sets.get(component_key, set())
        contains_query = query_signature_id in component_signature_set
        generated_as_residual = False
        if contains_query:
            generated_as_residual = truthy(row_values.get("query_in_seed_before_holdout", False))
            cluster_size = int_or_none(row_values.get("cluster_size"))
            if cluster_size is None or cluster_size != max(0, len(component_signature_set) - 1):
                generated_as_residual = False
        preserve_required = preserve_required_self_containing and pair in required_positive_pairs
        keep_row = (not contains_query) or generated_as_residual or preserve_required
        keep_mask.append(keep_row)
        if contains_query and generated_as_residual:
            residual_rows_preserved += 1
        elif contains_query and preserve_required:
            required_rows_preserved += 1
        elif contains_query:
            dropped_pairs.add(pair)
    dropped_required = sorted(required_positive_pairs & dropped_pairs)
    if dropped_required:
        raise ValueError(f"Self-filter would drop required positive pairs: {dropped_required[:5]}")
    filtered = rows.loc[keep_mask].copy()
    return filtered, {
        "self_containing_rows_dropped": int(len(rows) - len(filtered)),
        "self_containing_residual_rows_preserved": int(residual_rows_preserved),
        "self_containing_required_rows_preserved": int(required_rows_preserved),
    }


def _apply_name_count_feature_overlay(
    rows: pd.DataFrame,
    *,
    name_count_features_by_pair: Mapping[tuple[str, str], Mapping[str, float]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing name-count rarity features from computed generated-row features."""

    if not name_count_features_by_pair:
        return rows, {"name_count_feature_cells_filled": 0, "name_count_feature_cells_missing_after_fill": 0}
    enriched = rows.copy()
    filled = 0
    for column in NAME_COUNT_RARITY_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = ""
        missing_mask = enriched[column].astype(str).str.strip() == ""
        if not missing_mask.any():
            continue
        pairs = list(
            zip(
                enriched.loc[missing_mask, "query_group_id"].astype(str),
                enriched.loc[missing_mask, "candidate_component_key"].astype(str),
                strict=True,
            )
        )
        values = [name_count_features_by_pair.get(pair, {}).get(column, "") for pair in pairs]
        filled += sum(1 for value in values if value != "")
        enriched.loc[missing_mask, column] = values
    missing_after = int(
        sum(int((enriched[column].astype(str).str.strip() == "").sum()) for column in NAME_COUNT_RARITY_FEATURE_COLUMNS)
    )
    if missing_after:
        for column in NAME_COUNT_RARITY_FEATURE_COLUMNS:
            missing_mask = enriched[column].astype(str).str.strip() == ""
            enriched.loc[missing_mask, column] = 0.0
    return enriched, {
        "name_count_feature_cells_filled": int(filled),
        "name_count_feature_cells_missing_after_fill": int(missing_after),
    }


def _refresh_group_label_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Refresh group-level positive metadata after overlaying labels."""

    refreshed = rows.copy()
    labels = pd.to_numeric(refreshed["label"], errors="coerce").fillna(0).astype(int)
    refreshed["label"] = labels
    if "binary_safe_link_target" in refreshed.columns:
        refreshed["binary_safe_link_target"] = labels

    for _query_group_id, index in refreshed.groupby("query_group_id", sort=False).groups.items():
        group = refreshed.loc[index]
        group_labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int)
        positive = group.loc[group_labels == 1].copy()
        positive_keys = sorted(positive["candidate_component_key"].astype(str).tolist())
        refreshed.loc[index, "positive_candidate_count"] = int(len(positive_keys))
        refreshed.loc[index, "positive_candidate_keys"] = "|".join(positive_keys)
        refreshed.loc[index, "group_has_positive"] = int(bool(positive_keys))
        if positive_keys:
            positive_ranks = pd.to_numeric(positive["retrieval_rank"], errors="coerce")
            refreshed.loc[index, "best_positive_retrieval_rank"] = int(positive_ranks.min())
        else:
            refreshed.loc[index, "best_positive_retrieval_rank"] = ""
    return refreshed


def _sync_hwang_candidate_level_files(
    *,
    output_root: Path,
    promoted_rows: pd.DataFrame,
    name_compat_correction_pairs: set[tuple[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Refresh H-Wang companion label files to match promoted row labels."""

    manifest_path = output_root / HWANG_CANDIDATE_LEVEL_MANIFEST_RELATIVE_PATH
    clean_overrides_path = output_root / HWANG_CLEAN_OVERRIDES_RELATIVE_PATH
    summary_path = output_root / HWANG_CANDIDATE_LEVEL_SUMMARY_RELATIVE_PATH
    if not manifest_path.exists() or not clean_overrides_path.exists():
        return {}

    labels = pd.to_numeric(promoted_rows["label"], errors="coerce").fillna(0).astype(int)
    rows = promoted_rows.copy()
    rows["label"] = labels
    row_targets = rows.groupby(rows["query_group_id"].astype(str))["label"].max()
    positive_rows_by_query = rows.groupby(rows["query_group_id"].astype(str))["label"].sum().to_dict()
    active_positive_pairs = _pair_set(rows.loc[labels == 1])

    manifest = _read_rows(manifest_path)
    manifest_query_ids = set(manifest["query_group_id"].astype(str))
    row_query_ids = set(row_targets.index)
    if manifest_query_ids != row_query_ids:
        raise ValueError(
            "H-Wang manifest query set differs from promoted rows: "
            f"manifest_only={sorted(manifest_query_ids - row_query_ids)[:5]} "
            f"row_only={sorted(row_query_ids - manifest_query_ids)[:5]}"
        )

    name_compat_queries = {query_id for query_id, _candidate_key in name_compat_correction_pairs}
    manifest_rows: list[dict[str, Any]] = []
    clean_override_rows: list[dict[str, Any]] = []
    for raw_row in manifest.to_dict(orient="records"):
        row = {str(key): value for key, value in raw_row.items()}
        query_group_id = str(row["query_group_id"])
        reviewed_candidate_key = str(row.get("reviewed_candidate_component_key") or "")
        positive_rows_after = int(positive_rows_by_query.get(query_group_id, 0))
        manual_safe_target = int(row_targets.loc[query_group_id])
        reviewed_candidate_survived = int(
            bool(reviewed_candidate_key) and (query_group_id, reviewed_candidate_key) in active_positive_pairs
        )
        label_action = str(row.get("label_action") or "keep_surviving_raw_labels")
        if manual_safe_target and query_group_id in name_compat_queries and not reviewed_candidate_survived:
            label_action = "name_compat_manual_positive"
        elif label_action == "force_no_positive" and manual_safe_target:
            label_action = "promoted_surviving_positive"
        row.update(
            {
                "dataset": str(row.get("dataset") or "h_wang"),
                "reviewed_candidate_survived": str(reviewed_candidate_survived),
                "label_action": label_action,
                "positive_rows_after_candidate_relabel": str(positive_rows_after),
                "manual_safe_target": str(manual_safe_target),
            }
        )
        manifest_rows.append(row)
        clean_override_rows.append(
            {
                "query_group_id": query_group_id,
                "manual_safe_target": str(manual_safe_target),
                "manual_assessment": label_action,
                "correction_type": str(row.get("correction_type") or "none"),
                "review_source_path": str(row.get("review_source_path") or ""),
            }
        )

    manifest_fieldnames = [
        "query_group_id",
        "dataset",
        "correction_type",
        "reviewed_candidate_component_key",
        "reviewed_candidate_survived",
        "raw_positive_rows_before_candidate_relabel",
        "label_action",
        "review_source_path",
        "positive_rows_after_candidate_relabel",
        "manual_safe_target",
    ]
    label_action_counts = Counter(str(row["label_action"]) for row in manifest_rows)
    correction_type_counts = Counter(str(row["correction_type"]) for row in manifest_rows)
    reviewed_positive_rows = [
        row for row in manifest_rows if str(row["correction_type"]) in {"top1_should_link", "non_top1_should_link"}
    ]
    summary = {
        "hwang_rows_path": str(output_root / HWANG_ROW_RELATIVE_PATH),
        "hwang_clean_overrides_path": str(clean_overrides_path),
        "manifest_path": str(manifest_path),
        "apply": True,
        "queries": int(len(row_targets)),
        "rows": int(len(promoted_rows)),
        "positive_rows_before_candidate_relabel": int(labels.sum()),
        "positive_rows_after_candidate_relabel": int(labels.sum()),
        "positive_queries_after_candidate_relabel": int(row_targets.sum()),
        "label_action_counts": {str(key): int(value) for key, value in sorted(label_action_counts.items())},
        "correction_type_counts": {str(key): int(value) for key, value in sorted(correction_type_counts.items())},
        "reviewed_positive_corrections": int(len(reviewed_positive_rows)),
        "reviewed_positive_corrections_survived": int(
            sum(int(str(row["reviewed_candidate_survived"]) or 0) for row in reviewed_positive_rows)
        ),
        "reviewed_positive_corrections_missing_after_filter": int(
            sum(1 for row in reviewed_positive_rows if int(str(row["reviewed_candidate_survived"]) or 0) == 0)
        ),
        "manifest_queries_dropped_by_initial_only_rereview": 0,
        "raw_positive_queries_before_candidate_relabel": int(row_targets.sum()),
        "no_positive_queries_after_candidate_relabel": int(len(row_targets) - int(row_targets.sum())),
        "name_compat_manual_positive_queries": int(len(name_compat_queries)),
        "name_compat_manual_positive_rows": int(
            sum(1 for pair in name_compat_correction_pairs if pair in active_positive_pairs)
        ),
    }
    if not dry_run:
        _write_plain_rows(clean_overrides_path, clean_override_rows, fieldnames=list(clean_override_rows[0]))
        _write_plain_rows(manifest_path, manifest_rows, fieldnames=manifest_fieldnames)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def promote_slice_rows(
    *,
    generated_rows: pd.DataFrame,
    official_rows: pd.DataFrame,
    corrections: pd.DataFrame,
    fieldnames: Sequence[str],
    slice_key: str,
    extra_positive_pairs: set[tuple[str, str]] | None = None,
    component_signature_ids: Mapping[str, Sequence[str]] | None = None,
    name_count_features_by_pair: Mapping[tuple[str, str], Mapping[str, float]] | None = None,
    preserve_required_self_containing: bool = False,
    materialize: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return officialized name-compat rows and a promotion summary for one slice."""

    generated_query_ids = set(generated_rows["query_group_id"].astype(str))
    official_query_ids = set(official_rows["query_group_id"].astype(str))
    if generated_query_ids != official_query_ids:
        raise ValueError(
            "Generated and official query sets differ for "
            f"{slice_key}: generated_only={sorted(generated_query_ids - official_query_ids)[:5]} "
            f"official_only={sorted(official_query_ids - generated_query_ids)[:5]}"
        )

    correction_positive_pairs = _correction_pairs(corrections, slice_key=slice_key)
    generated_pairs = _pair_set(generated_rows)
    missing_correction_pairs = sorted(correction_positive_pairs - generated_pairs)
    if missing_correction_pairs:
        raise ValueError(
            f"Name-compat corrections missing from generated rows for {slice_key}: " f"{missing_correction_pairs[:5]}"
        )

    official_positive_pairs = _positive_pairs_from_rows(official_rows)
    protected_positive_pairs = official_positive_pairs | correction_positive_pairs
    extra_positive_pairs = set(extra_positive_pairs or set()) - protected_positive_pairs
    required_positive_pairs = protected_positive_pairs | extra_positive_pairs
    promoted = generated_rows.copy()
    base_group_ids = _base_group_id_by_query(official_rows)
    if "base_group_id" in fieldnames:
        if "base_group_id" not in promoted.columns:
            promoted["base_group_id"] = ""
        missing_base = promoted["base_group_id"].astype(str).str.strip() == ""
        promoted.loc[missing_base, "base_group_id"] = (
            promoted.loc[missing_base, "query_group_id"].astype(str).map(base_group_ids)
        )
        if promoted.loc[missing_base, "base_group_id"].isna().any():
            missing_queries = sorted(
                promoted.loc[promoted["base_group_id"].isna(), "query_group_id"].astype(str).unique().tolist()
            )
            raise ValueError(f"Missing base_group_id mapping for {slice_key}: {missing_queries[:5]}")

    promoted, append_summary = _append_missing_official_positive_rows(
        promoted,
        official_rows,
        required_positive_pairs=protected_positive_pairs,
    )
    promoted, self_filter_summary = _drop_self_containing_rows(
        promoted,
        component_signature_ids=component_signature_ids,
        required_positive_pairs=protected_positive_pairs,
        preserve_required_self_containing=preserve_required_self_containing,
    )
    promoted_pairs = list(
        zip(
            promoted["query_group_id"].astype(str),
            promoted["candidate_component_key"].astype(str),
            strict=True,
        )
    )
    promoted["label"] = [int(pair in required_positive_pairs) for pair in promoted_pairs]
    promoted = _enrich_promoted_prerequisites(promoted)
    promoted, name_count_summary = _apply_name_count_feature_overlay(
        promoted,
        name_count_features_by_pair=name_count_features_by_pair,
    )
    promoted = _refresh_group_label_metadata(promoted)
    if materialize:
        materialize_frame = promoted.astype(object).where(pd.notna(promoted), None).replace({"": None})
        records = materialize_frame.to_dict(orient="records")
        promoted = pd.DataFrame(materialize_derived_rows(records))

    query_order = {
        query_id: order for order, query_id in enumerate(generated_rows["query_group_id"].astype(str).drop_duplicates())
    }
    promoted["_query_order"] = promoted["query_group_id"].astype(str).map(query_order)
    promoted["_retrieval_rank_sort"] = pd.to_numeric(promoted["retrieval_rank"], errors="coerce").fillna(999999)
    promoted = promoted.sort_values(
        by=["_query_order", "_retrieval_rank_sort", "candidate_component_key"],
        kind="mergesort",
    ).drop(columns=["_query_order", "_retrieval_rank_sort"])
    for field in fieldnames:
        if field not in promoted.columns:
            promoted[field] = ""
    promoted = promoted[[str(field) for field in fieldnames]].copy()

    labels = pd.to_numeric(promoted["label"], errors="coerce").fillna(0).astype(int)
    generated_raw_labels = pd.to_numeric(generated_rows["label"], errors="coerce").fillna(0).astype(int)
    generated_raw_positive_pairs = _pair_set(generated_rows.loc[generated_raw_labels == 1])
    promoted_pairs_set = _pair_set(promoted)
    summary = {
        "slice_key": str(slice_key),
        "official_rows_before": int(len(official_rows)),
        "generated_rows": int(len(generated_rows)),
        "promoted_rows": int(len(promoted)),
        "official_positive_pairs": int(len(official_positive_pairs)),
        "name_compat_correction_pairs": int(len(correction_positive_pairs)),
        "extra_positive_pairs": int(len(extra_positive_pairs)),
        "extra_positive_pairs_survived": int(len(extra_positive_pairs & promoted_pairs_set)),
        "required_positive_pairs": int(len(required_positive_pairs)),
        "generated_raw_positive_rows": int(generated_raw_labels.sum()),
        "generated_raw_positive_pairs_not_required": int(len(generated_raw_positive_pairs - required_positive_pairs)),
        "promoted_positive_rows": int(labels.sum()),
        "promoted_positive_queries": int(promoted.loc[labels == 1, "query_group_id"].astype(str).nunique()),
        "promoted_no_positive_queries": int(promoted["query_group_id"].astype(str).nunique())
        - int(promoted.loc[labels == 1, "query_group_id"].astype(str).nunique()),
        **append_summary,
        **self_filter_summary,
        **name_count_summary,
    }
    return promoted, summary


def _selected_configs(slice_keys: Sequence[str] | None) -> list[SlicePromotionConfig]:
    """Resolve the requested slice subset."""

    configs_by_slice = {config.slice_key: config for config in SLICE_CONFIGS}
    if not slice_keys:
        return list(SLICE_CONFIGS)
    unknown = sorted(set(slice_keys) - set(configs_by_slice))
    if unknown:
        raise ValueError(f"Unknown slice keys: {unknown}; known={sorted(configs_by_slice)}")
    return [configs_by_slice[str(slice_key)] for slice_key in slice_keys]


def promote_eval_rows(
    *,
    bundle_root: Path,
    output_root: Path,
    summary_path: Path,
    slice_keys: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote configured eval rows and write a JSON summary."""

    bundle_root = bundle_root.resolve()
    output_root = output_root.resolve()
    corrections_path = bundle_root / NAME_COMPAT_CORRECTIONS_PATH
    corrections = _read_rows(corrections_path) if corrections_path.exists() else pd.DataFrame()
    slice_summaries: list[dict[str, Any]] = []
    for config in _selected_configs(slice_keys):
        generated_path = config.generated_rows_path
        official_path = bundle_root / config.row_relative_path
        output_path = output_root / config.row_relative_path
        if not generated_path.exists():
            raise FileNotFoundError(f"Generated name-compat rows are missing: {generated_path}")
        generated_rows = _read_rows(generated_path)
        official_rows = _read_rows(official_path)
        fieldnames = _read_fieldnames(official_path)
        component_keys = set(generated_rows["candidate_component_key"].astype(str))
        component_keys.update(official_rows["candidate_component_key"].astype(str))
        component_keys.discard("")
        component_signature_ids = _load_component_signature_ids(
            config.predicted_clusters_path,
            component_keys=component_keys,
        )
        missing_name_count_columns = [
            column
            for column in NAME_COUNT_RARITY_FEATURE_COLUMNS
            if column in fieldnames
            and (column not in generated_rows.columns or (generated_rows[column].astype(str).str.strip() == "").any())
        ]
        name_count_features_by_pair = (
            _build_name_count_features_by_pair(
                generated_rows,
                component_signature_ids=component_signature_ids,
                signatures_path=config.signatures_path,
            )
            if missing_name_count_columns
            else None
        )
        promoted_rows, summary = promote_slice_rows(
            generated_rows=generated_rows,
            official_rows=official_rows,
            corrections=corrections,
            fieldnames=fieldnames,
            slice_key=config.slice_key,
            component_signature_ids=component_signature_ids,
            name_count_features_by_pair=name_count_features_by_pair,
        )
        if config.row_relative_path == HWANG_ROW_RELATIVE_PATH:
            hwang_manifest_summary = _sync_hwang_candidate_level_files(
                output_root=output_root,
                promoted_rows=promoted_rows,
                name_compat_correction_pairs=_correction_pairs(corrections, slice_key=config.slice_key),
                dry_run=bool(dry_run),
            )
            if hwang_manifest_summary:
                summary["hwang_candidate_level_manifest"] = hwang_manifest_summary
        summary.update(
            {
                "dataset": config.dataset,
                "generated_rows_path": str(generated_path.relative_to(REPO_ROOT)).replace("/", "\\"),
                "output_path": str(output_path.relative_to(REPO_ROOT)).replace("/", "\\")
                if output_path.is_relative_to(REPO_ROOT)
                else str(output_path),
                "dry_run": bool(dry_run),
            }
        )
        if not dry_run:
            _write_rows(output_path, promoted_rows, fieldnames=fieldnames)
        slice_summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    overall = {
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "dry_run": bool(dry_run),
        "slice_summaries": slice_summaries,
        "total_promoted_rows": int(sum(item["promoted_rows"] for item in slice_summaries)),
        "total_generated_rows": int(sum(item["generated_rows"] for item in slice_summaries)),
        "total_name_compat_correction_pairs": int(
            sum(item["name_compat_correction_pairs"] for item in slice_summaries)
        ),
        "total_official_positive_rows_appended": int(
            sum(item["official_positive_rows_appended"] for item in slice_summaries)
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return overall


def main() -> None:
    """Run the promotion CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--slice", dest="slices", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = promote_eval_rows(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        summary_path=args.summary_json,
        slice_keys=args.slices,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
