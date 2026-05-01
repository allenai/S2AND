"""Validate the active official joint safe-link bundle."""

from __future__ import annotations

# ruff: noqa: E402
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

try:
    from scripts.joint_safe_link_dataset_contract import (
        CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH,
        DEFAULT_FILTER_POLICY,
        FILTER_POLICY_RELATIVE_PATH,
        apply_self_containment_filter,
        compare_ledger_to_current_bundle,
    )
    from scripts.joint_safe_link_official_stack import load_bundle
except ImportError:  # pragma: no cover - direct script execution path
    from joint_safe_link_dataset_contract import (  # type: ignore
        CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH,
        CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH,
        DEFAULT_FILTER_POLICY,
        FILTER_POLICY_RELATIVE_PATH,
        apply_self_containment_filter,
        compare_ledger_to_current_bundle,
    )
    from joint_safe_link_official_stack import load_bundle  # type: ignore

from scripts.reranker_dataset.bundle import CLASSIC_GATE_ONLY_CALIBRATION_SURFACE, RerankerBundleContract
from scripts.reranker_dataset.schema import FeatureSchema
from scripts.single_letter_retrieval_utils import invert_signature_to_cluster_id, load_preferred_signature_to_cluster_id

BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
NEW_BLOCK_DATASETS = ("j_smith", "a_khan", "a_silva", "s_gupta")
GIANT_STEP2_DIRS = {
    "a_khan": REPO_ROOT / "scratch" / "a_khan_multi_letter_v12_15000",
    "a_silva": REPO_ROOT / "scratch" / "a_silva_multi_letter_v12_15000",
    "h_wang": REPO_ROOT / "scratch" / "h_wang_multi_letter_v12_15000",
    "j_smith": REPO_ROOT / "scratch" / "j_smith_multi_letter_v12_15000",
    "s_gupta": REPO_ROOT / "scratch" / "s_gupta_multi_letter_v12_15000",
    "s_lee": REPO_ROOT / "scratch" / "s_lee_multi_letter_v12_15000",
    "s_park": REPO_ROOT / "scratch" / "s_park_multi_letter_v12_15000",
}
REQUIRED_LOCAL_FILES = (
    Path("bundle.json"),
    Path("README.md"),
    Path("PROVENANCE.md"),
    Path("training")
    / (
        "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_"
        "neg100_plus_reviewed_splitpos_hardneg_rows.csv.gz"
    ),
    Path("calibration") / "classic_gate_possible_manual_w5_rows.csv.gz",
    Path("calibration") / "classic_gate_possible_manual_w5_base_groups.csv",
    Path("calibration") / "stratified_eval_test_split" / "combined_query_split_assignments.csv",
    Path("calibration") / "stratified_eval_test_split" / "stratum_balance.csv",
    Path("calibration") / "stratified_eval_test_split" / "summary.json",
    Path("calibration") / "stratified_eval_test_split" / "report.md",
    Path("calibration") / "total_error_4score_2margin_gate" / "selected_gate.json",
    Path("calibration") / "total_error_4score_2margin_gate" / "gate_candidate_metrics.csv",
    Path("calibration") / "total_error_4score_2margin_gate" / "summary.json",
    Path("calibration") / "total_error_4score_2margin_gate" / "report.md",
    Path("calibration") / "greedy_best_check23_feature_selection" / "exact_2000_tree_confirmation.json",
    Path("calibration") / "greedy_best_check23_feature_selection" / "trajectory.csv",
    Path("calibration") / "greedy_best_check23_feature_selection" / "summary.json",
    FILTER_POLICY_RELATIVE_PATH,
    CUSTOM_LABEL_LEDGER_RELATIVE_PATH,
    CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH,
    CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH,
    CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH,
    Path("test") / "classic_gate_internal_eval_base_groups.csv",
    Path("test") / "s2and_eval_rows.csv.gz",
    Path("test") / "hwang_eval_rows.csv.gz",
    Path("test") / "hwang_candidate_level_label_overrides.csv",
    Path("test") / "hwang_candidate_level_label_overrides_summary.json",
    Path("test") / "s_park_eval_rows.csv.gz",
    Path("test") / "s_lee_eval_rows.csv.gz",
    Path("test") / "j_smith_eval_rows.csv.gz",
    Path("test") / "a_khan_eval_rows.csv.gz",
    Path("test") / "a_silva_eval_rows.csv.gz",
    Path("test") / "s_gupta_eval_rows.csv.gz",
    Path("test") / "training_s2and_source_reviewed_eval_rows.csv.gz",
    Path("test") / "s2and_extra_no_positive_eval_rows.csv.gz",
    Path("test") / "hwang_cleaned_eval_overrides.csv",
)
PATH_KEYS = {
    "path",
    "calibration_path",
    "evaluation_path",
    "s_park_eval_path",
    "s_lee_eval_path",
    "manual_holdout_candidates_path",
    "assignments_path",
    "stratum_balance_path",
    "summary_path",
    "report_path",
    "selected_gate_path",
    "candidate_metrics_path",
    "exact_confirmation_path",
    "trajectory_path",
    "selection_artifact",
    "filter_policy_path",
    "custom_label_ledger_path",
    "custom_label_ledger_summary_path",
    "custom_label_ledger_comparison_path",
    "custom_label_ledger_report_path",
    "augmentation_summary_path",
}


def _resolve_local_bundle_path(bundle_root: Path, path_like: str) -> Path:
    """Resolve one stored bundle path and require that it stays inside the bundle root."""

    path = Path(path_like)
    if path.is_absolute():
        raise ValueError(f"Bundle path must be relative, got absolute path: {path_like}")
    resolved = (bundle_root / path).resolve()
    try:
        resolved.relative_to(bundle_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Bundle path escapes bundle root: {path_like}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"Bundle path does not exist: {path_like}")
    return resolved


def _bundle_path_entries(node: Any, *, prefix: str = "") -> list[tuple[str, str]]:
    """Collect the explicit file-path entries from bundle metadata."""

    entries: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            node_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key in PATH_KEYS and isinstance(value, str):
                entries.append((node_prefix, value))
                continue
            if key == "extra_eval_paths" and isinstance(value, dict):
                for dataset_name, path_like in value.items():
                    entries.append((f"{node_prefix}.{dataset_name}", str(path_like)))
                continue
            entries.extend(_bundle_path_entries(value, prefix=node_prefix))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            entries.extend(_bundle_path_entries(value, prefix=f"{prefix}[{index}]"))
    return entries


def _hardlinked_relative_files(root: Path) -> list[str]:
    """Return bundle-relative files that still have multiple hard links."""

    linked: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if int(path.stat().st_nlink) > 1:
            linked.append(str(path.relative_to(root)).replace("/", "\\"))
    return sorted(linked)


def _empty_feature_coverage() -> dict[str, Any]:
    """Return an empty coverage accumulator."""

    return {
        "rows": 0,
        "query_ids": set(),
        "rows_with_missing_features": 0,
        "missing_feature_cells": 0,
        "columns_with_missing": set(),
        "columns_absent": set(),
    }


def _raw_feature_columns_for_validation(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the active feature columns that should exist as raw CSV fields on disk."""

    runtime_derived_columns = {
        "cluster_size_log_capped",
        "query_first_prefix_match",
        "anchor_evidence_count",
        "strong_positive_anchor_score",
        "weak_residual_anchor_score",
        "sparse_relative_winner_score",
        "query_view__full",
        "query_view__initial_only",
    }
    return tuple(column for column in feature_columns if column not in runtime_derived_columns)


def _training_dataset_names(path: Path) -> tuple[str, ...]:
    """Return the dataset blocks represented in the persisted training rows."""

    datasets = (
        pd.read_csv(path, usecols=["dataset"], low_memory=False)["dataset"].dropna().astype(str).unique().tolist()
    )
    return tuple(sorted(str(value) for value in datasets))


def _finalize_feature_coverage(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert coverage accumulators into JSON-serializable summaries."""

    return {
        dataset: {
            "rows": int(payload["rows"]),
            "queries": int(len(payload["query_ids"])),
            "rows_with_missing_features": int(payload["rows_with_missing_features"]),
            "missing_feature_cells": int(payload["missing_feature_cells"]),
            "columns_with_missing": sorted(str(value) for value in payload["columns_with_missing"]),
            "columns_absent": sorted(str(value) for value in payload["columns_absent"]),
        }
        for dataset, payload in stats.items()
    }


def _accumulate_feature_coverage(
    accumulator: dict[str, Any],
    dataset_rows: pd.DataFrame,
    *,
    present_feature_columns: tuple[str, ...],
) -> None:
    """Accumulate feature coverage counts from one filtered frame."""

    accumulator["rows"] += int(len(dataset_rows))
    accumulator["query_ids"].update(dataset_rows["query_group_id"].astype(str))
    missing = dataset_rows.loc[:, list(present_feature_columns)].isna()
    accumulator["rows_with_missing_features"] += int(missing.any(axis=1).sum())
    accumulator["missing_feature_cells"] += int(missing.to_numpy().sum())
    accumulator["columns_with_missing"].update(
        column for column in present_feature_columns if bool(missing[column].any())
    )


def _feature_coverage_failures(coverage_sections: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Return feature coverage failures for active raw feature columns."""

    failures: list[str] = []
    for split_name, section in coverage_sections.items():
        for dataset, payload in section.items():
            absent = payload["columns_absent"]
            missing = payload["columns_with_missing"]
            if absent:
                failures.append(f"{split_name}:{dataset}:absent:{absent}")
            if missing:
                failures.append(f"{split_name}:{dataset}:missing:{missing}")
    return failures


def _load_component_signature_lookup(dataset_name: str) -> dict[str, frozenset[str]]:
    """Load component membership for self-containment validation."""

    labeled_clusters_path = REPO_ROOT / "data" / dataset_name / f"{dataset_name}_clusters.json"
    if labeled_clusters_path.exists():
        clusters = json.loads(labeled_clusters_path.read_text(encoding="utf-8"))
        return {
            str(cluster_id): frozenset(str(signature_id) for signature_id in cluster["signature_ids"])
            for cluster_id, cluster in clusters.items()
        }
    step2_dir = GIANT_STEP2_DIRS.get(dataset_name)
    if step2_dir is not None and step2_dir.exists():
        signature_to_cluster_id, _assignment_info = load_preferred_signature_to_cluster_id(step2_dir)
        preferred_clusters = invert_signature_to_cluster_id(signature_to_cluster_id)
        lookup = {
            str(cluster_id): frozenset(str(signature_id) for signature_id in signature_ids)
            for cluster_id, signature_ids in preferred_clusters.items()
        }
        predicted_clusters_path = step2_dir / "predicted_clusters.json"
        if predicted_clusters_path.exists():
            predicted_clusters = json.loads(predicted_clusters_path.read_text(encoding="utf-8"))
            for block_key, cluster_map in predicted_clusters.items():
                for component_key, signature_ids in dict(cluster_map).items():
                    signatures = frozenset(str(signature_id) for signature_id in signature_ids)
                    lookup.setdefault(str(component_key), signatures)
                    lookup.setdefault(f"{block_key}::{component_key}", signatures)
        return lookup
    predicted_clusters_path = GIANT_STEP2_DIRS.get(dataset_name, Path()) / "predicted_clusters.json"
    if predicted_clusters_path.exists():
        predicted_clusters = json.loads(predicted_clusters_path.read_text(encoding="utf-8"))
        lookup: dict[str, frozenset[str]] = {}
        for block_key, cluster_map in predicted_clusters.items():
            for component_key, signature_ids in dict(cluster_map).items():
                signatures = frozenset(str(signature_id) for signature_id in signature_ids)
                lookup[str(component_key)] = signatures
                lookup[f"{block_key}::{component_key}"] = signatures
        return lookup
    raise FileNotFoundError(f"No component signature source found for dataset={dataset_name!r}")


def _candidate_signature_ids(
    component_lookup: dict[str, frozenset[str]],
    *,
    candidate_component_key: Any,
    candidate_cluster_id: Any,
) -> frozenset[str]:
    """Return raw candidate member signatures, trying both persisted component ids."""

    component_key = str(candidate_component_key)
    if component_key in component_lookup:
        return component_lookup[component_key]
    cluster_id = str(candidate_cluster_id)
    if cluster_id in component_lookup:
        return component_lookup[cluster_id]
    if "::" in component_key:
        component_suffix = component_key.split("::", 1)[1]
        if component_suffix in component_lookup:
            return component_lookup[component_suffix]
    return frozenset()


def _truthy_validation_flag(value: Any) -> bool:
    """Return whether a persisted validation flag is true."""

    text = str(value).strip().lower()
    if text in {"true", "t", "yes"}:
        return True
    try:
        return float(text) == 1.0
    except ValueError:
        return False


def _int_or_none(value: Any) -> int | None:
    """Return an integer parsed from persisted row text, if available."""

    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def _is_allowed_generated_residual_self_containing_row(
    row: dict[str, Any],
    *,
    component_signature_ids: frozenset[str],
) -> bool:
    """Return whether a generated self-containing row records residual holdout materialization."""

    if not _truthy_validation_flag(row.get("query_in_seed_before_holdout", False)):
        return False
    cluster_size = _int_or_none(row.get("cluster_size", ""))
    return cluster_size == max(0, len(component_signature_ids) - 1)


def _is_allowed_residual_loo_self_containing_row(row: dict[str, Any]) -> bool:
    """Return whether a self-containing row is an explicitly materialized residual LOO positive."""

    if int(row["label"]) != 1:
        return False
    source = str(row.get("source", ""))
    split = str(row.get("split", ""))
    return (source == "labeled_loo" and split in {"train_loo", "eval_loo"}) or (
        source == "s2and_rescue_manual_review" and split in {"train", "calibration_fit", "calibration_check", "test"}
    )


def _is_allowed_required_positive_self_containing_row(row: dict[str, Any]) -> bool:
    """Return whether a manual required positive row was intentionally preserved for accounting."""

    if int(row["label"]) != 1:
        return False
    positive_candidate_keys = str(row.get("positive_candidate_keys", "")).strip()
    if positive_candidate_keys == "" or positive_candidate_keys.lower() == "nan":
        return False
    return str(row["candidate_component_key"]) in positive_candidate_keys


def _summarize_self_containing_candidate_rows(
    row_paths: tuple[Path, ...],
    *,
    component_lookup_loader: Any = _load_component_signature_lookup,
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Return rows where the candidate component still contains the query signature."""

    usecols = [
        "dataset",
        "source",
        "split",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
        "query_in_seed_before_holdout",
        "cluster_size",
        "positive_candidate_keys",
    ]
    selected_columns = set(usecols)
    component_lookup_cache: dict[str, dict[str, frozenset[str]]] = {}
    file_summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_positive_rows = 0
    total_allowed_loo_rows = 0
    total_allowed_generated_residual_rows = 0
    total_allowed_required_positive_rows = 0
    for row_path in row_paths:
        file_rows = 0
        file_positive_rows = 0
        file_allowed_loo_rows = 0
        file_allowed_generated_residual_rows = 0
        file_allowed_required_positive_rows = 0
        sample_rows: list[dict[str, Any]] = []
        allow_required_positive_rows = "training" not in {part.lower() for part in row_path.parts}
        reader = pd.read_csv(
            row_path,
            usecols=lambda column: column in selected_columns,
            compression="gzip" if row_path.suffix == ".gz" else None,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            chunk["dataset"] = chunk["dataset"].astype(str)
            chunk["query_group_id"] = chunk["query_group_id"].astype(str)
            chunk["query_signature_id"] = chunk["query_signature_id"].astype(str)
            chunk["candidate_component_key"] = chunk["candidate_component_key"].astype(str)
            chunk["candidate_cluster_id"] = chunk["candidate_cluster_id"].astype(str)
            chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int)
            if "source" not in chunk.columns:
                chunk["source"] = ""
            if "split" not in chunk.columns:
                chunk["split"] = ""
            if "query_in_seed_before_holdout" not in chunk.columns:
                chunk["query_in_seed_before_holdout"] = ""
            if "cluster_size" not in chunk.columns:
                chunk["cluster_size"] = ""
            if "positive_candidate_keys" not in chunk.columns:
                chunk["positive_candidate_keys"] = ""
            chunk["source"] = chunk["source"].astype(str)
            chunk["split"] = chunk["split"].astype(str)
            for dataset_name, dataset_rows in chunk.groupby("dataset", sort=False):
                dataset_name_text = str(dataset_name)
                lookup = component_lookup_cache.get(dataset_name_text)
                if lookup is None:
                    lookup = component_lookup_loader(dataset_name_text)
                    component_lookup_cache[dataset_name_text] = lookup
                filter_result = apply_self_containment_filter(
                    dataset_rows.to_dict(orient="records"),
                    contains_query_signature=lambda row, active_lookup=lookup: str(row["query_signature_id"])
                    in _candidate_signature_ids(
                        active_lookup,
                        candidate_component_key=row["candidate_component_key"],
                        candidate_cluster_id=row["candidate_cluster_id"],
                    ),
                )
                for row in filter_result.dropped_rows:
                    component_signature_ids = _candidate_signature_ids(
                        lookup,
                        candidate_component_key=row["candidate_component_key"],
                        candidate_cluster_id=row["candidate_cluster_id"],
                    )
                    if _is_allowed_generated_residual_self_containing_row(
                        row,
                        component_signature_ids=component_signature_ids,
                    ):
                        file_allowed_generated_residual_rows += 1
                        total_allowed_generated_residual_rows += 1
                        continue
                    if _is_allowed_residual_loo_self_containing_row(row):
                        file_allowed_loo_rows += 1
                        total_allowed_loo_rows += 1
                        continue
                    if allow_required_positive_rows and _is_allowed_required_positive_self_containing_row(row):
                        file_allowed_required_positive_rows += 1
                        total_allowed_required_positive_rows += 1
                        continue
                    file_rows += 1
                    total_rows += 1
                    if int(row["label"]) == 1:
                        file_positive_rows += 1
                        total_positive_rows += 1
                    if len(sample_rows) < 5:
                        sample_rows.append(
                            {
                                "dataset": str(dataset_name),
                                "query_group_id": str(row["query_group_id"]),
                                "query_signature_id": str(row["query_signature_id"]),
                                "candidate_component_key": str(row["candidate_component_key"]),
                                "candidate_cluster_id": str(row["candidate_cluster_id"]),
                                "label": int(row["label"]),
                            }
                        )
        file_summaries.append(
            {
                "path": str(row_path),
                "self_containing_rows": int(file_rows),
                "self_containing_positive_rows": int(file_positive_rows),
                "allowed_residual_loo_self_containing_rows": int(file_allowed_loo_rows),
                "allowed_generated_residual_self_containing_rows": int(file_allowed_generated_residual_rows),
                "allowed_required_positive_self_containing_rows": int(file_allowed_required_positive_rows),
                "samples": sample_rows,
            }
        )
    return {
        "self_containing_rows": int(total_rows),
        "self_containing_positive_rows": int(total_positive_rows),
        "allowed_residual_loo_self_containing_rows": int(total_allowed_loo_rows),
        "allowed_generated_residual_self_containing_rows": int(total_allowed_generated_residual_rows),
        "allowed_required_positive_self_containing_rows": int(total_allowed_required_positive_rows),
        "files": file_summaries,
    }


def _expected_metric_value(expected_metrics: dict[str, Any], primary_key: str, fallback_key: str) -> float:
    """Return the expected gate value, allowing older unbucketed metric keys."""

    if primary_key in expected_metrics:
        return float(expected_metrics[primary_key])
    return float(expected_metrics[fallback_key])


def _summarize_total_error_gate_consistency(bundle: Any) -> dict[str, Any]:
    """Validate that selected-gate artifacts match the frozen expected metrics."""

    gate_asset = bundle.assets["calibration"].get("total_error_4score_2margin_gate")
    if not isinstance(gate_asset, dict):
        return {"checks": []}
    selected_gate = json.loads(
        _resolve_local_bundle_path(bundle.root, str(gate_asset["selected_gate_path"])).read_text(encoding="utf-8")
    )
    expected_metrics = dict(bundle.expected_metrics["classic"])
    comparisons = (
        (
            "score_thresholds",
            "multi_candidate|multi_letter_first",
            "multi_candidate_multi_letter_score_threshold",
            "score_threshold",
        ),
        (
            "score_thresholds",
            "multi_candidate|single_letter_first",
            "multi_candidate_single_letter_score_threshold",
            "score_threshold",
        ),
        (
            "score_thresholds",
            "single_candidate|multi_letter_first",
            "single_candidate_multi_letter_score_threshold",
            "single_candidate_score_threshold",
        ),
        (
            "score_thresholds",
            "single_candidate|single_letter_first",
            "single_candidate_single_letter_score_threshold",
            "single_candidate_score_threshold",
        ),
        (
            "margin_thresholds",
            "multi_candidate|multi_letter_first",
            "multi_candidate_multi_letter_margin_threshold",
            "margin_threshold",
        ),
        (
            "margin_thresholds",
            "multi_candidate|single_letter_first",
            "multi_candidate_single_letter_margin_threshold",
            "margin_threshold",
        ),
    )
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for section, bucket_key, expected_key, fallback_key in comparisons:
        if bucket_key not in selected_gate[section]:
            continue
        expected_value = _expected_metric_value(expected_metrics, expected_key, fallback_key)
        actual_value = float(selected_gate[section][bucket_key])
        diff = abs(actual_value - expected_value)
        check = {
            "section": section,
            "bucket": bucket_key,
            "expected_metric_key": expected_key if expected_key in expected_metrics else fallback_key,
            "artifact_value": actual_value,
            "expected_value": expected_value,
            "absolute_difference": diff,
        }
        checks.append(check)
        if diff > 1e-12:
            failures.append(check)
    if failures:
        raise ValueError(f"Total-error gate artifact does not match expected metrics: {failures}")
    return {
        "selected_gate_name": str(selected_gate["name"]),
        "checks": checks,
    }


def _summarize_hwang_candidate_level_label_consistency(
    *,
    hwang_rows_path: Path,
    clean_overrides_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate H-Wang clean targets against post-filter candidate-level labels."""

    hwang_rows = pd.read_csv(
        hwang_rows_path,
        compression="gzip" if hwang_rows_path.suffix == ".gz" else None,
        usecols=["query_group_id", "label"],
        low_memory=False,
    )
    hwang_rows["label"] = pd.to_numeric(hwang_rows["label"], errors="coerce").fillna(0).astype(int)
    row_targets = (
        hwang_rows.groupby(hwang_rows["query_group_id"].astype(str))["label"]
        .max()
        .rename("candidate_level_safe_target")
    )

    clean_overrides = pd.read_csv(clean_overrides_path, low_memory=False)
    clean_overrides["query_group_id"] = clean_overrides["query_group_id"].astype(str)
    clean_overrides["manual_safe_target"] = (
        pd.to_numeric(clean_overrides["manual_safe_target"], errors="coerce").fillna(0).astype(int)
    )
    override_targets = clean_overrides.set_index("query_group_id")["manual_safe_target"]

    manifest = pd.read_csv(manifest_path, low_memory=False)
    manifest["query_group_id"] = manifest["query_group_id"].astype(str)
    manifest["manual_safe_target"] = (
        pd.to_numeric(manifest["manual_safe_target"], errors="coerce").fillna(0).astype(int)
    )
    manifest_targets = manifest.set_index("query_group_id")["manual_safe_target"]

    expected_queries = set(row_targets.index)
    override_queries = set(override_targets.index)
    manifest_queries = set(manifest_targets.index)
    failures: list[str] = []
    if expected_queries != override_queries:
        failures.append(
            "hwang_clean_override_query_set_mismatch:"
            f"missing={sorted(expected_queries - override_queries)[:5]} "
            f"extra={sorted(override_queries - expected_queries)[:5]}"
        )
    if expected_queries != manifest_queries:
        failures.append(
            "hwang_manifest_query_set_mismatch:"
            f"missing={sorted(expected_queries - manifest_queries)[:5]} "
            f"extra={sorted(manifest_queries - expected_queries)[:5]}"
        )
    common_override_queries = sorted(expected_queries & override_queries)
    if common_override_queries:
        override_diff = row_targets.loc[common_override_queries] != override_targets.loc[common_override_queries]
        if bool(override_diff.any()):
            failures.append(
                "hwang_clean_override_target_mismatch:"
                f"{list(row_targets.loc[common_override_queries][override_diff].index[:5])}"
            )
    common_manifest_queries = sorted(expected_queries & manifest_queries)
    if common_manifest_queries:
        manifest_diff = row_targets.loc[common_manifest_queries] != manifest_targets.loc[common_manifest_queries]
        if bool(manifest_diff.any()):
            failures.append(
                "hwang_manifest_target_mismatch:"
                f"{list(row_targets.loc[common_manifest_queries][manifest_diff].index[:5])}"
            )
    if failures:
        raise ValueError(f"H-Wang candidate-level label validation failed: {failures}")

    return {
        "queries": int(len(row_targets)),
        "positive_queries": int(row_targets.sum()),
        "no_positive_queries": int((row_targets == 0).sum()),
        "positive_rows": int(hwang_rows["label"].sum()),
        "manifest_label_action_counts": {
            str(key): int(value) for key, value in manifest["label_action"].value_counts().sort_index().items()
        },
    }


def _summarize_dataset_contract(bundle_root: Path) -> dict[str, Any]:
    """Validate the canonical filter policy and custom-label ledger against current rows."""

    filter_policy_path = bundle_root / FILTER_POLICY_RELATIVE_PATH
    ledger_path = bundle_root / CUSTOM_LABEL_LEDGER_RELATIVE_PATH
    filter_policy = json.loads(filter_policy_path.read_text(encoding="utf-8"))
    if filter_policy != DEFAULT_FILTER_POLICY:
        raise ValueError(
            "Dataset contract filter policy differs from the canonical policy: "
            f"actual={filter_policy} expected={DEFAULT_FILTER_POLICY}"
        )
    ledger = pd.read_csv(ledger_path, low_memory=False, keep_default_na=False)
    comparison = compare_ledger_to_current_bundle(ledger, bundle_root=bundle_root)
    if int(comparison["fatal_mismatch_count"]) != 0:
        raise ValueError(
            "Custom label ledger does not reproduce current bundle labels: "
            f"fatal_mismatch_count={comparison['fatal_mismatch_count']}"
        )
    stored_summary = json.loads((bundle_root / CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH).read_text(encoding="utf-8"))
    if int(stored_summary["comparison_fatal_mismatch_count"]) != 0:
        raise ValueError(f"Stored custom label ledger summary contains mismatches: {stored_summary}")
    return {
        "filter_policy": filter_policy,
        "custom_label_ledger": {
            "rows": int(len(ledger)),
            "slice_counts": {
                str(key): int(value) for key, value in ledger["slice_key"].value_counts().sort_index().items()
            },
            "action_counts": {
                str(key): int(value) for key, value in ledger["action"].value_counts().sort_index().items()
            },
            "comparison_fatal_mismatch_count": int(comparison["fatal_mismatch_count"]),
            "comparison_slices": {
                str(slice_key): {
                    "ledger_rows": int(payload["ledger_rows"]),
                    "active_queries": int(payload["active_rows"]["queries"]),
                    "active_positive_rows": int(payload["active_rows"]["positive_rows"]),
                    "fatal_mismatch_count": int(payload["fatal_mismatch_count"]),
                }
                for slice_key, payload in sorted(comparison["slices"].items())
            },
        },
    }


def _summarize_reranker_bundle_contracts(bundle_root: Path) -> dict[str, Any]:
    """Validate optional reranker schema and bundle-contract artifacts."""

    feature_schema_path = bundle_root / "feature_schema.json"
    bundle_contract_path = bundle_root / "bundle_contract.json"
    calibrator_path = bundle_root / "calibrator.pkl"
    calibrator_summary_path = bundle_root / "calibrator_summary.json"
    feature_schema = (
        FeatureSchema.from_json_dict(json.loads(feature_schema_path.read_text(encoding="utf-8")))
        if feature_schema_path.exists()
        else None
    )
    bundle_contract = RerankerBundleContract.read_json(bundle_contract_path) if bundle_contract_path.exists() else None
    if feature_schema is not None and bundle_contract is not None:
        feature_schema.assert_matches(bundle_contract.feature_schema.feature_columns)

    if calibrator_summary_path.exists() != calibrator_path.exists():
        raise ValueError("Reranker calibrator artifacts must include both calibrator.pkl and calibrator_summary.json")
    if (
        bundle_contract is not None
        and bundle_contract.calibration_surface != CLASSIC_GATE_ONLY_CALIBRATION_SURFACE
        and not calibrator_path.exists()
    ):
        raise ValueError(
            "Reranker calibrator artifacts are required when calibration_surface is "
            f"{bundle_contract.calibration_surface!r}"
        )

    calibrator_feature_schema: FeatureSchema | None = None
    calibrator_metadata: dict[str, Any] | None = None
    calibrator_summary: dict[str, Any] | None = None
    if calibrator_path.exists():
        if feature_schema is None or bundle_contract is None:
            raise ValueError("Reranker calibrator artifacts require feature_schema.json and bundle_contract.json")
        with calibrator_path.open("rb") as handle:
            calibrator_payload = pickle.load(handle)  # noqa: S301
        if not isinstance(calibrator_payload, dict):
            raise ValueError("calibrator.pkl must contain a metadata dictionary")
        calibrator_feature_schema = FeatureSchema.from_json_dict(dict(calibrator_payload["feature_schema"]))
        feature_schema.assert_matches(calibrator_feature_schema.feature_columns)
        calibrator_metadata = dict(calibrator_payload.get("calibration", {}))
        calibrator_summary = json.loads(calibrator_summary_path.read_text(encoding="utf-8"))
        if not isinstance(calibrator_summary, dict):
            raise ValueError("calibrator_summary.json must contain an object")
        if calibrator_summary != calibrator_metadata:
            raise ValueError("calibrator_summary.json does not match calibrator.pkl calibration metadata")
        calibrator_schema_digest = str(calibrator_metadata.get("feature_schema_digest", ""))
        if calibrator_schema_digest != calibrator_feature_schema.digest:
            raise ValueError(
                "Calibrator feature schema digest mismatch: "
                f"metadata={calibrator_schema_digest!r} schema={calibrator_feature_schema.digest!r}"
            )
        calibrator_surface = str(calibrator_metadata.get("surface", ""))
        if calibrator_surface != bundle_contract.calibration_surface:
            raise ValueError(
                "Calibrator surface mismatch: "
                f"metadata={calibrator_surface!r} bundle_contract={bundle_contract.calibration_surface!r}"
            )
        overlap = calibrator_metadata.get("inner_split_group_overlap_with_training")
        if overlap is not None and int(overlap) != 0:
            raise ValueError(f"Calibrator inner split overlaps training groups: {overlap}")

    return {
        "feature_schema_present": feature_schema is not None,
        "bundle_contract_present": bundle_contract is not None,
        "feature_schema_digest": feature_schema.digest if feature_schema is not None else None,
        "bundle_contract_feature_schema_digest": (
            bundle_contract.feature_schema.digest if bundle_contract is not None else None
        ),
        "calibration_surface": bundle_contract.calibration_surface if bundle_contract is not None else None,
        "calibrator_present": calibrator_path.exists(),
        "calibrator_summary_present": calibrator_summary_path.exists(),
        "calibrator_feature_schema_digest": (
            calibrator_feature_schema.digest if calibrator_feature_schema is not None else None
        ),
        "calibrator_summary_feature_schema_digest": (
            str(calibrator_summary["feature_schema_digest"]) if calibrator_summary is not None else None
        ),
        "calibrator_surface": str(calibrator_metadata["surface"]) if calibrator_metadata is not None else None,
    }


def _summarize_active_feature_coverage(
    path: Path,
    *,
    feature_columns: tuple[str, ...],
    datasets: tuple[str, ...],
    chunksize: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Summarize active classic feature completeness for selected datasets in one row file."""

    read_header_kwargs: dict[str, Any] = {"low_memory": False, "nrows": 0}
    if path.suffix == ".gz":
        read_header_kwargs["compression"] = "gzip"
    available_columns = set(pd.read_csv(path, **read_header_kwargs).columns)
    present_feature_columns = tuple(column for column in feature_columns if column in available_columns)
    absent_feature_columns = tuple(column for column in feature_columns if column not in available_columns)
    usecols = ["dataset", "query_group_id", *present_feature_columns]
    accumulators = {dataset: _empty_feature_coverage() for dataset in datasets}
    for accumulator in accumulators.values():
        accumulator["columns_absent"].update(absent_feature_columns)
    read_kwargs: dict[str, Any] = {"low_memory": False, "usecols": usecols}
    if path.suffix == ".gz":
        read_kwargs["compression"] = "gzip"
    if chunksize is not None:
        read_kwargs["chunksize"] = int(chunksize)
    reader = pd.read_csv(path, **read_kwargs)
    frames = reader if chunksize is not None else (reader,)
    dataset_filter = set(datasets)
    for frame in frames:
        frame["dataset"] = frame["dataset"].astype(str)
        filtered = frame[frame["dataset"].isin(dataset_filter)].copy()
        if filtered.empty:
            continue
        for dataset, dataset_rows in filtered.groupby("dataset", sort=False):
            accumulator = accumulators[str(dataset)]
            _accumulate_feature_coverage(
                accumulator,
                dataset_rows,
                present_feature_columns=present_feature_columns,
            )
    unmatched_datasets = [dataset for dataset, accumulator in accumulators.items() if int(accumulator["rows"]) == 0]
    if unmatched_datasets:
        raise ValueError(f"Dataset filter matched zero rows in {path}: datasets={unmatched_datasets}")
    return _finalize_feature_coverage(accumulators)


def build_validation_payload(bundle_root: Path = BUNDLE_ROOT) -> dict[str, object]:
    """Return a compact validation payload for one official bundle."""

    bundle_root = bundle_root.resolve()
    bundle = load_bundle(bundle_root)
    gate_split = bundle.assets["calibration"]["classic_gate_split"]
    extra_eval_paths = bundle.models["classic"].get("extra_eval_paths", {})
    required_files = [str(path).replace("/", "\\") for path in REQUIRED_LOCAL_FILES]
    missing_required_files = [
        relative_path for relative_path in required_files if not (bundle.root / relative_path).exists()
    ]
    if missing_required_files:
        raise FileNotFoundError(f"Bundle is missing required local files: {missing_required_files}")

    path_entries = _bundle_path_entries(
        {
            "assets": bundle.assets,
            "models": bundle.models,
        }
    )
    resolved_entries = {
        entry_key: str(_resolve_local_bundle_path(bundle.root, path_value).relative_to(bundle.root)).replace("/", "\\")
        for entry_key, path_value in path_entries
    }

    hardlinked_files = _hardlinked_relative_files(bundle.root)
    if hardlinked_files:
        raise ValueError(f"Bundle still contains hard-linked files: {hardlinked_files[:10]}")

    feature_columns = _raw_feature_columns_for_validation(
        tuple(str(column) for column in bundle.models["classic"]["feature_columns"])
    )
    training_path = _resolve_local_bundle_path(bundle.root, str(bundle.models["classic"]["train_path"]))
    training_datasets = _training_dataset_names(training_path)
    calibration_path = _resolve_local_bundle_path(
        bundle.root,
        str(bundle.assets["calibration"]["classic_gate_source"]["path"]),
    )
    train_feature_coverage = _summarize_active_feature_coverage(
        training_path,
        feature_columns=feature_columns,
        datasets=training_datasets,
        chunksize=200000,
    )
    calibration_feature_coverage = _summarize_active_feature_coverage(
        calibration_path,
        feature_columns=feature_columns,
        datasets=NEW_BLOCK_DATASETS,
    )
    eval_feature_coverage = {
        dataset: _summarize_active_feature_coverage(
            _resolve_local_bundle_path(bundle.root, str(path_like)),
            feature_columns=feature_columns,
            datasets=(str(dataset),),
        )[str(dataset)]
        for dataset, path_like in extra_eval_paths.items()
    }
    coverage_sections = {
        "train": train_feature_coverage,
        "calibration": calibration_feature_coverage,
        "eval": eval_feature_coverage,
    }
    coverage_failures = _feature_coverage_failures(coverage_sections)
    if coverage_failures:
        raise ValueError(f"Active classic raw feature coverage failed: {coverage_failures}")

    total_error_gate_consistency = _summarize_total_error_gate_consistency(bundle)

    self_containment_row_paths = tuple(
        _resolve_local_bundle_path(bundle.root, path_value)
        for path_value in (
            str(bundle.models["classic"]["train_path"]),
            str(bundle.assets["calibration"]["classic_gate_source"]["path"]),
            str(bundle.models["classic"]["s2and_eval_path"]),
            str(bundle.models["classic"]["hwang_eval_path"]),
            str(bundle.models["classic"]["s_park_eval_path"]),
            str(bundle.models["classic"]["s_lee_eval_path"]),
            *[str(path_like) for path_like in extra_eval_paths.values()],
        )
    )
    self_containment = _summarize_self_containing_candidate_rows(self_containment_row_paths)
    if int(self_containment["self_containing_rows"]) > 0:
        failing_files = [
            {
                "path": str(Path(file_summary["path"]).relative_to(bundle.root)).replace("/", "\\"),
                "rows": int(file_summary["self_containing_rows"]),
                "positive_rows": int(file_summary["self_containing_positive_rows"]),
                "samples": file_summary["samples"],
            }
            for file_summary in self_containment["files"]
            if int(file_summary["self_containing_rows"]) > 0
        ]
        raise ValueError(f"Self-containing candidate rows are forbidden: {failing_files[:10]}")

    hwang_candidate_level_labels = _summarize_hwang_candidate_level_label_consistency(
        hwang_rows_path=_resolve_local_bundle_path(bundle.root, str(bundle.assets["test"]["hwang_eval"]["path"])),
        clean_overrides_path=_resolve_local_bundle_path(
            bundle.root,
            str(bundle.assets["test"]["hwang_clean_overrides"]["path"]),
        ),
        manifest_path=_resolve_local_bundle_path(
            bundle.root,
            str(bundle.assets["test"]["hwang_candidate_level_label_overrides"]["path"]),
        ),
    )
    dataset_contract = _summarize_dataset_contract(bundle.root)
    reranker_bundle_contracts = _summarize_reranker_bundle_contracts(bundle.root)

    return {
        "entrypoint_kind": "validator",
        "bundle_root": str(bundle.root.relative_to(REPO_ROOT)),
        "bundle_name": bundle.bundle_name,
        "bundle_file": str((bundle.root / "bundle.json").relative_to(REPO_ROOT)),
        "documentation_files": [
            str((bundle.root / "README.md").relative_to(REPO_ROOT)),
            str((bundle.root / "PROVENANCE.md").relative_to(REPO_ROOT)),
        ],
        "required_local_files": required_files,
        "resolved_bundle_paths": resolved_entries,
        "classic_expected_metric_keys": sorted(bundle.expected_metrics["classic"]),
        "gate_calibration_groups": int(gate_split["calibration_groups"]),
        "gate_evaluation_groups": int(gate_split["evaluation_groups"]),
        "extra_eval_datasets": sorted(str(key) for key in extra_eval_paths),
        "training_datasets": list(training_datasets),
        "hardlinked_file_count": 0,
        "active_feature_coverage": coverage_sections,
        "total_error_gate_consistency": total_error_gate_consistency,
        "self_containing_candidate_rows": self_containment,
        "hwang_candidate_level_labels": hwang_candidate_level_labels,
        "dataset_contract": dataset_contract,
        "reranker_bundle_contracts": reranker_bundle_contracts,
        "note": (
            "Classic-only bundle with fully local copied assets plus canonical classic "
            "derived features materialized for every non-runtime active feature; absent active raw "
            "features, present active feature NaNs, stale total-error gate artifacts, H-Wang "
            "query-level targets that disagree with candidate-level labels, dataset-contract custom "
            "label mismatches, and non-residual self-containing candidate rows are all fatal validation "
            "failures. Generated residual rows are allowed only when query_in_seed_before_holdout and "
            "cluster_size prove holdout materialization; manual required calibration/eval positives are allowed "
            "only when positive_candidate_keys marks the row as required; residual public LOO positive rows are "
            "allowed when source=labeled_loo and split is train_loo/eval_loo or source=s2and_rescue_manual_review."
        ),
    }


def main() -> None:
    print(json.dumps(build_validation_payload(), indent=2))


if __name__ == "__main__":
    main()
