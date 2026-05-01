"""Sync bundle metadata from on-disk assets plus one replay summary."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from joint_safe_link_official_stack import compare_to_expected, expected_metrics_from_summary


def _read_csv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, compression=compression, low_memory=False)


def _resolve_bundle_path(bundle_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else bundle_root / path


def _row_counts(df: pd.DataFrame, *, include_query_polarity: bool = False) -> dict[str, int]:
    counts = {
        "rows": int(len(df)),
        "queries": int(df["query_group_id"].astype(str).nunique()),
    }
    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="coerce").fillna(0)
        counts["positive_rows"] = int(labels.sum())
        if include_query_polarity:
            query_labels = labels.groupby(df["query_group_id"].astype(str)).max()
            counts["positive_queries"] = int((query_labels > 0).sum())
            counts["negative_queries"] = int((query_labels <= 0).sum())
    return counts


def _query_polarity_counts(df: pd.DataFrame) -> tuple[int, int]:
    if df.empty:
        return 0, 0
    labels = (
        df.assign(label_numeric=pd.to_numeric(df["label"], errors="coerce").fillna(0))
        .groupby("query_group_id", sort=False)["label_numeric"]
        .max()
    )
    return int((labels > 0).sum()), int((labels <= 0).sum())


def _sync_path_backed_row_asset(bundle_root: Path, asset: dict[str, Any]) -> dict[str, int]:
    include_query_polarity = "positive_queries" in asset or "negative_queries" in asset
    row_counts = _row_counts(
        _read_csv(_resolve_bundle_path(bundle_root, str(asset["path"]))),
        include_query_polarity=include_query_polarity,
    )
    asset.update(row_counts)
    return row_counts


def _sync_test_assets(payload: dict[str, Any], *, bundle_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    test_assets = payload["assets"]["test"]
    synced: dict[str, Any] = {}

    eval_asset_keys = {
        "s2and_eval": spec.get("s2and_eval_path"),
        "hwang_eval": spec.get("hwang_eval_path"),
        "s_park_eval": spec.get("s_park_eval_path"),
        "s_lee_eval": spec.get("s_lee_eval_path"),
    }
    for asset_key, path_like in eval_asset_keys.items():
        if path_like is None or asset_key not in test_assets:
            continue
        synced[asset_key] = _sync_path_backed_row_asset(bundle_root, test_assets[asset_key])

    extra_eval_paths = dict(spec.get("extra_eval_paths") or {})
    for dataset_name, _path_like in extra_eval_paths.items():
        asset_key = f"{dataset_name}_eval"
        if asset_key not in test_assets:
            continue
        synced[asset_key] = _sync_path_backed_row_asset(bundle_root, test_assets[asset_key])

    if "hwang_clean_overrides" in test_assets:
        override_asset = test_assets["hwang_clean_overrides"]
        override_df = _read_csv(_resolve_bundle_path(bundle_root, str(override_asset["path"])))
        override_counts = {
            "queries": int(override_df["query_group_id"].astype(str).nunique()),
            "positive_overrides": int(
                pd.to_numeric(override_df["manual_safe_target"], errors="coerce").fillna(0).sum()
            ),
        }
        override_asset.update(override_counts)
        synced["hwang_clean_overrides"] = override_counts
    if "hwang_candidate_level_label_overrides" in test_assets:
        manifest_asset = test_assets["hwang_candidate_level_label_overrides"]
        summary_path = manifest_asset.get("summary_path")
        if summary_path:
            manifest_summary = json.loads(
                _resolve_bundle_path(bundle_root, str(summary_path)).read_text(encoding="utf-8")
            )
            manifest_counts = {
                "queries": int(manifest_summary["queries"]),
                "positive_queries_after_candidate_relabel": int(
                    manifest_summary["positive_queries_after_candidate_relabel"]
                ),
                "reviewed_positive_corrections": int(manifest_summary["reviewed_positive_corrections"]),
                "reviewed_positive_corrections_survived": int(
                    manifest_summary["reviewed_positive_corrections_survived"]
                ),
            }
            manifest_asset.update(manifest_counts)
            synced["hwang_candidate_level_label_overrides"] = manifest_counts
    return synced


def _sync_new_block_manual_split(
    payload: dict[str, Any],
    *,
    gate_source_df: pd.DataFrame,
    calibration_groups: set[str],
    evaluation_groups: set[str],
) -> dict[str, Any]:
    split_payload = payload.get("new_block_manual_calibration_split", {})
    per_dataset = split_payload.get("per_dataset")
    if not isinstance(per_dataset, dict):
        return {}

    synced: dict[str, Any] = {}
    gate_source_df = gate_source_df.copy()
    gate_source_df["dataset"] = gate_source_df["dataset"].astype(str)
    gate_source_df["base_group_id"] = gate_source_df["base_group_id"].astype(str)
    for dataset_name, dataset_payload in per_dataset.items():
        dataset_rows = gate_source_df[gate_source_df["dataset"] == str(dataset_name)].copy()
        dataset_base_groups = set(dataset_rows["base_group_id"].astype(str))
        calibration_base_groups = dataset_base_groups & calibration_groups
        evaluation_base_groups = dataset_base_groups & evaluation_groups
        calibration_rows = dataset_rows[dataset_rows["base_group_id"].astype(str).isin(calibration_base_groups)].copy()
        evaluation_rows = dataset_rows[dataset_rows["base_group_id"].astype(str).isin(evaluation_base_groups)].copy()
        total_positive_queries, total_negative_queries = _query_polarity_counts(dataset_rows)
        calibration_positive_queries, calibration_negative_queries = _query_polarity_counts(calibration_rows)
        evaluation_positive_queries, evaluation_negative_queries = _query_polarity_counts(evaluation_rows)
        refreshed = {
            "total_base_groups": int(len(dataset_base_groups)),
            "calibration_base_groups": int(len(calibration_base_groups)),
            "evaluation_base_groups": int(len(evaluation_base_groups)),
            "total_positive_queries": total_positive_queries,
            "total_negative_queries": total_negative_queries,
            "calibration_positive_queries": calibration_positive_queries,
            "calibration_negative_queries": calibration_negative_queries,
            "evaluation_positive_queries": evaluation_positive_queries,
            "evaluation_negative_queries": evaluation_negative_queries,
        }
        dataset_payload.update(refreshed)
        synced[str(dataset_name)] = refreshed
    return synced


def _sync_stratified_eval_test_split(payload: dict[str, Any], *, bundle_root: Path) -> dict[str, Any]:
    """Refresh promoted stratified split counts from the assignment artifact."""

    split_asset = payload.get("assets", {}).get("calibration", {}).get("stratified_eval_test_split")
    if not isinstance(split_asset, dict):
        return {}
    assignments_path = split_asset.get("assignments_path")
    if not assignments_path:
        return {}
    assignments = _read_csv(_resolve_bundle_path(bundle_root, str(assignments_path)))
    split_counts = assignments["split"].value_counts().to_dict()
    refreshed = {
        "queries": int(len(assignments)),
        "calibration_fit_queries": int(split_counts.get("calibration_fit", 0)),
        "calibration_check_queries": int(split_counts.get("calibration_check", 0)),
        "test_queries": int(split_counts.get("test", 0)),
    }
    if "stratum_key" in assignments.columns:
        refreshed["observed_strata"] = int(assignments["stratum_key"].nunique())
    stratum_balance_path = split_asset.get("stratum_balance_path")
    if stratum_balance_path:
        stratum_balance = _read_csv(_resolve_bundle_path(bundle_root, str(stratum_balance_path)))
        refreshed["strata_too_small_for_all_splits"] = int((stratum_balance["total"] < 3).sum())
        refreshed["strata_with_missing_split"] = int((stratum_balance["missing_split_count"] > 0).sum())
    split_asset.update(refreshed)
    return refreshed


def _format_threshold_lines(thresholds: dict[str, Any]) -> list[str]:
    """Return Markdown bullets for one threshold mapping."""

    return [f"- `{key}`: `{float(value):.12g}`" for key, value in sorted(thresholds.items())]


def _metric_summary_line(label: str, metrics: dict[str, Any] | None) -> str:
    """Return one compact report line for a split metric payload."""

    if not metrics:
        return f"- {label}: unavailable"
    return (
        f"- {label} errors: `{int(metrics['errors'])}` / `{int(metrics['n_queries'])}`; "
        f"BA: `{float(metrics['balanced_accuracy']):.6f}`"
    )


def _write_total_error_gate_artifacts_from_summary(
    payload: dict[str, Any],
    *,
    bundle_root: Path,
    summary: dict[str, Any],
) -> bool:
    """Write promoted gate artifacts from the current replay summary, when present."""

    gate_asset = payload.get("assets", {}).get("calibration", {}).get("total_error_4score_2margin_gate")
    if not isinstance(gate_asset, dict):
        return False
    promoted_gate = summary.get("abstain_rule", {}).get("promoted_stratified_gate")
    if not isinstance(promoted_gate, dict) or "selected_gate" not in promoted_gate:
        return False

    selected_gate = dict(promoted_gate["selected_gate"])
    selected_gate_path = gate_asset.get("selected_gate_path")
    if selected_gate_path:
        path = _resolve_bundle_path(bundle_root, str(selected_gate_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selected_gate, indent=2) + "\n", encoding="utf-8")

    candidate_metrics = list(promoted_gate.get("candidate_metrics") or [])
    candidate_metrics_path = gate_asset.get("candidate_metrics_path")
    if candidate_metrics_path:
        path = _resolve_bundle_path(bundle_root, str(candidate_metrics_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(candidate_metrics).to_csv(path, index=False)

    stratified_overall = summary.get("stratified_eval_test_split", {}).get("overall", {})
    artifact_summary = {
        "selected_gate": selected_gate,
        "selection_key": promoted_gate.get("selection_key", {}),
        "fit_metrics": promoted_gate.get("fit_metrics") or stratified_overall.get("calibration_fit", {}),
        "check_metrics": promoted_gate.get("check_metrics") or stratified_overall.get("calibration_check", {}),
        "test_metrics": stratified_overall.get("test", {}),
        "candidate_count": len(candidate_metrics),
    }
    summary_path = gate_asset.get("summary_path")
    if summary_path:
        path = _resolve_bundle_path(bundle_root, str(summary_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact_summary, indent=2) + "\n", encoding="utf-8")

    report_path = gate_asset.get("report_path")
    if report_path:
        score_lines = _format_threshold_lines(dict(selected_gate["score_thresholds"]))
        margin_lines = _format_threshold_lines(dict(selected_gate["margin_thresholds"]))
        report_lines = [
            "# Total-Error Four-Score / Two-Margin Gate",
            "",
            f"Selected gate: `{selected_gate['name']}`",
            f"Lambda penalty: `{float(selected_gate['lambda_penalty'])}`",
            "",
            "Score thresholds:",
            *score_lines,
            "",
            "Margin thresholds:",
            *margin_lines,
            "",
            "Selection metrics:",
            _metric_summary_line("fit", artifact_summary["fit_metrics"]),
            _metric_summary_line("check", artifact_summary["check_metrics"]),
            _metric_summary_line("test", artifact_summary["test_metrics"]),
            "",
            "Artifacts:",
            "- `selected_gate.json`",
            "- `gate_candidate_metrics.csv`",
            "- `summary.json`",
        ]
        path = _resolve_bundle_path(bundle_root, str(report_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return True


def _sync_total_error_gate_asset(payload: dict[str, Any], *, bundle_root: Path) -> dict[str, Any]:
    """Refresh promoted total-error gate metadata from the selected-gate artifact."""

    gate_asset = payload.get("assets", {}).get("calibration", {}).get("total_error_4score_2margin_gate")
    if not isinstance(gate_asset, dict):
        return {}
    selected_gate_path = gate_asset.get("selected_gate_path")
    if not selected_gate_path:
        return {}
    selected_gate = json.loads(_resolve_bundle_path(bundle_root, str(selected_gate_path)).read_text(encoding="utf-8"))
    refreshed = {
        "selected_gate_name": str(selected_gate["name"]),
        "lambda_penalty": float(selected_gate["lambda_penalty"]),
        "score_thresholds": dict(selected_gate["score_thresholds"]),
        "margin_thresholds": dict(selected_gate["margin_thresholds"]),
    }
    gate_asset.update(refreshed)
    return refreshed


def _sync_dataset_contract_asset(payload: dict[str, Any], *, bundle_root: Path) -> dict[str, Any]:
    """Refresh dataset-contract metadata from the compiled summary artifact."""

    asset = payload.get("assets", {}).get("dataset_contract")
    if not isinstance(asset, dict):
        return {}
    summary_path = asset.get("custom_label_ledger_summary_path")
    if not summary_path:
        return {}
    summary = json.loads(_resolve_bundle_path(bundle_root, str(summary_path)).read_text(encoding="utf-8"))
    refreshed = {
        "custom_label_ledger_rows": int(summary["ledger_rows"]),
        "comparison_fatal_mismatch_count": int(summary["comparison_fatal_mismatch_count"]),
        "label_slice_counts": dict(summary["slice_counts"]),
    }
    asset.update(refreshed)
    return refreshed


def _write_verification_json(
    *,
    summary: dict[str, Any],
    expected_metrics: dict[str, float],
    verification_json_path: Path,
) -> dict[str, Any]:
    """Write a verification payload aligned to one replay summary."""

    verification = {
        "summary": summary,
        "expected": expected_metrics,
        "deltas": compare_to_expected(summary, expected_metrics),
    }
    verification_json_path.parent.mkdir(parents=True, exist_ok=True)
    verification_json_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    return verification


def sync_bundle_metadata(
    bundle_root: Path,
    summary: dict[str, Any],
    *,
    created_on: str | None = None,
    verification_json_path: Path | None = None,
) -> dict[str, Any]:
    """Refresh one bundle's counts and frozen metrics from disk plus replay summary."""

    bundle_root = bundle_root.resolve()
    bundle_json_path = bundle_root / "bundle.json"
    payload = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    spec = dict(payload["models"]["classic"])

    for training_asset in payload["assets"].get("training", {}).values():
        if isinstance(training_asset, dict) and "path" in training_asset:
            _sync_path_backed_row_asset(bundle_root, training_asset)

    calibration_asset = payload["assets"]["calibration"]["classic_gate_source"]
    gate_source_counts = _sync_path_backed_row_asset(bundle_root, calibration_asset)
    gate_source_df = _read_csv(_resolve_bundle_path(bundle_root, str(calibration_asset["path"])))

    calibration_groups_df = _read_csv(
        _resolve_bundle_path(bundle_root, str(spec["classic_gate_calibration_base_groups_path"]))
    )
    internal_groups_df = _read_csv(
        _resolve_bundle_path(bundle_root, str(spec["classic_gate_internal_eval_base_groups_path"]))
    )
    calibration_groups = set(calibration_groups_df["base_group_id"].astype(str))
    evaluation_groups = set(internal_groups_df["base_group_id"].astype(str))
    payload["assets"]["calibration"]["classic_gate_split"]["calibration_groups"] = int(len(calibration_groups))
    payload["assets"]["calibration"]["classic_gate_split"]["evaluation_groups"] = int(len(evaluation_groups))

    test_asset_counts = _sync_test_assets(payload, bundle_root=bundle_root, spec=spec)
    new_block_split_counts = _sync_new_block_manual_split(
        payload,
        gate_source_df=gate_source_df,
        calibration_groups=calibration_groups,
        evaluation_groups=evaluation_groups,
    )
    stratified_split_counts = _sync_stratified_eval_test_split(payload, bundle_root=bundle_root)
    _write_total_error_gate_artifacts_from_summary(payload, bundle_root=bundle_root, summary=summary)
    total_error_gate = _sync_total_error_gate_asset(payload, bundle_root=bundle_root)
    dataset_contract = _sync_dataset_contract_asset(payload, bundle_root=bundle_root)

    refreshed_expected_metrics = expected_metrics_from_summary(summary)
    payload.setdefault("expected_metrics", {})["classic"] = refreshed_expected_metrics
    payload["created_on"] = created_on or date.today().isoformat()
    bundle_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    refreshed_verification_path: str | None = None
    if verification_json_path is not None:
        verification_json_path = verification_json_path.resolve()
        _write_verification_json(
            summary=summary,
            expected_metrics=refreshed_expected_metrics,
            verification_json_path=verification_json_path,
        )
        refreshed_verification_path = str(verification_json_path)

    return {
        "bundle_root": str(bundle_root),
        "created_on": str(payload["created_on"]),
        "training_assets": payload["assets"].get("training", {}),
        "classic_gate_source": gate_source_counts,
        "classic_gate_split": {
            "calibration_groups": int(len(calibration_groups)),
            "evaluation_groups": int(len(evaluation_groups)),
        },
        "test_assets": test_asset_counts,
        "new_block_manual_calibration_split": new_block_split_counts,
        "stratified_eval_test_split": stratified_split_counts,
        "total_error_4score_2margin_gate": total_error_gate,
        "dataset_contract": dataset_contract,
        "expected_metrics": refreshed_expected_metrics,
        "verification_json_path": refreshed_verification_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--verification-json", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    verification_json_path = (
        args.verification_json.resolve()
        if args.verification_json is not None
        else args.summary_json.resolve().with_name("verification.json")
    )
    sync_summary = sync_bundle_metadata(
        args.bundle_root,
        summary,
        verification_json_path=verification_json_path,
    )
    print(json.dumps(sync_summary, indent=2))


if __name__ == "__main__":
    main()
