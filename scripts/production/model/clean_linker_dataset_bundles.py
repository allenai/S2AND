"""Clean linker dataset bundles after manual label repair.

This migration removes weak ``unlabeled_singleton_orcid`` rows from the
canonical label parquet files and rebuilds query-level split metadata from the
surviving active labels. It is intentionally bundle-level: downstream training
should validate clean data instead of depending on hidden runtime filtering.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from s2and.incremental_linking_training.classic import (
    CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET,
    UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE,
    _active_stratified_label_metadata,
    _drop_shadowed_calibration_source_rows,
    _refresh_stratified_metadata_from_active_labels,
    _validate_unique_stratified_candidate_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLE_ROOTS = (REPO_ROOT / "s2and" / "data" / "s2and_and_big_blocks_linker_dataset_20260525",)
DEFAULT_REPORT_PATH = REPO_ROOT / "scratch" / "clean_linker_dataset_bundles_report.json"


@dataclass(frozen=True)
class StratifiedSourceSpec:
    """A featureless label table participating in the promoted stratified split."""

    table_key: str
    source_key: str
    source_kind: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temp_path, index=False)
    temp_path.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _bundle_payload(bundle_root: Path) -> dict[str, Any]:
    return _read_json(bundle_root / "bundle.json")


def _bundle_child_path(bundle_root: Path, relative_path: str, *, context: str) -> Path:
    raw_path = Path(str(relative_path))
    if raw_path.is_absolute():
        raise ValueError(f"{context} must be relative to bundle root, got absolute path: {relative_path!r}")
    root = bundle_root.resolve()
    resolved = (root / raw_path).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{context} escapes bundle root: {relative_path!r}")
    return resolved


def _featureless_files(payload: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in payload["assets"]["featureless_rows"]["files"].items()}


def _without_unlabeled_singleton_orcid(rows: pd.DataFrame) -> pd.DataFrame:
    if "supervision_type" not in rows.columns:
        return rows
    return rows.loc[~rows["supervision_type"].astype(str).eq(UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE)].reset_index(
        drop=True
    )


def _classic_spec(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload["models"]["classic"])


def _stratified_source_specs(payload: dict[str, Any]) -> list[StratifiedSourceSpec]:
    files = _featureless_files(payload)
    sources: list[StratifiedSourceSpec] = []
    if "classic_gate_source_path" in files:
        sources.append(
            StratifiedSourceSpec(
                table_key="classic_gate_source_path",
                source_key="calibration_source",
                source_kind="calibration_source",
            )
        )
    for table_key, source_key in (
        ("s2and_eval_path", "s2and_eval"),
        ("hwang_eval_path", "hwang_eval"),
        ("s_park_eval_path", "s_park_eval"),
        ("s_lee_eval_path", "s_lee_eval"),
    ):
        if table_key in files:
            sources.append(StratifiedSourceSpec(table_key=table_key, source_key=source_key, source_kind="public_test"))
    for table_key in files:
        if not table_key.startswith("extra_eval_paths."):
            continue
        normalized = table_key.split(".", 1)[1]
        sources.append(
            StratifiedSourceSpec(
                table_key=f"extra_eval_paths.{normalized}",
                source_key=f"{normalized}_eval",
                source_kind="public_test",
            )
        )
    return sources


def _drop_unlabeled_singleton_orcid_labels(
    *,
    bundle_root: Path,
    payload: dict[str, Any],
    write: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for table_key, relative_path in _featureless_files(payload).items():
        path = _bundle_child_path(bundle_root, relative_path, context=f"featureless_rows.files[{table_key!r}]")
        rows = pd.read_parquet(path)
        if "supervision_type" not in rows.columns:
            reports.append(
                {
                    "table_key": table_key,
                    "path": relative_path,
                    "rows_before": int(len(rows)),
                    "rows_after": int(len(rows)),
                    "rows_removed": 0,
                    "positive_rows_removed": 0,
                    "negative_rows_removed": 0,
                    "queries_removed": 0,
                }
            )
            continue
        bad_mask = rows["supervision_type"].astype(str).eq(UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE)
        removed = rows.loc[bad_mask]
        cleaned = rows.loc[~bad_mask].reset_index(drop=True)
        labels = (
            pd.to_numeric(removed["label"], errors="coerce").fillna(0).astype(int)
            if "label" in removed
            else pd.Series(dtype=int)
        )
        reports.append(
            {
                "table_key": table_key,
                "path": relative_path,
                "rows_before": int(len(rows)),
                "rows_after": int(len(cleaned)),
                "rows_removed": int(bad_mask.sum()),
                "positive_rows_removed": int(labels.sum()) if len(removed) else 0,
                "negative_rows_removed": int(len(removed) - int(labels.sum())) if len(removed) else 0,
                "queries_removed": int(
                    len(set(rows["query_group_id"].astype(str)) - set(cleaned["query_group_id"].astype(str)))
                )
                if "query_group_id" in rows.columns
                else 0,
            }
        )
        if write and bad_mask.any():
            _write_parquet_atomic(cleaned, path)
    return reports


def _active_stratified_rows(bundle_root: Path, payload: dict[str, Any]) -> pd.DataFrame:
    files = _featureless_files(payload)
    frames: list[pd.DataFrame] = []
    for source in _stratified_source_specs(payload):
        rows = pd.read_parquet(
            _bundle_child_path(
                bundle_root,
                files[source.table_key],
                context=f"featureless_rows.files[{source.table_key!r}]",
            )
        )
        rows = _without_unlabeled_singleton_orcid(rows)
        if source.source_key == "calibration_source":
            rows["source_key"] = (
                rows["dataset"].astype(str).map(CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET).fillna("s2and_eval")
            )
        else:
            rows["source_key"] = source.source_key
        rows["source_kind"] = source.source_kind
        frames.append(rows)
    if not frames:
        raise ValueError(f"No stratified source label tables found in {bundle_root}")
    return _drop_shadowed_calibration_source_rows(pd.concat(frames, ignore_index=True))


def _refresh_split_assignments(
    *,
    bundle_root: Path,
    payload: dict[str, Any],
    write: bool,
) -> dict[str, Any]:
    split_assets = dict(payload["assets"]["splits"])
    assignments_relative = str(split_assets["assignments_path"])
    assignments_path = _bundle_child_path(bundle_root, assignments_relative, context="assets.splits.assignments_path")
    assignments = pd.read_csv(assignments_path, low_memory=False)
    original_columns = list(assignments.columns)
    rows = _active_stratified_rows(bundle_root, payload)
    active_keys = rows[["query_group_id", "source_key"]].drop_duplicates()
    filtered_assignments = assignments.merge(active_keys, on=["query_group_id", "source_key"], how="inner")
    filtered_keys = filtered_assignments[["query_group_id", "source_key"]].drop_duplicates()
    active_rows = rows.merge(filtered_keys, on=["query_group_id", "source_key"], how="inner")
    _validate_unique_stratified_candidate_rows(active_rows)
    _refreshed_rows, refreshed_assignments = _refresh_stratified_metadata_from_active_labels(
        active_rows,
        filtered_assignments,
    )
    ordered_columns = [column for column in original_columns if column in refreshed_assignments.columns]
    ordered_columns.extend(column for column in refreshed_assignments.columns if column not in ordered_columns)
    refreshed_assignments = refreshed_assignments[ordered_columns]
    if write:
        _write_csv_atomic(refreshed_assignments, assignments_path)

    summary_relative = str(split_assets.get("summary_path", "splits/summary.json"))
    summary_path = _bundle_child_path(bundle_root, summary_relative, context="assets.splits.summary_path")
    summary = _read_json(summary_path) if summary_path.exists() else {}
    removed_assignment_rows = int(len(assignments) - len(refreshed_assignments))
    summary["assignment_rows"] = int(len(refreshed_assignments))
    summary["split_counts"] = {
        str(split): int(count) for split, count in refreshed_assignments["split"].value_counts().sort_index().items()
    }
    if removed_assignment_rows:
        summary["removed_unlabeled_singleton_orcid_assignments"] = {
            "removed_assignment_rows": removed_assignment_rows,
            "source": f"drop_supervision_type:{UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE}",
        }
    else:
        summary.pop("removed_unlabeled_singleton_orcid_assignments", None)
    if write:
        _write_json(summary_path, summary)

    missing_by_source = (
        assignments.merge(active_keys, on=["query_group_id", "source_key"], how="left", indicator=True)
        .query("_merge == 'left_only'")["source_key"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    metadata = _active_stratified_label_metadata(active_rows)
    return {
        "assignments_path": assignments_relative,
        "assignment_rows_before": int(len(assignments)),
        "assignment_rows_after": int(len(refreshed_assignments)),
        "assignment_rows_removed": removed_assignment_rows,
        "missing_before_by_source_key": {str(key): int(value) for key, value in missing_by_source.items()},
        "split_counts": dict(summary["split_counts"]),
        "active_query_source_keys": int(len(filtered_keys)),
        "active_metadata_rows": int(len(metadata)),
    }


def _refresh_base_group_file(
    *,
    bundle_root: Path,
    relative_path: str,
    active_base_groups: set[str],
    write: bool,
) -> dict[str, Any]:
    path = _bundle_child_path(bundle_root, relative_path, context="classic base-group path")
    rows = pd.read_csv(path, low_memory=False)
    if "base_group_id" not in rows.columns:
        raise ValueError(f"{path} is missing required base_group_id column")
    filtered = rows[rows["base_group_id"].astype(str).isin(active_base_groups)].reset_index(drop=True)
    if write and len(filtered) != len(rows):
        _write_csv_atomic(filtered, path)
    return {
        "path": relative_path,
        "rows_before": int(len(rows)),
        "rows_after": int(len(filtered)),
        "rows_removed": int(len(rows) - len(filtered)),
    }


def _refresh_gate_base_groups(
    *,
    bundle_root: Path,
    payload: dict[str, Any],
    write: bool,
) -> list[dict[str, Any]]:
    spec = _classic_spec(payload)
    files = _featureless_files(payload)
    gate_key = "classic_gate_source_path"
    if gate_key not in files:
        return []
    gate_rows = pd.read_parquet(
        _bundle_child_path(bundle_root, files[gate_key], context=f"featureless_rows.files[{gate_key!r}]")
    )
    gate_rows = _without_unlabeled_singleton_orcid(gate_rows)
    active_base_groups = {str(value) for value in gate_rows["base_group_id"].dropna()}
    reports: list[dict[str, Any]] = []
    for spec_key in ("classic_gate_calibration_base_groups_path", "classic_gate_internal_eval_base_groups_path"):
        relative_path = spec.get(spec_key)
        if relative_path is None:
            continue
        reports.append(
            {
                "spec_key": spec_key,
                **_refresh_base_group_file(
                    bundle_root=bundle_root,
                    relative_path=str(relative_path),
                    active_base_groups=active_base_groups,
                    write=write,
                ),
            }
        )
    return reports


def _verify_bundle(bundle_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    bad_label_counts: dict[str, int] = {}
    for table_key, relative_path in _featureless_files(payload).items():
        path = _bundle_child_path(bundle_root, relative_path, context=f"featureless_rows.files[{table_key!r}]")
        columns = pd.read_parquet(path, columns=["supervision_type"]) if path.exists() else pd.DataFrame()
        if "supervision_type" not in columns.columns:
            continue
        bad_count = int(columns["supervision_type"].astype(str).eq(UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE).sum())
        if bad_count:
            bad_label_counts[table_key] = bad_count

    split_report = _refresh_split_assignments(bundle_root=bundle_root, payload=payload, write=False)
    base_group_reports = _refresh_gate_base_groups(bundle_root=bundle_root, payload=payload, write=False)
    errors: list[str] = []
    if bad_label_counts:
        errors.append(f"unlabeled_singleton_orcid rows remain: {bad_label_counts}")
    if split_report["assignment_rows_removed"]:
        errors.append(
            "split assignments do not match active source rows: "
            f"missing={split_report['assignment_rows_removed']}, "
            f"by_source={split_report['missing_before_by_source_key']}"
        )
    stale_base_groups = [report for report in base_group_reports if report["rows_removed"]]
    if stale_base_groups:
        errors.append(f"base-group split files contain inactive groups: {stale_base_groups}")
    return {
        "bad_label_counts": bad_label_counts,
        "split_verification": split_report,
        "base_group_verification": base_group_reports,
        "ok": not errors,
        "errors": errors,
    }


def migrate_bundle(bundle_root: Path, *, write: bool) -> dict[str, Any]:
    """Clean one bundle and return a reproducible report."""

    bundle_root = bundle_root.resolve()
    if not (bundle_root / "bundle.json").exists():
        raise FileNotFoundError(f"Missing bundle.json under {bundle_root}")
    payload = _bundle_payload(bundle_root)
    label_report = _drop_unlabeled_singleton_orcid_labels(
        bundle_root=bundle_root,
        payload=payload,
        write=write,
    )
    split_report = _refresh_split_assignments(bundle_root=bundle_root, payload=payload, write=write)
    base_group_report = _refresh_gate_base_groups(bundle_root=bundle_root, payload=payload, write=write)
    verification = _verify_bundle(bundle_root, _bundle_payload(bundle_root)) if write else {"ok": None, "dry_run": True}
    if write and verification["errors"]:
        raise RuntimeError(f"Bundle verification failed for {bundle_root}: {verification['errors']}")
    return {
        "bundle_root": str(bundle_root),
        "mode": "write" if write else "dry-run",
        "label_tables": label_report,
        "split_assignments": split_report,
        "base_group_files": base_group_report,
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_roots",
        nargs="*",
        type=Path,
        default=list(DEFAULT_BUNDLE_ROOTS),
        help="Bundle roots to clean. Defaults to the canonical linker bundle present in this checkout.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing files.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reports = [migrate_bundle(root, write=not args.dry_run) for root in args.bundle_roots]
    report = {
        "policy": f"drop_supervision_type:{UNLABELED_SINGLETON_ORCID_SUPERVISION_TYPE}",
        "bundles": reports,
    }
    if not args.dry_run:
        _write_json(args.report_path.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
