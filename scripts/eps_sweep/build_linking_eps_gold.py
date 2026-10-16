"""Build development-only pair constraints for giant-block EPS diagnostics."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from s2and.incremental_linking_training.classic import (
    _classic_stratified_eval_source_specs,
    _load_classic_stratified_eval_rows,
    load_bundle,
)
from scripts.production.model.train_linker_and_finalize import _load_initial_query_sources

LABEL_COLUMNS = [
    "row_index",
    "dataset",
    "query_group_id",
    "base_group_id",
    "query_signature_id",
    "query_view",
    "candidate_component_key",
    "candidate_cluster_id",
    "label",
    "retrieval_rank",
    "split",
    "supervision_type",
    "source_key",
    "source_kind",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit source and output paths."""
    parser = argparse.ArgumentParser(
        description="Expand linker candidate labels into query-member pair constraints for EPS clustering sweeps."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--task-splits",
        nargs="+",
        choices=["calibration_fit", "calibration_check"],
        default=["calibration_fit", "calibration_check"],
        help="Use declared development assignments; test queries remain excluded.",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Bound each dataset for a smoke run.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pair_left_right(query_ids: pd.Series, member_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    query = query_ids.astype(str)
    member = member_ids.astype(str)
    left = query.where(query <= member, member)
    right = member.where(query <= member, query)
    return left, right


def _load_label_rows(bundle_root: Path, datasets: set[str], task_splits: Sequence[str]) -> pd.DataFrame:
    """Use active labels and final lifecycle assignments instead of raw split tags."""
    bundle = load_bundle(bundle_root)
    spec = dict(bundle.models["classic"])
    rows, _assignments = _load_classic_stratified_eval_rows(
        bundle,
        spec,
        dict(spec["stratified_eval_test_split"]),
        splits=task_splits,
        include_calibration_source=True,
    )
    labels = rows.loc[rows["dataset"].astype(str).isin(datasets), LABEL_COLUMNS].copy()
    if labels.empty:
        raise ValueError(f"No development labels for datasets={sorted(datasets)}")
    source_names = {
        str(source["source_key"]): Path(source["path"]).name for source in _classic_stratified_eval_source_specs(spec)
    }
    labels["table_name"] = labels["source_key"].map(source_names)
    if labels["table_name"].isna().any():
        raise ValueError("Development labels reference an undeclared source table")
    labels["label"] = pd.to_numeric(labels["label"], errors="raise")
    if not labels["label"].isin([0, 1]).all():
        raise ValueError("Development labels must contain binary safe-action targets")
    labels["label"] = labels["label"].astype("int8")
    for column in [
        "dataset",
        "query_group_id",
        "query_signature_id",
        "query_view",
        "candidate_component_key",
        "split",
        "supervision_type",
        "source_key",
        "source_kind",
        "table_name",
    ]:
        if column in labels.columns:
            labels[column] = labels[column].astype(str)
    labels.attrs["task_splits"] = list(task_splits)
    return labels


def _load_members(bundle_root: Path, dataset: str) -> pd.DataFrame:
    path = bundle_root / "components" / f"{dataset}_members.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    members = pd.read_parquet(path, columns=["dataset", "candidate_component_key", "member_index", "signature_id"])
    members = members[members["dataset"].astype(str) == dataset].copy()
    members["candidate_component_key"] = members["candidate_component_key"].astype(str)
    members["member_signature_id"] = members["signature_id"].astype(str)
    return members.drop(columns=["signature_id"])


def _drop_conflicts(expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflict_keys = [
        "dataset",
        "table_name",
        "split",
        "query_signature_id",
        "member_signature_id",
    ]
    directed_nunique = expanded.groupby(conflict_keys, sort=False)["label"].nunique()
    directed_conflicts = directed_nunique[directed_nunique > 1].reset_index()[conflict_keys]
    if not directed_conflicts.empty:
        expanded = expanded.merge(
            directed_conflicts.assign(_directed_conflict=True),
            on=conflict_keys,
            how="left",
        )
    else:
        expanded["_directed_conflict"] = False

    undirected_keys = ["dataset", "table_name", "split", "pair_left", "pair_right"]
    undirected_nunique = expanded.groupby(undirected_keys, sort=False)["label"].nunique()
    undirected_conflicts = undirected_nunique[undirected_nunique > 1].reset_index()[undirected_keys]
    if not undirected_conflicts.empty:
        expanded = expanded.merge(
            undirected_conflicts.assign(_undirected_conflict=True),
            on=undirected_keys,
            how="left",
        )
    else:
        expanded["_undirected_conflict"] = False

    expanded["_directed_conflict"] = expanded["_directed_conflict"].fillna(False).astype(bool)
    expanded["_undirected_conflict"] = expanded["_undirected_conflict"].fillna(False).astype(bool)
    conflicts = expanded[expanded["_directed_conflict"] | expanded["_undirected_conflict"]].copy()
    clean = expanded[~(expanded["_directed_conflict"] | expanded["_undirected_conflict"])].copy()
    clean = clean.drop(columns=["_directed_conflict", "_undirected_conflict"])
    return clean, conflicts


def _add_weights(rows: pd.DataFrame) -> pd.DataFrame:
    query_keys = ["dataset", "table_name", "split", "query_group_id", "query_signature_id"]
    label_keys = [*query_keys, "label"]
    rows["_query_pair_count"] = rows.groupby(query_keys, sort=False)["label"].transform("size").astype("int64")
    rows["_query_label_pair_count"] = rows.groupby(label_keys, sort=False)["label"].transform("size").astype("int64")
    rows["_query_label_class_count"] = (
        rows.groupby(query_keys, sort=False)["label"].transform("nunique").astype("int64")
    )
    rows["weight_pair"] = 1.0
    rows["weight_query_balanced"] = 1.0 / rows["_query_pair_count"].astype(float)
    rows["weight_query_label_balanced"] = 1.0 / rows["_query_label_pair_count"].astype(float)
    rows["weight_query_class_balanced"] = (1.0 / rows["_query_label_class_count"].astype(float)) / rows[
        "_query_label_pair_count"
    ].astype(float)
    return rows.drop(columns=["_query_pair_count", "_query_label_pair_count", "_query_label_class_count"])


def _build_dataset_constraints(
    bundle_root: Path,
    dataset: str,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_labels = labels[labels["dataset"].astype(str) == dataset].copy()
    if dataset_labels.empty:
        raise ValueError(f"No labels for dataset={dataset}")
    members = _load_members(bundle_root, dataset)
    component_sizes = members.groupby("candidate_component_key", sort=False)["member_signature_id"].size()
    dataset_labels["component_size"] = (
        dataset_labels["candidate_component_key"].map(component_sizes).fillna(0).astype("int64")
    )
    missing_components = int((dataset_labels["component_size"] == 0).sum())
    dataset_labels = dataset_labels[dataset_labels["component_size"] > 0].copy()

    expanded = dataset_labels.merge(
        members,
        on=["dataset", "candidate_component_key"],
        how="inner",
        validate="many_to_many",
    )
    expanded = expanded[
        expanded["query_signature_id"].astype(str) != expanded["member_signature_id"].astype(str)
    ].copy()
    origins = _load_initial_query_sources(bundle_root).get(dataset, {})
    origin_matches = expanded["query_signature_id"].map(origins).eq(expanded["member_signature_id"])
    excluded_origin_rows = int(origin_matches.sum())
    expanded = expanded.loc[~origin_matches].copy()
    expanded["pair_left"], expanded["pair_right"] = _pair_left_right(
        expanded["query_signature_id"],
        expanded["member_signature_id"],
    )

    dedupe_keys = [
        "dataset",
        "table_name",
        "split",
        "source_key",
        "source_kind",
        "supervision_type",
        "query_group_id",
        "query_signature_id",
        "query_view",
        "member_signature_id",
        "pair_left",
        "pair_right",
        "label",
    ]
    expanded["occurrence_count"] = expanded.groupby(dedupe_keys, sort=False)["label"].transform("size").astype("int32")
    expanded = expanded.drop_duplicates(dedupe_keys).copy()
    clean, conflicts = _drop_conflicts(expanded)
    clean = _add_weights(clean)

    summary = {
        "dataset": dataset,
        "label_rows": int(len(dataset_labels)),
        "missing_component_label_rows": missing_components,
        "excluded_initial_origin_rows": excluded_origin_rows,
        "expanded_pair_rows_before_conflict_drop": int(len(expanded)),
        "conflict_pair_rows": int(len(conflicts)),
        "constraint_rows": int(len(clean)),
        "queries": int(clean["query_signature_id"].nunique()),
        "positive_rows": int((clean["label"] == 1).sum()),
        "negative_rows": int((clean["label"] == 0).sum()),
        "tables": {
            str(table): {
                "rows": int(len(table_rows)),
                "queries": int(table_rows["query_signature_id"].nunique()),
                "positive_rows": int((table_rows["label"] == 1).sum()),
                "negative_rows": int((table_rows["label"] == 0).sum()),
            }
            for table, table_rows in clean.groupby("table_name", sort=True)
        },
        "splits": {
            str(split): {
                "rows": int(len(split_rows)),
                "queries": int(split_rows["query_signature_id"].nunique()),
                "positive_rows": int((split_rows["label"] == 1).sum()),
                "negative_rows": int((split_rows["label"] == 0).sum()),
            }
            for split, split_rows in clean.groupby("split", sort=True)
        },
    }
    return clean, conflicts, summary


def main(argv: Sequence[str] | None = None) -> None:
    """Write portable development gold and conflict counts."""
    args = parse_args(argv)
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")
    bundle_root = args.bundle_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {str(dataset) for dataset in args.datasets}
    labels = _load_label_rows(bundle_root, datasets, args.task_splits)
    if args.max_queries is not None:
        query_keys = labels[["dataset", "query_group_id"]].drop_duplicates().sort_values(["dataset", "query_group_id"])
        selected = query_keys.groupby("dataset", sort=False).head(args.max_queries)
        labels = labels.merge(selected, on=["dataset", "query_group_id"], how="inner", validate="many_to_one")

    summaries = []
    output_paths: dict[str, str] = {}
    for dataset in sorted(datasets):
        output_path = output_dir / f"{dataset}_pair_constraints.parquet"
        conflict_path = output_dir / f"{dataset}_conflicts.parquet"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")
        constraints, conflicts, summary = _build_dataset_constraints(bundle_root, dataset, labels)
        constraints.to_parquet(output_path, index=False)
        output_paths[dataset] = str(output_path)
        summaries.append({**summary, "output_path": str(output_path)})
        if summary["conflict_pair_rows"] > 0:
            conflicts.to_parquet(conflict_path, index=False)
        elif conflict_path.exists() and args.overwrite:
            conflict_path.unlink()

    summary_payload = {
        "bundle_root": str(bundle_root),
        "datasets": sorted(datasets),
        "task_splits": list(args.task_splits),
        "max_queries_per_dataset": args.max_queries,
        "target_kind": "safe_action_proxy",
        "outputs": output_paths,
        "summaries": summaries,
    }
    _write_json(output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
