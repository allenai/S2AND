"""Validate linker source bundles before feature materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pa_dataset

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking_training.classic import (
    OfficialBundle,
    _promoted_stratified_gate_spec,
    _resolve_path,
    _validate_stratified_split_assignments,
)
from s2and.production_training_contract import REQUIRED_LINKER_TABLE_KEYS


def source_featureless_table_keys(bundle: OfficialBundle) -> tuple[str, ...]:
    """Return required and declared optional featureless source table keys."""

    files = bundle.assets.get("featureless_rows", {}).get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError("source bundle assets.featureless_rows.files must be a mapping")
    keys: list[str] = [key for key in REQUIRED_LINKER_TABLE_KEYS if key in files]
    for optional_key in ("s_park_eval_path", "s_lee_eval_path"):
        if optional_key in files:
            keys.append(optional_key)
    keys.extend(str(key) for key in files if str(key).startswith("extra_eval_paths."))
    return tuple(dict.fromkeys(keys))


def source_bundle_asset_file(
    bundle: OfficialBundle,
    asset_group: str,
    table_key: str,
) -> Path:
    """Resolve one declared source-bundle asset file."""

    files = dict(bundle.assets[asset_group]["files"])
    if table_key not in files:
        raise KeyError(f"Bundle asset group {asset_group!r} has no file for {table_key!r}")
    return _resolve_path(bundle, str(files[table_key]))


def read_source_parquet_rows(
    path: Path,
    *,
    query_ids: set[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a source table, optionally filtering it to lifecycle query IDs."""

    dataset = pa_dataset.dataset(path, format="parquet")
    schema_names = list(dataset.schema.names)
    if "dataset" not in schema_names:
        raise ValueError(f"Parquet source must contain a dataset column: {path}")
    if query_ids is not None and "query_group_id" not in schema_names:
        raise ValueError(f"Parquet source must contain query_group_id for lifecycle filtering: {path}")
    requested_columns = schema_names if columns is None else list(columns)
    if "dataset" not in requested_columns:
        requested_columns.append("dataset")
    unknown_columns = sorted(set(requested_columns) - set(schema_names))
    if unknown_columns:
        raise ValueError(f"Parquet source is missing requested columns {unknown_columns}: {path}")

    row_filter = None
    if query_ids is not None:
        row_filter = pa_dataset.field("query_group_id").isin(sorted(query_ids))
    rows = dataset.to_table(
        columns=requested_columns,
        filter=row_filter,
    ).to_pandas()
    if columns is not None:
        rows = rows[list(columns)]
    return rows


def open_source_arrow_dataset(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    name_counts_index_root: Path | None = None,
    require_name_counts_index: bool = True,
) -> ArrowDataset:
    """Open one source Arrow dataset and validate its name-count binding."""

    dataset_dir = (bundle.root / "datasets" / str(dataset_name)).resolve()
    dataset = ArrowDataset.open(
        dataset_dir,
        require_specter=True,
        require_name_counts_index=require_name_counts_index,
    )
    if name_counts_index_root is not None and dataset.name_counts_index is not None:
        expected = Path(name_counts_index_root).resolve()
        observed = Path(dataset.name_counts_index.path).resolve()
        if observed != expected:
            dataset.close()
            raise ValueError(f"Arrow dataset {dataset_name!r} binds name_counts_index {observed}, expected {expected}")
    return dataset


def validate_source_bundle_support_files(source_bundle: OfficialBundle) -> list[str]:
    """Validate support files copied before feature materialization starts."""

    required_files = [source_bundle.root / "bundle.json"]
    splits_dir = source_bundle.root / "splits"
    if not splits_dir.is_dir():
        raise ValueError(f"source bundle is missing splits directory: {splits_dir}")
    split_files = sorted(path for path in splits_dir.rglob("*") if path.is_file())
    if not split_files:
        raise ValueError(f"source bundle splits directory contains no files: {splits_dir}")
    required_files.extend(split_files)

    classic = source_bundle.models.get("classic")
    if not isinstance(classic, Mapping):
        raise ValueError("source bundle models.classic must be a mapping")
    direct_support_path = classic.get("classic_gate_internal_eval_base_groups_path")
    if not isinstance(direct_support_path, str) or not direct_support_path:
        raise ValueError("classic.classic_gate_internal_eval_base_groups_path is required")
    internal_eval_path = _resolve_path(source_bundle, direct_support_path)
    split_spec = classic.get("stratified_eval_test_split")
    if not isinstance(split_spec, Mapping):
        raise ValueError("classic.stratified_eval_test_split.assignments_path is required")
    assignments_path_value = split_spec.get("assignments_path")
    if not isinstance(assignments_path_value, str) or not assignments_path_value:
        raise ValueError("classic.stratified_eval_test_split.assignments_path is required")
    assignments_path = _resolve_path(source_bundle, assignments_path_value)
    gate_spec = _promoted_stratified_gate_spec(dict(classic))
    if gate_spec is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    required_files.extend((internal_eval_path, assignments_path))

    missing = sorted(str(path) for path in required_files if not path.is_file())
    if missing:
        raise ValueError(f"source bundle is missing required support files: {missing}")
    assignments = pd.read_csv(assignments_path)
    _validate_stratified_split_assignments(assignments)
    required_splits = {*gate_spec["calibration_splits"], str(gate_spec["test_split"])}
    missing_splits = sorted(required_splits - set(assignments["split"].astype(str)))
    if missing_splits:
        raise ValueError(f"Stratified split assignments omit configured calibration/test splits: {missing_splits}")
    return [str(path) for path in dict.fromkeys(required_files)]


def preflight_source_rows(
    source_bundle: OfficialBundle,
    *,
    name_counts_index_root: Path | None,
    arrow_stack: ExitStack,
) -> tuple[int, dict[str, ArrowDataset]]:
    """Validate source row presence and every referenced Arrow generation."""

    available_tables = source_featureless_table_keys(source_bundle)
    available_table_set = set(available_tables)
    missing_required = sorted(set(REQUIRED_LINKER_TABLE_KEYS) - available_table_set)
    if missing_required:
        raise ValueError(f"official linker source bundle is missing required tables: {missing_required}")

    source_rows = 0
    observed_datasets: set[str] = set()
    for table_key in available_tables:
        labels_path = source_bundle_asset_file(source_bundle, "featureless_rows", table_key)
        rows = read_source_parquet_rows(labels_path, columns=["dataset"])
        if rows.empty:
            raise ValueError(f"source table {table_key!r} is empty: {labels_path}")
        source_rows += len(rows)
        observed_datasets.update(rows["dataset"].astype(str))

    arrow_datasets: dict[str, ArrowDataset] = {}
    for dataset_name in sorted(observed_datasets):
        arrow_datasets[dataset_name] = arrow_stack.enter_context(
            open_source_arrow_dataset(
                source_bundle,
                dataset_name,
                name_counts_index_root=name_counts_index_root,
            )
        )

    name_count_hashes = {
        dataset.name_counts_index.manifest_sha256
        for dataset in arrow_datasets.values()
        if dataset.name_counts_index is not None
    }
    if len(name_count_hashes) > 1:
        raise ValueError(
            f"linker source datasets reference multiple name-count generations: {sorted(name_count_hashes)}"
        )
    return source_rows, arrow_datasets
