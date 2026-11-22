"""Validate linker source bundles before feature materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking_training.classic import (
    OfficialBundle,
    _classic_stratified_eval_source_specs,
    _drop_unlabeled_singleton_orcid_rows,
    _promoted_stratified_gate_spec,
    _resolve_path,
    _select_classic_stratified_eval_rows,
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


def _validate_singleton_orcid_labels(rows: pa.Table, table_key: str, labels_path: Path) -> None:
    """Require a positive label in at least 10% of marked views per dataset."""
    marked_rows = rows.filter(pa_dataset.field("supervision_type") == "negative_singleton_orcid")
    if len(marked_rows) < 10:
        return
    view_keys = ["dataset", "query_group_id"]
    marked_views = marked_rows.select(view_keys).to_pandas().drop_duplicates()
    marked_counts = marked_views.groupby("dataset", dropna=False).size()
    checked = marked_counts[marked_counts.ge(10)]
    if checked.empty:
        return
    candidates = rows.filter(pa_dataset.field("query_group_id").isin(marked_views["query_group_id"].unique()))
    eligible, _ = _drop_unlabeled_singleton_orcid_rows(candidates.to_pandas(), context=table_key)
    labels = pd.to_numeric(eligible["label"], errors="coerce")
    positive_views = eligible.loc[labels.eq(1), view_keys].drop_duplicates()
    positive_counts = marked_views.merge(positive_views, on=view_keys).groupby("dataset", dropna=False).size()
    for dataset, marked_count in checked.items():
        positive_count = int(positive_counts.get(dataset, 0))
        if 10 * positive_count >= marked_count:
            continue
        raise ValueError(
            f"source table {table_key!r} dataset {dataset!r} has {positive_count}/{marked_count} marked "
            f"negative_singleton_orcid views with a positive label (minimum 10%): {labels_path}. "
            "Mark unreviewed views unlabeled_singleton_orcid and remove split assignments for excluded queries."
        )


def _validate_source_split_coverage(source_bundle: OfficialBundle, label_tables: Mapping[Path, pa.Table]) -> None:
    """Check assigned groups and the calibration tables required by materialization."""
    classic = dict(source_bundle.models["classic"])
    gate = _promoted_stratified_gate_spec(classic)
    if gate is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    split_spec = classic["stratified_eval_test_split"]
    assignments = pd.read_csv(_resolve_path(source_bundle, str(split_spec["assignments_path"])))
    _validate_stratified_split_assignments(assignments)

    def read_source_rows(path: Path, query_ids: set[str]) -> pd.DataFrame:
        rows = label_tables[path.resolve()]
        return rows.filter(pa_dataset.field("query_group_id").isin(sorted(query_ids))).to_pandas()

    for splits, include_calibration_source in ((gate["calibration_splits"], True), ([gate["test_split"]], False)):
        selected = assignments.loc[
            assignments["split"].astype(str).isin(splits), ["query_group_id", "source_key", "base_group_id"]
        ]
        if selected.empty:
            raise ValueError(f"Stratified assignments contain no rows for splits={sorted(splits)}")
        matched = _select_classic_stratified_eval_rows(
            source_bundle,
            classic,
            selected,
            include_calibration_source=include_calibration_source,
            read_source_rows=read_source_rows,
        )
        if include_calibration_source:
            # Materialization still needs an eval file for each assigned calibration source.
            public_keys = set(matched.loc[matched["source_kind"].ne("calibration_source"), "source_key"])
            if missing := set(selected["source_key"]) - public_keys:
                raise ValueError(f"Eval tables have no eligible assigned calibration groups: {sorted(missing)}")


def validate_source_label_tables(
    source_bundle: OfficialBundle,
) -> tuple[int, frozenset[str]]:
    """Validate source labels, split coverage, and base group IDs without opening Arrow datasets."""

    available_tables = source_featureless_table_keys(source_bundle)
    available_table_set = set(available_tables)
    missing_required = sorted(set(REQUIRED_LINKER_TABLE_KEYS) - available_table_set)
    if missing_required:
        raise ValueError(f"official linker source bundle is missing required tables: {missing_required}")

    classic = dict(source_bundle.models["classic"])
    required_eval_keys = set(REQUIRED_LINKER_TABLE_KEYS) - {"train_path"}
    if missing_model_paths := sorted(required_eval_keys - classic.keys()):
        raise ValueError(f"classic is missing required source table paths: {missing_model_paths}")
    split_paths = {
        source["table_key"]: _resolve_path(source_bundle, source["path"]).resolve()
        for source in _classic_stratified_eval_source_specs(classic)
    }
    if undeclared_tables := sorted(split_paths.keys() ^ (available_table_set - {"train_path"})):
        raise ValueError(f"Stratified sources and featureless table keys differ: {undeclared_tables}")
    label_paths = {
        table_key: source_bundle_asset_file(source_bundle, "featureless_rows", table_key).resolve()
        for table_key in available_tables
    }
    for table_key, split_path in split_paths.items():
        labels_path = label_paths[table_key]
        if split_path != labels_path:
            raise ValueError(f"Stratified source path for {table_key!r} is {split_path}, expected {labels_path}")

    source_rows = 0
    observed_datasets: set[str] = set()
    label_tables: dict[Path, pa.Table] = {}
    for table_key, labels_path in label_paths.items():
        labels = pa_dataset.dataset(labels_path, format="parquet")
        check_singletons = "supervision_type" in labels.schema.names
        columns = ["dataset"]
        if table_key in split_paths:
            columns.extend(("base_group_id", "candidate_component_key"))
        if table_key in split_paths or check_singletons:
            columns.extend(("label", "query_group_id"))
        if check_singletons:
            columns.append("supervision_type")
        rows = labels.to_table(columns=columns)
        if not len(rows):
            raise ValueError(f"source table {table_key!r} is empty: {labels_path}")
        if table_key in split_paths:
            label_tables[labels_path] = rows
        if check_singletons:
            _validate_singleton_orcid_labels(rows, table_key, labels_path)
        source_rows += len(rows)
        observed_datasets.update(str(value) for value in rows["dataset"].unique().to_pylist())

    _validate_source_split_coverage(source_bundle, label_tables)

    return source_rows, frozenset(observed_datasets)


def preflight_source_rows(
    source_bundle: OfficialBundle,
    *,
    name_counts_index_root: Path | None,
    arrow_stack: ExitStack,
) -> tuple[int, dict[str, ArrowDataset]]:
    """Validate source rows and open the Arrow generations they reference."""
    source_rows, dataset_names = validate_source_label_tables(source_bundle)

    arrow_datasets: dict[str, ArrowDataset] = {}
    for dataset_name in sorted(dataset_names):
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
