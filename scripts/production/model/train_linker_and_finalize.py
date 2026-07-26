"""Train, calibrate, and evaluate the promoted joint-safe linker target.

This is the official replay entrypoint for the promoted LightGBM
linker/reranker target. It intentionally pins the promoted target JSON instead
of trusting bundle manifests whose classic model specs predate the promotion.

The flow starts from the self-contained Arrow+labels bundle, rebuilds the
promoted feature tables through Rust/Arrow query, summary, row-signal,
pairwise, and row-formula paths, then runs the train/calibrate/eval stack. The
active candidate-member contract is block-local for retrieval, pairwise
distance summaries, and appended `pw_*` aggregates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.ipc as pa_ipc

REPO_ROOT = Path(__file__).resolve().parents[3]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from s2and import feature_port  # noqa: E402
from s2and.arrow_inputs import ValidatedArrowInputs, validate_arrow_prediction_artifacts  # noqa: E402
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER, NORMALIZATION_VERSION  # noqa: E402
from s2and.incremental_linking.array_validation import as_retrieval_rank_uint16_1d  # noqa: E402
from s2and.incremental_linking.artifact import (  # noqa: E402
    load_incremental_linking_artifact,
    save_incremental_linking_artifact,
)
from s2and.incremental_linking.feature_block import (  # noqa: E402
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns  # noqa: E402
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view  # noqa: E402
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch  # noqa: E402
from s2and.incremental_linking.policy import require_arrow_name_counts_index_for_clusterer  # noqa: E402
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS  # noqa: E402
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features  # noqa: E402
from s2and.incremental_linking.runtime import compute_candidate_batch_pairwise_model_and_aggregate_stats  # noqa: E402
from s2and.incremental_linking_training.classic import (  # noqa: E402
    PROMOTED_NON_PAIRWISE_COLUMNS,
    PROMOTED_PAIRWISE_COLUMNS,
    SUPPORTED_PROMOTED_FEATURE_COLUMNS,
    WEIGHTED_ERROR_WEIGHTS,
    FittedClassicRun,
    OfficialBundle,
    _drop_unlabeled_singleton_orcid_rows,
    _load_classic_stratified_eval_rows,
    _promoted_stratified_gate_spec,
    _resolve_path,
    _validate_stratified_split_assignments,
    load_bundle,
    run_classic,
)
from s2and.incremental_linking_training.data_loading import load_clusterer  # noqa: E402
from s2and.production_bundle import finalize_production_bundle, production_version_from_bundle_dir  # noqa: E402
from s2and.production_model import pairwise_bundle_binding  # noqa: E402
from s2and.runtime import build_runtime_context  # noqa: E402
from s2and.rust_calls import get_constraint_labels_index_arrays_rust  # noqa: E402

DEFAULT_NAME_COUNTS_INDEX_ROOT: Path | None = None
DEFAULT_TOTAL_RAM_BYTES = 48 * 1024**3
PRODUCTION_MAX_EXEMPLARS = 4
PRODUCTION_PAIRWISE_MODEL_NAN_POLICY = "preserve"
PRODUCTION_PAIRWISE_AGGREGATE_NAN_POLICY = "zero"
PRODUCTION_ROW_NAN_POLICY = "finite"
EVALUATED_ARTIFACT_DIRNAME = "incremental_linker_artifact"
REQUIRED_TABLE_KEYS = (
    "train_path",
    "classic_gate_source_path",
    "s2and_eval_path",
    "hwang_eval_path",
)
INTEGER_OFFICIAL_METRIC_KEYS = frozenset(
    {
        "training_rows",
        "training_positive_rows",
        "stratified_test_queries",
        "stratified_test_errors",
        "stratified_test_false_abstain",
        "stratified_test_false_link",
        "stratified_test_wrong_candidate_link",
    }
)
FLOAT_OFFICIAL_METRIC_KEYS = frozenset(
    {
        "stratified_test_accuracy",
        "stratified_test_balanced_accuracy",
        "stratified_test_error_rate",
        "false_abstain_error_rate",
        "false_link_error_rate",
        "wrong_link_error_rate",
        "weighted_average_error",
    }
)
SUPPORTED_OFFICIAL_METRIC_KEYS = (
    INTEGER_OFFICIAL_METRIC_KEYS | FLOAT_OFFICIAL_METRIC_KEYS | {"weighted_average_error_weights"}
)


@dataclass
class ArrowRustDatasetContext:
    """Arrow-only dataset state shared across linker row tables for one dataset."""

    dataset_name: str
    row_component_scope: str
    pairwise_component_scope: str
    runtime_context: Any
    arrow_paths: ValidatedArrowInputs
    component_members: dict[str, tuple[str, ...]]
    cluster_seeds_require: dict[str, str]
    cluster_seeds_disallow: frozenset[tuple[str, str]]
    seed_constrained_signature_ids: frozenset[str]
    max_block_component_size: int


@dataclass
class ArrowRustPendingShard:
    """One table/dataset slice that still needs feature materialization."""

    table_key: str
    dataset_name: str
    rows: pd.DataFrame
    row_positions: np.ndarray
    partial_path: Path


@dataclass
class ArrowRustTablePlan:
    """Materialization state for one output feature table."""

    table_key: str
    labels_path: Path
    output_path: Path
    labels: pd.DataFrame
    partial_paths: list[Path]
    dataset_summaries: list[dict[str, Any]]
    label_filtering_summary: dict[str, Any]
    structural_cleaning_summary: dict[str, Any]
    started: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_finite_metric_values(
    value: Any,
    *,
    path: str,
    context: str,
    error_type: type[Exception],
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_metric_values(
                nested,
                path=f"{path}.{key}",
                context=context,
                error_type=error_type,
            )
    elif isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise error_type(f"{context} {path} must be numeric, got {value!r}")
    elif not math.isfinite(float(value)):
        raise error_type(f"{context} {path} must be finite, got {value!r}")


def _load_target(path: Path) -> dict[str, Any]:
    target = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(target, dict):
        raise ValueError(f"Promoted target must be a JSON object in {path}")
    raw_features = target.get("features")
    if not isinstance(raw_features, list) or any(
        not isinstance(feature, str) or not feature for feature in raw_features
    ):
        raise ValueError(f"Promoted target features must be a list of nonempty strings in {path}")
    feature_count = target.get("feature_count")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count < 0:
        raise ValueError(f"Promoted target feature_count must be a nonnegative integer in {path}")
    features = tuple(raw_features)
    if len(features) != feature_count:
        raise ValueError(f"Promoted target feature_count mismatch in {path}")
    if len(features) != len(set(features)):
        raise ValueError(f"Promoted target contains duplicate features in {path}")
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
    expected_features = promoted_linker_feature_columns()
    if features != expected_features:
        raise ValueError(
            "Promoted target features must exactly match promoted_linker_feature_columns() "
            f"in canonical order in {path}"
        )
    params = target.get("params")
    if not isinstance(params, Mapping) or not params:
        raise ValueError(f"Promoted target params must be a nonempty object in {path}")
    n_estimators = params.get("n_estimators")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError(f"Promoted target params.n_estimators must be a positive integer in {path}")
    metrics = target.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Promoted target metrics must be an object in {path}")
    _require_finite_metric_values(
        metrics,
        path="metrics",
        context="Promoted target metric",
        error_type=ValueError,
    )
    unknown_metrics = sorted(set(metrics) - SUPPORTED_OFFICIAL_METRIC_KEYS)
    if unknown_metrics:
        raise ValueError(f"Promoted target contains unknown metric keys: {unknown_metrics}")
    for key in INTEGER_OFFICIAL_METRIC_KEYS & set(metrics):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Promoted target metric metrics.{key} must be a nonnegative integer")
    weights = metrics.get("weighted_average_error_weights")
    if weights is not None:
        if not isinstance(weights, Mapping) or dict(weights) != dict(WEIGHTED_ERROR_WEIGHTS):
            raise ValueError(
                "Promoted target metric metrics.weighted_average_error_weights "
                f"must equal {dict(WEIGHTED_ERROR_WEIGHTS)}"
            )
    if metrics and set(metrics) != SUPPORTED_OFFICIAL_METRIC_KEYS:
        raise ValueError(
            "Promoted target metrics must be empty or contain the complete official metric set: "
            f"missing={sorted(SUPPORTED_OFFICIAL_METRIC_KEYS - set(metrics))}"
        )
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


def _source_featureless_table_keys(bundle: OfficialBundle) -> tuple[str, ...]:
    files = bundle.assets.get("featureless_rows", {}).get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError("source bundle assets.featureless_rows.files must be a mapping")
    keys: list[str] = [key for key in REQUIRED_TABLE_KEYS if key in files]
    for optional_key in ("s_park_eval_path", "s_lee_eval_path"):
        if optional_key in files:
            keys.append(optional_key)
    keys.extend(str(key) for key in files if str(key).startswith("extra_eval_paths."))
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


def _read_selected_parquet_rows(
    path: Path,
    *,
    datasets: set[str] | None,
    limit_rows: int | None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read selected rows through one bounded Arrow dataset scanner."""

    rows, _source_rows, _available_datasets = _scan_parquet_rows(
        path,
        datasets=datasets,
        limit_rows=limit_rows,
        columns=columns,
        inventory=False,
    )
    return rows


def _scan_parquet_rows(
    path: Path,
    *,
    datasets: set[str] | None,
    limit_rows: int | None,
    columns: Sequence[str] | None,
    inventory: bool,
) -> tuple[pd.DataFrame, int, set[str]]:
    """Scan once, optionally continuing after the row limit for selector inventory."""

    if limit_rows is not None and int(limit_rows) <= 0:
        raise ValueError("limit_rows must be > 0")
    dataset = pa_dataset.dataset(path, format="parquet")
    schema_names = list(dataset.schema.names)
    if "dataset" not in schema_names:
        raise ValueError(f"Parquet source must contain a dataset column: {path}")
    requested_columns = schema_names if columns is None else list(columns)
    if "dataset" not in requested_columns:
        requested_columns.append("dataset")
    unknown_columns = sorted(set(requested_columns) - set(schema_names))
    if unknown_columns:
        raise ValueError(f"Parquet source is missing requested columns {unknown_columns}: {path}")

    parts: list[pd.DataFrame] = []
    available_datasets: set[str] = set()
    scanned_rows = 0
    selected_rows = 0
    for batch in dataset.scanner(columns=requested_columns, batch_size=65_536).to_batches():
        scanned_rows += int(batch.num_rows)
        frame = batch.to_pandas()
        available_datasets.update(frame["dataset"].astype(str))
        if datasets is not None:
            frame = frame.loc[frame["dataset"].astype(str).isin(datasets)]
        if limit_rows is not None:
            frame = frame.iloc[: max(0, int(limit_rows) - selected_rows)]
        if not frame.empty:
            parts.append(frame)
            selected_rows += len(frame)
        if limit_rows is not None and selected_rows >= int(limit_rows) and not inventory:
            break

    if parts:
        rows = pd.concat(parts, ignore_index=True)
    else:
        rows = pa.Table.from_arrays(
            [pa.array([], type=dataset.schema.field(name).type) for name in requested_columns],
            names=requested_columns,
        ).to_pandas()
    if columns is not None:
        rows = rows[list(columns)]
    return rows, scanned_rows, available_datasets


def _clean_arrow_rust_structural_rows(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    rows: pd.DataFrame,
    component_membership_cache: dict[str, pd.DataFrame],
    name_counts_index_root: Path | None,
    arrow_paths_cache: Mapping[str, ValidatedArrowInputs] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove candidate rows with no non-query member using Arrow signature blocks."""

    required = {"dataset", "query_group_id", "query_signature_id", "candidate_component_key", "label"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{table_key}: arrow-rust structural cleaning missing columns: {missing}")
    started = time.perf_counter()
    keep_mask = np.ones(len(rows), dtype=bool)
    labels = pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(np.int8)
    query_ids_before = set(rows["query_group_id"].astype(str))
    positive_query_ids_before = set(rows.loc[labels == 1, "query_group_id"].astype(str))
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_name, dataset_rows in rows.groupby(rows["dataset"].astype(str), sort=False):
        membership = _arrow_component_membership_summary(
            source_bundle,
            str(dataset_name),
            cache=component_membership_cache,
            name_counts_index_root=name_counts_index_root,
            arrow_paths_cache=arrow_paths_cache,
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
        component_member_count = local["_component_member_count"].astype(np.int64)
        drop = (component_member_count == 0) | (
            (component_member_count == 1)
            & local["_component_single_member_signature_id"].astype(str).eq(local["query_signature_id"].astype(str))
        )
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
        "policy": "drop_candidate_rows_with_no_non_query_block_local_members_arrow",
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


def _block_local_member_ids_from_signature_blocks(
    component_key: str,
    member_ids: tuple[str, ...],
    signature_to_block: Mapping[str, str],
) -> tuple[str, ...]:
    if "::" not in str(component_key):
        return member_ids
    block_key, _cluster_id = str(component_key).split("::", 1)
    return tuple(
        signature_id for signature_id in member_ids if str(signature_to_block.get(str(signature_id), "")) == block_key
    )


def _signature_id_to_index(featurizer: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for index, signature_id in enumerate(featurizer.signature_ids()):
        out[str(signature_id)] = int(index)
    return out


def _resolve_arrow_manifest_path(raw_value: Any, *, dataset_dir: Path, bundle_root: Path) -> Path:
    raw_path = Path(str(raw_value))
    candidates = [raw_path] if raw_path.is_absolute() else []
    if not raw_path.is_absolute():
        candidates.extend((dataset_dir / raw_path, bundle_root / raw_path, REPO_ROOT / raw_path, raw_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Arrow manifest path does not exist: {raw_value}")


def _arrow_paths_for_dataset(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    name_counts_index_root: Path | None = None,
    require_name_counts_index: bool = True,
) -> ValidatedArrowInputs:
    dataset_dir = (bundle.root / "datasets" / str(dataset_name)).resolve()
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Arrow dataset manifest missing for {dataset_name!r}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_paths = manifest.get("paths", {})
    if not isinstance(raw_paths, Mapping):
        raise ValueError(f"Arrow dataset manifest paths must be a mapping: {manifest_path}")
    paths: dict[str, str] = {}
    for key, raw_value in raw_paths.items():
        paths[str(key)] = str(_resolve_arrow_manifest_path(raw_value, dataset_dir=dataset_dir, bundle_root=bundle.root))
    if name_counts_index_root is not None:
        name_counts_index = Path(name_counts_index_root).resolve()
        if not name_counts_index.exists():
            raise FileNotFoundError(f"name_counts_index root does not exist: {name_counts_index}")
        paths["name_counts_index"] = str(name_counts_index)
    return validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=require_name_counts_index,
        context=f"Arrow dataset {dataset_name!r} for linker train/calibrate/eval",
        producer_hint=(
            "include signatures/papers/paper_authors/specter Arrow files, raw-planner batch indexes, "
            "and name_counts_index in the bundle manifest"
        ),
    )


def _component_member_ids_by_key(path: Path) -> dict[str, tuple[str, ...]]:
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    out: dict[str, tuple[str, ...]] = {}
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        out[str(component_key)] = tuple(str(value) for value in group.sort_values("member_index")["signature_id"])
    return out


def _seed_constrained_signature_ids_from_maps(
    cluster_seeds_require: Mapping[str, str],
    cluster_seeds_disallow: Iterable[tuple[str, str]],
) -> frozenset[str]:
    signature_ids = {str(signature_id) for signature_id in cluster_seeds_require}
    for left, right in cluster_seeds_disallow:
        signature_ids.add(str(left))
        signature_ids.add(str(right))
    return frozenset(signature_ids)


def _load_arrow_seed_constraints(arrow_paths: Mapping[str, str]) -> tuple[dict[str, str], frozenset[tuple[str, str]]]:
    require_path = arrow_paths.get("cluster_seeds")
    cluster_seeds_require = read_cluster_seeds_arrow(Path(require_path)) if require_path else {}
    disallow_path = arrow_paths.get("cluster_seed_disallows")
    raw_disallows = read_cluster_seed_disallows_arrow(Path(disallow_path)) if disallow_path else ()
    cluster_seeds_disallow = frozenset((str(left), str(right)) for left, right in raw_disallows)
    return ({str(key): str(value) for key, value in cluster_seeds_require.items()}, cluster_seeds_disallow)


def _load_arrow_signature_blocks(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    name_counts_index_root: Path | None,
    arrow_paths: ValidatedArrowInputs | None = None,
) -> dict[str, str]:
    if arrow_paths is None:
        arrow_paths = _arrow_paths_for_dataset(
            bundle,
            dataset_name,
            name_counts_index_root=name_counts_index_root,
            require_name_counts_index=False,
        )
    path = Path(arrow_paths["signatures"])
    out: dict[str, str] = {}
    with pa_ipc.open_file(path) as reader:
        schema_names = set(reader.schema.names)
        if "author_block" not in schema_names:
            return out
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(["signature_id", "author_block"])
            signature_ids = batch.column(0).to_pylist()
            blocks = batch.column(1).to_pylist()
            out.update(
                {
                    str(signature_id): str(block or "")
                    for signature_id, block in zip(signature_ids, blocks, strict=True)
                    if signature_id is not None
                }
            )
    return out


def _arrow_component_membership_summary(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    cache: dict[str, pd.DataFrame],
    name_counts_index_root: Path | None,
    arrow_paths_cache: Mapping[str, ValidatedArrowInputs] | None = None,
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
        signature_to_block = _load_arrow_signature_blocks(
            bundle,
            dataset_name,
            name_counts_index_root=name_counts_index_root,
            arrow_paths=None if arrow_paths_cache is None else arrow_paths_cache.get(dataset_name),
        )

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


def _build_arrow_rust_dataset_context(
    *,
    source_bundle: OfficialBundle,
    dataset_name: str,
    name_counts_index_root: Path | None,
    arrow_paths: ValidatedArrowInputs | None = None,
) -> ArrowRustDatasetContext:
    started = time.perf_counter()
    if arrow_paths is None:
        arrow_paths = _arrow_paths_for_dataset(
            source_bundle,
            dataset_name,
            name_counts_index_root=name_counts_index_root,
        )
    member_path = _resolve_path(
        source_bundle,
        str(source_bundle.assets["candidate_members"]["datasets"][dataset_name]),
    )
    component_members = _component_member_ids_by_key(member_path)
    cluster_seeds_require, cluster_seeds_disallow = _load_arrow_seed_constraints(arrow_paths)
    seed_constrained_signature_ids = _seed_constrained_signature_ids_from_maps(
        cluster_seeds_require,
        cluster_seeds_disallow,
    )
    max_block_component_size = max((len(members) for members in component_members.values()), default=0)
    print(
        json.dumps(
            {
                "event": "arrow_rust_dataset_context_ready",
                "dataset": dataset_name,
                "components": int(len(component_members)),
                "component_scope": "block-local",
                "name_counts_index": arrow_paths.get("name_counts_index"),
                "cluster_seed_require_count": int(len(cluster_seeds_require)),
                "cluster_seed_disallow_count": int(len(cluster_seeds_disallow)),
                "seconds": round(float(time.perf_counter() - started), 3),
            }
        ),
        flush=True,
    )
    return ArrowRustDatasetContext(
        dataset_name=dataset_name,
        row_component_scope="block-local",
        pairwise_component_scope="block-local",
        runtime_context=build_runtime_context(
            "joint_safe_link_arrow_rust_featureization",
            backend="rust",
        ),
        arrow_paths=arrow_paths,
        component_members=component_members,
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=cluster_seeds_disallow,
        seed_constrained_signature_ids=seed_constrained_signature_ids,
        max_block_component_size=int(max_block_component_size),
    )


def _release_arrow_rust_dataset_context(context: ArrowRustDatasetContext) -> None:
    context.component_members.clear()
    gc.collect()


def _signature_indices_from_plan_ids(
    signature_ids: Sequence[Any],
    signature_id_to_index: Mapping[str, int],
    *,
    field_name: str,
) -> np.ndarray:
    out = np.empty(len(signature_ids), dtype=np.uint32)
    for index, signature_id in enumerate(signature_ids):
        key = str(signature_id)
        try:
            out[index] = int(signature_id_to_index[key])
        except KeyError as exc:
            raise KeyError(f"{field_name} contains signature_id missing from Arrow featurizer: {key}") from exc
    return out


def _row_signal_from_plan(plan: Mapping[str, Any], key: str, dtype: Any, row_count: int) -> np.ndarray:
    values = np.asarray(plan[key], dtype=dtype)
    if values.shape != (row_count,):
        raise ValueError(f"raw Arrow labeled plan {key!r} must have shape ({row_count},), got {values.shape}")
    return values


def _arrow_labeled_plan_to_batch_and_row_signals(
    *,
    plan: Mapping[str, Any],
    rows: pd.DataFrame,
    signature_id_to_index: Mapping[str, int],
    row_group_ids: Sequence[int],
) -> tuple[LinkerCandidateBatch, dict[str, Any]]:
    row_count = int(plan["row_count"])
    if row_count != len(rows):
        raise ValueError(f"raw Arrow labeled plan row_count mismatch: {row_count} != {len(rows)}")
    left = _signature_indices_from_plan_ids(
        plan.get("left_signature_ids", ()),
        signature_id_to_index,
        field_name="left_signature_ids",
    )
    right = _signature_indices_from_plan_ids(
        plan.get("right_signature_ids", ()),
        signature_id_to_index,
        field_name="right_signature_ids",
    )
    pair_row_indices = np.asarray(plan["pair_row_indices"], dtype=np.uint32)
    if not (len(left) == len(right) == len(pair_row_indices)):
        raise ValueError(
            "raw Arrow labeled plan pair arrays must have equal length: "
            f"left={len(left)} right={len(right)} rows={len(pair_row_indices)}"
        )
    row_component_keys = tuple(str(value) for value in plan["row_component_keys"])
    if len(row_component_keys) != row_count:
        raise ValueError(
            f"raw Arrow labeled plan row_component_keys length mismatch: {len(row_component_keys)} != {row_count}"
        )
    retrieval_scores = _row_signal_from_plan(plan, "retrieval_scores", np.float32, row_count)
    retrieval_ranks = as_retrieval_rank_uint16_1d(
        "retrieval_ranks",
        _row_signal_from_plan(plan, "retrieval_ranks", object, row_count),
    )
    batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left,
        right_signature_indices=right,
        pair_row_indices=pair_row_indices,
        row_query_signature_indices=np.asarray(row_group_ids, dtype=np.uint32),
        row_component_keys=row_component_keys,
        labels=rows["label"].to_numpy(dtype=np.int8, copy=False) if "label" in rows.columns else None,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
    )
    query_views = np.asarray(plan["row_query_views"], dtype=object)
    query_first_tokens = _row_signal_from_plan(plan, "row_query_first_tokens", object, row_count)
    row_signals: dict[str, Any] = {
        "retrieval_score": retrieval_scores,
        "retrieval_rank": retrieval_ranks.astype(np.float32, copy=False),
        "candidate_component_key": np.asarray(row_component_keys, dtype=object),
        "query_view": query_views,
        "query_author": np.asarray(plan["row_query_authors"], dtype=object),
        "first_name_bucket": np.asarray(
            [
                first_name_bucket_from_token_view(str(token or ""), str(view or ""))
                for token, view in zip(query_first_tokens, query_views, strict=True)
            ],
            dtype=object,
        ),
    }
    for raw_key, signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        row_signals[signal_key] = _row_signal_from_plan(plan, raw_key, dtype, row_count)
    for raw_key, signal_key in (
        ("row_query_has_specter", "query_has_specter"),
        ("row_query_has_name_counts", "query_has_name_counts"),
        ("row_candidate_has_affiliations", "candidate_has_affiliations"),
        ("row_candidate_has_coauthors", "candidate_has_coauthors"),
        ("row_candidate_has_specter_exemplars", "candidate_has_specter_exemplars"),
        ("row_candidate_has_name_counts", "candidate_has_name_counts"),
    ):
        row_signals[signal_key] = _row_signal_from_plan(plan, raw_key, np.float32, row_count)
    return batch, row_signals


def _resolve_arrow_rust_pair_labels(
    *,
    clusterer: Any,
    batch: LinkerCandidateBatch,
    featurizer: Any,
    n_jobs: int,
    pair_seed_bypass: np.ndarray | None = None,
    pair_ignore_disallow: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    pair_count = int(batch.pair_count)
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    started = time.perf_counter()
    if pair_seed_bypass is None:
        pair_seed_bypass = np.zeros(pair_count, dtype=bool)
    else:
        pair_seed_bypass = np.asarray(pair_seed_bypass, dtype=bool)
    if pair_ignore_disallow is None:
        if batch.labels is not None and pair_count:
            positive_rows = np.asarray(batch.labels, dtype=np.int8) == 1
            pair_ignore_disallow = positive_rows[np.asarray(batch.pair_row_indices, dtype=np.uint32)]
        else:
            pair_ignore_disallow = np.zeros(pair_count, dtype=bool)
    else:
        pair_ignore_disallow = np.asarray(pair_ignore_disallow, dtype=bool)
    if len(pair_seed_bypass) != pair_count:
        raise ValueError(f"pair_seed_bypass length {len(pair_seed_bypass)} != pair_count {pair_count}")
    if len(pair_ignore_disallow) != pair_count:
        raise ValueError(f"pair_ignore_disallow length {len(pair_ignore_disallow)} != pair_count {pair_count}")

    constraints_enabled = bool(getattr(clusterer, "use_default_constraints_as_supervision", True)) and pair_count > 0
    if constraints_enabled:
        labels = get_constraint_labels_index_arrays_rust(
            batch.left_signature_indices,
            batch.right_signature_indices,
            dont_merge_cluster_seeds=True,
            incremental_dont_use_cluster_seeds=False,
            num_threads=max(1, int(n_jobs)),
            featurizer=featurizer,
            suppress_orcid=True,
        )
    seed_bypass_indices = np.flatnonzero(pair_seed_bypass) if constraints_enabled else np.asarray([], dtype=np.int64)
    if len(seed_bypass_indices):
        labels[seed_bypass_indices] = get_constraint_labels_index_arrays_rust(
            batch.left_signature_indices[seed_bypass_indices],
            batch.right_signature_indices[seed_bypass_indices],
            dont_merge_cluster_seeds=True,
            incremental_dont_use_cluster_seeds=True,
            num_threads=max(1, int(n_jobs)),
            featurizer=featurizer,
            suppress_orcid=True,
        )
    disallow_ignored = 0
    if np.any(pair_ignore_disallow):
        disallowed = pair_ignore_disallow & np.asarray(
            [_constraint_label_is_disallow(float(label)) for label in labels],
            dtype=bool,
        )
        disallow_ignored = int(disallowed.sum())
        labels[disallowed] = np.nan
    return labels, {
        "constraint_pair_count": pair_count,
        "constraint_batch_calls": int(constraints_enabled),
        "constraint_seed_bypass_pair_count": int(len(seed_bypass_indices)),
        "constraint_seed_bypass_batch_calls": int(len(seed_bypass_indices) > 0),
        "constraint_disallow_ignored_pair_count": disallow_ignored,
        "constraint_seconds": round(float(time.perf_counter() - started), 3),
        "constraint_api_mode": "rust_index_arrays",
    }


def _assert_pairwise_model_supports_arrow_materialization(clusterer: Any, model_path: Path) -> None:
    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = tuple(str(value) for value in getattr(featurizer_info, "features_to_use", ()) or ())
        if "reference_features" in features_to_use:
            raise ValueError(
                f"Pairwise model {model_path} uses reference_features in {attr_name}; "
                "Arrow feature materialization does not support reference features."
            )


def _feature_nan_policy_summary() -> dict[str, str]:
    return {
        "pairwise_model_nan_policy": PRODUCTION_PAIRWISE_MODEL_NAN_POLICY,
        "pairwise_aggregate_nan_policy": PRODUCTION_PAIRWISE_AGGREGATE_NAN_POLICY,
        "row_nan_policy": PRODUCTION_ROW_NAN_POLICY,
    }


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


def _row_allows_seed_constraint_bypass(
    row: Any,
    *,
    seed_constraint_signature_ids: frozenset[str],
) -> bool:
    if _truthy_row_value(getattr(row, "query_in_seed_before_holdout", None)):
        return True
    query_signature_id = getattr(row, "query_signature_id", None)
    if query_signature_id is not None and str(query_signature_id) in seed_constraint_signature_ids:
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


def _has_query_seed_connection_from_maps(
    cluster_seeds_require: Mapping[str, str],
    cluster_seeds_disallow: Iterable[tuple[str, str]],
    *,
    query_signature_id: str,
    candidate_signature_ids: Sequence[str],
) -> bool:
    query_signature_id = str(query_signature_id)
    require = {str(signature_id): str(cluster_id) for signature_id, cluster_id in cluster_seeds_require.items()}
    disallow = {(str(left), str(right)) for left, right in cluster_seeds_disallow}
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


def _arrow_row_seed_bypass_mask(
    rows: pd.DataFrame,
    component_members: Mapping[str, Sequence[str]],
    *,
    cluster_seeds_require: Mapping[str, str],
    cluster_seeds_disallow: Iterable[tuple[str, str]],
    seed_constrained_signature_ids: frozenset[str],
) -> np.ndarray:
    row_seed_bypass = np.zeros(len(rows), dtype=bool)
    if not seed_constrained_signature_ids:
        return row_seed_bypass
    for row_index, row in enumerate(rows.itertuples(index=False)):
        row_any = cast(Any, row)
        query_signature_id = str(row_any.query_signature_id)
        component_key = str(row_any.candidate_component_key)
        active_member_ids = [
            str(signature_id)
            for signature_id in component_members.get(component_key, ())
            if str(signature_id) != query_signature_id
        ]
        if _row_allows_seed_constraint_bypass(
            row_any,
            seed_constraint_signature_ids=seed_constrained_signature_ids,
        ) and _has_query_seed_connection_from_maps(
            cluster_seeds_require,
            cluster_seeds_disallow,
            query_signature_id=query_signature_id,
            candidate_signature_ids=active_member_ids,
        ):
            row_seed_bypass[row_index] = True
    return row_seed_bypass


def _constraint_label_is_disallow(label: float) -> bool:
    if math.isnan(float(label)):
        return False
    return float(label) + float(LARGE_INTEGER) >= float(LARGE_DISTANCE)


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
) -> dict[str, Any]:
    if output_bundle_root.exists():
        raise ValueError(f"Fresh feature bundle output already exists: {output_bundle_root}")
    output_bundle_root.mkdir(parents=True)
    shutil.copytree(source_bundle.root / "splits", output_bundle_root / "splits")
    shutil.copy2(source_bundle.root / "bundle.json", output_bundle_root / "bundle.json")
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    payload["bundle_name"] = f"{payload['bundle_name']}_promoted_rust_recomputed_pw"
    payload["expected_metrics"] = {}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


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
        raise ValueError(f"Arrow feature materialization left unfilled row signals: {missing}")


def _pairwise_feature_values(pairwise_stats: Any) -> dict[str, np.ndarray]:
    pairwise_columns = tuple(pairwise_stats.aggregate_feature_columns)
    if pairwise_columns != PROMOTED_PAIRWISE_COLUMNS:
        raise ValueError("Rust pairwise aggregate column order mismatch in Arrow materialization")
    pairwise_matrix = pairwise_stats.feature_matrix().astype(np.float32, copy=False)
    return {
        column: np.asarray(pairwise_matrix[:, column_index], dtype=np.float32)
        for column_index, column in enumerate(pairwise_columns)
    }


def _assemble_promoted_feature_values(
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


def _materialize_arrow_rust_dataset_rows(
    *,
    context: ArrowRustDatasetContext,
    rows: pd.DataFrame,
    target_features: Sequence[str],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    max_exemplars: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    dataset_name = context.dataset_name
    dataset_rows = rows.reset_index(drop=True).copy()
    row_count = len(dataset_rows)
    group_codes = tuple(
        int(value) for value in pd.factorize(dataset_rows["query_group_id"].astype(str), sort=False)[0].tolist()
    )
    rust_module = feature_port._require_rust_runtime()  # noqa: SLF001
    plan_fn = rust_module.raw_arrow_labeled_candidate_plan
    plan_started = time.perf_counter()
    retrieval_ranks = as_retrieval_rank_uint16_1d(
        "retrieval_rank",
        pd.to_numeric(dataset_rows["retrieval_rank"], errors="raise").to_numpy(),
    )
    raw_plan = plan_fn(
        dict(context.arrow_paths),
        dataset_rows["query_signature_id"].astype(str).tolist(),
        dataset_rows["query_view"].astype(str).tolist(),
        dataset_rows["query_group_id"].astype(str).tolist(),
        dataset_rows["candidate_component_key"].astype(str).tolist(),
        retrieval_ranks.tolist(),
        context.component_members,
        orcid_enabled=False,
        num_threads=max(1, int(n_jobs)),
        max_exemplars=int(max_exemplars),
        name_counts_index=context.arrow_paths.native_name_counts_index,
    )
    raw_plan_seconds = float(time.perf_counter() - plan_started)
    signature_ids = tuple(str(signature_id) for signature_id in raw_plan["signature_ids"])
    featurizer_started = time.perf_counter()
    featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
        context.arrow_paths,
        expected_normalization_version=NORMALIZATION_VERSION,
        signature_ids=signature_ids,
        name_tuples=None,
        load_name_counts=True,
        preprocess=True,
        num_threads=max(1, int(n_jobs)),
    )
    featurizer_seconds = float(time.perf_counter() - featurizer_started)
    signature_id_to_index = _signature_id_to_index(featurizer)
    batch, row_signals = _arrow_labeled_plan_to_batch_and_row_signals(
        plan=raw_plan,
        rows=dataset_rows,
        signature_id_to_index=signature_id_to_index,
        row_group_ids=group_codes,
    )
    row_seed_bypass = _arrow_row_seed_bypass_mask(
        dataset_rows,
        context.component_members,
        cluster_seeds_require=context.cluster_seeds_require,
        cluster_seeds_disallow=context.cluster_seeds_disallow,
        seed_constrained_signature_ids=context.seed_constrained_signature_ids,
    )
    row_ignore_disallow = np.asarray(
        [_row_label_is_positive(row) for row in dataset_rows.itertuples(index=False)],
        dtype=bool,
    )
    pair_row_indices = np.asarray(batch.pair_row_indices, dtype=np.uint32)
    pair_labels, constraint_summary = _resolve_arrow_rust_pair_labels(
        clusterer=clusterer,
        batch=batch,
        featurizer=featurizer,
        n_jobs=n_jobs,
        pair_seed_bypass=row_seed_bypass[pair_row_indices],
        pair_ignore_disallow=(row_seed_bypass | row_ignore_disallow)[pair_row_indices],
    )
    fused_pairwise_started = time.perf_counter()
    fused_pairwise = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        None,
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
        runtime_context=context.runtime_context,
        featurizer=featurizer,
    )
    fused_pairwise_seconds = float(time.perf_counter() - fused_pairwise_started)
    overlap = sorted(set(row_signals) & set(fused_pairwise.row_signals))
    if overlap:
        raise ValueError(f"raw Arrow row signals overlap fused pairwise signals: {overlap}")
    row_signals.update(fused_pairwise.row_signals)
    _validate_row_signals(row_signals)

    non_pairwise_started = time.perf_counter()
    non_pairwise_features = {
        str(column): np.asarray(values, dtype=np.float32)
        for column, values in build_promoted_non_pairwise_row_features(batch, row_signals).items()
    }
    non_pairwise_seconds = float(time.perf_counter() - non_pairwise_started)
    feature_values = _assemble_promoted_feature_values(
        target_features=target_features,
        non_pairwise_features=non_pairwise_features,
        pairwise_stats=fused_pairwise.pairwise_stats,
    )
    raw_plan_telemetry = dict(raw_plan.get("telemetry", {}) or {})
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
        "full_summary_cache_size": 0,
        "residual_summary_cache_size": 0,
        "retrieval_policy": rust_module.DEFAULT_HYBRID_CENTROID_POLICY_NAME,
        "retrieval_max_block_component_size": int(context.max_block_component_size),
        "specter_embeddings": int(raw_plan_telemetry.get("specter_count", 0) or 0),
        "pairwise_model_nan_value": "nan"
        if math.isnan(float(pairwise_model_nan_value))
        else float(pairwise_model_nan_value),
        "pairwise_aggregate_nan_value": (
            "nan" if math.isnan(float(pairwise_aggregate_nan_value)) else float(pairwise_aggregate_nan_value)
        ),
        "row_nan_policy": PRODUCTION_ROW_NAN_POLICY,
        **constraint_summary,
        "raw_arrow_labeled_plan_seconds": round(raw_plan_seconds, 3),
        "raw_arrow_featurizer_seconds": round(featurizer_seconds, 3),
        "fused_pairwise_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_featurize_seconds": round(float(fused_pairwise.telemetry["feature_seconds"]), 3),
        "pairwise_model_predict_seconds": round(float(fused_pairwise.telemetry["predict_seconds"]), 3),
        "non_pairwise_formula_seconds": round(non_pairwise_seconds, 3),
        "rust_pairwise_aggregate_seconds": 0.0,
        "raw_arrow_labeled_plan_telemetry": raw_plan_telemetry,
        "seconds": round(float(time.perf_counter() - started), 3),
    }
    del pair_labels, fused_pairwise, featurizer
    gc.collect()
    return feature_values, summary


def _safe_dataset_filename(dataset_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(dataset_name))


def _write_arrow_rust_partial(
    *,
    shard: ArrowRustPendingShard,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> None:
    _write_arrow_rust_partial_frame(
        rows=shard.rows,
        row_positions=shard.row_positions,
        partial_path=shard.partial_path,
        dataset_features=dataset_features,
        target_features=target_features,
    )


def _write_arrow_rust_partial_frame(
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


def _finalize_arrow_rust_table_plan(
    *,
    plan: ArrowRustTablePlan,
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
        "label_filtering": plan.label_filtering_summary,
        "structural_cleaning": plan.structural_cleaning_summary,
        "seconds": round(float(time.perf_counter() - plan.started), 3),
        "mode": "arrow-rust",
    }


def _finalize_arrow_rust_bundle_metadata(
    *,
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    selected_keys: Sequence[str],
) -> OfficialBundle:
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    feature_count = int(target["feature_count"])
    tree_count = int(target["params"]["n_estimators"])
    payload["bundle_name"] = (
        f"{payload['bundle_name']}_arrow_rust_block_local_promoted_{feature_count}_{tree_count}trees"
    )
    assets = payload.setdefault("assets", {})
    if not isinstance(assets, dict):
        raise ValueError("bundle assets must be an object")
    corrected_feature_rows = assets.setdefault(
        "corrected_feature_rows",
        {
            "root": "features_corrected",
            "files": {},
        },
    )
    if not isinstance(corrected_feature_rows, dict):
        raise ValueError("assets.corrected_feature_rows must be an object")
    corrected_feature_rows.setdefault("root", "features_corrected")
    corrected_feature_files = corrected_feature_rows.setdefault("files", {})
    if not isinstance(corrected_feature_files, dict):
        raise ValueError("assets.corrected_feature_rows.files must be an object")
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("bundle models must be an object")
    classic_model = models.setdefault("classic", {})
    if not isinstance(classic_model, dict):
        raise ValueError("models.classic must be an object")
    extra_eval_paths = classic_model.setdefault("extra_eval_paths", {})
    if not isinstance(extra_eval_paths, dict):
        raise ValueError("models.classic.extra_eval_paths must be an object")
    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        relpath = _output_table_relpath(table_key, labels_path).as_posix()
        corrected_feature_files[table_key] = relpath
        if table_key.startswith("extra_eval_paths."):
            dataset_name = table_key.split(".", 1)[1]
            extra_eval_paths[dataset_name] = relpath
        else:
            classic_model[table_key] = relpath
    classic_model["feature_columns"] = list(target["features"])
    classic_model["best_params"] = dict(target["params"])
    payload["expected_metrics"] = {"classic": _target_expected_metrics(target)}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _bundle_with_promoted_target(load_bundle(output_bundle_root), target)


def _materialize_arrow_rust_feature_bundle(
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
    max_exemplars: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    name_counts_index_root: Path | None = None,
    prevalidated_arrow_paths: Mapping[str, ValidatedArrowInputs] | None = None,
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    _copy_bundle_support_files(source_bundle, output_bundle_root)
    table_key_set = set(table_keys) if table_keys is not None else None
    selected_keys = [
        table_key
        for table_key in _source_featureless_table_keys(source_bundle)
        if table_key_set is None or table_key in table_key_set
    ]
    materialized_keys: list[str] = []
    summaries: list[dict[str, Any]] = []
    target_features = tuple(str(feature) for feature in target["features"])
    table_plans: dict[str, ArrowRustTablePlan] = {}
    table_plan_order: list[str] = []
    pending_by_dataset: dict[str, list[ArrowRustPendingShard]] = {}
    component_membership_cache: dict[str, pd.DataFrame] = {}
    arrow_paths_cache: dict[str, ValidatedArrowInputs] = dict(prevalidated_arrow_paths or {})

    def append_empty_selection_summary(
        *,
        table_key: str,
        labels_path: Path,
        output_path: Path,
        label_filtering_summary: dict[str, Any],
        structural_cleaning_summary: dict[str, Any],
    ) -> None:
        summary = {
            "table_key": table_key,
            "labels_path": str(labels_path.relative_to(source_bundle.root)),
            "output_path": str(output_path),
            "rows": 0,
            "datasets": [],
            "seconds": 0.0,
            "mode": "arrow-rust",
            "skipped": "empty_selection",
            "label_filtering": label_filtering_summary,
            "structural_cleaning": structural_cleaning_summary,
        }
        summaries.append(summary)
        print(json.dumps({"event": "arrow_rust_table_featureization_skipped", **summary}), flush=True)

    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        output_relpath = _output_table_relpath(table_key, labels_path)
        output_path = output_bundle_root / output_relpath
        print(
            json.dumps(
                {
                    "event": "arrow_rust_table_featureization_start",
                    "table_key": table_key,
                    "output_path": str(output_path),
                }
            ),
            flush=True,
        )
        labels = _read_selected_parquet_rows(
            labels_path,
            datasets=datasets,
            limit_rows=limit_rows,
        )
        labels, label_filtering_summary = _drop_unlabeled_singleton_orcid_rows(
            labels,
            context=f"arrow-rust:{table_key}",
        )
        if labels.empty:
            append_empty_selection_summary(
                table_key=table_key,
                labels_path=labels_path,
                output_path=output_path,
                label_filtering_summary=label_filtering_summary,
                structural_cleaning_summary={
                    "rows_before": 0,
                    "rows_after": 0,
                    "rows_removed": 0,
                    "skipped": "empty_selection",
                },
            )
            continue
        input_dataset_names = tuple(dict.fromkeys(labels["dataset"].astype(str)))
        for dataset_name in input_dataset_names:
            if dataset_name not in arrow_paths_cache:
                arrow_paths_cache[dataset_name] = _arrow_paths_for_dataset(
                    source_bundle,
                    dataset_name,
                    name_counts_index_root=name_counts_index_root,
                )
        labels, structural_cleaning_summary = _clean_arrow_rust_structural_rows(
            source_bundle=source_bundle,
            table_key=table_key,
            rows=labels,
            component_membership_cache=component_membership_cache,
            name_counts_index_root=name_counts_index_root,
            arrow_paths_cache=arrow_paths_cache,
        )
        if labels.empty:
            append_empty_selection_summary(
                table_key=table_key,
                labels_path=labels_path,
                output_path=output_path,
                label_filtering_summary=label_filtering_summary,
                structural_cleaning_summary=structural_cleaning_summary,
            )
            continue

        partial_dir = output_path.parent / "_partial" / output_path.stem
        partial_dir.mkdir(parents=True)
        plan = ArrowRustTablePlan(
            table_key=table_key,
            labels_path=labels_path,
            output_path=output_path,
            labels=labels,
            partial_paths=[],
            dataset_summaries=[],
            label_filtering_summary=label_filtering_summary,
            structural_cleaning_summary=structural_cleaning_summary,
            started=time.perf_counter(),
        )
        table_plans[table_key] = plan
        table_plan_order.append(table_key)
        materialized_keys.append(table_key)
        for dataset_name, dataset_rows in labels.groupby(labels["dataset"].astype(str), sort=False):
            dataset_name = str(dataset_name)
            row_positions = dataset_rows.index.to_numpy(dtype=np.int64)
            partial_path = partial_dir / f"{_safe_dataset_filename(dataset_name)}.parquet"
            pending_by_dataset.setdefault(dataset_name, []).append(
                ArrowRustPendingShard(
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
                    "event": "arrow_rust_dataset_context_start",
                    "mode": "arrow-rust",
                    "dataset": dataset_name,
                    "shards": len(shards),
                    "rows": int(sum(len(shard.rows) for shard in shards)),
                    "tables": sorted({shard.table_key for shard in shards}),
                }
            ),
            flush=True,
        )
        context = _build_arrow_rust_dataset_context(
            source_bundle=source_bundle,
            dataset_name=dataset_name,
            name_counts_index_root=name_counts_index_root,
            arrow_paths=arrow_paths_cache[dataset_name],
        )
        try:
            for shard in shards:
                print(
                    json.dumps(
                        {
                            "event": "arrow_rust_dataset_featureization_start",
                            "mode": "arrow-rust",
                            "table_key": shard.table_key,
                            "dataset": shard.dataset_name,
                            "rows": int(len(shard.rows)),
                        }
                    ),
                    flush=True,
                )
                dataset_features, dataset_summary = _materialize_arrow_rust_dataset_rows(
                    context=context,
                    rows=shard.rows,
                    target_features=target_features,
                    clusterer=clusterer,
                    n_jobs=n_jobs,
                    total_ram_bytes=total_ram_bytes,
                    max_exemplars=max_exemplars,
                    pairwise_model_nan_value=float(pairwise_model_nan_value),
                    pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                )
                _write_arrow_rust_partial(
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
                            "event": "arrow_rust_dataset_featureization_done",
                            "mode": "arrow-rust",
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
            _release_arrow_rust_dataset_context(context)
            del context

    for table_key in table_plan_order:
        summary = _finalize_arrow_rust_table_plan(
            plan=table_plans[table_key],
            target_features=target_features,
            source_bundle=source_bundle,
        )
        summaries.append(summary)
        print(json.dumps({"event": "arrow_rust_table_featureization_done", **summary}), flush=True)

    _write_json(output_bundle_root / "featureization_summary.json", summaries)
    return (
        _finalize_arrow_rust_bundle_metadata(
            source_bundle=source_bundle,
            output_bundle_root=output_bundle_root,
            target=target,
            selected_keys=materialized_keys,
        ),
        summaries,
    )


def _save_evaluated_artifact(
    *,
    fitted: FittedClassicRun,
    artifact_dir: Path,
    target_spec: Mapping[str, Any],
    artifact_pairwise_bundle_binding: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_metadata = save_incremental_linking_artifact(
        fitted.model,
        artifact_dir,
        retrieval_top_k=fitted.retrieval_top_k,
        gate_config=fitted.gate_config,
        target_spec=target_spec,
        pairwise_bundle_binding=artifact_pairwise_bundle_binding,
    )
    loaded = load_incremental_linking_artifact(artifact_dir)
    return {
        "path": str(artifact_dir),
        "schema_version": artifact_metadata["schema_version"],
        "booster_sha256": artifact_metadata["booster_sha256"],
        "target_spec_digest": artifact_metadata["target_spec_digest"],
        "retrieval_top_k": loaded.retrieval_top_k,
    }


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
    target_metrics = target.get("metrics")
    if not isinstance(target_metrics, Mapping):
        raise RuntimeError("Official promoted target must contain a metrics object")
    if not target_metrics:
        raise RuntimeError("Official promoted target metrics must not be empty")
    observed_keys = set(observed)
    if observed_keys != SUPPORTED_OFFICIAL_METRIC_KEYS:
        raise RuntimeError(
            "Official promoted run produced an unexpected metric key set: "
            f"missing={sorted(SUPPORTED_OFFICIAL_METRIC_KEYS - observed_keys)} "
            f"extra={sorted(observed_keys - SUPPORTED_OFFICIAL_METRIC_KEYS)}"
        )
    target_keys = set(target_metrics)
    if target_keys != SUPPORTED_OFFICIAL_METRIC_KEYS:
        raise RuntimeError(
            "Official promoted target must contain the complete official metric set: "
            f"missing={sorted(SUPPORTED_OFFICIAL_METRIC_KEYS - target_keys)} "
            f"extra={sorted(target_keys - SUPPORTED_OFFICIAL_METRIC_KEYS)}"
        )

    for key in SUPPORTED_OFFICIAL_METRIC_KEYS:
        _require_finite_metric_values(
            observed[key],
            path=str(key),
            context="Official promoted metric",
            error_type=RuntimeError,
        )
        _require_finite_metric_values(
            target_metrics[key],
            path=f"target.{key}",
            context="Official promoted metric",
            error_type=RuntimeError,
        )
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


def _resolved_output_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    """Validate fresh output targets without creating them."""

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"--output-dir must name a new directory: {output_dir}")
    publish_to = getattr(args, "publish_to", None)
    publish_dir = publish_to.resolve() if publish_to is not None else None
    if publish_dir is not None:
        if publish_dir.exists():
            raise SystemExit(f"--publish-to must name a new directory: {publish_dir}")
        if publish_dir.is_relative_to(output_dir) or output_dir.is_relative_to(publish_dir):
            raise SystemExit("--output-dir and --publish-to must be separate directories")
    return output_dir, publish_dir, output_dir / EVALUATED_ARTIFACT_DIRNAME


def _assert_output_paths_outside_inputs(
    *,
    output_dir: Path,
    publish_dir: Path | None,
    source_bundle_root: Path,
    pairwise_model_path: Path,
) -> None:
    """Reject any write target nested beneath an immutable input bundle."""

    input_roots = {
        "--source-bundle-root": source_bundle_root.resolve(),
        "--pairwise-model-path": pairwise_model_path.resolve(),
    }
    output_paths = {
        "--output-dir": output_dir.resolve(),
        "--publish-to": None if publish_dir is None else publish_dir.resolve(),
    }
    collisions = [
        f"{output_option}={output_path} under {input_option}={input_root}"
        for output_option, output_path in output_paths.items()
        if output_path is not None
        for input_option, input_root in input_roots.items()
        if output_path.is_relative_to(input_root)
    ]
    if collisions:
        raise SystemExit("linker outputs must be outside immutable input bundles: " + "; ".join(collisions))


def _validate_source_bundle_support_files(
    source_bundle: OfficialBundle,
    *,
    require_training_contract: bool = False,
) -> list[str]:
    """Validate support files copied before feature materialization starts."""

    required_files = [source_bundle.root / "bundle.json"]
    splits_dir = source_bundle.root / "splits"
    if not splits_dir.is_dir():
        raise ValueError(f"source bundle is missing splits directory: {splits_dir}")
    split_files = sorted(path for path in splits_dir.rglob("*") if path.is_file())
    if not split_files:
        raise ValueError(f"source bundle splits directory contains no files: {splits_dir}")
    required_files.extend(split_files)

    assignments_path: Path | None = None
    internal_eval_path: Path | None = None
    gate_spec: dict[str, Any] | None = None
    classic = source_bundle.models.get("classic", {})
    if require_training_contract and not isinstance(classic, Mapping):
        raise ValueError("source bundle models.classic must be a mapping")
    if isinstance(classic, Mapping):
        direct_support_path = classic.get("classic_gate_internal_eval_base_groups_path")
        if isinstance(direct_support_path, str) and direct_support_path:
            internal_eval_path = _resolve_path(source_bundle, direct_support_path)
            required_files.append(internal_eval_path)
        split_spec = classic.get("stratified_eval_test_split")
        if isinstance(split_spec, Mapping):
            assignments_path_value = split_spec.get("assignments_path")
            if isinstance(assignments_path_value, str) and assignments_path_value:
                assignments_path = _resolve_path(source_bundle, assignments_path_value)
                required_files.append(assignments_path)
        gate_spec = _promoted_stratified_gate_spec(dict(classic))
        if require_training_contract:
            if internal_eval_path is None:
                raise ValueError("classic.classic_gate_internal_eval_base_groups_path is required")
            if not isinstance(split_spec, Mapping) or assignments_path is None:
                raise ValueError("classic.stratified_eval_test_split.assignments_path is required")
            if gate_spec is None:
                raise ValueError("classic.promoted_stratified_gate is required")

    missing = sorted(str(path) for path in required_files if not path.is_file())
    if missing:
        raise ValueError(f"source bundle is missing required support files: {missing}")
    if assignments_path is not None:
        if require_training_contract:
            _, assignments = _load_classic_stratified_eval_rows(
                source_bundle,
                dict(classic),
                dict(split_spec),
            )
        else:
            assignments = pd.read_csv(assignments_path)
            _validate_stratified_split_assignments(assignments)
        if gate_spec is not None:
            required_splits = {*gate_spec["calibration_splits"], str(gate_spec["test_split"])}
            missing_splits = sorted(required_splits - set(assignments["split"].astype(str)))
            if missing_splits:
                raise ValueError(
                    f"Stratified split assignments omit configured calibration/test splits: {missing_splits}"
                )
    return [str(path) for path in dict.fromkeys(required_files)]


def _require_publish_version_matches_pairwise(clusterer: Any, publish_dir: Path | None) -> None:
    """Reject an accidental release-version change before materialization."""

    if publish_dir is None:
        return
    publish_version = production_version_from_bundle_dir(publish_dir)
    pairwise_version = str(getattr(clusterer, "production_model_bundle_version", "") or "").strip()
    if publish_version is None:
        raise SystemExit("--publish-to must use a production_model_vX.Y directory name")
    if not pairwise_version:
        raise ValueError("Pairwise bundle does not declare production_model_bundle_version")
    if publish_version != pairwise_version:
        raise SystemExit(
            "--publish-to version disagrees with the pairwise bundle: "
            f"publish={publish_version!r}, pairwise={pairwise_version!r}"
        )


def _preflight_source_rows(
    source_bundle: OfficialBundle,
    *,
    table_keys: Sequence[str] | None,
    datasets: set[str] | None,
    limit_rows: int | None,
    require_full_tables: bool,
    name_counts_index_root: Path | None,
) -> tuple[dict[str, Any], dict[str, ValidatedArrowInputs]]:
    """Validate selectors, row presence, and every referenced Arrow generation."""

    available_tables = _source_featureless_table_keys(source_bundle)
    if not available_tables:
        raise ValueError("source bundle contains no supported featureless row tables")
    available_table_set = set(available_tables)
    if require_full_tables:
        missing_required = sorted(set(REQUIRED_TABLE_KEYS) - available_table_set)
        if missing_required:
            raise ValueError(f"official linker source bundle is missing required tables: {missing_required}")

    requested_tables = tuple(table_keys) if table_keys is not None else available_tables
    unknown_tables = sorted(set(requested_tables) - available_table_set)
    if unknown_tables:
        raise ValueError(f"unknown --tables selectors: {unknown_tables}; available={list(available_tables)}")
    if not requested_tables:
        raise ValueError("no source tables were selected")

    table_summaries: list[dict[str, Any]] = []
    observed_selector_matches: set[str] = set()
    observed_selected_datasets: set[str] = set()
    for table_key in requested_tables:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        try:
            selected_dataset_rows, source_rows, available_datasets = _scan_parquet_rows(
                labels_path,
                datasets=datasets,
                limit_rows=limit_rows,
                columns=["dataset"],
                inventory=True,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"source table {table_key!r} must contain a dataset column: {labels_path}") from exc
        if source_rows == 0:
            raise ValueError(f"source table {table_key!r} is empty: {labels_path}")
        if datasets is None:
            observed_selector_matches.update(available_datasets)
        else:
            observed_selector_matches.update(available_datasets & datasets)
        selected_rows = int(len(selected_dataset_rows))
        if selected_rows == 0:
            requested = "all datasets" if datasets is None else sorted(datasets)
            raise ValueError(f"source table {table_key!r} selected zero rows for datasets={requested}")
        selected_dataset_names = tuple(dict.fromkeys(selected_dataset_rows["dataset"].astype(str).tolist()))
        observed_selected_datasets.update(selected_dataset_names)
        table_summaries.append(
            {
                "table_key": table_key,
                "path": str(labels_path),
                "source_rows": source_rows,
                "selected_rows": selected_rows,
                "datasets": list(selected_dataset_names),
            }
        )

    if datasets is not None:
        unmatched_datasets = sorted(datasets - observed_selector_matches)
        if unmatched_datasets:
            raise ValueError(
                f"unknown or unmatched --datasets selectors: {unmatched_datasets}; "
                f"matched={sorted(observed_selected_datasets)}"
            )

    arrow_paths_by_dataset: dict[str, ValidatedArrowInputs] = {}
    for dataset_name in sorted(observed_selected_datasets):
        arrow_paths_by_dataset[dataset_name] = _arrow_paths_for_dataset(
            source_bundle,
            dataset_name,
            name_counts_index_root=name_counts_index_root,
        )

    name_count_hashes = {
        paths.name_counts_manifest.manifest_sha256
        for paths in arrow_paths_by_dataset.values()
        if paths.name_counts_manifest is not None
    }
    if len(name_count_hashes) > 1:
        raise ValueError(
            f"linker source datasets reference multiple name-count generations: {sorted(name_count_hashes)}"
        )
    summary = {
        "available_tables": list(available_tables),
        "selected_tables": list(requested_tables),
        "tables": table_summaries,
        "datasets": {
            dataset_name: {
                "generation_id": paths.generation_id,
                "normalization_version": paths.normalization_version,
                "name_counts_manifest_sha256": (
                    None if paths.name_counts_manifest is None else paths.name_counts_manifest.manifest_sha256
                ),
            }
            for dataset_name, paths in arrow_paths_by_dataset.items()
        },
        "total_selected_rows": sum(int(item["selected_rows"]) for item in table_summaries),
    }
    return summary, arrow_paths_by_dataset


def _assert_materialization_nonempty(
    summaries: Sequence[Mapping[str, Any]],
    *,
    require_full_tables: bool,
) -> None:
    """Reject successful-looking no-op or incomplete materialization."""

    rows_by_table = {str(item["table_key"]): int(item.get("rows", 0)) for item in summaries}
    total_rows = sum(rows_by_table.values())
    if total_rows <= 0:
        raise RuntimeError("linker feature materialization produced zero rows")
    if require_full_tables:
        missing_or_empty = sorted(key for key in REQUIRED_TABLE_KEYS if rows_by_table.get(key, 0) <= 0)
        if missing_or_empty:
            raise RuntimeError(
                f"official linker materialization has missing or empty required tables: {missing_or_empty}"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    command = str(args.command)
    is_preflight = command == "preflight"
    is_materialize = command == "materialize"
    is_candidate = command == "candidate"
    output_dir, publish_dir, artifact_dir = _resolved_output_paths(args)
    target = _load_target(args.target_json)
    selected_subset = args.limit_rows is not None or bool(args.tables) or bool(args.datasets)
    require_full_tables = command in {"candidate", "publish"} or (is_preflight and not selected_subset)
    if publish_dir is not None and not target["metrics"]:
        raise ValueError("Published target metrics must not be empty")
    pairwise_model_nan_value = float("nan")
    pairwise_aggregate_nan_value = 0.0
    feature_nan_policy = _feature_nan_policy_summary()
    source_bundle = load_bundle(args.source_bundle_root)
    _assert_output_paths_outside_inputs(
        output_dir=output_dir,
        publish_dir=publish_dir,
        source_bundle_root=source_bundle.root,
        pairwise_model_path=Path(args.pairwise_model_path),
    )
    support_files = _validate_source_bundle_support_files(
        source_bundle,
        require_training_contract=require_full_tables,
    )
    name_counts_index_root = (
        Path(args.arrow_name_counts_index_root) if args.arrow_name_counts_index_root is not None else None
    )
    preflight_summary, prevalidated_arrow_paths = _preflight_source_rows(
        source_bundle,
        table_keys=_parse_tables(args.tables),
        datasets=_parse_datasets(args.datasets),
        limit_rows=args.limit_rows,
        require_full_tables=require_full_tables,
        name_counts_index_root=name_counts_index_root,
    )
    clusterer = load_clusterer(args.pairwise_model_path, n_jobs=int(args.n_jobs))
    _assert_pairwise_model_supports_arrow_materialization(clusterer, args.pairwise_model_path)
    _require_publish_version_matches_pairwise(clusterer, publish_dir)
    for dataset_name, arrow_paths in prevalidated_arrow_paths.items():
        require_arrow_name_counts_index_for_clusterer(
            clusterer,
            arrow_paths,
            context=f"linker train/calibrate/eval preflight dataset {dataset_name!r}",
        )
    pairwise_binding = dict(pairwise_bundle_binding(Path(args.pairwise_model_path)))
    preflight_result = {
        "mode": "preflight",
        "source_bundle_root": str(source_bundle.root),
        "target_json": str(Path(args.target_json).resolve()),
        "pairwise_model_path": str(Path(args.pairwise_model_path).resolve()),
        "pairwise_bundle_binding": pairwise_binding,
        "output_dir": str(output_dir),
        "feature_nan_policy": feature_nan_policy,
        "max_exemplars": PRODUCTION_MAX_EXEMPLARS,
        "source": preflight_summary,
        "support_files": support_files,
    }
    if is_preflight:
        return preflight_result

    output_dir.mkdir(parents=True)
    _write_json(output_dir / "preflight.json", preflight_result)
    feature_bundle_root = output_dir / "arrow_rust_feature_bundle"
    feature_bundle, featureization_summaries = _materialize_arrow_rust_feature_bundle(
        source_bundle=source_bundle,
        output_bundle_root=feature_bundle_root,
        target=target,
        clusterer=clusterer,
        n_jobs=int(args.n_jobs),
        total_ram_bytes=int(args.total_ram_bytes),
        table_keys=_parse_tables(args.tables),
        datasets=_parse_datasets(args.datasets),
        limit_rows=args.limit_rows,
        max_exemplars=PRODUCTION_MAX_EXEMPLARS,
        pairwise_model_nan_value=pairwise_model_nan_value,
        pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
        name_counts_index_root=name_counts_index_root,
        prevalidated_arrow_paths=prevalidated_arrow_paths,
    )
    _assert_materialization_nonempty(featureization_summaries, require_full_tables=require_full_tables)
    if is_materialize:
        result = {
            "mode": "arrow-rust",
            "source_bundle_root": str(source_bundle.root),
            "feature_bundle_root": str(feature_bundle.root),
            "pairwise_model_path": str(args.pairwise_model_path),
            "feature_count": int(target["feature_count"]),
            "component_scope": "block-local",
            "feature_nan_policy": feature_nan_policy,
            "featureization": featureization_summaries,
        }
        _write_json(output_dir / "run_summary.json", result)
        return result

    run_output_dir = output_dir / "classic"
    started = time.perf_counter()
    active_params = dict(feature_bundle.models["classic"]["best_params"])

    fitted = run_classic(feature_bundle, run_output_dir, n_jobs=int(args.n_jobs))
    summary = fitted.summary
    observed = _observed_official_metrics(summary)
    deltas = _metric_deltas(observed, target)
    candidate_target_path = None
    if not is_candidate:
        _assert_no_metric_drift(observed, target)
    if is_candidate:
        candidate_target_path = output_dir / "candidate_target.json"
        candidate_target = {
            **target,
            "params": dict(active_params),
            "metrics": observed,
        }
        _write_json(candidate_target_path, candidate_target)

    artifact_summary = _save_evaluated_artifact(
        fitted=fitted,
        artifact_dir=artifact_dir,
        target_spec=candidate_target if is_candidate else target,
        artifact_pairwise_bundle_binding=pairwise_binding,
    )
    _write_json(output_dir / "artifact_summary.json", artifact_summary)
    production_bundle_summary = None
    if publish_dir is not None:
        production_bundle_summary = finalize_production_bundle(
            pairwise_bundle_dir=Path(args.pairwise_model_path),
            output_bundle_dir=publish_dir,
            incremental_linker_artifact_dir=artifact_dir,
            target_json=Path(args.target_json),
        )
    result = {
        "mode": "arrow-rust",
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
        "query_predictions": fitted.query_predictions,
        "classic_summary_path": str(run_output_dir / "summary.json"),
        "feature_nan_policy": feature_nan_policy,
    }
    result["artifact_dir"] = str(artifact_dir)
    result["artifact_summary"] = artifact_summary
    if production_bundle_summary is not None:
        result["production_bundle_dir"] = str(production_bundle_summary.bundle_dir)
        result["production_bundle_summary"] = {
            "bundle_status": production_bundle_summary.bundle_status,
            "bundle_version": production_bundle_summary.bundle_version,
            "files": list(production_bundle_summary.files),
            "manifest_path": str(production_bundle_summary.manifest_path),
        }
    if candidate_target_path is not None:
        result["candidate_target_path"] = str(candidate_target_path)
    result["component_scope"] = "block-local"
    if not is_candidate:
        result["metric_drift_check"] = "passed"
    _write_json(output_dir / "run_summary.json", result)
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-bundle-root", type=Path, required=True)
    parser.add_argument("--target-json", type=Path, required=True, help="Explicit promoted-linker target JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pairwise-model-path",
        type=Path,
        required=True,
        help="Explicit v5 pairwise_only native bundle produced by train_pairwise.py.",
    )
    parser.add_argument(
        "--arrow-name-counts-index-root",
        type=Path,
        default=DEFAULT_NAME_COUNTS_INDEX_ROOT,
        help=(
            "Optional override for the Arrow name_counts_index root. By default, each Arrow dataset "
            "manifest is the authority."
        ),
    )
    parser.add_argument("--n-jobs", type=_positive_int, default=20)
    parser.add_argument("--total-ram-bytes", type=_positive_int, default=DEFAULT_TOTAL_RAM_BYTES)


def _add_selectors(parser: argparse.ArgumentParser, *, require_limit: bool) -> None:
    parser.add_argument("--tables", nargs="+", help="Optional table keys to materialize.")
    parser.add_argument("--datasets", nargs="+", help="Optional dataset slugs to keep when materializing smoke checks.")
    parser.add_argument(
        "--limit-rows",
        type=_positive_int,
        required=require_limit,
        help="Per-table row limit for bounded checks.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="Validate inputs and write nothing.")
    _add_common_arguments(preflight)
    _add_selectors(preflight, require_limit=False)
    preflight.add_argument("--publish-to", type=Path, help="Fresh destination to validate without writing.")

    materialize = commands.add_parser("materialize", help="Run bounded Arrow/Rust materialization.")
    _add_common_arguments(materialize)
    _add_selectors(materialize, require_limit=True)

    candidate = commands.add_parser("candidate", help="Train and evaluate one release candidate.")
    _add_common_arguments(candidate)
    candidate.set_defaults(tables=None, datasets=None, limit_rows=None, publish_to=None)

    publish = commands.add_parser("publish", help="Train against a frozen target and publish the matching bundle.")
    _add_common_arguments(publish)
    publish.add_argument(
        "--publish-to",
        type=Path,
        required=True,
        help="Publish a complete production_model_vX.Y bundle to this new directory.",
    )
    publish.set_defaults(tables=None, datasets=None, limit_rows=None)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
