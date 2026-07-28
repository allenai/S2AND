"""Train the linker once, publish a complete bundle, reload, and evaluate."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc

REPO_ROOT = Path(__file__).resolve().parents[3]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from s2and import feature_port  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER  # noqa: E402
from s2and.incremental_linking.array_validation import as_retrieval_rank_uint16_1d  # noqa: E402
from s2and.incremental_linking.artifact import save_incremental_linking_artifact  # noqa: E402
from s2and.incremental_linking.feature_block import (  # noqa: E402
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
)
from s2and.incremental_linking.features import promoted_linker_feature_columns  # noqa: E402
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view  # noqa: E402
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch  # noqa: E402
from s2and.incremental_linking.policy import (  # noqa: E402
    PROMOTED_LINKER_MODEL_SUPPRESS_ORCID,
    require_arrow_name_counts_index_for_clusterer,
)
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS  # noqa: E402
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features  # noqa: E402
from s2and.incremental_linking.runtime import compute_candidate_batch_pairwise_model_and_aggregate_stats  # noqa: E402
from s2and.incremental_linking_training.classic import (  # noqa: E402
    DEFAULT_RETRIEVAL_TOP_K,
    PROMOTED_NON_PAIRWISE_COLUMNS,
    PROMOTED_PAIRWISE_COLUMNS,
    SUPPORTED_PROMOTED_FEATURE_COLUMNS,
    WEIGHTED_ERROR_WEIGHTS,
    OfficialBundle,
    _classic_stratified_eval_source_specs,
    _drop_unlabeled_singleton_orcid_rows,
    _filter_candidate_rows_to_retrieval_top_k,
    _promoted_stratified_gate_spec,
    _resolve_path,
    _validate_stratified_split_assignments,
    evaluate_classic,
    fit_classic,
    load_bundle,
)
from s2and.incremental_linking_training.data_loading import load_clusterer  # noqa: E402
from s2and.incremental_linking_training.source_bundle_preflight import (  # noqa: E402
    preflight_source_rows,
    read_source_parquet_rows,
    source_bundle_asset_file,
    source_featureless_table_keys,
    validate_source_bundle_support_files,
)
from s2and.production_bundle import finalize_production_bundle  # noqa: E402
from s2and.production_model import load_production_model, pairwise_bundle_binding  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    FLOAT_OFFICIAL_METRIC_KEYS,
    INTEGER_OFFICIAL_METRIC_KEYS,
    REQUIRED_LINKER_TABLE_KEYS,
    SUPPORTED_OFFICIAL_METRIC_KEYS,
    load_packaged_artifact_authority,
)
from s2and.runtime import build_runtime_context  # noqa: E402
from s2and.rust_calls import get_constraint_labels_index_arrays_rust  # noqa: E402

DEFAULT_TOTAL_RAM_BYTES = 48 * 1024**3
PRODUCTION_MAX_EXEMPLARS = 4
PRODUCTION_PAIRWISE_MODEL_NAN_POLICY = "preserve"
PRODUCTION_PAIRWISE_AGGREGATE_NAN_POLICY = "zero"
PRODUCTION_ROW_NAN_POLICY = "finite"
LINKER_STAGING_DIRNAME = ".incremental_linker_staging"
REQUIRED_TABLE_KEYS = REQUIRED_LINKER_TABLE_KEYS


@dataclass
class ArrowRustDatasetContext:
    """Arrow-only dataset state shared across linker row tables for one dataset."""

    dataset_name: str
    row_component_scope: str
    pairwise_component_scope: str
    runtime_context: Any
    arrow_dataset: ArrowDataset
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


def _output_table_relpath(labels_path: Path) -> Path:
    return Path("features_corrected") / labels_path.name


def _clean_arrow_rust_structural_rows(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    rows: pd.DataFrame,
    component_membership_cache: dict[str, pd.DataFrame],
    arrow_datasets: Mapping[str, ArrowDataset],
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
            arrow_datasets=arrow_datasets,
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


def _load_arrow_seed_constraints(
    bundle: OfficialBundle,
    dataset_name: str,
) -> tuple[dict[str, str], frozenset[tuple[str, str]]]:
    """Load explicit training constraints that are not part of dataset identity."""

    dataset_dir = (bundle.root / "datasets" / str(dataset_name)).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError(f"Arrow dataset manifest paths must be a mapping: {dataset_dir / 'manifest.json'}")

    def sidecar(key: str) -> Path | None:
        raw = paths.get(key)
        if raw is None:
            return None
        return _resolve_arrow_manifest_path(raw, dataset_dir=dataset_dir, bundle_root=bundle.root)

    require_path = sidecar("cluster_seeds")
    cluster_seeds_require = read_cluster_seeds_arrow(require_path) if require_path is not None else {}
    disallow_path = sidecar("cluster_seed_disallows")
    raw_disallows = read_cluster_seed_disallows_arrow(disallow_path) if disallow_path is not None else ()
    cluster_seeds_disallow = frozenset((str(left), str(right)) for left, right in raw_disallows)
    return ({str(key): str(value) for key, value in cluster_seeds_require.items()}, cluster_seeds_disallow)


def _load_arrow_signature_blocks(
    arrow_dataset: ArrowDataset,
) -> dict[str, str]:
    out: dict[str, str] = {}
    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            reader = pa_ipc.open_file(source)
            if "author_block" not in reader.schema.names:
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
    arrow_datasets: Mapping[str, ArrowDataset],
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
        signature_to_block = _load_arrow_signature_blocks(arrow_datasets[dataset_name])

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
    arrow_dataset: ArrowDataset,
) -> ArrowRustDatasetContext:
    started = time.perf_counter()
    member_path = _resolve_path(
        source_bundle,
        str(source_bundle.assets["candidate_members"]["datasets"][dataset_name]),
    )
    component_members = _component_member_ids_by_key(member_path)
    cluster_seeds_require, cluster_seeds_disallow = _load_arrow_seed_constraints(source_bundle, dataset_name)
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
                "name_counts_index": (
                    None if arrow_dataset.name_counts_index is None else str(arrow_dataset.name_counts_index.path)
                ),
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
        arrow_dataset=arrow_dataset,
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
            suppress_orcid=PROMOTED_LINKER_MODEL_SUPPRESS_ORCID,
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
            suppress_orcid=PROMOTED_LINKER_MODEL_SUPPRESS_ORCID,
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


def _rows_with_materialized_target_features(
    rows: pd.DataFrame,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> pd.DataFrame:
    """Return rows with every target feature replaced by materialized values."""

    output = rows.reset_index(drop=True).copy()
    for column in target_features:
        output[str(column)] = dataset_features[str(column)]
    return output


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


def _rows_in_current_retrieval_window(
    rows: pd.DataFrame,
    raw_plan: Mapping[str, Any],
    *,
    retrieval_top_k: int,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select rows by current native retrieval ranks and retain source positions."""

    if retrieval_top_k <= 0:
        raise ValueError(f"{context} retrieval_top_k must be positive")
    current_ranks = as_retrieval_rank_uint16_1d(
        "retrieval_ranks",
        _row_signal_from_plan(raw_plan, "retrieval_ranks", object, len(rows)),
    )
    selected_positions = np.flatnonzero(current_ranks <= retrieval_top_k).astype(np.int64, copy=False)
    selected_rows = rows.iloc[selected_positions].reset_index(drop=True).copy()
    selected_rows["retrieval_rank"] = current_ranks[selected_positions].astype(np.int64, copy=False)
    return selected_rows, selected_positions


def _materialize_arrow_rust_dataset_rows(
    *,
    context: ArrowRustDatasetContext,
    rows: pd.DataFrame,
    target_features: Sequence[str],
    name_tuples: frozenset[tuple[str, str]],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    max_exemplars: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    dataset_name = context.dataset_name
    source_rows = rows.reset_index(drop=True).copy()
    source_row_count = len(source_rows)
    rust_module = feature_port._require_rust_runtime()  # noqa: SLF001
    plan_fn = rust_module.raw_arrow_labeled_candidate_plan

    def build_raw_plan(candidate_rows: pd.DataFrame) -> Mapping[str, Any]:
        stored_ranks = as_retrieval_rank_uint16_1d(
            "retrieval_rank",
            pd.to_numeric(candidate_rows["retrieval_rank"], errors="raise").to_numpy(),
        )
        return plan_fn(
            context.arrow_dataset.native,
            candidate_rows["query_signature_id"].astype(str).tolist(),
            candidate_rows["query_view"].astype(str).tolist(),
            candidate_rows["query_group_id"].astype(str).tolist(),
            candidate_rows["candidate_component_key"].astype(str).tolist(),
            stored_ranks.tolist(),
            context.component_members,
            orcid_enabled=not PROMOTED_LINKER_MODEL_SUPPRESS_ORCID,
            num_threads=max(1, int(n_jobs)),
            max_exemplars=int(max_exemplars),
        )

    plan_started = time.perf_counter()
    raw_plan = build_raw_plan(source_rows)
    dataset_rows, selected_row_positions = _rows_in_current_retrieval_window(
        source_rows,
        raw_plan,
        retrieval_top_k=retrieval_top_k,
        context=f"arrow-rust:{dataset_name}",
    )
    if len(dataset_rows) != source_row_count:
        raw_plan = build_raw_plan(dataset_rows)
    raw_plan_seconds = float(time.perf_counter() - plan_started)
    row_count = len(dataset_rows)
    group_codes = tuple(
        int(value) for value in pd.factorize(dataset_rows["query_group_id"].astype(str), sort=False)[0].tolist()
    )
    signature_ids = tuple(str(signature_id) for signature_id in raw_plan["signature_ids"])
    featurizer_started = time.perf_counter()
    featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
        context.arrow_dataset,
        signature_ids=signature_ids,
        name_tuples=name_tuples,
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
        "rows_before_retrieval_window": int(source_row_count),
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
    return feature_values, summary, selected_row_positions


def _safe_dataset_filename(dataset_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(dataset_name))


def _write_arrow_rust_partial(
    *,
    shard: ArrowRustPendingShard,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
    selected_row_positions: np.ndarray,
) -> None:
    _write_arrow_rust_partial_frame(
        rows=shard.rows.iloc[selected_row_positions].reset_index(drop=True),
        row_positions=shard.row_positions[selected_row_positions],
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
    partial_output = _rows_with_materialized_target_features(rows, dataset_features, target_features)
    partial_output.insert(0, "_row_position", row_positions)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output.to_parquet(partial_path, index=False)
    del partial_output


def _finalize_arrow_rust_table_plan(
    *,
    plan: ArrowRustTablePlan,
    target_features: Sequence[str],
    source_bundle: OfficialBundle,
    retrieval_top_k: int,
) -> dict[str, Any]:
    parts = [pd.read_parquet(path) for path in plan.partial_paths]
    output = pd.concat(parts, axis=0, ignore_index=True)
    output = output.sort_values("_row_position", kind="stable").drop(columns=["_row_position"]).reset_index(drop=True)
    expected_row_count = sum(int(summary["rows"]) for summary in plan.dataset_summaries)
    if len(output) != expected_row_count:
        raise ValueError(f"{plan.table_key}: materialized row count mismatch: {len(output)} != {expected_row_count}")
    selected_output = _filter_candidate_rows_to_retrieval_top_k(
        output,
        retrieval_top_k=retrieval_top_k,
        context=f"arrow-rust:{plan.table_key}",
    )
    if len(selected_output) != len(output):
        raise ValueError(f"{plan.table_key}: materialized output contains rows outside current retrieval window")
    output = selected_output
    rows_before_retrieval_window = int(len(plan.labels))
    plan.label_filtering_summary["retrieval_window"] = {
        "retrieval_top_k": retrieval_top_k,
        "rows_before": rows_before_retrieval_window,
        "rows_after": int(len(output)),
        "rows_removed": rows_before_retrieval_window - int(len(output)),
    }
    _validate_materialized_target_features(output, target_features, context=plan.table_key)
    plan.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(plan.output_path, index=False)
    output_row_count = int(len(output))
    del parts, output
    gc.collect()
    return {
        "table_key": plan.table_key,
        "labels_path": str(plan.labels_path.relative_to(source_bundle.root)),
        "output_path": str(plan.output_path),
        "rows": output_row_count,
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
        labels_path = source_bundle_asset_file(source_bundle, "featureless_rows", table_key)
        relpath = _output_table_relpath(labels_path).as_posix()
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
    name_tuples: frozenset[tuple[str, str]],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    table_keys: Sequence[str],
    query_ids_by_table: Mapping[str, set[str]] | None = None,
    max_exemplars: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    arrow_datasets: Mapping[str, ArrowDataset],
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    _copy_bundle_support_files(source_bundle, output_bundle_root)
    classic_spec = source_bundle.models.get("classic")
    if not isinstance(classic_spec, Mapping):
        raise ValueError("Source bundle models.classic must be an object")
    retrieval_top_k = int(classic_spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    selected_keys = list(dict.fromkeys(table_keys))
    materialized_keys: list[str] = []
    summaries: list[dict[str, Any]] = []
    target_features = tuple(str(feature) for feature in target["features"])
    table_plans: dict[str, ArrowRustTablePlan] = {}
    table_plan_order: list[str] = []
    pending_by_dataset: dict[str, list[ArrowRustPendingShard]] = {}
    component_membership_cache: dict[str, pd.DataFrame] = {}

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
        labels_path = source_bundle_asset_file(source_bundle, "featureless_rows", table_key)
        output_relpath = _output_table_relpath(labels_path)
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
        labels = read_source_parquet_rows(
            labels_path,
            query_ids=(None if query_ids_by_table is None else query_ids_by_table.get(table_key)),
        )
        labels, label_filtering_summary = _drop_unlabeled_singleton_orcid_rows(
            labels,
            context=f"arrow-rust:{table_key}",
        )
        if labels.empty:
            label_filtering_summary["retrieval_window"] = {
                "retrieval_top_k": retrieval_top_k,
                "rows_before": 0,
                "rows_after": 0,
                "rows_removed": 0,
            }
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
        missing_datasets = sorted(set(input_dataset_names).difference(arrow_datasets))
        if missing_datasets:
            raise KeyError(f"preflight did not open Arrow datasets: {missing_datasets}")
        labels, structural_cleaning_summary = _clean_arrow_rust_structural_rows(
            source_bundle=source_bundle,
            table_key=table_key,
            rows=labels,
            component_membership_cache=component_membership_cache,
            arrow_datasets=arrow_datasets,
        )
        if labels.empty:
            label_filtering_summary["retrieval_window"] = {
                "retrieval_top_k": retrieval_top_k,
                "rows_before": 0,
                "rows_after": 0,
                "rows_removed": 0,
            }
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
            arrow_dataset=arrow_datasets[dataset_name],
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
                dataset_features, dataset_summary, selected_row_positions = _materialize_arrow_rust_dataset_rows(
                    context=context,
                    rows=shard.rows,
                    target_features=target_features,
                    name_tuples=name_tuples,
                    clusterer=clusterer,
                    n_jobs=n_jobs,
                    total_ram_bytes=total_ram_bytes,
                    max_exemplars=max_exemplars,
                    pairwise_model_nan_value=float(pairwise_model_nan_value),
                    pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                    retrieval_top_k=retrieval_top_k,
                )
                _write_arrow_rust_partial(
                    shard=shard,
                    dataset_features=dataset_features,
                    target_features=target_features,
                    selected_row_positions=selected_row_positions,
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
            retrieval_top_k=retrieval_top_k,
        )
        summaries.append(summary)
        print(json.dumps({"event": "arrow_rust_table_featureization_done", **summary}), flush=True)

    if not materialized_keys:
        raise RuntimeError("linker feature materialization produced zero rows")
    missing_or_empty = sorted(set(selected_keys) - set(materialized_keys))
    if missing_or_empty:
        raise RuntimeError(f"official linker materialization has missing or empty required tables: {missing_or_empty}")

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


def _validate_observed_official_metrics(metrics: Mapping[str, Any]) -> None:
    """Validate the official linker metrics before publishing the report."""

    observed_keys = set(metrics) if isinstance(metrics, Mapping) else set()
    if not isinstance(metrics, Mapping) or observed_keys != SUPPORTED_OFFICIAL_METRIC_KEYS:
        raise ValueError(
            "Linker evaluation observed_metrics must contain the complete official metric set: "
            f"missing={sorted(SUPPORTED_OFFICIAL_METRIC_KEYS - observed_keys)} "
            f"extra={sorted(observed_keys - SUPPORTED_OFFICIAL_METRIC_KEYS)}"
        )

    for key in INTEGER_OFFICIAL_METRIC_KEYS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Linker evaluation observed metric {key!r} must be a nonnegative integer")
    for key in FLOAT_OFFICIAL_METRIC_KEYS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"Linker evaluation observed metric {key!r} must be finite")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Linker evaluation observed metric {key!r} must be in [0, 1]")

    training_rows = metrics["training_rows"]
    training_positive_rows = metrics["training_positive_rows"]
    queries = metrics["stratified_test_queries"]
    if training_rows <= 0 or training_positive_rows > training_rows:
        raise ValueError("Linker evaluation observed training counts are inconsistent")
    if queries <= 0 or any(
        metrics[key] > queries
        for key in (
            "stratified_test_errors",
            "stratified_test_false_abstain",
            "stratified_test_false_link",
            "stratified_test_wrong_candidate_link",
        )
    ):
        raise ValueError("Linker evaluation observed test counts are inconsistent")

    weights = metrics["weighted_average_error_weights"]
    expected_weights = dict(WEIGHTED_ERROR_WEIGHTS)
    if not isinstance(weights, Mapping) or set(weights) != set(expected_weights):
        raise ValueError("Linker evaluation observed weighted-average error weights have invalid fields")
    for key, expected in expected_weights.items():
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"Linker evaluation observed weight {key!r} must be finite")
        if float(value) <= 0:
            raise ValueError(f"Linker evaluation observed weight {key!r} must be positive")
        if float(value) != expected:
            raise ValueError(f"Linker evaluation observed weight {key!r} must equal the official value {expected}")


def _resolved_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate fresh output targets without creating them."""

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"--output-dir must name a new directory: {output_dir}")
    pairwise_model_path = Path(args.pairwise_model_path).resolve()
    return (
        output_dir,
        output_dir / pairwise_model_path.name,
    )


def _assert_output_paths_outside_inputs(
    *,
    output_dir: Path,
    complete_model_dir: Path,
    source_bundle_root: Path,
    pairwise_model_path: Path,
) -> None:
    """Reject any write target nested beneath an immutable input bundle."""

    input_roots = {
        "--source-bundle-root": source_bundle_root.resolve(),
        "--pairwise-model-path": pairwise_model_path.resolve(),
    }
    collisions = [
        f"--output-dir={output_dir.resolve()} under {input_option}={input_root}"
        for input_option, input_root in input_roots.items()
        if output_dir.resolve().is_relative_to(input_root)
    ]
    if collisions:
        raise SystemExit("linker outputs must be outside immutable input bundles: " + "; ".join(collisions))
    if complete_model_dir == pairwise_model_path.resolve():
        raise SystemExit("complete model output must not replace the calibrated pairwise input")


def _release_table_plan(
    source_bundle: OfficialBundle,
) -> tuple[tuple[str, ...], dict[str, set[str]], tuple[str, ...]]:
    """Separate train/calibration rows from frozen evaluation tables."""

    classic = dict(source_bundle.models["classic"])
    split_spec = dict(classic["stratified_eval_test_split"])
    gate_spec = _promoted_stratified_gate_spec(classic)
    if gate_spec is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    assignments = pd.read_csv(_resolve_path(source_bundle, str(split_spec["assignments_path"])))
    _validate_stratified_split_assignments(assignments)
    calibration_splits = {str(split) for split in gate_spec["calibration_splits"]}
    calibration_assignments = assignments[assignments["split"].astype(str).isin(calibration_splits)]
    if calibration_assignments.empty:
        raise ValueError("Linker calibration splits contain no assigned queries")

    table_keys = source_featureless_table_keys(source_bundle)
    path_to_table_key = {
        source_bundle_asset_file(source_bundle, "featureless_rows", table_key).resolve(): table_key
        for table_key in table_keys
    }
    calibration_query_ids: dict[str, set[str]] = {}
    for source_spec in _classic_stratified_eval_source_specs(classic):
        if source_spec["source_kind"] == "calibration_source":
            continue
        table_key = path_to_table_key.get(_resolve_path(source_bundle, source_spec["path"]).resolve())
        if table_key is None:
            raise ValueError(f"Stratified source has no featureless table: {source_spec['path']}")
        query_ids = set(
            calibration_assignments.loc[
                calibration_assignments["source_key"].astype(str).eq(str(source_spec["source_key"])),
                "query_group_id",
            ].astype(str)
        )
        if query_ids:
            calibration_query_ids[table_key] = query_ids

    training_keys = tuple(
        dict.fromkeys(
            (
                "train_path",
                "classic_gate_source_path",
                *calibration_query_ids,
            )
        )
    )
    evaluation_keys = tuple(key for key in table_keys if key not in {"train_path", "classic_gate_source_path"})
    return training_keys, calibration_query_ids, evaluation_keys


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir, complete_model_dir = _resolved_output_paths(args)
    target = _load_target(args.target_json)
    artifact_authority = load_packaged_artifact_authority(
        name_counts_index_root=Path(args.name_counts_index_root),
    )
    artifact_hashes = artifact_authority.hashes
    name_tuples = artifact_authority.name_tuples.pairs
    pairwise_model_nan_value = float("nan")
    pairwise_aggregate_nan_value = 0.0
    feature_nan_policy = _feature_nan_policy_summary()
    source_bundle = load_bundle(args.source_bundle_root)
    training_table_keys, calibration_query_ids, evaluation_table_keys = _release_table_plan(source_bundle)
    _assert_output_paths_outside_inputs(
        output_dir=output_dir,
        complete_model_dir=complete_model_dir,
        source_bundle_root=source_bundle.root,
        pairwise_model_path=Path(args.pairwise_model_path),
    )
    validate_source_bundle_support_files(source_bundle)
    name_counts_index_root = Path(args.name_counts_index_root)
    with ExitStack() as arrow_stack:
        _, arrow_datasets = preflight_source_rows(
            source_bundle,
            name_counts_index_root=name_counts_index_root,
            arrow_stack=arrow_stack,
        )
        clusterer = load_clusterer(
            args.pairwise_model_path,
            n_jobs=int(args.n_jobs),
            expected_artifact_hashes=artifact_hashes,
        )
        for dataset_name, arrow_dataset in arrow_datasets.items():
            require_arrow_name_counts_index_for_clusterer(
                clusterer,
                arrow_dataset,
                context=f"linker train/calibrate/eval preflight dataset {dataset_name!r}",
            )
        pairwise_binding = dict(
            pairwise_bundle_binding(
                Path(args.pairwise_model_path),
                expected_artifact_hashes=artifact_hashes,
            )
        )
        output_dir.mkdir(parents=True)
        work = tempfile.TemporaryDirectory(prefix=".linker-release-", dir=output_dir)
        work_dir = Path(work.name)
        artifact_dir = work_dir / LINKER_STAGING_DIRNAME
        feature_bundle_root = work_dir / "training_features"
        feature_bundle, _featureization_summaries = _materialize_arrow_rust_feature_bundle(
            source_bundle=source_bundle,
            output_bundle_root=feature_bundle_root,
            target=target,
            name_tuples=name_tuples,
            clusterer=clusterer,
            n_jobs=int(args.n_jobs),
            total_ram_bytes=int(args.total_ram_bytes),
            table_keys=training_table_keys,
            query_ids_by_table=calibration_query_ids or None,
            max_exemplars=PRODUCTION_MAX_EXEMPLARS,
            pairwise_model_nan_value=pairwise_model_nan_value,
            pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
            arrow_datasets=arrow_datasets,
        )

        started = time.perf_counter()
        active_params = dict(feature_bundle.models["classic"]["best_params"])

        calibrated = fit_classic(feature_bundle, n_jobs=int(args.n_jobs))
        artifact_metadata = save_incremental_linking_artifact(
            calibrated.model,
            artifact_dir,
            retrieval_top_k=calibrated.retrieval_top_k,
            gate_config=calibrated.gate_config,
            target_spec=target,
            pairwise_bundle_binding=pairwise_binding,
        )
        finalized = finalize_production_bundle(
            pairwise_bundle_dir=Path(args.pairwise_model_path),
            output_bundle_dir=complete_model_dir,
            incremental_linker_artifact_dir=artifact_dir,
            target_json=Path(args.target_json),
            expected_artifact_hashes=artifact_hashes,
        )
        complete_model = load_production_model(
            complete_model_dir,
            expected_artifact_hashes=artifact_hashes,
        )
        if complete_model.incremental_linker_artifact is None:
            raise RuntimeError("Reloaded complete model does not contain an incremental linker artifact")
        shutil.rmtree(artifact_dir)

        evaluation_feature_bundle_root = work_dir / "evaluation_features"
        evaluation_feature_bundle, _evaluation_featureization_summaries = _materialize_arrow_rust_feature_bundle(
            source_bundle=source_bundle,
            output_bundle_root=evaluation_feature_bundle_root,
            target=target,
            name_tuples=name_tuples,
            clusterer=complete_model,
            n_jobs=int(args.n_jobs),
            total_ram_bytes=int(args.total_ram_bytes),
            table_keys=evaluation_table_keys,
            max_exemplars=PRODUCTION_MAX_EXEMPLARS,
            pairwise_model_nan_value=pairwise_model_nan_value,
            pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
            arrow_datasets=arrow_datasets,
        )
    evaluation = evaluate_classic(
        evaluation_feature_bundle,
        calibrated=calibrated,
        artifact=complete_model.incremental_linker_artifact,
        n_jobs=int(args.n_jobs),
    )
    summary = evaluation.summary
    observed = _observed_official_metrics(summary)
    _validate_observed_official_metrics(observed)
    artifact_summary = dict(artifact_metadata)
    result = {
        "mode": "arrow-rust",
        "complete_model_path": str(complete_model_dir),
        "model_manifest_path": str(finalized.manifest_path),
        "source_bundle_root": str(source_bundle.root),
        "target_json": str(Path(args.target_json).resolve()),
        "feature_count": int(target["feature_count"]),
        "n_estimators": int(active_params["n_estimators"]),
        "model_params": dict(active_params),
        "elapsed_seconds": round(float(time.perf_counter() - started), 3),
        "pairwise_bundle_binding": pairwise_binding,
        "input_artifact_hashes": artifact_hashes,
        "observed_metrics": observed,
        "query_predictions": evaluation.query_predictions,
        "feature_nan_policy": feature_nan_policy,
        "artifact_summary": artifact_summary,
        "component_scope": "block-local",
    }
    work.cleanup()
    _write_json(output_dir / "linker_evaluation_report.json", result)
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
        help="Calibrated pairwise-only bundle produced by the release workflow.",
    )
    parser.add_argument("--name-counts-index-root", type=Path, required=True)
    parser.add_argument("--n-jobs", type=_positive_int, default=20)
    parser.add_argument("--total-ram-bytes", type=_positive_int, default=DEFAULT_TOTAL_RAM_BYTES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_arguments(parser)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
