"""Train, calibrate, and evaluate the promoted joint-safe linker target.

This is the official replay entrypoint for the promoted LightGBM
linker/reranker target. It intentionally pins the promoted target JSON instead
of trusting bundle manifests whose classic model specs predate the promotion.

The default `arrow-rust` mode starts from the self-contained Arrow+labels
bundle, rebuilds the promoted feature tables through Rust/Arrow query,
summary, row-signal, pairwise, and row-formula paths, then runs the
train/calibrate/eval stack. The active candidate-member contract is
block-local for retrieval, pairwise distance summaries, and appended `pw_*`
aggregates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import s2and.incremental_linking.query_adapter as retrieval  # noqa: E402
from s2and import feature_port  # noqa: E402
from s2and import text as s2and_text  # noqa: E402
from s2and.arrow_inputs import validate_arrow_prediction_artifacts  # noqa: E402
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER, NAME_COUNTS_INDEX_PATH, NORMALIZATION_VERSION  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.incremental_linking.array_validation import as_retrieval_rank_uint16_1d  # noqa: E402
from s2and.incremental_linking.artifact import save_incremental_linking_artifact  # noqa: E402
from s2and.incremental_linking.contracts import (  # noqa: E402
    DEFAULT_RETRIEVAL_TOP_K,
    canonical_json_digest,
    promoted_linker_feature_schema_digest,
)
from s2and.incremental_linking.feature_block import (  # noqa: E402
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
)
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view  # noqa: E402
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch  # noqa: E402
from s2and.incremental_linking.query_adapter import name_count_rarity_features  # noqa: E402
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS  # noqa: E402
from s2and.incremental_linking.row_features import build_promoted_non_pairwise_row_features  # noqa: E402
from s2and.incremental_linking.runtime import compute_candidate_batch_pairwise_model_and_aggregate_stats  # noqa: E402
from s2and.incremental_linking_training.classic import (  # noqa: E402
    FROZEN_RETRIEVAL_POLICY,
    FROZEN_RETRIEVAL_POLICY_NAME,
    NAN_POLICY_CHOICES,
    PROMOTED_NON_PAIRWISE_COLUMNS,
    PROMOTED_PAIRWISE_COLUMNS,
    ROW_NAN_POLICY_CHOICES,
    SUPPORTED_PROMOTED_FEATURE_COLUMNS,
    WEIGHTED_ERROR_WEIGHTS,
    OfficialBundle,
    _apply_classic_train_holdout_filter,
    _apply_classic_train_row_cap,
    _build_classic_classifier,
    _classic_feature_matrix,
    _drop_unlabeled_singleton_orcid_rows,
    _fit_promoted_logistic_gate,
    _load_classic_stratified_eval_rows,
    _promoted_stratified_gate_spec,
    _query_first_token,
    _read_classic_holdout_identity_sets,
    _read_csv,
    _resolve_classic_monotone_constraints,
    _resolve_path,
    _score_classic_stratified_eval_test,
    load_bundle,
    run_classic,
)
from s2and.incremental_linking_training.data_loading import load_clusterer  # noqa: E402
from s2and.incremental_linking_training.query_support import (  # noqa: E402
    DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
    build_cluster_profile,
    build_labeled_retrieval_subblock_index,
    build_rust_hybrid_centroid_retriever,
    counter_query_overlap,
    rank_top_summaries_rust_hybrid_centroid,
    specter_exemplar_similarity,
    year_compatibility,
)
from s2and.model import (  # noqa: E402
    _build_incremental_constraint_backend,
)
from s2and.production_bundle import finalize_production_bundle, production_version_from_bundle_dir  # noqa: E402
from s2and.production_model import pairwise_bundle_binding  # noqa: E402
from s2and.runtime import build_runtime_context  # noqa: E402
from s2and.rust_calls import get_constraint_labels_index_arrays_rust  # noqa: E402

os.environ.setdefault("S2AND_BACKEND", "rust")

PACKAGE_DATA_ROOT = REPO_ROOT / "s2and" / "data"
DEFAULT_SOURCE_BUNDLE_ROOT = PACKAGE_DATA_ROOT / "s2and_and_big_blocks_linker_dataset_20260525"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "joint_safe_link_promoted_official_20260507"
DEFAULT_NAME_COUNTS_INDEX_ROOT: Path | None = None
DEFAULT_TOTAL_RAM_BYTES = 48 * 1024**3
REQUIRED_TABLE_KEYS = (
    "train_path",
    "classic_gate_source_path",
    "s2and_eval_path",
    "hwang_eval_path",
)
PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION = "precomputed_promoted_feature_bundle_v1"


@dataclass(frozen=True)
class ComponentMembers:
    """Candidate component members in both raw-id and Rust-index forms."""

    signature_ids: tuple[str, ...]
    signature_id_set: frozenset[str]
    signature_indices: np.ndarray


@dataclass
class MinimalRawDatasetContext:
    """Loaded raw dataset state shared across all linker row tables for one dataset."""

    dataset_name: str
    row_component_scope: str
    pairwise_component_scope: str
    dataset: ANDData
    runtime_context: Any
    constraint_backend: Any
    featurizer: Any
    signature_id_to_index: dict[str, int]
    component_details: dict[str, ComponentMembers]
    component_indices: dict[str, np.ndarray]
    pairwise_component_details: dict[str, ComponentMembers]
    pairwise_component_indices: dict[str, np.ndarray]
    component_keys_by_block: dict[str, tuple[str, ...]]
    feature_cache: dict[str, retrieval.QueryFeatures]
    paper_author_name_cache: dict[str, frozenset[str]]
    full_summary_cache: dict[str, retrieval.ClusterSummary]
    residual_summary_cache: dict[tuple[str, str], retrieval.ClusterSummary]
    rust_hybrid_centroid_retriever: Any
    retrieval_subblock_index: dict[str, Any]
    max_block_component_size: int


@dataclass
class ArrowRustDatasetContext:
    """Arrow-only dataset state shared across linker row tables for one dataset."""

    dataset_name: str
    row_component_scope: str
    pairwise_component_scope: str
    runtime_context: Any
    arrow_paths: dict[str, str]
    component_members: dict[str, tuple[str, ...]]
    cluster_seeds_require: dict[str, str]
    cluster_seeds_disallow: frozenset[tuple[str, str]]
    seed_constrained_signature_ids: frozenset[str]
    max_block_component_size: int


@dataclass
class MinimalRawPendingShard:
    """One table/dataset slice that still needs feature materialization."""

    table_key: str
    dataset_name: str
    rows: pd.DataFrame
    row_positions: np.ndarray
    partial_path: Path


@dataclass
class MinimalRawTablePlan:
    """Materialization state for one output feature table."""

    table_key: str
    labels_path: Path
    output_path: Path
    labels: pd.DataFrame
    required_output_columns: list[str]
    partial_dir: Path
    partial_paths: list[Path]
    dataset_summaries: list[dict[str, Any]]
    label_filtering_summary: dict[str, Any]
    structural_cleaning_summary: dict[str, Any]
    started: float


@dataclass(frozen=True)
class FusedDistanceStats:
    """Distance-summary adapter for `_fill_row_signal`."""

    count: int
    min_distance: float
    mean_distance: float
    top3_mean_distance: float
    top5_mean_distance: float

    def topk_mean_distance(self, top_k: int) -> float:
        if int(top_k) <= 3:
            return float(self.top3_mean_distance)
        return float(self.top5_mean_distance)


@dataclass(frozen=True)
class ProdTrainingData:
    """Final production fit rows and per-row weights."""

    rows: pd.DataFrame
    sample_weight: np.ndarray
    source_summaries: list[dict[str, Any]]
    train_holdout_filter_summary: dict[str, Any]
    train_filter_summary: dict[str, Any] | None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_target(path: Path) -> dict[str, Any]:
    target = json.loads(path.read_text(encoding="utf-8"))
    features = tuple(str(feature) for feature in target["features"])
    if len(features) != int(target["feature_count"]):
        raise ValueError(f"Promoted target feature_count mismatch in {path}")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_production_model_path(path: Path) -> str | None:
    name = path.name
    prefix = "production_model_v"
    suffix = ".pickle"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    if name.startswith(prefix):
        return name[len(prefix) :]
    return None


def _portable_repo_path(path: Path) -> str:
    raw_path = Path(path)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name


def _strip_local_paths(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): _strip_local_paths(value)
            for key, value in payload.items()
            if "path" not in str(key).lower() and "root" not in str(key).lower()
        }
    if isinstance(payload, list):
        return [_strip_local_paths(value) for value in payload]
    return payload


def _linker_artifact_audit_metadata(
    *,
    args: argparse.Namespace,
    target: Mapping[str, Any],
    feature_bundle: OfficialBundle,
    featureization_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pairwise_model_path = Path(args.pairwise_model_path)
    pairwise_model = {
        "path": _portable_repo_path(pairwise_model_path),
        "filename": pairwise_model_path.name,
        "version": _version_from_production_model_path(pairwise_model_path),
    }
    if pairwise_model_path.is_file():
        pairwise_model["sha256"] = _sha256_file(pairwise_model_path)
    elif pairwise_model_path.is_dir() and (pairwise_model_path / "manifest.json").exists():
        pairwise_model["manifest_sha256"] = _sha256_file(pairwise_model_path / "manifest.json")
    return {
        "artifact_name": "production_incremental_linker",
        "artifact_version": str(args.linker_artifact_version),
        "target_variant": str(target.get("variant", "")),
        "target_status": str(target.get("status", "")),
        "target_metrics": dict(target.get("metrics", {})),
        "pairwise_model": pairwise_model,
        "pairwise_bundle_binding": pairwise_bundle_binding(pairwise_model_path),
        "training_source_bundle": _portable_repo_path(Path(args.source_bundle_root)),
        "training_feature_mode": str(args.feature_mode),
        "precomputed_feature_bundle": (
            _portable_repo_path(Path(args.precomputed_feature_bundle_root))
            if args.precomputed_feature_bundle_root is not None
            else None
        ),
        "training_feature_bundle_name": str(feature_bundle.bundle_name),
        "feature_nan_policy": _feature_nan_policy_summary(args),
        "featureization": [_strip_local_paths(dict(summary)) for summary in featureization_summaries],
        "target_spec": _portable_repo_path(Path(args.target_json)),
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


def _bundle_with_classic_params(bundle: OfficialBundle, params: Mapping[str, Any]) -> OfficialBundle:
    models = copy.deepcopy(bundle.models)
    classic = dict(models["classic"])
    classic["best_params"] = dict(params)
    models["classic"] = classic
    return OfficialBundle(
        root=bundle.root,
        bundle_name=bundle.bundle_name,
        assets=copy.deepcopy(bundle.assets),
        models=models,
        expected_metrics=copy.deepcopy(bundle.expected_metrics),
    )


def _intify_hyperopt_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return int(value) if float(value).is_integer() else float(value)
    if hasattr(value, "is_integer") and value.is_integer():
        return int(value)
    return value


def _normalize_hyperopt_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _intify_hyperopt_value(value) for key, value in params.items()}


def _classic_hyperopt_search_space(base_params: Mapping[str, Any]) -> dict[str, Any]:
    from hyperopt import hp
    from hyperopt.pyll import scope

    space = {
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "learning_rate": hp.loguniform("learning_rate", math.log(0.005), math.log(0.08)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 16, 1)),
        "min_child_samples": scope.int(hp.qloguniform("min_child_samples", math.log(50), math.log(2000), 1)),
        "min_child_weight": hp.loguniform("min_child_weight", math.log(1e-3), math.log(10.0)),
        "min_split_gain": hp.uniform("min_split_gain", 0.0, 1.0),
        "n_estimators": scope.int(hp.quniform("n_estimators", 300, 1200, 25)),
        "num_leaves": scope.int(hp.qloguniform("num_leaves", math.log(31), math.log(512), 1)),
        "reg_alpha": hp.loguniform("reg_alpha", math.log(1e-4), math.log(32.0)),
        "reg_lambda": hp.loguniform("reg_lambda", math.log(1e-4), math.log(64.0)),
        "subsample": hp.uniform("subsample", 0.7, 1.0),
        "subsample_freq": scope.int(hp.quniform("subsample_freq", 0, 3, 1)),
    }
    return {key: value for key, value in space.items() if key in base_params}


def _hyperopt_loss(summary: Mapping[str, Any], metric: str) -> float:
    observed = _observed_official_metrics(summary)
    if metric == "weighted_average_error":
        return float(observed["weighted_average_error"])
    if metric == "stratified_test_errors":
        return float(observed["stratified_test_errors"])
    if metric == "stratified_test_error_rate":
        return float(observed["stratified_test_error_rate"])
    if metric == "stratified_test_balanced_accuracy":
        return -float(observed["stratified_test_balanced_accuracy"])
    raise ValueError(f"Unsupported hyperopt metric: {metric}")


def _run_classic_hyperopt(
    *,
    feature_bundle: OfficialBundle,
    output_dir: Path,
    base_params: Mapping[str, Any],
    hyperopt_evals: int,
    metric: str,
    seed: int,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune classic LightGBM params by running the full train/calibrate/eval stack."""

    from hyperopt import STATUS_OK, Trials, fmin, tpe

    if int(hyperopt_evals) <= 0:
        raise ValueError("hyperopt_evals must be positive when hyperopt is enabled")
    output_dir.mkdir(parents=True, exist_ok=True)
    search_space = _classic_hyperopt_search_space(base_params)
    if not search_space:
        raise ValueError("No tunable LightGBM parameters were present in base_params")
    trial_records: list[dict[str, Any]] = []

    def evaluate_params(resolved_params: Mapping[str, Any], *, source: str) -> float:
        trial_index = len(trial_records)
        trial_output_dir = output_dir / f"trial_{trial_index:03d}"
        print(
            json.dumps(
                {
                    "event": "classic_hyperopt_trial_start",
                    "trial": trial_index,
                    "source": source,
                    "output_dir": str(trial_output_dir),
                    "metric": metric,
                    "params": dict(resolved_params),
                }
            ),
            flush=True,
        )
        trial_summary = run_classic(
            _bundle_with_classic_params(feature_bundle, dict(resolved_params)),
            trial_output_dir,
            n_jobs=n_jobs,
        )
        observed = _observed_official_metrics(trial_summary)
        loss = _hyperopt_loss(trial_summary, metric)
        record = {
            "trial": trial_index,
            "source": source,
            "loss": float(loss),
            "metric": metric,
            "params": dict(resolved_params),
            "observed_metrics": observed,
            "classic_summary_path": str(trial_output_dir / "summary.json"),
        }
        trial_records.append(record)
        _write_json(output_dir / "trials.json", trial_records)
        print(json.dumps({"event": "classic_hyperopt_trial_done", **record}), flush=True)
        return float(loss)

    baseline_loss = evaluate_params(dict(base_params), source="base_params")

    def objective(params: Mapping[str, Any]) -> dict[str, Any]:
        resolved_params = dict(base_params)
        resolved_params.update(_normalize_hyperopt_params(params))
        loss = evaluate_params(resolved_params, source="tpe")
        return {"loss": float(loss), "status": STATUS_OK}

    trials = Trials()
    search_evals = max(0, int(hyperopt_evals) - 1)
    if search_evals:
        _ = fmin(
            fn=objective,
            space=search_space,
            algo=partial(tpe.suggest, n_startup_jobs=min(5, int(search_evals))),
            max_evals=int(search_evals),
            trials=trials,
            rstate=np.random.default_rng(int(seed)),
        )
    best_record = min(trial_records, key=lambda record: float(record["loss"]))
    best_params = dict(best_record["params"])
    summary = {
        "enabled": True,
        "hyperopt_evals": int(hyperopt_evals),
        "hyperopt_search_evals": int(search_evals),
        "hyperopt_trials_ran": int(len(trial_records)),
        "metric": metric,
        "seed": int(seed),
        "base_loss": float(baseline_loss),
        "best_loss": float(best_record["loss"]),
        "best_trial": int(best_record["trial"]),
        "best_source": str(best_record["source"]),
        "best_params": best_params,
        "trials_path": str(output_dir / "trials.json"),
    }
    _write_json(output_dir / "summary.json", summary)
    return best_params, summary


def _classic_table_keys(spec: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = [key for key in REQUIRED_TABLE_KEYS if key in spec]
    for optional_key in ("s_park_eval_path", "s_lee_eval_path"):
        if optional_key in spec:
            keys.append(optional_key)
    extra_eval_paths = spec.get("extra_eval_paths", {})
    if extra_eval_paths is not None:
        if not isinstance(extra_eval_paths, Mapping):
            raise ValueError("classic.extra_eval_paths must be a mapping")
        for dataset_name in extra_eval_paths:
            keys.append(f"extra_eval_paths.{dataset_name}")
    return tuple(dict.fromkeys(keys))


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


def _selected_row_positions(labels: pd.DataFrame, datasets: set[str] | None, limit_rows: int | None) -> np.ndarray:
    mask = np.ones(len(labels), dtype=bool)
    if datasets is not None:
        mask &= labels["dataset"].astype(str).isin(datasets).to_numpy()
    positions = np.flatnonzero(mask)
    if limit_rows is not None:
        if int(limit_rows) <= 0:
            raise ValueError("limit_rows must be > 0")
        positions = positions[: int(limit_rows)]
    return positions.astype(np.int64, copy=False)


def _read_selected_rows(
    *,
    labels_path: Path,
    corrected_path: Path,
    datasets: set[str] | None,
    limit_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_parquet(labels_path)
    corrected = pd.read_parquet(corrected_path)
    if len(labels) != len(corrected):
        raise ValueError(f"labels/corrected row count mismatch: {labels_path} {len(labels)} != {len(corrected)}")
    positions = _selected_row_positions(labels, datasets, limit_rows)
    labels = labels.iloc[positions].reset_index(drop=True)
    corrected = corrected.iloc[positions].reset_index(drop=True)
    for column in ("dataset", "query_group_id", "candidate_component_key", "retrieval_rank", "label"):
        if column not in corrected.columns:
            continue
        left = labels[column].astype(str).to_numpy()
        right = corrected[column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"labels/corrected identity mismatch for {column!r} in {labels_path.name}")
    return labels, corrected


def _load_raw_signature_blocks(bundle: OfficialBundle, dataset_name: str) -> dict[str, str]:
    raw_datasets = dict(bundle.assets["raw_metadata"]["datasets"])
    if dataset_name not in raw_datasets:
        raise KeyError(f"Minimal raw metadata is missing dataset {dataset_name!r}")
    raw_spec = dict(raw_datasets[dataset_name])
    signatures_path = _resolve_path(bundle, str(raw_spec["signatures_path"]))
    signatures = json.loads(signatures_path.read_text(encoding="utf-8"))
    return {
        str(signature_id): str((signature.get("author_info") or {}).get("block", ""))
        for signature_id, signature in signatures.items()
    }


def _minimal_raw_component_membership_summary(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    cache: dict[str, pd.DataFrame],
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
        signature_to_block = _load_raw_signature_blocks(bundle, dataset_name)

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


def _clean_minimal_raw_structural_rows(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    rows: pd.DataFrame,
    component_membership_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove candidate rows with no non-query member under the block-local contract."""

    required = {"dataset", "query_group_id", "query_signature_id", "candidate_component_key", "label"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{table_key}: minimal raw structural cleaning missing columns: {missing}")
    started = time.perf_counter()
    keep_mask = np.ones(len(rows), dtype=bool)
    labels = pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(np.int8)
    query_ids_before = set(rows["query_group_id"].astype(str))
    positive_query_ids_before = set(rows.loc[labels == 1, "query_group_id"].astype(str))
    dataset_summaries: list[dict[str, Any]] = []

    for dataset_name, dataset_rows in rows.groupby(rows["dataset"].astype(str), sort=False):
        membership = _minimal_raw_component_membership_summary(
            source_bundle,
            str(dataset_name),
            cache=component_membership_cache,
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
        drop = (local["_component_member_count"].astype(np.int64) == 1) & local[
            "_component_single_member_signature_id"
        ].astype(str).eq(local["query_signature_id"].astype(str))
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
        "policy": "drop_candidate_rows_with_no_non_query_block_local_members",
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


def _clean_arrow_rust_structural_rows(
    *,
    source_bundle: OfficialBundle,
    table_key: str,
    rows: pd.DataFrame,
    component_membership_cache: dict[str, pd.DataFrame],
    name_counts_index_root: Path | None,
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
        drop = (local["_component_member_count"].astype(np.int64) == 1) & local[
            "_component_single_member_signature_id"
        ].astype(str).eq(local["query_signature_id"].astype(str))
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


def _component_members_by_key(path: Path, signature_id_to_index: Mapping[str, int]) -> dict[str, np.ndarray]:
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    out: dict[str, np.ndarray] = {}
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        member_ids = group.sort_values("member_index")["signature_id"].astype(str)
        member_indices: list[int] = []
        for signature_id in member_ids:
            try:
                member_indices.append(int(signature_id_to_index[str(signature_id)]))
            except KeyError as exc:
                raise KeyError(f"component member signature_id missing from Rust featurizer: {signature_id}") from exc
        out[str(component_key)] = np.asarray(member_indices, dtype=np.uint32)
    return out


def _block_local_member_ids(
    dataset: ANDData,
    component_key: str,
    member_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return _block_local_member_ids_from_signature_blocks(
        component_key,
        member_ids,
        getattr(dataset, "signature_to_block", {}) or {},
    )


def _block_local_member_ids_from_signature_blocks(
    component_key: str,
    member_ids: tuple[str, ...],
    signature_to_block: Mapping[str, str],
) -> tuple[str, ...]:
    if "::" not in str(component_key):
        return member_ids
    block_key, _cluster_id = str(component_key).split("::", 1)
    filtered = tuple(
        signature_id for signature_id in member_ids if str(signature_to_block.get(str(signature_id), "")) == block_key
    )
    return filtered or member_ids


def _component_member_details_by_key(
    path: Path,
    signature_id_to_index: Mapping[str, int],
    *,
    dataset: ANDData,
) -> dict[str, ComponentMembers]:
    members = pd.read_parquet(path)
    required = {"candidate_component_key", "member_index", "signature_id"}
    missing = sorted(required - set(members.columns))
    if missing:
        raise ValueError(f"candidate member table {path} is missing columns: {missing}")
    out: dict[str, ComponentMembers] = {}
    for component_key, group in members.groupby("candidate_component_key", sort=False):
        member_ids = tuple(str(value) for value in group.sort_values("member_index")["signature_id"].astype(str))
        member_ids = _block_local_member_ids(dataset, str(component_key), member_ids)
        member_indices: list[int] = []
        for signature_id in member_ids:
            try:
                member_indices.append(int(signature_id_to_index[str(signature_id)]))
            except KeyError as exc:
                raise KeyError(f"component member signature_id missing from Rust featurizer: {signature_id}") from exc
        out[str(component_key)] = ComponentMembers(
            signature_ids=member_ids,
            signature_id_set=frozenset(member_ids),
            signature_indices=np.asarray(member_indices, dtype=np.uint32),
        )
    return out


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
) -> dict[str, str]:
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
    if "specter" not in paths and "specter2" in paths:
        paths["specter"] = paths["specter2"]
    if "specter_batch_index" not in paths and "specter2_batch_index" in paths:
        paths["specter_batch_index"] = paths["specter2_batch_index"]
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
) -> dict[str, str]:
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


def _candidate_batch_from_rows(
    rows: pd.DataFrame,
    component_members: Mapping[str, np.ndarray],
    signature_id_to_index: Mapping[str, int],
    *,
    row_group_ids: Sequence[int] | None = None,
) -> LinkerCandidateBatch:
    query_indices = np.empty(len(rows), dtype=np.uint32)
    member_arrays: list[np.ndarray] = []
    for row_offset, row in enumerate(rows.itertuples(index=False)):
        row_any = cast(Any, row)
        query_signature_id = str(row_any.query_signature_id)
        component_key = str(row_any.candidate_component_key)
        try:
            query_index = int(signature_id_to_index[query_signature_id])
        except KeyError as exc:
            raise KeyError(f"query_signature_id missing from Rust featurizer: {query_signature_id}") from exc
        try:
            members = component_members[component_key]
        except KeyError as exc:
            raise KeyError(f"candidate_component_key missing from members table: {component_key}") from exc
        query_indices[row_offset] = query_index
        active_members = members[members != query_index]
        member_arrays.append(np.ascontiguousarray(active_members, dtype=np.uint32))

    pair_count = int(sum(len(members) for members in member_arrays))
    left = np.empty(pair_count, dtype=np.uint32)
    right = np.empty(pair_count, dtype=np.uint32)
    owner_rows = np.empty(pair_count, dtype=np.uint32)
    offset = 0
    for row_offset, members in enumerate(member_arrays):
        stop = offset + len(members)
        left[offset:stop] = query_indices[row_offset]
        right[offset:stop] = members
        owner_rows[offset:stop] = row_offset
        offset = stop

    return LinkerCandidateBatch(
        row_count=len(rows),
        left_signature_indices=left,
        right_signature_indices=right,
        pair_row_indices=owner_rows,
        row_query_signature_indices=(
            np.asarray(row_group_ids, dtype=np.uint32) if row_group_ids is not None else query_indices
        ),
        row_component_keys=tuple(rows["candidate_component_key"].astype(str).tolist()),
        labels=rows["label"].to_numpy(dtype=np.int8, copy=False) if "label" in rows.columns else None,
        retrieval_scores=(
            rows["retrieval_score"].to_numpy(dtype=np.float32, copy=False)
            if "retrieval_score" in rows.columns
            else None
        ),
        retrieval_ranks=rows["retrieval_rank"].to_numpy(copy=False) if "retrieval_rank" in rows.columns else None,
    )


def _load_minimal_raw_specter_dataset(
    bundle: OfficialBundle,
    dataset_name: str,
    *,
    n_jobs: int,
) -> ANDData:
    raw_datasets = dict(bundle.assets["raw_metadata"]["datasets"])
    embedding_datasets = dict(bundle.assets.get("embeddings", {}).get("datasets", {}))
    if dataset_name not in raw_datasets:
        raise KeyError(f"Minimal raw metadata is missing dataset {dataset_name!r}")
    if dataset_name not in embedding_datasets:
        raise KeyError(f"Minimal SPECTER2 embeddings are missing dataset {dataset_name!r}")
    raw_spec = dict(raw_datasets[dataset_name])
    signatures_path = _resolve_path(bundle, str(raw_spec["signatures_path"]))
    papers_path = _resolve_path(bundle, str(raw_spec["papers_path"]))
    specter_path = _resolve_path(bundle, str(embedding_datasets[dataset_name]))
    dataset = ANDData(
        str(signatures_path),
        str(papers_path),
        name=f"joint_safe_link_minimal_raw_specter_{dataset_name}",
        mode="inference",
        specter_embeddings=str(specter_path),
        name_counts_index=NAME_COUNTS_INDEX_PATH,
        preprocess=True,
        n_jobs=max(1, int(n_jobs)),
        use_orcid_id=False,
        name_tuples="filtered",
    )
    return dataset


def _build_full_retrieval_summary_cache(
    *,
    dataset: ANDData,
    component_details: Mapping[str, ComponentMembers],
    feature_cache: dict[str, retrieval.QueryFeatures],
    paper_author_name_cache: dict[str, frozenset[str]],
    max_exemplars: int,
) -> dict[str, retrieval.ClusterSummary]:
    summaries: dict[str, retrieval.ClusterSummary] = {}
    for component_key, details in component_details.items():
        summaries[str(component_key)] = _build_summary_for_members(
            dataset=dataset,
            component_key=str(component_key),
            candidate_cluster_id=None,
            signature_ids=details.signature_ids,
            feature_cache=feature_cache,
            paper_author_name_cache=paper_author_name_cache,
            max_exemplars=max_exemplars,
        )
    return summaries


def _component_block_key(
    dataset: ANDData,
    component_key: str,
    details: ComponentMembers,
) -> str:
    if "::" in str(component_key):
        return str(component_key).split("::", 1)[0]
    signature_to_block = getattr(dataset, "signature_to_block", {}) or {}
    for signature_id in details.signature_ids:
        block_key = str(signature_to_block.get(str(signature_id), ""))
        if block_key:
            return block_key
    return str(component_key)


def _build_retrieval_subblock_index_for_components(
    *,
    dataset: ANDData,
    component_details: Mapping[str, ComponentMembers],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, tuple[str, ...]]]:
    block_to_component_keys: dict[str, list[str]] = {}
    component_signatures: dict[str, list[str]] = {}
    for component_key, details in component_details.items():
        key = str(component_key)
        block_key = _component_block_key(dataset, key, details)
        block_to_component_keys.setdefault(block_key, []).append(key)
        component_signatures[key] = list(details.signature_ids)
    index, diagnostics = build_labeled_retrieval_subblock_index(
        dataset=dataset,
        block_to_component_keys=block_to_component_keys,
        component_signatures=component_signatures,
    )
    component_keys_by_block = {
        block_key: tuple(component_keys) for block_key, component_keys in block_to_component_keys.items()
    }
    return index, diagnostics, component_keys_by_block


def _build_minimal_raw_dataset_context(
    *,
    source_bundle: OfficialBundle,
    dataset_name: str,
    clusterer: Any,
    n_jobs: int,
    max_exemplars: int,
) -> MinimalRawDatasetContext:
    started = time.perf_counter()
    dataset = _load_minimal_raw_specter_dataset(
        source_bundle,
        dataset_name,
        n_jobs=n_jobs,
    )
    runtime_context = build_runtime_context(
        "joint_safe_link_minimal_raw_featureization",
        backend="rust",
    )
    featurizer = feature_port._get_rust_featurizer(  # noqa: SLF001
        dataset,
        runtime_context=runtime_context,
    )
    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=bool(clusterer.use_default_constraints_as_supervision),
        runtime_context=runtime_context,
        suppress_orcid=True,
    )
    signature_id_to_index = _signature_id_to_index(featurizer)
    member_path = _resolve_path(
        source_bundle,
        str(source_bundle.assets["candidate_members"]["datasets"][dataset_name]),
    )
    row_component_scope = "block-local"
    pairwise_component_scope = "block-local"
    component_details = _component_member_details_by_key(
        member_path,
        signature_id_to_index,
        dataset=dataset,
    )
    component_indices = {
        component_key: details.signature_indices for component_key, details in component_details.items()
    }
    pairwise_component_details = component_details
    pairwise_component_indices = component_indices
    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    paper_author_name_cache: dict[str, frozenset[str]] = {}
    retrieval_subblock_index, retrieval_subblock_index_diagnostics, component_keys_by_block = (
        _build_retrieval_subblock_index_for_components(
            dataset=dataset,
            component_details=component_details,
        )
    )
    summary_started = time.perf_counter()
    full_summary_cache = _build_full_retrieval_summary_cache(
        dataset=dataset,
        component_details=component_details,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        max_exemplars=max_exemplars,
    )
    rust_hybrid_centroid_retriever = build_rust_hybrid_centroid_retriever(
        list(full_summary_cache.values()),
        include_exemplars=FROZEN_RETRIEVAL_POLICY.uses_exemplar_scoring(),
    )
    max_block_component_size = max((summary.size for summary in full_summary_cache.values()), default=0)
    summary_seconds = float(time.perf_counter() - summary_started)
    print(
        json.dumps(
            {
                "event": "minimal_raw_dataset_context_ready",
                "dataset": dataset_name,
                "components": int(len(component_details)),
                "specter_embeddings": int(len(dataset.specter_embeddings or {})),
                "retrieval_policy": FROZEN_RETRIEVAL_POLICY_NAME,
                "component_scope": "block-local",
                "row_component_scope": row_component_scope,
                "pairwise_component_scope": pairwise_component_scope,
                "retrieval_subblock_index": retrieval_subblock_index_diagnostics,
                "retrieval_summary_build_seconds": round(summary_seconds, 3),
                "seconds": round(float(time.perf_counter() - started), 3),
            }
        ),
        flush=True,
    )
    return MinimalRawDatasetContext(
        dataset_name=dataset_name,
        row_component_scope=row_component_scope,
        pairwise_component_scope=pairwise_component_scope,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        featurizer=featurizer,
        signature_id_to_index=signature_id_to_index,
        component_details=component_details,
        component_indices=component_indices,
        pairwise_component_details=pairwise_component_details,
        pairwise_component_indices=pairwise_component_indices,
        component_keys_by_block=component_keys_by_block,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        full_summary_cache=full_summary_cache,
        residual_summary_cache={},
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        retrieval_subblock_index=retrieval_subblock_index,
        max_block_component_size=int(max_block_component_size),
    )


def _release_minimal_raw_dataset_context(context: MinimalRawDatasetContext) -> None:
    context.feature_cache.clear()
    context.paper_author_name_cache.clear()
    context.full_summary_cache.clear()
    context.residual_summary_cache.clear()
    context.retrieval_subblock_index.clear()
    context.component_details.clear()
    context.component_indices.clear()
    context.component_keys_by_block.clear()
    context.pairwise_component_details.clear()
    context.pairwise_component_indices.clear()
    context.signature_id_to_index.clear()
    feature_port.clear_rust_featurizer_cache()
    gc.collect()


def _build_arrow_rust_dataset_context(
    *,
    source_bundle: OfficialBundle,
    dataset_name: str,
    name_counts_index_root: Path | None,
) -> ArrowRustDatasetContext:
    started = time.perf_counter()
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
    feature_port.clear_rust_featurizer_cache()
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
            "raw Arrow labeled plan row_component_keys length mismatch: " f"{len(row_component_keys)} != {row_count}"
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
            None,
            batch.left_signature_indices,
            batch.right_signature_indices,
            dont_merge_cluster_seeds=True,
            incremental_dont_use_cluster_seeds=False,
            num_threads=max(1, int(n_jobs)),
            featurizer=featurizer,
            runtime_context=None,
            suppress_orcid=True,
        )
    seed_bypass_indices = np.flatnonzero(pair_seed_bypass) if constraints_enabled else np.asarray([], dtype=np.int64)
    if len(seed_bypass_indices):
        labels[seed_bypass_indices] = get_constraint_labels_index_arrays_rust(
            None,
            batch.left_signature_indices[seed_bypass_indices],
            batch.right_signature_indices[seed_bypass_indices],
            dont_merge_cluster_seeds=True,
            incremental_dont_use_cluster_seeds=True,
            num_threads=max(1, int(n_jobs)),
            featurizer=featurizer,
            runtime_context=None,
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


def _assert_pairwise_model_is_raw_bundle_compatible(clusterer: Any, model_path: Path) -> None:
    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = tuple(str(value) for value in getattr(featurizer_info, "features_to_use", ()) or ())
        if "reference_features" in features_to_use:
            raise ValueError(
                f"Pairwise model {model_path} uses reference_features in {attr_name}; "
                "the minimal raw bundle intentionally does not store reference papers."
            )


def _component_id_parts(component_key: str, candidate_cluster_id: str | None = None) -> tuple[str, str]:
    if "::" in component_key:
        block_key, cluster_id = component_key.split("::", 1)
        return block_key, cluster_id
    cluster_id = str(candidate_cluster_id or component_key)
    if "_" in component_key:
        return component_key.split("_", 1)[0], cluster_id
    return component_key, cluster_id


def _finite_distance(value: float, *, empty_value: float = 1.0) -> float:
    value = float(value)
    return value if math.isfinite(value) else float(empty_value)


def _nan_value_from_policy(policy: str) -> float:
    if policy == "preserve":
        return float("nan")
    if policy == "zero":
        return 0.0
    raise ValueError(f"Unsupported NaN policy: {policy}")


def _feature_nan_policy_summary(args: argparse.Namespace) -> dict[str, str]:
    return {
        "pairwise_model_nan_policy": str(args.pairwise_model_nan_policy),
        "pairwise_aggregate_nan_policy": str(args.pairwise_aggregate_nan_policy),
        "row_nan_policy": str(args.row_nan_policy),
    }


def _score_candidate_summaries_with_frozen_rust_policy(
    *,
    query: retrieval.QueryFeatures,
    summaries: Mapping[str, retrieval.ClusterSummary],
    retriever: Any,
    n_jobs: int,
) -> dict[str, float]:
    """Score one query's candidate rows with the frozen Rust retrieval policy."""

    component_keys = [str(component_key) for component_key in summaries]
    if not component_keys:
        return {}
    override_summary: retrieval.ClusterSummary | None = None
    overridden_component_keys: list[str] = []
    summary_by_component = getattr(retriever, "summary_by_component", {})
    for component_key in component_keys:
        base_summary = summary_by_component.get(component_key)
        if base_summary is None:
            raise KeyError(f"Unknown component_key for frozen Rust retrieval: {component_key}")
        current_summary = summaries[component_key]
        if current_summary is not base_summary:
            overridden_component_keys.append(component_key)
            override_summary = current_summary
    if len(overridden_component_keys) > 1:
        raise ValueError(
            "Frozen Rust retrieval scoring supports at most one residual summary per query group; "
            f"got {overridden_component_keys}"
        )
    ranked = rank_top_summaries_rust_hybrid_centroid(
        query=query,
        retriever=retriever,
        component_keys=component_keys,
        override_summary=override_summary,
        num_threads=max(1, int(n_jobs)),
        weights=FROZEN_RETRIEVAL_POLICY.weights_for_query(query),
        scoring_config=FROZEN_RETRIEVAL_POLICY.scoring_config_for_query(query),
    )
    return {str(summary.component_key): round(float(score), 6) for score, summary in ranked}


def _current_retrieval_ranks_from_scores(
    retrieval_scores: Mapping[str, float],
    stored_retrieval_ranks: Mapping[str, int],
) -> dict[str, int]:
    """Return rank order induced by recomputed retrieval scores over the frozen candidate set."""

    ordered = sorted(
        retrieval_scores,
        key=lambda component_key: (
            -float(retrieval_scores[str(component_key)]),
            int(stored_retrieval_ranks[str(component_key)]),
            str(component_key),
        ),
    )
    return {str(component_key): rank for rank, component_key in enumerate(ordered, start=1)}


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


def _dataset_has_cluster_seed_constraints(dataset: ANDData) -> bool:
    return bool(getattr(dataset, "cluster_seeds_require", None)) or bool(
        getattr(dataset, "cluster_seeds_disallow", None)
    )


def _signature_has_seed_constraint(dataset: ANDData, signature_id: str) -> bool:
    signature_id = str(signature_id)
    require = getattr(dataset, "cluster_seeds_require", {}) or {}
    if signature_id in require:
        return True
    disallow = getattr(dataset, "cluster_seeds_disallow", set()) or set()
    return any(str(left) == signature_id or str(right) == signature_id for left, right in disallow)


def _seed_constrained_signature_ids(dataset: ANDData) -> frozenset[str]:
    signature_ids = {str(signature_id) for signature_id in (getattr(dataset, "cluster_seeds_require", {}) or {})}
    for left, right in getattr(dataset, "cluster_seeds_disallow", set()) or set():
        signature_ids.add(str(left))
        signature_ids.add(str(right))
    return frozenset(signature_ids)


def _row_allows_seed_constraint_bypass(
    dataset: ANDData,
    row: Any,
    *,
    seed_constraint_signature_ids: frozenset[str] | None = None,
) -> bool:
    if _truthy_row_value(getattr(row, "query_in_seed_before_holdout", None)):
        return True
    query_signature_id = getattr(row, "query_signature_id", None)
    if query_signature_id is not None:
        if seed_constraint_signature_ids is None:
            has_seed_constraint = _signature_has_seed_constraint(dataset, str(query_signature_id))
        else:
            has_seed_constraint = str(query_signature_id) in seed_constraint_signature_ids
        if has_seed_constraint:
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


def _has_query_seed_connection(
    dataset: ANDData,
    *,
    query_signature_id: str,
    candidate_signature_ids: Sequence[str],
) -> bool:
    return _has_query_seed_connection_from_maps(
        getattr(dataset, "cluster_seeds_require", {}) or {},
        getattr(dataset, "cluster_seeds_disallow", set()) or set(),
        query_signature_id=query_signature_id,
        candidate_signature_ids=candidate_signature_ids,
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
            cast(Any, None),
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


def _resolve_candidate_batch_pair_labels(
    *,
    clusterer: Any,
    dataset: ANDData,
    batch: LinkerCandidateBatch,
    index_to_signature_id: Mapping[int, str],
    runtime_context: Any,
    constraint_backend: Any,
    chunk_size: int,
    pair_seed_bypass: np.ndarray | None = None,
    pair_ignore_disallow: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    pair_count = int(batch.pair_count)
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    chunk_size = max(1, int(chunk_size))
    started = time.perf_counter()
    batch_calls = 0

    if pair_seed_bypass is None:
        pair_seed_bypass = np.zeros(pair_count, dtype=bool)
    else:
        pair_seed_bypass = np.asarray(pair_seed_bypass, dtype=bool)
    if pair_ignore_disallow is None:
        pair_ignore_disallow = np.zeros(pair_count, dtype=bool)
    else:
        pair_ignore_disallow = np.asarray(pair_ignore_disallow, dtype=bool)
    if len(pair_seed_bypass) != pair_count:
        raise ValueError(f"pair_seed_bypass length {len(pair_seed_bypass)} != pair_count {pair_count}")
    if len(pair_ignore_disallow) != pair_count:
        raise ValueError(f"pair_ignore_disallow length {len(pair_ignore_disallow)} != pair_count {pair_count}")

    for start in range(0, pair_count, chunk_size):
        stop = min(pair_count, start + chunk_size)
        pairs = [
            (
                index_to_signature_id[int(left_index)],
                index_to_signature_id[int(right_index)],
            )
            for left_index, right_index in zip(
                batch.left_signature_indices[start:stop],
                batch.right_signature_indices[start:stop],
                strict=True,
            )
        ]
        chunk_labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
            dataset,
            pairs,
            partial_supervision={},
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=False,
            constraint_backend=constraint_backend,
        )
        labels[start:stop] = np.asarray(chunk_labels, dtype=np.float64)
        batch_calls += 1

    seed_bypass_indices = np.flatnonzero(pair_seed_bypass)
    seed_bypass_batch_calls = 0
    for start in range(0, len(seed_bypass_indices), chunk_size):
        stop = min(len(seed_bypass_indices), start + chunk_size)
        chunk_indices = seed_bypass_indices[start:stop]
        pairs = [
            (
                index_to_signature_id[int(batch.left_signature_indices[int(index)])],
                index_to_signature_id[int(batch.right_signature_indices[int(index)])],
            )
            for index in chunk_indices
        ]
        chunk_labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
            dataset,
            pairs,
            partial_supervision={},
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=True,
            constraint_backend=constraint_backend,
        )
        labels[chunk_indices] = np.asarray(chunk_labels, dtype=np.float64)
        seed_bypass_batch_calls += 1

    disallow_ignored = np.zeros(pair_count, dtype=bool)
    if np.any(pair_ignore_disallow):
        disallow_ignored = pair_ignore_disallow & np.asarray(
            [_constraint_label_is_disallow(float(label)) for label in labels],
            dtype=bool,
        )
        labels[disallow_ignored] = np.nan
    return labels, {
        "constraint_pair_count": pair_count,
        "constraint_batch_calls": int(batch_calls),
        "constraint_seed_bypass_pair_count": int(len(seed_bypass_indices)),
        "constraint_seed_bypass_batch_calls": int(seed_bypass_batch_calls),
        "constraint_disallow_ignored_pair_count": int(disallow_ignored.sum()),
        "constraint_seconds": round(float(time.perf_counter() - started), 3),
    }


def _parquet_row_count_and_columns(path: Path) -> tuple[int, set[str]]:
    parquet_file = pq.ParquetFile(path)
    return int(parquet_file.metadata.num_rows), set(parquet_file.schema_arrow.names)


def _validate_reusable_parquet(
    path: Path,
    *,
    expected_rows: int,
    required_columns: Iterable[str],
    context: str,
) -> int:
    row_count, columns = _parquet_row_count_and_columns(path)
    if row_count != int(expected_rows):
        raise ValueError(f"{context}: reusable parquet row count mismatch: {row_count} != {expected_rows} ({path})")
    missing_columns = sorted(set(required_columns) - columns)
    if missing_columns:
        raise ValueError(f"{context}: reusable parquet is missing columns: {missing_columns[:10]} ({path})")
    return row_count


def _relative_bundle_asset_path(bundle: OfficialBundle, path: Path) -> str:
    """Return a portable bundle-relative path for a resolved asset path."""

    try:
        return str(path.resolve().relative_to(bundle.root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Precomputed feature path escapes bundle root: {path}") from exc


def _target_spec_digest(target: Mapping[str, Any]) -> str:
    """Return the stable digest for a promoted training target spec."""

    return canonical_json_digest(dict(target))


def _precomputed_table_metadata(bundle: OfficialBundle, target_features: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return validated table metadata for a portable precomputed feature bundle."""

    spec = dict(bundle.models["classic"])
    table_metadata: dict[str, dict[str, Any]] = {}
    for table_key in _classic_table_keys(spec):
        path = _asset_file(bundle, "corrected_feature_rows", table_key)
        row_count, columns = _parquet_row_count_and_columns(path)
        missing_features = sorted(set(str(feature) for feature in target_features) - columns)
        if missing_features:
            raise ValueError(f"{table_key}: precomputed table is missing target features: {missing_features[:10]}")
        table_metadata[table_key] = {
            "path": _relative_bundle_asset_path(bundle, path),
            "rows": int(row_count),
            "feature_count": int(len(target_features)),
        }
    return table_metadata


def _precomputed_promoted_bundle_metadata(
    *,
    bundle: OfficialBundle,
    target: Mapping[str, Any],
    source_mode: str,
) -> dict[str, Any]:
    """Build portable metadata for a validated precomputed promoted feature bundle."""

    target_features = tuple(str(feature) for feature in target["features"])
    return {
        "schema_version": PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION,
        "source_mode": str(source_mode),
        "target_spec_digest": _target_spec_digest(target),
        "feature_schema_digest": promoted_linker_feature_schema_digest(target_features),
        "feature_count": int(target["feature_count"]),
        "feature_columns": list(target_features),
        "tables": _precomputed_table_metadata(bundle, target_features),
    }


def _stamp_precomputed_promoted_bundle_metadata(
    *,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    source_mode: str,
) -> None:
    """Persist portable precomputed-feature metadata into `bundle.json`."""

    bundle = load_bundle(output_bundle_root)
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    payload["precomputed_promoted_feature_bundle"] = _precomputed_promoted_bundle_metadata(
        bundle=bundle,
        target=target,
        source_mode=source_mode,
    )
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_precomputed_promoted_feature_bundle(
    *,
    bundle_root: Path,
    target: Mapping[str, Any],
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    """Load and validate a portable precomputed promoted feature bundle."""

    root = bundle_root.resolve()
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    metadata = payload.get("precomputed_promoted_feature_bundle")
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "precomputed-promoted bundles must include precomputed_promoted_feature_bundle metadata; "
            "rerun materialization with --reuse-existing-features to stamp it"
        )
    if metadata.get("schema_version") != PRECOMPUTED_PROMOTED_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported precomputed promoted bundle schema_version: " f"{metadata.get('schema_version')!r}"
        )
    target_features = tuple(str(feature) for feature in target["features"])
    if tuple(str(feature) for feature in metadata.get("feature_columns", ())) != target_features:
        raise ValueError("Precomputed promoted bundle feature columns do not match target_json")
    if int(metadata.get("feature_count", -1)) != int(target["feature_count"]):
        raise ValueError("Precomputed promoted bundle feature_count does not match target_json")
    expected_target_digest = _target_spec_digest(target)
    if metadata.get("target_spec_digest") != expected_target_digest:
        raise ValueError("Precomputed promoted bundle target_spec_digest does not match target_json")
    expected_schema_digest = promoted_linker_feature_schema_digest(target_features)
    if metadata.get("feature_schema_digest") != expected_schema_digest:
        raise ValueError("Precomputed promoted bundle feature_schema_digest does not match target_json")

    raw_files = dict(payload.get("assets", {}).get("corrected_feature_rows", {}).get("files", {}))
    absolute_paths = sorted(str(path) for path in raw_files.values() if Path(str(path)).is_absolute())
    if absolute_paths:
        raise ValueError(f"Precomputed promoted bundle contains absolute feature paths: {absolute_paths[:5]}")

    bundle = _bundle_with_promoted_target(load_bundle(root), target)
    if tuple(str(feature) for feature in bundle.models["classic"]["feature_columns"]) != target_features:
        raise ValueError("Precomputed promoted bundle classic feature_columns do not match target_json")
    table_metadata = metadata.get("tables", {})
    if not isinstance(table_metadata, Mapping):
        raise ValueError("Precomputed promoted bundle metadata must include table row counts")
    featureization_summaries: list[dict[str, Any]] = []
    for table_key in _classic_table_keys(bundle.models["classic"]):
        if table_key not in table_metadata:
            raise ValueError(f"Precomputed promoted bundle metadata is missing table {table_key!r}")
        table_payload = dict(cast(Mapping[str, Any], table_metadata[table_key]))
        table_path = _asset_file(bundle, "corrected_feature_rows", table_key)
        if Path(str(table_payload.get("path", ""))).is_absolute():
            raise ValueError(f"{table_key}: precomputed table metadata path must be bundle-relative")
        if str(table_payload.get("path", "")) != _relative_bundle_asset_path(bundle, table_path):
            raise ValueError(f"{table_key}: precomputed table metadata path does not match bundle asset path")
        expected_rows = int(table_payload["rows"])
        row_count = _validate_reusable_parquet(
            table_path,
            expected_rows=expected_rows,
            required_columns=target_features,
            context=f"{table_key} precomputed promoted feature table",
        )
        featureization_summaries.append(
            {
                "table_key": table_key,
                "output_path": str(table_path.relative_to(bundle.root)),
                "rows": int(row_count),
                "mode": "precomputed-promoted",
                "reused": True,
            }
        )
    return bundle, featureization_summaries


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


def _required_materialized_output_columns(labels: pd.DataFrame, target_features: Sequence[str]) -> list[str]:
    """Return output columns, treating label columns as reusable feature columns."""

    columns = [str(column) for column in labels.columns]
    seen = set(columns)
    for feature in target_features:
        feature = str(feature)
        if feature not in seen:
            columns.append(feature)
            seen.add(feature)
    return columns


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
    *,
    reuse_existing_features: bool = False,
) -> dict[str, Any]:
    if output_bundle_root.exists() and not reuse_existing_features:
        shutil.rmtree(output_bundle_root)
    output_bundle_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bundle.root / "splits", output_bundle_root / "splits", dirs_exist_ok=True)
    shutil.copy2(source_bundle.root / "bundle.json", output_bundle_root / "bundle.json")
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    payload["bundle_name"] = f"{payload['bundle_name']}_promoted_rust_recomputed_pw"
    payload["expected_metrics"] = {}
    (output_bundle_root / "bundle.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_summary_for_members(
    *,
    dataset: ANDData,
    component_key: str,
    candidate_cluster_id: str | None,
    signature_ids: Sequence[str],
    feature_cache: dict[str, retrieval.QueryFeatures],
    paper_author_name_cache: dict[str, frozenset[str]] | None,
    max_exemplars: int,
) -> retrieval.ClusterSummary:
    block_key, cluster_id = _component_id_parts(component_key, candidate_cluster_id)
    return retrieval.build_cluster_summary(
        dataset=dataset,
        block_key=block_key,
        cluster_id=cluster_id,
        component_key=component_key,
        signature_ids=[str(signature_id) for signature_id in signature_ids],
        max_exemplars=max_exemplars,
        feature_cache=feature_cache,
        paper_author_name_cache=paper_author_name_cache,
        orcid_enabled=False,
    )


def _initialize_row_signal_arrays(row_count: int, rows: pd.DataFrame) -> dict[str, Any]:
    float_signal_names = (
        "retrieval_score",
        "retrieval_rank",
        "cluster_size",
        "named_signature_count",
        "candidate_year_min",
        "candidate_year_max",
        "candidate_year_range_missing",
        "query_year",
        "query_year_missing",
        "query_has_affiliations",
        "query_has_coauthors",
        "query_has_specter",
        "query_has_name_counts",
        "candidate_has_affiliations",
        "candidate_has_coauthors",
        "candidate_has_specter_exemplars",
        "candidate_has_name_counts",
        "candidate_cluster_max_paper_author_count",
        "paper_author_list_max_jaccard",
        "paper_author_list_max_containment",
        "paper_author_list_max_overlap_count",
        "local_author_window10_jaccard_max",
        "local_author_window10_overlap_count_max",
        "best_author_count_log_absdiff",
        "affiliation_overlap",
        "coauthor_overlap",
        "year_compatibility",
        "specter_exemplar_similarity",
        "min_distance",
        "mean_distance",
        "top3_mean_distance",
        "top5_mean_distance",
        "pair_count",
        "last_name_count_min_rarity",
        "candidate_last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
    )
    signals: dict[str, Any] = {name: np.full(row_count, np.nan, dtype=np.float32) for name in float_signal_names}
    signals["candidate_component_key"] = rows["candidate_component_key"].astype(str).to_numpy(dtype=object)
    signals["query_view"] = rows["query_view"].astype(str).to_numpy(dtype=object)
    signals["dominant_first_name"] = np.empty(row_count, dtype=object)
    signals["query_first_token"] = np.empty(row_count, dtype=object)
    signals["first_name_bucket"] = np.empty(row_count, dtype=object)
    signals["family_id"] = np.empty(row_count, dtype=object)
    return signals


def _fill_row_signal(
    *,
    row_signals: dict[str, Any],
    local_index: int,
    component_key: str,
    query: retrieval.QueryFeatures,
    summary: retrieval.ClusterSummary,
    stats: Any,
    retrieval_rank: int,
    retrieval_score: float,
    query_first_token_for_prefix: str,
) -> None:
    profile = build_cluster_profile(summary)
    rarity = name_count_rarity_features(query, summary)
    query_year_missing = query.year is None
    candidate_year_missing = summary.year_min is None or summary.year_max is None

    row_signals["retrieval_score"][local_index] = float(retrieval_score)
    row_signals["retrieval_rank"][local_index] = float(retrieval_rank)
    row_signals["cluster_size"][local_index] = float(summary.size)
    row_signals["named_signature_count"][local_index] = float(profile.family_named_count)
    row_signals["dominant_first_name"][local_index] = str(profile.dominant_first_name or "")
    row_signals["family_id"][local_index] = str(profile.family_id or component_key)
    row_signals["candidate_year_min"][local_index] = float(summary.year_min or 0)
    row_signals["candidate_year_max"][local_index] = float(summary.year_max or 0)
    row_signals["candidate_year_range_missing"][local_index] = float(candidate_year_missing)
    query_first_token = str(query_first_token_for_prefix or query.first or "")
    row_signals["query_first_token"][local_index] = query_first_token
    row_signals["first_name_bucket"][local_index] = first_name_bucket_from_token_view(
        query_first_token,
        row_signals["query_view"][local_index],
    )
    row_signals["query_year"][local_index] = float(query.year or 0)
    row_signals["query_year_missing"][local_index] = float(query_year_missing)
    row_signals["query_has_affiliations"][local_index] = float(query.has_affiliations)
    row_signals["query_has_coauthors"][local_index] = float(query.has_coauthors)
    row_signals["query_has_specter"][local_index] = float(
        bool(getattr(query, "has_specter", False)) and getattr(query, "specter", None) is not None
    )
    row_signals["query_has_name_counts"][local_index] = float(getattr(query, "name_counts", None) is not None)
    row_signals["candidate_has_affiliations"][local_index] = float(
        bool(summary.affiliation_counts) and summary.size > 0
    )
    row_signals["candidate_has_coauthors"][local_index] = float(bool(summary.coauthor_counts) and summary.size > 0)
    row_signals["candidate_has_specter_exemplars"][local_index] = float(
        bool(getattr(summary, "exemplar_vectors", ()) or ())
    )
    row_signals["candidate_has_name_counts"][local_index] = float(
        bool(getattr(summary, "name_counts_values", ()) or ())
    )
    row_signals["candidate_cluster_max_paper_author_count"][local_index] = float(
        getattr(summary, "max_paper_author_count", 0)
    )
    for signal_name, value in retrieval.raw_paper_evidence_features(query, summary).items():
        row_signals[signal_name][local_index] = float(value)
    row_signals["affiliation_overlap"][local_index] = round(
        float(counter_query_overlap(query.affiliation_terms, summary.affiliation_counts, summary.size)),
        6,
    )
    row_signals["coauthor_overlap"][local_index] = round(
        float(counter_query_overlap(query.coauthor_blocks, summary.coauthor_counts, summary.size)),
        6,
    )
    row_signals["year_compatibility"][local_index] = round(float(year_compatibility(query.year, summary)), 6)
    row_signals["specter_exemplar_similarity"][local_index] = round(
        float(specter_exemplar_similarity(query, summary)),
        6,
    )
    row_signals["min_distance"][local_index] = round(_finite_distance(stats.min_distance), 6)
    row_signals["mean_distance"][local_index] = round(_finite_distance(stats.mean_distance), 6)
    row_signals["top3_mean_distance"][local_index] = round(_finite_distance(stats.topk_mean_distance(3)), 6)
    row_signals["top5_mean_distance"][local_index] = round(_finite_distance(stats.topk_mean_distance(5)), 6)
    row_signals["pair_count"][local_index] = float(stats.count)
    for signal_name in (
        "last_name_count_min_rarity",
        "candidate_last_name_count_min_rarity",
        "last_first_name_count_min_rarity",
    ):
        row_signals[signal_name][local_index] = float(rarity.get(signal_name, 0.0) or 0.0)


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
        raise ValueError(f"Raw feature materialization left unfilled row signals: {missing}")


def _bool_row_signal(row_signals: Mapping[str, Any], name: str, row_count: int) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing row signal required for semantic NaN policy: {name}")
    values = np.asarray(row_signals[name], dtype=np.float32)
    if values.shape != (row_count,):
        raise ValueError(f"Row signal {name!r} must have shape ({row_count},), got {values.shape}")
    return values > 0.0


def _normalized_alpha_present_signal(
    row_signals: Mapping[str, Any],
    name: str,
    row_count: int,
    *,
    min_length: int = 1,
) -> np.ndarray:
    if name not in row_signals:
        raise KeyError(f"Missing row signal required for semantic NaN policy: {name}")
    values = np.asarray(row_signals[name], dtype=object)
    if values.shape != (row_count,):
        raise ValueError(f"Row signal {name!r} must have shape ({row_count},), got {values.shape}")
    present = np.zeros(row_count, dtype=bool)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        normalized = s2and_text.normalize_text(str(value), special_case_apostrophes=True)
        present[index] = len(normalized) >= min_length
    return present


def _singleton_query_group_mask(candidate_batch: LinkerCandidateBatch) -> np.ndarray:
    row_count = int(candidate_batch.row_count)
    if candidate_batch.row_query_signature_indices is None:
        return np.zeros(row_count, dtype=bool)
    query_indices = np.asarray(candidate_batch.row_query_signature_indices, dtype=np.uint32)
    if query_indices.shape != (row_count,):
        raise ValueError(f"row_query_signature_indices must have shape ({row_count},), got {query_indices.shape}")
    _unique, inverse, counts = np.unique(query_indices, return_inverse=True, return_counts=True)
    return counts[inverse] <= 1


def _semantic_row_nan_masks(
    row_signals: Mapping[str, Any],
    candidate_batch: LinkerCandidateBatch,
) -> dict[str, np.ndarray]:
    row_count = int(candidate_batch.row_count)
    pair_count = np.asarray(row_signals["pair_count"], dtype=np.float32)
    if pair_count.shape != (row_count,):
        raise ValueError(f"pair_count must have shape ({row_count},), got {pair_count.shape}")

    distance_missing = pair_count <= 0.0
    competitor_missing = _singleton_query_group_mask(candidate_batch)
    query_year_missing = np.asarray(row_signals["query_year_missing"], dtype=np.float32) > 0.0
    candidate_year_range_missing = np.asarray(row_signals["candidate_year_range_missing"], dtype=np.float32) > 0.0
    query_has_affiliations = _bool_row_signal(row_signals, "query_has_affiliations", row_count)
    query_has_coauthors = _bool_row_signal(row_signals, "query_has_coauthors", row_count)
    query_has_specter = _bool_row_signal(row_signals, "query_has_specter", row_count)
    query_has_name_counts = _bool_row_signal(row_signals, "query_has_name_counts", row_count)
    candidate_has_affiliations = _bool_row_signal(row_signals, "candidate_has_affiliations", row_count)
    candidate_has_coauthors = _bool_row_signal(row_signals, "candidate_has_coauthors", row_count)
    candidate_has_specter_exemplars = _bool_row_signal(row_signals, "candidate_has_specter_exemplars", row_count)
    candidate_has_name_counts = _bool_row_signal(row_signals, "candidate_has_name_counts", row_count)
    candidate_dominant_first_available = _normalized_alpha_present_signal(
        row_signals,
        "dominant_first_name",
        row_count,
    )
    query_name_count_missing = ~query_has_name_counts
    candidate_name_count_missing = ~candidate_has_name_counts
    name_count_missing = query_name_count_missing | candidate_name_count_missing
    query_first_any_available = _normalized_alpha_present_signal(
        row_signals,
        "query_first_token",
        row_count,
        min_length=1,
    )
    first_name_comparison_missing = ~query_first_any_available | ~candidate_dominant_first_available

    distance_available = ~distance_missing
    competitor_available = ~competitor_missing
    year_comparison_missing = query_year_missing | candidate_year_range_missing
    affiliation_comparison_missing = ~(query_has_affiliations & candidate_has_affiliations)
    coauthor_comparison_missing = ~(query_has_coauthors & candidate_has_coauthors)
    specter_comparison_missing = ~(query_has_specter & candidate_has_specter_exemplars)
    anchor_support_missing = ~(distance_available | competitor_available)
    strong_support_missing = distance_missing
    residual_support_missing = ~(distance_available | competitor_available)
    return {
        "min_distance": distance_missing,
        "retrieval_reciprocal_rank": np.zeros(row_count, dtype=bool),
        "specter_exemplar_similarity": specter_comparison_missing,
        "coauthor_overlap": coauthor_comparison_missing,
        "affiliation_overlap": affiliation_comparison_missing,
        "year_compatibility": year_comparison_missing,
        "candidate_year_span": candidate_year_range_missing,
        "year_gap_to_candidate_range": year_comparison_missing,
        "year_gap_signed_to_candidate_range": year_comparison_missing,
        "affiliation_contradiction_severity": ~query_has_affiliations,
        "same_dominant_first_as_best_top5": first_name_comparison_missing,
        "same_family_as_heuristic_choice": first_name_comparison_missing | distance_missing,
        "query_first_prefix_match_any_length": first_name_comparison_missing,
        "anchor_evidence_count": anchor_support_missing,
        "strong_positive_anchor_score": strong_support_missing,
        "weak_residual_anchor_score": residual_support_missing,
        "sparse_relative_winner_score": residual_support_missing,
        "last_name_count_min_rarity": name_count_missing,
        "last_first_name_count_min_rarity": name_count_missing,
        "top5_mean_distance": distance_missing,
        "cluster_size_log": np.zeros(row_count, dtype=bool),
        "candidate_dominant_first_name_length": ~candidate_dominant_first_available,
        "paper_author_list_max_jaccard": np.zeros(row_count, dtype=bool),
        "paper_author_list_max_containment": np.zeros(row_count, dtype=bool),
        "paper_author_list_max_overlap_count": np.zeros(row_count, dtype=bool),
        "local_author_window10_jaccard_max": np.zeros(row_count, dtype=bool),
        "local_author_window10_overlap_count_max": np.zeros(row_count, dtype=bool),
        "best_author_count_log_absdiff": np.zeros(row_count, dtype=bool),
        "candidate_cluster_max_paper_author_count": np.zeros(row_count, dtype=bool),
    }


def _apply_row_nan_policy(
    features: Mapping[str, np.ndarray],
    row_signals: Mapping[str, Any],
    candidate_batch: LinkerCandidateBatch,
    *,
    row_nan_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if row_nan_policy == "finite":
        return {str(column): np.asarray(values, dtype=np.float32) for column, values in features.items()}, {
            "row_nan_policy": "finite",
            "semantic_nan_counts": {},
            "semantic_nan_total": 0,
        }
    if row_nan_policy != "semantic":
        raise ValueError(f"Unsupported row_nan_policy: {row_nan_policy}")

    adjusted = {str(column): np.asarray(values, dtype=np.float32).copy() for column, values in features.items()}
    masks = _semantic_row_nan_masks(row_signals, candidate_batch)
    nan_counts: dict[str, int] = {}
    for column, mask in masks.items():
        if column not in adjusted:
            continue
        adjusted[column][mask] = np.nan
        nan_counts[column] = int(np.isnan(adjusted[column]).sum())
    return adjusted, {
        "row_nan_policy": "semantic",
        "semantic_nan_counts": nan_counts,
        "semantic_nan_total": int(sum(nan_counts.values())),
        "semantic_nan_feature_count": int(sum(count > 0 for count in nan_counts.values())),
    }


def _query_first_token_for_prefix(group: pd.DataFrame, base_query: retrieval.QueryFeatures) -> str:
    if "query_author" in group.columns:
        for value in group["query_author"].tolist():
            token = _query_first_token(value)
            if token:
                return token
    first = getattr(base_query, "first", None)
    if first:
        return str(first)
    if "query_first_token" in group.columns:
        for value in group["query_first_token"].tolist():
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                token = str(value).strip()
                if token:
                    return token
    return ""


def _pairwise_feature_values(pairwise_stats: Any) -> dict[str, np.ndarray]:
    pairwise_columns = tuple(pairwise_stats.aggregate_feature_columns)
    if pairwise_columns != PROMOTED_PAIRWISE_COLUMNS:
        raise ValueError("Rust pairwise aggregate column order mismatch in minimal raw materialization")
    pairwise_matrix = pairwise_stats.feature_matrix().astype(np.float32, copy=False)
    return {
        column: np.asarray(pairwise_matrix[:, column_index], dtype=np.float32)
        for column_index, column in enumerate(pairwise_columns)
    }


def _assemble_minimal_raw_feature_values(
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


def _materialize_minimal_raw_dataset_rows(
    *,
    context: MinimalRawDatasetContext,
    rows: pd.DataFrame,
    target_features: Sequence[str],
    clusterer: Any,
    n_jobs: int,
    total_ram_bytes: int,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_exemplars: int,
    max_top_k: int,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    row_nan_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    dataset_name = context.dataset_name
    dataset = context.dataset
    runtime_context = context.runtime_context
    constraint_backend = context.constraint_backend
    featurizer = context.featurizer
    signature_id_to_index = context.signature_id_to_index
    component_details = context.component_details
    component_indices = context.component_indices
    feature_cache = context.feature_cache
    full_summary_cache = context.full_summary_cache
    residual_summary_cache = context.residual_summary_cache

    dataset_rows = rows.reset_index(drop=True).copy()
    row_count = len(dataset_rows)
    row_signals = _initialize_row_signal_arrays(row_count, dataset_rows)
    group_codes = tuple(
        int(value) for value in pd.factorize(dataset_rows["query_group_id"].astype(str), sort=False)[0].tolist()
    )
    row_seed_bypass = np.zeros(row_count, dtype=bool)
    row_ignore_disallow = np.zeros(row_count, dtype=bool)
    seed_constraint_signature_ids = (
        _seed_constrained_signature_ids(dataset) if _dataset_has_cluster_seed_constraints(dataset) else frozenset()
    )
    batch = _candidate_batch_from_rows(
        dataset_rows,
        component_indices,
        signature_id_to_index,
        row_group_ids=group_codes,
    )

    component_cluster_ids = (
        dataset_rows[["candidate_component_key", "candidate_cluster_id"]]
        .drop_duplicates("candidate_component_key")
        .set_index("candidate_component_key")["candidate_cluster_id"]
        .astype(str)
        .to_dict()
    )

    def summary_for(component_key: str, query_signature_id: str | None) -> tuple[retrieval.ClusterSummary, list[str]]:
        details = component_details[str(component_key)]
        member_ids = list(details.signature_ids)
        if query_signature_id is not None and str(query_signature_id) in details.signature_id_set:
            cache_key = (str(component_key), str(query_signature_id))
            summary = residual_summary_cache.get(cache_key)
            active_member_ids = [signature_id for signature_id in member_ids if signature_id != str(query_signature_id)]
            if summary is None:
                summary = _build_summary_for_members(
                    dataset=dataset,
                    component_key=str(component_key),
                    candidate_cluster_id=component_cluster_ids.get(str(component_key)),
                    signature_ids=active_member_ids,
                    feature_cache=feature_cache,
                    paper_author_name_cache=context.paper_author_name_cache,
                    max_exemplars=max_exemplars,
                )
                residual_summary_cache[cache_key] = summary
            return summary, active_member_ids

        summary = full_summary_cache.get(str(component_key))
        if summary is None:
            summary = _build_summary_for_members(
                dataset=dataset,
                component_key=str(component_key),
                candidate_cluster_id=component_cluster_ids.get(str(component_key)),
                signature_ids=member_ids,
                feature_cache=feature_cache,
                paper_author_name_cache=context.paper_author_name_cache,
                max_exemplars=max_exemplars,
            )
            full_summary_cache[str(component_key)] = summary
        return summary, member_ids

    contexts: list[dict[str, Any]] = []

    for query_group_id, group in dataset_rows.groupby(dataset_rows["query_group_id"].astype(str), sort=False):
        query_signature_ids = set(group["query_signature_id"].astype(str))
        query_views = set(group["query_view"].astype(str))
        if len(query_signature_ids) != 1 or len(query_views) != 1:
            raise ValueError(f"{dataset_name}: query group {query_group_id!r} is not a single query/view")
        query_signature_id = next(iter(query_signature_ids))
        query_view = next(iter(query_views))
        base_query = retrieval.extract_query_features(
            dataset,
            query_signature_id,
            feature_cache=feature_cache,
            paper_author_name_cache=context.paper_author_name_cache,
            orcid_enabled=False,
        )
        query_first_token_for_prefix = _query_first_token_for_prefix(group, base_query)
        query = retrieval.mask_query_features(base_query, query_view, orcid_enabled=False)
        sorted_group = group.sort_values(["retrieval_rank", "candidate_component_key"], kind="stable")
        summaries: dict[str, retrieval.ClusterSummary] = {}
        retrieval_ranks: dict[str, int] = {}
        rows_by_component: dict[str, list[int]] = {}
        for row in sorted_group.itertuples():
            row_any = cast(Any, row)
            row_index = int(row_any.Index)
            component_key = str(row_any.candidate_component_key)
            summary, _active_member_ids = summary_for(component_key, query_signature_id)
            summaries[component_key] = summary
            retrieval_ranks[component_key] = int(row_any.retrieval_rank)
            rows_by_component.setdefault(component_key, []).append(row_index)
            if _row_label_is_positive(row):
                row_ignore_disallow[row_index] = True
            if (
                seed_constraint_signature_ids
                and _row_allows_seed_constraint_bypass(
                    dataset,
                    row,
                    seed_constraint_signature_ids=seed_constraint_signature_ids,
                )
                and _has_query_seed_connection(
                    dataset,
                    query_signature_id=str(query_signature_id),
                    candidate_signature_ids=_active_member_ids,
                )
            ):
                row_seed_bypass[row_index] = True
        retrieval_scores = _score_candidate_summaries_with_frozen_rust_policy(
            query=query,
            summaries=summaries,
            retriever=context.rust_hybrid_centroid_retriever,
            n_jobs=n_jobs,
        )
        current_retrieval_ranks = _current_retrieval_ranks_from_scores(retrieval_scores, retrieval_ranks)
        contexts.append(
            {
                "query": query,
                "query_first_token_for_prefix": query_first_token_for_prefix,
                "retrieval_scores": retrieval_scores,
                "retrieval_ranks": current_retrieval_ranks,
                "summaries": summaries,
                "rows_by_component": rows_by_component,
            }
        )

    index_to_signature_id = {int(index): str(signature_id) for signature_id, index in signature_id_to_index.items()}
    constraint_chunk_size = max(1, min(int(pair_batch_size), int(query_batch_pair_limit)))
    pair_labels, constraint_summary = _resolve_candidate_batch_pair_labels(
        clusterer=clusterer,
        dataset=dataset,
        batch=batch,
        index_to_signature_id=index_to_signature_id,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        chunk_size=constraint_chunk_size,
        pair_seed_bypass=row_seed_bypass[batch.pair_row_indices],
        pair_ignore_disallow=row_seed_bypass[batch.pair_row_indices] | row_ignore_disallow[batch.pair_row_indices],
    )
    fused_pairwise_started = time.perf_counter()
    fused_pairwise = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
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
        runtime_context=runtime_context,
        featurizer=featurizer,
    )
    fused_pairwise_seconds = float(time.perf_counter() - fused_pairwise_started)
    for query_context in contexts:
        query = query_context["query"]
        query_first_token_for_prefix = query_context["query_first_token_for_prefix"]
        retrieval_scores = query_context["retrieval_scores"]
        retrieval_ranks = query_context["retrieval_ranks"]
        summaries = query_context["summaries"]
        rows_by_component = query_context["rows_by_component"]
        for component_key, local_indices in rows_by_component.items():
            for local_index in local_indices:
                stats = FusedDistanceStats(
                    count=int(fused_pairwise.row_signals["pair_count"][local_index]),
                    min_distance=float(fused_pairwise.row_signals["min_distance"][local_index]),
                    mean_distance=float(fused_pairwise.row_signals["mean_distance"][local_index]),
                    top3_mean_distance=float(fused_pairwise.row_signals["top3_mean_distance"][local_index]),
                    top5_mean_distance=float(fused_pairwise.row_signals["top5_mean_distance"][local_index]),
                )
                _fill_row_signal(
                    row_signals=row_signals,
                    local_index=int(local_index),
                    component_key=str(component_key),
                    query=query,
                    summary=summaries[str(component_key)],
                    stats=stats,
                    retrieval_rank=int(retrieval_ranks[str(component_key)]),
                    retrieval_score=float(retrieval_scores[str(component_key)]),
                    query_first_token_for_prefix=str(query_first_token_for_prefix),
                )
    _validate_row_signals(row_signals)

    non_pairwise_started = time.perf_counter()
    non_pairwise_features = build_promoted_non_pairwise_row_features(batch, row_signals)
    non_pairwise_features, row_nan_summary = _apply_row_nan_policy(
        non_pairwise_features,
        row_signals,
        batch,
        row_nan_policy=str(row_nan_policy),
    )
    non_pairwise_seconds = float(time.perf_counter() - non_pairwise_started)
    feature_values = _assemble_minimal_raw_feature_values(
        target_features=target_features,
        non_pairwise_features=non_pairwise_features,
        pairwise_stats=fused_pairwise.pairwise_stats,
    )
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
        "full_summary_cache_size": int(len(full_summary_cache)),
        "residual_summary_cache_size": int(len(residual_summary_cache)),
        "retrieval_policy": FROZEN_RETRIEVAL_POLICY_NAME,
        "retrieval_max_block_component_size": int(context.max_block_component_size),
        "specter_embeddings": int(len(dataset.specter_embeddings or {})),
        "pairwise_model_nan_value": "nan"
        if math.isnan(float(pairwise_model_nan_value))
        else float(pairwise_model_nan_value),
        "pairwise_aggregate_nan_value": (
            "nan" if math.isnan(float(pairwise_aggregate_nan_value)) else float(pairwise_aggregate_nan_value)
        ),
        **row_nan_summary,
        **constraint_summary,
        "fused_pairwise_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_seconds": round(fused_pairwise_seconds, 3),
        "pairwise_model_featurize_seconds": round(float(fused_pairwise.telemetry["feature_seconds"]), 3),
        "pairwise_model_predict_seconds": round(float(fused_pairwise.telemetry["predict_seconds"]), 3),
        "non_pairwise_formula_seconds": round(non_pairwise_seconds, 3),
        "rust_pairwise_aggregate_seconds": 0.0,
        "seconds": round(float(time.perf_counter() - started), 3),
    }
    del pair_labels, fused_pairwise
    gc.collect()
    return feature_values, summary


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
    row_nan_policy: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    dataset_name = context.dataset_name
    dataset_rows = rows.reset_index(drop=True).copy()
    row_count = len(dataset_rows)
    group_codes = tuple(
        int(value) for value in pd.factorize(dataset_rows["query_group_id"].astype(str), sort=False)[0].tolist()
    )
    rust_module = feature_port._require_rust_runtime()  # noqa: SLF001
    plan_fn = getattr(rust_module, "raw_arrow_labeled_candidate_plan", None)
    if not callable(plan_fn):
        raise RuntimeError(
            "s2and_rust.raw_arrow_labeled_candidate_plan is required for --feature-mode arrow-rust; "
            "rebuild/install the current s2and-rust extension."
        )
    plan_started = time.perf_counter()
    retrieval_ranks = as_retrieval_rank_uint16_1d(
        "retrieval_rank",
        pd.to_numeric(dataset_rows["retrieval_rank"], errors="raise").to_numpy(),
    )
    raw_plan = plan_fn(
        context.arrow_paths,
        dataset_rows["query_signature_id"].astype(str).tolist(),
        dataset_rows["query_view"].astype(str).tolist(),
        dataset_rows["query_group_id"].astype(str).tolist(),
        dataset_rows["candidate_component_key"].astype(str).tolist(),
        retrieval_ranks.tolist(),
        context.component_members,
        orcid_enabled=False,
        num_threads=max(1, int(n_jobs)),
        max_exemplars=int(max_exemplars),
    )
    raw_plan_seconds = float(time.perf_counter() - plan_started)
    signature_ids = tuple(str(signature_id) for signature_id in raw_plan["signature_ids"])
    featurizer_started = time.perf_counter()
    featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
        context.arrow_paths,
        expected_normalization_version=NORMALIZATION_VERSION,
        signature_ids=signature_ids,
        name_tuples="filtered",
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
    non_pairwise_features = build_promoted_non_pairwise_row_features(batch, row_signals)
    non_pairwise_features, row_nan_summary = _apply_row_nan_policy(
        non_pairwise_features,
        row_signals,
        batch,
        row_nan_policy=str(row_nan_policy),
    )
    non_pairwise_seconds = float(time.perf_counter() - non_pairwise_started)
    feature_values = _assemble_minimal_raw_feature_values(
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
        "retrieval_policy": FROZEN_RETRIEVAL_POLICY_NAME,
        "retrieval_max_block_component_size": int(context.max_block_component_size),
        "specter_embeddings": int(raw_plan_telemetry.get("specter_count", 0) or 0),
        "pairwise_model_nan_value": "nan"
        if math.isnan(float(pairwise_model_nan_value))
        else float(pairwise_model_nan_value),
        "pairwise_aggregate_nan_value": (
            "nan" if math.isnan(float(pairwise_aggregate_nan_value)) else float(pairwise_aggregate_nan_value)
        ),
        **row_nan_summary,
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


def _write_minimal_raw_partial(
    *,
    shard: MinimalRawPendingShard,
    dataset_features: Mapping[str, np.ndarray],
    target_features: Sequence[str],
) -> None:
    _write_minimal_raw_partial_frame(
        rows=shard.rows,
        row_positions=shard.row_positions,
        partial_path=shard.partial_path,
        dataset_features=dataset_features,
        target_features=target_features,
    )


def _write_minimal_raw_partial_frame(
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


def _finalize_minimal_raw_table_plan(
    *,
    plan: MinimalRawTablePlan,
    target_features: Sequence[str],
    source_bundle: OfficialBundle,
    mode_label: str = "arrow-rust",
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
        "mode": mode_label,
    }


def _finalize_minimal_raw_bundle_metadata(
    *,
    source_bundle: OfficialBundle,
    output_bundle_root: Path,
    target: Mapping[str, Any],
    selected_keys: Sequence[str],
    stamp_precomputed_metadata: bool,
    source_mode: str = "arrow-rust",
) -> OfficialBundle:
    payload = json.loads((output_bundle_root / "bundle.json").read_text(encoding="utf-8"))
    feature_count = int(target["feature_count"])
    tree_count = int(target["params"]["n_estimators"])
    payload["bundle_name"] = (
        f"{payload['bundle_name']}_minimal_raw_block_local_promoted_{feature_count}_{tree_count}trees"
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
        relpath = str(_output_table_relpath(table_key, labels_path))
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
    if stamp_precomputed_metadata:
        _stamp_precomputed_promoted_bundle_metadata(
            output_bundle_root=output_bundle_root,
            target=target,
            source_mode=source_mode,
        )
    return _bundle_with_promoted_target(load_bundle(output_bundle_root), target)


def _materialize_minimal_raw_feature_bundle(
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
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_exemplars: int,
    max_top_k: int,
    reuse_existing_features: bool,
    pairwise_model_nan_value: float,
    pairwise_aggregate_nan_value: float,
    row_nan_policy: str,
    feature_mode: str = "arrow-rust",
    name_counts_index_root: Path | None = None,
) -> tuple[OfficialBundle, list[dict[str, Any]]]:
    _copy_bundle_support_files(
        source_bundle,
        output_bundle_root,
        reuse_existing_features=reuse_existing_features,
    )
    table_key_set = set(table_keys) if table_keys is not None else None
    selected_keys = [
        table_key
        for table_key in _source_featureless_table_keys(source_bundle)
        if table_key_set is None or table_key in table_key_set
    ]
    mode_label = str(feature_mode)
    materialized_keys: list[str] = []
    summaries: list[dict[str, Any]] = []
    target_features = tuple(str(feature) for feature in target["features"])
    table_plans: dict[str, MinimalRawTablePlan] = {}
    table_plan_order: list[str] = []
    pending_by_dataset: dict[str, list[MinimalRawPendingShard]] = {}
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
            "mode": mode_label,
            "skipped": "empty_selection",
            "label_filtering": label_filtering_summary,
            "structural_cleaning": structural_cleaning_summary,
        }
        summaries.append(summary)
        print(json.dumps({"event": "minimal_raw_table_featureization_skipped", **summary}), flush=True)

    for table_key in selected_keys:
        labels_path = _asset_file(source_bundle, "featureless_rows", table_key)
        output_relpath = _output_table_relpath(table_key, labels_path)
        output_path = output_bundle_root / output_relpath
        print(
            json.dumps(
                {
                    "event": "minimal_raw_table_featureization_start",
                    "table_key": table_key,
                    "output_path": str(output_path),
                }
            ),
            flush=True,
        )
        labels = pd.read_parquet(labels_path)
        positions = _selected_row_positions(labels, datasets, limit_rows)
        labels = labels.iloc[positions].reset_index(drop=True)
        labels, label_filtering_summary = _drop_unlabeled_singleton_orcid_rows(
            labels,
            context=f"{mode_label}:{table_key}",
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
        if mode_label == "arrow-rust":
            labels, structural_cleaning_summary = _clean_arrow_rust_structural_rows(
                source_bundle=source_bundle,
                table_key=table_key,
                rows=labels,
                component_membership_cache=component_membership_cache,
                name_counts_index_root=name_counts_index_root,
            )
        else:
            labels, structural_cleaning_summary = _clean_minimal_raw_structural_rows(
                source_bundle=source_bundle,
                table_key=table_key,
                rows=labels,
                component_membership_cache=component_membership_cache,
            )
        required_output_columns = _required_materialized_output_columns(labels, target_features)
        if labels.empty:
            append_empty_selection_summary(
                table_key=table_key,
                labels_path=labels_path,
                output_path=output_path,
                label_filtering_summary=label_filtering_summary,
                structural_cleaning_summary=structural_cleaning_summary,
            )
            continue
        if reuse_existing_features and output_path.exists():
            row_count = _validate_reusable_parquet(
                output_path,
                expected_rows=len(labels),
                required_columns=required_output_columns,
                context=f"{table_key} existing output",
            )
            summary = {
                "table_key": table_key,
                "labels_path": str(labels_path.relative_to(source_bundle.root)),
                "output_path": str(output_path),
                "rows": int(row_count),
                "datasets": [],
                "seconds": 0.0,
                "mode": mode_label,
                "reused": True,
                "label_filtering": label_filtering_summary,
                "structural_cleaning": structural_cleaning_summary,
            }
            summaries.append(summary)
            materialized_keys.append(table_key)
            print(json.dumps({"event": "minimal_raw_table_featureization_done", **summary}), flush=True)
            continue

        partial_dir = output_path.parent / "_partial" / output_path.stem
        if partial_dir.exists() and not reuse_existing_features:
            shutil.rmtree(partial_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)
        plan = MinimalRawTablePlan(
            table_key=table_key,
            labels_path=labels_path,
            output_path=output_path,
            labels=labels,
            required_output_columns=required_output_columns,
            partial_dir=partial_dir,
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
            if reuse_existing_features and partial_path.exists():
                row_count = _validate_reusable_parquet(
                    partial_path,
                    expected_rows=len(dataset_rows),
                    required_columns=["_row_position", *required_output_columns],
                    context=f"{table_key} {dataset_name} partial",
                )
                plan.dataset_summaries.append(
                    {
                        "dataset": dataset_name,
                        "rows": int(row_count),
                        "seconds": 0.0,
                        "mode": mode_label,
                        "reused": True,
                    }
                )
                plan.partial_paths.append(partial_path)
                print(
                    json.dumps(
                        {
                            "event": "minimal_raw_dataset_featureization_reused",
                            "table_key": table_key,
                            "dataset": dataset_name,
                            "rows": int(row_count),
                            "partial_path": str(partial_path),
                        }
                    ),
                    flush=True,
                )
                continue
            pending_by_dataset.setdefault(dataset_name, []).append(
                MinimalRawPendingShard(
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
                    "event": "minimal_raw_dataset_context_start",
                    "mode": mode_label,
                    "dataset": dataset_name,
                    "shards": len(shards),
                    "rows": int(sum(len(shard.rows) for shard in shards)),
                    "tables": sorted({shard.table_key for shard in shards}),
                }
            ),
            flush=True,
        )
        if mode_label == "arrow-rust":
            context = _build_arrow_rust_dataset_context(
                source_bundle=source_bundle,
                dataset_name=dataset_name,
                name_counts_index_root=name_counts_index_root,
            )
        else:
            context = _build_minimal_raw_dataset_context(
                source_bundle=source_bundle,
                dataset_name=dataset_name,
                clusterer=clusterer,
                n_jobs=n_jobs,
                max_exemplars=max_exemplars,
            )
        try:
            for shard in shards:
                print(
                    json.dumps(
                        {
                            "event": "minimal_raw_dataset_featureization_start",
                            "mode": mode_label,
                            "table_key": shard.table_key,
                            "dataset": shard.dataset_name,
                            "rows": int(len(shard.rows)),
                        }
                    ),
                    flush=True,
                )
                if mode_label == "arrow-rust":
                    dataset_features, dataset_summary = _materialize_arrow_rust_dataset_rows(
                        context=cast(ArrowRustDatasetContext, context),
                        rows=shard.rows,
                        target_features=target_features,
                        clusterer=clusterer,
                        n_jobs=n_jobs,
                        total_ram_bytes=total_ram_bytes,
                        max_exemplars=max_exemplars,
                        pairwise_model_nan_value=float(pairwise_model_nan_value),
                        pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                        row_nan_policy=str(row_nan_policy),
                    )
                else:
                    dataset_features, dataset_summary = _materialize_minimal_raw_dataset_rows(
                        context=cast(MinimalRawDatasetContext, context),
                        rows=shard.rows,
                        target_features=target_features,
                        clusterer=clusterer,
                        n_jobs=n_jobs,
                        total_ram_bytes=total_ram_bytes,
                        pair_batch_size=pair_batch_size,
                        query_batch_pair_limit=query_batch_pair_limit,
                        max_exemplars=max_exemplars,
                        max_top_k=max_top_k,
                        pairwise_model_nan_value=float(pairwise_model_nan_value),
                        pairwise_aggregate_nan_value=float(pairwise_aggregate_nan_value),
                        row_nan_policy=str(row_nan_policy),
                    )
                _write_minimal_raw_partial(
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
                            "event": "minimal_raw_dataset_featureization_done",
                            "mode": mode_label,
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
            if mode_label == "arrow-rust":
                _release_arrow_rust_dataset_context(cast(ArrowRustDatasetContext, context))
            else:
                _release_minimal_raw_dataset_context(cast(MinimalRawDatasetContext, context))
            del context

    for table_key in table_plan_order:
        summary = _finalize_minimal_raw_table_plan(
            plan=table_plans[table_key],
            target_features=target_features,
            source_bundle=source_bundle,
            mode_label=mode_label,
        )
        summaries.append(summary)
        print(json.dumps({"event": "minimal_raw_table_featureization_done", **summary}), flush=True)

    _write_json(output_bundle_root / "featureization_summary.json", summaries)
    return (
        _finalize_minimal_raw_bundle_metadata(
            source_bundle=source_bundle,
            output_bundle_root=output_bundle_root,
            target=target,
            selected_keys=materialized_keys,
            stamp_precomputed_metadata=table_keys is None and datasets is None and limit_rows is None,
            source_mode=mode_label,
        ),
        summaries,
    )


def _classic_candidate_training_rows(rows: pd.DataFrame, *, retrieval_rank_limit: int) -> pd.DataFrame:
    required = {"query_group_id", "retrieval_rank", "label"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Classic training rows are missing required columns: {missing}")
    out, _filter_summary = _drop_unlabeled_singleton_orcid_rows(
        rows,
        context="prod_training_rows",
    )
    out["retrieval_rank"] = pd.to_numeric(out["retrieval_rank"], errors="coerce")
    out = out[out["retrieval_rank"] <= int(retrieval_rank_limit)].copy()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(np.int8)
    return out


def _prepare_prod_training_data(
    bundle: OfficialBundle,
    *,
    holdout_importance_weight: float,
    retrieval_rank_limit: int,
) -> ProdTrainingData:
    """Build final production rows: train plus stratified calibration rows."""

    if float(holdout_importance_weight) <= 0.0:
        raise ValueError("holdout_importance_weight must be positive")
    spec = dict(bundle.models["classic"])
    train_path = _resolve_path(bundle, spec["train_path"])
    train_df = _classic_candidate_training_rows(
        _read_csv(train_path, compression="gzip"),
        retrieval_rank_limit=int(retrieval_rank_limit),
    )
    holdout_query_group_ids, holdout_base_group_ids, holdout_sources = _read_classic_holdout_identity_sets(
        bundle,
        spec,
    )
    train_df, train_holdout_filter_summary = _apply_classic_train_holdout_filter(
        train_df,
        holdout_query_group_ids=holdout_query_group_ids,
        holdout_base_group_ids=holdout_base_group_ids,
        holdout_sources=holdout_sources,
    )
    train_df, train_filter_summary = _apply_classic_train_row_cap(
        train_df,
        rule_name=spec.get("train_row_cap_rule"),
        min_train_limit=(int(spec["train_row_cap_min_limit"]) if "train_row_cap_min_limit" in spec else None),
    )

    train_df = train_df.copy()
    train_df["_prod_source_kind"] = "train"
    train_df["_prod_importance_weight"] = 1.0
    train_df["_prod_source_path"] = str(train_path.relative_to(bundle.root))
    frames = [train_df]

    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required for final production training")
    if spec.get("stratified_eval_test_split") is None:
        raise ValueError("classic.stratified_eval_test_split is required for final production training")
    split_spec = dict(spec["stratified_eval_test_split"])
    calibration_splits = tuple(str(split) for split in promoted_gate_config["calibration_splits"])
    split_rows, _assignments = _load_classic_stratified_eval_rows(bundle, spec, split_spec)
    split_values = split_rows["split"].astype(str)
    calibration_rows = split_rows[split_values.isin(calibration_splits)].copy()
    if calibration_rows.empty:
        raise ValueError(f"Final production training requires non-empty calibration splits: {list(calibration_splits)}")
    calibration_rows = _classic_candidate_training_rows(
        calibration_rows,
        retrieval_rank_limit=int(retrieval_rank_limit),
    )
    calibration_rows["_prod_source_kind"] = "stratified_calibration_" + calibration_rows["split"].astype(str)
    calibration_rows["_prod_importance_weight"] = float(holdout_importance_weight)
    assignments_path = _resolve_path(bundle, str(split_spec["assignments_path"]))
    calibration_rows["_prod_source_path"] = str(assignments_path.relative_to(bundle.root))
    frames.append(calibration_rows)

    combined = pd.concat(frames, ignore_index=True)
    source_kinds = combined["_prod_source_kind"].astype(str)
    importance_weights = pd.to_numeric(combined["_prod_importance_weight"], errors="raise").to_numpy(dtype=np.float32)
    query_group_ids = combined["query_group_id"].astype(str)
    group_sizes = query_group_ids.value_counts(sort=False)
    base_weights = (1.0 / query_group_ids.map(group_sizes).astype(float)).to_numpy(dtype=np.float32)
    sample_weight = (base_weights * importance_weights).astype(np.float32)

    source_summaries: list[dict[str, Any]] = []
    for source_name, source_rows in combined.groupby("_prod_source_kind", sort=False):
        source_indices = source_rows.index.to_numpy(dtype=np.int64)
        labels = pd.to_numeric(source_rows["label"], errors="coerce").fillna(0).astype(np.int8)
        paths = sorted(set(source_rows["_prod_source_path"].astype(str)))
        weight_values = sorted(float(value) for value in set(source_rows["_prod_importance_weight"].astype(float)))
        source_summaries.append(
            {
                "source": str(source_name),
                "paths": paths,
                "rows": int(len(source_rows)),
                "queries": int(source_rows["query_group_id"].astype(str).nunique()),
                "positive_rows": int(labels.sum()),
                "importance_weights": weight_values,
                "sample_weight_sum": round(float(sample_weight[source_indices].sum()), 6),
                **({"splits": sorted(set(source_rows["split"].astype(str)))} if "split" in source_rows.columns else {}),
                **(
                    {"source_keys": sorted(set(source_rows["source_key"].astype(str)))}
                    if "source_key" in source_rows.columns
                    else {}
                ),
            }
        )

    model_rows = combined.drop(columns=["_prod_source_kind", "_prod_importance_weight", "_prod_source_path"])
    if len(model_rows) != len(source_kinds):
        raise RuntimeError("Production training row metadata length mismatch")
    return ProdTrainingData(
        rows=model_rows,
        sample_weight=sample_weight,
        source_summaries=source_summaries,
        train_holdout_filter_summary=train_holdout_filter_summary,
        train_filter_summary=train_filter_summary,
    )


def _train_and_save_prod_artifact(
    *,
    feature_bundle: OfficialBundle,
    classic_summary: Mapping[str, Any],
    output_dir: Path,
    save_artifact_to: Path,
    artifact_audit_metadata: Mapping[str, Any] | None,
    holdout_importance_weight: float,
) -> dict[str, Any]:
    spec = dict(feature_bundle.models["classic"])
    retrieval_top_k = int(spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))
    if retrieval_top_k <= 0:
        raise ValueError("classic.retrieval_top_k must be positive")
    feature_columns = tuple(str(feature) for feature in spec["feature_columns"])
    monotone_constraints = _resolve_classic_monotone_constraints(spec, feature_columns)
    prod_training_data = _prepare_prod_training_data(
        feature_bundle,
        holdout_importance_weight=float(holdout_importance_weight),
        retrieval_rank_limit=retrieval_top_k,
    )
    train_matrix = _classic_feature_matrix(prod_training_data.rows, feature_columns).to_numpy(dtype=np.float32)
    train_labels = prod_training_data.rows["label"].to_numpy(dtype=np.int8, copy=False)
    model = _build_classic_classifier(spec["best_params"], monotone_constraints=monotone_constraints)
    started = time.perf_counter()
    model.fit(train_matrix, train_labels, sample_weight=prod_training_data.sample_weight)
    train_seconds = float(time.perf_counter() - started)

    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required to train the logistic artifact gate")
    if spec.get("stratified_eval_test_split") is None:
        raise ValueError("classic.stratified_eval_test_split is required to train the logistic artifact gate")
    split_spec = dict(spec["stratified_eval_test_split"])
    stratified_scores = _score_classic_stratified_eval_test(
        feature_bundle,
        spec,
        split_spec,
        model,
        feature_columns,
    )
    logistic_gate_result = _fit_promoted_logistic_gate(
        stratified_scores.rows,
        stratified_scores.choices,
        stratified_scores.probabilities,
        calibration_splits=(str(promoted_gate_config["test_split"]),),
    )
    logistic_gate_config = dict(logistic_gate_result["gate_config"])
    booster_training_splits = [str(split) for split in promoted_gate_config["calibration_splits"]]
    gate_calibration_splits = [str(promoted_gate_config["test_split"])]

    audit_metadata = dict(artifact_audit_metadata or {})
    audit_metadata["prod_training"] = {
        "policy": "train_plus_calibration_weighted_test_calibrated_logistic_gate",
        "holdout_importance_weight": float(holdout_importance_weight),
        "retrieval_rank_limit": retrieval_top_k,
        "booster_training_splits": booster_training_splits,
        "gate_calibration_splits": gate_calibration_splits,
        "rows": int(len(prod_training_data.rows)),
        "positive_rows": int(train_labels.sum()),
        "sample_weight_sum": round(float(prod_training_data.sample_weight.sum()), 6),
        "sources": prod_training_data.source_summaries,
        "train_holdout_filter_summary": prod_training_data.train_holdout_filter_summary,
        "train_filter_summary": prod_training_data.train_filter_summary,
        "params": dict(spec["best_params"]),
        "logistic_gate": {
            "calibration_metrics": dict(logistic_gate_result["calibration_metrics"]),
            "split_metrics": dict(logistic_gate_result["split_metrics"]),
            "training_summary": dict(logistic_gate_config.get("training_summary", {})),
        },
    }
    artifact_metadata = save_incremental_linking_artifact(
        model,
        Path(save_artifact_to),
        feature_columns=feature_columns,
        retrieval_top_k=retrieval_top_k,
        gate_config=logistic_gate_config,
        prediction_fixture_matrix=train_matrix[:5],
        audit_metadata=audit_metadata,
    )
    summary = {
        "path": str(Path(save_artifact_to)),
        "schema_version": artifact_metadata.schema_version,
        "feature_schema_digest": artifact_metadata.feature_schema_digest,
        "production_contract_digest": artifact_metadata.production_contract_digest,
        "retrieval_stack_digest": artifact_metadata.retrieval_stack_digest,
        "training_summary": {
            "rows": int(len(prod_training_data.rows)),
            "queries": int(prod_training_data.rows["query_group_id"].astype(str).nunique()),
            "positive_rows": int(train_labels.sum()),
            "sample_weight_sum": round(float(prod_training_data.sample_weight.sum()), 6),
            "holdout_importance_weight": float(holdout_importance_weight),
            "booster_training_splits": booster_training_splits,
            "gate_calibration_splits": gate_calibration_splits,
            "elapsed_seconds": round(train_seconds, 6),
            "sources": prod_training_data.source_summaries,
            "train_holdout_filter_summary": prod_training_data.train_holdout_filter_summary,
            "train_filter_summary": prod_training_data.train_filter_summary,
        },
    }
    _write_json(output_dir / "prod_artifact_summary.json", summary)
    return summary


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
    missing = sorted(set(target_metrics) - set(observed))
    if missing:
        raise RuntimeError(f"Official promoted run is missing target metrics: {missing}")

    def require_finite(value: Any, *, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                require_finite(nested, path=f"{path}.{key}")
        elif isinstance(value, float | np.floating) and not math.isfinite(float(value)):
            raise RuntimeError(f"Official promoted metric {path} must be finite, got {value!r}")

    for key in target_metrics:
        require_finite(observed[key], path=str(key))
        require_finite(target_metrics[key], path=f"target.{key}")
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


def _resolve_hyperopt_evals(args: argparse.Namespace) -> int:
    if args.hyperopt_evals is not None:
        resolved = int(args.hyperopt_evals)
    elif bool(args.hyperopt):
        resolved = 25
    else:
        resolved = 0
    if resolved < 0:
        raise SystemExit("--hyperopt-evals must be non-negative")
    if bool(args.hyperopt) and resolved == 0:
        raise SystemExit("--hyperopt requires --hyperopt-evals > 0")
    return resolved


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.allow_metric_drift and (args.save_artifact_to is not None or args.save_production_bundle_to is not None):
        raise SystemExit(
            "--allow-metric-drift is diagnostic-only and cannot be combined with "
            "--save-artifact-to or --save-production-bundle-to"
        )
    if args.limit_rows is not None and int(args.limit_rows) <= 0:
        raise SystemExit("--limit-rows must be > 0")
    target = _load_target(args.target_json)
    output_dir = args.output_dir.resolve()
    production_bundle_dir = (
        args.save_production_bundle_to.resolve() if args.save_production_bundle_to is not None else None
    )
    if production_bundle_dir is not None:
        if production_bundle_dir.exists():
            raise SystemExit(f"--save-production-bundle-to must name a new directory: {production_bundle_dir}")
        if output_dir == production_bundle_dir or output_dir.is_relative_to(production_bundle_dir):
            raise SystemExit("--output-dir must be outside --save-production-bundle-to")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairwise_model_nan_value = _nan_value_from_policy(str(args.pairwise_model_nan_policy))
    pairwise_aggregate_nan_value = _nan_value_from_policy(str(args.pairwise_aggregate_nan_policy))
    feature_nan_policy = _feature_nan_policy_summary(args)
    if args.feature_mode == "arrow-rust":
        if args.limit_rows is None and not args.run_full:
            raise SystemExit(f"unbounded {args.feature_mode} feature materialization requires --run-full")
        if not args.materialize_only and (args.limit_rows is not None or args.tables or args.datasets):
            raise SystemExit(
                f"limited/table-filtered {args.feature_mode} materialization is smoke-only; pass --materialize-only"
            )
        source_bundle = load_bundle(args.source_bundle_root)
        clusterer = load_clusterer(args.pairwise_model_path, n_jobs=int(args.n_jobs))
        clusterer.use_cache = False
        _assert_pairwise_model_is_raw_bundle_compatible(clusterer, args.pairwise_model_path)
        pair_batch_size = int(args.pair_batch_size) if args.pair_batch_size is not None else int(clusterer.batch_size)
        feature_bundle_root = output_dir / f"{str(args.feature_mode).replace('-', '_')}_feature_bundle"
        feature_bundle, featureization_summaries = _materialize_minimal_raw_feature_bundle(
            source_bundle=source_bundle,
            output_bundle_root=feature_bundle_root,
            target=target,
            clusterer=clusterer,
            n_jobs=int(args.n_jobs),
            total_ram_bytes=int(args.total_ram_bytes),
            table_keys=_parse_tables(args.tables),
            datasets=_parse_datasets(args.datasets),
            limit_rows=args.limit_rows,
            pair_batch_size=pair_batch_size,
            query_batch_pair_limit=int(args.query_batch_pair_limit),
            max_exemplars=int(args.max_exemplars),
            max_top_k=int(args.max_top_k),
            reuse_existing_features=bool(args.reuse_existing_features),
            pairwise_model_nan_value=pairwise_model_nan_value,
            pairwise_aggregate_nan_value=pairwise_aggregate_nan_value,
            row_nan_policy=str(args.row_nan_policy),
            feature_mode=str(args.feature_mode),
            name_counts_index_root=(
                Path(args.arrow_name_counts_index_root) if args.arrow_name_counts_index_root is not None else None
            ),
        )
        if args.materialize_only:
            result = {
                "mode": args.feature_mode,
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
    elif args.feature_mode == "precomputed-promoted":
        if args.precomputed_feature_bundle_root is None:
            raise SystemExit("--feature-mode precomputed-promoted requires --precomputed-feature-bundle-root")
        if not args.run_full:
            raise SystemExit("precomputed-promoted train/calibrate/eval requires --run-full")
        if args.materialize_only:
            raise SystemExit("precomputed-promoted does not materialize features")
        if args.limit_rows is not None or args.tables or args.datasets:
            raise SystemExit("precomputed-promoted requires a complete validated feature bundle")
        # The precomputed table still belongs to one exact pairwise model.
        # Loading through the staging-only door rejects complete/legacy bundles
        # before training or artifact publication starts.
        load_clusterer(args.pairwise_model_path, n_jobs=int(args.n_jobs))
        feature_bundle, featureization_summaries = _load_precomputed_promoted_feature_bundle(
            bundle_root=args.precomputed_feature_bundle_root,
            target=target,
        )
    else:
        raise ValueError(f"Unknown feature mode: {args.feature_mode}")

    if args.materialize_only:
        raise SystemExit("materialize-only is only valid with a materializing feature mode")

    run_output_dir = output_dir / "classic"
    started = time.perf_counter()
    active_params = dict(feature_bundle.models["classic"]["best_params"])
    hyperopt_evals = _resolve_hyperopt_evals(args)
    hyperopt_summary: dict[str, Any] | None = None
    if hyperopt_evals > 0:
        active_params, hyperopt_summary = _run_classic_hyperopt(
            feature_bundle=feature_bundle,
            output_dir=output_dir / "classic_hyperopt",
            base_params=active_params,
            hyperopt_evals=int(hyperopt_evals),
            metric=str(args.hyperopt_metric),
            seed=int(args.hyperopt_seed),
            n_jobs=int(args.n_jobs),
        )
        feature_bundle = _bundle_with_classic_params(feature_bundle, active_params)

    save_artifact_to = args.save_artifact_to.resolve() if args.save_artifact_to is not None else None
    if production_bundle_dir is not None:
        if save_artifact_to is None:
            save_artifact_to = output_dir / "production_incremental_linker"
        if save_artifact_to == production_bundle_dir or save_artifact_to.is_relative_to(production_bundle_dir):
            raise SystemExit("--save-artifact-to must be outside --save-production-bundle-to")
    artifact_audit_metadata = (
        _linker_artifact_audit_metadata(
            args=args,
            target=target,
            feature_bundle=feature_bundle,
            featureization_summaries=featureization_summaries,
        )
        if save_artifact_to is not None
        else None
    )
    summary = run_classic(
        feature_bundle,
        run_output_dir,
        n_jobs=int(args.n_jobs),
    )
    observed = _observed_official_metrics(summary)
    deltas = _metric_deltas(observed, target)
    if not args.allow_metric_drift:
        _assert_no_metric_drift(observed, target)
    if artifact_audit_metadata is not None:
        artifact_audit_metadata = {
            **artifact_audit_metadata,
            "classic_train_calibrate_eval": {
                "summary_artifact": "not bundled; observed metrics are embedded in this metadata",
                "observed_metrics": observed,
                "metric_deltas": deltas,
            },
            "hyperopt": hyperopt_summary or {"enabled": False},
        }
    prod_artifact_summary = None
    production_bundle_summary = None
    if save_artifact_to is not None:
        prod_artifact_summary = _train_and_save_prod_artifact(
            feature_bundle=feature_bundle,
            classic_summary=summary,
            output_dir=output_dir,
            save_artifact_to=save_artifact_to,
            artifact_audit_metadata=artifact_audit_metadata,
            holdout_importance_weight=float(args.prod_holdout_importance_weight),
        )
    if production_bundle_dir is not None:
        if save_artifact_to is None:
            raise RuntimeError("production bundle finalization requires an incremental linker artifact path")
        production_bundle_summary = finalize_production_bundle(
            pairwise_bundle_dir=Path(args.pairwise_model_path),
            output_bundle_dir=production_bundle_dir,
            incremental_linker_artifact_dir=save_artifact_to,
            target_json=Path(args.target_json),
            bundle_version=args.production_bundle_version or production_version_from_bundle_dir(production_bundle_dir),
            pairwise_model_version=_version_from_production_model_path(Path(args.pairwise_model_path)),
            incremental_linker_version=str(args.linker_artifact_version).removeprefix("v"),
        )
    result = {
        "mode": args.feature_mode,
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
        "classic_summary_path": str(run_output_dir / "summary.json"),
        "hyperopt": hyperopt_summary or {"enabled": False},
        "feature_nan_policy": feature_nan_policy,
    }
    if save_artifact_to is not None:
        result["artifact_dir"] = str(save_artifact_to)
        result["artifact_summary"] = dict(prod_artifact_summary or {})
    if production_bundle_summary is not None:
        result["production_bundle_dir"] = str(production_bundle_summary.bundle_dir)
        result["production_bundle_summary"] = {
            "bundle_status": production_bundle_summary.bundle_status,
            "bundle_version": production_bundle_summary.bundle_version,
            "files": list(production_bundle_summary.files),
            "manifest_path": str(production_bundle_summary.manifest_path),
        }
    if args.feature_mode == "arrow-rust":
        result["component_scope"] = "block-local"
    if not args.allow_metric_drift:
        result["metric_drift_check"] = "passed"
    _write_json(output_dir / "run_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-root", type=Path, default=DEFAULT_SOURCE_BUNDLE_ROOT)
    parser.add_argument("--target-json", type=Path, required=True, help="Explicit promoted-linker target JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pairwise-model-path",
        type=Path,
        required=True,
        help="Explicit v2 pairwise_only native bundle produced by train_pairwise.py.",
    )
    parser.add_argument("--save-artifact-to", type=Path, default=None)
    parser.add_argument(
        "--save-production-bundle-to",
        type=Path,
        default=None,
        help="Publish a complete production_model_vX.Y bundle to a new directory.",
    )
    parser.add_argument(
        "--production-bundle-version",
        default=None,
        help="Version recorded in the finalized production bundle; inferred from production_model_vX.Y if omitted.",
    )
    parser.add_argument("--linker-artifact-version", default="v1.2")
    parser.add_argument(
        "--prod-holdout-importance-weight",
        type=float,
        default=10.0,
        help="Final production fit multiplier for calibration/eval candidate rows.",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperopt over the initial train/calibrate/eval stack before the final production fit.",
    )
    parser.add_argument(
        "--hyperopt-evals",
        type=int,
        default=None,
        help="Number of train/calibrate/eval hyperopt trials. Passing a value enables hyperopt.",
    )
    parser.add_argument(
        "--hyperopt-metric",
        choices=(
            "weighted_average_error",
            "stratified_test_errors",
            "stratified_test_error_rate",
            "stratified_test_balanced_accuracy",
        ),
        default="weighted_average_error",
        help=(
            "Metric optimized by hyperopt. weighted_average_error uses "
            "0.25*false_abstain_error_rate, 1.0*false_link_error_rate, "
            "and 1.5*wrong_link_error_rate, divided by the total weight."
        ),
    )
    parser.add_argument("--hyperopt-seed", type=int, default=13)
    parser.add_argument(
        "--feature-mode",
        choices=("arrow-rust", "precomputed-promoted"),
        default="arrow-rust",
        help="Feature source for the official train/calibrate/eval run.",
    )
    parser.add_argument(
        "--arrow-name-counts-index-root",
        type=Path,
        default=DEFAULT_NAME_COUNTS_INDEX_ROOT,
        help=(
            "Optional override for the Arrow name_counts_index root used with --feature-mode arrow-rust. "
            "By default, each Arrow dataset manifest is the authority."
        ),
    )
    parser.add_argument(
        "--precomputed-feature-bundle-root",
        type=Path,
        default=None,
        help=(
            "Explicit portable precomputed promoted feature bundle root. Required only with "
            "--feature-mode precomputed-promoted."
        ),
    )
    parser.add_argument(
        "--pairwise-model-nan-policy",
        choices=NAN_POLICY_CHOICES,
        default="preserve",
        help=(
            "Missing-value policy for pairwise model feature matrices in minimal raw materialization. "
            "The production default preserves NaNs for the pairwise distance model internals."
        ),
    )
    parser.add_argument(
        "--pairwise-aggregate-nan-policy",
        choices=NAN_POLICY_CHOICES,
        default="zero",
        help=(
            "Missing-value policy for promoted pw_* aggregates. The production default reproduces "
            "prod12 dense zero-filled pairwise semantics; preserve uses nan-aware aggregation."
        ),
    )
    parser.add_argument(
        "--row-nan-policy",
        choices=ROW_NAN_POLICY_CHOICES,
        default="finite",
        help="Missing-value policy for promoted non-pw row features.",
    )
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-bytes", type=int, default=DEFAULT_TOTAL_RAM_BYTES)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--query-batch-pair-limit", type=int, default=200_000)
    parser.add_argument("--max-exemplars", type=int, default=4)
    parser.add_argument("--max-top-k", type=int, default=DEFAULT_CHOOSER_CACHE_MAX_TOP_K)
    parser.add_argument(
        "--tables", nargs="*", help="Optional table keys to materialize in feature rematerialization modes."
    )
    parser.add_argument("--datasets", nargs="*", help="Optional dataset slugs to keep when materializing smoke checks.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional per-table row limit for smoke checks.")
    parser.add_argument("--materialize-only", action="store_true", help="Stop after Rust feature materialization.")
    parser.add_argument(
        "--reuse-existing-features",
        action="store_true",
        help="Reuse already materialized output tables and dataset partials in the output directory.",
    )
    parser.add_argument("--run-full", action="store_true", help="Explicitly allow an unbounded official run.")
    parser.add_argument(
        "--allow-metric-drift",
        action="store_true",
        help="Diagnostic-only metric comparison override; incompatible with either artifact promotion option.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
