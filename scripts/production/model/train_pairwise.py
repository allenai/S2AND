"""Train the pairwise half of a native production model bundle.

This replaces the historical pickle dump flow. The output directory is a
pairwise-only ``production_model_vX.Y`` bundle stage. Run
``train_linker_and_finalize.py`` next with this stage as input to publish the
complete model into a separate, previously nonexistent directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from hyperopt import hp
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.consts import _PACKAGE_DATA_DIR, FEATURIZER_VERSION, NAME_COUNTS_INDEX_PATH  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.feature_cache import cached_featurize  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    TupleOfArrays,
    featurize,
    many_pairs_featurize,
    resolve_selection_pairs,
)
from s2and.model import Clusterer, FastCluster, PairwiseModeler  # noqa: E402
from s2and.name_counts_index import NameCountsIndex  # noqa: E402
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact  # noqa: E402
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts  # noqa: E402
from s2and.production_bundle import production_version_from_bundle_dir, write_pairwise_production_bundle  # noqa: E402

logger = logging.getLogger("s2and")

DEFAULT_SPECTER_SUFFIX = "_specter2.pkl"
DEFAULT_SIGNATURES_SUFFIX = "_signatures.json"
DEFAULT_SOURCE_DATASET_NAMES = ("aminer", "arnetminer", "inspire", "kisti", "orcid", "pubmed", "qian", "zbmath")
PAIRWISE_ONLY_DATASETS = frozenset({"medline", "augmented"})
DEFAULT_TRAIN_PAIRS_SIZE = 100_000
DEFAULT_VAL_TEST_SIZE = 10_000
DEFAULT_N_ITER = 50
DEFAULT_N_JOBS = 25
DEFAULT_CHUNK_SIZE = 100
DEFAULT_RANDOM_SEED = 1111


@dataclass(frozen=True)
class DatasetInputPlan:
    """Resolved immutable input files for one production training dataset."""

    files: Mapping[str, Path]
    sha256: Mapping[str, str]


@dataclass(frozen=True)
class PairwisePreflightPlan:
    """Validated, read-only launch plan created before artifact or dataset loading."""

    output_dir: Path
    dataset_names: tuple[str, ...]
    datasets: Mapping[str, DatasetInputPlan]
    feature_cache_dir: Path | None
    matrix_work_dir: Path
    total_ram_bytes: int | None


def _sha256_file(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _hash_source_files(source_files: Mapping[str, Path]) -> dict[str, str]:
    return {role: _sha256_file(path) for role, path in source_files.items()}


def _canonical_training_artifact_hashes(name_tuples_data_sha256: str) -> dict[str, str]:
    """Validate canonical count artifacts and return their immutable bindings."""

    orcid_counts = load_canonical_orcid_prefix_counts(_PACKAGE_DATA_DIR)
    if not orcid_counts.source_kind.startswith("redshift:"):
        raise RuntimeError(
            "production pairwise training requires warehouse-generated ORCID prefix counts; "
            f"observed source_kind={orcid_counts.source_kind!r}"
        )
    if orcid_counts.name_tuples_sha256 != name_tuples_data_sha256:
        raise RuntimeError("ORCID prefix counts were generated from a different canonical name-tuple artifact")
    name_counts_index = NameCountsIndex.open(NAME_COUNTS_INDEX_PATH)
    source_kind = name_counts_index.source_provenance.get("source_kind")
    if not isinstance(source_kind, str) or not source_kind.startswith("redshift:"):
        raise RuntimeError(
            "production pairwise training requires a warehouse-generated name-count index; "
            f"observed source_kind={source_kind!r}"
        )
    return {
        "name_tuples_data_sha256": name_tuples_data_sha256,
        "name_counts_manifest_sha256": name_counts_index.manifest_sha256,
        "orcid_prefix_counts_data_sha256": orcid_counts.data_sha256,
        "orcid_prefix_counts_manifest_sha256": orcid_counts.manifest_sha256,
    }


def _search_space() -> dict[str, Any]:
    return {
        "eps": hp.uniform("eps", 0, 1),
        "linkage": hp.choice("linkage", ["average"]),
    }


def _resolve_dataset_file(data_dir: Path, dataset_name: str, *candidates: str) -> Path:
    dataset_dir = data_dir / dataset_name
    for candidate in candidates:
        path = dataset_dir / candidate
        if path.is_file():
            return path.resolve()
    joined = ", ".join(str(dataset_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any dataset file for {dataset_name}: {joined}")


def _optional_dataset_file(data_dir: Path, dataset_name: str, *candidates: str) -> Path | None:
    dataset_dir = data_dir / dataset_name
    for candidate in candidates:
        path = dataset_dir / candidate
        if path.is_file():
            return path.resolve()
    return None


def _resolve_dataset_inputs(
    *,
    data_dir: Path,
    dataset_name: str,
    signatures_suffix: str,
    specter_suffix: str,
    include_test_pairs: bool,
) -> DatasetInputPlan:
    files = {
        "signatures": _resolve_dataset_file(
            data_dir,
            dataset_name,
            f"{dataset_name}{signatures_suffix}",
            signatures_suffix.lstrip("_"),
            "signatures.json",
        ),
        "papers": _resolve_dataset_file(data_dir, dataset_name, f"{dataset_name}_papers.json", "papers.json"),
        "specter_embeddings": _resolve_dataset_file(
            data_dir, dataset_name, f"{dataset_name}{specter_suffix}", specter_suffix.lstrip("_"), "specter.pickle"
        ),
    }
    if dataset_name in PAIRWISE_ONLY_DATASETS:
        files["train_pairs"] = _resolve_dataset_file(data_dir, dataset_name, "train_pairs.csv")
        if include_test_pairs:
            files["test_pairs"] = _resolve_dataset_file(data_dir, dataset_name, "test_pairs.csv")
        val_pairs = _optional_dataset_file(data_dir, dataset_name, "val_pairs.csv")
        if val_pairs is not None:
            files["val_pairs"] = val_pairs
    else:
        files["clusters"] = _resolve_dataset_file(
            data_dir, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"
        )
    return DatasetInputPlan(files=files, sha256=_hash_source_files(files))


def _positive_int_arg(args: argparse.Namespace, name: str) -> int:
    value = int(getattr(args, name))
    if value <= 0:
        option = name.replace("_", "-")
        raise SystemExit(f"--{option} must be positive, got {value}")
    return value


def _preflight_pairwise(args: argparse.Namespace) -> PairwisePreflightPlan:
    """Resolve and hash every launch input without loading artifacts or ANDData."""

    if bool(args.run_full) and bool(args.preflight_only):
        raise SystemExit("--run-full and --preflight-only are mutually exclusive")

    production_version = str(args.production_version)
    if not production_version or production_version != production_version.strip():
        raise SystemExit("--production-version must be nonempty and have no surrounding whitespace")

    for numeric_arg in (
        "n_iter",
        "cluster_n_iter",
        "n_jobs",
        "chunk_size",
        "train_pairs_size",
        "val_test_size",
    ):
        _positive_int_arg(args, numeric_arg)
    if int(args.random_seed) < 0:
        raise SystemExit(f"--random-seed must be non-negative, got {args.random_seed}")

    selected_raw = args.datasets
    if selected_raw is not None and len(selected_raw) == 0:
        raise SystemExit("--datasets requires at least one dataset name")
    selected = None if selected_raw is None else [str(value) for value in selected_raw]
    if selected is not None:
        if any(value != value.strip() or not value for value in selected):
            raise SystemExit("--datasets values must be nonempty names without surrounding whitespace")
        duplicates = sorted({value for value in selected if selected.count(value) > 1})
        if duplicates:
            raise SystemExit(f"--datasets contains duplicate names: {', '.join(duplicates)}")
        known_datasets = set(DEFAULT_SOURCE_DATASET_NAMES) | PAIRWISE_ONLY_DATASETS
        unknown = sorted(set(selected) - known_datasets)
        if unknown:
            raise SystemExit(f"--datasets contains unknown names: {', '.join(unknown)}")
    if selected is not None and bool(args.run_full):
        raise SystemExit("--run-full forbids --datasets; subset runs are non-publishable smoke tests")
    if selected is None and not bool(args.run_full) and not bool(args.preflight_only):
        raise SystemExit("full production training requires --run-full; smoke training requires --datasets")

    dataset_names = tuple(selected if selected is not None else (*DEFAULT_SOURCE_DATASET_NAMES, "augmented"))
    if len(dataset_names) == 0:
        raise SystemExit("At least one dataset is required")
    if all(name in PAIRWISE_ONLY_DATASETS for name in dataset_names):
        raise SystemExit("At least one clustered dataset is required to fit the clusterer")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir must name an existing directory: {data_dir}")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"--output-dir must name a new directory: {output_dir}")
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise SystemExit(f"--output-dir parent must be a directory: {output_dir.parent}")
    inferred_version = production_version_from_bundle_dir(output_dir)
    if inferred_version is not None and inferred_version != production_version:
        raise SystemExit(
            "--output-dir basename and --production-version disagree: "
            f"directory={output_dir.name!r} production_version={production_version!r}"
        )

    matrix_work_dir = Path(args.matrix_work_dir).resolve()
    if not matrix_work_dir.is_dir():
        raise SystemExit(f"--matrix-work-dir must name an existing directory: {matrix_work_dir}")

    feature_cache_dir = args.feature_cache_dir
    if selected is None and feature_cache_dir is not None:
        raise SystemExit("full release training does not use --feature-cache-dir")
    if feature_cache_dir is not None:
        cache_path = Path(feature_cache_dir).resolve()
        if cache_path.exists() and not cache_path.is_dir():
            raise SystemExit(f"--feature-cache-dir must name a directory: {cache_path}")
        if not cache_path.exists() and not cache_path.parent.is_dir():
            raise SystemExit(f"--feature-cache-dir parent must be an existing directory: {cache_path.parent}")
    else:
        cache_path = None
    if cache_path == output_dir:
        raise SystemExit("--feature-cache-dir must differ from --output-dir")

    requested_total_ram = args.total_ram_bytes
    if requested_total_ram is not None and int(requested_total_ram) <= 0:
        raise SystemExit(f"--total-ram-bytes must be positive, got {requested_total_ram}")

    resolved: dict[str, DatasetInputPlan] = {}
    for dataset_name in dataset_names:
        resolved[dataset_name] = _resolve_dataset_inputs(
            data_dir=data_dir,
            dataset_name=dataset_name,
            signatures_suffix=str(args.signatures_suffix),
            specter_suffix=str(args.specter_suffix),
            include_test_pairs=selected is not None,
        )

    test_manifest_sha256 = args.pairwise_test_manifest_sha256
    if selected is None:
        if (
            not isinstance(test_manifest_sha256, str)
            or len(test_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in test_manifest_sha256)
        ):
            raise SystemExit("full release training requires lowercase --pairwise-test-manifest-sha256")
    elif test_manifest_sha256 is not None:
        raise SystemExit("--pairwise-test-manifest-sha256 is only valid for full release training")

    return PairwisePreflightPlan(
        output_dir=output_dir,
        dataset_names=dataset_names,
        datasets=resolved,
        feature_cache_dir=cache_path,
        matrix_work_dir=matrix_work_dir,
        total_ram_bytes=None if requested_total_ram is None else int(requested_total_ram),
    )


def _training_config(
    args: argparse.Namespace,
    plan: PairwisePreflightPlan,
    *,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "chunk_size": int(args.chunk_size),
        "cluster_n_iter": int(args.cluster_n_iter),
        "dataset_inputs": {
            name: {
                "files": {
                    role: {"path": str(path), "sha256": plan.datasets[name].sha256[role]}
                    for role, path in plan.datasets[name].files.items()
                },
                "split_mode": "fixed_pairs" if "train_pairs" in plan.datasets[name].files else "random_blocks",
            }
            for name in plan.dataset_names
        },
        "features_to_use": list(DEFAULT_FEATURE_GROUPS),
        "featurizer_version": int(FEATURIZER_VERSION),
        "training_scope": "smoke_subset" if args.datasets is not None else "production_full",
        "input_artifact_hashes": {str(key): str(value) for key, value in artifact_hashes.items()},
        "n_iter": int(args.n_iter),
        "n_jobs": int(args.n_jobs),
        "nan_policy": "preserve_nan",
        "nameless_features_to_use": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "production_version": str(args.production_version),
        "pairwise_test_manifest_sha256": args.pairwise_test_manifest_sha256,
        "data_random_seed": int(args.random_seed),
        "model_random_seed": 42,
        "source_dataset_names": list(plan.dataset_names),
        "specter_suffix": str(args.specter_suffix),
        "signatures_suffix": str(args.signatures_suffix),
        "train_pairs_size": int(args.train_pairs_size),
        "uses_monotone_constraints": True,
        "val_test_size": int(args.val_test_size),
    }


def _require_disk_for_write(path: Path, payload_bytes: int) -> None:
    """Fail immediately when the next concrete matrix write cannot fit."""

    required_bytes = int(payload_bytes) + 4096
    free_bytes = int(shutil.disk_usage(path.parent).free)
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk for matrix write {path}: required_bytes={required_bytes} free_bytes={free_bytes}"
        )


def _stage_array(path: Path, values: np.ndarray) -> Path:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot stage object array at {path}")
    _require_disk_for_write(path, int(array.nbytes))
    np.save(path, array, allow_pickle=False)
    return path


def _stage_dataset_features(
    root: Path,
    dataset_name: str,
    *,
    train: TupleOfArrays,
    val: TupleOfArrays,
    test: TupleOfArrays | None = None,
) -> dict[str, Path]:
    dataset_root = root / dataset_name
    dataset_root.mkdir()
    arrays: dict[str, Path] = {}
    splits: list[tuple[str, TupleOfArrays]] = [("train", train), ("val", val)]
    if test is not None:
        splits.append(("test", test))
    for split_name, split in splits:
        features, labels, nameless_features = split
        if nameless_features is None:
            raise RuntimeError(f"Expected nameless {split_name} features for dataset {dataset_name!r}")
        for role, values in (
            (f"X_{split_name}", features),
            (f"y_{split_name}", labels),
            (f"nameless_X_{split_name}", nameless_features),
        ):
            arrays[role] = _stage_array(dataset_root / f"{role}.npy", values)
    return arrays


def _featurize_selection(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    *,
    n_jobs: int,
    chunk_size: int,
    nameless_featurizer_info: FeaturizationInfo,
    total_ram_bytes: int | None,
) -> tuple[TupleOfArrays, TupleOfArrays]:
    """Featurize release train/validation pairs and never resolve a test split."""

    train_pairs, val_pairs = resolve_selection_pairs(dataset)
    return (
        many_pairs_featurize(
            train_pairs,
            dataset,
            featurizer_info,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=np.nan,
            total_ram_bytes=total_ram_bytes,
        ),
        many_pairs_featurize(
            val_pairs,
            dataset,
            featurizer_info,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=np.nan,
            total_ram_bytes=total_ram_bytes,
        ),
    )


def _load_staged_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False, mmap_mode="r")


def _concatenate_staged_arrays(
    output_path: Path,
    arrays: list[Path],
) -> Path:
    if not arrays:
        raise ValueError(f"Cannot assemble empty union array at {output_path}")
    sources = [_load_staged_array(path) for path in arrays]
    trailing_shape = sources[0].shape[1:]
    if any(source.shape[1:] != trailing_shape for source in sources[1:]):
        raise ValueError("Cannot concatenate arrays with different shapes")
    shape = (sum(source.shape[0] for source in sources), *trailing_shape)
    if shape[0] <= 0:
        raise ValueError(f"Union array must contain at least one row: {output_path}")

    dtype = np.result_type(*(source.dtype for source in sources))
    _require_disk_for_write(output_path, int(np.prod(shape, dtype=np.int64) * dtype.itemsize))
    union = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
    offset = 0
    for source in sources:
        next_offset = offset + source.shape[0]
        union[offset:next_offset] = source
        offset = next_offset
    union.flush()
    return output_path


def _fit_pairwise_model(
    args: argparse.Namespace,
    arrays: Mapping[str, Path],
    *,
    feature_prefix: str,
    monotone_constraints: Any,
) -> PairwiseModeler:
    model = PairwiseModeler(
        n_iter=int(args.n_iter),
        n_jobs=int(args.n_jobs),
        monotone_constraints=monotone_constraints,
    )
    model.fit(
        _load_staged_array(arrays[f"{feature_prefix}X_train"]),
        _load_staged_array(arrays["y_train"]),
        _load_staged_array(arrays[f"{feature_prefix}X_val"]),
        _load_staged_array(arrays["y_val"]),
    )
    gc.collect()
    return model


def _smoke_pairwise_metrics(
    *,
    labels: np.ndarray,
    main_probabilities: np.ndarray,
    nameless_probabilities: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(labels).reshape(-1)
    main = np.asarray(main_probabilities, dtype=np.float64)[:, 1]
    nameless = np.asarray(nameless_probabilities, dtype=np.float64)[:, 1]
    if y.shape != main.shape or y.shape != nameless.shape:
        raise ValueError(
            "Pairwise test labels and probabilities must have the same row count: "
            f"labels={y.shape} main={main.shape} nameless={nameless.shape}"
        )
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("Pairwise test evaluation requires nonempty labels with both classes")
    probabilities = (main + nameless) / 2.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        probabilities > 0.5,
        beta=1.0,
        average="macro",
        zero_division=0,
    )
    metrics: dict[str, float | int] = {
        "rows": int(y.size),
        "auroc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
    if not all(np.isfinite(float(value)) for key, value in metrics.items() if key != "rows"):
        raise RuntimeError("Pairwise test evaluation produced a non-finite metric")
    return metrics


def _assert_sources_unchanged(plan: PairwisePreflightPlan) -> None:
    changed: list[str] = []
    for dataset_name in plan.dataset_names:
        dataset = plan.datasets[dataset_name]
        after = _hash_source_files(dataset.files)
        for role, expected_hash in dataset.sha256.items():
            if after[role] != expected_hash:
                changed.append(f"{dataset_name}:{role}")
    if changed:
        raise RuntimeError("Production training source files changed after preflight: " + ", ".join(changed))


def _publish_result(
    args: argparse.Namespace,
    plan: PairwisePreflightPlan,
    clusterer: Clusterer,
    training_config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if args.datasets is not None:
        return {
            "mode": "smoke",
            "bundle_dir": None,
            "bundle_status": "not_published",
            "bundle_version": str(args.production_version),
            "training_config": training_config,
            "training_summary": training_summary,
        }
    bundle = write_pairwise_production_bundle(
        clusterer,
        plan.output_dir,
        bundle_version=str(args.production_version),
        source_model_version=str(args.production_version),
        pairwise_training_config=training_config,
        pairwise_training_summary=training_summary,
    )
    return {
        "bundle_dir": str(bundle.bundle_dir),
        "bundle_status": bundle.bundle_status,
        "bundle_version": bundle.bundle_version,
        "manifest_path": str(bundle.manifest_path),
        "training_summary": training_summary,
    }


def train_pairwise_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Train pairwise models and write the pairwise production bundle stage."""

    plan = _preflight_pairwise(args)
    canonical_name_tuples = load_packaged_name_tuple_artifact()
    artifact_hashes = _canonical_training_artifact_hashes(canonical_name_tuples.data_sha256)
    training_config = _training_config(
        args,
        plan,
        artifact_hashes=artifact_hashes,
    )
    if bool(args.preflight_only):
        result = {
            "mode": "preflight",
            "output_dir": str(plan.output_dir),
            "training_config": training_config,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    os.environ["OMP_NUM_THREADS"] = str(int(args.n_jobs))

    featurizer_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    nameless_featurizer_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    monotone_constraints = featurizer_info.lightgbm_monotone_constraints
    nameless_monotone_constraints = nameless_featurizer_info.lightgbm_monotone_constraints

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".pairwise_v{args.production_version}_",
        dir=plan.matrix_work_dir,
    ) as matrix_work_dir_raw:
        matrix_work_dir = Path(matrix_work_dir_raw)
        staged_datasets: dict[str, dict[str, Path]] = {}
        anddatas: list[ANDData] = []
        for dataset_name in tqdm(plan.dataset_names, desc="Loading and featurizing datasets"):
            logger.info("processing dataset %s", dataset_name)
            dataset_input = plan.datasets[dataset_name]
            files = dataset_input.files
            anddata_kwargs: dict[str, Any] = {
                "signatures": str(files["signatures"]),
                "papers": str(files["papers"]),
                "name": dataset_name,
                "mode": "train",
                "specter_embeddings": str(files["specter_embeddings"]),
                "clusters": str(files["clusters"]) if "clusters" in files else None,
                "train_pairs": str(files["train_pairs"]) if "train_pairs" in files else None,
                "val_pairs": str(files["val_pairs"]) if "val_pairs" in files else None,
                "test_pairs": str(files["test_pairs"]) if "test_pairs" in files else None,
                "train_pairs_size": int(args.train_pairs_size),
                "val_pairs_size": int(args.val_test_size),
                "test_pairs_size": int(args.val_test_size),
                "random_seed": int(args.random_seed),
                "name_counts_index": NAME_COUNTS_INDEX_PATH,
                "n_jobs": int(args.n_jobs),
                "preprocess": True,
            }

            anddata = ANDData(**anddata_kwargs)
            if anddata.name_tuples != canonical_name_tuples.pairs:
                raise ValueError(f"Production training dataset {dataset_name!r} does not use the canonical name tuples")

            release_training = args.datasets is None
            if release_training:
                train, val = _featurize_selection(
                    anddata,
                    featurizer_info,
                    n_jobs=int(args.n_jobs),
                    chunk_size=int(args.chunk_size),
                    nameless_featurizer_info=nameless_featurizer_info,
                    total_ram_bytes=plan.total_ram_bytes,
                )
                staged = _stage_dataset_features(
                    matrix_work_dir,
                    dataset_name,
                    train=train,
                    val=val,
                )
                del train, val
            elif plan.feature_cache_dir is None:
                train, val, test = cast(
                    tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays],
                    featurize(
                        anddata,
                        featurizer_info,
                        n_jobs=int(args.n_jobs),
                        chunk_size=int(args.chunk_size),
                        nameless_featurizer_info=nameless_featurizer_info,
                        nan_value=np.nan,
                        total_ram_bytes=plan.total_ram_bytes,
                    ),
                )
            else:
                name_counts_provenance = anddata.name_counts_provenance
                if name_counts_provenance is None:
                    raise RuntimeError(f"Production training dataset {dataset_name!r} has no name-count manifest")
                source_key = {
                    **{f"{role}_sha256": dataset_input.sha256[role] for role in ("signatures", "papers")},
                    "specter_embeddings_sha256": dataset_input.sha256["specter_embeddings"],
                    "name_tuples_data_sha256": canonical_name_tuples.data_sha256,
                    "name_counts_manifest_sha256": name_counts_provenance["manifest_sha256"],
                    "normalization_version": anddata.normalization_version,
                }
                train, val, test = cached_featurize(
                    anddata,
                    featurizer_info,
                    source_key=source_key,
                    cache_dir=plan.feature_cache_dir,
                    n_jobs=int(args.n_jobs),
                    chunk_size=int(args.chunk_size),
                    nameless_featurizer_info=nameless_featurizer_info,
                    nan_value=np.nan,
                    total_ram_bytes=plan.total_ram_bytes,
                )
            if not release_training:
                staged = _stage_dataset_features(
                    matrix_work_dir,
                    dataset_name,
                    train=train,
                    val=val,
                    test=test,
                )
                del train, val, test
            staged_datasets[dataset_name] = staged
            if dataset_name not in PAIRWISE_ONLY_DATASETS:
                anddatas.append(anddata)
            del staged
            if dataset_name in PAIRWISE_ONLY_DATASETS:
                del anddata
            gc.collect()

        _assert_sources_unchanged(plan)
        validation_dataset_names = tuple(name for name in plan.dataset_names if name != "augmented")
        union_members = {
            "X_train": plan.dataset_names,
            "y_train": plan.dataset_names,
            "nameless_X_train": plan.dataset_names,
            "X_val": validation_dataset_names,
            "y_val": validation_dataset_names,
            "nameless_X_val": validation_dataset_names,
        }
        union_arrays = {
            role: _concatenate_staged_arrays(
                matrix_work_dir / f"union_{role}.npy",
                [staged_datasets[name][role] for name in names],
            )
            for role, names in union_members.items()
        }

        logger.info("fitting pairwise model")
        union_classifier = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="",
            monotone_constraints=monotone_constraints,
        )

        logger.info("fitting nameless pairwise model")
        nameless_union_classifier = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="nameless_",
            monotone_constraints=nameless_monotone_constraints,
        )

        pairwise_test_metrics: dict[str, dict[str, float | int]] = {}
        if args.datasets is not None:
            for dataset_name in plan.dataset_names:
                staged = staged_datasets[dataset_name]
                X_test = _load_staged_array(staged["X_test"])
                y_test = _load_staged_array(staged["y_test"])
                nameless_X_test = _load_staged_array(staged["nameless_X_test"])
                pairwise_test_metrics[dataset_name] = _smoke_pairwise_metrics(
                    labels=y_test,
                    main_probabilities=union_classifier.predict_proba(X_test),
                    nameless_probabilities=nameless_union_classifier.predict_proba(nameless_X_test),
                )
                del X_test, y_test, nameless_X_test

        logger.info("fitting clustering threshold")
        union_clusterer = Clusterer(
            featurizer_info,
            union_classifier.classifier,
            cluster_model=FastCluster(),
            search_space=_search_space(),
            n_iter=int(args.cluster_n_iter),
            n_jobs=int(args.n_jobs),
            nameless_classifier=nameless_union_classifier.classifier,
            nameless_featurizer_info=nameless_featurizer_info,
        )
        union_clusterer.feature_contract.update(artifact_hashes)
        union_clusterer.fit(anddatas)
        best_params = union_clusterer.best_params
        if best_params is None:
            raise RuntimeError("Clusterer fitting did not produce best clustering parameters.")

        training_summary: dict[str, Any] = {
            "best_clustering_params": dict(best_params),
            "elapsed_seconds": round(float(time.perf_counter() - started), 3),
            "main_pairwise_best_params": dict(union_classifier.best_params or {}),
            "main_train_rows": int(_load_staged_array(union_arrays["X_train"]).shape[0]),
            "main_val_rows": int(_load_staged_array(union_arrays["X_val"]).shape[0]),
            "nameless_pairwise_best_params": dict(nameless_union_classifier.best_params or {}),
            "nameless_train_rows": int(_load_staged_array(union_arrays["nameless_X_train"]).shape[0]),
            "nameless_val_rows": int(_load_staged_array(union_arrays["nameless_X_val"]).shape[0]),
        }
        if pairwise_test_metrics:
            training_summary["smoke_pairwise_test_metrics"] = pairwise_test_metrics

    result = _publish_result(
        args,
        plan,
        union_clusterer,
        training_config,
        training_summary,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-version", required=True, help="Version suffix for production_model_vX.Y.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--specter-suffix", default=DEFAULT_SPECTER_SUFFIX)
    parser.add_argument("--signatures-suffix", default=DEFAULT_SIGNATURES_SUFFIX)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--cluster-n-iter", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--train-pairs-size", type=int, default=DEFAULT_TRAIN_PAIRS_SIZE)
    parser.add_argument("--val-test-size", type=int, default=DEFAULT_VAL_TEST_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--matrix-work-dir", type=Path, required=True)
    parser.add_argument(
        "--pairwise-test-manifest-sha256",
        help="Sealed Stage-8 pair-manifest digest; the full trainer records but never opens the manifest.",
    )
    parser.add_argument(
        "--total-ram-bytes",
        type=int,
        default=None,
        help="Optional explicit RAM budget; autodetected RAM is safety-capped when omitted.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Run a non-publishable smoke test on this explicit dataset subset.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional featurized-split snapshot cache directory for repeated training experiments. "
        "Prefer a local unsynced directory; snapshots are content-addressed NPZ files.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and hash all local inputs and canonical artifacts without creating output or training.",
    )
    mode.add_argument("--run-full", action="store_true", help="Explicitly allow full production pairwise training.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = build_parser()
    train_pairwise_bundle(parser.parse_args(argv))


if __name__ == "__main__":
    main()
