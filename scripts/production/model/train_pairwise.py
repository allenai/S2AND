"""Train the pairwise half of a native production model bundle.

The output directory is a pairwise-only ``production_model_vX.Y`` bundle with
pending placeholder EPS. Validation-only calibration writes the calibrated
pairwise sibling consumed by ``train_linker_and_finalize.py``.
"""

from __future__ import annotations

import argparse
import gc
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
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.data import ANDData  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    TupleOfArrays,
    many_pairs_featurize,
    resolve_selection_pairs,
)
from s2and.model import Clusterer, FastCluster, PairwiseModeler  # noqa: E402
from s2and.production_bundle import write_pairwise_production_bundle  # noqa: E402
from s2and.production_bundle_contract import PENDING_EPS_CALIBRATION, PENDING_PAIRWISE_EPS  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    ModelDataset,
    ProductionArtifactAuthority,
    block_membership_sha256,
    load_model_plan,
    load_packaged_artifact_authority,
)

logger = logging.getLogger("s2and")

DEFAULT_SOURCE_DATASET_NAMES = ("aminer", "arnetminer", "inspire", "kisti", "orcid", "pubmed", "qian", "zbmath")
DEFAULT_TRAIN_PAIRS_SIZE = 100_000
DEFAULT_VALIDATION_PAIRS_SIZE = 10_000
DEFAULT_N_ITER = 50
DEFAULT_N_JOBS = 25
DEFAULT_CHUNK_SIZE = 100
DEFAULT_RANDOM_SEED = 1111


@dataclass(frozen=True, slots=True)
class PairwisePreflightPlan:
    """Validated, read-only launch plan created before artifact or dataset loading."""

    output_dir: Path
    release_version: str
    dataset_names: tuple[str, ...]
    datasets: Mapping[str, ModelDataset]
    model_plan_sha256: str
    matrix_work_dir: Path
    matrix_work_free_bytes: int
    total_ram_bytes: int | None


def _canonical_training_artifact_hashes(authority: ProductionArtifactAuthority) -> dict[str, str]:
    """Validate canonical count artifacts and return their immutable bindings."""

    orcid_counts = authority.orcid_prefix_counts
    if orcid_counts.name_tuples_sha256 != authority.name_tuples.data_sha256:
        raise RuntimeError("ORCID prefix counts were generated from a different canonical name-tuple artifact")
    return authority.hashes


def _positive_int_arg(args: argparse.Namespace, name: str) -> int:
    value = int(getattr(args, name))
    if value <= 0:
        option = name.replace("_", "-")
        raise SystemExit(f"--{option} must be positive, got {value}")
    return value


def _preflight_matrix_work_dir(path: Path) -> int:
    """Verify the scratch directory and return its measured free capacity."""

    if not path.is_dir():
        raise SystemExit(f"--matrix-work-dir must name an existing directory: {path}")
    if next(path.iterdir(), None) is not None:
        raise SystemExit(f"--matrix-work-dir must be empty: {path}")

    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".s2and_preflight_", dir=path) as probe:
            probe.write(b"s2and matrix-work preflight\n")
            probe.flush()
            os.fsync(probe.fileno())
    except OSError as exc:
        raise SystemExit(f"--matrix-work-dir is not writable: {path}: {exc}") from exc
    return int(shutil.disk_usage(path).free)


def _preflight_pairwise(args: argparse.Namespace) -> PairwisePreflightPlan:
    """Resolve every launch input without loading artifacts or ANDData."""

    for numeric_arg in (
        "n_iter",
        "n_jobs",
        "chunk_size",
        "train_pairs_size",
        "validation_pairs_size",
    ):
        _positive_int_arg(args, numeric_arg)
    if int(args.random_seed) < 0:
        raise SystemExit(f"--random-seed must be non-negative, got {args.random_seed}")

    if not bool(args.run_full):
        raise SystemExit("production training requires --run-full")
    dataset_names = (*DEFAULT_SOURCE_DATASET_NAMES, "augmented")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SystemExit(f"--output-dir must name a new directory: {output_dir}")
    if output_dir.parent.exists() and not output_dir.parent.is_dir():
        raise SystemExit(f"--output-dir parent must be a directory: {output_dir.parent}")
    matrix_work_dir = Path(args.matrix_work_dir).resolve()
    matrix_work_free_bytes = _preflight_matrix_work_dir(matrix_work_dir)

    requested_total_ram = args.total_ram_bytes
    if requested_total_ram is not None and int(requested_total_ram) <= 0:
        raise SystemExit(f"--total-ram-bytes must be positive, got {requested_total_ram}")

    if args.model_plan is None:
        raise SystemExit("production training requires --model-plan")
    model_plan = load_model_plan(Path(args.model_plan).resolve())
    expected_modes = {
        **{name: "random_blocks" for name in DEFAULT_SOURCE_DATASET_NAMES},
        "augmented": "fixed_pairs",
    }
    observed_modes = {name: dataset.split_mode for name, dataset in model_plan.datasets.items()}
    if observed_modes != expected_modes:
        raise SystemExit("model plan datasets disagree with the production dataset set")

    return PairwisePreflightPlan(
        output_dir=output_dir,
        release_version=model_plan.release_version,
        dataset_names=dataset_names,
        datasets=model_plan.datasets,
        model_plan_sha256=model_plan.sha256,
        matrix_work_dir=matrix_work_dir,
        matrix_work_free_bytes=matrix_work_free_bytes,
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
        "dataset_inputs": {
            name: {
                "files": {role: str(path) for role, path in plan.datasets[name].files.items()},
                "split_mode": plan.datasets[name].split_mode,
            }
            for name in plan.dataset_names
        },
        "features_to_use": list(DEFAULT_FEATURE_GROUPS),
        "training_scope": "production_full",
        "input_artifact_hashes": {str(key): str(value) for key, value in artifact_hashes.items()},
        "n_iter": int(args.n_iter),
        "n_jobs": int(args.n_jobs),
        "nan_policy": "preserve_nan",
        "nameless_features_to_use": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "model_plan_sha256": plan.model_plan_sha256,
        "data_random_seed": int(args.random_seed),
        "model_random_seed": 42,
        "source_dataset_names": list(plan.dataset_names),
        "train_pairs_size": int(args.train_pairs_size),
        "uses_monotone_constraints": True,
        "validation_pairs_size": int(args.validation_pairs_size),
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
) -> dict[str, Path]:
    dataset_root = root / dataset_name
    dataset_root.mkdir()
    arrays: dict[str, Path] = {}
    for split_name, split in (("train", train), ("val", val)):
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


def _finite_validation_roc_auc(value: Any) -> float:
    metric = float(value)
    if not np.isfinite(metric):
        raise RuntimeError("selected validation ROC AUC must be finite")
    return metric


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
) -> tuple[PairwiseModeler, float]:
    model = PairwiseModeler(
        n_iter=int(args.n_iter),
        n_jobs=int(args.n_jobs),
        monotone_constraints=monotone_constraints,
    )
    X_train = _load_staged_array(arrays[f"{feature_prefix}X_train"])
    y_train = _load_staged_array(arrays["y_train"])
    X_val = _load_staged_array(arrays[f"{feature_prefix}X_val"])
    y_val = _load_staged_array(arrays["y_val"])
    model.fit(X_train, y_train, X_val, y_val)
    validation_roc_auc = _finite_validation_roc_auc(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    gc.collect()
    return model, validation_roc_auc


def _publish_result(
    args: argparse.Namespace,
    plan: PairwisePreflightPlan,
    clusterer: Clusterer,
    training_config: Mapping[str, Any],
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    bundle = write_pairwise_production_bundle(
        clusterer,
        plan.output_dir,
        release_version=plan.release_version,
        eps_calibration=PENDING_EPS_CALIBRATION,
        pairwise_training_config=training_config,
        pairwise_training_summary=training_summary,
    )
    return {
        "bundle_dir": str(bundle.bundle_dir),
        "release_version": bundle.release_version,
        "eps_calibration": bundle.eps_calibration,
        "manifest_path": str(bundle.manifest_path),
        "training_summary": training_summary,
    }


def train_pairwise_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Train pairwise models and write the pairwise production bundle stage."""

    plan = _preflight_pairwise(args)
    artifact_authority = load_packaged_artifact_authority(
        name_counts_index_root=Path(args.name_counts_index_root),
    )
    canonical_name_tuples = artifact_authority.name_tuples
    artifact_hashes = _canonical_training_artifact_hashes(artifact_authority)
    training_config = _training_config(
        args,
        plan,
        artifact_hashes=artifact_hashes,
    )
    os.environ["OMP_NUM_THREADS"] = str(int(args.n_jobs))
    logger.info(
        "pairwise resources n_jobs=%d matrix_work_dir=%s free_bytes=%d total_ram_bytes=%s",
        int(args.n_jobs),
        plan.matrix_work_dir,
        plan.matrix_work_free_bytes,
        plan.total_ram_bytes,
    )

    featurizer_info = FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS))
    nameless_featurizer_info = FeaturizationInfo(features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS))
    monotone_constraints = featurizer_info.lightgbm_monotone_constraints
    nameless_monotone_constraints = nameless_featurizer_info.lightgbm_monotone_constraints

    cluster_test_splits: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=f".pairwise_v{plan.release_version}_",
        dir=plan.matrix_work_dir,
    ) as matrix_work_dir_raw:
        matrix_work_dir = Path(matrix_work_dir_raw)
        staged_datasets: dict[str, dict[str, Path]] = {}
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
                "test_pairs": None,
                "train_pairs_size": int(args.train_pairs_size),
                "val_pairs_size": int(args.validation_pairs_size),
                "random_seed": int(args.random_seed),
                "name_counts_index": artifact_authority.name_counts_index,
                "n_jobs": int(args.n_jobs),
                "preprocess": True,
                "name_tuples": canonical_name_tuples.pairs,
            }

            anddata = ANDData(**anddata_kwargs)
            observed_name_counts = anddata.name_counts_manifest_sha256
            expected_name_counts = artifact_hashes["name_counts_manifest_sha256"]
            if observed_name_counts != expected_name_counts:
                raise ValueError(
                    f"Production training dataset {dataset_name!r} name-count manifest mismatch: "
                    f"expected={expected_name_counts!r} observed={observed_name_counts!r}"
                )
            if anddata.name_tuples != canonical_name_tuples.pairs:
                raise ValueError(f"Production training dataset {dataset_name!r} does not use the canonical name tuples")

            if dataset_input.split_mode == "random_blocks":
                # Resolve the same split as pair selection, before allocating feature
                # matrices. Persist identities so Arrow row sorting cannot change it.
                train_blocks, val_blocks, test_blocks = anddata.split_cluster_signatures()
                cluster_test_splits[dataset_name] = {
                    "block_membership_sha256": block_membership_sha256(anddata.get_blocks()),
                    "test_block_ids": list(test_blocks),
                }
                del train_blocks, val_blocks, test_blocks

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
            staged_datasets[dataset_name] = staged
            del staged
            del anddata
            gc.collect()

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
        union_classifier, main_validation_roc_auc = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="",
            monotone_constraints=monotone_constraints,
        )

        logger.info("fitting nameless pairwise model")
        nameless_union_classifier, nameless_validation_roc_auc = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="nameless_",
            monotone_constraints=nameless_monotone_constraints,
        )

        logger.info("building uncalibrated pairwise bundle")
        union_clusterer = Clusterer(
            featurizer_info,
            union_classifier.classifier,
            cluster_model=FastCluster(linkage="average", eps=PENDING_PAIRWISE_EPS),
            n_jobs=int(args.n_jobs),
            nameless_classifier=nameless_union_classifier.classifier,
            nameless_featurizer_info=nameless_featurizer_info,
        )
        union_clusterer.feature_contract.update(artifact_hashes)

        training_summary: dict[str, Any] = {
            "cluster_test_splits": cluster_test_splits,
            "main_pairwise_best_params": dict(union_classifier.best_params or {}),
            "main_train_rows": int(_load_staged_array(union_arrays["X_train"]).shape[0]),
            "main_val_rows": int(_load_staged_array(union_arrays["X_val"]).shape[0]),
            "main_validation_roc_auc": main_validation_roc_auc,
            "nameless_pairwise_best_params": dict(nameless_union_classifier.best_params or {}),
            "nameless_train_rows": int(_load_staged_array(union_arrays["nameless_X_train"]).shape[0]),
            "nameless_val_rows": int(_load_staged_array(union_arrays["nameless_X_val"]).shape[0]),
            "nameless_validation_roc_auc": nameless_validation_roc_auc,
        }
        logger.info("pairwise training completed in %.3f seconds", time.perf_counter() - started)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--name-counts-index-root",
        type=Path,
        required=True,
        help="Explicit manifest-backed name-count index directory.",
    )
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--train-pairs-size", type=int, default=DEFAULT_TRAIN_PAIRS_SIZE)
    parser.add_argument(
        "--validation-pairs-size",
        type=int,
        default=DEFAULT_VALIDATION_PAIRS_SIZE,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--matrix-work-dir", type=Path, required=True)
    parser.add_argument(
        "--model-plan",
        type=Path,
        required=True,
        help="Training/validation inputs and EPS policy for this release run.",
    )
    parser.add_argument(
        "--total-ram-bytes",
        type=int,
        default=None,
        help="Optional explicit RAM budget; autodetected RAM is safety-capped when omitted.",
    )
    parser.add_argument(
        "--run-full",
        action="store_true",
        required=True,
        help="Acknowledge the full production training cost.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = build_parser()
    train_pairwise_bundle(parser.parse_args(argv))


if __name__ == "__main__":
    main()
