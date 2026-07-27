"""Train the pairwise half of a native production model bundle.

The output directory is a pairwise-only ``production_model_vX.Y`` bundle stage.
``train_linker_and_finalize.py`` then fits one fresh linker and writes, reloads,
and evaluates the final complete bundle.
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
from hyperopt import hp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import is_lowercase_sha256  # noqa: E402
from s2and._sha256 import sha256_file as _sha256_file  # noqa: E402
from s2and.consts import FEATURIZER_VERSION  # noqa: E402
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
from s2and.production_bundle import production_version_from_bundle_dir, write_pairwise_production_bundle  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    PAIRWISE_TRAINING_PLAN_SCHEMA_VERSION,
    ProductionArtifactAuthority,
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
TRAINING_PLAN_SCHEMA = PAIRWISE_TRAINING_PLAN_SCHEMA_VERSION


@dataclass(frozen=True)
class DatasetInputPlan:
    """Resolved immutable input files for one production training dataset."""

    split_mode: str
    files: Mapping[str, Path]
    sha256: Mapping[str, str]


@dataclass(frozen=True)
class PairwisePreflightPlan:
    """Validated, read-only launch plan created before artifact or dataset loading."""

    output_dir: Path
    dataset_names: tuple[str, ...]
    datasets: Mapping[str, DatasetInputPlan]
    matrix_work_dir: Path
    matrix_work_free_bytes: int
    total_ram_bytes: int | None
    source_manifest_sha256: str | None
    sealed_test_manifests: Mapping[str, Any]


def _hash_source_files(source_files: Mapping[str, Path]) -> dict[str, str]:
    return {role: _sha256_file(path) for role, path in source_files.items()}


def _canonical_training_artifact_hashes(authority: ProductionArtifactAuthority) -> dict[str, str]:
    """Validate canonical count artifacts and return their immutable bindings."""

    orcid_counts = authority.orcid_prefix_counts
    if not orcid_counts.source_kind.startswith("redshift:"):
        raise RuntimeError(
            "production pairwise training requires warehouse-generated ORCID prefix counts; "
            f"observed source_kind={orcid_counts.source_kind!r}"
        )
    if orcid_counts.name_tuples_sha256 != authority.name_tuples.data_sha256:
        raise RuntimeError("ORCID prefix counts were generated from a different canonical name-tuple artifact")
    name_counts_index = authority.name_counts_index
    source_kind = name_counts_index.source_provenance.get("source_kind")
    if not isinstance(source_kind, str) or not source_kind.startswith("redshift:"):
        raise RuntimeError(
            "production pairwise training requires a warehouse-generated name-count index; "
            f"observed source_kind={source_kind!r}"
        )
    return authority.hashes


def _search_space() -> dict[str, Any]:
    return {
        "eps": hp.uniform("eps", 0, 1),
        "linkage": hp.choice("linkage", ["average"]),
    }


def _sha256_digest(value: Any, *, label: str) -> str:
    """Return one lowercase SHA-256 digest."""

    if not is_lowercase_sha256(value):
        raise SystemExit(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verified_training_plan(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, DatasetInputPlan], str, dict[str, Any]]:
    """Load one digest-bound plan whose sealed test bindings contain no paths."""

    expected_sha256 = _sha256_digest(expected_sha256, label="--expected-training-plan-sha256")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise SystemExit(f"Training-plan SHA-256 mismatch: expected={expected_sha256} observed={observed_sha256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {"schema_version", "source_manifest_sha256", "datasets", "sealed_test_manifests"}
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise SystemExit(f"Training plan must contain exactly {sorted(expected_keys)}")
    if payload["schema_version"] != TRAINING_PLAN_SCHEMA:
        raise SystemExit(f"Training plan schema_version must be {TRAINING_PLAN_SCHEMA!r}")
    source_manifest_sha256 = _sha256_digest(
        payload["source_manifest_sha256"],
        label="Training plan source_manifest_sha256",
    )

    sealed = payload["sealed_test_manifests"]
    if not isinstance(sealed, Mapping) or set(sealed) != {"pairwise", "cluster"}:
        raise SystemExit("Training plan sealed_test_manifests must contain exactly pairwise and cluster")
    normalized_sealed: dict[str, Any] = {}
    for kind in ("pairwise", "cluster"):
        binding = sealed[kind]
        if not isinstance(binding, Mapping) or set(binding) != {"manifest_sha256", "members"}:
            raise SystemExit(f"Training plan sealed {kind} binding must contain exactly manifest_sha256 and members")
        members = binding["members"]
        if not isinstance(members, Mapping) or not members:
            raise SystemExit(f"Training plan sealed {kind} members must be a nonempty object")
        normalized_members: dict[str, dict[str, str]] = {}
        for dataset_name, roles in members.items():
            if not isinstance(dataset_name, str) or not dataset_name or not isinstance(roles, Mapping) or not roles:
                raise SystemExit(f"Training plan sealed {kind} members must map dataset names to file digests")
            normalized_members[dataset_name] = {
                str(role): _sha256_digest(digest, label=f"Training plan sealed {kind} {dataset_name}:{role}")
                for role, digest in roles.items()
            }
        normalized_sealed[kind] = {
            "manifest_sha256": _sha256_digest(
                binding["manifest_sha256"],
                label=f"Training plan sealed {kind} manifest_sha256",
            ),
            "members": normalized_members,
        }

    raw_datasets = payload["datasets"]
    if not isinstance(raw_datasets, list):
        raise SystemExit("Training plan datasets must be a list")
    by_name: dict[str, DatasetInputPlan] = {}
    common_roles = {"papers", "signatures", "specter_embeddings"}
    for spec in raw_datasets:
        if not isinstance(spec, Mapping) or set(spec) != {"name", "split_mode", "files"}:
            raise SystemExit("Each training-plan dataset must contain exactly name, split_mode, files")
        name = spec["name"]
        if not isinstance(name, str) or not name or name in by_name:
            raise SystemExit(f"Training plan contains an invalid or duplicate dataset name: {name!r}")
        expected_mode = "fixed_pairs" if name == "augmented" else "random_blocks"
        if spec["split_mode"] != expected_mode:
            raise SystemExit(f"Training plan dataset {name!r} must use split_mode={expected_mode!r}")
        roles = common_roles | ({"train_pairs", "val_pairs"} if expected_mode == "fixed_pairs" else {"clusters"})
        raw_files = spec["files"]
        if not isinstance(raw_files, Mapping) or set(raw_files) != roles:
            raise SystemExit(f"Training plan dataset {name!r} must declare exact file roles {sorted(roles)}")
        files: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for role in sorted(roles):
            file_spec = raw_files[role]
            if not isinstance(file_spec, Mapping) or set(file_spec) != {"path", "sha256"}:
                raise SystemExit(f"Training plan file {name}:{role} must contain exactly path and sha256")
            raw_path = file_spec["path"]
            if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
                raise SystemExit(f"Training plan file {name}:{role} path must be absolute")
            file_path = Path(raw_path).resolve()
            digest = _sha256_digest(file_spec["sha256"], label=f"Training plan file {name}:{role} sha256")
            observed = _sha256_file(file_path)
            if observed != digest:
                raise SystemExit(
                    f"Training plan file {name}:{role} SHA-256 mismatch: expected={digest} observed={observed}"
                )
            files[role] = file_path
            digests[role] = digest
        by_name[name] = DatasetInputPlan(split_mode=expected_mode, files=files, sha256=digests)

    expected_names = set((*DEFAULT_SOURCE_DATASET_NAMES, "augmented"))
    if set(by_name) != expected_names:
        raise SystemExit(
            "Training plan dataset names disagree with the production set: "
            f"missing={sorted(expected_names - set(by_name))} extra={sorted(set(by_name) - expected_names)}"
        )
    return by_name, source_manifest_sha256, normalized_sealed


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
    """Resolve and hash every launch input without loading artifacts or ANDData."""

    production_version = str(args.production_version)
    if not production_version or production_version != production_version.strip():
        raise SystemExit("--production-version must be nonempty and have no surrounding whitespace")

    for numeric_arg in (
        "n_iter",
        "cluster_n_iter",
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
    inferred_version = production_version_from_bundle_dir(output_dir)
    if inferred_version is not None and inferred_version != production_version:
        raise SystemExit(
            "--output-dir basename and --production-version disagree: "
            f"directory={output_dir.name!r} production_version={production_version!r}"
        )

    matrix_work_dir = Path(args.matrix_work_dir).resolve()
    matrix_work_free_bytes = _preflight_matrix_work_dir(matrix_work_dir)

    requested_total_ram = args.total_ram_bytes
    if requested_total_ram is not None and int(requested_total_ram) <= 0:
        raise SystemExit(f"--total-ram-bytes must be positive, got {requested_total_ram}")

    if args.training_plan is None or args.expected_training_plan_sha256 is None:
        raise SystemExit("production training requires --training-plan and --expected-training-plan-sha256")
    resolved, source_manifest_sha256, sealed_test_manifests = _verified_training_plan(
        Path(args.training_plan).resolve(),
        args.expected_training_plan_sha256,
    )

    return PairwisePreflightPlan(
        output_dir=output_dir,
        dataset_names=dataset_names,
        datasets=resolved,
        matrix_work_dir=matrix_work_dir,
        matrix_work_free_bytes=matrix_work_free_bytes,
        total_ram_bytes=None if requested_total_ram is None else int(requested_total_ram),
        source_manifest_sha256=source_manifest_sha256,
        sealed_test_manifests=sealed_test_manifests,
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
                "split_mode": plan.datasets[name].split_mode,
            }
            for name in plan.dataset_names
        },
        "features_to_use": list(DEFAULT_FEATURE_GROUPS),
        "featurizer_version": int(FEATURIZER_VERSION),
        "training_scope": "production_full",
        "input_artifact_hashes": {str(key): str(value) for key, value in artifact_hashes.items()},
        "n_iter": int(args.n_iter),
        "n_jobs": int(args.n_jobs),
        "nan_policy": "preserve_nan",
        "nameless_features_to_use": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "production_version": str(args.production_version),
        "pairwise_inputs_manifest_sha256": plan.source_manifest_sha256,
        "sealed_test_manifests": plan.sealed_test_manifests,
        "data_random_seed": int(args.random_seed),
        "model_random_seed": 42,
        "matrix_work_storage": {
            "path": str(plan.matrix_work_dir),
            "measured_free_bytes": plan.matrix_work_free_bytes,
        },
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
    bundle = write_pairwise_production_bundle(
        clusterer,
        plan.output_dir,
        bundle_version=str(args.production_version),
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
                "test_pairs": None,
                "train_pairs_size": int(args.train_pairs_size),
                "val_pairs_size": int(args.validation_pairs_size),
                "random_seed": int(args.random_seed),
                "name_counts_index": Path(args.name_counts_index_root),
                "n_jobs": int(args.n_jobs),
                "preprocess": True,
                "name_tuples": canonical_name_tuples.pairs,
            }

            anddata = ANDData(**anddata_kwargs)
            if anddata.name_tuples != canonical_name_tuples.pairs:
                raise ValueError(f"Production training dataset {dataset_name!r} does not use the canonical name tuples")

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
            if dataset_name != "augmented":
                anddatas.append(anddata)
            del staged
            if dataset_name == "augmented":
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
    parser.add_argument(
        "--name-counts-index-root",
        type=Path,
        required=True,
        help="Explicit manifest-backed name-count index directory.",
    )
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--cluster-n-iter", type=int, default=25)
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
        "--training-plan",
        type=Path,
        required=True,
        help="Test-path-free plan emitted by release_pairwise.py preflight-training-inputs.",
    )
    parser.add_argument(
        "--expected-training-plan-sha256",
        required=True,
        help="Expected SHA-256 of --training-plan.",
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
