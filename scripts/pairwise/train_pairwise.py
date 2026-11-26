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
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.data import ANDData, _map_fixed_pair_labels  # noqa: E402
from s2and.feature_port import (  # noqa: E402
    build_rust_featurizer_from_arrow_dataset,
    build_verified_rust_featurizer,
    featurize_pairs_with_rust,
)
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    TupleOfArrays,
    resolve_selection_pairs,
)
from s2and.model import Clusterer, FastCluster, PairwiseModeler  # noqa: E402
from s2and.model_pairwise import MAX_FAST_PLAN_LEAVES, _validated_classifier_features  # noqa: E402
from s2and.production_bundle import write_pairwise_production_bundle  # noqa: E402
from s2and.production_bundle_contract import PENDING_EPS_CALIBRATION, PENDING_PAIRWISE_EPS  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    ModelDataset,
    ProductionArtifactAuthority,
    block_membership_sha256,
    load_model_plan,
    load_packaged_artifact_authority,
)
from scripts.common.block_roles import load_block_roles  # noqa: E402
from scripts.common.identity_ledger import ledger_for_roles  # noqa: E402
from scripts.common.plan_datasets import (  # noqa: E402
    development_split_members,
    load_development_dataset,
    open_plan_arrow,
    plan_unknown_signatures,
    require_arrow_plan,
)
from scripts.common.unknown_signatures import (  # noqa: E402
    augmented_unknown_ids,
    exclude_unknown_pairs,
)
from scripts.pairwise.augmented_leakage import (  # noqa: E402
    require_declared_sources,
    require_training_members,
    require_validation_members,
    source_members,
)
from scripts.pairwise.training_pairs import TrainingBlock, training_pairs  # noqa: E402

logger = logging.getLogger("s2and")

# Historical plan examples use these names; each model plan declares its actual datasets.
DEFAULT_SOURCE_DATASET_NAMES = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
DEFAULT_TRAIN_PAIRS_SIZE = 100_000
DEFAULT_VALIDATION_PAIRS_SIZE = 10_000
DEFAULT_N_ITER = 50
DEFAULT_RANDOM_SEED = 1111


@dataclass(frozen=True, slots=True)
class PairwisePreflightPlan:
    """Validated, read-only launch plan created before artifact or dataset loading."""

    output_dir: Path
    release_version: str
    dataset_names: tuple[str, ...]
    datasets: Mapping[str, ModelDataset]
    model_plan_sha256: str
    model_plan_content_sha256: str
    arrow_dirs: Mapping[str, Path]
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
        "train_pairs_size",
        "validation_pairs_size",
    ):
        _positive_int_arg(args, numeric_arg)
    if int(args.random_seed) < 0:
        raise SystemExit(f"--random-seed must be non-negative, got {args.random_seed}")

    if not bool(args.run_full):
        raise SystemExit("production training requires --run-full")
    if getattr(args, "fixed_parameters", None) is not None:
        if getattr(args, "incumbent_parameters", None) is not None:
            raise SystemExit("--fixed-parameters replaces the search; do not combine it with --incumbent-parameters")
        _load_fixed_parameters(args.fixed_parameters)
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
    try:
        require_arrow_plan(model_plan.datasets)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    dataset_names = tuple(model_plan.datasets)
    if "augmented" in model_plan.datasets and model_plan.datasets["augmented"].split_mode != "fixed_pairs":
        raise SystemExit("model plan augmented dataset requires fixed train and validation pairs")
    if not any(dataset.split_mode == "random_blocks" for dataset in model_plan.datasets.values()):
        raise SystemExit("model plan requires a random-block dataset for EPS calibration")

    # The plan binds each Arrow manifest by digest, and ArrowDataset.open verifies the files.
    arrow_dirs = {name: model_plan.datasets[name].files["arrow"].parent for name in dataset_names}

    return PairwisePreflightPlan(
        output_dir=output_dir,
        release_version=model_plan.release_version,
        dataset_names=dataset_names,
        datasets=model_plan.datasets,
        model_plan_sha256=model_plan.sha256,
        model_plan_content_sha256=model_plan.content_sha256,
        arrow_dirs=arrow_dirs,
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
        "dataset_inputs": {
            name: {
                "arrow_dir": str(plan.arrow_dirs[name]),
                "files": {role: str(path) for role, path in plan.datasets[name].files.items()},
                "split_mode": plan.datasets[name].split_mode,
            }
            for name in plan.dataset_names
        },
        "featurization_backend": "rust_from_model_plan_arrow",
        "features_to_use": list(DEFAULT_FEATURE_GROUPS),
        "training_scope": "production_full",
        "input_artifact_hashes": {str(key): str(value) for key, value in artifact_hashes.items()},
        "n_iter": int(args.n_iter),
        "n_jobs": int(args.n_jobs),
        "nan_policy": "preserve_nan",
        "nameless_features_to_use": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "model_plan_sha256": plan.model_plan_sha256,
        "model_plan_content_sha256": plan.model_plan_content_sha256,
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
    train: TupleOfArrays | None = None,
    val: TupleOfArrays | None = None,
) -> dict[str, Path]:
    dataset_root = root / dataset_name
    dataset_root.mkdir()
    arrays: dict[str, Path] = {}
    for split_name, split in (("train", train), ("val", val)):
        if split is None:
            continue
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


def _featurize_mined_block(
    block: TrainingBlock,
    featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo,
    *,
    name_tuples: frozenset[tuple[str, str]],
    name_counts_sha256: str,
    n_jobs: int,
    total_ram_bytes: int | None,
) -> TupleOfArrays:
    if sha256_file(block.arrow_dir / "manifest.json") != block.arrow_manifest_sha256:
        raise ValueError(f"Mined pair Arrow dataset changed after verification: {block.arrow_dir}")
    with ArrowDataset.open(block.arrow_dir, require_specter=True, require_name_counts_index=True) as arrow:
        if arrow.name_counts_index is None or arrow.name_counts_index.manifest_sha256 != name_counts_sha256:
            raise ValueError(f"Wrong name-counts index for mined pairs in {block.arrow_dir}")
        rust = build_rust_featurizer_from_arrow_dataset(
            arrow,
            signature_ids=sorted({sid for pair in block.fit for sid in (pair.left, pair.right)}),
            name_tuples=name_tuples,
            num_threads=n_jobs,
            use_orcid_id=False,
        )
        return featurize_pairs_with_rust(
            rust,
            [(pair.left, pair.right, pair.label) for pair in block.fit],
            featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            n_jobs=n_jobs,
            nan_value=np.nan,
            total_ram_bytes=total_ram_bytes,
        )


def _featurize_selection(
    dataset: ANDData,
    arrow_dataset: ArrowDataset,
    featurizer_info: FeaturizationInfo,
    *,
    name_counts_manifest_sha256: str,
    n_jobs: int,
    nameless_featurizer_info: FeaturizationInfo,
    total_ram_bytes: int | None,
    unknown_signature_ids: frozenset[str] = frozenset(),
) -> tuple[TupleOfArrays, TupleOfArrays]:
    """Featurize release train/validation pairs and never resolve a test split.

    The Arrow-backed dataset selects the pairs. A Rust featurizer built only for the
    selected pairs' signatures computes their features.
    """

    train_pairs, val_pairs = resolve_selection_pairs(dataset)
    selected_counts = (len(train_pairs), len(val_pairs))
    train_pairs = exclude_unknown_pairs(train_pairs, unknown_signature_ids)
    val_pairs = exclude_unknown_pairs(val_pairs, unknown_signature_ids)
    logger.info(
        "unknown pair exclusions dataset=%s train=%d validation=%d",
        dataset.name,
        selected_counts[0] - len(train_pairs),
        selected_counts[1] - len(val_pairs),
    )
    rust_featurizer = build_verified_rust_featurizer(
        dataset,
        arrow_dataset,
        signature_ids={str(pair[index]) for pair in (*train_pairs, *val_pairs) for index in (0, 1)},
        name_counts_manifest_sha256=name_counts_manifest_sha256,
        num_threads=n_jobs,
    )
    train, val = (
        featurize_pairs_with_rust(
            rust_featurizer,
            pairs,
            featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            n_jobs=n_jobs,
            nan_value=np.nan,
            total_ram_bytes=total_ram_bytes,
        )
        for pairs in (train_pairs, val_pairs)
    )
    return train, val


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


def _load_head_parameters(path: Path | None, *, option: str, require_both: bool) -> dict[str, dict[str, Any]]:
    """Read per-head LightGBM parameters and keep every tree inside the native fast plan."""

    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = {"main", "nameless"}
    if (
        not isinstance(payload, dict)
        or not payload
        or not (set(payload) == heads if require_both else set(payload) <= heads)
        or any(not isinstance(parameters, dict) or not parameters for parameters in payload.values())
    ):
        scope = "both 'main' and 'nameless'" if require_both else "'main' and/or 'nameless'"
        raise SystemExit(f"{option} must map {scope} to LightGBM parameters")
    for head, parameters in payload.items():
        num_leaves = parameters.get("num_leaves")
        if (
            isinstance(num_leaves, bool)
            or not isinstance(num_leaves, int)
            or not 2 <= num_leaves <= MAX_FAST_PLAN_LEAVES
        ):
            raise SystemExit(
                f"{option} {head}.num_leaves must be an integer from 2 to {MAX_FAST_PLAN_LEAVES} "
                "so native scoring keeps a fast plan"
            )
    return payload


def _load_incumbent_parameters(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read optional LightGBM parameters that compete with each head's search winner."""

    return _load_head_parameters(path, option="--incumbent-parameters", require_both=False)


def _load_fixed_parameters(path: Path | None) -> dict[str, dict[str, Any]]:
    """Read the parameters that replace the search for both heads."""

    return _load_head_parameters(path, option="--fixed-parameters", require_both=True)


def _fit_pairwise_model(
    args: argparse.Namespace,
    arrays: Mapping[str, Path],
    *,
    feature_prefix: str,
    monotone_constraints: Any,
    feature_names: list[str],
    incumbent: Mapping[str, Any] | None = None,
    fixed: Mapping[str, Any] | None = None,
) -> tuple[PairwiseModeler, float, dict[str, Any]]:
    model = PairwiseModeler(
        n_iter=int(args.n_iter),
        n_jobs=int(args.n_jobs),
        monotone_constraints=monotone_constraints,
    )
    X_train = _load_staged_array(arrays[f"{feature_prefix}X_train"])
    y_train = _load_staged_array(arrays["y_train"])
    X_val = _load_staged_array(arrays[f"{feature_prefix}X_val"])
    y_val = _load_staged_array(arrays["y_val"])
    if fixed is not None:
        # Fit the search's own estimator with known parameters and skip the search, so a
        # refit on corrected data repeats the chosen recipe exactly.
        model.search_space = {}
        # The run's thread budget and the featurizer's constraint string override the file.
        applied = {**fixed, "n_jobs": int(args.n_jobs)}
        if monotone_constraints is not None:
            applied["monotone_constraints"] = monotone_constraints
        model.estimator.set_params(**applied)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        # Record what the booster used, so the summary can seed a later refit unchanged.
        model.best_params = applied
        validation_roc_auc = _finite_validation_roc_auc(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
        gc.collect()
        return model, validation_roc_auc, {"selected_source": "fixed", "fixed_validation_roc_auc": validation_roc_auc}
    # Named fits write feature names into the saved boosters, as in the released 1.3 model.
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    validation_roc_auc = _finite_validation_roc_auc(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    selection: dict[str, Any] = {"selected_source": "search", "search_validation_roc_auc": validation_roc_auc}
    if incumbent is not None:
        # A known configuration competes on the same train and validation rows; it wins ties,
        # so the search must improve on it to replace it.
        parameters = {
            **incumbent,
            "n_jobs": int(args.n_jobs),
            "random_state": 42,
            "monotone_constraints": monotone_constraints,
        }
        classifier = LGBMClassifier().set_params(**parameters).fit(X_train, y_train, feature_name=feature_names)
        # A named classifier warns on bare arrays; the checked path names the columns.
        probabilities = np.asarray(
            classifier.predict_proba(_validated_classifier_features(classifier, X_val)), dtype=np.float64
        )[:, 1]
        incumbent_roc_auc = _finite_validation_roc_auc(roc_auc_score(y_val, probabilities))
        selection["incumbent_validation_roc_auc"] = incumbent_roc_auc
        if incumbent_roc_auc >= validation_roc_auc:
            model.classifier = classifier
            model.best_params = dict(incumbent)
            validation_roc_auc = incumbent_roc_auc
            selection["selected_source"] = "incumbent"
    gc.collect()
    return model, validation_roc_auc, selection


def _publish_result(
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
    mined_csv = getattr(args, "mined_pairs", None)
    assignments = getattr(args, "block_role_assignments", None)
    evidence = getattr(args, "identity_evidence", None)
    mined_args = (mined_csv, assignments, evidence)
    if any(value is None for value in mined_args) and any(value is not None for value in mined_args):
        raise ValueError("Use --mined-pairs, --identity-evidence and --block-role-assignments together")
    unknown_inputs = {name: plan_unknown_signatures(plan.datasets[name]) for name in plan.dataset_names}
    unknown_ids = {name: source.ids for name, source in unknown_inputs.items()}
    augmented_members: dict[str, set[str]] = {}
    augmented_validation_members: dict[str, set[str]] = {}
    if "augmented" in plan.datasets:
        unknown_ids["augmented"] = augmented_unknown_ids(unknown_ids)
        for role, population, members in (
            ("train_pairs", "training", augmented_members),
            ("val_pairs", "validation", augmented_validation_members),
        ):
            if role not in plan.datasets["augmented"].files:
                continue
            pair_frame = _map_fixed_pair_labels(
                pd.read_csv(plan.datasets["augmented"].files[role]), f"augmented {population}"
            )
            members.update(
                source_members(
                    (str(left), str(right))
                    for left, right in pair_frame.iloc[:, :2].itertuples(index=False, name=None)
                    if str(left) not in unknown_ids["augmented"] and str(right) not in unknown_ids["augmented"]
                )
            )
            require_declared_sources(members, plan.dataset_names, context=f"Augmented {population}")
            del pair_frame
    # Copies outside each source fit population must not enter the fit, and validation
    # copies outside the source validation population must not select the model. Every
    # source is checked before artifacts load and before any dataset is featurized.
    for dataset_name in plan.dataset_names:
        if dataset_name in augmented_members or dataset_name in augmented_validation_members:
            training, validation = development_split_members(
                dataset_name, plan.datasets[dataset_name], random_seed=int(args.random_seed)
            )
            require_training_members(augmented_members, dataset_name, training)
            require_validation_members(augmented_validation_members, dataset_name, validation)
    incumbents = _load_incumbent_parameters(getattr(args, "incumbent_parameters", None))
    fixed_parameters = _load_fixed_parameters(getattr(args, "fixed_parameters", None))
    artifact_authority = load_packaged_artifact_authority(
        name_counts_index_root=Path(args.name_counts_index_root),
    )
    canonical_name_tuples = artifact_authority.name_tuples
    artifact_hashes = _canonical_training_artifact_hashes(artifact_authority)
    mined = None
    if mined_csv is not None:
        roles = load_block_roles(assignments)
        ledger = ledger_for_roles(evidence, roles)
        assert ledger is not None
        mined = training_pairs(
            roles, ledger, mined_csv, name_counts_sha256=artifact_hashes["name_counts_manifest_sha256"]
        )
    mined_blocks = mined.blocks if mined is not None else {}
    training_config = _training_config(
        args,
        plan,
        artifact_hashes=artifact_hashes,
    )
    training_config["unknown_signature_inputs"] = {name: source.binding() for name, source in unknown_inputs.items()}
    training_config["unknown_signature_policy"] = "exclude_pair_endpoints; retain_eps_inputs; score_known_members_only"
    training_config["incumbent_parameters"] = incumbents
    training_config["fixed_parameters"] = fixed_parameters
    if mined is not None:
        training_config["mined_pairs"] = mined.binding
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
    mined_summary = {
        block: {"split": spec.fold, "labeled_pairs": len(spec.fit), "unscored_pairs": len(spec.judge)}
        for block, spec in mined_blocks.items()
    }
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
            with open_plan_arrow(dataset_input) as arrow_dataset:
                # The same Arrow dataset selects pairs and computes their features.
                anddata = load_development_dataset(
                    dataset_name,
                    dataset_input,
                    arrow_dataset,
                    name_tuples=canonical_name_tuples.pairs,
                    random_seed=int(args.random_seed),
                    n_jobs=int(args.n_jobs),
                    train_pairs_size=int(args.train_pairs_size),
                    val_pairs_size=int(args.validation_pairs_size),
                )
                if anddata.name_tuples != canonical_name_tuples.pairs:
                    raise ValueError(
                        f"Production training dataset {dataset_name!r} does not use the canonical name tuples"
                    )

                if dataset_input.split_mode == "random_blocks":
                    # Resolve the same split as pair selection, before allocating feature matrices.
                    train_blocks, val_blocks, test_blocks = anddata.split_cluster_signatures()
                    require_training_members(
                        augmented_members,
                        dataset_name,
                        {signature_id for members in train_blocks.values() for signature_id in members},
                    )
                    require_validation_members(
                        augmented_validation_members,
                        dataset_name,
                        {signature_id for members in val_blocks.values() for signature_id in members},
                    )
                    cluster_test_splits[dataset_name] = {
                        "block_membership_sha256": block_membership_sha256(anddata.get_blocks()),
                        "test_block_ids": list(test_blocks),
                    }
                    del train_blocks, val_blocks, test_blocks

                train, val = _featurize_selection(
                    anddata,
                    arrow_dataset,
                    featurizer_info,
                    name_counts_manifest_sha256=artifact_hashes["name_counts_manifest_sha256"],
                    n_jobs=int(args.n_jobs),
                    nameless_featurizer_info=nameless_featurizer_info,
                    total_ram_bytes=plan.total_ram_bytes,
                    unknown_signature_ids=unknown_ids[dataset_name],
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

        mined_arrays: list[dict[str, Path]] = []
        if mined_blocks:
            mined_root = Path(tempfile.mkdtemp(dir=matrix_work_dir, prefix=".mined_pairs_"))
            for index, (block, spec) in enumerate(mined_blocks.items()):
                logger.info(
                    "mined pairs block=%s split=%s labeled=%d unscored=%d",
                    block,
                    spec.fold,
                    len(spec.fit),
                    len(spec.judge),
                )
                if not spec.fit:
                    continue
                features = _featurize_mined_block(
                    spec,
                    featurizer_info,
                    nameless_featurizer_info,
                    name_tuples=canonical_name_tuples.pairs,
                    name_counts_sha256=artifact_hashes["name_counts_manifest_sha256"],
                    n_jobs=int(args.n_jobs),
                    total_ram_bytes=plan.total_ram_bytes,
                )
                mined_arrays.append(_stage_dataset_features(mined_root, str(index), **{spec.fold: features}))
                del features

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
                [staged_datasets[name][role] for name in names]
                + [paths[role] for paths in mined_arrays if role in paths],
            )
            for role, names in union_members.items()
        }

        logger.info("fitting pairwise model")
        union_classifier, main_validation_roc_auc, main_selection = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="",
            monotone_constraints=monotone_constraints,
            feature_names=featurizer_info.get_feature_names(),
            incumbent=incumbents.get("main"),
            fixed=fixed_parameters.get("main"),
        )

        logger.info("fitting nameless pairwise model")
        nameless_union_classifier, nameless_validation_roc_auc, nameless_selection = _fit_pairwise_model(
            args,
            union_arrays,
            feature_prefix="nameless_",
            monotone_constraints=nameless_monotone_constraints,
            feature_names=nameless_featurizer_info.get_feature_names(),
            incumbent=incumbents.get("nameless"),
            fixed=fixed_parameters.get("nameless"),
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
            **{f"main_{key}": value for key, value in main_selection.items()},
            **{f"nameless_{key}": value for key, value in nameless_selection.items()},
        }
        if mined is not None:
            training_summary["mined_pairs_by_block"] = mined_summary
        logger.info("pairwise training completed in %.3f seconds", time.perf_counter() - started)
    result = _publish_result(
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
    # Require an explicit allocation; a fixed default can exceed the shared CPU budget.
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--train-pairs-size", type=int, default=DEFAULT_TRAIN_PAIRS_SIZE)
    parser.add_argument(
        "--validation-pairs-size",
        type=int,
        default=DEFAULT_VALIDATION_PAIRS_SIZE,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--incumbent-parameters",
        type=Path,
        default=None,
        help="Optional JSON of LightGBM parameters per head that compete with each search winner.",
    )
    parser.add_argument(
        "--fixed-parameters",
        type=Path,
        default=None,
        help="Optional JSON of LightGBM parameters for both heads; fits them directly instead of searching.",
    )
    parser.add_argument("--matrix-work-dir", type=Path, required=True)
    parser.add_argument("--mined-pairs", type=Path, help="CSV and sidecar from mine_pairwise_pairs.")
    parser.add_argument("--identity-evidence", type=Path, help="Identity ledger used to label mined pairs.")
    parser.add_argument("--block-role-assignments", type=Path, help="Master assignment that defines the mined fold.")
    parser.add_argument(
        "--model-plan",
        type=Path,
        required=True,
        help="Arrow datasets, labels, held-out blocks, and EPS policy for this release run.",
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
