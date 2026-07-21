"""Build native production model bundles from trained S2AND artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.consts import NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, _selected_feature_indices
from s2and.model_pairwise import _validated_classifier_features
from s2and.production_bundle_contract import (
    CLUSTERER_CONFIG_SCHEMA_VERSION,
    PAIRWISE_METADATA_SCHEMA_VERSION,
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    production_manifest_files,
)
from s2and.production_model import (
    _load_pairwise_staging_model,
    load_production_model,
    pairwise_bundle_binding,
    require_canonical_artifact_hashes,
)

PAIRWISE_FIXTURE_SEED = 921
PAIRWISE_FIXTURE_ROWS = 8


@dataclass(frozen=True)
class ProductionBundleSummary:
    """Files and status for a written production bundle."""

    bundle_dir: Path
    bundle_version: str
    bundle_status: str
    manifest_path: Path
    files: tuple[str, ...]


def production_version_from_bundle_dir(bundle_dir: Path) -> str | None:
    """Infer ``X.Y`` from a ``production_model_vX.Y`` directory name."""

    prefix = "production_model_v"
    name = Path(bundle_dir).name
    if name.startswith(prefix):
        return name[len(prefix) :]
    return None


def _validate_named_bundle_version(bundle_dir: Path, bundle_version: str) -> None:
    if not str(bundle_version):
        raise ValueError("Production bundle_version must be nonempty")
    inferred_version = production_version_from_bundle_dir(bundle_dir)
    if inferred_version is not None and inferred_version != str(bundle_version):
        raise ValueError(
            "Production bundle directory name and bundle_version disagree: "
            f"directory={Path(bundle_dir).name!r} bundle_version={str(bundle_version)!r}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _booster_from_model(model: Any) -> lgb.Booster:
    if isinstance(model, lgb.Booster):
        return model
    inner = getattr(model, "classifier", None)
    if inner is not None and inner is not model:
        return _booster_from_model(inner)
    booster = getattr(model, "booster_", None)
    if isinstance(booster, lgb.Booster):
        return booster
    booster = getattr(model, "_Booster", None)
    if isinstance(booster, lgb.Booster):
        return booster
    raise TypeError(f"Expected a fitted LightGBM model, got {type(model)!r}")


def _predict_proba(model: Any, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64, order="C")
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(
            predict_proba(_validated_classifier_features(model, matrix)),
            dtype=np.float64,
        )
    else:
        positive = np.asarray(_booster_from_model(model).predict(matrix), dtype=np.float64).reshape(-1)
        probabilities = np.column_stack((1.0 - positive, positive))
    if probabilities.ndim == 1:
        probabilities = np.column_stack((1.0 - probabilities, probabilities))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(f"Expected binary probability matrix, got shape={probabilities.shape}")
    return probabilities


def _featurization_info_payload(featurizer_info: FeaturizationInfo) -> dict[str, Any]:
    return {
        "features_to_use": [str(feature) for feature in featurizer_info.features_to_use],
        "featurizer_version": int(featurizer_info.featurizer_version),
    }


def _cluster_model_payload(clusterer: Clusterer) -> dict[str, Any]:
    cluster_model = clusterer.cluster_model
    return {
        "eps": float(cluster_model.eps),
        "family": type(cluster_model).__name__,
        "input_as_observation_matrix": bool(getattr(cluster_model, "input_as_observation_matrix", False)),
        "linkage": str(cluster_model.linkage),
        "preserve_input": bool(getattr(cluster_model, "preserve_input", True)),
    }


def _clusterer_config_payload(
    clusterer: Clusterer,
    *,
    nameless_featurizer_info: FeaturizationInfo,
) -> dict[str, Any]:
    raw_feature_contract = getattr(clusterer, "feature_contract", None)
    if not isinstance(raw_feature_contract, Mapping):
        raise ValueError("Production bundle export requires an explicit source feature_contract")
    feature_contract = dict(raw_feature_contract)
    source_normalization_version = feature_contract.get("normalization_version")
    if source_normalization_version is None:
        raise ValueError(
            "Production bundle export requires feature_contract['normalization_version']; "
            "missing provenance is legacy and cannot be relabeled"
        )
    if source_normalization_version != NORMALIZATION_VERSION:
        raise ValueError(
            "Production bundle normalization_version mismatch: "
            f"source={source_normalization_version!r} package={NORMALIZATION_VERSION!r}"
        )
    require_canonical_artifact_hashes(feature_contract, context="Production bundle export feature_contract")
    return {
        "batch_size": int(getattr(clusterer, "batch_size", 1_000_000)),
        "cluster_model": _cluster_model_payload(clusterer),
        "dont_merge_cluster_seeds": bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
        "feature_contract": feature_contract,
        "featurizer_info": _featurization_info_payload(clusterer.featurizer_info),
        "incremental_mean_min_hybrid_weight": float(getattr(clusterer, "incremental_mean_min_hybrid_weight", 0.5)),
        "incremental_precluster_broadcast_mode": str(
            getattr(clusterer, "incremental_precluster_broadcast_mode", "always")
        ),
        "incremental_seed_score_mode": str(getattr(clusterer, "incremental_seed_score_mode", "mean")),
        "n_iter": int(getattr(clusterer, "n_iter", 25)),
        "n_jobs": 1,
        "nameless_featurizer_info": _featurization_info_payload(nameless_featurizer_info),
        "random_state": int(getattr(clusterer, "random_state", 42)),
        "schema_version": CLUSTERER_CONFIG_SCHEMA_VERSION,
        "suppress_orcid": bool(getattr(clusterer, "suppress_orcid", False)),
        "use_default_constraints_as_supervision": bool(
            getattr(clusterer, "use_default_constraints_as_supervision", True)
        ),
        "val_blocks_size": getattr(clusterer, "val_blocks_size", None),
    }


def _pairwise_metadata_payload() -> dict[str, Any]:
    return {
        "class_labels": [0.0, 1.0],
        "distance_probability_column": "class_0",
        "model_family": "binary_lightgbm_pairwise_distance",
        "positive_probability_column": "class_1",
        "schema_version": PAIRWISE_METADATA_SCHEMA_VERSION,
    }


def _write_pairwise_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _booster_from_model(model).save_model(str(path))


def _write_pairwise_fixture(model: Any, path: Path, *, width: int, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    features = rng.normal(size=(PAIRWISE_FIXTURE_ROWS, int(width)))
    payload = {
        "atol": 1e-10,
        "expected_probabilities": _predict_proba(model, features).tolist(),
        "feature_source": "numpy_default_rng_normal",
        "features": features.tolist(),
        "rtol": 1e-10,
        "schema_version": PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
        "seed": int(seed),
    }
    _write_json(path, payload)


def _pairwise_reproducibility_present(bundle_dir: Path) -> bool:
    paths = [bundle_dir / relpath for relpath in PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES.values()]
    present = [path.is_file() for path in paths]
    if any(present) and not all(present):
        raise ValueError("Production bundles require both pairwise training reproducibility files or neither")
    return all(present)


def write_production_manifest(
    bundle_dir: Path,
    *,
    bundle_version: str,
    pairwise_model_version: str,
    include_incremental_linker: bool,
    incremental_linker_version: str | None = None,
) -> ProductionBundleSummary:
    """Write the bundle manifest for either pairwise-only or complete bundles."""

    bundle_dir = Path(bundle_dir)
    if include_incremental_linker:
        if not isinstance(incremental_linker_version, str) or not incremental_linker_version.strip():
            raise ValueError("Complete production manifests require a nonempty incremental_linker_version")
    elif incremental_linker_version is not None:
        raise ValueError("Pairwise-only production manifests cannot declare an incremental_linker_version")
    files = production_manifest_files(
        complete=include_incremental_linker,
        include_pairwise_reproducibility=_pairwise_reproducibility_present(bundle_dir),
    )
    sha256: dict[str, str] = {}
    for relpath in sorted(set(files.values())):
        path = bundle_dir / relpath
        if path.is_dir():
            continue
        if not path.exists():
            raise FileNotFoundError(f"Production bundle file is missing: {path}")
        sha256[relpath] = _sha256_file(path)

    status = "complete" if include_incremental_linker else "pairwise_only"
    manifest = {
        "bundle_status": status,
        "bundle_version": str(bundle_version),
        "files": files,
        "incremental_linker_version": incremental_linker_version if include_incremental_linker else None,
        "pairwise_model_version": str(pairwise_model_version),
        "schema_version": PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
        "sha256": sha256,
    }
    manifest_path = bundle_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return ProductionBundleSummary(
        bundle_dir=bundle_dir,
        bundle_version=str(bundle_version),
        bundle_status=status,
        manifest_path=manifest_path,
        files=tuple(sorted(sha256)),
    )


def _write_pairwise_production_bundle_stage(
    clusterer: Clusterer,
    bundle_dir: Path,
    *,
    bundle_version: str,
    source_model_version: str | None = None,
    pairwise_training_config: Mapping[str, Any] | None = None,
    pairwise_training_summary: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Write a pairwise bundle into a private staging directory."""

    if (pairwise_training_config is None) != (pairwise_training_summary is None):
        raise ValueError("Pairwise training config and summary must be provided together")
    nameless_featurizer_info = clusterer.nameless_featurizer_info
    if clusterer.nameless_classifier is None or nameless_featurizer_info is None:
        raise ValueError("Production bundles require a nameless pairwise model")

    bundle_dir = Path(bundle_dir)
    pairwise_dir = bundle_dir / "pairwise"
    source_version = str(source_model_version or bundle_version)
    main_width = len(_selected_feature_indices(clusterer.featurizer_info))
    nameless_width = len(_selected_feature_indices(nameless_featurizer_info))
    clusterer_payload = _clusterer_config_payload(
        clusterer,
        nameless_featurizer_info=nameless_featurizer_info,
    )
    pairwise_metadata_payload = _pairwise_metadata_payload()

    _write_pairwise_model(clusterer.classifier, pairwise_dir / "main.lgb")
    _write_pairwise_model(clusterer.nameless_classifier, pairwise_dir / "nameless.lgb")
    _write_pairwise_fixture(
        clusterer.classifier,
        pairwise_dir / "main_prediction_fixture.json",
        width=main_width,
        seed=PAIRWISE_FIXTURE_SEED,
    )
    _write_pairwise_fixture(
        clusterer.nameless_classifier,
        pairwise_dir / "nameless_prediction_fixture.json",
        width=nameless_width,
        seed=PAIRWISE_FIXTURE_SEED + 1,
    )
    _write_json(
        pairwise_dir / "metadata.json",
        pairwise_metadata_payload,
    )
    _write_json(bundle_dir / "clusterer.json", clusterer_payload)

    reproducibility_dir = bundle_dir / "reproducibility"
    if pairwise_training_config is not None:
        _write_json(reproducibility_dir / "pairwise_training_config.json", dict(pairwise_training_config))
    if pairwise_training_summary is not None:
        _write_json(reproducibility_dir / "pairwise_training_summary.json", dict(pairwise_training_summary))

    return write_production_manifest(
        bundle_dir,
        bundle_version=str(bundle_version),
        pairwise_model_version=source_version,
        include_incremental_linker=False,
    )


def write_pairwise_production_bundle(
    clusterer: Clusterer,
    bundle_dir: Path,
    *,
    bundle_version: str,
    source_model_version: str | None = None,
    pairwise_training_config: Mapping[str, Any] | None = None,
    pairwise_training_summary: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Atomically publish the pairwise stage of a native production bundle."""

    bundle_dir = Path(bundle_dir)
    _validate_named_bundle_version(bundle_dir, str(bundle_version))
    if bundle_dir.exists():
        raise FileExistsError(f"Production bundle output already exists; choose a new directory: {bundle_dir}")

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}.pairwise-staging-", dir=bundle_dir.parent))
    try:
        staged_summary = _write_pairwise_production_bundle_stage(
            clusterer,
            staging_dir,
            bundle_version=str(bundle_version),
            source_model_version=source_model_version,
            pairwise_training_config=pairwise_training_config,
            pairwise_training_summary=pairwise_training_summary,
        )
        _load_pairwise_staging_model(staging_dir)
        _fsync_tree(staging_dir)
        _publish_staged_bundle(staging_dir, bundle_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return ProductionBundleSummary(
        bundle_dir=bundle_dir,
        bundle_version=staged_summary.bundle_version,
        bundle_status=staged_summary.bundle_status,
        manifest_path=bundle_dir / "manifest.json",
        files=staged_summary.files,
    )


def _copy_path(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_pairwise_stage(source_bundle_dir: Path, output_bundle_dir: Path) -> None:
    for relpath in ("clusterer.json", "pairwise"):
        _copy_path(source_bundle_dir / relpath, output_bundle_dir / relpath)
    source_reproducibility = source_bundle_dir / "reproducibility"
    if source_reproducibility.exists():
        for path in source_reproducibility.iterdir():
            if path.name == "incremental_linker_training_target.json":
                continue
            _copy_path(path, output_bundle_dir / "reproducibility" / path.name)


def _fsync_tree(root: Path) -> None:
    for path in sorted((path for path in root.rglob("*") if path.is_file()), key=lambda item: item.as_posix()):
        if os.name == "nt":
            # The Windows CRT rejects fsync on a read-only descriptor. Staging
            # files are private, so temporarily add owner-write permission,
            # flush through a writable descriptor, then restore the copied mode.
            original_mode = path.stat().st_mode
            restore_mode = not bool(original_mode & stat.S_IWRITE)
            if restore_mode:
                path.chmod(original_mode | stat.S_IWRITE)
            try:
                with path.open("rb+") as handle:
                    os.fsync(handle.fileno())
            finally:
                if restore_mode:
                    path.chmod(original_mode)
        else:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    if os.name != "nt":
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in sorted(
            directories,
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _publish_staged_bundle(staging_dir: Path, destination: Path) -> None:
    """Rename one complete staging directory into a new destination."""

    lock_path = destination.parent / f".{destination.name}.publish.lock"
    with exclusive_file_lock(lock_path):
        if destination.exists():
            raise FileExistsError(f"Production bundle output already exists; choose a new directory: {destination}")
        os.replace(staging_dir, destination)
        fsync_directory(destination.parent)


def finalize_production_bundle(
    *,
    pairwise_bundle_dir: Path,
    output_bundle_dir: Path,
    incremental_linker_artifact_dir: Path,
    target_json: Path,
    bundle_version: str | None = None,
    pairwise_model_version: str | None = None,
    incremental_linker_version: str | None = None,
) -> ProductionBundleSummary:
    """Assemble a complete production bundle from pairwise and linker artifacts."""

    pairwise_bundle_dir = Path(pairwise_bundle_dir)
    output_bundle_dir = Path(output_bundle_dir)
    incremental_linker_artifact_dir = Path(incremental_linker_artifact_dir)
    target_json = Path(target_json)
    if not pairwise_bundle_dir.is_dir():
        raise FileNotFoundError(f"Pairwise bundle directory does not exist: {pairwise_bundle_dir}")
    if output_bundle_dir.exists():
        raise FileExistsError(
            f"Production bundle output already exists; finalization requires a new directory: {output_bundle_dir}"
        )
    if not incremental_linker_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Incremental linker artifact directory does not exist: {incremental_linker_artifact_dir}"
        )
    if not target_json.exists():
        raise FileNotFoundError(f"Incremental linker target JSON does not exist: {target_json}")

    inferred_version = production_version_from_bundle_dir(output_bundle_dir)
    if bundle_version is not None:
        _validate_named_bundle_version(output_bundle_dir, str(bundle_version))
    resolved_bundle_version = str(
        bundle_version or inferred_version or production_version_from_bundle_dir(pairwise_bundle_dir) or ""
    )
    if not resolved_bundle_version:
        raise ValueError("bundle_version is required when output_bundle_dir is not named production_model_vX.Y")

    pairwise_binding = pairwise_bundle_binding(pairwise_bundle_dir)
    linker_metadata = _read_json(incremental_linker_artifact_dir / "metadata.json")
    if linker_metadata.get("pairwise_bundle_binding") != pairwise_binding:
        raise ValueError("Incremental linker pairwise_bundle_binding does not match pairwise bundle")

    output_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_bundle_dir.name}.staging-", dir=output_bundle_dir.parent))
    try:
        _copy_pairwise_stage(pairwise_bundle_dir, staging_dir)
        _copy_path(incremental_linker_artifact_dir, staging_dir / "incremental_linker")
        target_destination = staging_dir / "reproducibility" / "incremental_linker_training_target.json"
        _copy_path(target_json, target_destination)
        staged_summary = write_production_manifest(
            staging_dir,
            bundle_version=resolved_bundle_version,
            pairwise_model_version=str(
                pairwise_model_version
                or production_version_from_bundle_dir(pairwise_bundle_dir)
                or resolved_bundle_version
            ),
            include_incremental_linker=True,
            incremental_linker_version=str(incremental_linker_version or resolved_bundle_version),
        )
        load_production_model(staging_dir)
        _fsync_tree(staging_dir)
        _publish_staged_bundle(staging_dir, output_bundle_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return ProductionBundleSummary(
        bundle_dir=output_bundle_dir,
        bundle_version=staged_summary.bundle_version,
        bundle_status=staged_summary.bundle_status,
        manifest_path=output_bundle_dir / "manifest.json",
        files=staged_summary.files,
    )
