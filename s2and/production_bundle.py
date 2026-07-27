"""Build native production model bundles from trained S2AND artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from s2and.consts import NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.model import Clusterer, _selected_feature_indices
from s2and.model_pairwise import _validated_classifier_features, lightgbm_booster
from s2and.production_bundle_contract import (
    CLUSTERER_CONFIG_SCHEMA_VERSION,
    PAIRWISE_ONLY_MANIFEST_FILES,
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PAIRWISE_PREDICTION_FIXTURE_TOLERANCE,
    PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    production_bundle_status,
    production_manifest_files,
)
from s2and.production_model import (
    _load_pairwise_staging_model,
    load_production_model,
    pairwise_bundle_binding,
    require_canonical_artifact_hashes,
    require_expected_artifact_hashes,
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


def _predict_proba(model: Any, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64, order="C")
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(
            predict_proba(_validated_classifier_features(model, matrix)),
            dtype=np.float64,
        )
    else:
        positive = np.asarray(lightgbm_booster(model).predict(matrix), dtype=np.float64).reshape(-1)
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
        "linkage": str(cluster_model.linkage),
    }


def _clusterer_config_payload(
    clusterer: Clusterer,
    *,
    nameless_featurizer_info: FeaturizationInfo,
    expected_artifact_hashes: Mapping[str, Any] | None,
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
    if expected_artifact_hashes is None:
        require_canonical_artifact_hashes(feature_contract, context="Production bundle export feature_contract")
    else:
        require_expected_artifact_hashes(
            feature_contract,
            expected_artifact_hashes,
            context="Production bundle export feature_contract",
        )
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
        "n_jobs": 1,
        "nameless_featurizer_info": _featurization_info_payload(nameless_featurizer_info),
        "random_state": int(getattr(clusterer, "random_state", 42)),
        "schema_version": CLUSTERER_CONFIG_SCHEMA_VERSION,
        "suppress_orcid": bool(getattr(clusterer, "suppress_orcid", False)),
        "use_default_constraints_as_supervision": bool(
            getattr(clusterer, "use_default_constraints_as_supervision", True)
        ),
    }


def _write_pairwise_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lightgbm_booster(model).save_model(str(path))


def _write_pairwise_fixture(model: Any, path: Path, *, width: int, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    features = rng.normal(size=(PAIRWISE_FIXTURE_ROWS, int(width)))
    payload = {
        "atol": PAIRWISE_PREDICTION_FIXTURE_TOLERANCE,
        "expected_probabilities": _predict_proba(model, features).tolist(),
        "feature_source": "numpy_default_rng_normal",
        "features": features.tolist(),
        "rtol": PAIRWISE_PREDICTION_FIXTURE_TOLERANCE,
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
    incremental_linker_version: str | None = None,
) -> ProductionBundleSummary:
    """Write the bundle manifest for either pairwise-only or complete bundles."""

    bundle_dir = Path(bundle_dir)
    for version_field, version in (
        ("bundle_version", bundle_version),
        ("pairwise_model_version", pairwise_model_version),
    ):
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Production model bundle {version_field} must be a nonempty string")
    if incremental_linker_version is not None and (
        not isinstance(incremental_linker_version, str) or not incremental_linker_version.strip()
    ):
        raise ValueError("Complete production manifests require a nonempty incremental_linker_version")
    files = production_manifest_files(
        incremental_linker_version=incremental_linker_version,
        include_pairwise_reproducibility=_pairwise_reproducibility_present(bundle_dir),
    )
    sha256: dict[str, str] = {}
    for relpath in sorted(set(files.values())):
        path = bundle_dir / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Production bundle file is missing or not a regular file: {path}")
        sha256[relpath] = _sha256_file(path)

    status = production_bundle_status(incremental_linker_version)
    manifest = {
        "bundle_version": str(bundle_version),
        "incremental_linker_version": incremental_linker_version,
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
    expected_artifact_hashes: Mapping[str, Any] | None = None
    if pairwise_training_config is not None:
        expected_artifact_hashes = pairwise_training_config.get("input_artifact_hashes")
        if not isinstance(expected_artifact_hashes, Mapping) or not expected_artifact_hashes:
            raise ValueError(
                "Pairwise training config must contain nonempty input_artifact_hashes "
                "when it is used to publish a production bundle"
            )
    main_width = len(_selected_feature_indices(clusterer.featurizer_info))
    nameless_width = len(_selected_feature_indices(nameless_featurizer_info))
    clusterer_payload = _clusterer_config_payload(
        clusterer,
        nameless_featurizer_info=nameless_featurizer_info,
        expected_artifact_hashes=expected_artifact_hashes,
    )
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
    _write_json(bundle_dir / "clusterer.json", clusterer_payload)

    reproducibility_dir = bundle_dir / "reproducibility"
    if pairwise_training_config is not None:
        _write_json(reproducibility_dir / "pairwise_training_config.json", dict(pairwise_training_config))
    if pairwise_training_summary is not None:
        _write_json(reproducibility_dir / "pairwise_training_summary.json", dict(pairwise_training_summary))

    return write_production_manifest(
        bundle_dir,
        bundle_version=str(bundle_version),
        pairwise_model_version=str(bundle_version),
    )


def write_pairwise_production_bundle(
    clusterer: Clusterer,
    bundle_dir: Path,
    *,
    bundle_version: str,
    pairwise_training_config: Mapping[str, Any] | None = None,
    pairwise_training_summary: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Atomically publish one versioned pairwise stage."""

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
            pairwise_training_config=pairwise_training_config,
            pairwise_training_summary=pairwise_training_summary,
        )
        expected_artifact_hashes = (
            pairwise_training_config.get("input_artifact_hashes") if pairwise_training_config is not None else None
        )
        _load_pairwise_staging_model(
            staging_dir,
            expected_artifact_hashes=expected_artifact_hashes,
        )
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


def finalize_pairwise_eps(
    *,
    source_bundle_dir: Path,
    output_bundle_dir: Path,
    expected_manifest_sha256: str,
    expected_old_eps: float,
    new_eps: float,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Write a fresh pairwise stage whose only semantic change is clustering EPS."""

    source_bundle_dir = Path(source_bundle_dir)
    output_bundle_dir = Path(output_bundle_dir)
    if output_bundle_dir.exists():
        raise FileExistsError(f"Pairwise EPS output already exists: {output_bundle_dir}")
    manifest_path = source_bundle_dir / "manifest.json"
    observed_manifest_sha256 = _sha256_file(manifest_path)
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Pairwise source manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256} observed={observed_manifest_sha256}"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("incremental_linker_version") is not None:
        raise ValueError("EPS finalization requires a pairwise-only bundle")
    _load_pairwise_staging_model(
        source_bundle_dir,
        expected_artifact_hashes=expected_artifact_hashes,
    )
    config_path = source_bundle_dir / "clusterer.json"
    config = _read_json(config_path)
    observed_eps = float(config["cluster_model"]["eps"])
    if observed_eps != float(expected_old_eps):
        raise ValueError(f"Pairwise source EPS mismatch: expected={expected_old_eps} observed={observed_eps}")
    if not np.isfinite(float(new_eps)) or not 0.0 <= float(new_eps) <= 1.0:
        raise ValueError(f"new_eps must be finite and in [0, 1], got {new_eps!r}")

    output_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_bundle_dir.name}.eps-staging-", dir=output_bundle_dir.parent))
    try:
        _copy_path(source_bundle_dir, staging_dir)
        config["cluster_model"]["eps"] = float(new_eps)
        _write_json(staging_dir / "clusterer.json", config)
        staged_summary = write_production_manifest(
            staging_dir,
            bundle_version=str(manifest["bundle_version"]),
            pairwise_model_version=str(manifest["pairwise_model_version"]),
        )
        source_files = {
            path.relative_to(source_bundle_dir).as_posix(): path
            for path in source_bundle_dir.rglob("*")
            if path.is_file()
        }
        staged_files = {
            path.relative_to(staging_dir).as_posix(): path for path in staging_dir.rglob("*") if path.is_file()
        }
        if set(source_files) != set(staged_files):
            raise RuntimeError("EPS finalization changed the pairwise bundle file inventory")
        for relpath in sorted(set(source_files) - {"clusterer.json", "manifest.json"}):
            if source_files[relpath].read_bytes() != staged_files[relpath].read_bytes():
                raise RuntimeError(f"EPS finalization changed immutable file {relpath}")
        _load_pairwise_staging_model(
            staging_dir,
            expected_artifact_hashes=expected_artifact_hashes,
        )
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


def _copy_path(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_pairwise_stage(
    source_bundle_dir: Path,
    output_bundle_dir: Path,
    source_manifest: Mapping[str, Any],
) -> None:
    declared_paths = set(source_manifest["sha256"])
    for relpath in PAIRWISE_ONLY_MANIFEST_FILES.values():
        _copy_path(source_bundle_dir / relpath, output_bundle_dir / relpath)
    for relpath in PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES.values():
        if relpath in declared_paths:
            _copy_path(source_bundle_dir / relpath, output_bundle_dir / relpath)


def _publish_staged_bundle(staging_dir: Path, destination: Path) -> None:
    """Rename one complete staging directory into a new destination."""

    if destination.exists():
        raise FileExistsError(f"Production bundle output already exists; choose a new directory: {destination}")
    try:
        os.replace(staging_dir, destination)
    except OSError:
        if destination.exists():
            raise FileExistsError(
                f"Production bundle output already exists; choose a new directory: {destination}"
            ) from None
        raise


def finalize_production_bundle(
    *,
    pairwise_bundle_dir: Path,
    output_bundle_dir: Path,
    incremental_linker_artifact_dir: Path,
    target_json: Path,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Assemble a complete bundle using the pairwise manifest as version authority."""

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

    pairwise_manifest = _read_json(pairwise_bundle_dir / "manifest.json")
    pairwise_bundle_version = str(pairwise_manifest.get("bundle_version") or "").strip()
    if not pairwise_bundle_version:
        raise ValueError("Pairwise manifest bundle_version must be a nonempty string")
    inferred_version = production_version_from_bundle_dir(output_bundle_dir)
    if inferred_version is not None and inferred_version != pairwise_bundle_version:
        raise ValueError(
            "Output directory bundle version disagrees with pairwise manifest: "
            f"output={inferred_version!r}, pairwise={pairwise_bundle_version!r}"
        )

    pairwise_binding = pairwise_bundle_binding(
        pairwise_bundle_dir,
        expected_artifact_hashes=expected_artifact_hashes,
    )
    linker_metadata = _read_json(incremental_linker_artifact_dir / "metadata.json")
    if linker_metadata.get("pairwise_bundle_binding_digest") != canonical_json_digest(pairwise_binding):
        raise ValueError("Incremental linker pairwise_bundle_binding_digest does not match pairwise bundle")
    target_spec_digest = canonical_json_digest(_read_json(target_json))
    if linker_metadata.get("target_spec_digest") != target_spec_digest:
        raise ValueError("Incremental linker target_spec_digest does not match target JSON")

    output_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_bundle_dir.name}.staging-", dir=output_bundle_dir.parent))
    try:
        _copy_pairwise_stage(pairwise_bundle_dir, staging_dir, pairwise_manifest)
        for filename in ("booster.lgb", "metadata.json"):
            _copy_path(
                incremental_linker_artifact_dir / filename,
                staging_dir / "incremental_linker" / filename,
            )
        target_destination = staging_dir / "reproducibility" / "incremental_linker_training_target.json"
        _copy_path(target_json, target_destination)
        staged_summary = write_production_manifest(
            staging_dir,
            bundle_version=pairwise_bundle_version,
            pairwise_model_version=str(pairwise_manifest["pairwise_model_version"]),
            incremental_linker_version=pairwise_bundle_version,
        )
        load_production_model(
            staging_dir,
            expected_artifact_hashes=expected_artifact_hashes,
        )
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
