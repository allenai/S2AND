"""Build native production model bundles from trained S2AND artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from s2and.consts import NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, _selected_feature_indices
from s2and.production_model import (
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    load_production_model,
)

PAIRWISE_METADATA_SCHEMA_VERSION = "s2and_pairwise_native_lightgbm_v1"
CLUSTERER_CONFIG_SCHEMA_VERSION = "s2and_clusterer_config_v1"
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
        probabilities = np.asarray(predict_proba(matrix), dtype=np.float64)
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
    bundle_version: str,
    source_model_version: str,
    nameless_featurizer_info: FeaturizationInfo,
    main_feature_count: int,
    nameless_feature_count: int,
) -> dict[str, Any]:
    feature_contract = dict(
        getattr(clusterer, "feature_contract", {"name_counts_last_first_initial_semantics": "initial_char"})
    )
    if not feature_contract:
        feature_contract = {"name_counts_last_first_initial_semantics": "initial_char"}
    # A bundle written by this package normalizes names with this package's policy;
    # readers assert the recorded version against their own (fail-fast contract).
    feature_contract.setdefault("normalization_version", NORMALIZATION_VERSION)
    return {
        "batch_size": int(getattr(clusterer, "batch_size", 1_000_000)),
        "best_params": dict(
            getattr(
                clusterer,
                "best_params",
                {
                    "eps": float(clusterer.cluster_model.eps),
                    "linkage": str(clusterer.cluster_model.linkage),
                },
            )
        ),
        "bundle_version": str(bundle_version),
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
        "pairwise": {
            "main_feature_count": int(main_feature_count),
            "nameless_feature_count": int(nameless_feature_count),
        },
        "random_state": int(getattr(clusterer, "random_state", 42)),
        "schema_version": CLUSTERER_CONFIG_SCHEMA_VERSION,
        "source_model_version": str(source_model_version),
        "suppress_orcid": bool(getattr(clusterer, "suppress_orcid", False)),
        "use_cache": bool(getattr(clusterer, "use_cache", False)),
        "use_default_constraints_as_supervision": bool(
            getattr(clusterer, "use_default_constraints_as_supervision", True)
        ),
        "val_blocks_size": getattr(clusterer, "val_blocks_size", None),
    }


def _pairwise_metadata_payload(
    clusterer: Clusterer,
    *,
    source_model_version: str,
    nameless_featurizer_info: FeaturizationInfo,
    main_feature_count: int,
    nameless_feature_count: int,
) -> dict[str, Any]:
    return {
        "class_labels": [0.0, 1.0],
        "distance_probability_column": "class_0",
        "main": {
            **_featurization_info_payload(clusterer.featurizer_info),
            "model_file": "main.lgb",
            "selected_feature_count": int(main_feature_count),
            "selected_feature_indices": list(_selected_feature_indices(clusterer.featurizer_info)),
        },
        "model_family": "binary_lightgbm_pairwise_distance",
        "nameless": {
            **_featurization_info_payload(nameless_featurizer_info),
            "model_file": "nameless.lgb",
            "selected_feature_count": int(nameless_feature_count),
            "selected_feature_indices": list(_selected_feature_indices(nameless_featurizer_info)),
        },
        "positive_probability_column": "class_1",
        "schema_version": PAIRWISE_METADATA_SCHEMA_VERSION,
        "source_model_version": str(source_model_version),
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


def _manifest_files(*, include_incremental_linker: bool) -> dict[str, str]:
    files = {
        "clusterer_config": "clusterer.json",
        "pairwise_main_fixture": "pairwise/main_prediction_fixture.json",
        "pairwise_main_model": "pairwise/main.lgb",
        "pairwise_metadata": "pairwise/metadata.json",
        "pairwise_nameless_fixture": "pairwise/nameless_prediction_fixture.json",
        "pairwise_nameless_model": "pairwise/nameless.lgb",
    }
    if include_incremental_linker:
        files.update(
            {
                "incremental_linker_booster": "incremental_linker/booster.lgb",
                "incremental_linker_dir": "incremental_linker",
                "incremental_linker_metadata": "incremental_linker/metadata.json",
                "incremental_linker_training_target": "reproducibility/incremental_linker_training_target.json",
            }
        )
    return files


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
    files = _manifest_files(include_incremental_linker=include_incremental_linker)
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
        "default_runtime_contract": (
            "load this directory once; normal prediction and Rust incremental prediction use artifacts from this bundle"
            if include_incremental_linker
            else "pairwise bundle stage; run linker training/finalization before using as the production runtime"
        ),
        "description": (
            f"Native production bundle: v{pairwise_model_version} pairwise model"
            + (
                f" plus v{incremental_linker_version} promoted incremental linker."
                if include_incremental_linker
                else "."
            )
        ),
        "files": files,
        "format": "native_lightgbm_json",
        "incremental_linker_version": str(incremental_linker_version) if include_incremental_linker else None,
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


def write_pairwise_production_bundle(
    clusterer: Clusterer,
    bundle_dir: Path,
    *,
    bundle_version: str,
    source_model_version: str | None = None,
    pairwise_training_config: Mapping[str, Any] | None = None,
    pairwise_training_summary: Mapping[str, Any] | None = None,
) -> ProductionBundleSummary:
    """Write the pairwise stage of a native production model bundle."""

    nameless_featurizer_info = clusterer.nameless_featurizer_info
    if clusterer.nameless_classifier is None or nameless_featurizer_info is None:
        raise ValueError("Production bundles require a nameless pairwise model")

    bundle_dir = Path(bundle_dir)
    stale_incremental_paths = [
        bundle_dir / "incremental_linker",
        bundle_dir / "reproducibility" / "incremental_linker_training_target.json",
    ]
    existing_stale_paths = [path for path in stale_incremental_paths if path.exists()]
    if existing_stale_paths:
        joined = ", ".join(str(path) for path in existing_stale_paths)
        raise ValueError(
            "Refusing to write a pairwise-only production bundle over existing incremental linker artifacts: "
            f"{joined}"
        )
    pairwise_dir = bundle_dir / "pairwise"
    source_version = str(source_model_version or bundle_version)
    main_width = len(_selected_feature_indices(clusterer.featurizer_info))
    nameless_width = len(_selected_feature_indices(nameless_featurizer_info))

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
        _pairwise_metadata_payload(
            clusterer,
            source_model_version=source_version,
            nameless_featurizer_info=nameless_featurizer_info,
            main_feature_count=main_width,
            nameless_feature_count=nameless_width,
        ),
    )
    _write_json(
        bundle_dir / "clusterer.json",
        _clusterer_config_payload(
            clusterer,
            bundle_version=str(bundle_version),
            source_model_version=source_version,
            nameless_featurizer_info=nameless_featurizer_info,
            main_feature_count=main_width,
            nameless_feature_count=nameless_width,
        ),
    )

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


def _rewrite_linker_target_spec(metadata_path: Path, *, target_spec: str) -> None:
    metadata = _read_json(metadata_path)
    audit_metadata = dict(metadata.get("audit_metadata", {}))
    audit_metadata["target_spec"] = str(target_spec)
    metadata["audit_metadata"] = audit_metadata
    _write_json(metadata_path, metadata)


def finalize_production_bundle(
    *,
    pairwise_bundle_dir: Path,
    output_bundle_dir: Path,
    incremental_linker_artifact_dir: Path,
    target_json: Path,
    bundle_version: str | None = None,
    pairwise_model_version: str | None = None,
    incremental_linker_version: str | None = None,
    validate: bool = True,
) -> ProductionBundleSummary:
    """Assemble a complete production bundle from pairwise and linker artifacts."""

    pairwise_bundle_dir = Path(pairwise_bundle_dir)
    output_bundle_dir = Path(output_bundle_dir)
    incremental_linker_artifact_dir = Path(incremental_linker_artifact_dir)
    target_json = Path(target_json)
    if not pairwise_bundle_dir.is_dir():
        raise FileNotFoundError(f"Pairwise bundle directory does not exist: {pairwise_bundle_dir}")
    if not incremental_linker_artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Incremental linker artifact directory does not exist: {incremental_linker_artifact_dir}"
        )
    if not target_json.exists():
        raise FileNotFoundError(f"Incremental linker target JSON does not exist: {target_json}")

    inferred_version = production_version_from_bundle_dir(output_bundle_dir)
    resolved_bundle_version = str(
        bundle_version or inferred_version or production_version_from_bundle_dir(pairwise_bundle_dir) or ""
    )
    if not resolved_bundle_version:
        raise ValueError("bundle_version is required when output_bundle_dir is not named production_model_vX.Y")

    _copy_pairwise_stage(pairwise_bundle_dir, output_bundle_dir)
    _copy_path(incremental_linker_artifact_dir, output_bundle_dir / "incremental_linker")
    target_destination = output_bundle_dir / "reproducibility" / "incremental_linker_training_target.json"
    _copy_path(target_json, target_destination)
    _rewrite_linker_target_spec(
        output_bundle_dir / "incremental_linker" / "metadata.json",
        target_spec=f"s2and/data/production_model_v{resolved_bundle_version}/reproducibility/"
        "incremental_linker_training_target.json",
    )

    summary = write_production_manifest(
        output_bundle_dir,
        bundle_version=resolved_bundle_version,
        pairwise_model_version=str(
            pairwise_model_version or production_version_from_bundle_dir(pairwise_bundle_dir) or resolved_bundle_version
        ),
        include_incremental_linker=True,
        incremental_linker_version=str(incremental_linker_version or resolved_bundle_version),
    )
    if validate:
        load_production_model(output_bundle_dir)
    return summary
