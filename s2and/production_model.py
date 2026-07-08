"""Production model bundle loading for packaged S2AND prediction artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import load_incremental_linking_artifact
from s2and.incremental_linking.contracts import validate_artifact_contract_metadata
from s2and.model import Clusterer, FastCluster, IncrementalBroadcastMode, IncrementalSeedScoreMode
from s2and.serialization import load_pickle_with_verified_label_encoder_compat
from s2and.thread_config import resolve_n_jobs

PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION = "s2and_production_model_bundle_v1"
PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION = "pairwise_prediction_fixture_v1"
DEFAULT_PRODUCTION_MODEL_DIR = Path(_PACKAGE_DATA_DIR) / "production_model_v1.21"
PUBLISHED_PRODUCTION_MODEL_RUNTIME_CLUSTER_EPS = 0.65
_RUNTIME_CLUSTER_EPS_OVERRIDE_VERSIONS = frozenset({"1.2", "1.21"})
_PRODUCTION_MODEL_PATH_PREFIX = "production_model_v"
_INCREMENTAL_BROADCAST_MODES = frozenset({"always", "never", "top1_consensus"})
_INCREMENTAL_SEED_SCORE_MODES = frozenset({"mean", "min", "mean_min_hybrid"})


def _load_rust_lightgbm_booster(model_path: str) -> Any:
    try:
        rust_module = importlib.import_module("s2and_rust")
    except ImportError as exc:
        from s2and.runtime import min_supported_rust_extension_version_string

        minimum = min_supported_rust_extension_version_string()
        raise RuntimeError(
            "RustLightGBMBooster requires s2and-rust>="
            f"{minimum}; the Rust extension is not importable. Rebuild the local extension or install the "
            "matching s2and-rust package."
        ) from exc
    booster_cls = getattr(rust_module, "RustLightGBMBooster", None)
    if booster_cls is None:
        from s2and.runtime import min_supported_rust_extension_version_string

        minimum = min_supported_rust_extension_version_string()
        found = getattr(rust_module, "__version__", None)
        found_version = "unknown" if found is None else str(found)
        raise RuntimeError(
            "RustLightGBMBooster requires s2and-rust>="
            f"{minimum}; found {found_version}. Rebuild the local extension or install the matching "
            "s2and-rust package."
        )

    return booster_cls(model_path)


class NativeLightGBMBinaryClassifier:
    """Small sklearn-compatible wrapper around a native LightGBM binary model.

    Scoring runs through the pure-Rust evaluator (``RustLightGBMBooster``),
    whose raw scores are parity-gated bit-exact against Python lightgbm
    (tests/test_rust_lightgbm_booster_parity.py). ``booster_`` lazily loads a
    Python ``lgb.Booster`` only for training-side consumers such as bundle
    writing and SHAP diagnostics; runtime scoring never touches it.
    """

    prediction_backend = "rust_lightgbm"

    def __init__(self, model_path: str | Path, *, n_jobs: int = 1, n_features: int | None = None) -> None:
        self.model_path = str(Path(model_path))
        self.n_jobs = resolve_n_jobs(n_jobs)
        self._scorer = _load_rust_lightgbm_booster(self.model_path)
        self._lazy_booster: lgb.Booster | None = None
        self._model_sha256: str | None = None
        self._set_feature_count(n_features)
        self._classes = np.asarray([0.0, 1.0])
        self.classes_ = self._classes.copy()
        self.fitted_ = True

    def _set_feature_count(self, n_features: int | None) -> None:
        self._n_features = int(n_features if n_features is not None else self._scorer.num_features())
        self._n_features_in = self._n_features
        self.n_features_in_ = self._n_features

    @property
    def booster_(self) -> lgb.Booster:
        if self._lazy_booster is None:
            self._lazy_booster = lgb.Booster(model_file=self.model_path)
        return self._lazy_booster

    def cache_fingerprint(self) -> tuple[str, str, int]:
        """Stable prediction-cache fingerprint that avoids materializing the Python booster."""
        if self._model_sha256 is None:
            self._model_sha256 = _sha256_file(Path(self.model_path))
        return ("native_lightgbm", self._model_sha256, self._n_features)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "model_path": self.model_path,
            "n_jobs": self.n_jobs,
            "n_features": self._n_features,
        }

    def set_params(self, **params: Any) -> NativeLightGBMBinaryClassifier:
        valid_params = {"model_path", "n_jobs", "n_features"}
        invalid = sorted(set(params) - valid_params)
        if invalid:
            raise ValueError(f"Invalid parameter(s) for NativeLightGBMBinaryClassifier: {invalid}")
        if "model_path" in params:
            model_path = str(Path(params["model_path"]))
            if model_path != self.model_path:
                self.model_path = model_path
                self._scorer = _load_rust_lightgbm_booster(self.model_path)
                self._lazy_booster = None
                self._model_sha256 = None
        if "n_jobs" in params:
            self.n_jobs = resolve_n_jobs(params["n_jobs"])
        if "model_path" in params or "n_features" in params:
            self._set_feature_count(cast(int | None, params.get("n_features")))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features_2d = np.asarray(features, dtype=np.float64, order="C")
        if features_2d.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={features_2d.shape}")
        if features_2d.shape[1] != self._n_features:
            raise ValueError(f"features must have {self._n_features} columns, got {features_2d.shape[1]}")
        positive = np.asarray(
            self._scorer.predict_proba_positive(features_2d, num_threads=self.n_jobs), dtype=np.float64
        ).reshape(-1)
        return np.column_stack((1.0 - positive, positive))

    def __deepcopy__(self, memo: dict[int, Any]) -> NativeLightGBMBinaryClassifier:
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.model_path = self.model_path
        copied.n_jobs = self.n_jobs
        copied._scorer = copy.deepcopy(self._scorer, memo)
        copied._lazy_booster = None
        copied._model_sha256 = self._model_sha256
        copied._set_feature_count(self._n_features)
        copied._classes = self._classes.copy()
        copied.classes_ = self.classes_.copy()
        copied.fitted_ = self.fitted_
        return copied

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # The Rust scorer and lazily-loaded Python booster are rebuilt from
        # model_path on unpickle.
        state["_scorer"] = None
        state["_lazy_booster"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._scorer = _load_rust_lightgbm_booster(self.model_path)
        self._lazy_booster = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _validate_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported production model bundle schema_version={manifest.get('schema_version')!r}")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Production model bundle manifest must contain a files object")
    expected_hashes = manifest.get("sha256", {})
    if not isinstance(expected_hashes, dict):
        raise ValueError("Production model bundle manifest sha256 must be an object")
    for relpath, expected in expected_hashes.items():
        path = bundle_dir / str(relpath)
        if not path.exists():
            raise FileNotFoundError(f"Production model bundle is missing {relpath}: {path}")
        observed = _sha256_file(path)
        if observed != str(expected):
            raise ValueError(f"Production model bundle checksum mismatch for {relpath}")
    return manifest


def _featurization_info_from_payload(payload: dict[str, Any]) -> FeaturizationInfo:
    return FeaturizationInfo(
        features_to_use=[str(value) for value in payload["features_to_use"]],
        featurizer_version=int(payload["featurizer_version"]),
    )


def _config_choice(payload: dict[str, Any], key: str, *, allowed: frozenset[str]) -> str:
    value = str(payload[key])
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported production model config {key}={value!r}; expected one of {allowed_values}")
    return value


def _validate_pairwise_fixture(classifier: NativeLightGBMBinaryClassifier, fixture_path: Path) -> None:
    fixture = _read_json(fixture_path)
    if fixture.get("schema_version") != PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported pairwise prediction fixture schema_version={fixture.get('schema_version')!r}")
    matrix = np.asarray(fixture["features"], dtype=np.float64)
    expected = np.asarray(fixture["expected_probabilities"], dtype=np.float64)
    observed = classifier.predict_proba(matrix)
    rtol = float(fixture.get("rtol", 1e-10))
    atol = float(fixture.get("atol", 1e-10))
    if observed.shape != expected.shape:
        raise ValueError(f"Pairwise prediction fixture shape mismatch: {observed.shape} != {expected.shape}")
    if not np.allclose(observed, expected, rtol=rtol, atol=atol):
        raise ValueError(f"Pairwise prediction fixture mismatch for {fixture_path}")


def _validate_incremental_linker_metadata(linker_dir: Path) -> None:
    metadata_path = linker_dir / "metadata.json"
    booster_path = linker_dir / "booster.lgb"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Incremental linker metadata is missing: {metadata_path}")
    if not booster_path.exists():
        raise FileNotFoundError(f"Incremental linker booster is missing: {booster_path}")
    metadata = _read_json(metadata_path)
    validate_artifact_contract_metadata(metadata)
    load_incremental_linking_artifact(linker_dir, require_rust_capabilities=False)


def _production_model_path_version(path: Path) -> str | None:
    name = path.name.removesuffix(".pickle")
    if not name.startswith(_PRODUCTION_MODEL_PATH_PREFIX):
        return None
    version = name.removeprefix(_PRODUCTION_MODEL_PATH_PREFIX)
    return version or None


def _production_runtime_cluster_eps(
    model_path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    clusterer_config: Mapping[str, Any] | None = None,
) -> float | None:
    versions: set[str] = set()
    path_version = _production_model_path_version(model_path)
    if path_version is not None:
        versions.add(path_version)
    for payload, keys in (
        (manifest, ("bundle_version", "pairwise_model_version")),
        (clusterer_config, ("bundle_version", "source_model_version")),
    ):
        if payload is None:
            continue
        for key in keys:
            value = payload.get(key)
            if value is not None:
                versions.add(str(value))
    if versions.isdisjoint(_RUNTIME_CLUSTER_EPS_OVERRIDE_VERSIONS):
        return None
    return PUBLISHED_PRODUCTION_MODEL_RUNTIME_CLUSTER_EPS


def _apply_production_runtime_cluster_eps(clusterer: Clusterer, eps: float | None) -> Clusterer:
    if eps is None:
        return clusterer
    if not isinstance(clusterer.cluster_model, FastCluster):
        raise TypeError(
            "Published production runtime cluster eps override requires "
            f"FastCluster, got {type(clusterer.cluster_model)!r}"
        )
    best_params = getattr(clusterer, "best_params", None)
    clusterer.best_params = dict(best_params or {})
    clusterer.best_params["eps"] = float(eps)
    clusterer.set_params({"eps": float(eps)})
    return clusterer


def _load_bundle_clusterer(bundle_dir: Path, *, require_incremental_linker: bool = True) -> Clusterer:
    manifest = _validate_manifest(bundle_dir)
    clusterer_config = _read_json(bundle_dir / str(manifest["files"]["clusterer_config"]))
    runtime_cluster_eps = _production_runtime_cluster_eps(
        bundle_dir,
        manifest=manifest,
        clusterer_config=clusterer_config,
    )

    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_featurizer_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_main_model"]),
        n_features=int(clusterer_config["pairwise"]["main_feature_count"]),
    )
    nameless_classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_nameless_model"]),
        n_features=int(clusterer_config["pairwise"]["nameless_feature_count"]),
    )
    _validate_pairwise_fixture(classifier, bundle_dir / str(manifest["files"]["pairwise_main_fixture"]))
    _validate_pairwise_fixture(nameless_classifier, bundle_dir / str(manifest["files"]["pairwise_nameless_fixture"]))

    cluster_model_config = clusterer_config["cluster_model"]
    cluster_model = FastCluster(
        linkage=str(cluster_model_config["linkage"]),
        eps=float(cluster_model_config["eps"]),
    )
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=classifier,
        val_blocks_size=clusterer_config.get("val_blocks_size"),
        cluster_model=cluster_model,
        search_space=None,
        n_iter=int(clusterer_config["n_iter"]),
        n_jobs=int(clusterer_config["n_jobs"]),
        use_cache=bool(clusterer_config["use_cache"]),
        use_default_constraints_as_supervision=bool(clusterer_config["use_default_constraints_as_supervision"]),
        random_state=int(clusterer_config["random_state"]),
        nameless_classifier=nameless_classifier,
        nameless_featurizer_info=nameless_featurizer_info,
        dont_merge_cluster_seeds=bool(clusterer_config["dont_merge_cluster_seeds"]),
        batch_size=int(clusterer_config["batch_size"]),
        suppress_orcid=bool(clusterer_config["suppress_orcid"]),
    )
    clusterer.feature_contract = dict(clusterer_config["feature_contract"])
    clusterer.best_params = dict(clusterer_config["best_params"])
    clusterer.incremental_precluster_broadcast_mode = cast(
        IncrementalBroadcastMode,
        _config_choice(
            clusterer_config,
            "incremental_precluster_broadcast_mode",
            allowed=_INCREMENTAL_BROADCAST_MODES,
        ),
    )
    clusterer.incremental_seed_score_mode = cast(
        IncrementalSeedScoreMode,
        _config_choice(
            clusterer_config,
            "incremental_seed_score_mode",
            allowed=_INCREMENTAL_SEED_SCORE_MODES,
        ),
    )
    clusterer.incremental_mean_min_hybrid_weight = float(clusterer_config["incremental_mean_min_hybrid_weight"])
    incremental_linker_relpath = manifest["files"].get("incremental_linker_dir")
    if incremental_linker_relpath is None:
        if require_incremental_linker:
            raise FileNotFoundError(
                f"Production model bundle is pairwise-only and has no incremental_linker: {bundle_dir}"
            )
    else:
        incremental_linker_dir = bundle_dir / str(incremental_linker_relpath)
        _validate_incremental_linker_metadata(incremental_linker_dir)
        clusterer.incremental_linker_artifact_dir = incremental_linker_dir
    clusterer.production_model_bundle_dir = bundle_dir
    clusterer.production_model_bundle_version = str(manifest["bundle_version"])
    clusterer.production_model_bundle_status = str(manifest.get("bundle_status", "complete"))
    return _apply_production_runtime_cluster_eps(clusterer, runtime_cluster_eps)


def load_production_model(path: str | Path | None = None, *, require_incremental_linker: bool = True) -> Clusterer:
    """Load the production model from a native bundle directory.

    Legacy pickle paths are accepted so older local scripts can migrate by
    changing only the imported loader first. New production defaults should pass
    the v1.21 bundle directory.
    Set ``require_incremental_linker=False`` only for training/finalization
    code that intentionally consumes a pairwise-only bundle stage.
    """

    model_path = Path(path) if path is not None else DEFAULT_PRODUCTION_MODEL_DIR
    if model_path.is_dir():
        return _load_bundle_clusterer(model_path, require_incremental_linker=require_incremental_linker)
    loaded = load_pickle_with_verified_label_encoder_compat(str(model_path))
    clusterer = loaded.get("clusterer") if isinstance(loaded, dict) else loaded
    if not isinstance(clusterer, Clusterer):
        raise TypeError(f"Expected a Clusterer in production model artifact, got {type(clusterer)!r}")
    runtime_cluster_eps = _production_runtime_cluster_eps(model_path)
    return _apply_production_runtime_cluster_eps(clusterer, runtime_cluster_eps)
