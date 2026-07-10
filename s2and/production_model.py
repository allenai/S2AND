"""Production model bundle loading for packaged S2AND prediction artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from s2and.consts import (
    _PACKAGE_DATA_DIR,
    FEATURIZER_VERSION,
    NORMALIZATION_VERSION,
    NORMALIZATION_VERSION_LEGACY_COMPAT,
    VALID_NORMALIZATION_VERSIONS,
)
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import IncrementalLinkingArtifact, load_incremental_linking_artifact
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.model import (
    Clusterer,
    FastCluster,
    IncrementalBroadcastMode,
    IncrementalSeedScoreMode,
    _selected_feature_indices,
)
from s2and.name_count_binding import NameCountsBinding
from s2and.production_bundle_contract import (
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    production_manifest_files,
)
from s2and.serialization import load_pickle_with_verified_label_encoder_compat
from s2and.thread_config import resolve_n_jobs

DEFAULT_PRODUCTION_MODEL_DECLARATION_SCHEMA_VERSION = "s2and_default_production_model_v1"
DEFAULT_PRODUCTION_MODEL_DECLARATION_PATH = Path(_PACKAGE_DATA_DIR) / "default_production_model.json"
PUBLISHED_PRODUCTION_MODEL_RUNTIME_CLUSTER_EPS = 0.65
_RUNTIME_CLUSTER_EPS_OVERRIDE_VERSIONS = frozenset({"1.2", "1.21"})
_PRODUCTION_MODEL_PATH_PREFIX = "production_model_v"
_INCREMENTAL_BROADCAST_MODES = frozenset({"always", "never", "top1_consensus"})
_INCREMENTAL_SEED_SCORE_MODES = frozenset({"mean", "min", "mean_min_hybrid"})
_CLUSTERER_CONFIG_FIELDS = frozenset(
    {
        "batch_size",
        "best_params",
        "bundle_version",
        "cluster_model",
        "dont_merge_cluster_seeds",
        "feature_contract",
        "featurizer_info",
        "incremental_mean_min_hybrid_weight",
        "incremental_precluster_broadcast_mode",
        "incremental_seed_score_mode",
        "n_iter",
        "n_jobs",
        "nameless_featurizer_info",
        "pairwise",
        "random_state",
        "schema_version",
        "source_model_version",
        "suppress_orcid",
        "use_cache",
        "use_default_constraints_as_supervision",
        "val_blocks_size",
    }
)


def _load_default_production_model_dir() -> Path:
    declaration = _read_json(DEFAULT_PRODUCTION_MODEL_DECLARATION_PATH)
    if declaration.get("schema_version") != DEFAULT_PRODUCTION_MODEL_DECLARATION_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported default production model declaration schema_version=" f"{declaration.get('schema_version')!r}"
        )
    directory = declaration.get("bundle_dir")
    if (
        not isinstance(directory, str)
        or not directory.startswith("production_model_v")
        or "/" in directory
        or "\\" in directory
    ):
        raise ValueError(f"Invalid default production model bundle_dir={directory!r}")
    declared_version = str(declaration.get("bundle_version", ""))
    if (
        not declared_version
        or ".." in declared_version
        or any(not (character.isalnum() or character in ".-_") for character in declared_version)
    ):
        raise ValueError(f"Invalid default production model bundle_version={declared_version!r}")
    if directory != f"production_model_v{declared_version}":
        raise ValueError("Default production model declaration bundle_dir and bundle_version disagree")
    return Path(_PACKAGE_DATA_DIR) / directory


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


def _lightgbm_text_feature_count(model_path: Path) -> int:
    with model_path.open("r", encoding="utf-8") as handle:
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            if line.startswith("max_feature_idx="):
                value = int(line.partition("=")[2].strip()) + 1
                if value <= 0:
                    raise ValueError(f"LightGBM model has invalid max_feature_idx in {model_path}")
                return value
    raise ValueError(f"LightGBM model is missing max_feature_idx header: {model_path}")


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
        self._scorer: Any | None = None
        self._lazy_booster: lgb.Booster | None = None
        self._model_sha256: str | None = None
        self._set_feature_count(n_features)
        self._classes = np.asarray([0.0, 1.0])
        self.classes_ = self._classes.copy()
        self.fitted_ = True

    def _set_feature_count(self, n_features: int | None) -> None:
        actual_features = _lightgbm_text_feature_count(Path(self.model_path))
        if n_features is not None and int(n_features) != actual_features:
            raise ValueError(
                "Native LightGBM feature-count mismatch: "
                f"metadata declares {int(n_features)} but booster contains {actual_features}"
            )
        self._n_features = actual_features
        self._n_features_in = self._n_features
        self.n_features_in_ = self._n_features

    @property
    def _rust_scorer(self) -> Any:
        if self._scorer is None:
            self._scorer = _load_rust_lightgbm_booster(self.model_path)
        return self._scorer

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
                self._scorer = None
                self._lazy_booster = None
                self._model_sha256 = None
        if "n_jobs" in params:
            self.n_jobs = resolve_n_jobs(params["n_jobs"])
        if "model_path" in params or "n_features" in params:
            self._set_feature_count(cast(int | None, params.get("n_features")))
        return self

    def predict_proba_positive(
        self,
        features: np.ndarray,
        *,
        max_rows_per_chunk: int | None = None,
    ) -> np.ndarray:
        raw_features = np.asarray(features)
        if raw_features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={raw_features.shape}")
        if raw_features.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError(f"features must be float32 or float64, got dtype={raw_features.dtype}")
        if not raw_features.flags.c_contiguous:
            raise ValueError("features must be C-contiguous")
        use_float32 = raw_features.dtype == np.float32
        features_2d = raw_features
        if features_2d.shape[1] != self._n_features:
            raise ValueError(f"features must have {self._n_features} columns, got {features_2d.shape[1]}")
        scorer = self._rust_scorer
        predict = scorer.predict_proba_positive_f32 if use_float32 else scorer.predict_proba_positive
        row_count = int(features_2d.shape[0])
        if max_rows_per_chunk is None:
            chunk_rows = max(1, row_count)
        else:
            chunk_rows = int(max_rows_per_chunk)
            if chunk_rows <= 0:
                raise ValueError(f"max_rows_per_chunk must be positive, got {max_rows_per_chunk}")
        if row_count == 0 or chunk_rows >= row_count:
            return np.asarray(predict(features_2d, num_threads=self.n_jobs), dtype=np.float64).reshape(-1)
        positive = np.empty(row_count, dtype=np.float64)
        for start in range(0, row_count, chunk_rows):
            stop = min(row_count, start + chunk_rows)
            positive[start:stop] = np.asarray(
                predict(features_2d[start:stop], num_threads=self.n_jobs),
                dtype=np.float64,
            ).reshape(-1)
        return positive

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        positive = self.predict_proba_positive(features)
        return np.column_stack((1.0 - positive, positive))

    def __deepcopy__(self, memo: dict[int, Any]) -> NativeLightGBMBinaryClassifier:
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        copied.model_path = self.model_path
        copied.n_jobs = self.n_jobs
        # RustLightGBMBooster is immutable after construction and prediction is
        # thread-safe. Share its model vectors while keeping wrapper params,
        # classes, and lazy Python state independent.
        copied._scorer = self._scorer
        copied._lazy_booster = None
        copied._model_sha256 = self._model_sha256
        copied._n_features = self._n_features
        copied._n_features_in = self._n_features_in
        copied.n_features_in_ = self.n_features_in_
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
        self._scorer = None
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


DEFAULT_PRODUCTION_MODEL_DIR = _load_default_production_model_dir()


def _validated_bundle_path(bundle_dir: Path, raw_relpath: Any, *, field: str) -> tuple[str, Path]:
    if not isinstance(raw_relpath, str) or not raw_relpath:
        raise ValueError(f"Production model bundle {field} must be a nonempty relative POSIX path")
    if "\\" in raw_relpath or ":" in raw_relpath:
        raise ValueError(f"Production model bundle {field} must use POSIX separators: {raw_relpath!r}")
    relpath = Path(raw_relpath)
    if relpath.is_absolute() or relpath.drive or any(part in {"", ".", ".."} for part in relpath.parts):
        raise ValueError(f"Production model bundle {field} escapes the bundle root: {raw_relpath!r}")
    bundle_root = bundle_dir.resolve()
    path = (bundle_dir / relpath).resolve()
    try:
        path.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError(f"Production model bundle {field} escapes the bundle root: {raw_relpath!r}") from exc
    normalized = relpath.as_posix()
    return normalized, path


def _runtime_files_on_disk(bundle_dir: Path, *, complete: bool) -> set[str]:
    candidates = [bundle_dir / "clusterer.json", bundle_dir / "pairwise", bundle_dir / "reproducibility"]
    if complete:
        candidates.append(bundle_dir / "incremental_linker")
    files: set[str] = set()
    root = bundle_dir.resolve()
    for candidate in candidates:
        if candidate.is_dir():
            descendants = (path for path in candidate.rglob("*") if path.is_file())
        elif candidate.is_file():
            descendants = iter((candidate,))
        else:
            continue
        for path in descendants:
            resolved = path.resolve()
            try:
                files.add(resolved.relative_to(root).as_posix())
            except ValueError as exc:
                raise ValueError(f"Production model runtime file escapes bundle root: {path}") from exc
    if not complete:
        # Same-path finalization may have installed this complete-only file
        # while the pairwise-only manifest remains the commit authority.
        files.discard("reproducibility/incremental_linker_training_target.json")
    return files


def _validate_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported production model bundle schema_version={manifest.get('schema_version')!r}")
    status = manifest.get("bundle_status")
    if status not in {"pairwise_only", "complete"}:
        raise ValueError(f"Unsupported production model bundle_status={status!r}")
    complete = status == "complete"
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Production model bundle manifest must contain a files object")
    optional_keys = set(PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES)
    present_optional_keys = optional_keys & set(files)
    if present_optional_keys and present_optional_keys != optional_keys:
        raise ValueError("Production model bundle must declare both pairwise reproducibility files or neither")
    expected_files = production_manifest_files(
        complete=complete,
        include_pairwise_reproducibility=present_optional_keys == optional_keys,
    )
    normalized_paths: dict[str, str] = {}
    for key, relpath in files.items():
        normalized, _ = _validated_bundle_path(bundle_dir, relpath, field=f"files[{key!r}]")
        previous = normalized_paths.setdefault(normalized, str(key))
        if previous != str(key):
            raise ValueError(
                f"Production model bundle files contain duplicate normalized path {normalized!r}: "
                f"{previous!r} and {key!r}"
            )
    if files != expected_files:
        missing = sorted(set(expected_files) - set(files))
        extra = sorted(set(files) - set(expected_files))
        changed = sorted(key for key in set(files) & set(expected_files) if files[key] != expected_files[key])
        raise ValueError(
            "Production model bundle files contract mismatch: " f"missing={missing} extra={extra} changed={changed}"
        )

    expected_hashes = manifest.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Production model bundle manifest sha256 must be an object")
    required_hashed_files = {str(value) for key, value in expected_files.items() if key != "incremental_linker_dir"}
    if set(expected_hashes) != required_hashed_files:
        raise ValueError(
            "Production model bundle checksum coverage mismatch: "
            f"missing={sorted(required_hashed_files - set(expected_hashes))} "
            f"extra={sorted(set(expected_hashes) - required_hashed_files)}"
        )
    for relpath, expected in expected_hashes.items():
        normalized, path = _validated_bundle_path(bundle_dir, relpath, field="sha256 key")
        if normalized != relpath:
            raise ValueError(f"Production model bundle checksum path is not normalized: {relpath!r}")
        if not path.is_file():
            raise FileNotFoundError(f"Production model bundle is missing {relpath}: {path}")
        if not isinstance(expected, str) or len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(f"Production model bundle has invalid SHA-256 for {relpath}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Production model bundle checksum mismatch for {relpath}")
    runtime_files = _runtime_files_on_disk(bundle_dir, complete=complete)
    if runtime_files != required_hashed_files:
        raise ValueError(
            "Production model bundle contains undeclared or missing runtime files: "
            f"missing={sorted(required_hashed_files - runtime_files)} "
            f"extra={sorted(runtime_files - required_hashed_files)}"
        )
    if complete != bool(manifest.get("incremental_linker_version")):
        raise ValueError("Production model bundle status and incremental_linker_version disagree")
    return manifest


def _featurization_info_from_payload(payload: dict[str, Any]) -> FeaturizationInfo:
    if not isinstance(payload, dict) or set(payload) != {"features_to_use", "featurizer_version"}:
        raise ValueError("Production featurization info must contain exactly features_to_use and featurizer_version")
    if not isinstance(payload["features_to_use"], list) or not all(
        isinstance(value, str) for value in payload["features_to_use"]
    ):
        raise ValueError("Production featurization info features_to_use must be a list of strings")
    if not isinstance(payload["featurizer_version"], int) or isinstance(payload["featurizer_version"], bool):
        raise ValueError("Production featurization info featurizer_version must be an integer")
    return FeaturizationInfo(
        features_to_use=[str(value) for value in payload["features_to_use"]],
        featurizer_version=int(payload["featurizer_version"]),
    )


def _require_featurizer_version_match(model_path: Path, versions: Mapping[str, int]) -> None:
    mismatched = {name: version for name, version in versions.items() if int(version) != FEATURIZER_VERSION}
    if not mismatched:
        return
    version_text = ", ".join(f"{name}={version}" for name, version in sorted(mismatched.items()))
    raise ValueError(
        f"Production model artifact {model_path} was trained with {version_text}, "
        f"but this package requires FEATURIZER_VERSION={FEATURIZER_VERSION}."
    )


def _require_finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Production model config {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Production model config {field} must be finite, got {value!r}")
    return number


def _validate_clusterer_config(manifest: Mapping[str, Any], payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "s2and_clusterer_config_v1":
        raise ValueError(f"Unsupported clusterer config schema_version={payload.get('schema_version')!r}")
    if set(payload) != _CLUSTERER_CONFIG_FIELDS:
        raise ValueError(
            "Production clusterer config field mismatch: "
            f"missing={sorted(_CLUSTERER_CONFIG_FIELDS - set(payload))} "
            f"extra={sorted(set(payload) - _CLUSTERER_CONFIG_FIELDS)}"
        )
    if str(payload["bundle_version"]) != str(manifest["bundle_version"]):
        raise ValueError("Production clusterer and manifest bundle_version disagree")
    if str(payload["source_model_version"]) != str(manifest["pairwise_model_version"]):
        raise ValueError("Production clusterer and manifest pairwise model version disagree")

    cluster_model = payload.get("cluster_model")
    if not isinstance(cluster_model, dict) or set(cluster_model) != {
        "eps",
        "family",
        "input_as_observation_matrix",
        "linkage",
        "preserve_input",
    }:
        raise ValueError("Production cluster_model must contain the exact FastCluster runtime configuration")
    if cluster_model["family"] != "FastCluster":
        raise ValueError(f"Unsupported production cluster_model family={cluster_model['family']!r}")
    if cluster_model["input_as_observation_matrix"] is not False or cluster_model["preserve_input"] is not True:
        raise ValueError("Production FastCluster requires condensed distances with preserve_input=true")
    eps = _require_finite_number(cluster_model["eps"], field="cluster_model.eps")
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"Production cluster_model.eps must be in [0, 1], got {eps!r}")
    if cluster_model["linkage"] not in {
        "average",
        "centroid",
        "complete",
        "median",
        "single",
        "ward",
        "weighted",
    }:
        raise ValueError(f"Unsupported production cluster_model linkage={cluster_model['linkage']!r}")

    best_params = payload.get("best_params")
    if not isinstance(best_params, dict) or set(best_params) != {"eps", "linkage"}:
        raise ValueError("Production best_params must contain exactly eps and linkage")
    best_eps = _require_finite_number(best_params["eps"], field="best_params.eps")
    if best_eps != eps or str(best_params["linkage"]) != str(cluster_model["linkage"]):
        raise ValueError("Production best_params contradict cluster_model configuration")

    hybrid_weight = _require_finite_number(
        payload["incremental_mean_min_hybrid_weight"],
        field="incremental_mean_min_hybrid_weight",
    )
    if not 0.0 <= hybrid_weight <= 1.0:
        raise ValueError("Production incremental_mean_min_hybrid_weight must be in [0, 1]")
    for field in ("batch_size", "n_jobs"):
        if not isinstance(payload[field], int) or isinstance(payload[field], bool) or int(payload[field]) <= 0:
            raise ValueError(f"Production model config {field} must be a positive integer")
    if not isinstance(payload["n_iter"], int) or isinstance(payload["n_iter"], bool) or int(payload["n_iter"]) < 0:
        raise ValueError("Production model config n_iter must be a non-negative integer")
    if not isinstance(payload["random_state"], int) or isinstance(payload["random_state"], bool):
        raise ValueError("Production model config random_state must be an integer")
    val_blocks_size = payload["val_blocks_size"]
    if val_blocks_size is not None and (
        not isinstance(val_blocks_size, int) or isinstance(val_blocks_size, bool) or val_blocks_size <= 0
    ):
        raise ValueError("Production model config val_blocks_size must be null or a positive integer")
    for field in (
        "dont_merge_cluster_seeds",
        "suppress_orcid",
        "use_cache",
        "use_default_constraints_as_supervision",
    ):
        if not isinstance(payload[field], bool):
            raise ValueError(f"Production model config {field} must be boolean")
    feature_contract = payload["feature_contract"]
    if not isinstance(feature_contract, Mapping) or not feature_contract:
        raise ValueError("Production model config feature_contract must be a nonempty object")


def _validate_pairwise_metadata(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
    clusterer_config: Mapping[str, Any],
    featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo,
) -> dict[str, Any]:
    metadata = _read_json(bundle_dir / str(manifest["files"]["pairwise_metadata"]))
    if metadata.get("schema_version") != "s2and_pairwise_native_lightgbm_v1":
        raise ValueError(f"Unsupported pairwise metadata schema_version={metadata.get('schema_version')!r}")
    if metadata.get("model_family") != "binary_lightgbm_pairwise_distance":
        raise ValueError(f"Unsupported pairwise model_family={metadata.get('model_family')!r}")
    if metadata.get("class_labels") != [0.0, 1.0]:
        raise ValueError("Pairwise metadata class_labels must be [0.0, 1.0]")
    if (
        metadata.get("distance_probability_column") != "class_0"
        or metadata.get("positive_probability_column") != "class_1"
    ):
        raise ValueError("Pairwise metadata probability-column contract is invalid")
    if str(metadata.get("source_model_version")) != str(clusterer_config["source_model_version"]):
        raise ValueError("Pairwise metadata and clusterer source_model_version disagree")
    feature_contract = clusterer_config["feature_contract"]
    if metadata.get("normalization_version") != feature_contract.get("normalization_version"):
        raise ValueError("Pairwise metadata and clusterer normalization_version disagree")

    pairwise_config = clusterer_config["pairwise"]
    if not isinstance(pairwise_config, Mapping) or set(pairwise_config) != {
        "main_feature_count",
        "nameless_feature_count",
    }:
        raise ValueError("Production clusterer pairwise config is invalid")
    for field in ("main_feature_count", "nameless_feature_count"):
        value = pairwise_config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Production clusterer pairwise {field} must be a positive integer")
    for name, info, count_key, model_key, expected_filename in (
        ("main", featurizer_info, "main_feature_count", "pairwise_main_model", "main.lgb"),
        (
            "nameless",
            nameless_featurizer_info,
            "nameless_feature_count",
            "pairwise_nameless_model",
            "nameless.lgb",
        ),
    ):
        section = metadata.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(f"Pairwise metadata {name} section must be an object")
        expected_indices = list(_selected_feature_indices(info))
        declared_count = int(pairwise_config[count_key])
        if section.get("features_to_use") != [str(value) for value in info.features_to_use]:
            raise ValueError(f"Pairwise metadata {name} ordered features contradict clusterer.json")
        if int(section.get("featurizer_version", -1)) != int(info.featurizer_version):
            raise ValueError(f"Pairwise metadata {name} featurizer_version contradicts clusterer.json")
        if section.get("selected_feature_indices") != expected_indices:
            raise ValueError(f"Pairwise metadata {name} selected_feature_indices contradict clusterer.json")
        if int(section.get("selected_feature_count", -1)) != declared_count or declared_count != len(expected_indices):
            raise ValueError(f"Pairwise metadata {name} feature count contradicts clusterer.json")
        manifest_model_filename = Path(str(manifest["files"][model_key])).name
        if section.get("model_file") != expected_filename or manifest_model_filename != expected_filename:
            raise ValueError(f"Pairwise metadata {name} model_file contradicts manifest")
    return metadata


def pairwise_bundle_binding(bundle_dir: str | Path) -> dict[str, Any]:
    """Return the immutable pairwise contract used to bind a linker artifact."""

    root = Path(bundle_dir)
    manifest = _validate_manifest(root)
    clusterer_config = _read_json(root / str(manifest["files"]["clusterer_config"]))
    _validate_clusterer_config(manifest, clusterer_config)
    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    pairwise_metadata = _validate_pairwise_metadata(
        root,
        manifest,
        clusterer_config,
        featurizer_info,
        nameless_info,
    )
    feature_contract = dict(clusterer_config["feature_contract"])
    ordered_feature_contract = {
        "feature_contract": feature_contract,
        "main": dict(pairwise_metadata["main"]),
        "nameless": dict(pairwise_metadata["nameless"]),
    }
    return {
        "normalization_version": str(feature_contract["normalization_version"]),
        "featurizer_version": int(featurizer_info.featurizer_version),
        "ordered_feature_contract_digest": canonical_json_digest(ordered_feature_contract),
        "main_booster_sha256": str(manifest["sha256"][str(manifest["files"]["pairwise_main_model"])]),
        "nameless_booster_sha256": str(manifest["sha256"][str(manifest["files"]["pairwise_nameless_model"])]),
    }


def _config_choice(payload: dict[str, Any], key: str, *, allowed: frozenset[str]) -> str:
    value = str(payload[key])
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported production model config {key}={value!r}; expected one of {allowed_values}")
    return value


def _validate_pairwise_fixture(
    classifier: NativeLightGBMBinaryClassifier,
    fixture_path: Path,
) -> None:
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


def _validate_incremental_linker_metadata(linker_dir: Path) -> IncrementalLinkingArtifact:
    metadata_path = linker_dir / "metadata.json"
    booster_path = linker_dir / "booster.lgb"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Incremental linker metadata is missing: {metadata_path}")
    if not booster_path.exists():
        raise FileNotFoundError(f"Incremental linker booster is missing: {booster_path}")
    return load_incremental_linking_artifact(linker_dir, require_rust_capabilities=False)


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


def _require_bundle_normalization_version(bundle_dir: Path, feature_contract: Mapping[str, Any]) -> None:
    """Fail fast when a bundle's normalization policy differs from this package's.

    An absent feature_contract["normalization_version"] means the bundle predates the
    normalization contract and implies "legacy_compat". Unlike the featurizer-version
    warning, a mismatch here is a hard error: normalization changes the name fields
    and count keys every feature consumes, and the rollback path is redeploying the
    matching package + artifact set (docs/normalization_migration_blocked.md, OD4).
    """

    bundle_version = feature_contract.get("normalization_version", NORMALIZATION_VERSION_LEGACY_COMPAT)
    if bundle_version not in VALID_NORMALIZATION_VERSIONS:
        raise ValueError(
            f"Production bundle {bundle_dir} has invalid feature_contract['normalization_version'] "
            f"{bundle_version!r}; expected one of {sorted(VALID_NORMALIZATION_VERSIONS)}"
        )
    if bundle_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Production bundle {bundle_dir} was built with normalization_version {bundle_version!r} "
            f"but this package implements {NORMALIZATION_VERSION!r}. Code, model, and artifacts move "
            "as one release unit; redeploy the matching package or rebuild the bundle "
            "(docs/normalization_migration_blocked.md)."
        )


def _load_bundle_clusterer(bundle_dir: Path, *, require_incremental_linker: bool = True) -> Clusterer:
    manifest = _validate_manifest(bundle_dir)
    clusterer_config = _read_json(bundle_dir / str(manifest["files"]["clusterer_config"]))
    _validate_clusterer_config(manifest, clusterer_config)
    feature_contract = clusterer_config["feature_contract"]
    _require_bundle_normalization_version(bundle_dir, feature_contract)
    runtime_cluster_eps = _production_runtime_cluster_eps(
        bundle_dir,
        manifest=manifest,
        clusterer_config=clusterer_config,
    )

    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_featurizer_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    NameCountsBinding.from_feature_contract(
        feature_contract,
        context=f"Production bundle {bundle_dir} feature_contract",
        required=any("name_counts" in info.features_to_use for info in (featurizer_info, nameless_featurizer_info)),
    )
    _require_featurizer_version_match(
        bundle_dir,
        {
            "featurizer_info": featurizer_info.featurizer_version,
            "nameless_featurizer_info": nameless_featurizer_info.featurizer_version,
        },
    )
    _validate_pairwise_metadata(
        bundle_dir,
        manifest,
        clusterer_config,
        featurizer_info,
        nameless_featurizer_info,
    )
    classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_main_model"]),
        n_features=int(clusterer_config["pairwise"]["main_feature_count"]),
    )
    nameless_classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_nameless_model"]),
        n_features=int(clusterer_config["pairwise"]["nameless_feature_count"]),
    )
    _validate_pairwise_fixture(
        classifier,
        bundle_dir / str(manifest["files"]["pairwise_main_fixture"]),
    )
    _validate_pairwise_fixture(
        nameless_classifier,
        bundle_dir / str(manifest["files"]["pairwise_nameless_fixture"]),
    )

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
    clusterer.feature_contract = dict(feature_contract)
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
        incremental_linker_artifact = _validate_incremental_linker_metadata(incremental_linker_dir)
        expected_binding = pairwise_bundle_binding(bundle_dir)
        if dict(incremental_linker_artifact.metadata.pairwise_bundle_binding) != expected_binding:
            raise ValueError("Incremental linker pairwise_bundle_binding does not match enclosing bundle")
        clusterer.incremental_linker_artifact_dir = incremental_linker_dir
        clusterer.incremental_linker_artifact = incremental_linker_artifact
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

    model_path = (Path(path) if path is not None else DEFAULT_PRODUCTION_MODEL_DIR).resolve()
    if model_path.is_dir():
        return _load_bundle_clusterer(model_path, require_incremental_linker=require_incremental_linker)
    loaded = load_pickle_with_verified_label_encoder_compat(str(model_path))
    clusterer = loaded.get("clusterer") if isinstance(loaded, dict) else loaded
    if not isinstance(clusterer, Clusterer):
        raise TypeError(f"Expected a Clusterer in production model artifact, got {type(clusterer)!r}")
    _require_bundle_normalization_version(model_path, getattr(clusterer, "feature_contract", None) or {})
    versions = {"featurizer_info": int(clusterer.featurizer_info.featurizer_version)}
    nameless_info = getattr(clusterer, "nameless_featurizer_info", None)
    if nameless_info is not None:
        versions["nameless_featurizer_info"] = int(nameless_info.featurizer_version)
    _require_featurizer_version_match(model_path, versions)
    runtime_cluster_eps = _production_runtime_cluster_eps(model_path)
    return _apply_production_runtime_cluster_eps(clusterer, runtime_cluster_eps)
