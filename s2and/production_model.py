"""Load explicit complete S2AND production bundles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from s2and.arrow_inputs import require_normalization_version
from s2and.consts import (
    FEATURIZER_VERSION,
    NORMALIZATION_VERSION,
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
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.production_bundle_contract import (
    CLUSTERER_CONFIG_SCHEMA_VERSION,
    PAIRWISE_METADATA_SCHEMA_VERSION,
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PAIRWISE_PREDICTION_FIXTURE_TOLERANCE,
    PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    production_manifest_files,
)
from s2and.runtime import load_s2and_rust_extension
from s2and.subblocking import canonical_orcid_prefix_counts_data_sha256
from s2and.thread_config import resolve_n_jobs

_INCREMENTAL_BROADCAST_MODES = frozenset({"always", "never", "top1_consensus"})
_INCREMENTAL_SEED_SCORE_MODES = frozenset({"mean", "min", "mean_min_hybrid"})
_CANONICAL_ARTIFACT_HASH_FIELDS = (
    "name_tuples_data_sha256",
    "orcid_prefix_counts_data_sha256",
)
_CLUSTERER_CONFIG_FIELDS = frozenset(
    {
        "batch_size",
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
        "random_state",
        "schema_version",
        "suppress_orcid",
        "use_default_constraints_as_supervision",
        "val_blocks_size",
    }
)


def _load_rust_lightgbm_booster(model_path: str) -> Any:
    return load_s2and_rust_extension().RustLightGBMBooster(model_path)


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
    return files


def _validate_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported production model bundle schema_version={manifest.get('schema_version')!r}")
    expected_manifest_fields = {
        "bundle_status",
        "bundle_version",
        "files",
        "incremental_linker_version",
        "pairwise_model_version",
        "schema_version",
        "sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError(
            "Production model manifest field mismatch: "
            f"missing={sorted(expected_manifest_fields - set(manifest))} "
            f"extra={sorted(set(manifest) - expected_manifest_fields)}"
        )
    status = manifest.get("bundle_status")
    if status not in {"pairwise_only", "complete"}:
        raise ValueError(f"Unsupported production model bundle_status={status!r}")
    complete = status == "complete"
    for version_field in ("bundle_version", "pairwise_model_version"):
        version = manifest.get(version_field)
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Production model bundle {version_field} must be a nonempty string")
    incremental_linker_version = manifest.get("incremental_linker_version")
    if complete:
        if not isinstance(incremental_linker_version, str) or not incremental_linker_version.strip():
            raise ValueError("Complete production model bundle incremental_linker_version must be a nonempty string")
    elif incremental_linker_version is not None:
        raise ValueError("Pairwise-only production model bundle incremental_linker_version must be null")
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
    if isinstance(value, bool):
        raise ValueError(f"Production model config {field} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Production model config {field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Production model config {field} must be finite, got {value!r}")
    return number


def canonical_artifact_hashes() -> dict[str, str]:
    """Return content hashes for the canonical artifacts used by production."""

    return {
        "name_tuples_data_sha256": load_packaged_name_tuple_artifact().data_sha256,
        "orcid_prefix_counts_data_sha256": canonical_orcid_prefix_counts_data_sha256(),
    }


def require_canonical_artifact_hashes(feature_contract: Mapping[str, Any], *, context: str) -> None:
    """Require a contract to match this package's two canonical data artifacts."""

    observed: dict[str, str] = {}
    for field in _CANONICAL_ARTIFACT_HASH_FIELDS:
        value = feature_contract.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{context} requires lowercase SHA-256 field {field!r}")
        observed[field] = value
    expected = canonical_artifact_hashes()
    for field in _CANONICAL_ARTIFACT_HASH_FIELDS:
        if observed[field] != expected[field]:
            raise ValueError(
                f"{context} {field} does not match the canonical artifact installed with this package: "
                f"contract={observed[field]} package={expected[field]}"
            )


def _validate_clusterer_config(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != CLUSTERER_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported clusterer config schema_version={payload.get('schema_version')!r}")
    if set(payload) != _CLUSTERER_CONFIG_FIELDS:
        raise ValueError(
            "Production clusterer config field mismatch: "
            f"missing={sorted(_CLUSTERER_CONFIG_FIELDS - set(payload))} "
            f"extra={sorted(set(payload) - _CLUSTERER_CONFIG_FIELDS)}"
        )
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
        "use_default_constraints_as_supervision",
    ):
        if not isinstance(payload[field], bool):
            raise ValueError(f"Production model config {field} must be boolean")
    feature_contract = payload["feature_contract"]
    if not isinstance(feature_contract, Mapping) or not feature_contract:
        raise ValueError("Production model config feature_contract must be a nonempty object")
    require_normalization_version(
        feature_contract.get("normalization_version"),
        context="Production model config feature_contract",
    )
    require_canonical_artifact_hashes(feature_contract, context="Production model config feature_contract")


def _validate_pairwise_metadata(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    metadata = _read_json(bundle_dir / str(manifest["files"]["pairwise_metadata"]))
    if metadata.get("schema_version") != PAIRWISE_METADATA_SCHEMA_VERSION:
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
    expected_metadata_fields = {
        "class_labels",
        "distance_probability_column",
        "model_family",
        "positive_probability_column",
        "schema_version",
    }
    if set(metadata) != expected_metadata_fields:
        raise ValueError(
            "Pairwise metadata field mismatch: "
            f"missing={sorted(expected_metadata_fields - set(metadata))} "
            f"extra={sorted(set(metadata) - expected_metadata_fields)}"
        )


def pairwise_bundle_binding(bundle_dir: str | Path) -> dict[str, Any]:
    """Return the immutable pairwise contract used to bind a linker artifact."""

    root = Path(bundle_dir)
    manifest = _validate_manifest(root)
    clusterer_config = _read_json(root / str(manifest["files"]["clusterer_config"]))
    _validate_clusterer_config(clusterer_config)
    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    _validate_pairwise_metadata(root, manifest)
    return _pairwise_binding_from_validated_parts(manifest, clusterer_config, featurizer_info, nameless_info)


def _pairwise_binding_from_validated_parts(
    manifest: Mapping[str, Any],
    clusterer_config: Mapping[str, Any],
    featurizer_info: FeaturizationInfo,
    nameless_info: FeaturizationInfo,
) -> dict[str, Any]:
    """Build the pairwise binding from bundle parts a caller already validated."""

    feature_contract = dict(clusterer_config["feature_contract"])
    ordered_feature_contract = {
        "feature_contract": feature_contract,
        "main": {
            "features_to_use": list(featurizer_info.features_to_use),
            "featurizer_version": int(featurizer_info.featurizer_version),
            "selected_feature_indices": list(_selected_feature_indices(featurizer_info)),
        },
        "nameless": {
            "features_to_use": list(nameless_info.features_to_use),
            "featurizer_version": int(nameless_info.featurizer_version),
            "selected_feature_indices": list(_selected_feature_indices(nameless_info)),
        },
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
    rtol = _require_finite_number(fixture.get("rtol"), field="pairwise prediction fixture rtol")
    atol = _require_finite_number(fixture.get("atol"), field="pairwise prediction fixture atol")
    if rtol != PAIRWISE_PREDICTION_FIXTURE_TOLERANCE or atol != PAIRWISE_PREDICTION_FIXTURE_TOLERANCE:
        raise ValueError(
            f"Pairwise prediction fixture tolerances must both equal {PAIRWISE_PREDICTION_FIXTURE_TOLERANCE}"
        )
    if observed.shape != expected.shape:
        raise ValueError(f"Pairwise prediction fixture shape mismatch: {observed.shape} != {expected.shape}")
    if not np.all(np.isfinite(expected)) or np.any((expected < 0) | (expected > 1)):
        raise ValueError(f"Pairwise prediction fixture probabilities must be finite and in [0, 1]: {fixture_path}")
    if not np.allclose(observed, expected, rtol=rtol, atol=atol):
        raise ValueError(f"Pairwise prediction fixture mismatch for {fixture_path}")


def _validate_incremental_linker_metadata(linker_dir: Path) -> IncrementalLinkingArtifact:
    metadata_path = linker_dir / "metadata.json"
    booster_path = linker_dir / "booster.lgb"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Incremental linker metadata is missing: {metadata_path}")
    if not booster_path.exists():
        raise FileNotFoundError(f"Incremental linker booster is missing: {booster_path}")
    return load_incremental_linking_artifact(linker_dir)


def _require_bundle_normalization_version(bundle_dir: Path, feature_contract: Mapping[str, Any]) -> None:
    """Require the one normalization contract implemented by this package."""

    bundle_version = feature_contract.get("normalization_version")
    if bundle_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Production bundle {bundle_dir} was built with normalization_version {bundle_version!r} "
            f"but this package implements {NORMALIZATION_VERSION!r}. Code, model, and artifacts move "
            "as one release unit; redeploy the matching package or rebuild the bundle "
            "(docs/normalization_migration_blocked.md)."
        )


def _load_bundle_clusterer(bundle_dir: Path, manifest: dict[str, Any]) -> Clusterer:
    clusterer_config = _read_json(bundle_dir / str(manifest["files"]["clusterer_config"]))
    _validate_clusterer_config(clusterer_config)
    feature_contract = clusterer_config["feature_contract"]
    _require_bundle_normalization_version(bundle_dir, feature_contract)
    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_featurizer_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    if any("name_counts" in info.features_to_use for info in (featurizer_info, nameless_featurizer_info)):
        NameCountsBinding.from_feature_contract(
            feature_contract,
            context=f"Production bundle {bundle_dir} feature_contract",
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
    )
    classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_main_model"]),
    )
    nameless_classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / str(manifest["files"]["pairwise_nameless_model"]),
    )
    for name, model, info in (
        ("main", classifier, featurizer_info),
        ("nameless", nameless_classifier, nameless_featurizer_info),
    ):
        expected_count = len(_selected_feature_indices(info))
        if model.n_features_in_ != expected_count:
            raise ValueError(
                f"Production {name} booster feature count {model.n_features_in_} contradicts "
                f"the clusterer featurizer count {expected_count}"
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
        use_default_constraints_as_supervision=bool(clusterer_config["use_default_constraints_as_supervision"]),
        random_state=int(clusterer_config["random_state"]),
        nameless_classifier=nameless_classifier,
        nameless_featurizer_info=nameless_featurizer_info,
        dont_merge_cluster_seeds=bool(clusterer_config["dont_merge_cluster_seeds"]),
        batch_size=int(clusterer_config["batch_size"]),
        suppress_orcid=bool(clusterer_config["suppress_orcid"]),
    )
    clusterer.feature_contract = dict(feature_contract)
    clusterer.best_params = {
        "eps": float(cluster_model_config["eps"]),
        "linkage": str(cluster_model_config["linkage"]),
    }
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
    if incremental_linker_relpath is not None:
        incremental_linker_dir = bundle_dir / str(incremental_linker_relpath)
        incremental_linker_artifact = _validate_incremental_linker_metadata(incremental_linker_dir)
        expected_binding = _pairwise_binding_from_validated_parts(
            manifest,
            clusterer_config,
            featurizer_info,
            nameless_featurizer_info,
        )
        if dict(incremental_linker_artifact.metadata.pairwise_bundle_binding) != expected_binding:
            raise ValueError("Incremental linker pairwise_bundle_binding does not match enclosing bundle")
        target_path = bundle_dir / str(manifest["files"]["incremental_linker_training_target"])
        if canonical_json_digest(_read_json(target_path)) != incremental_linker_artifact.metadata.target_spec_digest:
            raise ValueError("Incremental linker target_spec_digest does not match enclosing bundle target JSON")
        clusterer.incremental_linker_artifact_dir = incremental_linker_dir
        clusterer.incremental_linker_artifact = incremental_linker_artifact
    clusterer.production_model_bundle_dir = bundle_dir
    clusterer.production_model_bundle_version = str(manifest["bundle_version"])
    clusterer.production_model_bundle_status = str(manifest["bundle_status"])
    return clusterer


def _load_pairwise_staging_model(path: str | Path) -> Clusterer:
    """Load an internal pairwise-only bundle used during training/finalization."""

    bundle_dir = Path(path).resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Pairwise staging model must be a native bundle directory: {bundle_dir}")
    manifest = _validate_manifest(bundle_dir)
    if manifest["bundle_status"] != "pairwise_only":
        raise ValueError(
            f"Expected a pairwise_only production model bundle, got {manifest['bundle_status']!r}: {bundle_dir}"
        )
    return _load_bundle_clusterer(bundle_dir, manifest)


def load_production_model(path: str | Path | None = None) -> Clusterer:
    """Load a complete native production model bundle."""

    if path is None:
        raise ValueError("No default production model is declared; pass a complete native bundle path")
    bundle_dir = Path(path).resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Production model must be a complete native bundle directory: {bundle_dir}")
    manifest = _validate_manifest(bundle_dir)
    if manifest["bundle_status"] != "complete":
        raise ValueError(
            f"Expected a complete production model bundle, got {manifest['bundle_status']!r}: {bundle_dir}"
        )
    return _load_bundle_clusterer(bundle_dir, manifest)
