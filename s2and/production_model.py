"""Load explicit complete S2AND production bundles."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from s2and._sha256 import is_lowercase_sha256
from s2and._sha256 import sha256_file as _sha256_file
from s2and.arrow_inputs import require_normalization_version
from s2and.consts import (
    FEATURIZER_VERSION,
    NORMALIZATION_VERSION,
)
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import (
    IncrementalLinkingArtifact,
    _load_incremental_linking_artifact_from_verified_booster,
)
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.incremental_linking.policy import (
    NAME_COUNTS_MANIFEST_SHA256_FIELD,
    require_name_counts_manifest_sha256,
)
from s2and.model import (
    Clusterer,
    FastCluster,
    IncrementalBroadcastMode,
    IncrementalSeedScoreMode,
    _selected_feature_indices,
)
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.production_bundle_contract import (
    CLUSTERER_CONFIG_SCHEMA_VERSION,
    COMPLETE_MANIFEST_FILES,
    PAIRWISE_ONLY_MANIFEST_FILES,
    PAIRWISE_PREDICTION_FIXTURE_SCHEMA_VERSION,
    PAIRWISE_PREDICTION_FIXTURE_TOLERANCE,
    PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES,
    PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    production_bundle_status,
    production_manifest_files,
)
from s2and.runtime import load_s2and_rust_extension
from s2and.subblocking import canonical_orcid_prefix_counts_data_sha256
from s2and.thread_config import resolve_n_jobs

logger = logging.getLogger(__name__)

_INCREMENTAL_BROADCAST_MODES = frozenset({"always", "never", "top1_consensus"})
_INCREMENTAL_SEED_SCORE_MODES = frozenset({"mean", "min", "mean_min_hybrid"})
_CANONICAL_ARTIFACT_HASH_FIELDS = (
    "name_tuples_data_sha256",
    "orcid_prefix_counts_data_sha256",
)
EXPLICIT_RELEASE_ARTIFACT_HASH_FIELDS = frozenset(
    {
        "name_counts_manifest_sha256",
        "name_tuples_data_sha256",
        "orcid_prefix_counts_data_sha256",
    }
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
        "n_jobs",
        "nameless_featurizer_info",
        "random_state",
        "schema_version",
        "suppress_orcid",
        "use_default_constraints_as_supervision",
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
    whose supported split and missing-value cases are covered by deterministic
    Python/Rust parity tests (tests/test_rust_lightgbm_booster_parity.py).
    ``booster_`` lazily loads a Python ``lgb.Booster`` only for training-side
    consumers such as bundle writing and SHAP diagnostics; runtime scoring
    never touches it.
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
        actual_features = self._validated_feature_count(self.model_path, n_features)
        self._n_features = actual_features
        self._n_features_in = actual_features
        self.n_features_in_ = actual_features

    @staticmethod
    def _validated_feature_count(model_path: str, n_features: int | None) -> int:
        """Return the model's feature count after validating optional metadata."""

        actual_features = _lightgbm_text_feature_count(Path(model_path))
        if n_features is not None and int(n_features) != actual_features:
            raise ValueError(
                "Native LightGBM feature-count mismatch: "
                f"metadata declares {int(n_features)} but booster contains {actual_features}"
            )
        return actual_features

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

        model_path = str(Path(params["model_path"])) if "model_path" in params else self.model_path
        n_jobs = resolve_n_jobs(params["n_jobs"]) if "n_jobs" in params else self.n_jobs
        feature_count: int | None = None
        if "model_path" in params or "n_features" in params:
            feature_count = self._validated_feature_count(model_path, cast(int | None, params.get("n_features")))

        model_changed = model_path != self.model_path
        self.model_path = model_path
        self.n_jobs = n_jobs
        if feature_count is not None:
            self._n_features = feature_count
            self._n_features_in = feature_count
            self.n_features_in_ = feature_count
        if model_changed:
            self._scorer = None
            self._lazy_booster = None
            self._model_sha256 = None
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
    expected_manifest_fields = {
        "bundle_version",
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
    for version_field in ("bundle_version", "pairwise_model_version"):
        version = manifest.get(version_field)
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"Production model bundle {version_field} must be a nonempty string")
    incremental_linker_version = manifest.get("incremental_linker_version")
    if incremental_linker_version is not None and (
        not isinstance(incremental_linker_version, str) or not incremental_linker_version.strip()
    ):
        raise ValueError("Production model bundle incremental_linker_version must be null or a nonempty string")

    expected_hashes = manifest.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Production model bundle manifest sha256 must be an object")
    optional_paths = set(PAIRWISE_REPRODUCIBILITY_MANIFEST_FILES.values())
    present_optional_paths = optional_paths & set(expected_hashes)
    if present_optional_paths and present_optional_paths != optional_paths:
        raise ValueError("Production model bundle must declare both pairwise reproducibility files or neither")
    expected_files = production_manifest_files(
        incremental_linker_version=incremental_linker_version,
        include_pairwise_reproducibility=present_optional_paths == optional_paths,
    )
    required_hashed_files = {str(value) for value in expected_files.values()}
    if set(expected_hashes) != required_hashed_files:
        raise ValueError(
            "Production model bundle checksum coverage mismatch: "
            f"missing={sorted(required_hashed_files - set(expected_hashes))} "
            f"extra={sorted(set(expected_hashes) - required_hashed_files)}"
        )
    for relpath, expected in expected_hashes.items():
        path = bundle_dir / relpath
        if not path.is_file():
            raise FileNotFoundError(f"Production model bundle is missing {relpath}: {path}")
        if not is_lowercase_sha256(expected):
            raise ValueError(f"Production model bundle has invalid SHA-256 for {relpath}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"Production model bundle checksum mismatch for {relpath}")
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


def _require_matching_artifact_hashes(
    feature_contract: Mapping[str, Any],
    expected_artifact_hashes: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require selected feature-contract hashes to match one authority."""

    for field, expected in expected_artifact_hashes.items():
        if not is_lowercase_sha256(expected):
            raise ValueError(f"{context} expected artifact hash {field!r} must be a lowercase SHA-256")
        observed = feature_contract.get(field)
        if not is_lowercase_sha256(observed):
            raise ValueError(f"{context} requires lowercase SHA-256 field {field!r}")
        if observed != expected:
            raise ValueError(
                f"{context} {field} does not match the explicit artifact authority: "
                f"contract={observed} expected={expected}"
            )


def require_expected_artifact_hashes(
    feature_contract: Mapping[str, Any],
    expected_artifact_hashes: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require a feature contract to match the exact explicit release authority."""

    if not isinstance(expected_artifact_hashes, Mapping) or not expected_artifact_hashes:
        raise ValueError(f"{context} requires nonempty expected artifact hashes")
    if any(not isinstance(field, str) or not field for field in expected_artifact_hashes):
        raise ValueError(f"{context} expected artifact hash fields must be nonempty strings")
    observed_fields = set(expected_artifact_hashes)
    if observed_fields != EXPLICIT_RELEASE_ARTIFACT_HASH_FIELDS:
        raise ValueError(
            f"{context} explicit artifact authority field mismatch: "
            f"missing={sorted(EXPLICIT_RELEASE_ARTIFACT_HASH_FIELDS - observed_fields)} "
            f"extra={sorted(observed_fields - EXPLICIT_RELEASE_ARTIFACT_HASH_FIELDS)}"
        )
    _require_matching_artifact_hashes(
        feature_contract,
        expected_artifact_hashes,
        context=context,
    )


def require_canonical_artifact_hashes(feature_contract: Mapping[str, Any], *, context: str) -> None:
    """Require a contract to match this package's two canonical data artifacts."""

    expected = canonical_artifact_hashes()
    _require_matching_artifact_hashes(
        feature_contract,
        {field: expected[field] for field in _CANONICAL_ARTIFACT_HASH_FIELDS},
        context=f"{context} canonical artifacts installed with this package",
    )


def _validate_clusterer_config(
    payload: dict[str, Any],
    *,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("schema_version") != CLUSTERER_CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Unsupported clusterer config schema_version={payload.get('schema_version')!r}")
    if set(payload) != _CLUSTERER_CONFIG_FIELDS:
        raise ValueError(
            "Production clusterer config field mismatch: "
            f"missing={sorted(_CLUSTERER_CONFIG_FIELDS - set(payload))} "
            f"extra={sorted(set(payload) - _CLUSTERER_CONFIG_FIELDS)}"
        )
    cluster_model = payload.get("cluster_model")
    if not isinstance(cluster_model, dict) or set(cluster_model) != {"eps", "linkage"}:
        raise ValueError("Production cluster_model must contain the exact FastCluster runtime configuration")
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
    if not isinstance(payload["random_state"], int) or isinstance(payload["random_state"], bool):
        raise ValueError("Production model config random_state must be an integer")
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
    if expected_artifact_hashes is None:
        require_canonical_artifact_hashes(feature_contract, context="Production model config feature_contract")
    else:
        require_expected_artifact_hashes(
            feature_contract,
            expected_artifact_hashes,
            context="Production model config feature_contract",
        )


def pairwise_bundle_binding(
    bundle_dir: str | Path,
    *,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the immutable pairwise contract used to bind a linker artifact."""

    root = Path(bundle_dir)
    manifest = _validate_manifest(root)
    clusterer_config = _read_json(root / PAIRWISE_ONLY_MANIFEST_FILES["clusterer_config"])
    _validate_clusterer_config(
        clusterer_config,
        expected_artifact_hashes=expected_artifact_hashes,
    )
    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
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
        "main_booster_sha256": str(manifest["sha256"][PAIRWISE_ONLY_MANIFEST_FILES["pairwise_main_model"]]),
        "nameless_booster_sha256": str(manifest["sha256"][PAIRWISE_ONLY_MANIFEST_FILES["pairwise_nameless_model"]]),
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


def _validate_incremental_linker_metadata(
    linker_dir: Path,
    *,
    verified_booster_sha256: str,
) -> IncrementalLinkingArtifact:
    metadata_path = linker_dir / "metadata.json"
    booster_path = linker_dir / "booster.lgb"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Incremental linker metadata is missing: {metadata_path}")
    if not booster_path.exists():
        raise FileNotFoundError(f"Incremental linker booster is missing: {booster_path}")
    return _load_incremental_linking_artifact_from_verified_booster(
        linker_dir,
        booster_sha256=verified_booster_sha256,
    )


def _require_bundle_normalization_version(bundle_dir: Path, feature_contract: Mapping[str, Any]) -> None:
    """Require the one normalization contract implemented by this package."""

    bundle_version = feature_contract.get("normalization_version")
    if bundle_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Production bundle {bundle_dir} was built with normalization_version {bundle_version!r} "
            f"but this package implements {NORMALIZATION_VERSION!r}. Code, model, and artifacts move "
            "as one release unit; redeploy the matching package or rebuild the bundle "
            "(docs/data.md)."
        )


def _load_bundle_clusterer(
    bundle_dir: Path,
    manifest: dict[str, Any],
    *,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> Clusterer:
    clusterer_config = _read_json(bundle_dir / PAIRWISE_ONLY_MANIFEST_FILES["clusterer_config"])
    _validate_clusterer_config(
        clusterer_config,
        expected_artifact_hashes=expected_artifact_hashes,
    )
    feature_contract = clusterer_config["feature_contract"]
    _require_bundle_normalization_version(bundle_dir, feature_contract)
    featurizer_info = _featurization_info_from_payload(clusterer_config["featurizer_info"])
    nameless_featurizer_info = _featurization_info_from_payload(clusterer_config["nameless_featurizer_info"])
    if any("name_counts" in info.features_to_use for info in (featurizer_info, nameless_featurizer_info)):
        require_name_counts_manifest_sha256(
            feature_contract.get(NAME_COUNTS_MANIFEST_SHA256_FIELD),
            context=f"Production bundle {bundle_dir} feature_contract",
        )
    _require_featurizer_version_match(
        bundle_dir,
        {
            "featurizer_info": featurizer_info.featurizer_version,
            "nameless_featurizer_info": nameless_featurizer_info.featurizer_version,
        },
    )
    classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / PAIRWISE_ONLY_MANIFEST_FILES["pairwise_main_model"],
    )
    nameless_classifier = NativeLightGBMBinaryClassifier(
        bundle_dir / PAIRWISE_ONLY_MANIFEST_FILES["pairwise_nameless_model"],
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
        bundle_dir / PAIRWISE_ONLY_MANIFEST_FILES["pairwise_main_fixture"],
    )
    _validate_pairwise_fixture(
        nameless_classifier,
        bundle_dir / PAIRWISE_ONLY_MANIFEST_FILES["pairwise_nameless_fixture"],
    )

    cluster_model_config = clusterer_config["cluster_model"]
    effective_eps = float(cluster_model_config["eps"])
    cluster_model = FastCluster(
        linkage=str(cluster_model_config["linkage"]),
        eps=effective_eps,
    )
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=classifier,
        cluster_model=cluster_model,
        search_space=None,
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
        "eps": effective_eps,
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
    if manifest["incremental_linker_version"] is not None:
        incremental_linker_dir = bundle_dir / "incremental_linker"
        incremental_booster_relpath = COMPLETE_MANIFEST_FILES["incremental_linker_booster"]
        incremental_linker_artifact = _validate_incremental_linker_metadata(
            incremental_linker_dir,
            verified_booster_sha256=str(manifest["sha256"][incremental_booster_relpath]),
        )
        expected_binding = _pairwise_binding_from_validated_parts(
            manifest,
            clusterer_config,
            featurizer_info,
            nameless_featurizer_info,
        )
        if incremental_linker_artifact.pairwise_bundle_binding_digest != canonical_json_digest(expected_binding):
            raise ValueError("Incremental linker pairwise_bundle_binding_digest does not match enclosing bundle")
        target_path = bundle_dir / COMPLETE_MANIFEST_FILES["incremental_linker_training_target"]
        if canonical_json_digest(_read_json(target_path)) != incremental_linker_artifact.target_spec_digest:
            raise ValueError("Incremental linker target_spec_digest does not match enclosing bundle target JSON")
        clusterer.incremental_linker_artifact_dir = incremental_linker_dir
        clusterer.incremental_linker_artifact = incremental_linker_artifact
    clusterer.production_model_bundle_dir = bundle_dir
    clusterer.production_model_bundle_version = str(manifest["bundle_version"])
    clusterer.production_model_bundle_status = production_bundle_status(manifest["incremental_linker_version"])
    return clusterer


def _load_pairwise_staging_model(
    path: str | Path,
    *,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> Clusterer:
    """Load an internal pairwise-only bundle used during training/finalization."""

    bundle_dir = Path(path).resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Pairwise staging model must be a native bundle directory: {bundle_dir}")
    manifest = _validate_manifest(bundle_dir)
    status = production_bundle_status(manifest["incremental_linker_version"])
    if status != "pairwise_only":
        raise ValueError(f"Expected a pairwise_only production model bundle, got {status!r}: {bundle_dir}")
    return _load_bundle_clusterer(
        bundle_dir,
        manifest,
        expected_artifact_hashes=expected_artifact_hashes,
    )


def load_production_model(
    path: str | Path | None = None,
    *,
    expected_artifact_hashes: Mapping[str, Any] | None = None,
) -> Clusterer:
    """Load a complete native production model bundle."""

    if path is None:
        raise ValueError("No default production model is declared; pass a complete native bundle path")
    bundle_dir = Path(path).resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"Production model must be a complete native bundle directory: {bundle_dir}")
    manifest = _validate_manifest(bundle_dir)
    status = production_bundle_status(manifest["incremental_linker_version"])
    if status != "complete":
        raise ValueError(f"Expected a complete production model bundle, got {status!r}: {bundle_dir}")
    return _load_bundle_clusterer(
        bundle_dir,
        manifest,
        expected_artifact_hashes=expected_artifact_hashes,
    )
