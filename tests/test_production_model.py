from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.incremental_linking.artifact as incremental_artifact_module
import s2and.production_bundle as production_bundle_module
import s2and.production_model as production_model_module
import s2and.subblocking as subblocking_module
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, FastCluster, _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.model_pairwise import _validated_classifier_features
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import (
    NativeLightGBMBinaryClassifier,
    _config_choice,
    _load_pairwise_staging_model,
    _require_featurizer_version_match,
    load_production_model,
    pairwise_bundle_binding,
)
from tests.helpers import tiny_name_counts_index
from tests.promoted_linking_helpers import build_tiny_promoted_booster

_TEST_CANONICAL_ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "a" * 64,
    "orcid_prefix_counts_data_sha256": "b" * 64,
}


class _PythonLightGBMScorer:
    """Test double for the Rust scorer that executes the same saved model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = str(model_path)
        self.booster = lgb.Booster(model_file=self.model_path)

    def num_features(self) -> int:
        return int(self.booster.num_feature())

    def predict_proba_positive(self, features: np.ndarray, *, num_threads: int | None = None) -> np.ndarray:
        return np.asarray(self.booster.predict(features, num_threads=num_threads), dtype=np.float64)

    def predict_proba_positive_f32(self, features: np.ndarray, *, num_threads: int | None = None) -> np.ndarray:
        return self.predict_proba_positive(features, num_threads=num_threads)

    def __deepcopy__(self, memo: dict[int, Any]) -> _PythonLightGBMScorer:
        del memo
        return type(self)(self.model_path)


def _tiny_binary_booster(width: int, *, seed: int) -> lgb.LGBMClassifier:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(32, width))
    labels = np.asarray([0, 1] * 16, dtype=np.int8)
    classifier = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=1,
        learning_rate=0.2,
        num_leaves=3,
        min_child_samples=1,
        min_data_in_bin=1,
        force_col_wise=True,
        n_estimators=4,
        random_state=seed,
    )
    classifier.fit(matrix, labels)
    return classifier


@pytest.fixture
def synthetic_pairwise_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Clusterer]:
    monkeypatch.setattr(production_model_module, "_load_rust_lightgbm_booster", _PythonLightGBMScorer)
    monkeypatch.setattr(incremental_artifact_module, "_load_rust_lightgbm_booster", _PythonLightGBMScorer)
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: dict(_TEST_CANONICAL_ARTIFACT_HASHES),
    )
    main_info = FeaturizationInfo(["name_similarity"], featurizer_version=FEATURIZER_VERSION)
    nameless_info = FeaturizationInfo(["year_diff"], featurizer_version=FEATURIZER_VERSION)
    source_clusterer = Clusterer(
        main_info,
        _tiny_binary_booster(len(_selected_feature_indices(main_info)), seed=101),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        nameless_classifier=_tiny_binary_booster(len(_selected_feature_indices(nameless_info)), seed=102),
        nameless_featurizer_info=nameless_info,
        batch_size=100,
    )
    source_clusterer.feature_contract = {
        "normalization_version": NORMALIZATION_VERSION,
        **_TEST_CANONICAL_ARTIFACT_HASHES,
    }
    source_clusterer.best_params = {"eps": 0.5, "linkage": "average"}
    bundle_dir = tmp_path / "production_model_v9.9"
    write_pairwise_production_bundle(
        source_clusterer,
        bundle_dir,
        bundle_version="9.9",
        source_model_version="9.9",
    )
    return bundle_dir, source_clusterer


def _write_synthetic_linker(pairwise_bundle: Path, linker_dir: Path) -> Path:
    booster, _fixture = build_tiny_promoted_booster()
    gate_config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )
    target_spec: dict[str, Any] = {}
    save_incremental_linking_artifact(
        booster,
        linker_dir,
        gate_config=gate_config,
        target_spec=target_spec,
        pairwise_bundle_binding=pairwise_bundle_binding(pairwise_bundle),
    )
    target = linker_dir.parent / "target.json"
    target.write_text(json.dumps(target_spec) + "\n", encoding="utf-8")
    return target


def _refresh_manifest_checksum(bundle_dir: Path, relpath: str) -> None:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"][relpath] = hashlib.sha256((bundle_dir / relpath).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_dummy_inference_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode="inference",
        name_counts_index=tiny_name_counts_index(),
        preprocess=True,
        n_jobs=1,
    )


def _prepare_prediction_clusterer(clusterer):
    _ensure_lightgbm_fitted(clusterer.classifier)
    _ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.n_jobs = 1
    return clusterer


def _predict_dummy_block(clusterer, *, batching_threshold: int | None) -> dict[str, list[str]]:
    dataset = _load_dummy_inference_dataset(f"dummy-predict-{batching_threshold}")
    block = {
        "a sattar": [str(signature_index) for signature_index in range(9)],
    }
    predictions, dists = clusterer.predict(block, dataset, batching_threshold=batching_threshold)

    assert dists is None
    return predictions


def test_production_model_requires_explicit_complete_bundle_path() -> None:
    with pytest.raises(ValueError, match="No default production model is declared"):
        load_production_model()


def test_published_v121_runtime_eps_policy_matches_exact_artifact_identity() -> None:
    main_path = production_model_module.PAIRWISE_ONLY_MANIFEST_FILES["pairwise_main_model"]
    nameless_path = production_model_module.PAIRWISE_ONLY_MANIFEST_FILES["pairwise_nameless_model"]
    expected_hashes = dict(production_model_module._PUBLISHED_V121_PAIRWISE_SHA256)
    exact_manifest = {
        "bundle_version": "unrelated-name",
        "pairwise_model_version": "1.2",
        "sha256": expected_hashes,
    }
    stale_config = {"eps": production_model_module._PUBLISHED_V121_STORED_CLUSTER_EPS}

    assert (
        production_model_module._effective_cluster_eps(exact_manifest, stale_config)
        == production_model_module.PUBLISHED_V121_RUNTIME_CLUSTER_EPS
    )
    assert production_model_module._effective_cluster_eps(exact_manifest, {"eps": 0.65}) == 0.65
    assert production_model_module._effective_cluster_eps(exact_manifest, {"eps": 0.55}) == 0.55
    assert (
        production_model_module._effective_cluster_eps(
            {**exact_manifest, "pairwise_model_version": "1.3"},
            stale_config,
        )
        == production_model_module._PUBLISHED_V121_STORED_CLUSTER_EPS
    )
    changed_hashes = dict(expected_hashes)
    changed_hashes[main_path] = "0" * 64
    assert changed_hashes[nameless_path] == expected_hashes[nameless_path]
    assert (
        production_model_module._effective_cluster_eps(
            {**exact_manifest, "sha256": changed_hashes},
            stale_config,
        )
        == production_model_module._PUBLISHED_V121_STORED_CLUSTER_EPS
    )


def test_published_v121_runtime_eps_applies_to_staging_and_complete_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, _ = synthetic_pairwise_bundle
    config_path = source_bundle / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["cluster_model"]["eps"] = production_model_module._PUBLISHED_V121_STORED_CLUSTER_EPS
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(source_bundle, "clusterer.json")

    manifest_path = source_bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pairwise_model_version"] = "1.2"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    synthetic_booster_hashes = {
        path: manifest["sha256"][path] for path in production_model_module._PUBLISHED_V121_PAIRWISE_SHA256
    }
    monkeypatch.setattr(
        production_model_module,
        "_PUBLISHED_V121_PAIRWISE_SHA256",
        synthetic_booster_hashes,
    )

    staged = _load_pairwise_staging_model(source_bundle)
    assert staged.cluster_model.eps == 0.65
    assert staged.best_params["eps"] == 0.65

    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(source_bundle, linker_dir)
    complete_bundle = tmp_path / "production_model_v1.21"
    finalize_production_bundle(
        pairwise_bundle_dir=source_bundle,
        output_bundle_dir=complete_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="1.21",
        incremental_linker_version="1.21",
    )

    loaded = load_production_model(complete_bundle)
    assert loaded.cluster_model.eps == 0.65
    assert loaded.best_params["eps"] == 0.65
    assert loaded.production_model_bundle_version == "1.21"


def test_native_production_bundle_loads_as_mutable_clusterer(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = _load_pairwise_staging_model(bundle_dir)

    assert isinstance(clusterer.classifier, NativeLightGBMBinaryClassifier)
    assert isinstance(clusterer.nameless_classifier, NativeLightGBMBinaryClassifier)
    assert clusterer.incremental_linker_artifact_dir is None
    assert clusterer.production_model_bundle_version == "9.9"

    clusterer.n_jobs = 7
    clusterer.cluster_model.eps = 0.5

    assert clusterer.n_jobs == 7
    assert clusterer.classifier.n_jobs == 7
    assert clusterer.nameless_classifier.n_jobs == 7
    assert clusterer.cluster_model.eps == 0.5
    assert {
        field: clusterer.feature_contract[field] for field in _TEST_CANONICAL_ARTIFACT_HASHES
    } == _TEST_CANONICAL_ARTIFACT_HASHES


def test_production_name_count_model_requires_exact_binding(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config_path = bundle_dir / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["featurizer_info"]["features_to_use"] = ["name_counts"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")

    with pytest.raises(ValueError, match="requires name_counts_manifest_sha256"):
        _load_pairwise_staging_model(bundle_dir)


def test_native_lightgbm_set_params_rejects_unknown_params(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = _load_pairwise_staging_model(bundle_dir)

    with pytest.raises(ValueError, match="Invalid parameter"):
        clusterer.classifier.set_params(learning_rate=0.1)


def test_native_lightgbm_set_params_is_atomic_on_validation_failure(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    classifier = _load_pairwise_staging_model(bundle_dir).classifier
    replacement_path = tmp_path / "replacement.lgb"
    _tiny_binary_booster(classifier.n_features_in_ + 1, seed=103).booster_.save_model(str(replacement_path))
    original_params = classifier.get_params()
    original_scorer = classifier._rust_scorer
    original_booster = classifier.booster_
    original_fingerprint = classifier.cache_fingerprint()

    with pytest.raises(ValueError, match="feature-count mismatch"):
        classifier.set_params(
            model_path=replacement_path,
            n_features=classifier.n_features_in_,
        )

    assert classifier.get_params() == original_params
    assert classifier._scorer is original_scorer
    assert classifier._lazy_booster is original_booster
    assert classifier.cache_fingerprint() == original_fingerprint

    with pytest.raises(ValueError, match="must not be zero"):
        classifier.set_params(model_path=replacement_path, n_jobs=0)

    assert classifier.get_params() == original_params
    assert classifier._scorer is original_scorer
    assert classifier._lazy_booster is original_booster
    assert classifier.cache_fingerprint() == original_fingerprint


def test_native_lightgbm_deepcopy_does_not_require_model_path(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = _load_pairwise_staging_model(bundle_dir)
    classifier = clusterer.classifier
    features = np.zeros((2, classifier.n_features_in_), dtype=np.float64)
    expected = classifier.predict_proba(features)
    classifier.model_path = str(tmp_path / "missing_model.txt")

    copied = copy.deepcopy(classifier)

    np.testing.assert_allclose(copied.predict_proba(features), expected)
    assert copied.model_path == classifier.model_path
    assert copied._scorer is classifier._scorer
    copied.n_jobs = 3
    assert classifier.n_jobs != copied.n_jobs


def test_synthetic_native_pairwise_models_match_source_boosters(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    native_clusterer = _load_pairwise_staging_model(bundle_dir)

    rng = np.random.default_rng(921)
    main_width = len(_selected_feature_indices(source_clusterer.featurizer_info))
    assert source_clusterer.nameless_featurizer_info is not None
    nameless_width = len(_selected_feature_indices(source_clusterer.nameless_featurizer_info))
    main_features = rng.normal(size=(8, main_width))
    nameless_features = rng.normal(size=(8, nameless_width))
    source_main_features = _validated_classifier_features(source_clusterer.classifier, main_features)
    source_nameless_features = _validated_classifier_features(source_clusterer.nameless_classifier, nameless_features)

    np.testing.assert_allclose(
        native_clusterer.classifier.predict_proba(main_features)[:, 1],
        source_clusterer.classifier.predict_proba(source_main_features)[:, 1],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_clusterer.classifier.predict_proba_positive(main_features),
        native_clusterer.classifier.predict_proba(main_features)[:, 1],
        rtol=0,
        atol=0,
    )
    assert native_clusterer.nameless_classifier is not None
    assert source_clusterer.nameless_classifier is not None
    np.testing.assert_allclose(
        native_clusterer.nameless_classifier.predict_proba(nameless_features)[:, 1],
        source_clusterer.nameless_classifier.predict_proba(source_nameless_features)[:, 1],
        rtol=1e-10,
        atol=1e-10,
    )


def test_synthetic_native_clusterer_predict_matches_source_python(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(subblocking_module._LazyCanonicalOrcidPrefixCounts, "load", lambda _self: {})  # noqa: SLF001
    bundle_dir, source_clusterer = synthetic_pairwise_bundle

    for batching_threshold in (None, 7):
        native_clusterer = _prepare_prediction_clusterer(_load_pairwise_staging_model(bundle_dir))
        expected_clusterer = _prepare_prediction_clusterer(copy.deepcopy(source_clusterer))

        assert _predict_dummy_block(native_clusterer, batching_threshold=batching_threshold) == _predict_dummy_block(
            expected_clusterer,
            batching_threshold=batching_threshold,
        )


def test_native_clusterer_predict_is_explicitly_python_even_with_rust_environment(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    native_clusterer = _prepare_prediction_clusterer(_load_pairwise_staging_model(bundle_dir))

    monkeypatch.setenv("S2AND_BACKEND", "python")
    expected = _predict_dummy_block(
        _prepare_prediction_clusterer(copy.deepcopy(source_clusterer)),
        batching_threshold=None,
    )

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    assert _predict_dummy_block(native_clusterer, batching_threshold=None) == expected


def test_synthetic_native_clusterer_runtime_config_round_trips(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    native_clusterer = _load_pairwise_staging_model(bundle_dir)

    assert type(native_clusterer.cluster_model) is type(source_clusterer.cluster_model)
    assert native_clusterer.cluster_model.linkage == source_clusterer.cluster_model.linkage
    assert native_clusterer.cluster_model.eps == source_clusterer.cluster_model.eps
    assert native_clusterer.featurizer_info.features_to_use == source_clusterer.featurizer_info.features_to_use
    assert native_clusterer.featurizer_info.featurizer_version == source_clusterer.featurizer_info.featurizer_version
    assert native_clusterer.nameless_featurizer_info is not None
    assert source_clusterer.nameless_featurizer_info is not None
    assert (
        native_clusterer.nameless_featurizer_info.features_to_use
        == source_clusterer.nameless_featurizer_info.features_to_use
    )
    assert (
        native_clusterer.nameless_featurizer_info.featurizer_version
        == source_clusterer.nameless_featurizer_info.featurizer_version
    )
    assert native_clusterer.best_params == source_clusterer.best_params
    assert native_clusterer.batch_size == source_clusterer.batch_size
    assert native_clusterer.dont_merge_cluster_seeds == source_clusterer.dont_merge_cluster_seeds
    assert (
        native_clusterer.use_default_constraints_as_supervision
        == source_clusterer.use_default_constraints_as_supervision
    )
    assert native_clusterer.n_iter == source_clusterer.n_iter
    assert native_clusterer.random_state == source_clusterer.random_state

    assert getattr(native_clusterer, "suppress_orcid", False) == getattr(source_clusterer, "suppress_orcid", False)
    assert native_clusterer._incremental_experiment_config() == source_clusterer._incremental_experiment_config()


def test_featurizer_version_mismatch_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="FEATURIZER_VERSION"):
        _require_featurizer_version_match(
            tmp_path / "production_model_vtest",
            {"featurizer_info": -1, "nameless_featurizer_info": -1},
        )


def test_bundle_export_rejects_missing_normalization_provenance(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.pop("normalization_version")

    with pytest.raises(ValueError, match="missing provenance"):
        write_pairwise_production_bundle(
            source_clusterer,
            tmp_path / "missing-provenance",
            bundle_version="10.0",
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("name_tuples_data_sha256", None),
        ("name_tuples_data_sha256", "f" * 63),
        ("orcid_prefix_counts_data_sha256", "F" * 64),
        ("orcid_prefix_counts_data_sha256", "c" * 64),
    ),
)
def test_canonical_artifact_hash_contract_rejects_missing_malformed_or_mismatched_values(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: str | None,
) -> None:
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: dict(_TEST_CANONICAL_ARTIFACT_HASHES),
    )
    feature_contract = dict(_TEST_CANONICAL_ARTIFACT_HASHES)
    if invalid is None:
        feature_contract.pop(field)
    else:
        feature_contract[field] = invalid

    with pytest.raises(ValueError, match=field):
        production_model_module.require_canonical_artifact_hashes(
            feature_contract,
            context="test feature_contract",
        )


def test_bundle_export_rejects_missing_canonical_artifact_hash(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.pop("name_tuples_data_sha256")

    with pytest.raises(ValueError, match="name_tuples_data_sha256"):
        write_pairwise_production_bundle(
            source_clusterer,
            tmp_path / "missing-name-tuples-hash",
            bundle_version="10.0",
        )


def test_bundle_load_rejects_canonical_artifact_hash_mismatch(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config_path = bundle_dir / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    field = "orcid_prefix_counts_data_sha256"
    config["feature_contract"][field] = "c" * 64
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")

    with pytest.raises(ValueError, match=rf"{field} does not match"):
        _load_pairwise_staging_model(bundle_dir)


@pytest.mark.parametrize(
    ("path", "schema_version"),
    (
        ("manifest.json", "s2and_production_model_bundle_v2"),
        ("manifest.json", "s2and_production_model_bundle_v4"),
        ("clusterer.json", "s2and_clusterer_config_v2"),
        ("clusterer.json", "s2and_clusterer_config_v3"),
    ),
)
def test_bundle_rejects_previous_schema_versions(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    path: str,
    schema_version: str,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    artifact_path = bundle_dir / path
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if path == "clusterer.json":
        _refresh_manifest_checksum(bundle_dir, path)

    with pytest.raises(ValueError, match="schema_version"):
        _load_pairwise_staging_model(bundle_dir)


@pytest.mark.parametrize("field", tuple(_TEST_CANONICAL_ARTIFACT_HASHES))
def test_canonical_artifact_hashes_feed_pairwise_bundle_binding(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    field: str,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    original_binding = pairwise_bundle_binding(bundle_dir)
    changed_hashes = dict(_TEST_CANONICAL_ARTIFACT_HASHES)
    changed_hashes[field] = "c" * 64
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(changed_hashes))

    config_path = bundle_dir / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["feature_contract"][field] = changed_hashes[field]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")

    changed_binding = pairwise_bundle_binding(bundle_dir)
    assert changed_binding["ordered_feature_contract_digest"] != original_binding["ordered_feature_contract_digest"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bundle_status", "pairwise_only"),
        ("files", {"clusterer_config": "clusterer.json"}),
    ),
)
def test_manifest_rejects_redundant_legacy_state_fields(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    field: str,
    value: object,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest field mismatch"):
        _load_pairwise_staging_model(bundle_dir)


@pytest.mark.parametrize("field_name", ("bundle_version", "pairwise_model_version"))
def test_manifest_requires_nonempty_string_model_versions(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    field_name: str,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for invalid_value in (None, "   "):
        manifest = copy.deepcopy(original_manifest)
        manifest[field_name] = invalid_value
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=rf"{field_name} must be a nonempty string"):
            production_model_module._validate_manifest(bundle_dir)


def test_manifest_requires_null_or_nonempty_incremental_linker_version(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = copy.deepcopy(original_manifest)
    manifest["incremental_linker_version"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incremental_linker_version must be null or a nonempty string"):
        production_model_module._validate_manifest(bundle_dir)


def test_complete_manifest_requires_nonempty_string_incremental_linker_version(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    pairwise_bundle, _ = synthetic_pairwise_bundle
    complete_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(pairwise_bundle, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise_bundle,
        output_bundle_dir=complete_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )
    manifest_path = complete_bundle / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for invalid_value, message in ((None, "checksum coverage mismatch"), ("   ", "null or a nonempty string")):
        manifest = copy.deepcopy(original_manifest)
        manifest["incremental_linker_version"] = invalid_value
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            production_model_module._validate_manifest(complete_bundle)


def test_manifest_requires_complete_checksum_coverage(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sha256"]["pairwise/main.lgb"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum coverage mismatch"):
        _load_pairwise_staging_model(bundle_dir)


def test_manifest_ignores_undeclared_runtime_file(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    (bundle_dir / "pairwise" / "stale.lgb").write_text("stale", encoding="utf-8")

    clusterer = _load_pairwise_staging_model(bundle_dir)

    assert clusterer.production_model_bundle_status == "pairwise_only"


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda payload: payload["cluster_model"].update({"eps": float("nan")}), "must be finite"),
        (lambda payload: payload["cluster_model"].update({"eps": True}), "must be numeric"),
        (
            lambda payload: payload.update({"incremental_mean_min_hybrid_weight": False}),
            "must be numeric",
        ),
        (lambda payload: payload["cluster_model"].update({"family": "Other"}), "family"),
        (lambda payload: payload.update({"unknown_runtime_field": 1}), "field mismatch"),
        (
            lambda payload: payload.update({"incremental_mean_min_hybrid_weight": 2.0}),
            "hybrid_weight must be in",
        ),
    ),
)
def test_clusterer_config_rejects_nonfinite_unknown_and_contradictory_values(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    mutate: Any,
    message: str,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config_path = bundle_dir / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mutate(config)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")

    with pytest.raises(ValueError, match=message):
        _load_pairwise_staging_model(bundle_dir)


def test_pairwise_fixture_cannot_relax_tolerance_to_accept_wrong_predictions(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    fixture_path = bundle_dir / "pairwise" / "main_prediction_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["expected_probabilities"] = np.ones_like(fixture["expected_probabilities"]).tolist()
    fixture["rtol"] = 1e9
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "pairwise/main_prediction_fixture.json")

    with pytest.raises(ValueError, match="tolerances must both equal"):
        _load_pairwise_staging_model(bundle_dir)


def test_native_classifier_rejects_declared_booster_feature_count(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    model_path = bundle_dir / "pairwise" / "main.lgb"
    actual = _PythonLightGBMScorer(str(model_path)).num_features()

    with pytest.raises(ValueError, match="feature-count mismatch"):
        NativeLightGBMBinaryClassifier(model_path, n_features=actual + 1)


def test_native_classifier_scores_in_bounded_float32_chunks(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    model_path = bundle_dir / "pairwise" / "main.lgb"
    classifier = NativeLightGBMBinaryClassifier(model_path, n_jobs=1)

    class CountingScorer(_PythonLightGBMScorer):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.row_counts: list[int] = []

        def predict_proba_positive_f32(
            self,
            features: np.ndarray,
            *,
            num_threads: int | None = None,
        ) -> np.ndarray:
            self.row_counts.append(int(features.shape[0]))
            return super().predict_proba_positive_f32(features, num_threads=num_threads)

    scorer = CountingScorer(str(model_path))
    classifier._scorer = scorer  # noqa: SLF001
    matrix = np.ascontiguousarray(
        np.random.default_rng(20260709).normal(size=(5, scorer.num_features())),
        dtype=np.float32,
    )

    full = classifier.predict_proba_positive(matrix)
    chunked = classifier.predict_proba_positive(matrix, max_rows_per_chunk=2)

    np.testing.assert_array_equal(chunked, full)
    assert scorer.row_counts == [5, 2, 2, 1]
    with pytest.raises(ValueError, match="C-contiguous"):
        classifier.predict_proba_positive(matrix[:, ::-1], max_rows_per_chunk=2)


def test_failed_finalization_leaves_source_unchanged_and_output_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    original_manifest = (bundle_dir / "manifest.json").read_bytes()
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)

    def fail_validation(_path: Path) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(production_bundle_module, "load_production_model", fail_validation)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=output_bundle,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.8",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert (bundle_dir / "manifest.json").read_bytes() == original_manifest
    assert not (bundle_dir / "incremental_linker").exists()
    assert not output_bundle.exists()


def test_pairwise_stage_publication_failure_leaves_target_absent_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    real_replace = production_bundle_module.os.replace
    failed = False

    def fail_publish_once(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if not failed and Path(destination) == output_bundle:
            failed = True
            raise OSError("injected pairwise publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(production_bundle_module.os, "replace", fail_publish_once)
    with pytest.raises(OSError, match="injected pairwise publication failure"):
        write_pairwise_production_bundle(source_clusterer, output_bundle, bundle_version="9.8")
    assert not output_bundle.exists()

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = write_pairwise_production_bundle(source_clusterer, output_bundle, bundle_version="9.8")
    assert summary.bundle_status == "pairwise_only"
    assert _load_pairwise_staging_model(output_bundle).production_model_bundle_status == "pairwise_only"


def test_pairwise_stage_publication_requires_new_output(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    write_pairwise_production_bundle(source_clusterer, output_bundle, bundle_version="9.8")
    original_manifest = (output_bundle / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_pairwise_production_bundle(
            source_clusterer,
            output_bundle,
            bundle_version="9.8",
        )
    assert (output_bundle / "manifest.json").read_bytes() == original_manifest


@pytest.mark.parametrize(
    "relative_path",
    (
        "reproducibility/pairwise_training_config.json",
        "reproducibility/pairwise_training_summary.json",
    ),
)
def test_pairwise_reproducibility_files_are_manifest_bound(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    relative_path: str,
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    output_dir = tmp_path / "production_model_v9.8"
    write_pairwise_production_bundle(
        source_clusterer,
        output_dir,
        bundle_version="9.8",
        pairwise_training_config={"training_seed": 7},
        pairwise_training_summary={"pair_count": 11},
    )
    _load_pairwise_staging_model(output_dir)

    path = output_dir / relative_path
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_pairwise_staging_model(output_dir)


def test_finalization_publishes_with_one_rename_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    real_replace = production_bundle_module.os.replace

    clusterer_config = json.loads((bundle_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
    for derived_field in ("best_params", "bundle_version", "pairwise", "source_model_version"):
        assert derived_field not in clusterer_config
    with pytest.raises(ValueError, match="Expected a complete"):
        load_production_model(bundle_dir)

    failed = False

    def fail_publication_once(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if not failed and Path(destination) == output_bundle:
            failed = True
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(production_bundle_module.os, "replace", fail_publication_once)
    with pytest.raises(OSError, match="injected publication failure"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=output_bundle,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.8",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert not output_bundle.exists()
    assert not list(tmp_path.glob(".production_model_v9.8.staging-*"))
    assert _load_pairwise_staging_model(bundle_dir).production_model_bundle_status == "pairwise_only"
    assert not (bundle_dir / "incremental_linker").exists()

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    assert summary.bundle_status == "complete"
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_bundle_version == "9.8"
    assert loaded.production_model_bundle_status == "complete"
    assert loaded.incremental_linker_artifact_dir is not None
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"
    assert loaded.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    restored = pickle.loads(pickle.dumps(loaded))
    assert restored.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    assert copy.deepcopy(loaded).incremental_linker_artifact is loaded.incremental_linker_artifact

    with pytest.raises(FileExistsError, match="requires a new directory"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=output_bundle,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.8",
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_pairwise_production_bundle(
            source_clusterer,
            bundle_dir,
            bundle_version="9.9",
            source_model_version="9.9",
        )


def test_finalization_defaults_pairwise_version_from_source_manifest(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, _ = synthetic_pairwise_bundle
    source_manifest_path = source_bundle / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["pairwise_model_version"] = "1.2"
    source_manifest_path.write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(source_bundle, linker_dir)
    output_bundle = tmp_path / "production_model_v8.8"

    finalize_production_bundle(
        pairwise_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="8.8",
    )

    output_manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert source_bundle.name == "production_model_v9.9"
    assert output_manifest["bundle_version"] == "8.8"
    assert output_manifest["pairwise_model_version"] == "1.2"


def test_finalization_does_not_copy_undeclared_reproducibility_files(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, _ = synthetic_pairwise_bundle
    stale_path = source_bundle / "reproducibility" / "stale-sensitive.json"
    stale_path.parent.mkdir()
    stale_path.write_text('{"secret": true}\n', encoding="utf-8")
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(source_bundle, linker_dir)
    output_bundle = tmp_path / "production_model_v9.8"

    finalize_production_bundle(
        pairwise_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
    )

    assert not (output_bundle / "reproducibility" / stale_path.name).exists()


def test_finalization_rejects_linker_bound_to_different_pairwise_bundle(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    metadata_path = linker_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pairwise_bundle_binding_digest"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pairwise_bundle_binding_digest does not match"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=output_bundle,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.8",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert not output_bundle.exists()
    assert _load_pairwise_staging_model(bundle_dir).production_model_bundle_status == "pairwise_only"


def test_finalization_rejects_target_different_from_linker_training_target(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    target_json.write_text('{"variant": "wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="target_spec_digest does not match target JSON"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=tmp_path / "production_model_v9.8",
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.8",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )


def test_complete_bundle_rejects_target_tampering_even_with_refreshed_manifest(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )
    bundled_target = output_bundle / "reproducibility" / "incremental_linker_training_target.json"
    bundled_target.write_text('{"variant": "tampered"}\n', encoding="utf-8")
    _refresh_manifest_checksum(output_bundle, bundled_target.relative_to(output_bundle).as_posix())

    with pytest.raises(ValueError, match="target_spec_digest does not match enclosing bundle target JSON"):
        load_production_model(output_bundle)


def test_complete_bundle_rejects_pairwise_binding_tampering_even_with_refreshed_manifest(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )
    metadata_path = output_bundle / "incremental_linker" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pairwise_bundle_binding_digest"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(output_bundle, metadata_path.relative_to(output_bundle).as_posix())

    with pytest.raises(ValueError, match="pairwise_bundle_binding_digest does not match enclosing bundle"):
        load_production_model(output_bundle)


def test_normal_load_hashes_each_declared_file_exactly_once(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.8",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    manifest_hashes: list[str] = []
    standalone_artifact_hashes: list[str] = []
    real_manifest_sha256 = production_model_module._sha256_file
    real_artifact_sha256 = incremental_artifact_module._sha256_file

    def counting_manifest_sha256(path: Path) -> str:
        manifest_hashes.append(Path(path).resolve().as_posix())
        return real_manifest_sha256(path)

    def counting_artifact_sha256(path: Path) -> str:
        standalone_artifact_hashes.append(Path(path).resolve().as_posix())
        return real_artifact_sha256(path)

    monkeypatch.setattr(production_model_module, "_sha256_file", counting_manifest_sha256)
    monkeypatch.setattr(incremental_artifact_module, "_sha256_file", counting_artifact_sha256)
    load_production_model(output_bundle)

    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    expected = sorted((output_bundle / relpath).resolve().as_posix() for relpath in manifest["sha256"])
    assert sorted(manifest_hashes) == expected
    assert standalone_artifact_hashes == []


def test_bundle_directory_version_must_match_explicit_version(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, source_clusterer = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(source_bundle, linker_dir)
    mismatched_output = tmp_path / "production_model_v9.8"

    with pytest.raises(ValueError, match="directory name and bundle_version disagree"):
        finalize_production_bundle(
            pairwise_bundle_dir=source_bundle,
            output_bundle_dir=mismatched_output,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.9",
        )
    assert not mismatched_output.exists()

    with pytest.raises(ValueError, match="directory name and bundle_version disagree"):
        write_pairwise_production_bundle(
            source_clusterer,
            mismatched_output,
            bundle_version="9.9",
        )
    assert not mismatched_output.exists()

    with pytest.raises(ValueError, match="bundle_version must be nonempty"):
        write_pairwise_production_bundle(
            source_clusterer,
            tmp_path / "unnamed-bundle",
            bundle_version="",
        )


def test_manifest_writer_rejects_directory_at_required_file_path(tmp_path: Path) -> None:
    files = production_bundle_module.production_manifest_files(
        incremental_linker_version=None,
        include_pairwise_reproducibility=False,
    )
    for relpath in files.values():
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    (tmp_path / "clusterer.json").unlink()
    (tmp_path / "clusterer.json").mkdir()

    with pytest.raises(FileNotFoundError, match="missing or not a regular file"):
        production_bundle_module.write_production_manifest(
            tmp_path,
            bundle_version="9.9",
            pairwise_model_version="9.9",
        )
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("bundle_version", ""),
        ("bundle_version", "   "),
        ("pairwise_model_version", ""),
        ("pairwise_model_version", "   "),
    ),
)
def test_manifest_writer_rejects_empty_model_versions_before_writing(
    tmp_path: Path,
    field_name: str,
    invalid_value: str,
) -> None:
    versions = {
        "bundle_version": "9.9",
        "pairwise_model_version": "9.9",
    }
    versions[field_name] = invalid_value

    with pytest.raises(ValueError, match=rf"{field_name} must be a nonempty string"):
        production_bundle_module.write_production_manifest(tmp_path, **versions)

    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("invalid_version", (False, "   "))
def test_manifest_writer_rejects_invalid_state_discriminator(tmp_path: Path, invalid_version: Any) -> None:
    with pytest.raises(ValueError, match="require a nonempty incremental_linker_version"):
        production_bundle_module.write_production_manifest(
            tmp_path,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            incremental_linker_version=invalid_version,
        )


def test_finalize_production_bundle_rejects_invalid_incremental_linker_artifact(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, _ = synthetic_pairwise_bundle
    corrupt_linker = tmp_path / "corrupt_incremental_linker"
    target_json = _write_synthetic_linker(source_bundle, corrupt_linker)
    metadata_path = corrupt_linker / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["booster_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="booster_sha256 mismatch"):
        finalize_production_bundle(
            pairwise_bundle_dir=source_bundle,
            output_bundle_dir=tmp_path / "production_model_v9.8",
            incremental_linker_artifact_dir=corrupt_linker,
            target_json=target_json,
            bundle_version="9.8",
            pairwise_model_version="9.9",
            incremental_linker_version="9.8",
        )


def test_production_model_config_choice_rejects_unknown_literal() -> None:
    with pytest.raises(ValueError, match="incremental_seed_score_mode"):
        _config_choice(
            {"incremental_seed_score_mode": "unsupported"},
            "incremental_seed_score_mode",
            allowed=frozenset({"mean", "min", "mean_min_hybrid"}),
        )
