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
from s2and.data import ANDData
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.model_pairwise import _validated_classifier_features
from s2and.production_bundle import (
    finalize_pairwise_eps,
    finalize_production_bundle,
    write_pairwise_production_bundle,
)
from s2and.production_model import (
    NativeLightGBMBinaryClassifier,
    _config_choice,
    _load_pairwise_staging_model,
    _require_featurizer_version_match,
    load_production_model,
    pairwise_bundle_binding,
)
from tests.helpers import tiny_name_counts_index
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    tiny_binary_booster,
    write_synthetic_pairwise_bundle,
)

_TEST_CANONICAL_ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "a" * 64,
    "orcid_prefix_counts_data_sha256": "b" * 64,
}
_TEST_EXPLICIT_ARTIFACT_HASHES = {
    **_TEST_CANONICAL_ARTIFACT_HASHES,
    "name_counts_manifest_sha256": "c" * 64,
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
    bundle_dir = tmp_path / "production_model_v9.9"
    source_clusterer = write_synthetic_pairwise_bundle(
        bundle_dir,
        artifact_hashes=_TEST_CANONICAL_ARTIFACT_HASHES,
        bundle_version="9.9",
    )
    return bundle_dir, source_clusterer


def _write_synthetic_linker(
    pairwise_bundle: Path,
    linker_dir: Path,
    *,
    expected_artifact_hashes: dict[str, str] | None = None,
) -> Path:
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
        pairwise_bundle_binding=pairwise_bundle_binding(
            pairwise_bundle,
            expected_artifact_hashes=expected_artifact_hashes,
        ),
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


def test_clusterer_config_contains_only_inference_state(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config = json.loads((bundle_dir / "clusterer.json").read_text(encoding="utf-8"))

    assert set(config) == {
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
    assert set(config["cluster_model"]) == {"eps", "linkage"}


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
    tiny_binary_booster(classifier.n_features_in_ + 1, seed=103).booster_.save_model(str(replacement_path))
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


def test_canonical_artifact_hash_contract_rejects_missing_malformed_or_mismatched_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: dict(_TEST_CANONICAL_ARTIFACT_HASHES),
    )
    invalid_values = (
        ("name_tuples_data_sha256", None),
        ("name_tuples_data_sha256", "f" * 63),
        ("orcid_prefix_counts_data_sha256", "F" * 64),
        ("orcid_prefix_counts_data_sha256", "c" * 64),
    )
    for field, invalid in invalid_values:
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


def test_explicit_artifact_authority_requires_exact_fields() -> None:
    partial = dict(_TEST_EXPLICIT_ARTIFACT_HASHES)
    partial.pop("name_counts_manifest_sha256")
    authorities = (
        (partial, r"field mismatch: missing=\['name_counts_manifest_sha256'\]"),
        (
            {
                **_TEST_EXPLICIT_ARTIFACT_HASHES,
                "unreviewed_sha256": "e" * 64,
            },
            r"extra=\['unreviewed_sha256'\]",
        ),
    )
    for authority, message in authorities:
        with pytest.raises(ValueError, match=message):
            production_model_module.require_expected_artifact_hashes(
                _TEST_EXPLICIT_ARTIFACT_HASHES,
                authority,
                context="test feature_contract",
            )


def test_retired_orcid_manifest_hash_is_serialized_data_not_explicit_authority() -> None:
    retired_field = "orcid_prefix_counts_manifest_sha256"
    feature_contract = {
        **_TEST_EXPLICIT_ARTIFACT_HASHES,
        retired_field: "d" * 64,
    }

    production_model_module.require_expected_artifact_hashes(
        feature_contract,
        _TEST_EXPLICIT_ARTIFACT_HASHES,
        context="test feature_contract",
    )

    old_four_field_authority = {
        **_TEST_EXPLICIT_ARTIFACT_HASHES,
        retired_field: "d" * 64,
    }
    with pytest.raises(ValueError, match=rf"extra=\['{retired_field}'\]"):
        production_model_module.require_expected_artifact_hashes(
            feature_contract,
            old_four_field_authority,
            context="test feature_contract",
        )


def test_bundle_export_uses_training_artifact_hashes_without_loading_packaged_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    explicit_hashes = {
        "name_tuples_data_sha256": "e" * 64,
        "orcid_prefix_counts_data_sha256": "f" * 64,
        "name_counts_manifest_sha256": "1" * 64,
    }
    source_clusterer.feature_contract.update(explicit_hashes)
    package_loads = 0

    def packaged_hashes() -> dict[str, str]:
        nonlocal package_loads
        package_loads += 1
        return dict(_TEST_CANONICAL_ARTIFACT_HASHES)

    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", packaged_hashes)
    output_dir = tmp_path / "production_model_v10.0"

    write_pairwise_production_bundle(
        source_clusterer,
        output_dir,
        bundle_version="10.0",
        pairwise_training_config={"input_artifact_hashes": explicit_hashes},
        pairwise_training_summary={"pair_count": 11},
    )

    assert package_loads == 0
    with pytest.raises(ValueError, match="does not match"):
        _load_pairwise_staging_model(output_dir)
    assert package_loads == 1


def test_bundle_export_rejects_training_artifact_hash_mismatch(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    expected_hashes = dict(_TEST_EXPLICIT_ARTIFACT_HASHES)
    expected_hashes["name_tuples_data_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="explicit artifact authority"):
        write_pairwise_production_bundle(
            source_clusterer,
            tmp_path / "production_model_v10.0",
            bundle_version="10.0",
            pairwise_training_config={"input_artifact_hashes": expected_hashes},
            pairwise_training_summary={"pair_count": 11},
        )


def test_finalization_uses_explicit_artifact_authority_instead_of_package_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.update(_TEST_EXPLICIT_ARTIFACT_HASHES)
    pairwise_bundle = tmp_path / "production_model_v10.0"
    write_pairwise_production_bundle(
        source_clusterer,
        pairwise_bundle,
        bundle_version="10.0",
        pairwise_training_config={"input_artifact_hashes": dict(_TEST_EXPLICIT_ARTIFACT_HASHES)},
        pairwise_training_summary={"pair_count": 11},
    )
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(
        pairwise_bundle,
        linker_dir,
        expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES,
    )
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: {
            "name_tuples_data_sha256": "c" * 64,
            "orcid_prefix_counts_data_sha256": "d" * 64,
        },
    )

    with pytest.raises(ValueError, match="canonical artifacts installed with this package"):
        finalize_production_bundle(
            pairwise_bundle_dir=pairwise_bundle,
            output_bundle_dir=tmp_path / "default-authority",
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
        )

    output_bundle = tmp_path / "explicit-authority"
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES,
    )

    assert (
        load_production_model(
            output_bundle,
            expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES,
        ).production_model_bundle_status
        == "complete"
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


def test_canonical_artifact_hashes_feed_pairwise_bundle_binding(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    original_binding = pairwise_bundle_binding(bundle_dir)
    config_path = bundle_dir / "clusterer.json"
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    for field in _TEST_CANONICAL_ARTIFACT_HASHES:
        changed_hashes = dict(_TEST_CANONICAL_ARTIFACT_HASHES)
        changed_hashes[field] = "c" * 64
        monkeypatch.setattr(
            production_model_module,
            "canonical_artifact_hashes",
            lambda hashes=changed_hashes: dict(hashes),
        )
        config = copy.deepcopy(original_config)
        config["feature_contract"][field] = changed_hashes[field]
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _refresh_manifest_checksum(bundle_dir, "clusterer.json")
        changed_binding = pairwise_bundle_binding(bundle_dir)
        assert changed_binding["ordered_feature_contract_digest"] != original_binding["ordered_feature_contract_digest"]


def test_manifest_requires_nonempty_string_model_versions(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for field_name in ("bundle_version", "pairwise_model_version"):
        for invalid_value in (None, "   "):
            manifest = copy.deepcopy(original_manifest)
            manifest[field_name] = invalid_value
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with pytest.raises(ValueError, match=rf"{field_name} must be a nonempty string"):
                production_model_module._validate_manifest(bundle_dir)


def test_manifest_requires_valid_incremental_linker_version(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    pairwise_bundle, _ = synthetic_pairwise_bundle
    manifest_path = pairwise_bundle / "manifest.json"
    original_pairwise_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_pairwise_manifest = copy.deepcopy(original_pairwise_manifest)
    invalid_pairwise_manifest["incremental_linker_version"] = False
    manifest_path.write_text(json.dumps(invalid_pairwise_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incremental_linker_version must be null or a nonempty string"):
        production_model_module._validate_manifest(pairwise_bundle)
    manifest_path.write_text(json.dumps(original_pairwise_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    complete_bundle = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(pairwise_bundle, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise_bundle,
        output_bundle_dir=complete_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
    )
    complete_manifest_path = complete_bundle / "manifest.json"
    original_manifest = json.loads(complete_manifest_path.read_text(encoding="utf-8"))

    for invalid_value, message in ((None, "checksum coverage mismatch"), ("   ", "null or a nonempty string")):
        manifest = copy.deepcopy(original_manifest)
        manifest["incremental_linker_version"] = invalid_value
        complete_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def test_clusterer_config_rejects_nonfinite_unknown_and_contradictory_values(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config_path = bundle_dir / "clusterer.json"
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    invalid_configs = (
        (lambda payload: payload["cluster_model"].update({"eps": float("nan")}), "must be finite"),
        (lambda payload: payload["cluster_model"].update({"eps": True}), "must be numeric"),
        (lambda payload: payload.update({"incremental_mean_min_hybrid_weight": False}), "must be numeric"),
        (lambda payload: payload["cluster_model"].update({"family": "Other"}), "exact FastCluster"),
        (lambda payload: payload.update({"unknown_runtime_field": 1}), "field mismatch"),
        (
            lambda payload: payload.update({"incremental_mean_min_hybrid_weight": 2.0}),
            "hybrid_weight must be in",
        ),
    )
    for mutate, message in invalid_configs:
        config = copy.deepcopy(original_config)
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


def test_eps_finalization_failures_clean_stage_and_are_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "eps" / "production_model_v9.9"
    source_manifest = source_bundle / "manifest.json"
    expected_manifest_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    source_config = json.loads((source_bundle / "clusterer.json").read_text(encoding="utf-8"))
    expected_old_eps = float(source_config["cluster_model"]["eps"])
    new_eps = 0.37
    staging_pattern = f".{output_bundle.name}.staging-*"

    real_load = production_bundle_module._load_pairwise_staging_model

    def fail_staged_validation(bundle_dir: Path, **kwargs: Any) -> Any:
        if Path(bundle_dir) != source_bundle:
            raise ValueError("injected EPS validation failure")
        return real_load(bundle_dir, **kwargs)

    monkeypatch.setattr(production_bundle_module, "_load_pairwise_staging_model", fail_staged_validation)
    with pytest.raises(ValueError, match="injected EPS validation failure"):
        finalize_pairwise_eps(
            source_bundle_dir=source_bundle,
            output_bundle_dir=output_bundle,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_old_eps=expected_old_eps,
            new_eps=new_eps,
        )
    assert not output_bundle.exists()
    assert not list(output_bundle.parent.glob(staging_pattern))

    monkeypatch.setattr(production_bundle_module, "_load_pairwise_staging_model", real_load)
    real_replace = production_bundle_module.os.replace

    def fail_publication(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == output_bundle:
            raise OSError("injected EPS publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(production_bundle_module.os, "replace", fail_publication)
    with pytest.raises(OSError, match="injected EPS publication failure"):
        finalize_pairwise_eps(
            source_bundle_dir=source_bundle,
            output_bundle_dir=output_bundle,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_old_eps=expected_old_eps,
            new_eps=new_eps,
        )
    assert not output_bundle.exists()
    assert not list(output_bundle.parent.glob(staging_pattern))

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_old_eps=expected_old_eps,
        new_eps=new_eps,
    )

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert summary.bundle_dir == output_bundle
    assert summary.bundle_version == "9.9"
    assert summary.bundle_status == "pairwise_only"
    assert summary.manifest_path == output_bundle / "manifest.json"
    assert summary.files == tuple(sorted(manifest["sha256"]))
    assert not list(output_bundle.parent.glob(staging_pattern))
    published_config = json.loads((output_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert published_config["cluster_model"]["eps"] == new_eps


def test_pairwise_reproducibility_files_are_manifest_bound(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.update(_TEST_EXPLICIT_ARTIFACT_HASHES)
    output_dir = tmp_path / "production_model_v9.8"
    write_pairwise_production_bundle(
        source_clusterer,
        output_dir,
        bundle_version="9.8",
        pairwise_training_config={
            "training_seed": 7,
            "input_artifact_hashes": dict(_TEST_EXPLICIT_ARTIFACT_HASHES),
        },
        pairwise_training_summary={"pair_count": 11},
    )
    _load_pairwise_staging_model(output_dir)

    for relative_path in (
        "reproducibility/pairwise_training_config.json",
        "reproducibility/pairwise_training_summary.json",
    ):
        path = output_dir / relative_path
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="checksum mismatch"):
            _load_pairwise_staging_model(output_dir)
        path.write_bytes(original)


def test_finalization_publishes_with_one_rename_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    output_bundle = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    original_manifest = (bundle_dir / "manifest.json").read_bytes()
    real_replace = production_bundle_module.os.replace

    clusterer_config = json.loads((bundle_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
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
        )

    assert not output_bundle.exists()
    assert not list(output_bundle.parent.glob(".production_model_v9.9.staging-*"))
    assert (bundle_dir / "manifest.json").read_bytes() == original_manifest
    assert _load_pairwise_staging_model(bundle_dir).production_model_bundle_status == "pairwise_only"
    assert not (bundle_dir / "incremental_linker").exists()

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
    )

    assert summary.bundle_status == "complete"
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_bundle_version == "9.9"
    assert loaded.production_model_bundle_status == "complete"
    assert loaded.incremental_linker_artifact_dir is not None
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"
    assert loaded.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    restored = pickle.loads(pickle.dumps(loaded))
    assert restored.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    assert copy.deepcopy(loaded).incremental_linker_artifact is loaded.incremental_linker_artifact
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bundle_version"] == "9.9"
    assert manifest["pairwise_model_version"] == "9.9"
    assert manifest["incremental_linker_version"] == "9.9"

    with pytest.raises(FileExistsError, match="requires a new directory"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=output_bundle,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
        )
    with pytest.raises(FileExistsError, match="already exists"):
        write_pairwise_production_bundle(
            source_clusterer,
            bundle_dir,
            bundle_version="9.9",
        )


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
    output_bundle = tmp_path / "complete" / "production_model_v9.9"

    finalize_production_bundle(
        pairwise_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
    )

    assert not (output_bundle / "reproducibility" / stale_path.name).exists()


def test_finalization_rejects_linker_bound_to_different_pairwise_bundle(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "complete" / "production_model_v9.9"
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
            output_bundle_dir=tmp_path / "complete" / "production_model_v9.9",
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
        )


def test_complete_bundle_rejects_target_tampering_even_with_refreshed_manifest(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    output_bundle = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
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
    output_bundle = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
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
    output_bundle = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
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


def test_bundle_directory_version_must_match_derived_version(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    source_bundle, source_clusterer = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(source_bundle, linker_dir)
    mismatched_output = tmp_path / "production_model_v9.8"

    with pytest.raises(ValueError, match="bundle version disagrees with pairwise manifest"):
        finalize_production_bundle(
            pairwise_bundle_dir=source_bundle,
            output_bundle_dir=mismatched_output,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
        )
    assert not mismatched_output.exists()

    with pytest.raises(ValueError, match="directory name and bundle_version disagree"):
        write_pairwise_production_bundle(
            source_clusterer,
            mismatched_output,
            bundle_version="9.9",
        )
    assert not mismatched_output.exists()


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


def test_manifest_writer_rejects_invalid_versions_before_writing(tmp_path: Path) -> None:
    for field_name in ("bundle_version", "pairwise_model_version"):
        for invalid_value in ("", "   "):
            versions = {
                "bundle_version": "9.9",
                "pairwise_model_version": "9.9",
            }
            versions[field_name] = invalid_value
            with pytest.raises(ValueError, match=rf"{field_name} must be a nonempty string"):
                production_bundle_module.write_production_manifest(tmp_path, **versions)

    for invalid_version in (False, "   "):
        with pytest.raises(ValueError, match="require a nonempty incremental_linker_version"):
            production_bundle_module.write_production_manifest(
                tmp_path,
                bundle_version="9.9",
                pairwise_model_version="9.9",
                incremental_linker_version=invalid_version,
            )

    assert not (tmp_path / "manifest.json").exists()


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
            output_bundle_dir=tmp_path / "complete" / "production_model_v9.9",
            incremental_linker_artifact_dir=corrupt_linker,
            target_json=target_json,
        )


def test_production_model_config_choice_rejects_unknown_literal() -> None:
    with pytest.raises(ValueError, match="incremental_seed_score_mode"):
        _config_choice(
            {"incremental_seed_score_mode": "unsupported"},
            "incremental_seed_score_mode",
            allowed=frozenset({"mean", "min", "mean_min_hybrid"}),
        )
