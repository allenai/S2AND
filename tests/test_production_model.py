from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.incremental_linking.artifact as incremental_artifact_module
import s2and.production_bundle as production_bundle_module
import s2and.production_model as production_model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.data import ANDData
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.model_pairwise import _validated_classifier_features
from s2and.production_bundle import (
    finalize_pairwise_eps,
    finalize_production_bundle,
)
from s2and.production_bundle import (
    write_pairwise_production_bundle as _write_pairwise_production_bundle,
)
from s2and.production_bundle_contract import (
    CALIBRATED_EPS_CALIBRATION,
    LEGACY_PAIRWISE_FEATURE_CONTRACT_VERSION,
    PAIRWISE_FEATURE_CONTRACT_VERSION,
    PAIRWISE_ONLY_BUNDLE_KIND,
    PENDING_EPS_CALIBRATION,
    PRODUCTION_MODEL_FORMAT_VERSION,
    PRODUCTION_MODEL_KIND,
    EpsCalibration,
)
from s2and.production_model import (
    NativeLightGBMBinaryClassifier,
    load_production_model,
    pairwise_bundle_binding,
)
from s2and.production_model import (
    _load_pairwise_staging_model as _load_pairwise_staging_model_with_state,
)
from tests.helpers import tiny_name_counts_index
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    tiny_binary_booster,
    write_synthetic_pairwise_bundle,
)

_TEST_CANONICAL_ARTIFACT_HASHES = production_model_module.canonical_artifact_hashes()
_V132_ORCID_PREFIX_COUNTS_DATA_SHA256 = "fcb4cb7bdf30f07366b946b9fdb1bb0a807bdb00f7dda521575daea6b2b9d8ce"
_TEST_EXPLICIT_ARTIFACT_HASHES = {
    **_TEST_CANONICAL_ARTIFACT_HASHES,
    "name_counts_manifest_sha256": "c" * 64,
}


def write_pairwise_production_bundle(
    clusterer: Clusterer,
    bundle_dir: Path,
    *,
    release_version: str,
    eps_calibration: EpsCalibration = CALIBRATED_EPS_CALIBRATION,
    **kwargs: Any,
) -> Any:
    """Write the calibrated pairwise fixture used by most bundle tests."""

    return _write_pairwise_production_bundle(
        clusterer,
        bundle_dir,
        release_version=release_version,
        eps_calibration=eps_calibration,
        **kwargs,
    )


def _load_pairwise_staging_model(
    bundle_dir: Path,
    *,
    expected_eps_calibration: EpsCalibration = CALIBRATED_EPS_CALIBRATION,
    **kwargs: Any,
) -> Clusterer:
    """Load the calibrated pairwise fixture used by most bundle tests."""

    return _load_pairwise_staging_model_with_state(
        bundle_dir,
        expected_eps_calibration=expected_eps_calibration,
        **kwargs,
    )


@pytest.fixture(scope="module")
def pairwise_bundle_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Clusterer]:
    bundle_dir = tmp_path_factory.mktemp("pairwise_bundle_template") / "production_model_v9.9"
    source_clusterer = write_synthetic_pairwise_bundle(
        bundle_dir,
        artifact_hashes=_TEST_CANONICAL_ARTIFACT_HASHES,
        release_version="9.9",
    )
    return bundle_dir, source_clusterer


@pytest.fixture
def synthetic_pairwise_bundle(
    tmp_path: Path, pairwise_bundle_template: tuple[Path, Clusterer]
) -> tuple[Path, Clusterer]:
    template_path, clusterer = pairwise_bundle_template
    bundle_dir = tmp_path / template_path.name
    shutil.copytree(template_path, bundle_dir)
    return bundle_dir, copy.deepcopy(clusterer)


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


def test_native_classifier_replacement_updates_state_atomically(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    classifier = _load_pairwise_staging_model(bundle_dir).classifier
    wrong_width_path = tmp_path / "wrong_width.lgb"
    tiny_binary_booster(classifier.n_features_in_ + 1, seed=103).booster_.save_model(str(wrong_width_path))
    original_params = classifier.get_params()
    original_scorer = classifier._rust_scorer
    original_booster = classifier.booster_
    original_fingerprint = classifier.cache_fingerprint()

    with pytest.raises(ValueError, match="Invalid parameter"):
        classifier.set_params(learning_rate=0.1)
    with pytest.raises(ValueError, match="feature-count mismatch"):
        NativeLightGBMBinaryClassifier(wrong_width_path, n_features=classifier.n_features_in_)
    with pytest.raises(ValueError, match="feature-count mismatch"):
        classifier.set_params(
            model_path=wrong_width_path,
            n_features=classifier.n_features_in_,
        )

    assert classifier.get_params() == original_params
    assert classifier._scorer is original_scorer
    assert classifier._lazy_booster is original_booster
    assert classifier.cache_fingerprint() == original_fingerprint

    with pytest.raises(ValueError, match="must not be zero"):
        classifier.set_params(model_path=wrong_width_path, n_jobs=0)

    assert classifier.get_params() == original_params
    assert classifier._scorer is original_scorer
    assert classifier._lazy_booster is original_booster
    assert classifier.cache_fingerprint() == original_fingerprint

    features = np.random.default_rng(104).normal(size=(16, classifier.n_features_in_))
    previous_predictions = classifier.predict_proba_positive(features)
    replacement = tiny_binary_booster(classifier.n_features_in_, seed=103)
    replacement_path = tmp_path / "replacement.lgb"
    replacement.booster_.save_model(str(replacement_path))
    expected = replacement.booster_.predict(features)
    assert not np.allclose(previous_predictions, expected)

    assert classifier.set_params(model_path=replacement_path) is classifier

    np.testing.assert_allclose(classifier.predict_proba_positive(features), expected, rtol=1e-12, atol=1e-12)
    assert classifier.cache_fingerprint() != original_fingerprint
    assert classifier.cache_fingerprint()[1] == hashlib.sha256(replacement_path.read_bytes()).hexdigest()
    assert classifier.booster_ is not original_booster
    np.testing.assert_allclose(classifier.booster_.predict(features), expected, rtol=1e-12, atol=1e-12)

    classifier.model_path = str(tmp_path / "missing_model.txt")

    copied = copy.deepcopy(classifier)

    np.testing.assert_allclose(copied.predict_proba_positive(features), expected)
    assert copied.model_path == classifier.model_path
    assert copied._scorer is classifier._scorer
    copied.n_jobs = 3
    assert classifier.n_jobs != copied.n_jobs


def test_complete_bundle_native_inference_does_not_load_training_libraries(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    """Load, score, and cluster with a complete bundle without optimizer or training-library imports.

    scikit-learn, pandas, and LightGBM add about 1.5 s to each new prediction process.
    """
    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.update(_TEST_EXPLICIT_ARTIFACT_HASHES)
    pairwise_bundle = tmp_path / "explicit-pairwise"
    write_pairwise_production_bundle(
        source_clusterer,
        pairwise_bundle,
        release_version="9.9",
        pairwise_training_config={"input_artifact_hashes": dict(_TEST_EXPLICIT_ARTIFACT_HASHES)},
        pairwise_training_summary={"pair_count": 11},
    )
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(
        pairwise_bundle, linker_dir, expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES
    )
    output_bundle = tmp_path / "complete"
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES,
    )
    rng = np.random.default_rng(921)
    assert source_clusterer.nameless_featurizer_info is not None
    assert source_clusterer.nameless_classifier is not None
    main_features = rng.normal(size=(8, len(_selected_feature_indices(source_clusterer.featurizer_info))))
    nameless_features = rng.normal(size=(8, len(_selected_feature_indices(source_clusterer.nameless_featurizer_info))))
    fixture_path = tmp_path / "predictions.npz"
    np.savez(
        fixture_path,
        main_features=main_features,
        nameless_features=nameless_features,
        main_expected=source_clusterer.classifier.predict_proba(
            _validated_classifier_features(source_clusterer.classifier, main_features)
        ),
        nameless_expected=source_clusterer.nameless_classifier.predict_proba(
            _validated_classifier_features(source_clusterer.nameless_classifier, nameless_features)
        ),
    )
    script = textwrap.dedent(
        f"""
        import sys
        import numpy as np
        from s2and import runtime

        load_native = runtime.load_s2and_rust_extension

        def fail():
            raise AssertionError("Python model import must not load the Rust extension")

        runtime.load_s2and_rust_extension = fail
        import s2and.model
        runtime.load_s2and_rust_extension = load_native
        from s2and.model_pairwise import clone_cluster_model
        from s2and.production_model import load_production_model

        forbidden = [
            "seaborn", "matplotlib.pyplot", "hyperopt", "s2and.eval", "sklearn", "pandas", "lightgbm"
        ]
        loaded = [name for name in forbidden if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies: {{loaded}}"

        clusterer = load_production_model(
            {str(output_bundle)!r}, expected_artifact_hashes={_TEST_EXPLICIT_ARTIFACT_HASHES!r}
        )
        assert clusterer.incremental_linker_artifact is not None
        assert clusterer.search_space is None
        assert "hyperopt" not in sys.modules
        with np.load({str(fixture_path)!r}) as fixture:
            for name, classifier in [("main", clusterer.classifier), ("nameless", clusterer.nameless_classifier)]:
                np.testing.assert_allclose(
                    classifier.predict_proba(fixture[name + "_features"]),
                    fixture[name + "_expected"], rtol=1e-10, atol=1e-10,
                )
        cluster_model = clone_cluster_model(clusterer.cluster_model)
        labels = cluster_model.fit(np.asarray([0.1, 0.9, 0.9], dtype=np.float64)).labels_
        assert labels[0] == labels[1] != labels[2]
        assert clusterer.search_space is None
        loaded = [name for name in forbidden if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies after inference: {{loaded}}"
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_pairwise_bundle_preserves_predictions_and_runtime_settings(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    (bundle_dir / "pairwise" / "stale.lgb").write_text("ignored", encoding="utf-8")
    native_clusterer = _load_pairwise_staging_model(bundle_dir)

    assert type(native_clusterer.cluster_model) is type(source_clusterer.cluster_model)
    assert native_clusterer.cluster_model.linkage == source_clusterer.cluster_model.linkage
    assert native_clusterer.cluster_model.eps == source_clusterer.cluster_model.eps
    assert native_clusterer.featurizer_info.features_to_use == source_clusterer.featurizer_info.features_to_use
    assert native_clusterer.nameless_featurizer_info is not None
    assert source_clusterer.nameless_featurizer_info is not None
    assert (
        native_clusterer.nameless_featurizer_info.features_to_use
        == source_clusterer.nameless_featurizer_info.features_to_use
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
    expected_clusterer = _prepare_prediction_clusterer(copy.deepcopy(source_clusterer))
    _prepare_prediction_clusterer(native_clusterer)
    assert _predict_dummy_block(native_clusterer, batching_threshold=None) == _predict_dummy_block(
        expected_clusterer, batching_threshold=None
    )

    assert isinstance(native_clusterer.classifier, NativeLightGBMBinaryClassifier)
    assert isinstance(native_clusterer.nameless_classifier, NativeLightGBMBinaryClassifier)
    assert native_clusterer.incremental_linker_artifact_dir is None
    assert native_clusterer.production_model_release_version == "9.9"

    native_clusterer.n_jobs = 2
    native_clusterer.cluster_model.eps = 0.5

    assert native_clusterer.n_jobs == 2
    assert native_clusterer.classifier.n_jobs == 2
    assert native_clusterer.nameless_classifier.n_jobs == 2
    assert native_clusterer.cluster_model.eps == 0.5
    assert {
        field: native_clusterer.feature_contract[field] for field in _TEST_CANONICAL_ARTIFACT_HASHES
    } == _TEST_CANONICAL_ARTIFACT_HASHES


def test_canonical_artifact_hash_contract_rejects_missing_malformed_or_mismatched_values() -> None:
    invalid_values = (
        ("name_tuples_data_sha256", None, "name_tuples_data_sha256"),
        ("name_tuples_data_sha256", "f" * 63, "name_tuples_data_sha256"),
        ("orcid_prefix_counts_data_sha256", "F" * 64, "orcid_prefix_counts_data_sha256"),
        ("orcid_prefix_counts_data_sha256", "c" * 64, "Unknown orcid_prefix_counts_data_sha256"),
    )
    for field, invalid, message in invalid_values:
        feature_contract = dict(_TEST_CANONICAL_ARTIFACT_HASHES)
        if invalid is None:
            feature_contract.pop(field)
        else:
            feature_contract[field] = invalid
        with pytest.raises(ValueError, match=message):
            production_model_module.require_canonical_artifact_hashes(
                feature_contract,
                context="test feature_contract",
            )
    with pytest.raises(ValueError, match=r"extra=\['retired_sha256'\]"):
        production_model_module.require_canonical_artifact_hashes(
            {**_TEST_CANONICAL_ARTIFACT_HASHES, "retired_sha256": "f" * 64},
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
    with pytest.raises(ValueError, match=r"artifact hash field mismatch.*extra=\['retired_sha256'\]"):
        production_model_module.require_expected_artifact_hashes(
            {**_TEST_EXPLICIT_ARTIFACT_HASHES, "retired_sha256": "f" * 64},
            _TEST_EXPLICIT_ARTIFACT_HASHES,
            context="test feature_contract",
        )


def test_bundle_export_uses_explicit_name_tuple_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    explicit_hashes = {
        "name_tuples_data_sha256": "e" * 64,
        "orcid_prefix_counts_data_sha256": _V132_ORCID_PREFIX_COUNTS_DATA_SHA256,
        "name_counts_manifest_sha256": "1" * 64,
    }
    source_clusterer.feature_contract.update(explicit_hashes)
    package_loads = 0

    load_name_tuples = production_model_module.load_packaged_name_tuple_artifact

    def counted_load():
        nonlocal package_loads
        package_loads += 1
        return load_name_tuples()

    monkeypatch.setattr(production_model_module, "load_packaged_name_tuple_artifact", counted_load)
    output_dir = tmp_path / "production_model_v10.0"

    write_pairwise_production_bundle(
        source_clusterer,
        output_dir,
        release_version="10.0",
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
    source_clusterer.feature_contract.update(_TEST_EXPLICIT_ARTIFACT_HASHES)
    expected_hashes = dict(_TEST_EXPLICIT_ARTIFACT_HASHES)
    expected_hashes["name_tuples_data_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="explicit artifact authority"):
        write_pairwise_production_bundle(
            source_clusterer,
            tmp_path / "production_model_v10.0",
            release_version="10.0",
            pairwise_training_config={"input_artifact_hashes": expected_hashes},
            pairwise_training_summary={"pair_count": 11},
        )


def test_finalization_uses_explicit_artifact_authority_instead_of_package_defaults(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    explicit_hashes = {**_TEST_EXPLICIT_ARTIFACT_HASHES, "name_tuples_data_sha256": "e" * 64}
    source_clusterer.feature_contract.update(explicit_hashes)
    pairwise_bundle = tmp_path / "production_model_v10.0"
    write_pairwise_production_bundle(
        source_clusterer,
        pairwise_bundle,
        release_version="10.0",
        pairwise_training_config={"input_artifact_hashes": explicit_hashes},
        pairwise_training_summary={"pair_count": 11},
    )
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(
        pairwise_bundle,
        linker_dir,
        expected_artifact_hashes=explicit_hashes,
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
        expected_artifact_hashes=explicit_hashes,
    )

    loaded = load_production_model(
        output_bundle,
        expected_artifact_hashes=explicit_hashes,
    )
    assert loaded.incremental_linker_artifact is not None


@pytest.mark.parametrize("field", sorted(_TEST_CANONICAL_ARTIFACT_HASHES))
def test_canonical_artifact_hashes_feed_pairwise_bundle_binding(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    original_binding = pairwise_bundle_binding(bundle_dir)
    assert set(original_binding) == {
        "main_booster_sha256",
        "nameless_booster_sha256",
        "ordered_feature_contract_digest",
    }
    config_path = bundle_dir / "clusterer.json"
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "name_tuples_data_sha256":
        changed_hash = "c" * 64
        artifact = replace(production_model_module.load_packaged_name_tuple_artifact(), data_sha256=changed_hash)
        monkeypatch.setattr(production_model_module, "load_packaged_name_tuple_artifact", lambda: artifact)
    else:
        changed_hash = _V132_ORCID_PREFIX_COUNTS_DATA_SHA256
    original_config["feature_contract"][field] = changed_hash
    config_path.write_text(json.dumps(original_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")
    changed_binding = pairwise_bundle_binding(bundle_dir)
    assert changed_binding["ordered_feature_contract_digest"] != original_binding["ordered_feature_contract_digest"]


def test_pairwise_manifest_enforces_identity_and_lifecycle(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(original_manifest) == {
        "eps_calibration",
        "feature_contract_version",
        "feature_names",
        "format_version",
        "kind",
        "release_version",
        "sha256",
    }
    assert original_manifest["kind"] == PRODUCTION_MODEL_KIND
    assert original_manifest["format_version"] == PRODUCTION_MODEL_FORMAT_VERSION
    assert original_manifest["feature_contract_version"] == PAIRWISE_FEATURE_CONTRACT_VERSION
    assert original_manifest["release_version"] == "9.9"

    incomplete = copy.deepcopy(original_manifest)
    del incomplete["feature_contract_version"]
    invalid = (
        ({**original_manifest, "release_version": " "}, "release_version"),
        ({**original_manifest, "kind": "not-a-model"}, "bundle kind"),
        ({**original_manifest, "eps_calibration": None}, "Unsupported production bundle lifecycle state"),
        (incomplete, "manifest is missing fields"),
    )
    for manifest, message in invalid:
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _load_pairwise_staging_model(bundle_dir)

    # Accept a retired field from an older release.
    manifest = {**original_manifest, "generated_by_runtime": {"retired": True}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _load_pairwise_staging_model(bundle_dir)

    manifest = copy.deepcopy(original_manifest)
    del manifest["sha256"]["pairwise/main.lgb"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum inventory"):
        _load_pairwise_staging_model(bundle_dir)

    # Reject version changes before opening any booster.
    (bundle_dir / "pairwise" / "main.lgb").unlink()
    for field, value in (("format_version", True), ("feature_contract_version", 3)):
        manifest = {**original_manifest, field: value}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            _load_pairwise_staging_model(bundle_dir)


def test_pairwise_model_rejects_changed_column_order_with_equal_width(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    columns = manifest["feature_names"]["main"]
    columns[0], columns[1] = columns[1], columns[0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_names must match the runtime ordered feature names"):
        _load_pairwise_staging_model(bundle_dir)
    with pytest.raises(ValueError, match="feature_names must match the runtime ordered feature names"):
        pairwise_bundle_binding(bundle_dir)


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
        # The close-pair fields come as a pair, with the bonus on.
        (lambda payload: payload["cluster_model"].update({"close_pair_bonus": 8.0}), "exact FastCluster"),
        (
            lambda payload: payload["cluster_model"].update({"close_pair_bonus": 0.0, "close_pair_distance": 0.05}),
            "close-pair bonus needs",
        ),
        (
            lambda payload: payload["cluster_model"].update({"close_pair_bonus": 8.0, "close_pair_distance": 0.0}),
            "close-pair bonus needs",
        ),
        (
            lambda payload: payload["cluster_model"].update({"close_pair_bonus": 1e12, "close_pair_distance": 0.05}),
            "close-pair bonus needs",
        ),
        (
            lambda payload: payload["cluster_model"].update(
                {"linkage": "single", "close_pair_bonus": 8.0, "close_pair_distance": 0.05}
            ),
            "close-pair bonus needs",
        ),
        (lambda payload: payload.update({"unknown_runtime_field": 1}), "field mismatch"),
        (lambda payload: payload.update({"incremental_seed_score_mode": "unsupported"}), "incremental_seed_score_mode"),
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


def test_pairwise_fixture_preserves_edge_values_and_enforces_tolerance(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    for model_name in ("main", "nameless"):
        fixture_path = bundle_dir / "pairwise" / f"{model_name}_prediction_fixture.json"
        fixture = json.loads(
            fixture_path.read_text(encoding="utf-8"),
            parse_constant=lambda value, name=model_name: pytest.fail(
                f"{name} fixture contains nonstandard JSON value {value}"
            ),
        )
        features = np.asarray(fixture["features"], dtype=np.float64)
        assert features.shape[0] == 10, model_name
        assert np.isfinite(features[:8]).all(), model_name
        assert fixture["features"][-2] == [None] * features.shape[1], model_name
        assert np.isnan(features[-2]).all(), model_name
        np.testing.assert_array_equal(features[-1], np.zeros(features.shape[1]), err_msg=model_name)
        assert fixture["feature_source"] == "numpy_default_rng_normal_with_all_missing_and_zero_rows", model_name

        classifier = NativeLightGBMBinaryClassifier(bundle_dir / "pairwise" / f"{model_name}.lgb", n_jobs=1)
        production_model_module._validate_pairwise_fixture(classifier, fixture_path)

    fixture_path = bundle_dir / "pairwise" / "main_prediction_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture["expected_probabilities"] = np.ones_like(fixture["expected_probabilities"]).tolist()
    fixture["rtol"] = 1e9
    fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "pairwise/main_prediction_fixture.json")

    with pytest.raises(ValueError, match="tolerances must both equal"):
        _load_pairwise_staging_model(bundle_dir)


def test_native_classifier_scores_in_bounded_float32_chunks(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    model_path = bundle_dir / "pairwise" / "main.lgb"
    classifier = NativeLightGBMBinaryClassifier(model_path, n_jobs=1)

    native_scorer = classifier._rust_scorer

    class CountingScorer:
        row_counts: list[int] = []

        def num_features(self):
            return native_scorer.num_features()

        def predict_proba_positive_f32(
            self,
            features: np.ndarray,
            *,
            num_threads: int | None = None,
        ) -> np.ndarray:
            self.row_counts.append(int(features.shape[0]))
            return native_scorer.predict_proba_positive_f32(features, num_threads=num_threads)

    scorer = CountingScorer()
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
        write_pairwise_production_bundle(source_clusterer, output_bundle, release_version="9.8")
    assert not output_bundle.exists()

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = write_pairwise_production_bundle(source_clusterer, output_bundle, release_version="9.8")
    assert summary.bundle_dir == output_bundle
    assert _load_pairwise_staging_model(output_bundle).production_model_release_version == "9.8"


def test_eps_finalization_failures_clean_stage_and_are_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_bundle = tmp_path / "pending" / "production_model_v9.9"
    write_pairwise_production_bundle(
        source_clusterer,
        source_bundle,
        release_version="9.9",
        eps_calibration=PENDING_EPS_CALIBRATION,
    )
    output_bundle = tmp_path / "eps" / "production_model_v9.9"
    source_manifest = source_bundle / "manifest.json"
    expected_manifest_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
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
            new_eps=new_eps,
        )
    assert not output_bundle.exists()
    assert not list(output_bundle.parent.glob(staging_pattern))

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        expected_manifest_sha256=expected_manifest_sha256,
        new_eps=new_eps,
    )

    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert summary.bundle_dir == output_bundle
    assert summary.release_version == "9.9"
    assert summary.eps_calibration == CALIBRATED_EPS_CALIBRATION
    assert summary.manifest_path == output_bundle / "manifest.json"
    assert not list(output_bundle.parent.glob(staging_pattern))
    published_config = json.loads((output_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert published_config["cluster_model"]["eps"] == new_eps
    assert manifest["eps_calibration"] == CALIBRATED_EPS_CALIBRATION
    with pytest.raises(ValueError, match="lifecycle mismatch"):
        finalize_pairwise_eps(
            source_bundle_dir=output_bundle,
            output_bundle_dir=tmp_path / "double-calibrated" / "production_model_v9.9",
            expected_manifest_sha256=hashlib.sha256((output_bundle / "manifest.json").read_bytes()).hexdigest(),
            new_eps=0.41,
        )


def test_eps_finalization_writes_the_close_pair_bonus_and_the_loader_applies_it(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    source_bundle = tmp_path / "pending" / "production_model_v9.9"
    write_pairwise_production_bundle(
        source_clusterer, source_bundle, release_version="9.9", eps_calibration=PENDING_EPS_CALIBRATION
    )
    manifest_sha256 = hashlib.sha256((source_bundle / "manifest.json").read_bytes()).hexdigest()
    plain_bundle = tmp_path / "plain" / "production_model_v9.9"
    bonus_bundle = tmp_path / "bonus" / "production_model_v9.9"

    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=plain_bundle,
        expected_manifest_sha256=manifest_sha256,
        new_eps=0.41,
    )
    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=bonus_bundle,
        expected_manifest_sha256=manifest_sha256,
        new_eps=0.41,
        close_pair_bonus=8.0,
        close_pair_distance=0.05,
    )

    # Without the bonus the bundle keeps the two-field shape that older libraries read.
    plain_config = json.loads((plain_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert plain_config["cluster_model"] == {"eps": 0.41, "linkage": "average"}
    bonus_config = json.loads((bonus_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert bonus_config["cluster_model"] == {
        "close_pair_bonus": 8.0,
        "close_pair_distance": 0.05,
        "eps": 0.41,
        "linkage": "average",
    }
    plain, bonus = _load_pairwise_staging_model(plain_bundle), _load_pairwise_staging_model(bonus_bundle)
    assert (plain.cluster_model.close_pair_bonus, plain.cluster_model.close_pair_distance) == (0.0, 0.0)
    assert (bonus.cluster_model.close_pair_bonus, bonus.cluster_model.close_pair_distance) == (8.0, 0.05)
    assert bonus.best_params == bonus_config["cluster_model"]
    # Plain linkage replaces a bonus that the source bundle already carried.
    source_config_path = source_bundle / "clusterer.json"
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_config["cluster_model"] |= {"close_pair_bonus": 8.0, "close_pair_distance": 0.05}
    source_config_path.write_text(json.dumps(source_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(source_bundle, "clusterer.json")
    cleared_bundle = tmp_path / "cleared" / "production_model_v9.9"
    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=cleared_bundle,
        expected_manifest_sha256=hashlib.sha256((source_bundle / "manifest.json").read_bytes()).hexdigest(),
        new_eps=0.41,
    )
    assert json.loads((cleared_bundle / "clusterer.json").read_text(encoding="utf-8"))["cluster_model"] == {
        "eps": 0.41,
        "linkage": "average",
    }
    assert _load_pairwise_staging_model(cleared_bundle).cluster_model.close_pair_bonus == 0.0
    # A bundle written from the loaded model carries the same linkage settings.
    rewritten = tmp_path / "rewritten" / "production_model_v9.9"
    write_pairwise_production_bundle(bonus, rewritten, release_version="9.9")
    assert (
        json.loads((rewritten / "clusterer.json").read_text(encoding="utf-8"))["cluster_model"]
        == (bonus_config["cluster_model"])
    )


def test_bundle_export_rejects_unrecordable_correction_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    # A loaded model gets the class default back, so export refuses a setting it cannot record.
    _, source_clusterer = synthetic_pairwise_bundle

    # Reject settings that would silently revert after export.
    for setting, value in (
        ("name_conflict_repair_max_distance", None),
        ("orcid_name_waiver", None),
        ("orcid_join", False),
        ("name_conflict_repair_rounds", 1),
    ):
        message = "no longer exist" if setting == "name_conflict_repair_rounds" else f"cannot record {setting}"
        with monkeypatch.context() as changed:
            changed.setattr(source_clusterer, setting, value, raising=False)
            with pytest.raises(ValueError, match=message):
                write_pairwise_production_bundle(
                    source_clusterer, tmp_path / "rejected" / "production_model_v9.9", release_version="9.9"
                )

    source_clusterer.name_conflict_repair_max_distance = Clusterer.name_conflict_repair_max_distance
    source_clusterer.orcid_name_waiver = Clusterer.orcid_name_waiver
    monkeypatch.setattr(Clusterer, "year_gap_split", False)
    with pytest.raises(ValueError, match="cannot record year_gap_split"):
        write_pairwise_production_bundle(
            source_clusterer, tmp_path / "class" / "production_model_v9.9", release_version="9.9"
        )


def test_pending_pairwise_loader_requires_placeholder_eps(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    with pytest.raises(ValueError, match="No default production model is declared"):
        load_production_model()
    _, source_clusterer = synthetic_pairwise_bundle
    pending_bundle = tmp_path / "pending-eps" / "production_model_v9.9"
    write_pairwise_production_bundle(
        source_clusterer,
        pending_bundle,
        release_version="9.9",
        eps_calibration=PENDING_EPS_CALIBRATION,
    )
    pending_clusterer = _load_pairwise_staging_model(
        pending_bundle,
        expected_eps_calibration=PENDING_EPS_CALIBRATION,
    )
    assert pending_clusterer.best_params is None
    with pytest.raises(ValueError, match="lifecycle mismatch"):
        _load_pairwise_staging_model(pending_bundle)
    with pytest.raises(ValueError, match="lifecycle mismatch"):
        pairwise_bundle_binding(pending_bundle)

    config_path = pending_bundle / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["cluster_model"]["eps"] = 0.4
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(pending_bundle, "clusterer.json")

    with pytest.raises(ValueError, match="Pending pairwise bundle EPS must be 0.5"):
        _load_pairwise_staging_model(
            pending_bundle,
            expected_eps_calibration=PENDING_EPS_CALIBRATION,
        )


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
        release_version="9.8",
        pairwise_training_config={
            "training_seed": 7,
            "input_artifact_hashes": dict(_TEST_EXPLICIT_ARTIFACT_HASHES),
        },
        pairwise_training_summary={"pair_count": 11, "optional_diagnostic": float("nan")},
    )
    loaded = _load_pairwise_staging_model(output_dir, expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES)
    assert isinstance(loaded, Clusterer)
    summary = json.loads(
        (output_dir / "reproducibility" / "pairwise_training_summary.json").read_text(encoding="utf-8")
    )
    assert np.isnan(summary["optional_diagnostic"])

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
        _load_pairwise_staging_model(output_dir)


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
    stale_path = bundle_dir / "reproducibility" / "stale-sensitive.json"
    stale_path.parent.mkdir()
    stale_path.write_text('{"secret": true}', encoding="utf-8")
    real_replace = production_bundle_module.os.replace

    clusterer_config = json.loads((bundle_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
    with pytest.raises(ValueError, match="lifecycle mismatch"):
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
    _load_pairwise_staging_model(bundle_dir)
    assert not (bundle_dir / "incremental_linker").exists()

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
    )

    assert summary.bundle_dir == output_bundle
    assert not (output_bundle / "reproducibility" / stale_path.name).exists()
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_release_version == "9.9"
    assert loaded.incremental_linker_artifact_dir is not None
    assert loaded.incremental_linker_artifact is not None
    assert loaded.nameless_featurizer_info is not None
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"
    assert loaded.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    restored = pickle.loads(pickle.dumps(loaded))
    assert restored.incremental_linker_artifact.artifact_dir == output_bundle / "incremental_linker"
    assert copy.deepcopy(loaded).incremental_linker_artifact is loaded.incremental_linker_artifact
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "eps_calibration": CALIBRATED_EPS_CALIBRATION,
        "feature_contract_version": PAIRWISE_FEATURE_CONTRACT_VERSION,
        "feature_names": {
            "main": loaded.featurizer_info.get_feature_names(),
            "nameless": loaded.nameless_featurizer_info.get_feature_names(),
        },
        "format_version": PRODUCTION_MODEL_FORMAT_VERSION,
        "kind": PRODUCTION_MODEL_KIND,
        "release_version": "9.9",
        "sha256": manifest["sha256"],
    }

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
            release_version="9.9",
        )


@pytest.fixture
def finalization(tmp_path, synthetic_pairwise_bundle):
    """One pairwise/linker pair shared by finalization and complete-loader tests."""
    pairwise, _ = synthetic_pairwise_bundle
    linker = tmp_path / "linker"
    return {
        "pairwise_bundle_dir": pairwise,
        "output_bundle_dir": tmp_path / "complete" / "production_model_v9.9",
        "incremental_linker_artifact_dir": linker,
        "target_json": _write_synthetic_linker(pairwise, linker),
    }


@pytest.mark.parametrize("published", [False, True], ids=["staging", "published"])
def test_linker_identity_rejected_before_publication_and_after_rehash(finalization, published):
    output = finalization["output_bundle_dir"]
    if published:
        finalize_production_bundle(**finalization)
        paths = {
            "target": output / "reproducibility/incremental_linker_training_target.json",
            "metadata": output / "incremental_linker/metadata.json",
        }
    else:
        paths = {
            "target": finalization["target_json"],
            "metadata": finalization["incremental_linker_artifact_dir"] / "metadata.json",
        }

    cases = [
        ("target", "variant", "target_spec_digest"),
        ("metadata", "pairwise_bundle_binding_digest", "pairwise_bundle_binding_digest"),
    ]
    if not published:
        cases.append(("metadata", "booster_sha256", "booster_sha256 mismatch"))
    for kind, field, message in cases:
        path = paths[kind]
        original = path.read_bytes()
        payload = json.loads(original)
        payload[field] = "f" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        if published:
            _refresh_manifest_checksum(output, path.relative_to(output).as_posix())
        with pytest.raises(ValueError, match=message):
            load_production_model(output) if published else finalize_production_bundle(**finalization)
        assert output.exists() is published
        path.write_bytes(original)
        if published:
            _refresh_manifest_checksum(output, path.relative_to(output).as_posix())
    _load_pairwise_staging_model(finalization["pairwise_bundle_dir"])


def test_normal_load_hashes_each_declared_file_exactly_once(finalization, monkeypatch) -> None:
    output_bundle = finalization["output_bundle_dir"]
    finalize_production_bundle(**finalization)

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
    clusterer = load_production_model(output_bundle)

    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    for classifier, model_file in (
        (clusterer.classifier, "pairwise/main.lgb"),
        (clusterer.nameless_classifier, "pairwise/nameless.lgb"),
    ):
        assert classifier is not None
        expected_fingerprint = ("native_lightgbm", manifest["sha256"][model_file], classifier.n_features_in_)
        assert classifier.cache_fingerprint() == expected_fingerprint
        assert classifier.cache_fingerprint() == expected_fingerprint
    expected = sorted((output_bundle / relpath).resolve().as_posix() for relpath in manifest["sha256"])
    assert sorted(manifest_hashes) == expected
    assert standalone_artifact_hashes == []
    with monkeypatch.context() as restore:
        restore.setattr(production_model_module, "_sha256_file", real_manifest_sha256)
        restore.setattr(incremental_artifact_module, "_sha256_file", real_artifact_sha256)
        requested = load_production_model(output_bundle, n_jobs=3)
    assert clusterer.n_jobs == 1
    assert (requested.n_jobs, requested.classifier.n_jobs, requested.nameless_classifier.n_jobs) == (3, 3, 3)
    with pytest.raises(ValueError, match="n_jobs must not be zero"):
        load_production_model(output_bundle, n_jobs=0)


@pytest.mark.parametrize("explicit_authority", [False, True])
def test_load_production_model_rejects_an_unknown_orcid_prefix_hash(finalization, explicit_authority: bool) -> None:
    output_bundle = finalization["output_bundle_dir"]
    finalize_production_bundle(**finalization)
    path = output_bundle / "clusterer.json"
    config = json.loads(path.read_bytes())
    unknown_hash = "c" * 64
    hashes = {
        **config["feature_contract"],
        "orcid_prefix_counts_data_sha256": unknown_hash,
        "name_counts_manifest_sha256": "d" * 64,
    }
    config["feature_contract"] = hashes
    path.write_text(json.dumps(config), encoding="utf-8")
    _refresh_manifest_checksum(output_bundle, "clusterer.json")

    with pytest.raises(ValueError, match=f"Unknown orcid_prefix_counts_data_sha256.*{unknown_hash}"):
        load_production_model(output_bundle, expected_artifact_hashes=hashes if explicit_authority else None)


def test_manifest_writer_rejects_invalid_contracts_without_publishing(tmp_path: Path) -> None:
    for invalid_value in ("   ",):
        with pytest.raises(ValueError, match="release_version must be a nonempty trimmed string"):
            production_bundle_module.write_production_manifest(
                tmp_path,
                release_version=invalid_value,
                bundle_kind=PAIRWISE_ONLY_BUNDLE_KIND,
                eps_calibration=CALIBRATED_EPS_CALIBRATION,
            )

    for bundle_kind, eps_calibration in (
        ("complete", "pending"),
        ("pairwise_only", "unknown"),
        ("unknown", "calibrated"),
    ):
        with pytest.raises(ValueError, match="Unsupported production bundle lifecycle state"):
            production_bundle_module.write_production_manifest(
                tmp_path,
                release_version="9.9",
                bundle_kind=bundle_kind,
                eps_calibration=eps_calibration,
            )

    assert not (tmp_path / "manifest.json").exists()

    files = production_bundle_module.production_manifest_files(
        bundle_kind=PAIRWISE_ONLY_BUNDLE_KIND,
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
            release_version="9.9",
            bundle_kind=PAIRWISE_ONLY_BUNDLE_KIND,
            eps_calibration=CALIBRATED_EPS_CALIBRATION,
        )
    assert not (tmp_path / "manifest.json").exists()


def test_rebinding_the_name_count_index_changes_only_the_bound_metadata(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    """Bind a complete model to another index and keep every trained file."""
    from scripts.release.rebind_name_counts_index import rebind_name_counts_index
    from tests.helpers import tiny_name_counts_index_path

    _, source_clusterer = synthetic_pairwise_bundle
    source_clusterer.feature_contract.update(_TEST_EXPLICIT_ARTIFACT_HASHES)
    bundle_dir = tmp_path / "explicit-pairwise"
    write_pairwise_production_bundle(
        source_clusterer,
        bundle_dir,
        release_version="9.9",
        pairwise_training_config={"input_artifact_hashes": dict(_TEST_EXPLICIT_ARTIFACT_HASHES)},
        pairwise_training_summary={"pair_count": 11},
    )
    source = tmp_path / "complete" / "production_model_v9.9"
    linker_dir = tmp_path / "linker"
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=source,
        incremental_linker_artifact_dir=linker_dir,
        target_json=_write_synthetic_linker(
            bundle_dir, linker_dir, expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES
        ),
        expected_artifact_hashes=_TEST_EXPLICIT_ARTIFACT_HASHES,
    )
    output = tmp_path / "rebound" / "production_model_v9.10"

    report = rebind_name_counts_index(
        source_model=source,
        name_counts_index=Path(tiny_name_counts_index_path()),
        output_model=output,
        release_version="9.10",
    )

    loaded = load_production_model(output)
    index_hash = tiny_name_counts_index().manifest_sha256
    assert loaded.production_model_release_version == "9.10"
    assert loaded.feature_contract["name_counts_manifest_sha256"] == index_hash
    assert report["output_name_counts_manifest_sha256"] == index_hash != report["source_name_counts_manifest_sha256"]
    changed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.read_bytes() != (source / path.relative_to(output)).read_bytes()
    }
    assert changed == {"clusterer.json", "incremental_linker/metadata.json", "manifest.json"}
    # The source stays loadable, and a second run refuses to overwrite.
    load_production_model(source)
    with pytest.raises(FileExistsError):
        rebind_name_counts_index(
            source_model=source,
            name_counts_index=Path(tiny_name_counts_index_path()),
            output_model=output,
            release_version="9.10",
        )

    with pytest.raises(ValueError, match="inside the source"):
        rebind_name_counts_index(
            source_model=source,
            name_counts_index=Path(tiny_name_counts_index_path()),
            output_model=source / "rebound",
            release_version="9.10",
        )
    assert not (source / "rebound").exists()


@pytest.mark.parametrize("version, expected", [(1, 50.0), (2, np.nan)])
def test_loaded_pairwise_contract_replays_stamped_year(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    tmp_path: Path,
    version: int,
    expected: float,
) -> None:
    from s2and.featurizer import many_pairs_featurize
    from s2and.runtime import build_runtime_context
    from tests.helpers import build_arrow_training_dataset
    from tests.raw_arrow_helpers import base_arrow_paths, open_arrow_dataset
    from tests.test_feature_port_parity import _load_dataset_from_dir

    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if version == 1:
        manifest["release_version"] = "1.4"
        manifest["feature_contract_version"] = 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["feature_contract_version"] == version
    linker_dir = tmp_path / "linker"
    target = _write_synthetic_linker(bundle_dir, linker_dir)
    metadata_path = linker_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if version == 1:
        metadata["feature_contract_version"] = 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        assert metadata["feature_contract_version"] == 2
    complete_dir = tmp_path / "complete"
    finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=complete_dir,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target,
    )
    clusterer = load_production_model(complete_dir)
    assert clusterer.incremental_linker_artifact is not None
    legacy = clusterer.incremental_linker_artifact.legacy_paper_years
    assert legacy is (version == 1)
    paths = base_arrow_paths(tmp_path / "stamped_retrieval", years=[1800, 2015, 2010])
    with open_arrow_dataset(paths) as arrow:
        from s2and.runtime import load_s2and_rust_extension

        planner = load_s2and_rust_extension().RawBlockQueryCandidatePlanner.from_auto_queries(
            arrow.native,
            paths["cluster_seeds"],
            top_k=2,
            orcid_enabled=False,
            num_threads=1,
            legacy_paper_years=legacy,
        )
        try:
            plan = planner.plan(["q1"])
        finally:
            planner.release()
    assert plan["row_count"] > 0
    assert set(plan["row_query_year_missing"]) == {int(version == 2)}
    if legacy:
        assert set(plan["row_query_years"]) == {1800}
    dataset = _load_dataset_from_dir(str(Path(__file__).parent / "dummy"), "stamped")
    left, right = "0", "2"
    for signature, year in ((left, 1800), (right, 2010)):
        paper_id = str(dataset.signatures[signature].paper_id)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=year)
    assert clusterer.nameless_featurizer_info is not None
    info = clusterer.nameless_featurizer_info
    for backend in ("python", "rust"):
        runtime = build_runtime_context("stamped", backend=backend)
        source = dataset if backend == "python" else build_arrow_training_dataset(dataset, tmp_path / "stamped_arrow")
        features, _, _ = many_pairs_featurize(
            [(left, right, 1)],
            source,
            info,
            n_jobs=1,
            runtime_context=runtime,
        )
        np.testing.assert_equal(features[0, 0], expected)


@pytest.fixture
def contract_one_inference(
    tmp_path: Path, synthetic_pairwise_bundle: tuple[Path, Clusterer]
) -> Iterator[tuple[Clusterer, ArrowDataset]]:
    import pyarrow as pa

    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_name_counts_index
    from tests.helpers import tiny_name_counts_tuple, write_test_arrow_artifact_manifest
    from tests.raw_arrow_helpers import base_arrow_paths, write_ipc

    pairwise, _ = synthetic_pairwise_bundle
    manifest_path = pairwise / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(release_version="1.4", feature_contract_version=1)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    linker = tmp_path / "linker_contract_one"
    target = _write_synthetic_linker(pairwise, linker)
    metadata_path = linker / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_path.write_text(json.dumps({**metadata, "feature_contract_version": 1}), encoding="utf-8")
    complete = tmp_path / "complete_contract_one"
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise,
        output_bundle_dir=complete,
        incremental_linker_artifact_dir=linker,
        target_json=target,
    )
    clusterer = load_production_model(complete, n_jobs=1)
    clusterer.use_default_constraints_as_supervision = False
    arrow_root = tmp_path / "arrow_contract_one"
    paths = base_arrow_paths(arrow_root, years=[1800, 2015, 2010])
    specter = pa.table(
        {
            "paper_id": ["p_q", "p1", "p2"],
            "embedding": pa.array([[1.0, 0.0]] * 3, type=pa.list_(pa.float32(), 2)),
        }
    )
    paths["specter"] = write_ipc(arrow_root / "specter.arrow", specter)
    paths["specter_batch_index"], _ = write_arrow_batch_lookup_index(
        Path(paths["specter"]),
        arrow_root / "specter.bin",
        key_column="paper_id",
        table_name="specter",
    )
    paths["name_counts_index"], _ = write_name_counts_index(arrow_root, tiny_name_counts_tuple())
    write_test_arrow_artifact_manifest(arrow_root, paths)
    with ArrowDataset.open(arrow_root) as arrow:
        yield clusterer, arrow


def test_contract_one_prediction_and_eps_replay_stamped_year(
    contract_one_inference, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import s2and.incremental_linking.runtime as linker_runtime
    from s2and.model import distances
    from scripts.eps import sweep_eps_on_linking_gold as sweep

    clusterer, arrow = contract_one_inference
    pair_ids = ["q1", "s1"]
    blocks = {"a wang": pair_ids}
    year_diffs: list[float] = []
    row_years: list[int] = []
    original_bulk_predict = distances._predict_class0_with_runtime
    original_incremental_predict = linker_runtime.predict_pairwise_class0
    original_row_features = linker_runtime.build_promoted_non_pairwise_row_features_with_telemetry

    def record_features(classifier, features):
        if classifier is clusterer.nameless_classifier:
            year_diffs.extend(features[:, 0].tolist())

    def bulk_predict(classifier, features, **kwargs):
        record_features(classifier, features)
        return original_bulk_predict(classifier, features, **kwargs)

    def incremental_predict(classifier, features):
        record_features(classifier, features)
        return original_incremental_predict(classifier, features)

    def record_row_features(candidate_batch, signals, *, legacy_paper_years):
        assert legacy_paper_years
        row_years.extend(signals["query_year"].tolist())
        return original_row_features(candidate_batch, signals, legacy_paper_years=legacy_paper_years)

    # Use the supported unfused scoring paths to observe the real native feature matrices.
    monkeypatch.setattr(distances, "_native_block_scorers", lambda *_: None)
    monkeypatch.setattr(linker_runtime, "_FUSED_LINKER_SCORING", False)
    monkeypatch.setattr(distances, "_predict_class0_with_runtime", bulk_predict)
    monkeypatch.setattr(linker_runtime, "predict_pairwise_class0", incremental_predict)
    monkeypatch.setattr(linker_runtime, "build_promoted_non_pairwise_row_features_with_telemetry", record_row_features)
    clusters, _ = clusterer.predict_from_arrow(blocks, arrow, name_tuples=frozenset())
    assert sorted(sid for members in clusters.values() for sid in members) == sorted(pair_ids)
    assert year_diffs == [50.0]

    year_diffs.clear()
    result = clusterer.predict_incremental_from_arrow(
        pair_ids, arrow, cluster_seeds_require={"s1": "1"}, name_tuples=frozenset()
    )
    assert sorted(sid for members in result["clusters"].values() for sid in members) == sorted(pair_ids)
    assert row_years == [1800]
    assert year_diffs == [50.0]

    year_diffs.clear()
    args = argparse.Namespace(
        model_path=clusterer.production_model_bundle_dir,
        arrow_root=tmp_path,
        dataset="test",
        batching_threshold=10,
        pair_chunk_size=4,
        use_orcid_subblocking=False,
        suppress_orcid_constraints=True,
        total_ram_bytes=2**30,
        overwrite_dists=False,
        compute_missing_dists=True,
    )
    caches = sweep._ensure_distance_caches(args, clusterer, blocks, tmp_path / "cache", arrow)
    assert caches[0]["computed"]
    assert year_diffs == [50.0]


def test_contract_one_cached_production_route_skips_bulk_featurizer(contract_one_inference) -> None:
    clusterer, arrow = contract_one_inference
    blocks = {"a wang": ["q1", "s1"]}

    def cached_clusters(groups):
        return {f"{key}_{i}": [sid] for key, members in groups.items() for i, sid in enumerate(members)}

    clusters, _ = clusterer._predict_from_arrow_request(
        blocks,
        arrow,
        batching_threshold=1,
        name_tuples=frozenset(),
        bulk_clusters=cached_clusters,
    )
    assert sorted(sid for members in clusters.values() for sid in members) == ["q1", "s1"]


def test_contract_one_disallow_rescore_cache_miss_replays_stamped_year(
    contract_one_inference: tuple[Clusterer, ArrowDataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    from s2and import memory_budget
    from s2and.incremental_linking import production, runtime
    from s2and.runtime import build_runtime_context, load_s2and_rust_extension

    clusterer, arrow = contract_one_inference
    artifact = clusterer.incremental_linker_artifact
    assert artifact is not None
    seeds = {"s1": "c_match", "s2": "c_other"}
    seed_path = arrow.root / "cluster_seeds.arrow"
    observed: list[float] = []
    predict = runtime.predict_pairwise_class0

    def record(classifier, features):
        if classifier is clusterer.nameless_classifier:
            observed.extend(features[:, 0].tolist())
        return predict(classifier, features)

    monkeypatch.setattr(runtime, "_FUSED_LINKER_SCORING", False)
    monkeypatch.setattr(runtime, "predict_pairwise_class0", record)
    planner = load_s2and_rust_extension().RawBlockQueryCandidatePlanner.from_auto_queries(
        arrow.native, str(seed_path), top_k=2, orcid_enabled=False, num_threads=1, legacy_paper_years=True
    )
    try:
        context = production._QueryDisallowRescoreContext(
            clusterer=clusterer,
            artifact=artifact,
            planner=planner,
            arrow_dataset=arrow,
            cluster_seeds_path=seed_path,
            cluster_seed_disallows_path=None,
            name_tuples=frozenset(),
            retrieval_top_k=2,
            partial_supervision={},
            runtime_context=build_runtime_context("rescore", backend="rust"),
            total_ram_bytes=2**35,
            cluster_seeds_require=seeds,
            cluster_seed_representative_by_component={value: key for key, value in seeds.items()},
            orcid_fanout_by_query={},
            component_sizes=memory_budget.summarize_promoted_component_sizes({"c_match": 1, "c_other": 1}),
            memory_layout=production._promoted_incremental_memory_layout(clusterer, artifact),
            base_candidate_rows_per_query=2,
            base_pairs_per_query=2,
            featurizer_cache=production._QueryDisallowFeaturizerCache(),
        )
        result = production._rescore_query_disallow_endpoint(context, "q1", {"c_other"})
        assert result.telemetry["featurizer_build_count"] == 1
        assert observed == [50.0]
    finally:
        planner.release()


def test_contract_one_fused_distance_chunk_replays_stamped_year(
    contract_one_inference: tuple[Clusterer, ArrowDataset], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import s2and.model as model_module
    from s2and.feature_port import _get_rust_feature_data
    from s2and.model import distances
    from tests.helpers import build_arrow_training_dataset
    from tests.test_feature_port_parity import _load_dataset_from_dir

    clusterer, _ = contract_one_inference
    raw = _load_dataset_from_dir(str(Path(__file__).parent / "dummy"), "stamped_chunk")
    for sid, year in (("0", 1800), ("2", 2010)):
        paper_id = str(raw.signatures[sid].paper_id)
        raw.papers[paper_id] = raw.papers[paper_id]._replace(year=year)
    dataset = build_arrow_training_dataset(raw, tmp_path / "training")
    native = _get_rust_feature_data(dataset)
    ids = native.signature_ids()
    chunk = model_module._DistanceMatrixChunk(
        block_key="a wang",
        block_size=2,
        start_offset=0,
        index_i=np.array([0]),
        index_j=np.array([1]),
        pair_ids=None,
        labels=np.array([np.nan]),
        block_signature_indices=[ids.index("0"), ids.index("2")],
    )
    observed: list[float] = []
    original = distances._predict_and_combine

    def record(classifier, nameless, features, labels, nameless_features, *args, **kwargs):
        observed.extend(nameless_features[:, 0].tolist())
        return original(classifier, nameless, features, labels, nameless_features, *args, **kwargs)

    monkeypatch.setattr(distances, "_predict_and_combine", record)
    clusterer._predict_distance_matrix_chunk(chunk, dataset, dataset.runtime_context, "stamped")
    assert observed == [50.0]


def test_prebuilt_native_prediction_preserves_contract_one(
    contract_one_inference: tuple[Clusterer, ArrowDataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
    from s2and.model import distances

    clusterer, arrow = contract_one_inference
    ids = ["q1", "s1"]
    native = build_rust_featurizer_from_arrow_dataset(arrow, signature_ids=ids, name_tuples=frozenset(), num_threads=1)
    observed: list[float] = []
    predict = distances._predict_class0_with_runtime

    def record(classifier, features, **kwargs):
        if classifier is clusterer.nameless_classifier:
            observed.extend(features[:, 0].tolist())
        return predict(classifier, features, **kwargs)

    monkeypatch.setattr(distances, "_native_block_scorers", lambda *_: None)
    monkeypatch.setattr(distances, "_predict_class0_with_runtime", record)
    clusterer.predict_from_rust_featurizer({"a wang": ids}, native)
    assert observed == [50.0]


def test_background_planner_preserves_contract_one_stamped_query(
    contract_one_inference: tuple[Clusterer, ArrowDataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    from s2and.incremental_linking import production
    from s2and.incremental_linking.feature_block import temporary_cluster_seed_sidecars

    clusterer, arrow = contract_one_inference
    artifact = clusterer.incremental_linker_artifact
    assert artifact is not None
    seeds = {"s1": "c_match", "s2": "c_other"}
    monkeypatch.setattr(production, "_BACKGROUND_SEED_PLANNER_MIN_SIGNATURES", 1)
    background = production.start_background_seed_planner(
        clusterer,
        ["q1", "s1", "s2"],
        arrow_dataset=arrow,
        artifact=artifact,
        cluster_seeds_require=seeds,
        altered_cluster_signatures=["s1"],
        query_signature_ids=None,
        attachment_state=None,
    )
    assert background is not None
    try:
        final_seeds = {**seeds, "s1": "c_split"}
        with temporary_cluster_seed_sidecars(final_seeds, prefix="reloaded_seeds_") as sidecars:
            planner = background.take(Path(sidecars["cluster_seeds"]), None)
            assert planner is not None
            try:
                plan = planner.plan(["q1"])
            finally:
                planner.release()
        assert set(plan["row_component_keys"]) == set(final_seeds.values())
        assert set(plan["row_query_years"]) == {1800}
        assert set(plan["row_query_year_missing"]) == {0}
    finally:
        background.release()


@pytest.mark.parametrize(
    ("legacy_years", "contract"),
    [(True, LEGACY_PAIRWISE_FEATURE_CONTRACT_VERSION), (False, PAIRWISE_FEATURE_CONTRACT_VERSION)],
)
def test_pairwise_eps_publication_preserves_contract(
    tmp_path: Path, synthetic_pairwise_bundle: tuple[Path, Clusterer], legacy_years: bool, contract: int
) -> None:
    _, clusterer = synthetic_pairwise_bundle
    clusterer.featurizer_info.legacy_paper_years = legacy_years
    clusterer.nameless_featurizer_info.legacy_paper_years = legacy_years
    source, output = tmp_path / "pending", tmp_path / "calibrated"
    write_pairwise_production_bundle(clusterer, source, release_version="1.4", eps_calibration=PENDING_EPS_CALIBRATION)
    manifest = source / "manifest.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["feature_contract_version"] == contract
    finalize_pairwise_eps(
        source_bundle_dir=source,
        output_bundle_dir=output,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        new_eps=0.37,
    )
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["feature_contract_version"] == contract


@pytest.mark.parametrize("overlay", ["view", "seeds"])
def test_legacy_featurizer_seed_views_preserve_years(
    contract_one_inference: tuple[Clusterer, ArrowDataset], overlay: str
) -> None:
    from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
    from s2and.featurizer import FeaturizationInfo

    _, arrow = contract_one_inference
    ids = ["q1", "s1"]
    native = build_rust_featurizer_from_arrow_dataset(
        arrow,
        signature_ids=ids,
        name_tuples=frozenset(),
        preprocess=True,
        num_threads=1,
    ).with_legacy_paper_years()
    view = (
        native.view_with_cluster_seed_sidecars(ids, None, None, num_threads=1)
        if overlay == "view"
        else native.with_cluster_seeds({}, set())
    )
    assert dict(view.signature_years())["q1"] == 1800
    ordered = view.signature_ids()
    matrix = view.featurize_pairs_matrix_indexed(
        [(ordered.index("q1"), ordered.index("s1"))],
        FeaturizationInfo(["year_diff"]).selected_feature_indices(),
        1,
        np.nan,
    )
    assert np.asarray(matrix)[0, 0] == 50.0
