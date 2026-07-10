from __future__ import annotations

import copy
import hashlib
import json
import pickle
import stat
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.incremental_linking.artifact as incremental_artifact_module
import s2and.production_bundle as production_bundle_module
import s2and.production_model as production_model_module
import s2and.subblocking as subblocking_module
from s2and.arrow_inputs import MissingArrowArtifactError
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, FastCluster, _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import (
    DEFAULT_PRODUCTION_MODEL_DIR,
    NativeLightGBMBinaryClassifier,
    _config_choice,
    _production_runtime_cluster_eps,
    _require_featurizer_version_match,
    load_production_model,
    pairwise_bundle_binding,
)
from tests.helpers import import_s2and_rust, tiny_name_counts
from tests.promoted_linking_helpers import build_tiny_promoted_booster


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
        "name_counts_last_first_initial_semantics": "initial_char",
        "normalization_version": NORMALIZATION_VERSION,
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
    booster, fixture = build_tiny_promoted_booster()
    gate_config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )
    save_incremental_linking_artifact(
        booster,
        linker_dir,
        prediction_fixture_matrix=fixture,
        gate_config=gate_config,
        audit_metadata={"pairwise_bundle_binding": pairwise_bundle_binding(pairwise_bundle)},
    )
    target = linker_dir.parent / "target.json"
    target.write_text("{}\n", encoding="utf-8")
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
        load_name_counts=tiny_name_counts(),
        preprocess=True,
        n_jobs=1,
    )


def _prepare_prediction_clusterer(clusterer):
    _ensure_lightgbm_fitted(clusterer.classifier)
    _ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.n_jobs = 1
    clusterer.use_cache = False
    return clusterer


def _predict_dummy_block(clusterer, *, batching_threshold: int | None) -> dict[str, list[str]]:
    dataset = _load_dummy_inference_dataset(f"dummy-predict-{batching_threshold}")
    block = {
        "a sattar": [str(signature_index) for signature_index in range(9)],
    }
    predictions, dists = clusterer.predict(block, dataset, batching_threshold=batching_threshold)

    assert dists is None
    return predictions


@pytest.mark.skip(reason="the declared packaged default remains legacy until the canonical v1.3 bundle is available")
def test_packaged_default_production_bundle_smoke() -> None:
    assert load_production_model().production_model_bundle_dir == DEFAULT_PRODUCTION_MODEL_DIR


def test_native_production_bundle_loads_as_mutable_clusterer(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = load_production_model(bundle_dir, require_incremental_linker=False)

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


def test_production_name_count_model_requires_exact_binding(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    config_path = bundle_dir / "clusterer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["featurizer_info"]["features_to_use"] = ["name_counts"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "clusterer.json")

    with pytest.raises(ValueError, match="requires name-count provenance fields"):
        load_production_model(bundle_dir, require_incremental_linker=False)


def test_native_lightgbm_set_params_rejects_unknown_params(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = load_production_model(bundle_dir, require_incremental_linker=False)

    with pytest.raises(ValueError, match="Invalid parameter"):
        clusterer.classifier.set_params(learning_rate=0.1)


def test_native_lightgbm_deepcopy_does_not_require_model_path(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    clusterer = load_production_model(bundle_dir, require_incremental_linker=False)
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
    native_clusterer = load_production_model(bundle_dir, require_incremental_linker=False)

    rng = np.random.default_rng(921)
    main_width = len(_selected_feature_indices(source_clusterer.featurizer_info))
    assert source_clusterer.nameless_featurizer_info is not None
    nameless_width = len(_selected_feature_indices(source_clusterer.nameless_featurizer_info))
    main_features = rng.normal(size=(8, main_width))
    nameless_features = rng.normal(size=(8, nameless_width))

    np.testing.assert_allclose(
        native_clusterer.classifier.predict_proba(main_features)[:, 1],
        source_clusterer.classifier.predict_proba(main_features)[:, 1],
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
        source_clusterer.nameless_classifier.predict_proba(nameless_features)[:, 1],
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
        native_clusterer = _prepare_prediction_clusterer(
            load_production_model(bundle_dir, require_incremental_linker=False)
        )
        expected_clusterer = _prepare_prediction_clusterer(copy.deepcopy(source_clusterer))

        assert _predict_dummy_block(native_clusterer, batching_threshold=batching_threshold) == _predict_dummy_block(
            expected_clusterer,
            batching_threshold=batching_threshold,
        )


def test_native_clusterer_predict_rust_requires_arrow_paths(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    rust_available, rust_error = import_s2and_rust(required_method="from_arrow_paths")
    if not rust_available:
        raise pytest.skip.Exception(f"Rust runtime unavailable: {rust_error!r}")

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    bundle_dir, _ = synthetic_pairwise_bundle
    native_clusterer = _prepare_prediction_clusterer(
        load_production_model(bundle_dir, require_incremental_linker=False)
    )

    with pytest.raises(MissingArrowArtifactError, match="Rust production prediction requires complete Arrow artifacts"):
        _predict_dummy_block(native_clusterer, batching_threshold=None)


def test_native_clusterer_predict_auto_without_arrow_paths_uses_python(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    rust_available, rust_error = import_s2and_rust(required_method="from_arrow_paths")
    if not rust_available:
        raise pytest.skip.Exception(f"Rust runtime unavailable: {rust_error!r}")

    monkeypatch.delenv("S2AND_BACKEND", raising=False)
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    native_clusterer = _prepare_prediction_clusterer(
        load_production_model(bundle_dir, require_incremental_linker=False)
    )

    monkeypatch.setenv("S2AND_BACKEND", "python")
    expected = _predict_dummy_block(
        _prepare_prediction_clusterer(copy.deepcopy(source_clusterer)),
        batching_threshold=None,
    )

    monkeypatch.delenv("S2AND_BACKEND", raising=False)
    assert _predict_dummy_block(native_clusterer, batching_threshold=None) == expected


def test_synthetic_native_clusterer_runtime_config_round_trips(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, source_clusterer = synthetic_pairwise_bundle
    native_clusterer = load_production_model(bundle_dir, require_incremental_linker=False)

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
    assert native_clusterer.use_cache == source_clusterer.use_cache
    assert native_clusterer.n_iter == source_clusterer.n_iter
    assert native_clusterer.random_state == source_clusterer.random_state

    assert getattr(native_clusterer, "suppress_orcid", False) == getattr(source_clusterer, "suppress_orcid", False)
    assert native_clusterer._incremental_experiment_config() == source_clusterer._incremental_experiment_config()


def test_production_runtime_cluster_eps_policy_is_version_scoped(tmp_path: Path) -> None:
    assert _production_runtime_cluster_eps(tmp_path / "production_model_v1.2.pickle") == 0.65
    assert _production_runtime_cluster_eps(tmp_path / "production_model_v1.21") == 0.65
    assert (
        _production_runtime_cluster_eps(
            tmp_path / "production_model_v9.9",
            manifest={"bundle_version": "9.9", "pairwise_model_version": "1.2"},
            clusterer_config={"bundle_version": "9.9", "source_model_version": "9.9"},
        )
        == 0.65
    )
    assert (
        _production_runtime_cluster_eps(
            tmp_path / "production_model_v9.9",
            manifest={"bundle_version": "9.9", "pairwise_model_version": "9.9"},
            clusterer_config={"bundle_version": "9.9", "source_model_version": "9.9"},
        )
        is None
    )


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


@pytest.mark.parametrize("unsafe_path", ("../clusterer.json", "/tmp/clusterer.json", "pairwise\\main.lgb"))
def test_manifest_rejects_paths_outside_bundle(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    unsafe_path: str,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["clusterer_config"] = unsafe_path
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle root|POSIX separators"):
        load_production_model(bundle_dir, require_incremental_linker=False)


def test_manifest_requires_complete_checksum_coverage(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["sha256"]["pairwise/main.lgb"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum coverage mismatch"):
        load_production_model(bundle_dir, require_incremental_linker=False)


def test_manifest_rejects_undeclared_runtime_file(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    (bundle_dir / "pairwise" / "stale.lgb").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared or missing runtime files"):
        load_production_model(bundle_dir, require_incremental_linker=False)


def test_pairwise_metadata_must_match_clusterer_ordered_features(
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    metadata_path = bundle_dir / "pairwise" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["main"]["features_to_use"] = ["year_diff"]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_checksum(bundle_dir, "pairwise/metadata.json")

    with pytest.raises(ValueError, match="ordered features contradict"):
        load_production_model(bundle_dir, require_incremental_linker=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda payload: payload["cluster_model"].update({"eps": float("nan")}), "must be finite"),
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
        load_production_model(bundle_dir, require_incremental_linker=False)


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


def test_failed_finalization_leaves_pairwise_bundle_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    original_manifest = (bundle_dir / "manifest.json").read_bytes()
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)

    def fail_validation(_path: Path) -> None:
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(production_bundle_module, "load_production_model", fail_validation)
    with pytest.raises(RuntimeError, match="injected validation failure"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=bundle_dir,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert (bundle_dir / "manifest.json").read_bytes() == original_manifest
    assert not (bundle_dir / "incremental_linker").exists()


def test_bundle_fsync_accepts_read_only_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "model.lgb"
    artifact.write_bytes(b"immutable model")
    artifact.chmod(stat.S_IREAD)
    try:
        production_bundle_module._fsync_tree(tmp_path)  # noqa: SLF001
    finally:
        artifact.chmod(stat.S_IREAD | stat.S_IWRITE)


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
    assert load_production_model(output_bundle, require_incremental_linker=False).production_model_bundle_status == (
        "pairwise_only"
    )


def test_pairwise_stage_publication_is_immutable_or_byte_identical(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    _, source_clusterer = synthetic_pairwise_bundle
    output_bundle = tmp_path / "production_model_v9.8"
    write_pairwise_production_bundle(source_clusterer, output_bundle, bundle_version="9.8")
    original_manifest = (output_bundle / "manifest.json").read_bytes()

    write_pairwise_production_bundle(source_clusterer, output_bundle, bundle_version="9.8")
    assert (output_bundle / "manifest.json").read_bytes() == original_manifest

    with pytest.raises(FileExistsError, match="immutable"):
        write_pairwise_production_bundle(
            source_clusterer,
            output_bundle,
            bundle_version="9.8",
            source_model_version="9.7",
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
    load_production_model(output_dir, require_incremental_linker=False)

    path = output_dir / relative_path
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_production_model(output_dir, require_incremental_linker=False)


@pytest.mark.parametrize(
    ("failure_destination", "linker_published", "target_published"),
    [
        (Path("incremental_linker"), False, False),
        (Path("reproducibility/incremental_linker_training_target.json"), True, False),
        (Path("manifest.json"), True, True),
    ],
)
def test_in_place_finalization_commits_manifest_last_and_is_retry_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
    failure_destination: Path,
    linker_published: bool,
    target_published: bool,
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    real_replace = production_bundle_module.os.replace

    failed = False

    def fail_publication_once(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        destination_path = Path(destination)
        if not failed and destination_path == bundle_dir / failure_destination:
            failed = True
            raise OSError("injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(production_bundle_module.os, "replace", fail_publication_once)
    with pytest.raises(OSError, match="injected publication failure"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=bundle_dir,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert bundle_dir.is_dir()
    assert json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["bundle_status"] == ("pairwise_only")
    assert (bundle_dir / "incremental_linker" / "metadata.json").is_file() is linker_published
    assert (bundle_dir / "reproducibility" / "incremental_linker_training_target.json").is_file() is target_published
    assert load_production_model(bundle_dir, require_incremental_linker=False).production_model_bundle_status == (
        "pairwise_only"
    )

    monkeypatch.setattr(production_bundle_module.os, "replace", real_replace)
    summary = finalize_production_bundle(
        pairwise_bundle_dir=bundle_dir,
        output_bundle_dir=bundle_dir,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.9",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    assert summary.bundle_status == "complete"
    loaded = load_production_model(bundle_dir)
    assert loaded.production_model_bundle_status == "complete"
    assert loaded.incremental_linker_artifact.artifact_dir == bundle_dir / "incremental_linker"
    restored = pickle.loads(pickle.dumps(loaded))
    assert restored.incremental_linker_artifact.artifact_dir == bundle_dir / "incremental_linker"
    assert copy.deepcopy(loaded).incremental_linker_artifact is loaded.incremental_linker_artifact


def test_finalization_rejects_linker_bound_to_different_pairwise_bundle(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    bundle_dir, _ = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(bundle_dir, linker_dir)
    metadata_path = linker_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pairwise_bundle_binding"]["main_booster_sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pairwise_bundle_binding does not match"):
        finalize_production_bundle(
            pairwise_bundle_dir=bundle_dir,
            output_bundle_dir=bundle_dir,
            incremental_linker_artifact_dir=linker_dir,
            target_json=target_json,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            incremental_linker_version="9.9",
        )

    assert json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))["bundle_status"] == ("pairwise_only")


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


def test_manifest_writer_rejects_contradictory_bundle_states(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require a nonempty incremental_linker_version"):
        production_bundle_module.write_production_manifest(
            tmp_path,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            include_incremental_linker=True,
            incremental_linker_version=None,
        )
    with pytest.raises(ValueError, match="cannot declare an incremental_linker_version"):
        production_bundle_module.write_production_manifest(
            tmp_path,
            bundle_version="9.9",
            pairwise_model_version="9.9",
            include_incremental_linker=False,
            incremental_linker_version="9.9",
        )


def test_pairwise_stage_finalizes_into_loadable_production_bundle(
    tmp_path: Path,
    synthetic_pairwise_bundle: tuple[Path, Clusterer],
) -> None:
    output_bundle, source_clusterer = synthetic_pairwise_bundle
    linker_dir = tmp_path / "linker"
    target_json = _write_synthetic_linker(output_bundle, linker_dir)

    clusterer_config = json.loads((output_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
    with pytest.raises(FileNotFoundError, match="pairwise-only"):
        load_production_model(output_bundle)

    pairwise_only_clusterer = load_production_model(output_bundle, require_incremental_linker=False)
    assert pairwise_only_clusterer.production_model_bundle_status == "pairwise_only"

    final_summary = finalize_production_bundle(
        pairwise_bundle_dir=output_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
        bundle_version="9.9",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    assert final_summary.bundle_status == "complete"
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_bundle_version == "9.9"
    assert loaded.incremental_linker_artifact_dir is not None
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"

    with pytest.raises(ValueError, match="existing incremental linker artifacts"):
        write_pairwise_production_bundle(
            source_clusterer,
            output_bundle,
            bundle_version="9.9",
            source_model_version="9.9",
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
            output_bundle_dir=tmp_path / "production_model_v9.9",
            incremental_linker_artifact_dir=corrupt_linker,
            target_json=target_json,
            bundle_version="9.9",
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
