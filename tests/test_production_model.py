from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.model import _ensure_lightgbm_fitted, _selected_feature_indices
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import NativeLightGBMBinaryClassifier, _config_choice, load_production_model
from s2and.serialization import load_pickle_with_verified_label_encoder_compat
from tests.helpers import import_s2and_rust


def _load_dummy_inference_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode="inference",
        load_name_counts=True,
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


def test_native_production_bundle_loads_as_mutable_clusterer() -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")

    assert isinstance(clusterer.classifier, NativeLightGBMBinaryClassifier)
    assert isinstance(clusterer.nameless_classifier, NativeLightGBMBinaryClassifier)
    assert Path(clusterer.incremental_linker_artifact_dir).name == "incremental_linker"
    assert clusterer.production_model_bundle_version == "1.21"

    clusterer.n_jobs = 7
    clusterer.cluster_model.eps = 0.5

    assert clusterer.n_jobs == 7
    assert clusterer.classifier.n_jobs == 7
    assert clusterer.nameless_classifier.n_jobs == 7
    assert clusterer.cluster_model.eps == 0.5


def test_native_lightgbm_set_params_rejects_unknown_params() -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")

    with pytest.raises(ValueError, match="Invalid parameter"):
        clusterer.classifier.set_params(learning_rate=0.1)


def test_native_lightgbm_deepcopy_does_not_require_model_path(tmp_path: Path) -> None:
    clusterer = load_production_model("s2and/data/production_model_v1.21")
    classifier = clusterer.classifier
    features = np.zeros((2, classifier.n_features_in_), dtype=np.float64)
    classifier.model_path = str(tmp_path / "missing_model.txt")

    copied = copy.deepcopy(classifier)

    np.testing.assert_allclose(copied.predict_proba(features), classifier.predict_proba(features))
    assert copied.model_path == classifier.model_path


def test_native_pairwise_models_match_v12_pickle_fixture() -> None:
    native_clusterer = load_production_model("s2and/data/production_model_v1.21")
    legacy_clusterer = load_pickle_with_verified_label_encoder_compat("s2and/data/production_model_v1.2.pickle")[
        "clusterer"
    ]
    _ensure_lightgbm_fitted(legacy_clusterer.classifier)
    _ensure_lightgbm_fitted(legacy_clusterer.nameless_classifier)

    rng = np.random.default_rng(921)
    main_width = len(_selected_feature_indices(legacy_clusterer.featurizer_info))
    nameless_width = len(_selected_feature_indices(legacy_clusterer.nameless_featurizer_info))
    main_features = rng.normal(size=(8, main_width))
    nameless_features = rng.normal(size=(8, nameless_width))

    np.testing.assert_allclose(
        native_clusterer.classifier.predict_proba(main_features),
        legacy_clusterer.classifier.predict_proba(main_features),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        native_clusterer.nameless_classifier.predict_proba(nameless_features),
        legacy_clusterer.nameless_classifier.predict_proba(nameless_features),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_native_clusterer_predict_matches_v12_pickle(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    if backend == "rust":
        rust_available, rust_error = import_s2and_rust(required_method="from_dataset")
        if not rust_available:
            pytest.skip(f"Rust runtime unavailable: {rust_error!r}")

    monkeypatch.setenv("S2AND_BACKEND", backend)

    for batching_threshold in (None, 7):
        native_clusterer = _prepare_prediction_clusterer(
            load_production_model("s2and/data/production_model_v1.21", require_incremental_linker=False)
        )
        legacy_clusterer = _prepare_prediction_clusterer(
            load_pickle_with_verified_label_encoder_compat("s2and/data/production_model_v1.2.pickle")["clusterer"]
        )

        assert _predict_dummy_block(native_clusterer, batching_threshold=batching_threshold) == _predict_dummy_block(
            legacy_clusterer,
            batching_threshold=batching_threshold,
        )


def test_native_clusterer_runtime_config_matches_v12_pickle() -> None:
    native_clusterer = load_production_model("s2and/data/production_model_v1.21")
    legacy_clusterer = load_pickle_with_verified_label_encoder_compat("s2and/data/production_model_v1.2.pickle")[
        "clusterer"
    ]

    assert type(native_clusterer.cluster_model) is type(legacy_clusterer.cluster_model)
    assert native_clusterer.cluster_model.linkage == legacy_clusterer.cluster_model.linkage
    assert native_clusterer.cluster_model.eps == 0.65
    assert native_clusterer.featurizer_info.features_to_use == legacy_clusterer.featurizer_info.features_to_use
    assert native_clusterer.featurizer_info.featurizer_version == legacy_clusterer.featurizer_info.featurizer_version
    assert (
        native_clusterer.nameless_featurizer_info.features_to_use
        == legacy_clusterer.nameless_featurizer_info.features_to_use
    )
    assert (
        native_clusterer.nameless_featurizer_info.featurizer_version
        == legacy_clusterer.nameless_featurizer_info.featurizer_version
    )
    assert native_clusterer.best_params == {**legacy_clusterer.best_params, "eps": 0.65}
    assert native_clusterer.batch_size == legacy_clusterer.batch_size
    assert native_clusterer.dont_merge_cluster_seeds == legacy_clusterer.dont_merge_cluster_seeds
    assert (
        native_clusterer.use_default_constraints_as_supervision
        == legacy_clusterer.use_default_constraints_as_supervision
    )
    assert native_clusterer.use_cache == legacy_clusterer.use_cache
    assert native_clusterer.n_iter == legacy_clusterer.n_iter
    assert native_clusterer.random_state == legacy_clusterer.random_state

    assert getattr(native_clusterer, "suppress_orcid", False) == getattr(legacy_clusterer, "suppress_orcid", False)
    assert native_clusterer._incremental_experiment_config() == legacy_clusterer._incremental_experiment_config()


def test_pairwise_stage_finalizes_into_loadable_production_bundle(tmp_path: Path) -> None:
    source_bundle = Path("s2and/data/production_model_v1.21")
    source_clusterer = load_production_model(source_bundle)
    output_bundle = tmp_path / "production_model_v9.9"

    pairwise_summary = write_pairwise_production_bundle(
        source_clusterer,
        output_bundle,
        bundle_version="9.9",
        source_model_version="9.9",
        pairwise_training_config={"test": True},
        pairwise_training_summary={"rows": 1},
    )

    assert pairwise_summary.bundle_status == "pairwise_only"
    clusterer_config = json.loads((output_bundle / "clusterer.json").read_text(encoding="utf-8"))
    assert "incremental_phase_a_pair_batch_target_multiple" not in clusterer_config
    with pytest.raises(FileNotFoundError, match="pairwise-only"):
        load_production_model(output_bundle)

    pairwise_only_clusterer = load_production_model(output_bundle, require_incremental_linker=False)
    assert pairwise_only_clusterer.production_model_bundle_status == "pairwise_only"

    final_summary = finalize_production_bundle(
        pairwise_bundle_dir=output_bundle,
        output_bundle_dir=output_bundle,
        incremental_linker_artifact_dir=source_bundle / "incremental_linker",
        target_json=source_bundle / "reproducibility" / "incremental_linker_training_target.json",
        bundle_version="9.9",
        pairwise_model_version="9.9",
        incremental_linker_version="9.9",
    )

    assert final_summary.bundle_status == "complete"
    loaded = load_production_model(output_bundle)
    assert loaded.production_model_bundle_version == "9.9"
    assert Path(loaded.incremental_linker_artifact_dir) == output_bundle / "incremental_linker"

    with pytest.raises(ValueError, match="existing incremental linker artifacts"):
        write_pairwise_production_bundle(
            source_clusterer,
            output_bundle,
            bundle_version="9.9",
            source_model_version="9.9",
        )


def test_production_model_config_choice_rejects_unknown_literal() -> None:
    with pytest.raises(ValueError, match="incremental_seed_score_mode"):
        _config_choice(
            {"incremental_seed_score_mode": "unsupported"},
            "incremental_seed_score_mode",
            allowed=frozenset({"mean", "min", "mean_min_hybrid"}),
        )
