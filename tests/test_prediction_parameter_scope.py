"""Clustering overrides must cover nested phases without changing shared models."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.model_pairwise as pairwise_module
from s2and.feature_port import _get_rust_featurizer
from s2and.featurizer import FeaturizationInfo
from s2and.model import (
    Clusterer,
    FastCluster,
    _clusterer_with_prediction_params,
    _get_altered_presplit_cache_entry,
    _model_presplit_cache_fingerprint,
    _put_altered_presplit_cache_entry,
)
from s2and.runtime import build_runtime_context
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset
from tests.model_helpers import ConstantDistanceClassifier


def clusterer(eps: float = 0.5) -> Clusterer:
    """Create a small deterministic clustering model."""
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.6),
        cluster_model=FastCluster(eps=eps),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )


def partition(clusters: dict[str, list[str]]) -> set[frozenset[str]]:
    """Compare membership independently of generated cluster identifiers."""
    return {frozenset(values) for values in clusters.values()}


@pytest.mark.parametrize("route", ["python", "helper", "arrow", "prepared"])
def test_full_prediction_override_matches_configured_model(route: str, tmp_path: Path) -> None:
    """All public full-block entrances use overrides and preserve the source."""
    dataset = build_dummy_dataset("prediction_params")
    model = clusterer()
    expected = clusterer(0.8)
    blocks = {"a": ["0", "1"], "b": ["2", "3"]}
    original_estimator = model.cluster_model
    original_estimator.labels_ = np.array([17])
    arrow_training = None
    if route in {"arrow", "prepared"}:
        arrow_training = build_arrow_training_dataset(dataset, tmp_path / "arrow")

    def run(current: Clusterer, params: dict[str, Any] | None = None):
        if route == "python":
            return current.predict(blocks, dataset, cluster_model_params=params)[0]
        if route == "helper":
            return current.predict_helper(blocks, dataset, cluster_model_params=params)[0]
        assert arrow_training is not None
        assert arrow_training.arrow_dataset is not None
        if route == "arrow":
            return current.predict_from_arrow(
                blocks, arrow_training.arrow_dataset, cluster_model_params=params, name_tuples=frozenset()
            )[0]
        native = _get_rust_featurizer(arrow_training)
        return current.predict_from_rust_featurizer(blocks, native, cluster_model_params=params)[0]

    try:
        assert (
            partition(run(model, {"eps": 0.8}))
            == partition(run(expected))
            == {frozenset({"0", "1"}), frozenset({"2", "3"})}
        )
        assert len(run(model)) == 4
        assert model.cluster_model is original_estimator
        assert original_estimator.eps == 0.5
        np.testing.assert_array_equal(original_estimator.labels_, [17])
    finally:
        if arrow_training is not None:
            assert arrow_training.arrow_dataset is not None
            arrow_training.arrow_dataset.close()


def test_subblocked_override_reaches_initial_attachment_and_residuals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initial-only passes must share bulk EPS, including their residual fits."""
    dataset = build_dummy_dataset("prediction_params_initial")
    monkeypatch.setattr(
        Clusterer,
        "_build_subblocked_block_dict",
        lambda *args, **kwargs: {"full": ["0", "1"], "initial": ["2", "3"]},
    )
    monkeypatch.setattr(
        Clusterer,
        "_partition_subblocked_first_name_groups",
        lambda *args, **kwargs: ({"full": ["0", "1"]}, {"initial": ["2", "3"]}, False),
    )
    blocks = {"block": ["0", "1", "2", "3"]}
    model = clusterer()
    result, _ = model.predict(blocks, dataset, batching_threshold=2, cluster_model_params={"eps": 0.8})
    reference, _ = clusterer(0.8).predict(blocks, dataset, batching_threshold=2)
    assert partition(result) == partition(reference) == {frozenset({"0", "1", "2", "3"})}

    # Force every attachment into the real residual-clustering callback.
    monkeypatch.setattr(
        Clusterer, "_best_incremental_cluster", lambda *args, **kwargs: (None, float("inf"), float("inf"))
    )
    residual, _ = model.predict(blocks, dataset, batching_threshold=2, cluster_model_params={"eps": 0.8})
    reference, _ = clusterer(0.8).predict(blocks, dataset, batching_threshold=2)
    assert partition(residual) == partition(reference) == {frozenset({"0", "1"}), frozenset({"2", "3"})}
    assert model.cluster_model.eps == 0.5


def test_request_cache_is_owned_by_original_and_separates_effective_params() -> None:
    """The first overridden request must not discard its lazy presplit cache."""
    model = clusterer()
    first = _clusterer_with_prediction_params(model, {"eps": 0.8})
    first_key = _model_presplit_cache_fingerprint(first)
    _put_altered_presplit_cache_entry(first, first_key, [["a", "b"]])
    repeated = _clusterer_with_prediction_params(model, {"eps": 0.8})
    different = _clusterer_with_prediction_params(model, {"eps": 0.2})
    assert _get_altered_presplit_cache_entry(repeated, _model_presplit_cache_fingerprint(repeated)) == (("a", "b"),)
    assert _get_altered_presplit_cache_entry(different, _model_presplit_cache_fingerprint(different)) is None
    assert first.classifier is model.classifier
    assert first.subblocking_graph_config is model.subblocking_graph_config


def test_singleton_subblocks_still_use_request_threshold() -> None:
    """Two individually singleton blocks can interact in initial attachment."""
    dataset = build_dummy_dataset("prediction_params_singletons")
    dataset.signatures["1"] = dataset.signatures["1"]._replace(
        author_info_first="a",
        author_info_first_normalized_without_apostrophe="a",
    )
    blocks = {"full": ["0"], "initial": ["1"]}
    model = clusterer(0.8)
    actual, _ = model.predict(blocks, dataset, batching_threshold=10, cluster_model_params={"eps": 0.1})
    expected, _ = clusterer(0.1).predict(blocks, dataset, batching_threshold=10)
    assert partition(actual) == partition(expected) == {frozenset({"0"}), frozenset({"1"})}
    baseline, _ = model.predict(blocks, dataset, batching_threshold=10)
    assert partition(baseline) == {frozenset({"0", "1"})}


def test_arrow_subblocked_override_reaches_nested_residual_prediction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Arrow initial passes retain overrides when residuals re-enter prediction."""
    dataset = build_dummy_dataset("prediction_params_arrow_nested")
    dataset.specter_embeddings = {str(paper_id): np.ones(768, dtype=np.float32) for paper_id in dataset.papers}
    training = build_arrow_training_dataset(dataset, tmp_path / "arrow")
    arrow = training.arrow_dataset
    assert arrow is not None
    monkeypatch.setattr(
        Clusterer,
        "_build_arrow_subblocked_block_dict",
        lambda *args, **kwargs: {"full": ["0", "1"], "initial": ["2", "3"]},
    )
    monkeypatch.setattr(
        Clusterer,
        "_partition_subblocked_first_name_groups",
        lambda *args, **kwargs: ({"full": ["0", "1"]}, {"initial": ["2", "3"]}, False),
    )
    observed: list[float] = []

    def complete_with_residuals(self: Clusterer, signatures: list[str], arrow_dataset: Any, **kwargs: Any):
        observed.append(self.cluster_model.eps)
        residual = self._make_residual_clusterer(
            dataset,
            partial_supervision={},
            runtime_context=kwargs["runtime_context"],
            total_ram_bytes=None,
            arrow_dataset=arrow_dataset,
        )(signatures)
        seeds: dict[str, list[str]] = {}
        for signature_id, component in kwargs["cluster_seeds_require"].items():
            seeds.setdefault(str(component), []).append(signature_id)
        return {"clusters": {**seeds, **{f"residual_{key}": values for key, values in residual.items()}}}

    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", complete_with_residuals)
    model = clusterer()
    try:
        actual, _ = model.predict_from_arrow(
            {"block": ["0", "1", "2", "3"]},
            arrow,
            batching_threshold=2,
            cluster_model_params={"eps": 0.8},
            name_tuples=frozenset(),
        )
        assert partition(actual) == {frozenset({"0", "1"}), frozenset({"2", "3"})}
        assert observed == [0.8]
        assert model.cluster_model.eps == 0.5
    finally:
        arrow.close()


def test_arrow_altered_presplit_uses_override_and_reuses_original_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Overrides precede real presplitting, whose cache survives the request."""
    dataset = build_dummy_dataset("prediction_params_arrow_altered")
    dataset.specter_embeddings = {str(paper_id): np.ones(768, dtype=np.float32) for paper_id in dataset.papers}
    training = build_arrow_training_dataset(dataset, tmp_path / "arrow")
    arrow = training.arrow_dataset
    assert arrow is not None
    original_predict = Clusterer._predict_from_arrow
    presplit_thresholds: list[float] = []

    def stop_after_presplit(self: Clusterer, *args: Any, **kwargs: Any):
        if not kwargs["needs_subblocking"]:
            presplit_thresholds.append(self.cluster_model.eps)
            return original_predict(self, *args, **kwargs)
        clusters: dict[str, list[str]] = {}
        for signature_id, component in kwargs["prediction_cluster_seeds_require"].items():
            clusters.setdefault(str(component), []).append(signature_id)
        return clusters, None

    monkeypatch.setattr(Clusterer, "_predict_from_arrow", stop_after_presplit)
    model = clusterer()

    def run(eps: float):
        return model.predict_from_arrow(
            {"block": ["0", "1", "2"]},
            arrow,
            batching_threshold=2,
            cluster_model_params={"eps": eps},
            name_tuples=frozenset(),
            cluster_seeds_require={"0": "claimed", "1": "claimed", "2": "claimed"},
            altered_cluster_signatures=["0"],
        )[0]

    try:
        assert partition(run(0.8)) == {frozenset({"0", "1", "2"})}
        assert partition(run(0.8)) == {frozenset({"0", "1", "2"})}
        assert partition(run(0.1)) == {frozenset({"0"}), frozenset({"1"}), frozenset({"2"})}
        assert presplit_thresholds == [0.8, 0.1]
        assert model.cluster_model.eps == 0.5
    finally:
        arrow.close()


@pytest.mark.parametrize("preserve_input", [None, True, False])
def test_nested_precomputed_prediction_retains_explicit_input_policy(
    preserve_input: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored matrices stay intact by default; explicit ownership survives nesting."""
    dataset = build_dummy_dataset("prediction_params_matrix")
    params: dict[str, Any] = {"eps": 0.8}
    if preserve_input is not None:
        params["preserve_input"] = preserve_input
    model = _clusterer_with_prediction_params(clusterer(), params)
    distances = np.array([0.2, 0.7, 0.8, 0.5, 0.9, 0.6], dtype=np.float64)
    before = distances.copy()
    observed: list[bool] = []
    linkage = pairwise_module.linkage

    def checked_linkage(*args: Any, **kwargs: Any) -> np.ndarray:
        observed.append(kwargs["preserve_input"])
        return linkage(*args, **kwargs)

    monkeypatch.setattr(pairwise_module, "linkage", checked_linkage)
    model.predict_helper({"block": ["0", "1", "2", "3"]}, dataset, dists={"block": distances})
    assert observed == [True if preserve_input is None else preserve_input]
    if preserve_input is not False:
        np.testing.assert_array_equal(distances, before)


def test_failure_and_recorded_cluster_bypass_preserve_original() -> None:
    """Failed overrides cannot persist, and recorded predictions ignore them."""
    dataset = build_dummy_dataset("prediction_params_failure")
    model = clusterer()
    with pytest.raises(ValueError):
        model.predict_helper({"a": ["0", "1"]}, dataset, cluster_model_params={"invalid": 1})
    assert model.cluster_model.eps == 0.5
    expected, _ = model.predict({"a": ["0", "1"]}, dataset, use_s2_clusters=True)
    actual, _ = model.predict({"a": ["0", "1"]}, dataset, use_s2_clusters=True, cluster_model_params={"invalid": 1})
    assert actual == expected


def test_each_block_fits_a_fresh_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request prototype is never fitted or reused as a fitted block model."""
    dataset = build_dummy_dataset("prediction_params_clone")
    fitted: list[FastCluster] = []
    fit = FastCluster.fit

    def checked_fit(self: FastCluster, distances: np.ndarray) -> FastCluster:
        assert self.labels_ is None
        fitted.append(self)
        return fit(self, distances)

    monkeypatch.setattr(FastCluster, "fit", checked_fit)
    model = clusterer()
    model.predict_helper(
        {"a": ["0", "1"], "b": ["2", "3"]},
        dataset,
        dists={"a": np.array([0.6]), "b": np.array([0.6])},
        cluster_model_params={"eps": 0.8},
        runtime_context=build_runtime_context("params_clone", backend="python"),
    )
    assert len(fitted) == 2 and fitted[0] is not fitted[1]
    assert model.cluster_model.labels_ is None
