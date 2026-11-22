"""Caller supervision is finite, undirected, and validated before prediction."""

import numpy as np
import pytest

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.model_helpers import ConstantDistanceClassifier


def _clusterer() -> Clusterer:
    return Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )


def _assert_conflicting_supervision_is_rejected(method):
    """Even empty requests must not silently discard contradictory caller pairs."""
    values = (0.0, 10000.0)
    entries = [(("paper-a", "paper-b"), values[0]), (("paper-b", "paper-a"), values[1])]
    partial = dict(entries)
    original = list(partial.items())
    blocks = [] if "incremental" in method else {}
    # No usable dataset/featurizer: this must fail at the supervision boundary,
    # before leases, feature extraction, subblocking, or candidate selection.
    with pytest.raises(ValueError, match="partial_supervision has conflicting distances"):
        getattr(_clusterer(), method)(blocks, None, partial_supervision=partial)
    assert list(partial.items()) == original


def test_normalization_distinguishes_soft_distances_from_hard_values():
    from s2and.supervision import normalize_partial_supervision

    with pytest.raises(ValueError, match="too close to a hard value"):
        normalize_partial_supervision({("a", "b"): -5e-9})
    with pytest.raises(ValueError, match="too close to a hard value"):
        normalize_partial_supervision({("a", "b"): 10000.0 - 5e-7})

    distances = {("a", "b"): 0, ("a", "c"): 1e-6, ("a", "d"): 10000.0 - 1e-6, ("a", "e"): 10000.0}
    assert normalize_partial_supervision(distances) == distances

    value = float("nan")

    partial = {("paper-a", "paper-b"): value}
    with pytest.raises(ValueError, match="partial_supervision") as error:
        normalize_partial_supervision(partial)
    assert "paper-a" in str(error.value) and "paper-b" in str(error.value)
    assert partial[("paper-a", "paper-b")] is value


def test_normalization_accepts_equal_numeric_values_and_owns_its_result():
    from s2and.supervision import normalize_partial_supervision

    partial = {("b", "a"): 0, ("a", "b"): -0.0, ("c", "a"): -1.0}
    original = partial.copy()
    normalized = normalize_partial_supervision(partial)
    assert normalized == {("a", "b"): 0, ("a", "c"): -1.0}
    normalized[("a", "b")] = 0.7
    assert partial == original
    assert normalize_partial_supervision(None) == {}


def test_arrow_incremental_boundary_rejects_id_collisions_before_loading_data():
    partial = {(1, 2): 0.0, ("2", "1"): 10000.0}
    original = list(partial.items())
    with pytest.raises(ValueError, match="conflicting distances"):
        _clusterer().predict_incremental_from_arrow([], None, partial_supervision=partial)
    assert list(partial.items()) == original


@pytest.mark.parametrize("route", ["helper", "full"])
def test_python_prediction_normalizes_supervision_once_per_request(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested blocks reuse validated overrides without caching later requests."""
    import s2and.model as model_module
    from tests.helpers import build_dummy_dataset

    _assert_conflicting_supervision_is_rejected("predict_helper" if route == "helper" else "predict")
    dataset = build_dummy_dataset("supervision_normalization")
    clusterer = _clusterer()
    clusterer.use_default_constraints_as_supervision = False
    blocks = {"first": ["1", "0"], "second": ["3", "2"]}
    partial = {("1", "0"): 0.2, ("0", "1"): 0.2, ("3", "2"): 0.8}
    original = list(partial.items())
    normalization_calls = []
    normalize = model_module.normalize_partial_supervision

    def record_normalization(supervision, **kwargs):
        normalization_calls.append(supervision)
        return normalize(supervision, **kwargs)

    monkeypatch.setattr(model_module, "normalize_partial_supervision", record_normalization)
    method = clusterer.predict_helper if route == "helper" else clusterer.predict

    clusters, _ = method(blocks, dataset, partial_supervision=partial)

    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3"}),
    }
    assert list(partial.items()) == original
    assert len(normalization_calls) == 1

    partial[("3", "2")] = 0.2
    clusters, _ = method(blocks, dataset, partial_supervision=partial)

    assert {frozenset(members) for members in clusters.values()} == {frozenset({"0", "1"}), frozenset({"2", "3"})}
    assert len(normalization_calls) == 2
    assert partial[("3", "2")] == 0.2


@pytest.mark.parametrize("override", ["subclass", "instance"])
def test_bulk_prediction_preserves_predict_helper_overrides(override, monkeypatch):
    """The validation fast path must preserve existing custom helper dispatch."""
    from types import MethodType

    from tests.helpers import build_dummy_dataset

    calls = []

    def record_prediction(self, block_dict, dataset, **kwargs):
        calls.append(tuple(block_dict))
        return Clusterer.predict_helper(self, block_dict, dataset, **kwargs)

    class CustomClusterer(Clusterer):
        predict_helper = record_prediction

    clusterer_type = CustomClusterer if override == "subclass" else Clusterer
    clusterer = clusterer_type(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    if override == "instance":
        monkeypatch.setattr(clusterer, "predict_helper", MethodType(record_prediction, clusterer))
    dataset = build_dummy_dataset("supervision_helper_override")
    blocks = {"first": ["1", "0"], "second": ["3", "2"]}

    clusters, _ = clusterer.predict(blocks, dataset, partial_supervision={("1", "0"): 0.2, ("3", "2"): 0.2})

    assert calls == [("first", "second")]
    assert {frozenset(members) for members in clusters.values()} == {frozenset({"0", "1"}), frozenset({"2", "3"})}


@pytest.mark.parametrize(
    ("method", "value"),
    [
        ("make_distance_matrices", 0.2),
        ("make_distance_matrices_from_rust_featurizer", 10000.0),
        ("predict_from_rust_featurizer", 0.2),
        ("predict_from_arrow", 10000.0),
        ("predict_incremental_from_arrow", 0.0),
        ("predict_incremental_from_arrow", 0.2),
        ("predict_incremental_from_arrow", 10000.0),
    ],
)
def test_equal_reverse_supervision_matches_single_entry_on_real_public_routes(tmp_path, method, value):
    """Duplicates and input order cannot alter actual matrices or predictions."""
    from s2and.feature_port import _get_rust_featurizer
    from tests.helpers import build_arrow_training_dataset
    from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker

    if method != "predict_incremental_from_arrow" or value == 0.0:
        _assert_conflicting_supervision_is_rejected(method)
    incremental = "incremental" in method
    seeds = {"0": "seed"} if incremental else {}
    dataset = _dataset([("Ada", "", "Smith")] * 2, seeds)
    clusterer = _clusterer()
    data_arg = dataset
    extra = {}
    if "arrow" in method or "rust" in method:
        rust_dataset = build_arrow_training_dataset(dataset, tmp_path / "arrow")
        data_arg = rust_dataset.arrow_dataset if "arrow" in method else _get_rust_featurizer(rust_dataset)
    if "arrow" in method:
        extra = {"name_tuples": frozenset(), "cluster_seeds_require": seeds}
    if method == "predict_incremental_from_arrow":
        attach_tiny_incremental_linker(clusterer, tmp_path / "linker")

    results = []
    for ids, partial in [
        (["0", "1"], {("0", "1"): value}),
        (["1", "0"], {("1", "0"): value, ("0", "1"): value}),
    ]:
        original = list(partial.items())
        blocks = ids if incremental else {"one": ids}
        result = getattr(clusterer, method)(blocks, data_arg, partial_supervision=partial, **extra)
        assert list(partial.items()) == original
        if method.startswith("make_distance"):
            assert result["one"].shape == (1,)
            np.testing.assert_allclose(result["one"], [value])
        else:
            clusters = result["clusters"] if incremental else result[0]
            results.append({frozenset(members) for members in clusters.values()})
    if results:
        assert all(result == results[0] for result in results)
        if value in (0.0, 0.2):
            assert results[0] == {frozenset({"0", "1"})}
        elif value == 10000:
            assert results[0] == {frozenset({"0"}), frozenset({"1"})}
