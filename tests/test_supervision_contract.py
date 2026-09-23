"""Caller supervision is finite, undirected, and validated before prediction."""

from copy import deepcopy

import numpy as np
import pytest

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.model_helpers import ConstantDistanceClassifier

PUBLIC_METHODS = (
    "predict",
    "predict_helper",
    "make_distance_matrices",
    "predict_from_rust_featurizer",
    "make_distance_matrices_from_rust_featurizer",
    "predict_from_arrow",
    "predict_incremental",
    "predict_incremental_from_arrow",
)


def _clusterer() -> Clusterer:
    return Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )


@pytest.mark.parametrize("method", PUBLIC_METHODS)
@pytest.mark.parametrize("values", [(0.0, 10000.0), (0.2, 0.3)])
@pytest.mark.parametrize("reverse_entries", [False, True])
def test_public_boundaries_reject_conflicting_reverse_entries_before_work(method, values, reverse_entries):
    """Even empty requests must not silently discard contradictory caller pairs."""
    entries = [(("paper-a", "paper-b"), values[0]), (("paper-b", "paper-a"), values[1])]
    partial = dict(reversed(entries) if reverse_entries else entries)
    original = list(partial.items())
    blocks = [] if "incremental" in method else {}
    # No usable dataset/featurizer: this must fail at the supervision boundary,
    # before leases, feature extraction, subblocking, or candidate selection.
    with pytest.raises(ValueError, match="partial_supervision") as error:
        getattr(_clusterer(), method)(blocks, None, partial_supervision=partial)
    message = str(error.value)
    assert "paper-a" in message and "paper-b" in message
    assert all(repr(value) in message for value in values)
    assert list(partial.items()) == original


@pytest.mark.parametrize("method", PUBLIC_METHODS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_public_boundaries_reject_nonfinite_distances_before_work(method, value):
    partial = {("paper-a", "paper-b"): value}
    blocks = [] if "incremental" in method else {}
    with pytest.raises(ValueError, match="partial_supervision") as error:
        getattr(_clusterer(), method)(blocks, None, partial_supervision=partial)
    assert "paper-a" in str(error.value) and "paper-b" in str(error.value)
    assert partial[("paper-a", "paper-b")] is value


@pytest.mark.parametrize("value", [-1.0, -0.0, 0.2, 1.0, 10000.0])
def test_normalization_deduplicates_equal_reverse_values_and_owns_its_result(value):
    from s2and.supervision import normalize_partial_supervision

    partial = {("b", "a"): value, ("a", "b"): value, ("c", "a"): 0}
    original = deepcopy(partial)
    normalized = normalize_partial_supervision(partial)
    assert normalized == {("a", "b"): value, ("a", "c"): 0}
    assert partial == original
    assert normalized is not partial
    normalized[("a", "b")] = 0.7
    assert partial == original


def test_normalization_accepts_equal_numeric_values_and_empty_input():
    from s2and.supervision import normalize_partial_supervision

    assert normalize_partial_supervision({("b", "a"): 0, ("a", "b"): -0.0}) == {("a", "b"): 0}
    empty = {}
    assert normalize_partial_supervision(None) == {}
    assert normalize_partial_supervision(empty) == {}
    assert normalize_partial_supervision(empty) is not empty


@pytest.mark.parametrize("reverse_entries", [False, True])
@pytest.mark.parametrize("second_pair", [("1", "2"), ("2", "1")])
def test_arrow_id_normalization_checks_conflicts_before_collapsing_keys(reverse_entries, second_pair):
    from s2and.supervision import normalize_partial_supervision

    entries = [((1, 2), 0.0), (second_pair, 10000.0)]
    partial = dict(reversed(entries) if reverse_entries else entries)
    original = list(partial.items())
    with pytest.raises(ValueError, match="partial_supervision"):
        normalize_partial_supervision(partial, stringify_ids=True)
    assert list(partial.items()) == original
    assert normalize_partial_supervision({(1, 2): 0.2, ("2", "1"): 0.2}, stringify_ids=True) == {("1", "2"): 0.2}


@pytest.mark.parametrize("second_pair", [("1", "2"), ("2", "1")])
def test_arrow_incremental_boundary_rejects_id_collisions_before_loading_data(second_pair):
    partial = {(1, 2): 0.0, second_pair: 10000.0}
    original = list(partial.items())
    with pytest.raises(ValueError, match="partial_supervision"):
        _clusterer().predict_incremental_from_arrow([], None, partial_supervision=partial)
    assert list(partial.items()) == original


@pytest.mark.parametrize("method", PUBLIC_METHODS)
@pytest.mark.parametrize("value", [0.0, 0.2, 10000.0])
def test_equal_reverse_supervision_matches_single_entry_on_real_public_routes(tmp_path, method, value):
    """Duplicates and input order cannot alter actual matrices or predictions."""
    from s2and.feature_port import _get_rust_featurizer
    from s2and.incremental_linking.artifact import load_incremental_linking_artifact, save_incremental_linking_artifact
    from tests.helpers import build_arrow_training_dataset
    from tests.promoted_linking_helpers import (
        build_tiny_promoted_booster,
        synthetic_pairwise_bundle_binding,
        tiny_logistic_gate_config,
    )
    from tests.test_subblocked_seed_boundaries import _dataset

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
        booster, _ = build_tiny_promoted_booster()
        save_incremental_linking_artifact(
            booster,
            tmp_path / "linker",
            gate_config=tiny_logistic_gate_config(),
            target_spec={},
            pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
        )
        clusterer.incremental_linker_artifact = load_incremental_linking_artifact(tmp_path / "linker")

    results = []
    for ids, partial in [
        (["0", "1"], {("0", "1"): value}),
        (["1", "0"], {("1", "0"): value}),
        (["0", "1"], {("1", "0"): value, ("0", "1"): value}),
        (["1", "0"], {("0", "1"): value, ("1", "0"): value}),
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
        if value == 0:
            assert results[0] == {frozenset({"0", "1"})}
        elif value == 10000:
            assert results[0] == {frozenset({"0"}), frozenset({"1"})}
