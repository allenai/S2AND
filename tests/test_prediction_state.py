"""Regression coverage for prediction-owned synthetic seeds on shared objects."""

from collections.abc import Callable

import numpy as np
import pytest

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.prediction_state import PredictionState
from s2and.runtime import build_runtime_context
from tests.helpers import tiny_name_counts_index
from tests.model_helpers import ConstantDistanceClassifier


@pytest.fixture
def shared_prediction_objects() -> tuple[Clusterer, ANDData]:
    """Build real feature records with unrelated, preexisting seed state."""
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        cluster_seeds={"0": {"1": "require"}},
        name="prediction_state_isolation",
        name_counts_index=tiny_name_counts_index(),
    )
    dataset.altered_cluster_signatures = ["0"]
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=ConstantDistanceClassifier(0.0),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    return clusterer, dataset


def _assert_dataset_unchanged(dataset: ANDData) -> Callable[[], None]:
    """Capture both contents and identity of the reusable seed collections."""
    require = dataset.cluster_seeds_require
    disallow = dataset.cluster_seeds_disallow
    altered = dataset.altered_cluster_signatures
    require_contents = dict(require)
    disallow_contents = set(disallow)
    altered_contents = list(altered)

    def check() -> None:
        assert dataset.cluster_seeds_require is require
        assert dataset.cluster_seeds_require == require_contents
        assert dataset.cluster_seeds_disallow is disallow
        assert dataset.cluster_seeds_disallow == disallow_contents
        assert dataset.altered_cluster_signatures is altered
        assert dataset.altered_cluster_signatures == altered_contents

    return check


def test_predict_helper_replaces_source_groups_and_disallows_in_scoring(shared_prediction_objects) -> None:
    clusterer, dataset = shared_prediction_objects
    clusterer.use_default_constraints_as_supervision = True
    dataset.cluster_seeds_require = {"3": "source_a", "4": "source_b"}
    dataset.cluster_seeds_disallow = {("3", "4")}
    check = _assert_dataset_unchanged(dataset)

    clusters, _ = clusterer.predict_helper(
        {"allowed": ["3", "4"], "blocked": ["5", "6"]},
        dataset,
        prediction_state=PredictionState(
            cluster_seeds_require={"3": "request", "4": "request"},
            cluster_seeds_disallow={("5", "6")},
        ),
        runtime_context=build_runtime_context("request_disallow_test", backend="python"),
    )
    assert {frozenset(group) for group in clusters.values()} == {
        frozenset({"3", "4"}),
        frozenset({"5"}),
        frozenset({"6"}),
    }
    check()


def test_predict_helper_uses_request_require_groups_with_precomputed_distances(
    shared_prediction_objects: tuple[Clusterer, ANDData],
) -> None:
    """Request groups override source constraints in scoring and posthoc merging."""
    clusterer, dataset = shared_prediction_objects
    clusterer.use_default_constraints_as_supervision = True
    dataset.cluster_seeds_require = {"3": "source_a", "4": "source_b"}
    dataset.cluster_seeds_disallow = {("3", "4")}
    check = _assert_dataset_unchanged(dataset)
    clusters, _ = clusterer.predict_helper(
        {"block": ["3", "4"]},
        dataset,
        dists={"block": np.array([1.0])},
        prediction_state=PredictionState(cluster_seeds_require={"3": "request", "4": "request"}),
        runtime_context=build_runtime_context("request_require_test", backend="python"),
    )
    assert list(clusters.values()) == [["3", "4"]]
    check()
