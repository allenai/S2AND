"""Regression coverage for prediction-owned synthetic seeds on shared objects."""

from collections.abc import Callable
from typing import Any

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


def _run_subblocks(
    clusterer: Clusterer, dataset: ANDData, *, seed: str, queries: list[str], cluster: str
) -> dict[str, list[str]]:
    """Run real incremental inference with one synthetic seed component."""
    return clusterer._predict_subblocked_single_letter_incremental_groups(
        {f"subblock_{index}": [query] for index, query in enumerate(queries)},
        pred_clusters={cluster: [seed]},
        dataset=dataset,
        partial_supervision={},
        runtime_context=build_runtime_context("prediction_state_test", backend="python"),
    )


def test_subblock_predictions_interleave_without_sharing_seeds(
    shared_prediction_objects: tuple[Clusterer, ANDData], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nested request must not see or replace the outer synthetic seeds."""
    clusterer, dataset = shared_prediction_objects
    check = _assert_dataset_unchanged(dataset)
    original = clusterer._predict_incremental_python
    nested = False
    observed: list[dict[str, int | str]] = []
    scored_rows: list[int] = []
    score = clusterer.classifier.predict_proba

    def checked_score(features: np.ndarray) -> np.ndarray:
        check()
        scored_rows.append(len(features))
        return score(features)

    monkeypatch.setattr(clusterer.classifier, "predict_proba", checked_score)

    def interleaved(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal nested
        check()
        state = kwargs["prediction_state"]
        observed.append(dict(state.cluster_seeds_require))
        assert state.altered_cluster_signatures == []
        if not nested:
            nested = True
            assert _run_subblocks(clusterer, dataset, seed="6", queries=["7"], cluster="inner") == {"inner": ["6", "7"]}
            check()
            assert state.cluster_seeds_require == {"3": "outer"}
        return original(*args, **kwargs)

    monkeypatch.setattr(clusterer, "_predict_incremental_python", interleaved)
    expected = {"outer": ["3", "4", "5"]}
    assert _run_subblocks(clusterer, dataset, seed="3", queries=["4", "5"], cluster="outer") == expected
    check()
    assert observed == [{"3": "outer"}, {"6": "inner"}, {"3": "outer", "4": "outer"}]
    assert scored_rows == [1, 1, 2]
    monkeypatch.setattr(clusterer, "_predict_incremental_python", original)
    assert _run_subblocks(clusterer, dataset, seed="3", queries=["4", "5"], cluster="outer") == expected
    check()


def test_failed_subblock_prediction_leaves_shared_seeds_untouched(
    shared_prediction_objects: tuple[Clusterer, ANDData], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after one completed subblock must not leak its accumulated seeds."""
    clusterer, dataset = shared_prediction_objects
    check = _assert_dataset_unchanged(dataset)
    original = clusterer._predict_incremental_python
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        check()
        if calls == 2:
            assert kwargs["prediction_state"].cluster_seeds_require == {"3": "outer", "4": "outer"}
            raise RuntimeError("injected second-subblock failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(clusterer, "_predict_incremental_python", fail_second)
    with pytest.raises(RuntimeError, match="injected second-subblock failure"):
        _run_subblocks(clusterer, dataset, seed="3", queries=["4", "5"], cluster="outer")
    check()
    monkeypatch.setattr(clusterer, "_predict_incremental_python", original)
    assert _run_subblocks(clusterer, dataset, seed="6", queries=["7"], cluster="after_failure") == {
        "after_failure": ["6", "7"]
    }
    check()


@pytest.mark.parametrize("request_disallow", [False, True])
def test_predict_helper_scores_with_request_disallows(
    shared_prediction_objects: tuple[Clusterer, ANDData], request_disallow: bool
) -> None:
    """Both an explicit empty override and a request disallow supersede dataset seeds."""
    clusterer, dataset = shared_prediction_objects
    clusterer.use_default_constraints_as_supervision = True
    dataset.cluster_seeds_disallow = set() if request_disallow else {("3", "4")}
    check = _assert_dataset_unchanged(dataset)
    state = PredictionState(cluster_seeds_disallow={("3", "4")} if request_disallow else set())
    clusters, _ = clusterer.predict_helper(
        {"block": ["3", "4"]},
        dataset,
        prediction_state=state,
        runtime_context=build_runtime_context("request_disallow_test", backend="python"),
    )
    assert {frozenset(group) for group in clusters.values()} == (
        {frozenset({"3"}), frozenset({"4"})} if request_disallow else {frozenset({"3", "4"})}
    )
    check()


@pytest.mark.parametrize("precomputed", [False, True])
def test_predict_helper_uses_request_require_groups(
    shared_prediction_objects: tuple[Clusterer, ANDData], precomputed: bool
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
        dists={"block": np.array([1.0])} if precomputed else None,
        prediction_state=PredictionState(cluster_seeds_require={"3": "request", "4": "request"}),
        runtime_context=build_runtime_context("request_require_test", backend="python"),
    )
    assert list(clusters.values()) == [["3", "4"]]
    check()
