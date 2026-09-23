"""Public Python/Arrow regressions for effective caller and dataset constraints."""

from collections.abc import Mapping

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import load_incremental_linking_artifact, save_incremental_linking_artifact
from s2and.incremental_linking.feature_block_arrow import write_raw_arrow_batch_lookup_indexes
from s2and.model import Clusterer, FastCluster
from scripts.arrow_conversion_helpers import write_raw_planner_arrow_from_anddata
from tests.helpers import write_test_arrow_artifact_manifest
from tests.model_helpers import ConstantDistanceClassifier
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    synthetic_pairwise_bundle_binding,
    tiny_logistic_gate_config,
)
from tests.test_subblocked_seed_boundaries import _dataset, _predict


@pytest.fixture(autouse=True)
def _empty_orcid_priors(monkeypatch):
    # These fixtures contain no ORCIDs and do not need generated Stage 1 priors.
    monkeypatch.setattr("s2and.subblocking._resolved_orcid_prefix_counts", lambda counts: {})


def _partition(clusters: Mapping[object, list[str]]) -> list[list[str]]:
    return sorted(sorted(members) for members in clusters.values())


def _incremental_predictions(tmp_path, dataset, *, enabled, partial_supervision, broadcast="always"):
    """Use real retrieval and scoring with deterministic pair and linker models."""
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=enabled,
    )
    clusterer.incremental_precluster_broadcast_mode = broadcast
    signatures = list(dataset.signatures)
    yield (
        "python",
        _partition(
            clusterer.predict_incremental(signatures, dataset, partial_supervision=partial_supervision)["clusters"]
        ),
    )
    paths = write_raw_planner_arrow_from_anddata(dataset, tmp_path / "arrow", include_specter=True)
    paths, _ = write_raw_arrow_batch_lookup_indexes(paths, tmp_path / "arrow")
    write_test_arrow_artifact_manifest(tmp_path / "arrow", paths)
    booster, _ = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path / "linker",
        gate_config=tiny_logistic_gate_config(),
        target_spec={},
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    clusterer.incremental_linker_artifact = load_incremental_linking_artifact(tmp_path / "linker")
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        yield (
            "arrow",
            _partition(
                clusterer.predict_incremental_from_arrow(
                    signatures,
                    arrow_dataset,
                    cluster_seeds_require=dataset.cluster_seeds_require,
                    cluster_seeds_disallow=dataset.cluster_seeds_disallow,
                    partial_supervision=partial_supervision,
                    name_tuples=frozenset(),
                )["clusters"]
            ),
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("threshold", [None, 2])
@pytest.mark.parametrize("case", ["dataset_disallow", "caller_disallow", "zero_override", "soft_override"])
def test_bulk_uses_effective_constraints_before_real_subblocking(tmp_path, enabled, threshold, case):
    dataset = _dataset([("Ada", "", "Smith")] * 3, {})
    dataset.cluster_seeds_disallow = set() if case == "caller_disallow" else {("0", "1")}
    partial = {
        "dataset_disallow": {},
        "caller_disallow": {("0", "1"): LARGE_DISTANCE},
        "zero_override": {("0", "1"): 0.0},
        "soft_override": {("1", "0"): 0.3},
    }[case]
    disallowed = case == "caller_disallow" or (case == "dataset_disallow" and enabled)
    expected = {
        (False, False): [["0", "1", "2"]],
        (False, True): [["0", "1"], ["2"]],
        (True, False): [["0", "2"], ["1"]],
        (True, True): [["0"], ["1"], ["2"]],
    }[(disallowed, threshold is not None)]
    for route in ["python", "arrow"]:
        clusters = _predict(
            tmp_path / route,
            dataset,
            {"one": ["0", "1", "2"]},
            route,
            threshold=threshold,
            use_default_constraints_as_supervision=enabled,
            partial_supervision=partial,
        )
        assert _partition(clusters) == expected, route


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "case", ["dataset_seed_query", "dataset_query_query", "caller_seed_query", "caller_query_query"]
)
def test_incremental_negative_policy_preserves_seed_anchors(tmp_path, enabled, case):
    size = 3 if case.endswith("query_query") else 2
    dataset = _dataset([("Ada", "", "Smith")] * size, {"0": "claimed"})
    pair = ("1", "2") if case.endswith("query_query") else ("0", "1")
    dataset.cluster_seeds_disallow = {pair} if case.startswith("dataset") else set()
    partial = {pair: LARGE_DISTANCE} if case.startswith("caller") else {}
    disallowed = enabled or case.startswith("caller")
    if not disallowed:
        expected = [list(dataset.signatures)]
    elif case.endswith("query_query"):
        expected = [["0", "1"], ["2"]]
    else:
        expected = [["0"], ["1"]]
    for route, clusters in _incremental_predictions(tmp_path, dataset, enabled=enabled, partial_supervision=partial):
        assert clusters == expected, route
    assert dataset.cluster_seeds_require == {"0": "claimed"}
    assert dataset.cluster_seeds_disallow == ({pair} if case.startswith("dataset") else set())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("override", [0.0, 0.3])
def test_incremental_override_precedes_retrieval(tmp_path, enabled, reverse, override):
    dataset = _dataset([("Ada", "", "Smith")] * 2, {"0": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1")}
    pair = ("1", "0") if reverse else ("0", "1")
    partial = {pair: override}
    for route, clusters in _incremental_predictions(tmp_path, dataset, enabled=enabled, partial_supervision=partial):
        assert clusters == [["0", "1"]], route
    assert dataset.cluster_seeds_disallow == {("0", "1")}
    assert partial == {pair: override}


@pytest.mark.parametrize("enabled", [False, True])
def test_incremental_override_preserves_independent_dataset_negatives(tmp_path, enabled):
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1"), ("0", "2")}
    # Disable classic precluster score broadcasting to isolate each query's
    # constraints; the promoted linker already scores query attachments separately.
    expected = [["0", "1"], ["2"]] if enabled else [["0", "1", "2"]]
    for route, clusters in _incremental_predictions(
        tmp_path, dataset, enabled=enabled, partial_supervision={("1", "0"): 0.3}, broadcast="never"
    ):
        assert clusters == expected, route
    assert dataset.cluster_seeds_disallow == {("0", "1"), ("0", "2")}


@pytest.mark.parametrize("dataset_disallows", [set(), {("0", "1")}])
def test_disabled_dataset_disallow_cannot_split_altered_profile_during_seed_setup(dataset_disallows):
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"]
    dataset.cluster_seeds_disallow = dataset_disallows.copy()
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1),
        cluster_model=FastCluster(eps=0.4),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    # The mean distance to the intact claimed profile is .5, above eps. An
    # inactive negative must not create temporary pieces that admit this query
    # through the .1-scoring piece before restoring the original membership.
    result = clusterer.predict_incremental(
        ["0", "1", "2"],
        dataset,
        partial_supervision={("0", "2"): 0.1, ("1", "2"): 0.9},
    )
    assert _partition(result["clusters"]) == [["0", "1"], ["2"]]
    assert dataset.cluster_seeds_require == {"0": "claimed", "1": "claimed"}
    assert dataset.cluster_seeds_disallow == dataset_disallows


@pytest.mark.parametrize("enabled", [False, True])
def test_arrow_precomputed_distance_rejection_still_uses_supplied_dataset_disallows(tmp_path, enabled):
    """The effective-constraint gate does not relax the separate precomputed API."""
    dataset = _dataset([("Ada", "", "Smith")] * 3, {})
    dataset.cluster_seeds_disallow = {("0", "1")}
    with pytest.raises(ValueError, match="cluster_seeds_disallow cannot be used with precomputed dists"):
        _predict(
            tmp_path,
            dataset,
            {"one": ["0", "1", "2"]},
            "arrow",
            threshold=None,
            dists={"one": np.array([0.1, 0.1, 0.1])},
            use_default_constraints_as_supervision=enabled,
        )
