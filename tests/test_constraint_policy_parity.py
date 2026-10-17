"""Public Python/Arrow regressions for effective caller and dataset constraints."""

from collections.abc import Mapping

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, _predict, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.model_helpers import ConstantDistanceClassifier


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
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
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


@pytest.mark.parametrize("threshold", [None, 2])
@pytest.mark.parametrize(
    ("case", "enabled"),
    [
        ("dataset_disallow", False),
        ("dataset_disallow", True),
        ("caller_disallow", False),
        ("caller_disallow", True),
        ("zero_override", True),
        ("soft_override", True),
    ],
)
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
    # The cap cuts the no-middle "ada" records into the pieces 0-1 and 2. The reunite-only pass
    # then attaches record 2, so the subblocked result equals the full-block result.
    expected = [["0", "2"], ["1"]] if disallowed else [["0", "1", "2"]]
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


@pytest.mark.parametrize("override", [0.0, 0.3])
def test_incremental_override_precedes_retrieval(tmp_path, override):
    dataset = _dataset([("Ada", "", "Smith")] * 2, {"0": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1")}
    pair = ("0", "1")
    partial = {pair: override}
    for route, clusters in _incremental_predictions(tmp_path, dataset, enabled=True, partial_supervision=partial):
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
