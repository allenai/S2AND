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


def _incremental_predictions(tmp_path, dataset, *, enabled, partial_supervision):
    """Use real retrieval and scoring with deterministic pair and linker models."""
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=enabled,
    )
    signatures = list(dataset.signatures)
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


@pytest.mark.parametrize(("enabled", "threshold"), [(False, None), (False, 2), (True, 2)])
def test_bulk_uses_effective_constraints_before_real_subblocking(tmp_path, enabled, threshold):
    names = [
        (first, "", last)
        for first, last in (("Ada", "Smith"), ("Grace", "Hopper"), ("Mary", "Jackson"), ("Katherine", "Johnson"))
        for _ in range(3)
    ]
    dataset = _dataset(names, {})
    dataset.cluster_seeds_disallow = {("0", "1"), ("6", "7"), ("9", "10")}
    partial = {("3", "4"): LARGE_DISTANCE, ("6", "7"): 0.0, ("10", "9"): 0.3}
    blocks = {
        "dataset": ["0", "1", "2"],
        "caller": ["3", "4", "5"],
        "zero": ["6", "7", "8"],
        "soft": ["9", "10", "11"],
    }

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        "arrow",
        threshold=threshold,
        use_default_constraints_as_supervision=enabled,
        partial_supervision=partial,
    )

    # The cap splits each triple before the reunite pass restores allowed links.
    expected = [["0", "2"], ["1"]] if enabled else [["0", "1", "2"]]
    expected += [["3", "5"], ["4"], ["6", "7", "8"], ["9", "10", "11"]]
    assert _partition(clusters) == sorted(sorted(members) for members in expected)


@pytest.mark.parametrize(
    ("case", "enabled"),
    [
        ("dataset_seed_query", True),
        ("dataset_query_query", True),
        ("dataset_seed_query", False),
        ("caller_seed_query", False),
        ("caller_query_query", False),
    ],
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


def test_incremental_override_precedes_retrieval(tmp_path):
    dataset = _dataset([("Ada", "", "Smith")] * 2, {"0": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1")}
    pair = ("0", "1")
    partial = {pair: 0.0}
    for route, clusters in _incremental_predictions(tmp_path, dataset, enabled=True, partial_supervision=partial):
        assert clusters == [["0", "1"]], route
    assert dataset.cluster_seeds_disallow == {("0", "1")}
    assert partial == {pair: 0.0}


def test_incremental_override_preserves_independent_dataset_negatives(tmp_path):
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1"), ("0", "2")}
    for route, clusters in _incremental_predictions(
        tmp_path, dataset, enabled=True, partial_supervision={("1", "0"): 0.3}
    ):
        assert clusters == [["0", "1"], ["2"]], route
    assert dataset.cluster_seeds_disallow == {("0", "1"), ("0", "2")}
