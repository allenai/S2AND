"""Incremental prediction rejects negatives inside an original claimed profile."""

import re

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.model_helpers import ConstantDistanceClassifier

CONFLICT_MESSAGE = (
    "incremental_seed_disallow_conflict: seed_group='claimed', pair=('0', '1'); "
    "original seed members must remain together"
)


def _predict(tmp_path, dataset, route, *, enabled=True, partial=None, threshold=None, signatures=None, classifier=None):
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1) if classifier is None else classifier,
        cluster_model=FastCluster(eps=0.4),
        n_jobs=1,
        use_default_constraints_as_supervision=enabled,
    )
    requested = list(dataset.signatures) if signatures is None else signatures
    if route == "python":
        result = clusterer.predict_incremental(requested, dataset, partial_supervision=partial)
    else:
        prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
        attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
        with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
            result = clusterer.predict_incremental_from_arrow(
                requested,
                arrow_dataset,
                cluster_seeds_require=dataset.cluster_seeds_require,
                cluster_seeds_disallow=dataset.cluster_seeds_disallow,
                altered_cluster_signatures=dataset.altered_cluster_signatures,
                partial_supervision=partial,
                batching_threshold=threshold,
                name_tuples=frozenset(),
            )
    return sorted(sorted(members) for members in result["clusters"].values())


@pytest.mark.parametrize("altered", [False, True])
@pytest.mark.parametrize(
    ("enabled", "source", "override", "reject", "threshold"),
    [
        pytest.param(True, "dataset", None, True, None, id="dataset-active-None"),
        pytest.param(False, "dataset", None, False, None, id="dataset-disabled-None"),
        pytest.param(False, "dataset", None, False, 1, id="dataset-disabled-1"),
        pytest.param(True, "caller", LARGE_DISTANCE, True, None, id="caller-defaults-on-None"),
        pytest.param(False, "caller", LARGE_DISTANCE, True, None, id="caller-defaults-off-None"),
        pytest.param(True, "dataset", 0.0, False, None, id="zero-override-None"),
        pytest.param(True, "dataset", 0.0, False, 1, id="zero-override-1"),
        pytest.param(True, "dataset", 0.3, False, None, id="soft-override-None"),
        pytest.param(True, "dataset", 0.3, False, 1, id="soft-override-1"),
        pytest.param(True, "caller", LARGE_DISTANCE - 1, False, None, id="high-soft-distance-None"),
        pytest.param(True, "caller", LARGE_DISTANCE - 1, False, 1, id="high-soft-distance-1"),
    ],
)
def test_incremental_seed_conflict_policy_matches_routes(
    tmp_path, monkeypatch, altered, threshold, enabled, source, override, reject
):
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"] if altered else []
    dataset.cluster_seeds_disallow = {("0", "1")} if source == "dataset" else set()
    partial = {} if override is None else {("1", "0"): override}
    if reject:

        def reject_promoted_scoring(*_args, **_kwargs):
            raise AssertionError("Conflicting original seed membership reached promoted scoring")

        monkeypatch.setattr(
            "s2and.incremental_linking.production.runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
            reject_promoted_scoring,
        )
    errors = []
    for route in ["python", "arrow"]:
        if reject:
            try:
                _predict(
                    tmp_path / route,
                    dataset,
                    route,
                    enabled=enabled,
                    partial=partial,
                    threshold=threshold,
                    classifier=_RejectScoringClassifier(),
                )
            except ValueError as error:
                errors.append(str(error))
            else:
                errors.append(None)
        else:
            assert _predict(
                tmp_path / route,
                dataset,
                route,
                enabled=enabled,
                partial=partial,
                threshold=threshold,
            ) == [["0", "1", "2", "3"]]
    if reject:
        assert errors == [CONFLICT_MESSAGE, CONFLICT_MESSAGE]
    assert dataset.cluster_seeds_require == {"0": "claimed", "1": "claimed"}
    assert dataset.cluster_seeds_disallow == ({("0", "1")} if source == "dataset" else set())


@pytest.mark.parametrize("altered", [False, True])
@pytest.mark.parametrize("signatures", [["2", "3"], ["0", "1"], []], ids=["seeds-omitted", "seed-only", "empty"])
def test_incremental_seed_conflicts_rejected_before_query_early_returns(tmp_path, altered, signatures):
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"] if altered else []
    dataset.cluster_seeds_disallow = {("0", "1")}
    errors = []
    for route in ["python", "arrow"]:
        try:
            _predict(tmp_path / route, dataset, route, signatures=signatures)
        except ValueError as error:
            errors.append(str(error))
        else:
            errors.append(None)
    assert errors == [CONFLICT_MESSAGE, CONFLICT_MESSAGE]


@pytest.mark.parametrize("altered", [False, True])
@pytest.mark.parametrize("case", ["between_profiles", "query_seed", "query_query"])
def test_negatives_outside_an_original_seed_group_remain_valid(tmp_path, altered, case):
    seeds = {"0": "claimed", "1": "claimed"}
    size = 4 if case != "query_seed" else 3
    if case == "between_profiles":
        seeds["3"] = "other"
    dataset = _dataset([("Ada", "", "Smith")] * size, seeds)
    dataset.altered_cluster_signatures = ["0"] if altered else []
    pair = {"between_profiles": ("0", "3"), "query_seed": ("0", "2"), "query_query": ("2", "3")}[case]
    dataset.cluster_seeds_disallow = {pair}
    partial = {("0", "2"): 0.0, ("2", "3"): LARGE_DISTANCE} if case == "between_profiles" else {}
    expected = [["0", "1"], ["2"]] if case == "query_seed" else [["0", "1", "2"], ["3"]]
    for route in ["python", "arrow"]:
        assert _predict(tmp_path / route, dataset, route, partial=partial) == expected, route


class _RejectScoringClassifier(ConstantDistanceClassifier):
    def predict_proba(self, features):
        raise AssertionError("Conflicting original seed membership reached model scoring")


@pytest.mark.parametrize("reverse", [False, True])
def test_seed_conflict_error_is_independent_of_supervision_and_seed_order(tmp_path, reverse):
    seeds = [("0", "claimed"), ("1", "claimed"), ("2", "claimed")]
    pairs = [(("2", "1"), LARGE_DISTANCE), (("1", "0"), LARGE_DISTANCE)]
    if reverse:
        seeds.reverse()
        pairs.reverse()
    dataset = _dataset([("Ada", "", "Smith")] * 4, dict(seeds))
    for route in ["python", "arrow"]:
        with pytest.raises(ValueError, match=re.escape(CONFLICT_MESSAGE)):
            _predict(tmp_path / route, dataset, route, partial=dict(pairs))


@pytest.mark.parametrize("altered", [False, True])
def test_name_incompatibility_does_not_reject_original_claimed_membership(tmp_path, altered):
    dataset = _dataset([("Ada", "", "Smith"), ("Amy", "", "Jones")], {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"] if altered else []
    for route in ["python", "arrow"]:
        assert _predict(tmp_path / route, dataset, route) == [["0", "1"]]
