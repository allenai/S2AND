"""Incremental prediction rejects negatives inside an original claimed profile."""

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.model_helpers import ConstantDistanceClassifier

CONFLICT = "incremental_seed_disallow_conflict"


def _predict(tmp_path, dataset, *, enabled=True, partial=None, threshold=None, signatures=None, classifier=None):
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1) if classifier is None else classifier,
        cluster_model=FastCluster(eps=0.4),
        n_jobs=1,
        use_default_constraints_as_supervision=enabled,
    )
    requested = list(dataset.signatures) if signatures is None else signatures
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


class _RejectScoringClassifier(ConstantDistanceClassifier):
    def predict_proba(self, features):
        raise AssertionError("Conflicting original seed membership reached model scoring")


@pytest.mark.parametrize(
    ("enabled", "source"),
    [(True, "dataset"), (False, "caller")],
)
def test_original_seed_conflicts_are_rejected_before_scoring(tmp_path, monkeypatch, enabled, source):
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"] if enabled else []
    dataset.cluster_seeds_disallow = {("0", "1")} if source == "dataset" else set()
    partial = {("1", "0"): LARGE_DISTANCE} if source == "caller" else {}

    def reject_promoted_scoring(*_args, **_kwargs):
        raise AssertionError("Conflicting original seed membership reached promoted scoring")

    monkeypatch.setattr(
        "s2and.incremental_linking.production.runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        reject_promoted_scoring,
    )
    with pytest.raises(ValueError, match=CONFLICT):
        _predict(tmp_path, dataset, enabled=enabled, partial=partial, classifier=_RejectScoringClassifier())


@pytest.mark.parametrize(
    ("enabled", "threshold", "reverse_distance"),
    [(False, 1, None), (True, 1, 0.0), (True, None, LARGE_DISTANCE - 1)],
    ids=["disabled-defaults", "zero-override", "soft-override"],
)
def test_inactive_seed_conflicts_preserve_the_claimed_group(tmp_path, enabled, threshold, reverse_distance):
    seeds = {"0": "claimed", "1": "claimed"}
    dataset = _dataset([("Ada", "", "Smith")] * 4, seeds)
    dataset.cluster_seeds_disallow = {("0", "1")}
    dataset.altered_cluster_signatures = ["0"]
    partial = {("1", "0"): reverse_distance} if reverse_distance is not None else {}

    assert _predict(tmp_path, dataset, enabled=enabled, partial=partial, threshold=threshold) == [["0", "1", "2", "3"]]
    assert dataset.cluster_seeds_require == seeds
    assert dataset.cluster_seeds_disallow == {("0", "1")}


@pytest.mark.parametrize("signatures", [["2", "3"], []], ids=["seeds-omitted", "empty"])
def test_incremental_seed_conflicts_rejected_before_query_early_returns(tmp_path, signatures):
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "claimed", "1": "claimed"})
    dataset.cluster_seeds_disallow = {("0", "1")}
    with pytest.raises(ValueError, match=CONFLICT):
        _predict(tmp_path, dataset, signatures=signatures)


@pytest.mark.parametrize("case", ["between_profiles", "query_query"])
def test_negatives_outside_an_original_seed_group_remain_valid(tmp_path, case):
    seeds = {"0": "claimed", "1": "claimed"}
    if case == "between_profiles":
        seeds["3"] = "other"
    dataset = _dataset([("Ada", "", "Smith")] * 4, seeds)
    dataset.altered_cluster_signatures = ["0"]
    dataset.cluster_seeds_disallow = {("0", "3")} if case == "between_profiles" else {("2", "3")}
    partial = {("0", "2"): 0.0, ("2", "3"): LARGE_DISTANCE} if case == "between_profiles" else {}
    assert _predict(tmp_path, dataset, partial=partial) == [["0", "1", "2"], ["3"]]


def test_seed_conflict_error_chooses_the_same_pair_for_reverse_input_order(tmp_path):
    seeds = {"2": "claimed", "1": "claimed", "0": "claimed"}
    pairs = {("2", "1"): LARGE_DISTANCE, ("1", "0"): LARGE_DISTANCE}
    dataset = _dataset([("Ada", "", "Smith")] * 4, seeds)
    with pytest.raises(ValueError, match=r"pair=\('0', '1'\)"):
        _predict(tmp_path, dataset, partial=pairs)


def test_name_incompatibility_does_not_reject_original_claimed_membership(tmp_path):
    dataset = _dataset([("Ada", "", "Smith"), ("Amy", "", "Jones")], {"0": "claimed", "1": "claimed"})
    dataset.altered_cluster_signatures = ["0"]
    assert _predict(tmp_path, dataset) == [["0", "1"]]
