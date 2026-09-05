"""Regression coverage for conflicts introduced by classic seed attachment."""

from collections.abc import Iterator

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.model import Clusterer, _prediction_state_from_dataset
from s2and.runtime import build_runtime_context
from tests import test_cluster as cluster_test_helpers


@pytest.fixture
def classic() -> Iterator[tuple[Clusterer, ANDData]]:
    """Provide real feature/constraint machinery and deterministic pair scores."""
    fixture = cluster_test_helpers.TestClusterer()
    fixture.setUp()
    try:
        fixture._fill_python_featurizer_fields()
        dataset = fixture.dummy_dataset
        clusterer = fixture.dummy_clusterer
        clusterer.use_default_constraints_as_supervision = True
        clusterer.classifier = DummyClassifier(strategy="prior").fit(np.zeros((10, 6)), np.array([0] + [1] * 9))
        dataset.cluster_seeds_require = {"0": "seed"}
        dataset.cluster_seeds_disallow = set()
        dataset.altered_cluster_signatures = []
        dataset.name_tuples = set()
        for signature_id in ["0", "1", "2", "3"]:
            dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
                author_info_first="john",
                author_info_first_normalized_without_apostrophe="john",
                author_info_middle="",
                author_info_middle_normalized_without_apostrophe="",
                author_info_last="smith",
                author_info_last_normalized="smith",
                author_info_orcid=None,
            )
        yield clusterer, dataset
    finally:
        fixture.tearDown()


def _filter(
    classic: tuple[Clusterer, ANDData],
    *,
    links: dict[str, str] | None = None,
    scores: dict[str, float] | None = None,
    supervision: dict[tuple[str, str], int | float] | None = None,
    splits: dict[str, str] | None = None,
    restored: dict[int | str, int | str] | None = None,
) -> dict[str, int | str]:
    """Exercise accepted-link filtering using the same frozen state as public calls."""
    clusterer, dataset = classic
    return clusterer._filter_classic_incremental_links(
        dataset,
        {"1": "seed", "2": "seed"} if links is None else links,
        {"1": 0.1, "2": 0.2} if scores is None else scores,
        cluster_seeds_require=dataset.cluster_seeds_require if splits is None else splits,
        recluster_map={} if restored is None else restored,
        partial_supervision={} if supervision is None else supervision,
        runtime_context=build_runtime_context("cluster_predict_incremental", backend="python"),
        prediction_state=_prediction_state_from_dataset(dataset),
    )


@pytest.mark.parametrize("order", [["0", "1", "2"], ["2", "1", "0"]])
@pytest.mark.parametrize("broadcast", ["always", "never", "top1_consensus"])
@pytest.mark.parametrize("score_mode", ["mean", "min", "mean_min_hybrid"])
def test_public_incremental_preserves_query_disallow(classic, order, broadcast, score_mode):
    """A hard-negative pair cannot join a common seed through independent links."""
    clusterer, dataset = classic
    clusterer.incremental_precluster_broadcast_mode = broadcast
    clusterer.incremental_seed_score_mode = score_mode
    dataset.cluster_seeds_disallow = {("1", "2")}
    result = clusterer.predict_incremental(order, dataset)["clusters"]
    assert result["seed"] == ["0", "1"]
    assert any(members == ["2"] for members in result.values())
    assert dataset.cluster_seeds_require == {"0": "seed"}


def test_public_incremental_rejects_new_name_conflict(classic):
    clusterer, dataset = classic
    for signature_id, first in [("0", "j"), ("1", "john"), ("2", "james")]:
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_first=first, author_info_first_normalized_without_apostrophe=first
        )
    assert dataset.get_constraint("1", "2") == LARGE_DISTANCE
    result = clusterer.predict_incremental(["0", "1", "2"], dataset)["clusters"]
    assert result["seed"] == ["0", "1"]
    assert not any({"1", "2"}.issubset(members) for members in result.values())


def test_bulk_initial_attachment_uses_synthetic_seed_state(classic):
    clusterer, dataset = classic
    dataset.cluster_seeds_require = {}
    dataset.cluster_seeds_disallow = {("1", "2")}
    for signature_id in ["1", "2"]:
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_first="j", author_info_first_normalized_without_apostrophe="j"
        )
    result, _ = clusterer.predict({"full": ["0"], "initial": ["1", "2"]}, dataset, batching_threshold=10)
    assert any(set(members) == {"0", "1"} for members in result.values())
    assert not any({"1", "2"}.issubset(members) for members in result.values())
    assert dataset.cluster_seeds_require == {}


@pytest.mark.parametrize("default_constraints", [False, True])
@pytest.mark.parametrize("suppress_orcid", [False, True])
def test_filter_preserves_default_and_orcid_policy(classic, default_constraints, suppress_orcid):
    clusterer, dataset = classic
    clusterer.use_default_constraints_as_supervision = default_constraints
    clusterer.suppress_orcid = suppress_orcid
    for signature_id, first in [("1", "john"), ("2", "james")]:
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=first,
            author_info_orcid="0000-0002-1825-0097",
        )
    expected = {"1": "seed"} if default_constraints and suppress_orcid else {"1": "seed", "2": "seed"}
    assert _filter(classic) == expected


@pytest.mark.parametrize("default_constraints", [False, True])
def test_dataset_disallow_gate_and_reverse_supervision(classic, default_constraints):
    clusterer, dataset = classic
    clusterer.use_default_constraints_as_supervision = default_constraints
    dataset.cluster_seeds_disallow = {("1", "2")}
    expected = {"1": "seed"} if default_constraints else {"1": "seed", "2": "seed"}
    assert _filter(classic) == expected
    assert _filter(classic, supervision={("2", "1"): 0.3}) == {"1": "seed", "2": "seed"}
    dataset.cluster_seeds_disallow = set()
    assert _filter(classic, supervision={("2", "1"): LARGE_DISTANCE}) == {"1": "seed"}


def test_required_link_priority_and_contradictory_requires(classic):
    _, dataset = classic
    dataset.cluster_seeds_disallow = {("1", "2")}
    assert _filter(classic, supervision={("0", "2"): 0}) == {"2": "seed"}
    with pytest.raises(ValueError, match="Conflicting required classic incremental attachments"):
        _filter(classic, supervision={("0", "2"): 0, ("1", "0"): 0})


@pytest.mark.parametrize("default_constraints", [False, True])
def test_attachment_uses_direct_supervision_over_reverse_negative(classic, default_constraints):
    clusterer, _ = classic
    clusterer.use_default_constraints_as_supervision = default_constraints
    supervision = {("1", "2"): LARGE_DISTANCE, ("2", "1"): 0.1}
    assert _filter(classic, supervision=supervision) == {"1": "seed", "2": "seed"}
    assert _filter(classic, supervision=supervision, scores={"1": 0.2, "2": 0.1}) == {"2": "seed"}
    historical = {("0", "1"): LARGE_DISTANCE, ("1", "0"): 0.1}
    assert _filter(classic, links={"1": "seed"}, supervision=historical) == {"1": "seed"}


def test_overridden_reverse_zero_does_not_create_atomic_require_group(classic):
    links = {"2": "seed", "1": "seed", "3": "seed"}
    scores = {"1": 0.1, "2": 0.2, "3": 0.05}
    supervision = {("1", "2"): 0, ("2", "1"): 0.3, ("1", "3"): LARGE_DISTANCE}
    assert _filter(classic, links=links, scores=scores, supervision=supervision) == {"3": "seed", "2": "seed"}
    # Reversing score priority makes the zero direct and applicable. The
    # effective require group now abstains together when it conflicts with 3.
    scores.update({"1": 0.2, "2": 0.1})
    assert _filter(classic, links=links, scores=scores, supervision=supervision) == {"3": "seed"}


def test_query_require_component_is_never_partially_attached(classic):
    links = {"1": "seed", "2": "seed", "3": "seed"}
    scores = {"1": 0.1, "2": 0.05, "3": 0.3}
    supervision = {("2", "3"): 0, ("1", "3"): LARGE_DISTANCE}
    assert _filter(classic, links=links, scores=scores, supervision=supervision) == {"2": "seed", "3": "seed"}
    # A higher-priority seed requirement sends the entire conflicting free
    # query component to residual clustering, preserving its internal require.
    supervision[("1", "0")] = 0
    assert _filter(classic, links=links, scores=scores, supervision=supervision) == {"1": "seed"}
    supervision[("2", "0")] = 0
    with pytest.raises(ValueError, match="Conflicting required classic incremental attachments"):
        _filter(classic, links=links, scores=scores, supervision=supervision)


def test_restored_sibling_explicit_conflicts_and_historical_name_exception(classic):
    _, dataset = classic
    dataset.cluster_seeds_require = {"0": "claimed", "3": "claimed"}
    splits = {"0": "split_a", "3": "split_b"}
    restored: dict[int | str, int | str] = {"split_a": "claimed", "split_b": "claimed"}
    links = {"1": "split_a", "2": "split_b"}
    dataset.cluster_seeds_disallow = {("1", "2")}
    assert _filter(classic, links=links, splits=splits, restored=restored) == {"1": "split_a"}
    # A negative against a different historical split still forbids restoration.
    dataset.cluster_seeds_disallow = {("1", "3")}
    assert _filter(classic, links={"1": "split_a"}, splits=splits, restored=restored) == {}
    assert _filter(
        classic, links={"1": "split_a"}, splits=splits, restored=restored, supervision={("3", "1"): 0.1}
    ) == {"1": "split_a"}
    dataset.cluster_seeds_disallow = set()
    dataset.signatures["3"] = dataset.signatures["3"]._replace(
        author_info_first="james", author_info_first_normalized_without_apostrophe="james"
    )
    assert _filter(classic, links={"1": "split_a"}, splits=splits, restored=restored) == {"1": "split_a"}


def test_rejected_attachment_abstains_instead_of_selecting_second_seed(classic):
    clusterer, dataset = classic
    dataset.cluster_seeds_require = {"0": "seed", "3": "second"}
    dataset.cluster_seeds_disallow = {("1", "2")}
    # Both seeds are acceptable by score, but only the existing best is proposed.
    supervision = {("1", "0"): 0.1, ("2", "0"): 0.2, ("1", "3"): 0.3, ("2", "3"): 0.3}
    result = clusterer.predict_incremental(["0", "1", "2", "3"], dataset, partial_supervision=supervision)["clusters"]
    assert result["seed"] == ["0", "1"]
    assert result["second"] == ["3"]
    assert any(members == ["2"] for members in result.values())
