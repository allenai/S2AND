"""Verify completion decisions independently of inference backends."""

from unittest.mock import Mock

import pytest

from s2and.consts import LARGE_DISTANCE
from s2and.incremental_linking.completion import (
    complete_incremental_prediction,
    first_initials,
    next_unused_cluster_id,
    residual_first_initial_groups,
)
from s2and.incremental_linking.seed_assignment import SeedLinkAssignment
from s2and.prediction_state import PredictionState


def test_completion_preserves_order_allocates_ids_and_owns_results():
    assignment = SeedLinkAssignment({"0": ["seed0"], "2": ["seed2"], "profile": ["seedp"]}, ["a", "b", "c", "d"], [])
    backend_output = {"unrelated_id": ["a"], "0": ["c"]}
    callback = Mock(return_value=backend_output)
    state = PredictionState()
    result = complete_incremental_prediction(
        assignment,
        first_names={"a": "alice", "b": "bob", "c": "amy", "d": "dan"},
        orcids={},
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
        start_cluster_id=0,
        prediction_state=state,
        cluster_residuals=callback,
    )
    assert result == {
        "0": ["seed0"],
        "2": ["seed2"],
        "profile": ["seedp"],
        "1": ["a"],
        "3": ["c"],
        "4": ["b"],
        "5": ["d"],
    }
    assert list(result) == ["0", "2", "profile", "1", "3", "4", "5"]
    callback.assert_called_once_with(["a", "c"])
    assert state.telemetry["incremental_residual_phase_b"] == {
        "residual_phase_b_signature_count": 4,
        "residual_phase_b_group_count": 3,
        "residual_phase_b_pair_count_before": 6,
        "residual_phase_b_pair_count_after": 1,
        "residual_phase_b_pair_count_saved": 5,
    }
    result["0"].append("extra")
    result["1"].append("extra")
    assert assignment.clusters["0"] == ["seed0"]
    assert backend_output["unrelated_id"] == ["a"]
    assert assignment.residual_signature_ids == ["a", "b", "c", "d"]


@pytest.mark.parametrize("residuals", [[], ["a"]])
def test_empty_or_singleton_completion_skips_backend_and_replaces_telemetry(residuals):
    callback = Mock(side_effect=AssertionError("backend must not run"))
    state = PredictionState(telemetry={"incremental_residual_phase_b": {"stale": 1}, "other": {"keep": True}})
    result = complete_incremental_prediction(
        SeedLinkAssignment({"8": ["seed"]}, residuals, []),
        first_names={},
        orcids={},
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
        start_cluster_id=8,
        prediction_state=state,
        cluster_residuals=callback,
    )
    assert result == ({"8": ["seed"], "9": ["a"]} if residuals else {"8": ["seed"]})
    callback.assert_not_called()
    assert state.telemetry == {
        "other": {"keep": True},
        "incremental_residual_phase_b": {
            "residual_phase_b_signature_count": len(residuals),
            "residual_phase_b_group_count": len(residuals),
            "residual_phase_b_pair_count_before": 0,
            "residual_phase_b_pair_count_after": 0,
            "residual_phase_b_pair_count_saved": 0,
        },
    }


def test_callback_failure_propagates_without_mutating_assignment():
    assignment = SeedLinkAssignment({"1": ["seed"]}, ["a", "b"], [])
    failure = RuntimeError("residual backend failed")

    def failing_callback(group):
        group.append("mutation")
        raise failure

    with pytest.raises(RuntimeError) as caught:
        complete_incremental_prediction(
            assignment,
            first_names={},
            orcids={},
            partial_supervision={},
            use_default_constraints_as_supervision=True,
            suppress_orcid=False,
            start_cluster_id=1,
            prediction_state=PredictionState(),
            cluster_residuals=failing_callback,
        )
    assert caught.value is failure
    assert assignment == SeedLinkAssignment({"1": ["seed"]}, ["a", "b"], [])


@pytest.mark.parametrize(
    "orcids,suppress,supervision,expected",
    [
        ({}, False, {}, [["a", "c"], ["b"], ["d"]]),
        ({"a": "shared", "b": "shared"}, False, {}, [["a", "b", "c"], ["d"]]),
        ({"a": "shared", "b": "shared"}, True, {}, [["a", "c"], ["b"], ["d"]]),
        ({}, False, {("b", "d"): LARGE_DISTANCE - 1}, [["a", "c"], ["b", "d"]]),
        ({}, False, {("b", "d"): LARGE_DISTANCE}, [["a", "c"], ["b"], ["d"]]),
        ({}, False, {("outside", "a"): 0}, [["a", "c"], ["b"], ["d"]]),
    ],
)
def test_grouping_preserves_orcid_and_supervision_bridges(orcids, suppress, supervision, expected):
    assert (
        residual_first_initial_groups(
            ["a", "b", "c", "d"],
            first_names={"a": "alice", "b": "bob", "c": "amy", "d": "dan"},
            orcids=orcids,
            partial_supervision=supervision,
            use_default_constraints_as_supervision=True,
            suppress_orcid=suppress,
        )
        == expected
    )


@pytest.mark.parametrize(
    "names,enabled", [({}, True), ({"a": "amy", "b": " "}, True), ({"a": "amy", "b": "bob"}, False)]
)
def test_grouping_requires_complete_names_and_default_constraints(names, enabled):
    assert residual_first_initial_groups(
        ["a", "b"],
        first_names=names,
        orcids={},
        partial_supervision={},
        use_default_constraints_as_supervision=enabled,
        suppress_orcid=False,
    ) == [["a", "b"]]


def test_hyphenated_names_bridge_initial_groups_transitively():
    assert first_initials("  jean-luc pierre ") == frozenset({"j", "l", "p"})
    assert residual_first_initial_groups(
        ["j", "l", "jl", "p"],
        first_names={"j": "jean", "l": "luc", "jl": "jean-luc-pierre", "p": "pierre"},
        orcids={},
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        suppress_orcid=False,
    ) == [["j", "l", "jl", "p"]]


@pytest.mark.parametrize("suppress_orcid", [False, True])
def test_residual_partition_preserves_transitive_mixed_bridges(suppress_orcid: bool) -> None:
    """ORCID and reverse-pair supervision can connect several initial components."""
    groups = residual_first_initial_groups(
        ["d", "a", "e", "b", "c", "aa"],
        first_names={"a": "alice", "aa": "amy", "b": "bob", "c": "carol", "d": "dan", "e": "eve"},
        orcids={"aa": "shared", "b": "shared"},
        partial_supervision={("c", "b"): 0.25, ("e", "d"): LARGE_DISTANCE},
        use_default_constraints_as_supervision=True,
        suppress_orcid=suppress_orcid,
    )

    assert groups == (
        [["d"], ["a", "aa"], ["e"], ["b", "c"]] if suppress_orcid else [["d"], ["a", "b", "c", "aa"], ["e"]]
    )


def test_cluster_id_allocation_only_skips_exact_numeric_keys():
    assert next_unused_cluster_id({"4": [], "5": [], "07": [], "named": []}, 4) == 6
    assert next_unused_cluster_id({"07": []}, 7) == 7
