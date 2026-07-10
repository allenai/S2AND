from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import s2and.incremental_linking.production as production_module
import s2and.incremental_linking.runtime as runtime_module
from s2and.incremental_linking.feature_block import feature_block_signature_order_from_raw_candidate_plan
from s2and.incremental_linking.features import LinkerFeatureMatrix, promoted_linker_feature_columns
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.incremental_linking.runtime import LinkOrAbstainDecision


def test_promoted_incremental_window_signature_order_uses_feature_block_contract() -> None:
    assert (
        production_module.feature_block_signature_order_from_raw_candidate_plan
        is feature_block_signature_order_from_raw_candidate_plan
    )


def test_raw_arrow_plan_window_enabled_when_query_batch_is_smaller_than_query_count() -> None:
    query_batch_size = 2
    plan_window_size = production_module._raw_arrow_plan_window_size(  # noqa: SLF001
        query_count=9,
        query_batch_size=query_batch_size,
        plan_window_multiplier=production_module._RAW_ARROW_PLAN_WINDOW_MULTIPLIER,  # noqa: SLF001
    )

    assert production_module._RAW_ARROW_PLAN_WINDOW_MULTIPLIER > 1  # noqa: SLF001
    assert plan_window_size > query_batch_size
    assert int(plan_window_size > query_batch_size) == 1


def test_raw_arrow_plan_window_size_is_positive_for_empty_query_set() -> None:
    assert (
        production_module._raw_arrow_plan_window_size(  # noqa: SLF001
            query_count=0,
            query_batch_size=10,
            plan_window_multiplier=production_module._RAW_ARROW_PLAN_WINDOW_MULTIPLIER,  # noqa: SLF001
        )
        == 1
    )


def test_memory_safe_query_batch_shrinks_to_refreshed_limit(monkeypatch) -> None:
    calls: list[int] = []

    def fake_limits(**kwargs):
        query_count = int(kwargs["query_count"])
        calls.append(query_count)
        return SimpleNamespace(
            query_batch_size=min(2, query_count),
            single_query_exceeds_budget=False,
        )

    monkeypatch.setattr(production_module, "compute_promoted_incremental_limits", fake_limits)

    query_batch, limits = production_module._memory_safe_promoted_query_batch(  # noqa: SLF001
        ["q0", "q1", "q2", "q3"],
        orcid_fanout_by_query={},
        component_sizes={"c0": 1},
        retrieval_top_k=1,
        memory_layout=production_module._PromotedIncrementalMemoryLayout(  # noqa: SLF001
            final_matrix_feature_count=53,
            pairwise_matrix_feature_count=35,
            aggregate_feature_count=18,
        ),
        total_ram_bytes=1_000_000,
        base_candidate_rows_per_query=1,
        base_pairs_per_query=1,
    )

    assert query_batch == ["q0", "q1"]
    assert limits.query_batch_size == 2
    assert calls == [4, 2]


def test_batch_telemetry_aggregates_all_refreshed_limits() -> None:
    limits = [
        SimpleNamespace(
            query_batch_size=4,
            predicted_peak_delta_bytes=100,
            predicted_peak_rss_bytes=1_000,
            operational_estimate_source="default",
        ),
        SimpleNamespace(
            query_batch_size=2,
            predicted_peak_delta_bytes=250,
            predicted_peak_rss_bytes=1_250,
            operational_estimate_source="observed",
        ),
    ]

    telemetry = production_module.merge_promoted_incremental_batch_telemetry(
        [{"query_count": 2}, {"query_count": 2}],
        batch_sizes=[2, 2],
        configured_batch_size=4,
        final_limits_history=limits,
    )

    assert telemetry["memory_final_query_batch_size"] == 2
    assert telemetry["memory_final_predicted_peak_delta_bytes"] == 250
    assert telemetry["memory_final_predicted_peak_rss_bytes"] == 1_250
    assert telemetry["memory_final_operational_estimate_source"] == "__mixed__"


def _scored_query(
    signature_id: str,
    component_key: str | None,
    score: float | None,
    *,
    require_forced: bool = False,
):
    return production_module._ScoredQueryDecision(  # noqa: SLF001
        signature_id=signature_id,
        decision=LinkOrAbstainDecision(
            query_signature_index=0,
            action="link" if component_key is not None else "abstain",
            row_index=0 if component_key is not None else None,
            component_key=component_key,
            score=score,
            runner_up_score=None,
            score_margin=None,
        ),
        require_forced=require_forced,
    )


def test_global_query_disallow_resolution_is_score_ordered_and_input_order_invariant() -> None:
    initial = {
        "q-low": _scored_query("q-low", "shared", 0.90),
        "q-high": _scored_query("q-high", "shared", 0.95),
    }
    rescored: list[tuple[str, set[str]]] = []

    def rescore(signature_id: str, excluded_components: set[str]):
        rescored.append((signature_id, excluded_components))
        return _scored_query(signature_id, "runner-up", 0.80)

    expected = {"q-high": "shared", "q-low": "runner-up"}
    for ordered_initial in (initial, dict(reversed(list(initial.items())))):
        linked, telemetry = production_module._resolve_query_disallows_globally(  # noqa: SLF001
            ordered_initial,
            {"q-low": {"q-high"}, "q-high": {"q-low"}},
            rescore=rescore,
        )
        assert linked == expected
        assert telemetry["global_query_disallow_conflict_count"] == 1
        assert telemetry["global_query_disallow_rescore_count"] == 1
    assert rescored == [("q-low", {"shared"}), ("q-low", {"shared"})]


def test_global_query_disallow_resolution_prioritizes_require_and_avoids_unneeded_rescore() -> None:
    initial = {
        "q-require": _scored_query("q-require", "shared", 0.70, require_forced=True),
        "q-score": _scored_query("q-score", "shared", 0.99),
        "q-uncontended": _scored_query("q-uncontended", "other", 0.80),
    }
    rescored: list[str] = []

    def rescore(signature_id: str, excluded_components: set[str]):
        rescored.append(signature_id)
        assert excluded_components == {"shared"}
        return _scored_query(signature_id, None, None)

    linked, telemetry = production_module._resolve_query_disallows_globally(  # noqa: SLF001
        initial,
        {"q-require": {"q-score"}, "q-score": {"q-require"}},
        rescore=rescore,
    )

    assert linked == {"q-require": "shared", "q-uncontended": "other"}
    assert rescored == ["q-score"]
    assert telemetry["global_query_disallow_demoted_abstain_count"] == 1


def test_global_query_disallow_resolution_rejects_conflicting_requires() -> None:
    initial = {
        "q1": _scored_query("q1", "shared", 0.90, require_forced=True),
        "q2": _scored_query("q2", "shared", 0.80, require_forced=True),
    }

    try:
        production_module._resolve_query_disallows_globally(  # noqa: SLF001
            initial,
            {"q1": {"q2"}, "q2": {"q1"}},
            rescore=lambda _signature_id, _excluded: _scored_query("unused", None, None),
        )
    except ValueError as error:
        assert "cluster_seed_disallow_conflicts_with_require_constraint" in str(error)
    else:
        raise AssertionError("conflicting require decisions must fail")


def test_query_disallow_components_are_undirected_and_complete() -> None:
    components = production_module._query_disallow_components_by_id(  # noqa: SLF001
        {"q1": {"q2"}, "q2": {"q1", "q3"}, "q3": {"q2"}, "q4": {"q5"}, "q5": {"q4"}}
    )

    assert components["q1"] == frozenset({"q1", "q2", "q3"})
    assert components["q2"] is components["q1"]
    assert components["q3"] is components["q1"]
    assert components["q4"] == frozenset({"q4", "q5"})


def test_disallow_aware_query_batches_keep_small_components_together() -> None:
    components = production_module._query_disallow_components_by_id(  # noqa: SLF001
        {"q1": {"q4"}, "q4": {"q1"}, "q2": {"q5"}, "q5": {"q2"}}
    )

    batches = production_module._disallow_aware_query_batches(  # noqa: SLF001
        ["q1", "free-a", "q2", "q4", "free-b", "q5"],
        query_batch_size=3,
        disallow_components_by_id=components,
    )

    assert all(len(batch) <= 3 for batch in batches)
    assert sorted(signature_id for batch in batches for signature_id in batch) == [
        "free-a",
        "free-b",
        "q1",
        "q2",
        "q4",
        "q5",
    ]
    assert any({"q1", "q4"} <= set(batch) for batch in batches)
    assert any({"q2", "q5"} <= set(batch) for batch in batches)


def test_disallow_aware_query_batches_are_input_order_invariant_and_well_packed() -> None:
    partner_ids: dict[str, set[str]] = {}
    query_ids: list[str] = []
    for component_index, component_size in enumerate([51] * 10 + [49] * 10):
        component = [f"c{component_index:02d}-{member:02d}" for member in range(component_size)]
        query_ids.extend(component)
        for signature_id in component:
            partner_ids[signature_id] = set(component) - {signature_id}
    components = production_module._query_disallow_components_by_id(partner_ids)  # noqa: SLF001

    forward = production_module._disallow_aware_query_batches(  # noqa: SLF001
        query_ids,
        query_batch_size=100,
        disallow_components_by_id=components,
    )
    reverse = production_module._disallow_aware_query_batches(  # noqa: SLF001
        list(reversed(query_ids)),
        query_batch_size=100,
        disallow_components_by_id=components,
    )

    assert forward == reverse
    assert len(forward) == 10
    assert {len(batch) for batch in forward} == {100}


def test_disallow_aware_query_batches_never_exceed_fixed_slice_batch_count() -> None:
    partner_ids: dict[str, set[str]] = {}
    query_ids: list[str] = []
    components_by_members: list[set[str]] = []
    for component_index in range(3):
        component = {f"c{component_index}-{member}" for member in range(6)}
        components_by_members.append(component)
        query_ids.extend(sorted(component))
        for signature_id in component:
            partner_ids[signature_id] = component - {signature_id}
    components = production_module._query_disallow_components_by_id(partner_ids)  # noqa: SLF001

    batches = production_module._disallow_aware_query_batches(  # noqa: SLF001
        query_ids,
        query_batch_size=10,
        disallow_components_by_id=components,
    )

    assert len(batches) == 2
    assert sorted(len(batch) for batch in batches) == [8, 10]
    assert any(not any(component <= set(batch) for batch in batches) for component in components_by_members)


def test_query_batch_plan_windows_preserve_precomputed_batch_boundaries() -> None:
    windows = production_module._query_batch_plan_windows(  # noqa: SLF001
        [["q1", "q2"], ["q3"], ["q4", "q5"], ["q6"], ["q7"]],
        plan_window_multiplier=2,
    )

    assert windows == [[["q1", "q2"], ["q3"]], [["q4", "q5"], ["q6"]], [["q7"]]]


def test_resize_query_batch_plan_windows_applies_refreshed_batch_and_window_limits() -> None:
    windows = production_module._resize_query_batch_plan_windows(  # noqa: SLF001
        [["q1", "q2", "q3", "q4"], ["q5", "q6", "q7"], ["q8"]],
        query_batch_size=2,
        plan_window_multiplier=2,
    )

    assert windows == [
        [["q1", "q2"], ["q3", "q4"]],
        [["q5", "q6"], ["q7"]],
        [["q8"]],
    ]


def test_same_batch_disallow_score_tie_uses_signature_id_not_runtime_index() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=4,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11, 11], dtype=np.uint32),
        row_component_keys=("shared", "ten-other", "shared", "eleven-other"),
        retrieval_ranks=np.asarray([1, 2, 1, 2], dtype=np.uint16),
    )
    feature_matrix = LinkerFeatureMatrix(
        matrix=np.zeros((4, len(promoted_linker_feature_columns())), dtype=np.float32),
        feature_columns=promoted_linker_feature_columns(),
        candidate_batch=candidate_batch,
    )

    class Artifact:
        metadata = SimpleNamespace(
            gate_config=logistic_gate_config(
                feature_names=("chosen_probability",),
                weights=np.asarray([[-200.0, 0.0, 200.0]], dtype=np.float64),
                bias=np.asarray([100.0, -10.0, -100.0], dtype=np.float64),
                missing_values=np.asarray([0.0], dtype=np.float64),
                calibration_mode="test",
            )
        )

        @staticmethod
        def predict_probabilities(_matrix, *, num_threads=None):
            del num_threads
            return np.asarray([0.90, 0.80, 0.90, 0.70], dtype=np.float64)

    result = runtime_module._predict_incremental_link_or_abstain_compact(  # noqa: SLF001
        Artifact(),
        feature_matrix,
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"] * 4, dtype=object)},
        disallow_partner_query_indices={10: {11}, 11: {10}},
        disallow_query_priority_ids={10: "q-z", 11: "q-a"},
    )

    by_query = {decision.query_signature_index: decision.component_key for decision in result.decisions}
    assert by_query == {10: "ten-other", 11: "shared"}
