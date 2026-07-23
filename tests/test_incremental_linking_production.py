from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import s2and.incremental_linking.production as production_module
from s2and.incremental_linking.runtime import LinkOrAbstainDecision


def test_memory_safe_query_batch_shrinks_to_refreshed_limit(monkeypatch) -> None:
    calls: list[int] = []

    def fake_limits(**kwargs):
        query_count = int(kwargs["query_count"])
        calls.append(query_count)
        return SimpleNamespace(
            query_batch_size=min(2, query_count),
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
        ),
        SimpleNamespace(
            query_batch_size=2,
            predicted_peak_delta_bytes=250,
            predicted_peak_rss_bytes=1_250,
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


def test_scored_query_extraction_retains_require_constraint_priority() -> None:
    result = SimpleNamespace(
        compact_result=SimpleNamespace(decisions=(_scored_query("q-require", "required-component", 0.7).decision,)),
        pairwise_model_result=SimpleNamespace(row_signals={}),
        decision_row_signals={"constraint_require_count": np.asarray([1.0], dtype=np.float32)},
    )

    scored = production_module._scored_query_decisions_from_result(  # noqa: SLF001
        result,
        signature_ids_by_index=["q-require"],
        expected_query_signature_ids=["q-require"],
    )

    assert scored["q-require"].require_forced is True


def test_cross_batch_disallow_resolution_cannot_exclude_required_component() -> None:
    def extract_batch(signature_id: str, score: float, require_count: float):
        decision = _scored_query(signature_id, "shared", score).decision
        result = SimpleNamespace(
            compact_result=SimpleNamespace(decisions=(decision,)),
            pairwise_model_result=SimpleNamespace(row_signals={}),
            decision_row_signals={
                "constraint_require_count": np.asarray([require_count], dtype=np.float32),
            },
        )
        return production_module._scored_query_decisions_from_result(  # noqa: SLF001
            result,
            signature_ids_by_index=[signature_id],
            expected_query_signature_ids=[signature_id],
        )

    initial = {
        **extract_batch("q-require", 0.70, 1.0),
        **extract_batch("q-score", 0.99, 0.0),
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

    assert linked == {"q-require": "shared"}
    assert rescored == ["q-score"]
    assert telemetry["global_query_disallow_rescore_count"] == 1


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
