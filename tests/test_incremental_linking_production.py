from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import s2and.incremental_linking.production as production_module
from s2and.incremental_linking.runtime import LinkOrAbstainDecision


@pytest.mark.parametrize("query_disallow", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("external_fallback", [False, True])
def test_altered_profile_disallows_survive_native_planning_and_restoration(
    tmp_path, monkeypatch, query_disallow, batch_size, external_fallback
) -> None:
    """Exercise production sidecars, native retrieval, conflict rescore, and finalization."""
    from s2and.arrow_inputs import ArrowDataset
    from s2and.featurizer import FeaturizationInfo
    from s2and.model import Clusterer
    from tests.helpers import write_minimal_arrow_prediction_bundle

    write_minimal_arrow_prediction_bundle(tmp_path)
    queries = ["q1", "q2"] if query_disallow else ["q1"]
    disallow = ("q1", "q2") if query_disallow else ("0", "q1")
    seeds = {"0": "claimed_0", "1": "claimed_1"}
    original_inverse = {"claimed": ["0", "1"]}
    split_inverse = {"claimed_0": ["0"], "claimed_1": ["1"]}
    if external_fallback:
        seeds["2"] = "outside"
        original_inverse["outside"] = ["2"]
        split_inverse["outside"] = ["2"]
    dataset = production_module._DirectArrowIncrementalDataset(
        name_tuples=set(),
        cluster_seeds_require={"0": "claimed", "1": "claimed"},
        cluster_seeds_disallow={disallow},
        altered_cluster_signatures=["0"],
        max_seed_cluster_id=0,
        signatures={},
    )

    class SplitClusterer:
        n_jobs = 1
        suppress_orcid = True
        featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
        _finish_incremental_with_seed_links = Clusterer._finish_incremental_with_seed_links

        def _build_incremental_seed_setup(self, *args, **kwargs):
            return (
                seeds,
                {"claimed_0": "claimed", "claimed_1": "claimed"},
                original_inverse,
                split_inverse,
            )

    scored_plans = []

    def score_available_candidate(*args, **kwargs):
        plan = kwargs["raw_plan_bundle"]
        scored_plans.append((plan.query_signature_ids, plan.row_component_keys))
        signature_ids = kwargs["rust_featurizer"].signature_ids()
        decisions = []
        for offset, query in enumerate(plan.query_signature_ids):
            rows = np.flatnonzero(plan.row_query_offsets == offset)
            preferred = "claimed_0" if query == "q1" else "claimed_1"
            matching_rows = [int(row) for row in rows if plan.row_component_keys[int(row)] == preferred]
            row = matching_rows[0] if matching_rows else (int(rows[0]) if len(rows) else None)
            component = plan.row_component_keys[row] if row is not None else None
            decisions.append(
                LinkOrAbstainDecision(
                    signature_ids.index(query),
                    "link" if component else "abstain",
                    row,
                    component,
                    0.95 if query == "q1" else 0.90,
                    None,
                    None,
                )
            )
        return SimpleNamespace(
            compact_result=SimpleNamespace(decisions=decisions),
            decision_row_signals={},
            linked_signature_clusters={
                signature_ids[d.query_signature_index]: d.component_key for d in decisions if d.action == "link"
            },
            telemetry={
                "query_count": len(decisions),
                "candidate_row_count": plan.row_count,
                "pair_count": plan.pair_count,
            },
        )

    monkeypatch.setattr(
        production_module.feature_port,
        "build_rust_featurizer_from_arrow_dataset",
        lambda dataset, **kwargs: SimpleNamespace(signature_ids=lambda: list(kwargs["signature_ids"])),
    )
    monkeypatch.setattr(
        production_module.runtime_module,
        "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
        score_available_candidate,
    )
    with ArrowDataset.open(tmp_path) as arrow_dataset:
        result = production_module.predict_incremental_promoted_linker_from_arrow(
            SplitClusterer(),
            [*seeds, *queries],
            dataset,
            arrow_dataset=arrow_dataset,
            artifact=SimpleNamespace(artifact_dir=tmp_path, retrieval_top_k=2, feature_columns=("test",)),
            prevent_new_incompatibilities=False,
            partial_supervision={},
            runtime_context=SimpleNamespace(run_id="restored-disallow"),
            total_ram_bytes=32 * 1024**3,
            batching_threshold=batch_size,
        )
    assert not any(set(disallow) <= set(members) for members in result["clusters"].values())
    assert set(result["clusters"]["claimed"]) == ({"0", "1", "q1"} if query_disallow else {"0", "1"})
    if query_disallow:
        assert result["incremental_linker_telemetry"]["global_query_disallow_rescore_count"] == 1
        assert scored_plans[-1] == (("q2",), ("outside",) if external_fallback else ())
    else:
        assert scored_plans == [(("q1",), ("outside",) if external_fallback else ())]
    if external_fallback:
        assert set(result["clusters"]["outside"]) == {"2", "q2" if query_disallow else "q1"}


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
