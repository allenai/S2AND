from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TypedDict

import pytest

from s2and import memory_budget


def _compute_promoted_phase_a_limits(**kwargs):
    return memory_budget.compute_promoted_phase_a_limits(
        final_matrix_feature_count=53,
        pairwise_matrix_feature_count=35,
        aggregate_feature_count=18,
        **kwargs,
    )


class _MemoryBudgetKwargs(TypedDict):
    total_ram_bytes: int
    detect_cgroup_fn: Callable[[], tuple[int | None, str]]
    detect_total_fn: Callable[[], tuple[int | None, str]]
    current_rss_fn: Callable[[int], tuple[int, str]]


class _RustBatchPlanKwargs(_MemoryBudgetKwargs):
    base_chunk_pairs: int
    row_overhead_bytes: int


def test_resolve_total_ram_arg_overrides_autodetect():
    resolved, source = memory_budget.resolve_total_ram_bytes(
        4096,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
    )
    assert resolved == 4096
    assert source == "arg"


def test_emit_memory_telemetry_writes_jsonl(monkeypatch, tmp_path):
    output_path = tmp_path / "memory_telemetry.jsonl"
    monkeypatch.setenv(memory_budget.MEMORY_TELEMETRY_JSONL_ENV, str(output_path))
    memory_budget.emit_memory_telemetry({"stage": "test_stage", "value": 7})

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["event"] == "memory_telemetry"
    assert record["stage"] == "test_stage"
    assert record["value"] == 7


def test_emit_memory_telemetry_ignores_blank_env(monkeypatch, tmp_path):
    output_path = tmp_path / "memory_telemetry_env.jsonl"
    monkeypatch.setenv(memory_budget.MEMORY_TELEMETRY_JSONL_ENV, " ")
    assert memory_budget.memory_telemetry_jsonl_path() is None
    memory_budget.emit_memory_telemetry({"stage": "env_stage", "value": 11})
    assert not output_path.exists()


def test_resolve_total_ram_cgroup_uses_safety_factor():
    resolved, source = memory_budget.resolve_total_ram_bytes(
        None,
        detect_cgroup_fn=lambda: (10_000, "cgroup:test"),
        detect_total_fn=lambda: (None, "unavailable"),
        autodetect_safety_factor=0.8,
    )
    assert resolved == 8000
    assert source == "cgroup:test_80pct"


def test_resolve_total_ram_windows_fallback_uses_safety_factor(monkeypatch):
    monkeypatch.setattr(memory_budget, "_psutil_virtual_memory_total_bytes_best_effort", lambda: None)
    monkeypatch.setattr(memory_budget, "_is_windows", lambda: True)
    monkeypatch.setattr(memory_budget, "_windows_total_ram_bytes_best_effort", lambda: (10_000, "winapi:test"))
    resolved, source = memory_budget.resolve_total_ram_bytes(
        None,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
    )
    assert resolved == 8000
    assert source == "winapi:test_80pct"


def test_current_rss_windows_fallback_used_when_psutil_missing(monkeypatch):
    monkeypatch.setattr(memory_budget, "_psutil_process_rss_bytes_best_effort", lambda: None)
    monkeypatch.setattr(memory_budget, "_is_windows", lambda: True)
    monkeypatch.setattr(
        memory_budget,
        "_windows_process_working_set_bytes_best_effort",
        lambda: (654_321, "winapi:test"),
    )
    rss, source = memory_budget.current_rss_bytes_best_effort(10_000_000)
    assert rss == 654_321
    assert source == "winapi:test"


def test_compute_rust_batch_chunk_plan_respects_stage_budget():
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=1_000,
        total_pairs=20_000,
        total_ram_bytes=1_000_000,
        stage_budget_fraction=0.10,
        base_chunk_pairs=10_000,
        row_overhead_bytes=128,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    # available = 700_000 -> stage budget = 70_000 bytes. The configured fixed
    # overhead already exceeds that budget, so the helper keeps the chunk minimal.
    assert int(plan.chunk_pairs) == 1
    assert int(plan.predicted_chunk_bytes) == int(plan.bytes_per_pair_row)
    assert int(plan.predicted_stage_peak_rss_bytes) >= int(plan.predicted_stage_peak_delta_bytes)
    assert int(plan.predicted_stage_peak_delta_bytes) >= int(plan.predicted_chunk_bytes)


def test_compute_rust_batch_chunk_plan_accounts_for_borrowed_index_remap_arrays():
    common_kwargs = {
        "num_features": 10,
        "total_pairs": 1_000,
        "total_rows": 10,
        "total_ram_bytes": 10_000_000_000,
        "base_chunk_pairs": 0,
        "max_chunk_pairs": 0,
        "fixed_overhead_bytes": 0,
        "detect_cgroup_fn": lambda: (None, "unavailable"),
        "detect_total_fn": lambda: (None, "unavailable"),
        "current_rss_fn": lambda _total: (0, "rss:test"),
    }
    owned = memory_budget.compute_rust_batch_chunk_plan(**common_kwargs)
    borrowed = memory_budget.compute_rust_batch_chunk_plan(
        **common_kwargs,
        index_remap_bytes_per_pair=memory_budget.BORROWED_SIGNATURE_INDEX_REMAP_BYTES_PER_PAIR,
    )

    assert owned.chunk_pairs == borrowed.chunk_pairs == 1_000
    assert owned.index_remap_bytes_per_pair == 0
    assert owned.predicted_index_remap_bytes == 0
    assert borrowed.index_remap_bytes_per_pair == 8
    assert borrowed.predicted_index_remap_bytes == 8_000
    assert borrowed.bytes_per_pair_row == owned.bytes_per_pair_row + 8
    assert borrowed.predicted_chunk_bytes == owned.predicted_chunk_bytes + 8_000


def test_compute_rust_batch_chunk_plan_rejects_negative_index_remap_bytes():
    with pytest.raises(ValueError, match="index_remap_bytes_per_pair must be non-negative"):
        memory_budget.compute_rust_batch_chunk_plan(
            num_features=1,
            total_pairs=1,
            index_remap_bytes_per_pair=-1,
        )


def test_compute_rust_batch_chunk_plan_base_chunk_pairs_zero_allows_full_call():
    plan_big = memory_budget.compute_rust_batch_chunk_plan(
        num_features=1,
        total_pairs=500,
        total_ram_bytes=10_000_000_000,
        base_chunk_pairs=0,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )
    assert int(plan_big.chunk_pairs) == 500  # total_pairs is the only limit


def test_compute_rust_batch_chunk_plan_respects_named_max_chunk_pairs():
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=1,
        total_pairs=500_000,
        total_ram_bytes=10_000_000_000,
        base_chunk_pairs=0,
        max_chunk_pairs=10_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )

    derived_chunk_pairs = max(
        1,
        max(1, int(plan.stage_budget_bytes) - int(plan.fixed_overhead_bytes)) // max(1, int(plan.bytes_per_pair_row)),
    )
    assert derived_chunk_pairs > 10_000
    assert int(plan.max_chunk_pairs) == 10_000
    assert int(plan.chunk_pairs) == 10_000


def test_compute_promoted_phase_a_limits_uses_top_k_largest_components():
    limits = _compute_promoted_phase_a_limits(
        query_count=20,
        component_sizes=[100, 50, 25, 10],
        retrieval_top_k=3,
        total_ram_bytes=1_000_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1024,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )

    assert set(vars(limits)) == {
        "query_batch_size",
        "predicted_peak_delta_bytes",
        "predicted_peak_rss_bytes",
    }
    assert int(limits.query_batch_size) == 20
    assert int(limits.predicted_peak_delta_bytes) > 0
    assert int(limits.predicted_peak_rss_bytes) == 100_000_000 + int(limits.predicted_peak_delta_bytes)


def test_promoted_component_size_summary_is_reused_without_reinspecting_mapping():
    class CountingSizes(dict[str, int]):
        values_call_count = 0

        def values(self):
            self.values_call_count += 1
            return super().values()

    component_sizes = CountingSizes({f"c{index}": index + 1 for index in range(1_000)})
    summary = memory_budget.summarize_promoted_component_sizes(component_sizes)

    raw_limits = _compute_promoted_phase_a_limits(
        query_count=20,
        component_sizes=component_sizes,
        retrieval_top_k=25,
        total_ram_bytes=1_000_000_000,
        fixed_overhead_bytes=1024,
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )
    summarized_limits = [
        _compute_promoted_phase_a_limits(
            query_count=20,
            component_sizes=summary,
            retrieval_top_k=25,
            total_ram_bytes=1_000_000_000,
            fixed_overhead_bytes=1024,
            current_rss_fn=lambda _total: (100_000_000, "rss:test"),
        )
        for _ in range(3)
    ]

    assert component_sizes.values_call_count == 2
    assert all(limits == raw_limits for limits in summarized_limits)
    assert summary.component_count == 1_000
    assert summary.max_component_size == 1_000
    assert summary.top_k_totals(25) == (25, sum(range(976, 1_001)))


def test_promoted_resident_retrieval_payload_is_not_reserved_twice():
    kwargs = {
        "query_count": 4,
        "component_sizes": [10, 8, 6, 4],
        "retrieval_top_k": 4,
        "total_ram_bytes": 100_000_000,
        "max_query_batch_size": 4,
        "detect_cgroup_fn": lambda: (None, "unavailable"),
        "detect_total_fn": lambda: (None, "unavailable"),
        "current_rss_fn": lambda _total: (10_000_000, "rss:test"),
    }
    pending = _compute_promoted_phase_a_limits(**kwargs)
    resident = _compute_promoted_phase_a_limits(**kwargs, retrieval_payload_resident=True)

    assert resident.predicted_peak_delta_bytes < pending.predicted_peak_delta_bytes
    assert resident.predicted_peak_rss_bytes == 10_000_000 + resident.predicted_peak_delta_bytes
    assert pending.predicted_peak_rss_bytes == 10_000_000 + pending.predicted_peak_delta_bytes


def test_native_scorer_chunk_plan_handles_full_chunking_and_exhaustion() -> None:
    plan = memory_budget.compute_native_scorer_chunk_plan(
        row_count=1_000,
        feature_count=53,
        total_ram_bytes=100_000_000,
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )

    assert plan.chunk_rows == 1_000
    assert plan.chunk_count == 1
    assert plan.predicted_peak_delta_bytes == 1_000 * (53 * 4 + 8)

    plan = memory_budget.compute_native_scorer_chunk_plan(
        row_count=100_000,
        feature_count=70,
        total_ram_bytes=10_000_000,
        current_rss_fn=lambda _total: (1_000_000, "rss:test"),
    )

    assert 1 <= plan.chunk_rows < 100_000
    assert plan.chunk_count > 1
    assert plan.predicted_peak_delta_bytes <= plan.stage_budget_bytes

    with pytest.raises(MemoryError, match="cannot fit one scratch row"):
        memory_budget.compute_native_scorer_chunk_plan(
            row_count=10_000,
            feature_count=70,
            total_ram_bytes=1_000_000,
            safety_margin_fraction=0.1,
            stage_budget_fraction=0.5,
            current_rss_fn=lambda _total: (800_000, "rss:test"),
        )


def test_compute_promoted_phase_a_limits_shrinks_query_batch_under_tight_budget():
    limits = _compute_promoted_phase_a_limits(
        query_count=100,
        component_sizes=[20_000, 16_000, 12_000, 8_000, 4_000],
        retrieval_top_k=5,
        total_ram_bytes=100_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )

    assert int(limits.query_batch_size) < 100
    assert int(limits.query_batch_size) >= 1
    assert int(limits.predicted_peak_rss_bytes) > 10_000_000


def test_compute_promoted_phase_a_limits_uses_orcid_fanout_floor_above_top_k():
    baseline = _compute_promoted_phase_a_limits(
        query_count=10,
        component_sizes=[1] * 100,
        retrieval_top_k=25,
        total_ram_bytes=1_000_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1024,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )
    limits = _compute_promoted_phase_a_limits(
        query_count=10,
        component_sizes=[1] * 100,
        retrieval_top_k=25,
        total_ram_bytes=1_000_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1024,
        candidate_rows_per_query_floor=80,
        pairs_per_query_floor=80,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )

    assert int(limits.query_batch_size) == 10
    assert limits.predicted_peak_delta_bytes > baseline.predicted_peak_delta_bytes


def test_compute_promoted_phase_a_limits_uses_orcid_total_floor_for_mixed_batch():
    per_query_limits = _compute_promoted_phase_a_limits(
        query_count=10,
        component_sizes=[1] * 100,
        retrieval_top_k=25,
        total_ram_bytes=1_000_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1024,
        candidate_rows_per_query_floor=80,
        pairs_per_query_floor=80,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )
    limits = _compute_promoted_phase_a_limits(
        query_count=10,
        component_sizes=[1] * 100,
        retrieval_top_k=25,
        total_ram_bytes=1_000_000_000,
        stage_budget_fraction=0.50,
        fixed_overhead_bytes=1024,
        candidate_rows_per_query_floor=80,
        pairs_per_query_floor=80,
        candidate_rows_total_floor=305,
        pairs_total_floor=305,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )

    assert int(limits.query_batch_size) == 10
    assert limits.predicted_peak_delta_bytes < per_query_limits.predicted_peak_delta_bytes


def test_compute_promoted_phase_a_limits_fails_when_single_query_exceeds_budget():
    with pytest.raises(MemoryError, match="cannot fit a single query"):
        _compute_promoted_phase_a_limits(
            query_count=10,
            component_sizes=[2_000_000],
            retrieval_top_k=1,
            total_ram_bytes=100_000_000,
            stage_budget_fraction=0.10,
            fixed_overhead_bytes=1_000_000,
            detect_cgroup_fn=lambda: (None, "unavailable"),
            detect_total_fn=lambda: (None, "unavailable"),
            current_rss_fn=lambda _total: (10_000_000, "rss:test"),
        )


def test_compute_promoted_phase_a_limits_zero_queries_skip_work_and_single_query_budget():
    # All-seeded incremental request: every signature is already in a seed component,
    # so query_count == 0 and no batching_threshold is forwarded. This must not raise;
    # the planner short-circuits to a zero-size batch (no work to do).
    limits = _compute_promoted_phase_a_limits(
        query_count=0,
        component_sizes=[4],
        retrieval_top_k=50,
        total_ram_bytes=8_000_000_000,
        max_query_batch_size=None,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (100_000_000, "rss:test"),
    )

    assert int(limits.query_batch_size) == 0

    limits = _compute_promoted_phase_a_limits(
        query_count=0,
        component_sizes=[4],
        retrieval_top_k=50,
        total_ram_bytes=100_000_000,
        stage_budget_fraction=0.10,
        fixed_overhead_bytes=20_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )

    assert int(limits.query_batch_size) == 0
    assert int(limits.predicted_peak_delta_bytes) >= 20_000_000


def test_compute_promoted_phase_a_limits_rejects_explicit_nonpositive_threshold():
    # An explicit non-positive caller value is always invalid, including when
    # query_count == 0 (it must not be silently coerced to a 1-size batch).
    for query_count in (0, 5):
        with pytest.raises(ValueError, match="max_query_batch_size must be positive"):
            _compute_promoted_phase_a_limits(
                query_count=query_count,
                component_sizes=[4],
                retrieval_top_k=50,
                total_ram_bytes=8_000_000_000,
                max_query_batch_size=0,
                detect_cgroup_fn=lambda: (None, "unavailable"),
                detect_total_fn=lambda: (None, "unavailable"),
                current_rss_fn=lambda _total: (100_000_000, "rss:test"),
            )


def test_summarize_prediction_accuracy_flags_underprediction():
    summary = memory_budget.summarize_prediction_accuracy(
        stage_name="test_stage",
        predicted_peak_delta_bytes=100,
        rss_before_bytes=1000,
        rss_peak_bytes=1200,
        rss_after_bytes=1100,
    )
    assert int(summary.predicted_peak_delta_bytes) == 100
    assert int(summary.predicted_peak_rss_bytes) == 1100
    assert int(summary.observed_peak_delta_bytes) == 200
    assert bool(summary.underpredicted) is True
    assert float(summary.prediction_error_ratio) == 2.0


def test_compute_rust_batch_chunk_plan_adds_persistent_row_overhead():
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=16,
        total_pairs=100,
        total_rows=100,
        selected_feature_count=8,
        nameless_feature_count=2,
        total_ram_bytes=10_000_000,
        base_chunk_pairs=50,
        row_overhead_bytes=32,
        persistent_row_overhead_bytes=12,
        fixed_overhead_bytes=2048,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (1_000_000, "rss:test"),
    )
    assert int(plan.predicted_persistent_row_overhead_bytes) == 1200
    assert int(plan.predicted_fixed_overhead_bytes) == 2048
    expected = (
        int(plan.predicted_features_matrix_bytes)
        + int(plan.predicted_labels_bytes)
        + int(plan.predicted_chunk_bytes)
        + int(plan.predicted_persistent_row_overhead_bytes)
        + int(plan.predicted_fixed_overhead_bytes)
    )
    assert int(plan.predicted_stage_peak_delta_bytes) == expected


def test_rss_fallback_logs_warning(caplog, monkeypatch):
    """Fix #1: RSS fallback should log a warning when using the 50% assumption."""
    monkeypatch.setattr(memory_budget, "_psutil_process_rss_bytes_best_effort", lambda: None)
    monkeypatch.setattr(memory_budget, "_proc_status_rss_bytes_best_effort", lambda: (None, "unavailable"))
    monkeypatch.setattr(memory_budget, "_windows_process_working_set_bytes_best_effort", lambda: (None, "unavailable"))

    with caplog.at_level(logging.WARNING, logger="s2and"):
        rss, source = memory_budget.current_rss_bytes_best_effort(2_000_000)
    assert rss == 1_000_000
    assert source == "fallback_half_total"
    assert any("falling back to 50%" in r.message for r in caplog.records)


def test_rust_batch_selected_features_tighter_chunk_sizing():
    """Fix #6: Rust batch should use selected+nameless for chunk sizing, not full_feature_count."""
    common_kwargs: _RustBatchPlanKwargs = dict(
        total_ram_bytes=10_000_000,
        base_chunk_pairs=100_000,
        row_overhead_bytes=128,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (1_000_000, "rss:test"),
    )
    # No selected_feature_count (legacy): chunk_feature_count = full = 100
    # bytes_per_pair_row = 100*8 + 128 = 928
    legacy = memory_budget.compute_rust_batch_chunk_plan(num_features=100, total_pairs=50_000, **common_kwargs)
    assert int(legacy.bytes_per_pair_row) == 100 * 8 + 128

    # With selected=20, nameless=5: chunk_feature_count = 25
    # bytes_per_pair_row = 25*8 + 128 = 328
    selected = memory_budget.compute_rust_batch_chunk_plan(
        num_features=100,
        total_pairs=50_000,
        selected_feature_count=20,
        nameless_feature_count=5,
        **common_kwargs,
    )
    assert int(selected.bytes_per_pair_row) == 25 * 8 + 128
    # Tighter row estimate -> more pairs per chunk
    assert int(selected.chunk_pairs) >= int(legacy.chunk_pairs)


def test_degenerate_budget_logs_warning(caplog):
    """When RSS exceeds total_ram - safety_margin, a warning should be logged."""
    with caplog.at_level(logging.WARNING, logger="s2and"):
        snapshot = memory_budget.memory_snapshot_for_stage(
            total_ram_bytes=1_000_000,
            safety_margin_fraction=0.10,
            detect_cgroup_fn=lambda: (None, "unavailable"),
            detect_total_fn=lambda: (None, "unavailable"),
            # RSS > total - safety => degenerate
            current_rss_fn=lambda _total: (950_000, "rss:test"),
        )
    assert snapshot.available_bytes == 1
    assert any("degenerate" in r.message for r in caplog.records)


def test_effective_available_fraction_in_snapshot():
    """MemorySnapshot should include effective_available_fraction."""
    snapshot = memory_budget.memory_snapshot_for_stage(
        total_ram_bytes=1_000_000,
        safety_margin_fraction=0.10,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    # available = 1_000_000 - 200_000 - 100_000 = 700_000
    assert snapshot.available_bytes == 700_000
    expected_fraction = 700_000 / 1_000_000
    assert abs(snapshot.effective_available_fraction - expected_fraction) < 1e-6
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=64,
        total_pairs=1000,
        total_ram_bytes=1_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    assert float(plan.effective_available_fraction) == pytest.approx(0.7, abs=1e-6)
