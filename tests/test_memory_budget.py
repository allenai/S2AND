from __future__ import annotations

import logging

import pytest

from s2and import memory_budget


def test_resolve_total_ram_arg_overrides_autodetect():
    resolved, source = memory_budget.resolve_total_ram_bytes(
        4096,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
    )
    assert resolved == 4096
    assert source == "arg"


def test_resolve_total_ram_cgroup_uses_safety_factor():
    resolved, source = memory_budget.resolve_total_ram_bytes(
        None,
        detect_cgroup_fn=lambda: (10_000, "cgroup:test"),
        detect_total_fn=lambda: (None, "unavailable"),
    )
    assert resolved == 8000
    assert source == "cgroup:test_80pct"


def test_detect_total_ram_windows_fallback_used_when_psutil_missing(monkeypatch):
    monkeypatch.setattr(memory_budget, "_psutil_virtual_memory_total_bytes_best_effort", lambda: None)
    monkeypatch.setattr(memory_budget, "_is_windows", lambda: True)
    monkeypatch.setattr(memory_budget, "_windows_total_ram_bytes_best_effort", lambda: (123_456, "winapi:test"))
    total, source = memory_budget.detect_total_ram_bytes_best_effort()
    assert total == 123_456
    assert source == "winapi:test"


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


def test_resolve_total_ram_source_suffix_tracks_safety_factor():
    resolved, source = memory_budget.resolve_total_ram_bytes(
        None,
        detect_cgroup_fn=lambda: (10_000, "cgroup:test"),
        detect_total_fn=lambda: (None, "unavailable"),
        autodetect_safety_factor=0.75,
    )
    assert resolved == 7500
    assert source == "cgroup:test_75pct"


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


def test_compute_incremental_limits_uses_available_bytes():
    limits = memory_budget.compute_incremental_phase_split_limits(
        num_features=64,
        total_ram_bytes=1_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    # 10% safety margin on total_ram_bytes=1_000_000 => 100_000
    # available = 1_000_000 - 200_000 - 100_000 = 700_000
    assert int(limits["available_bytes"]) == 700_000
    assert 0 < int(limits["accumulator_budget_bytes"]) < int(limits["available_bytes"])
    assert int(limits["accumulator_budget_bytes"]) < int(limits["chunk_budget_bytes"])
    assert int(limits["chunk_budget_bytes"]) < int(limits["available_bytes"])
    assert int(limits["chunk_pairs"]) >= 1


def test_compute_incremental_limits_uses_default_phase_a_chunk_cap():
    limits = memory_budget.compute_incremental_phase_split_limits(
        num_features=1,
        total_ram_bytes=10_000_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )
    assert int(limits["derived_chunk_pairs"]) > memory_budget.PHASE_A_MAX_CHUNK_PAIRS_DEFAULT
    assert int(limits["max_chunk_pairs"]) == memory_budget.PHASE_A_MAX_CHUNK_PAIRS_DEFAULT
    assert int(limits["chunk_pairs"]) == memory_budget.PHASE_A_MAX_CHUNK_PAIRS_DEFAULT


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
    # available = 700_000 -> stage budget = 70_000 bytes
    # bytes_per_pair_row ~= 8,128 => chunk_pairs should clamp to 8
    assert int(plan["chunk_pairs"]) == 8
    assert int(plan["predicted_chunk_bytes"]) == 8 * int(plan["bytes_per_pair_row"])
    assert int(plan["predicted_stage_peak_delta_bytes"]) == int(plan["predicted_stage_peak_bytes"])
    assert int(plan["predicted_stage_peak_rss_bytes"]) >= int(plan["predicted_stage_peak_delta_bytes"])
    assert int(plan["predicted_stage_peak_bytes"]) >= int(plan["predicted_chunk_bytes"])


def test_compute_rust_batch_chunk_plan_base_chunk_pairs_zero_disables_floor():
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=1_000,
        total_pairs=20_000,
        total_ram_bytes=1_000_000,
        stage_budget_fraction=0.10,
        base_chunk_pairs=0,
        row_overhead_bytes=128,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    # Same setup as test_compute_rust_batch_chunk_plan_respects_stage_budget but
    # base_chunk_pairs=0 means the floor is disabled. chunk_pairs should equal
    # min(total_pairs, derived_chunk_pairs) — the base floor is not a candidate.
    assert int(plan["chunk_pairs"]) == 8  # still budget-limited, same as before
    # Now verify with generous RAM so derived_chunk_pairs > total_pairs:
    # chunk_pairs should equal total_pairs (not clamped by a base floor).
    plan_big = memory_budget.compute_rust_batch_chunk_plan(
        num_features=1,
        total_pairs=500,
        total_ram_bytes=10_000_000_000,
        base_chunk_pairs=0,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )
    assert int(plan_big["chunk_pairs"]) == 500  # total_pairs is the only limit


def test_summarize_prediction_accuracy_flags_underprediction():
    summary = memory_budget.summarize_prediction_accuracy(
        stage_name="test_stage",
        predicted_peak_delta_bytes=100,
        rss_before_bytes=1000,
        rss_peak_bytes=1200,
        rss_after_bytes=1100,
    )
    assert str(summary["prediction_contract_version"]) == "delta_v1"
    assert int(summary["predicted_peak_delta_bytes"]) == 100
    assert int(summary["predicted_peak_rss_bytes"]) == 1100
    assert int(summary["predicted_bytes"]) == 100
    assert int(summary["observed_peak_delta_bytes"]) == 200
    assert bool(summary["underpredicted"]) is True
    assert float(summary["prediction_error_ratio"]) == 2.0


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
    assert int(plan["predicted_persistent_row_overhead_bytes"]) == 1200
    assert int(plan["predicted_fixed_overhead_bytes"]) == 2048
    expected = (
        int(plan["predicted_features_matrix_bytes"])
        + int(plan["predicted_labels_bytes"])
        + int(plan["predicted_chunk_bytes"])
        + int(plan["predicted_persistent_row_overhead_bytes"])
        + int(plan["predicted_fixed_overhead_bytes"])
    )
    assert int(plan["predicted_stage_peak_delta_bytes"]) == expected


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


def test_incremental_limits_with_selected_feature_counts():
    """Fix #2: selected_feature_count should produce tighter bytes_per_pair."""
    common_kwargs = dict(
        total_ram_bytes=100_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (10_000_000, "rss:test"),
    )
    # Legacy (no selected counts): bytes_per_pair = 64*8*2 + 8 + 100 = 1132
    legacy = memory_budget.compute_incremental_phase_split_limits(num_features=64, **common_kwargs)
    assert int(legacy["bytes_per_pair"]) == 64 * 8 * 2 + 8 + 100

    # With selected counts: bytes_per_pair = (30+10)*8 + 8 + 100 = 428
    selected = memory_budget.compute_incremental_phase_split_limits(
        num_features=64, selected_feature_count=30, nameless_feature_count=10, **common_kwargs
    )
    assert int(selected["bytes_per_pair"]) == (30 + 10) * 8 + 8 + 100
    # Tighter estimate -> more pairs fit in the same budget
    assert int(selected["chunk_pairs"]) >= int(legacy["chunk_pairs"])


def test_rust_batch_selected_features_tighter_chunk_sizing():
    """Fix #6: Rust batch should use selected+nameless for chunk sizing, not full_feature_count."""
    common_kwargs = dict(
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
    assert int(legacy["bytes_per_pair_row"]) == 100 * 8 + 128

    # With selected=20, nameless=5: chunk_feature_count = 25
    # bytes_per_pair_row = 25*8 + 128 = 328
    selected = memory_budget.compute_rust_batch_chunk_plan(
        num_features=100,
        total_pairs=50_000,
        selected_feature_count=20,
        nameless_feature_count=5,
        **common_kwargs,
    )
    assert int(selected["bytes_per_pair_row"]) == 25 * 8 + 128
    # Tighter row estimate -> more pairs per chunk
    assert int(selected["chunk_pairs"]) >= int(legacy["chunk_pairs"])


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


def test_effective_available_fraction_in_incremental_limits():
    """compute_incremental_phase_split_limits should include effective_available_fraction."""
    limits = memory_budget.compute_incremental_phase_split_limits(
        num_features=64,
        total_ram_bytes=1_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    assert "effective_available_fraction" in limits
    assert float(limits["effective_available_fraction"]) == pytest.approx(0.7, abs=1e-6)


def test_effective_available_fraction_in_rust_batch_plan():
    """compute_rust_batch_chunk_plan should include effective_available_fraction."""
    plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=64,
        total_pairs=1000,
        total_ram_bytes=1_000_000,
        detect_cgroup_fn=lambda: (None, "unavailable"),
        detect_total_fn=lambda: (None, "unavailable"),
        current_rss_fn=lambda _total: (200_000, "rss:test"),
    )
    assert "effective_available_fraction" in plan
    assert float(plan["effective_available_fraction"]) == pytest.approx(0.7, abs=1e-6)


def test_gc_collect_and_log(caplog):
    """gc_collect_and_log should run without error."""
    with caplog.at_level(logging.INFO, logger="s2and"):
        memory_budget.gc_collect_and_log("test_stage")
    # We can't assert exact collection counts, but it should not raise.
