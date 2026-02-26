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
    assert int(limits["chunk_budget_bytes"]) == int(0.60 * 700_000)
    assert int(limits["accumulator_budget_bytes"]) == int(0.20 * 700_000)
    assert int(limits["chunk_pairs"]) >= 1


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


def test_resolve_rust_batch_prediction_params_env(monkeypatch):
    monkeypatch.setenv("S2AND_RUST_BATCH_BASE_CHUNK_PAIRS", "123")
    monkeypatch.setenv("S2AND_RUST_BATCH_ROW_OVERHEAD_BYTES", "45")
    monkeypatch.setenv("S2AND_RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES", "67")
    monkeypatch.setenv("S2AND_RUST_BATCH_FIXED_OVERHEAD_BYTES", "890")

    params = memory_budget.resolve_rust_batch_prediction_params()

    assert params["base_chunk_pairs"] == 123
    assert params["row_overhead_bytes"] == 45
    assert params["persistent_row_overhead_bytes"] == 67
    assert params["fixed_overhead_bytes"] == 890


def test_resolve_rust_batch_prediction_params_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("S2AND_RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES", "-1")
    with pytest.raises(ValueError, match="S2AND_RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES"):
        memory_budget.resolve_rust_batch_prediction_params()


def test_rss_fallback_logs_warning(caplog, monkeypatch):
    """Fix #1: RSS fallback should log a warning when using the 50% assumption."""
    import builtins
    import os

    _real_import = builtins.__import__

    def _no_psutil_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("mocked")
        return _real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_psutil_import)
    # Ensure /proc/self/status doesn't exist (it won't on Windows, but be explicit)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

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
    legacy = memory_budget.compute_rust_batch_chunk_plan(
        num_features=100, total_pairs=50_000, **common_kwargs
    )
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


def test_fallback_accumulator_max_entries_exists():
    """Fix #5: FALLBACK_ACCUMULATOR_MAX_ENTRIES constant should be defined."""
    assert hasattr(memory_budget, "FALLBACK_ACCUMULATOR_MAX_ENTRIES")
    assert memory_budget.FALLBACK_ACCUMULATOR_MAX_ENTRIES > 0
