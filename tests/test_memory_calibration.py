from __future__ import annotations

import json

import pytest

from s2and.memory_calibration import (
    effective_accumulator_entry_bytes,
    effective_rust_batch_persistent_row_overhead_bytes,
    extract_phase_a_sample,
    extract_rust_batch_sample,
    iter_memory_telemetry_records,
)


def test_iter_memory_telemetry_records_reads_jsonl_and_skips_blanks():
    records = list(
        iter_memory_telemetry_records(
            [
                json.dumps({"stage": "phase_a_seed_distances", "prediction_error_ratio": 1.25}),
                "",
                "  ",
                json.dumps({"stage": "pair_featurization_rust_batch", "prediction_error_ratio": 0.9}),
            ]
        )
    )

    assert [record["stage"] for record in records] == [
        "phase_a_seed_distances",
        "pair_featurization_rust_batch",
    ]


def test_iter_memory_telemetry_records_rejects_invalid_jsonl():
    with pytest.raises(ValueError, match="Invalid memory telemetry JSONL at line 2"):
        list(iter_memory_telemetry_records([json.dumps({"stage": "ok"}), "not json"]))


def test_extract_phase_a_sample_prefers_sampled_accumulator_count():
    record = {
        "stage": "phase_a_seed_distances",
        "chunk_features_peak_bytes": 1000,
        "phase_a_pair_buffer_peak_bytes": 200,
        "phase_a_fixed_overhead_bytes": 50,
        "observed_peak_delta_bytes": 2450,
        "accumulator_entries_peak": 60,
        "accumulator_entries_peak_sample": 6,
    }

    sample = extract_phase_a_sample(record)

    assert sample is not None
    assert sample.accumulator_entries_peak == 6
    assert effective_accumulator_entry_bytes(sample) == pytest.approx(200.0)
    assert extract_phase_a_sample({"stage": "pair_featurization_rust_batch"}) is None


def test_extract_phase_a_sample_uses_accumulator_count_when_sample_missing():
    record = {
        "stage": "phase_a_seed_distances",
        "chunk_features_peak_bytes": 100,
        "phase_a_fixed_overhead_bytes": 200,
        "observed_peak_delta_bytes": 900,
        "accumulator_entries_peak": 50,
    }

    sample = extract_phase_a_sample(record)

    assert sample is not None
    assert sample.accumulator_entries_peak == 50
    assert sample.phase_a_pair_buffer_peak_bytes == 0
    assert effective_accumulator_entry_bytes(sample) == pytest.approx(12.0)


def test_extract_phase_a_sample_rejects_malformed_matching_stage():
    with pytest.raises(ValueError, match="Invalid phase_a_seed_distances memory telemetry record"):
        extract_phase_a_sample(
            {
                "stage": "phase_a_seed_distances",
                "chunk_features_peak_bytes": 100,
                "phase_a_fixed_overhead_bytes": 200,
                "observed_peak_delta_bytes": 900,
            }
        )


def test_extract_rust_batch_sample_parses_pair_featurization_memory_contract():
    record = {
        "stage": "pair_featurization_rust_batch",
        "total_rows": 10,
        "predicted_features_matrix_bytes": 100,
        "predicted_labels_bytes": 20,
        "predicted_chunk_bytes": 30,
        "predicted_fixed_overhead_bytes": 40,
        "observed_peak_delta_bytes": 1190,
    }

    sample = extract_rust_batch_sample(record)

    assert sample is not None
    assert sample.total_rows == 10
    assert sample.predicted_features_matrix_bytes == 100
    assert sample.predicted_labels_bytes == 20
    assert sample.predicted_chunk_bytes == 30
    assert sample.predicted_fixed_overhead_bytes == 40
    assert sample.observed_peak_delta_bytes == 1190
    assert effective_rust_batch_persistent_row_overhead_bytes(sample) == pytest.approx(100.0)
    assert extract_rust_batch_sample({"stage": "phase_a_seed_distances"}) is None


def test_extract_rust_batch_sample_rejects_malformed_matching_stage():
    with pytest.raises(ValueError, match="Invalid pair_featurization_rust_batch memory telemetry record"):
        extract_rust_batch_sample(
            {
                "stage": "pair_featurization_rust_batch",
                "total_rows": True,
                "predicted_features_matrix_bytes": 100,
                "predicted_labels_bytes": 20,
                "predicted_chunk_bytes": 30,
                "predicted_fixed_overhead_bytes": 40,
                "observed_peak_delta_bytes": 1190,
            }
        )
