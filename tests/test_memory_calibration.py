from s2and.memory_calibration import (
    extract_phase_a_sample,
    extract_rust_batch_sample,
    iter_log_records,
    parse_kv_tokens,
)


def test_iter_log_records_rejoins_hard_wrapped_logging_lines():
    lines = [
        "2026-02-27 00:00:00,000 - s2and - INFO - Telemetry: phase_split_phase_a chunk_features_peak_bytes=1000",
        "phase_a_pair_buffer_peak_bytes=200 phase_a_fixed_overhead_bytes=50 "
        "observed_peak_delta_bytes=2450 accumulator_entries_peak_sample=6",
    ]
    records = list(iter_log_records(lines))
    assert len(records) == 1
    kv = parse_kv_tokens(records[0])
    assert kv["chunk_features_peak_bytes"] == "1000"
    assert kv["phase_a_pair_buffer_peak_bytes"] == "200"
    assert kv["phase_a_fixed_overhead_bytes"] == "50"
    assert kv["observed_peak_delta_bytes"] == "2450"
    assert kv["accumulator_entries_peak_sample"] == "6"


def test_extract_phase_a_sample_prefers_inferred_when_per_sample_missing_and_consistent():
    line = (
        "Telemetry: phase_split_phase_a "
        "chunk_features_peak_bytes=100 "
        "phase_a_fixed_overhead_bytes=200 "
        "observed_peak_delta_bytes=900 "
        "accumulator_entries_peak=50 "
        "predicted_peak_delta_bytes=800 "
        "accumulator_entry_bytes=10"
    )
    sample = extract_phase_a_sample(line)
    assert sample is not None
    assert sample.accumulator_entries_peak == 50
    assert sample.accumulator_entries_peak_parsed == 50
    assert sample.accumulator_entries_peak_inferred == 50


def test_extract_phase_a_sample_keeps_parsed_when_inference_mismatches():
    line = (
        "Telemetry: phase_split_phase_a "
        "chunk_features_peak_bytes=100 "
        "phase_a_fixed_overhead_bytes=200 "
        "observed_peak_delta_bytes=900 "
        "accumulator_entries_peak=60 "
        "predicted_peak_delta_bytes=800 "
        "accumulator_entry_bytes=10"
    )
    sample = extract_phase_a_sample(line)
    assert sample is not None
    assert sample.accumulator_entries_peak == 60
    assert sample.accumulator_entries_peak_parsed == 60
    assert sample.accumulator_entries_peak_inferred == 50


def test_extract_phase_a_sample_uses_inferred_when_parsed_missing():
    line = (
        "Telemetry: phase_split_phase_a "
        "chunk_features_peak_bytes=100 "
        "phase_a_fixed_overhead_bytes=200 "
        "observed_peak_delta_bytes=900 "
        "predicted_peak_delta_bytes=800 "
        "accumulator_entry_bytes=10"
    )
    sample = extract_phase_a_sample(line)
    assert sample is not None
    assert sample.accumulator_entries_peak == 50
    assert sample.accumulator_entries_peak_parsed is None
    assert sample.accumulator_entries_peak_inferred == 50


def test_extract_rust_batch_sample_parses_pair_featurization_memory_contract():
    line = (
        "Telemetry: pair_featurization_memory "
        "stage=pair_featurization_rust_batch "
        "total_rows=10 "
        "predicted_features_matrix_bytes=100 "
        "predicted_labels_bytes=20 "
        "predicted_chunk_bytes=30 "
        "predicted_fixed_overhead_bytes=40 "
        "observed_peak_delta_bytes=1190"
    )
    sample = extract_rust_batch_sample(line)
    assert sample is not None
    assert sample.total_rows == 10
    assert sample.predicted_features_matrix_bytes == 100
    assert sample.predicted_labels_bytes == 20
    assert sample.predicted_chunk_bytes == 30
    assert sample.predicted_fixed_overhead_bytes == 40
    assert sample.observed_peak_delta_bytes == 1190
