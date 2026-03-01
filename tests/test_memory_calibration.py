from s2and.memory_calibration import extract_phase_a_sample


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
