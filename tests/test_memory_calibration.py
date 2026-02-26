from s2and.memory_calibration import (
    effective_accumulator_entry_bytes,
    effective_rust_batch_persistent_row_overhead_bytes,
    extract_phase_a_sample,
    extract_rust_batch_sample,
    iter_log_records,
    percentile,
)


def test_extract_phase_a_sample_and_effective_bytes():
    line = (
        "Telemetry: phase_split_phase_a constraints_pairs_total=10 constraint_chunks_total=1 "
        "accumulator_entries_peak=100 chunk_features_peak_bytes=5000 accumulator_entry_bytes=300 "
        "phase_a_fixed_overhead_bytes=2000 model_predict_seconds=0.123 prediction_contract_version=delta_v1 "
        "predicted_peak_delta_bytes=9000 predicted_peak_rss_bytes=19000 predicted_bytes=9000 "
        "rss_before_bytes=10000 rss_peak_bytes=20000 rss_after_bytes=12000 observed_peak_delta_bytes=10000 "
        "prediction_error_ratio=1.111 underpredicted=False rss_source=psutil_process_rss"
    )
    sample = extract_phase_a_sample(line)
    assert sample is not None
    effective = effective_accumulator_entry_bytes(sample)
    assert effective is not None
    # residual = 10000 - (5000 + 2000) = 3000; entries=100 => 30 bytes/entry
    assert effective == 30.0


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    # k=(5-1)*0.95=3.8 => 4*(1-0.8)+5*0.8 = 4.8
    assert percentile(values, 95) == 4.8


def test_iter_log_records_joins_powershell_wrapped_lines():
    physical_lines = [
        "2026-02-25 18:33:44,471 - s2and - INFO - Telemetry: phase_split_phase_a constraints_pairs_total=18750000\n",
        "constraint_chunks_total=38 accumulator_entries_peak=3000000 chunk_features_peak_bytes=200000000\n",
        "phase_a_fixed_overhead_bytes=2097152 observed_peak_delta_bytes=352874496\n",
        "2026-02-25 18:33:45,133 - s2and - INFO - Next record\n",
    ]
    records = list(iter_log_records(physical_lines))
    assert len(records) == 2

    sample = extract_phase_a_sample(records[0])
    assert sample is not None
    assert sample.accumulator_entries_peak == 3_000_000


def test_extract_phase_a_sample_infers_peak_entries_from_prediction():
    line = (
        "Telemetry: phase_split_phase_a constraints_pairs_total=18750000 constraint_chunks_total=38 "
        "accumulator_entries_peak=3000000 chunk_features_peak_bytes=200000000 accumulator_entry_bytes=300 "
        "phase_a_fixed_overhead_bytes=2097152 predicted_peak_delta_bytes=437897152 observed_peak_delta_bytes=352874496 "
        "prediction_error_ratio=0.806 underpredicted=False"
    )
    sample = extract_phase_a_sample(line)
    assert sample is not None
    # (437897152 - 200000000 - 2097152) / 300 = 786000
    assert sample.accumulator_entries_peak == 786_000


def test_extract_rust_batch_sample_and_effective_overhead():
    line = (
        "Telemetry: pair_featurization_memory stage=pair_featurization_rust_batch "
        "prediction_contract_version=delta_v1 predicted_peak_delta_bytes=4000 predicted_peak_rss_bytes=9000 "
        "predicted_bytes=4000 total_rows=100 selected_feature_count=6 nameless_feature_count=0 "
        "predicted_features_matrix_bytes=1000 predicted_labels_bytes=800 predicted_chunk_bytes=600 "
        "predicted_persistent_row_overhead_bytes=700 predicted_fixed_overhead_bytes=300 "
        "rss_before_bytes=5000 rss_peak_bytes=9300 rss_after_bytes=7000 observed_peak_delta_bytes=4300 "
        "prediction_error_ratio=1.075 underpredicted=True rss_source=psutil_process_rss"
    )
    sample = extract_rust_batch_sample(line)
    assert sample is not None
    effective = effective_rust_batch_persistent_row_overhead_bytes(sample)
    assert effective is not None
    # residual = 4300 - (1000 + 800 + 600 + 300) = 1600; rows=100 => 16 bytes/row
    assert effective == 16.0


def test_extract_rust_batch_sample_ignores_non_rust_stage():
    line = (
        "Telemetry: pair_featurization_memory stage=pair_featurization_python_serial "
        "predicted_peak_delta_bytes=1000 observed_peak_delta_bytes=900"
    )
    assert extract_rust_batch_sample(line) is None
