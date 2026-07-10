from __future__ import annotations

import json

from scripts._rust_suite import summarize_memory_telemetry_cmd


def test_summarize_memory_telemetry_writes_json(tmp_path):
    log_path = tmp_path / "memory_telemetry.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 1,
                        "event": "memory_telemetry",
                        "stage": "phase_a_seed_distances",
                        "prediction_error_ratio": 1.25,
                        "underpredicted": True,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "event": "memory_telemetry",
                        "stage": "pair_featurization_rust_batch",
                        "prediction_error_ratio": 0.9,
                        "underpredicted": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    rc = summarize_memory_telemetry_cmd.main([str(log_path), "--write-json", str(output_path)])
    assert int(rc) == 0
    blob = json.loads(output_path.read_text(encoding="utf-8"))
    assert int(blob["schema_version"]) == 1

    phase_a = blob["stages"]["phase_a_seed_distances"]
    assert phase_a["stage"] == "phase_a_seed_distances"
    assert int(phase_a["matched_records"]) == 1
    assert int(phase_a["samples"]) == 1
    assert int(phase_a["underpredicted_count"]) == 1
    assert float(phase_a["underpredicted_fraction"]) == 1.0
    assert float(phase_a["ratio_summary"]["p95"]) == 1.25

    rust_batch = blob["stages"]["pair_featurization_rust_batch"]
    assert rust_batch["stage"] == "pair_featurization_rust_batch"
    assert int(rust_batch["matched_records"]) == 1
    assert int(rust_batch["samples"]) == 1
    assert int(rust_batch["underpredicted_count"]) == 0
    assert float(rust_batch["underpredicted_fraction"]) == 0.0
    assert float(rust_batch["ratio_summary"]["p95"]) == 0.9


def test_summarize_memory_telemetry_returns_nonzero_when_no_samples(tmp_path):
    log_path = tmp_path / "empty.jsonl"
    log_path.write_text(
        json.dumps({"schema_version": 1, "event": "memory_telemetry", "stage": "unrelated"}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    rc = summarize_memory_telemetry_cmd.main([str(log_path), "--write-json", str(output_path)])
    assert int(rc) != 0
    blob = json.loads(output_path.read_text(encoding="utf-8"))
    assert int(blob["stages"]["phase_a_seed_distances"]["samples"]) == 0
    assert int(blob["stages"]["pair_featurization_rust_batch"]["samples"]) == 0
