from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from s2and.consts import PROJECT_ROOT_PATH


def _load_rust_suite_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_memory_telemetry_writes_json(tmp_path):
    module = _load_rust_suite_module()
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-02-27 00:00:00,000 - s2and - INFO - Telemetry: phase_split_phase_a "
                "prediction_error_ratio=1.250 underpredicted=True",
                "2026-02-27 00:00:01,000 - s2and - INFO - Telemetry: pair_featurization_memory "
                "stage=pair_featurization_rust_batch prediction_error_ratio=0.900 underpredicted=False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    rc = module.main(["summarize-memory-telemetry", str(log_path), "--write-json", str(output_path)])
    assert int(rc) == 0
    blob = json.loads(output_path.read_text(encoding="utf-8"))
    assert int(blob["schema_version"]) == 1

    phase_a = blob["stages"]["phase_a_seed_distances"]
    assert int(phase_a["matched_records"]) == 1
    assert int(phase_a["samples"]) == 1
    assert int(phase_a["underpredicted_count"]) == 1
    assert float(phase_a["underpredicted_fraction"]) == 1.0
    assert float(phase_a["ratio_summary"]["p95"]) == 1.25

    rust_batch = blob["stages"]["pair_featurization_rust_batch"]
    assert int(rust_batch["matched_records"]) == 1
    assert int(rust_batch["samples"]) == 1
    assert int(rust_batch["underpredicted_count"]) == 0
    assert float(rust_batch["underpredicted_fraction"]) == 0.0
    assert float(rust_batch["ratio_summary"]["p95"]) == 0.9


def test_summarize_memory_telemetry_returns_nonzero_when_no_samples(tmp_path):
    module = _load_rust_suite_module()
    log_path = tmp_path / "empty.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-02-27 00:00:00,000 - s2and - INFO - not a telemetry line",
                "some continuation line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.json"

    rc = module.main(["summarize-memory-telemetry", str(log_path), "--write-json", str(output_path)])
    assert int(rc) != 0
    blob = json.loads(output_path.read_text(encoding="utf-8"))
    assert int(blob["stages"]["phase_a_seed_distances"]["samples"]) == 0
    assert int(blob["stages"]["pair_featurization_rust_batch"]["samples"]) == 0
