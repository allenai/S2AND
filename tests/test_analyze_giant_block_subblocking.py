"""Smoke-test the giant-block threshold sweep on a tiny Arrow block."""

import json
from pathlib import Path

from scripts import analyze_giant_block_subblocking as analysis
from tests.helpers import write_minimal_arrow_prediction_bundle


def test_threshold_sweep_uses_native_subblocking_on_one_arrow_block(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path / "arrow", include_specter=True)
    output_dir = tmp_path / "out"

    analysis.main(
        [
            "--arrow-dataset",
            str(tmp_path / "arrow"),
            "--block",
            "a lovelace",
            "--thresholds",
            "20",
            "4",
            "--output-dir",
            str(output_dir),
            "--n-jobs",
            "1",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    rows = {row["threshold"]: row for row in summary["rows"]}
    assert summary["metadata"]["signature_count"] == 13
    assert rows[20]["subblock_count"] == 1
    # One shared first name cannot split the block, so the native graph fallback must cap it.
    assert rows[4]["max_subblock_size"] <= 4
    assert rows[4]["specter_invocation_count"] >= 1
    assert (output_dir / "threshold_overview.png").is_file()
