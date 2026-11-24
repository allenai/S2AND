"""Smoke-test the giant-block threshold sweep on a tiny Arrow block."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.release.verification import analyze_giant_block_subblocking as analysis
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


def _sweep_args(tmp_path: Path, threshold: str, components: Path) -> list[str]:
    return [
        "--arrow-dataset",
        str(tmp_path / "arrow"),
        "--block",
        "a lovelace",
        "--thresholds",
        threshold,
        "--output-dir",
        str(tmp_path / f"out{threshold}"),
        "--n-jobs",
        "1",
        "--component-members-parquet",
        str(components),
    ]


def test_threshold_sweep_rejects_split_reviewed_component(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path / "arrow", include_specter=True)
    components = tmp_path / "components.parquet"
    signature_ids = [str(index) for index in range(10)] + ["q1", "q2", "seed1", "outside"]
    pd.DataFrame({"signature_id": signature_ids, "candidate_component_key": ["c"] * len(signature_ids)}).to_parquet(
        components
    )

    analysis.main(_sweep_args(tmp_path, "20", components))
    summary = json.loads((tmp_path / "out20" / "summary.json").read_text(encoding="utf-8"))
    assert summary["metadata"]["reviewed_component_signature_count"] == 13

    # A cap of 4 must split the 13-signature component, and the sweep must write no report.
    with pytest.raises(ValueError, match="split 1 reviewed components"):
        analysis.main(_sweep_args(tmp_path, "4", components))
    assert not (tmp_path / "out4" / "summary.json").exists()


def test_split_component_keys_ignores_whole_components() -> None:
    labels = {"a": "kept", "b": "kept", "c": "split", "d": "split"}
    placement = {"a": "s1", "b": "s1", "c": "s1", "d": "s2"}
    assert analysis.split_component_keys(labels, placement) == ["split"]
