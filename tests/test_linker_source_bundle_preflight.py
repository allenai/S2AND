"""Gate configuration validation at the real source-bundle preflight boundary."""

import json
from pathlib import Path

import pandas as pd
import pytest

from s2and.incremental_linking_training.classic import OfficialBundle, load_bundle
from s2and.incremental_linking_training.source_bundle_preflight import validate_source_bundle_support_files


def _source_bundle(root: Path) -> OfficialBundle:
    """Write otherwise-valid support files, including every configured split."""
    splits_dir = root / "splits"
    splits_dir.mkdir()
    pd.DataFrame({"base_group_id": ["internal"]}).to_csv(splits_dir / "internal_eval.csv", index=False)
    pd.DataFrame(
        {
            "query_group_id": ["q_fit", "q_check", "q_test"],
            "source_key": ["source"] * 3,
            "base_group_id": ["base_fit", "base_check", "base_test"],
            "split": ["calibration_fit", "calibration_check", "test"],
        }
    ).to_csv(splits_dir / "assignments.csv", index=False)
    classic = {
        "classic_gate_internal_eval_base_groups_path": "splits/internal_eval.csv",
        "stratified_eval_test_split": {"assignments_path": "splits/assignments.csv", "test_split": "test"},
        "promoted_stratified_gate": {
            "mode": "promoted_logistic_topk_multiclass_l2",
            "calibration_splits": ["calibration_fit", "calibration_check"],
            "test_split": "test",
        },
    }
    (root / "bundle.json").write_text(
        json.dumps(
            {"bundle_name": "gate-preflight", "assets": {}, "models": {"classic": classic}, "expected_metrics": {}}
        ),
        encoding="utf-8",
    )
    return load_bundle(root)


def test_source_support_preflight_requires_complete_gate_and_assignments(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path)

    files = validate_source_bundle_support_files(bundle)

    assert set(files) == {
        str(tmp_path / "bundle.json"),
        str(tmp_path / "splits" / "internal_eval.csv"),
        str(tmp_path / "splits" / "assignments.csv"),
    }

    splits = tmp_path / "splits"
    assignments = splits / "assignments.csv"
    complete = assignments.read_bytes()
    rows = pd.read_csv(assignments)
    rows.loc[rows["split"] != "calibration_check"].to_csv(assignments, index=False)
    with pytest.raises(ValueError, match="omit configured calibration/test splits"):
        validate_source_bundle_support_files(bundle)
    assignments.write_bytes(complete)

    classic = bundle.models["classic"]
    gate = classic.pop("promoted_stratified_gate")
    with pytest.raises(ValueError, match="promoted_stratified_gate is required"):
        validate_source_bundle_support_files(bundle)
    classic["promoted_stratified_gate"] = {**gate, "mode": "legacy_threshold_gate"}
    with pytest.raises(ValueError, match="promoted_stratified_gate.mode"):
        validate_source_bundle_support_files(bundle)

    classic.pop("classic_gate_internal_eval_base_groups_path")
    with pytest.raises(ValueError, match="classic_gate_internal_eval_base_groups_path"):
        validate_source_bundle_support_files(bundle)

    assignments.unlink()
    (splits / "internal_eval.csv").unlink()
    with pytest.raises(ValueError, match="contains no files"):
        validate_source_bundle_support_files(bundle)
