"""Gate configuration validation at the real source-bundle preflight boundary."""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and.incremental_linking_training.classic import OfficialBundle, load_bundle
from s2and.incremental_linking_training.source_bundle_preflight import validate_source_bundle_support_files


def _source_bundle(root: Path, gate_changes: dict[str, Any]) -> OfficialBundle:
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
            **gate_changes,
        },
    }
    (root / "bundle.json").write_text(
        json.dumps(
            {"bundle_name": "gate-preflight", "assets": {}, "models": {"classic": classic}, "expected_metrics": {}}
        ),
        encoding="utf-8",
    )
    return load_bundle(root)


def test_source_support_preflight_accepts_valid_gate_and_complete_assignments(tmp_path: Path) -> None:
    bundle = _source_bundle(tmp_path, {})

    files = validate_source_bundle_support_files(bundle)

    assert set(files) == {
        str(tmp_path / "bundle.json"),
        str(tmp_path / "splits" / "internal_eval.csv"),
        str(tmp_path / "splits" / "assignments.csv"),
    }


@pytest.mark.parametrize(
    ("gate_changes", "message"),
    [
        (
            {"calibration_splits": ["calibration_fit", "test"]},
            "classic.promoted_stratified_gate calibration_splits must not include test_split",
        ),
        (
            {"calibration_splits": ["calibration_fit", "calibration_fit"]},
            "classic.promoted_stratified_gate.calibration_splits must not contain duplicates",
        ),
        (
            {"mode": "legacy_threshold_gate"},
            "classic.promoted_stratified_gate.mode must be 'promoted_logistic_topk_multiclass_l2'",
        ),
    ],
    ids=["test-in-calibration", "duplicate-calibration", "unsupported-mode"],
)
def test_source_support_preflight_rejects_present_invalid_gate(
    tmp_path: Path, gate_changes: dict[str, Any], message: str
) -> None:
    bundle = _source_bundle(tmp_path, gate_changes)

    # All support files and referenced split names exist, and base/query IDs
    # are disjoint. Only validation of the present gate config can reject it.
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        validate_source_bundle_support_files(bundle)
