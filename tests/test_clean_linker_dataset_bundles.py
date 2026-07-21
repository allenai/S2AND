from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.production.model import clean_linker_dataset_bundles as clean_module
from scripts.production.model.clean_linker_dataset_bundles import _bundle_child_path, migrate_bundle


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _label_row(
    *,
    query_group_id: str,
    base_group_id: str,
    candidate_component_key: str,
    label: int,
    supervision_type: str,
    dataset: str = "h_wang",
    retrieval_rank: int = 1,
) -> dict[str, object]:
    return {
        "row_index": 0,
        "dataset": dataset,
        "query_group_id": query_group_id,
        "base_group_id": base_group_id,
        "query_signature_id": f"{query_group_id}:sig",
        "query_author": "author",
        "query_first_token": "a",
        "query_view": "full",
        "candidate_component_key": candidate_component_key,
        "candidate_cluster_id": candidate_component_key,
        "label": label,
        "retrieval_rank": retrieval_rank,
        "split": "test",
        "supervision_type": supervision_type,
        "source_key": "",
        "source_kind": "",
        "train_group_id": query_group_id,
    }


def test_bundle_child_path_rejects_absolute_paths(tmp_path: Path) -> None:
    absolute_path = tmp_path / "outside.parquet"

    with pytest.raises(ValueError, match="must be relative"):
        _bundle_child_path(tmp_path / "bundle", str(absolute_path), context="test path")


def test_bundle_child_path_rejects_parent_escapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes bundle root"):
        _bundle_child_path(tmp_path / "bundle", "../outside.parquet", context="test path")


def test_migrate_bundle_drops_singleton_orcid_and_refreshes_split_metadata(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    (bundle_root / "splits").mkdir(parents=True)

    _write_parquet(
        bundle_root / "labels" / "train.parquet",
        [
            _label_row(
                query_group_id="train_keep",
                base_group_id="b_train",
                candidate_component_key="c_train",
                label=1,
                supervision_type="manual",
            ),
            _label_row(
                query_group_id="train_drop",
                base_group_id="b_train_drop",
                candidate_component_key="c_train_drop",
                label=0,
                supervision_type="unlabeled_singleton_orcid",
            ),
        ],
    )
    _write_parquet(
        bundle_root / "labels" / "calibration_source.parquet",
        [
            _label_row(
                query_group_id="gate_keep",
                base_group_id="b_gate_keep",
                candidate_component_key="c_gate_keep",
                label=1,
                supervision_type="manual",
            ),
            _label_row(
                query_group_id="gate_drop",
                base_group_id="b_gate_drop",
                candidate_component_key="c_gate_drop",
                label=0,
                supervision_type="unlabeled_singleton_orcid",
            ),
        ],
    )
    _write_parquet(
        bundle_root / "labels" / "hwang_eval.parquet",
        [
            _label_row(
                query_group_id="hwang_keep",
                base_group_id="b_hwang_keep",
                candidate_component_key="c_hwang_keep",
                label=1,
                supervision_type="manual",
            ),
            _label_row(
                query_group_id="hwang_drop",
                base_group_id="b_hwang_drop",
                candidate_component_key="c_hwang_drop",
                label=0,
                supervision_type="unlabeled_singleton_orcid",
            ),
        ],
    )
    _write_parquet(
        bundle_root / "labels" / "s2and_eval.parquet",
        [
            _label_row(
                query_group_id="s2and_keep",
                base_group_id="b_s2and_keep",
                candidate_component_key="c_s2and_keep",
                label=1,
                supervision_type="manual",
                dataset="s2and",
            )
        ],
    )

    assignments = pd.DataFrame(
        [
            {"query_group_id": "hwang_keep", "source_key": "hwang_eval", "split": "test", "source_stratum": "x"},
            {"query_group_id": "hwang_drop", "source_key": "hwang_eval", "split": "test", "source_stratum": "x"},
            {
                "query_group_id": "s2and_keep",
                "source_key": "s2and_eval",
                "split": "calibration_fit",
                "source_stratum": "x",
            },
        ]
    )
    assignments.to_csv(bundle_root / "splits" / "combined_query_split_assignments.csv", index=False)
    pd.DataFrame({"base_group_id": ["b_gate_keep", "b_gate_drop"]}).to_csv(
        bundle_root / "splits" / "classic_gate_internal_eval_base_groups.csv",
        index=False,
    )
    (bundle_root / "splits" / "summary.json").write_text(
        json.dumps({"assignment_rows": 3, "split_counts": {"test": 2, "calibration_fit": 1}}),
        encoding="utf-8",
    )

    bundle_payload = {
        "bundle_name": "test_bundle",
        "expected_metrics": {},
        "assets": {
            "featureless_rows": {
                "files": {
                    "train_path": "labels/train.parquet",
                    "classic_gate_source_path": "labels/calibration_source.parquet",
                    "hwang_eval_path": "labels/hwang_eval.parquet",
                    "s2and_eval_path": "labels/s2and_eval.parquet",
                }
            },
            "splits": {
                "assignments_path": "splits/combined_query_split_assignments.csv",
                "summary_path": "splits/summary.json",
            },
        },
        "models": {
            "classic": {
                "train_path": "features_corrected/train.parquet",
                "classic_gate_source_path": "features_corrected/calibration_source.parquet",
                "hwang_eval_path": "features_corrected/hwang_eval.parquet",
                "s2and_eval_path": "features_corrected/s2and_eval.parquet",
                "classic_gate_internal_eval_base_groups_path": "splits/classic_gate_internal_eval_base_groups.csv",
                "stratified_eval_test_split": {"assignments_path": "splits/combined_query_split_assignments.csv"},
            }
        },
    }
    (bundle_root / "bundle.json").write_text(json.dumps(bundle_payload), encoding="utf-8")

    report = migrate_bundle(bundle_root, write=True)

    assert report["verification"]["ok"] is True
    assert pd.read_parquet(bundle_root / "labels" / "train.parquet")["query_group_id"].tolist() == ["train_keep"]
    assert pd.read_parquet(bundle_root / "labels" / "hwang_eval.parquet")["query_group_id"].tolist() == ["hwang_keep"]
    refreshed_assignments = pd.read_csv(bundle_root / "splits" / "combined_query_split_assignments.csv")
    assert set(refreshed_assignments["query_group_id"]) == {"hwang_keep", "s2and_keep"}
    assert refreshed_assignments["split"].value_counts().to_dict() == {"test": 1, "calibration_fit": 1}
    gate_groups = pd.read_csv(bundle_root / "splits" / "classic_gate_internal_eval_base_groups.csv")
    assert gate_groups["base_group_id"].tolist() == ["b_gate_keep"]


def test_main_dry_run_does_not_write_report_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_path = tmp_path / "report.json"
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_linker_dataset_bundles.py",
            "--dry-run",
            "--report-path",
            str(report_path),
            str(bundle_root),
        ],
    )
    monkeypatch.setattr(
        clean_module,
        "migrate_bundle",
        lambda root, write: {"bundle_root": str(root), "mode": "dry-run" if not write else "write"},
    )

    clean_module.main()

    assert not report_path.exists()
