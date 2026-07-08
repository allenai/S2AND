from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking_training.classic import load_bundle
from s2and.production_model import load_production_model
from tests.helpers import import_s2and_rust

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust(
    required_module_attrs=("RustLightGBMBooster",),
    prefer_site_packages=True,
)
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"RustLightGBMBooster unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)


def _run_cli(args: list[str], *, repo_root: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        f"Command failed: {[sys.executable, *args]}\n" f"stdout:\n{completed.stdout}\n" f"stderr:\n{completed.stderr}"
    )
    return completed


def _candidate_rows(
    *,
    prefix: str,
    base_prefix: str,
    dataset: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    base_columns = [
        "query_group_id",
        "dataset",
        "query_view",
        "query_first_token",
        "candidate_component_key",
        "base_group_id",
        "retrieval_rank",
        "label",
    ]
    output_columns = base_columns + [column for column in feature_columns if column not in base_columns]
    source_rows: list[dict[str, Any]] = [
        {
            "query_group_id": f"{prefix}_pos",
            "dataset": dataset,
            "query_view": "full",
            "query_first_token": "anna",
            "candidate_component_key": f"{prefix}_pos_correct",
            "base_group_id": f"{base_prefix}_pos",
            "retrieval_rank": 1,
            "label": 1,
            "min_distance": 0.05,
        },
        {
            "query_group_id": f"{prefix}_pos",
            "dataset": dataset,
            "query_view": "full",
            "query_first_token": "anna",
            "candidate_component_key": f"{prefix}_pos_wrong",
            "base_group_id": f"{base_prefix}_pos",
            "retrieval_rank": 2,
            "label": 0,
            "min_distance": 0.95,
        },
        {
            "query_group_id": f"{prefix}_neg",
            "dataset": dataset,
            "query_view": "full",
            "query_first_token": "bo",
            "candidate_component_key": f"{prefix}_neg_a",
            "base_group_id": f"{base_prefix}_neg",
            "retrieval_rank": 1,
            "label": 0,
            "min_distance": 0.85,
        },
        {
            "query_group_id": f"{prefix}_neg",
            "dataset": dataset,
            "query_view": "full",
            "query_first_token": "bo",
            "candidate_component_key": f"{prefix}_neg_b",
            "base_group_id": f"{base_prefix}_neg",
            "retrieval_rank": 2,
            "label": 0,
            "min_distance": 0.75,
        },
    ]
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        row = {column: source_row[column] for column in base_columns}
        for feature in feature_columns:
            if feature in base_columns:
                continue
            row[feature] = float(source_row.get(feature, 0.0))
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def _write_tiny_promoted_feature_bundle(feature_root: Path, target_path: Path) -> None:
    from scripts.production.model import linker_train_calibrate_eval as promoted_train

    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "features_corrected").mkdir(parents=True, exist_ok=True)
    (feature_root / "splits").mkdir(parents=True, exist_ok=True)

    feature_columns = list(promoted_linker_feature_columns())
    target = {
        "variant": "tiny_production_flow_smoke",
        "status": "test_fixture",
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "params": {
            "n_estimators": 3,
            "learning_rate": 0.2,
            "num_leaves": 2,
            "min_child_samples": 1,
            "min_data_in_leaf": 1,
            "force_col_wise": True,
        },
        "metrics": {},
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(target, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    paths = {
        "train_path": "features_corrected/train.parquet",
        "classic_gate_source_path": "features_corrected/classic_gate_source.parquet",
        "s2and_eval_path": "features_corrected/s2and_eval.parquet",
        "hwang_eval_path": "features_corrected/hwang_eval.parquet",
    }
    frames = {
        "train_path": _candidate_rows(
            prefix="train", base_prefix="train_base", dataset="qian", feature_columns=feature_columns
        ),
        "classic_gate_source_path": _candidate_rows(
            prefix="cal", base_prefix="cal_base", dataset="qian", feature_columns=feature_columns
        ),
        "s2and_eval_path": _candidate_rows(
            prefix="s2and", base_prefix="s2and_base", dataset="qian", feature_columns=feature_columns
        ),
        "hwang_eval_path": _candidate_rows(
            prefix="hwang", base_prefix="hwang_base", dataset="hwang", feature_columns=feature_columns
        ),
    }
    for key, rel_path in paths.items():
        frames[key].to_parquet(feature_root / rel_path, index=False)

    pd.DataFrame({"base_group_id": ["cal_base_neg"]}).to_csv(
        feature_root / "splits" / "classic_gate_internal_eval_base_groups.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "query_group_id": "s2and_pos",
                "source_key": "s2and_eval",
                "split": "test",
                "source_stratum": "s2and",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "s2and_neg",
                "source_key": "s2and_eval",
                "split": "test",
                "source_stratum": "s2and",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "hwang_pos",
                "source_key": "hwang_eval",
                "split": "calibration_fit",
                "source_stratum": "hwang",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "hwang_neg",
                "source_key": "hwang_eval",
                "split": "calibration_check",
                "source_stratum": "hwang",
                "first_name_bucket": "multi_letter_first",
            },
        ]
    ).to_csv(feature_root / "splits" / "assignments.csv", index=False)

    bundle_payload = {
        "bundle_name": "tiny_linker_feature_bundle",
        "assets": {
            "corrected_feature_rows": {"files": paths},
        },
        "models": {
            "classic": {
                **paths,
                "classic_gate_internal_eval_base_groups_path": "splits/classic_gate_internal_eval_base_groups.csv",
                "classic_gate_calibration_retrieval_limit": 25,
                "stratified_eval_test_split": {
                    "assignments_path": "splits/assignments.csv",
                    "split_order": ["calibration_fit", "calibration_check", "test"],
                    "test_split": "test",
                },
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit", "calibration_check"],
                    "test_split": "test",
                },
                "feature_columns": feature_columns,
                "best_params": dict(target["params"]),
            },
        },
        "expected_metrics": {"classic": {}},
    }
    (feature_root / "bundle.json").write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle_payload["precomputed_promoted_feature_bundle"] = promoted_train._precomputed_promoted_bundle_metadata(
        bundle=load_bundle(feature_root),
        target=target,
        source_mode="tiny-flow-pytest",
    )
    (feature_root / "bundle.json").write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@requires_rust_lightgbm
def test_tiny_qian_production_model_two_step_cli_flow(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_dir = tmp_path / "production_model_v9.8"

    _run_cli(
        [
            "scripts/production/model/train_pairwise.py",
            "--production-version",
            "9.8",
            "--output-dir",
            str(bundle_dir),
            "--data-dir",
            "tests",
            "--datasets",
            "qian",
            "--no-include-augmented",
            "--n-iter",
            "1",
            "--cluster-n-iter",
            "1",
            "--n-jobs",
            "1",
            "--chunk-size",
            "100",
            "--train-pairs-size",
            "50",
            "--val-test-size",
            "20",
            "--run-full",
        ],
        repo_root=repo_root,
    )

    pairwise_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert pairwise_manifest["bundle_status"] == "pairwise_only"
    with pytest.raises(FileNotFoundError, match="pairwise-only"):
        load_production_model(bundle_dir)
    assert load_production_model(bundle_dir, require_incremental_linker=False).production_model_bundle_status == (
        "pairwise_only"
    )

    feature_root = tmp_path / "tiny_linker_feature_bundle"
    target_path = bundle_dir / "reproducibility" / "incremental_linker_training_target.json"
    _write_tiny_promoted_feature_bundle(feature_root, target_path)

    _run_cli(
        [
            "scripts/production/model/train_linker_and_finalize.py",
            "--feature-mode",
            "precomputed-promoted",
            "--precomputed-feature-bundle-root",
            str(feature_root),
            "--target-json",
            str(target_path),
            "--pairwise-model-path",
            str(bundle_dir),
            "--save-production-bundle-to",
            str(bundle_dir),
            "--production-bundle-version",
            "9.8",
            "--linker-artifact-version",
            "v9.8",
            "--output-dir",
            str(tmp_path / "linker_run"),
            "--prod-holdout-importance-weight",
            "2.0",
            "--run-full",
            "--allow-metric-drift",
        ],
        repo_root=repo_root,
    )

    final_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["bundle_status"] == "complete"
    assert final_manifest["bundle_version"] == "9.8"

    clusterer = load_production_model(bundle_dir)
    assert clusterer.production_model_bundle_status == "complete"
    assert clusterer.incremental_linker_artifact_dir is not None
    assert Path(clusterer.incremental_linker_artifact_dir) == bundle_dir / "incremental_linker"
    artifact_metadata = json.loads((bundle_dir / "incremental_linker" / "metadata.json").read_text(encoding="utf-8"))
    assert artifact_metadata["gate_surface"] == "promoted_numpy_logistic_gate"
    assert artifact_metadata["gate_config"]["model_type"] == "multiclass_logistic_numpy_v1"
    assert len(artifact_metadata["gate_config"]["feature_names"]) == 240
    assert len(artifact_metadata["gate_config"]["weights"]) == 240
    assert len(artifact_metadata["gate_config"]["bias"]) == 3
