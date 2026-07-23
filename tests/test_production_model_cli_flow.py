from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and import production_model as production_model_module
from s2and.incremental_linking.artifact import ARTIFACT_SCHEMA_VERSION
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking_training.classic import load_bundle
from s2and.production_model import _load_pairwise_staging_model, load_production_model
from tests.helpers import import_s2and_rust, tiny_name_counts_provenance, tiny_name_counts_tuple

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust()
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"s2and_rust unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)
_SYNTHETIC_ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "1" * 64,
    "orcid_prefix_counts_data_sha256": "2" * 64,
}


def _run_cli(
    args: list[str],
    *,
    repo_root: Path,
    timeout: int = 300,
    env_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bootstrap = (
        "import json, runpy, sys; "
        "from s2and import production_model; "
        "artifact_hashes = json.loads(sys.argv.pop(1)); "
        "production_model.canonical_artifact_hashes = lambda: dict(artifact_hashes); "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, json.dumps(_SYNTHETIC_ARTIFACT_HASHES), *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env={**os.environ, "S2AND_BACKEND": "python", **dict(env_overrides or {})},
    )
    assert completed.returncode == 0, (
        f"Command failed: {[sys.executable, *args]}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
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


def _write_tiny_promoted_feature_bundle(
    feature_root: Path,
    target_path: Path,
    pairwise_bundle_dir: Path,
) -> None:
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
        "metrics": {
            "stratified_test_queries": 2,
            "stratified_test_accuracy": 0.5,
            "stratified_test_errors": 1,
            "stratified_test_false_abstain": 1,
            "stratified_test_false_link": 0,
            "stratified_test_wrong_candidate_link": 0,
        },
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
        table_path = feature_root / rel_path
        frames[key].to_parquet(table_path, index=False)

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

    pairwise_binding = production_model_module.pairwise_bundle_binding(pairwise_bundle_dir)
    for table_key, rel_path in paths.items():
        input_datasets = sorted(set(frames[table_key]["dataset"].astype(str)))
        identity = {
            "schema_version": promoted_train.MATERIALIZATION_IDENTITY_SCHEMA_VERSION,
            "source_bundle": {
                "bundle_json_sha256": "4" * 64,
                "labels_path": f"labels/{table_key}.parquet",
                "labels_sha256": "5" * 64,
            },
            "pairwise_bundle_binding": dict(pairwise_binding),
            "target_spec_digest": promoted_train._target_spec_digest(target),
            "feature_schema_digest": promoted_train.promoted_linker_feature_schema_digest(feature_columns),
            "feature_columns": feature_columns,
            "feature_policies": {
                "pairwise_model_nan_value": "nan",
                "pairwise_aggregate_nan_value": 0.0,
                "row_nan_policy": "finite",
                "max_exemplars": 4,
            },
            "selection": {
                "table_key": table_key,
                "datasets": None,
                "limit_rows": None,
                "selected_row_count": len(frames[table_key]),
                "selected_rows_digest": "9" * 64,
                "input_datasets": input_datasets,
            },
            "datasets": {
                dataset_name: {
                    "arrow": {
                        "generation_id": "6" * 64,
                        "normalization_version": "canonical_v2",
                        "name_counts_manifest_sha256": "7" * 64,
                    },
                    "candidate_members_path": f"components/{dataset_name}.parquet",
                    "candidate_members_sha256": "8" * 64,
                }
                for dataset_name in input_datasets
            },
        }
        reuse_metadata = promoted_train._materialization_reuse_metadata(
            identity,
            artifact={"kind": "complete_table", "table_key": table_key, "rows": len(frames[table_key])},
        )
        promoted_train._write_materialization_sidecar(feature_root / rel_path, reuse_metadata)

    bundle_payload["precomputed_promoted_feature_bundle"] = promoted_train._precomputed_promoted_bundle_metadata(
        bundle=load_bundle(feature_root),
        target=target,
        source_mode="tiny-flow-pytest",
        pairwise_model_binding=pairwise_binding,
    )
    (feature_root / "bundle.json").write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@requires_rust_lightgbm
def test_tiny_qian_production_model_two_step_cli_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: dict(_SYNTHETIC_ARTIFACT_HASHES),
    )
    repo_root = Path(__file__).resolve().parents[1]
    pairwise_bundle_dir = tmp_path / "pairwise_stage" / "production_model_v9.8"
    bundle_dir = tmp_path / "final" / "production_model_v9.8"
    canonical_data_dir = tmp_path / "canonical_data"
    name_counts_index_path, _metrics = write_name_counts_index(
        canonical_data_dir,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    path_config = tmp_path / "path_config.json"
    path_config.write_text(
        json.dumps({"main_data_dir": str(canonical_data_dir.resolve())}) + "\n",
        encoding="utf-8",
    )
    cli_env = {"S2AND_PATH_CONFIG": str(path_config)}

    _run_cli(
        [
            "scripts/production/model/train_pairwise.py",
            "--production-version",
            "9.8",
            "--output-dir",
            str(pairwise_bundle_dir),
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
        env_overrides=cli_env,
    )

    pairwise_manifest = json.loads((pairwise_bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert pairwise_manifest["incremental_linker_version"] is None
    assert "bundle_status" not in pairwise_manifest
    assert "files" not in pairwise_manifest
    # Pairwise lineage is manifest authority and need not match the staging
    # directory's release name (for example, a canonical v1.21 rewrap).
    pairwise_manifest["pairwise_model_version"] = "1.2"
    (pairwise_bundle_dir / "manifest.json").write_text(
        json.dumps(pairwise_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pairwise_config = json.loads((pairwise_bundle_dir / "clusterer.json").read_text(encoding="utf-8"))
    for field in ("name_tuples_data_sha256", "orcid_prefix_counts_data_sha256"):
        assert len(pairwise_config["feature_contract"][field]) == 64
    expected_name_count_contract = {
        "name_counts_manifest_sha256": hashlib.sha256(
            (Path(name_counts_index_path) / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    for field, expected in expected_name_count_contract.items():
        assert pairwise_config["feature_contract"][field] == expected
    with pytest.raises(ValueError, match="Expected a complete"):
        load_production_model(pairwise_bundle_dir)
    assert _load_pairwise_staging_model(pairwise_bundle_dir).production_model_bundle_status == "pairwise_only"

    feature_root = tmp_path / "tiny_linker_feature_bundle"
    target_path = tmp_path / "incremental_linker_training_target.json"
    _write_tiny_promoted_feature_bundle(feature_root, target_path, pairwise_bundle_dir)

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
            str(pairwise_bundle_dir),
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
        ],
        repo_root=repo_root,
        env_overrides=cli_env,
    )

    final_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["incremental_linker_version"] == "9.8"
    assert "bundle_status" not in final_manifest
    assert "files" not in final_manifest
    assert final_manifest["bundle_version"] == "9.8"
    assert final_manifest["pairwise_model_version"] == "1.2"
    assert _load_pairwise_staging_model(pairwise_bundle_dir).production_model_bundle_status == "pairwise_only"
    assert not (pairwise_bundle_dir / "incremental_linker").exists()
    assert (tmp_path / "linker_run" / "production_incremental_linker" / "metadata.json").is_file()

    clusterer = load_production_model(bundle_dir)
    assert clusterer.production_model_bundle_status == "complete"
    assert clusterer.incremental_linker_artifact_dir is not None
    assert Path(clusterer.incremental_linker_artifact_dir) == bundle_dir / "incremental_linker"
    artifact_metadata = json.loads((bundle_dir / "incremental_linker" / "metadata.json").read_text(encoding="utf-8"))
    assert set(artifact_metadata) == {
        "booster_sha256",
        "gate_config",
        "pairwise_bundle_binding_digest",
        "retrieval_top_k",
        "schema_version",
        "target_spec_digest",
    }
    assert artifact_metadata["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert artifact_metadata["gate_config"]["model_type"] == "multiclass_logistic_numpy_v1"
    assert len(artifact_metadata["gate_config"]["feature_names"]) == 240
    assert len(artifact_metadata["gate_config"]["weights"]) == 240
    assert len(artifact_metadata["gate_config"]["bias"]) == 3
