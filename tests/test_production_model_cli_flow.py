from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and import production_model as production_model_module
from s2and.incremental_linking.artifact import ARTIFACT_SCHEMA_VERSION
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking_training.classic import load_bundle
from s2and.production_model import _load_pairwise_staging_model, load_production_model
from scripts.production.model import train_linker_and_finalize as promoted_train
from tests.helpers import import_s2and_rust
from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust()
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"s2and_rust unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)
_SYNTHETIC_ARTIFACT_HASHES = {
    "name_tuples_data_sha256": "1" * 64,
    "orcid_prefix_counts_data_sha256": "2" * 64,
}


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
) -> None:
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
            "false_abstain_error_rate": 0.5,
            "false_link_error_rate": 0.0,
            "stratified_test_accuracy": 0.5,
            "stratified_test_balanced_accuracy": 0.5,
            "stratified_test_error_rate": 0.5,
            "stratified_test_errors": 1,
            "stratified_test_false_abstain": 1,
            "stratified_test_false_link": 0,
            "stratified_test_queries": 2,
            "stratified_test_wrong_candidate_link": 0,
            "training_positive_rows": 1,
            "training_rows": 4,
            "weighted_average_error": 0.045454545454545456,
            "weighted_average_error_weights": {
                "false_abstain_error_rate": 0.25,
                "false_link_error_rate": 1.0,
                "wrong_link_error_rate": 1.5,
            },
            "wrong_link_error_rate": 0.0,
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
                "base_group_id": "s2and_base_pos",
                "source_stratum": "s2and",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "s2and_neg",
                "source_key": "s2and_eval",
                "split": "test",
                "base_group_id": "s2and_base_neg",
                "source_stratum": "s2and",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "hwang_pos",
                "source_key": "hwang_eval",
                "split": "calibration_fit",
                "base_group_id": "hwang_base_pos",
                "source_stratum": "hwang",
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_group_id": "hwang_neg",
                "source_key": "hwang_eval",
                "split": "calibration_check",
                "base_group_id": "hwang_base_neg",
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


@requires_rust_lightgbm
def test_tiny_linker_finalization_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        production_model_module,
        "canonical_artifact_hashes",
        lambda: dict(_SYNTHETIC_ARTIFACT_HASHES),
    )
    pairwise_bundle_dir = tmp_path / "pairwise_stage" / "production_model_v9.8"
    write_synthetic_pairwise_bundle(
        pairwise_bundle_dir,
        artifact_hashes=_SYNTHETIC_ARTIFACT_HASHES,
        bundle_version="9.8",
        source_model_version="1.2",
    )
    bundle_dir = tmp_path / "final" / "production_model_v9.8"

    pairwise_manifest = json.loads((pairwise_bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert pairwise_manifest["incremental_linker_version"] is None
    assert "bundle_status" not in pairwise_manifest
    assert "files" not in pairwise_manifest
    assert pairwise_manifest["pairwise_model_version"] == "1.2"
    pairwise_config = json.loads((pairwise_bundle_dir / "clusterer.json").read_text(encoding="utf-8"))
    for field in ("name_tuples_data_sha256", "orcid_prefix_counts_data_sha256"):
        assert len(pairwise_config["feature_contract"][field]) == 64
    with pytest.raises(ValueError, match="Expected a complete"):
        load_production_model(pairwise_bundle_dir)
    assert _load_pairwise_staging_model(pairwise_bundle_dir).production_model_bundle_status == "pairwise_only"

    feature_root = tmp_path / "tiny_linker_feature_bundle"
    target_path = tmp_path / "incremental_linker_training_target.json"
    _write_tiny_promoted_feature_bundle(feature_root, target_path)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(
        promoted_train,
        "_assert_pairwise_model_supports_arrow_materialization",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        promoted_train,
        "_preflight_source_rows",
        lambda *_args, **_kwargs: ({"test_fixture": True}, {}),
    )

    def use_tiny_materialized_features(**kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        assert kwargs["source_bundle"].root == feature_root.resolve()
        return load_bundle(feature_root), [
            {"mode": "arrow-rust", "table_key": table_key, "rows": 1, "test_fixture": True}
            for table_key in promoted_train.REQUIRED_TABLE_KEYS
        ]

    monkeypatch.setattr(promoted_train, "_materialize_arrow_rust_feature_bundle", use_tiny_materialized_features)
    args = promoted_train.build_parser().parse_args(
        [
            "publish",
            "--source-bundle-root",
            str(feature_root),
            "--target-json",
            str(target_path),
            "--pairwise-model-path",
            str(pairwise_bundle_dir),
            "--publish-to",
            str(bundle_dir),
            "--output-dir",
            str(tmp_path / "linker_run"),
        ]
    )
    promoted_train.run(args)

    final_manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert final_manifest["incremental_linker_version"] == "9.8"
    assert "bundle_status" not in final_manifest
    assert "files" not in final_manifest
    assert final_manifest["bundle_version"] == "9.8"
    assert final_manifest["pairwise_model_version"] == "1.2"
    assert _load_pairwise_staging_model(pairwise_bundle_dir).production_model_bundle_status == "pairwise_only"
    assert not (pairwise_bundle_dir / "incremental_linker").exists()
    assert (tmp_path / "linker_run" / "incremental_linker_artifact" / "metadata.json").is_file()

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
