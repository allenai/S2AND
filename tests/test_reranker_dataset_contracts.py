from __future__ import annotations

import csv
import gzip
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from scripts.reranker_dataset.artifacts import ArtifactCacheKey, FilesystemArtifactStore, NullArtifactStore
from scripts.reranker_dataset.bundle import RerankerBundleContract
from scripts.reranker_dataset.schema import FeatureSchema
from scripts.reranker_dataset.staging import (
    FileRepairSummaryState,
    StageInputGroupsConfig,
    decompress_rows,
    fieldnames_with_materialized_derived_columns,
    stage_input_groups,
)
from scripts.validate_joint_safe_link_official_stack import _summarize_reranker_bundle_contracts


def _create_staged_groups_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE staged_groups (
            source_path TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            query_group_id TEXT NOT NULL,
            rows_before_total INTEGER NOT NULL,
            positive_rows_before_total INTEGER NOT NULL,
            rows_after_window_cap INTEGER NOT NULL,
            positive_rows_after_window_cap INTEGER NOT NULL,
            rows_blob BLOB NOT NULL
        )
        """
    )


def _write_gzip_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _staging_config(source_bundle_root: Path) -> StageInputGroupsConfig:
    return StageInputGroupsConfig(
        source_bundle_root=source_bundle_root,
        s2and_row_relative_path=Path("test") / "s2and_eval_rows.csv.gz",
        s2and_full_relabel_pre_filter_rows_path=source_bundle_root / "missing_pre_filter.csv.gz",
        window_size=25,
        read_initial_only_rereview_decisions=lambda: {},
        read_s2and_full_relabel_decisions=lambda: {},
        merge_initial_only_rereview_into_s2and_decisions=lambda decisions, _initial_only: decisions,
        apply_initial_only_rereview_to_group=lambda rows, *, decision: list(rows),
        apply_s2and_full_relabel_to_group=lambda rows, *, decisions: list(rows),
    )


def test_feature_schema_digest_changes_only_for_column_contract() -> None:
    base = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    same_columns_different_preset = FeatureSchema.from_columns(
        ("retrieval_rank", "retrieval_score"),
        preset="preset_b",
    )
    reordered = FeatureSchema.from_columns(("retrieval_score", "retrieval_rank"), preset="preset_a")
    extended = FeatureSchema.from_columns(
        ("retrieval_rank", "retrieval_score", "cluster_size"),
        preset="preset_a",
    )

    assert same_columns_different_preset.digest == base.digest
    assert reordered.digest != base.digest
    assert extended.digest != base.digest


def test_feature_schema_round_trip_rejects_digest_mismatch() -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    payload = schema.to_json_dict()

    assert FeatureSchema.from_json_dict(payload).digest == schema.digest

    payload["digest"] = "bad"
    with pytest.raises(ValueError, match="Feature schema digest mismatch"):
        FeatureSchema.from_json_dict(payload)


def test_staging_summary_state_merges_worker_results() -> None:
    parent = FileRepairSummaryState(path=r"test\demo_rows.csv.gz")
    parent.record_stage(dataset_name="demo", rows_before=3, positive_rows_before=1)

    worker = FileRepairSummaryState(path=r"test\demo_rows.csv.gz")
    worker.record_result(
        rows_before=3,
        positive_rows_before=1,
        rebuilt_rows=[{"label": 1}, {"label": 0}],
        group_summary={"query_group_id": "q1", "dataset": "demo"},
    )

    parent.merge_result_payload(worker.to_result_payload())
    payload = parent.to_payload()

    assert payload["rows_before"] == 3
    assert payload["rows_after"] == 2
    assert payload["rows_dropped"] == 1
    assert payload["groups_before"] == 1
    assert payload["groups_after"] == 1
    assert payload["positive_rows_before"] == 1
    assert payload["positive_rows_after"] == 1
    assert payload["groups_with_dropped_rows"] == 1
    assert payload["sample_dropped_groups"][0]["query_group_id"] == "q1"


def test_staging_fieldnames_preserve_materialized_derived_columns() -> None:
    fieldnames = ["query_group_id", "candidate_component_key", "label"]

    out = fieldnames_with_materialized_derived_columns(fieldnames)

    assert out[: len(fieldnames)] == fieldnames
    assert "top3_distance_best_gap" in out
    assert "top3_gap_to_heuristic_choice" in out
    assert "raw_max_affiliation_jaccard" in out
    assert "raw_max_coauthor_jaccard" in out
    assert "raw_max_title_jaccard" in out
    assert "raw_max_text_jaccard" in out
    assert len(out) == len(set(out))


def test_stage_input_groups_writes_window_capped_spool_rows(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    relative_path = Path("test") / "demo_rows.csv.gz"
    rows = [
        {
            "dataset": "demo",
            "query_group_id": "q1",
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": "1",
        },
        {
            "dataset": "demo",
            "query_group_id": "q1",
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": "0",
        },
        {
            "dataset": "demo",
            "query_group_id": "q1",
            "candidate_component_key": "c3",
            "retrieval_rank": "30",
            "label": "0",
        },
    ]
    _write_gzip_rows(source_root / relative_path, rows)
    connection = sqlite3.connect(tmp_path / "spool.sqlite3")
    try:
        _create_staged_groups_table(connection)

        fieldnames_by_path, file_summaries, ordered_source_paths = stage_input_groups(
            connection=connection,
            selected_row_paths=(relative_path,),
            limit_groups_per_file=None,
            config=_staging_config(source_root),
        )

        staged = connection.execute(
            """
            SELECT
                source_path,
                group_index,
                dataset_name,
                query_group_id,
                rows_before_total,
                positive_rows_before_total,
                rows_after_window_cap,
                positive_rows_after_window_cap,
                rows_blob
            FROM staged_groups
            """
        ).fetchone()
    finally:
        connection.close()

    source_path = r"test\demo_rows.csv.gz"
    assert ordered_source_paths == [source_path]
    assert fieldnames_by_path[source_path][: len(rows[0])] == list(rows[0])
    assert file_summaries[source_path].rows_before == 3
    assert file_summaries[source_path].positive_rows_before == 1
    assert staged[:8] == (source_path, 1, "demo", "q1", 3, 1, 2, 1)
    assert [row["candidate_component_key"] for row in decompress_rows(bytes(staged[8]))] == ["c1", "c2"]


def test_artifact_cache_key_invalidation_inputs_change_digest() -> None:
    base = ArtifactCacheKey.build(
        namespace="pairwise",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
        constraint_flags={"seed_bypass": True},
        candidate_signature_ids={"c1": ["s1", "s2"]},
    )
    suppress_orcid_changed = ArtifactCacheKey.build(
        namespace="pairwise",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=False,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
        constraint_flags={"seed_bypass": True},
        candidate_signature_ids={"c1": ["s1", "s2"]},
    )
    candidates_changed = ArtifactCacheKey.build(
        namespace="pairwise",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
        constraint_flags={"seed_bypass": True},
        candidate_signature_ids={"c1": ["s1", "s3"]},
    )

    assert suppress_orcid_changed.digest != base.digest
    assert candidates_changed.digest != base.digest


def test_null_artifact_store_is_explicitly_disabled() -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = NullArtifactStore()

    value, decision = store.get(key)
    write_decision = store.put(key, {"payload": 1})

    assert value is None
    assert decision.hit is False
    assert decision.reason == "cache_disabled"
    assert write_decision.hit is False
    assert write_decision.reason == "cache_disabled"


def test_filesystem_artifact_store_writes_content_addressed_artifacts(tmp_path) -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = FilesystemArtifactStore(tmp_path)
    payload = b'{"rows": 3}'

    miss_value, miss_decision = store.get(key)
    write_decision = store.put(key, payload, metadata={"kind": "fixture"})
    hit_value, hit_decision = store.get(key)

    assert miss_value is None
    assert miss_decision.hit is False
    assert miss_decision.reason == "cache_miss"
    assert write_decision.hit is False
    assert write_decision.reason == "cache_written"
    assert hit_value == payload
    assert hit_decision.hit is True
    assert hit_decision.reason == "cache_hit"
    assert store.artifact_path(key) == tmp_path / key.namespace / key.digest[:2] / key.digest / "artifact.bin"
    assert store.read_bytes(key) == payload


def test_filesystem_artifact_store_reports_incomplete_artifact_as_miss(tmp_path) -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = FilesystemArtifactStore(tmp_path)
    store.artifact_dir(key).mkdir(parents=True)
    store.artifact_path(key).write_bytes(b"payload")

    value, decision = store.get(key)

    assert value is None
    assert decision.hit is False
    assert decision.reason == "cache_incomplete"
    with pytest.raises(FileNotFoundError, match="cache_incomplete"):
        store.read_bytes(key)


def test_filesystem_artifact_store_rejects_metadata_digest_mismatch(tmp_path) -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = FilesystemArtifactStore(tmp_path)
    store.write_bytes(key, b"payload")
    metadata = json.loads(store.metadata_path(key).read_text(encoding="utf-8"))
    metadata["digest"] = "bad"
    store.metadata_path(key).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact metadata digest mismatch"):
        store.get(key)


def test_filesystem_artifact_store_rejects_key_metadata_mismatch(tmp_path) -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = FilesystemArtifactStore(tmp_path)
    store.write_bytes(key, b"payload")
    metadata = json.loads(store.metadata_path(key).read_text(encoding="utf-8"))
    metadata["key"]["dataset_digest"] = "dataset-b"
    store.metadata_path(key).write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact metadata key mismatch"):
        store.get(key)


def test_filesystem_artifact_store_rejects_payload_digest_mismatch(tmp_path) -> None:
    key = ArtifactCacheKey.build(
        namespace="component_summary",
        dataset_digest="dataset-a",
        model_digest="model-a",
        suppress_orcid=True,
        feature_schema_digest="schema-a",
        feature_preset="preset-a",
    )
    store = FilesystemArtifactStore(tmp_path)
    store.write_bytes(key, b"payload")
    store.artifact_path(key).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Artifact payload digest mismatch"):
        store.get(key)


def test_reranker_bundle_contract_round_trip(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    contract = RerankerBundleContract(
        feature_schema=schema,
        calibration_surface="classic_gate_only",
        migration_manifest={"row_engine": "legacy_bridge"},
    )
    path = tmp_path / "bundle_contract.json"

    contract.write_json(path)
    loaded = RerankerBundleContract.read_json(path)

    assert loaded.feature_schema.digest == schema.digest
    assert loaded.calibration_surface == "classic_gate_only"
    assert dict(loaded.migration_manifest) == {"row_engine": "legacy_bridge"}


def test_reranker_bundle_contract_rejects_nested_schema_digest_mismatch() -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    payload = RerankerBundleContract(feature_schema=schema).to_json_dict()
    payload["feature_schema"]["digest"] = "bad"

    with pytest.raises(ValueError, match="Feature schema digest mismatch"):
        RerankerBundleContract.from_json_dict(payload)


def test_validator_treats_reranker_contract_artifacts_as_optional(tmp_path) -> None:
    summary = _summarize_reranker_bundle_contracts(tmp_path)

    assert summary["feature_schema_present"] is False
    assert summary["bundle_contract_present"] is False


def test_validator_validates_optional_reranker_contract_artifacts(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(schema.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    RerankerBundleContract(feature_schema=schema).write_json(tmp_path / "bundle_contract.json")

    summary = _summarize_reranker_bundle_contracts(tmp_path)

    assert summary["feature_schema_present"] is True
    assert summary["bundle_contract_present"] is True
    assert summary["feature_schema_digest"] == schema.digest
    assert summary["bundle_contract_feature_schema_digest"] == schema.digest
    assert summary["calibration_surface"] == "classic_gate_only"
    assert summary["calibrator_present"] is False


def test_validator_validates_optional_calibrator_artifacts(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    calibration = {
        "enabled": True,
        "mode": "heldout",
        "surface": "ranker_heldout",
        "method": "isotonic",
        "feature_schema_digest": schema.digest,
        "inner_split_group_overlap_with_training": 0,
    }
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(schema.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    RerankerBundleContract(
        feature_schema=schema,
        calibration_surface="ranker_heldout",
    ).write_json(tmp_path / "bundle_contract.json")
    (tmp_path / "calibrator.pkl").write_bytes(
        pickle.dumps(
            {
                "calibrator": "placeholder",
                "feature_schema": schema.to_json_dict(),
                "calibration": calibration,
            }
        )
    )
    (tmp_path / "calibrator_summary.json").write_text(json.dumps(calibration), encoding="utf-8")

    summary = _summarize_reranker_bundle_contracts(tmp_path)

    assert summary["calibrator_present"] is True
    assert summary["calibrator_summary_present"] is True
    assert summary["calibrator_feature_schema_digest"] == schema.digest
    assert summary["calibrator_summary_feature_schema_digest"] == schema.digest
    assert summary["calibrator_surface"] == "ranker_heldout"


def test_validator_requires_calibrator_for_ranker_calibration_surface(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema.to_json_dict()), encoding="utf-8")
    RerankerBundleContract(feature_schema=schema, calibration_surface="ranker_heldout").write_json(
        tmp_path / "bundle_contract.json"
    )

    with pytest.raises(ValueError, match="calibrator artifacts are required"):
        _summarize_reranker_bundle_contracts(tmp_path)


def test_validator_rejects_calibrator_schema_digest_mismatch(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    calibration = {
        "enabled": True,
        "surface": "ranker_heldout",
        "feature_schema_digest": "bad",
        "inner_split_group_overlap_with_training": 0,
    }
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema.to_json_dict()), encoding="utf-8")
    RerankerBundleContract(feature_schema=schema, calibration_surface="ranker_heldout").write_json(
        tmp_path / "bundle_contract.json"
    )
    (tmp_path / "calibrator.pkl").write_bytes(
        pickle.dumps(
            {
                "calibrator": "placeholder",
                "feature_schema": schema.to_json_dict(),
                "calibration": calibration,
            }
        )
    )
    (tmp_path / "calibrator_summary.json").write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match="Calibrator feature schema digest mismatch"):
        _summarize_reranker_bundle_contracts(tmp_path)


def test_validator_rejects_calibrator_training_overlap(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    calibration = {
        "enabled": True,
        "surface": "ranker_heldout",
        "feature_schema_digest": schema.digest,
        "inner_split_group_overlap_with_training": 1,
    }
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema.to_json_dict()), encoding="utf-8")
    RerankerBundleContract(feature_schema=schema, calibration_surface="ranker_heldout").write_json(
        tmp_path / "bundle_contract.json"
    )
    (tmp_path / "calibrator.pkl").write_bytes(
        pickle.dumps(
            {
                "calibrator": "placeholder",
                "feature_schema": schema.to_json_dict(),
                "calibration": calibration,
            }
        )
    )
    (tmp_path / "calibrator_summary.json").write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match="Calibrator inner split overlaps training groups"):
        _summarize_reranker_bundle_contracts(tmp_path)


def test_validator_rejects_corrupt_optional_feature_schema_digest(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    payload = schema.to_json_dict()
    payload["digest"] = "bad"
    (tmp_path / "feature_schema.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Feature schema digest mismatch"):
        _summarize_reranker_bundle_contracts(tmp_path)
