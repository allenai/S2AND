from __future__ import annotations

import json

import pytest

from scripts.reranker_dataset.artifacts import ArtifactCacheKey, NullArtifactStore
from scripts.reranker_dataset.bundle import RerankerBundleContract
from scripts.reranker_dataset.schema import FeatureSchema
from scripts.validate_joint_safe_link_official_stack import _summarize_reranker_bundle_contracts


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


def test_validator_rejects_corrupt_optional_feature_schema_digest(tmp_path) -> None:
    schema = FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="preset_a")
    payload = schema.to_json_dict()
    payload["digest"] = "bad"
    (tmp_path / "feature_schema.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Feature schema digest mismatch"):
        _summarize_reranker_bundle_contracts(tmp_path)
