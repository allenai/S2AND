"""Held-out metrics use exact portable inputs and a complete native model."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_provenance import ArrowProvenanceError
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.model import Clusterer
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION
from s2and.production_model import load_production_model, pairwise_bundle_binding
from s2and.production_training_contract import load_model_plan, load_packaged_artifact_authority
from scripts.production.model import release_pairwise
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts, write_pairwise_dataset
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    tiny_logistic_gate_config,
    write_synthetic_pairwise_bundle,
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.fixture
def heldout_fixture(tmp_path: Path) -> argparse.Namespace:
    _tuple_path, counts_path, _orcid_path = write_pairwise_artifacts(tmp_path)
    authority = load_packaged_artifact_authority(name_counts_index_root=counts_path)
    pairwise_root = tmp_path / "snapshot" / "pairwise"
    source_root = pairwise_root / "sources"
    files = write_pairwise_dataset(source_root)
    signatures = json.loads(files["signatures"].read_text(encoding="utf-8"))
    keep = list(signatures)[:4]
    clusters = json.loads(files["clusters"].read_text(encoding="utf-8"))
    model_root = source_root / "model" / "qian"
    model_files = {
        "signatures": model_root / "qian_signatures.json",
        "clusters": model_root / "qian_clusters.json",
        "papers": model_root / files["papers"].name,
        "specter_embeddings": model_root / files["specter_embeddings"].name,
    }
    model_root.mkdir(parents=True)
    for role in ("papers", "specter_embeddings"):
        model_files[role].write_bytes(files[role].read_bytes())
    _write(model_files["signatures"], {key: value for key, value in signatures.items() if key not in keep})
    _write(
        model_files["clusters"],
        {key: value for key, value in clusters.items() if value["signature_ids"][0] not in keep},
    )
    _write(files["signatures"], {key: signatures[key] for key in keep})
    _write(files["clusters"], {key: value for key, value in clusters.items() if value["signature_ids"][0] in keep})
    blocks = files["signatures"].parent / "test_blocks.json"
    _write(blocks, {"a adams": keep})
    pairs = files["signatures"].parent / "test_pairs.json"
    rows = [
        {"signature_id_1": keep[left], "signature_id_2": keep[right], "label": label}
        for left, right, label in ((0, 1, 1), (0, 2, 0), (2, 3, 1))
    ]
    _write(pairs, rows)
    unknown = files["signatures"].parent / "restored_unknown_signature_ids_test.json"
    _write(unknown, [keep[3]])
    files.update(blocks=blocks, pairs=pairs, unknown_signatures=unknown)

    augmented_root = source_root / "augmented"
    augmented_files = {}
    for role in ("signatures", "papers", "specter_embeddings"):
        target = augmented_root / f"augmented_{files[role].name.partition('_')[2]}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if role == "signatures":
            augmented_signatures = {}
            for key in keep:
                signature = copy.deepcopy(signatures[key])
                signature["signature_id"] = f"qian___{key}"
                augmented_signatures[signature["signature_id"]] = signature
            _write(target, augmented_signatures)
        else:
            target.write_bytes(files[role].read_bytes())
        augmented_files[role] = target
    augmented_pairs = augmented_root / "test_pairs.json"
    _write(
        augmented_pairs,
        [
            {
                **row,
                "signature_id_1": f"qian___{row['signature_id_1']}",
                "signature_id_2": f"qian___{row['signature_id_2']}",
            }
            for row in rows
        ],
    )
    augmented_files["pairs"] = augmented_pairs
    arrow_root = pairwise_root / "arrow" / "evaluation"
    convert_release_arrow(source_root, "qian", arrow_root, counts_path)
    convert_release_arrow(source_root, "augmented", arrow_root, counts_path)

    def _relative_specs(dataset_files: dict[str, Path]) -> dict[str, dict[str, str]]:
        return {
            role: {"path": path.relative_to(pairwise_root).as_posix(), "sha256": sha256_file(path)}
            for role, path in sorted(dataset_files.items())
        }

    input_payload = {
        "release_version": "9.9",
        "sources": {
            "model": {"qian": _relative_specs(model_files)},
            "evaluation": {
                name: {
                    role: {"path": path.relative_to(pairwise_root).as_posix(), "sha256": sha256_file(path)}
                    for role, path in dataset_files.items()
                }
                for name, dataset_files in (("qian", files), ("augmented", augmented_files))
            },
        },
    }
    inputs_path = pairwise_root / "inputs.json"
    _write(inputs_path, input_payload)
    model_plan_path = pairwise_root / "model_plan.json"
    _write(
        model_plan_path,
        {
            "release_version": "9.9",
            "datasets": {"qian": _relative_specs(model_files)},
            "eps": {"grid": [0.5], "minimum_dataset_f1": 0.0, "minimum_signature_weighted_f1": 0.0},
        },
    )
    model_plan = load_model_plan(model_plan_path)
    stage = tmp_path / "pairwise-stage"
    source_clusterer = write_synthetic_pairwise_bundle(
        tmp_path / "model-source", artifact_hashes=authority.hashes, release_version="9.9"
    )
    write_pairwise_production_bundle(
        source_clusterer,
        stage,
        release_version="9.9",
        eps_calibration=CALIBRATED_EPS_CALIBRATION,
        pairwise_training_config={
            "training_scope": "production_full",
            "model_plan_sha256": model_plan.sha256,
            "model_plan_content_sha256": model_plan.content_sha256,
            "data_random_seed": 1111,
            "input_artifact_hashes": authority.hashes,
        },
        pairwise_training_summary={},
    )
    booster, _values = build_tiny_promoted_booster()
    linker = tmp_path / "linker"
    target = tmp_path / "target.json"
    _write(target, {})
    save_incremental_linking_artifact(
        booster,
        linker,
        target_spec={},
        gate_config=tiny_logistic_gate_config(),
        pairwise_bundle_binding=pairwise_bundle_binding(stage, expected_artifact_hashes=authority.hashes),
    )
    complete = tmp_path / "complete-model"
    finalize_production_bundle(
        pairwise_bundle_dir=stage,
        output_bundle_dir=complete,
        incremental_linker_artifact_dir=linker,
        target_json=target,
        expected_artifact_hashes=authority.hashes,
    )
    return argparse.Namespace(
        model=complete,
        inputs_json=inputs_path,
        arrow_root=arrow_root,
        name_counts_index_root=counts_path,
        output_report=tmp_path / "report.json",
        n_jobs=1,
        total_ram_bytes=1_000_000_000,
        files=files,
        pairwise_stage=stage,
        keep=keep,
        authority=authority,
    )


def test_heldout_scores_complete_model_exact_pairs_and_full_unknown_context(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    source_bytes = {path: path.read_bytes() for path in fixture.inputs_json.parent.rglob("*") if path.is_file()}
    prediction_calls = []
    predict = Clusterer.predict_from_rust_featurizer

    def record_prediction(self: Clusterer, blocks: Any, featurizer: Any, **kwargs: Any) -> Any:
        prediction_calls.append((blocks, kwargs))
        return predict(self, blocks, featurizer, **kwargs)

    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", record_prediction)
    parsed = release_pairwise.build_parser().parse_args(
        [
            "evaluate-heldout",
            "--inputs-json",
            str(fixture.inputs_json),
            "--model",
            str(fixture.model),
            "--arrow-root",
            str(fixture.arrow_root),
            "--name-counts-index-root",
            str(fixture.name_counts_index_root),
            "--output-report",
            str(fixture.output_report),
            "--n-jobs",
            "1",
            "--total-ram-bytes",
            str(fixture.total_ram_bytes),
        ]
    )
    report = parsed.handler(parsed)

    assert report["pairwise"]["aggregate"]["rows"] == 4
    assert set(report["pairwise"]["datasets"]) == {"qian", "augmented"}
    for dataset in report["pairwise"]["datasets"].values():
        assert dataset["metrics"]["rows"] == 2
        assert dataset["unknown_pairs_removed"] == 1
    assert set(report["cluster"]["datasets"]) == {"qian"}
    assert report["cluster"]["signature_count"] == 3
    assert report["cluster"]["datasets"]["qian"]["inference_signatures"] == 4
    assert report["cluster"]["datasets"]["qian"]["unknown_signatures"] == 1
    assert prediction_calls == [
        ({"a adams": fixture.keep}, {"cluster_seeds_disallow": set(), "total_ram_bytes": fixture.total_ram_bytes})
    ]
    loaded = load_production_model(fixture.model, expected_artifact_hashes=fixture.authority.hashes)
    assert loaded.incremental_linker_artifact is not None
    assert (
        report["clusterer_config"]["use_default_constraints_as_supervision"]
        == loaded.use_default_constraints_as_supervision
    )
    assert report["candidate_model_manifest_sha256"] == sha256_file(fixture.model / "manifest.json")
    assert report["evaluation_sources"]["qian"]["unknown_signatures"] == sha256_file(
        fixture.files["unknown_signatures"]
    )
    assert not any("baseline" in key for key in report)
    assert json.loads(fixture.output_report.read_text(encoding="utf-8")) == report
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


@pytest.mark.parametrize("role", ["pairs", "unknown_signatures"])
def test_heldout_rejects_changed_labels_or_masks_before_features(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    fixture = heldout_fixture
    fixture.files[role].write_bytes(fixture.files[role].read_bytes() + b" ")
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("changed inputs reach prediction")
    )
    with pytest.raises(ValueError, match="changed after release preparation"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_arrow_from_different_actual_source(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    papers = json.loads(fixture.files["papers"].read_text(encoding="utf-8"))
    next(iter(papers.values()))["title"] = "Changed source title"
    _write(fixture.files["papers"], papers)
    inputs = json.loads(fixture.inputs_json.read_text(encoding="utf-8"))
    for dataset in inputs["sources"]["evaluation"].values():
        if dataset["papers"]["path"] == fixture.files["papers"].relative_to(fixture.inputs_json.parent).as_posix():
            dataset["papers"]["sha256"] = sha256_file(fixture.files["papers"])
    _write(fixture.inputs_json, inputs)
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("wrong Arrow source reaches prediction")
    )
    with pytest.raises(ArrowProvenanceError, match="source"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_pairwise_only_stage(heldout_fixture: argparse.Namespace) -> None:
    fixture = heldout_fixture
    fixture.model = fixture.pairwise_stage
    with pytest.raises(ValueError, match="complete"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_another_training_plan(heldout_fixture: argparse.Namespace) -> None:
    fixture = heldout_fixture
    plan = fixture.inputs_json.parent / "model_plan.json"
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["eps"]["grid"] = [0.5, 0.6]
    _write(plan, payload)
    with pytest.raises(ValueError, match="different model plan"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_accepts_relocated_plan_with_same_content(heldout_fixture: argparse.Namespace, tmp_path: Path) -> None:
    fixture = heldout_fixture
    relocated = tmp_path / "relocated" / "pairwise"
    shutil.copytree(fixture.inputs_json.parent, relocated)
    plan = relocated / "model_plan.json"
    # Reformatted bytes keep the same files, roles, digests, and policy.
    plan.write_text(json.dumps(json.loads(plan.read_text(encoding="utf-8")), indent=2), encoding="utf-8")
    assert sha256_file(plan) != sha256_file(fixture.inputs_json.parent / "model_plan.json")
    fixture.inputs_json = relocated / "inputs.json"
    fixture.arrow_root = relocated / "arrow" / "evaluation"

    report = release_pairwise.evaluate_heldout(fixture)

    assert report["model_plan_sha256"] == sha256_file(plan)
    assert report["model_plan_content_sha256"] == load_model_plan(plan).content_sha256


def _legacy_record(original: Path) -> dict[str, Any]:
    return {"training_scope": "production_full", "model_plan_sha256": sha256_file(original), "data_random_seed": 1}


def test_byte_only_training_record_requires_original_plan_proof(
    heldout_fixture: argparse.Namespace, tmp_path: Path
) -> None:
    original = heldout_fixture.inputs_json.parent / "model_plan.json"
    relocated = tmp_path / "relocated" / "pairwise"
    shutil.copytree(original.parent, relocated)
    plan_path = relocated / "model_plan.json"
    plan_path.write_text(json.dumps(json.loads(plan_path.read_text(encoding="utf-8")), indent=2), encoding="utf-8")
    plan = load_model_plan(plan_path)
    record = _legacy_record(original)

    with pytest.raises(ValueError, match="requires --original-model-plan"):
        release_pairwise._require_trained_plan(record, plan, original_plan=None)
    release_pairwise._require_trained_plan(record, plan, original_plan=original)

    with pytest.raises(ValueError, match="bytes do not match"):
        release_pairwise._require_trained_plan(record, plan, original_plan=plan_path)
    changed = json.loads(plan_path.read_text(encoding="utf-8"))
    changed["eps"]["grid"] = [0.5, 0.6]
    _write(plan_path, changed)
    with pytest.raises(ValueError, match="different model plan"):
        release_pairwise._require_trained_plan(record, load_model_plan(plan_path), original_plan=original)
