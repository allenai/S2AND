"""Held-out metrics use the published Arrow plan and a complete native model."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from s2and._sha256 import sha256_file
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.model import Clusterer
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION
from s2and.production_model import load_production_model, pairwise_bundle_binding
from s2and.production_training_contract import (
    block_membership_sha256,
    load_model_plan,
    load_packaged_artifact_authority,
)
from scripts.production.model import release_pairwise
from scripts.production.model.plan_datasets import load_development_dataset, open_plan_arrow
from tests.fixture_templates import copy_template
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts, write_pairwise_dataset
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    tiny_logistic_gate_config,
    write_synthetic_pairwise_bundle,
)


def _write(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_plan(plan_path: Path, files: dict[str, dict[str, Path]]) -> None:
    """Write a relocatable plan whose paths resolve from the plan directory."""

    _write(
        plan_path,
        {
            "release_version": "9.9",
            "datasets": {
                name: {
                    role: {
                        "path": Path(os.path.relpath(path, plan_path.parent)).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for role, path in sorted(roles.items())
                }
                for name, roles in files.items()
            },
            "eps": {"grid": [0.5], "minimum_dataset_f1": 0.0, "minimum_signature_weighted_f1": 0.0},
        },
    )


def _build_heldout(tmp_path: Path) -> argparse.Namespace:
    _tuple_path, counts_path, _orcid_path = write_pairwise_artifacts(tmp_path)
    authority = load_packaged_artifact_authority(name_counts_index_root=counts_path)
    root = tmp_path / "dataset"
    source_root = tmp_path / "sources"
    sources = write_pairwise_dataset(source_root)
    signatures = json.loads(sources["signatures"].read_text(encoding="utf-8"))
    keep = list(signatures)[:4]
    development = [key for key in signatures if key not in keep]
    benchmark = convert_release_arrow(source_root, "qian", root / "benchmarks", counts_path)
    heldout = root / "pairwise" / "heldout" / "qian"
    rows = [
        {"signature_id_1": keep[left], "signature_id_2": keep[right], "label": label}
        for left, right, label in ((0, 1, 1), (0, 2, 0), (2, 3, 1))
    ]
    files = {
        "arrow": benchmark / "manifest.json",
        "clusters": benchmark / "qian_clusters.json",
        "heldout_blocks": _write(heldout / "test_blocks.json", {"a adams": keep}),
        "heldout_pairs": _write(heldout / "test_pairs.json", rows),
        "unknown_signatures": _write(benchmark / "restored_unknown_signature_ids_test.json", [keep[3]]),
    }

    # Augmented copies train on development signatures and test on held-out signatures.
    augmented_source = source_root / "augmented"
    augmented_signatures = {}
    for key in (*keep, *development[:4]):
        signature = copy.deepcopy(signatures[key])
        signature["signature_id"] = f"qian___{key}"
        augmented_signatures[signature["signature_id"]] = signature
    _write(augmented_source / "augmented_signatures.json", augmented_signatures)
    for role in ("papers", "specter_embeddings"):
        target = augmented_source / f"augmented_{sources[role].name.partition('_')[2]}"
        target.write_bytes(sources[role].read_bytes())
    augmented = convert_release_arrow(source_root, "augmented", root / "pairwise", counts_path)
    copies = [f"qian___{key}" for key in development[:4]]
    augmented_files = {
        "arrow": augmented / "manifest.json",
        "train_pairs": augmented / "train_pairs.csv",
        "val_pairs": augmented / "val_pairs.csv",
        "heldout_pairs": _write(
            augmented / "test_pairs.json",
            [
                {
                    **row,
                    "signature_id_1": f"qian___{row['signature_id_1']}",
                    "signature_id_2": f"qian___{row['signature_id_2']}",
                }
                for row in rows
            ],
        ),
    }
    augmented_files["train_pairs"].write_text(
        f"signature_id_1,signature_id_2,label\n{copies[0]},{copies[1]},1\n", encoding="utf-8"
    )
    augmented_files["val_pairs"].write_text(
        f"signature_id_1,signature_id_2,label\n{copies[2]},{copies[3]},0\n", encoding="utf-8"
    )
    plan_path = root / "pairwise" / "model_plan.json"
    _write_plan(plan_path, {"qian": files, "augmented": augmented_files})
    model_plan = load_model_plan(plan_path)
    with open_plan_arrow(model_plan.datasets["qian"]) as arrow_dataset:
        development_blocks = load_development_dataset(
            "qian",
            model_plan.datasets["qian"],
            arrow_dataset,
            name_tuples=authority.name_tuples.pairs,
            random_seed=1111,
            n_jobs=1,
        ).get_blocks()

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
        pairwise_training_summary={
            "cluster_test_splits": {
                "qian": {
                    "block_membership_sha256": block_membership_sha256(development_blocks),
                    "test_block_ids": ["a baker"],
                }
            }
        },
    )
    booster, _values = build_tiny_promoted_booster()
    linker = tmp_path / "linker"
    target = _write(tmp_path / "target.json", {})
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
        model_plan=plan_path,
        name_counts_index_root=counts_path,
        output_report=tmp_path / "report.json",
        n_jobs=1,
        total_ram_bytes=1_000_000_000,
        files=files,
        augmented_files=augmented_files,
        dataset_root=root,
        pairwise_stage=stage,
        keep=keep,
        development=development,
        authority=authority,
    )


@pytest.fixture(scope="module")
def _heldout_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, argparse.Namespace]:
    root = tmp_path_factory.mktemp("heldout_template")
    return root, _build_heldout(root)


@pytest.fixture
def heldout_fixture(_heldout_template: tuple[Path, argparse.Namespace], tmp_path: Path) -> argparse.Namespace:
    root, template = _heldout_template
    # The copy keeps the template depth because Arrow manifests find the count index by a relative path.
    copy_template(root, tmp_path)

    def rebase(value: Any) -> Any:
        if isinstance(value, Path) and value.is_relative_to(root):
            return tmp_path / value.relative_to(root)
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        return value

    return argparse.Namespace(**{key: rebase(value) for key, value in vars(template).items()})


def _rewrite_plan_digests(fixture: argparse.Namespace) -> None:
    """Refresh plan digests after a test edits a declared file."""

    payload = json.loads(fixture.model_plan.read_text(encoding="utf-8"))
    for roles in payload["datasets"].values():
        for spec in roles.values():
            spec["sha256"] = sha256_file(fixture.model_plan.parent / spec["path"])
    _write(fixture.model_plan, payload)


def test_heldout_scores_complete_model_exact_pairs_and_full_unknown_context(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    source_bytes = {path: path.read_bytes() for path in fixture.dataset_root.rglob("*") if path.is_file()}
    prediction_calls = []
    predict = Clusterer.predict_from_rust_featurizer

    def record_prediction(self: Clusterer, blocks: Any, featurizer: Any, **kwargs: Any) -> Any:
        prediction_calls.append((blocks, kwargs))
        return predict(self, blocks, featurizer, **kwargs)

    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", record_prediction)
    parsed = release_pairwise.build_parser().parse_args(
        [
            "evaluate-heldout",
            "--model-plan",
            str(fixture.model_plan),
            "--model",
            str(fixture.model),
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
    assert not any("baseline" in key for key in report)
    assert json.loads(fixture.output_report.read_text(encoding="utf-8")) == report
    assert {path: path.read_bytes() for path in source_bytes} == source_bytes


@pytest.mark.parametrize("role", ["heldout_pairs", "unknown_signatures", "arrow"])
def test_heldout_rejects_changed_plan_files_before_features(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    fixture = heldout_fixture
    fixture.files[role].write_bytes(fixture.files[role].read_bytes() + b" ")
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("changed inputs reach prediction")
    )
    with pytest.raises(ValueError, match="model file digest mismatch"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_pairwise_only_stage(heldout_fixture: argparse.Namespace) -> None:
    fixture = heldout_fixture
    fixture.model = fixture.pairwise_stage
    with pytest.raises(ValueError, match="complete"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_a_model_trained_on_other_blocks(heldout_fixture: argparse.Namespace) -> None:
    fixture = heldout_fixture
    excluded = _write(fixture.model_plan.parent / "excluded_signatures.json", [fixture.development[0]])
    payload = json.loads(fixture.model_plan.read_text(encoding="utf-8"))
    payload["datasets"]["qian"]["excluded_signatures"] = {"path": excluded.name, "sha256": sha256_file(excluded)}
    _write(fixture.model_plan, payload)
    with pytest.raises(ValueError, match="differ from the model's training population for 'qian'"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_accepts_a_relocated_dataset_root(heldout_fixture: argparse.Namespace, tmp_path: Path) -> None:
    fixture = heldout_fixture
    moved = tmp_path / "moved"
    # Arrow manifests find the count index by relative path, so both move together.
    counts = fixture.name_counts_index_root.relative_to(fixture.dataset_root.parent)
    shutil.copytree(fixture.dataset_root, moved / fixture.dataset_root.name)
    shutil.copytree(fixture.name_counts_index_root, moved / counts)
    fixture.model_plan = moved / fixture.dataset_root.name / "pairwise" / "model_plan.json"
    fixture.name_counts_index_root = moved / counts

    report = release_pairwise.evaluate_heldout(fixture)

    assert report["pairwise"]["aggregate"]["rows"] == 4


def test_heldout_rejects_outer_test_pairs_outside_the_heldout_blocks(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    rows = json.loads(fixture.files["heldout_pairs"].read_text(encoding="utf-8"))
    rows.append({"signature_id_1": fixture.keep[0], "signature_id_2": fixture.development[0], "label": 0})
    _write(fixture.files["heldout_pairs"], rows)
    _rewrite_plan_digests(fixture)
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("leaked inputs reach prediction")
    )
    with pytest.raises(ValueError, match="outside the held-out blocks"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_augmented_training_copies_of_heldout_signatures(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    fixture.augmented_files["train_pairs"].write_text(
        f"signature_id_1,signature_id_2,label\nqian___{fixture.keep[0]},qian___{fixture.development[0]},0\n",
        encoding="utf-8",
    )
    _rewrite_plan_digests(fixture)
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("leaked inputs reach prediction")
    )
    with pytest.raises(ValueError, match="held-out"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()


def test_heldout_rejects_an_undeclared_uncertainty_sidecar(
    heldout_fixture: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = heldout_fixture
    _write(
        fixture.augmented_files["train_pairs"].parent / "restored_unknown_signature_ids_late.json",
        [f"qian___{fixture.keep[0]}"],
    )
    monkeypatch.setattr(
        release_pairwise, "_pairwise_predictions", lambda *_args: pytest.fail("undeclared mask reaches scoring")
    )
    with pytest.raises(ValueError, match="Undeclared unknown-signature sidecar"):
        release_pairwise.evaluate_heldout(fixture)
    assert not fixture.output_report.exists()
