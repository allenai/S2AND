from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from s2and.arrow_inputs import build_arrow_artifact_manifest, write_arrow_artifact_manifest
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from scripts.production.model.linker_source_bundle import (
    MANIFEST_SCHEMA,
    SPEC_SCHEMA,
    assemble_source_bundle,
    validate_source_bundle,
)
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_arrow_root(
    root: Path,
    *,
    dataset: str,
    replay: bool,
    name_counts_source: Path | None = None,
) -> None:
    root.mkdir(parents=True)
    (root / "LICENSE.txt").write_text("fixture license\n", encoding="utf-8")
    if name_counts_source is None:
        write_name_counts_index(root, tiny_name_counts_tuple(), tiny_name_counts_provenance())
    else:
        shutil.copytree(name_counts_source, root / "name_counts_index")
    dataset_root = root / "datasets" / dataset if replay else root / dataset

    table_specs = {
        "signatures": (
            pa.table({"signature_id": pa.array(["s1"], type=pa.string())}),
            "signature_id",
        ),
        "papers": (
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
            "paper_id",
        ),
        "paper_authors": (
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
            "paper_id",
        ),
        "specter": (
            pa.table({"paper_id": pa.array(["p1"], type=pa.string())}),
            "paper_id",
        ),
    }
    paths: dict[str, str] = {}
    index_keys = {
        "signatures": "signatures_batch_index",
        "papers": "papers_batch_index",
        "paper_authors": "paper_authors_batch_index",
        "specter": "specter_batch_index",
    }
    for table_name, (table, key_column) in table_specs.items():
        filename = "specter2.arrow" if table_name == "specter" else f"{table_name}.arrow"
        table_path = dataset_root / filename
        paths[table_name] = write_arrow_ipc_table(table, table_path)
        index_key = index_keys[table_name]
        index_path = dataset_root / f"{table_name}.{index_key}.bin"
        paths[index_key], _metrics = write_arrow_batch_lookup_index(
            table_path,
            index_path,
            key_column=key_column,
            table_name=table_name,
        )
    paths["name_counts_index"] = str(root / "name_counts_index")
    dataset_manifest = build_arrow_artifact_manifest(
        paths,
        dataset_root,
        metadata={"dataset": dataset, "signature_count": 1, "paper_count": 1},
    )
    manifest_path = write_arrow_artifact_manifest(dataset_manifest, dataset_root)
    relative_manifest = manifest_path.relative_to(root).as_posix()
    root_manifest = {
        "schema": "inference_arrow_bundle_v1",
        "datasets": [dataset],
        "dataset_manifests": [
            {
                "dataset": dataset,
                "dataset_dir": dataset_root.relative_to(root).as_posix(),
                "manifest_path": relative_manifest,
                "manifest_size_bytes": manifest_path.stat().st_size,
                "manifest_sha256": _sha256(manifest_path),
                "validation_requirements": {
                    "require_embeddings": True,
                    "require_name_counts_index": True,
                },
            }
        ],
        "audit": {"dataset_count": 1},
    }
    _write_json(root / "manifest.json", root_manifest)


def _write_linker_inputs(root: Path, *, leak: bool) -> Path:
    label_paths = {
        "train_path": "labels/train.parquet",
        "classic_gate_source_path": "labels/calibration.parquet",
        "s2and_eval_path": "labels/s2and_eval.parquet",
        "hwang_eval_path": "labels/hwang_eval.parquet",
    }
    row_ids = {
        "train_path": ("train:q1", "train:b1"),
        "classic_gate_source_path": ("cal:q1", "cal:b1"),
        "s2and_eval_path": ("s2and:q1", "s2and:b1"),
        "hwang_eval_path": ("hwang:q1", "cal:b1" if leak else "hwang:b1"),
    }
    for key, relative_path in label_paths.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        query_group_id, base_group_id = row_ids[key]
        pd.DataFrame(
            {
                "query_group_id": [query_group_id],
                "base_group_id": [base_group_id],
                "dataset": ["replay_demo"],
                "query_view": ["full"],
                "candidate_component_key": ["component-1"],
                "retrieval_rank": [1],
                "label": [1],
            }
        ).to_parquet(path, index=False)

    candidate_path = root / "components" / "replay_demo_members.parquet"
    candidate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "candidate_component_key": ["component-1"],
            "member_index": [0],
            "signature_id": ["s1"],
        }
    ).to_parquet(candidate_path, index=False)

    assignments = pd.DataFrame(
        {
            "query_group_id": ["cal:q1", "s2and:q1", "hwang:q1"],
            "source_key": ["s2and_eval", "s2and_eval", "hwang_eval"],
            "split": ["calibration_fit", "calibration_fit", "test"],
            "base_group_id": ["cal:b1", "s2and:b1", "cal:b1" if leak else "hwang:b1"],
        }
    )
    assignments_path = root / "splits" / "assignments.csv"
    assignments_path.parent.mkdir(parents=True)
    assignments.to_csv(assignments_path, index=False)
    _write_json(root / "splits" / "summary.json", {"row_count": 3})
    pd.DataFrame({"base_group_id": ["cal:b1"]}).to_csv(root / "splits" / "internal_eval.csv", index=False)

    bundle = {
        "schema": "s2and_linker_replay_arrow_bundle_v1",
        "bundle_name": "mini-linker-source",
        "assets": {
            "candidate_members": {
                "root": "components",
                "datasets": {"replay_demo": "components/replay_demo_members.parquet"},
            },
            "featureless_rows": {"root": "labels", "files": label_paths},
            "splits": {
                "root": "splits",
                "assignments_path": "splits/assignments.csv",
                "summary_path": "splits/summary.json",
            },
        },
        "models": {
            "classic": {
                "classic_gate_source_path": label_paths["classic_gate_source_path"],
                "s2and_eval_path": label_paths["s2and_eval_path"],
                "hwang_eval_path": label_paths["hwang_eval_path"],
                "classic_gate_internal_eval_base_groups_path": "splits/internal_eval.csv",
                "stratified_eval_test_split": {
                    "assignments_path": "splits/assignments.csv",
                    "summary_path": "splits/summary.json",
                    "calibration_fit_split": "calibration_fit",
                    "test_split": "test",
                },
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit"],
                    "test_split": "test",
                },
            }
        },
        "expected_metrics": {},
    }
    _write_json(root / "bundle.json", bundle)

    members = [
        {"path": "bundle.json", "role": "bundle.definition"},
        {"path": "components/replay_demo_members.parquet", "role": "candidate_members.replay_demo"},
        *[{"path": path, "role": f"featureless_rows.{key}"} for key, path in label_paths.items()],
        {"path": "splits/assignments.csv", "role": "splits.assignments"},
        {"path": "splits/internal_eval.csv", "role": "splits.internal_eval"},
        {"path": "splits/summary.json", "role": "splits.summary"},
    ]
    spec_path = root.parent / "member_spec.json"
    _write_json(spec_path, {"schema": SPEC_SCHEMA, "members": members})
    return spec_path


def _inputs(tmp_path: Path, *, leak: bool = False) -> dict[str, Path]:
    source_root = tmp_path / "linker-input"
    benchmark_root = tmp_path / "benchmark-arrow"
    replay_root = tmp_path / "replay-arrow"
    source_root.mkdir()
    spec_path = _write_linker_inputs(source_root, leak=leak)
    _write_arrow_root(benchmark_root, dataset="benchmark_demo", replay=False)
    _write_arrow_root(
        replay_root,
        dataset="replay_demo",
        replay=True,
        name_counts_source=benchmark_root / "name_counts_index",
    )
    return {
        "member_spec_path": spec_path,
        "source_root": source_root,
        "benchmark_arrow_root": benchmark_root,
        "replay_arrow_root": replay_root,
        "output_source_bundle": tmp_path / "source-final",
        "output_data_root": tmp_path / "data-final",
    }


@pytest.fixture
def assembled(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    inputs = _inputs(tmp_path)
    return inputs, assemble_source_bundle(**inputs)


def test_assemble_and_validate_minimal_source_bundle(
    assembled: tuple[dict[str, Path], dict],
) -> None:
    inputs, report = assembled
    source_root = inputs["output_source_bundle"]
    data_root = inputs["output_data_root"]
    manifest = json.loads((source_root / "source_bundle_manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert {entry["role"] for entry in manifest["members"]} >= {
        "bundle.definition",
        "candidate_members.replay_demo",
        "featureless_rows.train_path",
        "splits.assignments",
    }
    assert report["selected_source_rows"] == 4
    assert validate_source_bundle(source_root, data_root) == report
    with pytest.raises(FileExistsError, match="must not exist"):
        assemble_source_bundle(**inputs)


def test_source_inventory_rejects_same_size_same_mtime_mutation(
    assembled: tuple[dict[str, Path], dict],
) -> None:
    inputs, _report = assembled
    path = inputs["output_source_bundle"] / "labels" / "train.parquet"
    original_stat = path.stat()
    mutated = bytearray(path.read_bytes())
    mutated[-1] ^= 1
    path.write_bytes(mutated)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert path.stat().st_size == original_stat.st_size
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_source_bundle(inputs["output_source_bundle"], inputs["output_data_root"])


def test_source_inventory_rejects_missing_member(
    assembled: tuple[dict[str, Path], dict],
) -> None:
    inputs, _report = assembled
    (inputs["output_source_bundle"] / "splits" / "summary.json").unlink()

    with pytest.raises(ValueError, match="is missing"):
        validate_source_bundle(inputs["output_source_bundle"], inputs["output_data_root"])


def test_source_inventory_rejects_undeclared_member(
    assembled: tuple[dict[str, Path], dict],
) -> None:
    inputs, _report = assembled
    (inputs["output_source_bundle"] / "unexpected.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(ValueError, match="undeclared"):
        validate_source_bundle(inputs["output_source_bundle"], inputs["output_data_root"])


def test_assembly_rejects_base_identity_leakage(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, leak=True)

    with pytest.raises(ValueError, match="base_group_id values in multiple splits"):
        assemble_source_bundle(**inputs)
