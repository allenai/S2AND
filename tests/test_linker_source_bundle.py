from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from s2and.arrow_inputs import (
    ARROW_COLLECTION_KIND,
    PUBLIC_DATA_KIND,
    build_arrow_artifact_manifest,
    write_arrow_artifact_manifest,
)
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from s2and.incremental_linking_training.classic import load_bundle
from s2and.incremental_linking_training.source_bundle_preflight import preflight_source_rows
from scripts.production.model.linker_source_bundle import assemble_source_bundle
from scripts.verification.validate_local_arrow_release import validate_release_root
from tests.helpers import tiny_name_counts_tuple


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
        write_name_counts_index(root, tiny_name_counts_tuple())
    else:
        shutil.copytree(name_counts_source, root / "name_counts_index")
    dataset_root = root / "datasets" / dataset if replay else root / dataset

    table_specs = {
        "signatures": (
            pa.table(
                {
                    "signature_id": pa.array(["s1"], type=pa.string()),
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "author_first": pa.array(["Ada"], type=pa.string()),
                    "author_middle": pa.array([""], type=pa.string()),
                    "author_last": pa.array(["Lovelace"], type=pa.string()),
                    "author_suffix": pa.array([""], type=pa.string()),
                    "author_affiliations": pa.array([[]], type=pa.list_(pa.string())),
                    "author_position": pa.array([0], type=pa.int64()),
                }
            ),
            "signature_id",
        ),
        "papers": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "title": pa.array(["Notes"], type=pa.string()),
                    "venue": pa.array([""], type=pa.string()),
                    "journal_name": pa.array([""], type=pa.string()),
                }
            ),
            "paper_id",
        ),
        "paper_authors": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "position": pa.array([0], type=pa.int64()),
                    "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
                }
            ),
            "paper_id",
        ),
        "specter": (
            pa.table(
                {
                    "paper_id": pa.array(["p1"], type=pa.string()),
                    "embedding": pa.FixedSizeListArray.from_arrays(
                        pa.array([1.0, 0.0], type=pa.float32()),
                        2,
                    ),
                }
            ),
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
    )
    manifest_path = write_arrow_artifact_manifest(dataset_manifest, dataset_root)
    relative_manifest = manifest_path.relative_to(root).as_posix()
    root_manifest = {
        "kind": ARROW_COLLECTION_KIND,
        "format_version": PUBLIC_DATA_FORMAT_VERSION,
        "dataset_manifests": {
            dataset: {
                "path": relative_manifest,
                "sha256": _sha256(manifest_path),
            }
        },
    }
    _write_json(root / "manifest.json", root_manifest)


def _write_linker_inputs(root: Path, *, leak: bool) -> None:
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


def _inputs(tmp_path: Path, *, leak: bool = False) -> dict[str, Path]:
    source_root = tmp_path / "linker-input"
    benchmark_root = tmp_path / "benchmark-arrow"
    replay_root = tmp_path / "replay-arrow"
    source_root.mkdir(parents=True)
    _write_linker_inputs(source_root, leak=leak)
    _write_arrow_root(benchmark_root, dataset="benchmark_demo", replay=False)
    _write_arrow_root(
        replay_root,
        dataset="replay_demo",
        replay=True,
        name_counts_source=benchmark_root / "name_counts_index",
    )
    model_files: dict[str, dict[str, str]] = {}
    for role in ("papers", "signatures", "specter_embeddings", "clusters"):
        path = tmp_path / "model-plan-inputs" / f"{role}.json"
        _write_json(path, {})
        model_files[role] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    model_plan = tmp_path / "model-plan.json"
    _write_json(
        model_plan,
        {
            "release_version": "1.3",
            "datasets": {"fixture": model_files},
            "eps": {
                "grid": [0.3, 0.6],
                "minimum_dataset_f1": 0.0,
                "minimum_signature_weighted_f1": 0.0,
            },
        },
    )
    return {
        "source_root": source_root,
        "benchmark_arrow_root": benchmark_root,
        "replay_arrow_root": replay_root,
        "name_counts_index": benchmark_root / "name_counts_index",
        "model_plan": model_plan,
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
    assert report["selected_source_rows"] == 4
    assert report["source_bundle"] == str(source_root.resolve())
    assert report["data_root"] == str(data_root.resolve())
    assert report["release_version"] == "1.3"
    assert report["name_counts_manifest_sha256"] == _sha256(inputs["name_counts_index"] / "manifest.json")
    assert (source_root / "bundle.json").is_file()
    assert (data_root / "manifest.json").is_file()
    assert sorted(
        path.relative_to(source_root).as_posix() for path in source_root.rglob("name_counts_index") if path.is_dir()
    ) == ["name_counts_index"]
    assert sorted(
        path.relative_to(data_root).as_posix() for path in data_root.rglob("name_counts_index") if path.is_dir()
    ) == ["name_counts_index"]
    assert validate_release_root(data_root)["replay_dataset_manifest_count"] == 1
    public_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    assert set(public_manifest) == {
        "kind",
        "release_version",
        "format_version",
        "dataset_manifests",
        "replay_bundles",
    }
    assert public_manifest["kind"] == PUBLIC_DATA_KIND
    assert public_manifest["release_version"] == "1.3"
    assert public_manifest["format_version"] == PUBLIC_DATA_FORMAT_VERSION
    with pytest.raises(FileExistsError, match="must not exist"):
        assemble_source_bundle(**inputs)


def test_assembly_rejects_source_leakage_and_published_arrow_inputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, leak=True)

    with pytest.raises(ValueError, match="base_group_id values in multiple splits"):
        assemble_source_bundle(**inputs)

    published_inputs = _inputs(tmp_path / "published")
    manifest_path = published_inputs["benchmark_arrow_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kind"] = PUBLIC_DATA_KIND
    manifest["release_version"] = "1.3"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="must be a generic Arrow collection"):
        assemble_source_bundle(**published_inputs)
    assert not published_inputs["output_source_bundle"].exists()
    assert not published_inputs["output_data_root"].exists()


def test_assembled_roots_are_movable_and_self_contained(
    assembled: tuple[dict[str, Path], dict],
    tmp_path: Path,
) -> None:
    inputs, _report = assembled
    moved_source = tmp_path / "moved" / "source"
    moved_data = tmp_path / "moved" / "data"
    shutil.copytree(inputs["output_source_bundle"], moved_source)
    shutil.copytree(inputs["output_data_root"], moved_data)

    assert validate_release_root(moved_data)["replay_dataset_manifest_count"] == 1
    moved_bundle = load_bundle(moved_source)
    with ExitStack() as arrow_stack:
        selected_rows, datasets = preflight_source_rows(
            moved_bundle,
            name_counts_index_root=moved_source / "name_counts_index",
            arrow_stack=arrow_stack,
        )
        assert selected_rows == 4
        assert set(datasets) == {"replay_demo"}


def test_assembly_rebinds_manifests_to_the_single_published_name_count_index(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    assemble_source_bundle(**inputs)

    data_root = inputs["output_data_root"]
    replay_root = data_root / "linker_replay"
    replay_manifest_path = replay_root / "manifest.json"
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    assert set(replay_manifest) == {"kind", "format_version", "dataset_manifests"}
    replay_dataset_entry = replay_manifest["dataset_manifests"]["replay_demo"]
    replay_dataset_manifest_path = replay_root / replay_dataset_entry["path"]
    assert replay_dataset_entry["sha256"] == _sha256(replay_dataset_manifest_path)

    replay_dataset_manifest = json.loads(replay_dataset_manifest_path.read_text(encoding="utf-8"))
    resolved_name_counts = (
        replay_dataset_manifest_path.parent / replay_dataset_manifest["paths"]["name_counts_index"]
    ).resolve()
    assert resolved_name_counts == (data_root / "name_counts_index").resolve()

    public_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    replay_entry = public_manifest["replay_bundles"]["linker_replay"]
    assert replay_entry["sha256"] == _sha256(replay_manifest_path)


def test_assembly_rejects_name_count_identity_mismatch_before_copy(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first, last, first_last, last_first_initial = tiny_name_counts_tuple()
    changed_first = dict(first)
    changed_key = next(iter(changed_first))
    changed_first[changed_key] += 1
    mappings = (
        changed_first,
        dict(last),
        dict(first_last),
        dict(last_first_initial),
    )
    alternate_index, _metrics = write_name_counts_index(tmp_path / "alternate", mappings)
    inputs["name_counts_index"] = Path(alternate_index)

    with pytest.raises(ValueError, match="does not match the authoritative index"):
        assemble_source_bundle(**inputs)

    assert not inputs["output_source_bundle"].exists()
    assert not inputs["output_data_root"].exists()


def test_assembly_rejects_reserved_support_root_collisions(tmp_path: Path) -> None:
    for case_id, reserved_name, is_directory in (
        ("datasets-directory", "datasets", True),
        ("manifest-file", "manifest.json", False),
        ("index-directory", "name_counts_index", True),
    ):
        inputs = _inputs(tmp_path / case_id)
        collision = inputs["source_root"] / reserved_name
        if is_directory:
            collision.mkdir()
        else:
            collision.write_text("{}\n", encoding="utf-8")

        with pytest.raises(ValueError, match="reserved assembled paths"):
            assemble_source_bundle(**inputs)

        assert not inputs["output_source_bundle"].exists(), case_id
        assert not inputs["output_data_root"].exists(), case_id


def test_assembly_rejects_output_path_overlaps_before_copy(tmp_path: Path) -> None:
    for case_id, configure in (
        (
            "inside-input",
            lambda inputs: inputs.__setitem__("output_source_bundle", inputs["source_root"] / "assembled"),
        ),
        (
            "outputs-overlap",
            lambda inputs: inputs.__setitem__(
                "output_data_root",
                inputs["output_source_bundle"] / "data",
            ),
        ),
    ):
        inputs = _inputs(tmp_path / case_id)
        configure(inputs)

        with pytest.raises(ValueError, match="output"):
            assemble_source_bundle(**inputs)

        assert not inputs["output_source_bundle"].exists(), case_id
        assert not inputs["output_data_root"].exists(), case_id
