from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from s2and.arrow_inputs import build_arrow_artifact_manifest, require_name_counts_index_artifact
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from scripts.verification.validate_local_arrow_release import validate_release_root
from tests.helpers import tiny_name_counts_tuple


def _touch_json(path: Path, payload: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _validate_required_release_files(release_root: Path, dataset_name: str) -> None:
    pa = pytest.importorskip("pyarrow")
    root_manifest = json.loads((release_root / "manifest.json").read_text(encoding="utf-8"))
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    manifest_paths = dataset_manifest.get("paths", {}) if isinstance(dataset_manifest, dict) else {}
    embedding_path = manifest_paths.get("specter", "specter.arrow")
    dataset_entries = {
        entry["dataset"]: entry for entry in root_manifest.get("dataset_manifests", []) if isinstance(entry, dict)
    }
    dataset_entry = dataset_entries[dataset_name]
    assert dataset_entry["manifest_sha256"] == hashlib.sha256(dataset_manifest_path.read_bytes()).hexdigest()
    assert dataset_entry["manifest_size_bytes"] == dataset_manifest_path.stat().st_size
    assert root_manifest["audit"]["dataset_count"] == 1
    assert root_manifest["audit"]["total_signature_count"] == dataset_manifest["signature_count"]
    required_paths = [
        release_root / "manifest.json",
        release_root / "LICENSE.txt",
        release_root / "name_counts_index" / "manifest.json",
        release_root / dataset_name / "manifest.json",
        release_root / dataset_name / "signatures.arrow",
        release_root / dataset_name / "papers.arrow",
        release_root / dataset_name / "paper_authors.arrow",
        release_root / dataset_name / str(embedding_path),
        release_root / dataset_name / "signatures.signatures_batch_index.bin",
    ]

    missing_paths = [path.relative_to(release_root) for path in required_paths if not path.exists()]
    assert missing_paths == []
    require_name_counts_index_artifact(
        release_root / "name_counts_index",
        context="release layout test",
        producer_hint="test fixture must include complete name_counts_index",
    )
    for key in ("signatures", "papers", "paper_authors", "specter"):
        path = release_root / dataset_name / manifest_paths[key]
        with pa.memory_map(str(path), "r") as source:
            assert pa.ipc.open_file(source).read_all().num_rows >= 1


def _write_root_manifest(release_root: Path, dataset_name: str, *, replay_bundles: list[dict] | None = None) -> None:
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest_bytes = dataset_manifest_path.read_bytes()
    root_manifest = {
        "schema": "inference_arrow_bundle_v1",
        "datasets": [dataset_name],
        "dataset_manifests": [
            {
                "dataset": dataset_name,
                "dataset_dir": dataset_name,
                "manifest_path": f"{dataset_name}/manifest.json",
                "manifest_size_bytes": len(dataset_manifest_bytes),
                "manifest_sha256": hashlib.sha256(dataset_manifest_bytes).hexdigest(),
                "validation_requirements": {
                    "require_embeddings": True,
                    "require_name_counts_index": True,
                },
            }
        ],
        "audit": {
            "dataset_count": 1,
            "total_signature_count": 1,
        },
    }
    if replay_bundles is not None:
        root_manifest["replay_bundles"] = replay_bundles
    _touch_json(release_root / "manifest.json", root_manifest)


def _build_arrow_release_fixture(tmp_path: Path, dataset_name: str = "s2and_mini") -> tuple[Path, str]:
    pa = pytest.importorskip("pyarrow")
    release_root = tmp_path / "release"

    write_name_counts_index(
        release_root,
        tiny_name_counts_tuple(),
    )

    for file_path in (release_root / "LICENSE.txt",):
        _touch_file(file_path)
    dataset_root = release_root / dataset_name
    write_arrow_ipc_table(
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
        dataset_root / "signatures.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "title": pa.array(["Notes"], type=pa.string()),
                "venue": pa.array([""], type=pa.string()),
                "journal_name": pa.array([""], type=pa.string()),
            }
        ),
        dataset_root / "papers.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Ada Lovelace"], type=pa.string()),
            }
        ),
        dataset_root / "paper_authors.arrow",
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        dataset_root / "specter2.arrow",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "signatures.arrow",
        dataset_root / "signatures.signatures_batch_index.bin",
        key_column="signature_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "papers.arrow",
        dataset_root / "papers.papers_batch_index.bin",
        key_column="paper_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "paper_authors.arrow",
        dataset_root / "paper_authors.paper_authors_batch_index.bin",
        key_column="paper_id",
    )
    write_arrow_batch_lookup_index(
        dataset_root / "specter2.arrow",
        dataset_root / "specter2.specter_batch_index.bin",
        key_column="paper_id",
    )
    generation_paths = {
        "signatures": str(dataset_root / "signatures.arrow"),
        "papers": str(dataset_root / "papers.arrow"),
        "paper_authors": str(dataset_root / "paper_authors.arrow"),
        "specter": str(dataset_root / "specter2.arrow"),
        "name_counts_index": str(release_root / "name_counts_index"),
        "signatures_batch_index": str(dataset_root / "signatures.signatures_batch_index.bin"),
        "papers_batch_index": str(dataset_root / "papers.papers_batch_index.bin"),
        "paper_authors_batch_index": str(dataset_root / "paper_authors.paper_authors_batch_index.bin"),
        "specter_batch_index": str(dataset_root / "specter2.specter_batch_index.bin"),
    }
    dataset_manifest = build_arrow_artifact_manifest(
        generation_paths,
        dataset_root,
        metadata={"signature_count": 1, "paper_count": 1},
    )
    _touch_json(dataset_root / "manifest.json", dataset_manifest)
    _write_root_manifest(release_root, dataset_name)
    return release_root, dataset_name


def _rewrite_dataset_manifest_paths(release_root: Path, dataset_name: str, paths: dict[str, str]) -> None:
    dataset_manifest_path = release_root / dataset_name / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    metadata = {
        key: value
        for key, value in dataset_manifest.items()
        if key not in {"normalization_version", "paths", "artifact_generation"}
    }
    dataset_root = dataset_manifest_path.parent
    resolved_paths = {
        key: str((dataset_root / value).resolve()) if not Path(value).is_absolute() else value
        for key, value in paths.items()
    }
    _touch_json(
        dataset_manifest_path,
        build_arrow_artifact_manifest(resolved_paths, dataset_root, metadata=metadata),
    )
    _write_root_manifest(release_root, dataset_name)


def test_arrow_release_layout_required_files(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)

    _validate_required_release_files(release_root, dataset_name)
    assert validate_release_root(release_root, include_replay_bundles=False) == {
        "release_root": str(release_root.resolve()),
        "dataset_manifest_count": 1,
        "replay_dataset_manifest_count": 0,
        "name_counts_index": str(release_root.resolve() / "name_counts_index"),
        "default_model_manifest": None,
        "network_access": False,
    }


def test_validate_release_root_reports_dataset_manifest_checksum_mismatch(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    root_manifest_path = release_root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["dataset_manifests"][0]["manifest_sha256"] = "0" * 64
    _touch_json(root_manifest_path, root_manifest)

    with pytest.raises(ValueError, match=f"root dataset {dataset_name} manifest_sha256 mismatch"):
        validate_release_root(release_root, include_replay_bundles=False)


def test_validate_release_root_rejects_default_model_outside_release(tmp_path: Path) -> None:
    release_root, _dataset_name = _build_arrow_release_fixture(tmp_path)
    _touch_json(
        release_root / "default_production_model.json",
        {"bundle_dir": "../outside", "bundle_version": "test"},
    )
    with pytest.raises(ValueError, match="outside release root"):
        validate_release_root(release_root, include_replay_bundles=False)


def test_validate_release_root_reports_missing_required_dataset_file(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    (release_root / dataset_name / "signatures.arrow").unlink()

    with pytest.raises(ValueError, match=r"root dataset s2and_mini paths\.signatures is missing"):
        validate_release_root(release_root, include_replay_bundles=False)


def test_validate_release_root_reports_missing_batch_index_path(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    dataset_manifest = json.loads((release_root / dataset_name / "manifest.json").read_text(encoding="utf-8"))
    paths = dataset_manifest["paths"]
    del paths["papers_batch_index"]
    _rewrite_dataset_manifest_paths(release_root, dataset_name, paths)

    with pytest.raises(
        ValueError,
        match=r"missing required Arrow artifacts.*papers_batch_index",
    ):
        validate_release_root(release_root, include_replay_bundles=False)


def test_validate_release_root_reports_missing_canonical_specter_batch_index(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    dataset_manifest = json.loads((release_root / dataset_name / "manifest.json").read_text(encoding="utf-8"))
    paths = dataset_manifest["paths"]
    del paths["specter_batch_index"]
    _rewrite_dataset_manifest_paths(release_root, dataset_name, paths)

    with pytest.raises(
        ValueError,
        match=r"must contain both specter and specter_batch_index",
    ):
        validate_release_root(release_root, include_replay_bundles=False)


def test_validate_release_root_reports_missing_replay_manifest_when_included(tmp_path: Path) -> None:
    release_root, dataset_name = _build_arrow_release_fixture(tmp_path)
    _write_root_manifest(
        release_root,
        dataset_name,
        replay_bundles=[{"bundle": "mini-replay", "manifest_path": "replay/manifest.json"}],
    )

    with pytest.raises(ValueError, match=r"replay bundle 0 manifest is missing: .*replay.*manifest\.json"):
        validate_release_root(release_root, include_replay_bundles=True)
