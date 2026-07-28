from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from s2and.arrow_inputs import MissingArrowArtifactError, require_name_counts_index_artifact
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from s2and.name_counts_manifest import ValidatedNameCountsManifest
from tests.helpers import import_s2and_rust, tiny_name_counts_tuple

HAS_RUST, RUST_MODULE = import_s2and_rust()

_REQUIRED_FILE_KEYS = ("first", "last", "first_last", "last_first_initial")
_REQUIRED_FILE_ENTRY_FIELDS = ("path", "byte_count", "sha256")
_REQUIRED_MANIFEST_FIELDS = ("schema_version", "normalization_version", "files")


def _write_index(root: Path) -> Path:
    path, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
    )
    return Path(path)


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_native_validation_rejects(index_dir: Path, expected_field: str) -> None:
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        ValidatedNameCountsManifest.load(index_dir)


def _remove_manifest_field(index_dir: Path, field: str) -> None:
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)


def test_valid_manifest_has_identical_python_and_rust_identity(tmp_path: Path) -> None:
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    index_dir = _write_index(tmp_path)

    python_manifest = ValidatedNameCountsManifest.load(index_dir)
    rust_index = RUST_MODULE.NameCountsIndex.open(str(index_dir))
    manifest_sha256 = hashlib.sha256((index_dir / "manifest.json").read_bytes()).hexdigest()

    assert set(_read_manifest(index_dir)) == set(_REQUIRED_MANIFEST_FIELDS)
    assert python_manifest.normalization_version == rust_index.normalization_version
    assert python_manifest.manifest_sha256 == manifest_sha256
    assert rust_index.name_counts_manifest_sha256 == manifest_sha256
    for file_key, file_entry in _read_manifest(index_dir)["files"].items():
        retained_file = python_manifest.files[file_key]
        assert os.path.samefile(retained_file.path, index_dir / file_entry["path"])
        assert retained_file.byte_count == file_entry["byte_count"]
        assert retained_file.sha256 == file_entry["sha256"]
    assert not hasattr(rust_index, "lookup_many")
    assert callable(rust_index._lookup_many_unique)


def test_required_manifest_fields_are_rejected(tmp_path: Path) -> None:
    for field in _REQUIRED_MANIFEST_FIELDS:
        index_dir = _write_index(tmp_path / field)
        _remove_manifest_field(index_dir, field)
        _assert_native_validation_rejects(index_dir, field)


@pytest.mark.parametrize("schema_version", ("name_counts_index_v1", "name_counts_index_v2"))
def test_previous_schema_versions_are_rejected(tmp_path: Path, schema_version: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["schema_version"] = schema_version
    _write_manifest(index_dir, manifest)

    _assert_native_validation_rejects(index_dir, "schema_version")


def test_parent_directory_is_not_a_compatible_index_path(tmp_path: Path) -> None:
    _write_index(tmp_path)

    _assert_native_validation_rejects(tmp_path, "does not contain manifest.json")


def test_required_file_entries_are_rejected(tmp_path: Path) -> None:
    for file_key in _REQUIRED_FILE_KEYS:
        index_dir = _write_index(tmp_path / file_key)
        manifest = _read_manifest(index_dir)
        manifest["files"].pop(file_key)
        _write_manifest(index_dir, manifest)
        _assert_native_validation_rejects(index_dir, file_key)


def test_required_file_entry_fields_are_rejected(tmp_path: Path) -> None:
    for field in _REQUIRED_FILE_ENTRY_FIELDS:
        index_dir = _write_index(tmp_path / field)
        manifest = _read_manifest(index_dir)
        manifest["files"]["first"].pop(field)
        _write_manifest(index_dir, manifest)
        _assert_native_validation_rejects(index_dir, field)


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    (
        ("root", "source_provenance"),
        ("files", "middle"),
        ("file_entry", "record_count"),
    ),
)
def test_unknown_manifest_fields_are_rejected(tmp_path: Path, mutation: str, expected_field: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    if mutation == "root":
        manifest["source_provenance"] = {}
    elif mutation == "files":
        manifest["files"]["middle"] = manifest["files"]["first"]
    elif mutation == "file_entry":
        manifest["files"]["first"]["record_count"] = 0
    else:  # pragma: no cover - parametrization invariant
        raise AssertionError(f"unknown mutation {mutation}")
    _write_manifest(index_dir, manifest)

    _assert_native_validation_rejects(index_dir, expected_field)


def test_material_contract_is_identical_at_python_and_rust_boundaries(tmp_path: Path) -> None:
    mutations = (
        ("byte_count_mismatch", "byte_count"),
        ("sha256_mismatch", "SHA-256"),
        ("generation_path", "must equal first.bin"),
        ("wrong_filename", "must equal first.bin"),
    )
    for mutation, expected_field in mutations:
        index_dir = _write_index(tmp_path / mutation / "index")
        manifest = _read_manifest(index_dir)
        if mutation == "byte_count_mismatch":
            manifest["files"]["first"]["byte_count"] += 1
        elif mutation == "sha256_mismatch":
            manifest["files"]["first"]["sha256"] = "f" * 64
        elif mutation == "generation_path":
            manifest["files"]["first"]["path"] = "generations/legacy/first.bin"
        elif mutation == "wrong_filename":
            manifest["files"]["first"]["path"] = "last.bin"
        else:  # pragma: no cover - mutation invariant
            raise AssertionError(f"unknown mutation {mutation}")
        _write_manifest(index_dir, manifest)
        _assert_native_validation_rejects(index_dir, expected_field)


def test_arrow_boundary_translates_representative_manifest_failure(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    _remove_manifest_field(index_dir, "normalization_version")

    with pytest.raises(MissingArrowArtifactError, match="normalization_version"):
        require_name_counts_index_artifact(
            index_dir,
            context="test Arrow boundary",
            producer_hint="write a complete test index",
        )


def test_python_index_translates_representative_native_failure(tmp_path: Path) -> None:
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    index_dir = _write_index(tmp_path)
    _remove_manifest_field(index_dir, "normalization_version")

    with pytest.raises((OSError, RuntimeError, ValueError), match="normalization_version"):
        NameCountsIndex.open(index_dir)
