from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from s2and.arrow_inputs import MissingArrowArtifactError, require_name_counts_index_artifact
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import import_s2and_rust, tiny_name_counts_tuple

HAS_RUST, RUST_MODULE = import_s2and_rust()

_REQUIRED_MANIFEST_FIELDS = ("kind", "format_version", "files")


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
        NameCountsIndex.open(index_dir)


def _remove_manifest_field(index_dir: Path, field: str) -> None:
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)


def test_valid_manifest_has_identical_python_and_rust_identity(tmp_path: Path) -> None:
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    index_dir = _write_index(tmp_path)

    python_index = NameCountsIndex.open(index_dir)
    rust_index = RUST_MODULE.NameCountsIndex.open(str(index_dir))
    manifest_sha256 = hashlib.sha256((index_dir / "manifest.json").read_bytes()).hexdigest()

    assert python_index.manifest_sha256 == manifest_sha256
    assert rust_index.name_counts_manifest_sha256 == manifest_sha256
    assert Path(python_index.path) == index_dir.resolve()
    assert callable(rust_index._lookup_many_unique)


def test_native_manifest_validation_rejects_structural_identity_and_material_corruption(tmp_path: Path) -> None:
    for field in _REQUIRED_MANIFEST_FIELDS:
        index_dir = _write_index(tmp_path / f"missing-{field}")
        _remove_manifest_field(index_dir, field)
        _assert_native_validation_rejects(index_dir, field)

    cases = (
        ("wrong-kind", "kind", "other", "kind"),
        ("unsupported-format", "format_version", 2, "format_version"),
        ("string-format", "format_version", "1", "expected u32"),
    )
    for case_id, field, value, expected in cases:
        index_dir = _write_index(tmp_path / case_id)
        manifest = _read_manifest(index_dir)
        manifest[field] = value
        _write_manifest(index_dir, manifest)

        _assert_native_validation_rejects(index_dir, expected)
    parent = tmp_path / "parent"
    _write_index(parent)
    _assert_native_validation_rejects(parent, "does not contain manifest.json")

    index_dir = _write_index(tmp_path / "missing-first")
    manifest = _read_manifest(index_dir)
    manifest["files"].pop("first")
    _write_manifest(index_dir, manifest)
    _assert_native_validation_rejects(index_dir, "first")

    index_dir = _write_index(tmp_path / "missing-sha256")
    manifest = _read_manifest(index_dir)
    manifest["files"]["first"].pop("sha256")
    _write_manifest(index_dir, manifest)
    _assert_native_validation_rejects(index_dir, "sha256")

    cases = (
        ("root", "source_provenance"),
        ("files", "middle"),
        ("file-entry", "record_count"),
    )
    for mutation, expected_field in cases:
        index_dir = _write_index(tmp_path / mutation)
        manifest = _read_manifest(index_dir)
        if mutation == "root":
            manifest["source_provenance"] = {}
        elif mutation == "files":
            manifest["files"]["middle"] = manifest["files"]["first"]
        else:
            manifest["files"]["first"]["record_count"] = 0
        _write_manifest(index_dir, manifest)

        _assert_native_validation_rejects(index_dir, expected_field)
    mutations = (
        ("byte_count_mismatch", "byte_count"),
        ("sha256_mismatch", "SHA-256"),
    )
    for mutation, expected_field in mutations:
        index_dir = _write_index(tmp_path / mutation / "index")
        manifest = _read_manifest(index_dir)
        if mutation == "byte_count_mismatch":
            manifest["files"]["first"]["byte_count"] += 1
        elif mutation == "sha256_mismatch":
            manifest["files"]["first"]["sha256"] = "f" * 64
        else:  # pragma: no cover - mutation invariant
            raise AssertionError(f"unknown mutation {mutation}")
        _write_manifest(index_dir, manifest)
        _assert_native_validation_rejects(index_dir, expected_field)

    index_dir = _write_index(tmp_path / "renamed-file")
    (index_dir / "first.bin").rename(index_dir / "renamed.bin")
    _assert_native_validation_rejects(index_dir, "first.bin")

    index_dir = _write_index(tmp_path / "format-before-payload")
    manifest = _read_manifest(index_dir)
    manifest["format_version"] = 2
    _write_manifest(index_dir, manifest)
    (index_dir / "first.bin").unlink()
    _assert_native_validation_rejects(index_dir, "format_version")


def test_arrow_boundary_translates_representative_manifest_failure(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    _remove_manifest_field(index_dir, "format_version")

    with pytest.raises(MissingArrowArtifactError, match="format_version"):
        require_name_counts_index_artifact(
            index_dir,
            context="test Arrow boundary",
            producer_hint="write a complete test index",
        )
