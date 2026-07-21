from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from s2and.arrow_inputs import MissingArrowArtifactError, require_name_counts_index_artifact
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_count_binding import NameCountsBinding
from s2and.name_counts_index import NameCountsIndex
from s2and.name_counts_manifest import ValidatedNameCountsManifest
from tests.helpers import import_s2and_rust, tiny_name_counts_provenance, tiny_name_counts_tuple

HAS_RUST, RUST_MODULE = import_s2and_rust()

_REQUIRED_PROVENANCE_FIELDS = (
    "schema_version",
    "normalization_version",
    "generation_id",
    "pickle_sha256",
    "source_snapshot_id",
    "source_kind",
    "source_query_sha256",
    "selected_rows_sha256",
    "selected_row_count",
    "source_row_count",
)
_REQUIRED_FILE_KEYS = ("first", "last", "first_last", "last_first_initial")
_REQUIRED_FILE_ENTRY_FIELDS = ("path", "byte_count", "sha256")
_REQUIRED_MANIFEST_FIELDS = ("schema_version", "normalization_version", "source_provenance", "files")


def _write_index(root: Path) -> Path:
    path, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    return Path(path)


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_authorities_reject(index_dir: Path, expected_field: str) -> None:
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        ValidatedNameCountsManifest.load(index_dir, context="test Python parser")
    if HAS_RUST:
        with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
            RUST_MODULE.NameCountsIndex.open(str(index_dir))


def _remove_manifest_field(index_dir: Path, field: str) -> None:
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)


def test_valid_manifest_has_identical_python_and_rust_binding(tmp_path: Path) -> None:
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    index_dir = _write_index(tmp_path)

    python_manifest = ValidatedNameCountsManifest.load(index_dir, context="test valid manifest")
    python_binding = NameCountsBinding.from_arrow_name_counts_index(index_dir, context="test valid binding")
    rust_index = RUST_MODULE.NameCountsIndex.open(str(index_dir))

    assert python_manifest.normalization_version == rust_index.normalization_version
    assert rust_index.manifest_sha256 == hashlib.sha256((index_dir / "manifest.json").read_bytes()).hexdigest()
    assert not hasattr(rust_index, "lookup_many")
    assert callable(rust_index._lookup_many_unique)
    assert rust_index.name_counts_provenance_binding == (
        python_binding.generation_id,
        python_binding.pickle_sha256,
        python_binding.source_snapshot_id,
        python_binding.selected_rows_sha256,
    )


def test_required_manifest_fields_are_rejected_by_python_and_rust(tmp_path: Path) -> None:
    for field in _REQUIRED_MANIFEST_FIELDS:
        index_dir = _write_index(tmp_path / field)
        _remove_manifest_field(index_dir, field)
        _assert_authorities_reject(index_dir, field)


def test_required_provenance_fields_are_rejected_by_python_and_rust(tmp_path: Path) -> None:
    for field in _REQUIRED_PROVENANCE_FIELDS:
        index_dir = _write_index(tmp_path / field)
        manifest = _read_manifest(index_dir)
        manifest["source_provenance"].pop(field)
        _write_manifest(index_dir, manifest)
        _assert_authorities_reject(index_dir, field)


def test_required_file_entries_are_rejected_by_python_and_rust(tmp_path: Path) -> None:
    for file_key in _REQUIRED_FILE_KEYS:
        index_dir = _write_index(tmp_path / file_key)
        manifest = _read_manifest(index_dir)
        manifest["files"].pop(file_key)
        _write_manifest(index_dir, manifest)
        _assert_authorities_reject(index_dir, file_key)


def test_required_file_entry_fields_are_rejected_by_python_and_rust(tmp_path: Path) -> None:
    for field in _REQUIRED_FILE_ENTRY_FIELDS:
        index_dir = _write_index(tmp_path / field)
        manifest = _read_manifest(index_dir)
        manifest["files"]["first"].pop(field)
        _write_manifest(index_dir, manifest)
        _assert_authorities_reject(index_dir, field)


def test_material_contract_is_identical_at_python_and_rust_boundaries(tmp_path: Path) -> None:
    mutations = (
        ("byte_count_mismatch", "byte_count"),
        ("sha256_mismatch", "SHA-256"),
        ("path_escape", "escapes"),
        ("missing_published_marker", "published marker"),
    )
    for mutation, expected_field in mutations:
        index_dir = _write_index(tmp_path / mutation / "index")
        manifest = _read_manifest(index_dir)
        if mutation == "byte_count_mismatch":
            manifest["files"]["first"]["byte_count"] += 1
        elif mutation == "sha256_mismatch":
            manifest["files"]["first"]["sha256"] = "f" * 64
        elif mutation == "path_escape":
            outside_dir = _write_index(tmp_path / mutation / "outside")
            outside_manifest = _read_manifest(outside_dir)
            manifest["files"]["first"]["path"] = str(
                (outside_dir / outside_manifest["files"]["first"]["path"]).resolve()
            )
        elif mutation == "missing_published_marker":
            first_path = index_dir / manifest["files"]["first"]["path"]
            (first_path.parent / ".published").unlink()
        else:  # pragma: no cover - mutation invariant
            raise AssertionError(f"unknown mutation {mutation}")
        _write_manifest(index_dir, manifest)
        _assert_authorities_reject(index_dir, expected_field)


def test_python_binding_translates_representative_manifest_failure(tmp_path: Path) -> None:
    index_dir = _write_index(tmp_path)
    _remove_manifest_field(index_dir, "normalization_version")

    with pytest.raises(ValueError, match="normalization_version"):
        NameCountsBinding.from_arrow_name_counts_index(index_dir, context="test Python binding")


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
