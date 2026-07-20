from __future__ import annotations

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


def _assert_all_boundaries_reject(index_dir: Path, expected_field: str) -> None:
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        ValidatedNameCountsManifest.load(index_dir, context="test Python parser")
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        NameCountsBinding.from_arrow_name_counts_index(index_dir, context="test Python binding")
    with pytest.raises(MissingArrowArtifactError, match=expected_field):
        require_name_counts_index_artifact(
            index_dir,
            context="test Arrow boundary",
            producer_hint="write a complete test index",
        )
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        NameCountsIndex.open(index_dir)
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        RUST_MODULE.NameCountsIndex.open(str(index_dir))


def test_valid_manifest_has_identical_python_and_rust_binding(tmp_path: Path) -> None:
    if not HAS_RUST:
        pytest.skip(f"Rust extension unavailable: {RUST_MODULE!r}")
    index_dir = _write_index(tmp_path)

    python_manifest = ValidatedNameCountsManifest.load(index_dir, context="test valid manifest")
    python_binding = NameCountsBinding.from_arrow_name_counts_index(index_dir, context="test valid binding")
    rust_index = RUST_MODULE.NameCountsIndex.open(str(index_dir))

    assert python_manifest.normalization_version == rust_index.normalization_version
    assert not hasattr(rust_index, "lookup_many")
    assert callable(rust_index._lookup_many_unique)
    assert rust_index.name_counts_provenance_binding == (
        python_binding.generation_id,
        python_binding.pickle_sha256,
        python_binding.source_snapshot_id,
        python_binding.selected_rows_sha256,
    )


@pytest.mark.parametrize("field", _REQUIRED_MANIFEST_FIELDS)
def test_required_manifest_fields_are_rejected_at_every_boundary(tmp_path: Path, field: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)

    _assert_all_boundaries_reject(index_dir, field)


@pytest.mark.parametrize("field", _REQUIRED_PROVENANCE_FIELDS)
def test_required_provenance_fields_are_rejected_at_every_boundary(tmp_path: Path, field: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["source_provenance"].pop(field)
    _write_manifest(index_dir, manifest)

    _assert_all_boundaries_reject(index_dir, field)


@pytest.mark.parametrize("file_key", _REQUIRED_FILE_KEYS)
def test_required_file_entries_are_rejected_at_every_boundary(tmp_path: Path, file_key: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["files"].pop(file_key)
    _write_manifest(index_dir, manifest)

    _assert_all_boundaries_reject(index_dir, file_key)


@pytest.mark.parametrize("field", _REQUIRED_FILE_ENTRY_FIELDS)
def test_required_file_entry_fields_are_rejected_at_every_boundary(tmp_path: Path, field: str) -> None:
    index_dir = _write_index(tmp_path)
    manifest = _read_manifest(index_dir)
    manifest["files"]["first"].pop(field)
    _write_manifest(index_dir, manifest)

    _assert_all_boundaries_reject(index_dir, field)


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    (
        ("byte_count_mismatch", "byte_count"),
        ("sha256_mismatch", "SHA-256"),
        ("path_escape", "escapes"),
        ("missing_published_marker", "published marker"),
    ),
)
def test_material_contract_is_identical_at_python_and_rust_boundaries(
    tmp_path: Path,
    mutation: str,
    expected_field: str,
) -> None:
    index_dir = _write_index(tmp_path / "index")
    manifest = _read_manifest(index_dir)
    if mutation == "byte_count_mismatch":
        manifest["files"]["first"]["byte_count"] += 1
    elif mutation == "sha256_mismatch":
        manifest["files"]["first"]["sha256"] = "f" * 64
    elif mutation == "path_escape":
        outside_dir = _write_index(tmp_path / "outside")
        outside_manifest = _read_manifest(outside_dir)
        manifest["files"]["first"]["path"] = str((outside_dir / outside_manifest["files"]["first"]["path"]).resolve())
    elif mutation == "missing_published_marker":
        first_path = index_dir / manifest["files"]["first"]["path"]
        (first_path.parent / ".published").unlink()
    else:  # pragma: no cover - parameter invariant
        raise AssertionError(f"unknown mutation {mutation}")
    _write_manifest(index_dir, manifest)

    _assert_all_boundaries_reject(index_dir, expected_field)
