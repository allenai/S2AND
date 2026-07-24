"""Outcome tests for native name-count index identity and provenance."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path
from typing import cast

import pytest

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _write_index(root: Path, *, generation_id: str = "generation-a") -> Path:
    provenance = tiny_name_counts_provenance()
    provenance["generation_id"] = generation_id
    path, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
        provenance,
    )
    return Path(path)


def test_open_reuses_one_exact_manifest_generation(tmp_path: Path) -> None:
    path = _write_index(tmp_path)

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second
    assert first.manifest_sha256 == first.source_provenance["manifest_sha256"]


def test_open_cache_retains_only_four_paths(tmp_path: Path) -> None:
    references: list[weakref.ReferenceType[NameCountsIndex]] = []
    for index in range(5):
        path = _write_index(tmp_path / str(index))
        opened = NameCountsIndex.open(path)
        references.append(weakref.ref(opened))
    del opened
    gc.collect()

    assert references[0]() is None
    assert all(reference() is not None for reference in references[1:])


def test_corrupt_published_material_is_rejected_on_open(tmp_path: Path) -> None:
    path = _write_index(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    first_path = path / manifest["files"]["first"]["path"]
    first_path.write_bytes(first_path.read_bytes() + b"corrupt")

    with pytest.raises((OSError, RuntimeError, ValueError), match="byte_count|SHA-256"):
        NameCountsIndex.open(path)


def test_runtime_provenance_is_read_only_and_bound_to_manifest(tmp_path: Path) -> None:
    index = NameCountsIndex.open(_write_index(tmp_path))

    assert index.source_provenance["manifest_sha256"] == index.manifest_sha256
    with pytest.raises(TypeError):
        cast(dict[str, object], index.source_provenance)["generation_id"] = "changed"


def test_writer_rejects_incomplete_audit_provenance(tmp_path: Path) -> None:
    provenance = tiny_name_counts_provenance()
    provenance.pop("source_query_sha256")

    with pytest.raises(ValueError, match="source_query_sha256"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple(), provenance)
