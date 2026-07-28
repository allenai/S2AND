"""Outcome tests for shared native name-count index handles."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import tiny_name_counts_tuple


def _write_index(root: Path) -> Path:
    path, _metrics = write_name_counts_index(root, tiny_name_counts_tuple())
    return Path(path)


def test_open_reuses_one_exact_manifest_identity(tmp_path: Path) -> None:
    path = _write_index(tmp_path)

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second


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
    first_path = path / "first.bin"
    first_path.write_bytes(first_path.read_bytes() + b"corrupt")

    with pytest.raises((OSError, RuntimeError, ValueError), match="byte_count|SHA-256"):
        NameCountsIndex.open(path)
