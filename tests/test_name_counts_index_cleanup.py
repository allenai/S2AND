"""Atomic write-once publication tests for the name-count index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from s2and.incremental_linking import feature_block_arrow
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _temporary_indexes(root: Path) -> list[Path]:
    return list(root.glob(".name_counts_index.*"))


def test_existing_target_is_never_reused_or_replaced(tmp_path: Path) -> None:
    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
    )
    manifest_path = Path(index_path) / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="target already exists"):
        write_name_counts_index(
            tmp_path,
            tiny_name_counts_tuple(),
            {**tiny_name_counts_provenance(), "generation_id": "replacement"},
        )

    assert manifest_path.read_bytes() == original_manifest
    assert _temporary_indexes(tmp_path) == []


def test_all_material_is_built_before_target_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_writer = feature_block_arrow._write_name_count_index_file
    written_kinds: list[str] = []

    def observe_write(path: Path, kind: str, mapping: Any, **kwargs: Any) -> dict[str, int]:
        assert not (tmp_path / "name_counts_index").exists()
        assert path.is_relative_to(tmp_path)
        assert path.parts[-4].startswith(".name_counts_index.")
        written_kinds.append(kind)
        return real_writer(path, kind, mapping, **kwargs)

    monkeypatch.setattr(feature_block_arrow, "_write_name_count_index_file", observe_write)

    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
    )

    assert written_kinds == ["first", "last", "first_last", "last_first_initial"]
    assert Path(index_path).is_dir()
    assert _temporary_indexes(tmp_path) == []


def test_failed_material_write_leaves_target_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_writer = feature_block_arrow._write_name_count_index_file

    def fail_on_last(path: Path, kind: str, mapping: Any, **kwargs: Any) -> dict[str, int]:
        if kind == "last":
            raise OSError("injected material failure")
        return real_writer(path, kind, mapping, **kwargs)

    monkeypatch.setattr(feature_block_arrow, "_write_name_count_index_file", fail_on_last)

    with pytest.raises(OSError, match="injected material failure"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple(), tiny_name_counts_provenance())

    assert not (tmp_path / "name_counts_index").exists()
    assert _temporary_indexes(tmp_path) == []


def test_failed_final_rename_leaves_target_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_rename = Path.rename

    def fail_publication(path: Path, target: str | Path) -> Path:
        if Path(target) == tmp_path / "name_counts_index":
            raise OSError("injected publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)

    with pytest.raises(OSError, match="injected publication failure"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple(), tiny_name_counts_provenance())

    assert not (tmp_path / "name_counts_index").exists()
    assert _temporary_indexes(tmp_path) == []
