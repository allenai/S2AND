"""Bounded-memory name-count index writer tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from s2and.incremental_linking import feature_block_arrow
from tests.helpers import patch_name_counts_artifact


def _read_index(path: Path) -> dict[str, float]:
    header_struct = feature_block_arrow._NAME_COUNTS_INDEX_HEADER_STRUCT
    record_struct = feature_block_arrow._NAME_COUNTS_INDEX_RECORD_STRUCT
    payload = path.read_bytes()
    magic, record_count, blob_offset, blob_size = header_struct.unpack_from(payload)
    assert magic == feature_block_arrow._NAME_COUNTS_INDEX_MAGIC
    blob = payload[blob_offset : blob_offset + blob_size]
    observed: dict[str, float] = {}
    for index in range(record_count):
        offset = header_struct.size + index * record_struct.size
        _hash_1, _hash_2, name_offset, name_length, reserved, count = record_struct.unpack_from(payload, offset)
        assert reserved == 0
        observed[blob[name_offset : name_offset + name_length].decode("utf-8")] = count
    return observed


def test_bounded_sort_runs_preserve_exact_binary_lookup_values(tmp_path: Path) -> None:
    mapping = {f"name {index}": float(index + 2) for index in range(11)}
    output_path = tmp_path / "first.bin"
    metrics = feature_block_arrow._write_name_count_index_file(
        output_path,
        "first",
        mapping,
        max_records_in_memory=3,
    )

    assert _read_index(output_path) == mapping
    assert metrics["sort_run_count"] == 4
    assert metrics["peak_buffered_records"] == 3
    assert metrics["record_count"] == len(mapping)
    assert metrics["temporary_byte_count"] > metrics["byte_count"]
    assert list(tmp_path.glob(".first.bin.*")) == []


def test_disk_preflight_fails_before_creating_index_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {"a": 2.0, "b": 3.0}
    patch_name_counts_artifact(monkeypatch, (mapping, {}, {}, {}))
    monkeypatch.setattr(
        feature_block_arrow.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="insufficient free disk"):
        feature_block_arrow.write_name_counts_index(tmp_path)

    index_dir = tmp_path / "name_counts_index"
    assert index_dir.is_dir()
    assert list(index_dir.iterdir()) == []


def test_name_count_fingerprint_is_order_independent_and_content_sensitive() -> None:
    forward = {"first": {"ada": 2, "grace": 3}}
    reverse = {"first": {"grace": 3, "ada": 2}}
    changed = {"first": {"grace": 4, "ada": 2}}
    assert feature_block_arrow._name_counts_arrow_fingerprint(forward) == (
        feature_block_arrow._name_counts_arrow_fingerprint(reverse)
    )
    assert feature_block_arrow._name_counts_arrow_fingerprint(forward) != (
        feature_block_arrow._name_counts_arrow_fingerprint(changed)
    )


@pytest.mark.parametrize(
    ("kind", "name"),
    (("first", "ada"), ("last", "李"), ("first_last", "anne marie o connor")),
)
def test_reused_kind_hash_seed_preserves_binary_index_hashes(kind: str, name: str) -> None:
    name_bytes = name.encode("utf-8")
    expected = (
        feature_block_arrow._fnv64_bytes(name_bytes),
        feature_block_arrow._fnv64_bytes(
            feature_block_arrow._NAME_COUNTS_INDEX_HASH_DOMAIN + kind.encode("utf-8") + b"\x00" + name_bytes
        ),
    )

    assert feature_block_arrow._name_counts_index_hashes(kind, name_bytes) == expected
