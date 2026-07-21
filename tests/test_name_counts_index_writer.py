"""Bounded-memory name-count index writer tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import s2and.name_counts_manifest as name_counts_manifest_module
from s2and.incremental_linking import feature_block_arrow
from tests.helpers import tiny_name_counts_provenance


class _DuplicateNameMapping(Mapping[str, float]):
    """Deliberately violate Mapping uniqueness to exercise streaming validation."""

    def __getitem__(self, key: str) -> float:
        if key != "ada":
            raise KeyError(key)
        return 2.0

    def __iter__(self) -> Iterator[str]:
        return iter(("ada", "ada"))

    def __len__(self) -> int:
        return 2


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


def test_writer_rejects_duplicate_logical_utf8_name_without_unbounded_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate UTF-8 name 'ada'"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            (_DuplicateNameMapping(), {}, {}, {}),
            tiny_name_counts_provenance(),
        )


def test_writer_rejects_keys_that_would_collide_after_stringification(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="name-count first keys must be strings"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({1: 2.0, "1": 3.0}, {}, {}, {}),
            tiny_name_counts_provenance(),
        )


@pytest.mark.parametrize("count", [float("nan"), float("inf"), float("-inf"), 0.0, -3.0])
def test_writer_rejects_nonfinite_and_nonpositive_counts(tmp_path: Path, count: float) -> None:
    with pytest.raises(ValueError, match="must be a finite positive number"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({"ada": count}, {}, {}, {}),
            tiny_name_counts_provenance(),
        )


def test_disk_preflight_fails_before_creating_index_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {"a": 2.0, "b": 3.0}
    monkeypatch.setattr(
        feature_block_arrow.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="insufficient free disk"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            (mapping, {}, {}, {}),
            tiny_name_counts_provenance(),
        )

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


def test_writer_never_hashes_name_count_material_under_publish_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tiny_name_counts_provenance()
    feature_block_arrow.write_name_counts_index(tmp_path, ({"ada": 2.0}, {}, {}, {}), provenance)
    real_sha256_file = feature_block_arrow._sha256_file
    real_manifest_sha256_file = name_counts_manifest_module._sha256_file
    lock_active = False

    def checked_sha256_file(path: Path) -> str:
        assert not lock_active, f"material hash ran under publication lock: {path}"
        return real_sha256_file(path)

    def checked_manifest_sha256_file(path: Path) -> str:
        assert not lock_active, f"manifest material hash ran under publication lock: {path}"
        return real_manifest_sha256_file(path)

    @contextmanager
    def observed_publish_lock(_index_dir: Path):
        nonlocal lock_active
        assert not lock_active
        lock_active = True
        try:
            yield
        finally:
            lock_active = False

    monkeypatch.setattr(feature_block_arrow, "_sha256_file", checked_sha256_file)
    monkeypatch.setattr(name_counts_manifest_module, "_sha256_file", checked_manifest_sha256_file)
    monkeypatch.setattr(feature_block_arrow, "_exclusive_name_counts_publish_lock", observed_publish_lock)

    _path, metrics = feature_block_arrow.write_name_counts_index(
        tmp_path,
        ({"ada": 3.0}, {}, {}, {}),
        {**provenance, "generation_id": "generation-b"},
    )

    assert metrics["reused"] is False


def test_cleanup_identifies_current_generation_without_material_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tiny_name_counts_provenance()
    index_path, _metrics = feature_block_arrow.write_name_counts_index(
        tmp_path,
        ({"ada": 2.0}, {}, {}, {}),
        provenance,
    )
    feature_block_arrow.write_name_counts_index(
        tmp_path,
        ({"ada": 3.0}, {}, {}, {}),
        {**provenance, "generation_id": "generation-b"},
        overwrite=True,
    )

    def unexpected_material_hash(_path: Path) -> str:
        pytest.fail("cleanup needs manifest reachability, not material hashing")

    monkeypatch.setattr("s2and.name_counts_manifest._sha256_file", unexpected_material_hash)

    assert feature_block_arrow.cleanup_stale_name_counts_generations(index_path) == {"removed_generation_count": 1}


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
