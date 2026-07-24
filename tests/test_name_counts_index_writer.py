"""Bounded-memory name-count index writer tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

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
    names = ("ada", "bea", "cy", "dee", "eve", "fay", "gia", "hal", "ian", "jay", "kim")
    mapping = {name: float(index + 2) for index, name in enumerate(names)}
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


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("first", "Ada"),
        ("first", "a"),
        ("first", "ada!"),
        ("first", "ada  marie"),
        ("last", ""),
        ("first_last", "adasmith"),
        ("first_last", "a smith"),
        ("last_first_initial", "smith ad"),
        ("last_first_initial", "smith A"),
    ),
)
def test_writer_rejects_noncanonical_name_count_keys(tmp_path: Path, kind: str, name: str) -> None:
    mappings = {mapping_kind: {} for mapping_kind in ("first", "last", "first_last", "last_first_initial")}
    mappings[kind] = {name: 2.0}

    with pytest.raises(ValueError, match=rf"name-count {kind} key .*canonical_v2"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            tuple(mappings[mapping_kind] for mapping_kind in ("first", "last", "first_last", "last_first_initial")),
            tiny_name_counts_provenance(),
        )


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("first", "ada"),
        ("first", "a b"),
        ("last", "van der berg"),
        ("first_last", "ada smith"),
        ("first_last", "a b smith"),
        ("first_last", "ada van der berg"),
        ("last_first_initial", "van der berg a"),
    ),
)
def test_name_count_key_contract_accepts_producible_canonical_keys(kind: str, name: str) -> None:
    assert feature_block_arrow._validated_name_count_entry(kind, name, 2.0) == (name, 2.0)


@pytest.mark.parametrize("count", [float("nan"), float("inf"), float("-inf"), 0.0, -3.0])
def test_writer_rejects_nonfinite_and_nonpositive_counts(tmp_path: Path, count: float) -> None:
    with pytest.raises(ValueError, match="must be a finite positive number"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({"ada": count}, {}, {}, {}),
            tiny_name_counts_provenance(),
        )


def test_fresh_writer_validates_each_name_count_entry_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 2.0, "grace": 3.0}, {"smith": 4.0}, {"ada smith": 2.0}, {"smith a": 2.0})
    validation_calls = 0
    original_validator = feature_block_arrow._validated_name_count_entry

    def counted_validator(kind: str, raw_name: object, raw_count: object) -> tuple[str, float]:
        nonlocal validation_calls
        validation_calls += 1
        return original_validator(kind, raw_name, raw_count)

    monkeypatch.setattr(feature_block_arrow, "_validated_name_count_entry", counted_validator)

    feature_block_arrow.write_name_counts_index(
        tmp_path,
        mappings,
        tiny_name_counts_provenance(),
    )

    assert validation_calls == sum(len(mapping) for mapping in mappings)


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
