"""Bounded-memory name-count index writer tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from s2and.incremental_linking import feature_block_arrow


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
        )


def test_writer_rejects_keys_that_would_collide_after_stringification(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="name-count first keys must be strings"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({1: 2.0, "1": 3.0}, {}, {}, {}),
        )


def test_writer_rejects_noncanonical_name_count_keys(tmp_path: Path) -> None:
    cases = (
        ("first-uppercase", "first", "Ada"),
        ("first-single-letter", "first", "a"),
        ("first-punctuation", "first", "ada!"),
        ("first-double-space", "first", "ada  marie"),
        ("last-empty", "last", ""),
        ("first-last-unseparated", "first_last", "adasmith"),
        ("first-last-short-first", "first_last", "a smith"),
        ("last-first-initial-long", "last_first_initial", "smith ad"),
        ("last-first-initial-uppercase", "last_first_initial", "smith A"),
    )
    mapping_kinds = ("first", "last", "first_last", "last_first_initial")
    for case_id, kind, name in cases:
        mappings = {mapping_kind: {} for mapping_kind in mapping_kinds}
        mappings[kind] = {name: 2.0}

        try:
            feature_block_arrow.write_name_counts_index(
                tmp_path / case_id,
                tuple(mappings[mapping_kind] for mapping_kind in mapping_kinds),
            )
        except ValueError as error:
            assert f"name-count {kind} key" in str(error) and "canonical_v2" in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: noncanonical key was accepted")


def test_name_count_key_contract_accepts_producible_canonical_keys() -> None:
    cases = (
        ("first-single-token", "first", "ada"),
        ("first-multiple-token", "first", "a b"),
        ("last-particle", "last", "van der berg"),
        ("first-last", "first_last", "ada smith"),
        ("first-multiple-last", "first_last", "a b smith"),
        ("first-last-particle", "first_last", "ada van der berg"),
        ("last-particle-initial", "last_first_initial", "van der berg a"),
    )
    for case_id, kind, name in cases:
        assert feature_block_arrow._validated_name_count_entry(kind, name, 2.0) == (name, 2.0), case_id


def test_writer_rejects_nonfinite_and_nonpositive_counts(tmp_path: Path) -> None:
    cases = (
        ("nan", float("nan")),
        ("zero", 0.0),
    )
    for case_id, count in cases:
        try:
            feature_block_arrow.write_name_counts_index(
                tmp_path / case_id,
                ({"ada": count}, {}, {}, {}),
            )
        except ValueError as error:
            assert "must be a finite positive number" in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: invalid count was accepted")


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
    )

    assert validation_calls == sum(len(mapping) for mapping in mappings)


def test_identical_counts_produce_byte_identical_flat_indexes(tmp_path: Path) -> None:
    mappings = (
        {"ada": 2.0, "grace": 3.0},
        {"lovelace": 4.0, "hopper": 5.0},
        {"ada lovelace": 2.0, "grace hopper": 3.0},
        {"lovelace a": 2.0, "hopper g": 3.0},
    )
    reversed_mappings = tuple(dict(reversed(tuple(mapping.items()))) for mapping in mappings)
    indexes = [
        Path(feature_block_arrow.write_name_counts_index(tmp_path / label, values)[0])
        for label, values in (("first", mappings), ("second", reversed_mappings))
    ]
    kinds = ("first", "last", "first_last", "last_first_initial")
    expected_files = {"manifest.json", *(f"{kind}.bin" for kind in kinds)}

    assert all({path.name for path in index.iterdir()} == expected_files for index in indexes)
    manifests = [(index / "manifest.json").read_bytes() for index in indexes]
    assert manifests[0] == manifests[1]
    assert len({hashlib.sha256(manifest).hexdigest() for manifest in manifests}) == 1
    manifest = json.loads(manifests[0])
    assert manifest["kind"] == "s2and_name_counts"
    assert manifest["format_version"] == 1
    assert all(set(entry) == {"byte_count", "sha256"} for entry in manifest["files"].values())


def test_reused_kind_hash_seed_preserves_binary_index_hashes() -> None:
    for case_id, kind, name in (
        ("first-ascii", "first", "ada"),
        ("last-unicode", "last", "李"),
        ("first-last-spaces", "first_last", "anne marie o connor"),
    ):
        name_bytes = name.encode("utf-8")
        expected = (
            feature_block_arrow._fnv64_bytes(name_bytes),
            feature_block_arrow._fnv64_bytes(
                feature_block_arrow._NAME_COUNTS_INDEX_HASH_DOMAIN + kind.encode("utf-8") + b"\x00" + name_bytes
            ),
        )

        assert feature_block_arrow._name_counts_index_hashes(kind, name_bytes) == expected, case_id
