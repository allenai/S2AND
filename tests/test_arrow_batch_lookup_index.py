from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
from s2and.incremental_linking.feature_block import (
    read_arrow_batch_lookup_index_batch_indices,
    validate_arrow_batch_lookup_index,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_raw_arrow_batch_lookup_indexes,
)


def _write_tiny_index(tmp_path: Path) -> tuple[str, Path]:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2", "s3"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    return path, index_path


def test_lookup_returns_every_batch_for_repeated_keys_and_omits_absent_keys(tmp_path: Path) -> None:
    """A paper's authors can span several batches, all of which must be selected."""
    import pyarrow as pa

    keys = ["paper-a", "paper-a", "paper-b", "paper-c", "paper-a", "paper-c", "paper-d"]
    path = write_arrow_ipc_table(
        pa.table({"paper_id": pa.array(keys, type=pa.string())}),
        tmp_path / "paper_authors.arrow",
        max_record_batch_rows=2,
    )
    index_path = tmp_path / "paper_authors.index"
    write_arrow_batch_lookup_index(path, index_path, key_column="paper_id", table_name="paper_authors")

    for requested, expected in (
        (["paper-a"], {0, 2}),
        (["paper-c", "paper-c", "missing"], {1, 2}),
        (["paper-a", "paper-d"], {0, 2, 3}),
        (["missing"], set()),
        ([], set()),
    ):
        assert (
            read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="paper_id", values=iter(requested))
            == expected
        )


def test_native_utf8_batch_hash_matches_python_fnv1a64() -> None:
    values = ["", "abc", "Renée", "李", "emoji-🧪"]

    assert feature_block_arrow_module._fnv64_utf8_batch(values) == [  # noqa: SLF001
        feature_block_arrow_module._fnv64_bytes(value.encode("utf-8"))  # noqa: SLF001
        for value in values
    ]


@pytest.mark.parametrize("arrow_type", ["string", "large_string"])
def test_vectorized_arrow_hash_matches_python_fnv1a64(arrow_type: str) -> None:
    import pyarrow as pa

    # Mixed lengths, multi-byte UTF-8, an empty key, and one long key exercise the byte-position loop.
    values = ["", "abc", "Renée", "李", "emoji-🧪", "x" * 300, "12345678-9"]
    array = pa.array(["skipped", *values, "tail"], type=getattr(pa, arrow_type)()).slice(1, len(values))

    hashes = feature_block_arrow_module._fnv64_arrow_utf8(array)  # noqa: SLF001

    assert hashes.tolist() == [feature_block_arrow_module._fnv64_bytes(v.encode("utf-8")) for v in values]  # noqa: SLF001


def test_vectorized_arrow_hash_uses_text_form_for_non_string_keys() -> None:
    import pyarrow as pa

    hashes = feature_block_arrow_module._fnv64_arrow_utf8(pa.array([7, 1234], type=pa.int64()))  # noqa: SLF001

    assert hashes.tolist() == [feature_block_arrow_module._fnv64_bytes(v) for v in (b"7", b"1234")]  # noqa: SLF001


def test_write_uses_bounded_sorted_runs_and_emits_v3_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

    keys = ["zeta", "alpha", "theta", "beta", "eta", "gamma", "delta"]
    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(keys, type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=2,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    buffered_run_sizes: list[int] = []
    original_run_writer = feature_block_arrow_module._write_arrow_batch_lookup_sort_run  # noqa: SLF001

    def record_run(run_path: Path, records: list[tuple[int, int]]) -> None:
        buffered_run_sizes.append(len(records))
        original_run_writer(run_path, records)

    monkeypatch.setattr(feature_block_arrow_module, "_ARROW_BATCH_LOOKUP_SORT_BUFFER_RECORDS", 2)
    monkeypatch.setattr(feature_block_arrow_module, "_write_arrow_batch_lookup_sort_run", record_run)

    write_arrow_batch_lookup_index(
        path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
        max_record_batch_rows=2,
    )

    source_path = Path(path)
    source_size = source_path.stat().st_size
    records = sorted(
        (feature_block_arrow_module._fnv64_bytes(key.encode("utf-8")), row_index // 2, 0)  # noqa: SLF001
        for row_index, key in enumerate(keys)
    )
    expected = bytearray(
        feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(  # noqa: SLF001
            feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_MAGIC,  # noqa: SLF001
            len(keys),
            source_size,
            feature_block_arrow_module._fnv64_bytes(b"signature_id"),  # noqa: SLF001
        )
    )
    for record in records:
        expected.extend(feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(*record))  # noqa: SLF001

    assert expected[:8] == b"S2ABI003"
    assert len(expected) == 32 + 16 * len(keys)
    assert buffered_run_sizes == [2, 2, 2, 1]
    assert index_path.read_bytes() == expected
    assert not list(tmp_path.glob(f".{index_path.name}.*"))


def test_validation_rejects_inexact_batch_index_body_length(tmp_path: Path) -> None:
    for mutation in ("truncated", "trailing"):
        case_root = tmp_path / mutation
        case_root.mkdir()
        path, index_path = _write_tiny_index(case_root)
        payload = index_path.read_bytes()
        if mutation == "truncated":
            index_path.write_bytes(payload[:-1])
        else:
            index_path.write_bytes(payload + b"\x00")

        with pytest.raises(ValueError, match="does not match expected length"):
            validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")


def test_validation_rejects_decreasing_batch_index_hashes(tmp_path: Path) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    payload = bytearray(index_path.read_bytes())
    header_size = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size  # noqa: SLF001
    record_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT  # noqa: SLF001
    records = [
        bytes(payload[offset : offset + record_struct.size])
        for offset in range(header_size, len(payload), record_struct.size)
    ]
    assert record_struct.unpack(records[0])[0] < record_struct.unpack(records[-1])[0]
    payload[header_size:] = b"".join(reversed(records))
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="key hashes are not nondecreasing"):
        validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")


def test_validation_rejects_out_of_bounds_record_batch_index(tmp_path: Path) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    payload = bytearray(index_path.read_bytes())
    header_size = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size  # noqa: SLF001
    record_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT  # noqa: SLF001
    key_hash, _batch_index, reserved = record_struct.unpack_from(payload, header_size)
    record_struct.pack_into(payload, header_size, key_hash, 3, reserved)
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="batch index 3 is out of bounds"):
        validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")


def test_write_reuse_validates_last_record(tmp_path: Path) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    payload = bytearray(index_path.read_bytes())
    header_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT  # noqa: SLF001
    record_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT  # noqa: SLF001
    record_count = int(header_struct.unpack_from(payload)[1])
    last_record_offset = header_struct.size + (record_count - 1) * record_struct.size
    key_hash, _batch_index, reserved = record_struct.unpack_from(payload, last_record_offset)
    record_struct.pack_into(payload, last_record_offset, key_hash, 3, reserved)
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="batch index 3 is out of bounds"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
        )


def test_raw_planner_index_rejects_source_changed_while_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    real_reader = feature_block_arrow_module._read_arrow_batch_lookup_records  # noqa: SLF001

    def mutating_reader(*args, **kwargs):
        result = real_reader(*args, **kwargs)
        arrow_path = Path(args[0])
        stat = arrow_path.stat()
        os.utime(arrow_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(feature_block_arrow_module, "_read_arrow_batch_lookup_records", mutating_reader)

    with pytest.raises(ValueError, match="changed while building batch lookup index"):
        write_arrow_batch_lookup_index(
            path,
            tmp_path / "signatures.signatures_batch_index.bin",
            key_column="signature_id",
            table_name="signatures",
        )


def test_raw_planner_index_rejects_source_changed_while_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    real_lower_bound = feature_block_arrow_module._arrow_batch_lookup_lower_bound  # noqa: SLF001

    def mutating_lower_bound(*args):
        result = real_lower_bound(*args)
        stat = Path(path).stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(feature_block_arrow_module, "_arrow_batch_lookup_lower_bound", mutating_lower_bound)

    with pytest.raises(ValueError, match="changed while reading batch lookup index"):
        read_arrow_batch_lookup_index_batch_indices(
            path,
            index_path,
            key_column="signature_id",
            values=["s1"],
        )


@pytest.mark.parametrize("operation", ["lookup", "validate"])
def test_old_index_format_requires_rebuilding(tmp_path: Path, operation: str) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    current = index_path.read_bytes()
    # A real v2 header includes the removed source fingerprint word.
    index_path.write_bytes(b"S2ABI002" + current[8:32] + struct.pack("<Q", 123) + current[32:])
    with pytest.raises(ValueError, match="invalid magic bytes.*Rebuild"):
        if operation == "lookup":
            read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="signature_id", values=["s1"])
        else:
            validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")


def test_structural_reuse_preserves_bytes_and_does_not_claim_content_validation(tmp_path: Path) -> None:
    import pyarrow as pa

    path, index_path = _write_tiny_index(tmp_path)
    original_index = index_path.read_bytes()
    original = Path(path).read_bytes()
    assert b"s2" in original
    Path(path).write_bytes(original.replace(b"s2", b"t2"))
    with pa.memory_map(path, "r") as source:
        assert pa.ipc.open_file(source).read_all()["signature_id"].to_pylist() == ["s1", "t2", "s3"]
    metrics = validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")
    assert metrics == {
        "magic": "S2ABI003",
        "record_count": 3,
        "source_size": len(original),
        "key_column_hash": feature_block_arrow_module._fnv64_bytes(b"signature_id"),  # noqa: SLF001
    }
    _, reused = write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", overwrite=False)
    assert reused["reused"] is True
    assert reused["magic"] == "S2ABI003"
    assert index_path.read_bytes() == original_index
    # The stale key remains reachable: the caller owns table/index pairing.
    assert read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="signature_id", values=["s2"]) == {
        1
    }
    assert (
        read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="signature_id", values=["t2"]) == set()
    )


@pytest.mark.parametrize("operation", ["lookup", "validate"])
def test_source_size_mismatch_is_still_rejected(tmp_path: Path, operation: str) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    data = bytearray(index_path.read_bytes())
    struct.pack_into("<Q", data, 16, Path(path).stat().st_size + 1)
    index_path.write_bytes(data)
    with pytest.raises(ValueError, match="stale.*indexed size"):
        if operation == "lookup":
            read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="signature_id", values=["s1"])
        else:
            validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")


def test_source_changed_while_publishing_preserves_previous_index_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, index_path = _write_tiny_index(tmp_path)
    previous = index_path.read_bytes()
    original_write = feature_block_arrow_module._write_sorted_arrow_batch_lookup_records  # noqa: SLF001

    def mutate_after_write(*args):
        original_write(*args)
        stat = Path(path).stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    monkeypatch.setattr(feature_block_arrow_module, "_write_sorted_arrow_batch_lookup_records", mutate_after_write)
    with pytest.raises(ValueError, match="changed while publishing batch lookup index"):
        write_arrow_batch_lookup_index(path, index_path, key_column="signature_id")
    assert index_path.read_bytes() == previous
    assert not list(tmp_path.glob(f".{index_path.name}.*"))


def test_raw_indexes_can_be_created_and_structurally_reused(tmp_path: Path) -> None:
    import pyarrow as pa

    paths = {
        "signatures": write_arrow_ipc_table(
            pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}), tmp_path / "signatures.arrow"
        ),
        "papers": write_arrow_ipc_table(
            pa.table({"paper_id": pa.array(["p1", "p2"], type=pa.string())}), tmp_path / "papers.arrow"
        ),
    }
    indexed, written = write_raw_arrow_batch_lookup_indexes(paths, tmp_path / "indexes", overwrite=False)
    again, reused = write_raw_arrow_batch_lookup_indexes(paths, tmp_path / "indexes", overwrite=False)
    assert again == indexed
    assert {"signatures_batch_index", "papers_batch_index"} <= indexed.keys()
    assert all(metrics["magic"] == "S2ABI003" and not metrics["reused"] for metrics in written.values())
    assert all(metrics["reused"] for metrics in reused.values())
