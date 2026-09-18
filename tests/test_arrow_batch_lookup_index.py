from __future__ import annotations

import os
from pathlib import Path

import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
from s2and.incremental_linking.feature_block import (
    ARROW_BATCH_LOOKUP_INDEX_UNFINGERPRINTED,
    read_arrow_batch_lookup_index_batch_indices,
    validate_arrow_batch_lookup_index,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_raw_arrow_batch_lookup_indexes,
)


def _fail_fingerprint(path_arg: Path, *, source_size: int) -> int:
    raise AssertionError(f"source fingerprint must not be computed for {path_arg} size={source_size}")


def _two_row_signatures_table(tmp_path: Path, pa) -> Path:
    return Path(
        write_arrow_ipc_table(
            pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
            tmp_path / "signatures.arrow",
            max_record_batch_rows=1,
        )
    )


def test_raw_planner_index_rejects_same_size_unsampled_middle_rewrite_python(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signature_ids = [f"key{index:013d}" for index in range(30_000)]
    path = write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(signature_ids, type=pa.string()),
                "payload": pa.array(["x" * 8] * len(signature_ids), type=pa.string()),
            }
        ),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1000,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    payload = Path(path).read_bytes()
    old_value = signature_ids[len(signature_ids) // 2].encode()
    new_value = b"new0000000000000"
    rewrite_offset = payload.index(old_value)
    assert len(old_value) == len(new_value)
    assert rewrite_offset > 65_536
    assert rewrite_offset < len(payload) - 65_536
    Path(path).write_bytes(payload[:rewrite_offset] + new_value + payload[rewrite_offset + len(old_value) :])

    with pytest.raises(ValueError, match="stale"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
        )
    with pytest.raises(ValueError, match="stale"):
        validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")
    with pytest.raises(ValueError, match="stale"):
        read_arrow_batch_lookup_index_batch_indices(
            path,
            index_path,
            key_column="signature_id",
            values=[signature_ids[0]],
        )


def test_raw_planner_index_rejects_source_changed_while_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")

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
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    real_fingerprint_once = feature_block_arrow_module._source_file_fingerprint_once  # noqa: SLF001

    def mutating_fingerprint_once(path_arg: Path, *, source_size: int) -> int:
        fingerprint = real_fingerprint_once(path_arg, source_size=source_size)
        stat = path_arg.stat()
        os.utime(path_arg, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return fingerprint

    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", mutating_fingerprint_once)

    with pytest.raises(ValueError, match="changed while reading batch lookup index"):
        read_arrow_batch_lookup_index_batch_indices(
            path,
            index_path,
            key_column="signature_id",
            values=["s1"],
        )


def test_write_batch_lookup_index_can_skip_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    path = _two_row_signatures_table(tmp_path, pa)
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", _fail_fingerprint)

    _, written = write_arrow_batch_lookup_index(
        path, index_path, key_column="signature_id", table_name="signatures", fingerprint_source=False
    )
    assert written["reused"] is False
    assert written["source_fingerprint"] == ARROW_BATCH_LOOKUP_INDEX_UNFINGERPRINTED
    assert written["source_fingerprint_kind"] == "size_only"
    assert feature_block_arrow_module.read_arrow_batch_lookup_index_batch_indices_for_request(
        path, index_path, key_column="signature_id", values=["s1"]
    ) == {0}


def test_skip_source_fingerprint_cannot_reuse_an_existing_index(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = _two_row_signatures_table(tmp_path, pa)
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")

    with pytest.raises(ValueError, match="cannot be combined with overwrite=False"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
            fingerprint_source=False,
        )
    # also refused when no index exists yet, so the rule does not depend on disk state
    with pytest.raises(ValueError, match="cannot be combined with overwrite=False"):
        write_arrow_batch_lookup_index(
            path,
            tmp_path / "missing.bin",
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
            fingerprint_source=False,
        )


def test_strict_readers_reject_unfingerprinted_index_with_explicit_message(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = _two_row_signatures_table(tmp_path, pa)
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(
        path, index_path, key_column="signature_id", table_name="signatures", fingerprint_source=False
    )
    message = "written with fingerprint_source=False"

    with pytest.raises(ValueError, match=message):
        validate_arrow_batch_lookup_index(path, index_path, key_column="signature_id")
    with pytest.raises(ValueError, match=message):
        read_arrow_batch_lookup_index_batch_indices(path, index_path, key_column="signature_id", values=["s1"])
    with pytest.raises(ValueError, match=message):
        write_arrow_batch_lookup_index(
            path, index_path, key_column="signature_id", table_name="signatures", overwrite=False
        )
    # a fresh fingerprinted write over it is the documented recovery
    _, rewritten = write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    assert rewritten["source_fingerprint_kind"] == "fnv1a64_full_file"
    _, reused = write_arrow_batch_lookup_index(
        path, index_path, key_column="signature_id", table_name="signatures", overwrite=False
    )
    assert reused["reused"] is True
    assert reused["source_fingerprint_kind"] == "fnv1a64_full_file"


def test_write_raw_arrow_batch_lookup_indexes_passes_skip_fingerprint_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    signatures = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    papers = write_arrow_ipc_table(
        pa.table({"paper_id": pa.array(["p1", "p2"], type=pa.string())}),
        tmp_path / "papers.arrow",
        max_record_batch_rows=1,
    )
    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", _fail_fingerprint)

    indexed_paths, metrics = write_raw_arrow_batch_lookup_indexes(
        {"signatures": signatures, "papers": papers},
        tmp_path / "idx",
        fingerprint_source=False,
    )
    assert {"signatures_batch_index", "papers_batch_index"} <= set(indexed_paths)
    assert {table_metrics["source_fingerprint_kind"] for table_metrics in metrics.values()} == {"size_only"}
    with pytest.raises(ValueError, match="cannot be combined with overwrite=False"):
        write_raw_arrow_batch_lookup_indexes(
            {"signatures": signatures},
            tmp_path / "idx2",
            overwrite=False,
            fingerprint_source=False,
        )


def test_request_time_batch_lookup_does_not_fingerprint_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")

    monkeypatch.setattr(feature_block_arrow_module, "_source_file_fingerprint_once", _fail_fingerprint)

    assert feature_block_arrow_module.read_arrow_batch_lookup_index_batch_indices_for_request(
        path,
        index_path,
        key_column="signature_id",
        values=["s1"],
    ) == {0}
