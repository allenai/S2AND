"""Raw-planner Arrow IPC, sidecar, and artifact IO helpers."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import mmap
import os
import shutil
import struct
import tempfile
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from s2and._atomic_io import fsync_directory
from s2and._sha256 import sha256_file as _sha256_file
from s2and.arrow_inputs import (
    RAW_PLANNER_ARROW_BATCH_INDEX_KEYS,
    RAW_PLANNER_ARROW_KEY_COLUMNS,
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    normalize_arrow_paths,
)
from s2and.arrow_schema import validate_arrow_schema
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block_contract import normalize_cluster_seed_disallow_pairs
from s2and.name_counts_index import NAME_COUNTS_INDEX_SCHEMA_VERSION
from s2and.text import canonicalize_name_text

ARROW_PHYSICAL_LAYOUT_SCHEMA_VERSION = "s2and_arrow_physical_v1"
ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION = "arrow_batch_lookup_index"
INCREMENTAL_QUERY_SIGNATURE_VIEWS = frozenset({"auto", "full", "initial_only"})
_NAME_COUNTS_INDEX_MAGIC = b"S2NCI001"
_ARROW_BATCH_LOOKUP_INDEX_MAGIC = b"S2ABI002"
_NAME_COUNTS_INDEX_HASH_DOMAIN = b"s2and-name-counts-index-v1\x00"
_ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN = b"s2and-arrow-batch-lookup-index-source\x00"
_ARROW_BATCH_LOOKUP_INDEX_SOURCE_READ_BYTES = 1024 * 1024
_NAME_COUNTS_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQ")
_NAME_COUNTS_INDEX_RECORD_STRUCT = struct.Struct("<QQQIId")
_NAME_COUNTS_SORT_RUN_RECORD_STRUCT = struct.Struct("<QQdI")
_NAME_COUNTS_SORT_BUFFER_RECORDS = 1_000_000
_NAME_COUNTS_WRITE_BUFFER_BYTES = 1024 * 1024
_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQQ")
_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT = struct.Struct("<QII")
_ARROW_BATCH_LOOKUP_SORT_BUFFER_RECORDS = 1_000_000
_ARROW_BATCH_LOOKUP_KEY_CHUNK_ROWS = 16_384
_ARROW_BATCH_LOOKUP_WRITE_BUFFER_BYTES = 1024 * 1024
_FNV64_OFFSET = 14695981039346656037
_FNV64_PRIME = 1099511628211
_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS = 2


@dataclass(frozen=True)
class IncrementalQuerySignatureRequest:
    """Typed query-signature request row for raw Arrow incremental scoring."""

    signature_id: str
    query_view: str
    query_author: str


@dataclass(frozen=True)
class _ArrowSourceSnapshot:
    size: int
    mtime_ns: int
    fingerprint: int


@dataclass(frozen=True)
class _ArrowSourceDigests:
    size: int
    mtime_ns: int
    sha256: str
    fingerprint: int | None


@dataclass
class _ArrowBatchLookupRecords:
    """Bounded-memory records and physical-layout facts for one index build."""

    buffered_records: list[tuple[int, int]]
    run_paths: list[Path]
    row_count: int
    max_batch_rows: int
    record_batch_count: int
    peak_buffered_records: int


def write_incremental_query_signatures_arrow(
    path: Path,
    signature_ids: Iterable[Any],
    *,
    query_views: Iterable[Any] | None = None,
    query_authors: Iterable[Any] | None = None,
) -> None:
    """Write the canonical Arrow incremental query-signature request table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _normalize_incremental_query_signature_requests(
        signature_ids,
        query_views=query_views,
        query_authors=query_authors,
    )
    table = pa.table(
        {
            "signature_id": pa.array([row.signature_id for row in rows], type=pa.string()),
            "query_view": pa.array([row.query_view for row in rows], type=pa.string()),
            "query_author": pa.array([row.query_author for row in rows], type=pa.string()),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_incremental_query_signatures_arrow(path: Path) -> tuple[IncrementalQuerySignatureRequest, ...]:
    """Read and validate a canonical Arrow incremental query-signature request table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    validate_arrow_schema(table.schema, table_name="incremental_query_signatures")
    return _normalize_incremental_query_signature_requests(
        table["signature_id"].to_pylist(),
        query_views=table["query_view"].to_pylist(),
        query_authors=table["query_author"].to_pylist(),
    )


def _normalize_incremental_query_signature_requests(
    signature_ids: Iterable[Any],
    *,
    query_views: Iterable[Any] | None = None,
    query_authors: Iterable[Any] | None = None,
) -> tuple[IncrementalQuerySignatureRequest, ...]:
    signature_id_values = tuple(signature_ids)
    if query_views is None:
        query_view_values: tuple[Any, ...] = ("auto",) * len(signature_id_values)
    else:
        query_view_values = tuple(query_views)
    if query_authors is None:
        query_author_values: tuple[Any, ...] = ("",) * len(signature_id_values)
    else:
        query_author_values = tuple(query_authors)
    if len(query_view_values) != len(signature_id_values):
        raise ValueError(
            "incremental query signatures Arrow query_view length must match signature_id length: "
            f"{len(query_view_values)} != {len(signature_id_values)}"
        )
    if len(query_author_values) != len(signature_id_values):
        raise ValueError(
            "incremental query signatures Arrow query_author length must match signature_id length: "
            f"{len(query_author_values)} != {len(signature_id_values)}"
        )

    rows: list[IncrementalQuerySignatureRequest] = []
    seen_signature_ids: set[str] = set()
    for signature_id_value, query_view_value, query_author_value in zip(
        signature_id_values,
        query_view_values,
        query_author_values,
        strict=True,
    ):
        if signature_id_value is None:
            raise ValueError("incremental query signatures Arrow cannot contain null signature_id values")
        if query_view_value is None:
            raise ValueError("incremental query signatures Arrow cannot contain null query_view values")
        if query_author_value is None:
            raise ValueError("incremental query signatures Arrow cannot contain null query_author values")
        signature_id = str(signature_id_value)
        query_view = str(query_view_value)
        query_author = str(query_author_value)
        if not signature_id:
            raise ValueError("incremental query signatures Arrow cannot contain empty signature_id values")
        if not query_view:
            raise ValueError(
                f"incremental query signatures Arrow cannot contain empty query_view values: {signature_id!r}"
            )
        if query_view not in INCREMENTAL_QUERY_SIGNATURE_VIEWS:
            raise ValueError(
                "incremental query signatures Arrow contains unknown query_view "
                f"{query_view!r}; expected one of {sorted(INCREMENTAL_QUERY_SIGNATURE_VIEWS)!r}"
            )
        if signature_id in seen_signature_ids:
            raise ValueError(f"incremental query signatures Arrow contains duplicate signature_id: {signature_id!r}")
        seen_signature_ids.add(signature_id)
        rows.append(
            IncrementalQuerySignatureRequest(
                signature_id=signature_id,
                query_view=query_view,
                query_author=query_author,
            )
        )
    return tuple(rows)


def write_cluster_seeds_arrow(path: Path, cluster_seeds_require: Mapping[Any, Any]) -> None:
    """Write the canonical Arrow cluster-seed table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    seen_signature_ids: set[str] = set()
    for signature_id, cluster_id in cluster_seeds_require.items():
        signature_key = str(signature_id)
        cluster_key = str(cluster_id)
        if not signature_key:
            raise ValueError("cluster seeds Arrow cannot contain empty signature_id values")
        if not cluster_key:
            raise ValueError(f"cluster seeds Arrow cannot contain empty cluster_id values: {signature_key!r}")
        if signature_key in seen_signature_ids:
            raise ValueError(f"cluster seeds Arrow contains duplicate signature_id: {signature_key!r}")
        seen_signature_ids.add(signature_key)
        items.append((signature_key, cluster_key))
    table = pa.table(
        {
            "signature_id": pa.array([signature_id for signature_id, _cluster_id in items], type=pa.string()),
            "cluster_id": pa.array([cluster_id for _signature_id, cluster_id in items], type=pa.string()),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_cluster_seeds_arrow(path: Path) -> dict[str, str]:
    """Read and validate a canonical Arrow cluster-seed table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    validate_arrow_schema(table.schema, table_name="cluster_seeds")
    rows: dict[str, str] = {}
    for index in range(table.num_rows):
        signature_value = table["signature_id"][index].as_py()
        cluster_value = table["cluster_id"][index].as_py()
        if signature_value is None or cluster_value is None:
            raise ValueError("cluster seeds Arrow cannot contain null signature_id or cluster_id values")
        signature_id = str(signature_value)
        cluster_id = str(cluster_value)
        if not signature_id:
            raise ValueError("cluster seeds Arrow cannot contain empty signature_id values")
        if not cluster_id:
            raise ValueError(f"cluster seeds Arrow cannot contain empty cluster_id values: {signature_id!r}")
        existing_cluster_id = rows.get(signature_id)
        if existing_cluster_id is not None:
            raise ValueError(
                f"cluster seeds Arrow contains duplicate signature_id: {signature_id!r} "
                f"({existing_cluster_id!r} and {cluster_id!r})"
            )
        rows[signature_id] = cluster_id
    return rows


def write_cluster_seed_disallows_arrow(path: Path, pairs: Iterable[tuple[Any, Any]]) -> None:
    """Write the canonical Arrow cluster-seed disallow table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_cluster_seed_disallow_pairs(pairs)
    table = pa.table(
        {
            "signature_id_1": pa.array([left for left, _right in normalized], type=pa.string()),
            "signature_id_2": pa.array([right for _left, right in normalized], type=pa.string()),
        }
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_cluster_seed_disallows_arrow(path: Path) -> tuple[tuple[str, str], ...]:
    """Read and validate a canonical Arrow cluster-seed disallow table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    validate_arrow_schema(table.schema, table_name="cluster_seed_disallows")
    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, right in zip(
        table["signature_id_1"].to_pylist(),
        table["signature_id_2"].to_pylist(),
        strict=True,
    ):
        if left is None or right is None:
            raise ValueError("cluster seed disallows Arrow cannot contain null signature ids")
        normalized_pair = normalize_cluster_seed_disallow_pairs([(str(left), str(right))])[0]
        if normalized_pair in seen_pairs:
            raise ValueError(f"cluster seed disallows Arrow contains duplicate pair: {normalized_pair!r}")
        seen_pairs.add(normalized_pair)
        rows.append(normalized_pair)
    return tuple(rows)


def write_altered_cluster_signatures_arrow(path: Path, signature_ids: Iterable[Any]) -> None:
    """Write the canonical Arrow altered-cluster-signature table."""

    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_unique_signature_ids(
        signature_ids,
        table_name="altered cluster signatures",
    )
    table = pa.table({"signature_id": pa.array(normalized, type=pa.string())})
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def read_altered_cluster_signatures_arrow(path: Path) -> tuple[str, ...]:
    """Read and validate a canonical Arrow altered-cluster-signature table."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    validate_arrow_schema(table.schema, table_name="altered_cluster_signatures")
    return _normalize_unique_signature_ids(
        table["signature_id"].to_pylist(),
        table_name="altered cluster signatures",
    )


def _normalize_unique_signature_ids(
    signature_ids: Iterable[Any],
    *,
    table_name: str,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in signature_ids:
        if value is None:
            raise ValueError(f"{table_name} Arrow cannot contain null signature_id values")
        signature_id = str(value)
        if not signature_id:
            raise ValueError(f"{table_name} Arrow cannot contain empty signature_id values")
        if signature_id in seen:
            raise ValueError(f"{table_name} Arrow contains duplicate signature_id: {signature_id!r}")
        seen.add(signature_id)
        normalized.append(signature_id)
    return tuple(normalized)


@contextmanager
def temporary_cluster_seed_sidecars(
    cluster_seeds_require: Mapping[Any, Any],
    *,
    prefix: str,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
) -> Iterator[dict[str, str]]:
    """Write and yield request-scoped cluster-seed sidecars."""

    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmpdir_path = Path(tmpdir)
        paths: dict[str, str] = {}
        cluster_seed_path = tmpdir_path / "cluster_seeds.arrow"
        write_cluster_seeds_arrow(cluster_seed_path, cluster_seeds_require)
        paths["cluster_seeds"] = str(cluster_seed_path)
        if cluster_seeds_disallow is not None:
            disallow_path = tmpdir_path / "cluster_seed_disallows.arrow"
            write_cluster_seed_disallows_arrow(disallow_path, cluster_seeds_disallow)
            paths["cluster_seed_disallows"] = str(disallow_path)
        yield paths


def _record_batch_limit_for_table(
    table_name: str,
    max_record_batch_rows: Mapping[str, int] | int | None,
) -> int | None:
    if max_record_batch_rows is None:
        return None
    if isinstance(max_record_batch_rows, Mapping):
        raw_limit = cast(Mapping[str, int], max_record_batch_rows).get(table_name)
        if raw_limit is None:
            return None
    else:
        raw_limit = max_record_batch_rows
    limit = int(raw_limit)
    if limit <= 0:
        raise ValueError(f"max_record_batch_rows must be positive for {table_name!r}: {limit}")
    return limit


def write_arrow_ipc_table(
    table: Any,
    path: str | Path,
    *,
    max_record_batch_rows: int | None = None,
) -> str:
    """Write one Arrow IPC file-format table and return its path."""

    import pyarrow as pa

    batch_limit = None if max_record_batch_rows is None else int(max_record_batch_rows)
    if batch_limit is not None and batch_limit <= 0:
        raise ValueError(f"max_record_batch_rows must be positive: {max_record_batch_rows}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(output_path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=batch_limit)
    return str(output_path)


def arrow_ipc_physical_layout(path: str | Path) -> dict[str, int]:
    """Return row and record-batch layout metrics for one Arrow IPC file."""

    import pyarrow as pa

    row_count = 0
    max_batch_rows = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        record_batch_count = int(reader.num_record_batches)
        for batch_index in range(record_batch_count):
            batch_rows = int(reader.get_batch(batch_index).num_rows)
            row_count += batch_rows
            max_batch_rows = max(max_batch_rows, batch_rows)
    return {
        "row_count": row_count,
        "record_batch_count": record_batch_count,
        "actual_max_batch_rows": max_batch_rows,
    }


def _raise_if_record_batch_limit_exceeded(
    *,
    arrow_path: str | Path,
    table_name: str,
    batch_index: int,
    batch_rows: int,
    max_record_batch_rows: int | None,
) -> None:
    if max_record_batch_rows is None or batch_rows <= max_record_batch_rows:
        return
    batch_label = f"record batch {batch_index}" if batch_index >= 0 else "at least one record batch"
    raise ValueError(
        f"{table_name}.arrow has {batch_label} with {batch_rows} rows, "
        f"exceeding the raw-planner limit of {max_record_batch_rows}: {arrow_path!s}. "
        "Rewrite the Arrow IPC file with bounded record batches before building lookup indexes."
    )


def _decode_arrow_batch_lookup_index_header(index_path: Path, header: bytes) -> dict[str, int | str]:
    if len(header) != _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size:
        raise ValueError(f"Arrow batch lookup index is truncated: {index_path!s}")
    magic = header[:8]
    if magic != _ARROW_BATCH_LOOKUP_INDEX_MAGIC:
        raise ValueError(f"Arrow batch lookup index has invalid magic bytes: {index_path!s}")
    (
        _magic,
        record_count,
        source_size,
        key_column_hash,
        source_fingerprint,
    ) = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.unpack(header)
    return {
        "magic": magic.decode("ascii"),
        "record_count": int(record_count),
        "source_size": int(source_size),
        "key_column_hash": int(key_column_hash),
        "source_fingerprint": int(source_fingerprint),
    }


def _batch_lookup_index_source_mismatch(
    header: Mapping[str, int | str],
    *,
    source_size: int,
    source_fingerprint: int | None,
) -> str | None:
    indexed_size = int(header["source_size"])
    indexed_fingerprint = int(header["source_fingerprint"])
    if indexed_size != int(source_size):
        return f"indexed size={indexed_size} current size={int(source_size)}"
    if source_fingerprint is None or indexed_fingerprint == int(source_fingerprint):
        return None
    return (
        "indexed size/fingerprint="
        f"({indexed_size}, {indexed_fingerprint}) current size/fingerprint="
        f"({int(source_size)}, {int(source_fingerprint)})"
    )


def _arrow_batch_lookup_record_hash(index_mmap: mmap.mmap, record_index: int) -> int:
    offset = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size + record_index * _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
    return int(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.unpack_from(index_mmap, offset)[0])


def _arrow_batch_lookup_record_batch_index(index_mmap: mmap.mmap, record_index: int) -> int:
    offset = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size + record_index * _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
    return int(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.unpack_from(index_mmap, offset)[1])


def _arrow_batch_lookup_lower_bound(index_mmap: mmap.mmap, record_count: int, key_hash: int) -> int:
    lower = 0
    upper = int(record_count)
    while lower < upper:
        midpoint = lower + (upper - lower) // 2
        if _arrow_batch_lookup_record_hash(index_mmap, midpoint) < key_hash:
            lower = midpoint + 1
        else:
            upper = midpoint
    return lower


def _source_snapshot_matches_stat(snapshot: _ArrowSourceSnapshot, stat_result: os.stat_result) -> bool:
    return snapshot.size == int(stat_result.st_size) and snapshot.mtime_ns == int(stat_result.st_mtime_ns)


def _raise_arrow_source_changed(path: Path, *, context: str) -> NoReturn:
    raise ValueError(f"Arrow IPC file changed while {context}: {path!s}")


def _read_arrow_batch_lookup_index_batch_indices(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    values: Iterable[Any],
    validate_source_fingerprint: bool,
    context: str,
) -> set[int]:
    keep_hashes = {_fnv64_bytes(str(value).encode("utf-8")) for value in values}
    if not keep_hashes:
        return set()
    arrow_path_obj = Path(arrow_path)
    index_path_obj = Path(index_path)
    source_stat_before = arrow_path_obj.stat()
    source_size = int(source_stat_before.st_size)
    source_snapshot: _ArrowSourceSnapshot | None = None
    if validate_source_fingerprint:
        source_snapshot = _stable_source_file_snapshot(arrow_path_obj, context=context)
        source_size = source_snapshot.size
    expected_key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
    with index_path_obj.open("rb") as infile:
        with mmap.mmap(infile.fileno(), 0, access=mmap.ACCESS_READ) as index_mmap:
            header = _decode_arrow_batch_lookup_index_header(
                index_path_obj,
                index_mmap[: _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size],
            )
            if int(header["key_column_hash"]) != expected_key_column_hash:
                raise ValueError(
                    f"Arrow batch lookup index '{index_path_obj!s}' was built for a different key column: "
                    f"indexed hash={int(header['key_column_hash'])} expected hash={expected_key_column_hash} "
                    f"key_column={key_column!r}"
                )
            if validate_source_fingerprint:
                if source_snapshot is None:
                    raise AssertionError("source snapshot must be populated")
                source_mismatch = _batch_lookup_index_source_mismatch(
                    header,
                    source_size=source_snapshot.size,
                    source_fingerprint=source_snapshot.fingerprint,
                )
                if source_mismatch is not None:
                    raise ValueError(
                        f"Arrow batch lookup index '{index_path_obj!s}' is stale for '{arrow_path_obj!s}': "
                        f"{source_mismatch}"
                    )
            elif int(header["source_size"]) != source_size:
                raise ValueError(
                    f"Arrow batch lookup index '{index_path_obj!s}' is stale for '{arrow_path_obj!s}': "
                    f"indexed size={int(header['source_size'])} current size={source_size}"
                )
            record_count = int(header["record_count"])
            expected_len = (
                _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size
                + record_count * _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
            )
            if len(index_mmap) != expected_len:
                raise ValueError(
                    f"Arrow batch lookup index '{index_path_obj!s}' length {len(index_mmap)} does not match "
                    f"expected length {expected_len} (record_count={record_count})"
                )
            batch_indices: set[int] = set()
            for key_hash in keep_hashes:
                record_index = _arrow_batch_lookup_lower_bound(index_mmap, record_count, key_hash)
                while (
                    record_index < record_count
                    and _arrow_batch_lookup_record_hash(index_mmap, record_index) == key_hash
                ):
                    batch_indices.add(_arrow_batch_lookup_record_batch_index(index_mmap, record_index))
                    record_index += 1
    source_stat_after = arrow_path_obj.stat()
    if validate_source_fingerprint:
        if source_snapshot is None:
            raise AssertionError("source snapshot must be populated")
        if not _source_snapshot_matches_stat(source_snapshot, source_stat_after):
            _raise_arrow_source_changed(arrow_path_obj, context=context)
    elif int(source_stat_before.st_size) != int(source_stat_after.st_size) or int(
        source_stat_before.st_mtime_ns
    ) != int(source_stat_after.st_mtime_ns):
        _raise_arrow_source_changed(arrow_path_obj, context=context)
    return batch_indices


def read_arrow_batch_lookup_index_batch_indices(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    values: Iterable[Any],
) -> set[int]:
    """Return Arrow record-batch indices with strict source fingerprint validation."""

    return _read_arrow_batch_lookup_index_batch_indices(
        arrow_path,
        index_path,
        key_column=key_column,
        values=values,
        validate_source_fingerprint=True,
        context="reading batch lookup index",
    )


def read_arrow_batch_lookup_index_batch_indices_for_request(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    values: Iterable[Any],
) -> set[int]:
    """Return Arrow record-batch indices for request-time filtered reads.

    This keeps sidecar magic/key/length/source-size checks but intentionally
    avoids hashing the whole Arrow IPC file on every prediction request. Use
    `validate_arrow_batch_lookup_index` for offline strict fingerprint checks.
    """

    return _read_arrow_batch_lookup_index_batch_indices(
        arrow_path,
        index_path,
        key_column=key_column,
        values=values,
        validate_source_fingerprint=False,
        context="reading request-time batch lookup index",
    )


def validate_arrow_batch_lookup_index(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    expected_row_count: int | None = None,
) -> dict[str, int | str]:
    """Validate an index and its record-batch references without reading Arrow rows."""

    return _validate_arrow_batch_lookup_index(
        arrow_path=Path(arrow_path),
        index_path=Path(index_path),
        key_column=key_column,
        expected_row_count=expected_row_count,
    )


def _validate_arrow_batch_lookup_index(
    *,
    arrow_path: Path,
    index_path: Path,
    key_column: str,
    expected_row_count: int | None = None,
    expected_arrow_byte_count: int | None = None,
    expected_arrow_sha256: str | None = None,
    expected_index_byte_count: int | None = None,
    expected_index_sha256: str | None = None,
    validate_source_fingerprint: bool = True,
) -> dict[str, int | str]:
    """Stream and validate one Arrow table/index pair."""

    context = (
        "validating generation-bound batch lookup index"
        if expected_arrow_sha256 is not None
        else "validating batch lookup index"
    )
    source_sha256: str | None = None
    if expected_arrow_sha256 is None:
        source_snapshot = _stable_source_file_snapshot(arrow_path, context=context)
        source_size = source_snapshot.size
        source_mtime_ns = source_snapshot.mtime_ns
        source_fingerprint: int | None = source_snapshot.fingerprint
    else:
        source_digests = (
            _stable_source_file_digests(arrow_path, context=context)
            if validate_source_fingerprint
            else _stable_source_file_sha256(arrow_path, context=context)
        )
        source_size = source_digests.size
        source_mtime_ns = source_digests.mtime_ns
        source_sha256 = source_digests.sha256
        source_fingerprint = source_digests.fingerprint
    if expected_arrow_byte_count is not None and source_size != expected_arrow_byte_count:
        raise ValueError(f"Arrow artifact generation source byte_count mismatch: {arrow_path}")
    if expected_arrow_sha256 is not None and source_sha256 != expected_arrow_sha256:
        raise ValueError(f"Arrow artifact generation source checksum mismatch: {arrow_path}")

    import pyarrow as pa

    with pa.memory_map(str(arrow_path), "r") as source:
        record_batch_count = int(pa.ipc.open_file(source).num_record_batches)
    source_stat = arrow_path.stat()
    if source_size != int(source_stat.st_size) or source_mtime_ns != int(source_stat.st_mtime_ns):
        _raise_arrow_source_changed(arrow_path, context=context)

    index_stat_before = index_path.stat()
    if expected_index_byte_count is not None and int(index_stat_before.st_size) != expected_index_byte_count:
        raise ValueError(f"Arrow artifact generation index byte_count mismatch: {index_path}")

    index_digest = hashlib.sha256() if expected_index_sha256 is not None else None
    with index_path.open("rb") as infile:
        header_bytes = infile.read(_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size)
        if index_digest is not None:
            index_digest.update(header_bytes)
        header = _decode_arrow_batch_lookup_index_header(index_path, header_bytes)
        key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
        if int(header["key_column_hash"]) != key_column_hash:
            raise ValueError(
                f"Arrow batch lookup index '{index_path!s}' was built for a different key column: "
                f"indexed hash={int(header['key_column_hash'])} expected hash={key_column_hash} "
                f"key_column={key_column!r}"
            )
        source_mismatch = _batch_lookup_index_source_mismatch(
            header,
            source_size=source_size,
            source_fingerprint=source_fingerprint,
        )
        if source_mismatch is not None:
            raise ValueError(
                f"Arrow batch lookup index '{index_path!s}' is stale for '{arrow_path!s}': {source_mismatch}"
            )
        record_count = int(header["record_count"])
        if expected_row_count is not None and record_count != int(expected_row_count):
            raise ValueError(
                f"Arrow batch lookup index row count mismatch for {arrow_path!s}: "
                f"index has {record_count} records, expected {int(expected_row_count)}. "
                "Rebuild it with overwrite=True."
            )
        expected_length = (
            _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size + record_count * _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
        )
        if int(index_stat_before.st_size) != expected_length:
            raise ValueError(
                f"Arrow batch lookup index '{index_path!s}' length {index_stat_before.st_size} does not match "
                f"expected length {expected_length} (record_count={record_count})"
            )

        previous_hash: int | None = None
        observed_count = 0
        while chunk := infile.read(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_READ_BYTES):
            if index_digest is not None:
                index_digest.update(chunk)
            if len(chunk) % _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size != 0:
                raise ValueError(f"Arrow batch lookup index is truncated: {index_path!s}")
            for key_hash, batch_index, _reserved in _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.iter_unpack(chunk):
                if previous_hash is not None and key_hash < previous_hash:
                    raise ValueError(
                        f"Arrow batch lookup index '{index_path!s}' key hashes are not nondecreasing "
                        f"at record {observed_count}: {key_hash} follows {previous_hash}"
                    )
                if batch_index >= record_batch_count:
                    raise ValueError(
                        f"Arrow batch lookup index '{index_path!s}' batch index {batch_index} is out of bounds "
                        f"at record {observed_count} for {record_batch_count} Arrow record batches"
                    )
                previous_hash = int(key_hash)
                observed_count += 1
        if observed_count != record_count:
            raise ValueError(
                f"Arrow batch lookup index '{index_path!s}' record count changed while validating: "
                f"expected {record_count}, observed {observed_count}"
            )

    index_stat_after = index_path.stat()
    if int(index_stat_before.st_size) != int(index_stat_after.st_size) or int(index_stat_before.st_mtime_ns) != int(
        index_stat_after.st_mtime_ns
    ):
        raise ValueError(f"Arrow batch lookup index changed while validating: {index_path!s}")
    if index_digest is not None and index_digest.hexdigest() != expected_index_sha256:
        raise ValueError(f"Arrow artifact generation index checksum mismatch: {index_path}")
    source_stat = arrow_path.stat()
    if source_size != int(source_stat.st_size) or source_mtime_ns != int(source_stat.st_mtime_ns):
        _raise_arrow_source_changed(arrow_path, context=context)
    return {
        "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
        "magic": str(header["magic"]),
        "record_count": record_count,
        "source_size": int(header["source_size"]),
        "key_column_hash": int(header["key_column_hash"]),
        "source_fingerprint": int(header["source_fingerprint"]),
    }


def _write_arrow_batch_lookup_sort_run(path: Path, records: list[tuple[int, int]]) -> None:
    """Sort and persist one temporary run in final-record binary form."""

    records.sort()
    buffer = bytearray()
    with path.open("wb") as output:
        for key_hash, batch_index in records:
            buffer.extend(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(key_hash, batch_index, 0))
            if len(buffer) >= _ARROW_BATCH_LOOKUP_WRITE_BUFFER_BYTES:
                output.write(buffer)
                buffer.clear()
        if buffer:
            output.write(buffer)
        output.flush()
        os.fsync(output.fileno())


def _iter_arrow_batch_lookup_sort_run(path: Path) -> Generator[tuple[int, int, int], None, None]:
    """Yield one exact sorted run in bounded chunks."""

    with path.open("rb") as source:
        while chunk := source.read(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_READ_BYTES):
            if len(chunk) % _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size != 0:
                raise ValueError(f"truncated Arrow batch lookup sort run: {path}")
            yield from _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.iter_unpack(chunk)


def _write_sorted_arrow_batch_lookup_records(
    output: Any,
    records: _ArrowBatchLookupRecords,
) -> None:
    """Write the exact sorted record body and verify its cardinality."""

    if len(records.run_paths) == 1:
        expected_size = records.row_count * _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
        observed_size = records.run_paths[0].stat().st_size
        if observed_size != expected_size:
            raise RuntimeError(f"Arrow batch lookup sort run has {observed_size} bytes, expected {expected_size}")
        with records.run_paths[0].open("rb") as source:
            shutil.copyfileobj(source, output, length=_ARROW_BATCH_LOOKUP_WRITE_BUFFER_BYTES)
        return

    run_iterators: list[Generator[tuple[int, int, int], None, None]] = []
    try:
        if records.run_paths:
            run_iterators = [_iter_arrow_batch_lookup_sort_run(path) for path in records.run_paths]
            sorted_records: Iterable[tuple[int, int, int]] = heapq.merge(*run_iterators)
        else:
            records.buffered_records.sort()
            sorted_records = ((key_hash, batch_index, 0) for key_hash, batch_index in records.buffered_records)

        written_records = 0
        buffer = bytearray()
        for key_hash, batch_index, reserved in sorted_records:
            buffer.extend(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(key_hash, batch_index, reserved))
            written_records += 1
            if len(buffer) >= _ARROW_BATCH_LOOKUP_WRITE_BUFFER_BYTES:
                output.write(buffer)
                buffer.clear()
        if buffer:
            output.write(buffer)
        if written_records != records.row_count:
            raise RuntimeError(
                f"Arrow batch lookup sort emitted {written_records} records, expected {records.row_count}"
            )
    finally:
        for run_iterator in run_iterators:
            run_iterator.close()


def _read_arrow_batch_lookup_records(
    arrow_path: Path,
    *,
    key_column: str,
    table_name: str,
    max_record_batch_rows: int | None,
    sort_run_dir: Path,
    max_records_in_memory: int,
) -> _ArrowBatchLookupRecords:
    import pyarrow as pa

    if max_records_in_memory < 1:
        raise ValueError("max_records_in_memory must be positive")
    buffered_records: list[tuple[int, int]] = []
    run_paths: list[Path] = []
    row_count = 0
    max_batch_rows = 0
    peak_buffered_records = 0

    def flush_run() -> None:
        run_path = sort_run_dir / f"run-{len(run_paths):06d}.bin"
        _write_arrow_batch_lookup_sort_run(run_path, buffered_records)
        run_paths.append(run_path)
        buffered_records.clear()

    with pa.memory_map(str(arrow_path), "r") as source:
        reader = pa.ipc.open_file(source)
        record_batch_count = int(reader.num_record_batches)
        key_column_index = reader.schema.get_field_index(key_column)
        if key_column_index < 0:
            raise KeyError(f"Arrow IPC file {arrow_path!s} is missing key column {key_column!r}")
        for batch_index in range(record_batch_count):
            batch = reader.get_batch(batch_index)
            batch_rows = int(batch.num_rows)
            max_batch_rows = max(max_batch_rows, batch_rows)
            _raise_if_record_batch_limit_exceeded(
                arrow_path=arrow_path,
                table_name=table_name,
                batch_index=batch_index,
                batch_rows=batch_rows,
                max_record_batch_rows=max_record_batch_rows,
            )
            keys = batch.column(key_column_index)
            row_count += batch_rows
            if keys.null_count:
                raise ValueError(
                    f"Arrow IPC file {arrow_path!s} contains null values in key column {key_column!r} "
                    f"for batch {batch_index}"
                )
            for offset in range(0, batch_rows, _ARROW_BATCH_LOOKUP_KEY_CHUNK_ROWS):
                key_chunk = keys.slice(offset, _ARROW_BATCH_LOOKUP_KEY_CHUNK_ROWS).to_pylist()
                key_hashes = _fnv64_utf8_batch([str(key) for key in key_chunk])
                for key_hash in key_hashes:
                    buffered_records.append((key_hash, batch_index))
                    peak_buffered_records = max(peak_buffered_records, len(buffered_records))
                    if len(buffered_records) == max_records_in_memory:
                        flush_run()
    if run_paths and buffered_records:
        flush_run()
    return _ArrowBatchLookupRecords(
        buffered_records=buffered_records,
        run_paths=run_paths,
        row_count=row_count,
        max_batch_rows=max_batch_rows,
        record_batch_count=record_batch_count,
        peak_buffered_records=peak_buffered_records,
    )


def write_arrow_batch_lookup_index(
    arrow_path: str | Path,
    index_path: str | Path,
    *,
    key_column: str,
    table_name: str = "arrow",
    max_record_batch_rows: int | None = None,
    overwrite: bool = True,
) -> tuple[str, dict[str, int | str | bool]]:
    """Write a Rust-readable key-hash to Arrow record-batch lookup index."""

    output_path = Path(index_path)
    if output_path.exists() and not overwrite:
        arrow_path_obj = Path(arrow_path)
        layout = arrow_ipc_physical_layout(arrow_path)
        _raise_if_record_batch_limit_exceeded(
            arrow_path=arrow_path,
            table_name=table_name,
            batch_index=-1,
            batch_rows=layout["actual_max_batch_rows"],
            max_record_batch_rows=max_record_batch_rows,
        )
        index_metrics = _validate_arrow_batch_lookup_index(
            arrow_path=arrow_path_obj,
            index_path=output_path,
            key_column=key_column,
            expected_row_count=int(layout["row_count"]),
        )
        return str(output_path), {
            "reused": True,
            "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
            **layout,
            "magic": index_metrics["magic"],
            "record_count": index_metrics["record_count"],
            "key_column_hash": index_metrics["key_column_hash"],
            "source_fingerprint": index_metrics["source_fingerprint"],
            "source_fingerprint_kind": "fnv1a64_full_file",
            "max_record_batch_rows": int(max_record_batch_rows or 0),
        }

    arrow_path_obj = Path(arrow_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: _ArrowBatchLookupRecords | None = None
    source_snapshot: _ArrowSourceSnapshot | None = None
    key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
    tmp_path: Path | None = None
    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=f".{output_path.name}.sort.",
    ) as sort_tmp_text:
        sort_tmp = Path(sort_tmp_text)
        for attempt in range(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS):
            source_stat_before = arrow_path_obj.stat()
            attempt_dir = sort_tmp / f"attempt-{attempt}"
            attempt_dir.mkdir()
            records = _read_arrow_batch_lookup_records(
                arrow_path_obj,
                key_column=key_column,
                table_name=table_name,
                max_record_batch_rows=max_record_batch_rows,
                sort_run_dir=attempt_dir,
                max_records_in_memory=_ARROW_BATCH_LOOKUP_SORT_BUFFER_RECORDS,
            )
            source_snapshot = _stable_source_file_snapshot(arrow_path_obj, context="building batch lookup index")
            if _source_snapshot_matches_stat(source_snapshot, source_stat_before):
                break
        else:
            _raise_arrow_source_changed(arrow_path_obj, context="building batch lookup index")
        if source_snapshot is None or records is None:
            raise AssertionError("source snapshot and sorted records must be populated")

        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as outfile:
                tmp_path = Path(outfile.name)
                outfile.write(
                    _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(
                        _ARROW_BATCH_LOOKUP_INDEX_MAGIC,
                        records.row_count,
                        source_snapshot.size,
                        key_column_hash,
                        source_snapshot.fingerprint,
                    )
                )
                _write_sorted_arrow_batch_lookup_records(outfile, records)
                outfile.flush()
                os.fsync(outfile.fileno())
            if not _source_snapshot_matches_stat(source_snapshot, arrow_path_obj.stat()):
                _raise_arrow_source_changed(arrow_path_obj, context="publishing batch lookup index")
            tmp_path.replace(output_path)
        except Exception:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
    return str(output_path), {
        "reused": False,
        "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
        "magic": _ARROW_BATCH_LOOKUP_INDEX_MAGIC.decode("ascii"),
        "row_count": records.row_count,
        "record_count": records.row_count,
        "key_column_hash": key_column_hash,
        "source_fingerprint": source_snapshot.fingerprint,
        "source_fingerprint_kind": "fnv1a64_full_file",
        "record_batch_count": records.record_batch_count,
        "actual_max_batch_rows": records.max_batch_rows,
        "max_record_batch_rows": int(max_record_batch_rows or 0),
    }


def write_raw_arrow_batch_lookup_indexes(
    paths: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> tuple[dict[str, str], dict[str, dict[str, int | str | bool]]]:
    """Write optional batch lookup indexes for raw Arrow planner inputs."""

    output_path = Path(output_dir) if output_dir is not None else None
    indexed_paths = normalize_arrow_paths(paths, omit_none=True)
    metrics: dict[str, dict[str, int | str | bool]] = {}
    for arrow_key, key_column in RAW_PLANNER_ARROW_KEY_COLUMNS.items():
        arrow_value = paths.get(arrow_key)
        if arrow_value is None:
            continue
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[arrow_key]
        batch_limit = _record_batch_limit_for_table(arrow_key, max_record_batch_rows)
        arrow_file_path = Path(str(arrow_value))
        index_file_path = (
            output_path / f"{arrow_file_path.stem}.{index_key}.bin"
            if output_path is not None
            else arrow_file_path.with_name(f"{arrow_file_path.stem}.{index_key}.bin")
        )
        index_file, index_metrics = write_arrow_batch_lookup_index(
            arrow_file_path,
            index_file_path,
            key_column=key_column,
            table_name=arrow_key,
            max_record_batch_rows=batch_limit,
            overwrite=overwrite,
        )
        indexed_paths[index_key] = index_file
        metrics[index_key] = index_metrics
    return indexed_paths, metrics


def raw_planner_arrow_physical_layout(
    paths: Mapping[str, Any],
    *,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
) -> dict[str, Any]:
    """Build manifest-ready physical-layout metadata for raw-planner Arrow inputs."""

    tables: dict[str, dict[str, int | str | bool]] = {}
    for table_name in RAW_PLANNER_ARROW_KEY_COLUMNS:
        path_value = paths.get(table_name)
        if path_value is None:
            continue
        batch_limit = _record_batch_limit_for_table(table_name, max_record_batch_rows)
        layout = arrow_ipc_physical_layout(path_value)
        _raise_if_record_batch_limit_exceeded(
            arrow_path=path_value,
            table_name=table_name,
            batch_index=-1,
            batch_rows=layout["actual_max_batch_rows"],
            max_record_batch_rows=batch_limit,
        )
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]
        index_path = paths.get(index_key)
        tables[table_name] = {
            "key": RAW_PLANNER_ARROW_KEY_COLUMNS[table_name],
            "max_record_batch_rows": int(batch_limit or 0),
            "batch_index_path_key": index_key,
            "batch_index_present": bool(index_path),
            **layout,
        }
    return {
        "schema": ARROW_PHYSICAL_LAYOUT_SCHEMA_VERSION,
        "optimized_for": "incremental_raw_candidate_planning",
        "tables": tables,
    }


def write_raw_planner_arrow_tables(
    tables: Mapping[str, Any],
    output_dir: str | Path,
    *,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Write raw-planner Arrow IPC tables and return paths keyed by table name."""

    output_path = Path(output_dir)
    paths: dict[str, str] = {}
    for name, table in tables.items():
        if (
            name in {"cluster_seeds", "cluster_seed_disallows"}
            and table.num_rows == 0
            and not include_empty_cluster_seeds
        ):
            continue
        path = output_path / f"{name}.arrow"
        if overwrite or not path.exists():
            write_arrow_ipc_table(
                table,
                path,
                max_record_batch_rows=_record_batch_limit_for_table(name, max_record_batch_rows),
            )
        paths[name] = str(path)
    return paths


def _validated_name_count_entry(kind: str, raw_name: Any, raw_count: Any) -> tuple[str, float]:
    """Return one strict name-count entry at the public artifact boundary."""

    if not isinstance(raw_name, str):
        raise TypeError(f"name-count {kind} keys must be strings, got {type(raw_name).__name__}: {raw_name!r}")
    if raw_name != canonicalize_name_text(raw_name):
        raise ValueError(f"name-count {kind} key {raw_name!r} must be canonical_v2 normalized")
    if kind == "first":
        structurally_valid = len(raw_name) > 1
    elif kind == "last":
        structurally_valid = bool(raw_name)
    elif kind == "first_last":
        first, separator, _last = raw_name.rpartition(" ")
        structurally_valid = bool(separator) and len(first) > 1
    elif kind == "last_first_initial":
        last, separator, initial = raw_name.rpartition(" ")
        structurally_valid = bool(last and separator) and len(initial) == 1
    else:
        raise ValueError(f"unknown name-count kind: {kind!r}")
    if not structurally_valid:
        raise ValueError(f"name-count {kind} key {raw_name!r} does not satisfy the canonical_v2 key contract")
    try:
        count: float = float(raw_count)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"name-count {kind} value for {raw_name!r} must be a finite positive number, got {raw_count!r}"
        ) from error
    if not math.isfinite(count) or count <= 0.0:
        raise ValueError(
            f"name-count {kind} value for {raw_name!r} must be a finite positive number, got {raw_count!r}"
        )
    return raw_name, count


def _fnv64_update(digest: int, value: bytes) -> int:
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _fnv64_bytes(value: bytes) -> int:
    return _fnv64_update(_FNV64_OFFSET, value)


def _fnv64_utf8_batch(values: Sequence[str]) -> list[int]:
    """Hash UTF-8 keys through the native batch boundary."""

    from s2and.runtime import load_s2and_rust_extension

    return [int(value) for value in load_s2and_rust_extension().fnv64_utf8_batch(list(values))]


def _source_file_fingerprint_once(path: Path, *, source_size: int) -> int:
    from s2and.runtime import load_s2and_rust_extension

    _sha256, fingerprint = load_s2and_rust_extension().arrow_source_file_digests(
        str(path),
        int(source_size),
        False,
    )
    return int(fingerprint)


def _source_file_digests_once(path: Path, *, source_size: int) -> tuple[str, int]:
    from s2and.runtime import load_s2and_rust_extension

    sha256, fingerprint = load_s2and_rust_extension().arrow_source_file_digests(
        str(path),
        int(source_size),
        True,
    )
    if not isinstance(sha256, str):  # pragma: no cover - native return contract
        raise RuntimeError("native Arrow source digest omitted requested SHA-256")
    return sha256, int(fingerprint)


def _source_file_sha256_once(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        while chunk := infile.read(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_READ_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_source_file_sha256(path: Path, *, context: str) -> _ArrowSourceDigests:
    for _attempt in range(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS):
        before = path.stat()
        sha256 = _source_file_sha256_once(path)
        after = path.stat()
        if int(before.st_size) == int(after.st_size) and int(before.st_mtime_ns) == int(after.st_mtime_ns):
            return _ArrowSourceDigests(
                size=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                sha256=sha256,
                fingerprint=None,
            )
    _raise_arrow_source_changed(path, context=context)


def _stable_source_file_digests(path: Path, *, context: str) -> _ArrowSourceDigests:
    for _attempt in range(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS):
        before = path.stat()
        sha256, fingerprint = _source_file_digests_once(path, source_size=int(before.st_size))
        after = path.stat()
        if int(before.st_size) == int(after.st_size) and int(before.st_mtime_ns) == int(after.st_mtime_ns):
            return _ArrowSourceDigests(
                size=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                sha256=sha256,
                fingerprint=int(fingerprint),
            )
    _raise_arrow_source_changed(path, context=context)


def _source_digests_match_stat(digests: _ArrowSourceDigests, stat_result: os.stat_result) -> bool:
    return digests.size == int(stat_result.st_size) and digests.mtime_ns == int(stat_result.st_mtime_ns)


def _stable_source_file_snapshot(path: Path, *, context: str) -> _ArrowSourceSnapshot:
    for _attempt in range(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS):
        before = path.stat()
        fingerprint = _source_file_fingerprint_once(path, source_size=int(before.st_size))
        after = path.stat()
        if int(before.st_size) == int(after.st_size) and int(before.st_mtime_ns) == int(after.st_mtime_ns):
            return _ArrowSourceSnapshot(
                size=int(after.st_size),
                mtime_ns=int(after.st_mtime_ns),
                fingerprint=int(fingerprint),
            )
    _raise_arrow_source_changed(path, context=context)


def _source_file_fingerprint(path: Path) -> int:
    return _stable_source_file_snapshot(path, context="fingerprinting source file").fingerprint


def _name_counts_index_hashes(kind: str, name_bytes: bytes) -> tuple[int, int]:
    return (
        _fnv64_bytes(name_bytes),
        _fnv64_update(_name_counts_index_kind_hash_seed(kind), name_bytes),
    )


def _name_counts_index_kind_hash_seed(kind: str) -> int:
    """Return the reusable second-hash seed for one index kind."""

    return _fnv64_bytes(_NAME_COUNTS_INDEX_HASH_DOMAIN + kind.encode("utf-8") + b"\x00")


def _write_name_count_sort_run(path: Path, records: list[tuple[int, int, bytes, float]]) -> int:
    records.sort(key=lambda item: (item[0], item[1], item[2]))
    buffer = bytearray()
    with path.open("wb") as output:
        for hash_1, hash_2, name_bytes, count in records:
            buffer.extend(_NAME_COUNTS_SORT_RUN_RECORD_STRUCT.pack(hash_1, hash_2, count, len(name_bytes)))
            buffer.extend(name_bytes)
            if len(buffer) >= _NAME_COUNTS_WRITE_BUFFER_BYTES:
                output.write(buffer)
                buffer.clear()
        if buffer:
            output.write(buffer)
        output.flush()
        os.fsync(output.fileno())
    return path.stat().st_size


def _iter_name_count_sort_run(path: Path) -> Generator[tuple[int, int, bytes, float], None, None]:
    with path.open("rb") as source:
        while header := source.read(_NAME_COUNTS_SORT_RUN_RECORD_STRUCT.size):
            if len(header) != _NAME_COUNTS_SORT_RUN_RECORD_STRUCT.size:
                raise ValueError(f"truncated name-count sort run header: {path}")
            hash_1, hash_2, count, name_length = _NAME_COUNTS_SORT_RUN_RECORD_STRUCT.unpack(header)
            name_bytes = source.read(name_length)
            if len(name_bytes) != name_length:
                raise ValueError(f"truncated name-count sort run name: {path}")
            yield hash_1, hash_2, name_bytes, count


def _write_sorted_name_count_records(
    path: Path,
    records: Iterable[tuple[int, int, bytes, float]],
    *,
    record_count: int,
) -> int:
    record_fd, record_tmp_text = tempfile.mkstemp(prefix=f".{path.name}.records.", dir=str(path.parent))
    blob_fd, blob_tmp_text = tempfile.mkstemp(prefix=f".{path.name}.blob.", dir=str(path.parent))
    os.close(record_fd)
    os.close(blob_fd)
    record_tmp = Path(record_tmp_text)
    blob_tmp = Path(blob_tmp_text)
    output_tmp = path.parent / f".{path.name}.tmp"
    written_records = 0
    blob_size = 0
    record_buffer = bytearray()
    blob_buffer = bytearray()
    previous_sort_key: tuple[int, int, bytes] | None = None
    try:
        with record_tmp.open("wb") as record_output, blob_tmp.open("wb") as blob_output:
            for hash_1, hash_2, name_bytes, count in records:
                sort_key = (hash_1, hash_2, name_bytes)
                if sort_key == previous_sort_key:
                    raise ValueError(f"name-count index contains duplicate UTF-8 name {name_bytes.decode('utf-8')!r}")
                previous_sort_key = sort_key
                record_buffer.extend(
                    _NAME_COUNTS_INDEX_RECORD_STRUCT.pack(
                        hash_1,
                        hash_2,
                        blob_size,
                        len(name_bytes),
                        0,
                        count,
                    )
                )
                blob_buffer.extend(name_bytes)
                blob_size += len(name_bytes)
                written_records += 1
                if len(record_buffer) >= _NAME_COUNTS_WRITE_BUFFER_BYTES:
                    record_output.write(record_buffer)
                    record_buffer.clear()
                if len(blob_buffer) >= _NAME_COUNTS_WRITE_BUFFER_BYTES:
                    blob_output.write(blob_buffer)
                    blob_buffer.clear()
            if record_buffer:
                record_output.write(record_buffer)
            if blob_buffer:
                blob_output.write(blob_buffer)
            record_output.flush()
            blob_output.flush()
            os.fsync(record_output.fileno())
            os.fsync(blob_output.fileno())
        if written_records != record_count:
            raise RuntimeError(f"name-count sort emitted {written_records} records, expected {record_count}")
        blob_offset = _NAME_COUNTS_INDEX_HEADER_STRUCT.size + (record_count * _NAME_COUNTS_INDEX_RECORD_STRUCT.size)
        with output_tmp.open("wb") as output:
            output.write(
                _NAME_COUNTS_INDEX_HEADER_STRUCT.pack(
                    _NAME_COUNTS_INDEX_MAGIC,
                    record_count,
                    blob_offset,
                    blob_size,
                )
            )
            with record_tmp.open("rb") as record_source:
                shutil.copyfileobj(record_source, output, length=_NAME_COUNTS_WRITE_BUFFER_BYTES)
            with blob_tmp.open("rb") as blob_source:
                shutil.copyfileobj(blob_source, output, length=_NAME_COUNTS_WRITE_BUFFER_BYTES)
            output.flush()
            os.fsync(output.fileno())
        output_tmp.replace(path)
        return record_tmp.stat().st_size + blob_tmp.stat().st_size
    finally:
        for temporary_path in (record_tmp, blob_tmp, output_tmp):
            temporary_path.unlink(missing_ok=True)


def _write_name_count_index_file(
    path: Path,
    kind: str,
    mapping: Mapping[Any, Any],
    *,
    max_records_in_memory: int = _NAME_COUNTS_SORT_BUFFER_RECORDS,
) -> dict[str, int]:
    """Write one exact index using bounded-memory sorted runs."""

    if max_records_in_memory < 1:
        raise ValueError("max_records_in_memory must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    record_count = len(mapping)
    buffered: list[tuple[int, int, bytes, float]] = []
    run_paths: list[Path] = []
    run_iterators: list[Generator[tuple[int, int, bytes, float], None, None]] = []
    run_bytes = 0
    peak_buffered_records = 0
    kind_hash_seed = _name_counts_index_kind_hash_seed(kind)
    try:
        for raw_name, raw_count in mapping.items():
            name, count = _validated_name_count_entry(kind, raw_name, raw_count)
            name_bytes = name.encode("utf-8")
            hash_1 = _fnv64_bytes(name_bytes)
            hash_2 = _fnv64_update(kind_hash_seed, name_bytes)
            buffered.append((hash_1, hash_2, name_bytes, count))
            peak_buffered_records = max(peak_buffered_records, len(buffered))
            if len(buffered) >= max_records_in_memory and record_count > max_records_in_memory:
                run_path = path.parent / f".{path.name}.run.{len(run_paths)}"
                run_bytes += _write_name_count_sort_run(run_path, buffered)
                run_paths.append(run_path)
                buffered = []

        if run_paths:
            if buffered:
                run_path = path.parent / f".{path.name}.run.{len(run_paths)}"
                run_bytes += _write_name_count_sort_run(run_path, buffered)
                run_paths.append(run_path)
                buffered = []
            run_iterators = [_iter_name_count_sort_run(run_path) for run_path in run_paths]
            sorted_records: Iterable[tuple[int, int, bytes, float]] = heapq.merge(
                *run_iterators,
                key=lambda item: (item[0], item[1], item[2]),
            )
        else:
            buffered.sort(key=lambda item: (item[0], item[1], item[2]))
            sorted_records = buffered
        assembly_tmp_bytes = _write_sorted_name_count_records(
            path,
            sorted_records,
            record_count=record_count,
        )
    finally:
        for run_iterator in run_iterators:
            run_iterator.close()
        for run_path in run_paths:
            run_path.unlink(missing_ok=True)
    return {
        "record_count": record_count,
        "byte_count": path.stat().st_size,
        "sort_run_count": len(run_paths),
        "peak_buffered_records": peak_buffered_records,
        "temporary_byte_count": run_bytes + assembly_tmp_bytes,
    }


def write_name_counts_index(
    output_dir: str | Path,
    mappings: tuple[Mapping[Any, Any], Mapping[Any, Any], Mapping[Any, Any], Mapping[Any, Any]],
) -> tuple[str, dict[str, int]]:
    """Publish one new flat global name-count index into an absent target."""

    output_path = Path(output_dir)
    index_dir = output_path / "name_counts_index"
    if index_dir.exists():
        raise FileExistsError(f"name-count index target already exists: {index_dir}")
    output_path.mkdir(parents=True, exist_ok=True)

    first_dict, last_dict, first_last_dict, last_first_initial_dict = mappings
    named_mappings = (
        ("first", first_dict),
        ("last", last_dict),
        ("first_last", first_last_dict),
        ("last_first_initial", last_first_initial_dict),
    )
    metrics: dict[str, int] = {}
    total_records = 0
    total_bytes = 0
    manifest_files: dict[str, dict[str, int | str]] = {}
    temporary_index_dir = Path(tempfile.mkdtemp(prefix=".name_counts_index.", dir=str(output_path)))
    try:
        for kind, mapping in named_mappings:
            filename = f"{kind}.bin"
            index_file = temporary_index_dir / filename
            file_metrics = _write_name_count_index_file(
                index_file,
                kind,
                mapping,
            )
            record_count = file_metrics["record_count"]
            byte_count = file_metrics["byte_count"]
            metrics[f"{kind}_count"] = record_count
            metrics[f"{kind}_bytes"] = byte_count
            total_records += record_count
            total_bytes += byte_count
            manifest_files[kind] = {
                "path": filename,
                "byte_count": byte_count,
                "sha256": _sha256_file(index_file),
            }
            metrics[f"{kind}_sort_run_count"] = file_metrics["sort_run_count"]
            metrics[f"{kind}_peak_buffered_records"] = file_metrics["peak_buffered_records"]
            metrics[f"{kind}_temporary_bytes"] = file_metrics["temporary_byte_count"]

        manifest = {
            "schema_version": NAME_COUNTS_INDEX_SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "files": manifest_files,
        }
        manifest_path = temporary_index_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with manifest_path.open("r+b") as manifest_input:
            os.fsync(manifest_input.fileno())
        fsync_directory(temporary_index_dir)
        if index_dir.exists():
            raise FileExistsError(f"name-count index target already exists: {index_dir}")
        temporary_index_dir.rename(index_dir)
        fsync_directory(output_path)
    finally:
        if temporary_index_dir.exists():
            shutil.rmtree(temporary_index_dir)
    metrics["row_count"] = total_records
    metrics["byte_count"] = total_bytes
    return str(index_dir), metrics
