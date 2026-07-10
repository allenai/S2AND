"""FeatureBlock Arrow IPC, sidecar, and artifact IO helpers."""

from __future__ import annotations

import hashlib
import heapq
import json
import mmap
import os
import shutil
import struct
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.arrow_inputs import (
    RAW_PLANNER_ARROW_BATCH_INDEX_KEYS,
    RAW_PLANNER_ARROW_KEY_COLUMNS,
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    normalize_arrow_paths,
)
from s2and.incremental_linking.feature_block_contract import (
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    _feature_block_specter_from_mapping,
    _optional_bool,
    _optional_int,
    _optional_str,
    _strict_string_tuple,
    feature_block_signature_order_from_raw_candidate_plan,
    filter_cluster_seed_disallows_for_signature_subset,
    normalize_cluster_seed_disallow_pairs,
)

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"
NAME_COUNTS_ARROW_MANIFEST_SCHEMA_VERSION = "name_counts_arrow_v1"
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
    _require_arrow_string_columns(
        table,
        "incremental query signatures",
        {"signature_id", "query_view", "query_author"},
    )
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
    missing_columns = sorted({"signature_id", "cluster_id"} - set(table.column_names))
    if missing_columns:
        raise ValueError(f"cluster seeds Arrow is missing required columns: {missing_columns}")
    _require_arrow_string_columns(table, "cluster seeds", {"signature_id", "cluster_id"})
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
    missing_columns = sorted({"signature_id_1", "signature_id_2"} - set(table.column_names))
    if missing_columns:
        raise ValueError(f"cluster seed disallows Arrow is missing required columns: {missing_columns}")
    _require_arrow_string_columns(table, "cluster seed disallows", {"signature_id_1", "signature_id_2"})
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
    _require_arrow_string_columns(table, "altered cluster signatures", {"signature_id"})
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


def cluster_seed_disallows_path_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> Path | None:
    """Return the explicit cluster-seed disallow sidecar path, if configured."""

    if arrow_paths is None:
        return None
    path_value = arrow_paths.get("cluster_seed_disallows")
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Arrow path cluster_seed_disallows={path} does not exist")
    return path


def cluster_seed_disallows_from_arrow_paths(arrow_paths: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    """Read explicit Arrow cluster-seed disallows, failing on stale configured paths."""

    path = cluster_seed_disallows_path_from_arrow_paths(arrow_paths)
    if path is None:
        return set()
    return set(read_cluster_seed_disallows_arrow(path))


@contextmanager
def temporary_arrow_paths_with_cluster_seeds(
    arrow_paths: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    *,
    prefix: str,
    reuse_existing_cluster_seeds_when_empty: bool = False,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
) -> Iterator[dict[str, str]]:
    """Yield Arrow paths with request-scoped cluster-seed sidecars."""

    paths = normalize_arrow_paths(arrow_paths)
    if (
        reuse_existing_cluster_seeds_when_empty
        and not cluster_seeds_require
        and cluster_seeds_disallow is None
        and paths.get("cluster_seeds") is not None
        and Path(paths["cluster_seeds"]).exists()
    ):
        yield paths
        return

    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmpdir_path = Path(tmpdir)
        reusable_cluster_seeds_path = (
            reuse_existing_cluster_seeds_when_empty
            and not cluster_seeds_require
            and paths.get("cluster_seeds") is not None
            and Path(paths["cluster_seeds"]).exists()
        )
        if not reusable_cluster_seeds_path:
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


def _read_arrow_batch_lookup_index_header(index_path: Path) -> dict[str, int | str]:
    with index_path.open("rb") as infile:
        header = infile.read(_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size)
    return _decode_arrow_batch_lookup_index_header(index_path, header)


def _batch_lookup_index_source_mismatch(
    header: Mapping[str, int | str],
    *,
    source_size: int,
    source_fingerprint: int,
) -> str | None:
    indexed_size = int(header["source_size"])
    indexed_fingerprint = int(header["source_fingerprint"])
    if indexed_size == int(source_size) and indexed_fingerprint == int(source_fingerprint):
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
    """Validate an existing batch lookup index without scanning Arrow record batches."""

    arrow_path_obj = Path(arrow_path)
    index_path_obj = Path(index_path)
    header = _read_arrow_batch_lookup_index_header(index_path_obj)
    source_snapshot = _stable_source_file_snapshot(arrow_path_obj, context="validating batch lookup index")
    key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
    if int(header["key_column_hash"]) != key_column_hash:
        raise ValueError(
            f"Arrow batch lookup index '{index_path_obj!s}' was built for a different key column: "
            f"indexed hash={int(header['key_column_hash'])} expected hash={key_column_hash} "
            f"key_column={key_column!r}"
        )
    source_mismatch = _batch_lookup_index_source_mismatch(
        header,
        source_size=source_snapshot.size,
        source_fingerprint=source_snapshot.fingerprint,
    )
    if source_mismatch is not None:
        raise ValueError(
            f"Arrow batch lookup index '{index_path_obj!s}' is stale for '{arrow_path_obj!s}': " f"{source_mismatch}"
        )
    if expected_row_count is not None and int(header["record_count"]) != int(expected_row_count):
        raise ValueError(
            f"Arrow batch lookup index row count mismatch for {arrow_path_obj!s}: "
            f"index has {int(header['record_count'])} records, expected {int(expected_row_count)}. "
            "Rebuild it with overwrite=True."
        )
    return {
        "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
        "magic": str(header["magic"]),
        "record_count": int(header["record_count"]),
        "source_size": int(header["source_size"]),
        "key_column_hash": int(header["key_column_hash"]),
        "source_fingerprint": int(header["source_fingerprint"]),
    }


def _read_arrow_batch_lookup_records(
    arrow_path: Path,
    *,
    key_column: str,
    table_name: str,
    max_record_batch_rows: int | None,
) -> tuple[list[tuple[int, int]], int, int, int]:
    import pyarrow as pa

    records: list[tuple[int, int]] = []
    row_count = 0
    max_batch_rows = 0
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
            keys = batch.column(key_column_index).to_pylist()
            row_count += len(keys)
            if any(key is None for key in keys):
                raise ValueError(
                    f"Arrow IPC file {arrow_path!s} contains null values in key column {key_column!r} "
                    f"for batch {batch_index}"
                )
            records.extend((_fnv64_bytes(str(key).encode("utf-8")), batch_index) for key in keys)
    return records, row_count, max_batch_rows, record_batch_count


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
        index_header = _read_arrow_batch_lookup_index_header(output_path)
        source_snapshot = _stable_source_file_snapshot(arrow_path_obj, context="validating reusable batch lookup index")
        key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
        source_mismatch = _batch_lookup_index_source_mismatch(
            index_header,
            source_size=source_snapshot.size,
            source_fingerprint=source_snapshot.fingerprint,
        )
        if int(index_header["key_column_hash"]) != key_column_hash or source_mismatch is not None:
            raise ValueError(
                f"Arrow batch lookup index is stale for {arrow_path_obj!s}: {output_path!s}. "
                "Rebuild it with overwrite=True."
            )
        layout_stat = arrow_path_obj.stat()
        layout = arrow_ipc_physical_layout(arrow_path)
        if not _source_snapshot_matches_stat(source_snapshot, layout_stat) or not _source_snapshot_matches_stat(
            source_snapshot,
            arrow_path_obj.stat(),
        ):
            _raise_arrow_source_changed(arrow_path_obj, context="validating reusable batch lookup index")
        _raise_if_record_batch_limit_exceeded(
            arrow_path=arrow_path,
            table_name=table_name,
            batch_index=-1,
            batch_rows=layout["actual_max_batch_rows"],
            max_record_batch_rows=max_record_batch_rows,
        )
        if int(index_header["record_count"]) != int(layout["row_count"]):
            raise ValueError(
                f"Arrow batch lookup index row count mismatch for {arrow_path_obj!s}: "
                f"index has {int(index_header['record_count'])} records, "
                f"Arrow file has {int(layout['row_count'])} rows. Rebuild it with overwrite=True."
            )
        return str(output_path), {
            "reused": True,
            "schema_version": ARROW_BATCH_LOOKUP_INDEX_SCHEMA_VERSION,
            **layout,
            "magic": str(index_header["magic"]),
            "record_count": int(index_header["record_count"]),
            "key_column_hash": int(index_header["key_column_hash"]),
            "source_fingerprint": int(index_header["source_fingerprint"]),
            "source_fingerprint_kind": "fnv1a64_full_file",
            "max_record_batch_rows": int(max_record_batch_rows or 0),
        }

    arrow_path_obj = Path(arrow_path)
    records: list[tuple[int, int]] = []
    row_count = 0
    max_batch_rows = 0
    record_batch_count = 0
    source_snapshot: _ArrowSourceSnapshot | None = None
    for _attempt in range(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_SNAPSHOT_ATTEMPTS):
        source_stat_before = arrow_path_obj.stat()
        records, row_count, max_batch_rows, record_batch_count = _read_arrow_batch_lookup_records(
            arrow_path_obj,
            key_column=key_column,
            table_name=table_name,
            max_record_batch_rows=max_record_batch_rows,
        )
        source_snapshot = _stable_source_file_snapshot(arrow_path_obj, context="building batch lookup index")
        if _source_snapshot_matches_stat(source_snapshot, source_stat_before):
            break
    else:
        _raise_arrow_source_changed(arrow_path_obj, context="building batch lookup index")
    if source_snapshot is None:
        raise AssertionError("source snapshot must be populated")

    records.sort()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    key_column_hash = _fnv64_bytes(str(key_column).encode("utf-8"))
    tmp_path: Path | None = None
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
                    len(records),
                    source_snapshot.size,
                    key_column_hash,
                    source_snapshot.fingerprint,
                )
            )
            for key_hash, batch_index in records:
                outfile.write(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(key_hash, batch_index, 0))
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
        "row_count": row_count,
        "record_count": len(records),
        "key_column_hash": key_column_hash,
        "source_fingerprint": source_snapshot.fingerprint,
        "source_fingerprint_kind": "fnv1a64_full_file",
        "record_batch_count": record_batch_count,
        "actual_max_batch_rows": max_batch_rows,
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


def write_feature_block_arrow_tables(
    feature_block: FeatureBlock,
    output_dir: str | Path,
    *,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Write `FeatureBlock` Arrow IPC tables and return paths keyed by table name."""

    output_path = Path(output_dir)
    tables = feature_block.to_arrow_tables()
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


def write_name_counts_arrow(output_dir: str | Path, *, overwrite: bool = False) -> tuple[str, dict[str, int | bool]]:
    """Publish the global name-count Arrow table as an immutable generation."""

    import pyarrow as pa

    from s2and.data import _load_name_counts_artifact, _validated_name_counts_provenance

    output_root = Path(output_dir)
    logical_output_path = output_root / "name_counts.arrow"
    counts_payload, raw_source_provenance = _load_name_counts_artifact()
    source_provenance = _validated_name_counts_provenance(
        raw_source_provenance,
        context="write_name_counts_arrow",
    )
    first_dict, last_dict, first_last_dict, last_first_initial_dict = counts_payload
    fingerprint = _name_counts_arrow_fingerprint(
        {
            "first": first_dict,
            "last": last_dict,
            "first_last": first_last_dict,
            "last_first_initial": last_first_initial_dict,
        }
    )
    expected_manifest = {
        "schema_version": NAME_COUNTS_ARROW_MANIFEST_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "normalization_version": source_provenance["normalization_version"],
        "source_provenance": source_provenance,
    }
    if not overwrite:
        current_path = _verified_name_counts_arrow_generation(logical_output_path, expected_manifest)
        if current_path is not None:
            return str(current_path), {"reused": True}

    metrics: dict[str, int | bool] = {"reused": False}
    generations_dir = output_root / "name_counts_arrow_generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation_name = f"gen-{uuid.uuid4().hex}"
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{generation_name}.", dir=str(generations_dir)))
    generation_dir = generations_dir / generation_name
    output_tmp = staging_dir / "name_counts.arrow"
    manifest_path = _name_artifact_manifest_path(logical_output_path)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{generation_name}.tmp")
    schema = pa.schema((pa.field("kind", pa.string()), pa.field("name", pa.string()), pa.field("count", pa.float64())))
    row_count = 0
    batch_size = 100_000
    try:
        with pa.OSFile(str(output_tmp), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for kind, mapping in (
                    ("first", first_dict),
                    ("last", last_dict),
                    ("first_last", first_last_dict),
                    ("last_first_initial", last_first_initial_dict),
                ):
                    metrics[f"{kind}_count"] = len(mapping)
                    names: list[str] = []
                    values: list[float] = []
                    for raw_name, raw_count in mapping.items():
                        names.append(str(raw_name))
                        values.append(float(raw_count))
                        if len(names) >= batch_size:
                            writer.write_batch(
                                pa.record_batch(
                                    [
                                        pa.array([kind] * len(names)),
                                        pa.array(names),
                                        pa.array(values, type=pa.float64()),
                                    ],
                                    schema=schema,
                                )
                            )
                            row_count += len(names)
                            names.clear()
                            values.clear()
                    if names:
                        writer.write_batch(
                            pa.record_batch(
                                [pa.array([kind] * len(names)), pa.array(names), pa.array(values, type=pa.float64())],
                                schema=schema,
                            )
                        )
                        row_count += len(names)
        with output_tmp.open("r+b") as output_input:
            os.fsync(output_input.fileno())
        arrow_sha256 = _sha256_file(output_tmp)
        arrow_byte_count = output_tmp.stat().st_size
        manifest = {
            **expected_manifest,
            "generation_id": generation_name,
            "row_count": row_count,
            "files": {
                "arrow": {
                    "path": f"name_counts_arrow_generations/{generation_name}/name_counts.arrow",
                    "byte_count": arrow_byte_count,
                    "sha256": arrow_sha256,
                }
            },
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with manifest_tmp.open("r+b") as manifest_input:
            os.fsync(manifest_input.fileno())
        marker_path = staging_dir / ".published"
        with marker_path.open("wb") as marker_output:
            marker_output.flush()
            os.fsync(marker_output.fileno())
        with _exclusive_name_counts_publish_lock(output_root):
            if not overwrite:
                current_path = _verified_name_counts_arrow_generation(logical_output_path, expected_manifest)
                if current_path is not None:
                    return str(current_path), {"reused": True}
            staging_dir.rename(generation_dir)
            manifest_tmp.replace(manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    metrics["row_count"] = row_count
    return str(generation_dir / "name_counts.arrow"), metrics


def _name_artifact_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.manifest.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_name_counts_arrow_generation(
    logical_output_path: Path,
    expected: Mapping[str, Any],
) -> Path | None:
    manifest_path = _name_artifact_manifest_path(logical_output_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        return None
    if not all(manifest.get(key) == value for key, value in expected.items()):
        return None
    files = manifest.get("files")
    arrow_entry = files.get("arrow") if isinstance(files, Mapping) else None
    if not isinstance(arrow_entry, Mapping):
        return None
    raw_path = arrow_entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    arrow_path = (logical_output_path.parent / raw_path).resolve()
    try:
        arrow_path.relative_to(logical_output_path.parent.resolve())
    except ValueError:
        return None
    if not arrow_path.is_file() or not (arrow_path.parent / ".published").is_file():
        return None
    expected_bytes = arrow_entry.get("byte_count")
    expected_sha256 = arrow_entry.get("sha256")
    if not isinstance(expected_bytes, int) or arrow_path.stat().st_size != expected_bytes:
        return None
    if not isinstance(expected_sha256, str) or _sha256_file(arrow_path) != expected_sha256:
        return None
    return arrow_path


def _fnv64_text(digest: int, value: str) -> int:
    raw = value.encode("utf-8")
    digest = _fnv64_update(digest, len(raw).to_bytes(8, "little", signed=False))
    return _fnv64_update(digest, raw)


def _name_counts_arrow_fingerprint(mappings: Mapping[str, Mapping[Any, Any]]) -> int:
    digest = _fnv64_bytes(b"s2and-name-counts-arrow-v2-order-independent\x00")
    for kind, mapping in sorted(mappings.items()):
        xor_accumulator = 0
        sum_accumulator = 0
        square_accumulator = 0
        kind_seed = _fnv64_text(_fnv64_bytes(b"s2and-name-count-entry-v1\x00"), kind)
        for raw_name, raw_count in mapping.items():
            entry_digest = _fnv64_text(kind_seed, str(raw_name))
            entry_digest = _fnv64_text(entry_digest, float(raw_count).hex())
            xor_accumulator ^= entry_digest
            sum_accumulator = (sum_accumulator + entry_digest) & 0xFFFFFFFFFFFFFFFF
            square_accumulator = (square_accumulator + (entry_digest * entry_digest)) & 0xFFFFFFFFFFFFFFFF
        digest = _fnv64_text(digest, kind)
        digest = _fnv64_update(digest, len(mapping).to_bytes(8, "little", signed=False))
        digest = _fnv64_update(digest, xor_accumulator.to_bytes(8, "little", signed=False))
        digest = _fnv64_update(digest, sum_accumulator.to_bytes(8, "little", signed=False))
        digest = _fnv64_update(digest, square_accumulator.to_bytes(8, "little", signed=False))
    return digest


def _fnv64_update(digest: int, value: bytes) -> int:
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _fnv64_bytes(value: bytes) -> int:
    return _fnv64_update(_FNV64_OFFSET, value)


def _source_file_fingerprint_once(path: Path, *, source_size: int) -> int:
    digest = _fnv64_bytes(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_HASH_DOMAIN)
    digest = _fnv64_update(digest, int(source_size).to_bytes(8, "little", signed=False))
    with path.open("rb") as infile:
        while True:
            chunk = infile.read(_ARROW_BATCH_LOOKUP_INDEX_SOURCE_READ_BYTES)
            if not chunk:
                break
            digest = _fnv64_update(digest, chunk)
    return digest


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


def _iter_name_count_sort_run(path: Path) -> Iterator[tuple[int, int, bytes, float]]:
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
    output_tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    written_records = 0
    blob_size = 0
    record_buffer = bytearray()
    blob_buffer = bytearray()
    try:
        with record_tmp.open("wb") as record_output, blob_tmp.open("wb") as blob_output:
            for hash_1, hash_2, name_bytes, count in records:
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
    run_bytes = 0
    peak_buffered_records = 0
    kind_hash_seed = _name_counts_index_kind_hash_seed(kind)
    try:
        for raw_name, raw_count in mapping.items():
            name_bytes = str(raw_name).encode("utf-8")
            hash_1 = _fnv64_bytes(name_bytes)
            hash_2 = _fnv64_update(kind_hash_seed, name_bytes)
            buffered.append((hash_1, hash_2, name_bytes, float(raw_count)))
            peak_buffered_records = max(peak_buffered_records, len(buffered))
            if len(buffered) >= max_records_in_memory and record_count > max_records_in_memory:
                run_path = path.parent / f".{path.name}.run.{len(run_paths)}.{uuid.uuid4().hex}"
                run_bytes += _write_name_count_sort_run(run_path, buffered)
                run_paths.append(run_path)
                buffered = []

        if run_paths:
            if buffered:
                run_path = path.parent / f".{path.name}.run.{len(run_paths)}.{uuid.uuid4().hex}"
                run_bytes += _write_name_count_sort_run(run_path, buffered)
                run_paths.append(run_path)
                buffered = []
            sorted_records: Iterable[tuple[int, int, bytes, float]] = heapq.merge(
                *(_iter_name_count_sort_run(run_path) for run_path in run_paths),
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
        for run_path in run_paths:
            run_path.unlink(missing_ok=True)
    return {
        "record_count": record_count,
        "byte_count": path.stat().st_size,
        "sort_run_count": len(run_paths),
        "peak_buffered_records": peak_buffered_records,
        "temporary_byte_count": run_bytes + assembly_tmp_bytes,
    }


def _name_count_index_disk_requirement(mapping: Mapping[Any, Any]) -> int:
    """Return a conservative peak disk requirement before any temporary write."""

    record_count = len(mapping)
    # UTF-8 uses at most four bytes per Unicode code point. This avoids
    # retaining encoded key copies during the preflight.
    maximum_name_bytes = sum(4 * len(str(raw_name)) for raw_name in mapping)
    sort_run_bytes = record_count * _NAME_COUNTS_SORT_RUN_RECORD_STRUCT.size + maximum_name_bytes
    assembly_bytes = record_count * _NAME_COUNTS_INDEX_RECORD_STRUCT.size + maximum_name_bytes
    final_bytes = _NAME_COUNTS_INDEX_HEADER_STRUCT.size + assembly_bytes
    return sort_run_bytes + assembly_bytes + final_bytes + (4 * _NAME_COUNTS_WRITE_BUFFER_BYTES)


def _name_counts_index_manifest_paths(index_dir: Path) -> dict[str, Path] | None:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError(f"name-count index manifest must contain an object: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"name-count index manifest is missing files: {manifest_path}")
    resolved: dict[str, Path] = {}
    for kind in ("first", "last", "first_last", "last_first_initial"):
        entry = files.get(kind)
        if not isinstance(entry, Mapping) or entry.get("path") is None:
            raise ValueError(f"name-count index manifest is missing files.{kind}.path: {manifest_path}")
        raw_path = Path(str(entry["path"]))
        resolved[kind] = raw_path if raw_path.is_absolute() else index_dir / raw_path
    return resolved


def _name_counts_index_complete(
    index_dir: Path,
    *,
    expected_fingerprint: int,
    expected_source_provenance: Mapping[str, Any],
) -> bool:
    manifest_paths = _name_counts_index_manifest_paths(index_dir)
    if manifest_paths is None or not all(path.exists() for path in manifest_paths.values()):
        return False
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("schema_version") != NAME_COUNTS_INDEX_SCHEMA_VERSION:
        return False
    if "fingerprint" not in manifest:
        return False
    return manifest.get("fingerprint") == expected_fingerprint and manifest.get("source_provenance") == dict(
        expected_source_provenance
    )


def _current_name_counts_generation_name(index_dir: Path) -> str | None:
    manifest_paths = _name_counts_index_manifest_paths(index_dir)
    if manifest_paths is None:
        return None
    generation_names: set[str] = set()
    generations_dir = index_dir / "generations"
    for path in manifest_paths.values():
        try:
            relative = path.relative_to(generations_dir)
        except ValueError:
            try:
                relative = path.resolve().relative_to(generations_dir.resolve())
            except ValueError:
                return None
        if len(relative.parts) < 2:
            return None
        generation_names.add(relative.parts[0])
    if len(generation_names) != 1:
        return None
    return next(iter(generation_names))


def cleanup_stale_name_counts_generations(index_dir: str | Path) -> dict[str, int]:
    """Delete published name-count generations not referenced by the current manifest."""

    index_path = Path(index_dir)
    with _exclusive_name_counts_publish_lock(index_path):
        generations_dir = index_path / "generations"
        if not generations_dir.exists():
            return {"removed_generation_count": 0}
        current_generation_name = _current_name_counts_generation_name(index_path)
        if current_generation_name is None:
            raise ValueError(
                f"refusing to clean name-count generations without a resolvable current manifest: {index_path}"
            )
        removed = 0
        for child in generations_dir.iterdir():
            if child.name.startswith(".") or not child.is_dir():
                continue
            if not (child / ".published").exists():
                continue
            if child.name == current_generation_name:
                continue
            shutil.rmtree(child)
            removed += 1
        return {"removed_generation_count": removed}


@contextmanager
def _exclusive_name_counts_publish_lock(index_dir: Path) -> Iterator[None]:
    """Serialize the short manifest publication boundary across processes."""

    with exclusive_file_lock(index_dir / ".publish.lock"):
        yield


def write_name_counts_index(output_dir: str | Path, *, overwrite: bool = False) -> tuple[str, dict[str, int | bool]]:
    """Write the global name-count lookup as exact-verified sorted binary indexes."""

    from s2and.data import _load_name_counts_artifact, _validated_name_counts_provenance

    index_dir = Path(output_dir) / "name_counts_index"
    manifest_path = index_dir / "manifest.json"

    counts, raw_source_provenance = _load_name_counts_artifact()
    source_provenance = _validated_name_counts_provenance(
        raw_source_provenance,
        context="write_name_counts_index",
    )
    first_dict, last_dict, first_last_dict, last_first_initial_dict = counts
    fingerprint = _name_counts_arrow_fingerprint(
        {
            "first": first_dict,
            "last": last_dict,
            "first_last": first_last_dict,
            "last_first_initial": last_first_initial_dict,
        }
    )
    if not overwrite and _name_counts_index_complete(
        index_dir,
        expected_fingerprint=fingerprint,
        expected_source_provenance=source_provenance,
    ):
        return str(index_dir), {"reused": True}

    mappings = (
        ("first", first_dict),
        ("last", last_dict),
        ("first_last", first_last_dict),
        ("last_first_initial", last_first_initial_dict),
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    required_free_bytes = sum(_name_count_index_disk_requirement(mapping) for _kind, mapping in mappings)
    free_bytes_at_preflight = shutil.disk_usage(index_dir).free
    if free_bytes_at_preflight < required_free_bytes:
        raise OSError(
            "insufficient free disk for name-count index generation: "
            f"required={required_free_bytes} free={free_bytes_at_preflight} path={index_dir}"
        )
    metrics: dict[str, int | bool] = {
        "reused": False,
        "required_free_bytes": required_free_bytes,
        "free_bytes_at_preflight": free_bytes_at_preflight,
    }
    total_records = 0
    total_bytes = 0
    manifest_files: dict[str, dict[str, int | str]] = {}
    generations_dir = index_dir / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation_name = f"gen-{uuid.uuid4().hex}"
    tmp_generation_dir = Path(tempfile.mkdtemp(prefix=f".{generation_name}.", dir=str(generations_dir)))
    generation_dir = generations_dir / generation_name
    tmp_manifest_path = index_dir / f".manifest.{generation_name}.json"
    try:
        for kind, mapping in mappings:
            filename = f"{kind}.bin"
            tmp_file = tmp_generation_dir / filename
            file_metrics = _write_name_count_index_file(
                tmp_file,
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
                "path": f"generations/{generation_name}/{filename}",
                "record_count": record_count,
                "byte_count": byte_count,
                "sha256": _sha256_file(tmp_file),
            }
            metrics[f"{kind}_sort_run_count"] = file_metrics["sort_run_count"]
            metrics[f"{kind}_peak_buffered_records"] = file_metrics["peak_buffered_records"]
            metrics[f"{kind}_temporary_bytes"] = file_metrics["temporary_byte_count"]

        manifest = {
            "schema_version": NAME_COUNTS_INDEX_SCHEMA_VERSION,
            "normalization_version": source_provenance["normalization_version"],
            "source_provenance": source_provenance,
            "magic": _NAME_COUNTS_INDEX_MAGIC.decode("ascii"),
            "fingerprint": fingerprint,
            "record_layout": "hash1:u64,hash2:u64,name_offset:u64,name_len:u32,reserved:u32,count:f64",
            "sort_order": "hash1,hash2,utf8_name_bytes",
            "hash": "fnv1a64(name_bytes), fnv1a64(domain + kind + NUL + name_bytes)",
            "exact_string_verification": True,
            "files": manifest_files,
        }
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with tmp_manifest_path.open("r+b") as manifest_input:
            os.fsync(manifest_input.fileno())
        marker_path = tmp_generation_dir / ".published"
        with marker_path.open("wb") as marker_output:
            marker_output.flush()
            os.fsync(marker_output.fileno())
        fsync_directory(tmp_generation_dir)

        with _exclusive_name_counts_publish_lock(index_dir):
            # Another writer may have completed the same generation while this
            # process was sorting. Reuse it rather than overwriting its manifest.
            if not overwrite and _name_counts_index_complete(
                index_dir,
                expected_fingerprint=fingerprint,
                expected_source_provenance=source_provenance,
            ):
                return str(index_dir), {"reused": True}
            tmp_generation_dir.rename(generation_dir)
            fsync_directory(generations_dir)
            for entry in manifest_files.values():
                path = index_dir / str(entry["path"])
                if not path.exists():
                    raise FileNotFoundError(f"name-count index generation is incomplete: {path}")
            tmp_manifest_path.replace(manifest_path)
            fsync_directory(index_dir)
    finally:
        if tmp_manifest_path.exists():
            tmp_manifest_path.unlink()
        if tmp_generation_dir.exists():
            shutil.rmtree(tmp_generation_dir)
        if generation_dir.exists():
            with _exclusive_name_counts_publish_lock(index_dir):
                if _current_name_counts_generation_name(index_dir) != generation_name:
                    shutil.rmtree(generation_dir)
                    fsync_directory(generations_dir)
    metrics["row_count"] = total_records
    metrics["byte_count"] = total_bytes
    return str(index_dir), metrics


def _arrow_rows_by_unique_key(
    rows: Iterable[Mapping[str, Any]],
    *,
    table_name: str,
    key_column: str,
) -> dict[str, Mapping[str, Any]]:
    rows_by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key_value = row.get(key_column)
        if key_value is None:
            raise ValueError(f"{table_name} Arrow cannot contain null {key_column} values")
        key = str(key_value)
        if not key:
            raise ValueError(f"{table_name} Arrow cannot contain empty {key_column} values")
        if key in rows_by_key:
            raise ValueError(f"{table_name} Arrow contains duplicate {key_column}: {key!r}")
        rows_by_key[key] = row
    return rows_by_key


def _require_arrow_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    missing_columns = sorted(required_columns.difference(table.column_names))
    if missing_columns:
        raise ValueError(f"{table_name} Arrow is missing required columns: {missing_columns}")


def _require_arrow_string_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    _require_arrow_columns(table, table_name, required_columns)
    pa = __import__("pyarrow")
    for column_name in sorted(required_columns):
        column_type = table[column_name].type
        if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
            raise ValueError(f"{table_name} Arrow column {column_name} expected string, got {column_type}")


def _require_arrow_int64_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    _require_arrow_columns(table, table_name, required_columns)
    pa = __import__("pyarrow")
    for column_name in sorted(required_columns):
        column_type = table[column_name].type
        if not pa.types.is_int64(column_type):
            raise ValueError(f"{table_name} Arrow column {column_name} expected int64, got {column_type}")


def _require_arrow_bool_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    _require_arrow_columns(table, table_name, required_columns)
    pa = __import__("pyarrow")
    for column_name in sorted(required_columns):
        column_type = table[column_name].type
        if not pa.types.is_boolean(column_type):
            raise ValueError(f"{table_name} Arrow column {column_name} expected bool, got {column_type}")


def _require_arrow_float_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    _require_arrow_columns(table, table_name, required_columns)
    pa = __import__("pyarrow")
    for column_name in sorted(required_columns):
        column_type = table[column_name].type
        if not (pa.types.is_float32(column_type) or pa.types.is_float64(column_type)):
            raise ValueError(f"{table_name} Arrow column {column_name} expected float, got {column_type}")


def _require_arrow_string_list_columns(table: Any, table_name: str, required_columns: set[str]) -> None:
    _require_arrow_columns(table, table_name, required_columns)
    pa = __import__("pyarrow")
    for column_name in sorted(required_columns):
        column_type = table[column_name].type
        value_type = getattr(column_type, "value_type", None)
        if not (pa.types.is_list(column_type) or pa.types.is_large_list(column_type)) or not (
            value_type is not None and (pa.types.is_string(value_type) or pa.types.is_large_string(value_type))
        ):
            raise ValueError(f"{table_name} Arrow column {column_name} expected list<string>, got {column_type}")


def feature_block_from_arrow_paths(
    paths: Mapping[str, Any],
    *,
    raw_candidate_plan: Mapping[str, Any],
    include_specter: bool = False,
) -> FeatureBlock:
    """Build a mini signal-only `FeatureBlock` from Arrow IPC inputs."""

    pa = __import__("pyarrow")
    pc = __import__("pyarrow.compute").compute
    from s2and.incremental_linking.retrieval import validate_raw_candidate_plan_schema

    validate_raw_candidate_plan_schema(raw_candidate_plan)
    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    selected_signature_ids = tuple(signature_order.signature_ids)
    selected_signature_id_set = set(selected_signature_ids)

    signatures_table = _read_arrow_ipc_table(pa, paths["signatures"])
    _require_arrow_string_columns(
        signatures_table,
        "signatures",
        {
            "signature_id",
            "paper_id",
            "author_first",
            "author_middle",
            "author_last",
            "author_suffix",
            "author_orcid",
        },
    )
    _require_arrow_int64_columns(signatures_table, "signatures", {"author_position"})
    _require_arrow_string_list_columns(signatures_table, "signatures", {"author_affiliations"})
    for optional_signature_string_column in ("author_block", "author_email"):
        if optional_signature_string_column in signatures_table.column_names:
            _require_arrow_string_columns(signatures_table, "signatures", {optional_signature_string_column})
    if "source_author_ids" in signatures_table.column_names:
        _require_arrow_string_list_columns(signatures_table, "signatures", {"source_author_ids"})
    signatures_table = _filter_arrow_table_by_values(pa, pc, signatures_table, "signature_id", selected_signature_ids)
    signatures_by_id = _arrow_rows_by_unique_key(
        signatures_table.to_pylist(),
        table_name="signatures",
        key_column="signature_id",
    )
    missing_signatures = [
        signature_id for signature_id in selected_signature_ids if signature_id not in signatures_by_id
    ]
    if missing_signatures:
        raise ValueError(f"Arrow signatures are missing raw-plan signature ids: {missing_signatures[:10]}")
    signature_rows = tuple(
        _feature_block_signature_from_arrow_row(signature_id, signatures_by_id[signature_id])
        for signature_id in selected_signature_ids
    )

    paper_ids = tuple(dict.fromkeys(row.paper_id for row in signature_rows))
    papers_table = _read_arrow_ipc_table(pa, paths["papers"])
    _require_arrow_string_columns(papers_table, "papers", {"journal_name", "paper_id", "title", "venue"})
    for optional_string_column in ("abstract", "predicted_language"):
        if optional_string_column in papers_table.column_names:
            _require_arrow_string_columns(papers_table, "papers", {optional_string_column})
    if "year" in papers_table.column_names:
        _require_arrow_int64_columns(papers_table, "papers", {"year"})
    if "is_reliable" in papers_table.column_names:
        _require_arrow_bool_columns(papers_table, "papers", {"is_reliable"})
    if "language_reliability" in papers_table.column_names:
        _require_arrow_float_columns(papers_table, "papers", {"language_reliability"})
    papers_table = _filter_arrow_table_by_values(pa, pc, papers_table, "paper_id", paper_ids)
    papers_by_id = _arrow_rows_by_unique_key(
        papers_table.to_pylist(),
        table_name="papers",
        key_column="paper_id",
    )
    missing_paper_ids = [paper_id for paper_id in paper_ids if paper_id not in papers_by_id]
    if missing_paper_ids:
        raise ValueError(f"Arrow papers are missing signature paper_ids: {missing_paper_ids[:10]}")
    paper_rows = tuple(_feature_block_paper_from_arrow_row(paper_id, papers_by_id[paper_id]) for paper_id in paper_ids)

    paper_author_rows: tuple[FeatureBlockPaperAuthor, ...] = ()
    paper_authors_path = paths.get("paper_authors")
    if paper_authors_path is not None:
        paper_authors_table = _read_arrow_ipc_table(pa, paper_authors_path)
        _require_arrow_int64_columns(paper_authors_table, "paper_authors", {"position"})
        _require_arrow_string_columns(paper_authors_table, "paper_authors", {"author_name", "paper_id"})
        paper_authors_table = _filter_arrow_table_by_values(pa, pc, paper_authors_table, "paper_id", paper_ids)
        paper_author_row_list: list[FeatureBlockPaperAuthor] = []
        seen_paper_author_positions: set[tuple[str, int]] = set()
        for row in paper_authors_table.to_pylist():
            paper_id_value = row.get("paper_id")
            if paper_id_value is None:
                raise ValueError("paper_authors Arrow cannot contain null paper_id values")
            paper_id = str(paper_id_value)
            if not paper_id:
                raise ValueError("paper_authors Arrow cannot contain empty paper_id values")
            if row.get("position") is None:
                raise ValueError("paper_authors Arrow cannot contain null position values")
            position = int(row["position"])
            author_name_value = row.get("author_name")
            if author_name_value is None:
                raise ValueError("paper_authors Arrow cannot contain null author_name values")
            author_name = str(author_name_value)
            if not author_name:
                raise ValueError("paper_authors Arrow cannot contain empty author_name values")
            key = (paper_id, position)
            if key in seen_paper_author_positions:
                raise ValueError(f"paper_authors Arrow contains duplicate (paper_id, position): {key!r}")
            seen_paper_author_positions.add(key)
            paper_author_row_list.append(
                FeatureBlockPaperAuthor(
                    paper_id=paper_id,
                    position=position,
                    author_name=author_name,
                )
            )
        paper_author_rows = tuple(paper_author_row_list)

    component_members = raw_candidate_plan.get("component_members")
    if not isinstance(component_members, Mapping):
        raise ValueError("raw candidate plan must include component_members")
    require_pairs = tuple(
        (str(signature_id), str(component_key))
        for component_key, members in component_members.items()
        for signature_id in members
        if str(signature_id) in selected_signature_id_set
    )
    disallow_pairs: tuple[tuple[str, str], ...] = ()
    disallow_path = paths.get("cluster_seed_disallows")
    if disallow_path is not None:
        disallow_rows = read_cluster_seed_disallows_arrow(Path(disallow_path))
        disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
            disallow_rows,
            selected_signature_id_set,
        )

    specter_paper_ids: tuple[str, ...] = ()
    specter_embeddings: np.ndarray | None = None
    specter_path = paths.get("specter")
    if include_specter and specter_path is not None:
        specter_table = _read_arrow_ipc_table(pa, specter_path)
        _require_arrow_columns(specter_table, "specter", {"embedding"})
        _require_arrow_string_columns(specter_table, "specter", {"paper_id"})
        specter_table = _filter_arrow_table_by_values(pa, pc, specter_table, "paper_id", paper_ids)
        specter_by_id = _arrow_rows_by_unique_key(
            specter_table.to_pylist(),
            table_name="specter",
            key_column="paper_id",
        )
        specter_payload: dict[str, Any] = {}
        for paper_id in paper_ids:
            row = specter_by_id.get(paper_id)
            if row is None:
                continue
            embedding = row.get("embedding")
            if embedding is None:
                raise ValueError("specter Arrow cannot contain null embedding values")
            specter_payload[paper_id] = embedding
        specter_paper_ids, specter_embeddings = _feature_block_specter_from_mapping(paper_ids, specter_payload)
    return FeatureBlock(
        signatures=signature_rows,
        papers=paper_rows,
        paper_authors=paper_author_rows,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def _read_arrow_ipc_table(pa: Any, path: Any) -> Any:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _filter_arrow_table_by_values(pa: Any, pc: Any, table: Any, column: str, values: Sequence[str]) -> Any:
    value_list = [str(value) for value in values]
    if not value_list:
        return table.slice(0, 0)
    value_set = pa.array(value_list, type=table[column].type)
    cast_values = value_set.to_pylist()
    if any(value is None for value in cast_values) or len(set(cast_values)) != len(value_list):
        raise ValueError(f"{column} filter values are not one-to-one after casting to Arrow type {table[column].type}")
    mask = pc.is_in(table[column], value_set=value_set)
    return table.filter(mask)


def _feature_block_signature_from_arrow_row(signature_id: str, row: Mapping[str, Any]) -> FeatureBlockSignature:
    return FeatureBlockSignature(
        signature_id=str(row.get("signature_id", signature_id)),
        paper_id=str(row["paper_id"]),
        author_first=_optional_str(row.get("author_first")),
        author_middle=_optional_str(row.get("author_middle")),
        author_last=_optional_str(row.get("author_last")),
        author_suffix=_optional_str(row.get("author_suffix")),
        author_affiliations=_strict_string_tuple(
            row.get("author_affiliations"),
            field_name="signatures.author_affiliations",
        ),
        author_orcid=_optional_str(row.get("author_orcid")),
        author_position=_optional_int(row.get("author_position"), field_name="signatures.author_position"),
        author_block=_optional_str(row.get("author_block")),
        author_email=_optional_str(row.get("author_email")),
        source_author_ids=_strict_string_tuple(
            row.get("source_author_ids"),
            field_name="signatures.source_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_paper_from_arrow_row(paper_id: str, row: Mapping[str, Any]) -> FeatureBlockPaper:
    is_reliable = _optional_bool(row.get("is_reliable"), field_name="papers.is_reliable")
    return FeatureBlockPaper(
        paper_id=str(row.get("paper_id", paper_id)),
        title=_optional_str(row.get("title")),
        abstract=_optional_str(row.get("abstract")),
        venue=_optional_str(row.get("venue")),
        journal_name=_optional_str(row.get("journal_name")),
        year=_optional_int(row.get("year"), field_name="papers.year"),
        predicted_language=_optional_str(row.get("predicted_language")),
        is_reliable=is_reliable,
        language_reliability=row.get("language_reliability"),
    )
