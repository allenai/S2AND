"""Build a canonical count index from reviewed weighted raw-name Parquet objects."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import s2and.text as text_normalizer
from s2and._sha256 import sha256_file
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block_arrow import _write_sorted_name_count_records
from s2and.name_counts_index import NameCountsIndex
from s2and.runtime import load_s2and_rust_extension
from s2and.text import canonicalize_name_parts, last_name_allows_joined_initials

from ._run_support import emit_jsonl, validate_input_file

KINDS = ("first", "last", "first_last", "last_first_initial")
_BATCH_ROWS = 100_000
# The D9 initials rule needs the last name, but each map holds one name part.
# The first map keeps both D9 outputs, and the join picks one by the last-name guard.
_CANONICAL_MAP_FIELDS = {
    "first": (("cfirst", pa.string()), ("cfirst_initials", pa.string())),
    "last": (("clast", pa.string()), ("initials_ok", pa.bool_())),
}


def _canonical_map_columns(part: str, raw: list[str | None]) -> list[list[Any]]:
    """Return the canonical map columns for one batch of distinct raw names.

    Args:
        part: ``first`` or ``last``.
        raw: Distinct raw values of that name part.

    Returns:
        The columns of ``_CANONICAL_MAP_FIELDS[part]``, in order.
    """
    if part == "first":
        return [
            [canonicalize_name_parts(value, None, None, last_allows_initials=False).first for value in raw],
            [canonicalize_name_parts(value, None, None, last_allows_initials=True).first for value in raw],
        ]
    return [
        [canonicalize_name_parts(None, None, value).last for value in raw],
        [last_name_allows_joined_initials(value) for value in raw],
    ]


def _aggregate(con: duckdb.DuckDBPyConnection, query: str) -> tuple[Any, ...]:
    """Return the single row that an aggregate query always produces."""
    row = con.execute(query).fetchone()
    if row is None:
        raise RuntimeError(f"Aggregate query returned no row: {query}")
    return row


def _source_files(manifest: Path, expected_rows: int) -> list[str]:
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("status", "complete") != "complete":
        raise ValueError("Require a complete source manifest")
    entries = source.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Source manifest files must be a nonempty list")
    files = []
    checksums: list[str] = []
    rows = 0
    for entry in entries:
        if not isinstance(entry, dict) or not {"path", "bytes", "rows", "sha256"}.issubset(entry):
            raise ValueError("Each source entry requires path, bytes, rows, and sha256")
        if not isinstance(entry["path"], str) or not isinstance(entry["sha256"], str):
            raise ValueError("Source paths and checksums must be strings")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        path = validate_input_file(path, option="source object")
        if str(path) in files:
            raise ValueError(f"Duplicate source object: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Source byte count differs: {path}")
        parquet = pq.ParquetFile(path)
        for column, dtype in (("first_name", pa.string()), ("last_name", pa.string()), ("mention_count", pa.int64())):
            if column not in parquet.schema_arrow.names or parquet.schema_arrow.field(column).type != dtype:
                raise ValueError(f"Source column dtype differs: {path}/{column}")
        if parquet.metadata.num_rows != entry["rows"]:
            raise ValueError(f"Source row count differs: {path}")
        rows += parquet.metadata.num_rows
        files.append(str(path))
        checksums.append(entry["sha256"])
    if rows != expected_rows:
        raise ValueError(f"Source rows {rows} differ from expected rows {expected_rows}")
    for path, checksum in zip(files, checksums, strict=True):
        if sha256_file(Path(path)) != checksum:
            raise ValueError(f"Source checksum differs: {path}")
    return files


def _sorted_records(reader: pa.RecordBatchReader) -> Iterator[tuple[int, int, bytes, float]]:
    for batch in reader:
        for h1, h2, key, value in zip(*(column.to_pylist() for column in batch.columns), strict=True):
            yield h1, h2, key.encode("utf-8"), float(value)


def build_name_counts_from_parquet(
    input_manifest: Path,
    output_dir: Path,
    *,
    expected_source_rows: int,
    expected_source_mentions: int,
    threads: int = 1,
    memory_limit: str = "12GB",
    resume: bool = False,
) -> dict[str, Any]:
    """Build or resume one immutable index with exact source population bounds."""
    if min(expected_source_rows, expected_source_mentions, threads) < 1:
        raise ValueError("Expected populations and threads must be positive integers")
    manifest = validate_input_file(input_manifest, option="--input-manifest")
    output = output_dir.resolve()
    identity = {
        "source_manifest_sha256": sha256_file(manifest),
        "producer_sha256": sha256_file(Path(__file__)),
        "normalizer_sha256": sha256_file(Path(text_normalizer.__file__)),
        "expected_source_rows": expected_source_rows,
        "expected_source_mentions": expected_source_mentions,
    }
    report_path = output / "run.json"
    if resume:
        previous = json.loads(validate_input_file(report_path, option="resume run record").read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Resume input bounds or producer identity differ")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    files = _source_files(manifest, expected_source_rows)
    index_dir = output / "name_counts_index"
    if index_dir.exists():
        if not resume or previous.get("status") != "complete":
            raise FileExistsError(f"Published index already exists: {index_dir}")
        index = NameCountsIndex.open(index_dir)
        if index.manifest_sha256 != previous["index_manifest_sha256"]:
            raise ValueError("Published index manifest differs from the completed run")
        emit_jsonl({"event": "name_counts_already_complete", "index": str(index_dir)})
        return previous
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        **identity,
        "status": "active",
        "pid": os.getpid(),
        "source_manifest": str(manifest),
        "source_objects": len(files),
        "threads": threads,
        "memory_limit": memory_limit,
        "started_utc": datetime.now(UTC).isoformat(),
        "stages": previous["stages"] if resume else {},
    }

    def emit(event: str, **values: object) -> None:
        report["elapsed_seconds"] = time.perf_counter() - started
        temporary_report = report_path.with_suffix(".partial")
        temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary_report.replace(report_path)
        emit_jsonl({"event": event, "seconds": report["elapsed_seconds"], **values})

    emit("name_counts_start", source_objects=len(files), expected_source_rows=expected_source_rows)
    try:
        with duckdb.connect(str(output / "counts.duckdb")) as con:
            con.execute(f"SET threads={threads}")
            con.execute("SET memory_limit=?", [memory_limit])
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET temp_directory=?", [str(output / "duckdb_tmp")])
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            if "raw" not in tables:
                con.execute(
                    "CREATE TABLE raw AS SELECT first_name,last_name,mention_count FROM read_parquet(?)", [files]
                )
            rows, mentions, bad = _aggregate(
                con,
                "SELECT count(*),sum(mention_count),count(*) FILTER "
                "(WHERE mention_count IS NULL OR mention_count<1) FROM raw",
            )
            if bad or rows != expected_source_rows or mentions != expected_source_mentions:
                raise ValueError(f"Source population differs: rows={rows}, mentions={mentions}, invalid_counts={bad}")
            report.update(source_rows=rows, source_mentions=mentions)
            emit("name_counts_source_ready", rows=rows, mentions=mentions)
            for column, part in (("first_name", "first"), ("last_name", "last")):
                table = f"map_{part}"
                if table in tables:
                    continue
                map_path = output / f"{table}.parquet"
                schema = pa.schema([(column, pa.string()), *_CANONICAL_MAP_FIELDS[part]])
                reader = con.execute(f"SELECT DISTINCT {column} FROM raw").to_arrow_reader(_BATCH_ROWS)
                count = 0
                with pq.ParquetWriter(map_path, schema, compression="zstd") as writer:
                    for batch in reader:
                        raw = batch.column(0).to_pylist()
                        columns = [raw, *_canonical_map_columns(part, raw)]
                        writer.write_table(pa.Table.from_pydict(dict(zip(schema.names, columns, strict=True)), schema))
                        count += len(raw)
                        if count % 1_000_000 == 0:
                            emit("name_counts_canonical_progress", part=part, rows=count)
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_parquet(?)", [str(map_path)])
                report["stages"][table] = {"rows": count}
                emit("name_counts_canonical_ready", part=part, rows=count)
            if "canonical" not in tables:
                con.execute(
                    "CREATE TABLE canonical AS SELECT "
                    "CASE WHEN l.initials_ok THEN f.cfirst_initials ELSE f.cfirst END AS cfirst,"
                    "clast,mention_count FROM raw r "
                    "JOIN map_first f ON r.first_name IS NOT DISTINCT FROM f.first_name "
                    "JOIN map_last l ON r.last_name IS NOT DISTINCT FROM l.last_name"
                )
            if _aggregate(con, "SELECT count(*) FROM canonical")[0] != rows:
                raise ValueError("Canonical map join changes the source population")
            specs = {
                "first": ("cfirst", "length(cfirst)>1"),
                "last": ("clast", "length(clast)>0"),
                "first_last": ("cfirst || ' ' || clast", "length(cfirst)>1 AND length(clast)>0"),
                "last_first_initial": ("clast || ' ' || left(cfirst,1)", "length(cfirst)>0 AND length(clast)>0"),
            }
            pending = output / ".name_counts_index"
            pending.mkdir(exist_ok=True)
            native = load_s2and_rust_extension()
            manifest_files = {}
            for kind, (expression, gate) in specs.items():
                table = f"counts_{kind}"
                if table not in tables:
                    con.execute(
                        f"CREATE TABLE {table} AS SELECT {expression} AS key,sum(mention_count)::BIGINT AS count "
                        f"FROM canonical WHERE {gate} GROUP BY 1 HAVING sum(mention_count)>1"
                    )
                count, total, minimum, maximum = _aggregate(
                    con, f"SELECT count(*),sum(count),min(count),max(count) FROM {table}"
                )
                if count and (minimum <= 1 or maximum > 2**53):
                    raise ValueError(f"Index counts exceed the exact positive-integer range: {kind}")
                hashed = output / f"hashed_{kind}.parquet"
                if not hashed.exists():
                    schema = pa.schema(
                        [("hash_1", pa.uint64()), ("hash_2", pa.uint64()), ("key", pa.string()), ("count", pa.int64())]
                    )
                    prefix = "s2and-name-counts-index-v1\x00" + kind + "\x00"
                    reader = con.execute(f"SELECT key,count FROM {table}").to_arrow_reader(_BATCH_ROWS)
                    partial_hash = hashed.with_suffix(".partial")
                    with pq.ParquetWriter(partial_hash, schema, compression="zstd") as writer:
                        for batch in reader:
                            keys = batch.column(0).to_pylist()
                            writer.write_table(
                                pa.table(
                                    {
                                        "hash_1": native.fnv64_utf8_batch(keys),
                                        "hash_2": native.fnv64_utf8_batch([prefix + key for key in keys]),
                                        "key": keys,
                                        "count": batch.column(1),
                                    },
                                    schema=schema,
                                )
                            )
                    partial_hash.rename(hashed)
                destination = pending / f"{kind}.bin"
                if not destination.exists():
                    reader = con.execute(
                        "SELECT hash_1,hash_2,key,count FROM read_parquet(?) ORDER BY hash_1,hash_2,key", [str(hashed)]
                    ).to_arrow_reader(_BATCH_ROWS)
                    _write_sorted_name_count_records(destination, _sorted_records(reader), record_count=count)
                manifest_files[kind] = {"byte_count": destination.stat().st_size, "sha256": sha256_file(destination)}
                report["stages"][kind] = {
                    "keys": count,
                    "sum_retained_counts": total or 0,
                    **manifest_files[kind],
                }
                emit("name_counts_kind_ready", kind=kind, keys=count)
            if sum(report["stages"][kind]["keys"] for kind in KINDS) == 0:
                raise ValueError("Source produces zero retained canonical keys")
            publication = {
                "kind": "s2and_name_counts",
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "files": manifest_files,
            }
            # Models bind this manifest's SHA-256, so its bytes must not depend on the OS.
            (pending / "manifest.json").write_text(
                json.dumps(publication, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
            )
            # Check every digest and record of the new index before publishing it.
            index = native.NameCountsIndex.open(str(pending), verify=True)
            index_sha256 = str(index.name_counts_manifest_sha256)
            # Windows blocks the directory move while the native index holds open files.
            del index
        pending.rename(index_dir)
    except (ValueError, OSError, RuntimeError, duckdb.Error, pa.ArrowException) as error:
        report.update(status="failed", error=str(error), exit_code=1, finished_utc=datetime.now(UTC).isoformat())
        emit("name_counts_failed", error=str(error))
        raise
    report.update(
        status="complete",
        exit_code=0,
        finished_utc=datetime.now(UTC).isoformat(),
        index_path=str(index_dir),
        index_manifest_sha256=index_sha256,
    )
    emit("name_counts_complete", source_rows=rows, index_path=str(index_dir))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Build one global index from the reviewed source manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-rows", type=int, required=True)
    parser.add_argument("--expected-source-mentions", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    build_name_counts_from_parquet(
        args.input_manifest,
        args.output_dir,
        expected_source_rows=args.expected_source_rows,
        expected_source_mentions=args.expected_source_mentions,
        threads=args.threads,
        memory_limit=args.memory_limit,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
