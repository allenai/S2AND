"""Export Athena author-row mass and build the canonical production count index.

The Parquet weight column retains the name ``mention_count`` for producer compatibility.
Here, that column counts author-table rows, not paper-author mentions.
Run without ``--execute`` to inspect the SQL and limits without AWS access.
A complete local resume needs no AWS SDK or credentials.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from s2and._sha256 import sha256_file

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def _save(path: Path, value: object) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _s3_parts(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError(f"Require an S3 object or prefix URI: {uri}")
    if any(character in uri for character in ("'", "\n", "\r")):
        raise ValueError("S3 URI contains a SQL delimiter")
    return parsed.netloc, parsed.path.lstrip("/")


def _thread_capacity() -> int:
    import psutil

    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        try:
            available = len(process.cpu_affinity())
        except (AttributeError, NotImplementedError):
            available = psutil.cpu_count(logical=True) or 1
    else:
        available = psutil.cpu_count(logical=True) or 1
    return max(1, available - 2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="AWS shared profile; omit for the standard credential chain")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--database", default="content_ext")
    parser.add_argument("--table", default="authors")
    parser.add_argument("--workgroup", required=True, help="Workgroup with a per-query scan cutoff")
    parser.add_argument("--query-output", required=True, help="S3 prefix for Athena query metadata")
    parser.add_argument("--unload-prefix", required=True, help="Empty S3 prefix reserved for this export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-author-rows", type=int, required=True)
    parser.add_argument("--expected-raw-groups", type=int)
    parser.add_argument("--max-scan-bytes", type=int, default=15_000_000_000)
    parser.add_argument("--max-query-seconds", type=int, default=900)
    parser.add_argument("--max-manifest-files", type=int, default=10_000)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument(
        "--price-per-tb-usd", type=float, default=5, help="Athena scan-price estimate; excludes S3 costs"
    )
    parser.add_argument("--threads", type=int, default=1, help="Local producer allocation within the shared CPU budget")
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--execute", action="store_true", help="Submit or resume the planned AWS query")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the saved query and local checkpoints. Complete local exports need no AWS SDK.",
    )
    return parser


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """Return the explicit source SQL and resource limits without AWS access."""
    if not all(_IDENTIFIER.fullmatch(value) for value in (args.database, args.table)):
        raise ValueError("Database and table require plain SQL identifiers")
    for uri in (args.query_output, args.unload_prefix):
        _s3_parts(uri)
    positive = (
        args.expected_author_rows,
        args.max_scan_bytes,
        args.max_query_seconds,
        args.max_manifest_files,
        args.poll_seconds,
        args.threads,
    )
    if min(positive) <= 0 or args.price_per_tb_usd < 0:
        raise ValueError("Population and resource bounds must be positive")
    if args.expected_raw_groups is not None and args.expected_raw_groups < 1:
        raise ValueError("Expected raw groups must be positive")
    capacity = _thread_capacity()
    if args.threads > capacity:
        raise ValueError(f"Local threads {args.threads} exceed the available shared budget {capacity}")
    table = f"{args.database}.{args.table}"
    destination = args.unload_prefix.rstrip("/") + "/"
    source_sql = (
        f"SELECT first_name, last_name, count(*) AS mention_count\nFROM {table}\nGROUP BY first_name, last_name"
    )
    normalizer_spec = importlib.util.find_spec("s2and.text")
    if normalizer_spec is None or normalizer_spec.origin is None:
        raise RuntimeError("Cannot locate the production name normalizer")
    return {
        "source_unit": "author_table_row",
        "normalizer_sha256": sha256_file(Path(normalizer_spec.origin)),
        "source_sql": source_sql,
        "population_audit_sql": (
            f"SELECT sum(author_rows) AS author_rows, count(*) AS raw_groups FROM "
            f"(SELECT count(*) AS author_rows FROM {table} GROUP BY first_name, last_name) AS raw_names"
        ),
        "sql": f"UNLOAD (\n{source_sql}\n) TO '{destination}'\nWITH (format='PARQUET', compression='SNAPPY')",
        "destination": destination,
        "database": args.database,
        "workgroup": args.workgroup,
        "region": args.region,
        "profile": args.profile,
        "query_output": args.query_output,
        "expected_author_rows": args.expected_author_rows,
        "expected_raw_groups": args.expected_raw_groups,
        "max_scan_bytes": args.max_scan_bytes,
        "max_query_seconds": args.max_query_seconds,
        "max_manifest_files": args.max_manifest_files,
        "poll_seconds": args.poll_seconds,
        "scan_price_estimate_upper_usd": args.max_scan_bytes / 1_000_000_000_000 * args.price_per_tb_usd,
        "local_thread_budget": capacity,
        "threads": args.threads,
        "thread_environment": {
            "OMP_NUM_THREADS": str(args.threads),
            "RAYON_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "memory_limit": args.memory_limit,
    }


def _wait_query(athena: Any, record: dict[str, Any], path: Path) -> None:
    execution_id = record["execution_id"]
    while True:
        execution = athena.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"]
        state = execution["Status"]["State"]
        stats = execution.get("Statistics", {})
        elapsed = time.time() - record["submitted_epoch"]
        record.update(query_state=state, scanned_bytes=stats.get("DataScannedInBytes", 0), elapsed_seconds=elapsed)
        _save(path, record)
        over_scan = record["scanned_bytes"] > record["plan"]["max_scan_bytes"]
        # A completed query can resume after the local process stays offline.
        query_seconds = stats.get("TotalExecutionTimeInMillis", 0) / 1000
        over_time = query_seconds > record["plan"]["max_query_seconds"] or (
            state not in _TERMINAL and elapsed > record["plan"]["max_query_seconds"]
        )
        if over_scan or over_time:
            if state not in _TERMINAL:
                athena.stop_query_execution(QueryExecutionId=execution_id)
            record.update(status="bound_exceeded")
            _save(path, record)
            raise RuntimeError(f"Author export exceeds the query bound: {execution_id}")
        if state in _TERMINAL:
            if state != "SUCCEEDED":
                record.update(status="failed", reason=execution["Status"].get("StateChangeReason", state))
                _save(path, record)
                raise RuntimeError(f"Author export {state}: {record['reason']}")
            record.update(status="exported", data_manifest_location=stats.get("DataManifestLocation"))
            _save(path, record)
            return
        time.sleep(record["plan"]["poll_seconds"])


def _manifest_paths(athena: Any, execution_id: str, destination: str, limit: int) -> list[str]:
    paths: list[str] = []
    token: dict[str, str] = {}
    while True:
        response = athena.get_query_results(
            QueryExecutionId=execution_id, QueryResultType="DATA_MANIFEST", MaxResults=1000, **token
        )
        for row in response["ResultSet"]["Rows"]:
            for cell in row["Data"]:
                value = cell.get("VarCharValue", "")
                if not value:
                    continue
                if not value.startswith(destination):
                    raise ValueError(f"Manifest entry is outside the export prefix: {value}")
                _s3_parts(value)
                paths.append(value)
                if len(paths) > limit:
                    raise ValueError("Athena manifest exceeds the file bound")
        if "NextToken" not in response:
            break
        token = {"NextToken": response["NextToken"]}
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Require nonempty unique Athena manifest paths")
    return paths


def _download_sources(s3: Any, paths: list[str], output: Path, record: dict[str, Any], record_path: Path) -> Path:
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    entries = record.setdefault("downloaded", [])
    if len(entries) > len(paths) or any(entry["s3_uri"] != paths[i] for i, entry in enumerate(entries)):
        raise ValueError("Saved downloads differ from the Athena manifest")
    for position, uri in enumerate(paths):
        path = source_dir / f"source_{position:06d}.parquet"
        if position < len(entries):
            entry = entries[position]
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Saved source object differs: {path}")
            continue
        bucket, key = _s3_parts(uri)
        response = s3.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        temporary = path.with_suffix(".partial")
        try:
            with temporary.open("wb") as handle:
                for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
                    handle.write(chunk)
        finally:
            body.close()
        if temporary.stat().st_size != response["ContentLength"]:
            raise ValueError(f"Downloaded byte count differs: {uri}")
        with pq.ParquetFile(temporary) as parquet:
            schema = parquet.schema_arrow
            expected = {"first_name": pa.string(), "last_name": pa.string(), "mention_count": pa.int64()}
            if set(schema.names) != set(expected) or any(
                schema.field(key).type != dtype for key, dtype in expected.items()
            ):
                raise ValueError(f"Author export schema differs: {uri}")
            rows = parquet.metadata.num_rows
        entry = {
            "s3_uri": uri,
            "path": str(path),
            "bytes": temporary.stat().st_size,
            "rows": rows,
            "sha256": sha256_file(temporary),
        }
        temporary.replace(path)
        entries.append(entry)
        _save(record_path, record)
    raw_groups = sum(entry["rows"] for entry in entries)
    expected_groups = record["plan"]["expected_raw_groups"]
    if expected_groups is not None and raw_groups != expected_groups:
        raise ValueError(f"Raw groups {raw_groups} differ from the expected population {expected_groups}")
    with duckdb.connect() as con:
        con.execute(f"SET threads={record['plan']['threads']}")
        con.execute("SET memory_limit=?", [record["plan"]["memory_limit"]])
        row = con.execute(
            "SELECT sum(mention_count),count(*) FILTER (WHERE mention_count IS NULL OR mention_count<1) "
            "FROM read_parquet(?)",
            [[entry["path"] for entry in entries]],
        ).fetchone()
    if row is None or row[1] or row[0] != record["plan"]["expected_author_rows"]:
        raise ValueError(f"Author-row population differs: {row}")
    manifest = source_dir / "manifest.json"
    _save(manifest, {"status": "complete", "source_unit": "author_table_row", "files": entries})
    record.update(raw_groups=raw_groups, author_rows=row[0], source_manifest=str(manifest), status="downloaded")
    _save(record_path, record)
    return manifest


def _aws_clients(args: argparse.Namespace, session: Any) -> tuple[Any, Any]:
    if session is None:
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        client_config = Config(
            connect_timeout=10, read_timeout=30, retries={"mode": "standard", "total_max_attempts": 3}
        )
        athena = session.client("athena", config=client_config)
        s3 = session.client("s3", config=client_config)
    else:
        athena, s3 = session.client("athena"), session.client("s3")
    return athena, s3


def execute_export(args: argparse.Namespace, run_plan: dict[str, Any], *, session: Any = None) -> dict[str, Any]:
    """Resume one bounded Athena export and the existing canonical producer."""
    os.environ.update(run_plan["thread_environment"])
    import s2and.text as text_normalizer

    from .generate_name_counts_from_parquet import build_name_counts_from_parquet

    if sha256_file(Path(text_normalizer.__file__)) != run_plan["normalizer_sha256"]:
        raise ValueError("Name normalizer changes after the plan")
    output = args.output_dir.resolve()
    record_path = output / "export.json"
    record: dict[str, Any]
    if args.resume:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["plan"] != run_plan:
            raise ValueError("Resume plan differs from the saved export")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        record = {"status": "prepared", "plan": run_plan, "client_request_token": str(uuid.uuid4())}
        _save(record_path, record)
        (output / "query.sql").write_text(run_plan["sql"] + "\n", encoding="utf-8", newline="\n")
    if record["status"] in {"failed", "bound_exceeded"}:
        raise RuntimeError(f"Saved Athena query is terminal: {record['status']}; start a fresh export")
    needs_remote = (
        "execution_id" not in record
        or record["status"] in {"prepared", "active"}
        or "manifest_paths" not in record
        or len(record.get("downloaded", [])) < len(record["manifest_paths"])
    )
    athena: Any
    s3: Any
    athena, s3 = _aws_clients(args, session) if needs_remote else (None, None)
    if "execution_id" not in record:
        assert athena is not None
        cutoff = athena.get_work_group(WorkGroup=args.workgroup)["WorkGroup"]["Configuration"].get(
            "BytesScannedCutoffPerQuery"
        )
        if cutoff is None or cutoff > args.max_scan_bytes:
            raise ValueError("Athena workgroup requires a per-query scan cutoff at or below --max-scan-bytes")
        record.setdefault("submitted_epoch", time.time())
        _save(record_path, record)
        response = athena.start_query_execution(
            QueryString=run_plan["sql"],
            ClientRequestToken=record["client_request_token"],
            QueryExecutionContext={"Database": args.database},
            WorkGroup=args.workgroup,
            ResultConfiguration={"OutputLocation": args.query_output},
        )
        record.update(status="active", execution_id=response["QueryExecutionId"])
        _save(record_path, record)
    if record["status"] in {"prepared", "active"}:
        _wait_query(athena, record, record_path)
    if record["status"] in {"failed", "bound_exceeded"}:
        raise RuntimeError(f"Saved Athena query is terminal: {record['status']}; start a fresh export")
    if "manifest_paths" not in record:
        record["manifest_paths"] = _manifest_paths(
            athena, record["execution_id"], run_plan["destination"], args.max_manifest_files
        )
        _save(record_path, record)
    manifest = _download_sources(s3, record["manifest_paths"], output, record, record_path)
    index_output = output / "canonical"
    result = build_name_counts_from_parquet(
        manifest,
        index_output,
        expected_source_rows=record["raw_groups"],
        expected_source_mentions=record["author_rows"],
        threads=args.threads,
        memory_limit=args.memory_limit,
        resume=(index_output / "run.json").exists(),
    )
    record.update(
        status="complete",
        finished_utc=datetime.now(UTC).isoformat(),
        index_path=result["index_path"],
        index_manifest_sha256=result["index_manifest_sha256"],
    )
    _save(record_path, record)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Print the plan before any query and require explicit execution."""
    args = _parser().parse_args(argv)
    run_plan = plan(args)
    print(json.dumps({"event": "author_counts_plan", **run_plan}, indent=2), flush=True)
    if args.resume and not args.execute:
        raise ValueError("--resume requires --execute")
    if args.execute:
        result = execute_export(args, run_plan)
        print(json.dumps({"event": "author_counts_complete", "index_path": result["index_path"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
