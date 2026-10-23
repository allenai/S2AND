"""Athena boundary and canonical-index tests for author-row counts."""

from __future__ import annotations

import builtins
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from scripts.production.counts import generate_author_counts_from_athena as exporter

ROWS: list[tuple[str | None, str | None, int]] = [
    ("Ána", "García", 1),
    ("Ana", "Garcia", 2),
    ("A", "Garcia", 3),
    (None, "Garcia", 2),
    ("", "", 5),
    ("Zoë", None, 2),
    ("Jean-Paul", "van der Berg", 2),
    ("Jean Paul", "van der Berg", 1),
    ("Anna Maria", "da Silva", 3),
    ("Anna-Marie", "da Silva", 2),
    ("Mary", "O'Neill", 4),
    ("HG", "Wells", 2),
    ("HG", "WELLS", 3),
    ("Jh", "Wells", 4),
    ("Jh", "WELLS", 2),
    ("Only", "Singleton", 1),
]


def _arguments(output: Path, *extra: str) -> list[str]:
    return [
        "--workgroup",
        "bounded",
        "--query-output",
        "s3://example/query/",
        "--unload-prefix",
        "s3://example/export/",
        "--output-dir",
        str(output),
        "--expected-author-rows",
        str(sum(row[2] for row in ROWS)),
        "--expected-raw-groups",
        str(len(ROWS)),
        "--threads",
        "1",
        *extra,
    ]


def _clients() -> tuple[Any, Any, Any, Any, Any]:
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber

    session = boto3.Session(
        aws_access_key_id="test", aws_secret_access_key="test", aws_session_token="test", region_name="us-west-2"
    )
    athena, s3 = session.client("athena"), session.client("s3")
    clients = {"athena": athena, "s3": s3}
    return SimpleNamespace(client=clients.__getitem__), athena, s3, Stubber(athena), Stubber(s3)


def _parquet(rows: list[tuple[str | None, str | None, int]]) -> bytes:
    stream = io.BytesIO()
    schema = pa.schema([("first_name", pa.string()), ("last_name", pa.string()), ("mention_count", pa.int64())])
    pq.write_table(
        pa.Table.from_pylist(
            [{"first_name": first, "last_name": last, "mention_count": count} for first, last, count in rows],
            schema=schema,
        ),
        stream,
    )
    return stream.getvalue()


def _export_responses(athena_stub: Any, s3_stub: Any, run_plan: dict[str, Any]) -> None:
    from botocore.response import StreamingBody
    from botocore.stub import ANY

    athena_stub.add_response(
        "get_work_group",
        {"WorkGroup": {"Name": "bounded", "Configuration": {"BytesScannedCutoffPerQuery": 100_000_000}}},
        {"WorkGroup": "bounded"},
    )
    athena_stub.add_response(
        "start_query_execution",
        {"QueryExecutionId": "export-1"},
        {
            "QueryString": run_plan["sql"],
            "ClientRequestToken": ANY,
            "QueryExecutionContext": {"Database": "content_ext"},
            "WorkGroup": "bounded",
            "ResultConfiguration": {"OutputLocation": "s3://example/query/"},
        },
    )
    athena_stub.add_response(
        "get_query_execution",
        {
            "QueryExecution": {
                "QueryExecutionId": "export-1",
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {"DataScannedInBytes": 12345, "TotalExecutionTimeInMillis": 10},
            }
        },
        {"QueryExecutionId": "export-1"},
    )
    for position, rows in enumerate((ROWS[:6], ROWS[6:])):
        uri = f"s3://example/export/data_{position}"
        result: dict[str, Any] = {"ResultSet": {"Rows": [{"Data": [{"VarCharValue": uri}]}]}}
        arguments = {"QueryExecutionId": "export-1", "QueryResultType": "DATA_MANIFEST", "MaxResults": 1000}
        if position == 0:
            result["NextToken"] = "page-2"
        else:
            arguments["NextToken"] = "page-2"
        athena_stub.add_response("get_query_results", result, arguments)
        data = _parquet(rows)
        s3_stub.add_response(
            "get_object",
            {"Body": StreamingBody(io.BytesIO(data), len(data)), "ContentLength": len(data)},
            {"Bucket": "example", "Key": f"export/data_{position}"},
        )


def test_plan_has_exact_raw_author_sql_and_does_not_execute(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "planned"
    assert exporter.main(_arguments(output)) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["source_unit"] == "author_table_row"
    assert value["source_sql"] == (
        "SELECT first_name, last_name, count(*) AS mention_count\n"
        "FROM content_ext.authors\nGROUP BY first_name, last_name"
    )
    assert "sum(author_rows)" in value["population_audit_sql"]
    assert value["scan_price_estimate_upper_usd"] == 0.075
    assert not output.exists()


def test_plan_rejects_sql_injection_and_excess_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = exporter._parser().parse_args(_arguments(tmp_path / "out", "--table", "authors WHERE true"))
    with pytest.raises(ValueError, match="plain SQL identifiers"):
        exporter.plan(args)
    args = exporter._parser().parse_args(_arguments(tmp_path / "out", "--threads", "3"))
    monkeypatch.setattr(exporter, "_thread_capacity", lambda: 2)
    with pytest.raises(ValueError, match="shared budget"):
        exporter.plan(args)


def test_export_builds_reference_bytes_and_completed_resume_uses_no_aws(tmp_path: Path) -> None:
    pytest.importorskip("duckdb", minversion="1.5")
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
    from tests.name_count_oracle import build_name_count_dicts

    output = tmp_path / "actual"
    args = exporter._parser().parse_args(_arguments(output, "--execute"))
    run_plan = exporter.plan(args)
    session, _, _, athena_stub, s3_stub = _clients()
    _export_responses(athena_stub, s3_stub, run_plan)
    with athena_stub, s3_stub:
        record = exporter.execute_export(args, run_plan, session=session)
        athena_stub.assert_no_pending_responses()
        s3_stub.assert_no_pending_responses()
        assert record["status"] == "complete"
        assert record["author_rows"] == sum(row[2] for row in ROWS)
        assert record["raw_groups"] == len(ROWS)
        mappings, _ = build_name_count_dicts([(first or "", last or "", count) for first, last, count in ROWS])
        expected, _ = write_name_counts_index(tmp_path / "reference", mappings)
        actual = Path(record["index_path"])
        files = ["manifest.json", "first.bin", "last.bin", "first_last.bin", "last_first_initial.bin"]
        before = {name: ((actual / name).read_bytes(), (actual / name).stat().st_mtime_ns) for name in files}
        for name in files:
            assert before[name][0] == (Path(expected) / name).read_bytes()
        args.resume = True
        exporter.execute_export(args, run_plan, session=session)
        assert before == {name: ((actual / name).read_bytes(), (actual / name).stat().st_mtime_ns) for name in files}
        source = output / "source/source_000000.parquet"
        source.write_bytes(source.read_bytes() + b"tamper")
        with pytest.raises(ValueError, match="Saved source object differs"):
            exporter.execute_export(args, run_plan, session=session)


def test_uncapped_workgroup_stops_before_query_submission(tmp_path: Path) -> None:
    pytest.importorskip("duckdb", minversion="1.5")
    args = exporter._parser().parse_args(_arguments(tmp_path / "out", "--execute"))
    run_plan = exporter.plan(args)
    session, _, _, athena_stub, s3_stub = _clients()
    athena_stub.add_response(
        "get_work_group", {"WorkGroup": {"Name": "bounded", "Configuration": {}}}, {"WorkGroup": "bounded"}
    )
    with athena_stub, s3_stub, pytest.raises(ValueError, match="per-query scan cutoff"):
        exporter.execute_export(args, run_plan, session=session)


def test_downloaded_and_complete_local_resume_need_no_aws_sdk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("duckdb", minversion="1.5")
    output = tmp_path / "local"
    source_dir = output / "source"
    source_dir.mkdir(parents=True)
    args = exporter._parser().parse_args(_arguments(output, "--execute", "--resume"))
    run_plan = exporter.plan(args)
    entries = []
    for position, rows in enumerate((ROWS[:6], ROWS[6:])):
        path = source_dir / f"source_{position:06d}.parquet"
        path.write_bytes(_parquet(rows))
        entries.append(
            {
                "path": str(path),
                "s3_uri": f"s3://example/export/data_{position}",
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "sha256": sha256_file(path),
            }
        )
    exporter._save(
        output / "export.json",
        {
            "status": "downloaded",
            "plan": run_plan,
            "execution_id": "cached-export",
            "downloaded": entries,
            "manifest_paths": [entry["s3_uri"] for entry in entries],
        },
    )
    original_import = builtins.__import__

    def reject_aws_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"boto3", "botocore"}:
            raise AssertionError("Local resume must not require the AWS SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_aws_import)
    first = exporter.execute_export(args, run_plan)
    assert first["status"] == "complete"
    index = Path(first["index_path"])
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in index.iterdir()}
    second = exporter.execute_export(args, run_plan)
    assert second["status"] == "complete"
    assert before == {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in index.iterdir()}


def test_resume_reuses_query_id_after_service_failure(tmp_path: Path) -> None:
    pytest.importorskip("duckdb", minversion="1.5")
    session, _, _, athena_stub, s3_stub = _clients()
    from botocore.exceptions import ClientError
    from botocore.stub import ANY

    output = tmp_path / "out"
    args = exporter._parser().parse_args(_arguments(output, "--execute"))
    run_plan = exporter.plan(args)
    athena_stub.add_response(
        "get_work_group",
        {"WorkGroup": {"Name": "bounded", "Configuration": {"BytesScannedCutoffPerQuery": 100_000_000}}},
        {"WorkGroup": "bounded"},
    )
    athena_stub.add_response(
        "start_query_execution",
        {"QueryExecutionId": "retained-query"},
        {
            "QueryString": run_plan["sql"],
            "ClientRequestToken": ANY,
            "QueryExecutionContext": {"Database": "content_ext"},
            "WorkGroup": "bounded",
            "ResultConfiguration": {"OutputLocation": "s3://example/query/"},
        },
    )
    athena_stub.add_client_error("get_query_execution", service_error_code="InternalServerException")
    with athena_stub, s3_stub:
        with pytest.raises(ClientError):
            exporter.execute_export(args, run_plan, session=session)
        record = json.loads((output / "export.json").read_text(encoding="utf-8"))
        assert record["execution_id"] == "retained-query"
        args.resume = True
        athena_stub.add_response(
            "get_query_execution",
            {"QueryExecution": {"QueryExecutionId": "retained-query", "Status": {"State": "FAILED"}}},
            {"QueryExecutionId": "retained-query"},
        )
        with pytest.raises(RuntimeError, match="Author export FAILED"):
            exporter.execute_export(args, run_plan, session=session)
        athena_stub.assert_no_pending_responses()


def test_resume_preserves_idempotency_token_after_lost_submission_response(tmp_path: Path) -> None:
    pytest.importorskip("duckdb", minversion="1.5")
    session, _, _, athena_stub, s3_stub = _clients()
    from botocore.exceptions import ClientError

    output = tmp_path / "out"
    args = exporter._parser().parse_args(_arguments(output, "--execute"))
    run_plan = exporter.plan(args)
    workgroup = {"WorkGroup": {"Name": "bounded", "Configuration": {"BytesScannedCutoffPerQuery": 100_000_000}}}
    athena_stub.add_response("get_work_group", workgroup, {"WorkGroup": "bounded"})
    athena_stub.add_client_error("start_query_execution", service_error_code="InternalServerException")
    with athena_stub, s3_stub:
        with pytest.raises(ClientError):
            exporter.execute_export(args, run_plan, session=session)
        saved = json.loads((output / "export.json").read_text(encoding="utf-8"))
        assert "execution_id" not in saved
        args.resume = True
        athena_stub.add_response("get_work_group", workgroup, {"WorkGroup": "bounded"})
        athena_stub.add_response(
            "start_query_execution",
            {"QueryExecutionId": "same-export"},
            {
                "QueryString": run_plan["sql"],
                "ClientRequestToken": saved["client_request_token"],
                "QueryExecutionContext": {"Database": "content_ext"},
                "WorkGroup": "bounded",
                "ResultConfiguration": {"OutputLocation": "s3://example/query/"},
            },
        )
        athena_stub.add_response(
            "get_query_execution",
            {"QueryExecution": {"Status": {"State": "FAILED"}}},
            {"QueryExecutionId": "same-export"},
        )
        with pytest.raises(RuntimeError, match="Author export FAILED"):
            exporter.execute_export(args, run_plan, session=session)
        athena_stub.assert_no_pending_responses()


def test_query_timeout_cancels_and_records_the_bound(tmp_path: Path) -> None:
    session, athena, _, athena_stub, _ = _clients()
    del session
    path = tmp_path / "export.json"
    record = {
        "execution_id": "slow",
        "submitted_epoch": time.time() - 10,
        "plan": {"max_query_seconds": 1, "max_scan_bytes": 100, "poll_seconds": 1},
    }
    athena_stub.add_response(
        "get_query_execution", {"QueryExecution": {"Status": {"State": "RUNNING"}}}, {"QueryExecutionId": "slow"}
    )
    athena_stub.add_response("stop_query_execution", {}, {"QueryExecutionId": "slow"})
    with athena_stub, pytest.raises(RuntimeError, match="query bound"):
        exporter._wait_query(athena, record, path)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "bound_exceeded"


def test_completed_query_can_resume_after_local_downtime(tmp_path: Path) -> None:
    _, athena, _, athena_stub, _ = _clients()
    path = tmp_path / "export.json"
    record = {
        "execution_id": "complete",
        "submitted_epoch": time.time() - 10,
        "plan": {"max_query_seconds": 1, "max_scan_bytes": 100, "poll_seconds": 1},
    }
    athena_stub.add_response(
        "get_query_execution",
        {
            "QueryExecution": {
                "Status": {"State": "SUCCEEDED"},
                "Statistics": {"TotalExecutionTimeInMillis": 500, "DataScannedInBytes": 50},
            }
        },
        {"QueryExecutionId": "complete"},
    )
    with athena_stub:
        exporter._wait_query(athena, record, path)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "exported"


def test_manifest_rejects_unrelated_objects(tmp_path: Path) -> None:
    del tmp_path
    _, athena, _, athena_stub, _ = _clients()
    athena_stub.add_response(
        "get_query_results",
        {"ResultSet": {"Rows": [{"Data": [{"VarCharValue": "s3://example/unrelated/data"}]}]}},
        {"QueryExecutionId": "export-1", "QueryResultType": "DATA_MANIFEST", "MaxResults": 1000},
    )
    with athena_stub, pytest.raises(ValueError, match="outside the export prefix"):
        exporter._manifest_paths(athena, "export-1", "s3://example/export/", 2)
