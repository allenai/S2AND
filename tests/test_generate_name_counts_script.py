"""Outcome tests for canonical name-count generation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from scripts.production.counts import generate_name_counts

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["first_name", "last_name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _csv_args(tmp_path: Path, rows: list[dict[str, object]]) -> list[str]:
    guardrails = _write_guardrails(tmp_path / "guardrails.json")
    return [
        "--input-csv",
        str(_write_rows(tmp_path / "rows.csv", rows)),
        "--guardrails-json",
        str(guardrails),
        "--output-dir",
        str(tmp_path),
    ]


def test_module_entrypoint_publishes_tiny_csv(tmp_path: Path) -> None:
    args = _csv_args(
        tmp_path,
        [
            {"first_name": "Abd-al", "last_name": "Sattar", "count": 4},
            {"first_name": "Abd al", "last_name": "Sattar", "count": 3},
            {"first_name": "", "last_name": "", "count": 2},
        ],
    )
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_name_counts", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((tmp_path / "name_counts_index" / "manifest.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert manifest["kind"] == "s2and_name_counts"
    assert manifest["format_version"] == PUBLIC_DATA_FORMAT_VERSION
    assert set(manifest["files"]) == {"first", "last", "first_last", "last_first_initial"}
    assert records[0]["event"] == "name_counts_plan"
    assert records[-1]["event"] == "name_counts_result"
    assert records[-1]["result"]["source_row_count"] == 3
    assert records[-1]["result"]["rejected_row_count"] == 1


def _write_guardrails(path: Path, **changes: int) -> Path:
    values = {
        "min_source_rows": 1,
        "max_source_rows": 100,
        "min_keys_per_mapping": 1,
        "max_keys_per_mapping": 100,
        **changes,
    }
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_empty_or_existing_publication_is_rejected(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(RuntimeError, match="zero source rows"):
        generate_name_counts.main(_csv_args(empty_root, []))
    assert not (empty_root / "name_counts_index").exists()

    populated_root = tmp_path / "populated"
    populated_root.mkdir()
    args = _csv_args(populated_root, [{"first_name": "Alice", "last_name": "Smith", "count": 2}])
    assert generate_name_counts.main(args) == 0
    with pytest.raises(FileExistsError, match="already exists"):
        generate_name_counts.main(args)


def test_builder_enforces_live_row_and_mapping_bounds() -> None:
    rows = [("Alice", "Smith", 2), ("Amy", "Jones", 2)]
    with pytest.raises(ValueError, match="max_source_rows=1"):
        generate_name_counts.build_name_count_dicts(rows, max_source_rows=1)
    with pytest.raises(ValueError, match="max_keys_per_mapping=1"):
        generate_name_counts.build_name_count_dicts(rows, max_keys_per_mapping=1)


def test_full_reviewed_csv_requires_guardrails_and_publishes(tmp_path: Path) -> None:
    source = tmp_path / "name_counts.csv"
    source.write_text("first_name,last_name,count\nAlice,Smith,2\n", encoding="utf-8")
    args = [
        "--input-csv",
        str(source),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    with pytest.raises(SystemExit):
        generate_name_counts.main(args)

    guardrails = _write_guardrails(tmp_path / "guardrails.json", max_source_rows=17)
    assert generate_name_counts.main([*args, "--guardrails-json", str(guardrails)]) == 0
    assert (tmp_path / "output" / "name_counts_index" / "manifest.json").is_file()

    source.write_text("first_name,last_name\nAlice,Smith\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        list(generate_name_counts._reviewed_csv_rows(source))


def test_invalid_guardrail_authority_fails_before_execution(tmp_path: Path) -> None:
    cases = [
        {"max_source_rows": 1},
        {
            "min_source_rows": 2,
            "max_source_rows": 1,
            "min_keys_per_mapping": 1,
            "max_keys_per_mapping": 2,
        },
    ]
    for index, values in enumerate(cases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        path = case_root / "guardrails.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        with pytest.raises(ValueError, match="guardrail"):
            generate_name_counts.main(
                [
                    "--input-csv",
                    str(path),
                    "--output-dir",
                    str(case_root / "output"),
                    "--guardrails-json",
                    str(path),
                ]
            )
