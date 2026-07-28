"""Outcome tests for canonical name-count generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import NAME_COUNTS_INDEX_SCHEMA_VERSION
from scripts.production.counts import generate_name_counts

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_module_entrypoint_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_name_counts", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--fixture-input" in completed.stdout
    assert "--input-csv" in completed.stdout


def _write_rows(path: Path, rows: object) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _fixture_args(tmp_path: Path, rows: object) -> list[str]:
    return [
        "--fixture-input",
        str(_write_rows(tmp_path / "rows.json", rows)),
        "--output-dir",
        str(tmp_path),
    ]


def test_module_entrypoint_publishes_fixture(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, [{"first_name": "Alice", "last_name": "Smith", "count": 2}])
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_name_counts", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "name_counts_index" / "manifest.json").is_file()


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


def test_fixture_publishes_verified_native_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        {"first_name": "Abd-al", "last_name": "Sattar", "count": 4},
        {"first_name": "Abd al", "last_name": "Sattar", "count": 3},
        {"first_name": "", "last_name": "", "count": 2},
    ]

    assert generate_name_counts.main(_fixture_args(tmp_path, rows)) == 0

    manifest = json.loads((tmp_path / "name_counts_index" / "manifest.json").read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert set(manifest) == {"schema_version", "normalization_version", "files"}
    assert manifest["schema_version"] == NAME_COUNTS_INDEX_SCHEMA_VERSION
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
    assert '"source_row_count": 3' in output
    assert '"rejected_row_count": 1' in output
    assert set(manifest["files"]) == {"first", "last", "first_last", "last_first_initial"}


def test_fixture_limit_changes_selected_content(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rows = [
        {"first_name": "Alice", "last_name": "Smith", "count": 2},
        {"first_name": "Amy", "last_name": "Jones", "count": 2},
    ]

    assert generate_name_counts.main([*_fixture_args(tmp_path, rows), "--limit", "1"]) == 0

    output = capsys.readouterr().out
    assert '"source_row_count": 1' in output
    for key in ("first", "last", "first_last", "last_first_initial"):
        assert f'"{key}": 1' in output


def test_empty_or_existing_publication_is_rejected(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(RuntimeError, match="zero source rows"):
        generate_name_counts.main(_fixture_args(empty_root, []))
    assert not (empty_root / "name_counts_index").exists()

    populated_root = tmp_path / "populated"
    populated_root.mkdir()
    args = _fixture_args(populated_root, [{"first_name": "Alice", "last_name": "Smith", "count": 2}])
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
    with pytest.raises(ValueError, match="--guardrails-json"):
        generate_name_counts.main(args)

    guardrails = _write_guardrails(tmp_path / "guardrails.json", max_source_rows=17)
    assert generate_name_counts.main([*args, "--guardrails-json", str(guardrails)]) == 0
    assert (tmp_path / "output" / "name_counts_index" / "manifest.json").is_file()

    source.write_text("first_name,last_name\nAlice,Smith\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        list(generate_name_counts._reviewed_csv_rows(source))


@pytest.mark.parametrize(
    "values",
    [
        {"max_source_rows": 1},
        {
            "min_source_rows": 2,
            "max_source_rows": 1,
            "min_keys_per_mapping": 1,
            "max_keys_per_mapping": 2,
        },
    ],
)
def test_invalid_guardrail_authority_fails_before_execution(tmp_path: Path, values: dict[str, int]) -> None:
    path = tmp_path / "guardrails.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="guardrail"):
        generate_name_counts.main(
            [
                "--input-csv",
                str(path),
                "--output-dir",
                str(tmp_path / "output"),
                "--guardrails-json",
                str(path),
                "--dry-run",
            ]
        )
