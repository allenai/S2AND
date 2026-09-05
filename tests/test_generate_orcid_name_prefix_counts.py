"""Outcome tests for canonical ORCID prefix-count generation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH
from s2and.orcid_prefix_counts import (
    ORCID_PREFIX_DATA_FILENAME,
    ORCID_PREFIX_MANIFEST_FILENAME,
    load_canonical_orcid_prefix_counts,
)
from scripts.production.counts import generate_orcid_name_prefix_counts as generator

ORCID_1 = "0000-0000-0000-0001"
NAME_TUPLES_PATH = Path(PROJECT_ROOT_PATH) / "s2and" / "data" / "s2and_name_tuples_canonical.txt"
NAME_TUPLES_SHA256 = hashlib.sha256(NAME_TUPLES_PATH.read_bytes()).hexdigest()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["raw_orcid", "orcid", "first_name", "middle"])
        writer.writeheader()
        writer.writerows(
            {
                "raw_orcid": row.get("raw_orcid", row.get("orcid")),
                "orcid": row.get("orcid"),
                "first_name": row.get("first_name"),
                "middle": row.get("middle"),
            }
            for row in rows
        )
    return path


def _csv_args(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    output_name: str = "publication",
) -> list[str]:
    guardrails = _write_guardrails(tmp_path / f"{output_name}-guardrails.json")
    return [
        "--input-csv",
        str(_write_rows(tmp_path / f"{output_name}-rows.csv", rows)),
        "--guardrails-json",
        str(guardrails),
        "--output-dir",
        str(tmp_path / output_name),
    ]


def _publication_rows() -> list[dict[str, object]]:
    return [
        {"orcid": f"0000-0000-0000-{index:04d}", "first_name": first, "middle": None}
        for index in range(10)
        for first in ("Alice", "Amy")
    ]


def test_module_entrypoint_publishes_tiny_csv(tmp_path: Path) -> None:
    args = _csv_args(tmp_path, _publication_rows())
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_orcid_name_prefix_counts", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "publication" / ORCID_PREFIX_MANIFEST_FILENAME).is_file()
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert records[0]["event"] == "orcid_prefix_plan"
    assert records[0]["plan"]["source"] == str((tmp_path / "publication-rows.csv").resolve())
    assert records[0]["plan"]["name_tuples_sha256"] == NAME_TUPLES_SHA256
    assert records[-1]["event"] == "orcid_prefix_result"
    output_dir = tmp_path / "publication"
    manifest = json.loads((output_dir / ORCID_PREFIX_MANIFEST_FILENAME).read_bytes())
    loaded = load_canonical_orcid_prefix_counts(output_dir)
    assert {path.name for path in output_dir.iterdir()} == {
        ORCID_PREFIX_DATA_FILENAME,
        ORCID_PREFIX_MANIFEST_FILENAME,
    }
    assert manifest == {"name_tuples_sha256": NAME_TUPLES_SHA256}
    assert loaded.data_sha256 == hashlib.sha256((output_dir / ORCID_PREFIX_DATA_FILENAME).read_bytes()).hexdigest()
    assert loaded.name_tuples_sha256 == NAME_TUPLES_SHA256
    with pytest.raises(TypeError):
        loaded.counts["al"] = {}  # type: ignore[index]


def _write_guardrails(path: Path, **changes: int) -> Path:
    values = {
        "min_source_rows": 1,
        "max_source_rows": 100,
        "max_names_per_orcid": 100,
        "max_pair_keys": 1_000_000,
        "min_orcid_pair_keys": 1,
        **changes,
    }
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _publish_tiny_csv(tmp_path: Path) -> Path:
    assert generator.main(_csv_args(tmp_path, _publication_rows())) == 0
    return tmp_path / "publication"


def test_existing_or_empty_publication_is_rejected_without_partial_output(tmp_path: Path) -> None:
    output_dir = _publish_tiny_csv(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        generator.main(
            _csv_args(
                tmp_path,
                [{"orcid": ORCID_1, "first_name": "Alice", "middle": None}],
                output_name=output_dir.name,
            )
        )

    empty_output = tmp_path / "empty"
    with pytest.raises(RuntimeError, match="zero source rows"):
        generator.main(_csv_args(tmp_path, [], output_name=empty_output.name))
    assert not empty_output.exists()


def test_full_run_rejects_invalid_expansion_guard_before_reading_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guardrails = _write_guardrails(tmp_path / "guardrails.json", max_names_per_orcid=1)
    export = _write_rows(tmp_path / "export.csv", [])
    monkeypatch.setattr(
        generator,
        "_load_reviewed_csv_rows",
        lambda *_args: pytest.fail("rows read before guardrail validation"),
    )

    with pytest.raises(ValueError, match="max_names_per_orcid must be at least 2"):
        generator.main(
            [
                "--input-csv",
                str(export),
                "--output-dir",
                str(tmp_path / "output"),
                "--guardrails-json",
                str(guardrails),
            ]
        )


def test_reviewed_csv_uses_full_guardrails_without_internal_warehouse_client(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export = tmp_path / "orcid-export.csv"
    rows = "".join(
        f"0000-0000-0000-{index:04d},0000-0000-0000-{index:04d},{first},\n"
        for index in range(10)
        for first in ("Alice", "Amy")
    )
    export.write_text(
        f"raw_orcid,orcid,first_name,middle\n{rows}",
        encoding="utf-8",
    )
    guardrails = _write_guardrails(tmp_path / "guardrails.json", min_source_rows=20)

    assert (
        generator.main(
            [
                "--input-csv",
                str(export),
                "--guardrails-json",
                str(guardrails),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 0
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert records[0]["event"] == "orcid_prefix_plan"
    assert records[0]["plan"]["source"] == str(export.resolve())
    assert (tmp_path / "output" / ORCID_PREFIX_DATA_FILENAME).is_file()


@pytest.mark.parametrize(
    "payload",
    (
        "raw_orcid,orcid,orcid,first_name,middle\n",
        "raw_orcid,orcid,first_name,middle,source\n",
        f"raw_orcid,orcid,first_name,middle\n{ORCID_1},{ORCID_1},Alice,,extra\n",
    ),
)
def test_reviewed_csv_rejects_ambiguous_columns(tmp_path: Path, payload: str) -> None:
    export = tmp_path / "orcid-export.csv"
    export.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="exact header|more values than columns"):
        list(generator._load_reviewed_csv_rows(export))


def test_builder_deduplicates_rows() -> None:
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, metrics = generator.build_prefix_counts_from_sorted_rows(
        rows,
        [],
        min_orcid_count=1,
    )
    duplicate_counts, duplicate_metrics = generator.build_prefix_counts_from_sorted_rows(
        [*rows, rows[-1]],
        [],
        min_orcid_count=1,
    )

    assert counts == duplicate_counts
    assert metrics["source_rows"] == 2
    assert duplicate_metrics["source_rows"] == 3
    assert metrics["orcid_pair_keys_after_threshold"] > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"max_names_per_orcid": 1}, "at least 2"),
        ({"max_names_per_orcid": 2}, "more than max_names_per_orcid=2"),
        ({"max_source_rows": 1}, "max_source_rows=1"),
        ({"max_pair_keys": 1}, "max_pair_keys=1"),
    ),
)
def test_builder_enforces_only_live_expansion_bounds(kwargs: dict[str, int], message: str) -> None:
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
        {"orcid": ORCID_1, "first_name": "Ava", "middle": None},
    ]

    with pytest.raises(ValueError, match=message):
        generator.build_prefix_counts_from_sorted_rows(rows, [], min_orcid_count=1, **kwargs)


def test_canonical_loader_rejects_semantically_invalid_counts(tmp_path: Path) -> None:
    output_dir = _publish_tiny_csv(tmp_path)
    data_path = output_dir / ORCID_PREFIX_DATA_FILENAME
    counts = json.loads(data_path.read_text())
    left = next(iter(counts))
    right = next(iter(counts[left]))
    counts[left][right] = 0
    data_path.write_text(json.dumps(counts))

    with pytest.raises(ValueError, match="positive integers"):
        load_canonical_orcid_prefix_counts(output_dir)


@pytest.mark.parametrize(
    "manifest",
    (
        {},
        {"name_tuples_sha256": NAME_TUPLES_SHA256.upper()},
        {"name_tuples_sha256": NAME_TUPLES_SHA256, "source": "extra"},
    ),
)
def test_canonical_loader_requires_exact_minimal_tuple_dependency(
    tmp_path: Path,
    manifest: dict[str, str],
) -> None:
    output_dir = _publish_tiny_csv(tmp_path)
    manifest_path = output_dir / ORCID_PREFIX_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase name_tuples_sha256"):
        load_canonical_orcid_prefix_counts(output_dir)


def test_prefix_pair_lookup_is_order_independent() -> None:
    expected = generator.prefix_pairs_for_names("alice", "amy")
    assert expected == generator.prefix_pairs_for_names("amy", "alice")
    assert all(left < right for left, right in expected)
    assert not generator.prefix_pairs_for_names("alice", "bob")
