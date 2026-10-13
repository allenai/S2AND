"""Parity and recovery tests for the bounded Parquet count producer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("duckdb")

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from scripts.production.counts import generate_name_counts_from_parquet as producer
from scripts.production.counts.generate_name_counts import build_name_count_dicts

ROWS = [
    ("Ána", "García", 1),
    ("Ana", "Garcia", 2),
    ("A", "Garcia", 3),
    (None, "Garcia", 2),
    ("", "", 5),
    ("Zoë", None, 2),
    ("Jean-Paul", "van der Berg", 2),
    ("Jean Paul", "van der Berg", 1),
    ("雪", "王", 2),
    ("Only", "Singleton", 1),
    ("Mary", "O'Neill", 4),
]


def _source(root: Path, rows: Sequence[tuple[str | None, str | None, int]] | None = None) -> Path:
    root.mkdir()
    entries = []
    schema = pa.schema([("first_name", pa.string()), ("last_name", pa.string()), ("mention_count", pa.int64())])
    selected = ROWS if rows is None else rows
    for position, batch_rows in enumerate((selected[:5], selected[5:])):
        path = root / f"source_{position}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [{"first_name": first, "last_name": last, "mention_count": count} for first, last, count in batch_rows],
                schema=schema,
            ),
            path,
        )
        entries.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "rows": len(batch_rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    return manifest


def _build(manifest: Path, output: Path, *, resume: bool = False) -> dict:
    return producer.build_name_counts_from_parquet(
        manifest,
        output,
        expected_source_rows=len(ROWS),
        expected_source_mentions=sum(row[2] for row in ROWS),
        threads=1,
        resume=resume,
    )


def test_module_cli_matches_the_production_writer_bytes_and_native_lookups(tmp_path: Path) -> None:
    manifest = _source(tmp_path / "source")
    output = tmp_path / "actual"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.production.counts.generate_name_counts_from_parquet",
            "--input-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--expected-source-rows",
            str(len(ROWS)),
            "--expected-source-mentions",
            str(sum(row[2] for row in ROWS)),
            "--threads",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    mappings, _ = build_name_count_dicts([(first or "", last or "", count) for first, last, count in ROWS])
    reference_path, _ = write_name_counts_index(tmp_path / "reference", mappings)
    actual = output / "name_counts_index"
    for filename in ("manifest.json", *(f"{kind}.bin" for kind in producer.KINDS)):
        assert (actual / filename).read_bytes() == (Path(reference_path) / filename).read_bytes()
    index = NameCountsIndex.open(actual)
    for position, mapping in enumerate(mappings):
        keys = [*mapping, "absent-key", None]
        columns = [[None] * len(keys) for _ in producer.KINDS]
        columns[position] = keys
        np.testing.assert_array_equal(index.lookup_many(*columns)[position], [*mapping.values(), 1.0, np.nan])
    assert mappings[0]["ana"] == 3
    assert "only singleton" not in mappings[2]
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["source_rows"] == len(ROWS)
    assert record["source_mentions"] == sum(row[2] for row in ROWS)


def test_completed_resume_preserves_every_artifact_and_rejects_changed_bounds(tmp_path: Path) -> None:
    manifest = _source(tmp_path / "source")
    output = tmp_path / "actual"
    _build(manifest, output)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output / "name_counts_index").iterdir()
    }
    _build(manifest, output, resume=True)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output / "name_counts_index").iterdir()
    }
    assert before == after
    with pytest.raises(FileExistsError, match="not empty"):
        _build(manifest, output)
    with pytest.raises(ValueError, match="identity differ"):
        producer.build_name_counts_from_parquet(
            manifest,
            output,
            expected_source_rows=len(ROWS),
            expected_source_mentions=999,
            resume=True,
        )


def test_initial_only_source_preserves_empty_informative_name_mappings(tmp_path: Path) -> None:
    rows = [("J", "Smith", 2), (None, "Smith", 1)]
    manifest = _source(tmp_path / "source", rows)
    output = tmp_path / "actual"
    result = producer.build_name_counts_from_parquet(
        manifest,
        output,
        expected_source_rows=2,
        expected_source_mentions=3,
        threads=1,
    )
    assert result["stages"]["first"]["keys"] == 0
    assert result["stages"]["first_last"]["keys"] == 0
    mappings, _ = build_name_count_dicts([(first or "", last, count) for first, last, count in rows])
    reference, _ = write_name_counts_index(tmp_path / "reference", mappings)
    for kind in producer.KINDS:
        assert (output / "name_counts_index" / f"{kind}.bin").read_bytes() == (
            Path(reference) / f"{kind}.bin"
        ).read_bytes()


def test_source_checksum_and_population_fail_before_publication(tmp_path: Path) -> None:
    manifest = _source(tmp_path / "source")
    with pytest.raises(ValueError, match="expected rows"):
        producer.build_name_counts_from_parquet(
            manifest,
            tmp_path / "wrong_rows",
            expected_source_rows=len(ROWS) + 1,
            expected_source_mentions=sum(row[2] for row in ROWS),
        )
    with pytest.raises(ValueError, match="Source population differs"):
        producer.build_name_counts_from_parquet(
            manifest,
            tmp_path / "wrong_mentions",
            expected_source_rows=len(ROWS),
            expected_source_mentions=999,
        )
    assert not (tmp_path / "wrong_mentions/name_counts_index").exists()
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum differs"):
        _build(manifest, tmp_path / "wrong_sha")
    assert not (tmp_path / "wrong_sha").exists()


def test_failed_binary_stage_resumes_existing_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _source(tmp_path / "source")
    output = tmp_path / "actual"
    original = producer._write_sorted_name_count_records
    calls = 0

    def fail_second_file(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("test disk failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(producer, "_write_sorted_name_count_records", fail_second_file)
    with pytest.raises(OSError, match="disk failure"):
        _build(manifest, output)
    first = output / ".name_counts_index/first.bin"
    first_bytes, first_mtime = first.read_bytes(), first.stat().st_mtime_ns
    assert not (output / "name_counts_index").exists()
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    monkeypatch.setattr(producer, "_write_sorted_name_count_records", original)
    _build(manifest, output, resume=True)
    restored = output / "name_counts_index/first.bin"
    assert (restored.read_bytes(), restored.stat().st_mtime_ns) == (first_bytes, first_mtime)
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["status"] == "complete"
