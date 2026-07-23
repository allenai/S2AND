"""Guardrail and publication tests for canonical name-count generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import NAME_COUNTS_PROVENANCE_SCHEMA_VERSION
from scripts.production.counts import generate_name_counts


def _fixture(path: Path) -> Path:
    path.write_text(
        json.dumps(
            [
                {"first_name": "Abd-al", "last_name": "Sattar", "count": 4},
                {"first_name": "Abd al", "last_name": "Sattar", "count": 3},
                {"first_name": "", "last_name": "", "count": 2},
            ]
        ),
        encoding="utf-8",
    )
    return path


def _run_fixture(tmp_path: Path, *extra: str) -> int:
    return generate_name_counts.main(
        [
            "--fixture-input",
            str(_fixture(tmp_path / "rows.json")),
            "--source-snapshot-id",
            "fixture-2026-07-09",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_dry_run_does_not_query_or_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        generate_name_counts,
        "_query_rows",
        lambda _limit: pytest.fail("dry-run must not query"),
    )

    assert (
        generate_name_counts.main(
            [
                "--run-full",
                "--source-snapshot-id",
                "snapshot-1",
                "--output-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not (tmp_path / "name_counts_index").exists()


def test_fixture_publishes_only_native_index_with_audit_provenance(tmp_path: Path) -> None:
    assert _run_fixture(tmp_path) == 0

    manifest = json.loads((tmp_path / "name_counts_index" / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["source_provenance"]
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
    assert provenance["schema_version"] == NAME_COUNTS_PROVENANCE_SCHEMA_VERSION
    assert provenance["source_snapshot_id"] == "fixture-2026-07-09"
    assert provenance["source_row_count"] == 3
    assert provenance["rejected_row_count"] == 1
    assert len(provenance["selected_rows_sha256"]) == 64
    assert set(manifest["files"]) == {"first", "last", "first_last", "last_first_initial"}
    assert not (tmp_path / "name_counts").exists()
    assert not list(tmp_path.rglob("*.pickle"))


def test_existing_publication_requires_explicit_overwrite(tmp_path: Path) -> None:
    _run_fixture(tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        _run_fixture(tmp_path)

    assert _run_fixture(tmp_path, "--overwrite") == 0


def test_fixture_limit_is_applied_before_aggregation(tmp_path: Path) -> None:
    assert _run_fixture(tmp_path, "--limit", "1") == 0

    manifest = json.loads((tmp_path / "name_counts_index" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_provenance"]["source_row_count"] == 1


def test_build_name_count_dicts_preserves_canonical_counts() -> None:
    mappings, metrics = generate_name_counts.build_name_count_dicts(
        [
            ("Abd-al", "Sattar", 4),
            ("abd-al", "sattar", 3),
        ]
    )

    first, last, first_last, last_first_initial = mappings
    assert first["abd al"] == 7
    assert last["sattar"] == 7
    assert first_last["abd al sattar"] == 7
    assert last_first_initial["sattar a"] == 7
    assert metrics["source_row_count"] == 2
