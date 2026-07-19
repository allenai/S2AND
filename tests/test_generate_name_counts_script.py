"""Guardrail and publication tests for canonical name-count generation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from s2and.consts import NORMALIZATION_VERSION
from scripts.production.counts import generate_name_counts


def test_import_is_side_effect_free_without_internal_pys2(tmp_path: Path) -> None:
    assert not (tmp_path / "name_counts").exists()
    assert callable(generate_name_counts.main)


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
    assert not (tmp_path / "name_counts").exists()


def test_fixture_generation_publishes_data_before_manifest(tmp_path: Path) -> None:
    fixture_path = tmp_path / "rows.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"first_name": "Abd-al", "last_name": "Sattar", "count": 4},
                {"first_name": "Abd al", "last_name": "Sattar", "count": 3},
                {"first_name": "", "last_name": "", "count": 2},
            ]
        ),
        encoding="utf-8",
    )

    assert (
        generate_name_counts.main(
            [
                "--fixture-input",
                str(fixture_path),
                "--source-snapshot-id",
                "fixture-2026-07-09",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    root = tmp_path / "name_counts"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    pickle_path = root / manifest["files"]["pickle"]
    provenance_path = root / manifest["files"]["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert pickle_path.is_file()
    assert provenance_path.is_file()
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
    assert provenance["generation_id"] == manifest["generation_id"]
    assert hashlib.sha256(pickle_path.read_bytes()).hexdigest() == manifest["pickle_sha256"]
    assert hashlib.sha256(provenance_path.read_bytes()).hexdigest() == manifest["provenance_sha256"]
    assert provenance_path.stat().st_size == manifest["provenance_byte_count"]
    assert provenance["source_row_count"] == 3
    assert provenance["selected_row_count"] == 3
    assert len(provenance["selected_rows_sha256"]) == 64
    assert provenance["rejected_row_count"] == 1

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_name_counts.main(
            [
                "--fixture-input",
                str(fixture_path),
                "--source-snapshot-id",
                "fixture-second",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_fixture_limit_is_bounded_and_reported(tmp_path: Path) -> None:
    fixture_path = tmp_path / "rows.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"first_name": "A", "last_name": "One", "count": 2},
                {"first_name": "B", "last_name": "Two", "count": 2},
            ]
        ),
        encoding="utf-8",
    )
    generate_name_counts.main(
        [
            "--fixture-input",
            str(fixture_path),
            "--source-snapshot-id",
            "fixture-limit",
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    manifest = json.loads((tmp_path / "name_counts" / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "name_counts" / manifest["files"]["provenance"]).read_text(encoding="utf-8"))
    assert provenance["source_row_count"] == 1


def test_failed_manifest_replace_removes_only_the_uncommitted_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    metrics = {
        "source_row_count": 1,
        "selected_row_count": 1,
        "selected_rows_sha256": "b" * 64,
        "rejected_row_count": 0,
    }
    generate_name_counts.publish_name_counts(
        mappings,
        output_dir=tmp_path,
        source_snapshot_id="first",
        source_kind="fixture",
        query_digest="a" * 64,
        row_metrics=metrics,
        overwrite=False,
    )
    root = tmp_path / "name_counts"
    original_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    original_replace = Path.replace

    def fail_manifest_replace(path: Path, target: Path) -> Path:
        if Path(target) == root / "manifest.json" and path.name.startswith(".manifest."):
            raise OSError("injected manifest replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected manifest replace failure"):
        generate_name_counts.publish_name_counts(
            mappings,
            output_dir=tmp_path,
            source_snapshot_id="second",
            source_kind="fixture",
            query_digest="a" * 64,
            row_metrics=metrics,
            overwrite=True,
        )

    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == original_manifest
    generations = [path for path in (root / "generations").iterdir() if path.is_dir()]
    assert [path.name for path in generations] == [original_manifest["generation_id"]]


def test_committed_generation_is_retained_after_a_superseding_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    metrics = {
        "source_row_count": 1,
        "selected_row_count": 1,
        "selected_rows_sha256": "b" * 64,
        "rejected_row_count": 0,
    }
    root = tmp_path / "name_counts"
    manifest_path = tmp_path / "name_counts" / "manifest.json"
    original_replace = Path.replace
    committed_generation_id: str | None = None
    superseding_generation_id = "superseding"

    def replace_then_supersede(path: Path, target: str | Path) -> Path:
        nonlocal committed_generation_id
        result = original_replace(path, target)
        if Path(target) != manifest_path or not path.name.startswith(".manifest."):
            return result
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        committed_generation_id = manifest["generation_id"]
        shutil.copytree(
            root / "generations" / committed_generation_id,
            root / "generations" / superseding_generation_id,
        )
        manifest["generation_id"] = superseding_generation_id
        for key, relative_path in manifest["files"].items():
            manifest["files"][key] = f"generations/{superseding_generation_id}/{Path(relative_path).name}"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "replace", replace_then_supersede)
    generate_name_counts.publish_name_counts(
        mappings,
        output_dir=tmp_path,
        source_snapshot_id="fixture",
        source_kind="fixture",
        query_digest="a" * 64,
        row_metrics=metrics,
        overwrite=False,
    )

    assert committed_generation_id is not None
    assert (root / "generations" / committed_generation_id).is_dir()
    assert (root / "generations" / superseding_generation_id).is_dir()


def test_failed_publication_cleans_uncommitted_generation_without_reading_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "name_counts"
    root.mkdir()
    manifest_path = root / "manifest.json"
    manifest_path.write_text("{", encoding="utf-8")
    original_replace = Path.replace

    def fail_manifest_replace(path: Path, target: Path) -> Path:
        if Path(target) == manifest_path and path.name.startswith(".manifest."):
            raise OSError("injected primary replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected primary replace failure") as exc_info:
        generate_name_counts.publish_name_counts(
            ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1}),
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_kind="fixture",
            query_digest="a" * 64,
            row_metrics={
                "source_row_count": 1,
                "selected_row_count": 1,
                "selected_rows_sha256": "b" * 64,
                "rejected_row_count": 0,
            },
            overwrite=True,
        )

    assert not getattr(exc_info.value, "__notes__", [])
    assert list((root / "generations").iterdir()) == []
