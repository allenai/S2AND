"""Guardrail and publication tests for canonical name-count generation."""

from __future__ import annotations

import hashlib
import json
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


@pytest.mark.parametrize("failure_kind", ["read_error", "malformed_json"])
def test_post_replace_manifest_inspection_failure_retains_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    metrics = {
        "source_row_count": 1,
        "selected_row_count": 1,
        "selected_rows_sha256": "b" * 64,
        "rejected_row_count": 0,
    }
    manifest_path = tmp_path / "name_counts" / "manifest.json"
    if failure_kind == "read_error":
        original_read_text = Path.read_text

        def fail_published_manifest_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == manifest_path and path.exists():
                raise OSError("injected post-replace read failure")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_published_manifest_read)
        expected_error = OSError
        expected_message = "Unable to read published name-count manifest"
    else:
        original_replace = Path.replace

        def corrupt_replaced_manifest(path: Path, target: Path) -> Path:
            result = original_replace(path, target)
            if Path(target) == manifest_path and path.name.startswith(".manifest."):
                Path(target).write_text("{", encoding="utf-8")
            return result

        monkeypatch.setattr(Path, "replace", corrupt_replaced_manifest)
        expected_error = ValueError
        expected_message = "manifest is invalid JSON"

    with pytest.raises(expected_error, match=expected_message):
        generate_name_counts.publish_name_counts(
            mappings,
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_kind="fixture",
            query_digest="a" * 64,
            row_metrics=metrics,
            overwrite=False,
        )

    generations = list((tmp_path / "name_counts" / "generations").iterdir())
    assert len(generations) == 1
    assert generations[0].is_dir()
    assert manifest_path.is_file()


def test_invalid_manifest_during_failed_publication_does_not_mask_primary_error(
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

    assert "Retained generation" in "\n".join(exc_info.value.__notes__)
    assert len(list((root / "generations").iterdir())) == 1
