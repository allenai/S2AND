from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from s2and.incremental_linking.feature_block import (
    cleanup_stale_name_counts_generations,
    write_name_counts_index,
)
from tests.helpers import tiny_name_counts_provenance


def _write_generation(index_dir: Path, generation_name: str) -> None:
    generation_dir = index_dir / "generations" / generation_name
    generation_dir.mkdir(parents=True)
    for filename in ("first.bin", "last.bin", "first_last.bin", "last_first_initial.bin"):
        (generation_dir / filename).write_bytes(b"index")
    (generation_dir / ".published").write_text("", encoding="utf-8")


def test_write_name_counts_index_retains_committed_generation_after_superseding_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    provenance = tiny_name_counts_provenance()
    index_dir = tmp_path / "name_counts_index"
    manifest_path = index_dir / "manifest.json"
    original_replace = Path.replace
    committed_generation_name: str | None = None
    superseding_generation_name = "gen-superseding"

    def replace_then_supersede(path: Path, target: str | Path) -> Path:
        nonlocal committed_generation_name
        result = original_replace(path, target)
        if Path(target) != manifest_path or not path.name.startswith(".manifest."):
            return result
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        committed_generation_name = Path(manifest["files"]["first"]["path"]).parts[1]
        shutil.copytree(
            index_dir / "generations" / committed_generation_name,
            index_dir / "generations" / superseding_generation_name,
        )
        for entry in manifest["files"].values():
            entry["path"] = f"generations/{superseding_generation_name}/{Path(entry['path']).name}"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "replace", replace_then_supersede)
    write_name_counts_index(tmp_path, mappings, provenance, overwrite=True)

    assert committed_generation_name is not None
    assert (index_dir / "generations" / committed_generation_name).is_dir()
    assert (index_dir / "generations" / superseding_generation_name).is_dir()


def test_write_name_counts_index_removes_generation_when_manifest_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    provenance = tiny_name_counts_provenance()
    manifest_path = tmp_path / "name_counts_index" / "manifest.json"
    original_replace = Path.replace

    def fail_manifest_replace(path: Path, target: str | Path) -> Path:
        if Path(target) == manifest_path and path.name.startswith(".manifest."):
            raise OSError("injected manifest replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected manifest replace failure"):
        write_name_counts_index(tmp_path, mappings, provenance, overwrite=True)

    assert not manifest_path.exists()
    assert list((manifest_path.parent / "generations").iterdir()) == []


def test_write_name_counts_index_keeps_manifest_absent_when_marker_write_fails(tmp_path, monkeypatch) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    provenance = tiny_name_counts_provenance()
    original_open = Path.open

    def fail_published_marker(path: Path, *args, **kwargs):
        if path.name == ".published":
            raise OSError("marker write failed")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_published_marker)

    try:
        write_name_counts_index(tmp_path, mappings, provenance, overwrite=True)
    except OSError as exc:
        assert "marker write failed" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("write_name_counts_index should fail when .published cannot be written")

    index_dir = tmp_path / "name_counts_index"
    manifest_path = index_dir / "manifest.json"
    assert not manifest_path.exists()
    generation_root = index_dir / "generations"
    generations = [path for path in generation_root.iterdir() if path.is_dir()]
    assert generations == []


@pytest.mark.parametrize("corruption", ("published_marker", "byte_count", "sha256", "manifest_shape"))
def test_write_name_counts_index_rebuilds_corrupted_matching_generation(
    tmp_path: Path,
    corruption: str,
) -> None:
    mappings = ({"ada": 1}, {"lovelace": 1}, {"ada lovelace": 1}, {"lovelace a": 1})
    provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path, mappings, provenance)
    index_dir = Path(index_path)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_first_path = str(manifest["files"]["first"]["path"])
    first_path = index_dir / original_first_path

    if corruption == "published_marker":
        (first_path.parent / ".published").unlink()
    elif corruption == "byte_count":
        manifest["files"]["first"]["byte_count"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "sha256":
        payload = bytearray(first_path.read_bytes())
        payload[-1] ^= 1
        first_path.write_bytes(payload)
    else:
        manifest_path.write_text("[]", encoding="utf-8")

    _index_path, rebuilt_metrics = write_name_counts_index(tmp_path, mappings, provenance)

    rebuilt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebuilt_metrics["reused"] is False
    assert rebuilt_manifest["files"]["first"]["path"] != original_first_path
    assert write_name_counts_index(tmp_path, mappings, provenance)[1] == {"reused": True}


def test_cleanup_stale_name_counts_generations_keeps_manifest_generation(tmp_path: Path) -> None:
    index_dir = tmp_path / "name_counts_index"
    _write_generation(index_dir, "gen-old")
    _write_generation(index_dir, "gen-current")
    manifest = {
        "files": {
            "first": {"path": "generations/gen-current/first.bin"},
            "last": {"path": "generations/gen-current/last.bin"},
            "first_last": {"path": "generations/gen-current/first_last.bin"},
            "last_first_initial": {"path": "generations/gen-current/last_first_initial.bin"},
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metrics = cleanup_stale_name_counts_generations(index_dir)

    assert metrics == {"removed_generation_count": 1}
    assert not (index_dir / "generations" / "gen-old").exists()
    assert (index_dir / "generations" / "gen-current").exists()


def test_cleanup_stale_name_counts_generations_refuses_missing_manifest(tmp_path: Path) -> None:
    index_dir = tmp_path / "name_counts_index"
    _write_generation(index_dir, "gen-old")

    with pytest.raises(ValueError, match="without a resolvable current manifest"):
        cleanup_stale_name_counts_generations(index_dir)

    assert (index_dir / "generations" / "gen-old").exists()


def test_cleanup_stale_name_counts_generations_refuses_manifest_outside_generations(tmp_path: Path) -> None:
    index_dir = tmp_path / "name_counts_index"
    _write_generation(index_dir, "gen-old")
    external = tmp_path / "external"
    external.mkdir()
    for filename in ("first.bin", "last.bin", "first_last.bin", "last_first_initial.bin"):
        (external / filename).write_bytes(b"index")
    manifest = {
        "files": {
            "first": {"path": str(external / "first.bin")},
            "last": {"path": str(external / "last.bin")},
            "first_last": {"path": str(external / "first_last.bin")},
            "last_first_initial": {"path": str(external / "last_first_initial.bin")},
        }
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="without a resolvable current manifest"):
        cleanup_stale_name_counts_generations(index_dir)

    assert (index_dir / "generations" / "gen-old").exists()
