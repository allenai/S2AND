from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts._rust_suite import transfer_mini_cmd


def test_transfer_mini_resolve_ingest_auto_is_backend_specific() -> None:
    assert transfer_mini_cmd._resolve_ingest("auto", "python") == "json"
    assert transfer_mini_cmd._resolve_ingest("auto", "rust") == "arrow"

    with pytest.raises(ValueError, match="Python-only"):
        transfer_mini_cmd._resolve_ingest("json", "rust")


def test_transfer_mini_prediction_arrow_manifest_does_not_require_clusters(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dummy"
    dataset_root.mkdir()
    paths: dict[str, str] = {}
    for key in ("signatures", "papers", "paper_authors"):
        artifact_path = dataset_root / f"{key}.arrow"
        artifact_path.touch()
        paths[key] = artifact_path.name
    (dataset_root / "manifest.json").write_text(json.dumps({"paths": paths}), encoding="utf-8")

    resolved, clusters_path = transfer_mini_cmd._resolve_arrow_dataset_paths(
        str(tmp_path),
        "dummy",
        require_clusters=False,
    )

    assert clusters_path is None
    assert set(resolved) == set(paths)
    assert all(Path(path).is_absolute() for path in resolved.values())


def test_transfer_mini_training_arrow_manifest_requires_clusters(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dummy"
    dataset_root.mkdir()
    signatures_path = dataset_root / "signatures.arrow"
    signatures_path.touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"signatures": signatures_path.name}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="no clusters ground truth"):
        transfer_mini_cmd._resolve_arrow_dataset_paths(str(tmp_path), "dummy", require_clusters=True)
