from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts._rust_suite import transfer_mini_cmd


def test_normalize_hyperopt_trial_vals_converts_only_integral_floats() -> None:
    assert transfer_mini_cmd._normalize_hyperopt_trial_vals(
        {"choice": [2.0, np.float32(5), 2.5, 3, "4", True], "ignored": "not-a-list"}
    ) == {"choice": [2, 5, 2.5, 3, "4", True]}


def test_transfer_mini_resolve_ingest_auto_is_backend_specific() -> None:
    assert transfer_mini_cmd._resolve_ingest("auto", "python") == "json"
    assert transfer_mini_cmd._resolve_ingest("auto", "rust") == "arrow"

    with pytest.raises(ValueError, match="Python-only"):
        transfer_mini_cmd._resolve_ingest("json", "rust")


def test_transfer_mini_compare_rejects_explicit_ingest() -> None:
    with pytest.raises(ValueError, match="--mode compare requires --ingest auto"):
        transfer_mini_cmd._resolve_workload(SimpleNamespace(mode="compare", ingest="arrow"))


def test_transfer_mini_rejects_borrowed_prediction_arrow_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["transfer_mini_cmd.py", "--prediction-arrow-data-dir", "converted"],
    )

    with pytest.raises(SystemExit) as exc_info:
        transfer_mini_cmd.main()

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert "--prediction-arrow-data-dir" in stderr


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


def test_transfer_mini_arrow_manifest_accepts_absolute_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dummy"
    dataset_root.mkdir()
    signatures_path = tmp_path / "shared" / "signatures.arrow"
    signatures_path.parent.mkdir()
    signatures_path.touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"signatures": str(signatures_path.resolve())}}),
        encoding="utf-8",
    )

    resolved, _ = transfer_mini_cmd._resolve_arrow_dataset_paths(str(tmp_path), "dummy", require_clusters=False)

    assert resolved == {"signatures": str(signatures_path.resolve())}


@pytest.mark.parametrize("stale_path", ["old-root/signatures.arrow", "dummy/signatures.arrow"])
def test_transfer_mini_arrow_manifest_rejects_legacy_path_fallbacks(tmp_path: Path, stale_path: str) -> None:
    dataset_root = tmp_path / "dummy"
    dataset_root.mkdir()
    (dataset_root / "signatures.arrow").touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"signatures": stale_path}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Cannot resolve arrow manifest path signatures"):
        transfer_mini_cmd._resolve_arrow_dataset_paths(str(tmp_path), "dummy", require_clusters=False)
