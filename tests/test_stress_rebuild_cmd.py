from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts._rust_suite import stress_rebuild_cmd


def test_parse_args_defaults_to_arrow_build_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stress_rebuild_cmd.py", "--dataset", "qian"])

    args = stress_rebuild_cmd._parse_args()

    assert args.build_path == "from_arrow_paths"
    assert args.arrow_data_root == stress_rebuild_cmd.DEFAULT_ARROW_DATA_ROOT
    assert args.specter_suffix == stress_rebuild_cmd.DEFAULT_ARROW_SPECTER_SUFFIX


def test_dataset_paths_resolve_json_datasets_under_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import s2and.consts as consts_module

    dataset_root = tmp_path / "s2and" / "data" / "aminer"
    dataset_root.mkdir(parents=True)
    (dataset_root / "aminer_signatures.json").write_text("{}", encoding="utf-8")
    (dataset_root / "aminer_papers.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(consts_module, "PROJECT_ROOT_PATH", str(tmp_path))

    paths = stress_rebuild_cmd._dataset_paths("aminer")  # noqa: SLF001

    assert paths["signatures"] == str(dataset_root / "aminer_signatures.json")
    assert paths["papers"] == str(dataset_root / "aminer_papers.json")
