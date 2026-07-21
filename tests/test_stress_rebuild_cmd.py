from __future__ import annotations

import sys

import pytest

from scripts._rust_suite import stress_rebuild_cmd


def test_parse_args_exposes_only_arrow_rebuild_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stress_rebuild_cmd.py", "--dataset", "qian"])

    args = stress_rebuild_cmd._parse_args()

    assert not hasattr(args, "build_path")
    assert args.arrow_data_root == stress_rebuild_cmd.DEFAULT_ARROW_DATA_ROOT
    assert args.specter_suffix == stress_rebuild_cmd.DEFAULT_ARROW_SPECTER_SUFFIX


def test_parse_args_rejects_removed_build_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["stress_rebuild_cmd.py", "--dataset", "qian", "--build-path", "from_arrow_paths"],
    )

    with pytest.raises(SystemExit):
        stress_rebuild_cmd._parse_args()
