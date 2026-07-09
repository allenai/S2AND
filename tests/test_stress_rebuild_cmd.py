from __future__ import annotations

import sys

import pytest

from scripts._rust_suite import stress_rebuild_cmd


def test_parse_args_defaults_to_arrow_build_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stress_rebuild_cmd.py", "--dataset", "qian"])

    args = stress_rebuild_cmd._parse_args()

    assert args.build_path == "from_arrow_paths"
    assert args.arrow_data_root == stress_rebuild_cmd.DEFAULT_ARROW_DATA_ROOT
    assert args.specter_suffix == stress_rebuild_cmd.DEFAULT_ARROW_SPECTER_SUFFIX
