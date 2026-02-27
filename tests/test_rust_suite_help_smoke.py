from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH


@pytest.mark.parametrize(
    "command_name",
    [
        "compare",
        "transfer-mini",
        "prod-inference",
        "featurizer-reuse",
        "largest-block",
        "big-block-incremental",
        "stress-rebuild",
    ],
)
def test_rust_suite_command_help_smoke(command_name: str) -> None:
    script_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    completed = subprocess.run(
        [sys.executable, str(script_path), command_name, "--help"],
        cwd=PROJECT_ROOT_PATH,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Command help failed for {command_name}\n" f"stdout:\n{completed.stdout}\n" f"stderr:\n{completed.stderr}"
    )
    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    assert "usage" in combined_output
