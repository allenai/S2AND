from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH

_SCRIPT_PATHS = [
    "scripts/_rust_suite/calibrate_phase_a_cmd.py",
    "scripts/_rust_suite/calibrate_rust_batch_cmd.py",
    "scripts/_rust_suite/compare_cmd.py",
    "scripts/_rust_suite/featurizer_reuse_cmd.py",
    "scripts/_rust_suite/largest_block_cmd.py",
    "scripts/_rust_suite/prod_inference_cmd.py",
    "scripts/_rust_suite/stress_rebuild_cmd.py",
    "scripts/_rust_suite/transfer_mini_cmd.py",
]


@pytest.mark.parametrize("relative_script_path", _SCRIPT_PATHS)
def test_rust_suite_subprocess_scripts_bootstrap_import_path(relative_script_path: str) -> None:
    repo_root = Path(PROJECT_ROOT_PATH)
    script_path = repo_root / relative_script_path

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    output = f"{completed.stdout}\n{completed.stderr}".lower()
    assert completed.returncode == 0, output
    assert "usage" in output
