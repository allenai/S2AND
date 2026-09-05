"""Run small shell contracts with native Bash, including Git Bash on Windows."""

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest


@cache
def bash_executable() -> str:
    """Find native Bash without accidentally selecting Windows' WSL launcher."""
    if os.name == "nt":
        git = shutil.which("git")
        candidate = Path(git).resolve().parent.parent / "bin" / "bash.exe" if git else None
        if candidate is not None and candidate.is_file():
            return str(candidate)
        pytest.skip("Git Bash is required for shell contract tests on Windows")
    executable = shutil.which("bash")
    if executable is None:
        pytest.skip("Bash is required for shell contract tests")
    return executable


def run_bash(
    script: str, *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a bounded script without startup files; retain output on failures."""
    return subprocess.run(
        [bash_executable(), "--noprofile", "--norc", "-s"],
        input=script,
        text=True,
        capture_output=True,
        timeout=15,
        cwd=cwd,
        env=os.environ | (env or {}),
    )
