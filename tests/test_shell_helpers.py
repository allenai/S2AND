"""Git Bash discovery for the supported Windows Git layouts."""

import os
from types import SimpleNamespace

import pytest

from tests import shell_helpers


@pytest.mark.parametrize("git_directory", ["cmd", "bin", "mingw64/bin"])
def test_windows_git_bash_discovery(tmp_path, monkeypatch, git_directory):
    git = tmp_path / "Git" / git_directory / "git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    bash = tmp_path / "Git" / "bin" / "bash.exe"
    bash.parent.mkdir(exist_ok=True)
    bash.touch()
    monkeypatch.setattr(shell_helpers, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(shell_helpers.shutil, "which", lambda name: str(git) if name == "git" else None)
    shell_helpers.bash_executable.cache_clear()
    try:
        assert shell_helpers.bash_executable() == str(bash.resolve())
    finally:
        shell_helpers.bash_executable.cache_clear()
