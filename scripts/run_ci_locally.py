#!/usr/bin/env python3
"""
Run CI steps locally using the ACTIVE virtual environment.

Order (matches your CI):
  1) uv sync --all-extras --dev [--frozen if uv.lock exists]  (ACTIVE venv)
  2) black checks via uvx --from black==24.8.0 ...
  3) mypy via scripts/mypy.sh when bash is available; otherwise `uv run mypy`
  4) pytest tests/ with coverage and PYTHONPATH=.

Key fix: resolve repo root (pyproject.toml) and run all commands from there.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def uv_exe() -> list[str]:
    uv_path = which("uv")
    if uv_path:
        return [uv_path]
    try:
        import uv  # noqa: F401
    except Exception:
        print("ERROR: 'uv' not found. Install uv first.", file=sys.stderr)
        sys.exit(2)
    return [sys.executable, "-m", "uv"]


def uvx_exe() -> list[str] | None:
    uvx_path = which("uvx")
    if uvx_path:
        return [uvx_path]
    try:
        import uvx  # noqa: F401

        return [sys.executable, "-m", "uvx"]
    except Exception:
        return None


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in [here] + list(here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here  # fallback


REPO = repo_root()


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def run_black_on(paths: list[str]) -> None:
    uvx = uvx_exe()
    if uvx:
        run(uvx + ["--from", "black==24.8.0", "black", *paths, "--check", "--line-length", "120"])
        return
    # Fallbacks if uvx missing
    try:
        run(uv_exe() + ["run", "--active", "black", *paths, "--check", "--line-length", "120"])
        return
    except subprocess.CalledProcessError:
        pass
    run([sys.executable, "-m", "black", *paths, "--check", "--line-length", "120"])


def main() -> None:
    # 1) Sync deps into ACTIVE venv
    lock_present = (REPO / "uv.lock").exists()
    sync_args = ["sync", "--active", "--all-extras", "--dev"]
    if lock_present:
        sync_args.append("--frozen")
    run(uv_exe() + sync_args)

    # 2) Black checks (same targets/flags as CI)
    run_black_on(["s2and"])
    script_files = sorted((REPO / "scripts").glob("*.py"))
    if script_files:
        run_black_on([str(p.relative_to(REPO)) for p in script_files])

    # 3) mypy — run type checking commands directly
    run(uv_exe() + ["run", "--active", "mypy", "s2and", "--ignore-missing-imports"])
    script_files = sorted((REPO / "scripts").glob("*.py"))
    if script_files:
        script_paths = [str(p.relative_to(REPO)) for p in script_files]
        run(uv_exe() + ["run", "--active", "mypy"] + script_paths + ["--ignore-missing-imports"])

    # 4) pytest — coverage flags, PYTHONPATH=.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    run(
        uv_exe()
        + [
            "run",
            "--active",
            "pytest",
            "tests/",
            "--cov=s2and",
            "--cov-report=term-missing",
            "--cov-fail-under=40",
        ],
        env=env,
    )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
