#!/usr/bin/env python3
"""
Cross-platform CI runner that uses `uv`.

Usage:
  python run_ci_locally.py

Requirements:
  - `uv` must be installed and on PATH (or importable as a module).
  - Dev dependencies should contain black, pytest, pytest-xdist (optional), coverage.
"""
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def which_uv():
    p = shutil.which("uv")
    if p:
        return ["uv"]
    # fallback to `python -m uv` if uv is installed as a package
    try:
        import uv  # type: ignore

        return [sys.executable, "-m", "uv"]
    except Exception:
        return None


def run(cmd, **kwargs):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def uv_cmd(args):
    uv = which_uv()
    if not uv:
        print("ERROR: 'uv' not found. Install it with: pip install --upgrade uv", file=sys.stderr)
        sys.exit(2)
    return uv + args


def main():
    # Ensure we run from repo root
    try:
        (ROOT / "pyproject.toml").exists() or (ROOT / "setup.py").exists()
    except Exception:
        pass

    # Sync deps (use frozen if lock exists)
    lock = ROOT / "uv.lock"
    sync_args = ["sync", "--dev"]
    if lock.exists():
        sync_args.append("--frozen")
    run(uv_cmd(sync_args))

    # -------------------------
    # Black checks (via uvx so it uses the cached resolver)
    # -------------------------
    # s2and
    run(uv_cmd(["x", "--from", "black==24.8.0", "black", "s2and", "--check", "--line-length", "120"]))

    # scripts/*.py (guard against no-match)
    script_files = list((ROOT / "scripts").glob("*.py"))
    if script_files:
        # pass as individual args (uvx forwards them)
        run(
            uv_cmd(
                ["x", "--from", "black==24.8.0", "black"]
                + [str(p) for p in script_files]
                + ["--check", "--line-length", "120"]
            )
        )

    # -------------------------
    # Tests: use xdist if available (parallel), otherwise single-run.
    # -------------------------
    # Check whether pytest-xdist is installed in the uv environment by trying to import it via `uv run python -c`
    try:
        run(
            uv_cmd(
                [
                    "run",
                    "python",
                    "-c",
                    "import importlib; import sys; sys.exit(0 if importlib.util.find_spec('xdist') else 1)",
                ]
            )
        )
        has_xdist = True
    except subprocess.CalledProcessError:
        has_xdist = False

    pytest_base = ["run", "pytest", "tests/", "--cov=s2and", "--cov-report=term-missing", "--cov-fail-under=40"]
    if has_xdist:
        pytest_base.insert(2, "-n")
        pytest_base.insert(3, "auto")  # places `-n auto` into args: ["run", "pytest", "-n", "auto", "tests/", ...]
    run(uv_cmd(pytest_base))

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
