#!/usr/bin/env python3
"""
Run CI steps locally using the ACTIVE virtual environment.

Order (matches your CI):
  1) uv sync --extra dev [--frozen if uv.lock exists]  (ACTIVE venv)
  2) ruff lint + format checks via uvx --from ruff==0.6.9 ...
  3) ty checks with tuned migration rules via uvx --from ty==0.0.18 ...
  4) pytest tests/ with coverage and PYTHONPATH=.

Key fix: resolve repo root (pyproject.toml) and run all commands from there.
"""

import os
import shutil
import subprocess
import sys
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
TY_VERSION = "0.0.18"
TY_BASE_IGNORES = [
    "unresolved-import",
    "unused-type-ignore-comment",
    "possibly-missing-attribute",
    "unresolved-global",
]
TY_SCRIPT_EXTRA_IGNORES = [
    "unresolved-reference",
    "unresolved-attribute",
]


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def ensure_rust_on_path() -> None:
    if shutil.which("cargo") or shutil.which("rustc"):
        return
    candidates: list[Path] = []
    if os.name == "nt":
        home = os.environ.get("USERPROFILE")
        if home:
            candidates.append(Path(home) / ".cargo" / "bin")
    else:
        home = os.environ.get("HOME")
        if home:
            candidates.append(Path(home) / ".cargo" / "bin")
    for candidate in candidates:
        if candidate.is_dir():
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"
            if shutil.which("cargo") or shutil.which("rustc"):
                return


def run_ruff_format_check_on(paths: list[str]) -> None:
    uvx = uvx_exe()
    if uvx:
        run(uvx + ["--from", "ruff==0.6.9", "ruff", "format", "--check", *paths])
        return
    # Fallbacks if uvx missing
    try:
        run(uv_exe() + ["run", "--active", "ruff", "format", "--check", *paths])
        return
    except subprocess.CalledProcessError:
        pass
    run([sys.executable, "-m", "ruff", "format", "--check", *paths])


def run_ruff_check_on(paths: list[str]) -> None:
    uvx = uvx_exe()
    if uvx:
        run(uvx + ["--from", "ruff==0.6.9", "ruff", "check", *paths])
        return
    try:
        run(uv_exe() + ["run", "--active", "ruff", "check", *paths])
        return
    except subprocess.CalledProcessError:
        pass
    run([sys.executable, "-m", "ruff", "check", *paths])


def run_ty_check_on(paths: list[str], *, script_mode: bool = False) -> None:
    ignore_rules = list(TY_BASE_IGNORES)
    if script_mode:
        ignore_rules.extend(TY_SCRIPT_EXTRA_IGNORES)
    ignore_args: list[str] = []
    for rule in ignore_rules:
        ignore_args.extend(["--ignore", rule])

    uvx = uvx_exe()
    if uvx:
        run(uvx + ["--from", f"ty=={TY_VERSION}", "ty", "check", *paths, *ignore_args])
        return
    try:
        run(uv_exe() + ["run", "--active", "--no-project", "ty", "check", *paths, *ignore_args])
        return
    except subprocess.CalledProcessError:
        pass
    run([sys.executable, "-m", "ty", "check", *paths, *ignore_args])


def main() -> None:
    # 1) Sync deps into ACTIVE venv
    lock_present = (REPO / "uv.lock").exists()
    sync_args = ["sync", "--active", "--extra", "dev"]
    if lock_present:
        sync_args.append("--frozen")
    run(uv_exe() + sync_args)

    # 1.5) Build Rust extension (required for parity tests)
    ensure_rust_on_path()
    run(
        uv_exe()
        + ["run", "--active", "--no-project", "--with", "maturin", "maturin", "develop", "-m", "s2and_rust/Cargo.toml"]
    )

    # 2) Ruff checks (same targets/flags as CI)
    run_ruff_check_on(["s2and", "scripts", "tests"])
    run_ruff_format_check_on(["s2and"])
    script_files = sorted((REPO / "scripts").glob("*.py"))
    if script_files:
        run_ruff_format_check_on([str(p.relative_to(REPO)) for p in script_files])

    # 3) ty — run type checking commands directly
    run_ty_check_on(["s2and"])
    script_files = sorted((REPO / "scripts").glob("*.py"))
    if script_files:
        script_paths = [str(p.relative_to(REPO)) for p in script_files]
        run_ty_check_on(script_paths, script_mode=True)

    # 4) pytest — coverage flags, PYTHONPATH=.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    run(
        uv_exe()
        + [
            "run",
            "--active",
            "--no-project",
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
