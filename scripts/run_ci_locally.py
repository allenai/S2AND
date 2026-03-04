#!/usr/bin/env python3
"""
Run local CI with close parity to `.github/workflows/main.yaml`.

Execution order:
  1) lint job:
     - uv sync --extra dev [--frozen if uv.lock exists]
     - ruff check / format checks
  2) typecheck-and-test matrix:
     - py-only lane
     - rust-enabled lane
"""

import os
import shutil
import subprocess
import sys
import time
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


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for directory in [here] + list(here.parents):
        if (directory / "pyproject.toml").exists():
            return directory
    return here


REPO = repo_root()
LANES = ["py-only", "rust-enabled"]
RUST_PARITY_TESTS = [
    "tests/test_feature_port_parity.py",
    "tests/test_rust_signature_preprocess.py",
    "tests/test_rust_batch_chunking.py",
    "tests/test_rust_from_json_paths.py",
]
TY_PYTHON_VERSION = "3.11"
TY_PYTHON_PLATFORM = os.environ.get("S2AND_CI_TY_PLATFORM", "linux")
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
MATURIN_RETRY_ATTEMPTS_WINDOWS = 3
MATURIN_RETRY_BACKOFF_SECONDS = 2.0


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def run_uv(args: list[str], *, env: dict[str, str] | None = None) -> None:
    run(uv_exe() + args, env=env)


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


def top_level_script_files() -> list[str]:
    return [str(path.relative_to(REPO)) for path in sorted((REPO / "scripts").glob("*.py"))]


def _rust_extension_artifacts() -> list[Path]:
    rust_package_dir = REPO / "s2and_rust" / "s2and_rust"
    if not rust_package_dir.is_dir():
        return []
    return sorted(rust_package_dir.glob("_s2and_rust*.pyd"))


def run_maturin_develop_with_retries() -> None:
    args = ["run", "--with", "maturin", "maturin", "develop", "-m", "s2and_rust/Cargo.toml"]
    attempts = MATURIN_RETRY_ATTEMPTS_WINDOWS if os.name == "nt" else 1
    for attempt in range(1, attempts + 1):
        try:
            print(f"[maturin] attempt {attempt}/{attempts}")
            run_uv(args)
            return
        except subprocess.CalledProcessError as exc:
            if attempt >= attempts:
                raise
            for artifact in _rust_extension_artifacts():
                try:
                    artifact.unlink()
                    print(f"[maturin] removed stale artifact before retry: {artifact}")
                except OSError as cleanup_exc:
                    print(
                        f"[maturin] cleanup warning (attempt {attempt}): could not remove {artifact}: {cleanup_exc}",
                        file=sys.stderr,
                    )
            sleep_seconds = MATURIN_RETRY_BACKOFF_SECONDS * float(attempt)
            print(
                f"[maturin] attempt {attempt} failed with exit code {exc.returncode}; retrying in {sleep_seconds:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_seconds)


def sync_deps(*, lock_present: bool, lane: str) -> None:
    args = ["sync", "--extra", "dev"]
    if lane == "rust-enabled":
        args.extend(["--extra", "rust"])
    if lock_present:
        args.append("--frozen")
    run_uv(args)


def run_lint_job(*, lock_present: bool) -> None:
    print("\n=== lint ===")
    sync_deps(lock_present=lock_present, lane="py-only")
    run_uv(["run", "ruff", "check", "s2and", "scripts", "tests"])
    run_uv(["run", "ruff", "format", "--check", "s2and"])
    script_files = top_level_script_files()
    if script_files:
        run_uv(["run", "ruff", "format", "--check", *script_files])


def run_ty_checks() -> None:
    ignore_args: list[str] = []
    for rule in TY_BASE_IGNORES:
        ignore_args.extend(["--ignore", rule])

    run_uv(
        [
            "run",
            "ty",
            "check",
            "s2and",
            *ignore_args,
            "--python-version",
            TY_PYTHON_VERSION,
            "--python-platform",
            TY_PYTHON_PLATFORM,
        ]
    )

    script_files = top_level_script_files()
    if script_files:
        script_ignore_args = list(ignore_args)
        for rule in TY_SCRIPT_EXTRA_IGNORES:
            script_ignore_args.extend(["--ignore", rule])
        run_uv(
            [
                "run",
                "ty",
                "check",
                *script_files,
                *script_ignore_args,
                "--python-version",
                TY_PYTHON_VERSION,
                "--python-platform",
                TY_PYTHON_PLATFORM,
            ]
        )


def run_typecheck_and_test_lane(*, lane: str, lock_present: bool) -> None:
    print(f"\n=== typecheck-and-test ({lane}) ===")
    sync_deps(lock_present=lock_present, lane=lane)

    if lane == "rust-enabled":
        ensure_rust_on_path()
        run_maturin_develop_with_retries()
        for parity_test in RUST_PARITY_TESTS:
            run_uv(["run", "pytest", "-q", parity_test])

    run_ty_checks()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    if lane == "py-only":
        env["S2AND_BACKEND"] = "python"

    run_uv(
        [
            "run",
            "pytest",
            "tests/",
            "--cov=s2and",
            "--cov-report=term-missing",
            "--cov-fail-under=40",
        ],
        env=env,
    )


def main() -> None:
    lock_present = (REPO / "uv.lock").exists()
    run_lint_job(lock_present=lock_present)
    for lane in LANES:
        run_typecheck_and_test_lane(lane=lane, lock_present=lock_present)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
