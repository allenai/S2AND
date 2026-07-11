#!/usr/bin/env python3
"""
Run local CI with close parity to `.github/workflows/main.yaml`.

Execution order:
  1) lint job:
     - uv sync --extra dev [--frozen if uv.lock exists]
     - version sync check
     - ruff check / format checks
  2) typecheck-and-test job:
     - run Rust formatting, Clippy, and native unit tests
     - build the required Rust extension
     - run Rust parity guardrails
     - run the full suite with Python orchestration
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
        import uv  # type: ignore  # noqa: F401
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
RUST_PARITY_TESTS = [
    "tests/test_feature_port_parity.py",
    "tests/test_rust_signature_preprocess.py",
    "tests/test_rust_batch_chunking.py",
]
PYTEST_REPORT_FLAGS = ["-ra"]
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
RUST_MANIFEST = "s2and_rust/Cargo.toml"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO), env=env)


def run_uv(args: list[str], *, env: dict[str, str] | None = None) -> None:
    run(uv_exe() + args, env=env)


def uv_run_args(*args: str) -> list[str]:
    return ["run", "--no-sync", *args]


def pytest_args(*args: str, quiet: bool = False) -> list[str]:
    cmd = uv_run_args("pytest")
    if quiet:
        cmd.append("-q")
    cmd.extend(PYTEST_REPORT_FLAGS)
    cmd.extend(args)
    return cmd


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


def pyo3_python_path() -> str:
    """Return the root uv environment's Python executable for PyO3 builds."""

    relative_path = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    candidate = (REPO / ".venv" / relative_path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"PyO3 requires the root uv environment's Python executable, but it is missing: {candidate}"
        )
    return str(candidate)


def run_native_rust_checks() -> None:
    """Run the native Rust gates enforced by hosted CI."""

    rust_env = os.environ.copy()
    rust_env["PYO3_PYTHON"] = pyo3_python_path()
    run_uv(
        uv_run_args("cargo", "fmt", "--manifest-path", RUST_MANIFEST, "--", "--check"),
        env=rust_env,
    )
    run_uv(
        uv_run_args(
            "cargo",
            "clippy",
            "--manifest-path",
            RUST_MANIFEST,
            "--lib",
            "--no-deps",
            "--",
            "-D",
            "clippy::correctness",
            "-D",
            "clippy::suspicious",
        ),
        env=rust_env,
    )
    run_uv(
        uv_run_args("cargo", "test", "--manifest-path", RUST_MANIFEST, "--lib"),
        env=rust_env,
    )


def top_level_script_files() -> list[str]:
    return [str(path.relative_to(REPO)) for path in sorted((REPO / "scripts").glob("*.py"))]


def _rust_extension_artifacts() -> list[Path]:
    rust_package_dir = REPO / "s2and_rust" / "s2and_rust"
    if not rust_package_dir.is_dir():
        return []
    return sorted(rust_package_dir.glob("_s2and_rust*.pyd"))


def run_maturin_develop_with_retries() -> None:
    args = uv_run_args("--with", "maturin", "maturin", "develop", "-m", "s2and_rust/Cargo.toml")
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


def sync_deps(*, lock_present: bool) -> None:
    args = ["sync", "--extra", "dev"]
    if lock_present:
        args.append("--frozen")
    args.extend(["--no-install-package", "s2and-rust"])
    run_uv(args)


def run_lint_job(*, lock_present: bool) -> None:
    print("\n=== lint ===")
    sync_deps(lock_present=lock_present)
    run_uv(uv_run_args("python", "scripts/sync_version.py", "--check"))
    run_uv(uv_run_args("ruff", "check", "s2and", "scripts", "tests"))
    run_uv(uv_run_args("ruff", "format", "--check", "s2and"))
    script_files = top_level_script_files()
    if script_files:
        run_uv(uv_run_args("ruff", "format", "--check", *script_files))


def run_ty_checks() -> None:
    ignore_args: list[str] = []
    for rule in TY_BASE_IGNORES:
        ignore_args.extend(["--ignore", rule])

    run_uv(
        uv_run_args(
            "ty",
            "check",
            "s2and",
            *ignore_args,
            "--python-version",
            TY_PYTHON_VERSION,
            "--python-platform",
            TY_PYTHON_PLATFORM,
        )
    )

    script_files = top_level_script_files()
    if script_files:
        script_ignore_args = list(ignore_args)
        for rule in TY_SCRIPT_EXTRA_IGNORES:
            script_ignore_args.extend(["--ignore", rule])
        run_uv(
            uv_run_args(
                "ty",
                "check",
                *script_files,
                *script_ignore_args,
                "--python-version",
                TY_PYTHON_VERSION,
                "--python-platform",
                TY_PYTHON_PLATFORM,
            )
        )


def run_typecheck_and_test_job(*, lock_present: bool) -> None:
    print("\n=== typecheck-and-test ===")
    sync_deps(lock_present=lock_present)

    ensure_rust_on_path()
    run_native_rust_checks()
    run_maturin_develop_with_retries()
    required_rust_env = os.environ.copy()
    required_rust_env["S2AND_TEST_REQUIRE_RUST"] = "1"
    required_rust_env.pop("S2AND_BACKEND", None)
    run_uv(
        uv_run_args("python", "scripts/verification/smoke_installed_rust_api.py"),
        env=required_rust_env,
    )
    for parity_test in RUST_PARITY_TESTS:
        run_uv(pytest_args(parity_test, quiet=True), env=required_rust_env)

    run_ty_checks()

    python_backend_env = required_rust_env.copy()
    python_backend_env["S2AND_BACKEND"] = "python"

    run_uv(
        pytest_args(
            "tests/",
            "--cov=s2and",
            "--cov-report=term-missing",
            "--cov-fail-under=40",
        ),
        env=python_backend_env,
    )


def main() -> None:
    lock_present = (REPO / "uv.lock").exists()
    run_lint_job(lock_present=lock_present)
    run_typecheck_and_test_job(lock_present=lock_present)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
