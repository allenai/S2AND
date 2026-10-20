#!/usr/bin/env python3
"""
Run local CI with close parity to `.github/workflows/main.yaml`.

With no argument, execution order is:
  1) lint job:
     - version sync check
     - generated feature-schema sync check
     - isolated, exactly pinned ruff check / format checks
  2) typecheck-and-test job:
     - sync dependencies, which builds and installs the required Rust extension
     - run Rust formatting, Clippy, and native unit tests
     - smoke-test the installed Rust extension
     - type-check s2and and the active scripts with ty
     - run the full suite with Python orchestration

Pass ``lint`` or ``typecheck-and-test`` to run only one hosted-CI job.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
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
PYTEST_REPORT_FLAGS = ["-ra"]
# ty's Python version and rule ignores come from [tool.ty] in pyproject.toml.
TY_PYTHON_PLATFORM = os.environ.get("S2AND_CI_TY_PLATFORM", "linux")
# Type-check the tracked nested script folders together with the top-level scripts.
ACTIVE_NESTED_SCRIPT_DIRS = ["scripts/eps_sweep", "scripts/production", "scripts/verification"]
TY_SCRIPT_EXTRA_IGNORES = [
    "unresolved-reference",
    "unresolved-attribute",
]
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
    if shutil.which("cargo"):
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
            if shutil.which("cargo"):
                return
    raise FileNotFoundError("cargo is required for native CI checks but was not found on PATH")


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
        uv_run_args(
            "cargo",
            "test",
            "--manifest-path",
            RUST_MANIFEST,
            "--lib",
            "--no-default-features",
        ),
        env=rust_env,
    )


def top_level_script_files() -> list[str]:
    return [str(path.relative_to(REPO)) for path in sorted((REPO / "scripts").glob("*.py"))]


def sync_deps(*, lock_present: bool) -> None:
    args = ["sync", "--extra", "dev"]
    if lock_present:
        args.append("--frozen")
    run_uv(args)


def exact_dev_tool_requirement(tool_name: str) -> str:
    """Return one exactly pinned tool requirement from the dev extra."""

    with (REPO / "pyproject.toml").open("rb") as source:
        pyproject = tomllib.load(source)
    dev_requirements = pyproject["project"]["optional-dependencies"]["dev"]
    prefix = f"{tool_name}=="
    matches = [str(requirement) for requirement in dev_requirements if str(requirement).startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected one exact {tool_name!r} pin in project.optional-dependencies.dev, found {matches}")
    return matches[0]


def isolated_tool_args(requirement: str, *args: str) -> list[str]:
    """Build a uv command for one isolated, exactly pinned development tool."""

    return ["tool", "run", "--isolated", requirement, *args]


def run_lint_job() -> None:
    print("\n=== lint ===")
    ruff_requirement = exact_dev_tool_requirement("ruff")
    run_uv(["run", "--no-project", "python", "scripts/sync_version.py", "--check"])
    run_uv(["run", "--no-project", "python", "scripts/sync_feature_schema.py", "--check"])
    run_uv(isolated_tool_args(ruff_requirement, "check", "s2and", "scripts", "tests"))
    run_uv(isolated_tool_args(ruff_requirement, "format", "--check", "s2and", "scripts", "tests"))


def run_ty_checks() -> None:
    platform_args = ["--python-platform", TY_PYTHON_PLATFORM]
    run_uv(uv_run_args("ty", "check", "s2and", *platform_args))

    script_files = [*top_level_script_files(), *ACTIVE_NESTED_SCRIPT_DIRS]
    if script_files:
        script_ignore_args: list[str] = []
        for rule in TY_SCRIPT_EXTRA_IGNORES:
            script_ignore_args.extend(["--ignore", rule])
        run_uv(uv_run_args("ty", "check", *script_files, *script_ignore_args, *platform_args))


def run_typecheck_and_test_job(*, lock_present: bool, pytest_workers: str) -> None:
    print("\n=== typecheck-and-test ===")
    # The locked environment builds s2and-rust from source, so cargo comes first.
    ensure_rust_on_path()
    sync_deps(lock_present=lock_present)
    run_native_rust_checks()
    run_uv(uv_run_args("python", "scripts/verification/smoke_installed_rust_api.py"))
    run_ty_checks()

    run_uv(
        pytest_args(
            "tests/",
            "-n",
            pytest_workers,
            *(["--maxprocesses", "8"] if pytest_workers in {"auto", "logical"} else []),
            "--cov=s2and",
            "--cov-report=term-missing",
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "job",
        nargs="?",
        choices=("lint", "typecheck-and-test"),
        help="run one CI job instead of the complete local mirror",
    )
    parser.add_argument(
        "--pytest-workers",
        default="auto",
        help="pytest-xdist workers ('auto' by default, capped at 8; explicit counts override the cap; '0' is serial)",
    )
    args = parser.parse_args(argv)
    lock_present = (REPO / "uv.lock").exists()
    if args.job in (None, "lint"):
        run_lint_job()
    if args.job in (None, "typecheck-and-test"):
        run_typecheck_and_test_job(lock_present=lock_present, pytest_workers=str(args.pytest_workers))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand failed with exit code {exc.returncode}", file=sys.stderr)
        sys.exit(exc.returncode)
