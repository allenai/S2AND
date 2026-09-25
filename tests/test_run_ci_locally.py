from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_ci_locally as run_ci


def test_lint_job_runs_version_sync_check(monkeypatch) -> None:
    calls: list[list[str]] = []

    monkeypatch.setattr(
        run_ci,
        "sync_deps",
        lambda *, lock_present: (_ for _ in ()).throw(AssertionError("lint must not sync project dependencies")),
    )
    monkeypatch.setattr(run_ci, "top_level_script_files", lambda: ["scripts/run_ci_locally.py"])
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append(args))

    run_ci.run_lint_job()

    ruff_requirement = run_ci.exact_dev_tool_requirement("ruff")
    assert calls == [
        ["run", "--no-project", "python", "scripts/sync_version.py", "--check"],
        ["run", "--no-project", "python", "scripts/sync_feature_schema.py", "--check"],
        ["tool", "run", "--isolated", ruff_requirement, "check", "s2and", "scripts", "tests"],
        ["tool", "run", "--isolated", ruff_requirement, "format", "--check", "s2and"],
        [
            "tool",
            "run",
            "--isolated",
            ruff_requirement,
            "format",
            "--check",
            "scripts/run_ci_locally.py",
        ],
    ]


def test_ensure_rust_on_path_rejects_rustc_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rustc_path = tmp_path / "rustc"

    monkeypatch.setattr(
        run_ci.shutil,
        "which",
        lambda command: str(rustc_path) if command == "rustc" else None,
    )
    monkeypatch.setenv("USERPROFILE" if run_ci.os.name == "nt" else "HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="cargo is required"):
        run_ci.ensure_rust_on_path()


@pytest.mark.parametrize("workers", ["auto", "12", "0"])
@pytest.mark.parametrize("smoke_fails", [False, True])
def test_ci_builds_native_runtime_and_requires_smoke_before_pytest(monkeypatch, smoke_fails, workers) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    lifecycle: list[str] = []

    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present: lifecycle.append("sync-and-build-rust"))
    monkeypatch.setattr(run_ci, "ensure_rust_on_path", lambda: lifecycle.append("ensure-rust"))
    monkeypatch.setattr(run_ci, "run_native_rust_checks", lambda: lifecycle.append("check-rust"))
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: lifecycle.append("typecheck"))
    failure = subprocess.CalledProcessError(7, "native smoke")

    def run_uv(args, *, env=None):
        calls.append((args, env))
        if smoke_fails:
            raise failure

    monkeypatch.setattr(run_ci, "run_uv", run_uv)
    if smoke_fails:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_ci.run_typecheck_and_test_job(lock_present=True, pytest_workers=workers)
        assert caught.value is failure
        assert lifecycle == ["ensure-rust", "sync-and-build-rust", "check-rust"]
        assert len(calls) == 1
        return
    run_ci.run_typecheck_and_test_job(lock_present=True, pytest_workers=workers)

    pytest_calls = [args for args, _env in calls if args[:3] == ["run", "--no-sync", "pytest"]]
    assert len(pytest_calls) == 1
    assert lifecycle == ["ensure-rust", "sync-and-build-rust", "check-rust", "typecheck"]
    assert calls[0][0] == ["run", "--no-sync", "python", "scripts/verification/smoke_installed_rust_api.py"]
    assert calls[0][1] is None
    assert all("-ra" in args for args in pytest_calls)
    assert pytest_calls[0] == [
        "run",
        "--no-sync",
        "pytest",
        "-ra",
        "tests/",
        "-n",
        workers,
        *(["--maxprocesses", "8"] if workers == "auto" else []),
        "--cov=s2and",
        "--cov-report=term-missing",
    ]
    assert calls[1][1] is None


@pytest.mark.parametrize(("lock_present", "frozen"), [(True, ["--frozen"]), (False, [])])
def test_sync_deps_installs_the_rust_extension(monkeypatch, lock_present: bool, frozen: list[str]) -> None:
    """The typecheck job imports the extension that the locked sync builds."""
    calls: list[list[str]] = []
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append(args))

    run_ci.sync_deps(lock_present=lock_present)

    assert calls == [["sync", "--extra", "dev", *frozen]]
