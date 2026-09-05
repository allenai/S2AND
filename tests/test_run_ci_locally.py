from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_ci_builds_native_runtime_and_requires_smoke_before_pytest(monkeypatch, smoke_fails) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    lifecycle: list[str] = []

    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present: None)
    monkeypatch.setattr(run_ci, "ensure_rust_on_path", lambda: lifecycle.append("ensure-rust"))
    monkeypatch.setattr(run_ci, "run_native_rust_checks", lambda: lifecycle.append("check-rust"))
    monkeypatch.setattr(run_ci, "run_maturin_develop_with_retries", lambda: lifecycle.append("build-rust"))
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: lifecycle.append("typecheck"))
    failure = subprocess.CalledProcessError(7, "native smoke")

    def run_uv(args, *, env=None):
        calls.append((args, env))
        if smoke_fails:
            raise failure

    monkeypatch.setattr(run_ci, "run_uv", run_uv)
    if smoke_fails:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_ci.run_typecheck_and_test_job(lock_present=True)
        assert caught.value is failure
        assert lifecycle == ["ensure-rust", "check-rust", "build-rust"]
        assert len(calls) == 1
        return
    run_ci.run_typecheck_and_test_job(lock_present=True)

    pytest_calls = [args for args, _env in calls if args[:3] == ["run", "--no-sync", "pytest"]]
    assert len(pytest_calls) == 1
    assert lifecycle == ["ensure-rust", "check-rust", "build-rust", "typecheck"]
    assert calls[0][0] == ["run", "--no-sync", "python", "scripts/verification/smoke_installed_rust_api.py"]
    assert calls[0][1] is not None
    assert "S2AND_BACKEND" not in calls[0][1]
    assert all("-ra" in args for args in pytest_calls)
    assert pytest_calls[0] == [
        "run",
        "--no-sync",
        "pytest",
        "-ra",
        "tests/",
        "--cov=s2and",
        "--cov-report=term-missing",
    ]
    assert calls[1][1] is not None
    assert calls[1][1]["S2AND_BACKEND"] == "python"


@pytest.mark.parametrize(
    ("platform", "failures", "expected_attempts", "expected_delays"),
    [("nt", 1, 2, [2.0]), ("nt", 3, 3, [2.0, 4.0]), ("posix", 1, 1, [])],
    ids=["retry-recovers", "retry-exhausted", "non-windows-no-retry"],
)
def test_maturin_retries_are_bounded_and_surface_final_failure(
    monkeypatch, tmp_path: Path, capsys, platform, failures, expected_attempts, expected_delays
) -> None:
    artifact = tmp_path / "stale.pyd"
    artifact.write_bytes(b"stale extension")
    attempts = []
    delays = []
    failure = subprocess.CalledProcessError(23, "maturin develop")

    def build(_args):
        attempts.append(1)
        if len(attempts) <= failures:
            raise failure
        assert not artifact.exists(), "Retry must clear the stale extension first"

    monkeypatch.setattr(run_ci, "os", SimpleNamespace(name=platform))
    monkeypatch.setattr(run_ci, "run_uv", build)
    monkeypatch.setattr(run_ci, "_rust_extension_artifacts", lambda: [artifact] if artifact.exists() else [])
    monkeypatch.setattr(run_ci.time, "sleep", delays.append)

    if failures >= expected_attempts:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_ci.run_maturin_develop_with_retries()
        assert caught.value is failure
    else:
        run_ci.run_maturin_develop_with_retries()

    assert len(attempts) == expected_attempts
    assert delays == expected_delays
    output = capsys.readouterr()
    assert f"attempt {expected_attempts}/" in output.out
    if expected_delays:
        assert "exit code 23" in output.err
