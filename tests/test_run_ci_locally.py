from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_run_ci_locally() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ci_locally.py"
    spec = importlib.util.spec_from_file_location("run_ci_locally", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lint_job_runs_version_sync_check(monkeypatch) -> None:
    run_ci = _load_run_ci_locally()
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
    run_ci = _load_run_ci_locally()
    rustc_path = tmp_path / "rustc"

    monkeypatch.setattr(
        run_ci.shutil,
        "which",
        lambda command: str(rustc_path) if command == "rustc" else None,
    )
    monkeypatch.setenv("USERPROFILE" if run_ci.os.name == "nt" else "HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="cargo is required"):
        run_ci.ensure_rust_on_path()


def test_typecheck_and_test_job_builds_required_rust_runtime(monkeypatch) -> None:
    run_ci = _load_run_ci_locally()
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    lifecycle: list[str] = []

    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present: None)
    monkeypatch.setattr(run_ci, "ensure_rust_on_path", lambda: lifecycle.append("ensure-rust"))
    monkeypatch.setattr(run_ci, "run_native_rust_checks", lambda: lifecycle.append("check-rust"))
    monkeypatch.setattr(run_ci, "run_maturin_develop_with_retries", lambda: lifecycle.append("build-rust"))
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: lifecycle.append("typecheck"))
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append((args, env)))

    run_ci.run_typecheck_and_test_job(lock_present=True)

    pytest_calls = [args for args, _env in calls if args[:3] == ["run", "--no-sync", "pytest"]]
    assert len(pytest_calls) == 1
    assert lifecycle == ["ensure-rust", "check-rust", "build-rust", "typecheck"]
    assert calls[0][0] == ["run", "--no-sync", "python", "scripts/verification/smoke_installed_rust_api.py"]
    assert calls[0][1] is not None
    assert calls[0][1]["S2AND_TEST_REQUIRE_RUST"] == "1"
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
        "--cov-fail-under=40",
    ]
    assert calls[1][1] is not None
    assert calls[1][1]["S2AND_BACKEND"] == "python"
    assert calls[1][1]["S2AND_TEST_REQUIRE_RUST"] == "1"
