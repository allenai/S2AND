from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


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

    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present: None)
    monkeypatch.setattr(run_ci, "top_level_script_files", lambda: ["scripts/run_ci_locally.py"])
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append(args))

    run_ci.run_lint_job(lock_present=True)

    assert calls[0] == ["run", "--no-sync", "python", "scripts/sync_version.py", "--check"]
    assert ["run", "--no-sync", "ruff", "check", "s2and", "scripts", "tests"] in calls


def test_typecheck_and_test_job_builds_required_rust_runtime(monkeypatch) -> None:
    run_ci = _load_run_ci_locally()
    calls: list[tuple[list[str], dict[str, str] | None]] = []
    lifecycle: list[str] = []

    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(run_ci, "sync_deps", lambda *, lock_present: None)
    monkeypatch.setattr(run_ci, "ensure_rust_on_path", lambda: lifecycle.append("ensure-rust"))
    monkeypatch.setattr(run_ci, "run_maturin_develop_with_retries", lambda: lifecycle.append("build-rust"))
    monkeypatch.setattr(run_ci, "run_ty_checks", lambda: lifecycle.append("typecheck"))
    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append((args, env)))

    run_ci.run_typecheck_and_test_job(lock_present=True)

    pytest_calls = [args for args, _env in calls if args[:3] == ["run", "--no-sync", "pytest"]]
    assert len(pytest_calls) == len(run_ci.RUST_PARITY_TESTS) + 1
    assert lifecycle == ["ensure-rust", "build-rust", "typecheck"]
    assert calls[0][0] == ["run", "--no-sync", "python", "scripts/verification/smoke_installed_rust_api.py"]
    assert calls[0][1] is not None
    assert calls[0][1]["S2AND_TEST_REQUIRE_RUST"] == "1"
    assert "S2AND_BACKEND" not in calls[0][1]
    assert all("-ra" in args for args in pytest_calls)
    assert pytest_calls[0] == ["run", "--no-sync", "pytest", "-q", "-ra", run_ci.RUST_PARITY_TESTS[0]]
    for _args, env in calls[1:-1]:
        assert env is not None
        assert env["S2AND_TEST_REQUIRE_RUST"] == "1"
        assert "S2AND_BACKEND" not in env
    assert pytest_calls[-1] == [
        "run",
        "--no-sync",
        "pytest",
        "-ra",
        "tests/",
        "--cov=s2and",
        "--cov-report=term-missing",
        "--cov-fail-under=40",
    ]
    assert calls[-1][1] is not None
    assert calls[-1][1]["S2AND_BACKEND"] == "python"
    assert calls[-1][1]["S2AND_TEST_REQUIRE_RUST"] == "1"


def test_rust_parity_test_paths_exist() -> None:
    run_ci = _load_run_ci_locally()

    for relative_path in run_ci.RUST_PARITY_TESTS:
        assert (run_ci.REPO / relative_path).is_file(), relative_path


def test_sync_deps_uses_only_defined_dev_extra(monkeypatch) -> None:
    run_ci = _load_run_ci_locally()
    calls: list[list[str]] = []

    monkeypatch.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append(args))

    run_ci.sync_deps(lock_present=True)
    run_ci.sync_deps(lock_present=False)

    assert calls == [
        ["sync", "--extra", "dev", "--frozen", "--no-install-package", "s2and-rust"],
        ["sync", "--extra", "dev", "--no-install-package", "s2and-rust"],
    ]
