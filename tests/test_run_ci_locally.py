"""CI policy, version/schema sync, release workflow gates, and distribution archive verifiers."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.production_bundle import _write_json
from scripts.dev import run_ci_locally as run_ci
from scripts.dev import sync_feature_schema, sync_version
from scripts.eps.sweep_curves import write_json
from scripts.release.verification.verify_production_model_distributions import verify_production_model_distributions
from scripts.release.verification.verify_rust_distribution_licenses import LICENSE_PATHS, verify_archive
from tests import shell_helpers
from tests.shell_helpers import run_bash

pytestmark = pytest.mark.tooling

PACKAGE = "s2and"
TUPLE_DATA_PATH = "s2and/data/s2and_name_tuples_canonical.txt"
ORCID_DATA_PATH = "s2and/data/first_k_letter_counts_from_orcid.json"
ORCID_MANIFEST_PATH = "s2and/data/first_k_letter_counts_from_orcid.manifest.json"
ORCID_V1_4_DATA_PATH = "s2and/data/orcid_prefix_counts_v1_4/first_k_letter_counts_from_orcid.json"
ORCID_V1_4_MANIFEST_PATH = "s2and/data/orcid_prefix_counts_v1_4/first_k_letter_counts_from_orcid.manifest.json"
DEFAULT_MODEL_PATH = "s2and/data/default_production_model.json"
MODEL_MEMBER_PATH = "s2and/data/production_model_v8.8/manifest.json"
GENERIC_MEMBER_PATH = "s2and/arrow_schema_contract.json"
MODULE_PATH = "s2and/__init__.py"
NATIVE_PATH = "s2and/_s2and_rust.cp311-win_amd64.pyd"
METADATA_PATH = "s2and-0.0.0.dist-info/METADATA"
REQUIRED_RUNTIME_PATHS = (
    TUPLE_DATA_PATH,
    ORCID_DATA_PATH,
    ORCID_MANIFEST_PATH,
    ORCID_V1_4_DATA_PATH,
    ORCID_V1_4_MANIFEST_PATH,
)
RELEASE_MEMBERS = {
    TUPLE_DATA_PATH: b"alice\talicia\n",
    ORCID_DATA_PATH: b"{}\n",
    ORCID_MANIFEST_PATH: b"{}\n",
    ORCID_V1_4_DATA_PATH: b'{"ale":{"alx":10}}\n',
    ORCID_V1_4_MANIFEST_PATH: b"{}\n",
    GENERIC_MEMBER_PATH: b"{}\n",
}
SOURCE_MEMBERS = {**RELEASE_MEMBERS, MODULE_PATH: b""}
WHEEL_MEMBERS = {**SOURCE_MEMBERS, NATIVE_PATH: b"native", METADATA_PATH: b"Name: s2and\n"}


def test_local_ci_runner_lifecycle_and_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, subtests: pytest.Subtests
) -> None:
    with subtests.test("lint-job-skips-project-sync"), monkeypatch.context() as mp:
        calls: list[list[str]] = []
        mp.setattr(
            run_ci,
            "sync_deps",
            lambda *, lock_present: (_ for _ in ()).throw(AssertionError("lint must not sync project dependencies")),
        )
        mp.setattr(run_ci, "run_uv", lambda args, *, env=None: calls.append(args))
        run_ci.run_lint_job()
        ruff_requirement = run_ci.exact_dev_tool_requirement("ruff")
        assert calls == [
            ["run", "--no-project", "python", "scripts/dev/sync_version.py", "--check"],
            ["run", "--no-project", "python", "scripts/dev/sync_feature_schema.py", "--check"],
            ["tool", "run", "--isolated", ruff_requirement, "check", "s2and", "scripts", "tests"],
            ["tool", "run", "--isolated", ruff_requirement, "format", "--check", "s2and", "scripts", "tests"],
        ]

    with subtests.test("ensure-rust-rejects-rustc-only"), monkeypatch.context() as mp:
        rustc_path = tmp_path / "rustc"
        mp.setattr(run_ci.shutil, "which", lambda command: str(rustc_path) if command == "rustc" else None)
        mp.setenv("USERPROFILE" if run_ci.os.name == "nt" else "HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="cargo is required"):
            run_ci.ensure_rust_on_path()

    with subtests.test("sync-deps-lock-flag"), monkeypatch.context() as mp:
        for lock_present, frozen in ((True, ["--frozen"]), (False, [])):
            sync_calls: list[list[str]] = []
            mp.setattr(run_ci, "run_uv", lambda args, *, env=None, _c=sync_calls: _c.append(args))
            run_ci.sync_deps(lock_present=lock_present)
            assert sync_calls == [["sync", "--extra", "dev", *frozen]]

    for workers, dist_args, max_processes, smoke_fails in (
        ("auto", ["--dist", "loadfile"], ["--maxprocesses", "8"], False),
        ("0", [], [], False),
        ("auto", ["--dist", "loadfile"], ["--maxprocesses", "8"], True),
    ):
        with subtests.test(workers=workers, smoke_fails=smoke_fails), monkeypatch.context() as mp:
            uv_calls: list[tuple[list[str], dict[str, str] | None]] = []
            lifecycle: list[str] = []
            mp.setattr(run_ci, "sync_deps", lambda *, lock_present, _l=lifecycle: _l.append("sync-and-build-rust"))
            mp.setattr(run_ci, "ensure_rust_on_path", lambda _l=lifecycle: _l.append("ensure-rust"))
            mp.setattr(run_ci, "run_native_rust_checks", lambda _l=lifecycle: _l.append("check-rust"))
            mp.setattr(run_ci, "run_ty_checks", lambda _l=lifecycle: _l.append("typecheck"))
            failure = subprocess.CalledProcessError(7, "native smoke")

            def _run_uv(args, *, env=None, _calls=uv_calls, _fail=smoke_fails, _exc=failure):
                _calls.append((args, env))
                if _fail:
                    raise _exc

            mp.setattr(run_ci, "run_uv", _run_uv)
            if smoke_fails:
                with pytest.raises(subprocess.CalledProcessError) as caught:
                    run_ci.run_typecheck_and_test_job(lock_present=True, pytest_workers=workers)
                assert caught.value is failure
                assert lifecycle == ["ensure-rust", "sync-and-build-rust", "check-rust"]
                assert len(uv_calls) == 1
            else:
                run_ci.run_typecheck_and_test_job(lock_present=True, pytest_workers=workers)
                assert lifecycle == ["ensure-rust", "sync-and-build-rust", "check-rust", "typecheck"]
                assert uv_calls[0][0] == [
                    "run",
                    "--no-sync",
                    "python",
                    "scripts/release/verification/smoke_installed_rust_api.py",
                ]
                assert uv_calls[1][0] == [
                    "run",
                    "--no-sync",
                    "pytest",
                    "-ra",
                    "tests/",
                    "-n",
                    workers,
                    *dist_args,
                    *max_processes,
                    "--cov=s2and",
                    "--cov-report=term-missing",
                ]


def test_feature_schema_line_endings_and_git_bash_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Native feature schema check & regeneration
    sync_feature_schema.main(["--check"])
    schema_out = tmp_path / "feature_schema.rs"
    monkeypatch.setattr(sync_feature_schema, "RUST_PATH", schema_out)
    sync_feature_schema.main([])
    generated = schema_out.read_text()
    schema_out.write_text(generated.replace("FIRST_NAMES_EQUAL: usize = 0", "FIRST_NAMES_EQUAL: usize = 1"))
    stale = schema_out.read_text()
    with pytest.raises(SystemExit, match="schema is stale"):
        sync_feature_schema.main(["--check"])
    assert schema_out.read_text() == stale
    sync_feature_schema.main([])
    sync_feature_schema.main(["--check"])
    assert schema_out.read_text() == generated

    # 2. Checksummed JSON artifacts always emit LF line endings
    for writer, path in ((_write_json, tmp_path / "bundle.json"), (write_json, tmp_path / "review.json")):
        writer(path, {"nested": {"value": 1}, "name": "x"})
        contents = path.read_bytes()
        assert b"\r" not in contents and contents.endswith(b"}\n")
    index_dir, _ = write_name_counts_index(
        tmp_path / "counts", ({"alex": 2}, {"smith": 3}, {"alex smith": 2}, {"smith a": 2})
    )
    assert b"\r" not in (Path(index_dir) / "manifest.json").read_bytes()

    # 3. Windows Git Bash discovery
    git = tmp_path / "Git" / "mingw64" / "bin" / "git.exe"
    git.parent.mkdir(parents=True)
    git.touch()
    bash = tmp_path / "Git" / "bin" / "bash.exe"
    bash.parent.mkdir(exist_ok=True)
    bash.touch()
    monkeypatch.setattr(shell_helpers, "os", SimpleNamespace(name="nt", environ=os.environ))
    monkeypatch.setattr(shell_helpers.shutil, "which", lambda name: str(git) if name == "git" else None)
    shell_helpers.bash_executable.cache_clear()
    try:
        assert shell_helpers.bash_executable() == str(bash.resolve())
    finally:
        shell_helpers.bash_executable.cache_clear()


def test_sync_version_and_pre_commit_hook(tmp_path: Path) -> None:
    # 1. Python/Cargo prerelease & final version rewriting + ambiguity rejection
    vroot = tmp_path / "version_repo"
    (vroot / "s2and_rust").mkdir(parents=True)
    (vroot / "VERSION").write_text("0.50.0\n", encoding="utf-8")
    (vroot / "pyproject.toml").write_text('[project]\nname = "s2and"\nversion = "0.49.0"\n', encoding="utf-8")
    (vroot / "s2and_rust" / "Cargo.toml").write_text(
        '[package]\nname = "s2and_rust"\nversion = "0.49.0"\n', encoding="utf-8"
    )
    (vroot / "s2and_rust" / "Cargo.lock").write_text(
        '[[package]]\nname = "s2and_rust"\nversion = "0.49.0"\n', encoding="utf-8"
    )
    (vroot / "uv.lock").write_text('[[package]]\nname = "s2and"\nversion = "0.49.0"\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="Version mismatch"):
        sync_version.verify_version("0.50.0b1", root=vroot)
    sync_version.sync_version("0.50.0b1", root=vroot)
    sync_version.verify_version("0.50.0b1", root=vroot)
    assert 'version = "0.50.0b1"' in (vroot / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0-beta.1"' in (vroot / "s2and_rust" / "Cargo.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0-beta.1"' in (vroot / "s2and_rust" / "Cargo.lock").read_text(encoding="utf-8")
    assert 'version = "0.50.0b1"' in (vroot / "uv.lock").read_text(encoding="utf-8")

    sync_version.sync_version("0.50.0", root=vroot)
    sync_version.verify_version("0.50.0", root=vroot)
    for rel in ("pyproject.toml", "s2and_rust/Cargo.toml", "s2and_rust/Cargo.lock", "uv.lock"):
        assert 'version = "0.50.0"' in (vroot / rel).read_text(encoding="utf-8")

    (vroot / "s2and_rust" / "Cargo.toml").write_text(
        '[package]\nname = "s2and_rust"\nversion = "0.49.0"\nversion = "0.49.0"\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="Expected one version match"):
        sync_version.sync_version("0.50.0", root=vroot)

    # 2. Exercise all 4 pre-commit hook branches inside one Bash invocation
    hook = (sync_version.ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    script = (
        r"""
set -euo pipefail
run_hook_case() {
    local case_dir="$1" staged="$2" check_st="$3" sync_st="$4" exp_st="$5"
    mkdir -p "$case_dir"
    (
        cd "$case_dir"
        export STAGED_FILES="$staged" CHECK_STATUS="$check_st" SYNC_STATUS="$sync_st"
        git() {
            case "$1" in
                rev-parse) pwd ;;
                diff) printf '%s\n' "$STAGED_FILES" ;;
                add) printf '<%s>\n' "${@:2}" > staged-files ;;
                *) return 97 ;;
            esac
        }
        uv() {
            printf '%s\n' "$*" >> commands
            case "$*" in
                'run python scripts/dev/sync_version.py --print-targets')
                    printf 'VERSION\r\nfolder with spaces/manifest.toml\r\nuv.lock\r\n' ;;
                'run python scripts/dev/sync_version.py --check') return "$CHECK_STATUS" ;;
                'run python scripts/dev/sync_version.py') return "$SYNC_STATUS" ;;
                'sync --extra dev') return 0 ;;
                'run --active --no-project ruff format scripts/dev/sync_version.py') return 0 ;;
                *) return 98 ;;
            esac
        }
        set +e
        (
"""
        + hook
        + r"""
        )
        actual=$?
        set -e
        test "$actual" -eq "$exp_st"
    )
}
run_hook_case c_unrelated README.md 1 0 0
run_hook_case c_synced VERSION 0 0 0
run_hook_case c_updated VERSION 1 0 0
run_hook_case c_failed VERSION 1 9 9
"""
    )
    hook_dir = tmp_path / "hook_cases"
    hook_dir.mkdir()
    completed = run_bash(script, cwd=hook_dir)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (hook_dir / "c_unrelated" / "commands").exists()
    assert (hook_dir / "c_synced" / "commands").read_text().splitlines() == [
        "run python scripts/dev/sync_version.py --print-targets",
        "run python scripts/dev/sync_version.py --check",
    ]
    assert (hook_dir / "c_updated" / "staged-files").read_text().splitlines() == [
        "<VERSION>",
        "<folder with spaces/manifest.toml>",
        "<uv.lock>",
    ]
    assert not (hook_dir / "c_failed" / "staged-files").exists()


def test_release_workflow_gates_and_dependency_graph(tmp_path: Path) -> None:
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "release-rust.yml"
    release_workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    # 1. Publication DAG & job conditions
    assert set(release_workflow["on"]) == {"push", "workflow_dispatch"}
    assert release_workflow["on"]["push"] == {"branches": ["main"]}
    jobs = release_workflow["jobs"]

    def prerequisites(job_name: str) -> set[str]:
        pending = list(jobs[job_name].get("needs", []))
        found: set[str] = set()
        while pending:
            dep = pending.pop()
            if dep not in found:
                found.add(dep)
                pending.extend(jobs[dep].get("needs", []))
        return found

    publish = f"publish-{PACKAGE}"
    assert {"release-smoke", "validate"} <= prerequisites(publish)
    assert jobs[publish]["environment"] == {"name": "pypi"}
    assert jobs[publish]["if"] == "success() && github.ref == 'refs/heads/main'"
    for name in ("source-distributions", "wheels-windows", "wheels-macos", "wheels-linux"):
        assert jobs[name]["needs"] == ["validate"]
        assert jobs[name]["if"] == "needs.validate.outputs.release == 'true'"
        assert name in prerequisites("release-smoke")

    # 2. Validate step ordering + ref & PyPI decision gates in one Bash session
    steps = jobs["validate"]["steps"]
    names = [step.get("name") for step in steps]
    assert steps[0]["uses"].startswith("actions/checkout@") and steps[0]["with"]["fetch-depth"] == "0"
    assert names[1] == "Decide whether to release"
    assert names.index("Require main for final releases") < names.index("Run the full CI policy")
    gate = next(step for step in steps if step.get("name") == "Require main for final releases")
    decide = next(step for step in steps if step.get("name") == "Decide whether to release")
    assert gate["if"] == "steps.decide.outputs.release == 'true'"
    assert gate.get("continue-on-error", "false") == "false"
    assert decide["id"] == "decide"

    repo = tmp_path / "git_repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (repo / "VERSION").write_text("0.9.0\n", encoding="utf-8")
    git("add", "VERSION")
    git("commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    (repo / "VERSION").write_text("1.0.0b6\n", encoding="utf-8")
    git("commit", "-qam", "bump")
    new = git("rev-parse", "HEAD")

    combined_bash = (
        "set -euo pipefail\n"
        f"release_gate() {{\n{gate['run']}\n}}\n"
        f"release_decision() {{\n{decide['run']}\n}}\n"
        r"""
curl() {
    test "${!#}" = "$PACKAGE_URL" || return 99
    printf '%s' "$FAKE_HTTP_STATUS"
}
check_release() {
    export GITHUB_EVENT_NAME="$1" BEFORE_SHA="$2" FAKE_HTTP_STATUS="$3"
    : > "$GITHUB_OUTPUT"
    set +e
    ( set -euo pipefail; release_decision )
    actual=$?
    set -e
    test "$actual" -eq "$4"
    test "$(cat "$GITHUB_OUTPUT")" = "$5"
}
check_release push "$OLD_SHA" 404 0 release=true
check_release push "$NEW_SHA" 404 0 release=false
check_release push "$OLD_SHA" 200 0 release=false
check_release push "$ZERO_SHA" 404 0 release=true
check_release push "$OLD_SHA" 503 1 ""
check_release workflow_dispatch "" 404 0 release=true
check_release workflow_dispatch "" 200 1 ""

mkdir -p "$REF_DIR" && cd "$REF_DIR"
check_ref() {
    printf '%s\n' "$1" > VERSION
    export GITHUB_REF="$2"
    set +e
    ( set -euo pipefail; release_gate )
    actual=$?
    set -e
    test "$actual" -eq "$3"
}
check_ref 1.0.0 refs/heads/main 0
check_ref 1.0.0 refs/tags/main 1
check_ref 1.0.0b1 refs/heads/feature 0
check_ref 1.0.0b1.post1 refs/heads/feature 1
"""
    )
    res = run_bash(
        combined_bash,
        cwd=repo,
        env={
            "PACKAGE_URL": f"https://pypi.org/pypi/{PACKAGE}/1.0.0b6/json",
            "GITHUB_OUTPUT": str(tmp_path / "github_output"),
            "OLD_SHA": old,
            "NEW_SHA": new,
            "ZERO_SHA": "0" * 40,
            "REF_DIR": str(tmp_path / "ref_check"),
        },
    )
    assert res.returncode == 0, res.stdout + res.stderr


def _write_package_data_config(root: Path, declared_paths: set[str]) -> None:
    declared = ", ".join("{ path = " + json.dumps(path) + " }" for path in sorted(declared_paths))
    (root / "pyproject.toml").write_text(f"[tool.maturin]\ninclude = [{declared}]\n", encoding="utf-8")


def _with_overrides(base: dict[str, bytes], overrides: dict[str, bytes | None] | None) -> dict[str, bytes]:
    members = dict(base)
    for path, content in (overrides or {}).items():
        if content is None:
            members.pop(path, None)
        else:
            members[path] = content
    return members


def _write_distribution_fixture(
    root: Path,
    *,
    wheel_overrides: dict[str, bytes | None] | None = None,
    sdist_overrides: dict[str, bytes | None] | None = None,
) -> tuple[Path, Path]:
    for path, content in SOURCE_MEMBERS.items():
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    _write_package_data_config(root, set(RELEASE_MEMBERS))

    dist_dir = root / "dist"
    dist_dir.mkdir()
    with zipfile.ZipFile(dist_dir / "s2and-0.0.0-cp311-cp311-win_amd64.whl", "w") as wheel:
        for path, content in _with_overrides(WHEEL_MEMBERS, wheel_overrides).items():
            wheel.writestr(path, content)
    with tarfile.open(dist_dir / "s2and-0.0.0.tar.gz", "w:gz") as sdist:
        for path, content in _with_overrides(SOURCE_MEMBERS, sdist_overrides).items():
            member = tarfile.TarInfo(f"s2and-0.0.0/{path}")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    return dist_dir, root


def test_distribution_and_license_archive_verifiers(tmp_path: Path, subtests: pytest.Subtests) -> None:
    # 1. Rust license notice verifier (wheel & sdist valid + missing + tampered)
    source = tmp_path / "license_source"
    base_notices = {path: f"license text: {path}".encode() for path in LICENSE_PATHS}
    for path, payload in base_notices.items():
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    for idx, (kind, damage) in enumerate(
        (("wheel", None), ("sdist", None), ("wheel", "missing"), ("sdist", "missing"), ("sdist", "changed"))
    ):
        with subtests.test(license_kind=kind, damage=damage):
            notices = dict(base_notices)
            if damage == "missing":
                del notices[LICENSE_PATHS[-1]]
            elif damage == "changed":
                notices[LICENSE_PATHS[-1]] = b"truncated notice"
            archive = tmp_path / (f"lic_{idx}.whl" if kind == "wheel" else f"lic_{idx}.tar.gz")
            if kind == "wheel":
                with zipfile.ZipFile(archive, "w") as wheel:
                    wheel.writestr("s2and_rust-1.0.dist-info/METADATA", "Name: s2and-rust\n")
                    for path, payload in notices.items():
                        wheel.writestr(f"s2and_rust-1.0.dist-info/licenses/{path}", payload)
            else:
                with tarfile.open(archive, "w:gz") as sdist:
                    for path, payload in notices.items():
                        member = tarfile.TarInfo(f"s2and_rust-1.0/{path}")
                        member.size = len(payload)
                        sdist.addfile(member, io.BytesIO(payload))
            if damage:
                with pytest.raises(ValueError, match="license notice"):
                    verify_archive(archive, source)
            else:
                verify_archive(archive, source)

    # 2. Production model distribution verifier: valid maturin layouts & rejection guards
    with subtests.test("valid-maturin-wheel-and-sdist-layouts"):
        dist_dir, source_root = _write_distribution_fixture(tmp_path / "valid_dist")
        licensed = {
            "s2and-0.0.0.dist-info/licenses/LICENSE": b"license\n",
            "s2and-0.0.0.dist-info/sboms/s2and_rust.cyclonedx.json": b"{}\n",
        }
        with zipfile.ZipFile(dist_dir / "s2and-0.0.0-cp313-cp313-musllinux_1_2_x86_64.whl", "w") as wheel:
            for path, payload in {**WHEEL_MEMBERS, **licensed}.items():
                if path != NATIVE_PATH:
                    wheel.writestr(path, payload)
            wheel.writestr("s2and/_s2and_rust.cpython-313-x86_64-linux-musl.so", b"native")
            wheel.writestr("s2and.libs/libgcc_s-f685abf1.so.1", b"runtime")
            wheel.writestr("s2and.libs/libstdc++-1b2c0963.so.6", b"runtime")
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

    for archive_kind, replacement, message in (
        ("sdist", None, "missing required distribution files"),
        ("wheel", b"tampered\n", "content differs from the source tree"),
    ):
        with subtests.test(drifted=archive_kind):
            overrides = {GENERIC_MEMBER_PATH: replacement}
            dist_dir, source_root = _write_distribution_fixture(
                tmp_path / f"drift_{archive_kind}",
                wheel_overrides=overrides if archive_kind == "wheel" else None,
                sdist_overrides=overrides if archive_kind == "sdist" else None,
            )
            with pytest.raises(ValueError, match=message):
                verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

    for platform, extra_lib in (
        ("musllinux_1_2_x86_64", "s2and.libs/libgcc_s.so.1"),
        ("win_amd64", "s2and.libs/libgcc_s-f685abf1.so.1"),
    ):
        with subtests.test(unexpected_bundled=platform):
            dist_dir, source_root = _write_distribution_fixture(
                tmp_path / f"lib_{platform}", wheel_overrides={extra_lib: b"unexpected"}
            )
            next(dist_dir.glob("*.whl")).rename(dist_dir / f"s2and-0.0.0-cp311-cp311-{platform}.whl")
            with pytest.raises(ValueError, match="unexpected package files"):
                verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

    with subtests.test("missing-runtime-paths-and-forbidden-model-paths"):
        dist_dir, source_root = _write_distribution_fixture(tmp_path / "missing_rt")
        _write_package_data_config(source_root, {GENERIC_MEMBER_PATH})
        with pytest.raises(ValueError, match="Release distributions require these paths") as error:
            verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)
        assert all(path in str(error.value) for path in REQUIRED_RUNTIME_PATHS)

        dist_dir2, source_root2 = _write_distribution_fixture(
            tmp_path / "forbidden_model", sdist_overrides={DEFAULT_MODEL_PATH: b"{}\n", MODEL_MEMBER_PATH: b"{}\n"}
        )
        with pytest.raises(ValueError, match="forbidden production model paths") as error2:
            verify_production_model_distributions(dist_dir=dist_dir2, source_root=source_root2)
        assert DEFAULT_MODEL_PATH in str(error2.value) and MODEL_MEMBER_PATH in str(error2.value)

    for archive_kind in ("wheel", "sdist"):
        with subtests.test(exact_inventory=archive_kind):
            unexpected = (
                {
                    "s2and/data/extra.json": b"{}",
                    "s2and/__pycache__/model.cpython-311.pyc": b"",
                    "scripts/extra.py": b"",
                }
                if archive_kind == "wheel"
                else {NATIVE_PATH: b"native", "s2and/data/extra.json": b"{}"}
            )
            overrides: dict[str, bytes | None] = {**unexpected, MODULE_PATH: None}
            if archive_kind == "wheel":
                overrides[NATIVE_PATH] = None
            dist_dir, source_root = _write_distribution_fixture(
                tmp_path / f"inv_{archive_kind}",
                wheel_overrides=overrides if archive_kind == "wheel" else None,
                sdist_overrides=overrides if archive_kind == "sdist" else None,
            )
            with pytest.raises(ValueError, match="unexpected package files") as error:
                verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)
            assert all(path in str(error.value) for path in unexpected)
            assert "missing source modules" in str(error.value) and MODULE_PATH in str(error.value)
            if archive_kind == "wheel":
                assert "must contain 1 native extension" in str(error.value)


def test_ty_covers_every_script_stage_folder() -> None:
    # A new stage folder missing from the hand list would skip type checking silently.
    repo = Path(run_ci.__file__).resolve().parents[2]
    stage_dirs = {
        f"scripts/{path.name}"
        for path in (repo / "scripts").iterdir()
        if path.is_dir() and any(path.rglob("*.py")) and path.name != "__pycache__"
    }
    assert stage_dirs == set(run_ci.ACTIVE_NESTED_SCRIPT_DIRS)
