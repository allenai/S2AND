"""Exercise the release decision and ref gates and verify the publication dependency graph."""

import subprocess
from pathlib import Path

import pytest
import yaml

from tests.shell_helpers import run_bash

PACKAGE = "s2and"


@pytest.fixture(scope="module")
def release_workflow():
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "release-rust.yml"
    # BaseLoader preserves GitHub's `on` key instead of interpreting YAML 1.1 booleans.
    return yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def validate_step(release_workflow, name):
    steps = release_workflow["jobs"]["validate"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_release_ref_gate_rejects_unapproved_refs(release_workflow, tmp_path: Path) -> None:
    steps = release_workflow["jobs"]["validate"]["steps"]
    names = [step.get("name") for step in steps]
    # The decision reads VERSION and git history, so only the checkout may run before it.
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[0]["with"]["fetch-depth"] == "0"
    assert names[1] == "Decide whether to release"
    gate = validate_step(release_workflow, "Require main for final releases")
    assert names.index("Require main for final releases") < names.index("Run the full CI policy")
    assert gate["if"] == "steps.decide.outputs.release == 'true'"
    assert gate.get("continue-on-error", "false") == "false"

    setup = r"""
check_ref() {
    printf '%s\n' "$1" > VERSION
    export GITHUB_REF="$2"
    set +e
    ( set -euo pipefail; release_gate )
    actual=$?
    set -e
    test "$actual" -eq "$3"
}
"""
    gates = """
check_ref 1.0.0 refs/heads/main 0
check_ref 1.0.0 refs/tags/main 1
check_ref 1.0.0b1 refs/heads/feature 0
check_ref 1.0.0b1.post1 refs/heads/feature 1
"""
    result = run_bash("set -euo pipefail\nrelease_gate() {\n" + gate["run"] + "\n}\n" + setup + gates, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def version_bump(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    path = tmp_path_factory.mktemp("release_version_bump")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (path / "VERSION").write_text("0.9.0\n", encoding="utf-8")
    git("add", "VERSION")
    git("commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    (path / "VERSION").write_text("1.0.0b6\n", encoding="utf-8")
    git("commit", "-qam", "bump")
    return path, old, git("rev-parse", "HEAD")


def _check_release_decisions(
    release_workflow, version_bump: tuple[Path, str, str], tmp_path: Path, event: str, checks: str
) -> None:
    decide = validate_step(release_workflow, "Decide whether to release")
    assert decide["id"] == "decide"
    repo, old, new = version_bump
    setup = """
curl() {
    test "${!#}" = "$PACKAGE_URL" || return 99
    printf '%s' "$FAKE_HTTP_STATUS"
}
check_release() {
    export BEFORE_SHA="$1" FAKE_HTTP_STATUS="$2"
    : > "$GITHUB_OUTPUT"
    set +e
    ( set -euo pipefail; release_decision )
    actual=$?
    set -e
    test "$actual" -eq "$3"
    test "$(cat "$GITHUB_OUTPUT")" = "$4"
}
"""
    result = run_bash(
        "set -euo pipefail\nrelease_decision() {\n" + decide["run"] + "\n}\n" + setup + checks,
        cwd=repo,
        env={
            "PACKAGE_URL": f"https://pypi.org/pypi/{PACKAGE}/1.0.0b6/json",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_OUTPUT": str(tmp_path / "github_output"),
            "OLD_SHA": old,
            "NEW_SHA": new,
            "ZERO_SHA": "0" * 40,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_push_releases_only_a_version_bump_missing_from_pypi(release_workflow, version_bump, tmp_path: Path) -> None:
    _check_release_decisions(
        release_workflow,
        version_bump,
        tmp_path,
        "push",
        """
check_release "$OLD_SHA" 404 0 release=true
check_release "$NEW_SHA" 404 0 release=false
check_release "$OLD_SHA" 200 0 release=false
check_release "$ZERO_SHA" 404 0 release=true
check_release "$OLD_SHA" 503 1 ""
""",
    )


def test_manual_release_requires_an_unpublished_version(release_workflow, version_bump, tmp_path: Path) -> None:
    _check_release_decisions(
        release_workflow,
        version_bump,
        tmp_path,
        "workflow_dispatch",
        """
check_release "" 404 0 release=true
check_release "" 200 1 ""
""",
    )


def test_release_publication_requires_successful_validation_and_smoke(release_workflow) -> None:
    # A merge to main starts a release; a manual dispatch can start one too.
    assert set(release_workflow["on"]) == {"push", "workflow_dispatch"}
    assert release_workflow["on"]["push"] == {"branches": ["main"]}
    jobs = release_workflow["jobs"]

    def prerequisites(job_name):
        pending = list(jobs[job_name].get("needs", []))
        found = set()
        while pending:
            dependency = pending.pop()
            if dependency not in found:
                found.add(dependency)
                pending.extend(jobs[dependency].get("needs", []))
        return found

    publish = f"publish-{PACKAGE}"
    assert {"release-smoke", "validate"} <= prerequisites(publish)
    assert jobs[publish]["environment"] == {"name": "pypi"}
    # Only main publishes, and publication still requires every prerequisite to succeed.
    assert jobs[publish]["if"] == "success() && github.ref == 'refs/heads/main'"
    # Every build waits for validate's decision, so a push that releases nothing builds nothing.
    for name in ("source-distributions", "wheels-windows", "wheels-macos", "wheels-linux"):
        assert jobs[name]["needs"] == ["validate"]
        assert jobs[name]["if"] == "needs.validate.outputs.release == 'true'"
        assert name in prerequisites("release-smoke")
