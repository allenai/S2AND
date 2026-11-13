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


@pytest.mark.parametrize(
    ("version", "ref", "allowed"),
    [
        ("1.0.0", "refs/heads/main", True),
        ("1.0.0", "refs/heads/main-backup", False),
        ("1.0.0", "refs/tags/main", False),
        ("1.0.0", "", False),
        # Pre-releases build on any branch; the publish jobs still require main.
        ("1.0.0b1", "refs/heads/feature", True),
        ("1.0.0rc2", "", True),
        ("1.0.0b1.post1", "refs/heads/feature", False),
    ],
)
def test_release_ref_gate_rejects_unapproved_refs(
    release_workflow, tmp_path: Path, version: str, ref: str, allowed: bool
) -> None:
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
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")
    result = run_bash("set -euo pipefail\n" + gate["run"], cwd=tmp_path, env={"GITHUB_REF": ref})
    assert (result.returncode == 0) is allowed, result.stdout + result.stderr


def git_repo_with_version_bump(path: Path) -> tuple[str, str]:
    """Commits VERSION 0.9.0, then 1.0.0b6; returns both commit ids."""

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
    return old, git("rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("event", "before", "status", "exit_ok", "release"),
    [
        # A push that changes VERSION releases a version PyPI lacks.
        ("push", "old", "404", True, "true"),
        # A push that keeps VERSION, or whose version PyPI has, releases nothing.
        ("push", "new", "404", True, "false"),
        ("push", "old", "200", True, "false"),
        # A push with no usable previous commit compares with nothing.
        ("push", "zero", "404", True, "true"),
        ("push", "old", "503", False, None),
        # A manual run releases only an unpublished version.
        ("workflow_dispatch", "", "404", True, "true"),
        ("workflow_dispatch", "", "200", False, None),
        ("workflow_dispatch", "", "503", False, None),
    ],
)
def test_release_runs_for_new_unpublished_versions(
    release_workflow, tmp_path: Path, event: str, before: str, status: str, exit_ok: bool, release
) -> None:
    decide = validate_step(release_workflow, "Decide whether to release")
    assert decide["id"] == "decide"
    old, new = git_repo_with_version_bump(tmp_path)
    before_sha = {"old": old, "new": new, "zero": "0" * 40, "": ""}[before]
    # A shell function stands in for curl so the test never contacts PyPI.
    fake_curl = 'curl() { printf "%s\\n" "$*" > curl_args; printf "%s" "$FAKE_HTTP_STATUS"; }\n'
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    env = {
        "FAKE_HTTP_STATUS": status,
        "GITHUB_EVENT_NAME": event,
        "BEFORE_SHA": before_sha,
        "GITHUB_OUTPUT": str(output),
    }
    result = run_bash("set -euo pipefail\n" + fake_curl + decide["run"], cwd=tmp_path, env=env)
    assert (result.returncode == 0) is exit_ok, result.stdout + result.stderr
    assert f"https://pypi.org/pypi/{PACKAGE}/1.0.0b6/json" in (tmp_path / "curl_args").read_text(encoding="utf-8")
    if release is not None:
        assert output.read_text(encoding="utf-8").split() == [f"release={release}"]


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
