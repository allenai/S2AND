"""Exercise the release ref gate and verify the publication dependency graph."""

from pathlib import Path

import pytest
import yaml

from tests.shell_helpers import run_bash


@pytest.fixture(scope="module")
def release_workflow():
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "release-rust.yml"
    # BaseLoader preserves GitHub's `on` key instead of interpreting YAML 1.1 booleans.
    return yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


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
    # The gate reads VERSION, so only the checkout may run before it.
    assert steps[0]["uses"].startswith("actions/checkout@")
    gate = steps[1]
    assert gate["name"] == "Require main for final releases"
    assert "if" not in gate
    assert gate.get("continue-on-error", "false") == "false"
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")
    result = run_bash("set -euo pipefail\n" + gate["run"], cwd=tmp_path, env={"GITHUB_REF": ref})
    assert (result.returncode == 0) is allowed, result.stdout + result.stderr


def test_release_publication_requires_successful_validation_and_smoke(release_workflow) -> None:
    assert set(release_workflow["on"]) == {"workflow_dispatch"}
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

    for name in ("publish-rust", "publish-s2and"):
        assert {"release-smoke", "validate"} <= prerequisites(name)
        assert jobs[name]["environment"] == {"name": "pypi"}
    # Only main publishes, and publication still requires every prerequisite to succeed.
    assert jobs["publish-rust"]["if"] == "success() && github.ref == 'refs/heads/main'"
    assert "if" not in jobs["publish-s2and"], "Publication must retain GitHub's default success dependency condition"
    assert "publish-rust" in prerequisites("publish-s2and")
    for name in ("source-distributions", "wheels-windows", "wheels-macos", "wheels-linux"):
        assert "validate" in prerequisites(name)
        assert name in prerequisites("release-smoke")
