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
    ("ref", "allowed"),
    [
        ("refs/heads/main", True),
        ("refs/heads/main-backup", False),
        ("refs/tags/main", False),
        ("", False),
    ],
)
def test_release_ref_gate_rejects_unapproved_refs(release_workflow, ref: str, allowed: bool) -> None:
    gate = release_workflow["jobs"]["validate"]["steps"][0]
    assert "if" not in gate
    assert gate.get("continue-on-error", "false") == "false"
    result = run_bash("set -euo pipefail\n" + gate["run"], env={"GITHUB_REF": ref})
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
        assert "if" not in jobs[name], "Publication must retain GitHub's default success dependency condition"
    assert "publish-rust" in prerequisites("publish-s2and")
    for name in ("source-distributions", "wheels-windows", "wheels-macos", "wheels-linux"):
        assert "validate" in prerequisites(name)
        assert name in prerequisites("release-smoke")
