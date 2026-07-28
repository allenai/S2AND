from pathlib import Path

import yaml


def test_release_workflow_preserves_publication_guards() -> None:
    workflow_path = Path(__file__).parents[1] / ".github" / "workflows" / "release-rust.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow[True] == {"workflow_dispatch": None}
    first_step = workflow["jobs"]["validate"]["steps"][0]
    assert first_step["run"] == 'test "$GITHUB_REF" = "refs/heads/main"'
    jobs = workflow["jobs"]
    assert jobs["publish-rust"]["needs"] == ["release-smoke"]
    assert jobs["publish-s2and"]["needs"] == ["publish-rust"]
    assert jobs["publish-rust"]["environment"] == jobs["publish-s2and"]["environment"] == {"name": "pypi"}
