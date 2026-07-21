import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import sync_version


def test_pre_commit_hook_collects_targets_without_bash_4_mapfile() -> None:
    hook = (sync_version.ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    assert "mapfile" not in hook
    assert 'VERSION_TARGETS="$(uv run python scripts/sync_version.py --print-targets)"' in hook
    assert "while IFS= read -r version_file" in hook
    assert "${version_file%$'\\r'}" in hook


def test_pre_commit_target_loop_strips_windows_crlf() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    hook = (sync_version.ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    loop_start = hook.index("VERSION_FILES=()")
    loop_end_marker = 'done <<< "$VERSION_TARGETS"'
    loop_end = hook.index(loop_end_marker, loop_start) + len(loop_end_marker)
    target_loop = hook[loop_start:loop_end]
    script = "\n".join(
        (
            "set -euo pipefail",
            "VERSION_TARGETS=\"$(printf 'VERSION\\r\\npyproject.toml\\r\\nuv.lock\\r\\n')\"",
            target_loop,
            "printf '<%s>\\n' \"${VERSION_FILES[@]}\"",
        )
    )

    completed = subprocess.run(
        [bash],
        check=True,
        capture_output=True,
        input=script.encode("utf-8"),
    )

    assert completed.stdout.decode("utf-8").splitlines() == ["<VERSION>", "<pyproject.toml>", "<uv.lock>"]


def _write_version_fixture(root: Path) -> None:
    (root / "s2and").mkdir()
    (root / "s2and_rust").mkdir()
    (root / "VERSION").write_text("0.50.0\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project.optional-dependencies]",
                "rust = [",
                '  "s2and-rust==0.49.0",',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "s2and_rust" / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "s2and-rust"',
                'version = "0.49.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "s2and_rust" / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                'name = "s2and_rust"',
                'version = "0.49.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "s2and" / "runtime.py").write_text(
        'REQUIRED_RUST_EXTENSION_VERSION = "0.49.0"\n',
        encoding="utf-8",
    )
    (root / "s2and_rust" / "Cargo.lock").write_text(
        "\n".join(
            [
                "[[package]]",
                'name = "s2and_rust"',
                'version = "0.49.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        "\n".join(
            [
                "[[package]]",
                'name = "s2and-rust"',
                'version = "0.49.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_sync_version_updates_rust_manifests_runtime_guard_and_lockfiles(tmp_path: Path) -> None:
    _write_version_fixture(tmp_path)

    with pytest.raises(SystemExit, match="Version mismatch"):
        sync_version.verify_version("0.50.0", root=tmp_path)

    sync_version.sync_version("0.50.0", root=tmp_path)
    sync_version.verify_version("0.50.0", root=tmp_path)

    assert '"s2and-rust==0.50.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "Cargo.toml").read_text(encoding="utf-8")
    assert 'REQUIRED_RUST_EXTENSION_VERSION = "0.50.0"' in (tmp_path / "s2and" / "runtime.py").read_text(
        encoding="utf-8"
    )
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "Cargo.lock").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "uv.lock").read_text(encoding="utf-8")


def test_sync_version_rejects_ambiguous_targets(tmp_path: Path) -> None:
    _write_version_fixture(tmp_path)
    (tmp_path / "s2and" / "runtime.py").write_text(
        "\n".join(
            [
                'REQUIRED_RUST_EXTENSION_VERSION = "0.50.0"',
                'REQUIRED_RUST_EXTENSION_VERSION = "0.50.0"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Expected one version match"):
        sync_version.sync_version("0.50.0", root=tmp_path)


@pytest.mark.parametrize(
    (
        "event_name",
        "ref",
        "version_changed",
        "force_build",
        "publish_s2and_requested",
        "publish_rust_requested",
        "expected",
    ),
    [
        (
            "pull_request",
            "refs/pull/80/merge",
            True,
            False,
            False,
            False,
            (True, True, False, False, False),
        ),
        (
            "pull_request",
            "refs/pull/80/merge",
            False,
            True,
            False,
            False,
            (True, True, True, False, False),
        ),
        (
            "push",
            "refs/heads/main",
            True,
            False,
            False,
            False,
            (True, True, True, True, True),
        ),
        (
            "workflow_dispatch",
            "refs/heads/main",
            False,
            False,
            True,
            False,
            (True, True, True, True, False),
        ),
        (
            "workflow_dispatch",
            "refs/heads/main",
            False,
            False,
            False,
            True,
            (False, True, False, False, True),
        ),
        (
            "workflow_dispatch",
            "refs/heads/main",
            False,
            True,
            False,
            False,
            (True, True, True, False, False),
        ),
        (
            "workflow_dispatch",
            "refs/heads/topic",
            False,
            False,
            True,
            True,
            (True, True, False, False, False),
        ),
    ],
)
def test_release_decisions_are_resolved_in_one_policy(
    event_name: str,
    ref: str,
    version_changed: bool,
    force_build: bool,
    publish_s2and_requested: bool,
    publish_rust_requested: bool,
    expected: tuple[bool, bool, bool, bool, bool],
) -> None:
    decisions = sync_version.release_decisions(
        event_name=event_name,
        ref=ref,
        version_changed=version_changed,
        force_build=force_build,
        publish_s2and_requested=publish_s2and_requested,
        publish_rust_requested=publish_rust_requested,
    )

    assert (
        decisions.build_s2and,
        decisions.build_rust,
        decisions.run_release_smoke,
        decisions.publish_s2and,
        decisions.publish_rust,
    ) == expected


def test_manual_release_policy_does_not_infer_a_version_change(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("0.50.0\n", encoding="utf-8")
    event_path = tmp_path / "event.json"
    event_path.write_text(
        '{"inputs": {"force_build": "false", "publish_s2and": "false", "publish_rust": "true"}}',
        encoding="utf-8",
    )

    decisions, before_version, current_version = sync_version.release_decisions_from_environment(
        root=tmp_path,
        environ={
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_REF": "refs/heads/main",
        },
    )

    assert before_version is None
    assert current_version == "0.50.0"
    assert decisions == sync_version.ReleaseDecisions(
        build_s2and=False,
        build_rust=True,
        run_release_smoke=False,
        publish_s2and=False,
        publish_rust=True,
    )


def test_release_policy_writes_only_final_workflow_outputs(tmp_path: Path) -> None:
    output_path = tmp_path / "github-output"
    decisions = sync_version.ReleaseDecisions(True, True, False, False, True)

    sync_version._write_github_outputs(output_path, decisions.github_outputs())

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "build_s2and=true",
        "build_rust=true",
        "run_release_smoke=false",
        "publish_s2and=false",
        "publish_rust=true",
    ]
