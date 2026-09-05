"""Check version updates and the complete pre-commit hook's failure boundaries."""

from pathlib import Path

import pytest

from scripts import sync_version
from tests.shell_helpers import run_bash


@pytest.mark.parametrize(
    ("staged", "check_status", "sync_status", "expected_status", "updates"),
    [
        ("README.md", "1", "0", 0, False),
        ("VERSION", "0", "0", 0, False),
        ("VERSION", "1", "0", 0, True),
        ("VERSION", "1", "9", 9, False),
    ],
    ids=["unrelated-change", "already-synchronized", "update-and-stage", "sync-failure"],
)
def test_pre_commit_hook_only_stages_successfully_synchronized_targets(
    tmp_path: Path, staged: str, check_status: str, sync_status: str, expected_status: int, updates: bool
) -> None:
    """Run the whole hook with command boundaries stubbed, never touching real Git."""
    hook = (sync_version.ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")
    commands = r"""
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
        'run python scripts/sync_version.py --print-targets')
            printf 'VERSION\r\nfolder with spaces/manifest.toml\r\nuv.lock\r\n' ;;
        'run python scripts/sync_version.py --check') return "$CHECK_STATUS" ;;
        'run python scripts/sync_version.py') return "$SYNC_STATUS" ;;
        'sync --extra dev') return 0 ;;
        'run --active --no-project cargo generate-lockfile --manifest-path s2and_rust/Cargo.toml') return 0 ;;
        'run --active --no-project ruff format scripts/sync_version.py') return 0 ;;
        *) return 98 ;;
    esac
}
"""
    completed = run_bash(
        commands + hook,
        cwd=tmp_path,
        env={"STAGED_FILES": staged, "CHECK_STATUS": check_status, "SYNC_STATUS": sync_status},
    )
    assert completed.returncode == expected_status, completed.stdout + completed.stderr
    staged_files = tmp_path / "staged-files"
    if updates:
        assert staged_files.read_text().splitlines() == ["<VERSION>", "<folder with spaces/manifest.toml>", "<uv.lock>"]
    else:
        assert not staged_files.exists()
    log = tmp_path / "commands"
    if staged != "VERSION":
        assert not log.exists(), "Unrelated commits must not run version tooling"
    elif check_status == "0":
        assert log.read_text().splitlines() == [
            "run python scripts/sync_version.py --print-targets",
            "run python scripts/sync_version.py --check",
        ]
    elif sync_status != "0":
        assert log.read_text().splitlines()[-1] == "run python scripts/sync_version.py"
    else:
        assert "sync --extra dev" in log.read_text().splitlines()


def _write_version_fixture(root: Path) -> None:
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


def test_sync_version_updates_rust_manifests_and_lockfiles(tmp_path: Path) -> None:
    _write_version_fixture(tmp_path)

    with pytest.raises(SystemExit, match="Version mismatch"):
        sync_version.verify_version("0.50.0", root=tmp_path)

    sync_version.sync_version("0.50.0", root=tmp_path)
    sync_version.verify_version("0.50.0", root=tmp_path)

    assert '"s2and-rust==0.50.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "Cargo.toml").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "s2and_rust" / "Cargo.lock").read_text(encoding="utf-8")
    assert 'version = "0.50.0"' in (tmp_path / "uv.lock").read_text(encoding="utf-8")


def test_sync_version_rejects_ambiguous_targets(tmp_path: Path) -> None:
    _write_version_fixture(tmp_path)
    (tmp_path / "s2and_rust" / "Cargo.toml").write_text(
        '[package]\nname = "s2and_rust"\nversion = "0.49.0"\nversion = "0.49.0"\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Expected one version match"):
        sync_version.sync_version("0.50.0", root=tmp_path)
