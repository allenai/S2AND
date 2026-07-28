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
