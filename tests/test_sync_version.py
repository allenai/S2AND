from pathlib import Path

import pytest

from scripts import sync_version


def _write_version_fixture(root: Path) -> None:
    (root / "s2and").mkdir()
    (root / "s2and_rust").mkdir()
    (root / "docs").mkdir()
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
    (root / "README.md").write_text("echo 0.49.0 > VERSION\n", encoding="utf-8")
    (root / "docs" / "development.md").write_text("echo 0.40.0 > VERSION\n", encoding="utf-8")
    (root / "docs" / "release_notes.md").write_text(
        "\n".join(
            [
                "# Release Notes",
                "",
                "## 0.49.0",
                "",
                "- Ships the package as `0.49.0` and pins optional Rust installs to `s2and-rust==0.49.0`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (root / "s2and_rust" / "README.md").write_text(
        "\n".join(
            [
                "# s2and-rust",
                "",
                "This checkout is `0.49.0`, so use a local build when working from this tree",
                "until the matching packages are published.",
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
    assert "echo 0.50.0 > VERSION" in (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "echo 0.50.0 > VERSION" in (tmp_path / "docs" / "development.md").read_text(encoding="utf-8")
    release_notes = (tmp_path / "docs" / "release_notes.md").read_text(encoding="utf-8")
    assert "## 0.49.0" in release_notes
    assert "package as `0.49.0` and pins optional Rust installs to `s2and-rust==0.49.0`" in release_notes
    assert "This checkout is `0.50.0`" in (tmp_path / "s2and_rust" / "README.md").read_text(encoding="utf-8")
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


def test_pre_commit_stages_sync_version_targets() -> None:
    hook_text = (sync_version.ROOT / ".githooks" / "pre-commit").read_text(encoding="utf-8")

    for target in sync_version.version_targets():
        assert target.relative_path.as_posix() in hook_text
    assert "docs/release_notes.md" not in hook_text
