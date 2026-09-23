"""Source digest used to reject stale native extensions before collection."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.rust_build_freshness import REBUILD_COMMAND, rust_source_digest

_TREE = {
    "Cargo.toml": b"[package]\n",
    "Cargo.lock": b"# lock\n",
    "build.rs": b"fn main() {}\n",
    "src/lib.rs": b"pub fn f() {}\n",
    "src/nested/mod.rs": b"pub fn g() {}\n",
    "vendor/dep/src/lib.rs": b"pub fn h() {}\n",
}


def _write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_digest_is_independent_of_file_creation_order(tmp_path: Path) -> None:
    forward = _write_tree(tmp_path / "forward", _TREE)
    reverse = _write_tree(tmp_path / "reverse", dict(reversed(list(_TREE.items()))))

    assert rust_source_digest(forward) == rust_source_digest(reverse)
    assert rust_source_digest(forward).startswith("v1:")


@pytest.mark.parametrize(
    "change",
    [
        "modify_source",
        "modify_manifest",
        "modify_lockfile",
        "modify_build_script",
        "add_source",
        "delete_source",
        "rename_source",
        "modify_vendor",
    ],
)
def test_digest_changes_for_every_build_input_change(tmp_path: Path, change: str) -> None:
    root = _write_tree(tmp_path, _TREE)
    baseline = rust_source_digest(root)

    if change == "modify_source":
        (root / "src/lib.rs").write_bytes(b"pub fn f() { }\n")
    elif change == "modify_manifest":
        (root / "Cargo.toml").write_bytes(b"[package]\nname = 'x'\n")
    elif change == "modify_lockfile":
        (root / "Cargo.lock").write_bytes(b"# updated lock\n")
    elif change == "modify_build_script":
        (root / "build.rs").write_bytes(b"fn main() { }\n")
    elif change == "add_source":
        (root / "src/new.rs").write_bytes(b"")
    elif change == "delete_source":
        (root / "src/nested/mod.rs").unlink()
    elif change == "rename_source":
        (root / "src/nested/mod.rs").rename(root / "src/nested/other.rs")
    else:
        (root / "vendor/dep/src/lib.rs").write_bytes(b"pub fn h() { }\n")

    assert rust_source_digest(root) != baseline


def test_digest_ignores_build_outputs_outside_declared_inputs(tmp_path: Path) -> None:
    root = _write_tree(tmp_path, _TREE)
    baseline = rust_source_digest(root)

    _write_tree(root, {"target/release/lib.dll": b"binary", "s2and_rust/__init__.py": b""})

    assert rust_source_digest(root) == baseline


def _session_project(root: Path, *, workspace_shim: bool) -> tuple[Path, Path]:
    """Copy the real session hook into a small independent pytest project."""
    source_root = _write_tree(root / "s2and_rust", _TREE)
    tests = root / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    for name in ("conftest.py", "rust_build_freshness.py"):
        shutil.copyfile(Path(__file__).with_name(name), tests / name)
    (tests / "test_collection.py").write_text(
        "from pathlib import Path\nPath('collected').touch()\ndef test_collected():\n    pass\n",
        encoding="utf-8",
    )
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    if workspace_shim:
        (source_root / "__init__.py").write_text("from .s2and_rust import *\n", encoding="utf-8")
        package = source_root / "s2and_rust"
    else:
        package = root / "installed" / "s2and_rust"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ._s2and_rust import *\n", encoding="utf-8")
    return source_root, package / "_s2and_rust.py"


def _run_session(root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "installed")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini", "tests"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize("workspace_shim", [True, False], ids=["workspace-shim", "installed-package"])
def test_session_rejects_source_edit_before_collection(tmp_path: Path, workspace_shim: bool) -> None:
    """Exercise the real hook and import path, without compiling a fake crate."""
    source_root, native = _session_project(tmp_path, workspace_shim=workspace_shim)
    digest = rust_source_digest(source_root)
    native.write_text(f"def get_build_info():\n    return {{'source_digest': {digest!r}}}\n", encoding="utf-8")

    fresh = _run_session(tmp_path)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert (tmp_path / "collected").exists()
    (tmp_path / "collected").unlink()

    (source_root / "src/lib.rs").write_bytes(b"pub fn changed() {}\n")

    stale = _run_session(tmp_path)
    output = stale.stdout + stale.stderr
    assert stale.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert str(native) in output
    assert REBUILD_COMMAND in output
    assert "different sources" in output


@pytest.mark.parametrize("workspace_shim", [True, False], ids=["workspace-shim", "installed-package"])
def test_session_rejects_extension_without_source_digest(tmp_path: Path, workspace_shim: bool) -> None:
    _, native = _session_project(tmp_path, workspace_shim=workspace_shim)
    native.write_text("def get_build_info():\n    return {'crate_version': '1.0.0'}\n", encoding="utf-8")

    result = _run_session(tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert "embedded=None" in output
    assert str(native) in output
    assert REBUILD_COMMAND in output


def test_session_explains_missing_extension(tmp_path: Path) -> None:
    _session_project(tmp_path, workspace_shim=True)

    result = _run_session(tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert "Cannot import the s2and_rust extension" in output
    assert REBUILD_COMMAND in output


def test_real_build_script_tracks_directories_and_current_package_root(tmp_path: Path) -> None:
    """Exercise Cargo invalidation and the Rust/Python digest agreement."""
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("Cargo is required to exercise native build-script invalidation")
    source = _write_tree(
        tmp_path / "crate",
        {
            "Cargo.toml": (
                b'[package]\nname = "s2and_freshness_probe"\nversion = "0.0.0"\nedition = "2021"\n'
                b'[build-dependencies]\nsha2 = "0.10"\n'
            ),
            "src/main.rs": b'fn main() { println!("{}", env!("S2AND_RUST_SOURCE_DIGEST")); }\n',
            "vendor/keep.rs": b"// Vendored input\n",
        },
    )
    shutil.copyfile(Path(__file__).resolve().parents[1] / "s2and_rust/build.rs", source / "build.rs")
    target = tmp_path / "target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYO3_PYTHON"] = sys.executable
    command = [cargo, "run", "--offline", "--quiet", "--manifest-path", str(source / "Cargo.toml")]

    def assert_current_digest() -> str:
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True, check=False, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        digest = rust_source_digest(source)
        assert result.stdout.strip() == digest
        return digest

    baseline = assert_current_digest()
    for relative in ("src/new.rs", "vendor/new.rs"):
        added = source / relative
        added.write_bytes(b"// Newly added, unreferenced build input\n")
        assert assert_current_digest() != baseline
        added.unlink()
        assert assert_current_digest() == baseline

    # A compiled build script may be reused with a shared Cargo target. Its
    # digest must use the current manifest directory supplied at execution.
    alternate = tmp_path / "alternate"
    shutil.copytree(source, alternate)
    (alternate / "src/new.rs").write_bytes(b"// Only in the current package\n")
    build_script_name = "build-script-build.exe" if os.name == "nt" else "build-script-build"
    build_script = next((target / "debug/build").glob(f"s2and_freshness_probe-*/{build_script_name}"))
    env["CARGO_MANIFEST_DIR"] = str(alternate)
    env["OUT_DIR"] = str(target / "debug/build/direct/out")
    result = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"cargo:rustc-env=S2AND_RUST_SOURCE_DIGEST={rust_source_digest(alternate)}" in result.stdout
