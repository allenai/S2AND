"""Source digest used to reject stale native extensions before collection."""

import importlib.machinery
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.rust_build_freshness import REBUILD_COMMAND, rust_source_digest, stale_rust_extension_message

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


def _uv_cache_project(root: Path) -> tuple[Path, Path]:
    """Use the native package's real cache keys with an offline wheel builder."""
    source = _write_tree(root / "package", _TREE)
    native_manifest = Path(__file__).resolve().parents[1] / "s2and_rust/pyproject.toml"
    keys = tomllib.loads(native_manifest.read_text(encoding="utf-8"))["tool"]["uv"]["cache-keys"]
    cache_keys = ", ".join("{ file = " + json.dumps(key["file"]) + " }" for key in keys)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "s2and-rust"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        f"[tool.uv]\ncache-keys = [{cache_keys}]\n",
        encoding="utf-8",
    )
    (source / "backend.py").write_text(
        "from pathlib import Path\n"
        "from zipfile import ZipFile\n"
        "\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    counter = Path(__file__).resolve().parent / 'build_count.txt'\n"
        "    count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "    counter.write_text(str(count))\n"
        "    name = 's2and_rust-0.0.0-py3-none-any.whl'\n"
        "    dist = 's2and_rust-0.0.0.dist-info'\n"
        "    with ZipFile(Path(wheel_directory) / name, 'w') as wheel:\n"
        "        wheel.writestr('freshness_probe.py', f'BUILD_COUNT = {count}\\n')\n"
        "        wheel.writestr(dist + '/METADATA', "
        "'Metadata-Version: 2.1\\nName: s2and-rust\\nVersion: 0.0.0\\n')\n"
        "        wheel.writestr(dist + '/WHEEL', "
        "'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
        "        wheel.writestr(dist + '/RECORD', '')\n"
        "    return name\n",
        encoding="utf-8",
    )
    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "freshness-probe-host"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["s2and-rust"]\n[project.optional-dependencies]\ndev = []\n'
        '[tool.uv.sources]\ns2and-rust = { path = "../package" }\n',
        encoding="utf-8",
    )
    return source, project


def _sync_uv_cache_project(project: Path, *, recovery: bool = False) -> int:
    """Run uv against only the fixture's environment and local cache."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to exercise local package cache invalidation")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(project / ".venv")
    arguments = shlex.split(REBUILD_COMMAND)[1:] if recovery else ["sync", "--extra", "dev"]
    result = subprocess.run(
        [
            uv,
            "--cache-dir",
            str(project.parent / "cache"),
            *arguments,
            "--project",
            str(project),
            "--python",
            sys.executable,
            "--offline",
            "--no-python-downloads",
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return int((project.parent / "package/build_count.txt").read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative", ["src/lib.rs.orig", "src/.swp", "vendor/other/README.md"])
def test_uv_rebuilds_when_digest_tracks_non_rust_files(tmp_path: Path, relative: str) -> None:
    """Changes rejected by the freshness guard must also invalidate uv's wheel."""
    source, project = _uv_cache_project(tmp_path)
    baseline = rust_source_digest(source)
    assert _sync_uv_cache_project(project) == 1
    assert _sync_uv_cache_project(project) == 1

    added = source / relative
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_bytes(b"new build input\n")
    added_digest = rust_source_digest(source)
    assert added_digest != baseline
    assert _sync_uv_cache_project(project) == 2

    added.write_bytes(b"changed build input\n")
    assert rust_source_digest(source) != added_digest
    assert _sync_uv_cache_project(project) == 3

    added.unlink()
    assert rust_source_digest(source) == baseline
    assert _sync_uv_cache_project(project) == 4


def test_recovery_command_rebuilds_an_unchanged_cached_package(tmp_path: Path) -> None:
    """The suggested recovery must force a build even when cache keys match."""
    _, project = _uv_cache_project(tmp_path)
    assert _sync_uv_cache_project(project) == 1
    assert _sync_uv_cache_project(project) == 1
    assert _sync_uv_cache_project(project, recovery=True) == 2


def _session_project(root: Path) -> tuple[Path, Path]:
    """Copy the real session hook into a small independent pytest project.

    Like the repository, the project root holds an ``s2and_rust/`` source
    directory without ``__init__.py``; the package imports from ``installed/``.
    """
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


def test_session_rejects_source_edit_before_collection(tmp_path: Path) -> None:
    """Exercise the real hook and import path, without compiling a fake crate."""
    source_root, native = _session_project(tmp_path)
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


def test_session_rejects_extension_without_source_digest(tmp_path: Path) -> None:
    _, native = _session_project(tmp_path)
    native.write_text("def get_build_info():\n    return {'crate_version': '1.0.0'}\n", encoding="utf-8")

    result = _run_session(tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert "embedded=None" in output
    assert str(native) in output
    assert REBUILD_COMMAND in output


def test_session_explains_missing_native_module(tmp_path: Path) -> None:
    _session_project(tmp_path)

    result = _run_session(tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert "Cannot import the s2and_rust extension" in output
    assert REBUILD_COMMAND in output


def test_uninstalled_extension_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without an install, the checkout's source directory imports as a namespace package."""
    source_root = _write_tree(tmp_path / "s2and_rust", _TREE)
    spec = importlib.machinery.PathFinder.find_spec("s2and_rust", [str(tmp_path)])
    assert spec is not None and spec.loader is None
    monkeypatch.setitem(sys.modules, "s2and_rust", importlib.util.module_from_spec(spec))

    message = stale_rust_extension_message(source_root)

    assert message is not None
    assert "not the compiled extension" in message
    assert str(source_root) in message
    assert REBUILD_COMMAND in message


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
