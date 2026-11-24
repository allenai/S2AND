"""Source digest and package-cache checks that reject stale native extensions before collection."""

from __future__ import annotations

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
from types import ModuleType

import pytest

from tests.rust_build_freshness import (
    CLEAN_COMMAND,
    REBUILD_COMMAND,
    duplicate_package_metadata_message,
    rust_source_digest,
    stale_rust_extension_message,
)

pytestmark = pytest.mark.tooling

_TREE = {
    "Cargo.toml": b"[package]\n",
    "Cargo.lock": b"# lock\n",
    "build.rs": b"fn main() {}\n",
    "src/lib.rs": b"pub fn f() {}\n",
    "src/nested/mod.rs": b"pub fn g() {}\n",
    "vendor/cld2/Cargo.toml": b"[package]\n",
    "vendor/cld2/src/lib.rs": b"pub fn h() {}\n",
    "vendor/cld2-sys/Cargo.toml": b"[package]\n",
    "vendor/cld2-sys/build.rs": b"fn main() {}\n",
    "vendor/cld2-sys/src/lib.rs": b"pub fn ffi() {}\n",
    "vendor/cld2-sys/cld2/internal/detect.cc": b"int detect() { return 0; }\n",
}


def _write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _link_directory(link: Path, target: Path) -> None:
    """Exercise Windows junctions without requiring symlink privileges."""
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def _unsupported_source_path(root: Path) -> Path:
    name = "\udcff.rs" if os.name == "nt" else os.fsdecode(b"\xff.rs")
    return root / "src" / name


def _install_native_stub(monkeypatch: pytest.MonkeyPatch, native_path: Path, build_info: dict[str, str]) -> None:
    native = ModuleType("s2and._s2and_rust")
    native.__file__ = str(native_path)
    monkeypatch.setattr(native, "get_build_info", lambda: build_info, raising=False)
    monkeypatch.setitem(sys.modules, "s2and._s2and_rust", native)


def test_source_digest_invariants_and_all_build_input_mutations(tmp_path: Path, subtests: pytest.Subtests) -> None:
    forward = _write_tree(tmp_path / "forward", _TREE)
    reverse = _write_tree(tmp_path / "reverse", dict(reversed(list(_TREE.items()))))
    baseline = rust_source_digest(forward)
    assert baseline == rust_source_digest(reverse)
    assert baseline.startswith("v1:")

    # Ignored outputs, editor swap files, and standalone vendor lockfiles do not affect the digest.
    _write_tree(
        reverse,
        {
            "target/release/lib.dll": b"binary",
            "s2and_rust/__init__.py": b"",
            "vendor/cld2/Cargo.lock": b"standalone vendor resolution",
            "vendor/cld2/target/generated.rs": b"pub fn unused_output() {}",
            "vendor/cld2/README.md": b"documentation",
            "src/lib.rs.orig": b"editor backup",
            "src/.swp": b"editor state",
        },
    )
    assert rust_source_digest(reverse) == baseline

    older_ns = 1_600_000_000_000_000_000
    newer_ns = older_ns + 100_000_000_000
    for change in (
        "modify_source",
        "modify_manifest",
        "modify_lockfile",
        "modify_build_script",
        "add_source",
        "delete_source",
        "rename_source",
        "modify_vendor",
        "modify_vendor_manifest",
        "modify_vendor_cpp",
        "modify_vendor_build",
    ):
        with subtests.test(change=change):
            root = _write_tree(tmp_path / change, _TREE)
            if change in {"modify_source", "add_source", "delete_source"}:
                for path in (root / "src").rglob("*.rs"):
                    os.utime(path, ns=(older_ns, older_ns))
                os.utime(root / "src/lib.rs", ns=(newer_ns, newer_ns))
            if change == "modify_source":
                (root / "src/lib.rs").write_bytes(b"pub fn f() { }\n")
                os.utime(root / "src/lib.rs", ns=(newer_ns, newer_ns))
            elif change == "modify_manifest":
                (root / "Cargo.toml").write_bytes(b"[package]\nname = 'x'\n")
            elif change == "modify_lockfile":
                (root / "Cargo.lock").write_bytes(b"# updated lock\n")
            elif change == "modify_build_script":
                (root / "build.rs").write_bytes(b"fn main() { }\n")
            elif change == "add_source":
                (root / "src/new.rs").write_bytes(b"")
                os.utime(root / "src/new.rs", ns=(older_ns, older_ns))
            elif change == "delete_source":
                (root / "src/nested/mod.rs").unlink()
            elif change == "rename_source":
                (root / "src/nested/mod.rs").rename(root / "src/nested/other.rs")
            elif change == "modify_vendor":
                (root / "vendor/cld2/src/lib.rs").write_bytes(b"pub fn h() { }\n")
            elif change == "modify_vendor_cpp":
                (root / "vendor/cld2-sys/cld2/internal/detect.cc").write_bytes(b"int detect() { return 1; }\n")
            elif change == "modify_vendor_build":
                (root / "vendor/cld2-sys/build.rs").write_bytes(b"fn main() { }\n")
            else:
                (root / "vendor/cld2/Cargo.toml").write_bytes(b"[package]\nname = 'cld2'\n")
            assert rust_source_digest(root) != baseline, change


def test_freshness_and_duplicate_metadata_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subtests: pytest.Subtests
) -> None:
    with subtests.test("linked-source-directory"):
        source = _write_tree(tmp_path / "linked_crate", _TREE)
        _install_native_stub(monkeypatch, tmp_path / "native.pyd", {"source_digest": rust_source_digest(source)})
        external = tmp_path / "external"
        external.mkdir()
        (external / "module.rs").write_bytes(b"pub fn linked() {}\n")
        _link_directory(source / "src/linked", external)
        with pytest.raises(ValueError, match="linked native source directories are unsupported") as error:
            rust_source_digest(source)
        message = stale_rust_extension_message(source)
        assert message == (
            f"Cannot verify s2and_rust source freshness: {error.value}. "
            "Correct the native source inputs before rebuilding."
        )
        assert "INTERNALERROR" not in message

    with subtests.test("invalid-utf8-source-name"):
        source = _write_tree(tmp_path / "utf8_crate", _TREE)
        _install_native_stub(monkeypatch, tmp_path / "native.pyd", {"source_digest": rust_source_digest(source)})
        _unsupported_source_path(source).write_bytes(b"// unused module\n")
        with pytest.raises(ValueError, match="native source paths must be valid UTF-8") as error:
            rust_source_digest(source)
        message = stale_rust_extension_message(source)
        assert message == (
            f"Cannot verify s2and_rust source freshness: {error.value}. "
            "Correct the native source inputs before rebuilding."
        )

    with subtests.test("missing-digest-and-missing-or-uninstalled-module"):
        source = _write_tree(tmp_path / "missing_digest", _TREE)
        native = tmp_path / "installed" / "_s2and_rust.pyd"
        _install_native_stub(monkeypatch, native, {"crate_version": "1.0.0"})
        message = stale_rust_extension_message(source)
        assert message is not None
        assert "different sources" in message and str(native) in message
        assert REBUILD_COMMAND in message and CLEAN_COMMAND in message

        monkeypatch.setitem(sys.modules, "s2and._s2and_rust", None)
        missing_msg = stale_rust_extension_message(tmp_path)
        assert missing_msg is not None and "Cannot import the s2and_rust extension" in missing_msg
        assert REBUILD_COMMAND in missing_msg

        ns_parent = tmp_path / "ns_parent"
        ns_source = _write_tree(ns_parent / "s2and_rust", _TREE)
        spec = importlib.machinery.PathFinder.find_spec("s2and_rust", [str(ns_parent)])
        assert spec is not None and spec.loader is None
        monkeypatch.setitem(sys.modules, "s2and._s2and_rust", importlib.util.module_from_spec(spec))
        ns_msg = stale_rust_extension_message(ns_source)
        assert ns_msg is not None and "not the compiled extension" in ns_msg and REBUILD_COMMAND in ns_msg

    with subtests.test("duplicate-metadata-stale-egg-info"):

        def _write_metadata(root: Path, directory: str, version: str) -> Path:
            metadata = root / directory
            metadata.mkdir(parents=True)
            name = "PKG-INFO" if directory.endswith(".egg-info") else "METADATA"
            (metadata / name).write_text(f"Metadata-Version: 2.1\nName: s2and\nVersion: {version}\n", encoding="utf-8")
            return root

        site_packages = _write_metadata(tmp_path / "site-packages", "s2and-1.0.0b6.dist-info", "1.0.0b6")
        checkout = _write_metadata(tmp_path / "checkout", "s2and.egg-info", "1.0.0b4")
        assert duplicate_package_metadata_message([str(site_packages)]) is None
        assert duplicate_package_metadata_message([str(site_packages), str(site_packages / ".")]) is None
        dup_msg = duplicate_package_metadata_message([str(checkout), str(site_packages)])
        assert (
            dup_msg is not None and "1.0.0b4 in" in dup_msg and "1.0.0b6 in" in dup_msg and "s2and.egg-info" in dup_msg
        )


def _uv_cache_project(root: Path) -> tuple[Path, Path, list[Path]]:
    """Use the native package's real cache keys and license inventory with an offline wheel builder."""
    source = _write_tree(root / "package", _TREE)
    native_manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
    parsed = tomllib.loads(native_manifest.read_text(encoding="utf-8"))
    keys = parsed["tool"]["uv"]["cache-keys"]
    license_files = parsed["project"]["license-files"]
    cache_key_set = {key["file"] for key in keys}
    assert set(license_files) <= cache_key_set, "Every packaged license-file must appear in [tool.uv].cache-keys"

    cache_keys = ", ".join("{ file = " + json.dumps(key["file"].removeprefix("s2and_rust/")) + " }" for key in keys)
    (source / "pyproject.toml").write_text(
        '[project]\nname = "s2and"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        f"[tool.uv]\ncache-keys = [{cache_keys}]\n",
        encoding="utf-8",
    )
    (source / "backend.py").write_text(
        "from pathlib import Path\n"
        "from zipfile import ZipFile\n\n"
        "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
        "    counter = Path(__file__).resolve().parent / 'build_count.txt'\n"
        "    count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "    counter.write_text(str(count))\n"
        "    name = 's2and-0.0.0-py3-none-any.whl'\n"
        "    dist = 's2and-0.0.0.dist-info'\n"
        "    with ZipFile(Path(wheel_directory) / name, 'w') as wheel:\n"
        "        wheel.writestr('freshness_probe.py', f'BUILD_COUNT = {count}\\n')\n"
        "        wheel.writestr(dist + '/METADATA', 'Metadata-Version: 2.1\\nName: s2and\\nVersion: 0.0.0\\n')\n"
        "        wheel.writestr(dist + '/WHEEL', 'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
        "        wheel.writestr(dist + '/RECORD', '')\n"
        "    return name\n",
        encoding="utf-8",
    )
    license_paths = [source / relative.removeprefix("s2and_rust/") for relative in license_files]
    for path in license_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Original notice\n", encoding="utf-8")

    project = root / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "freshness-probe-host"\nversion = "0.0.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["s2and"]\n[project.optional-dependencies]\ndev = []\n'
        '[tool.uv.sources]\ns2and = { path = "../package" }\n',
        encoding="utf-8",
    )
    return source, project, license_paths


def _sync_uv_cache_project(project: Path, *, recovery: bool = False) -> int:
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


def test_uv_cache_tracks_native_inputs_and_licenses_only(tmp_path: Path) -> None:
    """Verify uv cache hits, ignored-file immunity, source/vendor/license invalidation, and force-rebuild."""
    source, project, license_paths = _uv_cache_project(tmp_path)
    baseline = rust_source_digest(source)
    assert _sync_uv_cache_project(project) == 1, "initial build"
    assert _sync_uv_cache_project(project) == 1, "unchanged fixture must stay cached"

    ignored = [
        source / relative
        for relative in ("src/lib.rs.orig", "src/.swp", "vendor/cld2/Cargo.lock", "vendor/cld2/target/generated.rs")
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"irrelevant file\n")
    assert rust_source_digest(source) == baseline
    assert _sync_uv_cache_project(project) == 1, "ignored files must not bust cache"

    (source / "src/nested/mod.rs").write_bytes(b"pub fn g() { /* changed */ }\n")
    assert _sync_uv_cache_project(project) == 2, "nested Rust source edit must rebuild"

    (source / "vendor/cld2/src/lib.rs").write_bytes(b"pub fn h() { /* changed */ }\n")
    assert _sync_uv_cache_project(project) == 3, "vendored source edit must rebuild"

    license_paths[-1].write_text("Updated notice\n", encoding="utf-8")
    assert _sync_uv_cache_project(project) == 4, "packaged license edit must rebuild"
    assert _sync_uv_cache_project(project) == 4, "unchanged after license rebuild must stay cached"
    assert _sync_uv_cache_project(project, recovery=True) == 5, "recovery must rebuild a cached package"


def test_session_rejects_source_edit_before_collection(tmp_path: Path) -> None:
    """Exercise the real pytest sessionstart hook before collection."""
    source_root = _write_tree(tmp_path / "s2and_rust", _TREE)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("", encoding="utf-8")
    for name in ("conftest.py", "rust_build_freshness.py"):
        shutil.copyfile(Path(__file__).with_name(name), tests / name)
    (tests / "test_collection.py").write_text(
        "from pathlib import Path\nPath('collected').touch()\ndef test_collected():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    package = tmp_path / "installed" / "s2and"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("from ._s2and_rust import *\n", encoding="utf-8")
    native = package / "_s2and_rust.py"
    digest = rust_source_digest(source_root)
    native.write_text(f"def get_build_info():\n    return {{'source_digest': {digest!r}}}\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path / "installed")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-c", "pytest.ini", "tests"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    fresh = _run()
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert (tmp_path / "collected").exists()
    (tmp_path / "collected").unlink()

    (source_root / "src/lib.rs").write_bytes(b"pub fn changed() {}\n")
    stale = _run()
    output = stale.stdout + stale.stderr
    assert stale.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert not (tmp_path / "collected").exists()
    assert str(native) in output and REBUILD_COMMAND in output and CLEAN_COMMAND in output


def test_real_build_script_tracks_directories_and_current_package_root(tmp_path: Path) -> None:
    """Exercise Cargo invalidation, CLEAN_COMMAND recovery, and Rust/Python digest parity."""
    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("Cargo is required to exercise native build-script invalidation")
    source = _write_tree(
        tmp_path / "crate",
        {
            **{key: value for key, value in _TREE.items() if key.startswith("vendor/cld2-sys/")},
            "Cargo.toml": (
                b'[package]\nname = "s2and_freshness_probe"\nversion = "0.0.0"\nedition = "2021"\n'
                b'[dependencies]\ncld2 = { path = "vendor/cld2" }\n'
                b'cld2-sys = { path = "vendor/cld2-sys" }\n'
                b'[build-dependencies]\nsha2 = "0.10"\n'
            ),
            "src/main.rs": b'fn main() { println!("{} {}", env!("S2AND_RUST_SOURCE_DIGEST"), cld2::value()); }\n',
            "vendor/cld2/Cargo.toml": b'[package]\nname = "cld2"\nversion = "0.0.0"\nedition = "2021"\n',
            "vendor/cld2/src/lib.rs": b"pub fn value() -> u8 { 1 }\n",
            "vendor/cld2-sys/Cargo.toml": b'[package]\nname = "cld2-sys"\nversion = "0.0.0"\nedition = "2021"\n',
        },
    )
    shutil.copyfile(Path(__file__).resolve().parents[1] / "s2and_rust/build.rs", source / "build.rs")
    target = tmp_path / "target"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["PYO3_PYTHON"] = sys.executable
    command = [cargo, "run", "--release", "--offline", "--quiet", "--manifest-path", str(source / "Cargo.toml")]

    def assert_current_digest(expected_value: int = 1) -> str:
        result = subprocess.run(command, cwd=source, env=env, capture_output=True, text=True, check=False, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        digest = rust_source_digest(source)
        assert result.stdout.strip() == f"{digest} {expected_value}"
        return digest

    baseline = assert_current_digest()
    added = source / "vendor/cld2/src/new.rs"
    added.write_bytes(b"// Newly added, unreferenced build input\n")
    assert assert_current_digest() != baseline
    added.unlink()

    vendor_source = source / "vendor/cld2/src/lib.rs"
    original_times = vendor_source.stat()
    vendor_source.write_bytes(b"pub fn value() -> u8 { 2 }\n")
    os.utime(vendor_source, ns=(original_times.st_atime_ns, original_times.st_mtime_ns))
    assert rust_source_digest(source) != baseline
    main_source = source / "src/main.rs"
    main_source.write_bytes(main_source.read_bytes() + b"// Rebuild the root crate.\n")
    clean_command = shlex.split(CLEAN_COMMAND)[3:]
    clean_command[0] = cargo
    clean_command[clean_command.index("s2and_rust")] = "s2and_freshness_probe"
    clean_command[clean_command.index("s2and_rust/Cargo.toml")] = str(source / "Cargo.toml")
    cleaned = subprocess.run(
        clean_command, cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    assert_current_digest(expected_value=2)

    alternate = tmp_path / "alternate"
    shutil.copytree(source, alternate)
    (alternate / "src/new.rs").write_bytes(b"// Only in the current package\n")
    suffix = ".exe" if os.name == "nt" else ""
    build_root = target / "release/build"
    build_script = next(
        path
        for pattern in (
            f"s2and_freshness_probe-*/build-script-build{suffix}",
            f"s2and_freshness_probe/*/out/build_script_build{suffix}",
        )
        for path in build_root.glob(pattern)
    )
    env["CARGO_MANIFEST_DIR"] = str(alternate)
    env["OUT_DIR"] = str(target / "release/build/direct/out")
    result = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"cargo:rustc-env=S2AND_RUST_SOURCE_DIGEST={rust_source_digest(alternate)}" in result.stdout

    invalid_source = _unsupported_source_path(alternate)
    invalid_source.write_bytes(b"// unsupported path\n")
    invalid = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert invalid.returncode != 0 and "native source paths must be valid UTF-8" in invalid.stderr
    invalid_source.unlink()

    external = tmp_path / "external"
    external.mkdir()
    _link_directory(alternate / "src/linked", external)
    linked = subprocess.run(
        [str(build_script)], cwd=source, env=env, capture_output=True, text=True, check=False, timeout=30
    )
    assert linked.returncode != 0 and "linked native source directories are unsupported" in linked.stderr
