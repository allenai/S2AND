"""Verify Python distributions against the fixed release asset contract.

Every declared package-data file must appear byte-for-byte in both the wheel and
sdist. Canonical tuple and ORCID assets are always required. Packaged default
model declarations and ``production_model_v*`` paths are always forbidden
because the production model is an external release artifact.

The ``s2and`` package inventory must be exact. Maturin packs every file under
``s2and/`` that Git does not ignore, so each archive may hold only the source
modules, the declared data files, and, in a wheel, one native extension.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

CANONICAL_TUPLE_PATHS = frozenset(
    {
        "s2and/data/s2and_name_tuples_canonical.txt",
    }
)
CANONICAL_ORCID_PATHS = frozenset(
    {
        "s2and/data/first_k_letter_counts_from_orcid.json",
        "s2and/data/first_k_letter_counts_from_orcid.manifest.json",
    }
)
DEFAULT_PRODUCTION_MODEL_PATH = "s2and/data/default_production_model.json"
REQUIRED_RUNTIME_PATHS = CANONICAL_TUPLE_PATHS | CANONICAL_ORCID_PATHS
# Maturin names the module with a platform tag, e.g. _s2and_rust.cp313-win_amd64.pyd.
NATIVE_EXTENSION_PATTERN = re.compile(r"s2and/_s2and_rust\.[^/]+\.(?:pyd|so)")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _declared_package_data_paths(source_root: Path) -> set[str]:
    config_path = source_root / "pyproject.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    patterns = [
        entry["path"].removeprefix("s2and/")
        for entry in config["tool"]["maturin"]["include"]
        if entry["path"].startswith("s2and/")
    ]
    package_root = source_root / "s2and"
    expected: set[str] = set()
    missing: list[str] = []
    for pattern in patterns:
        matches = sorted(path for path in package_root.glob(str(pattern)) if path.is_file())
        if not matches:
            missing.append(str(pattern))
        expected.update(path.relative_to(source_root).as_posix() for path in matches)
    if missing:
        raise FileNotFoundError(f"Declared package-data patterns match no source files: {missing}")
    return expected


def _load_expected_paths(source_root: Path) -> set[str]:
    expected = _declared_package_data_paths(source_root)
    missing_runtime = sorted(REQUIRED_RUNTIME_PATHS - expected)
    if missing_runtime:
        raise ValueError(
            f"Release distributions require these paths in package-data and the source tree: {missing_runtime}"
        )
    return expected


def _source_module_paths(source_root: Path) -> set[str]:
    package_root = source_root / "s2and"
    return {path.relative_to(source_root).as_posix() for path in package_root.rglob("*.py") if path.is_file()}


def _inventory_errors(
    *,
    archive_name: str,
    package_paths: set[str],
    allowed: set[str],
    native_extensions: int,
) -> list[str]:
    """Describe unexpected package files and a wrong native extension count."""
    errors: list[str] = []
    unexpected = sorted(package_paths - allowed)
    if unexpected:
        errors.append(f"{archive_name} contains unexpected package files: {unexpected}")
    missing_modules = sorted(path for path in allowed - package_paths if path.endswith(".py"))
    if missing_modules:
        errors.append(f"{archive_name} is missing source modules: {missing_modules}")
    native = sorted(path for path in package_paths if NATIVE_EXTENSION_PATTERN.fullmatch(path))
    if len(native) != native_extensions:
        errors.append(f"{archive_name} must contain {native_extensions} native extension(s); found {native}")
    return errors


def _forbidden_model_paths(paths: Iterable[str]) -> set[str]:
    prefix = "s2and/data/"
    forbidden: set[str] = set()
    for path in paths:
        if path == DEFAULT_PRODUCTION_MODEL_PATH:
            forbidden.add(path)
            continue
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        top_level = relative.split("/", 1)[0]
        if top_level.startswith("production_model_v"):
            forbidden.add(path)
    return forbidden


def _verify_archive(
    *,
    archive_name: str,
    paths: set[str],
    read_bytes: Callable[[str], bytes],
    expected: set[str],
    source_root: Path,
) -> None:
    missing = sorted(expected - paths)
    if missing:
        raise ValueError(f"{archive_name} missing required distribution files: {missing}")

    forbidden = sorted(_forbidden_model_paths(paths))
    if forbidden:
        raise ValueError(f"{archive_name} contains forbidden production model paths: {forbidden}")

    mismatched: list[str] = []
    for path in sorted(expected):
        archive_payload = read_bytes(path)
        source_payload = (source_root / path).read_bytes()
        if _sha256_bytes(archive_payload) != _sha256_bytes(source_payload):
            mismatched.append(path)
    if mismatched:
        raise ValueError(f"{archive_name} content differs from the source tree for: {mismatched}")


def _sdist_package_path(member_name: str, *, archive_root: str) -> str | None:
    """Return a package path only from the sdist's canonical package directory."""

    parts = PurePosixPath(member_name).parts
    if len(parts) < 2 or parts[0] != archive_root or parts[1] != "s2and":
        return None
    return PurePosixPath(*parts[1:]).as_posix()


def verify_production_model_distributions(*, dist_dir: Path, source_root: Path) -> None:
    """Validate every release wheel and source archive under ``dist_dir``."""

    source_root = Path(source_root)
    expected = _load_expected_paths(source_root)
    package_files = expected | _source_module_paths(source_root)
    wheels = sorted(Path(dist_dir).glob("s2and-*.whl"))
    if not wheels and not list(Path(dist_dir).glob("s2and-*.tar.gz")):
        raise ValueError("expected at least one s2and distribution")
    for wheel_path in wheels:
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel_file_names = [name for name in wheel.namelist() if not name.endswith("/")]
            wheel_paths = set(wheel_file_names)
            if len(wheel_paths) != len(wheel_file_names):
                raise ValueError(f"{wheel_path.name} contains duplicate package file paths")
            _verify_archive(
                archive_name=wheel_path.name,
                paths=wheel_paths,
                read_bytes=wheel.read,
                expected=expected,
                source_root=source_root,
            )
            dist_info = {path for path in wheel_paths if path.split("/", 1)[0].endswith(".dist-info")}
            native = {path for path in wheel_paths if NATIVE_EXTENSION_PATTERN.fullmatch(path)}
            errors = _inventory_errors(
                archive_name=wheel_path.name,
                package_paths=wheel_paths - dist_info,
                allowed=package_files | native,
                native_extensions=1,
            )
            if errors:
                raise ValueError("; ".join(errors))

    sdists = sorted(Path(dist_dir).glob("s2and-*.tar.gz"))
    for sdist_path in sdists:
        with tarfile.open(sdist_path, "r:gz") as sdist:
            archive_root = sdist_path.name.removesuffix(".tar.gz")
            package_members: dict[str, tarfile.TarInfo] = {}
            for member in sdist.getmembers():
                if not member.isfile():
                    continue
                package_path = _sdist_package_path(member.name, archive_root=archive_root)
                if package_path is None:
                    continue
                if package_path in package_members:
                    raise ValueError(f"{sdist_path.name} contains duplicate package path: {package_path}")
                package_members[package_path] = member

            def read_sdist_bytes(
                path: str,
                members: dict[str, tarfile.TarInfo] = package_members,
                archive_name: str = sdist_path.name,
            ) -> bytes:
                extracted = sdist.extractfile(members[path])
                if extracted is None:
                    raise ValueError(f"{archive_name} package file is not readable: {path}")
                return extracted.read()

            _verify_archive(
                archive_name=sdist_path.name,
                paths=set(package_members),
                read_bytes=read_sdist_bytes,
                expected=expected,
                source_root=source_root,
            )
            errors = _inventory_errors(
                archive_name=sdist_path.name,
                package_paths=set(package_members),
                allowed=package_files,
                native_extensions=0,
            )
            if errors:
                raise ValueError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    args = parser.parse_args()
    verify_production_model_distributions(dist_dir=args.dist_dir, source_root=args.source_root)
    print("Verified release inventory and content digests in the distributions.")


if __name__ == "__main__":
    main()
