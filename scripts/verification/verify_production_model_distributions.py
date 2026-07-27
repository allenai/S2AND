"""Verify Python distributions against the fixed release asset contract.

Every declared package-data file must appear byte-for-byte in both the wheel and
sdist. Canonical tuple and ORCID assets are always required. Packaged default
model declarations and ``production_model_v*`` paths are always forbidden
because the production model is an external release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

CANONICAL_TUPLE_PATHS = frozenset(
    {
        "s2and/data/s2and_name_tuples_canonical.txt",
        "s2and/data/s2and_name_tuples_canonical.txt.meta.json",
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _declared_package_data_paths(source_root: Path) -> set[str]:
    config_path = source_root / "pyproject.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    patterns = config["tool"]["setuptools"]["package-data"]["s2and"]
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

    unhydrated: list[str] = []
    unhydrated_source: list[str] = []
    mismatched: list[str] = []
    for path in sorted(expected):
        archive_payload = read_bytes(path)
        source_payload = (source_root / path).read_bytes()
        if archive_payload[: len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX:
            unhydrated.append(path)
            continue
        if source_payload[: len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX:
            unhydrated_source.append(path)
            continue
        if _sha256_bytes(archive_payload) != _sha256_bytes(source_payload):
            mismatched.append(path)
    if unhydrated:
        raise ValueError(f"{archive_name} contains unhydrated LFS pointers: {unhydrated}")
    if unhydrated_source:
        # Distinguished from content drift: the archive is fine and the checkout
        # is the problem, so the operator needs `git lfs pull`, not a rebuild.
        raise ValueError(
            f"Source tree contains unhydrated LFS pointers, so {archive_name} content "
            f"cannot be verified against it: {unhydrated_source}"
        )
    if mismatched:
        raise ValueError(f"{archive_name} content differs from the source tree for: {mismatched}")


def _sdist_package_path(member_name: str, *, archive_root: str) -> str | None:
    """Return a package path only from the sdist's canonical package directory."""

    parts = PurePosixPath(member_name).parts
    if len(parts) < 2 or parts[0] != archive_root or parts[1] != "s2and":
        return None
    return PurePosixPath(*parts[1:]).as_posix()


def verify_production_model_distributions(*, dist_dir: Path, source_root: Path) -> None:
    """Validate the single release wheel and sdist under ``dist_dir``."""

    source_root = Path(source_root)
    expected = _load_expected_paths(source_root)
    wheels = sorted(Path(dist_dir).glob("s2and-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected exactly one s2and wheel, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_file_names = [name for name in wheel.namelist() if not name.endswith("/")]
        wheel_paths = set(wheel_file_names)
        if len(wheel_paths) != len(wheel_file_names):
            raise ValueError(f"{wheels[0].name} contains duplicate package file paths")
        _verify_archive(
            archive_name=wheels[0].name,
            paths=wheel_paths,
            read_bytes=wheel.read,
            expected=expected,
            source_root=source_root,
        )

    sdists = sorted(Path(dist_dir).glob("s2and-*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError(f"expected exactly one s2and sdist, found {len(sdists)}")
    with tarfile.open(sdists[0], "r:gz") as sdist:
        archive_root = sdists[0].name.removesuffix(".tar.gz")
        package_members: dict[str, tarfile.TarInfo] = {}
        for member in sdist.getmembers():
            if not member.isfile():
                continue
            package_path = _sdist_package_path(member.name, archive_root=archive_root)
            if package_path is None:
                continue
            if package_path in package_members:
                raise ValueError(f"{sdists[0].name} contains duplicate package path: {package_path}")
            package_members[package_path] = member

        def read_sdist_bytes(path: str) -> bytes:
            extracted = sdist.extractfile(package_members[path])
            if extracted is None:
                raise ValueError(f"{sdists[0].name} package file is not readable: {path}")
            return extracted.read()

        _verify_archive(
            archive_name=sdists[0].name,
            paths=set(package_members),
            read_bytes=read_sdist_bytes,
            expected=expected,
            source_root=source_root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    args = parser.parse_args()
    verify_production_model_distributions(dist_dir=args.dist_dir, source_root=args.source_root)
    print("Verified release inventory and content digests in the wheel and sdist.")


if __name__ == "__main__":
    main()
