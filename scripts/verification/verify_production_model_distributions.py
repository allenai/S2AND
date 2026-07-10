"""Verify that Python distributions contain exactly the intended model assets."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
FORBIDDEN_LEGACY_RUNTIME_PATHS = frozenset(
    {
        "s2and/data/first_k_letter_counts_from_orcid.json",
        "s2and/data/first_k_letter_counts_from_orcid.meta.json",
    }
)


def _load_expected_paths(source_root: Path) -> set[str]:
    declaration_path = source_root / "s2and" / "data" / "default_production_model.json"
    if not declaration_path.is_file():
        return set()
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    bundle_dir_name = str(declaration["bundle_dir"])
    bundle_version = str(declaration["bundle_version"])
    if bundle_dir_name != f"production_model_v{bundle_version}":
        raise ValueError("Default production model declaration bundle_dir and bundle_version disagree")
    default_dir = declaration_path.parent / bundle_dir_name
    if not default_dir.is_dir():
        raise FileNotFoundError(f"Declared default production bundle is missing: {default_dir}")
    expected = {
        "s2and/data/default_production_model.json",
        *(path.relative_to(source_root).as_posix() for path in sorted(default_dir.rglob("*")) if path.is_file()),
    }
    return expected


def _production_model_asset_paths(paths: Iterable[str]) -> set[str]:
    prefix = "s2and/data/"
    assets: set[str] = set()
    for path in paths:
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        top_level = relative.split("/", 1)[0]
        if top_level.startswith("production_model_v"):
            assets.add(path)
    return assets


def _verify_archive(
    *,
    archive_name: str,
    paths: set[str],
    read_prefix: Callable[[str, int], bytes],
    expected: set[str],
) -> None:
    forbidden = sorted(paths.intersection(FORBIDDEN_LEGACY_RUNTIME_PATHS))
    if forbidden:
        raise ValueError(f"{archive_name} contains forbidden legacy runtime artifacts: {forbidden}")

    missing = sorted(expected - paths)
    if missing:
        raise ValueError(f"{archive_name} missing production model files: {missing}")

    undeclared = sorted(_production_model_asset_paths(paths) - expected)
    if undeclared:
        raise ValueError(f"{archive_name} contains undeclared production model assets: {undeclared}")

    unhydrated = sorted(path for path in expected if read_prefix(path, len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX)
    if unhydrated:
        raise ValueError(f"{archive_name} contains unhydrated LFS pointers: {unhydrated}")


def _sdist_package_path(member_name: str) -> str | None:
    parts = PurePosixPath(member_name).parts
    try:
        package_index = parts.index("s2and")
    except ValueError:
        return None
    return PurePosixPath(*parts[package_index:]).as_posix()


def verify_production_model_distributions(*, dist_dir: Path, source_root: Path) -> None:
    """Validate the single wheel and sdist under ``dist_dir``."""

    expected = _load_expected_paths(Path(source_root))
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
            read_prefix=lambda path, size: wheel.read(path)[:size],
            expected=expected,
        )

    sdists = sorted(Path(dist_dir).glob("s2and-*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError(f"expected exactly one s2and sdist, found {len(sdists)}")
    with tarfile.open(sdists[0], "r:gz") as sdist:
        package_members: dict[str, tarfile.TarInfo] = {}
        for member in sdist.getmembers():
            if not member.isfile():
                continue
            package_path = _sdist_package_path(member.name)
            if package_path is None:
                continue
            if package_path in package_members:
                raise ValueError(f"{sdists[0].name} contains duplicate package path: {package_path}")
            package_members[package_path] = member

        def read_sdist_prefix(path: str, size: int) -> bytes:
            extracted = sdist.extractfile(package_members[path])
            if extracted is None:
                raise ValueError(f"{sdists[0].name} production model file is not readable: {path}")
            return extracted.read(size)

        _verify_archive(
            archive_name=sdists[0].name,
            paths=set(package_members),
            read_prefix=read_sdist_prefix,
            expected=expected,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    args = parser.parse_args()
    verify_production_model_distributions(dist_dir=args.dist_dir, source_root=args.source_root)
    print("Verified hydrated declared production models in the wheel and sdist.")


if __name__ == "__main__":
    main()
