"""Verify that Python distributions contain the required runtime and model assets.

Two properties are checked, and they are deliberately separate:

1. **Inventory** — every file declared in ``[tool.setuptools.package-data]``, plus
   any phase-permitted default production bundle, is present in both the wheel
   and the sdist, and no undeclared ``production_model_v*`` asset has leaked in.
   Deriving the baseline from ``pyproject.toml`` keeps this check from drifting
   away from what setuptools actually ships.

2. **Content** — every expected member's archive bytes SHA-256 match the source
   file. Inventory alone is fail-open: an archive can carry a stale or truncated
   copy of a declared path and still list it.

Inventory derived purely from ``pyproject.toml`` cannot notice a *removed
declaration*, because the expectation disappears along with the requirement.
That matters here because the canonical ORCID artifacts are intentionally
undeclared during the code-only phase and are declared in one Stage 1 promotion
commit. ``--phase`` therefore names the release phase whose runtime-artifact
contract applies, and each phase asserts both required and forbidden paths. It
is required rather than defaulted: a permissive default is exactly the mistake
this guard exists to catch. The release-candidate phase also forbids packaged
default-model declarations and ``production_model_v*`` directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# phase -> (paths that must be declared and shipped, paths that must not be)
PHASE_CONTRACTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "code_only": (CANONICAL_TUPLE_PATHS, CANONICAL_ORCID_PATHS),
    "release_candidate": (
        CANONICAL_TUPLE_PATHS | CANONICAL_ORCID_PATHS,
        frozenset({DEFAULT_PRODUCTION_MODEL_PATH}),
    ),
}


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


def _apply_phase_contract(expected: set[str], *, phase: str, source_root: Path) -> None:
    """Assert the phase's required and forbidden runtime artifacts."""

    try:
        required, forbidden = PHASE_CONTRACTS[phase]
    except KeyError:
        raise ValueError(f"Unknown release phase {phase!r}; expected one of {sorted(PHASE_CONTRACTS)}") from None

    undeclared = sorted(required - expected)
    if undeclared:
        raise ValueError(
            f"Release phase {phase!r} requires these paths to be declared in "
            f"pyproject.toml package-data and present in the source tree: {undeclared}"
        )

    declared_forbidden = sorted(forbidden & expected)
    if declared_forbidden:
        raise ValueError(f"Release phase {phase!r} forbids declaring these paths: {declared_forbidden}")

    present_forbidden = sorted(path for path in forbidden if (source_root / path).is_file())
    if present_forbidden:
        raise ValueError(
            f"Release phase {phase!r} forbids these source files; promote them together "
            f"with their package-data declarations: {present_forbidden}"
        )


def _load_expected_paths(source_root: Path, *, phase: str) -> set[str]:
    expected = _declared_package_data_paths(source_root)
    _apply_phase_contract(expected, phase=phase, source_root=source_root)
    declaration_path = source_root / DEFAULT_PRODUCTION_MODEL_PATH
    if declaration_path.is_file():
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        bundle_dir_name = str(declaration["bundle_dir"])
        bundle_version = str(declaration["bundle_version"])
        if bundle_dir_name != f"production_model_v{bundle_version}":
            raise ValueError("Default production model declaration bundle_dir and bundle_version disagree")
        default_dir = declaration_path.parent / bundle_dir_name
        if not default_dir.is_dir():
            raise FileNotFoundError(f"Declared default production bundle is missing: {default_dir}")
        expected |= {
            DEFAULT_PRODUCTION_MODEL_PATH,
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
    read_bytes: Callable[[str], bytes],
    expected: set[str],
    phase: str,
    source_root: Path,
) -> None:
    missing = sorted(expected - paths)
    if missing:
        raise ValueError(f"{archive_name} missing required distribution files: {missing}")

    undeclared = sorted(_production_model_asset_paths(paths) - expected)
    if undeclared:
        raise ValueError(f"{archive_name} contains undeclared production model assets: {undeclared}")

    _, forbidden = PHASE_CONTRACTS[phase]
    shipped_forbidden = sorted(forbidden & paths)
    if shipped_forbidden:
        raise ValueError(f"{archive_name} ships paths forbidden in release phase {phase!r}: {shipped_forbidden}")

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


def verify_production_model_distributions(*, dist_dir: Path, source_root: Path, phase: str) -> None:
    """Validate the single wheel and sdist under ``dist_dir`` for one release phase."""

    source_root = Path(source_root)
    expected = _load_expected_paths(source_root, phase=phase)
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
            phase=phase,
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
                raise ValueError(f"{sdists[0].name} production model file is not readable: {path}")
            return extracted.read()

        _verify_archive(
            archive_name=sdists[0].name,
            paths=set(package_members),
            read_bytes=read_sdist_bytes,
            expected=expected,
            phase=phase,
            source_root=source_root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--phase",
        required=True,
        choices=sorted(PHASE_CONTRACTS),
        help="Release phase whose runtime-artifact contract applies.",
    )
    args = parser.parse_args()
    verify_production_model_distributions(dist_dir=args.dist_dir, source_root=args.source_root, phase=args.phase)
    print(f"Verified {args.phase} inventory, content digests, and model selection in the wheel and sdist.")


if __name__ == "__main__":
    main()
