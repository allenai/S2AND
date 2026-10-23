#!/usr/bin/env python3
"""
Sync versions from VERSION into package manifests and lockfiles.

Usage:
  uv run python scripts/sync_version.py
  uv run python scripts/sync_version.py --check
  uv run python scripts/sync_version.py --print-targets
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


@dataclass(frozen=True)
class VersionTarget:
    name: str
    relative_path: Path
    pattern: str
    # Cargo requires SemVer pre-release spelling (1.0.0-beta.1); Python files use PEP 440 (1.0.0b1).
    cargo: bool = False

    def path(self, root: Path) -> Path:
        return root / self.relative_path

    def spelled(self, version: str) -> str:
        return cargo_version(version) if self.cargo else version


PEP440_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?"
CARGO_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:-(?:alpha|beta|rc)\.[0-9]+)?"
_CARGO_PRERELEASE_TAGS = {"a": "alpha", "b": "beta", "rc": "rc"}


def cargo_version(version: str) -> str:
    """Spell a PEP 440 release or pre-release the way Cargo requires, e.g. 1.0.0b1 -> 1.0.0-beta.1."""
    match = re.fullmatch(r"([0-9]+\.[0-9]+\.[0-9]+)(?:(a|b|rc)([0-9]+))?", version)
    if match is None:
        raise SystemExit(f"Unsupported version for Cargo: {version}")
    base, tag, number = match.groups()
    return base if tag is None else f"{base}-{_CARGO_PRERELEASE_TAGS[tag]}.{number}"


def version_targets() -> tuple[VersionTarget, ...]:
    return (
        VersionTarget(
            name="pyproject",
            relative_path=Path("pyproject.toml"),
            pattern=rf'(?m)^(?P<prefix>version = ")(?P<version>{PEP440_PATTERN})(?P<suffix>"\s*)$',
        ),
        VersionTarget(
            name="cargo_toml",
            relative_path=Path("s2and_rust") / "Cargo.toml",
            pattern=rf'(?m)^(?P<prefix>version = ")(?P<version>{CARGO_PATTERN})(?P<suffix>"\s*)$',
            cargo=True,
        ),
        VersionTarget(
            name="cargo_lock",
            relative_path=Path("s2and_rust") / "Cargo.lock",
            pattern=(
                rf'(?m)^(?P<prefix>\[\[package\]\]\r?\nname = "s2and_rust"\r?\nversion = ")'
                rf'(?P<version>{CARGO_PATTERN})(?P<suffix>")'
            ),
            cargo=True,
        ),
        VersionTarget(
            name="uv_lock",
            relative_path=Path("uv.lock"),
            pattern=(
                rf'(?m)^(?P<prefix>\[\[package\]\]\r?\nname = "s2and"\r?\nversion = ")'
                rf'(?P<version>{PEP440_PATTERN})(?P<suffix>")'
            ),
        ),
    )


def read_version(root: Path = ROOT) -> str:
    version_file = root / "VERSION"
    if not version_file.exists():
        raise SystemExit(f"VERSION file not found: {version_file}")
    version = _read_text(version_file).strip()
    if not re.fullmatch(PEP440_PATTERN, version):
        raise SystemExit(f"VERSION must be X.Y.Z or a PEP 440 pre-release such as X.Y.Zb1. Got: {version}")
    return version


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _single_match(path: Path, pattern: str) -> tuple[str, re.Match[str]]:
    text = _read_text(path)
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise SystemExit(f"Expected one version match in {path} for pattern: {pattern}; found {len(matches)}")
    return text, matches[0]


def _version_from_match(match: re.Match[str]) -> str:
    groups = match.groupdict()
    version = groups.get("version")
    if version is None:
        raise SystemExit("Internal error: version pattern has no version group")
    return version


def _replacement_from_match(match: re.Match[str], version: str) -> str:
    groups = match.groupdict()
    return f"{groups['prefix']}{version}{groups['suffix']}"


def sync_target(root: Path, target: VersionTarget, version: str) -> None:
    path = target.path(root)
    text, match = _single_match(path, target.pattern)
    replacement = _replacement_from_match(match, target.spelled(version))
    new_text = text[: match.start()] + replacement + text[match.end() :]
    if new_text != text:
        _write_text(path, new_text)


def check_target(root: Path, target: VersionTarget, expected: str) -> None:
    path = target.path(root)
    _, match = _single_match(path, target.pattern)
    found = _version_from_match(match)
    expected = target.spelled(expected)
    if found != expected:
        raise SystemExit(f"Version mismatch in {path} ({target.name}): found {found}, expected {expected}")


def sync_version(version: str, root: Path = ROOT) -> None:
    for target in version_targets():
        sync_target(root, target, version)


def verify_version(version: str, root: Path = ROOT) -> None:
    for target in version_targets():
        check_target(root, target, version)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Only verify versions match VERSION.")
    mode.add_argument(
        "--print-targets",
        action="store_true",
        help="Print repository-relative files updated or regenerated by version sync.",
    )
    args = parser.parse_args()

    if args.print_targets:
        print("VERSION")
        for target in version_targets():
            print(target.relative_path.as_posix())
        return

    version = read_version()
    if args.check:
        verify_version(version)
        print(f"OK: package versions match {version}")
        return
    sync_version(version)
    verify_version(version)
    print(f"Updated package versions to {version}")
    print("Both lockfiles are updated. Next: run `uv sync --extra dev` to rebuild the environment.")


if __name__ == "__main__":
    main()
