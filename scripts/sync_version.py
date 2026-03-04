#!/usr/bin/env python3
"""
Sync versions from VERSION into package manifests.

Usage:
  uv run python scripts/sync_version.py
  uv run python scripts/sync_version.py --check
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"

FILES = {
    "pyproject_rust_extra": ROOT / "pyproject.toml",
    "rust_pyproject": ROOT / "s2and_rust" / "pyproject.toml",
    "cargo_toml": ROOT / "s2and_rust" / "Cargo.toml",
}

PATTERNS = {
    "pyproject_rust_extra": r'(?m)^\s+"s2and-rust>=([0-9]+\.[0-9]+\.[0-9]+)",\s*$',
    "rust_pyproject": r'(?m)^version = "([0-9]+\.[0-9]+\.[0-9]+)"\s*$',
    "cargo_toml": r'(?m)^version = "([0-9]+\.[0-9]+\.[0-9]+)"\s*$',
}


def read_version() -> str:
    if not VERSION_FILE.exists():
        raise SystemExit(f"VERSION file not found: {VERSION_FILE}")
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        raise SystemExit(f"VERSION must be semver (X.Y.Z). Got: {version}")
    return version


def replace_once(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Expected one match in {path} for pattern: {pattern}")
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")


def check_version(path: Path, pattern: str, expected: str) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if not match:
        raise SystemExit(f"Missing version in {path} for pattern: {pattern}")
    found = match.group(1)
    if found != expected:
        raise SystemExit(f"Version mismatch in {path}: found {found}, expected {expected}")


def sync_version(version: str) -> None:
    replace_once(
        FILES["pyproject_rust_extra"],
        PATTERNS["pyproject_rust_extra"],
        f'  "s2and-rust>={version}",',
    )
    replace_once(
        FILES["rust_pyproject"],
        PATTERNS["rust_pyproject"],
        f'version = "{version}"',
    )
    replace_once(
        FILES["cargo_toml"],
        PATTERNS["cargo_toml"],
        f'version = "{version}"',
    )


def verify_version(version: str) -> None:
    check_version(FILES["pyproject_rust_extra"], PATTERNS["pyproject_rust_extra"], version)
    check_version(FILES["rust_pyproject"], PATTERNS["rust_pyproject"], version)
    check_version(FILES["cargo_toml"], PATTERNS["cargo_toml"], version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify versions match VERSION.")
    args = parser.parse_args()

    version = read_version()
    if args.check:
        verify_version(version)
        print(f"OK: versions match {version}")
        return

    sync_version(version)
    verify_version(version)
    print(f"Updated versions to {version}")
    print(
        "Next: run `uv sync --extra dev` and "
        "`uv run --active --no-project cargo generate-lockfile --manifest-path s2and_rust/Cargo.toml`."
    )


if __name__ == "__main__":
    main()
