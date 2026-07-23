#!/usr/bin/env python3
"""
Sync versions from VERSION into package manifests and runtime guards.

Usage:
  uv run python scripts/sync_version.py
  uv run python scripts/sync_version.py --check
  uv run python scripts/sync_version.py --print-targets
  uv run python scripts/sync_version.py --release-policy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"


@dataclass(frozen=True)
class VersionTarget:
    name: str
    relative_path: Path
    pattern: str

    def path(self, root: Path) -> Path:
        return root / self.relative_path


@dataclass(frozen=True)
class ReleaseDecisions:
    """Final build, smoke, and publish decisions for one release event."""

    build_s2and: bool
    build_rust: bool
    run_release_smoke: bool
    publish_s2and: bool
    publish_rust: bool

    def github_outputs(self) -> dict[str, bool]:
        """Return the exact booleans consumed by the release workflow."""

        return {
            "build_s2and": self.build_s2and,
            "build_rust": self.build_rust,
            "run_release_smoke": self.run_release_smoke,
            "publish_s2and": self.publish_s2and,
            "publish_rust": self.publish_rust,
        }


SEMVER_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+"


def version_targets() -> tuple[VersionTarget, ...]:
    return (
        VersionTarget(
            name="pyproject_rust_extra",
            relative_path=Path("pyproject.toml"),
            pattern=rf'(?m)^(?P<indent>\s*)"s2and-rust(?:==|>=)(?P<version>{SEMVER_PATTERN})(?P<suffix>",\s*)$',
        ),
        VersionTarget(
            name="rust_pyproject",
            relative_path=Path("s2and_rust") / "pyproject.toml",
            pattern=rf'(?m)^(?P<prefix>version = ")(?P<version>{SEMVER_PATTERN})(?P<suffix>"\s*)$',
        ),
        VersionTarget(
            name="cargo_toml",
            relative_path=Path("s2and_rust") / "Cargo.toml",
            pattern=rf'(?m)^(?P<prefix>version = ")(?P<version>{SEMVER_PATTERN})(?P<suffix>"\s*)$',
        ),
        VersionTarget(
            name="runtime_required",
            relative_path=Path("s2and") / "runtime.py",
            pattern=rf'(?m)^(?P<prefix>REQUIRED_RUST_EXTENSION_VERSION = ")'
            rf'(?P<version>{SEMVER_PATTERN})(?P<suffix>"\r?)$',
        ),
        VersionTarget(
            name="cargo_lock",
            relative_path=Path("s2and_rust") / "Cargo.lock",
            pattern=(
                rf'(?m)^(?P<prefix>\[\[package\]\]\r?\nname = "s2and_rust"\r?\nversion = ")'
                rf'(?P<version>{SEMVER_PATTERN})(?P<suffix>")'
            ),
        ),
        VersionTarget(
            name="uv_lock",
            relative_path=Path("uv.lock"),
            pattern=(
                rf'(?m)^(?P<prefix>\[\[package\]\]\r?\nname = "s2and-rust"\r?\nversion = ")'
                rf'(?P<version>{SEMVER_PATTERN})(?P<suffix>")'
            ),
        ),
    )


def read_version(root: Path = ROOT) -> str:
    version_file = root / "VERSION"
    if not version_file.exists():
        raise SystemExit(f"VERSION file not found: {version_file}")
    version = _read_text(version_file).strip()
    if not re.match(r"^[0-9]+\.[0-9]+\.[0-9]+$", version):
        raise SystemExit(f"VERSION must be semver (X.Y.Z). Got: {version}")
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
    if groups.get("indent") is not None:
        return f'{groups["indent"]}"s2and-rust=={version}{groups["suffix"]}'
    return f"{groups['prefix']}{version}{groups['suffix']}"


def sync_target(root: Path, target: VersionTarget, version: str) -> None:
    path = target.path(root)
    text, match = _single_match(path, target.pattern)
    replacement = _replacement_from_match(match, version)
    new_text = text[: match.start()] + replacement + text[match.end() :]
    if new_text != text:
        _write_text(path, new_text)


def check_target(root: Path, target: VersionTarget, expected: str) -> None:
    path = target.path(root)
    _, match = _single_match(path, target.pattern)
    found = _version_from_match(match)
    if found != expected:
        raise SystemExit(f"Version mismatch in {path} ({target.name}): found {found}, expected {expected}")


def sync_version(version: str, root: Path = ROOT) -> None:
    for target in version_targets():
        sync_target(root, target, version)


def verify_version(version: str, root: Path = ROOT) -> None:
    for target in version_targets():
        check_target(root, target, version)


def release_decisions(
    *,
    event_name: str,
    ref: str,
    version_changed: bool,
    force_build: bool,
    publish_s2and_requested: bool,
    publish_rust_requested: bool,
) -> ReleaseDecisions:
    """Resolve release policy once for all workflow jobs."""

    on_main = ref == "refs/heads/main"
    publish_s2and = on_main and event_name == "workflow_dispatch" and publish_s2and_requested
    publish_rust = on_main and event_name == "workflow_dispatch" and publish_rust_requested
    publish_requested = publish_s2and_requested or publish_rust_requested
    build_s2and = version_changed or force_build or publish_requested
    build_rust = version_changed or force_build or publish_requested
    run_release_smoke = (
        publish_s2and
        or publish_rust
        or (event_name == "push" and on_main and version_changed)
        or (event_name == "pull_request" and force_build)
        or (event_name == "workflow_dispatch" and on_main and force_build)
    )
    return ReleaseDecisions(
        build_s2and=build_s2and,
        build_rust=build_rust,
        run_release_smoke=run_release_smoke,
        publish_s2and=publish_s2and,
        publish_rust=publish_rust,
    )


def _requested(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _event_payload(path_value: str) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"GitHub event payload must be an object: {path}")
    return payload


def _version_at_revision(root: Path, revision: str) -> str | None:
    if not revision:
        return None
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:VERSION"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not re.fullmatch(SEMVER_PATTERN, value):
        raise ValueError(f"VERSION at revision {revision!r} is not semver: {value!r}")
    return value


def release_decisions_from_environment(
    *,
    root: Path = ROOT,
    environ: Mapping[str, str] | None = None,
) -> tuple[ReleaseDecisions, str | None, str]:
    """Read one GitHub event and return final release decisions."""

    environment = os.environ if environ is None else environ
    event_name = environment.get("GITHUB_EVENT_NAME", "")
    event = _event_payload(environment.get("GITHUB_EVENT_PATH", ""))
    ref = environment.get("GITHUB_REF", "")
    inputs = event.get("inputs", {}) if event_name == "workflow_dispatch" else {}
    if not isinstance(inputs, Mapping):
        raise ValueError("GitHub workflow_dispatch inputs must be an object")
    force_build = _requested(inputs.get("force_build"))
    publish_s2and_requested = _requested(inputs.get("publish_s2and"))
    publish_rust_requested = _requested(inputs.get("publish_rust"))
    if event_name == "pull_request":
        pull_request = event.get("pull_request", {})
        if not isinstance(pull_request, Mapping):
            raise ValueError("GitHub pull_request payload must be an object")
        labels = pull_request.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError("GitHub pull_request labels must be a list")
        force_build = any(
            isinstance(label, Mapping) and str(label.get("name", "")).lower() == "force-build" for label in labels
        )
        base = pull_request.get("base", {})
        before = str(base.get("sha", "")) if isinstance(base, Mapping) else ""
    elif event_name == "push":
        before = environment.get("BEFORE_SHA", "")
    else:
        before = ""
    if before.startswith("0000000"):
        before = ""

    current_version = read_version(root)
    before_version = _version_at_revision(root, before)
    version_changed = event_name in {"pull_request", "push"} and before_version != current_version
    decisions = release_decisions(
        event_name=event_name,
        ref=ref,
        version_changed=version_changed,
        force_build=force_build,
        publish_s2and_requested=publish_s2and_requested,
        publish_rust_requested=publish_rust_requested,
    )
    return decisions, before_version, current_version


def _write_github_outputs(path: Path, values: Mapping[str, bool]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for key, value in values.items():
            output.write(f"{key}={'true' if value else 'false'}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Only verify versions match VERSION.")
    mode.add_argument(
        "--print-targets",
        action="store_true",
        help="Print repository-relative files updated or regenerated by version sync.",
    )
    mode.add_argument(
        "--release-policy",
        action="store_true",
        help="Write final GitHub release build/publish decisions to GITHUB_OUTPUT.",
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
        print(f"OK: versions and runtime guards match {version}")
        return
    if args.release_policy:
        verify_version(version)
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            raise SystemExit("GITHUB_OUTPUT is required with --release-policy")
        decisions, before_version, current_version = release_decisions_from_environment()
        _write_github_outputs(Path(output_path), decisions.github_outputs())
        print(f"version: {before_version} -> {current_version}")
        for key, value in decisions.github_outputs().items():
            print(f"{key}: {value}")
        return

    sync_version(version)
    verify_version(version)
    print(f"Updated versions and runtime guards to {version}")
    print(
        "Next: run `uv sync --extra dev` and "
        "`uv run --active --no-project cargo generate-lockfile --manifest-path s2and_rust/Cargo.toml`."
    )


if __name__ == "__main__":
    main()
