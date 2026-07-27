"""Standard-library-only validation for transported release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import urlopen

RELEASE_EVIDENCE_MANIFEST_SCHEMA = "s2and_release_evidence_manifest_v1"
RELEASE_EVIDENCE_ROLES = frozenset(
    {
        "cluster_evaluation_report",
        "complete_model_manifest",
        "data_manifest",
        "linker_evaluation_report",
        "pairwise_evaluation_report",
        "parity_evaluation_report",
        "performance_evaluation_report",
        "release_spec",
        "subblocking_evaluation_report",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_release_evidence_manifest(
    payload: Any,
    manifest_path: Path,
    *,
    require_urls: bool = False,
    verify_local_members: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate the transport and local-integrity contract for release evidence."""

    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "members"}
        or payload["schema_version"] != RELEASE_EVIDENCE_MANIFEST_SCHEMA
        or not isinstance(payload["members"], Mapping)
    ):
        raise ValueError(f"Release-evidence manifest must be a {RELEASE_EVIDENCE_MANIFEST_SCHEMA!r} object")

    members = payload["members"]
    missing = sorted(RELEASE_EVIDENCE_ROLES - set(members))
    extra = sorted(set(members) - RELEASE_EVIDENCE_ROLES)
    if missing or extra:
        raise ValueError(f"Release-evidence manifest roles disagree with contract: missing={missing} extra={extra}")

    root = manifest_path.resolve().parent
    validated: dict[str, dict[str, Any]] = {}
    for role in sorted(members):
        member = members[role]
        base_keys = {"path", "sha256", "size_bytes"}
        allowed_keys = {frozenset(base_keys), frozenset({*base_keys, "url"})}
        if (
            not isinstance(member, Mapping)
            or frozenset(member) not in allowed_keys
            or (require_urls and "url" not in member)
        ):
            raise ValueError(
                f"Release-evidence manifest member {role!r} must contain path, sha256, size_bytes"
                f"{', and url' if require_urls else ', with optional url'}"
            )
        raw_path = member["path"]
        relative_path = PurePosixPath(raw_path) if isinstance(raw_path, str) else PurePosixPath()
        if (
            not isinstance(raw_path, str)
            or not raw_path
            or "\\" in raw_path
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != raw_path
        ):
            raise ValueError(f"Release-evidence manifest member {role!r} path must be normalized relative POSIX")
        expected_sha256 = member["sha256"]
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"Release-evidence manifest member {role!r} sha256 must be lowercase SHA-256")
        expected_size = member["size_bytes"]
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError(f"Release-evidence manifest member {role!r} size_bytes must be nonnegative")
        if "url" in member and (
            not isinstance(member["url"], str)
            or not member["url"].startswith("https://")
            or member["url"].strip() != member["url"]
        ):
            raise ValueError(f"Release-evidence manifest member {role!r} url must be an HTTPS URL")

        path = (root / Path(*relative_path.parts)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Release-evidence manifest member {role!r} escapes its manifest root")
        if verify_local_members:
            if path.stat().st_size != expected_size:
                raise ValueError(f"Release-evidence manifest member {role!r} size mismatch")
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    f"Release-evidence manifest member {role!r} SHA-256 mismatch: "
                    f"expected={expected_sha256} observed={observed_sha256}"
                )
        validated[role] = dict(member)
    return validated


def stage_release_evidence(
    manifest_path: Path,
    expected_manifest_sha256: str,
    output_root: Path,
) -> Path:
    """Download a pinned transport manifest's members into one fresh local root."""

    manifest_bytes = manifest_path.read_bytes()
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Release-evidence manifest SHA-256 mismatch: "
            f"expected={expected_manifest_sha256} observed={observed_manifest_sha256}"
        )
    payload = json.loads(manifest_bytes)
    members = validate_release_evidence_manifest(
        payload,
        manifest_path,
        require_urls=True,
        verify_local_members=False,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    staged_manifest = output_root / "evidence_manifest.json"
    staged_manifest.write_bytes(manifest_bytes)
    for member in members.values():
        destination = output_root / Path(*PurePosixPath(member["path"]).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(member["url"], timeout=60) as response, destination.open("wb") as output:  # noqa: S310
            shutil.copyfileobj(response, output)
    validate_release_evidence_manifest(payload, staged_manifest, require_urls=True)
    return staged_manifest


def main(argv: list[str] | None = None) -> None:
    """Stage a transport manifest using only the Python standard library."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    staged_manifest = stage_release_evidence(
        args.manifest,
        args.expected_manifest_sha256,
        args.output_root,
    )
    print(json.dumps({"evidence_manifest": str(staged_manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
