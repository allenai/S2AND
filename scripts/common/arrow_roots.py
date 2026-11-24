"""Read, copy and rebind Arrow collection roots that several stages package."""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import (
    _load_artifact_specs,
    build_arrow_artifact_manifest,
    read_arrow_collection_root,
    write_arrow_artifact_manifest,
)
from scripts.intake.convert_to_arrow import write_root_manifest


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON file that must hold one object."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def generic_dataset_manifests(root: Path) -> dict[str, Path]:
    """Return the dataset manifests of a generic Arrow collection without replay bundles."""
    dataset_manifests, replay_bundles, release_version = read_arrow_collection_root(root / "manifest.json")
    if release_version is not None or replay_bundles:
        raise ValueError(f"Assembly input must be a generic Arrow collection without replay bundles: {root}")
    return dataset_manifests


def resolved_manifest_artifact_path(manifest_path: Path, raw_path: Any, *, key: str) -> Path:
    """Resolve one manifest-relative artifact path and keep it inside the dataset directory."""
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{manifest_path} paths.{key} must be a non-empty string")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"{manifest_path} paths.{key} must be manifest-relative")
    resolved = (manifest_path.parent / path).resolve()
    if key != "name_counts_index":
        try:
            resolved.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(f"{manifest_path} paths.{key} escapes the dataset directory") from exc
    return resolved


def validate_root_name_counts_binding(
    root: Path,
    *,
    authoritative_manifest_sha256: str,
    dataset_manifests: Mapping[str, Path] | None = None,
) -> None:
    """Require every dataset manifest to locate the authoritative count index."""

    entries = generic_dataset_manifests(root) if dataset_manifests is None else dataset_manifests
    for dataset, manifest_path in entries.items():
        manifest = read_json_object(manifest_path)
        paths = manifest.get("paths")
        if not isinstance(paths, Mapping):
            raise ValueError(f"{manifest_path} must contain paths")
        observed_index = resolved_manifest_artifact_path(
            manifest_path,
            paths.get("name_counts_index"),
            key="name_counts_index",
        )
        observed_manifest = observed_index / "manifest.json"
        if not observed_manifest.is_file():
            raise FileNotFoundError(observed_manifest)
        observed_sha256 = sha256_file(observed_manifest)
        # A dataset manifest records where the index is, not which one it is.
        if observed_sha256 != authoritative_manifest_sha256:
            raise ValueError(
                f"{manifest_path} ({dataset}) name_counts_index does not match the authoritative index: "
                f"resolved_manifest_sha256={observed_sha256} "
                f"authoritative_manifest_sha256={authoritative_manifest_sha256}"
            )


def copy_arrow_root_without_name_counts(source: Path, destination: Path) -> None:
    """Copy one Arrow root while omitting its replaceable shared count index."""

    def ignore(source_dir: str, names: list[str]) -> set[str]:
        if Path(source_dir).resolve() == source.resolve() and "name_counts_index" in names:
            return {"name_counts_index"}
        return set()

    shutil.copytree(source, destination, ignore=ignore)


def reject_filesystem_links(root: Path, *, context: str) -> None:
    """Reject symlinks and junctions, because a copy must not depend on paths outside the root."""

    for directory, names, files in os.walk(root):
        for name in (*names, *files):
            path = Path(directory) / name
            metadata = path.lstat()
            if path.is_symlink() or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"{context} contains a filesystem link: {path}")


def _rebind_dataset_manifest(manifest_path: Path, *, name_counts_index: Path) -> None:
    """Verify payloads before a shared-index path update."""

    # A rebuild must not repair invalid metadata or changed payloads.
    _load_artifact_specs(manifest_path.parent)
    manifest = read_json_object(manifest_path)
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError(f"{manifest_path} must contain paths")
    resolved_paths = {
        str(key): (
            name_counts_index
            if str(key) == "name_counts_index"
            else resolved_manifest_artifact_path(manifest_path, value, key=str(key))
        )
        for key, value in paths.items()
    }
    rebuilt = build_arrow_artifact_manifest(resolved_paths, manifest_path.parent)
    # Manifests from before 1.0.0b11 also list the index among their files. Drop that entry:
    # after relocation it could name another index.
    manifest["files"] = {key: entry for key, entry in manifest["files"].items() if key != "name_counts_index"}
    if rebuilt["files"] != manifest["files"]:
        raise ValueError(f"Arrow artifact inventory mismatch before index relocation: {manifest_path}")
    manifest["paths"]["name_counts_index"] = rebuilt["paths"]["name_counts_index"]
    write_arrow_artifact_manifest(manifest, manifest_path.parent)


def rebind_dataset_manifests(root: Path, *, name_counts_index: Path) -> None:
    """Verify each original generation before relocating its shared-index path."""

    entries = generic_dataset_manifests(root)
    for manifest_path in entries.values():
        _rebind_dataset_manifest(manifest_path, name_counts_index=name_counts_index)
    write_root_manifest(
        root,
        dataset_manifests={
            dataset: manifest_path.relative_to(root.resolve()).as_posix() for dataset, manifest_path in entries.items()
        },
    )
