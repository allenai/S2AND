#!/usr/bin/env python
"""Validate a local Arrow release root without network access or table scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    require_name_counts_index_artifact,
    validate_arrow_prediction_artifacts,
)

ROOT_MANIFEST_SCHEMA = "inference_arrow_bundle_v1"
ROOT_HELPER_FILES = (
    "LICENSE.txt",
    "production_model_v1.21/manifest.json",
)
DECLARED_DIRECTORY_KEYS = frozenset({"name_counts_index"})


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _manifest_path(path_value: Any, base_dir: Path) -> Path:
    path_text = str(path_value).replace("\\", "/")
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_error(errors: list[str], message: str) -> None:
    errors.append(message)


def _require_file(path: Path, errors: list[str], *, label: str) -> None:
    if not path.exists():
        _record_error(errors, f"{label} is missing: {path}")
    elif not path.is_file():
        _record_error(errors, f"{label} is not a file: {path}")


def _validate_name_counts_index(path: Path, errors: list[str], *, label: str) -> None:
    try:
        require_name_counts_index_artifact(
            path,
            context=label,
            producer_hint="run scripts/convert_to_arrow.py name-counts-index or refresh the release checkout",
        )
    except (OSError, TypeError, ValueError) as exc:
        _record_error(errors, str(exc))


def _dataset_manifest_entries(root_manifest: Mapping[str, Any], root_manifest_path: Path) -> list[Mapping[str, Any]]:
    raw_entries = root_manifest.get("dataset_manifests")
    if not isinstance(raw_entries, list):
        raise ValueError(f"{root_manifest_path} is missing dataset_manifests list")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"{root_manifest_path} dataset_manifests[{index}] must be an object")
        entries.append(entry)
    return entries


def _validate_entry_manifest_checksum(
    entry: Mapping[str, Any],
    manifest_path: Path,
    errors: list[str],
    *,
    label: str,
) -> None:
    expected_size = entry.get("manifest_size_bytes")
    if expected_size is not None and int(expected_size) != manifest_path.stat().st_size:
        _record_error(
            errors,
            f"{label} manifest_size_bytes mismatch for {manifest_path}: "
            f"{expected_size} != {manifest_path.stat().st_size}",
        )
    expected_sha = entry.get("manifest_sha256")
    if expected_sha is not None:
        observed_sha = _sha256(manifest_path)
        if str(expected_sha) != observed_sha:
            _record_error(
                errors,
                f"{label} manifest_sha256 mismatch for {manifest_path}: {expected_sha} != {observed_sha}",
            )


def _validate_dataset_manifest(
    release_root: Path,
    entry: Mapping[str, Any],
    errors: list[str],
    *,
    label_prefix: str,
) -> int:
    dataset = str(entry.get("dataset") or "<unknown>")
    label = f"{label_prefix} dataset {dataset}"
    manifest_path_value = entry.get("manifest_path")
    if manifest_path_value is None:
        _record_error(errors, f"{label} is missing manifest_path")
        return 0
    manifest_path = _manifest_path(manifest_path_value, release_root)
    _require_file(manifest_path, errors, label=f"{label} manifest")
    if not manifest_path.is_file():
        return 0

    _validate_entry_manifest_checksum(entry, manifest_path, errors, label=label)
    manifest = _load_json_object(manifest_path)
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        _record_error(errors, f"{label} manifest is missing paths mapping: {manifest_path}")
        return 0

    requirements = entry.get("validation_requirements")
    require_name_counts_index = isinstance(requirements, Mapping) and bool(
        requirements.get("require_name_counts_index")
    )
    resolved_paths = {
        str(key): str(_manifest_path(path_value, manifest_path.parent)) for key, path_value in paths.items()
    }
    try:
        validate_arrow_prediction_artifacts(
            resolved_paths,
            require_specter=True,
            require_name_counts_index=require_name_counts_index,
            require_batch_indexes=True,
            context=label,
        )
    except MissingArrowArtifactError as exc:
        _record_error(errors, str(exc))

    for key, path_value in paths.items():
        resolved = _manifest_path(path_value, manifest_path.parent)
        if str(key) in DECLARED_DIRECTORY_KEYS:
            _validate_name_counts_index(resolved, errors, label=f"{label} {key}")
        else:
            _require_file(resolved, errors, label=f"{label} paths.{key}")
    return 1


def _validate_replay_bundles(release_root: Path, root_manifest: Mapping[str, Any], errors: list[str]) -> int:
    raw_bundles = root_manifest.get("replay_bundles", [])
    if raw_bundles is None:
        return 0
    if not isinstance(raw_bundles, list):
        _record_error(errors, f"{release_root / 'manifest.json'} replay_bundles must be a list")
        return 0

    validated = 0
    for index, bundle in enumerate(raw_bundles):
        if not isinstance(bundle, Mapping):
            _record_error(errors, f"replay_bundles[{index}] must be an object")
            continue
        manifest_path_value = bundle.get("manifest_path")
        if manifest_path_value is None:
            _record_error(errors, f"replay_bundles[{index}] is missing manifest_path")
            continue
        manifest_path = _manifest_path(manifest_path_value, release_root)
        _require_file(manifest_path, errors, label=f"replay bundle {index} manifest")
        if not manifest_path.is_file():
            continue
        _validate_entry_manifest_checksum(bundle, manifest_path, errors, label=f"replay bundle {index}")
        nested_manifest = _load_json_object(manifest_path)
        for entry in _dataset_manifest_entries(nested_manifest, manifest_path):
            validated += _validate_dataset_manifest(
                manifest_path.parent,
                entry,
                errors,
                label_prefix=f"replay bundle {bundle.get('bundle') or index}",
            )
    return validated


def validate_release_root(release_root: Path, *, include_replay_bundles: bool = True) -> dict[str, Any]:
    """Return validation metrics for a local Arrow release root.

    Raises:
        ValueError: If the release root has missing or inconsistent local artifacts.
    """

    resolved_root = release_root.resolve()
    errors: list[str] = []
    root_manifest_path = resolved_root / "manifest.json"
    _require_file(root_manifest_path, errors, label="root manifest")
    if not root_manifest_path.is_file():
        raise ValueError("\n".join(errors))

    root_manifest = _load_json_object(root_manifest_path)
    if root_manifest.get("schema") != ROOT_MANIFEST_SCHEMA:
        _record_error(
            errors,
            f"root manifest schema mismatch: {root_manifest.get('schema')!r} != {ROOT_MANIFEST_SCHEMA!r}",
        )

    for helper in ROOT_HELPER_FILES:
        _require_file(resolved_root / helper, errors, label=f"root helper {helper}")
    _validate_name_counts_index(resolved_root / "name_counts_index", errors, label="root name_counts_index")

    entries = _dataset_manifest_entries(root_manifest, root_manifest_path)
    expected_count = (
        root_manifest.get("audit", {}).get("dataset_count") if isinstance(root_manifest.get("audit"), Mapping) else None
    )
    if expected_count is not None and int(expected_count) != len(entries):
        _record_error(errors, f"root audit.dataset_count mismatch: {expected_count} != {len(entries)}")

    validated_datasets = 0
    for entry in entries:
        validated_datasets += _validate_dataset_manifest(resolved_root, entry, errors, label_prefix="root")

    validated_replay_datasets = (
        _validate_replay_bundles(resolved_root, root_manifest, errors) if include_replay_bundles else 0
    )

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "release_root": str(resolved_root),
        "dataset_manifest_count": validated_datasets,
        "replay_dataset_manifest_count": validated_replay_datasets,
        "name_counts_index": str(resolved_root / "name_counts_index"),
        "network_access": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a local Arrow release root with manifest/file checks only; no S3 or table scans."
    )
    parser.add_argument("--release-root", type=Path, default=Path("s2and/data"))
    parser.add_argument("--skip-replay-bundles", action="store_true")
    parser.add_argument("--write-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        metrics = validate_release_root(args.release_root, include_replay_bundles=not args.skip_replay_bundles)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
