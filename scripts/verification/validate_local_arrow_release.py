#!/usr/bin/env python
"""Validate a local Arrow release root without network access or table scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    ValidatedArrowInputs,
    _validate_arrow_publication_artifacts_with_retained_name_counts,
    require_name_counts_index_artifact,
)

ROOT_MANIFEST_SCHEMA = "inference_arrow_bundle_v1"
ROOT_HELPER_FILES = ("LICENSE.txt",)
DECLARED_DIRECTORY_KEYS = frozenset({"name_counts_index"})
_NameCountsGenerationKey = tuple[Path, str]


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
            producer_hint=(
                "run python -m scripts.production.counts.generate_name_counts or refresh the release checkout"
            ),
        )
    except (OSError, TypeError, ValueError) as exc:
        _record_error(errors, str(exc))


def _name_counts_generation_key(path: Path) -> _NameCountsGenerationKey | None:
    manifest_path = path / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        return None
    return path.resolve(), hashlib.sha256(manifest_bytes).hexdigest()


def _validate_default_model_declaration(release_root: Path, errors: list[str]) -> str | None:
    declaration_path = release_root / "default_production_model.json"
    if not declaration_path.is_file():
        return None
    try:
        declaration = _load_json_object(declaration_path)
    except (OSError, TypeError, ValueError) as exc:
        _record_error(errors, str(exc))
        return None
    bundle_dir_value = declaration.get("bundle_dir")
    if not isinstance(bundle_dir_value, str) or not bundle_dir_value.strip():
        _record_error(errors, f"{declaration_path} is missing non-empty bundle_dir")
        return None
    bundle_dir = _manifest_path(bundle_dir_value, release_root).resolve()
    try:
        bundle_dir.relative_to(release_root)
    except ValueError:
        _record_error(errors, f"default production bundle is outside release root: {bundle_dir}")
        return None
    bundle_manifest_path = bundle_dir / "manifest.json"
    _require_file(bundle_manifest_path, errors, label="declared default production model manifest")
    if not bundle_manifest_path.is_file():
        return None
    try:
        bundle_manifest = _load_json_object(bundle_manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        _record_error(errors, str(exc))
        return None
    declared_version = declaration.get("bundle_version")
    manifest_version = bundle_manifest.get("bundle_version")
    if declared_version is not None and str(declared_version) != str(manifest_version):
        _record_error(
            errors,
            f"default production bundle_version mismatch: declaration={declared_version!r} "
            f"manifest={manifest_version!r}",
        )
    return str(bundle_manifest_path)


def _dataset_manifest_entries(root_manifest: Mapping[str, Any], root_manifest_path: Path) -> list[Mapping[str, Any]]:
    raw_entries = root_manifest.get("dataset_manifests")
    if not isinstance(raw_entries, list):
        raise ValueError(f"{root_manifest_path} is missing dataset_manifests list")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"{root_manifest_path} dataset_manifests[{index}] must be an object")
        entries.append(cast(Mapping[str, Any], entry))
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
    validated_name_counts: dict[_NameCountsGenerationKey, ValidatedArrowInputs],
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
    name_counts_path = Path(resolved_paths["name_counts_index"]) if "name_counts_index" in resolved_paths else None
    name_counts_key = None if name_counts_path is None else _name_counts_generation_key(name_counts_path)
    retained_name_counts = None if name_counts_key is None else validated_name_counts.get(name_counts_key)
    try:
        validated = _validate_arrow_publication_artifacts_with_retained_name_counts(
            resolved_paths,
            require_specter=True,
            require_name_counts_index=require_name_counts_index,
            context=label,
            retained_name_counts=retained_name_counts,
        )
        validated_manifest = validated.name_counts_manifest
        if validated_manifest is not None:
            validated_name_counts[(validated_manifest.index_dir, validated_manifest.manifest_sha256)] = validated
    except MissingArrowArtifactError as exc:
        _record_error(errors, str(exc))

    for key, path_value in paths.items():
        resolved = _manifest_path(path_value, manifest_path.parent)
        if str(key) not in DECLARED_DIRECTORY_KEYS:
            _require_file(resolved, errors, label=f"{label} paths.{key}")
    return 1


def _validate_replay_bundles(
    release_root: Path,
    root_manifest: Mapping[str, Any],
    errors: list[str],
    *,
    validated_name_counts: dict[_NameCountsGenerationKey, ValidatedArrowInputs],
) -> int:
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
        bundle = cast(Mapping[str, Any], bundle)
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
                validated_name_counts=validated_name_counts,
            )
    return validated


def validate_release_root(release_root: Path, *, include_replay_bundles: bool = True) -> dict[str, Any]:
    """Return validation metrics for a local Arrow release root.

    Raises:
        ValueError: If the release root has missing or inconsistent local artifacts.
    """

    resolved_root = release_root.resolve()
    errors: list[str] = []
    validated_name_counts: dict[_NameCountsGenerationKey, ValidatedArrowInputs] = {}
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
    default_model_manifest = _validate_default_model_declaration(resolved_root, errors)

    entries = _dataset_manifest_entries(root_manifest, root_manifest_path)
    expected_count = (
        root_manifest.get("audit", {}).get("dataset_count") if isinstance(root_manifest.get("audit"), Mapping) else None
    )
    if expected_count is not None and int(expected_count) != len(entries):
        _record_error(errors, f"root audit.dataset_count mismatch: {expected_count} != {len(entries)}")

    validated_datasets = 0
    for entry in entries:
        validated_datasets += _validate_dataset_manifest(
            resolved_root,
            entry,
            errors,
            label_prefix="root",
            validated_name_counts=validated_name_counts,
        )

    validated_replay_datasets = (
        _validate_replay_bundles(
            resolved_root,
            root_manifest,
            errors,
            validated_name_counts=validated_name_counts,
        )
        if include_replay_bundles
        else 0
    )
    root_name_counts_path = resolved_root / "name_counts_index"
    root_name_counts_key = _name_counts_generation_key(root_name_counts_path)
    if root_name_counts_key is None or root_name_counts_key not in validated_name_counts:
        _validate_name_counts_index(root_name_counts_path, errors, label="root name_counts_index")

    if errors:
        raise ValueError("\n".join(errors))
    return {
        "release_root": str(resolved_root),
        "dataset_manifest_count": validated_datasets,
        "replay_dataset_manifest_count": validated_replay_datasets,
        "name_counts_index": str(resolved_root / "name_counts_index"),
        "default_model_manifest": default_model_manifest,
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
