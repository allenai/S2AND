#!/usr/bin/env python
"""Validate a local Arrow release root without network access or table scans."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from s2and.arrow_inputs import (
    ARROW_COLLECTION_KIND,
    PUBLIC_DATA_KIND,
    ArrowDataset,
    read_arrow_collection_root,
    require_name_counts_index_artifact,
)

ROOT_HELPER_FILES = ("LICENSE.txt",)


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


def _validate_dataset_manifest(
    manifest_path: Path,
    publication_root: Path,
    dataset: str,
    errors: list[str],
    *,
    label_prefix: str,
) -> int:
    label = f"{label_prefix} dataset {dataset}"
    manifest = _load_json_object(manifest_path)
    paths = manifest.get("paths")
    if not isinstance(paths, Mapping):
        _record_error(errors, f"{label} manifest is missing paths mapping: {manifest_path}")
        return 0
    name_counts_path = paths.get("name_counts_index")
    if name_counts_path is not None:
        observed_name_counts = _manifest_path(name_counts_path, manifest_path.parent).resolve()
        expected_name_counts = (publication_root / "name_counts_index").resolve()
        if observed_name_counts != expected_name_counts:
            _record_error(
                errors,
                f"{label} paths.name_counts_index must resolve to the publication root index: "
                f"observed={observed_name_counts} expected={expected_name_counts}",
            )

    try:
        with ArrowDataset.open(
            manifest_path.parent,
            require_specter=True,
            require_name_counts_index=True,
        ):
            pass
    except (OSError, TypeError, ValueError) as exc:
        _record_error(errors, str(exc))
    return 1


def _validate_replay_bundles(
    release_root: Path,
    replay_bundles: Mapping[str, Path],
    errors: list[str],
) -> int:
    validated = 0
    for bundle_name, manifest_path in replay_bundles.items():
        nested_datasets, _nested_replays, release_version = read_arrow_collection_root(manifest_path)
        if release_version is not None:
            raise ValueError(f"{manifest_path} kind must be {ARROW_COLLECTION_KIND!r}")
        for dataset, dataset_manifest_path in nested_datasets.items():
            validated += _validate_dataset_manifest(
                dataset_manifest_path,
                release_root,
                dataset,
                errors,
                label_prefix=f"replay bundle {bundle_name}",
            )
    return validated


def validate_release_root(release_root: Path) -> dict[str, Any]:
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

    dataset_manifests, replay_bundles, release_version = read_arrow_collection_root(root_manifest_path)
    if release_version is None:
        raise ValueError(f"{root_manifest_path} kind must be {PUBLIC_DATA_KIND!r}")

    for helper in ROOT_HELPER_FILES:
        _require_file(resolved_root / helper, errors, label=f"root helper {helper}")

    validated_datasets = 0
    for dataset, manifest_path in dataset_manifests.items():
        validated_datasets += _validate_dataset_manifest(
            manifest_path,
            resolved_root,
            dataset,
            errors,
            label_prefix="root",
        )

    validated_replay_datasets = _validate_replay_bundles(
        resolved_root,
        replay_bundles,
        errors,
    )
    root_name_counts_path = resolved_root / "name_counts_index"
    _validate_name_counts_index(root_name_counts_path, errors, label="root name_counts_index")

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
    parser.add_argument("--write-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        metrics = validate_release_root(args.release_root)
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
