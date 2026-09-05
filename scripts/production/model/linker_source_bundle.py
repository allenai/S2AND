"""Assemble the reviewed linker source bundle and public data root."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import (  # noqa: E402
    build_arrow_artifact_manifest,
    read_arrow_collection_root,
    require_name_counts_index_artifact,
    write_arrow_artifact_manifest,
)
from s2and.incremental_linking_training.classic import load_bundle  # noqa: E402
from s2and.incremental_linking_training.source_bundle_preflight import (  # noqa: E402
    preflight_source_rows,
    validate_source_bundle_support_files,
)
from s2and.production_training_contract import load_model_plan  # noqa: E402
from scripts.convert_to_arrow import _write_root_manifest  # noqa: E402
from scripts.verification.validate_local_arrow_release import validate_release_root  # noqa: E402

_RESERVED_SUPPORT_ROOT_NAMES = frozenset({"LICENSE.txt", "datasets", "manifest.json", "name_counts_index"})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _dataset_manifests(root: Path) -> dict[str, Path]:
    dataset_manifests, replay_bundles, release_version = read_arrow_collection_root(root / "manifest.json")
    if release_version is not None or replay_bundles:
        raise ValueError(f"Assembly input must be a generic Arrow collection without replay bundles: {root}")
    return dataset_manifests


def _resolved_manifest_artifact_path(manifest_path: Path, raw_path: Any, *, key: str) -> Path:
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


def _validate_root_name_counts_binding(
    root: Path,
    *,
    authoritative_manifest_sha256: str,
) -> None:
    """Require every dataset manifest to bind the authoritative count index."""

    for dataset, manifest_path in _dataset_manifests(root).items():
        manifest = _read_json(manifest_path)
        paths = manifest.get("paths")
        if not isinstance(paths, Mapping):
            raise ValueError(f"{manifest_path} must contain paths")
        observed_index = _resolved_manifest_artifact_path(
            manifest_path,
            paths.get("name_counts_index"),
            key="name_counts_index",
        )
        observed_manifest = observed_index / "manifest.json"
        if not observed_manifest.is_file():
            raise FileNotFoundError(observed_manifest)
        observed_sha256 = sha256_file(observed_manifest)

        files = manifest.get("files")
        file_entry = files.get("name_counts_index") if isinstance(files, Mapping) else None
        declared_sha256 = file_entry.get("sha256") if isinstance(file_entry, Mapping) else None
        if observed_sha256 != authoritative_manifest_sha256 or declared_sha256 != authoritative_manifest_sha256:
            raise ValueError(
                f"{manifest_path} ({dataset}) name_counts_index does not match the authoritative index: "
                f"resolved_manifest_sha256={observed_sha256} "
                f"declared_manifest_sha256={declared_sha256!r} "
                f"authoritative_manifest_sha256={authoritative_manifest_sha256}"
            )


def _copy_arrow_root_without_name_counts(source: Path, destination: Path) -> None:
    """Copy one Arrow root while omitting its replaceable shared count index."""

    def ignore(source_dir: str, names: list[str]) -> set[str]:
        if Path(source_dir).resolve() == source.resolve() and "name_counts_index" in names:
            return {"name_counts_index"}
        return set()

    shutil.copytree(source, destination, ignore=ignore)


def _rebind_dataset_manifests(root: Path, *, name_counts_index: Path) -> None:
    """Rebuild dataset manifests for their final shared-index location."""

    entries = _dataset_manifests(root)
    for manifest_path in entries.values():
        manifest = _read_json(manifest_path)
        paths = manifest.get("paths")
        if not isinstance(paths, Mapping):
            raise ValueError(f"{manifest_path} must contain paths")
        resolved_paths = {
            str(key): (
                name_counts_index
                if str(key) == "name_counts_index"
                else _resolved_manifest_artifact_path(manifest_path, value, key=str(key))
            )
            for key, value in paths.items()
        }
        write_arrow_artifact_manifest(
            build_arrow_artifact_manifest(
                resolved_paths,
                manifest_path.parent,
            ),
            manifest_path.parent,
        )
    _write_root_manifest(
        root,
        dataset_manifests={
            dataset: manifest_path.relative_to(root.resolve()).as_posix() for dataset, manifest_path in entries.items()
        },
    )


def _validate_support_root(source_root: Path) -> None:
    conflicts = sorted(child.name for child in source_root.iterdir() if child.name in _RESERVED_SUPPORT_ROOT_NAMES)
    if conflicts:
        raise ValueError(f"linker support root contains reserved assembled paths: {conflicts}")


def assemble_source_bundle(
    *,
    source_root: Path,
    benchmark_arrow_root: Path,
    replay_arrow_root: Path,
    name_counts_index: Path,
    model_plan: Path,
    output_source_bundle: Path,
    output_data_root: Path,
) -> dict[str, Any]:
    """Copy reviewed inputs into fresh source and public-data roots."""

    source_root, benchmark_arrow_root, replay_arrow_root, name_counts_index, model_plan = (
        path.resolve() for path in (source_root, benchmark_arrow_root, replay_arrow_root, name_counts_index, model_plan)
    )
    release_version = load_model_plan(model_plan).release_version
    output_source_bundle, output_data_root = output_source_bundle.resolve(), output_data_root.resolve()
    if output_source_bundle.exists() or output_data_root.exists():
        raise FileExistsError("assembly output directories must not exist")
    if output_source_bundle.is_relative_to(output_data_root) or output_data_root.is_relative_to(output_source_bundle):
        raise ValueError("assembly output directories must not overlap")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    for arrow_root in (benchmark_arrow_root, replay_arrow_root):
        if not arrow_root.is_dir():
            raise FileNotFoundError(arrow_root)
    input_roots = (source_root, benchmark_arrow_root, replay_arrow_root, name_counts_index)
    for output in (output_source_bundle, output_data_root):
        if any(output.is_relative_to(input_root) for input_root in input_roots):
            raise ValueError(f"assembly output must not be nested beneath an input root: {output}")
    _validate_support_root(source_root)
    require_name_counts_index_artifact(
        name_counts_index,
        context="linker source assembly",
        producer_hint="pass the reviewed name_counts_index directory with --name-counts-index",
    )
    authoritative_manifest_sha256 = sha256_file(name_counts_index / "manifest.json")
    for arrow_root in (benchmark_arrow_root, replay_arrow_root):
        _validate_root_name_counts_binding(
            arrow_root,
            authoritative_manifest_sha256=authoritative_manifest_sha256,
        )

    _copy_arrow_root_without_name_counts(replay_arrow_root, output_source_bundle)
    shutil.copytree(name_counts_index, output_source_bundle / "name_counts_index")
    shutil.copytree(source_root, output_source_bundle, dirs_exist_ok=True)
    _rebind_dataset_manifests(
        output_source_bundle,
        name_counts_index=output_source_bundle / "name_counts_index",
    )

    _copy_arrow_root_without_name_counts(benchmark_arrow_root, output_data_root)
    shutil.copytree(name_counts_index, output_data_root / "name_counts_index")
    _copy_arrow_root_without_name_counts(replay_arrow_root, output_data_root / "linker_replay")
    _rebind_dataset_manifests(
        output_data_root,
        name_counts_index=output_data_root / "name_counts_index",
    )
    _rebind_dataset_manifests(
        output_data_root / "linker_replay",
        name_counts_index=output_data_root / "name_counts_index",
    )
    _write_root_manifest(
        output_data_root,
        dataset_manifests={
            dataset: manifest_path.relative_to(output_data_root.resolve()).as_posix()
            for dataset, manifest_path in _dataset_manifests(output_data_root).items()
        },
        replay_bundles={"linker_replay": "linker_replay/manifest.json"},
        release_version=release_version,
    )

    bundle = load_bundle(output_source_bundle)
    validate_source_bundle_support_files(bundle)
    with ExitStack() as arrow_stack:
        selected_source_rows, _arrow_datasets = preflight_source_rows(
            bundle,
            name_counts_index_root=output_source_bundle / "name_counts_index",
            arrow_stack=arrow_stack,
        )
    validate_release_root(output_data_root)
    return {
        "data_root": str(output_data_root),
        "name_counts_manifest_sha256": authoritative_manifest_sha256,
        "release_version": release_version,
        "selected_source_rows": selected_source_rows,
        "source_bundle": str(output_source_bundle),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "source-root",
        "benchmark-arrow-root",
        "replay-arrow-root",
        "name-counts-index",
        "model-plan",
        "output-source-bundle",
        "output-data-root",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    print(json.dumps(assemble_source_bundle(**vars(parser.parse_args(argv))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
