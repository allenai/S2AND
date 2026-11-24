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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import (  # noqa: E402
    ArrowDataset,
    read_arrow_collection_root,
    require_name_counts_index_artifact,
)
from s2and.incremental_linking_training.classic import OfficialBundle, load_bundle  # noqa: E402
from s2and.incremental_linking_training.source_bundle_preflight import (  # noqa: E402
    preflight_source_rows,
    validate_source_bundle_support_files,
    validate_source_label_tables,
)
from s2and.production_training_contract import load_model_plan  # noqa: E402
from scripts.common.arrow_roots import (  # noqa: E402
    copy_arrow_root_without_name_counts,
    generic_dataset_manifests,
    read_json_object,
    rebind_dataset_manifests,
    reject_filesystem_links,
    resolved_manifest_artifact_path,
    validate_root_name_counts_binding,
)
from scripts.intake.convert_to_arrow import write_root_manifest  # noqa: E402
from scripts.release.verification.validate_local_arrow_release import validate_release_root  # noqa: E402

_RESERVED_SUPPORT_ROOT_NAMES = frozenset({"LICENSE.txt", "datasets", "manifest.json", "name_counts_index"})


def _portable_dataset_manifests(root: Path) -> dict[str, Path]:
    """Return dataset manifests from a generic or public source bundle."""

    entries, replay_bundles, _release_version = read_arrow_collection_root(root / "manifest.json")
    for bundle_name, manifest_path in replay_bundles.items():
        nested, nested_replays, release_version = read_arrow_collection_root(manifest_path)
        if nested_replays or release_version is not None:
            raise ValueError(f"Replay manifest must contain a generic Arrow collection: {manifest_path}")
        entries.update({f"{bundle_name}/{name}": path for name, path in nested.items()})
    return entries


def validate_label_replay_datasets(bundle: OfficialBundle, replay_manifests: Mapping[str, Path]) -> None:
    """Require a replay dataset for every dataset referenced by source labels."""
    _, label_datasets = validate_source_label_tables(bundle)
    if missing := sorted(label_datasets - replay_manifests.keys()):
        raise ValueError(f"Source label datasets are absent from the replay collection: {missing}")


def _validate_support_root(source_root: Path) -> None:
    conflicts = sorted(child.name for child in source_root.iterdir() if child.name in _RESERVED_SUPPORT_ROOT_NAMES)
    if conflicts:
        raise ValueError(f"linker support root contains reserved assembled paths: {conflicts}")


def validate_bundle_paths(root: Path) -> None:
    """Reject support paths that cannot survive a directory relocation."""

    bundle = read_json_object(root / "bundle.json")

    def check(value: Any, field: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"bundle.{field} must be a non-empty relative path")
        relative = Path(value)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"bundle.{field} escapes the source bundle: {value!r}")
        if not resolved.exists():
            raise FileNotFoundError(f"bundle.{field}: {resolved}")

    def assets(value: Any, field: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                assets(child, f"{field}.{key}")
        else:
            check(value, field)

    def model_paths(value: Any, field: str) -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            child_field = f"{field}.{key}"
            if str(key).endswith("_path") and child is not None:
                check(child, child_field)
            elif str(key).endswith("_paths") and isinstance(child, Mapping):
                assets(child, child_field)
            else:
                model_paths(child, child_field)

    assets(bundle["assets"], "assets")
    model_paths(bundle["models"], "models")
    runtime_root = bundle.get("runtime_contract", {}).get("arrow_dataset_root")
    if runtime_root is not None:
        check(runtime_root, "runtime_contract.arrow_dataset_root")


def validate_portable_source_bundle(root: Path) -> dict[str, Any]:
    """Verify all operational paths and payloads within one portable root."""

    root = root.resolve()
    reject_filesystem_links(root, context="portable source bundle")
    validate_bundle_paths(root)
    entries = _portable_dataset_manifests(root)
    shared_index = root / "name_counts_index"
    index_sha256 = sha256_file(shared_index / "manifest.json")
    for manifest_path in entries.values():
        if not manifest_path.is_relative_to(root):
            raise ValueError(f"dataset manifest escapes the source bundle: {manifest_path}")
        manifest = read_json_object(manifest_path)
        for key, value in manifest["paths"].items():
            resolved = resolved_manifest_artifact_path(manifest_path, value, key=key)
            if not resolved.is_relative_to(root):
                raise ValueError(f"{manifest_path} paths.{key} escapes the source bundle")
            if key == "name_counts_index" and resolved != shared_index:
                raise ValueError(f"{manifest_path} must bind the shared name_counts_index")
    validate_root_name_counts_binding(root, authoritative_manifest_sha256=index_sha256, dataset_manifests=entries)
    bundle = load_bundle(root)
    validate_source_bundle_support_files(bundle)
    with ExitStack() as arrow_stack:
        selected_rows, datasets = preflight_source_rows(
            bundle,
            name_counts_index_root=shared_index,
            arrow_stack=arrow_stack,
        )
        opened_roots = {dataset.root.resolve() for dataset in datasets.values()}
        # Unreferenced datasets can become active after a future label update.
        for manifest_path in entries.values():
            dataset_root = manifest_path.parent.resolve()
            if dataset_root not in opened_roots:
                arrow_stack.enter_context(
                    ArrowDataset.open(dataset_root, require_specter=True, require_name_counts_index=True)
                )
                opened_roots.add(dataset_root)
    return {
        "dataset_count": len(entries),
        "datasets": sorted(entries),
        "name_counts_manifest_sha256": index_sha256,
        "selected_source_rows": selected_rows,
        "source_bundle": str(root),
    }


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
        verify=True,
    )
    authoritative_manifest_sha256 = sha256_file(name_counts_index / "manifest.json")
    replay_manifests = generic_dataset_manifests(replay_arrow_root)
    validate_root_name_counts_binding(benchmark_arrow_root, authoritative_manifest_sha256=authoritative_manifest_sha256)
    validate_root_name_counts_binding(
        replay_arrow_root,
        authoritative_manifest_sha256=authoritative_manifest_sha256,
        dataset_manifests=replay_manifests,
    )

    bundle = load_bundle(source_root)
    validate_source_bundle_support_files(bundle)
    validate_label_replay_datasets(bundle, replay_manifests)

    copy_arrow_root_without_name_counts(replay_arrow_root, output_source_bundle)
    shutil.copytree(name_counts_index, output_source_bundle / "name_counts_index")
    shutil.copytree(source_root, output_source_bundle, dirs_exist_ok=True)
    rebind_dataset_manifests(
        output_source_bundle,
        name_counts_index=output_source_bundle / "name_counts_index",
    )

    copy_arrow_root_without_name_counts(benchmark_arrow_root, output_data_root)
    shutil.copytree(name_counts_index, output_data_root / "name_counts_index")
    copy_arrow_root_without_name_counts(replay_arrow_root, output_data_root / "linker_replay")
    rebind_dataset_manifests(
        output_data_root,
        name_counts_index=output_data_root / "name_counts_index",
    )
    rebind_dataset_manifests(
        output_data_root / "linker_replay",
        name_counts_index=output_data_root / "name_counts_index",
    )
    write_root_manifest(
        output_data_root,
        dataset_manifests={
            dataset: manifest_path.relative_to(output_data_root.resolve()).as_posix()
            for dataset, manifest_path in generic_dataset_manifests(output_data_root).items()
        },
        replay_bundles={"linker_replay": "linker_replay/manifest.json"},
        release_version=release_version,
    )

    source_report = validate_portable_source_bundle(output_source_bundle)
    validate_release_root(output_data_root)
    return {
        "data_root": str(output_data_root),
        "name_counts_manifest_sha256": authoritative_manifest_sha256,
        "release_version": release_version,
        "selected_source_rows": source_report["selected_source_rows"],
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
