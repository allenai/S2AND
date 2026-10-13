"""Prepare corrected research data with one shared inference count index."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import (  # noqa: E402
    ArrowDataset,
    read_arrow_collection_root,
    require_name_counts_index_artifact,
)
from s2and.arrow_provenance import verify_conversion_provenance  # noqa: E402
from s2and.production_training_contract import _unit_float, load_model_plan  # noqa: E402
from scripts.convert_to_arrow import _write_root_manifest  # noqa: E402
from scripts.production.model.linker_source_bundle import (  # noqa: E402
    _copy_arrow_root_without_name_counts,
    _dataset_manifests,
    _read_json,
    _rebind_dataset_manifests,
    _validate_bundle_paths,
    _validate_copy_output,
    _validate_root_name_counts_binding,
    validate_portable_source_bundle,
)
from scripts.production.model.unknown_signatures import load_unknown_signatures  # noqa: E402
from scripts.verification.validate_local_arrow_release import validate_release_root  # noqa: E402

ARROW_SOURCE_ROLES = ("signatures", "papers", "specter_embeddings")
MODEL_ROLES = frozenset((*ARROW_SOURCE_ROLES, "clusters", "train_pairs", "val_pairs"))
DEFAULT_EPS_GRID = tuple(step / 100 for step in range(101))


def _relative_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _local_path(root: Path, value: Any, *, boundary: Path) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Require a relative input path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(boundary.resolve()):
        raise ValueError(f"Input path escapes the dataset snapshot: {value!r}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _copy_file(source: Path, destination: Path, *, expected_sha256: str | None = None) -> str:
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_file(destination) != digest:
        raise ValueError(f"Copied file SHA-256 mismatch: {destination}")
    return digest


def _copy_corrected_sources(
    source: Path, destination: Path, *, snapshot: Path, inventory: dict[str, str]
) -> dict[tuple[str, str, str], Path]:
    reusable: dict[tuple[str, str, str], Path] = {}
    for dataset in sorted(path for path in source.iterdir() if path.is_dir()):
        name = dataset.name
        if not (dataset / f"{name}_signatures.json").is_file():
            continue
        for filename in (f"{name}_papers.json", f"{name}_specter2.pkl"):
            if not (dataset / filename).is_file():
                raise FileNotFoundError(dataset / filename)
        has_clusters = (dataset / f"{name}_clusters.json").is_file()
        has_fixed_pairs = (dataset / "train_pairs.csv").is_file() and any(
            (dataset / filename).is_file() for filename in ("test_pairs.csv", "test_pairs.json")
        )
        if not has_clusters and not has_fixed_pairs:
            raise ValueError(f"Require cluster labels or train/test fixed pairs: {dataset}")
        for file in sorted(dataset.iterdir()):
            if not file.is_file() or file.name.startswith("preparation_report") or file.suffix == ".lock":
                continue
            target = destination / name / file.name
            digest = _copy_file(file, target)
            inventory[target.relative_to(snapshot).as_posix()] = digest
            reusable[(name, file.name, digest)] = target
    if not reusable:
        raise ValueError(f"Corrected source contains no complete datasets: {source}")
    return reusable


def _copy_current_tree(source: Path, destination: Path, *, snapshot: Path, inventory: dict[str, str]) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Current replay source contains no files: {source}")
    for file in files:
        target = destination / file.relative_to(source)
        inventory[target.relative_to(snapshot).as_posix()] = _copy_file(file, target)


def _validate_corrected_source_inventory(root: Path, corrected: Path) -> int:
    inventory_path = corrected / "manifest.json"
    inventory = _read_json(inventory_path)["files"]
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Corrected source inventory must contain files")
    actual = {
        path.relative_to(root).as_posix() for path in corrected.rglob("*") if path.is_file() and path != inventory_path
    }
    replay_root = root / "corrected_replay"
    if replay_root.exists():
        actual.update(path.relative_to(root).as_posix() for path in replay_root.rglob("*") if path.is_file())
    if set(inventory) != actual:
        raise ValueError("Corrected source inventory does not match the current source files")
    for relative, digest in inventory.items():
        path = _local_path(root, relative, boundary=root)
        if sha256_file(path) != digest:
            raise ValueError(f"Corrected source SHA-256 mismatch: {path}")
    return len(inventory)


def validate_corrected_datasets(root: Path) -> dict[str, Any]:
    """Validate the public, pairwise, and incremental inputs within one root."""

    root = root.resolve()
    index = root / "name_counts_index"
    index_dirs = [path for path in root.rglob("name_counts_index") if path.is_dir()]
    if index_dirs != [index]:
        raise ValueError(f"Require one shared name_counts_index directory: {index_dirs}")
    public = validate_release_root(root)
    incremental = validate_portable_source_bundle(root)
    pairwise = root / "pairwise"
    inputs = _read_json(pairwise / "inputs.json")
    if _local_path(pairwise, inputs["name_counts_index"], boundary=root) != index:
        raise ValueError("Pairwise inputs must use the shared name_counts_index")
    corrected = _local_path(pairwise, inputs["corrected_source_root"], boundary=root)
    corrected_source_file_count = _validate_corrected_source_inventory(root, corrected)
    if _local_path(pairwise, inputs["linker_source_bundle"], boundary=root) != root:
        raise ValueError("Pairwise inputs must reference the snapshot incremental bundle")
    model_plan = _read_json(pairwise / "model_plan.json")
    load_model_plan(pairwise / "model_plan.json")
    expected_model = {
        name: {role: spec for role, spec in files.items() if role in MODEL_ROLES}
        for name, files in inputs["sources"]["model"].items()
    }
    if model_plan["datasets"] != expected_model or model_plan["release_version"] != inputs["release_version"]:
        raise ValueError("Model plan differs from the corrected pairwise inputs")
    index_digest = sha256_file(index / "manifest.json")
    for mode, sources in inputs["sources"].items():
        collection = _local_path(pairwise, inputs["arrow"][mode], boundary=root)
        manifests = _dataset_manifests(collection)
        if set(manifests) != set(sources):
            raise ValueError(f"Pairwise source and Arrow dataset populations differ: {mode}")
        _validate_root_name_counts_binding(collection, authoritative_manifest_sha256=index_digest)
        for name, files in sources.items():
            resolved_files = {}
            for role, spec in files.items():
                path = _local_path(pairwise, spec["path"], boundary=root)
                resolved_files[role] = path
                if sha256_file(path) != spec["sha256"]:
                    raise ValueError(f"Pairwise source SHA-256 mismatch: {path}")
            if "unknown_signatures" in resolved_files:
                unknown = load_unknown_signatures(resolved_files)
                if unknown.path != resolved_files["unknown_signatures"]:
                    raise ValueError(f"Unknown-signature sidecar must remain beside source labels: {name}")
            verify_conversion_provenance(
                manifests[name].parent,
                expected_source_sha256={role: files[role]["sha256"] for role in ARROW_SOURCE_ROLES},
            )
            with ArrowDataset.open(
                manifests[name].parent, require_specter=True, require_name_counts_index=True
            ) as data:
                if data.name_counts_index is None or Path(data.name_counts_index.path).resolve() != index:
                    raise ValueError(f"Pairwise dataset must use the shared count index: {name}")
    return {
        "root": str(root),
        "release_version": inputs["release_version"],
        "corrected_datasets": sorted(path.name for path in corrected.iterdir() if path.is_dir()),
        "corrected_source_file_count": corrected_source_file_count,
        "model_datasets": list(inputs["sources"]["model"]),
        "evaluation_datasets": list(inputs["sources"]["evaluation"]),
        "public_dataset_count": public["dataset_manifest_count"],
        "incremental_dataset_count": public["replay_dataset_manifest_count"],
        "incremental_source_rows": incremental["selected_source_rows"],
        "name_counts_manifest_sha256": index_digest,
    }


def package_corrected_datasets(
    *,
    public_data_root: Path,
    corrected_source_root: Path,
    pairwise_inputs_root: Path,
    linker_source_bundle: Path,
    name_counts_index: Path,
    output_root: Path,
    corrected_replay_root: Path | None = None,
    corrected_replay_embeddings_root: Path | None = None,
    release_version: str | None = None,
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
    minimum_dataset_f1: float = 0.0,
    minimum_signature_weighted_f1: float = 0.0,
) -> dict[str, Any]:
    """Copy final records and prepared populations into a fresh dataset snapshot."""

    public, corrected, pairwise, linker, index, output = (
        path.resolve()
        for path in (
            public_data_root,
            corrected_source_root,
            pairwise_inputs_root,
            linker_source_bundle,
            name_counts_index,
            output_root,
        )
    )
    if output.exists():
        raise FileExistsError(output)
    if (corrected_replay_root is None) != (corrected_replay_embeddings_root is None):
        raise ValueError("Pass both current replay raw and embedding roots")
    replay_sources = (
        {}
        if corrected_replay_root is None or corrected_replay_embeddings_root is None
        else {
            "raw": corrected_replay_root.resolve(),
            "embeddings": corrected_replay_embeddings_root.resolve(),
        }
    )
    for source in (public, corrected, pairwise, linker, index, *replay_sources.values()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Dataset snapshot input and output directories must not overlap")
        _validate_copy_output(source, output)
    license_source = public / "LICENSE.txt"
    if not license_source.is_file():
        raise FileNotFoundError(license_source)
    grid = [_unit_float(value) for value in eps_grid]
    if not grid or len(grid) != len(set(grid)):
        raise ValueError("Require a nonempty EPS grid without duplicates")
    minimum_dataset_f1 = _unit_float(minimum_dataset_f1)
    minimum_signature_weighted_f1 = _unit_float(minimum_signature_weighted_f1)
    require_name_counts_index_artifact(
        index, context="corrected dataset package", producer_hint="pass the common index"
    )
    inputs = _read_json(pairwise / "inputs.json")
    version = inputs["release_version"] if release_version is None else release_version
    public_manifests, _old_replays, public_version = read_arrow_collection_root(public / "manifest.json")
    if public_version is not None and public_version != version:
        raise ValueError("Public data and pairwise release versions differ")
    for name in public_manifests:
        if Path(name).name != name or name in {".", ".."} or ":" in name or "\\" in name:
            raise ValueError(f"Invalid benchmark dataset name: {name}")
    index_digest = sha256_file(index / "manifest.json")
    _validate_root_name_counts_binding(
        public, authoritative_manifest_sha256=index_digest, dataset_manifests=public_manifests
    )
    _validate_root_name_counts_binding(linker, authoritative_manifest_sha256=index_digest)
    _validate_bundle_paths(linker, resolve_links=False)

    _copy_arrow_root_without_name_counts(linker, output)
    _copy_file(license_source, output / "LICENSE.txt")
    shutil.copytree(index, output / "name_counts_index")
    _rebind_dataset_manifests(output, name_counts_index=output / "name_counts_index")
    shutil.copyfile(output / "manifest.json", output / "replay_manifest.json")
    inventory: dict[str, str] = {}
    reusable = _copy_corrected_sources(corrected, output / "corrected_sources", snapshot=output, inventory=inventory)
    for role, source in replay_sources.items():
        _copy_current_tree(source, output / "corrected_replay" / role, snapshot=output, inventory=inventory)
    (output / "corrected_sources" / "manifest.json").write_text(
        json.dumps({"files": inventory}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    destination_pairwise = output / "pairwise"
    destination_pairwise.mkdir()
    for sources in inputs["sources"].values():
        for files in sources.values():
            for role, spec in files.items():
                source = _local_path(pairwise, spec["path"], boundary=pairwise)
                # Source label paths also determine the implicit uncertainty sidecar.
                target = (
                    reusable.get((source.parent.name, source.name, spec["sha256"]))
                    if role in {"papers", "specter_embeddings"}
                    else None
                )
                if target is None:
                    target = destination_pairwise / spec["path"]
                    _copy_file(source, target, expected_sha256=spec["sha256"])
                elif sha256_file(source) != spec["sha256"]:
                    raise ValueError(f"Source SHA-256 mismatch: {source}")
                spec["path"] = _relative_path(target, destination_pairwise)
    for mode, relative in inputs["arrow"].items():
        source = _local_path(pairwise, relative, boundary=pairwise)
        target = destination_pairwise / relative
        _copy_arrow_root_without_name_counts(source, target)
        _rebind_dataset_manifests(target, name_counts_index=output / "name_counts_index")
        inputs["arrow"][mode] = _relative_path(target, destination_pairwise)
    inputs.update(
        release_version=version,
        name_counts_index="../name_counts_index",
        corrected_source_root="../corrected_sources",
        linker_source_bundle="..",
        scope="Corrected pairwise inputs and exact evaluation populations",
    )
    (destination_pairwise / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    model_plan = {
        "release_version": version,
        "datasets": {
            name: {role: spec for role, spec in files.items() if role in MODEL_ROLES}
            for name, files in inputs["sources"]["model"].items()
        },
        "eps": {
            "grid": grid,
            "minimum_dataset_f1": minimum_dataset_f1,
            "minimum_signature_weighted_f1": minimum_signature_weighted_f1,
        },
    }
    (destination_pairwise / "model_plan.json").write_text(json.dumps(model_plan, indent=2) + "\n", encoding="utf-8")
    benchmarks = output / "benchmarks"
    for name, manifest_path in public_manifests.items():
        shutil.copytree(manifest_path.parent, benchmarks / name)
    _write_root_manifest(benchmarks, dataset_manifests={name: f"{name}/manifest.json" for name in public_manifests})
    _rebind_dataset_manifests(benchmarks, name_counts_index=output / "name_counts_index")
    _write_root_manifest(
        output,
        dataset_manifests={name: f"benchmarks/{name}/manifest.json" for name in public_manifests},
        replay_bundles={"incremental": "replay_manifest.json"},
        release_version=version,
    )
    return validate_corrected_datasets(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "public-data-root",
        "corrected-source-root",
        "pairwise-inputs-root",
        "linker-source-bundle",
        "name-counts-index",
        "output-root",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--release-version")
    parser.add_argument("--corrected-replay-root", type=Path)
    parser.add_argument("--corrected-replay-embeddings-root", type=Path)
    parser.add_argument("--eps-grid", type=float, nargs="+", default=DEFAULT_EPS_GRID)
    parser.add_argument("--minimum-dataset-f1", type=float, default=0.0)
    parser.add_argument("--minimum-signature-weighted-f1", type=float, default=0.0)
    print(json.dumps(package_corrected_datasets(**vars(parser.parse_args(argv))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
