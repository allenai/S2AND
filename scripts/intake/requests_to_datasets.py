"""Convert production Arrow request bundles into a training snapshot."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections.abc import Collection, Sequence
from pathlib import Path, PurePosixPath
from typing import TypedDict

from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset, build_arrow_artifact_manifest, write_arrow_artifact_manifest
from s2and.arrow_schema import validate_arrow_file_schema
from s2and.arrow_tables import write_raw_arrow_batch_lookup_indexes
from scripts.common.arrow_roots import (
    generic_dataset_manifests,
    read_json_object,
    resolved_manifest_artifact_path,
    write_root_manifest,
)
from scripts.common.block_roles import normalize_block_key

REQUEST_SUFFIX = ".arrow.tar.gz"
DATASET_TABLES = ("signatures", "papers", "paper_authors", "specter")
SOURCE_RECORD = "request_source.json"


class IntakeResult(TypedDict):
    source_bundle_sha256: str
    reused: bool


def request_dataset_name(bundle: Path) -> str:
    """Return the snapshot dataset name for a production bundle filename."""
    if not bundle.name.endswith(REQUEST_SUFFIX):
        raise ValueError(f"Expected a {REQUEST_SUFFIX} request bundle: {bundle}")
    name = normalize_block_key(bundle.name.removesuffix(REQUEST_SUFFIX)).replace(" ", "_")
    if not name or name in {".", ".."} or any(char in name for char in "/\\:"):
        raise ValueError(f"Invalid request bundle block name: {bundle}")
    return name


def extract_request_tables(bundle: Path, target: Path, tables: Collection[str]) -> dict[str, str]:
    """Copy selected regular Arrow members to fixed paths and validate their schemas."""
    expected = {f"{table}.arrow": table for table in tables}
    paths: dict[str, str] = {}
    with tarfile.open(bundle, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name).name
            if name not in expected:
                continue
            table = expected[name]
            if not member.isfile() or table in paths:
                raise ValueError(f"Expected one regular {name} member in {bundle}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not read {member.name} from {bundle}")
            destination = target / name
            with source, destination.open("xb") as handle:
                shutil.copyfileobj(source, handle, length=1 << 22)
            validate_arrow_file_schema(str(destination), table_name=table)
            paths[table] = str(destination)
    if missing := set(tables) - paths.keys():
        raise ValueError(f"Missing tables in {bundle}: {sorted(missing)}")
    return paths


def _open_dataset(root: Path) -> None:
    with ArrowDataset.open(root, require_specter=True, require_name_counts_index=True):
        pass


def convert_requests(bundles: Sequence[Path], root: Path, name_counts_index: Path) -> dict[str, IntakeResult]:
    """Validate and publish a collection of request datasets, reusing completed blocks."""
    by_name = {request_dataset_name(bundle): bundle for bundle in bundles}
    if not by_name or len(by_name) != len(bundles):
        raise ValueError("Supply at least one request bundle and no duplicate block names")
    root, name_counts_index = root.resolve(), name_counts_index.resolve()
    with exclusive_file_lock(root / ".intake.lock"):
        return _convert_datasets(by_name, root, name_counts_index)


def _convert_datasets(by_name: dict[str, Path], root: Path, name_counts_index: Path) -> dict[str, IntakeResult]:
    index_digest = sha256_file(name_counts_index / "manifest.json")
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    published = generic_dataset_manifests(root) if (root / "manifest.json").exists() else {}
    manifests = {name: path.relative_to(root).as_posix() for name, path in published.items()}
    for path in published.values():
        index = resolved_manifest_artifact_path(
            path, read_json_object(path)["paths"]["name_counts_index"], key="name_counts_index"
        )
        if index != name_counts_index:
            raise ValueError(f"Existing dataset uses a different name-counts index: {path}")

    results: dict[str, IntakeResult] = {}
    for name, bundle in sorted(by_name.items()):
        target = datasets / name
        manifest = target / "manifest.json"
        digest = sha256_file(bundle)
        record = {"source_bundle_sha256": digest, "name_counts_index_sha256": index_digest}
        reused = target.exists()
        if reused:
            if read_json_object(target / SOURCE_RECORD) != {**record, "dataset_manifest_sha256": sha256_file(manifest)}:
                raise ValueError(f"Existing dataset has different inputs or manifest: {target}")
            index = resolved_manifest_artifact_path(
                manifest, read_json_object(manifest)["paths"]["name_counts_index"], key="name_counts_index"
            )
            if index != name_counts_index:
                raise ValueError(f"Existing dataset uses a different name-counts index: {target}")
            _open_dataset(target)
        else:
            staging = Path(tempfile.mkdtemp(dir=datasets, prefix=f".{name}."))
            paths = extract_request_tables(bundle, staging, DATASET_TABLES)
            paths, _ = write_raw_arrow_batch_lookup_indexes(paths, staging)
            paths["name_counts_index"] = str(name_counts_index)
            staged_manifest = write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, staging), staging)
            _open_dataset(staging)
            record["dataset_manifest_sha256"] = sha256_file(staged_manifest)
            (staging / SOURCE_RECORD).write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            staging.rename(target)
        manifests[name] = manifest.relative_to(root).as_posix()
        results[name] = {"source_bundle_sha256": digest, "reused": reused}

    write_root_manifest(root, dataset_manifests=manifests)
    return results


def main(argv: list[str] | None = None) -> int:
    """Convert one directory or an explicit list of bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name-counts-index", required=True, type=Path)
    args = parser.parse_args(argv)
    bundles = args.requests
    if len(bundles) == 1 and bundles[0].is_dir():
        bundles = sorted(bundles[0].glob(f"*{REQUEST_SUFFIX}"))
    print(json.dumps(convert_requests(bundles, args.output_root, args.name_counts_index), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
