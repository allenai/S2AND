"""Convert production Arrow request bundles into a training snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Collection, Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc

from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import is_lowercase_sha256, sha256_file
from s2and.arrow_inputs import (
    ArrowDataset,
    build_arrow_artifact_manifest,
    require_name_counts_index_artifact,
    write_arrow_artifact_manifest,
)
from s2and.arrow_schema import validate_arrow_file_schema
from s2and.arrow_tables import write_raw_arrow_batch_lookup_indexes
from scripts.common.arrow_roots import (
    _replace_json,
    generic_dataset_manifests,
    read_json_object,
    resolved_manifest_artifact_path,
    write_root_manifest,
)
from scripts.common.block_roles import normalize_block_key
from scripts.common.email_redaction import redact_source_tree, scan_tree_for_plaintext_emails
from scripts.common.identity_ledger import SNAPSHOT_BINDING as SNAPSHOT_BINDING
from scripts.intake.gate_public_snapshot import REDACTED_ABSTRACT_MARKER, load_paper_id_set

REQUEST_SUFFIX = ".arrow.tar.gz"
DATASET_TABLES = ("signatures", "papers", "paper_authors", "specter")
SOURCE_RECORD = "request_source.json"
INTAKE_LOCK = ".intake.lock"


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


def _scan_dataset(root: Path) -> None:
    if scan_tree_for_plaintext_emails(root, max_findings=1):
        raise ValueError(f"Plaintext email remains in snapshot dataset: {root.name}")


def _validate_block(path: Path, name: str) -> None:
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        keys = {
            normalize_block_key(value)
            for batch in range(reader.num_record_batches)
            for value in pc.unique(reader.get_batch(batch).column("author_block")).to_pylist()
        }
    if keys != {normalize_block_key(name)}:
        raise ValueError(f"Bundle {name!r} does not match its author_block values: {sorted(keys)}")


def _gate_tables(
    paths: dict[str, str], allowed: frozenset[str], abstract_allowed: frozenset[str]
) -> tuple[int, set[str]]:
    paper_ids = pa.array(sorted(allowed), type=pa.string())
    abstract_ids = pa.array(sorted(abstract_allowed), type=pa.string())
    signatures = 0
    filtered_papers: set[str] = set()
    for name, filename in paths.items():
        path = Path(filename)
        gated = path.with_suffix(".gated")
        with pa.memory_map(filename, "r") as source:
            reader = pa.ipc.open_file(source)
            with pa.OSFile(str(gated), "wb") as output, pa.ipc.new_file(output, reader.schema) as writer:
                for index in range(reader.num_record_batches):
                    batch = reader.get_batch(index)
                    keep = pc.is_in(batch.column("paper_id"), value_set=paper_ids)
                    if name == "papers":
                        removed = pc.unique(batch.column("paper_id").filter(pc.invert(keep))).to_pylist()
                        filtered_papers.update(paper.strip() for paper in removed if paper)
                    batch = batch.filter(keep)
                    if name == "signatures":
                        signatures += batch.num_rows
                    if name == "papers" and "abstract" in batch.schema.names:
                        field = batch.schema.get_field_index("abstract")
                        values = batch.column(field)
                        redact = pc.and_kleene(
                            pc.invert(pc.is_in(batch.column("paper_id"), value_set=abstract_ids)),
                            pc.not_equal(values, ""),
                        )
                        batch = batch.set_column(
                            field, "abstract", pc.if_else(redact, REDACTED_ABSTRACT_MARKER, values)
                        )
                    writer.write_batch(batch)
        gated.replace(path)
    return signatures, filtered_papers


def _permission_digest(papers: frozenset[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(papers), separators=(",", ":")).encode("utf-8")).hexdigest()


def convert_requests(
    bundles: Sequence[Path],
    root: Path,
    name_counts_index: Path,
    *,
    allowed_papers: frozenset[str],
    abstract_allowed_papers: frozenset[str],
) -> dict[str, IntakeResult]:
    """Validate and publish a collection of request datasets, reusing completed blocks."""
    by_name = {request_dataset_name(bundle): bundle for bundle in bundles}
    if not by_name or len(by_name) != len(bundles):
        raise ValueError("Supply at least one request bundle and no duplicate block names")
    root, name_counts_index = root.resolve(), name_counts_index.resolve()
    index_digest = sha256_file(name_counts_index / "manifest.json")
    permissions = {"s2ag": allowed_papers, "abstracts": abstract_allowed_papers}
    permission_sha256 = {key: _permission_digest(papers) for key, papers in permissions.items()}
    with exclusive_file_lock(root / INTAKE_LOCK):
        binding_path = root / SNAPSHOT_BINDING
        prior = {}
        if binding_path.exists() or (root / "manifest.json").exists():
            prior = read_json_object(binding_path)
            if prior.get("name_counts_index_sha256") != index_digest or "snapshot_id" not in prior:
                raise ValueError("Existing snapshot has different name-counts index or lacks its intake binding")
            for key, papers in permissions.items():
                prior_papers = frozenset(prior["permission_papers"][key])
                if _permission_digest(prior_papers) != prior["permission_sha256"][key] or not prior_papers <= papers:
                    raise ValueError(f"Existing snapshot requires a superset of its prior {key} paper permissions")
            snapshot_id = prior["snapshot_id"]
        else:
            snapshot_id = uuid4().hex
        require_name_counts_index_artifact(
            name_counts_index, context="request intake", producer_hint="Supply the production index", verify=True
        )
        binding = {
            "snapshot_id": snapshot_id,
            "name_counts_index_sha256": index_digest,
            "permission_sha256": permission_sha256,
        }
        intake = {**prior, **binding, "permission_papers": {key: sorted(papers) for key, papers in permissions.items()}}
        results = _convert_datasets(
            by_name,
            root,
            name_counts_index,
            binding,
            allowed_papers,
            abstract_allowed_papers,
            initial_intake=None if prior else intake,
        )
        if prior and prior != intake:
            _replace_json(binding_path, intake)
        return results


def _convert_datasets(
    by_name: dict[str, Path],
    root: Path,
    name_counts_index: Path,
    binding: dict[str, Any],
    allowed_papers: frozenset[str],
    abstract_allowed_papers: frozenset[str],
    *,
    initial_intake: dict[str, Any] | None,
) -> dict[str, IntakeResult]:
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    published = generic_dataset_manifests(root) if (root / "manifest.json").exists() else {}
    manifests = {name: path.relative_to(root).as_posix() for name, path in published.items()}

    def verify_retained(manifest: Path) -> dict[str, Any]:
        record = read_json_object(manifest.parent / SOURCE_RECORD)
        permissions = record.get("permission_sha256", {})
        if (
            record.get("snapshot_id") != binding["snapshot_id"]
            or record.get("name_counts_index_sha256") != binding["name_counts_index_sha256"]
            or permissions.keys() != binding["permission_sha256"].keys()
            or not all(is_lowercase_sha256(digest) for digest in permissions.values())
            or record.get("dataset_manifest_sha256") != sha256_file(manifest)
        ):
            raise ValueError(f"Existing dataset has different inputs or manifest: {manifest.parent}")
        index = resolved_manifest_artifact_path(
            manifest, read_json_object(manifest)["paths"]["name_counts_index"], key="name_counts_index"
        )
        if index != name_counts_index:
            raise ValueError(f"Existing dataset uses a different name-counts index: {manifest}")
        _open_dataset(manifest.parent)
        _scan_dataset(manifest.parent)
        return record

    retained = {name: verify_retained(path) for name, path in published.items()}
    results: dict[str, IntakeResult] = {}
    with ExitStack() as staged:
        pending = {}
        for name, bundle in sorted(by_name.items()):
            target = datasets / name
            manifest = target / "manifest.json"
            digest = sha256_file(bundle)
            record = {**binding, "source_bundle_sha256": digest}
            reused = target.exists()
            if reused:
                prior = retained[name] if name in retained else verify_retained(manifest)
                if prior["source_bundle_sha256"] != digest:
                    raise ValueError(f"Existing dataset has a different source bundle: {target}")
                if name not in retained and prior["permission_sha256"] != binding["permission_sha256"]:
                    raise ValueError(f"Unpublished dataset has different paper permissions: {target}")
            else:
                staging = Path(staged.enter_context(tempfile.TemporaryDirectory(dir=datasets, prefix=f".{name}.")))
                paths = extract_request_tables(bundle, staging, DATASET_TABLES)
                _validate_block(Path(paths["signatures"]), name)
                signatures, filtered_papers = _gate_tables(paths, allowed_papers, abstract_allowed_papers)
                if not signatures:
                    raise ValueError(f"No signatures remain after paper permission filtering for {name!r}")
                redact_source_tree(staging)
                _scan_dataset(staging)
                paths, _ = write_raw_arrow_batch_lookup_indexes(paths, staging)
                paths["name_counts_index"] = str(name_counts_index)
                staged_manifest = write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, staging), staging)
                _open_dataset(staging)
                record["dataset_manifest_sha256"] = sha256_file(staged_manifest)
                record["permission_filtered_papers"] = sorted(filtered_papers)
                _replace_json(staging / SOURCE_RECORD, record)
                pending[staging] = target
            manifests[name] = manifest.relative_to(root).as_posix()
            results[name] = {"source_bundle_sha256": digest, "reused": reused}
        if initial_intake is not None:
            # Publish the receipt only once all conversions have succeeded.
            _replace_json(root / SNAPSHOT_BINDING, initial_intake)
        for staging, target in pending.items():
            staging.rename(target)

    write_root_manifest(root, dataset_manifests=manifests)
    return results


def main(argv: list[str] | None = None) -> int:
    """Convert one directory or an explicit list of bundles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name-counts-index", required=True, type=Path)
    parser.add_argument("--allowed-papers", required=True, type=Path)
    parser.add_argument("--abstract-allowed-papers", required=True, type=Path)
    args = parser.parse_args(argv)
    bundles = args.requests
    if len(bundles) == 1 and bundles[0].is_dir():
        bundles = sorted(bundles[0].glob(f"*{REQUEST_SUFFIX}"))
    results = convert_requests(
        bundles,
        args.output_root,
        args.name_counts_index,
        allowed_papers=load_paper_id_set(args.allowed_papers),
        abstract_allowed_papers=load_paper_id_set(args.abstract_allowed_papers),
    )
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
