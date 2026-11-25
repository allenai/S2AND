"""Resolve blind gate papers from the frozen Arrow dataset and production seeds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from scripts.common.arrow_roots import generic_dataset_manifests, read_json_object
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, BlockRoles, normalize_block_key
from scripts.common.identity_ledger import IdentityEvidence, IdentityLedger, evidence_identities
from scripts.common.linker_seeds import PRODUCTION_MAX_EXEMPLARS, members_by_component, read_frozen_seeds


def _paper_items(arrow: ArrowDataset, signature_ids: set[str]) -> dict[str, dict[str, Any]]:
    def read_rows(key: str, id_column: str, ids: set[str], columns: list[str]) -> list[dict[str, Any]]:
        with arrow.use() as lease, lease.open_file(key) as file, pa.PythonFile(file, mode="r") as source:
            table = pa.ipc.open_file(source).read_all()
            table = table.select([column for column in columns if column in table.column_names])
            return table.filter(
                pc.is_in(table[id_column], value_set=pa.array(sorted(ids), type=pa.string()))
            ).to_pylist()

    names = ["author_first", "author_middle", "author_last", "author_suffix"]
    signatures = read_rows(
        "signatures", "signature_id", signature_ids, ["signature_id", "paper_id", *names, "author_affiliations"]
    )
    if {row["signature_id"] for row in signatures} != signature_ids:
        raise ValueError("Frozen query or exemplar signatures are missing from the Arrow dataset")
    paper_ids = {row["paper_id"] for row in signatures}
    papers = {
        row["paper_id"]: row
        for row in read_rows("papers", "paper_id", paper_ids, ["paper_id", "title", "abstract", "venue", "year"])
    }
    authors: dict[str, list[str]] = defaultdict(list)
    author_rows = read_rows("paper_authors", "paper_id", paper_ids, ["paper_id", "position", "author_name"])
    for row in sorted(author_rows, key=lambda row: (row["paper_id"], row["position"])):
        authors[row["paper_id"]].append(row["author_name"])
    return {
        row["signature_id"]: {
            **papers[row["paper_id"]],
            "signature_id": row["signature_id"],
            "authors": authors[row["paper_id"]],
            "author_name": " ".join(row[field] for field in names if row[field]),
            "affiliations": row.get("author_affiliations"),
        }
        for row in signatures
    }


def frozen_packet_papers(
    signature_ids_by_block: Mapping[str, set[str]], *, snapshot: Path | None, evidence: IdentityEvidence | None
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load every displayed paper from the frozen Arrow collection."""
    if evidence is not None:
        if snapshot is not None and snapshot.resolve() != evidence.snapshot_root.resolve():
            raise ValueError("Packet snapshot differs from the identity evidence snapshot")
        snapshot = evidence.snapshot_root
    if snapshot is None:
        raise ValueError("Blind packet displays require --snapshot or ledger-backed identity evidence")
    manifests = generic_dataset_manifests(snapshot)
    by_block = {normalize_block_key(dataset): path for dataset, path in manifests.items()}
    if len(by_block) != len(manifests):
        raise ValueError("Duplicate canonical block keys in the packet snapshot")
    papers = {}
    for block, signatures in signature_ids_by_block.items():
        if block not in by_block:
            raise ValueError(f"Packet block {block!r} is absent from the frozen snapshot")
        manifest = by_block[block]
        if evidence is not None and sha256_file(manifest) != evidence.blocks[block]["dataset_manifest_sha256"]:
            raise ValueError(f"Packet snapshot generation differs from the identity evidence for {block!r}")
        with ArrowDataset.open(manifest.parent) as arrow:
            papers.update(((block, sid), paper) for sid, paper in _paper_items(arrow, signatures).items())
    return papers


def resolve_gate_exemplars(
    items: Sequence[Mapping[str, Any]],
    roles: BlockRoles,
    task_manifest_path: Path | None,
    *,
    control_ledger: IdentityLedger | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    """Resolve frozen seed membership and prefer verified control exemplar references."""
    by_block: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        block = normalize_block_key(item["block_key"])
        if roles.role_for(block) == "gate_thresh":
            by_block[block].append(item)
    if not by_block:
        return list(items), {}
    if task_manifest_path is None:
        raise ValueError("Ledger-backed gate packets require --linker-task-manifest")
    manifest = read_json_object(task_manifest_path)
    blocks = manifest["tables"]["gate_thresh"]["blocks"]
    bindings = manifest[BLOCK_BINDINGS_FIELD]
    if any(bindings.get(block) != binding for block, binding in roles.bindings_for(by_block).items()):
        raise ValueError("Frozen gate task block provenance does not match the identity ledger")
    resolved: dict[int, Mapping[str, Any]] = {}
    seed_digests = {}
    for block, block_items in by_block.items():
        spec = blocks[block]
        members = members_by_component(read_frozen_seeds(spec))
        keys = {str(candidate["candidate_key"]) for item in block_items for candidate in item["candidates"]}
        if missing := keys - members.keys():
            raise ValueError(f"Gate candidates have no frozen seed members in {block!r}: {sorted(missing)}")
        if control_ledger is None:
            exemplar_ids = {key: members[key][:PRODUCTION_MAX_EXEMPLARS] for key in keys}
        else:
            rows = control_ledger.for_block(block).rows
            # Eligible controls have one component identity. Preserve seed order within each group.
            exemplar_ids = {
                key: sorted(members[key], key=lambda sid: not evidence_identities(rows[sid]))[:PRODUCTION_MAX_EXEMPLARS]
                for key in keys
            }
        with ArrowDataset.open(spec["arrow_dir"]) as arrow:
            if arrow.generation_id != spec["arrow_generation_id"]:
                raise ValueError(f"Frozen linker Arrow generation changed for {block!r}")
        for item in block_items:
            candidates = []
            for candidate in item["candidates"]:
                key = str(candidate["candidate_key"])
                candidates.append(
                    {
                        "candidate_key": key,
                        "member_signature_ids": members[key],
                        "exemplars": [{"signature_id": sid} for sid in exemplar_ids[key]],
                    }
                )
            resolved[id(item)] = {**item, "candidates": candidates}
        seed_digests[block] = spec["seeds_sha256"]
    return [resolved.get(id(item), item) for item in items], seed_digests
