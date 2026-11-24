"""Check both sealed test halves against the 1.4 training signature inventory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from scripts.production.data.block_roles import FINAL_TEST_ROLES, load_block_roles, normalize_block_key


def _read_sealed_block(root: Path, block: str) -> tuple[set[str], set[str]]:
    signatures: set[str] = set()
    orcids: set[str] = set()
    with (
        ArrowDataset.open(root) as dataset,
        dataset.use() as lease,
        lease.open_file("signatures") as infile,
        pa.PythonFile(infile, mode="r") as source,
    ):
        reader = ipc.open_file(source)
        columns = ["signature_id", "author_block"]
        if "author_orcid" in reader.schema.names:
            columns.append("author_orcid")
        for index in range(reader.num_record_batches):
            for row in reader.get_batch(index).select(columns).to_pylist():
                signature = row["signature_id"]
                if not signature or normalize_block_key(row["author_block"]) != block:
                    raise ValueError(f"Invalid signature or unexpected block in {root}: expected {block!r}")
                if signature.startswith("visibleq:"):
                    continue
                if signature in signatures:
                    raise ValueError(f"Duplicate signature {signature!r} in {root}")
                signatures.add(signature)
                if orcid := normalize_orcid(row.get("author_orcid") or ""):
                    orcids.add(orcid)
    if not signatures:
        raise ValueError(f"Sealed block {block!r} has no real signatures")
    return signatures, orcids


def check_training_overlap(
    roles_path: Path, sealed_blocks: Mapping[str, Path], training_paths: Sequence[Path]
) -> dict[str, Any]:
    """Count distinct signature and ORCID overlaps for each sealed half.

    Args:
        roles_path: Frozen block assignments with per-block ORCIDs.
        sealed_blocks: Arrow roots for every block in both sealed halves.
        training_paths: 1.4 training inventory shards with string signature IDs and nullable ORCIDs.

    Returns:
        Counts per sealed half and digests of the role manifest and training shards.
    """
    roles = load_block_roles(roles_path)
    by_block = {normalize_block_key(block): root for block, root in sealed_blocks.items()}
    expected = {block for block, role in roles.block_to_role.items() if role in FINAL_TEST_ROLES}
    if len(by_block) != len(sealed_blocks) or by_block.keys() != expected:
        raise ValueError(
            "The input must contain every sealed block exactly once. "
            f"Missing {sorted(expected - by_block.keys())}. Unexpected {sorted(by_block.keys() - expected)}."
        )
    if not training_paths or len({path.resolve() for path in training_paths}) != len(training_paths):
        raise ValueError("Supply distinct 1.4 training inventory shards")
    signatures = {role: set[str]() for role in FINAL_TEST_ROLES}
    orcids = {role: set[str]() for role in FINAL_TEST_ROLES}
    for block, root in sorted(by_block.items()):
        role = roles.role_for(block)
        block_signatures, block_orcids = _read_sealed_block(root, block)
        if any(block_signatures & seen for seen in signatures.values()):
            raise ValueError(f"Signature IDs occur in multiple sealed blocks including {block!r}")
        signatures[role].update(block_signatures)
        orcids[role].update(block_orcids | roles.block_orcids[block])
    if any(not ids for ids in signatures.values()):
        raise ValueError("Both sealed halves must contain signatures")

    signature_overlap = {role: set[str]() for role in FINAL_TEST_ROLES}
    orcid_overlap = {role: set[str]() for role in FINAL_TEST_ROLES}
    training = []
    for path in training_paths:
        parquet = pq.ParquetFile(path)
        columns = ["signature_id", "orcid"]
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise ValueError(f"Expected signature_id and orcid columns in {path}")
        count = 0
        for batch in parquet.iter_batches(columns=columns):
            ids = batch.column("signature_id").to_pylist()
            row_orcids = batch.column("orcid").to_pylist()
            if any(not isinstance(sid, str) or not sid.strip() for sid in ids):
                raise ValueError(f"Expected nonempty string signature IDs in {path}")
            if any(orcid is not None and not isinstance(orcid, str) for orcid in row_orcids):
                raise ValueError(f"Expected nullable string ORCIDs in {path}")
            batch_ids = set(ids)
            batch_orcids = {oid for raw in row_orcids if raw and (oid := normalize_orcid(raw))}
            for role in FINAL_TEST_ROLES:
                signature_overlap[role].update(signatures[role] & batch_ids)
                orcid_overlap[role].update(orcids[role] & batch_orcids)
            count += len(ids)
        training.append({"path": str(path), "rows": count, "sha256": sha256_file(path)})
    if not any(shard["rows"] for shard in training):
        raise ValueError("The 1.4 training inventory is empty")
    return {
        "block_role_assignments_sha256": roles.sha256,
        "training_shards": training,
        "passed": not any(signature_overlap.values()) and not any(orcid_overlap.values()),
        "sealed_roles": {
            role: {
                "signatures": len(signatures[role]),
                "orcids": len(orcids[role]),
                "signature_overlap": len(signature_overlap[role]),
                "orcid_overlap": len(orcid_overlap[role]),
            }
            for role in FINAL_TEST_ROLES
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Report overlap and exit with status one when either half overlaps training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--blocks-json", type=Path, required=True, help="Map both halves' blocks to Arrow directories.")
    parser.add_argument("--training-parquet", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.blocks_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(path, str) or not path for path in raw.values()):
        raise ValueError("Expected a JSON object mapping sealed block keys to Arrow directories")
    blocks = {block: args.blocks_json.parent / path for block, path in raw.items()}
    report = check_training_overlap(args.block_role_assignments, blocks, args.training_parquet)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
