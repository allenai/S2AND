"""Check both sealed test halves against the 1.4 training signature inventory."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from s2and.text.normalize import ORCID_PATTERN
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import FINAL_TEST_ROLES, load_block_roles, normalize_block_key


def canonical_id(value: object, field: str, path: Path) -> str:
    """Require an unscoped, canonical source ID."""
    if (
        not isinstance(value, str)
        or not value
        or any(char.isspace() for char in value)
        or value != value.lower()
        or "___" in value
        or ":" in value
        or re.fullmatch(r"[0-9]+[.]0+", value)
    ):
        raise ValueError(f"Invalid {field} {value!r} in {path}; expected a canonical source ID")
    return value


def checked_orcid(raw: object, path: Path) -> str | None:
    """Normalize known ORCID formats and reject malformed values."""
    if raw is None:
        return None
    if isinstance(raw, str):
        bare = raw
        for prefix in ("https://orcid.org/", "http://orcid.org/"):
            if bare.startswith(prefix):
                bare = bare.removeprefix(prefix)
                break
        if ORCID_PATTERN.fullmatch(bare) and (orcid := normalize_orcid(bare)):
            return orcid
    raise ValueError(f"Invalid ORCID {raw!r} in {path}")


def paper_position(paper: object, position: object, path: Path) -> tuple[str, int]:
    """Validate a corpus paper ID and integral author position."""
    if type(paper) is int and paper >= 0:
        paper = str(paper)
    if type(position) is not int or position < 0:
        raise ValueError(f"Invalid position {position!r} in {path}")
    return canonical_id(paper, "paper_id", path), position


def _read_sealed_block(root: Path, block: str) -> tuple[set[str], set[tuple[str, int]], set[str]]:
    signatures: set[str] = set()
    positions: set[tuple[str, int]] = set()
    orcids: set[str] = set()
    with (
        ArrowDataset.open(root) as dataset,
        dataset.use() as lease,
        lease.open_file("signatures") as infile,
        pa.PythonFile(infile, mode="r") as source,
    ):
        reader = ipc.open_file(source)
        columns = ["signature_id", "author_block", "paper_id", "author_position"]
        if "author_orcid" in reader.schema.names:
            columns.append("author_orcid")
        for index in range(reader.num_record_batches):
            for row in reader.get_batch(index).select(columns).to_pylist():
                raw = row["signature_id"]
                if normalize_block_key(row["author_block"]) != block:
                    raise ValueError(f"Unexpected block in {root}: expected {block!r}")
                if isinstance(raw, str) and raw.startswith("visibleq:"):
                    continue
                signature = canonical_id(raw, "signature_id", root)
                if signature in signatures:
                    raise ValueError(f"Duplicate signature {signature!r} in {root}")
                signatures.add(signature)
                positions.add(paper_position(row["paper_id"], row["author_position"], root))
                if orcid := checked_orcid(row.get("author_orcid"), root):
                    orcids.add(orcid)
    return signatures, positions, orcids


def check_training_overlap(
    roles_path: Path,
    sealed_blocks: Mapping[str, Path],
    training_paths: Sequence[Path],
    *,
    allow_no_training_orcids: bool = False,
) -> dict[str, Any]:
    """Count distinct signature, paper-position and ORCID overlaps for both sealed halves."""
    roles = load_block_roles(roles_path)
    counts = read_json_object(roles_path).get("block_signature_counts")
    by_block = {normalize_block_key(block): root for block, root in sealed_blocks.items()}
    expected = {block for block, role in roles.block_to_role.items() if role in FINAL_TEST_ROLES}
    if len(by_block) != len(sealed_blocks) or by_block.keys() != expected:
        raise ValueError(
            "The input must contain every sealed block exactly once. "
            f"Missing {sorted(expected - by_block.keys())}. Unexpected {sorted(by_block.keys() - expected)}."
        )
    if not isinstance(counts, dict) or any(
        type(counts.get(block)) is not int or counts[block] < 1 for block in expected
    ):
        raise ValueError("Re-freeze to record a valid signature count for every sealed block")
    if not training_paths or len({path.resolve() for path in training_paths}) != len(training_paths):
        raise ValueError("Supply distinct 1.4 training inventory shards")
    signatures = {role: set[str]() for role in FINAL_TEST_ROLES}
    positions = {role: set[tuple[str, int]]() for role in FINAL_TEST_ROLES}
    orcids = {role: set[str]() for role in FINAL_TEST_ROLES}
    for block, root in sorted(by_block.items()):
        role = roles.role_for(block)
        block_signatures, block_positions, block_orcids = _read_sealed_block(root, block)
        if len(block_signatures) != counts[block]:
            raise ValueError(f"Sealed block {block!r} has {len(block_signatures)} signatures, expected {counts[block]}")
        if any(block_signatures & seen for seen in signatures.values()):
            raise ValueError(f"Signature IDs occur in multiple sealed blocks including {block!r}")
        signatures[role].update(block_signatures)
        positions[role].update(block_positions)
        orcids[role].update(block_orcids | roles.block_orcids[block])
    if any(not ids for ids in signatures.values()):
        raise ValueError("Both sealed halves must contain signatures")

    signature_overlap = {role: set[str]() for role in FINAL_TEST_ROLES}
    position_overlap = {role: set[tuple[str, int]]() for role in FINAL_TEST_ROLES}
    orcid_overlap = {role: set[str]() for role in FINAL_TEST_ROLES}
    training_orcids: set[str] = set()
    training = []
    for path in training_paths:
        parquet = pq.ParquetFile(path)
        columns = ["signature_id", "orcid"]
        available = set(parquet.schema_arrow.names)
        if not set(columns) <= available:
            raise ValueError(f"Expected signature_id and orcid columns in {path}")
        position_columns = {"paper_id", "position"} & available
        if len(position_columns) == 1:
            raise ValueError(f"Provide both paper_id and position columns in {path}")
        columns.extend(sorted(position_columns))
        count = 0
        compared_positions = 0
        for batch in parquet.iter_batches(columns=columns):
            ids = {canonical_id(sid, "signature_id", path) for sid in batch.column("signature_id").to_pylist()}
            batch_orcids = {
                oid for raw in batch.column("orcid").to_pylist() if (oid := checked_orcid(raw, path)) is not None
            }
            batch_positions = set()
            if position_columns:
                for paper, position in zip(batch.column("paper_id"), batch.column("position"), strict=True):
                    pid, pos = paper.as_py(), position.as_py()
                    if pid is not None or pos is not None:
                        batch_positions.add(paper_position(pid, pos, path))
                        compared_positions += 1
            training_orcids.update(batch_orcids)
            for role in FINAL_TEST_ROLES:
                signature_overlap[role].update(signatures[role] & ids)
                position_overlap[role].update(positions[role] & batch_positions)
                orcid_overlap[role].update(orcids[role] & batch_orcids)
            count += batch.num_rows
        training.append(
            {
                "path": str(path),
                "rows": count,
                "rows_with_paper_position": compared_positions,
                "sha256": sha256_file(path),
            }
        )
    if not any(shard["rows"] for shard in training):
        raise ValueError("The 1.4 training inventory is empty")
    return {
        "block_role_assignments_sha256": roles.sha256,
        "training_shards": training,
        "distinct_training_orcids": len(training_orcids),
        "passed": (bool(training_orcids) or allow_no_training_orcids)
        and not any(signature_overlap.values())
        and not any(position_overlap.values())
        and not any(orcid_overlap.values()),
        "sealed_roles": {
            role: {
                "signatures": len(signatures[role]),
                "orcids": len(orcids[role]),
                "signature_overlap": len(signature_overlap[role]),
                "paper_position_overlap": len(position_overlap[role]),
                "orcid_overlap": len(orcid_overlap[role]),
            }
            for role in FINAL_TEST_ROLES
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Report sealed overlap and fail when training evidence is insufficient."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--blocks-json", type=Path, required=True, help="Map both halves' blocks to Arrow directories.")
    parser.add_argument("--training-parquet", type=Path, action="append", required=True)
    parser.add_argument("--allow-no-training-orcids", action="store_true")
    args = parser.parse_args(argv)
    raw = read_json_object(args.blocks_json)
    if any(not isinstance(path, str) or not path for path in raw.values()):
        raise ValueError("Expected a JSON object mapping sealed block keys to Arrow directories")
    blocks = {block: args.blocks_json.parent / path for block, path in raw.items()}
    report = check_training_overlap(
        args.block_role_assignments,
        blocks,
        args.training_parquet,
        allow_no_training_orcids=args.allow_no_training_orcids,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
