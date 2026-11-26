"""Check the sealed test role against the 1.4 training signature inventory."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.text import normalize_orcid
from s2and.text.normalize import ORCID_PATTERN
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import FINAL_TEST_ROLE, load_block_roles
from scripts.common.identity_ledger import evidence_identities, load_identity_ledger


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


def check_training_overlap(
    roles_path: Path,
    training_paths: Sequence[Path],
    *,
    allow_no_training_orcids: bool = False,
    identity_evidence: Path,
) -> dict[str, Any]:
    """Count distinct signature, paper-position and ORCID overlaps for the sealed test role."""
    roles = load_block_roles(roles_path)
    ledger = load_identity_ledger(identity_evidence, roles)
    counts = read_json_object(roles_path).get("block_signature_counts")
    expected = {block for block, role in roles.block_to_role.items() if role == FINAL_TEST_ROLE}
    by_block = ledger.evidence.dataset_dirs(expected)
    if not isinstance(counts, dict) or any(
        type(counts.get(block)) is not int or counts[block] < 1 for block in expected
    ):
        raise ValueError("Re-freeze to record a valid signature count for every sealed block")
    if not training_paths or len({path.resolve() for path in training_paths}) != len(training_paths):
        raise ValueError("Supply distinct 1.4 training inventory shards")
    signatures: set[str] = set()
    positions: set[tuple[str, int]] = set()
    orcids: set[str] = set()
    for block, root in sorted(by_block.items()):
        block_rows = ledger.for_block(block, dataset_dir=root).rows
        block_signatures = set(block_rows)
        block_positions = {(row["corpus_paper_id"], row["position"]) for row in block_rows.values()}
        block_orcids = {orcid for row in block_rows.values() for orcid in evidence_identities(row)}
        if len(block_signatures) != counts[block]:
            raise ValueError(f"Sealed block {block!r} has {len(block_signatures)} signatures, expected {counts[block]}")
        if block_signatures & signatures:
            raise ValueError(f"Signature IDs occur in multiple sealed blocks including {block!r}")
        signatures.update(block_signatures)
        positions.update(block_positions)
        orcids.update(block_orcids | roles.block_orcids[block])
    if not signatures:
        raise ValueError("The sealed role must contain signatures")

    signature_overlap: set[str] = set()
    position_overlap: set[tuple[str, int]] = set()
    orcid_overlap: set[str] = set()
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
            signature_overlap.update(signatures & ids)
            position_overlap.update(positions & batch_positions)
            orcid_overlap.update(orcids & batch_orcids)
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
    compared_positions = sum(shard["rows_with_paper_position"] for shard in training)
    if compared_positions == 0:
        raise ValueError("The 1.4 training inventory contains no comparable corpus positions")
    return {
        "compared_corpus_positions": compared_positions,
        "block_role_assignments_sha256": roles.sha256,
        **ledger.bindings,
        "benchmark_overlap": "unobservable",
        "training_shards": training,
        "distinct_training_orcids": len(training_orcids),
        "passed": (bool(training_orcids) or allow_no_training_orcids)
        and not signature_overlap
        and not position_overlap
        and not orcid_overlap,
        "sealed_roles": {
            FINAL_TEST_ROLE: {
                "signatures": len(signatures),
                "orcids": len(orcids),
                "signature_overlap": len(signature_overlap),
                "paper_position_overlap": len(position_overlap),
                "orcid_overlap": len(orcid_overlap),
            }
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Report sealed overlap and fail when training evidence is insufficient."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--training-parquet", type=Path, action="append", required=True)
    parser.add_argument("--allow-no-training-orcids", action="store_true")
    args = parser.parse_args(argv)
    report = check_training_overlap(
        args.block_role_assignments,
        args.training_parquet,
        allow_no_training_orcids=args.allow_no_training_orcids,
        identity_evidence=args.identity_evidence,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
