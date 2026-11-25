"""Build the role allocator input from exported signature evidence and source strata."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.common.block_roles import normalize_block_key
from scripts.common.identity_ledger import (
    EVIDENCE_SHA256_FIELD,
    MentionEvidence,
    evidence_identities,
    load_identity_evidence,
)
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS

SCHEMA = pa.schema(
    [
        ("block_key", pa.string()),
        ("signature_count", pa.int64()),
        ("orcids", pa.list_(pa.string())),
        ("orcid_mention_counts", pa.map_(pa.string(), pa.int64())),
        ("source_split", pa.string()),
        ("stratum", pa.string()),
    ]
)
GIANT_SIGNATURE_COUNT = 15_000


def size_band(count: int) -> str:
    """Return the intake size band for an exported signature count."""
    return "<=1000" if count <= 1000 else "1001-15000" if count <= GIANT_SIGNATURE_COUNT else ">15000"


def mention_counts(rows: dict[str, MentionEvidence]) -> Counter[str]:
    """Count each distinct corpus position once per non-excluded identity."""
    mentions = {
        (row["corpus_paper_id"], row["position"], orcid)
        for row in rows.values()
        if not row["reviewed_error"]
        for orcid in evidence_identities(row) - EXCLUDED_ORCID_GROUPS
    }
    return Counter(orcid for _, _, orcid in mentions)


def _source_splits(path: Path, blocks: set[str]) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    result = {}
    for batch in parquet.iter_batches(columns=["cluster_block_key", "split"]):
        for item in batch.to_pylist():
            block = normalize_block_key(item["cluster_block_key"])
            if block not in blocks:
                continue
            if block in result or item["split"] not in {"development", "holdout"}:
                raise ValueError(f"Duplicate or invalid source split for {block!r}")
            result[block] = item["split"]
    if missing := blocks - result.keys():
        raise ValueError(f"Source split missing for exported blocks: {sorted(missing)}")
    return result


def _read_origins(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"surname", "broad"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected surname and broad columns in {path}")
        origins = {}
        for row in reader:
            surname, broad = normalize_block_key(row["surname"]), (row["broad"] or "").strip()
            if not surname or not broad or "|" in broad or (surname in origins and origins[surname] != broad):
                raise ValueError(f"Invalid or conflicting origin for surname {surname!r} in {path}")
            origins[surname] = broad
        return origins


def build_freeze_input(
    evidence_path: Path, gold_blocks: Path, output: Path, *, surname_origin_csv: Path | None = None
) -> list[dict[str, Any]]:
    """Write signatures, source split, stratum, and both sources of identities per exported block."""
    evidence = load_identity_evidence(evidence_path)
    origins = _read_origins(surname_origin_csv) if surname_origin_csv is not None else {}
    source_splits = _source_splits(gold_blocks, set(evidence.blocks))
    rows = []
    for block, entry in sorted(evidence.blocks.items()):
        identities = mention_counts(evidence.read_block(block))
        count = entry["signature_count"]
        surname = block.partition(" ")[2]
        rows.append(
            {
                "block_key": block,
                "signature_count": count,
                "orcids": sorted(identities),
                "orcid_mention_counts": sorted(identities.items()),
                "source_split": source_splits[block],
                "stratum": f"{size_band(count)}|{origins.get(surname, 'other')}",
            }
        )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = SCHEMA.with_metadata(
        {
            EVIDENCE_SHA256_FIELD: evidence.sha256,
            "snapshot_sha256": evidence.snapshot_sha256,
            "intake_sha256": evidence.intake_sha256,
        }
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), output)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Generate the Parquet input for assign_block_roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", required=True, type=Path)
    parser.add_argument("--gold-blocks", required=True, type=Path)
    parser.add_argument("--surname-origin-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = build_freeze_input(
        args.identity_evidence, args.gold_blocks, args.output, surname_origin_csv=args.surname_origin_csv
    )
    print(json.dumps({"blocks": len(rows), "signatures": sum(row["signature_count"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
