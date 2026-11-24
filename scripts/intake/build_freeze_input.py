"""Build the role allocator input from exported signature evidence and source strata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import normalize_block_key
from scripts.common.identity_ledger import (
    EVIDENCE_SHA256_FIELD,
    MentionEvidence,
    evidence_identities,
    load_identity_evidence,
)

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


def mention_counts(rows: dict[str, MentionEvidence]) -> Counter[str]:
    """Count each distinct corpus position once per non-excluded identity."""
    mentions = {
        (row["corpus_paper_id"], row["position"], orcid)
        for row in rows.values()
        if not row["excluded"] and not row["reviewed_error"]
        for orcid in evidence_identities(row)
    }
    return Counter(orcid for _, _, orcid in mentions)


def _metadata(path: Path, blocks: set[str]) -> dict[str, tuple[str, str | None]]:
    parquet = pq.ParquetFile(path)
    columns = ["cluster_block_key", "split"]
    optional = [name for name in ("stratum", "script", "region", "script_region") if name in parquet.schema_arrow.names]
    result = {}
    for batch in parquet.iter_batches(columns=columns + optional):
        for item in batch.to_pylist():
            block = normalize_block_key(item["cluster_block_key"])
            if block not in blocks:
                continue
            if block in result or item["split"] not in {"development", "holdout"}:
                raise ValueError(f"Duplicate or invalid source split for {block!r}")
            region = item.get("script_region") or "/".join(
                item[name] for name in ("script", "region") if item.get(name)
            )
            result[block] = (item["split"], item.get("stratum") or region or None)
    if missing := blocks - result.keys():
        raise ValueError(f"Source split missing for exported blocks: {sorted(missing)}")
    return result


def build_freeze_input(
    evidence_path: Path, gold_blocks: Path, output: Path, *, strata: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Write signatures, source split, stratum, and both sources of identities per exported block."""
    evidence = load_identity_evidence(evidence_path)
    overrides = {normalize_block_key(block): stratum for block, stratum in (strata or {}).items()}
    if len(overrides) != len(strata or {}) or overrides.keys() - evidence.blocks.keys():
        raise ValueError("Strata overrides must name unique exported blocks")
    metadata = _metadata(gold_blocks, set(evidence.blocks))
    rows = []
    for block, entry in sorted(evidence.blocks.items()):
        identities = mention_counts(evidence.read_block(block))
        count = entry["signature_count"]
        size = "<=1000" if count <= 1000 else "1001-15000" if count <= 15000 else ">15000"
        source_split, region = metadata[block]
        rows.append(
            {
                "block_key": block,
                "signature_count": count,
                "orcids": sorted(identities),
                "orcid_mention_counts": sorted(identities.items()),
                "source_split": source_split,
                "stratum": overrides.get(block, f"{size}/{region}" if region else size),
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
    parser.add_argument("--strata-json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    strata = read_json_object(args.strata_json) if args.strata_json else None
    rows = build_freeze_input(args.identity_evidence, args.gold_blocks, args.output, strata=strata)
    print(json.dumps({"blocks": len(rows), "signatures": sum(row["signature_count"] for row in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
