"""Join gold and deposited identities onto a manifest-backed Arrow snapshot."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.text import normalize_orcid
from scripts.common.arrow_roots import _replace_json, generic_dataset_manifests, read_json_object
from scripts.common.block_roles import normalize_block_key
from scripts.common.identity_ledger import EvidenceBlock, GoldEvidence, MentionEvidence, evidence_identities
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS
from scripts.intake.prepare_identity_labels import (
    SignatureJoin,
    _normalize_paper_id,
    _optional_str,
    _read_arrow_signatures,
    _strict_nonneg_int,
    canonical_gold_block_filter,
)
from scripts.intake.requests_to_datasets import SNAPSHOT_BINDING

EVIDENCE_SCHEMA = pa.schema(
    [
        ("block_key", pa.string()),
        ("signature_id", pa.string()),
        ("corpus_paper_id", pa.string()),
        ("position", pa.int64()),
        ("gold", pa.list_(pa.struct([("orcid", pa.string()), ("scored", pa.bool_()), ("split", pa.string())]))),
        ("deposited_orcid", pa.string()),
        ("excluded", pa.bool_()),
        ("reviewed_error", pa.bool_()),
        ("evidence_state", pa.string()),
    ]
)


def _read_gold(gold: Path, blocks: Collection[str]) -> dict[str, list[pa.RecordBatch]]:
    source = ds.dataset(gold, format="parquet")
    required = {"cluster_block_key", "corpus_paper_id", "position", "orcid", "split", "scored"}
    if missing := required - set(source.schema.names):
        raise ValueError(f"Gold evidence is missing columns: {sorted(missing)}")
    names = {"author_first", "author_middle", "author_last", "first", "middle", "last"}
    columns = sorted(required | (names & set(source.schema.names)))
    selected = source.scanner(columns=columns, filter=canonical_gold_block_filter("cluster_block_key", blocks))
    partitions: dict[str, list[pa.RecordBatch]] = defaultdict(list)
    for batch in selected.to_batches():
        keys = batch.column("cluster_block_key")
        for raw_key in pc.unique(keys).to_pylist():
            block = normalize_block_key(raw_key)
            if block in blocks:
                partitions[block].append(batch.filter(pc.equal(keys, raw_key)))
    return partitions


def _normalized_orcid(value: Any) -> str | None:
    text = _optional_str(value)
    return normalize_orcid(text) if text else None


def _block_rows(
    directory: Path, block: str, gold: list[pa.RecordBatch], reviewed_errors: Collection[str]
) -> tuple[list[MentionEvidence], Counter[str]]:
    join = SignatureJoin(_read_arrow_signatures(directory))
    by_signature: dict[str, list[GoldEvidence]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for batch in gold:
        for item in batch.to_pylist():
            counts["gold_rows_scanned"] += 1
            paper = _normalize_paper_id(item["corpus_paper_id"])
            position = _strict_nonneg_int(item["position"])
            orcid = _normalized_orcid(item["orcid"])
            if paper is None or position is None or orcid is None:
                counts["invalid_gold_key_or_orcid"] += 1
                continue
            matched, reason = join.match(
                paper,
                position,
                normalize_block_key(item["cluster_block_key"]),
                (
                    _optional_str(item.get("author_first", item.get("first"))),
                    _optional_str(item.get("author_middle", item.get("middle"))),
                    _optional_str(item.get("author_last", item.get("last"))),
                ),
            )
            if matched is None:
                counts[reason] += 1
                continue
            by_signature[matched["signature_id"]].append(
                {"orcid": orcid, "scored": item["scored"] is True, "split": _optional_str(item["split"])}
            )

    known = {row["signature_id"] for row in join.rows}
    if len(known) != len(join.rows) or set(reviewed_errors) - known:
        raise ValueError(f"Duplicate signatures or unknown reviewed errors in {block!r}")
    rows: list[MentionEvidence] = []
    for raw in join.rows:
        sid = raw["signature_id"]
        paper = _normalize_paper_id(raw["paper_id"])
        position = _strict_nonneg_int(raw["author_position"])
        if not sid or paper is None or position is None or raw["block_norm"] != block:
            raise ValueError(f"Invalid Arrow signature key or author_block in {block!r}: {sid!r}")
        row: MentionEvidence = {
            "block_key": block,
            "signature_id": sid,
            "corpus_paper_id": paper,
            "position": position,
            "gold": by_signature.get(sid, []),
            "deposited_orcid": _normalized_orcid(raw.get("author_orcid")),
            "excluded": False,
            "reviewed_error": sid in reviewed_errors,
            "evidence_state": "none",
        }
        identities = evidence_identities(row)
        state: Literal["none", "single", "conflict"] = (
            "none" if not identities else "single" if len(identities) == 1 else "conflict"
        )
        row["evidence_state"] = state
        row["excluded"] = not identities.isdisjoint(EXCLUDED_ORCID_GROUPS)
        counts[state] += 1
        counts["excluded"] += row["excluded"]
        counts["reviewed_error"] += row["reviewed_error"]
        counts["deposited_only"] += row["deposited_orcid"] is not None and not row["gold"]
        counts["unscored_gold"] += bool(row["gold"]) and not any(entry["scored"] for entry in row["gold"])
        rows.append(row)
    return rows, counts


def build_identity_evidence(
    snapshot: Path, gold: Path, output: Path, reviewed_errors: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    """Project and scan gold once, then write one verified evidence partition per exported block."""
    datasets = generic_dataset_manifests(snapshot)
    blocks = {normalize_block_key(name): (name, path) for name, path in datasets.items()}
    if len(blocks) != len(datasets):
        raise ValueError("Duplicate canonical block keys in snapshot")
    errors = {normalize_block_key(key): values for key, values in (reviewed_errors or {}).items()}
    if len(errors) != len(reviewed_errors or {}) or errors.keys() - blocks.keys():
        raise ValueError("Reviewed errors must name unique exported blocks")
    binding = read_json_object(snapshot / SNAPSHOT_BINDING)
    if not binding.get("permission_sha256"):
        raise ValueError("Identity evidence requires a permission-gated intake snapshot")
    snapshot_digest = sha256_file(snapshot / "manifest.json")
    intake_digest = sha256_file(snapshot / SNAPSHOT_BINDING)
    if output.exists():
        raise FileExistsError(output)
    gold_partitions = _read_gold(gold, blocks)
    output.mkdir(parents=True)
    metadata: dict[str, EvidenceBlock] = {}
    audits: dict[str, dict[str, int]] = {}
    for index, (block, (dataset, path)) in enumerate(sorted(blocks.items())):
        manifest = read_json_object(path)
        if sha256_file(path.parent / "signatures.arrow") != manifest["files"]["signatures"]["sha256"]:
            raise ValueError(f"Snapshot signature digest mismatch: {block!r}")
        rows, audit = _block_rows(path.parent, block, gold_partitions.pop(block, []), errors.get(block, ()))
        destination = output / f"block-{index:06d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=EVIDENCE_SCHEMA), destination)
        metadata[block] = {
            "path": destination.name,
            "sha256": sha256_file(destination),
            "signature_count": len(rows),
            "dataset_manifest_sha256": sha256_file(path),
            "dataset": dataset,
        }
        audits[block] = dict(audit)
    result = {
        "format_version": 1,
        "snapshot_sha256": snapshot_digest,
        "intake_sha256": intake_digest,
        "name_counts_index_sha256": binding["name_counts_index_sha256"],
        "blocks": metadata,
        "audit": audits,
    }
    _replace_json(output / "manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Build evidence for all canonical blocks in a snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--gold-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reviewed-errors-json", type=Path)
    args = parser.parse_args(argv)
    reviewed = read_json_object(args.reviewed_errors_json) if args.reviewed_errors_json else None
    result = build_identity_evidence(args.snapshot, args.gold_parquet, args.output_dir, reviewed)
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
