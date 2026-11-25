"""Join gold and deposited identities onto a manifest-backed Arrow snapshot."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import sha256_file
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.text import canonicalize_name_parts, normalize_orcid
from scripts.common.arrow_roots import _replace_json, generic_dataset_manifests, read_json_object
from scripts.common.block_roles import normalize_block_key
from scripts.common.identity_ledger import (
    EvidenceBlock,
    GoldEvidence,
    MentionEvidence,
    RejectedGoldEvidence,
    evidence_identities,
    load_identity_evidence,
)
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, excluded_orcid_groups_digest
from scripts.intake.prepare_identity_labels import (
    SignatureJoin,
    _normalize_paper_id,
    _optional_str,
    _read_arrow_signatures,
    _strict_nonneg_int,
    canonical_gold_block_filter,
)
from scripts.intake.requests_to_datasets import INTAKE_LOCK, SNAPSHOT_BINDING, SOURCE_RECORD

_GOLD_FIELDS = [("orcid", pa.string()), ("scored", pa.bool_()), ("split", pa.string())]

DEFAULT_MAX_UNMATCHED_GOLD = 0.02

EVIDENCE_SCHEMA = pa.schema(
    [
        ("block_key", pa.string()),
        ("signature_id", pa.string()),
        ("corpus_paper_id", pa.string()),
        ("position", pa.int64()),
        ("gold", pa.list_(pa.struct(_GOLD_FIELDS))),
        ("rejected_gold", pa.list_(pa.struct([*_GOLD_FIELDS, ("reason", pa.string())]))),
        ("deposited_orcid", pa.string()),
        ("excluded", pa.bool_()),
        ("reviewed_error", pa.bool_()),
        ("evidence_state", pa.string()),
    ]
)


def _read_gold(gold: Path, joins: dict[str, SignatureJoin]) -> dict[str, list[dict[str, Any]]]:
    source = ds.dataset(gold, format="parquet")
    required = {"cluster_block_key", "corpus_paper_id", "position", "orcid", "split", "scored"}
    if missing := required - set(source.schema.names):
        raise ValueError(f"Gold evidence is missing columns: {sorted(missing)}")
    names = {"author_first", "author_middle", "author_last", "first", "middle", "last"}
    columns = sorted(required | (names & set(source.schema.names)))
    positions: dict[tuple[str, int], set[str]] = defaultdict(set)
    for block, join in joins.items():
        for key in join.positions:
            positions[key].add(block)
    papers = {paper for paper, _ in positions}
    paper_type = source.schema.field("corpus_paper_id").type
    predicate = canonical_gold_block_filter("cluster_block_key", joins)
    if pa.types.is_integer(paper_type) or pa.types.is_floating(paper_type):
        values = pa.array(sorted(int(paper) for paper in papers if paper.isdecimal()), type=paper_type)
        predicate |= ds.field("corpus_paper_id").isin(values)
    else:
        predicate |= pc.utf8_trim_whitespace(ds.field("corpus_paper_id")).isin(sorted(papers))
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in source.scanner(columns=columns, filter=predicate).to_batches():
        for row in batch.to_pylist():
            block = normalize_block_key(row["cluster_block_key"])
            paper = _normalize_paper_id(row["corpus_paper_id"])
            position = _strict_nonneg_int(row["position"])
            destinations = (
                positions.get((paper, position), set()) if paper is not None and position is not None else set()
            )
            if block in joins:
                destinations = destinations | {block}
            for destination in destinations:
                partitions[destination].append(row)
    return partitions


def _normalized_orcid(value: Any) -> str | None:
    text = _optional_str(value)
    return normalize_orcid(text) if text else None


def _signature_names(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        _optional_str(row.get("author_first")),
        _optional_str(row.get("author_middle")),
        _optional_str(row.get("author_last")),
    )


def _rotated_key(row: dict[str, Any]) -> str | None:
    name = canonicalize_name_parts(*_signature_names(row))
    if name.first and name.last:
        return f"{name.last[0]} {name.first.split()[0]}"
    return None


def _block_rows(
    join: SignatureJoin,
    block: str,
    gold: list[dict[str, Any]],
    reviewed_errors: Collection[str],
    allowed_papers: Collection[str],
) -> tuple[list[MentionEvidence], Counter[str]]:
    by_signature: dict[str, list[GoldEvidence]] = defaultdict(list)
    rejected: dict[str, list[RejectedGoldEvidence]] = defaultdict(list)
    by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in join.rows:
        by_paper[row["paper_id_norm"]].append(row)
    counts: Counter[str] = Counter()
    for item in gold:
        counts["gold_rows_scanned"] += 1
        gold_block = normalize_block_key(item["cluster_block_key"])
        counts["gold_rows_for_block"] += gold_block == block
        paper = _normalize_paper_id(item["corpus_paper_id"])
        position = _strict_nonneg_int(item["position"])
        orcid = _normalized_orcid(item["orcid"])
        if paper is None or position is None or orcid is None:
            counts["invalid_gold_key_or_orcid"] += 1
            continue
        if paper not in allowed_papers:
            counts["permission_filtered_gold_rows"] += gold_block == block
            continue
        names = (
            _optional_str(item.get("author_first", item.get("first"))),
            _optional_str(item.get("author_middle", item.get("middle"))),
            _optional_str(item.get("author_last", item.get("last"))),
        )
        entry: GoldEvidence = {"orcid": orcid, "scored": item["scored"] is True, "split": _optional_str(item["split"])}
        matched, reason = join.match(paper, position, gold_block, names)
        if matched is not None and gold_block == block:
            by_signature[matched["signature_id"]].append(entry)
            counts["matched_gold_rows"] += 1
            continue
        reason = reason or "dropped_block_key_mismatch"
        candidates = [row for row in by_paper.get(paper, ()) if row["pos_norm"] == position]
        if reason == "dropped_block_key_mismatch" and any(gold_block == _rotated_key(row) for row in candidates):
            reason = "dropped_name_order_rotation"
        elif gold_block == block and reason == "dropped_ambiguous_paper_block":
            # The gold row could refer to any of this paper's mentions in the block.
            candidates = by_paper[paper]
        elif gold_block == block and reason == "dropped_unmatched_paper_position" and paper in by_paper:
            # Gold v2 has no author names to resolve a shifted position.
            candidates = by_paper[paper]
            reason = "dropped_position_shift"
        counts[reason] += 1
        for candidate in candidates:
            rejected[candidate["signature_id"]].append({**entry, "reason": reason})

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
            "rejected_gold": rejected.get(sid, []),
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
    snapshot: Path,
    gold: Path,
    output: Path,
    reviewed_errors: dict[str, list[str]] | None = None,
    *,
    max_unmatched_gold: float = DEFAULT_MAX_UNMATCHED_GOLD,
    append: bool = False,
) -> dict[str, Any]:
    """Write verified evidence partitions for new blocks and preserve existing ones."""
    snapshot, output = snapshot.resolve(), output.resolve()
    with exclusive_file_lock(snapshot / INTAKE_LOCK):
        return _write_evidence(snapshot, gold, output, reviewed_errors, max_unmatched_gold, append=append)


def _write_evidence(
    snapshot: Path,
    gold: Path,
    output: Path,
    reviewed_errors: dict[str, list[str]] | None,
    max_unmatched_gold: float,
    *,
    append: bool,
) -> dict[str, Any]:
    if not 0 <= max_unmatched_gold <= 1:
        raise ValueError("Maximum unmatched gold share must be in [0, 1]")
    datasets = generic_dataset_manifests(snapshot)
    blocks = {normalize_block_key(name): (name, path) for name, path in datasets.items()}
    if len(blocks) != len(datasets):
        raise ValueError("Duplicate canonical block keys in snapshot")
    errors = {normalize_block_key(key): values for key, values in (reviewed_errors or {}).items()}
    if len(errors) != len(reviewed_errors or {}) or errors.keys() - blocks.keys():
        raise ValueError("Reviewed errors must name unique exported blocks")
    binding = read_json_object(snapshot / SNAPSHOT_BINDING)
    if not binding.get("permission_sha256") or not binding.get("snapshot_id"):
        raise ValueError("Identity evidence requires a permission-gated intake snapshot")
    allowed_papers = frozenset(binding["permission_papers"]["s2ag"])
    snapshot_digest = sha256_file(snapshot / "manifest.json")
    index_digest = binding["name_counts_index_sha256"]
    intake_digest = canonical_json_digest([binding["snapshot_id"], index_digest])
    gold_digest = sha256_file(gold)
    manifest_path = output / "manifest.json"
    metadata: dict[str, EvidenceBlock] = {}
    audits: dict[str, dict[str, int]] = {}
    if append:
        prior = load_identity_evidence(manifest_path)
        payload = read_json_object(manifest_path)
        if prior.intake_sha256 != intake_digest or payload["gold_sha256"] != gold_digest:
            raise ValueError("Evidence append requires the same intake and gold source")
        if prior.blocks.keys() - blocks.keys():
            raise ValueError("Prior evidence blocks are missing from the snapshot")
        for block, entry in prior.blocks.items():
            if block in errors and canonical_json_digest(sorted(errors[block])) != entry["reviewed_errors_sha256"]:
                raise ValueError(f"Reviewed errors changed for retained block {block!r}")
            prior.read_block(block, dataset_dir=blocks[block][1].parent)
        metadata.update(prior.blocks)
        audits.update(payload["audit"])
    elif output.exists():
        raise FileExistsError(f"Evidence already exists: {output}. Pass --append to add blocks.")
    new_blocks = {block: spec for block, spec in sorted(blocks.items()) if block not in metadata}
    joins = {}
    for block, (_, path) in new_blocks.items():
        manifest = read_json_object(path)
        if sha256_file(path.parent / "signatures.arrow") != manifest["files"]["signatures"]["sha256"]:
            raise ValueError(f"Snapshot signature digest mismatch: {block!r}")
        joins[block] = SignatureJoin(_read_arrow_signatures(path.parent))
    gold_partitions = _read_gold(gold, joins) if joins else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as tmp:
        staging = Path(tmp) / "evidence"
        staging.mkdir()
        for index, (block, (dataset, path)) in enumerate(new_blocks.items(), start=len(metadata)):
            rows, audit = _block_rows(
                joins[block], block, gold_partitions.pop(block, []), errors.get(block, ()), allowed_papers
            )
            if append and not audit["matched_gold_rows"]:
                raise ValueError(f"No gold rows matched the appended block {block!r}")
            unmatched = audit["dropped_unmatched_paper_position"] + audit["dropped_position_shift"]
            total = audit["gold_rows_for_block"] - audit["permission_filtered_gold_rows"]
            if total and unmatched / total > max_unmatched_gold:
                raise ValueError(
                    f"{block!r} has {unmatched}/{total} unmatched gold rows ({unmatched / total:.2%}); "
                    f"--max-unmatched-gold is {max_unmatched_gold:.2%}"
                )
            destination = staging / f"block-{index:06d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=EVIDENCE_SCHEMA), destination)
            source = path.parent / SOURCE_RECORD
            metadata[block] = {
                "path": destination.name,
                "sha256": sha256_file(destination),
                "signature_count": len(rows),
                "dataset_manifest_sha256": sha256_file(path),
                "dataset": dataset,
                "permission_sha256": read_json_object(source)["permission_sha256"],
                "request_source_sha256": sha256_file(source),
                "reviewed_errors_sha256": canonical_json_digest(sorted(errors.get(block, ()))),
            }
            audits[block] = dict(audit)
        if not sum(audit.get("matched_gold_rows", 0) for audit in audits.values()):
            raise ValueError("No gold rows matched the intake snapshot")
        result = {
            "format_version": 1,
            "excluded_orcid_groups_sha256": excluded_orcid_groups_digest(),
            "snapshot_sha256": snapshot_digest,
            "intake_sha256": intake_digest,
            "name_counts_index_sha256": index_digest,
            "snapshot_root": str(snapshot),
            "gold_sha256": gold_digest,
            "blocks": metadata,
            "audit": audits,
        }
        if append:
            for entry in new_blocks:
                filename = metadata[entry]["path"]
                (staging / filename).replace(output / filename)
            _replace_json(manifest_path, result)
        else:
            _replace_json(staging / "manifest.json", result)
            staging.rename(output)
    return result


def main(argv: list[str] | None = None) -> int:
    """Build evidence for all canonical blocks in a snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--gold-parquet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reviewed-errors-json", type=Path)
    parser.add_argument("--append", action="store_true", help="Verify retained blocks and add new evidence partitions.")
    parser.add_argument("--max-unmatched-gold", type=float, default=DEFAULT_MAX_UNMATCHED_GOLD)
    args = parser.parse_args(argv)
    reviewed = read_json_object(args.reviewed_errors_json) if args.reviewed_errors_json else None
    result = build_identity_evidence(
        args.snapshot,
        args.gold_parquet,
        args.output_dir,
        reviewed,
        max_unmatched_gold=args.max_unmatched_gold,
        append=args.append,
    )
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
