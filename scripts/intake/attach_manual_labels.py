"""Re-attach corpus-keyed manual labels to a verified stage-zero snapshot."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.text import canonicalize_name_parts, canonicalize_name_text  # noqa: E402
from scripts.common.arrow_roots import (  # noqa: E402
    generic_dataset_manifests,
    read_json_object,
    resolved_manifest_artifact_path,
)
from scripts.common.identity_ledger import EVIDENCE_SHA256_FIELD, IdentityEvidence, load_identity_evidence  # noqa: E402

Status = Literal["exact", "shifted", "ambiguous", "orphaned"]
Resolution = tuple[Status, str | None]
MentionKey = tuple[str, int]
NameKey = tuple[str, str]
_STATUSES: tuple[Status, ...] = ("exact", "shifted", "ambiguous", "orphaned")
_CORPUS_MENTION = re.compile(r"([0-9]+)-([0-9]+)")


class TableReport(TypedDict):
    counts: dict[Status, int]
    by_source: dict[str, dict[Status, int]]
    attached: int
    attached_by_source: dict[str, dict[Status, int]]


def _name_key(first: str | None, last: str | None) -> NameKey | None:
    name = canonicalize_name_parts(first, None, last)
    return (name.last, name.first[0]) if name.first and name.last else None


def _author_name_keys(first: str | None, last: str | None) -> set[NameKey]:
    return {key for key in (_name_key(first, last), _name_key(last, first)) if key is not None}


def _resolve_anchor(
    position: int,
    first: str | None,
    last: str | None,
    signatures: Mapping[tuple[int, str], set[NameKey]],
    other_authors: Mapping[int, set[str]],
) -> Resolution:
    key = _name_key(first, last)
    if key is None:
        return "orphaned", None
    surname, initial = key
    compatible: set[tuple[int, str | None]] = {slot for slot, keys in signatures.items() if key in keys}
    # An author-list-only name can rule out a shift but cannot supply a signature ID.
    for pos, names in other_authors.items():
        if any(
            (name.endswith(f" {surname}") and name.startswith(initial))
            or (name.startswith(f"{surname} ") and name[len(surname) + 1 :].startswith(initial))
            for name in names
        ):
            compatible.add((pos, None))
    exact = {slot for slot in compatible if slot[0] == position}
    matches = exact or compatible
    if len(matches) > 1:
        return "ambiguous", None
    if matches:
        ((_, signature_id),) = matches
        if signature_id is not None:
            return ("exact" if exact else "shifted"), signature_id
    return "orphaned", None


def _arrow_rows(path: Path, columns: list[str], selector: str, values: set[str]) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all().select(columns)
        return table.filter(pc.is_in(table[selector], value_set=pa.array(sorted(values), type=pa.string()))).to_pylist()


def _resolve_mentions(anchors_path: Path, snapshot: Path, evidence: IdentityEvidence) -> dict[MentionKey, Resolution]:
    columns = ["corpus_paper_id", "position", "first", "last", "anchor_status"]
    anchors = pq.read_table(anchors_path, columns=columns).to_pylist()
    anchor_papers = {str(row["corpus_paper_id"]) for row in anchors if row["anchor_status"] == "anchored"}
    manifests = generic_dataset_manifests(snapshot)
    if {entry["dataset"] for entry in evidence.blocks.values()} != manifests.keys():
        raise ValueError("Identity evidence does not cover the snapshot datasets")

    signatures: dict[str, dict[tuple[int, str], set[NameKey]]] = defaultdict(dict)
    author_lists: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for block, entry in evidence.blocks.items():
        manifest_path = manifests[entry["dataset"]]
        rows = evidence.read_block(block, dataset_dir=manifest_path.parent)
        mentions = {sid: row for sid, row in rows.items() if row["corpus_paper_id"] in anchor_papers}
        if not mentions:
            continue
        manifest = read_json_object(manifest_path)
        paths = manifest["paths"]
        signature_path = resolved_manifest_artifact_path(manifest_path, paths["signatures"], key="signatures")
        for row in _arrow_rows(
            signature_path, ["signature_id", "author_first", "author_last"], "signature_id", set(mentions)
        ):
            sid = row["signature_id"]
            mention = mentions[sid]
            keys = _author_name_keys(row["author_first"], row["author_last"])
            if keys:
                signatures[mention["corpus_paper_id"]].setdefault((mention["position"], sid), set()).update(keys)
        authors_path = resolved_manifest_artifact_path(manifest_path, paths["paper_authors"], key="paper_authors")
        if sha256_file(authors_path) != manifest["files"]["paper_authors"]["sha256"]:
            raise ValueError(f"Snapshot author list digest mismatch for {block!r}")
        for row in _arrow_rows(authors_path, ["paper_id", "position", "author_name"], "paper_id", anchor_papers):
            author_lists[row["paper_id"]][row["position"]].add(canonicalize_name_text(row["author_name"]))

    for paper, names in signatures.items():
        if paper in author_lists:
            for position, _ in names:
                author_lists[paper].pop(position, None)

    resolved: dict[MentionKey, Resolution] = {}
    for row in anchors:
        paper, position = str(row["corpus_paper_id"]), row["position"]
        if (paper, position) in resolved:
            raise ValueError(f"Duplicate manual label anchor for paper {paper!r} at position {position}")
        resolved[paper, position] = (
            _resolve_anchor(position, row["first"], row["last"], signatures.get(paper, {}), author_lists.get(paper, {}))
            if row["anchor_status"] == "anchored"
            else ("orphaned", None)
        )
    return resolved


def _label_members(row: Mapping[str, Any], *, pair: bool) -> list[MentionKey]:
    if pair:
        return [(str(row[f"corpus_paper_id_{side}"]), row[f"position_{side}"]) for side in ("a", "b")]
    members = []
    for raw_id in row["member_signature_ids"] or ():
        match = _CORPUS_MENTION.fullmatch(raw_id)
        if match is None:
            raise ValueError(f"Invalid corpus member {raw_id!r} in source {row['source_id']!r}")
        members.append((match[1], int(match[2])))
    return members


def _resolve_label(
    members: Sequence[MentionKey], mentions: Mapping[MentionKey, Resolution]
) -> tuple[Status, list[str]]:
    resolutions = [mentions.get(member, ("orphaned", None)) for member in members]
    ids = [sid for _, sid in resolutions if sid is not None]
    statuses = {status for status, _ in resolutions}
    if "ambiguous" in statuses or len(set(ids)) != len(ids):
        return "ambiguous", []
    if not ids or len(ids) != len(members):
        return "orphaned", []
    return ("shifted" if "shifted" in statuses else "exact"), ids


def _write_labels(source: Path, output: Path, mentions: Mapping[MentionKey, Resolution], *, pair: bool) -> TableReport:
    parquet = pq.ParquetFile(source)
    id_fields = (
        [pa.field(f"snapshot_signature_id_{side}", pa.string()) for side in ("a", "b")]
        if pair
        else [pa.field("snapshot_member_signature_ids", pa.list_(pa.string()))]
    )
    schema = parquet.schema_arrow
    for field in [pa.field("status", pa.string()), pa.field("attached", pa.bool_()), *id_fields]:
        schema = schema.append(field)
    by_source: dict[str, Counter[Status]] = defaultdict(Counter)
    attached: dict[str, Counter[Status]] = defaultdict(Counter)
    with pq.ParquetWriter(output, schema) as writer:
        for batch in parquet.iter_batches(batch_size=8192):
            rows = batch.to_pylist()
            for row in rows:
                source_id = row["source_id"]
                if not source_id or row["id_space"] != "corpus" or not isinstance(row["usable"], bool):
                    raise ValueError(f"Invalid corpus label provenance in {source}: {source_id!r}")
                status, ids = _resolve_label(_label_members(row, pair=pair), mentions)
                keep = row["usable"] and bool(ids)
                row.update(status=status, attached=keep)
                by_source[source_id][status] += 1
                if keep:
                    attached[source_id][status] += 1
                if pair:
                    row["snapshot_signature_id_a"], row["snapshot_signature_id_b"] = ids if keep else (None, None)
                else:
                    row["snapshot_member_signature_ids"] = ids if keep else None
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return {
        "counts": {status: sum(counts[status] for counts in by_source.values()) for status in _STATUSES},
        "by_source": {name: {status: counts[status] for status in _STATUSES} for name, counts in by_source.items()},
        "attached": sum(counts.total() for counts in attached.values()),
        "attached_by_source": {name: {status: attached[name][status] for status in _STATUSES} for name in by_source},
    }


def attach_manual_labels(
    anchors: Path, pairs: Path, changes: Path, snapshot: Path, evidence_manifest: Path, output: Path
) -> dict[str, TableReport]:
    """Attach usable manual labels and audit the match status of each source row.

    Args:
        anchors: Corpus mention anchor Parquet file.
        pairs: Corpus pair label Parquet file.
        changes: Corpus cluster change Parquet file.
        snapshot: Root of the stage-zero Arrow collection.
        evidence_manifest: Identity evidence manifest for that snapshot.
        output: Fresh directory for remapped Parquet files, report and manifest.

    Returns:
        Match and attachment counts grouped by label table and source.
    """
    if output.exists():
        raise FileExistsError(f"Manual label attachment output already exists: {output}")
    evidence = load_identity_evidence(evidence_manifest)
    snapshot_digest = sha256_file(snapshot / "manifest.json")
    if snapshot_digest != evidence.snapshot_sha256:
        raise ValueError("Identity evidence was built for a different snapshot")
    inputs = {"mention_anchors": anchors, "pair_labels_corpus": pairs, "cluster_changes_corpus": changes}
    input_digests = {name: sha256_file(path) for name, path in inputs.items()}
    mentions = _resolve_mentions(anchors, snapshot, evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as temporary:
        staging = Path(temporary) / "attached"
        staging.mkdir()
        report = {
            "pair_labels": _write_labels(pairs, staging / "pair_labels.parquet", mentions, pair=True),
            "cluster_changes": _write_labels(changes, staging / "cluster_changes.parquet", mentions, pair=False),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "format_version": 1,
            "inputs_sha256": input_digests,
            "snapshot_sha256": snapshot_digest,
            EVIDENCE_SHA256_FIELD: evidence.sha256,
            "outputs_sha256": {path.name: sha256_file(path) for path in staging.iterdir()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run one manual label attachment from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mention-anchors", "pair-labels", "cluster-changes", "snapshot", "identity-evidence", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    report = attach_manual_labels(
        args.mention_anchors,
        args.pair_labels,
        args.cluster_changes,
        args.snapshot,
        args.identity_evidence,
        args.output_dir,
    )
    print(json.dumps({name: {**table["counts"], "attached": table["attached"]} for name, table in report.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
