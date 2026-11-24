"""Mine 1.4 errors and random controls from frozen pairwise training blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import chain, groupby
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import load_block_roles, normalize_block_key
from scripts.common.identity_ledger import IdentityLedger, ledger_for_roles
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, read_external_reference_identity_evidence

Pair = tuple[str, str]


@dataclass(frozen=True)
class BlockInput:
    arrow_dir: Path
    labels: Path | None
    assignments: Path


def _mention_orcids(root: Path, block: str) -> dict[str, str | None]:
    signatures: dict[str, str | None] = {}
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
        for batch_index in range(reader.num_record_batches):
            for row in reader.get_batch(batch_index).select(columns).to_pylist():
                signature = row["signature_id"]
                if not signature or normalize_block_key(row["author_block"]) != block:
                    raise ValueError(f"Expected real signatures from block {block!r} in {root}")
                if signature.startswith("visibleq:"):
                    continue
                if signature in signatures:
                    raise ValueError(f"Duplicate signature {signature!r} in {root}")
                signatures[signature] = normalize_orcid(row.get("author_orcid") or "")
    return signatures


def _eligible_mentions(
    spec: BlockInput, block: str, frozen_orcids: frozenset[str], homes: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int]:
    deposited = _mention_orcids(spec.arrow_dir, block)
    if spec.labels is None:
        raise ValueError("Legacy pairwise inputs require labels")
    identities, scored = read_external_reference_identity_evidence(spec.labels, require_scored=True)
    if unknown := identities.keys() - deposited.keys():
        raise ValueError(f"Joined labels contain signatures outside {block!r}: {sorted(unknown)[:5]}")
    if frozen_orcids and frozen_orcids.isdisjoint(chain(identities.values(), deposited.values())):
        raise ValueError(f"Neither label nor Arrow input names a frozen ORCID in {block!r}")
    eligible = {}
    for sid, orcid in deposited.items():
        gold = identities.get(sid)
        if any(homes.get(oid, "pairwise_train") != "pairwise_train" for oid in (gold, orcid) if oid):
            continue
        eligible[sid] = gold or orcid or f"signature:{sid}"
    labels = {sid: oid for sid, oid in scored.items() if sid in eligible and oid not in EXCLUDED_ORCID_GROUPS}
    table = pq.read_table(spec.assignments, columns=["signature_id", "cluster_id"])
    components: dict[str, str] = {}
    for sid, cid in zip(table["signature_id"].to_pylist(), table["cluster_id"].to_pylist(), strict=True):
        if sid not in deposited or sid in components or type(cid) not in {str, int} or cid == "":
            raise ValueError(f"Invalid or duplicate assignment for {sid!r} in {spec.assignments}")
        components[sid] = str(cid)
    if components.keys() != deposited.keys():
        raise ValueError(f"Expected 1.4 assignments for every real signature in {block!r}")
    return eligible, labels, components, len(deposited) - len(eligible)


def _ledger_mentions(
    ledger: IdentityLedger, spec: BlockInput, block: str
) -> tuple[dict[str, str], dict[str, str], dict[str, str], int]:
    view = ledger.for_block(block, dataset_dir=spec.arrow_dir)
    manifest = read_json_object(spec.assignments.parent / "manifest.json")
    ledger.require_bindings(manifest["binding"], context=f"{block} pairwise assignments")
    if manifest["files"].get(spec.assignments.name) != sha256_file(spec.assignments):
        raise ValueError(f"Assignment digest mismatch: {spec.assignments}")
    decisions = {sid: view.eligibility(sid) for sid in view.rows}
    eligible = {
        sid: next(iter(decision.identities), f"signature:{sid}")
        for sid, decision in decisions.items()
        if decision.fit_ok
    }
    gold = {sid: decision.score_orcid for sid, decision in decisions.items() if decision.score_orcid is not None}
    rows = pq.read_table(spec.assignments, columns=["signature_id", "cluster_id"]).to_pylist()
    components = {row["signature_id"]: row["cluster_id"] for row in rows}
    if len(components) != len(rows) or components.keys() != eligible.keys() or not all(components.values()):
        raise ValueError(f"Expected assignments for every eligible pairwise signature in {block!r}")
    return eligible, gold, components, len(view.rows) - len(eligible)


@dataclass
class PairFrame:
    """Index a grouped pair population with one prefix count per mention."""

    signatures: list[str]
    spans: list[tuple[int, int]]
    ends: list[int]

    @property
    def size(self) -> int:
        return self.ends[-1] if self.ends else 0

    def pair(self, index: int) -> Pair:
        span = bisect_right(self.ends, index)
        right, start = self.spans[span]
        base = self.ends[span - 1] if span else 0
        return self.signatures[start + index - base], self.signatures[right]


def _grouped_frame(
    signatures: list[str], major: Mapping[str, str], minor: Mapping[str, str], *, include_unlabeled: bool = False
) -> PairFrame:
    ordered = sorted(signatures, key=lambda sid: (major[sid], sid not in minor, minor.get(sid, ""), sid))
    frame = PairFrame(ordered, [], [])
    for _, group in groupby(enumerate(ordered), key=lambda item: major[item[1]]):
        start = 0
        previous_minor = None
        minor_start = 0
        for right, sid in group:
            key = minor.get(sid)
            if key != previous_minor or include_unlabeled:
                minor_start = right
                previous_minor = key
            if right == 0 or major[ordered[right - 1]] != major[sid]:
                start = right
                minor_start = right
            count = (
                right - start
                if include_unlabeled and key is None
                else (0 if include_unlabeled else minor_start - start)
            )
            if count:
                frame.spans.append((right, start))
                frame.ends.append(frame.size + count)
    return frame


def _random_pair(signatures: list[str], index: int) -> Pair:
    right = (1 + math.isqrt(1 + 8 * index)) // 2
    return signatures[index - math.comb(right, 2)], signatures[right]


def _sample_frame(
    pair_at: Callable[[int], Pair],
    population: int,
    identities: Mapping[str, str],
    excluded_pairs: set[Pair],
    *,
    per_block: int,
    per_identity: int,
    frame_seed: str,
    rng: random.Random,
) -> tuple[list[Pair], list[Pair]]:
    # Freeze a bounded candidate frame before sampling, so probabilities account for identity caps.
    frame: list[Pair] = []
    counts: Counter[str] = Counter()
    frame_size = per_block * 4
    proposals = random.Random(frame_seed).sample(range(population), min(population, max(1024, frame_size * 16)))
    for index in proposals:
        a, b = pair_at(index)
        pair = (a, b) if a < b else (b, a)
        keys = {identities[a], identities[b]}
        if pair in excluded_pairs or any(counts[key] >= per_identity for key in keys):
            continue
        counts.update(keys)
        frame.append(pair)
        if len(frame) >= frame_size:
            break
    return rng.sample(frame, min(per_block, len(frame))), frame


def mine_pairs(
    roles_path: Path,
    blocks: Mapping[str, BlockInput],
    output_path: Path,
    *,
    per_block: int,
    identity_evidence: Path | None = None,
    per_identity: int = 25,
    seed: int = 20261123,
) -> int:
    """Write deduplicated pairs sampled within each capped stratum frame."""
    if min(per_block, per_identity) <= 0:
        raise ValueError("Pair caps must be positive")
    roles = load_block_roles(roles_path)
    ledger = ledger_for_roles(identity_evidence, roles)
    normalized = {normalize_block_key(block): spec for block, spec in blocks.items()}
    if not normalized or len(normalized) != len(blocks):
        raise ValueError("Expected at least one block and no duplicate normalized keys")
    for block in normalized:
        if roles.role_for(block) != "pairwise_train":
            raise ValueError(f"Block {block!r} is outside pairwise_train")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists() or output_path.exists():
        raise FileExistsError(output_path)

    homes = {
        orcid: roles.orcid_home_role.get(orcid, role)
        for key, role in roles.block_to_role.items()
        for orcid in roles.block_orcids[key]
    }
    rng = random.Random(seed)
    counts: dict[str, Any] = {}
    rows: list[tuple[str, str, str, str, float]] = []
    for block, spec in sorted(normalized.items()):
        identities, gold, clusters, excluded = (
            _ledger_mentions(ledger, spec, block)
            if ledger is not None
            else _eligible_mentions(spec, block, roles.block_orcids[block], homes)
        )
        signatures = sorted(identities)
        labeled = sorted(gold)
        false_merges = _grouped_frame(labeled, clusters, gold)
        missed_links = _grouped_frame(labeled, gold, clusters)
        merged_unlabeled = _grouped_frame(signatures, clusters, gold, include_unlabeled=True)
        stratum_counts = {}
        seen_pairs: set[Pair] = set()
        for stratum, population, pair_at in (
            ("false_merge", false_merges.size, false_merges.pair),
            ("missed_link", missed_links.size, missed_links.pair),
            ("merged_unlabeled", merged_unlabeled.size, merged_unlabeled.pair),
            ("random", math.comb(len(signatures), 2), partial(_random_pair, signatures)),
        ):
            sample, frame = _sample_frame(
                pair_at,
                population,
                identities,
                seen_pairs,
                per_block=per_block,
                per_identity=per_identity,
                frame_seed=f"{block}:{stratum}",
                rng=rng,
            )
            frame_size = len(frame)
            inclusion = len(sample) / frame_size if frame_size else 0.0
            stratum_counts[stratum] = {
                "eligible_pairs": population,
                "frame_pairs": frame_size,
                "reporting_population_pairs": frame_size,
                "sampled_pairs": len(sample),
                "inclusion_probability": inclusion,
            }
            seen_pairs.update(frame)
            rows.extend((block, left, right, stratum, inclusion) for left, right in sorted(sample))
        counts[block] = {
            "eligible_signatures": len(signatures),
            "excluded_non_home_signatures": excluded,
            "ineligible_signatures": excluded,
            "labels_sha256": sha256_file(spec.labels) if spec.labels is not None else None,
            "assignments_sha256": sha256_file(spec.assignments),
            "strata": stratum_counts,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_key", "signature_id_1", "signature_id_2", "stratum", "inclusion_probability"])
        writer.writerows(rows)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "block_role_assignments_sha256": roles.sha256,
                **(ledger.bindings if ledger is not None else {}),
                "seed": seed,
                "role": "pairwise_train",
                "pairs_per_block_per_stratum": per_block,
                "pairs_per_identity_per_stratum": per_identity,
                "probability_scope": "within_capped_stratum_frame",
                "reporting_population": "disjoint capped stratum frames fixed before sampling",
                "output_sha256": sha256_file(output_path),
                "blocks": counts,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Sample pairs from frozen training blocks and their stored 1.4 assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument(
        "--blocks-json", type=Path, required=True, help="Map block keys to arrow_dir, labels and assignments."
    )
    parser.add_argument("--pairs-per-block", type=int, required=True, help="Maximum per stratum and block.")
    parser.add_argument("--pairs-per-identity", type=int, default=25, help="Maximum per stratum and identity.")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = read_json_object(args.blocks_json)
    blocks = {
        block: BlockInput(
            arrow_dir=args.blocks_json.parent / spec["arrow_dir"],
            labels=args.blocks_json.parent / spec["labels"] if "labels" in spec else None,
            assignments=args.blocks_json.parent / spec["assignments"],
        )
        for block, spec in raw.items()
    }
    count = mine_pairs(
        args.block_role_assignments,
        blocks,
        args.output,
        per_block=args.pairs_per_block,
        identity_evidence=args.identity_evidence,
        per_identity=args.pairs_per_identity,
        seed=args.seed,
    )
    print(json.dumps({"blocks": len(blocks), "pairs": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
