"""Mine 1.4 errors and random controls from frozen fit_pool blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from itertools import chain, groupby
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._sha256 import is_lowercase_sha256, sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, load_block_roles, normalize_block_key
from scripts.common.identity_ledger import BlockEligibility, ledger_for_roles
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, read_external_reference_identity_evidence
from scripts.common.stage0_invariants import (
    JUDGE_ONLY_STRATUM,
    PAIR_STRATA,
    PairId,
    PairStratum,
    pairwise_fold,
    require_assignment_receipt,
    require_fit_pool_coverage,
)

DEFAULT_PAIRS_PER_BLOCK = 2_000
DEFAULT_VERIFIED_PAIR_FRACTION = 0.05
DEFAULT_PAIRS_PER_BLOCK_FLOOR = 100
PAIR_COLUMNS = ("block_key", "signature_id_1", "signature_id_2", "stratum", "inclusion_probability")


@dataclass(frozen=True)
class SamplingPolicy:
    """Keep the sampling inputs needed to replay each block."""

    seed: int
    pairs_per_block: int
    verified_pair_fraction: float
    pairs_per_block_floor: int
    pairs_per_identity_per_stratum: int

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or type(self.pairs_per_block) is not int
            or type(self.pairs_per_block_floor) is not int
            or type(self.pairs_per_identity_per_stratum) is not int
            or type(self.verified_pair_fraction) not in {int, float}
            or min(self.pairs_per_block, self.pairs_per_identity_per_stratum) <= 0
            or not 0 <= self.verified_pair_fraction <= 1
            or self.pairs_per_block_floor < 0
        ):
            raise ValueError("Invalid pair sampling policy")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> SamplingPolicy:
        """Require the mining policy fields before replaying a sample."""
        if cls.__dataclass_fields__.keys() - manifest.keys():
            raise ValueError("Missing pair sampling policy. Re-mine pairs.")
        return cls(
            manifest["seed"],
            manifest["pairs_per_block"],
            manifest["verified_pair_fraction"],
            manifest["pairs_per_block_floor"],
            manifest["pairs_per_identity_per_stratum"],
        )


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
    spec: BlockInput, block: str, frozen_orcids: frozenset[str], identity_roles: Mapping[str, frozenset[str]]
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
        if any(identity_roles.get(oid, frozenset({"fit_pool"})) != {"fit_pool"} for oid in (gold, orcid) if oid):
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


def ledger_population(view: BlockEligibility) -> tuple[dict[str, str], dict[str, str], int]:
    """Return eligible identities, scoring labels and excluded mentions."""
    decisions = {sid: view.eligibility(sid) for sid in view.rows}
    eligible = {
        sid: next(iter(decision.identities), f"signature:{sid}")
        for sid, decision in decisions.items()
        if decision.fit_ok
    }
    gold = {sid: decision.score_orcid for sid, decision in decisions.items() if decision.score_orcid is not None}
    return eligible, gold, len(view.rows) - len(eligible)


def read_assignment_partition(
    path: Path, block: str, eligible: Collection[str], manifest: Mapping[str, Any]
) -> tuple[dict[str, str], str]:
    """Verify the frozen assignment file and read its eligible partition."""
    digest = sha256_file(path)
    files = manifest.get("files")
    if not isinstance(files, dict) or files.get(path.name) != digest:
        raise ValueError(f"Assignment digest mismatch: {path}")
    rows = pq.read_table(path, columns=["signature_id", "cluster_id"]).to_pylist()
    components = {}
    for row in rows:
        sid, cid = row["signature_id"], row["cluster_id"]
        if sid in components or type(cid) not in {str, int} or cid == "":
            raise ValueError(f"Invalid or duplicate assignment for {sid!r} in {path}")
        components[sid] = str(cid)
    if components.keys() != set(eligible):
        raise ValueError(f"Expected assignments for every eligible pairwise signature in {block!r}")
    return components, digest


@dataclass
class PairFrame:
    """Index a grouped pair population with one prefix count per mention."""

    signatures: list[str]
    spans: list[tuple[int, int]]
    ends: list[int]

    @property
    def size(self) -> int:
        return self.ends[-1] if self.ends else 0

    def pair(self, index: int) -> PairId:
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


def _random_pair(signatures: list[str], index: int) -> PairId:
    right = (1 + math.isqrt(1 + 8 * index)) // 2
    return signatures[index - math.comb(right, 2)], signatures[right]


def _capped_frame(
    pair_at: Callable[[int], PairId],
    population: int,
    identities: Mapping[str, str],
    excluded_pairs: set[PairId],
    *,
    per_block: int,
    per_identity: int,
    rng: random.Random,
) -> tuple[list[PairId], Counter[str]]:
    # Freeze a bounded candidate frame before sampling, so probabilities account for identity caps.
    frame: list[PairId] = []
    drops: Counter[str] = Counter()
    if not per_block:
        if population:
            drops["zero_cap"] = population
        return frame, drops
    counts: Counter[str] = Counter()
    frame_size = per_block * 4
    proposals = rng.sample(range(population), min(population, max(1024, frame_size * 16)))
    for examined, index in enumerate(proposals, 1):
        a, b = pair_at(index)
        pair = (a, b) if a < b else (b, a)
        keys = {identities[a], identities[b]}
        if pair in excluded_pairs:
            drops["cross_stratum_duplicate"] += 1
            continue
        if any(counts[key] >= per_identity for key in keys):
            drops["identity_cap"] += 1
            continue
        counts.update(keys)
        frame.append(pair)
        if len(frame) >= frame_size:
            if examined < population:
                drops["frame_budget"] = population - examined
            return frame, drops
    if len(proposals) < population:
        drops["proposal_budget"] = population - len(proposals)
    return frame, drops


def _stratum_quotas(frame_sizes: list[int], budget: int) -> list[int]:
    """Split the block budget evenly, redistributing slots from exhausted frames."""
    quotas = [0] * len(frame_sizes)
    active = [i for i, size in enumerate(frame_sizes) if size]
    while active and budget:
        share, remainder = divmod(budget, len(active))
        for offset, i in enumerate(active):
            count = min(frame_sizes[i] - quotas[i], share + (offset < remainder))
            quotas[i] += count
            budget -= count
        active = [i for i in active if quotas[i] < frame_sizes[i]]
    return quotas


@dataclass(frozen=True)
class BlockSample:
    """Hold one block's selected pairs and sampling report."""

    pairs: dict[PairId, tuple[PairStratum, float]]
    counts: dict[str, Any]


def select_block_pairs(
    block: str,
    identities: Mapping[str, str],
    gold: Mapping[str, str],
    clusters: Mapping[str, str],
    excluded: int,
    policy: SamplingPolicy,
) -> BlockSample:
    """Derive one block's fitted and judge samples from frozen inputs."""
    rng = random.Random(f"{policy.seed}:{block}")
    signatures, labeled = sorted(identities), sorted(gold)
    verified_pairs = math.comb(len(labeled), 2)
    cap = max(
        min(policy.pairs_per_block, int(policy.verified_pair_fraction * verified_pairs)),
        min(policy.pairs_per_block_floor, verified_pairs),
    )
    judge_cap = policy.pairs_per_block
    false_merges = _grouped_frame(labeled, clusters, gold)
    missed_links = _grouped_frame(labeled, gold, clusters)
    merged_unlabeled = _grouped_frame(signatures, clusters, gold, include_unlabeled=True)
    populations = (
        (false_merges.size, false_merges.pair),
        (missed_links.size, missed_links.pair),
        (verified_pairs, partial(_random_pair, labeled)),
        (merged_unlabeled.size, merged_unlabeled.pair),
    )
    seen_pairs: set[PairId] = set()
    frames = []
    for stratum, (population, pair_at) in zip(PAIR_STRATA, populations, strict=True):
        frame, drops = _capped_frame(
            pair_at,
            population,
            identities,
            seen_pairs,
            per_block=judge_cap if stratum == JUDGE_ONLY_STRATUM else cap,
            per_identity=policy.pairs_per_identity_per_stratum,
            rng=rng,
        )
        frames.append((frame, drops))
        seen_pairs.update(frame)
    quotas = [*_stratum_quotas([len(frame) for frame, _ in frames[:-1]], cap), min(judge_cap, len(frames[-1][0]))]
    strata = {}
    pairs = {}
    for stratum, (population, _), (frame, drops), quota in zip(PAIR_STRATA, populations, frames, quotas, strict=True):
        sampled = rng.sample(frame, quota)
        inclusion = quota / len(frame) if frame else 0.0
        if quota < len(frame):
            drops["sampling_budget"] = len(frame) - quota
        strata[stratum] = {
            "eligible_pairs": population,
            "frame_pairs": len(frame),
            "reporting_population_pairs": len(frame),
            "sampled_pairs": quota,
            "dropped_pairs": sum(drops.values()),
            "drops": dict(drops),
            "inclusion_probability": inclusion,
        }
        pairs.update((pair, (stratum, inclusion)) for pair in sorted(sampled))
    return BlockSample(
        pairs,
        {
            "eligible_signatures": len(signatures),
            "excluded_non_home_signatures": excluded,
            "ineligible_signatures": excluded,
            "verified_signatures": len(labeled),
            "verified_within_block_pairs": verified_pairs,
            "pair_cap": cap,
            "judge_pair_cap": judge_cap,
            "strata": strata,
        },
    )


def mine_pairs(
    roles_path: Path,
    blocks: Mapping[str, BlockInput],
    output_path: Path,
    *,
    per_block: int = DEFAULT_PAIRS_PER_BLOCK,
    verified_pair_fraction: float = DEFAULT_VERIFIED_PAIR_FRACTION,
    per_block_floor: int = DEFAULT_PAIRS_PER_BLOCK_FLOOR,
    identity_evidence: Path | None = None,
    per_identity: int = 25,
    seed: int = 20261123,
) -> int:
    """Write capped stratum samples and counts for every frozen fit_pool block."""
    policy = SamplingPolicy(seed, per_block, verified_pair_fraction, per_block_floor, per_identity)
    roles = load_block_roles(roles_path)
    ledger = ledger_for_roles(identity_evidence, roles)
    normalized = {normalize_block_key(block): spec for block, spec in blocks.items()}
    if len(normalized) != len(blocks):
        raise ValueError("Duplicate normalized block keys in pairwise inputs")
    require_fit_pool_coverage(roles, normalized.keys())
    if not normalized:
        raise ValueError("Expected at least one frozen fit_pool block")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists() or output_path.exists():
        raise FileExistsError(output_path)

    assignment_manifests = {}
    shared_binding: dict[str, str] = {}
    if ledger is not None:
        intake_index = ledger.evidence.name_counts_index_sha256
        shared_binding["index_sha256"] = intake_index
        for block, spec in sorted(normalized.items()):
            manifest = read_json_object(spec.assignments.parent / "manifest.json")
            binding = manifest.get("binding")
            require_assignment_receipt(roles, block, binding, intake_index_sha256=intake_index)
            model_digest = manifest["binding"].get("model_sha256")
            if (
                not is_lowercase_sha256(model_digest)
                or shared_binding.setdefault("model_sha256", model_digest) != model_digest
            ):
                raise ValueError(f"{block!r} pairwise assignments lack a shared model_sha256")
            assignment_manifests[block] = manifest

    counts: dict[str, Any] = {}
    rows: list[tuple[str, str, str, str, float]] = []
    for block, spec in sorted(normalized.items()):
        if ledger is None:
            identities, gold, clusters, excluded = _eligible_mentions(
                spec, block, roles.block_orcids[block], roles.identity_roles
            )
            assignments_digest = sha256_file(spec.assignments)
            assignment_input = {}
        else:
            view = ledger.for_block(block, dataset_dir=spec.arrow_dir)
            identities, gold, excluded = ledger_population(view)
            receipt = assignment_manifests[block]
            clusters, assignments_digest = read_assignment_partition(spec.assignments, block, identities, receipt)
            assignment_input = {
                "assignments_path": str(spec.assignments.resolve()),
                "assignment_binding": receipt["binding"],
            }
        sample = select_block_pairs(block, identities, gold, clusters, excluded, policy)
        rows.extend(
            (block, left, right, stratum, inclusion) for (left, right), (stratum, inclusion) in sample.pairs.items()
        )
        counts[block] = {
            "fold": pairwise_fold(roles, block),
            **assignment_input,
            **sample.counts,
            "labels_sha256": sha256_file(spec.labels) if spec.labels is not None else None,
            "assignments_sha256": assignments_digest,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAIR_COLUMNS)
        writer.writerows(rows)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                ROLE_SHA256_FIELD: roles.sha256,
                **({BLOCK_BINDINGS_FIELD: roles.bindings_for(normalized.keys())} if ledger is not None else {}),
                **shared_binding,
                "role": "fit_pool",
                **asdict(policy),
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
    parser.add_argument(
        "--pairs-per-block", type=int, default=DEFAULT_PAIRS_PER_BLOCK, help="Cap per fitted and judge sample."
    )
    parser.add_argument("--verified-pair-fraction", type=float, default=DEFAULT_VERIFIED_PAIR_FRACTION)
    parser.add_argument("--pairs-per-block-floor", type=int, default=DEFAULT_PAIRS_PER_BLOCK_FLOOR)
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
        verified_pair_fraction=args.verified_pair_fraction,
        per_block_floor=args.pairs_per_block_floor,
        identity_evidence=args.identity_evidence,
        per_identity=args.pairs_per_identity,
        seed=args.seed,
    )
    print(json.dumps({"blocks": len(blocks), "pairs": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
