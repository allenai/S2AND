"""Mine 1.4 errors and random controls from frozen fit_pool blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from functools import partial
from itertools import groupby
from pathlib import Path
from typing import Any

from s2and._sha256 import sha256_file
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, load_block_roles, normalize_block_key
from scripts.common.identity_ledger import BlockEligibility, IdentityLedger, load_identity_ledger
from scripts.common.linker_seeds import read_seed_components
from scripts.common.stage0_invariants import (
    JUDGE_ONLY_STRATUM,
    PAIR_STRATA,
    PairId,
    PairStratum,
    pairwise_fold,
    read_fit_pool_queries,
    require_assignment_coverage,
    require_assignment_receipts,
    require_fit_pool_coverage,
)

DEFAULT_PAIRS_PER_BLOCK = 2_000
DEFAULT_VERIFIED_PAIR_FRACTION = 0.05
DEFAULT_PAIRS_PER_BLOCK_FLOOR = 100
PAIR_COLUMNS = ("block_key", "signature_id_1", "signature_id_2", "stratum", "inclusion_probability")


@dataclass(frozen=True)
class SamplingPolicy:
    """Keep the sampling inputs needed to replay each block."""

    seed: int = 20261123
    pairs_per_block: int = DEFAULT_PAIRS_PER_BLOCK
    verified_pair_fraction: float = DEFAULT_VERIFIED_PAIR_FRACTION
    pairs_per_block_floor: int = DEFAULT_PAIRS_PER_BLOCK_FLOOR
    pairs_per_identity_per_stratum: int = 25

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
    def from_args(cls, args: argparse.Namespace) -> SamplingPolicy:
        """Read the common pair sampling options."""
        return cls(
            seed=getattr(args, "seed", cls.seed),
            pairs_per_block=getattr(args, "pairs_per_block", cls.pairs_per_block),
            verified_pair_fraction=getattr(args, "verified_pair_fraction", cls.verified_pair_fraction),
            pairs_per_block_floor=getattr(args, "pairs_per_block_floor", cls.pairs_per_block_floor),
            pairs_per_identity_per_stratum=getattr(
                args, "pairs_per_identity_per_stratum", cls.pairs_per_identity_per_stratum
            ),
        )


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    """Configure the same sampling policy for reports and training."""
    parser.add_argument("--pairs-per-block", type=int, default=DEFAULT_PAIRS_PER_BLOCK)
    parser.add_argument("--verified-pair-fraction", type=float, default=DEFAULT_VERIFIED_PAIR_FRACTION)
    parser.add_argument("--pairs-per-block-floor", type=int, default=DEFAULT_PAIRS_PER_BLOCK_FLOOR)
    parser.add_argument("--pairs-per-identity", type=int, dest="pairs_per_identity_per_stratum", default=25)
    parser.add_argument("--seed", type=int, default=20261123, help="Seed for choosing mined pairs.")


@dataclass(frozen=True)
class BlockInput:
    arrow_dir: Path
    assignments: Path


def _read_assignment_partition(path: Path, receipt: Mapping[str, object]) -> tuple[dict[str, str], str]:
    """Verify the frozen assignment file and read its partition."""
    digest = sha256_file(path)
    files = receipt.get("files")
    if not isinstance(files, dict) or files.get(path.name) != digest:
        raise ValueError(f"Assignment digest mismatch: {path}")
    return read_seed_components(path), digest


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
    *,
    excluded_queries: int,
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
            "excluded_query_signatures": excluded_queries,
            "ineligible_signatures": excluded + excluded_queries,
            "verified_signatures": len(labeled),
            "verified_within_block_pairs": verified_pairs,
            "pair_cap": cap,
            "judge_pair_cap": judge_cap,
            "strata": strata,
        },
    )


@dataclass(frozen=True)
class FrozenMiningBlock:
    """Hold the shared mining and replay results for a frozen block."""

    sample: BlockSample
    gold: dict[str, str]
    assignments_digest: str


def sample_frozen_block(
    view: BlockEligibility, path: Path, receipt: Mapping[str, object], policy: SamplingPolicy
) -> FrozenMiningBlock:
    """Mine one block using its frozen non-query assignments and evidence."""
    block = view.block
    queries = read_fit_pool_queries(path.with_name("queries.json"), block, receipt)
    decisions = {sid: view.eligibility(sid) for sid in view.rows}
    eligible = {sid for sid, decision in decisions.items() if decision.fit_ok}
    clusters, digest = _read_assignment_partition(path, receipt)
    require_assignment_coverage(block, clusters.keys(), eligible, queries)
    if ineligible_queries := queries - eligible:
        raise ValueError(f"Ineligible query signatures in {block!r}: {sorted(ineligible_queries)[:5]}")
    identities = {sid: next(iter(decisions[sid].identities), f"signature:{sid}") for sid in clusters}
    gold = {sid: orcid for sid in clusters if (orcid := decisions[sid].score_orcid) is not None}
    sample = select_block_pairs(
        block, identities, gold, clusters, len(view.rows) - len(eligible), policy, excluded_queries=len(queries)
    )
    return FrozenMiningBlock(sample, gold, digest)


def fit_pool_receipts(ledger: IdentityLedger, blocks: Mapping[str, BlockInput]) -> tuple[dict[str, Any], str]:
    """Read partition receipts and require one seed model and intake index."""
    receipts = {block: read_json_object(spec.assignments.with_name("manifest.json")) for block, spec in blocks.items()}
    model = require_assignment_receipts(
        ledger.roles,
        {block: receipt.get("binding") for block, receipt in receipts.items()},
        intake_index_sha256=ledger.evidence.name_counts_index_sha256,
    )
    return receipts, model


def mine_pairs(
    roles_path: Path,
    blocks: Mapping[str, BlockInput],
    output_path: Path,
    *,
    per_block: int = DEFAULT_PAIRS_PER_BLOCK,
    verified_pair_fraction: float = DEFAULT_VERIFIED_PAIR_FRACTION,
    per_block_floor: int = DEFAULT_PAIRS_PER_BLOCK_FLOOR,
    identity_evidence: Path,
    per_identity: int = 25,
    seed: int = 20261123,
) -> int:
    """Write capped stratum samples and counts for every frozen fit_pool block."""
    policy = SamplingPolicy(seed, per_block, verified_pair_fraction, per_block_floor, per_identity)
    roles = load_block_roles(roles_path)
    ledger = load_identity_ledger(identity_evidence, roles)
    normalized = {normalize_block_key(block): spec for block, spec in blocks.items()}
    if len(normalized) != len(blocks):
        raise ValueError("Duplicate normalized block keys in pairwise inputs")
    require_fit_pool_coverage(roles, normalized.keys())
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists() or output_path.exists():
        raise FileExistsError(output_path)

    assignment_manifests, model_digest = fit_pool_receipts(ledger, normalized)

    counts: dict[str, Any] = {}
    rows: list[tuple[str, str, str, str, float]] = []
    for block, spec in sorted(normalized.items()):
        view = ledger.for_block(block, dataset_dir=spec.arrow_dir)
        receipt = assignment_manifests[block]
        frozen = sample_frozen_block(view, spec.assignments, receipt, policy)
        sample = frozen.sample
        rows.extend(
            (block, left, right, stratum, inclusion) for (left, right), (stratum, inclusion) in sample.pairs.items()
        )
        counts[block] = {
            "fold": pairwise_fold(roles, block),
            "assignments_path": str(spec.assignments.resolve()),
            "assignment_binding": receipt["binding"],
            **sample.counts,
            "assignments_sha256": frozen.assignments_digest,
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
                BLOCK_BINDINGS_FIELD: roles.bindings_for(normalized.keys()),
                "index_sha256": ledger.evidence.name_counts_index_sha256,
                "model_sha256": model_digest,
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


def read_fit_pool_inputs(ledger: IdentityLedger, inputs_path: Path) -> dict[str, BlockInput]:
    """Resolve the complete fit pool from the linker partitions and identity evidence."""
    inputs = {}
    for raw_block, entry in read_json_object(inputs_path).items():
        block = normalize_block_key(raw_block)
        if ledger.roles.role_for(block) == "gate_thresh":
            continue
        if (
            block in inputs
            or not isinstance(entry, dict)
            or not isinstance(entry.get("seeds"), str)
            or not entry["seeds"]
        ):
            raise ValueError(f"Expected one partition path for {block!r} in {inputs_path}")
        inputs[block] = inputs_path.parent / entry["seeds"]
    require_fit_pool_coverage(ledger.roles, inputs.keys())
    return {
        block: BlockInput(directory, inputs[block])
        for block, directory in ledger.evidence.dataset_dirs(inputs.keys()).items()
    }


def main(argv: list[str] | None = None) -> int:
    """Sample pairs from frozen training blocks and their stored 1.4 assignments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--block-inputs", type=Path, required=True, help="Linker inputs.json with frozen seeds.")
    add_sampling_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ledger = load_identity_ledger(args.identity_evidence, load_block_roles(args.block_role_assignments))
    blocks = read_fit_pool_inputs(ledger, args.block_inputs)
    count = mine_pairs(
        args.block_role_assignments,
        blocks,
        args.output,
        per_block=args.pairs_per_block,
        verified_pair_fraction=args.verified_pair_fraction,
        per_block_floor=args.pairs_per_block_floor,
        identity_evidence=args.identity_evidence,
        per_identity=args.pairs_per_identity_per_stratum,
        seed=args.seed,
    )
    print(json.dumps({"blocks": len(blocks), "pairs": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
