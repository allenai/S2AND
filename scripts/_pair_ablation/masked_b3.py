"""ORCID-masked B-cubed helpers for giant linker candidate components.

ORCID is used only to choose and score evaluation members.  Callers are
responsible for disabling ORCID constraints on the pairwise clusterer before
building or scoring features.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from s2and.eval import b3_precision_recall_fscore
from scripts._pair_ablation.evaluation import (
    B3EvaluationPlan,
    B3PlanBlock,
    BlockLinkage,
    predicted_clusters_at_threshold,
)


@dataclass(frozen=True, slots=True)
class MaskedB3Selection:
    """A bounded component sample and the ORCID labels hidden from clustering."""

    plan: B3EvaluationPlan
    target_gold: dict[str, str]
    stats: dict[str, object]


def _stable_rank(seed: int, *parts: str) -> bytes:
    payload = "\0".join((str(int(seed)), *parts)).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _pair_count(size: int) -> int:
    return size * (size - 1) // 2


def _size_bin(size: int) -> int:
    return int(math.log2(max(1, size)))


def select_masked_orcid_components(
    *,
    dataset: str,
    components: Mapping[str, Sequence[str]],
    orcid_by_signature: Mapping[str, str],
    pair_budget: int,
    max_block_size: int,
    random_seed: int,
) -> MaskedB3Selection:
    """Select complete repeated-ORCID groups and bounded component distractors.

    Selection is round-robin across group-size and fragmentation strata.  A
    selected ORCID contributes all of its component-covered signatures and all
    components containing those signatures.  Each component retains every
    selected target and is deterministically filled with distractors up to
    ``max_block_size``.
    """

    if not dataset:
        raise ValueError("dataset must not be empty")
    if pair_budget <= 0:
        raise ValueError("pair_budget must be positive")
    if max_block_size < 2:
        raise ValueError("max_block_size must be at least two")

    normalized_components: dict[str, tuple[str, ...]] = {}
    component_by_signature: dict[str, str] = {}
    for raw_key, raw_members in components.items():
        key = str(raw_key)
        members = tuple(str(member) for member in raw_members)
        if not key:
            raise ValueError("component keys must not be empty")
        if not members:
            raise ValueError(f"component {key!r} must not be empty")
        if len(members) != len(set(members)):
            raise ValueError(f"component {key!r} contains duplicate signatures")
        for signature_id in members:
            previous = component_by_signature.setdefault(signature_id, key)
            if previous != key:
                raise ValueError(f"signature {signature_id!r} belongs to components {previous!r} and {key!r}")
        normalized_components[key] = members

    normalized_orcids = {
        str(signature_id): str(orcid) for signature_id, orcid in orcid_by_signature.items() if str(orcid)
    }
    global_counts = Counter(normalized_orcids.values())
    repeated_orcids = {orcid for orcid, count in global_counts.items() if count >= 2}
    global_members: dict[str, list[str]] = defaultdict(list)
    covered_members: dict[str, list[str]] = defaultdict(list)
    for signature_id, orcid in normalized_orcids.items():
        if orcid not in repeated_orcids:
            continue
        global_members[orcid].append(signature_id)
        if signature_id in component_by_signature:
            covered_members[orcid].append(signature_id)

    eligible = {orcid: tuple(sorted(members)) for orcid, members in covered_members.items() if len(members) >= 2}
    if not eligible:
        raise ValueError(f"{dataset} has no repeated ORCID with at least two component-covered signatures")

    group_components = {
        orcid: frozenset(component_by_signature[signature_id] for signature_id in members)
        for orcid, members in eligible.items()
    }
    component_cost = {
        key: _pair_count(min(len(members), max_block_size)) for key, members in normalized_components.items()
    }
    strata: dict[tuple[bool, int], list[str]] = defaultdict(list)
    for orcid, members in eligible.items():
        strata[(len(group_components[orcid]) > 1, _size_bin(len(members)))].append(orcid)
    for values in strata.values():
        values.sort(key=lambda orcid: (_stable_rank(random_seed, dataset, "group", orcid), orcid))

    ordered_groups: list[str] = []
    offsets = {key: 0 for key in strata}
    while True:
        added = False
        for stratum in sorted(strata):
            offset = offsets[stratum]
            if offset < len(strata[stratum]):
                ordered_groups.append(strata[stratum][offset])
                offsets[stratum] = offset + 1
                added = True
        if not added:
            break

    selected_groups: list[str] = []
    selected_components: set[str] = set()
    used_pairs = 0
    for orcid in ordered_groups:
        new_components = group_components[orcid].difference(selected_components)
        added_pairs = sum(component_cost[key] for key in new_components)
        if used_pairs + added_pairs > pair_budget:
            continue
        selected_groups.append(orcid)
        selected_components.update(new_components)
        used_pairs += added_pairs
    if not selected_groups:
        cheapest = min(
            eligible,
            key=lambda orcid: (
                sum(component_cost[key] for key in group_components[orcid]),
                _stable_rank(random_seed, dataset, "fallback", orcid),
                orcid,
            ),
        )
        selected_groups.append(cheapest)
        selected_components.update(group_components[cheapest])
        used_pairs = sum(component_cost[key] for key in selected_components)

    selected_orcids = set(selected_groups)
    target_gold = {signature_id: orcid for orcid in selected_groups for signature_id in eligible[orcid]}
    plan_blocks: list[B3PlanBlock] = []
    plan_gold: list[tuple[str, str]] = []
    for component_key in sorted(selected_components):
        members = normalized_components[component_key]
        targets = [signature_id for signature_id in members if signature_id in target_gold]
        if len(targets) > max_block_size:
            raise ValueError(
                f"component {component_key!r} has {len(targets)} selected targets, "
                f"exceeding max_block_size={max_block_size}"
            )
        distractors = [signature_id for signature_id in members if signature_id not in target_gold]
        distractors.sort(
            key=lambda signature_id: (
                _stable_rank(random_seed, dataset, "distractor", component_key, signature_id),
                signature_id,
            )
        )
        selected_members = tuple(targets + distractors[: max_block_size - len(targets)])
        plan_blocks.append(B3PlanBlock(component_key, selected_members))
        plan_gold.extend(
            (
                signature_id,
                (
                    f"orcid:{target_gold[signature_id]}"
                    if signature_id in target_gold
                    else f"masked-singleton:{signature_id}"
                ),
            )
            for signature_id in selected_members
        )

    actual_pairs = sum(_pair_count(len(block.signatures)) for block in plan_blocks)
    if actual_pairs != used_pairs:
        raise AssertionError(f"masked B3 pair accounting mismatch: actual={actual_pairs} expected={used_pairs}")
    if actual_pairs > pair_budget and len(selected_groups) != 1:
        raise AssertionError(f"masked B3 selection exceeded pair budget: {actual_pairs} > {pair_budget}")
    plan = B3EvaluationPlan(
        dataset=dataset,
        role="heldout_full",
        evaluation_seed=int(random_seed),
        pair_budget=int(pair_budget),
        blocks=tuple(plan_blocks),
        gold_assignments=tuple(plan_gold),
    )
    plan.identity_payload()

    repeated_signature_count = sum(global_counts[orcid] for orcid in repeated_orcids)
    covered_repeated_signature_count = sum(len(covered_members[orcid]) for orcid in repeated_orcids)
    selected_fragmented = sum(len(group_components[orcid]) > 1 for orcid in selected_groups)
    stats: dict[str, object] = {
        "dataset": dataset,
        "pair_budget": int(pair_budget),
        "max_block_size": int(max_block_size),
        "component_count": len(normalized_components),
        "component_signature_count": len(component_by_signature),
        "orcid_signature_count": len(normalized_orcids),
        "repeated_orcid_group_count": len(repeated_orcids),
        "repeated_orcid_signature_count": repeated_signature_count,
        "component_covered_repeated_orcid_signature_count": covered_repeated_signature_count,
        "component_covered_repeated_orcid_signature_fraction": (
            covered_repeated_signature_count / repeated_signature_count
        ),
        "eligible_orcid_group_count": len(eligible),
        "eligible_fragmented_orcid_group_count": sum(len(group_components[orcid]) > 1 for orcid in eligible),
        "selected_orcid_group_count": len(selected_groups),
        "selected_fragmented_orcid_group_count": selected_fragmented,
        "selected_orcid_signature_count": len(target_gold),
        "selected_component_count": len(selected_components),
        "selected_block_signature_count": sum(len(block.signatures) for block in plan_blocks),
        "selected_pair_count": actual_pairs,
        "selected_orcids_sha256": hashlib.sha256("\n".join(sorted(selected_orcids)).encode("utf-8")).hexdigest(),
    }
    return MaskedB3Selection(plan=plan, target_gold=target_gold, stats=stats)


def masked_b3_for_threshold(
    linkages: Mapping[str, BlockLinkage],
    target_gold: Mapping[str, str],
    threshold: float,
    *,
    dataset_prefix: str,
) -> tuple[float, float, float]:
    """Compute B-cubed after projecting predicted clusters to masked targets."""

    predicted_full = predicted_clusters_at_threshold(
        linkages,
        threshold,
        dataset_prefix=dataset_prefix,
    )
    target_ids = {str(signature_id) for signature_id in target_gold}
    predicted: dict[str, list[tuple[str, str]]] = {}
    observed: set[str] = set()
    for cluster_id, members in predicted_full.items():
        kept = [member for member in members if member[1] in target_ids]
        if kept:
            predicted[cluster_id] = kept
            observed.update(member[1] for member in kept)
    if observed != target_ids:
        raise ValueError(
            "masked B3 predicted/target mismatch: "
            f"missing={sorted(target_ids - observed)[:5]} orphan={sorted(observed - target_ids)[:5]}"
        )
    truth: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for signature_id, orcid in target_gold.items():
        truth[f"{dataset_prefix}:orcid:{orcid}"].append((dataset_prefix, str(signature_id)))
    precision, recall, f1, *_ = b3_precision_recall_fscore(dict(truth), predicted)
    return float(precision), float(recall), float(f1)


def masked_component_ceiling(
    plan: B3EvaluationPlan,
    target_gold: Mapping[str, str],
    *,
    dataset_prefix: str,
) -> tuple[float, float, float]:
    """Return the ceiling when ORCID members can merge only inside a component."""

    predicted: dict[str, list[tuple[str, str]]] = defaultdict(list)
    observed: set[str] = set()
    for block in plan.blocks:
        for signature_id in block.signatures:
            if signature_id not in target_gold:
                continue
            orcid = target_gold[signature_id]
            predicted[f"{dataset_prefix}:{block.block_key}:orcid:{orcid}"].append((dataset_prefix, signature_id))
            observed.add(signature_id)
    if observed != set(target_gold):
        raise ValueError("masked component ceiling plan does not contain every target")
    truth: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for signature_id, orcid in target_gold.items():
        truth[f"{dataset_prefix}:orcid:{orcid}"].append((dataset_prefix, signature_id))
    precision, recall, f1, *_ = b3_precision_recall_fscore(dict(truth), dict(predicted))
    return float(precision), float(recall), float(f1)


__all__ = [
    "MaskedB3Selection",
    "masked_b3_for_threshold",
    "masked_component_ceiling",
    "select_masked_orcid_components",
]
