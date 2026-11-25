"""Assign production blocks into disjoint S2AND 1.5 training, tuning, and test pools.

Features:
- Canonicalizes all block keys (matching both ``a khan`` and ``a_khan``) and ORCIDs
  (stripping invalid values and ``EXCLUDED_ORCID_GROUPS``).
- Permanently excludes ``f matsen`` and pins all legacy 1.4 giant/reviewed blocks
  (including ``s feldman``) to ``dev_test``.
- Uses stable seeded SHA-256 hashing and per-stratum proportional allocation so strata
  never alias to roles and overflow blocks go to an explicit ``unassigned`` list rather than
  leaking into ``final_test``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._atomic_io import exclusive_file_lock  # noqa: E402
from s2and._sha256 import sha256_file  # noqa: E402
from s2and.text import normalize_orcid  # noqa: E402
from scripts.common.arrow_roots import _replace_json, read_json_object  # noqa: E402
from scripts.common.block_roles import (  # noqa: E402
    FINAL_TEST_ROLES,
    assigned_identity_roles,
    load_block_roles,
    normalize_block_key,
)
from scripts.common.identity_ledger import (  # noqa: E402
    EVIDENCE_SHA256_FIELD,
    FREEZE_BINDING,
    evidence_identities,
    load_identity_evidence,
)
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, excluded_orcid_groups_digest  # noqa: E402
from scripts.intake.build_freeze_input import mention_counts  # noqa: E402
from scripts.intake.requests_to_datasets import INTAKE_LOCK  # noqa: E402

PERMANENT_EXCLUDED_BLOCKS = frozenset({"f matsen"})
LEGACY_DEV_TEST_BLOCKS = frozenset(
    {"a khan", "a silva", "h wang", "j smith", "s feldman", "s gupta", "s lee", "s park"}
)
VALID_ROLES = (
    "pairwise_train",
    "pairwise_val",
    "linker_train",
    "gate_fit",
    "gate_thresh",
    "eps_cal",
    "dev_test",
    *FINAL_TEST_ROLES,
)
ASSIGNABLE_ROLES = (
    *FINAL_TEST_ROLES,
    "eps_cal",
    "gate_thresh",
    "gate_fit",
    "linker_train",
    "pairwise_val",
    "pairwise_train",
    "dev_test",
)
TRAINING_AND_TUNING_ROLES = frozenset(
    {"pairwise_train", "pairwise_val", "linker_train", "gate_fit", "gate_thresh", "eps_cal"}
)


@dataclass(frozen=True)
class BlockSpec:
    """Metadata for one candidate production block."""

    block_key: str
    signature_count: int
    orcids: frozenset[str] = frozenset()
    stratum: str = "default"
    source_split: str | None = None  # Optional Gold split: "development" or "holdout"
    orcid_mention_counts: Mapping[str, int] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt
        return root

    def union(self, left: str, right: str) -> None:
        ra, rb = self.find(left), self.find(right)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _clean_orcids(orcids: Collection[str]) -> frozenset[str]:
    cleaned: set[str] = set()
    for raw in orcids:
        norm = normalize_orcid(str(raw)) if raw else None
        if norm and norm not in EXCLUDED_ORCID_GROUPS:
            cleaned.add(norm)
    return frozenset(cleaned)


def _hash_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def build_orcid_components(blocks: Sequence[BlockSpec]) -> list[tuple[str, ...]]:
    """Group blocks connected by any shared valid, non-excluded ORCID into atomic components."""
    norm_keys = [normalize_block_key(block.block_key) for block in blocks]
    if len(norm_keys) != len(set(norm_keys)):
        raise ValueError("Duplicate normalized block keys in input")
    uf = _UnionFind(sorted(norm_keys))
    orcid_owner: dict[str, str] = {}
    for block in sorted(blocks, key=lambda item: normalize_block_key(item.block_key)):
        key = normalize_block_key(block.block_key)
        for orcid in _clean_orcids(block.orcids):
            first = orcid_owner.get(orcid)
            if first is None:
                orcid_owner[orcid] = key
            else:
                uf.union(first, key)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in sorted(norm_keys):
        grouped[uf.find(key)].append(key)
    return [tuple(sorted(members)) for _, members in sorted(grouped.items())]


def verify_role_assignments(
    blocks: Sequence[BlockSpec],
    block_to_role: Mapping[str, str],
    *,
    excluded_blocks: Collection[str] = PERMANENT_EXCLUDED_BLOCKS,
    dev_test_blocks: Collection[str] = LEGACY_DEV_TEST_BLOCKS,
    closure_mode: Literal["strict", "block_only"] = "strict",
    enforce_source_splits: bool = False,
) -> dict[str, Any]:
    """Verify strict block, ORCID, excluded-block, source-split, and legacy dev_test invariants."""
    by_key = {normalize_block_key(block.block_key): block for block in blocks}
    norm_mapping = {normalize_block_key(k): r for k, r in block_to_role.items()}
    excluded_set = {normalize_block_key(k) for k in excluded_blocks} | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_set = {normalize_block_key(k) for k in dev_test_blocks} | set(LEGACY_DEV_TEST_BLOCKS)
    _validate_source_splits(blocks, enforce_source_splits)

    for key, role in norm_mapping.items():
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown role {role!r} for block {key!r}")
        if key in excluded_set:
            raise ValueError(f"Excluded block {key!r} must not appear in active assignments")
        if key not in by_key:
            raise ValueError(f"Assigned block {key!r} missing from block metadata")
        if enforce_source_splits:
            split = by_key[key].source_split
            if role in FINAL_TEST_ROLES and split != "holdout":
                raise ValueError(f"Block {key!r} has source_split={split!r} and cannot enter {role!r}")
            if role in TRAINING_AND_TUNING_ROLES and split != "development":
                raise ValueError(f"Block {key!r} has source_split={split!r} and cannot enter {role!r}")

    for key in dev_set:
        if key in by_key and key not in excluded_set and norm_mapping.get(key) != "dev_test":
            raise ValueError(
                f"Legacy/reviewed block {key!r} must be assigned to 'dev_test', got {norm_mapping.get(key)!r}"
            )

    orcids_by_role: dict[str, set[str]] = defaultdict(set)
    for key, role in norm_mapping.items():
        orcids_by_role[role].update(_clean_orcids(by_key[key].orcids))

    raw_counts = Counter(orcid for orcids in orcids_by_role.values() for orcid in orcids)
    raw_overlap = {orcid for orcid, count in raw_counts.items() if count > 1}
    if closure_mode == "strict" and raw_overlap:
        raise ValueError(f"ORCID overlap across roles: {sorted(raw_overlap)[:5]}")

    role_counts = {role: sum(1 for r in norm_mapping.values() if r == role) for role in VALID_ROLES}
    role_signatures = {
        role: sum(by_key[key].signature_count for key, r in norm_mapping.items() if r == role) for role in VALID_ROLES
    }
    strata = sorted({block.stratum for block in blocks})
    counts = Counter((role, by_key[key].stratum) for key, role in norm_mapping.items())
    return {
        "total_assigned_blocks": len(norm_mapping),
        "role_block_counts": role_counts,
        "role_signature_counts": role_signatures,
        "role_block_counts_by_stratum": {
            role: {stratum: counts[role, stratum] for stratum in strata} for role in VALID_ROLES
        },
        "closure_mode": closure_mode,
        "cross_role_orcid_overlap": 0,
        "raw_cross_role_orcid_overlap": len(raw_overlap),
    }


def _validate_source_splits(blocks: Sequence[BlockSpec], enforce: bool) -> None:
    if enforce:
        for block in blocks:
            if block.source_split not in {"development", "holdout"}:
                raise ValueError(
                    f"Block {block.block_key!r} has missing or invalid source_split: {block.source_split!r}"
                )


def _allocate_strata(
    components: Sequence[tuple[str, ...]],
    by_key: Mapping[str, BlockSpec],
    remaining: dict[str, int],
    block_to_role: dict[str, str],
    *,
    seed: int,
    enforce_source_splits: bool,
) -> None:
    groups: dict[tuple[str, str], deque[tuple[str, ...]]] = defaultdict(deque)
    sizes: Counter[tuple[str, str]] = Counter()
    for component in sorted(components, key=lambda comp: (_hash_rank(seed, comp[0]), comp[0])):
        primary = max(component, key=lambda key: (by_key[key].signature_count, key))
        splits = {by_key[key].source_split for key in component} if enforce_source_splits else set()
        split = "holdout" if splits == {"holdout"} else "development" if splits == {"development"} else ""
        group = (by_key[primary].stratum, split)
        groups[group].append(component)
        sizes[group] += len(component)

    for role, target in remaining.items():
        allowed = "holdout" if role in FINAL_TEST_ROLES else "development"
        eligible = {
            group for group, size in sizes.items() if size and (not enforce_source_splits or group[1] == allowed)
        }
        total = sum(sizes[group] for group in eligible)
        target = min(target, total)
        # Spend the largest outstanding proportional quota before taking another block.
        deficits = {group: target * sizes[group] for group in eligible}
        tie_break = {group: _hash_rank(seed, f"{role}:{group}") for group in eligible}
        assigned = 0
        while assigned < target and eligible:
            group = max(eligible, key=lambda group: (deficits[group], tie_break[group]))
            component = groups[group].popleft()
            for key in component:
                block_to_role[key] = role
            assigned += len(component)
            sizes[group] -= len(component)
            deficits[group] -= total * len(component)
            if not groups[group]:
                eligible.remove(group)
        remaining[role] = max(0, remaining[role] - assigned)


def _reserve_role_strata(
    components: Sequence[tuple[str, ...]],
    by_key: Mapping[str, BlockSpec],
    remaining: dict[str, int],
    assignments: dict[str, str],
    *,
    seed: int,
    enforce_source_splits: bool,
) -> list[tuple[str, ...]]:
    groups: dict[tuple[str, str], deque[tuple[str, ...]]] = defaultdict(deque)
    for component in sorted(components, key=lambda comp: (_hash_rank(seed, comp[0]), comp[0])):
        splits = {by_key[key].source_split for key in component}
        split = next(iter(splits)) if enforce_source_splits and len(splits) == 1 else None
        if enforce_source_splits and split is None:
            continue
        primary = max(component, key=lambda key: (by_key[key].signature_count, key))
        groups[by_key[primary].stratum, split or ""].append(component)

    covered = {(role, by_key[key].stratum) for key, role in assignments.items()}
    reserved = set()
    paired = all(role in remaining for role in FINAL_TEST_ROLES)
    for (stratum, split), queue in sorted(groups.items(), key=lambda item: (len(item[1]), item[0])):
        for role in remaining:
            if paired and role in FINAL_TEST_ROLES:
                continue
            expected_split = "holdout" if role in FINAL_TEST_ROLES else "development"
            if split and split != expected_split:
                continue
            if queue and (role, stratum) not in covered and len(queue[0]) <= remaining[role]:
                component = queue.popleft()
                reserved.add(component[0])
                remaining[role] -= len(component)
                for key in component:
                    assignments[key] = role
                    covered.add((role, by_key[key].stratum))
    return [component for component in components if component[0] not in reserved]


def _allocate_sealed_pairs(
    components: Sequence[tuple[str, ...]],
    by_key: Mapping[str, BlockSpec],
    remaining: dict[str, int],
    assignments: dict[str, str],
    *,
    seed: int,
    enforce_source_splits: bool,
) -> list[tuple[str, ...]]:
    left, right = FINAL_TEST_ROLES
    if left not in remaining or right not in remaining:
        return list(components)
    strata: dict[str, deque[tuple[str, ...]]] = defaultdict(deque)
    for component in sorted(components, key=lambda comp: (_hash_rank(seed, comp[0]), comp[0])):
        if enforce_source_splits and any(by_key[key].source_split != "holdout" for key in component):
            continue
        primary = max(component, key=lambda key: (by_key[key].signature_count, key))
        strata[by_key[primary].stratum].append(component)
    counts = Counter((role, by_key[key].stratum) for key, role in assignments.items())
    sizes = {stratum: sum(map(len, queue)) for stratum, queue in strata.items()}
    total = sum(sizes.values())
    pair_target = min(remaining[left], remaining[right])
    assigned: set[str] = set()

    def take(role: str, queue: deque[tuple[str, ...]]) -> None:
        component = queue.popleft()
        assigned.add(component[0])
        remaining[role] -= len(component)
        for key in component:
            assignments[key] = role
            counts[role, by_key[key].stratum] += 1

    # Cover feasible rare strata before spending proportional quotas.
    for stratum in sorted(
        strata,
        key=lambda value: (
            len(strata[value]) == 1 and all(not counts[role, value] for role in FINAL_TEST_ROLES),
            sizes[value],
            value,
        ),
    ):
        queue = strata[stratum]
        missing = [role for role in FINAL_TEST_ROLES if not counts[role, stratum]]
        if len(queue) == 1 and len(missing) == 2:
            role = max(missing, key=remaining.__getitem__)
            if len(queue[0]) <= remaining[role]:
                take(role, queue)
        elif len(queue) >= len(missing) and all(
            len(component) <= remaining[role] for role, component in zip(missing, queue, strict=False)
        ):
            for role in missing:
                take(role, queue)

    # Fill both halves together so rounding cannot send the rare stratum to one half.
    deficits = {
        stratum: pair_target * size - total * max(counts[left, stratum], counts[right, stratum])
        for stratum, size in sizes.items()
    }
    while remaining[left] and remaining[right]:
        eligible = [
            stratum
            for stratum, queue in strata.items()
            if len(queue) >= 2 and len(queue[0]) <= remaining[left] and len(queue[1]) <= remaining[right]
        ]
        if not eligible:
            break
        stratum = max(eligible, key=lambda value: (deficits[value], value))
        queue = strata[stratum]
        deficits[stratum] -= total * max(len(queue[0]), len(queue[1]))
        take(left, queue)
        take(right, queue)
    return [component for component in components if component[0] not in assigned]


def assign_block_roles(
    blocks: Sequence[BlockSpec],
    role_targets: Mapping[str, int],
    *,
    seed: int = 20261123,
    extra_dev_test_blocks: Collection[str] = (),
    pinned_roles: Mapping[str, str] | None = None,
    excluded_blocks: Collection[str] = (),
    closure_mode: Literal["strict", "block_only"] = "strict",
    enforce_source_splits: bool = False,
    require_pinned_present: bool = False,
) -> dict[str, Any]:
    """Deterministically assign blocks into 1.5 pipeline roles with per-stratum proportional balance.

    Args:
        blocks: Candidate blocks (canonicalized on load so ``a khan`` == ``a_khan``).
        role_targets: Desired block counts per role in ``ASSIGNABLE_ROLES``. Overflow beyond
            targets is placed in ``unassigned`` (never dumped into ``final_test``).
        seed: Seed for stable SHA-256 component ranking.
        extra_dev_test_blocks: Additional reviewed blocks (e.g. 1.4 Batches 1-6) pinned to ``dev_test``.
        pinned_roles: Optional explicit ``{block_key: role}`` pins (e.g. prior frozen assignments).
        excluded_blocks: Extra blocks to exclude alongside ``PERMANENT_EXCLUDED_BLOCKS``.
        closure_mode: Use block-only allocation or require identity-disjoint blocks.
        enforce_source_splits: When True, both final test halves accept only ``holdout`` blocks and
            training/tuning roles accept only ``development`` blocks.
        require_pinned_present: When True, raise if any explicitly pinned/dev_test/excluded block
            is absent from ``blocks``.
    """
    role_targets = dict(role_targets)
    if "final_test" in role_targets:
        if any(role in role_targets for role in FINAL_TEST_ROLES):
            raise ValueError("Specify an aggregate final_test target or individual half targets")
        total = role_targets.pop("final_test")
        role_targets.update(final_test_a=(total + 1) // 2, final_test_b=total // 2)
    invalid_targets = sorted(set(role_targets) - set(ASSIGNABLE_ROLES))
    if invalid_targets:
        raise ValueError(f"Unsupported target roles: {invalid_targets}")
    for role, count in role_targets.items():
        if count < 0:
            raise ValueError(f"Target count for {role!r} must be >= 0, got {count}")
    _validate_source_splits(blocks, enforce_source_splits)

    excluded_set = {normalize_block_key(k) for k in excluded_blocks} | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_test_set = {normalize_block_key(k) for k in extra_dev_test_blocks} | set(LEGACY_DEV_TEST_BLOCKS)

    # Canonicalize input blocks, mapping normalized key -> original BlockSpec (with normalized key).
    norm_blocks: list[BlockSpec] = []
    seen_keys: set[str] = set()
    for block in blocks:
        key = normalize_block_key(block.block_key)
        if not key:
            raise ValueError(f"Empty block key in input: {block!r}")
        if key in seen_keys:
            raise ValueError(f"Duplicate normalized block key in input: {key!r}")
        seen_keys.add(key)
        norm_blocks.append(
            BlockSpec(
                block_key=key,
                signature_count=block.signature_count,
                orcids=_clean_orcids(block.orcids),
                stratum=block.stratum,
                source_split=block.source_split,
                orcid_mention_counts=block.orcid_mention_counts,
            )
        )

    pins: dict[str, str] = {}
    for raw_key, role in (pinned_roles or {}).items():
        key = normalize_block_key(raw_key)
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown pinned role {role!r} for block {raw_key!r}")
        pins[key] = role
    for key in dev_test_set:
        if key in pins and pins[key] != "dev_test":
            raise ValueError(f"Block {key!r} is a legacy/reviewed dev_test block and cannot be pinned to {pins[key]!r}")
        pins[key] = "dev_test"

    if require_pinned_present:
        missing_pins = sorted(
            (
                {normalize_block_key(k) for k in (pinned_roles or {})}
                | {normalize_block_key(k) for k in extra_dev_test_blocks}
            )
            - seen_keys
        )
        if missing_pins:
            raise ValueError(f"Pinned/excluded blocks missing from input: {missing_pins}")

    if conflicting := excluded_set & {normalize_block_key(key) for key in (pinned_roles or {})}:
        raise ValueError(f"Cannot exclude pinned blocks: {sorted(conflicting)}")

    active_blocks = [block for block in norm_blocks if block.block_key not in excluded_set]
    recorded_exclusions = sorted((seen_keys & excluded_set) | {normalize_block_key(k) for k in excluded_blocks})
    by_key = {block.block_key: block for block in active_blocks}

    if closure_mode == "strict":
        components = build_orcid_components(active_blocks)
    elif closure_mode == "block_only":
        components = [(block.block_key,) for block in sorted(active_blocks, key=lambda b: b.block_key)]
    else:
        raise ValueError(f"Unsupported closure_mode: {closure_mode!r}")

    block_to_role: dict[str, str] = {}
    free_components: list[tuple[str, ...]] = []
    for component in components:
        component_pins = {pins[key] for key in component if key in pins}
        if len(component_pins) > 1:
            raise ValueError(
                f"ORCID-linked component {component} has conflicting pinned roles: {sorted(component_pins)}"
            )
        if component_pins:
            role = next(iter(component_pins))
            for key in component:
                if (
                    enforce_source_splits
                    and role == "dev_test"
                    and by_key[key].source_split == "holdout"
                    and key not in pins
                ):
                    raise ValueError(f"Unpinned holdout block {key!r} cannot follow an ORCID component into dev_test")
                block_to_role[key] = role
        else:
            free_components.append(component)

    target_order = [role for role in ASSIGNABLE_ROLES if role in role_targets]
    if not target_order:
        raise ValueError("role_targets must specify at least one assignable role")

    remaining = {
        role: max(0, int(role_targets[role]) - sum(1 for r in block_to_role.values() if r == role))
        for role in target_order
    }

    strata_by_role = {
        role: {
            block.stratum
            for block in active_blocks
            if block_to_role.get(block.block_key) == role
            or (
                (role not in FINAL_TEST_ROLES or block.block_key not in block_to_role)
                and (
                    not enforce_source_splits
                    or block.source_split == ("holdout" if role in FINAL_TEST_ROLES else "development")
                )
            )
        }
        for role in target_order
    }
    if closure_mode != "block_only" and all(role in role_targets for role in FINAL_TEST_ROLES):
        sealed_counts = Counter(
            block.stratum
            for block in active_blocks
            if block_to_role.get(block.block_key) in FINAL_TEST_ROLES
            or (block.block_key not in block_to_role and (not enforce_source_splits or block.source_split == "holdout"))
        )
        for role in FINAL_TEST_ROLES:
            pinned_strata = {by_key[key].stratum for key, assigned in block_to_role.items() if assigned == role}
            required = pinned_strata | {stratum for stratum in strata_by_role[role] if sealed_counts[stratum] > 1}
            if role_targets[role] < len(required):
                raise ValueError("Sealed targets must allow at least one block per stratum in each half")

    free_components = _allocate_sealed_pairs(
        free_components, by_key, remaining, block_to_role, seed=seed, enforce_source_splits=enforce_source_splits
    )
    # Reserve coverage before proportional sampling can discard a rare stratum.
    free_components = _reserve_role_strata(
        free_components, by_key, remaining, block_to_role, seed=seed, enforce_source_splits=enforce_source_splits
    )
    if enforce_source_splits:
        cohorts = {
            FINAL_TEST_ROLES[0]: FINAL_TEST_ROLES,
            "pairwise_train": TRAINING_AND_TUNING_ROLES,
            "dev_test": ("dev_test",),
        }
    else:
        cohorts = {FINAL_TEST_ROLES[0]: ASSIGNABLE_ROLES}
    targets_by_cohort = {
        cohort: {role: count for role, count in remaining.items() if role in members}
        for cohort, members in cohorts.items()
    }
    # Reserve the total sample before role rounding so rare strata reach small roles.
    selected: dict[str, str] = {}
    _allocate_strata(
        free_components,
        by_key,
        {cohort: sum(targets.values()) for cohort, targets in targets_by_cohort.items()},
        selected,
        seed=seed,
        enforce_source_splits=enforce_source_splits,
    )
    selected_components: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for component in free_components:
        if component[0] in selected:
            selected_components[selected[component[0]]].append(component)
    for cohort, targets in targets_by_cohort.items():
        _allocate_strata(
            selected_components[cohort],
            by_key,
            targets,
            block_to_role,
            seed=seed,
            enforce_source_splits=enforce_source_splits,
        )
        remaining.update(targets)

    verification = verify_role_assignments(
        active_blocks,
        block_to_role,
        excluded_blocks=excluded_set,
        dev_test_blocks=dev_test_set,
        closure_mode=closure_mode,
        enforce_source_splits=enforce_source_splits,
    )
    roles: dict[str, list[str]] = {role: [] for role in VALID_ROLES}
    for key, role in sorted(block_to_role.items()):
        roles[role].append(key)
    multi_block_components = [list(component) for component in components if len(component) > 1]
    component_sizes = sorted((len(c) for c in components), reverse=True)
    shortfalls = {role: remaining[role] for role in target_order if remaining[role] > 0}

    stratum_shortfalls = {}
    for role in target_order:
        if role_targets[role] > 0:
            counts = verification["role_block_counts_by_stratum"][role]
            missing = sorted(stratum for stratum in strata_by_role[role] if not counts[stratum])
            if missing:
                stratum_shortfalls[role] = missing

    return {
        "seed": int(seed),
        "closure_mode": closure_mode,
        "enforce_source_splits": bool(enforce_source_splits),
        "excluded_blocks": recorded_exclusions,
        "role_targets": {role: int(role_targets[role]) for role in target_order},
        "roles": roles,
        "block_to_role": dict(sorted(block_to_role.items())),
        "block_orcids": {key: sorted(by_key[key].orcids) for key in sorted(block_to_role)},
        "block_signature_counts": {key: by_key[key].signature_count for key in sorted(block_to_role)},
        "unassigned": sorted(by_key.keys() - block_to_role.keys()),
        "shortfalls": shortfalls,
        "stratum_shortfalls": stratum_shortfalls,
        "cross_block_components": multi_block_components,
        "component_size_summary": {
            "total_components": len(components),
            "multi_block_components": len(multi_block_components),
            "largest_component_size": component_sizes[0] if component_sizes else 0,
            "top_5_component_sizes": component_sizes[:5],
        },
        "verification": verification,
    }


def load_blocks(path: Path, *, require_orcid_lists: bool = False) -> list[BlockSpec]:
    """Load block metadata from JSON or Parquet."""
    rows: Iterable[dict[str, Any]]
    if path.suffix.casefold() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        fields = {
            "block_key",
            "cluster_block_key",
            "signature_count",
            "orcids",
            "orcid_mention_counts",
            "size_band",
            "stratum",
            "script_region",
            "script",
            "region",
            "source_split",
            "split",
        }
        columns = sorted(fields & set(parquet.schema_arrow.names))
        rows = (
            dict(zip(columns, values, strict=True))
            for batch in parquet.iter_batches(columns=columns)
            for values in zip(*(batch.column(name).to_pylist() for name in columns), strict=True)
        )
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise ValueError(f"Expected a list of block specs in {path}")
        rows = cast(list[dict[str, Any]], raw)
    specs: list[BlockSpec] = []
    for item in rows:
        key = item.get("block_key") or item.get("cluster_block_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Missing block_key in {path}")
        orcids = item.get("orcids", None if require_orcid_lists else [])
        if not isinstance(orcids, list):
            raise ValueError(f"Expected an ORCID list for {key!r} in {path}")
        mention_counts = item.get("orcid_mention_counts")
        if mention_counts is None:
            mention_counts = Counter(orcid for raw in orcids if (orcid := normalize_orcid(str(raw))))
        elif isinstance(mention_counts, (dict, list)):
            normalized_counts: dict[str, int] = {}
            entries = mention_counts.items() if isinstance(mention_counts, dict) else mention_counts
            for raw_orcid, mentions in entries:
                orcid = normalize_orcid(str(raw_orcid))
                if not orcid or orcid in normalized_counts or type(mentions) is not int or mentions < 1:
                    raise ValueError(f"Invalid or duplicate ORCID mention count for {raw_orcid!r} in block {key!r}")
                normalized_counts[orcid] = mentions
            if missing := _clean_orcids(orcids) - normalized_counts.keys():
                raise ValueError(f"Missing ORCID mention counts for {sorted(missing)} in block {key!r}")
            mention_counts = normalized_counts
        else:
            raise ValueError(f"Expected ORCID mention counts for {key!r} in {path}")
        count = int(item["signature_count"])
        size = item.get("size_band") or ("<=1000" if count <= 1000 else "1001-15000" if count <= 15000 else ">15000")
        region = item.get("script_region") or "/".join(
            str(item[field]) for field in ("script", "region") if item.get(field)
        )
        stratum = item.get("stratum") or "/".join(str(value) for value in (size, region) if value) or "default"
        specs.append(
            BlockSpec(
                block_key=key,
                signature_count=count,
                orcids=frozenset(str(orcid) for orcid in orcids if orcid),
                stratum=str(stratum),
                source_split=item.get("source_split") or item.get("split"),
                orcid_mention_counts=mention_counts,
            )
        )
    return specs


def freeze_identity_roles(
    blocks_path: Path,
    evidence_path: Path,
    targets: Mapping[str, int],
    *,
    prior_path: Path | None = None,
    fresh: bool = False,
    extra_pins: Mapping[str, str] | None = None,
    dev_test_blocks: Collection[str] = (),
    excluded_blocks: Collection[str] = (),
    seed: int = 20261123,
) -> dict[str, Any]:
    """Freeze only exported blocks and discard every cross-role identity from scoring."""
    import pyarrow.parquet as pq

    evidence = load_identity_evidence(evidence_path)
    if evidence.snapshot_sha256 != sha256_file(evidence.snapshot_root / "manifest.json"):
        raise ValueError("Snapshot has changed; append its evidence before freezing roles")
    receipts = [
        read_json_object(path)["assignments_sha256"]
        for root in (evidence.snapshot_root, evidence.path.parent)
        if (path := root / FREEZE_BINDING).exists()
    ]
    if fresh and prior_path is not None:
        raise ValueError("Use either a prior role manifest or --fresh")
    if receipts and not fresh:
        if prior_path is None:
            raise ValueError("Snapshot or evidence already has a freeze binding; pass the prior manifest or --fresh")
        if set(receipts) != {sha256_file(prior_path)}:
            raise ValueError("The prior role manifest does not match the latest freeze binding")
    metadata = pq.read_schema(blocks_path).metadata or {}
    for digest_field, digest in {
        EVIDENCE_SHA256_FIELD: evidence.sha256,
        "snapshot_sha256": evidence.snapshot_sha256,
        "intake_sha256": evidence.intake_sha256,
    }.items():
        if metadata.get(digest_field.encode()) != digest.encode():
            raise ValueError(f"Freeze input {digest_field} does not match the identity evidence")
    blocks = load_blocks(blocks_path, require_orcid_lists=True)
    if {block.block_key for block in blocks} != evidence.blocks.keys():
        raise ValueError("Freeze input must cover exactly the exported blocks")
    evidence_blocks = evidence.block_digests()
    prior = read_json_object(prior_path) if prior_path else None
    pins: dict[str, str] = {}
    prior_exclusions: list[str] = []
    prior_unassigned: set[str] = set()
    if prior_path is not None and prior is not None:
        if prior.get("format_version") != 2 or prior.get("intake_sha256") != evidence.intake_sha256:
            raise ValueError("Re-freeze requires a prior ledger-backed manifest from the same intake")
        if any(evidence_blocks.get(block) != digest for block, digest in prior["evidence_blocks"].items()):
            raise ValueError("Prior evidence partitions changed or are missing")
        pins.update(load_block_roles(prior_path).block_to_role)
        prior_exclusions = prior["excluded_blocks"]
        prior_unassigned = set(prior["unassigned"])
        prior_blocks = prior["evidence_blocks"].keys()
        requested_changes = {
            normalize_block_key(block) for block in (*dev_test_blocks, *excluded_blocks)
        } & prior_blocks
        if requested_changes:
            raise ValueError(f"Re-freeze cannot move prior blocks: {sorted(requested_changes)}")
    extra = {normalize_block_key(block): role for block, role in (extra_pins or {}).items()}
    if len(extra) != len(extra_pins or {}):
        raise ValueError("Duplicate normalized keys in --extra-pins-json")
    if prior is not None and extra.keys() & (prior["evidence_blocks"].keys() - prior_unassigned):
        raise ValueError("Extra pins may only name new or previously unassigned exported blocks")
    pins.update(extra)
    roles = assign_block_roles(
        blocks,
        targets,
        seed=seed,
        extra_dev_test_blocks=dev_test_blocks,
        pinned_roles=pins,
        excluded_blocks=(*prior_exclusions, *excluded_blocks),
        closure_mode="block_only",
        enforce_source_splits=True,
        require_pinned_present=True,
    )
    mapping = roles["block_to_role"]
    block_orcids = {block.block_key: sorted(block.orcids) for block in blocks}
    identity_roles = assigned_identity_roles(block_orcids, mapping)
    cross_role = {orcid for orcid, assigned in identity_roles.items() if len(assigned) > 1}
    identities_by_role: dict[str, set[str]] = defaultdict(set)
    mentions_by_role: Counter[str] = Counter()
    for block in blocks:
        key = block.block_key
        evidence_rows = evidence.read_block(key)
        counts = mention_counts(evidence_rows)
        if (
            counts != block.orcid_mention_counts
            or set(counts) != block.orcids
            or len(evidence_rows) != block.signature_count
        ):
            raise ValueError(f"Freeze input differs from the exported identity evidence for {key!r}")
        if key not in mapping:
            continue
        role = mapping[key]
        identities_by_role[role].update(cross_role & counts.keys())
        if cross_role & counts.keys():
            mentions_by_role[role] += len(
                {
                    (row["corpus_paper_id"], row["position"])
                    for row in evidence_rows.values()
                    if not row["reviewed_error"] and not evidence_identities(row).isdisjoint(cross_role)
                }
            )
    for key in ("closure_mode", "cross_block_components", "component_size_summary"):
        roles.pop(key)
    roles["verification"].pop("closure_mode")
    roles["verification"]["raw_cross_role_orcid_overlap"] = len(cross_role)
    roles["verification"]["cross_role_identities_by_role"] = {
        role: len(identities_by_role[role]) for role in VALID_ROLES
    }
    roles["verification"]["cross_role_mentions_by_role"] = {role: mentions_by_role[role] for role in VALID_ROLES}
    return {
        **roles,
        "format_version": 2,
        "excluded_orcid_groups_sha256": excluded_orcid_groups_digest(),
        EVIDENCE_SHA256_FIELD: evidence.sha256,
        "snapshot_sha256": evidence.snapshot_sha256,
        "intake_sha256": evidence.intake_sha256,
        "evidence_blocks": evidence_blocks,
        "block_orcids": block_orcids,
        "block_source_splits": {block.block_key: block.source_split for block in blocks},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        "--blocks-json",
        type=Path,
        required=True,
        help="Parquet produced by build_freeze_input.",
    )
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True, help="JSON dict of {role: target_block_count}.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument(
        "--pinned-roles-json",
        type=Path,
        default=None,
        help="Prior assignments.json to preserve on a re-freeze.",
    )
    parser.add_argument("--extra-pins-json", type=Path, help="Pin new or previously unassigned blocks to roles.")
    parser.add_argument("--dev-test-blocks", nargs="*", default=())
    parser.add_argument("--exclude-blocks", nargs="*", default=())
    parser.add_argument("--fresh", action="store_true", help="Start a new freeze lineage for this intake.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing assignments.json.")
    args = parser.parse_args(argv)

    evidence = load_identity_evidence(args.identity_evidence)
    with exclusive_file_lock(evidence.snapshot_root / INTAKE_LOCK):
        out_json = args.output_dir / "assignments.json"
        if out_json.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing {out_json}; pass --overwrite or use a fresh directory"
            )

        if out_json.exists() and (
            args.pinned_roles_json is None or sha256_file(args.pinned_roles_json) != sha256_file(out_json)
        ):
            raise ValueError("Pass the existing assignments.json as the prior manifest when re-freezing")
        result = freeze_identity_roles(
            args.blocks,
            args.identity_evidence,
            read_json_object(args.targets_json),
            prior_path=args.pinned_roles_json,
            fresh=args.fresh,
            extra_pins=read_json_object(args.extra_pins_json) if args.extra_pins_json else None,
            dev_test_blocks=args.dev_test_blocks,
            excluded_blocks=args.exclude_blocks,
            seed=args.seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _replace_json(out_json, result)
        with (args.output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["block_key", "role"])
            for key, role in result["block_to_role"].items():
                writer.writerow([key, role])
        receipt = {"assignments_sha256": sha256_file(out_json)}
        for root in (evidence.snapshot_root, evidence.path.parent):
            _replace_json(root / FREEZE_BINDING, receipt)
    print(
        json.dumps(
            {
                **result["verification"],
                "shortfalls": result["shortfalls"],
                "stratum_shortfalls": result["stratum_shortfalls"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
