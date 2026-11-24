"""Assign production blocks into disjoint S2AND 1.5 training, tuning, and test pools.

Features:
- Canonicalizes all block keys (matching both ``a khan`` and ``a_khan``) and ORCIDs
  (stripping invalid values and ``EXCLUDED_ORCID_GROUPS``).
- Permanently excludes ``f matsen`` and pins all legacy 1.4 giant/reviewed blocks
  (including ``s feldman``) to ``dev_test``.
- Supports ``closure_mode="strict"`` (zero cross-role ORCID overlap via union-find closure)
  and ``closure_mode="home_pool"`` (avoids real-graph mega-component percolation by assigning
  each cross-pool ORCID one home pool whose mentions in other pools are marked unscored,
  mirroring ``orcid-blocks-v2/cross_holders.parquet``).
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
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.text import normalize_orcid  # noqa: E402
from scripts.common.block_roles import (  # noqa: E402
    FINAL_TEST_ROLES,
    in_orcid_home_role,
    load_block_roles,
    normalize_block_key,
)
from scripts.intake.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS  # noqa: E402

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
    closure_mode: Literal["strict", "home_pool"] = "strict",
    enforce_source_splits: bool = False,
    orcid_home_role: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify strict block, ORCID, excluded-block, source-split, and legacy dev_test invariants."""
    by_key = {normalize_block_key(block.block_key): block for block in blocks}
    norm_mapping = {normalize_block_key(k): r for k, r in block_to_role.items()}
    excluded_set = {normalize_block_key(k) for k in excluded_blocks} | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_set = {normalize_block_key(k) for k in dev_test_blocks} | set(LEGACY_DEV_TEST_BLOCKS)
    home_roles = orcid_home_role or {}
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
    for orcid, role in home_roles.items():
        if orcid not in orcids_by_role.get(role, ()):
            raise ValueError(f"ORCID {orcid!r} is missing from its home role {role!r}")
    scored_counts = Counter(
        orcid
        for role, orcids in orcids_by_role.items()
        for orcid in orcids
        if in_orcid_home_role(orcid, role, home_roles)
    )
    raw_overlap = {orcid for orcid, count in raw_counts.items() if count > 1}
    scored_overlap = {orcid for orcid, count in scored_counts.items() if count > 1}
    if scored_overlap or (closure_mode == "strict" and raw_overlap):
        raise ValueError(f"ORCID overlap across roles: {sorted(scored_overlap or raw_overlap)[:5]}")

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
        "cross_role_orcid_overlap": len(scored_overlap),
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
    for stratum in sorted(strata, key=lambda value: (sizes[value], value)):
        queue = strata[stratum]
        missing = [role for role in FINAL_TEST_ROLES if not counts[role, stratum]]
        if len(queue) >= len(missing) and all(
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
    pinned_home_roles: Mapping[str, str] | None = None,
    excluded_blocks: Collection[str] = (),
    closure_mode: Literal["strict", "home_pool"] = "strict",
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
        pinned_home_roles: Previously scored homes to preserve, including single-role ORCIDs.
        excluded_blocks: Extra blocks to exclude alongside ``PERMANENT_EXCLUDED_BLOCKS``.
        closure_mode: ``"strict"`` enforces zero cross-role ORCID overlap via union-find closure;
            ``"home_pool"`` avoids mega-component percolation and outputs ``orcid_home_role``
            so cross-pool ORCID mentions can be marked unscored.
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
                | {normalize_block_key(k) for k in excluded_blocks}
            )
            - seen_keys
        )
        if missing_pins:
            raise ValueError(f"Pinned/excluded blocks missing from input: {missing_pins}")

    if conflicting := excluded_set & {normalize_block_key(key) for key in (pinned_roles or {})}:
        raise ValueError(f"Cannot exclude pinned blocks: {sorted(conflicting)}")

    active_blocks = [block for block in norm_blocks if block.block_key not in excluded_set]
    present_excluded = sorted(block.block_key for block in norm_blocks if block.block_key in excluded_set)
    by_key = {block.block_key: block for block in active_blocks}

    if closure_mode == "strict":
        components = build_orcid_components(active_blocks)
    elif closure_mode == "home_pool":
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
            if not enforce_source_splits
            or block.source_split == ("holdout" if role in FINAL_TEST_ROLES else "development")
            or block_to_role.get(block.block_key) == role
        }
        for role in target_order
    }
    if all(role in role_targets for role in FINAL_TEST_ROLES) and any(
        role_targets[role] < len(strata_by_role[role]) for role in FINAL_TEST_ROLES
    ):
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

    # Keep scored identities in evaluation when an author crosses into a development pool.
    orcid_home_role: dict[str, str] = {}
    if closure_mode == "home_pool" or pinned_home_roles:
        orcid_role_mentions: dict[str, Counter[str]] = defaultdict(Counter)
        evaluation_roles = {*FINAL_TEST_ROLES, "dev_test"}
        pinned_homes = pinned_home_roles or {}
        for key, role in block_to_role.items():
            for orcid in by_key[key].orcids:
                count = by_key[key].orcid_mention_counts.get(orcid, 1)
                if type(count) is not int or count < 1:
                    raise ValueError(f"Invalid ORCID mention count in block {key!r}: {count!r}")
                orcid_role_mentions[orcid][role] += count
        role_priority = {role: idx for idx, role in enumerate(ASSIGNABLE_ROLES)}
        for orcid, home in pinned_homes.items():
            if home not in orcid_role_mentions.get(orcid, {}):
                raise ValueError(f"Pinned ORCID {orcid!r} is missing from its home role {home!r}")
        for orcid, role_map in sorted(orcid_role_mentions.items()):
            home = pinned_homes.get(orcid)
            if closure_mode == "home_pool" and len(role_map) > 1:
                orcid_home_role[orcid] = home or max(
                    role_map, key=lambda role: (role in evaluation_roles, role_map[role], -role_priority[role])
                )

    verification = verify_role_assignments(
        active_blocks,
        block_to_role,
        excluded_blocks=excluded_set,
        dev_test_blocks=dev_test_set,
        closure_mode=closure_mode,
        enforce_source_splits=enforce_source_splits,
        orcid_home_role=orcid_home_role,
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
        "excluded_blocks": present_excluded,
        "role_targets": {role: int(role_targets[role]) for role in target_order},
        "roles": roles,
        "block_to_role": dict(sorted(block_to_role.items())),
        "block_orcids": {key: sorted(by_key[key].orcids) for key in sorted(block_to_role)},
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
        "orcid_home_role": orcid_home_role,
        "verification": verification,
    }


def load_blocks(path: Path, *, require_orcid_lists: bool = False) -> list[BlockSpec]:
    """Load block metadata from JSON or Parquet."""
    if path.suffix.casefold() == ".parquet":
        import pyarrow.parquet as pq

        raw = pq.read_table(path).to_pylist()
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of block specs in {path}")
    specs: list[BlockSpec] = []
    for item in raw:
        key = item.get("block_key") or item.get("cluster_block_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Missing block_key in {path}")
        orcids = item.get("orcids", None if require_orcid_lists else [])
        if not isinstance(orcids, list):
            raise ValueError(f"Expected an ORCID list for {key!r} in {path}")
        mention_counts = item.get("orcid_mention_counts")
        if mention_counts is None:
            mention_counts = Counter(orcid for raw in orcids if (orcid := normalize_orcid(str(raw))))
        elif isinstance(mention_counts, dict):
            normalized_counts: dict[str, int] = {}
            for raw_orcid, mentions in mention_counts.items():
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks",
        "--blocks-json",
        type=Path,
        required=True,
        help="JSON or Parquet with block_key, signature_count, orcids, size/script/region bands, and split.",
    )
    parser.add_argument("--targets-json", type=Path, required=True, help="JSON dict of {role: target_block_count}.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--closure-mode", choices=["strict", "home_pool"], default="strict")
    parser.add_argument("--enforce-source-splits", action="store_true")
    parser.add_argument(
        "--pinned-roles-json",
        type=Path,
        default=None,
        help="Optional JSON mapping {block_key: role} or prior assignments.json to preserve.",
    )
    parser.add_argument("--dev-test-blocks", nargs="*", default=())
    parser.add_argument("--exclude-blocks", nargs="*", default=())
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing assignments.json.")
    args = parser.parse_args(argv)

    out_json = args.output_dir / "assignments.json"
    if out_json.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {out_json}; pass --overwrite or use a fresh directory")

    pinned_roles: dict[str, str] | None = None
    pinned_home_roles: dict[str, str] | None = None
    excluded_blocks = list(args.exclude_blocks)
    if args.pinned_roles_json is not None:
        raw_pins = json.loads(args.pinned_roles_json.read_text(encoding="utf-8"))
        if isinstance(raw_pins, dict) and isinstance(raw_pins.get("block_to_role"), dict):
            prior = load_block_roles(args.pinned_roles_json)
            pinned_roles = prior.block_to_role
            pinned_home_roles = {
                orcid: prior.orcid_home_role.get(orcid, pinned_roles[block])
                for block, orcids in prior.block_orcids.items()
                for orcid in orcids
            }
            prior_exclusions = raw_pins.get("excluded_blocks", [])
            if not isinstance(prior_exclusions, list) or any(not isinstance(block, str) for block in prior_exclusions):
                raise ValueError(f"Expected an excluded block list in {args.pinned_roles_json}")
            excluded_blocks.extend(prior_exclusions)
        elif isinstance(raw_pins, dict):
            pinned_roles = {str(k): str(v) for k, v in raw_pins.items()}
        else:
            raise ValueError(f"Expected JSON mapping in {args.pinned_roles_json}")

    blocks = load_blocks(args.blocks, require_orcid_lists=True)
    targets = json.loads(args.targets_json.read_text(encoding="utf-8"))
    result = assign_block_roles(
        blocks,
        targets,
        seed=args.seed,
        extra_dev_test_blocks=args.dev_test_blocks,
        pinned_roles=pinned_roles,
        pinned_home_roles=pinned_home_roles,
        excluded_blocks=excluded_blocks,
        closure_mode=args.closure_mode,
        enforce_source_splits=args.enforce_source_splits,
        require_pinned_present=args.pinned_roles_json is not None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    with (args.output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_key", "role"])
        for key, role in result["block_to_role"].items():
            writer.writerow([key, role])
    print(json.dumps(result["verification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
