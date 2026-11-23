"""Assign production blocks into disjoint S2AND 1.5 training, tuning, and test pools.

Closes blocks over shared verified ORCID identities (``cross_holders``) before
assignment so no author identity ever spans two pools. Permanently excludes
``f_matsen`` and pins legacy 1.4 giant and reviewed blocks to ``dev_test``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PERMANENT_EXCLUDED_BLOCKS = frozenset({"f_matsen"})
LEGACY_DEV_TEST_BLOCKS = frozenset({"a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park"})
VALID_ROLES = (
    "pairwise_train",
    "pairwise_val",
    "linker_train",
    "gate_fit",
    "gate_thresh",
    "eps_cal",
    "dev_test",
    "final_test",
)
ASSIGNABLE_ROLES = (
    "final_test",
    "eps_cal",
    "gate_thresh",
    "gate_fit",
    "linker_train",
    "pairwise_val",
    "pairwise_train",
)


@dataclass(frozen=True)
class BlockSpec:
    """Metadata for one candidate production block."""

    block_key: str
    signature_count: int
    orcids: frozenset[str] = frozenset()
    stratum: str = "default"


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


def build_orcid_components(blocks: Sequence[BlockSpec]) -> list[tuple[str, ...]]:
    """Group blocks connected by any shared verified ORCID into atomic components."""
    keys = sorted(block.block_key for block in blocks)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate block keys in input")
    uf = _UnionFind(keys)
    orcid_owner: dict[str, str] = {}
    for block in sorted(blocks, key=lambda item: item.block_key):
        for orcid in block.orcids:
            first = orcid_owner.get(orcid)
            if first is None:
                orcid_owner[orcid] = block.block_key
            else:
                uf.union(first, block.block_key)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        grouped[uf.find(key)].append(key)
    return [tuple(sorted(members)) for _, members in sorted(grouped.items())]


def verify_role_assignments(
    blocks: Sequence[BlockSpec],
    block_to_role: Mapping[str, str],
    *,
    excluded_blocks: Collection[str] = PERMANENT_EXCLUDED_BLOCKS,
    dev_test_blocks: Collection[str] = LEGACY_DEV_TEST_BLOCKS,
) -> dict[str, Any]:
    """Verify strict block, ORCID, excluded-block, and legacy dev_test invariants."""
    by_key = {block.block_key: block for block in blocks}
    excluded_set = set(excluded_blocks) | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_set = set(dev_test_blocks) | set(LEGACY_DEV_TEST_BLOCKS)

    for key, role in block_to_role.items():
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown role {role!r} for block {key!r}")
        if key in excluded_set:
            raise ValueError(f"Excluded block {key!r} must not appear in active assignments")

    for key in dev_set:
        if key in by_key and key not in excluded_set and block_to_role.get(key) != "dev_test":
            raise ValueError(
                f"Legacy/reviewed block {key!r} must be assigned to 'dev_test', got {block_to_role.get(key)!r}"
            )

    orcids_by_role: dict[str, set[str]] = defaultdict(set)
    for key, role in block_to_role.items():
        if key not in by_key:
            raise ValueError(f"Assigned block {key!r} missing from block metadata")
        orcids_by_role[role].update(by_key[key].orcids)

    roles_present = sorted(orcids_by_role)
    for idx, role_a in enumerate(roles_present):
        for role_b in roles_present[idx + 1 :]:
            overlap = orcids_by_role[role_a] & orcids_by_role[role_b]
            if overlap:
                raise ValueError(
                    f"ORCID overlap between {role_a!r} and {role_b!r}: "
                    f"count={len(overlap)}, sample={sorted(overlap)[:5]}"
                )

    role_counts = {role: sum(1 for r in block_to_role.values() if r == role) for role in VALID_ROLES}
    role_signatures = {
        role: sum(by_key[key].signature_count for key, r in block_to_role.items() if r == role) for role in VALID_ROLES
    }
    return {
        "total_assigned_blocks": len(block_to_role),
        "role_block_counts": role_counts,
        "role_signature_counts": role_signatures,
        "cross_role_orcid_overlap": 0,
    }


def assign_block_roles(
    blocks: Sequence[BlockSpec],
    role_targets: Mapping[str, int],
    *,
    seed: int = 20261123,
    extra_dev_test_blocks: Collection[str] = (),
    pinned_roles: Mapping[str, str] | None = None,
    excluded_blocks: Collection[str] = (),
) -> dict[str, Any]:
    """Deterministically assign blocks into ORCID-disjoint 1.5 pipeline roles.

    Args:
        blocks: Candidate blocks with signature counts, verified ORCID sets, and strata.
        role_targets: Desired block counts for roles in ``ASSIGNABLE_ROLES`` (e.g.
            ``{"final_test": 16, "eps_cal": 18, "gate_thresh": 12, "gate_fit": 16,
            "linker_train": 14, "pairwise_val": 10, "pairwise_train": 14}``).
        seed: Random seed for reproducible stratified component ordering.
        extra_dev_test_blocks: Additional previously reviewed blocks (e.g. 1.4 Batches 1-6)
            to pin permanently to ``dev_test``.
        pinned_roles: Optional explicit ``{block_key: role}`` overrides.
        excluded_blocks: Extra blocks to exclude alongside ``PERMANENT_EXCLUDED_BLOCKS``.

    Returns:
        Full assignment payload including ``block_to_role``, ``roles``, ``excluded_blocks``,
        ``cross_block_components``, and ``verification``.
    """
    invalid_targets = sorted(set(role_targets) - set(ASSIGNABLE_ROLES))
    if invalid_targets:
        raise ValueError(f"Unsupported target roles (use extra_dev_test_blocks for dev_test): {invalid_targets}")
    for role, count in role_targets.items():
        if count < 0:
            raise ValueError(f"Target count for {role!r} must be >= 0, got {count}")

    excluded_set = set(PERMANENT_EXCLUDED_BLOCKS) | set(excluded_blocks)
    dev_test_set = set(LEGACY_DEV_TEST_BLOCKS) | set(extra_dev_test_blocks)
    pins = dict(pinned_roles or {})
    for key in dev_test_set:
        if key in pins and pins[key] != "dev_test":
            raise ValueError(f"Block {key!r} is a legacy/reviewed dev_test block and cannot be pinned to {pins[key]!r}")
        pins[key] = "dev_test"

    active_blocks = [block for block in blocks if block.block_key not in excluded_set]
    present_excluded = sorted(block.block_key for block in blocks if block.block_key in excluded_set)
    by_key = {block.block_key: block for block in active_blocks}
    components = build_orcid_components(active_blocks)

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
            if role not in VALID_ROLES:
                raise ValueError(f"Unknown pinned role: {role!r}")
            for key in component:
                block_to_role[key] = role
        else:
            free_components.append(component)

    # Group free components by dominant stratum and shuffle deterministically within each stratum.
    rng = random.Random(seed)
    by_stratum: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for component in free_components:
        primary = max(component, key=lambda key: (by_key[key].signature_count, key))
        by_stratum[by_key[primary].stratum].append(component)
    ordered_free: list[tuple[str, ...]] = []
    stratum_queues: list[list[tuple[str, ...]]] = []
    for stratum in sorted(by_stratum):
        items = list(by_stratum[stratum])
        rng.shuffle(items)
        stratum_queues.append(items)
    # Interleave strata round-robin so every pool receives balanced stratum representation.
    while any(stratum_queues):
        for queue in stratum_queues:
            if queue:
                ordered_free.append(queue.pop(0))

    remaining = {
        role: max(0, int(role_targets.get(role, 0)) - sum(1 for r in block_to_role.values() if r == role))
        for role in ASSIGNABLE_ROLES
        if role in role_targets
    }
    target_order = [role for role in ASSIGNABLE_ROLES if role in role_targets]
    if not target_order:
        raise ValueError("role_targets must specify at least one assignable role")

    for component in ordered_free:
        # Pick the role with the highest remaining deficit ratio (tie-break by ASSIGNABLE_ROLES priority).
        candidates = [role for role in target_order if remaining.get(role, 0) > 0]
        if candidates:
            chosen = max(
                candidates,
                key=lambda role: (remaining[role] / max(1, int(role_targets[role])), -target_order.index(role)),
            )
            remaining[chosen] = max(0, remaining[chosen] - len(component))
        else:
            # Overflow goes to the role with the largest configured target (typically pairwise_train or linker_train).
            chosen = max(target_order, key=lambda role: (int(role_targets[role]), -target_order.index(role)))
        for key in component:
            block_to_role[key] = chosen

    verification = verify_role_assignments(
        active_blocks,
        block_to_role,
        excluded_blocks=excluded_set,
        dev_test_blocks=dev_test_set,
    )
    roles: dict[str, list[str]] = {role: [] for role in VALID_ROLES}
    for key, role in sorted(block_to_role.items()):
        roles[role].append(key)
    multi_block_components = [list(component) for component in components if len(component) > 1]
    return {
        "seed": int(seed),
        "excluded_blocks": present_excluded,
        "role_targets": {role: int(role_targets[role]) for role in target_order},
        "roles": roles,
        "block_to_role": dict(sorted(block_to_role.items())),
        "cross_block_components": multi_block_components,
        "verification": verification,
    }


def _load_blocks_json(path: Path) -> list[BlockSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list of block specs in {path}")
    specs: list[BlockSpec] = []
    for item in raw:
        specs.append(
            BlockSpec(
                block_key=str(item["block_key"]),
                signature_count=int(item["signature_count"]),
                orcids=frozenset(str(o) for o in item.get("orcids", ())),
                stratum=str(item.get("stratum", "default")),
            )
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks-json",
        type=Path,
        required=True,
        help="JSON list of {block_key, signature_count, orcids, stratum}.",
    )
    parser.add_argument("--targets-json", type=Path, required=True, help="JSON dict of {role: target_block_count}.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--dev-test-blocks", nargs="*", default=())
    parser.add_argument("--exclude-blocks", nargs="*", default=())
    args = parser.parse_args(argv)

    blocks = _load_blocks_json(args.blocks_json)
    targets = json.loads(args.targets_json.read_text(encoding="utf-8"))
    result = assign_block_roles(
        blocks,
        targets,
        seed=args.seed,
        extra_dev_test_blocks=args.dev_test_blocks,
        excluded_blocks=args.exclude_blocks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assignments.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    with (args.output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_key", "role"])
        for key, role in result["block_to_role"].items():
            writer.writerow([key, role])
    print(json.dumps(result["verification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
