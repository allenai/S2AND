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
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.text import normalize_orcid  # noqa: E402
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS  # noqa: E402
from scripts.production.data.prepare_identity_labels import normalize_block_key  # noqa: E402

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
) -> dict[str, Any]:
    """Verify strict block, ORCID, excluded-block, source-split, and legacy dev_test invariants."""
    by_key = {normalize_block_key(block.block_key): block for block in blocks}
    norm_mapping = {normalize_block_key(k): r for k, r in block_to_role.items()}
    excluded_set = {normalize_block_key(k) for k in excluded_blocks} | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_set = {normalize_block_key(k) for k in dev_test_blocks} | set(LEGACY_DEV_TEST_BLOCKS)

    for key, role in norm_mapping.items():
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown role {role!r} for block {key!r}")
        if key in excluded_set:
            raise ValueError(f"Excluded block {key!r} must not appear in active assignments")
        if key not in by_key:
            raise ValueError(f"Assigned block {key!r} missing from block metadata")
        if enforce_source_splits and by_key[key].source_split is not None:
            split = by_key[key].source_split
            if role == "final_test" and split != "holdout":
                raise ValueError(f"Block {key!r} has source_split={split!r} and cannot enter 'final_test'")
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

    roles_present = sorted(orcids_by_role)
    cross_role_orcids: set[str] = set()
    for idx, role_a in enumerate(roles_present):
        for role_b in roles_present[idx + 1 :]:
            overlap = orcids_by_role[role_a] & orcids_by_role[role_b]
            if overlap:
                cross_role_orcids.update(overlap)
                if closure_mode == "strict":
                    raise ValueError(
                        f"ORCID overlap between {role_a!r} and {role_b!r}: "
                        f"count={len(overlap)}, sample={sorted(overlap)[:5]}"
                    )

    role_counts = {role: sum(1 for r in norm_mapping.values() if r == role) for role in VALID_ROLES}
    role_signatures = {
        role: sum(by_key[key].signature_count for key, r in norm_mapping.items() if r == role) for role in VALID_ROLES
    }
    return {
        "total_assigned_blocks": len(norm_mapping),
        "role_block_counts": role_counts,
        "role_signature_counts": role_signatures,
        "closure_mode": closure_mode,
        "cross_role_orcid_overlap": 0 if closure_mode == "strict" else len(cross_role_orcids),
    }


def _eligible_roles_for_component(
    component: tuple[str, ...],
    by_key: Mapping[str, BlockSpec],
    candidate_roles: Sequence[str],
    *,
    enforce_source_splits: bool,
) -> list[str]:
    if not enforce_source_splits:
        return list(candidate_roles)
    splits = {by_key[k].source_split for k in component if by_key[k].source_split is not None}
    if not splits:
        return list(candidate_roles)
    eligible: list[str] = []
    for role in candidate_roles:
        if role == "final_test":
            if splits == {"holdout"}:
                eligible.append(role)
        elif role in TRAINING_AND_TUNING_ROLES:
            if splits == {"development"}:
                eligible.append(role)
        else:
            eligible.append(role)
    return eligible


def assign_block_roles(
    blocks: Sequence[BlockSpec],
    role_targets: Mapping[str, int],
    *,
    seed: int = 20261123,
    extra_dev_test_blocks: Collection[str] = (),
    pinned_roles: Mapping[str, str] | None = None,
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
        excluded_blocks: Extra blocks to exclude alongside ``PERMANENT_EXCLUDED_BLOCKS``.
        closure_mode: ``"strict"`` enforces zero cross-role ORCID overlap via union-find closure;
            ``"home_pool"`` avoids mega-component percolation and outputs ``orcid_home_role``
            so cross-pool ORCID mentions can be marked unscored.
        enforce_source_splits: When True, ``final_test`` accepts only ``holdout`` blocks and
            training/tuning roles accept only ``development`` blocks.
        require_pinned_present: When True, raise if any explicitly pinned/dev_test/excluded block
            is absent from ``blocks``.
    """
    invalid_targets = sorted(set(role_targets) - set(ASSIGNABLE_ROLES))
    if invalid_targets:
        raise ValueError(f"Unsupported target roles: {invalid_targets}")
    for role, count in role_targets.items():
        if count < 0:
            raise ValueError(f"Target count for {role!r} must be >= 0, got {count}")

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
            {normalize_block_key(k) for k in (pinned_roles or {})}
            | {normalize_block_key(k) for k in extra_dev_test_blocks}
            | {normalize_block_key(k) for k in excluded_blocks} - seen_keys
        )
        if missing_pins:
            raise ValueError(f"Pinned/excluded blocks missing from input: {missing_pins}")

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

    # Allocate within each stratum proportionally to remaining per-role targets using stable SHA-256 order.
    by_stratum: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for component in free_components:
        primary = max(component, key=lambda key: (by_key[key].signature_count, key))
        by_stratum[by_key[primary].stratum].append(component)

    unassigned: list[str] = []
    role_stratum_counts: dict[str, dict[str, int]] = {role: defaultdict(int) for role in target_order}
    for key, role in block_to_role.items():
        if role in role_stratum_counts:
            role_stratum_counts[role][by_key[key].stratum] += 1

    # Process each stratum independently so every role gets proportional share of every stratum.
    total_free_blocks = sum(len(comp) for comp in free_components)
    for stratum in sorted(by_stratum):
        stratum_comps = sorted(by_stratum[stratum], key=lambda comp: (_hash_rank(seed, comp[0]), comp[0]))
        stratum_size = sum(len(comp) for comp in stratum_comps)
        stratum_fraction = stratum_size / max(1, total_free_blocks)
        for component in stratum_comps:
            eligible = [
                role
                for role in _eligible_roles_for_component(
                    component, by_key, target_order, enforce_source_splits=enforce_source_splits
                )
                if remaining.get(role, 0) > 0
            ]
            if not eligible:
                unassigned.extend(component)
                continue
            # Score each eligible role by its proportional stratum deficit first, then overall remaining ratio.
            chosen = max(
                eligible,
                key=lambda role: (
                    (int(role_targets[role]) * stratum_fraction - role_stratum_counts[role][stratum])
                    / max(1.0, int(role_targets[role]) * stratum_fraction),
                    remaining[role] / max(1, int(role_targets[role])),
                    -target_order.index(role),
                ),
            )
            for key in component:
                block_to_role[key] = chosen
                role_stratum_counts[chosen][by_key[key].stratum] += 1
            remaining[chosen] = max(0, remaining[chosen] - len(component))

    verification = verify_role_assignments(
        active_blocks,
        block_to_role,
        excluded_blocks=excluded_set,
        dev_test_blocks=dev_test_set,
        closure_mode=closure_mode,
        enforce_source_splits=enforce_source_splits,
    )

    # For home_pool mode, assign every multi-pool ORCID a deterministic home role (largest mention count,
    # preferring dev_test/final_test over training pools so held-out identities stay clean).
    orcid_home_role: dict[str, str] = {}
    if closure_mode == "home_pool":
        orcid_role_blocks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for key, role in block_to_role.items():
            for orcid in by_key[key].orcids:
                orcid_role_blocks[orcid][role] += 1
        role_priority = {role: idx for idx, role in enumerate(ASSIGNABLE_ROLES)}
        for orcid, role_map in sorted(orcid_role_blocks.items()):
            if len(role_map) > 1:
                home = max(role_map, key=lambda r: (role_map[r], -role_priority.get(r, 99)))
                orcid_home_role[orcid] = home

    roles: dict[str, list[str]] = {role: [] for role in VALID_ROLES}
    for key, role in sorted(block_to_role.items()):
        roles[role].append(key)
    multi_block_components = [list(component) for component in components if len(component) > 1]
    component_sizes = sorted((len(c) for c in components), reverse=True)
    shortfalls = {role: remaining[role] for role in target_order if remaining[role] > 0}

    return {
        "seed": int(seed),
        "closure_mode": closure_mode,
        "enforce_source_splits": bool(enforce_source_splits),
        "excluded_blocks": present_excluded,
        "role_targets": {role: int(role_targets[role]) for role in target_order},
        "roles": roles,
        "block_to_role": dict(sorted(block_to_role.items())),
        "unassigned": sorted(unassigned),
        "shortfalls": shortfalls,
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
                source_split=str(item["source_split"]) if item.get("source_split") is not None else None,
            )
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks-json",
        type=Path,
        required=True,
        help="JSON list of {block_key, signature_count, orcids, stratum, source_split?}.",
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
    if args.pinned_roles_json is not None:
        raw_pins = json.loads(args.pinned_roles_json.read_text(encoding="utf-8"))
        if isinstance(raw_pins, dict) and isinstance(raw_pins.get("block_to_role"), dict):
            pinned_roles = {str(k): str(v) for k, v in raw_pins["block_to_role"].items()}
        elif isinstance(raw_pins, dict):
            pinned_roles = {str(k): str(v) for k, v in raw_pins.items()}
        else:
            raise ValueError(f"Expected JSON mapping in {args.pinned_roles_json}")

    blocks = _load_blocks_json(args.blocks_json)
    targets = json.loads(args.targets_json.read_text(encoding="utf-8"))
    result = assign_block_roles(
        blocks,
        targets,
        seed=args.seed,
        extra_dev_test_blocks=args.dev_test_blocks,
        pinned_roles=pinned_roles,
        excluded_blocks=args.exclude_blocks,
        closure_mode=args.closure_mode,
        enforce_source_splits=args.enforce_source_splits,
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
