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
import tempfile
from collections import Counter, defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._atomic_io import exclusive_file_lock  # noqa: E402
from s2and._sha256 import sha256_file  # noqa: E402
from s2and.text import normalize_orcid  # noqa: E402
from scripts.common.arrow_roots import _replace_json, read_json_object  # noqa: E402
from scripts.common.block_roles import (  # noqa: E402
    FINAL_TEST_ROLE,
    VALID_ROLES,
    BlockRoles,
    assigned_identity_roles,
    assignment_digest,
    load_block_roles,
    normalize_block_key,
)
from scripts.common.identity_ledger import (  # noqa: E402
    EVIDENCE_SHA256_FIELD,
    FREEZE_BINDING,
    FREEZE_SHA256_FIELD,
    SNAPSHOT_BINDING,
    IdentityLedger,
    evidence_identities,
    gate_veto_identities,
    load_identity_evidence,
)
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, excluded_orcid_groups_digest  # noqa: E402
from scripts.intake.build_freeze_input import GIANT_SIGNATURE_COUNT, mention_counts, size_band  # noqa: E402
from scripts.intake.requests_to_datasets import INTAKE_LOCK  # noqa: E402

PERMANENT_EXCLUDED_BLOCKS = frozenset({"f matsen"})
LEGACY_DEV_TEST_BLOCKS = frozenset(
    {"a khan", "a silva", "h wang", "j smith", "s feldman", "s gupta", "s lee", "s park"}
)
ASSIGNABLE_ROLES = (
    FINAL_TEST_ROLE,
    "eps_cal",
    "gate_thresh",
    "fit_pool",
    "dev_test",
)
TRAINING_AND_TUNING_ROLES = frozenset({"fit_pool", "gate_thresh", "eps_cal"})
DEFAULT_PAIRWISE_VALIDATION_FRACTION = 0.15
DEFAULT_GATE_TRAINING_BLOCK_COUNT = 20


@dataclass(frozen=True)
class BlockSpec:
    """Metadata for one candidate production block."""

    block_key: str
    signature_count: int
    orcids: frozenset[str] = frozenset()
    stratum: str = "default"
    source_split: str | None = None  # Optional Gold split: "development" or "holdout"
    orcid_mention_counts: Mapping[str, int] = field(default_factory=dict)


def _clean_orcids(orcids: Collection[str]) -> frozenset[str]:
    cleaned: set[str] = set()
    for raw in orcids:
        norm = normalize_orcid(str(raw)) if raw else None
        if norm and norm not in EXCLUDED_ORCID_GROUPS:
            cleaned.add(norm)
    return frozenset(cleaned)


def _hash_rank(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def verify_role_assignments(
    blocks: Sequence[BlockSpec],
    block_to_role: Mapping[str, str],
    *,
    excluded_blocks: Collection[str] = PERMANENT_EXCLUDED_BLOCKS,
    dev_test_blocks: Collection[str] = LEGACY_DEV_TEST_BLOCKS,
    enforce_source_splits: bool = False,
) -> dict[str, Any]:
    """Verify block, exclusion, source-split, and legacy dev_test invariants."""
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
            if role == FINAL_TEST_ROLE and split != "holdout":
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


@dataclass
class _GiantCaps:
    remaining: dict[str, int]
    giants: set[str]

    def allows(self, role: str, key: str) -> bool:
        return role not in self.remaining or key not in self.giants or self.remaining[role] > 0

    def take(self, role: str, key: str) -> None:
        if role in self.remaining and key in self.giants:
            self.remaining[role] -= 1


def _allocate_strata(
    blocks: Sequence[str],
    by_key: Mapping[str, BlockSpec],
    remaining: dict[str, int],
    block_to_role: dict[str, str],
    *,
    seed: int,
    enforce_source_splits: bool,
    giant_caps: _GiantCaps | None = None,
) -> None:
    groups: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    sizes: Counter[tuple[str, str]] = Counter()
    for key in sorted(blocks, key=lambda key: (_hash_rank(seed, key), key)):
        spec = by_key[key]
        group = (spec.stratum, (spec.source_split or "") if enforce_source_splits else "")
        groups[group].append(key)
        sizes[group] += 1

    for role, target in remaining.items():
        allowed = "holdout" if role == FINAL_TEST_ROLE else "development"
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
            queue = groups[group]
            if giant_caps is None or giant_caps.allows(role, queue[0]):
                key = queue.popleft()
            else:
                candidate = next((comp for comp in queue if giant_caps.allows(role, comp)), None)
                if candidate is None:
                    eligible.remove(group)
                    continue
                key = candidate
                queue.remove(key)
            if giant_caps is not None:
                giant_caps.take(role, key)
            block_to_role[key] = role
            assigned += 1
            sizes[group] -= 1
            deficits[group] -= total
            if not groups[group]:
                eligible.remove(group)
        remaining[role] = max(0, remaining[role] - assigned)


def _reserve_role_strata(
    blocks: Sequence[str],
    by_key: Mapping[str, BlockSpec],
    remaining: dict[str, int],
    assignments: dict[str, str],
    *,
    seed: int,
    enforce_source_splits: bool,
    giant_caps: _GiantCaps | None = None,
) -> list[str]:
    groups: dict[tuple[str, str], deque[str]] = defaultdict(deque)
    for key in sorted(blocks, key=lambda key: (_hash_rank(seed, key), key)):
        spec = by_key[key]
        split = (spec.source_split or "") if enforce_source_splits else ""
        groups[spec.stratum, split].append(key)

    covered = {(role, by_key[key].stratum) for key, role in assignments.items()}
    reserved = set()
    for (stratum, split), queue in sorted(groups.items(), key=lambda item: (len(item[1]), item[0])):
        for role in remaining:
            expected_split = "holdout" if role == FINAL_TEST_ROLE else "development"
            if split and split != expected_split:
                continue
            if (role, stratum) not in covered and remaining[role]:
                key = next((key for key in queue if giant_caps is None or giant_caps.allows(role, key)), None)
                if key is None:
                    continue
                queue.remove(key)
                if giant_caps is not None:
                    giant_caps.take(role, key)
                reserved.add(key)
                remaining[role] -= 1
                assignments[key] = role
                covered.add((role, stratum))
    return [key for key in blocks if key not in reserved]


def _pairwise_validation_fold(
    blocks: Mapping[str, BlockSpec],
    assignments: Mapping[str, str],
    fraction: float,
    seed: int,
    *,
    prior_fit: Collection[str] = (),
    prior_validation: Collection[str] = (),
) -> list[str]:
    fit = {key for key, role in assignments.items() if role == "fit_pool"}
    target = min(max(0, len(fit) - 1), int(len(fit) * fraction + 0.5))
    if fraction and len(fit) >= 2:
        target = max(1, target)
    # Keep at least one fitting block, including when the prior pool was empty.
    return _select_fit_pool_blocks(blocks, fit.difference(prior_fit), target, seed, "validation", prior_validation)


def _select_fit_pool_blocks(
    blocks: Mapping[str, BlockSpec],
    eligible: Collection[str],
    target: int,
    seed: int,
    subset: str,
    prior: Collection[str] = (),
) -> list[str]:
    """Fill the requested subset proportionally across the eligible strata."""
    selected: dict[str, str] = {}
    _allocate_strata(
        sorted(eligible),
        blocks,
        {subset: max(0, target - len(prior))},
        selected,
        seed=seed,
        enforce_source_splits=False,
    )
    return sorted(set(prior) | selected.keys())


def assign_block_roles(
    blocks: Sequence[BlockSpec],
    role_targets: Mapping[str, int],
    *,
    seed: int = 20261123,
    extra_dev_test_blocks: Collection[str] = (),
    fit_pool_blocks: Collection[str] = (),
    pinned_roles: Mapping[str, str] | None = None,
    excluded_blocks: Collection[str] = (),
    giant_caps: Mapping[str, int] | None = None,
    pairwise_validation_fraction: float = DEFAULT_PAIRWISE_VALIDATION_FRACTION,
    gate_training_block_count: int = DEFAULT_GATE_TRAINING_BLOCK_COUNT,
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
        fit_pool_blocks: Leakage-test blocks to pin to ``fit_pool``.
        pinned_roles: Optional explicit ``{block_key: role}`` pins (e.g. prior frozen assignments).
        excluded_blocks: Extra blocks to exclude alongside ``PERMANENT_EXCLUDED_BLOCKS``.
        giant_caps: Maximum number of blocks above 15,000 signatures per specified role, including pins.
        pairwise_validation_fraction: Fraction of fit_pool blocks held out from pairwise fitting.
        gate_training_block_count: Target number of fit_pool blocks marked for gate training.
        enforce_source_splits: When True, final_test accepts only ``holdout`` blocks and
            training/tuning roles accept only ``development`` blocks.
        require_pinned_present: When True, raise if any explicitly pinned/dev_test/excluded block
            is absent from ``blocks``.
    """
    role_targets = dict(role_targets)
    invalid_targets = sorted(set(role_targets) - set(ASSIGNABLE_ROLES))
    if invalid_targets:
        raise ValueError(f"Unsupported target roles: {invalid_targets}")
    for role, count in role_targets.items():
        if count < 0:
            raise ValueError(f"Target count for {role!r} must be >= 0, got {count}")
    if not 0 <= pairwise_validation_fraction < 1:
        raise ValueError("Pairwise validation fraction must be in [0, 1)")
    if gate_training_block_count < 0:
        raise ValueError("Gate training block count must be nonnegative")
    caps = dict(giant_caps or {})
    for role, cap in caps.items():
        if role not in VALID_ROLES or type(cap) is not int or cap < 0:
            raise ValueError(f"Invalid giant cap for role {role!r}: {cap!r}")
    _validate_source_splits(blocks, enforce_source_splits)

    excluded_set = {normalize_block_key(k) for k in excluded_blocks} | set(PERMANENT_EXCLUDED_BLOCKS)
    dev_test_set = {normalize_block_key(k) for k in extra_dev_test_blocks} | set(LEGACY_DEV_TEST_BLOCKS)
    fit_pool_set = {normalize_block_key(k) for k in fit_pool_blocks}
    if overlap := fit_pool_set & dev_test_set:
        raise ValueError(f"Blocks cannot be pinned to both fit_pool and dev_test: {sorted(overlap)}")

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
        if key in pins:
            raise ValueError(f"Duplicate normalized pin for {key!r}")
        pins[key] = role
    for key in fit_pool_set:
        if key in pins and pins[key] != "fit_pool":
            raise ValueError(f"Block {key!r} is pinned to {pins[key]!r} and fit_pool")
        pins[key] = "fit_pool"
    for key in dev_test_set:
        if key in pins and pins[key] != "dev_test":
            raise ValueError(f"Block {key!r} is a legacy/reviewed dev_test block and cannot be pinned to {pins[key]!r}")
        pins[key] = "dev_test"

    if require_pinned_present:
        missing_pins = sorted(
            (
                {normalize_block_key(k) for k in (pinned_roles or {})}
                | {normalize_block_key(k) for k in extra_dev_test_blocks}
                | fit_pool_set
            )
            - seen_keys
        )
        if missing_pins:
            raise ValueError(f"Pinned/excluded blocks missing from input: {missing_pins}")

    if conflicting := excluded_set & ({normalize_block_key(key) for key in (pinned_roles or {})} | fit_pool_set):
        raise ValueError(f"Cannot exclude pinned blocks: {sorted(conflicting)}")

    active_blocks = [block for block in norm_blocks if block.block_key not in excluded_set]
    recorded_exclusions = sorted((seen_keys & excluded_set) | {normalize_block_key(k) for k in excluded_blocks})
    by_key = {block.block_key: block for block in active_blocks}

    giant_budget = (
        _GiantCaps(caps.copy(), {key for key, block in by_key.items() if block.signature_count > GIANT_SIGNATURE_COUNT})
        if caps
        else None
    )
    block_to_role = {key: pins[key] for key in sorted(by_key.keys() & pins.keys())}
    if giant_budget is not None:
        for key, role in block_to_role.items():
            if not giant_budget.allows(role, key):
                raise ValueError(f"Pinned blocks exceed the giant cap for {role!r}: {key!r}")
            giant_budget.take(role, key)
    free_blocks = sorted(by_key.keys() - block_to_role.keys())

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
                (role != FINAL_TEST_ROLE or block.block_key not in block_to_role)
                and (
                    not enforce_source_splits
                    or block.source_split == ("holdout" if role == FINAL_TEST_ROLE else "development")
                )
            )
        }
        for role in target_order
    }
    # Reserve coverage before proportional sampling can discard a rare stratum.
    free_blocks = _reserve_role_strata(
        free_blocks,
        by_key,
        remaining,
        block_to_role,
        seed=seed,
        enforce_source_splits=enforce_source_splits,
        giant_caps=giant_budget,
    )
    if giant_budget is not None:
        _allocate_strata(
            free_blocks,
            by_key,
            remaining,
            block_to_role,
            seed=seed,
            enforce_source_splits=enforce_source_splits,
            giant_caps=giant_budget,
        )
    else:
        cohorts = (
            {FINAL_TEST_ROLE: (FINAL_TEST_ROLE,), "fit_pool": TRAINING_AND_TUNING_ROLES, "dev_test": ("dev_test",)}
            if enforce_source_splits
            else {FINAL_TEST_ROLE: ASSIGNABLE_ROLES}
        )
        targets_by_cohort = {
            cohort: {role: count for role, count in remaining.items() if role in members}
            for cohort, members in cohorts.items()
        }
        # Reserve the total sample before role rounding so rare strata reach small roles.
        selected: dict[str, str] = {}
        _allocate_strata(
            free_blocks,
            by_key,
            {cohort: sum(targets.values()) for cohort, targets in targets_by_cohort.items()},
            selected,
            seed=seed,
            enforce_source_splits=enforce_source_splits,
        )
        selected_blocks: dict[str, list[str]] = defaultdict(list)
        for key, cohort in selected.items():
            selected_blocks[cohort].append(key)
        for cohort, targets in targets_by_cohort.items():
            _allocate_strata(
                selected_blocks[cohort],
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
        enforce_source_splits=enforce_source_splits,
    )
    roles: dict[str, list[str]] = {role: [] for role in VALID_ROLES}
    for key, role in sorted(block_to_role.items()):
        roles[role].append(key)
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
        "enforce_source_splits": bool(enforce_source_splits),
        "excluded_blocks": recorded_exclusions,
        "role_targets": {role: int(role_targets[role]) for role in target_order},
        "giant_caps": caps,
        "roles": roles,
        "block_to_role": dict(sorted(block_to_role.items())),
        "pairwise_validation_fraction": pairwise_validation_fraction,
        "pairwise_validation_blocks": _pairwise_validation_fold(
            by_key, block_to_role, pairwise_validation_fraction, seed
        ),
        "gate_training_block_count": gate_training_block_count,
        "gate_training_blocks": _select_fit_pool_blocks(
            by_key, roles["fit_pool"], gate_training_block_count, seed, "gate_training"
        ),
        "block_orcids": {key: sorted(by_key[key].orcids) for key in sorted(block_to_role)},
        "block_signature_counts": {key: by_key[key].signature_count for key in sorted(block_to_role)},
        "unassigned": sorted(by_key.keys() - block_to_role.keys()),
        "shortfalls": shortfalls,
        "stratum_shortfalls": stratum_shortfalls,
        "verification": verification,
    }


def load_blocks(path: Path) -> list[BlockSpec]:
    """Load frozen block metadata from Parquet."""
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
    specs: list[BlockSpec] = []
    for item in rows:
        key = item.get("block_key") or item.get("cluster_block_key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Missing block_key in {path}")
        orcids = item.get("orcids")
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
        size = item.get("size_band") or size_band(count)
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
    fit_pool_blocks: Collection[str] = (),
    giant_caps: Mapping[str, int] | None = None,
    pairwise_validation_fraction: float | None = None,
    gate_training_block_count: int | None = None,
    seed: int = 20261123,
) -> dict[str, Any]:
    """Freeze only exported blocks and discard every cross-role identity from scoring."""
    import pyarrow.parquet as pq

    evidence = load_identity_evidence(evidence_path)
    if evidence.snapshot_sha256 != sha256_file(evidence.snapshot_root / "manifest.json"):
        raise ValueError("Snapshot has changed; append its evidence before freezing roles")
    receipts = evidence.assignment_receipts()
    if fresh and prior_path is None:
        raise ValueError("--fresh requires the master assignment via --pinned-roles-json")
    if fresh and (
        targets
        or extra_pins
        or fit_pool_blocks
        or giant_caps is not None
        or pairwise_validation_fraction is not None
        or gate_training_block_count is not None
    ):
        raise ValueError("--fresh reuses the master assignment and cannot take allocation options")
    if receipts:
        if prior_path is None:
            raise ValueError(
                "Snapshot or evidence already has a freeze binding; pass the prior manifest even with --fresh"
            )
        if receipts != {assignment_digest(prior_path)}:
            raise ValueError("The prior role manifest does not match the latest freeze binding")
    metadata = pq.read_schema(blocks_path).metadata or {}
    for digest_field, digest in {
        EVIDENCE_SHA256_FIELD: evidence.sha256,
        "snapshot_sha256": evidence.snapshot_sha256,
        "intake_sha256": evidence.intake_sha256,
    }.items():
        if metadata.get(digest_field.encode()) != digest.encode():
            raise ValueError(f"Freeze input {digest_field} does not match the identity evidence")
    blocks = load_blocks(blocks_path)
    if {block.block_key for block in blocks} != evidence.blocks.keys():
        raise ValueError("Freeze input must cover exactly the exported blocks")
    evidence_blocks = evidence.block_digests()
    prior = read_json_object(prior_path) if prior_path else None
    pins: dict[str, str] = {}
    prior_exclusions: list[str] = []
    prior_unassigned: set[str] = set()
    prior_fit: Collection[str] = ()
    prior_validation: Collection[str] = ()
    prior_gate_training: frozenset[str] = frozenset()
    if prior_path is not None and prior is not None:
        prior_roles = load_block_roles(prior_path)
        if prior.get("format_version") != 2 or prior.get("intake_sha256") != evidence.intake_sha256:
            raise ValueError("Re-freeze requires a prior ledger-backed manifest from the same intake")
        if any(evidence_blocks.get(block) != digest for block, digest in prior["evidence_blocks"].items()):
            raise ValueError("Prior evidence partitions changed or are missing")
        pins.update(prior_roles.block_to_role)
        prior_fit = prior["roles"]["fit_pool"]
        prior_validation = prior_roles.pairwise_validation_blocks
        prior_gate_training = prior_roles.gate_training_blocks
        prior_exclusions = prior["excluded_blocks"]
        prior_unassigned = set(prior["unassigned"])
        prior_blocks = prior["evidence_blocks"].keys()
        if fresh and (
            prior_blocks != evidence_blocks.keys()
            or prior_roles.source_splits != {block.block_key: block.source_split for block in blocks}
            or prior_roles.evidence_sha256 != evidence.sha256
            or prior_roles.snapshot_sha256 != evidence.snapshot_sha256
        ):
            raise ValueError("--fresh must use the master assignment's evidence, blocks and source splits")
        requested_changes = {normalize_block_key(block) for block in fit_pool_blocks} & (
            prior_blocks - prior_unassigned
        )
        if requested_changes:
            raise ValueError(f"Re-freeze cannot move prior blocks: {sorted(requested_changes)}")
    extra = {normalize_block_key(block): role for block, role in (extra_pins or {}).items()}
    if len(extra) != len(extra_pins or {}):
        raise ValueError("Duplicate normalized keys in --extra-pins-json")
    if prior is not None and extra.keys() & (prior["evidence_blocks"].keys() - prior_unassigned):
        raise ValueError("Extra pins may only name new or previously unassigned exported blocks")
    pins.update(extra)
    fraction = (
        pairwise_validation_fraction
        if pairwise_validation_fraction is not None
        else prior["pairwise_validation_fraction"]
        if prior is not None
        else DEFAULT_PAIRWISE_VALIDATION_FRACTION
    )
    gate_count = (
        gate_training_block_count
        if gate_training_block_count is not None
        else prior["gate_training_block_count"]
        if prior is not None
        else DEFAULT_GATE_TRAINING_BLOCK_COUNT
    )
    block_orcids = {block.block_key: sorted(block.orcids) for block in blocks}
    excluded_reasons: dict[str, str] = dict(prior.get("excluded_block_reasons", {})) if prior is not None else {}
    if fresh:
        assert prior is not None
        roles = prior
    else:
        while True:
            roles = assign_block_roles(
                blocks,
                targets,
                seed=seed,
                fit_pool_blocks=fit_pool_blocks,
                pinned_roles=pins,
                excluded_blocks=(*prior_exclusions, *excluded_reasons),
                giant_caps=giant_caps if giant_caps is not None else prior["giant_caps"] if prior is not None else None,
                pairwise_validation_fraction=fraction,
                gate_training_block_count=gate_count,
                enforce_source_splits=True,
                require_pinned_present=True,
            )
            mapping = roles["block_to_role"]
            ledger = IdentityLedger(
                evidence,
                BlockRoles(
                    mapping,
                    "",
                    {block: frozenset(orcids) for block, orcids in block_orcids.items()},
                    identity_roles=assigned_identity_roles(block_orcids, mapping),
                    source_splits={block.block_key: cast(str, block.source_split) for block in blocks},
                ),
            )
            sparse = {
                block
                for block in roles["roles"]["eps_cal"]
                if max(Counter(ledger.for_block(block).scored().values()).values(), default=0) < 2
            }
            if not sparse:
                break
            # Keep surviving assignments while filling any resulting EPS shortfall.
            excluded_reasons.update(dict.fromkeys(sorted(sparse), "no_repeated_scored_orcid"))
            pins = {block: role for block, role in mapping.items() if block not in sparse}
        if prior_fit:
            by_key = {block.block_key: block for block in blocks}
            roles["pairwise_validation_blocks"] = _pairwise_validation_fold(
                by_key,
                roles["block_to_role"],
                fraction,
                seed,
                prior_fit=prior_fit,
                prior_validation=prior_validation,
            )
            if gate_training_block_count is None:
                roles["gate_training_blocks"] = _select_fit_pool_blocks(
                    by_key,
                    set(roles["roles"]["fit_pool"]) - prior_gate_training,
                    gate_count,
                    seed,
                    "gate_training",
                    prior_gate_training,
                )
    mapping = roles["block_to_role"]
    identity_roles = assigned_identity_roles(block_orcids, mapping)
    cross_role = {orcid for orcid, assigned in identity_roles.items() if len(assigned) > 1}
    identities_by_role: dict[str, set[str]] = defaultdict(set)
    mentions_by_role: Counter[str] = Counter()
    gate_block_orcids = {}
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
        if role == "gate_thresh":
            gate_block_orcids[key] = sorted(
                {orc for row in evidence_rows.values() for orc in gate_veto_identities(row)}
            )
        identities_by_role[role].update(cross_role & counts.keys())
        if cross_role & counts.keys():
            mentions_by_role[role] += len(
                {
                    (row["corpus_paper_id"], row["position"])
                    for row in evidence_rows.values()
                    if not row["reviewed_error"] and not evidence_identities(row).isdisjoint(cross_role)
                }
            )
    if fresh:
        return roles
    roles["verification"]["raw_cross_role_orcid_overlap"] = len(cross_role)
    roles["verification"]["cross_role_identities_by_role"] = {
        role: len(identities_by_role[role]) for role in VALID_ROLES
    }
    roles["verification"]["cross_role_mentions_by_role"] = {role: mentions_by_role[role] for role in VALID_ROLES}
    return {
        **roles,
        **({"excluded_block_reasons": excluded_reasons} if excluded_reasons else {}),
        "format_version": 2,
        "excluded_orcid_groups_sha256": excluded_orcid_groups_digest(),
        EVIDENCE_SHA256_FIELD: evidence.sha256,
        "snapshot_sha256": evidence.snapshot_sha256,
        "intake_sha256": evidence.intake_sha256,
        "evidence_blocks": evidence_blocks,
        "block_orcids": block_orcids,
        "gate_block_orcids": gate_block_orcids,
        "block_source_splits": {block.block_key: block.source_split for block in blocks},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=Path, required=True, help="Parquet produced by build_freeze_input.")
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, help="JSON dict of {role: target_block_count}.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument(
        "--pinned-roles-json",
        type=Path,
        default=None,
        help="Prior assignments.json to preserve on a re-freeze.",
    )
    parser.add_argument("--extra-pins-json", type=Path, help="Pin new or previously unassigned blocks to roles.")
    parser.add_argument("--fit-pool-blocks", nargs="*", default=(), help="Pin leakage-test blocks to fit_pool.")
    parser.add_argument("--giant-caps-json", type=Path, help="JSON dict of {role: maximum_giant_block_count}.")
    parser.add_argument(
        "--pairwise-validation-fraction", type=float, help="Holdout fraction within fit_pool (default 0.15)."
    )
    parser.add_argument(
        "--gate-training-block-count", type=int, help="Target gate training blocks within fit_pool (default 20)."
    )
    parser.add_argument(
        "--fresh", action="store_true", help="Reproduce the master assignment without allocating blocks."
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing assignments.json.")
    args = parser.parse_args(argv)
    if args.targets_json is None and not args.fresh:
        parser.error("--targets-json is required unless --fresh reuses the master assignment")

    evidence = load_identity_evidence(args.identity_evidence)
    with exclusive_file_lock(evidence.snapshot_root / INTAKE_LOCK):
        out_json = args.output_dir / "assignments.json"
        if out_json.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing {out_json}; pass --overwrite or use a fresh directory"
            )

        if out_json.exists() and (
            args.pinned_roles_json is None or assignment_digest(args.pinned_roles_json) != assignment_digest(out_json)
        ):
            raise ValueError("Pass the existing assignments.json as the prior manifest when re-freezing")
        result = freeze_identity_roles(
            args.blocks,
            args.identity_evidence,
            read_json_object(args.targets_json) if args.targets_json else {},
            prior_path=args.pinned_roles_json,
            fresh=args.fresh,
            extra_pins=read_json_object(args.extra_pins_json) if args.extra_pins_json else None,
            fit_pool_blocks=args.fit_pool_blocks,
            giant_caps=read_json_object(args.giant_caps_json) if args.giant_caps_json else None,
            pairwise_validation_fraction=args.pairwise_validation_fraction,
            gate_training_block_count=args.gate_training_block_count,
            seed=args.seed,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            # Preserve the master digest, including the JSON file's original formatting.
            with tempfile.TemporaryDirectory(dir=args.output_dir) as staging:
                copy = Path(staging) / out_json.name
                copy.write_bytes(args.pinned_roles_json.read_bytes())
                copy.replace(out_json)
        else:
            _replace_json(out_json, result)
        with (args.output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["block_key", "role"])
            for key, role in result["block_to_role"].items():
                writer.writerow([key, role])
        receipt = {"assignments_sha256": assignment_digest(out_json)}
        intake_path = evidence.snapshot_root / SNAPSHOT_BINDING
        intake = read_json_object(intake_path)
        _replace_json(intake_path, {**intake, FREEZE_SHA256_FIELD: receipt["assignments_sha256"]})
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
