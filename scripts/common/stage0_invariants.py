"""Check the invariants at the stage-0 miner-to-trainer handoff."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal, cast

from s2and._sha256 import is_lowercase_sha256
from scripts.common.block_roles import BlockRoles

PairId = tuple[str, str]
Fold = Literal["train", "val"]
FitStratum = Literal["false_merge", "missed_link", "random"]
PairStratum = Literal[FitStratum, "merged_unlabeled"]
JUDGE_ONLY_STRATUM = "merged_unlabeled"
PAIR_STRATA: tuple[PairStratum, ...] = ("false_merge", "missed_link", "random", JUDGE_ONLY_STRATUM)


def require_fit_pool_coverage(roles: BlockRoles, blocks: Collection[str]) -> None:
    """Require exactly the frozen fit pool, including blocks with no sampled pairs."""
    required = {block for block, role in roles.block_to_role.items() if role == "fit_pool"}
    observed = set(blocks)
    if observed != required:
        raise ValueError(
            f"Expected all fit_pool blocks: missing={sorted(required - observed)}, "
            f"outside fit_pool={sorted(observed - required)}"
        )


def require_intake_index(intake_sha256: str, index_sha256: object, *, context: str) -> None:
    """Require a name-counts index digest to match the intake index."""
    if not is_lowercase_sha256(index_sha256) or index_sha256 != intake_sha256:
        raise ValueError(f"{context} name-counts index does not match intake")


def require_assignment_receipt(roles: BlockRoles, block: str, binding: object, *, intake_index_sha256: str) -> None:
    """Require a block's assignment receipt to name its freeze and intake index."""
    context = f"{block!r} pairwise assignments"
    if not isinstance(binding, dict):
        raise ValueError(f"Missing assignment receipt for {block!r}")
    receipt = cast(Mapping[str, object], binding)
    roles.require_assignment(receipt, context=context)
    roles.require_bindings(receipt, {block}, context=context)
    require_intake_index(intake_index_sha256, receipt.get("index_sha256"), context=context)


def pairwise_fold(roles: BlockRoles, block: str) -> Fold:
    """Return the fold chosen by the master assignment."""
    return "val" if block in roles.pairwise_validation_blocks else "train"


def require_pairwise_fold(roles: BlockRoles, block: str, declared: object) -> Fold:
    """Check the manifest's fold and return the frozen fold."""
    fold = pairwise_fold(roles, block)
    if declared != fold:
        raise ValueError(f"Mined fold for {block!r} is {declared!r}, expected {fold!r}")
    return fold


def require_unique_pair(block: str, stratum: str, pair: PairId, seen: Collection[PairId]) -> None:
    """Require one canonical pair ID across all of a block's strata."""
    left, right = pair
    if not left or not right or left >= right:
        raise ValueError(f"Invalid mined pair {pair!r} in {block!r} stratum {stratum!r}")
    if pair in seen:
        raise ValueError(f"Duplicate mined pair {pair!r} in {block!r} stratum {stratum!r}")
