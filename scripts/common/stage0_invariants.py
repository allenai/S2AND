"""Check the invariants at the stage-0 miner-to-trainer handoff."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal, cast

from s2and._sha256 import is_lowercase_sha256
from s2and.incremental_linking.contracts import canonical_json_digest
from scripts.common.block_roles import BlockRoles
from scripts.common.identity_ledger import Eligibility

PairId = tuple[str, str]
Fold = Literal["train", "val"]
FitStratum = Literal["false_merge", "missed_link", "random"]
JUDGE_ONLY_STRATUM = "merged_unlabeled"
PAIR_STRATA = ("false_merge", "missed_link", "random", JUDGE_ONLY_STRATUM)
SAMPLING_DROP_REASON = "sampling_budget"


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


def require_pair_purpose(block: str, stratum: str, pair: PairId, left: Eligibility, right: Eligibility) -> int | None:
    """Return the fitted pair's ledger label or confirm its judge-only purpose."""
    context = f"{block!r} stratum {stratum!r} pair {pair!r}"
    if not left.fit_ok or not right.fit_ok:
        raise ValueError(f"Ineligible signature in {context}")
    a, b = left.score_orcid, right.score_orcid
    if stratum == JUDGE_ONLY_STRATUM:
        if a is not None and b is not None:
            raise ValueError(f"Judge-only pair has two ledger scores in {context}")
        return None
    if a is None or b is None:
        raise ValueError(f"Fitted pair requires two ledger scores in {context}")
    return int(a == b)


def pair_ids_sha256(pairs: Collection[PairId]) -> str:
    """Bind the sampled pair IDs independently of their CSV row order."""
    return canonical_json_digest(sorted(pairs))


def require_pair_conservation(
    block: str,
    fold: Fold,
    strata: object,
    observed: Mapping[tuple[Fold, str], list[PairId]],
) -> None:
    """Reconcile sampled IDs and frame counts in each block, stratum and fold."""
    if not isinstance(strata, dict):
        raise ValueError(f"Missing mined strata for {block!r} fold {fold!r}")
    for stratum in sorted(set(strata) | set(PAIR_STRATA) | {key[1] for key in observed}):
        context = f"{block!r} stratum {stratum!r} fold {fold!r}"
        counts = strata.get(stratum)
        if stratum not in PAIR_STRATA or not isinstance(counts, dict):
            raise ValueError(f"Invalid mined stratum in {context}")
        frame, sampled, dropped = (counts.get(key) for key in ("frame_pairs", "sampled_pairs", "dropped_pairs"))
        if (
            type(frame) is not int
            or type(sampled) is not int
            or type(dropped) is not int
            or min(frame, sampled, dropped) < 0
            or sampled + dropped != frame
            or counts.get("drop_reason") != (SAMPLING_DROP_REASON if dropped else None)
        ):
            raise ValueError(f"Mined frame counts do not reconcile in {context}")
        ids = observed.get((fold, stratum), [])
        if any(key_fold != fold and key_stratum == stratum for key_fold, key_stratum in observed):
            raise ValueError(f"Mined pair assigned to the wrong fold in {context}")
        if len(ids) != sampled:
            raise ValueError(f"Mined pair count differs from the manifest in {context}: {len(ids)} != {sampled}")
        if pair_ids_sha256(ids) != counts.get("sampled_pair_ids_sha256"):
            raise ValueError(f"Mined pair IDs differ from the manifest in {context}")
