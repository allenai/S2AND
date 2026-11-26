"""Check invariants shared by stage-0 producers and consumers."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Literal, cast

from s2and._sha256 import is_lowercase_sha256
from scripts.common.block_roles import ROLE_SHA256_FIELD, BlockRoles

PairId = tuple[str, str]
Fold = Literal["train", "val"]
FitStratum = Literal["false_merge", "missed_link", "random"]
PairStratum = Literal[FitStratum, "merged_unlabeled"]
JUDGE_ONLY_STRATUM = "merged_unlabeled"
PAIR_STRATA: tuple[PairStratum, ...] = ("false_merge", "missed_link", "random", JUDGE_ONLY_STRATUM)


def require_fit_pool_coverage(roles: BlockRoles, blocks: Collection[str]) -> None:
    """Require exactly the frozen fit pool, including blocks with no sampled pairs."""
    required = {block for block, role in roles.block_to_role.items() if role == "fit_pool"}
    if not required:
        raise ValueError("Expected at least one frozen fit_pool block")
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


def require_assignment_receipts(roles: BlockRoles, bindings: Mapping[str, object], *, intake_index_sha256: str) -> str:
    """Require the frozen assignment receipts to agree on the model and index."""
    model = None
    for block, binding in sorted(bindings.items()):
        context = f"{block!r} pairwise assignments"
        if not isinstance(binding, dict):
            raise ValueError(f"Missing assignment receipt for {block!r}")
        receipt = cast(Mapping[str, object], binding)
        roles.require_assignment(receipt, context=context)
        roles.require_bindings(receipt, {block}, context=context)
        require_intake_index(intake_index_sha256, receipt.get("index_sha256"), context=context)
        digest = receipt.get("model_sha256")
        if not is_lowercase_sha256(digest) or (model is not None and digest != model):
            raise ValueError(f"{context} lack a shared model_sha256")
        model = digest
    assert model is not None  # Coverage requires a nonempty fit pool.
    return model


def require_linker_input_binding(
    roles: BlockRoles, binding: object, *, index_sha256: str | None = None
) -> dict[str, str]:
    """Check the model, index, and role binding shared by linker inputs."""
    if not isinstance(binding, dict):
        raise ValueError("Missing linker_input_binding. Rebuild frozen linker tasks.")
    roles.require_assignment(cast(Mapping[str, object], binding), context="Linker inputs")
    index = binding.get("index_sha256")
    if not is_lowercase_sha256(index) or (index_sha256 is not None and index != index_sha256):
        raise ValueError("Linker inputs use a different name-counts index")
    model = binding.get("model_sha256")
    if not is_lowercase_sha256(model):
        raise ValueError("Linker inputs lack a valid model_sha256")
    return {ROLE_SHA256_FIELD: roles.sha256, "index_sha256": index, "model_sha256": model}


def require_linker_input_receipt(
    roles: BlockRoles,
    block: str,
    receipt: Mapping[str, object],
    common: Mapping[str, str],
    files: Mapping[Path, str],
) -> None:
    """Match a block's input receipt to the frozen binding and files in use."""
    binding = receipt.get("binding")
    if not isinstance(binding, dict):
        raise ValueError(f"Missing linker input receipt binding for {block!r}")
    roles.require_bindings(cast(Mapping[str, object], binding), {block}, context="Linker inputs")
    if any(binding.get(key) != value for key, value in common.items()):
        raise ValueError(f"Input model, index, or role digest mismatch for {block!r}")
    digests = receipt.get("files")
    for path, digest in files.items():
        if not isinstance(digests, dict) or digests.get(path.name) != digest:
            raise ValueError(f"Linker input digest mismatch: {path}")


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
