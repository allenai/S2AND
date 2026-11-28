"""Sample fitted pairs from frozen seed partitions and the identity ledger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from s2and._sha256 import sha256_file
from s2and.incremental_linking.contracts import canonical_json_digest
from scripts.common.identity_ledger import IdentityLedger, require_evidence_for_roles
from scripts.common.stage0_invariants import JUDGE_ONLY_STRATUM, FitStratum, Fold, pairwise_fold, require_intake_index
from scripts.pairwise.mine_pairwise_pairs import (
    SamplingPolicy,
    fit_pool_receipts,
    read_fit_pool_inputs,
    sample_frozen_block,
)

DEFAULT_MIN_MINED_VALIDATION_PAIRS = 1


@dataclass(frozen=True, slots=True)
class FitPair:
    """Carry a ledger-labeled pair into its frozen fold."""

    block: str
    left: str
    right: str
    stratum: FitStratum
    label: int
    fold: Fold


@dataclass(frozen=True, slots=True)
class TrainingBlock:
    """Bind a block's fitted pairs to the verified Arrow dataset."""

    arrow_dir: Path
    arrow_manifest_sha256: str
    fold: Fold
    fit: list[FitPair]
    judge_count: int


class TrainingPairsBinding(TypedDict):
    block_role_assignments_sha256: str
    identity_evidence_sha256: str
    inputs_sha256: str
    model_sha256: str
    receipt_sha256_by_block: dict[str, str]
    sampling: dict[str, int | float]
    pairwise_validation_blocks: list[str]


@dataclass(frozen=True, slots=True)
class TrainValPairs:
    """Hold every frozen fit_pool block and the training provenance."""

    blocks: dict[str, TrainingBlock]
    binding: TrainingPairsBinding


def training_pairs(
    ledger: IdentityLedger,
    inputs_path: Path,
    *,
    name_counts_sha256: str,
    policy: SamplingPolicy,
    min_mined_validation_pairs: int = DEFAULT_MIN_MINED_VALIDATION_PAIRS,
) -> TrainValPairs:
    """Sample each fit_pool partition and assign fitted pairs to frozen folds."""
    if min_mined_validation_pairs < 1:
        raise ValueError("Minimum mined validation pairs must be positive")
    roles = ledger.roles
    require_evidence_for_roles(ledger.evidence, roles)
    require_intake_index(ledger.evidence.name_counts_index_sha256, name_counts_sha256, context="Pairwise trainer")
    blocks = read_fit_pool_inputs(ledger, inputs_path)
    receipts, model = fit_pool_receipts(ledger, blocks)

    result: dict[str, TrainingBlock] = {}
    for block, spec in sorted(blocks.items()):
        fold = pairwise_fold(roles, block)
        view = ledger.for_block(block, dataset_dir=spec.arrow_dir)
        frozen = sample_frozen_block(view, spec.assignments, receipts[block], policy)
        fit = [
            FitPair(block, left, right, stratum, int(frozen.gold[left] == frozen.gold[right]), fold)
            for (left, right), (stratum, _) in sorted(frozen.sample.pairs.items())
            if stratum != JUDGE_ONLY_STRATUM
        ]
        result[block] = TrainingBlock(
            spec.arrow_dir,
            ledger.evidence.blocks[block]["dataset_manifest_sha256"],
            fold,
            fit,
            len(frozen.sample.pairs) - len(fit),
        )

    validation_pairs = sum(len(spec.fit) for spec in result.values() if spec.fold == "val")
    if validation_pairs < min_mined_validation_pairs:
        raise ValueError(
            f"Expected at least {min_mined_validation_pairs} fitted mined validation pairs, found {validation_pairs}"
        )

    return TrainValPairs(
        result,
        TrainingPairsBinding(
            block_role_assignments_sha256=roles.sha256,
            identity_evidence_sha256=ledger.evidence.sha256,
            inputs_sha256=sha256_file(inputs_path),
            model_sha256=model,
            receipt_sha256_by_block={
                block: canonical_json_digest(receipt) for block, receipt in sorted(receipts.items())
            },
            sampling=asdict(policy),
            pairwise_validation_blocks=sorted(roles.pairwise_validation_blocks),
        ),
    )
