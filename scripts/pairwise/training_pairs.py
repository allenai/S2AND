"""Read the frozen mining handoff and partition fitted and judge-only pairs."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from s2and._sha256 import sha256_file
from scripts.common.arrow_roots import generic_dataset_manifests, read_json_object
from scripts.common.block_roles import BlockRoles, normalize_block_key
from scripts.common.identity_ledger import IdentityLedger
from scripts.common.stage0_invariants import (
    JUDGE_ONLY_STRATUM,
    PAIR_STRATA,
    FitStratum,
    Fold,
    PairId,
    require_assignment_receipt,
    require_fit_pool_coverage,
    require_intake_index,
    require_pair_conservation,
    require_pair_purpose,
    require_pairwise_fold,
    require_unique_pair,
)


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
class JudgePair:
    """Carry a judge-only pair without a label or training fold."""

    block: str
    left: str
    right: str


@dataclass(frozen=True, slots=True)
class TrainingBlock:
    """Bind a block's partitioned pairs to the verified Arrow dataset."""

    arrow_dir: Path
    arrow_manifest_sha256: str
    fold: Fold
    fit: list[FitPair]
    judge: list[JudgePair]


class TrainingPairsBinding(TypedDict):
    block_role_assignments_sha256: str
    identity_evidence_sha256: str
    csv_sha256: str
    manifest_sha256: str
    pairwise_validation_blocks: list[str]


@dataclass(frozen=True, slots=True)
class TrainValPairs:
    """Hold every frozen fit_pool block and the training provenance."""

    blocks: dict[str, TrainingBlock]
    binding: TrainingPairsBinding


def _read_pair_ids(mined_csv: Path, blocks: Collection[str]) -> dict[str, dict[PairId, str]]:
    pairs_by_block: dict[str, dict[PairId, str]] = {block: {} for block in blocks}
    columns = {"block_key", "signature_id_1", "signature_id_2", "stratum"}
    with mined_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not columns.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing block, stratum or signature IDs in mined pairs: {mined_csv}")
        for row in reader:
            block = normalize_block_key(row["block_key"])
            stratum = row["stratum"]
            if block not in pairs_by_block:
                raise ValueError(f"Mined pair outside fit_pool: {block!r} stratum {stratum!r}")
            if stratum not in PAIR_STRATA:
                raise ValueError(f"Invalid mined stratum {stratum!r} in {block!r}")
            pair = (row["signature_id_1"], row["signature_id_2"])
            require_unique_pair(block, stratum, pair, pairs_by_block[block].keys())
            pairs_by_block[block][pair] = stratum
    return pairs_by_block


def training_pairs(
    roles: BlockRoles, ledger: IdentityLedger, mined_csv: Path, *, name_counts_sha256: str
) -> TrainValPairs:
    """Verify the complete mining handoff before returning any pairs for training."""
    roles.require_assignment(ledger.bindings, context="Identity ledger")
    manifest_path = mined_csv.with_suffix(mined_csv.suffix + ".manifest.json")
    manifest = read_json_object(manifest_path)
    roles.require_assignment(manifest, context="Mined pairs")
    raw_blocks = manifest.get("blocks")
    if manifest.get("role") != "fit_pool" or not isinstance(raw_blocks, dict):
        raise ValueError("Mined pair manifest must describe fit_pool blocks")
    blocks = cast(dict[str, dict[str, object]], raw_blocks)
    require_fit_pool_coverage(roles, blocks.keys())
    roles.require_bindings(manifest, blocks.keys(), context="Mined pairs")
    intake_index = ledger.evidence.name_counts_index_sha256
    require_intake_index(intake_index, name_counts_sha256, context="Pairwise trainer")
    for block, spec in sorted(blocks.items()):
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid mined block record for {block!r}")
        require_assignment_receipt(roles, block, spec.get("assignment_binding"), intake_index_sha256=intake_index)
    require_intake_index(intake_index, manifest.get("index_sha256"), context="Mined pairs")

    mined_digest = sha256_file(mined_csv)
    if manifest.get("output_sha256") != mined_digest:
        raise ValueError(f"Mined pair CSV digest mismatch: {mined_csv}")
    pairs_by_block = _read_pair_ids(mined_csv, blocks)
    snapshot = ledger.evidence.snapshot_root
    if sha256_file(snapshot / "manifest.json") != ledger.evidence.snapshot_sha256:
        raise ValueError("Mined pair snapshot does not match the identity ledger")
    dataset_manifests = generic_dataset_manifests(snapshot)

    result: dict[str, TrainingBlock] = {}
    for block, spec in sorted(blocks.items()):
        fold = require_pairwise_fold(roles, block, spec.get("fold"))
        evidence = ledger.evidence.blocks[block]
        arrow_dir = dataset_manifests[evidence["dataset"]].parent
        view = ledger.for_block(block, dataset_dir=arrow_dir)
        fit: list[FitPair] = []
        judge: list[JudgePair] = []
        block_pairs = pairs_by_block.pop(block)
        decisions = {
            sid: view.eligibility(sid) for sid in {sid for pair in block_pairs for sid in pair} & view.rows.keys()
        }
        for (left, right), stratum in sorted(block_pairs.items()):
            if left not in decisions or right not in decisions:
                raise ValueError(f"Unknown endpoint in {block!r} stratum {stratum!r} pair {(left, right)!r}")
            label = require_pair_purpose(block, stratum, (left, right), decisions[left], decisions[right])
            if label is None:
                judge.append(JudgePair(block, left, right))
            else:
                fit.append(FitPair(block, left, right, cast(FitStratum, stratum), label, fold))

        observed: dict[tuple[Fold, str], list[PairId]] = defaultdict(list)
        for pair in fit:
            observed[pair.fold, pair.stratum].append((pair.left, pair.right))
        observed[fold, JUDGE_ONLY_STRATUM] = [(pair.left, pair.right) for pair in judge]
        require_pair_conservation(block, fold, spec.get("strata"), observed)
        result[block] = TrainingBlock(arrow_dir, evidence["dataset_manifest_sha256"], fold, fit, judge)

    return TrainValPairs(
        result,
        TrainingPairsBinding(
            block_role_assignments_sha256=roles.sha256,
            identity_evidence_sha256=ledger.evidence.sha256,
            csv_sha256=mined_digest,
            manifest_sha256=sha256_file(manifest_path),
            pairwise_validation_blocks=sorted(roles.pairwise_validation_blocks),
        ),
    )
