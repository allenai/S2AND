"""Read the frozen mining handoff and partition fitted and judge-only pairs."""

from __future__ import annotations

import csv
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
    PairStratum,
    require_assignment_receipt,
    require_fit_pool_coverage,
    require_intake_index,
    require_pairwise_fold,
    require_unique_pair,
)
from scripts.pairwise.mine_pairwise_pairs import (
    PAIR_COLUMNS,
    SamplingPolicy,
    ledger_population,
    read_assignment_partition,
    select_block_pairs,
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


def _read_pairs(mined_csv: Path, blocks: Collection[str]) -> dict[str, dict[PairId, tuple[PairStratum, float]]]:
    pairs_by_block: dict[str, dict[PairId, tuple[PairStratum, float]]] = {block: {} for block in blocks}
    with mined_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if set(fieldnames) != set(PAIR_COLUMNS) or len(fieldnames) != len(PAIR_COLUMNS):
            raise ValueError(f"Invalid mined pair CSV columns: {mined_csv}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Invalid mined pair CSV row {reader.line_num}: {mined_csv}")
            block = normalize_block_key(row["block_key"])
            stratum = row["stratum"]
            if block not in pairs_by_block:
                raise ValueError(f"Mined pair outside fit_pool: {block!r} stratum {stratum!r}")
            if stratum not in PAIR_STRATA:
                raise ValueError(f"Invalid mined stratum {stratum!r} in {block!r}")
            pair = (row["signature_id_1"], row["signature_id_2"])
            require_unique_pair(block, stratum, pair, pairs_by_block[block].keys())
            pairs_by_block[block][pair] = (cast(PairStratum, stratum), float(row["inclusion_probability"]))
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
    policy = SamplingPolicy.from_manifest(manifest)

    mined_digest = sha256_file(mined_csv)
    if manifest.get("output_sha256") != mined_digest:
        raise ValueError(f"Mined pair CSV digest mismatch: {mined_csv}")
    pairs_by_block = _read_pairs(mined_csv, blocks)
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
        identities, gold, excluded = ledger_population(view)
        raw_path = spec.get("assignments_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Missing frozen assignments path for {block!r}. Re-mine pairs.")
        path = manifest_path.parent / raw_path
        receipt = read_json_object(path.parent / "manifest.json")
        if receipt.get("binding") != spec.get("assignment_binding"):
            raise ValueError(f"Mined assignment binding differs from the frozen receipt for {block!r}")
        clusters, assignments_digest = read_assignment_partition(path, block, identities, receipt)
        if spec.get("assignments_sha256") != assignments_digest:
            raise ValueError(f"Mined assignment digest differs from the frozen partition for {block!r}")
        sample = select_block_pairs(block, identities, gold, clusters, excluded, policy)
        observed = pairs_by_block.pop(block)
        if observed != sample.pairs:
            pair = next(
                pair
                for pair in sorted(observed.keys() | sample.pairs.keys())
                if observed.get(pair) != sample.pairs.get(pair)
            )
            raise ValueError(
                f"Mined pair selection differs from frozen inputs in {block!r} fold {fold!r} for {pair!r}: "
                f"observed {observed.get(pair)!r}, expected {sample.pairs.get(pair)!r}"
            )
        for field, expected in sample.counts.items():
            if spec.get(field) != expected:
                raise ValueError(f"Mined {field} report differs from frozen inputs in {block!r} fold {fold!r}")

        fit: list[FitPair] = []
        judge: list[JudgePair] = []
        for (left, right), (stratum, _) in sorted(sample.pairs.items()):
            if stratum == JUDGE_ONLY_STRATUM:
                judge.append(JudgePair(block, left, right))
            else:
                fit.append(FitPair(block, left, right, stratum, int(gold[left] == gold[right]), fold))
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
