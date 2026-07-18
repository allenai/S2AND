"""Block loading, clustering-threshold selection, and B-cubed evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from s2and.eval import b3_precision_recall_fscore
from s2and.incremental_linking.feature_block_arrow import _read_arrow_ipc_table

B3_EVALUATION_PLAN_SCHEMA = "pair_ablation_b3_evaluation_plan_v1"
B3_MEMBER_IDENTITY_VERSION = "dataset_signature_tuple_v1"
B3EvaluationRole = Literal["calibration", "heldout_test", "heldout_full"]


@dataclass(frozen=True)
class GoldBlockData:
    """One cluster-labeled domain's signatures, S2 blocks, and gold authors."""

    dataset: str
    blocks: dict[str, list[str]]
    cluster_by_signature: dict[str, str]
    full_name_by_signature: dict[str, str]


@dataclass(frozen=True, slots=True)
class B3PlanBlock:
    """One evaluation block with its exact signature order."""

    block_key: str
    signatures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class B3EvaluationPlan:
    """Versioned, content-addressable block and gold-label evaluation plan."""

    dataset: str
    role: B3EvaluationRole
    evaluation_seed: int
    pair_budget: int | None
    blocks: tuple[B3PlanBlock, ...]
    gold_assignments: tuple[tuple[str, str], ...]

    def blocks_dict(self) -> dict[str, list[str]]:
        """Materialize the exact ordered block mapping expected by clustering."""

        return {block.block_key: list(block.signatures) for block in self.blocks}

    def gold_dict(self) -> dict[str, str]:
        """Materialize gold assignments for signatures selected by this plan."""

        return dict(self.gold_assignments)

    def identity_payload(self) -> dict[str, object]:
        """Return the semantic identity used by raw-feature caches and results."""

        gold: dict[str, str] = {}
        for signature_id, cluster_id in self.gold_assignments:
            if signature_id in gold:
                raise ValueError(f"B3 plan contains duplicate gold assignment for {signature_id!r}")
            gold[signature_id] = cluster_id
        seen_blocks: set[str] = set()
        seen_signatures: set[str] = set()
        for block in self.blocks:
            if block.block_key in seen_blocks:
                raise ValueError(f"B3 plan contains duplicate block key {block.block_key!r}")
            seen_blocks.add(block.block_key)
            overlap = seen_signatures.intersection(block.signatures)
            if overlap:
                raise ValueError(f"B3 plan contains signatures in multiple blocks: {sorted(overlap)[:5]}")
            seen_signatures.update(block.signatures)
        if set(gold) != seen_signatures:
            raise ValueError(
                "B3 plan gold/signature mismatch: "
                f"missing_gold={sorted(seen_signatures - set(gold))[:5]} "
                f"orphan_gold={sorted(set(gold) - seen_signatures)[:5]}"
            )
        return {
            "schema": B3_EVALUATION_PLAN_SCHEMA,
            "dataset": self.dataset,
            "role": self.role,
            "evaluation_seed": self.evaluation_seed,
            "pair_budget": self.pair_budget,
            "blocks": [
                {
                    "block_key": block.block_key,
                    "signatures": list(block.signatures),
                    "gold_cluster_ids": [gold[signature_id] for signature_id in block.signatures],
                }
                for block in self.blocks
            ],
        }

    @property
    def plan_digest(self) -> str:
        """Return a stable SHA-256 digest of the complete evaluation identity."""

        encoded = json.dumps(
            self.identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class B3DomainEvaluationPlans:
    """Reusable calibration and held-out plans for one public dataset."""

    calibration: B3EvaluationPlan
    heldout: B3EvaluationPlan


@dataclass(frozen=True)
class BlockLinkage:
    """One block's signature order and fitted hierarchical linkage matrix."""

    signatures: tuple[str, ...]
    tree: np.ndarray | None


def load_gold_block_data(dataset: str, signatures_path: Path, clusters_path: Path) -> GoldBlockData:
    """Load only columns needed for sampling and B-cubed evaluation."""

    import pyarrow as pa

    table = _read_arrow_ipc_table(pa, signatures_path)
    required = {
        "signature_id",
        "author_block",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
    }
    missing = sorted(required.difference(table.column_names))
    if missing:
        raise ValueError(f"{dataset} signatures Arrow is missing columns: {missing}")

    selected = table.select(sorted(required)).to_pylist()
    blocks: dict[str, list[str]] = {}
    full_names: dict[str, str] = {}
    for row in selected:
        signature_id = str(row["signature_id"])
        if signature_id in full_names:
            raise ValueError(f"{dataset} has duplicate signature_id={signature_id!r}")
        block = row["author_block"]
        if block is None or str(block) == "":
            raise ValueError(f"{dataset} signature {signature_id!r} has no S2 block")
        blocks.setdefault(str(block), []).append(signature_id)
        full_names[signature_id] = " ".join(
            str(row[column]).strip()
            for column in ("author_first", "author_middle", "author_last", "author_suffix")
            if row[column] is not None and str(row[column]).strip()
        )

    raw_clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not isinstance(raw_clusters, dict):
        raise TypeError(f"{dataset} clusters must contain an object")
    cluster_by_signature: dict[str, str] = {}
    for cluster_id, cluster in raw_clusters.items():
        for raw_signature_id in cluster["signature_ids"]:
            signature_id = str(raw_signature_id)
            previous = cluster_by_signature.setdefault(signature_id, str(cluster_id))
            if previous != str(cluster_id):
                raise ValueError(
                    f"{dataset} signature {signature_id!r} belongs to both {previous!r} and {cluster_id!r}"
                )
    runtime_ids = set(full_names)
    missing_gold = sorted(runtime_ids.difference(cluster_by_signature))
    orphan_gold = sorted(set(cluster_by_signature).difference(runtime_ids))
    if missing_gold or orphan_gold:
        raise ValueError(
            f"{dataset} runtime/gold signature mismatch: missing_gold={missing_gold[:5]} "
            f"orphan_gold={orphan_gold[:5]}"
        )
    return GoldBlockData(dataset, blocks, cluster_by_signature, full_names)


def split_blocks_like_anddata(
    blocks: Mapping[str, Sequence[str]],
    *,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Reproduce the default 80/10/10 block split used by ``ANDData``."""

    block_keys = list(blocks)
    if len(block_keys) < 10:
        raise ValueError("At least ten blocks are required for the default 80/10/10 split")
    block_sizes = np.asarray([len(blocks[key]) for key in block_keys], dtype=np.int64).reshape(-1, 1)
    strata = KMeans(n_clusters=1, random_state=random_seed, n_init=10).fit(block_sizes).labels_
    train_keys, val_test_keys, _, val_test_strata = train_test_split(
        block_keys,
        strata,
        test_size=0.2,
        stratify=strata,
        random_state=random_seed,
    )
    val_keys, test_keys = train_test_split(
        val_test_keys,
        test_size=0.5,
        stratify=val_test_strata,
        random_state=random_seed,
    )

    def materialize(keys: Sequence[str]) -> dict[str, list[str]]:
        return {key: list(blocks[key]) for key in keys}

    return materialize(train_keys), materialize(val_keys), materialize(test_keys)


def select_blocks_with_pair_budget(
    blocks: Mapping[str, Sequence[str]],
    *,
    pair_budget: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Select whole blocks deterministically up to a soft within-block pair budget."""

    if pair_budget <= 0:
        raise ValueError("pair_budget must be positive")
    candidates = [(key, list(signatures)) for key, signatures in blocks.items() if len(signatures) > 1]
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(candidates))
    selected: dict[str, list[str]] = {}
    used_pairs = 0
    for index in order:
        key, signatures = candidates[int(index)]
        block_pairs = len(signatures) * (len(signatures) - 1) // 2
        if used_pairs + block_pairs > pair_budget:
            continue
        selected[key] = signatures
        used_pairs += block_pairs
        if used_pairs >= pair_budget:
            break
    if not selected and candidates:
        key, signatures = min(candidates, key=lambda item: (len(item[1]), item[0]))
        selected[key] = signatures
    return selected


def _make_b3_evaluation_plan(
    gold: GoldBlockData,
    *,
    role: B3EvaluationRole,
    evaluation_seed: int,
    pair_budget: int | None,
    blocks: Mapping[str, Sequence[str]],
) -> B3EvaluationPlan:
    plan_blocks: list[B3PlanBlock] = []
    gold_assignments: list[tuple[str, str]] = []
    seen_signatures: set[str] = set()
    for raw_block_key, raw_signatures in blocks.items():
        block_key = str(raw_block_key)
        signatures = tuple(str(signature_id) for signature_id in raw_signatures)
        if len(signatures) != len(set(signatures)):
            raise ValueError(f"B3 plan block {block_key!r} contains duplicate signature IDs")
        overlap = seen_signatures.intersection(signatures)
        if overlap:
            raise ValueError(f"B3 plan contains signatures in multiple blocks: {sorted(overlap)[:5]}")
        seen_signatures.update(signatures)
        for signature_id in signatures:
            try:
                cluster_id = str(gold.cluster_by_signature[signature_id])
            except KeyError as exc:
                raise ValueError(f"Missing gold cluster for signature {signature_id!r}") from exc
            gold_assignments.append((signature_id, cluster_id))
        plan_blocks.append(B3PlanBlock(block_key, signatures))
    return B3EvaluationPlan(
        dataset=str(gold.dataset),
        role=role,
        evaluation_seed=int(evaluation_seed),
        pair_budget=None if pair_budget is None else int(pair_budget),
        blocks=tuple(plan_blocks),
        gold_assignments=tuple(gold_assignments),
    )


def build_b3_evaluation_plans(
    public_gold: Mapping[str, GoldBlockData],
    *,
    evaluation_seed: int,
    threshold_pairs_per_domain: int,
    b3_scope: str,
) -> dict[str, B3DomainEvaluationPlans]:
    """Freeze reusable calibration and held-out B3 plans for every public domain."""

    if threshold_pairs_per_domain <= 0:
        raise ValueError("threshold_pairs_per_domain must be positive")
    if b3_scope not in {"test", "full"}:
        raise ValueError(f"Unknown B3 scope: {b3_scope}")
    plans: dict[str, B3DomainEvaluationPlans] = {}
    for raw_dataset, gold in public_gold.items():
        dataset = str(raw_dataset)
        if dataset != gold.dataset:
            raise ValueError(f"Public-gold mapping key {dataset!r} does not match dataset {gold.dataset!r}")
        _, validation_blocks, test_blocks = split_blocks_like_anddata(
            gold.blocks,
            random_seed=evaluation_seed,
        )
        calibration_blocks = select_blocks_with_pair_budget(
            validation_blocks,
            pair_budget=threshold_pairs_per_domain,
            random_seed=evaluation_seed,
        )
        heldout_blocks = test_blocks if b3_scope == "test" else gold.blocks
        heldout_role: B3EvaluationRole = "heldout_test" if b3_scope == "test" else "heldout_full"
        plans[dataset] = B3DomainEvaluationPlans(
            calibration=_make_b3_evaluation_plan(
                gold,
                role="calibration",
                evaluation_seed=evaluation_seed,
                pair_budget=threshold_pairs_per_domain,
                blocks=calibration_blocks,
            ),
            heldout=_make_b3_evaluation_plan(
                gold,
                role=heldout_role,
                evaluation_seed=evaluation_seed,
                pair_budget=None,
                blocks=heldout_blocks,
            ),
        )
    return plans


def build_block_linkages(
    blocks: Mapping[str, Sequence[str]],
    distances: Mapping[str, np.ndarray],
) -> dict[str, BlockLinkage]:
    """Fit each average-linkage tree once so many thresholds can be scanned cheaply."""

    if set(blocks) != set(distances):
        raise ValueError("blocks and distances must have identical keys")
    output: dict[str, BlockLinkage] = {}
    for block_key, raw_signatures in blocks.items():
        signatures = tuple(str(value) for value in raw_signatures)
        expected_pairs = len(signatures) * (len(signatures) - 1) // 2
        values = np.asarray(distances[block_key], dtype=np.float64)
        if values.shape != (expected_pairs,):
            raise ValueError(f"distance shape mismatch for block={block_key!r}: {values.shape} != ({expected_pairs},)")
        tree = None if len(signatures) <= 1 else linkage(values, "average", preserve_input=True)
        output[str(block_key)] = BlockLinkage(signatures, tree)
    return output


def predicted_clusters_at_threshold(
    linkages: Mapping[str, BlockLinkage],
    threshold: float,
    *,
    dataset_prefix: str,
) -> dict[str, list[tuple[str, str]]]:
    """Cut trees and return clusters with dataset-namespaced member identities."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    output: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for block_key, block in linkages.items():
        labels = (
            np.ones(len(block.signatures), dtype=np.int64)
            if block.tree is None
            else fcluster(block.tree, t=float(threshold), criterion="distance")
        )
        for signature_id, label in zip(block.signatures, labels, strict=True):
            output[f"{dataset_prefix}:{block_key}:{int(label)}"].append((dataset_prefix, signature_id))
    return dict(output)


def true_clusters_for_blocks(
    blocks: Mapping[str, Sequence[str]],
    cluster_by_signature: Mapping[str, str],
    *,
    dataset_prefix: str,
) -> dict[str, list[tuple[str, str]]]:
    """Return gold clusters with dataset-namespaced members for ``blocks``."""

    output: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for signatures in blocks.values():
        for raw_signature_id in signatures:
            signature_id = str(raw_signature_id)
            try:
                cluster_id = cluster_by_signature[signature_id]
            except KeyError as exc:
                raise ValueError(f"Missing gold cluster for signature {signature_id!r}") from exc
            output[f"{dataset_prefix}:{cluster_id}"].append((dataset_prefix, signature_id))
    return dict(output)


def b3_for_threshold(
    linkages_by_dataset: Mapping[str, Mapping[str, BlockLinkage]],
    blocks_by_dataset: Mapping[str, Mapping[str, Sequence[str]]],
    cluster_lookup_by_dataset: Mapping[str, Mapping[str, str]],
    threshold: float,
) -> tuple[float, float, float]:
    """Compute signature-weighted B-cubed over one or more disjoint domains."""

    if set(linkages_by_dataset) != set(blocks_by_dataset) or set(blocks_by_dataset) != set(cluster_lookup_by_dataset):
        raise ValueError("B-cubed dataset mappings must have identical keys")
    predicted: dict[str, list[tuple[str, str]]] = {}
    truth: dict[str, list[tuple[str, str]]] = {}
    for dataset in blocks_by_dataset:
        predicted.update(
            predicted_clusters_at_threshold(linkages_by_dataset[dataset], threshold, dataset_prefix=dataset)
        )
        truth.update(
            true_clusters_for_blocks(
                blocks_by_dataset[dataset],
                cluster_lookup_by_dataset[dataset],
                dataset_prefix=dataset,
            )
        )
    precision, recall, f1, *_ = b3_precision_recall_fscore(truth, predicted)
    return float(precision), float(recall), float(f1)


def tune_b3_threshold(
    linkages_by_dataset: Mapping[str, Mapping[str, BlockLinkage]],
    blocks_by_dataset: Mapping[str, Mapping[str, Sequence[str]]],
    cluster_lookup_by_dataset: Mapping[str, Mapping[str, str]],
    thresholds: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Return the threshold with best B-cubed F1, breaking ties toward the smaller value."""

    if not thresholds:
        raise ValueError("thresholds must not be empty")
    scored = []
    for threshold in sorted({float(value) for value in thresholds}):
        precision, recall, f1 = b3_for_threshold(
            linkages_by_dataset,
            blocks_by_dataset,
            cluster_lookup_by_dataset,
            threshold,
        )
        scored.append((f1, -threshold, precision, recall))
    best_f1, negative_threshold, precision, recall = max(scored)
    threshold = -negative_threshold
    return threshold, {"precision": precision, "recall": recall, "f1": best_f1}
