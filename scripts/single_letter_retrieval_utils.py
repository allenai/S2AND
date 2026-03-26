"""Shared helpers for single-letter retrieval and chooser experiments.

These utilities sit below the experiment runners so giant-block and `h_wang`
evaluators can share the same candidate-summary and ranking behavior without
depending on one another's private helpers.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import scripts.eval_cluster_retrieval as retrieval
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore


def select_query_ids(
    query_ids: list[str],
    *,
    limit_queries: int | None,
    seed: int,
) -> list[str]:
    """Select a deterministic query subset for pilot runs."""

    if limit_queries is None or limit_queries >= len(query_ids):
        return list(query_ids)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(query_ids, dtype=object), size=int(limit_queries), replace=False)
    return sorted(str(query_id) for query_id in selected.tolist())


def invert_signature_to_cluster_id(signature_to_cluster_id: dict[str, str]) -> dict[str, list[str]]:
    """Invert a signature -> cluster map into cluster -> sorted signatures."""

    clusters: dict[str, list[str]] = {}
    for signature_id, cluster_id in signature_to_cluster_id.items():
        clusters.setdefault(str(cluster_id), []).append(str(signature_id))
    for cluster_id in clusters:
        clusters[cluster_id] = sorted(clusters[cluster_id])
    return clusters


def build_seed_summaries(
    *,
    dataset: Any,
    seed_clusters: dict[str, list[str]],
    block_key: str,
    max_exemplars: int,
) -> tuple[list[retrieval.ClusterSummary], dict[str, int], float]:
    """Build retrieval summaries for persisted seed clusters."""

    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    summaries: list[retrieval.ClusterSummary] = []
    cluster_sizes: dict[str, int] = {}
    start = time.perf_counter()
    for cluster_id, signature_ids in seed_clusters.items():
        cluster_sizes[str(cluster_id)] = len(signature_ids)
        summaries.append(
            retrieval.build_cluster_summary(
                dataset=dataset,
                block_key=block_key,
                cluster_id=str(cluster_id),
                component_key=str(cluster_id),
                signature_ids=[str(signature_id) for signature_id in signature_ids],
                max_exemplars=max_exemplars,
                feature_cache=feature_cache,
                orcid_enabled=False,
            )
        )
    build_ms = (time.perf_counter() - start) * 1000.0
    return summaries, cluster_sizes, build_ms


def rank_top_summaries(
    *,
    method: str,
    query: retrieval.QueryFeatures,
    candidate_summaries: list[retrieval.ClusterSummary],
    max_block_component_size: int,
    max_ranked_clusters: int,
) -> list[tuple[float, retrieval.ClusterSummary]]:
    """Score and rank candidate summaries for one retrieval method."""

    if not candidate_summaries:
        return []
    if max_ranked_clusters <= 0:
        raise ValueError("max_ranked_clusters must be positive")
    scores = np.fromiter(
        (
            retrieval.score_summary(method, query, summary, max_block_component_size=max_block_component_size)
            for summary in candidate_summaries
        ),
        dtype=np.float32,
        count=len(candidate_summaries),
    )
    top_n = min(int(max_ranked_clusters), len(candidate_summaries))
    top_indices = np.argpartition(-scores, top_n - 1)[:top_n]
    top_indices = sorted(
        top_indices.tolist(),
        key=lambda idx: (-float(scores[idx]), candidate_summaries[idx].component_key),
    )
    return [(float(scores[idx]), candidate_summaries[idx]) for idx in top_indices]


def hit_any_supported_at_k(ranked_cluster_ids: list[str], supported_cluster_ids: frozenset[str], k: int) -> int:
    """Return whether any supported cluster appears within the first `k` ranks."""

    return int(any(cluster_id in supported_cluster_ids for cluster_id in ranked_cluster_ids[:k]))


def hit_any_supported_within_signature_budget(
    ranked_cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    supported_cluster_ids: frozenset[str],
    signature_budget: int,
) -> int:
    """Return whether any supported cluster appears before the signature budget is exhausted."""

    if signature_budget <= 0:
        return 0
    materialized_signatures = 0
    for cluster_id in ranked_cluster_ids:
        next_total = materialized_signatures + int(cluster_sizes[cluster_id])
        if next_total > signature_budget:
            break
        materialized_signatures = next_total
        if cluster_id in supported_cluster_ids:
            return 1
    return 0


def materialized_signature_count_at_k(
    ranked_cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    k: int,
) -> int:
    """Return how many signatures are materialized in the first `k` clusters."""

    return sum(int(cluster_sizes[cluster_id]) for cluster_id in ranked_cluster_ids[:k])


def prefix_count_within_signature_budget(
    ranked_cluster_ids: list[str],
    cluster_sizes: dict[str, int],
    signature_budget: int,
) -> int:
    """Return the largest ranked prefix that fits within the signature budget."""

    if signature_budget <= 0:
        return 0
    materialized_signatures = 0
    cluster_count = 0
    for cluster_id in ranked_cluster_ids:
        next_total = materialized_signatures + int(cluster_sizes[cluster_id])
        if next_total > signature_budget:
            break
        materialized_signatures = next_total
        cluster_count += 1
    return int(cluster_count)
