"""Shared helpers for single-letter retrieval candidate generation.

These helpers keep deterministic query selection, seed-summary construction,
and top-k retrieval ranking out of the larger reranker pipeline module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import scripts.eval_cluster_retrieval as retrieval
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore

try:
    import s2and_rust
except ImportError:  # pragma: no cover - Rust extension optional
    s2and_rust = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RustHybridCentroidRetrieverHandle:
    """Cached Python + Rust state for exact `hybrid_centroid` retrieval."""

    retriever: Any
    summary_by_component: dict[str, retrieval.ClusterSummary]


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


def build_rust_hybrid_centroid_retriever(
    candidate_summaries: list[retrieval.ClusterSummary],
) -> RustHybridCentroidRetrieverHandle:
    """Build the optional Rust-backed exact retriever for `hybrid_centroid`."""

    if s2and_rust is None or not hasattr(s2and_rust, "RustHybridCentroidRetriever"):
        raise RuntimeError("RustHybridCentroidRetriever is unavailable; build/install s2and_rust first")
    return RustHybridCentroidRetrieverHandle(
        retriever=s2and_rust.RustHybridCentroidRetriever(candidate_summaries),
        summary_by_component={str(summary.component_key): summary for summary in candidate_summaries},
    )


def rank_top_summaries_rust_hybrid_centroid(
    *,
    query: retrieval.QueryFeatures,
    max_ranked_clusters: int,
    retriever: RustHybridCentroidRetrieverHandle,
    component_keys: list[str] | None = None,
    max_block_component_size: int | None = None,
    override_summary: retrieval.ClusterSummary | None = None,
    num_threads: int | None = None,
) -> list[tuple[float, retrieval.ClusterSummary]]:
    """Score and rank candidate summaries with the optional Rust `hybrid_centroid` path."""

    if max_ranked_clusters <= 0:
        raise ValueError("max_ranked_clusters must be positive")
    if component_keys is None:
        ranked_component_keys, scores = retriever.retriever.top_k_hybrid_centroid(
            query,
            top_k=int(max_ranked_clusters),
            num_threads=None if num_threads is None else int(num_threads),
        )
    else:
        if max_block_component_size is None:
            raise ValueError("max_block_component_size is required when component_keys are provided")
        ranked_component_keys, scores = retriever.retriever.top_k_hybrid_centroid_subset(
            query,
            component_keys,
            top_k=int(max_ranked_clusters),
            max_block_component_size=int(max_block_component_size),
            num_threads=None if num_threads is None else int(num_threads),
            override_summary=override_summary,
        )
    summary_by_component = dict(retriever.summary_by_component)
    if override_summary is not None:
        summary_by_component[str(override_summary.component_key)] = override_summary
    return [
        (float(score), summary_by_component[str(component_key)])
        for component_key, score in zip(ranked_component_keys, scores, strict=True)
    ]
