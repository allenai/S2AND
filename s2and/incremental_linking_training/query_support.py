"""Training/replay query, retrieval, and row-signal helpers for the promoted linker."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

import s2and_rust
from s2and.data import ANDData
from s2and.incremental_linking.query_adapter import (
    ClusterSummary,
    QueryFeatures,
    RustHybridCentroidRetrieverHandle,
    build_cluster_summary,
    build_rust_hybrid_centroid_retriever,
)
from s2and.incremental_linking.row_features import GENERIC_FAMILY_MIN_COUNT, GENERIC_FAMILY_MIN_RATIO
from s2and.subblocking import make_subblocks_with_telemetry
from s2and.text import normalize_text

DEFAULT_CHOOSER_CACHE_MAX_TOP_K = 25
_MIDDLE_INITIAL_CONFLICT_SCORE = float(
    s2and_rust.RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE  # type: ignore[unresolved-attribute]
)
_YEAR_SCORE_DECAY_YEARS = float(
    s2and_rust.RETRIEVAL_YEAR_SCORE_DECAY_YEARS  # type: ignore[unresolved-attribute]
)
_YEAR_SCORE_RANGE_GAP = int(
    s2and_rust.RETRIEVAL_YEAR_SCORE_RANGE_GAP  # type: ignore[unresolved-attribute]
)
_YEAR_SCORE_RANGE_PENALTY = float(
    s2and_rust.RETRIEVAL_YEAR_SCORE_RANGE_PENALTY  # type: ignore[unresolved-attribute]
)


@dataclass(frozen=True)
class ClusterProfile:
    """Family metadata derived from a candidate cluster summary."""

    cluster_id: str
    family_id: str
    dominant_first_name: str | None
    family_dominance_ratio: float
    family_named_count: int


def _safe_compute_block(name: str) -> str:
    normalized_name = normalize_text(name or "")
    if not normalized_name:
        return ""
    from s2and.text import compute_block

    return compute_block(normalized_name)


def _subblock_tokens(subblock_key: str) -> list[str]:
    values: set[str] = set()
    for raw_token in str(subblock_key).split(","):
        token = str(raw_token).strip().split("|", 1)[0].strip()
        if len(token) > 1:
            values.add(token)
    return sorted(values)


def build_labeled_retrieval_subblock_index(
    *,
    dataset: ANDData,
    block_to_component_keys: dict[str, list[str]],
    component_signatures: dict[str, list[str]],
    maximum_size: int = 15_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen full-query candidate-gate index for labeled datasets."""

    signature_to_subblock: dict[str, str] = {}
    subblock_to_components: dict[str, set[str]] = defaultdict(set)
    subblock_tokens_by_subblock: dict[str, list[str]] = {}
    prefix_to_subblocks: dict[int, dict[str, set[str]]] = {
        2: defaultdict(set),
        3: defaultdict(set),
        4: defaultdict(set),
    }
    telemetry_rows: list[dict[str, Any]] = []

    for block_key, component_keys in block_to_component_keys.items():
        block_signature_ids = sorted(
            {
                str(signature_id)
                for component_key in component_keys
                for signature_id in component_signatures[str(component_key)]
            }
        )
        subblocks, telemetry = make_subblocks_with_telemetry(
            block_signature_ids,
            dataset,
            maximum_size=int(maximum_size),
            compute_block_fn=_safe_compute_block,
        )
        local_signature_to_subblock: dict[str, str] = {}
        for local_subblock_key, signature_ids in dict(subblocks).items():
            global_subblock_key = f"{block_key}::{local_subblock_key}"
            for signature_id in signature_ids:
                local_signature_to_subblock[str(signature_id)] = global_subblock_key
                signature_to_subblock[str(signature_id)] = global_subblock_key
            tokens = _subblock_tokens(str(local_subblock_key))
            subblock_tokens_by_subblock[global_subblock_key] = tokens
            for token in tokens:
                for prefix_len in (2, 3, 4):
                    prefix = token[: min(len(token), prefix_len)]
                    if len(prefix) >= 2:
                        prefix_to_subblocks[prefix_len][prefix].add(global_subblock_key)
        for component_key in component_keys:
            for signature_id in component_signatures[str(component_key)]:
                subblock_key = local_signature_to_subblock.get(str(signature_id))
                if subblock_key is not None:
                    subblock_to_components[subblock_key].add(str(component_key))
        telemetry_rows.append(
            {
                "block_key": str(block_key),
                "input_signature_count": int(telemetry["input_signature_count"]),
                "final_subblock_count": int(telemetry["final_subblock_count"]),
                "final_specter_labeled_subblock_count": int(telemetry["final_specter_labeled_subblock_count"]),
                "specter_invocation_count": int(telemetry["specter_invocation_count"]),
            }
        )

    diagnostics = {
        "blocks": int(len(block_to_component_keys)),
        "subblocks": int(len(subblock_to_components)),
        "mean_final_subblock_count_per_block": round(
            float(statistics.mean(int(row["final_subblock_count"]) for row in telemetry_rows)),
            6,
        )
        if telemetry_rows
        else 0.0,
        "blocks_with_specter_subblocks": int(
            sum(1 for row in telemetry_rows if int(row["final_specter_labeled_subblock_count"]) > 0)
        ),
        "blocks_with_specter_invocations": int(
            sum(1 for row in telemetry_rows if int(row["specter_invocation_count"]) > 0)
        ),
    }
    index = {
        "signature_to_subblock": signature_to_subblock,
        "subblock_to_components": {key: sorted(value) for key, value in subblock_to_components.items()},
        "subblock_tokens_by_subblock": subblock_tokens_by_subblock,
        "prefix_to_subblocks": {
            prefix_len: {key: sorted(value) for key, value in mapping.items()}
            for prefix_len, mapping in prefix_to_subblocks.items()
        },
    }
    return index, diagnostics


def build_cluster_profile(summary: ClusterSummary) -> ClusterProfile:
    """Build generic family metadata from a retrieval summary."""

    family_named_count = int(sum(summary.first_name_counts.values()))
    dominant_first_name = None
    family_dominance_ratio = 0.0
    family_id = str(summary.component_key)
    if summary.first_name_counts and family_named_count > 0:
        dominant_first_name, dominant_count = max(
            summary.first_name_counts.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
        family_dominance_ratio = float(dominant_count / family_named_count)
        if int(family_named_count) >= int(GENERIC_FAMILY_MIN_COUNT) and float(family_dominance_ratio) >= float(
            GENERIC_FAMILY_MIN_RATIO
        ):
            family_id = str(dominant_first_name)
    return ClusterProfile(
        cluster_id=str(summary.component_key),
        family_id=str(family_id),
        dominant_first_name=str(dominant_first_name) if dominant_first_name is not None else None,
        family_dominance_ratio=float(family_dominance_ratio),
        family_named_count=int(family_named_count),
    )


def counter_query_overlap(query_values: frozenset[str], counter: Counter[str], size: int) -> float:
    """Return average per-query-token coverage in one cluster counter."""

    if size <= 0 or not query_values or not counter:
        return 0.0
    overlap = sum(float(counter[value]) / float(size) for value in query_values if value in counter)
    return float(overlap / float(len(query_values)))


def middle_initial_compatibility(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return the promoted middle-initial compatibility signal."""

    if not query.middle_initials or not summary.middle_initial_counts or summary.size <= 0:
        return 0.0
    overlap = query.middle_initials.intersection(summary.middle_initial_counts.keys())
    if overlap:
        return float(
            sum(float(summary.middle_initial_counts[value]) / float(summary.size) for value in overlap)
            / float(len(query.middle_initials))
        )
    return _MIDDLE_INITIAL_CONFLICT_SCORE


def year_compatibility(query_year: int | None, summary: ClusterSummary) -> float:
    """Return the promoted publication-year compatibility signal."""

    if query_year is None or summary.year_mean is None:
        return 0.0
    distance = abs(float(query_year) - float(summary.year_mean))
    score = max(0.0, 1.0 - (distance / _YEAR_SCORE_DECAY_YEARS))
    if summary.year_min is not None and summary.year_max is not None:
        if (
            query_year < int(summary.year_min) - _YEAR_SCORE_RANGE_GAP
            or query_year > int(summary.year_max) + _YEAR_SCORE_RANGE_GAP
        ):
            score -= _YEAR_SCORE_RANGE_PENALTY
    return float(score)


def title_overlap(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return query-title overlap against the cluster title counter."""

    return float(counter_query_overlap(query.title_terms, summary.title_counts, summary.size))


def specter_exemplar_similarity(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return max SPECTER similarity to cluster exemplars."""

    query_vector = getattr(query, "specter", None)
    exemplar_vectors = list(getattr(summary, "exemplar_vectors", []) or [])
    if query_vector is None or not exemplar_vectors:
        return 0.0
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 0.0:
        return 0.0
    best = 0.0
    for exemplar in exemplar_vectors:
        exemplar_norm = float(np.linalg.norm(exemplar))
        if exemplar_norm <= 0.0:
            continue
        best = max(best, float(np.dot(query_vector, exemplar) / float(query_norm * exemplar_norm)))
    return float(best)


def rank_top_summaries_rust_hybrid_centroid(
    *,
    query: QueryFeatures,
    retriever: RustHybridCentroidRetrieverHandle,
    component_keys: list[str],
    num_threads: int,
    override_summary: ClusterSummary | None = None,
) -> list[tuple[float, ClusterSummary]]:
    """Score a known component subset with the frozen Rust hybrid-centroid path."""

    if not component_keys:
        return []
    ranked_component_keys, scores = retriever.retriever.top_k_hybrid_centroid_subset(
        query,
        component_keys,
        top_k=len(component_keys),
        num_threads=max(1, int(num_threads)),
        override_summary=override_summary,
    )
    return [
        (
            float(score),
            (
                override_summary
                if override_summary is not None and str(component_key) == str(override_summary.component_key)
                else retriever.summary_by_component[str(component_key)]
            ),
        )
        for component_key, score in zip(ranked_component_keys, scores, strict=True)
    ]


__all__ = [
    "DEFAULT_CHOOSER_CACHE_MAX_TOP_K",
    "ClusterProfile",
    "build_cluster_profile",
    "build_cluster_summary",
    "build_labeled_retrieval_subblock_index",
    "build_rust_hybrid_centroid_retriever",
    "counter_query_overlap",
    "middle_initial_compatibility",
    "rank_top_summaries_rust_hybrid_centroid",
    "specter_exemplar_similarity",
    "title_overlap",
    "year_compatibility",
]
