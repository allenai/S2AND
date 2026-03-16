from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import s2and.data as s2and_data_module
import s2and.subblocking as s2and_subblocking_module
from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.subblocking import (
    _signature_affiliation_feature_keys,
    _signature_coauthor_blocks_for_specter,
    _signature_name_parts_for_subblocking,
)
from s2and.text import normalize_text, same_prefix_tokens

TOP_KS = (1, 5, 10, 20, 50, 100)
DEFAULT_SIGNATURE_BUDGETS = (25, 50, 100, 250, 500, 1000)
QUERY_VIEW_INFORMATION_PRIORITY = {
    "initial_only_nearly_empty": 0,
    "initial_only_sparse_metadata": 1,
    "initial_only_no_specter": 2,
    "initial_only": 3,
    "full": 4,
}
_ORIGINAL_COMPUTE_BLOCK = s2and_data_module.compute_block


def _safe_compute_block(name: str | None) -> str:
    normalized_name = normalize_text(name or "")
    if not normalized_name:
        return ""
    return _ORIGINAL_COMPUTE_BLOCK(normalized_name)


def _install_safe_compute_block_patch() -> None:
    # Some real datasets contain blank coauthor names. For this retrieval experiment,
    # treat them as missing instead of failing ingest or query feature extraction.
    s2and_data_module.compute_block = _safe_compute_block
    s2and_subblocking_module.compute_block = _safe_compute_block


@dataclass(frozen=True)
class QueryCase:
    dataset: str
    block_key: str
    component_key: str
    cluster_id: str
    heldout_signature_id: str
    block_size: int
    component_size: int
    initial_info_bucket: str


@dataclass(frozen=True)
class QueryFeatures:
    first: str
    middle: str
    first_initial: str
    middle_initials: frozenset[str]
    coauthor_blocks: frozenset[str]
    affiliation_terms: frozenset[str]
    venue_terms: frozenset[str]
    year: int | None
    orcid: str | None
    specter: np.ndarray | None
    has_specter: bool
    has_coauthors: bool
    has_affiliations: bool
    has_full_first: bool
    has_middle: bool


@dataclass
class ClusterSummary:
    component_key: str
    cluster_id: str
    block_key: str
    size: int
    first_name_counts: Counter[str]
    middle_initial_counts: Counter[str]
    coauthor_counts: Counter[str]
    affiliation_counts: Counter[str]
    venue_counts: Counter[str]
    year_values: list[int]
    year_min: int | None
    year_max: int | None
    year_mean: float | None
    orcid_values: frozenset[str]
    specter_centroid: np.ndarray | None
    exemplar_vectors: list[np.ndarray]


def _resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(
        f"Missing dataset file for '{dataset_name}'. Tried '{preferred_path}' and '{fallback_path}'."
    )


def _resolve_specter_file(data_root: str, dataset_name: str) -> str | None:
    candidates = (
        f"{dataset_name}_specter.pickle",
        "specter.pickle",
        f"{dataset_name}_specter2.pkl",
        "specter2.pkl",
    )
    for candidate in candidates:
        path = os.path.join(data_root, dataset_name, candidate)
        if os.path.exists(path):
            return path
    return None


def _safe_mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _counter_query_overlap(query_values: frozenset[str], counter: Counter[str], size: int) -> float:
    if size <= 0 or not query_values or not counter:
        return 0.0
    overlap = sum(float(counter[value]) / float(size) for value in query_values if value in counter)
    return overlap / float(len(query_values))


def _middle_initial_score(query_initials: frozenset[str], counter: Counter[str], size: int) -> float:
    if not query_initials or not counter or size <= 0:
        return 0.0
    overlap = query_initials.intersection(counter.keys())
    if overlap:
        return sum(float(counter[value]) / float(size) for value in overlap) / float(len(query_initials))
    return -0.25


def _first_name_score(query_first: str, counter: Counter[str], size: int) -> float:
    if size <= 0 or len(query_first) <= 1 or not counter:
        return 0.0
    best = 0.0
    for first_name, count in counter.items():
        if len(first_name) <= 1:
            continue
        if same_prefix_tokens(query_first, first_name):
            best = max(best, float(count) / float(size))
    return best


def _year_score(query_year: int | None, summary: ClusterSummary) -> float:
    if query_year is None or summary.year_mean is None:
        return 0.0
    distance = abs(float(query_year) - float(summary.year_mean))
    score = max(0.0, 1.0 - (distance / 15.0))
    if summary.year_min is not None and summary.year_max is not None:
        if query_year < summary.year_min - 10 or query_year > summary.year_max + 10:
            score -= 0.15
    return score


def _size_prior(size: int, max_block_component_size: int) -> float:
    if size <= 0 or max_block_component_size <= 0:
        return 0.0
    return math.log1p(size) / math.log1p(max_block_component_size)


def _normalize_term_set(text: str | None) -> frozenset[str]:
    normalized = normalize_text(text or "")
    if not normalized:
        return frozenset()
    return frozenset(token for token in normalized.split() if token)


def _nonempty_feature_values(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(value.strip() for value in values if value is not None and value.strip())


def _resolve_sampling_query_view(query_views: list[str], sampling_query_view: str | None = None) -> str:
    if sampling_query_view is not None:
        return sampling_query_view
    if not query_views:
        return "full"
    return min(query_views, key=lambda view: QUERY_VIEW_INFORMATION_PRIORITY.get(view, math.inf))


def _normalize_signature_budgets(signature_budgets: Iterable[int]) -> tuple[int, ...]:
    normalized = sorted({int(budget) for budget in signature_budgets if int(budget) > 0})
    if not normalized:
        raise ValueError("signature_budgets must contain at least one positive integer")
    return tuple(normalized)


def _get_specter_vector(dataset: ANDData, paper_id: Any) -> np.ndarray | None:
    key = str(paper_id)
    vector = dataset.specter_embeddings.get(key)
    if vector is None:
        vector = dataset.specter_embeddings.get(paper_id)
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def _extract_query_features(
    dataset: ANDData,
    signature_id: str,
    *,
    feature_cache: dict[str, QueryFeatures] | None = None,
) -> QueryFeatures:
    if feature_cache is not None and signature_id in feature_cache:
        return feature_cache[signature_id]

    signature = dataset.signatures[signature_id]
    first, middle = _signature_name_parts_for_subblocking(signature)
    coauthor_blocks = _nonempty_feature_values(_signature_coauthor_blocks_for_specter(signature, dataset))
    affiliation_terms = _nonempty_feature_values(_signature_affiliation_feature_keys(signature))
    paper = dataset.papers.get(str(signature.paper_id))
    venue_terms = frozenset()
    year = None
    if paper is not None:
        venue_terms = _normalize_term_set(" ".join(part for part in [paper.venue, paper.journal_name] if part))
        year = paper.year
    specter = _get_specter_vector(dataset, signature.paper_id)
    middle_tokens = [token for token in middle.split() if token]
    middle_initials = frozenset(token[0] for token in middle_tokens)
    orcid = signature.author_info_orcid or None
    features = QueryFeatures(
        first=first,
        middle=middle,
        first_initial=first[:1],
        middle_initials=middle_initials,
        coauthor_blocks=coauthor_blocks,
        affiliation_terms=affiliation_terms,
        venue_terms=venue_terms,
        year=year,
        orcid=orcid,
        specter=specter,
        has_specter=specter is not None,
        has_coauthors=bool(coauthor_blocks),
        has_affiliations=bool(affiliation_terms),
        has_full_first=len(first) > 1,
        has_middle=bool(middle_tokens),
    )
    if feature_cache is not None:
        feature_cache[signature_id] = features
    return features


def _mask_query_features(base: QueryFeatures, view: str) -> QueryFeatures:
    if view == "full":
        return base

    first = base.first_initial
    masked = QueryFeatures(
        first=first,
        middle="",
        first_initial=base.first_initial,
        middle_initials=frozenset(),
        coauthor_blocks=base.coauthor_blocks,
        affiliation_terms=base.affiliation_terms,
        venue_terms=base.venue_terms,
        year=base.year,
        orcid=base.orcid,
        specter=base.specter,
        has_specter=base.has_specter,
        has_coauthors=base.has_coauthors,
        has_affiliations=base.has_affiliations,
        has_full_first=False,
        has_middle=False,
    )
    if view == "initial_only":
        return masked
    if view == "initial_only_no_specter":
        return QueryFeatures(**{**asdict(masked), "specter": None, "has_specter": False})
    if view == "initial_only_sparse_metadata":
        return QueryFeatures(
            **{
                **asdict(masked),
                "coauthor_blocks": frozenset(),
                "affiliation_terms": frozenset(),
                "has_coauthors": False,
                "has_affiliations": False,
            }
        )
    if view == "initial_only_nearly_empty":
        return QueryFeatures(
            **{
                **asdict(masked),
                "coauthor_blocks": frozenset(),
                "affiliation_terms": frozenset(),
                "specter": None,
                "has_specter": False,
                "has_coauthors": False,
                "has_affiliations": False,
            }
        )
    raise ValueError(f"Unknown query view: {view}")


def _select_exemplars(vectors: list[np.ndarray], max_exemplars: int) -> list[np.ndarray]:
    if max_exemplars <= 0 or not vectors:
        return []
    if len(vectors) <= max_exemplars:
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]

    stack = np.vstack(vectors)
    centroid = np.mean(stack, axis=0)
    selected_indices: list[int] = [int(np.argmax(np.linalg.norm(stack - centroid, axis=1)))]
    while len(selected_indices) < max_exemplars:
        best_index = None
        best_distance = -1.0
        for idx in range(len(vectors)):
            if idx in selected_indices:
                continue
            candidate = stack[idx]
            min_distance = min(float(np.linalg.norm(candidate - stack[selected])) for selected in selected_indices)
            if min_distance > best_distance:
                best_distance = min_distance
                best_index = idx
        if best_index is None:
            break
        selected_indices.append(best_index)
    return [np.asarray(vectors[idx], dtype=np.float32) for idx in selected_indices]


def _build_cluster_summary(
    dataset: ANDData,
    block_key: str,
    cluster_id: str,
    component_key: str,
    signature_ids: list[str],
    max_exemplars: int,
    *,
    feature_cache: dict[str, QueryFeatures] | None = None,
) -> ClusterSummary:
    first_name_counts: Counter[str] = Counter()
    middle_initial_counts: Counter[str] = Counter()
    coauthor_counts: Counter[str] = Counter()
    affiliation_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    year_values: list[int] = []
    orcid_values: set[str] = set()
    specter_vectors: list[np.ndarray] = []

    for signature_id in signature_ids:
        features = _extract_query_features(dataset, signature_id, feature_cache=feature_cache)
        if len(features.first) > 1:
            first_name_counts[features.first] += 1
        for initial in features.middle_initials:
            middle_initial_counts[initial] += 1
        for block in features.coauthor_blocks:
            coauthor_counts[block] += 1
        for term in features.affiliation_terms:
            affiliation_counts[term] += 1
        for term in features.venue_terms:
            venue_counts[term] += 1
        if features.year is not None:
            year_values.append(int(features.year))
        if features.orcid is not None:
            orcid_values.add(features.orcid)
        if features.specter is not None:
            specter_vectors.append(features.specter)

    centroid = None
    if specter_vectors:
        centroid = np.mean(np.vstack(specter_vectors), axis=0).astype(np.float32)

    return ClusterSummary(
        component_key=component_key,
        cluster_id=cluster_id,
        block_key=block_key,
        size=len(signature_ids),
        first_name_counts=first_name_counts,
        middle_initial_counts=middle_initial_counts,
        coauthor_counts=coauthor_counts,
        affiliation_counts=affiliation_counts,
        venue_counts=venue_counts,
        year_values=year_values,
        year_min=min(year_values) if year_values else None,
        year_max=max(year_values) if year_values else None,
        year_mean=_safe_mean(year_values),
        orcid_values=frozenset(orcid_values),
        specter_centroid=centroid,
        exemplar_vectors=_select_exemplars(specter_vectors, max_exemplars=max_exemplars),
    )


def _score_summary(method: str, query: QueryFeatures, summary: ClusterSummary, max_block_component_size: int) -> float:
    size_prior = _size_prior(summary.size, max_block_component_size)
    coauthor_score = _counter_query_overlap(query.coauthor_blocks, summary.coauthor_counts, summary.size)
    affiliation_score = _counter_query_overlap(query.affiliation_terms, summary.affiliation_counts, summary.size)
    venue_score = _counter_query_overlap(query.venue_terms, summary.venue_counts, summary.size)
    middle_score = _middle_initial_score(query.middle_initials, summary.middle_initial_counts, summary.size)
    first_name_score = _first_name_score(query.first, summary.first_name_counts, summary.size)
    year_score = _year_score(query.year, summary)
    centroid_score = _cosine_similarity(query.specter, summary.specter_centroid)
    exemplar_score = max(
        (_cosine_similarity(query.specter, vector) for vector in summary.exemplar_vectors),
        default=0.0,
    )

    if method == "size_prior":
        return size_prior
    if method == "coauthor_sparse":
        return coauthor_score
    if method == "specter_centroid":
        return centroid_score
    if method == "hybrid_centroid":
        return (
            0.42 * centroid_score
            + 0.23 * coauthor_score
            + 0.12 * affiliation_score
            + 0.06 * venue_score
            + 0.05 * middle_score
            + 0.07 * first_name_score
            + 0.03 * year_score
            + 0.02 * size_prior
        )
    if method == "hybrid_exemplar_4":
        return (
            0.40 * exemplar_score
            + 0.23 * coauthor_score
            + 0.12 * affiliation_score
            + 0.06 * venue_score
            + 0.05 * middle_score
            + 0.07 * first_name_score
            + 0.05 * year_score
            + 0.02 * size_prior
        )
    raise ValueError(f"Unknown method: {method}")


def _has_middle_initial_conflict(query: QueryFeatures, summary: ClusterSummary) -> bool:
    return (
        bool(query.middle_initials)
        and bool(summary.middle_initial_counts)
        and query.middle_initials.isdisjoint(summary.middle_initial_counts.keys())
    )


def _has_impossible_year_conflict(query: QueryFeatures, summary: ClusterSummary, max_year_gap: int = 35) -> bool:
    if query.year is None or summary.year_min is None or summary.year_max is None:
        return False
    return query.year < summary.year_min - max_year_gap or query.year > summary.year_max + max_year_gap


def _apply_hard_filters(
    query: QueryFeatures,
    candidate_summaries: list[ClusterSummary],
) -> tuple[list[ClusterSummary], dict[str, int]]:
    filtered_summaries = list(candidate_summaries)
    orcid_filter_applied = 0
    middle_initial_filter_applied = 0
    year_range_filter_applied = 0

    if query.orcid is not None:
        orcid_matches = [summary for summary in filtered_summaries if query.orcid in summary.orcid_values]
        if orcid_matches:
            orcid_filter_applied = 1
            filtered_summaries = orcid_matches

    middle_before = len(filtered_summaries)
    middle_filtered = [summary for summary in filtered_summaries if not _has_middle_initial_conflict(query, summary)]
    if middle_filtered:
        middle_initial_filter_applied = int(len(middle_filtered) < middle_before)
        filtered_summaries = middle_filtered

    year_before = len(filtered_summaries)
    year_filtered = [summary for summary in filtered_summaries if not _has_impossible_year_conflict(query, summary)]
    if year_filtered:
        year_range_filter_applied = int(len(year_filtered) < year_before)
        filtered_summaries = year_filtered

    return filtered_summaries, {
        "orcid_filter_applied": orcid_filter_applied,
        "middle_initial_filter_applied": middle_initial_filter_applied,
        "year_range_filter_applied": year_range_filter_applied,
        "scored_candidate_components": len(filtered_summaries),
        "scored_candidate_signatures": sum(summary.size for summary in filtered_summaries),
    }


def _materialized_signature_count(ranked_summaries: list[ClusterSummary], k: int) -> int:
    return sum(summary.size for summary in ranked_summaries[:k])


def _materialized_cluster_count(ranked_summaries: list[ClusterSummary], k: int) -> int:
    return min(k, len(ranked_summaries))


def _materialized_signature_fraction(
    ranked_summaries: list[ClusterSummary],
    k: int,
    candidate_signatures: int,
) -> float:
    if candidate_signatures <= 0:
        return 0.0
    return _materialized_signature_count(ranked_summaries, k) / float(candidate_signatures)


def _hit_within_signature_budget(
    ranked_summaries: list[ClusterSummary],
    true_component_key: str,
    signature_budget: int,
) -> int:
    if signature_budget <= 0:
        return 0
    materialized_signatures = 0
    for summary in ranked_summaries:
        next_total = materialized_signatures + summary.size
        if next_total > signature_budget:
            break
        materialized_signatures = next_total
        if summary.component_key == true_component_key:
            return 1
    return 0


def _rank_summaries(
    method: str,
    query: QueryFeatures,
    candidate_summaries: list[ClusterSummary],
    max_block_component_size: int,
) -> list[tuple[float, ClusterSummary]]:
    scored = [
        (_score_summary(method, query, summary, max_block_component_size=max_block_component_size), summary)
        for summary in candidate_summaries
    ]
    scored.sort(key=lambda item: (-item[0], item[1].component_key))
    return scored


def _block_bucket(size: int) -> str:
    if size < 10:
        return "2_9"
    if size < 50:
        return "10_49"
    if size < 200:
        return "50_199"
    return "200_plus"


def _component_bucket(size: int) -> str:
    if size <= 2:
        return "2"
    if size <= 5:
        return "3_5"
    if size <= 10:
        return "6_10"
    if size <= 20:
        return "11_20"
    return "21_plus"


def _initial_info_bucket(features: QueryFeatures) -> str:
    if features.has_specter and (features.has_coauthors or features.has_affiliations):
        return "rich"
    if features.has_specter:
        return "specter_only"
    if features.has_coauthors or features.has_affiliations:
        return "metadata_only"
    return "sparse"


def _choose_heldout_signature(signature_ids: list[str], seed: int) -> str:
    rng = random.Random(seed)
    return rng.choice(signature_ids)


def _stable_component_seed(component_key: str, base_seed: int) -> int:
    digest = hashlib.sha256(component_key.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) + int(base_seed)


def _round_robin_sample(cases: list[QueryCase], limit: int, seed: int) -> list[QueryCase]:
    if limit <= 0 or len(cases) <= limit:
        return cases
    grouped: dict[tuple[str, str, str], list[QueryCase]] = defaultdict(list)
    for case in cases:
        key = (_block_bucket(case.block_size), _component_bucket(case.component_size), case.initial_info_bucket)
        grouped[key].append(case)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    ordered_keys = sorted(grouped.keys())
    selected: list[QueryCase] = []
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            if not grouped[key]:
                continue
            selected.append(grouped[key].pop())
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _compute_signature_feature_counts(
    dataset: ANDData,
    *,
    feature_cache: dict[str, QueryFeatures] | None = None,
) -> dict[str, int]:
    counts = {
        "full_first": 0,
        "middle": 0,
        "specter": 0,
        "coauthors": 0,
        "affiliations": 0,
    }
    for signature_id in dataset.signatures:
        features = _extract_query_features(dataset, signature_id, feature_cache=feature_cache)
        if features.has_full_first:
            counts["full_first"] += 1
        if features.has_middle:
            counts["middle"] += 1
        if features.has_specter:
            counts["specter"] += 1
        if features.has_coauthors:
            counts["coauthors"] += 1
        if features.has_affiliations:
            counts["affiliations"] += 1
    return counts


def _build_query_cases(
    dataset_name: str,
    dataset: ANDData,
    limit_queries: int,
    seed: int,
    sampling_query_view: str,
    *,
    feature_cache: dict[str, QueryFeatures] | None = None,
) -> tuple[list[QueryCase], dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    if dataset.clusters is None:
        raise RuntimeError(f"Dataset '{dataset_name}' has no clusters.")

    components: dict[str, list[str]] = {}
    missing_cluster_signature_ids = 0
    for cluster_id, cluster_info in dataset.clusters.items():
        signatures_by_block: dict[str, list[str]] = defaultdict(list)
        for signature_id in cluster_info["signature_ids"]:
            signature_key = str(signature_id)
            block_key = dataset.signature_to_block.get(signature_key)
            if block_key is None:
                missing_cluster_signature_ids += 1
                continue
            signatures_by_block[block_key].append(signature_key)
        for block_key, signature_ids in signatures_by_block.items():
            component_key = f"{block_key}::{cluster_id}"
            components[component_key] = signature_ids

    block_to_component_keys: dict[str, list[str]] = defaultdict(list)
    block_sizes: dict[str, int] = defaultdict(int)
    for component_key, signature_ids in components.items():
        block_key, _cluster_id = component_key.split("::", 1)
        block_to_component_keys[block_key].append(component_key)
        block_sizes[block_key] += len(signature_ids)

    census = {
        "blocks": len(block_to_component_keys),
        "components": len(components),
        "eligible_components": 0,
        "signatures_total": len(dataset.signatures),
        "missing_cluster_signature_ids": missing_cluster_signature_ids,
        "signature_feature_counts": _compute_signature_feature_counts(dataset, feature_cache=feature_cache),
        "eligible_query_feature_counts": {
            "full_first": 0,
            "middle": 0,
            "specter": 0,
            "coauthors": 0,
            "affiliations": 0,
        },
        "block_size_buckets": Counter(),
        "component_size_buckets": Counter(),
    }
    for block_size in block_sizes.values():
        census["block_size_buckets"][_block_bucket(block_size)] += 1

    all_cases: list[QueryCase] = []
    for component_key, signature_ids in components.items():
        component_size = len(signature_ids)
        census["component_size_buckets"][_component_bucket(component_size)] += 1
        block_key, cluster_id = component_key.split("::", 1)
        block_size = block_sizes[block_key]
        if component_size < 2:
            continue
        census["eligible_components"] += 1
        heldout_signature_id = _choose_heldout_signature(
            signature_ids,
            seed=_stable_component_seed(component_key, seed),
        )
        heldout_features = _extract_query_features(dataset, heldout_signature_id, feature_cache=feature_cache)
        if heldout_features.has_full_first:
            census["eligible_query_feature_counts"]["full_first"] += 1
        if heldout_features.has_middle:
            census["eligible_query_feature_counts"]["middle"] += 1
        if heldout_features.has_specter:
            census["eligible_query_feature_counts"]["specter"] += 1
        if heldout_features.has_coauthors:
            census["eligible_query_feature_counts"]["coauthors"] += 1
        if heldout_features.has_affiliations:
            census["eligible_query_feature_counts"]["affiliations"] += 1
        sampling_features = _mask_query_features(heldout_features, sampling_query_view)
        all_cases.append(
            QueryCase(
                dataset=dataset_name,
                block_key=block_key,
                component_key=component_key,
                cluster_id=cluster_id,
                heldout_signature_id=heldout_signature_id,
                block_size=block_size,
                component_size=component_size,
                initial_info_bucket=_initial_info_bucket(sampling_features),
            )
        )

    sampled = _round_robin_sample(all_cases, limit=limit_queries, seed=seed)
    return sampled, census, block_to_component_keys, components


def _format_float(value: float) -> float:
    return round(float(value), 6)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "queries": len(rows),
        "recall": {},
        "mrr": None,
        "rank_mean": None,
        "rank_median": None,
        "latency_ms_mean": None,
        "latency_ms_median": None,
        "latency_ms_p95": None,
        "query_feature_latency_ms_mean": None,
        "view_prepare_latency_ms_mean": None,
        "ranking_latency_ms_mean": None,
        "candidate_components_mean": None,
        "candidate_components_median": None,
        "scored_candidate_components_mean": None,
        "scored_candidate_components_median": None,
        "scored_candidate_signatures_mean": None,
        "candidate_component_distribution": {},
        "hard_filter_rates": {},
        "materialized_clusters": {},
        "materialized_signatures": {},
        "materialized_signature_fraction": {},
        "recall_under_signature_budget": {},
    }
    if not rows:
        return summary

    ranks = [int(row["true_rank"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    candidate_counts = [int(row["candidate_components"]) for row in rows]
    scored_candidate_counts = [int(row["scored_candidate_components"]) for row in rows]
    scored_candidate_signatures = [int(row["scored_candidate_signatures"]) for row in rows]
    summary["mrr"] = _format_float(statistics.mean(1.0 / rank for rank in ranks))
    summary["rank_mean"] = _format_float(statistics.mean(ranks))
    summary["rank_median"] = _format_float(statistics.median(ranks))
    summary["latency_ms_mean"] = _format_float(statistics.mean(latencies))
    summary["latency_ms_median"] = _format_float(statistics.median(latencies))
    summary["latency_ms_p95"] = _format_float(float(np.percentile(latencies, 95)))
    summary["query_feature_latency_ms_mean"] = _format_float(
        statistics.mean(float(row["query_feature_latency_ms"]) for row in rows)
    )
    summary["view_prepare_latency_ms_mean"] = _format_float(
        statistics.mean(float(row["view_prepare_latency_ms"]) for row in rows)
    )
    summary["ranking_latency_ms_mean"] = _format_float(
        statistics.mean(float(row["ranking_latency_ms"]) for row in rows)
    )
    summary["candidate_components_mean"] = _format_float(statistics.mean(candidate_counts))
    summary["candidate_components_median"] = _format_float(statistics.median(candidate_counts))
    summary["scored_candidate_components_mean"] = _format_float(statistics.mean(scored_candidate_counts))
    summary["scored_candidate_components_median"] = _format_float(statistics.median(scored_candidate_counts))
    summary["scored_candidate_signatures_mean"] = _format_float(statistics.mean(scored_candidate_signatures))
    summary["candidate_component_distribution"] = {
        "eq_1_rate": _format_float(sum(1 for value in candidate_counts if value == 1) / len(candidate_counts)),
        "lt_3_rate": _format_float(sum(1 for value in candidate_counts if value < 3) / len(candidate_counts)),
        "ge_3_rate": _format_float(sum(1 for value in candidate_counts if value >= 3) / len(candidate_counts)),
        "ge_6_rate": _format_float(sum(1 for value in candidate_counts if value >= 6) / len(candidate_counts)),
    }
    summary["hard_filter_rates"] = {
        "orcid_exact": _format_float(statistics.mean(int(row["orcid_filter_applied"]) for row in rows)),
        "middle_initial": _format_float(statistics.mean(int(row["middle_initial_filter_applied"]) for row in rows)),
        "year_range": _format_float(statistics.mean(int(row["year_range_filter_applied"]) for row in rows)),
    }

    for k in TOP_KS:
        hits = [int(row[f"hit@{k}"]) for row in rows]
        summary["recall"][str(k)] = _format_float(sum(hits) / len(hits))
        materialized_clusters = [int(row[f"materialized_clusters@{k}"]) for row in rows]
        materialized = [int(row[f"materialized_signatures@{k}"]) for row in rows]
        materialized_fraction = [float(row[f"materialized_signature_fraction@{k}"]) for row in rows]
        summary["materialized_clusters"][str(k)] = {
            "mean": _format_float(statistics.mean(materialized_clusters)),
            "median": _format_float(statistics.median(materialized_clusters)),
        }
        summary["materialized_signatures"][str(k)] = {
            "mean": _format_float(statistics.mean(materialized)),
            "median": _format_float(statistics.median(materialized)),
        }
        summary["materialized_signature_fraction"][str(k)] = {
            "mean": _format_float(statistics.mean(materialized_fraction)),
            "median": _format_float(statistics.median(materialized_fraction)),
        }

    budget_keys = sorted(int(key.split("@", 1)[1]) for key in rows[0].keys() if key.startswith("hit_budget@"))
    for budget in budget_keys:
        hits = [int(row[f"hit_budget@{budget}"]) for row in rows]
        summary["recall_under_signature_budget"][str(budget)] = _format_float(sum(hits) / len(hits))

    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_failures_rows(rows: list[dict[str, Any]], *, top_k: int = 20) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get(f"hit@{top_k}", 0)) == 0]


def _build_dataset_census_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset_name, dataset_diagnostics in diagnostics.items():
        if "census" in dataset_diagnostics:
            payload[dataset_name] = dataset_diagnostics["census"]
    return payload


def _build_summary_payload(
    args: argparse.Namespace,
    all_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "config": {
            "datasets": list(args.datasets),
            "query_views": list(args.query_views),
            "methods": list(args.methods),
            "limit_queries": int(args.limit_queries),
            "seed": int(args.seed),
            "n_jobs": int(args.n_jobs),
            "sampling_query_view": str(args.sampling_query_view),
            "signature_budgets": [int(budget) for budget in args.signature_budgets],
            "backend_env": os.environ.get("S2AND_BACKEND", "auto"),
            "latency_definition": (
                "query_features + view_mask + hard_filters + ranking; " "excludes summary build and persisted-index I/O"
            ),
        },
        "overall": {},
        "overall_candidate_floor": {},
        "by_dataset": {},
        "diagnostics": diagnostics,
    }

    for method in args.methods:
        for query_view in args.query_views:
            method_rows = [row for row in all_rows if row["method"] == method and row["query_view"] == query_view]
            summary["overall"][f"{method}::{query_view}"] = _aggregate_metrics(method_rows)

    for floor in (3, 6):
        floor_summary: dict[str, Any] = {}
        for method in args.methods:
            for query_view in args.query_views:
                floor_rows = [
                    row
                    for row in all_rows
                    if row["method"] == method
                    and row["query_view"] == query_view
                    and int(row["candidate_components"]) >= floor
                ]
                floor_summary[f"{method}::{query_view}"] = _aggregate_metrics(floor_rows)
        summary["overall_candidate_floor"][f"ge_{floor}"] = floor_summary

    for dataset_name in args.datasets:
        dataset_summary: dict[str, Any] = {}
        for method in args.methods:
            for query_view in args.query_views:
                rows = [
                    row
                    for row in all_rows
                    if row["dataset"] == dataset_name and row["method"] == method and row["query_view"] == query_view
                ]
                dataset_summary[f"{method}::{query_view}"] = _aggregate_metrics(rows)
        summary["by_dataset"][dataset_name] = dataset_summary

    return summary


def _write_progress(
    output_dir: Path,
    args: argparse.Namespace,
    all_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    summary = _build_summary_payload(args=args, all_rows=all_rows, diagnostics=diagnostics)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "diagnostics.json", diagnostics)
    _write_json(output_dir / "dataset_census.json", _build_dataset_census_payload(diagnostics))
    _write_csv(output_dir / "per_query.csv", all_rows)
    _write_csv(output_dir / "failures_topk.csv", _build_failures_rows(all_rows))
    return summary


def _load_dataset(data_root: str, dataset_name: str, n_jobs: int) -> ANDData:
    _install_safe_compute_block_patch()
    signatures_path = _resolve_dataset_file(
        data_root,
        dataset_name,
        f"{dataset_name}_signatures.json",
        "signatures.json",
    )
    papers_path = _resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_papers.json", "papers.json")
    clusters_path = _resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_clusters.json", "clusters.json")
    specter_path = _resolve_specter_file(data_root, dataset_name)

    return ANDData(
        signatures=signatures_path,
        papers=papers_path,
        name=dataset_name,
        mode="inference",
        specter_embeddings=specter_path,
        clusters=clusters_path,
        block_type="s2",
        n_jobs=n_jobs,
        load_name_counts=False,
        preprocess=True,
        random_seed=13,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
    )


def _evaluate_dataset(
    dataset_name: str,
    dataset: ANDData,
    query_views: list[str],
    methods: list[str],
    max_exemplars: int,
    limit_queries: int,
    seed: int,
    sampling_query_view: str,
    signature_budgets: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feature_cache: dict[str, QueryFeatures] = {}
    query_cases, census, block_to_component_keys, full_component_signatures = _build_query_cases(
        dataset_name=dataset_name,
        dataset=dataset,
        limit_queries=limit_queries,
        seed=seed,
        sampling_query_view=sampling_query_view,
        feature_cache=feature_cache,
    )

    full_summaries: dict[str, ClusterSummary] = {}
    build_full_start = time.perf_counter()
    for component_key, signature_ids in full_component_signatures.items():
        block_key, cluster_id = component_key.split("::", 1)
        full_summaries[component_key] = _build_cluster_summary(
            dataset=dataset,
            block_key=block_key,
            cluster_id=cluster_id,
            component_key=component_key,
            signature_ids=signature_ids,
            max_exemplars=max_exemplars,
            feature_cache=feature_cache,
        )
    full_summary_build_ms = (time.perf_counter() - build_full_start) * 1000.0

    residual_cache: dict[tuple[str, str], ClusterSummary] = {}
    residual_build_times_ms: list[float] = []
    results: list[dict[str, Any]] = []

    for query_case in query_cases:
        query_feature_start = time.perf_counter()
        base_query = _extract_query_features(dataset, query_case.heldout_signature_id, feature_cache=feature_cache)
        query_feature_latency_ms = (time.perf_counter() - query_feature_start) * 1000.0
        component_keys_in_block = block_to_component_keys[query_case.block_key]
        raw_candidate_summaries: list[ClusterSummary] = []
        for component_key in component_keys_in_block:
            if component_key != query_case.component_key:
                raw_candidate_summaries.append(full_summaries[component_key])
                continue
            residual_key = (component_key, query_case.heldout_signature_id)
            if residual_key not in residual_cache:
                signature_ids = [
                    signature_id
                    for signature_id in full_component_signatures[component_key]
                    if signature_id != query_case.heldout_signature_id
                ]
                block_key, cluster_id = component_key.split("::", 1)
                start = time.perf_counter()
                residual_cache[residual_key] = _build_cluster_summary(
                    dataset=dataset,
                    block_key=block_key,
                    cluster_id=cluster_id,
                    component_key=component_key,
                    signature_ids=signature_ids,
                    max_exemplars=max_exemplars,
                    feature_cache=feature_cache,
                )
                residual_build_times_ms.append((time.perf_counter() - start) * 1000.0)
            raw_candidate_summaries.append(residual_cache[residual_key])

        max_block_component_size = max((summary.size for summary in raw_candidate_summaries), default=0)
        candidate_signatures = sum(summary.size for summary in raw_candidate_summaries)

        for query_view in query_views:
            view_prepare_start = time.perf_counter()
            query = _mask_query_features(base_query, query_view)
            candidate_summaries, filter_state = _apply_hard_filters(query, raw_candidate_summaries)
            view_prepare_latency_ms = (time.perf_counter() - view_prepare_start) * 1000.0
            for method in methods:
                ranking_start = time.perf_counter()
                ranked = _rank_summaries(
                    method,
                    query,
                    candidate_summaries,
                    max_block_component_size=max_block_component_size,
                )
                ranking_latency_ms = (time.perf_counter() - ranking_start) * 1000.0
                latency_ms = query_feature_latency_ms + view_prepare_latency_ms + ranking_latency_ms
                ranked_summaries = [summary for _score, summary in ranked]
                ranked_keys = [summary.component_key for summary in ranked_summaries]
                score_by_component_key = {summary.component_key: score for score, summary in ranked}
                true_rank = (
                    ranked_keys.index(query_case.component_key) + 1
                    if query_case.component_key in ranked_keys
                    else len(ranked_keys) + 1
                )

                row: dict[str, Any] = {
                    "dataset": dataset_name,
                    "block_key": query_case.block_key,
                    "component_key": query_case.component_key,
                    "cluster_id": query_case.cluster_id,
                    "heldout_signature_id": query_case.heldout_signature_id,
                    "query_view": query_view,
                    "method": method,
                    "true_rank": true_rank,
                    "candidate_components": len(raw_candidate_summaries),
                    "candidate_signatures": candidate_signatures,
                    "scored_candidate_components": filter_state["scored_candidate_components"],
                    "scored_candidate_signatures": filter_state["scored_candidate_signatures"],
                    "latency_ms": _format_float(latency_ms),
                    "query_feature_latency_ms": _format_float(query_feature_latency_ms),
                    "view_prepare_latency_ms": _format_float(view_prepare_latency_ms),
                    "ranking_latency_ms": _format_float(ranking_latency_ms),
                    "block_size": query_case.block_size,
                    "component_size": query_case.component_size,
                    "sampling_info_bucket": query_case.initial_info_bucket,
                    "initial_info_bucket": _initial_info_bucket(query),
                    "has_full_first": int(query.has_full_first),
                    "has_middle": int(query.has_middle),
                    "has_specter": int(query.has_specter),
                    "has_coauthors": int(query.has_coauthors),
                    "has_affiliations": int(query.has_affiliations),
                    "source_has_full_first": int(base_query.has_full_first),
                    "source_has_middle": int(base_query.has_middle),
                    "source_has_specter": int(base_query.has_specter),
                    "source_has_coauthors": int(base_query.has_coauthors),
                    "source_has_affiliations": int(base_query.has_affiliations),
                    "orcid_filter_applied": filter_state["orcid_filter_applied"],
                    "middle_initial_filter_applied": filter_state["middle_initial_filter_applied"],
                    "year_range_filter_applied": filter_state["year_range_filter_applied"],
                    "top_component_keys@5": "|".join(ranked_keys[:5]),
                    "top_component_scores@5": "|".join(f"{score:.6f}" for score, _summary in ranked[:5]),
                    "true_component_score": (
                        _format_float(score_by_component_key[query_case.component_key])
                        if query_case.component_key in score_by_component_key
                        else None
                    ),
                }
                for k in TOP_KS:
                    row[f"hit@{k}"] = int(query_case.component_key in ranked_keys[:k])
                    row[f"materialized_clusters@{k}"] = _materialized_cluster_count(ranked_summaries, k)
                    row[f"materialized_signatures@{k}"] = _materialized_signature_count(ranked_summaries, k)
                    row[f"materialized_signature_fraction@{k}"] = _format_float(
                        _materialized_signature_fraction(ranked_summaries, k, candidate_signatures)
                    )
                for signature_budget in signature_budgets:
                    row[f"hit_budget@{signature_budget}"] = _hit_within_signature_budget(
                        ranked_summaries,
                        query_case.component_key,
                        signature_budget,
                    )
                results.append(row)

    diagnostics = {
        "census": census,
        "query_cases": len(query_cases),
        "full_summary_build_ms_total": _format_float(full_summary_build_ms),
        "full_summary_build_ms_per_component": _format_float(full_summary_build_ms / max(1, len(full_summaries))),
        "residual_summary_build_ms_mean": _format_float(statistics.mean(residual_build_times_ms))
        if residual_build_times_ms
        else None,
        "resolved_backend": dataset.runtime_context.resolved_backend,
        "sampling_query_view": sampling_query_view,
        "signature_budgets": list(signature_budgets),
        "specter_loaded": bool(dataset.specter_embeddings),
    }
    return results, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact within-block cluster retrieval summaries.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aminer", "arnetminer", "inspire", "inventors_s2and", "kisti", "orcid", "pubmed", "qian", "zbmath"],
    )
    parser.add_argument(
        "--query-views",
        nargs="+",
        default=["full", "initial_only", "initial_only_no_specter", "initial_only_sparse_metadata"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["size_prior", "coauthor_sparse", "specter_centroid", "hybrid_centroid", "hybrid_exemplar_4"],
    )
    parser.add_argument("--limit-queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-exemplars", type=int, default=4)
    parser.add_argument("--sampling-query-view", type=str, default=None)
    parser.add_argument("--signature-budgets", nargs="+", type=int, default=list(DEFAULT_SIGNATURE_BUDGETS))
    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT_PATH, "data"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT_PATH, "scratch", "cluster_retrieval"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sampling_query_view = _resolve_sampling_query_view(list(args.query_views), args.sampling_query_view)
    args.signature_budgets = list(_normalize_signature_budgets(args.signature_budgets))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    print(
        "Config: "
        f"backend={os.environ.get('S2AND_BACKEND', 'auto')} "
        f"datasets={args.datasets} "
        f"limit_queries={args.limit_queries} "
        f"query_views={args.query_views} "
        f"sampling_query_view={args.sampling_query_view} "
        f"signature_budgets={args.signature_budgets} "
        f"methods={args.methods}"
    )

    for dataset_name in args.datasets:
        dataset_start = time.perf_counter()
        try:
            dataset = _load_dataset(
                data_root=args.data_root,
                dataset_name=dataset_name,
                n_jobs=int(args.n_jobs),
            )
            dataset_rows, dataset_diagnostics = _evaluate_dataset(
                dataset_name=dataset_name,
                dataset=dataset,
                query_views=list(args.query_views),
                methods=list(args.methods),
                max_exemplars=int(args.max_exemplars),
                limit_queries=int(args.limit_queries),
                seed=int(args.seed),
                sampling_query_view=str(args.sampling_query_view),
                signature_budgets=tuple(int(budget) for budget in args.signature_budgets),
            )
            dataset_diagnostics["dataset_wall_seconds"] = _format_float(time.perf_counter() - dataset_start)
            diagnostics[dataset_name] = dataset_diagnostics
            all_rows.extend(dataset_rows)
            print(
                f"[{dataset_name}] backend={dataset.runtime_context.resolved_backend} "
                f"queries={dataset_diagnostics['query_cases']} "
                f"wall_s={dataset_diagnostics['dataset_wall_seconds']}"
            )
        except Exception as exc:
            diagnostics[dataset_name] = {
                "error": repr(exc),
                "dataset_wall_seconds": _format_float(time.perf_counter() - dataset_start),
            }
            print(f"[{dataset_name}] error={repr(exc)}")
        finally:
            summary = _write_progress(output_dir=output_dir, args=args, all_rows=all_rows, diagnostics=diagnostics)

    for method in args.methods:
        for query_view in args.query_views:
            metrics = summary["overall"][f"{method}::{query_view}"]
            print(
                f"{method} [{query_view}] "
                f"R@1={metrics['recall'].get('1', 0):.3f} "
                f"R@5={metrics['recall'].get('5', 0):.3f} "
                f"R@10={metrics['recall'].get('10', 0):.3f} "
                f"R@20={metrics['recall'].get('20', 0):.3f} "
                f"R@50={metrics['recall'].get('50', 0):.3f} "
                f"R@100={metrics['recall'].get('100', 0):.3f} "
                f"lat_ms_mean={metrics['latency_ms_mean']} "
                f"lat_ms_p95={metrics['latency_ms_p95']}"
            )


if __name__ == "__main__":
    main()
