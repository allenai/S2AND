"""Runtime-safe query and summary construction for incremental linking."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

import s2and_rust
from s2and.data import ANDData
from s2and.incremental_linking.gate_buckets import QueryView, normalize_query_views
from s2and.incremental_linking.retrieval import LinkerRetrievalBatch
from s2and.subblocking import signature_affiliation_feature_keys, signature_name_parts_for_subblocking
from s2and.text import (
    canonicalize_name_parts,
    compute_block,
    normalize_text,
    normalize_title,
    same_prefix_tokens,
)
from s2and.text import (
    name_counts as pairwise_name_counts,
)
from s2and.text import (
    normalize_orcid as normalize_orcid_text,
)

EMPTY_STRING_SET: frozenset[str] = frozenset()
PAIRWISE_NAME_COUNT_FEATURE_NAMES: tuple[str, ...] = (
    "first_name_count_min",
    "last_first_name_count_min",
    "last_name_count_min",
    "last_first_initial_count_min",
    "first_name_count_max",
    "last_first_name_count_max",
)
NAME_COUNT_RARITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "first_name_count_min_rarity",
    "last_first_name_count_min_rarity",
    "last_name_count_min_rarity",
    "last_first_initial_count_min_rarity",
    "first_name_count_max_rarity",
    "last_first_name_count_max_rarity",
    "first_prefix_x_last_first_name_count_min_rarity",
    "candidate_first_name_count_min_rarity",
    "candidate_last_first_name_count_min_rarity",
    "candidate_last_name_count_min_rarity",
    "candidate_last_first_initial_count_min_rarity",
)


def normalize_orcid(value: Any) -> str | None:
    """Return a canonical ORCID value, or None."""

    return normalize_orcid_text(value)


@dataclass(frozen=True)
class QueryFeatures:
    """Retrieval query features consumed by the Rust hybrid centroid retriever."""

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
    title_terms: frozenset[str] = EMPTY_STRING_SET
    name_counts: Any | None = None
    paper_author_count: int = 0
    paper_author_names: frozenset[str] = EMPTY_STRING_SET
    author_position: int | None = None
    local10_author_names: frozenset[str] = EMPTY_STRING_SET
    signature_id: str = ""
    query_author: str = ""


@dataclass(frozen=True)
class ClusterSummary:
    """Seed-cluster summary consumed by the Rust hybrid centroid retriever."""

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
    title_counts: Counter[str] = field(default_factory=Counter)
    name_counts_values: tuple[Any, ...] = field(default_factory=tuple)
    non_mega_coauthor_counts: Counter[str] = field(default_factory=Counter)
    max_paper_author_count: int = 0
    member_paper_author_names: tuple[frozenset[str], ...] = ()
    member_paper_author_counts: tuple[int, ...] = ()
    member_author_positions: tuple[int | None, ...] = ()
    member_local10_author_names: tuple[frozenset[str], ...] = ()
    member_signature_ids: tuple[str, ...] = ()
    member_title_terms: tuple[frozenset[str], ...] = ()


@dataclass(frozen=True)
class RustHybridCentroidRetrieverHandle:
    """Rust retriever plus Python summary lookup for row-signal completion."""

    retriever: Any
    summary_by_component: dict[str, ClusterSummary]


@dataclass(frozen=True)
class IncrementalLinkerInputs:
    """Inputs needed by the private production link-or-abstain runtime."""

    queries: tuple[QueryFeatures, ...]
    query_by_signature_id: dict[str, QueryFeatures]
    query_views: tuple[QueryView, ...]
    query_view_by_signature_id: dict[str, QueryView]
    retriever: RustHybridCentroidRetrieverHandle
    summary_by_component: dict[str, ClusterSummary]


def _normalize_term_set(value: Any, *, is_title: bool = False) -> frozenset[str]:
    normalizer = normalize_title if is_title else normalize_text
    normalized = normalizer(str(value or ""))
    if not normalized:
        return EMPTY_STRING_SET
    return frozenset(token for token in normalized.split() if token)


def _nonempty_feature_values(values: Sequence[str] | None) -> frozenset[str]:
    if not values:
        return EMPTY_STRING_SET
    return frozenset(str(value) for value in values if str(value or ""))


def _normalized_author_records(authors: Any) -> tuple[tuple[int, str], ...]:
    if not authors:
        return ()
    records: list[tuple[int, int, str]] = []
    for index, author in enumerate(authors):
        raw_position = getattr(author, "position", index)
        raw_name = getattr(author, "author_name", None)
        if isinstance(author, Mapping):
            raw_position = author.get("position", index)
            raw_name = author.get("author_name") or author.get("name")
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            position = index
        normalized = normalize_text(str(raw_name or ""))
        records.append((position, index, normalized))
    records.sort(key=lambda item: (item[0], item[1]))
    return tuple((position, normalized) for position, _index, normalized in records)


def _normalized_author_name_set(authors: Any) -> frozenset[str]:
    names = {name for _position, name in _normalized_author_records(authors) if name}
    return frozenset(names)


def _local_author_name_set(authors: Any, center_position: int | None, *, radius: int) -> frozenset[str]:
    if center_position is None:
        return EMPTY_STRING_SET
    return frozenset(
        name
        for position, name in _normalized_author_records(authors)
        if name and position != center_position and abs(position - center_position) <= radius
    )


def _signature_author_position(signature: Any) -> int | None:
    raw_position = getattr(signature, "author_info_position", None)
    if raw_position is None:
        return None
    try:
        return int(raw_position)
    except (TypeError, ValueError):
        return None


def _safe_compute_block(name: str) -> str:
    normalized_name = normalize_text(name or "")
    if not normalized_name:
        return ""
    return compute_block(normalized_name)


def _signature_coauthor_blocks(signature: Any, dataset: ANDData) -> frozenset[str]:
    coauthor_blocks = signature.author_info_coauthor_blocks
    if coauthor_blocks is not None:
        return _nonempty_feature_values(coauthor_blocks)

    coauthors = signature.author_info_coauthors
    if coauthors is None:
        author_position = _signature_author_position(signature)
        if author_position is None:
            return EMPTY_STRING_SET
        paper = dataset.papers.get(str(signature.paper_id))
        if paper is None:
            return EMPTY_STRING_SET
        coauthors = []
        for author in paper.authors:
            raw_position = author.get("position") if isinstance(author, Mapping) else getattr(author, "position", None)
            if raw_position is None:
                position = None
            else:
                try:
                    position = int(raw_position)
                except (TypeError, ValueError):
                    position = None
            if position != author_position:
                raw_name = (
                    author.get("author_name") if isinstance(author, Mapping) else getattr(author, "author_name", None)
                )
                coauthors.append(raw_name)
    return _nonempty_feature_values([_safe_compute_block(str(author or "")) for author in coauthors])


def _get_specter_vector(dataset: ANDData, paper_id: Any) -> np.ndarray | None:
    if dataset.specter_embeddings is None:
        return None
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


def _signature_query_author(signature: Any) -> str:
    """Return canonical author text for query-level gate features."""

    stored_first = getattr(signature, "author_info_first_normalized_without_apostrophe", None)
    stored_middle = getattr(signature, "author_info_middle_normalized_without_apostrophe", None)
    stored_last = getattr(signature, "author_info_last_normalized", None)
    if stored_first is None or stored_middle is None or stored_last is None:
        canonical = canonicalize_name_parts(
            getattr(signature, "author_info_first", None),
            getattr(signature, "author_info_middle", None),
            getattr(signature, "author_info_last", None),
        )
    else:
        canonical = (stored_first, stored_middle, stored_last)
    stored_suffix = getattr(signature, "author_info_suffix_normalized", None)
    suffix = normalize_text(getattr(signature, "author_info_suffix", None)) if stored_suffix is None else stored_suffix
    return " ".join(part for part in (*canonical, suffix) if part)


def extract_query_features(
    dataset: ANDData,
    signature_id: str,
    *,
    feature_cache: dict[str, QueryFeatures] | None = None,
    paper_author_name_cache: dict[str, frozenset[str]] | None = None,
    orcid_enabled: bool = False,
) -> QueryFeatures:
    """Extract production retrieval features for one signature."""

    if feature_cache is not None and signature_id in feature_cache:
        features = feature_cache[signature_id]
    else:
        signature = dataset.signatures[signature_id]
        first, middle = signature_name_parts_for_subblocking(signature)
        coauthor_blocks = _signature_coauthor_blocks(signature, dataset)
        affiliation_terms = _nonempty_feature_values(signature_affiliation_feature_keys(signature))
        paper = dataset.papers.get(str(signature.paper_id))
        venue_terms = EMPTY_STRING_SET
        title_terms = EMPTY_STRING_SET
        year = None
        paper_author_count = 0
        paper_author_names = EMPTY_STRING_SET
        local10_author_names = EMPTY_STRING_SET
        author_position = _signature_author_position(signature)
        if paper is not None:
            venue_terms = _normalize_term_set(" ".join(part for part in [paper.venue, paper.journal_name] if part))
            title_terms = _normalize_term_set(getattr(paper, "title", None), is_title=True)
            year = paper.year
            authors = getattr(paper, "authors", None)
            paper_author_count = len(authors) if authors is not None else 0
            local10_author_names = _local_author_name_set(authors, author_position, radius=10)
            if paper_author_name_cache is not None and str(signature.paper_id) in paper_author_name_cache:
                paper_author_names = paper_author_name_cache[str(signature.paper_id)]
            else:
                paper_author_names = _normalized_author_name_set(authors)
                if paper_author_name_cache is not None:
                    paper_author_name_cache[str(signature.paper_id)] = paper_author_names
        specter = _get_specter_vector(dataset, signature.paper_id)
        middle_tokens = [token for token in middle.split() if token]
        features = QueryFeatures(
            first=first,
            middle=middle,
            first_initial=first[:1],
            middle_initials=frozenset(token[0] for token in middle_tokens),
            coauthor_blocks=coauthor_blocks,
            affiliation_terms=affiliation_terms,
            venue_terms=venue_terms,
            year=year,
            orcid=normalize_orcid(signature.author_info_orcid),
            specter=specter,
            has_specter=specter is not None,
            has_coauthors=bool(coauthor_blocks),
            has_affiliations=bool(affiliation_terms),
            has_full_first=len(first) > 1,
            has_middle=bool(middle_tokens),
            title_terms=title_terms,
            name_counts=getattr(signature, "author_info_name_counts", None),
            paper_author_count=int(paper_author_count),
            paper_author_names=paper_author_names,
            author_position=author_position,
            local10_author_names=local10_author_names,
            signature_id=str(signature_id),
            query_author=_signature_query_author(signature),
        )
        if feature_cache is not None:
            feature_cache[signature_id] = features

    if orcid_enabled or features.orcid is None:
        return features
    return replace(features, orcid=None)


def mask_query_features(base: QueryFeatures, view: QueryView, *, orcid_enabled: bool = False) -> QueryFeatures:
    """Apply the promoted retrieval query-view policy."""

    if view == "full":
        return base if orcid_enabled else replace(base, orcid=None)

    first = base.first_initial
    masked = QueryFeatures(
        first=first,
        middle="",
        first_initial=base.first_initial,
        middle_initials=EMPTY_STRING_SET,
        coauthor_blocks=base.coauthor_blocks,
        affiliation_terms=base.affiliation_terms,
        venue_terms=base.venue_terms,
        year=base.year,
        orcid=base.orcid if orcid_enabled else None,
        specter=base.specter,
        has_specter=base.has_specter,
        has_coauthors=base.has_coauthors,
        has_affiliations=base.has_affiliations,
        has_full_first=False,
        has_middle=False,
        title_terms=base.title_terms,
        name_counts=base.name_counts,
        paper_author_count=base.paper_author_count,
        paper_author_names=base.paper_author_names,
        author_position=base.author_position,
        local10_author_names=base.local10_author_names,
        signature_id=base.signature_id,
        query_author=base.query_author,
    )
    if view == "initial_only":
        return masked
    raise ValueError(f"Unknown query view: {view}")


def query_view_for_features(base: QueryFeatures) -> QueryView:
    """Return the production query view implied by extracted query evidence."""

    return "full" if bool(base.has_full_first) else "initial_only"


def _resolve_query_views(
    query_signature_ids: Sequence[str],
    base_query_by_signature_id: Mapping[str, QueryFeatures],
    query_view: str | Sequence[str] | None,
) -> tuple[QueryView, ...]:
    if query_view is None:
        return tuple(
            query_view_for_features(base_query_by_signature_id[str(signature_id)])
            for signature_id in query_signature_ids
        )
    normalized = normalize_query_views(query_view, len(query_signature_ids))
    if isinstance(normalized, str):
        return tuple(normalized for _signature_id in query_signature_ids)
    return normalized


def _safe_mean(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values)) / float(len(values))


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


def _cluster_summary_from_feature_rows(
    *,
    cluster_id: str,
    component_key: str,
    feature_rows: Iterable[tuple[str, QueryFeatures]],
    max_exemplars: int,
    block_key: str,
) -> ClusterSummary:
    """Build one seed-cluster summary from pre-extracted member features."""

    first_name_counts: Counter[str] = Counter()
    middle_initial_counts: Counter[str] = Counter()
    coauthor_counts: Counter[str] = Counter()
    non_mega_coauthor_counts: Counter[str] = Counter()
    affiliation_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()
    year_values: list[int] = []
    orcid_values: set[str] = set()
    specter_vectors: list[np.ndarray] = []
    name_counts_values: list[Any] = []
    paper_author_counts: list[int] = []
    member_paper_author_names: list[frozenset[str]] = []
    member_paper_author_counts: list[int] = []
    member_author_positions: list[int | None] = []
    member_local10_author_names: list[frozenset[str]] = []
    member_signature_ids: list[str] = []
    member_title_terms: list[frozenset[str]] = []

    for signature_id, features in feature_rows:
        if len(features.first) > 1:
            first_name_counts[features.first] += 1
        for initial in features.middle_initials:
            middle_initial_counts[initial] += 1
        for block in features.coauthor_blocks:
            coauthor_counts[block] += 1
            if int(features.paper_author_count) < 50:
                non_mega_coauthor_counts[block] += 1
        for term in features.affiliation_terms:
            affiliation_counts[term] += 1
        for term in features.venue_terms:
            venue_counts[term] += 1
        for term in features.title_terms:
            title_counts[term] += 1
        if features.year is not None:
            year_values.append(int(features.year))
        if features.orcid is not None:
            orcid_values.add(features.orcid)
        if features.specter is not None:
            specter_vectors.append(features.specter)
        if features.name_counts is not None:
            name_counts_values.append(features.name_counts)
        paper_author_counts.append(int(features.paper_author_count))
        member_paper_author_names.append(features.paper_author_names)
        member_paper_author_counts.append(int(features.paper_author_count))
        member_author_positions.append(features.author_position)
        member_local10_author_names.append(features.local10_author_names)
        member_signature_ids.append(str(signature_id))
        member_title_terms.append(features.title_terms)

    centroid = None
    if specter_vectors:
        centroid = np.mean(np.vstack(specter_vectors), axis=0).astype(np.float32)

    return ClusterSummary(
        component_key=component_key,
        cluster_id=cluster_id,
        block_key=str(block_key),
        size=len(member_signature_ids),
        first_name_counts=first_name_counts,
        middle_initial_counts=middle_initial_counts,
        coauthor_counts=coauthor_counts,
        non_mega_coauthor_counts=non_mega_coauthor_counts,
        affiliation_counts=affiliation_counts,
        venue_counts=venue_counts,
        year_values=year_values,
        year_min=min(year_values) if year_values else None,
        year_max=max(year_values) if year_values else None,
        year_mean=_safe_mean(year_values),
        orcid_values=frozenset(orcid_values),
        specter_centroid=centroid,
        exemplar_vectors=_select_exemplars(specter_vectors, max_exemplars=max_exemplars),
        title_counts=title_counts,
        name_counts_values=tuple(name_counts_values),
        max_paper_author_count=max(paper_author_counts) if paper_author_counts else 0,
        member_paper_author_names=tuple(member_paper_author_names),
        member_paper_author_counts=tuple(member_paper_author_counts),
        member_author_positions=tuple(member_author_positions),
        member_local10_author_names=tuple(member_local10_author_names),
        member_signature_ids=tuple(member_signature_ids),
        member_title_terms=tuple(member_title_terms),
    )


def build_cluster_summary(
    dataset: ANDData,
    *,
    cluster_id: str,
    component_key: str,
    signature_ids: Sequence[str],
    max_exemplars: int,
    feature_cache: dict[str, QueryFeatures] | None = None,
    paper_author_name_cache: dict[str, frozenset[str]] | None = None,
    orcid_enabled: bool = False,
    block_key: str = "incremental",
) -> ClusterSummary:
    """Build one seed-cluster summary for Rust retrieval."""

    feature_rows = (
        (
            str(signature_id),
            extract_query_features(
                dataset,
                str(signature_id),
                feature_cache=feature_cache,
                paper_author_name_cache=paper_author_name_cache,
                orcid_enabled=orcid_enabled,
            ),
        )
        for signature_id in signature_ids
    )
    return _cluster_summary_from_feature_rows(
        cluster_id=cluster_id,
        component_key=component_key,
        feature_rows=feature_rows,
        max_exemplars=max_exemplars,
        block_key=block_key,
    )


def raw_paper_evidence_features(query: QueryFeatures, summary: ClusterSummary) -> dict[str, float]:
    """Return member-level raw paper evidence for giant-paper candidate rows."""

    query_author_names = query.paper_author_names
    query_local10_names = query.local10_author_names
    query_author_count = int(query.paper_author_count)
    query_signature_id = str(getattr(query, "signature_id", "") or "")
    best_author_jaccard = 0.0
    best_author_containment = 0.0
    best_author_overlap = 0.0
    best_local10_jaccard = 0.0
    best_local10_overlap_count = 0.0
    best_author_count_log_absdiff: float | None = None
    member_local10_author_names = summary.member_local10_author_names or (
        (EMPTY_STRING_SET,) * len(summary.member_paper_author_names)
    )
    member_signature_ids = summary.member_signature_ids or (("",) * len(summary.member_paper_author_names))

    for (
        candidate_names,
        candidate_count,
        candidate_local10_names,
        candidate_signature_id,
    ) in zip(
        summary.member_paper_author_names,
        summary.member_paper_author_counts,
        member_local10_author_names,
        member_signature_ids,
        strict=True,
    ):
        same_signature = query_signature_id and query_signature_id == str(candidate_signature_id)
        intersection = len(query_author_names & candidate_names)
        union = len(query_author_names | candidate_names)
        jaccard = float(intersection / union) if union else 0.0
        denominator = min(len(query_author_names), len(candidate_names))
        containment = float(intersection / denominator) if denominator else 0.0
        best_author_jaccard = max(best_author_jaccard, jaccard)
        best_author_containment = max(best_author_containment, containment)
        best_author_overlap = max(best_author_overlap, float(intersection))

        count_delta = abs(math.log1p(query_author_count) - math.log1p(int(candidate_count)))
        best_author_count_log_absdiff = (
            count_delta if best_author_count_log_absdiff is None else min(best_author_count_log_absdiff, count_delta)
        )

        # Exclude the query's own membership row from the local10-window features
        # only. The paper-author-list features above intentionally include the
        # self-match (see test_local10_evidence_ignores_query_signature_member);
        # extending the skip to them is a speculative guard for a caller that
        # does not exist -- production/training already drop the query upstream.
        if same_signature:
            continue
        local10_intersection = len(query_local10_names & candidate_local10_names)
        local10_union = len(query_local10_names | candidate_local10_names)
        if local10_union:
            best_local10_jaccard = max(best_local10_jaccard, float(local10_intersection / local10_union))
        best_local10_overlap_count = max(best_local10_overlap_count, float(local10_intersection))

    return {
        "paper_author_list_max_jaccard": round(best_author_jaccard, 6),
        "paper_author_list_max_containment": round(best_author_containment, 6),
        "paper_author_list_max_overlap_count": round(best_author_overlap, 6),
        "local_author_window10_jaccard_max": round(best_local10_jaccard, 6),
        "local_author_window10_overlap_count_max": round(best_local10_overlap_count, 6),
        "best_author_count_log_absdiff": round(float(best_author_count_log_absdiff or 0.0), 6),
    }


def build_rust_hybrid_centroid_retriever(
    candidate_summaries: Sequence[ClusterSummary],
    *,
    include_exemplars: bool = True,
) -> RustHybridCentroidRetrieverHandle:
    """Build the Rust hybrid centroid retriever for promoted linker candidates."""

    summaries = list(candidate_summaries)
    return RustHybridCentroidRetrieverHandle(
        retriever=s2and_rust.RustHybridCentroidRetriever(
            summaries,
            include_exemplars=bool(include_exemplars),
        ),
        summary_by_component={str(summary.component_key): summary for summary in summaries},
    )


def _seed_members_by_cluster(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, list[str]]:
    members_by_cluster: dict[str, list[str]] = {}
    for signature_id, cluster_id in cluster_seeds_require.items():
        members_by_cluster.setdefault(str(cluster_id), []).append(str(signature_id))
    return members_by_cluster


def build_incremental_linker_inputs(
    *,
    dataset: ANDData,
    query_signature_ids: Sequence[str],
    cluster_seeds_require: Mapping[str, int | str],
    query_view: str | Sequence[str] | None = "initial_only",
    max_exemplars: int = 4,
    orcid_enabled: bool = False,
) -> IncrementalLinkerInputs:
    """Build queries and the seed-cluster retriever for private incremental linking.

    `orcid_enabled` is disabled by default for labeled calibration/evaluation
    callers. Production promoted linking passes it explicitly as enabled.
    """

    feature_cache: dict[str, QueryFeatures] = {}
    paper_author_name_cache: dict[str, frozenset[str]] = {}
    base_query_by_signature_id = {
        str(signature_id): extract_query_features(
            dataset,
            str(signature_id),
            feature_cache=feature_cache,
            paper_author_name_cache=paper_author_name_cache,
            orcid_enabled=orcid_enabled,
        )
        for signature_id in query_signature_ids
    }
    query_views = _resolve_query_views(query_signature_ids, base_query_by_signature_id, query_view)
    query_view_by_signature_id = {
        str(signature_id): current_query_view
        for signature_id, current_query_view in zip(query_signature_ids, query_views, strict=True)
    }
    query_by_signature_id = {
        str(signature_id): mask_query_features(
            base_query_by_signature_id[str(signature_id)],
            query_view_by_signature_id[str(signature_id)],
            orcid_enabled=orcid_enabled,
        )
        for signature_id in query_signature_ids
    }
    summaries = [
        build_cluster_summary(
            dataset,
            cluster_id=component_key,
            component_key=component_key,
            signature_ids=signature_ids,
            max_exemplars=max_exemplars,
            feature_cache=feature_cache,
            paper_author_name_cache=paper_author_name_cache,
            orcid_enabled=orcid_enabled,
        )
        for component_key, signature_ids in _seed_members_by_cluster(cluster_seeds_require).items()
    ]
    retriever = build_rust_hybrid_centroid_retriever(summaries, include_exemplars=max_exemplars > 0)
    return IncrementalLinkerInputs(
        queries=tuple(query_by_signature_id[str(signature_id)] for signature_id in query_signature_ids),
        query_by_signature_id=query_by_signature_id,
        query_views=query_views,
        query_view_by_signature_id=query_view_by_signature_id,
        retriever=retriever,
        summary_by_component=retriever.summary_by_component,
    )


def _name_count_rarity(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        count = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(count) or count <= 0.0:
        return 0.0
    return float(1.0 / math.sqrt(count))


def _name_count_attr(value: Any, field_name: str) -> float | None:
    raw_value = getattr(value, field_name, None)
    if raw_value is None:
        return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _candidate_name_count_rarity_features(summary: ClusterSummary) -> dict[str, float]:
    minima: dict[str, float] = {}
    for candidate_name_counts in tuple(getattr(summary, "name_counts_values", ()) or ()):
        for field_name in ("first", "first_last", "last", "last_first_initial"):
            value = _name_count_attr(candidate_name_counts, field_name)
            if value is None:
                continue
            minima[field_name] = min(value, minima.get(field_name, value))

    return {
        "candidate_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first")), 6),
        "candidate_last_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first_last")), 6),
        "candidate_last_name_count_min_rarity": round(_name_count_rarity(minima.get("last")), 6),
        "candidate_last_first_initial_count_min_rarity": round(
            _name_count_rarity(minima.get("last_first_initial")),
            6,
        ),
    }


def name_count_rarity_features(query: QueryFeatures, summary: ClusterSummary) -> dict[str, float]:
    candidate_features = _candidate_name_count_rarity_features(summary)
    query_name_counts = getattr(query, "name_counts", None)
    candidate_name_counts_values = tuple(getattr(summary, "name_counts_values", ()) or ())
    if query_name_counts is None or not candidate_name_counts_values:
        return {
            **{column: 0.0 for column in NAME_COUNT_RARITY_FEATURE_COLUMNS if column not in candidate_features},
            **candidate_features,
        }

    observed_minima: dict[str, float] = {}
    for candidate_name_counts in candidate_name_counts_values:
        if candidate_name_counts is None:
            continue
        values = pairwise_name_counts(query_name_counts, candidate_name_counts)
        for feature_name, raw_value in zip(PAIRWISE_NAME_COUNT_FEATURE_NAMES, values, strict=True):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            observed_minima[feature_name] = min(value, observed_minima.get(feature_name, value))

    features = {
        f"{feature_name}_rarity": round(_name_count_rarity(observed_minima.get(feature_name)), 6)
        for feature_name in PAIRWISE_NAME_COUNT_FEATURE_NAMES
    }
    if not bool(query.has_full_first):
        for column in (
            "first_name_count_min_rarity",
            "last_first_name_count_min_rarity",
            "last_first_initial_count_min_rarity",
            "first_name_count_max_rarity",
            "last_first_name_count_max_rarity",
        ):
            features[column] = 0.0
    first_prefix_match = 0.0
    query_first = str(query.first or "")
    if len(query_first) > 1 and int(summary.size) > 0:
        for candidate_first, count in summary.first_name_counts.items():
            if len(candidate_first) > 1 and same_prefix_tokens(query_first, candidate_first):
                first_prefix_match = max(first_prefix_match, float(count) / float(summary.size))
    features["first_prefix_x_last_first_name_count_min_rarity"] = round(
        float(first_prefix_match) * float(features["last_first_name_count_min_rarity"]),
        6,
    )
    return {**features, **candidate_features}


def build_name_count_rarity_row_signals(
    retrieval_batch: LinkerRetrievalBatch,
    *,
    query_signature_id_by_index: Mapping[int, str],
    query_by_signature_id: Mapping[str, QueryFeatures],
    summary_by_component: Mapping[str, ClusterSummary],
) -> dict[str, np.ndarray]:
    """Build name-count rarity row signals for retrieved candidate rows."""

    candidate_batch = retrieval_batch.candidate_batch
    row_count = int(candidate_batch.row_count)
    query_indices = candidate_batch.row_query_signature_indices
    component_keys = candidate_batch.row_component_keys
    if query_indices is None or component_keys is None:
        raise ValueError("retrieval batch must include row query indices and component keys")

    signals = {
        "last_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_last_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "first_prefix_x_last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_cluster_max_paper_author_count": np.zeros(row_count, dtype=np.float32),
        "paper_author_list_max_jaccard": np.zeros(row_count, dtype=np.float32),
        "paper_author_list_max_containment": np.zeros(row_count, dtype=np.float32),
        "paper_author_list_max_overlap_count": np.zeros(row_count, dtype=np.float32),
        "local_author_window10_jaccard_max": np.zeros(row_count, dtype=np.float32),
        "local_author_window10_overlap_count_max": np.zeros(row_count, dtype=np.float32),
        "best_author_count_log_absdiff": np.zeros(row_count, dtype=np.float32),
    }
    for row_index, (query_index, component_key) in enumerate(zip(query_indices, component_keys, strict=True)):
        query_signature_id = query_signature_id_by_index.get(int(query_index))
        if query_signature_id is None:
            raise KeyError(f"Missing query signature id for index {int(query_index)}")
        query = query_by_signature_id[str(query_signature_id)]
        summary = summary_by_component[str(component_key)]
        rarity = name_count_rarity_features(query, summary)
        for signal_name in (
            "last_name_count_min_rarity",
            "candidate_last_name_count_min_rarity",
            "candidate_last_first_name_count_min_rarity",
            "last_first_name_count_min_rarity",
            "first_prefix_x_last_first_name_count_min_rarity",
        ):
            signals[signal_name][row_index] = float(rarity.get(signal_name, 0.0) or 0.0)
        signals["candidate_cluster_max_paper_author_count"][row_index] = float(summary.max_paper_author_count)
        for signal_name, value in raw_paper_evidence_features(query, summary).items():
            signals[signal_name][row_index] = float(value)
    return signals
