"""Raw metadata similarity features shared by reranker dataset builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from s2and.text import normalize_text

try:
    from scripts.single_letter_reranker_utils import RAW_METADATA_SIMILARITY_FEATURE_COLUMNS
except ImportError:  # pragma: no cover - direct script execution path
    from single_letter_reranker_utils import RAW_METADATA_SIMILARITY_FEATURE_COLUMNS  # type: ignore

RAW_TEXT_STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "using",
    "based",
    "into",
    "over",
    "under",
    "between",
    "without",
    "within",
    "study",
    "analysis",
    "approach",
    "method",
    "methods",
    "result",
    "results",
    "paper",
    "system",
    "systems",
}


@dataclass
class RawSimilarityFeatureCache:
    """Token caches used while generating raw metadata similarity features."""

    affiliation_tokens_by_signature_id: dict[str, frozenset[str]] = field(default_factory=dict)
    coauthor_names_by_paper_and_last: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict)
    title_tokens_by_paper_id: dict[str, frozenset[str]] = field(default_factory=dict)
    text_tokens_by_paper_id: dict[str, frozenset[str]] = field(default_factory=dict)


def _raw_tokens(value: Any) -> frozenset[str]:
    normalized = normalize_text(str(value or ""))
    if not normalized:
        return frozenset()
    return frozenset(token for token in normalized.split() if len(token) >= 3 and token not in RAW_TEXT_STOPWORDS)


def _raw_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _signature_last_token(signature: Any) -> str:
    tokens = normalize_text(str(getattr(signature, "author_info_last", "") or "")).split()
    return str(tokens[-1]) if tokens else ""


def _cached_signature_affiliation_tokens(
    cache: RawSimilarityFeatureCache,
    *,
    signature_id: str,
    signature: Any,
) -> frozenset[str]:
    cached = cache.affiliation_tokens_by_signature_id.get(str(signature_id))
    if cached is not None:
        return cached
    affiliations = getattr(signature, "author_info_affiliations", None) or []
    tokens = _raw_tokens(" ".join(str(value) for value in affiliations if value))
    cache.affiliation_tokens_by_signature_id[str(signature_id)] = tokens
    return tokens


def _cached_paper_author_name_set(
    cache: RawSimilarityFeatureCache,
    paper: Any,
    *,
    excluded_last_token: str,
) -> frozenset[str]:
    if paper is None:
        return frozenset()
    paper_id = str(getattr(paper, "paper_id", "") or "")
    key = (paper_id, str(excluded_last_token))
    cached = cache.coauthor_names_by_paper_and_last.get(key)
    if cached is not None:
        return cached
    names: set[str] = set()
    for author in getattr(paper, "authors", []) or []:
        name = normalize_text(str(getattr(author, "author_name", "") or ""))
        if not name:
            continue
        name_tokens = name.split()
        if excluded_last_token and excluded_last_token in name_tokens:
            continue
        names.add(" ".join(name_tokens))
    result = frozenset(names)
    if paper_id:
        cache.coauthor_names_by_paper_and_last[key] = result
    return result


def _cached_paper_title_tokens(cache: RawSimilarityFeatureCache, paper: Any) -> frozenset[str]:
    if paper is None:
        return frozenset()
    paper_id = str(getattr(paper, "paper_id", "") or "")
    cached = cache.title_tokens_by_paper_id.get(paper_id)
    if cached is not None:
        return cached
    tokens = _raw_tokens(getattr(paper, "title", None))
    if paper_id:
        cache.title_tokens_by_paper_id[paper_id] = tokens
    return tokens


def _cached_paper_text_tokens(
    cache: RawSimilarityFeatureCache,
    paper: Any,
    *,
    raw_paper_text_by_id: dict[str, str],
) -> frozenset[str]:
    if paper is None:
        return frozenset()
    paper_id = str(getattr(paper, "paper_id", "") or "")
    cached = cache.text_tokens_by_paper_id.get(paper_id)
    if cached is not None:
        return cached
    tokens = _raw_tokens(raw_paper_text_by_id.get(paper_id, getattr(paper, "title", "") or ""))
    if paper_id:
        cache.text_tokens_by_paper_id[paper_id] = tokens
    return tokens


def raw_similarity_feature_zeros() -> dict[str, float]:
    """Return an explicit all-zero raw metadata similarity feature payload."""

    return {feature_name: 0.0 for feature_name in RAW_METADATA_SIMILARITY_FEATURE_COLUMNS}


def raw_similarity_features_by_component(
    *,
    dataset: Any,
    query_signature_id: str,
    candidate_signature_ids_by_component: dict[str, list[str]],
    raw_paper_text_by_id: dict[str, str] | None = None,
    cache: RawSimilarityFeatureCache | None = None,
) -> dict[str, dict[str, float]]:
    """Compute max raw-metadata similarities, excluding the query signature from candidates."""

    cache = cache or RawSimilarityFeatureCache()
    raw_paper_text_by_id = raw_paper_text_by_id or {}
    query_signature = dataset.signatures.get(str(query_signature_id))
    if query_signature is None:
        return {component_key: raw_similarity_feature_zeros() for component_key in candidate_signature_ids_by_component}
    query_paper = dataset.papers.get(str(query_signature.paper_id))
    query_last = _signature_last_token(query_signature)
    query_affiliation_tokens = _cached_signature_affiliation_tokens(
        cache,
        signature_id=str(query_signature_id),
        signature=query_signature,
    )
    query_coauthor_names = _cached_paper_author_name_set(cache, query_paper, excluded_last_token=query_last)
    query_title_tokens = _cached_paper_title_tokens(cache, query_paper)
    query_text_tokens = _cached_paper_text_tokens(cache, query_paper, raw_paper_text_by_id=raw_paper_text_by_id)

    features_by_component: dict[str, dict[str, float]] = {}
    for component_key, candidate_signature_ids in candidate_signature_ids_by_component.items():
        max_affiliation = 0.0
        max_coauthor = 0.0
        max_title = 0.0
        max_text = 0.0
        for signature_id in candidate_signature_ids:
            if str(signature_id) == str(query_signature_id):
                continue
            candidate_signature = dataset.signatures.get(str(signature_id))
            if candidate_signature is None:
                continue
            candidate_paper = dataset.papers.get(str(candidate_signature.paper_id))
            max_affiliation = max(
                max_affiliation,
                _raw_jaccard(
                    query_affiliation_tokens,
                    _cached_signature_affiliation_tokens(
                        cache,
                        signature_id=str(signature_id),
                        signature=candidate_signature,
                    ),
                ),
            )
            max_coauthor = max(
                max_coauthor,
                _raw_jaccard(
                    query_coauthor_names,
                    _cached_paper_author_name_set(cache, candidate_paper, excluded_last_token=query_last),
                ),
            )
            max_title = max(
                max_title,
                _raw_jaccard(query_title_tokens, _cached_paper_title_tokens(cache, candidate_paper)),
            )
            max_text = max(
                max_text,
                _raw_jaccard(
                    query_text_tokens,
                    _cached_paper_text_tokens(cache, candidate_paper, raw_paper_text_by_id=raw_paper_text_by_id),
                ),
            )
        features_by_component[str(component_key)] = {
            "raw_max_affiliation_jaccard": round(float(max_affiliation), 6),
            "raw_max_coauthor_jaccard": round(float(max_coauthor), 6),
            "raw_max_title_jaccard": round(float(max_title), 6),
            "raw_max_text_jaccard": round(float(max_text), 6),
        }
    return features_by_component
