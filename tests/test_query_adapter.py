from __future__ import annotations

from types import SimpleNamespace

import pytest

from s2and.incremental_linking.query_adapter import build_cluster_summary, extract_query_features
from s2and.incremental_linking_training import counter_query_overlap, title_overlap


def _signature(paper_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        paper_id=paper_id,
        author_info_first="Alice",
        author_info_middle="",
        author_info_first_normalized_without_apostrophe="alice",
        author_info_middle_normalized_without_apostrophe="",
        author_info_position=0,
        author_info_coauthor_blocks=[],
        author_info_coauthors=[],
        author_info_affiliations_n_grams={},
        author_info_affiliations=[],
        author_info_orcid=None,
        author_info_name_counts=None,
    )


def test_title_and_venue_terms_keep_single_character_tokens() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq"), "c": _signature("pc")},
        papers={
            "pq": SimpleNamespace(title="A M Study", venue="Series A", journal_name=None, year=2020),
            "pc": SimpleNamespace(title="A Different Study", venue="A", journal_name=None, year=2021),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        dataset,
        cluster_id="cluster",
        component_key="component",
        signature_ids=("c",),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    assert query.title_terms == frozenset({"a", "m", "study"})
    assert query.venue_terms == frozenset({"series", "a"})
    assert summary.title_counts["a"] == 1
    assert summary.venue_counts["a"] == 1
    assert title_overlap(query, summary) == pytest.approx(2.0 / 3.0)
    assert counter_query_overlap(query.venue_terms, summary.venue_counts, summary.size) == pytest.approx(0.5)
