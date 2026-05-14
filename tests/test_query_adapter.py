from __future__ import annotations

from types import SimpleNamespace

import pytest

from s2and.incremental_linking.query_adapter import (
    build_cluster_summary,
    extract_query_features,
    raw_paper_evidence_features,
)
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


def test_cluster_summary_tracks_non_mega_coauthors_separately() -> None:
    small_signature = SimpleNamespace(
        **{
            **_signature("p_small").__dict__,
            "author_info_coauthor_blocks": ["shared coauthor", "small only"],
        }
    )
    mega_signature = SimpleNamespace(
        **{
            **_signature("p_mega").__dict__,
            "author_info_coauthor_blocks": ["shared coauthor", "mega only"],
        }
    )
    dataset = SimpleNamespace(
        signatures={"small": small_signature, "mega": mega_signature},
        papers={
            "p_small": SimpleNamespace(
                title="Small Paper",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[SimpleNamespace(author_name="Alice"), SimpleNamespace(author_name="Bob")],
            ),
            "p_mega": SimpleNamespace(
                title="Mega Paper",
                venue=None,
                journal_name=None,
                year=2021,
                authors=[SimpleNamespace(author_name=f"Author {index}") for index in range(50)],
            ),
        },
        specter_embeddings=None,
    )

    summary = build_cluster_summary(
        dataset,
        cluster_id="cluster",
        component_key="component",
        signature_ids=("small", "mega"),
        max_exemplars=4,
        feature_cache={},
        orcid_enabled=False,
        block_key="block",
    )

    assert summary.max_paper_author_count == 50
    assert summary.coauthor_counts["shared coauthor"] == 2
    assert summary.coauthor_counts["mega only"] == 1
    assert summary.non_mega_coauthor_counts["shared coauthor"] == 1
    assert summary.non_mega_coauthor_counts["small only"] == 1
    assert "mega only" not in summary.non_mega_coauthor_counts


def test_raw_paper_evidence_features_use_author_lists_and_local_windows() -> None:
    dataset = SimpleNamespace(
        signatures={
            "q": _signature("pq"),
            "c_match": SimpleNamespace(**{**_signature("pc_match").__dict__, "author_info_position": 1}),
            "c_other": SimpleNamespace(**{**_signature("pc_other").__dict__, "author_info_position": 0}),
        },
        papers={
            "pq": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_match": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_other": SimpleNamespace(
                title="Different Topic",
                venue=None,
                journal_name=None,
                year=2021,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Dana Kim", position=1),
                ],
            ),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        dataset,
        cluster_id="cluster",
        component_key="component",
        signature_ids=("c_other", "c_match"),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    features = raw_paper_evidence_features(query, summary)

    assert query.paper_author_names == frozenset({"alice smith", "bob jones", "carol lee"})
    assert query.author_position == 0
    assert query.local10_author_names == frozenset({"bob jones", "carol lee"})
    assert features["paper_author_list_max_jaccard"] == pytest.approx(1.0)
    assert features["paper_author_list_max_containment"] == pytest.approx(1.0)
    assert features["paper_author_list_max_overlap_count"] == pytest.approx(3.0)
    assert features["local_author_window10_jaccard_max"] == pytest.approx(1.0 / 3.0)
    assert features["local_author_window10_overlap_count_max"] == pytest.approx(1.0)
    assert features["best_author_count_log_absdiff"] == pytest.approx(0.0)


def test_local10_evidence_ignores_query_signature_member() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq")},
        papers={
            "pq": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        dataset,
        cluster_id="cluster",
        component_key="component",
        signature_ids=("q",),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    features = raw_paper_evidence_features(query, summary)

    assert features["paper_author_list_max_jaccard"] == pytest.approx(1.0)
    assert features["local_author_window10_jaccard_max"] == pytest.approx(0.0)
    assert features["local_author_window10_overlap_count_max"] == pytest.approx(0.0)
