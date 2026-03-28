from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import scripts.single_letter_reranker_utils as reranker_utils
import scripts.single_letter_retrieval_utils as retrieval_utils
from tests.helpers import build_cluster_summary, build_query_features

s2and_rust = pytest.importorskip("s2and_rust", reason="s2and_rust is unavailable")

def test_rust_hybrid_centroid_retriever_matches_python_window() -> None:
    assert hasattr(s2and_rust, "RustHybridCentroidRetriever")

    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        year=2012,
        orcid="orcid-1",
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    matching_best = build_cluster_summary(
        component_key="c1",
        size=4,
        first_name_counts=Counter({"alice": 4}),
        middle_initial_counts=Counter({"b": 4}),
        coauthor_counts=Counter({"a smith": 3}),
        affiliation_counts=Counter({"lab": 2}),
        year_min=2010,
        year_max=2013,
        year_mean=2011.5,
        orcid_values=frozenset({"orcid-1"}),
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    matching_runner_up = build_cluster_summary(
        component_key="c2",
        size=4,
        first_name_counts=Counter({"alicia": 4}),
        middle_initial_counts=Counter({"b": 4}),
        coauthor_counts=Counter({"a smith": 1}),
        affiliation_counts=Counter({"lab": 1}),
        year_min=2000,
        year_max=2002,
        year_mean=2001.0,
        orcid_values=frozenset({"orcid-1"}),
        specter_centroid=np.asarray([0.7, 0.3], dtype=np.float32),
    )
    filtered_out = build_cluster_summary(
        component_key="c3",
        size=4,
        first_name_counts=Counter({"bob": 4}),
        middle_initial_counts=Counter({"z": 4}),
        year_min=1990,
        year_max=1992,
        year_mean=1991.0,
        orcid_values=frozenset({"orcid-2"}),
        specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    candidate_summaries = [matching_best, matching_runner_up, filtered_out]

    python_ranked, python_scores, _python_ranks, _state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_summaries,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
    )

    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(candidate_summaries)
    rust_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        num_threads=1,
    )

    assert [summary.component_key for _score, summary in rust_ranked] == python_ranked
    assert [score for score, _summary in rust_ranked] == pytest.approx(
        [python_scores[component_key] for component_key in python_ranked],
        rel=1e-5,
        abs=1e-5,
    )


def test_build_retrieval_window_rust_handle_matches_python() -> None:
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        year=2012,
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    candidate_summaries = [
        build_cluster_summary(
            component_key="c1",
            size=4,
            first_name_counts=Counter({"alice": 4}),
            middle_initial_counts=Counter({"b": 4}),
            year_min=2010,
            year_max=2013,
            year_mean=2011.5,
            specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        build_cluster_summary(
            component_key="c2",
            size=4,
            first_name_counts=Counter({"alicia": 4}),
            middle_initial_counts=Counter({"b": 4}),
            year_min=2000,
            year_max=2002,
            year_mean=2001.0,
            specter_centroid=np.asarray([0.7, 0.3], dtype=np.float32),
        ),
    ]

    python_window = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_summaries,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
    )
    rust_window = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_summaries,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
        rust_hybrid_centroid_retriever=retrieval_utils.build_rust_hybrid_centroid_retriever(candidate_summaries),
    )

    assert rust_window[0] == python_window[0]
    assert rust_window[1] == pytest.approx(python_window[1], rel=1e-5, abs=1e-5)
    assert rust_window[2] == python_window[2]
    assert rust_window[3] == python_window[3]


def test_build_retrieval_window_rust_handle_supports_single_summary_override() -> None:
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        year=2012,
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    base_summaries = [
        build_cluster_summary(
            component_key="c1",
            size=5,
            first_name_counts=Counter({"alice": 5}),
            middle_initial_counts=Counter({"b": 5}),
            year_min=2008,
            year_max=2014,
            year_mean=2011.0,
            specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        ),
        build_cluster_summary(
            component_key="c2",
            size=3,
            first_name_counts=Counter({"alicia": 3}),
            middle_initial_counts=Counter({"b": 3}),
            year_min=2000,
            year_max=2003,
            year_mean=2001.0,
            specter_centroid=np.asarray([0.7, 0.3], dtype=np.float32),
        ),
    ]
    residual_c1 = build_cluster_summary(
        component_key="c1",
        size=1,
        first_name_counts=Counter({"alice": 1}),
        middle_initial_counts=Counter({"b": 1}),
        year_min=2012,
        year_max=2012,
        year_mean=2012.0,
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    query_candidate_summaries = [residual_c1, base_summaries[1]]

    python_window = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=query_candidate_summaries,
        max_block_component_size=5,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
    )
    rust_window = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=query_candidate_summaries,
        max_block_component_size=5,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
        rust_hybrid_centroid_retriever=retrieval_utils.build_rust_hybrid_centroid_retriever(base_summaries),
    )

    assert rust_window[0] == python_window[0]
    assert rust_window[1] == pytest.approx(python_window[1], rel=1e-5, abs=1e-5)
    assert rust_window[2] == python_window[2]
    assert rust_window[3] == python_window[3]
