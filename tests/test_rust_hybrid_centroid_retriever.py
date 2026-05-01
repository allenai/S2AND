from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import scripts.eval_cluster_retrieval as retrieval
import scripts.single_letter_reranker_utils as reranker_utils
import scripts.single_letter_retrieval_utils as retrieval_utils
from tests.helpers import build_cluster_summary, build_query_features

s2and_rust = pytest.importorskip("s2and_rust", reason="s2and_rust is unavailable")

BASELINE_HYBRID_WEIGHTS = retrieval_utils.DEFAULT_RUST_HYBRID_CENTROID_WEIGHTS
RETRIEVAL_ENGINE_STATE_KEYS = {
    "retrieval_engine_rust_method_count",
    "retrieval_engine_python_method_count",
    "retrieval_engine_fallback_count",
}


def _legacy_retrieval_state(state: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in state.items() if key not in RETRIEVAL_ENGINE_STATE_KEYS}


def test_default_retrieval_scoring_constants_match_rust_exports() -> None:
    assert tuple(s2and_rust.RETRIEVAL_FEATURE_ORDER) == retrieval.HYBRID_FEATURE_ORDER
    assert tuple(s2and_rust.RETRIEVAL_FEATURE_ORDER) == retrieval_utils.RUST_HYBRID_CENTROID_FEATURE_ORDER
    assert retrieval.DEFAULT_HYBRID_CENTROID_WEIGHTS == pytest.approx(tuple(s2and_rust.DEFAULT_HYBRID_CENTROID_WEIGHTS))
    assert retrieval_utils.DEFAULT_RUST_HYBRID_CENTROID_WEIGHTS == pytest.approx(
        tuple(s2and_rust.DEFAULT_HYBRID_CENTROID_WEIGHTS)
    )
    assert retrieval.DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS == pytest.approx(
        tuple(s2and_rust.DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS)
    )
    assert retrieval.RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE == pytest.approx(
        s2and_rust.RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE
    )
    assert retrieval.RETRIEVAL_YEAR_SCORE_DECAY_YEARS == pytest.approx(s2and_rust.RETRIEVAL_YEAR_SCORE_DECAY_YEARS)
    assert retrieval.RETRIEVAL_YEAR_SCORE_RANGE_GAP == s2and_rust.RETRIEVAL_YEAR_SCORE_RANGE_GAP
    assert retrieval.RETRIEVAL_YEAR_SCORE_RANGE_PENALTY == pytest.approx(s2and_rust.RETRIEVAL_YEAR_SCORE_RANGE_PENALTY)
    assert retrieval.RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP == s2and_rust.RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP


def test_rust_name_compatible_subblock_selector_preserves_same_subblock_and_backfill() -> None:
    if not hasattr(s2and_rust, "RustNameCompatibleSubblockSelector"):
        pytest.skip("RustNameCompatibleSubblockSelector is unavailable")
    selector = s2and_rust.RustNameCompatibleSubblockSelector(
        {
            "signature_to_subblock": {"q1": "legacy_bucket"},
            "subblock_to_components": {
                "legacy_bucket": ["c1"],
                "ali_bucket": ["c2"],
                "bob_bucket": ["c3"],
                "ann_bucket": ["c4"],
            },
            "subblock_tokens_by_subblock": {
                "legacy_bucket": ["zz"],
                "ali_bucket": ["ali"],
                "bob_bucket": ["bob"],
                "ann_bucket": ["ann"],
            },
        }
    )

    selected = selector.select(
        "q1",
        "alice",
        ["c1", "c2", "c3", "c4"],
        global_backfill_count=1,
    )

    assert selected == ["c1", "c2", "c3"]


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
    assert _legacy_retrieval_state(rust_window[3]) == _legacy_retrieval_state(python_window[3])
    assert rust_window[3]["retrieval_engine_rust_method_count"] == 1
    assert python_window[3]["retrieval_engine_python_method_count"] == 1


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
    assert _legacy_retrieval_state(rust_window[3]) == _legacy_retrieval_state(python_window[3])
    assert rust_window[3]["retrieval_engine_rust_method_count"] == 1
    assert python_window[3]["retrieval_engine_python_method_count"] == 1


def test_weighted_hybrid_centroid_matches_default_weights() -> None:
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
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(candidate_summaries)

    default_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        num_threads=1,
    )
    weighted_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        num_threads=1,
        weights=BASELINE_HYBRID_WEIGHTS,
    )

    assert [summary.component_key for _score, summary in weighted_ranked] == [
        summary.component_key for _score, summary in default_ranked
    ]
    assert [score for score, _summary in weighted_ranked] == pytest.approx(
        [score for score, _summary in default_ranked],
        rel=1e-5,
        abs=1e-5,
    )


def test_weighted_hybrid_centroid_rejects_seven_weight_vector() -> None:
    query = build_query_features(first="alice")
    candidate_summaries = [build_cluster_summary(component_key="c1", size=1, first_name_counts=Counter({"alice": 1}))]
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(candidate_summaries)

    with pytest.raises(ValueError, match="Expected 5 retrieval weights"):
        retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
            query=query,
            max_ranked_clusters=1,
            retriever=retriever,
            num_threads=1,
            weights=(0.42, 0.23, 0.12, 0.05, 0.07, 0.03, 0.02),
        )


def test_rank_top_summaries_rust_hybrid_centroid_uses_runtime_thread_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features(first="alice")
    summary = build_cluster_summary(component_key="c1", size=1, first_name_counts=Counter({"alice": 1}))
    captured: dict[str, int | None] = {}

    class _FakeRustRetriever:
        def top_k_hybrid_centroid(
            self,
            _query: object,
            *,
            top_k: int,
            num_threads: int | None,
        ) -> tuple[list[str], list[float]]:
            assert top_k == 1
            captured["num_threads"] = num_threads
            return ["c1"], [0.5]

    monkeypatch.setenv("RAYON_NUM_THREADS", "7")
    handle = retrieval_utils.RustHybridCentroidRetrieverHandle(
        retriever=_FakeRustRetriever(),
        summary_by_component={"c1": summary},
    )

    ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=1,
        retriever=handle,
    )

    assert captured["num_threads"] == 7
    assert ranked == [(0.5, summary)]


def test_experimental_first_name_mode_exact_only_breaks_prefix_tie() -> None:
    query = build_query_features(first="huabin")
    candidate_summaries = [
        build_cluster_summary(component_key="a_prefix", size=4, first_name_counts=Counter({"hua": 4})),
        build_cluster_summary(component_key="b_exact", size=4, first_name_counts=Counter({"huabin": 4})),
    ]
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(candidate_summaries)

    baseline_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        component_keys=["a_prefix", "b_exact"],
        max_block_component_size=4,
        num_threads=1,
        weights=(0.0, 0.0, 0.0, 0.0, 1.0),
    )
    exact_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        component_keys=["a_prefix", "b_exact"],
        max_block_component_size=4,
        num_threads=1,
        weights=(0.0, 0.0, 0.0, 0.0, 1.0),
        scoring_config=retrieval_utils.RustHybridCentroidScoringConfig(first_name_mode="exact_only"),
    )

    assert [summary.component_key for _score, summary in baseline_ranked] == ["a_prefix", "b_exact"]
    assert [summary.component_key for _score, summary in exact_ranked] == ["b_exact", "a_prefix"]


def test_experimental_specter_mode_can_use_exemplar_vectors() -> None:
    query = build_query_features(specter=np.asarray([1.0, 0.0], dtype=np.float32))
    centroid_favorite = build_cluster_summary(
        component_key="c1",
        size=4,
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        exemplar_vectors=[np.asarray([0.0, 1.0], dtype=np.float32)],
    )
    exemplar_favorite = build_cluster_summary(
        component_key="c2",
        size=4,
        specter_centroid=np.asarray([0.6, 0.8], dtype=np.float32),
        exemplar_vectors=[np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(
        [centroid_favorite, exemplar_favorite],
        include_exemplars=True,
    )

    centroid_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        component_keys=["c1", "c2"],
        max_block_component_size=4,
        num_threads=1,
        weights=(1.0, 0.0, 0.0, 0.0, 0.0),
    )
    exemplar_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=2,
        retriever=retriever,
        component_keys=["c1", "c2"],
        max_block_component_size=4,
        num_threads=1,
        weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        scoring_config=retrieval_utils.RustHybridCentroidScoringConfig(specter_mode="exemplar_max"),
    )

    assert [summary.component_key for _score, summary in centroid_ranked] == ["c1", "c2"]
    assert [summary.component_key for _score, summary in exemplar_ranked] == ["c2", "c1"]


def test_experimental_coauthor_idf_downweights_common_overlap() -> None:
    query = build_query_features(
        coauthor_blocks=frozenset({"common", "rare"}),
        has_coauthors=True,
    )
    common_heavy = build_cluster_summary(
        component_key="common_heavy",
        size=1,
        coauthor_counts=Counter({"common": 1}),
    )
    rare_match = build_cluster_summary(
        component_key="rare_match",
        size=1,
        coauthor_counts=Counter({"rare": 1}),
    )
    common_other_1 = build_cluster_summary(
        component_key="common_other_1",
        size=4,
        coauthor_counts=Counter({"common": 4}),
    )
    common_other_2 = build_cluster_summary(
        component_key="common_other_2",
        size=4,
        coauthor_counts=Counter({"common": 3}),
    )
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever(
        [common_heavy, rare_match, common_other_1, common_other_2]
    )

    baseline_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=4,
        retriever=retriever,
        component_keys=["common_heavy", "rare_match", "common_other_1", "common_other_2"],
        max_block_component_size=4,
        num_threads=1,
        weights=(0.0, 1.0, 0.0, 0.0, 0.0),
    )
    idf_ranked = retrieval_utils.rank_top_summaries_rust_hybrid_centroid(
        query=query,
        max_ranked_clusters=4,
        retriever=retriever,
        component_keys=["common_heavy", "rare_match", "common_other_1", "common_other_2"],
        max_block_component_size=4,
        num_threads=1,
        weights=(0.0, 1.0, 0.0, 0.0, 0.0),
        scoring_config=retrieval_utils.RustHybridCentroidScoringConfig(coauthor_use_idf=True),
    )

    assert [summary.component_key for _score, summary in baseline_ranked][0] == "common_heavy"
    assert [summary.component_key for _score, summary in idf_ranked][0] == "rare_match"


def test_rust_chooser_summary_features_include_existing_and_new_fast_signals() -> None:
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        year=2012,
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
        coauthor_blocks=frozenset({"a smith"}),
        affiliation_terms=frozenset({"lab"}),
        venue_terms=frozenset({"neurips"}),
        title_terms=frozenset({"graph", "learning"}),
        has_coauthors=True,
        has_affiliations=True,
    )
    summary = build_cluster_summary(
        component_key="c1",
        size=4,
        first_name_counts=Counter({"alice": 4}),
        middle_initial_counts=Counter({"b": 4}),
        coauthor_counts=Counter({"a smith": 3}),
        affiliation_counts=Counter({"lab": 2}),
        venue_counts=Counter({"neurips": 3}),
        title_counts=Counter({"graph": 4, "learning": 2}),
        year_min=2010,
        year_max=2013,
        year_mean=2011.5,
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        exemplar_vectors=[np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    retriever = retrieval_utils.build_rust_hybrid_centroid_retriever([summary], include_exemplars=True)

    payload = retrieval_utils.compute_chooser_summary_features_rust_hybrid_centroid(
        query=query,
        component_keys=["c1"],
        summary_by_component={"c1": summary},
        retriever=retriever,
        num_threads=1,
    )

    assert payload["c1"]["middle_initial_compatibility"] == pytest.approx(1.0)
    assert payload["c1"]["coauthor_overlap"] == pytest.approx(0.75)
    assert payload["c1"]["affiliation_overlap"] == pytest.approx(0.5)
    assert payload["c1"]["venue_overlap"] == pytest.approx(0.75)
    assert payload["c1"]["year_compatibility"] == pytest.approx(0.9666666667)
    assert payload["c1"]["title_overlap"] == pytest.approx(0.75)
    assert payload["c1"]["specter_centroid_similarity"] == pytest.approx(1.0)
    assert payload["c1"]["specter_exemplar_similarity"] == pytest.approx(1.0)
