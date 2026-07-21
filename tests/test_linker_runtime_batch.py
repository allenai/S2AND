from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.query_adapter import mask_query_features
from s2and.incremental_linking.retrieval import build_linker_retrieval_batch_rust
from s2and.incremental_linking.row_features import (
    build_promoted_non_pairwise_row_features,
    build_promoted_non_pairwise_row_features_with_telemetry,
)
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import build_cluster_summary, build_query_features
from tests.linker_row_feature_reference import build_promoted_non_pairwise_row_features_python_reference

s2and_rust = load_s2and_rust_extension()


def test_pair_plan_rejects_values_above_uint16_rank_limit() -> None:
    query = build_query_features(first="alice")
    retriever = s2and_rust.RustHybridCentroidRetriever(
        [build_cluster_summary(component_key="c1", first_name_counts=Counter({"alice": 1}))],
        include_exemplars=True,
    )

    with pytest.raises(ValueError, match="retrieval_ranks"):
        retriever.top_k_hybrid_centroid_pair_plan(
            [query],
            np.asarray([0], dtype=np.uint32),
            {"c1": np.asarray([1], dtype=np.uint32)},
            int(np.iinfo(np.uint16).max) + 1,
            1,
        )


def _base_row_signals(row_count: int) -> dict[str, object]:
    return {
        "candidate_component_key": np.asarray([f"c{index}" for index in range(row_count)], dtype=object),
        "cluster_size": np.asarray([4.0] * row_count, dtype=np.float32),
        "named_signature_count": np.asarray([4.0] * row_count, dtype=np.float32),
        "dominant_first_name": np.asarray(["alice"] * row_count, dtype=object),
        "candidate_year_min": np.asarray([2010] * row_count, dtype=np.float32),
        "candidate_year_max": np.asarray([2015] * row_count, dtype=np.float32),
        "candidate_year_range_missing": np.zeros(row_count, dtype=np.float32),
        "query_first_token": np.asarray(["al"] * row_count, dtype=object),
        "query_year": np.asarray([2012] * row_count, dtype=np.float32),
        "query_year_missing": np.zeros(row_count, dtype=np.float32),
        "query_has_affiliations": np.ones(row_count, dtype=np.float32),
        "affiliation_overlap": np.asarray([0.8, 0.2, 0.5], dtype=np.float32)[:row_count],
        "coauthor_overlap": np.asarray([0.7, 0.1, 0.4], dtype=np.float32)[:row_count],
        "year_compatibility": np.asarray([1.0, 0.5, 0.8], dtype=np.float32)[:row_count],
        "specter_exemplar_similarity": np.asarray([0.8, 0.2, 0.5], dtype=np.float32)[:row_count],
        "min_distance": np.asarray([0.1, 0.4, 0.2], dtype=np.float32)[:row_count],
        "mean_distance": np.asarray([0.2, 0.45, 0.25], dtype=np.float32)[:row_count],
        "top3_mean_distance": np.asarray([0.15, 0.43, 0.22], dtype=np.float32)[:row_count],
        "top5_mean_distance": np.asarray([0.12, 0.42, 0.21], dtype=np.float32)[:row_count],
        "last_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_last_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "first_prefix_x_last_first_name_count_min_rarity": np.zeros(row_count, dtype=np.float32),
        "candidate_cluster_max_paper_author_count": np.asarray([4.0, 6.0, 8.0], dtype=np.float32)[:row_count],
        "paper_author_list_max_jaccard": np.asarray([1.0, 0.5, 0.0], dtype=np.float32)[:row_count],
        "paper_author_list_max_containment": np.asarray([1.0, 0.75, 0.0], dtype=np.float32)[:row_count],
        "paper_author_list_max_overlap_count": np.asarray([3.0, 2.0, 0.0], dtype=np.float32)[:row_count],
        "local_author_window10_jaccard_max": np.asarray([0.5, 0.25, 0.0], dtype=np.float32)[:row_count],
        "local_author_window10_overlap_count_max": np.asarray([3.0, 2.0, 0.0], dtype=np.float32)[:row_count],
        "best_author_count_log_absdiff": np.asarray([0.0, 0.5, 1.0], dtype=np.float32)[:row_count],
    }


def test_rust_retrieval_batch_returns_flat_pair_plan() -> None:
    query = build_query_features(first="alice", has_coauthors=True, has_affiliations=True)
    summaries = [
        build_cluster_summary(
            component_key="c1",
            size=2,
            first_name_counts=Counter({"alice": 2}),
            coauthor_counts=Counter({"a smith": 2}),
            affiliation_counts=Counter({"lab": 2}),
        ),
        build_cluster_summary(component_key="c2", size=1, first_name_counts=Counter({"bob": 1})),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        component_member_indices_by_key={"c1": [1, 2], "c2": [3]},
        top_k=2,
        query_view="initial_only",
        n_jobs=1,
    )

    assert batch.candidate_batch.row_count == 2
    assert batch.candidate_batch.pair_count == 3
    assert batch.candidate_batch.left_signature_indices.tolist() == [9, 9, 9]
    assert batch.candidate_batch.pair_row_indices.tolist() == [0, 0, 1]
    assert batch.row_signals["query_view"].tolist() == ["initial_only", "initial_only"]
    assert batch.row_signals["first_name_bucket"].tolist() == ["multi_letter_first", "multi_letter_first"]
    assert "affiliation_overlap" in batch.row_signals


def test_rust_retrieval_batch_preserves_single_character_title_and_venue_terms() -> None:
    query = build_query_features(
        first="alice",
        title_terms=frozenset({"a", "m", "study"}),
        venue_terms=frozenset({"series", "a"}),
        has_full_first=True,
    )
    summaries = [
        build_cluster_summary(
            component_key="c1",
            size=1,
            first_name_counts=Counter({"alice": 1}),
            title_counts=Counter({"a": 1, "study": 1}),
            venue_counts=Counter({"a": 1}),
        )
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        component_member_indices_by_key={"c1": [1]},
        top_k=1,
        query_view="full",
        n_jobs=1,
    )

    assert batch.candidate_batch.row_component_keys == ("c1",)
    np.testing.assert_allclose(batch.row_signals["title_overlap"], [2.0 / 3.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(batch.row_signals["venue_overlap"], [0.5], rtol=1e-6, atol=1e-6)


def test_rust_retrieval_batch_has_stable_retrieval_order() -> None:
    queries = [
        build_query_features(first="alice", has_coauthors=True, has_affiliations=True),
        build_query_features(first="bob", has_coauthors=True, has_affiliations=True),
    ]
    summaries = [
        build_cluster_summary(
            component_key="c_alice",
            size=3,
            first_name_counts=Counter({"alice": 3}),
            coauthor_counts=Counter({"a smith": 3}),
            affiliation_counts=Counter({"lab": 3}),
        ),
        build_cluster_summary(
            component_key="c_bob",
            size=2,
            first_name_counts=Counter({"bob": 2}),
            coauthor_counts=Counter({"b smith": 2}),
            affiliation_counts=Counter({"dept": 2}),
        ),
        build_cluster_summary(
            component_key="c_tie_a",
            size=1,
            first_name_counts=Counter({"alex": 1}),
        ),
        build_cluster_summary(
            component_key="c_tie_b",
            size=1,
            first_name_counts=Counter({"alex": 1}),
        ),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=queries,
        query_signature_indices=np.asarray([9, 10], dtype=np.uint32),
        component_member_indices_by_key={
            "c_alice": [1, 2, 3],
            "c_bob": [4, 5],
            "c_tie_a": [6],
            "c_tie_b": [7],
        },
        top_k=4,
        query_view=["initial_only", "initial_only"],
        n_jobs=2,
    )

    row_component_keys = batch.candidate_batch.row_component_keys
    retrieval_scores = batch.candidate_batch.retrieval_scores
    assert row_component_keys is not None
    assert retrieval_scores is not None
    assert list(row_component_keys) == [
        "c_alice",
        "c_bob",
        "c_tie_a",
        "c_tie_b",
        "c_alice",
        "c_bob",
        "c_tie_a",
        "c_tie_b",
    ]
    assert float(retrieval_scores[0]) > 0.0
    assert float(retrieval_scores[4]) == pytest.approx(float(retrieval_scores[0]))
    np.testing.assert_allclose(retrieval_scores[[1, 2, 3, 5, 6, 7]], 0.0, rtol=1e-6, atol=1e-6)


def test_rust_retrieval_batch_rejects_unknown_query_view_before_retrieval() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval query_view"):
        build_linker_retrieval_batch_rust(
            retriever=object(),
            queries=[],
            query_signature_indices=np.asarray([], dtype=np.uint32),
            component_member_indices_by_key={},
            top_k=1,
            query_view="typo",
            n_jobs=1,
        )


def test_rust_retrieval_batch_rejects_unknown_per_query_view_before_retrieval() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval query_view"):
        build_linker_retrieval_batch_rust(
            retriever=object(),
            queries=[object()],
            query_signature_indices=np.asarray([1], dtype=np.uint32),
            component_member_indices_by_key={},
            top_k=1,
            query_view=["initial_only_no_specter"],
            n_jobs=1,
        )


def test_retrieval_batch_rejects_duplicate_query_signature_indices_for_per_query_views() -> None:
    class RetrieverShouldNotRun:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("duplicate query_signature_indices should be rejected before retrieval")

    with pytest.raises(ValueError, match="query_signature_indices must be unique"):
        build_linker_retrieval_batch_rust(
            retriever=RetrieverShouldNotRun(),
            queries=[object(), object()],
            query_signature_indices=np.asarray([7, 7], dtype=np.uint32),
            component_member_indices_by_key={},
            top_k=1,
            query_view=["full", "initial_only"],
            n_jobs=1,
        )


def test_rust_retrieval_batch_rejects_missing_pair_plan_contract_keys() -> None:
    class ContractBreakingRetriever:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            del args, kwargs
            return {"row_count": 0}

    with pytest.raises(RuntimeError, match="violated its result contract.*row_orcid_match"):
        build_linker_retrieval_batch_rust(
            retriever=ContractBreakingRetriever(),
            queries=[],
            query_signature_indices=np.asarray([], dtype=np.uint32),
            component_member_indices_by_key={},
            top_k=1,
            query_view="initial_only",
            n_jobs=1,
        )


def test_rust_retrieval_batch_rejects_missing_consumed_row_signal_key() -> None:
    empty_plan = {
        "row_count": 0,
        "left_signature_indices": np.asarray([], dtype=np.uint32),
        "right_signature_indices": np.asarray([], dtype=np.uint32),
        "pair_row_indices": np.asarray([], dtype=np.uint32),
        "row_query_signature_indices": np.asarray([], dtype=np.uint32),
        "row_component_keys": [],
        "retrieval_scores": np.asarray([], dtype=np.float32),
        "retrieval_ranks": np.asarray([], dtype=np.uint16),
        "row_query_first_tokens": [],
        "row_component_sizes": np.asarray([], dtype=np.float32),
        "row_named_signature_counts": np.asarray([], dtype=np.float32),
        "row_dominant_first_names": [],
        "row_candidate_year_min": np.asarray([], dtype=np.int32),
        "row_candidate_year_max": np.asarray([], dtype=np.int32),
        "row_candidate_year_range_missing": np.asarray([], dtype=np.uint8),
        "row_query_years": np.asarray([], dtype=np.int32),
        "row_query_year_missing": np.asarray([], dtype=np.uint8),
        "row_query_has_affiliations": np.asarray([], dtype=np.uint8),
        "row_query_has_coauthors": np.asarray([], dtype=np.uint8),
        "row_orcid_match": np.asarray([], dtype=np.uint8),
        "middle_initial_compatibility": np.asarray([], dtype=np.float32),
        "affiliation_overlap": np.asarray([], dtype=np.float32),
        "coauthor_overlap": np.asarray([], dtype=np.float32),
        "venue_overlap": np.asarray([], dtype=np.float32),
        "year_compatibility": np.asarray([], dtype=np.float32),
        "title_overlap": np.asarray([], dtype=np.float32),
        "specter_centroid_similarity": np.asarray([], dtype=np.float32),
    }

    class ContractBreakingRetriever:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            del args, kwargs
            return dict(empty_plan)

    with pytest.raises(RuntimeError, match="violated its result contract.*specter_exemplar_similarity"):
        build_linker_retrieval_batch_rust(
            retriever=ContractBreakingRetriever(),
            queries=[],
            query_signature_indices=np.asarray([], dtype=np.uint32),
            component_member_indices_by_key={},
            top_k=1,
            query_view="initial_only",
            n_jobs=1,
        )


def test_full_query_view_changes_same_initial_retrieval_order() -> None:
    base_query = build_query_features(first="alice", has_full_first=True)
    initial_query = mask_query_features(base_query, "initial_only")
    full_query = mask_query_features(base_query, "full")
    summaries = [
        build_cluster_summary(component_key="c_adam", size=1, first_name_counts=Counter({"adam": 1})),
        build_cluster_summary(component_key="c_alice", size=1, first_name_counts=Counter({"alice": 1})),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    initial_batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[initial_query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        component_member_indices_by_key={"c_adam": [1], "c_alice": [2]},
        top_k=2,
        query_view="initial_only",
        n_jobs=1,
    )
    assert initial_batch.candidate_batch.row_component_keys == ("c_adam", "c_alice")
    initial_scores = initial_batch.candidate_batch.retrieval_scores
    assert initial_scores is not None
    np.testing.assert_allclose(initial_scores, [0.0, 0.0], rtol=1e-6, atol=1e-6)

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[full_query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        component_member_indices_by_key={"c_adam": [1], "c_alice": [2]},
        top_k=2,
        query_view="full",
        n_jobs=1,
    )

    assert batch.candidate_batch.row_component_keys == ("c_alice", "c_adam")
    assert batch.row_signals["query_view"].tolist() == ["full", "full"]
    assert batch.row_signals["query_first_token"].tolist() == ["alice", "alice"]
    retrieval_scores = batch.candidate_batch.retrieval_scores
    assert retrieval_scores is not None
    assert float(retrieval_scores[0]) > float(retrieval_scores[1])


def test_rust_retrieval_batch_orcid_override_returns_all_matches_without_middle_or_year_filters() -> None:
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"q"}),
        year=2024,
        orcid="0000-0001-2345-6789",
        has_full_first=True,
    )
    summaries = [
        build_cluster_summary(
            component_key="orcid_match",
            first_name_counts=Counter({"alice": 1}),
            middle_initial_counts=Counter({"q": 1}),
            year_min=2024,
            year_max=2024,
            orcid_values=frozenset({"0000-0001-2345-6789"}),
        ),
        build_cluster_summary(
            component_key="orcid_middle_conflict",
            first_name_counts=Counter({"alice": 1}),
            middle_initial_counts=Counter({"z": 1}),
            year_min=2024,
            year_max=2024,
            orcid_values=frozenset({"0000-0001-2345-6789"}),
        ),
        build_cluster_summary(
            component_key="orcid_year_conflict",
            first_name_counts=Counter({"alice": 1}),
            middle_initial_counts=Counter({"q": 1}),
            year_min=1900,
            year_max=1900,
            orcid_values=frozenset({"0000-0001-2345-6789"}),
        ),
        build_cluster_summary(
            component_key="non_orcid_candidate",
            first_name_counts=Counter({"alice": 1}),
            middle_initial_counts=Counter({"q": 1}),
            year_min=2024,
            year_max=2024,
        ),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        query_signature_ids=["q1"],
        component_member_indices_by_key={
            "orcid_match": [1],
            "orcid_middle_conflict": [2],
            "orcid_year_conflict": [3],
            "non_orcid_candidate": [4],
        },
        top_k=1,
        query_view="full",
        n_jobs=1,
        query_candidate_component_keys_by_signature_id={"q1": ["non_orcid_candidate"]},
    )

    expected = {"orcid_match", "orcid_middle_conflict", "orcid_year_conflict"}
    row_component_keys = batch.candidate_batch.row_component_keys
    assert row_component_keys is not None
    assert set(row_component_keys) == expected
    assert "non_orcid_candidate" not in row_component_keys
    assert batch.row_signals["orcid_match"].tolist() == [1.0, 1.0, 1.0]


def test_fixed_rust_retrieval_rescues_high_coverage_mega_candidates() -> None:
    query = build_query_features(
        first="alice",
        coauthor_blocks=frozenset({"a one", "b two", "c three", "d four"}),
        has_coauthors=True,
    )
    summaries = [
        build_cluster_summary(
            component_key="c_rescue",
            size=1,
            coauthor_counts=Counter({"a one": 1, "b two": 1, "c three": 1, "d four": 1}),
            non_mega_coauthor_counts=Counter(),
            max_paper_author_count=100,
        ),
        build_cluster_summary(
            component_key="c_nonmega",
            size=1,
            coauthor_counts=Counter({"a one": 1}),
            non_mega_coauthor_counts=Counter({"a one": 1}),
            max_paper_author_count=4,
        ),
        build_cluster_summary(
            component_key="c_partial",
            size=1,
            coauthor_counts=Counter({"a one": 1}),
            non_mega_coauthor_counts=Counter(),
            max_paper_author_count=100,
        ),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)

    keys, scores = retriever.top_k_hybrid_centroid_subset(
        query,
        ["c_rescue", "c_nonmega", "c_partial"],
        top_k=3,
        num_threads=1,
    )

    assert keys == ["c_rescue", "c_nonmega", "c_partial"]
    assert scores[0] > scores[1] > 0.0
    assert scores[2] == pytest.approx(0.0)


def test_rust_retrieval_batch_applies_name_compatible_full_first_window() -> None:
    query = build_query_features(first="alice", has_full_first=True)
    summaries = [
        build_cluster_summary(component_key="c_same", size=1, first_name_counts=Counter({"alice": 1})),
        build_cluster_summary(component_key="c_name", size=1, first_name_counts=Counter({"alice": 1})),
        build_cluster_summary(component_key="c_backfill", size=1, first_name_counts=Counter({"carol": 1})),
        build_cluster_summary(component_key="c_other", size=1, first_name_counts=Counter({"bob": 1})),
    ]
    retriever = s2and_rust.RustHybridCentroidRetriever(summaries, include_exemplars=True)
    retrieval_subblock_index = {
        "signature_to_subblock": {"q1": "block::zz"},
        "subblock_to_components": {
            "block::zz": ["c_same"],
            "block::ally": ["c_name"],
            "block::bob": ["c_other"],
        },
        "subblock_tokens_by_subblock": {
            "block::zz": ["zz"],
            "block::ally": ["ally"],
            "block::bob": ["bob"],
        },
    }

    batch = build_linker_retrieval_batch_rust(
        retriever=retriever,
        queries=[query],
        query_signature_indices=np.asarray([9], dtype=np.uint32),
        query_signature_ids=["q1"],
        component_member_indices_by_key={
            "c_same": [1],
            "c_name": [2],
            "c_backfill": [3],
            "c_other": [4],
        },
        top_k=4,
        query_view="full",
        n_jobs=1,
        retrieval_subblock_index=retrieval_subblock_index,
        query_candidate_component_keys_by_signature_id={"q1": ["c_same", "c_name", "c_backfill"]},
        full_first_global_backfill_count=2,
        name_tuples={("alice", "ally")},
    )

    row_component_keys = batch.candidate_batch.row_component_keys
    assert row_component_keys is not None
    assert set(row_component_keys) == {"c_same", "c_name", "c_backfill"}
    assert "c_other" not in row_component_keys


def test_promoted_non_pairwise_row_features_derive_group_columns() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c0", "c1", "c2"),
    )
    row_signals = _base_row_signals(3)
    row_signals["retrieval_score"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1, 2, 1], dtype=np.float32)
    row_signals["family_id"] = np.asarray(["alice", "alice", "alice"], dtype=object)

    features = build_promoted_non_pairwise_row_features(candidate_batch, row_signals)

    assert tuple(features)  # column order is owned by the promoted schema constant
    assert "retrieval_rank_fraction" not in features
    assert "retrieval_score_gap_vs_best_competitor" not in features
    np.testing.assert_allclose(features["retrieval_reciprocal_rank"], [1.0, 0.5, 1.0])
    np.testing.assert_allclose(features["cluster_size_log"], [np.log1p(4.0)] * 3)
    np.testing.assert_allclose(features["candidate_year_span"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(features["year_gap_to_candidate_range"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(features["candidate_dominant_first_name_length"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(features["candidate_cluster_max_paper_author_count"], [4.0, 6.0, 8.0])
    np.testing.assert_allclose(features["paper_author_list_max_jaccard"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(features["paper_author_list_max_containment"], [1.0, 0.75, 0.0])
    np.testing.assert_allclose(features["paper_author_list_max_overlap_count"], [3.0, 2.0, 0.0])
    np.testing.assert_allclose(features["local_author_window10_jaccard_max"], [0.5, 0.25, 0.0])
    np.testing.assert_allclose(features["local_author_window10_overlap_count_max"], [3.0, 2.0, 0.0])
    np.testing.assert_allclose(features["best_author_count_log_absdiff"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(features["query_first_prefix_match_any_length"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(features["same_dominant_first_as_best_top5"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(features["same_family_as_heuristic_choice"], [1.78, 1.38, 1.49])


def test_promoted_non_pairwise_row_features_use_current_score_order_for_rank_features() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 10], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 10], dtype=np.uint32),
        row_component_keys=("old_rank1", "current_winner", "third"),
    )
    row_signals = _base_row_signals(3)
    row_signals["candidate_component_key"] = np.asarray(["old_rank1", "current_winner", "third"], dtype=object)
    row_signals["retrieval_score"] = np.asarray([0.50, 0.90, 0.20], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1, 2, 3], dtype=np.float32)
    row_signals["family_id"] = np.asarray(["alice", "bob", "bob"], dtype=object)

    features = build_promoted_non_pairwise_row_features(candidate_batch, row_signals)

    np.testing.assert_allclose(features["retrieval_rank"], [2.0, 1.0, 3.0])
    np.testing.assert_allclose(features["retrieval_reciprocal_rank"], [0.5, 1.0, 1.0 / 3.0], atol=1e-6)
    np.testing.assert_allclose(features["same_family_as_heuristic_choice"], [1.38, 1.48, 0.99], rtol=1e-6)
    np.testing.assert_allclose(features["strong_positive_anchor_score"], [0.09, 0.12, 0.16], rtol=1e-6)
    np.testing.assert_allclose(features["weak_residual_anchor_score"], [0.0, 0.248, 0.224], rtol=1e-6)
    np.testing.assert_allclose(features["sparse_relative_winner_score"], [0.0, 0.248, 0.0], rtol=1e-6)


def test_rust_promoted_non_pairwise_row_features_match_python_reference() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c0", "c1", "c2"),
    )
    row_signals = _base_row_signals(3)
    row_signals["retrieval_score"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1, 2, 1], dtype=np.float32)
    row_signals["family_id"] = np.asarray(["alice", "alice", "alice"], dtype=object)
    row_signals["query_first_token"] = np.asarray(["Él", "El", "Zoë"], dtype=object)
    row_signals["dominant_first_name"] = np.asarray(["Élodie", "Elodie", "Zoe"], dtype=object)

    rust_features = build_promoted_non_pairwise_row_features(candidate_batch, row_signals)
    python_features = build_promoted_non_pairwise_row_features_python_reference(candidate_batch, row_signals)

    assert rust_features.keys() == python_features.keys()
    for column in rust_features:
        np.testing.assert_allclose(rust_features[column], python_features[column], rtol=1e-6, atol=1e-6)


def test_promoted_non_pairwise_soft_year_features_are_raw_primitives() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11, 11], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3, 4], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 10], dtype=np.uint32),
        row_component_keys=("c0", "c1", "c2"),
    )
    row_signals = _base_row_signals(3)
    row_signals["query_year"] = np.asarray([2008.0, 2020.0, 2012.0], dtype=np.float32)
    row_signals["candidate_year_min"] = np.asarray([2010.0, 2010.0, 2010.0], dtype=np.float32)
    row_signals["candidate_year_max"] = np.asarray([2015.0, 2015.0, 2015.0], dtype=np.float32)
    row_signals["retrieval_score"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    features = build_promoted_non_pairwise_row_features(candidate_batch, row_signals)

    np.testing.assert_allclose(features["year_gap_to_candidate_range"], [2.0, 5.0, 0.0])
    np.testing.assert_allclose(features["year_gap_signed_to_candidate_range"], [-2.0, 5.0, 0.0])


def test_promoted_non_pairwise_row_features_reports_generated_family_ids() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c0", "c1", "c2"),
    )
    row_signals = _base_row_signals(3)
    row_signals["retrieval_score"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1, 2, 1], dtype=np.float32)

    _features, telemetry = build_promoted_non_pairwise_row_features_with_telemetry(candidate_batch, row_signals)

    assert telemetry["generated_family_id_count"] == 3
    assert telemetry["generic_family_override_count"] == 3


def test_promoted_row_features_reject_missing_rust_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([10], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c1",),
        retrieval_scores=np.asarray([0.9], dtype=np.float32),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
    )
    row_signals = _base_row_signals(1)
    row_signals["retrieval_score"] = np.asarray([0.9], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1.0], dtype=np.float32)
    original = s2and_rust.promoted_linker_non_pairwise_features

    def stale_result(payload):
        result = dict(original(payload))
        del result["min_distance"]
        return result

    monkeypatch.setattr(s2and_rust, "promoted_linker_non_pairwise_features", stale_result)
    with pytest.raises(RuntimeError, match="missing columns.*min_distance"):
        build_promoted_non_pairwise_row_features(candidate_batch, row_signals)


def test_promoted_row_features_reject_incomplete_rust_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([10], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c1",),
        retrieval_scores=np.asarray([0.9], dtype=np.float32),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
    )
    row_signals = _base_row_signals(1)
    row_signals["retrieval_score"] = np.asarray([0.9], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1.0], dtype=np.float32)
    original = s2and_rust.promoted_linker_non_pairwise_features

    def stale_result(payload):
        result = dict(original(payload))
        result["telemetry"] = {"generated_family_id_count": 0}
        return result

    monkeypatch.setattr(s2and_rust, "promoted_linker_non_pairwise_features", stale_result)
    with pytest.raises(RuntimeError, match="telemetry schema mismatch"):
        build_promoted_non_pairwise_row_features(candidate_batch, row_signals)


def test_promoted_non_pairwise_row_features_treats_family_id_none_as_missing() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c0", "c1", "c2"),
    )
    row_signals = _base_row_signals(3)
    row_signals["retrieval_score"] = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    row_signals["retrieval_rank"] = np.asarray([1, 2, 1], dtype=np.float32)
    row_signals["family_id"] = None

    _features, telemetry = build_promoted_non_pairwise_row_features_with_telemetry(candidate_batch, row_signals)

    assert telemetry["generated_family_id_count"] == 3
