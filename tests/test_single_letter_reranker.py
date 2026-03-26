from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.build_single_letter_reranker_dataset as reranker_build
import scripts.eval_single_letter_ranker as s2and_ranker_eval
import scripts.single_letter_reranker_utils as reranker_utils
from tests.helpers import build_cluster_summary, build_query_features


def _base_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": "labeled",
        "dataset": "d1",
        "query_source": "labeled",
        "query_view": "initial_only",
        "natural_query_view": "initial_only",
        "query_group_id": "g1",
        "query_id": "q1",
        "query_signature_id": "q1",
        "normalized_orcid": None,
        "orcid_group_size": None,
        "orcid_group_size_bucket": None,
        "split": "all",
        "block_key": "b",
        "block_size": 10,
        "component_size": 3,
        "sampling_info_bucket": "rich",
        "supervision_type": "labeled",
        "positive_candidate_count": 1,
        "positive_candidate_keys": "c1",
        "group_has_positive": 1,
        "best_positive_retrieval_rank": 1,
        "support_type": "labeled",
        "query_in_seed_before_holdout": 0,
        "candidate_component_key": "c1",
        "candidate_cluster_id": "c1",
        "best_competitor_component_key": "c2",
        "family_id": "fam1",
        "dominant_first_name": "alice",
        "label": 1,
        "candidate_count": 2,
        "candidate_signatures": 5,
        "scored_candidate_components": 2,
        "scored_candidate_signatures": 5,
        "orcid_filter_applied": 0,
        "middle_initial_filter_applied": 0,
        "year_range_filter_applied": 0,
        "retrieval_rank": 1,
        "retrieval_score": 0.8,
        "cluster_size": 3,
        "pair_count": 3,
        "min_distance": 0.1,
        "mean_distance": 0.2,
        "top3_mean_distance": 0.2,
        "top5_mean_distance": 0.2,
        "top10_mean_distance": 0.2,
        "top20_mean_distance": 0.2,
        "count_normalized_confidence": 0.8,
        "retrieval_score_gap_vs_best_competitor": 0.1,
        "retrieval_rank_gap_vs_best_competitor": -1.0,
        "top3_mean_delta_vs_best_competitor": 0.2,
        "top5_mean_delta_vs_best_competitor": 0.2,
        "cluster_size_ratio_vs_best_competitor": 1.5,
        "same_family_vs_best_competitor": 0,
        "dominant_name_ratio": 0.9,
        "named_signature_count": 3,
        "confident_family_flag": 1,
        "same_family_as_top1": 1,
        "middle_initial_compatibility": 1.0,
        "affiliation_overlap": 0.5,
        "coauthor_overlap": 0.5,
        "venue_overlap": 0.0,
        "year_compatibility": 1.0,
        "family_instability_flag": 0,
        "fragment_flag": 0,
        "query_has_specter": 1,
        "query_has_coauthors": 1,
        "query_has_affiliations": 1,
        "query_has_middle": 0,
        "query_has_full_first": 0,
    }
    row.update(overrides)
    return row


def test_build_training_matrix_drops_all_negative_groups() -> None:
    rows = [
        _base_row(query_group_id="g1", candidate_component_key="c1", label=1),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            label=0,
            retrieval_rank=2,
            retrieval_score=0.7,
            best_competitor_component_key="c1",
        ),
        _base_row(
            query_group_id="g2",
            candidate_component_key="c3",
            label=0,
            positive_candidate_count=0,
            positive_candidate_keys="",
            group_has_positive=0,
            best_positive_retrieval_rank=None,
            retrieval_rank=1,
        ),
        _base_row(
            query_group_id="g2",
            candidate_component_key="c4",
            label=0,
            positive_candidate_count=0,
            positive_candidate_keys="",
            group_has_positive=0,
            best_positive_retrieval_rank=None,
            retrieval_rank=2,
            best_competitor_component_key="c3",
        ),
    ]
    training_matrix = reranker_utils.build_training_matrix(rows, seed=7)
    assert training_matrix.features.shape == (2, len(reranker_utils.FEATURE_COLUMNS))
    assert training_matrix.group_ids == ["g1"]
    assert training_matrix.dropped_all_negative_group_ids == ["g2"]
    assert training_matrix.sample_weights.tolist() == [0.5, 0.5]


def test_generalized_feature_matrix_uses_group_relative_features() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            retrieval_score=0.9,
            min_distance=0.1,
            mean_distance=0.2,
            top3_mean_distance=0.2,
            top5_mean_distance=0.2,
            affiliation_overlap=0.5,
            coauthor_overlap=0.6,
            venue_overlap=0.1,
            year_compatibility=0.9,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            retrieval_score=0.7,
            min_distance=0.4,
            mean_distance=0.5,
            top3_mean_distance=0.5,
            top5_mean_distance=0.5,
            affiliation_overlap=0.2,
            coauthor_overlap=0.1,
            venue_overlap=0.0,
            year_compatibility=0.7,
            best_competitor_component_key="c1",
        ),
    ]
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v2")
    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=feature_columns)
    index = {column: idx for idx, column in enumerate(feature_columns)}
    assert matrix.shape == (2, len(feature_columns))
    assert matrix[0, index["retrieval_rank_fraction"]] == pytest.approx(0.0)
    assert matrix[1, index["retrieval_rank_fraction"]] == pytest.approx(1.0)
    assert matrix[0, index["top5_distance_best_gap"]] == pytest.approx(0.0)
    assert matrix[1, index["top5_distance_best_gap"]] == pytest.approx(0.3)
    assert matrix[0, index["same_family_as_best_top5"]] == pytest.approx(1.0)
    assert matrix[1, index["same_family_as_best_top5"]] == pytest.approx(0.0)
    assert "cluster_size" not in feature_columns


def test_generalized_v5_feature_matrix_exposes_heuristic_decision_stats() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            top3_mean_distance=0.25,
            top5_mean_distance=0.25,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            top3_mean_distance=0.21,
            top5_mean_distance=0.215,
            confident_family_flag=1,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v5")
    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=feature_columns)
    index = {column: idx for idx, column in enumerate(feature_columns)}
    assert matrix.shape == (2, len(feature_columns))
    assert matrix[0, index["is_retrieval_top1"]] == pytest.approx(1.0)
    assert matrix[1, index["is_best_top5"]] == pytest.approx(1.0)
    assert matrix[0, index["is_heuristic_choice"]] == pytest.approx(1.0)
    assert matrix[0, index["heuristic_prefers_top1"]] == pytest.approx(1.0)
    assert matrix[0, index["heuristic_cross_family_top1_vs_best_top5"]] == pytest.approx(1.0)
    assert matrix[0, index["heuristic_margin_threshold"]] == pytest.approx(0.04)
    assert matrix[0, index["heuristic_margin_slack"]] == pytest.approx(-0.005)


def test_feature_matrix_exposes_new_manual_residual_features() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            dominant_first_name="alice",
            retrieval_rank=1,
            pair_count=3,
            coauthor_overlap=0.5,
            top5_mean_distance=0.2,
            top10_mean_distance=0.25,
            top20_mean_distance=0.4,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            dominant_first_name="alina",
            label=0,
            retrieval_rank=2,
            pair_count=1,
            coauthor_overlap=0.2,
            top5_mean_distance=0.1,
            top10_mean_distance=0.12,
            top20_mean_distance=0.15,
            best_competitor_component_key="c1",
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c3",
            dominant_first_name="bob",
            family_id="fam3",
            label=0,
            retrieval_rank=3,
            pair_count=5,
            coauthor_overlap=0.1,
            top5_mean_distance=0.3,
            top10_mean_distance=0.31,
            top20_mean_distance=0.35,
            best_competitor_component_key="c1",
        ),
    ]
    feature_columns = (
        "coarse_family_pair_count_top50",
        "candidate_pair_share_within_coarse_family",
        "coarse_family_top5_best_gap",
        "coauthor_gap_to_best_same_coarse_family",
        "distance_spread_top20_minus_top5",
    )
    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=feature_columns)
    index = {column: idx for idx, column in enumerate(feature_columns)}
    assert matrix[0, index["coarse_family_pair_count_top50"]] == pytest.approx(4.0)
    assert matrix[1, index["coarse_family_pair_count_top50"]] == pytest.approx(4.0)
    assert matrix[0, index["candidate_pair_share_within_coarse_family"]] == pytest.approx(0.75)
    assert matrix[1, index["candidate_pair_share_within_coarse_family"]] == pytest.approx(0.25)
    assert matrix[0, index["coarse_family_top5_best_gap"]] == pytest.approx(0.1)
    assert matrix[1, index["coarse_family_top5_best_gap"]] == pytest.approx(0.0)
    assert matrix[0, index["coauthor_gap_to_best_same_coarse_family"]] == pytest.approx(0.3)
    assert matrix[1, index["coauthor_gap_to_best_same_coarse_family"]] == pytest.approx(0.0)
    assert matrix[0, index["distance_spread_top20_minus_top5"]] == pytest.approx(0.2)
    assert matrix[1, index["distance_spread_top20_minus_top5"]] == pytest.approx(0.05)


def test_generalized_v6_feature_matrix_keeps_only_core_heuristic_stats() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            top3_mean_distance=0.25,
            top5_mean_distance=0.25,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            top3_mean_distance=0.21,
            top5_mean_distance=0.215,
            confident_family_flag=1,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v6")
    assert "is_heuristic_choice" not in feature_columns
    assert "same_family_as_heuristic_choice" not in feature_columns
    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=feature_columns)
    index = {column: idx for idx, column in enumerate(feature_columns)}
    assert matrix[0, index["is_retrieval_top1"]] == pytest.approx(1.0)
    assert matrix[1, index["is_best_top5"]] == pytest.approx(1.0)
    assert matrix[0, index["heuristic_margin_slack"]] == pytest.approx(-0.005)
    assert matrix[1, index["top5_gap_to_retrieval_top1"]] == pytest.approx(-0.035)


def test_generalized_v8_feature_matrix_exposes_candidate_specific_top1_override_geometry() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            top3_mean_distance=0.25,
            top5_mean_distance=0.25,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam1",
            retrieval_rank=2,
            top3_mean_distance=0.22,
            top5_mean_distance=0.22,
            confident_family_flag=1,
            same_family_as_top1=1,
            label=0,
            best_competitor_component_key="c1",
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c3",
            family_id="fam2",
            retrieval_rank=3,
            top3_mean_distance=0.205,
            top5_mean_distance=0.215,
            confident_family_flag=1,
            same_family_as_top1=0,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v8")
    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=feature_columns)
    index = {column: idx for idx, column in enumerate(feature_columns)}
    assert matrix.shape == (3, len(feature_columns))
    assert matrix[0, index["cross_family_with_top1"]] == pytest.approx(0.0)
    assert matrix[1, index["cross_family_with_top1"]] == pytest.approx(0.0)
    assert matrix[2, index["cross_family_with_top1"]] == pytest.approx(1.0)
    assert matrix[0, index["override_slack_vs_top1"]] == pytest.approx(-0.01)
    assert matrix[1, index["override_slack_vs_top1"]] == pytest.approx(0.02)
    assert matrix[2, index["override_slack_vs_top1"]] == pytest.approx(-0.005)
    assert matrix[0, index["beats_top1_after_penalty"]] == pytest.approx(0.0)
    assert matrix[1, index["beats_top1_after_penalty"]] == pytest.approx(1.0)
    assert matrix[2, index["beats_top1_after_penalty"]] == pytest.approx(0.0)


def test_generalized_v9_feature_preset_prunes_low_shap_columns() -> None:
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v9")
    assert "override_slack_vs_top1" in feature_columns
    assert "min_distance_best_gap" in feature_columns
    assert "query_has_specter" not in feature_columns
    assert "query_has_coauthors" not in feature_columns
    assert "query_has_affiliations" not in feature_columns
    assert "query_has_middle" not in feature_columns
    assert "query_has_full_first" not in feature_columns
    assert "middle_initial_compatibility" not in feature_columns
    assert "heuristic_top1_vs_best_top5_margin" not in feature_columns
    assert "heuristic_margin_slack" not in feature_columns
    assert "heuristic_prefers_top1" not in feature_columns
    assert "heuristic_cross_family_top1_vs_best_top5" not in feature_columns
    assert "confident_family_flag" not in feature_columns
    assert "family_instability_flag" not in feature_columns
    assert "beats_top1_after_penalty" not in feature_columns
    assert len(feature_columns) < len(reranker_utils.resolve_feature_columns(feature_preset="generalized_v8"))


def test_make_candidate_rows_uses_non_self_best_competitor_anchor() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary_a = build_cluster_summary(component_key="c1", size=3)
    summary_b = build_cluster_summary(component_key="c2", size=2)
    stats_a = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        dominant_name_ratio=0.9,
        named_signature_count=3,
        count=3,
        sum_distance=0.6,
        min_distance=0.1,
        top_smallest_neg_heap=[-0.1, -0.2, -0.3],
    )
    stats_b = reranker_utils.ClusterPairwiseStats(
        cluster_id="c2",
        retrieval_rank=2,
        retrieval_score=0.7,
        cluster_size=2,
        family_id="fam2",
        dominant_name_ratio=0.8,
        named_signature_count=2,
        count=2,
        sum_distance=1.0,
        min_distance=0.4,
        top_smallest_neg_heap=[-0.4, -0.6],
    )
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c1"}),
        support_type="labeled",
        block_size=5,
        component_size=3,
        sampling_info_bucket="rich",
    )
    rows = reranker_utils.make_candidate_rows(
        query_case=query_case,
        query_view="initial_only",
        query_features=query,
        shortlist_component_keys=["c1", "c2"],
        retrieval_scores={"c1": 0.8, "c2": 0.7},
        retrieval_ranks={"c1": 1, "c2": 2},
        retrieval_window_state={
            "scored_candidate_components": 2,
            "scored_candidate_signatures": 5,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        summary_by_component={"c1": summary_a, "c2": summary_b},
        stats_by_component={"c1": stats_a, "c2": stats_b},
    )
    top_row = next(row for row in rows if row["candidate_component_key"] == "c1")
    assert top_row["best_competitor_component_key"] == "c2"
    assert top_row["retrieval_score_gap_vs_best_competitor"] != 0.0


def test_make_candidate_rows_uses_zero_competitor_gaps_for_single_candidate_groups() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary = build_cluster_summary(component_key="c1", size=3)
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        dominant_name_ratio=0.9,
        named_signature_count=3,
        count=3,
        sum_distance=0.6,
        min_distance=0.1,
        top_smallest_neg_heap=[-0.1, -0.2, -0.3],
    )
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c1"}),
        support_type="labeled",
        block_size=3,
        component_size=3,
        sampling_info_bucket="rich",
    )
    rows = reranker_utils.make_candidate_rows(
        query_case=query_case,
        query_view="initial_only",
        query_features=query,
        shortlist_component_keys=["c1"],
        retrieval_scores={"c1": 0.8},
        retrieval_ranks={"c1": 1},
        retrieval_window_state={
            "scored_candidate_components": 1,
            "scored_candidate_signatures": 3,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        summary_by_component={"c1": summary},
        stats_by_component={"c1": stats},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["best_competitor_component_key"] is None
    assert row["retrieval_score_gap_vs_best_competitor"] == pytest.approx(0.0)
    assert row["retrieval_rank_gap_vs_best_competitor"] == pytest.approx(0.0)


def test_choose_generic_heuristic_keeps_top1_when_cross_family_margin_is_small() -> None:
    retrieval_top1 = _base_row(
        candidate_component_key="c1",
        family_id="fam1",
        retrieval_rank=1,
        top5_mean_distance=0.25,
        confident_family_flag=1,
        label=1,
    )
    challenger = _base_row(
        candidate_component_key="c2",
        family_id="fam2",
        retrieval_rank=2,
        top5_mean_distance=0.23,
        confident_family_flag=1,
        label=0,
        best_competitor_component_key="c1",
    )
    chosen_component, _score = reranker_utils.choose_generic_heuristic(
        [retrieval_top1, challenger],
        window_size=10,
    )
    assert chosen_component == "c1"


def test_build_training_matrix_rejection_sampling_enriches_target_groups() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=2,
            top3_mean_distance=0.2,
            top5_mean_distance=0.2,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
            top3_mean_distance=0.3,
            top5_mean_distance=0.3,
            confident_family_flag=1,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    training_matrix = reranker_utils.build_training_matrix(
        rows,
        seed=7,
        feature_preset="generalized_v5",
        enrichment_profile="heuristic_error_regions_v1",
        enrichment_rounds=2,
    )
    assert training_matrix.group_repeat_counts == {"g1": 3}
    assert training_matrix.extra_group_copies == 2
    assert training_matrix.groups_with_extra_copies == 1
    assert training_matrix.features.shape[0] == 6


def test_training_matrix_group_sizes_expand_repeated_groups() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=2,
            top3_mean_distance=0.2,
            top5_mean_distance=0.2,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
            top3_mean_distance=0.3,
            top5_mean_distance=0.3,
            confident_family_flag=1,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    training_matrix = reranker_utils.build_training_matrix(
        rows,
        seed=7,
        feature_preset="generalized_v7",
        enrichment_profile="heuristic_override_regions_v2",
        enrichment_rounds=2,
    )
    assert s2and_ranker_eval._training_matrix_group_sizes(training_matrix) == [2, 2]  # noqa: SLF001


def test_s2and_hard_region_enrichment_targets_deeper_override_groups() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            confident_family_flag=0,
            retrieval_rank=26,
            top3_mean_distance=0.19,
            top5_mean_distance=0.19,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
            confident_family_flag=1,
            top3_mean_distance=0.34,
            top5_mean_distance=0.34,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    training_matrix = reranker_utils.build_training_matrix(
        rows,
        seed=7,
        feature_preset="generalized_v7",
        enrichment_profile="s2and_hard_regions_v1",
        enrichment_rounds=2,
    )
    assert training_matrix.group_repeat_counts == {"g1": 3}
    assert training_matrix.extra_group_copies == 2
    assert training_matrix.groups_with_extra_copies == 1


def test_fit_ranker_with_hyperopt_smoke() -> None:
    train_rows = [
        _base_row(query_group_id="g1", candidate_component_key="c1", label=1, retrieval_rank=1),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            label=0,
            retrieval_rank=2,
            retrieval_score=0.2,
            top5_mean_distance=0.8,
            count_normalized_confidence=0.1,
            best_competitor_component_key="c1",
        ),
        _base_row(query_group_id="g2", candidate_component_key="c3", label=1, retrieval_rank=1),
        _base_row(
            query_group_id="g2",
            candidate_component_key="c4",
            label=0,
            retrieval_rank=2,
            retrieval_score=0.2,
            top5_mean_distance=0.85,
            count_normalized_confidence=0.05,
            best_competitor_component_key="c3",
        ),
    ]
    validation_rows = [
        _base_row(query_group_id="g3", candidate_component_key="c5", label=1, retrieval_rank=1),
        _base_row(
            query_group_id="g3",
            candidate_component_key="c6",
            label=0,
            retrieval_rank=2,
            retrieval_score=0.3,
            top5_mean_distance=0.75,
            count_normalized_confidence=0.1,
            best_competitor_component_key="c5",
        ),
    ]
    feature_columns = reranker_utils.resolve_feature_columns(feature_preset="generalized_v7")
    training_matrix = reranker_utils.build_training_matrix(
        train_rows,
        seed=7,
        feature_columns=feature_columns,
        enrichment_profile="none",
        enrichment_rounds=0,
    )
    model, summary = s2and_ranker_eval._fit_ranker_with_hyperopt(  # noqa: SLF001
        training_matrix=training_matrix,
        validation_rows=validation_rows,
        feature_columns=feature_columns,
        seed=7,
        hyperopt_evals=0,
        n_jobs=1,
    )
    scores = s2and_ranker_eval._predict_scores(  # noqa: SLF001
        model,
        reranker_utils.build_feature_matrix(validation_rows, feature_columns=feature_columns),
    )
    assert len(scores) == 2
    assert summary["model_type"] == "lgbm_ranker"
    assert summary["hyperopt_trials_ran"] == 0


def test_override_only_enrichment_skips_tight_keep_top1_groups() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            top3_mean_distance=0.25,
            top5_mean_distance=0.25,
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            top3_mean_distance=0.21,
            top5_mean_distance=0.215,
            confident_family_flag=1,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]
    training_matrix = reranker_utils.build_training_matrix(
        rows,
        seed=7,
        feature_preset="generalized_v6",
        enrichment_profile="heuristic_override_regions_v2",
        enrichment_rounds=3,
    )
    assert training_matrix.group_repeat_counts == {"g1": 1}
    assert training_matrix.extra_group_copies == 0
    assert training_matrix.groups_with_extra_copies == 0


def test_resolve_feature_columns_rejects_unknown_preset() -> None:
    with pytest.raises(ValueError, match="Unknown feature preset"):
        reranker_utils.resolve_feature_columns(feature_preset="missing")


def test_resolve_feature_columns_supports_small_core_3_preset() -> None:
    assert reranker_utils.resolve_feature_columns(feature_preset="small_core_3") == (
        "min_distance_rank_fraction",
        "top3_distance_rank_fraction",
        "is_heuristic_choice",
    )


def test_resolve_load_name_counts_auto_checks_primary_and_nameless_models() -> None:
    clusterer_with_name_counts = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=["name_counts", "coauthor_similarity"])
    )
    clusterer_with_nameless_name_counts = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=["coauthor_similarity"]),
        nameless_featurizer_info=SimpleNamespace(features_to_use=["name_counts"]),
    )
    clusterer_without_name_counts = SimpleNamespace(
        featurizer_info=SimpleNamespace(features_to_use=["coauthor_similarity"]),
        nameless_featurizer_info=SimpleNamespace(features_to_use=["year_diff"]),
    )
    assert (
        reranker_utils._resolve_load_name_counts(  # noqa: SLF001
            load_name_counts="auto",
            clusterer=clusterer_with_name_counts,
        )
        is True
    )
    assert (
        reranker_utils._resolve_load_name_counts(  # noqa: SLF001
            load_name_counts="auto",
            clusterer=clusterer_with_nameless_name_counts,
        )
        is True
    )
    assert (
        reranker_utils._resolve_load_name_counts(  # noqa: SLF001
            load_name_counts="auto",
            clusterer=clusterer_without_name_counts,
        )
        is False
    )
    assert reranker_utils._resolve_load_name_counts(load_name_counts="auto", clusterer=None) is True  # noqa: SLF001
    with pytest.raises(ValueError, match="Unknown load_name_counts mode"):
        reranker_utils._resolve_load_name_counts(load_name_counts="never", clusterer=None)  # noqa: SLF001


def test_filter_query_sequence_by_id_set_is_strict() -> None:
    queries = [
        SimpleNamespace(query_id="q2"),
        SimpleNamespace(query_id="q1"),
    ]
    filtered = reranker_build._filter_query_sequence_by_id_set(  # noqa: SLF001
        queries,
        selected_query_ids={"q1"},
    )
    assert [query.query_id for query in filtered] == ["q1"]
    with pytest.raises(ValueError, match="Unknown query IDs requested"):
        reranker_build._filter_query_sequence_by_id_set(  # noqa: SLF001
            queries,
            selected_query_ids={"q3"},
        )


def test_require_labeled_name_counts_source_rejects_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reranker_build,
        "inspect_json_ingest_name_counts_source",
        lambda dataset: {
            "name_counts_source": "none",
            "signatures_total": 10,
            "signatures_with_counts": 0,
            "artifact_configured": False,
            "rust_can_overlay_signature_counts": True,
        },
    )
    with pytest.raises(RuntimeError, match="name_counts_source=none"):
        reranker_build._require_labeled_name_counts_source(SimpleNamespace(), dataset_name="demo")  # noqa: SLF001


def test_compute_query_cluster_stats_batched_matches_single_query_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distance_by_pair = {
        ("q1", "s1"): 0.3,
        ("q1", "s2"): 0.1,
        ("q1", "s3"): 0.2,
        ("q2", "s4"): 0.6,
        ("q2", "s5"): 0.4,
        ("q2", "s6"): 0.5,
    }

    class _FakeClusterer:
        classifier = object()
        nameless_classifier = None
        featurizer_info = object()
        nameless_featurizer_info = None
        n_jobs = 1
        use_cache = False

        @staticmethod
        def _resolve_constraint_batch(  # noqa: PLR6301
            _dataset: Any,
            batch_pairs: list[tuple[str, str]],
            *,
            partial_supervision: dict[tuple[str, str], int | float],
            runtime_context: Any,
            incremental_dont_use_cluster_seeds: bool,
            constraint_backend: Any,
        ) -> tuple[list[float], dict[str, Any]]:
            del partial_supervision, runtime_context, incremental_dont_use_cluster_seeds, constraint_backend
            return [float("nan")] * len(batch_pairs), {}

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        features = np.asarray(
            [[distance_by_pair[(str(left), str(right))]] for left, right, _label in signature_pairs],
            dtype=np.float64,
        )
        labels = np.asarray([float("nan")] * len(signature_pairs), dtype=np.float64)
        return features, labels, None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        **_kwargs,
    ):
        del labels, _kwargs
        return np.asarray(features[:, 0], dtype=np.float64), 0.25

    monkeypatch.setattr(reranker_utils, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(reranker_utils, "_predict_and_combine", fake_predict_and_combine)

    request_one = reranker_utils.QueryClusterStatsRequest(
        query_signature_id="q1",
        shortlist_component_keys=("c1", "c2"),
        candidate_signature_ids_by_component={
            "c1": ["s1", "s2"],
            "c2": ["s3"],
        },
        retrieval_ranks={"c1": 1, "c2": 2},
        retrieval_scores={"c1": 0.9, "c2": 0.8},
        summary_by_component={
            "c1": build_cluster_summary(component_key="c1", size=2),
            "c2": build_cluster_summary(component_key="c2", size=1),
        },
    )
    request_two = reranker_utils.QueryClusterStatsRequest(
        query_signature_id="q2",
        shortlist_component_keys=("d1", "d2"),
        candidate_signature_ids_by_component={
            "d1": ["s4"],
            "d2": ["s5", "s6"],
        },
        retrieval_ranks={"d1": 1, "d2": 2},
        retrieval_scores={"d1": 0.7, "d2": 0.6},
        summary_by_component={
            "d1": build_cluster_summary(component_key="d1", size=1),
            "d2": build_cluster_summary(component_key="d2", size=2),
        },
    )

    clusterer = _FakeClusterer()
    single_one_stats, single_one_diag = reranker_utils.compute_query_cluster_stats(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        query_signature_id=request_one.query_signature_id,
        shortlist_component_keys=list(request_one.shortlist_component_keys),
        candidate_signature_ids_by_component=request_one.candidate_signature_ids_by_component,
        retrieval_ranks=request_one.retrieval_ranks,
        retrieval_scores=request_one.retrieval_scores,
        summary_by_component=request_one.summary_by_component,
        pair_batch_size=10,
        max_top_k=5,
    )
    single_two_stats, single_two_diag = reranker_utils.compute_query_cluster_stats(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        query_signature_id=request_two.query_signature_id,
        shortlist_component_keys=list(request_two.shortlist_component_keys),
        candidate_signature_ids_by_component=request_two.candidate_signature_ids_by_component,
        retrieval_ranks=request_two.retrieval_ranks,
        retrieval_scores=request_two.retrieval_scores,
        summary_by_component=request_two.summary_by_component,
        pair_batch_size=10,
        max_top_k=5,
    )
    batch_results = reranker_utils.compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        requests=[request_one, request_two],
        pair_batch_size=10,
        max_top_k=5,
    )

    assert len(batch_results) == 2
    for single_stats, single_diag, (batch_stats, batch_diag) in (
        (single_one_stats, single_one_diag, batch_results[0]),
        (single_two_stats, single_two_diag, batch_results[1]),
    ):
        assert batch_diag["pair_count"] == single_diag["pair_count"]
        for component_key, single_component_stats in single_stats.items():
            batch_component_stats = batch_stats[component_key]
            assert batch_component_stats.count == single_component_stats.count
            assert batch_component_stats.min_distance == pytest.approx(single_component_stats.min_distance)
            assert batch_component_stats.mean_distance == pytest.approx(single_component_stats.mean_distance)
            assert batch_component_stats.topk_mean_distance(3) == pytest.approx(
                single_component_stats.topk_mean_distance(3)
            )


def test_flush_prepared_query_requests_materializes_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c1"}),
        support_type="labeled",
        block_size=3,
        component_size=2,
        sampling_info_bucket="initial_only",
    )
    prepared_request = reranker_build.PreparedQueryRowsRequest(
        query_case=query_case,
        block_component_count=2,
        view_payloads=(
            reranker_build.PreparedViewPayload(
                query_view="initial_only",
                query=build_query_features(has_coauthors=True, has_affiliations=True),
                shortlist_component_keys=("c1", "c2"),
                retrieval_scores={"c1": 0.8, "c2": 0.7},
                retrieval_ranks={"c1": 1, "c2": 2},
                retrieval_window_state={
                    "scored_candidate_components": 2,
                    "scored_candidate_signatures": 3,
                    "orcid_filter_applied": 0,
                    "middle_initial_filter_applied": 0,
                    "year_range_filter_applied": 0,
                },
            ),
        ),
        union_summary_by_component={
            "c1": build_cluster_summary(component_key="c1", size=2),
            "c2": build_cluster_summary(component_key="c2", size=1),
        },
        retrieval_window_state_base={},
        stats_request=reranker_utils.QueryClusterStatsRequest(
            query_signature_id="q1",
            shortlist_component_keys=("c1", "c2"),
            candidate_signature_ids_by_component={"c1": ["s1", "s2"], "c2": ["s3"]},
            retrieval_ranks={"c1": 1, "c2": 2},
            retrieval_scores={"c1": 0.8, "c2": 0.7},
            summary_by_component={
                "c1": build_cluster_summary(component_key="c1", size=2),
                "c2": build_cluster_summary(component_key="c2", size=1),
            },
        ),
        estimated_pair_count=3,
    )
    stats_by_component = {
        "c1": reranker_utils.ClusterPairwiseStats(
            cluster_id="c1",
            retrieval_rank=1,
            retrieval_score=0.8,
            cluster_size=2,
            family_id="c1",
            count=2,
            sum_distance=0.4,
            min_distance=0.1,
            top_smallest_neg_heap=[-0.1, -0.3],
        ),
        "c2": reranker_utils.ClusterPairwiseStats(
            cluster_id="c2",
            retrieval_rank=2,
            retrieval_score=0.7,
            cluster_size=1,
            family_id="c2",
            count=1,
            sum_distance=0.5,
            min_distance=0.5,
            top_smallest_neg_heap=[-0.5],
        ),
    }
    monkeypatch.setattr(
        reranker_build,
        "compute_query_cluster_stats_batched",
        lambda **_kwargs: [
            (stats_by_component, {"pair_count": 3, "featurize_seconds": 1.5, "model_predict_seconds": 0.25})
        ],
    )

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    pair_counts: list[int] = []
    featurize_seconds: list[float] = []
    model_seconds: list[float] = []
    reranker_build._flush_prepared_query_requests(  # noqa: SLF001
        clusterer=SimpleNamespace(),
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        prepared_requests=[prepared_request],
        pair_batch_size=10,
        max_top_k=5,
        rows=rows,
        query_group_metadata_rows=metadata_rows,
        pair_counts=pair_counts,
        featurize_seconds=featurize_seconds,
        model_seconds=model_seconds,
    )

    assert len(rows) == 2
    assert len(metadata_rows) == 1
    assert pair_counts == [3]
    assert featurize_seconds == [1.5]
    assert model_seconds == [0.25]


def test_build_retrieval_window_supports_exemplar_method() -> None:
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

    centroid_ranked, _, _, _ = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[centroid_favorite, exemplar_favorite],
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
    )
    exemplar_ranked, _, _, _ = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[centroid_favorite, exemplar_favorite],
        max_block_component_size=4,
        retrieval_approach="all__hybrid_exemplar_4",
        max_ranked_clusters=2,
    )

    assert centroid_ranked == ["c1", "c2"]
    assert exemplar_ranked == ["c2", "c1"]


def test_build_retrieval_window_all_union_preserves_both_lanes() -> None:
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
        specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
        exemplar_vectors=[np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    shared_runner_up = build_cluster_summary(
        component_key="c3",
        size=4,
        specter_centroid=np.asarray([0.8, 0.6], dtype=np.float32),
        exemplar_vectors=[np.asarray([0.8, 0.6], dtype=np.float32)],
    )

    ranked_component_keys, _scores, retrieval_ranks, _state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[centroid_favorite, exemplar_favorite, shared_runner_up],
        max_block_component_size=4,
        retrieval_approach="all_union__hybrid_centroid__hybrid_exemplar_4",
        max_ranked_clusters=2,
    )

    assert ranked_component_keys == ["c1", "c2"]
    assert retrieval_ranks == {"c1": 1, "c2": 2}


def test_build_retrieval_window_ambiguous_union_expands_on_same_family_near_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features()
    primary_row = build_cluster_summary(component_key="c1", first_name_counts=Counter({"alice": 4}), size=4)
    challenger_row = build_cluster_summary(component_key="c2", first_name_counts=Counter({"alice": 4}), size=4)
    other_row = build_cluster_summary(component_key="c3", first_name_counts=Counter({"bob": 4}), size=4)

    monkeypatch.setattr(
        reranker_utils.retrieval,
        "apply_hard_filters",
        lambda query, candidate_summaries: (
            candidate_summaries,
            {
                "orcid_filter_applied": 0,
                "middle_initial_filter_applied": 0,
                "year_range_filter_applied": 0,
                "scored_candidate_components": len(candidate_summaries),
                "scored_candidate_signatures": sum(summary.size for summary in candidate_summaries),
            },
        ),
    )

    def fake_rank_top_summaries(*, method: str, **_kwargs: Any) -> list[tuple[float, Any]]:
        if method == "hybrid_centroid":
            return [(0.90, primary_row), (0.87, challenger_row), (0.30, other_row)]
        if method == "hybrid_exemplar_4":
            return [(0.91, challenger_row), (0.89, primary_row), (0.20, other_row)]
        raise AssertionError(f"Unexpected method {method!r}")

    monkeypatch.setattr(reranker_utils, "rank_top_summaries", fake_rank_top_summaries)

    centroid_ranked, _, _, _ = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[primary_row, challenger_row, other_row],
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=1,
    )
    ambiguous_ranked, _, _, _ = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[primary_row, challenger_row, other_row],
        max_block_component_size=4,
        retrieval_approach="ambiguous_union__hybrid_centroid__hybrid_exemplar_4",
        max_ranked_clusters=1,
    )

    assert centroid_ranked[0] == "c1"
    assert ambiguous_ranked[0] == "c2"


def test_build_retrieval_window_ambiguous_union_stays_on_primary_when_not_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features()
    primary_row = build_cluster_summary(component_key="c1", first_name_counts=Counter({"alice": 4}), size=4)
    challenger_row = build_cluster_summary(component_key="c2", first_name_counts=Counter({"bob": 4}), size=4)
    other_row = build_cluster_summary(component_key="c3", first_name_counts=Counter({"cara": 4}), size=4)

    monkeypatch.setattr(
        reranker_utils.retrieval,
        "apply_hard_filters",
        lambda query, candidate_summaries: (
            candidate_summaries,
            {
                "orcid_filter_applied": 0,
                "middle_initial_filter_applied": 0,
                "year_range_filter_applied": 0,
                "scored_candidate_components": len(candidate_summaries),
                "scored_candidate_signatures": sum(summary.size for summary in candidate_summaries),
            },
        ),
    )

    def fake_rank_top_summaries(*, method: str, **_kwargs: Any) -> list[tuple[float, Any]]:
        if method == "hybrid_centroid":
            return [(0.90, primary_row), (0.70, challenger_row), (0.30, other_row)]
        if method == "hybrid_exemplar_4":
            return [(0.95, challenger_row), (0.40, primary_row), (0.20, other_row)]
        raise AssertionError(f"Unexpected method {method!r}")

    monkeypatch.setattr(reranker_utils, "rank_top_summaries", fake_rank_top_summaries)

    ambiguous_ranked, _, _, _ = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=[primary_row, challenger_row, other_row],
        max_block_component_size=4,
        retrieval_approach="ambiguous_union__hybrid_centroid__hybrid_exemplar_4",
        max_ranked_clusters=1,
    )

    assert ambiguous_ranked[0] == "c1"


def test_write_pooled_shap_artifacts_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        s2and_ranker_eval,
        "_ranker_shap_values",
        lambda model, features: np.zeros_like(features, dtype=np.float32),
    )

    def fake_summary_plot(shap_values, X, feature_names, shap_plot_type, outpath, fig_num=None):
        del shap_values, X, feature_names, shap_plot_type, fig_num
        Path(outpath).write_bytes(b"png")

    monkeypatch.setattr(s2and_ranker_eval.shap_utils, "_safe_summary_plot", fake_summary_plot)  # noqa: SLF001

    shap_runs = [
        {
            "heldout_dataset": "d1",
            "model": object(),
            "rows": [
                _base_row(query_group_id="g1", query_id="q1", candidate_component_key="c1", label=1, retrieval_rank=1),
                _base_row(
                    query_group_id="g1",
                    query_id="q1",
                    candidate_component_key="c2",
                    label=0,
                    retrieval_rank=2,
                    best_competitor_component_key="c1",
                ),
            ],
        },
        {
            "heldout_dataset": "d2",
            "model": object(),
            "rows": [
                _base_row(query_group_id="g2", query_id="q2", candidate_component_key="c3", label=1, retrieval_rank=1),
                _base_row(
                    query_group_id="g2",
                    query_id="q2",
                    candidate_component_key="c4",
                    label=0,
                    retrieval_rank=2,
                    best_competitor_component_key="c3",
                ),
            ],
        },
    ]

    summary = s2and_ranker_eval._write_pooled_shap_artifacts(  # noqa: SLF001
        shap_runs=shap_runs,
        feature_preset="generalized_v8",
        output_dir=tmp_path,
        max_rows=3,
        seed=7,
        shap_plot_type="dot",
    )

    assert summary["enabled"] is True
    assert summary["selected_rows"] >= 2
    assert Path(summary["plot_path"]).exists()
    assert Path(summary["rows_path"]).exists()
    assert Path(summary["values_path"]).exists()
    loaded = np.load(summary["values_path"], allow_pickle=True)
    expected_feature_count = len(reranker_utils.resolve_feature_columns(feature_preset="generalized_v8"))
    assert loaded["shap_values"].shape[1] == expected_feature_count


def test_read_string_id_file_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "ids.txt"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected at least one ID"):
        reranker_build._read_string_id_file(path)  # noqa: SLF001


def test_build_dataset_main_rejects_labeled_limit_queries_plus_query_id_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_id_file = tmp_path / "query_ids.txt"
    query_id_file.write_text("q1\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_single_letter_reranker_dataset.py",
            "labeled",
            "--output-dir",
            str(tmp_path / "out"),
            "--limit-queries",
            "1",
            "--query-id-file",
            str(query_id_file),
        ],
    )
    with pytest.raises(ValueError, match="Use either --limit-queries or --query-id-file for `labeled`"):
        reranker_build.main()


def test_build_dataset_main_rejects_custom_query_views_for_any_input_h_wang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_single_letter_reranker_dataset.py",
            "h_wang",
            "--data-dir",
            str(tmp_path / "data"),
            "--step2-dir",
            str(tmp_path / "step2"),
            "--output-dir",
            str(tmp_path / "out"),
            "--query-source",
            "orcid_any_input",
            "--query-views",
            "full",
        ],
    )
    with pytest.raises(ValueError, match="orcid_any_input queries use their natural view"):
        reranker_build.main()


def test_build_h_wang_any_input_query_cases_assigns_expected_supervision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_map = {
        "q_pos": SimpleNamespace(has_full_first=True),
        "q_seeded": SimpleNamespace(has_full_first=False),
        "q_single": SimpleNamespace(has_full_first=True),
        "q_unresolved": SimpleNamespace(has_full_first=False),
        "orcid1_seed": SimpleNamespace(has_full_first=True),
        "orcid2_seedmate": SimpleNamespace(has_full_first=False),
        "orcid4_other": SimpleNamespace(has_full_first=False),
    }

    def fake_extract_query_features(
        dataset: Any,
        query_id: str,
        *,
        feature_cache: dict[str, Any],
        orcid_enabled: bool,
    ) -> Any:
        del dataset, feature_cache, orcid_enabled
        return feature_map[str(query_id)]

    monkeypatch.setattr(reranker_build.retrieval, "extract_query_features", fake_extract_query_features)
    raw_signatures = {
        "q_pos": {"author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0001"]}},
        "orcid1_seed": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0001"]}
        },
        "q_seeded": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0002"]}
        },
        "orcid2_seedmate": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0002"]}
        },
        "q_single": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0003"]}
        },
        "q_unresolved": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0004"]}
        },
        "orcid4_other": {
            "author_info": {"block": "b1", "source_id_source": "ORCID", "source_ids": ["0000-0000-0000-0004"]}
        },
    }
    dataset = SimpleNamespace(signatures={signature_id: object() for signature_id in raw_signatures})
    signature_to_cluster_id = {
        "orcid1_seed": "c1",
        "q_seeded": "c2",
        "orcid2_seedmate": "c2",
        "q_unresolved": "c3",
    }
    seed_cluster_counts_by_orcid = {
        "0000000000000001": Counter({"c1": 1}),
        "0000000000000002": Counter({"c2": 2}),
        "0000000000000004": Counter({"c3": 1}),
    }

    query_cases, summary = reranker_build._build_h_wang_any_input_query_cases(  # noqa: SLF001
        raw_signatures=raw_signatures,
        dataset=dataset,
        target_block="b1",
        signature_to_cluster_id=signature_to_cluster_id,
        seed_cluster_counts_by_orcid=seed_cluster_counts_by_orcid,
        limit_queries=None,
        query_id_file=None,
        seed=7,
    )

    cases_by_id = {query_case.query_id: query_case for query_case in query_cases}
    assert cases_by_id["q_pos"].supervision_type == "positive_repeat_orcid"
    assert cases_by_id["q_pos"].positive_component_keys == frozenset({"c1"})
    assert cases_by_id["q_pos"].natural_query_view == "full"
    assert cases_by_id["q_seeded"].supervision_type == "positive_repeat_orcid"
    assert cases_by_id["q_seeded"].positive_component_keys == frozenset({"c2"})
    assert cases_by_id["q_seeded"].query_in_seed_before_holdout is True
    assert cases_by_id["q_seeded"].natural_query_view == "initial_only"
    assert cases_by_id["q_single"].supervision_type == "negative_singleton_orcid"
    assert cases_by_id["q_single"].positive_component_keys == frozenset()
    assert cases_by_id["q_unresolved"].supervision_type == "unresolved_repeat_orcid"
    assert cases_by_id["q_unresolved"].positive_component_keys == frozenset()
    assert summary["supervision_type_counts"] == {
        key: int(value)
        for key, value in sorted(Counter(query_case.supervision_type for query_case in query_cases).items())
    }


def test_select_rows_supports_new_metadata_filters() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            query_view="full",
            query_source="orcid_any_input",
            supervision_type="positive_repeat_orcid",
            split="train",
        ),
        _base_row(
            query_group_id="g2",
            query_view="initial_only",
            query_source="orcid_any_input",
            supervision_type="negative_singleton_orcid",
            split="dev",
            retrieval_rank=2,
        ),
        _base_row(
            query_group_id="g3",
            query_source="labeled",
            supervision_type="labeled",
            split="all",
            retrieval_rank=3,
        ),
    ]
    selected = reranker_utils.select_rows(
        rows,
        query_views=["full", "initial_only"],
        query_sources=["orcid_any_input"],
        supervision_types=["negative_singleton_orcid"],
        splits=["dev"],
        window_size=2,
    )
    assert [row["query_group_id"] for row in selected] == ["g2"]


def test_select_reject_threshold_balances_positive_and_negative_dev_rows() -> None:
    rows = [
        {
            "query_group_id": "p1",
            "supervision_type": "positive_repeat_orcid",
            "model_margin": 0.9,
            "model_correct": 1,
            "query_view": "full",
        },
        {
            "query_group_id": "n1",
            "supervision_type": "negative_singleton_orcid",
            "model_margin": 0.2,
            "model_correct": 0,
            "query_view": "initial_only",
        },
    ]

    summary = s2and_ranker_eval._select_reject_threshold(rows)  # noqa: SLF001

    assert summary["threshold"] == pytest.approx(0.2)
    assert summary["balanced_accuracy"] == pytest.approx(1.0)
    assert summary["positive_accuracy"] == pytest.approx(1.0)
    assert summary["negative_reject_accuracy"] == pytest.approx(1.0)
    assert summary["per_view"]["full"]["positive_accuracy"] == pytest.approx(1.0)
    assert summary["per_view"]["initial_only"]["negative_reject_accuracy"] == pytest.approx(1.0)


def test_eval_single_letter_ranker_defaults_to_all_labeled_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_single_letter_ranker.py",
            "--dataset-root",
            "rows",
            "--window-size",
            "25",
            "--output-dir",
            "out",
        ],
    )
    parsed = s2and_ranker_eval.parse_args()
    assert parsed.datasets == list(reranker_utils.DEFAULT_LABELED_DATASETS)
    assert parsed.feature_preset == "generalized_v8"
    assert parsed.hyperopt_evals is None
    assert parsed.training_source_mode == "s2and_only"


def test_resolve_hyperopt_evals_defaults_screen_to_zero() -> None:
    assert (
        s2and_ranker_eval._resolve_hyperopt_evals(  # noqa: SLF001
            requested_hyperopt_evals=None,
            run_mode="screen",
        )
        == 0
    )
    assert (
        s2and_ranker_eval._resolve_hyperopt_evals(  # noqa: SLF001
            requested_hyperopt_evals=None,
            run_mode="full",
        )
        == 20
    )
    assert (
        s2and_ranker_eval._resolve_hyperopt_evals(  # noqa: SLF001
            requested_hyperopt_evals=7,
            run_mode="screen",
        )
        == 7
    )


def test_summarize_query_group_rows_exposes_cached_sampler_fields() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            query_id="q1",
            candidate_component_key="c1",
            family_id="fam1",
            label=0,
            retrieval_rank=1,
            candidate_count=3,
            candidate_signatures=9,
            scored_candidate_components=3,
            scored_candidate_signatures=9,
            positive_candidate_count=1,
            positive_candidate_keys="c2",
            best_positive_retrieval_rank=2,
            support_type="labeled",
            confident_family_flag=1,
        ),
        _base_row(
            query_group_id="g1",
            query_id="q1",
            candidate_component_key="c2",
            family_id="fam2",
            label=1,
            retrieval_rank=2,
            candidate_count=3,
            candidate_signatures=9,
            scored_candidate_components=3,
            scored_candidate_signatures=9,
            positive_candidate_count=1,
            positive_candidate_keys="c2",
            best_positive_retrieval_rank=2,
            support_type="labeled",
            confident_family_flag=1,
            best_competitor_component_key="c1",
        ),
    ]
    summary = reranker_utils.summarize_query_group_rows(rows, block_component_count=7)
    assert summary["query_group_id"] == "g1"
    assert summary["block_component_count"] == 7
    assert summary["candidate_count"] == 3
    assert summary["best_positive_retrieval_rank"] == 2
    assert summary["recoverable_non_top1"] == 1
    assert summary["cross_family_top1_vs_positive"] == 1
    assert summary["retrieval_top1_component_key"] == "c1"


def test_hard_blocks_sampler_prefers_recoverable_cross_family_cases() -> None:
    metadata_rows = [
        {
            "dataset": "d1",
            "query_group_id": "q_easy",
            "query_id": "q_easy",
            "block_size": 40,
            "block_component_count": 5,
            "component_size": 4,
            "sampling_info_bucket": "rich",
            "candidate_count": 5,
            "best_positive_retrieval_rank": 1,
            "best_positive_rank_bucket": "1",
            "recoverable_non_top1": 0,
            "cross_family_top1_vs_positive": 0,
        },
        {
            "dataset": "d1",
            "query_group_id": "q_cross",
            "query_id": "q_cross",
            "block_size": 80,
            "block_component_count": 12,
            "component_size": 6,
            "sampling_info_bucket": "rich",
            "candidate_count": 25,
            "best_positive_retrieval_rank": 3,
            "best_positive_rank_bucket": "2_3",
            "recoverable_non_top1": 1,
            "cross_family_top1_vs_positive": 1,
        },
        {
            "dataset": "d1",
            "query_group_id": "q_recoverable",
            "query_id": "q_recoverable",
            "block_size": 60,
            "block_component_count": 10,
            "component_size": 5,
            "sampling_info_bucket": "metadata_only",
            "candidate_count": 20,
            "best_positive_retrieval_rank": 6,
            "best_positive_rank_bucket": "4_10",
            "recoverable_non_top1": 1,
            "cross_family_top1_vs_positive": 0,
        },
    ]
    selected, summary = reranker_build._select_query_group_metadata_rows(  # noqa: SLF001
        metadata_rows,
        labeled_query_sampler="hard_blocks_v1",
        limit_query_groups=2,
        seed=7,
    )
    assert [row["query_group_id"] for row in selected] == ["q_cross", "q_recoverable"]
    assert summary["selected_recoverable_non_top1_count"] == 2
    assert summary["selected_cross_family_top1_vs_positive_count"] == 1


def test_hard_blocks_sampler_filters_tiny_and_single_candidate_cases() -> None:
    metadata_rows = [
        {
            "dataset": "d1",
            "query_group_id": "q_small_block",
            "query_id": "q_small_block",
            "block_size": 5,
            "block_component_count": 3,
            "component_size": 3,
            "sampling_info_bucket": "rich",
            "candidate_count": 10,
            "best_positive_retrieval_rank": 2,
            "best_positive_rank_bucket": "2_3",
            "recoverable_non_top1": 1,
            "cross_family_top1_vs_positive": 1,
        },
        {
            "dataset": "d1",
            "query_group_id": "q_single_candidate",
            "query_id": "q_single_candidate",
            "block_size": 50,
            "block_component_count": 3,
            "component_size": 3,
            "sampling_info_bucket": "rich",
            "candidate_count": 1,
            "best_positive_retrieval_rank": 1,
            "best_positive_rank_bucket": "1",
            "recoverable_non_top1": 0,
            "cross_family_top1_vs_positive": 0,
        },
        {
            "dataset": "d1",
            "query_group_id": "q_keep",
            "query_id": "q_keep",
            "block_size": 50,
            "block_component_count": 3,
            "component_size": 3,
            "sampling_info_bucket": "rich",
            "candidate_count": 8,
            "best_positive_retrieval_rank": 2,
            "best_positive_rank_bucket": "2_3",
            "recoverable_non_top1": 1,
            "cross_family_top1_vs_positive": 0,
        },
    ]
    selected, summary = reranker_build._select_query_group_metadata_rows(  # noqa: SLF001
        metadata_rows,
        labeled_query_sampler="hard_blocks_v1",
        limit_query_groups=None,
        seed=7,
    )
    assert [row["query_group_id"] for row in selected] == ["q_keep"]
    assert summary["filtered_small_block_count"] == 1
    assert summary["filtered_single_candidate_preview_count"] == 1


def test_materialized_derived_rows_are_used_directly_by_feature_builder() -> None:
    rows = [
        _base_row(query_group_id="g1", candidate_component_key="c1", retrieval_rank=1),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            label=0,
            retrieval_rank=2,
            best_competitor_component_key="c1",
        ),
    ]
    materialized_rows = reranker_utils.materialize_derived_rows(rows)
    materialized_rows[0]["override_slack_vs_top1"] = 123.0
    matrix = reranker_utils.build_feature_matrix(
        materialized_rows,
        feature_columns=("override_slack_vs_top1",),
    )
    assert matrix[0, 0] == pytest.approx(123.0)


def test_load_dataset_rows_prefers_derived_cache_and_selected_ids(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "d1"
    base_rows = [
        _base_row(query_group_id="g1", candidate_component_key="c1", retrieval_rank=1),
        _base_row(
            query_group_id="g2",
            candidate_component_key="c2",
            retrieval_rank=1,
            query_id="q2",
            best_competitor_component_key="c1",
        ),
    ]
    reranker_utils.write_rows_csv(dataset_dir / "rows.csv", base_rows)
    derived_rows = reranker_utils.materialize_derived_rows(base_rows)
    derived_rows[0]["override_slack_vs_top1"] = 77.0
    reranker_utils.write_materialized_rows_csv(dataset_dir / "rows_derived.csv", derived_rows)
    rows, rows_source_by_dataset = s2and_ranker_eval._load_dataset_rows(  # noqa: SLF001
        tmp_path,
        ["d1"],
        rows_source="auto",
        selected_query_group_ids={"g1"},
    )
    assert rows_source_by_dataset == {"d1": "derived"}
    assert len(rows) == 1
    assert rows[0]["query_group_id"] == "g1"
    assert rows[0]["override_slack_vs_top1"] == pytest.approx(77.0)
