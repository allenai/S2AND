from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.eval_single_letter_ranker as s2and_ranker_eval
import scripts.giant_block_cluster_retrieval_task as giant_task
import scripts.reranker_dataset as reranker_dataset
import scripts.reranker_dataset.build as reranker_build
import scripts.single_letter_reranker_utils as reranker_utils
import scripts.single_letter_retrieval_utils as retrieval_utils
from s2and.data import NameCounts
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
        "query_first_token": "alice",
        "query_first_initial": "a",
        "query_year": 2000,
        "_audit_normalized_orcid": None,
        "_audit_orcid_group_size": None,
        "_audit_orcid_group_size_bucket": None,
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
        "candidate_year_min": 1999,
        "candidate_year_max": 2001,
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
        "same_family_as_top1": 1,
        "middle_initial_compatibility": 1.0,
        "affiliation_overlap": 0.5,
        "coauthor_overlap": 0.5,
        "venue_overlap": 0.0,
        "year_compatibility": 1.0,
        "title_overlap": 0.0,
        "specter_centroid_similarity": 0.0,
        "specter_exemplar_similarity": 0.0,
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


def test_audit_columns_not_in_any_preset() -> None:
    audit_columns = set(reranker_utils.AUDIT_ORCID_METADATA_COLUMNS)
    active_preset_columns = {
        column for feature_columns in reranker_utils.FEATURE_PRESETS.values() for column in feature_columns
    }
    legacy_columns = {"normalized_orcid", "orcid_group_size", "orcid_group_size_bucket"}

    assert audit_columns <= set(reranker_utils.ROW_COLUMNS)
    assert audit_columns <= set(reranker_utils.QUERY_GROUP_METADATA_COLUMNS)
    assert "_audit_orcid_group_size" in reranker_utils.INT_COLUMNS
    assert "_audit_orcid_group_size" in reranker_utils.QUERY_GROUP_METADATA_INT_COLUMNS
    assert audit_columns.isdisjoint(active_preset_columns)
    assert all(column not in reranker_utils.NUMERIC_FEATURE_COLUMNS for column in audit_columns)
    assert legacy_columns.isdisjoint(reranker_utils.ROW_COLUMNS)
    assert legacy_columns.isdisjoint(reranker_utils.QUERY_GROUP_METADATA_COLUMNS)


def test_chooser_compatibility_helpers_use_retrieval_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_utils.retrieval, "RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE", -0.75)
    monkeypatch.setattr(reranker_utils.retrieval, "RETRIEVAL_YEAR_SCORE_DECAY_YEARS", 20.0)
    monkeypatch.setattr(reranker_utils.retrieval, "RETRIEVAL_YEAR_SCORE_RANGE_GAP", 5)
    monkeypatch.setattr(reranker_utils.retrieval, "RETRIEVAL_YEAR_SCORE_RANGE_PENALTY", 0.2)
    query = build_query_features(middle_initials=frozenset({"a"}))
    conflicting_summary = build_cluster_summary(
        component_key="conflict",
        size=4,
        middle_initial_counts=Counter({"b": 4}),
    )
    dated_summary = build_cluster_summary(
        component_key="dated",
        size=4,
        year_min=2000,
        year_max=2002,
        year_mean=2000.0,
    )

    assert reranker_utils._middle_initial_compatibility(query, conflicting_summary) == pytest.approx(  # noqa: SLF001
        -0.75
    )
    assert reranker_utils._year_compatibility(2010, dated_summary) == pytest.approx(0.3)  # noqa: SLF001


def test_classify_subblocks_partitions_mixed_first_name_lengths() -> None:
    dataset = SimpleNamespace(
        signatures={
            "s_initial": SimpleNamespace(
                author_info_first_normalized_without_apostrophe="a",
                author_info_first="A",
            ),
            "s_full": SimpleNamespace(
                author_info_first_normalized_without_apostrophe="alice",
                author_info_first="Alice",
            ),
            "s_empty": SimpleNamespace(
                author_info_first_normalized_without_apostrophe="",
                author_info_first="",
            ),
        }
    )

    multi_letter, single_letter = giant_task._classify_subblocks(  # noqa: SLF001
        {
            "full": ["s_full"],
            "initial": ["s_initial"],
            "mixed": ["s_initial", "s_full", "s_empty"],
        },
        dataset,
    )

    assert multi_letter == {
        "full": ["s_full"],
        "mixed::multi_letter": ["s_full"],
    }
    assert single_letter == {
        "initial": ["s_initial"],
        "mixed::single_letter": ["s_initial", "s_empty"],
    }


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


def test_default_feature_preset_uses_official_best21_bundle() -> None:
    assert reranker_utils.DEFAULT_FEATURE_PRESET == "classic_best21"
    assert reranker_utils.resolve_feature_columns(feature_preset=reranker_utils.DEFAULT_FEATURE_PRESET) == (
        "min_distance",
        "top3_distance_best_gap",
        "retrieval_rank",
        "top20_mean_distance",
        "top3_gap_to_heuristic_choice",
        "top1_strongest_contradiction",
        "retrieval_score",
        "title_overlap",
        "retrieval_top1_score",
        "specter_exemplar_similarity",
        "initial_only_x_venue_overlap",
        "count_normalized_confidence",
        "distance_spread_mean_minus_top5",
        "affiliation_contradiction_severity",
        "cluster_size_log_capped",
        "coauthor_gap_to_best_same_coarse_family",
        "near_tied_alternative_count",
        "pair_count",
        "retrieval_score_rank_fraction",
        "top3_distance_rank_fraction",
        "middle_initial_compatibility",
    )


def test_load_preferred_signature_to_cluster_id_prefers_reconciled(tmp_path: Path) -> None:
    step2_dir = tmp_path / "step2"
    step2_dir.mkdir()
    (step2_dir / retrieval_utils.RAW_SIGNATURE_TO_CLUSTER_ID_FILENAME).write_text(
        json.dumps({"q1": "c_old", "m1": "c_old", "m2": "c_new"}),
        encoding="utf-8",
    )
    (step2_dir / retrieval_utils.RECONCILED_SIGNATURE_TO_CLUSTER_ID_FILENAME).write_text(
        json.dumps({"q1": "c_new", "m1": "c_old", "m2": "c_new"}),
        encoding="utf-8",
    )
    (step2_dir / retrieval_utils.RECONCILED_SIGNATURE_TO_CLUSTER_ID_SUMMARY_FILENAME).write_text(
        json.dumps({"queries_moved": 1}),
        encoding="utf-8",
    )

    mapping, info = retrieval_utils.load_preferred_signature_to_cluster_id(step2_dir)

    assert mapping == {"q1": "c_new", "m1": "c_old", "m2": "c_new"}
    assert info["source"] == "reconciled"
    assert info["assignment_count"] == 3
    assert str(info["path"]).endswith(retrieval_utils.RECONCILED_SIGNATURE_TO_CLUSTER_ID_FILENAME)
    assert str(info["summary_path"]).endswith(retrieval_utils.RECONCILED_SIGNATURE_TO_CLUSTER_ID_SUMMARY_FILENAME)


def test_materialize_derived_rows_adds_feature_gap_columns() -> None:
    rows = [
        _base_row(
            block_key="a smith",
            query_view="initial_only",
            query_first_token="a",
            query_first_initial="a",
            query_year=2000,
            candidate_component_key="c1",
            dominant_first_name="alice",
            retrieval_rank=1,
            retrieval_score=0.92,
            cluster_size=6,
            count_normalized_confidence=0.92,
            title_overlap=1.0,
            coauthor_overlap=0.5,
            affiliation_overlap=0.0,
            venue_overlap=0.7,
            year_compatibility=0.0,
            candidate_year_min=2012,
            candidate_year_max=2015,
        ),
        _base_row(
            block_key="a smith",
            query_view="initial_only",
            query_first_token="a",
            query_first_initial="a",
            query_year=2000,
            candidate_component_key="c2",
            dominant_first_name="adam",
            family_id="fam2",
            best_competitor_component_key="c1",
            retrieval_rank=2,
            retrieval_score=0.91,
            cluster_size=4,
            count_normalized_confidence=0.81,
            title_overlap=0.85,
            coauthor_overlap=0.6,
            affiliation_overlap=0.7,
            venue_overlap=0.3,
            year_compatibility=0.8,
            candidate_year_min=1998,
            candidate_year_max=2002,
            label=0,
        ),
        _base_row(
            block_key="a smith",
            query_view="initial_only",
            query_first_token="a",
            query_first_initial="a",
            query_year=2000,
            candidate_component_key="c3",
            dominant_first_name="brian",
            family_id="fam3",
            best_competitor_component_key="c1",
            retrieval_rank=3,
            retrieval_score=0.905,
            cluster_size=5,
            count_normalized_confidence=0.75,
            title_overlap=1.0,
            coauthor_overlap=0.0,
            affiliation_overlap=0.0,
            venue_overlap=0.1,
            year_compatibility=0.0,
            candidate_year_min=2018,
            candidate_year_max=2019,
            label=0,
        ),
    ]

    materialized = reranker_utils.materialize_derived_rows(rows)
    by_key = {row["candidate_component_key"]: row for row in materialized}
    top1 = by_key["c1"]
    conflict = by_key["c3"]

    assert top1["retrieval_top1_score"] == pytest.approx(0.92)
    assert top1["retrieval_top1_margin"] == pytest.approx(0.01)
    assert top1["near_tied_alternative_count"] == pytest.approx(2.0)
    assert top1["top1_minus_runnerup_title_overlap"] == pytest.approx(0.15)
    assert top1["top1_minus_runnerup_coauthor_overlap"] == pytest.approx(-0.1)
    assert top1["top1_minus_runnerup_venue_overlap"] == pytest.approx(0.4)
    assert top1["top1_minus_runnerup_year_compatibility"] == pytest.approx(-0.8)
    assert top1["top1_minus_runnerup_count_normalized_confidence"] == pytest.approx(0.11)
    assert top1["top1_minus_runnerup_cluster_size"] == pytest.approx(2.0)
    assert top1["exact_anchor_evidence_flag"] == pytest.approx(1.0)
    assert top1["top1_exact_anchor_evidence_flag"] == pytest.approx(1.0)
    assert top1["initial_only_x_title_overlap"] == pytest.approx(1.0)
    assert top1["year_mismatch_severity"] == pytest.approx(1.0)
    assert top1["affiliation_contradiction_severity"] == pytest.approx(1.0)
    assert top1["candidate_contradiction_count"] == pytest.approx(2.0)
    assert top1["candidate_contradiction_score"] == pytest.approx(1.0)
    assert top1["exact_title_identity_conflict_flag"] == pytest.approx(1.0)
    assert top1["top1_contradiction_count"] == pytest.approx(2.0)
    assert top1["top1_strongest_contradiction"] == pytest.approx(1.0)
    assert top1["top1_exact_title_identity_conflict_flag"] == pytest.approx(1.0)
    assert top1["plausible_conflicting_candidate_count"] == pytest.approx(1.0)
    assert top1["anchor_evidence_count"] == pytest.approx(5.0)
    assert top1["strong_positive_anchor_score"] == pytest.approx(0.1722)
    assert top1["weak_residual_anchor_score"] == pytest.approx(0.54)
    assert top1["sparse_relative_winner_score"] == pytest.approx(0.0)
    assert top1["cluster_size_log_capped"] == pytest.approx(0.369756)
    assert top1["query_view__full"] == pytest.approx(0.0)
    assert top1["query_view__initial_only"] == pytest.approx(1.0)
    assert conflict["exact_title_identity_conflict_flag"] == pytest.approx(1.0)


def test_build_feature_matrix_neutralizes_blank_candidate_year_bounds() -> None:
    rows = [
        _base_row(
            query_year=2020,
            candidate_year_min="",
            candidate_year_max="",
            year_compatibility=0.25,
        ),
        _base_row(
            query_year="",
            candidate_year_min=1990,
            candidate_year_max=1995,
            year_compatibility=0.25,
        ),
    ]

    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=("year_mismatch_severity",))

    assert matrix.shape == (2, 1)
    assert matrix[:, 0] == pytest.approx([0.0, 0.0])


def test_build_feature_matrix_scores_complete_year_range_contradictions() -> None:
    rows = [
        _base_row(
            query_year=2020,
            candidate_year_min=1990,
            candidate_year_max=1995,
            year_compatibility=1.0,
        )
    ]

    matrix = reranker_utils.build_feature_matrix(rows, feature_columns=("year_mismatch_severity",))

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(1.0)


def test_rank_fraction_map_shares_ties_and_neutralizes_all_equal_groups() -> None:
    rows = [
        _base_row(candidate_component_key="c1", retrieval_rank=1, affiliation_overlap=0.6, coauthor_overlap=0.0),
        _base_row(candidate_component_key="c2", retrieval_rank=2, affiliation_overlap=0.6, coauthor_overlap=0.0),
        _base_row(candidate_component_key="c3", retrieval_rank=3, affiliation_overlap=0.1, coauthor_overlap=0.0),
    ]

    tied = reranker_utils._rank_fraction_map(rows, column="affiliation_overlap", higher_is_better=True)
    assert tied[("g1", "c1")] == 0.25
    assert tied[("g1", "c2")] == 0.25
    assert tied[("g1", "c3")] == 1.0

    all_equal = reranker_utils._rank_fraction_map(rows, column="coauthor_overlap", higher_is_better=True)
    assert all_equal == {("g1", "c1"): 0.5, ("g1", "c2"): 0.5, ("g1", "c3"): 0.5}


def test_count_normalized_confidence_uses_top5_quality_and_sqrt_support() -> None:
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=9,
        count=9,
        sum_distance=1.3,
        min_distance=0.05,
        top_smallest_neg_heap=[-0.05, -0.1, -0.15, -0.4, -0.6],
    )

    actual = reranker_utils.count_normalized_confidence(stats, max_pair_count_in_group=99)

    expected_support = np.sqrt(np.log1p(9.0) / np.log1p(99.0))
    expected_quality = 1.0 - ((0.05 + 0.1 + 0.15 + 0.4 + 0.6) / 5.0)
    assert actual == pytest.approx(expected_support * expected_quality)


def test_materialize_derived_rows_uses_wider_cross_family_heuristic_margin() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            candidate_component_key="c1",
            family_id="fam1",
            retrieval_rank=1,
            top3_mean_distance=0.255,
            top5_mean_distance=0.255,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            retrieval_score=0.79,
            top3_mean_distance=0.21,
            top5_mean_distance=0.21,
            label=0,
            best_competitor_component_key="c1",
        ),
    ]

    materialized = reranker_utils.materialize_derived_rows(rows)
    by_key = {row["candidate_component_key"]: row for row in materialized}
    top1 = by_key["c1"]

    assert top1["heuristic_cross_family_top1_vs_best_top5"] == pytest.approx(1.0)
    assert top1["heuristic_top1_vs_best_top5_margin"] == pytest.approx(0.045)
    assert top1["heuristic_margin_threshold"] == pytest.approx(0.05)
    assert top1["heuristic_margin_slack"] == pytest.approx(-0.005)
    assert top1["heuristic_prefers_top1"] == pytest.approx(1.0)


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
        family_dominance_ratio=0.9,
        family_named_count=3,
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
        family_dominance_ratio=0.8,
        family_named_count=2,
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


def test_make_candidate_rows_ports_pairwise_name_count_rarity_features() -> None:
    query = build_query_features(
        first="alice",
        has_full_first=True,
        name_counts=NameCounts(first=10, first_last=4, last=100, last_first_initial=20),
    )
    summary = build_cluster_summary(
        component_key="c1",
        size=2,
        first_name_counts=Counter({"alice": 2}),
        name_counts_values=(
            NameCounts(first=8, first_last=4, last=100, last_first_initial=15),
            NameCounts(first=1000, first_last=900, last=100, last_first_initial=80),
        ),
    )
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=2,
        family_id="fam1",
        family_dominance_ratio=1.0,
        family_named_count=2,
        count=2,
        sum_distance=0.3,
        min_distance=0.1,
        top_smallest_neg_heap=[-0.1, -0.2],
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
        component_size=2,
        sampling_info_bucket="rich",
    )

    rows = reranker_utils.make_candidate_rows(
        query_case=query_case,
        query_view="full",
        query_features=query,
        shortlist_component_keys=["c1"],
        retrieval_scores={"c1": 0.8},
        retrieval_ranks={"c1": 1},
        retrieval_window_state={
            "scored_candidate_components": 1,
            "scored_candidate_signatures": 2,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        summary_by_component={"c1": summary},
        stats_by_component={"c1": stats},
    )

    row = rows[0]
    assert row["first_name_count_min_rarity"] == pytest.approx(1.0 / np.sqrt(8.0), abs=1e-6)
    assert row["last_first_name_count_min_rarity"] == pytest.approx(0.5)
    assert row["last_name_count_min_rarity"] == pytest.approx(0.1)
    assert row["last_first_initial_count_min_rarity"] == pytest.approx(1.0 / np.sqrt(15.0), abs=1e-6)
    assert row["first_name_count_max_rarity"] == pytest.approx(1.0 / np.sqrt(10.0), abs=1e-6)
    assert row["last_first_name_count_max_rarity"] == pytest.approx(0.5)
    assert row["first_prefix_x_last_first_name_count_min_rarity"] == pytest.approx(0.5)
    assert row["candidate_first_name_count_min_rarity"] == pytest.approx(1.0 / np.sqrt(8.0), abs=1e-6)
    assert row["candidate_last_first_name_count_min_rarity"] == pytest.approx(0.5)
    assert row["candidate_last_name_count_min_rarity"] == pytest.approx(0.1)
    assert row["candidate_last_first_initial_count_min_rarity"] == pytest.approx(1.0 / np.sqrt(15.0), abs=1e-6)


def test_make_candidate_rows_does_not_leak_full_first_rarity_for_initial_only_queries() -> None:
    query = build_query_features(
        first="a",
        has_full_first=False,
        name_counts=NameCounts(first=2, first_last=1, last=25, last_first_initial=1),
    )
    summary = build_cluster_summary(
        component_key="c1",
        size=1,
        first_name_counts=Counter({"alice": 1}),
        name_counts_values=(NameCounts(first=2, first_last=1, last=25, last_first_initial=1),),
    )
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=1,
        count=1,
        sum_distance=0.1,
        min_distance=0.1,
        top_smallest_neg_heap=[-0.1],
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
        component_size=1,
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
            "scored_candidate_signatures": 1,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        summary_by_component={"c1": summary},
        stats_by_component={"c1": stats},
    )

    row = rows[0]
    assert row["first_name_count_min_rarity"] == pytest.approx(0.0)
    assert row["last_first_name_count_min_rarity"] == pytest.approx(0.0)
    assert row["last_first_initial_count_min_rarity"] == pytest.approx(0.0)
    assert row["first_name_count_max_rarity"] == pytest.approx(0.0)
    assert row["last_first_name_count_max_rarity"] == pytest.approx(0.0)
    assert row["first_prefix_x_last_first_name_count_min_rarity"] == pytest.approx(0.0)
    assert row["last_name_count_min_rarity"] == pytest.approx(0.2)
    assert row["candidate_first_name_count_min_rarity"] == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-6)
    assert row["candidate_last_first_name_count_min_rarity"] == pytest.approx(1.0)


def test_make_candidate_rows_uses_zero_competitor_gaps_for_single_candidate_groups() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary = build_cluster_summary(component_key="c1", size=3)
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        family_dominance_ratio=0.9,
        family_named_count=3,
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


def test_make_candidate_rows_materializes_raw_similarity_features() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary = build_cluster_summary(component_key="c1", size=3)
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        family_dominance_ratio=0.9,
        family_named_count=3,
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
        raw_similarity_features_by_component={
            "c1": {
                "raw_max_affiliation_jaccard": 0.125,
                "raw_max_coauthor_jaccard": 0.25,
                "raw_max_title_jaccard": 1 / 3,
                "raw_max_text_jaccard": 0.5,
            }
        },
    )

    assert rows[0]["raw_max_affiliation_jaccard"] == pytest.approx(0.125)
    assert rows[0]["raw_max_coauthor_jaccard"] == pytest.approx(0.25)
    assert rows[0]["raw_max_title_jaccard"] == pytest.approx(0.333333)
    assert rows[0]["raw_max_text_jaccard"] == pytest.approx(0.5)
    feature_matrix = reranker_utils.build_feature_matrix(
        rows,
        feature_columns=reranker_utils.RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    )
    assert feature_matrix.tolist()[0] == pytest.approx([0.125, 0.25, 0.333333, 0.5])


def test_reranker_dataset_bridge_candidate_rows_match_legacy_csv_bytes(tmp_path: Path) -> None:
    query = build_query_features(
        first="alice",
        year=2020,
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
        has_coauthors=True,
        has_affiliations=True,
        has_full_first=True,
        title_terms=frozenset({"graph", "model"}),
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
        component_size=2,
        sampling_info_bucket="rich",
        normalized_orcid="0000000000000001",
        orcid_group_size=2,
        orcid_group_size_bucket="2",
        split="train",
    )
    summary_by_component = {
        "c1": build_cluster_summary(
            component_key="c1",
            size=2,
            first_name_counts=Counter({"alice": 2}),
            coauthor_counts=Counter({"a smith": 1}),
            affiliation_counts=Counter({"lab": 1}),
            title_counts=Counter({"graph": 2, "model": 1}),
            year_min=2019,
            year_max=2021,
            specter_centroid=np.asarray([0.9, 0.1], dtype=np.float32),
        ),
        "c2": build_cluster_summary(
            component_key="c2",
            size=1,
            first_name_counts=Counter({"anna": 1}),
            title_counts=Counter({"network": 1}),
            year_min=2001,
            year_max=2001,
            specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
        ),
    }
    stats_by_component = {
        "c1": reranker_utils.ClusterPairwiseStats(
            cluster_id="c1",
            retrieval_rank=1,
            retrieval_score=0.81,
            cluster_size=2,
            family_id="fam1",
            family_dominance_ratio=1.0,
            family_named_count=2,
            count=2,
            sum_distance=0.4,
            min_distance=0.1,
            top_smallest_neg_heap=[-0.1, -0.3],
        ),
        "c2": reranker_utils.ClusterPairwiseStats(
            cluster_id="c2",
            retrieval_rank=2,
            retrieval_score=0.42,
            cluster_size=1,
            family_id="fam2",
            family_dominance_ratio=1.0,
            family_named_count=1,
            count=1,
            sum_distance=0.7,
            min_distance=0.7,
            top_smallest_neg_heap=[-0.7],
        ),
    }
    shortlist_component_keys = ["c1", "c2"]
    retrieval_scores = {"c1": 0.81, "c2": 0.42}
    retrieval_ranks = {"c1": 1, "c2": 2}
    retrieval_window_state = {
        "scored_candidate_components": 2,
        "scored_candidate_signatures": 3,
        "orcid_filter_applied": 0,
        "middle_initial_filter_applied": 0,
        "year_range_filter_applied": 1,
    }
    raw_similarity_features_by_component = {
        "c1": {
            "raw_max_affiliation_jaccard": 0.5,
            "raw_max_coauthor_jaccard": 0.25,
            "raw_max_title_jaccard": 0.75,
            "raw_max_text_jaccard": 0.125,
        },
        "c2": {
            "raw_max_affiliation_jaccard": 0.0,
            "raw_max_coauthor_jaccard": 0.0,
            "raw_max_title_jaccard": 0.1,
            "raw_max_text_jaccard": 0.2,
        },
    }

    legacy_rows = reranker_utils.make_candidate_rows(
        query_case=query_case,
        query_view="initial_only",
        query_features=query,
        shortlist_component_keys=shortlist_component_keys,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
        retrieval_window_state=retrieval_window_state,
        summary_by_component=summary_by_component,
        stats_by_component=stats_by_component,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )
    bridge_rows = reranker_dataset.generate_candidate_rows(
        query_case=query_case,
        query_view="initial_only",
        query_features=query,
        shortlist_component_keys=shortlist_component_keys,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
        retrieval_window_state=retrieval_window_state,
        summary_by_component=summary_by_component,
        stats_by_component=stats_by_component,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )
    prepared_request = reranker_build.PreparedQueryRowsRequest(
        query_case=query_case,
        block_component_count=2,
        view_payloads=(
            reranker_build.PreparedViewPayload(
                query_view="initial_only",
                query=query,
                shortlist_component_keys=tuple(shortlist_component_keys),
                retrieval_scores=retrieval_scores,
                retrieval_ranks=retrieval_ranks,
                retrieval_window_state=retrieval_window_state,
            ),
        ),
        union_summary_by_component=summary_by_component,
        retrieval_window_state_base={},
        stats_request=reranker_utils.QueryClusterStatsRequest(
            query_signature_id="q1",
            shortlist_component_keys=tuple(shortlist_component_keys),
            candidate_signature_ids_by_component={"c1": ["s1", "s2"], "c2": ["s3"]},
            retrieval_ranks=retrieval_ranks,
            retrieval_scores=retrieval_scores,
            summary_by_component=summary_by_component,
        ),
        estimated_pair_count=3,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )
    prepared_rows = reranker_build._materialize_query_rows_from_prepared(  # noqa: SLF001
        prepared_request,
        stats_by_component=stats_by_component,
    )

    legacy_path = tmp_path / "legacy.csv"
    bridge_path = tmp_path / "bridge.csv"
    prepared_path = tmp_path / "prepared.csv"
    reranker_utils.write_rows_csv(legacy_path, legacy_rows)
    reranker_utils.write_rows_csv(bridge_path, bridge_rows)
    reranker_utils.write_rows_csv(prepared_path, prepared_rows)

    assert bridge_rows == legacy_rows
    assert prepared_rows == legacy_rows
    assert bridge_path.read_bytes() == legacy_path.read_bytes()
    assert prepared_path.read_bytes() == legacy_path.read_bytes()


def test_make_candidate_rows_filters_components_with_any_disallow_pair() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary_a = build_cluster_summary(component_key="c1", size=3)
    summary_b = build_cluster_summary(component_key="c2", size=2)
    stats_a = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        family_dominance_ratio=0.9,
        family_named_count=3,
        disallow_pair_count=1,
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
        family_dominance_ratio=0.8,
        family_named_count=2,
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
        positive_component_keys=frozenset({"c2"}),
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

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_component_key"] == "c2"
    assert row["candidate_count"] == 1
    assert row["candidate_signatures"] == 2
    assert row["label"] == 1
    assert row["positive_candidate_count"] == 1
    assert row["group_has_positive"] == 1
    assert row["best_positive_retrieval_rank"] == 2
    assert row["best_competitor_component_key"] is None


def test_make_candidate_rows_preserves_known_positive_with_disallow_pair() -> None:
    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary_a = build_cluster_summary(component_key="c1", size=3)
    summary_b = build_cluster_summary(component_key="c2", size=2)
    stats_a = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        family_dominance_ratio=0.9,
        family_named_count=3,
        disallow_pair_count=1,
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
        family_dominance_ratio=0.8,
        family_named_count=2,
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

    rows_by_component = {row["candidate_component_key"]: row for row in rows}
    assert set(rows_by_component) == {"c1", "c2"}
    assert rows_by_component["c1"]["label"] == 1
    assert rows_by_component["c2"]["label"] == 0
    assert rows_by_component["c1"]["positive_candidate_count"] == 1
    assert rows_by_component["c1"]["positive_candidate_keys"] == "c1"
    assert rows_by_component["c1"]["group_has_positive"] == 1
    assert rows_by_component["c1"]["best_positive_retrieval_rank"] == 1


def test_make_candidate_rows_does_not_promote_hard_require_constraints_to_positive() -> None:
    """Candidate labels should come from explicit positives, not pairwise constraints."""

    query = build_query_features(has_coauthors=True, has_affiliations=True)
    summary_a = build_cluster_summary(component_key="c1", size=3)
    summary_b = build_cluster_summary(component_key="c2", size=2)
    stats_a = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=3,
        family_id="fam1",
        family_dominance_ratio=0.9,
        family_named_count=3,
        require_pair_count=1,
        count=3,
        sum_distance=0.2,
        min_distance=0.0,
        top_smallest_neg_heap=[-0.0, -0.1, -0.1],
    )
    stats_b = reranker_utils.ClusterPairwiseStats(
        cluster_id="c2",
        retrieval_rank=2,
        retrieval_score=0.7,
        cluster_size=2,
        family_id="fam2",
        family_dominance_ratio=0.8,
        family_named_count=2,
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
        positive_component_keys=frozenset(),
        support_type="unresolved",
        block_size=5,
        component_size=0,
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

    rows_by_component = {row["candidate_component_key"]: row for row in rows}
    assert rows_by_component["c1"]["label"] == 0
    assert rows_by_component["c2"]["label"] == 0
    assert rows_by_component["c1"]["positive_candidate_count"] == 0
    assert rows_by_component["c1"]["positive_candidate_keys"] == ""
    assert rows_by_component["c1"]["group_has_positive"] == 0
    assert rows_by_component["c1"]["best_positive_retrieval_rank"] is None


def test_make_candidate_rows_includes_title_and_specter_summary_features() -> None:
    query = build_query_features(
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
        title_terms=frozenset({"graph", "learning"}),
    )
    summary = build_cluster_summary(
        component_key="c1",
        size=2,
        title_counts=Counter({"graph": 2, "learning": 1}),
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
        exemplar_vectors=[np.asarray([0.8, 0.2], dtype=np.float32), np.asarray([1.0, 0.0], dtype=np.float32)],
    )
    stats = reranker_utils.ClusterPairwiseStats(
        cluster_id="c1",
        retrieval_rank=1,
        retrieval_score=0.8,
        cluster_size=2,
        family_id="fam1",
        family_dominance_ratio=1.0,
        family_named_count=2,
        count=2,
        sum_distance=0.4,
        min_distance=0.1,
        top_smallest_neg_heap=[-0.1, -0.3],
    )
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c1"}),
        support_type="labeled",
        block_size=2,
        component_size=2,
        sampling_info_bucket="rich",
    )

    rows = reranker_utils.make_candidate_rows(
        query_case=query_case,
        query_view="full",
        query_features=query,
        shortlist_component_keys=["c1"],
        retrieval_scores={"c1": 0.8},
        retrieval_ranks={"c1": 1},
        retrieval_window_state={
            "scored_candidate_components": 1,
            "scored_candidate_signatures": 2,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        summary_by_component={"c1": summary},
        stats_by_component={"c1": stats},
    )

    assert rows[0]["title_overlap"] == pytest.approx(0.75)
    assert rows[0]["specter_centroid_similarity"] == pytest.approx(1.0)
    assert rows[0]["specter_exemplar_similarity"] == pytest.approx(1.0)


def test_choose_generic_heuristic_keeps_top1_when_cross_family_margin_is_small() -> None:
    retrieval_top1 = _base_row(
        candidate_component_key="c1",
        family_id="fam1",
        retrieval_rank=1,
        top5_mean_distance=0.25,
        label=1,
    )
    challenger = _base_row(
        candidate_component_key="c2",
        family_id="fam2",
        retrieval_rank=2,
        top5_mean_distance=0.23,
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
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
            top3_mean_distance=0.3,
            top5_mean_distance=0.3,
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
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
            top3_mean_distance=0.3,
            top5_mean_distance=0.3,
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
            family_id="c1",
            retrieval_rank=26,
            top3_mean_distance=0.19,
            top5_mean_distance=0.19,
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=1,
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
        ),
        _base_row(
            query_group_id="g1",
            candidate_component_key="c2",
            family_id="fam2",
            retrieval_rank=2,
            top3_mean_distance=0.21,
            top5_mean_distance=0.215,
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


def test_compute_query_cluster_stats_batched_matches_individual_batched_requests(
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
    [(single_one_stats, single_one_diag)] = reranker_utils.compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        requests=[request_one],
        pair_batch_size=10,
        max_top_k=5,
    )
    [(single_two_stats, single_two_diag)] = reranker_utils.compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        requests=[request_two],
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


def test_compute_query_cluster_stats_batched_recomputes_positive_components_without_seed_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_calls: list[bool] = []

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
            del partial_supervision, runtime_context, constraint_backend
            resolve_calls.append(bool(incremental_dont_use_cluster_seeds))
            base_value = 0.1 if incremental_dont_use_cluster_seeds else 0.7
            return [float(base_value)] * len(batch_pairs), {}

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        features = np.asarray([[float(label)] for _left, _right, label in signature_pairs], dtype=np.float64)
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
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(reranker_utils, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(reranker_utils, "_predict_and_combine", fake_predict_and_combine)

    request = reranker_utils.QueryClusterStatsRequest(
        query_signature_id="q1",
        shortlist_component_keys=("c_pos", "c_neg"),
        candidate_signature_ids_by_component={
            "c_pos": ["s1"],
            "c_neg": ["s2"],
        },
        retrieval_ranks={"c_pos": 1, "c_neg": 2},
        retrieval_scores={"c_pos": 0.9, "c_neg": 0.8},
        summary_by_component={
            "c_pos": build_cluster_summary(component_key="c_pos", size=1),
            "c_neg": build_cluster_summary(component_key="c_neg", size=1),
        },
        incremental_dont_use_cluster_seeds_component_keys=frozenset({"c_pos"}),
    )

    [(stats, _diag)] = reranker_utils.compute_query_cluster_stats_batched(
        clusterer=_FakeClusterer(),
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        requests=[request],
        pair_batch_size=10,
        max_top_k=5,
    )

    assert stats["c_pos"].min_distance == pytest.approx(0.1)
    assert stats["c_neg"].min_distance == pytest.approx(0.7)
    assert resolve_calls == [False, True]


def test_compute_query_cluster_stats_batched_ignores_disallow_constraints_for_positive_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_pairs: list[tuple[str, str, float]] = []

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
            return [-(100000.0 - 10000.0)] * len(batch_pairs), {}

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        captured_pairs.extend((left, right, float(label)) for left, right, label in signature_pairs)
        features = np.asarray([[0.3] for _left, _right, _label in signature_pairs], dtype=np.float64)
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
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(reranker_utils, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(reranker_utils, "_predict_and_combine", fake_predict_and_combine)

    request = reranker_utils.QueryClusterStatsRequest(
        query_signature_id="q1",
        shortlist_component_keys=("c_pos", "c_neg"),
        candidate_signature_ids_by_component={
            "c_pos": ["s1"],
            "c_neg": ["s2"],
        },
        retrieval_ranks={"c_pos": 1, "c_neg": 2},
        retrieval_scores={"c_pos": 0.9, "c_neg": 0.8},
        summary_by_component={
            "c_pos": build_cluster_summary(component_key="c_pos", size=1),
            "c_neg": build_cluster_summary(component_key="c_neg", size=1),
        },
        ignore_disallow_constraints_component_keys=frozenset({"c_pos"}),
    )

    [(stats, _diag)] = reranker_utils.compute_query_cluster_stats_batched(
        clusterer=_FakeClusterer(),
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        requests=[request],
        pair_batch_size=10,
        max_top_k=5,
    )

    assert stats["c_pos"].min_distance == pytest.approx(0.3)
    assert stats["c_neg"].min_distance == pytest.approx(0.3)
    assert stats["c_pos"].disallow_pair_count == 0
    assert stats["c_neg"].disallow_pair_count == 1
    assert len(captured_pairs) == 2
    assert np.isnan(captured_pairs[0][2])
    assert captured_pairs[1][2] == pytest.approx(-(100000.0 - 10000.0))


def test_seed_constraint_bypass_component_keys_empty_for_ordinary_positive() -> None:
    ordinary_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c_pos"}),
        support_type="labeled",
        block_size=3,
        component_size=1,
        sampling_info_bucket="rich",
    )
    seeded_dataset = SimpleNamespace(
        cluster_seeds_require={"q1": 7, "s_pos": 7},
        cluster_seeds_disallow=set(),
    )
    no_seed_dataset = SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set())
    candidate_signature_ids_by_component = {"c_pos": ["s_pos"]}

    assert (
        reranker_utils.seed_constraint_bypass_component_keys(
            dataset=seeded_dataset,
            query_case=ordinary_case,
            candidate_signature_ids_by_component=candidate_signature_ids_by_component,
        )
        == frozenset()
    )
    assert (
        reranker_utils.seed_constraint_bypass_component_keys(
            dataset=no_seed_dataset,
            query_case=replace(ordinary_case, query_in_seed_before_holdout=True),
            candidate_signature_ids_by_component=candidate_signature_ids_by_component,
        )
        == frozenset()
    )


def test_seed_constraint_bypass_component_keys_keeps_seeded_loo_self_containing_positive() -> None:
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled_loo",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c_pos", "c_unconnected", "c_other_seed_cluster"}),
        support_type="self_containing_loo",
        block_size=5,
        component_size=2,
        sampling_info_bucket="rich",
        split="eval_loo",
        query_in_seed_before_holdout=True,
    )
    dataset = SimpleNamespace(
        cluster_seeds_require={"q1": 7, "s_pos": 7, "s_other_seed": 9},
        cluster_seeds_disallow={("q1", "s_disallowed")},
    )

    assert reranker_utils.seed_constraint_bypass_component_keys(
        dataset=dataset,
        query_case=query_case,
        candidate_signature_ids_by_component={
            "c_pos": ["s_pos", "s_other"],
            "c_unconnected": ["s_other"],
            "c_other_seed_cluster": ["s_other_seed"],
            "c_negative": ["s_disallowed"],
        },
    ) == frozenset({"c_negative", "c_pos"})


def test_seed_constraint_bypass_component_keys_includes_seeded_negative_component() -> None:
    query_case = reranker_utils.RerankerQueryCase(
        source="labeled_loo",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c_pos"}),
        support_type="self_containing_loo",
        block_size=5,
        component_size=2,
        sampling_info_bucket="rich",
        split="eval_loo",
        query_in_seed_before_holdout=True,
    )
    dataset = SimpleNamespace(
        cluster_seeds_require={"q1": 7, "s_neg": 7},
        cluster_seeds_disallow=set(),
    )

    assert reranker_utils.seed_constraint_bypass_component_keys(
        dataset=dataset,
        query_case=query_case,
        candidate_signature_ids_by_component={
            "c_pos": ["s_pos"],
            "c_neg_seeded": ["s_neg"],
        },
    ) == frozenset({"c_neg_seeded"})


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


def test_prepare_query_rows_request_materializes_raw_similarity_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_case = reranker_utils.RerankerQueryCase(
        source="generated",
        dataset="d1",
        query_id="q1",
        query_signature_id="q1",
        block_key="b",
        positive_component_keys=frozenset({"c1"}),
        support_type="orcid",
        block_size=2,
        component_size=1,
        sampling_info_bucket="initial_only",
    )
    dataset = SimpleNamespace(
        signatures={
            "q1": SimpleNamespace(
                paper_id="p1",
                author_info_last="Smith",
                author_info_affiliations=["AI Lab"],
            ),
            "s2": SimpleNamespace(
                paper_id="p2",
                author_info_last="Jones",
                author_info_affiliations=["AI Lab"],
            ),
        },
        papers={
            "p1": SimpleNamespace(
                paper_id="p1",
                title="Neural Graph Models",
                authors=[
                    SimpleNamespace(author_name="Alice Smith"),
                    SimpleNamespace(author_name="Bob Lee"),
                ],
            ),
            "p2": SimpleNamespace(
                paper_id="p2",
                title="Neural Graph Signal",
                authors=[
                    SimpleNamespace(author_name="Carol Jones"),
                    SimpleNamespace(author_name="Bob Lee"),
                ],
            ),
        },
    )

    monkeypatch.setattr(reranker_build.retrieval, "mask_query_features", lambda query, *_args, **_kwargs: query)
    monkeypatch.setattr(
        reranker_build,
        "build_retrieval_window",
        lambda **_kwargs: (
            ["c1"],
            {"c1": 0.9},
            {"c1": 1},
            {
                "scored_candidate_components": 1,
                "scored_candidate_signatures": 1,
                "orcid_filter_applied": 0,
                "middle_initial_filter_applied": 0,
                "year_range_filter_applied": 0,
            },
        ),
    )

    prepared_request = reranker_build._prepare_query_rows_request(  # noqa: SLF001
        dataset=dataset,
        query_case=query_case,
        block_component_count=1,
        base_query=build_query_features(has_coauthors=True, has_affiliations=True),
        query_views=["initial_only"],
        raw_candidate_summaries=[build_cluster_summary(component_key="c1", size=1)],
        summary_by_component={"c1": build_cluster_summary(component_key="c1", size=1)},
        candidate_signature_ids_by_component={"c1": ["q1", "s2"]},
        retrieval_approach="test",
        retrieval_engine="auto",
        window_size=5,
        raw_paper_text_by_id={
            "p1": "alpha beta shared",
            "p2": "alpha gamma shared",
        },
        raw_similarity_feature_cache=reranker_build._RawSimilarityFeatureCache(),  # noqa: SLF001
    )
    rows = reranker_build._materialize_query_rows_from_prepared(  # noqa: SLF001
        prepared_request,
        stats_by_component={
            "c1": reranker_utils.ClusterPairwiseStats(
                cluster_id="c1",
                retrieval_rank=1,
                retrieval_score=0.9,
                cluster_size=1,
                family_id="c1",
                count=1,
                sum_distance=0.2,
                min_distance=0.2,
                top_smallest_neg_heap=[-0.2],
            )
        },
    )

    assert len(rows) == 1
    assert rows[0]["raw_max_affiliation_jaccard"] == pytest.approx(1.0)
    assert rows[0]["raw_max_coauthor_jaccard"] == pytest.approx(0.5)
    assert rows[0]["raw_max_title_jaccard"] == pytest.approx(0.5)
    assert rows[0]["raw_max_text_jaccard"] == pytest.approx(0.5)


def test_flush_prepared_query_requests_streams_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    rows_output_path = tmp_path / "rows.csv"
    query_groups_output_path = tmp_path / "query_groups.csv"
    summary_accumulator = reranker_build._QueryGroupSummaryAccumulator()  # noqa: SLF001
    pair_counts: list[int] = []
    featurize_seconds: list[float] = []
    model_seconds: list[float] = []

    flushed_row_count, flushed_query_group_count = reranker_build._flush_prepared_query_requests(  # noqa: SLF001
        clusterer=SimpleNamespace(),
        dataset=SimpleNamespace(),
        runtime_context=SimpleNamespace(),
        constraint_backend=SimpleNamespace(),
        prepared_requests=[prepared_request],
        pair_batch_size=10,
        max_top_k=5,
        pair_counts=pair_counts,
        featurize_seconds=featurize_seconds,
        model_seconds=model_seconds,
        rows_output_path=rows_output_path,
        query_group_metadata_output_path=query_groups_output_path,
        query_group_summary_accumulator=summary_accumulator,
    )

    streamed_rows = reranker_utils.read_rows_csv(rows_output_path)
    streamed_metadata_rows = reranker_utils.read_query_group_metadata_csv(query_groups_output_path)
    legacy_columns = {"normalized_orcid", "orcid_group_size", "orcid_group_size_bucket"}

    assert flushed_row_count == 2
    assert flushed_query_group_count == 1
    assert len(streamed_rows) == 2
    assert len(streamed_metadata_rows) == 1
    assert set(reranker_utils.AUDIT_ORCID_METADATA_COLUMNS) <= set(streamed_rows[0])
    assert set(reranker_utils.AUDIT_ORCID_METADATA_COLUMNS) <= set(streamed_metadata_rows[0])
    assert legacy_columns.isdisjoint(streamed_rows[0])
    assert legacy_columns.isdisjoint(streamed_metadata_rows[0])
    assert pair_counts == [3]
    assert featurize_seconds == [1.5]
    assert model_seconds == [0.25]
    assert summary_accumulator.to_summary() == reranker_utils.summarize_dataset_rows(streamed_rows)


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


def test_build_retrieval_window_plain_all_skips_profile_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features(specter=np.asarray([1.0, 0.0], dtype=np.float32))
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4, specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32)),
        build_cluster_summary(component_key="c2", size=4, specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32)),
    ]

    def fail_profile_build(_summary: Any) -> Any:
        raise AssertionError("build_cluster_profile should not run for all__hybrid_centroid")

    monkeypatch.setattr(reranker_utils, "build_cluster_profile", fail_profile_build)

    ranked_component_keys, _scores, _ranks, _state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_rows,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
    )

    assert ranked_component_keys == ["c1", "c2"]


def test_build_retrieval_window_strict_rust_uses_rust_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    query = build_query_features()
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=4),
    ]
    captured: dict[str, Any] = {}

    def fake_rank_top_summaries_rust_hybrid_centroid(**kwargs: Any) -> list[tuple[float, Any]]:
        captured.update(kwargs)
        return [(2.0, candidate_rows[1]), (1.0, candidate_rows[0])]

    monkeypatch.setattr(
        reranker_utils,
        "rank_top_summaries_rust_hybrid_centroid",
        fake_rank_top_summaries_rust_hybrid_centroid,
    )

    ranked_component_keys, _scores, _ranks, state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_rows,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
        rust_hybrid_centroid_retriever=retrieval_utils.RustHybridCentroidRetrieverHandle(
            retriever=SimpleNamespace(),
            summary_by_component={str(summary.component_key): summary for summary in candidate_rows},
        ),
        retrieval_engine="rust",
    )

    assert ranked_component_keys == ["c2", "c1"]
    assert captured["component_keys"] == ["c1", "c2"]
    assert state["retrieval_engine_rust_method_count"] == 1
    assert state["retrieval_engine_python_method_count"] == 0
    assert state["retrieval_engine_fallback_count"] == 0


def test_build_retrieval_window_strict_rust_raises_instead_of_falling_back() -> None:
    query = build_query_features()
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=4),
    ]

    with pytest.raises(ValueError, match="Strict Rust retrieval requires"):
        reranker_utils.build_retrieval_window(
            query=query,
            raw_candidate_summaries=candidate_rows,
            max_block_component_size=4,
            retrieval_approach="all__hybrid_centroid",
            max_ranked_clusters=2,
            retrieval_engine="rust",
        )


def test_build_retrieval_window_auto_records_rust_fallback() -> None:
    query = build_query_features()
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=4),
    ]

    ranked_component_keys, _scores, _ranks, state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_rows,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
        rust_hybrid_centroid_retriever=retrieval_utils.RustHybridCentroidRetrieverHandle(
            retriever=SimpleNamespace(),
            summary_by_component={"c1": candidate_rows[0]},
        ),
        retrieval_engine="auto",
    )

    assert ranked_component_keys == ["c1", "c2"]
    assert state["retrieval_engine_rust_method_count"] == 0
    assert state["retrieval_engine_python_method_count"] == 1
    assert state["retrieval_engine_fallback_count"] == 1


def test_frozen_best_rust_hybrid_centroid_policy_matches_expected_contract() -> None:
    policy = retrieval_utils.FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY

    assert retrieval_utils.FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME == "h_wang_any_input_v2"
    assert retrieval_utils.RUST_HYBRID_CENTROID_FEATURE_ORDER == (
        "centroid",
        "coauthor",
        "affiliation",
        "middle",
        "first_name",
    )
    assert policy.full_candidate_strategy == "name_compat_plus_global_backfill5"
    assert policy.full_weights == pytest.approx((0.527232, 0.223412, 0.146909, 0.009439, 0.093007))
    assert policy.initial_only_weights == pytest.approx((0.520012, 0.220264, 0.109278, 0.150447, 0.0))
    assert policy.full_scoring_config is not None
    assert policy.full_scoring_config.first_name_mode == "exact_then_prefix_half"
    assert policy.initial_only_scoring_config is not None
    assert policy.initial_only_scoring_config.specter_mode == "max_centroid_exemplar"
    assert policy.uses_exemplar_scoring() is True


def test_build_retrieval_window_applies_frozen_full_candidate_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features(first="alice", has_full_first=True)
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=4),
        build_cluster_summary(component_key="c3", size=4),
    ]
    policy = retrieval_utils.FrozenRustHybridCentroidPolicy(
        full_weights=(0.44, 0.18, 0.13, 0.03, 0.10),
        initial_only_weights=(0.49, 0.16, 0.14, 0.10, 0.0),
        full_scoring_config=retrieval_utils.RustHybridCentroidScoringConfig(first_name_mode="exact_then_prefix_half"),
        full_candidate_strategy="family2_plus_global_backfill1",
    )
    retrieval_subblock_index = {
        "signature_to_subblock": {"q1": "alice_bucket"},
        "subblock_to_components": {
            "alice_bucket": ["c1"],
            "other_bucket": ["c2", "c3"],
        },
        "prefix_to_subblocks": {
            2: {"al": ["alice_bucket"]},
            3: {"ali": ["alice_bucket"]},
            4: {"alic": ["alice_bucket"]},
        },
    }
    captured: dict[str, Any] = {}

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

    def fake_rank_top_summaries_rust_hybrid_centroid(**kwargs: Any) -> list[tuple[float, Any]]:
        captured.update(kwargs)
        return [(1.0, candidate_rows[0])]

    monkeypatch.setattr(
        reranker_utils,
        "rank_top_summaries_rust_hybrid_centroid",
        fake_rank_top_summaries_rust_hybrid_centroid,
    )

    ranked_component_keys, _scores, _ranks, state = reranker_utils.build_retrieval_window(
        query=query,
        raw_candidate_summaries=candidate_rows,
        max_block_component_size=4,
        retrieval_approach="all__hybrid_centroid",
        max_ranked_clusters=2,
        rust_hybrid_centroid_retriever=retrieval_utils.RustHybridCentroidRetrieverHandle(
            retriever=SimpleNamespace(),
            summary_by_component={str(summary.component_key): summary for summary in candidate_rows},
        ),
        frozen_rust_hybrid_centroid_policy=policy,
        query_signature_id="q1",
        retrieval_subblock_index=retrieval_subblock_index,
    )

    assert ranked_component_keys == ["c1"]
    assert captured["component_keys"] == ["c1", "c2"]
    assert state["preselected_candidate_components"] == 2
    assert captured["weights"] == pytest.approx(policy.full_weights)
    assert state["candidate_components"] == 3
    assert state["scored_candidate_components"] == 2


def test_configure_runtime_environment_requires_strict_rust_name_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(reranker_utils.STRICT_RUST_NAME_COMPAT_ENV, raising=False)

    reranker_utils.configure_runtime_environment(n_jobs=2, backend="rust")

    assert reranker_utils.os.environ[reranker_utils.STRICT_RUST_NAME_COMPAT_ENV] == "1"


def test_name_compat_candidate_strategy_preserves_same_subblock_and_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(reranker_utils.STRICT_RUST_NAME_COMPAT_ENV, raising=False)
    monkeypatch.setattr(
        reranker_utils,
        "build_rust_name_compatible_subblock_selector",
        lambda _index: (_ for _ in ()).throw(RuntimeError("rust selector unavailable")),
    )
    query = build_query_features(first="alice", has_full_first=True)
    retrieval_subblock_index = {
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
        "prefix_to_subblocks": {},
    }

    selected = reranker_utils._select_component_keys_for_candidate_strategy(  # noqa: SLF001
        query=query,
        query_signature_id="q1",
        component_keys=["c1", "c2", "c3", "c4"],
        strategy="name_compat_plus_global_backfill1",
        retrieval_subblock_index=retrieval_subblock_index,
        max_ranked_clusters=2,
    )

    assert selected == ["c1", "c2", "c3"]
    assert retrieval_subblock_index[reranker_utils.RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_COUNT_KEY] == 1
    assert "rust selector unavailable" in str(
        retrieval_subblock_index[reranker_utils.RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_REASON_KEY]
    )


def test_name_compat_candidate_strategy_requires_rust_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(reranker_utils.STRICT_RUST_NAME_COMPAT_ENV, "1")
    monkeypatch.setattr(
        reranker_utils,
        "build_rust_name_compatible_subblock_selector",
        lambda _index: (_ for _ in ()).throw(RuntimeError("rust selector unavailable")),
    )
    query = build_query_features(first="alice", has_full_first=True)
    retrieval_subblock_index = {
        "signature_to_subblock": {"q1": "legacy_bucket"},
        "subblock_to_components": {"legacy_bucket": ["c1"]},
        "subblock_tokens_by_subblock": {"legacy_bucket": ["alice"]},
        "prefix_to_subblocks": {},
    }

    with pytest.raises(RuntimeError, match="rust selector unavailable"):
        reranker_utils._select_component_keys_for_candidate_strategy(  # noqa: SLF001
            query=query,
            query_signature_id="q1",
            component_keys=["c1"],
            strategy="name_compat_plus_global_backfill1",
            retrieval_subblock_index=retrieval_subblock_index,
            max_ranked_clusters=2,
        )


def test_build_retrieval_window_rust_engine_requires_name_compat_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(reranker_utils.STRICT_RUST_NAME_COMPAT_ENV, raising=False)
    monkeypatch.setattr(
        reranker_utils,
        "build_rust_name_compatible_subblock_selector",
        lambda _index: (_ for _ in ()).throw(RuntimeError("rust selector unavailable")),
    )
    query = build_query_features(first="alice", has_full_first=True)
    candidate_rows = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=4),
    ]
    policy = retrieval_utils.FrozenRustHybridCentroidPolicy(
        full_weights=(0.44, 0.18, 0.13, 0.03, 0.10),
        initial_only_weights=(0.49, 0.16, 0.14, 0.10, 0.0),
        full_candidate_strategy="name_compat_plus_global_backfill1",
    )
    retrieval_subblock_index = {
        "signature_to_subblock": {"q1": "legacy_bucket"},
        "subblock_to_components": {"legacy_bucket": ["c1"]},
        "subblock_tokens_by_subblock": {"legacy_bucket": ["alice"]},
        "prefix_to_subblocks": {},
    }

    with pytest.raises(RuntimeError, match="rust selector unavailable"):
        reranker_utils.build_retrieval_window(
            query=query,
            raw_candidate_summaries=candidate_rows,
            max_block_component_size=4,
            retrieval_approach="all__hybrid_centroid",
            max_ranked_clusters=2,
            frozen_rust_hybrid_centroid_policy=policy,
            query_signature_id="q1",
            retrieval_subblock_index=retrieval_subblock_index,
            retrieval_engine="rust",
        )

    assert reranker_utils.RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY not in retrieval_subblock_index


def test_name_compat_candidate_strategy_uses_rust_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = build_query_features(first="alice", has_full_first=True)
    retrieval_subblock_index = {
        "signature_to_subblock": {"q1": "alice"},
        "subblock_to_components": {"alice": ["c1"], "ali": ["c2"]},
        "subblock_tokens_by_subblock": {"alice": ["alice"], "ali": ["ali"]},
        "prefix_to_subblocks": {},
    }
    captured: dict[str, Any] = {}

    class _FakeRustSelector:
        def select(
            self,
            query_signature_id: str,
            query_first: str,
            component_keys: list[str],
            *,
            global_backfill_count: int,
        ) -> list[str]:
            captured["query_signature_id"] = query_signature_id
            captured["query_first"] = query_first
            captured["component_keys"] = list(component_keys)
            captured["global_backfill_count"] = global_backfill_count
            return ["c2", "c3"]

    monkeypatch.setattr(
        reranker_utils,
        "build_rust_name_compatible_subblock_selector",
        lambda index: _FakeRustSelector(),
    )

    selected = reranker_utils._select_component_keys_for_candidate_strategy(  # noqa: SLF001
        query=query,
        query_signature_id="q1",
        component_keys=["c1", "c2", "c3"],
        strategy="name_compat_plus_global_backfill4",
        retrieval_subblock_index=retrieval_subblock_index,
        max_ranked_clusters=2,
    )

    assert selected == ["c2", "c3"]
    assert captured == {
        "query_signature_id": "q1",
        "query_first": "alice",
        "component_keys": ["c1", "c2", "c3"],
        "global_backfill_count": 4,
    }
    assert reranker_utils.RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY in retrieval_subblock_index


def test_build_labeled_retrieval_subblock_index_uses_subblocks_for_component_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component_signatures = {
        "b::c1": ["s1", "s2"],
        "b::c2": ["s3"],
    }
    block_to_component_keys = {"b": ["b::c1", "b::c2"]}

    monkeypatch.setattr(
        reranker_utils,
        "make_subblocks_with_telemetry",
        lambda signature_ids, _dataset, maximum_size: (
            {"alice": ["s1", "s2"], "alicia": ["s3"]},
            {
                "input_signature_count": len(signature_ids),
                "final_subblock_count": 2,
                "final_specter_labeled_subblock_count": 0,
                "specter_invocation_count": 0,
            },
        ),
    )

    index, diagnostics = reranker_utils.build_labeled_retrieval_subblock_index(
        dataset=SimpleNamespace(),
        block_to_component_keys=block_to_component_keys,
        component_signatures=component_signatures,
        maximum_size=15000,
    )

    assert index["signature_to_subblock"] == {
        "s1": "b::alice",
        "s2": "b::alice",
        "s3": "b::alicia",
    }
    assert index["subblock_to_components"] == {
        "b::alice": ["b::c1"],
        "b::alicia": ["b::c2"],
    }
    assert index["subblock_tokens_by_subblock"] == {
        "b::alice": ["alice"],
        "b::alicia": ["alicia"],
    }
    assert index["prefix_to_subblocks"][2]["al"] == ["b::alice", "b::alicia"]
    assert diagnostics["blocks"] == 1
    assert diagnostics["subblocks"] == 2


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


def test_load_query_metadata_requires_audit_orcid_keys(tmp_path: Path) -> None:
    path = tmp_path / "query_set.json"
    path.write_text(
        json.dumps(
            {
                "query_rows": [
                    {
                        "query_id": "q1",
                        "_audit_normalized_orcid": "0000000000000001",
                        "_audit_orcid_group_size": 2,
                        "query_subblock_key": "single",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = reranker_build.load_query_metadata(path)

    assert rows["q1"]["_audit_normalized_orcid"] == "0000000000000001"
    assert rows["q1"]["_audit_orcid_group_size"] == 2


def test_load_query_metadata_rejects_legacy_orcid_keys(tmp_path: Path) -> None:
    path = tmp_path / "query_set.json"
    path.write_text(
        json.dumps(
            {
                "query_rows": [
                    {
                        "query_id": "q1",
                        "normalized_orcid": "0000000000000001",
                        "orcid_group_size": 2,
                        "query_subblock_key": "single",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="_audit_normalized_orcid"):
        reranker_build.load_query_metadata(path)


def test_normalize_giant_block_dataset_label_collapses_block_name() -> None:
    assert reranker_build._normalize_giant_block_dataset_label("S Park") == "s_park"  # noqa: SLF001
    assert reranker_build._normalize_giant_block_dataset_label("h wang") == "h_wang"  # noqa: SLF001


def test_build_dataset_main_rejects_labeled_limit_queries_plus_query_id_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_id_file = tmp_path / "query_ids.txt"
    query_id_file.write_text("q1\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m scripts.reranker_dataset.build",
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
            "python -m scripts.reranker_dataset.build",
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


def test_build_dataset_main_rejects_custom_query_views_for_any_input_giant_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m scripts.reranker_dataset.build",
            "giant_block",
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


def test_build_dataset_main_rejects_nonpositive_min_candidates_for_giant_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "python -m scripts.reranker_dataset.build",
            "giant_block",
            "--data-dir",
            str(tmp_path / "data"),
            "--step2-dir",
            str(tmp_path / "step2"),
            "--output-dir",
            str(tmp_path / "out"),
            "--query-source",
            "orcid_any_input",
            "--min-candidates-per-query-group",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="min_candidates_per_query_group must be positive"):
        reranker_build.main()


def test_build_giant_block_any_input_query_cases_assigns_expected_supervision(
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

    query_cases, summary = reranker_build._build_giant_block_any_input_query_cases(  # noqa: SLF001
        dataset_label="s_park",
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
    assert cases_by_id["q_pos"].source == "s_park"
    assert cases_by_id["q_pos"].dataset == "s_park"
    assert cases_by_id["q_pos"].supervision_type == "positive_repeat_orcid"
    assert cases_by_id["q_pos"].positive_component_keys == frozenset({"c1"})
    assert cases_by_id["q_pos"].natural_query_view == "full"
    assert cases_by_id["q_seeded"].supervision_type == "positive_repeat_orcid"
    assert cases_by_id["q_seeded"].positive_component_keys == frozenset({"c2"})
    assert cases_by_id["q_seeded"].query_in_seed_before_holdout is True
    assert cases_by_id["q_seeded"].natural_query_view == "initial_only"
    assert cases_by_id["q_single"].supervision_type == "unlabeled_singleton_orcid"
    assert cases_by_id["q_single"].positive_component_keys == frozenset()
    assert cases_by_id["q_unresolved"].supervision_type == "unresolved_repeat_orcid"
    assert cases_by_id["q_unresolved"].positive_component_keys == frozenset()
    assert summary["supervision_type_counts"] == {
        key: int(value)
        for key, value in sorted(Counter(query_case.supervision_type for query_case in query_cases).items())
    }


def test_build_giant_block_supported_query_cases_uses_dataset_label() -> None:
    query_cases, summary = reranker_build._build_giant_block_supported_query_cases(  # noqa: SLF001
        dataset_label="s_park",
        query_metadata={
            "q1": {
                "_audit_normalized_orcid": "0000000000000001",
                "_audit_orcid_group_size": 2,
                "query_subblock_key": "single",
            }
        },
        signature_to_cluster_id={"seeded": "c1"},
        seed_cluster_counts_by_orcid={"0000000000000001": Counter({"c1": 1})},
        limit_queries=None,
        query_id_file=None,
        seed=13,
    )

    assert len(query_cases) == 1
    assert query_cases[0].source == "s_park"
    assert query_cases[0].dataset == "s_park"
    assert query_cases[0].query_source == "supported_single_letter"
    assert summary["query_count"] == 1


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
            supervision_type="unlabeled_singleton_orcid",
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
        supervision_types=["unlabeled_singleton_orcid"],
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
            "supervision_type": "reviewed_no_positive",
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


def test_select_reject_threshold_excludes_single_candidate_groups() -> None:
    rows = [
        {
            "query_group_id": "p1",
            "supervision_type": "positive_repeat_orcid",
            "model_margin": 0.9,
            "model_correct": 1,
            "query_view": "full",
            "has_runner_up": 1,
        },
        {
            "query_group_id": "n1",
            "supervision_type": "reviewed_no_positive",
            "model_margin": 0.2,
            "model_correct": 0,
            "query_view": "initial_only",
            "has_runner_up": 1,
        },
        {
            "query_group_id": "s1",
            "supervision_type": "unlabeled_singleton_orcid",
            "model_margin": None,
            "model_correct": 0,
            "query_view": "full",
            "has_runner_up": 0,
        },
    ]

    summary = s2and_ranker_eval._select_reject_threshold(rows)  # noqa: SLF001

    assert summary["queries"] == 3
    assert summary["eligible_queries"] == 2
    assert summary["singleton_candidate_group_count"] == 1
    assert summary["threshold"] == pytest.approx(0.2)
    assert summary["per_view"]["full"]["singleton_candidate_group_count"] == 1


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


def test_query_group_metadata_summary_matches_row_summary() -> None:
    rows = [
        _base_row(
            query_group_id="g1",
            query_id="q1",
            query_view="full",
            natural_query_view="full",
            candidate_component_key="c1",
            label=1,
            retrieval_rank=1,
            positive_candidate_count=1,
            positive_candidate_keys="c1",
            candidate_count=2,
        ),
        _base_row(
            query_group_id="g1",
            query_id="q1",
            query_view="full",
            natural_query_view="full",
            candidate_component_key="c2",
            label=0,
            retrieval_rank=2,
            best_competitor_component_key="c1",
            positive_candidate_count=1,
            positive_candidate_keys="c1",
            candidate_count=2,
        ),
        _base_row(
            query_group_id="g2",
            query_id="q2",
            query_view="initial_only",
            candidate_component_key="c3",
            label=0,
            retrieval_rank=1,
            positive_candidate_count=0,
            positive_candidate_keys="",
            group_has_positive=0,
            best_positive_retrieval_rank=None,
            candidate_count=2,
        ),
        _base_row(
            query_group_id="g2",
            query_id="q2",
            query_view="initial_only",
            candidate_component_key="c4",
            label=0,
            retrieval_rank=2,
            best_competitor_component_key="c3",
            positive_candidate_count=0,
            positive_candidate_keys="",
            group_has_positive=0,
            best_positive_retrieval_rank=None,
            candidate_count=2,
        ),
        _base_row(
            query_group_id="g3",
            query_id="q3",
            query_view="full",
            natural_query_view="full",
            candidate_component_key="c5",
            label=1,
            retrieval_rank=1,
            best_competitor_component_key=None,
            candidate_count=1,
            candidate_signatures=3,
            scored_candidate_components=1,
            scored_candidate_signatures=3,
            positive_candidate_count=1,
            positive_candidate_keys="c5",
        ),
    ]

    metadata_rows = [
        reranker_utils.summarize_query_group_rows(group_rows, block_component_count=3)
        for group_rows in reranker_utils.group_rows(rows).values()
    ]

    assert reranker_build._summarize_query_group_metadata_rows(metadata_rows) == reranker_utils.summarize_dataset_rows(  # noqa: SLF001
        rows
    )


def test_flush_prepared_query_requests_filters_single_candidate_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_requests = [
        SimpleNamespace(block_component_count=3, stats_request=object()),
        SimpleNamespace(block_component_count=3, stats_request=object()),
    ]
    materialized_rows_by_request = {
        id(prepared_requests[0]): [
            _base_row(
                query_group_id="g_keep",
                query_id="q_keep",
                candidate_component_key="c1",
                candidate_cluster_id="c1",
                candidate_count=2,
                positive_candidate_count=1,
                positive_candidate_keys="c1",
                best_positive_retrieval_rank=1,
                label=1,
                retrieval_rank=1,
                best_competitor_component_key="c2",
            ),
            _base_row(
                query_group_id="g_keep",
                query_id="q_keep",
                candidate_component_key="c2",
                candidate_cluster_id="c2",
                candidate_count=2,
                positive_candidate_count=1,
                positive_candidate_keys="c1",
                best_positive_retrieval_rank=1,
                label=0,
                retrieval_rank=2,
                best_competitor_component_key="c1",
            ),
        ],
        id(prepared_requests[1]): [
            _base_row(
                query_group_id="g_drop",
                query_id="q_drop",
                candidate_component_key="c3",
                candidate_cluster_id="c3",
                candidate_count=1,
                candidate_signatures=1,
                scored_candidate_components=1,
                scored_candidate_signatures=1,
                positive_candidate_count=1,
                positive_candidate_keys="c3",
                best_positive_retrieval_rank=1,
                label=1,
                retrieval_rank=1,
                best_competitor_component_key=None,
            )
        ],
    }

    def fake_compute_query_cluster_stats_batched(**kwargs: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        assert len(kwargs["requests"]) == 2
        return [
            ({}, {"pair_count": 2, "featurize_seconds": 1.25, "model_predict_seconds": 0.5}),
            ({}, {"pair_count": 1, "featurize_seconds": 0.75, "model_predict_seconds": 0.25}),
        ]

    def fake_materialize_query_rows_from_prepared(
        prepared_request: Any,
        *,
        stats_by_component: dict[str, Any],
        rust_hybrid_centroid_retriever: Any | None = None,
    ) -> list[dict[str, Any]]:
        del stats_by_component, rust_hybrid_centroid_retriever
        return list(materialized_rows_by_request[id(prepared_request)])

    monkeypatch.setattr(
        reranker_build,
        "compute_query_cluster_stats_batched",
        fake_compute_query_cluster_stats_batched,
    )
    monkeypatch.setattr(
        reranker_build,
        "_materialize_query_rows_from_prepared",
        fake_materialize_query_rows_from_prepared,
    )

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    pair_counts: list[int] = []
    featurize_seconds: list[float] = []
    model_seconds: list[float] = []
    filtered_stats = {"query_groups": 0, "rows": 0}
    summary_accumulator = reranker_build._QueryGroupSummaryAccumulator()  # noqa: SLF001

    written_rows, written_query_groups = reranker_build._flush_prepared_query_requests(  # noqa: SLF001
        clusterer=None,
        dataset=None,
        runtime_context=None,
        constraint_backend=None,
        prepared_requests=prepared_requests,
        pair_batch_size=64,
        max_top_k=250,
        pair_counts=pair_counts,
        featurize_seconds=featurize_seconds,
        model_seconds=model_seconds,
        rows=rows,
        query_group_metadata_rows=metadata_rows,
        query_group_summary_accumulator=summary_accumulator,
        min_candidates_per_query_group=2,
        filtered_query_group_stats=filtered_stats,
    )

    assert written_rows == 2
    assert written_query_groups == 1
    assert [row["query_group_id"] for row in rows] == ["g_keep", "g_keep"]
    assert [row["query_group_id"] for row in metadata_rows] == ["g_keep"]
    assert pair_counts == [2, 1]
    assert featurize_seconds == [1.25, 0.75]
    assert model_seconds == [0.5, 0.25]
    assert filtered_stats == {"query_groups": 1, "rows": 1}
    assert summary_accumulator.query_groups == 1
    assert summary_accumulator.row_count == 2


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


def test_reranker_dataset_build_parser_defaults_follow_top25_cache_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "labeled", "--output-dir", str(tmp_path)],
    )

    args = reranker_build.parse_args()

    assert args.query_views == list(reranker_utils.DEFAULT_QUERY_VIEWS)
    assert args.retrieval_engine == "auto"
    assert args.window_size == reranker_utils.DEFAULT_RETRIEVAL_WINDOW_SIZE
    assert args.max_top_k == reranker_utils.DEFAULT_CHOOSER_CACHE_MAX_TOP_K
    assert reranker_utils.DEFAULT_RETRIEVAL_WINDOW_SIZE == 25
    assert reranker_utils.DEFAULT_CHOOSER_CACHE_MAX_TOP_K == 25
    assert reranker_utils.DEFAULT_CANDIDATE_WINDOW_SENSITIVITY == (5, 25)
    assert reranker_utils.DEFAULT_H_WANG_WINDOW_SENSITIVITY == (5, 25)


def test_eval_single_letter_ranker_parser_defaults_to_default_feature_preset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset-root",
            str(tmp_path),
            "--window-size",
            "50",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    args = s2and_ranker_eval.parse_args()

    assert args.feature_preset == reranker_utils.DEFAULT_FEATURE_PRESET


def test_inner_split_group_id_prefers_audit_orcid_and_preserves_dataset_scope() -> None:
    assert (
        s2and_ranker_eval._inner_split_group_id(  # noqa: SLF001
            _base_row(dataset="d1", query_id="q1", _audit_normalized_orcid="0001", query_view="full")
        )
        == "d1:orcid:0001"
    )
    assert (
        s2and_ranker_eval._inner_split_group_id(  # noqa: SLF001
            _base_row(dataset="d2", query_id="q2", _audit_normalized_orcid="0001", query_view="initial_only")
        )
        == "d2:orcid:0001"
    )
    assert (
        s2and_ranker_eval._inner_split_group_id(  # noqa: SLF001
            _base_row(dataset="d1", query_id="q1", _audit_normalized_orcid=None, query_view="full")
        )
        == "d1:query:q1"
    )


def test_write_ranker_contracts_persists_feature_schema_and_bundle_contract(tmp_path: Path) -> None:
    schema = s2and_ranker_eval.FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="small_core_3")

    s2and_ranker_eval._write_ranker_contracts(tmp_path, {"feature_schema": schema.to_json_dict()})  # noqa: SLF001

    loaded_schema = s2and_ranker_eval.FeatureSchema.from_json_dict(
        json.loads((tmp_path / "feature_schema.json").read_text(encoding="utf-8"))
    )
    loaded_contract = s2and_ranker_eval.RerankerBundleContract.read_json(tmp_path / "bundle_contract.json")
    raw_contract = json.loads((tmp_path / "bundle_contract.json").read_text(encoding="utf-8"))
    assert loaded_schema.digest == schema.digest
    assert loaded_contract.feature_schema.digest == schema.digest
    assert loaded_contract.calibration_surface == "classic_gate_only"
    assert raw_contract["calibration_surface"] == "classic_gate_only"
    assert raw_contract["migration_manifest"]["calibration_surface"] == "classic_gate_only"


def test_write_ranker_contracts_rejects_corrupt_feature_schema_digest(tmp_path: Path) -> None:
    schema = s2and_ranker_eval.FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="small_core_3")
    corrupt_schema = schema.to_json_dict()
    corrupt_schema["digest"] = "bad"

    with pytest.raises(ValueError, match="Feature schema digest mismatch"):
        s2and_ranker_eval._write_ranker_contracts(tmp_path, {"feature_schema": corrupt_schema})  # noqa: SLF001

    assert not (tmp_path / "feature_schema.json").exists()
    assert not (tmp_path / "bundle_contract.json").exists()


def test_write_ranker_contracts_persists_heldout_calibrator_metadata(tmp_path: Path) -> None:
    schema = s2and_ranker_eval.FeatureSchema.from_columns(("retrieval_rank", "retrieval_score"), preset="small_core_3")
    train_summary = {
        "feature_schema": schema.to_json_dict(),
        "calibration": {
            "enabled": True,
            "surface": "ranker_heldout",
            "method": "isotonic",
            "feature_schema_digest": schema.digest,
        },
    }
    calibrator = s2and_ranker_eval.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit([0.0, 1.0], [0, 1])
    model = SimpleNamespace(s2and_score_calibrator_=calibrator)

    s2and_ranker_eval._write_ranker_contracts(tmp_path, train_summary)  # noqa: SLF001
    s2and_ranker_eval._write_calibrator_artifact(tmp_path, model, train_summary)  # noqa: SLF001

    loaded_contract = s2and_ranker_eval.RerankerBundleContract.read_json(tmp_path / "bundle_contract.json")
    payload = pickle.loads((tmp_path / "calibrator.pkl").read_bytes())
    assert loaded_contract.calibration_surface == "ranker_heldout"
    assert payload["feature_schema"]["digest"] == schema.digest
    assert json.loads((tmp_path / "calibrator_summary.json").read_text(encoding="utf-8"))["method"] == "isotonic"


def test_fit_ranker_for_split_keeps_mixed_views_on_same_side(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Any]] = []
    for query_id in ("q1", "q2", "q3", "q4"):
        for query_view in ("full", "initial_only"):
            group_id = f"d1:{query_id}:{query_view}"
            rows.append(
                _base_row(
                    dataset="d1",
                    query_id=query_id,
                    query_group_id=group_id,
                    query_view=query_view,
                    natural_query_view=query_view,
                    candidate_component_key=f"{query_id}:{query_view}:pos",
                    label=1,
                    retrieval_rank=1,
                    positive_candidate_count=1,
                    positive_candidate_keys=f"{query_id}:{query_view}:pos",
                    best_competitor_component_key=f"{query_id}:{query_view}:neg",
                )
            )
            rows.append(
                _base_row(
                    dataset="d1",
                    query_id=query_id,
                    query_group_id=group_id,
                    query_view=query_view,
                    natural_query_view=query_view,
                    candidate_component_key=f"{query_id}:{query_view}:neg",
                    label=0,
                    retrieval_rank=2,
                    best_competitor_component_key=f"{query_id}:{query_view}:pos",
                )
            )

    real_build_training_matrix = reranker_utils.build_training_matrix
    captured_training_split_ids: list[set[str]] = []
    captured_validation_split_ids: set[str] = set()

    def fake_build_training_matrix(input_rows: Any, **kwargs: Any) -> Any:
        captured_training_split_ids.append(
            {s2and_ranker_eval._inner_split_group_id(row) for row in input_rows}  # noqa: SLF001
        )
        return real_build_training_matrix(input_rows, **kwargs)

    class _FakeRanker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def fit(self, *args: Any, **kwargs: Any) -> _FakeRanker:
            del args, kwargs
            return self

    def fake_fit_ranker_with_hyperopt(
        *,
        validation_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del kwargs
        captured_validation_split_ids.update(
            {s2and_ranker_eval._inner_split_group_id(row) for row in validation_rows}  # noqa: SLF001
        )
        return object(), {"best_params": {}, "train_seconds": 0.0}

    monkeypatch.setattr(s2and_ranker_eval, "build_training_matrix", fake_build_training_matrix)
    monkeypatch.setattr(s2and_ranker_eval, "_fit_ranker_with_hyperopt", fake_fit_ranker_with_hyperopt)  # noqa: SLF001
    monkeypatch.setattr(s2and_ranker_eval, "LGBMRanker", _FakeRanker)

    _model, train_summary = s2and_ranker_eval._fit_ranker_for_split(  # noqa: SLF001
        train_rows=rows,
        query_views=["full", "initial_only"],
        window_size=2,
        seed=7,
        feature_preset="small_core_3",
        enrichment_profile="none",
        enrichment_rounds=0,
        hyperopt_evals=0,
        inner_validation_fraction=0.5,
        n_jobs=1,
    )

    assert captured_training_split_ids
    assert captured_training_split_ids[0].isdisjoint(captured_validation_split_ids)
    assert train_summary["query_views"] == ["full", "initial_only"]
    assert train_summary["feature_schema"]["digest"] == train_summary["feature_schema_digest"]


def test_fit_ranker_for_split_keeps_same_orcid_queries_on_same_side(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Any]] = []
    orcid_by_query = {"q1": "0001", "q2": "0001", "q3": "0003", "q4": None}
    for query_id, normalized_orcid in orcid_by_query.items():
        for query_view in ("full", "initial_only"):
            group_id = f"d1:{query_id}:{query_view}"
            rows.append(
                _base_row(
                    dataset="d1",
                    query_id=query_id,
                    _audit_normalized_orcid=normalized_orcid,
                    query_group_id=group_id,
                    query_view=query_view,
                    natural_query_view=query_view,
                    candidate_component_key=f"{query_id}:{query_view}:pos",
                    label=1,
                    retrieval_rank=1,
                    positive_candidate_count=1,
                    positive_candidate_keys=f"{query_id}:{query_view}:pos",
                    best_competitor_component_key=f"{query_id}:{query_view}:neg",
                )
            )
            rows.append(
                _base_row(
                    dataset="d1",
                    query_id=query_id,
                    _audit_normalized_orcid=normalized_orcid,
                    query_group_id=group_id,
                    query_view=query_view,
                    natural_query_view=query_view,
                    candidate_component_key=f"{query_id}:{query_view}:neg",
                    label=0,
                    retrieval_rank=2,
                    best_competitor_component_key=f"{query_id}:{query_view}:pos",
                )
            )

    real_build_training_matrix = reranker_utils.build_training_matrix
    captured_search_train_query_ids: set[str] = set()
    captured_validation_query_ids: set[str] = set()

    def fake_build_training_matrix(input_rows: Any, **kwargs: Any) -> Any:
        if not captured_search_train_query_ids:
            captured_search_train_query_ids.update(str(row["query_id"]) for row in input_rows)
        return real_build_training_matrix(input_rows, **kwargs)

    class _FakeRanker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def fit(self, *args: Any, **kwargs: Any) -> _FakeRanker:
            del args, kwargs
            return self

    def fake_fit_ranker_with_hyperopt(
        *,
        validation_rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del kwargs
        captured_validation_query_ids.update(str(row["query_id"]) for row in validation_rows)
        return object(), {"best_params": {}, "train_seconds": 0.0}

    monkeypatch.setattr(s2and_ranker_eval, "build_training_matrix", fake_build_training_matrix)
    monkeypatch.setattr(s2and_ranker_eval, "_fit_ranker_with_hyperopt", fake_fit_ranker_with_hyperopt)  # noqa: SLF001
    monkeypatch.setattr(s2and_ranker_eval, "LGBMRanker", _FakeRanker)

    _model, train_summary = s2and_ranker_eval._fit_ranker_for_split(  # noqa: SLF001
        train_rows=rows,
        query_views=["full", "initial_only"],
        window_size=2,
        seed=3,
        feature_preset="small_core_3",
        enrichment_profile="none",
        enrichment_rounds=0,
        hyperopt_evals=0,
        inner_validation_fraction=0.5,
        n_jobs=1,
    )

    assert {"q1", "q2"}.issubset(captured_search_train_query_ids) or {"q1", "q2"}.issubset(
        captured_validation_query_ids
    )
    assert train_summary["inner_split_group_policy"] == "dataset_orcid_or_query_id"


def test_fit_ranker_for_split_can_fit_heldout_calibrator(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, Any]] = []
    for index in range(8):
        query_id = f"q{index}"
        group_id = f"d1:{query_id}:initial_only"
        rows.append(
            _base_row(
                dataset="d1",
                query_id=query_id,
                query_group_id=group_id,
                candidate_component_key=f"{query_id}:pos",
                label=1,
                retrieval_rank=1,
                positive_candidate_count=1,
                positive_candidate_keys=f"{query_id}:pos",
                best_competitor_component_key=f"{query_id}:neg",
                min_distance=0.1,
                top3_mean_distance=0.1,
            )
        )
        rows.append(
            _base_row(
                dataset="d1",
                query_id=query_id,
                query_group_id=group_id,
                candidate_component_key=f"{query_id}:neg",
                label=0,
                retrieval_rank=2,
                best_competitor_component_key=f"{query_id}:pos",
                min_distance=0.8,
                top3_mean_distance=0.8,
            )
        )

    class _FakeRanker:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def fit(self, *args: Any, **kwargs: Any) -> _FakeRanker:
            del args, kwargs
            return self

        def predict(self, features: Any) -> np.ndarray:
            array = np.asarray(features, dtype=np.float64)
            return np.clip(1.0 - array[:, 0], 0.0, 1.0)

    def fake_fit_ranker_with_hyperopt(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
        del kwargs
        return object(), {"best_params": {}, "train_seconds": 0.0}

    monkeypatch.setattr(s2and_ranker_eval, "_fit_ranker_with_hyperopt", fake_fit_ranker_with_hyperopt)  # noqa: SLF001
    monkeypatch.setattr(s2and_ranker_eval, "LGBMRanker", _FakeRanker)

    model, train_summary = s2and_ranker_eval._fit_ranker_for_split(  # noqa: SLF001
        train_rows=rows,
        query_views=["initial_only"],
        window_size=2,
        seed=11,
        feature_preset="small_core_3",
        enrichment_profile="none",
        enrichment_rounds=0,
        hyperopt_evals=0,
        inner_validation_fraction=0.25,
        n_jobs=1,
        calibrator_mode="heldout",
    )

    calibration = train_summary["calibration"]
    assert hasattr(model, "s2and_score_calibrator_")
    assert calibration["enabled"] is True
    assert calibration["surface"] == "ranker_heldout"
    assert calibration["feature_schema_digest"] == train_summary["feature_schema_digest"]
    assert calibration["inner_split_group_overlap_with_training"] == 0
    assert train_summary["rows_reserved_for_calibration"] == calibration["rows"]
    assert train_summary["calibration_inner_split_group_count"] == calibration["groups"]
