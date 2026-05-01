from __future__ import annotations

from argparse import Namespace
from collections import Counter
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import scripts.eval_cluster_retrieval as retrieval
from s2and.data import ANDData, NameCounts
from tests.helpers import build_cluster_summary, build_query_features


def test_hybrid_centroid_default_ranking_matches_golden_scores():
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
        coauthor_blocks=frozenset({"a smith"}),
        affiliation_terms=frozenset({"lab"}),
    )
    winner = build_cluster_summary(
        component_key="winner",
        size=4,
        first_name_counts=Counter({"alice": 4}),
        middle_initial_counts=Counter({"b": 4}),
        coauthor_counts=Counter({"a smith": 3}),
        affiliation_counts=Counter({"lab": 2}),
        specter_centroid=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    runner_up = build_cluster_summary(
        component_key="runner_up",
        size=4,
        first_name_counts=Counter({"alice": 4}),
        middle_initial_counts=Counter({"c": 4}),
        specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
    )

    ranked = retrieval.rank_summaries(
        "hybrid_centroid",
        query,
        [runner_up, winner],
        max_block_component_size=4,
    )

    assert [summary.component_key for _score, summary in ranked] == ["winner", "runner_up"]
    assert [score for score, _summary in ranked] == pytest.approx([0.7725, 0.0575])


def test_hybrid_scores_penalize_middle_initial_conflict():
    query = build_query_features(middle_initials=frozenset({"a"}))
    empty_summary = build_cluster_summary(component_key="empty", size=4)
    conflicting_summary = build_cluster_summary(
        component_key="conflict",
        size=4,
        middle_initial_counts=Counter({"b": 1}),
    )

    centroid_empty = retrieval.score_summary("hybrid_centroid", query, empty_summary, max_block_component_size=4)
    centroid_conflict = retrieval.score_summary(
        "hybrid_centroid",
        query,
        conflicting_summary,
        max_block_component_size=4,
    )
    exemplar_empty = retrieval.score_summary("hybrid_exemplar_4", query, empty_summary, max_block_component_size=4)
    exemplar_conflict = retrieval.score_summary(
        "hybrid_exemplar_4",
        query,
        conflicting_summary,
        max_block_component_size=4,
    )

    assert centroid_conflict < centroid_empty
    assert exemplar_conflict < exemplar_empty


def test_apply_hard_filters_uses_orcid_middle_and_year_rules():
    query = build_query_features(middle_initials=frozenset({"a"}), year=2000)
    good = build_cluster_summary(
        component_key="good",
        size=3,
        middle_initial_counts=Counter({"a": 1}),
        year_min=1998,
        year_max=2002,
        year_mean=2000.0,
    )
    middle_conflict = build_cluster_summary(
        component_key="middle_conflict",
        size=3,
        middle_initial_counts=Counter({"b": 1}),
        year_min=1998,
        year_max=2002,
        year_mean=2000.0,
    )
    year_conflict = build_cluster_summary(
        component_key="year_conflict",
        size=3,
        middle_initial_counts=Counter(),
        year_min=1900,
        year_max=1910,
        year_mean=1905.0,
    )

    filtered, stats = retrieval.apply_hard_filters(query, [good, middle_conflict, year_conflict])

    assert [summary.component_key for summary in filtered] == ["good"]
    assert stats["middle_initial_filter_applied"] == 1
    assert stats["year_range_filter_applied"] == 1

    orcid_query = build_query_features(orcid="orcid-1")
    orcid_match = build_cluster_summary(component_key="orcid", size=2, orcid_values=frozenset({"orcid-1"}))
    non_match = build_cluster_summary(component_key="other", size=2, orcid_values=frozenset({"orcid-2"}))
    filtered_orcid, stats_orcid = retrieval.apply_hard_filters(orcid_query, [orcid_match, non_match])

    assert [summary.component_key for summary in filtered_orcid] == ["orcid"]
    assert stats_orcid["orcid_filter_applied"] == 1


def test_orcid_disabled_query_features_strip_orcid_and_skip_filters(monkeypatch):
    monkeypatch.setattr(retrieval, "_signature_name_parts_for_subblocking", lambda signature: ("alice", "beth"))
    monkeypatch.setattr(retrieval, "_signature_coauthor_blocks_for_specter", lambda signature, dataset: [])
    monkeypatch.setattr(retrieval, "_signature_affiliation_feature_keys", lambda signature: [])
    monkeypatch.setattr(retrieval, "_get_specter_vector", lambda dataset, paper_id: None)

    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={"s1": SimpleNamespace(paper_id="p1", author_info_orcid="0000-0001")},
            papers={"p1": SimpleNamespace(venue=None, journal_name=None, year=None)},
        ),
    )

    enabled = retrieval.extract_query_features(dataset, "s1", orcid_enabled=True)
    disabled = retrieval.extract_query_features(dataset, "s1", orcid_enabled=False)
    default_disabled = retrieval.extract_query_features(dataset, "s1")

    assert enabled.orcid == "0000-0001"
    assert disabled.orcid is None
    assert default_disabled.orcid is None
    assert retrieval.mask_query_features(enabled, "full").orcid is None
    assert retrieval.mask_query_features(enabled, "full", orcid_enabled=False).orcid is None
    assert retrieval.mask_query_features(enabled, "initial_only", orcid_enabled=False).orcid is None
    assert (
        retrieval.build_cluster_summary(
            dataset=dataset,
            block_key="b",
            cluster_id="c",
            component_key="b::c",
            signature_ids=["s1"],
            max_exemplars=1,
        ).orcid_values
        == frozenset()
    )

    matching = build_cluster_summary(component_key="matching", size=2, orcid_values=frozenset({"0000-0001"}))
    non_matching = build_cluster_summary(component_key="other", size=2, orcid_values=frozenset({"0000-0002"}))
    filtered, stats = retrieval.apply_hard_filters(disabled, [matching, non_matching])

    assert [summary.component_key for summary in filtered] == ["matching", "other"]
    assert stats["orcid_filter_applied"] == 0


def test_mask_query_features_preserves_name_counts_for_rarity_features():
    name_counts = NameCounts(first=10, first_last=3, last=100, last_first_initial=12)
    query = build_query_features(
        first="alice",
        middle_initials=frozenset({"b"}),
        name_counts=name_counts,
    )

    masked = retrieval.mask_query_features(query, "initial_only")

    assert masked.first == "a"
    assert masked.middle_initials == frozenset()
    assert masked.name_counts == name_counts


def test_extract_query_features_cache_keeps_canonical_orcid_across_modes(monkeypatch):
    monkeypatch.setattr(retrieval, "_signature_name_parts_for_subblocking", lambda signature: ("alice", "beth"))
    monkeypatch.setattr(retrieval, "_signature_coauthor_blocks_for_specter", lambda signature, dataset: [])
    monkeypatch.setattr(retrieval, "_signature_affiliation_feature_keys", lambda signature: [])
    monkeypatch.setattr(retrieval, "_get_specter_vector", lambda dataset, paper_id: None)

    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={"s1": SimpleNamespace(paper_id="p1", author_info_orcid="0000-0001")},
            papers={"p1": SimpleNamespace(venue=None, journal_name=None, year=None)},
        ),
    )
    feature_cache = {}

    disabled = retrieval.extract_query_features(dataset, "s1", feature_cache=feature_cache, orcid_enabled=False)
    enabled = retrieval.extract_query_features(dataset, "s1", feature_cache=feature_cache, orcid_enabled=True)
    disabled_again = retrieval.extract_query_features(dataset, "s1", feature_cache=feature_cache, orcid_enabled=False)

    assert disabled.orcid is None
    assert enabled.orcid == "0000-0001"
    assert disabled_again.orcid is None
    assert feature_cache["s1"].orcid == "0000-0001"


def test_documented_baselines_do_not_add_hidden_side_signals():
    query = build_query_features(has_affiliations=True)
    no_coauthor = build_cluster_summary(component_key="no_coauthor", size=2)
    affiliation_only = retrieval.ClusterSummary(
        component_key="affiliation_only",
        cluster_id="affiliation_only",
        block_key="b",
        size=10,
        first_name_counts=Counter(),
        middle_initial_counts=Counter(),
        coauthor_counts=Counter(),
        affiliation_counts=Counter({"lab": 3}),
        venue_counts=Counter(),
        year_values=[],
        year_min=None,
        year_max=None,
        year_mean=None,
        orcid_values=frozenset(),
        specter_centroid=None,
        exemplar_vectors=[],
    )

    assert (
        retrieval.score_summary("coauthor_sparse", query, no_coauthor, max_block_component_size=10)
        == retrieval.score_summary("coauthor_sparse", query, affiliation_only, max_block_component_size=10)
        == 0.0
    )
    assert (
        retrieval.score_summary("specter_centroid", query, no_coauthor, max_block_component_size=10)
        == retrieval.score_summary("specter_centroid", query, affiliation_only, max_block_component_size=10)
        == 0.0
    )


def test_materialized_signature_count_uses_residual_summary_sizes():
    ranked_summaries = [
        build_cluster_summary(component_key="c1", size=4),
        build_cluster_summary(component_key="c2", size=3),
        build_cluster_summary(component_key="c3", size=2),
    ]

    assert retrieval._materialized_signature_count(ranked_summaries, 1) == 4
    assert retrieval._materialized_signature_count(ranked_summaries, 2) == 7


def test_build_query_cases_counts_block_buckets_once_per_block(monkeypatch):
    dummy_features = build_query_features()
    monkeypatch.setattr(retrieval, "extract_query_features", lambda dataset, signature_id, **_: dummy_features)

    dataset = cast(
        ANDData,
        SimpleNamespace(
            clusters={
                "c1": {"signature_ids": ["s1", "s2"]},
                "c2": {"signature_ids": ["s3", "s4"]},
            },
            signature_to_block={
                "s1": "block-a",
                "s2": "block-a",
                "s3": "block-a",
                "s4": "block-a",
            },
            signatures={signature_id: object() for signature_id in ["s1", "s2", "s3", "s4"]},
        ),
    )

    _cases, census, _block_to_component_keys, _components = retrieval._build_query_cases(
        dataset_name="dummy",
        dataset=dataset,
        limit_queries=10,
        seed=13,
        sampling_query_view="full",
    )

    assert census["blocks"] == 1
    assert census["block_size_buckets"] == Counter({"2_9": 1})


def test_build_query_cases_uses_signature_level_census_and_sampling_view(monkeypatch):
    feature_by_signature = {
        "singleton": build_query_features(has_full_first=True),
        "eligible_a": build_query_features(has_coauthors=True),
        "eligible_b": build_query_features(),
    }
    monkeypatch.setattr(
        retrieval,
        "extract_query_features",
        lambda dataset, signature_id, **_: feature_by_signature[signature_id],
    )

    dataset = cast(
        ANDData,
        SimpleNamespace(
            clusters={
                "c1": {"signature_ids": ["eligible_a", "eligible_b"]},
                "c2": {"signature_ids": ["singleton"]},
            },
            signature_to_block={
                "eligible_a": "block-a",
                "eligible_b": "block-a",
                "singleton": "block-b",
            },
            signatures={signature_id: object() for signature_id in feature_by_signature},
        ),
    )

    cases, census, _block_to_component_keys, _components = retrieval._build_query_cases(
        dataset_name="dummy",
        dataset=dataset,
        limit_queries=10,
        seed=13,
        sampling_query_view="initial_only_sparse_metadata",
    )

    assert census["signature_feature_counts"]["full_first"] == 1
    assert census["signature_feature_counts"]["coauthors"] == 1
    assert census["eligible_query_feature_counts"]["full_first"] == 0
    assert len(cases) == 1
    assert cases[0].initial_info_bucket == "sparse"


def test_extract_query_features_drops_empty_coauthor_blocks(monkeypatch):
    monkeypatch.setattr(retrieval, "_signature_name_parts_for_subblocking", lambda signature: ("alice", "beth"))
    monkeypatch.setattr(
        retrieval,
        "_signature_coauthor_blocks_for_specter",
        lambda signature, dataset: ["", "a smith", " "],
    )
    monkeypatch.setattr(retrieval, "_signature_affiliation_feature_keys", lambda signature: ["lab"])
    monkeypatch.setattr(retrieval, "_get_specter_vector", lambda dataset, paper_id: None)

    dataset = cast(
        ANDData,
        SimpleNamespace(
            signatures={"s1": SimpleNamespace(paper_id="p1", author_info_orcid=None)},
            papers={"p1": SimpleNamespace(venue=None, journal_name=None, year=None)},
        ),
    )

    features = retrieval.extract_query_features(dataset, "s1")

    assert features.coauthor_blocks == frozenset({"a smith"})
    assert features.has_coauthors is True
    assert features.has_middle is True


def test_build_summary_payload_reports_candidate_floor_slice():
    rows = [
        {
            "dataset": "dummy",
            "query_view": "initial_only",
            "method": "hybrid_centroid",
            "true_rank": 1,
            "candidate_components": 1,
            "candidate_signatures": 5,
            "scored_candidate_components": 1,
            "scored_candidate_signatures": 5,
            "latency_ms": 0.1,
            "query_feature_latency_ms": 0.02,
            "view_prepare_latency_ms": 0.03,
            "ranking_latency_ms": 0.05,
            "hit@1": 1,
            "hit@5": 1,
            "hit@10": 1,
            "hit@20": 1,
            "hit@50": 1,
            "hit@100": 1,
            "hit_budget@25": 1,
            "hit_budget@50": 1,
            "materialized_signatures@1": 5,
            "materialized_signatures@5": 5,
            "materialized_signatures@10": 5,
            "materialized_signatures@20": 5,
            "materialized_signatures@50": 5,
            "materialized_signatures@100": 5,
            "materialized_clusters@1": 1,
            "materialized_clusters@5": 1,
            "materialized_clusters@10": 1,
            "materialized_clusters@20": 1,
            "materialized_clusters@50": 1,
            "materialized_clusters@100": 1,
            "materialized_signature_fraction@1": 1.0,
            "materialized_signature_fraction@5": 1.0,
            "materialized_signature_fraction@10": 1.0,
            "materialized_signature_fraction@20": 1.0,
            "materialized_signature_fraction@50": 1.0,
            "materialized_signature_fraction@100": 1.0,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 0,
            "year_range_filter_applied": 0,
        },
        {
            "dataset": "dummy",
            "query_view": "initial_only",
            "method": "hybrid_centroid",
            "true_rank": 2,
            "candidate_components": 3,
            "candidate_signatures": 9,
            "scored_candidate_components": 2,
            "scored_candidate_signatures": 6,
            "latency_ms": 0.2,
            "query_feature_latency_ms": 0.02,
            "view_prepare_latency_ms": 0.03,
            "ranking_latency_ms": 0.15,
            "hit@1": 0,
            "hit@5": 1,
            "hit@10": 1,
            "hit@20": 1,
            "hit@50": 1,
            "hit@100": 1,
            "hit_budget@25": 1,
            "hit_budget@50": 1,
            "materialized_signatures@1": 3,
            "materialized_signatures@5": 6,
            "materialized_signatures@10": 6,
            "materialized_signatures@20": 6,
            "materialized_signatures@50": 6,
            "materialized_signatures@100": 6,
            "materialized_clusters@1": 1,
            "materialized_clusters@5": 2,
            "materialized_clusters@10": 2,
            "materialized_clusters@20": 2,
            "materialized_clusters@50": 2,
            "materialized_clusters@100": 2,
            "materialized_signature_fraction@1": 0.333333,
            "materialized_signature_fraction@5": 0.666667,
            "materialized_signature_fraction@10": 0.666667,
            "materialized_signature_fraction@20": 0.666667,
            "materialized_signature_fraction@50": 0.666667,
            "materialized_signature_fraction@100": 0.666667,
            "orcid_filter_applied": 0,
            "middle_initial_filter_applied": 1,
            "year_range_filter_applied": 0,
        },
    ]
    args = Namespace(
        datasets=["dummy"],
        query_views=["initial_only"],
        methods=["hybrid_centroid"],
        limit_queries=2,
        seed=13,
        n_jobs=1,
        sampling_query_view="initial_only_sparse_metadata",
        signature_budgets=[25, 50],
        disable_orcid_id=True,
    )

    summary = retrieval._build_summary_payload(args=args, all_rows=rows, diagnostics={})

    assert summary["overall"]["hybrid_centroid::initial_only"]["queries"] == 2
    assert summary["overall"]["hybrid_centroid::initial_only"]["mrr"] == 0.75
    assert summary["overall_candidate_floor"]["ge_3"]["hybrid_centroid::initial_only"]["queries"] == 1
    assert summary["overall"]["hybrid_centroid::initial_only"]["candidate_component_distribution"]["eq_1_rate"] == 0.5
    assert summary["overall"]["hybrid_centroid::initial_only"]["recall_under_signature_budget"]["25"] == 1.0
    assert (
        summary["overall"]["hybrid_centroid::initial_only"]["materialized_signature_fraction"]["5"]["mean"] == 0.833333
    )
    assert summary["config"]["orcid_enabled"] is False
    assert summary["config"]["orcid_mode"] == "disabled"


def test_failure_and_census_artifact_helpers():
    rows = [
        {"dataset": "d1", "hit@20": 1, "component_key": "ok"},
        {"dataset": "d1", "hit@20": 0, "component_key": "fail"},
    ]
    diagnostics = {
        "d1": {"census": {"blocks": 3}},
        "d2": {"error": "boom"},
    }

    assert retrieval._build_failures_rows(rows) == [rows[1]]
    assert retrieval._build_dataset_census_payload(diagnostics) == {"d1": {"blocks": 3}}
