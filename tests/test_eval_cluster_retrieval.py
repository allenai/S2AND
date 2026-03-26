from __future__ import annotations

from argparse import Namespace
from collections import Counter
from types import SimpleNamespace

import scripts.eval_cluster_retrieval as retrieval


def _query_features(
    *,
    middle_initials: frozenset[str] = frozenset(),
    year: int | None = None,
    orcid: str | None = None,
    has_coauthors: bool = False,
    has_affiliations: bool = False,
    has_full_first: bool = False,
    has_middle: bool = False,
) -> retrieval.QueryFeatures:
    return retrieval.QueryFeatures(
        first="a",
        middle="",
        first_initial="a",
        middle_initials=middle_initials,
        coauthor_blocks=frozenset({"a smith"}) if has_coauthors else frozenset(),
        affiliation_terms=frozenset({"lab"}) if has_affiliations else frozenset(),
        venue_terms=frozenset(),
        year=year,
        orcid=orcid,
        specter=None,
        has_specter=False,
        has_coauthors=has_coauthors,
        has_affiliations=has_affiliations,
        has_full_first=has_full_first,
        has_middle=has_middle,
    )


def _cluster_summary(
    *,
    component_key: str,
    size: int = 1,
    middle_initial_counts: Counter[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    year_mean: float | None = None,
    orcid_values: frozenset[str] = frozenset(),
) -> retrieval.ClusterSummary:
    return retrieval.ClusterSummary(
        component_key=component_key,
        cluster_id=component_key,
        block_key="b",
        size=size,
        first_name_counts=Counter(),
        middle_initial_counts=middle_initial_counts or Counter(),
        coauthor_counts=Counter(),
        affiliation_counts=Counter(),
        venue_counts=Counter(),
        year_values=[],
        year_min=year_min,
        year_max=year_max,
        year_mean=year_mean,
        orcid_values=orcid_values,
        specter_centroid=None,
        exemplar_vectors=[],
    )


def test_hybrid_scores_penalize_middle_initial_conflict():
    query = _query_features(middle_initials=frozenset({"a"}))
    empty_summary = _cluster_summary(component_key="empty", size=4)
    conflicting_summary = _cluster_summary(
        component_key="conflict",
        size=4,
        middle_initial_counts=Counter({"b": 1}),
    )

    centroid_empty = retrieval._score_summary("hybrid_centroid", query, empty_summary, max_block_component_size=4)
    centroid_conflict = retrieval._score_summary(
        "hybrid_centroid",
        query,
        conflicting_summary,
        max_block_component_size=4,
    )
    exemplar_empty = retrieval._score_summary("hybrid_exemplar_4", query, empty_summary, max_block_component_size=4)
    exemplar_conflict = retrieval._score_summary(
        "hybrid_exemplar_4",
        query,
        conflicting_summary,
        max_block_component_size=4,
    )

    assert centroid_conflict < centroid_empty
    assert exemplar_conflict < exemplar_empty


def test_apply_hard_filters_uses_orcid_middle_and_year_rules():
    query = _query_features(middle_initials=frozenset({"a"}), year=2000)
    good = _cluster_summary(
        component_key="good",
        size=3,
        middle_initial_counts=Counter({"a": 1}),
        year_min=1998,
        year_max=2002,
        year_mean=2000.0,
    )
    middle_conflict = _cluster_summary(
        component_key="middle_conflict",
        size=3,
        middle_initial_counts=Counter({"b": 1}),
        year_min=1998,
        year_max=2002,
        year_mean=2000.0,
    )
    year_conflict = _cluster_summary(
        component_key="year_conflict",
        size=3,
        middle_initial_counts=Counter(),
        year_min=1900,
        year_max=1910,
        year_mean=1905.0,
    )

    filtered, stats = retrieval._apply_hard_filters(query, [good, middle_conflict, year_conflict])

    assert [summary.component_key for summary in filtered] == ["good"]
    assert stats["middle_initial_filter_applied"] == 1
    assert stats["year_range_filter_applied"] == 1

    orcid_query = _query_features(orcid="orcid-1")
    orcid_match = _cluster_summary(component_key="orcid", size=2, orcid_values=frozenset({"orcid-1"}))
    non_match = _cluster_summary(component_key="other", size=2, orcid_values=frozenset({"orcid-2"}))
    filtered_orcid, stats_orcid = retrieval._apply_hard_filters(orcid_query, [orcid_match, non_match])

    assert [summary.component_key for summary in filtered_orcid] == ["orcid"]
    assert stats_orcid["orcid_filter_applied"] == 1


def test_documented_baselines_do_not_add_hidden_side_signals():
    query = _query_features(has_affiliations=True)
    no_coauthor = _cluster_summary(component_key="no_coauthor", size=2)
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
        retrieval._score_summary("coauthor_sparse", query, no_coauthor, max_block_component_size=10)
        == retrieval._score_summary("coauthor_sparse", query, affiliation_only, max_block_component_size=10)
        == 0.0
    )
    assert (
        retrieval._score_summary("specter_centroid", query, no_coauthor, max_block_component_size=10)
        == retrieval._score_summary("specter_centroid", query, affiliation_only, max_block_component_size=10)
        == 0.0
    )


def test_materialized_signature_count_uses_residual_summary_sizes():
    ranked_summaries = [
        _cluster_summary(component_key="c1", size=4),
        _cluster_summary(component_key="c2", size=3),
        _cluster_summary(component_key="c3", size=2),
    ]

    assert retrieval._materialized_signature_count(ranked_summaries, 1) == 4
    assert retrieval._materialized_signature_count(ranked_summaries, 2) == 7


def test_build_query_cases_counts_block_buckets_once_per_block(monkeypatch):
    dummy_features = _query_features()
    monkeypatch.setattr(retrieval, "_extract_query_features", lambda dataset, signature_id, **_: dummy_features)

    dataset = SimpleNamespace(
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
        "singleton": _query_features(has_full_first=True),
        "eligible_a": _query_features(has_coauthors=True),
        "eligible_b": _query_features(),
    }
    monkeypatch.setattr(
        retrieval,
        "_extract_query_features",
        lambda dataset, signature_id, **_: feature_by_signature[signature_id],
    )

    dataset = SimpleNamespace(
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

    dataset = SimpleNamespace(
        signatures={"s1": SimpleNamespace(paper_id="p1", author_info_orcid=None)},
        papers={"p1": SimpleNamespace(venue=None, journal_name=None, year=None)},
    )

    features = retrieval._extract_query_features(dataset, "s1")

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
