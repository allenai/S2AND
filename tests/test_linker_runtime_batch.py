from __future__ import annotations

import numpy as np
import pytest

from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.row_features import (
    build_promoted_non_pairwise_row_features,
    build_promoted_non_pairwise_row_features_with_telemetry,
)
from s2and.runtime import load_s2and_rust_extension
from tests.linker_row_feature_reference import build_promoted_non_pairwise_row_features_python_reference

s2and_rust = load_s2and_rust_extension()


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
