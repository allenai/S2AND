from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.logistic_gate import (
    build_logistic_gate_matrix,
    build_runtime_logistic_gate_matrix,
    default_logistic_gate_feature_names,
    load_logistic_gate_config,
    logistic_gate_config,
    ranked_query_rows,
)


def test_load_logistic_gate_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported logistic gate model_type"):
        load_logistic_gate_config({"model_type": "unknown"})


def test_logistic_gate_config_rejects_feature_outside_runtime_safe_universe() -> None:
    with pytest.raises(ValueError, match="unsupported runtime features.*top_raw_unwired_source"):
        logistic_gate_config(
            feature_names=("top_raw_unwired_source",),
            weights=np.zeros((1, 3), dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
            missing_values=np.zeros(1, dtype=np.float64),
            calibration_mode="test",
        )


def test_load_logistic_gate_config_rejects_feature_outside_runtime_safe_universe() -> None:
    config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.zeros((1, 3), dtype=np.float64),
        bias=np.zeros(3, dtype=np.float64),
        missing_values=np.zeros(1, dtype=np.float64),
        calibration_mode="test",
    )
    config["feature_names"] = ["top_raw_unwired_source"]

    with pytest.raises(ValueError, match="unsupported runtime features.*top_raw_unwired_source"):
        load_logistic_gate_config(config)


def test_ranked_query_rows_groups_in_first_seen_order_without_repeated_scans() -> None:
    query_rows = ranked_query_rows(
        np.asarray(["q2", "q1", "q2", "q1", "q3", "q2"], dtype=object),
        np.asarray([0.4, 0.9, 0.8, 0.7, 0.1, 0.8], dtype=np.float64),
        retrieval_ranks=np.asarray([2, 1, 1, 2, 1, 3], dtype=np.int64),
        component_keys=cast(Any, np.asarray(["b", "a", "a", "b", "z", "c"], dtype=object)),
    )

    assert [group.tolist() for group in query_rows.groups] == [[0, 2, 5], [1, 3], [4]]
    assert [group.tolist() for group in query_rows.ranked_groups] == [[2, 5, 0], [1, 3], [4]]
    np.testing.assert_array_equal(query_rows.best_rows, np.asarray([2, 1, 4], dtype=np.int64))
    np.testing.assert_allclose(query_rows.runner_up_scores[:2], np.asarray([0.8, 0.7], dtype=np.float64))
    assert np.isnan(query_rows.runner_up_scores[2])
    np.testing.assert_allclose(query_rows.score_margins[:2], np.asarray([0.0, 0.2], dtype=np.float64))
    assert np.isnan(query_rows.score_margins[2])
    np.testing.assert_array_equal(query_rows.has_runner_up, np.asarray([True, True, False]))


def test_build_logistic_gate_matrix_covers_feature_families() -> None:
    feature_names = (
        "chosen_probability",
        "second_probability",
        "score_margin",
        "has_runner_up",
        "raw_candidate_count",
        "prob_top",
        "prob_second",
        "prob_third",
        "prob_top2_gap",
        "prob_top2_share",
        "prob_count_within_0_05",
        "candidate_kind_multi_candidate",
        "candidate_kind_single_candidate",
        "first_name_bucket_multi_letter_first",
        "query_view_full",
        "gate_bucket_multi_candidate|multi_letter_first",
        "top_raw_min_distance",
        "delta_top_second_min_distance",
        "list_mean_min_distance",
        "list_std_min_distance",
        "list_min_min_distance",
        "list_max_min_distance",
        "top_meta_retrieval_rank",
        "top_meta_query_first_token_len",
        "top_meta_query_author_len",
    )
    probabilities = np.asarray([0.4, 0.9, 0.8, 0.7, 0.1, 0.8], dtype=np.float64)
    feature_values = {
        "retrieval_rank": np.asarray([2, 1, 1, 2, 1, 3], dtype=np.int64),
        "first_name_bucket": np.asarray(
            [
                "single_letter_first",
                "single_letter_first",
                "multi_letter_first",
                "single_letter_first",
                "multi_letter_first",
                "multi_letter_first",
            ],
            dtype=object,
        ),
        "query_view": np.asarray(["full", "initial_only", "full", "initial_only", "full", "full"], dtype=object),
        "query_first_token": np.asarray(["A", "B", "Ada", "B", None, "Ada"], dtype=object),
        "query_author": np.asarray(["A Z", "B C", "Ada Lovelace", "B C", None, "Ada L"], dtype=object),
        "min_distance": np.asarray([10.0, 1.0, 4.0, 3.0, 7.0, 5.0], dtype=np.float64),
    }

    matrix, query_rows = build_logistic_gate_matrix(
        feature_names,
        query_indices=np.asarray(["q2", "q1", "q2", "q1", "q3", "q2"], dtype=object),
        probabilities=probabilities,
        feature_values=feature_values,
        retrieval_ranks=feature_values["retrieval_rank"],
        component_keys=cast(Any, np.asarray(["b", "a", "a", "b", "z", "c"], dtype=object)),
    )

    assert [group.tolist() for group in query_rows.ranked_groups] == [[2, 5, 0], [1, 3], [4]]
    expected = np.asarray(
        [
            [
                0.8,
                0.8,
                0.0,
                1.0,
                3.0,
                0.8,
                0.8,
                0.4,
                0.0,
                0.8,
                2.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                4.0,
                -1.0,
                19.0 / 3.0,
                np.std(np.asarray([10.0, 4.0, 5.0], dtype=np.float64)),
                4.0,
                10.0,
                1.0,
                3.0,
                12.0,
            ],
            [
                0.9,
                0.7,
                0.2,
                1.0,
                2.0,
                0.9,
                0.7,
                0.0,
                0.2,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                -2.0,
                2.0,
                1.0,
                1.0,
                3.0,
                1.0,
                1.0,
                3.0,
            ],
            [
                0.1,
                np.nan,
                np.nan,
                0.0,
                1.0,
                0.1,
                0.0,
                0.0,
                0.1,
                1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                7.0,
                np.nan,
                7.0,
                0.0,
                7.0,
                7.0,
                1.0,
                0.0,
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(matrix, expected)


def test_build_logistic_gate_matrix_treats_nan_query_text_as_missing() -> None:
    matrix, _query_rows = build_logistic_gate_matrix(
        ("top_meta_query_first_token_len", "top_meta_query_author_len"),
        query_indices=np.asarray(["q1", "q2"], dtype=object),
        probabilities=np.asarray([0.9, 0.8], dtype=np.float64),
        feature_values={
            "query_first_token": np.asarray([np.nan, None], dtype=object),
            "query_author": np.asarray([np.float32(np.nan), None], dtype=object),
        },
    )

    np.testing.assert_allclose(matrix, np.zeros((2, 2), dtype=np.float64))


def test_build_logistic_gate_matrix_rejects_duplicate_feature_names() -> None:
    with pytest.raises(ValueError, match="feature_names must be unique"):
        build_logistic_gate_matrix(
            ("chosen_probability", "chosen_probability"),
            query_indices=np.asarray(["q1", "q2"], dtype=object),
            probabilities=np.asarray([0.9, 0.8], dtype=np.float64),
            feature_values={},
        )


def test_build_runtime_logistic_gate_matrix_rejects_absent_selected_numeric_source() -> None:
    gate = load_logistic_gate_config(
        logistic_gate_config(
            feature_names=("top_raw_min_distance",),
            weights=np.zeros((1, 3), dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
            missing_values=np.zeros(1, dtype=np.float64),
            calibration_mode="test",
        )
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([0], dtype=np.uint32),
        row_component_keys=("candidate",),
    )
    feature_matrix = LinkerFeatureMatrix(
        matrix=np.empty((1, 0), dtype=np.float32),
        feature_columns=(),
        candidate_batch=candidate_batch,
    )

    with pytest.raises(ValueError, match="top_raw_min_distance.*requires.*min_distance"):
        build_runtime_logistic_gate_matrix(
            gate,
            feature_matrix,
            probabilities=np.asarray([0.9], dtype=np.float64),
            row_signals=None,
        )


def test_build_logistic_gate_matrix_rejects_absent_gate_bucket_source() -> None:
    with pytest.raises(ValueError, match="gate_bucket_.*require.*first_name_bucket"):
        build_logistic_gate_matrix(
            ("gate_bucket_single_candidate|multi_letter_first",),
            query_indices=np.asarray(["q1"], dtype=object),
            probabilities=np.asarray([0.9], dtype=np.float64),
            feature_values={},
        )


def test_default_logistic_gate_feature_names_excludes_unwired_runtime_features() -> None:
    names = set(default_logistic_gate_feature_names())
    unwired = {
        "cluster_size_log_capped",
        "first_name_exact_match",
        "paper_author_position_jaccard_score",
        "pw_min_jaro",
        "pw_min_last_first_initial_count_min",
    }
    derived = {
        f"{prefix}_{feature}"
        for prefix in ("top_raw", "delta_top_second", "list_mean", "list_std", "list_min", "list_max")
        for feature in unwired
    }

    assert names.isdisjoint(derived)
