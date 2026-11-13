from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.logistic_gate import (
    _PROBABILITY_FEATURES,
    build_logistic_gate_matrix,
    build_runtime_logistic_gate_matrix,
    default_logistic_gate_feature_names,
    load_logistic_gate_config,
    logistic_gate_config,
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

    assert [group.tolist() for group in query_rows.groups] == [[0, 2, 5], [1, 3], [4]]
    assert [group.tolist() for group in query_rows.ranked_groups] == [[2, 5, 0], [1, 3], [4]]
    np.testing.assert_array_equal(query_rows.best_rows, np.asarray([2, 1, 4], dtype=np.int64))
    np.testing.assert_allclose(query_rows.runner_up_scores[:2], np.asarray([0.8, 0.7], dtype=np.float64))
    assert np.isnan(query_rows.runner_up_scores[2])
    np.testing.assert_allclose(query_rows.score_margins[:2], np.asarray([0.0, 0.2], dtype=np.float64))
    assert np.isnan(query_rows.score_margins[2])
    np.testing.assert_array_equal(query_rows.has_runner_up, np.asarray([True, True, False]))
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


def _per_query_probability_feature(name: str, probs: np.ndarray) -> float:
    """Compute one probability feature with the per-query definition."""
    top = float(probs[0])
    second = float(probs[1]) if len(probs) > 1 else 0.0
    third = float(probs[2]) if len(probs) > 2 else 0.0
    prob_sum = float(np.sum(probs))
    if name == "prob_top2_share":
        return float(np.sum(probs[:2])) / prob_sum if prob_sum > 0 else 0.0
    if name == "prob_entropy":
        if len(probs) <= 1 or prob_sum <= 0.0:
            return 0.0
        shares = probs / prob_sum
        return -float(np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)))) / math.log(len(probs))
    if name == "prob_effective_candidates":
        if prob_sum <= 0.0:
            return 0.0
        shares = probs / prob_sum
        return math.exp(-float(np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0)))))
    simple = {
        "prob_top": top,
        "prob_second": second,
        "prob_third": third,
        "prob_top2_gap": top - second,
        "prob_second_third_gap": second - third,
        "prob_top_share": top / prob_sum if prob_sum > 0 else 0.0,
        "prob_sum": prob_sum,
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_p90": float(np.quantile(probs, 0.9)),
        "prob_p75": float(np.quantile(probs, 0.75)),
        "prob_count_ge_0_25": float((probs >= 0.25).sum()),
        "prob_count_ge_0_50": float((probs >= 0.50).sum()),
        "prob_count_ge_0_75": float((probs >= 0.75).sum()),
        "prob_count_within_0_01": float((probs >= top - 0.01).sum()),
        "prob_count_within_0_05": float((probs >= top - 0.05).sum()),
        "prob_count_within_0_10": float((probs >= top - 0.10).sum()),
    }
    return simple[name]


@pytest.mark.parametrize("seed", range(5))
def test_build_logistic_gate_matrix_matches_per_query_definitions(seed: int) -> None:
    """Grouped NumPy reductions must equal the per-query definitions bit for bit.

    Source x has missing values and takes the per-source path. Sources y and z have none and are
    reduced together, which needs contiguous rows to keep NumPy's pairwise sums.
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 40, size=30)
    lengths[:3] = [1, 2, 3]
    if seed == 4:
        # Many queries share each length, and one length has a single query.
        lengths = np.concatenate([rng.choice([9, 17], size=300), [5]])
    query_indices = np.repeat(np.arange(len(lengths)), lengths)
    rng.shuffle(query_indices)
    row_count = len(query_indices)
    probabilities = rng.random(row_count)
    if seed == 1:
        probabilities = rng.choice([0.0, -0.0, 0.25, 0.5, 0.75, 1.0], size=row_count)
    if seed == 2:
        probabilities[rng.random(row_count) < 0.1] = np.nan
    values = {}
    for source, specials in (
        ("x", [0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324]),
        ("y", [0.0, -0.0, 5e-324, 1e300, -1e300]),
        ("z", [0.0, -0.0, 1.0]),
    ):
        source_values = rng.normal(size=row_count) * 10.0 ** int(rng.integers(-3, 4))
        special = rng.random(row_count) < 0.25
        source_values[special] = rng.choice(specials, size=int(special.sum()))
        values[source] = source_values
    values["x"][query_indices == 0] = np.nan
    sources = ("x", "y", "z")
    prefixes = ("delta_top_second", "list_mean", "list_std", "list_min", "list_max")
    feature_names = [*_PROBABILITY_FEATURES] + [f"{prefix}_{source}" for source in sources for prefix in prefixes]

    # The 1e300 values overflow np.std in both implementations, which must still agree.
    with np.errstate(all="ignore"):
        matrix, query_rows = build_logistic_gate_matrix(
            feature_names,
            query_indices=query_indices,
            probabilities=probabilities,
            feature_values=values,
            retrieval_ranks=rng.integers(0, 9, size=row_count),
        )
        expected = np.full_like(matrix, np.nan)
        for query_pos, (group, ranked) in enumerate(zip(query_rows.groups, query_rows.ranked_groups, strict=True)):
            for column, name in enumerate(_PROBABILITY_FEATURES):
                expected[query_pos, column] = _per_query_probability_feature(name, probabilities[ranked])
            for source_position, source in enumerate(sources):
                column = len(_PROBABILITY_FEATURES) + source_position * len(prefixes)
                source_values = values[source]
                if len(ranked) > 1:
                    expected[query_pos, column] = float(source_values[ranked[0]] - source_values[ranked[1]])
                finite = source_values[group][np.isfinite(source_values[group])]
                if len(finite):
                    expected[query_pos, column + 1 : column + 5] = [
                        np.mean(finite),
                        np.std(finite),
                        np.min(finite),
                        np.max(finite),
                    ]

    same_bits = matrix.view(np.uint64) == expected.view(np.uint64)
    assert np.all(same_bits | (np.isnan(matrix) & np.isnan(expected)))
    all_missing = next(pos for pos, group in enumerate(query_rows.groups) if query_indices[group[0]] == 0)
    x_list_columns = slice(len(_PROBABILITY_FEATURES) + 1, len(_PROBABILITY_FEATURES) + 5)
    assert np.isnan(matrix[all_missing, x_list_columns]).all()
