"""Regression tests for promoted train/calibrate/eval helper functions."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_joint_safe_link_promoted_train_calibrate_eval import (
    OfficialBundle,
    _apply_classic_gate,
    _apply_classic_train_holdout_filter,
    _apply_classic_train_row_cap,
    _augmented_feature_matrix,
    _classic_feature_matrix,
    _evaluate_classic_manual_holdout,
    _fit_promoted_stratified_total_error_gate,
    _fit_score_margin_gate,
    _fit_single_candidate_score_gate,
    _iter_extra_eval_paths,
    _normalize_augmented_feature_frame,
    _resolve_classic_monotone_constraints,
    _resolve_path,
    _score_abstain_rule,
    _score_classic_stratified_eval_test_choices,
    _score_eval_candidate_rows,
    _summarize_classic_stratified_predictions,
    _summarize_training_gate_buckets,
    _summary_key_for_eval_dataset,
    format_classic_selected_gate_tables,
    load_bundle,
)
from scripts.run_joint_safe_link_promoted_train_calibrate_eval import (
    _read_csv as _read_official_table,
)


def test_normalize_augmented_feature_frame_derives_query_first_features() -> None:
    """Augmented frame normalization should derive missing first-name features."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "query_view": "full",
            },
            {
                "query_author": "H. Wang",
                "dominant_first_name": "huijuan",
                "query_view": "initial_only",
            },
        ]
    )
    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=("query_first_prefix_match_any_length",),
    )
    assert out["query_first_prefix_match_any_length"].tolist() == [1.0, 1.0]


def test_normalize_augmented_feature_frame_overwrites_stale_runtime_features() -> None:
    """Runtime-derived features should be recomputed from raw prerequisites when available."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "cluster_size": 17,
                "query_first_prefix_match_any_length": 0.0,
                "cluster_size_log": 0.0,
            }
        ]
    )

    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=("query_first_prefix_match_any_length", "cluster_size_log"),
    )

    assert out.iloc[0]["query_first_prefix_match_any_length"] == 1.0
    assert out.iloc[0]["cluster_size_log"] == pytest.approx(np.log1p(17))


def test_normalize_augmented_feature_frame_derives_anchor_evidence_features() -> None:
    """Promoted anchor formulas should be derived from raw candidate-row evidence."""

    df = pd.DataFrame(
        [
            {
                "min_distance": 0.1,
                "specter_exemplar_similarity": 0.8,
                "title_overlap": 0.3,
                "coauthor_overlap": 0.3,
                "affiliation_overlap": 0.1,
                "venue_overlap": 0.4,
                "year_compatibility": 0.95,
                "retrieval_score_gap_vs_best_competitor": 0.06,
                "candidate_contradiction_score": 0.2,
                "same_family_as_top1": 1.0,
                "candidate_pair_share_within_coarse_family": 0.25,
                "cluster_size": 2,
                "named_signature_count": 2,
                "retrieval_rank": 1,
                "anchor_evidence_count": 0.0,
            }
        ]
    )

    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=(
            "anchor_evidence_count",
            "strong_positive_anchor_score",
            "weak_residual_anchor_score",
            "sparse_relative_winner_score",
        ),
    )

    assert out.iloc[0]["anchor_evidence_count"] == pytest.approx(2.0)
    assert out.iloc[0]["strong_positive_anchor_score"] == pytest.approx(0.18, abs=1e-6)
    assert out.iloc[0]["weak_residual_anchor_score"] == pytest.approx(0.2936, abs=1e-6)
    assert out.iloc[0]["sparse_relative_winner_score"] == pytest.approx(0.05872, abs=1e-6)


def test_normalize_augmented_feature_frame_derives_promoted_primitives() -> None:
    """Promoted runtime primitives should be recomputed from raw evidence."""

    df = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "query_author": "C. Zhang",
                "candidate_component_key": "c1",
                "dominant_first_name": "chen",
                "retrieval_rank": 2,
                "retrieval_score": 0.75,
                "top5_mean_distance": 0.4,
                "cluster_size": 200,
                "candidate_year_min": 2010,
                "candidate_year_max": 2015,
                "candidate_year_range_missing": 0,
                "query_year": 2007,
                "query_year_missing": 0,
            }
        ]
    )

    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=(
            "retrieval_reciprocal_rank",
            "cluster_size_log",
            "candidate_year_span",
            "year_gap_to_candidate_range",
            "year_gap_signed_to_candidate_range",
            "candidate_dominant_first_name_length",
            "query_first_prefix_match_any_length",
            "same_dominant_first_as_best_top5",
            "same_family_as_heuristic_choice",
        ),
    )

    assert out.iloc[0]["retrieval_reciprocal_rank"] == pytest.approx(0.5)
    assert out.iloc[0]["cluster_size_log"] == pytest.approx(np.log1p(200))
    assert out.iloc[0]["candidate_year_span"] == pytest.approx(5.0)
    assert out.iloc[0]["year_gap_to_candidate_range"] == pytest.approx(3.0)
    assert out.iloc[0]["year_gap_signed_to_candidate_range"] == pytest.approx(-3.0)
    assert out.iloc[0]["candidate_dominant_first_name_length"] == pytest.approx(4.0)
    assert out.iloc[0]["query_first_prefix_match_any_length"] == pytest.approx(1.0)
    assert out.iloc[0]["same_dominant_first_as_best_top5"] == pytest.approx(1.0)
    assert out.iloc[0]["same_family_as_heuristic_choice"] == pytest.approx(1.35)


def test_classic_feature_matrix_uses_query_first_token_without_query_author() -> None:
    """Public eval files should derive prefix match from query_first_token."""

    df = pd.DataFrame(
        [
            {"query_first_token": "Hanbing", "dominant_first_name": "hanbing"},
            {"dominant_first_name": "hanbing"},
        ]
    )

    features = _classic_feature_matrix(df, ("query_first_prefix_match_any_length",))

    assert features["query_first_prefix_match_any_length"].tolist() == [1.0, 0.0]


def test_official_table_loader_reads_parquet_with_usecols(tmp_path: Path) -> None:
    """Official stack loaders should accept parquet paths from the three-phase bundle."""

    path = tmp_path / "rows.parquet"
    pd.DataFrame([{"query_group_id": "001", "label": 1, "min_distance": 0.2}]).to_parquet(path, index=False)

    loaded = _read_official_table(path, usecols=["query_group_id", "label"])

    assert loaded.to_dict(orient="records") == [{"query_group_id": "001", "label": 1}]


def test_promoted_stratified_loader_prefers_public_rows_over_shadowing_calibration_rows(
    tmp_path: Path,
) -> None:
    """Calibration-source rows must not duplicate active public test rows."""

    bundle_root = tmp_path / "bundle"
    calibration_path = bundle_root / "calibration" / "gate_rows.csv.gz"
    hwang_path = bundle_root / "test" / "hwang_eval_rows.csv.gz"
    s2and_path = bundle_root / "test" / "s2and_eval_rows.csv.gz"
    split_path = bundle_root / "calibration" / "stratified_eval_test_split" / "combined_query_split_assignments.csv"
    calibration_path.parent.mkdir(parents=True)
    hwang_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)

    row_columns = ["query_group_id", "dataset", "query_view", "candidate_component_key", "retrieval_rank", "label"]
    pd.DataFrame(
        [
            {
                "query_group_id": "h:q1",
                "dataset": "h_wang",
                "query_view": "full",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 0,
            }
        ],
        columns=row_columns,
    ).to_csv(calibration_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "query_group_id": "h:q1",
                "dataset": "h_wang",
                "query_view": "full",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 1,
            }
        ],
        columns=row_columns,
    ).to_csv(hwang_path, index=False, compression="gzip")
    pd.DataFrame([], columns=row_columns).to_csv(s2and_path, index=False, compression="gzip")
    pd.DataFrame(
        [
            {
                "query_group_id": "h:q1",
                "source_key": "hwang_eval",
                "split": "test",
                "source_stratum": "hwang_block",
                "has_positive_candidate": False,
                "positive_rank_bucket": "no_positive",
                "first_name_bucket": "multi_letter_first",
                "multiple_candidates": False,
                "stratum_key": "stale",
            }
        ]
    ).to_csv(split_path, index=False)
    spec = {
        "classic_gate_source_path": "calibration/gate_rows.csv.gz",
        "s2and_eval_path": "test/s2and_eval_rows.csv.gz",
        "hwang_eval_path": "test/hwang_eval_rows.csv.gz",
    }
    bundle = OfficialBundle(
        root=bundle_root,
        bundle_name="demo",
        assets={},
        models={"classic": spec},
        expected_metrics={},
    )

    class AlwaysLinkModel:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.repeat([[0.0, 1.0]], repeats=len(features), axis=0)

    choices, assignments = _score_classic_stratified_eval_test_choices(
        bundle,
        spec,
        {"assignments_path": "calibration/stratified_eval_test_split/combined_query_split_assignments.csv"},
        AlwaysLinkModel(),  # type: ignore[arg-type]
        (),
    )

    choice = choices.set_index("query_case_id").loc["h:q1"]
    assignment = assignments.set_index("query_group_id").loc["h:q1"]
    assert choice["chosen_candidate_target"] == 1
    assert choice["query_safe_target"] == 1
    assert choice["positive_rank_bucket"] == "positive_first"
    assert assignment["positive_rank_bucket"] == "positive_first"


def test_fit_score_margin_gate_finds_perfect_gate() -> None:
    """Score+margin fitting should recover a gate with perfect balanced accuracy when available."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q1",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.2,
            },
            {
                "query_case_id": "q2",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.05,
            },
        ]
    )
    fitted = _fit_score_margin_gate(
        query_choices,
        reference_score_threshold=0.8,
        reference_margin_threshold=0.15,
        score_grid_size=5,
        margin_grid_size=5,
    )
    assert fitted["metrics"]["balanced_accuracy"] == 1.0


def test_fit_single_candidate_score_gate_finds_perfect_gate() -> None:
    """Single-candidate gate fitting should learn a score-only abstain threshold."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_pos",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_neg",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
        ]
    )

    fitted = _fit_single_candidate_score_gate(
        query_choices,
        reference_score_threshold=0.5,
        score_grid_size=5,
    )

    assert fitted["metrics"]["balanced_accuracy"] == 1.0
    assert 0.1 < fitted["single_candidate_score_threshold"] <= 0.9


def test_apply_classic_gate_uses_single_candidate_threshold() -> None:
    """Missing runner-up margin should use the calibrated single-candidate gate."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_neg_single",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_pos_single",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_multi",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.4,
                "score_margin": 0.05,
                "has_runner_up": 1,
            },
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.8,
        margin_threshold=0.1,
        single_candidate_score_threshold=0.5,
    ).set_index("query_case_id")

    assert predictions.loc["q_neg_single", "predicted_action"] == "abstain"
    assert predictions.loc["q_neg_single", "correct"] == 1
    assert predictions.loc["q_pos_single", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_pos_single", "correct"] == 1
    assert predictions.loc["q_multi", "predicted_action"] == "abstain"


def test_apply_classic_gate_supports_bucketed_thresholds() -> None:
    """Fixed bucketed gates should use candidate kind and first-name length."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_multi_margin_rescue",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.2,
                "score_margin": 0.6,
                "has_runner_up": 1,
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_case_id": "q_single_letter_margin_not_rescued",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.2,
                "score_margin": 0.6,
                "has_runner_up": 1,
                "first_name_bucket": "single_letter_first",
            },
            {
                "query_case_id": "q_single_letter_accept",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.5,
                "score_margin": np.nan,
                "has_runner_up": 0,
                "first_name_bucket": "single_letter_first",
            },
            {
                "query_case_id": "q_single_multi_abstain",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.5,
                "score_margin": np.nan,
                "has_runner_up": 0,
                "first_name_bucket": "multi_letter_first",
            },
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.8,
        margin_threshold=0.4,
        single_candidate_score_threshold=0.7,
        bucketed_score_thresholds={
            "multi_candidate|multi_letter_first": 0.9,
            "multi_candidate|single_letter_first": 0.3,
            "single_candidate|multi_letter_first": 0.7,
            "single_candidate|single_letter_first": 0.4,
        },
        bucketed_margin_threshold=0.4,
        bucketed_margin_thresholds={
            "multi_candidate|multi_letter_first": 0.4,
            "multi_candidate|single_letter_first": 0.7,
        },
    ).set_index("query_case_id")

    assert predictions.loc["q_multi_margin_rescue", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_single_letter_margin_not_rescued", "predicted_action"] == "abstain"
    assert predictions.loc["q_single_letter_accept", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_single_multi_abstain", "predicted_action"] == "abstain"


def test_promoted_total_error_gate_fits_fixed_grid_on_full_calibration() -> None:
    """Promoted gate calibration should fit one fixed-grid gate on all calibration splits."""

    choices = pd.DataFrame(
        [
            {
                "query_case_id": "fit_pos",
                "split": "calibration_fit",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.2,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "fit_neg",
                "split": "calibration_fit",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.01,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "check_pos",
                "split": "calibration_check",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.85,
                "score_margin": 0.15,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "check_neg",
                "split": "calibration_check",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.05,
                "score_margin": 0.0,
                "has_runner_up": 1,
            },
        ]
    )
    config = {
        "mode": "full_calibration_fixed_grid_4score_2margin",
        "calibration_splits": ["calibration_fit", "calibration_check"],
        "test_split": "test",
        "fixed_grid_step": 0.1,
    }

    result = _fit_promoted_stratified_total_error_gate(choices, config)

    assert result["gate"].name == "full_calibration_fixed_grid_0.1"
    assert result["calibration_metrics"]["errors"] == 0
    assert result["calibration_split_metrics"]["calibration_fit"]["errors"] == 0
    assert result["calibration_split_metrics"]["calibration_check"]["errors"] == 0
    assert result["threshold_grid_points"] == 11


def test_promoted_gate_selects_on_weighted_error_not_total_errors() -> None:
    """Wrong links should be costlier than false abstains during promoted gate calibration."""

    rows: list[dict[str, object]] = []
    for index, (query_target, candidate_target, score) in enumerate(
        [
            (1, 1, 0.9),
            (1, 1, 0.5),
            (1, 1, 0.5),
            (1, 1, 0.5),
            (0, 0, 0.5),
            (1, 0, 0.5),
        ]
    ):
        rows.append(
            {
                "query_case_id": f"fit_{index}",
                "split": "calibration_fit",
                "candidate_kind": "single_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": query_target,
                "chosen_candidate_target": candidate_target,
                "chosen_probability": score,
                "score_margin": np.nan,
                "has_runner_up": 0,
            }
        )
    config = {
        "mode": "full_calibration_fixed_grid_4score_2margin",
        "calibration_splits": ["calibration_fit"],
        "test_split": "test",
        "fixed_grid_step": 0.1,
    }

    result = _fit_promoted_stratified_total_error_gate(pd.DataFrame(rows), config)

    bucket = "single_candidate|multi_letter_first"
    assert result["gate"].score_thresholds[bucket] == pytest.approx(0.6)
    assert result["selection_key"]["calibration_weighted_average_error"] == pytest.approx((4 * 0.25) / (6 * 2.75))
    assert result["selection_key"]["calibration_errors"] == 4
    assert result["bucket_metrics"][bucket]["errors"] == 4


def test_apply_classic_gate_requires_positive_query_target_for_correct_link() -> None:
    """A stale positive candidate row should not overcredit a query-level manual negative."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "manual_negative_with_stale_positive_row",
                "query_safe_target": 0,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.99,
                "score_margin": 0.9,
                "has_runner_up": 1,
            }
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.5,
        margin_threshold=0.5,
    )

    assert predictions.iloc[0]["predicted_action"] == "link_candidate"
    assert predictions.iloc[0]["correct"] == 0


def test_score_abstain_rule_uses_repaired_query_targets() -> None:
    """Internal gate diagnostics should follow final targets, not stale supervision type."""

    rows = pd.DataFrame(
        [
            {
                "query_case_id": "q_repaired_singleton",
                "dataset": "demo",
                "supervision_type": "negative_singleton_orcid",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.8,
                "has_runner_up": 1,
                "top1_correct": 1,
            },
            {
                "query_case_id": "q_true_negative",
                "dataset": "demo",
                "supervision_type": "negative_singleton_orcid",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.1,
                "has_runner_up": 1,
                "top1_correct": 0,
            },
        ]
    )

    metrics = _score_abstain_rule(rows, score_threshold=0.5, margin_threshold=0.5)

    assert metrics["positive_queries"] == 1
    assert metrics["negative_queries"] == 1
    assert metrics["balanced_accuracy"] == 1.0


def test_apply_classic_train_row_cap_preserves_positive_queries() -> None:
    """Classic row-cap filtering should keep every positive query represented."""

    train_df = pd.DataFrame(
        [
            {"query_group_id": "q1", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q1", "retrieval_rank": 2, "label": 0},
            {"query_group_id": "q1", "retrieval_rank": 6, "label": 1},
            {"query_group_id": "q1", "retrieval_rank": 8, "label": 0},
            {"query_group_id": "q2", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q2", "retrieval_rank": 3, "label": 1},
            {"query_group_id": "q2", "retrieval_rank": 7, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 5, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 7, "label": 0},
        ]
    )
    selected, summary = _apply_classic_train_row_cap(
        train_df,
        rule_name="max_of_min_limit_and_first_positive_rank",
        min_train_limit=5,
    )
    assert selected.groupby("query_group_id", sort=False)["retrieval_rank"].max().to_dict() == {
        "q1": 6,
        "q2": 3,
        "q3": 5,
    }
    assert summary is not None
    assert summary["lost_positive_queries"] == 0
    assert summary["positive_rows_after"] == 2


def test_apply_classic_train_holdout_filter_removes_eval_identities() -> None:
    """Classic training should drop rows that share query or base IDs with held-out rows."""

    train_df = pd.DataFrame(
        [
            {"query_group_id": "q_keep", "base_group_id": "b_keep", "label": 1},
            {"query_group_id": "q_exact", "base_group_id": "b_train", "label": 1},
            {"query_group_id": "q_base_pos", "base_group_id": "b_eval", "label": 1},
            {"query_group_id": "q_base_neg", "base_group_id": "b_eval", "label": 0},
        ]
    )

    filtered, summary = _apply_classic_train_holdout_filter(
        train_df,
        holdout_query_group_ids={"q_exact"},
        holdout_base_group_ids={"b_eval"},
        holdout_sources=[{"source": "demo", "query_groups": 1, "base_groups": 1}],
    )

    assert filtered["query_group_id"].tolist() == ["q_keep"]
    assert summary["rows_removed"] == 3
    assert summary["queries_removed"] == 3
    assert summary["positive_rows_removed"] == 2
    assert summary["positive_queries_removed"] == 2
    assert summary["overlapping_query_groups"] == 1
    assert summary["overlapping_base_groups"] == 1
    assert summary["holdout_sources"] == [{"source": "demo", "query_groups": 1, "base_groups": 1}]


def test_evaluate_classic_manual_holdout_scores_fresh_candidates() -> None:
    """Classic manual holdout evaluation should use fresh candidate scores, not frozen top1 metadata."""

    manual_holdout = pd.DataFrame(
        [
            {
                "query_case_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "rescue",
                "candidate_component_key": "wrong",
                "retrieval_rank": 1,
                "binary_safe_link_target": 0,
            },
            {
                "query_case_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "rescue",
                "candidate_component_key": "right",
                "retrieval_rank": 2,
                "binary_safe_link_target": 1,
            },
            {
                "query_case_id": "q2",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "easy",
                "candidate_component_key": "negative",
                "retrieval_rank": 1,
                "binary_safe_link_target": 0,
            },
            {
                "query_case_id": "q2",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "easy",
                "candidate_component_key": "distractor",
                "retrieval_rank": 2,
                "binary_safe_link_target": 0,
            },
        ]
    )
    summary = _evaluate_classic_manual_holdout(
        manual_holdout,
        probabilities=np.array([0.1, 0.9, 0.1, 0.05], dtype=np.float32),
        score_threshold=0.2,
        margin_threshold=0.2,
    )
    assert summary["overall"]["balanced_accuracy"] == 1.0
    assert summary["by_bucket"]["rescue"]["positive_recall"] == 1.0
    assert summary["by_bucket"]["easy"]["negative_recall"] == 1.0


def test_augmented_feature_matrix_respects_ablation_columns() -> None:
    """Augmented feature matrices should honor ablated promoted feature lists."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "title_overlap": 0.25,
                "cluster_size": 17,
                "model_score": 0.91,
                "gap_to_top1": 0.01,
                "candidate_rank": 2,
            }
        ]
    )
    features = _augmented_feature_matrix(
        df,
        feature_columns=(
            "title_overlap",
            "cluster_size",
            "query_first_prefix_match_any_length",
        ),
    )
    assert list(features.columns) == [
        "title_overlap",
        "cluster_size",
        "query_first_prefix_match_any_length",
    ]


def test_classic_feature_matrix_supports_augmented_union_features() -> None:
    """Classic feature matrices should derive declared runtime features only."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "title_overlap": 0.25,
                "cluster_size": 17,
                "count_normalized_confidence": 0.6,
            }
        ]
    )
    features = _classic_feature_matrix(
        df,
        (
            "title_overlap",
            "cluster_size",
            "cluster_size_log",
        ),
    )
    assert list(features.columns) == [
        "title_overlap",
        "cluster_size",
        "cluster_size_log",
    ]
    assert features.iloc[0]["title_overlap"] == 0.25
    assert features.iloc[0]["cluster_size"] == 17.0
    assert features.iloc[0]["cluster_size_log"] == pytest.approx(np.log1p(17))


def test_classic_feature_matrix_preserves_present_derivable_features() -> None:
    """Stored feature columns should be used as-is even when they are derivable."""

    df = pd.DataFrame([{"cluster_size": 17, "cluster_size_log": 99.0}])

    features = _classic_feature_matrix(df, ("cluster_size_log",))

    assert features.iloc[0]["cluster_size_log"] == 99.0


def test_classic_feature_matrix_rejects_missing_required_features() -> None:
    """Absent non-runtime features should fail instead of becoming zero-valued signals."""

    df = pd.DataFrame([{"title_overlap": 0.25}])

    with pytest.raises(ValueError, match="missing required feature inputs"):
        _classic_feature_matrix(df, ("title_overlap", "cluster_size"))


def test_classic_feature_matrix_preserves_missing_feature_cells() -> None:
    """Present active features with missing values should remain NaN for LightGBM."""

    df = pd.DataFrame([{"title_overlap": None, "cluster_size": 17}])

    features = _classic_feature_matrix(df, ("title_overlap", "cluster_size"))

    assert np.isnan(features.iloc[0]["title_overlap"])
    assert features.iloc[0]["cluster_size"] == 17.0


def test_classic_feature_matrix_rejects_non_numeric_feature_cells() -> None:
    """Present active features with malformed values should still fail."""

    df = pd.DataFrame([{"title_overlap": "not-a-number", "cluster_size": 17}])

    with pytest.raises(ValueError, match="non-numeric feature values"):
        _classic_feature_matrix(df, ("title_overlap", "cluster_size"))


def test_classic_feature_matrix_rejects_infinite_feature_cells() -> None:
    """LightGBM feature matrices should not contain infinities."""

    df = pd.DataFrame([{"title_overlap": np.inf, "cluster_size": 17}])

    with pytest.raises(ValueError, match="infinite feature values"):
        _classic_feature_matrix(df, ("title_overlap", "cluster_size"))


def test_load_bundle_requires_explicit_root(tmp_path: Path) -> None:
    """Bundle loading should not silently default to historical feature tables."""

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    (bundle_root / "bundle.json").write_text(
        '{"bundle_name":"demo","assets":{},"models":{"classic":{}},"expected_metrics":{}}',
        encoding="utf-8",
    )

    bundle = load_bundle(bundle_root)

    assert bundle.root == bundle_root.resolve()
    assert bundle.bundle_name == "demo"
    assert bundle.models == {"classic": {}}
    assert bundle.expected_metrics == {}


def test_stratified_scoring_recomputes_stale_manual_target_from_active_labels(tmp_path: Path) -> None:
    """Classic stratified scoring should refresh stale split targets from active labels."""

    bundle_root = tmp_path / "bundle"

    def write_rows(relative_path: str, rows: list[dict[str, object]]) -> None:
        path = bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    common_row = {
        "dataset": "demo",
        "query_view": "full",
        "candidate_component_key": "c1",
        "retrieval_rank": 1,
        "label": 0,
    }
    write_rows("calibration/gate_rows.csv.gz", [{"query_group_id": "cal:q1", **common_row}])
    write_rows("test/s2and_eval_rows.csv.gz", [{"query_group_id": "s2:q1", **common_row}])
    write_rows("test/hwang_eval_rows.csv.gz", [{"query_group_id": "h:q1", **common_row}])
    write_rows(
        "test/s2and_rescue_reviewed_eval_rows.csv.gz",
        [{"query_group_id": "rescue:q1", **common_row}],
    )
    split_root = bundle_root / "calibration" / "stratified_eval_test_split"
    split_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "query_group_id": "rescue:q1",
                "source_key": "s2and_rescue_reviewed_eval",
                "split": "test",
                "manual_safe_target": 1,
                "stratum_key": "s2and_block|has_pos=1|positive_first|multi_letter_first|multi_cand=0",
            }
        ]
    ).to_csv(split_root / "combined_query_split_assignments.csv", index=False)
    spec = {
        "classic_gate_source_path": "calibration/gate_rows.csv.gz",
        "s2and_eval_path": "test/s2and_eval_rows.csv.gz",
        "hwang_eval_path": "test/hwang_eval_rows.csv.gz",
        "extra_eval_paths": {
            "s2and_rescue_reviewed": "test/s2and_rescue_reviewed_eval_rows.csv.gz",
        },
    }
    bundle = OfficialBundle(
        root=bundle_root,
        bundle_name="demo",
        assets={},
        models={"classic": spec},
        expected_metrics={},
    )

    class AlwaysLinkModel:
        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.repeat([[0.0, 1.0]], repeats=len(features), axis=0)

    choices, _assignments = _score_classic_stratified_eval_test_choices(
        bundle,
        spec,
        {"assignments_path": "calibration/stratified_eval_test_split/combined_query_split_assignments.csv"},
        AlwaysLinkModel(),  # type: ignore[arg-type]
        (),
    )

    row = choices.set_index("query_case_id").loc["rescue:q1"]
    assert row["query_safe_target"] == 0
    assert row["retrieved_window_safe_target"] == 0
    assert row["query_safe_target_source"] == "retrieved_window"
    assert row["manual_safe_target"] == 0
    assert bool(row["manual_safe_target_matches_active_label"])


def test_bundle_path_resolution_rejects_absolute_paths_outside_bundle(tmp_path: Path) -> None:
    """Runtime bundle path resolution should not allow stale external assets."""

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    inside_path = bundle_root / "rows.csv"
    inside_path.write_text("query_group_id,label\nq1,0\n", encoding="utf-8")
    outside_path = tmp_path / "old_bundle_rows.csv"
    outside_path.write_text("query_group_id,label\nq1,1\n", encoding="utf-8")
    bundle = OfficialBundle(root=bundle_root, bundle_name="demo", assets={}, models={}, expected_metrics={})

    assert _resolve_path(bundle, inside_path) == inside_path.resolve()
    with pytest.raises(ValueError, match="escapes bundle root"):
        _resolve_path(bundle, outside_path)


def test_resolve_classic_monotone_constraints_requires_explicit_opt_in() -> None:
    """Classic monotone constraints should only activate when the bundle specifies them."""

    feature_columns = ("title_overlap", "cluster_size", "min_distance")
    assert _resolve_classic_monotone_constraints({}, feature_columns) is None
    assert _resolve_classic_monotone_constraints(
        {"monotone_constraints": [1, 0, 0]},
        feature_columns,
    ) == [1, 0, 0]


def test_extra_eval_paths_support_dynamic_dataset_mapping() -> None:
    """Classic bundle specs should support arbitrary extra eval dataset mappings."""

    spec = {
        "s_park_eval_path": "test/s_park_eval_rows.csv.gz",
        "extra_eval_paths": {
            "j_smith": "test/j_smith_eval_rows.csv.gz",
            "a_silva": "test/a_silva_eval_rows.csv.gz",
        },
    }
    assert _iter_extra_eval_paths(spec) == (
        ("s_park", "test/s_park_eval_rows.csv.gz"),
        ("j_smith", "test/j_smith_eval_rows.csv.gz"),
        ("a_silva", "test/a_silva_eval_rows.csv.gz"),
    )
    assert _summary_key_for_eval_dataset("a_silva") == "overall_a_silva_eval"


def test_format_classic_selected_gate_tables_includes_requested_breakdowns() -> None:
    """Official table rendering should expose dataset and requested factor metrics."""

    metric = {
        "n_queries": 2,
        "n_positive_queries": 1,
        "n_negative_queries": 1,
        "balanced_accuracy": 0.75,
        "error_rate": 0.25,
        "false_abstain": 1,
        "false_link": 0,
        "wrong_candidate_link": 0,
    }
    all_negative_metric = {
        "n_queries": 5,
        "n_positive_queries": 0,
        "n_negative_queries": 5,
        "balanced_accuracy": 0.5,
        "error_rate": 0.0,
        "false_abstain": 0,
        "false_link": 0,
        "wrong_candidate_link": 0,
    }
    summary = {
        "stratified_eval_test_split": {
            "test_breakdowns": {
                "source_key": {"all_negative_eval": all_negative_metric, "demo_eval": metric},
                "has_positive_candidate": {"True": metric},
                "positive_rank_bucket": {"positive_first": metric},
                "first_name_bucket": {"multi_letter_first": metric},
                "multiple_candidates": {"True": metric},
            }
        }
    }

    tables = format_classic_selected_gate_tables(summary)

    assert "## By Dataset Slice, Selected Gate" in tables
    assert (
        "| slice | queries | positive queries | negative queries | BA | error rate | "
        "false abstain | false link | wrong link |"
    ) in tables
    assert "| demo_eval | 2 | 1 | 1 | 0.7500 | 0.2500 | 1 | 0 | 0 |" in tables
    assert "| all_negative_eval | 5 | 0 | 5 | n/a | 0.0000 | 0 | 0 | 0 |" in tables
    assert "BA is n/a for single-class slices." in tables
    assert "| has_positive_candidate | True | 2 | 1 | 1 | 0.2500 | 1 | 0 | 0 |" in tables


def test_summarize_training_gate_buckets_counts_queries_and_rows() -> None:
    """Training bucket counts should reflect the post-filter query windows."""

    train_df = pd.DataFrame(
        [
            {
                "query_group_id": "multi_multi",
                "query_first_token": "anna",
                "query_view": "full",
                "candidate_component_key": "c1",
            },
            {
                "query_group_id": "multi_multi",
                "query_first_token": "anna",
                "query_view": "full",
                "candidate_component_key": "c2",
            },
            {
                "query_group_id": "single_single",
                "query_first_token": "a",
                "query_view": "initial_only",
                "candidate_component_key": "c3",
            },
        ]
    )

    summary = _summarize_training_gate_buckets(train_df)

    assert summary["query_counts"]["multi_candidate|multi_letter_first"] == 1
    assert summary["row_counts"]["multi_candidate|multi_letter_first"] == 2
    assert summary["query_counts"]["single_candidate|single_letter_first"] == 1
    assert summary["row_counts"]["single_candidate|single_letter_first"] == 1


def test_stratified_summary_tracks_gate_bucket_split_counts_and_errors() -> None:
    """The promoted split summary should expose joint calibration-bucket metrics."""

    predictions = pd.DataFrame(
        [
            {
                "query_case_id": "fit_pos",
                "split": "calibration_fit",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "predicted_action": "link_candidate",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "correct": 1,
            },
            {
                "query_case_id": "check_neg",
                "split": "calibration_check",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "single_letter_first",
                "predicted_action": "abstain",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "correct": 1,
            },
            {
                "query_case_id": "test_false_abstain",
                "split": "test",
                "candidate_kind": "single_candidate",
                "first_name_bucket": "multi_letter_first",
                "predicted_action": "abstain",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "correct": 0,
            },
            {
                "query_case_id": "test_wrong_link",
                "split": "test",
                "candidate_kind": "single_candidate",
                "first_name_bucket": "single_letter_first",
                "predicted_action": "link_candidate",
                "query_safe_target": 1,
                "chosen_candidate_target": 0,
                "correct": 0,
            },
        ]
    )
    assignments = pd.DataFrame(
        [
            {"query_group_id": "fit_pos", "split": "calibration_fit"},
            {"query_group_id": "check_neg", "split": "calibration_check"},
            {"query_group_id": "test_false_abstain", "split": "test"},
            {"query_group_id": "test_wrong_link", "split": "test"},
        ]
    )

    summary = _summarize_classic_stratified_predictions(
        predictions,
        assignments,
        {"split_order": ("calibration_fit", "calibration_check", "test")},
    )

    assert summary["gate_bucket_split_counts"]["multi_candidate|multi_letter_first"]["calibration_fit"] == 1
    assert summary["gate_bucket_split_counts"]["multi_candidate|single_letter_first"]["calibration_check"] == 1
    test_breakdowns = summary["test_breakdowns"]["gate_bucket"]
    assert test_breakdowns["single_candidate|multi_letter_first"]["false_abstain"] == 1
    assert test_breakdowns["single_candidate|single_letter_first"]["wrong_candidate_link"] == 1


def test_format_classic_selected_gate_tables_includes_calibration_bucket_table() -> None:
    """Selected-gate markdown should include the 4 calibration buckets."""

    bucket_metric = {
        "n_queries": 4,
        "n_positive_queries": 3,
        "n_negative_queries": 1,
        "balanced_accuracy": 0.5,
        "error_rate": 0.5,
        "errors": 2,
        "false_abstain": 1,
        "false_link": 0,
        "wrong_candidate_link": 1,
    }
    summary = {
        "training_summary": {
            "gate_bucket_query_counts": {"multi_candidate|multi_letter_first": 7},
            "gate_bucket_row_counts": {"multi_candidate|multi_letter_first": 25},
        },
        "abstain_rule": {
            "bucketed_score_thresholds": {
                "multi_candidate|multi_letter_first": 0.82,
                "multi_candidate|single_letter_first": 0.04,
                "single_candidate|multi_letter_first": 0.01,
                "single_candidate|single_letter_first": 0.50,
            },
            "bucketed_margin_thresholds": {
                "multi_candidate|multi_letter_first": 0.16,
                "multi_candidate|single_letter_first": 0.41,
            },
            "promoted_stratified_gate": {
                "calibration_splits": ["calibration_fit", "calibration_check"],
                "test_split": "test",
            },
        },
        "stratified_eval_test_split": {
            "gate_bucket_split_counts": {
                "multi_candidate|multi_letter_first": {
                    "calibration_fit": 3,
                    "calibration_check": 2,
                    "test": 4,
                }
            },
            "test_breakdowns": {
                "source_key": {},
                "gate_bucket": {"multi_candidate|multi_letter_first": bucket_metric},
            },
        },
    }

    tables = format_classic_selected_gate_tables(summary)

    assert "## By Calibration Bucket, Selected Gate" in tables
    assert (
        "| bucket | score threshold | margin threshold | train queries | train rows | calibration fit | "
        "calibration check | calibration total | test queries | test positive queries | test negative queries | "
        "test errors | test error rate | false abstain | false link | wrong link |"
    ) in tables
    assert (
        "| multi_candidate\\|multi_letter_first | 0.8200 | 0.1600 | 7 | 25 | 3 | 2 | 5 | "
        "4 | 3 | 1 | 2 | 0.5000 | 1 | 0 | 1 |"
    ) in tables


def test_score_eval_candidate_rows_defaults_to_w5_and_w25_only() -> None:
    """Official classic eval scoring should only materialize the retained window limits."""

    df = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 1,
            },
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c2",
                "retrieval_rank": 10,
                "label": 0,
            },
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c3",
                "retrieval_rank": 30,
                "label": 0,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c4",
                "retrieval_rank": 3,
                "label": 0,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c5",
                "retrieval_rank": 20,
                "label": 1,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c6",
                "retrieval_rank": 40,
                "label": 0,
            },
        ]
    )

    scored = _score_eval_candidate_rows(
        df,
        probabilities=np.array([0.9, 0.4, 0.1, 0.3, 0.8, 0.2], dtype=np.float32),
        include_margin=True,
    )

    assert sorted(scored["retrieval_rank_limit"].astype(int).unique().tolist()) == [5, 25]
