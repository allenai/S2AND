"""Regression tests for promoted train/calibrate/eval helper functions."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking.logistic_gate import (
    LOGISTIC_GATE_CLASSES,
    LOGISTIC_GATE_ERROR_WEIGHTS,
    NumpyLogisticGate,
    logistic_gate_config,
)
from s2and.incremental_linking_training.classic import (
    OfficialBundle,
    _apply_classic_train_holdout_filter,
    _apply_classic_train_row_cap,
    _build_classic_classifier,
    _classic_feature_matrix,
    _drop_unlabeled_singleton_orcid_rows,
    _evaluate_logistic_manual_holdout,
    _evaluate_logistic_scored_windows,
    _fit_multiclass_logistic_gate,
    _fold_standard_scaler_into_logistic,
    _iter_extra_eval_paths,
    _promoted_stratified_gate_spec,
    _resolve_classic_monotone_constraints,
    _resolve_path,
    _score_classic_stratified_eval_test,
    _score_eval_candidate_rows,
    _score_query_choices,
    _summarize_classic_stratified_predictions,
    _summarize_training_gate_buckets,
    _summary_key_for_eval_dataset,
    format_classic_selected_gate_tables,
    load_bundle,
)
from s2and.incremental_linking_training.classic import (
    _read_csv as _read_official_table,
)


def test_score_query_choices_preserves_float64_probability_tie_breaking() -> None:
    rows = pd.DataFrame(
        [
            {
                "query_group_id": "q",
                "dataset": "unit",
                "query_view": "full",
                "candidate_component_key": "slightly_higher",
                "retrieval_rank": 2,
                "label": 1,
            },
            {
                "query_group_id": "q",
                "dataset": "unit",
                "query_view": "full",
                "candidate_component_key": "float32_tie_lower_rank",
                "retrieval_rank": 1,
                "label": 0,
            },
        ]
    )

    choices = _score_query_choices(
        rows,
        np.asarray([0.500000020, 0.500000019], dtype=np.float64),
        query_id_column="query_group_id",
        include_margin=True,
    )

    assert choices.loc[0, "chosen_candidate_component_key"] == "slightly_higher"
    assert choices.loc[0, "chosen_candidate_target"] == 1


@pytest.mark.parametrize(
    "labels",
    (
        np.asarray([0, 2] * 30, dtype=np.int8),
        np.asarray([1, 2] * 30, dtype=np.int8),
    ),
)
def test_fold_standard_scaler_binary_logistic_matches_sklearn(labels: np.ndarray) -> None:
    rng = np.random.default_rng(20260518 + int(labels[0]))
    matrix = rng.normal(size=(len(labels), 5)).astype(np.float64)
    matrix[:, 0] += np.where(labels == int(labels.max()), 0.7, -0.7)
    matrix[0, 1] = np.nan

    model, scaler, medians = _fit_multiclass_logistic_gate(matrix, labels, c_value=0.2)
    filled = matrix.copy()
    row_indices, col_indices = np.nonzero(~np.isfinite(filled))
    filled[row_indices, col_indices] = medians[col_indices]
    weights, bias = _fold_standard_scaler_into_logistic(scaler, model)
    gate = NumpyLogisticGate(
        feature_names=tuple(f"f{index}" for index in range(matrix.shape[1])),
        weights=weights,
        bias=bias,
        missing_values=medians,
        classes=LOGISTIC_GATE_CLASSES,
        error_weights=LOGISTIC_GATE_ERROR_WEIGHTS,
    )

    expected = np.zeros((len(labels), len(LOGISTIC_GATE_CLASSES)), dtype=np.float64)
    sklearn_probabilities = model.predict_proba(scaler.transform(filled))
    for class_position, class_value in enumerate(model.classes_):
        expected[:, int(class_value)] = sklearn_probabilities[:, class_position]

    np.testing.assert_allclose(gate.predict_proba(matrix), expected, rtol=1e-12, atol=1e-12)


def test_official_table_loader_reads_parquet_with_usecols(tmp_path: Path) -> None:
    """Official stack loaders should accept parquet paths from the three-phase bundle."""

    path = tmp_path / "rows.parquet"
    pd.DataFrame([{"query_group_id": "001", "label": 1, "min_distance": 0.2}]).to_parquet(path, index=False)

    loaded = _read_official_table(path, usecols=["query_group_id", "label"])

    assert loaded.to_dict(orient="records") == [{"query_group_id": "001", "label": 1}]


def test_drop_unlabeled_singleton_orcid_rows_reports_removed_queries() -> None:
    rows = pd.DataFrame(
        [
            {
                "query_group_id": "drop_me",
                "label": 0,
                "supervision_type": "unlabeled_singleton_orcid",
            },
            {
                "query_group_id": "keep_me",
                "label": 1,
                "supervision_type": "positive_repeat_orcid",
            },
            {
                "query_group_id": "mixed",
                "label": 0,
                "supervision_type": "unlabeled_singleton_orcid",
            },
            {
                "query_group_id": "mixed",
                "label": 1,
                "supervision_type": "positive_repeat_orcid",
            },
        ]
    )

    cleaned, summary = _drop_unlabeled_singleton_orcid_rows(rows, context="unit")

    assert cleaned["query_group_id"].tolist() == ["keep_me", "mixed"]
    assert summary["rows_removed"] == 2
    assert summary["negative_rows_removed"] == 2
    assert summary["queries_removed"] == 1


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

    scored = _score_classic_stratified_eval_test(
        bundle,
        spec,
        {"assignments_path": "calibration/stratified_eval_test_split/combined_query_split_assignments.csv"},
        AlwaysLinkModel(),  # type: ignore[arg-type]
        (),
    )

    choice = scored.choices.set_index("query_case_id").loc["h:q1"]
    assignment = scored.assignments.set_index("query_group_id").loc["h:q1"]
    assert choice["chosen_candidate_target"] == 1
    assert choice["query_safe_target"] == 1
    assert choice["positive_rank_bucket"] == "positive_first"
    assert assignment["positive_rank_bucket"] == "positive_first"


def test_promoted_stratified_gate_spec_rejects_removed_threshold_calibration() -> None:
    """Promoted gate config should not silently preserve old threshold calibration settings."""

    with pytest.raises(ValueError, match="no longer supports threshold calibration keys"):
        _promoted_stratified_gate_spec(
            {
                "stratified_eval_test_split": {"test_split": "test"},
                "promoted_stratified_gate": {
                    "mode": "promoted_logistic_topk_multiclass_l2",
                    "calibration_splits": ["calibration_fit"],
                    "test_split": "test",
                    "fixed_grid_step": 0.1,
                },
            }
        )

    with pytest.raises(ValueError, match="mode must be"):
        _promoted_stratified_gate_spec(
            {
                "promoted_stratified_gate": {
                    "mode": "full_calibration_fixed_grid_4score_2margin",
                    "calibration_splits": ["calibration_fit"],
                }
            }
        )


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


def test_apply_classic_train_row_cap_matches_numeric_query_ids() -> None:
    train_df = pd.DataFrame(
        [
            {"query_group_id": 1, "retrieval_rank": 1, "label": 0},
            {"query_group_id": 1, "retrieval_rank": 6, "label": 1},
            {"query_group_id": 2, "retrieval_rank": 1, "label": 0},
            {"query_group_id": 2, "retrieval_rank": 3, "label": 1},
        ]
    )

    selected, summary = _apply_classic_train_row_cap(
        train_df,
        rule_name="max_of_min_limit_and_first_positive_rank",
        min_train_limit=5,
    )

    assert selected.groupby("query_group_id", sort=False)["retrieval_rank"].max().to_dict() == {1: 6, 2: 3}
    assert summary is not None
    assert summary["queries_with_row_cap_above_min"] == 1
    assert summary["lost_positive_queries"] == 0


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


def test_evaluate_logistic_manual_holdout_scores_fresh_candidates() -> None:
    """Manual holdout evaluation should score fresh candidates through the logistic gate."""

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
    gate_config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[-10.0, 0.0, 10.0]], dtype=np.float64),
        bias=np.asarray([5.0, 0.0, -5.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="unit_test_manual_holdout",
        error_weights=LOGISTIC_GATE_ERROR_WEIGHTS,
    )

    summary = _evaluate_logistic_manual_holdout(
        manual_holdout,
        probabilities=np.array([0.1, 0.9, 0.1, 0.05], dtype=np.float64),
        gate_config=gate_config,
    )

    assert summary["overall"]["balanced_accuracy"] == 1.0
    assert summary["by_bucket"]["rescue"]["positive_recall"] == 1.0
    assert summary["by_bucket"]["easy"]["negative_recall"] == 1.0


def test_logistic_eval_summaries_accept_empty_candidate_tables() -> None:
    """Empty eval inputs should produce zero-count summaries, not schema KeyErrors."""

    gate_config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="unit_test_empty_eval",
        error_weights=LOGISTIC_GATE_ERROR_WEIGHTS,
    )
    candidate_rows = pd.DataFrame(
        columns=["query_group_id", "dataset", "query_view", "candidate_component_key", "retrieval_rank", "label"]
    )
    window_summary = _evaluate_logistic_scored_windows(
        candidate_rows,
        probabilities=np.asarray([], dtype=np.float64),
        gate_config=gate_config,
    )

    assert window_summary["5"]["overall"]["n_queries"] == 0
    assert window_summary["25"]["overall"]["n_queries"] == 0

    manual_holdout = pd.DataFrame(
        columns=[
            "query_case_id",
            "dataset",
            "query_view",
            "review_bucket",
            "candidate_component_key",
            "retrieval_rank",
            "binary_safe_link_target",
        ]
    )
    manual_summary = _evaluate_logistic_manual_holdout(
        manual_holdout,
        probabilities=np.asarray([], dtype=np.float64),
        gate_config=gate_config,
    )

    assert manual_summary["overall"]["n_queries"] == 0
    assert manual_summary["by_bucket"] == {}


def test_classic_feature_matrix_preserves_present_values_and_missing_cells() -> None:
    """Stored values stay authoritative and missing cells remain NaN for LightGBM."""

    df = pd.DataFrame([{"title_overlap": None, "cluster_size": 17, "cluster_size_log": 99.0}])
    features = _classic_feature_matrix(df, ("title_overlap", "cluster_size", "cluster_size_log"))

    assert np.isnan(features.iloc[0]["title_overlap"])
    assert features.iloc[0]["cluster_size"] == 17.0
    assert features.iloc[0]["cluster_size_log"] == 99.0


def test_classic_feature_matrix_rejects_invalid_feature_tables() -> None:
    """Missing, non-numeric, and infinite active features fail explicitly."""

    cases = (
        ({"cluster_size": 17}, ("cluster_size", "cluster_size_log"), r"missing_features=\['cluster_size_log'\]"),
        ({"title_overlap": 0.25}, ("title_overlap", "cluster_size"), "missing required feature inputs"),
        ({"title_overlap": "not-a-number", "cluster_size": 17}, ("title_overlap", "cluster_size"), "non-numeric"),
        ({"title_overlap": np.inf, "cluster_size": 17}, ("title_overlap", "cluster_size"), "infinite"),
    )
    for row, features, message in cases:
        with pytest.raises(ValueError, match=message):
            _classic_feature_matrix(pd.DataFrame([row]), features)


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

    scored = _score_classic_stratified_eval_test(
        bundle,
        spec,
        {"assignments_path": "calibration/stratified_eval_test_split/combined_query_split_assignments.csv"},
        AlwaysLinkModel(),  # type: ignore[arg-type]
        (),
    )

    row = scored.choices.set_index("query_case_id").loc["rescue:q1"]
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
    assert scored["second_probability"].dtype == np.dtype(np.float64)
    assert scored["score_margin"].dtype == np.dtype(np.float64)


def test_score_eval_candidate_rows_uses_positions_not_index_labels() -> None:
    df = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "dataset": "s2and",
                "query_view": "full",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 0,
            },
            {
                "query_group_id": "q1",
                "dataset": "s2and",
                "query_view": "full",
                "candidate_component_key": "c2",
                "retrieval_rank": 2,
                "label": 1,
            },
            {
                "query_group_id": "q2",
                "dataset": "s2and",
                "query_view": "full",
                "candidate_component_key": "c3",
                "retrieval_rank": 1,
                "label": 1,
            },
        ],
        index=[10, 20, 30],
    )

    scored = _score_eval_candidate_rows(
        df,
        probabilities=np.array([0.2, 0.9, 0.8], dtype=np.float32),
        include_margin=True,
        limits=(5,),
    )

    chosen = scored.set_index("query_case_id")["chosen_candidate_component_key"].to_dict()
    assert chosen == {"q1": "c2", "q2": "c3"}


def test_build_classic_classifier_uses_configured_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("s2and.thread_config.os.cpu_count", lambda: 8)

    classifier = _build_classic_classifier({}, n_jobs=-1)

    assert classifier.get_params()["n_jobs"] == 8
