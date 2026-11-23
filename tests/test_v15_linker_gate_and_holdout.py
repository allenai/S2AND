from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking_training.classic import (
    _fit_promoted_logistic_gate,
    _promoted_stratified_gate_spec,
    _score_query_choices,
    build_whole_author_holdout_rows,
    select_logistic_gate_error_weights,
)
from scripts.production.model import train_linker_and_finalize as promoted_train


def _synthetic_gate_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(20261123)
    rows: list[dict] = []
    probs: list[float] = []
    for idx in range(60):
        split = "calibration_fit" if idx < 40 else "calibration_check"
        qid = f"q_{idx:03d}"
        is_safe = idx % 3 != 0
        wrong_top = (idx % 6) == 1
        for rank in (1, 2):
            if is_safe:
                label = 1 if (rank == 2 if wrong_top else rank == 1) else 0
                base_p = 0.85 if label == 1 else 0.15
                if wrong_top and rank == 1:
                    base_p = 0.62
                if wrong_top and rank == 2:
                    base_p = 0.55
            else:
                label = 0
                base_p = 0.25 if rank == 1 else 0.18
            prob = float(np.clip(base_p + rng.normal(0.0, 0.05), 0.01, 0.99))
            rows.append(
                {
                    "query_group_id": qid,
                    "dataset": "unit",
                    "query_view": "full" if idx % 2 == 0 else "initial_only",
                    "candidate_component_key": f"blk::c_{idx}_{rank}",
                    "retrieval_rank": rank,
                    "label": label,
                    "split": split,
                    "first_name_bucket": "full" if idx % 2 == 0 else "initial",
                    "pw_mean_prob": prob,
                    "pw_max_prob": min(0.99, prob + 0.03),
                    "pw_min_prob": max(0.01, prob - 0.03),
                    "component_size": 3 if rank == 1 else 2,
                }
            )
            probs.append(prob)
    candidate_rows = pd.DataFrame(rows)
    probabilities = np.asarray(probs, dtype=np.float64)
    choices = _score_query_choices(candidate_rows, probabilities, query_id_column="query_group_id", include_margin=True)
    split_map = (
        candidate_rows[["query_group_id", "split"]]
        .drop_duplicates()
        .rename(columns={"query_group_id": "query_case_id"})
    )
    choices = choices.merge(split_map, on="query_case_id", how="left")
    return candidate_rows, choices, probabilities


def test_promoted_stratified_gate_spec_validates_fit_and_validation_splits() -> None:
    spec = {
        "promoted_stratified_gate": {
            "calibration_splits": ["calibration_fit", "calibration_check"],
            "fit_splits": ["calibration_fit"],
            "validation_splits": ["calibration_check"],
            "test_split": "test",
        }
    }
    parsed = _promoted_stratified_gate_spec(spec)
    assert parsed is not None
    assert parsed["fit_splits"] == ["calibration_fit"]
    assert parsed["validation_splits"] == ["calibration_check"]

    overlap_spec = {
        "promoted_stratified_gate": {
            "calibration_splits": ["calibration_fit", "calibration_check"],
            "fit_splits": ["calibration_fit"],
            "validation_splits": ["calibration_fit"],
            "test_split": "test",
        }
    }
    with pytest.raises(ValueError, match="must be disjoint"):
        _promoted_stratified_gate_spec(overlap_spec)


def test_fit_promoted_logistic_gate_separates_fit_and_validation_and_tunes_w_fa() -> None:
    rows, choices, probs = _synthetic_gate_data()
    res1 = _fit_promoted_logistic_gate(
        rows,
        choices,
        probs,
        calibration_splits=("calibration_fit", "calibration_check"),
        fit_splits=("calibration_fit",),
        validation_splits=("calibration_check",),
        selection_feature_count=4,
    )
    assert res1["validation_metrics"] is not None
    summary = res1["gate_config"]["training_summary"]
    assert summary["fit_splits"] == ["calibration_fit"]
    assert summary["validation_splits"] == ["calibration_check"]
    assert summary["fit_queries"] == 40
    assert summary["validation_queries"] == 20

    # Flipping targets ONLY in validation_splits must not change the fitted gate weights!
    flipped_choices = choices.copy()
    val_mask = flipped_choices["split"].eq("calibration_check")
    flipped_choices.loc[val_mask, "query_safe_target"] = 1 - flipped_choices.loc[val_mask, "query_safe_target"]
    res2 = _fit_promoted_logistic_gate(
        rows,
        flipped_choices,
        probs,
        calibration_splits=("calibration_fit", "calibration_check"),
        fit_splits=("calibration_fit",),
        validation_splits=("calibration_check",),
        selection_feature_count=4,
    )
    assert res1["gate_config"]["weights"] == res2["gate_config"]["weights"]
    assert res1["gate_config"]["bias"] == res2["gate_config"]["bias"]

    # Out-of-sample false_abstain weight sweep on calibration_check.
    val_qids = set(choices.loc[val_mask, "query_case_id"].astype(str))
    row_val_mask = rows["query_group_id"].astype(str).isin(val_qids).to_numpy()
    tuned = select_logistic_gate_error_weights(
        res1["gate_config"],
        rows.loc[row_val_mask].reset_index(drop=True),
        choices.loc[val_mask].reset_index(drop=True),
        probs[row_val_mask],
        false_abstain_weights=(0.5, 1.0, 2.0),
    )
    assert tuned["selected_false_abstain_weight"] in (0.5, 1.0, 2.0)
    assert len(tuned["sweep"]) == 3


def test_build_whole_author_holdout_rows_removes_all_author_fragments_and_reranks() -> None:
    rows = pd.DataFrame(
        [
            # q1 (author A1) will be held out: c1 and c3 both contain A1 -> both removed; c2, c4 remain reranked 1..2.
            {"query_group_id": "q1", "candidate_component_key": "c1", "retrieval_rank": 1, "label": 1},
            {"query_group_id": "q1", "candidate_component_key": "c2", "retrieval_rank": 2, "label": 0},
            {"query_group_id": "q1", "candidate_component_key": "c3", "retrieval_rank": 3, "label": 1},
            {"query_group_id": "q1", "candidate_component_key": "c4", "retrieval_rank": 4, "label": 0},
            # q2 (author A2) will be held out and has ONLY A2's cluster -> dropped as empty.
            {"query_group_id": "q2", "candidate_component_key": "c5", "retrieval_rank": 1, "label": 1},
            # q3 is NOT held out -> untouched.
            {"query_group_id": "q3", "candidate_component_key": "c6", "retrieval_rank": 1, "label": 1},
        ]
    )
    out, summary = build_whole_author_holdout_rows(
        rows,
        query_to_author={"q1": "A1", "q2": "A2", "q3": "A3"},
        component_to_authors={"c1": {"A1"}, "c2": {"A9"}, "c3": {"A1", "A8"}, "c4": {"A7"}, "c5": {"A2"}, "c6": {"A3"}},
        holdout_query_ids={"q1", "q2"},
        retrieval_top_k=5,
    )
    assert summary == {
        "holdout_queries_requested": 2,
        "holdout_queries_retained": 1,
        "holdout_queries_dropped_empty": 1,
        "author_rows_removed": 3,
    }
    q1 = out[out["query_group_id"] == "q1"]
    assert q1["candidate_component_key"].tolist() == ["c2", "c4"]
    assert q1["retrieval_rank"].tolist() == [1, 2]
    assert q1["label"].tolist() == [0, 0]
    assert "q2" not in set(out["query_group_id"])
    q3 = out[out["query_group_id"] == "q3"]
    assert q3["label"].tolist() == [1]


def test_smoke_mode_helpers_and_cli_arg(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "query_group_id": [f"q{i}" for i in range(10) for _ in range(2)],
            "label": [1 if (i % 2 == 0 and j == 0) else 0 for i in range(10) for j in range(2)],
        }
    )
    sampled = promoted_train._sample_query_groups_for_smoke(labels, max_query_groups=4)  # noqa: SLF001
    assert sampled["query_group_id"].nunique() == 4
    # Preserves both positive-bearing and all-negative queries.
    has_pos = sampled.groupby("query_group_id")["label"].max() > 0
    assert int(has_pos.sum()) == 2
    assert int((~has_pos).sum()) == 2

    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame(
        {
            "query_group_id": [f"q{i}" for i in range(8)],
            "source_key": ["s2and_eval"] * 8,
            "split": ["calibration_fit"] * 4 + ["test"] * 4,
            "has_positive_candidate": [True, True, False, False, True, True, False, False],
        }
    ).to_csv(assignments_path, index=False)
    by_source = promoted_train._subset_smoke_assignments_file(assignments_path, max_query_groups=2)  # noqa: SLF001
    reloaded = pd.read_csv(assignments_path)
    assert len(reloaded) == 4
    assert len(by_source["s2and_eval"]) == 4

    parser = promoted_train.build_parser()
    parsed = parser.parse_args(
        [
            "--source-bundle-root",
            "src",
            "--output-dir",
            "out",
            "--pairwise-model-path",
            "pw",
            "--target-json",
            "t.json",
            "--name-counts-index-root",
            "nci",
            "--n-jobs",
            "2",
            "--max-query-groups-per-table",
            "25",
        ]
    )
    assert parsed.max_query_groups_per_table == 25
