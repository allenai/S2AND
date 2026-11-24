from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking_training import classic as classic_training
from s2and.incremental_linking_training.classic import (
    _fit_promoted_logistic_gate,
    _promoted_stratified_gate_spec,
    _score_query_choices,
    build_whole_author_holdout_rows,
    report_logistic_gate_error_weights,
)
from scripts.eps_sweep import apply_reviewed_eps as apply_eps
from scripts.eps_sweep.orcid_b3 import REFERENCE_POLICY_ALL_IDENTITIES
from scripts.production.data.block_roles import load_block_roles
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
            "tune_false_abstain_weights": [0.20, 0.25, 0.30],
            "test_split": "test",
        }
    }
    parsed = _promoted_stratified_gate_spec(spec)
    assert parsed is not None
    assert parsed["fit_splits"] == ["calibration_fit"]
    assert parsed["validation_splits"] == ["calibration_check"]
    assert parsed["tune_false_abstain_weights"] == [0.20, 0.25, 0.30]

    # validation_splits without fit_splits must fail early in the parser!
    with pytest.raises(ValueError, match="requires explicit fit_splits"):
        _promoted_stratified_gate_spec(
            {
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit", "calibration_check"],
                    "validation_splits": ["calibration_check"],
                    "test_split": "test",
                }
            }
        )

    with pytest.raises(ValueError, match="must be disjoint"):
        _promoted_stratified_gate_spec(
            {
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit", "calibration_check"],
                    "fit_splits": ["calibration_fit"],
                    "validation_splits": ["calibration_fit"],
                    "test_split": "test",
                }
            }
        )


def test_fit_promoted_logistic_gate_separates_fit_and_reports_weight_grid() -> None:
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

    # Default grid includes the production 0.25 baseline and rejects empty choices.
    val_qids = set(choices.loc[val_mask, "query_case_id"].astype(str))
    row_val_mask = rows["query_group_id"].astype(str).isin(val_qids).to_numpy()
    config = {
        **res1["gate_config"],
        "weights": [[0.0] * 3] * 4,
        "bias": np.log([0.10, 0.025, 0.875]).tolist(),
    }
    report = report_logistic_gate_error_weights(
        config,
        rows.loc[row_val_mask].reset_index(drop=True),
        choices.loc[val_mask].reset_index(drop=True),
        probs[row_val_mask],
    )
    assert report["configured_false_abstain_weight"] == 0.25
    assert "gate_config" not in report
    assert res1["gate_config"]["error_weights"]["false_abstain"] == 0.25
    baseline = next(trial for trial in report["sweep"] if trial["false_abstain_weight"] == 0.25)
    links = baseline["link_count"]
    assert baseline["gold_precision"] == pytest.approx(baseline["gold_correct_links"] / links)
    assert baseline["gold_recall"] == pytest.approx(baseline["gold_correct_links"] / report["gold_linkable_queries"])
    assert any(trial["added_links"] for trial in report["sweep"])
    assert report["sweep"][0]["removed_links"]
    for trial in report["sweep"]:
        links += len(trial["added_links"]) - len(trial["removed_links"])
        assert trial["link_count"] == links
    with pytest.raises(ValueError, match="non-empty"):
        report_logistic_gate_error_weights(res1["gate_config"], rows.iloc[:0], choices.iloc[:0], np.array([]))


@pytest.mark.parametrize("configured_weight", [None, 0.34])
def test_fit_classic_uses_frozen_roles_and_keeps_the_configured_gate_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_weight: float | None
) -> None:
    rows, _, probabilities = _synthetic_gate_data()
    rows = rows.assign(feature=probabilities, candidate_gold_scored=True)
    rows.loc[rows["query_group_id"].isin({"q_000", "q_049"}), "candidate_gold_scored"] = False
    gate_fit = rows.loc[rows["split"].eq("calibration_fit")]
    gate_thresh = rows.loc[rows["split"].eq("calibration_check")]
    linker_train = gate_fit.assign(query_group_id=lambda frame: "train_" + frame["query_group_id"])
    fitted_rows = []

    class Ranker:
        def fit(self, features: np.ndarray, labels: np.ndarray, *, sample_weight: np.ndarray) -> None:
            fitted_rows.append(len(features))

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            positive = np.asarray(features)[:, 0]
            return np.column_stack((1 - positive, positive))

    monkeypatch.setattr(classic_training, "_build_classic_classifier", lambda *_args, **_kwargs: Ranker())
    gate = {"false_abstain_weight": configured_weight} if configured_weight is not None else {}
    bundle = classic_training.OfficialBundle(
        root=tmp_path,
        bundle_name="fixture",
        assets={},
        expected_metrics={},
        models={
            "classic": {
                "feature_columns": ["feature"],
                "best_params": {},
                "promoted_stratified_gate": gate,
                "stratified_eval_test_split": {},
            }
        },
    )
    result = classic_training.fit_classic(
        bundle,
        n_jobs=1,
        holdout_identities=classic_training.ClassicHoldoutIdentities(frozenset(), frozenset(), ()),
        frozen_task_rows={"linker_train": linker_train, "gate_fit": gate_fit, "gate_thresh": gate_thresh},
    )
    assert fitted_rows == [78]
    assert result.gate_config["training_summary"]["fit_queries"] == 39
    assert result.gate_config["training_summary"]["validation_queries"] == 20
    assert result.gate_config["error_weights"]["false_abstain"] == (configured_weight or 0.25)
    report = result.abstain_rule_summary["promoted_logistic_gate"]["false_abstain_weight_report"]
    assert report["n_queries"] == 20
    assert report["configured_false_abstain_weight"] == result.gate_config["error_weights"]["false_abstain"]


def test_build_whole_author_holdout_rows_guards_membership_features_and_labels() -> None:
    rows = pd.DataFrame(
        [
            {"query_group_id": "q1", "candidate_component_key": "c1", "retrieval_rank": "1", "label": 1},
            {"query_group_id": "q1", "candidate_component_key": "c2", "retrieval_rank": "2", "label": 0},
            {"query_group_id": "q1", "candidate_component_key": "c3", "retrieval_rank": "10", "label": 1},
            {"query_group_id": "q1", "candidate_component_key": "c4", "retrieval_rank": "3", "label": 0},
            {"query_group_id": "q2", "candidate_component_key": "c5", "retrieval_rank": "1", "label": 1},
            {"query_group_id": "q3", "candidate_component_key": "c6", "retrieval_rank": "1", "label": 1},
        ]
    )
    membership = {"c1": {"A1"}, "c2": {"A9"}, "c3": {"A1", "A8"}, "c4": {"A7"}, "c5": {"A2"}, "c6": {"A3"}}
    out, summary = build_whole_author_holdout_rows(
        rows,
        query_to_author={"q1": "A1", "q2": "A2", "q3": "A3"},
        component_to_authors=membership,
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

    # Rejects missing component_to_authors key (never silently manufactures false negatives!).
    with pytest.raises(KeyError, match="missing from component_to_authors"):
        build_whole_author_holdout_rows(
            rows,
            query_to_author={"q1": "A1"},
            component_to_authors={"c2": {"A9"}},
            holdout_query_ids={"q1"},
        )
    # Rejects incomplete membership where a label==1 row survives author removal.
    with pytest.raises(ValueError, match="still has a label==1"):
        build_whole_author_holdout_rows(
            rows,
            query_to_author={"q1": "A1"},
            component_to_authors={"c1": set(), "c2": set(), "c3": set(), "c4": set()},
            holdout_query_ids={"q1"},
        )
    # Rejects materialized feature frames.
    with pytest.raises(ValueError, match="featureless pre-materialization"):
        build_whole_author_holdout_rows(
            rows.assign(pw_mean_prob=0.5),
            query_to_author={"q1": "A1"},
            component_to_authors=membership,
            holdout_query_ids={"q1"},
        )
    # Rejects missing requested holdout query IDs.
    with pytest.raises(KeyError, match="absent from candidate_rows"):
        build_whole_author_holdout_rows(
            rows,
            query_to_author={"q_missing": "A9"},
            component_to_authors=membership,
            holdout_query_ids={"q_missing"},
        )


def test_smoke_mode_cap_per_source_and_block_disjoint_eps_review_guard(tmp_path: Path) -> None:
    assignments_path = tmp_path / "assignments.csv"
    # 3 splits x 10 rows = 30 rows for one source_key, with string "True"/"False" booleans!
    pd.DataFrame(
        {
            "query_group_id": [f"q{i:02d}" for i in range(30)],
            "source_key": ["s2and_eval"] * 30,
            "split": ["calibration_fit"] * 10 + ["calibration_check"] * 10 + ["test"] * 10,
            "has_positive_candidate": ["True" if i % 2 == 0 else "False" for i in range(30)],
        }
    ).to_csv(assignments_path, index=False)
    by_source = promoted_train._subset_smoke_assignments_file(assignments_path, max_query_groups=6)  # noqa: SLF001
    # Total per source_key is capped at 6 (2 per split), NOT 18!
    assert len(by_source["s2and_eval"]) == 6
    reloaded = pd.read_csv(assignments_path)
    for _split, grp in reloaded.groupby("split"):
        assert len(grp) == 2
        flags = promoted_train._parse_bool_flag(grp["has_positive_candidate"])  # noqa: SLF001
        assert int(flags.sum()) == 1
        assert int((~flags).sum()) == 1

    # Verify apply_reviewed_eps block-role manifest verifier accepts clean eps_cal and rejects bad manifests.
    role_manifest = tmp_path / "assignments.json"
    role_manifest.write_text(
        json.dumps(
            {
                "roles": {"eps_cal": ["a_smith", "b_lee"], "final_test": ["c_park"]},
                "verification": {"cross_role_orcid_overlap": 0},
            }
        ),
        encoding="utf-8",
    )
    eps_cal = apply_eps._eps_cal_blocks(load_block_roles(role_manifest))  # noqa: SLF001
    assert eps_cal == frozenset({"a smith", "b lee"})

    # Rejects block-disjoint policy when no role manifest is provided.
    with pytest.raises(ValueError, match="requires --block-role-assignments"):
        apply_eps._require_test_separated_reference(  # noqa: SLF001
            tmp_path,
            "a_smith",
            {"reference": {"reference_policy": REFERENCE_POLICY_ALL_IDENTITIES}, "arrow_root": str(tmp_path)},
            lambda *_a: frozenset(),
            None,  # type: ignore[arg-type]
            verified_eps_cal_blocks=None,
        )
