from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.greedy_single_letter_feature_elimination as greedy_elimination


def test_read_fixed_best_params_uses_h_wang_train_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "h_wang": {
                    "train_summary": {
                        "best_params": {
                            "learning_rate": 0.1,
                            "n_estimators": 123,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert greedy_elimination._read_fixed_best_params(summary_path) == {  # noqa: SLF001
        "learning_rate": 0.1,
        "n_estimators": 123,
    }


def test_choose_best_candidate_result_breaks_ties_by_feature_order() -> None:
    results = [
        greedy_elimination.CandidateResult(
            position=2,
            removed_feature="feature_c",
            remaining_features=("a", "b"),
            h_wang_accuracy=0.7,
            fit_seconds=1.0,
            predict_seconds=0.1,
        ),
        greedy_elimination.CandidateResult(
            position=0,
            removed_feature="feature_a",
            remaining_features=("b", "c"),
            h_wang_accuracy=0.7,
            fit_seconds=1.0,
            predict_seconds=0.1,
        ),
        greedy_elimination.CandidateResult(
            position=1,
            removed_feature="feature_b",
            remaining_features=("a", "c"),
            h_wang_accuracy=0.69,
            fit_seconds=1.0,
            predict_seconds=0.1,
        ),
    ]
    accepted = greedy_elimination._choose_best_candidate_result(results)  # noqa: SLF001
    assert accepted.removed_feature == "feature_a"


def test_parse_args_defaults_to_corrected_round_robin_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "greedy_single_letter_feature_elimination.py",
            "--output-dir",
            "scratch/out",
        ],
    )
    parsed = greedy_elimination.parse_args()
    assert parsed.dataset_root == Path("scratch/s2and_round_robin_matched_20260325_rows")
    assert parsed.fixed_summary_json == Path(
        "scratch/s2and_ranker_all6_k50_v8_roundrobinmatched_full_20260325/summary.json"
    )
    assert parsed.feature_preset == "generalized_v8"
    assert parsed.window_size == 50
    assert parsed.max_workers == 20
    assert parsed.lgbm_n_jobs == 1
