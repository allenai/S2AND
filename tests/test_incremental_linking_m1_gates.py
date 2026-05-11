from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts import run_joint_safe_link_promoted_train_calibrate_eval as promoted_train
from scripts.run_joint_safe_link_promoted_train_calibrate_eval import run_classic


def test_incremental_linking_runtime_imports_stay_runtime_safe() -> None:
    runtime_root = Path("s2and/incremental_linking")
    scripts_imports: list[str] = []
    model_imports: list[str] = []
    for path in runtime_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "scripts" or alias.name.startswith("scripts."):
                        scripts_imports.append(str(path))
                    if alias.name == "s2and.model" or alias.name.startswith("s2and.model."):
                        model_imports.append(str(path))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "scripts" or module.startswith("scripts."):
                    scripts_imports.append(str(path))
                if module == "s2and.model" or module.startswith("s2and.model."):
                    model_imports.append(str(path))

    assert scripts_imports == []
    assert model_imports == []


def test_promoted_training_defaults_to_minimal_raw_specter_source() -> None:
    parser = promoted_train.build_parser()
    parser_defaults = vars(parser.parse_args([]))
    feature_mode_action = next(action for action in parser._actions if action.dest == "feature_mode")  # noqa: SLF001

    assert promoted_train.DEFAULT_SOURCE_BUNDLE_ROOT.name == "joint_safe_link_minimal_raw_specter_20260507a"
    assert promoted_train.DEFAULT_TARGET_JSON.relative_to(promoted_train.REPO_ROOT) == Path(
        "s2and/data/production_incremental_linker_v1.2/training_target.json"
    )
    assert parser_defaults["feature_mode"] == "minimal-raw-rust"
    assert feature_mode_action.choices == ("minimal-raw-rust", "rust-recompute-pw")
    assert "promoted_feature_bundle_root" not in parser_defaults
    assert parser_defaults["prod_holdout_importance_weight"] == 10.0
    assert parser_defaults["hyperopt"] is False
    assert parser_defaults["hyperopt_evals"] is None
    assert parser_defaults["hyperopt_metric"] == "weighted_average_error"
    assert "minimal_raw_component_scope" not in parser_defaults
    assert "minimal_raw_compare_pw_scopes" not in parser_defaults


def test_promoted_training_uses_extracted_training_helpers() -> None:
    source = inspect.getsource(promoted_train)
    disallowed_imports = (
        "scripts.eval_cluster_retrieval",
        "scripts.giant_block_cluster_retrieval_task",
        "scripts.single_letter_retrieval_utils",
        "scripts.retrieval_policy",
    )

    assert all(value not in source for value in disallowed_imports)
    assert "s2and.incremental_linking_training" in source


def test_run_classic_keeps_artifact_export_hook() -> None:
    signature = inspect.signature(run_classic)

    assert "save_artifact_to" in signature.parameters
    assert signature.parameters["save_artifact_to"].default is None


def test_hyperopt_loss_uses_weighted_average_error() -> None:
    summary = {
        "training_summary": {"rows": 10, "positive_rows": 3},
        "abstain_rule": {"promoted_stratified_gate": {"selected_gate_name": "gate"}},
        "stratified_eval_test_split": {
            "overall": {
                "test": {
                    "n_queries": 100,
                    "accuracy": 0.9,
                    "balanced_accuracy": 0.9,
                    "error_rate": 0.1,
                    "errors": 10,
                    "false_abstain": 4,
                    "false_link": 3,
                    "wrong_candidate_link": 3,
                }
            }
        },
    }

    observed = promoted_train._observed_official_metrics(summary)  # noqa: SLF001

    assert observed["false_abstain_error_rate"] == pytest.approx(0.04)
    assert observed["false_link_error_rate"] == pytest.approx(0.03)
    assert observed["wrong_link_error_rate"] == pytest.approx(0.03)
    assert observed["weighted_average_error"] == pytest.approx(((0.25 * 0.04) + 0.03 + (1.5 * 0.03)) / 2.75)
    assert promoted_train._hyperopt_loss(summary, "weighted_average_error") == pytest.approx(  # noqa: SLF001
        observed["weighted_average_error"]
    )
    assert promoted_train._metric_deltas(  # noqa: SLF001
        {"weighted_average_error_weights": observed["weighted_average_error_weights"]},
        {"metrics": {"weighted_average_error_weights": observed["weighted_average_error_weights"]}},
    ) == {"weighted_average_error_weights": True}


def test_hyperopt_includes_base_params_as_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="test",
        assets={},
        models={"classic": {"feature_columns": ["f0"], "best_params": {"n_estimators": 10}}},
        expected_metrics={},
    )
    calls: list[dict[str, Any]] = []

    def fake_run_classic(feature_bundle: Any, output_dir: Path) -> dict[str, Any]:
        calls.append({"params": dict(feature_bundle.models["classic"]["best_params"]), "output_dir": output_dir})
        return {
            "training_summary": {"rows": 3, "positive_rows": 1},
            "abstain_rule": {"promoted_stratified_gate": {"selected_gate_name": "gate"}},
            "stratified_eval_test_split": {
                "overall": {
                    "test": {
                        "accuracy": 1.0,
                        "balanced_accuracy": 1.0,
                        "error_rate": 0.0,
                        "n_queries": 3,
                        "errors": 0,
                        "false_abstain": 0,
                        "false_link": 0,
                        "wrong_candidate_link": 0,
                    }
                }
            },
        }

    monkeypatch.setattr(promoted_train, "run_classic", fake_run_classic)

    best_params, summary = promoted_train._run_classic_hyperopt(  # noqa: SLF001
        feature_bundle=bundle,
        output_dir=tmp_path / "hyperopt",
        base_params={"n_estimators": 10},
        hyperopt_evals=1,
        metric="weighted_average_error",
        seed=13,
    )

    assert calls == [{"params": {"n_estimators": 10}, "output_dir": tmp_path / "hyperopt" / "trial_000"}]
    assert best_params == {"n_estimators": 10}
    assert summary["base_loss"] == 0.0
    assert summary["best_source"] == "base_params"
    assert summary["hyperopt_search_evals"] == 0
    assert summary["hyperopt_trials_ran"] == 1


def _write_candidate_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def test_prepare_prod_training_data_weights_calibration_and_eval_rows(tmp_path: Path) -> None:
    _write_candidate_rows(
        tmp_path / "train.csv.gz",
        [
            {"query_group_id": "q_train", "base_group_id": "b_train", "retrieval_rank": 1, "label": 0, "f0": 0.1},
            {"query_group_id": "q_train", "base_group_id": "b_train", "retrieval_rank": 2, "label": 1, "f0": 0.2},
            {"query_group_id": "q_calib", "base_group_id": "b_calib", "retrieval_rank": 1, "label": 1, "f0": 0.3},
        ],
    )
    _write_candidate_rows(
        tmp_path / "calib.csv.gz",
        [
            {"query_group_id": "q_calib", "base_group_id": "b_calib", "retrieval_rank": 1, "label": 1, "f0": 0.4},
            {"query_group_id": "q_calib", "base_group_id": "b_calib", "retrieval_rank": 2, "label": 0, "f0": 0.5},
        ],
    )
    _write_candidate_rows(
        tmp_path / "s2and.csv.gz",
        [
            {"query_group_id": "q_s2and", "base_group_id": "b_s2and", "retrieval_rank": 1, "label": 1, "f0": 0.6},
            {"query_group_id": "q_s2and", "base_group_id": "b_s2and", "retrieval_rank": 30, "label": 0, "f0": 0.7},
        ],
    )
    _write_candidate_rows(
        tmp_path / "hwang.csv.gz",
        [
            {"query_group_id": "q_hwang", "base_group_id": "b_hwang", "retrieval_rank": 1, "label": 0, "f0": 0.8},
        ],
    )
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="test",
        assets={},
        models={
            "classic": {
                "train_path": "train.csv.gz",
                "classic_gate_source_path": "calib.csv.gz",
                "s2and_eval_path": "s2and.csv.gz",
                "hwang_eval_path": "hwang.csv.gz",
                "feature_columns": ["f0"],
                "best_params": {"n_estimators": 10},
            }
        },
        expected_metrics={},
    )

    prod_data = promoted_train._prepare_prod_training_data(  # noqa: SLF001
        bundle,
        holdout_importance_weight=10.0,
    )

    assert prod_data.rows["query_group_id"].tolist() == [
        "q_train",
        "q_train",
        "q_calib",
        "q_calib",
        "q_s2and",
        "q_hwang",
    ]
    assert prod_data.sample_weight.tolist() == pytest.approx([0.5, 0.5, 5.0, 5.0, 10.0, 10.0])
    summaries = {summary["source"]: summary for summary in prod_data.source_summaries}
    assert summaries["train"]["sample_weight_sum"] == pytest.approx(1.0)
    assert summaries["classic_gate_source"]["sample_weight_sum"] == pytest.approx(10.0)
    assert summaries["s2and"]["sample_weight_sum"] == pytest.approx(10.0)
    assert summaries["hwang"]["sample_weight_sum"] == pytest.approx(10.0)
    assert prod_data.train_holdout_filter_summary["rows_removed"] == 1


def test_run_uses_hyperopt_params_and_saves_only_final_prod_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = {
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 10},
        "metrics": {"stratified_test_errors": 0},
    }
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="test",
        assets={},
        models={"classic": {"feature_columns": ["f0"], "best_params": {"n_estimators": 10}}},
        expected_metrics={},
    )
    run_classic_calls: list[dict[str, Any]] = []
    prod_calls: list[dict[str, Any]] = []

    def fake_hyperopt(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        assert kwargs["base_params"] == {"n_estimators": 10}
        assert kwargs["metric"] == "weighted_average_error"
        return {"n_estimators": 42}, {"enabled": True, "best_params": {"n_estimators": 42}}

    def fake_run_classic(feature_bundle: Any, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        run_classic_calls.append(
            {
                "params": dict(feature_bundle.models["classic"]["best_params"]),
                "save_artifact_to": kwargs.get("save_artifact_to"),
                "output_dir": output_dir,
            }
        )
        return {
            "training_summary": {"rows": 3, "positive_rows": 1},
            "abstain_rule": {"promoted_stratified_gate": {"selected_gate_name": "gate"}},
            "stratified_eval_test_split": {
                "overall": {
                    "test": {
                        "accuracy": 1.0,
                        "balanced_accuracy": 1.0,
                        "error_rate": 0.0,
                        "n_queries": 3,
                        "errors": 0,
                        "false_abstain": 0,
                        "false_link": 0,
                        "wrong_candidate_link": 0,
                    }
                }
            },
        }

    def fake_train_prod(**kwargs: Any) -> dict[str, Any]:
        prod_calls.append(
            {
                "params": dict(kwargs["feature_bundle"].models["classic"]["best_params"]),
                "holdout_importance_weight": kwargs["holdout_importance_weight"],
                "save_artifact_to": kwargs["save_artifact_to"],
            }
        )
        return {"path": str(kwargs["save_artifact_to"]), "training_summary": {"rows": 9}}

    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_promoted_feature_bundle",
        lambda **_kwargs: (bundle, [{"mode": "rust-recompute-pw"}]),
    )
    monkeypatch.setattr(promoted_train, "_run_classic_hyperopt", fake_hyperopt)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "run_classic", fake_run_classic)
    monkeypatch.setattr(promoted_train, "_train_and_save_prod_artifact", fake_train_prod)  # noqa: SLF001

    artifact_dir = tmp_path / "artifact"
    args = promoted_train.build_parser().parse_args(
        [
            "--feature-mode",
            "rust-recompute-pw",
            "--run-full",
            "--hyperopt-evals",
            "2",
            "--save-artifact-to",
            str(artifact_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    result = promoted_train.run(args)

    assert run_classic_calls == [
        {"params": {"n_estimators": 42}, "save_artifact_to": None, "output_dir": tmp_path / "out" / "classic"}
    ]
    assert prod_calls == [
        {
            "params": {"n_estimators": 42},
            "holdout_importance_weight": 10.0,
            "save_artifact_to": artifact_dir.resolve(),
        }
    ]
    assert result["n_estimators"] == 42
    assert result["artifact_summary"]["path"] == str(artifact_dir.resolve())
    assert result["metric_drift_check"] == "skipped_after_hyperopt_param_search"
