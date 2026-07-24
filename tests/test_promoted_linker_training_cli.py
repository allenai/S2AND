from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from s2and.arrow_inputs import ValidatedArrowInputs
from scripts.production.model import linker_train_calibrate_eval as promoted_train
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

REQUIRED_TRAINING_ARGS = (
    "--pairwise-model-path",
    "pairwise-stage",
    "--target-json",
    "target.json",
)


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


def test_negative_limit_rows_is_rejected() -> None:
    args = promoted_train.build_parser().parse_args([*REQUIRED_TRAINING_ARGS, "--limit-rows", "-1"])

    with pytest.raises(SystemExit, match="--limit-rows must be > 0"):
        promoted_train.run(args)


@pytest.mark.parametrize(
    "removed_args",
    (
        ("--feature-mode", "precomputed-promoted"),
        ("--precomputed-feature-bundle-root", "features"),
        ("--reuse-existing-features",),
    ),
)
def test_removed_cached_feature_interfaces_are_rejected(removed_args: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit):
        promoted_train.build_parser().parse_args([*REQUIRED_TRAINING_ARGS, *removed_args])


def test_existing_production_bundle_output_is_rejected_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: {})  # noqa: SLF001
    production_bundle = tmp_path / "production_model_v9.9"
    production_bundle.mkdir()
    run_output = tmp_path / "must_not_be_created"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--save-production-bundle-to",
            str(production_bundle),
            "--output-dir",
            str(run_output),
        ]
    )

    with pytest.raises(SystemExit, match="must name a new directory"):
        promoted_train.run(args)

    assert not run_output.exists()


@pytest.mark.parametrize("save_option", ["--save-artifact-to", "--save-production-bundle-to"])
def test_metric_drift_override_cannot_promote_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    save_option: str,
) -> None:
    def fail_load(_path: Path) -> dict[str, Any]:
        raise AssertionError("promotion validation must run before loading the target")

    monkeypatch.setattr(promoted_train, "_load_target", fail_load)  # noqa: SLF001
    output_dir = tmp_path / "must_not_be_created"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--allow-metric-drift",
            save_option,
            str(tmp_path / "promotion"),
            "--output-dir",
            str(output_dir),
        ]
    )

    with pytest.raises(SystemExit, match="--allow-metric-drift is diagnostic-only"):
        promoted_train.run(args)

    assert not output_dir.exists()


@pytest.mark.parametrize("retrieval_rank", [-1, 0, 65536])
def test_arrow_rust_materialization_rejects_invalid_retrieval_rank(
    monkeypatch: pytest.MonkeyPatch,
    retrieval_rank: int,
) -> None:
    class RustModule:
        @staticmethod
        def raw_arrow_labeled_candidate_plan(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("rank validation should run before Rust planning")

    monkeypatch.setattr(promoted_train.feature_port, "_require_rust_runtime", lambda: RustModule)
    context = promoted_train.ArrowRustDatasetContext(
        dataset_name="dummy",
        row_component_scope="block-local",
        pairwise_component_scope="block-local",
        runtime_context=SimpleNamespace(),
        arrow_paths={},
        component_members={},
        cluster_seeds_require={},
        cluster_seeds_disallow=frozenset(),
        seed_constrained_signature_ids=frozenset(),
        max_block_component_size=1,
    )
    rows = pd.DataFrame(
        [
            {
                "query_signature_id": "q1",
                "query_view": "full",
                "query_group_id": "g1",
                "candidate_component_key": "c1",
                "retrieval_rank": retrieval_rank,
                "label": 0,
            }
        ]
    )

    with pytest.raises(ValueError, match="retrieval_rank"):
        promoted_train._materialize_arrow_rust_dataset_rows(  # noqa: SLF001
            context=context,
            rows=rows,
            target_features=[],
            clusterer=SimpleNamespace(),
            n_jobs=1,
            total_ram_bytes=1,
            max_exemplars=1,
            pairwise_model_nan_value=0.0,
            pairwise_aggregate_nan_value=0.0,
            row_nan_policy="zero",
        )


def test_arrow_rust_materialization_passes_concrete_paths_to_native_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dataset = build_dummy_dataset("native_planner_boundary", mode="train")
    arrow_dataset = build_arrow_training_dataset(source_dataset, tmp_path / "arrow")
    arrow_paths = arrow_dataset.arrow_paths
    assert isinstance(arrow_paths, ValidatedArrowInputs)
    real_rust_module = promoted_train.feature_port._require_rust_runtime()  # noqa: SLF001
    captured: dict[str, Any] = {}

    class CapturingRustModule:
        @staticmethod
        def raw_arrow_labeled_candidate_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["name_counts_index"] = kwargs.get("name_counts_index")
            plan = real_rust_module.raw_arrow_labeled_candidate_plan(*args, **kwargs)
            captured["reused_name_counts_index"] = plan["telemetry"]["reused_name_counts_index"]
            return plan

    monkeypatch.setattr(promoted_train.feature_port, "_require_rust_runtime", lambda: CapturingRustModule)

    def stop_after_native_plan(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("native planner accepted concrete Arrow paths")

    monkeypatch.setattr(
        promoted_train.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        stop_after_native_plan,
    )
    context = promoted_train.ArrowRustDatasetContext(
        dataset_name="native_planner_boundary",
        row_component_scope="block-local",
        pairwise_component_scope="block-local",
        runtime_context=SimpleNamespace(),
        arrow_paths=arrow_paths,
        component_members={"candidate": ("1",)},
        cluster_seeds_require={},
        cluster_seeds_disallow=frozenset(),
        seed_constrained_signature_ids=frozenset(),
        max_block_component_size=1,
    )
    rows = pd.DataFrame(
        [
            {
                "query_signature_id": "0",
                "query_view": "full",
                "query_group_id": "group",
                "candidate_component_key": "candidate",
                "retrieval_rank": 1,
                "label": 0,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="native planner accepted concrete Arrow paths"):
        promoted_train._materialize_arrow_rust_dataset_rows(  # noqa: SLF001
            context=context,
            rows=rows,
            target_features=[],
            clusterer=SimpleNamespace(),
            n_jobs=1,
            total_ram_bytes=1,
            max_exemplars=1,
            pairwise_model_nan_value=0.0,
            pairwise_aggregate_nan_value=0.0,
            row_nan_policy="zero",
        )

    assert captured["name_counts_index"] is arrow_paths.native_name_counts_index
    assert captured["reused_name_counts_index"] is True


def test_finalized_arrow_materialization_bundle_creates_corrected_feature_asset_group(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    labels_path = source_root / "labels" / "train.parquet"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"query_group_id": "q1", "retrieval_rank": 1, "label": 1}]).to_parquet(labels_path, index=False)
    payload = {
        "bundle_name": "arrow_source",
        "assets": {
            "featureless_rows": {
                "root": "labels",
                "files": {
                    "train_path": "labels/train.parquet",
                },
            },
        },
        "models": {
            "classic": {
                "feature_columns": [],
                "best_params": {},
            },
        },
        "expected_metrics": {},
    }
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    bundle = promoted_train._finalize_arrow_rust_bundle_metadata(  # noqa: SLF001
        source_bundle=promoted_train.load_bundle(source_root),
        output_bundle_root=output_root,
        target={"feature_count": 1, "features": ["f0"], "params": {"n_estimators": 10}, "metrics": {}},
        selected_keys=["train_path"],
    )

    assert bundle.assets["corrected_feature_rows"]["root"] == "features_corrected"
    feature_path = "features_corrected/train.parquet"
    assert bundle.assets["corrected_feature_rows"]["files"] == {"train_path": feature_path}
    assert bundle.models["classic"]["train_path"] == feature_path


def test_hyperopt_loss_uses_weighted_average_error() -> None:
    summary = {
        "training_summary": {"rows": 10, "positive_rows": 3},
        "abstain_rule": {"promoted_logistic_gate": {"mode": "promoted_logistic_topk_multiclass_l2"}},
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


@pytest.mark.parametrize(
    ("observed", "target", "message"),
    (
        ({}, {"metrics": {"score": 1.0}}, "missing target metrics"),
        ({"score": float("nan")}, {"metrics": {"score": 1.0}}, "must be finite"),
        ({"score": float("inf")}, {"metrics": {"score": 1.0}}, "must be finite"),
        ({"score": float("-inf")}, {"metrics": {"score": 1.0}}, "must be finite"),
        ({}, {"metrics": {}}, "must not be empty"),
    ),
)
def test_metric_gate_rejects_missing_and_nonfinite_values(
    observed: dict[str, float],
    target: dict[str, dict[str, float]],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        promoted_train._assert_no_metric_drift(observed, target)  # noqa: SLF001


def test_metric_gate_runs_before_artifact_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = {
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 1},
        "metrics": {"stratified_test_errors": 0},
    }
    bundle = promoted_train.OfficialBundle(
        root=tmp_path,
        bundle_name="gate-order",
        assets={},
        models={"classic": {"feature_columns": ["f0"], "best_params": {"n_estimators": 1}}},
        expected_metrics={},
    )
    summary = {
        "training_summary": {"rows": 3, "positive_rows": 1},
        "stratified_eval_test_split": {
            "overall": {
                "test": {
                    "accuracy": 0.0,
                    "balanced_accuracy": 0.0,
                    "error_rate": 1.0,
                    "n_queries": 1,
                    "errors": 1,
                    "false_abstain": 1,
                    "false_link": 0,
                    "wrong_candidate_link": 0,
                }
            }
        },
    }
    promoted = False

    def fail_if_promoted(**_kwargs: Any) -> dict[str, Any]:
        nonlocal promoted
        promoted = True
        raise AssertionError("artifact promotion ran before metric gate")

    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: (bundle, [{"mode": "arrow-rust"}]),
    )
    monkeypatch.setattr(
        promoted_train,
        "_assert_pairwise_model_supports_arrow_materialization",
        lambda *_args: None,
    )
    monkeypatch.setattr(promoted_train, "run_classic", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(promoted_train, "_train_and_save_prod_artifact", fail_if_promoted)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path: {"test": "binding"})
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: SimpleNamespace())
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--run-full",
            "--save-artifact-to",
            str(tmp_path / "artifact"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    with pytest.raises(RuntimeError, match="drifted from target metrics"):
        promoted_train.run(args)

    assert not promoted
    assert not (tmp_path / "artifact").exists()


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

    def fake_run_classic(feature_bundle: Any, output_dir: Path, **_kwargs: Any) -> dict[str, Any]:
        calls.append({"params": dict(feature_bundle.models["classic"]["best_params"]), "output_dir": output_dir})
        return {
            "training_summary": {"rows": 3, "positive_rows": 1},
            "abstain_rule": {"promoted_logistic_gate": {"mode": "promoted_logistic_topk_multiclass_l2"}},
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


def test_prepare_prod_training_data_weights_calibration_rows_and_leaves_test_for_gate(tmp_path: Path) -> None:
    _write_candidate_rows(
        tmp_path / "train.csv.gz",
        [
            {
                "query_group_id": "q_train",
                "base_group_id": "b_train",
                "candidate_component_key": "c_train_1",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 0,
                "f0": 0.1,
            },
            {
                "query_group_id": "q_train",
                "base_group_id": "b_train",
                "candidate_component_key": "c_train_2",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 2,
                "label": 1,
                "f0": 0.2,
            },
            {
                "query_group_id": "q_calib",
                "base_group_id": "b_calib",
                "candidate_component_key": "c_train_shadowed",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 1,
                "f0": 0.3,
            },
            {
                "query_group_id": "q_unlabeled",
                "base_group_id": "b_unlabeled",
                "candidate_component_key": "c_unlabeled",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 0,
                "supervision_type": "unlabeled_singleton_orcid",
                "f0": 0.9,
            },
        ],
    )
    _write_candidate_rows(
        tmp_path / "calib.csv.gz",
        [
            {
                "query_group_id": "q_calib",
                "base_group_id": "b_calib",
                "candidate_component_key": "c_calib_pos",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 1,
                "f0": 0.4,
            },
            {
                "query_group_id": "q_calib",
                "base_group_id": "b_calib",
                "candidate_component_key": "c_calib_neg",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 2,
                "label": 0,
                "f0": 0.5,
            },
        ],
    )
    _write_candidate_rows(
        tmp_path / "s2and.csv.gz",
        [
            {
                "query_group_id": "q_s2and",
                "base_group_id": "b_s2and",
                "candidate_component_key": "c_s2and",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 1,
                "f0": 0.6,
            },
            {
                "query_group_id": "q_s2and",
                "base_group_id": "b_s2and",
                "candidate_component_key": "c_s2and_late",
                "dataset": "unit",
                "query_view": "full",
                "retrieval_rank": 30,
                "label": 0,
                "f0": 0.7,
            },
        ],
    )
    _write_candidate_rows(
        tmp_path / "hwang.csv.gz",
        [
            {
                "query_group_id": "q_hwang",
                "base_group_id": "b_hwang",
                "candidate_component_key": "c_hwang",
                "dataset": "hwang",
                "query_view": "full",
                "retrieval_rank": 1,
                "label": 0,
                "f0": 0.8,
            },
        ],
    )
    (tmp_path / "splits").mkdir()
    pd.DataFrame(
        [
            {"query_group_id": "q_calib", "source_key": "s2and_eval", "split": "calibration_fit"},
            {"query_group_id": "q_hwang", "source_key": "hwang_eval", "split": "test"},
        ]
    ).to_csv(tmp_path / "splits" / "assignments.csv", index=False)
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
                "stratified_eval_test_split": {
                    "assignments_path": "splits/assignments.csv",
                    "test_split": "test",
                },
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit"],
                    "test_split": "test",
                },
                "feature_columns": ["f0"],
                "best_params": {"n_estimators": 10},
            }
        },
        expected_metrics={},
    )

    prod_data = promoted_train._prepare_prod_training_data(  # noqa: SLF001
        bundle,
        holdout_importance_weight=10.0,
        retrieval_rank_limit=promoted_train.DEFAULT_RETRIEVAL_TOP_K,
    )

    assert prod_data.rows["query_group_id"].tolist() == [
        "q_train",
        "q_train",
        "q_calib",
        "q_calib",
    ]
    assert prod_data.sample_weight.tolist() == pytest.approx([0.5, 0.5, 5.0, 5.0])
    summaries = {summary["source"]: summary for summary in prod_data.source_summaries}
    assert summaries["train"]["sample_weight_sum"] == pytest.approx(1.0)
    assert summaries["stratified_calibration_calibration_fit"]["sample_weight_sum"] == pytest.approx(10.0)
    assert summaries["stratified_calibration_calibration_fit"]["splits"] == ["calibration_fit"]
    assert summaries["stratified_calibration_calibration_fit"]["source_keys"] == ["s2and_eval"]
    assert prod_data.train_holdout_filter_summary["rows_removed"] == 1
    assert "q_unlabeled" not in set(prod_data.rows["query_group_id"].astype(str))


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
                "output_dir": output_dir,
            }
        )
        assert set(kwargs) == {"n_jobs"}
        return {
            "training_summary": {"rows": 3, "positive_rows": 1},
            "abstain_rule": {"promoted_logistic_gate": {"mode": "promoted_logistic_topk_multiclass_l2"}},
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
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: SimpleNamespace(batch_size=10))
    monkeypatch.setattr(promoted_train, "_assert_pairwise_model_supports_arrow_materialization", lambda *_args: None)  # noqa: SLF001
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: (bundle, [{"mode": "arrow-rust"}]),
    )
    monkeypatch.setattr(promoted_train, "_run_classic_hyperopt", fake_hyperopt)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "run_classic", fake_run_classic)
    monkeypatch.setattr(promoted_train, "_train_and_save_prod_artifact", fake_train_prod)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path: {"test": "binding"})

    artifact_dir = tmp_path / "artifact"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
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

    assert run_classic_calls == [{"params": {"n_estimators": 42}, "output_dir": tmp_path / "out" / "classic"}]
    assert prod_calls == [
        {
            "params": {"n_estimators": 42},
            "holdout_importance_weight": 10.0,
            "save_artifact_to": artifact_dir.resolve(),
        }
    ]
    assert result["n_estimators"] == 42
    assert result["artifact_summary"]["path"] == str(artifact_dir.resolve())
    assert result["metric_drift_check"] == "passed"
