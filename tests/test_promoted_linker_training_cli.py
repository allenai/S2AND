from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking.runtime import CandidateBatchPairwiseModelResult
from s2and.incremental_linking_training import classic as classic_training
from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from scripts.linker import train_linker_and_finalize as promoted_train
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, write_minimal_arrow_prediction_bundle
from tests.test_linker_source_bundle import _write_arrow_root, _write_linker_inputs, _write_singleton_orcid_views

pytestmark = pytest.mark.tooling

COMMON_TRAINING_ARGS = (
    "--source-bundle-root",
    "source-bundle",
    "--output-dir",
    "linker-run",
    "--pairwise-model-path",
    "pairwise-stage",
    "--target-json",
    "target.json",
    "--name-counts-index-root",
    "name-counts-index",
    "--n-jobs",
    "1",
)
EXPLICIT_ARTIFACT_HASHES = {
    "name_counts_manifest_sha256": "a" * 64,
    "name_tuples_data_sha256": "b" * 64,
    "orcid_prefix_counts_data_sha256": "c" * 64,
}
FULL_MATERIALIZATION_SUMMARIES = tuple(
    {"table_key": key, "rows": 1, "label_filtering": {"train_holdout": {"rows_removed": 0}}}
    for key in promoted_train.REQUIRED_TABLE_KEYS
)


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        promoted_train,
        "read_classic_holdout_identities",
        lambda _bundle: classic_training.ClassicHoldoutIdentities(frozenset(), frozenset(), ()),
    )
    monkeypatch.setattr(
        promoted_train,
        "load_packaged_artifact_authority",
        lambda **_kwargs: SimpleNamespace(
            hashes=dict(EXPLICIT_ARTIFACT_HASHES),
            name_tuples=SimpleNamespace(pairs=frozenset()),
        ),
    )
    monkeypatch.setattr(
        promoted_train,
        "preflight_source_rows",
        lambda *_args, **_kwargs: (4, {}),
    )
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path, **_kwargs: {"test": "binding"})
    monkeypatch.setattr(
        promoted_train,
        "validate_source_bundle_support_files",
        lambda *_args, **_kwargs: ["test-support"],
    )
    monkeypatch.setattr(
        promoted_train,
        "_release_table_plan",
        lambda _bundle: (
            ("train_path", "classic_gate_source_path"),
            {},
            ("s2and_eval_path", "hwang_eval_path"),
        ),
    )


def test_linker_commands_require_name_counts_root() -> None:
    arguments = list(COMMON_TRAINING_ARGS)
    option_index = arguments.index("--name-counts-index-root")
    del arguments[option_index : option_index + 2]

    parser = promoted_train.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)

    arguments = list(COMMON_TRAINING_ARGS)
    option_index = arguments.index("--n-jobs")
    del arguments[option_index : option_index + 2]
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_output_paths_cannot_be_nested_under_input_bundles(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    pairwise_root = tmp_path / "pairwise"

    with pytest.raises(SystemExit, match="immutable input bundles"):
        promoted_train._assert_output_paths_outside_inputs(  # noqa: SLF001
            output_dir=pairwise_root / "linker-run",
            complete_model_dir=pairwise_root / "linker-run" / "production_model_v9.9",
            source_bundle_root=source_root,
            pairwise_model_path=pairwise_root,
        )


def test_release_table_plan_keeps_frozen_test_queries_out_of_training(tmp_path: Path) -> None:
    paths = {
        "train_path": "rows/train.parquet",
        "classic_gate_source_path": "rows/gate.parquet",
        "s2and_eval_path": "rows/s2and.parquet",
        "hwang_eval_path": "rows/hwang.parquet",
    }
    for relative_path in paths.values():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assignments_path = tmp_path / "splits" / "assignments.csv"
    assignments_path.parent.mkdir()
    pd.DataFrame(
        {
            "query_group_id": ["calibration-query", "frozen-test-query"],
            "source_key": ["hwang_eval", "s2and_eval"],
            "split": ["calibration_fit", "test"],
            "base_group_id": ["calibration-base", "test-base"],
        }
    ).to_csv(assignments_path, index=False)
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="lifecycle",
        assets={"featureless_rows": {"files": paths}},
        models={
            "classic": {
                **paths,
                "stratified_eval_test_split": {"assignments_path": "splits/assignments.csv"},
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit"],
                    "test_split": "test",
                },
            }
        },
        expected_metrics={},
    )

    training_keys, query_ids, evaluation_keys = promoted_train._release_table_plan(bundle)  # noqa: SLF001

    assert training_keys == ("train_path", "classic_gate_source_path", "hwang_eval_path")
    assert query_ids == {"hwang_eval_path": {"calibration-query"}}
    assert evaluation_keys == ("s2and_eval_path", "hwang_eval_path")


def test_run_rejects_name_count_binding_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = {
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 1},
        "metrics": {"stratified_test_errors": 0},
    }
    bundle = promoted_train.OfficialBundle(
        root=(tmp_path / "source").resolve(),
        bundle_name="preflight",
        assets={},
        models={},
        expected_metrics={},
    )
    dataset_root = tmp_path / "arrow"
    write_minimal_arrow_prediction_bundle(dataset_root)
    clusterer = SimpleNamespace()
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: clusterer)

    def reject_binding(actual_clusterer: Any, actual_dataset: Any, *, context: str) -> None:
        assert actual_clusterer is clusterer
        assert actual_dataset is arrow_dataset
        assert "toy" in context
        raise ValueError("name-count generation mismatch")

    monkeypatch.setattr(promoted_train, "require_arrow_name_counts_index_for_clusterer", reject_binding)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: pytest.fail("binding mismatch reached materialization"),
    )
    output_dir = tmp_path / "output"
    args = promoted_train.build_parser().parse_args(
        [
            *COMMON_TRAINING_ARGS,
            "--output-dir",
            str(output_dir),
        ]
    )

    with ArrowDataset.open(dataset_root) as arrow_dataset:
        monkeypatch.setattr(
            promoted_train,
            "preflight_source_rows",
            lambda *_args, **_kwargs: (1, {"toy": arrow_dataset}),
        )
        with pytest.raises(ValueError, match="name-count generation mismatch"):
            promoted_train.run(args)

    assert not output_dir.exists()


def test_arrow_rust_materialization_passes_dataset_handle_to_native_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dataset = build_dummy_dataset("native_planner_boundary", mode="train")
    training_dataset = build_arrow_training_dataset(source_dataset, tmp_path / "arrow")
    arrow_dataset = training_dataset.arrow_dataset
    assert isinstance(arrow_dataset, ArrowDataset)
    real_rust_module = promoted_train.feature_port._require_rust_runtime()  # noqa: SLF001
    captured: dict[str, Any] = {}

    class CapturingPlanner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["native_dataset"] = args[0]
            self.planner = real_rust_module.RawArrowLabeledCandidatePlanner(*args, **kwargs)

        def plan(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            plan = self.planner.plan(*args, **kwargs)
            captured["reused_name_counts_index"] = plan["telemetry"]["reused_name_counts_index"]
            return plan

    class CapturingRustModule:
        RawArrowLabeledCandidatePlanner = CapturingPlanner

    monkeypatch.setattr(promoted_train.feature_port, "_require_rust_runtime", lambda: CapturingRustModule)

    def stop_after_native_plan(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("native planner accepted the retained Arrow dataset")

    monkeypatch.setattr(
        promoted_train.feature_port,
        "build_rust_featurizer_from_arrow_dataset",
        stop_after_native_plan,
    )
    context = promoted_train.ArrowRustDatasetContext(
        dataset_name="native_planner_boundary",
        row_component_scope="block-local",
        pairwise_component_scope="block-local",
        runtime_context=SimpleNamespace(),
        arrow_dataset=arrow_dataset,
        component_members={"candidate": ("1",)},
        cluster_seeds_require={},
        cluster_seeds_disallow=frozenset(),
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

    try:
        with pytest.raises(RuntimeError, match="native planner accepted the retained Arrow dataset"):
            promoted_train._materialize_arrow_rust_dataset_rows(  # noqa: SLF001
                context=context,
                rows=rows,
                target_features=[],
                name_tuples=frozenset(),
                clusterer=SimpleNamespace(),
                n_jobs=1,
                total_ram_bytes=1,
                max_exemplars=1,
                pairwise_model_nan_value=0.0,
                pairwise_aggregate_nan_value=0.0,
            )

        assert captured["native_dataset"] is arrow_dataset.native
        assert captured["reused_name_counts_index"] is True
    finally:
        arrow_dataset.close()


def test_finalized_arrow_materialization_bundle_loads_all_optional_eval_paths(tmp_path: Path) -> None:
    source_classic = {"retrieval_top_k": 7, "classic_gate_calibration_retrieval_limit": 5}
    retrieval_top_k = 7
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_files = {
        "train_path": "labels/train.parquet",
        "s_park_eval_path": "labels/s_park.parquet",
        "s_lee_eval_path": "labels/s_lee.parquet",
        "extra_eval_paths.j_smith": "labels/j_smith.parquet",
    }
    for index, relpath in enumerate(source_files.values()):
        path = source_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"query_group_id": f"q{index}", "retrieval_rank": 1, "label": 1}]).to_parquet(
            path,
            index=False,
        )
    payload = {
        "bundle_name": "arrow_source",
        "assets": {
            "featureless_rows": {
                "root": "labels",
                "files": source_files,
            },
        },
        "models": {"classic": source_classic},
        "expected_metrics": {},
    }
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_bytes = {
        path.relative_to(source_root): path.read_bytes() for path in source_root.rglob("*") if path.is_file()
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "bundle.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for relpath in source_files.values():
        output_path = output_root / "features_corrected" / Path(relpath).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"query_group_id": Path(relpath).stem}]).to_parquet(output_path, index=False)

    source_bundle = promoted_train.load_bundle(source_root)
    assert source_bundle.root == source_root.resolve()
    assert source_bundle.bundle_name == "arrow_source"
    assert source_bundle.models == {"classic": source_classic}
    assert source_bundle.expected_metrics == {}

    bundle = promoted_train._finalize_arrow_rust_bundle_metadata(  # noqa: SLF001
        source_bundle=source_bundle,
        output_bundle_root=output_root,
        target={"feature_count": 1, "features": ["f0"], "params": {"n_estimators": 10}, "metrics": {}},
        selected_keys=list(source_files),
        retrieval_top_k=retrieval_top_k,
    )

    assert bundle.models["classic"]["retrieval_top_k"] == retrieval_top_k
    assert bundle.models["classic"]["classic_gate_calibration_retrieval_limit"] == retrieval_top_k
    assert bundle.models["classic"]["feature_columns"] == ["f0"]
    assert bundle.models["classic"]["best_params"] == {"n_estimators": 10}
    persisted = json.loads((output_root / "bundle.json").read_text(encoding="utf-8"))
    assert persisted["models"]["classic"]["retrieval_top_k"] == retrieval_top_k
    assert persisted["models"]["classic"]["classic_gate_calibration_retrieval_limit"] == retrieval_top_k
    assert source_bundle.models == {"classic": source_classic}
    assert {path: (source_root / path).read_bytes() for path in source_bytes} == source_bytes
    assert bundle.assets["corrected_feature_rows"]["root"] == "features_corrected"
    assert bundle.models["classic"]["s_park_eval_path"] == "features_corrected/s_park.parquet"
    assert bundle.models["classic"]["s_lee_eval_path"] == "features_corrected/s_lee.parquet"
    assert bundle.models["classic"]["extra_eval_paths"] == {
        "j_smith": "features_corrected/j_smith.parquet",
    }
    optional_paths = classic_training._iter_extra_eval_paths(bundle.models["classic"])  # noqa: SLF001

    assert [dataset for dataset, _path in optional_paths] == ["s_park", "s_lee", "j_smith"]
    assert classic_training._summary_key_for_eval_dataset("j_smith") == "overall_j_smith_eval"
    for _dataset, path in optional_paths:
        assert len(pd.read_parquet(classic_training._resolve_path(bundle, path))) == 1  # noqa: SLF001


def _valid_observed_metrics() -> dict[str, Any]:
    return promoted_train._observed_official_metrics(  # noqa: SLF001
        {
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
    )


def test_observed_official_metrics_validate_weighted_error_and_counts() -> None:
    observed = _valid_observed_metrics()
    promoted_train._validate_observed_official_metrics(observed)  # noqa: SLF001

    assert observed["false_abstain_error_rate"] == pytest.approx(0.04)
    assert observed["false_link_error_rate"] == pytest.approx(0.03)
    assert observed["wrong_link_error_rate"] == pytest.approx(0.03)
    assert observed["weighted_average_error"] == pytest.approx(((0.25 * 0.04) + 0.03 + (1.5 * 0.03)) / 2.75)

    incomplete = observed.copy()
    incomplete.pop("weighted_average_error")
    with pytest.raises(ValueError, match="complete official metric set"):
        promoted_train._validate_observed_official_metrics(incomplete)  # noqa: SLF001

    for field, value, message in (
        ("stratified_test_accuracy", float("nan"), "must be finite"),
        ("stratified_test_accuracy", 1.1, r"must be in \[0, 1\]"),
        ("stratified_test_errors", -1, "nonnegative integer"),
        ("training_rows", 0, "training counts are inconsistent"),
        ("stratified_test_errors", 101, "test counts are inconsistent"),
    ):
        with pytest.raises(ValueError, match=message):
            promoted_train._validate_observed_official_metrics(observed | {field: value})  # noqa: SLF001


def test_observed_official_metrics_reject_invalid_error_weights() -> None:
    observed = _valid_observed_metrics()
    for value, message in ((0.0, "must be positive"), (2.0, "must equal the official value")):
        weights = {**promoted_train.WEIGHTED_ERROR_WEIGHTS, "false_link_error_rate": value}
        with pytest.raises(ValueError, match=message):
            promoted_train._validate_observed_official_metrics(  # noqa: SLF001
                observed | {"weighted_average_error_weights": weights},
            )


def test_query_prediction_export_is_deterministic(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "base_group_id": ["base-2", "base-1"],
            "chosen_candidate_component_key": ["candidate-2", "candidate-1"],
            "chosen_probability": [0.25, 0.75],
            "correct": [0, 1],
            "predicted_action": ["abstain", "link_candidate"],
            "query_case_id": ["query-2", "query-1"],
            "query_safe_target": [1, 1],
            "query_safe_target_source": ["retrieved_window", "manual_override"],
            "source_key": ["source", "source"],
            "split": ["test", "test"],
        }
    )
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    first = classic_training._write_query_predictions(rows, first_path)  # noqa: SLF001
    second = classic_training._write_query_predictions(rows.iloc[::-1], second_path)  # noqa: SLF001

    assert first["sha256"] == second["sha256"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["columns"][:5] == ["source_key", "query_case_id", "base_group_id", "split", "label"]
    written = pd.read_csv(first_path)
    assert written["query_safe_target"].tolist() == [1, 1]
    assert written["query_safe_target_source"].tolist() == ["manual_override", "retrieved_window"]


@pytest.mark.parametrize("problem", [None, "unreviewed-singletons", "missing-test-group", "frozen", "non-top25"])
def test_release_serializes_complete_bundle_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    problem: str | None,
) -> None:
    target = {
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 1},
        "metrics": {"stratified_test_errors": 0},
    }
    source_root = tmp_path / "source"
    _write_arrow_root(source_root, dataset="replay_demo", replay=True)
    _write_linker_inputs(source_root, leak=False)
    _write_singleton_orcid_views(
        source_root, view_count=10, positive_view_count=0 if problem == "unreviewed-singletons" else 1
    )
    if problem == "missing-test-group":
        labels = source_root / "labels" / "hwang_eval.parquet"
        pd.read_parquet(labels).assign(query_group_id="hwang:other").to_parquet(labels, index=False)
    if problem == "non-top25":
        source_manifest = source_root / "bundle.json"
        source_config = json.loads(source_manifest.read_text(encoding="utf-8"))
        source_config["models"]["classic"]["retrieval_top_k"] = 7
        source_manifest.write_text(json.dumps(source_config), encoding="utf-8")
    bundle = promoted_train.load_bundle(source_root)
    bundle.models["classic"]["best_params"] = {"n_estimators": 1}
    map_path = bundle.root / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    map_path.parent.mkdir(parents=True)
    map_path.write_bytes(b"")
    map_digest = hashlib.sha256(b"").hexdigest()
    summary = {
        "training_summary": {"rows": 3, "positive_rows": 1},
        "abstain_rule": {"calibration_metrics": {"accuracy": 0.75}},
        "overall_s2and_eval": {"w25": {"overall": {"accuracy": 0.5}}},
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
    order: list[str] = []

    def save_artifact(model: Any, artifact_dir: Path, **kwargs: Any) -> dict[str, Any]:
        order.append("serialize")
        assert model is calibrated.model
        assert kwargs["target_spec"] == target
        artifact_dir.mkdir()
        (artifact_dir / "booster.lgb").write_bytes(b"calibrated-booster\n")
        return {
            "booster_sha256": "a" * 64,
            "pairwise_bundle_binding_digest": "b" * 64,
            "target_spec_digest": "c" * 64,
            "retrieval_top_k": 25,
        }

    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(
        promoted_train,
        "load_packaged_artifact_authority",
        lambda **_kwargs: SimpleNamespace(
            hashes=dict(EXPLICIT_ARTIFACT_HASHES), name_tuples=SimpleNamespace(pairs=frozenset())
        ),
    )
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path, **_kwargs: {"test": "binding"})
    monkeypatch.setattr(promoted_train, "require_arrow_name_counts_index_for_clusterer", lambda *_a, **_kw: None)

    training_authorities: list[Any] = []
    stage_summaries: dict[str, list[dict[str, Any]]] = {}

    def materialize(**_kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        stage = "materialize_training" if "materialize_training" not in order else "materialize_evaluation"
        if stage == "materialize_training":
            training_authorities.append(_kwargs["holdout_identities"])
            assert _kwargs["query_ids_by_table"] == {"s2and_eval_path": {"cal:q1", "s2and:q1"}}
        else:
            assert "holdout_identities" not in _kwargs
            assert set(_kwargs["table_keys"]) == {"s2and_eval_path", "hwang_eval_path"}
        order.append(stage)
        stage_summaries[stage] = [{**item, "stage": stage} for item in FULL_MATERIALIZATION_SUMMARIES]
        return bundle, stage_summaries[stage]

    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        materialize,
    )
    output_dir = tmp_path / "out"
    query_predictions_bytes = b"query_case_id\nquery-1\n"
    calibrated = classic_training.CalibratedClassicModel(
        model=cast(Any, object()),
        gate_config={"model_type": "test"},
        retrieval_top_k=25,
        training_summary=summary["training_summary"],
        abstain_rule_summary={},
    )

    def fit_linker(_bundle: Any, **_kwargs: Any) -> Any:
        if problem == "frozen":
            assert set(_kwargs["frozen_task_rows"]) == {"fit_pool", "gate_thresh"}
            assert len(_kwargs["gate_training_blocks"]) == 1
        else:
            assert _kwargs["holdout_identities"] is training_authorities[0]
            assert _kwargs["pre_materialization_holdout_summary"] == {"rows_removed": 0}
        order.append("fit")
        return calibrated

    def finalize_bundle(**kwargs: Any) -> Any:
        order.append("finalize")
        complete_model_dir = Path(kwargs["output_bundle_dir"])
        complete_model_dir.mkdir()
        manifest_path = complete_model_dir / "manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(manifest_path=manifest_path)

    loaded_artifact = SimpleNamespace()

    def load_complete(path: Path, **_kwargs: Any) -> Any:
        order.append("reload")
        assert Path(path).is_dir()
        return SimpleNamespace(incremental_linker_artifact=loaded_artifact)

    def evaluate_linker(_bundle: Any, **kwargs: Any) -> Any:
        order.append("evaluate")
        assert kwargs["artifact"] is loaded_artifact
        return classic_training.ClassicEvaluation(
            summary=summary,
            query_predictions={
                "sha256": hashlib.sha256(query_predictions_bytes).hexdigest(),
                "bytes": len(query_predictions_bytes),
                "rows": 1,
                "columns": ["query_case_id"],
            },
        )

    monkeypatch.setattr(promoted_train, "fit_linker", fit_linker)
    monkeypatch.setattr(promoted_train, "evaluate_linker", evaluate_linker)
    monkeypatch.setattr(promoted_train, "save_incremental_linking_artifact", save_artifact)
    monkeypatch.setattr(promoted_train, "finalize_production_bundle", finalize_bundle)
    monkeypatch.setattr(promoted_train, "load_production_model", load_complete)
    validate_observed_metrics = promoted_train._validate_observed_official_metrics  # noqa: SLF001

    def validate_metrics(metrics: dict[str, Any], *, ranker_refit: bool) -> None:
        order.append("validate")
        validate_observed_metrics(metrics, ranker_refit=ranker_refit)

    monkeypatch.setattr(promoted_train, "_validate_observed_official_metrics", validate_metrics)
    monkeypatch.setattr(
        promoted_train,
        "load_clusterer",
        lambda *_args, **_kwargs: SimpleNamespace(production_model_release_version="9.9"),
    )
    args = [
        *COMMON_TRAINING_ARGS,
        "--source-bundle-root",
        str(source_root),
        "--name-counts-index-root",
        str(source_root / "name_counts_index"),
        "--output-dir",
        str(output_dir),
    ]
    if problem in {"frozen", "non-top25"}:
        role_names = ("fit_pool", "gate_thresh")
        roles = tmp_path / "roles.json"
        roles.write_text(
            json.dumps(assign_block_roles([BlockSpec(f"test {i}", 2) for i in range(2)], dict.fromkeys(role_names, 1))),
            encoding="utf-8",
        )
        manifest = tmp_path / "tasks.json"
        if problem == "frozen":
            manifest.write_text("{}", encoding="utf-8")
        args.extend(["--linker-task-manifest", str(manifest), "--block-role-assignments", str(roles)])

        def materialize_frozen_tasks(
            path: Path, *_args: Any, **_kwargs: Any
        ) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
            assert problem == "frozen", "Fetched frozen task rows before validating the source window"
            assert path == manifest
            order.append("materialize_training")
            stage_summaries["materialize_training"] = []
            return {role: pd.DataFrame() for role in role_names}, []

        monkeypatch.setattr(promoted_train, "_materialize_frozen_linker_tasks", materialize_frozen_tasks)
        if problem == "non-top25":
            with pytest.raises(ValueError, match="Frozen linker tasks require top-25"):
                promoted_train.main(args)
            assert not order
            return
    elif problem is not None:
        message = {
            "unreviewed-singletons": "0/10 marked negative_singleton_orcid views",
            "missing-test-group": "did not all match source rows.*hwang:q1",
        }[problem]
        with pytest.raises(ValueError, match=message):
            promoted_train.main(args)
        assert not output_dir.exists()
        assert not order
        return

    promoted_train.main(args)

    result = json.loads(capsys.readouterr().out)
    assert order == [
        "materialize_training",
        "fit",
        "serialize",
        "finalize",
        "reload",
        "materialize_evaluation",
        "evaluate",
        "validate",
    ]
    assert result["pairwise_bundle_binding"] == {"test": "binding"}
    assert result["initial_query_sources_sha256"] == map_digest
    assert result["complete_model_path"] == str(output_dir / "pairwise-stage")
    assert result["query_predictions"]["rows"] == 1
    assert result["evaluation_summary"] == summary
    assert result["feature_materialization_summaries"] == {
        "training": stage_summaries["materialize_training"],
        "evaluation": stage_summaries["materialize_evaluation"],
    }
    assert set(output_dir.iterdir()) == {
        output_dir / "pairwise-stage",
        output_dir / "linker_evaluation_report.json",
    }
    written_report = json.loads((output_dir / "linker_evaluation_report.json").read_text(encoding="utf-8"))
    assert written_report == result


def _fixed_ranker_fixture(tmp_path: Path, *, counts_hash: str | None = None) -> tuple[Path, Path, str]:
    from s2and._sha256 import sha256_file
    from s2and.incremental_linking.artifact import save_incremental_linking_artifact
    from s2and.production_bundle import finalize_production_bundle
    from s2and.production_model import canonical_artifact_hashes, pairwise_bundle_binding
    from tests.promoted_linking_helpers import (
        build_tiny_promoted_booster,
        tiny_logistic_gate_config,
        write_synthetic_pairwise_bundle,
    )

    pairwise = tmp_path / "pairwise"
    hashes = canonical_artifact_hashes()
    if counts_hash is not None:
        hashes["name_counts_manifest_sha256"] = counts_hash
    write_synthetic_pairwise_bundle(pairwise, artifact_hashes=hashes, release_version="1.4")
    manifest_path = pairwise / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps({**manifest, "feature_contract_version": 1}), encoding="utf-8")
    target = tmp_path / "target.json"
    target_payload = {
        "features": list(promoted_train.promoted_linker_feature_columns()),
        "feature_count": len(promoted_train.promoted_linker_feature_columns()),
        "params": {"n_estimators": 6},
        "metrics": {},
    }
    target.write_text(json.dumps(target_payload), encoding="utf-8")
    booster, _ = build_tiny_promoted_booster()
    artifact = tmp_path / "artifact"
    save_incremental_linking_artifact(
        booster,
        artifact,
        gate_config=tiny_logistic_gate_config(),
        target_spec=target_payload,
        pairwise_bundle_binding=pairwise_bundle_binding(pairwise),
    )
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_path.write_text(json.dumps({**metadata, "feature_contract_version": 1}), encoding="utf-8")
    complete = tmp_path / "fixed"
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise,
        output_bundle_dir=complete,
        incremental_linker_artifact_dir=artifact,
        target_json=target,
    )
    return pairwise, complete, sha256_file(complete / "incremental_linker" / "booster.lgb")


def _fixed_ranker_source_fixture(tmp_path: Path) -> Path:
    from tests.test_build_linker_task_tables import _ORCIDS, _block_inputs

    source = tmp_path / "source"
    _write_linker_inputs(source, leak=False)
    dataset = "replay_demo"
    block = _block_inputs(tmp_path / "replay", "a ross", _ORCIDS[0], query_year=1800, recent_seed_year=1820)
    shutil.copytree(block["arrow_dir"], source / "datasets" / dataset)
    components = source / "components" / f"{dataset}_members.parquet"
    pd.read_parquet(components).assign(signature_id="s00").to_parquet(components, index=False)
    for name, sid, label in [
        ("train", "q_full", 1),
        ("calibration", "q_initial", 1),
        ("s2and_eval", "q_full", 0),
        ("hwang_eval", "q_outside_home", 0),
    ]:
        path = source / "labels" / f"{name}.parquet"
        pd.read_parquet(path).assign(query_signature_id=sid, label=label, first_name_bucket="full").to_parquet(
            path, index=False
        )
    map_path = source / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    map_path.parent.mkdir()
    map_path.write_bytes(b"")
    return source


@pytest.mark.parametrize("check", ["run", "contract", "opt_in", "source_manifest", "legacy_frozen"])
def test_fixed_ranker_run_preserves_booster_and_calibrates_gate(
    tmp_path: Path, check: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from s2and._sha256 import sha256_file
    from s2and.production_model import load_production_model

    source = _fixed_ranker_source_fixture(tmp_path)
    counts = source / "datasets" / "replay_demo" / "name_counts_index" / "name_counts_index"
    pairwise, fixed, digest = _fixed_ranker_fixture(tmp_path, counts_hash=sha256_file(counts / "manifest.json"))
    output = tmp_path / "out"
    args = promoted_train.build_parser().parse_args(
        [
            "--source-bundle-root",
            str(source),
            "--output-dir",
            str(output),
            "--pairwise-model-path",
            str(pairwise),
            "--target-json",
            str(tmp_path / "target.json"),
            "--name-counts-index-root",
            str(counts),
            "--fixed-ranker-bundle",
            str(fixed),
            "--expected-ranker-sha256",
            digest,
            "--n-jobs",
            "1",
            *([] if check == "opt_in" else ["--use-legacy-gate-calibration"]),
        ]
    )
    if check == "legacy_frozen":
        roles = tmp_path / "roles.json"
        roles.write_text(json.dumps(assign_block_roles([BlockSpec("a ross", 1, frozenset())], {"fit_pool": 1})))
        args.use_legacy_gate_calibration = False
        args.linker_task_manifest = tmp_path / "frozen.json"
        args.block_role_assignments = roles

        def reject_task_read(*args, **kwargs):
            pytest.fail("contract-1 ranker reached frozen task materialization")

        monkeypatch.setattr(promoted_train, "_materialize_frozen_linker_tasks", reject_task_read)
        with pytest.raises(ValueError, match="Contract-1 fixed ranker requires legacy calibration"):
            promoted_train.run(args)
        assert not output.exists()
        return
    if check == "opt_in":
        try:
            with pytest.raises(ValueError, match="requires --use-legacy-gate-calibration"):
                promoted_train.run(args)
        finally:
            assert not output.exists()
    elif check == "source_manifest":
        path = fixed / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**payload, "format_version": 999}), encoding="utf-8")
        with pytest.raises(ValueError, match="format_version"):
            promoted_train._load_fixed_ranker(fixed, digest)
    elif check == "contract":
        path = pairwise / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**manifest, "feature_contract_version": 2}), encoding="utf-8")
        with pytest.raises(ValueError, match="Fixed ranker pairwise binding"):
            promoted_train.run(args)
        assert not output.exists()
    else:
        target_path = Path(args.target_json)
        target = json.loads(target_path.read_text(encoding="utf-8"))
        target["params"]["n_estimators"] = 999
        target_path.write_text(json.dumps(target), encoding="utf-8")
        pairwise_year_diffs: list[float] = []
        query_years: list[int] = []
        linker_year_gaps: list[float] = []
        year_indices = FeaturizationInfo(["year_diff"]).selected_feature_indices()
        compute_pairs = promoted_train.compute_candidate_batch_pairwise_model_and_aggregate_stats
        build_rows = promoted_train.build_promoted_non_pairwise_row_features

        def record_pairwise_years(
            dataset: ANDData | None, batch: LinkerCandidateBatch, **kwargs: Any
        ) -> CandidateBatchPairwiseModelResult:
            pairs = list(
                zip(batch.left_signature_indices.tolist(), batch.right_signature_indices.tolist(), strict=True)
            )
            matrix = kwargs["featurizer"].featurize_pairs_matrix_indexed(pairs, year_indices, 1, np.nan)
            pairwise_year_diffs.extend(np.asarray(matrix)[:, 0].tolist())
            return compute_pairs(dataset, batch, **kwargs)

        def record_linker_years(
            batch: LinkerCandidateBatch, row_signals: Mapping[str, Any], *, legacy_paper_years: bool = False
        ) -> dict[str, np.ndarray]:
            features = build_rows(batch, row_signals, legacy_paper_years=legacy_paper_years)
            query_years.extend(row_signals["query_year"].tolist())
            linker_year_gaps.extend(features["year_gap_to_candidate_range"].tolist())
            return features

        monkeypatch.setattr(
            promoted_train, "compute_candidate_batch_pairwise_model_and_aggregate_stats", record_pairwise_years
        )
        monkeypatch.setattr(promoted_train, "build_promoted_non_pairwise_row_features", record_linker_years)
        report = promoted_train.run(args)
        assert set(pairwise_year_diffs) == {20.0}
        assert set(query_years) == {1800}
        assert set(linker_year_gaps) == {20.0}
        training_tables = {summary["table_key"] for summary in report["feature_materialization_summaries"]["training"]}
        assert training_tables == {"classic_gate_source_path", "s2and_eval_path"}
        complete = Path(report["complete_model_path"])
        assert report["evaluation_summary"]["training_summary"]["ranker_refit"] is False
        fixed_booster, _, _ = promoted_train._load_fixed_ranker(fixed, digest)
        assert report["n_estimators"] == report["model_params"]["n_estimators"] == fixed_booster.num_trees()
        assert "training_rows" not in report["observed_metrics"]
        assert report["observed_metrics"]["stratified_test_queries"] == 1
        assert report["query_predictions"]["rows"] == 1
        booster = Path("incremental_linker/booster.lgb")
        assert (complete / booster).read_bytes() == (fixed / booster).read_bytes()
        reloaded = load_production_model(complete).incremental_linker_artifact
        assert reloaded is not None and reloaded.feature_contract_version == 1
        assert report["artifact_summary"]["gate_config"]["training_summary"]["calibration_queries"] == 2


@pytest.mark.parametrize("corrupt", ["digest", "staged_copy"])
def test_fixed_ranker_publication_rechecks_booster_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: str
) -> None:
    import shutil

    from s2and.incremental_linking import artifact as artifact_module
    from s2and.incremental_linking.artifact import save_incremental_linking_artifact
    from tests.promoted_linking_helpers import tiny_logistic_gate_config

    _, fixed, digest = _fixed_ranker_fixture(tmp_path)
    source = fixed / "incremental_linker" / "booster.lgb"
    output = tmp_path / "new-artifact"
    if corrupt == "digest":
        digest = digest[::-1]
    else:
        # Keep the source digest valid and alter only the staged copy, so a check that hashes the source passes.
        real_copyfile = shutil.copyfile

        def copy_then_alter(src: Path, dst: Path) -> None:
            real_copyfile(src, dst)
            with Path(dst).open("ab") as staged:
                staged.write(b"x")

        monkeypatch.setattr(artifact_module.shutil, "copyfile", copy_then_alter)
    with pytest.raises(ValueError, match="SHA-256 changed before publication"):
        save_incremental_linking_artifact(
            None,
            output,
            fixed_booster=(source, digest),
            gate_config=tiny_logistic_gate_config(),
            target_spec={},
            pairwise_bundle_binding={"source": "fixture"},
        )
    assert not output.exists()
