from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking_training import classic as classic_training
from scripts.production.model import train_linker_and_finalize as promoted_train
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, write_minimal_arrow_prediction_bundle

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
)
EXPLICIT_ARTIFACT_HASHES = {
    "name_counts_manifest_sha256": "a" * 64,
    "name_tuples_data_sha256": "b" * 64,
    "orcid_prefix_counts_data_sha256": "c" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_MATERIALIZATION_SUMMARIES = tuple({"table_key": key, "rows": 1} for key in promoted_train.REQUIRED_TABLE_KEYS)


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
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
        "_preflight_source_rows",
        lambda *_args, **_kwargs: (4, {}),
    )
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path, **_kwargs: {"test": "binding"})
    monkeypatch.setattr(
        promoted_train,
        "_validate_source_bundle_support_files",
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


def test_canonical_cli_help_and_import_leave_backend_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    help_result = subprocess.run(
        [sys.executable, "scripts/production/model/train_linker_and_finalize.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--pairwise-model-path" in help_result.stdout

    with pytest.raises(SystemExit) as help_exit:
        promoted_train.build_parser().parse_args(["--help"])
    assert help_exit.value.code == 0
    assert "Explicit v5 pairwise_only native bundle" in capsys.readouterr().out

    for backend in (None, "python"):
        env = os.environ.copy()
        if backend is None:
            env.pop("S2AND_BACKEND", None)
        else:
            env["S2AND_BACKEND"] = backend
        import_result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "import scripts.production.model.train_linker_and_finalize; "
                    "print(os.environ.get('S2AND_BACKEND'))"
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert import_result.returncode == 0, import_result.stderr
        assert import_result.stdout.strip() == str(backend)


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


def test_linker_commands_require_name_counts_root() -> None:
    arguments = list(COMMON_TRAINING_ARGS)
    option_index = arguments.index("--name-counts-index-root")
    del arguments[option_index : option_index + 2]

    with pytest.raises(SystemExit):
        promoted_train.build_parser().parse_args(arguments)


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


def test_source_support_preflight_requires_declared_split_files(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    (tmp_path / "splits").mkdir()
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="support",
        assets={},
        models={"classic": {}},
        expected_metrics={},
    )

    with pytest.raises(ValueError, match="contains no files"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001

    (tmp_path / "splits" / "placeholder.csv").write_text("unused\n", encoding="utf-8")
    with pytest.raises(ValueError, match="classic_gate_internal_eval_base_groups_path"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001


def test_source_support_preflight_rejects_leaky_split_contract(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    assignments_path = splits_dir / "assignments.csv"
    internal_eval_path = splits_dir / "internal_eval.csv"
    pd.DataFrame({"base_group_id": ["internal"]}).to_csv(internal_eval_path, index=False)
    pd.DataFrame(
        [
            {"query_group_id": "q1", "source_key": "s", "base_group_id": "b1", "split": "calibration_fit"},
            {"query_group_id": "q2", "source_key": "s", "base_group_id": "b1", "split": "test"},
        ]
    ).to_csv(assignments_path, index=False)
    classic = {
        "classic_gate_internal_eval_base_groups_path": "splits/internal_eval.csv",
        "stratified_eval_test_split": {"assignments_path": "splits/assignments.csv", "test_split": "test"},
        "promoted_stratified_gate": {
            "calibration_splits": ["calibration_fit"],
            "test_split": "test",
        },
    }
    bundle = promoted_train.OfficialBundle(
        root=tmp_path.resolve(),
        bundle_name="support",
        assets={},
        models={"classic": classic},
        expected_metrics={},
    )

    with pytest.raises(ValueError, match="base_group_id values in multiple splits"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001

    assignments = pd.read_csv(assignments_path)
    assignments.loc[1, "base_group_id"] = "b2"
    assignments.to_csv(assignments_path, index=False)
    classic["promoted_stratified_gate"]["calibration_splits"] = [
        "calibration_fit",
        "calibration_check",
    ]
    with pytest.raises(ValueError, match="omit configured calibration/test splits"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001

    classic["promoted_stratified_gate"]["calibration_splits"] = ["calibration_fit", "test"]
    with pytest.raises(ValueError, match="must not include test_split"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001


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
        [
            {
                "query_group_id": "calibration-query",
                "source_key": "hwang_eval",
                "split": "calibration_fit",
                "base_group_id": "calibration-base",
            },
            {
                "query_group_id": "frozen-test-query",
                "source_key": "s2and_eval",
                "split": "test",
                "base_group_id": "test-base",
            },
        ]
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
    monkeypatch.setattr(promoted_train, "_assert_pairwise_model_supports_arrow_materialization", lambda *_args: None)

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
            "_preflight_source_rows",
            lambda *_args, **_kwargs: (1, {"toy": arrow_dataset}),
        )
        with pytest.raises(ValueError, match="name-count generation mismatch"):
            promoted_train.run(args)

    assert not output_dir.exists()


def test_partial_target_metrics_are_rejected_when_loaded(tmp_path: Path) -> None:
    target_path = tmp_path / "partial-target.json"
    features = list(promoted_train.promoted_linker_feature_columns())
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "features": features,
                "feature_count": len(features),
                "params": {"n_estimators": 1},
                "metrics": {"stratified_test_accuracy": 1.0},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="complete official metric set"):
        promoted_train._load_target(target_path)  # noqa: SLF001


def test_target_schema_is_required_when_loaded(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be"):
        promoted_train._load_target(target_path)  # noqa: SLF001


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

    class CapturingRustModule:
        @staticmethod
        def raw_arrow_labeled_candidate_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
            captured["native_dataset"] = args[0]
            plan = real_rust_module.raw_arrow_labeled_candidate_plan(*args, **kwargs)
            captured["reused_name_counts_index"] = plan["telemetry"]["reused_name_counts_index"]
            return plan

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
    for relpath in source_files.values():
        output_path = output_root / "features_corrected" / Path(relpath).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"query_group_id": Path(relpath).stem}]).to_parquet(output_path, index=False)

    bundle = promoted_train._finalize_arrow_rust_bundle_metadata(  # noqa: SLF001
        source_bundle=promoted_train.load_bundle(source_root),
        output_bundle_root=output_root,
        target={"feature_count": 1, "features": ["f0"], "params": {"n_estimators": 10}, "metrics": {}},
        selected_keys=list(source_files),
    )

    assert bundle.assets["corrected_feature_rows"]["root"] == "features_corrected"
    assert bundle.models["classic"]["s_park_eval_path"] == "features_corrected/s_park.parquet"
    assert bundle.models["classic"]["s_lee_eval_path"] == "features_corrected/s_lee.parquet"
    assert bundle.models["classic"]["extra_eval_paths"] == {
        "j_smith": "features_corrected/j_smith.parquet",
    }
    optional_paths = classic_training._iter_extra_eval_paths(bundle.models["classic"])  # noqa: SLF001
    assert [dataset for dataset, _path in optional_paths] == ["s_park", "s_lee", "j_smith"]
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


def test_observed_metrics_use_weighted_average_error() -> None:
    observed = _valid_observed_metrics()

    promoted_train._validate_observed_official_metrics(observed)  # noqa: SLF001

    assert observed["false_abstain_error_rate"] == pytest.approx(0.04)
    assert observed["false_link_error_rate"] == pytest.approx(0.03)
    assert observed["wrong_link_error_rate"] == pytest.approx(0.03)
    assert observed["weighted_average_error"] == pytest.approx(((0.25 * 0.04) + 0.03 + (1.5 * 0.03)) / 2.75)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_observed_metric_validation_requires_exact_official_keys(change: str) -> None:
    observed = _valid_observed_metrics()
    if change == "missing":
        observed.pop("weighted_average_error")
    else:
        observed["unexpected_metric"] = 0.0

    with pytest.raises(ValueError, match="complete official metric set"):
        promoted_train._validate_observed_official_metrics(observed)  # noqa: SLF001


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stratified_test_accuracy", float("nan"), "must be finite"),
        ("stratified_test_accuracy", 1.1, r"must be in \[0, 1\]"),
        ("stratified_test_errors", -1, "must be a nonnegative integer"),
        ("training_rows", 0, "training counts are inconsistent"),
        ("stratified_test_queries", 0, "test counts are inconsistent"),
        ("stratified_test_errors", 101, "test counts are inconsistent"),
        (
            "weighted_average_error_weights",
            {
                **promoted_train.WEIGHTED_ERROR_WEIGHTS,
                "false_link_error_rate": 0.0,
            },
            "must be positive",
        ),
        (
            "weighted_average_error_weights",
            {
                **promoted_train.WEIGHTED_ERROR_WEIGHTS,
                "false_link_error_rate": 2.0,
            },
            "must equal the official value",
        ),
    ],
)
def test_observed_metric_validation_rejects_malformed_values(
    field: str,
    value: Any,
    message: str,
) -> None:
    observed = _valid_observed_metrics()
    observed[field] = value

    with pytest.raises(ValueError, match=message):
        promoted_train._validate_observed_official_metrics(observed)  # noqa: SLF001


def test_query_prediction_export_is_deterministic(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        [
            {
                "base_group_id": "base-2",
                "chosen_candidate_component_key": "candidate-2",
                "chosen_probability": 0.25,
                "correct": 0,
                "predicted_action": "abstain",
                "query_case_id": "query-2",
                "query_safe_target": 1,
                "query_safe_target_source": "retrieved_window",
                "source_key": "source",
                "split": "test",
            },
            {
                "base_group_id": "base-1",
                "chosen_candidate_component_key": "candidate-1",
                "chosen_probability": 0.75,
                "correct": 1,
                "predicted_action": "link_candidate",
                "query_case_id": "query-1",
                "query_safe_target": 1,
                "query_safe_target_source": "manual_override",
                "source_key": "source",
                "split": "test",
            },
        ]
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


def test_release_serializes_complete_bundle_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = {
        "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 1},
        "metrics": {"stratified_test_errors": 0},
    }
    bundle = promoted_train.OfficialBundle(
        root=(tmp_path / "source").resolve(),
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
    order: list[str] = []

    def save_artifact(model: Any, artifact_dir: Path, **kwargs: Any) -> dict[str, Any]:
        order.append("serialize")
        assert model is calibrated.model
        assert kwargs["target_spec"] == target
        artifact_dir.mkdir()
        (artifact_dir / "booster.lgb").write_bytes(b"calibrated-booster\n")
        (artifact_dir / "metadata.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")
        return {
            "schema_version": "test",
            "booster_sha256": "a" * 64,
            "pairwise_bundle_binding_digest": "b" * 64,
            "target_spec_digest": "c" * 64,
            "retrieval_top_k": 25,
        }

    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)

    def materialize(**_kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        stage = "materialize_training" if "materialize_training" not in order else "materialize_evaluation"
        order.append(stage)
        return bundle, list(FULL_MATERIALIZATION_SUMMARIES)

    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        materialize,
    )
    monkeypatch.setattr(
        promoted_train,
        "_assert_pairwise_model_supports_arrow_materialization",
        lambda *_args: None,
    )
    output_dir = tmp_path / "out"
    query_predictions_bytes = b"query_case_id\nquery-1\n"
    calibrated = classic_training.CalibratedClassicModel(
        model=object(),
        gate_config={"model_type": "test"},
        retrieval_top_k=25,
        training_summary=summary["training_summary"],
        abstain_rule_summary={},
    )

    def fit_classic(_bundle: Any, **_kwargs: Any) -> Any:
        order.append("fit")
        return calibrated

    def finalize_bundle(**kwargs: Any) -> Any:
        order.append("finalize")
        complete_model_dir = Path(kwargs["output_bundle_dir"])
        complete_model_dir.mkdir()
        manifest_path = complete_model_dir / "manifest.json"
        manifest_path.write_text('{"schema_version":"s2and_production_model_bundle_v5"}\n', encoding="utf-8")
        return SimpleNamespace(manifest_path=manifest_path)

    loaded_artifact = SimpleNamespace()

    def load_complete(path: Path, **_kwargs: Any) -> Any:
        order.append("reload")
        assert Path(path).is_dir()
        return SimpleNamespace(incremental_linker_artifact=loaded_artifact)

    def evaluate_classic(_bundle: Any, **kwargs: Any) -> Any:
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

    monkeypatch.setattr(promoted_train, "fit_classic", fit_classic)
    monkeypatch.setattr(promoted_train, "evaluate_classic", evaluate_classic)
    monkeypatch.setattr(promoted_train, "save_incremental_linking_artifact", save_artifact)
    monkeypatch.setattr(promoted_train, "finalize_production_bundle", finalize_bundle)
    monkeypatch.setattr(promoted_train, "load_production_model", load_complete)
    validate_observed_metrics = promoted_train._validate_observed_official_metrics  # noqa: SLF001

    def validate_metrics(metrics: dict[str, Any]) -> None:
        order.append("validate")
        validate_observed_metrics(metrics)

    monkeypatch.setattr(promoted_train, "_validate_observed_official_metrics", validate_metrics)
    monkeypatch.setattr(
        promoted_train,
        "load_clusterer",
        lambda *_args, **_kwargs: SimpleNamespace(production_model_bundle_version="9.9"),
    )
    args = promoted_train.build_parser().parse_args(
        [
            *COMMON_TRAINING_ARGS,
            "--output-dir",
            str(output_dir),
        ]
    )

    result = promoted_train.run(args)
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
    assert result["schema_version"] == promoted_train.LINKER_EVALUATION_REPORT_SCHEMA
    assert result["pairwise_bundle_binding"] == {"test": "binding"}
    assert result["complete_model_path"] == str(output_dir / "pairwise-stage")
    assert result["query_predictions"]["rows"] == 1
    assert set(output_dir.iterdir()) == {
        output_dir / "pairwise-stage",
        output_dir / "linker_evaluation_report.json",
    }
    assert (
        json.loads((output_dir / "linker_evaluation_report.json").read_text(encoding="utf-8"))["query_predictions"]
        == result["query_predictions"]
    )
