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
    "--source-bundle-root",
    "source-bundle",
    "--output-dir",
    "linker-run",
    "--pairwise-model-path",
    "pairwise-stage",
    "--target-json",
    "target.json",
)
FULL_MATERIALIZATION_SUMMARIES = tuple({"table_key": key, "rows": 1} for key in promoted_train.REQUIRED_TABLE_KEYS)


def _stub_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        promoted_train,
        "_preflight_source_rows",
        lambda *_args, **_kwargs: ({"total_selected_rows": 4}, {}),
    )
    monkeypatch.setattr(promoted_train, "pairwise_bundle_binding", lambda _path: {"test": "binding"})
    monkeypatch.setattr(
        promoted_train,
        "_validate_source_bundle_support_files",
        lambda *_args, **_kwargs: ["test-support"],
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


def test_linker_modes_reject_ignored_actions_and_allow_selector_preflight() -> None:
    parser = promoted_train.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*REQUIRED_TRAINING_ARGS, "--preflight-only", "--run-full"])

    for extra_args, message in (
        (["--preflight-only", "--materialize-only"], "mutually exclusive"),
        (["--materialize-only", "--publish-to", "production_model_v9.9"], "cannot be combined"),
        (["--materialize-only", "--allow-metric-drift"], "requires a full"),
    ):
        args = parser.parse_args([*REQUIRED_TRAINING_ARGS, *extra_args])
        with pytest.raises(SystemExit, match=message):
            promoted_train._validate_run_mode(args)  # noqa: SLF001

    selector_preflight = parser.parse_args(
        [*REQUIRED_TRAINING_ARGS, "--preflight-only", "--datasets", "qian", "--limit-rows", "10"]
    )
    promoted_train._validate_run_mode(selector_preflight)  # noqa: SLF001


@pytest.mark.parametrize("selector", ["--tables", "--datasets"])
def test_selectors_require_at_least_one_value(selector: str) -> None:
    with pytest.raises(SystemExit):
        promoted_train.build_parser().parse_args([*REQUIRED_TRAINING_ARGS, selector])


def test_source_preflight_rejects_unknown_selectors_and_reports_valid_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    labels_path = source_root / "labels" / "train.parquet"
    labels_path.parent.mkdir(parents=True)
    pd.DataFrame({"dataset": ["qian", "qian"]}).to_parquet(labels_path, index=False)
    bundle = promoted_train.OfficialBundle(
        root=source_root.resolve(),
        bundle_name="preflight",
        assets={"featureless_rows": {"files": {"train_path": "labels/train.parquet"}}},
        models={},
        expected_metrics={},
    )
    arrow_paths = ValidatedArrowInputs(
        paths={},
        generation_id="fixture-generation",
        normalization_version=promoted_train.NORMALIZATION_VERSION,
    )
    monkeypatch.setattr(promoted_train, "_arrow_paths_for_dataset", lambda *_args, **_kwargs: arrow_paths)

    with pytest.raises(ValueError, match="unknown --tables"):
        promoted_train._preflight_source_rows(  # noqa: SLF001
            bundle,
            table_keys=("unknown",),
            datasets=None,
            limit_rows=1,
            require_full_tables=False,
            name_counts_index_root=None,
        )
    with pytest.raises(ValueError, match="selected zero rows"):
        promoted_train._preflight_source_rows(  # noqa: SLF001
            bundle,
            table_keys=("train_path",),
            datasets={"missing"},
            limit_rows=1,
            require_full_tables=False,
            name_counts_index_root=None,
        )

    summary, resolved_paths = promoted_train._preflight_source_rows(  # noqa: SLF001
        bundle,
        table_keys=("train_path",),
        datasets={"qian"},
        limit_rows=1,
        require_full_tables=False,
        name_counts_index_root=None,
    )

    assert summary["total_selected_rows"] == 1
    assert summary["selected_tables"] == ["train_path"]
    assert resolved_paths == {"qian": arrow_paths}


def test_limited_source_preflight_validates_only_datasets_that_will_be_materialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    labels_path = source_root / "labels" / "train.parquet"
    labels_path.parent.mkdir(parents=True)
    pd.DataFrame({"dataset": ["qian", "pubmed"]}).to_parquet(labels_path, index=False)
    bundle = promoted_train.OfficialBundle(
        root=source_root.resolve(),
        bundle_name="preflight",
        assets={"featureless_rows": {"files": {"train_path": "labels/train.parquet"}}},
        models={},
        expected_metrics={},
    )
    qian_paths = ValidatedArrowInputs(
        paths={},
        generation_id="qian-generation",
        normalization_version=promoted_train.NORMALIZATION_VERSION,
    )

    def resolve_paths(_bundle: Any, dataset_name: str, **_kwargs: Any) -> ValidatedArrowInputs:
        assert dataset_name == "qian"
        return qian_paths

    monkeypatch.setattr(promoted_train, "_arrow_paths_for_dataset", resolve_paths)

    summary, resolved_paths = promoted_train._preflight_source_rows(  # noqa: SLF001
        bundle,
        table_keys=("train_path",),
        datasets={"qian", "pubmed"},
        limit_rows=1,
        require_full_tables=False,
        name_counts_index_root=None,
    )

    assert summary["tables"][0]["datasets"] == ["qian"]
    assert resolved_paths == {"qian": qian_paths}


def test_limited_parquet_read_does_not_call_unbounded_pandas_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame(
        {
            "dataset": ["qian", "pubmed", "qian"],
            "value": [1, 2, 3],
        }
    ).to_parquet(path, index=False)
    monkeypatch.setattr(
        promoted_train.pd,
        "read_parquet",
        lambda *_args, **_kwargs: pytest.fail("bounded read used pd.read_parquet"),
    )

    rows = promoted_train._read_selected_parquet_rows(  # noqa: SLF001
        path,
        datasets={"qian"},
        limit_rows=1,
    )

    assert rows.to_dict("records") == [{"dataset": "qian", "value": 1}]


def test_output_paths_cannot_be_nested_under_input_bundles(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    pairwise_root = tmp_path / "pairwise"

    with pytest.raises(SystemExit, match="immutable input bundles"):
        promoted_train._assert_output_paths_outside_inputs(  # noqa: SLF001
            output_dir=pairwise_root / "linker-run",
            publish_dir=source_root / "production_model_v9.9",
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
        models={},
        expected_metrics={},
    )

    with pytest.raises(ValueError, match="contains no files"):
        promoted_train._validate_source_bundle_support_files(bundle)  # noqa: SLF001

    (tmp_path / "splits" / "placeholder.csv").write_text("unused\n", encoding="utf-8")
    with pytest.raises(ValueError, match="classic_gate_internal_eval_base_groups_path"):
        promoted_train._validate_source_bundle_support_files(  # noqa: SLF001
            bundle,
            require_training_contract=True,
        )


def test_source_support_preflight_rejects_leaky_split_contract(tmp_path: Path) -> None:
    (tmp_path / "bundle.json").write_text("{}", encoding="utf-8")
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    assignments_path = splits_dir / "assignments.csv"
    pd.DataFrame(
        [
            {"query_group_id": "q1", "source_key": "s", "base_group_id": "b1", "split": "calibration_fit"},
            {"query_group_id": "q2", "source_key": "s", "base_group_id": "b1", "split": "test"},
        ]
    ).to_csv(assignments_path, index=False)
    classic = {
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


def test_preflight_only_does_not_create_output_or_materialize(
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
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(promoted_train, "_assert_pairwise_model_supports_arrow_materialization", lambda *_args: None)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: pytest.fail("preflight-only must not materialize features"),
    )
    output_dir = tmp_path / "output"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--output-dir",
            str(output_dir),
            "--preflight-only",
        ]
    )

    result = promoted_train.run(args)

    assert result["mode"] == "preflight"
    assert result["source"]["total_selected_rows"] == 4
    assert not output_dir.exists()


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
    arrow_paths = ValidatedArrowInputs(
        paths={},
        generation_id="fixture-generation",
        normalization_version=promoted_train.NORMALIZATION_VERSION,
    )
    clusterer = SimpleNamespace()
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        promoted_train,
        "_preflight_source_rows",
        lambda *_args, **_kwargs: ({"total_selected_rows": 1}, {"toy": arrow_paths}),
    )
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: clusterer)
    monkeypatch.setattr(promoted_train, "_assert_pairwise_model_supports_arrow_materialization", lambda *_args: None)

    def reject_binding(actual_clusterer: Any, actual_paths: Any, *, context: str) -> None:
        assert actual_clusterer is clusterer
        assert actual_paths is arrow_paths
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
            *REQUIRED_TRAINING_ARGS,
            "--output-dir",
            str(output_dir),
            "--preflight-only",
        ]
    )

    with pytest.raises(ValueError, match="name-count generation mismatch"):
        promoted_train.run(args)

    assert not output_dir.exists()


def test_materialization_summary_must_be_nonempty_and_complete() -> None:
    with pytest.raises(RuntimeError, match="produced zero rows"):
        promoted_train._assert_materialization_nonempty([], require_full_tables=False)  # noqa: SLF001
    with pytest.raises(RuntimeError, match="missing or empty required tables"):
        promoted_train._assert_materialization_nonempty(  # noqa: SLF001
            [{"table_key": "train_path", "rows": 1}],
            require_full_tables=True,
        )


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


def test_publish_destination_must_be_fresh_and_separate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    existing_publish = tmp_path / "production_model_v9.9"
    existing_publish.mkdir()
    parser = promoted_train.build_parser()

    with pytest.raises(SystemExit, match="must name a new directory"):
        promoted_train._resolved_output_paths(  # noqa: SLF001
            parser.parse_args([*REQUIRED_TRAINING_ARGS, "--publish-to", str(existing_publish)])
        )

    nested_publish = output_dir / "production_model_v9.9"
    with pytest.raises(SystemExit, match="separate directories"):
        promoted_train._resolved_output_paths(  # noqa: SLF001
            parser.parse_args(
                [
                    *REQUIRED_TRAINING_ARGS,
                    "--output-dir",
                    str(output_dir),
                    "--publish-to",
                    str(nested_publish),
                ]
            )
        )


def test_publish_version_must_match_pairwise_before_materialization(
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
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        promoted_train,
        "load_clusterer",
        lambda *_args, **_kwargs: SimpleNamespace(production_model_bundle_version="9.9"),
    )
    monkeypatch.setattr(promoted_train, "_assert_pairwise_model_supports_arrow_materialization", lambda *_args: None)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: pytest.fail("version mismatch reached materialization"),
    )
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--run-full",
            "--output-dir",
            str(tmp_path / "run"),
            "--publish-to",
            str(tmp_path / "production_model_v8.8"),
        ]
    )

    with pytest.raises(SystemExit, match="version disagrees with the pairwise bundle"):
        promoted_train.run(args)


def test_empty_target_metrics_are_rejected_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = {
        "features": ["f0"],
        "feature_count": 1,
        "params": {"n_estimators": 1},
        "metrics": {},
    }
    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(
        promoted_train,
        "load_bundle",
        lambda _path: pytest.fail("bundle loading ran before target metric validation"),
    )
    output_dir = tmp_path / "must_not_be_created"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--run-full",
            "--output-dir",
            str(output_dir),
        ]
    )

    with pytest.raises(ValueError, match="must not be empty without --allow-metric-drift"):
        promoted_train.run(args)

    assert not output_dir.exists()


def test_partial_target_metrics_are_rejected_when_loaded(tmp_path: Path) -> None:
    target_path = tmp_path / "partial-target.json"
    features = list(promoted_train.promoted_linker_feature_columns())
    target_path.write_text(
        json.dumps(
            {
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


def test_metric_drift_override_cannot_promote_before_loading_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(_path: Path) -> dict[str, Any]:
        raise AssertionError("promotion validation must run before loading the target")

    monkeypatch.setattr(promoted_train, "_load_target", fail_load)  # noqa: SLF001
    output_dir = tmp_path / "must_not_be_created"
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--allow-metric-drift",
            "--publish-to",
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


def test_observed_metrics_use_weighted_average_error() -> None:
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
    assert promoted_train._metric_deltas(  # noqa: SLF001
        {"weighted_average_error_weights": observed["weighted_average_error_weights"]},
        {"metrics": {"weighted_average_error_weights": observed["weighted_average_error_weights"]}},
    ) == {"weighted_average_error_weights": True}


def test_metric_gate_requires_complete_finite_official_metrics() -> None:
    observed = promoted_train._observed_official_metrics(  # noqa: SLF001
        {
            "training_summary": {"rows": 10, "positive_rows": 3},
            "stratified_eval_test_split": {
                "overall": {
                    "test": {
                        "n_queries": 10,
                        "accuracy": 0.9,
                        "balanced_accuracy": 0.8,
                        "error_rate": 0.1,
                        "errors": 1,
                        "false_abstain": 1,
                        "false_link": 0,
                        "wrong_candidate_link": 0,
                    }
                }
            },
        }
    )
    target = {"metrics": dict(observed)}
    promoted_train._assert_no_metric_drift(observed, target)  # noqa: SLF001

    incomplete_target = {"metrics": dict(observed)}
    incomplete_target["metrics"].pop("stratified_test_error_rate")
    with pytest.raises(RuntimeError, match="complete official metric set"):
        promoted_train._assert_no_metric_drift(observed, incomplete_target)  # noqa: SLF001

    incomplete_observed = dict(observed)
    incomplete_observed.pop("stratified_test_error_rate")
    with pytest.raises(RuntimeError, match="unexpected metric key set"):
        promoted_train._assert_no_metric_drift(incomplete_observed, target)  # noqa: SLF001

    nonfinite_target = {"metrics": {**observed, "stratified_test_error_rate": float("nan")}}
    with pytest.raises(RuntimeError, match="must be finite"):
        promoted_train._assert_no_metric_drift(observed, nonfinite_target)  # noqa: SLF001


@pytest.mark.parametrize("allow_metric_drift", [False, True])
def test_metric_gate_prevents_unapproved_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    allow_metric_drift: bool,
) -> None:
    target = {
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
    target["metrics"] = promoted_train._observed_official_metrics(summary)  # noqa: SLF001
    target["metrics"]["stratified_test_errors"] = 0
    promoted = False

    def fail_if_promoted(**_kwargs: Any) -> dict[str, Any]:
        nonlocal promoted
        promoted = True
        raise AssertionError("artifact promotion ran before metric gate")

    monkeypatch.setattr(promoted_train, "_load_target", lambda _path: target)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_bundle", lambda _path: bundle)
    _stub_preflight(monkeypatch)
    monkeypatch.setattr(
        promoted_train,
        "_materialize_arrow_rust_feature_bundle",
        lambda **_kwargs: (bundle, list(FULL_MATERIALIZATION_SUMMARIES)),
    )
    monkeypatch.setattr(
        promoted_train,
        "_assert_pairwise_model_supports_arrow_materialization",
        lambda *_args: None,
    )
    fitted = promoted_train.FittedClassicRun(
        summary=summary,
        model=object(),
        gate_config={"model_type": "test"},
        retrieval_top_k=25,
    )
    monkeypatch.setattr(promoted_train, "run_classic", lambda *_args, **_kwargs: fitted)
    monkeypatch.setattr(promoted_train, "_save_evaluated_artifact", fail_if_promoted)  # noqa: SLF001
    monkeypatch.setattr(promoted_train, "load_clusterer", lambda *_args, **_kwargs: SimpleNamespace())
    args = promoted_train.build_parser().parse_args(
        [
            *REQUIRED_TRAINING_ARGS,
            "--run-full",
            "--output-dir",
            str(tmp_path / "out"),
            *(["--allow-metric-drift"] if allow_metric_drift else []),
        ]
    )

    if allow_metric_drift:
        result = promoted_train.run(args)
        assert Path(result["candidate_target_path"]).is_file()
        assert "artifact_dir" not in result
    else:
        with pytest.raises(RuntimeError, match="drifted from target metrics"):
            promoted_train.run(args)

    assert not promoted
    assert not (tmp_path / "out" / promoted_train.EVALUATED_ARTIFACT_DIRNAME).exists()


def test_saved_artifact_uses_exact_evaluated_model_and_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = object()
    gate_config = {"model_type": "evaluated-gate"}
    fitted = promoted_train.FittedClassicRun(
        summary={},
        model=model,
        gate_config=gate_config,
        retrieval_top_k=17,
    )
    captured: dict[str, Any] = {}

    def save(actual_model: Any, artifact_dir: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(model=actual_model, artifact_dir=artifact_dir, kwargs=kwargs)
        return {
            "schema_version": "test",
            "booster_sha256": "a" * 64,
            "target_spec_digest": "b" * 64,
        }

    monkeypatch.setattr(promoted_train, "save_incremental_linking_artifact", save)
    monkeypatch.setattr(
        promoted_train,
        "load_incremental_linking_artifact",
        lambda _path: SimpleNamespace(retrieval_top_k=17),
    )
    artifact_dir = tmp_path / promoted_train.EVALUATED_ARTIFACT_DIRNAME

    summary = promoted_train._save_evaluated_artifact(  # noqa: SLF001
        fitted=fitted,
        artifact_dir=artifact_dir,
        target_spec={"features": ["f0"]},
        artifact_pairwise_bundle_binding={"manifest": "digest"},
    )

    assert captured["model"] is model
    assert captured["artifact_dir"] == artifact_dir
    assert captured["kwargs"]["gate_config"] is gate_config
    assert captured["kwargs"]["retrieval_top_k"] == 17
    assert summary["booster_sha256"] == "a" * 64
