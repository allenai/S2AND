"""Tests for promoted Arrow/Rust feature materialization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking_training.classic import OfficialBundle
from scripts.production.model import train_linker_and_finalize as promoted_train
from scripts.production.model.train_linker_and_finalize import (
    _arrow_row_seed_bypass_mask,
    _clean_arrow_rust_structural_rows,
    _load_target,
    _resolve_arrow_rust_pair_labels,
    _row_label_is_positive,
    _write_arrow_rust_partial_frame,
)
from tests.helpers import write_minimal_arrow_prediction_bundle


def test_load_target_requires_exact_promoted_feature_order(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"
    feature_columns = list(promoted_train.promoted_linker_feature_columns())
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "params": {"n_estimators": 10},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    target = _load_target(target_path)

    assert target["features"] == feature_columns

    reordered_features = feature_columns.copy()
    reordered_features[0], reordered_features[1] = reordered_features[1], reordered_features[0]
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": len(reordered_features),
                "features": reordered_features,
                "params": {"n_estimators": 10},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical order"):
        _load_target(target_path)


def test_load_target_rejects_removed_promoted_features(tmp_path: Path) -> None:
    target_path = tmp_path / "target.json"

    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": 1,
                "features": ["pw_max_email_prefix_equal"],
                "params": {"n_estimators": 10},
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown features"):
        _load_target(target_path)


@pytest.mark.parametrize("params", [{}, {"n_estimators": True}, {"n_estimators": 0}, {"n_estimators": 1.5}])
def test_load_target_rejects_invalid_params(tmp_path: Path, params: dict[str, Any]) -> None:
    feature_columns = list(promoted_train.promoted_linker_feature_columns())
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "params": params,
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="params"):
        _load_target(target_path)


def test_load_target_rejects_nonfinite_metrics(tmp_path: Path) -> None:
    feature_columns = list(promoted_train.promoted_linker_feature_columns())
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "params": {"n_estimators": 10},
                "metrics": {"score": float("nan")},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be finite"):
        _load_target(target_path)


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"score": 1.0}, "unknown metric keys"),
        ({"stratified_test_accuracy": None}, "must be numeric"),
        ({"stratified_test_accuracy": True}, "must be numeric"),
        ({"stratified_test_errors": 1.5}, "nonnegative integer"),
        ({"weighted_average_error_weights": {"wrong": 1.0}}, "must equal"),
    ],
)
def test_load_target_rejects_invalid_metric_schema(
    tmp_path: Path,
    metrics: dict[str, Any],
    message: str,
) -> None:
    feature_columns = list(promoted_train.promoted_linker_feature_columns())
    target_path = tmp_path / "target.json"
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "params": {"n_estimators": 10},
                "metrics": metrics,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load_target(target_path)


def test_load_target_rejects_duplicate_features(tmp_path) -> None:
    target_path = tmp_path / "duplicate_target.json"
    target_path.write_text(
        json.dumps(
            {
                "schema_version": promoted_train.LINKER_TARGET_SCHEMA,
                "feature_count": 2,
                "features": ["min_distance", "min_distance"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate features"):
        _load_target(target_path)


def test_arrow_rust_partial_writer_reuses_label_columns_as_features(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "retrieval_rank": [1.0, 2.0],
            "query_group_id": ["q1", "q1"],
            "label": [1, 0],
        }
    )
    partial_path = tmp_path / "partial.parquet"

    _write_arrow_rust_partial_frame(
        rows=rows,
        row_positions=np.asarray([7, 8], dtype=np.int64),
        partial_path=partial_path,
        dataset_features={
            "retrieval_rank": np.asarray([1.0, 2.0], dtype=np.float32),
            "title_overlap": np.asarray([0.4, 0.1], dtype=np.float32),
        },
        target_features=("retrieval_rank", "title_overlap"),
    )

    out = pd.read_parquet(partial_path)

    assert out.columns.tolist() == ["_row_position", "retrieval_rank", "query_group_id", "label", "title_overlap"]
    assert out["retrieval_rank"].tolist() == [1.0, 2.0]
    assert out["title_overlap"].tolist() == pytest.approx([0.4, 0.1])


def test_fresh_materialization_writes_one_bundle_without_identity_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    labels_path = source_root / "labels" / "train.parquet"
    labels_path.parent.mkdir(parents=True)
    (source_root / "splits").mkdir()
    labels = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "q1",
                "candidate_component_key": "candidate",
                "retrieval_rank": 1,
                "label": 0,
            },
            {
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "q1",
                "candidate_component_key": "unreachable",
                "retrieval_rank": 30,
                "label": 1,
            },
        ]
    )
    labels.to_parquet(labels_path, index=False)
    bundle_payload = {
        "bundle_name": "fresh_source",
        "assets": {
            "featureless_rows": {"files": {"train_path": "labels/train.parquet"}},
        },
        "models": {"classic": {"feature_columns": [], "best_params": {}, "retrieval_top_k": 1}},
        "expected_metrics": {},
    }
    (source_root / "bundle.json").write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_bundle = OfficialBundle(
        root=source_root.resolve(),
        bundle_name="fresh_source",
        assets=bundle_payload["assets"],
        models=bundle_payload["models"],
        expected_metrics={},
    )
    target = {
        "feature_count": 1,
        "features": ["min_distance"],
        "params": {"n_estimators": 1},
        "metrics": {},
    }
    monkeypatch.setattr(
        promoted_train,
        "_clean_arrow_rust_structural_rows",
        lambda **kwargs: (
            kwargs["rows"],
            {"rows_before": len(kwargs["rows"]), "rows_after": len(kwargs["rows"]), "rows_removed": 0},
        ),
    )
    monkeypatch.setattr(
        promoted_train,
        "_build_arrow_rust_dataset_context",
        lambda **_kwargs: cast(Any, SimpleNamespace()),
    )
    monkeypatch.setattr(promoted_train, "_release_arrow_rust_dataset_context", lambda _context: None)

    def fake_materialize(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        assert kwargs["rows"]["candidate_component_key"].tolist() == ["candidate"]
        return (
            {"min_distance": np.asarray([0.5], dtype=np.float32)},
            {"dataset": "toy", "rows": 1, "seconds": 0.0, "mode": "arrow-rust"},
        )

    monkeypatch.setattr(promoted_train, "_materialize_arrow_rust_dataset_rows", fake_materialize)

    dataset_root = source_root / "datasets" / "toy"
    write_minimal_arrow_prediction_bundle(dataset_root)
    with ArrowDataset.open(dataset_root) as arrow_dataset:
        bundle, summaries = promoted_train._materialize_arrow_rust_feature_bundle(  # noqa: SLF001
            source_bundle=source_bundle,
            output_bundle_root=output_root,
            target=target,
            name_tuples=frozenset(),
            clusterer=SimpleNamespace(),
            n_jobs=1,
            total_ram_bytes=1,
            table_keys=("train_path",),
            max_exemplars=4,
            pairwise_model_nan_value=np.nan,
            pairwise_aggregate_nan_value=0.0,
            arrow_datasets={"toy": arrow_dataset},
        )

    output_path = output_root / "features_corrected" / "train.parquet"
    output = pd.read_parquet(output_path)
    assert bundle.root == output_root.resolve()
    assert output["min_distance"].tolist() == pytest.approx([0.5])
    assert summaries[0]["rows"] == 1
    assert summaries[0]["label_filtering"]["retrieval_window"] == {
        "retrieval_top_k": 1,
        "rows_before": 2,
        "rows_after": 1,
        "rows_removed": 1,
    }
    assert list(output_root.rglob("*.materialization.json")) == []


def test_fresh_materialization_rejects_an_existing_output_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "splits").mkdir(parents=True)
    (source_root / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": "source",
                "assets": {},
                "models": {"classic": {}},
                "expected_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("existing output", encoding="utf-8")
    source_bundle = OfficialBundle(source_root, "source", {}, {"classic": {}}, {})

    with pytest.raises(ValueError, match="already exists"):
        promoted_train._copy_bundle_support_files(source_bundle, output_root)  # noqa: SLF001

    assert marker.read_text(encoding="utf-8") == "existing output"


def test_block_local_member_ids_drop_foreign_members() -> None:
    signature_blocks = {"local": "block-a", "foreign-1": "block-b", "foreign-2": "block-b"}

    assert promoted_train._block_local_member_ids_from_signature_blocks(  # noqa: SLF001
        "block-a::mixed",
        ("local", "foreign-1"),
        signature_blocks,
    ) == ("local",)
    assert (
        promoted_train._block_local_member_ids_from_signature_blocks(  # noqa: SLF001
            "block-a::foreign",
            ("foreign-1", "foreign-2"),
            signature_blocks,
        )
        == ()
    )


def test_arrow_rust_structural_cleaning_drops_all_foreign_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    pd.DataFrame(
        [
            {"candidate_component_key": "block-a::foreign", "member_index": 0, "signature_id": "f1"},
            {"candidate_component_key": "block-a::foreign", "member_index": 1, "signature_id": "f2"},
        ]
    ).to_parquet(components_dir / "toy_members.parquet", index=False)
    bundle = OfficialBundle(
        root=tmp_path,
        bundle_name="toy",
        assets={"candidate_members": {"datasets": {"toy": "components/toy_members.parquet"}}},
        models={},
        expected_metrics={},
    )
    rows = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "query_group_id": "q1:full",
                "query_signature_id": "q1",
                "candidate_component_key": "block-a::foreign",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(
        promoted_train,
        "_load_arrow_signature_blocks",
        lambda *_args, **_kwargs: {"q1": "block-a", "f1": "block-b", "f2": "block-b"},
    )

    dataset_root = tmp_path / "datasets" / "toy"
    write_minimal_arrow_prediction_bundle(dataset_root)
    with ArrowDataset.open(dataset_root) as arrow_dataset:
        cleaned, summary = _clean_arrow_rust_structural_rows(
            source_bundle=bundle,
            table_key="train_path",
            rows=rows,
            component_membership_cache={},
            arrow_datasets={"toy": arrow_dataset},
        )

    assert cleaned.empty
    assert summary["rows_removed"] == 1
    assert summary["positive_rows_removed"] == 1


def test_arrow_rust_structural_cleaning_drops_self_only_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    pd.DataFrame(
        [
            {"candidate_component_key": "toy block::self", "member_index": 0, "signature_id": "q1"},
            {"candidate_component_key": "toy block::with_neighbor", "member_index": 0, "signature_id": "q1"},
            {"candidate_component_key": "toy block::with_neighbor", "member_index": 1, "signature_id": "n1"},
            {"candidate_component_key": "toy block::other", "member_index": 0, "signature_id": "n2"},
            {"candidate_component_key": "plain_self", "member_index": 0, "signature_id": "q3"},
        ]
    ).to_parquet(components_dir / "toy_members.parquet", index=False)
    bundle = OfficialBundle(
        root=tmp_path,
        bundle_name="toy",
        assets={
            "candidate_members": {"datasets": {"toy": "components/toy_members.parquet"}},
        },
        models={},
        expected_metrics={},
    )
    rows = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "query_group_id": "q1:full",
                "query_signature_id": "q1",
                "candidate_component_key": "toy block::self",
                "label": 1,
            },
            {
                "dataset": "toy",
                "query_group_id": "q1:full",
                "query_signature_id": "q1",
                "candidate_component_key": "toy block::with_neighbor",
                "label": 0,
            },
            {
                "dataset": "toy",
                "query_group_id": "q2:full",
                "query_signature_id": "q2",
                "candidate_component_key": "toy block::other",
                "label": 1,
            },
            {
                "dataset": "toy",
                "query_group_id": "q3:full",
                "query_signature_id": "q3",
                "candidate_component_key": "plain_self",
                "label": 0,
            },
        ]
    )

    monkeypatch.setattr(
        promoted_train,
        "_load_arrow_signature_blocks",
        lambda *_args, **_kwargs: {
            "q1": "toy block",
            "n1": "toy block",
            "q2": "toy block",
            "n2": "toy block",
            "q3": "toy block",
        },
    )

    dataset_root = tmp_path / "datasets" / "toy"
    write_minimal_arrow_prediction_bundle(dataset_root)
    with ArrowDataset.open(dataset_root) as arrow_dataset:
        cleaned, summary = _clean_arrow_rust_structural_rows(
            source_bundle=bundle,
            table_key="train_path",
            rows=rows,
            component_membership_cache={},
            arrow_datasets={"toy": arrow_dataset},
        )

    assert cleaned["candidate_component_key"].tolist() == ["toy block::with_neighbor", "toy block::other"]
    assert summary["rows_removed"] == 2
    assert summary["positive_rows_removed"] == 1
    assert summary["negative_rows_removed"] == 1
    assert summary["queries_removed"] == 1
    assert summary["positive_queries_changed_or_removed"] == 1


def test_arrow_rust_positive_label_marks_training_disallow_ignore() -> None:
    assert _row_label_is_positive(SimpleNamespace(label=1))
    assert _row_label_is_positive(SimpleNamespace(label=1.0))
    assert not _row_label_is_positive(SimpleNamespace(label=0))
    assert not _row_label_is_positive(SimpleNamespace(label=np.nan))


def test_arrow_rust_row_seed_bypass_uses_manifest_seed_constraints() -> None:
    rows = pd.DataFrame(
        [
            {
                "query_signature_id": "q",
                "candidate_component_key": "c_match",
                "split": "train",
                "query_in_seed_before_holdout": 0,
            },
            {
                "query_signature_id": "q",
                "candidate_component_key": "c_other",
                "split": "train",
                "query_in_seed_before_holdout": 0,
            },
        ]
    )

    mask = _arrow_row_seed_bypass_mask(
        rows,
        {"c_match": ("q", "m1"), "c_other": ("other",)},
        cluster_seeds_require={"q": "seed_cluster", "m1": "seed_cluster", "other": "different"},
        cluster_seeds_disallow=frozenset(),
        seed_constrained_signature_ids=frozenset({"q", "m1", "other"}),
    )

    np.testing.assert_array_equal(mask, np.asarray([True, False]))


def test_arrow_rust_pair_label_resolution_applies_seed_bypass_and_disallow_ignore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_constraints(
        left: np.ndarray,
        right: np.ndarray,
        *,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: int,
        featurizer: Any,
        suppress_orcid: bool,
    ) -> np.ndarray:
        assert dont_merge_cluster_seeds is True
        assert num_threads == 2
        assert featurizer == "featurizer"
        assert suppress_orcid is True
        calls.append(
            {
                "left": np.asarray(left).tolist(),
                "right": np.asarray(right).tolist(),
                "seed_bypass": bool(incremental_dont_use_cluster_seeds),
            }
        )
        if incremental_dont_use_cluster_seeds:
            return np.asarray([-90_000.0, -90_000.0], dtype=np.float64)
        return np.asarray([np.nan, -100_000.0, -90_000.0], dtype=np.float64)

    monkeypatch.setattr(promoted_train, "get_constraint_labels_index_arrays_rust", fake_constraints)
    batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([0, 0, 0], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        row_query_signature_indices=np.asarray([0, 0, 0], dtype=np.uint32),
    )

    labels, summary = _resolve_arrow_rust_pair_labels(
        clusterer=SimpleNamespace(use_default_constraints_as_supervision=True),
        batch=batch,
        featurizer="featurizer",
        n_jobs=2,
        pair_seed_bypass=np.asarray([False, True, True]),
        pair_ignore_disallow=np.asarray([False, True, False]),
    )

    assert calls == [
        {"left": [0, 0, 0], "right": [1, 2, 3], "seed_bypass": False},
        {"left": [0, 0], "right": [2, 3], "seed_bypass": True},
    ]
    np.testing.assert_array_equal(np.isnan(labels), np.asarray([True, True, False]))
    assert labels[2] == pytest.approx(-90_000.0)
    assert summary["constraint_seed_bypass_pair_count"] == 2
    assert summary["constraint_seed_bypass_batch_calls"] == 1
    assert summary["constraint_disallow_ignored_pair_count"] == 1
