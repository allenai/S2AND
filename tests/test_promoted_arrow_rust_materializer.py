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
    _write_arrow_rust_partial_frame,
)
from tests.helpers import write_minimal_arrow_prediction_bundle


@pytest.fixture
def target_payload() -> dict[str, Any]:
    """Return the smallest valid promoted training target."""
    features = list(promoted_train.promoted_linker_feature_columns())
    return {"feature_count": len(features), "features": features, "params": {"n_estimators": 10}, "metrics": {}}


def test_load_target_requires_exact_promoted_feature_order(tmp_path: Path, target_payload) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target_payload), encoding="utf-8")
    assert _load_target(target_path)["features"] == target_payload["features"]
    target_payload["features"][0], target_payload["features"][1] = (
        target_payload["features"][1],
        target_payload["features"][0],
    )
    target_path.write_text(json.dumps(target_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical order"):
        _load_target(target_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"feature_count": 1, "features": ["not_a_promoted_feature"]}, "unknown features"),
        ({"feature_count": 2, "features": ["min_distance", "min_distance"]}, "duplicate features"),
        ({"params": {}}, "params"),
        ({"params": {"n_estimators": True}}, "params"),
        ({"params": {"n_estimators": 0}}, "params"),
        ({"params": {"n_estimators": 1.5}}, "params"),
        ({"metrics": {"score": float("nan")}}, "must be finite"),
        ({"metrics": {"score": 1.0}}, "unknown metric keys"),
        ({"metrics": {"stratified_test_accuracy": None}}, "must be numeric"),
        ({"metrics": {"stratified_test_accuracy": True}}, "must be numeric"),
        ({"metrics": {"stratified_test_errors": 1.5}}, "nonnegative integer"),
        ({"metrics": {"weighted_average_error_weights": {"wrong": 1.0}}}, "must equal"),
    ],
)
def test_load_target_rejects_invalid_contract(tmp_path: Path, target_payload, overrides, message: str) -> None:
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target_payload | overrides), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_target(target_path)


def test_arrow_rust_partial_writer_replaces_label_columns_with_materialized_features(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "retrieval_rank": [2.0, 1.0],
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


def test_current_retrieval_window_uses_native_ranks_instead_of_source_ranks() -> None:
    rows = pd.DataFrame(
        {
            "candidate_component_key": ["old-top", "current-top"],
            "retrieval_rank": [1, 30],
        }
    )

    selected, source_positions = promoted_train._rows_in_current_retrieval_window(  # noqa: SLF001
        rows,
        {"retrieval_ranks": np.asarray([2, 1], dtype=np.uint16)},
        retrieval_top_k=1,
        context="test",
    )

    assert source_positions.tolist() == [1]
    assert selected["candidate_component_key"].tolist() == ["current-top"]
    assert selected["retrieval_rank"].tolist() == [1]


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
        "feature_count": 2,
        "features": ["retrieval_rank", "min_distance"],
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

    def fake_materialize(**kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
        assert kwargs["rows"]["candidate_component_key"].tolist() == ["candidate", "unreachable"]
        return (
            {
                "retrieval_rank": np.asarray([1.0], dtype=np.float32),
                "min_distance": np.asarray([0.7], dtype=np.float32),
            },
            {"dataset": "toy", "rows": 1, "seconds": 0.0, "mode": "arrow-rust"},
            np.asarray([1], dtype=np.int64),
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
    assert output["candidate_component_key"].tolist() == ["unreachable"]
    assert output["retrieval_rank"].tolist() == [1]
    assert output["min_distance"].tolist() == pytest.approx([0.7])
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


def test_arrow_rust_structural_cleaning_preserves_only_nonself_local_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real membership table covers self-only, mixed, unrelated, and foreign components."""
    members = {
        "toy block::self": ["q1"],
        "toy block::with_neighbor": ["q1", "n1"],
        "toy block::other": ["n2"],
        "plain_self": ["q3"],
        "toy block::foreign": ["f1", "f2"],
    }
    components = tmp_path / "components"
    components.mkdir()
    pd.DataFrame(
        [
            {"candidate_component_key": component, "member_index": index, "signature_id": sid}
            for component, signature_ids in members.items()
            for index, sid in enumerate(signature_ids)
        ]
    ).to_parquet(components / "toy_members.parquet", index=False)
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
                "query_group_id": f"{query}:full",
                "query_signature_id": query,
                "candidate_component_key": component,
                "label": label,
            }
            for query, component, label in [
                ("q1", "toy block::self", 1),
                ("q1", "toy block::with_neighbor", 0),
                ("q2", "toy block::other", 1),
                ("q3", "plain_self", 0),
                ("q4", "toy block::foreign", 1),
            ]
        ]
    )
    monkeypatch.setattr(
        promoted_train,
        "_load_arrow_signature_blocks",
        lambda *_args, **_kwargs: (
            dict.fromkeys(["q1", "q2", "q3", "q4", "n1", "n2"], "toy block") | {"f1": "foreign", "f2": "foreign"}
        ),
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
    assert summary["rows_removed"] == 3
    assert summary["positive_rows_removed"] == 2
    assert summary["negative_rows_removed"] == 1
    assert summary["queries_removed"] == 2
    assert summary["positive_queries_changed_or_removed"] == 2


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
        cluster_seeds_require={"q": "seed_cluster", "m1": "seed_cluster", "other": "different"},
    )

    np.testing.assert_array_equal(mask, np.asarray([True, True]))


@pytest.mark.parametrize(
    ("query_id", "expected"),
    [
        ("q", True),
        ("unseeded", False),
    ],
    ids=["input-seeded-query", "unseeded-query-with-loo-annotation"],
)
def test_arrow_rust_row_seed_bypass_requires_input_membership(
    query_id: str,
    expected: bool,
) -> None:
    rows = pd.DataFrame(
        [
            {
                "query_signature_id": query_id,
                "candidate_component_key": "c",
                "split": "loo",
                "query_in_seed_before_holdout": 1,
                "label": 1,
            }
        ]
    )
    mask = _arrow_row_seed_bypass_mask(
        rows,
        cluster_seeds_require={"q": "seed", "neighbor": "other_seed"},
    )
    np.testing.assert_array_equal(mask, np.asarray([expected]))


def test_arrow_rust_pair_label_resolution_applies_seed_bypass_without_erasing_disallows(
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
    )

    assert calls == [
        {"left": [0, 0, 0], "right": [1, 2, 3], "seed_bypass": False},
        {"left": [0, 0], "right": [2, 3], "seed_bypass": True},
    ]
    np.testing.assert_array_equal(np.isnan(labels), np.asarray([True, False, False]))
    assert labels[1] == pytest.approx(-90_000.0)
    assert labels[2] == pytest.approx(-90_000.0)
    assert summary["constraint_seed_bypass_pair_count"] == 2
    assert summary["constraint_seed_bypass_batch_calls"] == 1
