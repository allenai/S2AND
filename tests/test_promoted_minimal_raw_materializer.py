from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import s2and.incremental_linking.query_adapter as retrieval
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking_training.classic import OfficialBundle
from s2and.incremental_linking_training.query_support import build_rust_hybrid_centroid_retriever
from scripts.production.model import linker_train_calibrate_eval as promoted_train
from scripts.production.model.linker_train_calibrate_eval import (
    _apply_row_nan_policy,
    _arrow_paths_for_dataset,
    _arrow_row_seed_bypass_mask,
    _clean_minimal_raw_structural_rows,
    _component_member_details_by_key,
    _has_query_seed_connection,
    _load_target,
    _query_first_token_for_prefix,
    _resolve_arrow_rust_pair_labels,
    _resolve_candidate_batch_pair_labels,
    _row_allows_seed_constraint_bypass,
    _row_label_is_positive,
    _score_candidate_summaries_with_frozen_rust_policy,
    _write_minimal_raw_partial_frame,
)
from tests.helpers import import_s2and_rust, patch_tiny_name_counts_loader


class _ConstraintClusterer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[tuple[str, str], ...], bool]] = []

    def _resolve_constraint_batch(
        self,
        _dataset,
        pairs,
        *,
        partial_supervision,
        runtime_context,
        incremental_dont_use_cluster_seeds,
        constraint_backend,
    ):
        assert partial_supervision == {}
        assert runtime_context is None
        assert constraint_backend is None
        self.calls.append(
            (
                tuple((str(left), str(right)) for left, right in pairs),
                bool(incremental_dont_use_cluster_seeds),
            )
        )
        return [-90_000.0 for _pair in pairs], {}


def test_load_target_accepts_current_supported_promoted_features(tmp_path) -> None:
    target_path = tmp_path / "current_target.json"
    target_path.write_text(
        json.dumps(
            {
                "feature_count": 3,
                "features": ["min_distance", "pw_max_affiliation_overlap", "strong_positive_anchor_score"],
            }
        ),
        encoding="utf-8",
    )

    target = _load_target(target_path)

    assert target["features"] == ["min_distance", "pw_max_affiliation_overlap", "strong_positive_anchor_score"]


def test_load_target_rejects_removed_promoted_features(tmp_path) -> None:
    target_path = tmp_path / "unsupported_target.json"
    target_path.write_text(
        json.dumps(
            {
                "feature_count": 1,
                "features": ["pw_max_email_prefix_equal"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown features"):
        _load_target(target_path)


def test_semantic_row_nan_policy_marks_undefined_non_pairwise_features() -> None:
    batch = LinkerCandidateBatch(
        row_count=4,
        left_signature_indices=np.asarray([], dtype=np.uint32),
        right_signature_indices=np.asarray([], dtype=np.uint32),
        pair_row_indices=np.asarray([], dtype=np.uint32),
        row_query_signature_indices=np.asarray([0, 0, 1, 2], dtype=np.uint32),
    )
    row_signals = {
        "pair_count": np.asarray([2.0, 2.0, 0.0, 2.0], dtype=np.float32),
        "query_year_missing": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "candidate_year_range_missing": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "query_has_affiliations": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "candidate_has_affiliations": np.zeros(4, dtype=np.float32),
        "query_has_coauthors": np.zeros(4, dtype=np.float32),
        "candidate_has_coauthors": np.zeros(4, dtype=np.float32),
        "query_has_title_terms": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "candidate_has_title_terms": np.zeros(4, dtype=np.float32),
        "query_has_venue_terms": np.zeros(4, dtype=np.float32),
        "candidate_has_venue_terms": np.zeros(4, dtype=np.float32),
        "query_has_specter": np.zeros(4, dtype=np.float32),
        "candidate_has_specter_exemplars": np.zeros(4, dtype=np.float32),
        "query_has_name_counts": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "candidate_has_name_counts": np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "query_first_token": np.asarray(["alex", "bo", "", "c"], dtype=object),
        "dominant_first_name": np.asarray(["alex", "", "casey", "c"], dtype=object),
    }
    features = {
        column: np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        for column in (
            "min_distance",
            "specter_exemplar_similarity",
            "coauthor_overlap",
            "affiliation_overlap",
            "year_compatibility",
            "candidate_year_span",
            "year_gap_to_candidate_range",
            "year_gap_signed_to_candidate_range",
            "same_dominant_first_as_best_top5",
            "same_family_as_heuristic_choice",
            "query_first_prefix_match_any_length",
            "affiliation_contradiction_severity",
            "anchor_evidence_count",
            "strong_positive_anchor_score",
            "weak_residual_anchor_score",
            "sparse_relative_winner_score",
            "last_name_count_min_rarity",
            "last_first_name_count_min_rarity",
            "top5_mean_distance",
        )
    }

    adjusted, summary = _apply_row_nan_policy(
        features,
        row_signals,
        batch,
        row_nan_policy="semantic",
    )

    distance_nan = np.asarray([False, False, True, False])
    np.testing.assert_array_equal(np.isnan(adjusted["min_distance"]), distance_nan)
    np.testing.assert_array_equal(np.isnan(adjusted["top5_mean_distance"]), distance_nan)
    np.testing.assert_array_equal(
        np.isnan(adjusted["specter_exemplar_similarity"]),
        np.asarray([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["coauthor_overlap"]),
        np.asarray([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["affiliation_overlap"]),
        np.asarray([True, True, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["year_compatibility"]),
        np.asarray([True, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["candidate_year_span"]),
        np.asarray([True, False, True, False]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["year_gap_to_candidate_range"]),
        np.asarray([True, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["year_gap_signed_to_candidate_range"]),
        np.asarray([True, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["affiliation_contradiction_severity"]),
        np.asarray([True, False, True, True]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["same_dominant_first_as_best_top5"]),
        np.asarray([False, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["same_family_as_heuristic_choice"]),
        np.asarray([False, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["query_first_prefix_match_any_length"]),
        np.asarray([False, True, True, False]),
    )
    composite_nan = np.asarray([False, False, True, False])
    np.testing.assert_array_equal(np.isnan(adjusted["anchor_evidence_count"]), composite_nan)
    np.testing.assert_array_equal(np.isnan(adjusted["strong_positive_anchor_score"]), composite_nan)
    np.testing.assert_array_equal(np.isnan(adjusted["weak_residual_anchor_score"]), composite_nan)
    np.testing.assert_array_equal(np.isnan(adjusted["sparse_relative_winner_score"]), composite_nan)
    assert adjusted["anchor_evidence_count"][0] == pytest.approx(0.1)
    assert adjusted["anchor_evidence_count"][1] == pytest.approx(0.2)
    assert adjusted["anchor_evidence_count"][3] == pytest.approx(0.4)
    assert adjusted["affiliation_contradiction_severity"][1] == pytest.approx(0.2)
    np.testing.assert_array_equal(
        np.isnan(adjusted["last_name_count_min_rarity"]),
        np.asarray([False, True, True, False]),
    )
    np.testing.assert_array_equal(
        np.isnan(adjusted["last_first_name_count_min_rarity"]),
        np.asarray([False, True, True, False]),
    )
    assert summary["row_nan_policy"] == "semantic"
    assert summary["semantic_nan_total"] > 0


def test_minimal_raw_partial_writer_reuses_label_columns_as_features(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "retrieval_rank": [1.0, 2.0],
            "query_group_id": ["q1", "q1"],
            "label": [1, 0],
        }
    )
    partial_path = tmp_path / "partial.parquet"

    _write_minimal_raw_partial_frame(
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


def test_semantic_row_nan_policy_uses_feature_direct_sources() -> None:
    batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.asarray([], dtype=np.uint32),
        right_signature_indices=np.asarray([], dtype=np.uint32),
        pair_row_indices=np.asarray([], dtype=np.uint32),
        row_query_signature_indices=np.asarray([0, 0], dtype=np.uint32),
    )
    row_signals = {
        "pair_count": np.zeros(2, dtype=np.float32),
        "query_year_missing": np.ones(2, dtype=np.float32),
        "candidate_year_range_missing": np.ones(2, dtype=np.float32),
        "query_has_affiliations": np.zeros(2, dtype=np.float32),
        "candidate_has_affiliations": np.zeros(2, dtype=np.float32),
        "query_has_coauthors": np.zeros(2, dtype=np.float32),
        "candidate_has_coauthors": np.zeros(2, dtype=np.float32),
        "query_has_title_terms": np.zeros(2, dtype=np.float32),
        "candidate_has_title_terms": np.zeros(2, dtype=np.float32),
        "query_has_venue_terms": np.zeros(2, dtype=np.float32),
        "candidate_has_venue_terms": np.zeros(2, dtype=np.float32),
        "query_has_specter": np.zeros(2, dtype=np.float32),
        "candidate_has_specter_exemplars": np.zeros(2, dtype=np.float32),
        "query_has_name_counts": np.ones(2, dtype=np.float32),
        "candidate_has_name_counts": np.ones(2, dtype=np.float32),
        "query_first_token": np.asarray(["alex", "alex"], dtype=object),
        "dominant_first_name": np.asarray(["alex", "alex"], dtype=object),
    }
    features = {
        column: np.asarray([0.1, 0.2], dtype=np.float32)
        for column in (
            "anchor_evidence_count",
            "strong_positive_anchor_score",
            "weak_residual_anchor_score",
            "sparse_relative_winner_score",
        )
    }

    adjusted, _summary = _apply_row_nan_policy(
        features,
        row_signals,
        batch,
        row_nan_policy="semantic",
    )

    assert not np.isnan(adjusted["anchor_evidence_count"]).any()
    assert not np.isnan(adjusted["weak_residual_anchor_score"]).any()
    assert not np.isnan(adjusted["sparse_relative_winner_score"]).any()
    assert np.isnan(adjusted["strong_positive_anchor_score"]).all()


def test_minimal_raw_constraint_resolution_bypasses_seed_constraints_and_ignores_disallow() -> None:
    clusterer = _ConstraintClusterer()
    batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([0, 0, 0], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0, 0], dtype=np.uint32),
    )

    labels, summary = _resolve_candidate_batch_pair_labels(
        clusterer=clusterer,
        dataset=cast(Any, SimpleNamespace()),
        batch=batch,
        index_to_signature_id={0: "q", 1: "a", 2: "b", 3: "c"},
        runtime_context=None,
        constraint_backend=None,
        chunk_size=2,
        pair_seed_bypass=np.asarray([False, True, True]),
        pair_ignore_disallow=np.asarray([False, True, False]),
    )

    assert clusterer.calls == [
        ((("q", "a"), ("q", "b")), False),
        ((("q", "c"),), False),
        ((("q", "b"), ("q", "c")), True),
    ]
    assert labels[0] == pytest.approx(-90_000.0)
    assert np.isnan(labels[1])
    assert labels[2] == pytest.approx(-90_000.0)
    assert summary["constraint_pair_count"] == 3
    assert summary["constraint_batch_calls"] == 2
    assert summary["constraint_seed_bypass_pair_count"] == 2
    assert summary["constraint_seed_bypass_batch_calls"] == 1
    assert summary["constraint_disallow_ignored_pair_count"] == 1


def test_minimal_raw_component_members_default_to_block_local_component_keys(tmp_path) -> None:
    members_path = tmp_path / "members.parquet"
    pd.DataFrame(
        [
            {"candidate_component_key": "m muller::284283", "member_index": 0, "signature_id": "a"},
            {"candidate_component_key": "m muller::284283", "member_index": 1, "signature_id": "b"},
            {"candidate_component_key": "m muller::284283", "member_index": 2, "signature_id": "c"},
            {"candidate_component_key": "other::1", "member_index": 0, "signature_id": "d"},
        ]
    ).to_parquet(members_path, index=False)
    dataset = SimpleNamespace(signature_to_block={"a": "g muller", "b": "m muller", "c": "m muller", "d": "x"})

    details = _component_member_details_by_key(
        members_path,
        {"a": 0, "b": 1, "c": 2, "d": 3},
        dataset=cast(Any, dataset),
    )

    assert details["m muller::284283"].signature_ids == ("b", "c")
    assert details["m muller::284283"].signature_indices.tolist() == [1, 2]
    assert details["other::1"].signature_ids == ("d",)


def test_minimal_raw_structural_cleaning_drops_self_only_candidates(tmp_path) -> None:
    components_dir = tmp_path / "components"
    raw_dir = tmp_path / "raw" / "toy"
    components_dir.mkdir()
    raw_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"candidate_component_key": "toy block::self", "member_index": 0, "signature_id": "q1"},
            {"candidate_component_key": "toy block::with_neighbor", "member_index": 0, "signature_id": "q1"},
            {"candidate_component_key": "toy block::with_neighbor", "member_index": 1, "signature_id": "n1"},
            {"candidate_component_key": "toy block::other", "member_index": 0, "signature_id": "n2"},
            {"candidate_component_key": "plain_self", "member_index": 0, "signature_id": "q3"},
        ]
    ).to_parquet(components_dir / "toy_members.parquet", index=False)
    signatures = {
        signature_id: {
            "signature_id": signature_id,
            "paper_id": index,
            "author_info": {"block": "toy block"},
        }
        for index, signature_id in enumerate(("q1", "n1", "q2", "n2", "q3"), start=1)
    }
    (raw_dir / "signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    (raw_dir / "papers.json").write_text("{}", encoding="utf-8")
    bundle = OfficialBundle(
        root=tmp_path,
        bundle_name="toy",
        assets={
            "candidate_members": {"datasets": {"toy": "components/toy_members.parquet"}},
            "raw_metadata": {
                "datasets": {
                    "toy": {
                        "signatures_path": "raw/toy/signatures.json",
                        "papers_path": "raw/toy/papers.json",
                    }
                }
            },
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

    cleaned, summary = _clean_minimal_raw_structural_rows(
        source_bundle=bundle,
        table_key="train_path",
        rows=rows,
        component_membership_cache={},
    )

    assert cleaned["candidate_component_key"].tolist() == ["toy block::with_neighbor", "toy block::other"]
    assert summary["rows_removed"] == 2
    assert summary["positive_rows_removed"] == 1
    assert summary["negative_rows_removed"] == 1
    assert summary["queries_removed"] == 1
    assert summary["positive_queries_changed_or_removed"] == 1


def test_minimal_raw_positive_label_marks_training_disallow_ignore() -> None:
    assert _row_label_is_positive(SimpleNamespace(label=1))
    assert _row_label_is_positive(SimpleNamespace(label=1.0))
    assert not _row_label_is_positive(SimpleNamespace(label=0))
    assert not _row_label_is_positive(SimpleNamespace(label=np.nan))


def test_minimal_raw_seed_bypass_detects_seeded_query_component() -> None:
    dataset = SimpleNamespace(
        cluster_seeds_require={"q": "seed_cluster", "m1": "seed_cluster", "other": "different"},
        cluster_seeds_disallow=set(),
    )
    row = SimpleNamespace(
        query_signature_id="q",
        split="train",
        source="",
        source_key="",
        source_kind="training",
        support_type="",
        supervision_type="",
        query_in_seed_before_holdout=0,
    )

    assert _row_allows_seed_constraint_bypass(
        cast(Any, dataset),
        row,
        seed_constraint_signature_ids=frozenset({"q", "m1", "other"}),
    )
    assert _has_query_seed_connection(cast(Any, dataset), query_signature_id="q", candidate_signature_ids=["m1"])
    assert not _has_query_seed_connection(cast(Any, dataset), query_signature_id="q", candidate_signature_ids=["other"])


def test_minimal_raw_seed_bypass_keeps_loo_marker_without_query_seed_flag() -> None:
    dataset = SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow={("q", "m1")})
    row = SimpleNamespace(
        query_signature_id="q",
        split="eval_loo",
        source="",
        source_key="s2and_eval",
        source_kind="public_test",
        support_type="",
        supervision_type="",
        query_in_seed_before_holdout=0,
    )

    assert _row_allows_seed_constraint_bypass(
        cast(Any, dataset), row, seed_constraint_signature_ids=frozenset({"q", "m1"})
    )
    assert _has_query_seed_connection(cast(Any, dataset), query_signature_id="q", candidate_signature_ids=["m1"])


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
        _dataset: Any,
        left: np.ndarray,
        right: np.ndarray,
        *,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: int,
        featurizer: Any,
        runtime_context: Any,
        suppress_orcid: bool,
    ) -> np.ndarray:
        assert dont_merge_cluster_seeds is True
        assert num_threads == 2
        assert featurizer == "featurizer"
        assert runtime_context is None
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


def test_arrow_paths_use_manifest_name_counts_index_unless_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "bundle"
    dataset_dir = bundle_root / "datasets" / "toy"
    dataset_dir.mkdir(parents=True)
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "signatures.signatures_batch_index.bin",
        "papers.papers_batch_index.bin",
        "paper_authors.paper_authors_batch_index.bin",
        "specter.specter_batch_index.bin",
    ):
        (dataset_dir / filename).write_bytes(b"placeholder")
    patch_tiny_name_counts_loader(monkeypatch)
    manifest_index, _metrics = write_name_counts_index(bundle_root)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "paths": {
                    "signatures": "signatures.arrow",
                    "papers": "papers.arrow",
                    "paper_authors": "paper_authors.arrow",
                    "specter": "specter.arrow",
                    "signatures_batch_index": "signatures.signatures_batch_index.bin",
                    "papers_batch_index": "papers.papers_batch_index.bin",
                    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin",
                    "specter_batch_index": "specter.specter_batch_index.bin",
                    "name_counts_index": "name_counts_index",
                }
            }
        ),
        encoding="utf-8",
    )
    bundle = OfficialBundle(
        root=bundle_root.resolve(),
        bundle_name="toy_bundle",
        assets={},
        models={},
        expected_metrics={},
    )

    paths = _arrow_paths_for_dataset(bundle, "toy")
    assert paths["name_counts_index"] == str(Path(manifest_index).resolve())

    override_index, _metrics = write_name_counts_index(tmp_path / "override")
    paths = _arrow_paths_for_dataset(bundle, "toy", name_counts_index_root=Path(override_index))
    assert paths["name_counts_index"] == str(Path(override_index).resolve())

    manifest_path.write_text(
        json.dumps(
            {
                "paths": {
                    "signatures": "signatures.arrow",
                    "papers": "papers.arrow",
                    "paper_authors": "paper_authors.arrow",
                    "specter": "specter.arrow",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="signatures_batch_index"):
        _arrow_paths_for_dataset(bundle, "toy", name_counts_index_root=Path(override_index))

    manifest_path.write_text(
        json.dumps(
            {
                "paths": {
                    "signatures": "signatures.arrow",
                    "papers": "papers.arrow",
                    "paper_authors": "paper_authors.arrow",
                    "specter": "specter.arrow",
                    "signatures_batch_index": "signatures.signatures_batch_index.bin",
                    "papers_batch_index": "papers.papers_batch_index.bin",
                    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin",
                    "specter_batch_index": "specter.specter_batch_index.bin",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="name_counts_index"):
        _arrow_paths_for_dataset(bundle, "toy")
    paths = _arrow_paths_for_dataset(bundle, "toy", name_counts_index_root=Path(override_index))
    assert paths["name_counts_index"] == str(Path(override_index).resolve())


def test_arrow_paths_alias_specter2_manifest_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "bundle"
    dataset_dir = bundle_root / "datasets" / "toy"
    dataset_dir.mkdir(parents=True)
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter2.arrow",
        "signatures.signatures_batch_index.bin",
        "papers.papers_batch_index.bin",
        "paper_authors.paper_authors_batch_index.bin",
        "specter2.specter_batch_index.bin",
    ):
        (dataset_dir / filename).write_bytes(b"placeholder")
    patch_tiny_name_counts_loader(monkeypatch)
    manifest_index, _metrics = write_name_counts_index(bundle_root)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "paths": {
                    "signatures": "signatures.arrow",
                    "papers": "papers.arrow",
                    "paper_authors": "paper_authors.arrow",
                    "specter2": "specter2.arrow",
                    "signatures_batch_index": "signatures.signatures_batch_index.bin",
                    "papers_batch_index": "papers.papers_batch_index.bin",
                    "paper_authors_batch_index": "paper_authors.paper_authors_batch_index.bin",
                    "specter2_batch_index": "specter2.specter_batch_index.bin",
                    "name_counts_index": "name_counts_index",
                }
            }
        ),
        encoding="utf-8",
    )
    bundle = OfficialBundle(
        root=bundle_root.resolve(),
        bundle_name="toy_bundle",
        assets={},
        models={},
        expected_metrics={},
    )

    paths = _arrow_paths_for_dataset(bundle, "toy")

    assert paths["specter"] == str((dataset_dir / "specter2.arrow").resolve())
    assert paths["specter_batch_index"] == str((dataset_dir / "specter2.specter_batch_index.bin").resolve())
    assert paths["name_counts_index"] == str(Path(manifest_index).resolve())


def test_minimal_raw_query_first_prefix_uses_full_author_before_masked_view() -> None:
    group = pd.DataFrame(
        [
            {
                "query_author": "Jianping Wang",
                "query_first_token": "j",
            }
        ]
    )

    assert _query_first_token_for_prefix(group, cast(Any, SimpleNamespace(first="j"))) == "jianping"


def test_minimal_raw_retrieval_score_uses_frozen_rust_policy() -> None:
    rust_available, rust_payload = import_s2and_rust(
        required_module_attrs=("RustHybridCentroidRetriever",),
        prefer_site_packages=True,
    )
    if not rust_available:
        raise pytest.skip.Exception(f"RustHybridCentroidRetriever unavailable: {rust_payload!r}")
    query = retrieval.QueryFeatures(
        first="john",
        middle="",
        first_initial="j",
        middle_initials=frozenset(),
        coauthor_blocks=frozenset(),
        affiliation_terms=frozenset(),
        venue_terms=frozenset(),
        year=None,
        orcid=None,
        specter=np.asarray([1.0, 0.0], dtype=np.float32),
        has_specter=True,
        has_coauthors=False,
        has_affiliations=False,
        has_full_first=True,
        has_middle=False,
    )
    summary = retrieval.ClusterSummary(
        component_key="c1",
        cluster_id="c1",
        block_key="c",
        size=1,
        first_name_counts=Counter({"john": 1}),
        middle_initial_counts=Counter(),
        coauthor_counts=Counter(),
        affiliation_counts=Counter(),
        venue_counts=Counter(),
        year_values=[],
        year_min=None,
        year_max=None,
        year_mean=None,
        orcid_values=frozenset(),
        specter_centroid=np.asarray([0.0, 1.0], dtype=np.float32),
        exemplar_vectors=[np.asarray([1.0, 0.0], dtype=np.float32)],
        title_counts=Counter(),
        name_counts_values=(),
    )
    retriever = build_rust_hybrid_centroid_retriever([summary], include_exemplars=True)

    scores = _score_candidate_summaries_with_frozen_rust_policy(
        query=query,
        summaries={"c1": summary},
        retriever=retriever,
        n_jobs=1,
    )

    assert scores["c1"] == pytest.approx(0.620239, abs=1e-6)
