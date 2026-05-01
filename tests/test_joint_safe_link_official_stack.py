"""Regression tests for the official joint safe-link shared stack helpers."""

from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.rebuild_joint_safe_link_official_stack as rebuild_stack
import scripts.reranker_dataset.official_rows as promote_name_compat
import scripts.reranker_dataset.staging as reranker_staging
import scripts.validate_joint_safe_link_official_stack as validate_stack
from scripts.joint_safe_link_dataset_contract import (
    NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH,
    _apply_name_compat_manual_positive_corrections,
    _build_name_compat_manual_positive_correction_ledger,
    _build_review_ledger_from_frame,
    _compare_ledger_slice,
    apply_hard_disallow_component_filter,
    apply_retrieval_rank_filter,
    apply_self_containment_filter,
)
from scripts.joint_safe_link_initial_only_rereview import (
    InitialOnlyRereviewDecision,
    read_initial_only_rereview_decisions,
    resolve_reviewed_safe_component_keys,
)
from scripts.joint_safe_link_official_stack import (
    DEFAULT_PACKAGE_DIR,
    _apply_classic_gate,
    _apply_classic_train_holdout_filter,
    _apply_classic_train_row_cap,
    _augmented_feature_matrix,
    _classic_feature_matrix,
    _classic_monotone_constraints_for_features,
    _evaluate_classic_manual_holdout,
    _fit_promoted_stratified_total_error_gate,
    _fit_score_margin_gate,
    _fit_single_candidate_score_gate,
    _iter_extra_eval_paths,
    _normalize_augmented_feature_frame,
    _resolve_classic_monotone_constraints,
    _score_abstain_rule,
    _score_eval_candidate_rows,
    _select_negative_training_groups,
    _summary_key_for_eval_dataset,
    compare_to_expected,
    expected_metrics_from_summary,
    format_classic_selected_gate_tables,
    load_bundle,
)
from scripts.reranker_dataset.raw_similarity import raw_similarity_features_by_component
from scripts.sync_joint_safe_link_official_bundle_metadata import sync_bundle_metadata
from scripts.validate_joint_safe_link_official_stack import (
    _feature_coverage_failures,
    _raw_feature_columns_for_validation,
    _summarize_active_feature_coverage,
    _summarize_hwang_candidate_level_label_consistency,
    _summarize_self_containing_candidate_rows,
)


def test_configure_official_rust_backend_sets_strict_rust(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Official rebuild should force strict Rust backend settings."""

    monkeypatch.delenv("S2AND_BACKEND", raising=False)
    monkeypatch.delenv(rebuild_stack.STRICT_RUST_NAME_COMPAT_ENV, raising=False)
    monkeypatch.setattr(
        rebuild_stack,
        "detect_rust_runtime_capabilities",
        lambda: SimpleNamespace(core_runtime_available=True, reason="native_extension_available"),
    )

    rebuild_stack._configure_official_rust_backend(n_jobs=3)

    assert rebuild_stack.os.environ["S2AND_BACKEND"] == "rust"
    assert rebuild_stack.os.environ[rebuild_stack.STRICT_RUST_NAME_COMPAT_ENV] == "1"
    assert rebuild_stack.os.environ["OMP_NUM_THREADS"] == "3"
    assert rebuild_stack.os.environ["RAYON_NUM_THREADS"] == "3"
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "official_rust_backend_required"
    assert event["backend"] == "rust"
    assert event["strict_name_compat_selector"] is True


def test_configure_official_rust_backend_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Official rebuild should not fall back to Python when Rust is unavailable."""

    monkeypatch.setattr(
        rebuild_stack,
        "detect_rust_runtime_capabilities",
        lambda: SimpleNamespace(core_runtime_available=False, reason="missing_extension"),
    )

    with pytest.raises(RuntimeError, match="requires the Rust backend"):
        rebuild_stack._configure_official_rust_backend(n_jobs=3)


def test_select_negative_training_groups_supports_named_filters() -> None:
    """Named negative filters should keep the expected training groups."""

    top1 = pd.DataFrame(
        [
            {
                "train_group_id": "g1",
                "title_overlap": 0.0,
                "coauthor_overlap": 0.0,
                "affiliation_overlap": 0.0,
                "count_normalized_confidence": 0.2,
            },
            {
                "train_group_id": "g2",
                "title_overlap": 0.04,
                "coauthor_overlap": 0.0,
                "affiliation_overlap": 0.0,
                "count_normalized_confidence": 0.35,
            },
            {
                "train_group_id": "g3",
                "title_overlap": 0.2,
                "coauthor_overlap": 1.0,
                "affiliation_overlap": 0.0,
                "count_normalized_confidence": 0.2,
            },
        ]
    )
    assert _select_negative_training_groups(top1, filter_name="strict") == {"g1"}
    assert _select_negative_training_groups(top1, filter_name="better") == {"g1", "g2"}
    assert _select_negative_training_groups(top1, filter_name="medium") == {"g1", "g2"}


def test_normalize_augmented_feature_frame_derives_query_first_features() -> None:
    """Augmented frame normalization should derive missing first-name features."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "query_view": "full",
            },
            {
                "query_author": "H. Wang",
                "dominant_first_name": "huijuan",
                "query_view": "initial_only",
            },
        ]
    )
    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=("query_view", "query_first_prefix_match"),
    )
    assert out["query_first_prefix_match"].tolist() == [1.0, 0.0]


def test_normalize_augmented_feature_frame_overwrites_stale_runtime_features() -> None:
    """Runtime-derived features should be recomputed from raw prerequisites when available."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "query_first_prefix_match": 0.0,
                "cluster_size": 17,
                "cluster_size_log_capped": 0.0,
            }
        ]
    )

    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=("query_first_prefix_match", "cluster_size_log_capped"),
    )

    assert out.iloc[0]["query_first_prefix_match"] == 1.0
    assert out.iloc[0]["cluster_size_log_capped"] > 0.0


def test_normalize_augmented_feature_frame_derives_anchor_evidence_features() -> None:
    """Promoted anchor formulas should be derived from raw candidate-row evidence."""

    df = pd.DataFrame(
        [
            {
                "min_distance": 0.1,
                "specter_exemplar_similarity": 0.8,
                "title_overlap": 0.3,
                "coauthor_overlap": 0.3,
                "affiliation_overlap": 0.1,
                "venue_overlap": 0.4,
                "year_compatibility": 0.95,
                "retrieval_score_gap_vs_best_competitor": 0.06,
                "candidate_contradiction_score": 0.2,
                "same_family_as_top1": 1.0,
                "candidate_pair_share_within_coarse_family": 0.25,
                "cluster_size": 2,
                "named_signature_count": 2,
                "retrieval_rank": 1,
                "anchor_evidence_count": 0.0,
            }
        ]
    )

    out = _normalize_augmented_feature_frame(
        df,
        feature_columns=(
            "anchor_evidence_count",
            "strong_positive_anchor_score",
            "weak_residual_anchor_score",
            "sparse_relative_winner_score",
        ),
    )

    assert out.iloc[0]["anchor_evidence_count"] == pytest.approx(7.0)
    assert out.iloc[0]["strong_positive_anchor_score"] == pytest.approx(0.47067, abs=1e-6)
    assert out.iloc[0]["weak_residual_anchor_score"] == pytest.approx(0.6506, abs=1e-6)
    assert out.iloc[0]["sparse_relative_winner_score"] == pytest.approx(0.09759, abs=1e-6)


def test_classic_feature_matrix_uses_query_first_token_without_query_author() -> None:
    """Public eval files should derive prefix match from query_first_token."""

    df = pd.DataFrame(
        [
            {"query_first_token": "Hanbing", "dominant_first_name": "hanbing"},
            {"dominant_first_name": "hanbing"},
        ]
    )

    features = _classic_feature_matrix(df, ("query_first_prefix_match",))

    assert features["query_first_prefix_match"].tolist() == [1.0, 0.0]


def test_self_containing_candidate_validation_finds_query_membership(tmp_path: Path) -> None:
    """Rows whose raw candidate component contains the query signature should be fatal."""

    row_path = tmp_path / "rows.csv.gz"
    fieldnames = [
        "dataset",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
    ]
    with gzip.open(row_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "c1",
                "candidate_cluster_id": "cluster1",
                "label": 1,
            }
        )
        writer.writerow(
            {
                "dataset": "toy",
                "query_group_id": "q2",
                "query_signature_id": "s9",
                "candidate_component_key": "c2",
                "candidate_cluster_id": "cluster2",
                "label": 0,
            }
        )

    def load_lookup(dataset_name: str) -> dict[str, frozenset[str]]:
        assert dataset_name == "toy"
        return {
            "c1": frozenset({"s1", "s2"}),
            "cluster2": frozenset({"s3", "s4"}),
        }

    summary = _summarize_self_containing_candidate_rows(
        (row_path,),
        component_lookup_loader=load_lookup,
        chunksize=1,
    )

    assert summary["self_containing_rows"] == 1
    assert summary["self_containing_positive_rows"] == 1
    assert summary["files"][0]["samples"][0]["query_group_id"] == "q1"


def test_self_containing_candidate_validation_allows_eval_residual_loo(tmp_path: Path) -> None:
    """Reviewed eval LOO positives should not fail raw self-containment validation."""

    row_path = tmp_path / "rows.csv.gz"
    fieldnames = [
        "source",
        "split",
        "dataset",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
    ]
    with gzip.open(row_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source": "labeled_loo",
                "split": "eval_loo",
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "c1",
                "candidate_cluster_id": "cluster1",
                "label": 1,
            }
        )

    summary = _summarize_self_containing_candidate_rows(
        (row_path,),
        component_lookup_loader=lambda _dataset_name: {"c1": frozenset({"s1", "s2"})},
        chunksize=1,
    )

    assert summary["self_containing_rows"] == 0
    assert summary["allowed_residual_loo_self_containing_rows"] == 1


def test_self_containing_candidate_validation_allows_reviewed_rescue_positive(tmp_path: Path) -> None:
    """Reviewed rescue positives are residual LOO rows materialized without the query signature."""

    row_path = tmp_path / "rows.csv.gz"
    fieldnames = [
        "source",
        "split",
        "dataset",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
    ]
    with gzip.open(row_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "source": "s2and_rescue_manual_review",
                "split": "calibration_check",
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "c1",
                "candidate_cluster_id": "cluster1",
                "label": 1,
            }
        )

    summary = _summarize_self_containing_candidate_rows(
        (row_path,),
        component_lookup_loader=lambda _dataset_name: {"c1": frozenset({"s1", "s2"})},
        chunksize=1,
    )

    assert summary["self_containing_rows"] == 0
    assert summary["allowed_residual_loo_self_containing_rows"] == 1


def test_self_containing_candidate_validation_allows_generated_residual_holdout(tmp_path: Path) -> None:
    """Generated residual holdout rows should not fail raw self-containment validation."""

    row_path = tmp_path / "rows.csv.gz"
    fieldnames = [
        "dataset",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
        "query_in_seed_before_holdout",
        "cluster_size",
    ]
    with gzip.open(row_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "own_cluster_residual",
                "candidate_cluster_id": "cluster1",
                "label": 0,
                "query_in_seed_before_holdout": 1,
                "cluster_size": 1,
            }
        )

    summary = _summarize_self_containing_candidate_rows(
        (row_path,),
        component_lookup_loader=lambda _dataset_name: {"own_cluster_residual": frozenset({"s1", "s2"})},
        chunksize=1,
    )

    assert summary["self_containing_rows"] == 0
    assert summary["allowed_generated_residual_self_containing_rows"] == 1


def test_self_containing_candidate_validation_uses_preferred_step2_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict-surface validation should use reconciled component assignments when present."""

    step2_dir = tmp_path / "step2"
    step2_dir.mkdir()
    (step2_dir / "reconciled_signature_to_cluster_id.json").write_text(
        json.dumps({"s1": "c_reconciled", "s2": "c_reconciled"}),
        encoding="utf-8",
    )
    (step2_dir / "predicted_clusters.json").write_text(
        json.dumps({"block": {"c_reconciled": ["s1"], "c_raw_only": ["s3"]}}),
        encoding="utf-8",
    )
    monkeypatch.setitem(validate_stack.GIANT_STEP2_DIRS, "toy_preferred", step2_dir)

    lookup = validate_stack._load_component_signature_lookup("toy_preferred")

    assert lookup["c_reconciled"] == frozenset({"s1", "s2"})
    assert lookup["c_raw_only"] == frozenset({"s3"})


def test_self_containing_candidate_validation_allows_required_manual_positive(tmp_path: Path) -> None:
    """Manual calibration/eval positives can be preserved when group metadata marks them required."""

    row_path = tmp_path / "rows.csv.gz"
    fieldnames = [
        "dataset",
        "query_group_id",
        "query_signature_id",
        "candidate_component_key",
        "candidate_cluster_id",
        "label",
        "positive_candidate_keys",
    ]
    with gzip.open(row_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "toy",
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "legacy_positive",
                "candidate_cluster_id": "cluster1",
                "label": 1,
                "positive_candidate_keys": "legacy_positive",
            }
        )

    summary = _summarize_self_containing_candidate_rows(
        (row_path,),
        component_lookup_loader=lambda _dataset_name: {"legacy_positive": frozenset({"s1", "s2"})},
        chunksize=1,
    )

    assert summary["self_containing_rows"] == 0
    assert summary["allowed_required_positive_self_containing_rows"] == 1


def test_contract_retrieval_rank_filter_matches_official_window_policy() -> None:
    """The centralized filter policy should own the official top-k row cap."""

    rows = [
        {"query_group_id": "q1", "candidate_component_key": "c1", "retrieval_rank": "1", "label": "1"},
        {"query_group_id": "q1", "candidate_component_key": "c2", "retrieval_rank": "25", "label": "0"},
        {"query_group_id": "q1", "candidate_component_key": "c3", "retrieval_rank": "26", "label": "1"},
    ]

    result = apply_retrieval_rank_filter(rows)

    assert [row["candidate_component_key"] for row in result.kept_rows] == ["c1", "c2"]
    assert result.rows_before == 3
    assert result.rows_after == 2
    assert result.positive_rows_before == 2
    assert result.positive_rows_after == 1
    assert result.dropped_candidate_component_keys == ("c3",)
    assert result.dropped_positive_candidate_component_keys == ("c3",)


def test_contract_self_containment_filter_tracks_dropped_positive_candidates() -> None:
    """The centralized self-filter should preserve order and report dropped positive candidates."""

    rows = [
        {"candidate_component_key": "c1", "label": "0"},
        {"candidate_component_key": "c2", "label": "1"},
        {"candidate_component_key": "c3", "label": "1"},
    ]

    result = apply_self_containment_filter(
        rows,
        contains_query_signature=lambda row: row["candidate_component_key"] in {"c2"},
    )

    assert [row["candidate_component_key"] for row in result.kept_rows] == ["c1", "c3"]
    assert result.dropped_candidate_component_keys == ("c2",)
    assert result.dropped_positive_candidate_component_keys == ("c2",)
    assert result.positive_rows_after == 1


def test_contract_hard_disallow_filter_drops_components_with_any_disallow_pair() -> None:
    """The centralized hard-disallow filter should own component removal after pairwise scoring."""

    result = apply_hard_disallow_component_filter(
        ["c1", "c2", "c3"],
        disallow_pair_count_by_component={"c1": 0, "c2": 2, "c3": 1},
    )

    assert result.kept_component_keys == ("c1",)
    assert result.dropped_component_keys == ("c2", "c3")
    assert result.components_before == 3
    assert result.components_after == 1


def test_contract_hard_disallow_filter_preserves_known_positive_components() -> None:
    """Known positives should survive hard-disallow filtering."""

    result = apply_hard_disallow_component_filter(
        ["c1", "c2", "c3"],
        disallow_pair_count_by_component={"c1": 0, "c2": 2, "c3": 1},
        preserve_component_keys={"c2"},
    )

    assert result.kept_component_keys == ("c1", "c2")
    assert result.dropped_component_keys == ("c3",)
    assert result.components_before == 3
    assert result.components_after == 2


def test_initial_only_rereview_collapse_is_conservative_for_conflicts(tmp_path: Path) -> None:
    """Conflicting model-visible reviews should be dropped instead of cast as labels."""

    review_path = tmp_path / "aggregate.tsv"
    fieldnames = [
        "query_case_id",
        "dataset",
        "active_contract_decision",
        "active_contract_safe_positive_component_keys",
        "evidence_contract_issue",
    ]
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                {
                    "query_case_id": "toy:1:initial_only",
                    "dataset": "toy",
                    "active_contract_decision": "link",
                    "active_contract_safe_positive_component_keys": "c2",
                    "evidence_contract_issue": "none",
                },
                {
                    "query_case_id": "toy:1:initial_only",
                    "dataset": "toy",
                    "active_contract_decision": "abstain",
                    "active_contract_safe_positive_component_keys": "",
                    "evidence_contract_issue": "packet_too_weak",
                },
                {
                    "query_case_id": "toy:2:initial_only",
                    "dataset": "toy",
                    "active_contract_decision": "abstain",
                    "active_contract_safe_positive_component_keys": "",
                    "evidence_contract_issue": "raw_query_author_needed",
                },
                {
                    "query_case_id": "toy:3:initial_only",
                    "dataset": "toy",
                    "active_contract_decision": "abstain",
                    "active_contract_safe_positive_component_keys": "",
                    "evidence_contract_issue": "packet_too_weak",
                },
            ]
        )

    decisions = read_initial_only_rereview_decisions(review_path)

    assert decisions["toy:1:initial_only"].action == "drop_query"
    assert decisions["toy:1:initial_only"].reason_bucket == "conflicting_model_visible_reviews"
    assert decisions["toy:2:initial_only"].action == "drop_query"
    assert decisions["toy:2:initial_only"].reason_bucket == "feature_contract_failure"
    assert decisions["toy:3:initial_only"].action == "force_no_positive"


def test_initial_only_rereview_resolves_pipe_bearing_component_keys() -> None:
    """Pipe-bearing component keys should stay atomic when they exist in the row surface."""

    component_key = "hon, hong|middle=a, hong|middle=b, hong_943"

    assert resolve_reviewed_safe_component_keys(
        (component_key,),
        candidate_component_keys={component_key, "other"},
    ) == (component_key,)
    assert resolve_reviewed_safe_component_keys(
        ("c1|c2",),
        candidate_component_keys={"c1", "c2", "c3"},
    ) == ("c1", "c2")


def test_initial_only_rereview_updates_group_labels_and_metadata() -> None:
    """Collapsed re-review decisions should relabel active rows under the visible-evidence contract."""

    rows = [
        {
            "query_group_id": "toy:1:initial_only",
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": "1",
        },
        {
            "query_group_id": "toy:1:initial_only",
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": "0",
        },
    ]
    decision = InitialOnlyRereviewDecision(
        query_group_id="toy:1:initial_only",
        dataset="toy",
        action="candidate_positive",
        reason_bucket="unanimous_model_visible_link",
        safe_component_key_texts=("c2",),
        reviewed_row_count=1,
        review_decisions=("link",),
        evidence_issues=("none",),
    )

    relabeled = rebuild_stack._apply_initial_only_rereview_to_group(rows, decision=decision)

    assert [row["label"] for row in relabeled] == ["0", "1"]
    assert {row["positive_candidate_keys"] for row in relabeled} == {"c2"}
    assert {row["best_positive_retrieval_rank"] for row in relabeled} == {"2"}


def test_rebuilt_row_writer_reapplies_initial_only_rereview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final row writing should not leak stale rebuilt labels from a resumable spool."""

    bundle_root = tmp_path / "bundle"
    output_relpath = Path("test") / "demo_rows.csv.gz"
    (bundle_root / "test").mkdir(parents=True)
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE staged_groups (source_path TEXT, group_index INTEGER, rows_blob BLOB)")
    connection.execute("CREATE TABLE rebuilt_groups (source_path TEXT, group_index INTEGER, rows_blob BLOB)")
    connection.execute(
        "INSERT INTO staged_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        (
            str(output_relpath).replace("/", "\\"),
            1,
            rebuild_stack._compress_rows(
                [
                    {
                        "query_group_id": "toy:1:initial_only",
                        "candidate_component_key": "c1",
                        "retrieval_rank": "1",
                        "label": "1",
                    }
                ]
            ),
        ),
    )
    connection.execute(
        "INSERT INTO rebuilt_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        (
            str(output_relpath).replace("/", "\\"),
            1,
            rebuild_stack._compress_rows(
                [
                    {
                        "query_group_id": "toy:1:initial_only",
                        "candidate_component_key": "c1",
                        "retrieval_rank": "1",
                        "label": "1",
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(rebuild_stack, "DEST_BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(
        rebuild_stack,
        "read_initial_only_rereview_decisions",
        lambda: {
            "toy:1:initial_only": InitialOnlyRereviewDecision(
                query_group_id="toy:1:initial_only",
                dataset="toy",
                action="force_no_positive",
                reason_bucket="packet_too_weak",
                safe_component_key_texts=(),
                reviewed_row_count=1,
                review_decisions=("abstain",),
                evidence_issues=("packet_too_weak",),
            )
        },
    )

    rebuild_stack._write_rebuilt_row_files(
        connection=connection,
        fieldnames_by_path={
            str(output_relpath).replace("/", "\\"): [
                "query_group_id",
                "candidate_component_key",
                "retrieval_rank",
                "label",
            ]
        },
        ordered_source_paths=[str(output_relpath).replace("/", "\\")],
    )

    rewritten = pd.read_csv(bundle_root / output_relpath, compression="gzip")
    assert rewritten["label"].tolist() == [0]


def test_rebuilt_row_writer_restores_staged_labels_from_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resumed rebuilds should not write labels invented during prior feature materialization."""

    bundle_root = tmp_path / "bundle"
    output_relpath = Path("test") / "demo_rows.csv.gz"
    (bundle_root / "test").mkdir(parents=True)
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE staged_groups (source_path TEXT, group_index INTEGER, rows_blob BLOB)")
    connection.execute("CREATE TABLE rebuilt_groups (source_path TEXT, group_index INTEGER, rows_blob BLOB)")
    source_path = str(output_relpath).replace("/", "\\")
    connection.execute(
        "INSERT INTO staged_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        (
            source_path,
            1,
            rebuild_stack._compress_rows(
                [
                    {
                        "query_group_id": "toy:1:full",
                        "candidate_component_key": "c1",
                        "retrieval_rank": "1",
                        "label": "0",
                        "positive_candidate_count": "0",
                        "positive_candidate_keys": "",
                        "group_has_positive": "0",
                        "best_positive_retrieval_rank": "",
                    }
                ]
            ),
        ),
    )
    connection.execute(
        "INSERT INTO rebuilt_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        (
            source_path,
            1,
            rebuild_stack._compress_rows(
                [
                    {
                        "query_group_id": "toy:1:full",
                        "candidate_component_key": "c1",
                        "retrieval_rank": "1",
                        "label": "1",
                        "positive_candidate_count": "1",
                        "positive_candidate_keys": "c1",
                        "group_has_positive": "1",
                        "best_positive_retrieval_rank": "1",
                    }
                ]
            ),
        ),
    )
    monkeypatch.setattr(rebuild_stack, "DEST_BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(rebuild_stack, "read_initial_only_rereview_decisions", lambda: {})

    rebuild_stack._write_rebuilt_row_files(
        connection=connection,
        fieldnames_by_path={
            source_path: [
                "query_group_id",
                "candidate_component_key",
                "retrieval_rank",
                "label",
                "positive_candidate_count",
                "positive_candidate_keys",
                "group_has_positive",
                "best_positive_retrieval_rank",
            ]
        },
        ordered_source_paths=[source_path],
    )

    rewritten = pd.read_csv(bundle_root / output_relpath, compression="gzip")
    assert rewritten["label"].tolist() == [0]
    assert rewritten["positive_candidate_count"].tolist() == [0]
    assert rewritten["group_has_positive"].tolist() == [0]


def test_rebuilt_materialization_preserves_staged_labels_over_require_constraints() -> None:
    """Feature rematerialization must not convert hard constraints into official labels."""

    rows = [
        {
            "query_group_id": "toy:1:full",
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": 1,
            "binary_safe_link_target": 1,
            "positive_candidate_count": 1,
            "positive_candidate_keys": "c1",
            "group_has_positive": 1,
            "best_positive_retrieval_rank": 1,
        },
        {
            "query_group_id": "toy:1:full",
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": 0,
            "binary_safe_link_target": 0,
            "positive_candidate_count": 1,
            "positive_candidate_keys": "c1",
            "group_has_positive": 1,
            "best_positive_retrieval_rank": 1,
        },
    ]
    original_by_component = {
        "c1": {
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": "0",
            "binary_safe_link_target": "0",
        },
        "c2": {
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": "1",
            "binary_safe_link_target": "1",
        },
    }

    rebuild_stack._restore_staged_labels_and_metadata(rows, original_by_component=original_by_component)

    assert [row["label"] for row in rows] == ["0", "1"]
    assert [row["binary_safe_link_target"] for row in rows] == ["0", "1"]
    assert {row["positive_candidate_count"] for row in rows} == {1}
    assert {row["positive_candidate_keys"] for row in rows} == {"c2"}
    assert {row["group_has_positive"] for row in rows} == {1}
    assert {row["best_positive_retrieval_rank"] for row in rows} == {2}


def test_review_ledger_uses_initial_only_rereview_override() -> None:
    """Legacy review ledgers should not retain superseded initial-only positives."""

    reviews = pd.DataFrame(
        [
            {
                "query_case_id": "toy:1:initial_only",
                "dataset": "toy",
                "split": "test",
                "query_view": "initial_only",
                "manual_assessment": "safe_link",
                "safe_positive_component_keys": "old_positive",
                "correction_type": "add_positive",
                "reason_bucket": "",
                "review_file": "scratch/old.tsv",
                "notes": "",
            }
        ]
    )
    decisions = {
        "toy:1:initial_only": InitialOnlyRereviewDecision(
            query_group_id="toy:1:initial_only",
            dataset="toy",
            action="drop_query",
            reason_bucket="feature_contract_failure",
            safe_component_key_texts=(),
            reviewed_row_count=2,
            review_decisions=("abstain",),
            evidence_issues=("raw_query_author_needed",),
        )
    }

    ledger_rows = _build_review_ledger_from_frame(
        reviews=reviews,
        ledger_source="legacy_review",
        slice_key="demo",
        review_path=Path("scratch/old.tsv"),
        initial_only_decisions=decisions,
    )

    assert ledger_rows == [
        {
            "ledger_source": "legacy_review",
            "slice_key": "demo",
            "dataset": "toy",
            "split": "test",
            "query_group_id": "toy:1:initial_only",
            "query_view": "initial_only",
            "decision_scope": "query",
            "candidate_component_key": "",
            "target_label": "",
            "action": "drop_query",
            "reason_bucket": "feature_contract_failure",
            "review_source_path": "scratch/old.tsv",
            "notes": "model_visible_initial_only_rereview rows=2; decisions=abstain; issues=raw_query_author_needed",
        }
    ]


def test_s2and_full_relabel_updates_group_labels_and_metadata() -> None:
    """S2AND staging should recover reviewed labels from the pre-filter row surface."""

    rows = [
        {
            "query_group_id": "arnetminer:1:full",
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": "1",
        },
        {
            "query_group_id": "arnetminer:1:full",
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": "0",
        },
    ]
    decisions = {
        "arnetminer:1:full": rebuild_stack.S2ANDFullRelabelDecision(
            safe_component_keys=("c2",),
            split="test",
            correction_type="replace_positives",
        )
    }

    relabeled = rebuild_stack._apply_s2and_full_relabel_to_group(rows, decisions=decisions)

    assert [row["label"] for row in relabeled] == ["0", "1"]
    assert {row["positive_candidate_keys"] for row in relabeled} == {"c2"}
    assert {row["best_positive_retrieval_rank"] for row in relabeled} == {"2"}


def test_s2and_assignment_rows_use_review_split_and_active_labels() -> None:
    """Promoted split assignments should be reconstructed for rematerialized S2AND queries."""

    rows = [
        {
            "query_group_id": "arnetminer:1:full",
            "base_group_id": "arnetminer:1",
            "dataset": "arnetminer",
            "query_source": "labeled",
            "query_view": "full",
            "support_type": "labeled",
            "query_first_token": "Alice",
            "candidate_component_key": "c1",
            "retrieval_rank": "1",
            "label": "0",
        },
        {
            "query_group_id": "arnetminer:1:full",
            "base_group_id": "arnetminer:1",
            "dataset": "arnetminer",
            "query_source": "labeled",
            "query_view": "full",
            "support_type": "labeled",
            "query_first_token": "Alice",
            "candidate_component_key": "c2",
            "retrieval_rank": "2",
            "label": "1",
        },
    ]
    decisions = {
        "arnetminer:1:full": rebuild_stack.S2ANDFullRelabelDecision(
            safe_component_keys=("c2",),
            split="calibration_check",
            correction_type="add_positive",
        )
    }

    assignments = rebuild_stack._s2and_assignment_rows_from_active_rows(rows, decisions=decisions)

    assert assignments == [
        {
            "query_group_id": "arnetminer:1:full",
            "base_group_id": "arnetminer:1",
            "dataset": "arnetminer",
            "source_key": "s2and_eval",
            "source_kind": "public_test",
            "source_priority": "1",
            "query_source": "labeled",
            "query_view": "full",
            "support_type": "labeled",
            "source_stratum": "s2and_block",
            "has_positive_candidate": "True",
            "positive_first": "False",
            "positive_rank_bucket": "positive_not_first",
            "raw_has_positive_candidate": "True",
            "raw_positive_first": "False",
            "manual_safe_target": "1",
            "correction_type": "add_positive",
            "first_name_bucket": "multi_letter_first",
            "multiple_candidates": "True",
            "candidate_count": "2",
            "min_positive_rank": "2",
            "min_retrieval_rank": "1",
            "max_retrieval_rank": "2",
            "positive_candidate_rows": "1",
            "split": "calibration_check",
            "stratum_key": "s2and_block|has_pos=1|positive_not_first|multi_letter_first|multi_cand=1",
        }
    ]


def test_stratified_split_refresh_prunes_assignments_for_dropped_source_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Promoted split assignments should follow active row files after re-review drops."""

    bundle_root = tmp_path / "bundle"
    s2and_path = bundle_root / rebuild_stack.S2AND_ROW_RELATIVE_PATH
    hwang_path = bundle_root / rebuild_stack.HWANG_ROW_RELATIVE_PATH
    split_root = bundle_root / "calibration" / "stratified_eval_test_split"
    s2and_path.parent.mkdir(parents=True, exist_ok=True)
    hwang_path.parent.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True)

    def write_gzip_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_gzip_rows(
        s2and_path,
        [
            {
                "query_group_id": "s2:q1:initial_only",
                "base_group_id": "s2:q1",
                "dataset": "arnetminer",
                "query_source": "labeled",
                "query_view": "initial_only",
                "support_type": "labeled",
                "query_first_token": "A",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 1,
            }
        ],
        [
            "query_group_id",
            "base_group_id",
            "dataset",
            "query_source",
            "query_view",
            "support_type",
            "query_first_token",
            "candidate_component_key",
            "retrieval_rank",
            "label",
        ],
    )
    write_gzip_rows(
        hwang_path,
        [{"query_group_id": "h:q1", "candidate_component_key": "h1", "label": 0}],
        ["query_group_id", "candidate_component_key", "label"],
    )
    assignment_fieldnames = [
        "query_group_id",
        "base_group_id",
        "dataset",
        "source_key",
        "source_kind",
        "source_priority",
        "query_source",
        "query_view",
        "support_type",
        "source_stratum",
        "has_positive_candidate",
        "positive_first",
        "positive_rank_bucket",
        "raw_has_positive_candidate",
        "raw_positive_first",
        "manual_safe_target",
        "correction_type",
        "first_name_bucket",
        "multiple_candidates",
        "candidate_count",
        "min_positive_rank",
        "min_retrieval_rank",
        "max_retrieval_rank",
        "positive_candidate_rows",
        "split",
        "stratum_key",
    ]
    with (split_root / "combined_query_split_assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assignment_fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "query_group_id": "h:q1",
                    "base_group_id": "h:q1",
                    "dataset": "h_wang",
                    "source_key": "hwang_eval",
                    "split": "test",
                },
                {
                    "query_group_id": "h:dropped",
                    "base_group_id": "h:dropped",
                    "dataset": "h_wang",
                    "source_key": "hwang_eval",
                    "split": "test",
                },
            ]
        )
    (split_root / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rebuild_stack, "DEST_BUNDLE_ROOT", bundle_root)
    monkeypatch.setattr(
        rebuild_stack,
        "_read_s2and_full_relabel_decisions",
        lambda: {
            "s2:q1:initial_only": rebuild_stack.S2ANDFullRelabelDecision(
                safe_component_keys=("c1",),
                split="test",
                correction_type="add_positive",
            )
        },
    )
    monkeypatch.setattr(rebuild_stack, "read_initial_only_rereview_decisions", lambda: {})

    summary = rebuild_stack._refresh_s2and_stratified_split_from_reviews(
        selected_row_paths=(rebuild_stack.S2AND_ROW_RELATIVE_PATH, rebuild_stack.HWANG_ROW_RELATIVE_PATH),
    )

    assert summary is not None
    assert summary["pruned_assignment_counts"] == {"hwang_eval": 1}
    refreshed = pd.read_csv(split_root / "combined_query_split_assignments.csv")
    assert set(refreshed["query_group_id"]) == {"h:q1", "s2:q1:initial_only"}


def test_hwang_candidate_level_label_consistency_rejects_stale_query_target(tmp_path: Path) -> None:
    """H-Wang query targets should not declare positives absent from candidate labels."""

    hwang_rows_path = tmp_path / "hwang_eval_rows.csv.gz"
    with gzip.open(hwang_rows_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_group_id", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"query_group_id": "q1", "label": 1},
                {"query_group_id": "q1", "label": 0},
                {"query_group_id": "q2", "label": 0},
            ]
        )
    clean_overrides_path = tmp_path / "hwang_cleaned_eval_overrides.csv"
    with clean_overrides_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_group_id", "manual_safe_target", "manual_assessment", "correction_type"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "query_group_id": "q1",
                    "manual_safe_target": 1,
                    "manual_assessment": "keep_surviving_raw_labels",
                    "correction_type": "none",
                },
                {
                    "query_group_id": "q2",
                    "manual_safe_target": 1,
                    "manual_assessment": "stale_query_level_positive",
                    "correction_type": "none",
                },
            ]
        )
    manifest_path = tmp_path / "hwang_candidate_level_label_overrides.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_group_id", "manual_safe_target", "label_action"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"query_group_id": "q1", "manual_safe_target": 1, "label_action": "keep_surviving_raw_labels"},
                {"query_group_id": "q2", "manual_safe_target": 0, "label_action": "keep_surviving_raw_labels"},
            ]
        )

    with pytest.raises(ValueError, match="hwang_clean_override_target_mismatch"):
        _summarize_hwang_candidate_level_label_consistency(
            hwang_rows_path=hwang_rows_path,
            clean_overrides_path=clean_overrides_path,
            manifest_path=manifest_path,
        )


def test_hwang_candidate_level_relabel_applies_after_self_filter(tmp_path: Path) -> None:
    """H-Wang rebuilds should re-derive query targets from surviving reviewed candidates."""

    bundle_root = tmp_path / "bundle"
    test_root = bundle_root / "test"
    test_root.mkdir(parents=True)
    hwang_rows_path = test_root / "hwang_eval_rows.csv.gz"
    with gzip.open(hwang_rows_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_group_id", "candidate_component_key", "label"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"query_group_id": "h:q1", "candidate_component_key": "c1", "label": 0},
                {"query_group_id": "h:q1", "candidate_component_key": "c2", "label": 0},
                {"query_group_id": "h:q2", "candidate_component_key": "c3", "label": 1},
                {"query_group_id": "h:q3", "candidate_component_key": "c4", "label": 0},
            ]
        )
    manifest_path = test_root / "hwang_candidate_level_label_overrides.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_group_id",
                "dataset",
                "correction_type",
                "reviewed_candidate_component_key",
                "reviewed_candidate_survived",
                "raw_positive_rows_before_candidate_relabel",
                "label_action",
                "review_source_path",
                "positive_rows_after_candidate_relabel",
                "manual_safe_target",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "query_group_id": "h:q1",
                    "dataset": "h_wang",
                    "correction_type": "top1_should_link",
                    "reviewed_candidate_component_key": "c1",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 0,
                    "label_action": "reviewed_positive_missing_after_filter",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 0,
                    "manual_safe_target": 0,
                },
                {
                    "query_group_id": "h:q2",
                    "dataset": "h_wang",
                    "correction_type": "should_abstain",
                    "reviewed_candidate_component_key": "",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 1,
                    "label_action": "keep_surviving_raw_labels",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 1,
                    "manual_safe_target": 1,
                },
                {
                    "query_group_id": "h:q3",
                    "dataset": "h_wang",
                    "correction_type": "non_top1_should_link",
                    "reviewed_candidate_component_key": "missing",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 0,
                    "label_action": "keep_surviving_raw_labels",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 0,
                    "manual_safe_target": 0,
                },
            ]
        )
    per_file_summaries = [
        {
            "path": "test/hwang_eval_rows.csv.gz",
            "positive_rows_before": 1,
            "positive_rows_after": 1,
            "positive_groups_after": 1,
        }
    ]

    summary = rebuild_stack._apply_hwang_candidate_level_label_overrides(
        bundle_root=bundle_root,
        selected_row_paths=(rebuild_stack.HWANG_ROW_RELATIVE_PATH,),
        per_file_summaries=per_file_summaries,
    )

    assert summary is not None
    assert summary["positive_rows_after_candidate_relabel"] == 1
    assert summary["positive_queries_after_candidate_relabel"] == 1
    assert summary["label_action_counts"] == {
        "add_reviewed_positive": 1,
        "force_no_positive": 1,
        "reviewed_positive_missing_after_filter": 1,
    }
    rewritten_rows = pd.read_csv(hwang_rows_path, compression="gzip")
    assert rewritten_rows.set_index(["query_group_id", "candidate_component_key"])["label"].to_dict() == {
        ("h:q1", "c1"): 1,
        ("h:q1", "c2"): 0,
        ("h:q2", "c3"): 0,
        ("h:q3", "c4"): 0,
    }
    clean_overrides = pd.read_csv(test_root / "hwang_cleaned_eval_overrides.csv")
    assert clean_overrides.set_index("query_group_id")["manual_safe_target"].to_dict() == {
        "h:q1": 1,
        "h:q2": 0,
        "h:q3": 0,
    }
    assert per_file_summaries[0]["positive_rows_after"] == 1
    assert per_file_summaries[0]["positive_groups_after"] == 1


def test_hwang_candidate_level_relabel_allows_initial_only_dropped_manifest_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H-Wang post-relabel should tolerate manifest rows removed by initial-only quarantine."""

    bundle_root = tmp_path / "bundle"
    test_root = bundle_root / "test"
    test_root.mkdir(parents=True)
    hwang_rows_path = test_root / "hwang_eval_rows.csv.gz"
    with gzip.open(hwang_rows_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_group_id", "candidate_component_key", "label"])
        writer.writeheader()
        writer.writerow({"query_group_id": "h:q1", "candidate_component_key": "c1", "label": 1})
        writer.writerow({"query_group_id": "h:force:initial_only", "candidate_component_key": "c2", "label": 1})
    manifest_path = test_root / "hwang_candidate_level_label_overrides.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_group_id",
                "dataset",
                "correction_type",
                "reviewed_candidate_component_key",
                "reviewed_candidate_survived",
                "raw_positive_rows_before_candidate_relabel",
                "label_action",
                "review_source_path",
                "positive_rows_after_candidate_relabel",
                "manual_safe_target",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "query_group_id": "h:q1",
                    "dataset": "h_wang",
                    "correction_type": "none",
                    "reviewed_candidate_component_key": "",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 1,
                    "label_action": "keep_surviving_raw_labels",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 1,
                    "manual_safe_target": 1,
                },
                {
                    "query_group_id": "h:force:initial_only",
                    "dataset": "h_wang",
                    "correction_type": "top1_should_link",
                    "reviewed_candidate_component_key": "c2",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 1,
                    "label_action": "add_reviewed_positive",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 1,
                    "manual_safe_target": 1,
                },
                {
                    "query_group_id": "h:drop:initial_only",
                    "dataset": "h_wang",
                    "correction_type": "should_abstain",
                    "reviewed_candidate_component_key": "",
                    "reviewed_candidate_survived": 0,
                    "raw_positive_rows_before_candidate_relabel": 1,
                    "label_action": "force_no_positive",
                    "review_source_path": "scratch/demo.tsv",
                    "positive_rows_after_candidate_relabel": 0,
                    "manual_safe_target": 0,
                },
            ]
        )
    monkeypatch.setattr(
        rebuild_stack,
        "read_initial_only_rereview_decisions",
        lambda: {
            "h:force:initial_only": InitialOnlyRereviewDecision(
                query_group_id="h:force:initial_only",
                dataset="h_wang",
                action="force_no_positive",
                reason_bucket="packet_too_weak",
                safe_component_key_texts=(),
                reviewed_row_count=1,
                review_decisions=("abstain",),
                evidence_issues=("packet_too_weak",),
            ),
            "h:drop:initial_only": InitialOnlyRereviewDecision(
                query_group_id="h:drop:initial_only",
                dataset="h_wang",
                action="drop_query",
                reason_bucket="feature_contract_failure",
                safe_component_key_texts=(),
                reviewed_row_count=1,
                review_decisions=("abstain",),
                evidence_issues=("raw_query_author_needed",),
            ),
        },
    )
    per_file_summaries = [
        {
            "path": "test/hwang_eval_rows.csv.gz",
            "positive_rows_before": 1,
            "positive_rows_after": 1,
            "positive_groups_after": 1,
        }
    ]

    summary = rebuild_stack._apply_hwang_candidate_level_label_overrides(
        bundle_root=bundle_root,
        selected_row_paths=(rebuild_stack.HWANG_ROW_RELATIVE_PATH,),
        per_file_summaries=per_file_summaries,
    )

    assert summary is not None
    assert summary["manifest_queries_dropped_by_initial_only_rereview"] == 1
    rewritten_manifest = pd.read_csv(manifest_path)
    assert rewritten_manifest["query_group_id"].tolist() == ["h:q1", "h:force:initial_only"]
    rewritten_rows = pd.read_csv(hwang_rows_path, compression="gzip")
    assert rewritten_rows.set_index("query_group_id")["label"].to_dict()["h:force:initial_only"] == 0


def test_dataset_contract_slice_comparison_accepts_matching_decisions() -> None:
    """Centralized label-contract decisions should compare cleanly against active rows."""

    ledger = pd.DataFrame(
        [
            {
                "slice_key": "demo",
                "query_group_id": "q1",
                "candidate_component_key": "c1",
                "target_label": 1,
                "action": "candidate_positive",
            },
            {
                "slice_key": "demo",
                "query_group_id": "q1",
                "candidate_component_key": "c2",
                "target_label": 0,
                "action": "candidate_negative",
            },
            {
                "slice_key": "demo",
                "query_group_id": "q2",
                "candidate_component_key": "",
                "target_label": 0,
                "action": "force_no_positive",
            },
            {
                "slice_key": "demo",
                "query_group_id": "q3",
                "candidate_component_key": "",
                "target_label": 1,
                "action": "query_target_from_surviving_labels",
            },
        ]
    )
    active_rows = pd.DataFrame(
        [
            {"query_group_id": "q1", "candidate_component_key": "c1", "label": 1},
            {"query_group_id": "q1", "candidate_component_key": "c2", "label": 0},
            {"query_group_id": "q2", "candidate_component_key": "c3", "label": 0},
            {"query_group_id": "q3", "candidate_component_key": "c4", "label": 1},
        ]
    )

    comparison = _compare_ledger_slice(ledger=ledger, slice_key="demo", active_rows=active_rows)

    assert comparison["fatal_mismatch_count"] == 0
    assert comparison["candidate_positive_checks"] == 1
    assert comparison["candidate_negative_checks"] == 1
    assert comparison["force_no_positive_checks"] == 1
    assert comparison["query_target_checks"] == 1


def test_name_compat_manual_positive_corrections_replace_stale_no_positive(tmp_path: Path) -> None:
    """Manual name-compatible positives should override stale no-positive query rows."""

    correction_path = tmp_path / NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH
    correction_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ledger_source": "name_compat_manual_positive_corrections",
                "slice_key": "s_lee_eval",
                "dataset": "s_lee",
                "split": "dev",
                "query_group_id": "s_lee:1:full",
                "query_view": "full",
                "decision_scope": "candidate",
                "candidate_component_key": "su , suk_1",
                "target_label": 1,
                "action": "candidate_positive",
                "reason_bucket": "manual_name_compat_exact_orcid",
                "review_source_path": str(NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH).replace("/", "\\"),
                "notes": "Manual review: shared ORCID.",
            }
        ]
    ).to_csv(correction_path, index=False)
    active_rows = [
        {
            "ledger_source": "s_lee_eval_active_labels",
            "slice_key": "s_lee_eval",
            "dataset": "s_lee",
            "split": "dev",
            "query_group_id": "s_lee:1:full",
            "query_view": "full",
            "decision_scope": "query",
            "candidate_component_key": "",
            "target_label": 0,
            "action": "force_no_positive",
            "reason_bucket": "active_reviewed_row_labels",
            "review_source_path": "test\\s_lee_eval_rows.csv.gz",
            "notes": "",
        },
        {
            "ledger_source": "s_lee_eval_active_labels",
            "slice_key": "s_lee_eval",
            "dataset": "s_lee",
            "split": "dev",
            "query_group_id": "s_lee:2:full",
            "query_view": "full",
            "decision_scope": "query",
            "candidate_component_key": "",
            "target_label": 0,
            "action": "force_no_positive",
            "reason_bucket": "active_reviewed_row_labels",
            "review_source_path": "test\\s_lee_eval_rows.csv.gz",
            "notes": "",
        },
    ]

    corrections = _build_name_compat_manual_positive_correction_ledger(tmp_path)
    corrected_rows = _apply_name_compat_manual_positive_corrections(active_rows, corrections)

    assert [(row["query_group_id"], row["action"]) for row in corrected_rows] == [
        ("s_lee:2:full", "force_no_positive"),
        ("s_lee:1:full", "candidate_positive"),
    ]
    assert corrected_rows[1]["candidate_component_key"] == "su , suk_1"


def test_name_compat_manual_positive_corrections_replace_stale_query_targets() -> None:
    """Manual name-compatible positives should replace stale query targets and duplicate active positives."""

    active_rows = [
        {
            "ledger_source": "hwang_eval_active_labels",
            "slice_key": "hwang_eval",
            "dataset": "h_wang",
            "split": "dev",
            "query_group_id": "h_wang:1:full",
            "query_view": "full",
            "decision_scope": "query",
            "candidate_component_key": "",
            "target_label": 0,
            "action": "query_target_from_surviving_labels",
            "reason_bucket": "active_reviewed_row_labels",
            "review_source_path": "test\\hwang_eval_rows.csv.gz",
            "notes": "",
        },
        {
            "ledger_source": "hwang_eval_active_labels",
            "slice_key": "hwang_eval",
            "dataset": "h_wang",
            "split": "dev",
            "query_group_id": "h_wang:1:full",
            "query_view": "full",
            "decision_scope": "candidate",
            "candidate_component_key": "hai_1",
            "target_label": 1,
            "action": "candidate_positive",
            "reason_bucket": "active_reviewed_row_labels",
            "review_source_path": "test\\hwang_eval_rows.csv.gz",
            "notes": "",
        },
    ]
    corrections = [
        {
            "ledger_source": "name_compat_manual_positive_corrections",
            "slice_key": "hwang_eval",
            "dataset": "h_wang",
            "split": "dev",
            "query_group_id": "h_wang:1:full",
            "query_view": "full",
            "decision_scope": "candidate",
            "candidate_component_key": "hai_1",
            "target_label": 1,
            "action": "candidate_positive",
            "reason_bucket": "manual_name_compat_exact_orcid_non_self",
            "review_source_path": "scratch\\audit.csv",
            "notes": "verified",
        }
    ]

    corrected_rows = _apply_name_compat_manual_positive_corrections(active_rows, corrections)

    assert corrected_rows == corrections


def test_promote_name_compat_rows_overlays_labels_without_orcid_leakage() -> None:
    """Generated ORCID labels should be replaced by official positives plus manual corrections."""

    generated_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "candidate_component_key": "old_positive",
                "retrieval_rank": 1,
                "label": 0,
                "cluster_size": 3,
                "family_id": "ana",
                "dominant_first_name": "ana",
                "query_first_token": "ana",
            },
            {
                "query_group_id": "q1",
                "candidate_component_key": "raw_orcid_only",
                "retrieval_rank": 2,
                "label": 1,
                "cluster_size": 4,
                "family_id": "ann",
                "dominant_first_name": "ann",
                "query_first_token": "ana",
            },
            {
                "query_group_id": "q2",
                "candidate_component_key": "manual_positive",
                "retrieval_rank": 1,
                "label": 0,
                "cluster_size": 2,
                "family_id": "bo",
                "dominant_first_name": "bo",
                "query_first_token": "bo",
            },
        ]
    )
    official_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "candidate_component_key": "old_positive",
                "retrieval_rank": 1,
                "label": 1,
                "base_group_id": "b1",
            },
            {
                "query_group_id": "q2",
                "candidate_component_key": "stale_negative",
                "retrieval_rank": 1,
                "label": 0,
                "base_group_id": "b2",
            },
        ]
    )
    corrections = pd.DataFrame(
        [
            {
                "slice_key": "demo_eval",
                "query_group_id": "q2",
                "candidate_component_key": "manual_positive",
                "target_label": "1",
                "action": "candidate_positive",
            }
        ]
    )

    promoted, summary = promote_name_compat.promote_slice_rows(
        generated_rows=generated_rows,
        official_rows=official_rows,
        corrections=corrections,
        fieldnames=[
            "query_group_id",
            "candidate_component_key",
            "retrieval_rank",
            "label",
            "base_group_id",
            "positive_candidate_count",
            "positive_candidate_keys",
            "group_has_positive",
            "best_positive_retrieval_rank",
            "named_signature_count",
            "confident_family_flag",
            "first_name_expansion_compatibility",
        ],
        slice_key="demo_eval",
        materialize=False,
    )

    labels = dict(
        zip(
            promoted["candidate_component_key"].astype(str),
            promoted["label"].astype(int),
            strict=True,
        )
    )
    assert labels == {"old_positive": 1, "raw_orcid_only": 0, "manual_positive": 1}
    assert promoted.groupby("query_group_id")["base_group_id"].first().to_dict() == {"q1": "b1", "q2": "b2"}
    assert summary["generated_raw_positive_rows"] == 1
    assert summary["name_compat_correction_pairs"] == 1
    assert summary["promoted_positive_rows"] == 2


def test_promote_name_compat_rows_preserves_official_positive_missing_generated() -> None:
    """Known official positives missed by the promoted retriever should stay visible and counted."""

    generated_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "candidate_component_key": "new_top1",
                "retrieval_rank": 1,
                "label": 0,
                "cluster_size": 3,
                "family_id": "c1",
                "dominant_first_name": "amy",
                "query_first_token": "amy",
            }
        ]
    )
    official_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "candidate_component_key": "old_positive",
                "retrieval_rank": 2,
                "label": 1,
                "base_group_id": "b1",
                "cluster_size": 2,
                "family_id": "c2",
                "dominant_first_name": "amy",
                "query_first_token": "amy",
            }
        ]
    )

    promoted, summary = promote_name_compat.promote_slice_rows(
        generated_rows=generated_rows,
        official_rows=official_rows,
        corrections=pd.DataFrame(),
        fieldnames=[
            "query_group_id",
            "candidate_component_key",
            "retrieval_rank",
            "label",
            "base_group_id",
            "positive_candidate_count",
            "positive_candidate_keys",
            "group_has_positive",
            "best_positive_retrieval_rank",
        ],
        slice_key="demo_eval",
        materialize=False,
    )

    assert promoted["candidate_component_key"].astype(str).tolist() == ["new_top1", "old_positive"]
    assert promoted["label"].astype(int).tolist() == [0, 1]
    assert summary["required_positive_pairs_missing_generated"] == 1
    assert summary["official_positive_rows_appended"] == 1


def test_promote_name_compat_rows_can_trust_extra_training_positives() -> None:
    """Training promotion can opt into generated positives without protecting self rows."""

    generated_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "official_positive",
                "retrieval_rank": 1,
                "label": 0,
                "cluster_size": 3,
                "family_id": "ann",
                "dominant_first_name": "ann",
                "query_first_token": "ann",
            },
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "generated_positive",
                "retrieval_rank": 2,
                "label": 1,
                "cluster_size": 4,
                "family_id": "ana",
                "dominant_first_name": "ana",
                "query_first_token": "ann",
            },
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "self_generated_positive",
                "retrieval_rank": 3,
                "label": 1,
                "cluster_size": 4,
                "family_id": "ann",
                "dominant_first_name": "ann",
                "query_first_token": "ann",
            },
        ]
    )
    official_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "official_positive",
                "retrieval_rank": 1,
                "label": 1,
                "base_group_id": "b1",
            }
        ]
    )

    promoted, summary = promote_name_compat.promote_slice_rows(
        generated_rows=generated_rows,
        official_rows=official_rows,
        corrections=pd.DataFrame(),
        fieldnames=[
            "query_group_id",
            "query_signature_id",
            "candidate_component_key",
            "retrieval_rank",
            "label",
            "base_group_id",
            "positive_candidate_count",
            "positive_candidate_keys",
            "group_has_positive",
            "best_positive_retrieval_rank",
        ],
        slice_key="demo_train",
        extra_positive_pairs=promote_name_compat._positive_pairs_from_rows(generated_rows),
        component_signature_ids={
            "official_positive": ("s2",),
            "generated_positive": ("s3",),
            "self_generated_positive": ("s1", "s4"),
        },
        materialize=False,
    )

    labels = dict(
        zip(
            promoted["candidate_component_key"].astype(str),
            promoted["label"].astype(int),
            strict=True,
        )
    )
    assert labels == {"official_positive": 1, "generated_positive": 1}
    assert summary["self_containing_rows_dropped"] == 1
    assert summary["extra_positive_pairs"] == 2
    assert summary["extra_positive_pairs_survived"] == 1
    assert summary["promoted_positive_rows"] == 2


def test_promote_name_compat_rows_preserves_generated_residual_holdout() -> None:
    """A generated own-cluster residual row is not a self-link when the query was held out."""

    generated_rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "own_cluster_residual",
                "retrieval_rank": 1,
                "label": 1,
                "query_in_seed_before_holdout": 1,
                "cluster_size": 1,
                "family_id": "ann",
                "dominant_first_name": "ann",
                "query_first_token": "ann",
            }
        ]
    )
    official_rows = generated_rows.assign(label=0).copy()

    promoted, summary = promote_name_compat.promote_slice_rows(
        generated_rows=generated_rows,
        official_rows=official_rows,
        corrections=pd.DataFrame(),
        fieldnames=[
            "query_group_id",
            "query_signature_id",
            "candidate_component_key",
            "retrieval_rank",
            "label",
            "positive_candidate_count",
            "positive_candidate_keys",
            "group_has_positive",
            "best_positive_retrieval_rank",
        ],
        slice_key="demo_train",
        extra_positive_pairs=promote_name_compat._positive_pairs_from_rows(generated_rows),
        component_signature_ids={"own_cluster_residual": ("s1", "s4")},
        materialize=False,
    )

    assert promoted["label"].astype(int).tolist() == [1]
    assert summary["self_containing_rows_dropped"] == 0
    assert summary["self_containing_residual_rows_preserved"] == 1
    assert summary["extra_positive_pairs_survived"] == 1


def test_promote_name_compat_rows_can_preserve_required_eval_self_positive() -> None:
    """Eval/calibration promotion can keep legacy required positives for accounting."""

    rows = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "query_signature_id": "s1",
                "candidate_component_key": "legacy_positive",
                "retrieval_rank": 1,
                "label": 1,
            }
        ]
    )

    promoted, summary = promote_name_compat.promote_slice_rows(
        generated_rows=rows,
        official_rows=rows,
        corrections=pd.DataFrame(),
        fieldnames=["query_group_id", "query_signature_id", "candidate_component_key", "retrieval_rank", "label"],
        slice_key="demo_eval",
        component_signature_ids={"legacy_positive": ("s1", "s2")},
        preserve_required_self_containing=True,
        materialize=False,
    )

    assert promoted["label"].astype(int).tolist() == [1]
    assert summary["self_containing_rows_dropped"] == 0
    assert summary["self_containing_required_rows_preserved"] == 1


def test_promote_name_compat_rows_normalizes_float_year_text() -> None:
    """Promotion should feed integer year text to derived feature materialization."""

    rows = pd.DataFrame(
        [
            {
                "query_year": "2025.0",
                "candidate_year_min": 2019.0,
                "candidate_year_max": "2026.0",
            }
        ]
    )

    enriched = promote_name_compat._enrich_promoted_prerequisites(rows)

    assert enriched.loc[0, "query_year"] == "2025"
    assert enriched.loc[0, "candidate_year_min"] == "2019"
    assert enriched.loc[0, "candidate_year_max"] == "2026"


def test_promote_name_compat_rows_requires_corrections_in_generated_surface() -> None:
    """Manual name-compat corrections are only valid when the promoted retrieval surface contains them."""

    generated_rows = pd.DataFrame(
        [{"query_group_id": "q1", "candidate_component_key": "negative", "retrieval_rank": 1, "label": 0}]
    )
    official_rows = pd.DataFrame(
        [{"query_group_id": "q1", "candidate_component_key": "negative", "retrieval_rank": 1, "label": 0}]
    )
    corrections = pd.DataFrame(
        [
            {
                "slice_key": "demo_eval",
                "query_group_id": "q1",
                "candidate_component_key": "manual_positive",
                "target_label": "1",
                "action": "candidate_positive",
            }
        ]
    )

    with pytest.raises(ValueError, match="corrections missing from generated rows"):
        promote_name_compat.promote_slice_rows(
            generated_rows=generated_rows,
            official_rows=official_rows,
            corrections=corrections,
            fieldnames=["query_group_id", "candidate_component_key", "retrieval_rank", "label"],
            slice_key="demo_eval",
            materialize=False,
        )


def test_promote_name_compat_syncs_hwang_companion_targets(tmp_path: Path) -> None:
    """H-Wang clean and candidate manifests should track promoted row-level positives."""

    output_root = tmp_path / "bundle"
    test_root = output_root / "test"
    test_root.mkdir(parents=True)
    hwang_rows_path = test_root / "hwang_eval_rows.csv.gz"
    with gzip.open(hwang_rows_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_group_id", "candidate_component_key", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"query_group_id": "h:q1", "candidate_component_key": "new_positive", "label": 1},
                {"query_group_id": "h:q1", "candidate_component_key": "old_reviewed", "label": 0},
                {"query_group_id": "h:q2", "candidate_component_key": "negative", "label": 0},
            ]
        )
    promoted_rows = pd.read_csv(hwang_rows_path, compression="gzip")
    (test_root / "hwang_cleaned_eval_overrides.csv").write_text(
        "\n".join(
            [
                "query_group_id,manual_safe_target,manual_assessment,correction_type,review_source_path",
                "h:q1,0,force_no_positive,should_abstain,scratch/review.tsv",
                "h:q2,1,keep_surviving_raw_labels,none,scratch/review.tsv",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (test_root / "hwang_candidate_level_label_overrides.csv").write_text(
        "\n".join(
            [
                "query_group_id,dataset,correction_type,reviewed_candidate_component_key,"
                "reviewed_candidate_survived,raw_positive_rows_before_candidate_relabel,label_action,"
                "review_source_path,positive_rows_after_candidate_relabel,manual_safe_target",
                "h:q1,h_wang,should_abstain,old_reviewed,0,0,force_no_positive,scratch/review.tsv,0,0",
                "h:q2,h_wang,none,,0,1,keep_surviving_raw_labels,scratch/review.tsv,1,1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = promote_name_compat._sync_hwang_candidate_level_files(
        output_root=output_root,
        promoted_rows=promoted_rows,
        name_compat_correction_pairs={("h:q1", "new_positive")},
        dry_run=False,
    )

    clean_targets = pd.read_csv(test_root / "hwang_cleaned_eval_overrides.csv").set_index("query_group_id")
    manifest = pd.read_csv(test_root / "hwang_candidate_level_label_overrides.csv").set_index("query_group_id")
    assert clean_targets["manual_safe_target"].to_dict() == {"h:q1": 1, "h:q2": 0}
    assert manifest.loc["h:q1", "label_action"] == "name_compat_manual_positive"
    assert manifest.loc["h:q1", "manual_safe_target"] == 1
    assert manifest.loc["h:q2", "manual_safe_target"] == 0
    assert summary["positive_queries_after_candidate_relabel"] == 1
    assert summary["name_compat_manual_positive_rows"] == 1
    _summarize_hwang_candidate_level_label_consistency(
        hwang_rows_path=hwang_rows_path,
        clean_overrides_path=test_root / "hwang_cleaned_eval_overrides.csv",
        manifest_path=test_root / "hwang_candidate_level_label_overrides.csv",
    )


def test_dataset_worker_uses_from_dataset_for_slee_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S. Lee worker should avoid the unstable Windows native JSON ingest path."""

    monkeypatch.setattr(rebuild_stack, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(rebuild_stack.os, "name", "nt", raising=False)
    calls: list[dict[str, str]] = []

    def fake_run(command: list[str], *, check: bool, cwd: str, env: dict[str, str]) -> None:
        del command, check, cwd
        calls.append(dict(env))
        rebuild_stack._worker_summary_path("s_lee").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(rebuild_stack.subprocess, "run", fake_run)
    monkeypatch.setattr(
        rebuild_stack,
        "_merge_worker_summary",
        lambda *, dataset_name, file_summary_states, summary_path: None,
    )

    rebuild_stack._run_dataset_worker_subprocess(
        dataset_name="s_lee",
        spool_db_path=tmp_path / "spool.sqlite3",
        pair_batch_size=10,
        query_batch_pair_limit=20,
        file_summary_states={},
    )

    assert len(calls) == 1
    assert calls[0][rebuild_stack.RUST_BUILD_PATH_ENV] == "from_dataset"


def test_dataset_worker_retries_access_violation_with_from_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native access violations should get one Rust from_dataset retry."""

    monkeypatch.setattr(rebuild_stack, "TELEMETRY_DIR", tmp_path)
    calls: list[dict[str, str]] = []

    def fake_run(command: list[str], *, check: bool, cwd: str, env: dict[str, str]) -> None:
        del command, check, cwd
        calls.append(dict(env))
        if len(calls) == 1:
            raise rebuild_stack.subprocess.CalledProcessError(3221225477, "worker")
        rebuild_stack._worker_summary_path("a_silva").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(rebuild_stack.subprocess, "run", fake_run)
    monkeypatch.setattr(
        rebuild_stack,
        "_merge_worker_summary",
        lambda *, dataset_name, file_summary_states, summary_path: None,
    )

    rebuild_stack._run_dataset_worker_subprocess(
        dataset_name="a_silva",
        spool_db_path=tmp_path / "spool.sqlite3",
        pair_batch_size=10,
        query_batch_pair_limit=20,
        file_summary_states={},
    )

    assert len(calls) == 2
    assert calls[0].get(rebuild_stack.RUST_BUILD_PATH_ENV) != "from_dataset"
    assert calls[1][rebuild_stack.RUST_BUILD_PATH_ENV] == "from_dataset"


def test_dataset_contract_slice_comparison_rejects_wrong_candidate_label() -> None:
    """The contract comparison should fail if a candidate-level custom label drifts."""

    ledger = pd.DataFrame(
        [
            {
                "slice_key": "demo",
                "query_group_id": "q1",
                "candidate_component_key": "c1",
                "target_label": 1,
                "action": "candidate_positive",
            }
        ]
    )
    active_rows = pd.DataFrame([{"query_group_id": "q1", "candidate_component_key": "c1", "label": 0}])

    comparison = _compare_ledger_slice(ledger=ledger, slice_key="demo", active_rows=active_rows)

    assert comparison["fatal_mismatch_count"] == 1
    assert comparison["mismatches"]["candidate_positive_wrong_label"] == [
        {"query_group_id": "q1", "candidate_component_key": "c1"}
    ]


def test_fit_score_margin_gate_finds_perfect_gate() -> None:
    """Score+margin fitting should recover a gate with perfect balanced accuracy when available."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q1",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.2,
            },
            {
                "query_case_id": "q2",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.05,
            },
        ]
    )
    fitted = _fit_score_margin_gate(
        query_choices,
        reference_score_threshold=0.8,
        reference_margin_threshold=0.15,
        score_grid_size=5,
        margin_grid_size=5,
    )
    assert fitted["metrics"]["balanced_accuracy"] == 1.0


def test_fit_single_candidate_score_gate_finds_perfect_gate() -> None:
    """Single-candidate gate fitting should learn a score-only abstain threshold."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_pos",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_neg",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
        ]
    )

    fitted = _fit_single_candidate_score_gate(
        query_choices,
        reference_score_threshold=0.5,
        score_grid_size=5,
    )

    assert fitted["metrics"]["balanced_accuracy"] == 1.0
    assert 0.1 < fitted["single_candidate_score_threshold"] <= 0.9


def test_apply_classic_gate_uses_single_candidate_threshold() -> None:
    """Missing runner-up margin should use the calibrated single-candidate gate."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_neg_single",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_pos_single",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": np.nan,
                "has_runner_up": 0,
            },
            {
                "query_case_id": "q_multi",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.4,
                "score_margin": 0.05,
                "has_runner_up": 1,
            },
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.8,
        margin_threshold=0.1,
        single_candidate_score_threshold=0.5,
    ).set_index("query_case_id")

    assert predictions.loc["q_neg_single", "predicted_action"] == "abstain"
    assert predictions.loc["q_neg_single", "correct"] == 1
    assert predictions.loc["q_pos_single", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_pos_single", "correct"] == 1
    assert predictions.loc["q_multi", "predicted_action"] == "abstain"


def test_apply_classic_gate_supports_bucketed_thresholds() -> None:
    """Fixed bucketed gates should use candidate kind and first-name length."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "q_multi_margin_rescue",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.2,
                "score_margin": 0.6,
                "has_runner_up": 1,
                "first_name_bucket": "multi_letter_first",
            },
            {
                "query_case_id": "q_single_letter_margin_not_rescued",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.2,
                "score_margin": 0.6,
                "has_runner_up": 1,
                "first_name_bucket": "single_letter_first",
            },
            {
                "query_case_id": "q_single_letter_accept",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.5,
                "score_margin": np.nan,
                "has_runner_up": 0,
                "first_name_bucket": "single_letter_first",
            },
            {
                "query_case_id": "q_single_multi_abstain",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.5,
                "score_margin": np.nan,
                "has_runner_up": 0,
                "first_name_bucket": "multi_letter_first",
            },
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.8,
        margin_threshold=0.4,
        single_candidate_score_threshold=0.7,
        bucketed_score_thresholds={
            "multi_candidate|multi_letter_first": 0.9,
            "multi_candidate|single_letter_first": 0.3,
            "single_candidate|multi_letter_first": 0.7,
            "single_candidate|single_letter_first": 0.4,
        },
        bucketed_margin_threshold=0.4,
        bucketed_margin_thresholds={
            "multi_candidate|multi_letter_first": 0.4,
            "multi_candidate|single_letter_first": 0.7,
        },
    ).set_index("query_case_id")

    assert predictions.loc["q_multi_margin_rescue", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_single_letter_margin_not_rescued", "predicted_action"] == "abstain"
    assert predictions.loc["q_single_letter_accept", "predicted_action"] == "link_candidate"
    assert predictions.loc["q_single_multi_abstain", "predicted_action"] == "abstain"


def test_promoted_total_error_gate_selects_on_check_errors() -> None:
    """Promoted non-fixed gate calibration should fit on one split and select on check."""

    choices = pd.DataFrame(
        [
            {
                "query_case_id": "fit_pos",
                "split": "calibration_fit",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.2,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "fit_neg",
                "split": "calibration_fit",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.01,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "check_pos",
                "split": "calibration_check",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.85,
                "score_margin": 0.15,
                "has_runner_up": 1,
            },
            {
                "query_case_id": "check_neg",
                "split": "calibration_check",
                "candidate_kind": "multi_candidate",
                "first_name_bucket": "multi_letter_first",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.05,
                "score_margin": 0.0,
                "has_runner_up": 1,
            },
        ]
    )
    config = {
        "mode": "total_error_4score_2margin",
        "fit_split": "calibration_fit",
        "selection_split": "calibration_check",
        "test_split": "test",
        "score_grid_size": 7,
        "margin_grid_size": 7,
        "lambda_grid": [0.0],
        "reference_score_thresholds": {
            "multi_candidate|multi_letter_first": 0.5,
            "multi_candidate|single_letter_first": 0.5,
            "single_candidate|multi_letter_first": 0.5,
            "single_candidate|single_letter_first": 0.5,
        },
        "reference_margin_thresholds": {
            "multi_candidate|multi_letter_first": 0.5,
            "multi_candidate|single_letter_first": 0.5,
        },
    }

    result = _fit_promoted_stratified_total_error_gate(choices, config)

    assert result["gate"].name == "total_error_4score_2margin_lambda_0"
    assert result["check_metrics"]["errors"] == 0
    assert result["fit_metrics"]["errors"] == 0


def test_apply_classic_gate_requires_positive_query_target_for_correct_link() -> None:
    """A stale positive candidate row should not overcredit a query-level manual negative."""

    query_choices = pd.DataFrame(
        [
            {
                "query_case_id": "manual_negative_with_stale_positive_row",
                "query_safe_target": 0,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.99,
                "score_margin": 0.9,
                "has_runner_up": 1,
            }
        ]
    )

    predictions = _apply_classic_gate(
        query_choices,
        score_threshold=0.5,
        margin_threshold=0.5,
    )

    assert predictions.iloc[0]["predicted_action"] == "link_candidate"
    assert predictions.iloc[0]["correct"] == 0


def test_score_abstain_rule_uses_repaired_query_targets() -> None:
    """Internal gate diagnostics should follow final targets, not stale supervision type."""

    rows = pd.DataFrame(
        [
            {
                "query_case_id": "q_repaired_singleton",
                "dataset": "demo",
                "supervision_type": "negative_singleton_orcid",
                "query_safe_target": 1,
                "chosen_candidate_target": 1,
                "chosen_probability": 0.9,
                "score_margin": 0.8,
                "has_runner_up": 1,
                "top1_correct": 1,
            },
            {
                "query_case_id": "q_true_negative",
                "dataset": "demo",
                "supervision_type": "negative_singleton_orcid",
                "query_safe_target": 0,
                "chosen_candidate_target": 0,
                "chosen_probability": 0.1,
                "score_margin": 0.1,
                "has_runner_up": 1,
                "top1_correct": 0,
            },
        ]
    )

    metrics = _score_abstain_rule(rows, score_threshold=0.5, margin_threshold=0.5)

    assert metrics["positive_queries"] == 1
    assert metrics["negative_queries"] == 1
    assert metrics["balanced_accuracy"] == 1.0


def test_apply_classic_train_row_cap_preserves_positive_queries() -> None:
    """Classic row-cap filtering should keep every positive query represented."""

    train_df = pd.DataFrame(
        [
            {"query_group_id": "q1", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q1", "retrieval_rank": 2, "label": 0},
            {"query_group_id": "q1", "retrieval_rank": 6, "label": 1},
            {"query_group_id": "q1", "retrieval_rank": 8, "label": 0},
            {"query_group_id": "q2", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q2", "retrieval_rank": 3, "label": 1},
            {"query_group_id": "q2", "retrieval_rank": 7, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 1, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 5, "label": 0},
            {"query_group_id": "q3", "retrieval_rank": 7, "label": 0},
        ]
    )
    selected, summary = _apply_classic_train_row_cap(
        train_df,
        rule_name="max_of_min_limit_and_first_positive_rank",
        min_train_limit=5,
    )
    assert selected.groupby("query_group_id", sort=False)["retrieval_rank"].max().to_dict() == {
        "q1": 6,
        "q2": 3,
        "q3": 5,
    }
    assert summary is not None
    assert summary["lost_positive_queries"] == 0
    assert summary["positive_rows_after"] == 2


def test_apply_classic_train_holdout_filter_removes_eval_identities() -> None:
    """Classic training should drop rows that share query or base IDs with held-out rows."""

    train_df = pd.DataFrame(
        [
            {"query_group_id": "q_keep", "base_group_id": "b_keep", "label": 1},
            {"query_group_id": "q_exact", "base_group_id": "b_train", "label": 1},
            {"query_group_id": "q_base_pos", "base_group_id": "b_eval", "label": 1},
            {"query_group_id": "q_base_neg", "base_group_id": "b_eval", "label": 0},
        ]
    )

    filtered, summary = _apply_classic_train_holdout_filter(
        train_df,
        holdout_query_group_ids={"q_exact"},
        holdout_base_group_ids={"b_eval"},
        holdout_sources=[{"source": "demo", "query_groups": 1, "base_groups": 1}],
    )

    assert filtered["query_group_id"].tolist() == ["q_keep"]
    assert summary["rows_removed"] == 3
    assert summary["queries_removed"] == 3
    assert summary["positive_rows_removed"] == 2
    assert summary["positive_queries_removed"] == 2
    assert summary["overlapping_query_groups"] == 1
    assert summary["overlapping_base_groups"] == 1
    assert summary["holdout_sources"] == [{"source": "demo", "query_groups": 1, "base_groups": 1}]


def test_evaluate_classic_manual_holdout_scores_fresh_candidates() -> None:
    """Classic manual holdout evaluation should use fresh candidate scores, not frozen top1 metadata."""

    manual_holdout = pd.DataFrame(
        [
            {
                "query_case_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "rescue",
                "candidate_component_key": "wrong",
                "retrieval_rank": 1,
                "binary_safe_link_target": 0,
            },
            {
                "query_case_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "rescue",
                "candidate_component_key": "right",
                "retrieval_rank": 2,
                "binary_safe_link_target": 1,
            },
            {
                "query_case_id": "q2",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "easy",
                "candidate_component_key": "negative",
                "retrieval_rank": 1,
                "binary_safe_link_target": 0,
            },
            {
                "query_case_id": "q2",
                "dataset": "demo",
                "query_view": "full",
                "review_bucket": "easy",
                "candidate_component_key": "distractor",
                "retrieval_rank": 2,
                "binary_safe_link_target": 0,
            },
        ]
    )
    summary = _evaluate_classic_manual_holdout(
        manual_holdout,
        probabilities=np.array([0.1, 0.9, 0.1, 0.05], dtype=np.float32),
        score_threshold=0.2,
        margin_threshold=0.2,
    )
    assert summary["overall"]["balanced_accuracy"] == 1.0
    assert summary["by_bucket"]["rescue"]["positive_recall"] == 1.0
    assert summary["by_bucket"]["easy"]["negative_recall"] == 1.0


def test_augmented_feature_matrix_respects_ablation_columns() -> None:
    """Augmented feature matrices should honor ablated feature lists and keep query-view dummies."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "query_view": "full",
                "title_overlap": 0.25,
                "cluster_size": 17,
                "model_score": 0.91,
                "gap_to_top1": 0.01,
                "candidate_rank": 2,
            }
        ]
    )
    features = _augmented_feature_matrix(
        df,
        feature_columns=(
            "title_overlap",
            "cluster_size",
            "query_first_prefix_match",
            "query_view",
        ),
    )
    assert list(features.columns) == [
        "title_overlap",
        "cluster_size",
        "query_first_prefix_match",
        "query_view__full",
        "query_view__initial_only",
    ]


def test_classic_feature_matrix_supports_augmented_union_features() -> None:
    """Classic feature matrices should derive declared runtime features only."""

    df = pd.DataFrame(
        [
            {
                "query_author": "Hanbing Wang",
                "dominant_first_name": "hanbing",
                "query_view": "full",
                "title_overlap": 0.25,
                "cluster_size": 17,
                "count_normalized_confidence": 0.6,
            }
        ]
    )
    features = _classic_feature_matrix(
        df,
        (
            "title_overlap",
            "cluster_size",
            "cluster_size_log_capped",
            "query_view__full",
            "query_view__initial_only",
        ),
    )
    assert list(features.columns) == [
        "title_overlap",
        "cluster_size",
        "cluster_size_log_capped",
        "query_view__full",
        "query_view__initial_only",
    ]
    assert features.iloc[0]["title_overlap"] == 0.25
    assert features.iloc[0]["cluster_size"] == 17.0
    assert features.iloc[0]["cluster_size_log_capped"] > 0.0
    assert features.iloc[0]["query_view__full"] == 1.0
    assert features.iloc[0]["query_view__initial_only"] == 0.0


def test_classic_feature_matrix_rejects_missing_required_features() -> None:
    """Absent non-runtime features should fail instead of becoming zero-valued signals."""

    df = pd.DataFrame([{"title_overlap": 0.25}])

    with pytest.raises(ValueError, match="missing required feature inputs"):
        _classic_feature_matrix(df, ("title_overlap", "cluster_size"))


def test_classic_feature_matrix_rejects_missing_feature_cells() -> None:
    """Present active features with missing values should fail instead of becoming zero-valued signals."""

    df = pd.DataFrame([{"title_overlap": None, "cluster_size": 17}])

    with pytest.raises(ValueError, match="missing/non-numeric feature values"):
        _classic_feature_matrix(df, ("title_overlap", "cluster_size"))


def test_raw_similarity_features_exclude_query_signature_from_candidate_component() -> None:
    """Raw metadata features must not leak self-similarity from residual LOO rows."""

    query_signature = SimpleNamespace(
        paper_id="p1",
        author_info_last="Smith",
        author_info_affiliations=["one two three"],
    )
    self_signature = SimpleNamespace(
        paper_id="p1",
        author_info_last="Smith",
        author_info_affiliations=["one two three"],
    )
    candidate_signature = SimpleNamespace(
        paper_id="p2",
        author_info_last="Smith",
        author_info_affiliations=["one four"],
    )
    unrelated_signature = SimpleNamespace(
        paper_id="p3",
        author_info_last="Jones",
        author_info_affiliations=["eight nine"],
    )
    query_paper = SimpleNamespace(
        paper_id="p1",
        title="alpha beta",
        authors=[
            SimpleNamespace(author_name="Alice Smith"),
            SimpleNamespace(author_name="Carol Jones"),
        ],
    )
    candidate_paper = SimpleNamespace(
        paper_id="p2",
        title="alpha epsilon",
        authors=[
            SimpleNamespace(author_name="Alice Smith"),
            SimpleNamespace(author_name="Carol Jones"),
            SimpleNamespace(author_name="Dan Brown"),
        ],
    )
    unrelated_paper = SimpleNamespace(
        paper_id="p3",
        title="zeta eta",
        authors=[SimpleNamespace(author_name="Eve Stone")],
    )
    resources = rebuild_stack.DatasetResources(
        dataset_name="demo",
        dataset=SimpleNamespace(
            signatures={
                "q": query_signature,
                "self": self_signature,
                "m": candidate_signature,
                "n": unrelated_signature,
            },
            papers={"p1": query_paper, "p2": candidate_paper, "p3": unrelated_paper},
        ),
        runtime_context=None,
        constraint_backend=None,
        component_signatures={"c1": ["self", "m", "n"]},
        raw_paper_text_by_id={
            "p1": "alpha beta gamma delta",
            "p2": "alpha beta epsilon",
            "p3": "zeta eta theta",
        },
    )

    features = raw_similarity_features_by_component(
        dataset=resources.dataset,
        query_signature_id="q",
        candidate_signature_ids_by_component={"c1": ["q", "m", "n"]},
        raw_paper_text_by_id=resources.raw_paper_text_by_id,
        cache=resources.raw_similarity_feature_cache,
    )

    assert features["c1"]["raw_max_affiliation_jaccard"] == pytest.approx(0.25)
    assert features["c1"]["raw_max_coauthor_jaccard"] == pytest.approx(0.5)
    assert features["c1"]["raw_max_title_jaccard"] == pytest.approx(1 / 3)
    assert features["c1"]["raw_max_text_jaccard"] == pytest.approx(0.4)


def test_load_bundle_defaults_to_active_20260428p_bundle() -> None:
    """The shared stack should default to the active 20260428p bundle."""

    bundle = load_bundle()
    assert DEFAULT_PACKAGE_DIR.name == "joint_safe_link_official_stack_20260428p"
    assert bundle.root == DEFAULT_PACKAGE_DIR
    assert bundle.bundle_name == "joint_safe_link_official_stack_20260428p"
    assert "manual_holdout_candidates_path" not in bundle.models["classic"]
    assert "monotone_constraints" not in bundle.models["classic"]
    assert bundle.models["classic"]["feature_columns"] == [
        "min_distance",
        "retrieval_score_gap_vs_best_competitor",
        "specter_exemplar_similarity",
        "top5_distance_best_gap",
        "retrieval_score",
        "affiliation_contradiction_severity",
        "venue_overlap_rank_fraction",
        "coarse_family_top5_best_gap",
        "same_family_as_best_top5",
        "same_family_as_heuristic_choice",
        "same_family_as_top1",
        "query_first_prefix_match",
        "retrieval_score_best_gap",
        "specter_exemplar_rank_fraction",
        "cluster_size_log_capped",
        "anchor_evidence_count",
        "strong_positive_anchor_score",
        "weak_residual_anchor_score",
        "sparse_relative_winner_score",
        "query_view__initial_only",
        "last_name_count_min_rarity",
        "candidate_last_name_count_min_rarity",
        "candidate_last_first_name_count_min_rarity",
        "last_first_name_count_min_rarity",
        "first_prefix_x_last_first_name_count_min_rarity",
        "raw_max_affiliation_jaccard",
        "raw_max_coauthor_jaccard",
        "raw_max_title_jaccard",
        "raw_max_text_jaccard",
        "affiliation_overlap",
        "venue_overlap",
        "coauthor_overlap",
        "middle_initial_compatibility",
        "title_overlap",
        "exact_anchor_evidence_flag",
        "year_compatibility",
        "year_mismatch_severity",
        "specter_centroid_similarity",
        "top5_mean_distance",
        "distance_spread_top5_minus_min",
        "query_view__full",
        "min_distance_rank_fraction",
        "mean_distance_rank_fraction",
        "top3_distance_rank_fraction",
        "top5_distance_rank_fraction",
        "retrieval_rank_fraction",
        "retrieval_score_rank_fraction",
        "affiliation_overlap_rank_fraction",
        "coauthor_overlap_rank_fraction",
        "year_compatibility_rank_fraction",
        "title_overlap_rank_fraction",
        "specter_centroid_rank_fraction",
    ]
    assert bundle.models["classic"]["extra_eval_paths"] == {
        "j_smith": "test\\j_smith_eval_rows.csv.gz",
        "a_khan": "test\\a_khan_eval_rows.csv.gz",
        "a_silva": "test\\a_silva_eval_rows.csv.gz",
        "s_gupta": "test\\s_gupta_eval_rows.csv.gz",
        "training_s2and_source_reviewed": "test/training_s2and_source_reviewed_eval_rows.csv.gz",
        "s2and_extra_no_positive": "test/s2and_extra_no_positive_eval_rows.csv.gz",
        "s2and_rescue_reviewed": "test\\s2and_rescue_reviewed_eval_rows.csv.gz",
    }
    training_asset = bundle.assets["training"]["classic_train_union21_plus_s_lee_raw"]
    assert int(training_asset["rows"]) == 413866
    assert int(training_asset["queries"]) == 81692
    assert int(training_asset["positive_rows"]) == 25519
    assert training_asset["label_repair_manifest_path"] == ("training\\singleton_near_distance_repair_manifest.csv")
    active_training_asset = bundle.assets["training"][
        "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_neg100_plus_reviewed_splitpos_hardneg"
    ]
    assert int(active_training_asset["rows"]) == 1653165
    assert int(active_training_asset["queries"]) == 82775
    assert int(active_training_asset["positive_rows"]) == 101543
    assert int(active_training_asset["positive_queries"]) == 72011
    assert int(active_training_asset["negative_queries"]) == 10764
    assert "fixed_bucketed_gate" not in bundle.models["classic"]
    promoted_gate = bundle.models["classic"]["promoted_stratified_gate"]
    assert promoted_gate["mode"] == "total_error_4score_2margin"
    assert promoted_gate["fit_split"] == "calibration_fit"
    assert promoted_gate["selection_split"] == "calibration_check"
    assert promoted_gate["reference_score_thresholds"]["multi_candidate|multi_letter_first"] == 0.8255340123176578
    promoted_split = bundle.assets["calibration"]["stratified_eval_test_split"]
    assert promoted_split["calibration_fit_queries"] == 5514
    assert promoted_split["calibration_check_queries"] == 5513
    assert promoted_split["test_queries"] == 9888
    assert bundle.assets["calibration"]["total_error_4score_2margin_gate"]["selected_gate_name"] == (
        "total_error_4score_2margin_lambda_0"
    )
    assert bundle.assets["calibration"]["greedy_best_check23_feature_selection"]["selected_feature_count"] == 23
    assert bundle.assets["calibration"]["best_by_check_minus_new26_feature_selection"]["selected_feature_count"] == 26
    assert bundle.assets["calibration"]["best_by_test_minus_new15_feature_selection"]["selected_feature_count"] == 15
    dataset_contract = bundle.assets["dataset_contract"]
    assert dataset_contract["custom_label_ledger_rows"] == 42079
    assert dataset_contract["comparison_fatal_mismatch_count"] == 0
    assert dataset_contract["label_slice_counts"]["hwang_eval"] == 5169

    assert dataset_contract["label_slice_counts"]["new_block_calibration_source"] == 10639
    assert dataset_contract["label_slice_counts"]["s2and_rescue_reviewed_eval"] == 2000
    assert dataset_contract["label_slice_counts"]["s2and_rescue_reviewed_train"] == 1018
    assert dataset_contract["label_slice_counts"]["s2and_singleton_reviewed_train"] == 424
    assert dataset_contract["label_slice_counts"]["s_park_eval"] == 4660
    assert (bundle.root / "bundle.json").exists()
    assert (bundle.root / "README.md").exists()
    assert (bundle.root / "PROVENANCE.md").exists()
    assert int(bundle.assets["calibration"]["classic_gate_source"]["queries"]) == 8009
    assert int(bundle.assets["calibration"]["classic_gate_source"]["positive_queries"]) == 6210
    assert int(bundle.assets["calibration"]["classic_gate_source"]["negative_queries"]) == 1799
    assert int(bundle.assets["calibration"]["classic_gate_split"]["calibration_groups"]) == 4154
    assert int(bundle.assets["calibration"]["classic_gate_split"]["evaluation_groups"]) == 4151
    assert int(bundle.assets["test"]["s2and_eval"]["rows"]) == 11116
    assert int(bundle.assets["test"]["s2and_eval"]["queries"]) == 1675
    assert int(bundle.assets["test"]["s2and_eval"]["positive_rows"]) == 1593
    assert int(bundle.assets["test"]["s2and_eval"]["positive_queries"]) == 1488
    assert int(bundle.assets["test"]["s2and_eval"]["negative_queries"]) == 187
    assert int(bundle.assets["test"]["hwang_eval"]["positive_rows"]) == 6932
    hwang_candidate_asset = bundle.assets["test"]["hwang_candidate_level_label_overrides"]
    assert int(hwang_candidate_asset["positive_queries_after_candidate_relabel"]) == 4442
    assert int(hwang_candidate_asset["reviewed_positive_corrections_survived"]) == 364
    assert int(bundle.assets["test"]["j_smith_eval"]["queries"]) == 70
    assert int(bundle.assets["test"]["a_khan_eval"]["queries"]) == 149
    assert int(bundle.assets["test"]["a_silva_eval"]["queries"]) == 641
    assert int(bundle.assets["test"]["s_gupta_eval"]["queries"]) == 139
    reviewed_asset = bundle.assets["test"]["training_s2and_source_reviewed_eval"]
    assert int(reviewed_asset["rows"]) == 35
    assert int(reviewed_asset["queries"]) == 13
    assert int(reviewed_asset["positive_rows"]) == 2
    extra_s2and_asset = bundle.assets["test"]["s2and_extra_no_positive_eval"]
    assert int(extra_s2and_asset["rows"]) == 1
    assert int(extra_s2and_asset["queries"]) == 1
    assert int(extra_s2and_asset["positive_rows"]) == 0
    rescue_asset = bundle.assets["test"]["s2and_rescue_reviewed_eval"]
    assert int(rescue_asset["rows"]) == 8510
    assert int(rescue_asset["queries"]) == 1905
    assert int(rescue_asset["positive_rows"]) == 1572
    assert int(rescue_asset["positive_queries"]) == 1541
    assert int(rescue_asset["negative_queries"]) == 364
    assert not (bundle.root / "manifest.json").exists()
    assert not (bundle.root / "model_specs.json").exists()
    assert not (bundle.root / "expected_metrics.json").exists()


def test_rebuild_row_paths_include_active_classic_train_path() -> None:
    """The official rebuild must regenerate the row file the classic model trains on."""

    bundle = load_bundle()
    train_path = Path(str(bundle.models["classic"]["train_path"]).replace("\\", "/"))

    assert train_path in rebuild_stack.ROW_RELATIVE_PATHS
    assert rebuild_stack.S2AND_RESCUE_REVIEWED_ROW_RELATIVE_PATH in rebuild_stack.ROW_RELATIVE_PATHS


def test_classic_monotone_constraints_match_active_feature_order() -> None:
    """Classic monotone constraints should stay aligned with the active feature order."""

    feature_columns = (
        "min_distance",
        "retrieval_score_gap_vs_best_competitor",
        "specter_exemplar_similarity",
        "top5_distance_best_gap",
        "retrieval_score",
        "affiliation_contradiction_severity",
        "venue_overlap_rank_fraction",
        "coarse_family_top5_best_gap",
        "same_family_as_best_top5",
        "same_family_as_heuristic_choice",
        "same_family_as_top1",
        "query_first_prefix_match",
        "retrieval_score_best_gap",
        "specter_exemplar_rank_fraction",
        "cluster_size_log_capped",
        "anchor_evidence_count",
        "strong_positive_anchor_score",
        "weak_residual_anchor_score",
        "sparse_relative_winner_score",
    )
    assert _classic_monotone_constraints_for_features(feature_columns) == [
        -1,
        0,
        1,
        0,
        0,
        -1,
        -1,
        0,
        0,
        0,
        0,
        0,
        0,
        -1,
        0,
        1,
        1,
        1,
        1,
    ]


def test_resolve_classic_monotone_constraints_requires_explicit_opt_in() -> None:
    """Classic monotone constraints should only activate when the bundle specifies them."""

    feature_columns = ("title_overlap", "cluster_size", "query_view__full")
    assert _resolve_classic_monotone_constraints({}, feature_columns) is None
    assert _resolve_classic_monotone_constraints(
        {"monotone_constraints": [1, 0, 0]},
        feature_columns,
    ) == [1, 0, 0]


def test_validator_raw_feature_columns_ignore_runtime_derived_columns() -> None:
    """Feature coverage validation should not require runtime-derived columns on disk."""

    assert _raw_feature_columns_for_validation(
        (
            "title_overlap",
            "cluster_size_log_capped",
            "query_first_prefix_match",
            "query_view__full",
            "query_view__initial_only",
        )
    ) == ("title_overlap",)


def test_validator_reports_absent_columns_separately_from_missing_cells(tmp_path: Path) -> None:
    """Coverage validation should separate absent columns from actual NaNs."""

    path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {
                "dataset": "demo",
                "query_group_id": "demo:q1",
                "title_overlap": 0.5,
            }
        ]
    ).to_csv(path, index=False)

    coverage = _summarize_active_feature_coverage(
        path,
        feature_columns=("title_overlap", "runtime_zero_feature"),
        datasets=("demo",),
    )

    assert coverage["demo"]["rows_with_missing_features"] == 0
    assert coverage["demo"]["missing_feature_cells"] == 0
    assert coverage["demo"]["columns_with_missing"] == []
    assert coverage["demo"]["columns_absent"] == ["runtime_zero_feature"]
    assert _feature_coverage_failures({"eval": coverage}) == [
        "eval:demo:absent:['runtime_zero_feature']",
    ]


def test_validator_rejects_zero_row_dataset_filter(tmp_path: Path) -> None:
    """Dataset filters that match no rows should fail instead of summarizing the whole file."""

    path = tmp_path / "rows.csv"
    pd.DataFrame(
        [
            {
                "dataset": "inspire",
                "query_group_id": "inspire:q1",
                "title_overlap": 0.5,
            },
            {
                "dataset": "kisti",
                "query_group_id": "kisti:q2",
                "title_overlap": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="training_s2and_source_reviewed"):
        _summarize_active_feature_coverage(
            path,
            feature_columns=("title_overlap",),
            datasets=("training_s2and_source_reviewed",),
        )


def test_extra_eval_paths_support_dynamic_dataset_mapping() -> None:
    """Classic bundle specs should support arbitrary extra eval dataset mappings."""

    spec = {
        "s_park_eval_path": "test/s_park_eval_rows.csv.gz",
        "extra_eval_paths": {
            "j_smith": "test/j_smith_eval_rows.csv.gz",
            "a_silva": "test/a_silva_eval_rows.csv.gz",
        },
    }
    assert _iter_extra_eval_paths(spec) == (
        ("s_park", "test/s_park_eval_rows.csv.gz"),
        ("j_smith", "test/j_smith_eval_rows.csv.gz"),
        ("a_silva", "test/a_silva_eval_rows.csv.gz"),
    )
    assert _summary_key_for_eval_dataset("a_silva") == "overall_a_silva_eval"


def test_compare_to_expected_supports_dynamic_window_keys() -> None:
    """Expected-metric comparison should support w5/w25 keys, not just w50/w250."""

    summary = {
        "manual_holdout": {"overall": {"balanced_accuracy": 0.8}},
        "overall_s2and_eval": {
            "5": {"overall": {"balanced_accuracy": 0.71}},
            "25": {"overall": {"balanced_accuracy": 0.73}},
        },
        "hwang_cleaned_eval": {
            "w5": {"cleaned_balanced_accuracy": 0.81},
            "w25": {"cleaned_balanced_accuracy": 0.79},
        },
        "overall_s_park_eval": {
            "5": {"overall": {"balanced_accuracy": 0.77}},
            "25": {"overall": {"balanced_accuracy": 0.75}},
        },
        "overall_s_lee_eval": {
            "5": {"overall": {"balanced_accuracy": 0.79}},
            "25": {"overall": {"balanced_accuracy": 0.78}},
        },
        "overall_j_smith_eval": {
            "5": {"overall": {"balanced_accuracy": 0.66}},
            "25": {"overall": {"balanced_accuracy": 0.64}},
        },
        "abstain_rule": {
            "score_threshold": 0.9,
            "margin_threshold": 0.8,
            "single_candidate_score_threshold": 0.7,
        },
    }
    expected = {
        "manual_holdout_overall_balanced_accuracy": 0.75,
        "s2and_w5_balanced_accuracy": 0.7,
        "s2and_w25_balanced_accuracy": 0.7,
        "hwang_clean_w5_balanced_accuracy": 0.8,
        "hwang_clean_w25_balanced_accuracy": 0.8,
        "s_park_w5_balanced_accuracy": 0.75,
        "s_park_w25_balanced_accuracy": 0.74,
        "s_lee_w5_balanced_accuracy": 0.76,
        "s_lee_w25_balanced_accuracy": 0.77,
        "j_smith_w5_balanced_accuracy": 0.61,
        "j_smith_w25_balanced_accuracy": 0.6,
        "score_threshold": 0.85,
        "margin_threshold": 0.75,
        "single_candidate_score_threshold": 0.6,
    }
    deltas = compare_to_expected(summary, expected)
    assert deltas == {
        "manual_holdout_overall_balanced_accuracy": 0.050000000000000044,
        "s2and_w5_balanced_accuracy": 0.010000000000000009,
        "s2and_w25_balanced_accuracy": 0.030000000000000027,
        "hwang_clean_w5_balanced_accuracy": 0.010000000000000009,
        "hwang_clean_w25_balanced_accuracy": -0.010000000000000009,
        "s_park_w5_balanced_accuracy": 0.020000000000000018,
        "s_park_w25_balanced_accuracy": 0.010000000000000009,
        "s_lee_w5_balanced_accuracy": 0.030000000000000027,
        "s_lee_w25_balanced_accuracy": 0.010000000000000009,
        "j_smith_w5_balanced_accuracy": 0.050000000000000044,
        "j_smith_w25_balanced_accuracy": 0.040000000000000036,
        "score_threshold": 0.050000000000000044,
        "margin_threshold": 0.050000000000000044,
        "single_candidate_score_threshold": 0.09999999999999998,
    }


def test_expected_metrics_from_summary_captures_dynamic_eval_sections() -> None:
    """Frozen expected metrics should be derived from the replay summary shape."""

    summary = {
        "manual_holdout": {"overall": {"balanced_accuracy": 0.8}},
        "overall_s2and_eval": {
            "5": {"overall": {"balanced_accuracy": 0.71}},
            "25": {"overall": {"balanced_accuracy": 0.73}},
        },
        "hwang_cleaned_eval": {
            "w5": {"cleaned_balanced_accuracy": 0.81},
            "w25": {"cleaned_balanced_accuracy": 0.79},
        },
        "overall_s_park_eval": {
            "5": {"overall": {"balanced_accuracy": 0.77}},
            "25": {"overall": {"balanced_accuracy": 0.75}},
        },
        "overall_s_lee_eval": {
            "5": {"overall": {"balanced_accuracy": 0.79}},
            "25": {"overall": {"balanced_accuracy": 0.78}},
        },
        "overall_j_smith_eval": {
            "5": {"overall": {"balanced_accuracy": 0.66}},
            "25": {"overall": {"balanced_accuracy": 0.64}},
        },
        "abstain_rule": {
            "score_threshold": 0.9,
            "margin_threshold": 0.8,
            "single_candidate_score_threshold": 0.7,
        },
    }

    assert expected_metrics_from_summary(summary) == {
        "manual_holdout_overall_balanced_accuracy": 0.8,
        "s2and_w5_balanced_accuracy": 0.71,
        "s2and_w25_balanced_accuracy": 0.73,
        "s_park_w5_balanced_accuracy": 0.77,
        "s_park_w25_balanced_accuracy": 0.75,
        "s_lee_w5_balanced_accuracy": 0.79,
        "s_lee_w25_balanced_accuracy": 0.78,
        "j_smith_w5_balanced_accuracy": 0.66,
        "j_smith_w25_balanced_accuracy": 0.64,
        "hwang_clean_w5_balanced_accuracy": 0.81,
        "hwang_clean_w25_balanced_accuracy": 0.79,
        "score_threshold": 0.9,
        "margin_threshold": 0.8,
        "single_candidate_score_threshold": 0.7,
    }


def test_expected_metrics_from_summary_captures_bucketed_gate_and_stratified_split() -> None:
    """Frozen expected metrics should include the promoted bucketed gate and held-out split metrics."""

    summary = {
        "overall_s2and_eval": {
            "5": {"overall": {"balanced_accuracy": 0.71}},
        },
        "hwang_cleaned_eval": {},
        "abstain_rule": {
            "score_threshold": 0.82,
            "margin_threshold": 0.16,
            "single_candidate_score_threshold": 0.50,
            "bucketed_score_thresholds": {
                "multi_candidate|multi_letter_first": 0.82,
                "multi_candidate|single_letter_first": 0.04,
                "single_candidate|multi_letter_first": 0.01,
                "single_candidate|single_letter_first": 0.50,
            },
            "bucketed_margin_thresholds": {
                "multi_candidate|multi_letter_first": 0.16,
                "multi_candidate|single_letter_first": 0.41,
            },
        },
        "stratified_eval_test_split": {
            "overall": {
                "test": {
                    "balanced_accuracy": 0.92,
                    "accuracy": 0.95,
                    "error_rate": 0.05,
                }
            }
        },
    }

    expected = expected_metrics_from_summary(summary)

    assert expected["multi_candidate_single_letter_score_threshold"] == 0.04
    assert expected["single_candidate_multi_letter_score_threshold"] == 0.01
    assert expected["multi_candidate_multi_letter_margin_threshold"] == 0.16
    assert expected["multi_candidate_single_letter_margin_threshold"] == 0.41
    assert expected["stratified_test_balanced_accuracy"] == 0.92
    assert expected["stratified_test_accuracy"] == 0.95
    assert expected["stratified_test_error_rate"] == 0.05


def test_format_classic_selected_gate_tables_includes_requested_breakdowns() -> None:
    """Official table rendering should expose dataset and requested factor metrics."""

    metric = {
        "n_queries": 2,
        "n_positive_queries": 1,
        "n_negative_queries": 1,
        "balanced_accuracy": 0.75,
        "error_rate": 0.25,
        "false_abstain": 1,
        "false_link": 0,
        "wrong_candidate_link": 0,
    }
    all_negative_metric = {
        "n_queries": 5,
        "n_positive_queries": 0,
        "n_negative_queries": 5,
        "balanced_accuracy": 0.5,
        "error_rate": 0.0,
        "false_abstain": 0,
        "false_link": 0,
        "wrong_candidate_link": 0,
    }
    summary = {
        "stratified_eval_test_split": {
            "test_breakdowns": {
                "source_key": {"all_negative_eval": all_negative_metric, "demo_eval": metric},
                "has_positive_candidate": {"True": metric},
                "positive_rank_bucket": {"positive_first": metric},
                "first_name_bucket": {"multi_letter_first": metric},
                "multiple_candidates": {"True": metric},
            }
        }
    }

    tables = format_classic_selected_gate_tables(summary)

    assert "## By Dataset Slice, Selected Gate" in tables
    assert (
        "| slice | queries | positive queries | negative queries | BA | error rate | "
        "false abstain | false link | wrong link |"
    ) in tables
    assert "| demo_eval | 2 | 1 | 1 | 0.7500 | 0.2500 | 1 | 0 | 0 |" in tables
    assert "| all_negative_eval | 5 | 0 | 5 | n/a | 0.0000 | 0 | 0 | 0 |" in tables
    assert "BA is n/a for single-class slices." in tables
    assert "| has_positive_candidate | True | 2 | 1 | 1 | 0.2500 | 1 | 0 | 0 |" in tables


def test_sync_bundle_metadata_refreshes_counts_and_expected_metrics(tmp_path: Path) -> None:
    """Bundle-metadata sync should match on-disk files plus the replay summary."""

    bundle_root = tmp_path / "bundle"
    (bundle_root / "training").mkdir(parents=True)
    (bundle_root / "calibration").mkdir(parents=True)
    (bundle_root / "test").mkdir(parents=True)

    def _write_gzip_csv(relative_path: str, rows: list[dict[str, object]]) -> None:
        path = bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_csv(relative_path: str, rows: list[dict[str, object]]) -> None:
        path = bundle_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    _write_gzip_csv(
        "training/train_rows.csv.gz",
        [
            {"query_group_id": "train:q1", "label": 1},
            {"query_group_id": "train:q1", "label": 0},
            {"query_group_id": "train:q2", "label": 0},
        ],
    )
    _write_gzip_csv(
        "calibration/gate_rows.csv.gz",
        [
            {
                "dataset": "j_smith",
                "query_group_id": "j_smith:q1:full",
                "base_group_id": "j_smith:q1",
                "label": 1,
            },
            {
                "dataset": "j_smith",
                "query_group_id": "j_smith:q2:full",
                "base_group_id": "j_smith:q2",
                "label": 0,
            },
        ],
    )
    _write_csv("calibration/calibration_groups.csv", [{"base_group_id": "j_smith:q1"}])
    _write_csv("test/internal_groups.csv", [{"base_group_id": "j_smith:q2"}])
    _write_gzip_csv(
        "test/s2and_eval_rows.csv.gz",
        [{"query_group_id": "s2:q1", "label": 1}],
    )
    _write_gzip_csv(
        "test/hwang_eval_rows.csv.gz",
        [{"query_group_id": "h:q1", "label": 0}],
    )
    _write_gzip_csv(
        "test/s_park_eval_rows.csv.gz",
        [{"query_group_id": "p:q1", "label": 1}],
    )
    _write_gzip_csv(
        "test/s_lee_eval_rows.csv.gz",
        [{"query_group_id": "l:q1", "label": 0}],
    )
    _write_gzip_csv(
        "test/j_smith_eval_rows.csv.gz",
        [
            {"query_group_id": "j_smith:q1:full", "label": 1},
            {"query_group_id": "j_smith:q2:full", "label": 0},
        ],
    )
    _write_csv(
        "test/hwang_cleaned_eval_overrides.csv",
        [
            {
                "query_group_id": "h:q1",
                "manual_safe_target": 1,
                "manual_assessment": "possible",
                "correction_type": "none",
                "review_source_path": "scratch/demo.tsv",
            }
        ],
    )
    _write_csv(
        "test/hwang_candidate_level_label_overrides.csv",
        [
            {
                "query_group_id": "h:q1",
                "dataset": "h_wang",
                "correction_type": "top1_should_link",
                "reviewed_candidate_component_key": "c1",
                "reviewed_candidate_survived": 1,
                "raw_positive_rows_before_candidate_relabel": 0,
                "label_action": "add_reviewed_positive",
                "review_source_path": "scratch/demo.tsv",
                "positive_rows_after_candidate_relabel": 1,
                "manual_safe_target": 1,
            }
        ],
    )
    (bundle_root / "test/hwang_candidate_level_label_overrides_summary.json").write_text(
        json.dumps(
            {
                "queries": 1,
                "positive_queries_after_candidate_relabel": 1,
                "reviewed_positive_corrections": 1,
                "reviewed_positive_corrections_survived": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_root / "dataset_contract").mkdir()
    (bundle_root / "dataset_contract/custom_label_ledger_summary.json").write_text(
        json.dumps(
            {
                "ledger_rows": 12,
                "comparison_fatal_mismatch_count": 0,
                "slice_counts": {"demo": 12},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    bundle_payload = {
        "bundle_name": "demo_bundle",
        "created_on": "2026-01-01",
        "assets": {
            "training": {
                "classic_train_union21_plus_s_lee_raw": {
                    "path": "training/train_rows.csv.gz",
                    "rows": 0,
                    "queries": 0,
                    "positive_rows": 0,
                }
            },
            "dataset_contract": {
                "filter_policy_path": "dataset_contract/filter_policy.json",
                "custom_label_ledger_path": "dataset_contract/custom_label_ledger.csv",
                "custom_label_ledger_summary_path": "dataset_contract/custom_label_ledger_summary.json",
                "custom_label_ledger_comparison_path": "dataset_contract/custom_label_ledger_comparison.json",
                "custom_label_ledger_report_path": "dataset_contract/custom_label_ledger_report.md",
                "custom_label_ledger_rows": 0,
                "comparison_fatal_mismatch_count": 1,
                "label_slice_counts": {},
            },
            "calibration": {
                "classic_gate_source": {
                    "path": "calibration/gate_rows.csv.gz",
                    "rows": 0,
                    "queries": 0,
                    "positive_rows": 0,
                },
                "classic_gate_split": {
                    "calibration_path": "calibration/calibration_groups.csv",
                    "evaluation_path": "test/internal_groups.csv",
                    "calibration_groups": 0,
                    "evaluation_groups": 0,
                },
                "total_error_4score_2margin_gate": {
                    "selected_gate_path": "calibration/total_error_4score_2margin_gate/selected_gate.json",
                    "candidate_metrics_path": (
                        "calibration/total_error_4score_2margin_gate/gate_candidate_metrics.csv"
                    ),
                    "summary_path": "calibration/total_error_4score_2margin_gate/summary.json",
                    "report_path": "calibration/total_error_4score_2margin_gate/report.md",
                },
            },
            "test": {
                "s2and_eval": {
                    "path": "test/s2and_eval_rows.csv.gz",
                    "rows": 0,
                    "queries": 0,
                    "positive_rows": 0,
                    "positive_queries": 99,
                    "negative_queries": 99,
                },
                "hwang_eval": {"path": "test/hwang_eval_rows.csv.gz", "rows": 0, "queries": 0, "positive_rows": 0},
                "s_park_eval": {"path": "test/s_park_eval_rows.csv.gz", "rows": 0, "queries": 0, "positive_rows": 0},
                "s_lee_eval": {"path": "test/s_lee_eval_rows.csv.gz", "rows": 0, "queries": 0, "positive_rows": 0},
                "j_smith_eval": {"path": "test/j_smith_eval_rows.csv.gz", "rows": 0, "queries": 0, "positive_rows": 0},
                "hwang_clean_overrides": {
                    "path": "test/hwang_cleaned_eval_overrides.csv",
                    "queries": 0,
                    "positive_overrides": 0,
                },
                "hwang_candidate_level_label_overrides": {
                    "path": "test/hwang_candidate_level_label_overrides.csv",
                    "summary_path": "test/hwang_candidate_level_label_overrides_summary.json",
                    "queries": 0,
                    "positive_queries_after_candidate_relabel": 0,
                    "reviewed_positive_corrections": 0,
                    "reviewed_positive_corrections_survived": 0,
                },
            },
        },
        "models": {
            "classic": {
                "train_path": "training/train_rows.csv.gz",
                "classic_gate_source_path": "calibration/gate_rows.csv.gz",
                "classic_gate_calibration_base_groups_path": "calibration/calibration_groups.csv",
                "classic_gate_internal_eval_base_groups_path": "test/internal_groups.csv",
                "s2and_eval_path": "test/s2and_eval_rows.csv.gz",
                "hwang_eval_path": "test/hwang_eval_rows.csv.gz",
                "s_park_eval_path": "test/s_park_eval_rows.csv.gz",
                "s_lee_eval_path": "test/s_lee_eval_rows.csv.gz",
                "extra_eval_paths": {
                    "j_smith": "test/j_smith_eval_rows.csv.gz",
                },
            }
        },
        "expected_metrics": {
            "classic": {
                "score_threshold": 0.0,
                "margin_threshold": 0.0,
            }
        },
        "new_block_manual_calibration_split": {
            "per_dataset": {
                "j_smith": {
                    "total_base_groups": 0,
                    "calibration_base_groups": 0,
                    "evaluation_base_groups": 0,
                    "total_positive_queries": 0,
                    "total_negative_queries": 0,
                    "calibration_positive_queries": 0,
                    "calibration_negative_queries": 0,
                    "evaluation_positive_queries": 0,
                    "evaluation_negative_queries": 0,
                }
            }
        },
    }
    (bundle_root / "bundle.json").write_text(json.dumps(bundle_payload, indent=2) + "\n", encoding="utf-8")

    summary = {
        "overall_s2and_eval": {
            "5": {"overall": {"balanced_accuracy": 0.71}},
            "25": {"overall": {"balanced_accuracy": 0.73}},
        },
        "hwang_cleaned_eval": {
            "w5": {"cleaned_balanced_accuracy": 0.81},
            "w25": {"cleaned_balanced_accuracy": 0.79},
        },
        "overall_s_park_eval": {
            "5": {"overall": {"balanced_accuracy": 0.77}},
            "25": {"overall": {"balanced_accuracy": 0.75}},
        },
        "overall_s_lee_eval": {
            "5": {"overall": {"balanced_accuracy": 0.79}},
            "25": {"overall": {"balanced_accuracy": 0.78}},
        },
        "overall_j_smith_eval": {
            "5": {"overall": {"balanced_accuracy": 0.66}},
            "25": {"overall": {"balanced_accuracy": 0.64}},
        },
        "abstain_rule": {
            "score_threshold": 0.9,
            "margin_threshold": 0.8,
            "single_candidate_score_threshold": 0.7,
            "promoted_stratified_gate": {
                "selected_gate": {
                    "name": "demo_gate",
                    "score_thresholds": {
                        "multi_candidate|multi_letter_first": 0.9,
                        "multi_candidate|single_letter_first": 0.2,
                        "single_candidate|multi_letter_first": 0.3,
                        "single_candidate|single_letter_first": 0.7,
                    },
                    "margin_thresholds": {
                        "multi_candidate|multi_letter_first": 0.8,
                        "multi_candidate|single_letter_first": 0.6,
                    },
                    "lambda_penalty": 0.0,
                },
                "selection_key": {
                    "check_errors": 2,
                    "check_false_link": 1,
                    "check_wrong_candidate_link": 0,
                    "check_false_abstain": 1,
                    "total_threshold_drift": 0.4,
                },
                "fit_metrics": {
                    "n_queries": 10,
                    "errors": 1,
                    "balanced_accuracy": 0.9,
                },
                "check_metrics": {
                    "n_queries": 8,
                    "errors": 2,
                    "balanced_accuracy": 0.8,
                },
                "candidate_metrics": [
                    {
                        "name": "demo_gate",
                        "lambda_penalty": 0.0,
                        "check_errors": 2,
                    }
                ],
            },
        },
    }

    verification_path = tmp_path / "replay" / "verification.json"
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    verification_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "expected": {
                    "s2and_w5_balanced_accuracy": 0.5,
                    "s2and_w25_balanced_accuracy": 0.5,
                    "score_threshold": 0.1,
                    "margin_threshold": 0.1,
                },
                "deltas": {
                    "s2and_w5_balanced_accuracy": 0.21,
                    "s2and_w25_balanced_accuracy": 0.23,
                    "score_threshold": 0.8,
                    "margin_threshold": 0.7,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sync_summary = sync_bundle_metadata(
        bundle_root,
        summary,
        created_on="2026-04-20",
        verification_json_path=verification_path,
    )
    updated_bundle = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    updated_verification = json.loads(verification_path.read_text(encoding="utf-8"))

    assert updated_bundle["created_on"] == "2026-04-20"
    assert updated_bundle["assets"]["training"]["classic_train_union21_plus_s_lee_raw"] == {
        "path": "training/train_rows.csv.gz",
        "rows": 3,
        "queries": 2,
        "positive_rows": 1,
    }
    assert updated_bundle["assets"]["calibration"]["classic_gate_source"] == {
        "path": "calibration/gate_rows.csv.gz",
        "rows": 2,
        "queries": 2,
        "positive_rows": 1,
    }
    assert updated_bundle["assets"]["calibration"]["classic_gate_split"]["calibration_groups"] == 1
    assert updated_bundle["assets"]["calibration"]["classic_gate_split"]["evaluation_groups"] == 1
    assert updated_bundle["assets"]["dataset_contract"]["custom_label_ledger_rows"] == 12
    assert updated_bundle["assets"]["dataset_contract"]["comparison_fatal_mismatch_count"] == 0
    assert updated_bundle["assets"]["dataset_contract"]["label_slice_counts"] == {"demo": 12}
    assert updated_bundle["assets"]["test"]["j_smith_eval"]["queries"] == 2
    assert updated_bundle["assets"]["test"]["hwang_clean_overrides"] == {
        "path": "test/hwang_cleaned_eval_overrides.csv",
        "queries": 1,
        "positive_overrides": 1,
    }
    assert updated_bundle["assets"]["test"]["hwang_candidate_level_label_overrides"] == {
        "path": "test/hwang_candidate_level_label_overrides.csv",
        "summary_path": "test/hwang_candidate_level_label_overrides_summary.json",
        "queries": 1,
        "positive_queries_after_candidate_relabel": 1,
        "reviewed_positive_corrections": 1,
        "reviewed_positive_corrections_survived": 1,
    }
    assert updated_bundle["assets"]["test"]["s2and_eval"] == {
        "path": "test/s2and_eval_rows.csv.gz",
        "rows": 1,
        "queries": 1,
        "positive_rows": 1,
        "positive_queries": 1,
        "negative_queries": 0,
    }
    assert updated_bundle["new_block_manual_calibration_split"]["per_dataset"]["j_smith"] == {
        "total_base_groups": 2,
        "calibration_base_groups": 1,
        "evaluation_base_groups": 1,
        "total_positive_queries": 1,
        "total_negative_queries": 1,
        "calibration_positive_queries": 1,
        "calibration_negative_queries": 0,
        "evaluation_positive_queries": 0,
        "evaluation_negative_queries": 1,
    }
    assert updated_bundle["expected_metrics"]["classic"] == {
        "s2and_w5_balanced_accuracy": 0.71,
        "s2and_w25_balanced_accuracy": 0.73,
        "s_park_w5_balanced_accuracy": 0.77,
        "s_park_w25_balanced_accuracy": 0.75,
        "s_lee_w5_balanced_accuracy": 0.79,
        "s_lee_w25_balanced_accuracy": 0.78,
        "j_smith_w5_balanced_accuracy": 0.66,
        "j_smith_w25_balanced_accuracy": 0.64,
        "hwang_clean_w5_balanced_accuracy": 0.81,
        "hwang_clean_w25_balanced_accuracy": 0.79,
        "score_threshold": 0.9,
        "margin_threshold": 0.8,
        "single_candidate_score_threshold": 0.7,
    }
    selected_gate = json.loads(
        (bundle_root / "calibration/total_error_4score_2margin_gate/selected_gate.json").read_text(encoding="utf-8")
    )
    gate_summary = json.loads(
        (bundle_root / "calibration/total_error_4score_2margin_gate/summary.json").read_text(encoding="utf-8")
    )
    gate_metrics = pd.read_csv(bundle_root / "calibration/total_error_4score_2margin_gate/gate_candidate_metrics.csv")
    gate_report = (bundle_root / "calibration/total_error_4score_2margin_gate/report.md").read_text(encoding="utf-8")
    assert updated_bundle["assets"]["calibration"]["total_error_4score_2margin_gate"]["selected_gate_name"] == (
        "demo_gate"
    )
    assert selected_gate["score_thresholds"]["multi_candidate|multi_letter_first"] == 0.9
    assert gate_summary["candidate_count"] == 1
    assert gate_metrics["name"].tolist() == ["demo_gate"]
    assert "Selected gate: `demo_gate`" in gate_report
    assert sync_summary["classic_gate_split"] == {
        "calibration_groups": 1,
        "evaluation_groups": 1,
    }
    assert sync_summary["dataset_contract"] == {
        "custom_label_ledger_rows": 12,
        "comparison_fatal_mismatch_count": 0,
        "label_slice_counts": {"demo": 12},
    }
    assert sync_summary["verification_json_path"] == str(verification_path.resolve())
    assert updated_verification["summary"] == summary
    assert updated_verification["expected"] == updated_bundle["expected_metrics"]["classic"]
    assert updated_verification["deltas"] == compare_to_expected(summary, updated_bundle["expected_metrics"]["classic"])


def test_score_eval_candidate_rows_defaults_to_w5_and_w25_only() -> None:
    """Official classic eval scoring should only materialize the retained window limits."""

    df = pd.DataFrame(
        [
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c1",
                "retrieval_rank": 1,
                "label": 1,
            },
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c2",
                "retrieval_rank": 10,
                "label": 0,
            },
            {
                "query_group_id": "q1",
                "dataset": "demo",
                "query_view": "full",
                "candidate_component_key": "c3",
                "retrieval_rank": 30,
                "label": 0,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c4",
                "retrieval_rank": 3,
                "label": 0,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c5",
                "retrieval_rank": 20,
                "label": 1,
            },
            {
                "query_group_id": "q2",
                "dataset": "demo",
                "query_view": "initial_only",
                "candidate_component_key": "c6",
                "retrieval_rank": 40,
                "label": 0,
            },
        ]
    )

    scored = _score_eval_candidate_rows(
        df,
        probabilities=np.array([0.9, 0.4, 0.1, 0.3, 0.8, 0.2], dtype=np.float32),
        include_margin=True,
    )

    assert sorted(scored["retrieval_rank_limit"].astype(int).unique().tolist()) == [5, 25]


def test_resume_helpers_reuse_saved_spool_outputs(tmp_path, monkeypatch) -> None:
    """Resume helpers should recover staged counts from spool and reuse saved worker summaries."""

    source_root = tmp_path / "source"
    telemetry_dir = tmp_path / "telemetry"
    source_root.mkdir()
    telemetry_dir.mkdir()

    relative_path = Path("test") / "demo_rows.csv.gz"
    source_path = r"test\demo_rows.csv.gz"
    row_file = source_root / relative_path
    row_file.parent.mkdir(parents=True)
    original_rows = [
        {
            "dataset": "demo",
            "query_group_id": "q1",
            "candidate_component_key": "c1",
            "retrieval_rank": 1,
            "label": 1,
        },
        {
            "dataset": "demo",
            "query_group_id": "q1",
            "candidate_component_key": "c2",
            "retrieval_rank": 2,
            "label": 0,
        },
    ]
    with gzip.open(row_file, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(original_rows[0]))
        writer.writeheader()
        writer.writerows(original_rows)

    monkeypatch.setattr(rebuild_stack, "SOURCE_BUNDLE_ROOT", source_root)
    monkeypatch.setattr(rebuild_stack, "TELEMETRY_DIR", telemetry_dir)

    spool_db_path = tmp_path / "group_rebuild_spool.sqlite3"
    connection = rebuild_stack._connect_spool_db(spool_db_path)
    try:
        connection.execute(
            """
            INSERT INTO staged_groups (
                source_path,
                group_index,
                dataset_name,
                query_group_id,
                rows_before_total,
                positive_rows_before_total,
                rows_after_window_cap,
                positive_rows_after_window_cap,
                rows_blob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_path,
                1,
                "demo",
                "q1",
                2,
                1,
                2,
                1,
                sqlite3.Binary(reranker_staging.compress_rows(original_rows)),
            ),
        )
        connection.execute(
            "INSERT INTO rebuilt_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
            (
                source_path,
                1,
                sqlite3.Binary(reranker_staging.compress_rows([original_rows[0]])),
            ),
        )
        connection.commit()

        staging_config = reranker_staging.StageInputGroupsConfig(
            source_bundle_root=source_root,
            s2and_row_relative_path=rebuild_stack.S2AND_ROW_RELATIVE_PATH,
            s2and_full_relabel_pre_filter_rows_path=source_root / "missing_pre_filter.csv.gz",
            window_size=rebuild_stack.WINDOW_SIZE,
            read_initial_only_rereview_decisions=lambda: {},
            read_s2and_full_relabel_decisions=lambda: {},
            merge_initial_only_rereview_into_s2and_decisions=lambda decisions, _initial_only: decisions,
            apply_initial_only_rereview_to_group=lambda rows, *, decision: list(rows),
            apply_s2and_full_relabel_to_group=lambda rows, *, decisions: list(rows),
        )
        fieldnames_by_path, file_summaries, ordered_source_paths = (
            reranker_staging.load_staged_input_groups_from_spool(
                connection,
                selected_row_paths=(relative_path,),
                config=staging_config,
            )
        )
        assert fieldnames_by_path[source_path][: len(original_rows[0])] == list(original_rows[0])
        assert "top3_distance_best_gap" in fieldnames_by_path[source_path]
        assert ordered_source_paths == [source_path]
        assert file_summaries[source_path].groups_before == 1
        assert file_summaries[source_path].rows_before == 2
        assert file_summaries[source_path].positive_rows_before == 1
        assert dict(file_summaries[source_path].per_dataset_groups) == {"demo": 1}
        assert rebuild_stack._dataset_rebuild_is_complete(
            connection,
            dataset_name="demo",
            ordered_source_paths=ordered_source_paths,
        )
    finally:
        connection.close()

    worker_summary = reranker_staging.FileRepairSummaryState(path=source_path)
    worker_summary.record_result(
        rows_before=2,
        positive_rows_before=1,
        rebuilt_rows=[{"label": 1}],
        group_summary={"query_group_id": "q1", "dataset": "demo"},
    )
    summary_path = rebuild_stack._worker_summary_path("demo")
    summary_path.write_text(
        json.dumps({"dataset": "demo", "files": [worker_summary.to_result_payload()]}) + "\n",
        encoding="utf-8",
    )

    rebuild_stack._merge_worker_summary(
        dataset_name="demo",
        file_summary_states=file_summaries,
        summary_path=summary_path,
    )
    payload = file_summaries[source_path].to_payload()
    assert payload["rows_after"] == 1
    assert payload["positive_rows_after"] == 1
    assert payload["groups_with_dropped_rows"] == 1
    assert payload["sample_dropped_groups"][0]["query_group_id"] == "q1"


def test_replace_with_retry_retries_permission_error(tmp_path, monkeypatch) -> None:
    """Final row-file replacement should retry transient Windows access-denied failures."""

    output_path = tmp_path / "demo_rows.csv.gz"
    temp_path = tmp_path / "demo_rows.csv.gz.tmp"
    output_path.write_text("old", encoding="utf-8")
    temp_path.write_text("new", encoding="utf-8")

    call_count = {"replace": 0}
    real_replace = Path.replace

    def flaky_replace(self: Path, target: str | Path) -> Path:
        if self == temp_path and Path(target) == output_path and call_count["replace"] == 0:
            call_count["replace"] += 1
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(rebuild_stack.time, "sleep", lambda _: None)

    rebuild_stack._replace_with_retry(
        temp_path,
        output_path,
        max_attempts=2,
        initial_delay_seconds=0.01,
        max_delay_seconds=0.01,
    )

    assert call_count["replace"] == 1
    assert output_path.read_text(encoding="utf-8") == "new"
    assert not temp_path.exists()
