from __future__ import annotations

import json
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import s2and.incremental_linking.query_adapter as retrieval
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.incremental_linking_training import build_rust_hybrid_centroid_retriever
from scripts.joint_safe_link_official_stack import OfficialBundle
from scripts.run_joint_safe_link_promoted_train_calibrate_eval import (
    _clean_minimal_raw_structural_rows,
    _component_member_details_by_key,
    _enable_fasttext_language_detection,
    _has_query_seed_connection,
    _load_target,
    _query_first_token_for_prefix,
    _resolve_candidate_batch_pair_labels,
    _row_allows_seed_constraint_bypass,
    _row_label_is_positive,
    _score_candidate_summaries_with_frozen_rust_policy,
)


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


def test_load_target_accepts_historical_supported_promoted_features(tmp_path) -> None:
    target_path = tmp_path / "historical_target.json"
    target_path.write_text(
        json.dumps(
            {
                "feature_count": 2,
                "features": ["min_distance", "pw_max_email_prefix_equal"],
            }
        ),
        encoding="utf-8",
    )

    target = _load_target(target_path)

    assert target["features"] == ["min_distance", "pw_max_email_prefix_equal"]


def test_load_target_rejects_unsupported_historical_features(tmp_path) -> None:
    target_path = tmp_path / "unsupported_target.json"
    target_path.write_text(
        json.dumps(
            {
                "feature_count": 1,
                "features": ["min_distance_rank_fraction"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown features"):
        _load_target(target_path)


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
        dataset=SimpleNamespace(),
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
        dataset=dataset,
    )

    assert details["m muller::284283"].signature_ids == ("b", "c")
    assert details["m muller::284283"].signature_indices.tolist() == [1, 2]
    assert details["other::1"].signature_ids == ("d",)

    frozen_details = _component_member_details_by_key(
        members_path,
        {"a": 0, "b": 1, "c": 2, "d": 3},
        dataset=dataset,
        component_scope="frozen",
    )

    assert frozen_details["m muller::284283"].signature_ids == ("a", "b", "c")
    assert frozen_details["m muller::284283"].signature_indices.tolist() == [0, 1, 2]
    assert frozen_details["other::1"].signature_ids == ("d",)


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


def test_minimal_raw_loader_enables_fasttext_language_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")

    _enable_fasttext_language_detection()

    assert os.environ["S2AND_SKIP_FASTTEXT"] == "0"


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
        dataset,
        row,
        seed_constraint_signature_ids=frozenset({"q", "m1", "other"}),
    )
    assert _has_query_seed_connection(dataset, query_signature_id="q", candidate_signature_ids=["m1"])
    assert not _has_query_seed_connection(dataset, query_signature_id="q", candidate_signature_ids=["other"])


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

    assert _row_allows_seed_constraint_bypass(dataset, row, seed_constraint_signature_ids=frozenset({"q", "m1"}))
    assert _has_query_seed_connection(dataset, query_signature_id="q", candidate_signature_ids=["m1"])


def test_minimal_raw_query_first_prefix_uses_full_author_before_masked_view() -> None:
    group = pd.DataFrame(
        [
            {
                "query_author": "Jianping Wang",
                "query_first_token": "j",
            }
        ]
    )

    assert _query_first_token_for_prefix(group, SimpleNamespace(first="j")) == "jianping"


def test_minimal_raw_retrieval_score_uses_frozen_rust_policy() -> None:
    pytest.importorskip("s2and_rust")
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
        max_block_component_size=1,
        n_jobs=1,
    )

    assert scores["c1"] == pytest.approx(0.620239, abs=1e-6)
