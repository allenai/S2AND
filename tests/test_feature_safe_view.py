from __future__ import annotations

import math
from typing import Any

import pytest

import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import (
    _get_rust_featurizer,
    clear_rust_featurizer_cache,
    get_constraint_rust,
    get_constraints_matrix_indexed_rust,
)
from s2and.runtime import build_runtime_context
from scripts import single_letter_reranker_utils as reranker_utils
from tests.helpers import build_cluster_summary, build_query_features, import_s2and_rust

_ORCID = "0000-0000-0000-0001"
_NORMALIZED_ORCID = "0000000000000001"
_HAS_RUST, _RUST_IMPORT_PAYLOAD = import_s2and_rust(required_method="from_dataset", prefer_site_packages=True)


def _signature(
    signature_id: str,
    *,
    paper_id: int,
    first: str,
    middle: str = "",
    last: str = "Smith",
    orcid: str | None = _ORCID,
) -> dict[str, Any]:
    author_info: dict[str, Any] = {
        "position": 0,
        "block": f"{first[:1].lower()} {last.lower()}",
        "first": first,
        "middle": middle,
        "last": last,
        "suffix": None,
        "email": None,
        "affiliations": [],
        "given_block": f"{first[:1].lower()} {last.lower()}",
    }
    if orcid is not None:
        author_info["source_id_source"] = "ORCID"
        author_info["source_ids"] = [orcid]
    return {
        "signature_id": signature_id,
        "paper_id": paper_id,
        "author_info": author_info,
    }


def _paper(paper_id: int, title: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": "",
        "journal_name": "",
        "venue": "",
        "year": 2020,
        "authors": [{"position": 0, "author_name": "Alice Smith"}],
        "references": [],
    }


def _feature_safe_dataset() -> ANDData:
    signatures = {
        "same_a": _signature("same_a", paper_id=1, first="Alice"),
        "same_b": _signature("same_b", paper_id=2, first="Alice"),
        "last_a": _signature("last_a", paper_id=3, first="Alice", last="Smith"),
        "last_b": _signature("last_b", paper_id=4, first="Alice", last="Jones"),
        "first_a": _signature("first_a", paper_id=5, first="Alice", last="Smith"),
        "first_b": _signature("first_b", paper_id=6, first="Bob", last="Smith"),
        "middle_a": _signature("middle_a", paper_id=7, first="Alice", middle="Marie", last="Smith"),
        "middle_b": _signature("middle_b", paper_id=8, first="Alice", middle="Zoe", last="Smith"),
    }
    papers = {str(paper_id): _paper(paper_id, f"Paper {paper_id}") for paper_id in range(1, 9)}
    clusters = {
        f"cluster_{signature_id}": {
            "cluster_id": f"cluster_{signature_id}",
            "signature_ids": [signature_id],
            "model_version": -1,
        }
        for signature_id in signatures
    }
    return ANDData(
        signatures,
        papers,
        name="feature_safe_view",
        clusters=clusters,
        load_name_counts=False,
        preprocess=True,
        name_tuples=set(),
        n_jobs=1,
    )


def _require_rust() -> None:
    if not _HAS_RUST:
        raise pytest.skip.Exception(f"s2and_rust extension not built/installed: {_RUST_IMPORT_PAYLOAD}")


def test_get_constraint_suppress_orcid_python() -> None:
    dataset = _feature_safe_dataset()

    assert dataset.get_constraint("same_a", "same_b") == 0
    assert dataset.get_constraint("same_a", "same_b", suppress_orcid=True) is None


def test_get_constraint_suppress_orcid_rust_parity() -> None:
    _require_rust()
    dataset = _feature_safe_dataset()
    clear_rust_featurizer_cache()

    assert get_constraint_rust(dataset, "same_a", "same_b") == dataset.get_constraint("same_a", "same_b")
    assert get_constraint_rust(
        dataset,
        "same_a",
        "same_b",
        suppress_orcid=True,
    ) == dataset.get_constraint("same_a", "same_b", suppress_orcid=True)


def test_cached_rust_featurizer_respects_suppress_orcid_per_call() -> None:
    _require_rust()
    dataset = _feature_safe_dataset()
    clear_rust_featurizer_cache()
    rust_featurizer = _get_rust_featurizer(dataset, use_cache=False)
    signature_index = {str(sig_id): idx for idx, sig_id in enumerate(rust_featurizer.signature_ids())}
    indexed_pairs = [(signature_index["same_a"], signature_index["same_b"])]

    default_values = get_constraints_matrix_indexed_rust(
        dataset,
        indexed_pairs,
        featurizer=rust_featurizer,
        suppress_orcid=False,
    )
    suppressed_values = get_constraints_matrix_indexed_rust(
        dataset,
        indexed_pairs,
        featurizer=rust_featurizer,
        suppress_orcid=True,
    )
    default_values_again = get_constraints_matrix_indexed_rust(
        dataset,
        indexed_pairs,
        featurizer=rust_featurizer,
        suppress_orcid=False,
    )

    assert default_values == [0.0]
    assert suppressed_values == [None]
    assert default_values_again == [0.0]


def test_orcid_positive_label_count_unchanged() -> None:
    dataset = _feature_safe_dataset()
    runtime_context = build_runtime_context("feature_safe_view_test")
    backend_default = model_module._build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=True,
        runtime_context=runtime_context,
        use_cache=False,
        suppress_orcid=False,
    )
    backend_suppressed = model_module._build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=True,
        runtime_context=runtime_context,
        use_cache=False,
        suppress_orcid=True,
    )
    default_labels, _default_telemetry = model_module._resolve_constraint_labels_batch(
        dataset,
        [("same_a", "same_b")],
        constraint_backend=backend_default,
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        incremental_dont_use_cluster_seeds=False,
        runtime_context=runtime_context,
        use_cache=False,
    )
    suppressed_labels, _suppressed_telemetry = model_module._resolve_constraint_labels_batch(
        dataset,
        [("same_a", "same_b")],
        constraint_backend=backend_suppressed,
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        incremental_dont_use_cluster_seeds=False,
        runtime_context=runtime_context,
        use_cache=False,
    )

    assert default_labels == [float(-LARGE_INTEGER)]
    assert math.isnan(suppressed_labels[0])

    query_case = reranker_utils.RerankerQueryCase(
        source="labeled",
        dataset="h_wang_fixture",
        query_id="q1",
        query_signature_id="same_a",
        block_key="a smith",
        positive_component_keys=frozenset({"same_orcid_component"}),
        support_type="orcid",
        block_size=3,
        component_size=1,
        sampling_info_bucket="tiny",
        normalized_orcid=_NORMALIZED_ORCID,
        orcid_group_size=2,
        orcid_group_size_bucket="2",
    )
    retrieval_window_state = {
        "scored_candidate_components": 2,
        "scored_candidate_signatures": 2,
        "orcid_filter_applied": 0,
        "middle_initial_filter_applied": 0,
        "year_range_filter_applied": 0,
    }

    def _stats(component_key: str, distance: float, *, rank: int, score: float) -> reranker_utils.ClusterPairwiseStats:
        return reranker_utils.ClusterPairwiseStats(
            cluster_id=component_key,
            retrieval_rank=rank,
            retrieval_score=score,
            cluster_size=1,
            count=1,
            sum_distance=distance,
            min_distance=distance,
            top_smallest_neg_heap=[-distance],
        )

    def _rows(positive_distance: float) -> list[dict[str, Any]]:
        return reranker_utils.make_candidate_rows(
            query_case=query_case,
            query_view="full",
            query_features=build_query_features(first="alice", orcid=_NORMALIZED_ORCID),
            shortlist_component_keys=["same_orcid_component", "negative_component"],
            retrieval_scores={"same_orcid_component": 0.9, "negative_component": 0.7},
            retrieval_ranks={"same_orcid_component": 1, "negative_component": 2},
            retrieval_window_state=retrieval_window_state,
            summary_by_component={
                "same_orcid_component": build_cluster_summary(component_key="same_orcid_component", size=1),
                "negative_component": build_cluster_summary(component_key="negative_component", size=1),
            },
            stats_by_component={
                "same_orcid_component": _stats(
                    "same_orcid_component",
                    positive_distance,
                    rank=1,
                    score=0.9,
                ),
                "negative_component": _stats("negative_component", 0.8, rank=2, score=0.7),
            },
        )

    rows_before = _rows(0.0)
    rows_after = _rows(0.6)

    assert sum(int(row["label"]) for row in rows_before) == 1
    assert sum(int(row["label"]) for row in rows_after) == 1
    assert [row["positive_candidate_keys"] for row in rows_before] == [
        row["positive_candidate_keys"] for row in rows_after
    ]


def test_other_constraints_intact_under_suppress_orcid() -> None:
    dataset = _feature_safe_dataset()

    assert dataset.get_constraint("last_a", "last_b", suppress_orcid=True) == LARGE_DISTANCE
    assert dataset.get_constraint("first_a", "first_b", suppress_orcid=True) == LARGE_DISTANCE
    assert dataset.get_constraint("middle_a", "middle_b", suppress_orcid=True) == LARGE_DISTANCE
