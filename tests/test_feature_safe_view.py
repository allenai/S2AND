from __future__ import annotations

import math
from typing import Any

import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import (
    _get_rust_featurizer,
    get_constraints_matrix_indexed_rust,
)
from s2and.runtime import build_runtime_context
from tests.helpers import build_arrow_training_dataset

_ORCID = "0000-0000-0000-0001"


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
        name_counts_index=None,
        preprocess=True,
        name_tuples=set(),
        n_jobs=1,
    )


def test_python_and_cached_rust_constraints_respect_suppress_orcid_per_call(tmp_path) -> None:
    source_dataset = _feature_safe_dataset()
    dataset = build_arrow_training_dataset(source_dataset, tmp_path, name_counts="empty")
    rust_featurizer = _get_rust_featurizer(dataset)
    signature_index = {str(sig_id): idx for idx, sig_id in enumerate(rust_featurizer.signature_ids())}
    pairs = [(f"{kind}_a", f"{kind}_b") for kind in ("same", "last", "first", "middle")]
    indexed_pairs = [(signature_index[left], signature_index[right]) for left, right in pairs]

    # Toggle back to prove suppression is scoped to a call on the cached handle.
    for suppress in (False, True, False):
        expected = [None, LARGE_DISTANCE, LARGE_DISTANCE, LARGE_DISTANCE] if suppress else [0, 0, 0, 0]
        assert [source_dataset.get_constraint(*pair, suppress_orcid=suppress) for pair in pairs] == expected
        assert (
            get_constraints_matrix_indexed_rust(indexed_pairs, featurizer=rust_featurizer, suppress_orcid=suppress)
            == expected
        )


def test_suppress_orcid_removes_orcid_supervision() -> None:
    dataset = _feature_safe_dataset()
    runtime_context = build_runtime_context("feature_safe_view_test", backend="python")
    for suppress in (False, True):
        backend = model_module._build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=True,
            runtime_context=runtime_context,
            suppress_orcid=suppress,
        )
        labels, _telemetry = model_module._resolve_constraint_labels_batch(
            dataset,
            [("same_a", "same_b")],
            constraint_backend=backend,
            partial_supervision={},
            use_default_constraints_as_supervision=True,
            constraint_policy=model_module._ConstraintPolicy(),
            runtime_context=runtime_context,
        )
        assert math.isnan(labels[0]) if suppress else labels == [float(-LARGE_INTEGER)]
