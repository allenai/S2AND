"""FeatureBlock conversion helpers for writing Arrow tables from ANDData."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from s2and.incremental_linking.feature_block_arrow import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    write_feature_block_arrow_tables,
    write_name_counts_index,
)
from s2and.incremental_linking.feature_block_contract import (
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    _optional_bool,
    _optional_int,
    _optional_str,
    _strict_string_tuple,
    filter_cluster_seed_disallows_for_signature_subset,
)
from s2and.name_counts_manifest import NAME_COUNTS_PROVENANCE_SCHEMA_VERSION


def bounded_name_count_mappings_from_signature_payloads(
    signatures: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Count canonical name keys in one exact bounded signature payload."""

    from s2and.text import canonical_name_count_keys, canonicalize_name_parts

    counters: dict[str, Counter[str]] = {
        key: Counter() for key in ("first", "last", "first_last", "last_first_initial")
    }
    for signature in signatures.values():
        if not isinstance(signature, Mapping):
            raise TypeError("bounded signatures must contain object records")
        author_info = signature.get("author_info") or {}
        if not isinstance(author_info, Mapping):
            raise TypeError("bounded signature author_info must be an object")
        keys = canonical_name_count_keys(
            canonicalize_name_parts(
                author_info.get("first"),
                author_info.get("middle"),
                author_info.get("last"),
            )
        )
        for key, value in keys.items():
            if value is not None:
                counters[key][value] += 1
    return (
        dict(counters["first"]),
        dict(counters["last"]),
        dict(counters["first_last"]),
        dict(counters["last_first_initial"]),
    )


def write_bounded_name_counts_index(
    signatures: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[str, str]:
    """Write a canonical bounded name-count index and return its logical digest."""

    from s2and.consts import NORMALIZATION_VERSION

    mappings = bounded_name_count_mappings_from_signature_payloads(signatures)
    encoded_signatures = json.dumps(signatures, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    records_sha256 = hashlib.sha256(encoded_signatures).hexdigest()
    provenance = {
        "schema_version": NAME_COUNTS_PROVENANCE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "generation_id": f"bounded-{records_sha256[:16]}",
        "source_snapshot_id": f"bounded-json-{records_sha256[:16]}",
        "source_kind": "verification:bounded-json",
        "source_query_sha256": hashlib.sha256(b"bounded-name-counts-v1").hexdigest(),
        "selected_rows_sha256": records_sha256,
        "source_row_count": len(signatures),
    }
    index_path, _metrics = write_name_counts_index(
        output_dir,
        mappings,
        provenance,
        overwrite=True,
    )
    logical_payload = json.dumps(
        {"mappings": mappings, "provenance": provenance},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return index_path, hashlib.sha256(logical_payload).hexdigest()


def write_feature_block_arrow_from_anddata(
    dataset: Any,
    output_dir: str | Path,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Build a `FeatureBlock` from `ANDData` and write complete Arrow IPC tables."""

    feature_block = feature_block_from_anddata(
        dataset,
        signature_ids=signature_ids,
        query_signature_ids=query_signature_ids,
        cluster_seeds_require=cluster_seeds_require,
        include_specter=include_specter,
    )
    return write_feature_block_arrow_tables(
        feature_block,
        output_dir,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        max_record_batch_rows=max_record_batch_rows,
        overwrite=overwrite,
    )


def feature_block_from_anddata(
    dataset: Any,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
) -> FeatureBlock:
    """Build a `FeatureBlock` view from an existing `ANDData`-like object."""

    resolved_signature_ids = tuple(
        str(value) for value in (dataset.signatures.keys() if signature_ids is None else signature_ids)
    )
    signatures = tuple(
        _feature_block_signature_from_anddata(dataset, signature_id) for signature_id in resolved_signature_ids
    )
    papers = _feature_block_papers_from_anddata(dataset, signatures)
    paper_authors = _feature_block_paper_authors_from_papers(
        papers_by_id={paper.paper_id: paper for paper in papers}, dataset=dataset
    )

    signature_id_set = set(resolved_signature_ids)
    source_cluster_seeds = dict(
        getattr(dataset, "cluster_seeds_require", {}) if cluster_seeds_require is None else cluster_seeds_require
    )
    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in source_cluster_seeds.items()
        if str(signature_id) in signature_id_set
    )
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        getattr(dataset, "cluster_seeds_disallow", set()),
        signature_id_set,
    )
    specter_paper_ids, specter_embeddings = _feature_block_specter_from_anddata(
        dataset,
        papers,
        include_specter=include_specter,
    )
    return FeatureBlock(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=tuple(str(value) for value in query_signature_ids),
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def _feature_block_signature_from_anddata(dataset: Any, signature_id: str) -> FeatureBlockSignature:
    signature = dataset.signatures[str(signature_id)]
    return FeatureBlockSignature(
        signature_id=str(signature_id),
        paper_id=str(signature.paper_id),
        author_first=_optional_str(getattr(signature, "author_info_first", None)),
        author_middle=_optional_str(getattr(signature, "author_info_middle", None)),
        author_last=_optional_str(getattr(signature, "author_info_last", None)),
        author_suffix=_optional_str(getattr(signature, "author_info_suffix", None)),
        author_affiliations=_strict_string_tuple(
            getattr(signature, "author_info_affiliations", None),
            field_name="signatures.author_info_affiliations",
        ),
        author_orcid=_optional_str(getattr(signature, "author_info_orcid", None)),
        author_position=_optional_int(
            getattr(signature, "author_info_position", None),
            field_name="signatures.author_info_position",
        ),
        author_block=_optional_str(getattr(signature, "author_info_block", None)),
        author_email=_optional_str(getattr(signature, "author_info_email", None)),
        source_author_ids=_strict_string_tuple(
            getattr(signature, "sourced_author_ids", None),
            field_name="signatures.sourced_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_papers_from_anddata(
    dataset: Any,
    signatures: Sequence[FeatureBlockSignature],
) -> tuple[FeatureBlockPaper, ...]:
    papers: list[FeatureBlockPaper] = []
    seen: set[str] = set()
    for signature in signatures:
        paper_id = str(signature.paper_id)
        if paper_id in seen:
            continue
        paper = getattr(dataset, "papers", {}).get(paper_id)
        if paper is None:
            raise ValueError(f"ANDData papers are missing signature paper_id: {paper_id!r}")
        seen.add(paper_id)
        is_reliable = _optional_bool(getattr(paper, "is_reliable", None), field_name="papers.is_reliable")
        papers.append(
            FeatureBlockPaper(
                paper_id=paper_id,
                title=_optional_str(getattr(paper, "title", None)),
                abstract="Has Abstract" if bool(getattr(paper, "has_abstract", False)) else "",
                venue=_optional_str(getattr(paper, "venue", None)),
                journal_name=_optional_str(getattr(paper, "journal_name", None)),
                year=_optional_int(getattr(paper, "year", None), field_name="papers.year"),
                predicted_language=_optional_str(getattr(paper, "predicted_language", None)),
                is_reliable=is_reliable,
                language_reliability=getattr(paper, "language_reliability", None),
            )
        )
    return tuple(papers)


def _feature_block_paper_authors_from_papers(
    *,
    papers_by_id: Mapping[str, FeatureBlockPaper],
    dataset: Any,
) -> tuple[FeatureBlockPaperAuthor, ...]:
    rows: list[FeatureBlockPaperAuthor] = []
    dataset_papers = getattr(dataset, "papers", {})
    for paper_id in papers_by_id:
        paper = dataset_papers[str(paper_id)]
        for index, author in enumerate(getattr(paper, "authors", None) or ()):
            position = _optional_int(getattr(author, "position", index), field_name="papers.authors.position")
            rows.append(
                FeatureBlockPaperAuthor(
                    paper_id=paper_id,
                    position=index if position is None else position,
                    author_name=str(getattr(author, "author_name", "") or ""),
                )
            )
    return tuple(rows)


def _feature_block_specter_from_anddata(
    dataset: Any,
    papers: Sequence[FeatureBlockPaper],
    *,
    include_specter: bool,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not include_specter:
        return (), None
    specter = getattr(dataset, "specter_embeddings", None)
    if specter is None:
        return (), None
    paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper in papers:
        vector = specter.get(str(paper.paper_id))
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper.paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors in a FeatureBlock must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper.paper_id!r}"
            )
        paper_ids.append(str(paper.paper_id))
        vectors.append(array)
    if not vectors:
        return (), np.empty((0, 1), dtype=np.float32)
    return tuple(paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
