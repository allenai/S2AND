"""Packaged FeatureBlock conversion helpers for writing Arrow IPC tables.

These let a service (e.g. the author-disambiguation service) produce the Arrow IPC
artifacts that ``Clusterer.predict_from_arrow_paths`` consumes, without depending on the
unpackaged ``scripts/`` tree.

- ``write_feature_block_arrow_from_rows`` is the preferred production path: build the tables
  straight from service-native rows, no ``ANDData``.
- ``write_feature_block_arrow_from_anddata`` is a compatibility bridge for callers that already
  hold an ``ANDData`` (also re-exported by ``scripts/arrow_conversion_helpers.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from s2and.incremental_linking.feature_block_arrow import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    write_feature_block_arrow_tables,
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
    normalize_cluster_seed_disallow_pairs,
)


def write_feature_block_arrow_from_rows(
    output_dir: str | Path,
    *,
    signatures: Sequence[Mapping[str, Any]],
    papers: Sequence[Mapping[str, Any]],
    paper_authors: Sequence[Mapping[str, Any]],
    specter_embeddings: Mapping[str, Sequence[float]] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    query_signature_ids: Sequence[Any] = (),
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Build a FeatureBlock from service-native rows and write Arrow IPC tables (no ANDData).

    Row mappings use the Arrow column names (see the schema contract). cluster_seeds_require maps
    signature_id -> cluster_id (already grouped); cluster_seeds_disallow is an iterable of
    signature-id pairs.
    """

    feature_block = feature_block_from_rows(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        specter_embeddings=specter_embeddings,
        cluster_seeds_require=cluster_seeds_require,
        cluster_seeds_disallow=cluster_seeds_disallow,
        query_signature_ids=query_signature_ids,
    )
    return write_feature_block_arrow_tables(
        feature_block,
        output_dir,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        max_record_batch_rows=max_record_batch_rows,
        overwrite=overwrite,
    )


def feature_block_from_rows(
    *,
    signatures: Sequence[Mapping[str, Any]],
    papers: Sequence[Mapping[str, Any]],
    paper_authors: Sequence[Mapping[str, Any]],
    specter_embeddings: Mapping[str, Sequence[float]] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    query_signature_ids: Sequence[Any] = (),
) -> FeatureBlock:
    """Build a FeatureBlock view from service-native row mappings (no ANDData)."""

    block_signatures = tuple(_feature_block_signature_from_row(row) for row in signatures)
    signature_id_set = {sig.signature_id for sig in block_signatures}
    block_papers = tuple(_feature_block_paper_from_row(row) for row in papers)
    block_paper_authors = tuple(_feature_block_paper_author_from_row(row) for row in paper_authors)

    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in (cluster_seeds_require or {}).items()
        if str(signature_id) in signature_id_set
    )
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow or ()),
        signature_id_set,
    )
    specter_paper_ids, specter_matrix = _specter_from_mapping(block_papers, specter_embeddings)
    return FeatureBlock(
        signatures=block_signatures,
        papers=block_papers,
        paper_authors=block_paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=tuple(str(value) for value in query_signature_ids),
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_matrix,
    )


def _feature_block_signature_from_row(row: Mapping[str, Any]) -> FeatureBlockSignature:
    return FeatureBlockSignature(
        signature_id=str(row["signature_id"]),
        paper_id=str(row["paper_id"]),
        author_first=_optional_str(row.get("author_first")),
        author_middle=_optional_str(row.get("author_middle")),
        author_last=_optional_str(row.get("author_last")),
        author_suffix=_optional_str(row.get("author_suffix")),
        author_affiliations=_strict_string_tuple(
            row.get("author_affiliations"), field_name="signatures.author_affiliations"
        ),
        author_orcid=_optional_str(row.get("author_orcid")),
        author_position=_optional_int(row.get("author_position"), field_name="signatures.author_position"),
        author_block=_optional_str(row.get("author_block")),
        author_email=_optional_str(row.get("author_email")),
        source_author_ids=_strict_string_tuple(
            row.get("source_author_ids"),
            field_name="signatures.source_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_paper_from_row(row: Mapping[str, Any]) -> FeatureBlockPaper:
    return FeatureBlockPaper(
        paper_id=str(row["paper_id"]),
        title=_optional_str(row.get("title")),
        abstract="Has Abstract" if bool(row.get("has_abstract", False)) else "",
        venue=_optional_str(row.get("venue")),
        journal_name=_optional_str(row.get("journal_name")),
        year=_optional_int(row.get("year"), field_name="papers.year"),
        predicted_language=_optional_str(row.get("predicted_language")),
        is_reliable=_optional_bool(row.get("is_reliable"), field_name="papers.is_reliable"),
    )


def _feature_block_paper_author_from_row(row: Mapping[str, Any]) -> FeatureBlockPaperAuthor:
    position = _optional_int(row.get("position"), field_name="paper_authors.position")
    return FeatureBlockPaperAuthor(
        paper_id=str(row["paper_id"]),
        position=0 if position is None else position,
        author_name=str(row.get("author_name", "") or ""),
    )


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
        papers.append(
            FeatureBlockPaper(
                paper_id=paper_id,
                title=_optional_str(getattr(paper, "title", None)),
                abstract="Has Abstract" if bool(getattr(paper, "has_abstract", False)) else "",
                venue=_optional_str(getattr(paper, "venue", None)),
                journal_name=_optional_str(getattr(paper, "journal_name", None)),
                year=_optional_int(getattr(paper, "year", None), field_name="papers.year"),
                predicted_language=_optional_str(getattr(paper, "predicted_language", None)),
                is_reliable=_optional_bool(getattr(paper, "is_reliable", None), field_name="papers.is_reliable"),
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
    return _specter_from_mapping(papers, getattr(dataset, "specter_embeddings", None))


def _specter_from_mapping(
    papers: Sequence[FeatureBlockPaper],
    specter_embeddings: Mapping[str, Sequence[float]] | None,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not specter_embeddings:
        return (), None
    paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper in papers:
        vector = specter_embeddings.get(str(paper.paper_id))
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
        return (), None
    return tuple(paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
