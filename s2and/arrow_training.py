"""Arrow-native training ingestion.

Builds a train-mode :class:`~s2and.data.ANDData` from the same Arrow
prediction artifacts the inference runtime consumes (``signatures.arrow``,
``papers.arrow``, ``paper_authors.arrow``, ``specter.arrow`` plus the
raw-planner batch indexes and the ``name_counts_index/`` sidecar), and routes
featurization through ``RustFeaturizer.from_arrow_paths`` — the only Rust
featurizer constructor.

What stays outside Arrow, by design:

- **Ground-truth clusters** remain JSON (the converters copy
  ``{dataset}_clusters.json`` verbatim next to the Arrow tables); pass that
  path/dict as ``clusters``.
- **Iteration order**: Arrow tables are sorted by id, so split construction
  and pair sampling are deterministic but not pair-identical to a JSON-ingested
  dataset whose insertion order differed. Feature values for any given pair are
  parity-gated identical (tests/test_arrow_training_ingestion.py).

Arrow text and name columns carry source/raw preprocessing inputs, and Rust
performs canonical preprocessing while building the scoring view. See
``docs/rust/arrow_dataset_spec.md`` for the authoritative contract.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from s2and.arrow_inputs import (
    require_normalization_version,
    validate_arrow_training_artifacts,
)
from s2and.arrow_schema import validate_arrow_file_schema, validate_arrow_schema
from s2and.data import ANDData, Author, Paper, Signature

logger = logging.getLogger("s2and")

if TYPE_CHECKING:
    import pandas as pd


def _iter_arrow_rows(
    path: str | Path,
    *,
    table_name: str,
    required_columns: set[str],
) -> Any:
    """Yield rows one record batch at a time from a memory-mapped IPC file."""

    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        validate_arrow_schema(reader.schema, table_name=table_name)
        missing_from_loader = sorted(required_columns.difference(reader.schema.names))
        if missing_from_loader:
            raise ValueError(f"Arrow {table_name} table is missing loader-required columns: {missing_from_loader}")
        row_index = 0
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            for row in batch.to_pylist():
                yield row_index, row
                row_index += 1


def _required_id_value(raw_value: Any, table_name: str, column_name: str, row_index: int) -> str:
    if raw_value is None:
        raise ValueError(f"Arrow {table_name} table contains null {column_name} at row {row_index}")
    value = str(raw_value)
    if value == "":
        raise ValueError(f"Arrow {table_name} table contains empty {column_name} at row {row_index}")
    return value


def load_signatures_from_arrow(path: str | Path) -> dict[str, Signature]:
    """Read the minimal Python signature metadata Rust-backed training needs."""

    required_columns = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_affiliations",
        "author_position",
        "author_block",
    }
    signatures: dict[str, Signature] = {}
    for row_index, row in _iter_arrow_rows(
        path,
        table_name="signatures",
        required_columns=required_columns,
    ):
        signature_id = _required_id_value(row.get("signature_id"), "signatures", "signature_id", row_index)
        if signature_id in signatures:
            raise ValueError(f"signatures Arrow contains duplicate signature_id={signature_id!r}")
        paper_id = _required_id_value(row.get("paper_id"), "signatures", "paper_id", row_index)
        author_block = _required_id_value(row.get("author_block"), "signatures", "author_block", row_index)
        orcid = row.get("author_orcid")
        signatures[signature_id] = Signature(
            author_info_first=row["author_first"],
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle=row["author_middle"],
            author_info_middle_normalized_without_apostrophe=None,
            author_info_last_normalized=None,
            author_info_last=row["author_last"],
            author_info_suffix_normalized=None,
            author_info_suffix=row["author_suffix"],
            author_info_coauthors=None,
            author_info_coauthor_blocks=None,
            author_info_full_name=None,
            author_info_affiliations=list(row["author_affiliations"] or []),
            author_info_affiliations_n_grams=None,
            author_info_coauthor_n_grams=None,
            author_info_email=row.get("author_email"),
            author_info_orcid=str(orcid) if orcid else None,
            author_info_name_counts=None,
            author_info_position=int(row["author_position"]),
            author_info_block=author_block,
            author_info_given_block=None,
            author_info_estimated_gender=None,
            author_info_estimated_ethnicity=None,
            paper_id=paper_id,
            sourced_author_source=None,
            sourced_author_ids=list(row.get("source_author_ids") or []),
            author_id=None,
            signature_id=signature_id,
        )
    if not signatures:
        raise ValueError(f"Arrow signatures table has no rows: {path}")
    return signatures


def load_papers_from_arrow(
    papers_path: str | Path,
    paper_authors_path: str | Path,
    *,
    needed_paper_ids: set[str] | None = None,
) -> dict[str, Paper]:
    """Read the minimal Python paper metadata Rust-backed training needs.

    The Arrow bundle stores the abstract as a has-abstract sentinel, which is
    fine for training because ``ANDData`` reduces the abstract to a boolean.
    """

    authors_by_paper_id: dict[str, list[Author]] = {}
    paper_author_keys: set[tuple[str, int]] = set()
    for row_index, row in _iter_arrow_rows(
        paper_authors_path,
        table_name="paper_authors",
        required_columns={"paper_id", "position", "author_name"},
    ):
        paper_id = _required_id_value(row.get("paper_id"), "paper_authors", "paper_id", row_index)
        if needed_paper_ids is not None and paper_id not in needed_paper_ids:
            continue
        raw_author_name = row.get("author_name")
        if not isinstance(raw_author_name, str) or not raw_author_name.strip():
            raise ValueError(f"Arrow paper_authors table contains empty author_name at row {row_index}")
        raw_position = row.get("position")
        if raw_position is None:
            raise ValueError(f"Arrow paper_authors table contains null position at row {row_index}")
        position = int(raw_position)
        author_key = (paper_id, position)
        if author_key in paper_author_keys:
            raise ValueError(f"paper_authors Arrow contains duplicate (paper_id, position)=({paper_id!r}, {position})")
        paper_author_keys.add(author_key)
        authors_by_paper_id.setdefault(paper_id, []).append(Author(author_name=raw_author_name, position=position))
    del paper_author_keys

    papers: dict[str, Paper] = {}
    source_has_papers = False
    for row_index, row in _iter_arrow_rows(
        papers_path,
        table_name="papers",
        required_columns={"paper_id", "title", "venue", "journal_name"},
    ):
        source_has_papers = True
        paper_id = _required_id_value(row.get("paper_id"), "papers", "paper_id", row_index)
        if needed_paper_ids is not None and paper_id not in needed_paper_ids:
            continue
        if paper_id in papers:
            raise ValueError(f"papers Arrow contains duplicate paper_id={paper_id!r}")
        papers[paper_id] = Paper(
            title=row["title"],
            has_abstract=row.get("abstract") not in {"", None},
            in_signatures=True,
            is_english=None,
            is_reliable=None,
            language_reliability=None,
            predicted_language=None,
            title_ngrams_words=None,
            authors=authors_by_paper_id.pop(paper_id, []),
            venue=row["venue"],
            journal_name=row["journal_name"],
            title_ngrams_chars=None,
            venue_ngrams=None,
            journal_ngrams=None,
            year=row.get("year"),
            paper_id=paper_id,
        )
    if not source_has_papers:
        raise ValueError(f"Arrow papers table has no rows: {papers_path}")
    if needed_paper_ids is None:
        unknown_author_paper_ids = sorted(authors_by_paper_id)
        if unknown_author_paper_ids:
            raise ValueError(
                "paper_authors Arrow references paper_id values absent from papers Arrow: "
                f"{unknown_author_paper_ids[:10]}"
            )
    else:
        missing_paper_ids = sorted(needed_paper_ids.difference(papers))
        if missing_paper_ids:
            raise ValueError(
                f"signatures Arrow references paper_id values absent from papers Arrow: {missing_paper_ids[:10]}"
            )
    return papers


def build_training_anddata_from_arrow(
    arrow_paths: Mapping[str, Any],
    name: str,
    *,
    expected_normalization_version: str,
    clusters: str | dict | None = None,
    train_pairs: str | pd.DataFrame | None = None,
    val_pairs: str | pd.DataFrame | None = None,
    test_pairs: str | pd.DataFrame | None = None,
    block_type: str = "s2",
    train_pairs_size: int = 30_000,
    val_pairs_size: int = 5_000,
    test_pairs_size: int = 5_000,
    random_seed: int = 1111,
    n_jobs: int = 1,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
) -> ANDData:
    """Build a fully initialized Rust-backed train ``ANDData``.

    The bundle must include raw-planner indexes and ``name_counts_index``.
    Python SPECTER and name-count values are never materialized; Rust reads
    them directly from the one immutable ``dataset.arrow_paths`` mapping.
    """

    expected_version = require_normalization_version(
        expected_normalization_version,
        context="arrow-native training ingestion",
    )
    ingest_start = time.perf_counter()
    path_keys = {str(key) for key in arrow_paths}
    normalized_arrow_paths = validate_arrow_training_artifacts(
        arrow_paths,
        require_specter="specter" in path_keys,
        require_name_counts_index=True,
        expected_normalization_version=expected_version,
        context="arrow-native training ingestion",
        producer_hint=(
            "include signatures, papers, paper_authors, raw-planner batch indexes, "
            "name_counts_index, and model-required specter"
        ),
    )
    if "specter" in normalized_arrow_paths:
        validate_arrow_file_schema(normalized_arrow_paths["specter"], table_name="specter")
    training_arrow_paths = normalized_arrow_paths.without("query_signatures")
    signatures = load_signatures_from_arrow(training_arrow_paths["signatures"])
    needed_paper_ids = {str(signature.paper_id) for signature in signatures.values()}
    papers = load_papers_from_arrow(
        training_arrow_paths["papers"],
        training_arrow_paths["paper_authors"],
        needed_paper_ids=needed_paper_ids,
    )
    del needed_paper_ids

    dataset = ANDData._from_validated_arrow_training(
        signatures=signatures,
        papers=papers,
        name=name,
        arrow_paths=training_arrow_paths,
        clusters=clusters,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        block_type=block_type,
        train_pairs_size=train_pairs_size,
        val_pairs_size=val_pairs_size,
        test_pairs_size=test_pairs_size,
        random_seed=random_seed,
        n_jobs=n_jobs,
        name_tuples=name_tuples,
    )
    logger.debug(
        "Telemetry stage: stage=arrow_training_ingest seconds=%.3f signatures=%d papers=%d specter=%s",
        time.perf_counter() - ingest_start,
        len(signatures),
        len(papers),
        "yes" if training_arrow_paths.get("specter") else "no",
    )
    return dataset
