"""Arrow-native training ingestion.

Builds a train-mode :class:`~s2and.data.ANDData` from an open
:class:`~s2and.arrow_inputs.ArrowDataset` and routes featurization through its
retained native resource.

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
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_schema import validate_arrow_schema
from s2and.data import ANDData, Author, Paper, Signature

logger = logging.getLogger("s2and")

if TYPE_CHECKING:
    import pandas as pd


def _iter_arrow_rows(
    source: Any,
    *,
    table_name: str,
    required_columns: set[str],
) -> Any:
    """Yield rows one record batch at a time from a memory-mapped IPC file."""

    import pyarrow as pa

    source_context = pa.memory_map(str(source), "r") if isinstance(source, str | Path) else nullcontext(source)
    with source_context as opened:
        reader = pa.ipc.open_file(opened)
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


def load_signatures_from_arrow(
    path: str | Path | BinaryIO,
    *,
    use_orcid_id: bool = True,
) -> dict[str, Signature]:
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
            author_info_orcid=str(orcid) if use_orcid_id and orcid else None,
            author_info_name_counts=None,
            author_info_position=int(row["author_position"]),
            author_info_block=author_block,
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
    papers_path: str | Path | BinaryIO,
    paper_authors_path: str | Path | BinaryIO,
    *,
    needed_paper_ids: set[str] | None = None,
) -> dict[str, Paper]:
    """Read the minimal Python paper metadata Rust-backed training needs.

    The Arrow bundle stores the abstract as a has-abstract sentinel, which is
    fine for training because ``ANDData`` reduces the abstract to a boolean.
    """

    authors_by_paper_id: dict[str, list[Author]] = {}
    for row_index, row in _iter_arrow_rows(
        paper_authors_path,
        table_name="paper_authors",
        required_columns={"paper_id", "position", "author_name"},
    ):
        paper_id = _required_id_value(row.get("paper_id"), "paper_authors", "paper_id", row_index)
        if needed_paper_ids is not None and paper_id not in needed_paper_ids:
            continue
        raw_author_name = row.get("author_name")
        if raw_author_name is None:
            raise ValueError(f"Arrow paper_authors table contains null author_name at row {row_index}")
        if not isinstance(raw_author_name, str):
            raise ValueError(f"Arrow paper_authors table contains non-string author_name at row {row_index}")
        raw_position = row.get("position")
        if raw_position is None:
            raise ValueError(f"Arrow paper_authors table contains null position at row {row_index}")
        position = int(raw_position)
        authors_by_paper_id.setdefault(paper_id, []).append(Author(author_name=raw_author_name, position=position))
    for paper_id, authors in authors_by_paper_id.items():
        _validate_unique_author_positions(paper_id, authors)

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


def _validate_unique_author_positions(paper_id: str, authors: list[Author]) -> None:
    """Reject duplicate positions while retaining state for only one paper."""

    if len(authors) < 2:
        return
    positions: set[int] = set()
    for author in authors:
        if author.position in positions:
            raise ValueError(
                f"paper_authors Arrow contains duplicate (paper_id, position)=({paper_id!r}, {author.position})"
            )
        positions.add(author.position)


def build_training_anddata_from_arrow(
    arrow_dataset: ArrowDataset,
    name: str,
    *,
    clusters: str | dict | None = None,
    cluster_seeds: str | dict | None = None,
    altered_cluster_signatures: str | list | set | None = None,
    train_pairs: str | pd.DataFrame | None = None,
    val_pairs: str | pd.DataFrame | None = None,
    test_pairs: str | pd.DataFrame | None = None,
    train_pairs_size: int = 30_000,
    val_pairs_size: int = 5_000,
    test_pairs_size: int = 5_000,
    random_seed: int = 1111,
    n_jobs: int = 1,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    use_orcid_id: bool = True,
) -> ANDData:
    """Build a fully initialized Rust-backed train ``ANDData``.

    ``arrow_dataset`` must remain open while the returned dataset is used and
    must include ``name_counts_index``.
    Python SPECTER and name-count values are never materialized; Rust reads
    them directly from the retained native dataset resource.
    Set ``use_orcid_id=False`` to remove ORCID evidence from both the
    Python-visible signatures and the Rust featurizer.
    Training cluster-seed constraints remain explicit inputs rather than part
    of the immutable dataset identity.
    """

    if not isinstance(arrow_dataset, ArrowDataset):
        raise TypeError("build_training_anddata_from_arrow requires an open ArrowDataset")
    ingest_start = time.perf_counter()
    with arrow_dataset.use(
        require_name_counts_index=True,
    ) as lease:
        has_specter = lease.has("specter")
        with lease.open_file("signatures") as signatures_source:
            signatures = load_signatures_from_arrow(
                signatures_source,
                use_orcid_id=use_orcid_id,
            )

    dataset = ANDData._from_arrow_training(
        signatures=signatures,
        name=name,
        arrow_dataset=arrow_dataset,
        clusters=clusters,
        cluster_seeds=cluster_seeds,
        altered_cluster_signatures=altered_cluster_signatures,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_pairs_size=train_pairs_size,
        val_pairs_size=val_pairs_size,
        test_pairs_size=test_pairs_size,
        random_seed=random_seed,
        n_jobs=n_jobs,
        name_tuples=name_tuples,
        use_orcid_id=use_orcid_id,
    )
    logger.debug(
        "Telemetry stage: stage=arrow_training_ingest seconds=%.3f signatures=%d papers=%d specter=%s",
        time.perf_counter() - ingest_start,
        len(signatures),
        len(dataset._arrow_paper_ids) if dataset._arrow_paper_ids is not None else 0,
        "yes" if has_specter else "no",
    )
    return dataset
