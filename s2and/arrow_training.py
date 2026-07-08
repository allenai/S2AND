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

Text columns hold ANDData-preprocessed values; re-ingesting with
``preprocess=True`` relies on the same normalize-idempotency contract the
Rust Arrow readers already assume (docs/rust/ingest_source_policy_inventory.md).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import validate_arrow_prediction_artifacts
from s2and.data import NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR, ANDData
from s2and.runtime import RUST_FEATURIZER_ARROW_PATHS_ATTR

logger = logging.getLogger("s2and")

_REQUIRED_TABLE_KEYS = ("signatures", "papers", "paper_authors")


def _read_arrow_table(path: str | Path) -> Any:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _require_columns(table: Any, table_name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(table.column_names))
    if missing:
        raise ValueError(f"Arrow {table_name} table is missing required columns: {missing}")


def load_signatures_dict_from_arrow(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``signatures.arrow`` into the JSON-shaped dict ``ANDData`` accepts."""

    table = _read_arrow_table(path)
    _require_columns(
        table,
        "signatures",
        {
            "signature_id",
            "paper_id",
            "author_first",
            "author_middle",
            "author_last",
            "author_suffix",
            "author_affiliations",
            "author_position",
        },
    )
    signatures: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        signature_id = str(row["signature_id"])
        orcid = row.get("author_orcid")
        signatures[signature_id] = {
            "signature_id": signature_id,
            "paper_id": str(row["paper_id"]),
            "author_info": {
                "first": row["author_first"],
                "middle": row["author_middle"],
                "last": row["author_last"],
                "suffix": row["author_suffix"],
                "affiliations": list(row["author_affiliations"] or []),
                "email": row.get("author_email"),
                "position": row["author_position"],
                "block": row.get("author_block"),
                # ANDData derives author_info_orcid from source_ids +
                # source_id_source; the Arrow column stores the derived value,
                # so reconstruct the source shape it expects.
                "source_ids": [orcid] if orcid else None,
                "source_id_source": "ORCID" if orcid else None,
            },
            "sourced_author_ids": list(row.get("source_author_ids") or []),
        }
    if not signatures:
        raise ValueError(f"Arrow signatures table has no rows: {path}")
    return signatures


def load_papers_dict_from_arrow(
    papers_path: str | Path,
    paper_authors_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Read ``papers.arrow`` + ``paper_authors.arrow`` into the JSON paper shape.

    The Arrow bundle stores the abstract as a has-abstract sentinel, which is
    fine for training because ``ANDData`` reduces the abstract to a boolean.
    """

    authors_table = _read_arrow_table(paper_authors_path)
    _require_columns(authors_table, "paper_authors", {"paper_id", "position", "author_name"})
    authors_by_paper_id: dict[str, list[dict[str, Any]]] = {}
    for row in authors_table.to_pylist():
        authors_by_paper_id.setdefault(str(row["paper_id"]), []).append(
            {"author_name": row["author_name"], "position": int(row["position"])}
        )

    papers_table = _read_arrow_table(papers_path)
    _require_columns(papers_table, "papers", {"paper_id", "title", "venue", "journal_name"})
    papers: dict[str, dict[str, Any]] = {}
    for row in papers_table.to_pylist():
        paper_id = str(row["paper_id"])
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": row["title"],
            "abstract": row.get("abstract") or "",
            "venue": row["venue"],
            "journal_name": row["journal_name"],
            "year": row.get("year"),
            "authors": authors_by_paper_id.get(paper_id, []),
        }
    if not papers:
        raise ValueError(f"Arrow papers table has no rows: {papers_path}")
    return papers


def load_specter_tuple_from_arrow(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read ``specter.arrow`` into the ``(matrix, keys)`` tuple ANDData accepts."""

    table = _read_arrow_table(path)
    _require_columns(table, "specter", {"paper_id", "embedding"})
    keys = [str(value) for value in table.column("paper_id").to_pylist()]
    embedding_column = table.column("embedding").combine_chunks()
    dimension = int(embedding_column.type.list_size)
    flat = np.asarray(embedding_column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    matrix = flat.reshape(len(keys), dimension)
    return matrix, keys


def attach_training_arrow_featurizer_paths(
    dataset: ANDData,
    arrow_paths: Mapping[str, Any],
) -> dict[str, str]:
    """Validate and attach Arrow paths so featurization uses ``from_arrow_paths``.

    Requires the raw-planner batch indexes and the ``name_counts_index/``
    sidecar because training featurization loads name counts in Rust.
    """

    semantics = getattr(dataset, "name_counts_last_first_initial_semantics", None)
    if semantics != NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR:
        raise ValueError(
            "Arrow-backed featurization requires name_counts_last_first_initial_semantics="
            f"{NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR!r}, got {semantics!r}"
        )
    path_keys = {str(key) for key in arrow_paths}
    normalized = validate_arrow_prediction_artifacts(
        arrow_paths,
        require_specter="specter" in path_keys or "specter2" in path_keys,
        require_name_counts_index=True,
        require_batch_indexes=True,
        context="arrow-native training featurization",
        producer_hint=(
            "write the training bundle with signatures/papers/paper_authors tables, raw-planner "
            "batch indexes (write_raw_arrow_batch_lookup_indexes), specter, and the "
            "name_counts_index sidecar (write_name_counts_index)"
        ),
    )
    normalized.pop("query_signatures", None)
    setattr(dataset, RUST_FEATURIZER_ARROW_PATHS_ATTR, normalized)
    # Rust production prediction (Clusterer.predict / cluster_eval) resolves
    # explicit dataset.arrow_paths; an Arrow-ingested training dataset carries
    # the same artifacts, so expose them for prediction too.
    dataset.arrow_paths = dict(normalized)
    return normalized


def build_training_anddata_from_arrow(
    arrow_paths: Mapping[str, Any],
    name: str,
    *,
    clusters: str | dict | None = None,
    mode: str = "train",
    load_name_counts: bool | dict = False,
    attach_rust_featurizer: bool = True,
    **anddata_kwargs: Any,
) -> ANDData:
    """Build a train-mode ``ANDData`` from Arrow artifacts.

    ``arrow_paths`` must contain at least ``signatures``, ``papers``, and
    ``paper_authors`` (``specter`` strongly recommended); with
    ``attach_rust_featurizer=True`` (default) it must also carry the batch
    indexes and ``name_counts_index`` so featurization can run through
    ``RustFeaturizer.from_arrow_paths``. Ground-truth ``clusters`` (or fixed
    train/val/test pairs via ``anddata_kwargs``) come from JSON exactly as in
    the JSON training path.

    ``load_name_counts`` defaults to False because the Rust featurizer reads
    name counts from the ``name_counts_index`` sidecar; pass True (or a dict)
    only if Python-side per-signature name counts are needed (e.g. for the
    Python reference featurizer).
    """

    missing = [key for key in _REQUIRED_TABLE_KEYS if key not in arrow_paths]
    if missing:
        raise ValueError(f"arrow_paths is missing required tables: {missing}")

    ingest_start = time.perf_counter()
    path_keys = {str(key) for key in arrow_paths}
    normalized_arrow_paths = validate_arrow_prediction_artifacts(
        arrow_paths,
        require_specter="specter" in path_keys or "specter2" in path_keys,
        require_name_counts_index=False,
        require_batch_indexes=False,
        context="arrow-native training ingestion",
        producer_hint="include signatures, papers, paper_authors, and optional specter/specter2 Arrow tables",
    )
    signatures = load_signatures_dict_from_arrow(normalized_arrow_paths["signatures"])
    papers = load_papers_dict_from_arrow(normalized_arrow_paths["papers"], normalized_arrow_paths["paper_authors"])
    specter = (
        load_specter_tuple_from_arrow(normalized_arrow_paths["specter"])
        if normalized_arrow_paths.get("specter")
        else None
    )
    logger.debug(
        "Telemetry stage: stage=arrow_training_ingest seconds=%.3f signatures=%d papers=%d specter=%s",
        time.perf_counter() - ingest_start,
        len(signatures),
        len(papers),
        "yes" if specter is not None else "no",
    )

    dataset = ANDData(
        signatures=signatures,
        papers=papers,
        name=name,
        mode=mode,
        clusters=clusters,
        specter_embeddings=specter,
        load_name_counts=load_name_counts,
        rust_arrow_featurization=attach_rust_featurizer,
        **anddata_kwargs,
    )
    if attach_rust_featurizer:
        attach_training_arrow_featurizer_paths(dataset, normalized_arrow_paths)
    return dataset
