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
from s2and.incremental_linking.feature_block_arrow import (
    _arrow_rows_by_unique_key,
    _read_arrow_ipc_table,
    _require_arrow_columns,
)
from s2and.runtime import RUST_FEATURIZER_ARROW_PATHS_ATTR, dataset_stage_uses_rust

logger = logging.getLogger("s2and")

_REQUIRED_TABLE_KEYS = ("signatures", "papers", "paper_authors")


def _read_arrow_table(path: str | Path) -> Any:
    import pyarrow as pa

    return _read_arrow_ipc_table(pa, path)


def _required_id_value(raw_value: Any, table_name: str, column_name: str, row_index: int) -> str:
    if raw_value is None:
        raise ValueError(f"Arrow {table_name} table contains null {column_name} at row {row_index}")
    value = str(raw_value)
    if value == "":
        raise ValueError(f"Arrow {table_name} table contains empty {column_name} at row {row_index}")
    return value


def load_signatures_dict_from_arrow(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read ``signatures.arrow`` into the JSON-shaped dict ``ANDData`` accepts."""

    required_columns = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_affiliations",
        "author_position",
    }
    signatures: dict[str, dict[str, Any]] = {}
    table = _read_arrow_table(path)
    _require_arrow_columns(table, "signatures", required_columns)
    rows = table.to_pylist()
    _arrow_rows_by_unique_key(rows, table_name="signatures", key_column="signature_id")
    for row_index, row in enumerate(rows):
        signature_id = _required_id_value(row.get("signature_id"), "signatures", "signature_id", row_index)
        paper_id = _required_id_value(row.get("paper_id"), "signatures", "paper_id", row_index)
        orcid = row.get("author_orcid")
        signatures[signature_id] = {
            "signature_id": signature_id,
            "paper_id": paper_id,
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

    authors_by_paper_id: dict[str, list[dict[str, Any]]] = {}
    authors_table = _read_arrow_table(paper_authors_path)
    _require_arrow_columns(authors_table, "paper_authors", {"paper_id", "position", "author_name"})
    for row_index, row in enumerate(authors_table.to_pylist()):
        paper_id = _required_id_value(row.get("paper_id"), "paper_authors", "paper_id", row_index)
        authors_by_paper_id.setdefault(paper_id, []).append(
            {"author_name": row["author_name"], "position": int(row["position"])}
        )

    papers: dict[str, dict[str, Any]] = {}
    papers_table = _read_arrow_table(papers_path)
    _require_arrow_columns(papers_table, "papers", {"paper_id", "title", "venue", "journal_name"})
    paper_rows = papers_table.to_pylist()
    _arrow_rows_by_unique_key(paper_rows, table_name="papers", key_column="paper_id")
    for row_index, row in enumerate(paper_rows):
        paper_id = _required_id_value(row.get("paper_id"), "papers", "paper_id", row_index)
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
    _require_arrow_columns(table, "specter", {"paper_id", "embedding"})
    rows = table.to_pylist()
    _arrow_rows_by_unique_key(rows, table_name="specter", key_column="paper_id")
    keys = [str(row["paper_id"]) for row in rows]
    embedding_column = table.column("embedding").combine_chunks()
    if embedding_column.null_count > 0 or embedding_column.values.null_count > 0:
        raise ValueError("specter Arrow cannot contain null embedding values")
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
    load_python_specter: bool | None = None,
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

    ``load_python_specter`` defaults to whether Python featurization will be
    used for the resolved dataset runtime. Rust-backed datasets read the
    embedding Arrow table directly; Python-backed datasets need the embeddings
    loaded into ``dataset.specter_embeddings``.
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
        if (load_python_specter is True and normalized_arrow_paths.get("specter"))
        else None
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
    if load_python_specter is None:
        rust_featurization_active = bool(
            attach_rust_featurizer and dataset_stage_uses_rust(dataset.runtime_context, dataset)
        )
        load_python_specter = not rust_featurization_active
    if load_python_specter and specter is None and normalized_arrow_paths.get("specter"):
        specter = load_specter_tuple_from_arrow(normalized_arrow_paths["specter"])
        loaded_specter = ANDData.maybe_load_specter(specter)
        needed_keys = set(dataset.papers.keys())
        dataset.specter_embeddings = {
            key: value for key, value in (loaded_specter or {}).items() if str(key) in needed_keys
        }
    logger.debug(
        "Telemetry stage: stage=arrow_training_ingest seconds=%.3f signatures=%d papers=%d specter=%s "
        "python_specter_loaded=%s",
        time.perf_counter() - ingest_start,
        len(signatures),
        len(papers),
        "yes" if normalized_arrow_paths.get("specter") else "no",
        "yes" if specter is not None else "no",
    )
    return dataset
