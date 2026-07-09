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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import validate_arrow_prediction_artifacts
from s2and.data import NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR, ANDData
from s2and.runtime import RUST_FEATURIZER_ARROW_PATHS_ATTR, dataset_stage_uses_rust

logger = logging.getLogger("s2and")

_REQUIRED_TABLE_KEYS = ("signatures", "papers", "paper_authors")
_ARROW_INGEST_BATCH_ROWS = 512


def _read_arrow_table(path: str | Path) -> Any:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


@contextmanager
def _arrow_file_reader(path: str | Path) -> Iterator[Any]:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        yield pa.ipc.open_file(source)


def _require_column_names(column_names: set[str], table_name: str, columns: set[str]) -> None:
    missing = sorted(columns - column_names)
    if missing:
        raise ValueError(f"Arrow {table_name} table is missing required columns: {missing}")


def _require_columns(table: Any, table_name: str, columns: set[str]) -> None:
    _require_column_names(set(table.column_names), table_name, columns)


def _batch_column_values(batch: Any, column_name: str) -> list[Any]:
    column_index = batch.schema.get_field_index(column_name)
    if column_index < 0:
        raise KeyError(column_name)
    return batch.column(column_index).to_pylist()


def _optional_batch_column_values(batch: Any, column_name: str) -> list[Any]:
    column_index = batch.schema.get_field_index(column_name)
    if column_index < 0:
        return [None] * batch.num_rows
    return batch.column(column_index).to_pylist()


def _iter_record_batch_slices(reader: Any) -> Iterator[Any]:
    for batch_index in range(reader.num_record_batches):
        batch = reader.get_batch(batch_index)
        for offset in range(0, batch.num_rows, _ARROW_INGEST_BATCH_ROWS):
            yield batch.slice(offset, _ARROW_INGEST_BATCH_ROWS)


def _required_id_value(raw_value: Any, table_name: str, column_name: str, row_index: int) -> str:
    if raw_value is None:
        raise ValueError(f"Arrow {table_name} table contains null {column_name} at row {row_index}")
    value = str(raw_value)
    if value == "":
        raise ValueError(f"Arrow {table_name} table contains empty {column_name} at row {row_index}")
    return value


def _validate_unique_required_id_column(table: Any, table_name: str, column_name: str) -> None:
    seen: set[str] = set()
    for row_index, raw_value in enumerate(table.column(column_name).to_pylist()):
        value = _required_id_value(raw_value, table_name, column_name, row_index)
        if value in seen:
            raise ValueError(f"Arrow {table_name} table contains duplicate {column_name}: {value!r}")
        seen.add(value)


def _validate_required_id_column(table: Any, table_name: str, column_name: str) -> None:
    for row_index, raw_value in enumerate(table.column(column_name).to_pylist()):
        _required_id_value(raw_value, table_name, column_name, row_index)


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
    seen_signature_ids: set[str] = set()
    row_offset = 0
    with _arrow_file_reader(path) as reader:
        _require_column_names(set(reader.schema.names), "signatures", required_columns)
        for batch in _iter_record_batch_slices(reader):
            signature_id_values = _batch_column_values(batch, "signature_id")
            paper_id_values = _batch_column_values(batch, "paper_id")
            first_values = _batch_column_values(batch, "author_first")
            middle_values = _batch_column_values(batch, "author_middle")
            last_values = _batch_column_values(batch, "author_last")
            suffix_values = _batch_column_values(batch, "author_suffix")
            affiliation_values = _batch_column_values(batch, "author_affiliations")
            position_values = _batch_column_values(batch, "author_position")
            email_values = _optional_batch_column_values(batch, "author_email")
            block_values = _optional_batch_column_values(batch, "author_block")
            orcid_values = _optional_batch_column_values(batch, "author_orcid")
            source_author_id_values = _optional_batch_column_values(batch, "source_author_ids")
            for local_index in range(batch.num_rows):
                row_index = row_offset + local_index
                signature_id = _required_id_value(
                    signature_id_values[local_index],
                    "signatures",
                    "signature_id",
                    row_index,
                )
                if signature_id in seen_signature_ids:
                    raise ValueError(f"Arrow signatures table contains duplicate signature_id: {signature_id!r}")
                seen_signature_ids.add(signature_id)
                paper_id = _required_id_value(
                    paper_id_values[local_index],
                    "signatures",
                    "paper_id",
                    row_index,
                )
                orcid = orcid_values[local_index]
                signatures[signature_id] = {
                    "signature_id": signature_id,
                    "paper_id": paper_id,
                    "author_info": {
                        "first": first_values[local_index],
                        "middle": middle_values[local_index],
                        "last": last_values[local_index],
                        "suffix": suffix_values[local_index],
                        "affiliations": list(affiliation_values[local_index] or []),
                        "email": email_values[local_index],
                        "position": position_values[local_index],
                        "block": block_values[local_index],
                        # ANDData derives author_info_orcid from source_ids +
                        # source_id_source; the Arrow column stores the derived value,
                        # so reconstruct the source shape it expects.
                        "source_ids": [orcid] if orcid else None,
                        "source_id_source": "ORCID" if orcid else None,
                    },
                    "sourced_author_ids": list(source_author_id_values[local_index] or []),
                }
            row_offset += batch.num_rows
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
    row_offset = 0
    with _arrow_file_reader(paper_authors_path) as reader:
        _require_column_names(set(reader.schema.names), "paper_authors", {"paper_id", "position", "author_name"})
        for batch in _iter_record_batch_slices(reader):
            paper_id_values = _batch_column_values(batch, "paper_id")
            position_values = _batch_column_values(batch, "position")
            author_name_values = _batch_column_values(batch, "author_name")
            for local_index in range(batch.num_rows):
                row_index = row_offset + local_index
                paper_id = _required_id_value(
                    paper_id_values[local_index],
                    "paper_authors",
                    "paper_id",
                    row_index,
                )
                authors_by_paper_id.setdefault(paper_id, []).append(
                    {"author_name": author_name_values[local_index], "position": int(position_values[local_index])}
                )
            row_offset += batch.num_rows

    papers: dict[str, dict[str, Any]] = {}
    seen_paper_ids: set[str] = set()
    row_offset = 0
    with _arrow_file_reader(papers_path) as reader:
        _require_column_names(set(reader.schema.names), "papers", {"paper_id", "title", "venue", "journal_name"})
        for batch in _iter_record_batch_slices(reader):
            paper_id_values = _batch_column_values(batch, "paper_id")
            title_values = _batch_column_values(batch, "title")
            venue_values = _batch_column_values(batch, "venue")
            journal_name_values = _batch_column_values(batch, "journal_name")
            abstract_values = _optional_batch_column_values(batch, "abstract")
            year_values = _optional_batch_column_values(batch, "year")
            for local_index in range(batch.num_rows):
                row_index = row_offset + local_index
                paper_id = _required_id_value(
                    paper_id_values[local_index],
                    "papers",
                    "paper_id",
                    row_index,
                )
                if paper_id in seen_paper_ids:
                    raise ValueError(f"Arrow papers table contains duplicate paper_id: {paper_id!r}")
                seen_paper_ids.add(paper_id)
                papers[paper_id] = {
                    "paper_id": paper_id,
                    "title": title_values[local_index],
                    "abstract": abstract_values[local_index] or "",
                    "venue": venue_values[local_index],
                    "journal_name": journal_name_values[local_index],
                    "year": year_values[local_index],
                    "authors": authors_by_paper_id.get(paper_id, []),
                }
            row_offset += batch.num_rows
    if not papers:
        raise ValueError(f"Arrow papers table has no rows: {papers_path}")
    return papers


def load_specter_tuple_from_arrow(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read ``specter.arrow`` into the ``(matrix, keys)`` tuple ANDData accepts."""

    table = _read_arrow_table(path)
    _require_columns(table, "specter", {"paper_id", "embedding"})
    _validate_unique_required_id_column(table, "specter", "paper_id")
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
