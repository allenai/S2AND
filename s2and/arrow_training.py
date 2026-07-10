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

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import (
    require_normalization_version,
    validate_arrow_prediction_artifacts,
    verified_arrow_artifact_generation,
)
from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    ANDData,
    _validated_name_counts_provenance,
)
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
        missing = sorted(required_columns.difference(reader.schema.names))
        if missing:
            raise ValueError(f"Arrow {table_name} table is missing required columns: {missing}")
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
    for row_index, row in _iter_arrow_rows(
        path,
        table_name="signatures",
        required_columns=required_columns,
    ):
        signature_id = _required_id_value(row.get("signature_id"), "signatures", "signature_id", row_index)
        if signature_id in signatures:
            raise ValueError(f"signatures Arrow contains duplicate signature_id={signature_id!r}")
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
    paper_author_keys: set[tuple[str, int]] = set()
    for row_index, row in _iter_arrow_rows(
        paper_authors_path,
        table_name="paper_authors",
        required_columns={"paper_id", "position", "author_name"},
    ):
        paper_id = _required_id_value(row.get("paper_id"), "paper_authors", "paper_id", row_index)
        raw_author_name = row.get("author_name")
        if not isinstance(raw_author_name, str) or not raw_author_name.strip():
            raise ValueError(f"Arrow paper_authors table contains empty author_name at row {row_index}")
        raw_position = row.get("position")
        if raw_position is None:
            raise ValueError(f"Arrow paper_authors table contains null position at row {row_index}")
        position = int(raw_position)
        author_key = (paper_id, position)
        if author_key in paper_author_keys:
            raise ValueError(
                "paper_authors Arrow contains duplicate " f"(paper_id, position)=({paper_id!r}, {position})"
            )
        paper_author_keys.add(author_key)
        authors_by_paper_id.setdefault(paper_id, []).append({"author_name": raw_author_name, "position": position})

    papers: dict[str, dict[str, Any]] = {}
    for row_index, row in _iter_arrow_rows(
        papers_path,
        table_name="papers",
        required_columns={"paper_id", "title", "venue", "journal_name"},
    ):
        paper_id = _required_id_value(row.get("paper_id"), "papers", "paper_id", row_index)
        if paper_id in papers:
            raise ValueError(f"papers Arrow contains duplicate paper_id={paper_id!r}")
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
    unknown_author_paper_ids = sorted(set(authors_by_paper_id).difference(papers))
    if unknown_author_paper_ids:
        raise ValueError(
            "paper_authors Arrow references paper_id values absent from papers Arrow: "
            f"{unknown_author_paper_ids[:10]}"
        )
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
    *,
    expected_normalization_version: str,
) -> dict[str, str]:
    """Validate and attach Arrow paths so featurization uses ``from_arrow_paths``.

    Requires the raw-planner batch indexes and the ``name_counts_index/``
    sidecar because training featurization loads name counts in Rust.
    """

    expected_version = require_normalization_version(
        expected_normalization_version,
        context="arrow-native training attachment",
    )
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
        strict_batch_index_validation=True,
        expected_normalization_version=expected_version,
        require_canonical_manifest=True,
        context="arrow-native training featurization",
        producer_hint=(
            "write the training bundle with signatures/papers/paper_authors tables, raw-planner "
            "batch indexes (write_raw_arrow_batch_lookup_indexes), specter, and the "
            "name_counts_index sidecar (write_name_counts_index)"
        ),
    )
    normalized.pop("query_signatures", None)
    index_manifest = json.loads((Path(normalized["name_counts_index"]) / "manifest.json").read_text(encoding="utf-8"))
    dataset.name_counts_provenance = _validated_name_counts_provenance(
        index_manifest.get("source_provenance"),
        context="arrow-native training name_counts_index",
    )
    # Rust consumes aliases while building. Freezing provides mutation safety
    # and an O(1) repeat fingerprint for the native featurizer cache.
    if isinstance(getattr(dataset, "name_tuples", None), set):
        dataset.name_tuples = frozenset(dataset.name_tuples)
    from s2and import feature_port

    feature_port.evict_rust_featurizer(dataset)
    setattr(dataset, RUST_FEATURIZER_ARROW_PATHS_ATTR, normalized)
    setattr(dataset, feature_port.RUST_FEATURIZER_NORMALIZATION_VERSION_ATTR, expected_version)
    verified_generation = verified_arrow_artifact_generation(normalized)
    if verified_generation is None:
        # Legacy/ad-hoc bundles have no immutable full-digest inventory. Their
        # cache key must fingerprint the material files on every lookup.
        if hasattr(dataset, feature_port.RUST_FEATURIZER_ARTIFACT_GENERATION_ATTR):
            delattr(dataset, feature_port.RUST_FEATURIZER_ARTIFACT_GENERATION_ATTR)
    else:
        setattr(
            dataset,
            feature_port.RUST_FEATURIZER_ARTIFACT_GENERATION_ATTR,
            verified_generation,
        )
    # Rust production prediction (Clusterer.predict / cluster_eval) resolves
    # explicit dataset.arrow_paths; an Arrow-ingested training dataset carries
    # the same artifacts, so expose them for prediction too.
    dataset.arrow_paths = dict(normalized)
    return normalized


def build_training_anddata_from_arrow(
    arrow_paths: Mapping[str, Any],
    name: str,
    *,
    expected_normalization_version: str,
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

    expected_version = require_normalization_version(
        expected_normalization_version,
        context="arrow-native training ingestion",
    )
    ingest_start = time.perf_counter()
    path_keys = {str(key) for key in arrow_paths}
    normalized_arrow_paths = validate_arrow_prediction_artifacts(
        arrow_paths,
        require_specter="specter" in path_keys or "specter2" in path_keys,
        require_name_counts_index=False,
        require_batch_indexes=False,
        expected_normalization_version=expected_version,
        require_canonical_manifest=True,
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
        attach_training_arrow_featurizer_paths(
            dataset,
            normalized_arrow_paths,
            expected_normalization_version=expected_version,
        )
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
