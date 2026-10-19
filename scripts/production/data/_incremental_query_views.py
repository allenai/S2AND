"""Create private initial-only source records in a copied Arrow dataset.

The helper preserves original rows and clones each query paper separately.
Native retrieval, pair features, and name counts then use the visible names.
Use this helper only on an unpublished physical copy without open readers.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import (
    ARROW_DATASET_KIND,
    RAW_PLANNER_ARROW_BATCH_INDEX_KEYS,
    RAW_PLANNER_ARROW_KEY_COLUMNS,
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    write_arrow_artifact_manifest,
)
from s2and.arrow_schema import validate_arrow_schema
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block_arrow import (
    read_arrow_batch_lookup_index_batch_indices_for_request,
    write_arrow_batch_lookup_index,
)
from s2and.text import canonicalize_name_parts, canonicalize_name_text
from scripts.production.model.initial_query_sources import derived_query_id as _derived_id


@dataclass(frozen=True)
class InitialQueryViews:
    """Derived IDs and visible names, keyed by original signature ID."""

    source_to_initial: dict[str, str]
    query_authors: dict[str, str]
    query_first_tokens: dict[str, str]
    source_paper_ids: dict[str, str]
    initial_paper_ids: dict[str, str]


def mask_initial_query_rows(
    signature_rows: list[dict[str, Any]], paper_author_rows: list[dict[str, Any]], query: str
) -> None:
    """Mask one local query and its own roster row without a schema change."""
    matches = [row for row in signature_rows if row["signature_id"] == query]
    if len(matches) != 1:
        raise ValueError("Initial-only source requires exactly one query row")
    signature = matches[0]
    parts = canonicalize_name_parts(signature["author_first"], signature["author_middle"], signature["author_last"])
    roster = [row for row in paper_author_rows if row["paper_id"] == signature["paper_id"]]
    position = signature["author_position"]
    if position is None and roster:
        raise ValueError("Cannot mask a nonempty query roster without author_position")
    own_rows = [row for row in roster if row["position"] == position]
    if len(own_rows) > 1:
        raise ValueError("Query roster repeats author_position")
    # Coauthor features read every other roster row, so an unmatched full name would stay visible.
    if roster and not own_rows:
        raise ValueError("Query roster has no row at author_position")
    signature["author_first"] = parts.first[:1]
    signature["author_middle"] = ""
    author = " ".join(
        value for value in (parts.first[:1], parts.last, canonicalize_name_text(signature["author_suffix"])) if value
    )
    for row in own_rows:
        row["author_name"] = author


def _selected_table(paths: Mapping[str, Path], table_name: str, values: set[str]) -> pa.Table:
    key = RAW_PLANNER_ARROW_KEY_COLUMNS[table_name]
    batches = read_arrow_batch_lookup_index_batch_indices_for_request(
        paths[table_name], paths[RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]], key_column=key, values=values
    )
    with pa.memory_map(str(paths[table_name]), "r") as source:
        reader = pa.ipc.open_file(source)
        validate_arrow_schema(reader.schema, table_name=table_name)
        selected = []
        lookup = pc.SetLookupOptions(value_set=pa.array(sorted(values), type=pa.string()))
        for index in sorted(batches):
            batch = reader.get_batch(index)
            selected.append(
                pa.Table.from_batches([batch]).filter(pc.call_function("is_in", [batch.column(key)], options=lookup))
            )
        return pa.concat_tables(selected) if selected else pa.Table.from_batches([], schema=reader.schema)


def _unique_rows(table: pa.Table, key: str, required: set[str], *, optional: bool = False) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        value = row[key]
        if value in records:
            raise ValueError(f"Source table repeats {key}: {value}")
        records[value] = row
    if not optional and set(records) != required:
        raise ValueError(f"Source table requires exact {key} coverage: missing={sorted(required - records.keys())}")
    return records


def _append_table(source_path: Path, output_path: Path, additions: pa.Table, table_name: str) -> None:
    with pa.memory_map(str(source_path), "r") as source, pa.OSFile(str(output_path), "wb") as sink:
        reader = pa.ipc.open_file(source)
        if not reader.schema.equals(additions.schema, check_metadata=True):
            raise ValueError(f"Derived {table_name} schema differs from source")
        with pa.ipc.new_file(sink, reader.schema) as writer:
            for index in range(reader.num_record_batches):
                writer.write_batch(reader.get_batch(index))
            writer.write_table(additions, max_chunksize=RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS[table_name])


def append_initial_query_views(dataset_dir: Path, source_query_ids: Sequence[str]) -> InitialQueryViews:
    """Append initial-only clones to a private dataset and rebind its manifest.

    Args:
        dataset_dir: Unpublished physical copy with a valid manifest and batch indexes.
        source_query_ids: Explicit, unique source query signature IDs.

    Returns:
        Stable clone IDs and canonical visible names, keyed by original IDs.

    Raises:
        ValueError: A source row is missing, repeated, or already projected.
    """
    if any(not isinstance(query, str) or not query for query in source_query_ids):
        raise ValueError("Require unique nonempty source query IDs")
    queries = sorted(source_query_ids)
    if len(queries) != len(set(queries)):
        raise ValueError("Require unique nonempty source query IDs")
    if not queries:
        return InitialQueryViews({}, {}, {}, {}, {})
    root = Path(dataset_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != ARROW_DATASET_KIND or manifest.get("format_version") != PUBLIC_DATA_FORMAT_VERSION:
        raise ValueError("Require a current Arrow dataset manifest")
    paths = {key: (root / value).resolve() for key, value in manifest["paths"].items()}
    for table_name in RAW_PLANNER_ARROW_KEY_COLUMNS:
        if table_name not in paths:
            continue
        for key in (table_name, RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]):
            if not paths[key].is_relative_to(root):
                raise ValueError(f"Derived source path escapes the physical dataset copy: {key}")
    signature_table = _selected_table(paths, "signatures", set(queries))
    signatures = _unique_rows(signature_table, "signature_id", set(queries))
    paper_ids = {row["paper_id"] for row in signatures.values()}
    if any(not isinstance(paper_id, str) or not paper_id for paper_id in paper_ids):
        raise ValueError("Source queries require nonempty paper IDs")
    papers_table = _selected_table(paths, "papers", paper_ids)
    papers = _unique_rows(papers_table, "paper_id", paper_ids)
    roster_table = _selected_table(paths, "paper_authors", paper_ids)
    rosters: dict[str, list[dict[str, Any]]] = {paper_id: [] for paper_id in paper_ids}
    for row in roster_table.to_pylist():
        rosters[row["paper_id"]].append(row)
    specter_table = _selected_table(paths, "specter", paper_ids) if "specter" in paths else None
    embeddings = {} if specter_table is None else _unique_rows(specter_table, "paper_id", paper_ids, optional=True)
    clone_signatures = []
    clone_papers = []
    clone_rosters = []
    clone_embeddings = []
    source_to_initial = {}
    query_authors = {}
    first_tokens = {}
    source_paper_ids = {}
    initial_paper_ids = {}
    for query in queries:
        signature = dict(signatures[query])
        paper_id = signature["paper_id"]
        roster = [dict(row) for row in rosters[paper_id]]
        mask_initial_query_rows([signature], roster, query)
        parts = canonicalize_name_parts(signature["author_first"], signature["author_middle"], signature["author_last"])
        query_authors[query] = " ".join(
            value
            for value in (parts.first, parts.middle, parts.last, canonicalize_name_text(signature["author_suffix"]))
            if value
        )
        first_tokens[query] = parts.first
        clone_id = _derived_id("visibleq:", query, paper_id)
        clone_paper = _derived_id("visiblep:", query, paper_id)
        source_to_initial[query] = clone_id
        source_paper_ids[query] = paper_id
        initial_paper_ids[query] = clone_paper
        signature.update(signature_id=clone_id, paper_id=clone_paper)
        clone_signatures.append(signature)
        clone_papers.append({**papers[paper_id], "paper_id": clone_paper})
        clone_rosters.extend({**row, "paper_id": clone_paper} for row in roster)
        if paper_id in embeddings:
            clone_embeddings.append({**embeddings[paper_id], "paper_id": clone_paper})
    if _selected_table(paths, "signatures", set(source_to_initial.values())).num_rows:
        raise ValueError("Derived query IDs already occur in the dataset")
    if _selected_table(paths, "papers", {row["paper_id"] for row in clone_papers}).num_rows:
        raise ValueError("Derived paper IDs already occur in the dataset")
    additions = {
        "signatures": pa.Table.from_pylist(clone_signatures, schema=signature_table.schema),
        "papers": pa.Table.from_pylist(clone_papers, schema=papers_table.schema),
        "paper_authors": pa.Table.from_pylist(clone_rosters, schema=roster_table.schema),
    }
    if specter_table is not None and clone_embeddings:
        additions["specter"] = pa.Table.from_pylist(clone_embeddings, schema=specter_table.schema)
    with tempfile.TemporaryDirectory(prefix=".initial-query-views-", dir=root) as temporary:
        staged_paths: dict[str, Path] = {}
        for table_name, table in additions.items():
            staged = Path(temporary) / f"{table_name}.arrow"
            _append_table(paths[table_name], staged, table, table_name)
            staged_paths[table_name] = staged
            index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]
            index_path, _metrics = write_arrow_batch_lookup_index(
                staged,
                Path(temporary) / f"{index_key}.bin",
                key_column=RAW_PLANNER_ARROW_KEY_COLUMNS[table_name],
                table_name=table_name,
            )
            staged_paths[index_key] = Path(index_path)
        for key, staged in staged_paths.items():
            manifest["files"][key] = {"byte_count": staged.stat().st_size, "sha256": sha256_file(staged)}
        for key, staged in staged_paths.items():
            staged.replace(paths[key])
        write_arrow_artifact_manifest(manifest, root)
    return InitialQueryViews(source_to_initial, query_authors, first_tokens, source_paper_ids, initial_paper_ids)
