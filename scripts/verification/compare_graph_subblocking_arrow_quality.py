"""Compare Python graph subblocking with Rust Arrow graph subblocking.

This is the durable verification entry point for the Rust-native Arrow graph
fallback. It stops at subblock construction, writes a JSON summary plus
inspection artifacts, and requires either ``--limit`` or ``--allow-full`` so
large real-data runs are explicit.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset, read_arrow_collection_root  # noqa: E402
from s2and.subblocking import (  # noqa: E402
    GraphSubblockingConfig,
    _make_subblocks_with_telemetry_arrow_rust,
    _read_arrow_rows_by_values,
    make_dataset_graph_subblocking_cluster_fn,
    make_subblocks_with_telemetry,
)
from s2and.text import compute_block, normalize_text  # noqa: E402
from scripts.production.model.release_pairwise import _load_evaluation_plan  # noqa: E402
from scripts.production.model.run_binding import load_run_binding, require_run_binding_matches  # noqa: E402
from scripts.verification.validate_local_arrow_release import validate_release_root  # noqa: E402

_DEFAULT_GRAPH_CONFIG = GraphSubblockingConfig()


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-root", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--run-binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--specter-pickle", type=Path, default=None)
    parser.add_argument("--python-source", choices=("raw", "arrow"), default="raw")
    parser.add_argument("--comparison-mode", choices=("python-vs-rust", "rust-only"), default="python-vs-rust")
    parser.add_argument("--component-members-parquet", type=Path, default=None)
    parser.add_argument("--maximum-size", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-full", action="store_true")
    parser.add_argument("--sample-mode", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--neighbor-mode", choices=("projection", "exact"), default="projection")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--min-edge-score", type=float, default=0.30)
    parser.add_argument("--specter-weight", type=float, default=1.0)
    parser.add_argument("--coauthor-weight", type=float, default=0.35)
    parser.add_argument("--affiliation-weight", type=float, default=0.20)
    parser.add_argument("--max-exact-knn-group-size", type=int, default=25_000)
    parser.add_argument("--projection-count", type=int, default=12)
    parser.add_argument("--projection-window", type=int, default=12)
    parser.add_argument("--max-candidate-edges", type=int, default=5_000_000)
    parser.add_argument("--pack-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--component-pack-strategy",
        choices=("edge-greedy", "aggregate-greedy", "size"),
        default="edge-greedy",
    )
    parser.add_argument(
        "--sparse-evidence-edges",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_edges,
    )
    parser.add_argument(
        "--sparse-evidence-max-posting-size",
        type=int,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_max_posting_size,
    )
    parser.add_argument(
        "--sparse-evidence-neighbors",
        type=int,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_neighbors,
    )
    parser.add_argument(
        "--sparse-evidence-min-weight",
        type=float,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_min_weight,
    )
    parser.add_argument(
        "--sparse-evidence-include-coauthors",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_include_coauthors,
    )
    parser.add_argument(
        "--sparse-evidence-include-affiliations",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_GRAPH_CONFIG.sparse_evidence_include_affiliations,
    )
    parser.add_argument("--component-pack-top-k", type=int, default=8)
    parser.add_argument("--local-move-passes", type=int, default=0)
    parser.add_argument("--adaptive-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-projection-max-group-size", type=int, default=5_000)
    parser.add_argument("--adaptive-projection-count", type=int, default=24)
    parser.add_argument("--adaptive-projection-window", type=int, default=24)
    parser.add_argument("--orcid-subblocking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-diff-subblocks", type=int, default=30)
    args = parser.parse_args()
    if args.limit is None and not args.allow_full:
        parser.error("provide --limit for a bounded run, or --allow-full to run the full block explicitly")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if (
        args.comparison_mode == "python-vs-rust"
        and args.python_source == "raw"
        and (args.raw_root is None or args.specter_pickle is None)
    ):
        parser.error("--python-source raw requires --raw-root and --specter-pickle")
    return args


def _required_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path


def _log_progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}", file=sys.stderr, flush=True)


def _load_json_map(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object-shaped JSON in {path}")
    return {str(key): value for key, value in payload.items()}


def _select_signature_ids(
    all_signature_ids: Iterable[str],
    *,
    limit: int | None,
    sample_mode: str,
    seed: int,
) -> list[str]:
    signature_ids = sorted(str(signature_id) for signature_id in all_signature_ids)
    if limit is None or len(signature_ids) <= limit:
        return signature_ids
    if sample_mode == "first":
        return signature_ids[:limit]
    rng = random.Random(seed)
    return sorted(rng.sample(signature_ids, limit))


def _paper_author_blocks(paper: Mapping[str, Any] | None, signature_position: int | None) -> tuple[str, ...]:
    if paper is None:
        return ()
    out: list[str] = []
    for author in paper.get("authors") or []:
        try:
            position = int(author.get("position", -1))
        except (TypeError, ValueError):
            position = -1
        if signature_position is not None and position == signature_position:
            continue
        block = compute_block(normalize_text(str(author.get("author_name") or "")))
        if block:
            out.append(block)
    return tuple(sorted(set(out)))


def _signature_from_raw(signature_id: str, row: Mapping[str, Any], papers: Mapping[str, Any]) -> SimpleNamespace:
    author_info = row.get("author_info") or {}
    paper_id = str(row.get("paper_id"))
    position_raw = author_info.get("position")
    try:
        position = int(position_raw) if position_raw is not None else None
    except (TypeError, ValueError):
        position = None
    affiliations = tuple(str(value) for value in (author_info.get("affiliations") or ()))
    paper = papers.get(paper_id)
    return SimpleNamespace(
        signature_id=str(row.get("signature_id", signature_id)),
        paper_id=paper_id,
        author_info_first=author_info.get("first"),
        author_info_middle=author_info.get("middle"),
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle_normalized_without_apostrophe=None,
        author_info_affiliations=affiliations,
        author_info_affiliations_n_grams=None,
        author_info_coauthor_blocks=_paper_author_blocks(paper, position),
        author_info_coauthors=None,
        author_info_orcid=author_info.get("orcid"),
        author_info_position=position,
    )


def _load_specter_subset(path: Path, needed_paper_ids: set[str]) -> dict[str, np.ndarray]:
    with path.open("rb") as infile:
        payload = pickle.load(infile)
    if isinstance(payload, tuple) and len(payload) == 2:
        matrix, paper_ids = payload
        matrix = np.asarray(matrix, dtype=np.float32)
        if len(paper_ids) != matrix.shape[0]:
            raise ValueError(f"SPECTER tuple ids={len(paper_ids)} rows={matrix.shape[0]}")
        return {
            str(paper_id): np.asarray(matrix[index], dtype=np.float32)
            for index, paper_id in enumerate(paper_ids)
            if str(paper_id) in needed_paper_ids
        }
    if not isinstance(payload, Mapping):
        raise ValueError(f"Unsupported SPECTER pickle payload type: {type(payload).__name__}")
    out: dict[str, np.ndarray] = {}
    for paper_id in needed_paper_ids:
        value = payload.get(paper_id)
        if value is None:
            try:
                value = payload.get(int(paper_id))
            except ValueError:
                value = None
        if value is not None:
            out[paper_id] = np.asarray(value, dtype=np.float32)
    return out


def load_lightweight_dataset(
    raw_root: Path,
    specter_pickle: Path,
    *,
    limit: int | None,
    sample_mode: str,
    seed: int,
) -> tuple[SimpleNamespace, list[str]]:
    signatures_raw = _load_json_map(_required_path(raw_root / "signatures.json"))
    signature_ids = _select_signature_ids(
        signatures_raw.keys(),
        limit=limit,
        sample_mode=sample_mode,
        seed=seed,
    )
    paper_ids = {str(signatures_raw[signature_id].get("paper_id")) for signature_id in signature_ids}
    papers_raw = _load_json_map(_required_path(raw_root / "papers.json"))
    papers = {paper_id: papers_raw[paper_id] for paper_id in paper_ids if paper_id in papers_raw}
    signatures = {
        signature_id: _signature_from_raw(signature_id, signatures_raw[signature_id], papers)
        for signature_id in signature_ids
    }
    dataset = SimpleNamespace(
        signatures=signatures,
        papers=papers,
        specter_embeddings=_load_specter_subset(_required_path(specter_pickle), paper_ids),
        random_seed=int(seed),
    )
    return dataset, signature_ids


def _read_arrow_column_values(source_file: BinaryIO, column_name: str) -> list[Any]:
    pa = __import__("pyarrow")
    with pa.PythonFile(source_file, mode="r") as source:
        reader = pa.ipc.open_file(source)
        column_index = reader.schema.get_field_index(column_name)
        if column_index < 0:
            raise ValueError(f"Arrow file is missing required column: {column_name!r}")
        values: list[Any] = []
        for batch_index in range(reader.num_record_batches):
            values.extend(reader.get_batch(batch_index).column(column_index).to_pylist())
    return values


def _read_arrow_rows(
    source_file: BinaryIO,
    required_columns: set[str],
    *,
    optional_columns: Sequence[str] = (),
) -> list[dict[str, Any]]:
    pa = __import__("pyarrow")
    with pa.PythonFile(source_file, mode="r") as source:
        reader = pa.ipc.open_file(source)
        selected_columns = required_columns.union(set(optional_columns).intersection(reader.schema.names))
        column_indices = {
            column_name: reader.schema.get_field_index(column_name) for column_name in sorted(selected_columns)
        }
        missing_columns = [column_name for column_name, column_index in column_indices.items() if column_index < 0]
        if missing_columns:
            raise ValueError(f"Arrow file is missing required columns: {missing_columns!r}")
        rows: list[dict[str, Any]] = []
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            columns = {
                column_name: batch.column(column_index).to_pylist()
                for column_name, column_index in column_indices.items()
            }
            for row_index in range(batch.num_rows):
                rows.append({column_name: values[row_index] for column_name, values in columns.items()})
    return rows


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise ValueError("Arrow list field unexpectedly decoded as a scalar string")
    return tuple(str(item) for item in value if item is not None)


def _paper_author_rows_by_paper(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_positions: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError("paper_authors Arrow cannot contain empty paper_id values")
        position_raw = row.get("position")
        if position_raw is None:
            raise ValueError("paper_authors Arrow cannot contain null position values")
        position = int(position_raw)
        if position in seen_positions[paper_id]:
            raise ValueError(f"paper_authors Arrow contains duplicate (paper_id, position): ({paper_id!r}, {position})")
        seen_positions[paper_id].add(position)
        author_name_value = row.get("author_name")
        if author_name_value is None:
            raise ValueError("paper_authors Arrow cannot contain null author_name values")
        author_name = str(author_name_value)
        out[paper_id].append({"position": position, "author_name": author_name})
    for authors in out.values():
        authors.sort(key=lambda author: int(author["position"]))
    return dict(out)


def _signatures_from_arrow_rows(
    signature_ids: Sequence[str],
    signature_rows_by_id: Mapping[str, Mapping[str, Any]],
    papers: Mapping[str, Any],
) -> dict[str, SimpleNamespace]:
    signatures: dict[str, SimpleNamespace] = {}
    for signature_id in signature_ids:
        row = signature_rows_by_id[signature_id]
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            raise ValueError("signatures Arrow cannot contain empty paper_id values")
        position_raw = row.get("author_position")
        position = None if position_raw is None else int(position_raw)
        affiliations = _string_tuple(row.get("author_affiliations"))
        paper = papers.get(paper_id)
        signatures[signature_id] = SimpleNamespace(
            signature_id=signature_id,
            paper_id=paper_id,
            author_info_first=None if row.get("author_first") is None else str(row.get("author_first")),
            author_info_middle=None if row.get("author_middle") is None else str(row.get("author_middle")),
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle_normalized_without_apostrophe=None,
            author_info_affiliations=affiliations,
            author_info_affiliations_n_grams=None,
            author_info_coauthor_blocks=_paper_author_blocks(paper, position),
            author_info_coauthors=None,
            author_info_orcid=None if row.get("author_orcid") is None else str(row.get("author_orcid")),
            author_info_position=position,
        )
    return signatures


def load_lightweight_dataset_from_arrow(
    arrow_lease: Any,
    *,
    limit: int | None,
    sample_mode: str,
    seed: int,
    include_specter: bool = True,
) -> tuple[SimpleNamespace, list[str]]:
    """Load bounded Python comparison state from one retained generation."""

    lease = arrow_lease
    signature_columns = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_affiliations",
        "author_position",
    }
    if limit is None:
        with lease.open_file("signatures") as source_file:
            signature_rows = _read_arrow_rows(
                source_file,
                signature_columns,
                optional_columns=("author_orcid",),
            )
        signature_ids = _select_signature_ids(
            (str(row["signature_id"]) for row in signature_rows if row.get("signature_id") is not None),
            limit=None,
            sample_mode=sample_mode,
            seed=seed,
        )
    else:
        with lease.open_file("signatures") as source_file:
            all_signature_ids = [
                str(signature_id)
                for signature_id in _read_arrow_column_values(source_file, "signature_id")
                if signature_id is not None
            ]
        signature_ids = _select_signature_ids(
            all_signature_ids,
            limit=limit,
            sample_mode=sample_mode,
            seed=seed,
        )
        signature_rows = _read_arrow_rows_by_values(
            lease,
            "signatures",
            "signature_id",
            signature_ids,
            required_columns=signature_columns,
            optional_columns=("author_orcid",),
            table_name="signatures",
        )
    signature_rows_by_id = {str(row["signature_id"]): row for row in signature_rows}
    missing_signature_ids = [signature_id for signature_id in signature_ids if signature_id not in signature_rows_by_id]
    if missing_signature_ids:
        raise ValueError(f"signatures Arrow is missing selected signature ids: {missing_signature_ids[:10]}")
    paper_ids = tuple(dict.fromkeys(str(row["paper_id"]) for row in signature_rows_by_id.values()))
    paper_author_columns = {"paper_id", "position", "author_name"}
    if limit is None:
        paper_id_set = set(paper_ids)
        with lease.open_file("paper_authors") as source_file:
            paper_author_rows = [
                row
                for row in _read_arrow_rows(source_file, paper_author_columns)
                if str(row.get("paper_id") or "") in paper_id_set
            ]
    else:
        paper_author_rows = _read_arrow_rows_by_values(
            lease,
            "paper_authors",
            "paper_id",
            paper_ids,
            required_columns=paper_author_columns,
            table_name="paper_authors",
        )
    paper_authors_by_paper = _paper_author_rows_by_paper(paper_author_rows)
    papers = {paper_id: {"authors": paper_authors_by_paper.get(paper_id, [])} for paper_id in paper_ids}
    specter_embeddings = {}
    if include_specter:
        specter_columns = {"paper_id", "embedding"}
        if limit is None:
            paper_id_set = set(paper_ids)
            with lease.open_file("specter") as source_file:
                specter_rows = [
                    row
                    for row in _read_arrow_rows(source_file, specter_columns)
                    if str(row.get("paper_id") or "") in paper_id_set
                ]
        else:
            specter_rows = _read_arrow_rows_by_values(
                lease,
                "specter",
                "paper_id",
                paper_ids,
                required_columns=specter_columns,
                table_name="specter",
            )
        specter_embeddings = {
            str(row["paper_id"]): np.asarray(row["embedding"], dtype=np.float32)
            for row in specter_rows
            if row.get("paper_id") is not None and row.get("embedding") is not None
        }
    return (
        SimpleNamespace(
            signatures=_signatures_from_arrow_rows(signature_ids, signature_rows_by_id, papers),
            papers=papers,
            specter_embeddings=specter_embeddings,
            random_seed=int(seed),
        ),
        signature_ids,
    )


def _graph_config(args: argparse.Namespace) -> GraphSubblockingConfig:
    return GraphSubblockingConfig(
        neighbor_mode=str(args.neighbor_mode),
        neighbors=int(args.neighbors),
        min_edge_score=float(args.min_edge_score),
        specter_weight=float(args.specter_weight),
        coauthor_weight=float(args.coauthor_weight),
        affiliation_weight=float(args.affiliation_weight),
        max_exact_knn_group_size=int(args.max_exact_knn_group_size),
        projection_count=int(args.projection_count),
        projection_window=int(args.projection_window),
        max_candidate_edges=int(args.max_candidate_edges),
        pack_components=bool(args.pack_components),
        component_pack_strategy=str(args.component_pack_strategy),
        sparse_evidence_edges=bool(args.sparse_evidence_edges),
        sparse_evidence_max_posting_size=int(args.sparse_evidence_max_posting_size),
        sparse_evidence_neighbors=int(args.sparse_evidence_neighbors),
        sparse_evidence_min_weight=float(args.sparse_evidence_min_weight),
        sparse_evidence_include_coauthors=bool(args.sparse_evidence_include_coauthors),
        sparse_evidence_include_affiliations=bool(args.sparse_evidence_include_affiliations),
        component_pack_top_k=int(args.component_pack_top_k),
        local_move_passes=int(args.local_move_passes),
        adaptive_projection=bool(args.adaptive_projection),
        adaptive_projection_max_group_size=int(args.adaptive_projection_max_group_size),
        adaptive_projection_count=int(args.adaptive_projection_count),
        adaptive_projection_window=int(args.adaptive_projection_window),
    )


def load_signature_ids_from_arrow(
    arrow_dataset: ArrowDataset,
    *,
    limit: int | None,
    sample_mode: str,
    seed: int,
) -> list[str]:
    with arrow_dataset.use() as lease, lease.open_file("signatures") as source_file:
        all_signature_ids = [
            str(signature_id)
            for signature_id in _read_arrow_column_values(source_file, "signature_id")
            if signature_id is not None
        ]
    return _select_signature_ids(
        all_signature_ids,
        limit=limit,
        sample_mode=sample_mode,
        seed=seed,
    )


def _signature_to_subblock(subblocks: Mapping[str, Iterable[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for subblock_key, signature_ids in subblocks.items():
        for signature_id in signature_ids:
            key = str(signature_id)
            if key in out:
                raise ValueError(f"Signature {key} appears in more than one subblock")
            out[key] = str(subblock_key)
    return out


def _partition_metrics(subblocks: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    sizes = np.array([len(list(signature_ids)) for signature_ids in subblocks.values()], dtype=np.int64)
    if sizes.size == 0:
        raise ValueError("Expected at least one subblock")
    return {
        "subblock_count": int(sizes.size),
        "max_subblock_size": int(sizes.max()),
        "median_subblock_size": float(np.median(sizes)),
        "p95_subblock_size": float(np.percentile(sizes, 95)),
        "within_subblock_pair_count": int(sum(int(size) * (int(size) - 1) // 2 for size in sizes)),
    }


def _validated_partition(
    subblocks: Mapping[str, list[str]],
    signature_ids: Sequence[str],
    maximum_size: int,
    component_labels: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed_ids = set(_signature_to_subblock(subblocks))
    expected_ids = {str(signature_id) for signature_id in signature_ids}
    if observed_ids != expected_ids:
        raise ValueError("Subblocking output lost or added signatures")
    partition = _partition_metrics(subblocks)
    if partition["max_subblock_size"] > maximum_size:
        raise ValueError("Subblocking output exceeds maximum_size")
    components = _component_preservation_metrics(subblocks, component_labels)
    if components and components["component_preserved_count"] != components["repeated_component_count"]:
        raise ValueError("Subblocking output split a required component")
    return partition, components


def _load_component_labels(path: Path | None, selected_signature_ids: set[str]) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_parquet(_required_path(path), columns=["signature_id", "candidate_component_key"])
    labels: dict[str, str] = {}
    for signature_id, component_key in zip(table["signature_id"], table["candidate_component_key"], strict=True):
        signature_key = str(signature_id)
        if signature_key in selected_signature_ids:
            labels[signature_key] = str(component_key)
    return labels


def _component_preservation_metrics(
    subblocks: Mapping[str, list[str]],
    component_labels: Mapping[str, str],
) -> dict[str, Any]:
    if not component_labels:
        return {}
    signature_to_subblock = _signature_to_subblock(subblocks)
    component_to_signatures: dict[str, list[str]] = defaultdict(list)
    missing_label_count = 0
    for signature_id in signature_to_subblock:
        component_key = component_labels.get(signature_id)
        if component_key is None:
            missing_label_count += 1
            continue
        component_to_signatures[component_key].append(signature_id)
    repeated_components = [values for values in component_to_signatures.values() if len(values) > 1]
    total_pair_count = 0
    preserved_pair_count = 0
    preserved_component_count = 0
    fragment_counts = []
    for signature_ids in repeated_components:
        subblock_counts = Counter(signature_to_subblock[signature_id] for signature_id in signature_ids)
        fragment_counts.append(len(subblock_counts))
        preserved_component_count += int(len(subblock_counts) == 1)
        total_pair_count += len(signature_ids) * (len(signature_ids) - 1) // 2
        preserved_pair_count += sum(count * (count - 1) // 2 for count in subblock_counts.values())
    return {
        "component_label_count": int(len(component_labels)),
        "component_missing_label_count": int(missing_label_count),
        "repeated_component_count": int(len(repeated_components)),
        "component_preserved_count": int(preserved_component_count),
        "component_pair_count": int(total_pair_count),
        "component_pair_preserved_count": int(preserved_pair_count),
        "component_pair_recall": float(preserved_pair_count / total_pair_count) if total_pair_count else 0.0,
        "component_mean_fragment_count": float(np.mean(fragment_counts)) if fragment_counts else 0.0,
        "component_max_fragment_count": int(max(fragment_counts)) if fragment_counts else 0,
    }


def _subblock_diff_rows(
    source_subblocks: Mapping[str, list[str]],
    other_subblocks: Mapping[str, list[str]],
    *,
    source_label: str,
    other_label: str,
    top_n: int,
) -> list[dict[str, Any]]:
    other_by_signature = _signature_to_subblock(other_subblocks)
    other_sets = {key: set(values) for key, values in other_subblocks.items()}
    rows: list[dict[str, Any]] = []
    for source_key, source_members_list in source_subblocks.items():
        source_members = set(str(value) for value in source_members_list)
        overlap_counts = Counter(other_by_signature[signature_id] for signature_id in source_members)
        if not overlap_counts:
            continue
        best_other_key, best_overlap = overlap_counts.most_common(1)[0]
        best_other_members = other_sets[best_other_key]
        union_size = len(source_members | best_other_members)
        rows.append(
            {
                "source": source_label,
                "other": other_label,
                "source_subblock": source_key,
                "source_size": len(source_members),
                "best_other_subblock": best_other_key,
                "best_other_size": len(best_other_members),
                "best_overlap": int(best_overlap),
                "best_jaccard": float(best_overlap / union_size) if union_size else 1.0,
                "diff_signature_count": int(len(source_members) - best_overlap),
                "top_other_subblocks": json.dumps(overlap_counts.most_common(8)),
            }
        )
    rows.sort(key=lambda row: (-row["diff_signature_count"], row["best_jaccard"], -row["source_size"]))
    return rows[:top_n]


def _partition_change_metrics(
    python_subblocks: Mapping[str, list[str]],
    rust_subblocks: Mapping[str, list[str]],
) -> dict[str, Any]:
    python_sets = {frozenset(str(value) for value in values) for values in python_subblocks.values()}
    rust_sets = {frozenset(str(value) for value in values) for values in rust_subblocks.values()}
    python_by_signature = _signature_to_subblock(python_subblocks)
    rust_by_signature = _signature_to_subblock(rust_subblocks)
    unmatched_python_sets = [members for members in python_sets if members not in rust_sets]
    affected_members = set().union(*unmatched_python_sets) if unmatched_python_sets else set()
    return {
        "same_unlabeled_partition": bool(python_sets == rust_sets),
        "exact_matched_python_subblock_count": int(len(python_sets) - len(unmatched_python_sets)),
        "unmatched_python_subblock_count": int(len(unmatched_python_sets)),
        "affected_signature_count": int(len(affected_members)),
        "affected_signature_fraction": float(len(affected_members) / len(python_by_signature)),
        "key_changed_signature_count": int(
            sum(
                1
                for signature_id, python_key in python_by_signature.items()
                if rust_by_signature[signature_id] != python_key
            )
        ),
    }


def _write_subblocks(path: Path, subblocks: Mapping[str, list[str]]) -> None:
    serializable = {str(key): [str(value) for value in values] for key, values in sorted(subblocks.items())}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _run_python_subblocking(
    args: argparse.Namespace,
    dataset: SimpleNamespace,
    signature_ids: Sequence[str],
    config: GraphSubblockingConfig,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    fallback = make_dataset_graph_subblocking_cluster_fn(config=config)
    return make_subblocks_with_telemetry(
        signature_ids,
        dataset,
        maximum_size=int(args.maximum_size),
        specter_cluster_fn=fallback,
        use_orcid_subblocking=bool(args.orcid_subblocking),
    )


def _run_rust_subblocking(
    args: argparse.Namespace,
    arrow_dataset: ArrowDataset,
    signature_ids: Sequence[str],
    config: GraphSubblockingConfig,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    return _make_subblocks_with_telemetry_arrow_rust(
        arrow_dataset,
        signature_ids,
        maximum_size=int(args.maximum_size),
        graph_subblocking_config=config,
        graph_subblocking_random_seed=int(args.seed),
        use_orcid_subblocking=bool(args.orcid_subblocking),
    )


def _baseline_deltas(summary: dict[str, Any], baseline_path: Path | None) -> dict[str, Any]:
    if baseline_path is None:
        return {}
    baseline = json.loads(_required_path(baseline_path).read_text(encoding="utf-8"))
    deltas = {}
    for path in (
        ("rust", "component_preservation", "component_pair_recall"),
        ("rust", "seconds"),
        ("rust", "telemetry", "specter_invocation_count"),
    ):
        current: Any = summary
        old: Any = baseline
        for key in path:
            current = current.get(key, {}) if isinstance(current, dict) else {}
            old = old.get(key, {}) if isinstance(old, dict) else {}
        if isinstance(current, int | float) and isinstance(old, int | float):
            deltas[".".join(path)] = float(current) - float(old)
    return deltas


def _validated_release_subblocking_inputs(
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    config: GraphSubblockingConfig,
) -> None:
    public_data_root = Path(args.public_data_root).resolve()
    validate_release_root(public_data_root)
    dataset_manifests, _replay_bundles, _release_version = read_arrow_collection_root(
        public_data_root / "manifest.json"
    )
    dataset_manifest = dataset_manifests.get(plan["dataset"])
    if dataset_manifest is None:
        raise ValueError("Subblocking dataset is not declared by the public-data root")
    expected_arrow_root = dataset_manifest.parent
    if Path(args.arrow_root).resolve() != expected_arrow_root:
        raise ValueError("Subblocking Arrow dataset does not match the frozen evaluation plan")

    expected_component_path, expected_component_sha256 = plan["component_members"]
    if args.component_members_parquet is None:
        raise ValueError("Release subblocking requires the frozen component-members input")
    component_path = Path(args.component_members_parquet).resolve()
    if component_path != expected_component_path or sha256_file(component_path) != expected_component_sha256:
        raise ValueError("Subblocking component-members input does not match the frozen evaluation plan")
    if args.raw_root is not None or args.specter_pickle is not None or args.baseline_summary is not None:
        raise ValueError("Release subblocking does not accept unbound raw or baseline inputs")
    observed_workload = {
        "allow_full": bool(args.allow_full),
        "comparison_mode": str(args.comparison_mode),
        "graph_config": config.__dict__,
        "limit": args.limit,
        "maximum_size": int(args.maximum_size),
        "orcid_subblocking": bool(args.orcid_subblocking),
        "python_source": str(args.python_source),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "top_diff_subblocks": int(args.top_diff_subblocks),
    }
    if observed_workload != plan["workload"]:
        raise ValueError("Subblocking workload does not match the frozen evaluation plan")


def main() -> None:
    args = parse_args()
    report_path = args.output_dir / "subblocking_evaluation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Report output already exists: {report_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_plan_path = Path(args.evaluation_plan).resolve()
    evaluation_plan = _load_evaluation_plan(evaluation_plan_path)
    binding = load_run_binding(Path(args.run_binding).resolve())
    require_run_binding_matches(
        binding,
        evaluation_plan=evaluation_plan_path,
        public_data_root=Path(args.public_data_root).resolve(),
    )
    report_identity = {
        "run_binding_sha256": binding["run_binding_sha256"],
    }
    config = _graph_config(args)
    _validated_release_subblocking_inputs(args, evaluation_plan.subblocking, config)
    _log_progress(
        f"starting comparison_mode={args.comparison_mode} python_source={args.python_source} "
        f"maximum_size={args.maximum_size} limit={args.limit}"
    )
    load_start = time.perf_counter()
    if args.comparison_mode == "rust-only":
        with ArrowDataset.open(args.arrow_root, require_specter=True) as arrow_dataset:
            signature_ids = load_signature_ids_from_arrow(
                arrow_dataset,
                limit=args.limit,
                sample_mode=str(args.sample_mode),
                seed=int(args.seed),
            )
            component_labels = _load_component_labels(args.component_members_parquet, set(signature_ids))
            load_seconds = time.perf_counter() - load_start
            _log_progress(f"loaded signatures={len(signature_ids)} component_labels={len(component_labels)}")
            rust_start = time.perf_counter()
            _log_progress("running Rust graph subblocking")
            rust_subblocks, rust_telemetry = _run_rust_subblocking(args, arrow_dataset, signature_ids, config)
            rust_seconds = time.perf_counter() - rust_start
        rust_subblocks_path = args.output_dir / "rust_subblocks.json"
        _write_subblocks(rust_subblocks_path, rust_subblocks)
        rust_partition, rust_components = _validated_partition(
            rust_subblocks,
            signature_ids,
            int(args.maximum_size),
            component_labels,
        )
        summary = {
            **report_identity,
            "inputs": {
                "comparison_mode": "rust-only",
                "python_source": None,
                "raw_root": None,
                "specter_pickle": None,
                "arrow_root": str(args.arrow_root),
                "component_members_parquet": str(args.component_members_parquet)
                if args.component_members_parquet is not None
                else None,
                "limit": args.limit,
                "allow_full": bool(args.allow_full),
                "sample_mode": str(args.sample_mode),
                "seed": int(args.seed),
                "maximum_size": int(args.maximum_size),
                "orcid_subblocking": bool(args.orcid_subblocking),
                "load_seconds": float(load_seconds),
            },
            "graph_config": config.__dict__,
            "counts": {
                "signature_count": int(len(signature_ids)),
            },
            "rust": {
                "seconds": float(rust_seconds),
                "telemetry": rust_telemetry,
                "partition": rust_partition,
                "component_preservation": rust_components,
            },
            "artifacts": {
                "summary": str(report_path),
                "rust_subblocks": str(rust_subblocks_path),
            },
        }
        summary["baseline_deltas"] = _baseline_deltas(summary, args.baseline_summary)
        _write_fresh_json(report_path, summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    dataset: SimpleNamespace | None = None
    if args.python_source == "arrow":
        with ArrowDataset.open(args.arrow_root, require_specter=True) as arrow_dataset:
            with arrow_dataset.use(require_specter=True) as arrow_lease:
                dataset, signature_ids = load_lightweight_dataset_from_arrow(
                    arrow_lease,
                    limit=args.limit,
                    sample_mode=str(args.sample_mode),
                    seed=int(args.seed),
                    include_specter=True,
                )
    else:
        if args.raw_root is None or args.specter_pickle is None:
            raise ValueError("--python-source raw requires --raw-root and --specter-pickle")
        dataset, signature_ids = load_lightweight_dataset(
            args.raw_root,
            args.specter_pickle,
            limit=args.limit,
            sample_mode=str(args.sample_mode),
            seed=int(args.seed),
        )
    component_labels = _load_component_labels(args.component_members_parquet, set(signature_ids))
    load_seconds = time.perf_counter() - load_start
    _log_progress(f"loaded signatures={len(signature_ids)} component_labels={len(component_labels)}")
    source_label = "python"
    source_start = time.perf_counter()
    if dataset is None:
        raise RuntimeError("Python comparison requires a loaded Python dataset")
    _log_progress("running Python graph subblocking")
    source_subblocks, source_telemetry = _run_python_subblocking(args, dataset, signature_ids, config)
    source_hook_telemetry = {}
    source_seconds = time.perf_counter() - source_start
    with ArrowDataset.open(args.arrow_root, require_specter=True) as arrow_dataset:
        rust_start = time.perf_counter()
        _log_progress("running Rust graph subblocking")
        rust_subblocks, rust_telemetry = _run_rust_subblocking(args, arrow_dataset, signature_ids, config)
        rust_seconds = time.perf_counter() - rust_start
    if set(_signature_to_subblock(source_subblocks)) != set(_signature_to_subblock(rust_subblocks)):
        raise ValueError(f"{source_label} and Rust partitions cover different signature IDs")

    source_diff_path = args.output_dir / f"diff_heavy_{source_label}_subblocks.csv"
    rust_diff_path = args.output_dir / "diff_heavy_rust_subblocks.csv"
    pd.DataFrame(
        _subblock_diff_rows(
            source_subblocks,
            rust_subblocks,
            source_label=source_label,
            other_label="rust",
            top_n=int(args.top_diff_subblocks),
        )
    ).to_csv(source_diff_path, index=False)
    pd.DataFrame(
        _subblock_diff_rows(
            rust_subblocks,
            source_subblocks,
            source_label="rust",
            other_label=source_label,
            top_n=int(args.top_diff_subblocks),
        )
    ).to_csv(rust_diff_path, index=False)
    _write_subblocks(args.output_dir / f"{source_label}_subblocks.json", source_subblocks)
    _write_subblocks(args.output_dir / "rust_subblocks.json", rust_subblocks)
    source_partition, source_components = _validated_partition(
        source_subblocks,
        signature_ids,
        int(args.maximum_size),
        component_labels,
    )
    rust_partition, rust_components = _validated_partition(
        rust_subblocks,
        signature_ids,
        int(args.maximum_size),
        component_labels,
    )
    source_subblocks_path = args.output_dir / f"{source_label}_subblocks.json"
    rust_subblocks_path = args.output_dir / "rust_subblocks.json"

    summary = {
        **report_identity,
        "inputs": {
            "comparison_mode": "python-vs-rust",
            "python_source": str(args.python_source),
            "raw_root": str(args.raw_root) if args.raw_root is not None else None,
            "specter_pickle": str(args.specter_pickle) if args.specter_pickle is not None else None,
            "arrow_root": str(args.arrow_root),
            "component_members_parquet": str(args.component_members_parquet)
            if args.component_members_parquet is not None
            else None,
            "limit": args.limit,
            "allow_full": bool(args.allow_full),
            "sample_mode": str(args.sample_mode),
            "seed": int(args.seed),
            "maximum_size": int(args.maximum_size),
            "orcid_subblocking": bool(args.orcid_subblocking),
            "load_seconds": float(load_seconds),
        },
        "graph_config": config.__dict__,
        "counts": {
            "signature_count": int(len(signature_ids)),
            **_partition_change_metrics(source_subblocks, rust_subblocks),
        },
        source_label: {
            "seconds": float(source_seconds),
            "telemetry": source_telemetry,
            "hook_telemetry": source_hook_telemetry,
            "partition": source_partition,
            "component_preservation": source_components,
        },
        "rust": {
            "seconds": float(rust_seconds),
            "telemetry": rust_telemetry,
            "partition": rust_partition,
            "component_preservation": rust_components,
        },
        "artifacts": {
            "summary": str(report_path),
            f"{source_label}_diff_csv": str(source_diff_path),
            "rust_diff_csv": str(rust_diff_path),
            f"{source_label}_subblocks": str(source_subblocks_path),
            "rust_subblocks": str(rust_subblocks_path),
        },
    }
    summary["baseline_deltas"] = _baseline_deltas(summary, args.baseline_summary)
    _write_fresh_json(report_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
