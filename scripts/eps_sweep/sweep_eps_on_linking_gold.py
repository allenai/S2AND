"""Sweep FastCluster EPS values against linker-derived pair gold.

This runner consumes Arrow linker-replay datasets, plans target-bearing blocks
from Arrow signature metadata, computes pairwise distances through the
Arrow/Rust featurizer path, and sweeps FastCluster EPS values by reusing one
linkage tree per subblock.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from s2and.arrow_inputs import ValidatedArrowInputs
from s2and.incremental_linking_training.classic import _drop_unlabeled_singleton_orcid_rows
from scripts.eps_sweep.common import (
    DEFAULT_ARROW_ROOT,
    DEFAULT_GOLD_ROOT,
    DEFAULT_LINKER_BUNDLE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    load_arrow_paths,
    sha1_text,
    write_json,
)

WEIGHT_COLUMNS = [
    "weight_pair",
    "weight_query_balanced",
    "weight_query_label_balanced",
    "weight_query_class_balanced",
]


@dataclass
class ArrowPlanningState:
    """Minimal Arrow-derived state needed for target-block subblocking."""

    name: str
    signatures: dict[str, Any]
    block_dict: dict[str, list[str]]
    random_seed: int = 42
    block_type: str = "s2"

    def get_blocks(self) -> dict[str, list[str]]:
        """Return a copy of the raw S2 block mapping."""

        return {str(block_key): list(signature_ids) for block_key, signature_ids in self.block_dict.items()}


@dataclass(frozen=True)
class CachedLinkageBlock:
    """One selected subblock with a reusable FastCluster linkage tree."""

    block_key: str
    signature_ids: list[str]
    linkage_matrix: Any | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Cache Arrow/Rust pairwise distances, sweep EPS, and score linker-derived pair gold."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_LINKER_BUNDLE_ROOT)
    parser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    parser.add_argument("--model-path", type=Path, required=True, help="Complete native production bundle path.")
    parser.add_argument("--gold-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "rust"], default="rust")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batching-threshold", type=int, default=5000)
    parser.add_argument("--pair-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--use-orcid-subblocking",
        action="store_true",
        help="Use ORCID values during subblock merging. Defaults off for EPS selection.",
    )
    orcid_constraint_group = parser.add_mutually_exclusive_group()
    orcid_constraint_group.add_argument(
        "--suppress-orcid-constraints",
        action="store_true",
        default=True,
        help="Disable same-ORCID hard-link distance constraints. This is the default for EPS selection.",
    )
    orcid_constraint_group.add_argument(
        "--use-orcid-constraints",
        dest="suppress_orcid_constraints",
        action="store_false",
        help="Enable same-ORCID hard-link distance constraints for old-artifact parity checks.",
    )
    parser.add_argument("--eps-start", type=float, default=0.40)
    parser.add_argument("--eps-stop", type=float, default=0.80)
    parser.add_argument("--eps-step", type=float, default=0.05)
    parser.add_argument("--eps-values", nargs="*", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Plan subblocks and pair counts without pairwise work.")
    parser.add_argument(
        "--compute-missing-dists",
        action="store_true",
        help="Compute and cache missing Arrow/Rust pairwise distance matrices. Without this, missing caches fail fast.",
    )
    parser.add_argument("--overwrite-dists", action="store_true")
    parser.add_argument(
        "--max-subblocks",
        type=int,
        default=None,
        help="Optional smoke limit after subblock selection.",
    )
    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help="Permit --compute-missing-dists without --max-subblocks.",
    )
    parser.add_argument(
        "--subblock-selection",
        choices=["gold-heavy", "smallest"],
        default="gold-heavy",
        help="Ordering used with --max-subblocks.",
    )
    parser.add_argument("--write-assignments", action="store_true")
    return parser.parse_args(argv)


def _signature_ids_digest(signature_ids: Iterable[str]) -> str:
    """Return a stable digest for a sequence of signature ids."""

    return sha1_text("\n".join(str(sig_id) for sig_id in signature_ids))


def _safe_cache_name(dataset: str, index: int, block_key: str) -> str:
    """Return a filesystem-safe distance cache filename."""

    return f"{dataset}_{index:03d}_{sha1_text(block_key)[:12]}.pkl"


def _eps_values(args: argparse.Namespace, model_eps: float | None) -> list[float]:
    """Return sorted EPS values, including model EPS when it falls in range."""

    if args.eps_values:
        values = [round(float(value), 10) for value in args.eps_values]
    else:
        count = int(round((args.eps_stop - args.eps_start) / args.eps_step))
        values = [round(args.eps_start + i * args.eps_step, 10) for i in range(count + 1)]
        values = [value for value in values if value <= args.eps_stop + 1e-9]
    if model_eps is not None and args.eps_start <= model_eps <= args.eps_stop:
        values.append(round(float(model_eps), 10))
    return sorted(set(values))


def _validate_args(args: argparse.Namespace) -> None:
    """Validate cheap CLI invariants before doing I/O."""

    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")
    if args.batching_threshold <= 0:
        raise ValueError("--batching-threshold must be > 0")
    if args.pair_chunk_size <= 0:
        raise ValueError("--pair-chunk-size must be > 0")
    if args.eps_step <= 0:
        raise ValueError("--eps-step must be > 0")
    if args.eps_stop < args.eps_start:
        raise ValueError("--eps-stop must be >= --eps-start")
    if args.max_subblocks is not None and args.max_subblocks <= 0:
        raise ValueError("--max-subblocks must be > 0 when set")
    if args.compute_missing_dists and args.max_subblocks is None and not args.allow_full_run:
        raise ValueError("--compute-missing-dists requires --max-subblocks or explicit --allow-full-run")


def _gold_path(args: argparse.Namespace) -> Path:
    """Return the gold constraints path."""

    if args.gold_path is not None:
        return args.gold_path
    return DEFAULT_GOLD_ROOT / f"{args.dataset}_pair_constraints.parquet"


def _output_dir(args: argparse.Namespace) -> Path:
    """Return the output directory for one dataset."""

    if args.output_dir is not None:
        return args.output_dir
    return DEFAULT_OUTPUT_ROOT / args.dataset


def _load_gold(path: Path) -> pd.DataFrame:
    """Load and validate pair gold constraints."""

    if not path.exists():
        raise FileNotFoundError(path)
    gold = pd.read_parquet(path)
    required = {
        "dataset",
        "table_name",
        "split",
        "source_key",
        "supervision_type",
        "query_signature_id",
        "member_signature_id",
        "label",
        *WEIGHT_COLUMNS,
    }
    missing = sorted(required - set(gold.columns))
    if missing:
        raise ValueError(f"Gold file is missing required columns: {missing}")
    for column in [
        "dataset",
        "table_name",
        "split",
        "source_key",
        "supervision_type",
        "query_signature_id",
        "member_signature_id",
        "query_view",
    ]:
        if column in gold.columns:
            gold[column] = gold[column].astype(str)
    gold["label"] = pd.to_numeric(gold["label"], errors="raise").astype("int8")
    for column in WEIGHT_COLUMNS:
        gold[column] = pd.to_numeric(gold[column], errors="raise").astype(float)
    gold, filter_summary = _drop_unlabeled_singleton_orcid_rows(gold, context=f"eps_gold:{path.name}")
    if filter_summary["rows_removed"]:
        logging.info(
            "dropped unlabeled_singleton_orcid rows from gold path=%s rows_removed=%d rows_after=%d",
            path,
            filter_summary["rows_removed"],
            filter_summary["rows_after"],
        )
    return gold


def _optional_str(value: Any) -> str | None:
    """Return a string or None for nullable Arrow/Pandas scalar values."""

    if value is None or pd.isna(value):
        return None
    return str(value)


def _read_arrow_signatures_for_planning(
    args: argparse.Namespace,
    arrow_paths: ValidatedArrowInputs,
    target_signature_ids: set[str],
) -> ArrowPlanningState:
    """Read Arrow signature metadata needed for target-bearing base blocks."""

    signatures_path = Path(arrow_paths["signatures"])
    with signatures_path.open("rb") as infile:
        table = ipc.open_file(infile).read_all()
    required = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_position",
        "author_block",
    }
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"Arrow signatures table is missing planning columns: {missing}")

    selected_columns = required.union({"author_orcid"}.intersection(table.column_names))
    table = table.select(sorted(selected_columns))
    target_values = pa.array(sorted(target_signature_ids), type=table["signature_id"].type)
    pc_any = cast(Any, pc)
    target_table = table.filter(pc_any.is_in(table["signature_id"], value_set=target_values))
    found_target_ids = {str(signature_id) for signature_id in target_table["signature_id"].to_pylist()}
    missing_target_ids = sorted(target_signature_ids - found_target_ids)
    if missing_target_ids:
        raise ValueError(f"Arrow signatures table is missing target signature ids: {missing_target_ids[:10]}")

    target_blocks = sorted({_optional_str(block_key) or "" for block_key in target_table["author_block"].to_pylist()})
    block_column = pc.fill_null(table["author_block"], "")
    block_values = pa.array(target_blocks, type=block_column.type)
    data = table.filter(pc_any.is_in(block_column, value_set=block_values)).to_pydict()
    signatures: dict[str, Any] = {}
    block_dict: dict[str, list[str]] = defaultdict(list)
    row_count = len(data["signature_id"])
    for index in range(row_count):
        signature_id = str(data["signature_id"][index])
        block_key = _optional_str(data["author_block"][index]) or ""
        block_dict[block_key].append(signature_id)
        signatures[signature_id] = SimpleNamespace(
            signature_id=signature_id,
            paper_id=_optional_str(data["paper_id"][index]),
            author_info_first=_optional_str(data["author_first"][index]),
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle=_optional_str(data["author_middle"][index]),
            author_info_middle_normalized_without_apostrophe=None,
            author_info_last=_optional_str(data["author_last"][index]),
            author_info_last_normalized=None,
            author_info_suffix=_optional_str(data["author_suffix"][index]),
            author_info_suffix_normalized=None,
            author_info_orcid=_optional_str(data.get("author_orcid", [None] * row_count)[index]),
            author_info_position=data["author_position"][index],
        )

    return ArrowPlanningState(
        name=f"{args.dataset}_linking_eps_arrow_plan",
        signatures=signatures,
        block_dict=dict(block_dict),
        random_seed=42,
    )


def _select_subblock_rows(
    rows: list[dict[str, Any]],
    selection: str,
    max_subblocks: int | None,
) -> list[dict[str, Any]]:
    """Order and optionally limit selected subblock rows."""

    if max_subblocks is None:
        return rows
    if selection == "gold-heavy":
        ordered = sorted(
            rows,
            key=lambda row: (-int(row["intra_gold_pair_count"]), int(row["pair_count"]), str(row["block_key"])),
        )
    elif selection == "smallest":
        rows_with_gold = [row for row in rows if int(row["intra_gold_pair_count"]) > 0 and int(row["pair_count"]) > 0]
        fallback_rows = [row for row in rows if row not in rows_with_gold]
        ordered = sorted(
            rows_with_gold,
            key=lambda row: (int(row["pair_count"]), int(row["signature_count"]), str(row["block_key"])),
        ) + sorted(
            fallback_rows,
            key=lambda row: (int(row["pair_count"]), int(row["signature_count"]), str(row["block_key"])),
        )
    else:
        raise ValueError(f"Unknown subblock selection: {selection}")
    return ordered[:max_subblocks]


def _make_arrow_specter_cluster_fn(
    clusterer: Any,
    arrow_paths: Mapping[str, str],
    signature_ids: Sequence[str],
) -> Any:
    """Return the Arrow-backed graph fallback used by S2AND subblocking."""

    from s2and.subblocking import (
        _resolve_graph_subblocking_config,
        make_arrow_graph_subblocking_cluster_fn,
    )

    config = _resolve_graph_subblocking_config(getattr(clusterer, "subblocking_graph_config", None))
    return make_arrow_graph_subblocking_cluster_fn(
        arrow_paths,
        signature_ids,
        config=config,
        random_seed=int(getattr(clusterer, "random_state", 0) or 0),
    )


def _build_arrow_subblocked_block_dict(
    *,
    args: argparse.Namespace,
    clusterer: Any,
    planning_state: ArrowPlanningState,
    block_dict: Mapping[str, list[str]],
    target_signature_ids: set[str],
    arrow_paths: Mapping[str, str],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Subblock only target-bearing base blocks using Arrow-backed fallback."""

    from s2and.subblocking import make_subblocks_with_telemetry

    threshold = int(args.batching_threshold)
    subblocked: dict[str, list[str]] = {}
    telemetry_rows: list[dict[str, Any]] = []
    for block_key, signature_ids in sorted(block_dict.items()):
        signature_ids = [str(signature_id) for signature_id in signature_ids]
        if not (target_signature_ids & set(signature_ids)):
            continue
        if len(signature_ids) <= threshold:
            subblocked[str(block_key)] = signature_ids
            continue

        started = time.perf_counter()
        specter_cluster_fn = _make_arrow_specter_cluster_fn(clusterer, arrow_paths, signature_ids)
        subblocks, telemetry = make_subblocks_with_telemetry(
            signature_ids,
            planning_state,
            maximum_size=threshold,
            specter_cluster_fn=specter_cluster_fn,
            use_orcid_subblocking=bool(args.use_orcid_subblocking),
        )
        seconds = time.perf_counter() - started
        for subblock_key in sorted(subblocks):
            subblock_signatures = [str(signature_id) for signature_id in subblocks[subblock_key]]
            if len(subblock_signatures) > threshold:
                raise ValueError(
                    f"Arrow subblocking produced oversized subblock {block_key}|subblock={subblock_key}: "
                    f"{len(subblock_signatures)} > {threshold}"
                )
            subblocked[f"{block_key}|subblock={subblock_key}"] = subblock_signatures

        telemetry_rows.append(
            {
                "block_key": str(block_key),
                "input_signature_count": len(signature_ids),
                "output_subblock_count": len(subblocks),
                "seconds": round(seconds, 3),
                "arrow_graph_load_seconds": round(float(getattr(specter_cluster_fn, "load_seconds", 0.0)), 3),
                "arrow_graph_stats": getattr(specter_cluster_fn, "stats", []),
                "subblocking_telemetry": telemetry,
            }
        )
    return subblocked, telemetry_rows


def _plan_subblocks(
    args: argparse.Namespace,
    clusterer: Any,
    planning_state: ArrowPlanningState,
    gold: pd.DataFrame,
    arrow_paths: Mapping[str, str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Plan target subblocks from Arrow metadata and Arrow-backed subblocking."""

    target_signature_ids = set(gold["query_signature_id"].astype(str)) | set(gold["member_signature_id"].astype(str))
    block_dict = planning_state.get_blocks()
    subblocked, subblocking_telemetry = _build_arrow_subblocked_block_dict(
        args=args,
        clusterer=clusterer,
        planning_state=planning_state,
        block_dict=block_dict,
        target_signature_ids=target_signature_ids,
        arrow_paths=arrow_paths,
    )
    selected: dict[str, list[str]] = {}
    subblock_rows: list[dict[str, Any]] = []
    signature_to_subblock: dict[str, str] = {}
    for block_key, signature_ids in sorted(subblocked.items()):
        signature_ids = [str(sig_id) for sig_id in signature_ids]
        for signature_id in signature_ids:
            signature_to_subblock[signature_id] = block_key
        target_count = len(target_signature_ids & set(signature_ids))
        if target_count == 0:
            continue
        pair_count = len(signature_ids) * (len(signature_ids) - 1) // 2
        selected[block_key] = signature_ids
        subblock_rows.append(
            {
                "block_key": block_key,
                "signature_count": len(signature_ids),
                "target_signature_count": target_count,
                "pair_count": pair_count,
            }
        )

    gold_subblocks = pd.DataFrame(
        {
            "_query_subblock": gold["query_signature_id"].map(signature_to_subblock),
            "_member_subblock": gold["member_signature_id"].map(signature_to_subblock),
            "label": gold["label"].to_numpy(),
        }
    )
    gold_subblocks = gold_subblocks[
        gold_subblocks["_query_subblock"].notna()
        & gold_subblocks["_member_subblock"].notna()
        & (gold_subblocks["_query_subblock"] == gold_subblocks["_member_subblock"])
    ].copy()
    intra_counts = gold_subblocks.groupby("_query_subblock", sort=False).size()
    intra_positive_counts = (
        gold_subblocks[gold_subblocks["label"].astype(int) == 1].groupby("_query_subblock", sort=False).size()
    )
    intra_negative_counts = (
        gold_subblocks[gold_subblocks["label"].astype(int) == 0].groupby("_query_subblock", sort=False).size()
    )
    for row in subblock_rows:
        block_key = str(row["block_key"])
        row["intra_gold_pair_count"] = int(intra_counts.get(block_key, 0))
        row["intra_gold_positive_count"] = int(intra_positive_counts.get(block_key, 0))
        row["intra_gold_negative_count"] = int(intra_negative_counts.get(block_key, 0))

    selected_rows = _select_subblock_rows(subblock_rows, args.subblock_selection, args.max_subblocks)
    selected = {str(row["block_key"]): selected[str(row["block_key"])] for row in selected_rows}

    selected_signature_ids = {sig_id for signature_ids in selected.values() for sig_id in signature_ids}
    covered_target_signature_ids = target_signature_ids & selected_signature_ids
    plan = {
        "dataset": args.dataset,
        "gold_rows": int(len(gold)),
        "gold_queries": int(gold["query_signature_id"].nunique()),
        "gold_target_signature_count": int(len(target_signature_ids)),
        "covered_target_signature_count": int(len(covered_target_signature_ids)),
        "missing_target_signature_count": int(len(target_signature_ids - covered_target_signature_ids)),
        "base_blocks": {block_key: len(signature_ids) for block_key, signature_ids in block_dict.items()},
        "selected_subblock_count": len(selected),
        "selected_signature_count": int(sum(len(ids) for ids in selected.values())),
        "selected_pair_count": int(sum(row["pair_count"] for row in selected_rows)),
        "subblock_selection": args.subblock_selection,
        "subblocking_backend": "arrow-graph",
        "subblocking_telemetry": subblocking_telemetry,
        "subblocks": selected_rows,
    }
    return selected, plan


def _cache_metadata(
    args: argparse.Namespace,
    block_key: str,
    signature_ids: list[str],
    arrow_paths_digest: str,
) -> dict[str, Any]:
    """Return metadata that must match for a cached distance vector."""

    model_fingerprint = _model_fingerprint(args)
    return {
        "dataset": args.dataset,
        **model_fingerprint,
        "arrow_root": str(args.arrow_root.resolve()),
        "arrow_paths_digest": arrow_paths_digest,
        "block_key": block_key,
        "signature_count": len(signature_ids),
        "signature_ids_digest": _signature_ids_digest(signature_ids),
        "pair_count": len(signature_ids) * (len(signature_ids) - 1) // 2,
        "batching_threshold": int(args.batching_threshold),
        "pair_chunk_size": int(args.pair_chunk_size),
        "use_orcid_subblocking": bool(args.use_orcid_subblocking),
        "suppress_orcid_constraints": bool(args.suppress_orcid_constraints),
        "distance_source": "arrow-rust",
    }


def _hash_file(path: Path) -> str:
    """Return a SHA256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_stat_cache_key(path: Path) -> tuple[Any, ...]:
    if not path.exists():
        return ("missing", str(path))
    stat = path.stat()
    if path.is_file():
        return ("file", str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if path.is_dir():
        entries: list[tuple[str, int, int]] = []
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            child_stat = child.stat()
            entries.append(
                (
                    child.relative_to(path).as_posix(),
                    int(child_stat.st_size),
                    int(child_stat.st_mtime_ns),
                )
            )
        return ("dir", str(path), tuple(entries))
    return ("other", str(path), int(stat.st_size), int(stat.st_mtime_ns))


def _hash_directory(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    digest.update(b"s2and-directory-fingerprint-v1\0")
    total_size = 0
    max_mtime_ns = int(path.stat().st_mtime_ns)
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        child_stat = child.stat()
        rel_path = child.relative_to(path).as_posix()
        total_size += int(child_stat.st_size)
        max_mtime_ns = max(max_mtime_ns, int(child_stat.st_mtime_ns))
        rel_bytes = rel_path.encode("utf-8")
        digest.update(len(rel_bytes).to_bytes(8, "little", signed=False))
        digest.update(rel_bytes)
        digest.update(int(child_stat.st_size).to_bytes(8, "little", signed=False))
        with child.open("rb") as infile:
            for chunk in iter(lambda: infile.read(1024 * 1024), b""):
                digest.update(chunk)
    return total_size, max_mtime_ns, digest.hexdigest()


def _arrow_paths_content_digest(arrow_paths: Mapping[str, str]) -> str:
    """Digest Arrow path identity and file metadata for cache invalidation."""

    digest = hashlib.sha256()
    digest.update(b"s2and-eps-arrow-paths-v2\0")
    for key, raw_path in sorted((str(key), str(value)) for key, value in arrow_paths.items()):
        path = Path(raw_path)
        resolved = path.resolve() if path.exists() else path
        key_bytes = key.encode("utf-8")
        path_bytes = str(resolved).encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little", signed=False))
        digest.update(key_bytes)
        digest.update(len(path_bytes).to_bytes(8, "little", signed=False))
        digest.update(path_bytes)
        if path.is_file():
            stat = path.stat()
            digest.update(b"file\0")
            digest.update(int(stat.st_size).to_bytes(8, "little", signed=False))
            digest.update(int(stat.st_mtime_ns).to_bytes(8, "little", signed=False))
        elif path.is_dir():
            stat_key = _path_stat_cache_key(path)
            digest.update(b"dir\0")
            digest.update(repr(stat_key).encode("utf-8"))
        else:
            digest.update(b"missing\0")
    return digest.hexdigest()


def _model_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    """Return a stable fingerprint for the model artifact used by distance caches."""

    model_path = args.model_path.resolve()
    model_stat = model_path.stat() if model_path.exists() else None
    if model_stat is None:
        return {"model_path": str(model_path), "model_size": None, "model_mtime_ns": None, "model_sha256": None}
    cache_key = _path_stat_cache_key(model_path)
    cached = getattr(args, "_s2and_model_fingerprint_cache", None)
    if cached is not None and cached[0] == cache_key:
        return dict(cached[1])
    if model_path.is_dir():
        model_size, model_mtime_ns, model_sha256 = _hash_directory(model_path)
    else:
        model_size = int(model_stat.st_size)
        model_mtime_ns = int(model_stat.st_mtime_ns)
        model_sha256 = _hash_file(model_path)
    fingerprint = {
        "model_path": str(model_path),
        "model_size": model_size,
        "model_mtime_ns": model_mtime_ns,
        "model_sha256": model_sha256,
    }
    try:
        args._s2and_model_fingerprint_cache = (cache_key, dict(fingerprint))
    except Exception:
        pass
    return fingerprint


def _distance_cache_path(cache_dir: Path, dataset: str, index: int, block_key: str) -> Path:
    """Return the cache path for one subblock."""

    return cache_dir / _safe_cache_name(dataset, index, block_key)


def _load_cached_distance(path: Path, expected_metadata: Mapping[str, Any]) -> Any:
    """Load one cached distance matrix after validating metadata."""

    with path.open("rb") as infile:
        payload = pickle.load(infile)
    metadata = dict(payload.get("metadata", {}))
    for key in [
        "dataset",
        "model_path",
        "model_size",
        "model_mtime_ns",
        "model_sha256",
        "arrow_root",
        "arrow_paths_digest",
        "distance_source",
        "block_key",
        "signature_count",
        "signature_ids_digest",
        "use_orcid_subblocking",
        "suppress_orcid_constraints",
    ]:
        if metadata.get(key) != expected_metadata.get(key):
            raise ValueError(f"Distance cache metadata mismatch for {path}: key={key}")
    return payload["dist"]


def _build_arrow_featurizer(
    clusterer: Any,
    arrow_paths: ValidatedArrowInputs,
    signature_ids: Sequence[str],
) -> Any:
    """Build a Rust featurizer from Arrow paths for the requested signatures."""

    from s2and.arrow_inputs import require_feature_contract_normalization_version
    from s2and.feature_port import build_rust_featurizer_from_arrow_paths

    return build_rust_featurizer_from_arrow_paths(
        arrow_paths,
        expected_normalization_version=require_feature_contract_normalization_version(
            clusterer,
            context="epsilon sweep Arrow featurizer",
        ),
        signature_ids=signature_ids,
        name_tuples=None,
        load_name_counts="name_counts_index" in arrow_paths,
        preprocess=True,
        num_threads=int(clusterer.n_jobs),
    )


def _ensure_distance_caches(
    args: argparse.Namespace,
    clusterer: Any,
    selected_subblocks: dict[str, list[str]],
    cache_dir: Path,
    arrow_paths: ValidatedArrowInputs,
) -> list[dict[str, Any]]:
    """Ensure Arrow/Rust distance caches exist for selected subblocks."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    original_batch_size = clusterer.batch_size
    clusterer.batch_size = int(args.pair_chunk_size)
    arrow_paths_digest = _arrow_paths_content_digest(arrow_paths)
    cache_rows: list[dict[str, Any]] = []
    rows_by_block: dict[str, dict[str, Any]] = {}
    blocks_to_compute: dict[str, list[str]] = {}
    try:
        for index, (block_key, signature_ids) in enumerate(sorted(selected_subblocks.items())):
            metadata = _cache_metadata(args, block_key, signature_ids, arrow_paths_digest)
            cache_path = _distance_cache_path(cache_dir, args.dataset, index, block_key)
            row = {
                **metadata,
                "cache_path": str(cache_path),
                "cache_exists_before": cache_path.exists(),
                "computed": False,
                "seconds": 0.0,
            }
            if cache_path.exists() and not args.overwrite_dists:
                _load_cached_distance(cache_path, metadata)
                cache_rows.append(row)
                continue
            if len(signature_ids) <= 1:
                cache_rows.append(row)
                continue
            if not args.compute_missing_dists:
                raise SystemExit(
                    f"Missing distance cache for block {block_key!r}: {cache_path}. "
                    "Pass --compute-missing-dists to run Arrow/Rust pairwise prediction."
                )
            cache_rows.append(row)
            rows_by_block[block_key] = row
            blocks_to_compute[block_key] = signature_ids

        if blocks_to_compute:
            featurizer_signature_ids = list(
                dict.fromkeys(
                    signature_id for signature_ids in blocks_to_compute.values() for signature_id in signature_ids
                )
            )
            featurizer_started = time.perf_counter()
            rust_featurizer = _build_arrow_featurizer(clusterer, arrow_paths, featurizer_signature_ids)
            featurizer_seconds = time.perf_counter() - featurizer_started
            logging.info(
                "built Arrow/Rust featurizer signatures=%d blocks=%d seconds=%.3f",
                len(featurizer_signature_ids),
                len(blocks_to_compute),
                featurizer_seconds,
            )

        for block_key, signature_ids in blocks_to_compute.items():
            started = time.perf_counter()
            dists = clusterer.make_distance_matrices_from_rust_featurizer(
                {block_key: signature_ids},
                rust_featurizer,
                pair_chunk_size=int(args.pair_chunk_size),
            )
            seconds = time.perf_counter() - started
            metadata = _cache_metadata(args, block_key, signature_ids, arrow_paths_digest)
            cache_path = Path(rows_by_block[block_key]["cache_path"])
            with cache_path.open("wb") as outfile:
                pickle.dump({"metadata": metadata, "dist": dists[block_key]}, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            row = rows_by_block[block_key]
            row["computed"] = True
            row["seconds"] = round(seconds, 3)
            row["featurizer_batch_signature_count"] = len(featurizer_signature_ids)
            row["featurizer_batch_seconds"] = round(featurizer_seconds, 3)
            logging.info(
                "cached Arrow/Rust distances block=%s signatures=%d pairs=%d seconds=%.3f path=%s",
                block_key,
                len(signature_ids),
                metadata["pair_count"],
                seconds,
                cache_path,
            )
    finally:
        clusterer.batch_size = original_batch_size
    return cache_rows


def _normalize_cluster_labels(labels: Any, expected_count: int) -> list[int]:
    """Return FastCluster labels as stable Python ints."""

    import numpy as np

    normalized = np.asarray(labels).copy()
    if len(normalized) != expected_count:
        raise ValueError(f"Cluster model returned {len(normalized)} labels for {expected_count} signatures")
    if len(normalized) > 0:
        max_label = int(normalized.max())
        outlier_locations = np.where(normalized == -1)[0]
        for offset, location in enumerate(outlier_locations):
            normalized[location] = max_label + 1 + offset
    return [int(label) for label in normalized]


def _linkage_from_cached_distance(
    *,
    clusterer: Any,
    block_key: str,
    signature_ids: Sequence[str],
    dist: Any,
) -> Any | None:
    """Build the FastCluster linkage tree once for a cached distance vector."""

    import numpy as np
    from fastcluster import linkage

    from s2and.model_pairwise import FastCluster

    if len(signature_ids) <= 1:
        return None

    cluster_model = clusterer.cluster_model
    if not isinstance(cluster_model, FastCluster):
        raise ValueError(f"Cached linkage EPS sweep requires FastCluster, got {type(cluster_model).__name__}")
    if bool(getattr(cluster_model, "input_as_observation_matrix", False)):
        raise ValueError("EPS sweep expects cached condensed distance vectors, not observation matrices")
    dist_array = np.asarray(dist)
    expected_pair_count = len(signature_ids) * (len(signature_ids) - 1) // 2
    if dist_array.ndim != 1 or len(dist_array) != expected_pair_count:
        raise ValueError(
            f"Cached distance vector for block {block_key} must be 1-D with {expected_pair_count} pairs, "
            f"got shape={dist_array.shape}"
        )
    linkage_method = str(cluster_model.linkage)
    cluster_model_name = type(cluster_model).__name__
    logging.info(
        "Starting linkage build for block %s using %s (signatures=%d)",
        block_key,
        cluster_model_name,
        len(signature_ids),
    )
    started = time.perf_counter()
    linkage_matrix = linkage(dist_array, linkage_method, preserve_input=True)
    logging.info(
        "Finished linkage build for block %s using %s in %.3fs",
        block_key,
        cluster_model_name,
        time.perf_counter() - started,
    )
    return linkage_matrix


def _labels_from_cached_linkage(
    *,
    linkage_matrix: Any | None,
    signature_ids: Sequence[str],
    eps: float,
) -> list[int]:
    """Cut a cached FastCluster linkage tree at one EPS value."""

    from scipy.cluster.hierarchy import fcluster

    if len(signature_ids) <= 1:
        return [0]
    if linkage_matrix is None:
        raise ValueError("Missing linkage matrix for multi-signature subblock")
    labels = fcluster(linkage_matrix, t=float(eps), criterion="distance")
    return _normalize_cluster_labels(labels, len(signature_ids))


def _build_linkage_blocks(
    *,
    args: argparse.Namespace,
    clusterer: Any,
    selected_subblocks: dict[str, list[str]],
    cache_dir: Path,
    arrow_paths: Mapping[str, str],
) -> tuple[dict[str, CachedLinkageBlock], list[dict[str, Any]]]:
    """Load cached distances once and build reusable linkage trees."""

    linkage_blocks: dict[str, CachedLinkageBlock] = {}
    linkage_rows: list[dict[str, Any]] = []
    arrow_paths_digest = _arrow_paths_content_digest(arrow_paths)
    for index, (block_key, signature_ids) in enumerate(sorted(selected_subblocks.items())):
        started = time.perf_counter()
        signature_ids = [str(signature_id) for signature_id in signature_ids]
        pair_count = len(signature_ids) * (len(signature_ids) - 1) // 2
        linkage_matrix = None
        metadata = _cache_metadata(args, block_key, signature_ids, arrow_paths_digest)
        if len(signature_ids) > 1:
            cache_path = _distance_cache_path(cache_dir, args.dataset, index, block_key)
            dist = _load_cached_distance(cache_path, metadata)
            linkage_matrix = _linkage_from_cached_distance(
                clusterer=clusterer,
                block_key=block_key,
                signature_ids=signature_ids,
                dist=dist,
            )
        linkage_blocks[block_key] = CachedLinkageBlock(
            block_key=block_key,
            signature_ids=signature_ids,
            linkage_matrix=linkage_matrix,
        )
        linkage_rows.append(
            {
                "block_key": block_key,
                "signature_count": len(signature_ids),
                "pair_count": pair_count,
                "linkage_computed": len(signature_ids) > 1,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    return linkage_blocks, linkage_rows


def _cluster_assignments_for_eps(
    *,
    eps: float,
    linkage_blocks: Mapping[str, CachedLinkageBlock],
) -> dict[str, str]:
    """Cluster selected subblocks by cutting cached linkage trees at one EPS value."""

    assignments: dict[str, str] = {}
    for block_key, linkage_block in sorted(linkage_blocks.items()):
        labels = _labels_from_cached_linkage(
            linkage_matrix=linkage_block.linkage_matrix,
            signature_ids=linkage_block.signature_ids,
            eps=eps,
        )
        for signature_id, label in zip(linkage_block.signature_ids, labels, strict=True):
            assignments[str(signature_id)] = f"{block_key}_{label}"
    return assignments


def _signature_to_subblock(selected_subblocks: dict[str, list[str]]) -> dict[str, str]:
    """Return signature id to selected subblock key."""

    return {
        str(signature_id): str(block_key)
        for block_key, signature_ids in selected_subblocks.items()
        for signature_id in signature_ids
    }


def _score_one_scope(
    scored: pd.DataFrame,
    weight_column: str,
    scope_type: str,
    scope_value: str,
    eps: float,
) -> dict[str, Any]:
    """Score one grouped scope for a single weighting scheme."""

    weights = scored[weight_column].astype(float)
    assigned = scored["_assigned"].astype(bool)
    positives = scored["label"].astype(int) == 1
    negatives = scored["label"].astype(int) == 0
    same = scored["_same_cluster"].astype(bool)
    total_weight = float(weights[assigned].sum())
    pos_weight = float(weights[assigned & positives].sum())
    neg_weight = float(weights[assigned & negatives].sum())
    pos_same_weight = float(weights[assigned & positives & same].sum())
    neg_separate_weight = float(weights[assigned & negatives & ~same].sum())
    correct_weight = pos_same_weight + neg_separate_weight
    return {
        "eps": eps,
        "scope_type": scope_type,
        "scope_value": scope_value,
        "weight": weight_column,
        "rows": int(len(scored)),
        "assigned_rows": int(assigned.sum()),
        "missing_assignment_rows": int((~assigned).sum()),
        "positive_rows": int((assigned & positives).sum()),
        "negative_rows": int((assigned & negatives).sum()),
        "total_weight": total_weight,
        "positive_weight": pos_weight,
        "negative_weight": neg_weight,
        "accuracy": correct_weight / total_weight if total_weight else None,
        "positive_recall": pos_same_weight / pos_weight if pos_weight else None,
        "false_split_rate": 1.0 - (pos_same_weight / pos_weight) if pos_weight else None,
        "negative_separation": neg_separate_weight / neg_weight if neg_weight else None,
        "false_merge_rate": 1.0 - (neg_separate_weight / neg_weight) if neg_weight else None,
    }


def _score_assignments(
    gold: pd.DataFrame,
    assignments: dict[str, str],
    signature_to_subblock: dict[str, str],
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score pair constraints against predicted cluster assignments."""

    scored = gold.copy()
    scored["_query_cluster"] = scored["query_signature_id"].map(assignments)
    scored["_member_cluster"] = scored["member_signature_id"].map(assignments)
    scored["_query_subblock"] = scored["query_signature_id"].map(signature_to_subblock)
    scored["_member_subblock"] = scored["member_signature_id"].map(signature_to_subblock)
    scored["_assigned"] = (
        scored["_query_cluster"].notna()
        & scored["_member_cluster"].notna()
        & scored["_query_subblock"].notna()
        & scored["_member_subblock"].notna()
    )
    scored["_same_subblock"] = scored["_assigned"] & (scored["_query_subblock"] == scored["_member_subblock"])
    scored["_same_cluster"] = scored["_assigned"] & (scored["_query_cluster"] == scored["_member_cluster"])
    scoreable = scored[scored["_same_subblock"]].copy()
    rows: list[dict[str, Any]] = []
    scope_specs = [
        ("overall", None),
        ("split", "split"),
        ("table_name", "table_name"),
        ("source_key", "source_key"),
        ("supervision_type", "supervision_type"),
        ("query_view", "query_view"),
    ]
    for weight_column in WEIGHT_COLUMNS:
        for scope_type, column in scope_specs:
            if column is None:
                rows.append(_score_one_scope(scoreable, weight_column, scope_type, "all", eps))
                continue
            if column not in scoreable.columns:
                continue
            for value, group in scoreable.groupby(column, sort=True):
                rows.append(_score_one_scope(group, weight_column, scope_type, str(value), eps))

    boundary = scored[scored["_assigned"] & ~scored["_same_subblock"]].copy()
    boundary_rows = []
    for weight_column in WEIGHT_COLUMNS:
        weights = boundary[weight_column].astype(float)
        positives = boundary["label"].astype(int) == 1
        negatives = boundary["label"].astype(int) == 0
        boundary_rows.append(
            {
                "eps": eps,
                "weight": weight_column,
                "boundary_rows": int(len(boundary)),
                "boundary_positive_rows": int(positives.sum()),
                "boundary_negative_rows": int(negatives.sum()),
                "boundary_weight": float(weights.sum()),
                "boundary_positive_weight": float(weights[positives].sum()),
                "boundary_negative_weight": float(weights[negatives].sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boundary_rows)


def _format_float_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Format float metrics for TSV outputs."""

    output = frame.copy()
    for column in [
        "accuracy",
        "positive_recall",
        "false_split_rate",
        "negative_separation",
        "false_merge_rate",
        "total_weight",
        "positive_weight",
        "negative_weight",
    ]:
        if column in output.columns:
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.8f}")
    return output


def _loggable_metric(value: Any) -> float:
    """Return a numeric metric value for logging."""

    return float("nan") if pd.isna(value) else float(value)


def _configure_runtime_environment(args: argparse.Namespace) -> None:
    """Configure backend/thread environment variables."""

    os.environ["S2AND_BACKEND"] = args.backend
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    os.environ["RAYON_NUM_THREADS"] = str(args.n_jobs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run one dataset EPS sweep and return the summary payload."""

    _validate_args(args)
    _configure_runtime_environment(args)

    output_dir = _output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "eps_sweep.log", encoding="utf-8"),
        ],
    )

    from s2and.production_model import load_production_model

    arrow_paths = load_arrow_paths(args.arrow_root, args.dataset)
    gold = _load_gold(_gold_path(args))
    gold = gold[gold["dataset"].astype(str) == str(args.dataset)].copy()
    if gold.empty:
        raise ValueError(f"No gold rows for dataset={args.dataset}")

    clusterer = load_production_model(args.model_path)
    clusterer.use_cache = False
    clusterer.n_jobs = int(args.n_jobs)
    clusterer.suppress_orcid = bool(args.suppress_orcid_constraints)
    model_eps = None
    best_params = getattr(clusterer, "best_params", None)
    if isinstance(best_params, Mapping) and "eps" in best_params:
        model_eps = float(best_params["eps"])

    target_signature_ids = set(gold["query_signature_id"].astype(str)) | set(gold["member_signature_id"].astype(str))
    planning_state = _read_arrow_signatures_for_planning(args, arrow_paths, target_signature_ids)
    selected_subblocks, plan = _plan_subblocks(args, clusterer, planning_state, gold, arrow_paths)
    plan["model_eps"] = model_eps
    plan["eps_values"] = _eps_values(args, model_eps)
    plan["gold_path"] = str(_gold_path(args).resolve())
    plan["model_path"] = str(args.model_path.resolve())
    plan["bundle_root"] = str(args.bundle_root.resolve())
    plan["arrow_root"] = str(args.arrow_root.resolve())
    plan["arrow_manifest"] = arrow_paths["manifest"]
    plan["distance_source"] = "arrow-rust"
    plan["planning_source"] = "Arrow signatures with Arrow graph subblocking"
    write_json(output_dir / "plan.json", plan)
    pd.DataFrame(plan["subblocks"]).to_csv(output_dir / "target_subblocks.tsv", sep="\t", index=False)

    if args.dry_run:
        return {"plan": plan, "outputs": {"plan": str(output_dir / "plan.json")}}

    cache_dir = output_dir / "distance_caches"
    cache_rows = _ensure_distance_caches(args, clusterer, selected_subblocks, cache_dir, arrow_paths)
    pd.DataFrame(cache_rows).to_csv(output_dir / "distance_caches.tsv", sep="\t", index=False)
    linkage_blocks, linkage_rows = _build_linkage_blocks(
        args=args,
        clusterer=clusterer,
        selected_subblocks=selected_subblocks,
        cache_dir=cache_dir,
        arrow_paths=arrow_paths,
    )
    pd.DataFrame(linkage_rows).to_csv(output_dir / "linkage_builds.tsv", sep="\t", index=False)

    signature_to_subblock = _signature_to_subblock(selected_subblocks)
    score_frames: list[pd.DataFrame] = []
    boundary_frames: list[pd.DataFrame] = []
    assignment_dir = output_dir / "assignments"
    if args.write_assignments:
        assignment_dir.mkdir(exist_ok=True)
    for eps in _eps_values(args, model_eps):
        started = time.perf_counter()
        assignments = _cluster_assignments_for_eps(
            eps=eps,
            linkage_blocks=linkage_blocks,
        )
        seconds = time.perf_counter() - started
        scores, boundary_scores = _score_assignments(gold, assignments, signature_to_subblock, eps)
        scores["cluster_seconds"] = round(seconds, 6)
        boundary_scores["cluster_seconds"] = round(seconds, 6)
        score_frames.append(scores)
        boundary_frames.append(boundary_scores)
        if args.write_assignments:
            assignment_path = assignment_dir / f"assignments_eps_{str(eps).replace('.', 'p')}.parquet"
            pd.DataFrame(
                {"signature_id": list(assignments.keys()), "cluster_id": list(assignments.values())}
            ).to_parquet(assignment_path, index=False)
        overall = scores[
            (scores["scope_type"] == "overall")
            & (scores["scope_value"] == "all")
            & (scores["weight"] == "weight_query_class_balanced")
        ].iloc[0]
        logging.info(
            "eps=%s rows=%d pos_recall=%.6f neg_sep=%.6f false_merge=%.6f seconds=%.3f",
            eps,
            int(overall["assigned_rows"]),
            _loggable_metric(overall["positive_recall"]),
            _loggable_metric(overall["negative_separation"]),
            _loggable_metric(overall["false_merge_rate"]),
            seconds,
        )

    scores_all = pd.concat(score_frames, ignore_index=True)
    boundary_all = pd.concat(boundary_frames, ignore_index=True)
    overall = scores_all[(scores_all["scope_type"] == "overall") & (scores_all["scope_value"] == "all")].copy()
    scores_all.to_parquet(output_dir / "eps_scores.parquet", index=False)
    boundary_all.to_parquet(output_dir / "eps_boundary_pairs.parquet", index=False)
    overall.to_parquet(output_dir / "eps_overall.parquet", index=False)
    _format_float_columns(scores_all).to_csv(output_dir / "eps_scores.tsv", sep="\t", index=False)
    _format_float_columns(boundary_all).to_csv(output_dir / "eps_boundary_pairs.tsv", sep="\t", index=False)
    _format_float_columns(overall).to_csv(output_dir / "eps_overall.tsv", sep="\t", index=False)

    summary = {
        "plan": plan,
        "distance_cache_rows": cache_rows,
        "linkage_rows": linkage_rows,
        "outputs": {
            "plan": str(output_dir / "plan.json"),
            "target_subblocks": str(output_dir / "target_subblocks.tsv"),
            "distance_caches": str(output_dir / "distance_caches.tsv"),
            "linkage_builds": str(output_dir / "linkage_builds.tsv"),
            "eps_scores": str(output_dir / "eps_scores.tsv"),
            "eps_boundary_pairs": str(output_dir / "eps_boundary_pairs.tsv"),
            "eps_overall": str(output_dir / "eps_overall.tsv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _compact_stdout_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact payload for CLI stdout."""

    plan = summary.get("plan", {})
    outputs = summary.get("outputs", {})
    if not isinstance(plan, Mapping):
        return dict(summary)
    compact: dict[str, Any] = {
        "dataset": plan.get("dataset"),
        "distance_source": plan.get("distance_source"),
        "arrow_manifest": plan.get("arrow_manifest"),
        "selected_subblock_count": plan.get("selected_subblock_count"),
        "selected_signature_count": plan.get("selected_signature_count"),
        "selected_pair_count": plan.get("selected_pair_count"),
        "eps_values": plan.get("eps_values"),
        "outputs": outputs,
    }
    subblocks = plan.get("subblocks")
    if isinstance(subblocks, list):
        compact["subblocks"] = subblocks
    if "distance_cache_rows" in summary:
        compact["distance_cache_rows"] = summary["distance_cache_rows"]
    return compact


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweep CLI."""

    summary = run(parse_args(argv))
    print(json.dumps(_compact_stdout_summary(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
