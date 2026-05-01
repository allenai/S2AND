"""Build and reuse persisted candidate-level datasets for the single-letter reranker."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scripts.eval_cluster_retrieval as retrieval
    from scripts.giant_block_cluster_retrieval_task import (
        DEFAULT_MODEL_PATH,
        _read_json,
        _required_file,
        load_clusterer,
    )
    from scripts.giant_block_cluster_retrieval_task import load_dataset as load_giant_block_dataset
    from scripts.reranker_dataset.raw_similarity import (
        RawSimilarityFeatureCache as _RawSimilarityFeatureCache,
    )
    from scripts.reranker_dataset.raw_similarity import (
        raw_similarity_features_by_component as _raw_similarity_features_by_component,
    )
    from scripts.reranker_dataset.rows import generate_candidate_rows
    from scripts.single_letter_reranker_utils import (
        DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
        DEFAULT_LABELED_DATASETS,
        DEFAULT_QUERY_VIEWS,
        DEFAULT_RETRIEVAL_APPROACH,
        DEFAULT_RETRIEVAL_WINDOW_SIZE,
        RETRIEVAL_ENGINE_CHOICES,
        QueryClusterStatsRequest,
        RerankerQueryCase,
        append_query_group_metadata_csv,
        append_rows_csv,
        block_size_bucket,
        build_component_summaries,
        build_labeled_query_cases,
        build_labeled_retrieval_subblock_index,
        build_retrieval_window,
        component_size_bucket,
        compute_query_cluster_stats_batched,
        configure_runtime_environment,
        group_rows,
        load_labeled_dataset,
        load_retrieval_subblock_index,
        materialize_derived_rows,
        read_query_group_metadata_csv,
        read_rows_csv,
        seed_constraint_bypass_component_keys,
        summarize_dataset_rows,
        summarize_query_group_rows,
        write_json,
        write_materialized_rows_csv,
        write_query_group_metadata_csv,
        write_rows_csv,
    )
    from scripts.single_letter_retrieval_utils import (
        FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY,
        FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME,
        build_rust_hybrid_centroid_retriever,
        build_seed_summaries,
        invert_signature_to_cluster_id,
        load_preferred_signature_to_cluster_id,
        select_query_ids,
    )
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore
    from giant_block_cluster_retrieval_task import (  # type: ignore
        DEFAULT_MODEL_PATH,
        _read_json,
        _required_file,
        load_clusterer,
    )
    from giant_block_cluster_retrieval_task import load_dataset as load_giant_block_dataset  # type: ignore
    from reranker_dataset.raw_similarity import (  # type: ignore
        RawSimilarityFeatureCache as _RawSimilarityFeatureCache,
    )
    from reranker_dataset.raw_similarity import (  # type: ignore
        raw_similarity_features_by_component as _raw_similarity_features_by_component,
    )
    from reranker_dataset.rows import generate_candidate_rows  # type: ignore
    from single_letter_reranker_utils import (  # type: ignore
        DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
        DEFAULT_LABELED_DATASETS,
        DEFAULT_QUERY_VIEWS,
        DEFAULT_RETRIEVAL_APPROACH,
        DEFAULT_RETRIEVAL_WINDOW_SIZE,
        RETRIEVAL_ENGINE_CHOICES,
        QueryClusterStatsRequest,
        RerankerQueryCase,
        append_query_group_metadata_csv,
        append_rows_csv,
        block_size_bucket,
        build_component_summaries,
        build_labeled_query_cases,
        build_labeled_retrieval_subblock_index,
        build_retrieval_window,
        component_size_bucket,
        compute_query_cluster_stats_batched,
        configure_runtime_environment,
        group_rows,
        load_labeled_dataset,
        load_retrieval_subblock_index,
        materialize_derived_rows,
        read_query_group_metadata_csv,
        read_rows_csv,
        seed_constraint_bypass_component_keys,
        summarize_dataset_rows,
        summarize_query_group_rows,
        write_json,
        write_materialized_rows_csv,
        write_query_group_metadata_csv,
        write_rows_csv,
    )
    from single_letter_retrieval_utils import (  # type: ignore
        FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY,
        FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME,
        build_rust_hybrid_centroid_retriever,
        build_seed_summaries,
        invert_signature_to_cluster_id,
        load_preferred_signature_to_cluster_id,
        select_query_ids,
    )

from s2and.feature_port import inspect_json_ingest_name_counts_source
from s2and.model import _apply_dataset_name_count_semantics_for_prediction, _build_incremental_constraint_backend
from s2and.runtime import build_runtime_context

ORCID_PATTERN = re.compile(r"(\d{4}-?\d{4}-?\d{4}-?[\dXx]{4})")
DEFAULT_QUERY_BATCH_PAIR_LIMIT = 200_000
GIANT_BLOCK_QUERY_SOURCE_CHOICES = ("supported_single_letter", "orcid_any_input")
PARTIAL_ROWS_FILENAME = "rows.partial.csv"
PARTIAL_QUERY_GROUPS_FILENAME = "query_groups.partial.csv"
BUILD_PROGRESS_FILENAME = "progress.json"
BUILD_DONE_FILENAME = "done.json"


def _normalize_orcid(orcid: str | None) -> str | None:
    """Normalize ORCID to the compact uppercase 16-character form."""

    if not orcid:
        return None
    matches = ORCID_PATTERN.findall(str(orcid))
    if not matches:
        return None
    return matches[0].upper().replace("-", "")


def extract_signature_orcid(signature_payload: dict[str, Any]) -> str | None:
    """Return the normalized ORCID for a raw extracted signature payload."""

    author_info = signature_payload.get("author_info", {})
    if str(author_info.get("source_id_source", "")) != "ORCID":
        return None
    source_ids = author_info.get("source_ids") or []
    if len(source_ids) == 0:
        return None
    return _normalize_orcid(str(source_ids[0]))


def load_query_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load the supported-query metadata payload keyed by query ID."""

    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    rows = payload.get("query_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid query-set payload at {path}: expected list under 'query_rows'")
    required_keys = {"query_id", "_audit_normalized_orcid", "_audit_orcid_group_size", "query_subblock_key"}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Invalid query-set payload at {path}: query_rows[{index}] is not an object")
        missing_keys = sorted(required_keys - set(row))
        if missing_keys:
            raise RuntimeError(
                f"Invalid query-set payload at {path}: query_rows[{index}] missing required keys {missing_keys}"
            )
        metadata_by_id[str(row["query_id"])] = dict(row)
    return metadata_by_id


def build_orcid_seed_cluster_counts(
    *,
    raw_signatures: dict[str, Any],
    signature_to_cluster_id: dict[str, str],
) -> dict[str, Counter[str]]:
    """Count supported seed-cluster memberships by normalized ORCID."""

    counts_by_orcid: dict[str, Counter[str]] = defaultdict(Counter)
    for signature_id, cluster_id in signature_to_cluster_id.items():
        signature_payload = raw_signatures.get(str(signature_id))
        if not isinstance(signature_payload, dict):
            continue
        normalized_orcid = extract_signature_orcid(signature_payload)
        if normalized_orcid is None:
            continue
        counts_by_orcid[normalized_orcid][str(cluster_id)] += 1
    return counts_by_orcid


@dataclass(frozen=True)
class PreparedViewPayload:
    """One masked query-view payload ready for row materialization."""

    query_view: str
    query: Any
    shortlist_component_keys: tuple[str, ...]
    retrieval_scores: dict[str, float]
    retrieval_ranks: dict[str, int]
    retrieval_window_state: dict[str, int]


@dataclass(frozen=True)
class PreparedQueryRowsRequest:
    """Prepared query-window request that can be scored in a batched pass."""

    query_case: RerankerQueryCase
    block_component_count: int
    view_payloads: tuple[PreparedViewPayload, ...]
    union_summary_by_component: dict[str, retrieval.ClusterSummary]
    retrieval_window_state_base: dict[str, int]
    stats_request: QueryClusterStatsRequest
    estimated_pair_count: int
    raw_similarity_features_by_component: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class _QueryGroupSummaryAccumulator:
    """Streaming summary state for one query-group metadata output."""

    query_groups: int = 0
    row_count: int = 0
    positive_rows: int = 0
    dropped_all_negative_group_count: int = 0
    candidate_counts: list[int] = field(default_factory=list)
    best_positive_retrieval_ranks: list[int] = field(default_factory=list)
    per_supervision_type: dict[str, dict[str, int]] = field(default_factory=dict)
    per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    per_view: dict[str, dict[str, int]] = field(default_factory=dict)

    def _update_bucket(
        self,
        bucket: dict[str, dict[str, int]],
        key: str,
        *,
        candidate_count: int,
        positive_candidate_count: int,
        group_has_positive: bool,
        include_negative_count: bool = False,
    ) -> None:
        state = bucket.setdefault(
            str(key),
            {
                "query_groups": 0,
                "row_count": 0,
                "positive_rows": 0,
                "dropped_all_negative_group_count": 0,
            },
        )
        state["query_groups"] += 1
        state["row_count"] += int(candidate_count)
        state["positive_rows"] += int(positive_candidate_count)
        if include_negative_count and not group_has_positive:
            state["dropped_all_negative_group_count"] += 1

    def update(self, metadata_row: dict[str, Any]) -> None:
        """Fold one persisted query-group metadata row into the summary."""

        candidate_count = int(metadata_row["candidate_count"])
        positive_candidate_count = int(metadata_row["positive_candidate_count"])
        group_has_positive = bool(int(metadata_row["group_has_positive"]))
        self.query_groups += 1
        self.row_count += int(candidate_count)
        self.positive_rows += int(positive_candidate_count)
        self.dropped_all_negative_group_count += int(not group_has_positive)
        self.candidate_counts.append(int(candidate_count))
        if metadata_row["best_positive_retrieval_rank"] is not None:
            self.best_positive_retrieval_ranks.append(int(metadata_row["best_positive_retrieval_rank"]))
        self._update_bucket(
            self.per_supervision_type,
            str(metadata_row.get("supervision_type", "labeled")),
            candidate_count=candidate_count,
            positive_candidate_count=positive_candidate_count,
            group_has_positive=group_has_positive,
        )
        self._update_bucket(
            self.per_split,
            str(metadata_row.get("split", "all")),
            candidate_count=candidate_count,
            positive_candidate_count=positive_candidate_count,
            group_has_positive=group_has_positive,
        )
        self._update_bucket(
            self.per_view,
            str(metadata_row["query_view"]),
            candidate_count=candidate_count,
            positive_candidate_count=positive_candidate_count,
            group_has_positive=group_has_positive,
            include_negative_count=True,
        )

    def to_summary(self) -> dict[str, Any]:
        """Return the persisted dataset summary from the accumulated metadata."""

        summary = {
            "query_groups": int(self.query_groups),
            "row_count": int(self.row_count),
            "positive_rows": int(self.positive_rows),
            "positive_rate": round(float(self.positive_rows / max(1, self.row_count)), 6),
            "dropped_all_negative_group_count": int(self.dropped_all_negative_group_count),
            "candidate_window_coverage": round(
                float((self.query_groups - self.dropped_all_negative_group_count) / max(1, self.query_groups)),
                6,
            ),
            "candidate_count_mean": round(float(statistics.mean(self.candidate_counts)), 6)
            if self.candidate_counts
            else 0.0,
            "candidate_count_median": round(float(statistics.median(self.candidate_counts)), 6)
            if self.candidate_counts
            else 0.0,
            "best_positive_retrieval_rank_mean": round(float(statistics.mean(self.best_positive_retrieval_ranks)), 6)
            if self.best_positive_retrieval_ranks
            else None,
            "per_supervision_type": {},
            "per_split": {},
            "per_view": {},
        }
        for supervision_type, bucket in sorted(self.per_supervision_type.items()):
            summary["per_supervision_type"][str(supervision_type)] = {
                "query_groups": int(bucket["query_groups"]),
                "row_count": int(bucket["row_count"]),
                "positive_rows": int(bucket["positive_rows"]),
            }
        for split, bucket in sorted(self.per_split.items()):
            summary["per_split"][str(split)] = {
                "query_groups": int(bucket["query_groups"]),
                "row_count": int(bucket["row_count"]),
                "positive_rows": int(bucket["positive_rows"]),
            }
        for query_view, bucket in sorted(self.per_view.items()):
            summary["per_view"][str(query_view)] = {
                "query_groups": int(bucket["query_groups"]),
                "row_count": int(bucket["row_count"]),
                "positive_rows": int(bucket["positive_rows"]),
                "positive_rate": round(float(bucket["positive_rows"] / max(1, bucket["row_count"])), 6),
                "dropped_all_negative_group_count": int(bucket["dropped_all_negative_group_count"]),
                "candidate_window_coverage": round(
                    float(
                        (bucket["query_groups"] - bucket["dropped_all_negative_group_count"])
                        / max(1, bucket["query_groups"])
                    ),
                    6,
                ),
            }
        return summary


def _summarize_query_group_metadata_rows(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a dataset from query-group metadata rows."""

    accumulator = _QueryGroupSummaryAccumulator()
    for metadata_row in metadata_rows:
        accumulator.update(metadata_row)
    return accumulator.to_summary()


def _utc_timestamp() -> str:
    """Return an ISO-style UTC timestamp."""

    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written watcher reads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _current_rss_bytes() -> int | None:
    """Return the current process RSS when available."""

    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _normalize_giant_block_dataset_label(value: str) -> str:
    """Return a stable dataset label for one giant-block build."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if not normalized:
        raise ValueError(f"Could not derive a giant-block dataset label from {value!r}")
    return normalized


def _reset_giant_block_stream_outputs(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Reset managed streaming output files for one giant-block build."""

    rows_partial_path = output_dir / PARTIAL_ROWS_FILENAME
    query_groups_partial_path = output_dir / PARTIAL_QUERY_GROUPS_FILENAME
    progress_path = output_dir / BUILD_PROGRESS_FILENAME
    done_path = output_dir / BUILD_DONE_FILENAME
    managed_paths = [
        output_dir / "rows.csv",
        output_dir / "query_groups.csv",
        output_dir / "summary.json",
        output_dir / "run_summary.json",
        rows_partial_path,
        query_groups_partial_path,
        progress_path,
        done_path,
    ]
    for managed_path in managed_paths:
        if managed_path.exists():
            managed_path.unlink()
    write_rows_csv(rows_partial_path, [])
    write_query_group_metadata_csv(query_groups_partial_path, [])
    return rows_partial_path, query_groups_partial_path, progress_path, done_path


def _write_giant_block_progress(
    progress_path: Path,
    *,
    status: str,
    query_source: str,
    total_queries: int,
    processed_queries: int,
    written_rows: int,
    written_query_groups: int,
    last_query_id: str | None,
    started_at: str,
) -> None:
    """Persist a lightweight progress heartbeat for the long-running build."""

    _atomic_write_json(
        progress_path,
        {
            "status": str(status),
            "query_source": str(query_source),
            "total_queries": int(total_queries),
            "processed_queries": int(processed_queries),
            "written_rows": int(written_rows),
            "written_query_groups": int(written_query_groups),
            "last_query_id": (str(last_query_id) if last_query_id is not None else None),
            "started_at": str(started_at),
            "updated_at": _utc_timestamp(),
            "rss_bytes": _current_rss_bytes(),
        },
    )


def _require_labeled_name_counts_source(dataset: Any, *, dataset_name: str) -> dict[str, Any]:
    """Fail fast when labeled Rust JSON ingest would drop name-count features."""

    plan = inspect_json_ingest_name_counts_source(dataset)
    name_counts_source = str(plan["name_counts_source"])
    if name_counts_source == "none":
        raise RuntimeError(
            "Labeled reranker dataset build requires Rust name-count features, but JSON ingest "
            f"would use name_counts_source=none for dataset={dataset_name!r}. "
            "Rebuild after fixing name-count loading rather than silently emitting degraded rows."
        )
    return {
        "name_counts_source": name_counts_source,
        "signatures_total": int(plan["signatures_total"]),
        "signatures_with_counts": int(plan["signatures_with_counts"]),
        "artifact_configured": bool(plan["artifact_configured"]),
        "rust_can_overlay_signature_counts": bool(plan["rust_can_overlay_signature_counts"]),
    }


def _read_string_id_file(path: Path) -> set[str]:
    """Read a newline-delimited string-ID file."""

    selected_ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not selected_ids:
        raise ValueError(f"Expected at least one ID in {path}")
    return selected_ids


def _write_string_id_file(path: Path, values: list[str]) -> None:
    """Write a newline-delimited string-ID file with stable ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    unique_values = sorted({str(value) for value in values})
    path.write_text("\n".join(unique_values) + "\n", encoding="utf-8")


def _filter_query_sequence_by_id_set(queries: list[Any], *, selected_query_ids: set[str] | None) -> list[Any]:
    """Filter query-like objects by ``query_id`` with strict unknown-ID validation."""

    if selected_query_ids is None:
        return list(queries)
    query_id_to_query = {str(query.query_id): query for query in queries}
    unknown_query_ids = sorted(selected_query_ids.difference(query_id_to_query))
    if unknown_query_ids:
        raise ValueError(f"Unknown query IDs requested: {unknown_query_ids[:10]}")
    return [query_id_to_query[query_id] for query_id in sorted(selected_query_ids)]


def _load_raw_paper_text_by_id(papers_path: Path, *, needed_paper_ids: set[str]) -> dict[str, str]:
    """Load title-plus-abstract raw text for papers retained by the active dataset."""

    if not needed_paper_ids:
        return {}
    raw_papers = _read_json(papers_path)
    text_by_id: dict[str, str] = {}
    for paper_id in needed_paper_ids:
        paper = raw_papers.get(str(paper_id))
        if not isinstance(paper, dict):
            continue
        text_by_id[str(paper_id)] = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    return text_by_id


def _residual_summary(
    *,
    dataset: Any,
    component_key: str,
    heldout_signature_id: str,
    component_signatures: dict[str, list[str]],
    feature_cache: dict[str, retrieval.QueryFeatures],
    max_exemplars: int,
    cache: dict[tuple[str, str], retrieval.ClusterSummary],
) -> retrieval.ClusterSummary:
    """Return the cached true-component residual summary for one held-out query."""

    cache_key = (str(component_key), str(heldout_signature_id))
    if cache_key in cache:
        return cache[cache_key]
    block_key, cluster_id = component_key.split("::", 1)
    signature_ids = [
        signature_id for signature_id in component_signatures[component_key] if signature_id != heldout_signature_id
    ]
    cache[cache_key] = retrieval.build_cluster_summary(
        dataset=dataset,
        block_key=block_key,
        cluster_id=cluster_id,
        component_key=component_key,
        signature_ids=signature_ids,
        max_exemplars=max_exemplars,
        feature_cache=feature_cache,
        orcid_enabled=False,
    )
    return cache[cache_key]


def _residual_seed_summary(
    *,
    dataset: Any,
    block_key: str,
    cluster_id: str,
    heldout_signature_id: str,
    seed_clusters: dict[str, list[str]],
    feature_cache: dict[str, retrieval.QueryFeatures],
    max_exemplars: int,
    cache: dict[tuple[str, str], retrieval.ClusterSummary],
) -> retrieval.ClusterSummary:
    """Return the cached residual seed summary for one held-out query."""

    cache_key = (str(cluster_id), str(heldout_signature_id))
    if cache_key in cache:
        return cache[cache_key]
    signature_ids = [
        signature_id for signature_id in seed_clusters[str(cluster_id)] if signature_id != heldout_signature_id
    ]
    cache[cache_key] = retrieval.build_cluster_summary(
        dataset=dataset,
        block_key=str(block_key),
        cluster_id=str(cluster_id),
        component_key=str(cluster_id),
        signature_ids=signature_ids,
        max_exemplars=max_exemplars,
        feature_cache=feature_cache,
        orcid_enabled=False,
    )
    return cache[cache_key]


def _orcid_group_size_bucket(orcid_group_size: int) -> str:
    """Bucket ORCID group sizes for cached metadata and split stratification."""

    size = int(orcid_group_size)
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    return "4+"


def _any_input_query_view(base_query: retrieval.QueryFeatures) -> str:
    """Return the natural runtime query view for one giant-block query."""

    return "full" if bool(base_query.has_full_first) else "initial_only"


def _split_orcid_groups_for_giant_block(
    query_metadata_by_id: dict[str, dict[str, Any]],
    *,
    seed: int,
    dev_fraction: float = 0.2,
) -> dict[str, str]:
    """Assign one stable train/dev split per ORCID group."""

    orcid_group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_meta in query_metadata_by_id.values():
        orcid_group_rows[str(query_meta["_audit_normalized_orcid"])].append(dict(query_meta))

    grouped_orcids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for normalized_orcid, rows in orcid_group_rows.items():
        natural_views = {str(row["natural_query_view"]) for row in rows}
        if len(natural_views) == 1:
            view_bucket = next(iter(natural_views))
        else:
            view_bucket = "mixed"
        size_bucket = str(rows[0]["_audit_orcid_group_size_bucket"])
        grouped_orcids[(view_bucket, size_bucket)].append(str(normalized_orcid))

    rng = random.Random(int(seed))
    split_by_orcid: dict[str, str] = {}
    for strat_key, orcids in grouped_orcids.items():
        del strat_key
        ordered_orcids = sorted(orcids)
        rng.shuffle(ordered_orcids)
        dev_count = int(round(len(ordered_orcids) * float(dev_fraction)))
        if len(ordered_orcids) > 1:
            dev_count = max(1, min(len(ordered_orcids) - 1, dev_count))
        else:
            dev_count = 0
        dev_orcids = set(ordered_orcids[:dev_count])
        for normalized_orcid in ordered_orcids:
            split_by_orcid[str(normalized_orcid)] = "dev" if normalized_orcid in dev_orcids else "train"
    return split_by_orcid


def _build_giant_block_any_input_query_cases(
    *,
    dataset_label: str,
    raw_signatures: dict[str, Any],
    dataset: Any,
    target_block: str,
    signature_to_cluster_id: dict[str, str],
    seed_cluster_counts_by_orcid: dict[str, Counter[str]],
    limit_queries: int | None,
    query_id_file: Path | None,
    seed: int,
) -> tuple[list[RerankerQueryCase], dict[str, Any]]:
    """Build weakly supervised giant-block query cases for any-input development."""

    explicit_query_ids = _read_string_id_file(query_id_file) if query_id_file is not None else None
    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    query_cases: list[RerankerQueryCase] = []
    query_metadata_by_id: dict[str, dict[str, Any]] = {}
    orcid_to_signature_ids: dict[str, list[str]] = defaultdict(list)
    for signature_id, signature_payload in raw_signatures.items():
        author_info = signature_payload.get("author_info", {})
        if str(author_info.get("block", "")) != str(target_block):
            continue
        if str(signature_id) not in dataset.signatures:
            continue
        normalized_orcid = extract_signature_orcid(signature_payload)
        if normalized_orcid is None:
            continue
        orcid_to_signature_ids[normalized_orcid].append(str(signature_id))

    for normalized_orcid, signature_ids in sorted(orcid_to_signature_ids.items()):
        signature_ids_sorted = sorted(str(signature_id) for signature_id in signature_ids)
        orcid_group_size = int(len(signature_ids_sorted))
        group_size_bucket = _orcid_group_size_bucket(orcid_group_size)
        for query_id in signature_ids_sorted:
            if explicit_query_ids is not None and str(query_id) not in explicit_query_ids:
                continue
            base_query = retrieval.extract_query_features(
                dataset,
                str(query_id),
                feature_cache=feature_cache,
                orcid_enabled=False,
            )
            natural_query_view = _any_input_query_view(base_query)
            positive_support = Counter(seed_cluster_counts_by_orcid.get(str(normalized_orcid), Counter()))
            query_in_seed_before_holdout = str(query_id) in signature_to_cluster_id
            if query_in_seed_before_holdout:
                cluster_id = str(signature_to_cluster_id[str(query_id)])
                if cluster_id in positive_support:
                    positive_support[cluster_id] -= 1
                    if int(positive_support[cluster_id]) <= 0:
                        del positive_support[cluster_id]
            positive_component_keys = frozenset(
                str(cluster_id) for cluster_id in positive_support if int(positive_support[cluster_id]) > 0
            )
            if orcid_group_size == 1:
                supervision_type = "unlabeled_singleton_orcid"
                support_type = "unlabeled"
            elif positive_component_keys:
                supervision_type = "positive_repeat_orcid"
                support_type = "unique" if len(positive_component_keys) == 1 else "ambiguous"
            else:
                supervision_type = "unresolved_repeat_orcid"
                support_type = "unresolved"
            query_case = RerankerQueryCase(
                source=str(dataset_label),
                dataset=str(dataset_label),
                query_source="orcid_any_input",
                query_id=str(query_id),
                query_signature_id=str(query_id),
                normalized_orcid=str(normalized_orcid),
                orcid_group_size=int(orcid_group_size),
                orcid_group_size_bucket=str(group_size_bucket),
                split="all",
                block_key=str(target_block),
                positive_component_keys=frozenset(positive_component_keys),
                supervision_type=str(supervision_type),
                support_type=str(support_type),
                query_in_seed_before_holdout=bool(query_in_seed_before_holdout),
                natural_query_view=str(natural_query_view),
                block_size=0,
                component_size=0,
                sampling_info_bucket=str(natural_query_view),
            )
            query_cases.append(query_case)
            query_metadata_by_id[str(query_id)] = {
                "_audit_normalized_orcid": str(normalized_orcid),
                "_audit_orcid_group_size": int(orcid_group_size),
                "_audit_orcid_group_size_bucket": str(group_size_bucket),
                "natural_query_view": str(natural_query_view),
            }

    if explicit_query_ids is None:
        selected_query_ids = set(
            select_query_ids(
                [query_case.query_id for query_case in query_cases],
                limit_queries=limit_queries,
                seed=seed,
            )
        )
        query_cases = [query_case for query_case in query_cases if str(query_case.query_id) in selected_query_ids]
        query_metadata_by_id = {
            query_id: metadata for query_id, metadata in query_metadata_by_id.items() if query_id in selected_query_ids
        }

    split_by_orcid = _split_orcid_groups_for_giant_block(query_metadata_by_id, seed=seed)
    query_cases = [
        RerankerQueryCase(
            source=query_case.source,
            dataset=query_case.dataset,
            query_source=query_case.query_source,
            query_id=query_case.query_id,
            query_signature_id=query_case.query_signature_id,
            normalized_orcid=query_case.normalized_orcid,
            orcid_group_size=query_case.orcid_group_size,
            orcid_group_size_bucket=query_case.orcid_group_size_bucket,
            split=str(split_by_orcid.get(str(query_case.normalized_orcid), "train")),
            block_key=query_case.block_key,
            positive_component_keys=query_case.positive_component_keys,
            supervision_type=query_case.supervision_type,
            support_type=query_case.support_type,
            query_in_seed_before_holdout=query_case.query_in_seed_before_holdout,
            natural_query_view=query_case.natural_query_view,
            block_size=query_case.block_size,
            component_size=query_case.component_size,
            sampling_info_bucket=query_case.sampling_info_bucket,
        )
        for query_case in query_cases
    ]
    supervision_counts = Counter(str(query_case.supervision_type) for query_case in query_cases)
    split_counts = Counter(str(query_case.split) for query_case in query_cases)
    natural_view_counts = Counter(str(query_case.natural_query_view) for query_case in query_cases)
    selected_orcid_groups = {str(query_case.normalized_orcid) for query_case in query_cases}
    return sorted(query_cases, key=lambda query_case: str(query_case.query_id)), {
        "query_count": int(len(query_cases)),
        "orcid_group_count": int(len(selected_orcid_groups)),
        "supervision_type_counts": {str(key): int(value) for key, value in sorted(supervision_counts.items())},
        "split_counts": {str(key): int(value) for key, value in sorted(split_counts.items())},
        "natural_query_view_counts": {str(key): int(value) for key, value in sorted(natural_view_counts.items())},
    }


def _build_giant_block_supported_query_cases(
    *,
    dataset_label: str,
    query_metadata: dict[str, dict[str, Any]],
    signature_to_cluster_id: dict[str, str],
    seed_cluster_counts_by_orcid: dict[str, Counter[str]],
    limit_queries: int | None,
    query_id_file: Path | None,
    seed: int,
) -> tuple[list[RerankerQueryCase], dict[str, Any]]:
    """Build supported single-letter giant-block query cases."""

    explicit_query_ids = _read_string_id_file(query_id_file) if query_id_file is not None else None
    query_cases: list[RerankerQueryCase] = []
    support_type_counts: Counter[str] = Counter()

    for query_id, query_meta in sorted(query_metadata.items()):
        normalized_orcid = str(query_meta["_audit_normalized_orcid"])
        positive_support = Counter(seed_cluster_counts_by_orcid.get(normalized_orcid, Counter()))
        query_in_seed_before_holdout = str(query_id) in signature_to_cluster_id
        if query_in_seed_before_holdout:
            cluster_id = str(signature_to_cluster_id[str(query_id)])
            if cluster_id in positive_support:
                positive_support[cluster_id] -= 1
                if int(positive_support[cluster_id]) <= 0:
                    del positive_support[cluster_id]
        positive_component_keys = frozenset(
            str(cluster_id) for cluster_id, count in positive_support.items() if int(count) > 0
        )
        if not positive_component_keys:
            continue
        support_type = "unique" if len(positive_component_keys) == 1 else "ambiguous"
        support_type_counts[support_type] += 1
        query_cases.append(
            RerankerQueryCase(
                source=str(dataset_label),
                dataset=str(dataset_label),
                query_source="supported_single_letter",
                query_id=str(query_id),
                query_signature_id=str(query_id),
                normalized_orcid=normalized_orcid,
                orcid_group_size=int(query_meta["_audit_orcid_group_size"]),
                orcid_group_size_bucket=_orcid_group_size_bucket(int(query_meta["_audit_orcid_group_size"])),
                split="all",
                block_key=str(query_meta["query_subblock_key"]),
                positive_component_keys=positive_component_keys,
                supervision_type="positive_repeat_orcid",
                support_type=str(support_type),
                query_in_seed_before_holdout=bool(query_in_seed_before_holdout),
                natural_query_view="initial_only",
                block_size=0,
                component_size=0,
                sampling_info_bucket="initial_only",
            )
        )

    query_cases = _filter_query_sequence_by_id_set(query_cases, selected_query_ids=explicit_query_ids)
    if explicit_query_ids is None:
        selected_query_ids = set(
            select_query_ids(
                [query_case.query_id for query_case in query_cases],
                limit_queries=limit_queries,
                seed=seed,
            )
        )
        query_cases = [query_case for query_case in query_cases if str(query_case.query_id) in selected_query_ids]
    query_cases.sort(key=lambda query_case: str(query_case.query_id))
    support_type_counts = Counter(str(query_case.support_type) for query_case in query_cases)
    return query_cases, {
        "query_count": int(len(query_cases)),
        "support_type_counts": {str(key): int(value) for key, value in sorted(support_type_counts.items())},
        "natural_query_view_counts": {"initial_only": int(len(query_cases))},
        "supervision_type_counts": {"positive_repeat_orcid": int(len(query_cases))},
    }


def _positive_rank_priority(rank: int | None) -> int:
    """Return the hard-block sampler priority bucket for one positive rank."""

    if rank is None:
        return 0
    if int(rank) <= 3:
        return 4
    if int(rank) <= 10:
        return 3
    if int(rank) <= 25:
        return 2
    if int(rank) <= 50:
        return 1
    return 0


def _hard_blocks_v1_sort_key(metadata_row: dict[str, Any]) -> tuple[Any, ...]:
    """Return the deterministic hard-case priority for one cached query group."""

    return (
        -int(metadata_row["recoverable_non_top1"]),
        -int(metadata_row["cross_family_top1_vs_positive"]),
        -int(_positive_rank_priority(metadata_row["best_positive_retrieval_rank"])),
        -int(metadata_row["candidate_count"]),
        -int(metadata_row["block_component_count"]),
        -int(metadata_row["block_size"]),
        -int(metadata_row["component_size"]),
        str(metadata_row["query_group_id"]),
    )


def _round_robin_query_group_sample(
    metadata_rows: list[dict[str, Any]],
    *,
    limit_query_groups: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply the original round-robin balancing to cached query-group metadata."""

    if limit_query_groups is None or int(limit_query_groups) <= 0 or len(metadata_rows) <= int(limit_query_groups):
        return sorted(metadata_rows, key=lambda row: str(row["query_group_id"]))
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        key = (
            block_size_bucket(int(row["block_size"])),
            component_size_bucket(int(row["component_size"])),
            str(row["sampling_info_bucket"]),
        )
        grouped[key].append(dict(row))
    rng = random.Random(int(seed))
    for values in grouped.values():
        rng.shuffle(values)
    ordered_keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    while len(selected) < int(limit_query_groups):
        progressed = False
        for key in ordered_keys:
            if not grouped[key]:
                continue
            selected.append(grouped[key].pop())
            progressed = True
            if len(selected) >= int(limit_query_groups):
                break
        if not progressed:
            break
    return sorted(selected, key=lambda row: str(row["query_group_id"]))


def _select_query_group_metadata_rows(
    metadata_rows: list[dict[str, Any]],
    *,
    labeled_query_sampler: str,
    limit_query_groups: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select cached query groups using the requested sampler."""

    if labeled_query_sampler == "round_robin_v1":
        selected_rows = _round_robin_query_group_sample(
            metadata_rows,
            limit_query_groups=limit_query_groups,
            seed=seed,
        )
        return selected_rows, {
            "profile": "round_robin_v1",
            "requested_limit_query_groups": (int(limit_query_groups) if limit_query_groups is not None else None),
            "selected_query_group_count": int(len(selected_rows)),
        }

    filtered_rows: list[dict[str, Any]] = []
    filtered_small_block_count = 0
    filtered_small_component_set_count = 0
    filtered_single_candidate_preview_count = 0
    for row in metadata_rows:
        if int(row["block_size"]) < 10:
            filtered_small_block_count += 1
            continue
        if int(row["block_component_count"]) < 2:
            filtered_small_component_set_count += 1
            continue
        if int(row["candidate_count"]) < 2:
            filtered_single_candidate_preview_count += 1
            continue
        filtered_rows.append(dict(row))
    if not filtered_rows:
        raise RuntimeError("Hard-case labeled sampler removed every eligible cached query group")
    ordered_rows = sorted(filtered_rows, key=_hard_blocks_v1_sort_key)
    if limit_query_groups is not None and int(limit_query_groups) > 0:
        ordered_rows = ordered_rows[: int(limit_query_groups)]
    selected_positive_buckets = Counter(str(row["best_positive_rank_bucket"]) for row in ordered_rows)
    return ordered_rows, {
        "profile": "hard_blocks_v1",
        "requested_limit_query_groups": (int(limit_query_groups) if limit_query_groups is not None else None),
        "cases_before_filtering": int(len(metadata_rows)),
        "cases_after_filtering": int(len(filtered_rows)),
        "selected_query_group_count": int(len(ordered_rows)),
        "filtered_small_block_count": int(filtered_small_block_count),
        "filtered_single_component_block_count": int(filtered_small_component_set_count),
        "filtered_single_candidate_preview_count": int(filtered_single_candidate_preview_count),
        "selected_recoverable_non_top1_count": int(sum(int(row["recoverable_non_top1"]) for row in ordered_rows)),
        "selected_cross_family_top1_vs_positive_count": int(
            sum(int(row["cross_family_top1_vs_positive"]) for row in ordered_rows)
        ),
        "selected_candidate_count_mean": round(
            float(statistics.mean(int(row["candidate_count"]) for row in ordered_rows)),
            6,
        ),
        "selected_block_component_count_mean": round(
            float(statistics.mean(int(row["block_component_count"]) for row in ordered_rows)),
            6,
        ),
        "selected_positive_rank_buckets": {
            str(bucket): int(count) for bucket, count in sorted(selected_positive_buckets.items())
        },
    }


def _prepare_query_rows_request(
    *,
    dataset: Any,
    query_case: RerankerQueryCase,
    block_component_count: int,
    base_query: retrieval.QueryFeatures,
    query_views: list[str],
    raw_candidate_summaries: list[retrieval.ClusterSummary],
    summary_by_component: dict[str, retrieval.ClusterSummary],
    candidate_signature_ids_by_component: dict[str, list[str]],
    retrieval_approach: str,
    retrieval_engine: str,
    window_size: int,
    retrieval_window_state_base: dict[str, int] | None = None,
    rust_hybrid_centroid_retriever: Any | None = None,
    frozen_rust_hybrid_centroid_policy: Any | None = None,
    retrieval_subblock_index: dict[str, Any] | None = None,
    raw_paper_text_by_id: dict[str, str] | None = None,
    raw_similarity_feature_cache: _RawSimilarityFeatureCache | None = None,
) -> PreparedQueryRowsRequest:
    """Prepare one query request for later batched pairwise scoring."""

    max_block_component_size = max((summary.size for summary in raw_candidate_summaries), default=0)
    view_payloads: list[PreparedViewPayload] = []
    union_component_keys: set[str] = set()
    for query_view in query_views:
        query = retrieval.mask_query_features(base_query, query_view, orcid_enabled=False)
        ranked_component_keys, retrieval_scores, retrieval_ranks, retrieval_window_state = build_retrieval_window(
            query=query,
            raw_candidate_summaries=raw_candidate_summaries,
            max_block_component_size=max_block_component_size,
            retrieval_approach=retrieval_approach,
            retrieval_engine=retrieval_engine,
            max_ranked_clusters=window_size,
            rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
            frozen_rust_hybrid_centroid_policy=frozen_rust_hybrid_centroid_policy,
            query_signature_id=str(query_case.query_signature_id),
            retrieval_subblock_index=retrieval_subblock_index,
        )
        shortlist_component_keys = tuple(str(component_key) for component_key in ranked_component_keys[:window_size])
        union_component_keys.update(shortlist_component_keys)
        view_payloads.append(
            PreparedViewPayload(
                query_view=str(query_view),
                query=query,
                shortlist_component_keys=shortlist_component_keys,
                retrieval_scores={str(key): float(value) for key, value in retrieval_scores.items()},
                retrieval_ranks={str(key): int(value) for key, value in retrieval_ranks.items()},
                retrieval_window_state={str(key): int(value) for key, value in retrieval_window_state.items()},
            )
        )

    union_component_keys_ordered = tuple(
        sorted(
            union_component_keys,
            key=lambda component_key: (
                min(payload.retrieval_ranks.get(component_key, window_size + 1) for payload in view_payloads),
                str(component_key),
            ),
        )
    )
    union_summary_by_component = {
        component_key: summary_by_component[component_key] for component_key in union_component_keys_ordered
    }
    union_signature_ids_by_component = {
        component_key: [str(signature_id) for signature_id in candidate_signature_ids_by_component[component_key]]
        for component_key in union_component_keys_ordered
    }
    union_retrieval_ranks = {
        component_key: min(payload.retrieval_ranks.get(component_key, window_size + 1) for payload in view_payloads)
        for component_key in union_component_keys_ordered
    }
    union_retrieval_scores = {
        component_key: max(
            float(payload.retrieval_scores.get(component_key, float("-inf"))) for payload in view_payloads
        )
        for component_key in union_component_keys_ordered
    }
    seed_bypass_component_keys = seed_constraint_bypass_component_keys(
        dataset=dataset,
        query_case=query_case,
        candidate_signature_ids_by_component=union_signature_ids_by_component,
    )
    estimated_pair_count = sum(len(signature_ids) for signature_ids in union_signature_ids_by_component.values())
    raw_similarity_features_by_component = _raw_similarity_features_by_component(
        dataset=dataset,
        query_signature_id=str(query_case.query_signature_id),
        candidate_signature_ids_by_component=union_signature_ids_by_component,
        raw_paper_text_by_id=raw_paper_text_by_id,
        cache=raw_similarity_feature_cache,
    )
    return PreparedQueryRowsRequest(
        query_case=query_case,
        block_component_count=int(block_component_count),
        view_payloads=tuple(view_payloads),
        union_summary_by_component=union_summary_by_component,
        retrieval_window_state_base=dict(retrieval_window_state_base or {}),
        stats_request=QueryClusterStatsRequest(
            query_signature_id=str(query_case.query_signature_id),
            shortlist_component_keys=union_component_keys_ordered,
            candidate_signature_ids_by_component=union_signature_ids_by_component,
            retrieval_ranks=union_retrieval_ranks,
            retrieval_scores=union_retrieval_scores,
            summary_by_component=union_summary_by_component,
            incremental_dont_use_cluster_seeds_component_keys=seed_bypass_component_keys,
            ignore_disallow_constraints_component_keys=seed_bypass_component_keys,
        ),
        estimated_pair_count=int(estimated_pair_count),
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )


def _materialize_query_rows_from_prepared(
    prepared_request: PreparedQueryRowsRequest,
    *,
    stats_by_component: dict[str, Any],
    rust_hybrid_centroid_retriever: Any | None = None,
) -> list[dict[str, Any]]:
    """Materialize candidate rows for one prepared query after batch scoring."""

    rows: list[dict[str, Any]] = []
    retrieval_state_prefix = dict(prepared_request.retrieval_window_state_base)
    for payload in prepared_request.view_payloads:
        shortlist_component_keys = list(payload.shortlist_component_keys)
        shortlist_summary_by_component = {
            component_key: prepared_request.union_summary_by_component[component_key]
            for component_key in shortlist_component_keys
        }
        shortlist_stats_by_component = {
            component_key: stats_by_component[component_key] for component_key in shortlist_component_keys
        }
        shortlist_raw_similarity_features_by_component = {
            component_key: prepared_request.raw_similarity_features_by_component[component_key]
            for component_key in shortlist_component_keys
            if component_key in prepared_request.raw_similarity_features_by_component
        }
        retrieval_window_state = dict(retrieval_state_prefix)
        retrieval_window_state.update(payload.retrieval_window_state)
        rows.extend(
            generate_candidate_rows(
                query_case=prepared_request.query_case,
                query_view=str(payload.query_view),
                query_features=payload.query,
                shortlist_component_keys=shortlist_component_keys,
                retrieval_scores=payload.retrieval_scores,
                retrieval_ranks=payload.retrieval_ranks,
                retrieval_window_state=retrieval_window_state,
                summary_by_component=shortlist_summary_by_component,
                stats_by_component=shortlist_stats_by_component,
                rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
                raw_similarity_features_by_component=shortlist_raw_similarity_features_by_component,
            )
        )
    return rows


def _flush_prepared_query_requests(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    prepared_requests: list[PreparedQueryRowsRequest],
    pair_batch_size: int,
    max_top_k: int,
    pair_counts: list[int],
    featurize_seconds: list[float],
    model_seconds: list[float],
    rust_hybrid_centroid_retriever: Any | None = None,
    rows: list[dict[str, Any]] | None = None,
    query_group_metadata_rows: list[dict[str, Any]] | None = None,
    rows_output_path: Path | None = None,
    query_group_metadata_output_path: Path | None = None,
    query_group_summary_accumulator: _QueryGroupSummaryAccumulator | None = None,
    min_candidates_per_query_group: int = 1,
    filtered_query_group_stats: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Score and materialize one prepared-query batch."""

    if not prepared_requests:
        return 0, 0
    if int(min_candidates_per_query_group) <= 0:
        raise ValueError(f"min_candidates_per_query_group must be positive, got {min_candidates_per_query_group}")
    batch_results = compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        requests=[request.stats_request for request in prepared_requests],
        pair_batch_size=int(pair_batch_size),
        max_top_k=int(max_top_k),
    )
    if len(batch_results) != len(prepared_requests):
        raise RuntimeError(
            "Prepared query batch size did not match scored results: "
            f"prepared={len(prepared_requests)} results={len(batch_results)}"
        )
    batch_rows_to_write: list[dict[str, Any]] = []
    batch_query_group_metadata_rows: list[dict[str, Any]] = []
    for prepared_request, (stats_by_component, pairwise_diagnostics) in zip(
        prepared_requests,
        batch_results,
        strict=True,
    ):
        query_rows = _materialize_query_rows_from_prepared(
            prepared_request,
            stats_by_component=stats_by_component,
            rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        )
        for query_group_rows in group_rows(query_rows).values():
            metadata_row = summarize_query_group_rows(
                query_group_rows,
                block_component_count=int(prepared_request.block_component_count),
            )
            if int(metadata_row["candidate_count"]) < int(min_candidates_per_query_group):
                if filtered_query_group_stats is not None:
                    filtered_query_group_stats["query_groups"] = (
                        int(filtered_query_group_stats.get("query_groups", 0)) + 1
                    )
                    filtered_query_group_stats["rows"] = int(filtered_query_group_stats.get("rows", 0)) + int(
                        len(query_group_rows)
                    )
                continue
            batch_rows_to_write.extend(query_group_rows)
            batch_query_group_metadata_rows.append(metadata_row)
            if query_group_summary_accumulator is not None:
                query_group_summary_accumulator.update(metadata_row)
        pair_counts.append(int(pairwise_diagnostics["pair_count"]))
        featurize_seconds.append(float(pairwise_diagnostics["featurize_seconds"]))
        model_seconds.append(float(pairwise_diagnostics["model_predict_seconds"]))
    if rows is not None:
        rows.extend(batch_rows_to_write)
    if query_group_metadata_rows is not None:
        query_group_metadata_rows.extend(batch_query_group_metadata_rows)
    if rows_output_path is not None:
        append_rows_csv(rows_output_path, batch_rows_to_write)
    if query_group_metadata_output_path is not None:
        append_query_group_metadata_csv(query_group_metadata_output_path, batch_query_group_metadata_rows)
    return int(len(batch_rows_to_write)), int(len(batch_query_group_metadata_rows))


def _build_labeled_master_for_dataset(
    *,
    data_root: Path,
    dataset_name: str,
    clusterer: Any,
    output_dir: Path,
    query_views: list[str],
    retrieval_approach: str,
    retrieval_engine: str,
    window_size: int,
    max_exemplars: int,
    limit_queries: int | None,
    query_id_file: Path | None,
    n_jobs: int,
    seed: int,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_top_k: int,
    write_derived_cache: bool,
) -> dict[str, Any]:
    """Build one persisted labeled master-cache artifact."""

    configure_runtime_environment(n_jobs=n_jobs, backend="rust")
    dataset_start = time.perf_counter()
    dataset = load_labeled_dataset(
        data_root,
        dataset_name,
        n_jobs=n_jobs,
        clusterer=clusterer,
        load_name_counts="auto",
    )
    dataset_load_ms = (time.perf_counter() - dataset_start) * 1000.0
    name_counts_summary = _require_labeled_name_counts_source(dataset, dataset_name=dataset_name)
    _apply_dataset_name_count_semantics_for_prediction(clusterer, dataset)
    runtime_context = build_runtime_context("single_letter_reranker_dataset")
    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=clusterer.use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        use_cache=clusterer.use_cache,
        suppress_orcid=True,
    )
    query_cases, census, block_to_component_keys, component_signatures = build_labeled_query_cases(
        dataset_name,
        dataset,
        seed=seed,
        sampling_query_view=query_views[0],
        limit_queries=limit_queries,
    )
    explicit_query_ids = _read_string_id_file(query_id_file) if query_id_file is not None else None
    query_cases = _filter_query_sequence_by_id_set(query_cases, selected_query_ids=explicit_query_ids)
    full_summaries, feature_cache, full_summary_build_ms = build_component_summaries(
        dataset,
        component_signatures,
        max_exemplars=max_exemplars,
    )
    frozen_rust_hybrid_centroid_policy = None
    rust_hybrid_centroid_retriever = None
    retrieval_subblock_index = None
    retrieval_subblock_index_build_ms = 0.0
    retrieval_subblock_index_diagnostics = None
    if "hybrid_centroid" in {str(value) for value in str(retrieval_approach).split("__")}:
        frozen_rust_hybrid_centroid_policy = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY
        if str(frozen_rust_hybrid_centroid_policy.full_candidate_strategy) != "global":
            subblock_index_start = time.perf_counter()
            retrieval_subblock_index, retrieval_subblock_index_diagnostics = build_labeled_retrieval_subblock_index(
                dataset=dataset,
                block_to_component_keys=block_to_component_keys,
                component_signatures=component_signatures,
            )
            retrieval_subblock_index_build_ms = (time.perf_counter() - subblock_index_start) * 1000.0
        try:
            rust_hybrid_centroid_retriever = build_rust_hybrid_centroid_retriever(
                list(full_summaries.values()),
                include_exemplars=frozen_rust_hybrid_centroid_policy.uses_exemplar_scoring(),
            )
        except RuntimeError as exc:
            raise RuntimeError("The frozen Rust hybrid-centroid retriever is required for labeled row builds") from exc

    residual_cache: dict[tuple[str, str], retrieval.ClusterSummary] = {}
    rows: list[dict[str, Any]] = []
    query_group_metadata_rows: list[dict[str, Any]] = []
    pair_counts: list[int] = []
    featurize_seconds: list[float] = []
    model_seconds: list[float] = []
    prepared_requests: list[PreparedQueryRowsRequest] = []
    prepared_request_pair_count = 0
    raw_similarity_feature_cache = _RawSimilarityFeatureCache()

    for query_case in query_cases:
        base_query = retrieval.extract_query_features(
            dataset,
            query_case.query_signature_id,
            feature_cache=feature_cache,
            orcid_enabled=False,
        )
        component_keys_in_block = block_to_component_keys[query_case.block_key]
        raw_candidate_summaries: list[retrieval.ClusterSummary] = []
        summary_by_component: dict[str, retrieval.ClusterSummary] = {}
        candidate_signature_ids_by_component: dict[str, list[str]] = {}
        positive_component_key = next(iter(query_case.positive_component_keys))
        for component_key in component_keys_in_block:
            if component_key == positive_component_key:
                summary = _residual_summary(
                    dataset=dataset,
                    component_key=component_key,
                    heldout_signature_id=query_case.query_signature_id,
                    component_signatures=component_signatures,
                    feature_cache=feature_cache,
                    max_exemplars=max_exemplars,
                    cache=residual_cache,
                )
                signature_ids = [
                    signature_id
                    for signature_id in component_signatures[component_key]
                    if signature_id != query_case.query_signature_id
                ]
            else:
                summary = full_summaries[component_key]
                signature_ids = component_signatures[component_key]
            raw_candidate_summaries.append(summary)
            summary_by_component[component_key] = summary
            candidate_signature_ids_by_component[component_key] = list(signature_ids)

        prepared_request = _prepare_query_rows_request(
            dataset=dataset,
            query_case=query_case,
            block_component_count=len(component_keys_in_block),
            base_query=base_query,
            query_views=query_views,
            raw_candidate_summaries=raw_candidate_summaries,
            summary_by_component=summary_by_component,
            candidate_signature_ids_by_component=candidate_signature_ids_by_component,
            retrieval_approach=retrieval_approach,
            retrieval_engine=retrieval_engine,
            window_size=window_size,
            rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
            frozen_rust_hybrid_centroid_policy=frozen_rust_hybrid_centroid_policy,
            retrieval_subblock_index=retrieval_subblock_index,
            raw_similarity_feature_cache=raw_similarity_feature_cache,
        )
        prepared_requests.append(prepared_request)
        prepared_request_pair_count += int(prepared_request.estimated_pair_count)
        if prepared_request_pair_count >= int(query_batch_pair_limit):
            _ = _flush_prepared_query_requests(
                clusterer=clusterer,
                dataset=dataset,
                runtime_context=runtime_context,
                constraint_backend=constraint_backend,
                prepared_requests=prepared_requests,
                pair_batch_size=pair_batch_size,
                max_top_k=max_top_k,
                rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
                rows=rows,
                query_group_metadata_rows=query_group_metadata_rows,
                pair_counts=pair_counts,
                featurize_seconds=featurize_seconds,
                model_seconds=model_seconds,
            )
            prepared_requests = []
            prepared_request_pair_count = 0

    _ = _flush_prepared_query_requests(
        clusterer=clusterer,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        prepared_requests=prepared_requests,
        pair_batch_size=pair_batch_size,
        max_top_k=max_top_k,
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        rows=rows,
        query_group_metadata_rows=query_group_metadata_rows,
        pair_counts=pair_counts,
        featurize_seconds=featurize_seconds,
        model_seconds=model_seconds,
    )

    dataset_output_dir = output_dir / dataset_name
    write_rows_csv(dataset_output_dir / "rows.csv", rows)
    write_query_group_metadata_csv(dataset_output_dir / "query_groups.csv", query_group_metadata_rows)
    if write_derived_cache:
        write_materialized_rows_csv(
            dataset_output_dir / "rows_derived.csv",
            materialize_derived_rows(rows),
        )
    summary = summarize_dataset_rows(rows)
    summary.update(
        {
            "dataset": dataset_name,
            "dataset_load_ms": round(float(dataset_load_ms), 6),
            "full_summary_build_ms": round(float(full_summary_build_ms), 6),
            "retrieval_subblock_index_build_ms": round(float(retrieval_subblock_index_build_ms), 6),
            "query_case_count": int(len(query_cases)),
            "query_group_metadata_count": int(len(query_group_metadata_rows)),
            "pair_count_mean": round(float(statistics.mean(pair_counts)), 6) if pair_counts else 0.0,
            "pair_count_p95": round(float(np.percentile(pair_counts, 95)), 6) if pair_counts else 0.0,
            "pair_featurize_seconds_mean": round(float(statistics.mean(featurize_seconds)), 6)
            if featurize_seconds
            else 0.0,
            "pair_model_seconds_mean": round(float(statistics.mean(model_seconds)), 6) if model_seconds else 0.0,
            "census": census,
            "name_counts": name_counts_summary,
            "cache_role": "master",
            "requested_limit_queries": (int(limit_queries) if limit_queries is not None else None),
            "selected_query_id_filter_applied": bool(query_id_file is not None),
            "query_batch_pair_limit": int(query_batch_pair_limit),
            "max_top_k": int(max_top_k),
            "write_derived_cache": bool(write_derived_cache),
            "retrieval_subblock_index": retrieval_subblock_index_diagnostics,
            "frozen_retrieval_policy": (
                frozen_rust_hybrid_centroid_policy.to_summary_payload(
                    policy_name=FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME
                )
                if frozen_rust_hybrid_centroid_policy is not None
                else None
            ),
        }
    )
    write_json(dataset_output_dir / "summary.json", summary)
    return summary


def _build_giant_block_rows(
    *,
    data_dir: Path,
    step2_dir: Path,
    targets_dir: Path | None,
    dataset_label: str | None,
    clusterer: Any,
    output_dir: Path,
    query_views: list[str],
    query_source: str,
    retrieval_approach: str,
    retrieval_engine: str,
    window_size: int,
    max_exemplars: int,
    limit_queries: int | None,
    query_id_file: Path | None,
    n_jobs: int,
    seed: int,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_top_k: int,
    min_candidates_per_query_group: int,
) -> dict[str, Any]:
    """Build reranker rows for one extracted giant block."""

    configure_runtime_environment(n_jobs=n_jobs, backend="rust")
    raw_signatures = _read_json(_required_file(data_dir, "signatures.json"))
    signature_to_cluster_id, step2_assignment_info = load_preferred_signature_to_cluster_id(step2_dir)
    seed_cluster_counts_by_orcid = build_orcid_seed_cluster_counts(
        raw_signatures=raw_signatures,
        signature_to_cluster_id=signature_to_cluster_id,
    )

    dataset_start = time.perf_counter()
    dataset, load_info = load_giant_block_dataset(data_dir, block_key=None, n_jobs=n_jobs, clusterer=clusterer)
    dataset_load_ms = (time.perf_counter() - dataset_start) * 1000.0
    raw_paper_text_by_id = _load_raw_paper_text_by_id(
        _required_file(data_dir, "papers.json"),
        needed_paper_ids={str(paper_id) for paper_id in dataset.papers},
    )
    resolved_dataset_label = _normalize_giant_block_dataset_label(
        dataset_label if dataset_label is not None else str(load_info["target_block"])
    )
    _apply_dataset_name_count_semantics_for_prediction(clusterer, dataset)
    runtime_context = build_runtime_context("single_letter_reranker_giant_block")
    constraint_backend = _build_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=clusterer.use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        use_cache=clusterer.use_cache,
        suppress_orcid=True,
    )
    seed_clusters = invert_signature_to_cluster_id(signature_to_cluster_id)
    seed_summary_list, cluster_sizes, seed_summary_build_ms = build_seed_summaries(
        dataset=dataset,
        seed_clusters=seed_clusters,
        block_key=str(load_info["target_block"]),
        max_exemplars=max_exemplars,
    )
    frozen_rust_hybrid_centroid_policy = None
    retrieval_subblock_index = None
    if "hybrid_centroid" in {str(value) for value in str(retrieval_approach).split("__")}:
        frozen_rust_hybrid_centroid_policy = FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY
        if str(frozen_rust_hybrid_centroid_policy.full_candidate_strategy) != "global":
            retrieval_subblock_index = load_retrieval_subblock_index(step2_dir)
    include_rust_exemplars = bool(
        frozen_rust_hybrid_centroid_policy is not None and frozen_rust_hybrid_centroid_policy.uses_exemplar_scoring()
    )
    rust_hybrid_centroid_retriever = None
    if "hybrid_centroid" in {str(value) for value in str(retrieval_approach).split("__")}:
        try:
            rust_hybrid_centroid_retriever = build_rust_hybrid_centroid_retriever(
                seed_summary_list,
                include_exemplars=include_rust_exemplars,
            )
        except RuntimeError:
            rust_hybrid_centroid_retriever = None
    if frozen_rust_hybrid_centroid_policy is not None and rust_hybrid_centroid_retriever is None:
        raise RuntimeError("The frozen Rust hybrid-centroid retriever is required for giant-block row builds")
    if str(query_source) == "supported_single_letter":
        if targets_dir is None:
            raise ValueError("supported_single_letter giant-block rows require --targets-dir")
        query_metadata = load_query_metadata(_required_file(targets_dir, "query_set.json"))
        query_cases, query_source_summary = _build_giant_block_supported_query_cases(
            dataset_label=resolved_dataset_label,
            query_metadata=query_metadata,
            signature_to_cluster_id=signature_to_cluster_id,
            seed_cluster_counts_by_orcid=seed_cluster_counts_by_orcid,
            limit_queries=limit_queries,
            query_id_file=query_id_file,
            seed=seed,
        )
    elif str(query_source) == "orcid_any_input":
        query_cases, query_source_summary = _build_giant_block_any_input_query_cases(
            dataset_label=resolved_dataset_label,
            raw_signatures=raw_signatures,
            dataset=dataset,
            target_block=str(load_info["target_block"]),
            signature_to_cluster_id=signature_to_cluster_id,
            seed_cluster_counts_by_orcid=seed_cluster_counts_by_orcid,
            limit_queries=limit_queries,
            query_id_file=query_id_file,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported giant-block query_source: {query_source}")

    summary_by_component = {summary.component_key: summary for summary in seed_summary_list}
    rows_partial_path, query_groups_partial_path, progress_path, done_path = _reset_giant_block_stream_outputs(
        output_dir
    )
    summary_accumulator = _QueryGroupSummaryAccumulator()
    pair_counts: list[int] = []
    featurize_seconds: list[float] = []
    model_seconds: list[float] = []
    filtered_query_group_stats = {"query_groups": 0, "rows": 0}
    prepared_requests: list[PreparedQueryRowsRequest] = []
    prepared_request_pair_count = 0
    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    raw_similarity_feature_cache = _RawSimilarityFeatureCache()
    raw_candidate_summaries = list(seed_summary_list)
    residual_summary_cache: dict[tuple[str, str], retrieval.ClusterSummary] = {}
    processed_queries = 0
    written_rows = 0
    written_query_groups = 0
    started_at = _utc_timestamp()
    _write_giant_block_progress(
        progress_path,
        status="running",
        query_source=str(query_source),
        total_queries=int(len(query_cases)),
        processed_queries=int(processed_queries),
        written_rows=int(written_rows),
        written_query_groups=int(written_query_groups),
        last_query_id=None,
        started_at=started_at,
    )

    for query_case in query_cases:
        base_query = retrieval.extract_query_features(
            dataset,
            query_case.query_id,
            feature_cache=feature_cache,
            orcid_enabled=False,
        )
        query_views_for_case = (
            [str(query_case.natural_query_view or query_views[0])]
            if str(query_source) == "orcid_any_input"
            else [str(value) for value in query_views]
        )
        candidate_signature_ids_by_component = {key: list(value) for key, value in seed_clusters.items()}
        query_summary_by_component = dict(summary_by_component)
        query_candidate_summaries = list(raw_candidate_summaries)
        if bool(query_case.query_in_seed_before_holdout):
            cluster_id = signature_to_cluster_id.get(str(query_case.query_signature_id))
            if cluster_id is not None and str(cluster_id) in candidate_signature_ids_by_component:
                residual_signature_ids = [
                    signature_id
                    for signature_id in candidate_signature_ids_by_component[str(cluster_id)]
                    if str(signature_id) != str(query_case.query_signature_id)
                ]
                if residual_signature_ids:
                    candidate_signature_ids_by_component[str(cluster_id)] = residual_signature_ids
                    query_summary_by_component[str(cluster_id)] = _residual_seed_summary(
                        dataset=dataset,
                        block_key=str(load_info["target_block"]),
                        cluster_id=str(cluster_id),
                        heldout_signature_id=str(query_case.query_signature_id),
                        seed_clusters=seed_clusters,
                        feature_cache=feature_cache,
                        max_exemplars=max_exemplars,
                        cache=residual_summary_cache,
                    )
                    query_candidate_summaries = [
                        query_summary_by_component[str(summary.component_key)] for summary in raw_candidate_summaries
                    ]
                else:
                    del candidate_signature_ids_by_component[str(cluster_id)]
                    del query_summary_by_component[str(cluster_id)]
                    query_candidate_summaries = [
                        summary for summary in raw_candidate_summaries if str(summary.component_key) != str(cluster_id)
                    ]
        retrieval_window_state_base = {
            "candidate_components": int(len(query_candidate_summaries)),
            "candidate_signatures": int(sum(summary.size for summary in query_candidate_summaries)),
        }
        positive_component_sizes = [
            len(candidate_signature_ids_by_component[str(component_key)])
            for component_key in query_case.positive_component_keys
            if str(component_key) in candidate_signature_ids_by_component
        ]
        resolved_query_case = replace(
            query_case,
            block_key=str(load_info["target_block"]),
            block_size=int(sum(len(signature_ids) for signature_ids in candidate_signature_ids_by_component.values())),
            component_size=int(max(positive_component_sizes, default=0)),
            sampling_info_bucket=str(query_views_for_case[0]),
        )
        prepared_request = _prepare_query_rows_request(
            dataset=dataset,
            query_case=resolved_query_case,
            block_component_count=int(len(query_candidate_summaries)),
            base_query=base_query,
            query_views=query_views_for_case,
            raw_candidate_summaries=query_candidate_summaries,
            summary_by_component=query_summary_by_component,
            candidate_signature_ids_by_component=candidate_signature_ids_by_component,
            retrieval_approach=retrieval_approach,
            retrieval_engine=retrieval_engine,
            window_size=window_size,
            retrieval_window_state_base=retrieval_window_state_base,
            rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
            frozen_rust_hybrid_centroid_policy=frozen_rust_hybrid_centroid_policy,
            retrieval_subblock_index=retrieval_subblock_index,
            raw_paper_text_by_id=raw_paper_text_by_id,
            raw_similarity_feature_cache=raw_similarity_feature_cache,
        )
        prepared_requests.append(prepared_request)
        prepared_request_pair_count += int(prepared_request.estimated_pair_count)
        if prepared_request_pair_count >= int(query_batch_pair_limit):
            flushed_row_count, flushed_query_group_count = _flush_prepared_query_requests(
                clusterer=clusterer,
                dataset=dataset,
                runtime_context=runtime_context,
                constraint_backend=constraint_backend,
                prepared_requests=prepared_requests,
                pair_batch_size=pair_batch_size,
                max_top_k=max_top_k,
                rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
                pair_counts=pair_counts,
                featurize_seconds=featurize_seconds,
                model_seconds=model_seconds,
                rows_output_path=rows_partial_path,
                query_group_metadata_output_path=query_groups_partial_path,
                query_group_summary_accumulator=summary_accumulator,
                min_candidates_per_query_group=int(min_candidates_per_query_group),
                filtered_query_group_stats=filtered_query_group_stats,
            )
            processed_queries += int(len(prepared_requests))
            written_rows += int(flushed_row_count)
            written_query_groups += int(flushed_query_group_count)
            _write_giant_block_progress(
                progress_path,
                status="running",
                query_source=str(query_source),
                total_queries=int(len(query_cases)),
                processed_queries=int(processed_queries),
                written_rows=int(written_rows),
                written_query_groups=int(written_query_groups),
                last_query_id=str(prepared_requests[-1].query_case.query_id),
                started_at=started_at,
            )
            prepared_requests = []
            prepared_request_pair_count = 0

    flushed_row_count, flushed_query_group_count = _flush_prepared_query_requests(
        clusterer=clusterer,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        prepared_requests=prepared_requests,
        pair_batch_size=pair_batch_size,
        max_top_k=max_top_k,
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        pair_counts=pair_counts,
        featurize_seconds=featurize_seconds,
        model_seconds=model_seconds,
        rows_output_path=rows_partial_path,
        query_group_metadata_output_path=query_groups_partial_path,
        query_group_summary_accumulator=summary_accumulator,
        min_candidates_per_query_group=int(min_candidates_per_query_group),
        filtered_query_group_stats=filtered_query_group_stats,
    )
    if prepared_requests:
        processed_queries += int(len(prepared_requests))
        written_rows += int(flushed_row_count)
        written_query_groups += int(flushed_query_group_count)
    summary = summary_accumulator.to_summary()
    summary.update(
        {
            "dataset": resolved_dataset_label,
            "target_block": str(load_info["target_block"]),
            "query_source": str(query_source),
            "dataset_load_ms": round(float(dataset_load_ms), 6),
            "seed_summary_build_ms": round(float(seed_summary_build_ms), 6),
            "query_count": int(len(query_cases)),
            "query_group_metadata_count": int(written_query_groups),
            "query_source_summary": query_source_summary,
            "min_candidates_per_query_group": int(min_candidates_per_query_group),
            "filtered_query_groups_min_candidates_count": int(filtered_query_group_stats["query_groups"]),
            "filtered_rows_min_candidates_count": int(filtered_query_group_stats["rows"]),
            "pair_count_mean": round(float(statistics.mean(pair_counts)), 6) if pair_counts else 0.0,
            "pair_count_p95": round(float(np.percentile(pair_counts, 95)), 6) if pair_counts else 0.0,
            "pair_featurize_seconds_mean": round(float(statistics.mean(featurize_seconds)), 6)
            if featurize_seconds
            else 0.0,
            "pair_model_seconds_mean": round(float(statistics.mean(model_seconds)), 6) if model_seconds else 0.0,
            "query_batch_pair_limit": int(query_batch_pair_limit),
            "max_top_k": int(max_top_k),
            "step2_assignment": dict(step2_assignment_info),
            "frozen_retrieval_policy": (
                frozen_rust_hybrid_centroid_policy.to_summary_payload(
                    policy_name=FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME
                )
                if frozen_rust_hybrid_centroid_policy is not None
                else None
            ),
        }
    )
    _write_giant_block_progress(
        progress_path,
        status="completed",
        query_source=str(query_source),
        total_queries=int(len(query_cases)),
        processed_queries=int(processed_queries),
        written_rows=int(written_rows),
        written_query_groups=int(written_query_groups),
        last_query_id=(str(query_cases[-1].query_id) if query_cases else None),
        started_at=started_at,
    )
    rows_partial_path.replace(output_dir / "rows.csv")
    query_groups_partial_path.replace(output_dir / "query_groups.csv")
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(
        done_path,
        {
            "status": "completed",
            "query_source": str(query_source),
            "query_count": int(len(query_cases)),
            "row_count": int(summary["row_count"]),
            "query_group_metadata_count": int(written_query_groups),
            "completed_at": _utc_timestamp(),
        },
    )
    return summary


def _load_all_query_group_metadata(dataset_root: Path, datasets: list[str]) -> list[dict[str, Any]]:
    """Load cached query-group metadata rows across the requested datasets."""

    metadata_rows: list[dict[str, Any]] = []
    for dataset_name in datasets:
        metadata_rows.extend(read_query_group_metadata_csv(dataset_root / dataset_name / "query_groups.csv"))
    return metadata_rows


def _build_selected_query_groups_artifact(
    *,
    dataset_root: Path,
    datasets: list[str],
    output_dir: Path,
    labeled_query_sampler: str,
    limit_query_groups: int | None,
    seed: int,
) -> dict[str, Any]:
    """Select cached labeled query groups and persist the chosen IDs."""

    metadata_rows = _load_all_query_group_metadata(dataset_root, datasets)
    selected_rows, sampler_summary = _select_query_group_metadata_rows(
        metadata_rows,
        labeled_query_sampler=labeled_query_sampler,
        limit_query_groups=limit_query_groups,
        seed=seed,
    )
    selected_query_group_ids = [str(row["query_group_id"]) for row in selected_rows]
    _write_string_id_file(output_dir / "selected_query_groups.txt", selected_query_group_ids)
    write_query_group_metadata_csv(output_dir / "selected_query_groups.csv", selected_rows)
    summary = {
        "mode": "select",
        "dataset_root": str(dataset_root),
        "datasets": [str(value) for value in datasets],
        "labeled_query_sampler": str(labeled_query_sampler),
        "selected_query_group_count": int(len(selected_rows)),
        "selected_by_dataset": {
            str(dataset_name): int(sum(1 for row in selected_rows if str(row["dataset"]) == str(dataset_name)))
            for dataset_name in datasets
        },
        "sampler_summary": sampler_summary,
        "selected_query_groups_file": str(output_dir / "selected_query_groups.txt"),
        "selected_query_groups_csv": str(output_dir / "selected_query_groups.csv"),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _materialize_derived_dataset_root(
    *,
    dataset_root: Path,
    output_root: Path,
    datasets: list[str],
    selected_query_group_ids: set[str] | None,
) -> dict[str, Any]:
    """Write derived-feature caches for the requested dataset root."""

    dataset_summaries: dict[str, Any] = {}
    for dataset_name in datasets:
        rows = read_rows_csv(dataset_root / dataset_name / "rows.csv")
        filtered_rows = [
            row
            for row in rows
            if selected_query_group_ids is None or str(row["query_group_id"]) in selected_query_group_ids
        ]
        materialized_rows = materialize_derived_rows(filtered_rows)
        output_dataset_dir = output_root / dataset_name
        write_materialized_rows_csv(output_dataset_dir / "rows_derived.csv", materialized_rows)
        metadata_path = dataset_root / dataset_name / "query_groups.csv"
        if metadata_path.exists():
            metadata_rows = read_query_group_metadata_csv(metadata_path)
            if selected_query_group_ids is not None:
                metadata_rows = [row for row in metadata_rows if str(row["query_group_id"]) in selected_query_group_ids]
            write_query_group_metadata_csv(output_dataset_dir / "query_groups.csv", metadata_rows)
        dataset_summaries[dataset_name] = {
            "rows_in": int(len(rows)),
            "rows_out": int(len(materialized_rows)),
            "selected_query_group_count": int(len({str(row["query_group_id"]) for row in materialized_rows})),
            "rows_derived_path": str(output_dataset_dir / "rows_derived.csv"),
        }
    summary = {
        "mode": "materialize-derived",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "datasets": [str(value) for value in datasets],
        "selected_query_group_filter_applied": bool(selected_query_group_ids is not None),
        "dataset_summaries": dataset_summaries,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    labeled_parser = subparsers.add_parser("labeled", help="Build labeled-dataset master-cache rows.")
    labeled_parser.add_argument("--data-root", type=Path, default=Path("data"))
    labeled_parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_LABELED_DATASETS))
    labeled_parser.add_argument("--output-dir", type=Path, required=True)
    labeled_parser.add_argument("--query-id-file", type=Path, default=None)
    labeled_parser.add_argument("--write-derived-cache", action="store_true")

    select_parser = subparsers.add_parser("select", help="Select cached labeled query groups.")
    select_parser.add_argument("--dataset-root", type=Path, required=True)
    select_parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_LABELED_DATASETS))
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument(
        "--labeled-query-sampler",
        choices=("round_robin_v1", "hard_blocks_v1"),
        required=True,
    )
    select_parser.add_argument("--limit-query-groups", type=int, default=None)
    select_parser.add_argument("--seed", type=int, default=13)

    materialize_parser = subparsers.add_parser(
        "materialize-derived",
        help="Materialize derived numeric feature columns from cached rows.",
    )
    materialize_parser.add_argument("--dataset-root", type=Path, required=True)
    materialize_parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_LABELED_DATASETS))
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--selected-query-groups-file", type=Path, default=None)

    giant_block_parser = subparsers.add_parser("giant_block", help="Build generic giant-block reranker rows.")
    giant_block_parser.add_argument("--data-dir", type=Path, required=True)
    giant_block_parser.add_argument("--step2-dir", type=Path, required=True)
    giant_block_parser.add_argument("--targets-dir", type=Path, default=None)
    giant_block_parser.add_argument("--output-dir", type=Path, required=True)
    giant_block_parser.add_argument("--query-id-file", type=Path, default=None)
    giant_block_parser.add_argument(
        "--query-source",
        choices=GIANT_BLOCK_QUERY_SOURCE_CHOICES,
        default="supported_single_letter",
    )
    giant_block_parser.add_argument(
        "--dataset-label",
        type=str,
        default=None,
        help="Optional dataset/source label for emitted rows. Defaults to the normalized target block.",
    )
    giant_block_parser.add_argument("--min-candidates-per-query-group", type=int, default=1)

    h_wang_parser = subparsers.add_parser("h_wang", help="Build `h_wang` reranker rows.")
    h_wang_parser.add_argument("--data-dir", type=Path, required=True)
    h_wang_parser.add_argument("--step2-dir", type=Path, required=True)
    h_wang_parser.add_argument("--targets-dir", type=Path, default=None)
    h_wang_parser.add_argument("--output-dir", type=Path, required=True)
    h_wang_parser.add_argument("--query-id-file", type=Path, default=None)
    h_wang_parser.add_argument(
        "--query-source",
        choices=GIANT_BLOCK_QUERY_SOURCE_CHOICES,
        default="supported_single_letter",
    )
    h_wang_parser.add_argument("--min-candidates-per-query-group", type=int, default=1)

    for subparser in (labeled_parser, giant_block_parser, h_wang_parser):
        subparser.add_argument("--query-views", nargs="+", default=list(DEFAULT_QUERY_VIEWS))
        subparser.add_argument("--retrieval-approach", default=DEFAULT_RETRIEVAL_APPROACH)
        subparser.add_argument("--retrieval-engine", choices=sorted(RETRIEVAL_ENGINE_CHOICES), default="auto")
        subparser.add_argument("--window-size", type=int, default=DEFAULT_RETRIEVAL_WINDOW_SIZE)
        subparser.add_argument("--max-exemplars", type=int, default=4)
        subparser.add_argument("--max-top-k", type=int, default=DEFAULT_CHOOSER_CACHE_MAX_TOP_K)
        subparser.add_argument("--limit-queries", type=int, default=None)
        subparser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
        subparser.add_argument("--n-jobs", type=int, default=8)
        subparser.add_argument("--seed", type=int, default=13)
        subparser.add_argument("--pair-batch-size", type=int, default=None)
        subparser.add_argument("--query-batch-pair-limit", type=int, default=DEFAULT_QUERY_BATCH_PAIR_LIMIT)
        subparser.add_argument("--use-cache", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Run the requested cache-build command."""

    args = parse_args()
    if args.mode == "select":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _build_selected_query_groups_artifact(
            dataset_root=args.dataset_root,
            datasets=[str(value) for value in args.datasets],
            output_dir=args.output_dir,
            labeled_query_sampler=str(args.labeled_query_sampler),
            limit_query_groups=args.limit_query_groups,
            seed=int(args.seed),
        )
        return
    if args.mode == "materialize-derived":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selected_query_group_ids = (
            _read_string_id_file(args.selected_query_groups_file)
            if args.selected_query_groups_file is not None
            else None
        )
        _materialize_derived_dataset_root(
            dataset_root=args.dataset_root,
            output_root=args.output_dir,
            datasets=[str(value) for value in args.datasets],
            selected_query_group_ids=selected_query_group_ids,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.mode in {"labeled", "giant_block", "h_wang"}
        and args.limit_queries is not None
        and args.query_id_file is not None
    ):
        raise ValueError(f"Use either --limit-queries or --query-id-file for `{args.mode}`, not both")
    if (
        args.mode in {"giant_block", "h_wang"}
        and str(args.query_source) == "supported_single_letter"
        and args.targets_dir is None
    ):
        raise ValueError(f"`{args.mode}` supported_single_letter mode requires --targets-dir")
    if (
        args.mode in {"giant_block", "h_wang"}
        and str(args.query_source) == "orcid_any_input"
        and list(args.query_views) != list(DEFAULT_QUERY_VIEWS)
    ):
        raise ValueError(
            f"`{args.mode}` orcid_any_input queries use their natural view; do not pass custom --query-views"
        )

    clusterer = load_clusterer(args.model_path, n_jobs=args.n_jobs)
    clusterer.use_cache = bool(args.use_cache)
    pair_batch_size = int(args.pair_batch_size) if args.pair_batch_size is not None else int(clusterer.batch_size)
    query_batch_pair_limit = int(args.query_batch_pair_limit)
    if query_batch_pair_limit <= 0:
        raise ValueError(f"query_batch_pair_limit must be positive, got {query_batch_pair_limit}")
    if args.mode in {"giant_block", "h_wang"} and int(args.min_candidates_per_query_group) <= 0:
        raise ValueError(f"min_candidates_per_query_group must be positive, got {args.min_candidates_per_query_group}")

    if args.mode == "labeled":
        dataset_summaries = {}
        for dataset_name in args.datasets:
            dataset_summaries[str(dataset_name)] = _build_labeled_master_for_dataset(
                data_root=args.data_root,
                dataset_name=str(dataset_name),
                clusterer=clusterer,
                output_dir=args.output_dir,
                query_views=[str(value) for value in args.query_views],
                retrieval_approach=str(args.retrieval_approach),
                retrieval_engine=str(args.retrieval_engine),
                window_size=int(args.window_size),
                max_exemplars=int(args.max_exemplars),
                limit_queries=args.limit_queries,
                query_id_file=args.query_id_file,
                n_jobs=int(args.n_jobs),
                seed=int(args.seed),
                pair_batch_size=pair_batch_size,
                query_batch_pair_limit=query_batch_pair_limit,
                max_top_k=int(args.max_top_k),
                write_derived_cache=bool(args.write_derived_cache),
            )
        write_json(
            args.output_dir / "summary.json",
            {
                "mode": "labeled",
                "cache_role": "master",
                "datasets": dataset_summaries,
                "query_views": [str(value) for value in args.query_views],
                "window_size": int(args.window_size),
                "max_top_k": int(args.max_top_k),
                "pair_batch_size": int(pair_batch_size),
                "query_batch_pair_limit": int(query_batch_pair_limit),
                "use_cache": bool(clusterer.use_cache),
                "write_derived_cache": bool(args.write_derived_cache),
            },
        )
        return

    summary = _build_giant_block_rows(
        data_dir=args.data_dir,
        step2_dir=args.step2_dir,
        targets_dir=args.targets_dir,
        dataset_label=("h_wang" if args.mode == "h_wang" else args.dataset_label),
        clusterer=clusterer,
        output_dir=args.output_dir,
        query_views=[str(value) for value in args.query_views],
        query_source=str(args.query_source),
        retrieval_approach=str(args.retrieval_approach),
        retrieval_engine=str(args.retrieval_engine),
        window_size=int(args.window_size),
        max_exemplars=int(args.max_exemplars),
        limit_queries=args.limit_queries,
        query_id_file=args.query_id_file,
        n_jobs=int(args.n_jobs),
        seed=int(args.seed),
        pair_batch_size=pair_batch_size,
        query_batch_pair_limit=query_batch_pair_limit,
        max_top_k=int(args.max_top_k),
        min_candidates_per_query_group=int(args.min_candidates_per_query_group),
    )
    write_json(args.output_dir / "run_summary.json", summary)


if __name__ == "__main__":
    main()
