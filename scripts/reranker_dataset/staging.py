"""Official bundle row-staging helpers for reranker dataset rebuilds."""

from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from scripts.joint_safe_link_dataset_contract import apply_retrieval_rank_filter
    from scripts.single_letter_reranker_utils import (
        DERIVED_FEATURE_COLUMNS,
        NAME_COUNT_RARITY_FEATURE_COLUMNS,
        RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from joint_safe_link_dataset_contract import apply_retrieval_rank_filter  # type: ignore
    from single_letter_reranker_utils import (  # type: ignore
        DERIVED_FEATURE_COLUMNS,
        NAME_COUNT_RARITY_FEATURE_COLUMNS,
        RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    )


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def normalize_bundle_relpath(path_like: str | Path) -> str:
    """Return the bundle-relative path format used in official CSV metadata."""

    return str(path_like).replace("/", "\\")


def compress_rows(rows: list[dict[str, Any]]) -> bytes:
    """Compress staged row dictionaries for the rebuild spool database."""

    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return zlib.compress(payload, level=1)


def decompress_rows(blob: bytes) -> list[dict[str, Any]]:
    """Decompress staged row dictionaries from the rebuild spool database."""

    return list(json.loads(zlib.decompress(blob).decode("utf-8")))


@dataclass
class FileRepairSummaryState:
    """Before/after counters for one official row file during rebuild."""

    path: str
    groups_before: int = 0
    groups_after: int = 0
    rows_before: int = 0
    rows_after: int = 0
    positive_groups_before: int = 0
    positive_groups_after: int = 0
    positive_rows_before: int = 0
    positive_rows_after: int = 0
    groups_with_dropped_rows: int = 0
    groups_fully_dropped: int = 0
    groups_lost_all_positives: int = 0
    per_dataset_groups: Counter[str] = field(default_factory=Counter)
    sample_dropped_groups: list[dict[str, Any]] = field(default_factory=list)

    def record_stage(self, *, dataset_name: str, rows_before: int, positive_rows_before: int) -> None:
        """Record input-stage counts for one query group."""

        self.groups_before += 1
        self.rows_before += int(rows_before)
        self.positive_rows_before += int(positive_rows_before)
        self.per_dataset_groups[str(dataset_name)] += 1
        if int(positive_rows_before) > 0:
            self.positive_groups_before += 1

    def record_result(
        self,
        *,
        rows_before: int,
        positive_rows_before: int,
        rebuilt_rows: list[dict[str, Any]],
        group_summary: dict[str, Any],
    ) -> None:
        """Record rebuilt output counts for one query group."""

        rows_after = int(len(rebuilt_rows))
        positive_rows_after = int(sum(1 for row in rebuilt_rows if _to_int(row.get("label")) == 1))
        if rows_after < int(rows_before):
            self.groups_with_dropped_rows += 1
            if len(self.sample_dropped_groups) < 20:
                self.sample_dropped_groups.append(group_summary)
        if rows_after > 0:
            self.groups_after += 1
        else:
            self.groups_fully_dropped += 1
        self.rows_after += rows_after
        self.positive_rows_after += positive_rows_after
        if positive_rows_after > 0:
            self.positive_groups_after += 1
        elif int(positive_rows_before) > 0:
            self.groups_lost_all_positives += 1

    def to_payload(self) -> dict[str, Any]:
        """Return the full summary payload for final rebuild reports."""

        return {
            "path": str(self.path),
            "groups_before": int(self.groups_before),
            "groups_after": int(self.groups_after),
            "rows_before": int(self.rows_before),
            "rows_after": int(self.rows_after),
            "rows_dropped": int(self.rows_before - self.rows_after),
            "positive_groups_before": int(self.positive_groups_before),
            "positive_groups_after": int(self.positive_groups_after),
            "positive_rows_before": int(self.positive_rows_before),
            "positive_rows_after": int(self.positive_rows_after),
            "positive_rows_dropped": int(self.positive_rows_before - self.positive_rows_after),
            "groups_with_dropped_rows": int(self.groups_with_dropped_rows),
            "groups_fully_dropped": int(self.groups_fully_dropped),
            "groups_lost_all_positives": int(self.groups_lost_all_positives),
            "per_dataset_groups": {str(key): int(value) for key, value in sorted(self.per_dataset_groups.items())},
            "sample_dropped_groups": list(self.sample_dropped_groups),
        }

    def to_result_payload(self) -> dict[str, Any]:
        """Return worker-owned result counters for parent-process merge."""

        return {
            "path": str(self.path),
            "groups_after": int(self.groups_after),
            "rows_after": int(self.rows_after),
            "positive_groups_after": int(self.positive_groups_after),
            "positive_rows_after": int(self.positive_rows_after),
            "groups_with_dropped_rows": int(self.groups_with_dropped_rows),
            "groups_fully_dropped": int(self.groups_fully_dropped),
            "groups_lost_all_positives": int(self.groups_lost_all_positives),
            "sample_dropped_groups": list(self.sample_dropped_groups),
        }

    def merge_result_payload(self, payload: dict[str, Any]) -> None:
        """Merge a worker result payload without changing staged input counters."""

        self.groups_after += int(_to_int(payload.get("groups_after")))
        self.rows_after += int(_to_int(payload.get("rows_after")))
        self.positive_groups_after += int(_to_int(payload.get("positive_groups_after")))
        self.positive_rows_after += int(_to_int(payload.get("positive_rows_after")))
        self.groups_with_dropped_rows += int(_to_int(payload.get("groups_with_dropped_rows")))
        self.groups_fully_dropped += int(_to_int(payload.get("groups_fully_dropped")))
        self.groups_lost_all_positives += int(_to_int(payload.get("groups_lost_all_positives")))
        for sample in payload.get("sample_dropped_groups", []):
            if len(self.sample_dropped_groups) >= 20:
                break
            self.sample_dropped_groups.append(dict(sample))


@dataclass(frozen=True)
class StageInputGroupsConfig:
    """Runtime dependencies for official row staging."""

    source_bundle_root: Path
    s2and_row_relative_path: Path
    s2and_full_relabel_pre_filter_rows_path: Path
    window_size: int
    read_initial_only_rereview_decisions: Callable[[], dict[str, Any]]
    read_s2and_full_relabel_decisions: Callable[[], dict[str, Any]]
    merge_initial_only_rereview_into_s2and_decisions: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    apply_initial_only_rereview_to_group: Callable[..., list[dict[str, Any]]]
    apply_s2and_full_relabel_to_group: Callable[..., list[dict[str, Any]]]


def is_s2and_eval_row_path(path_like: str | Path, *, config: StageInputGroupsConfig) -> bool:
    """Return whether a selected official row path is the S2AND eval slice."""

    return normalize_bundle_relpath(path_like) == normalize_bundle_relpath(config.s2and_row_relative_path)


def source_rows_path_for_rebuild(relative_path: Path, *, config: StageInputGroupsConfig) -> Path:
    """Return the row source used for one rebuild input."""

    if is_s2and_eval_row_path(relative_path, config=config) and config.s2and_full_relabel_pre_filter_rows_path.exists():
        return config.s2and_full_relabel_pre_filter_rows_path
    return config.source_bundle_root / relative_path


def fieldnames_with_materialized_derived_columns(fieldnames: list[str]) -> list[str]:
    """Return CSV fieldnames with every official materialized feature preserved."""

    existing = {str(field) for field in fieldnames}
    materialized_columns = (
        set(DERIVED_FEATURE_COLUMNS)
        | set(RAW_METADATA_SIMILARITY_FEATURE_COLUMNS)
        | set(NAME_COUNT_RARITY_FEATURE_COLUMNS)
    )
    return [
        *fieldnames,
        *(column for column in sorted(materialized_columns) if column not in existing),
    ]


def load_selected_row_headers(
    selected_row_paths: tuple[Path, ...],
    *,
    config: StageInputGroupsConfig,
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    """Load selected row-file headers and initialize per-file stage counters."""

    fieldnames_by_path: dict[str, list[str]] = {}
    file_summaries: dict[str, FileRepairSummaryState] = {}
    ordered_source_paths: list[str] = []
    for relative_path in selected_row_paths:
        source_path = normalize_bundle_relpath(relative_path)
        ordered_source_paths.append(source_path)
        file_summaries[source_path] = FileRepairSummaryState(path=source_path)
        input_path = source_rows_path_for_rebuild(relative_path, config=config)
        with gzip.open(input_path, "rt", encoding="utf-8", newline="") as src_handle:
            reader = csv.DictReader(src_handle)
            fieldnames = [str(value) for value in reader.fieldnames or []]
            if not fieldnames:
                raise ValueError(f"CSV has no header: {input_path}")
            fieldnames_by_path[source_path] = fieldnames_with_materialized_derived_columns(fieldnames)
    return fieldnames_by_path, file_summaries, ordered_source_paths


def _source_path_placeholders(source_paths: list[str] | tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    normalized_paths = tuple(str(path) for path in source_paths)
    if not normalized_paths:
        raise ValueError("At least one source path is required.")
    placeholders = ", ".join("?" for _ in normalized_paths)
    return placeholders, normalized_paths


def hydrate_stage_summaries_from_spool(
    connection: sqlite3.Connection,
    *,
    file_summaries: dict[str, FileRepairSummaryState],
    ordered_source_paths: list[str],
) -> None:
    """Recover staged input counters from an existing rebuild spool database."""

    placeholders, params = _source_path_placeholders(ordered_source_paths)
    rows = connection.execute(
        f"""
        SELECT
            source_path,
            dataset_name,
            COUNT(*) AS group_count,
            SUM(rows_before_total) AS rows_before,
            SUM(positive_rows_before_total) AS positive_rows_before,
            SUM(CASE WHEN positive_rows_before_total > 0 THEN 1 ELSE 0 END) AS positive_groups_before
        FROM staged_groups
        WHERE source_path IN ({placeholders})
        GROUP BY source_path, dataset_name
        ORDER BY source_path, dataset_name
        """,
        params,
    )
    for source_path, dataset_name, group_count, rows_before, positive_rows_before, positive_groups_before in rows:
        state = file_summaries.get(str(source_path))
        if state is None:
            continue
        state.groups_before += int(_to_int(group_count))
        state.rows_before += int(_to_int(rows_before))
        state.positive_rows_before += int(_to_int(positive_rows_before))
        state.positive_groups_before += int(_to_int(positive_groups_before))
        state.per_dataset_groups[str(dataset_name)] += int(_to_int(group_count))
    missing_source_paths = [
        source_path for source_path in ordered_source_paths if file_summaries[source_path].groups_before == 0
    ]
    if missing_source_paths:
        missing_display = ", ".join(missing_source_paths[:5])
        raise FileNotFoundError(f"Spool db missing staged groups for selected row files: {missing_display}")


def load_staged_input_groups_from_spool(
    connection: sqlite3.Connection,
    *,
    selected_row_paths: tuple[Path, ...],
    config: StageInputGroupsConfig,
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    """Load row-file headers and recover staged counters from an existing spool."""

    fieldnames_by_path, file_summaries, ordered_source_paths = load_selected_row_headers(
        selected_row_paths,
        config=config,
    )
    hydrate_stage_summaries_from_spool(
        connection,
        file_summaries=file_summaries,
        ordered_source_paths=ordered_source_paths,
    )
    return fieldnames_by_path, file_summaries, ordered_source_paths


def preflight_s2and_full_relabel_decisions(
    *,
    selected_row_paths: tuple[Path, ...],
    decisions: dict[str, Any],
    config: StageInputGroupsConfig,
    sample_size: int = 10,
) -> None:
    """Fail before staging if active S2AND relabel groups lack decisions."""

    if not any(is_s2and_eval_row_path(path, config=config) for path in selected_row_paths):
        return
    input_path = source_rows_path_for_rebuild(config.s2and_row_relative_path, config=config)
    if input_path != config.s2and_full_relabel_pre_filter_rows_path:
        return

    active_query_ids: set[str] = set()
    sample_by_query_id: dict[str, dict[str, str]] = {}
    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_group_id = str(row["query_group_id"])
            active_query_ids.add(query_group_id)
            if len(sample_by_query_id) < sample_size and query_group_id not in decisions:
                sample_by_query_id[query_group_id] = {
                    "query_group_id": query_group_id,
                    "dataset": str(row.get("dataset", "")),
                }

    missing_query_ids = sorted(active_query_ids - set(decisions))
    if not missing_query_ids:
        return
    sample = [
        sample_by_query_id.get(query_id, {"query_group_id": query_id}) for query_id in missing_query_ids[:sample_size]
    ]
    raise ValueError(
        "Missing S2AND full relabel decisions for active query groups before rebuild staging: "
        f"missing_count={len(missing_query_ids)} sample={sample}"
    )


def _insert_staged_rows(connection: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO staged_groups (
            source_path,
            group_index,
            dataset_name,
            query_group_id,
            rows_before_total,
            positive_rows_before_total,
            rows_after_window_cap,
            positive_rows_after_window_cap,
            rows_blob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


def stage_input_groups(
    *,
    connection: sqlite3.Connection,
    selected_row_paths: tuple[Path, ...],
    limit_groups_per_file: int | None,
    config: StageInputGroupsConfig,
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    """Stage selected official row groups into the rebuild spool database."""

    fieldnames_by_path, file_summaries, ordered_source_paths = load_selected_row_headers(
        selected_row_paths,
        config=config,
    )
    initial_only_rereview_decisions = dict(config.read_initial_only_rereview_decisions())
    s2and_relabel_decisions = (
        config.merge_initial_only_rereview_into_s2and_decisions(
            dict(config.read_s2and_full_relabel_decisions()),
            initial_only_rereview_decisions,
        )
        if any(
            is_s2and_eval_row_path(path, config=config)
            and config.s2and_full_relabel_pre_filter_rows_path.exists()
            for path in selected_row_paths
        )
        else {}
    )
    preflight_s2and_full_relabel_decisions(
        selected_row_paths=selected_row_paths,
        decisions=s2and_relabel_decisions,
        config=config,
    )
    insert_rows: list[tuple[Any, ...]] = []
    flush_every = 500

    for relative_path in selected_row_paths:
        source_path = normalize_bundle_relpath(relative_path)
        staged_group_count = 0
        input_path = source_rows_path_for_rebuild(relative_path, config=config)
        relabel_s2and = (
            is_s2and_eval_row_path(relative_path, config=config)
            and input_path == config.s2and_full_relabel_pre_filter_rows_path
        )
        with gzip.open(input_path, "rt", encoding="utf-8", newline="") as src_handle:
            reader = csv.DictReader(src_handle)
            current_group_id: str | None = None
            current_rows: list[dict[str, Any]] = []

            def flush_group(
                group_rows: list[dict[str, Any]],
                *,
                target_source_path: str = source_path,
                relabel_group_s2and: bool = relabel_s2and,
            ) -> None:
                nonlocal staged_group_count
                if not group_rows:
                    return
                initial_only_decision = initial_only_rereview_decisions.get(str(group_rows[0]["query_group_id"]))
                if initial_only_decision is not None:
                    group_rows = config.apply_initial_only_rereview_to_group(
                        group_rows,
                        decision=initial_only_decision,
                    )
                    if not group_rows:
                        return
                elif relabel_group_s2and:
                    group_rows = config.apply_s2and_full_relabel_to_group(
                        group_rows,
                        decisions=s2and_relabel_decisions,
                    )
                    if not group_rows:
                        return
                if limit_groups_per_file is not None and staged_group_count >= int(limit_groups_per_file):
                    raise StopIteration
                staged_group_count += 1
                dataset_name = str(group_rows[0]["dataset"])
                query_group_id = str(group_rows[0]["query_group_id"])
                rows_before_total = int(len(group_rows))
                positive_rows_before_total = int(sum(1 for row in group_rows if _to_int(row.get("label")) == 1))
                window_filter = apply_retrieval_rank_filter(group_rows, retrieval_rank_limit=int(config.window_size))
                rows_after_window_cap_rows = list(window_filter.kept_rows)
                positive_rows_after_window_cap = int(window_filter.positive_rows_after)
                if (
                    initial_only_decision is not None
                    and initial_only_decision.action == "candidate_positive"
                    and positive_rows_after_window_cap == 0
                ):
                    return
                file_summaries[target_source_path].record_stage(
                    dataset_name=dataset_name,
                    rows_before=rows_before_total,
                    positive_rows_before=positive_rows_before_total,
                )
                insert_rows.append(
                    (
                        target_source_path,
                        int(staged_group_count),
                        dataset_name,
                        query_group_id,
                        rows_before_total,
                        positive_rows_before_total,
                        int(len(rows_after_window_cap_rows)),
                        positive_rows_after_window_cap,
                        sqlite3.Binary(compress_rows(rows_after_window_cap_rows)),
                    )
                )
                if len(insert_rows) >= flush_every:
                    _insert_staged_rows(connection, insert_rows)
                    insert_rows.clear()

            try:
                for row in reader:
                    group_id = str(row["query_group_id"])
                    if current_group_id is None:
                        current_group_id = group_id
                    if group_id != current_group_id:
                        flush_group(current_rows)
                        current_rows = [dict(row)]
                        current_group_id = group_id
                    else:
                        current_rows.append(dict(row))
                if current_rows:
                    flush_group(current_rows)
            except StopIteration:
                pass
        print(
            json.dumps(
                {
                    "event": "staged_row_file",
                    "path": source_path,
                    "groups": int(file_summaries[source_path].groups_before),
                    "rows": int(file_summaries[source_path].rows_before),
                }
            ),
            flush=True,
        )

    _insert_staged_rows(connection, insert_rows)
    return fieldnames_by_path, file_summaries, ordered_source_paths
