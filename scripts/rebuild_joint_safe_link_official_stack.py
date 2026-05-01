from __future__ import annotations

# ruff: noqa: E402, E501
import argparse
import csv
import gc
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

try:
    from scripts.joint_safe_link_dataset_contract import (
        apply_retrieval_rank_filter,
        retrieval_rank_limit_from_policy,
        write_contract_artifacts,
    )
    from scripts.joint_safe_link_official_stack import (
        compare_to_expected,
        expected_metrics_from_summary,
        load_bundle,
        run_classic,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from joint_safe_link_dataset_contract import (  # type: ignore
        apply_retrieval_rank_filter,
        retrieval_rank_limit_from_policy,
        write_contract_artifacts,
    )
    from joint_safe_link_official_stack import (  # type: ignore
        compare_to_expected,
        expected_metrics_from_summary,
        load_bundle,
        run_classic,
    )

import scripts.eval_cluster_retrieval as retrieval
from s2and.data import ANDData
from s2and.feature_port import RUST_BUILD_PATH_ENV, clear_rust_featurizer_cache
from s2and.model import _apply_dataset_name_count_semantics_for_prediction, _build_incremental_constraint_backend
from s2and.runtime import build_runtime_context, detect_rust_runtime_capabilities
from scripts.giant_block_cluster_retrieval_task import load_clusterer
from scripts.joint_safe_link_initial_only_rereview import (
    InitialOnlyRereviewDecision,
    read_initial_only_rereview_decisions,
    resolve_reviewed_safe_component_keys,
)
from scripts.reranker_dataset.raw_similarity import RawSimilarityFeatureCache, raw_similarity_features_by_component
from scripts.reranker_dataset.rows import generate_candidate_rows
from scripts.single_letter_reranker_utils import (
    DERIVED_FEATURE_COLUMNS,
    NAME_COUNT_RARITY_FEATURE_COLUMNS,
    RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    STRICT_RUST_NAME_COMPAT_ENV,
    QueryClusterStatsRequest,
    RerankerQueryCase,
    _resolve_dataset_file,
    _resolve_specter_file,
    build_component_index,
    compute_query_cluster_stats_batched,
    materialize_derived_rows,
    seed_constraint_bypass_component_keys,
)

SOURCE_BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
DEST_BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
SCRATCH_OUT = REPO_ROOT / "scratch" / "joint_safe_link_official_classic_20260428p"
TELEMETRY_DIR = REPO_ROOT / "scratch" / "promote_joint_safe_link_official_stack_20260428p_active_rebuild"
MODEL_PATH = REPO_ROOT / "data" / "production_model_v1.2.pickle"
CREATED_ON = "2026-04-21"
N_JOBS = 20
WINDOW_SIZE = retrieval_rank_limit_from_policy()
PAIR_BATCH_SIZE = 100_000
QUERY_BATCH_PAIR_LIMIT = 200_000
MAX_TOP_K = 25
MAX_EXEMPLARS = 4
SPOOL_DB_FILENAME = "group_rebuild_spool.sqlite3"
WORKER_SUMMARY_PREFIX = "dataset_summary_"
FILE_REPLACE_MAX_ATTEMPTS = 30
FILE_REPLACE_INITIAL_DELAY_SECONDS = 0.5
WINDOWS_ACCESS_VIOLATION_EXIT_CODES = {3221225477, -1073741819}
WINDOWS_FROM_DATASET_WORKER_DATASETS = {"s_lee"}


def _configure_official_rust_backend(*, n_jobs: int) -> None:
    """Force the official rebuild path onto the Rust backend or fail fast."""

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = "rust"
    os.environ[STRICT_RUST_NAME_COMPAT_ENV] = "1"
    thread_count = str(max(1, int(n_jobs)))
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["RAYON_NUM_THREADS"] = thread_count
    capabilities = detect_rust_runtime_capabilities()
    if not capabilities.core_runtime_available:
        raise RuntimeError(
            "Official joint safe-link rebuild requires the Rust backend, but the Rust runtime is unavailable "
            f"(reason={capabilities.reason})."
        )
    print(
        json.dumps(
            {
                "event": "official_rust_backend_required",
                "backend": "rust",
                "threads": int(thread_count),
                "strict_name_compat_selector": True,
                "capability_reason": capabilities.reason,
            }
        ),
        flush=True,
    )


FILE_REPLACE_MAX_DELAY_SECONDS = 5.0
OFFICIAL_CLASSIC_TRAIN_ROW_RELATIVE_PATH = Path("training") / (
    "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_"
    "neg100_plus_reviewed_splitpos_hardneg_rows.csv.gz"
)
S2AND_ROW_RELATIVE_PATH = Path("test") / "s2and_eval_rows.csv.gz"
S2AND_RESCUE_REVIEWED_ROW_RELATIVE_PATH = Path("test") / "s2and_rescue_reviewed_eval_rows.csv.gz"
ROW_RELATIVE_PATHS = (
    OFFICIAL_CLASSIC_TRAIN_ROW_RELATIVE_PATH,
    Path("calibration") / "classic_gate_possible_manual_w5_rows.csv.gz",
    S2AND_ROW_RELATIVE_PATH,
    Path("test") / "hwang_eval_rows.csv.gz",
    Path("test") / "s_park_eval_rows.csv.gz",
    Path("test") / "s_lee_eval_rows.csv.gz",
    Path("test") / "j_smith_eval_rows.csv.gz",
    Path("test") / "a_khan_eval_rows.csv.gz",
    Path("test") / "a_silva_eval_rows.csv.gz",
    Path("test") / "s_gupta_eval_rows.csv.gz",
    Path("test") / "training_s2and_source_reviewed_eval_rows.csv.gz",
    Path("test") / "s2and_extra_no_positive_eval_rows.csv.gz",
    S2AND_RESCUE_REVIEWED_ROW_RELATIVE_PATH,
)
STRATIFIED_SPLIT_SOURCE_KEY_BY_ROW_RELATIVE_PATH = {
    str(OFFICIAL_CLASSIC_TRAIN_ROW_RELATIVE_PATH).replace("/", "\\"): "s2and_rescue_reviewed_train",
    str(Path("calibration") / "classic_gate_possible_manual_w5_rows.csv.gz").replace("/", "\\"): ("calibration_source"),
    str(S2AND_ROW_RELATIVE_PATH).replace("/", "\\"): "s2and_eval",
    str(Path("test") / "hwang_eval_rows.csv.gz").replace("/", "\\"): "hwang_eval",
    str(Path("test") / "s_park_eval_rows.csv.gz").replace("/", "\\"): "s_park_eval",
    str(Path("test") / "s_lee_eval_rows.csv.gz").replace("/", "\\"): "s_lee_eval",
    str(Path("test") / "j_smith_eval_rows.csv.gz").replace("/", "\\"): "j_smith_eval",
    str(Path("test") / "a_khan_eval_rows.csv.gz").replace("/", "\\"): "a_khan_eval",
    str(Path("test") / "a_silva_eval_rows.csv.gz").replace("/", "\\"): "a_silva_eval",
    str(Path("test") / "s_gupta_eval_rows.csv.gz").replace("/", "\\"): "s_gupta_eval",
    str(Path("test") / "training_s2and_source_reviewed_eval_rows.csv.gz").replace("/", "\\"): (
        "training_s2and_source_reviewed_eval"
    ),
    str(Path("test") / "s2and_extra_no_positive_eval_rows.csv.gz").replace("/", "\\"): ("s2and_extra_no_positive_eval"),
    str(S2AND_RESCUE_REVIEWED_ROW_RELATIVE_PATH).replace("/", "\\"): "s2and_rescue_reviewed_eval",
}
S2AND_FULL_RELABEL_REVIEW_ROOT = REPO_ROOT / "scratch" / "s2and_full_relabel_20260424"
S2AND_FULL_RELABEL_PRE_FILTER_ROWS_PATH = (
    S2AND_FULL_RELABEL_REVIEW_ROOT / "bundle_backups_before_apply" / "s2and_eval_rows.csv.gz"
)
S2AND_RESIDUAL_LOO_SOURCE = "labeled_loo"
S2AND_RESIDUAL_LOO_SPLIT = "eval_loo"
HWANG_ROW_RELATIVE_PATH = Path("test") / "hwang_eval_rows.csv.gz"
HWANG_CLEAN_OVERRIDES_RELATIVE_PATH = Path("test") / "hwang_cleaned_eval_overrides.csv"
HWANG_CANDIDATE_LEVEL_MANIFEST_RELATIVE_PATH = Path("test") / "hwang_candidate_level_label_overrides.csv"
HWANG_CANDIDATE_LEVEL_SUMMARY_RELATIVE_PATH = Path("test") / "hwang_candidate_level_label_overrides_summary.json"
GIANT_DATASET_DIRS = {
    "a_khan": Path(r"D:\data\a_khan"),
    "a_silva": Path(r"D:\data\a_silva"),
    "h_wang": Path(r"D:\data\h_wang"),
    "j_smith": Path(r"D:\data\j_smith"),
    "s_gupta": Path(r"D:\data\s_gupta"),
    "s_lee": Path(r"D:\data\s_lee"),
    "s_park": Path(r"D:\data\s_park"),
}
GIANT_STEP2_DIRS = {
    "a_khan": REPO_ROOT / "scratch" / "a_khan_multi_letter_v12_15000",
    "a_silva": REPO_ROOT / "scratch" / "a_silva_multi_letter_v12_15000",
    "h_wang": REPO_ROOT / "scratch" / "h_wang_multi_letter_v12_15000",
    "j_smith": REPO_ROOT / "scratch" / "j_smith_multi_letter_v12_15000",
    "s_gupta": REPO_ROOT / "scratch" / "s_gupta_multi_letter_v12_15000",
    "s_lee": REPO_ROOT / "scratch" / "s_lee_multi_letter_v12_15000",
    "s_park": REPO_ROOT / "scratch" / "s_park_multi_letter_v12_15000",
}
LABELED_DATA_ROOT = REPO_ROOT / "data"


@dataclass
class DatasetResources:
    dataset_name: str
    dataset: ANDData
    runtime_context: Any
    constraint_backend: Any
    component_signatures: dict[str, list[str]]
    raw_paper_text_by_id: dict[str, str]
    raw_similarity_feature_cache: RawSimilarityFeatureCache = field(default_factory=RawSimilarityFeatureCache)


@dataclass
class FileRepairSummaryState:
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
class PreparedExistingGroup:
    source_path: str
    group_index: int
    dataset_name: str
    query_group_id: str
    rows_before_total: int
    positive_rows_before_total: int
    rows_after_window_cap: int
    positive_rows_after_window_cap: int
    rows_after_self_filter: int
    positive_rows_after_self_filter: int
    self_containing_candidate_component_keys: tuple[str, ...]
    self_containing_positive_component_keys: tuple[str, ...]
    residual_loo_positive_component_keys: tuple[str, ...]
    original_by_component: dict[str, dict[str, Any]]
    query_case: RerankerQueryCase
    query_view: str
    query_features: retrieval.QueryFeatures
    shortlist_component_keys: tuple[str, ...]
    retrieval_scores: dict[str, float]
    retrieval_ranks: dict[str, int]
    retrieval_window_state: dict[str, int]
    summary_by_component: dict[str, retrieval.ClusterSummary]
    raw_similarity_features_by_component: dict[str, dict[str, float]]
    stats_request: QueryClusterStatsRequest
    estimated_pair_count: int


@dataclass(frozen=True)
class EmptyExistingGroup:
    source_path: str
    group_index: int
    dataset_name: str
    query_group_id: str
    rows_before_total: int
    positive_rows_before_total: int
    rows_after_window_cap: int
    positive_rows_after_window_cap: int
    self_containing_candidate_component_keys: tuple[str, ...] = ()
    self_containing_positive_component_keys: tuple[str, ...] = ()
    residual_loo_positive_component_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class S2ANDFullRelabelDecision:
    """Manual S2AND review decision for one query group."""

    safe_component_keys: tuple[str, ...] | None
    split: str
    correction_type: str


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(float(value))


def _optional_str(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _has_value(value: Any) -> bool:
    return value not in (None, "")


def _normalize_bundle_relpath(path_like: str | Path) -> str:
    return str(path_like).replace("/", "\\")


def _is_s2and_eval_row_path(path_like: str | Path) -> bool:
    return _normalize_bundle_relpath(path_like) == _normalize_bundle_relpath(S2AND_ROW_RELATIVE_PATH)


def _is_s2and_rescue_reviewed_row_path(path_like: str | Path) -> bool:
    return _normalize_bundle_relpath(path_like) == _normalize_bundle_relpath(S2AND_RESCUE_REVIEWED_ROW_RELATIVE_PATH)


def _source_rows_path_for_rebuild(relative_path: Path) -> Path:
    """Return the row source used for one rebuild input.

    S2AND eval needs the manually reviewed pre-self-filter candidate surface;
    the active row file no longer contains many reviewed positive candidates.
    """

    if _is_s2and_eval_row_path(relative_path) and S2AND_FULL_RELABEL_PRE_FILTER_ROWS_PATH.exists():
        return S2AND_FULL_RELABEL_PRE_FILTER_ROWS_PATH
    return SOURCE_BUNDLE_ROOT / relative_path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def _worker_summary_path(dataset_name: str) -> Path:
    safe_name = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(dataset_name))
    return TELEMETRY_DIR / f"{WORKER_SUMMARY_PREFIX}{safe_name}.json"


def _split_component_keys(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _read_s2and_full_relabel_decisions(
    review_root: Path = S2AND_FULL_RELABEL_REVIEW_ROOT,
) -> dict[str, S2ANDFullRelabelDecision]:
    """Load complete S2AND manual relabel decisions."""

    review_path = review_root / "all_reviews_merged.tsv"
    if not review_path.exists():
        raise FileNotFoundError(f"Missing S2AND full relabel review file: {review_path}")
    decisions: dict[str, S2ANDFullRelabelDecision] = {}
    with review_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required_fields = {
            "query_case_id",
            "manual_assessment",
            "safe_positive_component_keys",
            "split",
            "correction_type",
        }
        missing = sorted(required_fields - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"S2AND full relabel review file missing columns: {missing}")
        for row in reader:
            query_group_id = str(row["query_case_id"])
            if query_group_id in decisions:
                raise ValueError(f"Duplicate S2AND relabel review for query_group_id={query_group_id!r}")
            if str(row["manual_assessment"]) == "impossible":
                safe_component_keys = None
            else:
                safe_component_keys = _split_component_keys(row.get("safe_positive_component_keys"))
            decisions[query_group_id] = S2ANDFullRelabelDecision(
                safe_component_keys=safe_component_keys,
                split=str(row["split"]),
                correction_type=str(row.get("correction_type") or ""),
            )
    if not decisions:
        raise ValueError(f"S2AND full relabel review file has no rows: {review_path}")
    return decisions


def _merge_initial_only_rereview_into_s2and_decisions(
    decisions: dict[str, S2ANDFullRelabelDecision],
    initial_only_decisions: dict[str, InitialOnlyRereviewDecision],
) -> dict[str, S2ANDFullRelabelDecision]:
    """Overlay model-visible initial-only decisions on promoted S2AND split metadata."""

    merged = dict(decisions)
    for query_group_id, rereview_decision in initial_only_decisions.items():
        current = merged.get(query_group_id)
        if current is None:
            continue
        if rereview_decision.action == "drop_query":
            safe_component_keys: tuple[str, ...] | None = None
        elif rereview_decision.action == "force_no_positive":
            safe_component_keys = ()
        elif rereview_decision.action == "candidate_positive":
            safe_component_keys = rereview_decision.safe_component_key_texts
        else:
            raise ValueError(
                f"Unsupported initial-only re-review action for {query_group_id!r}: {rereview_decision.action!r}"
            )
        merged[query_group_id] = S2ANDFullRelabelDecision(
            safe_component_keys=safe_component_keys,
            split=current.split,
            correction_type=f"initial_only_model_visible_{rereview_decision.reason_bucket}",
        )
    return merged


def _apply_initial_only_rereview_to_group(
    rows: list[dict[str, Any]],
    *,
    decision: InitialOnlyRereviewDecision,
) -> list[dict[str, Any]]:
    """Apply one collapsed model-visible initial-only decision to candidate rows."""

    if not rows:
        return []
    query_group_id = str(rows[0]["query_group_id"])
    if query_group_id != decision.query_group_id:
        raise ValueError(
            "Initial-only re-review decision was applied to the wrong query group: "
            f"rows={query_group_id!r} decision={decision.query_group_id!r}"
        )
    if decision.action == "drop_query":
        return []
    relabeled_rows = [dict(row) for row in rows]
    if decision.action == "force_no_positive":
        safe_keys: set[str] = set()
    elif decision.action == "candidate_positive":
        safe_keys = set(
            resolve_reviewed_safe_component_keys(
                decision.safe_component_key_texts,
                candidate_component_keys={str(row["candidate_component_key"]) for row in rows},
            )
        )
        if not safe_keys:
            return []
    else:
        raise ValueError(f"Unsupported initial-only re-review action: {decision.action!r}")
    for row in relabeled_rows:
        label = int(str(row["candidate_component_key"]) in safe_keys)
        row["label"] = str(label)
        if "binary_safe_link_target" in row:
            row["binary_safe_link_target"] = str(label)
    _refresh_group_label_metadata(relabeled_rows)
    return relabeled_rows


def _refresh_group_label_metadata(rows: list[dict[str, Any]]) -> None:
    """Refresh query-level positive metadata after candidate-level relabeling."""

    positive_rows = [row for row in rows if _to_int(row.get("label")) == 1]
    positive_keys = tuple(sorted(str(row["candidate_component_key"]) for row in positive_rows))
    positive_key_text = "|".join(positive_keys)
    positive_ranks = [_to_int(row.get("retrieval_rank")) for row in positive_rows]
    best_positive_rank = min(positive_ranks) if positive_ranks else ""
    for row in rows:
        row["positive_candidate_count"] = str(len(positive_keys))
        row["positive_candidate_keys"] = positive_key_text
        row["group_has_positive"] = str(int(bool(positive_keys)))
        row["best_positive_retrieval_rank"] = str(best_positive_rank) if best_positive_rank != "" else ""


def _normalize_letters(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalpha())


def _to_boolish(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return True
    if text in {"false", "f", "no", "n", "", "none", "nan"}:
        return False
    return bool(_to_int(value))


def _s2and_first_name_bucket(row: dict[str, Any]) -> str:
    token = _normalize_letters(row.get("query_first_token"))
    if not token:
        author = str(row.get("query_author", ""))
        letters: list[str] = []
        for character in author:
            if character.isalpha():
                letters.append(character.lower())
            elif letters:
                break
        token = "".join(letters)
    if str(row.get("query_view", "")) == "initial_only" or len(token) <= 1:
        return "single_letter_first"
    return "multi_letter_first"


def _s2and_stratum_key(row: dict[str, Any]) -> str:
    return (
        f"{row['source_stratum']}|has_pos={int(_to_boolish(row['has_positive_candidate']))}|"
        f"{row['positive_rank_bucket']}|{row['first_name_bucket']}|"
        f"multi_cand={int(_to_boolish(row['multiple_candidates']))}"
    )


def _group_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_group_id"]), []).append(row)
    return grouped


def _s2and_assignment_rows_from_active_rows(
    rows: list[dict[str, Any]],
    *,
    decisions: dict[str, S2ANDFullRelabelDecision],
) -> list[dict[str, Any]]:
    """Build promoted split assignment rows for active S2AND eval rows."""

    assignment_rows: list[dict[str, Any]] = []
    for query_group_id, group in sorted(_group_rows_by_query(rows).items()):
        decision = decisions.get(query_group_id)
        if decision is None:
            raise KeyError(f"S2AND relabel decision missing for active query_group_id={query_group_id!r}")
        if decision.safe_component_keys is None:
            raise ValueError(f"Impossible S2AND query survived active rows: {query_group_id!r}")
        ordered = sorted(
            group, key=lambda row: (_to_int(row.get("retrieval_rank")), str(row["candidate_component_key"]))
        )
        first = ordered[0]
        positive_rows = [row for row in ordered if _to_int(row.get("label")) == 1]
        has_positive = bool(positive_rows)
        min_retrieval_rank = min(_to_int(row.get("retrieval_rank")) for row in ordered)
        max_retrieval_rank = max(_to_int(row.get("retrieval_rank")) for row in ordered)
        if has_positive:
            min_positive_rank: int | str = min(_to_int(row.get("retrieval_rank")) for row in positive_rows)
            positive_first = int(min_positive_rank) == int(min_retrieval_rank)
            positive_rank_bucket = "positive_first" if positive_first else "positive_not_first"
        else:
            min_positive_rank = ""
            positive_first = False
            positive_rank_bucket = "no_positive"
        candidate_count = len({str(row["candidate_component_key"]) for row in ordered})
        payload: dict[str, Any] = {
            "query_group_id": str(query_group_id),
            "base_group_id": str(first.get("base_group_id") or query_group_id),
            "dataset": str(first.get("dataset", "")),
            "source_key": "s2and_eval",
            "source_kind": "public_test",
            "source_priority": "1",
            "query_source": str(first.get("query_source", "")),
            "query_view": str(first.get("query_view", "")),
            "support_type": str(first.get("support_type", "")),
            "source_stratum": "s2and_block",
            "has_positive_candidate": str(bool(has_positive)),
            "positive_first": str(bool(positive_first)),
            "positive_rank_bucket": positive_rank_bucket,
            "raw_has_positive_candidate": str(bool(has_positive)),
            "raw_positive_first": str(bool(positive_first)),
            "manual_safe_target": str(int(has_positive)),
            "correction_type": str(decision.correction_type),
            "first_name_bucket": _s2and_first_name_bucket(first),
            "multiple_candidates": str(bool(candidate_count > 1)),
            "candidate_count": str(candidate_count),
            "min_positive_rank": str(min_positive_rank),
            "min_retrieval_rank": str(min_retrieval_rank),
            "max_retrieval_rank": str(max_retrieval_rank),
            "positive_candidate_rows": str(len(positive_rows)),
            "split": str(decision.split),
        }
        payload["stratum_key"] = _s2and_stratum_key(payload)
        assignment_rows.append(payload)
    return assignment_rows


def _refresh_stratum_balance_rows(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in assignments:
        grouped.setdefault(str(row["stratum_key"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for stratum_key, group in sorted(grouped.items()):
        total = len(group)
        split_counts = Counter(str(row["split"]) for row in group)
        payload: dict[str, Any] = {"stratum_key": stratum_key, "total": str(total)}
        missing_split_count = 0
        for split, target in (
            ("calibration_fit", 0.25),
            ("calibration_check", 0.25),
            ("test", 0.50),
        ):
            count = int(split_counts.get(split, 0))
            if count == 0:
                missing_split_count += 1
            expected = total * float(target)
            payload[f"{split}_count"] = str(count)
            payload[f"{split}_expected"] = str(expected)
            payload[f"{split}_delta"] = str(count - expected)
        payload["missing_split_count"] = str(missing_split_count)
        rows.append(payload)
    return rows


def _max_abs_assignment_marginal_delta(assignments: list[dict[str, Any]], factor: str) -> float:
    total = max(1, len(assignments))
    full_counts = Counter(str(row.get(factor, "")) for row in assignments)
    max_delta = 0.0
    for split in ("calibration_fit", "calibration_check", "test"):
        split_rows = [row for row in assignments if str(row.get("split")) == split]
        split_total = max(1, len(split_rows))
        split_counts = Counter(str(row.get(factor, "")) for row in split_rows)
        for value in set(full_counts) | set(split_counts):
            max_delta = max(
                max_delta,
                abs(float(split_counts.get(value, 0)) / split_total - float(full_counts.get(value, 0)) / total),
            )
    return float(max_delta)


def _apply_s2and_full_relabel_to_group(
    rows: list[dict[str, Any]],
    *,
    decisions: dict[str, S2ANDFullRelabelDecision],
) -> list[dict[str, Any]]:
    """Apply S2AND manual review labels to one staged query group."""

    if not rows:
        return []
    query_group_id = str(rows[0]["query_group_id"])
    decision = decisions.get(query_group_id)
    if decision is None:
        raise KeyError(f"S2AND relabel decision missing for query_group_id={query_group_id!r}")
    if decision.safe_component_keys is None:
        return []
    candidate_keys = {str(row["candidate_component_key"]) for row in rows}
    missing_safe_keys = sorted(set(decision.safe_component_keys) - candidate_keys)
    if missing_safe_keys:
        raise ValueError(
            "S2AND reviewed safe candidates are absent from the pre-filter row surface: "
            f"query_group_id={query_group_id!r} missing={missing_safe_keys}"
        )
    safe_keys = set(decision.safe_component_keys)
    relabeled_rows = [dict(row) for row in rows]
    for row in relabeled_rows:
        label = int(str(row["candidate_component_key"]) in safe_keys)
        row["label"] = str(label)
        if "binary_safe_link_target" in row:
            row["binary_safe_link_target"] = str(label)
    _refresh_group_label_metadata(relabeled_rows)
    return relabeled_rows


def _source_path_placeholders(source_paths: list[str] | tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    normalized_paths = tuple(str(path) for path in source_paths)
    if not normalized_paths:
        raise ValueError("At least one source path is required.")
    placeholders = ", ".join("?" for _ in normalized_paths)
    return placeholders, normalized_paths


def _load_selected_row_headers(
    selected_row_paths: tuple[Path, ...],
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    fieldnames_by_path: dict[str, list[str]] = {}
    file_summaries: dict[str, FileRepairSummaryState] = {}
    ordered_source_paths: list[str] = []
    for relative_path in selected_row_paths:
        source_path = _normalize_bundle_relpath(relative_path)
        ordered_source_paths.append(source_path)
        file_summaries[source_path] = FileRepairSummaryState(path=source_path)
        input_path = _source_rows_path_for_rebuild(relative_path)
        with gzip.open(input_path, "rt", encoding="utf-8", newline="") as src_handle:
            reader = csv.DictReader(src_handle)
            fieldnames = [str(value) for value in reader.fieldnames or []]
            if not fieldnames:
                raise ValueError(f"CSV has no header: {input_path}")
            fieldnames_by_path[source_path] = _fieldnames_with_materialized_derived_columns(fieldnames)
    return fieldnames_by_path, file_summaries, ordered_source_paths


def _fieldnames_with_materialized_derived_columns(fieldnames: list[str]) -> list[str]:
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


def _hydrate_stage_summaries_from_spool(
    connection: sqlite3.Connection,
    *,
    file_summaries: dict[str, FileRepairSummaryState],
    ordered_source_paths: list[str],
) -> None:
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


def _load_staged_input_groups_from_spool(
    connection: sqlite3.Connection,
    *,
    selected_row_paths: tuple[Path, ...],
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    fieldnames_by_path, file_summaries, ordered_source_paths = _load_selected_row_headers(selected_row_paths)
    _hydrate_stage_summaries_from_spool(
        connection,
        file_summaries=file_summaries,
        ordered_source_paths=ordered_source_paths,
    )
    return fieldnames_by_path, file_summaries, ordered_source_paths


def _selected_datasets(connection: sqlite3.Connection, *, ordered_source_paths: list[str]) -> list[str]:
    placeholders, params = _source_path_placeholders(ordered_source_paths)
    rows = connection.execute(
        f"""
        SELECT dataset_name
        FROM staged_groups
        WHERE source_path IN ({placeholders})
        GROUP BY dataset_name
        ORDER BY COUNT(*) DESC, dataset_name ASC
        """,
        params,
    )
    return [str(dataset_name) for (dataset_name,) in rows]


def _dataset_rebuild_is_complete(
    connection: sqlite3.Connection,
    *,
    dataset_name: str,
    ordered_source_paths: list[str],
) -> bool:
    placeholders, params = _source_path_placeholders(ordered_source_paths)
    staged_total, rebuilt_total = connection.execute(
        f"""
        SELECT
            COUNT(*) AS staged_total,
            SUM(CASE WHEN rebuilt.source_path IS NOT NULL THEN 1 ELSE 0 END) AS rebuilt_total
        FROM staged_groups AS staged
        LEFT JOIN rebuilt_groups AS rebuilt
          ON rebuilt.source_path = staged.source_path
         AND rebuilt.group_index = staged.group_index
        WHERE staged.dataset_name = ?
          AND staged.source_path IN ({placeholders})
        """,
        (str(dataset_name), *params),
    ).fetchone()
    return int(_to_int(staged_total)) > 0 and int(_to_int(rebuilt_total)) == int(_to_int(staged_total))


def _is_retryable_replace_error(exc: OSError) -> bool:
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in {5, 32}


def _replace_with_retry(
    temp_path: Path,
    output_path: Path,
    *,
    max_attempts: int = FILE_REPLACE_MAX_ATTEMPTS,
    initial_delay_seconds: float = FILE_REPLACE_INITIAL_DELAY_SECONDS,
    max_delay_seconds: float = FILE_REPLACE_MAX_DELAY_SECONDS,
) -> None:
    delay_seconds = float(initial_delay_seconds)
    for attempt in range(1, int(max_attempts) + 1):
        try:
            temp_path.replace(output_path)
            return
        except OSError as exc:
            if not _is_retryable_replace_error(exc) or attempt >= int(max_attempts):
                raise
            print(
                json.dumps(
                    {
                        "event": "replace_retry",
                        "path": _display_path(output_path),
                        "attempt": int(attempt),
                        "max_attempts": int(max_attempts),
                        "delay_seconds": float(delay_seconds),
                        "winerror": getattr(exc, "winerror", None),
                        "message": str(exc),
                    }
                ),
                flush=True,
            )
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.5, float(max_delay_seconds))


def _compress_rows(rows: list[dict[str, Any]]) -> bytes:
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return zlib.compress(payload, level=1)


def _decompress_rows(blob: bytes) -> list[dict[str, Any]]:
    return list(json.loads(zlib.decompress(blob).decode("utf-8")))


def _select_metadata_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metadata_fields = (
        "query_signature_id",
        "query_id",
        "block_key",
        "query_source",
        "split",
        "support_type",
        "sampling_info_bucket",
        "supervision_type",
        "candidate_cluster_id",
        "family_id",
        "block_size",
        "component_size",
    )
    return max(
        rows,
        key=lambda row: sum(1 for field in metadata_fields if _has_value(row.get(field))),
    )


def _infer_query_signature_id(rows: list[dict[str, Any]]) -> str:
    for field_name in ("query_signature_id", "query_id"):
        for row in rows:
            value = row.get(field_name)
            if _has_value(value):
                return str(value)
    query_group_id = str(rows[0].get("query_group_id", ""))
    query_group_parts = query_group_id.rsplit(":", 2)
    if len(query_group_parts) == 3 and _has_value(query_group_parts[1]):
        return str(query_group_parts[1])
    base_group_id = str(rows[0].get("base_group_id", ""))
    if ":" in base_group_id:
        candidate = base_group_id.rsplit(":", 1)[-1]
        if _has_value(candidate):
            return str(candidate)
    return ""


def _build_query_case(rows: list[dict[str, Any]], *, metadata_row: dict[str, Any] | None = None) -> RerankerQueryCase:
    if metadata_row is None:
        metadata_row = _select_metadata_row(rows)
    positive_component_keys = frozenset(
        str(row["candidate_component_key"]) for row in rows if _to_int(row.get("label")) == 1
    )
    return RerankerQueryCase(
        source=str(metadata_row["source"]),
        dataset=str(metadata_row["dataset"]),
        query_id=str(metadata_row["query_id"]),
        query_signature_id=str(metadata_row["query_signature_id"]),
        block_key=str(metadata_row["block_key"]),
        positive_component_keys=positive_component_keys,
        support_type=str(metadata_row["support_type"]),
        block_size=_to_int(metadata_row.get("block_size")),
        component_size=_to_int(metadata_row.get("component_size")),
        sampling_info_bucket=str(metadata_row["sampling_info_bucket"]),
        query_source=str(metadata_row["query_source"]),
        normalized_orcid=_optional_str(metadata_row.get("_audit_normalized_orcid")),
        orcid_group_size=_optional_int(metadata_row.get("_audit_orcid_group_size")),
        orcid_group_size_bucket=_optional_str(metadata_row.get("_audit_orcid_group_size_bucket")),
        split=str(metadata_row["split"]),
        supervision_type=str(metadata_row["supervision_type"]),
        query_in_seed_before_holdout=bool(_to_int(metadata_row.get("query_in_seed_before_holdout"))),
        natural_query_view=_optional_str(metadata_row.get("natural_query_view")),
    )


def _load_labeled_dataset(dataset_name: str) -> ANDData:
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = "rust"
    os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
    os.environ["RAYON_NUM_THREADS"] = str(N_JOBS)
    return ANDData(
        signatures=_resolve_dataset_file(
            LABELED_DATA_ROOT,
            dataset_name,
            f"{dataset_name}_signatures.json",
            "signatures.json",
        ),
        papers=_resolve_dataset_file(LABELED_DATA_ROOT, dataset_name, f"{dataset_name}_papers.json", "papers.json"),
        name=dataset_name,
        mode="inference",
        specter_embeddings=_resolve_specter_file(LABELED_DATA_ROOT, dataset_name),
        clusters=_resolve_dataset_file(
            LABELED_DATA_ROOT,
            dataset_name,
            f"{dataset_name}_clusters.json",
            "clusters.json",
        ),
        block_type="s2",
        n_jobs=N_JOBS,
        load_name_counts=True,
        preprocess=True,
        random_seed=13,
        name_tuples="filtered",
        use_orcid_id=False,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )


def _load_giant_dataset(dataset_name: str) -> ANDData:
    data_dir = GIANT_DATASET_DIRS[dataset_name]
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = "rust"
    os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
    os.environ["RAYON_NUM_THREADS"] = str(N_JOBS)
    return ANDData(
        signatures=str(data_dir / "signatures.json"),
        papers=str(data_dir / "papers.json"),
        name=dataset_name,
        mode="inference",
        specter_embeddings=str(data_dir / "specter.pickle"),
        clusters=None,
        cluster_seeds=str(data_dir / "cluster_seeds.json"),
        altered_cluster_signatures=None,
        block_type="s2",
        n_jobs=N_JOBS,
        load_name_counts=True,
        preprocess=True,
        random_seed=13,
        name_tuples="filtered",
        use_orcid_id=False,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )


def _load_giant_component_signatures(dataset_name: str) -> dict[str, list[str]]:
    predicted_clusters = json.loads(
        (GIANT_STEP2_DIRS[dataset_name] / "predicted_clusters.json").read_text(encoding="utf-8")
    )
    component_signatures: dict[str, list[str]] = {}
    for _subblock_key, cluster_map in predicted_clusters.items():
        for component_key, signature_ids in dict(cluster_map).items():
            component_signatures[str(component_key)] = [str(value) for value in signature_ids]
    return component_signatures


class DatasetResourceManager:
    def __init__(self, *, clusterer: Any):
        self.clusterer = clusterer
        self._resources: dict[str, DatasetResources] = {}

    def get(self, dataset_name: str) -> DatasetResources:
        dataset_name = str(dataset_name)
        existing = self._resources.get(dataset_name)
        if existing is not None:
            return existing
        load_started = time.perf_counter()
        print(json.dumps({"event": "dataset_load_start", "dataset": dataset_name}), flush=True)
        if dataset_name in GIANT_DATASET_DIRS:
            dataset = _load_giant_dataset(dataset_name)
            component_signatures = _load_giant_component_signatures(dataset_name)
        else:
            dataset = _load_labeled_dataset(dataset_name)
            component_signatures, _block_to_component_keys, _block_sizes, _missing = build_component_index(dataset)
        raw_paper_text_by_id = _load_raw_paper_text_by_id(
            dataset_name,
            needed_paper_ids={str(paper_id) for paper_id in dataset.papers},
        )
        _apply_dataset_name_count_semantics_for_prediction(self.clusterer, dataset)
        runtime_context = build_runtime_context(f"joint_safe_link_{dataset_name}_constraint_repair")
        constraint_backend = _build_incremental_constraint_backend(
            dataset,
            use_default_constraints_as_supervision=self.clusterer.use_default_constraints_as_supervision,
            runtime_context=runtime_context,
            use_cache=self.clusterer.use_cache,
            suppress_orcid=True,
        )
        resources = DatasetResources(
            dataset_name=dataset_name,
            dataset=dataset,
            runtime_context=runtime_context,
            constraint_backend=constraint_backend,
            component_signatures=component_signatures,
            raw_paper_text_by_id=raw_paper_text_by_id,
        )
        self._resources[dataset_name] = resources
        print(
            json.dumps(
                {
                    "event": "dataset_load_complete",
                    "dataset": dataset_name,
                    "seconds": round(time.perf_counter() - load_started, 3),
                    "signatures": len(dataset.signatures),
                    "raw_paper_texts": len(raw_paper_text_by_id),
                    "components": len(component_signatures),
                }
            ),
            flush=True,
        )
        return resources

    def release_all(self) -> list[str]:
        released = list(self._resources)
        self._resources.clear()
        clear_rust_featurizer_cache()
        gc.collect()
        return released


def _component_signature_ids(resources: DatasetResources, *, component_key: str, query_signature_id: str) -> list[str]:
    signature_ids = resources.component_signatures.get(str(component_key))
    if signature_ids is None:
        raise KeyError(f"Unknown component {component_key!r} in dataset {resources.dataset_name!r}")
    return [signature_id for signature_id in signature_ids if str(signature_id) != str(query_signature_id)]


def _raw_papers_path_for_dataset(dataset_name: str) -> Path:
    """Return the raw papers JSON path for the official rebuild dataset."""

    if dataset_name in GIANT_DATASET_DIRS:
        return GIANT_DATASET_DIRS[dataset_name] / "papers.json"
    return Path(_resolve_dataset_file(LABELED_DATA_ROOT, dataset_name, f"{dataset_name}_papers.json", "papers.json"))


def _load_raw_paper_text_by_id(dataset_name: str, *, needed_paper_ids: set[str]) -> dict[str, str]:
    """Load title-plus-abstract text for papers retained by the active ANDData object."""

    if not needed_paper_ids:
        return {}
    raw_papers_path = _raw_papers_path_for_dataset(dataset_name)
    with raw_papers_path.open("r", encoding="utf-8") as handle:
        raw_papers = json.load(handle)
    text_by_id: dict[str, str] = {}
    for paper_id in needed_paper_ids:
        paper = raw_papers.get(str(paper_id))
        if not isinstance(paper, dict):
            continue
        text_by_id[str(paper_id)] = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    return text_by_id


def _raw_similarity_features_by_component(
    resources: DatasetResources,
    *,
    query_signature_id: str,
    candidate_signature_ids_by_component: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Compatibility wrapper around the shared raw-similarity implementation."""

    return raw_similarity_features_by_component(
        dataset=resources.dataset,
        query_signature_id=str(query_signature_id),
        candidate_signature_ids_by_component=candidate_signature_ids_by_component,
        raw_paper_text_by_id=resources.raw_paper_text_by_id,
        cache=resources.raw_similarity_feature_cache,
    )


def _component_contains_query_signature(
    resources: DatasetResources,
    *,
    component_key: str,
    query_signature_id: str,
) -> bool:
    signature_ids = resources.component_signatures.get(str(component_key))
    if signature_ids is None:
        raise KeyError(f"Unknown component {component_key!r} in dataset {resources.dataset_name!r}")
    return str(query_signature_id) in {str(signature_id) for signature_id in signature_ids}


def _should_keep_self_containing_positive_as_residual_loo(*, source_path: str, row: dict[str, Any]) -> bool:
    """Return whether a self-containing positive row should be materialized as residual LOO."""

    return _to_int(row.get("label")) == 1 and (
        _is_s2and_eval_row_path(source_path)
        or _is_s2and_rescue_reviewed_row_path(source_path)
        or str(row.get("source", "")) in {"labeled_loo", "s2and_rescue_manual_review"}
    )


def _component_summary(
    resources: DatasetResources,
    *,
    component_key: str,
    block_key: str,
    query_signature_id: str,
    feature_cache: dict[str, retrieval.QueryFeatures],
    full_summary_cache: dict[str, retrieval.ClusterSummary],
) -> retrieval.ClusterSummary:
    signature_ids = resources.component_signatures.get(str(component_key))
    if signature_ids is None:
        raise KeyError(f"Unknown component {component_key!r} in dataset {resources.dataset_name!r}")
    if str(query_signature_id) not in signature_ids:
        cached = full_summary_cache.get(str(component_key))
        if cached is not None:
            return cached
        summary_signature_ids = list(signature_ids)
    else:
        summary_signature_ids = [
            signature_id for signature_id in signature_ids if str(signature_id) != str(query_signature_id)
        ]
    if "::" in str(component_key):
        summary_block_key, cluster_id = str(component_key).split("::", 1)
    else:
        summary_block_key = str(block_key)
        cluster_id = str(component_key)
    summary = retrieval.build_cluster_summary(
        dataset=resources.dataset,
        block_key=str(summary_block_key),
        cluster_id=str(cluster_id),
        component_key=str(component_key),
        signature_ids=summary_signature_ids,
        max_exemplars=MAX_EXEMPLARS,
        feature_cache=feature_cache,
        orcid_enabled=False,
    )
    if str(query_signature_id) not in signature_ids:
        full_summary_cache[str(component_key)] = summary
    return summary


def _connect_spool_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-200000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS staged_groups (
            source_path TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            dataset_name TEXT NOT NULL,
            query_group_id TEXT NOT NULL,
            rows_before_total INTEGER NOT NULL,
            positive_rows_before_total INTEGER NOT NULL,
            rows_after_window_cap INTEGER NOT NULL,
            positive_rows_after_window_cap INTEGER NOT NULL,
            rows_blob BLOB NOT NULL,
            PRIMARY KEY (source_path, group_index)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_staged_groups_dataset ON staged_groups (dataset_name, source_path, group_index)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rebuilt_groups (
            source_path TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            rows_blob BLOB NOT NULL,
            PRIMARY KEY (source_path, group_index)
        )
        """
    )
    return connection


def _dataset_result_file_summaries(
    connection: sqlite3.Connection, *, dataset_name: str
) -> dict[str, FileRepairSummaryState]:
    source_paths = [
        str(source_path)
        for (source_path,) in connection.execute(
            "SELECT DISTINCT source_path FROM staged_groups WHERE dataset_name = ? ORDER BY source_path",
            (str(dataset_name),),
        )
    ]
    return {source_path: FileRepairSummaryState(path=source_path) for source_path in source_paths}


def _write_worker_summary(*, dataset_name: str, file_summaries: dict[str, FileRepairSummaryState]) -> Path:
    payload = {
        "dataset": str(dataset_name),
        "files": [file_summaries[source_path].to_result_payload() for source_path in sorted(file_summaries)],
    }
    output_path = _worker_summary_path(dataset_name)
    _write_json(output_path, payload)
    return output_path


def _merge_worker_summary(
    *,
    dataset_name: str,
    file_summary_states: dict[str, FileRepairSummaryState],
    summary_path: Path,
) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(payload.get("dataset")) != str(dataset_name):
        raise ValueError(f"Worker summary dataset mismatch: expected {dataset_name!r}, got {payload.get('dataset')!r}")
    for file_payload in payload.get("files", []):
        source_path = str(file_payload["path"])
        file_summary_states[source_path].merge_result_payload(dict(file_payload))


def _run_dataset_worker(
    *,
    spool_db_path: Path,
    dataset_name: str,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_top_k: int,
) -> None:
    connection = _connect_spool_db(spool_db_path)
    try:
        clusterer = load_clusterer(MODEL_PATH, n_jobs=N_JOBS)
        clusterer.use_cache = False
        resource_manager = DatasetResourceManager(clusterer=clusterer)
        file_summaries = _dataset_result_file_summaries(connection, dataset_name=str(dataset_name))
        _process_dataset_groups(
            connection=connection,
            file_summaries=file_summaries,
            dataset_name=str(dataset_name),
            clusterer=clusterer,
            resource_manager=resource_manager,
            pair_batch_size=int(pair_batch_size),
            query_batch_pair_limit=int(query_batch_pair_limit),
            max_top_k=int(max_top_k),
            release_resources=False,
        )
        summary_path = _write_worker_summary(dataset_name=str(dataset_name), file_summaries=file_summaries)
        print(
            json.dumps(
                {
                    "event": "dataset_worker_summary",
                    "dataset": str(dataset_name),
                    "path": str(summary_path.relative_to(REPO_ROOT)).replace("/", "\\"),
                }
            ),
            flush=True,
        )
    finally:
        connection.close()


def _run_dataset_worker_subprocess(
    *,
    dataset_name: str,
    spool_db_path: Path,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    file_summary_states: dict[str, FileRepairSummaryState],
) -> None:
    summary_path = _worker_summary_path(dataset_name)
    if summary_path.exists():
        summary_path.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-bundle-root",
        str(SOURCE_BUNDLE_ROOT),
        "--dest-bundle-root",
        str(DEST_BUNDLE_ROOT),
        "--scratch-out",
        str(SCRATCH_OUT),
        "--telemetry-dir",
        str(TELEMETRY_DIR),
        "--spool-db-path",
        str(spool_db_path),
        "--worker-dataset",
        str(dataset_name),
        "--pair-batch-size",
        str(int(pair_batch_size)),
        "--query-batch-pair-limit",
        str(int(query_batch_pair_limit)),
    ]
    env = os.environ.copy()
    initial_build_path = None
    if os.name == "nt" and str(dataset_name) in WINDOWS_FROM_DATASET_WORKER_DATASETS:
        initial_build_path = "from_dataset"
        env[RUST_BUILD_PATH_ENV] = initial_build_path

    print(
        json.dumps(
            {
                "event": "dataset_worker_start",
                "dataset": str(dataset_name),
                "command": command,
                "rust_build_path": initial_build_path or env.get(RUST_BUILD_PATH_ENV, "default"),
            }
        ),
        flush=True,
    )
    try:
        subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=env)
    except subprocess.CalledProcessError as exc:
        if (
            int(exc.returncode) not in WINDOWS_ACCESS_VIOLATION_EXIT_CODES
            or env.get(RUST_BUILD_PATH_ENV) == "from_dataset"
        ):
            raise
        retry_env = os.environ.copy()
        retry_env[RUST_BUILD_PATH_ENV] = "from_dataset"
        print(
            json.dumps(
                {
                    "event": "dataset_worker_retry_after_native_access_violation",
                    "dataset": str(dataset_name),
                    "returncode": int(exc.returncode),
                    "rust_build_path": "from_dataset",
                }
            ),
            flush=True,
        )
        subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=retry_env)
    if not summary_path.exists():
        raise FileNotFoundError(f"Worker summary missing for dataset={dataset_name!r}: {summary_path}")
    _merge_worker_summary(
        dataset_name=str(dataset_name),
        file_summary_states=file_summary_states,
        summary_path=summary_path,
    )
    print(
        json.dumps(
            {
                "event": "dataset_worker_complete",
                "dataset": str(dataset_name),
                "summary_path": _display_path(summary_path),
            }
        ),
        flush=True,
    )


def _stage_input_groups(
    *,
    connection: sqlite3.Connection,
    selected_row_paths: tuple[Path, ...],
    limit_groups_per_file: int | None,
) -> tuple[dict[str, list[str]], dict[str, FileRepairSummaryState], list[str]]:
    fieldnames_by_path, file_summaries, ordered_source_paths = _load_selected_row_headers(selected_row_paths)
    initial_only_rereview_decisions = read_initial_only_rereview_decisions()
    s2and_relabel_decisions = (
        _merge_initial_only_rereview_into_s2and_decisions(
            _read_s2and_full_relabel_decisions(),
            initial_only_rereview_decisions,
        )
        if any(
            _is_s2and_eval_row_path(path) and S2AND_FULL_RELABEL_PRE_FILTER_ROWS_PATH.exists()
            for path in selected_row_paths
        )
        else {}
    )
    insert_rows: list[tuple[Any, ...]] = []
    flush_every = 500

    for relative_path in selected_row_paths:
        source_path = _normalize_bundle_relpath(relative_path)
        staged_group_count = 0
        input_path = _source_rows_path_for_rebuild(relative_path)
        relabel_s2and = _is_s2and_eval_row_path(relative_path) and input_path == S2AND_FULL_RELABEL_PRE_FILTER_ROWS_PATH
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
                    group_rows = _apply_initial_only_rereview_to_group(
                        group_rows,
                        decision=initial_only_decision,
                    )
                    if not group_rows:
                        return
                elif relabel_group_s2and:
                    group_rows = _apply_s2and_full_relabel_to_group(
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
                window_filter = apply_retrieval_rank_filter(group_rows, retrieval_rank_limit=int(WINDOW_SIZE))
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
                        sqlite3.Binary(_compress_rows(rows_after_window_cap_rows)),
                    )
                )
                if len(insert_rows) >= flush_every:
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
                        insert_rows,
                    )
                    connection.commit()
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

    if insert_rows:
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
            insert_rows,
        )
        connection.commit()
    return fieldnames_by_path, file_summaries, ordered_source_paths


def _prepare_existing_group(
    *,
    source_path: str,
    group_index: int,
    resources: DatasetResources,
    rows: list[dict[str, Any]],
    rows_before_total: int,
    positive_rows_before_total: int,
    rows_after_window_cap: int,
    positive_rows_after_window_cap: int,
    feature_cache: dict[str, retrieval.QueryFeatures],
    full_summary_cache: dict[str, retrieval.ClusterSummary],
) -> PreparedExistingGroup | EmptyExistingGroup:
    ordered_rows = sorted(
        rows,
        key=lambda row: (_to_int(row.get("retrieval_rank")), str(row.get("candidate_component_key"))),
    )
    original_by_component: dict[str, dict[str, Any]] = {}
    for row in ordered_rows:
        component_key = str(row["candidate_component_key"])
        if component_key in original_by_component:
            raise ValueError(f"Duplicate candidate_component_key within query group {row.get('query_group_id')!r}")
        original_by_component[component_key] = dict(row)
    metadata_row = dict(_select_metadata_row(ordered_rows))
    query_signature_id = str(metadata_row.get("query_signature_id") or "")
    if not query_signature_id:
        inferred_query_signature_id = _infer_query_signature_id(ordered_rows)
        if inferred_query_signature_id in resources.dataset.signatures:
            query_signature_id = inferred_query_signature_id
            metadata_row["query_signature_id"] = query_signature_id
            if not _has_value(metadata_row.get("query_id")):
                metadata_row["query_id"] = query_signature_id
            if not _has_value(metadata_row.get("query_source")):
                base_group_id = str(metadata_row.get("base_group_id", ""))
                if ":" in base_group_id:
                    metadata_row["query_source"] = base_group_id.split(":", 1)[0]
            if not _has_value(metadata_row.get("block_key")):
                signature_block_key = getattr(resources.dataset, "signature_to_block", {}).get(query_signature_id)
                if not _has_value(signature_block_key):
                    signature_block_key = getattr(
                        resources.dataset.signatures[query_signature_id],
                        "author_info_block",
                        "",
                    )
                metadata_row["block_key"] = signature_block_key
        else:
            query_signature_id = ""
    if not query_signature_id:
        raise ValueError(
            f"Query group {metadata_row.get('query_group_id')!r} in dataset {metadata_row.get('dataset')!r} "
            "has no populated query_signature_id in any row."
        )
    contains_query_by_component = {
        str(row["candidate_component_key"]): _component_contains_query_signature(
            resources,
            component_key=str(row["candidate_component_key"]),
            query_signature_id=query_signature_id,
        )
        for row in ordered_rows
    }
    residual_loo_positive_component_keys = tuple(
        str(row["candidate_component_key"])
        for row in ordered_rows
        if contains_query_by_component[str(row["candidate_component_key"])]
        and _should_keep_self_containing_positive_as_residual_loo(source_path=source_path, row=row)
    )
    residual_loo_positive_key_set = set(residual_loo_positive_component_keys)
    dropped_self_containing_rows = [
        row
        for row in ordered_rows
        if contains_query_by_component[str(row["candidate_component_key"])]
        and str(row["candidate_component_key"]) not in residual_loo_positive_key_set
    ]
    self_containing_candidate_component_keys = tuple(
        str(row["candidate_component_key"]) for row in dropped_self_containing_rows
    )
    self_containing_positive_component_keys = tuple(
        str(row["candidate_component_key"]) for row in dropped_self_containing_rows if _to_int(row.get("label")) == 1
    )
    filtered_rows = [
        row
        for row in ordered_rows
        if str(row["candidate_component_key"]) not in set(self_containing_candidate_component_keys)
    ]
    if not filtered_rows:
        return EmptyExistingGroup(
            source_path=str(source_path),
            group_index=int(group_index),
            dataset_name=str(resources.dataset_name),
            query_group_id=str(metadata_row["query_group_id"]),
            rows_before_total=int(rows_before_total),
            positive_rows_before_total=int(positive_rows_before_total),
            rows_after_window_cap=int(rows_after_window_cap),
            positive_rows_after_window_cap=int(positive_rows_after_window_cap),
            self_containing_candidate_component_keys=self_containing_candidate_component_keys,
            self_containing_positive_component_keys=self_containing_positive_component_keys,
            residual_loo_positive_component_keys=residual_loo_positive_component_keys,
        )

    query_view_values = {str(row.get("query_view")) for row in filtered_rows}
    if len(query_view_values) != 1:
        raise ValueError(
            f"Expected exactly one query_view per query group {metadata_row.get('query_group_id')!r}, "
            f"got {sorted(query_view_values)}"
        )
    query_view = next(iter(query_view_values))
    if residual_loo_positive_component_keys:
        metadata_row["source"] = S2AND_RESIDUAL_LOO_SOURCE
        metadata_row["split"] = S2AND_RESIDUAL_LOO_SPLIT
    query_case = _build_query_case(filtered_rows, metadata_row=metadata_row)
    base_query = retrieval.extract_query_features(
        resources.dataset,
        query_signature_id,
        feature_cache=feature_cache,
        orcid_enabled=False,
    )
    query_features = retrieval.mask_query_features(base_query, query_view, orcid_enabled=False)
    shortlist_component_keys = tuple(str(row["candidate_component_key"]) for row in filtered_rows)
    retrieval_ranks = {str(row["candidate_component_key"]): _to_int(row.get("retrieval_rank")) for row in filtered_rows}
    retrieval_scores = {
        str(row["candidate_component_key"]): _to_float(row.get("retrieval_score")) for row in filtered_rows
    }
    block_key = str(metadata_row.get("block_key") or "")
    summary_by_component = {
        component_key: _component_summary(
            resources,
            component_key=component_key,
            block_key=block_key,
            query_signature_id=query_signature_id,
            feature_cache=feature_cache,
            full_summary_cache=full_summary_cache,
        )
        for component_key in shortlist_component_keys
    }
    candidate_signature_ids_by_component = {
        component_key: _component_signature_ids(
            resources,
            component_key=component_key,
            query_signature_id=query_signature_id,
        )
        for component_key in shortlist_component_keys
    }
    raw_similarity_features_by_component = _raw_similarity_features_by_component(
        resources,
        query_signature_id=str(query_signature_id),
        candidate_signature_ids_by_component=candidate_signature_ids_by_component,
    )
    seed_bypass_component_keys = seed_constraint_bypass_component_keys(
        dataset=resources.dataset,
        query_case=query_case,
        candidate_signature_ids_by_component=candidate_signature_ids_by_component,
    )
    retrieval_window_state = {
        "candidate_components": int(len(shortlist_component_keys)),
        "candidate_signatures": int(sum(len(value) for value in candidate_signature_ids_by_component.values())),
        "scored_candidate_components": int(len(shortlist_component_keys)),
        "scored_candidate_signatures": int(sum(len(value) for value in candidate_signature_ids_by_component.values())),
        "orcid_filter_applied": _to_int(metadata_row.get("orcid_filter_applied")),
        "middle_initial_filter_applied": _to_int(metadata_row.get("middle_initial_filter_applied")),
        "year_range_filter_applied": _to_int(metadata_row.get("year_range_filter_applied")),
    }
    stats_request = QueryClusterStatsRequest(
        query_signature_id=str(query_signature_id),
        shortlist_component_keys=shortlist_component_keys,
        candidate_signature_ids_by_component={
            str(component_key): [
                str(signature_id) for signature_id in candidate_signature_ids_by_component[component_key]
            ]
            for component_key in shortlist_component_keys
        },
        retrieval_ranks={
            str(component_key): int(retrieval_ranks[component_key]) for component_key in shortlist_component_keys
        },
        retrieval_scores={
            str(component_key): float(retrieval_scores[component_key]) for component_key in shortlist_component_keys
        },
        summary_by_component={
            str(component_key): summary_by_component[component_key] for component_key in shortlist_component_keys
        },
        incremental_dont_use_cluster_seeds_component_keys=seed_bypass_component_keys,
        ignore_disallow_constraints_component_keys=seed_bypass_component_keys,
    )
    estimated_pair_count = int(sum(len(value) for value in candidate_signature_ids_by_component.values()))
    return PreparedExistingGroup(
        source_path=str(source_path),
        group_index=int(group_index),
        dataset_name=str(resources.dataset_name),
        query_group_id=str(metadata_row["query_group_id"]),
        rows_before_total=int(rows_before_total),
        positive_rows_before_total=int(positive_rows_before_total),
        rows_after_window_cap=int(rows_after_window_cap),
        positive_rows_after_window_cap=int(positive_rows_after_window_cap),
        rows_after_self_filter=int(len(filtered_rows)),
        positive_rows_after_self_filter=int(sum(1 for row in filtered_rows if _to_int(row.get("label")) == 1)),
        self_containing_candidate_component_keys=self_containing_candidate_component_keys,
        self_containing_positive_component_keys=self_containing_positive_component_keys,
        residual_loo_positive_component_keys=residual_loo_positive_component_keys,
        original_by_component=original_by_component,
        query_case=query_case,
        query_view=str(query_view),
        query_features=query_features,
        shortlist_component_keys=shortlist_component_keys,
        retrieval_scores={str(key): float(value) for key, value in retrieval_scores.items()},
        retrieval_ranks={str(key): int(value) for key, value in retrieval_ranks.items()},
        retrieval_window_state=retrieval_window_state,
        summary_by_component=summary_by_component,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
        stats_request=stats_request,
        estimated_pair_count=estimated_pair_count,
    )


def _materialize_existing_group(
    prepared_group: PreparedExistingGroup,
    *,
    stats_by_component: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rebuilt_rows = generate_candidate_rows(
        query_case=prepared_group.query_case,
        query_view=prepared_group.query_view,
        query_features=prepared_group.query_features,
        shortlist_component_keys=list(prepared_group.shortlist_component_keys),
        retrieval_scores=prepared_group.retrieval_scores,
        retrieval_ranks=prepared_group.retrieval_ranks,
        retrieval_window_state=prepared_group.retrieval_window_state,
        summary_by_component=prepared_group.summary_by_component,
        stats_by_component=stats_by_component,
        rust_hybrid_centroid_retriever=None,
        raw_similarity_features_by_component=prepared_group.raw_similarity_features_by_component,
    )
    rebuilt_rows = materialize_derived_rows(rebuilt_rows)
    merged_rows: list[dict[str, Any]] = []
    for rebuilt_row in rebuilt_rows:
        component_key = str(rebuilt_row["candidate_component_key"])
        merged = dict(prepared_group.original_by_component[component_key])
        merged.update(rebuilt_row)
        merged_rows.append(merged)
    dropped_components = [
        component_key
        for component_key in prepared_group.original_by_component
        if component_key not in {str(row["candidate_component_key"]) for row in merged_rows}
    ]
    _restore_staged_labels_and_metadata(
        merged_rows,
        original_by_component=prepared_group.original_by_component,
    )
    return merged_rows, {
        "dataset": str(prepared_group.dataset_name),
        "query_group_id": str(prepared_group.query_group_id),
        "rows_before_total": int(prepared_group.rows_before_total),
        "rows_after_window_cap": int(prepared_group.rows_after_window_cap),
        "rows_after_self_filter": int(prepared_group.rows_after_self_filter),
        "rows_after": int(len(merged_rows)),
        "positive_rows_before_total": int(prepared_group.positive_rows_before_total),
        "positive_rows_after_window_cap": int(prepared_group.positive_rows_after_window_cap),
        "positive_rows_after_self_filter": int(prepared_group.positive_rows_after_self_filter),
        "positive_rows_after": int(sum(1 for row in merged_rows if _to_int(row.get("label")) == 1)),
        "window_cap_rows_dropped": int(prepared_group.rows_before_total - prepared_group.rows_after_window_cap),
        "self_filter_rows_dropped": int(prepared_group.rows_after_window_cap - prepared_group.rows_after_self_filter),
        "self_filter_positive_rows_dropped": int(
            prepared_group.positive_rows_after_window_cap - prepared_group.positive_rows_after_self_filter
        ),
        "self_containing_candidate_count": int(len(prepared_group.self_containing_candidate_component_keys)),
        "self_containing_positive_candidate_count": int(len(prepared_group.self_containing_positive_component_keys)),
        "residual_loo_positive_candidate_count": int(len(prepared_group.residual_loo_positive_component_keys)),
        "dropped_self_containing_candidate_component_keys": list(
            prepared_group.self_containing_candidate_component_keys
        ),
        "dropped_self_containing_positive_component_keys": list(prepared_group.self_containing_positive_component_keys),
        "residual_loo_positive_component_keys": list(prepared_group.residual_loo_positive_component_keys),
        "dropped_candidate_count": int(len(dropped_components)),
        "dropped_candidate_component_keys": dropped_components,
    }


def _restore_staged_labels_and_metadata(
    rows: list[dict[str, Any]],
    *,
    original_by_component: dict[str, dict[str, Any]],
) -> None:
    """Preserve staged/manual labels after feature rematerialization."""

    positive_component_keys: list[str] = []
    positive_retrieval_ranks: list[int] = []
    labels_by_component: dict[str, int] = {}
    for row in rows:
        component_key = str(row["candidate_component_key"])
        original = original_by_component[component_key]
        label = int(_to_int(original.get("label")))
        labels_by_component[component_key] = label
        if label == 1:
            positive_component_keys.append(component_key)
            positive_retrieval_ranks.append(_to_int(row.get("retrieval_rank")))

    positive_key_text = "|".join(sorted(positive_component_keys))
    best_positive_retrieval_rank = min(positive_retrieval_ranks) if positive_retrieval_ranks else None
    for row in rows:
        component_key = str(row["candidate_component_key"])
        label = labels_by_component[component_key]
        row["label"] = str(label)
        if "binary_safe_link_target" in row or "binary_safe_link_target" in original_by_component[component_key]:
            row["binary_safe_link_target"] = str(label)
        row["positive_candidate_count"] = int(len(positive_component_keys))
        row["positive_candidate_keys"] = positive_key_text
        row["group_has_positive"] = int(bool(positive_component_keys))
        row["best_positive_retrieval_rank"] = (
            int(best_positive_retrieval_rank) if best_positive_retrieval_rank is not None else None
        )


def _flush_prepared_groups(
    *,
    connection: sqlite3.Connection,
    file_summaries: dict[str, FileRepairSummaryState],
    clusterer: Any,
    resources: DatasetResources,
    prepared_groups: list[PreparedExistingGroup],
    pair_batch_size: int,
    max_top_k: int,
) -> None:
    if not prepared_groups:
        return
    batch_results = compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=resources.dataset,
        runtime_context=resources.runtime_context,
        constraint_backend=resources.constraint_backend,
        requests=[prepared_group.stats_request for prepared_group in prepared_groups],
        pair_batch_size=int(pair_batch_size),
        max_top_k=int(max_top_k),
    )
    if len(batch_results) != len(prepared_groups):
        raise RuntimeError(
            "Prepared group batch size did not match scored results: "
            f"prepared={len(prepared_groups)} results={len(batch_results)}"
        )
    insert_rows: list[tuple[str, int, bytes]] = []
    for prepared_group, (stats_by_component, pairwise_diagnostics) in zip(prepared_groups, batch_results, strict=True):
        merged_rows, group_summary = _materialize_existing_group(prepared_group, stats_by_component=stats_by_component)
        group_summary.update(
            {
                "pair_count": int(pairwise_diagnostics["pair_count"]),
                "featurize_seconds": float(pairwise_diagnostics["featurize_seconds"]),
                "model_predict_seconds": float(pairwise_diagnostics["model_predict_seconds"]),
            }
        )
        insert_rows.append(
            (
                str(prepared_group.source_path),
                int(prepared_group.group_index),
                _compress_rows(merged_rows),
            )
        )
        file_summaries[str(prepared_group.source_path)].record_result(
            rows_before=int(prepared_group.rows_before_total),
            positive_rows_before=int(prepared_group.positive_rows_before_total),
            rebuilt_rows=merged_rows,
            group_summary=group_summary,
        )
    connection.executemany(
        "INSERT OR REPLACE INTO rebuilt_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        insert_rows,
    )
    connection.commit()


def _record_empty_group_result(
    *,
    connection: sqlite3.Connection,
    file_summaries: dict[str, FileRepairSummaryState],
    source_path: str,
    group_index: int,
    dataset_name: str,
    query_group_id: str,
    rows_before_total: int,
    positive_rows_before_total: int,
    rows_after_window_cap: int,
    positive_rows_after_window_cap: int,
    self_containing_candidate_component_keys: tuple[str, ...] = (),
    self_containing_positive_component_keys: tuple[str, ...] = (),
    residual_loo_positive_component_keys: tuple[str, ...] = (),
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO rebuilt_groups (source_path, group_index, rows_blob) VALUES (?, ?, ?)",
        (str(source_path), int(group_index), _compress_rows([])),
    )
    connection.commit()
    group_summary = {
        "dataset": str(dataset_name),
        "query_group_id": str(query_group_id),
        "rows_before_total": int(rows_before_total),
        "rows_after_window_cap": int(rows_after_window_cap),
        "rows_after_self_filter": 0,
        "rows_after": 0,
        "positive_rows_before_total": int(positive_rows_before_total),
        "positive_rows_after_window_cap": int(positive_rows_after_window_cap),
        "positive_rows_after_self_filter": 0,
        "positive_rows_after": 0,
        "window_cap_rows_dropped": int(rows_before_total - rows_after_window_cap),
        "self_filter_rows_dropped": int(len(self_containing_candidate_component_keys)),
        "self_filter_positive_rows_dropped": int(len(self_containing_positive_component_keys)),
        "self_containing_candidate_count": int(len(self_containing_candidate_component_keys)),
        "self_containing_positive_candidate_count": int(len(self_containing_positive_component_keys)),
        "residual_loo_positive_candidate_count": int(len(residual_loo_positive_component_keys)),
        "dropped_self_containing_candidate_component_keys": list(self_containing_candidate_component_keys),
        "dropped_self_containing_positive_component_keys": list(self_containing_positive_component_keys),
        "residual_loo_positive_component_keys": list(residual_loo_positive_component_keys),
        "dropped_candidate_count": int(rows_after_window_cap),
        "dropped_candidate_component_keys": list(self_containing_candidate_component_keys),
        "pair_count": 0,
        "featurize_seconds": 0.0,
        "model_predict_seconds": 0.0,
    }
    file_summaries[str(source_path)].record_result(
        rows_before=int(rows_before_total),
        positive_rows_before=int(positive_rows_before_total),
        rebuilt_rows=[],
        group_summary=group_summary,
    )


def _process_dataset_groups(
    *,
    connection: sqlite3.Connection,
    file_summaries: dict[str, FileRepairSummaryState],
    dataset_name: str,
    clusterer: Any,
    resource_manager: DatasetResourceManager,
    pair_batch_size: int,
    query_batch_pair_limit: int,
    max_top_k: int,
    release_resources: bool = True,
) -> None:
    total_groups = int(
        connection.execute(
            "SELECT COUNT(*) FROM staged_groups WHERE dataset_name = ?",
            (str(dataset_name),),
        ).fetchone()[0]
    )
    resources = resource_manager.get(dataset_name)
    print(
        json.dumps(
            {
                "event": "dataset_activation",
                "dataset": str(dataset_name),
                "groups": int(total_groups),
                "pair_batch_size": int(pair_batch_size),
                "query_batch_pair_limit": int(query_batch_pair_limit),
            }
        ),
        flush=True,
    )
    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    full_summary_cache: dict[str, retrieval.ClusterSummary] = {}
    prepared_groups: list[PreparedExistingGroup] = []
    prepared_pair_count = 0
    processed_groups = 0
    cursor = connection.execute(
        """
        SELECT
            source_path,
            group_index,
            query_group_id,
            rows_before_total,
            positive_rows_before_total,
            rows_after_window_cap,
            positive_rows_after_window_cap,
            rows_blob
        FROM staged_groups
        WHERE dataset_name = ?
        ORDER BY source_path, group_index
        """,
        (str(dataset_name),),
    )
    for (
        source_path,
        group_index,
        query_group_id,
        rows_before_total,
        positive_rows_before_total,
        rows_after_window_cap,
        positive_rows_after_window_cap,
        rows_blob,
    ) in cursor:
        processed_groups += 1
        if int(rows_after_window_cap) <= 0:
            _record_empty_group_result(
                connection=connection,
                file_summaries=file_summaries,
                source_path=str(source_path),
                group_index=int(group_index),
                dataset_name=str(dataset_name),
                query_group_id=str(query_group_id),
                rows_before_total=int(rows_before_total),
                positive_rows_before_total=int(positive_rows_before_total),
                rows_after_window_cap=int(rows_after_window_cap),
                positive_rows_after_window_cap=int(positive_rows_after_window_cap),
            )
        else:
            rows = _decompress_rows(bytes(rows_blob))
            prepared_group = _prepare_existing_group(
                source_path=str(source_path),
                group_index=int(group_index),
                resources=resources,
                rows=rows,
                rows_before_total=int(rows_before_total),
                positive_rows_before_total=int(positive_rows_before_total),
                rows_after_window_cap=int(rows_after_window_cap),
                positive_rows_after_window_cap=int(positive_rows_after_window_cap),
                feature_cache=feature_cache,
                full_summary_cache=full_summary_cache,
            )
            if isinstance(prepared_group, EmptyExistingGroup):
                _record_empty_group_result(
                    connection=connection,
                    file_summaries=file_summaries,
                    source_path=str(prepared_group.source_path),
                    group_index=int(prepared_group.group_index),
                    dataset_name=str(prepared_group.dataset_name),
                    query_group_id=str(prepared_group.query_group_id),
                    rows_before_total=int(prepared_group.rows_before_total),
                    positive_rows_before_total=int(prepared_group.positive_rows_before_total),
                    rows_after_window_cap=int(prepared_group.rows_after_window_cap),
                    positive_rows_after_window_cap=int(prepared_group.positive_rows_after_window_cap),
                    self_containing_candidate_component_keys=prepared_group.self_containing_candidate_component_keys,
                    self_containing_positive_component_keys=prepared_group.self_containing_positive_component_keys,
                    residual_loo_positive_component_keys=prepared_group.residual_loo_positive_component_keys,
                )
            else:
                prepared_groups.append(prepared_group)
                prepared_pair_count += int(prepared_group.estimated_pair_count)
            if prepared_pair_count >= int(query_batch_pair_limit):
                _flush_prepared_groups(
                    connection=connection,
                    file_summaries=file_summaries,
                    clusterer=clusterer,
                    resources=resources,
                    prepared_groups=prepared_groups,
                    pair_batch_size=pair_batch_size,
                    max_top_k=max_top_k,
                )
                prepared_groups = []
                prepared_pair_count = 0
        if processed_groups % 1000 == 0 or processed_groups == total_groups:
            print(
                json.dumps(
                    {
                        "event": "dataset_progress",
                        "dataset": str(dataset_name),
                        "processed_groups": int(processed_groups),
                        "total_groups": int(total_groups),
                        "pending_groups": int(len(prepared_groups)),
                        "pending_estimated_pairs": int(prepared_pair_count),
                    }
                ),
                flush=True,
            )
    if prepared_groups:
        _flush_prepared_groups(
            connection=connection,
            file_summaries=file_summaries,
            clusterer=clusterer,
            resources=resources,
            prepared_groups=prepared_groups,
            pair_batch_size=pair_batch_size,
            max_top_k=max_top_k,
        )
    released = resource_manager.release_all() if release_resources else []
    print(
        json.dumps(
            {
                "event": "dataset_release",
                "dataset": str(dataset_name),
                "released_datasets": released,
            }
        ),
        flush=True,
    )


def _write_rebuilt_row_files(
    *,
    connection: sqlite3.Connection,
    fieldnames_by_path: dict[str, list[str]],
    ordered_source_paths: list[str],
) -> None:
    initial_only_rereview_decisions = read_initial_only_rereview_decisions()
    for source_path in ordered_source_paths:
        relative_path = Path(source_path)
        output_path = DEST_BUNDLE_ROOT / relative_path
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        fieldnames = fieldnames_by_path[source_path]
        with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as dst_handle:
            writer = csv.DictWriter(dst_handle, fieldnames=fieldnames)
            writer.writeheader()
            cursor = connection.execute(
                """
                SELECT rebuilt_groups.rows_blob, staged_groups.rows_blob
                FROM rebuilt_groups
                JOIN staged_groups
                  ON staged_groups.source_path = rebuilt_groups.source_path
                 AND staged_groups.group_index = rebuilt_groups.group_index
                WHERE rebuilt_groups.source_path = ?
                ORDER BY rebuilt_groups.group_index
                """,
                (str(source_path),),
            )
            for rows_blob, staged_rows_blob in cursor:
                rows = _decompress_rows(bytes(rows_blob))
                if rows:
                    staged_rows = _decompress_rows(bytes(staged_rows_blob))
                    _restore_staged_labels_and_metadata(
                        rows,
                        original_by_component={str(row["candidate_component_key"]): row for row in staged_rows},
                    )
                    initial_only_decision = initial_only_rereview_decisions.get(str(rows[0]["query_group_id"]))
                    if initial_only_decision is not None:
                        rows = _apply_initial_only_rereview_to_group(rows, decision=initial_only_decision)
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})
        _replace_with_retry(temp_path, output_path)
        print(json.dumps({"event": "wrote_row_file", "path": source_path}), flush=True)


def _refresh_file_summary_counts_from_outputs(
    *,
    file_summary_states: dict[str, FileRepairSummaryState],
    ordered_source_paths: list[str],
) -> None:
    """Refresh post-write counts after deterministic label/drop overrides."""

    for source_path in ordered_source_paths:
        state = file_summary_states[source_path]
        row_count = 0
        positive_row_count = 0
        positive_query_ids: set[str] = set()
        query_ids: set[str] = set()
        output_path = DEST_BUNDLE_ROOT / Path(source_path)
        with gzip.open(output_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                query_group_id = str(row["query_group_id"])
                query_ids.add(query_group_id)
                row_count += 1
                if _to_int(row.get("label")) == 1:
                    positive_row_count += 1
                    positive_query_ids.add(query_group_id)
        state.groups_after = int(len(query_ids))
        state.rows_after = int(row_count)
        state.positive_groups_after = int(len(positive_query_ids))
        state.positive_rows_after = int(positive_row_count)
        state.groups_fully_dropped = max(int(state.groups_fully_dropped), int(state.groups_before - state.groups_after))


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read one CSV or gzipped CSV file as dictionaries."""

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader], list(reader.fieldnames or [])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write one CSV or gzipped CSV file atomically."""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    else:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)
    _replace_with_retry(temp_path, path)


def _summarize_initial_only_rereview_application(
    *,
    selected_row_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Summarize how collapsed initial-only re-review decisions landed in active rows."""

    decisions = read_initial_only_rereview_decisions()
    action_counts = Counter(decision.action for decision in decisions.values())
    reason_counts = Counter(decision.reason_bucket for decision in decisions.values())
    per_file: list[dict[str, Any]] = []
    overall = Counter()
    present_query_ids: set[str] = set()
    for relative_path in selected_row_paths:
        row_path = DEST_BUNDLE_ROOT / relative_path
        if not row_path.exists():
            continue
        active_rows_by_query: dict[str, list[dict[str, Any]]] = {}
        with gzip.open(row_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                query_group_id = str(row.get("query_group_id", ""))
                if query_group_id in decisions:
                    active_rows_by_query.setdefault(query_group_id, []).append(dict(row))
        file_counts = Counter()
        for query_group_id, rows in active_rows_by_query.items():
            decision = decisions[query_group_id]
            labels_by_candidate = {str(row["candidate_component_key"]): _to_int(row.get("label")) for row in rows}
            positive_target = max(labels_by_candidate.values(), default=0)
            if decision.action == "drop_query":
                file_counts["drop_query_still_active"] += 1
            elif decision.action == "force_no_positive":
                if positive_target == 0:
                    file_counts["force_no_positive_applied"] += 1
                else:
                    file_counts["force_no_positive_mismatch"] += 1
            elif decision.action == "candidate_positive":
                resolved_safe_keys = resolve_reviewed_safe_component_keys(
                    decision.safe_component_key_texts,
                    candidate_component_keys=set(labels_by_candidate),
                )
                if not resolved_safe_keys:
                    file_counts["candidate_positive_safe_key_missing_from_active_rows"] += 1
                elif any(labels_by_candidate.get(key, 0) == 1 for key in resolved_safe_keys):
                    file_counts["candidate_positive_applied"] += 1
                else:
                    file_counts["candidate_positive_mismatch"] += 1
        active_query_ids = set(active_rows_by_query)
        present_query_ids.update(active_query_ids)
        overall.update(file_counts)
        per_file.append(
            {
                "path": _normalize_bundle_relpath(relative_path),
                "matched_active_queries": int(len(active_query_ids)),
                "counts": {str(key): int(value) for key, value in sorted(file_counts.items())},
            }
        )
    return {
        "decisions_loaded": int(len(decisions)),
        "decision_action_counts": {str(key): int(value) for key, value in sorted(action_counts.items())},
        "decision_reason_counts": {str(key): int(value) for key, value in sorted(reason_counts.items())},
        "matched_selected_query_ids": int(len(present_query_ids)),
        "absent_from_selected_files_by_action": {
            str(key): int(value)
            for key, value in sorted(
                Counter(
                    decision.action
                    for query_group_id, decision in decisions.items()
                    if query_group_id not in present_query_ids
                ).items()
            )
        },
        "overall_counts": {str(key): int(value) for key, value in sorted(overall.items())},
        "files": per_file,
    }


def _write_s2and_split_report(
    *,
    assignments: list[dict[str, Any]],
    stratum_balance: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    """Write a compact report for the promoted split after S2AND refresh."""

    worst = sorted(
        stratum_balance,
        key=lambda row: (
            max(
                abs(float(row["calibration_fit_delta"])),
                abs(float(row["calibration_check_delta"])),
                abs(float(row["test_delta"])),
            ),
            int(float(row["total"])),
        ),
        reverse=True,
    )
    lines = [
        "# Eval/Test Stratified Split",
        "",
        f"Total unique query groups: `{summary['unique_query_groups']}`",
        "",
        "Split sizes:",
    ]
    for split, count in summary["split_counts"].items():
        lines.append(f"- `{split}`: `{count}` ({summary['split_ratios'][split]:.3%})")
    lines.extend(
        [
            "",
            "Requested strata:",
            "- source: `s2and_block` vs `other_blocks`",
            "- candidate-positive presence",
            "- positive first vs positive not first vs no positive",
            "- single-letter vs multi-letter first name",
            "- one vs multiple candidates",
            "",
            f"Observed full-stratum cells: `{summary['observed_strata']}`",
            (
                "Cells too small to put at least one query in all three splits (`n < 3`): "
                f"`{summary['strata_too_small_for_all_splits']}`"
            ),
            (
                "Cells with at least one missing split after best-effort allocation: "
                f"`{summary['strata_with_missing_split']}`"
            ),
            "",
            "Maximum absolute marginal proportion delta from the full combined pool:",
        ]
    )
    for factor, value in sorted(summary["max_abs_marginal_delta"].items()):
        lines.append(f"- `{factor}`: `{float(value):.4f}`")
    lines.extend(
        [
            "",
            "Worst stratum count deltas:",
            "",
            "| stratum | total | fit | check | test | max abs count delta | missing split count |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst[:20]:
        max_abs_delta = max(
            abs(float(row["calibration_fit_delta"])),
            abs(float(row["calibration_check_delta"])),
            abs(float(row["test_delta"])),
        )
        lines.append(
            f"| `{row['stratum_key']}` | {int(float(row['total']))} | "
            f"{int(float(row['calibration_fit_count']))} | {int(float(row['calibration_check_count']))} | "
            f"{int(float(row['test_count']))} | {max_abs_delta:.2f} | "
            f"{int(float(row['missing_split_count']))} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `s2and_block` includes the public `s2and_eval` source.",
            "- Public S2AND split rows were refreshed from the full manual relabel ledger after residual LOO "
            "candidate materialization.",
        ]
    )
    report_path = DEST_BUNDLE_ROOT / "calibration" / "stratified_eval_test_split" / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _active_query_ids_by_stratified_source_key(
    *,
    selected_row_paths: tuple[Path, ...],
) -> dict[str, set[str]]:
    """Return active query ids for row files represented in the promoted split."""

    active_query_ids: dict[str, set[str]] = {}
    for relative_path in selected_row_paths:
        source_key = STRATIFIED_SPLIT_SOURCE_KEY_BY_ROW_RELATIVE_PATH.get(_normalize_bundle_relpath(relative_path))
        if source_key is None or source_key == "s2and_rescue_reviewed_train":
            continue
        row_path = DEST_BUNDLE_ROOT / relative_path
        if not row_path.exists():
            continue
        query_ids: set[str] = set()
        with gzip.open(row_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                query_ids.add(str(row["query_group_id"]))
        active_query_ids[source_key] = query_ids
    return active_query_ids


def _refresh_s2and_stratified_split_from_reviews(
    *,
    selected_row_paths: tuple[Path, ...],
) -> dict[str, Any] | None:
    """Refresh promoted split assignments for S2AND eval after row rematerialization."""

    if not any(_is_s2and_eval_row_path(path) for path in selected_row_paths):
        return None
    split_root = DEST_BUNDLE_ROOT / "calibration" / "stratified_eval_test_split"
    assignments_path = split_root / "combined_query_split_assignments.csv"
    stratum_balance_path = split_root / "stratum_balance.csv"
    summary_path = split_root / "summary.json"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing promoted split assignments: {assignments_path}")

    decisions = _merge_initial_only_rereview_into_s2and_decisions(
        _read_s2and_full_relabel_decisions(),
        read_initial_only_rereview_decisions(),
    )
    s2and_rows, _s2and_fieldnames = _read_csv_rows(DEST_BUNDLE_ROOT / S2AND_ROW_RELATIVE_PATH)
    assignments, assignment_fieldnames = _read_csv_rows(assignments_path)
    s2and_assignments = _s2and_assignment_rows_from_active_rows(s2and_rows, decisions=decisions)
    if not s2and_assignments:
        raise ValueError("S2AND eval row file has no active query groups after residual LOO materialization.")
    active_query_ids_by_source_key = _active_query_ids_by_stratified_source_key(selected_row_paths=selected_row_paths)
    pruned_assignment_counts: Counter[str] = Counter()
    non_s2and_assignments: list[dict[str, Any]] = []
    for row in assignments:
        source_key = str(row.get("source_key"))
        if source_key == "s2and_eval":
            continue
        active_query_ids = active_query_ids_by_source_key.get(source_key)
        if active_query_ids is not None and str(row.get("query_group_id")) not in active_query_ids:
            pruned_assignment_counts[source_key] += 1
            continue
        non_s2and_assignments.append(row)
    updated_assignments = [*non_s2and_assignments, *s2and_assignments]
    stratum_balance = _refresh_stratum_balance_rows(updated_assignments)

    assignment_fieldnames = [
        *assignment_fieldnames,
        *(field for field in s2and_assignments[0] if field not in set(assignment_fieldnames)),
    ]
    _write_csv_rows(assignments_path, updated_assignments, assignment_fieldnames)
    _write_csv_rows(stratum_balance_path, stratum_balance, list(stratum_balance[0]))

    old_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    split_counts = Counter(str(row["split"]) for row in updated_assignments)
    total = len(updated_assignments)
    summary = dict(old_summary)
    summary.update(
        {
            "unique_query_groups": int(total),
            "split_counts": {
                "calibration_fit": int(split_counts.get("calibration_fit", 0)),
                "calibration_check": int(split_counts.get("calibration_check", 0)),
                "test": int(split_counts.get("test", 0)),
            },
            "observed_strata": int(len(stratum_balance)),
            "strata_too_small_for_all_splits": int(sum(1 for row in stratum_balance if int(float(row["total"])) < 3)),
            "strata_with_missing_split": int(
                sum(1 for row in stratum_balance if int(float(row["missing_split_count"])) > 0)
            ),
            "max_abs_marginal_delta": {
                factor: _max_abs_assignment_marginal_delta(updated_assignments, factor)
                for factor in [
                    "source_stratum",
                    "has_positive_candidate",
                    "positive_rank_bucket",
                    "first_name_bucket",
                    "multiple_candidates",
                ]
            },
            "s2and_full_relabel_review_root": str(S2AND_FULL_RELABEL_REVIEW_ROOT.relative_to(REPO_ROOT)).replace(
                "\\",
                "/",
            ),
            "s2and_residual_loo_materialization": True,
            "pruned_assignment_counts": {
                str(key): int(value) for key, value in sorted(pruned_assignment_counts.items())
            },
        }
    )
    summary["split_ratios"] = {split: float(count / max(1, total)) for split, count in summary["split_counts"].items()}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_s2and_split_report(assignments=updated_assignments, stratum_balance=stratum_balance, summary=summary)

    return {
        "assignments_path": _display_path(assignments_path),
        "s2and_queries": int(len(s2and_assignments)),
        "s2and_positive_queries": int(sum(_to_boolish(row["has_positive_candidate"]) for row in s2and_assignments)),
        "queries": int(total),
        "split_counts": dict(summary["split_counts"]),
        "observed_strata": int(summary["observed_strata"]),
        "pruned_assignment_counts": dict(summary["pruned_assignment_counts"]),
    }


def _hwang_label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return query and positive-row counts for H-Wang row labels."""

    positive_rows_by_query: Counter[str] = Counter()
    query_ids: set[str] = set()
    positive_rows = 0
    for row in rows:
        query_group_id = str(row["query_group_id"])
        query_ids.add(query_group_id)
        label = _to_int(row.get("label"))
        if label > 0:
            positive_rows += label
            positive_rows_by_query[query_group_id] += label
    return {
        "queries": int(len(query_ids)),
        "positive_rows": int(positive_rows),
        "positive_queries": int(sum(1 for value in positive_rows_by_query.values() if int(value) > 0)),
    }


def _display_path(path: Path) -> str:
    """Return a compact path for JSON rebuild telemetry."""

    try:
        return str(path.relative_to(REPO_ROOT)).replace("/", "\\")
    except ValueError:
        return str(path)


def _apply_hwang_candidate_level_label_overrides(
    *,
    bundle_root: Path,
    selected_row_paths: tuple[Path, ...],
    per_file_summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Re-apply H-Wang reviewed labels after rebuilding no-self-filtered rows.

    H-Wang manual corrections are candidate-level facts. A query-level positive
    target is only valid when the reviewed candidate survived the current
    candidate filter.
    """

    selected_paths = {_normalize_bundle_relpath(path) for path in selected_row_paths}
    if _normalize_bundle_relpath(HWANG_ROW_RELATIVE_PATH) not in selected_paths:
        return None

    hwang_rows_path = bundle_root / HWANG_ROW_RELATIVE_PATH
    manifest_path = bundle_root / HWANG_CANDIDATE_LEVEL_MANIFEST_RELATIVE_PATH
    clean_overrides_path = bundle_root / HWANG_CLEAN_OVERRIDES_RELATIVE_PATH
    summary_path = bundle_root / HWANG_CANDIDATE_LEVEL_SUMMARY_RELATIVE_PATH
    for required_path in (hwang_rows_path, manifest_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing H-Wang candidate-level relabel input: {required_path}")

    hwang_rows, hwang_fieldnames = _read_csv_rows(hwang_rows_path)
    manifest_rows, manifest_fieldnames = _read_csv_rows(manifest_path)
    required_manifest_fields = {
        "query_group_id",
        "correction_type",
        "reviewed_candidate_component_key",
        "review_source_path",
    }
    missing_manifest_fields = sorted(required_manifest_fields - set(manifest_fieldnames))
    if missing_manifest_fields:
        raise ValueError(f"H-Wang candidate-level manifest missing columns: {missing_manifest_fields}")
    if not hwang_rows:
        raise ValueError("H-Wang row file is empty")
    if not manifest_rows:
        raise ValueError("H-Wang candidate-level manifest is empty")

    rows_by_query: dict[str, list[dict[str, Any]]] = {}
    candidates_by_query: dict[str, set[str]] = {}
    raw_positive_rows_by_query: Counter[str] = Counter()
    initial_only_rereview_decisions = read_initial_only_rereview_decisions()
    for row in hwang_rows:
        query_group_id = str(row["query_group_id"])
        candidate_key = str(row["candidate_component_key"])
        rows_by_query.setdefault(query_group_id, []).append(row)
        candidates_by_query.setdefault(query_group_id, set()).add(candidate_key)
        label = _to_int(row.get("label"))
        row["label"] = str(label)
        if label > 0:
            raw_positive_rows_by_query[query_group_id] += label

    dropped_by_initial_only_rereview_ids = {
        query_group_id
        for query_group_id in rows_by_query
        if initial_only_rereview_decisions.get(query_group_id) is not None
        and initial_only_rereview_decisions[query_group_id].action == "drop_query"
    }
    if dropped_by_initial_only_rereview_ids:
        hwang_rows = [
            row for row in hwang_rows if str(row["query_group_id"]) not in dropped_by_initial_only_rereview_ids
        ]
        for query_group_id in dropped_by_initial_only_rereview_ids:
            rows_by_query.pop(query_group_id, None)
            candidates_by_query.pop(query_group_id, None)
            raw_positive_rows_by_query.pop(query_group_id, None)

    hwang_query_ids = set(rows_by_query)
    manifest_query_ids = {str(row["query_group_id"]) for row in manifest_rows}
    missing_from_manifest_ids = hwang_query_ids - manifest_query_ids
    missing_from_rows_ids = manifest_query_ids - hwang_query_ids
    allowed_initial_only_dropped_manifest_ids = {
        query_group_id
        for query_group_id in missing_from_rows_ids
        if initial_only_rereview_decisions.get(query_group_id) is not None
        and initial_only_rereview_decisions[query_group_id].action == "drop_query"
    }
    unexpected_missing_from_rows = missing_from_rows_ids - allowed_initial_only_dropped_manifest_ids
    if missing_from_manifest_ids or unexpected_missing_from_rows:
        missing_from_manifest = sorted(missing_from_manifest_ids)[:10]
        missing_from_rows = sorted(unexpected_missing_from_rows)[:10]
        raise ValueError(
            "H-Wang relabel manifest query set mismatch: "
            f"missing_from_manifest={missing_from_manifest} missing_from_rows={missing_from_rows}"
        )
    manifest_rows = [
        row for row in manifest_rows if str(row["query_group_id"]) not in allowed_initial_only_dropped_manifest_ids
    ]

    updated_manifest_rows: list[dict[str, Any]] = []
    clean_override_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        query_group_id = str(manifest_row["query_group_id"])
        correction_type = str(manifest_row.get("correction_type") or "none")
        candidate_key = str(manifest_row.get("reviewed_candidate_component_key") or "")
        initial_only_decision = initial_only_rereview_decisions.get(query_group_id)
        label_action = "keep_surviving_raw_labels"
        reviewed_candidate_survived = 0
        if initial_only_decision is not None and initial_only_decision.action == "force_no_positive":
            label_action = "force_no_positive"
            for row in rows_by_query[query_group_id]:
                row["label"] = "0"
        elif initial_only_decision is not None and initial_only_decision.action == "candidate_positive":
            safe_keys = set(
                resolve_reviewed_safe_component_keys(
                    initial_only_decision.safe_component_key_texts,
                    candidate_component_keys=candidates_by_query[query_group_id],
                )
            )
            reviewed_candidate_survived = int(bool(safe_keys))
            label_action = "add_reviewed_positive" if safe_keys else "reviewed_positive_missing_after_filter"
            for row in rows_by_query[query_group_id]:
                row["label"] = str(int(str(row["candidate_component_key"]) in safe_keys))
        elif correction_type in {"top1_should_link", "non_top1_should_link"}:
            if candidate_key and candidate_key in candidates_by_query[query_group_id]:
                label_action = "add_reviewed_positive"
                reviewed_candidate_survived = 1
                for row in rows_by_query[query_group_id]:
                    if str(row["candidate_component_key"]) == candidate_key:
                        row["label"] = "1"
            else:
                label_action = "reviewed_positive_missing_after_filter"
        elif correction_type == "should_abstain":
            label_action = "force_no_positive"
            for row in rows_by_query[query_group_id]:
                row["label"] = "0"

        positive_rows_after = sum(_to_int(row.get("label")) for row in rows_by_query[query_group_id])
        manual_safe_target = 1 if positive_rows_after > 0 else 0
        updated_manifest_row = dict(manifest_row)
        updated_manifest_row.update(
            {
                "dataset": str(manifest_row.get("dataset") or "h_wang"),
                "reviewed_candidate_survived": str(reviewed_candidate_survived),
                "raw_positive_rows_before_candidate_relabel": str(
                    int(raw_positive_rows_by_query.get(query_group_id, 0))
                ),
                "label_action": label_action,
                "positive_rows_after_candidate_relabel": str(int(positive_rows_after)),
                "manual_safe_target": str(manual_safe_target),
            }
        )
        updated_manifest_rows.append(updated_manifest_row)
        clean_override_rows.append(
            {
                "query_group_id": query_group_id,
                "manual_safe_target": str(manual_safe_target),
                "manual_assessment": label_action,
                "correction_type": correction_type,
                "review_source_path": str(manifest_row.get("review_source_path") or ""),
            }
        )

    manifest_output_fieldnames = [
        "query_group_id",
        "dataset",
        "correction_type",
        "reviewed_candidate_component_key",
        "reviewed_candidate_survived",
        "raw_positive_rows_before_candidate_relabel",
        "label_action",
        "review_source_path",
        "positive_rows_after_candidate_relabel",
        "manual_safe_target",
    ]
    label_counts_after = _hwang_label_counts(hwang_rows)
    label_action_counts = Counter(str(row["label_action"]) for row in updated_manifest_rows)
    correction_type_counts = Counter(str(row["correction_type"]) for row in updated_manifest_rows)
    reviewed_positive_rows = [
        row
        for row in updated_manifest_rows
        if str(row["correction_type"]) in {"top1_should_link", "non_top1_should_link"}
    ]
    summary = {
        "hwang_rows_path": _display_path(bundle_root / HWANG_ROW_RELATIVE_PATH),
        "hwang_clean_overrides_path": _display_path(clean_overrides_path),
        "manifest_path": _display_path(manifest_path),
        "apply": True,
        "queries": int(label_counts_after["queries"]),
        "rows": int(len(hwang_rows)),
        "positive_rows_before_candidate_relabel": int(sum(raw_positive_rows_by_query.values())),
        "positive_rows_after_candidate_relabel": int(label_counts_after["positive_rows"]),
        "positive_queries_after_candidate_relabel": int(label_counts_after["positive_queries"]),
        "label_action_counts": {str(key): int(value) for key, value in sorted(label_action_counts.items())},
        "correction_type_counts": {str(key): int(value) for key, value in sorted(correction_type_counts.items())},
        "reviewed_positive_corrections": int(len(reviewed_positive_rows)),
        "reviewed_positive_corrections_survived": int(
            sum(_to_int(row["reviewed_candidate_survived"]) for row in reviewed_positive_rows)
        ),
        "reviewed_positive_corrections_missing_after_filter": int(
            sum(1 for row in reviewed_positive_rows if _to_int(row["reviewed_candidate_survived"]) == 0)
        ),
        "manifest_queries_dropped_by_initial_only_rereview": int(len(allowed_initial_only_dropped_manifest_ids)),
        "raw_positive_queries_before_candidate_relabel": int(
            sum(1 for value in raw_positive_rows_by_query.values() if int(value) > 0)
        ),
        "no_positive_queries_after_candidate_relabel": int(
            int(label_counts_after["queries"]) - int(label_counts_after["positive_queries"])
        ),
    }

    _write_csv_rows(hwang_rows_path, hwang_rows, hwang_fieldnames)
    _write_csv_rows(clean_overrides_path, clean_override_rows, list(clean_override_rows[0]))
    _write_csv_rows(manifest_path, updated_manifest_rows, manifest_output_fieldnames)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for file_summary in per_file_summaries:
        if _normalize_bundle_relpath(file_summary["path"]) != _normalize_bundle_relpath(HWANG_ROW_RELATIVE_PATH):
            continue
        file_summary["positive_rows_after"] = int(label_counts_after["positive_rows"])
        file_summary["positive_groups_after"] = int(label_counts_after["positive_queries"])
        file_summary["positive_rows_dropped"] = int(
            _to_int(file_summary.get("positive_rows_before")) - int(label_counts_after["positive_rows"])
        )
        break

    print(json.dumps({"event": "applied_hwang_candidate_level_relabel", **summary}), flush=True)
    return summary


def _update_hwang_clean_override_asset_counts(payload: dict[str, Any]) -> None:
    """Refresh H-Wang clean override counts in bundle metadata."""

    asset = payload.get("assets", {}).get("test", {}).get("hwang_clean_overrides")
    if not isinstance(asset, dict) or "path" not in asset:
        return
    path = DEST_BUNDLE_ROOT / Path(str(asset["path"]))
    if not path.exists():
        return
    rows, _fieldnames = _read_csv_rows(path)
    asset["queries"] = int(len({str(row["query_group_id"]) for row in rows}))
    asset["positive_overrides"] = int(sum(_to_int(row.get("manual_safe_target")) for row in rows))


def _update_hwang_candidate_level_asset_counts(payload: dict[str, Any]) -> None:
    """Refresh H-Wang candidate-level manifest counts in bundle metadata."""

    asset = payload.get("assets", {}).get("test", {}).get("hwang_candidate_level_label_overrides")
    if not isinstance(asset, dict):
        return
    summary_path_like = asset.get("summary_path")
    if not summary_path_like:
        return
    summary_path = DEST_BUNDLE_ROOT / Path(str(summary_path_like))
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    asset["queries"] = int(summary["queries"])
    asset["positive_queries_after_candidate_relabel"] = int(summary["positive_queries_after_candidate_relabel"])
    asset["reviewed_positive_corrections"] = int(summary["reviewed_positive_corrections"])
    asset["reviewed_positive_corrections_survived"] = int(summary["reviewed_positive_corrections_survived"])


def _update_dataset_contract_asset_counts(payload: dict[str, Any]) -> None:
    """Refresh dataset-contract counts in bundle metadata."""

    asset = payload.get("assets", {}).get("dataset_contract")
    if not isinstance(asset, dict):
        return
    summary_path_like = asset.get("custom_label_ledger_summary_path")
    if not summary_path_like:
        return
    summary_path = DEST_BUNDLE_ROOT / Path(str(summary_path_like))
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    asset["custom_label_ledger_rows"] = int(summary["ledger_rows"])
    asset["comparison_fatal_mismatch_count"] = int(summary["comparison_fatal_mismatch_count"])
    asset["label_slice_counts"] = dict(summary["slice_counts"])


def _update_stratified_split_asset_counts(payload: dict[str, Any]) -> None:
    """Refresh promoted stratified split counts in bundle metadata."""

    asset = payload.get("assets", {}).get("calibration", {}).get("stratified_eval_test_split")
    if not isinstance(asset, dict):
        return
    assignments_path_like = asset.get("assignments_path")
    summary_path_like = asset.get("summary_path")
    if not assignments_path_like:
        return
    assignments_path = DEST_BUNDLE_ROOT / Path(str(assignments_path_like))
    if not assignments_path.exists():
        return
    assignments, _fieldnames = _read_csv_rows(assignments_path)
    split_counts = Counter(str(row.get("split", "")) for row in assignments)
    asset["queries"] = int(len(assignments))
    asset["calibration_fit_queries"] = int(split_counts.get("calibration_fit", 0))
    asset["calibration_check_queries"] = int(split_counts.get("calibration_check", 0))
    asset["test_queries"] = int(split_counts.get("test", 0))
    if summary_path_like:
        summary_path = DEST_BUNDLE_ROOT / Path(str(summary_path_like))
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            asset["observed_strata"] = int(summary.get("observed_strata", asset.get("observed_strata", 0)))
            asset["strata_too_small_for_all_splits"] = int(
                summary.get(
                    "strata_too_small_for_all_splits",
                    asset.get("strata_too_small_for_all_splits", 0),
                )
            )
            asset["strata_with_missing_split"] = int(
                summary.get("strata_with_missing_split", asset.get("strata_with_missing_split", 0))
            )


def _write_bundle_metadata(
    *,
    per_file_summaries: list[dict[str, Any]],
    expected_metrics: dict[str, float] | None = None,
) -> None:
    payload = json.loads((DEST_BUNDLE_ROOT / "bundle.json").read_text(encoding="utf-8"))
    payload["bundle_name"] = DEST_BUNDLE_ROOT.name
    payload["created_on"] = CREATED_ON
    summary_by_relpath = {_normalize_bundle_relpath(item["path"]): item for item in per_file_summaries}

    def update_asset_counts(node: Any, *, relative_path: Path, file_summary: dict[str, Any]) -> bool:
        if isinstance(node, dict):
            path_value = node.get("path")
            if isinstance(path_value, str) and _normalize_bundle_relpath(path_value) == _normalize_bundle_relpath(
                relative_path
            ):
                node["rows"] = int(file_summary["rows_after"])
                node["queries"] = int(file_summary["groups_after"])
                node["positive_rows"] = int(file_summary["positive_rows_after"])
                return True
            for value in node.values():
                if update_asset_counts(value, relative_path=relative_path, file_summary=file_summary):
                    return True
        elif isinstance(node, list):
            for value in node:
                if update_asset_counts(value, relative_path=relative_path, file_summary=file_summary):
                    return True
        return False

    for relative_path in ROW_RELATIVE_PATHS:
        file_summary = summary_by_relpath.get(_normalize_bundle_relpath(relative_path))
        if file_summary is None:
            continue
        if not update_asset_counts(payload.get("assets", {}), relative_path=relative_path, file_summary=file_summary):
            raise KeyError(f"Failed to update bundle counts for {relative_path}")
    _update_hwang_clean_override_asset_counts(payload)
    _update_hwang_candidate_level_asset_counts(payload)
    _update_dataset_contract_asset_counts(payload)
    _update_stratified_split_asset_counts(payload)
    if expected_metrics is not None:
        payload["expected_metrics"]["classic"] = expected_metrics
    payload["notes"] = [
        "Rebuilds the active official bundle by recomputing every persisted query group under the current constraint logic.",
        "Processes row groups dataset-by-dataset across training, calibration, and eval so each ANDData reload happens once per dataset.",
        "Caps the persisted retrieval surface at 25 neighbors and drops any candidate row whose query-to-cluster pair set contains at least one hard-disallow pair or whose raw candidate component contains the query signature.",
        "For public S2AND eval, reviewed positive candidates whose raw component contains the query signature are materialized as residual leave-one-out positives with the query signature removed from feature computation.",
        "Applies the 2026-04-29 model-visible initial-only re-review as a dataset-level override: unanimous links stay positive, packet-too-weak abstains become no-positive, and feature-contract failures, impossible cases, or conflicting reviews are dropped.",
        "Applies H-Wang reviewed corrections at candidate level after self-candidate filtering and regenerates cleaned query targets from surviving candidate labels.",
        "Replays the official classic pipeline after the rebuild using only w5 and w25 evaluation windows.",
    ]
    (DEST_BUNDLE_ROOT / "bundle.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_documentation(*, classic_summary: dict[str, Any], repair_summary: dict[str, Any]) -> None:
    bundle_name = DEST_BUNDLE_ROOT.name
    stack_id = bundle_name.removeprefix("joint_safe_link_official_stack_")
    in_place_rebuild = SOURCE_BUNDLE_ROOT == DEST_BUNDLE_ROOT
    hwang_relabel = repair_summary.get("hwang_candidate_level_relabel")
    hwang_relabel_lines: list[str] = []
    hwang_provenance_step = ""
    if isinstance(hwang_relabel, dict):
        hwang_relabel_lines = [
            (
                "- H-Wang candidate-level relabel after self-candidate filtering: "
                f"{int(hwang_relabel['queries'])} queries, "
                f"{int(hwang_relabel['positive_queries_after_candidate_relabel'])} positive queries, "
                f"{int(hwang_relabel['positive_rows_after_candidate_relabel'])} positive rows"
            )
        ]
        hwang_provenance_step = (
            " Apply H-Wang reviewed corrections at candidate level after the self-candidate filter, so stale "
            "query-level positives are no longer treated as linkable when the reviewed candidate was removed."
        )
    if in_place_rebuild:
        intro_line = f"This bundle refreshes `{bundle_name}` in place."
        rebuild_origin = "It reuses the existing bundle root and rewrites the persisted row files in place:"
        source_line = f"Source bundle: `{SOURCE_BUNDLE_ROOT.name}` (rebuilt in place)."
        first_step = "1. Stage all persisted query groups from the active bundle once, trimming the stored row surface to the top 25 retrieval neighbors."
    else:
        intro_line = f"This bundle supersedes `{SOURCE_BUNDLE_ROOT.name}`."
        rebuild_origin = "It is a full standalone copy of the source bundle with one full-surface rebuild:"
        source_line = f"Source bundle: `{SOURCE_BUNDLE_ROOT.name}`."
        first_step = "1. Copy the full source bundle into the new stack so the output stays physically standalone."
    verification_lines = [
        "- `uv run python scripts\\rebuild_joint_safe_link_official_stack.py`",
        "- `uv run python scripts\\run_joint_safe_link_official_classic.py`",
        "- `uv run python scripts\\validate_joint_safe_link_official_stack.py`",
    ]
    readme_lines = [
        f"# Joint Safe-Link Official Stack ({stack_id})",
        "",
        intro_line,
        "",
        rebuild_origin,
        "every persisted training, calibration, and eval query group was regenerated from source data under the current",
        "constraint logic, candidate rows were capped to the top 25 retrieval neighbors, and any remaining candidate row",
        "with at least one hard-disallow query-to-signature pair or with the query signature inside the raw candidate",
        "component was dropped, except reviewed public-S2AND positives that are feature-materialized as residual",
        "leave-one-out candidates with the query signature removed.",
        "",
        "Key details:",
        "",
        f"- rebuilt row files: {len(ROW_RELATIVE_PATHS)}",
        f"- query groups before rebuild: {repair_summary['groups_before_total']}",
        f"- query groups after rebuild: {repair_summary['groups_after_total']}",
        f"- rows before rebuild: {repair_summary['rows_before_total']}",
        f"- rows after rebuild: {repair_summary['rows_after_total']}",
        f"- rows dropped: {repair_summary['rows_dropped_total']}",
        f"- positive rows dropped: {repair_summary['positive_rows_dropped_total']}",
        *hwang_relabel_lines,
        f"- classic score threshold: {classic_summary['abstain_rule']['score_threshold']:.12f}",
        f"- classic margin threshold: {classic_summary['abstain_rule']['margin_threshold']:.12f}",
        "",
        "Verification:",
        "",
        *verification_lines,
    ]
    (DEST_BUNDLE_ROOT / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    provenance_lines = [
        f"# Joint Safe-Link {stack_id} Provenance",
        "",
        source_line,
        "",
        "Build steps:",
        "",
        first_step,
        "2. Reload each originating dataset exactly once and score batched query-group requests across train, calibration, and eval.",
        "3. Drop any candidate row whose query-to-cluster pair set contains at least one hard-disallow pair or whose raw candidate component contains the query signature, aside from reviewed S2AND residual LOO positives.",
        "4. Materialize reviewed public-S2AND self-containing positives as residual leave-one-out candidates by "
        "removing the query signature from feature computation, then refresh S2AND promoted split assignments.",
        f"5. Rewrite the rebuilt row files in place and update `bundle.json` counts.{hwang_provenance_step}",
        "6. Replay the official classic pipeline and freeze the resulting expected metrics into `bundle.json`.",
        "",
        f"Rebuild telemetry directory: `{TELEMETRY_DIR.relative_to(REPO_ROOT)}`.",
        f"Classic replay directory: `{SCRATCH_OUT.relative_to(REPO_ROOT)}`.",
    ]
    (DEST_BUNDLE_ROOT / "PROVENANCE.md").write_text("\n".join(provenance_lines) + "\n", encoding="utf-8")


def _aggregate_repair_summary(
    per_file: list[dict[str, Any]], *, pair_batch_size: int, query_batch_pair_limit: int
) -> dict[str, Any]:
    return {
        "files": per_file,
        "groups_before_total": int(sum(_to_int(item.get("groups_before")) for item in per_file)),
        "groups_after_total": int(sum(_to_int(item.get("groups_after")) for item in per_file)),
        "rows_before_total": int(sum(_to_int(item.get("rows_before")) for item in per_file)),
        "rows_after_total": int(sum(_to_int(item.get("rows_after")) for item in per_file)),
        "rows_dropped_total": int(sum(_to_int(item.get("rows_dropped")) for item in per_file)),
        "positive_groups_before_total": int(sum(_to_int(item.get("positive_groups_before")) for item in per_file)),
        "positive_groups_after_total": int(sum(_to_int(item.get("positive_groups_after")) for item in per_file)),
        "positive_rows_before_total": int(sum(_to_int(item.get("positive_rows_before")) for item in per_file)),
        "positive_rows_after_total": int(sum(_to_int(item.get("positive_rows_after")) for item in per_file)),
        "positive_rows_dropped_total": int(sum(_to_int(item.get("positive_rows_dropped")) for item in per_file)),
        "groups_with_dropped_rows_total": int(sum(_to_int(item.get("groups_with_dropped_rows")) for item in per_file)),
        "groups_fully_dropped_total": int(sum(_to_int(item.get("groups_fully_dropped")) for item in per_file)),
        "groups_lost_all_positives_total": int(
            sum(_to_int(item.get("groups_lost_all_positives")) for item in per_file)
        ),
        "window_size": int(WINDOW_SIZE),
        "pair_batch_size": int(pair_batch_size),
        "query_batch_pair_limit": int(query_batch_pair_limit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild an official joint safe-link bundle in place or into a new root."
    )
    parser.add_argument("--source-bundle-root", type=Path, default=SOURCE_BUNDLE_ROOT)
    parser.add_argument("--dest-bundle-root", type=Path, default=DEST_BUNDLE_ROOT)
    parser.add_argument("--scratch-out", type=Path, default=SCRATCH_OUT)
    parser.add_argument("--telemetry-dir", type=Path, default=TELEMETRY_DIR)
    parser.add_argument("--spool-db-path", type=Path, default=None)
    parser.add_argument("--worker-dataset", type=str, default=None)
    parser.add_argument(
        "--row-file",
        dest="row_files",
        action="append",
        default=[],
        help="Bundle-relative row file to rebuild. Repeat to select multiple files. Default: rebuild all row files.",
    )
    parser.add_argument(
        "--limit-groups-per-file",
        type=int,
        default=None,
        help="Optional cap for sampled rebuilds. Intended for smoke tests only.",
    )
    parser.add_argument(
        "--skip-classic",
        action="store_true",
        help="Skip the classic replay and expected-metric rewrite. Intended for sampled rebuilds only.",
    )
    parser.add_argument("--repair-mode", choices=("full",), default="full")
    parser.add_argument("--pair-batch-size", type=int, default=PAIR_BATCH_SIZE)
    parser.add_argument("--query-batch-pair-limit", type=int, default=QUERY_BATCH_PAIR_LIMIT)
    return parser.parse_args()


def main() -> None:
    global SOURCE_BUNDLE_ROOT
    global DEST_BUNDLE_ROOT
    global SCRATCH_OUT
    global TELEMETRY_DIR

    args = parse_args()
    SOURCE_BUNDLE_ROOT = Path(args.source_bundle_root).resolve()
    DEST_BUNDLE_ROOT = Path(args.dest_bundle_root).resolve()
    SCRATCH_OUT = Path(args.scratch_out).resolve()
    TELEMETRY_DIR = Path(args.telemetry_dir).resolve()
    spool_db_path = (
        Path(args.spool_db_path).resolve() if args.spool_db_path is not None else TELEMETRY_DIR / SPOOL_DB_FILENAME
    )
    selected_row_paths = tuple(Path(value) for value in args.row_files) if args.row_files else ROW_RELATIVE_PATHS

    _configure_official_rust_backend(n_jobs=N_JOBS)
    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    pair_batch_size = int(args.pair_batch_size)
    query_batch_pair_limit = int(args.query_batch_pair_limit)
    if pair_batch_size <= 0:
        raise ValueError(f"pair_batch_size must be positive, got {pair_batch_size}")
    if query_batch_pair_limit <= 0:
        raise ValueError(f"query_batch_pair_limit must be positive, got {query_batch_pair_limit}")
    if args.worker_dataset:
        if not spool_db_path.exists():
            raise FileNotFoundError(f"Worker spool db does not exist: {spool_db_path}")
        _run_dataset_worker(
            spool_db_path=spool_db_path,
            dataset_name=str(args.worker_dataset),
            pair_batch_size=pair_batch_size,
            query_batch_pair_limit=query_batch_pair_limit,
            max_top_k=MAX_TOP_K,
        )
        return

    in_place_rebuild = SOURCE_BUNDLE_ROOT == DEST_BUNDLE_ROOT
    resuming_from_spool = DEST_BUNDLE_ROOT.exists() and spool_db_path.exists()
    if DEST_BUNDLE_ROOT.exists() and not spool_db_path.exists() and not in_place_rebuild:
        raise FileExistsError(f"Destination bundle already exists without a resumable spool db: {DEST_BUNDLE_ROOT}")

    if resuming_from_spool:
        print(
            json.dumps(
                {
                    "event": "resume_from_spool",
                    "dest_bundle_root": str(DEST_BUNDLE_ROOT.relative_to(REPO_ROOT)).replace("/", "\\"),
                    "spool_db": str(spool_db_path.relative_to(REPO_ROOT)).replace("/", "\\"),
                }
            ),
            flush=True,
        )
    elif in_place_rebuild:
        print(
            json.dumps(
                {
                    "event": "rebuild_in_place",
                    "bundle_root": str(DEST_BUNDLE_ROOT.relative_to(REPO_ROOT)).replace("/", "\\"),
                }
            ),
            flush=True,
        )
    else:
        print(f"Copying {SOURCE_BUNDLE_ROOT.name} -> {DEST_BUNDLE_ROOT.name}", flush=True)
        shutil.copytree(SOURCE_BUNDLE_ROOT, DEST_BUNDLE_ROOT, copy_function=shutil.copy2)
        if spool_db_path.exists():
            spool_db_path.unlink()

    connection = _connect_spool_db(spool_db_path)
    try:
        if resuming_from_spool:
            fieldnames_by_path, file_summary_states, ordered_source_paths = _load_staged_input_groups_from_spool(
                connection,
                selected_row_paths=selected_row_paths,
            )
        else:
            fieldnames_by_path, file_summary_states, ordered_source_paths = _stage_input_groups(
                connection=connection,
                selected_row_paths=selected_row_paths,
                limit_groups_per_file=args.limit_groups_per_file,
            )
        datasets = _selected_datasets(connection, ordered_source_paths=ordered_source_paths)
        print(json.dumps({"event": "staged_datasets", "datasets": datasets}), flush=True)

        for dataset_name in datasets:
            summary_path = _worker_summary_path(str(dataset_name))
            if (
                resuming_from_spool
                and summary_path.exists()
                and _dataset_rebuild_is_complete(
                    connection,
                    dataset_name=str(dataset_name),
                    ordered_source_paths=ordered_source_paths,
                )
            ):
                _merge_worker_summary(
                    dataset_name=str(dataset_name),
                    file_summary_states=file_summary_states,
                    summary_path=summary_path,
                )
                print(
                    json.dumps(
                        {
                            "event": "dataset_worker_reused",
                            "dataset": str(dataset_name),
                            "summary_path": _display_path(summary_path),
                        }
                    ),
                    flush=True,
                )
            else:
                _run_dataset_worker_subprocess(
                    dataset_name=str(dataset_name),
                    spool_db_path=spool_db_path,
                    pair_batch_size=pair_batch_size,
                    query_batch_pair_limit=query_batch_pair_limit,
                    file_summary_states=file_summary_states,
                )

        _write_rebuilt_row_files(
            connection=connection,
            fieldnames_by_path=fieldnames_by_path,
            ordered_source_paths=ordered_source_paths,
        )
        _refresh_file_summary_counts_from_outputs(
            file_summary_states=file_summary_states,
            ordered_source_paths=ordered_source_paths,
        )
    finally:
        connection.close()

    per_file_summaries = [file_summary_states[source_path].to_payload() for source_path in ordered_source_paths]
    hwang_candidate_level_relabel = _apply_hwang_candidate_level_label_overrides(
        bundle_root=DEST_BUNDLE_ROOT,
        selected_row_paths=tuple(Path(path) for path in ordered_source_paths),
        per_file_summaries=per_file_summaries,
    )
    s2and_stratified_split_refresh = _refresh_s2and_stratified_split_from_reviews(
        selected_row_paths=tuple(Path(path) for path in ordered_source_paths),
    )
    dataset_contract_summary = write_contract_artifacts(DEST_BUNDLE_ROOT)
    initial_only_rereview_summary = _summarize_initial_only_rereview_application(
        selected_row_paths=tuple(Path(path) for path in ordered_source_paths),
    )
    _write_json(TELEMETRY_DIR / "initial_only_rereview_application_summary.json", initial_only_rereview_summary)
    repair_summary = _aggregate_repair_summary(
        per_file_summaries,
        pair_batch_size=pair_batch_size,
        query_batch_pair_limit=query_batch_pair_limit,
    )
    if hwang_candidate_level_relabel is not None:
        repair_summary["hwang_candidate_level_relabel"] = hwang_candidate_level_relabel
    if s2and_stratified_split_refresh is not None:
        repair_summary["s2and_stratified_split_refresh"] = s2and_stratified_split_refresh
    repair_summary["dataset_contract"] = dataset_contract_summary
    repair_summary["initial_only_rereview"] = initial_only_rereview_summary
    _write_json(TELEMETRY_DIR / "repair_summary.json", repair_summary)
    _write_bundle_metadata(per_file_summaries=per_file_summaries)

    if args.skip_classic:
        payload = {
            "bundle_root": str(DEST_BUNDLE_ROOT.relative_to(REPO_ROOT)).replace("/", "\\"),
            "repair_summary": repair_summary,
            "classic_replay_skipped": True,
            "spool_db": str(spool_db_path.relative_to(REPO_ROOT)).replace("/", "\\"),
        }
        _write_json(TELEMETRY_DIR / "sample_run_summary.json", payload)
        print(json.dumps(payload, indent=2))
        return

    bundle = load_bundle(DEST_BUNDLE_ROOT)
    classic_summary = run_classic(bundle, SCRATCH_OUT)
    expected_metrics = expected_metrics_from_summary(classic_summary)
    _write_bundle_metadata(per_file_summaries=per_file_summaries, expected_metrics=expected_metrics)
    try:
        from scripts.sync_joint_safe_link_official_bundle_metadata import sync_bundle_metadata
    except ImportError:  # pragma: no cover - direct script execution path
        from sync_joint_safe_link_official_bundle_metadata import sync_bundle_metadata  # type: ignore

    sync_bundle_metadata(
        DEST_BUNDLE_ROOT,
        classic_summary,
        created_on=CREATED_ON,
        verification_json_path=SCRATCH_OUT / "metadata_sync_verification.json",
    )
    bundle = load_bundle(DEST_BUNDLE_ROOT)
    verification = {
        "summary": classic_summary,
        "expected": bundle.expected_metrics["classic"],
        "deltas": compare_to_expected(classic_summary, bundle.expected_metrics["classic"]),
        "repair_summary": repair_summary,
    }
    _write_json(SCRATCH_OUT / "verification.json", verification)
    _write_documentation(classic_summary=classic_summary, repair_summary=repair_summary)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
