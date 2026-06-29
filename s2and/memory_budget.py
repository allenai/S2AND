from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("s2and")

MEMORY_TELEMETRY_JSONL_ENV = "S2AND_MEMORY_TELEMETRY_JSONL"
_MEMORY_TELEMETRY_LOCK = threading.Lock()

AUTODETECT_RAM_SAFETY_FACTOR = 0.8
DEFAULT_SAFETY_MARGIN_FRACTION = 0.10
# Rust batch featurization defaults.
RUST_BATCH_BASE_CHUNK_PAIRS = 0  # Disabled - rely on memory-budget-derived limit
RUST_BATCH_MAX_CHUNK_PAIRS = 10_000
RUST_BATCH_STAGE_BUDGET_FRACTION = 0.25
RUST_BATCH_ROW_OVERHEAD_BYTES = 128
# Bundle 4 calibration (4 workload shapes: 37, 37, 49, 49 bytes/row); P95 = 49; 52 provides ~6% margin.
RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES = 52
RUST_BATCH_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PROMOTED_PHASE_A_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PROMOTED_PHASE_A_RETRIEVAL_PAIR_BYTES = 16
PROMOTED_PHASE_A_RETRIEVAL_ROW_BYTES = 512
PROMOTED_PHASE_A_PAIR_LABEL_BYTES = 8
PROMOTED_PHASE_A_DISTANCE_ROW_BYTES = 96
PROMOTED_PHASE_A_DEFAULT_FEATURE_COUNT = 70
PROMOTED_PHASE_A_DEFAULT_AGGREGATE_FEATURE_COUNT = 31
PROMOTED_PHASE_A_STAGE_BUDGET_FRACTION = 0.50


@dataclass(frozen=True)
class MemorySnapshot:
    total_ram_bytes: int
    total_ram_source: str
    current_rss_bytes: int
    current_rss_source: str
    safety_margin_bytes: int
    available_bytes: int
    # What fraction of total_ram_bytes is actually usable after subtracting RSS and safety margin.
    effective_available_fraction: float


@dataclass(frozen=True)
class RustBatchChunkPlan:
    total_ram_bytes: int
    total_ram_source: str
    current_rss_bytes: int
    current_rss_source: str
    available_bytes: int
    effective_available_fraction: float
    safety_margin_bytes: int
    stage_budget_fraction: float
    stage_budget_bytes: int
    base_chunk_pairs: int
    max_chunk_pairs: int
    row_overhead_bytes: int
    persistent_row_overhead_bytes: int
    fixed_overhead_bytes: int
    bytes_per_pair_row: int
    derived_chunk_pairs: int
    chunk_pairs: int
    total_rows: int
    full_feature_count: int
    selected_feature_count: int
    nameless_feature_count: int
    predicted_chunk_bytes: int
    predicted_features_matrix_bytes: int
    predicted_labels_bytes: int
    predicted_persistent_row_overhead_bytes: int
    predicted_fixed_overhead_bytes: int
    predicted_selected_features_bytes: int
    predicted_nameless_features_bytes: int
    predicted_stage_peak_delta_bytes: int
    predicted_stage_peak_rss_bytes: int


@dataclass(frozen=True)
class PromotedPhaseALimits:
    total_ram_bytes: int
    total_ram_source: str
    current_rss_bytes: int
    current_rss_source: str
    available_bytes: int
    effective_available_fraction: float
    safety_margin_bytes: int
    stage_budget_fraction: float
    stage_budget_bytes: int
    query_count: int
    max_query_batch_size: int
    query_batch_size: int
    component_count: int
    retrieval_top_k: int
    candidate_rows_per_query: int
    conservative_pairs_per_query: int
    hard_query_batch_size: int
    observed_query_count: int
    observed_candidate_rows_per_query: int
    observed_pairs_per_query: int
    observed_safety_multiplier: float
    operational_candidate_rows_per_query: int
    operational_pairs_per_query: int
    operational_estimate_source: str
    max_component_size: int
    predicted_candidate_rows_per_batch: int
    predicted_pairs_per_batch: int
    hard_predicted_candidate_rows_per_batch: int
    hard_predicted_pairs_per_batch: int
    retrieval_pair_bytes: int
    retrieval_row_bytes: int
    pair_label_bytes: int
    distance_row_bytes: int
    final_matrix_feature_count: int
    aggregate_feature_count: int
    fixed_overhead_bytes: int
    predicted_retrieval_pair_arrays_bytes: int
    predicted_retrieval_row_bytes: int
    predicted_pair_label_bytes: int
    predicted_aggregate_bytes: int
    predicted_distance_row_bytes: int
    predicted_final_matrix_bytes: int
    predicted_fixed_overhead_bytes: int
    predicted_persistent_bytes: int
    predicted_pair_chunk_bytes: int
    predicted_peak_delta_bytes: int
    predicted_peak_rss_bytes: int
    pair_chunk_pairs: int
    pair_chunk_count: int
    pair_chunk_stage_budget_bytes: int
    single_query_predicted_persistent_bytes: int
    single_query_exceeds_budget: bool


@dataclass(frozen=True)
class PredictionAccuracySummary:
    stage_name: str
    prediction_contract_version: str
    predicted_peak_delta_bytes: int
    predicted_peak_rss_bytes: int
    predicted_bytes: int
    rss_before_bytes: int
    rss_peak_bytes: int
    rss_after_bytes: int
    observed_peak_delta_bytes: int
    observed_end_delta_bytes: int
    prediction_error_ratio: float
    underpredicted: bool


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        converted = item()
        if converted is None or isinstance(converted, str | int | float | bool):
            return converted
    return str(value)


def emit_memory_telemetry(record: Mapping[str, Any]) -> None:
    """Append one structured memory telemetry record when configured by env var."""

    output_path_raw = os.environ.get(MEMORY_TELEMETRY_JSONL_ENV)
    if not output_path_raw:
        return
    output_path = Path(output_path_raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "event": "memory_telemetry",
    }
    for key, value in record.items():
        payload[str(key)] = _json_safe_value(value)

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    with _MEMORY_TELEMETRY_LOCK:
        with output_path.open("a", encoding="utf-8") as telemetry_file:
            telemetry_file.write(encoded)


def compute_stage_budget_bytes(available_bytes: int, stage_fraction: float) -> int:
    return max(1, int(float(stage_fraction) * float(max(1, int(available_bytes)))))


def compute_chunk_size(
    *,
    item_bytes: int,
    budget_bytes: int,
    fixed_overhead_bytes: int = 0,
    hard_limit_items: int | None = None,
    soft_limit_items: int | None = None,
) -> tuple[int, int]:
    """Return ``(chunk_size, derived_chunk_size)`` for a fixed byte budget."""

    budget_after_fixed = max(1, int(budget_bytes) - max(0, int(fixed_overhead_bytes)))
    derived_chunk_size = max(1, budget_after_fixed // max(1, int(item_bytes)))
    candidates = [derived_chunk_size]
    if hard_limit_items is not None and int(hard_limit_items) > 0:
        candidates.append(int(hard_limit_items))
    if soft_limit_items is not None and int(soft_limit_items) > 0:
        candidates.append(int(soft_limit_items))
    return max(1, min(candidates)), derived_chunk_size


def resolve_rust_batch_prediction_params() -> dict[str, int]:
    return {
        "base_chunk_pairs": RUST_BATCH_BASE_CHUNK_PAIRS,
        "max_chunk_pairs": RUST_BATCH_MAX_CHUNK_PAIRS,
        "row_overhead_bytes": RUST_BATCH_ROW_OVERHEAD_BYTES,
        "persistent_row_overhead_bytes": RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES,
        "fixed_overhead_bytes": RUST_BATCH_FIXED_OVERHEAD_BYTES,
    }


def validate_positive_total_ram_bytes(total_ram_bytes: int, *, source: str) -> int:
    try:
        parsed = int(total_ram_bytes)
    except ValueError as exc:
        raise ValueError(
            f"Invalid total_ram_bytes={total_ram_bytes!r} from {source}; expected a positive integer"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"Invalid total_ram_bytes={total_ram_bytes!r} from {source}; expected a positive integer")
    return parsed


def _is_windows() -> bool:
    return os.name == "nt"


def _psutil_virtual_memory_total_bytes_best_effort() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        total = int(psutil.virtual_memory().total)
    except Exception:
        return None
    if total > 0:
        return total
    return None


def _psutil_process_rss_bytes_best_effort() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    try:
        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        return None
    if rss >= 0:
        return rss
    return None


def _proc_meminfo_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    meminfo_path = "/proc/meminfo"
    if not os.path.exists(meminfo_path):
        return None, "unavailable"
    try:
        with open(meminfo_path, encoding="utf-8") as meminfo_file:
            for line in meminfo_file:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024, "proc_meminfo"
    except Exception:
        pass
    return None, "unavailable"


def _proc_status_rss_bytes_best_effort() -> tuple[int | None, str]:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return None, "unavailable"
    try:
        with open(status_path, encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024, "proc_status_vmrss"
    except Exception:
        pass
    return None, "unavailable"


def _windows_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    if not _is_windows():
        return None, "unavailable"
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None, "unavailable"
        if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, "unavailable"
        total = int(status.ullTotalPhys)
        if total > 0:
            return total, "winapi_globalmemorystatusex"
    except Exception as exc:
        logger.debug("Windows total RAM detection failed: %s", exc)
    return None, "unavailable"


def _windows_process_working_set_bytes_best_effort() -> tuple[int | None, str]:
    if not _is_windows():
        return None, "unavailable"
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None, "unavailable"
        process_handle = windll.kernel32.GetCurrentProcess()
        if not windll.psapi.GetProcessMemoryInfo(process_handle, ctypes.byref(counters), counters.cb):
            return None, "unavailable"
        rss = int(counters.WorkingSetSize)
        if rss >= 0:
            return rss, "winapi_process_working_set"
    except Exception as exc:
        logger.debug("Windows RSS detection failed: %s", exc)
    return None, "unavailable"


def detect_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    total = _psutil_virtual_memory_total_bytes_best_effort()
    if total is not None:
        return total, "psutil.virtual_memory"

    win_total, win_source = _windows_total_ram_bytes_best_effort()
    if win_total is not None:
        return win_total, str(win_source)

    proc_total, proc_source = _proc_meminfo_total_ram_bytes_best_effort()
    if proc_total is not None:
        return proc_total, str(proc_source)

    return None, "unavailable"


def detect_cgroup_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    cgroup_paths = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for cgroup_path in cgroup_paths:
        if not os.path.exists(cgroup_path):
            continue
        try:
            with open(cgroup_path, encoding="utf-8") as cgroup_file:
                raw_value = cgroup_file.read().strip().lower()
        except OSError:
            continue

        if raw_value in {"", "max"}:
            continue
        try:
            parsed = int(raw_value)
        except ValueError:
            continue
        if parsed <= 0:
            continue
        # cgroup "unlimited" values are often very large sentinel integers.
        if parsed >= (1 << 60):
            continue
        return parsed, f"cgroup:{cgroup_path}"

    return None, "unavailable"


def current_rss_bytes_best_effort(total_ram_bytes: int) -> tuple[int, str]:
    rss = _psutil_process_rss_bytes_best_effort()
    if rss is not None:
        return rss, "psutil_process_rss"

    win_rss, win_source = _windows_process_working_set_bytes_best_effort()
    if win_rss is not None:
        return win_rss, str(win_source)

    proc_rss, proc_source = _proc_status_rss_bytes_best_effort()
    if proc_rss is not None:
        return proc_rss, str(proc_source)

    logger.warning(
        "Unable to determine process RSS (psutil unavailable and no platform RSS fallback available); "
        "falling back to 50%% of total_ram_bytes=%d. Memory budgeting may be inaccurate. "
        "Install psutil for reliable RSS measurement.",
        total_ram_bytes,
    )
    return int(0.5 * total_ram_bytes), "fallback_half_total"


def gc_collect_and_log(stage_name: str) -> None:
    """Hint the garbage collector between stages to reduce stale RSS inflation.

    Calling ``gc.collect()`` encourages Python to release reference-counted
    objects that would otherwise inflate RSS seen by the next stage's snapshot.
    """
    import gc

    collected = gc.collect()
    if collected > 0:
        logger.info("Inter-stage GC after %s: collected %d objects", stage_name, collected)


def resolve_total_ram_bytes(
    total_ram_bytes: int | None = None,
    *,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    autodetect_safety_factor: float = AUTODETECT_RAM_SAFETY_FACTOR,
) -> tuple[int, str]:
    def _safety_factor_suffix(factor: float) -> str:
        percent_value = float(factor) * 100.0
        if float(percent_value).is_integer():
            return f"{int(percent_value)}pct"
        return f"{percent_value:g}pct"

    if total_ram_bytes is not None:
        return validate_positive_total_ram_bytes(total_ram_bytes, source="arg"), "arg"

    detect_cgroup = detect_cgroup_fn or detect_cgroup_total_ram_bytes_best_effort
    detect_total = detect_total_fn or detect_total_ram_bytes_best_effort

    cgroup_limit_bytes, cgroup_source = detect_cgroup()
    if cgroup_limit_bytes is not None:
        capped_cgroup = max(1, int(float(cgroup_limit_bytes) * autodetect_safety_factor))
        return capped_cgroup, f"{cgroup_source}_{_safety_factor_suffix(autodetect_safety_factor)}"

    detected, source = detect_total()
    if detected is None:
        raise RuntimeError("Unable to determine total RAM for chunked incremental; pass total_ram_bytes explicitly.")
    capped_detected = max(1, int(float(detected) * autodetect_safety_factor))
    return capped_detected, f"{source}_{_safety_factor_suffix(autodetect_safety_factor)}"


def memory_snapshot_for_stage(
    *,
    total_ram_bytes: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> MemorySnapshot:
    resolved_total_ram_bytes, total_ram_source = resolve_total_ram_bytes(
        total_ram_bytes,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
    )
    resolve_rss = current_rss_fn or current_rss_bytes_best_effort
    current_rss_bytes, current_rss_source = resolve_rss(resolved_total_ram_bytes)
    safety_margin_bytes = int(float(safety_margin_fraction) * float(resolved_total_ram_bytes))
    raw_available = resolved_total_ram_bytes - current_rss_bytes - safety_margin_bytes
    available_bytes = max(1, raw_available)
    if raw_available <= 0:
        effective_pct = 100.0 * float(current_rss_bytes) / float(max(1, resolved_total_ram_bytes))
        logger.warning(
            "Memory budget is degenerate: current_rss_bytes=%d (%.1f%% of total_ram_bytes=%d) "
            "exceeds usable headroom (safety_margin=%.0f%%). "
            "Chunk sizes will be minimal and throughput will be severely degraded. "
            "Consider passing a larger total_ram_bytes or reducing process memory usage.",
            current_rss_bytes,
            effective_pct,
            resolved_total_ram_bytes,
            safety_margin_fraction * 100.0,
        )
    effective_available_fraction = float(available_bytes) / float(max(1, resolved_total_ram_bytes))
    return MemorySnapshot(
        total_ram_bytes=resolved_total_ram_bytes,
        total_ram_source=total_ram_source,
        current_rss_bytes=current_rss_bytes,
        current_rss_source=current_rss_source,
        safety_margin_bytes=safety_margin_bytes,
        available_bytes=available_bytes,
        effective_available_fraction=effective_available_fraction,
    )


def compute_rust_batch_chunk_plan(
    *,
    num_features: int,
    total_pairs: int,
    total_rows: int | None = None,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_ram_bytes: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    stage_budget_fraction: float = RUST_BATCH_STAGE_BUDGET_FRACTION,
    base_chunk_pairs: int | None = None,
    max_chunk_pairs: int | None = None,
    row_overhead_bytes: int | None = None,
    persistent_row_overhead_bytes: int | None = None,
    fixed_overhead_bytes: int | None = None,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> RustBatchChunkPlan:
    resolved = resolve_rust_batch_prediction_params()
    if base_chunk_pairs is None:
        base_chunk_pairs = resolved["base_chunk_pairs"]
    if max_chunk_pairs is None:
        max_chunk_pairs = resolved["max_chunk_pairs"]
    if row_overhead_bytes is None:
        row_overhead_bytes = resolved["row_overhead_bytes"]
    if persistent_row_overhead_bytes is None:
        persistent_row_overhead_bytes = resolved["persistent_row_overhead_bytes"]
    if fixed_overhead_bytes is None:
        fixed_overhead_bytes = resolved["fixed_overhead_bytes"]

    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    stage_budget_bytes = compute_stage_budget_bytes(snapshot.available_bytes, stage_budget_fraction)
    full_feature_count = max(1, num_features)
    selected_feature_count_bounded = full_feature_count
    if selected_feature_count is not None:
        selected_feature_count_bounded = max(1, min(full_feature_count, selected_feature_count))
    nameless_feature_count_bounded = max(0, min(full_feature_count, nameless_feature_count))
    # Use selected + nameless for chunk sizing (upper bound on columns Rust produces).
    # When selected_feature_count is None, selected_feature_count_bounded == full_feature_count,
    # so behavior is unchanged for callers that don't specify feature counts.
    chunk_feature_count = max(1, selected_feature_count_bounded + nameless_feature_count_bounded)
    bytes_per_pair_row = max(1, chunk_feature_count * 8 + row_overhead_bytes)
    bounded_total_pairs = max(1, total_pairs)
    if int(max_chunk_pairs) < 0:
        raise ValueError(f"Invalid max_chunk_pairs={max_chunk_pairs}; expected >= 0")
    hard_limit_pairs = bounded_total_pairs
    if int(max_chunk_pairs) > 0:
        hard_limit_pairs = min(hard_limit_pairs, int(max_chunk_pairs))
    bounded_total_rows = bounded_total_pairs
    if total_rows is not None:
        bounded_total_rows = max(1, total_rows)
    chunk_pairs, derived_chunk_pairs = compute_chunk_size(
        item_bytes=bytes_per_pair_row,
        budget_bytes=stage_budget_bytes,
        fixed_overhead_bytes=fixed_overhead_bytes,
        hard_limit_items=hard_limit_pairs,
        soft_limit_items=base_chunk_pairs,
    )

    predicted_chunk_bytes = chunk_pairs * bytes_per_pair_row
    predicted_selected_features_bytes = bounded_total_rows * (selected_feature_count_bounded * 8)
    predicted_nameless_features_bytes = bounded_total_rows * (nameless_feature_count_bounded * 8)
    predicted_features_matrix_bytes = predicted_selected_features_bytes + predicted_nameless_features_bytes
    predicted_labels_bytes = bounded_total_rows * 8
    predicted_persistent_row_overhead_bytes = bounded_total_rows * max(0, persistent_row_overhead_bytes)
    predicted_fixed_overhead_bytes = max(0, fixed_overhead_bytes)
    predicted_stage_peak_delta_bytes = (
        predicted_features_matrix_bytes
        + predicted_labels_bytes
        + predicted_chunk_bytes
        + predicted_persistent_row_overhead_bytes
        + predicted_fixed_overhead_bytes
    )
    predicted_stage_peak_rss_bytes = snapshot.current_rss_bytes + predicted_stage_peak_delta_bytes
    return RustBatchChunkPlan(
        total_ram_bytes=snapshot.total_ram_bytes,
        total_ram_source=snapshot.total_ram_source,
        current_rss_bytes=snapshot.current_rss_bytes,
        current_rss_source=snapshot.current_rss_source,
        available_bytes=snapshot.available_bytes,
        effective_available_fraction=snapshot.effective_available_fraction,
        safety_margin_bytes=snapshot.safety_margin_bytes,
        stage_budget_fraction=float(stage_budget_fraction),
        stage_budget_bytes=stage_budget_bytes,
        base_chunk_pairs=max(0, base_chunk_pairs),
        max_chunk_pairs=max(0, int(max_chunk_pairs)),
        row_overhead_bytes=max(0, row_overhead_bytes),
        persistent_row_overhead_bytes=max(0, persistent_row_overhead_bytes),
        fixed_overhead_bytes=predicted_fixed_overhead_bytes,
        bytes_per_pair_row=bytes_per_pair_row,
        derived_chunk_pairs=derived_chunk_pairs,
        chunk_pairs=chunk_pairs,
        total_rows=bounded_total_rows,
        full_feature_count=full_feature_count,
        selected_feature_count=selected_feature_count_bounded,
        nameless_feature_count=nameless_feature_count_bounded,
        predicted_chunk_bytes=predicted_chunk_bytes,
        predicted_features_matrix_bytes=predicted_features_matrix_bytes,
        predicted_labels_bytes=predicted_labels_bytes,
        predicted_persistent_row_overhead_bytes=predicted_persistent_row_overhead_bytes,
        predicted_fixed_overhead_bytes=predicted_fixed_overhead_bytes,
        predicted_selected_features_bytes=predicted_selected_features_bytes,
        predicted_nameless_features_bytes=predicted_nameless_features_bytes,
        predicted_stage_peak_delta_bytes=predicted_stage_peak_delta_bytes,
        predicted_stage_peak_rss_bytes=predicted_stage_peak_rss_bytes,
    )


def _largest_sum(values: list[int], count: int) -> int:
    if count <= 0 or not values:
        return 0
    return int(sum(sorted((max(0, int(value)) for value in values), reverse=True)[:count]))


def compute_promoted_phase_a_limits(
    *,
    query_count: int,
    component_sizes: Mapping[Any, int] | list[int] | tuple[int, ...],
    retrieval_top_k: int,
    total_ram_bytes: int | None = None,
    max_query_batch_size: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    stage_budget_fraction: float = PROMOTED_PHASE_A_STAGE_BUDGET_FRACTION,
    final_matrix_feature_count: int = PROMOTED_PHASE_A_DEFAULT_FEATURE_COUNT,
    aggregate_feature_count: int = PROMOTED_PHASE_A_DEFAULT_AGGREGATE_FEATURE_COUNT,
    retrieval_pair_bytes: int = PROMOTED_PHASE_A_RETRIEVAL_PAIR_BYTES,
    retrieval_row_bytes: int = PROMOTED_PHASE_A_RETRIEVAL_ROW_BYTES,
    pair_label_bytes: int = PROMOTED_PHASE_A_PAIR_LABEL_BYTES,
    distance_row_bytes: int = PROMOTED_PHASE_A_DISTANCE_ROW_BYTES,
    fixed_overhead_bytes: int = PROMOTED_PHASE_A_FIXED_OVERHEAD_BYTES,
    observed_query_count: int = 0,
    observed_candidate_rows_per_query: int | None = None,
    observed_pairs_per_query: int | None = None,
    candidate_rows_per_query_floor: int | None = None,
    pairs_per_query_floor: int | None = None,
    candidate_rows_total_floor: int | None = None,
    pairs_total_floor: int | None = None,
    observed_safety_multiplier: float = 2.0,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> PromotedPhaseALimits:
    """Compute conservative query-batch limits for promoted incremental Phase A.

    The planner sizes the retrieval-owned ``LinkerCandidateBatch`` before Rust
    retrieval allocation and reuses the Rust pair chunk planner for pair scoring.
    ``component_sizes`` should contain the current seeded component sizes.
    """

    parsed_query_count = int(query_count)
    if parsed_query_count < 0:
        raise ValueError(f"query_count must be >= 0, got {query_count}")
    parsed_top_k = int(retrieval_top_k)
    if parsed_top_k <= 0:
        raise ValueError(f"retrieval_top_k must be positive, got {retrieval_top_k}")
    size_values: Iterable[int] = (
        cast(Iterable[int], component_sizes.values()) if isinstance(component_sizes, Mapping) else component_sizes
    )
    sizes = [max(0, int(value)) for value in size_values]
    sizes = [size for size in sizes if size > 0]
    component_count = len(sizes)
    top_k_candidate_rows_per_query = min(parsed_top_k, component_count)
    top_k_pairs_per_query = _largest_sum(sizes, top_k_candidate_rows_per_query)
    max_component_size = max(sizes, default=0)
    parsed_observed_query_count = max(0, int(observed_query_count))
    multiplier = float(observed_safety_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError(f"observed_safety_multiplier must be finite and >= 1.0, got {observed_safety_multiplier}")
    observed_rows = 0 if observed_candidate_rows_per_query is None else max(0, int(observed_candidate_rows_per_query))
    observed_pairs = 0 if observed_pairs_per_query is None else max(0, int(observed_pairs_per_query))
    row_floor = 0 if candidate_rows_per_query_floor is None else max(0, int(candidate_rows_per_query_floor))
    pair_floor = 0 if pairs_per_query_floor is None else max(0, int(pairs_per_query_floor))
    row_total_floor = 0 if candidate_rows_total_floor is None else max(0, int(candidate_rows_total_floor))
    pair_total_floor = 0 if pairs_total_floor is None else max(0, int(pairs_total_floor))
    row_total_per_query_floor = (
        int(math.ceil(float(row_total_floor) / float(parsed_query_count)))
        if parsed_query_count > 0 and row_total_floor > 0
        else 0
    )
    pair_total_per_query_floor = (
        int(math.ceil(float(pair_total_floor) / float(parsed_query_count)))
        if parsed_query_count > 0 and pair_total_floor > 0
        else 0
    )
    candidate_rows_per_query = min(
        component_count,
        max(
            top_k_candidate_rows_per_query,
            row_floor,
            observed_rows if parsed_observed_query_count > 0 and observed_candidate_rows_per_query is not None else 0,
        ),
    )
    conservative_pairs_per_query = max(
        top_k_pairs_per_query,
        pair_floor,
        observed_pairs if parsed_observed_query_count > 0 and observed_pairs_per_query is not None else 0,
    )
    if parsed_observed_query_count > 0 and observed_candidate_rows_per_query is not None:
        observed_operational_rows = int(math.ceil(float(observed_rows) * multiplier))
        row_floor_for_operational = row_floor if row_total_per_query_floor == 0 else row_total_per_query_floor
        operational_candidate_rows_per_query = min(
            candidate_rows_per_query,
            max(row_floor_for_operational, observed_operational_rows),
        )
    elif row_total_per_query_floor > 0:
        operational_candidate_rows_per_query = max(top_k_candidate_rows_per_query, row_total_per_query_floor)
    else:
        operational_candidate_rows_per_query = candidate_rows_per_query
    if parsed_observed_query_count > 0 and observed_pairs_per_query is not None:
        observed_operational_pairs = int(math.ceil(float(observed_pairs) * multiplier))
        pair_floor_for_operational = pair_floor if pair_total_per_query_floor == 0 else pair_total_per_query_floor
        operational_pairs_per_query = min(
            conservative_pairs_per_query,
            max(pair_floor_for_operational, observed_operational_pairs),
        )
        operational_estimate_source = "observed_probe"
    elif (
        row_floor > top_k_candidate_rows_per_query
        or pair_floor > top_k_pairs_per_query
        or row_total_floor > parsed_query_count * top_k_candidate_rows_per_query
        or pair_total_floor > parsed_query_count * top_k_pairs_per_query
    ):
        operational_pairs_per_query = (
            max(top_k_pairs_per_query, pair_total_per_query_floor)
            if pair_total_per_query_floor > 0
            else conservative_pairs_per_query
        )
        operational_estimate_source = "orcid_fanout"
    else:
        operational_pairs_per_query = conservative_pairs_per_query
        operational_estimate_source = "top_k_largest_components"

    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    stage_budget_bytes = compute_stage_budget_bytes(snapshot.available_bytes, stage_budget_fraction)
    # An explicitly supplied batch size must be positive, regardless of query_count.
    # A None threshold with query_count == 0 is valid: the planner short-circuits to a
    # zero-size batch below (no queries to batch), so it must not be rejected here.
    if max_query_batch_size is not None and int(max_query_batch_size) <= 0:
        raise ValueError(f"max_query_batch_size must be positive, got {max_query_batch_size}")
    max_batch = parsed_query_count if max_query_batch_size is None else int(max_query_batch_size)
    max_batch = max(1, min(parsed_query_count if parsed_query_count > 0 else 1, max_batch))

    row_state_bytes_per_row = (
        int(aggregate_feature_count) * 3 * 8 + int(distance_row_bytes) + int(final_matrix_feature_count) * 4
    )
    hard_persistent_bytes_per_query = conservative_pairs_per_query * (
        int(retrieval_pair_bytes) + int(pair_label_bytes)
    ) + candidate_rows_per_query * (int(retrieval_row_bytes) + row_state_bytes_per_row)
    operational_persistent_bytes_per_query = operational_pairs_per_query * (
        int(retrieval_pair_bytes) + int(pair_label_bytes)
    ) + operational_candidate_rows_per_query * (int(retrieval_row_bytes) + row_state_bytes_per_row)
    single_query_predicted_persistent_bytes = int(fixed_overhead_bytes) + hard_persistent_bytes_per_query
    if parsed_query_count > 0 and single_query_predicted_persistent_bytes > stage_budget_bytes:
        raise MemoryError(
            "Promoted incremental linker cannot fit a single query under the memory budget: "
            f"single_query_predicted_persistent_bytes={int(single_query_predicted_persistent_bytes)} "
            f"stage_budget_bytes={int(stage_budget_bytes)} "
            f"total_ram_bytes={int(snapshot.total_ram_bytes)} "
            f"current_rss_bytes={int(snapshot.current_rss_bytes)} "
            f"safety_margin_bytes={int(snapshot.safety_margin_bytes)}"
        )
    if parsed_query_count == 0:
        query_batch_size = 0
        hard_query_batch_size = 0
    elif operational_persistent_bytes_per_query <= 0:
        query_batch_size = max_batch
        hard_query_batch_size = max_batch
    else:
        query_batch_size, _ = compute_chunk_size(
            item_bytes=operational_persistent_bytes_per_query,
            budget_bytes=stage_budget_bytes,
            fixed_overhead_bytes=fixed_overhead_bytes,
            hard_limit_items=max_batch,
        )
        hard_query_batch_size, _ = compute_chunk_size(
            item_bytes=max(1, hard_persistent_bytes_per_query),
            budget_bytes=stage_budget_bytes,
            fixed_overhead_bytes=fixed_overhead_bytes,
            hard_limit_items=max_batch,
        )

    predicted_candidate_rows_per_batch = int(query_batch_size) * operational_candidate_rows_per_query
    predicted_pairs_per_batch = int(query_batch_size) * operational_pairs_per_query
    hard_predicted_candidate_rows_per_batch = int(query_batch_size) * candidate_rows_per_query
    hard_predicted_pairs_per_batch = int(query_batch_size) * conservative_pairs_per_query
    if int(query_batch_size) == parsed_query_count:
        if row_total_floor > 0 and str(operational_estimate_source) == "orcid_fanout":
            predicted_candidate_rows_per_batch = row_total_floor
        else:
            predicted_candidate_rows_per_batch = max(predicted_candidate_rows_per_batch, row_total_floor)
        if pair_total_floor > 0 and str(operational_estimate_source) == "orcid_fanout":
            predicted_pairs_per_batch = pair_total_floor
        else:
            predicted_pairs_per_batch = max(predicted_pairs_per_batch, pair_total_floor)
        hard_predicted_candidate_rows_per_batch = max(hard_predicted_candidate_rows_per_batch, row_total_floor)
        hard_predicted_pairs_per_batch = max(hard_predicted_pairs_per_batch, pair_total_floor)
    predicted_retrieval_pair_arrays_bytes = predicted_pairs_per_batch * int(retrieval_pair_bytes)
    predicted_retrieval_row_bytes = predicted_candidate_rows_per_batch * int(retrieval_row_bytes)
    predicted_pair_label_bytes = predicted_pairs_per_batch * int(pair_label_bytes)
    predicted_aggregate_bytes = predicted_candidate_rows_per_batch * int(aggregate_feature_count) * 3 * 8
    predicted_distance_row_bytes = predicted_candidate_rows_per_batch * int(distance_row_bytes)
    predicted_final_matrix_bytes = predicted_candidate_rows_per_batch * int(final_matrix_feature_count) * 4
    predicted_fixed_overhead_bytes = int(fixed_overhead_bytes)
    predicted_persistent_bytes = (
        predicted_retrieval_pair_arrays_bytes
        + predicted_retrieval_row_bytes
        + predicted_pair_label_bytes
        + predicted_aggregate_bytes
        + predicted_distance_row_bytes
        + predicted_final_matrix_bytes
        + predicted_fixed_overhead_bytes
    )

    # Match s2and.incremental_linking.linker_pairwise.compute_linker_pair_chunk_plan
    # without importing the incremental-linking package from this core utility.
    pair_memory_feature_count = max(1, int(final_matrix_feature_count) + int(aggregate_feature_count) * 3 + 1)
    pair_plan = compute_rust_batch_chunk_plan(
        num_features=pair_memory_feature_count,
        total_pairs=max(0, predicted_pairs_per_batch),
        total_rows=max(0, predicted_candidate_rows_per_batch),
        total_ram_bytes=snapshot.total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=lambda: (snapshot.total_ram_bytes, snapshot.total_ram_source),
        detect_total_fn=lambda: (snapshot.total_ram_bytes, snapshot.total_ram_source),
        current_rss_fn=lambda _total: (snapshot.current_rss_bytes, snapshot.current_rss_source),
    )
    predicted_pair_chunk_bytes = int(pair_plan.predicted_chunk_bytes)
    predicted_peak_delta_bytes = predicted_persistent_bytes + predicted_pair_chunk_bytes
    predicted_peak_rss_bytes = snapshot.current_rss_bytes + predicted_peak_delta_bytes
    pair_chunk_pairs = int(pair_plan.chunk_pairs)
    pair_chunk_count = (
        int((predicted_pairs_per_batch + pair_chunk_pairs - 1) // pair_chunk_pairs)
        if predicted_pairs_per_batch > 0
        else 0
    )
    return PromotedPhaseALimits(
        total_ram_bytes=snapshot.total_ram_bytes,
        total_ram_source=snapshot.total_ram_source,
        current_rss_bytes=snapshot.current_rss_bytes,
        current_rss_source=snapshot.current_rss_source,
        available_bytes=snapshot.available_bytes,
        effective_available_fraction=snapshot.effective_available_fraction,
        safety_margin_bytes=snapshot.safety_margin_bytes,
        stage_budget_fraction=float(stage_budget_fraction),
        stage_budget_bytes=stage_budget_bytes,
        query_count=parsed_query_count,
        max_query_batch_size=max_batch,
        query_batch_size=int(query_batch_size),
        component_count=component_count,
        retrieval_top_k=parsed_top_k,
        candidate_rows_per_query=candidate_rows_per_query,
        conservative_pairs_per_query=conservative_pairs_per_query,
        hard_query_batch_size=int(hard_query_batch_size),
        observed_query_count=parsed_observed_query_count,
        observed_candidate_rows_per_query=observed_rows,
        observed_pairs_per_query=observed_pairs,
        observed_safety_multiplier=multiplier,
        operational_candidate_rows_per_query=operational_candidate_rows_per_query,
        operational_pairs_per_query=operational_pairs_per_query,
        operational_estimate_source=operational_estimate_source,
        max_component_size=max_component_size,
        predicted_candidate_rows_per_batch=predicted_candidate_rows_per_batch,
        predicted_pairs_per_batch=predicted_pairs_per_batch,
        hard_predicted_candidate_rows_per_batch=hard_predicted_candidate_rows_per_batch,
        hard_predicted_pairs_per_batch=hard_predicted_pairs_per_batch,
        retrieval_pair_bytes=int(retrieval_pair_bytes),
        retrieval_row_bytes=int(retrieval_row_bytes),
        pair_label_bytes=int(pair_label_bytes),
        distance_row_bytes=int(distance_row_bytes),
        final_matrix_feature_count=int(final_matrix_feature_count),
        aggregate_feature_count=int(aggregate_feature_count),
        fixed_overhead_bytes=int(fixed_overhead_bytes),
        predicted_retrieval_pair_arrays_bytes=predicted_retrieval_pair_arrays_bytes,
        predicted_retrieval_row_bytes=predicted_retrieval_row_bytes,
        predicted_pair_label_bytes=predicted_pair_label_bytes,
        predicted_aggregate_bytes=predicted_aggregate_bytes,
        predicted_distance_row_bytes=predicted_distance_row_bytes,
        predicted_final_matrix_bytes=predicted_final_matrix_bytes,
        predicted_fixed_overhead_bytes=predicted_fixed_overhead_bytes,
        predicted_persistent_bytes=predicted_persistent_bytes,
        predicted_pair_chunk_bytes=predicted_pair_chunk_bytes,
        predicted_peak_delta_bytes=predicted_peak_delta_bytes,
        predicted_peak_rss_bytes=predicted_peak_rss_bytes,
        pair_chunk_pairs=pair_chunk_pairs,
        pair_chunk_count=pair_chunk_count,
        pair_chunk_stage_budget_bytes=int(pair_plan.stage_budget_bytes),
        single_query_predicted_persistent_bytes=single_query_predicted_persistent_bytes,
        single_query_exceeds_budget=single_query_predicted_persistent_bytes > stage_budget_bytes,
    )


def summarize_prediction_accuracy(
    *,
    stage_name: str,
    predicted_peak_delta_bytes: int | None = None,
    predicted_bytes: int | None = None,
    rss_before_bytes: int,
    rss_peak_bytes: int,
    rss_after_bytes: int,
) -> PredictionAccuracySummary:
    predicted_delta = predicted_peak_delta_bytes if predicted_peak_delta_bytes is not None else predicted_bytes
    if predicted_delta is None:
        raise ValueError("Either predicted_peak_delta_bytes or predicted_bytes must be provided.")

    bounded_predicted_delta = max(1, predicted_delta)
    bounded_before = max(0, rss_before_bytes)
    bounded_peak = max(bounded_before, rss_peak_bytes)
    bounded_after = max(0, rss_after_bytes)
    observed_peak_delta_bytes = max(0, bounded_peak - bounded_before)
    observed_end_delta_bytes = max(0, bounded_after - bounded_before)
    predicted_peak_rss_bytes = bounded_before + bounded_predicted_delta
    prediction_error_ratio = float(observed_peak_delta_bytes) / float(bounded_predicted_delta)
    return PredictionAccuracySummary(
        stage_name=stage_name,
        prediction_contract_version="delta_v1",
        predicted_peak_delta_bytes=bounded_predicted_delta,
        predicted_peak_rss_bytes=predicted_peak_rss_bytes,
        # Backward-compatible alias; prefer predicted_peak_delta_bytes.
        predicted_bytes=bounded_predicted_delta,
        rss_before_bytes=bounded_before,
        rss_peak_bytes=bounded_peak,
        rss_after_bytes=bounded_after,
        observed_peak_delta_bytes=observed_peak_delta_bytes,
        observed_end_delta_bytes=observed_end_delta_bytes,
        prediction_error_ratio=prediction_error_ratio,
        underpredicted=bool(observed_peak_delta_bytes > bounded_predicted_delta),
    )
