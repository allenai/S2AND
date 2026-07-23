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
_MEMORY_TELEMETRY_JSONL_PATH: Path | None = None

AUTODETECT_RAM_SAFETY_FACTOR = 0.8
DEFAULT_SAFETY_MARGIN_FRACTION = 0.10
# Rust batch featurization defaults.
RUST_BATCH_BASE_CHUNK_PAIRS = 0  # Disabled - rely on memory-budget-derived limit
RUST_BATCH_MAX_CHUNK_PAIRS = 10_000
RUST_BATCH_STAGE_BUDGET_FRACTION = 0.25
RUST_BATCH_ROW_OVERHEAD_BYTES = 128
BORROWED_SIGNATURE_INDEX_REMAP_BYTES_PER_PAIR = 2 * 4
# Bundle 4 calibration (4 workload shapes: 37, 37, 49, 49 bytes/row); P95 = 49; 52 provides ~6% margin.
RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES = 52
RUST_BATCH_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PROMOTED_PHASE_A_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PROMOTED_PHASE_A_RETRIEVAL_PAIR_BYTES = 16
PROMOTED_PHASE_A_RETRIEVAL_ROW_BYTES = 512
PROMOTED_PHASE_A_PAIR_LABEL_BYTES = 8
PROMOTED_PHASE_A_DISTANCE_ROW_BYTES = 96
PROMOTED_PHASE_A_STAGE_BUDGET_FRACTION = 0.50
NATIVE_SCORER_STAGE_BUDGET_FRACTION = 0.50


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
    index_remap_bytes_per_pair: int = 0
    predicted_index_remap_bytes: int = 0


@dataclass(frozen=True)
class PromotedPhaseALimits:
    """The three Phase-A values consumed outside the memory calculation."""

    query_batch_size: int
    predicted_peak_delta_bytes: int
    predicted_peak_rss_bytes: int


@dataclass(frozen=True)
class NativeScorerChunkPlan:
    """Bounded scratch plan for the Rust LightGBM owned-input copy."""

    total_ram_bytes: int
    current_rss_bytes: int
    available_bytes: int
    stage_budget_bytes: int
    row_count: int
    feature_count: int
    input_bytes_per_row: int
    output_bytes_per_row: int
    persistent_output_bytes: int
    chunk_rows: int
    chunk_count: int
    predicted_chunk_input_bytes: int
    predicted_chunk_output_bytes: int
    predicted_peak_delta_bytes: int
    predicted_peak_rss_bytes: int


@dataclass(frozen=True)
class PromotedComponentSizeSummary:
    """Immutable component-size statistics reused by Phase A RSS refreshes."""

    sizes_descending: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the precomputed ordering consumed by constant-time refreshes."""

        normalized = tuple(int(size) for size in self.sizes_descending)
        if any(size <= 0 for size in normalized):
            raise ValueError("Promoted component-size summaries must contain only positive sizes")
        if any(left < right for left, right in zip(normalized, normalized[1:], strict=False)):
            raise ValueError("Promoted component-size summaries must be sorted descending")
        object.__setattr__(self, "sizes_descending", normalized)

    @property
    def component_count(self) -> int:
        """Return the number of positive-size components."""

        return len(self.sizes_descending)

    @property
    def max_component_size(self) -> int:
        """Return the largest component size, or zero when there are none."""

        return self.sizes_descending[0] if self.sizes_descending else 0

    def top_k_totals(self, retrieval_top_k: int) -> tuple[int, int]:
        """Return candidate rows and pairs for the largest ``retrieval_top_k`` components."""

        count = min(max(0, int(retrieval_top_k)), self.component_count)
        return count, int(sum(self.sizes_descending[:count]))


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


def configure_memory_telemetry_jsonl(path: str | Path | None) -> None:
    """Configure the optional JSONL sink for structured memory telemetry."""

    global _MEMORY_TELEMETRY_JSONL_PATH
    _MEMORY_TELEMETRY_JSONL_PATH = None if path is None else Path(path)


def memory_telemetry_jsonl_path() -> Path | None:
    """Return the configured structured memory telemetry sink, if any."""

    if _MEMORY_TELEMETRY_JSONL_PATH is not None:
        return _MEMORY_TELEMETRY_JSONL_PATH
    env_path = os.environ.get(MEMORY_TELEMETRY_JSONL_ENV)
    if env_path is None or not env_path.strip():
        return None
    return Path(env_path)


def emit_memory_telemetry(record: Mapping[str, Any]) -> None:
    """Append one structured memory telemetry record when a sink is configured."""

    output_path = memory_telemetry_jsonl_path()
    if output_path is None:
        return
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


def compute_native_scorer_chunk_plan(
    *,
    row_count: int,
    feature_count: int,
    input_itemsize: int = 4,
    total_ram_bytes: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    stage_budget_fraction: float = NATIVE_SCORER_STAGE_BUDGET_FRACTION,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> NativeScorerChunkPlan:
    """Bound the native scorer's float32 input copy and float64 output arrays."""

    rows = int(row_count)
    features = int(feature_count)
    if rows < 0:
        raise ValueError(f"row_count must be non-negative, got {row_count}")
    if features <= 0:
        raise ValueError(f"feature_count must be positive, got {feature_count}")
    itemsize = int(input_itemsize)
    if itemsize not in {4, 8}:
        raise ValueError(f"input_itemsize must be 4 or 8, got {input_itemsize}")
    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    stage_budget_bytes = compute_stage_budget_bytes(snapshot.available_bytes, stage_budget_fraction)
    input_bytes_per_row = features * itemsize
    output_bytes_per_row = 8
    persistent_output_bytes = rows * output_bytes_per_row
    scratch_bytes_per_row = input_bytes_per_row + output_bytes_per_row
    if rows == 0:
        chunk_rows = 0
        chunk_count = 0
    elif rows * scratch_bytes_per_row <= stage_budget_bytes:
        chunk_rows = rows
        chunk_count = 1
    else:
        one_row_required_bytes = persistent_output_bytes + scratch_bytes_per_row
        if one_row_required_bytes > stage_budget_bytes:
            raise MemoryError(
                "Native LightGBM scorer cannot fit one scratch row under the memory budget: "
                f"persistent_output_bytes={persistent_output_bytes} "
                f"scratch_bytes_per_row={scratch_bytes_per_row} "
                f"stage_budget_bytes={stage_budget_bytes} "
                f"current_rss_bytes={snapshot.current_rss_bytes}"
            )
        chunk_rows, _ = compute_chunk_size(
            item_bytes=scratch_bytes_per_row,
            budget_bytes=stage_budget_bytes,
            fixed_overhead_bytes=persistent_output_bytes,
            hard_limit_items=rows,
        )
        chunk_count = (rows + chunk_rows - 1) // chunk_rows
    predicted_chunk_input_bytes = chunk_rows * input_bytes_per_row
    predicted_chunk_output_bytes = chunk_rows * output_bytes_per_row
    predicted_peak_delta_bytes = (
        predicted_chunk_input_bytes + predicted_chunk_output_bytes + (persistent_output_bytes if chunk_count > 1 else 0)
    )
    return NativeScorerChunkPlan(
        total_ram_bytes=snapshot.total_ram_bytes,
        current_rss_bytes=snapshot.current_rss_bytes,
        available_bytes=snapshot.available_bytes,
        stage_budget_bytes=stage_budget_bytes,
        row_count=rows,
        feature_count=features,
        input_bytes_per_row=input_bytes_per_row,
        output_bytes_per_row=output_bytes_per_row,
        persistent_output_bytes=persistent_output_bytes,
        chunk_rows=chunk_rows,
        chunk_count=chunk_count,
        predicted_chunk_input_bytes=predicted_chunk_input_bytes,
        predicted_chunk_output_bytes=predicted_chunk_output_bytes,
        predicted_peak_delta_bytes=predicted_peak_delta_bytes,
        predicted_peak_rss_bytes=snapshot.current_rss_bytes + predicted_peak_delta_bytes,
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
    index_remap_bytes_per_pair: int = 0,
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
    parsed_index_remap_bytes_per_pair = int(index_remap_bytes_per_pair)
    if parsed_index_remap_bytes_per_pair < 0:
        raise ValueError(f"index_remap_bytes_per_pair must be non-negative, got {index_remap_bytes_per_pair}")
    bytes_per_pair_row = max(
        1,
        chunk_feature_count * 8 + row_overhead_bytes + parsed_index_remap_bytes_per_pair,
    )
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
    predicted_index_remap_bytes = chunk_pairs * parsed_index_remap_bytes_per_pair
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
        index_remap_bytes_per_pair=parsed_index_remap_bytes_per_pair,
        predicted_index_remap_bytes=predicted_index_remap_bytes,
    )


def summarize_promoted_component_sizes(
    component_sizes: Mapping[Any, int] | Iterable[int],
) -> PromotedComponentSizeSummary:
    """Normalize and sort seeded component sizes once for repeated RSS planning."""

    size_values = (
        cast(Iterable[int], component_sizes.values()) if isinstance(component_sizes, Mapping) else component_sizes
    )
    return PromotedComponentSizeSummary(
        sizes_descending=tuple(
            sorted(
                (size for value in size_values if (size := max(0, int(value))) > 0),
                reverse=True,
            )
        )
    )


def compute_promoted_phase_a_limits(
    *,
    query_count: int,
    component_sizes: Mapping[Any, int] | list[int] | tuple[int, ...] | PromotedComponentSizeSummary,
    retrieval_top_k: int,
    final_matrix_feature_count: int,
    pairwise_matrix_feature_count: int,
    aggregate_feature_count: int,
    total_ram_bytes: int | None = None,
    max_query_batch_size: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    stage_budget_fraction: float = PROMOTED_PHASE_A_STAGE_BUDGET_FRACTION,
    retrieval_pair_bytes: int = PROMOTED_PHASE_A_RETRIEVAL_PAIR_BYTES,
    retrieval_row_bytes: int = PROMOTED_PHASE_A_RETRIEVAL_ROW_BYTES,
    pair_label_bytes: int = PROMOTED_PHASE_A_PAIR_LABEL_BYTES,
    distance_row_bytes: int = PROMOTED_PHASE_A_DISTANCE_ROW_BYTES,
    fixed_overhead_bytes: int = PROMOTED_PHASE_A_FIXED_OVERHEAD_BYTES,
    candidate_rows_per_query_floor: int | None = None,
    pairs_per_query_floor: int | None = None,
    candidate_rows_total_floor: int | None = None,
    pairs_total_floor: int | None = None,
    retrieval_payload_resident: bool = False,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> PromotedPhaseALimits:
    """Compute conservative query-batch limits for promoted incremental Phase A.

    The planner sizes the retrieval-owned ``LinkerCandidateBatch`` before Rust
    retrieval allocation and reuses the Rust pair chunk planner for pair scoring.
    ``component_sizes`` should contain the current seeded component sizes. Set
    ``retrieval_payload_resident`` after the raw plan has been allocated so its
    pair and row arrays are observed in RSS instead of reserved a second time.
    """

    parsed_query_count = int(query_count)
    if parsed_query_count < 0:
        raise ValueError(f"query_count must be >= 0, got {query_count}")
    parsed_top_k = int(retrieval_top_k)
    if parsed_top_k <= 0:
        raise ValueError(f"retrieval_top_k must be positive, got {retrieval_top_k}")
    parsed_final_matrix_feature_count = int(final_matrix_feature_count)
    parsed_pairwise_matrix_feature_count = int(pairwise_matrix_feature_count)
    parsed_aggregate_feature_count = int(aggregate_feature_count)
    if parsed_final_matrix_feature_count <= 0:
        raise ValueError("final_matrix_feature_count must be positive")
    if parsed_pairwise_matrix_feature_count <= 0:
        raise ValueError("pairwise_matrix_feature_count must be positive")
    if parsed_aggregate_feature_count < 0:
        raise ValueError("aggregate_feature_count must be nonnegative")
    component_summary = (
        component_sizes
        if isinstance(component_sizes, PromotedComponentSizeSummary)
        else summarize_promoted_component_sizes(component_sizes)
    )
    component_count = component_summary.component_count
    top_k_candidate_rows_per_query, top_k_pairs_per_query = component_summary.top_k_totals(parsed_top_k)
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
        max(top_k_candidate_rows_per_query, row_floor),
    )
    conservative_pairs_per_query = max(top_k_pairs_per_query, pair_floor)
    operational_candidate_rows_per_query = (
        max(top_k_candidate_rows_per_query, row_total_per_query_floor)
        if row_total_per_query_floor > 0
        else candidate_rows_per_query
    )
    operational_pairs_per_query = (
        max(top_k_pairs_per_query, pair_total_per_query_floor)
        if pair_total_per_query_floor > 0
        else conservative_pairs_per_query
    )
    orcid_floor_exceeds_top_k = (
        row_floor > top_k_candidate_rows_per_query
        or pair_floor > top_k_pairs_per_query
        or row_total_floor > parsed_query_count * top_k_candidate_rows_per_query
        or pair_total_floor > parsed_query_count * top_k_pairs_per_query
    )

    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    stage_budget_bytes = compute_stage_budget_bytes(snapshot.available_bytes, stage_budget_fraction)
    if max_query_batch_size is None:
        max_batch = 1 if parsed_query_count == 0 else parsed_query_count
    else:
        max_batch = int(max_query_batch_size)
    if max_batch <= 0:
        raise ValueError(f"max_query_batch_size must be positive, got {max_query_batch_size}")
    max_batch = max(1, min(parsed_query_count if parsed_query_count > 0 else 1, max_batch))

    scorer_input_bytes_per_row = parsed_final_matrix_feature_count * 4
    scorer_output_bytes_per_row = 8
    scorer_scratch_bytes_per_row = scorer_input_bytes_per_row + scorer_output_bytes_per_row
    row_state_bytes_per_row = (
        parsed_aggregate_feature_count * 3 * 8
        + int(distance_row_bytes)
        + parsed_final_matrix_feature_count * 4
        + scorer_output_bytes_per_row
    )
    pending_retrieval_pair_bytes = 0 if retrieval_payload_resident else int(retrieval_pair_bytes)
    pending_retrieval_row_bytes = 0 if retrieval_payload_resident else int(retrieval_row_bytes)
    hard_persistent_bytes_per_query = conservative_pairs_per_query * (
        pending_retrieval_pair_bytes + int(pair_label_bytes)
    ) + candidate_rows_per_query * (pending_retrieval_row_bytes + row_state_bytes_per_row)
    operational_persistent_bytes_per_query = operational_pairs_per_query * (
        pending_retrieval_pair_bytes + int(pair_label_bytes)
    ) + operational_candidate_rows_per_query * (pending_retrieval_row_bytes + row_state_bytes_per_row)
    single_query_predicted_persistent_bytes = (
        int(fixed_overhead_bytes) + hard_persistent_bytes_per_query + scorer_scratch_bytes_per_row
    )
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
    elif operational_persistent_bytes_per_query <= 0:
        query_batch_size = max_batch
    else:
        query_batch_size, _ = compute_chunk_size(
            item_bytes=operational_persistent_bytes_per_query,
            budget_bytes=stage_budget_bytes,
            fixed_overhead_bytes=int(fixed_overhead_bytes) + scorer_scratch_bytes_per_row,
            hard_limit_items=max_batch,
        )

    predicted_candidate_rows_per_batch = int(query_batch_size) * operational_candidate_rows_per_query
    predicted_pairs_per_batch = int(query_batch_size) * operational_pairs_per_query
    if int(query_batch_size) == parsed_query_count:
        if row_total_floor > 0 and orcid_floor_exceeds_top_k:
            predicted_candidate_rows_per_batch = row_total_floor
        else:
            predicted_candidate_rows_per_batch = max(predicted_candidate_rows_per_batch, row_total_floor)
        if pair_total_floor > 0 and orcid_floor_exceeds_top_k:
            predicted_pairs_per_batch = pair_total_floor
        else:
            predicted_pairs_per_batch = max(predicted_pairs_per_batch, pair_total_floor)
    predicted_persistent_bytes = (
        predicted_pairs_per_batch * (pending_retrieval_pair_bytes + int(pair_label_bytes))
        + predicted_candidate_rows_per_batch * (pending_retrieval_row_bytes + row_state_bytes_per_row)
        + int(fixed_overhead_bytes)
    )
    scorer_scratch_budget_bytes = max(1, stage_budget_bytes - predicted_persistent_bytes)
    scorer_chunk_rows = (
        min(
            predicted_candidate_rows_per_batch,
            max(1, scorer_scratch_budget_bytes // max(1, scorer_scratch_bytes_per_row)),
        )
        if predicted_candidate_rows_per_batch > 0
        else 0
    )

    # Match s2and.incremental_linking.linker_pairwise.compute_linker_pair_chunk_plan
    # without importing the incremental-linking package from this core utility.
    pair_memory_feature_count = max(
        1,
        parsed_pairwise_matrix_feature_count + parsed_aggregate_feature_count * 3 + 1,
    )
    pair_plan = compute_rust_batch_chunk_plan(
        num_features=pair_memory_feature_count,
        total_pairs=max(0, predicted_pairs_per_batch),
        total_rows=max(0, predicted_candidate_rows_per_batch),
        total_ram_bytes=snapshot.total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=lambda: (snapshot.total_ram_bytes, snapshot.total_ram_source),
        detect_total_fn=lambda: (snapshot.total_ram_bytes, snapshot.total_ram_source),
        current_rss_fn=lambda _total: (snapshot.current_rss_bytes, snapshot.current_rss_source),
        index_remap_bytes_per_pair=BORROWED_SIGNATURE_INDEX_REMAP_BYTES_PER_PAIR,
    )
    scorer_transient_bytes = scorer_chunk_rows * scorer_input_bytes_per_row + (
        scorer_chunk_rows * scorer_output_bytes_per_row if scorer_chunk_rows < predicted_candidate_rows_per_batch else 0
    )
    predicted_peak_delta_bytes = predicted_persistent_bytes + max(
        int(pair_plan.predicted_chunk_bytes),
        scorer_transient_bytes,
    )
    predicted_peak_rss_bytes = snapshot.current_rss_bytes + predicted_peak_delta_bytes
    return PromotedPhaseALimits(
        query_batch_size=int(query_batch_size),
        predicted_peak_delta_bytes=predicted_peak_delta_bytes,
        predicted_peak_rss_bytes=predicted_peak_rss_bytes,
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
