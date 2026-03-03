from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("s2and")

AUTODETECT_RAM_SAFETY_FACTOR = 0.8
DEFAULT_SAFETY_MARGIN_FRACTION = 0.10
# Approximate bytes/entry for Phase A's `signature_to_cluster_sum_count` accumulator.
# Calibrated on big-block telemetry (Feb 2026); tune via `scripts/rust_suite.py calibrate-phase-a`.
# Bundle 4 calibration (4 workload shapes): P95 ~192; 200 provides ~4% margin.
INCREMENTAL_ACCUMULATOR_ENTRY_BYTES = 200

# Rust batch featurization defaults.
RUST_BATCH_BASE_CHUNK_PAIRS = 10_000
RUST_BATCH_STAGE_BUDGET_FRACTION = 0.25
RUST_BATCH_ROW_OVERHEAD_BYTES = 128
# Bundle 4 calibration (4 workload shapes: 37, 37, 49, 49 bytes/row); P95 = 49; 52 provides ~6% margin.
RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES = 52
RUST_BATCH_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PHASE_A_FIXED_OVERHEAD_BYTES = 2 * (1 << 20)
PHASE_A_PAIR_BUFFER_ENTRY_BYTES = 80
PHASE_A_MAX_CHUNK_PAIRS_DEFAULT = 1_000_000
# Defensive upper bound on accumulator entries when no memory limits are provided.
# ~2 GB at 200 bytes/entry. Prevents unbounded growth if chunk_limits is None.
FALLBACK_ACCUMULATOR_MAX_ENTRIES = 10_000_000


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
class _MappingDataclass(Mapping[str, Any]):
    """Dataclass payload with read-only dict-like access for compatibility."""

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dataclass_fields__:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.__dataclass_fields__:
            return default
        return getattr(self, key)


@dataclass(frozen=True)
class IncrementalPhaseSplitLimits(_MappingDataclass):
    total_ram_bytes: int
    total_ram_source: str
    current_rss_bytes: int
    current_rss_source: str
    available_bytes: int
    effective_available_fraction: float
    safety_margin_bytes: int
    chunk_budget_bytes: int
    accumulator_budget_bytes: int
    bytes_per_pair: int
    derived_chunk_pairs: int
    chunk_pairs: int
    max_chunk_pairs: int
    accumulator_entry_bytes: int
    accumulator_warn: int
    accumulator_max: int


@dataclass(frozen=True)
class RustBatchChunkPlan(_MappingDataclass):
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
    predicted_stage_peak_bytes: int


@dataclass(frozen=True)
class PredictionAccuracySummary(_MappingDataclass):
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


def resolve_rust_batch_prediction_params() -> dict[str, int]:
    return {
        "base_chunk_pairs": RUST_BATCH_BASE_CHUNK_PAIRS,
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
        import psutil  # type: ignore
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
        import psutil  # type: ignore
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


def compute_incremental_phase_split_limits(
    num_features: int,
    *,
    selected_feature_count: int | None = None,
    nameless_feature_count: int = 0,
    total_ram_bytes: int | None = None,
    safety_margin_fraction: float = DEFAULT_SAFETY_MARGIN_FRACTION,
    chunk_budget_fraction: float = 0.60,
    accumulator_budget_fraction: float = 0.20,
    accumulator_entry_bytes: int = INCREMENTAL_ACCUMULATOR_ENTRY_BYTES,
    max_chunk_pairs: int | None = None,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    current_rss_fn: Callable[[int], tuple[int, str]] | None = None,
) -> IncrementalPhaseSplitLimits:
    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    chunk_budget_bytes = int(float(chunk_budget_fraction) * float(snapshot.available_bytes))
    accumulator_budget_bytes = int(float(accumulator_budget_fraction) * float(snapshot.available_bytes))

    if selected_feature_count is not None:
        effective_selected = max(1, min(num_features, selected_feature_count))
        effective_nameless = max(0, min(num_features, nameless_feature_count))
        bytes_per_pair = max(1, (effective_selected + effective_nameless) * 8 + 8 + 100)
    else:
        # Legacy: assume worst case (main + nameless at full width).
        bytes_per_pair = max(1, num_features * 8 * 2 + 8 + 100)
    derived_chunk_pairs = max(1, chunk_budget_bytes // bytes_per_pair)
    chunk_pairs = derived_chunk_pairs

    max_pairs = max_chunk_pairs
    if max_pairs is None:
        max_pairs = PHASE_A_MAX_CHUNK_PAIRS_DEFAULT
    if max_pairs < 0:
        raise ValueError(f"Invalid max_chunk_pairs={max_pairs}; expected >=0")
    if max_pairs > 0:
        chunk_pairs = min(chunk_pairs, max_pairs)

    accumulator_warn = max(1, accumulator_budget_bytes // accumulator_entry_bytes // 5)
    accumulator_max = max(accumulator_warn + 1, accumulator_budget_bytes // accumulator_entry_bytes)
    return IncrementalPhaseSplitLimits(
        total_ram_bytes=snapshot.total_ram_bytes,
        total_ram_source=snapshot.total_ram_source,
        current_rss_bytes=snapshot.current_rss_bytes,
        current_rss_source=snapshot.current_rss_source,
        available_bytes=snapshot.available_bytes,
        effective_available_fraction=snapshot.effective_available_fraction,
        safety_margin_bytes=snapshot.safety_margin_bytes,
        chunk_budget_bytes=chunk_budget_bytes,
        accumulator_budget_bytes=accumulator_budget_bytes,
        bytes_per_pair=bytes_per_pair,
        derived_chunk_pairs=derived_chunk_pairs,
        chunk_pairs=chunk_pairs,
        max_chunk_pairs=max_pairs,
        accumulator_entry_bytes=accumulator_entry_bytes,
        accumulator_warn=accumulator_warn,
        accumulator_max=accumulator_max,
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
    stage_budget_bytes = max(1, int(float(stage_budget_fraction) * float(snapshot.available_bytes)))
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
    derived_chunk_pairs = max(1, stage_budget_bytes // bytes_per_pair_row)
    bounded_base_chunk_pairs = max(1, base_chunk_pairs)
    bounded_total_pairs = max(1, total_pairs)
    bounded_total_rows = bounded_total_pairs
    if total_rows is not None:
        bounded_total_rows = max(1, total_rows)
    chunk_pairs = max(1, min(bounded_total_pairs, bounded_base_chunk_pairs, derived_chunk_pairs))

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
        base_chunk_pairs=bounded_base_chunk_pairs,
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
        # Backward-compatible alias; prefer predicted_stage_peak_delta_bytes.
        predicted_stage_peak_bytes=predicted_stage_peak_delta_bytes,
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
