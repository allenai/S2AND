from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("s2and")

AUTODETECT_RAM_SAFETY_FACTOR = 0.8
DEFAULT_SAFETY_MARGIN_FRACTION = 0.10
# Approximate bytes/entry for Phase A's `signature_to_cluster_sum_count` accumulator.
# Calibrated on big-block telemetry (Feb 2026); tune via `scripts/calibrate_phase_a_accumulator.py`.
INCREMENTAL_ACCUMULATOR_ENTRY_BYTES = 200

# Rust batch featurization defaults (override via S2AND_RUST_BATCH_* env vars).
RUST_BATCH_BASE_CHUNK_PAIRS = 10_000
RUST_BATCH_STAGE_BUDGET_FRACTION = 0.25
RUST_BATCH_ROW_OVERHEAD_BYTES = 128
RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES = 64
RUST_BATCH_FIXED_OVERHEAD_BYTES = 16 * (1 << 20)
PHASE_A_FIXED_OVERHEAD_BYTES = 2 * (1 << 20)
PHASE_A_PAIR_BUFFER_ENTRY_BYTES = 80
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


def _env_int(name: str, *, default: int, min_value: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    try:
        parsed = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw_value!r}; expected integer >= {min_value}.") from exc
    if parsed < int(min_value):
        raise ValueError(f"Invalid {name}={parsed}; expected >= {min_value}.")
    return int(parsed)


def resolve_rust_batch_prediction_params() -> dict[str, int]:
    return {
        "base_chunk_pairs": _env_int(
            "S2AND_RUST_BATCH_BASE_CHUNK_PAIRS",
            default=RUST_BATCH_BASE_CHUNK_PAIRS,
            min_value=1,
        ),
        "row_overhead_bytes": _env_int(
            "S2AND_RUST_BATCH_ROW_OVERHEAD_BYTES",
            default=RUST_BATCH_ROW_OVERHEAD_BYTES,
            min_value=0,
        ),
        "persistent_row_overhead_bytes": _env_int(
            "S2AND_RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES",
            default=RUST_BATCH_PERSISTENT_ROW_OVERHEAD_BYTES,
            min_value=0,
        ),
        "fixed_overhead_bytes": _env_int(
            "S2AND_RUST_BATCH_FIXED_OVERHEAD_BYTES",
            default=RUST_BATCH_FIXED_OVERHEAD_BYTES,
            min_value=0,
        ),
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


def detect_total_ram_bytes_best_effort() -> tuple[int | None, str]:
    try:
        import psutil  # type: ignore

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total, "psutil.virtual_memory"
    except Exception:
        pass

    meminfo_path = "/proc/meminfo"
    if os.path.exists(meminfo_path):
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
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss), "psutil_process_rss"
    except Exception:
        pass

    status_path = "/proc/self/status"
    if os.path.exists(status_path):
        try:
            with open(status_path, encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            return int(parts[1]) * 1024, "proc_status_vmrss"
        except Exception:
            pass

    logger.warning(
        "Unable to determine process RSS (psutil unavailable and /proc/self/status not found); "
        "falling back to 50%% of total_ram_bytes=%d. Memory budgeting may be inaccurate. "
        "Install psutil for reliable RSS measurement.",
        total_ram_bytes,
    )
    return int(0.5 * total_ram_bytes), "fallback_half_total"


def resolve_total_ram_bytes(
    total_ram_bytes: int | None = None,
    *,
    detect_cgroup_fn: Callable[[], tuple[int | None, str]] | None = None,
    detect_total_fn: Callable[[], tuple[int | None, str]] | None = None,
    autodetect_safety_factor: float = AUTODETECT_RAM_SAFETY_FACTOR,
) -> tuple[int, str]:
    if total_ram_bytes is not None:
        return validate_positive_total_ram_bytes(total_ram_bytes, source="arg"), "arg"

    detect_cgroup = detect_cgroup_fn or detect_cgroup_total_ram_bytes_best_effort
    detect_total = detect_total_fn or detect_total_ram_bytes_best_effort

    cgroup_limit_bytes, cgroup_source = detect_cgroup()
    if cgroup_limit_bytes is not None:
        capped_cgroup = max(1, int(float(cgroup_limit_bytes) * autodetect_safety_factor))
        return capped_cgroup, f"{cgroup_source}_80pct"

    detected, source = detect_total()
    if detected is None:
        raise RuntimeError("Unable to determine total RAM for chunked incremental; pass total_ram_bytes explicitly.")
    capped_detected = max(1, int(float(detected) * autodetect_safety_factor))
    return capped_detected, f"{source}_80pct"


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
    available_bytes = max(1, resolved_total_ram_bytes - current_rss_bytes - safety_margin_bytes)
    return MemorySnapshot(
        total_ram_bytes=resolved_total_ram_bytes,
        total_ram_source=total_ram_source,
        current_rss_bytes=current_rss_bytes,
        current_rss_source=current_rss_source,
        safety_margin_bytes=safety_margin_bytes,
        available_bytes=available_bytes,
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
) -> dict[str, int | str]:
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
        effective_selected = max(1, min(int(num_features), int(selected_feature_count)))
        effective_nameless = max(0, min(int(num_features), int(nameless_feature_count)))
        bytes_per_pair = max(1, (effective_selected + effective_nameless) * 8 + 8 + 100)
    else:
        # Legacy: assume worst case (main + nameless at full width).
        bytes_per_pair = max(1, int(num_features) * 8 * 2 + 8 + 100)
    derived_chunk_pairs = max(1, chunk_budget_bytes // bytes_per_pair)
    chunk_pairs = min(20_000_000, derived_chunk_pairs)

    max_pairs = max_chunk_pairs
    if max_pairs is None:
        env_raw = os.environ.get("S2AND_PHASE_A_MAX_CHUNK_PAIRS")
        if env_raw is not None:
            try:
                max_pairs = int(env_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid S2AND_PHASE_A_MAX_CHUNK_PAIRS={env_raw!r}; expected an integer (0 disables the cap)."
                ) from exc
        else:
            max_pairs = 500_000
    if max_pairs < 0:
        raise ValueError(f"Invalid max_chunk_pairs={max_pairs}; expected >=0")
    if max_pairs > 0:
        chunk_pairs = min(chunk_pairs, int(max_pairs))

    accumulator_warn = max(1, accumulator_budget_bytes // accumulator_entry_bytes // 5)
    accumulator_max = max(accumulator_warn + 1, accumulator_budget_bytes // accumulator_entry_bytes)
    return {
        "total_ram_bytes": snapshot.total_ram_bytes,
        "total_ram_source": snapshot.total_ram_source,
        "current_rss_bytes": snapshot.current_rss_bytes,
        "current_rss_source": snapshot.current_rss_source,
        "available_bytes": snapshot.available_bytes,
        "safety_margin_bytes": snapshot.safety_margin_bytes,
        "chunk_budget_bytes": chunk_budget_bytes,
        "accumulator_budget_bytes": accumulator_budget_bytes,
        "bytes_per_pair": bytes_per_pair,
        "derived_chunk_pairs": derived_chunk_pairs,
        "chunk_pairs": chunk_pairs,
        "max_chunk_pairs": int(max_pairs),
        "accumulator_entry_bytes": int(accumulator_entry_bytes),
        "accumulator_warn": accumulator_warn,
        "accumulator_max": accumulator_max,
    }


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
) -> dict[str, int | str | float]:
    resolved = resolve_rust_batch_prediction_params()
    if base_chunk_pairs is None:
        base_chunk_pairs = int(resolved["base_chunk_pairs"])
    if row_overhead_bytes is None:
        row_overhead_bytes = int(resolved["row_overhead_bytes"])
    if persistent_row_overhead_bytes is None:
        persistent_row_overhead_bytes = int(resolved["persistent_row_overhead_bytes"])
    if fixed_overhead_bytes is None:
        fixed_overhead_bytes = int(resolved["fixed_overhead_bytes"])

    snapshot = memory_snapshot_for_stage(
        total_ram_bytes=total_ram_bytes,
        safety_margin_fraction=safety_margin_fraction,
        detect_cgroup_fn=detect_cgroup_fn,
        detect_total_fn=detect_total_fn,
        current_rss_fn=current_rss_fn,
    )
    stage_budget_bytes = max(1, int(float(stage_budget_fraction) * float(snapshot.available_bytes)))
    full_feature_count = max(1, int(num_features))
    selected_feature_count_bounded = full_feature_count
    if selected_feature_count is not None:
        selected_feature_count_bounded = max(1, min(full_feature_count, int(selected_feature_count)))
    nameless_feature_count_bounded = max(0, min(full_feature_count, int(nameless_feature_count)))
    # Use selected + nameless for chunk sizing (upper bound on columns Rust produces).
    # When selected_feature_count is None, selected_feature_count_bounded == full_feature_count,
    # so behavior is unchanged for callers that don't specify feature counts.
    chunk_feature_count = max(1, selected_feature_count_bounded + nameless_feature_count_bounded)
    bytes_per_pair_row = max(1, chunk_feature_count * 8 + int(row_overhead_bytes))
    derived_chunk_pairs = max(1, stage_budget_bytes // bytes_per_pair_row)
    bounded_base_chunk_pairs = max(1, int(base_chunk_pairs))
    bounded_total_pairs = max(1, int(total_pairs))
    bounded_total_rows = bounded_total_pairs
    if total_rows is not None:
        bounded_total_rows = max(1, int(total_rows))
    chunk_pairs = max(1, min(bounded_total_pairs, bounded_base_chunk_pairs, derived_chunk_pairs))

    predicted_chunk_bytes = int(chunk_pairs) * int(bytes_per_pair_row)
    predicted_selected_features_bytes = int(bounded_total_rows) * int(selected_feature_count_bounded * 8)
    predicted_nameless_features_bytes = int(bounded_total_rows) * int(nameless_feature_count_bounded * 8)
    predicted_features_matrix_bytes = int(predicted_selected_features_bytes) + int(predicted_nameless_features_bytes)
    # many_pairs_featurize now allocates only selected columns (plus optional nameless columns)
    # and does not materialize full-matrix + slice copies anymore.
    predicted_slice_peak_bytes = 0
    predicted_labels_bytes = int(bounded_total_rows) * 8
    predicted_persistent_row_overhead_bytes = int(bounded_total_rows) * max(0, int(persistent_row_overhead_bytes))
    predicted_fixed_overhead_bytes = max(0, int(fixed_overhead_bytes))
    predicted_stage_peak_delta_bytes = (
        predicted_features_matrix_bytes
        + predicted_labels_bytes
        + predicted_chunk_bytes
        + predicted_persistent_row_overhead_bytes
        + predicted_fixed_overhead_bytes
    )
    predicted_stage_peak_rss_bytes = int(snapshot.current_rss_bytes) + int(predicted_stage_peak_delta_bytes)
    return {
        "total_ram_bytes": snapshot.total_ram_bytes,
        "total_ram_source": snapshot.total_ram_source,
        "current_rss_bytes": snapshot.current_rss_bytes,
        "current_rss_source": snapshot.current_rss_source,
        "available_bytes": snapshot.available_bytes,
        "safety_margin_bytes": snapshot.safety_margin_bytes,
        "stage_budget_fraction": float(stage_budget_fraction),
        "stage_budget_bytes": int(stage_budget_bytes),
        "base_chunk_pairs": int(bounded_base_chunk_pairs),
        "row_overhead_bytes": int(max(0, int(row_overhead_bytes))),
        "persistent_row_overhead_bytes": int(max(0, int(persistent_row_overhead_bytes))),
        "fixed_overhead_bytes": int(predicted_fixed_overhead_bytes),
        "bytes_per_pair_row": int(bytes_per_pair_row),
        "derived_chunk_pairs": int(derived_chunk_pairs),
        "chunk_pairs": int(chunk_pairs),
        "total_rows": int(bounded_total_rows),
        "full_feature_count": int(full_feature_count),
        "selected_feature_count": int(selected_feature_count_bounded),
        "nameless_feature_count": int(nameless_feature_count_bounded),
        "predicted_chunk_bytes": int(predicted_chunk_bytes),
        "predicted_features_matrix_bytes": int(predicted_features_matrix_bytes),
        "predicted_labels_bytes": int(predicted_labels_bytes),
        "predicted_persistent_row_overhead_bytes": int(predicted_persistent_row_overhead_bytes),
        "predicted_fixed_overhead_bytes": int(predicted_fixed_overhead_bytes),
        "predicted_selected_features_bytes": int(predicted_selected_features_bytes),
        "predicted_nameless_features_bytes": int(predicted_nameless_features_bytes),
        "predicted_slice_peak_bytes": int(predicted_slice_peak_bytes),
        "predicted_stage_peak_delta_bytes": int(predicted_stage_peak_delta_bytes),
        "predicted_stage_peak_rss_bytes": int(predicted_stage_peak_rss_bytes),
        # Backward-compatible alias; prefer predicted_stage_peak_delta_bytes.
        "predicted_stage_peak_bytes": int(predicted_stage_peak_delta_bytes),
    }


def summarize_prediction_accuracy(
    *,
    stage_name: str,
    predicted_peak_delta_bytes: int | None = None,
    predicted_bytes: int | None = None,
    rss_before_bytes: int,
    rss_peak_bytes: int,
    rss_after_bytes: int,
) -> dict[str, str | int | float | bool]:
    if predicted_peak_delta_bytes is None and predicted_bytes is None:
        raise ValueError("Either predicted_peak_delta_bytes or predicted_bytes must be provided.")
    if predicted_peak_delta_bytes is None:
        predicted_peak_delta_bytes = predicted_bytes

    bounded_predicted_delta = max(1, int(predicted_peak_delta_bytes))
    bounded_before = max(0, int(rss_before_bytes))
    bounded_peak = max(bounded_before, int(rss_peak_bytes))
    bounded_after = max(0, int(rss_after_bytes))
    observed_peak_delta_bytes = max(0, bounded_peak - bounded_before)
    observed_end_delta_bytes = max(0, bounded_after - bounded_before)
    predicted_peak_rss_bytes = bounded_before + bounded_predicted_delta
    prediction_error_ratio = float(observed_peak_delta_bytes) / float(bounded_predicted_delta)
    return {
        "stage_name": stage_name,
        "prediction_contract_version": "delta_v1",
        "predicted_peak_delta_bytes": bounded_predicted_delta,
        "predicted_peak_rss_bytes": predicted_peak_rss_bytes,
        # Backward-compatible alias; prefer predicted_peak_delta_bytes.
        "predicted_bytes": bounded_predicted_delta,
        "rss_before_bytes": bounded_before,
        "rss_peak_bytes": bounded_peak,
        "rss_after_bytes": bounded_after,
        "observed_peak_delta_bytes": observed_peak_delta_bytes,
        "observed_end_delta_bytes": observed_end_delta_bytes,
        "prediction_error_ratio": prediction_error_ratio,
        "underpredicted": bool(observed_peak_delta_bytes > bounded_predicted_delta),
    }
