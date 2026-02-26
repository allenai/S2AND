from __future__ import annotations

import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


def parse_kv_tokens(line: str) -> dict[str, str]:
    """Parses whitespace-delimited key=value tokens from a telemetry log line."""
    result: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            result[key] = value
    return result


_LOG_RECORD_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")


def iter_log_records(lines: Iterable[str]) -> Iterator[str]:
    """Yields reconstructed log records from raw physical lines.

    Some PowerShell redirections (notably `*> file.log`) can hard-wrap long lines
    to a terminal width, splitting a single Python `logging` record across
    multiple physical lines. Those continuation lines do not include the usual
    timestamp prefix, which breaks key=value token parsing.

    This function re-joins wrapped log output back into one record per timestamp
    prefix, concatenating continuation lines with spaces.
    """

    buffer: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if _LOG_RECORD_PREFIX_RE.match(line):
            if buffer is not None:
                yield buffer
            buffer = line
            continue

        if buffer is None:
            # Non-logging output (e.g., raw prints / warnings / progress bars).
            buffer = line.strip()
            continue

        continuation = line.strip()
        if not continuation:
            continue
        buffer = f"{buffer} {continuation}"

    if buffer is not None:
        yield buffer


@dataclass(frozen=True)
class PhaseASample:
    accumulator_entries_peak: int
    chunk_features_peak_bytes: int
    phase_a_pair_buffer_peak_bytes: int
    phase_a_fixed_overhead_bytes: int
    observed_peak_delta_bytes: int


@dataclass(frozen=True)
class RustBatchSample:
    total_rows: int
    predicted_features_matrix_bytes: int
    predicted_labels_bytes: int
    predicted_chunk_bytes: int
    predicted_fixed_overhead_bytes: int
    observed_peak_delta_bytes: int


def extract_phase_a_sample(line: str) -> PhaseASample | None:
    if "Telemetry: phase_split_phase_a" not in line:
        return None
    kv = parse_kv_tokens(line)
    try:
        chunk_features_peak_bytes = int(kv["chunk_features_peak_bytes"])
        phase_a_pair_buffer_peak_bytes = int(kv.get("phase_a_pair_buffer_peak_bytes", "0"))
        phase_a_fixed_overhead_bytes = int(kv["phase_a_fixed_overhead_bytes"])
        observed_peak_delta_bytes = int(kv["observed_peak_delta_bytes"])

        # Prefer the per-sample peak, since the aggregated Phase A telemetry can also
        # include a global peak that doesn't correspond to the predicted/observed deltas.
        accumulator_entries_peak: int | None = None
        if "accumulator_entries_peak_sample" in kv:
            accumulator_entries_peak = int(kv["accumulator_entries_peak_sample"])
        elif "accumulator_entries_peak" in kv:
            accumulator_entries_peak = int(kv["accumulator_entries_peak"])

        # Backward compatibility for older logs: infer the per-sample entry count from
        # the prediction formula when available.
        if accumulator_entries_peak is not None and "accumulator_entries_peak_sample" not in kv:
            predicted_raw = kv.get("predicted_peak_delta_bytes")
            entry_bytes_raw = kv.get("accumulator_entry_bytes")
            if predicted_raw is not None and entry_bytes_raw is not None:
                predicted_peak_delta_bytes = int(predicted_raw)
                accumulator_entry_bytes = int(entry_bytes_raw)
                residual = (
                    int(predicted_peak_delta_bytes) - int(chunk_features_peak_bytes) - int(phase_a_fixed_overhead_bytes)
                )
                if accumulator_entry_bytes > 0 and residual > 0 and residual % accumulator_entry_bytes == 0:
                    inferred_entries = residual // accumulator_entry_bytes
                    if inferred_entries > 0:
                        accumulator_entries_peak = int(inferred_entries)

        return PhaseASample(
            accumulator_entries_peak=int(accumulator_entries_peak or 0),
            chunk_features_peak_bytes=int(chunk_features_peak_bytes),
            phase_a_pair_buffer_peak_bytes=int(phase_a_pair_buffer_peak_bytes),
            phase_a_fixed_overhead_bytes=int(phase_a_fixed_overhead_bytes),
            observed_peak_delta_bytes=int(observed_peak_delta_bytes),
        )
    except (KeyError, ValueError):
        return None


def extract_rust_batch_sample(line: str) -> RustBatchSample | None:
    if "Telemetry: pair_featurization_memory" not in line:
        return None
    kv = parse_kv_tokens(line)
    if kv.get("stage") != "pair_featurization_rust_batch":
        return None
    try:
        return RustBatchSample(
            total_rows=int(kv["total_rows"]),
            predicted_features_matrix_bytes=int(kv["predicted_features_matrix_bytes"]),
            predicted_labels_bytes=int(kv["predicted_labels_bytes"]),
            predicted_chunk_bytes=int(kv["predicted_chunk_bytes"]),
            predicted_fixed_overhead_bytes=int(kv["predicted_fixed_overhead_bytes"]),
            observed_peak_delta_bytes=int(kv["observed_peak_delta_bytes"]),
        )
    except (KeyError, ValueError):
        return None


def effective_accumulator_entry_bytes(sample: PhaseASample) -> float | None:
    """Derives bytes-per-entry from the residual after subtracting modeled arrays."""
    if sample.accumulator_entries_peak <= 0:
        return None
    modeled_arrays_bytes = (
        int(sample.chunk_features_peak_bytes)
        + int(sample.phase_a_pair_buffer_peak_bytes)
        + int(sample.phase_a_fixed_overhead_bytes)
    )
    residual_bytes = int(sample.observed_peak_delta_bytes) - modeled_arrays_bytes
    if residual_bytes <= 0:
        return None
    return float(residual_bytes) / float(sample.accumulator_entries_peak)


def effective_rust_batch_persistent_row_overhead_bytes(sample: RustBatchSample) -> float | None:
    if sample.total_rows <= 0:
        return None
    modeled_without_persistent = (
        int(sample.predicted_features_matrix_bytes)
        + int(sample.predicted_labels_bytes)
        + int(sample.predicted_chunk_bytes)
        + int(sample.predicted_fixed_overhead_bytes)
    )
    residual_bytes = int(sample.observed_peak_delta_bytes) - modeled_without_persistent
    if residual_bytes <= 0:
        return None
    return float(residual_bytes) / float(sample.total_rows)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile, matching numpy's default convention closely."""
    if not values:
        raise ValueError("values must be non-empty")
    if p < 0 or p > 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return float(values_sorted[0])

    k = (len(values_sorted) - 1) * (float(p) / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(values_sorted[f])
    d = k - float(f)
    return float(values_sorted[f]) * (1.0 - d) + float(values_sorted[c]) * d
