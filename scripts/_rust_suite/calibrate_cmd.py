from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from s2and.memory_calibration import (
    effective_accumulator_entry_bytes,
    effective_rust_batch_persistent_row_overhead_bytes,
    extract_phase_a_sample,
    extract_rust_batch_sample,
    iter_log_records,
    percentile,
)

_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_UTF8_BOM = b"\xef\xbb\xbf"


def _open_text_log(path: Path):
    with path.open("rb") as fh:
        head = fh.read(4)
    if head.startswith(_UTF16_BOMS):
        return path.open("r", encoding="utf-16", errors="replace")
    if head.startswith(_UTF8_BOM):
        return path.open("r", encoding="utf-8-sig", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "mean": float(sum(values) / float(len(values))),
        "p50": float(percentile(values, 50)),
        "p95": float(percentile(values, 95)),
        "max": float(max(values)),
    }


class _CalibrationProfile:
    def __init__(
        self,
        *,
        description: str,
        record_matcher: str,
        sample_extractor: Callable[[str], Any | None],
        effective_fn: Callable[[Any], float | None],
        matched_record_key: str,
        recommended_key: str,
        summary_label: str,
        print_title: str,
    ) -> None:
        self.description = description
        self.record_matcher = record_matcher
        self.sample_extractor = sample_extractor
        self.effective_fn = effective_fn
        self.matched_record_key = matched_record_key
        self.recommended_key = recommended_key
        self.summary_label = summary_label
        self.print_title = print_title


_CALIBRATION_PROFILES: dict[str, _CalibrationProfile] = {
    "phase_a": _CalibrationProfile(
        description="Calibrate Phase A accumulator entry bytes from telemetry logs.",
        record_matcher="Telemetry: phase_split_phase_a",
        sample_extractor=extract_phase_a_sample,
        effective_fn=effective_accumulator_entry_bytes,
        matched_record_key="matched_phase_a_records",
        recommended_key="recommended_accumulator_entry_bytes_p95",
        summary_label="bytes_per_entry",
        print_title="Phase A accumulator calibration",
    ),
    "rust_batch": _CalibrationProfile(
        description="Calibrate Rust batch persistent row overhead bytes from telemetry logs.",
        record_matcher="Telemetry: pair_featurization_memory",
        sample_extractor=extract_rust_batch_sample,
        effective_fn=effective_rust_batch_persistent_row_overhead_bytes,
        matched_record_key="matched_rust_batch_records",
        recommended_key="recommended_rust_batch_persistent_row_overhead_bytes_p95",
        summary_label="bytes_per_row",
        print_title="Rust batch persistent row overhead calibration",
    ),
}


def run_calibration(argv: list[str], *, profile_key: str) -> int:
    profile = _CALIBRATION_PROFILES[profile_key]

    parser = argparse.ArgumentParser(description=profile.description)
    parser.add_argument("logs", nargs="+", help="Log file(s) containing telemetry lines.")
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Optional output path for a calibration JSON blob.",
    )
    args = parser.parse_args(argv)

    effective_values: list[float] = []
    matched_records = 0

    for raw_path in args.logs:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        with _open_text_log(path) as fh:
            for record in iter_log_records(fh):
                if profile.record_matcher not in record:
                    continue
                matched_records += 1
                sample = profile.sample_extractor(record)
                if sample is None:
                    continue
                effective = profile.effective_fn(sample)
                if effective is None:
                    continue
                effective_values.append(float(effective))

    summary = _summarize(effective_values)
    if not summary:
        print("No calibration samples found " f"({profile.matched_record_key}={matched_records}, effective_samples=0).")
        return 2

    recommended_p95 = int(math.ceil(summary["p95"]))
    print(
        f"{profile.print_title}:\n"
        f"- {profile.matched_record_key}={matched_records}\n"
        f"- effective_samples={len(effective_values)}\n"
        f"- {profile.summary_label}: min={summary['min']:.1f} mean={summary['mean']:.1f} "
        f"p50={summary['p50']:.1f} p95={summary['p95']:.1f} max={summary['max']:.1f}\n"
        f"- {profile.recommended_key}={recommended_p95}"
    )

    if args.write_json is not None:
        output_path = Path(args.write_json)
        blob = {
            "schema_version": 1,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            profile.matched_record_key: int(matched_records),
            "effective_samples": int(len(effective_values)),
            profile.recommended_key: int(recommended_p95),
            "summary": {k: float(v) for k, v in summary.items()},
            "inputs": [str(Path(p).resolve()) for p in args.logs],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0
