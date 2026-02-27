from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from s2and.memory_calibration import (
    effective_rust_batch_persistent_row_overhead_bytes,
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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Rust batch persistent row overhead bytes from telemetry logs."
    )
    parser.add_argument("logs", nargs="+", help="Log file(s) containing rust pair_featurization telemetry lines.")
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Optional output path for a calibration JSON blob.",
    )
    args = parser.parse_args(argv)

    effective_values: list[float] = []
    rust_batch_records = 0

    for raw_path in args.logs:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        with _open_text_log(path) as fh:
            for record in iter_log_records(fh):
                if "Telemetry: pair_featurization_memory" not in record:
                    continue
                sample = extract_rust_batch_sample(record)
                if sample is None:
                    continue
                rust_batch_records += 1
                effective = effective_rust_batch_persistent_row_overhead_bytes(sample)
                if effective is None:
                    continue
                effective_values.append(float(effective))

    summary = _summarize(effective_values)
    if not summary:
        print(f"No calibration samples found (matched Rust batch records={rust_batch_records}, effective_samples=0).")
        return 2

    recommended_p95 = int(math.ceil(summary["p95"]))
    print(
        "Rust batch persistent row overhead calibration:\n"
        f"- matched_rust_batch_records={rust_batch_records}\n"
        f"- effective_samples={len(effective_values)}\n"
        f"- bytes_per_row: min={summary['min']:.1f} mean={summary['mean']:.1f} "
        f"p50={summary['p50']:.1f} p95={summary['p95']:.1f} max={summary['max']:.1f}\n"
        f"- recommended_rust_batch_persistent_row_overhead_bytes_p95={recommended_p95}"
    )

    if args.write_json is not None:
        output_path = Path(args.write_json)
        blob = {
            "schema_version": 1,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "matched_rust_batch_records": int(rust_batch_records),
            "effective_samples": int(len(effective_values)),
            "recommended_rust_batch_persistent_row_overhead_bytes_p95": int(recommended_p95),
            "summary": {k: float(v) for k, v in summary.items()},
            "inputs": [str(Path(p).resolve()) for p in args.logs],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
