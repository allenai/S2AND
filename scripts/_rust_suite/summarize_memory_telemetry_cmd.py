from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from _rust_suite.calibrate_cmd import _open_text_log

from s2and.memory_calibration import iter_log_records, parse_kv_tokens, percentile


@dataclass(frozen=True)
class _RatioSummary:
    matched_records: int
    samples: int
    underpredicted_count: int
    underpredicted_fraction: float
    ratio_summary: dict[str, float]


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


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _collect_ratio_summary(
    *,
    log_paths: list[Path],
    record_matcher: str,
    stage_filter: str | None = None,
) -> _RatioSummary:
    matched_records = 0
    ratio_values: list[float] = []
    underpredicted_count = 0

    for path in log_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
        with _open_text_log(path) as fh:
            for record in iter_log_records(fh):
                if record_matcher not in record:
                    continue
                kv = parse_kv_tokens(record)
                if stage_filter is not None and kv.get("stage") != stage_filter:
                    continue
                matched_records += 1

                ratio_raw = kv.get("prediction_error_ratio")
                if ratio_raw is None:
                    continue
                try:
                    ratio_values.append(float(ratio_raw))
                except ValueError:
                    continue

                if _truthy(kv.get("underpredicted", "false")):
                    underpredicted_count += 1

    samples = len(ratio_values)
    underpredicted_fraction = float(underpredicted_count) / float(samples) if samples else 0.0
    return _RatioSummary(
        matched_records=matched_records,
        samples=samples,
        underpredicted_count=underpredicted_count,
        underpredicted_fraction=underpredicted_fraction,
        ratio_summary=_summarize(ratio_values),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Summarize memory prediction error ratios from telemetry logs.")
    parser.add_argument("logs", nargs="+", help="Log file(s) containing telemetry lines.")
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Optional output path for a JSON summary blob.",
    )
    args = parser.parse_args(argv)

    log_paths = [Path(raw) for raw in args.logs]
    phase_a = _collect_ratio_summary(
        log_paths=log_paths,
        record_matcher="Telemetry: phase_split_phase_a",
    )
    rust_batch = _collect_ratio_summary(
        log_paths=log_paths,
        record_matcher="Telemetry: pair_featurization_memory",
        stage_filter="pair_featurization_rust_batch",
    )

    blob = {
        "schema_version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "inputs": [str(p.resolve()) for p in log_paths],
        "stages": {
            "phase_a_seed_distances": {
                "record_matcher": "Telemetry: phase_split_phase_a",
                "matched_records": int(phase_a.matched_records),
                "samples": int(phase_a.samples),
                "underpredicted_count": int(phase_a.underpredicted_count),
                "underpredicted_fraction": float(phase_a.underpredicted_fraction),
                "ratio_summary": {k: float(v) for k, v in phase_a.ratio_summary.items()},
            },
            "pair_featurization_rust_batch": {
                "record_matcher": "Telemetry: pair_featurization_memory",
                "stage_filter": "pair_featurization_rust_batch",
                "matched_records": int(rust_batch.matched_records),
                "samples": int(rust_batch.samples),
                "underpredicted_count": int(rust_batch.underpredicted_count),
                "underpredicted_fraction": float(rust_batch.underpredicted_fraction),
                "ratio_summary": {k: float(v) for k, v in rust_batch.ratio_summary.items()},
            },
        },
    }

    if args.write_json is not None:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if phase_a.samples == 0 and rust_batch.samples == 0:
        print("No telemetry samples found (phase_a_samples=0, rust_batch_samples=0).")
        return 2

    def _format_stage(name: str, summary: _RatioSummary) -> str:
        ratio = summary.ratio_summary
        if not ratio:
            ratio_text = "ratio: (no samples)"
        else:
            ratio_text = (
                "ratio: "
                f"min={ratio['min']:.3f} mean={ratio['mean']:.3f} "
                f"p50={ratio['p50']:.3f} p95={ratio['p95']:.3f} max={ratio['max']:.3f}"
            )
        return (
            f"- {name}: matched_records={summary.matched_records} samples={summary.samples} "
            f"underpredicted={summary.underpredicted_count} "
            f"underpredicted_fraction={summary.underpredicted_fraction:.3f} {ratio_text}"
        )

    print(
        "Memory telemetry summary:\n"
        f"{_format_stage('phase_a_seed_distances', phase_a)}\n"
        f"{_format_stage('pair_featurization_rust_batch', rust_batch)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
