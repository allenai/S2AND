"""Benchmark batched Python-facing lookups against a name-count index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import psutil

from s2and.name_counts_index import _lookup_many_deduplicated
from s2and.text import canonical_name_count_keys, canonicalize_name_parts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5_000, help="0 means all signatures")
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--write-json", type=Path)
    return parser


def _rss() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _keys(path: Path, limit: int) -> tuple[list[str | None], ...]:
    payload = orjson.loads(path.read_bytes())
    selected = list(payload.values())
    if limit > 0:
        selected = selected[:limit]
    columns: tuple[list[str | None], ...] = ([], [], [], [])
    for signature in selected:
        author = signature["author_info"]
        keys = canonical_name_count_keys(
            canonicalize_name_parts(author.get("first"), author.get("middle"), author.get("last"))
        )
        columns[0].append(keys["first"])
        columns[1].append(keys["last"])
        columns[2].append(keys["first_last"])
        columns[3].append(keys["last_first_initial"])
    return columns


def _output_digest(columns: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    canonical_nan = struct.pack("<Q", 0x7FF8000000000000)
    for column in columns:
        for value in column:
            scalar = float(value)
            digest.update(canonical_nan if math.isnan(scalar) else struct.pack("<d", scalar))
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size < 1 or args.rounds < 1 or args.repetitions < 1 or args.limit < 0:
        raise ValueError("limit must be nonnegative and batch-size/rounds/repetitions must be positive")
    key_columns = _keys(args.signatures, args.limit)
    signature_count = len(key_columns[0])
    if signature_count == 0:
        raise ValueError("benchmark selected zero signatures")

    from s2and.runtime import load_s2and_rust_extension

    s2and_rust = load_s2and_rust_extension()

    rss_before_open = _rss()
    open_start = time.perf_counter()
    index = s2and_rust.NameCountsIndex.open(str(args.index.resolve()))
    open_seconds = time.perf_counter() - open_start
    rss_after_open = _rss()

    batch_seconds: list[float] = []
    round_seconds: list[float] = []
    digest = ""
    output_rows = 0
    peak_rss = rss_after_open
    for _round in range(args.rounds):
        round_start = time.perf_counter()
        round_digest = hashlib.sha256()
        for _repetition in range(args.repetitions):
            for start in range(0, signature_count, args.batch_size):
                end = min(start + args.batch_size, signature_count)
                batch_start = time.perf_counter()
                result = _lookup_many_deduplicated(index, *(column[start:end] for column in key_columns))
                batch_seconds.append(time.perf_counter() - batch_start)
                round_digest.update(_output_digest(result).encode("ascii"))
                output_rows += end - start
                peak_rss = max(peak_rss, _rss())
        round_seconds.append(time.perf_counter() - round_start)
        digest = round_digest.hexdigest()

    sorted_batches = sorted(batch_seconds)
    p95_index = min(len(sorted_batches) - 1, math.ceil(0.95 * len(sorted_batches)) - 1)
    total_lookups = output_rows * 4
    lookup_seconds = sum(batch_seconds)
    benchmark_seconds = sum(round_seconds)
    report: dict[str, Any] = {
        "artifact": {
            "index": str(args.index.resolve()),
            "normalization_version": index.normalization_version,
            "provenance_binding": index.name_counts_provenance_binding,
        },
        "input": {
            "signatures": str(args.signatures.resolve()),
            "signature_count": signature_count,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "repetitions": args.repetitions,
        },
        "metrics": {
            "open_seconds": open_seconds,
            "open_retained_rss_delta_bytes": rss_after_open - rss_before_open,
            "peak_rss_delta_bytes": peak_rss - rss_before_open,
            "lookup_seconds": lookup_seconds,
            "lookups": total_lookups,
            "lookups_per_second": total_lookups / lookup_seconds,
            "signatures_per_second": output_rows / lookup_seconds,
            "benchmark_seconds_including_digest_and_rss_sampling": benchmark_seconds,
            "benchmark_lookups_per_second": total_lookups / benchmark_seconds,
            "benchmark_signatures_per_second": output_rows / benchmark_seconds,
            "batch_seconds_p50": statistics.median(batch_seconds),
            "batch_seconds_p95": sorted_batches[p95_index],
            "output_digest": digest,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
