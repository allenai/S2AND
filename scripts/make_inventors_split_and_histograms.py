"""Create inventors histogram artifacts from clusters.json.

This script computes:
1) histogram of signatures per block
2) histogram of signatures per cluster

Block IDs are derived from cluster IDs by stripping the trailing numeric suffix:
`<block_id>-<n>` -> `<block_id>`.

Usage:
  - Small verification run:
      uv run --with ijson scripts/make_inventors_split_and_histograms.py --limit-clusters 100000
  - Full run:
      uv run --with ijson scripts/make_inventors_split_and_histograms.py --full-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import ijson
import matplotlib.pyplot as plt
import numpy as np

CLUSTER_LOG_INTERVAL = 250_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-path", type=Path, default=Path("data/inventors/clusters.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/inventors"))
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--make-split",
        action="store_true",
        help="Also write train_keys.json and test_keys.json based on derived block IDs.",
    )
    parser.add_argument(
        "--limit-clusters",
        type=int,
        default=None,
        help="Optional cap on clusters processed for quick checks.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required for unbounded processing of full files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_size <= 0:
        raise ValueError("--train-size must be > 0")
    if args.test_size <= 0:
        raise ValueError("--test-size must be > 0")
    if args.limit_clusters is not None and args.limit_clusters <= 0:
        raise ValueError("--limit-clusters must be > 0 when set")
    if not args.full_run and args.limit_clusters is None:
        raise ValueError("Refusing unbounded run without explicit confirmation. Use --full-run or --limit-clusters.")


def cluster_to_block_id(cluster_id: str) -> str:
    head, sep, tail = cluster_id.rpartition("-")
    if sep and tail.isdigit():
        return head
    return cluster_id


def iter_cluster_id_and_size(clusters_path: Path):
    """Yield `(cluster_id, cluster_size)` tuples from clusters.json."""
    current_cluster_size: int | None = None
    in_signature_ids = False

    with clusters_path.open("rb") as infile:
        for prefix, event, value in ijson.parse(infile):
            if event == "map_key" and value == "signature_ids":
                in_signature_ids = True
                current_cluster_size = 0
                continue

            if not in_signature_ids:
                continue

            if event == "string" and prefix.endswith(".signature_ids.item"):
                assert current_cluster_size is not None
                current_cluster_size += 1
            elif event == "end_array" and prefix.endswith(".signature_ids"):
                assert current_cluster_size is not None
                cluster_id = prefix.removesuffix(".signature_ids")
                yield cluster_id, current_cluster_size
                in_signature_ids = False
                current_cluster_size = None


def quantile_from_freq(freq: Counter[int], q: float) -> int:
    if not freq:
        raise ValueError("Cannot compute quantile from empty frequency map")
    total = sum(freq.values())
    threshold = max(1, math.ceil(total * q))
    running = 0
    for value in sorted(freq):
        running += freq[value]
        if running >= threshold:
            return value
    return max(freq)


def summary_from_freq(freq: Counter[int]) -> dict[str, float]:
    if not freq:
        raise ValueError("Cannot summarize empty frequency map")

    num_groups = sum(freq.values())
    num_signatures = sum(size * count for size, count in freq.items())
    mean = num_signatures / num_groups
    p50 = quantile_from_freq(freq, 0.50)
    p90 = quantile_from_freq(freq, 0.90)
    p99 = quantile_from_freq(freq, 0.99)
    min_size = min(freq)
    max_size = max(freq)

    return {
        "num_groups": float(num_groups),
        "num_signatures": float(num_signatures),
        "min": float(min_size),
        "max": float(max_size),
        "mean": float(mean),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }


def make_split(
    block_signature_counts: Counter[str], train_size: int, test_size: int, seed: int
) -> tuple[list[str], list[str]]:
    all_blocks = sorted(block_signature_counts.keys())
    required = train_size + test_size
    if len(all_blocks) < required:
        raise ValueError(
            f"Not enough unique blocks for requested split. Have={len(all_blocks)}, need={required} "
            f"(train={train_size}, test={test_size})"
        )

    rng = random.Random(seed)
    rng.shuffle(all_blocks)
    train_keys = all_blocks[:train_size]
    test_keys = all_blocks[train_size : train_size + test_size]
    return train_keys, test_keys


def plot_histogram_from_freq(
    freq: Counter[int],
    title: str,
    output_path: Path,
    x_label: str = "Signatures per group",
    y_label: str = "Count of groups",
) -> None:
    if not freq:
        raise ValueError(f"Cannot plot empty histogram for {output_path}")

    x = np.array(list(freq.keys()), dtype=np.int64)
    weights = np.array(list(freq.values()), dtype=np.int64)

    max_x = int(x.max())
    if max_x <= 1:
        bins = np.array([0.5, 1.5], dtype=float)
    else:
        bins = np.geomspace(1.0, float(max_x) + 1.0, num=80)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=bins, weights=weights, color="#2C7FB8", alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print("Counting signatures per cluster and per block from clusters.json...")

    block_signature_counts: Counter[str] = Counter()
    cluster_size_freq: Counter[int] = Counter()
    cluster_count = 0
    signatures_in_clusters = 0

    for cluster_id, cluster_size in iter_cluster_id_and_size(args.clusters_path):
        cluster_size_freq[cluster_size] += 1
        signatures_in_clusters += cluster_size
        block_signature_counts[cluster_to_block_id(cluster_id)] += cluster_size
        cluster_count += 1

        if cluster_count % CLUSTER_LOG_INTERVAL == 0:
            elapsed = time.time() - start
            print(
                f"  clusters processed={cluster_count:,}, unique_blocks={len(block_signature_counts):,}, "
                f"cluster_signatures={signatures_in_clusters:,}, elapsed={elapsed:,.1f}s"
            )
        if args.limit_clusters is not None and cluster_count >= args.limit_clusters:
            break

    elapsed = time.time() - start
    print(
        f"Done clusters: processed={cluster_count:,}, unique_blocks={len(block_signature_counts):,}, "
        f"cluster_signatures={signatures_in_clusters:,}, elapsed={elapsed:,.1f}s"
    )

    train_path = output_dir / "train_keys.json"
    test_path = output_dir / "test_keys.json"
    block_hist_path = output_dir / "signatures_per_block_hist.png"
    cluster_hist_path = output_dir / "signatures_per_cluster_hist.png"
    metrics_path = output_dir / "inventors_split_hist_metrics.json"

    if args.make_split:
        train_keys, test_keys = make_split(
            block_signature_counts=block_signature_counts,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        with train_path.open("w", encoding="utf-8") as outfile:
            json.dump(train_keys, outfile)
        with test_path.open("w", encoding="utf-8") as outfile:
            json.dump(test_keys, outfile)

    block_size_freq = Counter(block_signature_counts.values())
    plot_histogram_from_freq(
        freq=block_size_freq,
        title="Inventors: Signatures per Block",
        output_path=block_hist_path,
    )
    plot_histogram_from_freq(
        freq=cluster_size_freq,
        title="Inventors: Signatures per Cluster",
        output_path=cluster_hist_path,
    )

    metrics = {
        "inputs": {
            "clusters_path": str(args.clusters_path),
        },
        "run_config": {
            "full_run": args.full_run,
            "make_split": args.make_split,
            "limit_clusters": args.limit_clusters,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "seed": args.seed,
            "block_derivation": "cluster_id minus trailing '-<digits>'",
        },
        "counts": {
            "processed_clusters": cluster_count,
            "unique_blocks": len(block_signature_counts),
            "signatures_in_clusters": signatures_in_clusters,
        },
        "summary": {
            "signatures_per_block": summary_from_freq(block_size_freq),
            "signatures_per_cluster": summary_from_freq(cluster_size_freq),
        },
        "outputs": {
            "signatures_per_block_hist": str(block_hist_path),
            "signatures_per_cluster_hist": str(cluster_hist_path),
        },
    }
    if args.make_split:
        metrics["outputs"]["train_keys"] = str(train_path)
        metrics["outputs"]["test_keys"] = str(test_path)

    with metrics_path.open("w", encoding="utf-8") as outfile:
        json.dump(metrics, outfile, indent=2)

    print("Wrote outputs:")
    if args.make_split:
        print(f"  {train_path}")
        print(f"  {test_path}")
    print(f"  {block_hist_path}")
    print(f"  {cluster_hist_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
