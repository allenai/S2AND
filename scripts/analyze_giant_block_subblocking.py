"""Sweep subblocking thresholds on a giant Arrow block and write plots/tables.

This script evaluates candidate `maximum_size` thresholds using repeated ORCID groups as
an external preservation signal. It runs the native Arrow subblocker that
`Clusterer.predict_from_arrow(..., batching_threshold=...)` uses, with the default
`GraphSubblockingConfig`. It disables ORCID subblocking explicitly so the metric
reflects only name-based and fallback partitioning, not the ORCID co-location override.

It also reports how often the fallback was actually invoked while building subblocks,
plus the size of the downstream retrieval target:

`single-letter-first-name signatures UNION signatures that end up in a fallback-made subblock`.
The `*specter*` metric names are historical: fallback subblocks keep the `|specter=` key.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from s2and.arrow_inputs import ArrowDataset
from s2and.subblocking import (
    GraphSubblockingConfig,
    _make_subblocks_with_telemetry_arrow_rust,
    normalize_orcid_for_subblocking,
    signature_name_parts_for_subblocking,
    subblocking_name_parts,
)

Signature = SimpleNamespace


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrow-dataset",
        type=Path,
        required=True,
        help="Arrow dataset directory with manifest.json and SPECTER embeddings.",
    )
    parser.add_argument(
        "--block",
        default=None,
        help="Author block to analyze; the default analyzes every signature in the dataset.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        required=True,
        help="Subblocking thresholds to evaluate, for example: --thresholds 2000 4000 6000 7500 10000",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON/plot artifacts.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Native worker threads.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Graph subblocking seed; production uses the model's random_state (default: 42).",
    )
    return parser.parse_args(argv)


def _normalize_orcid(orcid: str | None) -> str | None:
    """Normalize ORCID using the subblocking/Rust-Arrow canonical key."""

    return normalize_orcid_for_subblocking(orcid)


def _build_repeated_orcid_groups(signatures: dict[str, Signature]) -> dict[str, list[str]]:
    """Group signatures by normalized ORCID and keep only repeated groups."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for signature_id, signature in signatures.items():
        normalized_orcid = _normalize_orcid(signature.author_info_orcid)
        if normalized_orcid is not None:
            grouped[normalized_orcid].append(str(signature_id))
    return {
        normalized_orcid: sorted(signature_ids)
        for normalized_orcid, signature_ids in grouped.items()
        if len(signature_ids) > 1
    }


def _single_letter_first_name_ids(
    signatures: dict[str, Signature], signature_ids: list[str], maximum_size: int
) -> set[str]:
    """Return signature IDs whose subblock-key first name is empty or one character.

    The key of a compound first name depends on the threshold, so call this once per threshold.
    """
    key_names = subblocking_name_parts(
        [signature_name_parts_for_subblocking(signatures[signature_id]) for signature_id in signature_ids],
        maximum_size,
    )
    return {
        signature_id for signature_id, (first, _middle) in zip(signature_ids, key_names, strict=True) if len(first) <= 1
    }


def _signature_to_subblock(subblocks: dict[str, list[str]]) -> dict[str, str]:
    """Invert a subblock mapping."""
    signature_to_subblock: dict[str, str] = {}
    for subblock_key, signature_ids in subblocks.items():
        for signature_id in signature_ids:
            signature_id = str(signature_id)
            if signature_id in signature_to_subblock:
                raise ValueError(f"Signature {signature_id} appeared in multiple subblocks")
            signature_to_subblock[signature_id] = subblock_key
    return signature_to_subblock


def _specter_signature_ids(subblocks: dict[str, list[str]]) -> set[str]:
    """Return signature IDs that end up in a subblock whose lineage includes SPECTER."""
    output: set[str] = set()
    for subblock_key, signature_ids in subblocks.items():
        if "|specter=" in subblock_key:
            output.update(str(signature_id) for signature_id in signature_ids)
    return output


def _orcid_preservation_metrics(
    repeated_orcid_groups: dict[str, list[str]],
    signature_to_subblock: dict[str, str],
) -> dict[str, float | int]:
    """Compute ORCID-group preservation metrics for one partition."""
    preserved_group_count = 0
    preserved_signature_count = 0
    total_signature_count = 0

    for signature_ids in repeated_orcid_groups.values():
        subblock_ids = {signature_to_subblock[signature_id] for signature_id in signature_ids}
        group_size = len(signature_ids)
        total_signature_count += group_size
        if len(subblock_ids) == 1:
            preserved_group_count += 1
            preserved_signature_count += group_size

    total_group_count = len(repeated_orcid_groups)
    if total_group_count == 0:
        return {
            "orcid_group_count": 0,
            "orcid_preserved_group_count": 0,
            "orcid_group_preservation_rate": 0.0,
            "orcid_repeated_signature_count": 0,
            "orcid_preserved_signature_count": 0,
            "orcid_signature_weighted_preservation_rate": 0.0,
        }

    return {
        "orcid_group_count": int(total_group_count),
        "orcid_preserved_group_count": int(preserved_group_count),
        "orcid_group_preservation_rate": float(preserved_group_count / total_group_count),
        "orcid_repeated_signature_count": int(total_signature_count),
        "orcid_preserved_signature_count": int(preserved_signature_count),
        "orcid_signature_weighted_preservation_rate": float(preserved_signature_count / total_signature_count),
    }


def _subblock_size_metrics(subblocks: dict[str, list[str]]) -> dict[str, float | int]:
    """Summarize final subblock sizes."""
    sizes = np.array(sorted((len(signature_ids) for signature_ids in subblocks.values()), reverse=True), dtype=np.int64)
    if len(sizes) == 0:
        raise ValueError("Expected at least one subblock")
    return {
        "subblock_count": int(len(sizes)),
        "max_subblock_size": int(sizes[0]),
        "median_subblock_size": float(np.median(sizes)),
        "p95_subblock_size": float(np.percentile(sizes, 95)),
        "sum_within_subblock_pairs": int(sum(int(size) * (int(size) - 1) // 2 for size in sizes)),
    }


def _render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a compact markdown summary table without external tabulate dependencies."""
    ordered_columns = [
        "threshold",
        "orcid_group_preservation_rate",
        "specter_invocation_count",
        "specter_input_signature_count",
        "final_specter_labeled_signature_count",
        "retrieval_target_signature_count",
        "retrieval_target_signature_fraction",
        "subblock_count",
        "max_subblock_size",
        "median_subblock_size",
    ]
    header = "| " + " | ".join(ordered_columns) + " |"
    separator = "| " + " | ".join("---" for _ in ordered_columns) + " |"
    body = []
    for row in rows:
        formatted = []
        for column in ordered_columns:
            value = row[column]
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        body.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *body, ""])


def _write_overview_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Write a four-panel overview plot for threshold selection."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    thresholds = summary_df["threshold"].to_numpy()

    ax = axes[0, 0]
    ax.plot(thresholds, summary_df["orcid_group_preservation_rate"], marker="o", label="Group preservation")
    ax.plot(
        thresholds,
        summary_df["orcid_signature_weighted_preservation_rate"],
        marker="s",
        label="Signature-weighted preservation",
    )
    ax.set_title("ORCID Preservation")
    ax.set_xlabel("Subblock threshold")
    ax.set_ylabel("Fraction preserved")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(thresholds, summary_df["specter_invocation_count"], marker="o", label="SPECTER invocations")
    ax.plot(
        thresholds,
        summary_df["specter_fallback_candidate_block_count"],
        marker="s",
        label="Fallback candidate blocks",
    )
    ax.set_title("SPECTER Fallback Frequency")
    ax.set_xlabel("Subblock threshold")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(thresholds, summary_df["specter_input_signature_count"], marker="o", label="SPECTER input signatures")
    ax.plot(
        thresholds,
        summary_df["final_specter_labeled_signature_count"],
        marker="s",
        label="Final SPECTER-subblock signatures",
    )
    ax.plot(
        thresholds,
        summary_df["retrieval_target_signature_count"],
        marker="^",
        label="Retrieval target signatures",
    )
    ax.set_title("SPECTER And Retrieval Target Size")
    ax.set_xlabel("Subblock threshold")
    ax.set_ylabel("Signature count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(thresholds, summary_df["subblock_count"], marker="o", label="Subblock count")
    ax.plot(thresholds, summary_df["max_subblock_size"], marker="s", label="Max subblock size")
    ax.plot(thresholds, summary_df["median_subblock_size"], marker="^", label="Median subblock size")
    ax.set_title("Subblock Shape")
    ax.set_xlabel("Subblock threshold")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _optional_str(value: Any) -> str | None:
    """Return None for missing Arrow strings and the string otherwise."""
    return None if value is None else str(value)


def load_block_signatures(arrow_dataset: ArrowDataset, block: str | None) -> dict[str, Signature]:
    """Read the name, paper, and ORCID fields of one block's signatures from Arrow.

    Normalized names stay empty, so ``signature_name_parts_for_subblocking`` canonicalizes the raw
    names exactly as native subblocking does.
    """
    columns = ["signature_id", "paper_id", "author_first", "author_middle", "author_last", "author_block"]
    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            table = ipc.open_file(source).read_all()
    missing = sorted(set(columns) - set(table.column_names))
    if missing:
        raise ValueError(f"Arrow signatures table is missing columns: {missing}")
    has_orcid = "author_orcid" in table.column_names
    data = table.select(columns + (["author_orcid"] if has_orcid else [])).to_pydict()
    signatures: dict[str, Signature] = {}
    for index, signature_id in enumerate(data["signature_id"]):
        if block is not None and data["author_block"][index] != block:
            continue
        signatures[str(signature_id)] = Signature(
            paper_id=_optional_str(data["paper_id"][index]),
            author_info_first=_optional_str(data["author_first"][index]),
            author_info_first_normalized=None,
            author_info_middle=_optional_str(data["author_middle"][index]),
            author_info_middle_normalized=None,
            author_info_last=_optional_str(data["author_last"][index]),
            author_info_orcid=_optional_str(data["author_orcid"][index]) if has_orcid else None,
        )
    if not signatures:
        raise ValueError(f"No signatures found for block {block!r}")
    return signatures


def _threshold_row(
    arrow_dataset: ArrowDataset,
    signatures: dict[str, Signature],
    all_signature_ids: list[str],
    repeated_orcid_groups: dict[str, list[str]],
    *,
    threshold: int,
    random_seed: int,
) -> dict[str, Any]:
    """Subblock one block natively at one threshold and summarize the partition."""
    threshold_start = time.perf_counter()
    subblocks, telemetry, _no_middle_dead_ends = _make_subblocks_with_telemetry_arrow_rust(
        arrow_dataset,
        all_signature_ids,
        maximum_size=int(threshold),
        graph_subblocking_config=GraphSubblockingConfig(),
        graph_subblocking_random_seed=random_seed,
        use_orcid_subblocking=False,
    )
    signature_to_subblock = _signature_to_subblock(subblocks)
    if set(signature_to_subblock) != set(all_signature_ids):
        raise ValueError("Native subblocking must return every input signature exactly once")
    single_letter_ids = _single_letter_first_name_ids(signatures, all_signature_ids, int(threshold))
    preservation_metrics = _orcid_preservation_metrics(repeated_orcid_groups, signature_to_subblock)
    subblock_metrics = _subblock_size_metrics(subblocks)
    final_specter_signature_ids = _specter_signature_ids(subblocks)
    retrieval_target_signature_ids = single_letter_ids.union(final_specter_signature_ids)
    row = {
        "threshold": int(threshold),
        **telemetry,
        **preservation_metrics,
        **subblock_metrics,
        "single_letter_first_name_signature_count": int(len(single_letter_ids)),
        "final_specter_signature_fraction": float(len(final_specter_signature_ids) / len(all_signature_ids)),
        "retrieval_target_signature_count": int(len(retrieval_target_signature_ids)),
        "retrieval_target_signature_fraction": float(len(retrieval_target_signature_ids) / len(all_signature_ids)),
        "retrieval_target_specter_only_signature_count": int(len(final_specter_signature_ids - single_letter_ids)),
        "elapsed_seconds": round(time.perf_counter() - threshold_start, 3),
    }
    print(
        f"threshold={threshold} preserved={row['orcid_group_preservation_rate']:.4f} "
        f"specter_invocations={row['specter_invocation_count']} "
        f"target_sigs={row['retrieval_target_signature_count']} "
        f"elapsed_s={row['elapsed_seconds']:.3f}"
    )
    return row


def main(argv: Sequence[str] | None = None) -> None:
    """Run the threshold sweep and write artifacts."""
    args = parse_args(argv)
    thresholds = sorted(set(int(threshold) for threshold in args.thresholds))
    if not thresholds:
        raise ValueError("Need at least one threshold")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Native and OpenMP pools read these variables when they first start.
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(args.n_jobs)))
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.n_jobs)))
    start = time.perf_counter()
    with ArrowDataset.open(args.arrow_dataset, require_specter=True) as arrow_dataset:
        signatures = load_block_signatures(arrow_dataset, args.block)
        all_signature_ids = sorted(signatures)
        repeated_orcid_groups = _build_repeated_orcid_groups(signatures)
        rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            rows.append(
                _threshold_row(
                    arrow_dataset,
                    signatures,
                    all_signature_ids,
                    repeated_orcid_groups,
                    threshold=threshold,
                    random_seed=int(args.random_seed),
                )
            )

    summary_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    summary_csv_path = output_dir / "threshold_summary.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "threshold_summary.md"
    plot_path = output_dir / "threshold_overview.png"

    summary_df.to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(_render_markdown_table(summary_df.to_dict(orient="records")), encoding="utf-8")
    _write_overview_plot(summary_df, plot_path)

    metadata = {
        "arrow_dataset": str(args.arrow_dataset),
        "block": args.block,
        "signature_count": int(len(all_signature_ids)),
        "paper_count": int(len({signature.paper_id for signature in signatures.values()})),
        "orcid_signature_count": int(sum(1 for signature in signatures.values() if signature.author_info_orcid)),
        "normalized_orcid_count": int(
            len(
                {
                    normalized_orcid
                    for normalized_orcid in (
                        _normalize_orcid(signature.author_info_orcid) for signature in signatures.values()
                    )
                    if normalized_orcid is not None
                }
            )
        ),
        "repeated_orcid_group_count": int(len(repeated_orcid_groups)),
        "repeated_orcid_signature_count": int(
            sum(len(signature_ids) for signature_ids in repeated_orcid_groups.values())
        ),
        "thresholds": thresholds,
        "artifacts": {
            "threshold_summary_csv": str(summary_csv_path),
            "threshold_summary_markdown": str(summary_md_path),
            "threshold_overview_plot": str(plot_path),
        },
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    summary_json_path.write_text(
        json.dumps({"metadata": metadata, "rows": summary_df.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_json_path}")


if __name__ == "__main__":
    main()
