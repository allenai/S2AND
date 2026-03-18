"""Sweep subblocking thresholds on a giant extracted block and write plots/tables.

This script evaluates candidate `maximum_size` thresholds using repeated ORCID groups as
an external preservation signal. Per-signature ORCIDs are masked before subblocking so the
metric reflects only name-based and SPECTER-based partitioning, not the ORCID co-location
override in `make_subblocks(...)`.

It also reports how often SPECTER fallback was actually invoked while building subblocks,
plus the size of the downstream retrieval target:

`single-letter-first-name signatures UNION signatures that end up in a SPECTER-made subblock`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from s2and.data import ANDData, Signature
from s2and.subblocking import _signature_name_parts_for_subblocking, make_subblocks_with_telemetry
from s2and.text import ORCID_PATTERN


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory with signatures/papers/specter files.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        required=True,
        help="Subblocking thresholds to evaluate, for example: --thresholds 2000 4000 6000 7500 10000",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON/plot artifacts.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Worker count for ANDData preprocessing.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for deterministic preprocessing.")
    return parser.parse_args()


def _required_file(dataset_root: Path, filename: str) -> Path:
    """Return a required file under `dataset_root`."""
    path = dataset_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def _normalize_orcid(orcid: str | None) -> str | None:
    """Normalize ORCID to its 16-character uppercase compact form."""
    if not orcid:
        return None
    matches = ORCID_PATTERN.findall(orcid)
    if not matches:
        return None
    return matches[0].upper().replace("-", "")


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


def _single_letter_first_name_ids(signatures: dict[str, Signature]) -> set[str]:
    """Return signature IDs whose normalized first name is empty or one character."""
    return {
        str(signature_id)
        for signature_id, signature in signatures.items()
        if len(_signature_name_parts_for_subblocking(signature)[0]) <= 1
    }


def _masked_signature_map(signatures: dict[str, Signature]) -> dict[str, Signature]:
    """Return a shallow ORCID-masked copy of `signatures` for subblocking evaluation."""
    return {
        str(signature_id): signature._replace(author_info_orcid=None) for signature_id, signature in signatures.items()
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


def load_dataset(dataset_root: Path, n_jobs: int, random_seed: int) -> ANDData:
    """Load the extracted block as ANDData."""
    signatures_path = _required_file(dataset_root, "signatures.json")
    papers_path = _required_file(dataset_root, "papers.json")
    specter_path = _required_file(dataset_root, "specter.pickle")

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ.setdefault("S2AND_BACKEND", "python")
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))

    return ANDData(
        signatures=str(signatures_path),
        papers=str(papers_path),
        name=f"{dataset_root.name}_threshold_sweep",
        mode="inference",
        clusters=None,
        specter_embeddings=str(specter_path),
        cluster_seeds=None,
        altered_cluster_signatures=None,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=int(n_jobs),
        load_name_counts=False,
        preprocess=True,
        random_seed=int(random_seed),
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )


def main() -> None:
    """Run the threshold sweep and write artifacts."""
    args = parse_args()
    thresholds = sorted(set(int(threshold) for threshold in args.thresholds))
    if not thresholds:
        raise ValueError("Need at least one threshold")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dataset = load_dataset(args.dataset_root, n_jobs=int(args.n_jobs), random_seed=int(args.random_seed))
    all_signature_ids = sorted(str(signature_id) for signature_id in dataset.signatures.keys())
    repeated_orcid_groups = _build_repeated_orcid_groups(dataset.signatures)
    single_letter_ids = _single_letter_first_name_ids(dataset.signatures)
    masked_dataset = SimpleNamespace(
        signatures=_masked_signature_map(dataset.signatures),
        papers=dataset.papers,
        specter_embeddings=dataset.specter_embeddings,
        random_seed=dataset.random_seed,
    )

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_start = time.perf_counter()
        subblocks, telemetry = make_subblocks_with_telemetry(
            all_signature_ids,
            masked_dataset,
            maximum_size=int(threshold),
        )
        signature_to_subblock = _signature_to_subblock(subblocks)
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
        rows.append(row)
        print(
            f"threshold={threshold} preserved={row['orcid_group_preservation_rate']:.4f} "
            f"specter_invocations={row['specter_invocation_count']} "
            f"target_sigs={row['retrieval_target_signature_count']} "
            f"elapsed_s={row['elapsed_seconds']:.3f}"
        )

    summary_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    summary_csv_path = output_dir / "threshold_summary.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "threshold_summary.md"
    plot_path = output_dir / "threshold_overview.png"

    summary_df.to_csv(summary_csv_path, index=False)
    summary_md_path.write_text(_render_markdown_table(summary_df.to_dict(orient="records")), encoding="utf-8")
    _write_overview_plot(summary_df, plot_path)

    specter_embeddings = dataset.specter_embeddings
    embedding_count = len(specter_embeddings) if isinstance(specter_embeddings, dict) else 0

    metadata = {
        "dataset_root": str(args.dataset_root),
        "signature_count": int(len(all_signature_ids)),
        "paper_count": int(len(dataset.papers)),
        "embedding_count": int(embedding_count),
        "single_letter_first_name_signature_count": int(len(single_letter_ids)),
        "orcid_signature_count": int(
            sum(1 for signature in dataset.signatures.values() if signature.author_info_orcid)
        ),
        "normalized_orcid_count": int(
            len(
                {
                    normalized_orcid
                    for normalized_orcid in (
                        _normalize_orcid(signature.author_info_orcid) for signature in dataset.signatures.values()
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
