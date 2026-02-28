#!/usr/bin/env python
"""Scan all dataset signature files and find the single largest block.

Block = author_info["block"] field in each signature.
Prints a ranked table per dataset and the overall winner.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Datasets that have standard {name}_signatures.json files
DATASET_NAMES = [
    "aminer",
    "arnetminer",
    "augmented",
    "inspire",
    "inventors_s2and",
    "kisti",
    "medline",
    "orcid",
    "pubmed",
    "qian",
    "zbmath",
]

# Also check s2and_mini subdirectories
S2AND_MINI_NAMES = [
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
]


def find_signature_files():
    """Yield (label, path) for every signatures JSON we can find."""
    for name in DATASET_NAMES:
        p = DATA_DIR / name / f"{name}_signatures.json"
        if p.exists():
            yield name, p

    for name in S2AND_MINI_NAMES:
        p = DATA_DIR / "s2and_mini" / name / f"{name}_signatures.json"
        if p.exists():
            yield f"s2and_mini/{name}", p


def count_blocks(sig_path: Path) -> Counter:
    """Load signatures JSON and count block sizes."""
    with open(sig_path, encoding="utf-8") as f:
        sigs = json.load(f)
    block_counts = Counter()
    for _sig_id, sig in sigs.items():
        block = sig.get("author_info", {}).get("block", "")
        block_counts[block] += 1
    return block_counts


def main():
    global_best_block = ""
    global_best_count = 0
    global_best_dataset = ""
    global_best_top5 = []

    results = []

    for label, path in find_signature_files():
        print(f"Scanning {label} ... ", end="", flush=True)
        try:
            bc = count_blocks(path)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        total_sigs = sum(bc.values())
        total_blocks = len(bc)
        top5 = bc.most_common(5)
        biggest_block, biggest_count = top5[0] if top5 else ("", 0)
        print(f"{total_sigs:,} sigs, {total_blocks:,} blocks, largest: {biggest_block!r} ({biggest_count:,})")

        results.append(
            {
                "dataset": label,
                "total_sigs": total_sigs,
                "total_blocks": total_blocks,
                "top5": top5,
            }
        )

        if biggest_count > global_best_count:
            global_best_count = biggest_count
            global_best_block = biggest_block
            global_best_dataset = label
            global_best_top5 = top5

    print("\n" + "=" * 80)
    print("OVERALL LARGEST BLOCK")
    print(f"  Dataset:    {global_best_dataset}")
    print(f"  Block:      {global_best_block!r}")
    print(f"  Signatures: {global_best_count:,}")
    print()

    print("Top 5 blocks in that dataset:")
    for rank, (block, count) in enumerate(global_best_top5, 1):
        print(f"  {rank}. {block!r}: {count:,}")

    print("\n" + "=" * 80)
    print("ALL DATASETS - LARGEST BLOCK SUMMARY")
    print(f"{'Dataset':<25} {'Total Sigs':>12} {'Blocks':>8} {'Largest Block':<25} {'Size':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["top5"][0][1] if x["top5"] else 0, reverse=True):
        top = r["top5"][0] if r["top5"] else ("", 0)
        print(f"{r['dataset']:<25} {r['total_sigs']:>12,} {r['total_blocks']:>8,} {top[0]!r:<25} {top[1]:>8,}")


if __name__ == "__main__":
    main()
