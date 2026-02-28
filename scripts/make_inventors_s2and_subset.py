"""Build a stratified inventors subset with merged cross-block clusters.

Workflow:
1) sample N seed blocks from `author_info.block`, stratified by block size
2) expand to closure over (block -> signatures -> clusters -> blocks)
3) merge blocks connected by shared clusters; choose canonical name as
   most frequent original block in each merged component
4) write subset data files:
   - <prefix>_clusters.json
   - <prefix>_signatures.json (with both `block` and `given_block` overwritten)
   - <prefix>_papers.json
   - <prefix>_train_keys.json / <prefix>_val_keys.json (80/20 split on merged blocks, stratified)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNATURE_LOG_INTERVAL = 500_000
CLUSTER_LOG_INTERVAL = 250_000
PAPER_LOG_INTERVAL = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=_PROJECT_ROOT / "data" / "inventors")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "scratch" / "inventors_s2and")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for written subset files. Defaults to output directory name.",
    )
    parser.add_argument("--n-blocks", type=int, default=1000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--sample-strata-bins", type=int, default=20)
    parser.add_argument("--split-strata-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required for unbounded full-dataset processing.",
    )
    parser.add_argument("--limit-signatures", type=int, default=None, help="Optional debug cap for signatures passes.")
    parser.add_argument("--limit-clusters", type=int, default=None, help="Optional debug cap for cluster passes.")
    parser.add_argument("--limit-papers", type=int, default=None, help="Optional debug cap for paper pass.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_blocks <= 0:
        raise ValueError("--n-blocks must be > 0")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.sample_strata_bins <= 1:
        raise ValueError("--sample-strata-bins must be > 1")
    if args.split_strata_bins <= 1:
        raise ValueError("--split-strata-bins must be > 1")

    if (
        not args.full_run
        and args.limit_signatures is None
        and args.limit_clusters is None
        and args.limit_papers is None
    ):
        raise ValueError("Refusing unbounded run without explicit confirmation. Use --full-run or --limit-*.")

    for name in ("limit_signatures", "limit_clusters", "limit_papers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0 when set")


def make_strata(values: np.ndarray, max_bins: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("Cannot build strata for empty input")
    unique_count = np.unique(values).size
    bins = int(min(max_bins, unique_count))
    if bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int32)

    x = np.log1p(values.astype(np.float64))
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int32)
    return np.digitize(x, edges[1:-1], right=True).astype(np.int32)


def stratified_take(
    items: np.ndarray, sizes: np.ndarray, n_take: int, seed: int, max_bins: int
) -> tuple[np.ndarray, int, bool]:
    if n_take >= items.shape[0]:
        raise ValueError(f"n_take={n_take} must be < number of items={items.shape[0]}")

    upper_bins = int(min(max_bins, np.unique(sizes).size))
    for bins in range(upper_bins, 1, -1):
        strata = make_strata(sizes, bins)
        counts = np.bincount(strata)
        if counts.size > 0 and int(counts.min()) < 2:
            continue
        try:
            selected, _ = train_test_split(
                items,
                train_size=n_take,
                random_state=seed,
                stratify=strata,
            )
            return np.array(selected), bins, True
        except ValueError:
            continue

    selected, _ = train_test_split(
        items,
        train_size=n_take,
        random_state=seed,
        stratify=None,
    )
    return np.array(selected), 1, False


def stratified_train_val_split(
    blocks: np.ndarray, sizes: np.ndarray, val_ratio: float, seed: int, max_bins: int
) -> tuple[list[str], list[str], int, bool]:
    n_val = int(round(blocks.shape[0] * val_ratio))
    if n_val <= 0 or n_val >= blocks.shape[0]:
        raise ValueError("Invalid val size derived from val_ratio and number of blocks")

    upper_bins = int(min(max_bins, np.unique(sizes).size))
    for bins in range(upper_bins, 1, -1):
        strata = make_strata(sizes, bins)
        counts = np.bincount(strata)
        if counts.size > 0 and int(counts.min()) < 2:
            continue
        if counts.size > n_val:
            continue
        try:
            train, val = train_test_split(
                blocks,
                test_size=val_ratio,
                random_state=seed,
                stratify=strata,
            )
            return list(train), list(val), bins, True
        except ValueError:
            continue

    train, val = train_test_split(
        blocks,
        test_size=val_ratio,
        random_state=seed,
        stratify=None,
    )
    return list(train), list(val), 1, False


def count_signatures_per_block(signatures_path: Path, limit_signatures: int | None = None) -> Counter[str]:
    block_counts: Counter[str] = Counter()
    count = 0
    with signatures_path.open("rb") as infile:
        for prefix, event, value in ijson.parse(infile):
            if event == "string" and prefix.endswith(".author_info.block"):
                block_counts[value] += 1
                count += 1
                if limit_signatures is not None and count >= limit_signatures:
                    break
    return block_counts


def collect_signature_ids_for_blocks(
    signatures_path: Path, block_set: set[str], limit_signatures: int | None = None
) -> set[str]:
    selected: set[str] = set()
    scanned = 0
    with signatures_path.open("rb") as infile:
        for signature_id, signature in ijson.kvitems(infile, ""):
            scanned += 1
            if signature["author_info"]["block"] in block_set:
                selected.add(signature_id)
            if limit_signatures is not None and scanned >= limit_signatures:
                break
    return selected


def collect_signature_blocks_for_ids(
    signatures_path: Path, signature_ids: set[str], limit_signatures: int | None = None
) -> set[str]:
    if not signature_ids:
        return set()
    needed = signature_ids.__contains__
    blocks: set[str] = set()
    scanned = 0
    found = 0
    target = len(signature_ids)
    with signatures_path.open("rb") as infile:
        for signature_id, signature in ijson.kvitems(infile, ""):
            scanned += 1
            if needed(signature_id):
                blocks.add(signature["author_info"]["block"])
                found += 1
                if limit_signatures is None and found >= target:
                    break
            if limit_signatures is not None and scanned >= limit_signatures:
                break
    return blocks


def expand_signatures_via_clusters(
    clusters_path: Path, seed_signature_ids: set[str], limit_clusters: int | None = None
) -> tuple[set[str], int]:
    if not seed_signature_ids:
        return set(), 0
    contains_seed = seed_signature_ids.__contains__
    expanded: set[str] = set()
    touched_clusters = 0
    scanned = 0
    with clusters_path.open("rb") as infile:
        for _cluster_id, cluster in ijson.kvitems(infile, ""):
            scanned += 1
            sig_ids = cluster.get("signature_ids", [])
            hit = False
            for sig_id in sig_ids:
                if contains_seed(sig_id):
                    hit = True
                    break
            if hit:
                touched_clusters += 1
                expanded.update(sig_ids)
            if limit_clusters is not None and scanned >= limit_clusters:
                break
    return expanded, touched_clusters


class DSU:
    def __init__(self, items: set[str]) -> None:
        self.parent: dict[str, str] = {x: x for x in items}
        self.rank: dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main() -> None:
    args = parse_args()
    validate_args(args)

    signatures_path = args.input_dir / "signatures.json"
    clusters_path = args.input_dir / "clusters.json"
    papers_path = args.input_dir / "papers.json"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = (args.output_prefix or output_dir.name).rstrip("_")
    if not output_prefix:
        raise ValueError("Resolved empty output prefix. Set --output-prefix explicitly.")
    out_clusters_path = output_dir / f"{output_prefix}_clusters.json"
    out_signatures_path = output_dir / f"{output_prefix}_signatures.json"
    out_papers_path = output_dir / f"{output_prefix}_papers.json"
    out_train_keys_path = output_dir / f"{output_prefix}_train_keys.json"
    out_val_keys_path = output_dir / f"{output_prefix}_val_keys.json"
    out_summary_path = output_dir / f"{output_prefix}_subset_summary.json"

    start = time.time()
    print("Pass 1/9: counting signatures per raw block (author_info.block)...")
    block_counts = count_signatures_per_block(signatures_path, limit_signatures=args.limit_signatures)
    if len(block_counts) <= args.n_blocks:
        raise ValueError(f"Need > {args.n_blocks} unique blocks to sample from; found {len(block_counts):,}.")

    all_blocks = np.array(list(block_counts.keys()))
    all_sizes = np.array([block_counts[b] for b in all_blocks], dtype=np.int32)
    sampled_blocks, sample_bins_used, sample_was_stratified = stratified_take(
        items=all_blocks,
        sizes=all_sizes,
        n_take=args.n_blocks,
        seed=args.seed,
        max_bins=args.sample_strata_bins,
    )
    seed_blocks = set(str(x) for x in sampled_blocks.tolist())
    print(
        f"Seed sampled blocks={len(seed_blocks):,} "
        f"(stratified={sample_was_stratified}, strata_bins={sample_bins_used})"
    )

    print("Pass 2/9-4/9: closure expansion over block/signature/cluster graph...")
    closure_blocks = set(seed_blocks)
    closure_iterations = 0
    while True:
        closure_iterations += 1
        print(f"  Closure iteration {closure_iterations}...")
        sig_ids_in_blocks = collect_signature_ids_for_blocks(
            signatures_path,
            closure_blocks,
            limit_signatures=args.limit_signatures,
        )
        expanded_sig_ids, touched_clusters = expand_signatures_via_clusters(
            clusters_path,
            sig_ids_in_blocks,
            limit_clusters=args.limit_clusters,
        )
        expanded_blocks = collect_signature_blocks_for_ids(
            signatures_path,
            expanded_sig_ids,
            limit_signatures=args.limit_signatures,
        )
        new_blocks = expanded_blocks - closure_blocks
        print(
            f"    blocks={len(closure_blocks):,}, sigs_in_blocks={len(sig_ids_in_blocks):,}, "
            f"touched_clusters={touched_clusters:,}, expanded_sigs={len(expanded_sig_ids):,}, "
            f"expanded_blocks={len(expanded_blocks):,}, new_blocks={len(new_blocks):,}"
        )
        if not new_blocks:
            final_raw_blocks = set(closure_blocks)
            final_signature_ids = set(sig_ids_in_blocks)
            break
        closure_blocks.update(new_blocks)

    print("Pass 5/9: loading final signatures and base counts...")
    final_signature_records: dict[str, dict] = {}
    final_signature_block: dict[str, str] = {}
    final_paper_ids: set[str] = set()
    raw_block_counts_in_subset: Counter[str] = Counter()

    scanned_sigs = 0
    keep_sig = final_signature_ids.__contains__
    with signatures_path.open("rb") as infile:
        for signature_id, signature in ijson.kvitems(infile, ""):
            scanned_sigs += 1
            if keep_sig(signature_id):
                final_signature_records[signature_id] = signature
                block = signature["author_info"]["block"]
                final_signature_block[signature_id] = block
                raw_block_counts_in_subset[block] += 1
                final_paper_ids.add(str(signature["paper_id"]))
            if scanned_sigs % SIGNATURE_LOG_INTERVAL == 0:
                elapsed = time.time() - start
                print(
                    f"  signatures scanned={scanned_sigs:,}, kept={len(final_signature_records):,}, "
                    f"elapsed={elapsed:,.1f}s"
                )
            if args.limit_signatures is not None and scanned_sigs >= args.limit_signatures:
                break

    print("Pass 6/9: scanning clusters, building block merges, writing subset clusters...")
    dsu = DSU(set(final_raw_blocks))
    kept_cluster_count = 0
    trimmed_cluster_count = 0
    scanned_clusters = 0
    keep_sig = final_signature_ids.__contains__

    with out_clusters_path.open("w", encoding="utf-8") as out:
        out.write("{")
        first = True
        with clusters_path.open("rb") as infile:
            for cluster_id, cluster in ijson.kvitems(infile, ""):
                scanned_clusters += 1
                sig_ids = cluster.get("signature_ids", [])
                selected_sig_ids: list[str] = []
                for sig_id in sig_ids:
                    if keep_sig(sig_id):
                        selected_sig_ids.append(sig_id)

                if selected_sig_ids:
                    if len(selected_sig_ids) < len(sig_ids):
                        trimmed_cluster_count += 1
                    cluster_out = dict(cluster)
                    cluster_out["signature_ids"] = selected_sig_ids

                    # Merge any raw blocks that co-occur in a kept cluster.
                    blocks_in_cluster = {final_signature_block[sid] for sid in selected_sig_ids}
                    blocks_list = list(blocks_in_cluster)
                    for i in range(1, len(blocks_list)):
                        dsu.union(blocks_list[0], blocks_list[i])

                    if not first:
                        out.write(",")
                    json.dump(cluster_id, out, ensure_ascii=True)
                    out.write(":")
                    json.dump(cluster_out, out, ensure_ascii=True)
                    first = False
                    kept_cluster_count += 1

                if scanned_clusters % CLUSTER_LOG_INTERVAL == 0:
                    elapsed = time.time() - start
                    print(
                        f"  clusters scanned={scanned_clusters:,}, kept={kept_cluster_count:,}, "
                        f"trimmed={trimmed_cluster_count:,}, elapsed={elapsed:,.1f}s"
                    )
                if args.limit_clusters is not None and scanned_clusters >= args.limit_clusters:
                    break
        out.write("}")

    # Determine canonical merged block name per component: most frequent raw block in subset.
    component_members: dict[str, list[str]] = {}
    for block in final_raw_blocks:
        root = dsu.find(block)
        component_members.setdefault(root, []).append(block)

    component_canonical: dict[str, str] = {}
    for root, members in component_members.items():
        members_sorted = sorted(members, key=lambda b: (-raw_block_counts_in_subset[b], b))
        component_canonical[root] = members_sorted[0]

    raw_to_canonical: dict[str, str] = {}
    for block in final_raw_blocks:
        raw_to_canonical[block] = component_canonical[dsu.find(block)]

    print("Pass 7/9: writing signatures with merged canonical block labels...")
    merged_block_counts: Counter[str] = Counter()
    with out_signatures_path.open("w", encoding="utf-8") as out:
        out.write("{")
        first = True
        for signature_id, signature in final_signature_records.items():
            canonical = raw_to_canonical[final_signature_block[signature_id]]
            signature["author_info"]["block"] = canonical
            signature["author_info"]["given_block"] = canonical
            merged_block_counts[canonical] += 1

            if not first:
                out.write(",")
            json.dump(signature_id, out, ensure_ascii=True)
            out.write(":")
            json.dump(signature, out, ensure_ascii=True)
            first = False
        out.write("}")

    print("Pass 8/9: writing papers...")
    keep_paper = final_paper_ids.__contains__
    kept_paper_count = 0
    scanned_papers = 0
    with out_papers_path.open("w", encoding="utf-8") as out:
        out.write("{")
        first = True
        with papers_path.open("rb") as infile:
            for paper_id, paper in ijson.kvitems(infile, ""):
                scanned_papers += 1
                if keep_paper(str(paper_id)):
                    if not first:
                        out.write(",")
                    json.dump(str(paper_id), out, ensure_ascii=True)
                    out.write(":")
                    json.dump(paper, out, ensure_ascii=True)
                    first = False
                    kept_paper_count += 1
                if scanned_papers % PAPER_LOG_INTERVAL == 0:
                    elapsed = time.time() - start
                    print(
                        f"  papers scanned={scanned_papers:,}, kept={kept_paper_count:,}, " f"elapsed={elapsed:,.1f}s"
                    )
                if args.limit_papers is not None and scanned_papers >= args.limit_papers:
                    break
        out.write("}")

    print("Pass 9/9: train/val split on merged blocks...")
    merged_blocks = np.array(sorted(merged_block_counts.keys()))
    merged_sizes = np.array([merged_block_counts[b] for b in merged_blocks], dtype=np.int32)
    train_blocks, val_blocks, split_bins_used, split_was_stratified = stratified_train_val_split(
        blocks=merged_blocks,
        sizes=merged_sizes,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_bins=args.split_strata_bins,
    )
    with out_train_keys_path.open("w", encoding="utf-8") as out:
        json.dump(train_blocks, out, ensure_ascii=True)
    with out_val_keys_path.open("w", encoding="utf-8") as out:
        json.dump(val_blocks, out, ensure_ascii=True)

    elapsed = time.time() - start
    summary = {
        "config": {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "output_prefix": output_prefix,
            "n_blocks": args.n_blocks,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "sample_strata_bins": args.sample_strata_bins,
            "split_strata_bins": args.split_strata_bins,
            "full_run": args.full_run,
            "limit_signatures": args.limit_signatures,
            "limit_clusters": args.limit_clusters,
            "limit_papers": args.limit_papers,
        },
        "counts": {
            "seed_sampled_blocks": len(seed_blocks),
            "closure_iterations": closure_iterations,
            "final_raw_blocks_before_merge": len(final_raw_blocks),
            "final_blocks_after_merge": int(len(merged_block_counts)),
            "kept_clusters": kept_cluster_count,
            "trimmed_clusters": trimmed_cluster_count,
            "kept_signatures": len(final_signature_records),
            "kept_papers": kept_paper_count,
            "train_blocks": len(train_blocks),
            "val_blocks": len(val_blocks),
        },
        "stratification": {
            "sample_was_stratified": sample_was_stratified,
            "sample_bins_used": sample_bins_used,
            "split_was_stratified": split_was_stratified,
            "split_bins_used": split_bins_used,
        },
        "outputs": {
            "clusters": str(out_clusters_path),
            "signatures": str(out_signatures_path),
            "papers": str(out_papers_path),
            "train_keys": str(out_train_keys_path),
            "val_keys": str(out_val_keys_path),
        },
        "timing_seconds": elapsed,
    }
    with out_summary_path.open("w", encoding="utf-8") as out:
        json.dump(summary, out, indent=2)

    print("Done. Wrote subset dataset:")
    print(f"  {out_clusters_path}")
    print(f"  {out_signatures_path}")
    print(f"  {out_papers_path}")
    print(f"  {out_train_keys_path}")
    print(f"  {out_val_keys_path}")
    print(f"  {out_summary_path}")
    print(f"Total elapsed: {elapsed:,.1f}s")


if __name__ == "__main__":
    main()
