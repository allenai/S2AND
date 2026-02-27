"""Benchmark paper preprocessing: threads vs processes via UniversalPool.

Tests UniversalPool with use_threads=True vs use_threads=False to measure
whether process-based parallelism helps for GIL-bound preprocessing work.

Separates pool creation time from work time so Windows spawn overhead
is visible but doesn't obscure the parallelism signal.

Usage:
    .venv/Scripts/python.exe -u scripts/bench_paper_preprocess_pool.py --dataset kisti
    .venv/Scripts/python.exe -u scripts/bench_paper_preprocess_pool.py --dataset kisti --n-jobs 8 --rounds 3 --serial
"""

import argparse
import json
import os
import platform
import sys
import time
from functools import partial

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset(dataset_name: str):
    """Load papers as Paper namedtuples with in_signatures flags set.

    Returns (papers_dict, n_signatures).
    """
    from s2and.data import Author, Paper

    sig_path = os.path.join(DATA_DIR, dataset_name, f"{dataset_name}_signatures.json")
    paper_path = os.path.join(DATA_DIR, dataset_name, f"{dataset_name}_papers.json")

    with open(sig_path) as f:
        raw_sigs = json.load(f)
    with open(paper_path) as f:
        raw_papers = json.load(f)

    paper_ids_from_sigs = set()
    for sig_dict in raw_sigs.values():
        paper_id = sig_dict.get("paper_id", sig_dict.get("paperId", ""))
        paper_ids_from_sigs.add(str(paper_id))

    papers = {}
    for paper_id_raw, p in raw_papers.items():
        paper_id_str = str(paper_id_raw)
        if paper_id_str not in paper_ids_from_sigs:
            continue
        try:
            paper_id = int(paper_id_str)
        except (TypeError, ValueError):
            continue
        authors_raw = p.get("authors", [])
        authors = []
        for i, a in enumerate(authors_raw):
            if isinstance(a, dict):
                authors.append(Author(position=a.get("position", i), author_name=a.get("author_name", "")))
            elif isinstance(a, list | tuple):
                authors.append(Author(position=a[0] if len(a) > 0 else i, author_name=a[1] if len(a) > 1 else ""))
            else:
                authors.append(Author(position=i, author_name=str(a)))
        paper = Paper(
            paper_id=paper_id,
            title=p.get("title", "") or "",
            authors=authors,
            venue=p.get("venue", "") or "",
            journal_name=p.get("journal_name", "") or "",
            year=p.get("year"),
            references=p.get("references"),
            has_abstract=bool(p.get("abstract", "") or p.get("has_abstract", False)),
            predicted_language=None,
            is_english=None,
            is_reliable=None,
            title_ngrams_words=None,
            title_ngrams_chars=None,
            venue_ngrams=None,
            journal_ngrams=None,
            reference_details=None,
            in_signatures=(paper_id_str in paper_ids_from_sigs),
        )
        papers[paper_id_str] = paper

    return papers, len(raw_sigs)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def bench_serial(papers_dict):
    """Serial baseline -- no pool, raw loop. Matches production n_jobs=1 path."""
    from s2and.data import preprocess_paper_1

    t0 = time.perf_counter()
    count = 0
    for item in papers_dict.items():
        preprocess_paper_1(item, preprocess=True)
        count += 1
    elapsed = time.perf_counter() - t0

    return elapsed, count


def bench(papers_dict, n_jobs, use_threads, chunk_size=1000):
    """Benchmark UniversalPool.imap for preprocess_paper_1.

    Returns (pool_create_secs, work_secs, n_papers_out).
    """
    from s2and.data import preprocess_paper_1
    from s2and.mp import UniversalPool

    func = partial(preprocess_paper_1, preprocess=True)

    # --- time pool creation separately ---
    t0 = time.perf_counter()
    pool = UniversalPool(processes=n_jobs, use_threads=use_threads)
    pool_time = time.perf_counter() - t0

    # --- time work ---
    t1 = time.perf_counter()
    count = 0
    with pool:
        for _key, _value in pool.imap(func, papers_dict.items(), chunk_size):
            count += 1
    work_time = time.perf_counter() - t1

    return pool_time, work_time, count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark paper preprocessing: threads vs processes via UniversalPool"
    )
    parser.add_argument("--dataset", default="kisti", help="Dataset name (default: kisti)")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--rounds", type=int, default=3, help="Rounds per config (default: 3)")
    parser.add_argument("--serial", action="store_true", help="Include serial (no-pool) baseline")
    parser.add_argument("--chunk-size", type=int, default=1000, help="imap chunk size (default: 1000)")
    args = parser.parse_args()

    print(f"Platform: {platform.system()} ({platform.platform()})")
    print(f"Python:   {sys.version}")
    print(f"Workers:  {args.n_jobs}    Chunk size: {args.chunk_size}    Rounds: {args.rounds}")
    print(flush=True)

    print(f"Loading dataset '{args.dataset}'...")
    papers, n_sigs = load_dataset(args.dataset)
    print(f"  {len(papers):,} papers ({n_sigs:,} signatures)")
    print(flush=True)

    # (label, use_threads) -- use_threads=None means serial
    configs = []
    if args.serial:
        configs.append(("serial (no pool)", None))
    configs.append((f"threads x{args.n_jobs}", True))
    configs.append((f"processes x{args.n_jobs}", False))

    results = {}
    for label, use_threads in configs:
        print(f"--- {label} ---")
        rounds_data = []
        for r in range(args.rounds):
            if use_threads is None:
                elapsed, count = bench_serial(papers)
                print(f"  round {r + 1}: {elapsed:.3f}s  ({count:,} papers)", flush=True)
                rounds_data.append({"work": elapsed, "pool": 0.0, "count": count})
            else:
                pool_t, work_t, count = bench(papers, args.n_jobs, use_threads=use_threads, chunk_size=args.chunk_size)
                total = pool_t + work_t
                print(
                    f"  round {r + 1}: pool={pool_t:.3f}s  work={work_t:.3f}s  total={total:.3f}s  ({count:,} papers)",
                    flush=True,
                )
                rounds_data.append({"work": work_t, "pool": pool_t, "count": count})
        results[label] = rounds_data
        print(flush=True)

    # --- Summary table ---
    print("=" * 75)
    print(f"Summary: {args.dataset} | {len(papers):,} papers | {args.rounds} rounds | chunk={args.chunk_size}")
    print("=" * 75)
    print(f"  {'Config':<25s}  {'Avg Work':>9s}  {'Best Work':>10s}  {'Avg Pool':>9s}  {'Avg Total':>10s}")
    print(f"  {'-' * 25}  {'-' * 9}  {'-' * 10}  {'-' * 9}  {'-' * 10}")
    for label, rounds_data in results.items():
        avg_work = sum(d["work"] for d in rounds_data) / len(rounds_data)
        best_work = min(d["work"] for d in rounds_data)
        avg_pool = sum(d["pool"] for d in rounds_data) / len(rounds_data)
        avg_total = avg_work + avg_pool
        print(f"  {label:<25s}  {avg_work:>8.3f}s  {best_work:>9.3f}s  {avg_pool:>8.3f}s  {avg_total:>9.3f}s")

    # --- Thread vs Process comparison ---
    thread_key = f"threads x{args.n_jobs}"
    proc_key = f"processes x{args.n_jobs}"
    if thread_key in results and proc_key in results:
        t_work = sum(d["work"] for d in results[thread_key]) / len(results[thread_key])
        p_work = sum(d["work"] for d in results[proc_key]) / len(results[proc_key])
        t_pool = sum(d["pool"] for d in results[thread_key]) / len(results[thread_key])
        p_pool = sum(d["pool"] for d in results[proc_key]) / len(results[proc_key])

        delta_work = t_work - p_work
        pct_work = (delta_work / t_work) * 100 if t_work > 0 else 0
        print()
        if delta_work > 0:
            print(f"  Work only: processes faster by {delta_work:.3f}s ({pct_work:.1f}%)")
        else:
            print(f"  Work only: threads faster by {-delta_work:.3f}s ({-pct_work:.1f}%)")

        t_total = t_work + t_pool
        p_total = p_work + p_pool
        delta_total = t_total - p_total
        if abs(delta_total - delta_work) > 0.01:
            winner = "processes" if delta_total > 0 else "threads"
            print(f"  Including pool creation: {winner} faster by {abs(delta_total):.3f}s total")


if __name__ == "__main__":
    main()
