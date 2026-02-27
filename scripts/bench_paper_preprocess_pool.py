"""Benchmark paper preprocessing: threads vs processes for UniversalPool.

Tests whether switching UniversalPool from ThreadPoolExecutor (default)
to ProcessPoolExecutor improves wall-clock time for preprocess_papers_parallel,
which is GIL-bound Python string work (normalize_text, get_text_ngrams, etc.).

NOTE: On Windows, spawn-context processes don't inherit module-level globals.
preprocess_paper_1 depends on `global_preprocess`, so we must either
(a) set it via an initializer, or (b) avoid ProcessPoolExecutor entirely.
This benchmark handles it by directly using multiprocessing.Pool with an
initializer for the process-based case.

Usage:
    .venv/Scripts/python.exe -u scripts/bench_paper_preprocess_pool.py --dataset kisti --n-jobs 8
    .venv/Scripts/python.exe -u scripts/bench_paper_preprocess_pool.py --dataset kisti --n-jobs 8 --rounds 3
    .venv/Scripts/python.exe -u scripts/bench_paper_preprocess_pool.py --dataset kisti --process-start-method spawn
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_raw_papers_and_signatures(dataset_name: str):
    """Load papers as namedtuples with in_signatures flags set."""
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


# --- Worker initializer for spawn-context processes ---


def _init_worker_global_preprocess(preprocess_flag: bool):
    """Set the module-level global that preprocess_paper_1 depends on."""
    import s2and.data as data_mod

    data_mod.global_preprocess = preprocess_flag


# --- Direct process pool that properly initializes workers ---

CHUNK_SIZE = 1000


def _resolve_process_start_method(requested: str) -> str:
    """Resolve process start method for this platform."""
    if requested != "auto":
        return requested
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return "spawn"
    return "fork"


def _preprocess_papers_with_processes(
    papers_dict, n_jobs: int, preprocess: bool = True, process_start_method: str = "auto"
):
    """Run preprocess_papers_parallel logic using ProcessPoolExecutor with initializer."""
    from collections import Counter

    import s2and.data as data_mod

    # Set in parent too (for consistency)
    data_mod.global_preprocess = preprocess

    output = {}
    start_method = _resolve_process_start_method(process_start_method)
    available = mp.get_all_start_methods()
    if start_method not in available:
        raise ValueError(f"Unsupported process start method '{start_method}' on this platform. Available: {available}")
    ctx = mp.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        mp_context=ctx,
        initializer=_init_worker_global_preprocess,
        initargs=(preprocess,),
    ) as executor:
        # Submit in chunks, collect in order
        items = list(papers_dict.items())
        futures = []
        for chunk_start in range(0, len(items), CHUNK_SIZE):
            chunk = items[chunk_start : chunk_start + CHUNK_SIZE]
            futures.append(executor.submit(_process_chunk, chunk))

        for fut in futures:
            for key, value in fut.result():
                output[key] = value

    # Ensure reference_details exists (mirrors preprocess_papers_parallel)
    if preprocess:
        empty_tuple = (Counter(), Counter(), Counter(), Counter())
        for k, v in output.items():
            if v.reference_details is None:
                output[k] = v._replace(reference_details=empty_tuple)

    return output


def _process_chunk(chunk):
    """Process a chunk of papers in a worker process."""
    from s2and.data import preprocess_paper_1

    results = []
    for item in chunk:
        results.append(preprocess_paper_1(item))
    return results


def bench_threads(papers_dict, n_jobs: int, preprocess: bool = True):
    """Run with threads (current default)."""
    from s2and.data import preprocess_papers_parallel

    papers_copy = copy.deepcopy(papers_dict)
    start = time.perf_counter()
    result = preprocess_papers_parallel(papers_copy, n_jobs, preprocess)
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def bench_serial(papers_dict, preprocess: bool = True):
    """Run single-threaded."""
    from s2and.data import preprocess_papers_parallel

    papers_copy = copy.deepcopy(papers_dict)
    start = time.perf_counter()
    result = preprocess_papers_parallel(papers_copy, 1, preprocess)
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def bench_processes(papers_dict, n_jobs: int, preprocess: bool = True, process_start_method: str = "auto"):
    """Run with process pool (properly initializing globals for spawn/fork)."""
    papers_copy = copy.deepcopy(papers_dict)
    start = time.perf_counter()
    result = _preprocess_papers_with_processes(
        papers_copy, n_jobs, preprocess, process_start_method=process_start_method
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def main():
    parser = argparse.ArgumentParser(description="Benchmark paper preprocessing: threads vs processes")
    parser.add_argument("--dataset", default="kisti", help="Dataset name (default: kisti)")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds per config (default: 3)")
    parser.add_argument("--single-thread", action="store_true", help="Also benchmark n_jobs=1 as baseline")
    parser.add_argument(
        "--process-start-method",
        default="auto",
        choices=["auto", "spawn", "fork", "forkserver"],
        help="Multiprocessing start method for process benchmark (default: auto).",
    )
    args = parser.parse_args()

    resolved_method = _resolve_process_start_method(args.process_start_method)
    print(f"Process start method: requested={args.process_start_method} resolved={resolved_method}", flush=True)
    print(f"Loading {args.dataset} papers...", flush=True)
    papers, n_sigs = load_raw_papers_and_signatures(args.dataset)
    print(f"  {len(papers)} papers (from {n_sigs} signatures)", flush=True)

    configs = []
    if args.single_thread:
        configs.append(("serial", "serial"))
    configs.append(("threads", "threads"))
    configs.append(("processes", "processes"))

    results = {}
    for label, mode in configs:
        display = f"{label} (n_jobs={'1' if mode == 'serial' else str(args.n_jobs)})"
        times = []
        for r in range(args.rounds):
            if mode == "serial":
                elapsed, count = bench_serial(papers)
            elif mode == "threads":
                elapsed, count = bench_threads(papers, args.n_jobs)
            else:
                elapsed, count = bench_processes(papers, args.n_jobs, process_start_method=args.process_start_method)
            times.append(elapsed)
            print(f"  {display} round {r+1}/{args.rounds}: {elapsed:.3f}s ({count} papers)", flush=True)
        avg = sum(times) / len(times)
        best = min(times)
        results[display] = {"times": times, "avg": avg, "best": best}

    print(flush=True)
    print("=" * 65, flush=True)
    print(f"Results: {args.dataset} ({len(papers)} papers, {args.rounds} rounds)", flush=True)
    print("=" * 65, flush=True)
    for label, data in results.items():
        times_str = ", ".join(f"{t:.3f}" for t in data["times"])
        print(f"  {label:35s}  avg={data['avg']:.3f}s  best={data['best']:.3f}s  [{times_str}]", flush=True)

    # Comparison
    labels = list(results.keys())
    thread_labels = [label_name for label_name in labels if "threads" in label_name]
    proc_labels = [label_name for label_name in labels if "processes" in label_name]
    if thread_labels and proc_labels:
        t_avg = results[thread_labels[0]]["avg"]
        p_avg = results[proc_labels[0]]["avg"]
        delta = t_avg - p_avg
        pct = (delta / t_avg) * 100 if t_avg > 0 else 0
        print(flush=True)
        if delta > 0:
            print(f"  >>> Processes faster by {delta:.3f}s ({pct:.1f}%)", flush=True)
        else:
            print(f"  >>> Threads faster by {-delta:.3f}s ({-pct:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
