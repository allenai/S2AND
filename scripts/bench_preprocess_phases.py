"""Benchmark preprocessing phases (papers, signatures).

Focus: compare serial vs threads vs processes via UniversalPool across OSes.

Phases:
  1) Papers: `preprocess_paper_1` across papers
  2) Signatures: `ANDData.preprocess_signatures` with swappable Python ngram backend

Notes:
  - Default `--limit-signatures` keeps the run small; set `--limit-signatures 0` for full dataset.
  - Signature benchmarking keeps all signature field normalization logic identical by calling the
    production method, but swaps `_python_signature_ngrams_batch` to test parallelism.

Usage:
  uv run python scripts/bench_preprocess_phases.py --dataset kisti --limit-signatures 0 --n-jobs 8 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any, TypeVar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

T = TypeVar("T")
PaperStageInputT = TypeVar("PaperStageInputT")
PaperStageOutputT = TypeVar("PaperStageOutputT")


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_limited_items(items: list[tuple[str, T]], limit: int) -> Iterator[tuple[str, T]]:
    if limit <= 0:
        yield from items
        return
    yield from items[:limit]


def _paper_id_from_raw_signature(sig: dict[str, Any]) -> str:
    paper_id = sig.get("paper_id", sig.get("paperId"))
    return str(paper_id) if paper_id is not None else ""


def load_dataset(*, data_dir: str, dataset: str, limit_signatures: int) -> tuple[dict[str, Any], dict[str, Any]]:
    sig_path = os.path.join(data_dir, dataset, f"{dataset}_signatures.json")
    paper_path = os.path.join(data_dir, dataset, f"{dataset}_papers.json")

    raw_sigs: dict[str, Any] = _load_json(sig_path)
    raw_papers: dict[str, Any] = _load_json(paper_path)

    sig_items = sorted(raw_sigs.items(), key=lambda kv: kv[0])
    limited_sigs = {k: v for k, v in _iter_limited_items(sig_items, limit_signatures)}

    needed_paper_ids = {str(_paper_id_from_raw_signature(sig)) for sig in limited_sigs.values()}
    needed_paper_ids.discard("")

    filtered_papers = {pid: paper for pid, paper in raw_papers.items() if str(pid) in needed_paper_ids}
    return limited_sigs, filtered_papers


def build_namedtuples(
    *,
    raw_signatures: dict[str, Any],
    raw_papers: dict[str, Any],
    use_orcid_id: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from s2and.data import Author, Paper, Signature

    signatures: dict[str, Any] = {}
    for signature_id, signature in raw_signatures.items():
        author_info = signature["author_info"]
        signatures[signature_id] = Signature(
            author_info_first=author_info["first"],
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle=author_info["middle"],
            author_info_middle_normalized_without_apostrophe=None,
            author_info_last_normalized=None,
            author_info_last=author_info["last"],
            author_info_suffix_normalized=None,
            author_info_suffix=author_info["suffix"],
            author_info_coauthors=None,
            author_info_coauthor_blocks=None,
            author_info_full_name=None,
            author_info_affiliations=author_info["affiliations"],
            author_info_affiliations_n_grams=None,
            author_info_coauthor_n_grams=None,
            author_info_email=author_info["email"],
            author_info_orcid=(
                author_info["source_ids"][0]
                if use_orcid_id and author_info.get("source_id_source") == "ORCID"
                else None
            ),
            author_info_name_counts=None,
            author_info_position=author_info["position"],
            author_info_block=author_info["block"],
            author_info_estimated_gender=author_info.get("estimated_gender", None),
            author_info_estimated_ethnicity=author_info.get("estimated_ethnicity", None),
            paper_id=signature.get("paper_id", signature.get("paperId")),
            sourced_author_source=signature.get("sourced_author_source", None),
            sourced_author_ids=signature.get("sourced_author_ids", []),
            author_id=signature.get("author_id", None),
            signature_id=signature["signature_id"],
        )

    papers: dict[str, Any] = {}
    for paper_id, paper in raw_papers.items():
        authors_raw = paper.get("authors", [])
        authors: list[Any] = []
        for i, author in enumerate(authors_raw):
            if isinstance(author, dict):
                authors.append(
                    Author(
                        position=author.get("position", i),
                        author_name=author.get("author_name", "") or "",
                    )
                )
            elif isinstance(author, list | tuple):
                authors.append(
                    Author(
                        position=author[0] if len(author) > 0 else i,
                        author_name=author[1] if len(author) > 1 else "",
                    )
                )
            else:
                authors.append(Author(position=i, author_name=str(author)))

        papers[str(paper_id)] = Paper(
            paper_id=paper.get("paper_id", int(paper_id) if str(paper_id).isdigit() else 0),
            title=paper.get("title", "") or "",
            authors=authors,
            venue=paper.get("venue", "") or "",
            journal_name=paper.get("journal_name", "") or "",
            year=paper.get("year"),
            has_abstract=bool(paper.get("abstract", "") or paper.get("has_abstract", False)),
            predicted_language=None,
            is_english=None,
            is_reliable=None,
            language_reliability=None,
            title_ngrams_words=None,
            title_ngrams_chars=None,
            venue_ngrams=None,
            journal_ngrams=None,
            in_signatures=True,
        )

    return signatures, papers


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}s"


@contextmanager
def _patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def _signature_ngrams_one(pair: tuple[str, str]) -> tuple[Counter, Counter]:
    from s2and.text import get_text_ngrams, get_text_ngrams_words

    coauthor_text, affiliation_text = pair
    coauthor_counter = (
        get_text_ngrams(coauthor_text, stopwords=None, use_bigrams=True, drop_short_tokens=False)
        if coauthor_text
        else Counter()
    )
    affiliation_counter = get_text_ngrams_words(affiliation_text, stopwords=set()) if affiliation_text else Counter()
    return coauthor_counter, affiliation_counter


def _run_paper_stage(
    *,
    label: str,
    items: list[PaperStageInputT],
    func: Callable[[PaperStageInputT], PaperStageOutputT],
    n_jobs: int,
    use_threads: bool | None,
    chunk_size: int,
) -> tuple[float, float, int]:
    from s2and.mp import UniversalPool

    if use_threads is None:
        t0 = time.perf_counter()
        out_count = 0
        for item in items:
            func(item)
            out_count += 1
        return 0.0, time.perf_counter() - t0, out_count

    t_pool0 = time.perf_counter()
    pool = UniversalPool(processes=n_jobs, use_threads=use_threads)
    pool_create = time.perf_counter() - t_pool0

    t1 = time.perf_counter()
    out_count = 0
    with pool:
        for _ in pool.imap(func, items, chunk_size):
            out_count += 1
    work = time.perf_counter() - t1
    _ = label  # keep param for symmetry/readability
    return pool_create, work, out_count


def _bench_phase(
    *,
    phase_name: str,
    run_once: Callable[[bool | None], tuple[float, float, int]],
    n_jobs: int,
    rounds: int,
    configs: Sequence[tuple[str, bool | None]] | None = None,
) -> None:
    resolved_configs = configs or [
        ("serial", None),
        (f"threads x{n_jobs}", True),
        (f"processes x{n_jobs}", False),
    ]

    results: dict[str, list[dict[str, float]]] = {}
    print()
    print("=" * 80)
    print(phase_name)
    print("=" * 80)
    for label, use_threads in resolved_configs:
        rows: list[dict[str, float]] = []
        print(f"--- {label} ---")
        for r in range(rounds):
            pool_t, work_t, count = run_once(use_threads)
            total_t = pool_t + work_t
            print(
                f"  round {r + 1}: pool={_fmt(pool_t)} work={_fmt(work_t)} total={_fmt(total_t)} ({count:,} items)",
                flush=True,
            )
            rows.append({"pool": pool_t, "work": work_t, "total": total_t})
        results[label] = rows
        print(flush=True)

    print("-" * 80)
    print(f"{'Config':<18s}  {'Avg Work':>9s}  {'Best Work':>10s}  {'Avg Pool':>9s}  {'Avg Total':>10s}")
    for label, rows in results.items():
        work_values = [row["work"] for row in rows]
        pool_values = [row["pool"] for row in rows]
        total_values = [row["total"] for row in rows]
        print(
            f"{label:<18s}  {_fmt(_avg(work_values)):>9s}  {_fmt(_min(work_values)):>10s}  "
            f"{_fmt(_avg(pool_values)):>9s}  {_fmt(_avg(total_values)):>10s}"
        )


def _bench_signatures_preprocess(
    *,
    base_signatures: dict[str, Any],
    papers: dict[str, Any],
    n_jobs: int,
    rounds: int,
    ngram_chunk_size: int,
    name_counts_index: Any = None,
    show_breakdown: bool = False,
    configs: Sequence[tuple[str, bool | None]] | None = None,
) -> None:
    import s2and.data as data_mod
    from s2and.data import ANDData
    from s2and.mp import UniversalPool
    from s2and.runtime import build_runtime_context
    from s2and.text import compute_block

    def _make_ds(signatures: dict[str, Any]) -> Any:
        ds = ANDData.__new__(ANDData)
        ds.runtime_context = build_runtime_context("bench_preprocess_signatures")
        ds.arrow_dataset = None
        ds.preprocess = True
        ds.signatures = signatures
        ds.papers = papers
        ds.compute_block_fn = compute_block
        ds.name_counts_index = name_counts_index
        ds.name_counts_loaded = name_counts_index is not None
        return ds

    def _tqdm_wrapper(orig_tqdm):
        def _wrapped(*args, **kwargs):
            kwargs["disable"] = True
            return orig_tqdm(*args, **kwargs)

        return _wrapped

    def _run_serial() -> tuple[float, float, int]:
        signatures = dict(base_signatures)
        ds = _make_ds(signatures)
        ngram_time = 0.0
        ngram_calls = 0
        ngram_items = 0
        orig_batch = data_mod._python_signature_ngrams_batch

        def _timed_batch(coauthor_texts: list[str], affiliation_texts: list[str]):
            nonlocal ngram_time, ngram_calls, ngram_items
            t_ng = time.perf_counter()
            res = orig_batch(coauthor_texts, affiliation_texts)
            ngram_time += time.perf_counter() - t_ng
            ngram_calls += 1
            ngram_items += len(coauthor_texts)
            return res

        t0 = time.perf_counter()
        with _patch_attr(data_mod, "_python_signature_ngrams_batch", _timed_batch):
            with _patch_attr(data_mod, "tqdm", _tqdm_wrapper(data_mod.tqdm)):
                ds.preprocess_signatures()
        work = time.perf_counter() - t0
        if show_breakdown:
            frac = (ngram_time / work * 100) if work > 0 else 0.0
            print(
                f"    breakdown: ngram_batch={_fmt(ngram_time)} ({frac:.1f}%) calls={ngram_calls} items={ngram_items}",
                flush=True,
            )
        return 0.0, work, len(signatures)

    def _run_pool(use_threads: bool) -> tuple[float, float, int]:
        signatures = dict(base_signatures)
        ds = _make_ds(signatures)

        t_pool0 = time.perf_counter()
        pool = UniversalPool(processes=n_jobs, use_threads=use_threads)
        pool_create = time.perf_counter() - t_pool0

        ngram_time = 0.0
        ngram_calls = 0
        ngram_items = 0

        def _batch_parallel(coauthor_texts: list[str], affiliation_texts: list[str]):
            nonlocal ngram_time, ngram_calls, ngram_items
            t_ng = time.perf_counter()
            pairs = list(zip(coauthor_texts, affiliation_texts, strict=True))
            results = list(pool.imap(_signature_ngrams_one, pairs, ngram_chunk_size))
            coauthor_counters = []
            affiliation_counters = []
            for co_ctr, aff_ctr in results:
                coauthor_counters.append(co_ctr)
                affiliation_counters.append(aff_ctr)
            ngram_time += time.perf_counter() - t_ng
            ngram_calls += 1
            ngram_items += len(coauthor_texts)
            return coauthor_counters, affiliation_counters

        with pool:
            with _patch_attr(data_mod, "_python_signature_ngrams_batch", _batch_parallel):
                with _patch_attr(data_mod, "tqdm", _tqdm_wrapper(data_mod.tqdm)):
                    t1 = time.perf_counter()
                    ds.preprocess_signatures()
                    work = time.perf_counter() - t1

        if show_breakdown:
            frac = (ngram_time / work * 100) if work > 0 else 0.0
            print(
                f"    breakdown: ngram_batch={_fmt(ngram_time)} ({frac:.1f}%) calls={ngram_calls} items={ngram_items}",
                flush=True,
            )
        return pool_create, work, len(signatures)

    def run_once(use_threads: bool | None) -> tuple[float, float, int]:
        if use_threads is None:
            return _run_serial()
        return _run_pool(use_threads)

    _bench_phase(
        phase_name="Signatures: ANDData.preprocess_signatures (ngram backend swap)",
        run_once=run_once,
        n_jobs=n_jobs,
        rounds=rounds,
        configs=configs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preprocessing phases across OS / pool modes.")
    parser.add_argument("--dataset", default="kisti", help="Dataset name (default: kisti)")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data"),
        help="Directory containing dataset subdirectories",
    )
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds per config (default: 1)")
    parser.add_argument(
        "--limit-signatures",
        type=int,
        default=5_000,
        help="Limit signatures (and thus papers) for quicker runs; 0 = full dataset (default: 5000)",
    )
    parser.add_argument("--chunk-size-paper1", type=int, default=1000, help="Paper stage 1 imap chunk size")
    parser.add_argument(
        "--signature-ngram-chunk-size",
        type=int,
        default=1000,
        help="Chunk size for signature ngram imap (threads/processes backends only)",
    )
    parser.add_argument(
        "--signature-breakdown",
        action="store_true",
        help="Print a time breakdown for signature preprocessing n-gram computation per config",
    )
    parser.add_argument(
        "--backend",
        choices=["python", "rust", "auto"],
        default="python",
        help="Set S2AND_BACKEND for this run (default: python)",
    )
    parser.add_argument(
        "--skip-signatures",
        action="store_true",
        help="Skip signatures preprocessing benchmarking",
    )
    parser.add_argument(
        "--skip-paper-benchmark",
        action="store_true",
        help="Preprocess papers once serially without benchmarking that phase",
    )
    parser.add_argument(
        "--signature-config",
        choices=["all", "serial", "threads", "processes"],
        default="all",
        help="Signature-preprocessing configuration(s) to benchmark",
    )
    parser.add_argument(
        "--name-counts-index",
        help="Optional canonical name_counts_index path to include in signature preprocessing",
    )
    args = parser.parse_args()

    os.environ["S2AND_BACKEND"] = args.backend

    print(f"Platform: {platform.system()} ({platform.platform()})")
    print(f"Python:   {sys.version}")
    print(f"Backend:  {args.backend}")
    print(f"Dataset:  {args.dataset}")
    print(f"Data dir: {args.data_dir}")
    print(f"Workers:  {args.n_jobs}    Rounds: {args.rounds}")
    print(f"Limit:    signatures={args.limit_signatures} (0 = full)")
    print(f"Counts:   {args.name_counts_index or 'disabled'}")
    print(flush=True)

    name_counts_index = None
    if args.name_counts_index:
        from s2and.name_counts_index import NameCountsIndex

        name_counts_index = NameCountsIndex.open(args.name_counts_index)

    print(f"Loading dataset '{args.dataset}'...")
    raw_sigs, raw_papers = load_dataset(
        data_dir=args.data_dir,
        dataset=args.dataset,
        limit_signatures=args.limit_signatures,
    )
    print(f"  raw: {len(raw_papers):,} papers | {len(raw_sigs):,} signatures")

    base_signatures, base_papers = build_namedtuples(
        raw_signatures=raw_sigs,
        raw_papers=raw_papers,
        use_orcid_id=True,
    )
    paper_items = list(base_papers.items())
    print(f"  namedtuples: {len(base_papers):,} papers | {len(base_signatures):,} signatures")
    print(flush=True)

    # --- Papers ---
    from s2and.data import preprocess_paper_1

    paper1_func = partial(preprocess_paper_1, preprocess=True)

    need_papers_preprocessed = not args.skip_signatures
    papers_preprocessed: dict[str, Any] | None = None

    def run_paper1(use_threads: bool | None) -> tuple[float, float, int]:
        nonlocal papers_preprocessed
        if use_threads is None and need_papers_preprocessed and papers_preprocessed is None:
            t0 = time.perf_counter()
            out: dict[str, Any] = {}
            for item in paper_items:
                k, v = paper1_func(item)
                out[k] = v
            elapsed = time.perf_counter() - t0
            papers_preprocessed = out
            return 0.0, elapsed, len(out)

        return _run_paper_stage(
            label="papers",
            items=paper_items,
            func=paper1_func,
            n_jobs=args.n_jobs,
            use_threads=use_threads,
            chunk_size=args.chunk_size_paper1,
        )

    if args.skip_paper_benchmark:
        papers_preprocessed = {key: value for key, value in map(paper1_func, paper_items)}
    else:
        _bench_phase(
            phase_name="Papers: preprocess_paper_1",
            run_once=run_paper1,
            n_jobs=args.n_jobs,
            rounds=args.rounds,
            configs=[
                (f"threads x{args.n_jobs}", True),
                (f"processes x{args.n_jobs}", False),
                ("serial", None),
            ],
        )

    if need_papers_preprocessed and papers_preprocessed is None:
        raise RuntimeError("Expected papers_preprocessed to be materialized during serial papers run.")

    # --- Signatures ---
    if not args.skip_signatures:
        signature_configs: Sequence[tuple[str, bool | None]] | None = {
            "all": None,
            "serial": [("serial", None)],
            "threads": [(f"threads x{args.n_jobs}", True)],
            "processes": [(f"processes x{args.n_jobs}", False)],
        }[args.signature_config]
        _bench_signatures_preprocess(
            base_signatures=base_signatures,
            papers=papers_preprocessed or {},
            n_jobs=args.n_jobs,
            rounds=args.rounds,
            ngram_chunk_size=args.signature_ngram_chunk_size,
            name_counts_index=name_counts_index,
            show_breakdown=args.signature_breakdown,
            configs=signature_configs,
        )


if __name__ == "__main__":
    main()
