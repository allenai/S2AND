"""Benchmark preprocessing phases (papers 1/2, papers 2/2, signatures).

Focus: compare serial vs threads vs processes via UniversalPool across OSes.

Phases:
  1) Papers 1/2: `preprocess_paper_1` across papers
  2) Papers 2/2: `preprocess_paper_2` reference-details computation
  3) Signatures: `ANDData.preprocess_signatures` with swappable Python ngram backend

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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, TypeVar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

T = TypeVar("T")


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


def load_dataset(*, dataset: str, limit_signatures: int) -> tuple[dict[str, Any], dict[str, Any]]:
    sig_path = os.path.join(DATA_DIR, dataset, f"{dataset}_signatures.json")
    paper_path = os.path.join(DATA_DIR, dataset, f"{dataset}_papers.json")

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
            author_info_first_normalized=None,
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
            author_info_given_block=author_info.get("given_block", None),
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
            references=paper.get("references"),
            has_abstract=bool(paper.get("abstract", "") or paper.get("has_abstract", False)),
            predicted_language=None,
            is_english=None,
            is_reliable=None,
            title_ngrams_words=None,
            title_ngrams_chars=None,
            venue_ngrams=None,
            journal_ngrams=None,
            reference_details=None,
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
    coauthor_counter = get_text_ngrams(coauthor_text, stopwords=None, use_bigrams=True) if coauthor_text else Counter()
    affiliation_counter = get_text_ngrams_words(affiliation_text, stopwords=set()) if affiliation_text else Counter()
    return coauthor_counter, affiliation_counter


def _run_paper_stage(
    *,
    label: str,
    items: list[tuple[str, Any]],
    func: Callable[[tuple[str, Any]], tuple[str, Any]],
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
        for _key, _value in pool.imap(func, items, chunk_size):
            out_count += 1
    work = time.perf_counter() - t1
    _ = label  # keep param for symmetry/readability
    return pool_create, work, out_count


def _build_reference_inputs(*, papers: dict[str, Any]) -> list[tuple[str, Any, list[Any]]]:
    from s2and.data import MiniPaper

    input_2: list[tuple[str, Any, list[Any]]] = []
    for key, value in papers.items():
        refs = value.references or []
        reference_papers = [
            MiniPaper(
                title=p.title,
                venue=p.venue,
                journal_name=p.journal_name,
                authors=[a.author_name for a in p.authors],
            )
            for p in (papers.get(str(rid)) for rid in refs)
            if p is not None
        ]
        input_2.append((key, value, reference_papers))
    return input_2


def _bench_phase(
    *,
    phase_name: str,
    run_once: Callable[[bool | None], tuple[float, float, int]],
    n_jobs: int,
    rounds: int,
    configs: list[tuple[str, bool | None]] | None = None,
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
    show_breakdown: bool = False,
) -> None:
    import s2and.data as data_mod
    from s2and.data import ANDData
    from s2and.mp import UniversalPool
    from s2and.runtime import build_runtime_context
    from s2and.rust_lifecycle import PYTHON_ONLY_POLICY

    def _make_ds(signatures: dict[str, Any]) -> Any:
        ds = ANDData.__new__(ANDData)
        ds.runtime_context = build_runtime_context("bench_preprocess_signatures", emit_startup_warning=False)
        ds.rust_lifecycle_policy = PYTHON_ONLY_POLICY
        ds.preprocess = True
        ds.signatures = signatures
        ds.papers = papers
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
                ds.preprocess_signatures(load_name_counts=False)
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
                    ds.preprocess_signatures(load_name_counts=False)
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preprocessing phases across OS / pool modes.")
    parser.add_argument("--dataset", default="kisti", help="Dataset name (default: kisti)")
    parser.add_argument("--n-jobs", type=int, default=8, help="Number of workers (default: 8)")
    parser.add_argument("--rounds", type=int, default=1, help="Rounds per config (default: 1)")
    parser.add_argument(
        "--limit-signatures",
        type=int,
        default=5_000,
        help="Limit signatures (and thus papers) for quicker runs; 0 = full dataset (default: 5000)",
    )
    parser.add_argument("--chunk-size-paper1", type=int, default=1000, help="Paper stage 1 imap chunk size")
    parser.add_argument("--chunk-size-paper2", type=int, default=100, help="Paper stage 2 imap chunk size")
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
        "--skip-paper2",
        action="store_true",
        help="Skip papers 2/2 (reference-details) benchmarking",
    )
    parser.add_argument(
        "--skip-signatures",
        action="store_true",
        help="Skip signatures preprocessing benchmarking",
    )
    args = parser.parse_args()

    os.environ["S2AND_BACKEND"] = args.backend

    print(f"Platform: {platform.system()} ({platform.platform()})")
    print(f"Python:   {sys.version}")
    print(f"Backend:  {args.backend}")
    print(f"Dataset:  {args.dataset}")
    print(f"Workers:  {args.n_jobs}    Rounds: {args.rounds}")
    print(f"Limit:    signatures={args.limit_signatures} (0 = full)")
    print(flush=True)

    print(f"Loading dataset '{args.dataset}'...")
    raw_sigs, raw_papers = load_dataset(dataset=args.dataset, limit_signatures=args.limit_signatures)
    print(f"  raw: {len(raw_papers):,} papers | {len(raw_sigs):,} signatures")

    base_signatures, base_papers = build_namedtuples(
        raw_signatures=raw_sigs,
        raw_papers=raw_papers,
        use_orcid_id=True,
    )
    paper_items = list(base_papers.items())
    print(f"  namedtuples: {len(base_papers):,} papers | {len(base_signatures):,} signatures")
    print(flush=True)

    # --- Papers 1/2 ---
    from s2and.data import preprocess_paper_1, preprocess_paper_2

    paper1_func = partial(preprocess_paper_1, preprocess=True)

    need_papers_preprocessed = not args.skip_paper2 or not args.skip_signatures
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
            label="papers 1/2",
            items=paper_items,
            func=paper1_func,
            n_jobs=args.n_jobs,
            use_threads=use_threads,
            chunk_size=args.chunk_size_paper1,
        )

    _bench_phase(
        phase_name="Papers 1/2: preprocess_paper_1",
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
        raise RuntimeError("Expected papers_preprocessed to be materialized during serial papers 1/2 run.")

    # --- Papers 2/2 (reference_details) ---
    if not args.skip_paper2:
        print()
        print("Building papers 2/2 input (reference-details)...", flush=True)
        t_build0 = time.perf_counter()
        input_2 = _build_reference_inputs(papers=papers_preprocessed or {})
        build_input_2 = time.perf_counter() - t_build0
        print(f"  input_2: {_fmt(build_input_2)} ({len(input_2):,} items)", flush=True)

        def run_paper2(use_threads: bool | None) -> tuple[float, float, int]:
            return _run_paper_stage(
                label="papers 2/2",
                items=input_2,
                func=preprocess_paper_2,
                n_jobs=args.n_jobs,
                use_threads=use_threads,
                chunk_size=args.chunk_size_paper2,
            )

        _bench_phase(
            phase_name=f"Papers 2/2: preprocess_paper_2 (reference_details) [build input: {_fmt(build_input_2)}]",
            run_once=run_paper2,
            n_jobs=args.n_jobs,
            rounds=args.rounds,
        )

    # --- Signatures ---
    if not args.skip_signatures:
        _bench_signatures_preprocess(
            base_signatures=base_signatures,
            papers=papers_preprocessed or {},
            n_jobs=args.n_jobs,
            rounds=args.rounds,
            ngram_chunk_size=args.signature_ngram_chunk_size,
            show_breakdown=args.signature_breakdown,
        )


if __name__ == "__main__":
    main()
