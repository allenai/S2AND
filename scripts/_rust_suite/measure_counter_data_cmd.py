"""
Measure how much memory the CounterData fields (title_ngrams_chars,
title_ngrams_words, venue_ngrams, journal_ngrams) contribute to the Rust
featurizer for a given dataset, without modifying any production code.

Method:
  1. Build ANDData for the dataset (same way as profile_transfer_mini).
  2. Build RustFeaturizer from original ANDData -> save to temp file -> record size.
  3. Zero out all 4 CounterData fields on every paper -> build RustFeaturizer again ->
     save to temp file -> record size.
  4. Load each from disk and record RSS delta.
  5. Report CounterData disk and in-memory contribution.

Usage:
  uv run python scripts/rust_suite.py measure-counter-data --dataset kisti
"""

import argparse
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.common import PROJECT_ROOT  # noqa: E402


def _rss_gb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1024**3)


def _import_rust_module():
    import s2and_rust

    return s2and_rust


def _build_anddata(dataset_name: str, data_dir: str, n_jobs: int = 1):
    from s2and.data import ANDData

    dataset_root = os.path.join(data_dir, dataset_name)
    sig_path = None
    papers_path = None
    specter_path = None
    clusters_path = None
    for f in os.listdir(dataset_root):
        fl = f.lower()
        if "signature" in fl and fl.endswith(".json"):
            sig_path = os.path.join(dataset_root, f)
        elif "paper" in fl and fl.endswith(".json"):
            papers_path = os.path.join(dataset_root, f)
        elif "specter" in fl or "specter2" in fl:
            specter_path = os.path.join(dataset_root, f)
        elif "cluster" in fl and fl.endswith(".json"):
            clusters_path = os.path.join(dataset_root, f)
    if sig_path is None or papers_path is None:
        raise FileNotFoundError(f"Could not find signatures/papers in {dataset_root}: {os.listdir(dataset_root)}")
    anddata = ANDData(
        signatures=sig_path,
        papers=papers_path,
        name=dataset_name,
        specter_embeddings=specter_path,
        clusters=clusters_path,
        n_jobs=n_jobs,
    )
    return anddata


def _strip_counter_data(anddata):
    """Return new papers dict with all 4 CounterData fields zeroed out."""
    empty = Counter()
    new_papers = {}
    for pid, paper in anddata.papers.items():
        new_papers[pid] = paper._replace(
            title_ngrams_chars=empty,
            title_ngrams_words=empty,
            venue_ngrams=empty,
            journal_ngrams=empty,
        )
    return new_papers


def _build_featurizer(anddata) -> Any:
    s2and_rust = _import_rust_module()
    return s2and_rust.RustFeaturizer.from_dataset(anddata)


def _load_featurizer(path: str) -> Any:
    s2and_rust = _import_rust_module()
    return s2and_rust.RustFeaturizer.load(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kisti")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    data_dir = args.data_dir
    dataset_name = args.dataset

    print(f"Building ANDData for {dataset_name}...")
    t0 = time.perf_counter()
    anddata = _build_anddata(dataset_name, data_dir, n_jobs=args.n_jobs)
    print(f"  ANDData built in {time.perf_counter()-t0:.1f}s  RSS={_rss_gb():.3f} GB")
    print(f"  papers={len(anddata.papers)}  signatures={len(anddata.signatures)}")

    # ---- Sample CounterData stats from a few papers ----
    sample_sizes = {"title_ngrams_chars": [], "title_ngrams_words": [], "venue_ngrams": [], "journal_ngrams": []}
    for p in list(anddata.papers.values())[:5000]:
        for f in sample_sizes:
            v = getattr(p, f, None)
            sample_sizes[f].append(len(v) if v else 0)
    print("\nCounterData field stats (first 5000 papers):")
    for f, sizes in sample_sizes.items():
        nonzero = sum(1 for s in sizes if s > 0)
        avg = sum(sizes) / len(sizes) if sizes else 0
        print(f"  {f}: avg_entries={avg:.1f}  papers_with_data={nonzero}/{len(sizes)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        path_full = os.path.join(tmpdir, "full.bin")
        path_stripped = os.path.join(tmpdir, "stripped.bin")

        # ---- Build FULL featurizer ----
        print("\nBuilding full featurizer (with CounterData)...")
        rss_before_full = _rss_gb()
        t0 = time.perf_counter()
        feat_full = _build_featurizer(anddata)
        rss_after_build_full = _rss_gb()
        feat_full.save(path_full)
        size_full = os.path.getsize(path_full)
        del feat_full
        rss_after_del_full = _rss_gb()
        print(f"  Build time: {time.perf_counter()-t0:.1f}s")
        print(
            "  RSS: "
            f"{rss_before_full:.3f} -> {rss_after_build_full:.3f} GB  "
            f"(delta={rss_after_build_full - rss_before_full:+.3f} GB)"
        )
        print(f"  Disk size: {size_full/1e6:.1f} MB")
        print(f"  RSS after del: {rss_after_del_full:.3f} GB")

        # ---- Build STRIPPED featurizer ----
        print("\nBuilding stripped featurizer (CounterData zeroed)...")
        orig_papers = anddata.papers
        anddata.papers = _strip_counter_data(anddata)
        rss_before_stripped = _rss_gb()
        t0 = time.perf_counter()
        feat_stripped = _build_featurizer(anddata)
        rss_after_build_stripped = _rss_gb()
        feat_stripped.save(path_stripped)
        size_stripped = os.path.getsize(path_stripped)
        del feat_stripped
        anddata.papers = orig_papers
        rss_after_del_stripped = _rss_gb()
        print(f"  Build time: {time.perf_counter()-t0:.1f}s")
        print(
            "  RSS: "
            f"{rss_before_stripped:.3f} -> {rss_after_build_stripped:.3f} GB  "
            f"(delta={rss_after_build_stripped - rss_before_stripped:+.3f} GB)"
        )
        print(f"  Disk size: {size_stripped/1e6:.1f} MB")
        print(f"  RSS after del: {rss_after_del_stripped:.3f} GB")

        # ---- Load from disk and compare RSS ----
        print("\nLoading full featurizer from disk...")
        rss_before_load_full = _rss_gb()
        feat_full_loaded = _load_featurizer(path_full)
        rss_after_load_full = _rss_gb()
        del feat_full_loaded
        load_delta_full = rss_after_load_full - rss_before_load_full

        print("\nLoading stripped featurizer from disk...")
        rss_before_load_stripped = _rss_gb()
        feat_stripped_loaded = _load_featurizer(path_stripped)
        rss_after_load_stripped = _rss_gb()
        del feat_stripped_loaded
        load_delta_stripped = rss_after_load_stripped - rss_before_load_stripped

        # ---- Summary ----
        disk_counter_data_mb = (size_full - size_stripped) / 1e6
        mem_counter_data_gb = load_delta_full - load_delta_stripped
        expansion = load_delta_full / (size_full / 1e9) if size_full > 0 else 0
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Disk size full:        {size_full/1e6:.1f} MB")
        print(f"Disk size stripped:    {size_stripped/1e6:.1f} MB")
        print(f"CounterData on disk:   {disk_counter_data_mb:.1f} MB")
        print("")
        print(f"Load RSS delta full:   {load_delta_full*1024:.0f} MB")
        print(f"Load RSS delta stripped: {load_delta_stripped*1024:.0f} MB")
        print(f"CounterData in memory: {mem_counter_data_gb*1024:.0f} MB")
        print(f"Disk->memory expansion (full): {expansion:.2f}x")
        print("")
        print("HYPOTHESIS: ~200-300 MB in-memory savings from compact CounterData")
        actual = mem_counter_data_gb * 1024
        if actual >= 150:
            print(f"RESULT: hypothesis SUPPORTED  (measured {actual:.0f} MB, >= 150 MB threshold)")
        else:
            print(f"RESULT: hypothesis NOT SUPPORTED  (measured {actual:.0f} MB, < 150 MB threshold)")


if __name__ == "__main__":
    main()
