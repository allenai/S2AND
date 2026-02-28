#!/usr/bin/env python
"""Profile Python vs Rust on the single largest block across all datasets.

This script:
1. Scans all datasets to find the single largest block (by signature count)
2. Loads the full dataset containing that block
3. Runs predict() on just that one block, for both Python and Rust backends
4. Measures: ANDData build time, prediction time (featurize + cluster), peak RSS
5. Captures cProfile hotspot analysis
6. Outputs JSON results and cProfile text files

Usage:
    # Auto-detect largest block across all datasets, compare Python vs Rust:
    uv run python scripts/rust_suite.py largest-block --mode compare

    # Specify dataset and block manually:
    uv run python scripts/rust_suite.py largest-block --mode compare \
        --dataset aminer --block "j wang"

    # Single backend run:
    uv run python scripts/rust_suite.py largest-block --mode single \
        --backend rust --dataset aminer --block "j wang"

    # Limit block size (use first N signatures from the block):
    uv run python scripts/rust_suite.py largest-block --mode compare \
        --max-block-size 1000

    # Add block-level quality metrics (requires clusters.json):
    uv run python scripts/rust_suite.py largest-block --mode compare \
        --dataset aminer --block "j wang" --quality-check

    # Sample constraint parity (python vs rust constraints on the same dataset instance):
    uv run python scripts/rust_suite.py largest-block --mode compare \
        --dataset aminer --block "j wang" --constraint-sample 50000 --constraint-sample-seed 43
"""

from __future__ import annotations

import argparse
import cProfile
import datetime
import hashlib
import io
import json
import os
import pstats
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts._rust_suite.common import (
        ProcessTreeRSSMonitor,
        build_run_metadata,
        collect_rust_extension_identity,
    )
else:
    try:
        from _rust_suite.common import ProcessTreeRSSMonitor, build_run_metadata, collect_rust_extension_identity
    except ModuleNotFoundError:
        _SCRIPTS_DIR = Path(__file__).resolve().parents[1]
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from _rust_suite.common import ProcessTreeRSSMonitor, build_run_metadata, collect_rust_extension_identity

RESULT_JSON_START = "===S2AND_LARGEST_BLOCK_RESULT_START==="
RESULT_JSON_END = "===S2AND_LARGEST_BLOCK_RESULT_END==="

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL_PATH = str(DATA_DIR / "production_model_v1.1.pickle")

# All known dataset directories (top-level under data/)
DATASET_CANDIDATES = [
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


def _cluster_membership_digest(cluster_to_signatures: dict[str, list[str]]) -> str:
    sorted_clusters = sorted([sorted(signatures) for signatures in cluster_to_signatures.values() if signatures])
    payload = json.dumps(sorted_clusters, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signature_to_cluster_fingerprint_map(cluster_to_signatures: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for signatures in cluster_to_signatures.values():
        members = sorted(signatures)
        fingerprint = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()
        for signature_id in members:
            mapping[signature_id] = fingerprint
    return mapping


def _pair_count_with_singleton_fix(cluster_size: int) -> int:
    if cluster_size <= 0:
        return 0
    if cluster_size == 1:
        return 1
    return cluster_size * (cluster_size - 1) // 2


def _pairwise_precision_recall_fscore_with_singleton_fix(
    true_clusters: dict[str, list[str]],
    pred_clusters: dict[str, list[str]],
) -> tuple[float, float, float]:
    """Efficient pairwise P/R/F1 matching s2and.eval.cluster_precision_recall_fscore semantics.

    s2and's implementation includes a singleton fix: a singleton cluster contributes one self-pair (s, s).
    """

    true_members = {s for members in true_clusters.values() for s in members}
    pred_members = {s for members in pred_clusters.values() for s in members}
    if true_members != pred_members:
        raise ValueError("Predictions do not cover all the signatures!")

    signature_to_true: dict[str, str] = {}
    true_singletons: set[str] = set()
    for cluster_id, members in true_clusters.items():
        if len(members) == 1:
            true_singletons.add(members[0])
        for signature_id in members:
            signature_to_true[signature_id] = cluster_id

    signature_to_pred: dict[str, str] = {}
    pred_singletons: set[str] = set()
    for cluster_id, members in pred_clusters.items():
        if len(members) == 1:
            pred_singletons.add(members[0])
        for signature_id in members:
            signature_to_pred[signature_id] = cluster_id

    # Denominators: number of positive pairs (including singleton self-pairs).
    true_positive_pairs = sum(_pair_count_with_singleton_fix(len(members)) for members in true_clusters.values())
    pred_positive_pairs = sum(_pair_count_with_singleton_fix(len(members)) for members in pred_clusters.values())

    # TP: sum over intersections of C(n, 2) + matching singleton self-pairs.
    intersection_counts: dict[tuple[str, str], int] = {}
    for signature_id in true_members:
        key = (signature_to_true[signature_id], signature_to_pred[signature_id])
        intersection_counts[key] = intersection_counts.get(key, 0) + 1

    true_positive = sum((count * (count - 1) // 2) for count in intersection_counts.values() if count > 1)
    true_positive += len(true_singletons.intersection(pred_singletons))

    precision = true_positive / pred_positive_pairs if pred_positive_pairs > 0 else 0.0
    recall = true_positive / true_positive_pairs if true_positive_pairs > 0 else 0.0
    f1 = 0.0 if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)
    return round(precision, 3), round(recall, 3), round(f1, 3)


# ---------------------------------------------------------------------------
# Block scanning
# ---------------------------------------------------------------------------


def _find_signatures_file(dataset_name: str) -> Path | None:
    """Find the signatures JSON file for a dataset."""
    p = DATA_DIR / dataset_name / f"{dataset_name}_signatures.json"
    if p.exists():
        return p
    return None


def _scan_blocks(sig_path: Path) -> Counter:
    """Load signatures JSON and count block sizes."""
    with open(sig_path, encoding="utf-8") as f:
        sigs = json.load(f)
    block_counts: Counter = Counter()
    for _sig_id, sig in sigs.items():
        block = sig.get("author_info", {}).get("block", "")
        block_counts[block] += 1
    return block_counts


def find_largest_block() -> tuple[str, str, int]:
    """Scan all datasets and return (dataset_name, block_key, block_size)."""
    best_dataset = ""
    best_block = ""
    best_count = 0

    for name in DATASET_CANDIDATES:
        sig_path = _find_signatures_file(name)
        if sig_path is None:
            continue
        bc = _scan_blocks(sig_path)
        if not bc:
            continue
        top_block, top_count = bc.most_common(1)[0]
        print(f"  {name}: largest block = {top_block!r} ({top_count:,} sigs)")
        if top_count > best_count:
            best_count = top_count
            best_block = top_block
            best_dataset = name

    return best_dataset, best_block, best_count


# ---------------------------------------------------------------------------
# Dataset path helpers
# ---------------------------------------------------------------------------


def _resolve_path(path_like: str) -> str:
    path = Path(path_like)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _build_data_paths(dataset_name: str, data_root: str) -> dict[str, str]:
    """Build paths to dataset files."""
    dataset_root = Path(_resolve_path(data_root)) / dataset_name
    return {
        "signatures": str((dataset_root / f"{dataset_name}_signatures.json").resolve()),
        "papers": str((dataset_root / f"{dataset_name}_papers.json").resolve()),
        "clusters": str((dataset_root / f"{dataset_name}_clusters.json").resolve()),
        "specter": str((dataset_root / f"{dataset_name}_specter.pickle").resolve()),
    }


def _check_paths(paths: dict[str, str], require_clusters: bool = False) -> None:
    """Validate that required paths exist."""
    required = ["signatures", "papers"]
    if require_clusters:
        required.append("clusters")
    for key in required:
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(f"Missing {key}: {paths[key]}")


def _write_profile_output(profiler: cProfile.Profile, output_path: str, elapsed_seconds: float) -> None:
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime")
    stats.print_stats(60)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_stream.getvalue())
        f.write(f"\nTotal runtime (predict only): {elapsed_seconds:.3f}s\n")


# ---------------------------------------------------------------------------
# Single-run logic
# ---------------------------------------------------------------------------


def _run_single(
    backend: str,
    dataset_name: str,
    block_key: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    data_root: str = str(DATA_DIR),
    max_block_size: int = 0,
    run_label: str = "",
    quality_check: bool = False,
    constraint_sample: int = 0,
    constraint_sample_seed: int = 42,
    emit_signature_map: bool = False,
    require_rust_release: bool = False,
) -> dict[str, Any]:
    """Run prediction on a single block and return metrics."""

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ["S2AND_BACKEND"] = backend
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")

    import s2and.model as model_module
    from s2and.data import ANDData
    from s2and.serialization import load_pickle_with_verified_label_encoder_compat

    resolved_model_path = _resolve_path(model_path)
    resolved_data_root = _resolve_path(data_root)
    paths = _build_data_paths(dataset_name, resolved_data_root)
    _check_paths(paths, require_clusters=False)

    # Load model
    model_artifact = load_pickle_with_verified_label_encoder_compat(resolved_model_path)
    clusterer = model_artifact["clusterer"]
    model_module._ensure_lightgbm_fitted(clusterer.classifier)
    model_module._ensure_lightgbm_fitted(clusterer.nameless_classifier)
    clusterer.use_cache = False
    clusterer.n_jobs = n_jobs

    # Check if we have clusters (needed for train mode / cluster_eval)
    has_clusters = os.path.exists(paths["clusters"])
    has_specter = os.path.exists(paths["specter"])
    rust_extension_identity: dict[str, Any] | None = None
    if backend == "rust":
        rust_extension_identity = collect_rust_extension_identity(
            require_release=bool(require_rust_release),
            fail_if_unavailable=False,
        )

    print(f"[{backend}] Building ANDData for {dataset_name}...")
    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        anddata_start = time.perf_counter()
        anddata = ANDData(
            signatures=paths["signatures"],
            papers=paths["papers"],
            name=dataset_name,
            mode="inference",
            specter_embeddings=paths["specter"] if has_specter else None,
            clusters=paths["clusters"] if has_clusters else None,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=100000,
            val_pairs_size=10000,
            test_pairs_size=10000,
            n_jobs=n_jobs,
            load_name_counts=True,
            preprocess=True,
            random_seed=42,
            name_tuples="filtered",
            use_orcid_id=True,
            use_sinonym_overwrite=True,
        )
        anddata_build_seconds = time.perf_counter() - anddata_start

        # Get the target block
        all_blocks = anddata.get_blocks()
        if block_key not in all_blocks:
            raise ValueError(
                f"Block {block_key!r} not found in {dataset_name}. "
                f"Available blocks ({len(all_blocks)}): {sorted(all_blocks.keys())[:10]}..."
            )

        block_sigs = all_blocks[block_key]
        original_block_size = len(block_sigs)

        # Optionally limit block size
        if max_block_size > 0 and len(block_sigs) > max_block_size:
            block_sigs = sorted(block_sigs)[:max_block_size]
            print(f"[{backend}] Trimmed block from {original_block_size} to {max_block_size} signatures")

        block_size = len(block_sigs)
        num_pairs = block_size * (block_size - 1) // 2
        print(f"[{backend}] Block {block_key!r}: {block_size} signatures, {num_pairs:,} pairs")
        print(f"[{backend}] ANDData built in {anddata_build_seconds:.1f}s")

        # Warm Rust featurizer if using Rust backend
        warm_rust_featurizer_seconds = 0.0
        if backend == "rust":
            from s2and.feature_port import warm_rust_featurizer

            print(f"[{backend}] Warming Rust featurizer...")
            warm_start = time.perf_counter()
            warm_rust_featurizer(anddata)
            warm_rust_featurizer_seconds = time.perf_counter() - warm_start
            print(f"[{backend}] Rust featurizer warm in {warm_rust_featurizer_seconds:.1f}s")

        # Run prediction with cProfile on just the target block
        single_block_dict = {block_key: block_sigs}

        print(f"[{backend}] Running predict on block {block_key!r} ({block_size} sigs, {num_pairs:,} pairs)...")
        profiler = cProfile.Profile()
        predict_start = time.perf_counter()
        profiler.enable()
        pred_clusters, _ = clusterer.predict_helper(
            single_block_dict,
            anddata,
            dists=None,
            cluster_model_params=None,
            partial_supervision={},
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=False,
        )
        profiler.disable()
        predict_seconds = time.perf_counter() - predict_start

    total_seconds = time.perf_counter() - total_start

    _write_profile_output(profiler, profile_output_path, predict_seconds)

    # Cluster statistics
    cluster_sizes = sorted([len(sigs) for sigs in pred_clusters.values()], reverse=True)
    num_clusters = len(pred_clusters)
    assigned_sigs = sum(cluster_sizes)
    cluster_membership_digest = _cluster_membership_digest(pred_clusters)
    signature_to_cluster_fingerprint = (
        _signature_to_cluster_fingerprint_map(pred_clusters) if emit_signature_map else None
    )

    quality_metrics: dict[str, Any] | None = None
    if quality_check:
        if anddata.clusters is None:
            raise FileNotFoundError(
                "Quality check requested but clusters.json was not loaded. "
                "Ensure <dataset>_clusters.json exists and re-run with --quality-check."
            )
        block_sig_set = set(block_sigs)
        signature_to_true_cluster_id: dict[str, str] = {}
        existing_map = getattr(anddata, "signature_to_cluster_id", None)
        if isinstance(existing_map, dict) and len(existing_map) > 0:
            signature_to_true_cluster_id = {sig: existing_map[sig] for sig in block_sigs}
        else:
            for cluster_id, cluster_info in anddata.clusters.items():
                for signature_id in cluster_info.get("signature_ids", []):
                    if signature_id in block_sig_set:
                        signature_to_true_cluster_id[signature_id] = str(cluster_id)
                if len(signature_to_true_cluster_id) == len(block_sig_set):
                    break
        missing = [sig for sig in block_sigs if sig not in signature_to_true_cluster_id]
        if missing:
            raise RuntimeError(
                f"Quality check failed: {len(missing)}/{len(block_sigs)} signatures missing from clusters.json"
            )

        true_clusters: dict[str, list[str]] = {}
        for signature_id, true_cluster_id in signature_to_true_cluster_id.items():
            true_clusters.setdefault(true_cluster_id, []).append(signature_id)
        true_cluster_sizes = sorted([len(sigs) for sigs in true_clusters.values()], reverse=True)

        from s2and.eval import b3_precision_recall_fscore

        b3_p, b3_r, b3_f1, _, _, _ = b3_precision_recall_fscore(true_clusters, pred_clusters)
        pw_p, pw_r, pw_f1 = _pairwise_precision_recall_fscore_with_singleton_fix(true_clusters, pred_clusters)
        quality_metrics = {
            "b3": {"precision": float(b3_p), "recall": float(b3_r), "f1": float(b3_f1)},
            "pairwise": {"precision": pw_p, "recall": pw_r, "f1": pw_f1},
            "true_num_clusters": int(len(true_clusters)),
            "true_cluster_sizes_top10": [int(x) for x in true_cluster_sizes[:10]],
        }

    constraint_parity: dict[str, Any] | None = None
    if constraint_sample > 0:
        if len(block_sigs) < 2:
            constraint_parity = {
                "sample_pairs_requested": int(constraint_sample),
                "sample_pairs_effective": 0,
                "mismatch_count": 0,
                "mismatch_rate": 0.0,
                "note": "Block too small for pair sampling.",
            }
        else:
            try:
                from s2and.feature_port import _get_rust_featurizer, get_constraint_rust
            except Exception as exc:  # pragma: no cover - rust extension optional
                raise RuntimeError(
                    "Constraint parity sampling requires the Rust extension. "
                    "Build it (maturin develop) or run with --constraint-sample 0."
                ) from exc

            rng = random.Random(int(constraint_sample_seed))
            n = len(block_sigs)
            max_pairs = n * (n - 1) // 2
            sample_pairs = min(int(constraint_sample), int(max_pairs))

            rust_featurizer = _get_rust_featurizer(anddata, use_cache=False)

            mismatch_count = 0
            mismatch_python_none = 0
            mismatch_rust_none = 0
            mismatch_value_diff = 0
            python_constraint_hits = 0
            rust_constraint_hits = 0
            examples: list[dict[str, Any]] = []

            dont_merge = bool(getattr(clusterer, "dont_merge_cluster_seeds", True))
            for _ in range(sample_pairs):
                i = rng.randrange(n)
                j = rng.randrange(n - 1)
                if j >= i:
                    j += 1
                if i > j:
                    i, j = j, i
                sig_a = block_sigs[i]
                sig_b = block_sigs[j]

                py_val = anddata.get_constraint(
                    sig_a,
                    sig_b,
                    dont_merge_cluster_seeds=dont_merge,
                    incremental_dont_use_cluster_seeds=False,
                )
                rust_val = get_constraint_rust(
                    anddata,
                    sig_a,
                    sig_b,
                    dont_merge_cluster_seeds=dont_merge,
                    incremental_dont_use_cluster_seeds=False,
                    featurizer=rust_featurizer,
                    use_cache=False,
                )

                if py_val is not None:
                    python_constraint_hits += 1
                if rust_val is not None:
                    rust_constraint_hits += 1

                if py_val != rust_val:
                    mismatch_count += 1
                    if py_val is None:
                        mismatch_python_none += 1
                    elif rust_val is None:
                        mismatch_rust_none += 1
                    else:
                        mismatch_value_diff += 1
                    if len(examples) < 10:
                        examples.append(
                            {
                                "sig_a": sig_a,
                                "sig_b": sig_b,
                                "python": py_val,
                                "rust": rust_val,
                            }
                        )

            constraint_parity = {
                "sample_pairs_requested": int(constraint_sample),
                "sample_pairs_effective": int(sample_pairs),
                "python_constraint_hits": int(python_constraint_hits),
                "rust_constraint_hits": int(rust_constraint_hits),
                "mismatch_count": int(mismatch_count),
                "mismatch_rate": round(mismatch_count / sample_pairs, 6) if sample_pairs > 0 else 0.0,
                "mismatch_python_none": int(mismatch_python_none),
                "mismatch_rust_none": int(mismatch_rust_none),
                "mismatch_value_diff": int(mismatch_value_diff),
                "mismatch_examples": examples,
            }

    result = {
        "backend": backend,
        "backend_label": run_label or backend,
        "dataset": dataset_name,
        "block_key": block_key,
        "original_block_size": original_block_size,
        "effective_block_size": block_size,
        "num_pairs": num_pairs,
        "max_block_size_limit": max_block_size,
        "n_jobs": n_jobs,
        "model_path": resolved_model_path,
        "data_root": resolved_data_root,
        "anddata_build_seconds": round(anddata_build_seconds, 3),
        "warm_rust_featurizer_seconds": round(warm_rust_featurizer_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_rss_gb": round(monitor.peak_gb, 3),
        "num_clusters": num_clusters,
        "assigned_signatures": assigned_sigs,
        "cluster_sizes_top10": cluster_sizes[:10],
        "cluster_membership_digest": cluster_membership_digest,
        "signature_to_cluster_fingerprint": signature_to_cluster_fingerprint,
        "quality_metrics": quality_metrics,
        "constraint_parity": constraint_parity,
        "profile_output_path": profile_output_path,
        "rust_extension_identity": rust_extension_identity,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }
    return result


# ---------------------------------------------------------------------------
# Compare mode: runs Python and Rust in subprocesses
# ---------------------------------------------------------------------------


def _extract_json_payload(stdout_text: str) -> dict[str, Any]:
    start = stdout_text.find(RESULT_JSON_START)
    end = stdout_text.find(RESULT_JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "Failed to parse result JSON markers from subprocess output.\n"
            f"STDOUT (last 2000 chars):\n{stdout_text[-2000:]}"
        )
    payload_text = stdout_text[start + len(RESULT_JSON_START) : end].strip()
    return json.loads(payload_text)


def _run_single_subprocess(
    backend: str,
    dataset_name: str,
    block_key: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str,
    data_root: str,
    max_block_size: int,
    run_label: str,
    timeout_seconds: int,
    quality_check: bool,
    constraint_sample: int,
    constraint_sample_seed: int,
    emit_signature_map: bool,
    require_rust_release: bool,
) -> dict[str, Any]:
    """Run a single backend in a subprocess (isolation for RSS measurement)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "single",
        "--backend",
        backend,
        "--dataset",
        dataset_name,
        "--block",
        block_key,
        "--n-jobs",
        str(n_jobs),
        "--profile-output-path",
        profile_output_path,
        "--model-path",
        model_path,
        "--data-root",
        data_root,
        "--require-rust-release",
        str(int(require_rust_release)),
        "--run-label",
        run_label or backend,
    ]
    if max_block_size > 0:
        cmd.extend(["--max-block-size", str(max_block_size)])
    if quality_check:
        cmd.append("--quality-check")
    if constraint_sample > 0:
        cmd.extend(["--constraint-sample", str(constraint_sample)])
        cmd.extend(["--constraint-sample-seed", str(constraint_sample_seed)])
    if emit_signature_map:
        cmd.append("--emit-signature-map")

    print(f"\n{'='*70}")
    print(f"Launching subprocess: {backend} (timeout={timeout_seconds}s = {timeout_seconds/3600:.1f}h)")
    print(f"  Dataset: {dataset_name}, Block: {block_key!r}")
    print(f"{'='*70}")

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired as e:
        print(f"TIMEOUT after {timeout_seconds}s for {backend}!")
        print(f"  Partial stdout (last 2000 chars): {(e.stdout or b'')[-2000:]}")
        print(f"  Partial stderr (last 2000 chars): {(e.stderr or b'')[-2000:]}")
        raise

    # Stream subprocess output
    if completed.stdout:
        in_payload = False
        suppressed = 0
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if stripped == RESULT_JSON_START:
                in_payload = True
                print(f"  [{backend}] {RESULT_JSON_START} (payload elided)")
                continue
            if stripped == RESULT_JSON_END:
                in_payload = False
                print(f"  [{backend}] {RESULT_JSON_END}")
                continue
            if in_payload:
                suppressed += 1
                continue
            print(f"  [{backend}] {line}")
        if suppressed > 0:
            print(f"  [{backend}] (suppressed {suppressed} JSON payload line(s))")
    if completed.stderr:
        for line in completed.stderr.splitlines():
            print(f"  [{backend} STDERR] {line}")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (returncode={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout[-3000:]}\nSTDERR:\n{completed.stderr[-3000:]}"
        )

    return _extract_json_payload(completed.stdout)


def _compare_runs(args: argparse.Namespace) -> None:
    scratch_dir = PROJECT_ROOT / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset
    block_key = args.block
    timeout_seconds = args.timeout_hours * 3600
    resolved_model_path = _resolve_path(args.model_path)
    resolved_data_root = _resolve_path(args.data_root)

    # Auto-detect largest block if not specified
    if not dataset_name or not block_key:
        print("Scanning all datasets for the largest block...")
        dataset_name, block_key, block_size = find_largest_block()
        print(f"\nLargest block: {block_key!r} in {dataset_name} ({block_size:,} signatures)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    python_profile_path = str(scratch_dir / f"profile_largest_block_python_{timestamp}.txt")
    rust_profile_path = str(scratch_dir / f"profile_largest_block_rust_{timestamp}.txt")

    # Run Python
    python_result = _run_single_subprocess(
        backend="python",
        dataset_name=dataset_name,
        block_key=block_key,
        n_jobs=args.n_jobs,
        profile_output_path=python_profile_path,
        model_path=resolved_model_path,
        data_root=resolved_data_root,
        max_block_size=args.max_block_size,
        run_label="python",
        timeout_seconds=timeout_seconds,
        quality_check=bool(args.quality_check),
        constraint_sample=int(args.constraint_sample),
        constraint_sample_seed=int(args.constraint_sample_seed),
        emit_signature_map=True,
        require_rust_release=bool(args.require_rust_release),
    )

    # Run Rust
    rust_result = _run_single_subprocess(
        backend="rust",
        dataset_name=dataset_name,
        block_key=block_key,
        n_jobs=args.n_jobs,
        profile_output_path=rust_profile_path,
        model_path=resolved_model_path,
        data_root=resolved_data_root,
        max_block_size=args.max_block_size,
        run_label="rust",
        timeout_seconds=timeout_seconds,
        quality_check=bool(args.quality_check),
        constraint_sample=int(args.constraint_sample),
        constraint_sample_seed=int(args.constraint_sample_seed),
        emit_signature_map=True,
        require_rust_release=bool(args.require_rust_release),
    )

    cluster_equivalent = (
        python_result.get("cluster_membership_digest") == rust_result.get("cluster_membership_digest")
        and python_result.get("cluster_membership_digest") is not None
    )
    python_partition = python_result.get("signature_to_cluster_fingerprint") or {}
    rust_partition = rust_result.get("signature_to_cluster_fingerprint") or {}
    signature_partition_diff_count = int(
        sum(
            1
            for signature_id, fingerprint in python_partition.items()
            if rust_partition.get(signature_id) != fingerprint
        )
    )
    block_size = int(python_result["effective_block_size"])
    signature_partition_diff_fraction = float(signature_partition_diff_count / max(1, block_size))

    quality_delta: dict[str, Any] | None = None
    py_quality = python_result.get("quality_metrics")
    rust_quality = rust_result.get("quality_metrics")
    if isinstance(py_quality, dict) and isinstance(rust_quality, dict):
        py_b3 = py_quality.get("b3")
        rust_b3 = rust_quality.get("b3")
        py_pw = py_quality.get("pairwise")
        rust_pw = rust_quality.get("pairwise")
        if (
            isinstance(py_b3, dict)
            and isinstance(rust_b3, dict)
            and isinstance(py_pw, dict)
            and isinstance(rust_pw, dict)
        ):
            quality_delta = {
                "b3_f1_delta": round(float(rust_b3.get("f1", 0.0)) - float(py_b3.get("f1", 0.0)), 6),
                "pairwise_f1_delta": round(float(rust_pw.get("f1", 0.0)) - float(py_pw.get("f1", 0.0)), 6),
            }

    # Keep summary JSON compact by default.
    python_result.pop("signature_to_cluster_fingerprint", None)
    rust_result.pop("signature_to_cluster_fingerprint", None)

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset:    {dataset_name}")
    print(f"Block:      {block_key!r}")
    print(f"Block size: {python_result['effective_block_size']} signatures")
    print(f"Pairs:      {python_result['num_pairs']:,}")
    print(f"Cluster equivalent (digest): {cluster_equivalent}")
    print(
        "Signature partition diff: "
        f"{signature_partition_diff_count}/{block_size} ({100.0 * signature_partition_diff_fraction:.3f}%)"
    )
    print()

    headers = ["Metric", "Python", "Rust", "Delta"]
    rows = []

    def _delta(py_val, rust_val, unit="s"):
        diff = rust_val - py_val
        pct = (diff / py_val * 100) if py_val > 0 else 0
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.3f}{unit} ({sign}{pct:.1f}%)"

    rows.append(
        (
            "ANDData build (s)",
            f"{python_result['anddata_build_seconds']:.3f}",
            f"{rust_result['anddata_build_seconds']:.3f}",
            _delta(python_result["anddata_build_seconds"], rust_result["anddata_build_seconds"]),
        )
    )
    rows.append(("Rust warm (s)", "N/A", f"{rust_result['warm_rust_featurizer_seconds']:.3f}", ""))
    rows.append(
        (
            "Predict (s)",
            f"{python_result['predict_seconds']:.3f}",
            f"{rust_result['predict_seconds']:.3f}",
            _delta(python_result["predict_seconds"], rust_result["predict_seconds"]),
        )
    )
    rows.append(
        (
            "Total (s)",
            f"{python_result['total_seconds']:.3f}",
            f"{rust_result['total_seconds']:.3f}",
            _delta(python_result["total_seconds"], rust_result["total_seconds"]),
        )
    )
    rows.append(
        (
            "Peak RSS (GB)",
            f"{python_result['peak_rss_gb']:.3f}",
            f"{rust_result['peak_rss_gb']:.3f}",
            _delta(python_result["peak_rss_gb"], rust_result["peak_rss_gb"], unit=" GB"),
        )
    )
    rows.append(("Clusters", str(python_result["num_clusters"]), str(rust_result["num_clusters"]), ""))

    # Print table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-|-".join("-" * w for w in col_widths)
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

    if isinstance(py_quality, dict) and isinstance(rust_quality, dict):
        py_b3 = py_quality.get("b3", {})
        rust_b3 = rust_quality.get("b3", {})
        py_pw = py_quality.get("pairwise", {})
        rust_pw = rust_quality.get("pairwise", {})
        print("\nQuality vs ground truth (block-level):")
        print(
            "  Python: "
            f"B3 F1={py_b3.get('f1')} (P={py_b3.get('precision')}, R={py_b3.get('recall')}) | "
            f"Pairwise F1={py_pw.get('f1')} (P={py_pw.get('precision')}, R={py_pw.get('recall')})"
        )
        print(
            "  Rust:   "
            f"B3 F1={rust_b3.get('f1')} (P={rust_b3.get('precision')}, R={rust_b3.get('recall')}) | "
            f"Pairwise F1={rust_pw.get('f1')} (P={rust_pw.get('precision')}, R={rust_pw.get('recall')})"
        )
        if quality_delta is not None:
            print(
                "  Delta (Rust-Python): "
                f"B3 F1={quality_delta.get('b3_f1_delta')} | Pairwise F1={quality_delta.get('pairwise_f1_delta')}"
            )

    python_constraint_parity = python_result.get("constraint_parity")
    rust_constraint_parity = rust_result.get("constraint_parity")
    if python_constraint_parity is not None or rust_constraint_parity is not None:
        print("\nConstraint parity sample (python vs rust constraints):")
        if isinstance(python_constraint_parity, dict):
            print(
                "  Python-run: "
                f"mismatch_rate={python_constraint_parity.get('mismatch_rate')} "
                f"mismatches={python_constraint_parity.get('mismatch_count')}/"
                f"{python_constraint_parity.get('sample_pairs_effective')} "
                f"python_hits={python_constraint_parity.get('python_constraint_hits')} "
                f"rust_hits={python_constraint_parity.get('rust_constraint_hits')}"
            )
        if isinstance(rust_constraint_parity, dict):
            print(
                "  Rust-run:   "
                f"mismatch_rate={rust_constraint_parity.get('mismatch_rate')} "
                f"mismatches={rust_constraint_parity.get('mismatch_count')}/"
                f"{rust_constraint_parity.get('sample_pairs_effective')} "
                f"python_hits={rust_constraint_parity.get('python_constraint_hits')} "
                f"rust_hits={rust_constraint_parity.get('rust_constraint_hits')}"
            )

    print(f"\nPython cProfile: {python_profile_path}")
    print(f"Rust cProfile:   {rust_profile_path}")

    # Speedup summary
    py_pred = python_result["predict_seconds"]
    rust_pred = rust_result["predict_seconds"]
    if rust_pred > 0:
        print(f"\nPredict speedup (Python/Rust): {py_pred/rust_pred:.2f}x")
    py_total = python_result["total_seconds"]
    rust_total = rust_result["total_seconds"]
    if rust_total > 0:
        print(f"Total speedup (Python/Rust):   {py_total/rust_total:.2f}x")

    # Write JSON
    if args.write_json:
        summary = {
            "dataset": dataset_name,
            "block_key": block_key,
            "effective_block_size": python_result["effective_block_size"],
            "num_pairs": python_result["num_pairs"],
            "n_jobs": args.n_jobs,
            "data_root": _resolve_path(args.data_root),
            "model_path": _resolve_path(args.model_path),
            "max_block_size_limit": args.max_block_size,
            "timeout_hours": args.timeout_hours,
            "cluster_equivalent": bool(cluster_equivalent),
            "signature_partition_diff_count": signature_partition_diff_count,
            "signature_partition_diff_fraction": round(signature_partition_diff_fraction, 6),
            "python": python_result,
            "rust": rust_result,
            "predict_speedup": round(py_pred / rust_pred, 3) if rust_pred > 0 else None,
            "total_speedup": round(py_total / rust_total, 3) if rust_total > 0 else None,
            "quality_delta": quality_delta,
            "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
        }
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nSummary JSON: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile Python vs Rust on the largest block across all datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "single"],
        default="compare",
        help="'compare' runs both backends in subprocesses; 'single' runs one backend in-process.",
    )
    parser.add_argument(
        "--backend",
        choices=["python", "rust"],
        default="python",
        help="Backend for --mode=single.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset name (e.g. 'aminer'). Empty = auto-detect largest block.",
    )
    parser.add_argument(
        "--block",
        default="",
        help="Block key (e.g. 'j wang'). Empty = auto-detect largest in dataset.",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=0,
        help="Limit block to first N signatures (0 = use full block).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to production model pickle.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DATA_DIR),
        help="Dataset root directory containing per-dataset files.",
    )
    parser.add_argument(
        "--profile-output-path",
        default="",
        help="cProfile output path (required for --mode=single).",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Output JSON path for results.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Label for this run (single mode).",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=4.0,
        help="Timeout per subprocess in hours (default: 4 hours).",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Compute block-level quality metrics vs ground truth clusters (requires clusters.json).",
    )
    parser.add_argument(
        "--constraint-sample",
        type=int,
        default=0,
        help="Sample N random pairs from the block and compare python vs rust constraint outputs.",
    )
    parser.add_argument(
        "--constraint-sample-seed",
        type=int,
        default=42,
        help="RNG seed for --constraint-sample (reproducible).",
    )
    parser.add_argument(
        "--emit-signature-map",
        action="store_true",
        help="Include signature->cluster fingerprint mapping in single-run JSON (debug; can be large).",
    )
    parser.add_argument(
        "--require-rust-release",
        type=int,
        choices=[0, 1],
        default=0,
        help="When backend=rust, fail if extension build reports debug_assertions.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        dataset_name = args.dataset
        block_key = args.block

        # Auto-detect if not specified
        if not dataset_name or not block_key:
            print("Scanning all datasets for the largest block...")
            dataset_name, block_key, block_size = find_largest_block()
            print(f"Largest block: {block_key!r} in {dataset_name} ({block_size:,} signatures)")

        if not args.profile_output_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            args.profile_output_path = str(
                PROJECT_ROOT / "scratch" / f"profile_largest_block_{args.backend}_{timestamp}.txt"
            )
            Path(args.profile_output_path).parent.mkdir(parents=True, exist_ok=True)

        result = _run_single(
            backend=args.backend,
            dataset_name=dataset_name,
            block_key=block_key,
            n_jobs=args.n_jobs,
            profile_output_path=args.profile_output_path,
            model_path=args.model_path,
            data_root=args.data_root,
            max_block_size=args.max_block_size,
            run_label=args.run_label or args.backend,
            quality_check=bool(args.quality_check),
            constraint_sample=int(args.constraint_sample),
            constraint_sample_seed=int(args.constraint_sample_seed),
            emit_signature_map=bool(args.emit_signature_map),
            require_rust_release=bool(args.require_rust_release),
        )

        print(f"\n[{args.backend}] Done in {result['total_seconds']:.1f}s")
        print(f"  Predict: {result['predict_seconds']:.1f}s")
        print(f"  Peak RSS: {result['peak_rss_gb']:.3f} GB")
        print(f"  Clusters: {result['num_clusters']}")
        print(f"  cProfile: {args.profile_output_path}")

        if args.write_json:
            output_path = Path(args.write_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, sort_keys=True)
            print(f"  JSON: {args.write_json}")

        # Also emit markers for subprocess extraction
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
        return

    _compare_runs(args)


if __name__ == "__main__":
    main()
