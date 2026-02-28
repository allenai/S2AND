from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts._rust_suite.common import ProcessTreeRSSMonitor, collect_rust_extension_identity
else:
    try:
        from _rust_suite.common import ProcessTreeRSSMonitor, collect_rust_extension_identity
    except ModuleNotFoundError:
        _SCRIPTS_DIR = Path(__file__).resolve().parents[1]
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from _rust_suite.common import ProcessTreeRSSMonitor, collect_rust_extension_identity

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_JSON_START = "===S2AND_BIG_BLOCK_RESULT_START==="
RESULT_JSON_END = "===S2AND_BIG_BLOCK_RESULT_END==="
DEFAULT_TOTAL_RAM_BYTES = 32 * 1024 * 1024 * 1024


def _resolve_existing_file(base_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required file under {base_dir}: tried {candidates}")


def _load_subset_payload(
    subset_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    signatures_path = _resolve_existing_file(
        subset_dir,
        [
            "signatures.json",
            f"{subset_dir.name}_signatures.json",
        ],
    )
    papers_path = _resolve_existing_file(
        subset_dir,
        [
            "papers.json",
            f"{subset_dir.name}_papers.json",
        ],
    )
    with signatures_path.open("r", encoding="utf-8") as infile:
        signatures = json.load(infile)
    with papers_path.open("r", encoding="utf-8") as infile:
        papers = json.load(infile)

    meta_path = subset_dir / "meta.json"
    target_block = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as infile:
            meta = json.load(infile)
        target_block = str(meta.get("target_block")) if meta.get("target_block") else None

    return signatures, papers, target_block


def _resolve_target_block(signatures: dict[str, dict[str, Any]], target_block: str | None) -> str:
    if target_block:
        return target_block
    block_counts = Counter(
        str(payload.get("author_info", {}).get("block", ""))
        for payload in signatures.values()
        if payload.get("author_info", {}).get("block") is not None
    )
    if not block_counts:
        raise RuntimeError("No signature blocks found while resolving target block")
    return str(block_counts.most_common(1)[0][0])


def _select_block_signature_ids(
    signatures: dict[str, dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    target_block: str,
    total_signatures: int,
    random_seed: int,
) -> list[str]:
    in_block = [
        signature_id
        for signature_id, payload in signatures.items()
        if str(payload.get("author_info", {}).get("block", "")) == target_block
    ]
    if len(in_block) < total_signatures:
        raise ValueError(
            f"Requested total_signatures={total_signatures} but block '{target_block}' "
            f"has only {len(in_block)} signatures."
        )
    rng = random.Random(random_seed)
    in_block_sorted = sorted(in_block)
    rng.shuffle(in_block_sorted)
    selected: list[str] = []
    skipped_invalid_paper = 0
    for signature_id in in_block_sorted:
        paper_id = str(signatures[signature_id].get("paper_id", ""))
        paper_payload = papers.get(paper_id)
        if paper_payload is None:
            skipped_invalid_paper += 1
            continue
        if not _paper_has_block_safe_author_names(paper_payload):
            skipped_invalid_paper += 1
            continue
        selected.append(signature_id)
        if len(selected) >= total_signatures:
            break
    if len(selected) < total_signatures:
        raise ValueError(
            f"Requested total_signatures={total_signatures}, but only {len(selected)} block-safe signatures were found "
            f"in block '{target_block}' (skipped_invalid_paper={skipped_invalid_paper})."
        )
    return selected


def _paper_has_block_safe_author_names(paper_payload: dict[str, Any]) -> bool:
    from s2and.text import compute_block

    authors = paper_payload.get("authors")
    if not isinstance(authors, list):
        return True
    for author in authors:
        if isinstance(author, dict):
            author_name = author.get("author_name", "")
        else:
            author_name = str(author)
        if str(author_name).strip() == "":
            return False
        try:
            compute_block(str(author_name))
        except Exception:
            return False
    return True


def _effective_seed_cluster_count(seed_signature_count: int, requested_seed_clusters: int) -> int:
    if seed_signature_count < 2:
        raise ValueError("Need at least 2 seed signatures")
    if requested_seed_clusters <= 0:
        raise ValueError("requested_seed_clusters must be > 0")
    effective = min(requested_seed_clusters, seed_signature_count // 2)
    if effective <= 0:
        raise ValueError("Unable to form seed clusters with at least 2 signatures each")
    return effective


def _build_cluster_seeds(seed_signature_ids: list[str], seed_cluster_count: int) -> dict[str, dict[str, str]]:
    if seed_cluster_count <= 0:
        raise ValueError("seed_cluster_count must be > 0")
    if len(seed_signature_ids) < 2 * seed_cluster_count:
        raise ValueError(
            "seed_signature_ids must have at least 2 signatures per seed cluster; "
            f"got signatures={len(seed_signature_ids)} clusters={seed_cluster_count}"
        )

    cluster_seeds: dict[str, dict[str, str]] = {}
    base_size = len(seed_signature_ids) // seed_cluster_count
    remainder = len(seed_signature_ids) % seed_cluster_count
    cursor = 0
    for cluster_idx in range(seed_cluster_count):
        cluster_size = base_size + (1 if cluster_idx < remainder else 0)
        members = seed_signature_ids[cursor : cursor + cluster_size]
        cursor += cluster_size
        if len(members) < 2:
            raise ValueError(f"Cluster {cluster_idx} has <2 members; invalid seed partition")
        root = members[0]
        cluster_seeds[root] = {member: "require" for member in members[1:]}
    return cluster_seeds


def _cluster_membership_digest(cluster_to_signatures: dict[str, list[str]]) -> str:
    sorted_clusters = sorted([sorted(signatures) for signatures in cluster_to_signatures.values() if signatures])
    payload = json.dumps(sorted_clusters, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cluster_size_summary(cluster_to_signatures: dict[str, list[str]]) -> dict[str, Any]:
    sizes = sorted((len(signatures) for signatures in cluster_to_signatures.values()), reverse=True)
    return {
        "cluster_count": int(len(sizes)),
        "largest_clusters_top10": [int(size) for size in sizes[:10]],
        "max_cluster_size": int(sizes[0]) if sizes else 0,
        "min_cluster_size": int(sizes[-1]) if sizes else 0,
    }


def _signature_to_cluster_fingerprint_map(cluster_to_signatures: dict[str, list[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for signatures in cluster_to_signatures.values():
        members = sorted(signatures)
        fingerprint = hashlib.sha1("|".join(members).encode("utf-8")).hexdigest()
        for signature_id in members:
            mapping[signature_id] = fingerprint
    return mapping


def _set_runtime_env(
    *,
    backend: str,
    n_jobs: int,
) -> None:
    if backend not in {"python", "rust", "auto"}:
        raise ValueError(f"Unsupported backend={backend}")
    os.environ["S2AND_BACKEND"] = backend
    os.environ["S2AND_SKIP_FASTTEXT"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_signatures <= 0:
        raise ValueError("--total-signatures must be > 0")
    if args.seed_signatures <= 0:
        raise ValueError("--seed-signatures must be > 0")
    if args.seed_signatures >= args.total_signatures:
        raise ValueError("--seed-signatures must be < --total-signatures")
    if args.seed_cluster_count <= 0:
        raise ValueError("--seed-cluster-count must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")
    if args.batching_threshold <= 0:
        raise ValueError("--batching-threshold must be > 0")
    if args.total_signatures > 4000 and not bool(args.full_run):
        raise ValueError("Refusing large run without explicit confirmation. Use --full-run for >4000 signatures.")


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    from s2and import model as model_module
    from s2and.data import ANDData
    from s2and.serialization import load_pickle_with_verified_label_encoder_compat

    _set_runtime_env(
        backend=args.backend,
        n_jobs=int(args.n_jobs),
    )
    rust_extension_identity: dict[str, Any] | None = None
    if args.backend in {"rust", "auto"}:
        rust_extension_identity = collect_rust_extension_identity(
            require_release=bool(args.require_rust_release),
            fail_if_unavailable=False,
        )

    subset_dir = Path(args.subset_dir)
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    signatures, papers, meta_target_block = _load_subset_payload(subset_dir)
    target_block = _resolve_target_block(signatures, args.target_block or meta_target_block)
    selected_signature_ids = _select_block_signature_ids(
        signatures=signatures,
        papers=papers,
        target_block=target_block,
        total_signatures=int(args.total_signatures),
        random_seed=int(args.random_seed),
    )
    seed_signature_ids = selected_signature_ids[: int(args.seed_signatures)]
    unassigned_signature_ids = selected_signature_ids[int(args.seed_signatures) :]
    effective_seed_cluster_count = _effective_seed_cluster_count(
        seed_signature_count=len(seed_signature_ids),
        requested_seed_clusters=int(args.seed_cluster_count),
    )
    cluster_seeds = _build_cluster_seeds(seed_signature_ids, effective_seed_cluster_count)

    filtered_signatures = {signature_id: signatures[signature_id] for signature_id in selected_signature_ids}
    selected_paper_ids = {
        str(filtered_signatures[signature_id]["paper_id"])
        for signature_id in selected_signature_ids
        if filtered_signatures[signature_id].get("paper_id") is not None
    }
    filtered_papers = {paper_id: payload for paper_id, payload in papers.items() if str(paper_id) in selected_paper_ids}

    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        anddata_start = time.perf_counter()
        anddata = ANDData(
            signatures=filtered_signatures,
            papers=filtered_papers,
            name=f"big_block_incremental_{args.backend}",
            mode="inference",
            clusters=None,
            specter_embeddings=None,
            cluster_seeds=cluster_seeds,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=1000,
            val_pairs_size=1000,
            test_pairs_size=1000,
            n_jobs=int(args.n_jobs),
            load_name_counts=False,
            preprocess=True,
            random_seed=int(args.random_seed),
            name_tuples="filtered",
            use_orcid_id=True,
            use_sinonym_overwrite=False,
            compute_reference_features=False,
        )
        anddata_build_seconds = time.perf_counter() - anddata_start

        model_artifact = load_pickle_with_verified_label_encoder_compat(str(model_path))
        clusterer = model_artifact["clusterer"]
        model_module._ensure_lightgbm_fitted(clusterer.classifier)
        model_module._ensure_lightgbm_fitted(clusterer.nameless_classifier)
        clusterer.use_cache = False
        clusterer.n_jobs = int(args.n_jobs)

        predict_start = time.perf_counter()
        incremental_result = clusterer.predict_incremental(
            selected_signature_ids,
            anddata,
            batching_threshold=int(args.batching_threshold),
            total_ram_bytes=int(args.total_ram_bytes),
        )
        pred_clusters = incremental_result["clusters"]
        predict_seconds = time.perf_counter() - predict_start

    total_runtime_seconds = time.perf_counter() - total_start
    assigned_signatures = int(sum(len(signatures_in_cluster) for signatures_in_cluster in pred_clusters.values()))
    result = {
        "mode": "single",
        "backend": args.backend,
        "subset_dir": str(subset_dir),
        "model_path": str(model_path),
        "target_block": target_block,
        "total_ram_bytes": int(args.total_ram_bytes),
        "total_signatures": int(len(selected_signature_ids)),
        "seed_signatures": int(len(seed_signature_ids)),
        "unassigned_signatures": int(len(unassigned_signature_ids)),
        "seed_clusters_requested": int(args.seed_cluster_count),
        "seed_clusters_effective": int(effective_seed_cluster_count),
        "batching_threshold": int(args.batching_threshold),
        "n_jobs": int(args.n_jobs),
        "random_seed": int(args.random_seed),
        "estimated_incremental_pairs": int(len(seed_signature_ids) * len(unassigned_signature_ids)),
        "anddata_build_seconds": round(anddata_build_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "total_runtime_seconds": round(total_runtime_seconds, 3),
        "peak_rss_gb": round(monitor.peak_gb, 3),
        "predicted_cluster_count": int(len(pred_clusters)),
        "assigned_signatures": assigned_signatures,
        "cluster_size_summary": _cluster_size_summary(pred_clusters),
        "cluster_membership_digest": _cluster_membership_digest(pred_clusters),
        "phase_b_mode": str(incremental_result["phase_b_mode"]),
        "phase_b_budget_bytes": int(incremental_result["phase_b_budget_bytes"]),
        "phase_b_required_bytes": int(incremental_result["phase_b_required_bytes"]),
        "rust_extension_identity": rust_extension_identity,
    }
    if int(args.emit_signature_map) == 1:
        result["signature_to_cluster_fingerprint"] = _signature_to_cluster_fingerprint_map(pred_clusters)
    if args.single_write_json:
        output_path = Path(args.single_write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=2, sort_keys=True)
        print(f"Wrote single-run JSON: {output_path}")
    return result


def _extract_single_result(stdout_text: str) -> dict[str, Any]:
    start = stdout_text.find(RESULT_JSON_START)
    end = stdout_text.find(RESULT_JSON_END)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Could not find single-run JSON markers in subprocess output")
    payload = stdout_text[start + len(RESULT_JSON_START) : end].strip()
    return json.loads(payload)


def _run_subprocess_single(
    script_path: Path,
    args: argparse.Namespace,
    *,
    batching_threshold_override: int | None = None,
) -> dict[str, Any]:
    batching_threshold = (
        int(args.batching_threshold) if batching_threshold_override is None else int(batching_threshold_override)
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "single",
        "--backend",
        args.backend,
        "--subset-dir",
        args.subset_dir,
        "--target-block",
        args.target_block,
        "--total-signatures",
        str(args.total_signatures),
        "--seed-signatures",
        str(args.seed_signatures),
        "--seed-cluster-count",
        str(args.seed_cluster_count),
        "--batching-threshold",
        str(batching_threshold),
        "--n-jobs",
        str(args.n_jobs),
        "--random-seed",
        str(args.random_seed),
        "--total-ram-bytes",
        str(args.total_ram_bytes),
        "--model-path",
        args.model_path,
        "--require-rust-release",
        str(int(args.require_rust_release)),
        "--emit-signature-map",
        "1",
        "--full-run",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Subprocess single-run failed (returncode={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return _extract_single_result(completed.stdout)


def _run_compare_phase_split(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()

    # Force baseline into monolithic helper path by disabling subblocking.
    monolithic_threshold = max(int(args.total_signatures), int(args.batching_threshold)) + 1
    baseline_result = _run_subprocess_single(
        script_path,
        args,
        batching_threshold_override=monolithic_threshold,
    )
    candidate_result = _run_subprocess_single(
        script_path,
        args,
        batching_threshold_override=int(args.batching_threshold),
    )

    cluster_equivalent = baseline_result["cluster_membership_digest"] == candidate_result["cluster_membership_digest"]
    baseline_partition = baseline_result.get("signature_to_cluster_fingerprint", {})
    candidate_partition = candidate_result.get("signature_to_cluster_fingerprint", {})
    signature_partition_diff_count = int(
        sum(
            1
            for signature_id, fingerprint in baseline_partition.items()
            if candidate_partition.get(signature_id) != fingerprint
        )
    )
    signature_partition_diff_fraction = float(signature_partition_diff_count / max(1, len(baseline_partition)))
    runtime_delta_seconds = float(candidate_result["total_runtime_seconds"]) - float(
        baseline_result["total_runtime_seconds"]
    )
    baseline_runtime = float(baseline_result["total_runtime_seconds"])
    runtime_delta_fraction = 0.0 if baseline_runtime <= 0 else runtime_delta_seconds / baseline_runtime

    baseline_peak = float(baseline_result["peak_rss_gb"])
    candidate_peak = float(candidate_result["peak_rss_gb"])
    rss_delta_gb = candidate_peak - baseline_peak
    rss_delta_fraction = 0.0 if baseline_peak <= 0 else rss_delta_gb / baseline_peak

    baseline_compact = dict(baseline_result)
    candidate_compact = dict(candidate_result)
    baseline_compact.pop("signature_to_cluster_fingerprint", None)
    candidate_compact.pop("signature_to_cluster_fingerprint", None)

    summary = {
        "mode": "compare_phase_split",
        "comparison": "phase_split_on_vs_monolithic_baseline",
        "backend": args.backend,
        "subset_dir": args.subset_dir,
        "target_block": baseline_result["target_block"],
        "total_signatures": int(args.total_signatures),
        "seed_signatures": int(args.seed_signatures),
        "seed_clusters_requested": int(args.seed_cluster_count),
        "seed_clusters_effective": int(baseline_result["seed_clusters_effective"]),
        "baseline_batching_threshold": int(monolithic_threshold),
        "candidate_batching_threshold": int(args.batching_threshold),
        "n_jobs": int(args.n_jobs),
        "random_seed": int(args.random_seed),
        "total_ram_bytes": int(args.total_ram_bytes),
        "baseline_phase_b_mode": str(baseline_result.get("phase_b_mode", "unknown")),
        "candidate_phase_b_mode": str(candidate_result.get("phase_b_mode", "unknown")),
        "baseline": baseline_compact,
        "candidate": candidate_compact,
        "cluster_equivalent": bool(cluster_equivalent),
        "signature_partition_diff_count": signature_partition_diff_count,
        "signature_partition_diff_fraction": round(signature_partition_diff_fraction, 6),
        "runtime_delta_seconds": round(runtime_delta_seconds, 3),
        "runtime_delta_fraction": round(runtime_delta_fraction, 6),
        "peak_rss_delta_gb": round(rss_delta_gb, 3),
        "peak_rss_delta_fraction": round(rss_delta_fraction, 6),
    }

    print("Phase-split exact comparison summary:")
    print(
        f"1. Baseline (monolithic): {baseline_result['total_runtime_seconds']}s | "
        f"peak RSS: {baseline_result['peak_rss_gb']} GB | "
        f"batching_threshold={monolithic_threshold}"
    )
    print(
        f"2. Candidate (phase-split): {candidate_result['total_runtime_seconds']}s | "
        f"peak RSS: {candidate_result['peak_rss_gb']} GB | "
        f"batching_threshold={int(args.batching_threshold)}"
    )
    print(f"3. Cluster equivalent: {cluster_equivalent}")
    print(
        "4. Signature partition diff: "
        f"{signature_partition_diff_count}/{len(baseline_partition)} ({100.0 * signature_partition_diff_fraction:.3f}%)"
    )
    print(f"5. Runtime delta (candidate-baseline): {summary['runtime_delta_seconds']}s")
    print(f"6. Peak RSS delta (candidate-baseline): {summary['peak_rss_delta_gb']} GB")
    print(
        "7. Phase B mode (baseline/candidate): "
        f"{summary['baseline_phase_b_mode']} / {summary['candidate_phase_b_mode']}"
    )

    if args.write_json:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True)
        print(f"Wrote compare JSON: {output_path}")

    if bool(args.fail_on_cluster_mismatch) and not cluster_equivalent:
        raise RuntimeError("Cluster equivalence check failed between phase-split and monolithic baseline runs")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile big-block incremental inference and compare variants "
            "with process-tree peak RSS and wall-clock metrics."
        )
    )
    parser.add_argument("--mode", choices=["single", "compare_phase_split"], default="compare_phase_split")
    parser.add_argument("--backend", choices=["python", "rust", "auto"], default="rust")
    parser.add_argument("--subset-dir", default=str(_PROJECT_ROOT / "scratch" / "inventors_topblock_15k"))
    parser.add_argument(
        "--target-block",
        default="",
        help="Optional block key. Empty -> meta target block or largest block.",
    )
    parser.add_argument("--total-signatures", type=int, default=4000)
    parser.add_argument("--seed-signatures", type=int, default=3000)
    parser.add_argument("--seed-cluster-count", type=int, default=500)
    parser.add_argument("--batching-threshold", type=int, default=1500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--total-ram-bytes",
        type=int,
        default=DEFAULT_TOTAL_RAM_BYTES,
        help="RAM budget for phase-split Phase A chunking.",
    )
    parser.add_argument("--model-path", default=str(_PROJECT_ROOT / "data" / "production_model_v1.1.pickle"))
    parser.add_argument(
        "--emit-signature-map",
        type=int,
        choices=[0, 1],
        default=0,
        help="Single-mode debug: include signature-to-cluster fingerprint map in JSON output.",
    )
    parser.add_argument("--write-json", default="", help="Compare-mode output JSON path.")
    parser.add_argument("--single-write-json", default="", help="Single-mode output JSON path.")
    parser.add_argument("--fail-on-cluster-mismatch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required for runs larger than 4000 signatures.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    if args.mode == "single":
        result = _run_single(args)
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
        return

    _run_compare_phase_split(args)


if __name__ == "__main__":
    main()
