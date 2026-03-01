import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from _rust_suite.common import (
    ProcessTreeRSSMonitor,
    build_run_metadata,
    collect_rust_extension_identity,
    extract_marked_json_payload,
)

RESULT_JSON_START = "===S2AND_COMPARE_RESULT_START==="
RESULT_JSON_END = "===S2AND_COMPARE_RESULT_END==="
LANGUAGE_FEATURE_NAMES = {
    "english_count",
    "same_language",
    "language_reliability_count",
}
FEATURES_TO_USE = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]


def _load_dataset_inputs(
    dataset: str,
    limit: int | None,
    project_root: str,
    *,
    force_paths: bool = False,
) -> tuple[Any, Any, tempfile.TemporaryDirectory[str] | None]:
    dataset_dir = Path(project_root) / "data" / dataset
    signatures_path = dataset_dir / f"{dataset}_signatures.json"
    papers_path = dataset_dir / f"{dataset}_papers.json"

    if not signatures_path.exists() or not papers_path.exists():
        raise FileNotFoundError(f"Expected dataset files at {signatures_path} and {papers_path}")

    if limit is None:
        return str(signatures_path), str(papers_path), None

    with signatures_path.open("r", encoding="utf-8") as f:
        signatures_all = json.load(f)
    signature_items = list(signatures_all.items())[:limit]
    signatures = {k: v for k, v in signature_items}
    needed_paper_ids = {str(v["paper_id"]) for _, v in signature_items}

    with papers_path.open("r", encoding="utf-8") as f:
        papers_all = json.load(f)
    papers = {k: v for k, v in papers_all.items() if str(k) in needed_paper_ids}

    if force_paths:
        limited_tmpdir = tempfile.TemporaryDirectory(prefix=f"s2and_compare_{dataset}_")
        limited_dir = Path(limited_tmpdir.name)
        limited_signatures_path = limited_dir / f"{dataset}_signatures_limited.json"
        limited_papers_path = limited_dir / f"{dataset}_papers_limited.json"
        with limited_signatures_path.open("w", encoding="utf-8") as f:
            json.dump(signatures, f)
        with limited_papers_path.open("w", encoding="utf-8") as f:
            json.dump(papers, f)
        return str(limited_signatures_path), str(limited_papers_path), limited_tmpdir

    return signatures, papers, None


def _make_pairs(signature_ids: list[str], pair_count: int, seed: int) -> list[tuple[str, str, float]]:
    if pair_count <= 0 or len(signature_ids) < 2:
        return []
    rng = np.random.RandomState(seed)
    n = len(signature_ids)
    pairs: list[tuple[str, str, float]] = []
    for _ in range(pair_count):
        first = int(rng.randint(0, n))
        second = int(rng.randint(0, n - 1))
        if second >= first:
            second += 1
        pairs.append((signature_ids[first], signature_ids[second], 0.0))
    return pairs


def _set_backend_env(
    backend: str,
    n_jobs: int,
) -> None:
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unsupported backend: {backend}")

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = backend


def _collect_rust_package_info(require_non_dev_rust: bool, require_rust_release: bool) -> dict[str, Any]:
    from s2and import feature_port

    module = feature_port.s2and_rust
    if module is None:
        raise RuntimeError("Rust run requested but s2and_rust extension is unavailable")

    version = str(getattr(module, "__version__", "unknown"))
    module_name = str(getattr(module, "__name__", "unknown"))
    module_path = str(getattr(module, "__file__", "unknown"))

    if require_non_dev_rust and "dev" in version.lower():
        raise RuntimeError(
            f"Loaded s2and_rust version looks like a dev build: version={version} module_path={module_path}"
        )

    extension_identity = collect_rust_extension_identity(
        require_release=bool(require_rust_release),
        fail_if_unavailable=True,
    )

    return {
        "module_name": module_name,
        "module_path": module_path,
        "version": version,
        "extension_identity": extension_identity,
    }


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    from s2and.consts import PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.featurizer import FeaturizationInfo, many_pairs_featurize

    _set_backend_env(
        args.backend,
        args.n_jobs,
    )

    rust_package = None
    if args.backend == "rust":
        rust_package = _collect_rust_package_info(
            bool(args.require_non_dev_rust),
            bool(args.require_rust_release),
        )

    # Rust inference defaults to JSON ingest; keep path inputs even on limited fixtures.
    force_path_inputs = args.backend == "rust"
    signatures_input, papers_input, _tmpdir = _load_dataset_inputs(
        args.dataset,
        args.limit,
        PROJECT_ROOT_PATH,
        force_paths=force_path_inputs,
    )
    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as rss_monitor:
        anddata_start = time.perf_counter()
        dataset = ANDData(
            signatures=signatures_input,
            papers=papers_input,
            name=f"{args.dataset}_compare_{args.backend}",
            mode="inference",
            clusters=None,
            specter_embeddings=None,
            cluster_seeds=None,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=1000,
            val_pairs_size=1000,
            test_pairs_size=1000,
            n_jobs=args.n_jobs,
            load_name_counts=False,
            preprocess=True,
            random_seed=42,
            name_tuples="filtered",
            use_orcid_id=True,
            use_sinonym_overwrite=False,
            compute_reference_features=False,
        )
        anddata_seconds = time.perf_counter() - anddata_start

        signature_ids = list(dataset.signatures.keys())
        pairs = _make_pairs(signature_ids, args.pair_count, args.seed)
        featurizer_info = FeaturizationInfo(features_to_use=FEATURES_TO_USE)
        featurize_start = time.perf_counter()
        features, _labels, _nameless_features = many_pairs_featurize(
            pairs,
            dataset,
            featurizer_info,
            n_jobs=args.n_jobs,
            use_cache=False,
            chunk_size=args.chunk_size,
            nameless_featurizer_info=None,
            nan_value=np.nan,
            delete_training_data=False,
        )
        featurize_seconds = time.perf_counter() - featurize_start

    total_seconds = time.perf_counter() - total_start
    output_features_path = Path(args.output_features_path)
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_features_path, features)

    return {
        "backend": args.backend,
        "dataset": args.dataset,
        "limit": args.limit,
        "pair_count_requested": args.pair_count,
        "pair_count_featurized": len(pairs),
        "n_jobs": args.n_jobs,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "total_runtime_seconds": round(total_seconds, 3),
        "anddata_build_seconds": round(anddata_seconds, 3),
        "featurize_seconds": round(featurize_seconds, 3),
        "peak_rss_gb": round(rss_monitor.peak_gb, 3),
        "feature_shape": [int(features.shape[0]), int(features.shape[1])],
        "feature_names": featurizer_info.get_feature_names(),
        "features_npy_path": str(output_features_path),
        "rust_package": rust_package,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }


def _extract_single_result(stdout_text: str) -> dict[str, Any]:
    return extract_marked_json_payload(stdout_text, RESULT_JSON_START, RESULT_JSON_END)


def _language_feature_indices(feature_names: list[str]) -> list[int]:
    return [idx for idx, name in enumerate(feature_names) if name in LANGUAGE_FEATURE_NAMES]


def _compute_feature_parity(
    python_features: np.ndarray,
    rust_features: np.ndarray,
    feature_names: list[str],
    *,
    non_language_rtol: float,
    non_language_atol: float,
    language_max_mismatch_fraction: float,
) -> dict[str, Any]:
    if python_features.shape != rust_features.shape:
        return {
            "pass": False,
            "shape_match": False,
            "python_shape": list(python_features.shape),
            "rust_shape": list(rust_features.shape),
            "reason": "feature shape mismatch",
        }

    if python_features.shape[1] != len(feature_names):
        return {
            "pass": False,
            "shape_match": True,
            "python_shape": list(python_features.shape),
            "rust_shape": list(rust_features.shape),
            "reason": "feature name count mismatch",
        }

    language_indices = _language_feature_indices(feature_names)
    all_indices = list(range(python_features.shape[1]))
    non_language_indices = [idx for idx in all_indices if idx not in set(language_indices)]

    close_matrix = np.isclose(
        python_features,
        rust_features,
        rtol=non_language_rtol,
        atol=non_language_atol,
        equal_nan=True,
    )

    non_language_mismatches = 0
    non_language_elements = 0
    if non_language_indices:
        non_language_view = close_matrix[:, non_language_indices]
        non_language_elements = int(non_language_view.size)
        non_language_mismatches = int(non_language_elements - int(non_language_view.sum()))

    language_mismatches = 0
    language_elements = 0
    if language_indices:
        language_view = close_matrix[:, language_indices]
        language_elements = int(language_view.size)
        language_mismatches = int(language_elements - int(language_view.sum()))

    language_mismatch_fraction = (
        0.0 if language_elements == 0 else float(language_mismatches) / float(language_elements)
    )

    non_language_pass = non_language_mismatches == 0
    language_pass = language_mismatch_fraction <= language_max_mismatch_fraction

    return {
        "pass": bool(non_language_pass and language_pass),
        "shape_match": True,
        "python_shape": list(python_features.shape),
        "rust_shape": list(rust_features.shape),
        "non_language": {
            "indices": non_language_indices,
            "elements": non_language_elements,
            "mismatches": non_language_mismatches,
            "rtol": non_language_rtol,
            "atol": non_language_atol,
            "pass": non_language_pass,
        },
        "language": {
            "indices": language_indices,
            "feature_names": [feature_names[idx] for idx in language_indices],
            "elements": language_elements,
            "mismatches": language_mismatches,
            "mismatch_fraction": language_mismatch_fraction,
            "max_mismatch_fraction": language_max_mismatch_fraction,
            "pass": language_pass,
        },
    }


def _run_subprocess_once(
    script_path: Path,
    backend: str,
    features_npy_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "single",
        "--backend",
        backend,
        "--dataset",
        args.dataset,
        "--limit",
        str(args.limit),
        "--pair-count",
        str(args.pair_count),
        "--n-jobs",
        str(args.n_jobs),
        "--chunk-size",
        str(args.chunk_size),
        "--seed",
        str(args.seed),
        "--require-non-dev-rust",
        str(args.require_non_dev_rust),
        "--require-rust-release",
        str(args.require_rust_release),
        "--output-features-path",
        str(features_npy_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _extract_single_result(completed.stdout)


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="s2and_compare_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        python_features_path = tmpdir_path / "python_features.npy"
        rust_features_path = tmpdir_path / "rust_features.npy"

        python_result = _run_subprocess_once(
            script_path=script_path,
            backend="python",
            features_npy_path=python_features_path,
            args=args,
        )
        rust_result = _run_subprocess_once(
            script_path=script_path,
            backend="rust",
            features_npy_path=rust_features_path,
            args=args,
        )

        python_features = np.load(python_result["features_npy_path"])
        rust_features = np.load(rust_result["features_npy_path"])
        feature_names = list(python_result["feature_names"])

        parity = _compute_feature_parity(
            python_features,
            rust_features,
            feature_names,
            non_language_rtol=args.non_language_rtol,
            non_language_atol=args.non_language_atol,
            language_max_mismatch_fraction=args.language_max_mismatch_fraction,
        )

    runtime_speedup = (
        None
        if float(rust_result["total_runtime_seconds"]) <= 0
        else float(python_result["total_runtime_seconds"]) / float(rust_result["total_runtime_seconds"])
    )
    rss_reduction_fraction = (
        None
        if float(python_result["peak_rss_gb"]) <= 0
        else (
            (float(python_result["peak_rss_gb"]) - float(rust_result["peak_rss_gb"]))
            / float(python_result["peak_rss_gb"])
        )
    )

    summary = {
        "dataset": args.dataset,
        "limit": args.limit,
        "pair_count": args.pair_count,
        "n_jobs": args.n_jobs,
        "seed": args.seed,
        "python": python_result,
        "rust": rust_result,
        "runtime_speedup_vs_python": (None if runtime_speedup is None else round(runtime_speedup, 6)),
        "rss_reduction_vs_python_fraction": (
            None if rss_reduction_fraction is None else round(rss_reduction_fraction, 6)
        ),
        "feature_parity": parity,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }

    print("Comparison summary:")
    print(
        "1. Python total runtime: "
        f"{python_result['total_runtime_seconds']}s | peak RSS: {python_result['peak_rss_gb']} GB"
    )
    print(f"2. Rust total runtime: {rust_result['total_runtime_seconds']}s | peak RSS: {rust_result['peak_rss_gb']} GB")
    print(
        "3. Feature parity: "
        f"non-language pass={parity.get('non_language', {}).get('pass', False)} | "
        f"language pass={parity.get('language', {}).get('pass', False)}"
    )
    if runtime_speedup is not None:
        print(f"4. Runtime speedup (python/rust): {runtime_speedup:.3f}x")
    if rss_reduction_fraction is not None:
        print(f"5. Peak RSS reduction vs python: {100.0 * rss_reduction_fraction:.2f}%")

    if args.write_json:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Wrote JSON summary: {output_path}")

    if bool(args.fail_on_parity_mismatch) and not bool(parity.get("pass", False)):
        raise RuntimeError("Feature parity check failed")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all-python path vs max-rust path on total runtime, process-tree peak RSS, "
            "and feature parity (language features allowed tiny drift)."
        )
    )
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--backend", choices=["python", "rust"], default="python")
    parser.add_argument("--dataset", default="inspire", help="Dataset directory name under data/")
    parser.add_argument("--limit", type=int, default=5000, help="Signature limit for quick stage checks")
    parser.add_argument("--pair-count", type=int, default=5000, help="Random pair count for featurization parity")
    parser.add_argument("--n-jobs", type=int, default=4, help="n_jobs for ANDData and featurization")
    parser.add_argument("--chunk-size", type=int, default=100, help="many_pairs_featurize chunk_size")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic pair sampling seed")
    parser.add_argument("--require-non-dev-rust", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--non-language-rtol", type=float, default=1e-6)
    parser.add_argument("--non-language-atol", type=float, default=1e-3)
    parser.add_argument(
        "--language-max-mismatch-fraction",
        type=float,
        default=0.005,
        help="Allowed mismatch fraction for language-sensitive features only",
    )
    parser.add_argument("--fail-on-parity-mismatch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--write-json", default="", help="Optional compare-mode output JSON path")
    parser.add_argument("--output-features-path", default="", help="Required for --mode single")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "single":
        if not args.output_features_path:
            raise ValueError("--output-features-path is required for --mode single")
        result = _run_single(args)
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
        return

    _run_compare(args)


if __name__ == "__main__":
    main()
