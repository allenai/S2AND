import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.common import (  # type: ignore  # noqa: E402
    PROJECT_ROOT,
    ProcessTreeRSSMonitor,
    build_run_metadata,
    collect_rust_extension_identity,
    extract_marked_json_payload,
    get_result_markers,
)

RESULT_JSON_START, RESULT_JSON_END = get_result_markers("compare")
LANGUAGE_FEATURE_NAMES = {
    "english_count",
    "same_language",
    "language_reliability_min",
}
PYTHON_EXECUTION_ROUTE = "ANDData.many_pairs_featurize"
RUST_EXECUTION_ROUTE = "RustFeaturizer.from_arrow_paths.featurize_pairs_matrix_indexed"
DEFAULT_JSON_DATA_ROOT = PROJECT_ROOT / "s2and" / "data-backup"


def _load_dataset_inputs(
    dataset: str,
    limit: int | None,
    data_root: str | Path,
    *,
    force_paths: bool = False,
) -> tuple[Any, Any, tempfile.TemporaryDirectory[str] | None]:
    dataset_dir = Path(data_root) / dataset
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


def _identity_digest(values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounded_name_count_mappings(
    signatures: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Count canonical name keys in the exact bounded comparison payload."""

    from scripts.arrow_conversion_helpers import bounded_name_count_mappings_from_signature_payloads

    return bounded_name_count_mappings_from_signature_payloads(signatures)


def _write_bounded_name_counts_index(
    signatures: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[str, str]:
    """Write a current canonical name-count index and return its logical digest."""

    from scripts.arrow_conversion_helpers import write_bounded_name_counts_index

    return write_bounded_name_counts_index(signatures, output_dir)


def _validate_bounded_args(args: argparse.Namespace) -> None:
    if args.limit is None or int(args.limit) <= 0:
        raise ValueError("--limit must be a positive integer; unbounded compare runs are not supported")
    if int(args.pair_count) < 0:
        raise ValueError("--pair-count must be non-negative")


def _write_arrow_artifact_manifest(paths: dict[str, str], output_dir: str | Path) -> Path:
    from s2and.arrow_inputs import _build_arrow_artifact_generation
    from s2and.consts import NORMALIZATION_VERSION

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": dict(paths),
                "artifact_generation": _build_arrow_artifact_generation(paths, root),
            },
            f,
            sort_keys=True,
        )
    return manifest_path


def _selected_feature_indices(featurizer_info: Any) -> list[int]:
    return sorted(
        {
            feature_index
            for feature_group in featurizer_info.features_to_use
            for feature_index in featurizer_info.feature_group_to_index[feature_group]
        }
    )


def _featurize_pairs_with_rust(
    rust_featurizer: Any,
    pairs: list[tuple[str, str, float]],
    selected_feature_indices: list[int],
    *,
    n_jobs: int,
) -> np.ndarray:
    signature_id_to_index = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }
    missing_signature_ids = sorted(
        {
            signature_id
            for left_signature_id, right_signature_id, _label in pairs
            for signature_id in (left_signature_id, right_signature_id)
            if signature_id not in signature_id_to_index
        }
    )
    if missing_signature_ids:
        raise ValueError("Arrow Rust featurizer is missing sampled signature ids: " f"{missing_signature_ids[:10]}")
    indexed_pairs = [
        (signature_id_to_index[left_signature_id], signature_id_to_index[right_signature_id])
        for left_signature_id, right_signature_id, _label in pairs
    ]
    return np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(
            indexed_pairs,
            selected_feature_indices,
            int(n_jobs),
            np.nan,
        ),
        dtype=np.float64,
    )


def _set_backend_env(
    backend: str,
    n_jobs: int,
) -> None:
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unsupported backend: {backend}")

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ["S2AND_BACKEND"] = backend


def _collect_rust_package_info(require_non_dev_rust: bool, require_rust_release: bool) -> dict[str, Any]:
    from s2and import feature_port

    module = feature_port._ensure_s2and_rust_loaded()  # noqa: SLF001

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
    from s2and.consts import NORMALIZATION_VERSION
    from s2and.data import ANDData
    from s2and.featurizer import DEFAULT_FEATURE_GROUPS, FeaturizationInfo, many_pairs_featurize

    _validate_bounded_args(args)
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

    signatures_input, papers_input, _tmpdir = _load_dataset_inputs(
        args.dataset,
        args.limit,
        args.data_root,
    )
    records_sha256 = _identity_digest(
        {
            "signatures": signatures_input,
            "papers": papers_input,
        }
    )
    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as rss_monitor:
        name_counts_start = time.perf_counter()
        name_counts_tmpdir = tempfile.TemporaryDirectory(prefix=f"s2and_compare_{args.dataset}_name_counts_")
        name_counts_index_path, name_counts_sha256 = _write_bounded_name_counts_index(
            signatures_input,
            name_counts_tmpdir.name,
        )
        name_counts_prepare_seconds = time.perf_counter() - name_counts_start

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
            name_counts_index=name_counts_index_path,
            preprocess=True,
            random_seed=42,
            name_tuples="filtered",
            use_orcid_id=True,
        )
        anddata_seconds = time.perf_counter() - anddata_start

        signature_ids = list(dataset.signatures.keys())
        pairs = _make_pairs(signature_ids, args.pair_count, args.seed)
        featurizer_info = FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS))
        arrow_prepare_seconds = 0.0
        rust_featurizer_build_seconds = 0.0
        if args.backend == "python":
            execution_route = PYTHON_EXECUTION_ROUTE
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
        else:
            from s2and.arrow_inputs import validate_arrow_prediction_artifacts
            from s2and.feature_port import build_rust_featurizer_from_arrow_paths
            from s2and.incremental_linking.feature_block_arrow import write_raw_arrow_batch_lookup_indexes
            from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata

            execution_route = RUST_EXECUTION_ROUTE
            with tempfile.TemporaryDirectory(prefix=f"s2and_compare_{args.dataset}_arrow_") as arrow_tmpdir:
                arrow_prepare_start = time.perf_counter()
                arrow_paths = write_feature_block_arrow_from_anddata(
                    dataset,
                    arrow_tmpdir,
                    signature_ids=signature_ids,
                    include_specter=False,
                )
                arrow_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(
                    arrow_paths,
                    arrow_tmpdir,
                )
                arrow_paths["name_counts_index"] = name_counts_index_path
                manifest_path = _write_arrow_artifact_manifest(arrow_paths, arrow_tmpdir)
                arrow_paths["manifest"] = str(manifest_path)
                validated_arrow_paths = validate_arrow_prediction_artifacts(
                    arrow_paths,
                    require_specter=False,
                    require_name_counts_index=True,
                    expected_normalization_version=NORMALIZATION_VERSION,
                    context="rust-suite compare",
                )
                arrow_prepare_seconds = time.perf_counter() - arrow_prepare_start

                rust_featurizer_build_start = time.perf_counter()
                rust_featurizer = build_rust_featurizer_from_arrow_paths(
                    validated_arrow_paths,
                    expected_normalization_version=NORMALIZATION_VERSION,
                    signature_ids=signature_ids,
                    name_tuples=getattr(dataset, "name_tuples", "filtered"),
                    load_name_counts=True,
                    preprocess=True,
                    num_threads=args.n_jobs,
                )
                rust_featurizer_build_seconds = time.perf_counter() - rust_featurizer_build_start

                featurize_start = time.perf_counter()
                features = _featurize_pairs_with_rust(
                    rust_featurizer,
                    pairs,
                    _selected_feature_indices(featurizer_info),
                    n_jobs=args.n_jobs,
                )
                featurize_seconds = time.perf_counter() - featurize_start

    total_seconds = time.perf_counter() - total_start
    output_features_path = Path(args.output_features_path)
    output_features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_features_path, features)

    return {
        "backend": args.backend,
        "dataset": args.dataset,
        "data_root": str(Path(args.data_root).resolve()),
        "limit": args.limit,
        "pair_count_requested": args.pair_count,
        "pair_count_featurized": len(pairs),
        "n_jobs": args.n_jobs,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "execution_route": execution_route,
        "records_sha256": records_sha256,
        "name_counts_sha256": name_counts_sha256,
        "signature_ids_sha256": _identity_digest(signature_ids),
        "pairs_sha256": _identity_digest(pairs),
        "total_runtime_seconds": round(total_seconds, 3),
        "anddata_build_seconds": round(anddata_seconds, 3),
        "name_counts_prepare_seconds": round(name_counts_prepare_seconds, 3),
        "arrow_prepare_seconds": round(arrow_prepare_seconds, 3),
        "rust_featurizer_build_seconds": round(rust_featurizer_build_seconds, 3),
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
    finite_integer_reference = np.isfinite(python_features) & (python_features == np.trunc(python_features))
    close_matrix[finite_integer_reference] = (
        python_features[finite_integer_reference] == rust_features[finite_integer_reference]
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
    language_pass = language_mismatches == 0

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
            "max_mismatch_fraction": 0.0,
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
        "--data-root",
        str(args.data_root),
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
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        stdout_tail = (exc.stdout or "")[-2000:]
        stderr_tail = (exc.stderr or "")[-4000:]
        raise RuntimeError(
            f"{backend} comparison subprocess failed with exit code {exc.returncode}; "
            f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
        ) from exc
    return _extract_single_result(completed.stdout)


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    _validate_bounded_args(args)
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

        if python_result["records_sha256"] != rust_result["records_sha256"]:
            raise RuntimeError("Python and Rust runs did not use the same bounded records")
        if python_result["name_counts_sha256"] != rust_result["name_counts_sha256"]:
            raise RuntimeError("Python and Rust runs did not use the same bounded name counts")
        if python_result["signature_ids_sha256"] != rust_result["signature_ids_sha256"]:
            raise RuntimeError("Python and Rust runs did not use the same bounded signature ids")
        if python_result["pairs_sha256"] != rust_result["pairs_sha256"]:
            raise RuntimeError("Python and Rust runs did not featurize the same sampled pairs")

        python_features = np.load(python_result["features_npy_path"])
        rust_features = np.load(rust_result["features_npy_path"])
        feature_names = list(python_result["feature_names"])

        parity = _compute_feature_parity(
            python_features,
            rust_features,
            feature_names,
            non_language_rtol=args.non_language_rtol,
            non_language_atol=args.non_language_atol,
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
        "data_root": str(Path(args.data_root).resolve()),
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
        "input_identity": {
            "records_sha256": python_result["records_sha256"],
            "name_counts_sha256": python_result["name_counts_sha256"],
            "signature_ids_sha256": python_result["signature_ids_sha256"],
            "pairs_sha256": python_result["pairs_sha256"],
            "pass": True,
        },
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
            "Compare Python reference featurization with the Arrow/Rust route on total runtime, process-tree peak RSS, "
            "and strict feature parity."
        )
    )
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--backend", choices=["python", "rust"], default="python")
    parser.add_argument("--dataset", default="qian", help="Legacy JSON dataset directory name")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_JSON_DATA_ROOT),
        help="Root containing per-dataset legacy JSON directories",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Signature limit for quick stage checks")
    parser.add_argument("--pair-count", type=int, default=5000, help="Random pair count for featurization parity")
    parser.add_argument("--n-jobs", type=int, default=4, help="n_jobs for ANDData and featurization")
    parser.add_argument("--chunk-size", type=int, default=100, help="many_pairs_featurize chunk_size")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic pair sampling seed")
    parser.add_argument("--require-non-dev-rust", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--non-language-rtol", type=float, default=0.0)
    parser.add_argument("--non-language-atol", type=float, default=1e-6)
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
