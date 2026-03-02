import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.common import (  # noqa: E402
    RSSMonitor,
    build_run_metadata,
    collect_rust_extension_identity,
    extract_marked_json_payload,
    get_result_markers,
)

RESULT_JSON_START, RESULT_JSON_END = get_result_markers("profile")
DEFAULT_DATA_ROOT = os.path.join("data", "s2and_mini")
DEFAULT_MODEL_PATH = os.path.join("data", "production_model_v1.1.pickle")


def _as_triplet(metrics: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = metrics.get(key, (float("nan"), float("nan"), float("nan")))
    if not isinstance(value, list | tuple) or len(value) < 3:
        return (float("nan"), float("nan"), float("nan"))
    return (float(value[0]), float(value[1]), float(value[2]))


def _fmt_triplet(values: tuple[float, float, float]) -> str:
    return f"({values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f})"


def _resolve_path(project_root: str, maybe_relative_path: str) -> str:
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(project_root) / candidate)


def _build_data_paths(project_root: str, dataset_name: str, data_root: str, specter_file: str) -> dict[str, str]:
    data_root_path = _resolve_path(project_root, data_root)
    dataset_root = os.path.join(data_root_path, dataset_name)
    specter_name = specter_file or f"{dataset_name}_specter.pickle"
    return {
        "signatures": os.path.join(dataset_root, f"{dataset_name}_signatures.json"),
        "papers": os.path.join(dataset_root, f"{dataset_name}_papers.json"),
        "clusters": os.path.join(dataset_root, f"{dataset_name}_clusters.json"),
        "specter": os.path.join(dataset_root, specter_name),
    }


def _write_profile_output(profiler: cProfile.Profile, output_path: str, elapsed_seconds: float) -> None:
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime")
    stats.print_stats(40)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(stats_stream.getvalue())
        f.write(f"\nTotal runtime (cluster_eval only): {elapsed_seconds:.3f}s\n")


def _single_run(
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    data_root: str = DEFAULT_DATA_ROOT,
    specter_file: str = "",
    rust_warm_featurizer_before_predict: int = 0,
    run_label: str | None = None,
    require_rust_release: bool = False,
) -> dict[str, Any]:
    if backend not in {"python", "rust"}:
        raise ValueError(f"Unsupported backend: {backend}")

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ["S2AND_BACKEND"] = backend
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    rust_extension_identity: dict[str, Any] | None = None
    if backend == "rust":
        rust_extension_identity = collect_rust_extension_identity(
            require_release=bool(require_rust_release),
            fail_if_unavailable=False,
        )

    from s2and.consts import PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.serialization import load_pickle_with_verified_label_encoder_compat

    resolved_model_path = _resolve_path(PROJECT_ROOT_PATH, model_path)
    clusterer = load_pickle_with_verified_label_encoder_compat(resolved_model_path)["clusterer"]
    clusterer.use_cache = False
    clusterer.n_jobs = n_jobs

    paths = _build_data_paths(PROJECT_ROOT_PATH, dataset_name, data_root, specter_file)
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} path for dataset '{dataset_name}': {path}")

    total_start = time.perf_counter()
    with RSSMonitor(interval_seconds=0.05) as monitor:
        anddata_start = time.perf_counter()
        anddata = ANDData(
            signatures=paths["signatures"],
            papers=paths["papers"],
            name=dataset_name,
            mode="train",
            specter_embeddings=paths["specter"],
            clusters=paths["clusters"],
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
        anddata_seconds = time.perf_counter() - anddata_start

        warm_rust_featurizer_seconds = 0.0
        if backend == "rust" and int(rust_warm_featurizer_before_predict) == 1:
            from s2and.feature_port import warm_rust_featurizer

            warm_start = time.perf_counter()
            warm_rust_featurizer(anddata)
            warm_rust_featurizer_seconds = time.perf_counter() - warm_start

        profiler = cProfile.Profile()
        prediction_start = time.perf_counter()
        profiler.enable()
        cluster_metrics, _ = cluster_eval(anddata, clusterer, split="test", use_s2_clusters=False)
        profiler.disable()
        prediction_seconds = time.perf_counter() - prediction_start
    total_seconds = time.perf_counter() - total_start

    _write_profile_output(profiler, profile_output_path, prediction_seconds)

    b3 = _as_triplet(cluster_metrics, "B3 (P, R, F1)")
    cluster = _as_triplet(cluster_metrics, "Cluster (P, R F1)")
    cluster_macro = _as_triplet(cluster_metrics, "Cluster Macro (P, R, F1)")
    return {
        "backend": backend,
        "backend_label": run_label or backend,
        "dataset": dataset_name,
        "n_jobs": n_jobs,
        "model_path": resolved_model_path,
        "data_root": _resolve_path(PROJECT_ROOT_PATH, data_root),
        "specter_file": specter_file or f"{dataset_name}_specter.pickle",
        "rust_warm_featurizer_before_predict": int(rust_warm_featurizer_before_predict) if backend == "rust" else 0,
        "rust_warm_featurizer_seconds": round(warm_rust_featurizer_seconds, 3),
        "total_latency_seconds": round(total_seconds, 3),
        "anddata_build_seconds": round(anddata_seconds, 3),
        "prediction_seconds": round(prediction_seconds, 3),
        "peak_rss_gb": round(monitor.peak_gb, 3),
        "b3": [round(v, 3) for v in b3],
        "cluster": [round(v, 3) for v in cluster],
        "cluster_macro": [round(v, 3) for v in cluster_macro],
        "profile_output_path": profile_output_path,
        "raw_cluster_metrics": cluster_metrics,
        "rust_extension_identity": rust_extension_identity,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }


def _render_markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Backend | Latency (s) | Peak RSS (GB) | B3 (P, R, F1) | Cluster (P, R, F1) | Cluster Macro (P, R, F1) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        b3 = _fmt_triplet(tuple(result["b3"]))
        cluster = _fmt_triplet(tuple(result["cluster"]))
        cluster_macro = _fmt_triplet(tuple(result["cluster_macro"]))
        lines.append(
            f"| {result['backend_label']} | {result['total_latency_seconds']:.3f} | {result['peak_rss_gb']:.3f} | "
            f"{b3} | {cluster} | {cluster_macro} |"
        )
    return "\n".join(lines)


def _run_single_subprocess(
    script_path: Path,
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    data_root: str = DEFAULT_DATA_ROOT,
    specter_file: str = "",
    rust_warm_featurizer_before_predict: int = 0,
    single_write_json: str = "",
    run_label: str = "",
    require_rust_release: int = 0,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "single",
        "--backend",
        backend,
        "--dataset-name",
        dataset_name,
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
    ]
    if specter_file:
        cmd.extend(["--specter-file", specter_file])
    if rust_warm_featurizer_before_predict in {0, 1}:
        cmd.extend(["--rust-warm-featurizer-before-predict", str(int(rust_warm_featurizer_before_predict))])
    if single_write_json:
        cmd.extend(["--single-write-json", single_write_json])
    if run_label:
        cmd.extend(["--run-label", run_label])
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return extract_marked_json_payload(completed.stdout, RESULT_JSON_START, RESULT_JSON_END)


def _compare_runs(args: argparse.Namespace) -> None:
    from s2and.consts import PROJECT_ROOT_PATH

    scratch_dir = Path(PROJECT_ROOT_PATH) / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    python_profile_path = str(scratch_dir / "profile_kisti_python_path.txt")
    rust_dataset_profile_path = str(scratch_dir / "profile_kisti_rust_from_dataset.txt")

    print("Running pure Python path...")
    python_result = _run_single_subprocess(
        script_path=script_path,
        backend="python",
        dataset_name=args.dataset_name,
        n_jobs=args.n_jobs,
        profile_output_path=python_profile_path,
        model_path=args.model_path,
        data_root=args.data_root,
        specter_file=args.specter_file,
        run_label="python",
        require_rust_release=args.require_rust_release,
    )

    print("Running Rust path (from_dataset)...")
    rust_dataset_result = _run_single_subprocess(
        script_path=script_path,
        backend="rust",
        dataset_name=args.dataset_name,
        n_jobs=args.n_jobs,
        profile_output_path=rust_dataset_profile_path,
        model_path=args.model_path,
        data_root=args.data_root,
        specter_file=args.specter_file,
        rust_warm_featurizer_before_predict=args.rust_warm_featurizer_before_predict,
        run_label="rust_from_dataset",
        require_rust_release=args.require_rust_release,
    )

    results = [python_result, rust_dataset_result]

    print("")
    print(_render_markdown_table(results))
    print("")
    print(f"Python profile output: {python_profile_path}")
    print(f"Rust profile output (from_dataset): {rust_dataset_profile_path}")

    if args.write_json:
        summary_path = Path(args.write_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset_name": args.dataset_name,
            "n_jobs": args.n_jobs,
            "model_path": args.model_path,
            "data_root": args.data_root,
            "specter_file": args.specter_file or f"{args.dataset_name}_specter.pickle",
            "rust_warm_featurizer_before_predict": args.rust_warm_featurizer_before_predict,
            "results": results,
            "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Summary JSON: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prod-style evaluation on pure Python path vs Rust path.")
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--backend", choices=["python", "rust"], default="python")
    parser.add_argument("--dataset-name", default="kisti")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Model artifact path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Dataset root directory containing per-dataset subfolders (relative to project root or absolute).",
    )
    parser.add_argument(
        "--specter-file",
        default="",
        help="Optional embedding filename under dataset folder. Defaults to <dataset>_specter.pickle.",
    )
    parser.add_argument(
        "--rust-warm-featurizer-before-predict",
        type=int,
        choices=[0, 1],
        default=0,
        help="When mode=single and backend=rust, call warm_rust_featurizer before cluster_eval.",
    )
    parser.add_argument(
        "--profile-output-path",
        default="",
        help="Required in single mode. Output path for cProfile summary.",
    )
    parser.add_argument(
        "--single-write-json",
        default="",
        help="Optional JSON path to persist single-run result payload.",
    )
    parser.add_argument(
        "--write-json",
        default="",
        help="Optional summary JSON path (compare mode).",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional run label (single mode), used in comparison table backend column.",
    )
    parser.add_argument(
        "--require-rust-release",
        type=int,
        choices=[0, 1],
        default=0,
        help="Fail rust runs when extension build reports debug_assertions.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        if not args.profile_output_path:
            raise ValueError("--profile-output-path is required for --mode single")
        result = _single_run(
            backend=args.backend,
            dataset_name=args.dataset_name,
            n_jobs=args.n_jobs,
            profile_output_path=args.profile_output_path,
            model_path=args.model_path,
            data_root=args.data_root,
            specter_file=args.specter_file,
            rust_warm_featurizer_before_predict=args.rust_warm_featurizer_before_predict,
            run_label=args.run_label or None,
            require_rust_release=bool(args.require_rust_release),
        )
        if args.single_write_json:
            output_path = Path(args.single_write_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
        return

    _compare_runs(args)


if __name__ == "__main__":
    main()
