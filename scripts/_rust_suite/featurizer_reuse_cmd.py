import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.common import RSSMonitor, collect_rust_extension_identity  # type: ignore  # noqa: E402

DEFAULT_ARROW_DATA_ROOT = os.path.join("s2and", "data")
DEFAULT_SPECTER_SUFFIX = "_specter2.pkl"
DEFAULT_ARROW_TOTAL_RAM_BYTES = 1_000_000_000_000


def _resolve_path(project_root: str, maybe_relative_path: str) -> str:
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(project_root) / candidate)


def _run_cluster_eval_arrow(
    arrow_paths: Mapping[str, str],
    clusterer: Any,
    cluster_eval_arrow_fn: Any,
    *,
    n_jobs: int,
) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    cluster_metrics, _ = cluster_eval_arrow_fn(
        arrow_paths,
        clusterer,
        random_seed=42,
        n_jobs=n_jobs,
        split="test",
        total_ram_bytes=DEFAULT_ARROW_TOTAL_RAM_BYTES,
    )
    return time.perf_counter() - start, cluster_metrics


def _iteration_metrics(iteration: int, prediction_seconds: float, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "prediction_seconds": round(prediction_seconds, 3),
        "b3_f1": round(float(metrics["B3 (P, R, F1)"][2]), 3),
        "cluster_f1": round(float(metrics["Cluster (P, R F1)"][2]), 3),
        "cluster_macro_f1": round(float(metrics["Cluster Macro (P, R, F1)"][2]), 3),
    }


def _prepare_rust_backend(n_jobs: int, require_rust_release: bool) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ["S2AND_BACKEND"] = "rust"
    return collect_rust_extension_identity(
        require_release=bool(require_rust_release),
        fail_if_unavailable=True,
    )


def _finalize_result(
    *,
    dataset_name: str,
    n_jobs: int,
    repeats: int,
    same_object_iterations: list[dict[str, Any]],
    reinstantiated_iterations: list[dict[str, Any]],
    same_peak_rss_gb: float,
    reinstantiated_peak_rss_gb: float,
    rust_extension_identity: dict[str, Any],
    input_format: str,
) -> dict[str, Any]:
    same_total = sum(item["prediction_seconds"] for item in same_object_iterations)
    reinst_total = sum(item["prediction_seconds"] for item in reinstantiated_iterations)

    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "input_format": input_format,
        "n_jobs": n_jobs,
        "repeats": repeats,
        "same_object": {
            "iterations": same_object_iterations,
            "total_prediction_seconds": round(same_total, 3),
            "mean_prediction_seconds": round(same_total / repeats, 3),
            "peak_rss_gb": round(same_peak_rss_gb, 3),
        },
        "reinstantiated_object": {
            "iterations": reinstantiated_iterations,
            "total_prediction_seconds": round(reinst_total, 3),
            "mean_prediction_seconds": round(reinst_total / repeats, 3),
            "peak_rss_gb": round(reinstantiated_peak_rss_gb, 3),
        },
        "rust_extension_identity": rust_extension_identity,
    }

    same_mean = float(result["same_object"]["mean_prediction_seconds"])
    reinst_mean = float(result["reinstantiated_object"]["mean_prediction_seconds"])
    result["delta_reinstantiated_minus_same_seconds"] = round(reinst_mean - same_mean, 3)
    return result


def _run_arrow_reuse_profile(
    *,
    dataset_name: str,
    n_jobs: int,
    repeats: int,
    model_path: str,
    require_rust_release: bool,
    arrow_data_root: str,
    specter_suffix: str,
) -> dict[str, Any]:
    rust_extension_identity = _prepare_rust_backend(n_jobs, require_rust_release)

    from s2and.consts import PROJECT_ROOT_PATH
    from s2and.production_model import load_production_model
    from scripts.eval_prod_models import cluster_eval_arrow, resolve_arrow_dataset_paths

    resolved_model_path = _resolve_path(PROJECT_ROOT_PATH, model_path)
    resolved_arrow_root = _resolve_path(PROJECT_ROOT_PATH, arrow_data_root)
    clusterer = load_production_model(resolved_model_path)
    clusterer.n_jobs = n_jobs

    same_arrow_paths = resolve_arrow_dataset_paths(resolved_arrow_root, dataset_name, specter_suffix)
    same_object_iterations: list[dict[str, Any]] = []
    with RSSMonitor(interval_seconds=0.05) as same_monitor:
        for iteration in range(1, repeats + 1):
            prediction_seconds, metrics = _run_cluster_eval_arrow(
                same_arrow_paths,
                clusterer,
                cluster_eval_arrow,
                n_jobs=n_jobs,
            )
            iteration_result = _iteration_metrics(iteration, prediction_seconds, metrics)
            iteration_result["arrow_predict_telemetry"] = dict(
                getattr(clusterer, "_last_arrow_predict_telemetry", {}) or {}
            )
            same_object_iterations.append(iteration_result)

    reinstantiated_iterations: list[dict[str, Any]] = []
    with RSSMonitor(interval_seconds=0.05) as reinstantiated_monitor:
        for iteration in range(1, repeats + 1):
            arrow_paths = resolve_arrow_dataset_paths(resolved_arrow_root, dataset_name, specter_suffix)
            prediction_seconds, metrics = _run_cluster_eval_arrow(
                arrow_paths,
                clusterer,
                cluster_eval_arrow,
                n_jobs=n_jobs,
            )
            iteration_result = _iteration_metrics(iteration, prediction_seconds, metrics)
            iteration_result["arrow_predict_telemetry"] = dict(
                getattr(clusterer, "_last_arrow_predict_telemetry", {}) or {}
            )
            reinstantiated_iterations.append(iteration_result)

    result = _finalize_result(
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        repeats=repeats,
        same_object_iterations=same_object_iterations,
        reinstantiated_iterations=reinstantiated_iterations,
        same_peak_rss_gb=same_monitor.peak_gb,
        reinstantiated_peak_rss_gb=reinstantiated_monitor.peak_gb,
        rust_extension_identity=rust_extension_identity,
        input_format="arrow",
    )
    result["arrow_data_root"] = resolved_arrow_root
    result["model_path"] = resolved_model_path
    result["specter_suffix"] = specter_suffix
    return result


def run_reuse_profile(
    *,
    dataset_name: str,
    n_jobs: int,
    repeats: int,
    model_path: str,
    require_rust_release: bool = False,
    input_format: str = "arrow",
    arrow_data_root: str = DEFAULT_ARROW_DATA_ROOT,
    specter_suffix: str = DEFAULT_SPECTER_SUFFIX,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if input_format != "arrow":
        raise ValueError("Rust featurizer reuse profiling requires --input-format arrow; JSON/ANDData uses Python")
    return _run_arrow_reuse_profile(
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        repeats=repeats,
        model_path=model_path,
        require_rust_release=require_rust_release,
        arrow_data_root=arrow_data_root,
        specter_suffix=specter_suffix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark repeated Arrow/Rust predictions and Rust featurizer reuse in one process."
    )
    parser.add_argument("--dataset-name", default="kisti")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--input-format", choices=["arrow"], default="arrow")
    parser.add_argument("--model-path", required=True, help="Complete native production bundle path.")
    parser.add_argument(
        "--arrow-data-root",
        default=DEFAULT_ARROW_DATA_ROOT,
        help="Arrow bundle root containing per-dataset manifests (relative to project root or absolute).",
    )
    parser.add_argument(
        "--specter-suffix",
        choices=["_specter.pickle", "_specter2.pkl"],
        default=DEFAULT_SPECTER_SUFFIX,
        help="Embedding/model suffix used to select the Arrow embedding file.",
    )
    parser.add_argument("--write-json", required=True)
    args = parser.parse_args()

    result = run_reuse_profile(
        dataset_name=args.dataset_name,
        n_jobs=args.n_jobs,
        repeats=args.repeats,
        model_path=args.model_path,
        require_rust_release=bool(args.require_rust_release),
        input_format=args.input_format,
        arrow_data_root=args.arrow_data_root,
        specter_suffix=args.specter_suffix,
    )

    output_path = Path(args.write_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
