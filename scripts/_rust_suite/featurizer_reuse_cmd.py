import argparse
import json
import os
import sys
import time
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
DEFAULT_JSON_DATA_ROOT = os.path.join("s2and", "data", "s2and_mini")
DEFAULT_SPECTER_SUFFIX = "_specter2.pkl"
DEFAULT_ARROW_TOTAL_RAM_BYTES = 1_000_000_000_000


def _resolve_path(project_root: str, maybe_relative_path: str) -> str:
    candidate = Path(maybe_relative_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(project_root) / candidate)


def _build_data_paths(project_root: str, dataset_name: str, data_root: str = DEFAULT_JSON_DATA_ROOT) -> dict[str, str]:
    data_root = _resolve_path(project_root, data_root)
    dataset_root = os.path.join(data_root, dataset_name)
    return {
        "signatures": os.path.join(dataset_root, f"{dataset_name}_signatures.json"),
        "papers": os.path.join(dataset_root, f"{dataset_name}_papers.json"),
        "clusters": os.path.join(dataset_root, f"{dataset_name}_clusters.json"),
        "specter": os.path.join(dataset_root, f"{dataset_name}_specter.pickle"),
    }


def _build_train_dataset(
    dataset_name: str,
    n_jobs: int,
    paths: dict[str, str],
    anddata_cls: Any,
    name_counts_index: Any,
) -> Any:
    return anddata_cls(
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
        name_counts_index=name_counts_index,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
    )


def _run_cluster_eval(dataset: Any, clusterer: Any, cluster_eval_fn: Any) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    cluster_metrics, _ = cluster_eval_fn(dataset, clusterer, split="test")
    return time.perf_counter() - start, cluster_metrics


def _run_cluster_eval_arrow(
    arrow_paths: dict[str, str],
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
    clusterer.use_cache = False
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


def _run_json_reuse_profile(
    *,
    dataset_name: str,
    n_jobs: int,
    repeats: int,
    model_path: str,
    require_rust_release: bool,
    json_data_root: str,
) -> dict[str, Any]:
    rust_extension_identity = _prepare_rust_backend(n_jobs, require_rust_release)

    from s2and.consts import NAME_COUNTS_INDEX_PATH, PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.feature_port import _rust_featurizer_build_count, clear_rust_featurizer_cache
    from s2and.production_model import load_production_model

    paths = _build_data_paths(PROJECT_ROOT_PATH, dataset_name, json_data_root)
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} path for dataset '{dataset_name}': {path}")

    resolved_model_path = _resolve_path(PROJECT_ROOT_PATH, model_path)
    clusterer = load_production_model(resolved_model_path)
    clusterer.use_cache = False
    clusterer.n_jobs = n_jobs

    clear_rust_featurizer_cache()
    same_object_iterations: list[dict[str, Any]] = []
    same_object_dataset = _build_train_dataset(dataset_name, n_jobs, paths, ANDData, NAME_COUNTS_INDEX_PATH)
    with RSSMonitor(interval_seconds=0.05) as same_monitor:
        for iteration in range(1, repeats + 1):
            prediction_seconds, metrics = _run_cluster_eval(same_object_dataset, clusterer, cluster_eval)
            iteration_result = _iteration_metrics(iteration, prediction_seconds, metrics)
            iteration_result["featurizer_build_count"] = int(_rust_featurizer_build_count(same_object_dataset))
            same_object_iterations.append(iteration_result)

    clear_rust_featurizer_cache()
    reinstantiated_iterations: list[dict[str, Any]] = []
    with RSSMonitor(interval_seconds=0.05) as reinstantiated_monitor:
        for iteration in range(1, repeats + 1):
            dataset = _build_train_dataset(dataset_name, n_jobs, paths, ANDData, NAME_COUNTS_INDEX_PATH)
            prediction_seconds, metrics = _run_cluster_eval(dataset, clusterer, cluster_eval)
            iteration_result = _iteration_metrics(iteration, prediction_seconds, metrics)
            iteration_result["featurizer_build_count"] = int(_rust_featurizer_build_count(dataset))
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
        input_format="json",
    )
    result["json_data_root"] = _resolve_path(PROJECT_ROOT_PATH, json_data_root)
    result["model_path"] = resolved_model_path
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
    json_data_root: str = DEFAULT_JSON_DATA_ROOT,
    specter_suffix: str = DEFAULT_SPECTER_SUFFIX,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if input_format == "arrow":
        return _run_arrow_reuse_profile(
            dataset_name=dataset_name,
            n_jobs=n_jobs,
            repeats=repeats,
            model_path=model_path,
            require_rust_release=require_rust_release,
            arrow_data_root=arrow_data_root,
            specter_suffix=specter_suffix,
        )
    if input_format == "json":
        return _run_json_reuse_profile(
            dataset_name=dataset_name,
            n_jobs=n_jobs,
            repeats=repeats,
            model_path=model_path,
            require_rust_release=require_rust_release,
            json_data_root=json_data_root,
        )
    raise ValueError(f"Unsupported input_format: {input_format}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark repeated KISTI train-mode predictions in one process. Defaults to Arrow/Rust "
            "evaluation; pass --input-format json for the legacy ANDData path."
        )
    )
    parser.add_argument("--dataset-name", default="kisti")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--input-format", choices=["arrow", "json"], default="arrow")
    parser.add_argument("--model-path", required=True, help="Complete native production bundle path.")
    parser.add_argument(
        "--arrow-data-root",
        default=DEFAULT_ARROW_DATA_ROOT,
        help="Arrow bundle root containing per-dataset manifests (relative to project root or absolute).",
    )
    parser.add_argument(
        "--json-data-root",
        default=DEFAULT_JSON_DATA_ROOT,
        help="JSON dataset root containing per-dataset directories (relative to project root or absolute).",
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
        json_data_root=args.json_data_root,
        specter_suffix=args.specter_suffix,
    )

    output_path = Path(args.write_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
