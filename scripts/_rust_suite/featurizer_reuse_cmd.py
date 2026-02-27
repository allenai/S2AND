import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts._rust_suite.common import RSSMonitor, collect_rust_extension_identity
else:
    try:
        from _rust_suite.common import RSSMonitor, collect_rust_extension_identity
    except ModuleNotFoundError:
        _SCRIPTS_DIR = Path(__file__).resolve().parents[1]
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from _rust_suite.common import RSSMonitor, collect_rust_extension_identity


def _build_data_paths(project_root: str, dataset_name: str) -> dict[str, str]:
    data_root = os.path.join(project_root, "data", "s2and_mini")
    dataset_root = os.path.join(data_root, dataset_name)
    return {
        "signatures": os.path.join(dataset_root, f"{dataset_name}_signatures.json"),
        "papers": os.path.join(dataset_root, f"{dataset_name}_papers.json"),
        "clusters": os.path.join(dataset_root, f"{dataset_name}_clusters.json"),
        "specter": os.path.join(dataset_root, f"{dataset_name}_specter.pickle"),
    }


def _build_train_dataset(dataset_name: str, n_jobs: int, paths: dict[str, str], anddata_cls: Any) -> Any:
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
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=True,
    )


def _run_cluster_eval(dataset: Any, clusterer: Any, cluster_eval_fn: Any) -> tuple[float, dict[str, Any]]:
    start = time.perf_counter()
    cluster_metrics, _ = cluster_eval_fn(dataset, clusterer, split="test", use_s2_clusters=False)
    return time.perf_counter() - start, cluster_metrics


def run_reuse_profile(
    *,
    dataset_name: str,
    n_jobs: int,
    repeats: int,
    require_rust_release: bool = False,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = "rust"
    rust_extension_identity = collect_rust_extension_identity(
        require_release=bool(require_rust_release),
        fail_if_unavailable=True,
    )

    from s2and.consts import PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.feature_port import _rust_featurizer_build_count, clear_rust_featurizer_cache
    from s2and.serialization import load_pickle_with_verified_label_encoder_compat

    paths = _build_data_paths(PROJECT_ROOT_PATH, dataset_name)
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {key} path for dataset '{dataset_name}': {path}")

    model_path = os.path.join(PROJECT_ROOT_PATH, "data", "production_model_v1.1.pickle")
    clusterer = load_pickle_with_verified_label_encoder_compat(model_path)["clusterer"]
    clusterer.use_cache = False
    clusterer.n_jobs = n_jobs

    clear_rust_featurizer_cache()
    same_object_iterations: list[dict[str, Any]] = []
    same_object_dataset = _build_train_dataset(dataset_name, n_jobs, paths, ANDData)
    with RSSMonitor(interval_seconds=0.05) as same_monitor:
        for iteration in range(1, repeats + 1):
            prediction_seconds, metrics = _run_cluster_eval(same_object_dataset, clusterer, cluster_eval)
            same_object_iterations.append(
                {
                    "iteration": iteration,
                    "prediction_seconds": round(prediction_seconds, 3),
                    "featurizer_build_count": int(_rust_featurizer_build_count(same_object_dataset)),
                    "b3_f1": round(float(metrics["B3 (P, R, F1)"][2]), 3),
                    "cluster_f1": round(float(metrics["Cluster (P, R F1)"][2]), 3),
                    "cluster_macro_f1": round(float(metrics["Cluster Macro (P, R, F1)"][2]), 3),
                }
            )

    clear_rust_featurizer_cache()
    reinstantiated_iterations: list[dict[str, Any]] = []
    with RSSMonitor(interval_seconds=0.05) as reinstantiated_monitor:
        for iteration in range(1, repeats + 1):
            dataset = _build_train_dataset(dataset_name, n_jobs, paths, ANDData)
            prediction_seconds, metrics = _run_cluster_eval(dataset, clusterer, cluster_eval)
            reinstantiated_iterations.append(
                {
                    "iteration": iteration,
                    "prediction_seconds": round(prediction_seconds, 3),
                    "featurizer_build_count": int(_rust_featurizer_build_count(dataset)),
                    "b3_f1": round(float(metrics["B3 (P, R, F1)"][2]), 3),
                    "cluster_f1": round(float(metrics["Cluster (P, R F1)"][2]), 3),
                    "cluster_macro_f1": round(float(metrics["Cluster Macro (P, R, F1)"][2]), 3),
                }
            )

    same_total = sum(item["prediction_seconds"] for item in same_object_iterations)
    reinst_total = sum(item["prediction_seconds"] for item in reinstantiated_iterations)

    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "n_jobs": n_jobs,
        "repeats": repeats,
        "same_object": {
            "iterations": same_object_iterations,
            "total_prediction_seconds": round(same_total, 3),
            "mean_prediction_seconds": round(same_total / repeats, 3),
            "peak_rss_gb": round(same_monitor.peak_gb, 3),
        },
        "reinstantiated_object": {
            "iterations": reinstantiated_iterations,
            "total_prediction_seconds": round(reinst_total, 3),
            "mean_prediction_seconds": round(reinst_total / repeats, 3),
            "peak_rss_gb": round(reinstantiated_monitor.peak_gb, 3),
        },
        "rust_extension_identity": rust_extension_identity,
    }

    same_mean = float(result["same_object"]["mean_prediction_seconds"])
    reinst_mean = float(result["reinstantiated_object"]["mean_prediction_seconds"])
    result["delta_reinstantiated_minus_same_seconds"] = round(reinst_mean - same_mean, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Microbenchmark repeated KISTI train-mode predictions in one process to compare "
            "same-object vs re-instantiated dataset behavior."
        )
    )
    parser.add_argument("--dataset-name", default="kisti")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--write-json", required=True)
    args = parser.parse_args()

    result = run_reuse_profile(
        dataset_name=args.dataset_name,
        n_jobs=args.n_jobs,
        repeats=args.repeats,
        require_rust_release=bool(args.require_rust_release),
    )

    output_path = Path(args.write_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
