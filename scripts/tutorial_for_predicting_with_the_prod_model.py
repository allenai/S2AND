# mypy: ignore-errors
"""
This script demonstrates how to use a complete native S2AND model bundle for clustering.

The Rust path uses Arrow inputs and calls
`Clusterer.predict_from_arrow_paths(...)`. JSON/ANDData input calls
`Clusterer.predict(...)` through Python.
You can also point `--data-root` to `tests` and run `--dataset qian`.

Examples:
  # Arrow release + explicit Rust API
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format arrow --dataset qian --arrow-data-root s2and/data

  # Bundled JSON fixture + explicit Python API
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format json --dataset qian --data-root tests --load-name-counts 0

  # Show subblocking + memory budget knobs (for large blocks)
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format json --dataset qian --batching-threshold 5000 --desired-memory-use 25000000
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and.arrow_inputs import ValidatedArrowInputs  # noqa: E402


def _resolve_root(project_root: str, maybe_relative: str) -> str:
    return maybe_relative if os.path.isabs(maybe_relative) else os.path.join(project_root, maybe_relative)


def _resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(
        f"Missing dataset file for '{dataset_name}'. Tried '{preferred_path}' and '{fallback_path}'."
    )


def _select_input_route(
    *,
    requested_input_format: str,
    dataset_name: str,
    arrow_data_root: str,
    specter_suffix: str,
    batching_threshold: int | None,
    desired_memory_use: int | None,
    resolve_arrow_dataset_paths,
) -> tuple[str, ValidatedArrowInputs | None]:
    """Resolve tutorial input routing without loading models or ANDData."""

    input_format = requested_input_format
    arrow_paths = None
    if input_format in {"auto", "arrow"}:
        try:
            arrow_paths = resolve_arrow_dataset_paths(arrow_data_root, dataset_name, specter_suffix)
        except FileNotFoundError:
            if input_format == "arrow":
                raise
            input_format = "json"
        else:
            if input_format == "auto" and (batching_threshold is not None or desired_memory_use is not None):
                input_format = "json"
            else:
                input_format = "arrow"

    if input_format == "arrow":
        if batching_threshold is not None or desired_memory_use is not None:
            raise ValueError("--batching-threshold and --desired-memory-use are JSON/ANDData tutorial knobs")
        if arrow_paths is None:
            raise RuntimeError("Arrow input selected without resolved Arrow paths")
    return input_format, arrow_paths


def _cluster_eval_with_predict_options(
    dataset,
    clusterer,
    *,
    split: str,
    use_s2_clusters: bool,
    batching_threshold: int | None,
    desired_memory_use: int | None,
):
    import numpy as np

    from s2and.eval import b3_precision_recall_fscore, pairwise_precision_recall_fscore

    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise ValueError("Split must be one of: train, val, test")

    cluster_to_signatures = dataset.construct_cluster_to_signatures(block_dict)
    pred_clusters, _ = clusterer.predict(
        block_dict,
        dataset,
        use_s2_clusters=use_s2_clusters,
        batching_threshold=batching_threshold,
        desired_memory_use=desired_memory_use,
    )

    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "cmacro"
    )

    def _mean_or_nan(xs):
        if len(xs) == 0:
            return float("nan")
        return float(np.round(np.mean(xs), 2))

    metrics["Pred bigger ratio (mean, count)"] = (_mean_or_nan(pred_bigger_ratios), len(pred_bigger_ratios))
    metrics["True bigger ratio (mean, count)"] = (_mean_or_nan(true_bigger_ratios), len(true_bigger_ratios))
    return metrics, b3_metrics_per_signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset by name (default: run all).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join("s2and", "data", "s2and_mini"),
        help=(
            "Root directory containing per-dataset subfolders. "
            "Supports both <dataset>_*.json naming (mini) and plain *.json naming (tests fixtures)."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "arrow", "json"],
        default="auto",
        help=(
            "Input route. auto uses Arrow when the requested dataset exists under --arrow-data-root, "
            "unless JSON-only subblocking knobs are supplied."
        ),
    )
    parser.add_argument(
        "--arrow-data-root",
        type=str,
        default=os.path.join("s2and", "data"),
        help="Arrow release root containing per-dataset manifests, relative to repo root or absolute.",
    )
    parser.add_argument(
        "--specter-suffix",
        choices=["_specter.pickle", "_specter2.pkl"],
        default="_specter2.pkl",
        help="Embedding/model suffix used to select the Arrow embedding file.",
    )
    parser.add_argument(
        "--arrow-total-ram-bytes",
        type=int,
        default=1_000_000_000_000,
        help="Memory budget passed to Clusterer.predict_from_arrow_paths.",
    )
    parser.add_argument(
        "--load-name-counts",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set 0 to skip name-count artifact loading (useful for lightweight fixtures).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for ANDData/clusterer (default: 4).",
    )
    parser.add_argument(
        "--use-cache",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Set 1 to enable the persistent pair-feature SQLite cache during cache-aware prediction paths. "
            "Same-process Rust featurizer reuse is independent of this flag."
        ),
    )
    parser.add_argument(
        "--batching-threshold",
        type=int,
        default=None,
        help=(
            "Optional subblocking threshold for Clusterer.predict. "
            "Blocks larger than this use subblocking/incremental flow."
        ),
    )
    parser.add_argument(
        "--desired-memory-use",
        type=int,
        default=None,
        help=(
            "Optional desired memory budget for subblocked predict path "
            "(signature-pair units). Requires --batching-threshold."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Evaluation split for reported metrics (default: test).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Complete native production model bundle directory, relative to repo root or absolute.",
    )
    args = parser.parse_args()

    if args.desired_memory_use is not None and args.batching_threshold is None:
        raise ValueError("--desired-memory-use requires --batching-threshold")

    from s2and.consts import FEATURIZER_VERSION, NAME_COUNTS_INDEX_PATH, PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS, FeaturizationInfo
    from s2and.production_model import load_production_model
    from scripts.eval_prod_models import cluster_eval_arrow, resolve_arrow_dataset_paths

    n_jobs = args.n_jobs
    use_cache = bool(args.use_cache)

    # Limit BLAS threads to keep things responsive
    os.environ["OMP_NUM_THREADS"] = f"{n_jobs}"

    data_original = _resolve_root(PROJECT_ROOT_PATH, args.data_root)
    arrow_data_root = _resolve_root(PROJECT_ROOT_PATH, args.arrow_data_root)

    random_seed = 42

    datasets = [
        "arnetminer",
        "inspire",
        "kisti",
        "pubmed",
        "qian",
        "zbmath",
    ]
    if args.dataset is not None:
        datasets = [args.dataset]

    # we also have this special second "nameless" model that doesn't use any name-based features
    # it helps to improve clustering performance by preventing model overreliance on names
    # we store all the information about the features in this convenient wrapper
    # note: we don't need these objects in this script, but they are useful for documentation purposes
    featurization_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    _ = (featurization_info, nameless_featurization_info)

    # The public loader accepts only an explicitly selected complete native bundle.
    model_path = _resolve_root(PROJECT_ROOT_PATH, args.model_path)
    clusterer = load_production_model(model_path)
    clusterer.use_cache = use_cache
    clusterer.n_jobs = n_jobs

    print(
        "Config: "
        f"runtime_context_default={os.environ.get('S2AND_BACKEND', 'python')}, "
        f"split={args.split}, n_jobs={n_jobs}, use_cache={int(use_cache)}, "
        f"batching_threshold={args.batching_threshold}, "
        f"desired_memory_use={args.desired_memory_use}, "
        f"load_name_counts={args.load_name_counts}, "
        f"input_format={args.input_format}"
    )
    print(f"Model: {model_path}")
    print(f"Data root: {data_original}")
    print(f"Arrow data root: {arrow_data_root}")

    cluster_metrics_all = []
    for dataset_name in datasets:
        input_format, arrow_paths = _select_input_route(
            requested_input_format=args.input_format,
            dataset_name=dataset_name,
            arrow_data_root=arrow_data_root,
            specter_suffix=args.specter_suffix,
            batching_threshold=args.batching_threshold,
            desired_memory_use=args.desired_memory_use,
            resolve_arrow_dataset_paths=resolve_arrow_dataset_paths,
        )

        if input_format == "arrow":
            if arrow_paths is None:
                raise RuntimeError("Arrow input selected without resolved Arrow paths")
            cluster_metrics, _b3_metrics_per_signature = cluster_eval_arrow(
                arrow_paths,
                clusterer,
                random_seed=random_seed,
                n_jobs=n_jobs,
                split=args.split,
                total_ram_bytes=int(args.arrow_total_ram_bytes),
            )
            print(f"[{dataset_name}] Arrow predict_from_arrow_paths (Rust)")
            print(cluster_metrics)
            cluster_metrics_all.append(cluster_metrics)
            continue

        signatures_path = _resolve_dataset_file(
            data_original, dataset_name, f"{dataset_name}_signatures.json", "signatures.json"
        )
        papers_path = _resolve_dataset_file(data_original, dataset_name, f"{dataset_name}_papers.json", "papers.json")
        clusters_path = _resolve_dataset_file(
            data_original, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"
        )
        try:
            specter_path = _resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_specter.pickle", "specter.pickle"
            )
        except FileNotFoundError:
            specter_path = None
            print(f"[{dataset_name}] No specter file found; embedding features will be missing.")

        anddata = ANDData(
            signatures=signatures_path,
            papers=papers_path,
            name=dataset_name,
            mode="train",
            specter_embeddings=specter_path,
            clusters=clusters_path,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=100000,
            val_pairs_size=10000,
            test_pairs_size=10000,
            n_jobs=n_jobs,
            name_counts_index=NAME_COUNTS_INDEX_PATH if args.load_name_counts else None,
            preprocess=True,
            random_seed=random_seed,
            name_tuples=None,
            use_orcid_id=True,
        )

        print(f"[{dataset_name}] ANDData predict (Python)")
        cluster_metrics, _b3_metrics_per_signature = _cluster_eval_with_predict_options(
            anddata,
            clusterer,
            split=args.split,
            use_s2_clusters=False,
            batching_threshold=args.batching_threshold,
            desired_memory_use=args.desired_memory_use,
        )
        print(cluster_metrics)
        cluster_metrics_all.append(cluster_metrics)

    b3s = [i["B3 (P, R, F1)"][-1] for i in cluster_metrics_all]
    print(b3s, sum(b3s) / len(b3s))

    for i in range(len(datasets)):
        print(f"Performance on {datasets[i]}: {cluster_metrics_all[i]['B3 (P, R, F1)']}")
        print()


if __name__ == "__main__":
    main()
