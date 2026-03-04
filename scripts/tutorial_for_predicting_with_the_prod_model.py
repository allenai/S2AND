# mypy: ignore-errors
"""
This script demonstrates how to use the production S2AND model (v1.1) for clustering.

Default examples use `data/s2and_mini/*`.
You can also point `--data-root` to `tests` and run `--dataset qian`.

Examples:
  # Bundled fixture + Rust backend
  uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --use-rust 1 --dataset qian --data-root tests --load-name-counts 0

  # Show subblocking + memory budget knobs (for large blocks)
  uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --use-rust 1 --dataset qian --batching-threshold 5000 --desired-memory-use 25000000

  # Warm Rust featurizer before prediction
  uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --use-rust 1 --dataset qian --warm-rust-featurizer-before-predict 1
"""

import argparse
import os


def _apply_backend_flag(use_rust: int | None) -> None:
    if use_rust is None:
        return
    os.environ["S2AND_BACKEND"] = "rust" if use_rust else "python"


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
        "--use-rust",
        type=int,
        choices=[0, 1],
        default=None,
        help="Set 1 to force Rust path, 0 to force Python path (default: env settings).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Run a single dataset by name (default: run all).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join("data", "s2and_mini"),
        help=(
            "Root directory containing per-dataset subfolders. "
            "Supports both <dataset>_*.json naming (mini) and plain *.json naming (tests fixtures)."
        ),
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
        help="Set 1 to enable pair-feature/Rust featurizer cache during prediction.",
    )
    parser.add_argument(
        "--warm-rust-featurizer-before-predict",
        type=int,
        choices=[0, 1],
        default=0,
        help="Set 1 to warm Rust featurizer once per dataset before running prediction.",
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
        default=os.path.join("data", "production_model_v1.1.pickle"),
        help="Model pickle path relative to repo root or absolute.",
    )
    args = parser.parse_args()

    if args.desired_memory_use is not None and args.batching_threshold is None:
        raise ValueError("--desired-memory-use requires --batching-threshold")

    _apply_backend_flag(args.use_rust)

    from s2and.consts import FEATURIZER_VERSION, PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.feature_port import warm_rust_featurizer
    from s2and.featurizer import FeaturizationInfo
    from s2and.serialization import load_pickle_with_verified_label_encoder_compat

    n_jobs = args.n_jobs
    use_cache = bool(args.use_cache)

    # Limit BLAS threads to keep things responsive
    os.environ["OMP_NUM_THREADS"] = f"{n_jobs}"

    data_original = _resolve_root(PROJECT_ROOT_PATH, args.data_root)

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

    features_to_use = [
        "name_similarity",
        "affiliation_similarity",
        "email_similarity",
        "coauthor_similarity",
        "venue_similarity",
        "year_diff",
        "title_similarity",
        # "reference_features",  # removed in the v1.1. model
        "misc_features",
        "name_counts",
        "embedding_similarity",
        "journal_similarity",
        "advanced_name_similarity",
    ]

    # we also have this special second "nameless" model that doesn't use any name-based features
    # it helps to improve clustering performance by preventing model overreliance on names
    nameless_features_to_use = [
        feature_name
        for feature_name in features_to_use
        if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
    ]

    # we store all the information about the features in this convenient wrapper
    # note: we don't need these objects in this script, but they are useful for documentation purposes
    featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION
    )
    _ = (featurization_info, nameless_featurization_info)

    # this defaults to prod 1.1 model; override via --model-path if needed
    model_path = _resolve_root(PROJECT_ROOT_PATH, args.model_path)
    clusterer = load_pickle_with_verified_label_encoder_compat(model_path)["clusterer"]
    clusterer.use_cache = use_cache
    clusterer.n_jobs = n_jobs

    print(
        "Config: "
        f"backend={os.environ.get('S2AND_BACKEND', 'auto')}, "
        f"split={args.split}, n_jobs={n_jobs}, use_cache={int(use_cache)}, "
        f"warm_rust_featurizer={args.warm_rust_featurizer_before_predict}, "
        f"batching_threshold={args.batching_threshold}, "
        f"desired_memory_use={args.desired_memory_use}, "
        f"load_name_counts={args.load_name_counts}"
    )
    print(f"Model: {model_path}")
    print(f"Data root: {data_original}")

    cluster_metrics_all = []
    for dataset_name in datasets:
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
            load_name_counts=bool(args.load_name_counts),
            preprocess=True,
            random_seed=random_seed,
            name_tuples="filtered",
            use_orcid_id=True,
            use_sinonym_overwrite=True,
        )

        if args.warm_rust_featurizer_before_predict:
            if anddata.runtime_context.resolved_backend == "rust":
                warm_rust_featurizer(anddata, use_cache=use_cache)
                print(f"[{dataset_name}] Warmed Rust featurizer")
            else:
                print(
                    f"[{dataset_name}] Skipping warm_rust_featurizer: "
                    f"resolved backend is {anddata.runtime_context.resolved_backend}"
                )

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
