# mypy: ignore-errors
"""
This script demonstrates how to use a complete native S2AND model bundle for clustering.

The Rust path uses Arrow inputs and calls
`Clusterer.predict_from_arrow(...)`. JSON/ANDData input calls
`Clusterer.predict(...)` through Python.
You can also point `--data-root` to `tests` and run `--dataset qian`. The
default `--data-root` (`s2and/data-backup/s2and_mini`) is an internal AI2 mini
bundle; pass `--data-root` explicitly otherwise.

Test-split metrics use the held-out blocks recorded in the bundle, and pair
sampling uses the trainer's recorded seed, so evaluation matches training.
Recorded test splits require the complete, exact model-plan population. A full
public benchmark does not match a bundle trained on a subset; use that run's
private release Arrow root for this tutorial, or release_pairwise.py's evaluation
plan commands for independently held-out populations. Do not disable the digest
check or re-split public data to claim recorded held-out metrics.
`--specter-suffix` selects the embedding generation for both routes; production
bundles use SPECTER2.

Update at v1.3 release: every example needs a complete v1.3 production bundle,
which is not yet published (Git LFS copy: artifacts/common/v1.3). The Arrow
example also needs public-format-1 data (the currently published Arrow release
is rejected), and the JSON examples need a public-format-1 name-count index at
`<main_data_dir>/name_counts_index` (see docs/data.md). Revise this note once
they run against published artifacts.

Examples:
  # Arrow release + explicit Rust API (owner-only: the internal pre-cut Arrow root
  # matches the bundle's recorded test blocks)
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format arrow --dataset qian --arrow-data-root <corrected-snapshot>/pairwise/arrow/release

  # Bundled JSON fixture + explicit Python API (the qian fixture ships SPECTER1
  # vectors, so this is a smoke run; its scores are not meaningful for SPECTER2)
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format json --dataset qian --data-root tests --specter-suffix _specter.pickle --split val

  # Show subblocking for large blocks (the largest qian fixture block has 320 signatures)
  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path path/to/complete-production-bundle \
      --input-format json --dataset qian --data-root tests --specter-suffix _specter.pickle --split val \
      --batching-threshold 100
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and.arrow_inputs import ArrowDataset  # noqa: E402


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


def _json_specter_path(data_root: str, dataset_name: str, specter_suffix: str) -> str:
    """Return the JSON-route embedding pickle for the selected SPECTER generation."""

    return _resolve_dataset_file(
        data_root,
        dataset_name,
        f"{dataset_name}{specter_suffix}",
        specter_suffix.lstrip("_"),
    )


def _select_input_route(
    *,
    requested_input_format: str,
    dataset_name: str,
    arrow_data_root: str,
    specter_suffix: str,
    resolve_arrow_dataset,
) -> tuple[str, ArrowDataset | None]:
    """Resolve tutorial input routing without loading models or ANDData."""

    input_format = requested_input_format
    arrow_dataset = None
    if input_format in {"auto", "arrow"}:
        try:
            arrow_dataset = resolve_arrow_dataset(arrow_data_root, dataset_name, specter_suffix)
        except FileNotFoundError:
            if input_format == "arrow":
                raise
            input_format = "json"
        else:
            input_format = "arrow"

    if input_format == "arrow":
        if arrow_dataset is None:
            raise RuntimeError("Arrow input selected without an open Arrow dataset")
    return input_format, arrow_dataset


def _cluster_eval_with_predict_options(
    dataset,
    clusterer,
    *,
    split: str,
    use_s2_clusters: bool,
    batching_threshold: int | None,
    test_split=None,
):
    import numpy as np

    from s2and.eval import b3_precision_recall_fscore, pairwise_precision_recall_fscore
    from s2and.production_training_contract import frozen_test_blocks

    if split == "test" and test_split is not None:
        block_dict = frozen_test_blocks(dataset.get_blocks(), test_split)
    else:
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
        default=os.path.join("s2and", "data-backup", "s2and_mini"),
        help=(
            "Root directory containing per-dataset subfolders. "
            "Supports both <dataset>_*.json naming (mini) and plain *.json naming (tests fixtures)."
        ),
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "arrow", "json"],
        default="auto",
        help="Input route. auto uses Arrow when the requested dataset exists under --arrow-data-root.",
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
        help=(
            "SPECTER generation for both routes: selects the Arrow embedding table, or the JSON "
            "<dataset><suffix> pickle (plain fixtures: the suffix without its leading underscore)."
        ),
    )
    parser.add_argument(
        "--arrow-total-ram-bytes",
        type=int,
        default=1_000_000_000_000,
        help="Memory budget passed to Clusterer.predict_from_arrow.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for ANDData/clusterer (default: 4).",
    )
    parser.add_argument(
        "--batching-threshold",
        type=int,
        default=None,
        help=(
            "Optional maximum subblock size for Python or Arrow/Rust prediction. "
            "Blocks larger than this use the route's native subblocking/incremental flow."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help=(
            "Evaluation split for reported metrics (default: test). test uses the bundle's recorded "
            "held-out blocks; train and val re-derive splits with the bundle's recorded seed."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Complete native production model bundle directory, relative to repo root or absolute.",
    )
    args = parser.parse_args()

    from s2and.consts import NAME_COUNTS_INDEX_PATH, PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS, FeaturizationInfo
    from s2and.production_model import load_production_model
    from scripts.eval_prod_models import (
        bundle_cluster_test_splits,
        bundle_data_random_seed,
        cluster_eval_arrow,
        resolve_arrow_dataset,
    )

    n_jobs = args.n_jobs

    # Limit BLAS threads to keep things responsive
    os.environ["OMP_NUM_THREADS"] = f"{n_jobs}"

    data_original = _resolve_root(PROJECT_ROOT_PATH, args.data_root)
    arrow_data_root = _resolve_root(PROJECT_ROOT_PATH, args.arrow_data_root)

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
    )
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
    )
    _ = (featurization_info, nameless_featurization_info)

    # The public loader accepts only an explicitly selected complete native bundle.
    model_path = _resolve_root(PROJECT_ROOT_PATH, args.model_path)
    clusterer = load_production_model(model_path, n_jobs=n_jobs)
    # Reuse the trainer's recorded split identities instead of guessing a seed.
    random_seed = bundle_data_random_seed(Path(model_path))
    recorded_test_splits = bundle_cluster_test_splits(Path(model_path)) if args.split == "test" else {}

    print(
        "Config: "
        f"split={args.split}, n_jobs={n_jobs}, "
        f"batching_threshold={args.batching_threshold}, "
        f"input_format={args.input_format}"
    )
    print(f"Model: {model_path}")
    print(f"Data root: {data_original}")
    print(f"Arrow data root: {arrow_data_root}")

    cluster_metrics_all = []
    for dataset_name in datasets:
        test_split = None
        if args.split == "test":
            if dataset_name not in recorded_test_splits:
                raise ValueError(f"Bundle records no cluster test split for {dataset_name!r}")
            test_split = recorded_test_splits[dataset_name]
        input_format, arrow_dataset = _select_input_route(
            requested_input_format=args.input_format,
            dataset_name=dataset_name,
            arrow_data_root=arrow_data_root,
            specter_suffix=args.specter_suffix,
            resolve_arrow_dataset=resolve_arrow_dataset,
        )

        if input_format == "arrow":
            if arrow_dataset is None:
                raise RuntimeError("Arrow input selected without an open Arrow dataset")
            with arrow_dataset:
                cluster_metrics, _b3_metrics_per_signature = cluster_eval_arrow(
                    arrow_dataset,
                    clusterer,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                    split=args.split,
                    total_ram_bytes=int(args.arrow_total_ram_bytes),
                    batching_threshold=args.batching_threshold,
                    test_split=test_split,
                )
            print(f"[{dataset_name}] Arrow predict_from_arrow (Rust)")
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
            specter_path = _json_specter_path(data_original, dataset_name, args.specter_suffix)
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
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=100000,
            val_pairs_size=10000,
            test_pairs_size=10000,
            n_jobs=n_jobs,
            name_counts_index=NAME_COUNTS_INDEX_PATH,
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
            test_split=test_split,
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
