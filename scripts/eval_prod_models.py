# mypy: ignore-errors
# ruff: noqa: E402

"""
Evaluate production S2AND models (SPECTER1 vs SPECTER2) on various datasets.


In this script we try to answer the question: if we deploy SPECTER2, will S2AND care?
Both with retraining and without retraining.

This is done with s2and-mini. Ai2 employee, find it at s3://ai2-s2-research/s2and/s2and-mini/

With retraining (random seed 42):

Performance with SPECTERv1 data, on arnetminer (B3): (0.922, 0.985, 0.952)
Performance with SPECTERv2 data, on arnetminer (B3): (0.93, 0.988, 0.958)

Performance with SPECTERv1 data, on inspire (B3): (0.958, 0.974, 0.966)
Performance with SPECTERv2 data, on inspire (B3): (0.995, 0.959, 0.977)

Performance with SPECTERv1 data, on kisti (B3): (0.951, 0.971, 0.961)
Performance with SPECTERv2 data, on kisti (B3): (0.946, 0.98, 0.963)

Performance with SPECTERv1 data, on pubmed (B3): (0.849, 0.988, 0.913)
Performance with SPECTERv2 data, on pubmed (B3): (0.86, 0.988, 0.92)

Performance with SPECTERv1 data, on qian (B3): (0.936, 0.943, 0.94)
Performance with SPECTERv2 data, on qian (B3): (0.95, 0.964, 0.957)

Performance with SPECTERv1 data, on zbmath (B3): (0.966, 0.984, 0.975)
Performance with SPECTERv2 data, on zbmath (B3): (0.975, 0.991, 0.983)

---

Without retraining,

Performance with SPECTERv1 data, on arnetminer (B3): (0.977, 0.982, 0.979)
Performance with SPECTERv2 data, on arnetminer (B3):

Performance with SPECTERv1 data, on inspire (B3): (0.993, 0.964, 0.978)
Performance with SPECTERv2 data, on inspire (B3):

Performance with SPECTERv1 data, on kisti (B3): (0.96, 0.957, 0.959)
Performance with SPECTERv2 data, on kisti (B3):

Performance with SPECTERv1 data, on pubmed (B3): (1.0, 0.968, 0.984)
Performance with SPECTERv2 data, on pubmed (B3):

Performance with SPECTERv1 data, on qian (B3): (0.985, 0.955, 0.969)
Performance with SPECTERv2 data, on qian (B3):

Performance with SPECTERv1 data, on zbmath (B3): (0.967, 0.955, 0.961)
Performance with SPECTERv2 data, on zbmath (B3):


Usage:
    # Evaluate on inventors_s2and (default)
    python scripts/eval_prod_models.py

    # Evaluate on s2and_mini datasets
    python scripts/eval_prod_models.py --dataset mini

    # Retrain from scratch instead of using prod models
    python scripts/eval_prod_models.py --train

    # Override seed / n_jobs
    python scripts/eval_prod_models.py --seed 42 --n_jobs 8
"""

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prod S2AND models (SPECTER1 vs SPECTER2)")
    parser.add_argument(
        "--dataset",
        choices=["inventors_s2and", "mini"],
        default="inventors_s2and",
        help="Which dataset(s) to evaluate on (default: inventors_s2and)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1, matches transfer_experiment_internal)",
    )
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain models from scratch instead of loading prod pickles",
    )
    return parser


import numpy as np

from s2and.consts import DEFAULT_CHUNK_SIZE, FEATURIZER_VERSION, PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import Clusterer, PairwiseModeler
from s2and.serialization import load_pickle_with_verified_label_encoder_compat
from s2and.warnings_utils import suppress_sklearn_feature_name_warnings

# specter suffix -> production model pickle
# v1.1 was trained on specter1 features, v1.2 on specter2
MODELS = {
    "_specter.pickle": "production_model_v1.1.pickle",
    "_specter2.pkl": "production_model_v1.2.pickle",
}
specter_suffixes = list(MODELS.keys())


def resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    """Try preferred filename, then fallback, raising FileNotFoundError if neither exists."""
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing dataset file. Tried '{preferred_path}' and '{fallback_path}'.")


# feature categories (all except reference_features)
features_to_use = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    # "reference_features",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]

# nameless model: no name-based features (prevents model overreliance on names)
nameless_features_to_use = [
    f for f in features_to_use if f not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]

featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
nameless_featurization_info = FeaturizationInfo(
    features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION
)


def main() -> None:
    args = _build_parser().parse_args()
    suppress_sklearn_feature_name_warnings()
    n_jobs = args.n_jobs
    random_seed = args.seed
    train_flag = bool(args.train)
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)

    if args.dataset == "mini":
        data_original = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")
        # aminer has too much variance; medline is pairwise only
        datasets = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
    else:
        data_original = os.path.join(PROJECT_ROOT_PATH, "data")
        datasets = ["inventors_s2and"]

    print(f"Config: dataset={args.dataset}, seed={random_seed}, n_jobs={n_jobs}, train={train_flag}")
    print(f"Datasets: {datasets}")
    print()

    results = {}
    for specter_suffix in specter_suffixes:
        clusterer = None

        if not train_flag:
            model_name = MODELS[specter_suffix]
            model_path = os.path.join(PROJECT_ROOT_PATH, "data", model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Missing model artifact at {model_path}. "
                    "Either use --train to retrain, or place the model pickle in data/."
                )
            print(f"=== specter_suffix: {specter_suffix}, model: {model_name} ===")
            clusterer = load_pickle_with_verified_label_encoder_compat(model_path)["clusterer"]
            clusterer.use_cache = False
            clusterer.n_jobs = n_jobs
        else:
            print(f"=== specter_suffix: {specter_suffix}, training from scratch ===")

        cluster_metrics_all = []
        for dataset_name in datasets:
            print(f"-- dataset: {dataset_name} --")
            signatures_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_signatures.json", "signatures.json"
            )
            papers_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_papers.json", "papers.json"
            )
            clusters_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"
            )
            embeddings_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}{specter_suffix}", specter_suffix.lstrip("_")
            )
            anddata = ANDData(
                signatures=signatures_path,
                papers=papers_path,
                name=dataset_name,
                mode="train",
                specter_embeddings=embeddings_path,
                clusters=clusters_path,
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
                random_seed=random_seed,
                name_tuples="filtered",
            )

            if train_flag:
                train, val, _test = featurize(
                    anddata,
                    featurization_info,
                    n_jobs=n_jobs,
                    use_cache=False,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    nameless_featurizer_info=nameless_featurization_info,
                    nan_value=np.nan,
                )
                X_train, y_train, nameless_X_train = train
                X_val, y_val, nameless_X_val = val

                pairwise_modeler = PairwiseModeler(
                    n_iter=25,
                    estimator=None,
                    search_space=None,
                    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
                    random_state=random_seed,
                )
                pairwise_modeler.fit(X_train, y_train, X_val, y_val)

                nameless_pairwise_modeler = PairwiseModeler(
                    n_iter=25,
                    estimator=None,
                    search_space=None,
                    monotone_constraints=nameless_featurization_info.lightgbm_monotone_constraints,
                    random_state=random_seed,
                )
                nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)

                clusterer = Clusterer(
                    featurization_info,
                    pairwise_modeler.classifier,
                    n_jobs=n_jobs,
                    use_cache=False,
                    nameless_classifier=nameless_pairwise_modeler.classifier,
                    nameless_featurizer_info=nameless_featurization_info,
                    random_state=random_seed,
                    use_default_constraints_as_supervision=False,
                )
                clusterer.fit(anddata)

            if clusterer is None:
                raise RuntimeError("Clusterer was not initialized. Check --train flag and model artifact path.")

            cluster_metrics, _b3_metrics_per_signature = cluster_eval(
                anddata,
                clusterer,
                split="test",
                use_s2_clusters=False,
            )
            print(cluster_metrics)
            cluster_metrics_all.append(cluster_metrics)

        results[specter_suffix] = cluster_metrics_all
        b3s = [m["B3 (P, R, F1)"][-1] for m in cluster_metrics_all]
        print(f"B3 F1s: {b3s}, mean: {sum(b3s) / len(b3s):.3f}")
        print()

    # summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    result_specter1 = results["_specter.pickle"]
    result_specter2 = results["_specter2.pkl"]

    for i, dataset_name in enumerate(datasets):
        print(f"Performance with SPECTERv1 data, on {dataset_name} (B3): {result_specter1[i]['B3 (P, R, F1)']}")
        print(f"Performance with SPECTERv2 data, on {dataset_name} (B3): {result_specter2[i]['B3 (P, R, F1)']}")
        print()


if __name__ == "__main__":
    main()
