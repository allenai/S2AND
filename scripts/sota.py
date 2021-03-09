from typing import Dict, Any, Optional
import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "8"

import copy
import numpy as np
import pandas as pd
import argparse
import logging
import pickle

logger = logging.getLogger("s2and")

from tqdm import tqdm

os.environ["S2AND_CACHE"] = os.path.join(CONFIG["main_data_dir"], ".feature_cache")

from sklearn.cluster import DBSCAN

from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.eval import pairwise_eval, cluster_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, NAME_COUNTS_PATH
from s2and.file_cache import cached_path
from hyperopt import hp


search_space = {
    "eps": hp.uniform("choice", 0, 1),
}

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

DATA_DIR = CONFIG["main_data_dir"]

FEATURES_TO_USE = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "reference_features",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]

NAMELESS_FEATURES_TO_USE = [
    feature_name
    for feature_name in FEATURES_TO_USE
    if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]

FEATURIZER_INFO = FeaturizationInfo(features_to_use=FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION)
NAMELESS_FEATURIZER_INFO = FeaturizationInfo(
    features_to_use=NAMELESS_FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION
)

PAIRWISE_ONLY_DATASETS = {"medline"}
BLOCK_TYPE = "original"
N_TRAIN_PAIRS_SIZE = 100000
N_VAL_TEST_SIZE = 10000
N_ITER = 25
USE_CACHE = True
PREPROCESS = True


def sota_helper(dataset, experiment_name, random_seed, use_s2_clusters=False):

    dataset_name = dataset["name"]

    pairwise_metrics = pairwise_eval(
        dataset["X_test"],
        dataset["y_test"],
        dataset["pairwise_modeler"],
        os.path.join(DATA_DIR, "experiments", experiment_name, "sota", f"seed_{random_seed}", "figs"),
        f"{dataset_name}",
        FEATURIZER_INFO.get_feature_names(),
        nameless_classifier=dataset["nameless_pairwise_modeler"],
        nameless_X=dataset["nameless_X_test"],
        nameless_feature_names=NAMELESS_FEATURIZER_INFO.get_feature_names(),
    )

    if dataset_name not in PAIRWISE_ONLY_DATASETS:
        cluster_metrics, b3_metrics_per_signature = cluster_eval(
            dataset["anddata"],
            dataset["clusterer"],
            split="test",
            use_s2_clusters=use_s2_clusters,
        )
    else:
        cluster_metrics = {
            "B3 (P, R, F1)": (None, None, None),
            "Cluster (P, R F1)": (None, None, None),
            "Cluster Macro (P, R, F1)": (None, None, None),
            "Pred bigger ratio (mean, count)": (None, None),
            "True bigger ratio (mean, count)": (None, None),
        }
        b3_metrics_per_signature = None

    metrics = {"pairwise": pairwise_metrics, "cluster": cluster_metrics}
    logger.info(f"{dataset_name}_sota_: {metrics}")
    if not os.path.exists(
        os.path.join(DATA_DIR, "experiments", experiment_name, "sota", f"seed_{random_seed}", "metrics")
    ):
        os.makedirs(
            os.path.join(
                DATA_DIR,
                "experiments",
                experiment_name,
                "sota",
                f"seed_{random_seed}",
                "metrics",
            )
        )
    with open(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            f"{dataset_name}_sota.json",
        ),
        "w",
    ) as _json_file:
        json.dump(metrics, _json_file, indent=4)
    return pairwise_metrics, cluster_metrics, b3_metrics_per_signature


def main(
    experiment_name: str,
    dont_use_nameless_model: bool,
    n_jobs: int,
    dont_use_monotone_constraints: bool,
    linkage: str,
    use_dbscan: bool,
    negative_one_for_nan: bool,
    random_seed: int,
    inspire_split: int,
    inspire_only: bool,
    aminer_only: bool,
):
    USE_NAMELESS_MODEL = not dont_use_nameless_model
    N_JOBS = n_jobs
    USE_MONOTONE_CONSTRAINTS = not dont_use_monotone_constraints
    logger.info(
        (
            f"USE_NAMELESS_MODEL={USE_NAMELESS_MODEL}, "
            f"N_JOBS={N_JOBS}, "
            f"USE_MONOTONE_CONSTRAINTS={USE_MONOTONE_CONSTRAINTS}, "
            f"linkage={linkage}, "
            f"use_dbscan={use_dbscan}, "
            f"negative_one_for_nan={negative_one_for_nan}, "
            f"random_seed={random_seed}"
        )
    )

    if inspire_only:
        DATASET_NAMES = ["inspire"]
    elif aminer_only:
        DATASET_NAMES = ["aminer"]
    else:
        DATASET_NAMES = [
            "kisti",
            "pubmed",
            "medline",
        ]

    FIXED_BLOCK = ["aminer"]
    FIXED_SIGNATURE = ["inspire"]

    if negative_one_for_nan:
        MONOTONE_CONSTRAINTS = None
        NAMELESS_MONOTONE_CONSTRAINTS = None
        NAN_VALUE = -1
    else:
        MONOTONE_CONSTRAINTS = FEATURIZER_INFO.lightgbm_monotone_constraints
        NAMELESS_MONOTONE_CONSTRAINTS = NAMELESS_FEATURIZER_INFO.lightgbm_monotone_constraints
        NAN_VALUE = np.nan

    with open(cached_path(NAME_COUNTS_PATH), "rb") as f:
        (
            first_dict,
            last_dict,
            first_last_dict,
            last_first_initial_dict,
        ) = pickle.load(f)
    name_counts = {
        "first_dict": first_dict,
        "last_dict": last_dict,
        "first_last_dict": first_last_dict,
        "last_first_initial_dict": last_first_initial_dict,
    }
    logger.info("loaded name counts")

    datasets: Dict[str, Any] = {}

    for dataset_name in tqdm(DATASET_NAMES, desc="Processing datasets and fitting base models"):
        logger.info("")
        logger.info(f"processing dataset {dataset_name}")
        clusters_path: Optional[str] = None
        train_blocks: Optional[str] = None
        val_blocks: Optional[str] = None
        test_blocks: Optional[str] = None
        train_pairs_path: Optional[str] = None
        val_pairs_path: Optional[str] = None
        test_pairs_path: Optional[str] = None
        train_signatures: Optional[str] = None
        val_signatures: Optional[str] = None
        test_signatures: Optional[str] = None

        if dataset_name in FIXED_BLOCK:
            logger.info("FIXED BLOCK")
            train_blocks_fname: str = "train_keys.json"
            val_blocks_fname: str = "val_keys.json"
            test_blocks_fname: str = "test_keys.json"

            logger.info(f"File names, FIXED BLOCK {train_blocks_fname, val_blocks_fname, test_blocks_fname}")
            clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")
            train_blocks = os.path.join(DATA_DIR, dataset_name, train_blocks_fname)
            if not os.path.exists(os.path.join(DATA_DIR, dataset_name, val_blocks_fname)):
                val_blocks = None
            test_blocks = os.path.join(DATA_DIR, dataset_name, test_blocks_fname)

        elif dataset_name in FIXED_SIGNATURE:
            train_sign_fname: str = "train_keys_" + str(inspire_split) + ".json"
            val_sign_fname: str = "val_keys_" + str(inspire_split) + ".json"
            test_sign_fname: str = "test_keys_" + str(inspire_split) + ".json"

            logger.info(f"File names, FIXED_SIGNATURE {train_sign_fname, val_sign_fname, test_sign_fname}")
            clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")
            train_signatures = os.path.join(DATA_DIR, dataset_name, train_sign_fname)
            if not os.path.exists(os.path.join(DATA_DIR, dataset_name, val_sign_fname)):
                val_signatures = None
            test_signatures = os.path.join(DATA_DIR, dataset_name, test_sign_fname)

        elif dataset_name not in PAIRWISE_ONLY_DATASETS:
            logger.info("CLUSTER with random split")
            clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")

        else:
            logger.info("Pairwise model")
            train_pairs_path = os.path.join(DATA_DIR, dataset_name, "train_pairs.csv")
            val_pairs_path = os.path.join(DATA_DIR, dataset_name, "val_pairs.csv")
            if not os.path.exists(val_pairs_path):
                val_pairs_path = None
            test_pairs_path = os.path.join(DATA_DIR, dataset_name, "test_pairs.csv")

        logger.info(f"loading dataset {dataset_name}")

        if dataset_name == "inspire" or dataset_name == "kisti":
            unit_of_data_split = "signatures"
        else:
            unit_of_data_split = "blocks"

        if dataset_name == "kisti":
            train_ratio = 0.4
            val_ratio = 0.1
            test_ratio = 0.5
        else:
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1

        logger.info(f"ratios {train_ratio, val_ratio, test_ratio}")
        logger.info(f"block keys {train_blocks, val_blocks, test_blocks}")
        logger.info(f"signature keys {train_signatures, val_signatures, test_signatures}")

        anddata = ANDData(
            signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
            papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
            clusters=clusters_path,
            block_type=BLOCK_TYPE,
            train_pairs=train_pairs_path,
            val_pairs=val_pairs_path,
            test_pairs=test_pairs_path,
            train_pairs_size=N_TRAIN_PAIRS_SIZE,
            val_pairs_size=N_VAL_TEST_SIZE,
            test_pairs_size=N_VAL_TEST_SIZE,
            n_jobs=N_JOBS,
            load_name_counts=name_counts,
            preprocess=PREPROCESS,
            random_seed=random_seed,
            train_blocks=train_blocks,
            val_blocks=val_blocks,
            test_blocks=test_blocks,
            train_signatures=train_signatures,
            val_signatures=val_signatures,
            test_signatures=test_signatures,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            unit_of_data_split=unit_of_data_split,
        )
        logger.info(f"dataset {dataset_name} loaded")

        logger.info(f"featurizing {dataset_name}")
        train, val, test = featurize(anddata, FEATURIZER_INFO, n_jobs=N_JOBS, use_cache=USE_CACHE, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=NAMELESS_FEATURIZER_INFO, nan_value=NAN_VALUE)  # type: ignore
        X_train, y_train, nameless_X_train = train
        X_val, y_val, nameless_X_val = val
        assert test is not None
        X_test, y_test, nameless_X_test = test
        logger.info(f"dataset {dataset_name} featurized")

        pairwise_modeler: Optional[PairwiseModeler] = None
        nameless_pairwise_modeler = None
        cluster: Optional[Clusterer] = None
        logger.info(f"fitting pairwise for {dataset_name}")
        pairwise_modeler = PairwiseModeler(
            n_iter=N_ITER,
            monotone_constraints=MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
            random_state=random_seed,
        )
        pairwise_modeler.fit(X_train, y_train, X_val, y_val)
        logger.info(f"pairwise fit for {dataset_name}")

        if USE_NAMELESS_MODEL:
            logger.info(f"nameless fitting pairwise for {dataset_name}")
            nameless_pairwise_modeler = PairwiseModeler(
                n_iter=N_ITER,
                monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
                random_state=random_seed,
            )
            nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)
            logger.info(f"nameless pairwise fit for {dataset_name}")

        distances_for_sparsity = [1 - pred[1] for pred in pairwise_modeler.predict_proba(X_train)]
        threshold = np.percentile(distances_for_sparsity, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        logger.info(f"Thresholds {threshold}")

        if dataset_name not in PAIRWISE_ONLY_DATASETS:
            logger.info(f"fitting clusterer for {dataset_name}")
            cluster = Clusterer(
                FEATURIZER_INFO,
                pairwise_modeler.classifier,
                cluster_model=FastCluster(linkage=linkage)
                if not use_dbscan
                else DBSCAN(min_samples=1, metric="precomputed"),
                search_space=search_space,
                n_jobs=N_JOBS,
                use_cache=USE_CACHE,
                nameless_classifier=nameless_pairwise_modeler.classifier
                if nameless_pairwise_modeler is not None
                else None,
                nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
                random_state=random_seed,
                use_default_constraints_as_supervision=False,
            )
            cluster.fit(anddata)
            logger.info(f"clusterer fit for {dataset_name}")
            logger.info(f"{dataset_name} best clustering parameters: " + str(cluster.best_params))

        dataset: Dict[str, Any] = {}
        dataset["anddata"] = anddata
        dataset["X_train"] = X_train
        dataset["y_train"] = y_train
        dataset["X_val"] = X_val
        dataset["y_val"] = y_val
        dataset["X_test"] = X_test
        dataset["y_test"] = y_test
        dataset["pairwise_modeler"] = pairwise_modeler
        dataset["nameless_X_train"] = nameless_X_train
        dataset["nameless_X_val"] = nameless_X_val
        dataset["nameless_X_test"] = nameless_X_test
        dataset["nameless_pairwise_modeler"] = nameless_pairwise_modeler
        dataset["clusterer"] = cluster
        dataset["name"] = anddata.name
        datasets[dataset_name] = dataset

    logger.info("")
    logger.info("making evaluation grids")

    b3_f1_grid = [["" for j in range(len(DATASET_NAMES) + 1)] for i in range(len(DATASET_NAMES) + 1)]

    for i in range(max(len(DATASET_NAMES), len(DATASET_NAMES))):
        if i < len(DATASET_NAMES):
            b3_f1_grid[0][i + 1] = DATASET_NAMES[i]
        if i < len(DATASET_NAMES):
            b3_f1_grid[i + 1][0] = DATASET_NAMES[i]

    pairwise_auroc_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid
    pairwise_f1_classification_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid
    pairwise_average_precisision_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid
    pairwise_macro_f1_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid

    # transfer of individual models
    logger.info("starting individual model evaluation")
    for _, source_dataset in tqdm(datasets.items(), desc="Evaluating individual models"):
        logger.info("")
        logger.info(f"evaluating source {source_dataset['name']} target {source_dataset['name']}")
        pairwise_metrics, cluster_metrics, _ = sota_helper(source_dataset, experiment_name, random_seed)
        b3_f1_grid[DATASET_NAMES.index(source_dataset["name"]) + 1][
            DATASET_NAMES.index(source_dataset["name"]) + 1
        ] = cluster_metrics["B3 (P, R, F1)"][2]
        pairwise_macro_f1_grid[DATASET_NAMES.index(source_dataset["name"]) + 1][
            DATASET_NAMES.index(source_dataset["name"]) + 1
        ] = cluster_metrics["Cluster Macro (P, R, F1)"][2]
        pairwise_auroc_grid[DATASET_NAMES.index(source_dataset["name"]) + 1][
            DATASET_NAMES.index(source_dataset["name"]) + 1
        ] = pairwise_metrics["AUROC"]
        pairwise_f1_classification_grid[DATASET_NAMES.index(source_dataset["name"]) + 1][
            DATASET_NAMES.index(source_dataset["name"]) + 1
        ] = pairwise_metrics["F1"]
        pairwise_average_precisision_grid[DATASET_NAMES.index(source_dataset["name"]) + 1][
            DATASET_NAMES.index(source_dataset["name"]) + 1
        ] = pairwise_metrics["Average Precision"]
    logger.info("finished individual model evaluation")

    # union
    logger.info("")
    logger.info("writing results to disk")
    print("B3 F1:")
    b3_df = pd.DataFrame(b3_f1_grid)
    print(b3_df)

    print()

    print("Pairwise Macro F1 (cluster):")
    pairwise_macro_f1_df = pd.DataFrame(pairwise_macro_f1_grid)
    print(pairwise_macro_f1_df)

    print()

    print("Pairwise AUROC:")
    pairwise_df = pd.DataFrame(pairwise_auroc_grid)
    print(pairwise_df)

    print()

    print("Pairwise classification F1:")
    pairwise_classification_f1_df = pd.DataFrame(pairwise_f1_classification_grid)
    print(pairwise_classification_f1_df)

    print()

    print("Pairwise AP:")
    pairwise_ap_df = pd.DataFrame(pairwise_average_precisision_grid)
    print(pairwise_ap_df)

    print()

    with open(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "full_grid.json",
        ),
        "w",
    ) as _json_file:
        json.dump(
            {
                "b3": b3_f1_grid,
                "pairwisef1": pairwise_macro_f1_grid,
                "auroc": pairwise_auroc_grid,
                "classificationf1": pairwise_f1_classification_grid,
                "averageprecision": pairwise_average_precisision_grid,
            },
            _json_file,
        )

    b3_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "b3.csv",
        ),
        index=False,
    )

    pairwise_macro_f1_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "pair_macro_f1_cluster.csv",
        ),
        index=False,
    )
    pairwise_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "pairwise_auc.csv",
        ),
        index=False,
    )

    pairwise_classification_f1_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "classification_f1.csv",
        ),
        index=False,
    )

    pairwise_ap_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            "sota",
            f"seed_{random_seed}",
            "metrics",
            "average_precision.csv",
        ),
        index=False,
    )

    return (
        b3_f1_grid,
        pairwise_macro_f1_grid,
        pairwise_auroc_grid,
        pairwise_f1_classification_grid,
        pairwise_average_precisision_grid,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="The name of the experiment (for writing results to disk)",
    )
    parser.add_argument(
        "--dont_use_nameless_model",
        action="store_true",
        help="Whether to use the nameless model",
    )
    parser.add_argument("--n_jobs", type=int, default=10, help="How many cpus to use")
    parser.add_argument(
        "--dont_use_monotone_constraints",
        action="store_true",
        help="Whether to use monotone constraints",
    )
    parser.add_argument("--linkage", type=str, default="average", help="What linkage function to use")
    parser.add_argument(
        "--use_dbscan",
        action="store_true",
        help="Whether to use DBSCAN instead of fastcluster",
    )
    parser.add_argument(
        "--negative_one_for_nan",
        action="store_true",
        help="Whether to use -1 as nans and no monotone constraints",
    )
    parser.add_argument("--random_seed", nargs="+", default=[1], type=int)

    parser.add_argument("--inspire_only", action="store_true")

    parser.add_argument("--aminer_only", action="store_true")

    parser.add_argument("--inspire_split", default=0, help="which split from inspire to evaluate 0/1/2", type=int)

    args = parser.parse_args()
    logger.info(args)

    multi_b3_grid = []
    multi_pairwise_macro_f1_grid = []

    multi_pairwise_auroc_grid = []
    multi_pairwise_class_f1_grid = []
    multi_average_precision_grid = []

    for seed in args.random_seed:
        print("run with seed:", seed)
        b3_f1_grid, pairwise_macro_f1, pairwise_auroc_grid, pairwise_class_f1, pairwise_average_precision = main(
            args.experiment_name,
            args.dont_use_nameless_model,
            args.n_jobs,
            args.dont_use_monotone_constraints,
            args.linkage,
            args.use_dbscan,
            args.negative_one_for_nan,
            int(seed),
            args.inspire_split,
            args.inspire_only,
            args.aminer_only,
        )
        multi_b3_grid.append(b3_f1_grid)
        multi_pairwise_macro_f1_grid.append(pairwise_macro_f1)

        multi_pairwise_auroc_grid.append(pairwise_auroc_grid)
        multi_pairwise_class_f1_grid.append(pairwise_class_f1)
        multi_average_precision_grid.append(pairwise_average_precision)

    average_b3_f1_grid = copy.deepcopy(multi_b3_grid[0])
    average_pairwise_f1_macro_grid = copy.deepcopy(multi_pairwise_macro_f1_grid[0])

    average_pairwise_auroc_grid = copy.deepcopy(multi_pairwise_auroc_grid[0])
    average_pairwise_class_f1_grid = copy.deepcopy(multi_pairwise_class_f1_grid[0])
    average_class_ap_grid = copy.deepcopy(multi_average_precision_grid[0])

    index = -1
    for b3, macro_f1, pairwise, class_f1, ap in zip(
        multi_b3_grid,
        multi_pairwise_macro_f1_grid,
        multi_pairwise_auroc_grid,
        multi_pairwise_class_f1_grid,
        multi_average_precision_grid,
    ):
        index += 1
        if index == 0:
            continue
        for i in range(1, len(average_b3_f1_grid)):
            for j in range(1, len(average_b3_f1_grid[0])):
                if i == j:
                    if b3[i][j] is not None:
                        average_b3_f1_grid[i][j] += b3[i][j]
                    if macro_f1[i][j] is not None:
                        average_pairwise_f1_macro_grid[i][j] += macro_f1[i][j]
                    average_pairwise_auroc_grid[i][j] += pairwise[i][j]
                    average_pairwise_class_f1_grid[i][j] += class_f1[i][j]
                    average_class_ap_grid[i][j] += ap[i][j]

    for i in range(1, len(average_b3_f1_grid)):
        for j in range(1, len(average_b3_f1_grid[0])):
            if i == j:
                if average_b3_f1_grid[i][j] is not None:
                    average_b3_f1_grid[i][j] = average_b3_f1_grid[i][j] / len(multi_b3_grid)
                if average_pairwise_f1_macro_grid[i][j] is not None:
                    average_pairwise_f1_macro_grid[i][j] = average_pairwise_f1_macro_grid[i][j] / len(
                        multi_pairwise_macro_f1_grid
                    )
                average_pairwise_auroc_grid[i][j] = average_pairwise_auroc_grid[i][j] / len(multi_pairwise_auroc_grid)
                average_pairwise_class_f1_grid[i][j] = average_pairwise_class_f1_grid[i][j] / len(
                    multi_pairwise_class_f1_grid
                )
                average_class_ap_grid[i][j] = average_class_ap_grid[i][j] / len(multi_average_precision_grid)

    print("Average B3 F1:")
    b3_df = pd.DataFrame(average_b3_f1_grid)
    print(b3_df)
    print()
    print("Average pairwise macro F1:")
    pair_macro_f1_df = pd.DataFrame(average_pairwise_f1_macro_grid)
    print(pair_macro_f1_df)
    print()
    print("Average Pairwise AUROC:")
    pairwise_df = pd.DataFrame(average_pairwise_auroc_grid)
    print(pairwise_df)
    print()
    print("Average Pairwise classification F1:")
    pairwise_class_f1_df = pd.DataFrame(average_pairwise_class_f1_grid)
    print(pairwise_class_f1_df)
    print()
    print("Average Precision:")
    pairwise_ap_df = pd.DataFrame(average_class_ap_grid)
    print(pairwise_ap_df)

    with open(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_full_grid.json"),
        "w",
    ) as _json_file:
        json.dump(
            {
                "b3": average_b3_f1_grid,
                "pair_macro_f1": average_pairwise_f1_macro_grid,
                "auroc": average_pairwise_auroc_grid,
                "class_f1": average_pairwise_class_f1_grid,
                "ap": average_class_ap_grid,
            },
            _json_file,
        )

    b3_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_b3.csv"),
        index=False,
    )
    pair_macro_f1_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_pairwise_macro(cluster)_f1.csv"),
        index=False,
    )
    pairwise_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_pairwise_auc.csv"),
        index=False,
    )
    pairwise_class_f1_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_class_f1.csv"),
        index=False,
    )
    pairwise_ap_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "sota", "average_mean_precision.csv"),
        index=False,
    )
