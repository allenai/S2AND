import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

DATA_DIR = CONFIG["main_data_dir"]
if not os.path.exists(DATA_DIR):
    PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT_PATH, "data")

os.environ["S2AND_CACHE"] = os.path.join(DATA_DIR, ".feature_cache")
os.environ["OMP_NUM_THREADS"] = "8"

import copy
import argparse
import logging
import pickle
from typing import Dict, Any, Optional, List

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.cluster import DBSCAN
from sklearn.linear_model import BayesianRidge, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline

from s2and.data import PDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.eval import pairwise_eval, cluster_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, NAME_COUNTS_PATH
from s2and.file_cache import cached_path
from hyperopt import hp

logger = logging.getLogger("s2and")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


N_VAL_TEST_SIZE = 10000
N_ITER = 25


def transfer_helper(
    source_dataset,
    target_dataset,
    experiment_name,
    random_seed,
    featurizer_info,
    nameless_featurizer_info,
    skip_shap=False,
):
    source_name = source_dataset["name"]
    target_name = target_dataset["name"]

    pairwise_metrics = pairwise_eval(
        target_dataset["X_test"],
        target_dataset["y_test"],
        source_dataset["pairwise_modeler"],
        os.path.join(DATA_DIR, "experiments", experiment_name, f"seed_{random_seed}", "figs"),
        f"{source_name}_to_{target_name}",
        featurizer_info.get_feature_names(),
        nameless_classifier=source_dataset["nameless_pairwise_modeler"],
        nameless_X=target_dataset["nameless_X_test"],
        nameless_feature_names=nameless_featurizer_info.get_feature_names(),
        skip_shap=skip_shap,
    )

    cluster_metrics, b3_metrics_per_signature = cluster_eval(
        target_dataset["PDData"],
        source_dataset["clusterer"],
        split="test",
    )

    metrics = {"pairwise": pairwise_metrics, "cluster": cluster_metrics}
    logger.info(f"{source_name}_to_{target_name}: {metrics}")
    with open(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            f"{source_name}_to_{target_name}.json",
        ),
        "w",
    ) as _json_file:
        json.dump(metrics, _json_file, indent=4)

    return pairwise_metrics, cluster_metrics, b3_metrics_per_signature


def main(
    experiment_name: str,
    use_nameless_model: bool,
    n_jobs: int,
    dont_use_monotone_constraints: bool,
    linkage: str,
    use_dbscan: bool,
    random_seed: int,
    n_train_pairs_size: int,
    feature_groups_to_skip: List[str],
    use_linear_pairwise_model: bool,
    use_cache: bool,
):
    if not os.path.exists(os.path.join(DATA_DIR, "experiments", experiment_name, f"seed_{random_seed}", "metrics")):
        os.makedirs(
            os.path.join(
                DATA_DIR,
                "experiments",
                experiment_name,
                f"seed_{random_seed}",
                "metrics",
            )
        )

    USE_NAMELESS_MODEL = use_nameless_model
    N_JOBS = n_jobs
    USE_MONOTONE_CONSTRAINTS = not dont_use_monotone_constraints
    N_TRAIN_PAIRS_SIZE = n_train_pairs_size
    USE_CACHE = use_cache
    logger.info(
        (
            f"USE_NAMELESS_MODEL={USE_NAMELESS_MODEL}, "
            f"N_JOBS={N_JOBS}, "
            f"USE_MONOTONE_CONSTRAINTS={USE_MONOTONE_CONSTRAINTS}, "
            f"linkage={linkage}, "
            f"use_dbscan={use_dbscan}, "
            f"random_seed={random_seed}, "
            f"N_TRAIN_PAIRS_SIZE={N_TRAIN_PAIRS_SIZE}, "
            f"feature_groups_to_skip={feature_groups_to_skip}, "
            f"use_linear_pairwise_model={use_linear_pairwise_model}, "
            f"USE_CACHE={USE_CACHE}, "
        )
    )

    FEATURES_TO_USE = [
        "author_similarity",
        "venue_similarity",
        "year_diff",
        "title_similarity",
        "abstract_similarity",
        "paper_quality",
    ]
    for feature_group in feature_groups_to_skip:
        FEATURES_TO_USE.remove(feature_group)

    NAMELESS_FEATURES_TO_USE = [
        feature_name
        for feature_name in FEATURES_TO_USE
        if feature_name not in {"title_similarity", "abstract_similarity"}
    ]

    FEATURIZER_INFO = FeaturizationInfo(features_to_use=FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION)
    NAMELESS_FEATURIZER_INFO = FeaturizationInfo(
        features_to_use=NAMELESS_FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION
    )

    SOURCE_DATASET_NAMES = ["s2_papers"]

    TARGET_DATASET_NAMES = ["s2_papers"]

    MONOTONE_CONSTRAINTS = FEATURIZER_INFO.lightgbm_monotone_constraints
    NAMELESS_MONOTONE_CONSTRAINTS = NAMELESS_FEATURIZER_INFO.lightgbm_monotone_constraints
    NAN_VALUE = np.nan

    cluster_search_space: Dict[str, Any] = {
        "eps": hp.uniform("choice", 0, 1),
    }
    pairwise_search_space: Optional[Dict[str, Any]] = None
    estimator: Any = None
    if use_linear_pairwise_model:
        estimator = make_pipeline(
            StandardScaler(),
            IterativeImputer(
                max_iter=20,
                random_state=random_seed,
                estimator=BayesianRidge(),
                skip_complete=True,
                add_indicator=True,
                n_nearest_features=10,
                verbose=0,
            ),
            LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10],
                solver="saga",
                random_state=random_seed,
                n_jobs=N_JOBS,
                verbose=0,
                max_iter=10000,
                tol=1e-3,
            ),
        )
        pairwise_search_space = {}

    DATASETS_TO_TRAIN = set(SOURCE_DATASET_NAMES).union(set(TARGET_DATASET_NAMES))

    logger.info("starting transfer experiment main, loading name counts")
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
    for dataset_name in tqdm(DATASETS_TO_TRAIN, desc="Processing datasets and fitting base models"):
        logger.info("")
        logger.info(f"processing dataset {dataset_name}")

        clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")
        train_pairs_path = None
        val_pairs_path = None
        test_pairs_path = None

        logger.info(f"loading dataset {dataset_name}")
        pddata = PDData(
            papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=None,
            clusters=clusters_path,
            train_pairs=train_pairs_path,
            val_pairs=val_pairs_path,
            test_pairs=test_pairs_path,
            train_pairs_size=np.maximum(N_TRAIN_PAIRS_SIZE, 100000),
            val_pairs_size=N_VAL_TEST_SIZE,
            test_pairs_size=N_VAL_TEST_SIZE,
            n_jobs=N_JOBS,
            load_name_counts=name_counts,
            random_seed=random_seed,
        )
        logger.info(f"dataset {dataset_name} loaded")

        logger.info(f"featurizing {dataset_name}")
        train, val, test = featurize(pddata, FEATURIZER_INFO, n_jobs=N_JOBS, use_cache=USE_CACHE, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=NAMELESS_FEATURIZER_INFO, nan_value=NAN_VALUE)  # type: ignore
        X_train, y_train, nameless_X_train = train
        # if we sampled more training pairs than required, then we downsample
        if len(y_train) > N_TRAIN_PAIRS_SIZE:
            np.random.seed(random_seed)
            subset_indices = np.random.choice(len(y_train), size=N_TRAIN_PAIRS_SIZE, replace=False)
            X_train = X_train[subset_indices, :]
            if nameless_X_train is not None:
                nameless_X_train = nameless_X_train[subset_indices, :]
            y_train = y_train[subset_indices]
        X_val, y_val, nameless_X_val = val
        assert test is not None
        X_test, y_test, nameless_X_test = test
        logger.info(f"dataset {dataset_name} featurized")

        pairwise_modeler: Optional[PairwiseModeler] = None
        nameless_pairwise_modeler = None
        cluster: Optional[Clusterer] = None
        if dataset_name in SOURCE_DATASET_NAMES:
            logger.info(f"fitting pairwise for {dataset_name}")
            pairwise_modeler = PairwiseModeler(
                n_iter=N_ITER,
                estimator=estimator,
                search_space=pairwise_search_space,
                monotone_constraints=MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
                random_state=random_seed,
            )
            pairwise_modeler.fit(X_train, y_train, X_val, y_val)
            logger.info(f"pairwise fit for {dataset_name}")

            if USE_NAMELESS_MODEL:
                logger.info(f"nameless fitting pairwise for {dataset_name}")
                nameless_pairwise_modeler = PairwiseModeler(
                    n_iter=N_ITER,
                    estimator=estimator,
                    search_space=pairwise_search_space,
                    monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
                    random_state=random_seed,
                )
                nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)
                logger.info(f"nameless pairwise fit for {dataset_name}")

            distances_for_sparsity = [1 - pred[1] for pred in pairwise_modeler.predict_proba(X_train)]
            threshold = np.percentile(distances_for_sparsity, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            logger.info(f"Thresholds {threshold}")

            logger.info(f"fitting clusterer for {dataset_name}")
            cluster = Clusterer(
                FEATURIZER_INFO,
                pairwise_modeler.classifier,
                cluster_model=FastCluster(linkage=linkage)
                if not use_dbscan
                else DBSCAN(min_samples=1, metric="precomputed"),
                search_space=cluster_search_space,
                n_jobs=N_JOBS,
                use_cache=USE_CACHE,
                nameless_classifier=nameless_pairwise_modeler.classifier
                if nameless_pairwise_modeler is not None
                else None,
                nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
                random_state=random_seed,
                use_default_constraints_as_supervision=False,
            )
            cluster.fit(pddata)
            logger.info(f"clusterer fit for {dataset_name}")
            logger.info(f"{dataset_name} best clustering parameters: " + str(cluster.best_params))

        dataset: Dict[str, Any] = {}
        dataset["PDData"] = pddata
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
        dataset["name"] = pddata.name
        datasets[dataset_name] = dataset

    logger.info("")
    logger.info("making evaluation grids")
    b3_f1_grid = [["" for j in range(len(TARGET_DATASET_NAMES) + 1)] for i in range(len(SOURCE_DATASET_NAMES) + 1)]

    for i in range(max(len(TARGET_DATASET_NAMES), len(SOURCE_DATASET_NAMES))):
        if i < len(TARGET_DATASET_NAMES):
            b3_f1_grid[0][i + 1] = TARGET_DATASET_NAMES[i]
        if i < len(SOURCE_DATASET_NAMES):
            b3_f1_grid[i + 1][0] = SOURCE_DATASET_NAMES[i]

    pairwise_auroc_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid
    true_bigger_ratios_and_counts_grid = copy.deepcopy(b3_f1_grid)
    pred_bigger_ratios_and_counts_grid = copy.deepcopy(b3_f1_grid)

    logger.info("starting individual model evaluation")
    for _, source_dataset in tqdm(datasets.items(), desc="Evaluating individual models"):
        for _, target_dataset in datasets.items():
            if not (source_dataset["name"] in SOURCE_DATASET_NAMES) or (
                not target_dataset["name"] in TARGET_DATASET_NAMES
            ):
                continue

            logger.info("")
            logger.info(f"evaluating source {source_dataset['name']} target {target_dataset['name']}")
            pairwise_metrics, cluster_metrics, _ = transfer_helper(
                source_dataset,
                target_dataset,
                experiment_name,
                random_seed,
                FEATURIZER_INFO,
                NAMELESS_FEATURIZER_INFO,
                skip_shap=use_linear_pairwise_model,  # skip SHAP if not using default model
            )

            b3_f1_grid[SOURCE_DATASET_NAMES.index(source_dataset["name"]) + 1][
                TARGET_DATASET_NAMES.index(target_dataset["name"]) + 1
            ] = cluster_metrics["B3 (P, R, F1)"][2]
            pairwise_auroc_grid[SOURCE_DATASET_NAMES.index(source_dataset["name"]) + 1][
                TARGET_DATASET_NAMES.index(target_dataset["name"]) + 1
            ] = pairwise_metrics["AUROC"]
            true_bigger_ratios_and_counts_grid[SOURCE_DATASET_NAMES.index(source_dataset["name"]) + 1][
                TARGET_DATASET_NAMES.index(target_dataset["name"]) + 1
            ] = cluster_metrics["True bigger ratio (mean, count)"]
            pred_bigger_ratios_and_counts_grid[SOURCE_DATASET_NAMES.index(source_dataset["name"]) + 1][
                TARGET_DATASET_NAMES.index(target_dataset["name"]) + 1
            ] = cluster_metrics["Pred bigger ratio (mean, count)"]
            logger.info(f"finished evaluating source {source_dataset['name']} target {target_dataset['name']}")
    logger.info("finished individual model evaluation")

    logger.info("")
    logger.info("writing results to disk")
    print("B3 F1:")
    b3_df = pd.DataFrame(b3_f1_grid)
    print(b3_df)

    print()

    print("Pairwise AUROC:")
    pairwise_df = pd.DataFrame(pairwise_auroc_grid)
    print(pairwise_df)

    print()

    print("True bigger:")
    true_bigger_df = pd.DataFrame(true_bigger_ratios_and_counts_grid)
    print(true_bigger_df)

    print()

    print("Pred bigger:")
    pred_bigger_df = pd.DataFrame(pred_bigger_ratios_and_counts_grid)
    print(pred_bigger_df)

    with open(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            "full_grid.json",
        ),
        "w",
    ) as _json_file:
        json.dump(
            {
                "b3": b3_f1_grid,
                "auroc": pairwise_auroc_grid,
                "true_bigger": true_bigger_ratios_and_counts_grid,
                "pred_bigger": pred_bigger_ratios_and_counts_grid,
            },
            _json_file,
        )

    b3_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            "b3.csv",
        ),
        index=False,
    )
    pairwise_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            "pairwise.csv",
        ),
        index=False,
    )
    true_bigger_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            "true_bigger.csv",
        ),
        index=False,
    )
    pred_bigger_df.to_csv(
        os.path.join(
            DATA_DIR,
            "experiments",
            experiment_name,
            f"seed_{random_seed}",
            "metrics",
            "pred_bigger.csv",
        ),
        index=False,
    )

    logger.info("transfer experiment script done")

    return b3_f1_grid, pairwise_auroc_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="The name of the experiment (for writing results to disk)",
    )
    parser.add_argument(
        "--use_nameless_model",
        action="store_true",
        help="Whether to use the nameless model",
    )
    parser.add_argument("--n_jobs", type=int, default=1, help="How many cpus to use")
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

    parser.add_argument("--random_seed", nargs="+", default=[1], type=int)
    parser.add_argument("--n_train_pairs_size", type=int, default=100000, help="How many training pairs per dataset")
    parser.add_argument("--feature_groups_to_skip", nargs="+", default=[], type=str)
    parser.add_argument(
        "--use_linear_pairwise_model", action="store_true", help="Whether to use a LogisticRegression pairwise model"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use this flag to enable the cache; cache makes things faster but uses a *lot* of ram",
    )

    args = parser.parse_args()
    logger.info(args)

    multi_b3_grid = []
    multi_pairwise_auroc_grid = []

    for seed in args.random_seed:
        print("run with seed:", seed)
        b3_f1_grid, pairwise_auroc_grid = main(
            args.experiment_name,
            args.use_nameless_model,
            args.n_jobs,
            args.dont_use_monotone_constraints,
            args.linkage,
            args.use_dbscan,
            int(seed),
            args.n_train_pairs_size,
            args.feature_groups_to_skip,
            args.use_linear_pairwise_model,
            args.use_cache,
        )
        multi_b3_grid.append(b3_f1_grid)
        multi_pairwise_auroc_grid.append(pairwise_auroc_grid)

    average_b3_f1_grid = copy.deepcopy(multi_b3_grid[0])
    average_pairwise_auroc_grid = copy.deepcopy(multi_pairwise_auroc_grid[0])

    index = -1
    for b3, pairwise in zip(multi_b3_grid, multi_pairwise_auroc_grid):
        index += 1
        if index == 0:
            continue
        for i in range(1, len(average_b3_f1_grid)):
            for j in range(1, len(average_b3_f1_grid[0])):
                if b3[i][j] is not None and b3[i][j] != "":
                    average_b3_f1_grid[i][j] += b3[i][j]
                if pairwise[i][j] is not None and pairwise[i][j] != "":
                    average_pairwise_auroc_grid[i][j] += pairwise[i][j]

    for i in range(1, len(average_b3_f1_grid)):
        for j in range(1, len(average_b3_f1_grid[0])):
            if average_b3_f1_grid[i][j] is not None and average_b3_f1_grid[i][j] != "":
                average_b3_f1_grid[i][j] = average_b3_f1_grid[i][j] / len(multi_b3_grid)
            if average_pairwise_auroc_grid[i][j] is not None and average_pairwise_auroc_grid[i][j] != "":
                average_pairwise_auroc_grid[i][j] = average_pairwise_auroc_grid[i][j] / len(multi_pairwise_auroc_grid)

    print("Average B3 F1:")
    b3_df = pd.DataFrame(average_b3_f1_grid)
    print(b3_df)
    print()
    print("Average Pairwise AUROC:")
    pairwise_df = pd.DataFrame(average_pairwise_auroc_grid)
    print(pairwise_df)
    with open(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "average_full_grid.json"),
        "w",
    ) as _json_file:
        json.dump({"b3": average_b3_f1_grid, "auroc": average_pairwise_auroc_grid}, _json_file)

    b3_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "average_b3.csv"),
        index=False,
    )
    pairwise_df.to_csv(
        os.path.join(DATA_DIR, "experiments", args.experiment_name, "average_pairwise.csv"),
        index=False,
    )
