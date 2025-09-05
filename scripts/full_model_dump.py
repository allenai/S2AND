"""
This is the file to use to retrain a full prod model on all datasets.

Ai2 employee, the complete data (with augmented dataset and specter2 pickles) is: s3://ai2-s2-research/s2and/s2and_release_12_23_20/
"""

import os
from s2and.consts import FEATURIZER_VERSION, PROJECT_ROOT_PATH

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["S2AND_CACHE"] = os.path.join(PROJECT_ROOT_PATH, "data", ".feature_cache")

import numpy as np
import logging
import pickle
from typing import Optional, Dict, Any
from tqdm import tqdm

from s2and.consts import FEATURIZER_VERSION, PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster

from hyperopt import hp

logger = logging.getLogger("s2and")

SPECTER_SUFFIX = ["_specter.pickle", "_specter2.pkl"][0]
SIGNATURES_SUFFIX = ["_signatures.json", "_signatures_with_s2aff.json"][0]

USE_CACHE = False

search_space = {
    "eps": hp.uniform("choice", 0, 1),
    "linkage": hp.choice("linkage", ["average"]),
}

DATA_DIR = os.path.join(PROJECT_ROOT_PATH, "data")

FEATURES_TO_USE = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    # "reference_features",  # speed up prod
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


SOURCE_DATASET_NAMES = ["aminer", "arnetminer", "inspire", "kisti", "orcid", "pubmed", "qian", "zbmath"]
PAIRWISE_ONLY_DATASETS = {"medline", "augmented"}

BLOCK_TYPE = "s2"
N_TRAIN_PAIRS_SIZE = 100000
N_VAL_TEST_SIZE = 10000
N_ITER = 50
N_JOBS = 25
USE_NAMELESS_MODEL = True

USE_AUGMENTATION = True
if USE_AUGMENTATION:
    SOURCE_DATASET_NAMES.append("augmented")

NEGATIVE_ONE_FOR_NAN = False
if NEGATIVE_ONE_FOR_NAN:
    MONOTONE_CONSTRAINTS = None
    NAMELESS_MONOTONE_CONSTRAINTS = None
    NAN_VALUE = -1
else:
    MONOTONE_CONSTRAINTS = FEATURIZER_INFO.lightgbm_monotone_constraints
    NAMELESS_MONOTONE_CONSTRAINTS = NAMELESS_FEATURIZER_INFO.lightgbm_monotone_constraints
    NAN_VALUE = np.nan  # type: ignore


def main():
    """
    This script is used to train and dump a model trained on all the datasets
    """
    datasets: Dict[str, Dict[str, Any]] = {}
    for dataset_name in tqdm(SOURCE_DATASET_NAMES, desc="Processing datasets and fitting base models"):
        logger.info(f"processing dataset {dataset_name}")
        clusters_path: Optional[str] = None
        if dataset_name not in PAIRWISE_ONLY_DATASETS:
            clusters_path = os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json")
            train_pairs_path = None
            val_pairs_path = None
            test_pairs_path = None
        else:
            train_pairs_path = os.path.join(DATA_DIR, dataset_name, "train_pairs.csv")
            val_pairs_path = os.path.join(DATA_DIR, dataset_name, "val_pairs.csv")
            if not os.path.exists(val_pairs_path):
                val_pairs_path = None
            test_pairs_path = os.path.join(DATA_DIR, dataset_name, "test_pairs.csv")

        logger.info(f"loading dataset {dataset_name}")
        anddata = ANDData(
            signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + SIGNATURES_SUFFIX),
            papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + SPECTER_SUFFIX),
            clusters=clusters_path,
            block_type=BLOCK_TYPE,
            train_pairs=train_pairs_path,
            val_pairs=val_pairs_path,
            test_pairs=test_pairs_path,
            train_pairs_size=N_TRAIN_PAIRS_SIZE,
            val_pairs_size=N_VAL_TEST_SIZE,
            test_pairs_size=N_VAL_TEST_SIZE,
            preprocess=True,
        )

        logger.info(f"featurizing {dataset_name}")
        train, val, test = featurize(
            anddata,
            FEATURIZER_INFO,
            n_jobs=N_JOBS,
            use_cache=USE_CACHE,
            chunk_size=100,
            nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
            nan_value=NAN_VALUE,
        )
        # Assert these are not None to help mypy with type checking
        assert train is not None
        assert val is not None
        assert test is not None

        X_train, y_train, nameless_X_train = train
        X_val, y_val, nameless_X_val = val
        X_test, y_test, nameless_X_test = test

        dataset: Dict[str, Any] = {}
        dataset["anddata"] = anddata
        dataset["X_train"] = X_train
        dataset["y_train"] = y_train
        dataset["X_val"] = X_val
        dataset["y_val"] = y_val
        dataset["X_test"] = X_test
        dataset["y_test"] = y_test
        dataset["nameless_X_train"] = nameless_X_train
        dataset["nameless_X_val"] = nameless_X_val
        dataset["nameless_X_test"] = nameless_X_test
        dataset["name"] = anddata.name
        datasets[dataset_name] = dataset

    anddatas = [
        datasets[dataset_name]["anddata"]
        for dataset_name in SOURCE_DATASET_NAMES
        if dataset_name not in PAIRWISE_ONLY_DATASETS
    ]

    # Type: ignore to help mypy understand these are numpy arrays, not ANDData objects
    X_train = np.vstack([datasets[dataset_name]["X_train"] for dataset_name in SOURCE_DATASET_NAMES])  # type: ignore
    y_train = np.hstack([datasets[dataset_name]["y_train"] for dataset_name in SOURCE_DATASET_NAMES])  # type: ignore
    X_val = np.vstack(  # type: ignore
        [datasets[dataset_name]["X_val"] for dataset_name in SOURCE_DATASET_NAMES if dataset_name not in {"augmented"}]
    )
    y_val = np.hstack(  # type: ignore
        [datasets[dataset_name]["y_val"] for dataset_name in SOURCE_DATASET_NAMES if dataset_name not in {"augmented"}]
    )

    nameless_X_train = np.vstack([datasets[dataset_name]["nameless_X_train"] for dataset_name in SOURCE_DATASET_NAMES])  # type: ignore
    nameless_X_val = np.vstack(  # type: ignore
        [
            datasets[dataset_name]["nameless_X_val"]
            for dataset_name in SOURCE_DATASET_NAMES
            if dataset_name not in {"augmented"}
        ]
    )

    logger.info("fitting pairwise")
    union_classifier = PairwiseModeler(n_iter=N_ITER, monotone_constraints=MONOTONE_CONSTRAINTS)
    union_classifier.fit(X_train, y_train, X_val, y_val)

    nameless_union_classifier = None
    if USE_NAMELESS_MODEL:
        logger.info("nameless fitting pairwise for " + str(SOURCE_DATASET_NAMES))
        nameless_union_classifier = PairwiseModeler(
            n_iter=N_ITER,
            monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS,
        )
        nameless_union_classifier.fit(nameless_X_train, y_train, nameless_X_val, y_val)
        logger.info("nameless pairwise fit for " + str(SOURCE_DATASET_NAMES))

    logger.info("fitting clusterer for")
    union_clusterer = Clusterer(
        FEATURIZER_INFO,
        union_classifier.classifier,
        cluster_model=FastCluster(),
        search_space=search_space,
        n_jobs=N_JOBS,
        use_cache=USE_CACHE,
        nameless_classifier=nameless_union_classifier.classifier if nameless_union_classifier is not None else None,
        nameless_featurizer_info=NAMELESS_FEATURIZER_INFO if nameless_union_classifier is not None else None,
    )
    union_clusterer.fit(anddatas)
    print(
        "best clustering parameters:",
        union_clusterer.best_params,
    )

    models = {}
    models["clusterer"] = union_clusterer

    with open(
        f"full_union_model_script_dump_average_no_refs_{FEATURIZER_VERSION}.pickle",
        "wb",
    ) as _pickle_file:
        pickle.dump(models, _pickle_file)
    logger.info("Done.")


if __name__ == "__main__":
    main()
