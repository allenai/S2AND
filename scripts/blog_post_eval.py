"""
This script generates results that are needed for the blog post.
Here are all the commands that one needs to run:

python scripts/blog_post_eval.py --random_seed 42 --experiment_name baseline
python scripts/blog_post_eval.py --random_seed 42 --experiment_name qian_only --single_dataset qian
python scripts/blog_post_eval.py --random_seed 42 --experiment_name exclude_augmented --exclude_augmented
python scripts/blog_post_eval.py --random_seed 42 --experiment_name dont_use_rules --dont_use_rules
python scripts/blog_post_eval.py --random_seed 42 --experiment_name dont_use_nameless_model --dont_use_nameless_model
python scripts/blog_post_eval.py --random_seed 42 --experiment_name dont_use_specter --feature_groups_to_skip embedding_similarity
python scripts/blog_post_eval.py --random_seed 42 --experiment_name dont_use_name_counts --feature_groups_to_skip name_counts
"""

from typing import Optional, List, Dict, Any

import os
import json

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["S2AND_CACHE"] = os.path.join(CONFIG["internal_data_dir"], ".feature_cache")

import numpy as np
import logging
import argparse
import pickle
import json
import pandas as pd
from tqdm import tqdm

from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.consts import FEATURIZER_VERSION, NAME_COUNTS_PATH
from s2and.eval import claims_eval
from s2and.file_cache import cached_path

from hyperopt import hp

logger = logging.getLogger("s2and")

search_space = {
    "eps": hp.uniform("choice", 0, 1),
    "linkage": hp.choice("linkage", ["average"]),
}


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

BLOCK_TYPE = "s2"
N_TRAIN_PAIRS_SIZE = 100000
N_VAL_TEST_SIZE = 10000
N_ITER = 50


def main(
    experiment_name: str,
    dont_use_nameless_model: bool,
    dont_use_rules: bool,
    dont_use_monotone_constraints: bool,
    exclude_augmented: bool,
    single_dataset: str,
    feature_groups_to_skip: List[str],
    n_jobs: int,
    random_seed: int,
):
    """
    This script is used to train and dump a model trained on all the datasets
    """
    DATA_DIR = CONFIG["internal_data_dir"]
    USE_NAMELESS_MODEL = not dont_use_nameless_model
    USE_RULES = not dont_use_rules
    USE_AUGMENTATION = not exclude_augmented
    USE_MONOTONE_CONSTRAINTS = not dont_use_monotone_constraints
    N_JOBS = n_jobs

    for feature_group in feature_groups_to_skip:
        FEATURES_TO_USE.remove(feature_group)

    NAMELESS_FEATURES_TO_USE = [
        feature_name
        for feature_name in FEATURES_TO_USE
        if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
    ]

    FEATURIZER_INFO = FeaturizationInfo(features_to_use=FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION)
    NAMELESS_FEATURIZER_INFO = FeaturizationInfo(
        features_to_use=NAMELESS_FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION
    )

    MONOTONE_CONSTRAINTS = FEATURIZER_INFO.lightgbm_monotone_constraints
    NAMELESS_MONOTONE_CONSTRAINTS = NAMELESS_FEATURIZER_INFO.lightgbm_monotone_constraints

    SOURCE_DATASET_NAMES = ["aminer", "arnetminer", "inspire", "kisti", "orcid", "pubmed", "qian", "zbmath"]
    PAIRWISE_ONLY_DATASETS = {"medline", "augmented"}

    if USE_AUGMENTATION:
        SOURCE_DATASET_NAMES.append("augmented")

    if single_dataset != "":
        SOURCE_DATASET_NAMES = [single_dataset]

    datasets = {}
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
            preprocess=True,
            random_seed=random_seed if random_seed is not None else 1111,
        )

        logger.info(f"featurizing {dataset_name}")
        train, val, _ = featurize(
            anddata,
            FEATURIZER_INFO,
            n_jobs=N_JOBS,
            use_cache=True,
            chunk_size=100,
            nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
            nan_value=np.nan,
        )
        X_train, y_train, nameless_X_train = train
        X_val, y_val, nameless_X_val = val

        dataset: Dict[Any, Any] = {}
        dataset["anddata"] = anddata
        dataset["X_train"] = X_train
        dataset["y_train"] = y_train
        dataset["X_val"] = X_val
        dataset["y_val"] = y_val
        dataset["nameless_X_train"] = nameless_X_train
        dataset["nameless_X_val"] = nameless_X_val

        datasets[dataset_name] = dataset

    anddatas = [
        datasets[dataset_name]["anddata"]
        for dataset_name in SOURCE_DATASET_NAMES
        if dataset_name not in PAIRWISE_ONLY_DATASETS
    ]

    X_train = np.vstack([datasets[dataset_name]["X_train"] for dataset_name in SOURCE_DATASET_NAMES])
    y_train = np.hstack([datasets[dataset_name]["y_train"] for dataset_name in SOURCE_DATASET_NAMES])
    X_val = np.vstack(
        [datasets[dataset_name]["X_val"] for dataset_name in SOURCE_DATASET_NAMES if dataset_name not in {"augmented"}]
    )
    y_val = np.hstack(
        [datasets[dataset_name]["y_val"] for dataset_name in SOURCE_DATASET_NAMES if dataset_name not in {"augmented"}]
    )

    nameless_X_train = np.vstack([datasets[dataset_name]["nameless_X_train"] for dataset_name in SOURCE_DATASET_NAMES])
    nameless_X_val = np.vstack(
        [
            datasets[dataset_name]["nameless_X_val"]
            for dataset_name in SOURCE_DATASET_NAMES
            if dataset_name not in {"augmented"}
        ]
    )

    logger.info("fitting pairwise")
    union_classifier = PairwiseModeler(
        n_iter=N_ITER,
        monotone_constraints=MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
        random_state=random_seed if random_seed is not None else 42,
    )
    union_classifier.fit(X_train, y_train, X_val, y_val)

    nameless_union_classifier = None
    if USE_NAMELESS_MODEL:
        logger.info("nameless fitting pairwise for " + str(SOURCE_DATASET_NAMES))
        nameless_union_classifier = PairwiseModeler(
            n_iter=N_ITER,
            monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
            random_state=random_seed if random_seed is not None else 42,
        )
        nameless_union_classifier.fit(nameless_X_train, y_train, nameless_X_val, y_val)
        logger.info("nameless pairwise fit for " + str(SOURCE_DATASET_NAMES))

    logger.info("fitting clusterer for")
    clusterer = Clusterer(
        FEATURIZER_INFO,
        union_classifier.classifier,
        cluster_model=FastCluster(),
        search_space=search_space,
        n_jobs=N_JOBS,
        nameless_classifier=nameless_union_classifier.classifier if nameless_union_classifier is not None else None,
        nameless_featurizer_info=NAMELESS_FEATURIZER_INFO if nameless_union_classifier is not None else None,
        use_default_constraints_as_supervision=USE_RULES,
        use_cache=True,
        random_state=random_seed if random_seed is not None else 42,
    )
    clusterer.fit(anddatas)
    print(
        "best clustering parameters:",
        clusterer.best_params,
    )

    # now working on the blocks
    CLAIMS_DATA_DIR = os.path.join(CONFIG["internal_data_dir"], "claims")
    BLOCK_DATASETS_DIR = os.path.join(CLAIMS_DATA_DIR, "block_datasets")

    with open(os.path.join(CLAIMS_DATA_DIR, "claims_pairs_remapped.json")) as _json_file:
        claims_pairs = json.load(_json_file)
    logger.info("Claims pairs loaded")

    clusterer.batch_size = 10000000

    block_keys = sorted(
        filter(
            lambda x: not x.endswith(".json")
            and not x.endswith(".pickle")
            and not x.endswith(".py")
            and not x.endswith(".vscode")
            and not x.endswith(".csv"),
            os.listdir(BLOCK_DATASETS_DIR),
        ),
        key=lambda x: os.path.getsize(os.path.join(os.path.join(BLOCK_DATASETS_DIR, x), "claims_signatures.json")),
    )
    # these had errors when manually evaluating
    for block_key in ["t_xiao", "m_dagostino", "s_tunster", "n_smith"]:
        block_keys.remove(block_key)

    # let's only keep the first ~130 for speed purposes
    block_keys = block_keys[:130]

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

    results_dict = {}
    for block_key in tqdm(block_keys):
        results = {}
        block_dir = os.path.join(BLOCK_DATASETS_DIR, block_key)
        logger.info(f"Loading dataset {block_key}")
        claims_dataset = ANDData(
            signatures=os.path.join(block_dir, "claims_signatures.json"),
            papers=os.path.join(block_dir, "claims_papers.json"),
            mode="inference",
            specter_embeddings=os.path.join(block_dir, "claims_specter.pickle"),
            block_type="s2",
            name=block_key.replace(" ", "_"),
            n_jobs=n_jobs,
            load_name_counts=name_counts,
        )
        logger.info("Dataset loaded")

        result = claims_eval(
            claims_dataset,
            clusterer,
            claims_pairs,
            os.path.join(BLOCK_DATASETS_DIR, claims_dataset.name),
            output_shap=False,
            optional_name=experiment_name,
        )
        results[block_key.replace(" ", "_")] = result
        logger.info(f"Claims eval output: {result}")

        with open(
            os.path.join(
                BLOCK_DATASETS_DIR,
                claims_dataset.name,
                f"results_{experiment_name}.json",
            ),
            "w",
        ) as _json_file:
            json.dump(results, _json_file)
        results_dict.update(results)

    pd.DataFrame(results_dict).T.to_csv(os.path.join(BLOCK_DATASETS_DIR, f"{experiment_name}.csv"))


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
    parser.add_argument(
        "--dont_use_rules",
        action="store_true",
        help="Whether to use the rules model",
    )
    parser.add_argument(
        "--dont_use_monotone_constraints",
        action="store_true",
        help="Whether to use monotone constraints",
    )
    parser.add_argument(
        "--exclude_augmented",
        action="store_true",
        help="Whether to exclude augmented in experiments",
    )
    parser.add_argument("--single_dataset", type=str, default="", help="Which single dataset to use")
    parser.add_argument("--feature_groups_to_skip", nargs="+", default=[], type=str)
    parser.add_argument("--n_jobs", type=int, default=25, help="How many cpus to use")
    parser.add_argument("--random_seed", nargs="?", type=int)

    args = parser.parse_args()
    logger.info(args)
    main(
        args.experiment_name,
        args.dont_use_nameless_model,
        args.dont_use_rules,
        args.dont_use_monotone_constraints,
        args.exclude_augmented,
        args.single_dataset,
        args.feature_groups_to_skip,
        args.n_jobs,
        args.random_seed,
    )
