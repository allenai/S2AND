"""
Results (s2aff is missing but it made things worse)
-------
                   aminer  arnetminer  inspire kisti   orcid  pubmed   qian   zbmath
specterV1           0.703       0.881    0.934  0.951  0.965   0.936  0.924   0.905
specterV2           0.743       0.897    0.931  0.961  0.966   0.920  0.948   0.905
+ filtered          0.747       0.897    0.934  0.959  0.964   0.920  0.949   0.908  
+ no extra prefix   0.414       0.366    0.545  0.321  0.678   0.380  0.288   0.500
"""

import os
import json

# this is from when we updated S2AND with SPECTER2 and also with S2AFF outputs
SPECTER_SUFFIX = ["_specter.pickle", "_specter2.pkl"][1]
SIGNATURES_SUFFIX = ["_signatures.json", "_signatures_with_s2aff.json"][0]
NAME_TUPLES_VARIANT = ["original", "filtered"][1]

CONFIG_LOCATION = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

DATA_DIR = CONFIG["internal_data_dir"]
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
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.cluster import DBSCAN
from sklearn.linear_model import BayesianRidge, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline

from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.eval import pairwise_eval, cluster_eval, facet_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, NAME_COUNTS_PATH
from s2and.file_cache import cached_path
from s2and.plotting_utils import plot_facets
from hyperopt import hp

logger = logging.getLogger("s2and")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

PAIRWISE_ONLY_DATASETS = {"medline", "augmented"}
BLOCK_TYPE = "s2"
N_VAL_TEST_SIZE = 10000
N_ITER = 50
PREPROCESS = True


def transfer_helper(
    source_dataset,
    target_dataset,
    experiment_name,
    random_seed,
    featurizer_info,
    nameless_featurizer_info,
    use_s2_clusters=False,
    skip_shap=False,
):
    source_name = source_dataset["name"]
    target_name = target_dataset["name"]

    if use_s2_clusters:
        pairwise_metrics = {
            "AUROC": None,
            "Average Precision": None,
            "F1": None,
            "Precision": None,
            "Recall": None,
        }
    else:
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

    if target_name not in PAIRWISE_ONLY_DATASETS and source_name not in PAIRWISE_ONLY_DATASETS:
        cluster_metrics, b3_metrics_per_signature = cluster_eval(
            target_dataset["anddata"],
            source_dataset["clusterer"],
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

    if not use_s2_clusters:
        metrics = {"pairwise": pairwise_metrics, "cluster": cluster_metrics}
        logger.info(f"{source_name}_to_{target_name}: {metrics}")
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


def summary_features_analysis(
    union_firstname_f1,
    union_affiliation_f1,
    union_email_f1,
    union_abstract_f1,
    union_venue_f1,
    union_references_f1,
    union_coauthors_f1,
    union_s2_firstname_f1,
    union_s2_affiliation_f1,
    union_s2_email_f1,
    union_s2_abstract_f1,
    union_s2_venue_f1,
    union_s2_references_f1,
    union_s2_coauthors_f1,
):
    """
    Aggregates differences in performance for s2and and s2,
    across different feature availability indicators.
    """

    feature_summary = []

    for s2and_feature_facet, s2_feature_facet in zip(
        [
            union_firstname_f1,
            union_affiliation_f1,
            union_email_f1,
            union_abstract_f1,
            union_venue_f1,
            union_references_f1,
            union_coauthors_f1,
        ],
        [
            union_s2_firstname_f1,
            union_s2_affiliation_f1,
            union_s2_email_f1,
            union_s2_abstract_f1,
            union_s2_venue_f1,
            union_s2_references_f1,
            union_s2_coauthors_f1,
        ],
    ):
        s2and_pres_avg = sum(s2and_feature_facet[1]) / len(s2and_feature_facet[1])
        s2and_abs_avg = sum(s2and_feature_facet[0]) / len(s2and_feature_facet[0])
        s2_pres_avg = sum(s2_feature_facet[1]) / len(s2_feature_facet[1])
        s2_abs_avg = sum(s2_feature_facet[0]) / len(s2_feature_facet[0])

        feature_summary.append(
            [
                s2and_pres_avg,
                s2and_abs_avg,
                s2_pres_avg,
                s2_abs_avg,
            ]
        )

    s2and_feature_facet_scores = {
        "first_name_diff": np.round(feature_summary[0][0], 3) - np.round(feature_summary[0][1], 3),
        "affilition_diff": np.round(feature_summary[1][0], 3) - np.round(feature_summary[1][1], 3),
        "email_diff": np.round(feature_summary[2][0], 3) - np.round(feature_summary[2][1], 3),
        "abstract_diff": np.round(feature_summary[3][0], 3) - np.round(feature_summary[3][1], 3),
        "venue_diff": np.round(feature_summary[4][0], 3) - np.round(feature_summary[4][1], 3),
        "references_diff": np.round(feature_summary[5][0], 3) - np.round(feature_summary[5][1], 3),
        "coauthors_diff": np.round(feature_summary[6][0], 3) - np.round(feature_summary[6][1], 3),
    }

    s2_feature_facet_scores = {
        "first_name_diff": np.round(feature_summary[0][2], 3) - np.round(feature_summary[0][3], 3),
        "affilition_diff": np.round(feature_summary[1][2], 3) - np.round(feature_summary[1][3], 3),
        "email_diff": np.round(feature_summary[2][2], 3) - np.round(feature_summary[2][3], 3),
        "abstract_diff": np.round(feature_summary[3][2], 3) - np.round(feature_summary[3][3], 3),
        "venue_diff": np.round(feature_summary[4][2], 3) - np.round(feature_summary[4][3], 3),
        "references_diff": np.round(feature_summary[5][2], 3) - np.round(feature_summary[5][3], 3),
        "coauthors_diff": np.round(feature_summary[6][2], 3) - np.round(feature_summary[6][3], 3),
    }

    return s2and_feature_facet_scores, s2_feature_facet_scores


def disparity_analysis(
    comb_s2and_facet_f1,
    comb_s2_facet_f1,
):
    """
    Studying disparity for different gender and ethnicity groups.

    Metric 1: Standard deviation
    Metric 2: Sum of difference from privileged group performance (found by max performance)

    Also finds the best and worst performing groups, synonymous to most and least privileged ones.
    """
    s2and_f1 = []
    s2_f1 = []
    keylist = []

    for facet, f1 in comb_s2and_facet_f1.items():
        s2and_average = sum(comb_s2and_facet_f1[facet]) / len(comb_s2and_facet_f1[facet])
        s2_average = sum(comb_s2_facet_f1[facet]) / len(comb_s2_facet_f1[facet])
        keylist.append(facet[0:3])
        s2and_f1.append(s2and_average)
        s2_f1.append(s2_average)

    print("facet", keylist, s2and_f1, s2_f1)

    s2and_deviation = np.std(s2and_f1)
    s2_deviation = np.std(s2_f1)
    s2and_max_performance = max(s2and_f1)
    s2_max_performance = max(s2_f1)
    s2and_min_performance = min(s2and_f1)
    s2_min_performance = min(s2_f1)
    s2and_max_group = keylist[s2and_f1.index(s2and_max_performance)]
    s2_max_group = keylist[s2_f1.index(s2_max_performance)]
    s2and_min_group = keylist[s2and_f1.index(s2and_min_performance)]
    s2_min_group = keylist[s2_f1.index(s2_min_performance)]

    s2and_sum_deviation = 0
    for group_f1 in s2and_f1:
        s2and_sum_deviation += s2and_max_performance - group_f1

    s2_sum_deviation = 0
    for group_f1 in s2_f1:
        s2_sum_deviation += s2_max_performance - group_f1

    disparity_scores = {
        "S2AND std": np.round(s2and_deviation, 3),
        "S2 std": np.round(s2_deviation, 3),
        "S2AND sum-diff": np.round(s2and_sum_deviation, 3),
        "S2 sum-diff": np.round(s2_sum_deviation, 3),
        "S2AND max-perf-group": s2and_max_group,
        "S2 max-perf-group": s2_max_group,
        "S2AND min-perf-group": s2and_min_group,
        "S2 min-perf-group": s2_min_group,
        "S2AND max-perf": np.round(s2and_max_performance, 3),
        "S2 max-perf": np.round(s2_max_performance, 3),
        "S2AND min-perf": np.round(s2and_min_performance, 3),
        "S2 min-perf": np.round(s2_min_performance, 3),
    }

    return disparity_scores


def update_facets(
    gender_f1,
    ethnicity_f1,
    author_num_f1,
    year_f1,
    block_len_f1,
    cluster_len_f1,
    homonymity_f1,
    synonymity_f1,
    firstname_f1,
    affiliation_f1,
    email_f1,
    abstract_f1,
    venue_f1,
    references_f1,
    coauthors_f1,
    comb_gender_f1,
    comb_ethnicity_f1,
    comb_author_num_f1,
    comb_year_f1,
    comb_block_len_f1,
    comb_cluster_len_f1,
    comb_homonymity_f1,
    comb_synonymity_f1,
    comb_firstname_f1,
    comb_affiliaition_f1,
    comb_email_f1,
    comb_abstract_f1,
    comb_venue_f1,
    comb_references_f1,
    comb_coauthors_f1,
):
    """
    Macro-average over individual facets.
    """
    for individual_facet, combined_facet in zip(
        [
            gender_f1,
            ethnicity_f1,
            author_num_f1,
            year_f1,
            block_len_f1,
            cluster_len_f1,
            homonymity_f1,
            synonymity_f1,
            firstname_f1,
            affiliation_f1,
            email_f1,
            abstract_f1,
            venue_f1,
            references_f1,
            coauthors_f1,
        ],
        [
            comb_gender_f1,
            comb_ethnicity_f1,
            comb_author_num_f1,
            comb_year_f1,
            comb_block_len_f1,
            comb_cluster_len_f1,
            comb_homonymity_f1,
            comb_synonymity_f1,
            comb_firstname_f1,
            comb_affiliaition_f1,
            comb_email_f1,
            comb_abstract_f1,
            comb_venue_f1,
            comb_references_f1,
            comb_coauthors_f1,
        ],
    ):
        for key, f1 in individual_facet.items():
            combined_facet[key].extend(f1)

    return (
        comb_gender_f1,
        comb_ethnicity_f1,
        comb_author_num_f1,
        comb_year_f1,
        comb_block_len_f1,
        comb_cluster_len_f1,
        comb_homonymity_f1,
        comb_synonymity_f1,
        comb_firstname_f1,
        comb_affiliaition_f1,
        comb_email_f1,
        comb_abstract_f1,
        comb_venue_f1,
        comb_references_f1,
        comb_coauthors_f1,
    )


def main(
    experiment_name: str,
    dont_use_nameless_model: bool,
    n_jobs: int,
    dont_use_monotone_constraints: bool,
    exclude_medline: bool,
    linkage: str,
    use_dbscan: bool,
    leave_self_in: bool,
    random_seed: int,
    skip_individual_models: bool,
    skip_union_models: bool,
    n_train_pairs_size: int,
    feature_groups_to_skip: List[str],
    use_linear_pairwise_model: bool,
    gender_ethnicity_available: bool,
    use_cache: bool,
):
    USE_NAMELESS_MODEL = not dont_use_nameless_model
    N_JOBS = n_jobs
    USE_MONOTONE_CONSTRAINTS = not dont_use_monotone_constraints
    LEAVE_SELF_OUT_FOR_UNION = not leave_self_in
    INDIVIDUAL_MODELS = not skip_individual_models
    UNION_MODELS = not skip_union_models
    N_TRAIN_PAIRS_SIZE = n_train_pairs_size
    USE_CACHE = use_cache
    logger.info(
        (
            f"USE_NAMELESS_MODEL={USE_NAMELESS_MODEL}, "
            f"N_JOBS={N_JOBS}, "
            f"USE_MONOTONE_CONSTRAINTS={USE_MONOTONE_CONSTRAINTS}, "
            f"exclude_medline={exclude_medline}, "
            f"linkage={linkage}, "
            f"use_dbscan={use_dbscan}, "
            f"LEAVE_SELF_OUT_FOR_UNION={LEAVE_SELF_OUT_FOR_UNION}, "
            f"random_seed={random_seed}, "
            f"INDIVIDUAL_MODELS={INDIVIDUAL_MODELS}, "
            f"UNION_MODELS={UNION_MODELS}, "
            f"N_TRAIN_PAIRS_SIZE={N_TRAIN_PAIRS_SIZE}, "
            f"feature_groups_to_skip={feature_groups_to_skip}, "
            f"use_linear_pairwise_model={use_linear_pairwise_model}, "
            f"USE_CACHE={USE_CACHE}, "
        )
    )

    FEATURES_TO_USE = [
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

    SOURCE_DATASET_NAMES = [
        "aminer",
        "arnetminer",
        "inspire",
        "kisti",
        "orcid",
        "pubmed",
        "qian",
        "zbmath",
        "augmented",
    ]

    TARGET_DATASET_NAMES = [
        "aminer",
        "arnetminer",
        "inspire",
        "kisti",
        "orcid",
        "pubmed",
        "qian",
        "zbmath",
        "augmented",
    ]

    DATASETS_FOR_UNION: List[str] = [
        "aminer",
        "arnetminer",
        "inspire",
        "kisti",
        "orcid",
        "pubmed",
        "qian",
        "zbmath",
        "augmented",
    ]

    if not exclude_medline:
        SOURCE_DATASET_NAMES.append("medline")
        TARGET_DATASET_NAMES.append("medline")
        DATASETS_FOR_UNION.append("medline")

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

    if UNION_MODELS:
        DATASETS_TO_TRAIN = set(SOURCE_DATASET_NAMES).union(set(TARGET_DATASET_NAMES)).union(set(DATASETS_FOR_UNION))
    else:
        DATASETS_TO_TRAIN = set(SOURCE_DATASET_NAMES).union(set(TARGET_DATASET_NAMES))

    if LEAVE_SELF_OUT_FOR_UNION:
        UNION_DATASETS_TO_TRAIN = set()
        for dataset_name in TARGET_DATASET_NAMES:
            one_left_out_dataset = set(DATASETS_FOR_UNION) - {dataset_name}
            UNION_DATASETS_TO_TRAIN.add(tuple(sorted(list(one_left_out_dataset))))
    else:
        UNION_DATASETS_TO_TRAIN = {tuple(DATASETS_FOR_UNION)}

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
            train_pairs_size=np.maximum(N_TRAIN_PAIRS_SIZE, 100000),
            val_pairs_size=N_VAL_TEST_SIZE,
            test_pairs_size=N_VAL_TEST_SIZE,
            n_jobs=N_JOBS,
            load_name_counts=name_counts,
            preprocess=PREPROCESS,
            random_seed=random_seed,
            name_tuples=NAME_TUPLES_VARIANT,
        )
        logger.info(f"dataset {dataset_name} loaded")

        logger.info(f"featurizing {dataset_name}")
        train, val, test = featurize(anddata, FEATURIZER_INFO, n_jobs=N_JOBS, use_cache=USE_CACHE, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=NAMELESS_FEATURIZER_INFO, nan_value=NAN_VALUE)  # type: ignore
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
        if INDIVIDUAL_MODELS and dataset_name in SOURCE_DATASET_NAMES:
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

            if dataset_name not in PAIRWISE_ONLY_DATASETS:
                logger.info(f"fitting clusterer for {dataset_name}")
                cluster = Clusterer(
                    FEATURIZER_INFO,
                    pairwise_modeler.classifier,
                    cluster_model=(
                        FastCluster(linkage=linkage) if not use_dbscan else DBSCAN(min_samples=1, metric="precomputed")
                    ),
                    search_space=cluster_search_space,
                    n_jobs=N_JOBS,
                    use_cache=USE_CACHE,
                    nameless_classifier=(
                        nameless_pairwise_modeler.classifier if nameless_pairwise_modeler is not None else None
                    ),
                    nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
                    random_state=random_seed,
                    use_default_constraints_as_supervision=True,
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

    if UNION_MODELS:
        unions = {}
        for dataset_name_tuple in tqdm(UNION_DATASETS_TO_TRAIN, desc="Fitting union models..."):
            logger.info("")
            logger.info("loading dataset for " + str(dataset_name_tuple))
            anddatas = [
                datasets[dataset_name]["anddata"]
                for dataset_name in dataset_name_tuple
                if dataset_name not in PAIRWISE_ONLY_DATASETS
            ]

            X_train = np.vstack([datasets[dataset_name]["X_train"] for dataset_name in dataset_name_tuple])
            y_train = np.hstack([datasets[dataset_name]["y_train"] for dataset_name in dataset_name_tuple])
            X_val = np.vstack(
                [
                    datasets[dataset_name]["X_val"]
                    for dataset_name in dataset_name_tuple
                    if dataset_name not in {"augmented"}
                ]
            )
            y_val = np.hstack(
                [
                    datasets[dataset_name]["y_val"]
                    for dataset_name in dataset_name_tuple
                    if dataset_name not in {"augmented"}
                ]
            )

            nameless_X_train = np.vstack(
                [datasets[dataset_name]["nameless_X_train"] for dataset_name in dataset_name_tuple]
            )
            nameless_X_val = np.vstack(
                [
                    datasets[dataset_name]["nameless_X_val"]
                    for dataset_name in dataset_name_tuple
                    if dataset_name not in {"augmented"}
                ]
            )
            logger.info("dataset loaded for " + str(dataset_name_tuple))

            logger.info("fitting pairwise for " + str(dataset_name_tuple))
            union_classifier = PairwiseModeler(
                n_iter=N_ITER,
                estimator=estimator,
                search_space=pairwise_search_space,
                monotone_constraints=MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
                random_state=random_seed,
            )
            union_classifier.fit(X_train, y_train, X_val, y_val)
            logger.info("pairwise fit for " + str(dataset_name_tuple))

            nameless_union_classifier = None
            if USE_NAMELESS_MODEL:
                logger.info("nameless fitting pairwise for " + str(dataset_name_tuple))
                nameless_union_classifier = PairwiseModeler(
                    n_iter=N_ITER,
                    estimator=estimator,
                    search_space=pairwise_search_space,
                    monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS if USE_MONOTONE_CONSTRAINTS else None,
                    random_state=random_seed,
                )
                nameless_union_classifier.fit(nameless_X_train, y_train, nameless_X_val, y_val)
                logger.info("nameless pairwise fit for " + str(dataset_name_tuple))

            union_clusterer: Optional[Clusterer] = None
            if len(anddatas) > 0:
                distances_for_sparsity = [1 - pred[1] for pred in union_classifier.predict_proba(X_train)]
                threshold = np.percentile(distances_for_sparsity, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                logger.info(f"Thresholds {threshold}")

                logger.info("fitting clusterer for " + str(dataset_name_tuple))
                union_clusterer = Clusterer(
                    FEATURIZER_INFO,
                    union_classifier.classifier,
                    cluster_model=(
                        FastCluster(linkage=linkage) if not use_dbscan else DBSCAN(min_samples=1, metric="precomputed")
                    ),
                    search_space=cluster_search_space,
                    n_jobs=N_JOBS,
                    use_cache=USE_CACHE,
                    nameless_classifier=(
                        nameless_union_classifier.classifier if nameless_union_classifier is not None else None
                    ),
                    nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
                    random_state=random_seed,
                    use_default_constraints_as_supervision=True,
                )
                union_clusterer.fit(anddatas)
                logger.info("clusterer fit for " + str(dataset_name_tuple))
                logger.info(f"{dataset_name_tuple} best clustering parameters: " + str(union_clusterer.best_params))

            models: Dict[str, Any] = {}
            models["pairwise_modeler"] = union_classifier
            models["nameless_pairwise_modeler"] = nameless_union_classifier
            models["clusterer"] = union_clusterer
            models["name"] = "union__" + "_".join(dataset_name_tuple)
            unions[dataset_name_tuple] = models

    logger.info("")
    logger.info("making evaluation grids")
    b3_f1_grid = [
        ["" for j in range(len(TARGET_DATASET_NAMES) + 1)]
        for i in range(len(SOURCE_DATASET_NAMES) + 1 + 2 * int(UNION_MODELS))
    ]

    for i in range(max(len(TARGET_DATASET_NAMES), len(SOURCE_DATASET_NAMES))):
        if i < len(TARGET_DATASET_NAMES):
            b3_f1_grid[0][i + 1] = TARGET_DATASET_NAMES[i]
        if i < len(SOURCE_DATASET_NAMES):
            b3_f1_grid[i + 1][0] = SOURCE_DATASET_NAMES[i]

    if UNION_MODELS:
        b3_f1_grid[len(SOURCE_DATASET_NAMES) + 1][0] = "union"
        b3_f1_grid[len(SOURCE_DATASET_NAMES) + 2][0] = "s2"

    pairwise_auroc_grid = copy.deepcopy(b3_f1_grid)  # makes a copy of the grid
    true_bigger_ratios_and_counts_grid = copy.deepcopy(b3_f1_grid)
    pred_bigger_ratios_and_counts_grid = copy.deepcopy(b3_f1_grid)

    # transfer of individual models
    if INDIVIDUAL_MODELS:
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

    # union
    if UNION_MODELS:
        union_gender_f1: Dict[str, List] = defaultdict(list)
        union_ethnicity_f1: Dict[str, List] = defaultdict(list)
        union_author_num_f1: Dict[int, List] = defaultdict(list)
        union_year_f1: Dict[int, List] = defaultdict(list)
        union_block_len_f1: Dict[int, List] = defaultdict(list)
        union_cluster_len_f1: Dict[int, List] = defaultdict(list)
        union_homonymity_f1: Dict[int, List] = defaultdict(list)
        union_synonymity_f1: Dict[int, List] = defaultdict(list)

        # features availability
        union_firstname_f1: Dict[str, List] = defaultdict(list)
        union_affiliation_f1: Dict[str, List] = defaultdict(list)
        union_email_f1: Dict[int, List] = defaultdict(list)
        union_abstract_f1: Dict[int, List] = defaultdict(list)
        union_venue_f1: Dict[int, List] = defaultdict(list)
        union_references_f1: Dict[int, List] = defaultdict(list)
        union_coauthors_f1: Dict[int, List] = defaultdict(list)

        union_s2_gender_f1: Dict[str, List] = defaultdict(list)
        union_s2_ethnicity_f1: Dict[str, List] = defaultdict(list)
        union_s2_author_num_f1: Dict[int, List] = defaultdict(list)
        union_s2_year_f1: Dict[int, List] = defaultdict(list)
        union_s2_block_len_f1: Dict[int, List] = defaultdict(list)
        union_s2_cluster_len_f1: Dict[int, List] = defaultdict(list)
        union_s2_homonymity_f1: Dict[int, List] = defaultdict(list)
        union_s2_synonymity_f1: Dict[int, List] = defaultdict(list)

        # features availability
        union_s2_firstname_f1: Dict[str, List] = defaultdict(list)
        union_s2_affiliation_f1: Dict[str, List] = defaultdict(list)
        union_s2_email_f1: Dict[int, List] = defaultdict(list)
        union_s2_abstract_f1: Dict[int, List] = defaultdict(list)
        union_s2_venue_f1: Dict[int, List] = defaultdict(list)
        union_s2_references_f1: Dict[int, List] = defaultdict(list)
        union_s2_coauthors_f1: Dict[int, List] = defaultdict(list)

        logger.info("started evaluating unions")
        for _, target_dataset in tqdm(datasets.items(), desc="Evaluating union models"):
            target_name = target_dataset["name"]
            if target_name not in TARGET_DATASET_NAMES:
                continue

            logger.info("")
            logger.info(f"evaluating union for {target_name}")
            if LEAVE_SELF_OUT_FOR_UNION:
                one_left_out_dataset = set(DATASETS_FOR_UNION) - {target_name}
                dataset_name_tuple = tuple(sorted(list(one_left_out_dataset)))

            else:
                dataset_name_tuple = tuple(DATASETS_FOR_UNION)

            source_dataset = unions[dataset_name_tuple]
            (
                pairwise_metrics,
                cluster_metrics,
                b3_metrics_per_signature,
            ) = transfer_helper(
                source_dataset,
                target_dataset,
                experiment_name,
                random_seed,
                FEATURIZER_INFO,
                NAMELESS_FEATURIZER_INFO,
                skip_shap=use_linear_pairwise_model,  # skip SHAP if not using default model
            )

            (
                s2_pairwise_metrics,
                s2_cluster_metrics,
                s2_b3_metrics_per_signature,
            ) = transfer_helper(
                source_dataset,
                target_dataset,
                experiment_name,
                random_seed,
                FEATURIZER_INFO,
                NAMELESS_FEATURIZER_INFO,
                use_s2_clusters=True,
                skip_shap=use_linear_pairwise_model,  # skip SHAP if not using default model
            )

            if b3_metrics_per_signature is not None:
                (
                    gender_f1,
                    ethnicity_f1,
                    author_num_f1,
                    year_f1,
                    block_len_f1,
                    cluster_len_f1,
                    homonymity_f1,
                    synonymity_f1,
                    firstname_f1,
                    affiliation_f1,
                    email_f1,
                    abstract_f1,
                    venue_f1,
                    references_f1,
                    coauthors_f1,
                    signature_wise_facets,
                ) = facet_eval(target_dataset["anddata"], b3_metrics_per_signature, BLOCK_TYPE)

                (
                    union_gender_f1,
                    union_ethnicity_f1,
                    union_author_num_f1,
                    union_year_f1,
                    union_block_len_f1,
                    union_cluster_len_f1,
                    union_homonymity_f1,
                    union_synonymity_f1,
                    union_firstname_f1,
                    union_affiliation_f1,
                    union_email_f1,
                    union_abstract_f1,
                    union_venue_f1,
                    union_references_f1,
                    union_coauthors_f1,
                ) = update_facets(
                    gender_f1,
                    ethnicity_f1,
                    author_num_f1,
                    year_f1,
                    block_len_f1,
                    cluster_len_f1,
                    homonymity_f1,
                    synonymity_f1,
                    firstname_f1,
                    affiliation_f1,
                    email_f1,
                    abstract_f1,
                    venue_f1,
                    references_f1,
                    coauthors_f1,
                    union_gender_f1,
                    union_ethnicity_f1,
                    union_author_num_f1,
                    union_year_f1,
                    union_block_len_f1,
                    union_cluster_len_f1,
                    union_homonymity_f1,
                    union_synonymity_f1,
                    union_firstname_f1,
                    union_affiliation_f1,
                    union_email_f1,
                    union_abstract_f1,
                    union_venue_f1,
                    union_references_f1,
                    union_coauthors_f1,
                )

            if s2_b3_metrics_per_signature is not None:
                (
                    s2_gender_f1,
                    s2_ethnicity_f1,
                    s2_author_num_f1,
                    s2_year_f1,
                    s2_block_len_f1,
                    s2_cluster_len_f1,
                    s2_homonymity_f1,
                    s2_synonymity_f1,
                    s2_firstname_f1,
                    s2_affiliation_f1,
                    s2_email_f1,
                    s2_abstract_f1,
                    s2_venue_f1,
                    s2_references_f1,
                    s2_coauthors_f1,
                    s2_signature_wise_facets,
                ) = facet_eval(target_dataset["anddata"], s2_b3_metrics_per_signature, BLOCK_TYPE)

                (
                    union_s2_gender_f1,
                    union_s2_ethnicity_f1,
                    union_s2_author_num_f1,
                    union_s2_year_f1,
                    union_s2_block_len_f1,
                    union_s2_cluster_len_f1,
                    union_s2_homonymity_f1,
                    union_s2_synonymity_f1,
                    union_s2_firstname_f1,
                    union_s2_affiliation_f1,
                    union_s2_email_f1,
                    union_s2_abstract_f1,
                    union_s2_venue_f1,
                    union_s2_references_f1,
                    union_s2_coauthors_f1,
                ) = update_facets(
                    s2_gender_f1,
                    s2_ethnicity_f1,
                    s2_author_num_f1,
                    s2_year_f1,
                    s2_block_len_f1,
                    s2_cluster_len_f1,
                    s2_homonymity_f1,
                    s2_synonymity_f1,
                    s2_firstname_f1,
                    s2_affiliation_f1,
                    s2_email_f1,
                    s2_abstract_f1,
                    s2_venue_f1,
                    s2_references_f1,
                    s2_coauthors_f1,
                    union_s2_gender_f1,
                    union_s2_ethnicity_f1,
                    union_s2_author_num_f1,
                    union_s2_year_f1,
                    union_s2_block_len_f1,
                    union_s2_cluster_len_f1,
                    union_s2_homonymity_f1,
                    union_s2_synonymity_f1,
                    union_s2_firstname_f1,
                    union_s2_affiliation_f1,
                    union_s2_email_f1,
                    union_s2_abstract_f1,
                    union_s2_venue_f1,
                    union_s2_references_f1,
                    union_s2_coauthors_f1,
                )

            b3_f1_grid[len(SOURCE_DATASET_NAMES) + 1][TARGET_DATASET_NAMES.index(target_name) + 1] = cluster_metrics[
                "B3 (P, R, F1)"
            ][2]
            pairwise_auroc_grid[len(SOURCE_DATASET_NAMES) + 1][TARGET_DATASET_NAMES.index(target_name) + 1] = (
                pairwise_metrics["AUROC"]
            )
            true_bigger_ratios_and_counts_grid[len(SOURCE_DATASET_NAMES) + 1][
                TARGET_DATASET_NAMES.index(target_name) + 1
            ] = cluster_metrics["True bigger ratio (mean, count)"]

            b3_f1_grid[len(SOURCE_DATASET_NAMES) + 2][TARGET_DATASET_NAMES.index(target_name) + 1] = s2_cluster_metrics[
                "B3 (P, R, F1)"
            ][2]
            pairwise_auroc_grid[len(SOURCE_DATASET_NAMES) + 2][TARGET_DATASET_NAMES.index(target_name) + 1] = (
                s2_pairwise_metrics["AUROC"]
            )

            true_bigger_ratios_and_counts_grid[len(SOURCE_DATASET_NAMES) + 1][
                TARGET_DATASET_NAMES.index(target_name) + 1
            ] = cluster_metrics["True bigger ratio (mean, count)"]

            pred_bigger_ratios_and_counts_grid[len(SOURCE_DATASET_NAMES) + 1][
                TARGET_DATASET_NAMES.index(target_name) + 1
            ] = cluster_metrics["Pred bigger ratio (mean, count)"]
            logger.info(f"finished evaluating union for {target_name}")
        logger.info("finished evaluating unions")

        if not os.path.exists(os.path.join(DATA_DIR, "experiments", experiment_name, "facets")):
            os.makedirs(os.path.join(DATA_DIR, "experiments", experiment_name, "facets"))

        s2and_feature_summary, s2_feature_summary = summary_features_analysis(
            union_firstname_f1,
            union_affiliation_f1,
            union_email_f1,
            union_abstract_f1,
            union_venue_f1,
            union_references_f1,
            union_coauthors_f1,
            union_s2_firstname_f1,
            union_s2_affiliation_f1,
            union_s2_email_f1,
            union_s2_abstract_f1,
            union_s2_venue_f1,
            union_s2_references_f1,
            union_s2_coauthors_f1,
        )

        if gender_ethnicity_available:
            gender_disparity = disparity_analysis(union_gender_f1, union_s2_gender_f1)
            ethnicity_disparity = disparity_analysis(union_ethnicity_f1, union_s2_ethnicity_f1)

            logger.info("")
            logger.info("disparity analysis")
            print("Gender Disparity in F1:")
            gender_disparity_df = pd.DataFrame(gender_disparity, index=[0])
            print(gender_disparity_df)

            print()

            print("Ethnicity Disparity in F1:")
            ethnicity_disparity_df = pd.DataFrame(ethnicity_disparity, index=[0])
            print(ethnicity_disparity_df)

            print()

            gender_disparity_df.to_csv(
                os.path.join(
                    DATA_DIR,
                    "experiments",
                    experiment_name,
                    "gender_disparity.csv",
                ),
                index=False,
            )
            ethnicity_disparity_df.to_csv(
                os.path.join(
                    DATA_DIR,
                    "experiments",
                    experiment_name,
                    "ethnicity_disparity.csv",
                ),
                index=False,
            )

        print("S2AND Feature effect in F1:")
        s2and_feature_df = pd.DataFrame(s2and_feature_summary, index=[0])
        print(s2and_feature_df)

        print()

        print("S2 Feature effect  in F1:")
        s2_feature_df = pd.DataFrame(s2_feature_summary, index=[0])
        print(s2_feature_df)

        s2and_feature_df.to_csv(
            os.path.join(
                DATA_DIR,
                "experiments",
                experiment_name,
                "s2and_feature_diff.csv",
            ),
            index=False,
        )
        s2_feature_df.to_csv(
            os.path.join(
                DATA_DIR,
                "experiments",
                experiment_name,
                "s2_feature_diff.csv",
            ),
            index=False,
        )

        facet_fig_path = os.path.join(DATA_DIR, "experiments", experiment_name, "facets")

        plot_facets(
            union_gender_f1,
            union_ethnicity_f1,
            union_author_num_f1,
            union_year_f1,
            union_block_len_f1,
            union_cluster_len_f1,
            union_homonymity_f1,
            union_synonymity_f1,
            union_s2_gender_f1,
            union_s2_ethnicity_f1,
            union_s2_author_num_f1,
            union_s2_year_f1,
            union_s2_block_len_f1,
            union_s2_cluster_len_f1,
            union_s2_homonymity_f1,
            union_s2_synonymity_f1,
            facet_fig_path,
            gender_ethnicity_available,
        )

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
        "--dont_use_nameless_model",
        action="store_true",
        help="Whether to use the nameless model",
    )
    parser.add_argument("--n_jobs", type=int, default=8, help="How many cpus to use")
    parser.add_argument(
        "--dont_use_monotone_constraints",
        action="store_true",
        help="Whether to use monotone constraints",
    )
    parser.add_argument(
        "--exclude_medline",
        action="store_true",
        help="Whether to exclude medline in experiments",
    )
    parser.add_argument("--linkage", type=str, default="average", help="What linkage function to use")
    parser.add_argument(
        "--use_dbscan",
        action="store_true",
        help="Whether to use DBSCAN instead of fastcluster",
    )
    parser.add_argument(
        "--leave_self_in",
        action="store_true",
        help="Whether to leave self in for union experiments",
    )
    parser.add_argument("--random_seed", nargs="+", default=[1], type=int)
    parser.add_argument(
        "--skip_individual_models", action="store_true", help="Whether to skip training/evaluating individual models"
    )
    parser.add_argument(
        "--skip_union_models", action="store_true", help="Whether to skip training/evaluating union models"
    )
    parser.add_argument("--n_train_pairs_size", type=int, default=100000, help="How many training pairs per dataset")
    parser.add_argument("--feature_groups_to_skip", nargs="+", default=[], type=str)
    parser.add_argument(
        "--use_linear_pairwise_model", action="store_true", help="Whether to use a LogisticRegression pairwise model"
    )
    parser.add_argument(
        "--gender_ethnicity_available",
        action="store_true",
        help="Do the signatures have estimated gender/ethnicity of author?",
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
            args.dont_use_nameless_model,
            args.n_jobs,
            args.dont_use_monotone_constraints,
            args.exclude_medline,
            args.linkage,
            args.use_dbscan,
            args.leave_self_in,
            int(seed),
            args.skip_individual_models,
            args.skip_union_models,
            args.n_train_pairs_size,
            args.feature_groups_to_skip,
            args.use_linear_pairwise_model,
            args.gender_ethnicity_available,
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
