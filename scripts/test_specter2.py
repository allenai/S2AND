# mypy: ignore-errors
"""
In this script we try to answer the question: if we deploy SPECTER2, will S2AND care?
Both with retraining and without retraining.

This is done with s2and-mini. Ai2 employee, find it at s3://ai2-s2-research/s2and/s2and-mini/

With retraining:

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

Without retraining, just using the original SPECTER1 model with SPECTER1/SPECTER2 features,
we have substantial performance drop for SPECTER2:

Performance with SPECTERv1 data, on arnetminer (B3): (0.977, 0.982, 0.979)
Performance with SPECTERv2 data, on arnetminer (B3): (0.602, 0.991, 0.749)

Performance with SPECTERv1 data, on inspire (B3): (0.993, 0.964, 0.978)
Performance with SPECTERv2 data, on inspire (B3): (0.959, 0.938, 0.948)

Performance with SPECTERv1 data, on kisti (B3): (0.96, 0.957, 0.959)
Performance with SPECTERv2 data, on kisti (B3): (0.875, 0.982, 0.925)

Performance with SPECTERv1 data, on pubmed (B3): (1.0, 0.968, 0.984)
Performance with SPECTERv2 data, on pubmed (B3): (0.947, 0.988, 0.967)

Performance with SPECTERv1 data, on qian (B3): (0.985, 0.955, 0.969)
Performance with SPECTERv2 data, on qian (B3): (0.635, 0.99, 0.774)

Performance with SPECTERv1 data, on zbmath (B3): (0.967, 0.955, 0.961)
Performance with SPECTERv2 data, on zbmath (B3): (0.944, 0.934, 0.939)

"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import pickle
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, PROJECT_ROOT_PATH
from s2and.featurizer import FeaturizationInfo, featurize
import numpy as np
from s2and.model import PairwiseModeler, Clusterer

data_original = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")

specter_suffixes = ["_specter.pickle", "_specter2.pkl"]

random_seed = 42
n_jobs = 4

TRAIN_FLAG = True

# aminer has too much variance
# medline is pairwise only
datasets = [
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
]

# to train the pairwise model, we define which feature categories to use
# here it is all of them
features_to_use = [
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

# we also have this special second "nameless" model that doesn't use any name-based features
# it helps to improve clustering performance by preventing model overreliance on names
nameless_features_to_use = [
    feature_name
    for feature_name in features_to_use
    if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]

# we store all the information about the features in this convenient wrapper
featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
nameless_featurization_info = FeaturizationInfo(
    features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION
)

# this is the prod 1.1 model, which we may or may not retrain
with open(os.path.join(PROJECT_ROOT_PATH, "data", "model_dump.pickle"), "rb") as f:
    clusterer = pickle.load(f)["clusterer"]
    clusterer.use_cache = False  # very important for this experiment!!!

results = {}
num_test_blocks = {}
for specter_suffix in specter_suffixes:
    cluster_metrics_all = []
    for dataset_name in datasets:
        anddata = ANDData(
            signatures=os.path.join(data_original, dataset_name, dataset_name + "_signatures.json"),
            papers=os.path.join(data_original, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=os.path.join(data_original, dataset_name, dataset_name + specter_suffix),
            clusters=os.path.join(data_original, dataset_name, dataset_name + "_clusters.json"),
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
        train_block_dict, val_block_dict, test_block_dict = anddata.split_blocks_helper(anddata.get_blocks())
        num_test_blocks[dataset_name] = len(test_block_dict)

        if TRAIN_FLAG:
            # now we can actually go and get the pairwise training, val and test data
            train, val, test = featurize(anddata, featurization_info, n_jobs=n_jobs, use_cache=False, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=nameless_featurization_info, nan_value=np.nan)  # type: ignore
            X_train, y_train, nameless_X_train = train
            X_val, y_val, nameless_X_val = val
            X_test, y_test, nameless_X_test = test

            # now we define and fit the pairwise modelers
            pairwise_modeler = PairwiseModeler(
                n_iter=25,  # number of hyperparameter search iterations
                estimator=None,  # this will use the default LightGBM classifier
                search_space=None,  # this will use the default LightGBM search space
                monotone_constraints=featurization_info.lightgbm_monotone_constraints,  # we use monotonicity constraints to make the model more sensible
                random_state=random_seed,
            )
            pairwise_modeler.fit(X_train, y_train, X_val, y_val)

            # as mentioned above, there are 2: one with all features and a nameless one
            nameless_pairwise_modeler = PairwiseModeler(
                n_iter=25,
                estimator=None,
                search_space=None,
                monotone_constraints=nameless_featurization_info.lightgbm_monotone_constraints,
                random_state=random_seed,
            )
            nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)

            # now we can fit the clusterer itself
            clusterer = Clusterer(
                featurization_info,
                pairwise_modeler.classifier,  # the actual pairwise classifier
                n_jobs=n_jobs,
                use_cache=False,
                nameless_classifier=nameless_pairwise_modeler.classifier,  # the nameless pairwise classifier
                nameless_featurizer_info=nameless_featurization_info,
                random_state=random_seed,
                use_default_constraints_as_supervision=False,  # this is an option used by the S2 production system but not in the S2AND paper
            )
            clusterer.fit(anddata)

        cluster_metrics, b3_metrics_per_signature = cluster_eval(
            anddata,
            clusterer,
            split="test",
            use_s2_clusters=False,
        )
        print(cluster_metrics)
        cluster_metrics_all.append(cluster_metrics)

    results[specter_suffix] = cluster_metrics_all
    b3s = [i["B3 (P, R, F1)"][-1] for i in cluster_metrics_all]
    print(b3s, sum(b3s) / len(b3s))

result_specter1 = results["_specter.pickle"]
result_specter2 = results["_specter2.pkl"]

for i in range(len(datasets)):
    print(f"Performance with SPECTERv1 data, on {datasets[i]} (B3): {result_specter1[i]['B3 (P, R, F1)']}")
    print(f"Performance with SPECTERv2 data, on {datasets[i]} (B3): {result_specter2[i]['B3 (P, R, F1)']}")
    print()


"""
Iterative predict version comparison
"""
import os

os.environ["OMP_NUM_THREADS"] = "8"

import pickle
import numpy as np
from s2and.data import ANDData
from s2and.eval import cluster_eval
import seaborn as sns
import json

CONFIG_LOCATION = os.path.abspath(os.path.join("data", "path_config.json"))
with open(CONFIG_LOCATION) as _json_file:
    CONFIG = json.load(_json_file)

os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import logging
import pickle

logger = logging.getLogger("s2and")

os.environ["S2AND_CACHE"] = os.path.join(CONFIG["internal_data_dir"], ".feature_cache")
from random import shuffle
from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer
from s2and.eval import b3_precision_recall_fscore
from s2and.consts import FEATURIZER_VERSION

sns.set(context="talk")

SPECTER_SUFFIX = ["_specter.pickle", "_specter2.pkl"]
SIGNATURES_SUFFIX = ["_signatures.json", "_signatures_with_s2aff.json"][0]

USE_CACHE = False
N_TRAIN_PAIRS_SIZE = 100000
N_VAL_TEST_SIZE = 10000
DATA_DIR = CONFIG["internal_data_dir"]

random_seed = 42
n_jobs = 8


# this is the prod 1.1 model
with open("data/model_dump.pickle", "rb") as f:
    clusterer1 = pickle.load(f)["clusterer"]
    clusterer1.use_cache = False

with open("data/model_dump_specter2.pickle", "rb") as f:
    clusterer2 = pickle.load(f)["clusterer"]
    clusterer2.use_cache = False


results = {}
for dataset_name in ["aminer", "arnetminer", "kisti", "qian", "zbmath", "inspire"]:
    anddata1 = ANDData(
        signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + SIGNATURES_SUFFIX),
        papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + SPECTER_SUFFIX[0]),
        clusters=os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json"),
        block_type="s2",
        train_pairs_size=N_TRAIN_PAIRS_SIZE,
        val_pairs_size=N_VAL_TEST_SIZE,
        test_pairs_size=N_VAL_TEST_SIZE,
        preprocess=True,
    )

    anddata2 = ANDData(
        signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + SIGNATURES_SUFFIX),
        papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + SPECTER_SUFFIX[1]),
        clusters=os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json"),
        block_type="s2",
        train_pairs_size=N_TRAIN_PAIRS_SIZE,
        val_pairs_size=N_VAL_TEST_SIZE,
        test_pairs_size=N_VAL_TEST_SIZE,
        preprocess=True,
    )

    _, _, test_block_dict = anddata1.split_blocks_helper(anddata1.get_blocks())

    b3_f1s1 = []
    b3_f1s2 = []

    """
    the way that clusterer.predict_incremental works is that we have to populate anddata.cluster_seeds
    so here is what we do:

    (0) pick a random block
    (1) split up the block signatures above into train and test 0.9 and 0.1
    (2) use clusterer1.predict for the train using anddata1
    (3) populate anddata1.cluster_seeds and anddata2.cluster_seeds with the results of (2) using its preferred format
        require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
    (4) use clusterer2.predict_incremental on the test using anddata2 VS use clusterer2.predict_incremental on the test set using anddata2
    (5) see which one is correct
    (6) clean up
    (7) repeat
    """
    anddata1.cluster_seeds_require = {}
    anddata2.cluster_seeds_require = {}

    for _ in range(1000):
        # (0) pick a random block
        random_key = np.random.choice(list(test_block_dict.keys()))
        random_block = test_block_dict[random_key]
        cluster_to_signatures = anddata1.construct_cluster_to_signatures({"to_predict": random_block})

        # (1) randomly split the block into train and test 90/10
        shuffle(random_block)
        random_block_train, random_block_test = (
            random_block[: int(0.9 * len(random_block))],
            random_block[int(0.9 * len(random_block)) :],
        )

        if len(random_block_test) > 0 and len(random_block_train) > 0:
            # (2) use clusterer1.predict for the train using anddata1 - for this we need to format the block signatures into the correct format for predict
            to_predict_block_dict = {"to_predict": random_block_train}
            train_cluster_prediction = clusterer1.predict(to_predict_block_dict, anddata1)[0]

            # (3) populate anddata1.cluster_seeds_require and anddata2.cluster_seeds_require with the results of (2) using its preferred format
            for i, (cluster_id, cluster) in enumerate(train_cluster_prediction.items()):
                for sig_id in cluster:
                    anddata1.cluster_seeds_require[sig_id] = i
                    anddata2.cluster_seeds_require[sig_id] = i

            anddata1.max_seed_cluster_id = i + 1  # wow this was a fun thing to find out about
            anddata2.max_seed_cluster_id = i + 1

            # (4) use clusterer2.predict_incremental on the test using anddata2 VS use clusterer1.predict_incremental on the test set using anddata2
            test_preds_1 = clusterer1.predict_incremental(random_block_test, anddata1)
            test_preds_2 = clusterer2.predict_incremental(random_block_test, anddata2)

            def add_missed_ones_to_test_preds(train_cluster_prediction, test_preds):
                all_the_values_in_test_preds = set()
                for v in test_preds.values():
                    all_the_values_in_test_preds.update(v)

                # for each list in train_cluster_prediction.values() that is not in all_the_values_in_test_preds, add it to test_preds_1
                for i, v in enumerate(train_cluster_prediction.values()):
                    if v[0] not in all_the_values_in_test_preds:
                        test_preds[f"added_in_{i}"] = v

                return test_preds

            test_preds_1 = add_missed_ones_to_test_preds(train_cluster_prediction, test_preds_1)
            test_preds_2 = add_missed_ones_to_test_preds(train_cluster_prediction, test_preds_2)

            # (5) log the overall B3 scores for each
            (
                b3_p,
                b3_r,
                b3_f1,
                b3_metrics_per_signature,
                pred_bigger_ratios,
                true_bigger_ratios,
            ) = b3_precision_recall_fscore(cluster_to_signatures, test_preds_1)

            b3_f1s1.append(b3_f1)

            (
                b3_p,
                b3_r,
                b3_f1,
                b3_metrics_per_signature,
                pred_bigger_ratios,
                true_bigger_ratios,
            ) = b3_precision_recall_fscore(cluster_to_signatures, test_preds_2)

            b3_f1s2.append(b3_f1)

            # (6) cleanup anddata1.cluster_seeds and anddata2.cluster_seeds
            anddata1.cluster_seeds_require = {}
            anddata2.cluster_seeds_require = {}

    results[dataset_name] = (b3_f1s1, b3_f1s2)


# save results to pickle
with open("specter_version_iterative_experiment_results.pickle", "wb") as f:
    pickle.dump(results, f)
