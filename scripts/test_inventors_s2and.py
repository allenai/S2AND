import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

from s2and.consts import DEFAULT_CHUNK_SIZE, FEATURIZER_VERSION, PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import Clusterer, PairwiseModeler
from s2and.serialization import load_pickle_with_verified_label_encoder_compat

data_original = os.path.join(PROJECT_ROOT_PATH, "data")

specter_suffixes = ["_specter.pickle", "_specter2.pkl"]

random_seed = 42
n_jobs = 4


TRAIN_FLAG = False
#  1.0, 1.1, 1.2. 1.2 uses specter2. the rest user specter1
MODELS = {
    "_specter.pickle": "production_model_v1.1.pickle",
    "_specter2.pkl": "production_model_v1.2.pickle",
}

# aminer has too much variance
# medline is pairwise only
datasets = [
    "inventors_s2and",
]


def resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing dataset file. Tried '{preferred_path}' and '{fallback_path}'.")


# to train the pairwise model, we define which feature categories to use
# here it is all of them except reference model.
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


results = {}
num_test_blocks = {}
for specter_suffix in specter_suffixes:
    clusterer = None

    # when not retraining, load the pre-trained model artifact
    if not TRAIN_FLAG:
        model_path = os.path.join(PROJECT_ROOT_PATH, "data", MODELS[specter_suffix])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Missing model artifact at "
                f"{model_path}. This repo clone does not include production model pickles. "
                "Either set TRAIN_FLAG=True to retrain, or place the model pickle in data/."
            )
        clusterer = load_pickle_with_verified_label_encoder_compat(model_path)["clusterer"]
        clusterer.use_cache = False  # very important for this experiment!!!
        clusterer.n_jobs = n_jobs

    print(f"=== specter_suffix: {specter_suffix} ===")
    cluster_metrics_all = []
    for dataset_name in datasets:
        print(f"-- dataset: {dataset_name} --")
        signatures_path = resolve_dataset_file(
            data_original,
            dataset_name,
            f"{dataset_name}_signatures.json",
            "signatures.json",
        )
        papers_path = resolve_dataset_file(
            data_original,
            dataset_name,
            f"{dataset_name}_papers.json",
            "papers.json",
        )
        clusters_path = resolve_dataset_file(
            data_original,
            dataset_name,
            f"{dataset_name}_clusters.json",
            "clusters.json",
        )
        embeddings_path = resolve_dataset_file(
            data_original,
            dataset_name,
            f"{dataset_name}{specter_suffix}",
            specter_suffix.lstrip("_"),
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
        train_block_dict, val_block_dict, test_block_dict = anddata.split_blocks_helper(anddata.get_blocks())
        num_test_blocks[dataset_name] = len(test_block_dict)

        if TRAIN_FLAG:
            # now we can actually go and get the pairwise training, val and test data
            train, val, test = featurize(
                anddata,
                featurization_info,
                n_jobs=n_jobs,
                use_cache=False,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=nameless_featurization_info,
                nan_value=np.nan,
            )  # type: ignore
            X_train, y_train, nameless_X_train = train
            X_val, y_val, nameless_X_val = val
            X_test, y_test, nameless_X_test = test

            # now we define and fit the pairwise modelers
            pairwise_modeler = PairwiseModeler(
                n_iter=25,  # number of hyperparameter search iterations
                estimator=None,  # this will use the default LightGBM classifier
                search_space=None,  # this will use the default LightGBM search space
                monotone_constraints=(
                    featurization_info.lightgbm_monotone_constraints
                ),  # we use monotonicity constraints to make the model more sensible
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
                use_default_constraints_as_supervision=False,  # used by S2 prod, not in S2AND paper
            )
            clusterer.fit(anddata)

        if clusterer is None:
            raise RuntimeError("Clusterer was not initialized. Check TRAIN_FLAG and model artifact path.")

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
