# mypy: ignore-errors
"""
This script demonstrates how to use the production S2AND model (v1.1) for clustering.

We will use the test sets of arnnetminer and pubmed datasets as examples.
"""

import os
import pickle
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.consts import FEATURIZER_VERSION, PROJECT_ROOT_PATH
from s2and.featurizer import FeaturizationInfo


def main() -> None:

    n_jobs = 4

    # Limit BLAS threads to keep things responsive
    os.environ["OMP_NUM_THREADS"] = f"{n_jobs}"

    data_original = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")

    random_seed = 42

    datasets = [
        "arnetminer",
        "inspire",
        "kisti",
        "pubmed",
        "qian",
        "zbmath",
    ]

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

    # this is the prod 1.1 model
    with open(os.path.join(PROJECT_ROOT_PATH, "data", "production_model_v1.1.pickle"), "rb") as f:
        clusterer = pickle.load(f)["clusterer"]
        clusterer.use_cache = False  # very important for this experiment!!!
        clusterer.n_jobs = n_jobs

    num_test_blocks = {}

    cluster_metrics_all = []
    for dataset_name in datasets:
        anddata = ANDData(
            signatures=os.path.join(data_original, dataset_name, dataset_name + "_signatures.json"),
            papers=os.path.join(data_original, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=os.path.join(data_original, dataset_name, dataset_name + "_specter.pickle"),
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
            use_orcid_id=True,
            use_sinonym_overwrite=True,
        )
        train_block_dict, val_block_dict, test_block_dict = anddata.split_blocks_helper(anddata.get_blocks())
        num_test_blocks[dataset_name] = len(test_block_dict)

        cluster_metrics, b3_metrics_per_signature = cluster_eval(
            anddata,
            clusterer,
            split="test",
            use_s2_clusters=False,
        )
        print(cluster_metrics)
        cluster_metrics_all.append(cluster_metrics)

        # cluster_to_signatures = anddata.construct_cluster_to_signatures(test_block_dict)

        # # now we need to print out the unique tuples of anddata(get_full_name_for_features(signature)) for the signatures that were clustered together
        # for cluster_id, signatures in cluster_to_signatures.items():
        #     full_names = [anddata.get_full_name_for_features(anddata.signatures[sig]) for sig in signatures]
        #     # also get the BLOCK author_info_block
        #     blocks = [anddata.signatures[sig].author_info_block for sig in signatures]
        #     print(f"Cluster {cluster_id}: {set(list(zip(full_names, blocks)))}")

    b3s = [i["B3 (P, R, F1)"][-1] for i in cluster_metrics_all]
    print(b3s, sum(b3s) / len(b3s))

    for i in range(len(datasets)):
        print(f"Performance on {datasets[i]}: {cluster_metrics_all[i]['B3 (P, R, F1)']}")
        print()


if __name__ == "__main__":
    main()
