# mypy: ignore-errors
"""
In this script we try to answer the question: if we deploy S2AFF, will S2AND care?

That is, if we use official linked ROR affiliation names instead of raw affiliations, will the S2AND
output change? Will we have to retrain?

Performance with original data, per dataset (B3): [0.979, 0.978, 0.959, 0.984, 0.969, 0.961] 0.9716666666666667
Performance with S2AFF-replaced data, per dataset (B3): [0.979, 0.978, 0.959, 0.984, 0.969, 0.961] 0.9716666666666667
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"

import pickle
import numpy as np
from s2and.data import ANDData
from s2and.eval import cluster_eval

data_original = "/net/nfs2.s2-research/phantasm/S2AND/s2and_mini/"
data_s2aff = "/home/sergey/S2AFF/data/s2and_mini/"

random_seed = 42
n_jobs = 4

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

# this is the prod 1.1 model
with open("data/production_model_v1.1.pickle", "rb") as f:
    clusterer = pickle.load(f)["clusterer"]
    clusterer.use_cache = False

with open("data/model_dump_specter2.pickle", "rb") as f:
    clusterer2 = pickle.load(f)["clusterer"]
    clusterer2.use_cache = False


def extract_a_nice_trials_object(clusterer):
    trials = clusterer.hyperopt_trials_store.trials
    eps = []
    losses = []
    for trial in trials:
        eps.append(trial["misc"]["vals"]["choice"][0])
        losses.append(-trial["result"]["loss"])
    # sort both by eps
    sort_indices = np.argsort(eps)
    eps = np.array(eps)[sort_indices]
    losses = np.array(losses)[sort_indices]
    return {i: j for i, j in zip(eps, losses)}


trials1 = extract_a_nice_trials_object(clusterer)
trials2 = extract_a_nice_trials_object(clusterer2)

results = {}
num_test_blocks = {}
for DATA_DIR in [data_original, data_s2aff]:
    cluster_metrics_all = []
    for dataset_name in datasets:
        anddata = ANDData(
            signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
            papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
            name=dataset_name,
            mode="train",
            specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
            clusters=os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json"),
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
        )
        train_block_dict, val_block_dict, test_block_dict = anddata.split_blocks_helper(anddata.get_blocks())
        num_test_blocks[dataset_name] = len(test_block_dict)

        # # intentionally mess up all affiliations
        # for key in anddata.signatures.keys():
        #     anddata.signatures[key].author_info_affiliations = []
        #     anddata.signatures.author_info_affiliations_n_grams = Counter()

        cluster_metrics, b3_metrics_per_signature = cluster_eval(
            anddata,
            clusterer,
            split="test",
            use_s2_clusters=False,
        )
        print(cluster_metrics)
        cluster_metrics_all.append(cluster_metrics)

    results[DATA_DIR] = cluster_metrics_all
    b3s = [i["B3 (P, R, F1)"][-1] for i in cluster_metrics_all]
    print(b3s, sum(b3s) / len(b3s))

result_og = results[data_original]
result_s2aff = results[data_s2aff]

for i in range(len(datasets)):
    print(f"Performance with original data, on {datasets[i]} (B3): {result_og[i]['B3 (P, R, F1)']}")
    print(f"Performance with S2AFF-replaced data, on {datasets[i]} dataset (B3): {result_s2aff[i]['B3 (P, R, F1)']}")
    print()


# dive in
from s2and.featurizer import featurize
from s2and.consts import DEFAULT_CHUNK_SIZE

dataset_name = "pubmed"

DATA_DIR = "/net/nfs2.s2-research/phantasm/S2AND/s2and_mini/"
anddata1 = ANDData(
    signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
    papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
    name=dataset_name,
    mode="train",
    specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
    clusters=os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json"),
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
)

DATA_DIR = "/home/sergey/S2AFF/data/s2and_mini/"
anddata2 = ANDData(
    signatures=os.path.join(DATA_DIR, dataset_name, dataset_name + "_signatures.json"),
    papers=os.path.join(DATA_DIR, dataset_name, dataset_name + "_papers.json"),
    name=dataset_name,
    mode="train",
    specter_embeddings=os.path.join(DATA_DIR, dataset_name, dataset_name + "_specter.pickle"),
    clusters=os.path.join(DATA_DIR, dataset_name, dataset_name + "_clusters.json"),
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
)


cluster_metrics1, b3_metrics_per_signature1 = cluster_eval(
    anddata1,
    clusterer,
    split="test",
    use_s2_clusters=False,
)

cluster_metrics2, b3_metrics_per_signature2 = cluster_eval(
    anddata2,
    clusterer,
    split="test",
    use_s2_clusters=False,
)

# find keys where b3_metrics_per_signature1 != b3_metrics_per_signature2
for key in b3_metrics_per_signature1.keys():
    if b3_metrics_per_signature1[key] != b3_metrics_per_signature2[key]:
        print("-----------")
        print("B3 (p, r, f1) for original aff:", b3_metrics_per_signature1[key])
        print("B3 (p, r, f1) for s2aff:", b3_metrics_per_signature2[key])
        # print the actual signatures from both anddata1 and anddata2
        print(
            "Original aff:",
            anddata1.signatures[key].author_info_affiliations,
            anddata1.signatures[key].author_info_full_name,
        )
        print(
            "S2AFF map:",
            anddata2.signatures[key].author_info_affiliations,
            anddata2.signatures[key].author_info_full_name,
        )


featurization_info = clusterer.featurizer_info
nameless_featurization_info = clusterer.nameless_featurizer_info

_, _, test1 = featurize(anddata1, featurization_info, n_jobs=4, use_cache=False, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=nameless_featurization_info, nan_value=np.nan)  # type: ignore
X_test1, y_test1, nameless_X_test1 = test1

_, _, test2 = featurize(anddata2, featurization_info, n_jobs=4, use_cache=False, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=nameless_featurization_info, nan_value=np.nan)  # type: ignore
X_test2, y_test2, nameless_X_test2 = test2

aff_ind = featurization_info.get_feature_names().index("affiliation_overlap")
diff = X_test2[:, aff_ind] - X_test1[:, aff_ind]
diff = diff[~np.isnan(diff)]
np.mean(diff != 0) * 100
