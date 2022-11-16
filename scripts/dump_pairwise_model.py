import pickle
from os.path import join
import numpy as np
from s2and.consts import CONFIG
from s2and.data import PDData
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import PairwiseModeler, Clusterer
from s2and.eval import pairwise_eval
from s2and.consts import PROJECT_ROOT_PATH


N_JOBS = 16
N_ITER = 25  # for the hyperopt
EPS = 0.7  # see comments below where clusterer is defined

# which features to use (all of them)
features_to_use = [
    "author_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "abstract_similarity",
    "paper_quality",
]


# load the main dataset
dataset = PDData(
    join(CONFIG["main_data_dir"], "papers.json"),
    clusters=join(CONFIG["main_data_dir"], "clusters.json"),
    name="paper_clustering_dataset",
    n_jobs=N_JOBS,
    balanced_pair_sample=False,
    train_pairs_size=5000000,
    val_pairs_size=5000000,
    test_pairs_size=5000000,
)

# make features
# the cache will make it faster to train multiple times - it stores the features on disk for you
# off for now
featurization_info = FeaturizationInfo(features_to_use=features_to_use)
feature_names = featurization_info.get_feature_names()
train, val, test = featurize(dataset, featurization_info, n_jobs=N_JOBS, use_cache=False)
X_train, y_train, _ = train
X_val, y_val, _ = val
X_test, y_test, _ = test


# train pairwise model on train+val and eval on test.
model = PairwiseModeler(n_iter=N_ITER, n_jobs=N_JOBS)
# no test set is reserved at all because we want the best model for prod possible
model.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), X_test, y_test)

# compute some shap plots for the model on X_test and y_test
# subsample for speed
np.random.seed(seed=0)
rand_ind = np.random.choice(len(X_test), 100000, replace=False)

metrics = pairwise_eval(
    X_test[rand_ind, :],
    y_test[rand_ind],
    model,
    join(PROJECT_ROOT_PATH, "data"),
    "prod_model",
    feature_names,
    thresh_for_f1=0.5,
)
print(metrics)
# {'AUROC': 0.999, 'Average Precision': 0.999, 'F1': 0.987, 'Precision': 0.987, 'Recall': 0.988}
"""
get the clusterer together
note that eps is set to a number we know is good from previous experiments
and a number that trades off precision with recall

Previous experiment results (P, R, B3 F1):
S2:  (0.99951, 0.99476, 0.99713)
us (per EPS):
0.10 (0.99993, 0.95101, 0.97486)
0.15 (0.99993, 0.96465, 0.98197)
0.20 (0.99984, 0.97364, 0.98656)
0.25 (0.99984, 0.98077, 0.99021)
0.30 (0.99984, 0.98557, 0.99265)
0.35 (0.99979, 0.99020, 0.99497)
0.40 (0.99970, 0.99257, 0.99612)
0.45 (0.99955, 0.99422, 0.99688)
0.50 (0.99935, 0.99592, 0.99763)
0.55 (0.99926, 0.99666, 0.99796)
0.60 (0.99922, 0.99735, 0.99829)
0.65 (0.99914, 0.99785, 0.99850)
0.70 (0.99889, 0.99833, 0.99861) <- max, chosen here
0.75 (0.99862, 0.99860, 0.99861)
0.80 (0.99809, 0.99876, 0.99842)
0.85 (0.99755, 0.99887, 0.99821)
"""
cluster = Clusterer(
    featurization_info,
    model.classifier,
    use_default_constraints_as_supervision=True,  # note that this may need to be off once we have prod constraints
    n_iter=N_ITER,
    n_jobs=N_JOBS,
)
cluster.set_params({"eps": EPS})

models = {}
models["clusterer"] = cluster

with open(
    join(CONFIG["main_data_dir"], "prod_model.pickle"),
    "wb",
) as f:
    pickle.dump(models, f)
