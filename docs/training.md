# Training and Evaluation

This document expands the root README's training example with the main steps for training, evaluating, saving, and reloading a model.

## Load a dataset in training mode

```python
from os.path import join

from s2and.data import ANDData

dataset_name = "pubmed"
parent_dir = f"s2and/data/{dataset_name}"

dataset = ANDData(
    signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    papers=join(parent_dir, f"{dataset_name}_papers.json"),
    clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    mode="train",
    block_type="s2",
    train_pairs_size=100000,
    val_pairs_size=10000,
    test_pairs_size=10000,
    name=dataset_name,
    n_jobs=8,
)
```

Training-mode preprocessing can take a while, especially on larger datasets.

## Featurize pairs and train the pairwise model

```python
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import PairwiseModeler

featurization_info = FeaturizationInfo()
train, val, test = featurize(dataset, featurization_info, n_jobs=8, use_cache=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

pairwise_model = PairwiseModeler(
    n_iter=25,
    calibrate=True,
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
)
pairwise_model.fit(X_train, y_train, X_val, y_val)
```

Why `use_cache=True` is often useful here:

- repeated training or evaluation runs often revisit the same pair sets
- the pair-feature cache avoids recomputing those rows

See [caching.md](caching.md) for the exact cache semantics.

## Evaluate the pairwise classifier

```python
from s2and.eval import pairwise_eval

pairwise_metrics = pairwise_eval(
    X_test,
    y_test,
    pairwise_model.classifier,
    figs_path="figs/",
    title="example",
)
print(pairwise_metrics)
```

This writes useful diagnostic plots such as ROC, PR, and SHAP outputs under `figs/`.

## Fit the clusterer

```python
from hyperopt import hp

from s2and.model import Clusterer, FastCluster

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)
```

S2AND uses agglomerative clustering with average linkage on top of the pairwise model.

## Evaluate clustering

```python
from s2and.eval import cluster_eval

metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

`metrics_per_signature` is useful when you want to slice performance by signature properties.

## Save and reload a trained model

Save:

```python
import pickle

with open("saved_model.pkl", "wb") as handle:
    pickle.dump(clusterer, handle)
```

Reload and predict:

```python
import pickle

from s2and.data import ANDData

with open("saved_model.pkl", "rb") as handle:
    clusterer = pickle.load(handle)

anddata = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter_embeddings.pkl",
    name="your_name_here",
    mode="inference",
    block_type="s2",
)

pred_clusters, pred_distance_matrices = clusterer.predict(anddata.get_blocks(), anddata)
```

`pred_distance_matrices` may be `None` when memory-optimized fused clustering paths are active.

## Reference scripts

- `scripts/transfer_experiment_seed_paper.py`: fuller transfer and evaluation workflow
- `scripts/tutorial_for_predicting_with_the_prod_model.py`: released-model inference example
- `scripts/README.md`: script catalog
