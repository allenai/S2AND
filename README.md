# S2AND
This repository provides access to the S2AND dataset and S2AND reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman.

The reference model is live on semanticscholar.org, and the trained model is available now as part of the data download (see below).

# Installation


## Developer setup

```bash
# From the repo root

# 1) Create a virtual env (uv will default to .venv if no name is given)
uv venv --python 3.11

# 2) Install project deps from pyproject.toml
#    - runtime only:
uv sync
#    - or include dev tools (tests, linters, type checker):
uv sync --extra dev

# 3) (Optional) Install the project in editable mode
uv run pip install -e . --no-deps
```

You can run everything **without activating** the venv:

```bash
uv run pytest
uv run python -c "import s2and; print(s2and.__version__)"
```

If you prefer to **activate**:

* macOS/Linux (bash/zsh):
  `source .venv/bin/activate`
* Windows PowerShell:
  `.venv\Scripts\Activate.ps1`
* Windows CMD:
  `.venv\Scripts\activate.bat`

## Installation (library users)

If you just want the package (no dev tools):

```bash
# with uv
uv pip install .

# or with pip
pip install .
```

## Running Tests

To run the tests, use the following command:

```bash
uv run pytest tests/
```

To run the entire CI suite mimicking the GH Actions, use the following command:
```bash
python scripts\run_ci_locally.py
```


## Data 
To obtain the S2AND dataset, run the following command after the package is installed (from inside the `S2AND` directory):  
```[Expected download size is: 50.4 GiB]```

`aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/`

Note that this software package comes with tools specifically designed to access and model the dataset.

## Configuration
Modify the config file at `data/path_config.json`. This file should look like this
```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
    "internal_data_dir": "ignore this one unless you work at AI2"
}
```
As the dummy file says, `main_data_dir` should be set to the location of wherever you downloaded the data to, and
`internal_data_dir` can be ignored, as it is used for some scripts that rely on unreleased data, internal to Semantic Scholar.

## How to use S2AND for loading data and training a model
Once you have downloaded the datasets, you can go ahead and load up one of them:

```python
from os.path import join
from s2and.data import ANDData

dataset_name = "pubmed"
parent_dir = f"data/{dataset_name}"
dataset = ANDData(
    signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    papers=join(parent_dir, f"{dataset_name}_papers.json"),
    mode="train",
    specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    block_type="s2",
    train_pairs_size=100000,
    val_pairs_size=10000,
    test_pairs_size=10000,
    name=dataset_name,
    n_jobs=8,
)
```

This may take a few minutes - there is a lot of text pre-processing to do.

The first step in the S2AND pipeline is to specify a featurizer and then train a binary classifier
that tries to guess whether two signatures are referring to the same person. 

We'll do hyperparameter selection with the validation set and then get the test area under ROC curve.

Here's how to do all that:

```python
from s2and.model import PairwiseModeler
from s2and.featurizer import FeaturizationInfo
from s2and.eval import pairwise_eval

featurization_info = FeaturizationInfo()
# the cache will make it faster to train multiple times - it stores the features on disk for you
train, val, test = featurize(dataset, featurization_info, n_jobs=8, use_cache=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

# calibration fits isotonic regression after the binary classifier is fit
# monotone constraints help the LightGBM classifier behave sensibly
pairwise_model = PairwiseModeler(
    n_iter=25, calibrate=True, monotone_constraints=featurization_info.lightgbm_monotone_constraints
)
# this does hyperparameter selection, which is why we need to pass in the validation set.
pairwise_model.fit(X_train, y_train, X_val, y_val)

# this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
pairwise_metrics = pairwise_eval(X_test, y_test, pairwise_model.classifier, figs_path='figs/', title='example')
print(pairwise_metrics)
```

The second stage in the S2AND pipeline is to tune hyperparameters for the clusterer on the validation data
and then evaluate the full clustering pipeline on the test blocks.

We use agglomerative clustering as implemented in `fastcluster` with average linkage.
There is only one hyperparameter to tune.

```python
from s2and.model import Clusterer, FastCluster
from hyperopt import hp

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)

# the metrics_per_signature are there so we can break out the facets if needed
metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

For a fuller example, please see the transfer script: `scripts/transfer_experiment.py`.

## How to use S2AND for predicting with a saved model
Assuming you have a clusterer already fit, you can dump the model to disk like so
```python
import pickle

with open("saved_model.pkl", "wb") as _pkl_file:
    pickle.dump(clusterer, _pkl_file)
```

You can then reload it, load a new dataset, and run prediction
```python
import pickle

with open("saved_model.pkl", "rb") as _pkl_file:
    clusterer = pickle.load(_pkl_file)

anddata = ANDData(
    signatures=signatures,
    papers=papers,
    specter_embeddings=paper_embeddings,
    name="your_name_here",
    mode="inference",
    block_type="s2",
)
pred_clusters, pred_distance_matrices = clusterer.predict(anddata.get_blocks(), anddata)
```

### Incremental prediction
There is a also a `predict_incremental` function on the `Clusterer`, that allows prediction for just a small set of *new* signatures. When instantiating `ANDData`, you can pass in `cluster_seeds`, which will be used instead of model predictions for those signatures. If you call `predict_incremental`, the full distance matrix will not be created, and the new signatures will simply be assigned to the cluster they have the lowest average distance to, as long as it is below the model's `eps`, or separately reclustered with the other unassigned signatures, if not within `eps` of any existing cluster.

## Reproducibility
The experiments in the paper were run with the python (3.7.9) package versions in `paper_experiments_env.txt`, in the branch `s2and_paper`. 

To install, run:
```bash
git checkout s2and_paper
pip install pip==21.0.0
pip install -r paper_experiments_env.txt --use-feature=fast-deps --use-deprecated=legacy-resolver
``` 

Then, Rerunning `scripts/paper_experiments.sh` on the branch `s2and_paper` should produce the same numbers as in the paper (we will udpate here if this becomes not true). 

Our trained, released models are in the `s3` folder referenced above, and are called `production_model.pickle` (very close to what is running on the Semantic Scholar website, except the production model doesn't compute the reference features) and `full_union_seed_*.pickle` (models trained during benchmark experiments). They can be loaded the same way as in the section above called "How to use S2AND for predicting with a saved model", except that the pickled object is a *dictionary*, with a `clusterer` key. *Important*: these pickles will only run on the branch `s2and_paper` and not on main.

Note that by default we are using the `--use_cache` flag, which will cache all the features so future reruns are faster. There are two things to be aware of: (a) the cache is stored in RAM and can be huge (100gb+) and (b) if you intend to change the features and rerun, you'll have to turn off the cache or the new features won't be used.

## Licensing
The code in this repo is released under the Apache 2.0 license. The dataset is released under ODC-BY (included in S3 bucket with the data). We would also like to acknowledge that some of the affiliations data comes directly from the Microsoft Academic Graph (https://aka.ms/msracad).

## Citation

If you use S2AND in your research, please cite [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421).

```
@inproceedings{subramanian2021s2and,
      title={{S}2{AND}: {A} {B}enchmark and {E}valuation {S}ystem for {A}uthor {N}ame {D}isambiguation}, 
      author={Subramanian, Shivashankar and King, Daniel and Downey, Doug and Feldman, Sergey},
      year={2021},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      booktitle = {{JCDL} '21: Proceedings of the {ACM/IEEE} Joint Conference on Digital Libraries in 2021},
      series = {JCDL '21}
}
```

S2AND is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
