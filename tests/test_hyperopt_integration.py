import numpy as np
from hyperopt import Trials, hp
from sklearn.linear_model import LogisticRegression

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, PairwiseModeler
from tests.helpers import build_dummy_dataset


def test_pairwise_modeler_hyperopt_small():
    rng = np.random.RandomState(0)
    X_train = rng.normal(size=(12, 3))
    y_train = np.array([0, 1] * 6)
    X_val = rng.normal(size=(6, 3))
    y_val = np.array([0, 1, 0, 1, 0, 1])

    estimator = LogisticRegression(max_iter=50, solver="liblinear")
    search_space = {"C": hp.uniform("C", 0.1, 1.0)}

    modeler = PairwiseModeler(
        estimator=estimator,
        search_space=search_space,
        n_iter=4,
        n_jobs=1,
        random_state=0,
    )

    trials = modeler.fit(X_train, y_train, X_val, y_val)
    assert modeler.best_params is not None
    assert "C" in modeler.best_params
    assert isinstance(trials, Trials)
    assert len(trials.trials) == 4

    probs = modeler.predict_proba(X_val)
    assert probs.shape == (6, 2)


def test_clusterer_default_search_space_fits_with_extracted_metrics(monkeypatch):
    """Exercise default Hyperopt setup, B3 scoring, and cached clustering together."""
    dataset = build_dummy_dataset("cluster_hyperopt_imports")
    block = {"tiny": ["0", "1", "2"]}
    monkeypatch.setattr(dataset, "split_cluster_signatures", lambda: ({}, block, {}))
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        classifier=None,
        n_iter=2,
        n_jobs=1,
        random_state=0,
    )
    distances = {"tiny": np.array([0.0, 1.0, 1.0])}

    assert clusterer.fit(dataset, val_dists_precomputed={dataset.name: distances}) is clusterer
    assert isinstance(clusterer.hyperopt_trials_store, Trials)
    assert len(clusterer.hyperopt_trials_store.trials) == 2
    assert clusterer.hyperopt_trials_store.losses() == [-1.0, -1.0]
    assert 0 <= clusterer.best_params["eps"] <= 1


def test_pairwise_modeler_resets_trials_when_search_space_empty():
    rng = np.random.RandomState(1)
    X_train = rng.normal(size=(12, 3))
    y_train = np.array([0, 1] * 6)
    X_val = rng.normal(size=(6, 3))
    y_val = np.array([0, 1, 0, 1, 0, 1])

    modeler = PairwiseModeler(
        estimator=LogisticRegression(max_iter=50, solver="liblinear"),
        search_space={"C": hp.uniform("C_reset_test", 0.1, 1.0)},
        n_iter=3,
        n_jobs=1,
        random_state=1,
    )

    first_trials = modeler.fit(X_train, y_train, X_val, y_val)
    assert isinstance(first_trials, Trials)
    assert len(first_trials.trials) == 3

    modeler.search_space = {}
    second_trials = modeler.fit(X_train, y_train, X_val, y_val)

    assert modeler.best_params == {}
    assert second_trials == {}
    assert modeler.hyperopt_trials_store == {}
