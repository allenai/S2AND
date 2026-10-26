import numpy as np
from hyperopt import Trials, hp
from sklearn.linear_model import LogisticRegression

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, PairwiseModeler
from tests.helpers import build_dummy_dataset


def test_deferred_calibration_matches_explicit_default_space(monkeypatch):
    """Deferring setup preserves optimizer samples, scores, EPS, and predictions."""
    dataset = build_dummy_dataset("deferred_cluster_calibration")
    block = {"tiny": ["0", "1", "2"]}
    monkeypatch.setattr(dataset, "split_cluster_signatures", lambda: ({}, block, {}))
    distances = {"tiny": np.array([0.2, 0.8, 0.9])}
    explicit_space = {"eps": hp.uniform("eps", 0, 1)}
    deferred, explicit = [
        Clusterer(
            FeaturizationInfo(["year_diff"]),
            classifier=None,
            search_space=space,
            n_iter=8,
            n_jobs=1,
            random_state=0,
        )
        for space in (None, explicit_space)
    ]
    assert deferred.search_space is None
    for clusterer in (deferred, explicit):
        assert clusterer.fit(dataset, val_dists_precomputed={dataset.name: distances}) is clusterer
        assert isinstance(clusterer.hyperopt_trials_store, Trials)
        assert len(clusterer.hyperopt_trials_store.trials) == 8
        assert 0 <= clusterer.best_params["eps"] <= 1
    assert explicit.search_space is explicit_space
    assert deferred.hyperopt_trials_store.vals == explicit.hyperopt_trials_store.vals
    assert deferred.hyperopt_trials_store.losses() == explicit.hyperopt_trials_store.losses()
    assert len(set(deferred.hyperopt_trials_store.losses())) > 1
    assert deferred.best_params == explicit.best_params
    assert deferred.predict(block, dataset, dists=distances)[0] == explicit.predict(block, dataset, dists=distances)[0]


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
    assert modeler.best_params == {"C": first_trials.best_trial["misc"]["vals"]["C_reset_test"][0]}
    assert 0.1 <= modeler.best_params["C"] <= 1.0
    assert modeler.estimator.get_params()["C"] == modeler.best_params["C"]
    assert modeler.predict_proba(X_val).shape == (6, 2)

    modeler.search_space = {}
    second_trials = modeler.fit(X_train, y_train, X_val, y_val)

    assert modeler.best_params == {}
    assert second_trials == {}
    assert modeler.hyperopt_trials_store == {}


def test_pairwise_modeler_feature_names_change_only_the_saved_names():
    """Named final fits write names into the booster and keep every tree and prediction."""
    from lightgbm import LGBMClassifier

    rng = np.random.RandomState(3)
    X_train = rng.normal(size=(400, 3))
    X_train[rng.rand(*X_train.shape) < 0.1] = np.nan
    y_train = (np.nan_to_num(X_train[:, 0]) + rng.normal(size=400) > 0).astype(int)
    X_val = rng.normal(size=(50, 3))
    names = ["first_names_equal", "year_diff", "specter_cosine_sim"]
    fitted = []
    for feature_names in (None, names):
        modeler = PairwiseModeler(
            estimator=LGBMClassifier(n_estimators=20, num_leaves=7, n_jobs=1, verbose=-1, random_state=0),
            search_space={},
            n_jobs=1,
        )
        modeler.fit(X_train, y_train, X_val, y_train[:50], feature_names=feature_names)
        fitted.append(modeler)
    unnamed, named = fitted

    assert named.classifier.booster_.feature_name() == names
    unnamed_text = unnamed.classifier.booster_.model_to_string()
    for index, name in enumerate(names):
        unnamed_text = unnamed_text.replace(f"Column_{index}", name)
    assert named.classifier.booster_.model_to_string() == unnamed_text
    np.testing.assert_array_equal(named.predict_proba(X_val), unnamed.predict_proba(X_val))
