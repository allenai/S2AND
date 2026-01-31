import numpy as np
from functools import partial

from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression

from s2and.model import PairwiseModeler, intify


def test_hyperopt_fmin_space_eval_with_tpe_partial():
    search_space = {
        "eps": hp.uniform("eps", 0.0, 1.0),
        "max_depth": scope.int(hp.quniform("max_depth", 1, 3, 1)),
        "learning_rate": hp.loguniform("learning_rate", -7, 0),
    }

    def obj(params):
        params = {k: intify(v) for k, v in params.items()}
        return (params["eps"] - 0.2) ** 2 + (params["max_depth"] - 2) ** 2 + params["learning_rate"]

    trials = Trials()
    _ = fmin(
        fn=obj,
        space=search_space,
        algo=partial(tpe.suggest, n_startup_jobs=2),
        max_evals=5,
        trials=trials,
        rstate=np.random.default_rng(0),
    )

    assert len(trials.trials) == 5
    best_params = space_eval(search_space, trials.argmin)
    best_params = {k: intify(v) for k, v in best_params.items()}
    assert set(best_params.keys()) == {"eps", "max_depth", "learning_rate"}
    assert isinstance(best_params["max_depth"], int)


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
    assert trials is not None
    assert len(trials.trials) == 4

    probs = modeler.predict_proba(X_val)
    assert probs.shape == (6, 2)
