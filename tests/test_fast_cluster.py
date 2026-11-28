"""Check that FastCluster keeps the scikit-learn parameter API without inheriting from scikit-learn."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import DBSCAN

from s2and.model_pairwise import FastCluster, clone_cluster_model


def test_get_and_set_params_follow_the_scikit_learn_contract() -> None:
    model = FastCluster(linkage="single", eps=0.3)

    assert model.get_params() == {
        "close_pair_bonus": 0.0,
        "close_pair_distance": 0.0,
        "eps": 0.3,
        "input_as_observation_matrix": False,
        "linkage": "single",
        "preserve_input": True,
    }
    assert model.get_params(deep=False) == model.get_params()
    assert model.set_params(eps=0.7, preserve_input=False) is model
    assert (model.eps, model.preserve_input) == (0.7, False)
    with pytest.raises(ValueError, match="Invalid parameter 'n_jobs'"):
        model.set_params(n_jobs=4)


def test_clones_are_unfitted_with_the_same_parameters() -> None:
    model = FastCluster(linkage="complete", eps=0.4).fit(np.asarray([0.1, 0.9, 0.9]))
    assert model.labels_ is not None

    for copied in (clone_cluster_model(model), clone(model)):
        assert type(copied) is FastCluster
        assert copied is not model
        assert copied.get_params() == model.get_params()
        assert copied.labels_ is None

    fallback = DBSCAN(eps=0.2, metric="precomputed")
    copied_fallback = clone_cluster_model(fallback)
    assert type(copied_fallback) is DBSCAN
    assert copied_fallback is not fallback
    assert copied_fallback.get_params() == fallback.get_params()
