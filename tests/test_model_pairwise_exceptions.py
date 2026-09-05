import numpy as np
import pytest
from scipy.spatial.distance import pdist
from sklearn.base import clone

from s2and.model import FastCluster


def test_fastcluster_invalid_linkage_raises_value_error():
    with pytest.raises(ValueError, match="linkage"):
        FastCluster(linkage="invalid")


def test_fastcluster_fit_rejects_1d_input_when_observation_matrix_expected():
    clusterer = FastCluster(input_as_observation_matrix=True)
    with pytest.raises(ValueError, match="one-dimensional"):
        clusterer.fit(np.array([0.1, 0.2, 0.3]))


def test_fastcluster_fit_rejects_2d_input_when_distance_matrix_expected():
    clusterer = FastCluster(input_as_observation_matrix=False)
    with pytest.raises(ValueError, match="two-dimensional"):
        clusterer.fit(np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_fastcluster_fit_rejects_inputs_above_2_dimensions():
    clusterer = FastCluster(input_as_observation_matrix=False)
    with pytest.raises(ValueError, match="one-dimensional or two-dimensional"):
        clusterer.fit(np.zeros((2, 2, 2)))


def test_fastcluster_transform_raises_not_implemented_error():
    clusterer = FastCluster()
    with pytest.raises(NotImplementedError, match="no inductive mode"):
        clusterer.transform(np.array([0.1, 0.2, 0.3]))


@pytest.mark.parametrize("observations", [False, True], ids=["condensed", "observations"])
def test_fastcluster_clone_refit_and_input_preservation(observations: bool) -> None:
    """Cloned models retain their linkage and threshold without sharing fitted labels."""
    points = np.asarray([[0.0], [0.2], [0.5], [2.0]])
    values = points if observations else pdist(points)
    original_values = values.copy()
    source = FastCluster(linkage="complete", eps=0.4, input_as_observation_matrix=observations)
    source_labels = source.fit_transform(values).copy()
    copied = clone(source)

    assert copied is not source
    assert copied.labels_ is None
    copied.set_params(eps=0.5)
    assert copied.fit(values) is copied
    # Complete linkage keeps the third point out at 0.4; single or average
    # linkage would merge it. The exact 0.5 boundary then joins the first three.
    np.testing.assert_array_equal(
        source_labels[:, None] == source_labels,
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ],
    )
    np.testing.assert_array_equal(
        copied.labels_[:, None] == copied.labels_,
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
            [False, False, False, True],
        ],
    )
    np.testing.assert_array_equal(source.labels_, source_labels)
    np.testing.assert_array_equal(values, original_values)

    # Refit must replace, rather than reuse, labels from the previous population.
    next_values = np.asarray([[0.0], [2.0]]) if observations else np.asarray([2.0])
    next_labels = copied.fit_transform(next_values)
    assert len(next_labels) == 2
    assert next_labels[0] != next_labels[1]
