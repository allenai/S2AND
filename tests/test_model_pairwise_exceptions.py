import numpy as np
import pytest

from s2and.model import FastCluster, VotingClassifier


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


def test_voting_classifier_invalid_voting_type_raises_value_error():
    classifier = VotingClassifier(estimators=[], voting="invalid")
    with pytest.raises(ValueError, match="Voting type must be one of"):
        classifier.predict(np.array([[0.0], [1.0]]))
