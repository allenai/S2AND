import unittest
import pytest
import numpy as np
import pickle

from s2and.data import ANDData
from s2and.model import Clusterer
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.consts import LARGE_DISTANCE
from sklearn.linear_model import LogisticRegression


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures_incompatible.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            cluster_seeds={"1": {"2": "require"}},
            altered_cluster_signatures=["1", "2"],
            name="dummy",
            load_name_counts=True,
        )

        features_to_use = [
            "affiliation_similarity",
        ]
        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X = np.vstack([np.ones((100, 1)), np.random.uniform(0, 0.5, (100, 1))])
        y = np.vstack([np.ones((100, 1)), np.zeros((100, 1))]).squeeze()
        clf = LogisticRegression().fit(X, y)
        self.dummy_clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=clf,
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

    def test_predict_incremental(self):
        block = ["3"]
        output = self.dummy_clusterer.predict_incremental(block, self.dummy_dataset)
        expected_output = {"0": ["1", "2"], "1": ["3"]}
        assert output == expected_output

        block = ["3"]
        output = self.dummy_clusterer.predict_incremental(
            block, self.dummy_dataset, prevent_new_incompatibilities=False
        )
        expected_output = {"0": ["1", "2", "3"]}
        assert output == expected_output
