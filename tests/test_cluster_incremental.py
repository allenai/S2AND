import unittest
import pytest
import numpy as np
import pickle

from s2and.data import ANDData
from s2and.model import Clusterer
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.consts import LARGE_DISTANCE
import lightgbm as lgb


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            cluster_seeds={"6": {"7": "require"}, "3": {"4": "require"}},
            name="dummy",
            load_name_counts=True,
        )

        features_to_use = [
            "year_diff",
            "misc_features",
        ]
        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 6))
        y_random = np.random.randint(0, 6, 10)
        self.dummy_clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

    def test_predict_incremental(self):
        # base clustering of the random model would be
        # {'0': ['0', '1', '2'], '1': ['3', '4', '5', '8'], '2': ['6', '7']}

        block = ["3", "4", "5", "6", "7", "8"]
        output = self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=3)
        expected_output = {"0": ["6", "7", "5"], "1": ["3", "4", "8"]}
        assert output == expected_output

        block = ["3", "4", "5", "6", "7", "8"]
        output = self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=None)
        expected_output = {"0": ["6", "7"], "1": ["3", "4", "5", "8"]}
        assert output == expected_output

        self.dummy_dataset.cluster_seeds_disallow = {("5", "7"), ("8", "4"), ("5", "4"), ("8", "7")}
        output = self.dummy_clusterer.predict_incremental(block, self.dummy_dataset)
        expected_output = {"0": ["6", "7"], "1": ["3", "4"], "2": ["5", "8"]}
        assert output == expected_output

        self.dummy_dataset.altered_cluster_signatures = ["1", "5"]
        self.dummy_dataset.cluster_seeds_require = {"1": 0, "2": 0, "5": 0, "6": 1, "7": 1}
        block = ["3", "4", "8"]
        output = self.dummy_clusterer.predict_incremental(block, self.dummy_dataset, batching_threshold=None)
        expected_output = {"0": ["1", "2", "5", "8"], "1": ["6", "7", "3", "4"]}
        assert output == expected_output
