import unittest
import pytest
import numpy as np
import pickle

import s2and.featurizer
from s2and.data import ANDData
from s2and.model import Clusterer
from s2and.featurizer import FeaturizationInfo
from s2and.consts import LARGE_DISTANCE
import lightgbm as lgb


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if "global_dataset" in dir(s2and.featurizer):
            del s2and.featurizer.global_dataset

        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            cluster_seeds="tests/dummy/cluster_seeds.json",
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
            use_default_constraints_as_supervision=False,
        )

    def test_get_constraints(self):
        block = {
            "a sattar": ["0", "1", "2"],
        }
        constraint_1 = self.dummy_dataset.get_constraint("0", "1", low_value=0, high_value=2)
        constraint_2 = self.dummy_dataset.get_constraint("1", "0", low_value=0, high_value=2)
        constraint_3 = self.dummy_dataset.get_constraint("1", "2", low_value=0, high_value=2)
        constraint_4 = self.dummy_dataset.get_constraint("2", "1", low_value=0, high_value=2)

        self.assertIs(constraint_1, LARGE_DISTANCE)
        self.assertIs(constraint_2, LARGE_DISTANCE)
        self.assertIs(constraint_3, 0)
        self.assertIs(constraint_4, 0)

    def test_make_distance_matrix_fastcluster(self):
        block = {
            "a sattar": ["0", "1", "2"],
        }
        partial_supervision = {("0", "1"): 1.1, ("1", "2"): 1e-6}
        distance_matrices = self.dummy_clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dummy_dataset,
            partial_supervision=partial_supervision,
        )
        distance_matrix = distance_matrices["a sattar"]
        self.assertEqual(distance_matrix[0], np.float16(1.1))
        self.assertEqual(distance_matrix[1], np.float16(0.3))
        self.assertEqual(distance_matrix[2], np.float16(1e-6))

        distance_matrices = self.dummy_clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dummy_dataset,
            partial_supervision={},
        )
        distance_matrix = distance_matrices["a sattar"]
        self.assertEqual(distance_matrix[0], np.float16(0.3))
        self.assertEqual(distance_matrix[1], np.float16(0.3))
        self.assertEqual(distance_matrix[2], np.float16(0.3))

    def test_subblocking(self):
        block = {
            "a sattar": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        }
        prediction_full, _ = self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=None)
        # all go together
        self.assertEqual(prediction_full["a sattar_1"], block["a sattar"])

        # now with batching
        # interestingly, this causes an odd outcome where the subblock clustering is different
        prediction_full, _ = self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=7)
        prediction_subblock_1, _ = self.dummy_clusterer.predict(
            {"a sattar|subblock=ab": ["0", "1", "2"]}, self.dummy_dataset
        )
        self.assertEqual(prediction_full["a sattar|subblock=ab_1"], prediction_subblock_1["a sattar|subblock=ab_1"])

        # stricter batching - just making sure it doesn't break
        self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=2)
        self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=1)
