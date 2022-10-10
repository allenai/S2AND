import unittest
import pytest
import numpy as np

from s2and.data import PDData
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.consts import LARGE_INTEGER


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            load_name_counts=False,
            balanced_pair_sample=False,
        )

        features_to_use = [
            "author_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            "abstract_similarity"
        ]
        self.featurizer = FeaturizationInfo(features_to_use=features_to_use)

    def check_features_array_equal(self, array_1, array_2):
        assert len(array_1) == len(array_2)
        for i in range(len(array_1)):
            both_nan = np.isnan(array_1[i]) and np.isnan(array_2[i])
            if not both_nan:
                self.assertAlmostEqual(array_1[i], array_2[i], msg=i)

    def test_featurizer(self):
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
            ("3", "2", 0),
            ("3", "2", -1),
        ]
        
        self.dataset.train_pairs_size = 100
        self.dataset.val_pairs_size = 50
        self.dataset.test_pairs_size = 3
        self.dataset.random_seed = 1111
        (
            train_block_dict,
            val_block_dict,
            test_block_dict,
        ) = self.dataset.split_cluster_papers()
        _, _, test_pairs = self.dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        test_pair_neg_1 = list(test_pairs[0])
        test_pair_neg_1[-1] = -1
        test_pairs.append(tuple(test_pair_neg_1))
        
        # single thread
        features, labels, _ = many_pairs_featurize(
            test_pairs, self.dataset, self.featurizer, 1, False, 1, nan_value=-1
        )
        # multi thread: currently broken???
        # TODO: fix the multithreading global issue...
        features, _, _ = many_pairs_featurize(
            test_pairs, self.dataset, self.featurizer, 2, False, 1, nan_value=-1
        )

        expected_features_1 = [-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, -1.0]
        expected_features_2 = [-1.0, -1.0, -1.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0]
        expected_features_3 = [
            0.06319702602230483,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            -1.0
        ]
        self.check_features_array_equal(list(features[0, :]), expected_features_1)
        self.check_features_array_equal(list(features[1, :]), expected_features_2)
        self.check_features_array_equal(list(features[2, :]), expected_features_3)
