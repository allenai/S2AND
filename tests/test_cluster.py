import unittest
import pytest
import numpy as np

from s2and.data import PDData
from s2and.model import Clusterer
from s2and.featurizer import FeaturizationInfo
import lightgbm as lgb


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            load_name_counts=True
        )

        features_to_use = [
            "year_diff",
            "title_similarity",
        ]
        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 3))
        y_random = np.random.randint(0, 3, 10)
        self.clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=False,
        )

    def test_make_distance_matrix_fastcluster(self):
        block = {
            'reviewerlistfor': [
                '84177344',
                '49188235',
                '214237506',
                '217917498', 
                '1473469382'
            ]
        }
        partial_supervision = {("84177344", "49188235"): 1.1, ("49188235", "214237506"): 1e-6}
        distance_matrices = self.clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dataset,
            partial_supervision=partial_supervision,
        )
        distance_matrix = distance_matrices["reviewerlistfor"]
        self.assertEqual(distance_matrix[0], np.float16(1.1))
        self.assertEqual(distance_matrix[1], np.float16(0.3))
        self.assertEqual(distance_matrix[4], np.float16(1e-6))

        distance_matrices = self.clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dataset,
            partial_supervision={},
        )
        distance_matrix = distance_matrices["reviewerlistfor"]
        self.assertEqual(distance_matrix[0], np.float16(0.3))
        self.assertEqual(distance_matrix[1], np.float16(0.3))
        self.assertEqual(distance_matrix[4], np.float16(0.3))