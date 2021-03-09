import unittest
import pytest
import numpy as np

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.consts import LARGE_INTEGER


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy",
            load_name_counts=True,
        )

        features_to_use = [
            "name_similarity",
            "affiliation_similarity",
            "email_similarity",
            "coauthor_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            "reference_features",
            "misc_features",
            "name_counts",
            "journal_similarity",
            "advanced_name_similarity",
        ]
        self.dummy_featurizer = FeaturizationInfo(features_to_use=features_to_use)

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
        features, labels, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, 2, False, 1, nan_value=-1
        )

        expected_features_1 = [
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            4.0,
            0.0,
            0.03067484662576687,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            82081.0,
            12.0,
            807.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            0.7777777777777778,
            0.8,
            0.7777777777777778,
            0.5407407407407407,
        ]
        expected_features_2 = [
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            6.0,
            0.02857142857142857,
            0.09615384615384616,
            0.25757575757575757,
            0.34615384615384615,
            0.8181818181818182,
            0.2222222222222222,
            0.0,
            0.5,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            23425.0,
            12.0,
            807.0,
            1.0,
            82081.0,
            20.0,
            -1.0,
            0.7777777777777778,
            0.8,
            0.7777777777777778,
            0.5407407407407407,
        ]
        expected_features_3 = [
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.2,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            6.0,
            0.0,
            0.058823529411764705,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            1.0,
            -1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            2.0,
            23425.0,
            12.0,
            807.0,
            1.0,
            82081.0,
            20.0,
            -1.0,
            0.7777777777777778,
            0.8,
            0.7777777777777778,
            0.5407407407407407,
        ]
        self.check_features_array_equal(list(features[0, :]), expected_features_1)
        self.check_features_array_equal(list(features[1, :]), expected_features_2)
        self.check_features_array_equal(list(features[2, :]), expected_features_3)
        self.assertEqual(features[3, 0], -LARGE_INTEGER)

    def test_get_constraint(self):
        first_constraint = self.dummy_dataset.get_constraint("0", "8", high_value=100)
        assert first_constraint == 100
        middle_constraint = self.dummy_dataset.get_constraint("6", "8", high_value=100)
        assert middle_constraint == 100
        no_constraint = self.dummy_dataset.get_constraint("0", "1")
        assert no_constraint is None