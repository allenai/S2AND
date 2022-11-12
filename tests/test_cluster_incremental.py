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
            cluster_seeds={"84177344": {"49188235": "require"}, "214237506": {"217917498": "require"}},
            name="test_dataset",
            load_name_counts=False,
            balanced_pair_sample=False,
        )

        features_to_use = [
            "year_diff",
            "title_similarity",
        ]

        featurizer_info = FeaturizationInfo(features_to_use=features_to_use)
        np.random.seed(1)
        X_random = np.random.random((10, 9))
        y_random = np.random.randint(0, 9, 10)
        self.clusterer = Clusterer(
            featurizer_info=featurizer_info,
            classifier=lgb.LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=True,
        )

    def test_predict_incremental(self):
        block_papers = ["49188235", "84177344", "214237506", "217917498", "210102606", "1400649365030178816"]
        output = self.clusterer.predict_incremental(block_papers, self.dataset)
        expected_output = {
            "0": ["84177344", "49188235", "210102606", "1400649365030178816"],
            "1": ["214237506", "217917498"],
        }
        assert output == expected_output

        self.dataset.cluster_seeds_disallow = {("84177344", "210102606"), ("49188235", "1400649365030178816")}
        output = self.clusterer.predict_incremental(block_papers, self.dataset)
        expected_output = {
            "0": ["84177344", "49188235"],
            "1": ["214237506", "217917498", "210102606", "1400649365030178816"],
        }
        assert output == expected_output
