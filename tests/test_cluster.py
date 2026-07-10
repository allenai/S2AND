import os
import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
from lightgbm import LGBMClassifier

from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from tests.helpers import tiny_name_counts_index


class TestClusterer(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._previous_s2and_backend = os.environ.get("S2AND_BACKEND")
        os.environ["S2AND_BACKEND"] = "python"
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            cluster_seeds="tests/dummy/cluster_seeds.json",
            name="dummy",
            name_counts_index=tiny_name_counts_index(),
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
            classifier=LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1, verbosity=-1).fit(
                X_random, y_random
            ),
            n_jobs=1,
            use_cache=False,
            use_default_constraints_as_supervision=False,
        )

    def tearDown(self):
        if self._previous_s2and_backend is None:
            os.environ.pop("S2AND_BACKEND", None)
        else:
            os.environ["S2AND_BACKEND"] = self._previous_s2and_backend
        super().tearDown()

    def _fill_python_featurizer_fields(self):
        for signature_id, signature in list(self.dummy_dataset.signatures.items()):
            replacements = {}
            if signature.author_info_first_normalized_without_apostrophe is None:
                replacements["author_info_first_normalized_without_apostrophe"] = (
                    (signature.author_info_first or "").replace("'", "").lower()
                )
            if signature.author_info_middle_normalized_without_apostrophe is None:
                replacements["author_info_middle_normalized_without_apostrophe"] = ""
            if signature.author_info_affiliations_n_grams is None:
                replacements["author_info_affiliations_n_grams"] = Counter()
            if signature.author_info_coauthor_blocks is None:
                replacements["author_info_coauthor_blocks"] = set()
            if signature.author_info_coauthor_n_grams is None:
                replacements["author_info_coauthor_n_grams"] = Counter()
            if signature.author_info_coauthors is None:
                replacements["author_info_coauthors"] = set()
            if replacements:
                self.dummy_dataset.signatures[signature_id] = signature._replace(**replacements)
        for paper_id, paper in list(self.dummy_dataset.papers.items()):
            replacements = {}
            if paper.venue_ngrams is None:
                replacements["venue_ngrams"] = Counter()
            if paper.title_ngrams_words is None:
                replacements["title_ngrams_words"] = Counter()
            if paper.title_ngrams_chars is None:
                replacements["title_ngrams_chars"] = Counter()
            if paper.journal_ngrams is None:
                replacements["journal_ngrams"] = Counter()
            if paper.is_reliable is None:
                replacements["is_reliable"] = False
            if replacements:
                self.dummy_dataset.papers[paper_id] = paper._replace(**replacements)

    def test_get_constraints(self):
        constraint_1 = self.dummy_dataset.get_constraint("0", "1", low_value=0, high_value=2)
        constraint_2 = self.dummy_dataset.get_constraint("1", "0", low_value=0, high_value=2)
        constraint_3 = self.dummy_dataset.get_constraint("1", "2", low_value=0, high_value=2)
        constraint_4 = self.dummy_dataset.get_constraint("2", "1", low_value=0, high_value=2)

        self.assertIs(constraint_1, LARGE_DISTANCE)
        self.assertIs(constraint_2, LARGE_DISTANCE)
        self.assertIs(constraint_3, 0)
        self.assertIs(constraint_4, 0)

    def test_incremental_dont_use_cluster_seeds_keeps_explicit_disallow(self):
        self.dummy_dataset.cluster_seeds_disallow = {("0", "1")}
        self.dummy_dataset.cluster_seeds_require = {"0": 0, "1": 1}
        constraint = self.dummy_dataset.get_constraint(
            "0",
            "1",
            low_value=0,
            high_value=2,
            incremental_dont_use_cluster_seeds=True,
        )
        self.assertIs(constraint, LARGE_DISTANCE)

    def test_incremental_dont_use_cluster_seeds_ignores_cross_seed_disallow(self):
        self.dummy_dataset.cluster_seeds_disallow = set()
        self.dummy_dataset.cluster_seeds_require = {"0": 0, "1": 1}
        constraint = self.dummy_dataset.get_constraint(
            "0",
            "1",
            low_value=0,
            high_value=2,
            incremental_dont_use_cluster_seeds=True,
        )
        self.assertIsNone(constraint)

    def test_cluster_one_block_empty_returns_no_labels(self):
        labels = self.dummy_clusterer._cluster_one_block([], np.zeros((0, 0)), None, self.dummy_dataset, set())

        self.assertEqual(labels, [])

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
        np.testing.assert_allclose(
            distance_matrix,
            np.asarray([1.1, 0.3, 1e-6], dtype=np.float64),
            rtol=0,
            atol=1e-3,
        )

        distance_matrices = self.dummy_clusterer.make_distance_matrices(
            block_dict=block,
            dataset=self.dummy_dataset,
            partial_supervision={},
        )
        distance_matrix = distance_matrices["a sattar"]
        np.testing.assert_allclose(
            distance_matrix,
            np.asarray([0.3, 0.3, 0.3], dtype=np.float64),
            rtol=0,
            atol=1e-3,
        )

    def test_subblocking(self):
        self._fill_python_featurizer_fields()
        block = {
            "a sattar": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        }
        prediction_full, _ = self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=None)
        # all go together
        self.assertEqual(prediction_full["a sattar_1"], block["a sattar"])

        # now with batching
        # interestingly, this causes an odd outcome where the subblock clustering is different
        # The checked-in legacy ORCID priors are intentionally unavailable to
        # the strict production loader until the replacement snapshot is
        # published. This unit test exercises the subblocking algorithm with a
        # small explicit prior fixture rather than crossing that artifact
        # boundary.
        with patch("s2and.subblocking._resolved_orcid_prefix_counts", return_value={}):
            prediction_full, _ = self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=7)
            prediction_subblock_1, _ = self.dummy_clusterer.predict(
                {"a sattar|subblock=ab": ["0", "1", "2"]}, self.dummy_dataset
            )
            self.assertTrue(
                set(prediction_full["a sattar|subblock=ab_1"]).issubset(
                    set(prediction_subblock_1["a sattar|subblock=ab_1"])
                )
            )

            # stricter batching - just making sure it doesn't break
            self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=2)
            self.dummy_clusterer.predict(block, self.dummy_dataset, batching_threshold=1)

    def test_fused_path_equivalence(self):
        """The fused cluster-and-free path (dists=None) must produce
        identical pred_clusters as the two-phase path (precomputed dists)."""
        block = {
            "a sattar": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        }
        # two-phase: build dists first, then cluster from precomputed
        dists = self.dummy_clusterer.make_distance_matrices(block, self.dummy_dataset)
        precomputed_result, _ = self.dummy_clusterer.predict_helper(block, self.dummy_dataset, dists=dists)

        # fused: dists=None triggers cluster-and-free
        fused_result, fused_dists = self.dummy_clusterer.predict_helper(block, self.dummy_dataset, dists=None)

        self.assertIsNone(fused_dists)
        self.assertEqual(precomputed_result, fused_result)

    def test_predict_helper_missing_precomputed_dists_raises(self):
        block = {
            "a sattar": ["0", "1", "2"],
        }
        with pytest.raises(KeyError, match="Missing precomputed distance matrix for block"):
            self.dummy_clusterer.predict_helper(block, self.dummy_dataset, dists={})
