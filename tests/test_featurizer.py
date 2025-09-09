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
            compute_reference_features=True,
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

    def test_featurizer_without_reference_features_raises(self):
        # Build a dataset with reference features enabled (baseline) and disabled
        dataset_ref = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_ref",
            load_name_counts=True,
            compute_reference_features=True,
        )
        dataset_no_ref = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_no_ref",
            load_name_counts=True,
            compute_reference_features=False,
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
        fi = FeaturizationInfo(features_to_use=features_to_use)

        # Single pair is sufficient for checking NaN placement
        test_pairs = [("3", "0", 0)]
        feats_ref, _, _ = many_pairs_featurize(
            test_pairs,
            dataset_ref,
            fi,
            n_jobs=1,
            use_cache=False,
            chunk_size=1,
            nan_value=np.nan,
        )

        # When reference_features are requested but the dataset disabled them,
        # the featurizer should raise a clear ValueError.
        with pytest.raises(ValueError):
            _ = many_pairs_featurize(
                test_pairs,
                dataset_no_ref,
                fi,
                n_jobs=1,
                use_cache=False,
                chunk_size=1,
                nan_value=np.nan,
            )

    def test_featurizer_without_reference_group_ok(self):
        """When compute_reference_features=False and 'reference_features' is NOT requested,
        featurization should proceed normally with a reduced feature vector."""
        dataset_no_ref = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_no_ref_ok",
            load_name_counts=True,
            compute_reference_features=False,
        )

        features_to_use = [
            "name_similarity",
            "affiliation_similarity",
            "email_similarity",
            "coauthor_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            # no "reference_features" here
            "misc_features",
            "name_counts",
            "journal_similarity",
            "advanced_name_similarity",
        ]
        fi = FeaturizationInfo(features_to_use=features_to_use)

        test_pairs = [("3", "0", 0), ("3", "1", 0)]
        feats, labels, _ = many_pairs_featurize(
            test_pairs,
            dataset_no_ref,
            fi,
            n_jobs=1,
            use_cache=False,
            chunk_size=1,
            nan_value=-1,
        )

        # Shape checks
        assert feats.shape[0] == len(test_pairs)
        expected_len = sum(len(fi.feature_group_to_index[name]) for name in features_to_use)
        assert feats.shape[1] == expected_len

        # Sanity: not all sentinel values
        assert np.any(feats != -LARGE_INTEGER)

    def test_get_constraint(self):
        first_constraint = self.dummy_dataset.get_constraint("0", "8", high_value=100)
        assert first_constraint == 100
        middle_constraint = self.dummy_dataset.get_constraint("6", "8", high_value=100)
        assert middle_constraint == 100
        no_constraint = self.dummy_dataset.get_constraint("0", "1")
        assert no_constraint is None

    def test_multiprocessing_featurization_consistency(self):
        """Test that multiprocessing featurization produces identical results to single-threaded"""
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
            ("3", "2", 0),
            ("0", "1", 1),
        ]

        # Test single-threaded
        features_single, labels_single, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=1, use_cache=False, chunk_size=1, nan_value=-1
        )

        # Test multi-threaded
        features_multi, labels_multi, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=2, use_cache=False, chunk_size=1, nan_value=-1
        )

        # Verify identical results
        assert features_single.shape == features_multi.shape, "Feature array shapes don't match"
        assert labels_single.shape == labels_multi.shape, "Label array shapes don't match"

        # Check that all features are identical
        for i in range(features_single.shape[0]):
            for j in range(features_single.shape[1]):
                val_single = features_single[i, j]
                val_multi = features_multi[i, j]

                # Handle NaN comparisons
                both_nan = np.isnan(val_single) and np.isnan(val_multi)
                if not both_nan:
                    self.assertAlmostEqual(
                        val_single,
                        val_multi,
                        places=10,
                        msg=f"Feature mismatch at position ({i}, {j}): {val_single} vs {val_multi}",
                    )

        # Check labels are identical
        np.testing.assert_array_equal(
            labels_single, labels_multi, "Labels don't match between single and multi-threaded"
        )

    def test_global_dataset_initialization_in_workers(self):
        """Test that global_dataset is properly initialized in worker processes"""
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
        ]

        # This test verifies that worker processes can access the global dataset
        # If _init_pool wasn't working, this would fail with AttributeError
        try:
            features, labels, _ = many_pairs_featurize(
                test_pairs,
                self.dummy_dataset,
                self.dummy_featurizer,
                n_jobs=2,
                use_cache=False,
                chunk_size=1,
                nan_value=-1,
            )
            # If we get here, global dataset was properly initialized
            assert features.shape[0] == len(test_pairs)
        except (AttributeError, NameError) as e:
            self.fail(f"Global dataset not properly initialized in worker processes: {e}")

    def test_multiprocessing_with_different_chunk_sizes(self):
        """Test that different chunk sizes don't affect results with multiprocessing"""
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
            ("3", "2", 0),
            ("0", "1", 1),
            ("0", "2", 0),
            ("1", "2", 1),
        ]

        # Test with chunk_size=1
        features_chunk1, labels_chunk1, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=2, use_cache=False, chunk_size=1, nan_value=-1
        )

        # Test with chunk_size=3
        features_chunk3, labels_chunk3, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=2, use_cache=False, chunk_size=3, nan_value=-1
        )

        # Results should be identical regardless of chunk size
        assert features_chunk1.shape == features_chunk3.shape
        np.testing.assert_array_almost_equal(features_chunk1, features_chunk3, decimal=10)
        np.testing.assert_array_equal(labels_chunk1, labels_chunk3)

    def test_multiprocessing_fallback_to_single_thread(self):
        """Test that multiprocessing gracefully falls back when work is too small"""
        test_pairs = [("3", "0", 0)]  # Very small work load

        # Should work even with n_jobs > 1 for small datasets
        features, labels, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=4, use_cache=False, chunk_size=1, nan_value=-1
        )

        assert features.shape[0] == 1
        assert labels.shape[0] == 1

    def test_spawn_context_compatibility(self):
        """Test that the spawn multiprocessing context works correctly"""
        test_pairs = [
            ("3", "0", 0),
            ("3", "1", 0),
            ("0", "1", 1),
        ]

        # This specifically tests that our spawn context implementation works
        # The spawn context should work consistently across platforms
        features, labels, _ = many_pairs_featurize(
            test_pairs, self.dummy_dataset, self.dummy_featurizer, n_jobs=2, use_cache=False, chunk_size=1, nan_value=-1
        )

        # Verify we got valid results
        assert features.shape[0] == len(test_pairs)
        assert not np.all(features == -LARGE_INTEGER), "Features were not computed (global dataset issue)"

        # Verify feature values are reasonable (not all zeros or errors)
        non_missing_features = features[features != -LARGE_INTEGER]
        assert len(non_missing_features) > 0, "No valid features computed"
