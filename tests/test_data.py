import unittest
import pytest

from s2and.data import PDData


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dataset = PDData(
            "tests/test_dataset/papers.json",
            clusters="tests/test_dataset/clusters.json",
            name="test_dataset",
            balanced_pair_sample=False
        )

    def test_split_pairs_within_blocks(self):
        # Test random sampling within blocks
        self.dataset.train_pairs_size = 100
        self.dataset.val_pairs_size = 50
        self.dataset.test_pairs_size = 50
        self.dataset.random_seed = 1111
        (
            train_block_dict,
            val_block_dict,
            test_block_dict,
        ) = self.dataset.split_cluster_papers()
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )

        assert len(train_pairs) == 100 and len(val_pairs) == 50 and len(test_pairs) == 50
        assert (
            train_pairs[0] == ('1409183546974670848', '1401326657880461312', 0)
            and val_pairs[0] == ('35668499', '212140021', 1)
            and test_pairs[0] == ('65517728', '1458733052', 0)
        )

        # Test adding the all test pairs flag to the test above
        self.dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert len(train_pairs) == 100, len(val_pairs) == 50 and len(test_pairs) == 72


    def test_initialization(self):
        dataset = PDData(papers={}, name="", mode="inference", load_name_counts=False)
        assert dataset.paper_to_cluster_id is None
        assert dataset.all_test_pairs_flag

    def test_construct_cluster_to_signatures(self):
        cluster_to_signatures = self.dataset.construct_cluster_to_papers({"a": ["20797514", "51804247"], "b": ["7355243", "65911307"]})
        expected_cluster_to_signatures = {'PM_51910': ['20797514'],
            'PM_114482': ['51804247'],
            'PM_221928': ['7355243'],
            'PM_146905': ['65911307']
        }
        assert cluster_to_signatures == expected_cluster_to_signatures