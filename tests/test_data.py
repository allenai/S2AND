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
            load_name_counts=False
        )
        self.dummy_dataset = PDData(
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy",
            load_name_counts=False,
            preprocess=False,
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
            and test_pairs[0] == ('7355243', '51939837', 0)
        )

        # Test adding the all test pairs flag to the test above
        self.dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert len(train_pairs) == 100, len(val_pairs) == 50 and len(test_pairs) == 72


    def test_blocks(self):
        original_blocks = self.dummy_dataset.get_original_blocks()
        s2_blocks = self.dummy_dataset.get_s2_blocks()

        expected_original_blocks = {
            "a sattar": ["0", "1", "2"],
            "a konovalov": ["3", "4", "5", "6", "7", "8"],
        }
        expected_s2_blocks = {
            "a sattary": ["0", "1", "2"],
            "a konovalov": ["3", "4", "5", "6", "7", "8"],
        }

        self.dummy_dataset.block_type = "s2"
        s2_blocks_2 = self.dummy_dataset.get_blocks()
        self.dummy_dataset.block_type = "original"
        original_blocks_2 = self.dummy_dataset.get_blocks()
        self.dummy_dataset.block_type = "dummy"
        with pytest.raises(Exception):
            blocks = self.dummy_dataset.get_blocks()
        self.dummy_dataset.block_type = "s2"

        assert original_blocks == expected_original_blocks
        assert original_blocks_2 == expected_original_blocks
        assert s2_blocks == expected_s2_blocks
        assert s2_blocks_2 == expected_s2_blocks

    def test_initialization(self):
        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                clusters={},
                name="",
                mode="train",
                train_blocks=[],
                block_type="s2",
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                clusters={},
                name="",
                mode="train",
                unit_of_data_split="blocks",
                pair_sampling_block=False,
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                name="",
                mode="train",
                clusters={},
                train_pairs=[],
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                name="",
                mode="train",
                clusters=None,
                train_pairs=None,
                train_blocks=None,
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                name="",
                mode="train",
                train_blocks=[],
                train_pairs=[],
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={},
                papers={},
                name="",
                mode="train",
                train_blocks=[],
                clusters=None,
                load_name_counts=False,
                preprocess=False,
            )

        dataset = ANDData(authors={}, papers={}, name="", mode="inference", load_name_counts=False, preprocess=False)
        assert dataset.paper_to_cluster_id is None

        dataset = ANDData(authors={}, papers={}, name="", mode="inference", load_name_counts=False, preprocess=False)
        assert dataset.pair_sampling_block
        assert not dataset.pair_sampling_balanced_classes
        assert not dataset.pair_sampling_balanced_homonym_synonym
        assert dataset.all_test_pairs_flag
        assert dataset.block_type == "s2"

        with pytest.raises(Exception):
            dataset = ANDData(
                authors={}, papers={}, clusters={}, name="", mode="dummy", load_name_counts=False, preprocess=False
            )

    def test_construct_cluster_to_signatures(self):
        cluster_to_signatures = self.dummy_dataset.construct_cluster_to_papers({"a": ["0", "1"], "b": ["3", "4"]})
        expected_cluster_to_signatures = {"1": ["0", "1"], "3": ["3", "4"]}
        assert cluster_to_signatures == expected_cluster_to_signatures