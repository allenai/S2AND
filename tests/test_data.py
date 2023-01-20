import unittest
import pytest

from s2and.data import ANDData


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.qian_dataset = ANDData(
            "tests/qian/signatures.json",
            # "tests/qian/papers.json",
            {},
            clusters="tests/qian/clusters.json",
            name="qian",
            load_name_counts=False,
            preprocess=False,
        )
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            # "tests/dummy/papers.json",
            {},
            clusters="tests/dummy/clusters.json",
            name="dummy",
            load_name_counts=False,
            preprocess=False,
        )

    def test_split_pairs_within_blocks(self):
        # Test random sampling within blocks
        self.qian_dataset.pair_sampling_block = True
        self.qian_dataset.pair_sampling_balanced_classes = False
        self.qian_dataset.pair_sampling_balanced_homonym_synonym = False
        self.qian_dataset.train_pairs_size = 1000
        self.qian_dataset.val_pairs_size = 500
        self.qian_dataset.test_pairs_size = 500
        self.qian_dataset.random_seed = 1111
        (
            train_block_dict,
            val_block_dict,
            test_block_dict,
        ) = self.qian_dataset.split_cluster_signatures()
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )

        assert len(train_pairs) == 1000 and len(val_pairs) == 500 and len(test_pairs) == 500
        assert (
            train_pairs[0] == ("5259", "5270", 1)
            and val_pairs[0] == ("3830", "3847", 1)
            and test_pairs[0] == ("1050", "1063", 1)
        )

        # Test balanced pos/neg sampling within blocks
        self.qian_dataset.pair_sampling_block = True
        self.qian_dataset.pair_sampling_balanced_classes = True
        self.qian_dataset.pair_sampling_balanced_homonym_synonym = False
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert sum([int(pair[2]) for pair in train_pairs]) == 500
        assert len(train_pairs) == 1000 and len(val_pairs) == 500 and len(test_pairs) == 500
        assert (
            train_pairs[0] == ("5694", "5702", 1)
            and val_pairs[0] == ("781", "787", 1)
            and test_pairs[0] == ("2428", "2581", 0)
        )

        # Test balanced pos/neg and homonym/synonym sampling within blocks
        self.qian_dataset.pair_sampling_block = True
        self.qian_dataset.pair_sampling_balanced_classes = True
        self.qian_dataset.pair_sampling_balanced_homonym_synonym = True
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert sum([int(pair[2]) for pair in train_pairs]) == 500
        assert len(train_pairs) == 1000 and len(val_pairs) == 429 and len(test_pairs) == 376
        assert (
            train_pairs[0] == ("4389", "4493", 0)
            and val_pairs[0] == ("621", "636", 0)
            and test_pairs[0] == ("2550", "2622", 0)
        )

        # Test adding the all test pairs flag to the test above
        self.qian_dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert len(train_pairs) == 1000, len(val_pairs) == 429 and len(test_pairs) == 7244

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
                signatures={},
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
                signatures={},
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
                signatures={},
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
                signatures={},
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
                signatures={},
                papers={},
                name="",
                mode="train",
                train_blocks=[],
                clusters=None,
                load_name_counts=False,
                preprocess=False,
            )

        dataset = ANDData(signatures={}, papers={}, name="", mode="inference", load_name_counts=False, preprocess=False)
        assert dataset.signature_to_cluster_id is None

        dataset = ANDData(signatures={}, papers={}, name="", mode="inference", load_name_counts=False, preprocess=False)
        assert dataset.pair_sampling_block
        assert not dataset.pair_sampling_balanced_classes
        assert not dataset.pair_sampling_balanced_homonym_synonym
        assert dataset.all_test_pairs_flag
        assert dataset.block_type == "s2"

        with pytest.raises(Exception):
            dataset = ANDData(
                signatures={}, papers={}, clusters={}, name="", mode="dummy", load_name_counts=False, preprocess=False
            )

    def test_construct_cluster_to_signatures(self):
        cluster_to_signatures = self.dummy_dataset.construct_cluster_to_signatures({"a": ["0", "1"], "b": ["3", "4"]})
        expected_cluster_to_signatures = {"1": ["0", "1"], "3": ["3", "4"]}
        assert cluster_to_signatures == expected_cluster_to_signatures
