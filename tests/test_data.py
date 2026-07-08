import unittest
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import s2and.data as data_module
from s2and.data import ANDData
from s2and.rust_lifecycle import PYTHON_ONLY_POLICY


def test_maybe_load_list_empty_file_returns_empty_list(tmp_path):
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")

    assert ANDData.maybe_load_list(str(empty_path)) == []


def test_preprocess_signatures_drops_empty_normalized_affiliations() -> None:
    dataset = ANDData(
        signatures={
            "s1": {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "a lovelace",
                    "first": "Ada",
                    "middle": "",
                    "last": "Lovelace",
                    "suffix": None,
                    "email": None,
                    "affiliations": [",", "\u00a0", "Analytical Engine Lab"],
                },
            }
        },
        papers={
            "1": {
                "paper_id": 1,
                "title": "Notes",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 1843,
                "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
                "references": [],
            }
        },
        name="empty_normalized_affiliations",
        mode="inference",
        load_name_counts=False,
        preprocess=True,
        n_jobs=1,
    )

    assert dataset.signatures["s1"].author_info_affiliations == ["analytical engine lab"]
    assert "" not in dataset.signatures["s1"].author_info_affiliations


def test_anddata_passes_arrow_featurization_to_rust_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        data_module,
        "build_runtime_context",
        lambda _operation: SimpleNamespace(
            requested_backend="auto",
            resolved_backend="rust",
            use_rust=False,
            run_id="test-run",
            source="default",
        ),
    )

    def _capture_lifecycle_policy(**kwargs: Any):
        captured.update(kwargs)
        return PYTHON_ONLY_POLICY

    monkeypatch.setattr(data_module, "build_rust_lifecycle_policy", _capture_lifecycle_policy)

    ANDData(
        signatures={
            "s1": {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "a lovelace",
                    "first": "Ada",
                    "middle": "",
                    "last": "Lovelace",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                },
            }
        },
        papers={
            "1": {
                "paper_id": 1,
                "title": "Notes",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 1843,
                "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
                "references": [],
            }
        },
        name="rust_lifecycle_capability",
        mode="inference",
        load_name_counts=False,
        preprocess=False,
        name_tuples=set(),
        rust_arrow_featurization=True,
    )

    assert captured["backend"] == "rust"
    assert captured["arrow_featurization"] is True


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
        self.qian_dataset.pair_sampling_mode = "within_block_random"
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
        self.qian_dataset.pair_sampling_mode = "within_block_balanced_classes"
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
        self.qian_dataset.pair_sampling_mode = "within_block_balanced_homonym_synonym"
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert sum([int(pair[2]) for pair in train_pairs]) == 500
        assert len(train_pairs) == 1000 and len(val_pairs) == 500 and len(test_pairs) == 500
        assert (
            train_pairs[0] == ("4389", "4493", 0)
            and val_pairs[0] == ("185", "197", 0)
            and test_pairs[0] == ("2431", "2437", 1)
        )

        # Test adding the all test pairs flag to the test above
        self.qian_dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert len(train_pairs) == 1000
        assert len(val_pairs) == 500
        assert len(test_pairs) == 7244

    def test_split_pairs_global_balanced_classes_uses_split_signatures(self):
        self.qian_dataset.pair_sampling_mode = "global_balanced_classes"
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

        expected_train_pairs = self.qian_dataset.pair_sampling(
            self.qian_dataset.train_pairs_size,
            [signature for signatures in train_block_dict.values() for signature in signatures],
            train_block_dict,
        )
        expected_val_pairs = self.qian_dataset.pair_sampling(
            self.qian_dataset.val_pairs_size,
            [signature for signatures in val_block_dict.values() for signature in signatures],
            val_block_dict,
        )
        expected_test_pairs = self.qian_dataset.pair_sampling(
            self.qian_dataset.test_pairs_size,
            [signature for signatures in test_block_dict.values() for signature in signatures],
            test_block_dict,
        )

        assert train_pairs == expected_train_pairs
        assert val_pairs == expected_val_pairs
        assert test_pairs == expected_test_pairs
        assert train_pairs

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
        with pytest.raises(ValueError):
            self.dummy_dataset.get_blocks()
        self.dummy_dataset.block_type = "s2"

        assert original_blocks == expected_original_blocks
        assert original_blocks_2 == expected_original_blocks
        assert s2_blocks == expected_s2_blocks
        assert s2_blocks_2 == expected_s2_blocks

    def test_initialization(self):
        with pytest.raises(ValueError):
            dataset = ANDData(
                signatures={},
                papers={},
                clusters={},
                name="",
                mode="train",
                unit_of_data_split="blocks",
                pair_sampling_mode="global_balanced_classes",
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(ValueError):
            dataset = ANDData(
                signatures={},
                papers={},
                name="",
                mode="train",
                clusters={},
                train_pairs=cast(Any, []),
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(ValueError):
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

        with pytest.raises(ValueError):
            dataset = ANDData(
                signatures={},
                papers={},
                name="",
                mode="train",
                train_blocks=[],
                train_pairs=cast(Any, []),
                load_name_counts=False,
                preprocess=False,
            )

        with pytest.raises(ValueError):
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

        with pytest.raises(ValueError):
            dataset = ANDData(
                signatures={}, papers={}, clusters={}, name="", mode="dummy", load_name_counts=False, preprocess=False
            )

    def test_construct_cluster_to_signatures(self):
        cluster_to_signatures = self.dummy_dataset.construct_cluster_to_signatures({"a": ["0", "1"], "b": ["3", "4"]})
        expected_cluster_to_signatures = {"1": ["0", "1"], "3": ["3", "4"]}
        assert cluster_to_signatures == expected_cluster_to_signatures

    def test_multiprocessing_preprocessing_consistency(self):
        """Test that multiprocessing preprocessing produces identical results to single-threaded"""
        # Create datasets with same data but different n_jobs settings
        dataset_single = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_single",
            load_name_counts=False,
            preprocess=True,
            n_jobs=1,
        )

        dataset_multi = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_multi",
            load_name_counts=False,
            preprocess=True,
            n_jobs=2,
        )

        # Verify that at least one paper was processed (has title normalization)
        assert len(dataset_single.papers) > 0 and len(dataset_multi.papers) > 0

        # Compare that papers are preprocessed identically
        for paper_id in dataset_single.papers:
            paper_single = dataset_single.papers[paper_id]
            paper_multi = dataset_multi.papers[paper_id]

            # Check that key preprocessed fields are identical
            assert paper_single.title == paper_multi.title, f"Title mismatch for paper {paper_id}"
            assert (
                paper_single.predicted_language == paper_multi.predicted_language
            ), f"Language mismatch for paper {paper_id}"
            assert paper_single.is_english == paper_multi.is_english, f"is_english mismatch for paper {paper_id}"
            assert paper_single.is_reliable == paper_multi.is_reliable, f"is_reliable mismatch for paper {paper_id}"

            # Check ngrams are identical
            if paper_single.title_ngrams_words is not None and paper_multi.title_ngrams_words is not None:
                assert (
                    paper_single.title_ngrams_words == paper_multi.title_ngrams_words
                ), f"Title ngrams mismatch for paper {paper_id}"


def test_compute_signature_name_counts_uses_single_character_initial():
    load_name_counts = {
        "first_dict": {},
        "last_dict": {"smith": 11},
        "first_last_dict": {},
        "last_first_initial_dict": {"smith m": 17},
    }
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="dummy_name_counts_initial",
        mode="inference",
        load_name_counts=load_name_counts,
        preprocess=False,
    )
    signature = next(iter(dataset.signatures.values()))._replace(
        author_info_first="Michael",
        author_info_middle="",
        author_info_last="Smith",
        author_info_first_normalized_without_apostrophe="michael",
        author_info_middle_normalized_without_apostrophe="",
        author_info_last_normalized="smith",
    )
    counts = dataset._compute_signature_name_counts(
        signature,
        first_raw="Michael",
        middle_raw="",
        first_without_apostrophe="michael",
        last_normalized="smith",
    )
    assert counts.last_first_initial == 17


def test_empty_altered_cluster_signatures_file_loads_as_empty_list(tmp_path):
    altered_path = tmp_path / "altered_cluster_signatures.txt"
    altered_path.write_text("", encoding="utf-8")

    dataset = ANDData(
        signatures={},
        papers={},
        name="empty_altered",
        mode="inference",
        cluster_seeds={"1": {"2": "require"}},
        altered_cluster_signatures=str(altered_path),
        load_name_counts=False,
        preprocess=False,
    )

    assert dataset.altered_cluster_signatures == []


def test_pair_sampling_invalid_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown pair_sampling_mode"):
        ANDData(
            signatures={},
            papers={},
            clusters={},
            name="invalid_pair_sampling_mode",
            mode="train",
            pair_sampling_mode="global_unbalanced",  # type: ignore[arg-type]
            load_name_counts=False,
            preprocess=False,
        )


def test_pair_sampling_rejects_mixed_canonical_and_legacy_flags():
    with pytest.raises(ValueError, match="Set either pair_sampling_mode or legacy"):
        ANDData(
            signatures={},
            papers={},
            clusters={},
            name="mixed_pair_sampling",
            mode="train",
            pair_sampling_mode="within_block_random",
            pair_sampling_block=True,
            load_name_counts=False,
            preprocess=False,
        )


def test_inference_dataset_with_clusters_initializes_signature_to_cluster_id():
    dataset = ANDData(
        signatures={},
        papers={},
        clusters={},
        name="inference_with_clusters_signature_mapping",
        mode="inference",
        load_name_counts=False,
        preprocess=False,
    )

    assert hasattr(dataset, "signature_to_cluster_id")
    assert dataset.signature_to_cluster_id is None


def test_fixed_pairs_does_not_mutate_source_dataframes():
    train_pairs_df = pd.DataFrame(
        [("s1", "s2", "YES"), ("s3", "s4", "NO")],
        columns=["signature_id_1", "signature_id_2", "label"],
    )
    val_pairs_df = pd.DataFrame(
        [("s5", "s6", "1"), ("s7", "s8", "0")],
        columns=["signature_id_1", "signature_id_2", "label"],
    )
    test_pairs_df = pd.DataFrame(
        [("s9", "s10", 1), ("s11", "s12", 0)],
        columns=["signature_id_1", "signature_id_2", "label"],
    )
    dataset = ANDData(
        signatures={},
        papers={},
        name="fixed_pairs_copy_safety",
        mode="train",
        clusters=None,
        train_pairs=train_pairs_df,
        val_pairs=val_pairs_df,
        test_pairs=test_pairs_df,
        load_name_counts=False,
        preprocess=False,
    )

    assert dataset.train_pairs is not None
    assert dataset.val_pairs is not None
    assert dataset.test_pairs is not None
    train_before = dataset.train_pairs.copy(deep=True)
    val_before = dataset.val_pairs.copy(deep=True)
    test_before = dataset.test_pairs.copy(deep=True)

    train_pairs, val_pairs, test_pairs = dataset.fixed_pairs()

    assert dataset.train_pairs.equals(train_before)
    assert dataset.val_pairs.equals(val_before)
    assert dataset.test_pairs.equals(test_before)

    all_labels = [int(pair[2]) for pair in train_pairs + val_pairs + test_pairs]
    assert set(all_labels).issubset({0, 1})
