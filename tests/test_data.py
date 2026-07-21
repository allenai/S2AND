import json
import unittest
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import s2and.data as data_module
from s2and.data import ANDData
from s2and.name_tuple_artifact import build_name_tuple_artifact_metadata
from tests.helpers import tiny_name_counts_provenance


def test_maybe_load_list_empty_file_returns_empty_list(tmp_path):
    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("", encoding="utf-8")

    assert ANDData.maybe_load_list(str(empty_path)) == []


def test_maybe_load_json_reads_utf8_text(tmp_path):
    json_path = tmp_path / "unicode.json"
    json_path.write_text(json.dumps({"name": "José"}, ensure_ascii=False), encoding="utf-8")

    assert ANDData.maybe_load_json(str(json_path)) == {"name": "José"}


def test_maybe_load_list_reads_utf8_text(tmp_path):
    list_path = tmp_path / "unicode.txt"
    list_path.write_text("José\nZoë", encoding="utf-8")

    assert ANDData.maybe_load_list(str(list_path)) == ["José", "Zoë"]


@pytest.mark.parametrize(
    "payload",
    [
        {1: np.asarray([1.0]), "1": np.asarray([2.0])},
        (np.asarray([[1.0], [2.0]]), [1, "1"]),
    ],
)
def test_maybe_load_specter_rejects_keys_that_collide_as_strings(payload):
    with pytest.raises(ValueError, match="collide after string normalization"):
        ANDData.maybe_load_specter(payload)


def test_maybe_load_specter_normalizes_tuple_keys_to_strings():
    loaded = ANDData.maybe_load_specter((np.asarray([[1.0, 2.0], [3.0, 4.0]]), [1, 2]))

    assert loaded is not None
    assert set(loaded) == {"1", "2"}
    np.testing.assert_array_equal(loaded["1"], np.asarray([1.0, 2.0]))


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
        name_counts_index=None,
        preprocess=True,
        n_jobs=1,
    )

    assert dataset.signatures["s1"].author_info_affiliations == ["analytical engine lab"]
    assert "" not in dataset.signatures["s1"].author_info_affiliations


def test_name_tuples_none_uses_canonical_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        data_module,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(pairs=frozenset({("bill", "william")})),
    )
    dataset = ANDData(
        signatures={},
        papers={},
        name="canonical_name_tuple_default",
        mode="inference",
        name_counts_index=None,
        preprocess=False,
        name_tuples=None,
    )

    assert dataset.name_tuples == {("bill", "william")}


def test_name_tuples_rejects_string_sentinel() -> None:
    with pytest.raises(TypeError, match="set/frozenset"):
        ANDData(
            signatures={},
            papers={},
            name="invalid_name_tuple_sentinel",
            mode="inference",
            name_counts_index=None,
            preprocess=False,
            name_tuples="filtered",  # type: ignore[arg-type]
        )


def test_custom_name_tuples_are_stored_as_unordered_pairs() -> None:
    dataset = ANDData(
        signatures={},
        papers={},
        name="canonical_custom_name_tuples",
        mode="inference",
        name_counts_index=None,
        preprocess=False,
        name_tuples={("william", "bill")},
    )

    assert dataset.name_tuples == {("bill", "william")}


def test_name_tuple_loader_rejects_invalid_rows_with_valid_binding_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(data_module, "_PACKAGE_DATA_DIR", str(tmp_path))
    data_bytes = b"alice,bob,carol\n"
    (tmp_path / "invalid.txt").write_bytes(data_bytes)
    metadata = build_name_tuple_artifact_metadata(
        source_filename="source.txt",
        source_bytes=b"source\n",
        data_filename="invalid.txt",
        data_bytes=data_bytes,
        pair_count=1,
        generated_at="2026-07-10T00:00:00+00:00",
        input_pair_count=1,
        dropped_identity=0,
        dropped_prefix_compatible=0,
        dropped_empty=0,
        dropped_duplicate_canonical=0,
    )
    (tmp_path / "invalid.txt.meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid.txt:1"):
        data_module._load_name_tuples_from_file("invalid.txt")


def test_signature_full_name_uses_only_canonical_fields() -> None:
    dataset = ANDData(
        signatures={
            "s1": {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "d smith",
                    "first": "Dr.",
                    "middle": "",
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                },
            }
        },
        papers={
            "1": {
                "paper_id": 1,
                "title": "Part 1",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2020,
                "authors": [{"position": 0, "author_name": "Dr. Smith"}],
                "references": [],
            }
        },
        name="canonical_full_name",
        mode="inference",
        name_counts_index=None,
        preprocess=True,
        n_jobs=1,
        name_tuples=set(),
    )

    signature = dataset.signatures["s1"]
    assert signature.author_info_first_normalized_without_apostrophe == ""
    assert signature.author_info_full_name == "smith"
    assert dataset.papers["1"].title == "part 1"
    assert dataset.papers["1"].title_ngrams_words["1"] == 1


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.qian_dataset = ANDData(
            "tests/qian/signatures.json",
            # "tests/qian/papers.json",
            {},
            clusters="tests/qian/clusters.json",
            name="qian",
            name_counts_index=None,
            preprocess=False,
        )
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            # "tests/dummy/papers.json",
            {},
            clusters="tests/dummy/clusters.json",
            name="dummy",
            name_counts_index=None,
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

        # Test balanced pos/neg and homonym/synonym sampling within blocks
        self.qian_dataset.pair_sampling_mode = "within_block_balanced_homonym_synonym"
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert sum([int(pair[2]) for pair in train_pairs]) == 500
        assert len(train_pairs) == 1000 and len(val_pairs) == 500 and len(test_pairs) == 500

        # Test adding the all test pairs flag to the test above
        self.qian_dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = self.qian_dataset.split_pairs(
            train_block_dict, val_block_dict, test_block_dict
        )
        assert len(train_pairs) == 1000
        assert len(val_pairs) == 500
        assert len(test_pairs) == 7244

    def test_split_cluster_signatures_accepts_float_ratios_close_to_one(self):
        self.qian_dataset.train_ratio = 0.7
        self.qian_dataset.val_ratio = 0.2
        self.qian_dataset.test_ratio = 0.1

        train_blocks, val_blocks, test_blocks = self.qian_dataset.split_cluster_signatures()

        self.assertEqual(
            set(train_blocks) | set(val_blocks) | set(test_blocks),
            set(self.qian_dataset.get_blocks()),
        )

    def test_split_cluster_signatures_rejects_out_of_range_ratios(self):
        self.qian_dataset.train_ratio = 1.1
        self.qian_dataset.val_ratio = -0.1
        self.qian_dataset.test_ratio = 0.0

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.qian_dataset.split_cluster_signatures()

    def test_fixed_block_split_allows_empty_partitions(self):
        block_ids = list(self.qian_dataset.get_blocks())
        self.qian_dataset.train_blocks = block_ids
        self.qian_dataset.val_blocks = []
        self.qian_dataset.test_blocks = []

        train_blocks, val_blocks, test_blocks = self.qian_dataset.split_cluster_signatures_fixed()

        self.assertEqual(set(train_blocks), set(block_ids))
        self.assertEqual(val_blocks, {})
        self.assertEqual(test_blocks, {})

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
                name_counts_index=None,
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
                name_counts_index=None,
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
                name_counts_index=None,
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
                name_counts_index=None,
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
                name_counts_index=None,
                preprocess=False,
            )

        dataset = ANDData(signatures={}, papers={}, name="", mode="inference", name_counts_index=None, preprocess=False)
        assert dataset.signature_to_cluster_id is None

        dataset = ANDData(signatures={}, papers={}, name="", mode="inference", name_counts_index=None, preprocess=False)
        assert dataset.pair_sampling_mode == "within_block_random"
        assert dataset.all_test_pairs_flag
        assert dataset.block_type == "s2"

        with pytest.raises(ValueError):
            dataset = ANDData(
                signatures={}, papers={}, clusters={}, name="", mode="dummy", name_counts_index=None, preprocess=False
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
            name_counts_index=None,
            preprocess=True,
            n_jobs=1,
        )

        dataset_multi = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy_multi",
            name_counts_index=None,
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


def test_preprocessing_name_counts_use_single_character_initial(tmp_path):
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

    index_path, _metrics = write_name_counts_index(
        tmp_path,
        ({}, {"sattar": 11}, {}, {"sattar a": 17, "sattar abdul": 41}),
        tiny_name_counts_provenance(),
    )
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="dummy_name_counts_initial",
        mode="inference",
        name_counts_index=index_path,
        preprocess=False,
    )
    signature_id = next(iter(dataset.signatures))
    dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
        author_info_first="Abdul",
        author_info_middle="",
        author_info_last="Sattar",
    )
    dataset.preprocess = True
    dataset.preprocess_signatures()

    counts = dataset.signatures[signature_id].author_info_name_counts
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
        name_counts_index=None,
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
            name_counts_index=None,
            preprocess=False,
        )


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
        name_counts_index=None,
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


@pytest.mark.parametrize("invalid_split", ["train", "val", "test"])
def test_fixed_pairs_rejects_unknown_labels(invalid_split):
    pair_frames = {
        split_name: pd.DataFrame(
            [(f"{split_name}1", f"{split_name}2", "YES")],
            columns=["signature_id_1", "signature_id_2", "label"],
        )
        for split_name in ("train", "val", "test")
    }
    pair_frames[invalid_split].loc[0, "label"] = "MAYBE"
    dataset = ANDData(
        signatures={},
        papers={},
        name="fixed_pairs_invalid_label",
        mode="train",
        clusters=None,
        train_pairs=pair_frames["train"],
        val_pairs=pair_frames["val"],
        test_pairs=pair_frames["test"],
        name_counts_index=None,
        preprocess=False,
    )

    with pytest.raises(ValueError, match=rf"Unknown fixed-pair labels.*{invalid_split}.*MAYBE"):
        dataset.fixed_pairs()
