import json
import unittest
from functools import cached_property, partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import s2and.data as data_module
from s2and.data import ANDData
from tests.helpers import build_dummy_dataset, tiny_name_counts_index


def test_time_split_groups_placeholder_years_with_missing_dates() -> None:
    dataset = build_dummy_dataset("time_split")
    dataset.unit_of_data_split = "time"
    ids = list(dataset.signatures)
    dataset.train_ratio = 1 / len(ids)
    dataset.val_ratio = 1 / len(ids)
    dataset.test_ratio = 1 - 2 / len(ids)
    years = [1800, 1799, None, *([2010] * (len(ids) - 3))]
    for signature, year in zip(ids, years, strict=True):
        paper_id = str(dataset.signatures[signature].paper_id)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=year)

    train, val, test = dataset.split_cluster_signatures()

    assert [sid for block in train.values() for sid in block] == [ids[0]]
    assert [sid for block in val.values() for sid in block] == [ids[2]]
    assert {sid for block in test.values() for sid in block} == set(ids) - {ids[0], ids[2]}


@pytest.mark.parametrize(
    "seeds, expected",
    [
        (
            {"0": {"1": "require"}, "2": {"3": "require"}, "1": {"2": "require"}},
            {"0": 0, "1": 0, "2": 0, "3": 0},
        ),
        (
            {"0": {"1": "require", "2": "require"}, "3": {"4": "require"}},
            {"0": 0, "1": 0, "2": 0, "3": 1, "4": 1},
        ),
        ({"0": {"1": "require"}, "1": {"0": "require"}}, {"0": 0, "1": 0}),
        ({"0": {"0": "require"}}, {"0": 0}),
        ({"0": {"1": "disallow"}}, {}),
    ],
    ids=["bridge", "disjoint", "cycle", "self", "disallow_only"],
)
def test_required_seed_components_preserve_transitive_constraints(seeds, expected):
    dataset = ANDData(
        signatures="tests/dummy/signatures.json",
        papers="tests/dummy/papers.json",
        name="seed_components",
        mode="inference",
        cluster_seeds=seeds,
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
    )

    assert dataset.cluster_seeds_require == expected
    assert dataset.max_seed_cluster_id == len(set(expected.values()))
    for left, left_component in expected.items():
        for right, right_component in expected.items():
            constraint = "require" if left_component == right_component else "disallow"
            assert dataset.get_constraint(left, right) == data_module.CLUSTER_SEEDS_LOOKUP[constraint]


def test_unpreprocessed_signatures_skip_coauthor_sets_that_subblocking_recomputes():
    """Coauthor sets are quadratic on huge author lists, so only preprocessing builds them."""

    from s2and.subblocking import _signature_coauthor_blocks_for_specter

    def load(preprocess: bool) -> ANDData:
        return ANDData(
            signatures="tests/dummy/signatures.json",
            papers="tests/dummy/papers.json",
            name="coauthors",
            mode="inference",
            name_counts_index=None,
            name_tuples=set(),
            preprocess=preprocess,
        )

    raw, prepared = load(False), load(True)
    assert any(signature.author_info_coauthor_blocks for signature in prepared.signatures.values())
    for signature_id, signature in raw.signatures.items():
        assert signature.author_info_coauthors is None
        assert signature.author_info_coauthor_blocks is None
        # Paper author names are normalized either way, so subblocking evidence is unchanged.
        expected = prepared.signatures[signature_id].author_info_coauthor_blocks
        assert set(_signature_coauthor_blocks_for_specter(signature, raw)) == expected - {""}


def test_explicit_seed_disallow_survives_connected_require_groups():
    dataset = ANDData(
        signatures="tests/dummy/signatures.json",
        papers="tests/dummy/papers.json",
        name="seed_explicit_disallow",
        mode="inference",
        cluster_seeds={"0": {"1": "require", "2": "disallow"}, "1": {"2": "require"}},
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
    )

    assert dataset.cluster_seeds_require == {"0": 0, "1": 0, "2": 0}
    assert dataset.cluster_seeds_disallow == {("0", "2")}
    for ignore_seeds in (False, True):
        for left, right in (("0", "2"), ("2", "0")):
            assert (
                dataset.get_constraint(left, right, incremental_dont_use_cluster_seeds=ignore_seeds)
                == data_module.CLUSTER_SEEDS_LOOKUP["disallow"]
            )


def test_split_ratios_reject_default_isclose_near_miss() -> None:
    with pytest.raises(ValueError, match="must add to 1"):
        data_module._validate_split_ratios(5e-10, 0.5, 0.5000000004)


def test_anddata_uses_only_s2_blocks_and_ignores_legacy_given_block() -> None:
    signatures = {}
    papers = {}
    for index, legacy_block in enumerate(("legacy one", "legacy two"), start=1):
        signature_id = f"s{index}"
        paper_id = f"p{index}"
        signatures[signature_id] = {
            "signature_id": signature_id,
            "paper_id": paper_id,
            "author_info": {
                "position": 0,
                "block": "a canonical",
                "given_block": legacy_block,
                "first": "Ada",
                "middle": None,
                "last": "Canonical",
                "suffix": None,
                "affiliations": [],
                "email": None,
            },
        }
        papers[paper_id] = {
            "paper_id": paper_id,
            "title": f"Paper {index}",
            "abstract": None,
            "authors": [{"position": 0, "author_name": "Ada Canonical"}],
            "venue": None,
            "journal_name": None,
            "year": 2026,
        }

    dataset = ANDData(
        signatures=signatures,
        papers=papers,
        name="s2_only_blocks",
        mode="inference",
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
    )

    assert dataset.get_blocks() == {"a canonical": ["s1", "s2"]}


def test_split_ratios_do_not_create_a_partition_for_tolerated_roundoff() -> None:
    train, val, test = data_module._split_train_val_test(
        ["a", "b"],
        1.0 - 5e-13,
        0.0,
        0.0,
        random_seed=0,
    )

    assert train == ["a", "b"]
    assert val == []
    assert test == []


def test_input_loaders_read_utf8_text(tmp_path):
    json_path = tmp_path / "unicode.json"
    json_path.write_text(json.dumps({"name": "José"}, ensure_ascii=False), encoding="utf-8")

    assert ANDData.maybe_load_json(str(json_path)) == {"name": "José"}

    list_path = tmp_path / "unicode.txt"
    list_path.write_text("José\nZoë", encoding="utf-8")

    assert ANDData.maybe_load_list(str(list_path)) == ["José", "Zoë"]


def test_specter_normalizes_keys_and_rejects_collisions():
    loaded = ANDData.maybe_load_specter((np.asarray([[1.0, 2.0], [3.0, 4.0]]), [1, 2]))

    assert loaded is not None
    assert set(loaded) == {"1", "2"}
    np.testing.assert_array_equal(loaded["1"], np.asarray([1.0, 2.0]))

    cases = (
        ("mapping", {1: np.asarray([1.0]), "1": np.asarray([2.0])}),
        ("tuple", (np.asarray([[1.0], [2.0]]), [1, "1"])),
    )
    for _case_id, payload in cases:
        with pytest.raises(ValueError, match="collide after string normalization"):
            ANDData.maybe_load_specter(payload)


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
    assert signature.author_info_first_normalized == ""
    assert signature.author_info_full_name == "smith"
    assert dataset.papers["1"].title == "part 1"
    assert dataset.papers["1"].title_ngrams_words["1"] == 1


def test_split_pairs_global_balanced_classes_routes_split_signatures() -> None:
    dataset = ANDData.__new__(ANDData)
    dataset.pair_sampling_mode = "global_balanced_classes"
    dataset.train_pairs_size, dataset.val_pairs_size, dataset.test_pairs_size = 11, 12, 13
    dataset.all_test_pairs_flag = True
    calls: list[tuple[int, list[str], dict[str, list[str]], bool]] = []

    def record_call(
        sample_size: int,
        signature_ids: list[str],
        blocks: dict[str, list[str]],
        all_pairs: bool = False,
    ) -> list[tuple[str, str, int]]:
        calls.append((sample_size, signature_ids, blocks, all_pairs))
        return [(signature_ids[0], signature_ids[-1], sample_size)]

    dataset.pair_sampling = cast(Any, record_call)
    train = {"train": ["t1", "t2"]}
    val = {"val": ["v1", "v2"]}
    test = {"test": ["x1", "x2"]}

    outputs = dataset.split_pairs(train, val, test)

    assert calls == [
        (11, ["t1", "t2"], train, False),
        (12, ["v1", "v2"], val, False),
        (13, ["x1", "x2"], test, True),
    ]
    assert outputs == ([("t1", "t2", 11)], [("v1", "v2", 12)], [("x1", "x2", 13)])


class TestData(unittest.TestCase):
    @cached_property
    def qian_dataset(self):
        return ANDData(
            "tests/qian/signatures.json",
            {},
            clusters="tests/qian/clusters.json",
            name="qian",
            name_counts_index=None,
            preprocess=False,
        )

    @cached_property
    def dummy_dataset(self):
        return ANDData(
            "tests/dummy/signatures.json",
            {},
            clusters="tests/dummy/clusters.json",
            name="dummy",
            name_counts_index=None,
            preprocess=False,
        )

    def test_split_pairs_within_blocks(self):
        dataset = self.dummy_dataset
        dataset.train_pairs_size = 2
        dataset.val_pairs_size = 2
        dataset.test_pairs_size = 2
        dataset.random_seed = 1111
        dataset.signature_to_cluster_id = {
            "0": "a",
            "1": "a",
            "2": "b",
            "3": "c",
            "4": "c",
            "5": "d",
            "6": "e",
            "7": "e",
            "8": "f",
        }
        dataset.signatures["0"] = dataset.signatures["0"]._replace(
            author_info_first=dataset.signatures["1"].author_info_first
        )
        train = {"train": ["0", "1", "2"]}
        validation = {"val": ["3", "4", "5"]}
        test = {"test": ["6", "7", "8"]}

        dataset.pair_sampling_mode = "within_block_random"
        sampled = dataset.split_pairs(train, validation, test)
        assert all(len(pairs) == 2 for pairs in sampled)

        dataset.pair_sampling_mode = "within_block_balanced_classes"
        balanced = dataset.split_pairs(train, validation, test)
        assert {pair[2] for pair in balanced[0]} == {0, 1}
        assert all(len(pairs) == 2 for pairs in balanced)

        dataset.pair_sampling_mode = "within_block_balanced_homonym_synonym"
        dataset.all_test_pairs_flag = True
        train_pairs, val_pairs, test_pairs = dataset.split_pairs(train, validation, test)
        assert {pair[2] for pair in train_pairs} == {0, 1}
        assert len(val_pairs) == 2
        assert len(test_pairs) == 3
        dataset.train_pairs_size = 3
        dataset.signatures["1"] = dataset.signatures["1"]._replace(author_info_first="Abdul Rahman")
        dataset.signatures["2"] = dataset.signatures["2"]._replace(author_info_first="Abdul Hadi")
        train_pairs, _, _ = dataset.split_pairs(train, validation, test)
        assert len(train_pairs) == 3

    def test_split_cluster_signatures_rejects_out_of_range_ratios(self):
        self.qian_dataset.train_ratio = 1.1
        self.qian_dataset.val_ratio = -0.1
        self.qian_dataset.test_ratio = 0.0

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.qian_dataset.split_cluster_signatures()

    def test_split_cluster_signatures_preserves_ratio_partitions(self):
        cases = [
            ((0.7, 0.2, 0.1), set()),
            ((1.0, 0.0, 0.0), {"val", "test"}),
            ((0.8, 0.2, 0.0), {"test"}),
            ((0.8, 0.0, 0.2), {"val"}),
        ]
        for unit_of_data_split in ("blocks", "signatures"):
            for ratios, empty_partitions in cases:
                with self.subTest(unit_of_data_split=unit_of_data_split, ratios=ratios):
                    self.qian_dataset.unit_of_data_split = unit_of_data_split
                    (
                        self.qian_dataset.train_ratio,
                        self.qian_dataset.val_ratio,
                        self.qian_dataset.test_ratio,
                    ) = ratios

                    train_blocks, val_blocks, test_blocks = self.qian_dataset.split_cluster_signatures()
                    self.assertEqual(
                        set(train_blocks) | set(val_blocks) | set(test_blocks),
                        set(self.qian_dataset.get_blocks()),
                    )

                    split_signatures = {
                        "train": {signature for block in train_blocks.values() for signature in block},
                        "val": {signature for block in val_blocks.values() for signature in block},
                        "test": {signature for block in test_blocks.values() for signature in block},
                    }
                    expected_signatures = set(self.qian_dataset.signatures)
                    self.assertEqual(set().union(*split_signatures.values()), expected_signatures)
                    self.assertFalse(split_signatures["train"] & split_signatures["val"])
                    self.assertFalse(split_signatures["train"] & split_signatures["test"])
                    self.assertFalse(split_signatures["val"] & split_signatures["test"])
                    for partition in empty_partitions:
                        self.assertEqual(split_signatures[partition], set())
                    for partition in {"train", "val", "test"} - empty_partitions:
                        self.assertTrue(split_signatures[partition])

    def test_fixed_block_split_allows_empty_partitions(self):
        block_ids = list(self.qian_dataset.get_blocks())
        self.qian_dataset.train_blocks = block_ids
        self.qian_dataset.val_blocks = []
        self.qian_dataset.test_blocks = []

        train_blocks, val_blocks, test_blocks = self.qian_dataset.split_cluster_signatures_fixed()

        self.assertEqual(set(train_blocks), set(block_ids))
        self.assertEqual(val_blocks, {})
        self.assertEqual(test_blocks, {})

    def test_dummy_blocks_and_cluster_membership(self):
        expected_blocks = {
            "a sattary": ["0", "1", "2"],
            "a konovalov": ["3", "4", "5", "6", "7", "8"],
        }

        assert self.dummy_dataset.get_blocks() == expected_blocks

        cluster_to_signatures = self.dummy_dataset.construct_cluster_to_signatures({"a": ["0", "1"], "b": ["3", "4"]})
        expected_cluster_to_signatures = {"1": ["0", "1"], "3": ["3", "4"]}
        assert cluster_to_signatures == expected_cluster_to_signatures

    def test_initialization(self):
        make_dataset = partial(
            ANDData,
            signatures={},
            papers={},
            name="initialization",
            mode="train",
            name_counts_index=None,
            preprocess=False,
        )
        for options, message in (
            (
                {"clusters": {}, "unit_of_data_split": "blocks", "pair_sampling_mode": "global_balanced_classes"},
                "Block-based cluster splits",
            ),
            ({"clusters": {}, "train_pairs": []}, "Set exactly one"),
            ({}, "Set exactly one"),
            ({"train_blocks": [], "train_pairs": []}, "both train_blocks and train_pairs"),
            ({"train_blocks": []}, "Train blocks still needs clusters"),
            ({"mode": "dummy", "clusters": {}}, "Unknown mode"),
            ({"clusters": {}, "pair_sampling_mode": "global_unbalanced"}, "Unknown pair_sampling_mode"),
        ):
            with pytest.raises(ValueError, match=message):
                make_dataset(**options)

        dataset = make_dataset(mode="inference")
        assert dataset.signature_to_cluster_id is None
        assert dataset.pair_sampling_mode == "within_block_random"
        assert dataset.all_test_pairs_flag

    def test_multiprocessing_preprocessing_consistency(self):
        dataset_single = build_dummy_dataset("preprocessing_single", n_jobs=1)
        dataset_multi = build_dummy_dataset("preprocessing_multi", n_jobs=2)

        assert dataset_single.papers
        assert dataset_single.papers == dataset_multi.papers
        assert dataset_single.signatures == dataset_multi.signatures


def test_preprocessing_name_counts_use_single_character_initial(tmp_path):
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

    index_path, _metrics = write_name_counts_index(
        tmp_path,
        ({}, {"sattar": 11}, {}, {"sattar a": 17}),
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


def test_default_name_counts_index_populates_canonical_features(monkeypatch: pytest.MonkeyPatch) -> None:
    canonical_index = tiny_name_counts_index()
    opened_paths = []

    def open_canonical_index(_cls, path):
        opened_paths.append(path)
        return canonical_index

    monkeypatch.setattr(data_module.NameCountsIndex, "open", classmethod(open_canonical_index))

    dataset = ANDData(
        signatures={
            "s1": {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "a sattary",
                    "first": "Abdul",
                    "middle": "",
                    "last": "Sattar",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                },
            }
        },
        papers={
            "1": {
                "paper_id": 1,
                "title": "A paper",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2026,
                "authors": [{"position": 0, "author_name": "Abdul Sattar"}],
                "references": [],
            }
        },
        name="default_name_counts",
        mode="inference",
        preprocess=True,
        n_jobs=1,
    )

    assert opened_paths == [data_module.NAME_COUNTS_INDEX_PATH]
    assert dataset.signatures["s1"].author_info_name_counts == data_module.NameCounts(10.0, 40.0, 60.0, 90.0)


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


def test_fixed_pairs_validate_and_preserve_source_dataframes():
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
    dataset.val_pairs = pd.DataFrame([("s2", "s1", 1)], columns=train_pairs_df.columns)
    with pytest.raises(ValueError, match="overlap by unordered signature pair"):
        dataset.fixed_pairs()

    dataset.val_pairs = pd.DataFrame([("s5", "s6", "MAYBE")], columns=train_pairs_df.columns)
    with pytest.raises(ValueError, match="Unknown fixed-pair labels"):
        dataset.fixed_pairs()


def test_fixed_train_val_pairs_matches_fixed_pairs_without_test_or_rng_side_effects():
    columns = ["signature_id_1", "signature_id_2", "label"]
    dataset = ANDData.__new__(ANDData)
    dataset.train_pairs = pd.DataFrame(
        [(f"s{index}", f"s{index + 1}", index % 2) for index in range(0, 20, 2)],
        columns=columns,
    )
    dataset.val_pairs = None
    dataset.test_pairs = None
    dataset.train_ratio = 0.8
    dataset.val_ratio = 0.1
    dataset.random_seed = 1111

    np.random.seed(7)
    state_before = np.random.get_state()
    selection_train, selection_val = dataset.fixed_train_val_pairs()
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]

    dataset.test_pairs = pd.DataFrame([("test-1", "test-2", 0)], columns=columns)
    full_train, full_val, _ = dataset.fixed_pairs()
    assert selection_train == full_train
    assert selection_val == full_val


def test_fixed_pair_csv_keeps_signature_ids_as_text(tmp_path):
    from s2and.data import _map_fixed_pair_labels

    path = tmp_path / "train_pairs.csv"
    path.write_text("signature_id_1,signature_id_2,label\n00,01,1.0\nNA,7,0.0\n", encoding="utf-8")

    frame = _map_fixed_pair_labels(ANDData.maybe_load_dataframe(str(path)), "train")

    assert frame.values.tolist() == [["00", "01", 1], ["NA", "7", 0]]


@pytest.mark.parametrize("seed_id", [-2, "-1"])
def test_negative_seed_ids_do_not_require_unseeded_signatures(seed_id):
    from tests.incremental_test_helpers import _dataset

    # Ada and Beth conflict by name; a seed ID equal to a missing-key sentinel must not force them together.
    dataset = _dataset([("Ada", "", "Smith"), ("Beth", "", "Smith")], {})
    dataset.cluster_seeds_require = {"1": seed_id}

    assert dataset.get_constraint("0", "1") == dataset.get_constraint("1", "0")
    assert dataset.get_constraint("0", "1") != 0
