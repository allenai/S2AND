"""Fixed-pair ingestion preserves named identities through featurization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize, resolve_selection_pairs


def test_reordered_fixed_pair_csv_preserves_splits_features_and_labels(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    train_path.write_text("label,signature_id_1,signature_id_2\nYES,2,0\n", encoding="utf-8")
    val_path = tmp_path / "val.csv"
    val_path.write_text("signature_id_2,label,signature_id_1\n4,NO,3\n", encoding="utf-8")
    test_path = tmp_path / "test.csv"
    test_path.write_text("signature_id_1,label,signature_id_2\n5,YES,6\n", encoding="utf-8")
    dataset = ANDData(
        signatures="tests/dummy/signatures.json",
        papers="tests/dummy/papers.json",
        name="fixed_pair_csv_order",
        mode="train",
        train_pairs=str(train_path),
        val_pairs=str(val_path),
        test_pairs=str(test_path),
        name_counts_index=None,
        name_tuples=set(),
        n_jobs=1,
    )

    train, val = resolve_selection_pairs(dataset)
    all_splits = dataset.fixed_pairs()
    assert train == all_splits[0]
    assert val == all_splits[1]
    assert [[(str(left), str(right), int(label)) for left, right, label in split] for split in all_splits] == [
        [("2", "0", 1)],
        [("3", "4", 0)],
        [("5", "6", 1)],
    ]

    info = FeaturizationInfo()
    observed_features, observed_labels, _ = many_pairs_featurize(train, dataset, info, n_jobs=1)
    expected_features, expected_labels, _ = many_pairs_featurize([("2", "0", 1)], dataset, info, n_jobs=1)
    np.testing.assert_array_equal(observed_features, expected_features)
    np.testing.assert_array_equal(observed_labels, expected_labels)
    assert observed_labels.tolist() == [1.0]
    assert observed_features[0, info.get_feature_names().index("year_diff")] == 2.0


@pytest.mark.parametrize("left_column", ["pair1", "pairs1"])
def test_legacy_fixed_pair_headers_preserve_automatic_validation_split(left_column: str) -> None:
    rows = [(f"left-{index}", f"right-{index}", index % 2) for index in range(20)]
    frame = pd.DataFrame(rows, columns=[left_column, "pair2", "label"])
    frame["dataset_name"] = "fixture"
    dataset = ANDData(
        signatures={},
        papers={},
        name="legacy_fixed_pair_headers",
        mode="train",
        train_pairs=frame[["label", "dataset_name", "pair2", left_column]],
        test_pairs=pd.DataFrame([("YES", "test-left", "test-right")], columns=["label", left_column, "pair2"]),
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
        random_seed=1111,
    )

    train, val = dataset.fixed_train_val_pairs()
    assert train and val
    assert {tuple(pair) for pair in train + val} == set(rows)
    full_train, full_val, test = dataset.fixed_pairs()
    assert train == full_train
    assert val == full_val
    assert [tuple(pair) for pair in test] == [("test-left", "test-right", 1)]


@pytest.mark.parametrize(
    "columns",
    [
        ["left", "right", "label"],
        ["signature_id_1", "signature_id_2", "pair1", "pair2", "label"],
        ["signature_id_1", "signature_id_2", "label", "label"],
    ],
    ids=["unnamed-identities", "ambiguous-identities", "duplicate-label"],
)
def test_fixed_pairs_reject_ambiguous_or_missing_named_columns(columns: list[str]) -> None:
    dataset = ANDData(
        signatures={},
        papers={},
        name="invalid_fixed_pair_columns",
        mode="train",
        train_pairs=pd.DataFrame([["YES"] * len(columns)], columns=columns),
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
    )

    with pytest.raises(ValueError, match="fixed-pair columns in train"):
        dataset.fixed_train_val_pairs()
