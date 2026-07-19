from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS
from scripts._pair_ablation.prepared import load_prepared
from scripts._pair_ablation.study import BASE_FAMILY, PAIR_COLUMNS


def _save(root: Path, relative: str, values: np.ndarray) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


def _write_prepared(root: Path) -> None:
    training = root / "training"
    training.mkdir(parents=True)
    pd.DataFrame(
        [
            ("aminer", BASE_FAMILY, "a", "b", 0),
            ("arnetminer", BASE_FAMILY, "c", "d", 1),
        ],
        columns=PAIR_COLUMNS,
    ).to_parquet(training / "catalog.parquet", index=False)
    (training / "feature_schema.json").write_text(
        json.dumps(
            {
                "featurizer_version": FEATURIZER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "main_feature_groups": list(DEFAULT_FEATURE_GROUPS),
                "nameless_feature_groups": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
            }
        ),
        encoding="utf-8",
    )
    _save(root, "training/main.npy", np.arange(66, dtype=np.float64).reshape(2, 33))
    _save(root, "training/nameless.npy", np.arange(34, dtype=np.float32).reshape(2, 17))

    for domain in ("aminer", "medline"):
        _save(root, f"evaluation/{domain}/main.npy", np.arange(132, dtype=np.float64).reshape(4, 33))
        _save(root, f"evaluation/{domain}/nameless.npy", np.arange(68, dtype=np.float32).reshape(4, 17))
        _save(root, f"evaluation/{domain}/labels.npy", np.array([0, 1, 0, 1], dtype=np.int8))

    _save(root, "b3/aminer/main.npy", np.arange(132, dtype=np.float64).reshape(4, 33))
    _save(root, "b3/aminer/nameless.npy", np.arange(68, dtype=np.float64).reshape(4, 17))
    _save(
        root,
        "b3/aminer/staged_labels.npy",
        np.array([np.nan, -100_000.0, np.nan, -90_000.0]),
    )
    _save(root, "b3/aminer/pair_offsets.npy", np.array([0, 1, 4], dtype=np.int64))
    _save(root, "b3/aminer/signature_offsets.npy", np.array([0, 2, 5], dtype=np.int64))
    _save(root, "b3/aminer/signature_ids.npy", np.array(["s0", "s1", "s2", "s3", "s4"]))
    _save(root, "b3/aminer/gold_cluster_ids.npy", np.array(["g0", "g0", "g1", "g1", "g2"]))


def test_load_prepared_maps_and_validates_the_minimal_layout(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    prepared = load_prepared(tmp_path)

    assert prepared.root == tmp_path
    assert prepared.catalog["feature_row"].tolist() == [0, 1]
    assert isinstance(prepared.training_main, np.memmap)
    assert isinstance(prepared.training_nameless, np.memmap)
    assert set(prepared.evaluation) == {"aminer", "medline"}
    assert set(prepared.b3) == {"aminer"}
    assert prepared.evaluation["aminer"].main.shape == (4, 33)
    assert prepared.b3["aminer"].signature_ids.tolist() == ["s0", "s1", "s2", "s3", "s4"]
    assert len(prepared.prepared_digest) == 64


def test_digest_uses_required_relative_paths_and_file_bytes(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    first = load_prepared(tmp_path).prepared_digest
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert load_prepared(tmp_path).prepared_digest == first

    values = np.load(tmp_path / "evaluation" / "medline" / "main.npy", allow_pickle=False)
    values[0, 0] = 999
    _save(tmp_path, "evaluation/medline/main.npy", values)
    assert load_prepared(tmp_path).prepared_digest != first


def test_nan_features_and_staged_constraints_are_allowed(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    values = np.load(tmp_path / "training" / "main.npy", allow_pickle=False)
    values[0, 0] = np.nan
    _save(tmp_path, "training/main.npy", values)
    prepared = load_prepared(tmp_path)
    assert np.isnan(prepared.training_main[0, 0])
    assert np.isnan(prepared.b3["aminer"].staged_labels[0])


@pytest.mark.parametrize(
    ("relative", "values", "match"),
    (
        ("training/main.npy", np.ones((3, 33)), "row count"),
        ("training/main.npy", np.ones((2, 0)), "nonempty-width"),
        ("training/nameless.npy", np.ones((2, 0)), "nonempty-width"),
        ("evaluation/medline/main.npy", np.ones((4, 4)), "width"),
        ("evaluation/medline/nameless.npy", np.ones((3, 17)), "row count"),
        ("evaluation/medline/main.npy", np.ones(4), "float32/float64"),
        ("evaluation/medline/main.npy", np.array([["x"]] * 4), "float32/float64"),
        (
            "evaluation/medline/main.npy",
            np.array([[np.inf] + [0.0] * 32] * 4),
            "infinity",
        ),
        (
            "evaluation/medline/main.npy",
            np.asfortranarray(np.ones((4, 33))),
            "C-contiguous",
        ),
        (
            "b3/aminer/staged_labels.npy",
            np.array([0.0, 1.0, np.inf, np.nan]),
            "infinity",
        ),
    ),
)
def test_matrix_contracts_fail_fast(
    tmp_path: Path,
    relative: str,
    values: np.ndarray,
    match: str,
) -> None:
    _write_prepared(tmp_path)
    _save(tmp_path, relative, values)
    with pytest.raises(ValueError, match=match):
        load_prepared(tmp_path)


@pytest.mark.parametrize(
    "values",
    (
        np.array([0, 1, 2, 1], dtype=np.int8),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([1, 1, 1, 1], dtype=np.int8),
        np.array([[0, 1], [0, 1]], dtype=np.int8),
    ),
)
def test_pair_labels_are_exact_binary_and_have_both_classes(
    tmp_path: Path,
    values: np.ndarray,
) -> None:
    _write_prepared(tmp_path)
    _save(tmp_path, "evaluation/medline/labels.npy", values)
    with pytest.raises(ValueError, match="labels|both classes"):
        load_prepared(tmp_path)


def test_evaluation_must_be_a_nonempty_known_subset(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    (tmp_path / "evaluation" / "unknown").mkdir()
    with pytest.raises(ValueError, match="known-domain subset"):
        load_prepared(tmp_path)


def test_b3_directories_exactly_match_loaded_gold_domains(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    (tmp_path / "b3" / "medline").mkdir()
    with pytest.raises(ValueError, match="loaded gold evaluation domains"):
        load_prepared(tmp_path)


@pytest.mark.parametrize(
    ("relative", "values", "match"),
    (
        (
            "b3/aminer/signature_ids.npy",
            np.array(["s0", "s1", "s2", "s3", "s3"]),
            "unique",
        ),
        (
            "b3/aminer/gold_cluster_ids.npy",
            np.array(["g0", "g0", "g1", "g1"]),
            "aligned",
        ),
        (
            "b3/aminer/signature_ids.npy",
            np.array([0, 1, 2, 3, 4]),
            "Unicode array",
        ),
        (
            "b3/aminer/pair_offsets.npy",
            np.array([1, 2, 4]),
            "never decrease",
        ),
        (
            "b3/aminer/signature_offsets.npy",
            np.array([0, 3, 5], dtype=np.int64),
            "nC2",
        ),
        (
            "b3/aminer/pair_offsets.npy",
            np.array([0.0, 1.0, 4.0]),
            "never decrease",
        ),
        (
            "b3/aminer/signature_offsets.npy",
            np.array([0, 2, 1, 3], dtype=np.uint8),
            "strictly increase",
        ),
        (
            "b3/aminer/signature_offsets.npy",
            np.array([0, 2**62, 5], dtype=np.int64),
            "strictly increase",
        ),
    ),
)
def test_b3_layout_contracts_fail_fast(
    tmp_path: Path,
    relative: str,
    values: np.ndarray,
    match: str,
) -> None:
    _write_prepared(tmp_path)
    _save(tmp_path, relative, values)
    with pytest.raises(ValueError, match=match):
        load_prepared(tmp_path)


def test_object_arrays_are_never_unpickled(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    _save(tmp_path, "evaluation/medline/labels.npy", np.array([0, 1, 0, 1], dtype=object))
    with pytest.raises(ValueError, match="Python objects|Object arrays"):
        load_prepared(tmp_path)


def test_feature_schema_and_unicode_ids_are_strict(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    (tmp_path / "training" / "feature_schema.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="feature schema"):
        load_prepared(tmp_path)

    _write_prepared(tmp_path / "bytes")
    _save(tmp_path / "bytes", "b3/aminer/signature_ids.npy", np.array([b"", b"s1", b"s2", b"s3", b"s4"]))
    with pytest.raises(ValueError, match="Unicode"):
        load_prepared(tmp_path / "bytes")


def test_pair_offsets_allow_singleton_blocks(tmp_path: Path) -> None:
    _write_prepared(tmp_path)
    _save(tmp_path, "b3/aminer/pair_offsets.npy", np.array([0, 0, 1, 4], dtype=np.int64))
    _save(tmp_path, "b3/aminer/signature_offsets.npy", np.array([0, 1, 3, 6], dtype=np.int64))
    _save(tmp_path, "b3/aminer/signature_ids.npy", np.array(["s0", "s1", "s2", "s3", "s4", "s5"]))
    _save(tmp_path, "b3/aminer/gold_cluster_ids.npy", np.array(["g0", "g1", "g1", "g2", "g2", "g3"]))
    prepared = load_prepared(tmp_path)
    assert prepared.b3["aminer"].pair_offsets.tolist() == [0, 0, 1, 4]
