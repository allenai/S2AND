from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts._pair_ablation.feature_artifact import (
    load_feature_store,
    pair_identity_digest,
    validate_binary_labels,
    write_feature_store,
)
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS

ARTIFACT_DIGEST = "a" * 64
MANIFEST_DIGEST = "b" * 64
MAIN_INDICES = (0, 2)
NAMELESS_INDICES = (1,)


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_domain": "qian",
                "source_family": "gold_cluster_uniform",
                "pair1": "1",
                "pair2": "2",
                "label": 1,
                "label_rule": "gold",
                "origin": "fixture",
                "group_id": "qian:1",
            },
            {
                "source_domain": "qian",
                "source_family": "gold_cluster_uniform",
                "pair1": "3",
                "pair2": "4",
                "label": 0,
                "label_rule": "gold",
                "origin": "fixture",
                "group_id": "qian:3",
            },
        ],
        columns=list(PAIR_COLUMNS),
    )


def _write(root: Path) -> tuple[Path, str]:
    pairs = _pairs()
    store = root / "qian"
    write_feature_store(
        store,
        domain="qian",
        pairs=pairs,
        main=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        nameless=np.asarray([[5.0], [6.0]], dtype=np.float64),
        labels=np.asarray([1, 0], dtype=np.int64),
        artifact_identity_digest=ARTIFACT_DIGEST,
        artifact_manifest_sha256=MANIFEST_DIGEST,
        main_feature_indices=MAIN_INDICES,
        nameless_feature_indices=NAMELESS_INDICES,
    )
    return store, pair_identity_digest(pairs)


def _load(store: Path, pair_digest: str):
    return load_feature_store(
        store,
        expected_domain="qian",
        expected_pair_digest=pair_digest,
        expected_artifact_identity_digest=ARTIFACT_DIGEST,
        expected_main_feature_indices=MAIN_INDICES,
        expected_nameless_feature_indices=NAMELESS_INDICES,
    )


def test_feature_store_round_trip_is_memory_mapped_and_verified(tmp_path: Path) -> None:
    store, pair_digest = _write(tmp_path)

    loaded = _load(store, pair_digest)

    assert loaded.domain == "qian"
    assert isinstance(loaded.main, np.memmap)
    assert isinstance(loaded.nameless, np.memmap)
    assert isinstance(loaded.labels, np.memmap)
    assert loaded.main.dtype == np.dtype("float32")
    assert loaded.nameless.dtype == np.dtype("float64")
    assert loaded.labels.dtype == np.dtype("int8")
    assert loaded.row_by_pair == {("1", "2"): 0, ("3", "4"): 1}


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        ([0.5, 1.0], "exact finite binary"),
        ([0, 256], "exact finite binary"),
        ([0, float("nan")], "exact finite binary"),
        ([0, float("inf")], "exact finite binary"),
        (["0", "1"], "numeric or Boolean"),
        ([[0, 1]], "one-dimensional"),
    ],
)
def test_binary_label_validator_rejects_coercible_or_invalid_values(labels: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_binary_labels(labels, context="labels")


@pytest.mark.parametrize("labels", [[0, 1], np.asarray([False, True]), np.asarray([0.0, 1.0])])
def test_binary_label_validator_accepts_exact_binary_values(labels: object) -> None:
    assert np.array_equal(validate_binary_labels(labels, context="labels"), np.asarray([0, 1], dtype=np.int8))


def test_feature_store_writer_rejects_label_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match"):
        write_feature_store(
            tmp_path / "qian",
            domain="qian",
            pairs=_pairs(),
            main=np.ones((2, 2), dtype=np.float32),
            nameless=np.ones((2, 1), dtype=np.float32),
            labels=np.asarray([0, 1]),
            artifact_identity_digest=ARTIFACT_DIGEST,
            artifact_manifest_sha256=MANIFEST_DIGEST,
            main_feature_indices=MAIN_INDICES,
            nameless_feature_indices=NAMELESS_INDICES,
        )


def test_feature_store_loader_rejects_content_hash_drift(tmp_path: Path) -> None:
    store, pair_digest = _write(tmp_path)
    main = np.load(store / "main.npy")
    main[0, 0] += 1
    np.save(store / "main.npy", main)

    with pytest.raises(ValueError, match="output hash mismatch"):
        _load(store, pair_digest)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "unknown", "Unsupported feature-store schema"),
        ("domain", "pubmed", "domain identity mismatch"),
        ("pair_digest", "c" * 64, "pair identity mismatch"),
        ("artifact_identity_digest", "d" * 64, "input-artifact identity mismatch"),
        ("main_feature_indices", [1, 2], "main feature indices mismatch"),
        ("output_dtypes", {"main.npy": "float64", "nameless.npy": "float64", "labels.npy": "int8"}, "dtype mismatch"),
        ("output_shapes", {"main.npy": [3, 2], "nameless.npy": [2, 1], "labels.npy": [2]}, "shape mismatch"),
    ],
)
def test_feature_store_loader_rejects_manifest_drift(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    store, pair_digest = _write(tmp_path)
    path = store / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load(store, pair_digest)


def test_feature_store_loader_rejects_expected_identity_drift(tmp_path: Path) -> None:
    store, pair_digest = _write(tmp_path)

    with pytest.raises(ValueError, match="pair identity mismatch"):
        load_feature_store(
            store,
            expected_domain="qian",
            expected_pair_digest="f" * 64,
            expected_artifact_identity_digest=ARTIFACT_DIGEST,
            expected_main_feature_indices=MAIN_INDICES,
            expected_nameless_feature_indices=NAMELESS_INDICES,
        )
