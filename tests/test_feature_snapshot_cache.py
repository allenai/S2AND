"""Tests for the training-time featurized-split snapshot cache."""

from __future__ import annotations

import hashlib
import os
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import s2and.feature_cache as feature_cache
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from scripts.production.model import train_pairwise
from scripts.production.model.train_pairwise import _feature_snapshot_source_key, _feature_source_hashes


def _dataset() -> ANDData:
    return ANDData(
        signatures="tests/dummy/signatures.json",
        papers="tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="snapshot_cache_dummy",
        mode="train",
        name_counts_index=None,
        name_tuples=set(),
        preprocess=True,
        n_jobs=1,
        train_pairs_size=100,
        val_pairs_size=50,
        test_pairs_size=50,
    )


def _split_pairs(dataset: ANDData):
    cluster_of = dataset.signature_to_cluster_id or {}
    pairs = [
        (str(first), str(second), float(cluster_of.get(first) == cluster_of.get(second)))
        for first, second in combinations(sorted(dataset.signatures), 2)
    ]
    return pairs[0::3], pairs[1::3], pairs[2::3]


def _source_key(marker: str = "0") -> dict[str, object]:
    return {
        "signatures_sha256": marker * 64,
        "papers_sha256": "1" * 64,
        "specter_embeddings_sha256": "2" * 64,
        "name_tuples_data_sha256": "3" * 64,
        "name_counts_manifest_sha256": "4" * 64,
        "normalization_version": "canonical_v2",
    }


@pytest.fixture()
def dataset(monkeypatch: pytest.MonkeyPatch) -> ANDData:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(feature_cache, "resolve_training_pairs", _split_pairs)
    return _dataset()


def _assert_same_splits(first, second) -> None:
    for first_split, second_split in zip(first, second, strict=True):
        for first_array, second_array in zip(first_split, second_split, strict=True):
            if first_array is None or second_array is None:
                assert first_array is second_array is None
            else:
                np.testing.assert_array_equal(first_array, second_array)


def test_cold_warm_and_uncached_outputs_are_identical(
    dataset: ANDData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    nameless_info = FeaturizationInfo(features_to_use=["year_diff"])
    pairs = _split_pairs(dataset)
    uncached = tuple(
        feature_cache.many_pairs_featurize(
            split_pairs,
            dataset,
            info,
            n_jobs=1,
            nameless_featurizer_info=nameless_info,
        )
        for split_pairs in pairs
    )
    cold = feature_cache.cached_featurize(
        dataset,
        info,
        source_key=_source_key(),
        cache_dir=tmp_path,
        nameless_featurizer_info=nameless_info,
    )

    snapshots = sorted(tmp_path.glob("*.npz"))
    assert len(snapshots) == 3
    assert {path.name.split("_", 1)[0] for path in snapshots} == {"train", "val", "test"}
    assert all(len(path.stem.rsplit("_", 1)[1]) == 64 for path in snapshots)
    for path in snapshots:
        with np.load(path, allow_pickle=False) as loaded:
            assert set(loaded.files) == {"X", "y", "nameless_X"}

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("warm snapshots must skip featurization")

    monkeypatch.setattr(feature_cache, "many_pairs_featurize", fail_if_recomputed)
    warm = feature_cache.cached_featurize(
        dataset,
        info,
        source_key=_source_key(),
        cache_dir=tmp_path,
        nameless_featurizer_info=nameless_info,
    )
    _assert_same_splits(uncached, cold)
    _assert_same_splits(cold, warm)


def test_losing_publisher_loads_winning_snapshot(
    dataset: ANDData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = feature_cache._publish_snapshot

    def publish_winner(path: Path, arrays: dict[str, np.ndarray]) -> bool:
        winner = {name: np.full(array.shape, 9.0, dtype=np.float64) for name, array in arrays.items()}
        assert original_publish(path, winner) is True
        return False

    monkeypatch.setattr(feature_cache, "_publish_snapshot", publish_winner)
    results = feature_cache.cached_featurize(
        dataset,
        FeaturizationInfo(features_to_use=["year_diff"]),
        source_key=_source_key(),
        cache_dir=tmp_path,
    )

    for features, labels, nameless in results:
        assert np.all(features == 9.0)
        assert np.all(labels == 9.0)
        assert nameless is None


def test_source_key_change_creates_new_snapshots(
    dataset: ANDData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_featurize(pairs, _dataset, info, *, nameless_featurizer_info=None, **_kwargs):
        nonlocal calls
        calls += 1
        rows = len(pairs)
        features = np.full((rows, len(info.selected_feature_indices())), calls, dtype=np.float64)
        labels = np.asarray([float(pair[2]) for pair in pairs], dtype=np.float64)
        nameless = (
            None
            if nameless_featurizer_info is None
            else np.zeros((rows, len(nameless_featurizer_info.selected_feature_indices())), dtype=np.float64)
        )
        return features, labels, nameless

    monkeypatch.setattr(feature_cache, "many_pairs_featurize", fake_featurize)
    info = FeaturizationInfo(features_to_use=["year_diff"])
    feature_cache.cached_featurize(dataset, info, source_key=_source_key("a"), cache_dir=tmp_path)
    feature_cache.cached_featurize(dataset, info, source_key=_source_key("a"), cache_dir=tmp_path)
    assert calls == 3
    assert len(list(tmp_path.glob("*.npz"))) == 3

    feature_cache.cached_featurize(dataset, info, source_key=_source_key("b"), cache_dir=tmp_path)
    assert calls == 6
    assert len(list(tmp_path.glob("*.npz"))) == 6


def test_snapshot_identity_covers_pairs_and_featurizer_options() -> None:
    info = FeaturizationInfo(features_to_use=["year_diff"], featurizer_version=10)
    pairs = [("a", "b", 1.0), ("c", "d", 0.0)]

    def key(
        *,
        source_key: dict[str, object] | None = None,
        featurizer_info: FeaturizationInfo = info,
        nameless_info: FeaturizationInfo | None = None,
        nan_value: float = np.nan,
        pair_list=pairs,
    ) -> str:
        return feature_cache._snapshot_key(
            source_key=_source_key() if source_key is None else source_key,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_info,
            nan_value=nan_value,
            pair_list_hash=feature_cache._hash_pair_list(pair_list),
        )

    base = key()
    assert key(source_key=_source_key("f")) != base
    assert key(featurizer_info=FeaturizationInfo(features_to_use=["year_diff"], featurizer_version=11)) != base
    assert key(nameless_info=FeaturizationInfo(features_to_use=["year_diff"])) != base
    assert key(nan_value=-1.0) != base
    assert key(pair_list=list(reversed(pairs))) != base
    assert key(source_key=dict(reversed(list(_source_key().items())))) == base


def test_invalid_cold_arrays_are_not_published(
    dataset: ANDData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_featurize(pairs, *_args, **_kwargs):
        return np.zeros((len(pairs), 1), dtype=np.float32), np.zeros(len(pairs)), None

    monkeypatch.setattr(feature_cache, "many_pairs_featurize", invalid_featurize)
    with pytest.raises(ValueError, match="dtype float32"):
        feature_cache.cached_featurize(
            dataset,
            FeaturizationInfo(features_to_use=["year_diff"]),
            source_key=_source_key(),
            cache_dir=tmp_path,
        )
    assert list(tmp_path.glob("*.npz")) == []


def test_corrupt_snapshot_raises_with_path(dataset: ANDData, tmp_path: Path) -> None:
    info = FeaturizationInfo(features_to_use=["year_diff"])
    feature_cache.cached_featurize(dataset, info, source_key=_source_key(), cache_dir=tmp_path)
    snapshot = sorted(tmp_path.glob("*.npz"))[0]
    snapshot.write_bytes(b"not an npz")

    with pytest.raises(ValueError, match="is unreadable"):
        feature_cache.cached_featurize(dataset, info, source_key=_source_key(), cache_dir=tmp_path)


@pytest.mark.parametrize(
    ("arrays", "nameless_width", "message"),
    [
        ({"X": np.zeros((2, 1)), "y": np.zeros(2), "extra": np.zeros(1)}, None, "has members"),
        ({"X": np.zeros((2, 1), dtype=np.float32), "y": np.zeros(2)}, None, "dtype float32"),
        ({"X": np.zeros((2, 2)), "y": np.zeros(2)}, None, r"expected \(2, 1\)"),
        ({"X": np.zeros((2, 1)), "y": np.zeros((2, 1))}, None, r"expected \(2,\)"),
        (
            {"X": np.zeros((2, 1)), "y": np.zeros(2), "nameless_X": np.zeros((1, 1))},
            1,
            "nameless_X has shape",
        ),
    ],
)
def test_snapshot_member_dtype_and_shape_validation(
    arrays: dict[str, np.ndarray],
    nameless_width: int | None,
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=message):
        feature_cache._validate_snapshot_arrays(
            arrays,
            path=tmp_path / "snapshot.npz",
            expected_rows=2,
            expected_width=1,
            expected_nameless_width=nameless_width,
        )


def test_publication_is_write_once_while_existing_snapshot_is_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "snapshot.npz"
    original_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def record_replace(source, destination) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(feature_cache.os, "replace", record_replace)
    first_published = feature_cache._publish_snapshot(path, {"X": np.zeros((1, 1)), "y": np.zeros(1)})
    with np.load(path, allow_pickle=False) as open_reader:
        second_published = feature_cache._publish_snapshot(path, {"X": np.ones((1, 1)), "y": np.ones(1)})
        np.testing.assert_array_equal(open_reader["X"], np.zeros((1, 1)))

    assert first_published is True
    assert second_published is False
    assert len(replacements) == 1
    assert all(source.parent == path.parent and destination == path for source, destination in replacements)
    with np.load(path, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(loaded["X"], np.zeros((1, 1)))
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_publication_cleans_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_savez(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(feature_cache.np, "savez", broken_savez)
    path = tmp_path / "snapshot.npz"
    with pytest.raises(OSError, match="disk full"):
        feature_cache._publish_snapshot(path, {"X": np.zeros((1, 1)), "y": np.zeros(1)})
    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_load_does_not_relabel_permission_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(feature_cache.np, "load", denied)
    with pytest.raises(PermissionError, match="denied"):
        feature_cache._load_snapshot(
            tmp_path / "snapshot.npz",
            expected_rows=2,
            expected_width=1,
            expected_nameless_width=None,
        )


def test_production_source_key_uses_exact_files_and_opened_artifacts(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("signatures.json", "papers.json", "specter.pkl")]
    payloads = [b"signatures", b"papers", b"pickle payload\x00including trailing bytes"]
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(payload)
    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="tiny",
            name_counts_provenance={"manifest_sha256": "4" * 64},
            normalization_version="canonical_v2",
        ),
    )

    assert _feature_snapshot_source_key(
        dataset,
        source_hashes=_feature_source_hashes(
            signatures_path=paths[0],
            papers_path=paths[1],
            specter_path=paths[2],
        ),
        name_tuples_data_sha256="3" * 64,
    ) == {
        "signatures_sha256": hashlib.sha256(payloads[0]).hexdigest(),
        "papers_sha256": hashlib.sha256(payloads[1]).hexdigest(),
        "specter_embeddings_sha256": hashlib.sha256(payloads[2]).hexdigest(),
        "name_tuples_data_sha256": "3" * 64,
        "name_counts_manifest_sha256": "4" * 64,
        "normalization_version": "canonical_v2",
    }


def test_cacheable_anddata_rejects_source_change_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / name for name in ("signatures.json", "papers.json", "specter.pkl")]
    for path in paths:
        path.write_bytes(b"original")

    dataset = cast(
        ANDData,
        SimpleNamespace(
            name="tiny",
            name_counts_provenance={"manifest_sha256": "4" * 64},
            normalization_version="canonical_v2",
        ),
    )

    def mutate_signatures(**_kwargs) -> ANDData:
        paths[0].write_bytes(b"changed")
        return dataset

    monkeypatch.setattr(train_pairwise, "ANDData", mutate_signatures)
    with pytest.raises(RuntimeError, match="signatures"):
        train_pairwise._build_cacheable_anddata(
            {
                "signatures": paths[0],
                "papers": paths[1],
                "specter_embeddings": paths[2],
            },
            name_tuples_data_sha256="3" * 64,
        )
