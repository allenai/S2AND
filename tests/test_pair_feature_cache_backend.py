from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import s2and.featurizer as featurizer_mod
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import build_dummy_dataset


def _patch_pair_cache_paths(featurizer_info: FeaturizationInfo, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db_path = cache_dir / featurizer_mod.PAIR_FEATURE_CACHE_DB_FILENAME
    featurizer_info.cache_directory = lambda _dataset_name: str(cache_dir)  # type: ignore[method-assign]
    featurizer_info.cache_db_path = lambda _dataset_name: str(cache_db_path)  # type: ignore[method-assign]
    return cache_db_path


def test_write_cache_persists_to_sqlite_and_load_cache_round_trips(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path = _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    feature_vector = np.arange(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    featurizer_info.write_cache({"a___b": feature_vector}, "dataset")

    loaded = featurizer_info.load_cache("dataset", ["a___b"])
    np.testing.assert_array_equal(loaded["a___b"], feature_vector)
    assert cache_db_path.exists()
    with sqlite3.connect(cache_db_path) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM cache_metadata"))
    assert metadata == {
        "feature_count": str(featurizer_mod.NUM_FEATURES),
        "schema_version": str(featurizer_mod.PAIR_FEATURE_CACHE_SCHEMA_VERSION),
    }


def test_load_cache_reads_only_requested_rows(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    features = {
        "a___b": np.zeros(featurizer_mod.NUM_FEATURES, dtype=np.float64),
        "c___d": np.ones(featurizer_mod.NUM_FEATURES, dtype=np.float64),
    }
    featurizer_info.write_cache(features, "dataset")

    loaded = featurizer_info.load_cache("dataset", ["c___d", "missing___pair"])

    assert set(loaded) == {"c___d"}
    np.testing.assert_array_equal(loaded["c___d"], features["c___d"])


def test_write_cache_rejects_feature_vector_with_wrong_shape(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")

    with pytest.raises(ValueError, match="unexpected shape"):
        featurizer_info.write_cache({"a___b": np.zeros(1, dtype=np.float64)}, "dataset")


def test_load_cache_rejects_incompatible_feature_count_metadata(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path = _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    feature_vector = np.zeros(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    featurizer_info.write_cache({"a___b": feature_vector}, "dataset")
    with sqlite3.connect(cache_db_path) as connection:
        connection.execute(
            "UPDATE cache_metadata SET value = ? WHERE key = ?",
            (str(featurizer_mod.NUM_FEATURES + 1), "feature_count"),
        )

    with pytest.raises(RuntimeError, match="Unsupported pair-feature cache feature count"):
        featurizer_info.load_cache("dataset", ["a___b"])


@pytest.mark.parametrize("metadata_key", ["schema_version", "feature_count"])
def test_load_cache_rejects_missing_required_metadata(tmp_path: Path, metadata_key: str) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path = _patch_pair_cache_paths(featurizer_info, tmp_path / metadata_key)
    feature_vector = np.zeros(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    featurizer_info.write_cache({"a___b": feature_vector}, "dataset")
    with sqlite3.connect(cache_db_path) as connection:
        connection.execute("DELETE FROM cache_metadata WHERE key = ?", (metadata_key,))

    with pytest.raises(RuntimeError, match=rf"missing required metadata.*{metadata_key}"):
        featurizer_info.load_cache("dataset", ["a___b"])


def test_load_cache_ignores_unrecognized_json_cache_files(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path = _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    legacy_cache_path = tmp_path / "pair_cache" / "all_features.json"
    legacy_feature = np.arange(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    legacy_cache_path.write_text(
        json.dumps({"features": {"legacy___pair": legacy_feature.tolist()}}),
        encoding="utf-8",
    )

    loaded = featurizer_info.load_cache("dataset", ["legacy___pair"])
    assert loaded == {}
    assert not cache_db_path.exists()


def test_many_pairs_featurize_reuses_persisted_pair_feature_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(featurizer_mod, "CACHE_ROOT", tmp_path)
    dataset = build_dummy_dataset("pair_feature_cache_roundtrip", name_counts_index=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0.0), ("0", "2", 0.0)]
    original_single_pair_featurize = featurizer_mod._single_pair_featurize
    call_count = {"count": 0}

    def _tracked_single_pair_featurize(*args, **kwargs):
        call_count["count"] += 1
        return original_single_pair_featurize(*args, **kwargs)

    monkeypatch.setattr(featurizer_mod, "_single_pair_featurize", _tracked_single_pair_featurize)

    many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=True,
        chunk_size=1,
        nan_value=np.nan,
    )
    first_run_call_count = int(call_count["count"])
    assert first_run_call_count == len(pairs)
    assert Path(featurizer_info.cache_db_path(dataset.name)).exists()

    many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=True,
        chunk_size=1,
        nan_value=np.nan,
    )
    assert int(call_count["count"]) == first_run_call_count


def test_pair_feature_cache_lookup_probes_reverse_for_legacy_rows() -> None:
    assert FeaturizationInfo.feature_cache_lookup_keys(("a", "b")) == ("a___b", "b___a")


def test_many_pairs_featurize_with_use_cache_false_does_not_write_pair_feature_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(featurizer_mod, "CACHE_ROOT", tmp_path)
    dataset = build_dummy_dataset("pair_feature_cache_disabled", name_counts_index=True)
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0.0)]

    many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
    )

    assert not Path(featurizer_info.cache_db_path(dataset.name)).exists()
