from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest

import s2and.featurizer as featurizer_mod
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import build_dummy_dataset


@pytest.fixture(autouse=True)
def _clear_pair_feature_cache_state() -> None:
    with featurizer_mod._CACHED_FEATURES_LOCK:
        featurizer_mod.CACHED_FEATURES.clear()
    yield
    with featurizer_mod._CACHED_FEATURES_LOCK:
        featurizer_mod.CACHED_FEATURES.clear()


def _patch_pair_cache_paths(featurizer_info: FeaturizationInfo, cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_db_path = cache_dir / featurizer_mod.PAIR_FEATURE_CACHE_DB_FILENAME
    legacy_cache_path = cache_dir / "all_features.json"
    featurizer_info.cache_directory = lambda _dataset_name: str(cache_dir)  # type: ignore[method-assign]
    featurizer_info.cache_db_path = lambda _dataset_name: str(cache_db_path)  # type: ignore[method-assign]
    featurizer_info.cache_storage_key = lambda _dataset_name: str(cache_db_path)  # type: ignore[method-assign]
    featurizer_info.cache_file_path = lambda _dataset_name: str(legacy_cache_path)  # type: ignore[method-assign]
    return cache_db_path, legacy_cache_path


def test_write_cache_persists_to_sqlite_and_load_cache_round_trips(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path, legacy_cache_path = _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    feature_vector = np.arange(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    cached_features = featurizer_info._fresh_cache_payload()
    cached_features["features"]["a___b"] = feature_vector
    cached_features["__new_features__"]["a___b"] = feature_vector

    featurizer_info.write_cache(cached_features, "dataset", incremental=True)

    loaded = featurizer_info.load_cache("dataset")
    np.testing.assert_array_equal(loaded["features"]["a___b"], feature_vector)
    assert cache_db_path.exists()
    assert not legacy_cache_path.exists()
    assert loaded["__cache_backend__"] == "sqlite"


def test_load_cache_reads_legacy_json_and_migrates_to_sqlite_on_write(tmp_path: Path) -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    cache_db_path, legacy_cache_path = _patch_pair_cache_paths(featurizer_info, tmp_path / "pair_cache")
    legacy_feature = np.arange(featurizer_mod.NUM_FEATURES, dtype=np.float64)
    new_feature = legacy_feature + 100.0
    legacy_cache_path.write_bytes(
        orjson.dumps(
            {
                "features": {"legacy___pair": legacy_feature.tolist()},
                "features_to_use": featurizer_info.features_to_use,
            }
        )
    )

    loaded = featurizer_info.load_cache("dataset")
    assert loaded["__cache_backend__"] == "legacy_json"
    loaded["features"]["new___pair"] = new_feature
    loaded["__new_features__"]["new___pair"] = new_feature

    featurizer_info.write_cache(loaded, "dataset", incremental=True)

    migrated = featurizer_info.load_cache("dataset")
    assert cache_db_path.exists()
    assert set(migrated["features"].keys()) == {"legacy___pair", "new___pair"}
    np.testing.assert_array_equal(migrated["features"]["legacy___pair"], legacy_feature)
    np.testing.assert_array_equal(migrated["features"]["new___pair"], new_feature)
    assert migrated["__cache_backend__"] == "sqlite"


def test_many_pairs_featurize_reuses_persisted_pair_feature_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(featurizer_mod, "CACHE_ROOT", tmp_path)
    dataset = build_dummy_dataset("pair_feature_cache_roundtrip", load_name_counts=True)
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

    with featurizer_mod._CACHED_FEATURES_LOCK:
        featurizer_mod.CACHED_FEATURES.clear()

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


def test_many_pairs_featurize_with_use_cache_false_does_not_write_pair_feature_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setattr(featurizer_mod, "CACHE_ROOT", tmp_path)
    dataset = build_dummy_dataset("pair_feature_cache_disabled", load_name_counts=True)
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
    assert not Path(featurizer_info.cache_file_path(dataset.name)).exists()
