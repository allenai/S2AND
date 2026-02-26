from __future__ import annotations

import numpy as np
import pytest

from s2and import feature_port
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize


def _dummy_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        load_name_counts=False,
        n_jobs=1,
    )


def test_python_backend_pair_featurization_makes_zero_rust_calls(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "python")
    dataset = _dummy_dataset("dummy_runtime_policy_python")
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0), ("0", "2", 0)]

    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected rust call")),
    )

    features, labels, _ = many_pairs_featurize(
        pairs,
        dataset,
        featurizer_info,
        n_jobs=1,
        use_cache=False,
        chunk_size=1,
        nan_value=np.nan,
    )

    assert features.shape[0] == len(pairs)
    assert labels.shape[0] == len(pairs)


def test_rust_backend_pair_featurization_fails_fast_on_rust_error(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    dataset = _dummy_dataset("dummy_runtime_policy_rust")
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    pairs = [("0", "1", 0), ("0", "2", 0)]

    class FailingRustFeaturizer:
        def signature_ids(self):
            return []

        def featurize_pairs_matrix_indexed(self, _pairs, _indices, _threads, _nan):
            raise RuntimeError("synthetic rust batch failure")

        def featurize_pairs_matrix(self, *_args, **_kwargs):
            raise RuntimeError("synthetic rust batch failure")

        def featurize_pairs(self, _pairs, num_threads=None):
            del num_threads
            raise RuntimeError("synthetic rust batch failure")

    monkeypatch.setattr(feature_port, "s2and_rust", object())
    monkeypatch.setattr(feature_port, "_get_rust_featurizer", lambda *_args, **_kwargs: FailingRustFeaturizer())


    with pytest.raises(RuntimeError, match="strict rust backend"):
        many_pairs_featurize(
            pairs,
            dataset,
            featurizer_info,
            n_jobs=1,
            use_cache=False,
            chunk_size=1,
            nan_value=np.nan,
        )


