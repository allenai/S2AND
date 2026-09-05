"""Require exact scored features, classifier batches, and stored distances."""

from typing import Any

import numpy as np
import pytest

import s2and.model as model_module
from s2and.feature_port import _get_rust_featurizer
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset


@pytest.fixture(scope="module")
def native_features(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Build a real Arrow featurizer with diverse paper and author metadata."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("S2AND_BACKEND", "python")
        dataset = build_dummy_dataset("sparse_features", name_counts_index=True)
        arrow = build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("sparse_features"))
    return _get_rust_featurizer(arrow)


@pytest.mark.parametrize("rows", [[], [0], [1, 4], [0, 1, 2, 3, 4]])
@pytest.mark.parametrize("main,nameless", [([3, 1, 3, 0], [1, 0]), ([0, 1], None), ([], [])])
def test_sparse_feature_rows_match_dense_bits(
    native_features: Any, rows: list[int], main: list[int], nameless: list[int] | None
) -> None:
    """Retain bit patterns and requested column order at a triangle row boundary."""
    kwargs = dict(
        featurizer=native_features,
        start_offset=2,
        max_pairs=5,
        main_indices=main,
        nameless_indices=nameless,
        num_threads=10,
    )
    dense = model_module._build_block_feature_matrices_indexed_rust([3, 0, 2, 1, 4], **kwargs)
    indices = np.asarray(rows, dtype=np.intp)
    sparse = model_module._build_block_feature_matrices_indexed_rust([3, 0, 2, 1, 4], **kwargs, scored_rows=indices)
    for actual, expected in zip(sparse, dense, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual.shape == expected.shape
            assert actual.flags.c_contiguous
            np.testing.assert_array_equal(
                np.ascontiguousarray(actual[indices]).view(np.uint64),
                np.ascontiguousarray(expected[indices]).view(np.uint64),
            )


class RecordingClassifier:
    """Make both feature content and call boundaries observable."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities that depend on each row's place in its call."""
        self.calls.append(features.copy())
        distances = 0.25 + np.arange(len(features)) / (2 * max(1, len(features)))
        return np.column_stack((distances, 1 - distances))


@pytest.mark.parametrize("chunk_size", [1, 5, 21])
@pytest.mark.parametrize("fastcluster", [False, True])
@pytest.mark.parametrize("constraints", [False, True])
def test_sparse_features_preserve_distances_and_classifier_calls(
    native_features: Any, monkeypatch: pytest.MonkeyPatch, chunk_size: int, fastcluster: bool, constraints: bool
) -> None:
    """Compare the dense and sparse paths with overrides and both storage forms."""
    ids = [str(index) for index in range(7)]
    pairs = [(left, right) for index, left in enumerate(ids) for right in ids[index + 1 :]]
    overrides = {pair: float(index % 2) for index, pair in enumerate(pairs) if index not in (2, 5, 13)}
    overrides[(pairs[0][1], pairs[0][0])] = 0.75
    info = FeaturizationInfo(features_to_use=["year_diff", "name_similarity"])
    main, nameless = RecordingClassifier(), RecordingClassifier()
    clusterer = Clusterer(
        featurizer_info=info,
        classifier=main,
        nameless_featurizer_info=info,
        nameless_classifier=nameless,
        cluster_model=FastCluster(linkage="average") if fastcluster else object(),
        n_jobs=10,
        batch_size=chunk_size,
        use_default_constraints_as_supervision=constraints,
    )
    kwargs = dict(
        signature_index_by_id={value: index for index, value in enumerate(ids)}, partial_supervision=overrides
    )
    sparse = clusterer._make_distance_matrices_from_verified_rust_featurizer({"block": ids}, native_features, **kwargs)
    expected_calls = [main.calls[:], nameless.calls[:]]
    main.calls.clear()
    nameless.calls.clear()
    original = model_module._build_block_feature_matrices_indexed_rust

    def force_dense(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("scored_rows", None)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "_build_block_feature_matrices_indexed_rust", force_dense)
    original_combine = model_module._predict_and_combine

    def combine_dense(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("compact_rows", None)
        return original_combine(*args, **kwargs)

    monkeypatch.setattr(model_module, "_predict_and_combine", combine_dense)
    dense = clusterer._make_distance_matrices_from_verified_rust_featurizer({"block": ids}, native_features, **kwargs)
    np.testing.assert_array_equal(sparse["block"].view(np.uint8), dense["block"].view(np.uint8))
    for actual, expected in zip((main.calls, nameless.calls), expected_calls, strict=True):
        assert len(actual) == len(expected)
        for actual_batch, expected_batch in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_batch.view(np.uint64), expected_batch.view(np.uint64))


@pytest.mark.parametrize("rows", [[], [0], [1, 4], [0, 1, 2, 3, 4]])
def test_compact_feature_rows_match_dense_bits(native_features: Any, rows: list[int]) -> None:
    """Compact projection preserves duplicate columns and row order exactly."""
    kwargs = dict(
        featurizer=native_features,
        start_offset=2,
        max_pairs=5,
        main_indices=[3, 1, 3, 0],
        nameless_indices=[1, 0],
        num_threads=10,
    )
    dense = model_module._build_block_feature_matrices_indexed_rust([3, 0, 2, 1, 4], **kwargs)
    indices = np.asarray(rows, dtype=np.intp)
    compact = model_module._build_block_feature_matrices_indexed_rust(
        [3, 0, 2, 1, 4], **kwargs, scored_rows=indices, compact=True
    )
    for actual, expected in zip(compact, dense, strict=True):
        np.testing.assert_array_equal(actual.view(np.uint64), expected[indices].view(np.uint64))


@pytest.mark.parametrize("copy_budget", [1, 96, 100000])
@pytest.mark.parametrize("rows", [[], [0], [0, 3, 5], list(range(7))])
@pytest.mark.parametrize("with_nameless", [False, True])
def test_compact_scoring_preserves_batches_and_owned_inputs(
    monkeypatch: pytest.MonkeyPatch, copy_budget: int, rows: list[int], with_nameless: bool
) -> None:
    """Check bits, call boundaries, writable owned inputs and mutation isolation."""
    monkeypatch.setattr(model_module, "_PREDICT_FEATURE_COPY_MAX_BYTES", copy_budget)
    matrix = np.arange(28, dtype=np.float64).reshape(7, 4)
    matrix[0] = [-0.0, np.nan, np.inf, -np.inf]
    labels = np.full(7, -model_module.LARGE_INTEGER, dtype=np.float64)
    indices = np.asarray(rows, dtype=np.intp)
    labels[indices] = np.nan

    class MutatingClassifier(RecordingClassifier):
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.flags.c_contiguous
            assert features.flags.writeable
            assert features.flags.owndata
            result = super().predict_proba(features)
            features[:] = 42.0
            return result

    observed = []
    for compact in (False, True):
        main = MutatingClassifier()
        nameless = MutatingClassifier() if with_nameless else None
        features = matrix[indices].copy() if compact else matrix.copy()
        nl_features = features.copy() if with_nameless else None
        result, _ = model_module._predict_and_combine(
            main,
            nameless,
            features,
            labels,
            nl_features,
            "test",
            num_threads=10,
            compact_rows=indices if compact else None,
        )
        observed.append((result, main.calls, None if nameless is None else nameless.calls))
        if len(rows) < len(labels):
            np.testing.assert_array_equal(
                features.view(np.uint64), (matrix[indices] if compact else matrix).view(np.uint64)
            )
    for actual, expected in zip(observed[0], observed[1], strict=True):
        if actual is None:
            assert expected is None
        elif isinstance(actual, list):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        else:
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
