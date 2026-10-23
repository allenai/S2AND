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
    dataset = build_dummy_dataset("sparse_features", name_counts_index=True)
    arrow = build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("sparse_features"))
    return _get_rust_featurizer(arrow)


@pytest.mark.parametrize(
    ("rows", "main", "nameless", "compact"),
    [(rows, [3, 1, 3, 0], [1, 0], compact) for rows in ([], [1, 4], [0, 1, 2, 3, 4]) for compact in (False, True)]
    + [([1, 4], [0, 1], None, False), ([1, 4], [], [], False)],
)
def test_sparse_feature_rows_match_dense_bits(
    native_features: Any, rows: list[int], main: list[int], nameless: list[int] | None, compact: bool
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
    sparse = model_module._build_block_feature_matrices_indexed_rust(
        [3, 0, 2, 1, 4], **kwargs, scored_rows=indices, compact=compact
    )
    for actual, expected in zip(sparse, dense, strict=True):
        if expected is None:
            assert actual is None
        else:
            assert actual.shape == (expected[indices].shape if compact else expected.shape)
            assert actual.flags.c_contiguous
            np.testing.assert_array_equal(
                np.ascontiguousarray(actual if compact else actual[indices]).view(np.uint64),
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


@pytest.mark.parametrize(
    ("chunk_size", "fastcluster", "constraints"),
    [(5, False, False), (5, False, True), (5, True, False), (5, True, True), (1, True, True), (21, False, True)],
)
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


@pytest.fixture(scope="module")
def native_block_scorers(native_features: Any, tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    """Two tiny native LightGBM scorers trained on the test block's own feature rows.

    Splits then fall inside the real value ranges, so scores differ across pairs.
    """
    from lightgbm import LGBMClassifier

    from s2and.production_model import NativeLightGBMBinaryClassifier
    from s2and.rust_calls import build_block_upper_triangle_feature_matrix_indexed_rust

    info = FeaturizationInfo(features_to_use=["year_diff", "name_similarity"])
    rows = build_block_upper_triangle_feature_matrix_indexed_rust(
        list(range(7)),
        featurizer=native_features,
        selected_indices=list(info.selected_feature_indices()),
        num_threads=1,
        nan_value=np.nan,
    )
    rng = np.random.default_rng(11)
    directory = tmp_path_factory.mktemp("native_block_scorers")
    scorers = []
    for index in range(2):
        labels = rng.integers(0, 2, size=len(rows))
        labels[0], labels[1] = 0, 1
        python_scorer = LGBMClassifier(
            n_estimators=8, num_leaves=4, min_child_samples=2, random_state=index, n_jobs=1, verbosity=-1
        ).fit(rows, labels)
        model_path = directory / f"model_{index}.txt"
        python_scorer.booster_.save_model(str(model_path))
        scorers.append(NativeLightGBMBinaryClassifier(model_path, n_jobs=1))
    return scorers[0], scorers[1]


@pytest.mark.parametrize(
    ("chunk_size", "constraints", "overrides", "with_nameless"),
    [
        (5, True, True, True),
        (21, True, True, True),
        (5, False, False, True),
        (3, True, False, False),
        (50, False, True, False),
    ],
)
def test_fused_block_scoring_matches_feature_matrix_path(
    native_features: Any,
    native_block_scorers: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
    constraints: bool,
    overrides: bool,
    with_nameless: bool,
) -> None:
    """Fused tiles must give the distance matrix the batch path gives, bit for bit."""
    ids = [str(index) for index in range(7)]
    pairs = [(left, right) for index, left in enumerate(ids) for right in ids[index + 1 :]]
    supervision = (
        {pair: float(index % 2) for index, pair in enumerate(pairs) if index not in (2, 5, 13)} if overrides else {}
    )
    info = FeaturizationInfo(features_to_use=["year_diff", "name_similarity"])
    main, nameless = native_block_scorers
    clusterer = Clusterer(
        featurizer_info=info,
        classifier=main,
        nameless_featurizer_info=info if with_nameless else None,
        nameless_classifier=nameless if with_nameless else None,
        cluster_model=FastCluster(linkage="average"),
        n_jobs=4,
        batch_size=chunk_size,
        use_default_constraints_as_supervision=constraints,
    )
    kwargs = dict(
        signature_index_by_id={value: index for index, value in enumerate(ids)}, partial_supervision=supervision
    )
    assert model_module._native_block_scorers(main, nameless if with_nameless else None) is not None
    fused = clusterer._make_distance_matrices_from_verified_rust_featurizer({"block": ids}, native_features, **kwargs)
    monkeypatch.setattr(model_module, "_FUSED_BLOCK_SCORING", False)
    batch = clusterer._make_distance_matrices_from_verified_rust_featurizer({"block": ids}, native_features, **kwargs)
    assert fused["block"].dtype == batch["block"].dtype == np.float64
    np.testing.assert_array_equal(fused["block"].view(np.uint64), batch["block"].view(np.uint64))
    # With constraints and overrides every pair is labeled; otherwise the scores
    # vary across pairs, so the comparison is not of constants.
    if not (constraints and overrides):
        assert len(np.unique(batch["block"])) > 3


@pytest.mark.parametrize(
    ("copy_budget", "rows", "with_nameless"),
    [(budget, [0, 3, 5], nameless) for budget in (1, 96, 100000) for nameless in (False, True)]
    + [(1, [], True), (96, [0], True), (100000, list(range(7)), True)],
)
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
