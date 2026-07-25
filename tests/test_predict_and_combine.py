import numpy as np
import pytest

import s2and.model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.model import _predict_and_combine


class FakeClassifier:
    def __init__(self, scale: float):
        self.scale = float(scale)
        self.last_shape: tuple[int, int] | None = None

    def predict_proba(self, features_2d: np.ndarray) -> np.ndarray:
        features_2d = np.asarray(features_2d, dtype=np.float64)
        self.last_shape = tuple(int(v) for v in features_2d.shape)  # type: ignore[assignment]
        col0 = np.sum(features_2d, axis=1) * float(self.scale)
        col1 = 1.0 - col0
        return np.stack([col0, col1], axis=1)


class FakePositiveClassifier:
    prediction_backend = "rust_lightgbm"

    def __init__(self) -> None:
        self.last_shape: tuple[int, int] | None = None
        self.last_dtype: np.dtype | None = None

    def predict_proba_positive(
        self,
        features_2d: np.ndarray,
        *,
        max_rows_per_chunk: int | None = None,
    ) -> np.ndarray:
        assert max_rows_per_chunk is None or max_rows_per_chunk >= 1
        self.last_dtype = features_2d.dtype
        features_2d = np.asarray(features_2d, dtype=np.float64)
        self.last_shape = tuple(int(v) for v in features_2d.shape)  # type: ignore[assignment]
        return np.asarray([0.2, 0.7], dtype=np.float64)[: features_2d.shape[0]]

    def predict_proba(self, features_2d: np.ndarray) -> np.ndarray:
        raise AssertionError("predict_proba should not be called when predict_proba_positive is available")


class NonnegativeClassifier:
    def __init__(self) -> None:
        self.seen_rows: list[np.ndarray] = []

    def predict_proba(self, features_2d: np.ndarray) -> np.ndarray:
        features_2d = np.asarray(features_2d, dtype=np.float64)
        if np.any(features_2d < 0):
            raise ValueError("negative input")
        self.seen_rows.extend(features_2d)
        class0 = features_2d[:, 0] / 10.0
        return np.stack([class0, 1.0 - class0], axis=1)


def test_predict_and_combine_all_predicted_rows():
    classifier = FakeClassifier(scale=0.1)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    labels = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)

    predictions, seconds = _predict_and_combine(
        classifier,
        None,
        features,
        labels,
        None,
        "batch",
    )

    assert classifier.last_shape == features.shape
    assert seconds >= 0.0
    assert np.allclose(predictions, np.sum(features, axis=1) * 0.1)


def test_predict_and_combine_respects_constrained_rows():
    classifier = FakeClassifier(scale=0.01)
    features = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [
            np.nan,
            0.0 - LARGE_INTEGER,
            np.nan,
            1.0 - LARGE_INTEGER,
            np.nan,
        ],
        dtype=np.float64,
    )

    predictions, _ = _predict_and_combine(
        classifier,
        None,
        features,
        labels,
        None,
        "batch",
    )

    assert classifier.last_shape == (3, 3)
    expected = np.zeros(5, dtype=np.float64)
    expected[[0, 2, 4]] = np.sum(features[[0, 2, 4], :], axis=1) * 0.01
    expected[1] = 0.0
    expected[3] = 1.0
    assert np.allclose(predictions, expected)


def test_predict_and_combine_uses_native_positive_probability_fast_path():
    classifier = FakePositiveClassifier()
    features = np.asarray([[1.0], [2.0]], dtype=np.float64)
    labels = np.asarray([np.nan, np.nan], dtype=np.float64)

    predictions, seconds = _predict_and_combine(
        classifier,
        None,
        features,
        labels,
        None,
        "batch",
    )

    assert classifier.last_shape == features.shape
    assert seconds >= 0.0
    assert np.allclose(predictions, [0.8, 0.3])


def test_predict_and_combine_preserves_float32_for_native_scorer():
    classifier = FakePositiveClassifier()
    features = np.asarray([[1.0], [2.0]], dtype=np.float32)
    labels = np.asarray([np.nan, np.nan], dtype=np.float64)

    predictions, _ = _predict_and_combine(classifier, None, features, labels, None, "batch")

    assert classifier.last_dtype == np.dtype(np.float32)
    assert np.allclose(predictions, [0.8, 0.3])


def test_predict_and_combine_averages_nameless_classifier():
    classifier = FakeClassifier(scale=0.1)
    nameless_classifier = FakeClassifier(scale=0.2)
    features = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    nameless_features = np.asarray([[10.0], [20.0], [30.0]], dtype=np.float64)
    labels = np.asarray([np.nan, np.nan, 0.0 - LARGE_INTEGER], dtype=np.float64)

    predictions, _ = _predict_and_combine(
        classifier,
        nameless_classifier,
        features,
        labels,
        nameless_features,
        "batch",
    )

    assert classifier.last_shape == (2, 2)
    assert nameless_classifier.last_shape == (2, 1)
    expected = np.zeros(3, dtype=np.float64)
    main_pred = np.sum(features[:2, :], axis=1) * 0.1
    nl_pred = np.sum(nameless_features[:2, :], axis=1) * 0.2
    expected[:2] = (main_pred + nl_pred) / 2.0
    expected[2] = 0.0
    assert np.allclose(predictions, expected)


def test_large_constrained_batch_never_sends_sentinel_rows_to_classifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_module, "_PREDICT_FEATURE_COPY_MAX_BYTES", 16)
    classifier = NonnegativeClassifier()
    nameless_classifier = NonnegativeClassifier()
    features = np.asarray(
        [
            [1.0, 2.0],
            [-LARGE_INTEGER, -LARGE_INTEGER],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )
    nameless_features = np.asarray(
        [
            [2.0],
            [-LARGE_INTEGER],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([np.nan, -LARGE_INTEGER, np.nan, np.nan], dtype=np.float64)

    predictions, _ = _predict_and_combine(
        classifier,
        nameless_classifier,
        features,
        labels,
        nameless_features,
        "batch",
    )

    assert np.array_equal(np.asarray(classifier.seen_rows), features[[0, 2, 3]])
    assert np.array_equal(np.asarray(nameless_classifier.seen_rows), nameless_features[[0, 2, 3]])
    assert np.allclose(predictions, [0.15, 0.0, 0.35, 0.55])
