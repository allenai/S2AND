import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.model import _predict_and_combine
from s2and.production_model import NativeLightGBMBinaryClassifier


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


@pytest.fixture(scope="module")
def trained_scorers(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[tuple[LGBMClassifier, NativeLightGBMBinaryClassifier], ...]:
    """Train distinct tiny models and load their actual native release scorers."""
    training = np.linspace(-4, 4, 16).reshape(-1, 1)
    directory = tmp_path_factory.mktemp("combined_native_scorers")
    scorers = []
    for index, labels in enumerate((training[:, 0] > 0, abs(training[:, 0]) > 2)):
        python_scorer = LGBMClassifier(
            n_estimators=4, num_leaves=3, min_child_samples=1, random_state=7, n_jobs=1, verbosity=-1
        ).fit(training, labels.astype(int))
        model_path = directory / f"model_{index}.txt"
        python_scorer.booster_.save_model(str(model_path))
        scorers.append((python_scorer, NativeLightGBMBinaryClassifier(model_path, n_jobs=1)))
    return tuple(scorers)


@pytest.mark.parametrize(
    "main_backend,nameless_backend",
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    ids=["python-python", "python-native", "native-python", "native-native"],
)
@pytest.mark.parametrize("copy_budget", [1, 1_000_000], ids=["chunked", "one-batch"])
def test_real_native_and_python_scorers_combine_distances_in_original_row_order(
    trained_scorers: tuple[tuple[LGBMClassifier, NativeLightGBMBinaryClassifier], ...],
    monkeypatch: pytest.MonkeyPatch,
    main_backend: int,
    nameless_backend: int,
    copy_budget: int,
) -> None:
    """Mixing scorer APIs must preserve class-0 averaging and supervised rows."""
    monkeypatch.setattr(model_module, "_PREDICT_FEATURE_COPY_MAX_BYTES", copy_budget)
    main_features = np.asarray([[-3.0], [-2.0], [0.5], [1.0], [3.0]], dtype=np.float32)
    nameless_features = np.asarray([[0.0], [-3.0], [3.0], [-1.0], [0.0]], dtype=np.float32)
    labels = np.asarray([np.nan, -LARGE_INTEGER, np.nan, 1 - LARGE_INTEGER, np.nan], dtype=np.float64)
    main_python, _ = trained_scorers[0]
    nameless_python, _ = trained_scorers[1]
    expected = (
        main_python.predict_proba(main_features)[:, 0] + nameless_python.predict_proba(nameless_features)[:, 0]
    ) / 2
    expected[[1, 3]] = [0.0, 1.0]
    assert np.ptp(expected[[0, 2, 4]]) > 0.1

    actual, _ = _predict_and_combine(
        trained_scorers[0][main_backend],
        trained_scorers[1][nameless_backend],
        main_features,
        labels,
        nameless_features,
        "native-combination",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(actual[[1, 3]], [0.0, 1.0])
