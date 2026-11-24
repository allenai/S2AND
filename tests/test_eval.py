import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.shap_utils as shap_utils
from s2and.eval import (
    b3_precision_recall_fscore,
    cluster_precision_recall_fscore,
    f1_score,
    pairwise_eval,
    pairwise_precision_recall_fscore,
    pairwise_probability_metrics,
)


def test_pairwise_probability_metrics_match_hand_calculated_ensemble() -> None:
    """Averaging once and predicting negative at exactly 0.5 produce known metrics."""
    metrics, probabilities = pairwise_probability_metrics(
        np.array([0, 1, 0, 1]),
        np.array([0.0, 0.6, 0.6, 1.0]),
        np.array([0.2, 0.4, 0.8, 0.6]),
        include_average_precision=True,
    )

    np.testing.assert_allclose(probabilities, [0.1, 0.5, 0.7, 0.8], rtol=0, atol=1e-15)
    # One true/false positive and one true/false negative give every macro metric 1/2.
    # Three of the four positive-negative pairs are ordered correctly (AUROC 3/4).
    expected = {"rows": 4, "auroc": 0.75, "macro_f1": 0.5, "macro_precision": 0.5, "macro_recall": 0.5}
    # Positive labels rank first and third.
    expected["average_precision"] = 5 / 6
    assert metrics == pytest.approx(expected)


@pytest.mark.parametrize(
    ("labels", "main", "nameless", "message"),
    [
        ([1, 1], [0.8, 0.9], [0.7, 0.8], "requires nonempty labels with both classes"),
        ([0, 1], [0.1, 0.9], [0.2], "must have equal shape"),
    ],
    ids=["single-class", "missing-nameless-score"],
)
def test_pairwise_probability_metrics_reject_unevaluable_populations(
    labels: list[int], main: list[float], nameless: list[float], message: str
) -> None:
    """Incomplete prediction populations must not produce a release metric report."""
    with pytest.raises(ValueError, match=message):
        pairwise_probability_metrics(np.array(labels), np.array(main), np.array(nameless))


def test_pairwise_eval_restores_plot_style_after_save_failure(tmp_path, monkeypatch) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    class DummyClf:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.2, 0.8]])

    def savefig(*args, **kwargs):
        assert mpl.rcParams["axes.labelsize"] == 18.0
        raise OSError("render failure")

    monkeypatch.setattr(plt, "savefig", savefig)
    with mpl.rc_context({"axes.labelsize": 7, "axes.facecolor": "pink", "font.size": 8}):
        before = mpl.rcParams.copy()
        try:
            with pytest.raises(OSError, match="render failure"):
                pairwise_eval(
                    np.ones((2, 1)), np.array([0, 1]), DummyClf(), str(tmp_path), "Style", ["x"], skip_shap=True
                )
            assert mpl.rcParams == before
        finally:
            plt.close("all")


class TestB3AndF1(unittest.TestCase):
    def test_b3_precision_recall_score(self):
        cluster_homogeneity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5, 6, 7], "3": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_1[0], 0.599)
        self.assertAlmostEqual(cluster_homogeneity_1[1], 0.695)
        self.assertAlmostEqual(cluster_homogeneity_1[2], 0.643)

        cluster_homogeneity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_2[0], 0.694)
        self.assertAlmostEqual(cluster_homogeneity_2[1], 0.695)
        self.assertAlmostEqual(cluster_homogeneity_2[2], 0.695)

        size_v_quantity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4, 5], "2": [6], "3": [7], "4": [8], "5": [9], "6": [10], "7": [11], "8": [12], "9": [13]},
        )
        self.assertAlmostEqual(size_v_quantity_1[0], 1.0)
        self.assertAlmostEqual(size_v_quantity_1[1], 0.692)
        self.assertAlmostEqual(size_v_quantity_1[2], 0.818)

        size_v_quantity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13]},
        )
        self.assertAlmostEqual(size_v_quantity_2[0], 1.0)
        self.assertAlmostEqual(size_v_quantity_2[1], 0.877)
        self.assertAlmostEqual(size_v_quantity_2[2], 0.934)

        self.assertEqual(f1_score(0, 1), 0.0)
        self.assertEqual(f1_score(1, 0), 0.0)
        self.assertAlmostEqual(f1_score(0.5, 0.5), 0.5)


def test_cluster_metrics_require_partitions_of_the_same_population() -> None:
    cases = (
        (
            "b3-truth-within",
            b3_precision_recall_fscore,
            {"t1": ["s1", "s1"], "t2": ["s2"]},
            {"p1": ["s1"], "p2": ["s2"]},
            "Ground-truth",
        ),
        (
            "cluster-truth-across",
            cluster_precision_recall_fscore,
            {"t1": ["s1"], "t2": ["s1", "s2"]},
            {"p1": ["s1"], "p2": ["s2"]},
            "Ground-truth",
        ),
        (
            "pairwise-prediction-within",
            lambda true_clus, pred_clus: pairwise_precision_recall_fscore(
                true_clus,
                pred_clus,
                {"block": ["s1", "s2"]},
            ),
            {"t1": ["s1"], "t2": ["s2"]},
            {"p1": ["s1", "s1"], "p2": ["s2"]},
            "Predicted",
        ),
        (
            "b3-prediction-across",
            b3_precision_recall_fscore,
            {"t1": ["s1"], "t2": ["s2"]},
            {"p1": ["s1"], "p2": ["s1", "s2"]},
            "Predicted",
        ),
    )
    for _case_id, metric, true_clus, pred_clus, label in cases:
        with pytest.raises(ValueError, match=f"{label} clustering must be a partition"):
            metric(true_clus, pred_clus)

    metrics = (
        ("b3", b3_precision_recall_fscore),
        ("cluster", cluster_precision_recall_fscore),
        (
            "pairwise",
            lambda true_clus, pred_clus: pairwise_precision_recall_fscore(
                true_clus,
                pred_clus,
                {"block": ["s1"]},
            ),
        ),
    )
    for _case_id, metric in metrics:
        with pytest.raises(ValueError, match="Predictions do not cover all the signatures"):
            metric({"t1": ["s1"]}, {"p1": ["s2"]})


class TestShapIntegration(unittest.TestCase):
    def setUp(self):
        class DummyExplainer:
            def __init__(self, model):
                del model

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        self._orig_shap_module = shap_utils._SHAP_MODULE
        shap_utils._SHAP_MODULE = SimpleNamespace(
            TreeExplainer=DummyExplainer,
            summary_plot=lambda *args, **kwargs: None,
        )

    def tearDown(self):
        shap_utils._SHAP_MODULE = self._orig_shap_module

    def test_pairwise_eval_validates_fitted_feature_names(self):
        import pandas as pd
        from lightgbm import LGBMClassifier

        rng = np.random.default_rng(7)
        X_train = pd.DataFrame(rng.random((20, 3)), columns=["f0", "f1", "f2"])
        y_train = rng.integers(0, 2, size=20)
        classifier = LGBMClassifier(n_estimators=8, random_state=7, verbosity=-1)
        classifier.fit(X_train, y_train)

        X = rng.random((4, 3))
        y = np.array([0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as td:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _ = pairwise_eval(
                    X=X,
                    y=y,
                    classifier=classifier,
                    figs_path=td,
                    title="Feature Schema",
                    shap_feature_names=["f0", "f1", "f2"],
                    skip_shap=True,
                )

        leaked = [w for w in caught if "X does not have valid feature names" in str(w.message)]
        self.assertEqual(leaked, [])

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "feature names do not match fitted schema"):
                pairwise_eval(
                    X=X,
                    y=y,
                    classifier=classifier,
                    figs_path=td,
                    title="Feature Schema Mismatch",
                    shap_feature_names=["f1", "f0", "f2"],
                    skip_shap=True,
                )

    def test_safe_summary_plot_falls_back_to_beeswarm(self):
        fake_shap = cast(Any, shap_utils._SHAP_MODULE)
        beeswarm_calls = []

        def fail_summary_plot(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("summary plot failed")

        class Explanation:
            def __init__(self, values=None, data=None, feature_names=None):
                self.values = values
                self.data = data
                self.feature_names = feature_names

        class Plots:
            @staticmethod
            def beeswarm(explanation, **kwargs):
                beeswarm_calls.append((explanation, kwargs))

        fake_shap.summary_plot = fail_summary_plot
        fake_shap.Explanation = Explanation
        fake_shap.plots = Plots

        X = np.ones((3, 3))
        with tempfile.TemporaryDirectory() as td:
            outputs = shap_utils.compute_shap_summary_plots(
                classifier=object(),
                X=X,
                shap_feature_names=["x", "y", "z"],
                shap_plot_type="dot",
                base_name="fallback",
                figs_path=td,
            )
            assert os.path.exists(outputs[0])

        assert len(beeswarm_calls) == 1
        assert outputs[0].endswith("fallback_shap.png")


def _small_binary_classification() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic tiny binary classification fixture."""

    from sklearn.datasets import make_classification

    return make_classification(
        n_samples=30,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )


def test_shap_values_select_random_forest_class_index():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _small_binary_classification()
    classifier = RandomForestClassifier(n_estimators=4, random_state=7).fit(X, y)
    raw_values = shap_utils._get_shap_module().TreeExplainer(classifier).shap_values(X[:3])
    assert raw_values.shape == (3, 4, 2)

    values = shap_utils._shap_values_for_tree_model(classifier, X[:3], class_index=1)

    np.testing.assert_allclose(values, raw_values[:, :, 1])


def test_shap_values_reject_invalid_shape_and_class_index(monkeypatch):
    class FakeExplainer:
        output = np.zeros((2, 3, 4, 5))

        def __init__(self, model):
            del model

        def shap_values(self, X):
            del X
            return self.output

    monkeypatch.setattr(shap_utils, "_SHAP_MODULE", SimpleNamespace(TreeExplainer=FakeExplainer))
    with pytest.raises(ValueError, match="unsupported shape"):
        shap_utils._shap_values_for_tree_model(object(), np.ones((2, 3)))

    FakeExplainer.output = np.zeros((2, 3, 2))
    with pytest.raises(ValueError, match="class_index 2 is out of range"):
        shap_utils._shap_values_for_tree_model(object(), np.ones((2, 3)), class_index=2)


def test_shap_values_reject_calibrated_voting_and_stacking_classifiers():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = _small_binary_classification()
    estimators = [
        ("shallow", DecisionTreeClassifier(max_depth=2, random_state=7)),
        ("deep", DecisionTreeClassifier(max_depth=3, random_state=8)),
    ]
    unsupported = [
        CalibratedClassifierCV(DecisionTreeClassifier(max_depth=2, random_state=7), cv=2).fit(X, y),
        VotingClassifier(estimators).fit(X, y),
        StackingClassifier(estimators, cv=2).fit(X, y),
    ]

    for classifier in unsupported:
        with pytest.raises(TypeError, match=rf"{type(classifier).__name__} is unsupported.*skip_shap=True"):
            shap_utils._shap_values_for_tree_model(classifier, X[:3])


if __name__ == "__main__":
    unittest.main()


def test_pairwise_eval_writes_only_requested_plots(tmp_path, monkeypatch):
    """Exercise scorer unwrapping and SHAP routing with one plotting fixture."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    shapes = []

    def shap_values(matrix):
        shapes.append(matrix.shape)
        return np.zeros_like(matrix)

    monkeypatch.setattr(
        shap_utils,
        "_SHAP_MODULE",
        SimpleNamespace(
            TreeExplainer=lambda model: SimpleNamespace(shap_values=shap_values),
            summary_plot=lambda *args, **kwargs: None,
        ),
    )
    scorer = SimpleNamespace(predict_proba=lambda matrix: np.tile([0.4, 0.6], (len(matrix), 1)))
    observed_styles = []
    real_savefig = plt.savefig

    def savefig(*args, **kwargs):
        observed_styles.append(mpl.rcParams["axes.labelsize"])
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", savefig)
    original_style = mpl.rcParams.copy()
    classifier = SimpleNamespace(classifier=scorer)
    features = np.ones((4, 3))
    labels = np.array([0, 1, 0, 1])
    try:
        pairwise_eval(
            features,
            labels,
            classifier,
            str(tmp_path),
            "Plots",
            ["a", "b", "c"],
            nameless_classifier=classifier,
            nameless_X=np.ones((4, 2)),
            nameless_feature_names=["d", "e"],
        )
        assert {path.name for path in tmp_path.glob("*.png")} == {
            "plots_roc.png",
            "plots_pr.png",
            "plots_shap_0.png",
            "plots_shap_1.png",
        }
        assert shapes == [(4, 3), (4, 2)]
        assert observed_styles[:2] == [18.0, 18.0]
        assert mpl.rcParams == original_style

        pairwise_eval(features, labels, scorer, str(tmp_path), "Single", ["a", "b", "c"])
        assert {path.name for path in tmp_path.glob("single*.png")} == {
            "single_roc.png",
            "single_pr.png",
            "single_shap.png",
        }
        assert shapes == [(4, 3), (4, 2), (4, 3)]
    finally:
        plt.close("all")
