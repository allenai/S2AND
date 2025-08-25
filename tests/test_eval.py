import unittest
import tempfile
import os
import numpy as np

from s2and.eval import (
    b3_precision_recall_fscore,
    f1_score,
    pairwise_eval,
)
import s2and.shap_utils as shap_utils


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

    def test_f1_score_edges(self):
        self.assertEqual(f1_score(0, 1), 0.0)
        self.assertEqual(f1_score(1, 0), 0.0)
        self.assertAlmostEqual(f1_score(0.5, 0.5), 0.5)


class TestShapIntegration(unittest.TestCase):
    def setUp(self):
        # backup originals
        self._orig_tree = shap_utils.shap.TreeExplainer
        self._orig_summary = shap_utils.shap.summary_plot
        self._orig_plots = getattr(shap_utils.shap, "plots", None)
        self._orig_expl = getattr(shap_utils.shap, "Explanation", None)

        # ensure plots namespace exists for fallback stub
        if not hasattr(shap_utils.shap, "plots"):

            class _Plots: ...

            shap_utils.shap.plots = _Plots()

    def tearDown(self):
        # restore
        shap_utils.shap.TreeExplainer = self._orig_tree
        shap_utils.shap.summary_plot = self._orig_summary
        if self._orig_plots is None:
            delattr(shap_utils.shap, "plots")
        else:
            shap_utils.shap.plots = self._orig_plots
        if self._orig_expl is None:
            if hasattr(shap_utils.shap, "Explanation"):
                delattr(shap_utils.shap, "Explanation")
        else:
            shap_utils.shap.Explanation = self._orig_expl

    # -------------------- pairwise_eval tests --------------------

    def test_pairwise_eval_writes_shap_single(self):
        # Dummy TreeExplainer that returns 2D array SHAP values
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer
        shap_utils.shap.summary_plot = lambda *a, **k: None

        class DummyClf:
            def predict_proba(self, X):
                p = np.zeros((X.shape[0], 2))
                p[:, 1] = 0.5
                return p

        X = np.ones((5, 4))
        y = np.array([0, 1, 0, 1, 0])

        clf = DummyClf()
        with tempfile.TemporaryDirectory() as td:
            _ = pairwise_eval(
                X=X,
                y=y,
                classifier=clf,
                figs_path=td,
                title="Test SHAP Single",
                shap_feature_names=[f"f{i}" for i in range(X.shape[1])],
                skip_shap=False,
            )
            base = "test_shap_single"
            self.assertTrue(os.path.exists(os.path.join(td, base + "_roc.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_pr.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_shap.png")))

    def test_pairwise_eval_writes_shap_nameless(self):
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer
        shap_utils.shap.summary_plot = lambda *a, **k: None

        class DummyClf:
            def predict_proba(self, X):
                p = np.zeros((X.shape[0], 2))
                p[:, 1] = 0.6
                return p

        X = np.ones((6, 3))
        y = np.array([1, 0, 1, 0, 1, 0])
        clf = DummyClf()
        nameless = DummyClf()
        nameless_X = np.ones((6, 2))

        with tempfile.TemporaryDirectory() as td:
            _ = pairwise_eval(
                X=X,
                y=y,
                classifier=clf,
                figs_path=td,
                title="Test SHAP Nameless",
                shap_feature_names=[f"f{i}" for i in range(X.shape[1])],
                nameless_classifier=nameless,
                nameless_X=nameless_X,
                nameless_feature_names=[f"nf{i}" for i in range(nameless_X.shape[1])],
                skip_shap=False,
            )
            base = "test_shap_nameless"
            self.assertTrue(os.path.exists(os.path.join(td, base + "_roc.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_pr.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_shap_0.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_shap_1.png")))

    def test_pairwise_eval_skip_shap(self):
        # ensure we don't create SHAP files when skip_shap=True
        class DummyClf:
            def predict_proba(self, X):
                p = np.zeros((X.shape[0], 2))
                p[:, 1] = 0.4
                return p

        X = np.ones((4, 3))
        y = np.array([0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as td:
            _ = pairwise_eval(
                X=X,
                y=y,
                classifier=DummyClf(),
                figs_path=td,
                title="Skip SHAP",
                shap_feature_names=["a", "b", "c"],
                skip_shap=True,
            )
            base = "skip_shap"
            self.assertTrue(os.path.exists(os.path.join(td, base + "_roc.png")))
            self.assertTrue(os.path.exists(os.path.join(td, base + "_pr.png")))
            self.assertFalse(os.path.exists(os.path.join(td, base + "_shap.png")))

    def test_pairwise_eval_wrapper_unwraps_classifier(self):
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer
        shap_utils.shap.summary_plot = lambda *a, **k: None

        class Inner:
            def predict_proba(self, X):
                p = np.zeros((X.shape[0], 2))
                p[:, 1] = 0.7
                return p

        class Wrapper:
            def __init__(self):
                self.classifier = Inner()

        X = np.ones((5, 2))
        y = np.array([0, 1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as td:
            _ = pairwise_eval(
                X=X,
                y=y,
                classifier=Wrapper(),
                figs_path=td,
                title="Wrapped",
                shap_feature_names=["f0", "f1"],
                skip_shap=False,
            )
            self.assertTrue(os.path.exists(os.path.join(td, "wrapped_roc.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "wrapped_pr.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "wrapped_shap.png")))

    # -------------------- shap_utils.compute_shap_summary_plots tests --------------------

    def test_compute_shap_summary_plots_voting_mean(self):
        # TreeExplainer stub
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer
        shap_utils.shap.summary_plot = lambda *a, **k: None

        # Voting-like classifier (estimators attribute)
        class DummyEstimator:
            pass

        class DummyVoting:
            def __init__(self):
                self.estimators = [DummyEstimator(), DummyEstimator()]

        X = np.ones((3, 4))
        names = [f"f{i}" for i in range(4)]
        with tempfile.TemporaryDirectory() as td:
            outs = shap_utils.compute_shap_summary_plots(
                classifier=DummyVoting(),
                X=X,
                shap_feature_names=names,
                shap_plot_type="dot",
                base_name="vote",
                figs_path=td,
            )
            self.assertEqual(len(outs), 1)
            self.assertTrue(outs[0].endswith("vote_shap_0.png"))
            self.assertTrue(os.path.exists(outs[0]))

    def test_compute_shap_summary_plots_calibrated(self):
        # TreeExplainer stub
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer
        shap_utils.shap.summary_plot = lambda *a, **k: None

        # Calibrated-like: expose fitted base estimator through calibrated_classifiers_
        class Fold:
            def __init__(self):
                self.base_estimator = object()

        class DummyCalibrated:
            def __init__(self):
                self.calibrated_classifiers_ = [Fold()]

        X = np.ones((2, 3))
        names = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as td:
            outs = shap_utils.compute_shap_summary_plots(
                classifier=DummyCalibrated(),
                X=X,
                shap_feature_names=names,
                shap_plot_type="dot",
                base_name="calib",
                figs_path=td,
            )
            self.assertEqual(len(outs), 1)
            self.assertTrue(outs[0].endswith("calib_shap.png"))
            self.assertTrue(os.path.exists(outs[0]))

    def test_safe_summary_plot_fallback_to_beeswarm(self):
        # Force summary_plot to raise; provide beeswarm + Explanation stubs
        shap_utils.shap.summary_plot = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))

        class Explanation:
            def __init__(self, values=None, data=None, feature_names=None):
                self.values = values
                self.data = data
                self.feature_names = feature_names

        shap_utils.shap.Explanation = Explanation

        class Plots:
            @staticmethod
            def beeswarm(*a, **k):
                return None

        shap_utils.shap.plots = Plots

        # TreeExplainer stub
        class DummyExplainer:
            def __init__(self, model):
                pass

            def shap_values(self, X):
                return np.zeros((X.shape[0], X.shape[1]))

        shap_utils.shap.TreeExplainer = DummyExplainer

        X = np.ones((3, 3))
        names = ["x", "y", "z"]
        with tempfile.TemporaryDirectory() as td:
            outs = shap_utils.compute_shap_summary_plots(
                classifier=object(),
                X=X,
                shap_feature_names=names,
                shap_plot_type="dot",
                base_name="fallback",
                figs_path=td,
            )
            self.assertEqual(len(outs), 1)
            self.assertTrue(os.path.exists(outs[0]))


if __name__ == "__main__":
    unittest.main()
