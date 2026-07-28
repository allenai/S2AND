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
    _shap_values_for_tree_model_preserving_booster_params,
    _write_claims_eval_shap_plots,
    b3_precision_recall_fscore,
    claims_eval,
    cluster_precision_recall_fscore,
    f1_score,
    facet_eval,
    pairwise_eval,
    pairwise_precision_recall_fscore,
)


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


def test_cluster_metrics_reject_non_partition_memberships() -> None:
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


def test_cluster_metrics_reject_unequal_coverage() -> None:
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


def test_facet_eval_includes_zero_homonymity_and_synonymity_buckets() -> None:
    signature = SimpleNamespace(
        paper_id="p1",
        author_info_block="block",
        author_info_full_name="alice smith",
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        author_info_first="Alice",
        author_info_affiliations=[],
        author_info_email=None,
        author_info_coauthors=[],
    )
    paper = SimpleNamespace(
        authors=[SimpleNamespace(author_name="Alice Smith")],
        year=2020,
        has_abstract=False,
        venue="",
        journal_name="",
    )
    dataset = SimpleNamespace(
        get_blocks=lambda: {"block": ["s1"]},
        clusters={"c1": {"signature_ids": ["s1"]}},
        signature_to_cluster_id={"s1": "c1"},
        signatures={"s1": signature},
        papers={"p1": paper},
    )

    result = facet_eval(cast(Any, dataset), {"s1": (0.8, 0.6, 0.7)})

    assert result.homonymity_f1 == {0: [0.7]}
    assert result.synonymity_f1 == {0: [0.7]}


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

    # -------------------- pairwise_eval tests --------------------

    def test_pairwise_eval_writes_shap_single(self):
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


def test_shap_values_for_lightgbm_are_2d():
    from lightgbm import LGBMClassifier

    X, y = _small_binary_classification()
    classifier = LGBMClassifier(n_estimators=4, random_state=7, verbosity=-1).fit(X, y)

    values = shap_utils._shap_values_for_tree_model(classifier, X[:3], class_index=1)

    assert values.shape == (3, 4)


def test_shap_values_for_native_lightgbm_are_2d():
    from lightgbm import LGBMClassifier

    from s2and.production_model import NativeLightGBMBinaryClassifier

    X, y = _small_binary_classification()
    classifier = LGBMClassifier(n_estimators=4, random_state=7, verbosity=-1).fit(X, y)
    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.txt")
        classifier.booster_.save_model(model_path)
        native_classifier = NativeLightGBMBinaryClassifier(model_path)

        values = shap_utils._shap_values_for_tree_model(native_classifier, X[:3], class_index=1)

    assert values.shape == (3, 4)


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


def _build_claims_eval_test_inputs(dists):
    class _Author:
        def __init__(self, author_name):
            self.author_name = author_name

    class _Paper:
        def __init__(self, title):
            self.title = title
            self.authors = [_Author("Test Author")]

    class _Signature:
        def __init__(self, affiliations):
            self.author_info_affiliations = affiliations

    class _Dataset:
        def __init__(self):
            self.mode = "inference"
            self._blocks = {"blk": ["p1___0", "p2___0"]}
            self.papers = {
                "p1": _Paper("Paper 1"),
                "p2": _Paper("Paper 2"),
            }
            self.signatures = {
                "p1___0": _Signature(["Org 1"]),
                "p2___0": _Signature(["Org 2"]),
            }

        def get_blocks(self):
            return self._blocks

    class _Clusterer:
        def __init__(self, dists_value):
            self._dists = dists_value

        def predict(self, _block_dict, _dataset):
            return {"blk_0": ["p1___0", "p2___0"]}, self._dists

    dataset = _Dataset()
    clusterer = _Clusterer(dists)
    claims_pairs = [("p1___0", "p2___0", 1, "blk", "blk")]
    return dataset, clusterer, claims_pairs


def test_claims_eval_skips_distance_dump_when_predict_returns_none():
    dataset, clusterer, claims_pairs = _build_claims_eval_test_inputs(dists=None)
    with tempfile.TemporaryDirectory() as td:
        output = claims_eval(
            dataset=dataset,
            clusterer=clusterer,
            claims_pairs=claims_pairs,
            directory_for_caching=td,
            optional_name="unit",
        )
        assert os.path.exists(os.path.join(td, "preds_unit.json"))
        assert not os.path.exists(os.path.join(td, "dists_unit.pkl"))
        assert output["total"] == 1


def test_claims_eval_handles_none_affiliations():
    dataset, clusterer, claims_pairs = _build_claims_eval_test_inputs(dists=None)
    dataset.signatures["p1___0"].author_info_affiliations = None
    output = claims_eval(
        dataset=dataset,
        clusterer=clusterer,
        claims_pairs=claims_pairs,
        directory_for_caching=None,
        optional_name="unit",
    )
    assert output["total"] == 1


def test_claims_eval_writes_distance_dump_when_available():
    dataset, clusterer, claims_pairs = _build_claims_eval_test_inputs(dists={"blk": np.array([0.5])})
    with tempfile.TemporaryDirectory() as td:
        output = claims_eval(
            dataset=dataset,
            clusterer=clusterer,
            claims_pairs=claims_pairs,
            directory_for_caching=td,
            optional_name="unit",
        )
        assert os.path.exists(os.path.join(td, "preds_unit.json"))
        assert os.path.exists(os.path.join(td, "dists_unit.pkl"))
        assert output["total"] == 1


def test_native_lightgbm_shap_routes_booster_and_restores_params(monkeypatch):
    import lightgbm as lgb

    from s2and.production_model import NativeLightGBMBinaryClassifier

    X, y = _small_binary_classification()
    classifier = lgb.LGBMClassifier(n_estimators=4, random_state=7, verbosity=-1).fit(X, y)
    captured_models = []

    def mutate_booster_params(model, features, class_index):
        assert class_index == 1
        captured_models.append(model)
        model.params["temporary"] = "shap"
        return np.zeros_like(features)

    monkeypatch.setattr(shap_utils, "_shap_values_for_tree_model", mutate_booster_params)

    with tempfile.TemporaryDirectory() as td:
        model_path = os.path.join(td, "model.txt")
        classifier.booster_.save_model(model_path)
        native_classifier = NativeLightGBMBinaryClassifier(model_path)
        original_params = dict(native_classifier.booster_.params)

        values = _shap_values_for_tree_model_preserving_booster_params(native_classifier, X[:2])

        np.testing.assert_array_equal(values, np.zeros((2, 4)))
        assert captured_models == [native_classifier.booster_]
        assert isinstance(captured_models[0], lgb.Booster)
        assert native_classifier.booster_.params == original_params


def test_write_claims_eval_shap_plots_requires_nameless_features(monkeypatch):
    class DummyFeatureInfo:
        def get_feature_names(self):
            return ["feature"]

    class DummyClusterer:
        classifier = object()
        nameless_classifier = object()
        featurizer_info = DummyFeatureInfo()
        nameless_featurizer_info = None

    def fake_many_pairs_featurize(*_args, **_kwargs):
        return np.ones((1, 1)), np.array([1]), None

    monkeypatch.setattr("s2and.eval.many_pairs_featurize", fake_many_pairs_featurize)

    with pytest.raises(ValueError, match="output_shap=True requires clusterer.nameless_featurizer_info"):
        _write_claims_eval_shap_plots(
            id1="p1___0",
            id2="p2___0",
            dataset=cast(Any, object()),
            clusterer=cast(Any, DummyClusterer()),
            directory_for_caching=".",
        )


if __name__ == "__main__":
    unittest.main()
