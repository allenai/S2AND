from __future__ import annotations

import os
from os.path import join
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

# NumPy 1.24+ removed aliases like np.bool/np.int/np.float. Avoid triggering
# numpy's __getattr__ FutureWarnings by checking the dict directly before
# setting compatibility aliases used by older SHAP code paths.
if np.__dict__.get("bool", None) is None:  # NumPy>=1.24
    np.bool = np.bool_  # type: ignore[attr-defined]
if np.__dict__.get("int", None) is None:
    np.int = np.int_  # type: ignore[attr-defined]
if np.__dict__.get("float", None) is None:
    np.float = np.float64  # type: ignore[attr-defined]
if np.__dict__.get("object", None) is None:
    np.object = np.object_  # type: ignore[attr-defined]
import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import shap
from sklearn.calibration import CalibratedClassifierCV

ArrayLike = Union[np.ndarray, "shap._explanation.Explanation"]  # type: ignore[attr-defined]


def _is_fitted_tree_estimator(est) -> bool:
    # RandomForest/ExtraTrees -> estimators_, DecisionTree -> tree_
    return hasattr(est, "estimators_") and len(getattr(est, "estimators_", [])) > 0 or hasattr(est, "tree_")


def _base_estimator(clf):
    """
    Return a *fitted* underlying estimator when given a CalibratedClassifierCV,
    compatible across sklearn versions.
    """
    if isinstance(clf, CalibratedClassifierCV):
        # Preferred: dig into per-fold CalibratedClassifier objects
        if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
            for cc in clf.calibrated_classifiers_:
                for attr in ("base_estimator", "classifier", "estimator", "clf"):
                    cand = getattr(cc, attr, None)
                    if cand is not None and _is_fitted_tree_estimator(cand):
                        return cand  # fitted
        # Fallbacks on the outer CV wrapper (only use if *fitted*)
        for attr in ("estimator", "base_estimator"):
            cand = getattr(clf, attr, None)
            if cand is not None and cand != "deprecated" and _is_fitted_tree_estimator(cand):
                return cand
    return clf


def _iter_estimators(clf):
    """
    Return underlying estimators for voting/stacking-style ensembles.
    Do NOT treat tree ensembles (RandomForest, ExtraTrees, GB*) as voting ensembles.
    """
    if hasattr(clf, "estimators") and clf.estimators:
        ests = clf.estimators
        if isinstance(ests[0], tuple):  # e.g., VotingClassifier: [('rf', rf), ...]
            return [e for _, e in ests]
        return list(ests)  # already a list of estimators
    return None


def _shap_values_for_tree_model(model, X, class_index: int = 1) -> np.ndarray:
    """
    Compute SHAP values for a (tree) model and return a 2D array (n_samples, n_features).
    Compatible with SHAP >=0.36.0 through latest:
      - TreeExplainer(...).shap_values(X) may return:
        * list of arrays (multiclass) -> we pick class_index
        * single 2D array
    """
    expl = shap.TreeExplainer(model)
    vals = expl.shap_values(X)
    if isinstance(vals, list):
        # multiclass case
        return np.asarray(vals[class_index])
    # SHAP >=0.39 can sometimes return Explanation objects from non-TreeExplainer, but we force TreeExplainer above.
    # Still, be defensive:
    if hasattr(vals, "values"):
        vals = vals.values
    return np.asarray(vals)


def _safe_summary_plot(
    shap_values: np.ndarray,
    X: Union[np.ndarray, Any],
    feature_names: Sequence[str],
    shap_plot_type: str,
    outpath: str,
    fig_num: Optional[int] = None,
) -> None:
    """
    Prefer legacy summary_plot for cross-version compatibility.
    Falls back to beeswarm if summary_plot misbehaves.
    """
    if fig_num is not None:
        plt.figure(fig_num)
    else:
        plt.figure()
    try:
        shap.summary_plot(
            shap_values,
            X,
            plot_type=shap_plot_type,
            feature_names=feature_names,
            show=False,
            max_display=len(feature_names),
        )
    except Exception:
        # Fallback to the new API if needed
        try:
            exp = shap.Explanation(values=shap_values, data=X, feature_names=list(feature_names))  # type: ignore
            shap.plots.beeswarm(exp, show=False, max_display=len(feature_names))
        except Exception:
            plt.close()
            raise
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()


def compute_shap_summary_plots(
    *,
    classifier,
    X,
    shap_feature_names: Sequence[str],
    shap_plot_type: str,
    base_name: str,
    figs_path: str,
    nameless_classifier=None,
    nameless_X=None,
    nameless_feature_names: Optional[Sequence[str]] = None,
    class_index: int = 1,
) -> List[str]:
    """
    Computes SHAP values and writes summary plots as PNGs.
    Returns list of written file paths.

    Behavior:
      - If `classifier` is an ensemble with estimators, compute per-estimator SHAP and average.
      - If `nameless_classifier` is provided, compute and plot *both* (named and nameless) models.
      - If a CalibratedClassifierCV is encountered, SHAP runs on its base_estimator.
      - Uses TreeExplainer for maximum cross-version stability.

    Parameters
    ----------
    classifier : fitted classifier
    X : array-like, shape (n_samples, n_features)
    shap_feature_names : list[str]
    shap_plot_type : e.g. "dot", "bar", "violin"
    base_name : str  (used in output filenames)
    figs_path : str  (directory for outputs)
    nameless_classifier : optional fitted classifier
    nameless_X : optional array-like for the nameless classifier
    nameless_feature_names : optional list[str]
    class_index : which class to visualize for multiclass/binary (default=1)

    Returns
    -------
    List[str] : list of saved file paths
    """
    outputs: List[str] = []
    assert shap_feature_names is not None

    # Branch 1: ensemble averaging
    ensemble = _iter_estimators(classifier)
    if ensemble:
        vals_list: List[np.ndarray] = []
        for c in ensemble:
            base = _base_estimator(c)
            vals_list.append(_shap_values_for_tree_model(base, X, class_index))
        mean_vals = np.mean(np.stack(vals_list, axis=0), axis=0)
        out = join(figs_path, f"{base_name}_shap_0.png")
        _safe_summary_plot(mean_vals, X, shap_feature_names, shap_plot_type, out, fig_num=2)
        outputs.append(out)
        return outputs

    # Branch 2: two-model (named + nameless)
    if nameless_classifier is not None:
        pairs: List[Tuple[object, np.ndarray, Sequence[str], str]] = []

        base_a = _base_estimator(classifier)
        vals_a = _shap_values_for_tree_model(base_a, X, class_index)
        pairs.append((classifier, X, shap_feature_names, f"{base_name}_shap_0.png"))

        assert (
            nameless_X is not None and nameless_feature_names is not None
        ), "Provide nameless_X and nameless_feature_names when nameless_classifier is set."
        base_b = _base_estimator(nameless_classifier)
        vals_b = _shap_values_for_tree_model(base_b, nameless_X, class_index)

        # plot A
        out_a = join(figs_path, pairs[0][3])
        _safe_summary_plot(vals_a, X, shap_feature_names, shap_plot_type, out_a, fig_num=2)
        outputs.append(out_a)

        # plot B
        out_b = join(figs_path, f"{base_name}_shap_1.png")
        _safe_summary_plot(vals_b, nameless_X, nameless_feature_names, shap_plot_type, out_b, fig_num=3)
        outputs.append(out_b)
        return outputs

    # Branch 3: single model
    base = _base_estimator(classifier)
    vals = _shap_values_for_tree_model(base, X, class_index)
    out = join(figs_path, f"{base_name}_shap.png")
    _safe_summary_plot(vals, X, shap_feature_names, shap_plot_type, out, fig_num=2)
    outputs.append(out)
    return outputs
