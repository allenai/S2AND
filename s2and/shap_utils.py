from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from os.path import join
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier

_SHAP_MODULE: Any | None = None


def _get_shap_module() -> Any:
    global _SHAP_MODULE
    if _SHAP_MODULE is not None:
        return _SHAP_MODULE
    try:
        _SHAP_MODULE = importlib.import_module("shap")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import shap. Install the supported SHAP dependency (for example, `uv sync`)."
        ) from exc
    return _SHAP_MODULE


def _shap_values_for_tree_model(model: Any, X: Any, class_index: int = 1) -> np.ndarray:
    """Return 2D SHAP values for a directly fitted tree model.

    Calibrated classifiers and voting or stacking ensembles are intentionally
    unsupported because their predictions are not explained by a single
    underlying tree model.

    Args:
        model: A fitted tree or LightGBM classifier.
        X: Two-dimensional feature rows accepted by the model.
        class_index: Output class to select from multi-output SHAP values.

    Returns:
        A ``(n_samples, n_features)`` SHAP value array.

    Raises:
        TypeError: If the classifier combines or calibrates multiple models.
        ValueError: If SHAP returns an unsupported shape or class index.
    """
    if getattr(model, "prediction_backend", None) == "rust_lightgbm":
        model = model.booster_
    if isinstance(model, (CalibratedClassifierCV, VotingClassifier, StackingClassifier)):
        raise TypeError(
            "SHAP diagnostics support directly fitted tree models only; "
            f"{type(model).__name__} is unsupported. Pass skip_shap=True."
        )

    shap = _get_shap_module()
    expl = shap.TreeExplainer(model)
    vals = np.asarray(expl.shap_values(X))
    if vals.ndim == 2:
        return vals
    if vals.ndim != 3:
        raise ValueError(f"TreeExplainer returned SHAP values with unsupported shape {vals.shape}; expected 2D or 3D")
    if not 0 <= class_index < vals.shape[2]:
        raise ValueError(f"class_index {class_index} is out of range for SHAP values with shape {vals.shape}")
    return vals[:, :, class_index]


def _safe_summary_plot(
    shap_values: np.ndarray,
    X: np.ndarray | Any,
    feature_names: Sequence[str],
    shap_plot_type: str,
    outpath: str,
    fig_num: int | None = None,
) -> None:
    """Write a SHAP summary plot, falling back to beeswarm if summary_plot fails."""
    shap = _get_shap_module()
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
        try:
            exp = shap.Explanation(values=shap_values, data=X, feature_names=list(feature_names))
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
    nameless_feature_names: Sequence[str] | None = None,
    class_index: int = 1,
) -> list[str]:
    """
    Computes SHAP values and writes summary plots as PNGs.
    Returns list of written file paths.

    Behavior:
      - If `nameless_classifier` is provided, compute and plot *both* (named and nameless) models.
      - Uses TreeExplainer on directly fitted tree models.
      - Calibrated classifiers and voting/stacking ensembles are unsupported.

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
    outputs: list[str] = []
    assert shap_feature_names is not None

    if nameless_classifier is not None:
        vals_a = _shap_values_for_tree_model(classifier, X, class_index)
        assert nameless_X is not None and nameless_feature_names is not None, (
            "Provide nameless_X and nameless_feature_names when nameless_classifier is set."
        )
        vals_b = _shap_values_for_tree_model(nameless_classifier, nameless_X, class_index)

        out_a = join(figs_path, f"{base_name}_shap_0.png")
        _safe_summary_plot(vals_a, X, shap_feature_names, shap_plot_type, out_a, fig_num=2)
        outputs.append(out_a)

        out_b = join(figs_path, f"{base_name}_shap_1.png")
        _safe_summary_plot(vals_b, nameless_X, nameless_feature_names, shap_plot_type, out_b, fig_num=3)
        outputs.append(out_b)
        return outputs

    vals = _shap_values_for_tree_model(classifier, X, class_index)
    out = join(figs_path, f"{base_name}_shap.png")
    _safe_summary_plot(vals, X, shap_feature_names, shap_plot_type, out, fig_num=2)
    outputs.append(out)
    return outputs
