import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # need this for circular import issues
    from s2and.data import ANDData
    from s2and.model import Clusterer

import copy
import os
from os.path import join

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from s2and.data import _resolve_signature_splits
from s2and.metrics import (
    b3_precision_recall_fscore as b3_precision_recall_fscore,
)
from s2and.metrics import (
    cluster_precision_recall_fscore as cluster_precision_recall_fscore,
)
from s2and.metrics import (
    f1_score as f1_score,
)
from s2and.metrics import (
    pairwise_precision_recall_fscore as pairwise_precision_recall_fscore,
)
from s2and.model_pairwise import _validated_classifier_features

logger = logging.getLogger("s2and")


def cluster_eval(
    dataset: "ANDData",
    clusterer: "Clusterer",
    split: str = "test",
) -> tuple[dict[str, tuple], dict[str, tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: ANDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by signature.
    """
    train_block_dict, val_block_dict, test_block_dict = _resolve_signature_splits(dataset)
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test}!")

    # block ground truth labels: cluster_to_signatures
    cluster_to_signatures = dataset.construct_cluster_to_signatures(block_dict)

    pred_clusters, _ = clusterer.predict(block_dict, dataset)

    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics: dict[str, tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "cmacro"
    )

    def _mean_or_nan(xs):
        if len(xs) == 0:
            return float("nan")
        return float(np.round(np.mean(xs), 2))

    metrics["Pred bigger ratio (mean, count)"] = (_mean_or_nan(pred_bigger_ratios), len(pred_bigger_ratios))
    metrics["True bigger ratio (mean, count)"] = (_mean_or_nan(true_bigger_ratios), len(true_bigger_ratios))

    return metrics, b3_metrics_per_signature


def incremental_cluster_eval(
    dataset: "ANDData", clusterer: "Clusterer", split: str = "test"
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation for the incremental clustering setting.
    This includes both time-split and random split of signatures.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Notes
    -----
    Partial supervision is materialized for all observed-signature pairs.
    If there are M observed signatures, this step builds M*(M-1)/2 pairs
    (quadratic time and memory in M).

    Parameters
    ----------
    dataset: ANDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by signature.
    """
    block_dict = dataset.get_blocks()
    (
        train_block_dict,
        val_block_dict,
        test_block_dict,
    ) = _resolve_signature_splits(dataset)
    # evaluation must happen only on test-signatures in blocks, so remove train/val signatures
    observed_signatures = set()
    for _, signatures in train_block_dict.items():
        for signature in signatures:
            observed_signatures.add(signature)

    # use entire block of signatures for predictions
    # NOTE: train/val/test block dicts can have overlapping signatures in the incremental case
    eval_block_dict_full = {}
    eval_block_dict_for_metrics: dict[str, list[str]]
    if split == "test":
        selected_signatures = {
            signature
            for blocks in (train_block_dict, val_block_dict, test_block_dict)
            for signatures in blocks.values()
            for signature in signatures
        }
        for block_key, _ in test_block_dict.items():
            eval_block_dict_full[block_key] = [
                signature for signature in block_dict[block_key] if signature in selected_signatures
            ]
        cluster_to_signatures = dataset.construct_cluster_to_signatures(test_block_dict)
        eval_block_dict_for_metrics = test_block_dict
        for _, signatures in val_block_dict.items():
            for signature in signatures:
                observed_signatures.add(signature)
    elif split == "val":
        cluster_to_signatures = dataset.construct_cluster_to_signatures(val_block_dict)
        eval_block_dict_for_metrics = val_block_dict
        eval_block_dict_full = copy.deepcopy(val_block_dict)
        for block_key, signatures in train_block_dict.items():
            if block_key in eval_block_dict_full:
                eval_block_dict_full[block_key].extend(signatures)
    else:
        raise Exception("Evaluation split must be one of: {val, test}!")

    partial_supervision: dict[tuple[str, str], int | float] = {}
    if dataset.signature_to_cluster_id is None:
        raise ValueError("cluster_eval requires dataset.signature_to_cluster_id")
    signature_to_cluster_id = dataset.signature_to_cluster_id
    list_obs_signatures = list(observed_signatures)
    observed_count = len(list_obs_signatures)
    observed_pair_count = observed_count * (observed_count - 1) // 2
    logger.debug(
        "Building incremental partial supervision for %d observed signatures (%d pairs).",
        observed_count,
        observed_pair_count,
    )
    # considers the supervision as distances
    for i, signature_i in enumerate(list_obs_signatures):
        for signature_j in list_obs_signatures[i + 1 : len(list_obs_signatures)]:
            if signature_to_cluster_id[signature_i] == signature_to_cluster_id[signature_j]:
                partial_supervision[(signature_i, signature_j)] = 0
            else:
                partial_supervision[(signature_i, signature_j)] = 1

    pred_clusters, _ = clusterer.predict(eval_block_dict_full, dataset, partial_supervision=partial_supervision)
    # to avoid sparsity in b3 computation, we use all the signatures' ground-truth
    full_cluster_to_signatures = dataset.construct_cluster_to_signatures(pred_clusters)

    scored_signatures = {signature for signatures in eval_block_dict_for_metrics.values() for signature in signatures}
    eval_only_pred_clusters = {}
    for cluster_key, signatures in pred_clusters.items():
        test_signatures = [signature for signature in signatures if signature in scored_signatures]
        assert len(set(test_signatures).intersection(observed_signatures)) == 0
        if len(test_signatures) > 0:
            eval_only_pred_clusters[cluster_key] = test_signatures

    b3_p, b3_r, b3_f1, b3_metrics_per_signature, _, _ = b3_precision_recall_fscore(
        full_cluster_to_signatures, pred_clusters, skip_signatures=observed_signatures
    )
    metrics = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, eval_block_dict_for_metrics, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, eval_block_dict_for_metrics, "cmacro"
    )

    return metrics, b3_metrics_per_signature


def pairwise_eval(
    X: np.ndarray,
    y: np.ndarray,
    classifier: Any,
    figs_path: str,
    title: str,
    shap_feature_names: list[str],
    thresh_for_f1: float = 0.5,
    shap_plot_type: str | None = "dot",
    nameless_classifier: Any | None = None,
    nameless_X: np.ndarray | None = None,
    nameless_feature_names: list[str] | None = None,
    skip_shap: bool = False,
) -> dict[str, float]:
    """
    Performs pairwise model evaluation, without using blocks.
    Also writes plots to the provided file path

    Parameters
    ----------
    X: np.array
        Feature matrix of features to do eval on.
    y: np.array
        Feature matrix of labels to do eval on.
    classifier: sklearn compatible classifier
        Classifier to do eval on.
        SHAP plots require a directly fitted tree or LightGBM classifier;
        pass ``skip_shap=True`` for calibrated, voting, stacking, or non-tree classifiers.
    figs_path: string
        Where to put the resulting evaluation figures.
    title: string
        Title to stick on all the plots and use for file name.
    shap_feature_names: List[str]
        List of feature names for the SHAP plots.
    thresh_for_f1: float
        Threshold for F1 computation. Defaults to 0.5.
    shap_plot_type: str
        Type of shap plot. Defaults to 'dot'.
        Can also be: 'bar', 'violin', 'compact_dot'
    nameless_classifier: sklearn compatible classifier
        Classifier to do eval on that doesn't use name features.
    nameless_X: np.array
        Feature matrix of features to do eval on excluding name features.
    nameless_feature_names: List[str]
        List of feature names for the SHAP plots excluding name features.
    skip_shap: bool
        Whether to skip SHAP entirely. Set this for classifiers that are not
        directly fitted tree or LightGBM models.

    Returns
    -------
    Dict: A dictionary of common pairwise metrics.
    """
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    base_name = title.lower().replace(" ", "_")
    if hasattr(classifier, "classifier"):
        classifier = classifier.classifier

    if nameless_classifier is not None and hasattr(nameless_classifier, "classifier"):
        nameless_classifier = nameless_classifier.classifier

    classifier_input = _validated_classifier_features(
        classifier,
        X,
        feature_names=shap_feature_names,
    )
    if nameless_classifier is not None:
        if nameless_X is None or nameless_feature_names is None:
            raise ValueError("nameless_X and nameless_feature_names are required with nameless_classifier")
        nameless_input = _validated_classifier_features(
            nameless_classifier,
            nameless_X,
            feature_names=nameless_feature_names,
        )
        y_prob = (
            classifier.predict_proba(classifier_input)[:, 1] + nameless_classifier.predict_proba(nameless_input)[:, 1]
        ) / 2
    else:
        y_prob = classifier.predict_proba(classifier_input)[:, 1]

    import matplotlib.pyplot as plt
    import seaborn as sns

    with sns.axes_style("darkgrid"), sns.plotting_context("talk"), sns.color_palette("deep"):
        # plot AUROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(0, figsize=(15, 15))
        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
        plt.xlim((-0.01, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {title}")
        plt.legend(loc="lower right")
        plt.savefig(join(figs_path, base_name + "_roc.png"))
        plt.clf()
        plt.close()

        # plot AUPR
        precision, recall, _ = precision_recall_curve(y, y_prob)
        avg_precision = average_precision_score(y, y_prob)

        plt.figure(1, figsize=(15, 15))
        # Standard: recall on x-axis, precision on y-axis
        plt.plot(
            recall,
            precision,
            lw=2,
            label=f"PR curve (AP = {avg_precision:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve for {title}")
        plt.legend(loc="lower left")
        plt.savefig(join(figs_path, base_name + "_pr.png"))
        plt.clf()
        plt.close()

        # plot SHAP -- delegate to central helper that normalizes across SHAP/NumPy versions
        if not skip_shap and shap_plot_type is not None:
            from s2and.shap_utils import compute_shap_summary_plots

            compute_shap_summary_plots(
                classifier=classifier,
                X=X,
                shap_feature_names=shap_feature_names,
                shap_plot_type=shap_plot_type,
                base_name=base_name,
                figs_path=figs_path,
                nameless_classifier=nameless_classifier,
                nameless_X=nameless_X,
                nameless_feature_names=nameless_feature_names,
            )

    pr, rc, f1, _ = precision_recall_fscore_support(
        y, y_prob > thresh_for_f1, beta=1.0, average="macro", zero_division=0
    )
    metrics = {
        "AUROC": np.round(roc_auc, 3),
        "Average Precision": np.round(avg_precision, 3),
        "F1": np.round(f1, 3),
        "Precision": np.round(pr, 3),
        "Recall": np.round(rc, 3),
    }

    return metrics


def pairwise_probability_metrics(
    labels: np.ndarray,
    main_positive: np.ndarray,
    nameless_positive: np.ndarray,
    *,
    include_average_precision: bool = False,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Compute the single pairwise metric contract shared by smoke and release runs.

    Both the pairwise-only smoke path in ``train_pairwise.py`` and the sealed
    release evaluator in ``release_pairwise.py`` must report identical keys
    computed identically, so that smoke evidence actually exercises the release
    report schema. Positive probabilities from the main and nameless models are
    averaged exactly once, and the F1 threshold is strict ``> 0.5``.

    Args:
        labels: Binary ground-truth labels, one per pair.
        main_positive: Positive-class probabilities from the main model.
        nameless_positive: Positive-class probabilities from the nameless model.
        include_average_precision: Whether to add ``average_precision``. Release
            reports omit it; smoke reports include it as diagnostic evidence.

    Returns:
        The metric mapping and the averaged positive probabilities.

    Raises:
        ValueError: If shapes disagree, labels are empty, or only one class is
            present.
        RuntimeError: If any computed metric is non-finite.
    """

    y = np.asarray(labels).reshape(-1)
    main = np.asarray(main_positive, dtype=np.float64).reshape(-1)
    nameless = np.asarray(nameless_positive, dtype=np.float64).reshape(-1)
    if y.shape != main.shape or y.shape != nameless.shape:
        raise ValueError(
            "Pair labels and both probability vectors must have equal shape: "
            f"labels={y.shape} main={main.shape} nameless={nameless.shape}"
        )
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("Pair evaluation requires nonempty labels with both classes")
    probabilities = (main + nameless) / 2.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        probabilities > 0.5,
        beta=1.0,
        average="macro",
        zero_division=0,
    )
    metrics: dict[str, float | int] = {
        "rows": int(y.size),
        "auroc": float(roc_auc_score(y, probabilities)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }
    if include_average_precision:
        metrics["average_precision"] = float(average_precision_score(y, probabilities))
    non_finite = sorted(key for key, value in metrics.items() if key != "rows" and not np.isfinite(float(value)))
    if non_finite:
        raise RuntimeError(f"Pair evaluation produced non-finite metrics: {non_finite}")
    return metrics, probabilities
