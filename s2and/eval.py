from typing import Dict, Optional, Any, List, Tuple, TYPE_CHECKING, Union

import logging

if TYPE_CHECKING:  # need this for circular import issues
    from s2and.model import Clusterer
    from s2and.data import PDData

from s2and.consts import ORPHAN_CLUSTER_KEY

import os
from os.path import join
from functools import reduce
from collections import defaultdict

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import copy


logger = logging.getLogger("s2and")

sns.set(context="talk")


def cluster_eval(
    dataset: "PDData",
    clusterer: "Clusterer",
    split: str = "test",
    use_s2_clusters: bool = False,
) -> Tuple[Dict[str, Tuple], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: PDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?
    use_s2_clusters: bool
        Whether to use the original S2 clusters

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by paper.
    """
    train_block_dict, val_block_dict, test_block_dict = dataset.split_blocks_helper(dataset.get_blocks())
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise Exception("Split must be one of: {train, val, test}!")

    # block ground truth labels: cluster_to_papers
    cluster_to_papers = dataset.construct_cluster_to_papers(block_dict)

    # predict
    pred_clusters, _ = clusterer.predict(block_dict, dataset, use_s2_clusters=use_s2_clusters)

    # get metrics
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_paper,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_papers, pred_clusters)
    metrics: Dict[str, Tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}

    metrics["Pred bigger ratio (mean, count)"] = (
        np.round(np.mean(pred_bigger_ratios), 2),
        len(pred_bigger_ratios),
    )
    metrics["True bigger ratio (mean, count)"] = (
        np.round(np.mean(true_bigger_ratios), 2),
        len(true_bigger_ratios),
    )

    return metrics, b3_metrics_per_paper


def incremental_cluster_eval(
    dataset: "PDData", clusterer: "Clusterer", split: str = "test"
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation for the incremental clustering setting.
    This includes both time-split and random split of papers.
    Returns B3, Cluster F1, and Cluster Macro F1.

    Parameters
    ----------
    dataset: PDData
        Dataset that has ground truth
    clusterer: Clusterer
        Clusterer object that will do predicting.
    split: string
        Which split in the dataset are we evaluating?

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by paper.
    """
    block_dict = dataset.get_blocks()
    (
        train_block_dict,
        val_block_dict,
        test_block_dict,
    ) = dataset.split_cluster_papers()
    # evaluation must happen only on test-papers in blocks, so remove train/val papers
    observed_papers = set()
    for _, papers in train_block_dict.items():
        for paper in papers:
            observed_papers.add(paper)

    # use entire block of papers for predictions
    # NOTE: train/val/test block dicts can have overlapping papers in the incremental case
    eval_block_dict_full = {}
    if split == "test":
        for block_key, _ in test_block_dict.items():
            eval_block_dict_full[block_key] = block_dict[block_key]
        for _, papers in val_block_dict.items():
            for paper in papers:
                observed_papers.add(paper)
    elif split == "val":
        eval_block_dict_full = copy.deepcopy(val_block_dict)
        for block_key, papers in train_block_dict.items():
            if block_key in eval_block_dict_full:
                eval_block_dict_full[block_key].extend(papers)
    else:
        raise Exception("Evaluation split must be one of: {val, test}!")

    partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {}
    list_obs_papers = list(observed_papers)
    # considers the supervision as distances
    for i, paper_i in enumerate(list_obs_papers):
        for paper_j in list_obs_papers[i + 1 : len(list_obs_papers)]:
            if dataset.paper_to_cluster_id[paper_i] == dataset.paper_to_cluster_id[paper_j]:
                if dataset.paper_to_cluster_id[paper_i] != ORPHAN_CLUSTER_KEY:
                    partial_supervision[(paper_i, paper_j)] = 0
            else:
                partial_supervision[(paper_i, paper_j)] = 1

    # predict on test-blocks
    pred_clusters, _ = clusterer.predict(eval_block_dict_full, dataset, partial_supervision=partial_supervision)
    # to avoid sparsity in b3 computation, we use all the papers' ground-truth
    full_cluster_to_papers = dataset.construct_cluster_to_papers(pred_clusters)

    eval_only_pred_clusters = {}
    for cluster_key, papers in pred_clusters.items():
        test_papers = list(set(papers).difference(observed_papers))
        assert len(set(test_papers).intersection(observed_papers)) == 0
        if len(test_papers) > 0:
            eval_only_pred_clusters[cluster_key] = test_papers

    # get metrics
    b3_p, b3_r, b3_f1, b3_metrics_per_paper, _, _ = b3_precision_recall_fscore(
        full_cluster_to_papers, pred_clusters, skip_papers=observed_papers
    )
    metrics = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}

    return metrics, b3_metrics_per_paper


def facet_eval(
    dataset: "PDData",
    metrics_per_paper: Dict[str, Tuple[float, float, float]],
    block_type: str = "original",
) -> Tuple[Dict[int, List], Dict[int, List], Dict[int, List], Dict[int, List], Dict[int, List], List[dict]]:
    """
    Extracts B3 per facets.
    The returned dictionaries are keyed by the metric itself. For example, the keys of the
    homonymity_f1 variable are floating points between 0 and 1 indicating the amount
    of homonymity. The values are the per-paper B3s that have this amount of homonymity.

    Parameters
    ----------
    dataset: PDData Input dataset
    metrics_per_paper: Dict
        B3 P/R/F1 per paper.
        Second output of cluster_eval function.
    block_type: string
        Whether to use Semantic Scholar ("s2") or "original" blocks

    Returns
    -------
    Dict: B3 F1 broken down by number of paper authors.
    Dict: B3 F1 broken down by year.
    Dict: B3 F1 broken down by block size.
    Dict: B3 F1 broken down by true cluster size.
    Dict: B3 F1 broken down by whether there is an abstract
    """
    block_len_dict = {}
    if block_type == "original":
        blocks = dataset.get_blocks()
    elif block_type == "s2":
        blocks = dataset.get_s2_blocks()
    else:
        raise Exception("block_type must one of: {'original', 's2'}!")

    for block_key, paper_ids in blocks.items():
        block_len_dict[block_key] = len(paper_ids)

    # we need to know the length of each cluster
    assert dataset.clusters is not None
    cluster_len_dict = {}
    for cluster_id, cluster_dict in dataset.clusters.items():
        if cluster_id != ORPHAN_CLUSTER_KEY:
            cluster_len_dict[cluster_id] = len(cluster_dict["paper_ids"])

    # Keep track of facet specific f-score performance
    author_num_f1 = defaultdict(list)
    year_f1 = defaultdict(list)
    block_len_f1 = defaultdict(list)
    cluster_len_f1 = defaultdict(list)
    abstract_f1 = defaultdict(list)

    paper_lookup = list()

    for paper_id, (p, r, f1) in metrics_per_paper.items():

        _paper_dict = dict()

        cluster_id = dataset.paper_to_cluster_id[paper_id]
        if cluster_id != ORPHAN_CLUSTER_KEY:
            paper = dataset.papers[str(paper_id)]

            author_num_f1[len(paper.authors)].append(f1)
            year_f1[paper.year].append(f1)
            cluster_len_f1[cluster_len_dict[cluster_id]].append(f1)

            # full first-name
            if paper.has_abstract:
                abstract_f1[1].append(f1)
                _paper_dict["abstract"] = 1
            else:
                abstract_f1[0].append(f1)
                _paper_dict["abstract"] = 0

            if block_type == "original":
                block_len_f1[block_len_dict[paper.block]].append(f1)
                _paper_dict["block size"] = block_len_dict[paper.author_info_given_block]
            elif block_type == "s2":
                # TODO: update author_info_block to whatever we use for original block
                block_len_f1[block_len_dict[paper.author_info_block]].append(f1)
                _paper_dict["block size"] = block_len_dict[paper.author_info_block]

            _paper_dict["paper_id"] = paper_id  # type: ignore
            _paper_dict["precision"] = p  # type: ignore
            _paper_dict["recall"] = r  # type: ignore
            _paper_dict["f1"] = f1  # type: ignore
            _paper_dict["#authors"] = len(paper.authors)
            _paper_dict["year"] = paper.year
            _paper_dict["cluster size"] = cluster_len_dict[cluster_id]

            paper_lookup.append(_paper_dict)

    return (
        dict(author_num_f1),
        dict(year_f1),
        dict(block_len_f1),
        dict(cluster_len_f1),
        dict(abstract_f1),
        paper_lookup,
    )


def pairwise_eval(
    X: np.array,
    y: np.array,
    classifier: Any,
    figs_path: str,
    title: str,
    shap_feature_names: List[str],
    thresh_for_f1: float = 0.5,
    shap_plot_type: Optional[str] = "dot",
    nameless_classifier: Optional[Any] = None,
    nameless_X: Optional[np.array] = None,
    nameless_feature_names: Optional[List[str]] = None,
    skip_shap: bool = False,
) -> Dict[str, float]:
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
        Whether to skip SHAP entirely.

    Returns
    -------
    Dict: A dictionary of common pairwise metrics.
    """
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    # filename base will be title but lower and underscores
    base_name = title.lower().replace(" ", "_")
    if hasattr(classifier, "classifier"):
        classifier = classifier.classifier

    if nameless_classifier is not None and hasattr(nameless_classifier, "classifier"):
        nameless_classifier = nameless_classifier.classifier

    if nameless_classifier is not None:
        y_prob = (classifier.predict_proba(X)[:, 1] + nameless_classifier.predict_proba(nameless_X)[:, 1]) / 2
    else:
        y_prob = classifier.predict_proba(X)[:, 1]

    # plot AUROC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(0, figsize=(15, 15))
    plt.plot(fpr, tpr, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
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
    plt.plot(
        precision,
        recall,
        lw=2,
        label="PR curve (average precision = %0.2f)" % avg_precision,
    )
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"PR Curve for {title}")
    plt.legend(loc="lower left")
    plt.savefig(join(figs_path, base_name + "_pr.png"))
    plt.clf()
    plt.close()

    # plot SHAP
    # note that SHAP doesn't support model stacking directly
    # so we have to approximate by getting SHAP values for each
    # of the models inside the stack
    if not skip_shap:
        from s2and.model import VotingClassifier  # avoid circular import

        if isinstance(classifier, VotingClassifier):
            shap_values_all = []
            for c in classifier.estimators:
                if isinstance(c, CalibratedClassifierCV):
                    shap_values_all.append(shap.TreeExplainer(c.base_estimator).shap_values(X)[1])
                else:
                    shap_values_all.append(shap.TreeExplainer(c).shap_values(X)[1])
            shap_values = [np.mean(shap_values_all, axis=0)]
        elif nameless_classifier is not None:
            shap_values = []
            for c, d in [(classifier, X), (nameless_classifier, nameless_X)]:
                if isinstance(classifier, CalibratedClassifierCV):
                    shap_values.append(shap.TreeExplainer(c.base_estimator).shap_values(d)[1])
                else:
                    shap_values.append(shap.TreeExplainer(c).shap_values(d)[1])
        elif isinstance(classifier, CalibratedClassifierCV):
            shap_values = shap.TreeExplainer(classifier.base_estimator).shap_values(X)[1]
        else:
            shap_values = shap.TreeExplainer(classifier).shap_values(X)[1]

        if isinstance(shap_values, list):
            for i, (shap_value, feature_names, d) in enumerate(
                zip(
                    shap_values,
                    [shap_feature_names, nameless_feature_names],
                    [X, nameless_X],
                )
            ):
                assert feature_names is not None, "neither feature_names should be None here"
                plt.figure(2 + i)
                shap.summary_plot(
                    shap_value,
                    d,
                    plot_type=shap_plot_type,
                    feature_names=feature_names,
                    show=False,
                    max_display=len(feature_names),
                )
                # plt.title(f"{i}: SHAP Values for {title}")
                plt.tight_layout()
                plt.savefig(join(figs_path, base_name + f"_shap_{i}.png"))
                plt.clf()
                plt.close()
        else:
            plt.figure(2)
            shap.summary_plot(
                shap_values,
                X,
                plot_type=shap_plot_type,
                feature_names=shap_feature_names,
                show=False,
                max_display=len(shap_feature_names),
            )
            # plt.title(f"SHAP Values for {title}")
            plt.tight_layout()
            plt.savefig(join(figs_path, base_name + "_shap.png"))
            plt.clf()
            plt.close()

    # collect metrics and return
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_prob > thresh_for_f1, beta=1.0, average="macro")
    metrics = {
        "AUROC": np.round(roc_auc, 3),
        "Average Precision": np.round(avg_precision, 3),
        "F1": np.round(f1, 3),
        "Precision": np.round(pr, 3),
        "Recall": np.round(rc, 3),
    }

    return metrics


def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def b3_precision_recall_fscore(true_clus, pred_clus, skip_papers=None):
    """
    Compute the B^3 variant of precision, recall and F-score.
    Modified from: https://github.com/glouppe/beard/blob/master/beard/metrics/clustering.py

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the ground-truth paper id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the predicted paper id assignments as values.
    skip_papers: List[string]
        in the incremental setting blocks can be partially supervised,
        hence those instances are not used for evaluation.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    Dict: P/R/F1 per paper

    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """

    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set(reduce(lambda x, y: x + y, true_clusters.values()))
    pcset = set(reduce(lambda x, y: x + y, pred_clusters.values()))

    if tcset != pcset:
        raise ValueError("Predictions do not cover all the papers!")

    # incremental evaluation contains partially observed papers
    # skip_papers are observed papers, which we skip for b3 calc.
    if skip_papers is not None:
        tcset = tcset.difference(skip_papers)

    # anything from the orphan cluster will also be skipped
    # but note that other positives will be penalized for joining to any orphans
    if ORPHAN_CLUSTER_KEY in true_clusters:
        to_skip = true_clusters[ORPHAN_CLUSTER_KEY]
        tcset = tcset.difference(to_skip)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    reverse_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            reverse_true_clusters[vi] = k

    reverse_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            reverse_pred_clusters[vi] = k

    intersections = {}
    per_paper_metrics = {}

    true_bigger_ratios, pred_bigger_ratios = [], []
    for item in list(tcset):
        pred_cluster_i = pred_clusters[reverse_pred_clusters[item]]
        true_cluster_i = true_clusters[reverse_true_clusters[item]]

        if len(pred_cluster_i) >= len(true_cluster_i):
            pred_bigger_ratios.append(len(pred_cluster_i) / len(true_cluster_i))
        else:
            true_bigger_ratios.append(len(true_cluster_i) / len(pred_cluster_i))

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection
        _precision = len(intersection) / len(pred_cluster_i)
        _recall = len(intersection) / len(true_cluster_i)
        precision += _precision
        recall += _recall
        per_paper_metrics[item] = (
            _precision,
            _recall,
            f1_score(_precision, _recall),
        )

    n_samples = len(tcset)
    precision /= n_samples
    recall /= n_samples

    f_score = f1_score(precision, recall)

    return (
        np.round(precision, 3),
        np.round(recall, 3),
        np.round(f_score, 3),
        per_paper_metrics,
        pred_bigger_ratios,
        true_bigger_ratios,
    )
