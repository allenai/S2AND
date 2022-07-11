from typing import Dict, Optional, Any, List, Tuple, TYPE_CHECKING, Union

import logging
import pickle
import json
import warnings
from collections import Counter

if TYPE_CHECKING:  # need this for circular import issues
    from s2and.model import Clusterer
    from s2and.data import PDData

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
from tqdm import tqdm

from s2and.featurizer import many_pairs_featurize

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

    Returns
    -------
    Dict: Dictionary of clusterwise metrics.
    Dict: Same as above but broken down by signature.
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

    # block ground truth labels: cluster_to_signatures
    cluster_to_signatures = dataset.construct_cluster_to_papers(block_dict)

    # predict
    pred_clusters, _ = clusterer.predict(block_dict, dataset, use_s2_clusters=use_s2_clusters)

    # get metrics
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics: Dict[str, Tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "cmacro"
    )
    metrics["Pred bigger ratio (mean, count)"] = (
        np.round(np.mean(pred_bigger_ratios), 2),
        len(pred_bigger_ratios),
    )
    metrics["True bigger ratio (mean, count)"] = (
        np.round(np.mean(true_bigger_ratios), 2),
        len(true_bigger_ratios),
    )

    return metrics, b3_metrics_per_signature


def incremental_cluster_eval(
    dataset: "PDData", clusterer: "Clusterer", split: str = "test"
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Performs clusterwise evaluation for the incremental clustering setting.
    This includes both time-split and random split of signatures.
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
    Dict: Same as above but broken down by signature.
    """
    block_dict = dataset.get_blocks()
    (
        train_block_dict,
        val_block_dict,
        test_block_dict,
    ) = dataset.split_cluster_papers()
    # evaluation must happen only on test-signatures in blocks, so remove train/val signatures
    observed_signatures = set()
    for _, signatures in train_block_dict.items():
        for signature in signatures:
            observed_signatures.add(signature)

    # use entire block of signatures for predictions
    # NOTE: train/val/test block dicts can have overlapping signatures in the incremental case
    eval_block_dict_full = {}
    if split == "test":
        for block_key, _ in test_block_dict.items():
            eval_block_dict_full[block_key] = block_dict[block_key]
        cluster_to_signatures = dataset.construct_cluster_to_papers(test_block_dict)
        for _, signatures in val_block_dict.items():
            for signature in signatures:
                observed_signatures.add(signature)
    elif split == "val":
        cluster_to_signatures = dataset.construct_cluster_to_papers(val_block_dict)
        eval_block_dict_full = copy.deepcopy(val_block_dict)
        for block_key, signatures in train_block_dict.items():
            if block_key in eval_block_dict_full:
                eval_block_dict_full[block_key].extend(signatures)
    else:
        raise Exception("Evaluation split must be one of: {val, test}!")

    partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {}
    list_obs_signatures = list(observed_signatures)
    # considers the supervision as distances
    for i, signature_i in enumerate(list_obs_signatures):
        for signature_j in list_obs_signatures[i + 1 : len(list_obs_signatures)]:
            if dataset.paper_to_cluster_id[signature_i] == dataset.paper_to_cluster_id[signature_j]:
                partial_supervision[(signature_i, signature_j)] = 0
            else:
                partial_supervision[(signature_i, signature_j)] = 1

    # predict on test-blocks
    pred_clusters, _ = clusterer.predict(eval_block_dict_full, dataset, partial_supervision=partial_supervision)
    # to avoid sparsity in b3 computation, we use all the signatures' ground-truth
    full_cluster_to_signatures = dataset.construct_cluster_to_papers(pred_clusters)

    eval_only_pred_clusters = {}
    for cluster_key, signatures in pred_clusters.items():
        test_signatures = list(set(signatures).difference(observed_signatures))
        assert len(set(test_signatures).intersection(observed_signatures)) == 0
        if len(test_signatures) > 0:
            eval_only_pred_clusters[cluster_key] = test_signatures

    # get metrics
    b3_p, b3_r, b3_f1, b3_metrics_per_signature, _, _ = b3_precision_recall_fscore(
        full_cluster_to_signatures, pred_clusters, skip_signatures=observed_signatures
    )
    metrics = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, test_block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, eval_only_pred_clusters, test_block_dict, "cmacro"
    )

    return metrics, b3_metrics_per_signature


def facet_eval(
    dataset: "PDData",
    metrics_per_signature: Dict[str, Tuple[float, float, float]],
    block_type: str = "original",
) -> Tuple[
    Dict[str, List],
    Dict[str, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    Dict[int, List],
    List[dict],
]:
    """
    Extracts B3 per facets.
    The returned dictionaries are keyed by the metric itself. For example, the keys of the
    homonymity_f1 variable are floating points between 0 and 1 indicating the amount
    of homonymity. The values are the per-signature B3s that have this amount of homonymity.

    Parameters
    ----------
    dataset: PDData Input dataset
    metrics_per_signature: Dict
        B3 P/R/F1 per signature.
        Second output of cluster_eval function.
    block_type: string
        Whether to use Semantic Scholar ("s2") or "original" blocks

    Returns
    -------
    Dict: B3 F1 broken down by number of paper authors.
    Dict: B3 F1 broken down by year.
    Dict: B3 F1 broken down by block size.
    Dict: B3 F1 broken down by true cluster size.
    Dict: B3 F1 broken down by within-block homonymity fraction.
          Definition (per signature): Fraction of same names but within different clusters.
    Dict: B3 F1 broken down by within-block synonymity fraction.
          Definition (per signature): Fraction of different names but within same clusters.
    """
    block_len_dict = {}
    if block_type == "original":
        blocks = dataset.get_original_blocks()
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
        cluster_len_dict[cluster_id] = len(cluster_dict["paper_ids"])

    # Keep track of facet specific f-score performance
    author_num_f1 = defaultdict(list)
    year_f1 = defaultdict(list)
    block_len_f1 = defaultdict(list)
    cluster_len_f1 = defaultdict(list)
    # keep track feature availability facet specific f-score
    firstname_f1 = defaultdict(list)
    affiliation_f1 = defaultdict(list)
    email_f1 = defaultdict(list)
    abstract_f1 = defaultdict(list)
    venue_f1 = defaultdict(list)
    coauthors_f1 = defaultdict(list)

    signature_lookup = list()

    for signature_key, (p, r, f1) in metrics_per_signature.items():

        _signature_dict = dict()

        cluster_id = dataset.paper_to_cluster_id[signature_key]
        signature = dataset.papers[signature_key]
        paper = dataset.papers[str(signature.paper_id)]

        author_num_f1[len(paper.authors)].append(f1)
        year_f1[paper.year].append(f1)
        cluster_len_f1[cluster_len_dict[cluster_id]].append(f1)

        # full first-name
        if signature.author_info_first is not None and len(signature.author_info_first.replace(".", "")) >= 2:
            firstname_f1[1].append(f1)
            _signature_dict["first name"] = 1
        else:
            firstname_f1[0].append(f1)
            _signature_dict["first name"] = 0

        if len(signature.author_info_affiliations) > 0:
            affiliation_f1[1].append(f1)
            _signature_dict["affiliation"] = 1
        else:
            affiliation_f1[0].append(f1)
            _signature_dict["affiliation"] = 0

        if signature.author_info_email not in {"", None}:
            email_f1[1].append(f1)
            _signature_dict["email"] = 1
        else:
            email_f1[0].append(f1)
            _signature_dict["email"] = 0

        if paper.has_abstract:
            abstract_f1[1].append(f1)
            _signature_dict["abstract"] = 1
        else:
            abstract_f1[0].append(f1)
            _signature_dict["abstract"] = 0

        if paper.venue not in {"", None} or paper.journal_name not in {"", None}:
            venue_f1[1].append(f1)
            _signature_dict["venue"] = 1
        else:
            venue_f1[0].append(f1)
            _signature_dict["venue"] = 0
            
        if len(signature.author_info_coauthors) > 0:
            coauthors_f1[1].append(f1)
            _signature_dict["multiple_authors"] = 1
        else:
            coauthors_f1[0].append(f1)
            _signature_dict["multiple_authors"] = 0

        if block_type == "original":
            block_len_f1[block_len_dict[signature.author_info_given_block]].append(f1)
            _signature_dict["block size"] = block_len_dict[signature.author_info_given_block]
        elif block_type == "s2":
            block_len_f1[block_len_dict[signature.author_info_block]].append(f1)
            _signature_dict["block size"] = block_len_dict[signature.author_info_block]

        _signature_dict["paper_id"] = signature_key
        _signature_dict["precision"] = p
        _signature_dict["recall"] = r
        _signature_dict["f1"] = f1
        _signature_dict["#authors"] = len(paper.authors)
        _signature_dict["year"] = paper.year
        _signature_dict["cluster size"] = cluster_len_dict[cluster_id]

        signature_lookup.append(_signature_dict)

    return (
        author_num_f1,
        year_f1,
        block_len_f1,
        cluster_len_f1,
        firstname_f1,
        affiliation_f1,
        email_f1,
        abstract_f1,
        venue_f1,
        coauthors_f1,
        signature_lookup,
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


def b3_precision_recall_fscore(true_clus, pred_clus, skip_signatures=None):
    """
    Compute the B^3 variant of precision, recall and F-score.
    Modified from: https://github.com/glouppe/beard/blob/master/beard/metrics/clustering.py

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the ground-truth signature id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array containing
        the predicted signature id assignments as values.
    skip_signatures: List[string]
        in the incremental setting blocks can be partially supervised,
        hence those instances are not used for evaluation.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    Dict: P/R/F1 per signature

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
        raise ValueError("Predictions do not cover all the signatures!")

    # incremental evaluation contains partially observed signatures
    # skip_signatures are observed signatures, which we skip for b3 calc.
    if skip_signatures is not None:
        tcset = tcset.difference(skip_signatures)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    rev_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            rev_true_clusters[vi] = k

    rev_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            rev_pred_clusters[vi] = k

    intersections = {}
    per_signature_metrics = {}
    n_samples = len(tcset)

    true_bigger_ratios, pred_bigger_ratios = [], []
    for item in list(tcset):
        pred_cluster_i = pred_clusters[rev_pred_clusters[item]]
        true_cluster_i = true_clusters[rev_true_clusters[item]]

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
        per_signature_metrics[item] = (
            _precision,
            _recall,
            f1_score(_precision, _recall),
        )

    precision /= n_samples
    recall /= n_samples

    f_score = f1_score(precision, recall)

    return (
        np.round(precision, 3),
        np.round(recall, 3),
        np.round(f_score, 3),
        per_signature_metrics,
        pred_bigger_ratios,
        true_bigger_ratios,
    )


def cluster_precision_recall_fscore(
    true_clus: Dict[str, List[str]], pred_clus: Dict[str, List[str]]
) -> Tuple[float, float, float]:
    """
    Compute cluster-wise pair-wise precision, recall and F-score.

    The function also contains the fix proposed in
    https://arxiv.org/pdf/1808.04216.pdf to handle singleton clusters.

    Parameters
    ----------
    true_clus: Dict
        dictionary with cluster id as keys and 1d array
        containing the ground-truth signature id assignments as values.
    pred_clus: Dict
        dictionary with cluster id as keys and 1d array
        containing the predicted signature id assignments as values.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1

    Reference
    ---------
    Levin, Michael, et al. "Citation-based bootstrapping for
    large-scale author disambiguation." Journal of the American Society for Information
    Science and Technology (2012): 1030-1047.
    """

    goldpairs = set()
    syspairs = set()

    for _, signatures in true_clus.items():
        if len(signatures) == 1:
            goldpairs.add((signatures[0], signatures[0]))
            continue

        sort_sign = sorted(signatures)

        for i in range(len(sort_sign) - 1):
            for j in range(i + 1, len(sort_sign)):
                goldpairs.add((sort_sign[i], sort_sign[j]))

    for _, signatures in pred_clus.items():

        if len(signatures) == 1:
            syspairs.add((signatures[0], signatures[0]))
            continue

        sort_sign = sorted(signatures)

        for i in range(len(sort_sign) - 1):
            for j in range(i + 1, len(sort_sign)):
                syspairs.add((sort_sign[i], sort_sign[j]))

    precision: float = len(goldpairs.intersection(syspairs)) / len(syspairs)
    recall: float = len(goldpairs.intersection(syspairs)) / len(goldpairs)

    return precision, recall, f1_score(precision, recall)


def pairwise_precision_recall_fscore(true_clus, pred_clus, test_block, strategy="cmacro"):
    """
    Compute the Pairwise precision, recall and F-score.

    Parameters
    ----------
    true_clusters: Dict
        dictionary with cluster id as keys and
        1d array containing the ground-truth signature id assignments as values.
    pred_clusters: Dict
        dictionary with cluster id as keys and
        1d array containing the predicted signature id assignments as values.
    test_block: Dict
        dictionary with block id as keys and 1d array
        containing signature ids as values (block assignment).
    strategy: string
        'clusters' is cluster-wise pairwise precision, recall
        and f1 scores. It is computed over all possible pairs in true and predicted
        clusters. 'cmacro' is computed over each block, and averaged finally.

    Returns
    -------
    float: calculated precision
    float: calculated recall
    float: calculated F1
    """

    true_clusters = true_clus.copy()
    pred_clusters = pred_clus.copy()

    tcset = set(reduce(lambda x, y: x + y, true_clusters.values()))
    pcset = set(reduce(lambda x, y: x + y, pred_clusters.values()))

    if tcset != pcset:
        raise ValueError("predictions do not cover all the signatures.")

    rev_true_clusters = {}
    for k, v in true_clusters.items():
        for vi in v:
            rev_true_clusters[vi] = k

    rev_pred_clusters = {}
    for k, v in pred_clusters.items():
        for vi in v:
            rev_pred_clusters[vi] = k

    if strategy == "clusters":

        precision, recall, f1 = cluster_precision_recall_fscore(true_clus, pred_clus)
        return np.round(precision, 3), np.round(recall, 3), np.round(f1, 3)

    elif strategy == "cmacro":

        mprecision = 0
        mrecall = 0
        mf1 = 0

        for _, signatures in test_block.items():

            gtruth_block = {}
            prediction_block = {}

            for sign in signatures:
                tclus = rev_true_clusters[sign]
                pclus = rev_pred_clusters[sign]
                if tclus not in gtruth_block:
                    gtruth_block[tclus] = list()
                gtruth_block[tclus].append(sign)
                if pclus not in prediction_block:
                    prediction_block[pclus] = list()
                prediction_block[pclus].append(sign)

            _mprecision, _mrecall, _mf1 = cluster_precision_recall_fscore(gtruth_block, prediction_block)

            mprecision += _mprecision
            mrecall += _mrecall
            mf1 += _mf1

        mprecision = mprecision / len(test_block)
        mrecall = mrecall / len(test_block)
        mf1 = mf1 / len(test_block)

        return np.round(mprecision, 3), np.round(mrecall, 3), np.round(mf1, 3)
    

def min_pair_edit(preds):
    """Find minimum number of cluster changes
    to fully correct a block with errors.

    Args:
        preds: Dictionary that has cluster assignments and claim pairs.

    Returns:
        min_edit_score: Minimum edit distance score from 0 to 1
        min_edit_count: Unnormalized count version of score above.
        number_of_mistaken_ids: Total number of signature ids that were part of wrong pairs
    """
    wrong = preds["sig_pairs_wrong"]
    right = preds["sig_pairs_right"]

    if len(wrong) == 0:
        return 0, 0, 0

    signature_to_cluster = dict()
    for key, value in preds.items():
        if not key.startswith("sig_pairs"):
            for v in value:
                signature_to_cluster[v[1]] = key

    all_clusters = set(list(signature_to_cluster.values()))
    all_clusters.update(["dummy"])

    tp_sigs = set()
    tn_sigs = set()
    for sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same in wrong + right:
        if gold_same:
            tp_sigs.add((sig_id_1, sig_id_2))
        else:
            tn_sigs.add((sig_id_1, sig_id_2))

    def eval_current_cluster(signature_to_cluster):
        tp, fp, tn, fn = 0, 0, 0, 0
        for s_id_1, s_id_2 in tp_sigs:
            same_cluster_pred = signature_to_cluster[s_id_1] == signature_to_cluster[s_id_2]
            if same_cluster_pred:
                tp += 1
            else:
                fn += 1

        for s_id_1, s_id_2 in tn_sigs:
            same_cluster_pred = signature_to_cluster[s_id_1] == signature_to_cluster[s_id_2]
            if same_cluster_pred:
                fp += 1
            else:
                tn += 1

        return -fp + -fn

    wrong_counts = Counter()
    for sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same in wrong:
        wrong_counts.update([sig_id_1, sig_id_2])

    worst_ids = [i[0] for i in wrong_counts.most_common(10000000)]

    steps = 0
    for worst_id in worst_ids:
        original_cluster_label = signature_to_cluster[worst_id]
        best_f1 = eval_current_cluster(signature_to_cluster)
        flip_tos = [i for i in all_clusters if i != original_cluster_label]
        best_flip_to = None
        for flip_to in flip_tos:
            signature_to_cluster[worst_id] = flip_to
            current_f1 = eval_current_cluster(signature_to_cluster)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_flip_to = flip_to

        if best_flip_to is not None:
            signature_to_cluster[worst_id] = best_flip_to

            # remake wrong and right
            wrong_new, right_new = [], []
            for sig_id_1, sig_id_2, title_1, title_2, _, gold_same in wrong + right:
                pred_same = signature_to_cluster[sig_id_1] == signature_to_cluster[sig_id_2]
                if pred_same == gold_same:
                    right_new.append([sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same])
                else:
                    wrong_new.append([sig_id_1, sig_id_2, title_1, title_2, pred_same, gold_same])

            wrong = wrong_new
            right = right_new
            steps += 1
        else:
            signature_to_cluster[worst_id] = original_cluster_label

        if len(wrong) == 0:
            break

    if len(wrong) != 0:
        print("something went wrong")

    return steps / (len(worst_ids) - 1), steps, len(worst_ids)
