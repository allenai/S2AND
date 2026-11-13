"""Pure clustering metrics shared by training and evaluation."""

import itertools
from collections import Counter

import numpy as np


def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def _validated_cluster_partition_members(clusters: dict, *, label: str) -> set:
    """Return partition members after rejecting repeated signature assignments."""

    counts = Counter(itertools.chain.from_iterable(clusters.values()))
    duplicates = sorted((signature_id for signature_id, count in counts.items() if count > 1), key=str)
    if duplicates:
        raise ValueError(f"{label} clustering must be a partition; duplicate signatures: {duplicates}")
    return set(counts)


def _validated_cluster_partition_coverage(true_clus: dict, pred_clus: dict) -> tuple[set, set]:
    """Validate two partitions and return their equally covered signature sets."""

    true_members = _validated_cluster_partition_members(true_clus, label="Ground-truth")
    predicted_members = _validated_cluster_partition_members(pred_clus, label="Predicted")
    if true_members != predicted_members:
        raise ValueError("Predictions do not cover all the signatures!")
    return true_members, predicted_members


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

    tcset, _pcset = _validated_cluster_partition_coverage(true_clusters, pred_clusters)

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
    if n_samples == 0:
        return (
            np.round(0.0, 3),
            np.round(0.0, 3),
            np.round(0.0, 3),
            per_signature_metrics,
            [],
            [],
        )

    true_bigger_ratios, pred_bigger_ratios = [], []
    for item in list(tcset):
        pred_cluster_id = rev_pred_clusters[item]
        true_cluster_id = rev_true_clusters[item]
        pred_cluster_i = pred_clusters[pred_cluster_id]
        true_cluster_i = true_clusters[true_cluster_id]

        if len(pred_cluster_i) >= len(true_cluster_i):
            pred_bigger_ratios.append(len(pred_cluster_i) / len(true_cluster_i))
        else:
            true_bigger_ratios.append(len(true_cluster_i) / len(pred_cluster_i))

        memo_key = (pred_cluster_id, true_cluster_id)
        if memo_key in intersections:
            intersection = intersections[memo_key]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[memo_key] = intersection
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
    true_clus: dict[str, list[str]], pred_clus: dict[str, list[str]]
) -> tuple[float, float, float]:
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
    Levin, Michael, et al. "Citation‐based bootstrapping for
    large‐scale author disambiguation." Journal of the American Society for Information
    Science and Technology (2012): 1030-1047.
    """

    _validated_cluster_partition_coverage(true_clus, pred_clus)

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

    overlap = len(goldpairs.intersection(syspairs))
    precision = overlap / len(syspairs) if len(syspairs) > 0 else 0.0
    recall = overlap / len(goldpairs) if len(goldpairs) > 0 else 0.0

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

    _validated_cluster_partition_coverage(true_clusters, pred_clusters)

    if strategy == "clusters":
        precision, recall, f1 = cluster_precision_recall_fscore(true_clus, pred_clus)
        return np.round(precision, 3), np.round(recall, 3), np.round(f1, 3)

    elif strategy == "cmacro":
        rev_true_clusters = {}
        for k, v in true_clusters.items():
            for vi in v:
                rev_true_clusters[vi] = k

        rev_pred_clusters = {}
        for k, v in pred_clusters.items():
            for vi in v:
                rev_pred_clusters[vi] = k

        if len(test_block) == 0:
            return np.round(0.0, 3), np.round(0.0, 3), np.round(0.0, 3)
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
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
