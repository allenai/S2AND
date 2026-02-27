import math
import random
from typing import Any

"""
Sampling code modified from:
https://github.com/glouppe/beard/blob/9fb268736d195dd0c27cd0ae2915d8e00bbb4e2c/examples/applications/author-disambiguation/sampling.py
"""


def sampling(
    same_name_different_cluster: list[tuple[str, str, int | float]],
    different_name_same_cluster: list[tuple[str, str, int | float]],
    same_name_same_cluster: list[tuple[str, str, int | float]],
    different_name_different_cluster: list[tuple[str, str, int | float]],
    sample_size: int,
    balanced_homonyms_and_synonyms: bool,
    random_seed: int,
) -> list[tuple[str, str, int | float]]:
    """
    Samples pairs from the input list of pairs computed exhaustively from pair_sampling.
    Two criteria includes whether balance pairs based on positive/negative classes only
    or also consider balancing homonyms and synonyms.

    Parameters
    ----------
    same_name_different_cluster: List
        list of signature pairs (s1, s2) with same name,
        but from different clusters--> (s1, s2, 0).

    different_name_same_cluster: List
        list of signature pairs (s1, s2) with different name,
        but from same cluster--> (s1, s2, 1).

    same_name_same_cluster: List
        list of signature pairs (s1, s2) with same name,
        also from same cluster--> (s1, s2, 1).

    different_name_different_cluster: List
        list of signature pairs (s1, s2) with different name,
        also from different clusters--> (s1, s2, 0).

    sample_size: int
        The desired sample size

    balanced_homonyms_and_synonyms: bool
        False -- balance for positive and negative classes
        True -- balance for homonyms and synonyms under positive and negative classes
             as well (i.e., same_name_different_cluster, different_name_same_cluster,
             same_name_same_cluster and different_name_different_cluster)

    random_seed: int
        random seed for sampling

    Returns
    -------
    List: list of sampled signature pairs
    """

    random.seed(random_seed)

    all_candidates = (
        same_name_different_cluster
        + different_name_same_cluster
        + same_name_same_cluster
        + different_name_different_cluster
    )
    target_size = min(sample_size, len(all_candidates))
    if target_size <= 0:
        return []

    if balanced_homonyms_and_synonyms:
        buckets = [
            same_name_different_cluster,
            different_name_same_cluster,
            same_name_same_cluster,
            different_name_different_cluster,
        ]
        per_bucket_target = int(math.ceil(target_size / 4))
        sampled_buckets = []
        leftovers = []
        for bucket in buckets:
            take = min(len(bucket), per_bucket_target)
            sampled = random.sample(bucket, take)
            sampled_buckets.append(sampled)
            sampled_set = set(sampled)
            leftovers.extend([pair for pair in bucket if pair not in sampled_set])
        pairs = [pair for sampled in sampled_buckets for pair in sampled]
        if len(pairs) < target_size and len(leftovers) > 0:
            pairs.extend(random.sample(leftovers, min(target_size - len(pairs), len(leftovers))))
    else:
        positive = same_name_same_cluster + different_name_same_cluster
        negative = same_name_different_cluster + different_name_different_cluster
        pos_target = int(math.ceil(target_size / 2))
        neg_target = target_size - pos_target

        pos_sample = random.sample(positive, min(len(positive), pos_target))
        neg_sample = random.sample(negative, min(len(negative), neg_target))
        pairs = pos_sample + neg_sample

        if len(pairs) < target_size:
            pos_left = [pair for pair in positive if pair not in set(pos_sample)]
            neg_left = [pair for pair in negative if pair not in set(neg_sample)]
            leftovers = pos_left + neg_left
            pairs.extend(random.sample(leftovers, min(target_size - len(pairs), len(leftovers))))

    if len(pairs) > target_size:
        pairs = random.sample(pairs, target_size)

    return random.sample(pairs, len(pairs))


def random_sampling(possible: list[Any], sample_size: int, random_seed: int) -> list[Any]:
    """
    Randomly samples a list

    Parameters
    ----------
    possible: List
        list of things to sample
    sample_size: int
        the sample size
    random_seed: int
        the random seed

    Returns
    -------
    List: the sample from the list
    """
    random.seed(random_seed)
    return random.sample(possible, sample_size)
