from typing import List, Tuple, Union, Any

import random
import math


"""
Sampling code modified from:
https://github.com/glouppe/beard/blob/9fb268736d195dd0c27cd0ae2915d8e00bbb4e2c/examples/applications/author-disambiguation/sampling.py
"""


def sampling(
    same_name_different_cluster: List[Tuple[str, str, Union[int, float]]],
    different_name_same_cluster: List[Tuple[str, str, Union[int, float]]],
    same_name_same_cluster: List[Tuple[str, str, Union[int, float]]],
    different_name_different_cluster: List[Tuple[str, str, Union[int, float]]],
    sample_size: int,
    balanced_homonyms_and_synonyms: bool,
    random_seed: int,
) -> List[Tuple[str, str, Union[int, float]]]:
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

    if balanced_homonyms_and_synonyms:
        same_name_different_cluster_pairs = random.sample(
            same_name_different_cluster,
            min(len(same_name_different_cluster), math.ceil(sample_size / 4)),
        )
        different_name_same_cluster_pairs = random.sample(
            different_name_same_cluster,
            min(len(different_name_same_cluster), math.ceil(sample_size / 4)),
        )
        same_name_same_cluster_pairs = random.sample(
            same_name_same_cluster,
            min(len(same_name_same_cluster), math.ceil(sample_size / 4)),
        )
        different_name_different_cluster_pairs = random.sample(
            different_name_different_cluster,
            min(len(different_name_different_cluster), math.ceil(sample_size / 4)),
        )
        pairs = (
            same_name_different_cluster_pairs
            + different_name_same_cluster_pairs
            + same_name_same_cluster_pairs
            + different_name_different_cluster_pairs
        )
    else:
        positive = same_name_same_cluster + different_name_same_cluster
        negative = same_name_different_cluster + different_name_different_cluster
        pairs = random.sample(positive, min(len(positive), math.ceil(sample_size / 2))) + random.sample(
            negative, min(len(negative), math.ceil(sample_size / 2))
        )

    return random.sample(pairs, len(pairs))


def random_sampling(possible: List[Any], sample_size: int, random_seed: int) -> List[Any]:
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
