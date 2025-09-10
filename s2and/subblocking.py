import random
import os
import json
import numpy as np
import pandas as pd
import logging
from itertools import combinations
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import genieclust
from s2and.consts import SPECTER_DIM, PROJECT_ROOT_PATH
from s2and.text import same_prefix_tokens


logger = logging.getLogger("s2and")


with open(os.path.join(PROJECT_ROOT_PATH, "data", "first_k_letter_counts_from_orcid.json"), "r") as f:
    FIRST_K_LETTER_COUNTS = json.load(f)


def cluster_with_specter(signature_ids, anddata, target_subblock_size=10000):
    """Helper function to cluster signature ids into subblocks using specter embeddings.
    Also tries to add simple embeddings of co-author blocks and affiliation n-grams.

    Args:
        signature_ids (list[str/int]): signature_ids
        anddata (s2and.data.ANDData): the anddata dataset
        target_subblock_size (int, optional): The desired maximum subblock size.
            If any of the resulting clusters are bigger than this, we chop them up randomly.
            Defaults to 10000.

    Returns:
        clusters: dict with keys as cluster_ids and values as list of signature_ids.
    """
    if len(signature_ids) == 0:
        return {}
    elif len(signature_ids) < target_subblock_size:
        return {"0": signature_ids}

    # extract all the specter stuff in order of the signatures
    X_specter = np.array(
        [
            anddata.specter_embeddings.get(str(anddata.signatures[i].paper_id), np.zeros(SPECTER_DIM))
            for i in signature_ids
        ]
    )

    try:
        # same for the co-author blocks
        X = MultiLabelBinarizer(sparse_output=True).fit_transform(
            [list(anddata.signatures[i].author_info_coauthor_blocks) for i in signature_ids]
        )
        X_svd = TruncatedSVD(n_components=SPECTER_DIM).fit_transform(X)

        # same for affiliations
        X = TfidfVectorizer(preprocessor=None, analyzer=lambda x: x).fit_transform(
            [list(anddata.signatures[i].author_info_affiliations_n_grams.keys()) for i in signature_ids]
        )
        X_svd2 = TruncatedSVD(n_components=SPECTER_DIM).fit_transform(X)

        # all together now
        X = X_specter + np.mean([X_svd, X_svd2], axis=0)
    except:
        X = X_specter

    # how many subblocks do we want given this data and target subblock size?
    # should be at least 2 if we end up here otherwise there is no point
    num_desired_subblocks = int(np.ceil(len(signature_ids) / target_subblock_size))

    # this can fail when X are all zeros
    try:
        g = genieclust.Genie(n_clusters=num_desired_subblocks, gini_threshold=0.01, exact=False)
        labels = g.fit_predict(X)
    except:
        labels = np.zeros(len(signature_ids), dtype=int)

    subblocks = defaultdict(list)
    for sig_id, label in zip(signature_ids, labels):
        subblocks[label].append(sig_id)
    # if any subblock is above the target size, just chop it up randomly into pieces that are below the target size
    for label, subblock in list(subblocks.items()):
        if len(subblock) > target_subblock_size:
            random.shuffle(subblock)
            num_new_subblocks = int(np.ceil(len(subblock) / target_subblock_size))
            c = 0
            for i in range(num_new_subblocks):
                subblocks[f"{label}.{i}"] = subblock[i * target_subblock_size : (i + 1) * target_subblock_size]
                c += len(subblocks[f"{label}.{i}"])
            del subblocks[label]

    # assert that the subblocks has a complete clustering of the input signature_ids
    assert sum([len(subblock) for subblock in subblocks.values()]) == len(signature_ids)

    return dict(subblocks)


def subdivide_helper(names, signature_ids, maximum_size, starting_k=2):
    """Helper function to subdivide a list of names into subblocks of maximum_size.
    Uses the first k letters of the names to subdivide. If the subblocks are still too big,
    then it will subdivide further by increasing k. Keeps going until the maximum_size is reached.
    If the maximum_size is reached and there are still some names left over, then those names
    will be put into their own subblock and returned separately.

    Args:
        names (list of strings): the names to subdivide
        signature_ids (list[str/int]): the signature_ids corresponding to the names
        maximum_size (int): the maximum size of each subblock allowed
        starting_k (int, optional): The starting k to use for the first subdivision.
            Defaults to 2.

    Returns:
        output: dict with keys as subblock names and values as list of signature_ids
        output_cant_subdivide: dict with keys as subblock names and values as list of signature_ids
            that cant be subdivided further
    """
    # start with 2 letters only, then subdivide further to 3 letters, etc until the maximum_size is reached
    n_signature_ids = len(signature_ids)
    if n_signature_ids == 0:
        return {}, {}
    output = {}
    output_cant_subdivide = {}
    k = starting_k
    max_len = max([len(name) for name in names])
    clean_break = False
    for k in range(starting_k, max_len + 1):
        # note: any time we take something like XYZ and make it into XYZA, XYZB, ...
        # we will have some leftover ones that are just XYZ. those will end up in their own subblock
        names_up_to_k = np.array([name[0:k] for name in names])
        # use Series.value_counts to avoid the deprecated pd.value_counts API
        counts_up_to_k = pd.Series(names_up_to_k).value_counts()
        # find the ones that are a good size, and then take the rest and subdivide further
        good_size_flag = counts_up_to_k < maximum_size
        counts_up_to_k_good_size = counts_up_to_k[good_size_flag]
        # the case where at this point *all* the newly made subblocks are too big
        # so it is a dead-end
        if len(counts_up_to_k_good_size) == 0:
            for name in counts_up_to_k.index:
                flag = names_up_to_k == name
                output_cant_subdivide[name] = signature_ids[flag]
            clean_break = True
            break
        # store each subblock in output
        for name in counts_up_to_k_good_size.index:
            flag = names_up_to_k == name
            output[name] = signature_ids[flag]
        # take the rest and subdivide further
        bad_names = set(counts_up_to_k[counts_up_to_k > maximum_size].index)
        bad_size_flag = np.array([i[0:k] in bad_names for i in names])
        names = names[bad_size_flag]
        signature_ids = signature_ids[bad_size_flag]
        k += 1
    # last ditch clean-up in case things didn't work out
    if len(names) > 0 and clean_break == False:
        output_cant_subdivide["final"] = signature_ids
    # assert that the combo of the output and output_cant_subdivide is a complete clustering of the input signature_ids
    assert (
        sum([len(subblock) for subblock in output.values()])
        + sum([len(subblock) for subblock in output_cant_subdivide.values()])
        == n_signature_ids
    )
    return output, output_cant_subdivide


def make_subblocks(signature_ids, anddata, maximum_size=7500, first_k_letter_counts_sorted=FIRST_K_LETTER_COUNTS):
    """Splits a list of signature IDs into subblocks based on name attributes.

    This function takes a list of signature IDs and splits them into subblocks of maximum_size.
    It first splits by first name initial letter. Then it recursively splits any subblocks larger than
    maximum_size using middle names and the SPECTER clustering algorithm. Finally, it merges any subblocks
    smaller than maximum_size that share name attributes.

    There is a special case for ORCIDs: we make sure that signatures with the same ORCID end up
    in the same subblock

    Args:
        signature_ids (list[str/int]): List of signature IDs.
        anddata (s2and.data.ANDData): Contains name attribute data for the signatures.
        maximum_size (int): Maximum size of any subblock. Default is 7500.
        first_k_letter_counts_sorted (dict): Dictionary of name letter counts, used for merging subblocks.
            Already included in the package. Default is FIRST_K_LETTER_COUNTS, which is imported
            in this file.

    Returns:
        dict: Dictionary of subblock keys mapped to lists of signature IDs.
    """
    logger.info("Beginning subblocking...")
    signature_ids = np.array(signature_ids)
    first_names = np.array(
        [anddata.signatures[i].author_info_first_normalized_without_apostrophe for i in signature_ids]
    )
    middle_names = np.array(
        [anddata.signatures[i].author_info_middle_normalized_without_apostrophe for i in signature_ids]
    )

    # set aside those that are only 1 letter long for a different treatment
    single_letter_first_names_flag = np.array([len(first_name) <= 1 for first_name in first_names])

    # first letter is
    first_letter = "?"  # could happen if all the first names are missing
    for name in first_names:
        if len(name) > 0:
            first_letter = name[0]
            break

    # first pass through the more-than-one-letter first names
    logger.info("First pass through the more-than-one-letter first names")
    output, output_cant_subdivide = subdivide_helper(
        first_names[~single_letter_first_names_flag], signature_ids[~single_letter_first_names_flag], maximum_size
    )

    # for each block in output_cant_subdivide, we need to subdivide it further using middle names
    if len(output_cant_subdivide) > 0:
        logger.info(
            "Subdividing the more-than-one-letter first names that could not be subdivided further using middle names"
        )
    output_for_specter = {}
    for key, sig_ids_loop in output_cant_subdivide.items():
        middle_names_loop = np.array(
            [anddata.signatures[i].author_info_middle_normalized_without_apostrophe for i in sig_ids_loop]
        )
        output_loop, output_cant_subdivide_loop = subdivide_helper(
            middle_names_loop, sig_ids_loop, maximum_size, starting_k=1
        )
        # the key in output loop should be pre-pended by the loop key
        for key_loop in list(output_loop.keys()):
            output_loop[key + "|middle=" + str(key_loop)] = output_loop.pop(key_loop)
        for key_loop in list(output_cant_subdivide_loop.keys()):
            output_cant_subdivide_loop[key + "|middle=" + str(key_loop)] = output_cant_subdivide_loop.pop(key_loop)
        # now update the output
        output.update(output_loop)
        output_for_specter.update(output_cant_subdivide_loop)

    # deal with the single (or zero) letter first names
    if len(first_names[single_letter_first_names_flag]) < maximum_size:
        if np.mean(single_letter_first_names_flag) > 0:
            output[first_letter] = signature_ids[single_letter_first_names_flag]
    else:
        logger.info("Subdividing the too-big single letter subblock using middle names")
        output_single_letter_first_name, output_cant_subdivide_single_letter_first_name = subdivide_helper(
            middle_names[single_letter_first_names_flag],
            signature_ids[single_letter_first_names_flag],
            maximum_size,
            starting_k=1,
        )
        # modify the key to indicate what this is
        for key in list(output_single_letter_first_name.keys()):
            output_single_letter_first_name[f"{first_letter}|middle=" + str(key)] = output_single_letter_first_name.pop(
                key
            )
        for key in list(output_cant_subdivide_single_letter_first_name.keys()):
            output_cant_subdivide_single_letter_first_name[f"{first_letter}|middle=" + str(key)] = (
                output_cant_subdivide_single_letter_first_name.pop(key)
            )
        output.update(output_single_letter_first_name)
        output_for_specter.update(
            output_cant_subdivide_single_letter_first_name
        )  # since it already went through the middle name step

    # for each subblock that STILL can't be subdivided, we must use SPECTER
    # which also does totally random sub-blocking in case things went awry
    if len(output_for_specter) > 0:
        logger.info(
            "Subdividing the subblocks that could not be subdivided via middle names using SPECTER (and random subblocking)"
        )
    for key, sig_ids_loop in output_for_specter.items():
        if len(sig_ids_loop) <= maximum_size:
            # edge case where the subblock is already fine
            output_loop[key] = sig_ids_loop
        else:
            output_loop = {}
            specter_clustering = cluster_with_specter(sig_ids_loop, anddata, target_subblock_size=maximum_size)
            # prepend the key to the specter_clustering keys
            for key_loop in list(specter_clustering.keys()):
                output_loop[key + "|specter=" + str(key_loop)] = specter_clustering.pop(key_loop)
        output.update(output_loop)

    """
    Merging too small subblocks back up to maximum_size
    If we found that the subblock Jame* was too big, afterwards some of the subblocks
    like James*, Jamen*, Jamek* etc may be too small and could be joined again while
    still being below the maximum size.
    
    This is done by looking at all the subblocks that are small enough, and then
    checking (a) they are plausible to be merged (b) their join is small enough.
    
    First step is to find candidates for merging.
    """
    logger.info("Starting to merge subblocks. First step is to find candidates for merging.")
    small_enough_keys = [k for k, v in output.items() if len(v) < maximum_size]
    # for each pair of keys in small_enough, look up the count in first_k_letter_counts_sorted
    # and keep only pairs where their sum is less than maximum subblock size
    # then sort descending by the count
    small_enough_pairs_counts = []
    for pair in list(combinations(small_enough_keys, 2)):
        # the addition of this pair can't be greater than the maximum size
        if len(output[pair[0]]) + len(output[pair[1]]) < maximum_size:
            pair_0_split = pair[0].split("|")
            pair_1_split = pair[1].split("|")

            first_name_1 = pair_0_split[0]
            first_name_2 = pair_1_split[0]

            if len(pair_0_split) > 1:
                middle_name_1 = pair_0_split[1].split("=")[1]
            else:
                middle_name_1 = None

            if len(pair_1_split) > 1:
                middle_name_2 = pair_1_split[1].split("=")[1]
            else:
                middle_name_2 = None

            # for more than single-letter first names
            # we consider merging the subblocks if they overlapping first k letters
            # however this may be not necessary as the constraints disallow
            # situations where not (a.startswith(b) or b.startswith(a))
            if len(first_name_1) > 1 and len(first_name_2) > 1:
                name_for_splits_1 = first_name_1
                name_for_splits_2 = first_name_2
            # then we have the situation where we have single letter first names and available middle name
            # here we'll use the middle names for the proposed merges
            elif (
                len(first_name_1) == 1
                and len(first_name_2) == 1
                and middle_name_1 is not None
                and middle_name_2 is not None
            ):
                name_for_splits_1 = middle_name_1
                name_for_splits_2 = middle_name_2
            # we don't really want to mix cases where one has 2 or more first name letters and the other doesn't
            # also it's weird when one has a middle name and the other doesn't (and they're both single letter)
            # so just skipping them all
            else:
                continue

            # if names are the same, then we give this a very high artificial count
            # and the count for this will be very high
            if name_for_splits_1 == name_for_splits_2:
                # we also have to add overlap between the middle names if they exist
                # to prioritize James W.E. to be joined with James W over being joined with James E
                if middle_name_1 is not None and middle_name_2 is not None:
                    score = 0
                    for i in range(1, len(middle_name_1)):
                        if middle_name_1[:i] == middle_name_2[:i]:
                            score = i
                else:
                    score = 0
                small_enough_pairs_counts.append((pair, 1e10 + score))
            # the name tuples allow the situation where prefixes match in either direction
            elif same_prefix_tokens(name_for_splits_1, name_for_splits_2):
                score = min(len(name_for_splits_1), len(name_for_splits_2))
                small_enough_pairs_counts.append((pair, 1e5 + score))
            # the other option is that the names are different but we have counts
            else:
                # TODO(s2and): Temporary compatibility tweak for hyphen-preserving first names.
                # The ORCID-derived first_k_letter_counts were generated with legacy normalization.
                # To preserve utility without regenerating, probe counts using token before first space.
                # Consider removing this once counts are regenerated with new logic.
                lookup_1 = name_for_splits_1.split(" ")[0]
                lookup_2 = name_for_splits_2.split(" ")[0]
                if lookup_1 in first_k_letter_counts_sorted and lookup_2 in first_k_letter_counts_sorted[lookup_1]:
                    small_enough_pairs_counts.append((pair, first_k_letter_counts_sorted[lookup_1][lookup_2]))

    small_enough_pairs_sorted = sorted(small_enough_pairs_counts, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
    # now we go down the list and merge until we reach merged subblocks not above maximum size
    # it's possible that when we merge subblock A and B, the resulting subblock is still below thresh
    # and then it's legal to merge A/B with C, so we have to keep track of all that
    merging_log = {}  # what we will actually merge after we're done. cluster id -> set of keys
    inverse_merging_log = {}  # need this to see if things are in the same subblock already
    cluster_id = 0
    # we'll use this to see how many tuples a key appears in
    # and if a proposed merge appears in more than one
    # then we have a problem and it shouldn't occur
    logger.info(f"Number of small enough pairs to consider for subblock merging: {len(small_enough_pairs_sorted)}")
    logger.info("Merging subblocks...")
    for pair, _ in small_enough_pairs_sorted:
        # see where both parts of the pair are in the merging log
        pair_1_cluster_id = inverse_merging_log.get(pair[0], None)
        pair_2_cluster_id = inverse_merging_log.get(pair[1], None)
        # if neither are in the log, then we can just add them to it
        if pair_1_cluster_id is None and pair_2_cluster_id is None:
            merging_log[cluster_id] = set(pair)
            inverse_merging_log[pair[0]] = cluster_id
            inverse_merging_log[pair[1]] = cluster_id
            cluster_id += 1
        # if both are in the merging log but they have the SAME cluster id, then we don't need to do anything
        elif pair_1_cluster_id is not None and pair_2_cluster_id is not None and pair_1_cluster_id == pair_2_cluster_id:
            continue
        else:
            # if both are in the merging log but they have DIFFERENT cluster ids
            # then we should check if their clusters can be joined legally
            if (
                pair_1_cluster_id is not None
                and pair_2_cluster_id is not None
                and pair_1_cluster_id != pair_2_cluster_id
            ):
                proposed_cluster = merging_log[pair_1_cluster_id].union(merging_log[pair_2_cluster_id])
            # if only one is in the merging log, then we should check if the other can be added to it legally
            elif pair_1_cluster_id is not None and pair_2_cluster_id is None:
                proposed_cluster = merging_log[pair_1_cluster_id].union(set(pair))
            # and vice versa
            elif pair_1_cluster_id is None and pair_2_cluster_id is not None:
                proposed_cluster = merging_log[pair_2_cluster_id].union(set(pair))
            else:
                raise ValueError("This should never happen")
            size_of_proposed = sum([len(output[k]) for k in proposed_cluster])
            if size_of_proposed <= maximum_size:
                if pair_1_cluster_id is not None:
                    merging_log[pair_1_cluster_id] = proposed_cluster
                    if pair_2_cluster_id is not None:
                        del merging_log[pair_2_cluster_id]
                    for k in proposed_cluster:
                        inverse_merging_log[k] = pair_1_cluster_id
                else:
                    merging_log[pair_2_cluster_id] = proposed_cluster
                    if pair_1_cluster_id is not None:
                        del merging_log[pair_1_cluster_id]
                    for k in proposed_cluster:
                        inverse_merging_log[k] = pair_2_cluster_id

    # double check that nothing weird happened: each key should only appear in one subblock
    counter_of_keys = defaultdict(int)
    for keys_to_merge in merging_log.values():
        for k in keys_to_merge:
            counter_of_keys[k] += 1

    assert all([v == 1 for v in counter_of_keys.values()])

    # now perform the actual merges
    for keys_to_merge in merging_log.values():
        key_of_keys = ", ".join(sorted(list(keys_to_merge)))
        signature_ids_stacked = np.hstack([output[k] for k in keys_to_merge])
        output[key_of_keys] = signature_ids_stacked
        # delete what was merged
        for k in keys_to_merge:
            del output[k]

    # values in output should be lists
    for k in list(output.keys()):
        output[k] = list(output[k])

    # final step: we need to make sure that sets of signature_ids with the same ORCID are in the same subblock
    # approach: find all the signature_ids with ORCIDs that appear more than once
    # AND are in different subblocks
    # then move around the individual signatures so that they are in the same subblock
    # 1: get a mapping from orcid -> (signature_id, subblock_id)
    orcid_to_sig_id_subblock_id = defaultdict(list)
    for subblock_id, sig_ids in output.items():
        for sig_id in sig_ids:
            orcid = anddata.signatures[sig_id].author_info_orcid
            if orcid is not None:
                orcid_to_sig_id_subblock_id[orcid].append((sig_id, subblock_id))
    # 2: for each orcid, if there is more than one unique subblock_id, then we need to move signature_ids around
    for orcid, sig_id_subblock_id in orcid_to_sig_id_subblock_id.items():
        unique_subblock_ids = list(set([i[1] for i in sig_id_subblock_id]))
        if len(unique_subblock_ids) > 1:
            # 3: pick a subblock that isn't already maximum size
            # if they are all maximum size, then pick the first one
            subblock_sizes = [len(output[k]) for k in unique_subblock_ids]
            # try to move into subblocks that
            # (a) are not SPECTER subblocks
            # (b) have more than 1 letter
            unique_subblock_ids = sorted(
                unique_subblock_ids,
                key=lambda x: x.count("specter") * 10 + x.count("|"),
            )
            if all([i == maximum_size for i in subblock_sizes]):
                subblock_id_to_move_to = unique_subblock_ids[0]
            else:
                subblock_id_to_move_to = [k for k in unique_subblock_ids if len(output[k]) < maximum_size][0]
            # 4: move the signature_ids around so that they are all in the same subblock
            # we take ONLY the signature ids that are not in the chosen subblock_id
            # and move them there, removing from their original subblock
            for sig_id, original_subblock_id in sig_id_subblock_id:
                if original_subblock_id != subblock_id_to_move_to:
                    output[subblock_id_to_move_to].append(sig_id)
                    output[original_subblock_id].remove(sig_id)
                    # unlikely, but if we emptied out the original subblock, then delete it
                    if len(output[original_subblock_id]) == 0:
                        del output[original_subblock_id]

    # let's assert that we have done a complete partition
    assert set(np.hstack([output[k] for k in output])) == set(signature_ids)

    # before the end, makes sure everything is a standard list
    for k in list(output.keys()):
        output[k] = list(output[k])

    average_subblock_length = np.mean([len(output[k]) for k in output])
    logger.info(
        f"Done subblocking. There are {len(output)} subblocks with an average of {average_subblock_length} signatures each."
    )
    return output
