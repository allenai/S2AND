"""
Note: This script won't run because it relies on an internal Semantic Scholar package
called pys2, and is here for documentation of how the prefix counts for subblocking were built.

TODO: rerun this when we update how names are normalized
"""

import os
import json
from collections import Counter
from itertools import combinations
from pys2.pys2 import _evaluate_redshift_query
from s2and.text import normalize_text, NAME_PREFIXES
from s2and.consts import PROJECT_ROOT_PATH

"""
Step 1: Get orcid name pairs from our internal databases
"""

query = """
select p.year, p.inserted paper_inserted, 
     pae.corpus_paper_id, pae.source, pae.orcid,  pae.position, pae.first_name, pa.middle, pae.last_name, 
     pa.corpus_author_id, au.ai2_id, pa.inserted pa_inserted, pa.updated pa_updated, pa.cluster_block_key, pa.model_version, pa.clusterer
from content_ext.paper_authors_orcids pae 
join content_ext.papers p 
     on pae.corpus_paper_id=p.corpus_paper_id  
join content_ext.paper_authors pa 
     on pae.corpus_paper_id=pa.corpus_paper_id 
     and pae.position=pa.position+1 and lower(pae.last_name)=lower(pa.last)
join content_ext.authors au 
   on pa.corpus_author_id=au.corpus_author_id
where pae.source in ('Crossref')
;
"""

df_all = _evaluate_redshift_query(query)

cache = {}  # type: ignore


def normalize_names(row):
    """This is basically the same as what's in row 456 and on in s2and/data.py
    TODO: if that changes due to, say, how dashes are treated, we have to rerun this
    """
    first = row["first_name"]
    middle = row["middle"]

    if (first, middle) in cache:
        return cache[(first, middle)]

    first_normalized_without_apostrophe = normalize_text(first or "", special_case_apostrophes=True)

    middle_normalized = normalize_text(middle or "")

    first_middle_normalized_split_without_apostrophe = (
        first_normalized_without_apostrophe + " " + middle_normalized
    ).split(" ")
    if first_middle_normalized_split_without_apostrophe[0] in NAME_PREFIXES:
        first_middle_normalized_split_without_apostrophe = first_middle_normalized_split_without_apostrophe[1:]

    author_info_first_normalized_without_apostrophe = first_middle_normalized_split_without_apostrophe[0]
    author_info_middle_normalized_without_apostrophe = " ".join(first_middle_normalized_split_without_apostrophe[1:])
    cache[(first, middle)] = (
        author_info_first_normalized_without_apostrophe,
        author_info_middle_normalized_without_apostrophe,
    )

    return author_info_first_normalized_without_apostrophe, author_info_middle_normalized_without_apostrophe


normed_first_second = df_all.apply(normalize_names, axis=1, result_type="expand")
df_all.loc[:, ["first_norm", "middle_norm"]] = normed_first_second.values
orcids = df_all[["cluster_block_key", "orcid", "first_norm", "middle_norm"]]

"""
Step 2: Get name pairs that are included in S2AND
"""
name_tuples = set()
with open(os.path.join(PROJECT_ROOT_PATH, "data", "s2and_name_tuples_filtered.txt"), "r") as f2:
    for line in f2:
        line_split = line.strip().split(",")
        name_tuples.add((line_split[0], line_split[1]))

"""
Step 3: Compute first k letter pair and how often they occur for both data sources
and combine them
"""

# orcid data
k_values = (2, 3, 4, 5)  # only care up to first 5 letters
orcid_first_k_letter_counts = Counter()  # type: ignore


# in each group, take all pairs of unique names and then count the number of times each first k letter combination occurs
# (name_1[:k], name_2[:k]) for k in range(2, 6) where k is the outer dictionary key
def group_update(group, k_values=k_values):
    names = [i for i in group["first_norm"].unique() if type(i) == str]
    if len(names) > 1:
        for name1, name2 in combinations(names, 2):
            if name1[0] == name2[0]:
                pairs = set()
                for k in k_values:
                    for j in k_values:
                        pair = (name1[:k], name2[:j])
                        if pair[0] != pair[1]:
                            pairs.add(pair)
                orcid_first_k_letter_counts.update(list(pairs))


groups = orcids.groupby("orcid")
groups.apply(group_update)

# name tuples data
name_tuples_first_k_letter_counts = Counter()  # type: ignore
for name1, name2 in name_tuples:
    if name1[0] == name2[0] and (name1, name2):
        pairs = set()
        for k in k_values:
            for j in k_values:
                pair = (name1[:k], name2[:j])
                if pair[0] != pair[1] and not (pair[0].startswith(pair[1]) or pair[1].startswith(pair[0])):
                    pairs.add(pair)
        name_tuples_first_k_letter_counts.update(list(pairs))

# we will have a special subblock merge rule for a.starts_with(b) and b.starts_with(a)
# where a and b are names so we can just remove all of those from the orcid_first_k_letter_counts
# to save space
orcid_first_k_letter_counts_filtered = {}
for (name1, name2), count in orcid_first_k_letter_counts.items():
    if not (name1.startswith(name2) or name2.startswith(name1)):
        # we also have a filter on this one where count has to be greater than 10
        if count >= 10:
            orcid_first_k_letter_counts_filtered[(name1, name2)] = count

# can't save a json where the keys are tuples so make a nested dict:
# outer key: tuple[0], inner key: tuple[1], value: count
# remove everything with count < 10 as it is too noisy
merged_first_k_letter_counts_sorted = {}  # type: ignore
for name_tuple, count in orcid_first_k_letter_counts_filtered.items():
    if name_tuple[0] not in merged_first_k_letter_counts_sorted:
        merged_first_k_letter_counts_sorted[name_tuple[0]] = {}
    merged_first_k_letter_counts_sorted[name_tuple[0]][name_tuple[1]] = count

print(len(merged_first_k_letter_counts_sorted))

# now add from the name_tuples but the count has to change a bit
# as these are just not as high numbers as the orcid ones
already_in = 0
for name_tuple, count in name_tuples_first_k_letter_counts.items():
    if count >= 2:
        if name_tuple[0] not in merged_first_k_letter_counts_sorted:
            merged_first_k_letter_counts_sorted[name_tuple[0]] = {}
        if name_tuple[1] not in merged_first_k_letter_counts_sorted[name_tuple[0]]:
            merged_first_k_letter_counts_sorted[name_tuple[0]][name_tuple[1]] = count
        else:
            already_in += 1

# save it
with open(os.path.join(PROJECT_ROOT_PATH, "data", "first_k_letter_counts_from_orcid.json"), "w") as f:
    json.dump(merged_first_k_letter_counts_sorted, f)
