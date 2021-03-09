import string
import re
from typing import List, Set, Tuple

from s2and.data import ANDData
from s2and.text import jaccard, normalize_text

# This file is helper functions for comparing aspects of author disambiguation to
# the Semantic Scholar production system as of writing

PUNCTUATION_RE = re.compile("[%s]" % re.escape(string.punctuation))
WHITESPACE = r"\s+"
STOPWORDS = {"of", "for", "and", "dept", "department", "univ", "university"}
MAX_YEAR_GAP = 10


def normalized_affiliation_tokens(s: List[str]) -> Set[str]:
    s_joined = PUNCTUATION_RE.sub(" ", " ".join(s)).strip().lower()
    s_split = set(re.split(WHITESPACE, s_joined))
    return s_split.difference(STOPWORDS)


def affiliation_fuzzy_match(cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData) -> float:
    affiliations_1 = [
        normalized_affiliation_tokens(dataset.signatures[signature].author_info_affiliations)
        for signature in cluster_candidate_1
    ]

    affiliations_2 = [
        normalized_affiliation_tokens(dataset.signatures[signature].author_info_affiliations)
        for signature in cluster_candidate_2
    ]

    affiliations_1_union = {token for affiliation in affiliations_1 for token in affiliation}
    affiliations_2_union = {token for affiliation in affiliations_2 for token in affiliation}

    affiliations_jaccard = jaccard(affiliations_1_union, affiliations_2_union)
    return affiliations_jaccard


def year_gap_is_small(cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData) -> bool:
    years_1 = set(
        [
            dataset.papers[str(dataset.signatures[signature].paper_id)].year
            for signature in cluster_candidate_1
            if dataset.papers[str(dataset.signatures[signature].paper_id)].year is not None
        ]
    )
    years_2 = set(
        [
            dataset.papers[str(dataset.signatures[signature].paper_id)].year
            for signature in cluster_candidate_2
            if dataset.papers[str(dataset.signatures[signature].paper_id)].year is not None
        ]
    )

    if len(years_1) == 0 or len(years_2) == 0:
        return True

    lowest_max_year = min(max(years_1), max(years_2))
    highest_min_year = max(min(years_1), min(years_2))

    is_small = (highest_min_year - lowest_max_year) < MAX_YEAR_GAP

    return is_small


def has_year_gap(cluster_candidate_1: List[str], dataset: ANDData) -> bool:
    years_1 = list(
        set(
            [
                dataset.papers[str(dataset.signatures[signature].paper_id)].year
                for signature in cluster_candidate_1
                if dataset.papers[str(dataset.signatures[signature].paper_id)].year is not None
            ]
        )
    )
    years_1 = sorted(years_1)
    for i in range(len(years_1) - 1):
        if years_1[i + 1] - years_1[i] >= MAX_YEAR_GAP:
            return True
    return False


def trusted_ids_are_compatible(
    cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData, orcid_only: bool = False
) -> bool:
    orcid_1 = set()
    orcid_2 = set()
    if not orcid_only:
        dblp_1 = set()
        dblp_2 = set()

    for signature_id in cluster_candidate_1:
        signature = dataset.signatures[signature_id]
        if signature.sourced_author_source == "ORCID":
            orcid_1.update(signature.sourced_author_ids)
        elif signature.sourced_author_source == "DBLP" and not orcid_only:
            dblp_1.update(signature.sourced_author_ids)

    for signature_id in cluster_candidate_2:
        signature = dataset.signatures[signature_id]
        if signature.sourced_author_source == "ORCID":
            orcid_2.update(signature.sourced_author_ids)
        elif signature.sourced_author_source == "DBLP" and not orcid_only:
            dblp_2.update(signature.sourced_author_ids)

    orcid_ok = orcid_2.issubset(orcid_1) if len(orcid_1) > len(orcid_2) else orcid_1.issubset(orcid_2)
    if orcid_only:
        return orcid_ok
    else:
        dblp_ok = dblp_2.issubset(dblp_1) if len(dblp_1) > len(dblp_2) else dblp_1.issubset(dblp_2)
        return orcid_ok and dblp_ok


def emails_match_exactly(cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData) -> bool:
    emails_1 = set(
        [
            dataset.signatures[signature].author_info_email
            for signature in cluster_candidate_1
            if dataset.signatures[signature].author_info_email is not None
        ]
    )
    emails_2 = set(
        [
            dataset.signatures[signature].author_info_email
            for signature in cluster_candidate_2
            if dataset.signatures[signature].author_info_email is not None
        ]
    )

    if len(emails_1) != 1 or len(emails_2) != 1:
        return False

    single_element_exact_match = emails_1 == emails_2
    return single_element_exact_match


def trusted_ids_match_exactly(
    cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData, orcid_only: bool = False
) -> bool:
    orcid_1 = set()
    orcid_2 = set()
    if not orcid_only:
        dblp_1 = set()
        dblp_2 = set()

    for signature_id in cluster_candidate_1:
        signature = dataset.signatures[signature_id]
        if signature.sourced_author_source == "ORCID":
            orcid_1.update(signature.sourced_author_ids)
        elif signature.sourced_author_source == "DBLP" and not orcid_only:
            dblp_1.update(signature.sourced_author_ids)

    for signature_id in cluster_candidate_2:
        signature = dataset.signatures[signature_id]
        if signature.sourced_author_source == "ORCID":
            orcid_2.update(signature.sourced_author_ids)
        elif signature.sourced_author_source == "DBLP" and not orcid_only:
            dblp_2.update(signature.sourced_author_ids)

    if not orcid_only:
        if len(dblp_1) != 1 and len(dblp_2) == 1:
            return False

        if len(dblp_1) == 1 and len(dblp_2) != 1:
            return False

        if len(dblp_1) == 1 and len(dblp_2) == 1 and dblp_1 == dblp_2:
            return True

    if len(orcid_1) != 1 and len(orcid_2) == 1:
        return False

    if len(orcid_1) == 1 and len(orcid_2) != 1:
        return False

    if len(orcid_1) == 1 and len(orcid_2) == 1 and orcid_1 == orcid_2:
        return True

    return False


def names_are_compatible(cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData) -> bool:
    last_names_first_initials_1 = set()
    full_first_names_1 = set()
    full_middle_names_1 = set()
    middle_initials_1 = set()
    for signature_id in cluster_candidate_1:
        signature = dataset.signatures[signature_id]
        first = normalize_text(signature.author_info_first)
        middle = normalize_text(signature.author_info_middle)
        last = normalize_text(signature.author_info_last)

        if first is not None and len(first) > 0:
            if len(first) > 1:
                full_first_names_1.add(first)

            if last is not None and len(last) > 0:
                last_names_first_initials_1.add(first[0] + " " + last)

        if middle is not None and len(middle) > 0:
            if len(middle) > 1:
                full_middle_names_1.add(middle)

            middle_initials_1.add(middle[0])

    last_names_first_initials_2 = set()
    full_first_names_2 = set()
    full_middle_names_2 = set()
    middle_initials_2 = set()
    for signature_id in cluster_candidate_2:
        signature = dataset.signatures[signature_id]
        first = normalize_text(signature.author_info_first)
        middle = normalize_text(signature.author_info_middle)
        last = normalize_text(signature.author_info_last)

        if first is not None and len(first) > 0:
            if len(first) > 1:
                full_first_names_2.add(first)

            if last is not None and len(last) > 0:
                last_names_first_initials_2.add(first[0] + " " + last)

        if middle is not None and len(middle) > 0:
            if len(middle) > 1:
                full_middle_names_2.add(middle)

            middle_initials_2.add(middle[0])

    first_last_ok = (
        last_names_first_initials_2.issubset(last_names_first_initials_1)
        if len(last_names_first_initials_1) > len(last_names_first_initials_2)
        else last_names_first_initials_1.issubset(last_names_first_initials_2)
    )
    full_first_ok = (
        full_first_names_2.issubset(full_first_names_1)
        if len(full_first_names_1) > len(full_first_names_2)
        else full_first_names_1.issubset(full_first_names_2)
    )
    full_middle_ok = (
        full_middle_names_2.issubset(full_middle_names_1)
        if len(full_middle_names_1) > len(full_middle_names_2)
        else full_middle_names_1.issubset(full_middle_names_2)
    )
    middle_initial_ok = (
        middle_initials_2.issubset(middle_initials_1)
        if len(middle_initials_1) > len(middle_initials_2)
        else middle_initials_1.issubset(middle_initials_2)
    )
    return first_last_ok and full_first_ok and full_middle_ok and middle_initial_ok


def sergeys_rule(
    cluster_candidate_1: List[str], cluster_candidate_2: List[str], dataset: ANDData, name_tuples: Set[Tuple[str, str]]
) -> bool:
    for signature_id_a in cluster_candidate_1:
        first_a = dataset.signatures[signature_id_a].author_info_first_normalized_without_apostrophe
        for signature_id_b in cluster_candidate_2:
            first_b = dataset.signatures[signature_id_b].author_info_first_normalized_without_apostrophe
            prefix = first_a.startswith(first_b) or first_b.startswith(first_a)
            known_alias = (first_a, first_b) in name_tuples
            if not prefix and not known_alias:
                return False
    return True
