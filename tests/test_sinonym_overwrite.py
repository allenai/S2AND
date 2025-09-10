import math
from typing import Dict, Any, Tuple

import pytest

from s2and.data import Signature, compute_sinonym_overwrite_allowlist


def make_sig(sig_id: str, paper_id: int, pos: int, first: str, middle: str, last: str) -> Tuple[str, Signature]:
    # Build a Signature with only the fields used by gating logic
    sig = Signature(
        author_info_first=first,
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle=middle,
        author_info_middle_normalized_without_apostrophe=None,
        author_info_last_normalized=None,
        author_info_last=last,
        author_info_suffix_normalized=None,
        author_info_suffix=None,
        author_info_first_normalized=None,
        author_info_coauthors=None,
        author_info_coauthor_blocks=None,
        author_info_full_name=None,
        author_info_affiliations=[],
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
        author_info_email=None,
        author_info_orcid=None,
        author_info_name_counts=None,
        author_info_position=pos,
        author_info_block="",
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=paper_id,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id=sig_id,
    )
    return sig_id, sig


def make_parsed(given: str, surname: str) -> Dict[str, Any]:
    return {
        "given_tokens": [t for t in given.split("-") if t],
        "surname_tokens": [t for t in surname.split("-") if t],
        "original_compound_surname": None,
        "middle_tokens": [],
    }


def build_per_paper(*entries: Tuple[int, int, str, str]) -> Dict[str, Dict[int, Any]]:
    """
    entries of (paper_id, pos, sinonym_given, sinonym_surname)
    """
    m: Dict[str, Dict[int, Any]] = {}
    for pid, pos, given, surname in entries:
        m.setdefault(str(pid), {})[pos] = make_parsed(given, surname)
    return m


def test_multi_author_ratio_pass():
    # Name: "ping zhang" (x=3, y=1) with ratio=3.0 => 3 >= 3*1 -> yes
    sigs = dict([
        make_sig("s1", 1, 0, "ping", "", "zhang"),  # target on multi-author paper 1
        make_sig("s2", 1, 1, "co", "", "author"),   # coauthor to make it multi-author
        make_sig("s3", 2, 0, "ping", "", "zhang"),  # target on multi-author paper 2
        make_sig("s4", 2, 1, "co2", "", "author2"),
        make_sig("s5", 3, 0, "ping", "", "zhang"),  # target on multi-author paper 3
        make_sig("s6", 3, 1, "co3", "", "author3"),
        make_sig("s7", 4, 0, "ping", "", "zhang"),  # target on multi-author paper 4 (not-flip)
        make_sig("s8", 4, 1, "co4", "", "author4"),
    ])
    # sinonym parses: three flips (zhang,ping) and one not-flip (ping,zhang)
    ppr = build_per_paper(
        (1, 0, "zhang", "ping"), (1, 1, "co", "author"),
        (2, 0, "zhang", "ping"), (2, 1, "co2", "author2"),
        (3, 0, "zhang", "ping"), (3, 1, "co3", "author3"),
        (4, 0, "ping", "zhang"), (4, 1, "co4", "author4"),
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "1" in allow and 0 in allow["1"]
    assert "2" in allow and 0 in allow["2"]
    assert "3" in allow and 0 in allow["3"]
    assert "4" in allow and 0 in allow["4"]  # apply to all occurrences


def test_multi_author_ratio_fail():
    # Name: "pei wang" (x=2, y=3) with ratio=3.0 => 2 >= 9? no
    sigs = dict([
        make_sig("s1", 10, 0, "pei", "", "wang"), make_sig("s2", 10, 1, "co", "", "author"),
        make_sig("s3", 11, 0, "pei", "", "wang"), make_sig("s4", 11, 1, "co", "", "author"),
        make_sig("s5", 12, 0, "pei", "", "wang"), make_sig("s6", 12, 1, "co", "", "author"),
        make_sig("s7", 13, 0, "pei", "", "wang"), make_sig("s8", 13, 1, "co", "", "author"),
        make_sig("s9", 14, 0, "pei", "", "wang"), make_sig("s10",14, 1, "co", "", "author"),
    ])
    # sinonym: two flips (wang,pei) and three not-flips (pei,wang)
    ppr = build_per_paper(
        (10, 0, "wang", "pei"), (10, 1, "co", "author"),
        (11, 0, "wang", "pei"), (11, 1, "co", "author"),
        (12, 0, "pei",  "wang"), (12, 1, "co", "author"),
        (13, 0, "pei",  "wang"), (13, 1, "co", "author"),
        (14, 0, "pei",  "wang"), (14, 1, "co", "author"),
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "10" not in allow or 0 not in allow["10"]
    assert "12" not in allow or 0 not in allow["12"]


def test_multi_author_all_flips_yes():
    # y=0, x>0 implies ratio passes
    sigs = dict([
        make_sig("s1", 20, 0, "yang", "", "peng"), make_sig("s2", 20, 1, "co", "", "author"),
        make_sig("s3", 21, 0, "yang", "", "peng"), make_sig("s4", 21, 1, "co", "", "author"),
    ])
    ppr = build_per_paper(
        (20, 0, "peng", "yang"), (20, 1, "co", "author"),
        (21, 0, "peng", "yang"), (21, 1, "co", "author"),
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "20" in allow and 0 in allow["20"]
    assert "21" in allow and 0 in allow["21"]


def test_single_author_only_flip_yes():
    # Only single-author evidence; a>0 -> overwrite
    sigs = dict([
        make_sig("s1", 30, 0, "yang", "", "peng"),
    ])
    ppr = build_per_paper(
        (30, 0, "peng", "yang"),
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "30" in allow and 0 in allow["30"]


def test_single_author_only_not_flip_no():
    # Only single-author evidence; b>0 -> do not overwrite
    sigs = dict([
        make_sig("s1", 31, 0, "xiaofei", "", "lu"),
    ])
    ppr = build_per_paper(
        (31, 0, "xiaofei", "lu"),
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "31" not in allow or 0 not in allow["31"]


def test_mixed_single_and_multi_prioritizes_multi():
    # Multi-author y>0 dominates over single-author a>many; expect no overwrite
    # Multi-author: y=2 (not flips), x=0 (no flips) -> do not overwrite
    # Single-author: several flips (a>0), but should be ignored since x+y>0
    sigs = dict([
        make_sig("m1", 40, 0, "pei", "", "wang"), make_sig("m2", 40, 1, "co", "", "author"),
        make_sig("m3", 41, 0, "pei", "", "wang"), make_sig("m4", 41, 1, "co", "", "author"),
        # Single-author occurrences for same name
        make_sig("s1", 42, 0, "pei", "", "wang"),
        make_sig("s2", 43, 0, "pei", "", "wang"),
    ])
    ppr = build_per_paper(
        (40, 0, "pei", "wang"), (40, 1, "co", "author"),
        (41, 0, "pei", "wang"), (41, 1, "co", "author"),
        (42, 0, "wang", "pei"),  # single-author flip
        (43, 0, "wang", "pei"),  # single-author flip
    )
    allow = compute_sinonym_overwrite_allowlist(sigs, ppr, min_ratio=3.0)
    assert "40" not in allow or 0 not in allow["40"]
    assert "41" not in allow or 0 not in allow["41"]
    # And should not apply to single-author occurrences either
    assert "42" not in allow or 0 not in allow["42"]
    assert "43" not in allow or 0 not in allow["43"]
