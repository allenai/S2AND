"""Targeted coverage for the 2026-07-04 correctness pass (work_plan section 2).

These lock the observable behavior changes that a Python-vs-Rust parity run
cannot by itself pin down: the self-citation same-paper guard, the
reference-list features being computed without reference_details, and the
Sinonym-overwrite invalidation of derived name fields.
"""

from __future__ import annotations

import math

import numpy as np

from s2and.data import ANDData, NameCounts, apply_sinonym_overwrites
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from tests.helpers import tiny_name_counts
from tests.test_sinonym_overwrite import make_parsed, make_sig

_ALL_FEATURES = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "reference_features",
    "misc_features",
    "name_counts",
    "journal_similarity",
    "advanced_name_similarity",
]
# reference_features occupies these (global) vector positions; the block is
# appended as [rd0, rd1, rd2, rd3, self_cite, references_jaccard].
_REF_INDICES = FeaturizationInfo().feature_group_to_index["reference_features"]
_SELF_CITE_IDX = _REF_INDICES[4]
_REF_JACCARD_IDX = _REF_INDICES[5]
_REF_COUNTER_IDXS = _REF_INDICES[:4]


def _dataset() -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="correctness_pass",
        load_name_counts=tiny_name_counts(),
        compute_reference_features=True,
    )


def _set_paper(dataset: ANDData, paper_id: int, *, references, reference_details) -> None:
    key = str(paper_id)
    dataset.papers[key] = dataset.papers[key]._replace(references=references, reference_details=reference_details)


def _featurize_pair(dataset: ANDData, sid_a: str, sid_b: str) -> np.ndarray:
    featurizer = FeaturizationInfo(features_to_use=_ALL_FEATURES)
    features, _, _ = many_pairs_featurize([(sid_a, sid_b, 1)], dataset, featurizer, 1, False, 1, nan_value=np.nan)
    return features[0]


def test_self_cite_zero_for_two_signatures_on_the_same_paper() -> None:
    dataset = _dataset()
    sid_a, sid_b = "0", "1"
    paper_a = dataset.signatures[sid_a].paper_id
    # Put both signatures on paper A, and make paper A cite itself.
    dataset.signatures[sid_b] = dataset.signatures[sid_b]._replace(paper_id=paper_a)
    _set_paper(dataset, paper_a, references=[paper_a], reference_details=None)

    features = _featurize_pair(dataset, sid_a, sid_b)
    # A paper citing itself is not a cross-paper self-citation.
    assert features[_SELF_CITE_IDX] == 0


def test_self_cite_one_for_distinct_papers_with_citation() -> None:
    dataset = _dataset()
    sid_a, sid_b = "0", "1"
    paper_a = dataset.signatures[sid_a].paper_id
    paper_b = dataset.signatures[sid_b].paper_id
    assert paper_a != paper_b
    # Paper A cites paper B.
    _set_paper(dataset, paper_a, references=[paper_b], reference_details=None)
    _set_paper(dataset, paper_b, references=[], reference_details=None)

    features = _featurize_pair(dataset, sid_a, sid_b)
    assert features[_SELF_CITE_IDX] == 1


def test_reference_list_features_computed_without_reference_details() -> None:
    dataset = _dataset()
    sid_a, sid_b = "0", "1"
    paper_a = dataset.signatures[sid_a].paper_id
    paper_b = dataset.signatures[sid_b].paper_id
    # reference_details is None (e.g. preprocess=False path) but the raw
    # references lists are present.
    _set_paper(dataset, paper_a, references=[paper_b, 999], reference_details=None)
    _set_paper(dataset, paper_b, references=[999], reference_details=None)

    features = _featurize_pair(dataset, sid_a, sid_b)
    # The four ngram-Counter features still need reference_details -> NaN.
    for idx in _REF_COUNTER_IDXS:
        assert math.isnan(features[idx])
    # The two reference-list features only need paper.references -> computed.
    assert features[_SELF_CITE_IDX] == 1  # A cites B, distinct papers
    assert features[_REF_JACCARD_IDX] == 1 / 2  # {B,999} vs {999}


def test_apply_sinonym_overwrites_invalidates_derived_name_fields() -> None:
    _, sig = make_sig("s1", 100, 0, "Qi", "", "Ou-Yang")
    # Simulate a fully-preprocessed signature: derived name fields populated.
    sig = sig._replace(
        author_info_first_normalized="qi",
        author_info_first_normalized_without_apostrophe="qi",
        author_info_middle_normalized_without_apostrophe="",
        author_info_last_normalized="ou yang",
        author_info_full_name="qi ou yang",
        author_info_name_counts=NameCounts(first=5.0, last=5.0, first_last=5.0, last_first_initial=5.0),
    )
    signatures = {"s1": sig}
    per_paper_results = {"100": {0: make_parsed("Xin", "Wang")}}

    updated_count = apply_sinonym_overwrites(signatures, per_paper_results)
    assert updated_count == 1

    updated = signatures["s1"]
    # Raw parts were overwritten...
    assert updated.author_info_last == "Wang"
    # ...and every field derived from the name parts was invalidated so a
    # post-init caller cannot read stale normalized names / counts.
    assert updated.author_info_first_normalized is None
    assert updated.author_info_first_normalized_without_apostrophe is None
    assert updated.author_info_middle_normalized_without_apostrophe is None
    assert updated.author_info_last_normalized is None
    assert updated.author_info_full_name is None
    assert updated.author_info_name_counts is None
