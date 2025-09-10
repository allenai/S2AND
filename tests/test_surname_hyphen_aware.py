import pytest

from s2and.data import (
    Signature,
    _lasts_equivalent_for_constraint,
    _canonicalize_last_for_counts,
    apply_sinonym_overwrites,
)


def test_last_equivalence_helper():
    # Hyphen/space variants should be equivalent for constraints
    assert _lasts_equivalent_for_constraint("ou yang", "ouyang") is True
    assert _lasts_equivalent_for_constraint("ouyang", "ou yang") is True
    assert _lasts_equivalent_for_constraint("li", "ouyang") is False


def test_canonicalize_last_for_counts():
    # Join internal spaces for compound surnames
    assert _canonicalize_last_for_counts("Ou-Yang", "ou yang") == "ouyang"
    # Gracefully handle normalized-only signal
    assert _canonicalize_last_for_counts(None, "ou yang") == "ouyang"
    # Non-compound surnames should pass through
    assert _canonicalize_last_for_counts("Smith", "smith") == "smith"


def test_apply_sinonym_overwrites_block_compound_surname():
    # Single signature with a compound surname; expect block to use joined surname
    sig = Signature(
        author_info_first="qi",
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle="",
        author_info_middle_normalized_without_apostrophe=None,
        author_info_last_normalized=None,
        author_info_last="yang",
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
        author_info_position=0,
        author_info_block="q yang",  # initial block (legacy)
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=1,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id="s1",
    )
    signatures = {"s1": sig}

    per_paper_results = {
        "1": {
            0: {
                "given_tokens": ["Qi"],
                "surname_tokens": ["Ou", "Yang"],
                "original_compound_surname": "Ou-Yang",
            }
        }
    }

    updated = apply_sinonym_overwrites(
        signatures,
        per_paper_results,
        overwrite_blocks=True,
        allow_overwrite_pos=None,
    )
    assert updated == 1
    new_sig = signatures["s1"]
    assert new_sig.author_info_block == "q ouyang"

