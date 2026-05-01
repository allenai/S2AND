import pytest

from s2and.data import (
    ANDData,
    Signature,
    _canonicalize_last_for_counts,
    _lasts_equivalent_for_constraint,
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


def _raw_signature(signature_id: str, *, paper_id: int, first: str, last: str) -> dict:
    return {
        "signature_id": signature_id,
        "paper_id": paper_id,
        "author_info": {
            "position": 0,
            "block": f"{first[:1].lower()} {last.lower()}",
            "first": first,
            "middle": "",
            "last": last,
            "suffix": None,
            "email": None,
            "affiliations": [],
        },
    }


def _raw_paper(paper_id: int, author_name: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": f"Paper {paper_id}",
        "abstract": "",
        "journal_name": "",
        "venue": "",
        "year": 2020,
        "authors": [{"position": 0, "author_name": author_name}],
        "references": [],
    }


def _constraint_dataset(*, name_tuples: set[tuple[str, str]] | None = None) -> ANDData:
    signatures = {
        "s1": _raw_signature("s1", paper_id=1, first="Qi-Xin", last="Ou-Yang"),
        "s2": _raw_signature("s2", paper_id=2, first="Qadir", last="Ou Yang"),
    }
    papers = {
        "1": _raw_paper(1, "Qi-Xin Ou-Yang"),
        "2": _raw_paper(2, "Qadir Ou Yang"),
    }
    return ANDData(
        signatures,
        papers,
        name="surname_hyphen_aware",
        mode="inference",
        load_name_counts=False,
        preprocess=False,
        name_tuples=name_tuples or set(),
        n_jobs=1,
    )


def test_constraint_treats_hyphen_and_space_last_names_as_equivalent():
    dataset = _constraint_dataset(name_tuples={("qi xin", "qadir")})

    assert dataset.get_constraint("s1", "s2") is None


@pytest.mark.parametrize(
    "name_tuples",
    [
        {("qi xin", "qadir")},
        {("qixin", "qadir")},
        {("qi", "qadir")},
    ],
)
def test_constraint_accepts_name_tuple_compatibility_forms(name_tuples):
    dataset = _constraint_dataset(name_tuples=name_tuples)

    assert dataset.get_constraint("s1", "s2") is None


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
    assert new_sig.author_info_block == "q ou yang"
