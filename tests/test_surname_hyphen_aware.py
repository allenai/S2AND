import pytest

from s2and.data import (
    ANDData,
    _canonicalize_last_for_counts,
    _lasts_equivalent_for_constraint,
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
