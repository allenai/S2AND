"""Canonical (canonical_v2) surname and first-name constraint behavior.

Pins the canonical semantics in docs/canonical_names.md.
"""

import json
from pathlib import Path

from s2and.data import ANDData
from s2and.text import (
    canonical_lasts_equivalent,
    canonical_surname_family,
    canonicalize_name_text,
    surname_variant_map,
)


def test_canonical_lasts_equivalent_is_space_insensitive_compare_policy():
    # Storage is spaced (D5); compare time treats joined and spaced spellings
    # as one surname, but different surnames stay different.
    assert canonical_lasts_equivalent("ou yang", "ouyang") is True
    assert canonical_lasts_equivalent("ouyang", "ou yang") is True
    assert canonical_lasts_equivalent("van der berg", "vanderberg") is True
    assert canonical_lasts_equivalent("li", "ouyang") is False
    assert canonical_lasts_equivalent("", "") is True


def test_canonical_lasts_equivalent_joins_listed_variants_and_sky_endings():
    # Blocking keys these spellings together, so the constraint must not veto them.
    assert canonical_lasts_equivalent("muller", "mueller") is True
    assert canonical_lasts_equivalent("tartakovskiy", "tartakovsky") is True
    assert canonical_lasts_equivalent("mueller", "muller schmidt") is False
    assert canonical_lasts_equivalent("xue", "xu") is False
    assert canonical_lasts_equivalent("kowalski", "kowalsky") is False


def test_canonical_surname_family_matches_shared_fixture():
    cases = json.loads((Path(__file__).parent / "fixtures" / "surname_family_examples.json").read_text("utf-8"))
    for case in cases["cases"]:
        assert canonical_surname_family(case["last"]) == case["family"], case


def test_surname_variant_list_is_canonical():
    variants = surname_variant_map()
    assert len(variants) > 200
    for variant, canonical in variants.items():
        assert canonicalize_name_text(variant) == variant
        assert canonicalize_name_text(canonical) == canonical


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


def _constraint_dataset(
    *,
    last_1: str = "Ou-Yang",
    last_2: str = "Ou Yang",
    name_tuples: set[tuple[str, str]] | None = None,
) -> ANDData:
    signatures = {
        "s1": _raw_signature("s1", paper_id=1, first="Qi-Xin", last=last_1),
        "s2": _raw_signature("s2", paper_id=2, first="Qadir", last=last_2),
    }
    papers = {
        "1": _raw_paper(1, f"Qi-Xin {last_1}"),
        "2": _raw_paper(2, f"Qadir {last_2}"),
    }
    return ANDData(
        signatures,
        papers,
        name="surname_hyphen_aware",
        mode="inference",
        name_counts_index=None,
        preprocess=False,
        name_tuples=name_tuples or set(),
        n_jobs=1,
    )


def test_constraint_treats_hyphen_and_space_last_names_as_equivalent():
    # Both variants canonicalize to "ou yang", so the last-name disallow does
    # not fire and the curated alias keeps the firsts compatible.
    dataset = _constraint_dataset(name_tuples={("qi xin", "qadir"), ("qadir", "qi xin")})

    assert dataset.get_constraint("s1", "s2") is None


def test_constraint_treats_joined_and_spaced_last_names_as_equivalent():
    # "ouyang" and "ou yang" are distinct canonical STRINGS (storage is spaced,
    # D5) but equivalent at compare time (canonical_lasts_equivalent): upstream
    # blocking groups surname spelling variants, and the within-block constraint
    # must not veto pairs that blocking deliberately grouped.
    dataset = _constraint_dataset(
        last_2="Ouyang",
        name_tuples={("qi xin", "qadir"), ("qadir", "qi xin")},
    )

    assert dataset.get_constraint("s1", "s2") is None


def test_constraint_disallows_genuinely_different_last_names():
    dataset = _constraint_dataset(
        last_2="Li",
        name_tuples={("qi xin", "qadir"), ("qadir", "qi xin")},
    )

    assert dataset.get_constraint("s1", "s2") is not None


def test_constraint_name_tuples_use_exact_canonical_forms_only():
    # The legacy joined ("qixin") and first-token ("qi") probing forms are
    # retired: a tuple curated in a non-canonical form no longer matches.
    for stale_tuples in ({("qixin", "qadir")}, {("qi", "qadir")}):
        dataset = _constraint_dataset(name_tuples=stale_tuples)
        assert dataset.get_constraint("s1", "s2") is not None

    dataset = _constraint_dataset(name_tuples={("qi xin", "qadir")})
    assert dataset.get_constraint("s1", "s2") is None
