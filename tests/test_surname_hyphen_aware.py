"""Canonical (canonical_v2) surname and first-name constraint behavior.

Historically this module pinned the legacy-compatibility shims
(`_lasts_equivalent_for_constraint`, `_canonicalize_last_for_counts`, and the
joined/first-token name-tuple probing forms). Those shims were removed by the
canonical_v2 cutover; this module now pins the canonical semantics that
replaced them (docs/normalization_migration_blocked.md, D4/D5).
"""

from s2and.data import ANDData
from s2and.text import canonical_lasts_equivalent, canonicalize_name_text


def test_canonical_last_treats_dash_and_space_variants_identically():
    # D4/D5: every dash-like character is a separator; canonical surnames are
    # spaced with particles preserved. Joined spellings stay distinct strings.
    assert canonicalize_name_text("Ou-Yang") == "ou yang"
    assert canonicalize_name_text("Ou–Yang") == "ou yang"
    assert canonicalize_name_text("Ou Yang") == "ou yang"
    assert canonicalize_name_text("Ouyang") == "ouyang"
    assert canonicalize_name_text("van-der-Berg") == "van der berg"


def test_canonical_lasts_equivalent_is_space_insensitive_compare_policy():
    # Storage is spaced (D5); compare time treats joined and spaced spellings
    # as one surname, but different surnames stay different.
    assert canonical_lasts_equivalent("ou yang", "ouyang") is True
    assert canonical_lasts_equivalent("ouyang", "ou yang") is True
    assert canonical_lasts_equivalent("van der berg", "vanderberg") is True
    assert canonical_lasts_equivalent("li", "ouyang") is False
    assert canonical_lasts_equivalent("", "") is True


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
