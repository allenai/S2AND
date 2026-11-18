"""Regressions for short given names read as joined initials in the hard name rule.

Production splits profiles such as Kristan Allan Hahn ("Ka" Hahn) and Marco Aurelio
Lisboa Leite ("Mal" Leite) because a joined-initials spelling is parsed as a given name.
The hard rule reads such a name as initials only against a name that spells exactly
those initials, so real short names keep their ordinary verdicts.
"""

from pathlib import Path

import pytest

from s2and.text import (
    canonical_names_conflict,
    canonicalize_name_parts,
    given_name_for_constraint,
    joined_initials_for_constraint,
)
from tests.incremental_test_helpers import _dataset, _predict


@pytest.mark.parametrize(
    ("first", "middle", "expected"),
    [
        ("Ka", "", "Jka"),
        ("Mal", None, "Jmal"),
        ("MArfan", "", "Wm arfan"),
        ("Ka", "A.", None),
        ("KA", "", None),
        ("Kristan", "", None),
        ("Dr", "", None),
        ("Md", "", None),
        ("Ed", "", None),
        ("K", "", None),
    ],
)
def test_joined_initials_reading(first, middle, expected):
    assert joined_initials_for_constraint(first, middle) == expected


def _conflict(a: tuple[str, str], b: tuple[str, str], last: str = "Leite") -> bool:
    name_a = canonicalize_name_parts(a[0], a[1], last)
    name_b = canonicalize_name_parts(b[0], b[1], last)
    values = given_name_for_constraint(*a), given_name_for_constraint(*b)
    return canonical_names_conflict(name_a, name_b, set(), cjk_given_names=values)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # Joined initials that spell the other name exactly.
        (("Ka", ""), ("Kristan", "Allan"), False),
        (("Ka", ""), ("Kristian", "A."), False),
        (("Mal", ""), ("Marco Aurelio Lisboa", ""), False),
        (("MArfan", ""), ("Mohammad", "Arfan"), False),
        (("Al", ""), ("Andrew", "L."), False),
        # Too few names on the other side, or other initials, keep the veto.
        (("Ka", ""), ("Kristian", ""), True),
        (("Mal", ""), ("Mark", ""), True),
        (("MArfan", ""), ("Mohammad", ""), True),
        (("MArfan", ""), ("Mohammad", "Arif"), True),
        (("Jim", ""), ("Jack", "M. T."), True),
        # Short romanized syllables on the other side keep the veto.
        (("Li", ""), ("Lei", ""), True),
        (("Ji", ""), ("Jong", "In"), True),
        (("Yu", ""), ("Young", "Uk"), True),
        # Two short names compare by the ordinary rule only.
        (("Ka", ""), ("Ki", ""), True),
        # Unrelated conflicts are unchanged.
        (("Kristan", ""), ("Kristian", ""), True),
        (("Ka", ""), ("Kristan", "Allan"), False),
    ],
)
def test_joined_initials_reconcile_only_exact_spellings(left, right, expected):
    assert _conflict(left, right) is expected
    assert _conflict(right, left) is expected


def test_joined_initials_keep_the_surname_rule():
    name_a = canonicalize_name_parts("Ka", "", "Hahn")
    name_b = canonicalize_name_parts("Kristan", "Allan", "Hahn-Smith")
    values = given_name_for_constraint("Ka", ""), given_name_for_constraint("Kristan", "Allan")
    assert canonical_names_conflict(name_a, name_b, set(), cjk_given_names=values)


# Every pair gets model distance 0.1, under EPS 0.5, so only the name rule keeps people apart.
@pytest.mark.parametrize("route", ["arrow", "python"])
@pytest.mark.parametrize(
    ("names", "clusters"),
    [
        ([("Ka", "", "Hahn"), ("Kristan", "Allan", "Hahn")], 1),
        ([("Mal", "", "Leite"), ("Marco Aurelio Lisboa", "", "Leite")], 1),
        ([("MArfan", "", "Ikram"), ("Mohammad", "Arfan", "Ikram")], 1),
        ([("Li", "", "Wang"), ("Lei", "", "Wang")], 2),
        ([("Ka", "", "Hahn"), ("Kristian", "", "Hahn")], 2),
    ],
)
def test_full_block_prediction_applies_the_joined_initials_reading(tmp_path: Path, route, names, clusters):
    dataset = _dataset(names, {})
    found = _predict(tmp_path, dataset, {"block": [str(i) for i in range(len(names))]}, route, threshold=None)
    assert len(found) == clusters
