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
        ("MArfan", "", "Wm arfan"),
        ("Ka", "A.", None),
        ("KA", "", None),
        ("Dr", "", None),
        ("Md", "", None),
    ],
)
def test_joined_initials_reading(first, middle, expected):
    assert joined_initials_for_constraint(first, middle) == expected


@pytest.mark.parametrize(
    ("first", "middle", "expected"),
    [
        ("Johanna", "IP", "Sip"),
        ("Johanna", "I.P.", "Sip"),
        # A vowel makes a mixed-case middle a name; ED stays a name; an all-capitals
        # first name carries no case.
        ("Johanna", "Ip", None),
        ("Edward", "ED", None),
        ("JOHANNA", "IP", None),
        ("Лев", "IP", "Sip"),
        ("Johanna IP", "", "Sip"),
        # Only the middle's last word counts: the spell step splits only that token.
        ("Johanna", "IP A", None),
        # Vowel-less mixed case reads as initials as in D9; "Mc", suffixes and roman numerals do not.
        ("Hendrik", "Hg", "Shg"),
        ("Jeffrey", "Mc", None),
        ("John", "III", None),
    ],
)
def test_spelled_middle_initials_value(first, middle, expected):
    assert given_name_for_constraint(first, middle) == expected


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
        (("Mal", ""), ("Marco Aurelio Lisboa", ""), False),
        (("MArfan", ""), ("Mohammad", "Arfan"), False),
        # A joined-capitals middle is spelled out ("Johanna IP" is J. I. P.).
        (("Jip", ""), ("Johanna", "IP"), False),
        # A joined-capitals last middle word after an initial is spelled out too.
        (("Johanna", "A IP"), ("Johanna", "A. Inge Petra"), False),
        # Too few names on the other side, or other initials, keep the veto.
        (("Ka", ""), ("Kristian", ""), True),
        (("MArfan", ""), ("Mohammad", "Arif"), True),
        # Short romanized syllables on the other side keep the veto.
        (("Ji", ""), ("Jong", "In"), True),
        # Two short names compare by the ordinary rule only.
        (("Ka", ""), ("Ki", ""), True),
    ],
)
def test_joined_initials_reconcile_only_exact_spellings(left, right, expected):
    assert _conflict(left, right) is expected
    assert _conflict(right, left) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (("Johanna", "IP"), ("Johanna", "I. P."), False),
        (("Johanna", "IP"), ("Johanna", "Karin Maria"), True),
        # A title-case middle stays a name.
        (("Johanna", "Ip"), ("Johanna", "Inge Petra"), True),
    ],
)
def test_joined_capitals_middle_reads_as_initials(left, right, expected):
    assert _conflict(left, right) is expected
    assert _conflict(right, left) is expected


def test_joined_initials_keep_the_surname_rule():
    name_a = canonicalize_name_parts("Ka", "", "Hahn")
    name_b = canonicalize_name_parts("Kristan", "Allan", "Hahn-Smith")
    values = given_name_for_constraint("Ka", ""), given_name_for_constraint("Kristan", "Allan")
    assert canonical_names_conflict(name_a, name_b, set(), cjk_given_names=values)


# Set model distances below the cut to exercise the name rule.
@pytest.mark.parametrize("route", ["arrow", "python"])
def test_full_block_prediction_applies_the_joined_initials_reading(tmp_path: Path, route: str) -> None:
    names = [
        ("Ka", "", "Hahn"),
        ("Kristan", "Allan", "Hahn"),
        ("MArfan", "", "Ikram"),
        ("Mohammad", "Arfan", "Ikram"),
        ("Ka", "", "Berg"),
        ("Kristian", "", "Berg"),
        ("Jip", "", "de Vries"),
        ("Johanna", "IP", "de Vries"),
        ("Johanna", "Inge Petra", "de Vries"),
    ]
    blocks = {"hahn": ["0", "1"], "ikram": ["2", "3"], "berg": ["4", "5"], "de vries": ["6", "7", "8"]}

    found = _predict(tmp_path, _dataset(names, {}), blocks, route, threshold=None)

    expected = [["0", "1"], ["2", "3"], ["4"], ["5"], ["6", "7", "8"]]
    assert sorted(sorted(members) for members in found.values()) == expected


@pytest.mark.parametrize("route", ["arrow", "python"])
def test_repair_rejoins_a_piece_that_a_joined_reading_keeps_apart(tmp_path: Path, route):
    # "Ec" spells "Edward C." but conflicts with a bare "Edward". With every model distance
    # at 0.1, linkage leaves "Ec" in a piece that the bare "Edward" mentions cannot join;
    # the repair's reading step merges it back, since the other piece spells "Ec".
    names = [("Ec", "", "Holmes")] + [("Edward", "C.", "Holmes")] * 6 + [("Edward", "", "Holmes")] * 5
    dataset = _dataset(names, {})
    found = _predict(tmp_path, dataset, {"block": [str(i) for i in range(len(names))]}, route, threshold=None)
    assert len(found) == 1


@pytest.mark.parametrize("route", ["arrow", "python"])
def test_reading_repair_keeps_required_seed_groups_apart_when_linkage_ignores_them(tmp_path: Path, route):
    # Two Edward C. mentions in different required seed groups; linkage ignores the seeds, but
    # the repair still may not join the groups, on either route.
    names = [("Ec", "", "Holmes")] + [("Edward", "C.", "Holmes")] * 6 + [("Edward", "", "Holmes")] * 5
    dataset = _dataset(names, {"1": "p1", "5": "p2"})
    found = _predict(
        tmp_path,
        dataset,
        {"block": [str(i) for i in range(len(names))]},
        route,
        threshold=None,
        incremental_dont_use_cluster_seeds=True,
    )
    assert not any({"1", "5"} <= set(members) for members in found.values())
