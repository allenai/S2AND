"""Compare canonical names for clustering constraints."""

import re
from collections.abc import Set

from .canonical_names import (
    _CJK_SCRIPT_TAGS,
    _JOINED_INITIALS_GIVEN_NAMES,
    _MIXED_CASE_NON_INITIALS,
    CanonicalNameParts,
    _joined_initials,
    canonical_lasts_equivalent,
    cjk_given_name_for_constraint,
    last_name_allows_joined_initials,
)
from .normalize import DROPPED_AFFIXES, NAME_PREFIXES

_RE_SHORT_GIVEN_NAME = re.compile(r"[A-Z][a-z]{1,2}")


_RE_INITIAL_THEN_NAME = re.compile(r"[A-Z]([A-Z][a-z]{2,})")


def is_joined_reading(value: str | None) -> bool:
    """Return whether a ``given_name_for_constraint`` value is a joined-initials reading."""
    return bool(value) and value[0] in "JW"  # type: ignore[index]


def joined_initials_for_constraint(first_raw: str | None, middle_raw: str | None) -> str | None:
    """Return the joined-initials reading of a raw given name, or None.

    "Ka" (no middle name) reads "Jka", meaning K. A.; "MArfan" reads "Wm arfan", meaning
    M. Arfan. Real short names ("Ka", "Mal", "Al") take the reading too. A pair verdict
    changes only when the other name spells exactly these initials (see
    `joined_initials_match`), but the changed verdicts can change which clusters linkage
    builds, and name-conflict repair reads the reading at cluster level.
    Rust `joined_initials_for_constraint` mirrors it.
    """
    first = (first_raw or "").strip()
    if (middle_raw or "").strip():
        return None
    if _RE_SHORT_GIVEN_NAME.fullmatch(first):
        lowered = first.lower()
        # ED and MD stay given names, as in the canonical initials parser.
        if (
            lowered in _MIXED_CASE_NON_INITIALS
            or lowered in NAME_PREFIXES
            or first.upper() in _JOINED_INITIALS_GIVEN_NAMES
        ):
            return None
        return "J" + lowered
    match = _RE_INITIAL_THEN_NAME.fullmatch(first)
    if match:
        return "W" + first[0].lower() + " " + match.group(1).lower()
    return None


# A middle name of these letters is a suffix, a roman numeral or the surname prefix "Mc",
# not initials, in any letter case.
_MIDDLE_NON_INITIALS = frozenset({"jr", "sr", "ii", "iii", "iv", "vi", "vii", "viii", "mc"})


def spelled_middle_initials_for_constraint(first_raw: str | None, middle_raw: str | None) -> str | None:
    """Return "S" plus the middle initials of a name with a joined-capitals middle, or None.

    "Johanna IP" gives "Sip". Canonical names keep "IP" as the one token "ip", so without
    this value "Jip" could not spell "Johanna IP". The middle is the last word of the raw
    middle field, or of the raw first field when the middle field is empty, so "A IP" gives
    "Sip". `spell_middle_initials` splits only the last middle token, so "IP A" gives None.
    The word must pass the D9 initials test (2-3 capitals, or vowel-less mixed case such as
    "Hg") and not be a suffix, a roman numeral or "Mc". The
    first name must pass the D9 letter-case test that last names use
    (`last_name_allows_joined_initials`): an all-capitals name carries no case. That test
    transliterates first, so "Лев IP" qualifies, and Python and Rust agree.
    Rust `spelled_middle_initials_for_constraint` mirrors it.
    """
    first, middle = first_raw or "", (middle_raw or "").strip()
    if not middle:
        words = first.split()
        if len(words) < 2:
            return None
        first, middle = " ".join(words[:-1]), words[-1]
    if not last_name_allows_joined_initials(first):
        return None
    initials = _joined_initials(middle.split()[-1])
    return None if initials is None or initials in _MIDDLE_NON_INITIALS else "S" + initials


def given_name_for_constraint(first_raw: str | None, middle_raw: str | None) -> str | None:
    """Return the CJK given-name value, else a joined-initials value, else None.

    The values never overlap: CJK given names are not ASCII, joined-initials readings
    have no middle name, and spelled middle initials need one.
    """
    cjk = cjk_given_name_for_constraint(first_raw, middle_raw)
    if cjk is not None:
        return cjk
    reading = joined_initials_for_constraint(first_raw, middle_raw)
    return reading if reading is not None else spelled_middle_initials_for_constraint(first_raw, middle_raw)


def spell_middle_initials(name: "CanonicalNameParts", value: str | None) -> "CanonicalNameParts":
    """Split a joined-capitals middle back into initials for the name rule.

    With an S value from `spelled_middle_initials_for_constraint`, "johanna ip" becomes
    "johanna i p", so it agrees with "Johanna I. P." and "Johanna Inge Petra". Other names
    return unchanged. Rust `spell_middle_tokens` mirrors it.
    """
    if not value or value[0] != "S":
        return name
    tokens = name.middle.split()
    if not tokens or tokens[-1] != value[1:]:
        return name
    return name._replace(middle=" ".join([*tokens[:-1], *value[1:]]))


def spelled_reading_forms(other: "CanonicalNameParts", other_value: str | None = None) -> tuple[str, ...]:
    """Return the joined-initials readings that a name spells exactly, at most two.

    The name needs a full first name. It spells the "J" reading of its first and middle
    initials when every later token is an initial or at least four letters, so "Jong In"
    spells no "Ji" and "Lei" spells no "Li". With exactly two tokens it also spells the
    "W" reading of its initial and second word. An ``other_value`` with tag S splits the
    joined middle back into initials first. Rust `spelled_reading_forms` mirrors it.
    """
    other = spell_middle_initials(other, other_value)
    tokens = other.first.split() + other.middle.split()
    if not tokens or len(tokens[0]) < 2:
        return ()
    forms: list[str] = []
    if all(len(token) == 1 or len(token) >= 4 for token in tokens[1:]):
        forms.append("J" + "".join(token[0] for token in tokens))
    if len(tokens) == 2:
        forms.append("W" + tokens[0][0] + " " + tokens[1])
    return tuple(forms)


def joined_initials_match(reading: str, other: "CanonicalNameParts", other_value: str | None = None) -> bool:
    """Return whether a joined-initials reading spells the other name exactly.

    For example "Jka" spells "Kristan Allan" but not "Kristian"; see `spelled_reading_forms`.
    """
    return reading in spelled_reading_forms(other, other_value)


def same_prefix_tokens(a: str, b: str) -> bool:
    """
    Symmetric multi-token “startswith”.
    Assumes that the inputs are already fully normalized, lower-cased and depunctuated.
    Also assumes that multi-tokens are SPACE separated, not anything else (like dashes).
    True ⇔ for every aligned pair of tokens (up to the shorter list),
           one token is a prefix of the other.
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    for x, y in zip(ta, tb, strict=False):
        if not (x.startswith(y) or y.startswith(x)):
            return False
    return True


def canonical_name_tuple_pair(first_a: str, first_b: str) -> tuple[str, str]:
    """Return the order-independent storage key for one first-name alias."""

    return (first_a, first_b) if first_a <= first_b else (first_b, first_a)


def first_names_name_compatible(first_a: str, first_b: str, name_tuples: Set[tuple[str, str]]) -> bool:
    """Return canonical first-name compatibility.

    Two canonical first fields are compatible when either is missing (unknown is
    not an incompatibility signal), when they are prefix-compatible per
    ``same_prefix_tokens``, or when the pair is a curated alias in the canonical
    name-tuple artifact.
    """

    if not first_a.split() or not first_b.split():
        return True

    if same_prefix_tokens(first_a, first_b):
        return True

    pair = canonical_name_tuple_pair(first_a, first_b)
    return pair in name_tuples or (pair[1], pair[0]) in name_tuples


def _uncovered_first_tail(tokens: list[str], other: list[str]) -> list[str]:
    """Return the later first-name tokens that the other first name does not cover.

    The other first name covers the tokens at its aligned positions. When its
    last token extends past the aligned token, the remainder also covers the
    next tokens, so ``xiaoqing`` covers ``xiao qing``. The first token is the
    first name itself and is never part of the tail.
    """

    start = max(len(other), 1)
    if start >= len(tokens):
        return []
    if other:
        aligned, last = tokens[len(other) - 1], other[-1]
        rest = last[len(aligned) :] if last.startswith(aligned) else ""
        while rest and start < len(tokens):
            token = tokens[start]
            if rest.startswith(token):
                rest = rest[len(token) :]
            elif token.startswith(rest):
                rest = ""
            else:
                break
            start += 1
    return tokens[start:]


def constraint_middle_tokens(first_a: str, middle_a: str, first_b: str, middle_b: str) -> tuple[list[str], list[str]]:
    """Return the middle-name tokens that the hard name constraint compares.

    A canonical first name keeps a dash-bound group, for example ``jean daniel``
    from ``Jean-Daniel``. When the first names match by prefix, or one is
    missing, the first-name check compares only the aligned tokens. The tokens
    that the other first name does not cover then come before the middle-name
    tokens, so ``jean daniel`` compares with ``jean`` + ``daniel`` as first
    ``jean`` with middle ``daniel``. Aligned tokens, such as the two
    ``pierre`` tokens of two ``jean pierre`` records, are not added. When the
    first names match only through an alias tuple, the alias covers the full
    first names and the middle-name tokens do not change. Call this function
    only for compatible first names.

    Args:
        first_a: Canonical first name of one record.
        middle_a: Canonical middle name of that record.
        first_b: Canonical first name of the other record.
        middle_b: Canonical middle name of the other record.

    Returns:
        The effective middle-name tokens of each record.
    """

    first_tokens_a, first_tokens_b = first_a.split(), first_b.split()
    middle_tokens_a, middle_tokens_b = middle_a.split(), middle_b.split()
    if first_tokens_a and first_tokens_b and not same_prefix_tokens(first_a, first_b):
        return middle_tokens_a, middle_tokens_b
    return (
        _uncovered_first_tail(first_tokens_a, first_tokens_b) + middle_tokens_a,
        _uncovered_first_tail(first_tokens_b, first_tokens_a) + middle_tokens_b,
    )


def middle_names_conflict(middle_a: list[str], middle_b: list[str]) -> bool:
    """Return whether two middle-name token lists violate the hard name rule.

    Shared affixes do not count. The initials of the remaining tokens must
    intersect. Full words must intersect, use the same letters, or match by
    aligned prefixes.

    Args:
        middle_a: Effective middle-name tokens of one record.
        middle_b: Effective middle-name tokens of the other record.

    Returns:
        True when the middle names conflict.
    """

    if not middle_a or not middle_b:
        return False
    overlapping_affixes = set(middle_b).intersection(middle_a).intersection(DROPPED_AFFIXES)
    kept_a = [word for word in middle_a if word and word not in overlapping_affixes]
    kept_b = [word for word in middle_b if word and word not in overlapping_affixes]
    words_a = {word for word in kept_a if len(word) > 1}
    words_b = {word for word in kept_b if len(word) > 1}
    initials_a = {word[0] for word in kept_a}
    initials_b = {word[0] for word in kept_b}
    conflicting_initials = bool(initials_a) and bool(initials_b) and not initials_a & initials_b
    conflicting_full_names = (
        bool(words_a)
        and bool(words_b)
        and not words_a & words_b
        and set("".join(words_a)) != set("".join(words_b))
        and not same_prefix_tokens(" ".join(middle_a), " ".join(middle_b))
    )
    return conflicting_initials or conflicting_full_names


def canonical_names_conflict(
    name_a: CanonicalNameParts,
    name_b: CanonicalNameParts,
    name_tuples: Set[tuple[str, str]],
    *,
    cjk_given_names: tuple[str | None, str | None] = (None, None),
) -> bool:
    """Return whether two canonical names violate the hard name constraint.

    This is the name part of ``ANDData._get_constraint`` and of its Rust
    equivalent: last names, CJK given names, first initials, first-name
    compatibility and effective middle names. Seed, explicit disallow and
    ORCID rules are not part of it.

    Args:
        name_a: Canonical name parts of one record.
        name_b: Canonical name parts of the other record.
        name_tuples: Canonical first-name alias pairs.
        cjk_given_names: Results of ``given_name_for_constraint`` in pair order: whole
            CJK given names or joined-initials readings.

    Returns:
        True when the two names cannot belong to the same person.
    """

    if not canonical_lasts_equivalent(name_a.last, name_b.last):
        return True
    given_a, given_b = cjk_given_names
    # Incomplete or cross-script CJK given names keep the ordinary rules below.
    if given_a and given_b and given_a[0] == given_b[0] and given_a[0] in _CJK_SCRIPT_TAGS:
        return given_a != given_b
    name_a, name_b = spell_middle_initials(name_a, given_a), spell_middle_initials(name_b, given_b)
    if not _ordinary_names_conflict(name_a, name_b, name_tuples):
        return False
    # A short given name may be joined initials ("Ka" for K. A.). It reconciles a
    # conflict only against a name whose initials it spells exactly.
    if is_joined_reading(given_a) and not is_joined_reading(given_b):
        return not joined_initials_match(given_a or "", name_b, given_b)
    if is_joined_reading(given_b) and not is_joined_reading(given_a):
        return not joined_initials_match(given_b or "", name_a, given_a)
    return True


def _ordinary_names_conflict(
    name_a: CanonicalNameParts, name_b: CanonicalNameParts, name_tuples: Set[tuple[str, str]]
) -> bool:
    """Return the first-initial, first-name and middle-name part of the hard rule."""
    if name_a.first and name_b.first and name_a.first[0] != name_b.first[0]:
        return True
    if not first_names_name_compatible(name_a.first, name_b.first, name_tuples):
        return True
    middle_a, middle_b = constraint_middle_tokens(name_a.first, name_a.middle, name_b.first, name_b.middle)
    return middle_names_conflict(middle_a, middle_b)
