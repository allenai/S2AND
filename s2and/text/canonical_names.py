"""Normalize canonical names and surname families."""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from text_unidecode import unidecode

from .normalize import DASH_CHARS, NAME_PREFIXES

# ------------------------ canonical_v2 name canonicalization ------------------------
# These functions are the live canonical normalization surface used by ingestion,
# feature extraction, subblocking, and query adaptation. The durable contract is in
# docs/canonical_names.md; decisions D1-D9 are recorded in the fixture's decisions registry.

# D3 apostrophe-like marks: ASCII apostrophe, backtick, spacing acute, curly quotes,
# modifier letters (okina/apostrophe), primes, saltillo, U+FE4D (classified with
# apostrophe-like marks by issue #39 despite its Unicode name), fullwidth apostrophe.
NAME_APOSTROPHE_LIKE_CHARS = frozenset("'`\u00b4\u2018\u2019\u02bb\u02bc\u2032\u2035\ua78c\ufe4d\uff07")


# Invisible formatting controls deleted before tokenization: soft hyphen (not a dash
# separator) and zero-width joiner.
_NAME_INVISIBLE_FORMAT_CHARS = "\u00ad\u200d"


# D10: Georgian and Armenian romanize with the tables of scholar's block keys
# (BlockKeyNormalizers), not text_unidecode. Both sides then agree on names such
# as ბერიძე -> beridze, where text_unidecode writes "berize".
# Georgian is the national system without apostrophes. Mkhedruli letters give
# lowercase. Mtavruli capitals give uppercase, which keeps the D9 case guard.
_GEORGIAN_ROMANIZATION = dict(
    zip(
        "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰჱჲჳჴჵჶ",
        (
            *("a", "b", "g", "d", "e", "v", "z", "t", "i", "k", "l", "m", "n", "o", "p", "zh"),
            *("r", "s", "t", "u", "p", "k", "gh", "q", "sh", "ch", "ts", "dz", "ts", "ch", "kh"),
            *("j", "h", "e", "y", "v", "kh", "o", "f"),
        ),
        strict=True,
    )
)


# Mtavruli sits at a fixed offset from Mkhedruli.
_GEORGIAN_MTAVRULI_OFFSET = 0x1C90 - 0x10D0


# Eastern Armenian as most Latin sources write it. `_romanize_armenian` holds
# the context rules for ու, a word-initial ե, and a word-initial ո.
_ARMENIAN_ROMANIZATION = dict(
    zip(
        "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ",
        (
            *("a", "b", "g", "d", "e", "z", "e", "y", "t", "zh", "i", "l", "kh", "ts", "k", "h"),
            *("dz", "gh", "ch", "m", "y", "n", "sh", "o", "ch", "p", "j", "r", "s", "v", "t"),
            *("r", "ts", "v", "p", "k", "o", "f"),
        ),
        strict=True,
    )
)


# Armenian capitals sit 0x30 below their small letters.
_ARMENIAN_SMALL_FROM_CAPITAL = {chr(ord(small) - 0x30): small for small in _ARMENIAN_ROMANIZATION}


# Scholar applies NFKC first, which splits the Armenian ligatures into letters.
_ARMENIAN_LIGATURES = {
    "և": "եւ",
    "ﬓ": "մն",
    "ﬔ": "մե",
    "ﬕ": "մի",
    "ﬖ": "վն",
    "ﬗ": "մխ",
}


_RE_ARMENIAN_LETTER = re.compile("[Ա-Ֆա-ևﬓ-ﬗ]")


# Scholar deletes these characters before it reads an Armenian letter's
# neighbors: punctuation, nonspacing marks, and format, control, private-use,
# and surrogate characters. The context rules skip them.
_ARMENIAN_CONTEXT_SKIPPED_CATEGORIES = frozenset({"Mn", "Cf", "Cc", "Co", "Cs"})


_CANONICAL_NAME_TRANSLATION = str.maketrans(
    {
        **{character: None for character in _NAME_INVISIBLE_FORMAT_CHARS},
        **{character: "'" for character in NAME_APOSTROPHE_LIKE_CHARS},
        **{character: "-" for character in DASH_CHARS},
        **_GEORGIAN_ROMANIZATION,
        **{
            chr(ord(letter) + _GEORGIAN_MTAVRULI_OFFSET): roman.upper()
            for letter, roman in _GEORGIAN_ROMANIZATION.items()
        },
    }
)


_RE_CANONICAL_NON_LETTER = re.compile(r"[^a-z]+")


# D9: ED is a frequent given name, and MD abbreviates Muhammad (D7).
_JOINED_INITIALS_GIVEN_NAMES = frozenset({"ED", "MD"})


# With a final period, these capital pairs abbreviate one given name, for example
# "PH." (Philippe), "GY." (György), and "YU." (Yuri).
_DOTTED_GIVEN_NAME_ABBREVIATIONS = frozenset(
    {"CH", "SH", "ZH", "KH", "TS", "YU", "YA", "YE", "GY", "NY", "SZ", "ZS", "CS", "PH", "TH"}
)


_RE_JOINED_INITIALS = re.compile(r"[A-Z]{2,3}")


# The D9 mixed-case branch: one capital and 1-2 lowercase letters, with no
# vowel and no y, such as "Jh". The pattern keeps periods, so "Ch." fails it.
_RE_MIXED_CASE_INITIALS = re.compile(r"[A-Z][a-z]{1,2}")


_MIXED_CASE_INITIALS_VOWELS = frozenset("aeiouy")


# These mixed-case strings are not initials. The 2026-10-16 review of mention
# counts and gold clusters found that each one is a given-name abbreviation, a
# title, a suffix, a real name, or a company word.
_MIXED_CASE_NON_INITIALS = frozenset(
    {
        # Abbreviations of one given name, for example "Md" (Muhammad) and "Ch" (Christian).
        *("ch", "chr", "cs", "dzh", "fr", "gh", "kh", "khr", "lj", "ll", "md", "mhd"),
        *("ph", "sh", "sk", "st", "sw", "sz", "th", "vl", "wm", "zh", "zs"),
        # Titles before the given name, for example "Mst" (Mosammat) and "Lt" (Lieutenant).
        *("ds", "hjh", "lt", "mst", "rr", "smt"),
        # Suffixes: "Jr" (Junior) and "Sr" (Senior).
        *("jr", "sr"),
        # Real names: "Ng" and "Lv" are surnames, and "Tsz" is a given name.
        *("lv", "ng", "tsz"),
        # A company word ("Ltd").
        "ltd",
    }
)


_RE_ASCII_LOWERCASE = re.compile(r"[a-z]")


_RE_ASCII_UPPERCASE = re.compile(r"[A-Z]")


# Han, kana and hangul code points. Transliteration writes each Han character as one
# pinyin syllable, so the syllables of two different given names often match by
# prefix ("zhong xi" and "zhong xiang"). The name rule compares such given names whole.
# Keep the ranges in sync with Rust `text_compat`.
_HAN_NAME_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x323AF),  # CJK Unified Ideographs Extensions B-H and Compatibility Supplement
)


_CJK_NAME_RANGES: tuple[tuple[int, int], ...] = (
    *_HAN_NAME_RANGES,
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3005, 0x3007),  # Ideographic iteration mark, closing mark and number zero
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7FF),  # Hangul Syllables and Hangul Jamo Extended-B
    (0xFF66, 0xFFDC),  # Halfwidth Katakana and Halfwidth Hangul
)


# Script tags of a whole CJK given name. Pinyin of a Han name and the romanization of its
# hangul or kana form differ ("minzhu" for 敏洙, "minsu" for 민수), so only names in one
# script are compared whole. Keep the ranges and tags in sync with Rust `text_compat`.
_HANGUL_NAME_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
    (0xFFA0, 0xFFDC),
)


_KANA_NAME_RANGES: tuple[tuple[int, int], ...] = ((0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F))


def _is_cjk_name_character(character: str) -> bool:
    code = ord(character)
    return any(start <= code <= end for start, end in _CJK_NAME_RANGES)


def _widen_kana(text: str) -> str:
    """Write halfwidth katakana and spacing voicing marks as their composable forms.

    NFC alone keeps halfwidth katakana apart from its fullwidth form, and a spacing voicing
    mark apart from the voiced kana. The halfwidth compatibility mappings have not changed
    since Unicode 1.1, so Python and Rust agree. Rust `widen_kana` mirrors it.
    """
    out: list[str] = []
    for character in text:
        if "｡" <= character <= "ﾟ":
            out.append(unicodedata.normalize("NFKC", character))
        elif character == "゛":
            out.append("゙")
        elif character == "゜":
            out.append("゚")
        else:
            out.append(character)
    return "".join(out)


# Tags of `given_name_for_constraint` values. H, K and G mark whole CJK given names.
# J marks a short title-case given name that may be joined initials ("Ka", "Mal"), and
# W a capital initial run into a given name ("MArfan"). S marks a full first name with a
# middle of joined capitals ("Johanna IP" gives "Sip"), which canonical names keep as one
# token. Rust `text_compat` mirrors them.
_CJK_SCRIPT_TAGS = frozenset("HKG")


def _cjk_script_tag(given: str) -> str:
    """Return "G" for a given name with hangul, else "K" with kana, else "H" (Han only)."""
    codes = [ord(character) for character in given]
    if any(start <= code <= end for code in codes for start, end in _HANGUL_NAME_RANGES):
        return "G"
    if any(start <= code <= end for code in codes for start, end in _KANA_NAME_RANGES):
        return "K"
    return "H"


def is_cjk_given_name(first_raw: str | None, middle_raw: str | None) -> bool:
    """Return whether a raw given name is written in Han, kana or hangul only.

    The given name is the raw first and middle fields together. It needs at least
    one CJK character, and its other characters must transliterate to no letter.
    Spaces, punctuation and digits may appear. Rust `is_cjk_given_name` mirrors it.

    Args:
        first_raw: Raw first name, or ``None``.
        middle_raw: Raw middle name, or ``None``.

    Returns:
        True for a given name such as ``重喜``, ``ひろし`` or ``민수``; False for
        ``Wei``, ``重喜 Shigeki`` or an empty name.
    """

    given = f"{first_raw or ''} {middle_raw or ''}"
    if given.isascii() or not any(map(_is_cjk_name_character, given)):
        return False
    other = "".join(" " if _is_cjk_name_character(character) else character for character in given)
    return not _canonical_name_tokens(_canonical_name_pretranslate(other))


def cjk_given_name_for_constraint(first_raw: str | None, middle_raw: str | None) -> str | None:
    """Prepare a whole CJK given name for the hard constraint.

    Args:
        first_raw: Raw first name, or ``None``.
        middle_raw: Raw middle name, or ``None``.

    Returns:
        A script tag (``_cjk_script_tag``) followed by the joined transliterated
        letters, an empty string for incomplete CJK evidence, or ``None`` when the
        given name is not CJK. The hard rule compares two values whole only when
        both are complete and share a script; otherwise the ordinary rules apply.
    """
    if not is_cjk_given_name(first_raw, middle_raw):
        return None

    raw = _canonical_name_pretranslate((first_raw or "") + (middle_raw or ""))
    given = unicodedata.normalize("NFC", _widen_kana("".join(raw.split())))
    previous = ""
    parts: list[str] = []
    for character in given:
        if character == "々":
            if not previous or not any(start <= ord(previous) <= end for start, end in _HAN_NAME_RANGES):
                return ""
            character = previous
        letters = _RE_CANONICAL_NON_LETTER.sub("", unidecode(character).lower())
        # A partial transliteration can make equivalent names look different. Any non-ASCII
        # character without letters marks the name incomplete, so the ordinary rules apply.
        # Unicode categories are not used: Python and Rust ship different Unicode versions.
        if not letters and not character.isascii():
            return ""
        parts.append(letters)
        previous = character
    return _cjk_script_tag(given) + "".join(parts)


class CanonicalNameParts(NamedTuple):
    """Canonical (canonical_v2) first/middle/last name fields."""

    first: str
    middle: str
    last: str


CanonicalNameParts.__module__ = "s2and.text"


def _romanize_armenian(text: str) -> str:
    """Romanize the Armenian letters of ``text`` with scholar's table (D10).

    ու is ``u``. A word-initial ե is ``ye``, and a word-initial ո is ``vo``
    unless վ follows. As in scholar, the neighbors of a letter skip
    punctuation, nonspacing marks, and format and control characters, and a
    word starts after a character that is not a letter (Unicode category L).
    A capital gives a capitalized romanization.
    """

    for ligature, letters in _ARMENIAN_LIGATURES.items():
        text = text.replace(ligature, letters)
    small = [_ARMENIAN_SMALL_FROM_CAPITAL.get(character, character) for character in text]
    # The nearest kept neighbor on each side; "" at either end of the text.
    kept = [
        not (category in _ARMENIAN_CONTEXT_SKIPPED_CATEGORIES or category[0] == "P")
        for category in map(unicodedata.category, text)
    ]
    previous_kept = [""] * len(text)
    last = ""
    for index in range(len(text)):
        previous_kept[index] = last
        if kept[index]:
            last = small[index]
    following_kept = [""] * len(text)
    last = ""
    for index in range(len(text) - 1, -1, -1):
        following_kept[index] = last
        if kept[index]:
            last = small[index]
    pieces: list[str] = []
    for index, character in enumerate(text):
        letter = small[index]
        roman = _ARMENIAN_ROMANIZATION.get(letter)
        if roman is None:
            pieces.append(character)
            continue
        previous = previous_kept[index]
        following = following_kept[index]
        word_start = not previous or not unicodedata.category(previous).startswith("L")
        if letter == "ո" and following == "ւ":
            roman = "u"
        elif letter == "ւ" and previous == "ո":
            roman = ""
        elif letter == "ե" and word_start:
            roman = "ye"
        elif letter == "ո" and word_start and following != "վ":
            roman = "vo"
        pieces.append(roman.capitalize() if character != letter else roman)
    return "".join(pieces)


def _canonical_name_pretranslate(raw: str | None) -> str:
    """Delete invisible format controls, unify apostrophe-like marks to ASCII
    apostrophe, unify dash-like characters to ASCII hyphen, and romanize
    Georgian and Armenian letters (D10) — all on the raw code points, before
    transliteration."""

    text = (raw or "").translate(_CANONICAL_NAME_TRANSLATION)
    # Most names are ASCII, so test that first and skip the regex.
    if not text.isascii() and _RE_ARMENIAN_LETTER.search(text) is not None:
        return _romanize_armenian(text)
    return text


def _canonical_name_tokens(pretranslated: str) -> list[str]:
    """Transliterate, lowercase, delete apostrophes/backticks, and split on
    everything that is not a letter (dashes included: binding is decided by the
    caller before this runs)."""

    ascii_text = unidecode(pretranslated).lower().replace("'", "").replace("`", "")
    return _RE_CANONICAL_NON_LETTER.sub(" ", ascii_text).split()


def canonicalize_name_text(raw: str | None) -> str:
    """Canonicalize a whole name string to spaced canonical_v2 tokens.

    Applies the canonical_v2 character pipeline (invisible-format deletion,
    apostrophe-like deletion, uniform dash separators, transliteration) without
    the first/middle split or title-prefix drop. This is the normalization used
    for canonical middle and last fields, and for artifact generators that
    operate on complete name strings (name tuples, ORCID prefix counts).
    """

    return " ".join(_canonical_name_tokens(_canonical_name_pretranslate(raw)))


def last_name_allows_joined_initials(last_raw: str | None) -> bool:
    """Return the D9 letter-case guard for one raw last name.

    Letter case comes from the canonical transliteration of the raw last name
    before lowercasing. ASCII ``a-z`` counts as lowercase and ``A-Z`` counts as
    uppercase, so ``á`` counts as lowercase. The pinned transliteration tables
    make this definition identical in Python and Rust.

    Args:
        last_raw: Raw last name, or ``None``.

    Returns:
        True when the transliterated last name has at least one lowercase
        letter and no more uppercase than lowercase letters.
    """

    transliterated = unidecode(_canonical_name_pretranslate(last_raw))
    lowercase_count = len(_RE_ASCII_LOWERCASE.findall(transliterated))
    return lowercase_count > 0 and lowercase_count >= len(_RE_ASCII_UPPERCASE.findall(transliterated))


def _joined_initials(first_raw: str | None) -> str | None:
    """Return the lowercase D9 initials of a raw first name, or ``None``.

    This applies every D9 condition except the last-name guard.
    """

    if not first_raw:
        return None
    joined = "".join(first_raw.split())
    letters = joined.replace(".", "")
    if _RE_JOINED_INITIALS.fullmatch(letters) is not None:
        if letters in _JOINED_INITIALS_GIVEN_NAMES:
            return None
        if len(letters) == 2 and letters in _DOTTED_GIVEN_NAME_ABBREVIATIONS and first_raw.strip().endswith("."):
            return None
        initials = letters.lower()
    elif _RE_MIXED_CASE_INITIALS.fullmatch(joined) is not None:
        initials = joined.lower()
        # A vowel or y can make a short name, such as "Ty" or "Bo".
        if not _MIXED_CASE_INITIALS_VOWELS.isdisjoint(initials) or initials in _MIXED_CASE_NON_INITIALS:
            return None
    else:
        return None
    # Titles keep the D7 drop: "DR" and "Dr" are not the initials "d r".
    return None if initials in NAME_PREFIXES else initials


def canonicalize_name_parts(
    first_raw: str | None,
    middle_raw: str | None,
    last_raw: str | None,
    *,
    last_allows_initials: bool | None = None,
) -> CanonicalNameParts:
    """Canonicalize raw first/middle/last per the canonical_v2 pipeline.

    Pipeline (docs/canonical_names.md):
    - None is missing/empty; NBSP is whitespace; soft hyphen and zero-width
      joiner are deleted, not separators.
    - Apostrophe-like marks are deleted globally (D2/D3); dash-like characters
      are uniform separators (D4).
    - At most one leading title-prefix token is dropped from first; ``md`` is
      retained as a given-name abbreviation (D7).
    - First/middle split (D1): a leading dash-bound group stays together in
      first as spaced tokens; otherwise the first token stays and later tokens
      spill into middle ahead of existing middle tokens. Space tokens after a
      dash-bound group still spill.
    - Last keeps normalized spaces with particles preserved (D5); suffix
      stripping is outside canonical_v2.
    - Joined initials (D9): when whitespace and periods are removed, a first
      of 2-3 ASCII capitals such as ``HG`` or ``J.H.`` becomes a first initial
      plus middle initials ahead of existing middle tokens. The last name must
      pass ``last_name_allows_joined_initials``. ``ED``, ``MD``, title
      prefixes, and dotted given-name abbreviations such as ``PH.`` do not
      change. A first of one capital and 1-2 lowercase letters with no vowel,
      no y, and no period, such as ``Jh``, changes the same way. Listed
      non-initials such as ``Md``, ``Ch``, and ``Jr`` do not change.

    Args:
        first_raw: Raw first name, or ``None``.
        middle_raw: Raw middle name, or ``None``.
        last_raw: Raw last name, or ``None``.
        last_allows_initials: The precomputed
            ``last_name_allows_joined_initials(last_raw)`` result. Supply it
            only when first and last names are canonicalized separately, such
            as in a count generator with one map for each distinct name. The
            default computes it from ``last_raw``.

    Returns:
        The canonical first, middle, and last fields.
    """

    middle_tokens = _canonical_name_tokens(_canonical_name_pretranslate(middle_raw))
    last_tokens = _canonical_name_tokens(_canonical_name_pretranslate(last_raw))
    initials = _joined_initials(first_raw)
    if initials is not None:
        if last_allows_initials is None:
            last_allows_initials = last_name_allows_joined_initials(last_raw)
        if last_allows_initials:
            return CanonicalNameParts(
                first=initials[0],
                middle=" ".join([*initials[1:], *middle_tokens]),
                last=" ".join(last_tokens),
            )

    first_clean = _canonical_name_pretranslate(first_raw)

    # Whitespace chunks of the raw first field, each normalized to tokens and
    # tagged with whether a dash bound it together.
    flattened: list[tuple[str, int]] = []
    dash_bound: list[bool] = []
    for group_index, chunk in enumerate(first_clean.split()):
        dash_bound.append("-" in chunk)
        for token in _canonical_name_tokens(chunk.replace("-", " ")):
            flattened.append((token, group_index))

    if flattened and flattened[0][0] in NAME_PREFIXES:
        flattened = flattened[1:]

    if not flattened:
        first_field_tokens: list[str] = []
        spilled_tokens: list[str] = []
    else:
        lead_group = flattened[0][1]
        if dash_bound[lead_group]:
            first_field_tokens = [token for token, group in flattened if group == lead_group]
            spilled_tokens = [token for token, group in flattened if group != lead_group]
        else:
            first_field_tokens = [flattened[0][0]]
            spilled_tokens = [token for token, _ in flattened[1:]]

    return CanonicalNameParts(
        first=" ".join(first_field_tokens),
        middle=" ".join(spilled_tokens + middle_tokens),
        last=" ".join(last_tokens),
    )


def canonical_name_count_keys(parts: CanonicalNameParts) -> dict[str, str | None]:
    """Build canonical_v2 count keys from canonical fields after gating (D6/D8).

    A None key means no lookup (NaN feature), never a sentinel count. ``first``
    and ``first_last`` require an informative first (string length > 1);
    ``last_first_initial`` requires first and last present and stays
    initial-char semantics.
    """

    first_informative = len(parts.first) > 1
    first_key = parts.first if first_informative else None
    last_key = parts.last if parts.last else None
    return {
        "first": first_key,
        "last": last_key,
        "first_last": f"{parts.first} {parts.last}" if (first_informative and last_key) else None,
        "last_first_initial": f"{parts.last} {parts.first[0]}" if (parts.first and last_key) else None,
    }


def canonical_lasts_equivalent(last_a: str, last_b: str) -> bool:
    """Compare-time equivalence for canonical last names (space-insensitive).

    Canonical surnames are STORED spaced with particles preserved (D5:
    ``ou yang``, ``van der berg``); at compare time, joined and spaced
    spellings of the same surname are treated as equivalent (``ou yang`` ==
    ``ouyang``). This is deliberate compare-time policy, not an artifact shim:
    upstream blocking groups surname spelling variants under one block key, and
    the within-block last-name constraint must not veto pairs that blocking
    deliberately grouped (ruled 2026-07-09 with the canonical_v2 cutover).
    For the same reason, listed spelling variants (``muller`` == ``mueller``)
    and Slavic -sky endings (``tartakovskiy`` == ``tartakovsky``) are
    equivalent; see ``canonical_surname_family``.
    """

    if last_a == last_b:
        return True
    return canonical_surname_family(last_a) == canonical_surname_family(last_b)


SURNAME_VARIANTS_FILENAME = "s2and_surname_variants_canonical.txt"


# A Slavic -sky ending after at least two letters; every spelling maps to -skii.
_SLAVIC_SKY_ENDING = re.compile(r"(?<=[a-z]{2})sk(?:iy|yi|yy|ii|ij|yj|y)\Z")


@lru_cache(maxsize=1)
def surname_variant_map() -> dict[str, str]:
    """Load the packaged spelling variants of one surname (``mueller`` -> ``muller``).

    Each line is ``variant,canonical`` in canonical last-name form without
    spaces. Blocking puts both spellings under one block key, so the
    last-name constraint must not veto them.

    Returns:
        A map from each variant spelling to its canonical spelling.

    Raises:
        ValueError: When the file is unsorted, malformed or chains variants.
    """

    path = Path(__file__).parent.parent / "data" / SURNAME_VARIANTS_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines != sorted(lines):
        raise ValueError(f"{path} must be sorted")
    variants: dict[str, str] = {}
    for line in lines:
        fields = line.split(",")
        if len(fields) != 2 or not all(re.fullmatch(r"[a-z]+", field) for field in fields):
            raise ValueError(f"{path}: malformed line {line!r}")
        variant, canonical = fields
        if variant == canonical or variant in variants:
            raise ValueError(f"{path}: repeated or self-mapped variant {variant!r}")
        variants[variant] = canonical
    chained = set(variants) & set(variants.values())
    if chained:
        raise ValueError(f"{path}: canonical spellings that are also variants: {sorted(chained)[:5]}")
    return variants


# Training compares millions of pairs over far fewer distinct last names, so cache the form.
@lru_cache(maxsize=1 << 16)
def canonical_surname_family(last: str) -> str:
    """Return the compare-time form of a canonical last name.

    Removes spaces, maps a listed spelling variant to its canonical spelling,
    and writes every Slavic -sky ending as -skii. Two last names with the same
    form never trigger the last-name constraint. Name counts and features keep
    the stored spelling.

    Args:
        last: A canonical last name.

    Returns:
        The last name in its compare-time form.
    """

    compact = last.replace(" ", "")
    compact = surname_variant_map().get(compact, compact)
    return _SLAVIC_SKY_ENDING.sub("skii", compact)
