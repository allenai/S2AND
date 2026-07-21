import logging
import re
from collections import Counter
from collections.abc import Set
from typing import TYPE_CHECKING, Any, NamedTuple

import jellyfish
import numpy as np
import pycld2 as cld2
from numpy import inner
from numpy.linalg import norm
from strsimpy.metric_lcs import MetricLCS
from text_unidecode import unidecode

from s2and.consts import NUMPY_NAN

if TYPE_CHECKING:
    from s2and.data import NameCounts

logger = logging.getLogger("s2and")


RE_NORMALIZE_WHOLE_NAME = re.compile(r"[^a-zA-Z\s]+")
RE_NORMALIZE_TITLE = re.compile(r"[^a-zA-Z0-9\s]+")

DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d"
NAME_DASH_CHARS = frozenset(DASH_CHARS)
ORCID_DASH_CLASS = re.escape(DASH_CHARS)
# Digit groups are matched with the explicit ASCII class [0-9] (not \d) so that
# Unicode digit code points (Arabic-Indic, etc.) are rejected, matching the Rust
# is_ascii_digit() behavior in s2and_rust/src/orcid.rs.
ORCID_PATTERN = re.compile(
    rf"(?i)(?<![0-9x])"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}]?"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}]?"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}]?"
    rf"[0-9]{{3}}[0-9x](?![0-9x])"
)

DROPPED_AFFIXES = {
    "ab",
    "am",
    "ap",
    "abu",
    "al",
    "auf",
    "aus",
    "bar",
    "bath",
    "bat",
    "bet",
    "bint",
    "dall",
    "dalla",
    "das",
    "de",
    "degli",
    "del",
    "dell",
    "della",
    "dem",
    "den",
    "der",
    "di",
    "do",
    "dos",
    "ds",
    "du",
    "el",
    "ibn",
    "im",
    "jr",
    "la",
    "las",
    "le",
    "los",
    "mac",
    "mc",
    "mhic",
    "mic",
    "ter",
    "und",
    "van",
    "vom",
    "von",
    "zu",
    "zum",
    "zur",
}


# Stop-words list must be updated for citations title/abstract related information
STOPWORDS = set(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ]
)

AFFILIATIONS_STOP_WORDS = STOPWORDS.union(
    {
        "university",
        "college",
        "lab",
        "organization",
        "department",
        "research",
        "institute",
        "school",
        "academy",
        "national",
        "laboratory",
    }
)

VENUE_STOP_WORDS = STOPWORDS.union(
    {
        "proceedings",
        "journal",
        "conference",
        "transactions",
        "international",
        "society",
        "letters",
        "official",
        "research",
        "association",
    }
)

NAME_PREFIXES = {"dr", "prof", "professor", "mr", "miss", "mrs", "ms", "mx", "sir", "phd", "doctor"}


def prefix_dist(string_1: str, string_2: str) -> float:
    if string_1 == string_2:
        return 0.0
    min_word, max_word = (string_1, string_2) if len(string_1) < len(string_2) else (string_2, string_1)
    min_len = len(min_word)
    for i in range(min_len, 0, -1):
        if min_word[:i] == max_word[:i]:
            return 1 - (i / min_len)
    return 1.0


metric_lcs = MetricLCS()
TEXT_FUNCTIONS = [
    (jellyfish.levenshtein_distance, "levenshtein"),
    (prefix_dist, "prefix"),
    (metric_lcs.distance, "lcs"),
    (jellyfish.jaro_winkler_similarity, "jaro"),
]


class LanguageDetection(NamedTuple):
    """CLD2 language detection result used by pairwise language features."""

    is_reliable: bool
    is_english: bool
    predicted_language: str
    language_reliability: float


def _unknown_language_detection() -> LanguageDetection:
    """Return the normalized unknown-language detection result."""

    return LanguageDetection(False, False, "un", 0.0)


def detect_language(text: str | None) -> LanguageDetection:
    """Detect title language with CLD2 only.

    `predicted_language` is CLD2's top language code when known, even if CLD2
    does not mark the detection reliable. `language_reliability` is the CLD2
    top-language percent divided by 100, but only when CLD2 reports the top
    language as reliable and known; otherwise it is 0.0.
    """

    text = text or ""
    if len(text.split()) <= 1:
        return _unknown_language_detection()

    if not [character.isupper() for character in text if character.isalpha()]:
        return _unknown_language_detection()

    try:
        # Titles are plain text. Pin the mode explicitly so Python matches the
        # Rust detector instead of inheriting pycld2's HTML-aware default.
        cld2_pred = cld2.detect(text, isPlainText=True)
    except (UnicodeError, cld2.error):
        logger.exception("cld2 language detection failed; using unknown language marker")
        return _unknown_language_detection()

    top_language = cld2_pred[2][0]
    predicted_language = str(top_language[1])
    if predicted_language == "un":
        return _unknown_language_detection()

    is_reliable = bool(cld2_pred[0])
    language_reliability = float(top_language[2]) / 100.0 if is_reliable else 0.0
    return LanguageDetection(
        is_reliable=is_reliable,
        is_english=predicted_language == "en",
        predicted_language=predicted_language,
        language_reliability=language_reliability,
    )


def normalize_text(text: str | None, special_case_apostrophes: bool = False) -> str:
    """
    Normalize text.

    Parameters
    ----------
    text: string
        the text to normalize
    special_case_apostrophie: bool
        whether to replace apostrophes with empty strings rather than spaces

    Returns
    -------
    string: the normalized text
    """
    if text is None or len(text) == 0:
        return ""

    norm_text = unidecode(text).lower()

    if special_case_apostrophes:
        norm_text = norm_text.replace("'", "")

    norm_text = RE_NORMALIZE_WHOLE_NAME.sub(" ", norm_text)
    norm_text = re.sub(r"\s+", " ", norm_text).strip()

    return norm_text


def normalize_title(text: str | None) -> str:
    """Normalize publication titles while preserving identifying digits.

    Person-name normalization intentionally removes digits. Titles have a
    different contract: section numbers, years, and formula subscripts such as
    ``Co3O4`` are evidence and must survive normalization.

    Args:
        text: Raw publication title, or ``None``.

    Returns:
        Lowercase transliterated title containing only ASCII letters, digits,
        and single spaces.
    """

    if not text:
        return ""
    normalized = unidecode(text).lower()
    normalized = RE_NORMALIZE_TITLE.sub(" ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_orcid(value: Any) -> str | None:
    """Return a canonical hyphenated ORCID, or None when no valid ORCID is present."""

    if value is None:
        return None
    match = ORCID_PATTERN.search(str(value).strip())
    if match is None:
        return None
    compact = "".join(character for character in match.group(0) if character not in NAME_DASH_CHARS).upper()
    return f"{compact[0:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:16]}"


def normalize_orcid_compact(value: Any) -> str | None:
    """Return a compact legacy ORCID key, or None when no valid ORCID is present."""

    normalized = normalize_orcid(value)
    return None if normalized is None else normalized.replace("-", "")


def has_name_dash(value: str | None) -> bool:
    """Return whether a raw name contains a dash-like character."""

    return any(character in NAME_DASH_CHARS for character in value or "")


# ------------------------ canonical_v2 name canonicalization ------------------------
# These functions are the live canonical normalization surface used by ingestion,
# feature extraction, subblocking, and query adaptation. The remaining release work
# in docs/normalization_migration_blocked.md concerns artifact regeneration and model
# retraining. Policy decisions D1-D8 are recorded in the fixture's decisions registry.

# D3 apostrophe-like marks: ASCII apostrophe, backtick, spacing acute, curly quotes,
# modifier letters (okina/apostrophe), primes, saltillo, U+FE4D (classified with
# apostrophe-like marks by issue #39 despite its Unicode name), fullwidth apostrophe.
NAME_APOSTROPHE_LIKE_CHARS = frozenset("'`\u00b4\u2018\u2019\u02bb\u02bc\u2032\u2035\ua78c\ufe4d\uff07")

# Invisible formatting controls deleted before tokenization: soft hyphen (not a dash
# separator) and zero-width joiner.
_NAME_INVISIBLE_FORMAT_CHARS = "\u00ad\u200d"

_CANONICAL_NAME_TRANSLATION = str.maketrans(
    {
        **{character: None for character in _NAME_INVISIBLE_FORMAT_CHARS},
        **{character: "'" for character in NAME_APOSTROPHE_LIKE_CHARS},
        **{character: "-" for character in DASH_CHARS},
    }
)

_RE_CANONICAL_NON_LETTER = re.compile(r"[^a-z]+")


class CanonicalNameParts(NamedTuple):
    """Canonical (canonical_v2) first/middle/last name fields."""

    first: str
    middle: str
    last: str


def _canonical_name_pretranslate(raw: str | None) -> str:
    """Delete invisible format controls, unify apostrophe-like marks to ASCII
    apostrophe, and unify dash-like characters to ASCII hyphen — all on the raw
    code points, before transliteration."""

    return (raw or "").translate(_CANONICAL_NAME_TRANSLATION)


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


def canonicalize_name_parts(
    first_raw: str | None,
    middle_raw: str | None,
    last_raw: str | None,
) -> CanonicalNameParts:
    """Canonicalize raw first/middle/last per the canonical_v2 pipeline.

    Pipeline (docs/normalization_migration_blocked.md):
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
    """

    first_clean = _canonical_name_pretranslate(first_raw)
    middle_tokens = _canonical_name_tokens(_canonical_name_pretranslate(middle_raw))
    last_tokens = _canonical_name_tokens(_canonical_name_pretranslate(last_raw))

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
    """

    if last_a == last_b:
        return True
    return last_a.replace(" ", "") == last_b.replace(" ", "")


def name_text_features(
    name_1: str,
    name_2: str,
    default_val: float = NUMPY_NAN,
) -> list[float]:
    """
    Computes various text similarity features for two names

    Parameters
    ----------
    name_1: string
        the first name
    name_2: string
        the second name
    default_val: float
        the default value to return when one or both of the names is empty

    Returns
    -------
    List[float]: a list of the various similarity scores for the two names
    """
    scores = []
    if name_1 is None or name_2 is None or len(name_1) == 0 or len(name_2) == 0:
        return [default_val] * len(TEXT_FUNCTIONS)

    for function, function_name in TEXT_FUNCTIONS:
        score = function(name_1, name_2)
        if function_name in {"levenshtein"}:
            score = score / max(len(name_1), len(name_2))
        scores.append(score)
    return scores


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors

    Parameters
    ----------
    a: np.ndarray
        the first vector
    b: np.ndarray
        the second vector

    Returns
    -------
    float: the cosine similarity of the two vectors
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_norm = norm(a)
    b_norm = norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    else:
        return inner(a, b) / (a_norm * b_norm)


def email_prefix_suffix(email: str) -> tuple[str | None, str | None]:
    """Return normalized email components, or missing values when malformed.

    A valid feature input has exactly one ``@``, nonempty local and domain
    components after edge-dot normalization, and no whitespace. Malformed
    values return ``(None, None)`` so neither exact-match feature can become
    positive evidence. This mirrors Rust ``email_parts``.
    """
    prefix_raw, separator, suffix_raw = email.partition("@")
    whitespace_parts = email.split()
    if (
        not separator
        or not prefix_raw
        or not suffix_raw
        or "@" in suffix_raw
        or len(whitespace_parts) != 1
        or whitespace_parts[0] != email
    ):
        return None, None
    prefix = prefix_raw.strip(".").lower()
    suffix = suffix_raw.strip(".").lower()
    if not prefix or not suffix:
        return None, None
    return prefix, suffix


def get_text_ngrams(
    text: str | None,
    use_unigrams: bool = False,
    use_bigrams: bool = True,
    stopwords: set[str] | None = STOPWORDS,
    drop_short_tokens: bool = True,
) -> Counter:
    """
    Get character bigrams, trigrams, quadgrams, and optionally unigrams for a piece of text.
    Note: respects word boundaries

    Parameters
    ----------
    text: string
        the text to get ngrams for
    use_unigrams: bool
        whether or not to include unigrams
    stopwords: Set
        The set of stopwords to filter out before computing character ngrams
    drop_short_tokens: bool
        Whether to drop word tokens of length <= 2 before computing character ngrams.

    Returns
    -------
    Counter: the ngrams present in the text
    """
    if text is None or len(text) == 0:
        return Counter()

    # The short-token filter is independent of stopword removal. Reference-author
    # ngrams use it with stopwords=None, while coauthor ngrams explicitly keep
    # short names like "li" and "wu".
    words = [word for word in text.split(" ") if not drop_short_tokens or len(word) > 2]
    if stopwords is not None:
        words = [word for word in words if word not in stopwords]
    text = " ".join(words)

    unigrams = []
    if use_unigrams:
        unigrams = filter(lambda x: " " not in x, text)

    bigrams = []
    if use_bigrams:
        bigrams = map(
            lambda x: "".join(x),
            filter(lambda x: " " not in x, zip(text, text[1:], strict=False)),
        )

    trigrams = map(
        lambda x: "".join(x),
        filter(lambda x: " " not in x, zip(text, text[1:], text[2:], strict=False)),
    )

    quadgrams = map(
        lambda x: "".join(x),
        filter(lambda x: " " not in x, zip(text, text[1:], text[2:], text[3:], strict=False)),
    )
    ngrams: Counter = Counter()
    ngrams.update(Counter(unigrams))
    ngrams.update(Counter(bigrams))
    ngrams.update(Counter(trigrams))
    ngrams.update(Counter(quadgrams))
    return ngrams


def get_text_ngrams_words(
    text: str | None,
    stopwords: set[str] = STOPWORDS,
    *,
    drop_short_tokens: bool = True,
) -> Counter:
    """
    Get word unigrams, bigrams, and trigrams for a piece of text.

    Parameters
    ----------
    text: string
        the text to get ngrams for
    stopwords: Set
        The set of stopwords to filter out before computing word ngrams
    drop_short_tokens: bool
        Whether to drop one-character tokens. Titles disable this so section
        and formula digits remain evidence.

    Returns
    -------
    Counter: the ngrams present in the text
    """
    if text is None or len(text) == 0:
        return Counter()
    text_split = [word for word in text.split() if word not in stopwords and (not drop_short_tokens or len(word) > 1)]
    unigrams = Counter(text_split)
    bigrams = map(
        lambda x: " ".join(x),
        zip(text_split, text_split[1:], strict=False),
    )
    trigrams = map(
        lambda x: " ".join(x),
        zip(text_split, text_split[1:], text_split[2:], strict=False),
    )
    ngrams: Counter = Counter()
    ngrams.update(unigrams)
    ngrams.update(Counter(bigrams))
    ngrams.update(Counter(trigrams))
    return ngrams


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
    name-tuple artifact. The legacy joined/first-token probing forms were retired
    with the canonical_v2 cutover.
    """

    if not first_a.split() or not first_b.split():
        # Missing first-name evidence is unknown, not an incompatibility signal.
        return True

    if same_prefix_tokens(first_a, first_b):
        return True

    pair = canonical_name_tuple_pair(first_a, first_b)
    return pair in name_tuples or (pair[1], pair[0]) in name_tuples


def equal(
    name_1: str | None,
    name_2: str | None,
    default_val: float = NUMPY_NAN,
) -> int | float:
    """
    Check if two names are exactly equal after lowercasing

    Parameters
    ----------
    name_1: string
        the first name
    name_2: string
        the second name
    default_val: float
        the default value to return when one or both of the names is empty

    Returns
    -------
    int: 0 (if unequal) or 1 (if equal)
    """
    if name_1 is None or name_2 is None:
        return default_val

    # Strip/lower first, then test emptiness, so whitespace-only inputs are
    # treated as empty (return default) rather than comparing equal as "".
    norm_1 = name_1.lower().strip()
    norm_2 = name_2.lower().strip()
    if len(norm_1) == 0 or len(norm_2) == 0:
        return default_val

    if norm_1 == "-" or norm_2 == "-":
        return default_val

    return 1 if norm_1 == norm_2 else 0


def equal_middle(
    name_1: str | None,
    name_2: str | None,
    default_val: float = NUMPY_NAN,
) -> int | float:
    """
    Checks if two middle names are equal. If either middle name is just an initial,
    just check euqality of initials

    Parameters
    ----------
    name_1: string
        first middle name string
    name_2: string
        second middle name string
    default_val: float
        the default value to return when one or both of the names is empty

    Returns
    -------
    int: 0 (if unequal) or 1 (if equal)
    """
    if name_1 is None or name_2 is None or len(name_1) == 0 or len(name_2) == 0:
        return default_val

    # When either side is a single-character initial, compare the sets of
    # token initials so a joined multi-token middle ("james lee") matches the
    # other side's initial for ANY of its tokens, not just the first one.
    if len(name_1) == 1 or len(name_2) == 1:
        for left_token in name_1.split():
            if not left_token:
                continue
            left_initial = left_token[0]
            for right_token in name_2.split():
                if right_token and right_token[0] == left_initial:
                    return 1
        return 0

    return 1 if name_1 == name_2 else 0


def equal_initial(
    name_1: str | None,
    name_2: str | None,
    default_val: float = NUMPY_NAN,
) -> int | float:
    """
    Checks if two initials are qual

    Parameters
    ----------
    name_1: string
        first initial
    name_2: string
        second initial
    default_val: float
        the default value to return when one or both of the names is empty

    Returns
    -------
    int: 0 (if unequal) or 1 (if equal)
    """
    if name_1 is None or name_2 is None or len(name_1) == 0 or len(name_2) == 0:
        return default_val

    if name_1.strip().lower()[0] == name_2.strip().lower()[0]:
        return 1
    else:
        return 0


def counter_jaccard(
    counter_1: Counter, counter_2: Counter, default_val: float = NUMPY_NAN, denominator_max: float = np.inf
) -> float:
    """
    Computes jaccard overlap between two Counters

    Parameters
    ----------
    counter_1: Counter
        first Counter
    counter_2: Counter
        second Counter
    default_val: float
        the default value to return when one or both of the Counters is empty

    Returns
    -------
    float: the jaccard overlap
    """
    if len(counter_1) == 0 or len(counter_2) == 0:
        return default_val

    intersection_sum = sum((counter_1 & counter_2).values())
    union_sum = sum(counter_1.values()) + sum(counter_2.values()) - intersection_sum
    score = intersection_sum / min(union_sum, denominator_max)
    return min(score, 1)


def jaccard(
    set_1: set,
    set_2: set,
    default_val: float = NUMPY_NAN,
) -> float:
    """
    Computes jaccard overlap between two sets

    Parameters
    ----------
    set_1: Set
        first Set
    set_2: Set
        second Set
    default_val: float
        the default value to return when one or both of the Sets is empty

    Returns
    -------
    float: the jaccard overlap
    """
    if len(set_1) == 0 or len(set_2) == 0:
        return default_val

    score = len(set_1.intersection(set_2)) / (len(set_1.union(set_2)))
    return score


def compute_block(name: str) -> str:
    """
    Compute block for a name.
    Override for other definition of blocks. This function gives block as first initial + last name.

    Parameters
    ----------
    name: string
        the name to block

    Returns
    -------
    string: the block string
    """
    if len(name) == 0:
        return ""

    name_parts = name.split(" ")
    if len(name_parts) == 1:
        return name_parts[0]
    block = name_parts[0][0] + " " + name_parts[-1]
    return block


def diff(value_1: float | None, value_2: float | None, default_val: float = NUMPY_NAN) -> float:
    """
    Compute absolute difference between two values.

    Parameters
    ----------
    value_1: float
        first value
    value_2: float
        second value
    default_val: float
        the default value to return when one or both of the values is empty

    Returns
    -------
    float: absolute difference
    """
    if value_1 is None or value_2 is None:
        return default_val

    return abs(float(value_1) - float(value_2))


def name_counts(
    counts_1: "NameCounts",
    counts_2: "NameCounts",
) -> list[int | float]:
    """
    Gets name counts for first, last, and first_last names.
    These counts were computed from the entire S2 corpus.

    Parameters
    ----------
    counts_1: NameCounts
        first NameCounts
    counts_2: NameCounts
        second NameCounts

    Returns
    -------
    List[int]: min/max for first, first_last, and min for last, last_first_initial
    """
    counts = []
    counts.append(
        [
            counts_1.first,  # can be nan
            counts_1.first_last,  # can be nan
            counts_1.last,
            counts_1.last_first_initial,
        ]
    )
    counts.append(
        [
            counts_2.first,  # can be nan
            counts_2.first_last,  # can be nan
            counts_2.last,
            counts_2.last_first_initial,
        ]
    )
    # fmin ignores one missing count but preserves NaN when both counts are missing.
    counts_array = np.array(counts, dtype=float)
    counts_min_max = list(np.fmin.reduce(counts_array, axis=0)) + list(np.max(counts_array[:, :2], axis=0))

    return counts_min_max
