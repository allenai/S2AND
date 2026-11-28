"""Normalize text, language, and identifiers."""

import logging
import re
from collections import Counter
from typing import Any, NamedTuple

import pycld2 as cld2
from text_unidecode import unidecode

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


class LanguageDetection(NamedTuple):
    """CLD2 language detection result used by pairwise language features."""

    is_reliable: bool
    is_english: bool
    predicted_language: str
    language_reliability: float


LanguageDetection.__module__ = "s2and.text"


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


def normalize_text(text: str | None) -> str:
    """Normalize free text such as venues, affiliations, and suffixes.

    Author names use ``canonicalize_name_parts`` instead. This normalizer keeps
    the trained model's text features stable: it replaces apostrophes with
    spaces rather than deleting them.

    Args:
        text: Raw text, or ``None``.

    Returns:
        Lowercase ASCII letters separated by single spaces.
    """
    if text is None or len(text) == 0:
        return ""

    norm_text = unidecode(text).lower()
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
