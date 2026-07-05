import logging
import os
import re
import threading
import warnings
from collections import Counter
from collections.abc import Set
from typing import TYPE_CHECKING, Any

import fasttext
import jellyfish
import numpy as np
import pycld2 as cld2
from numpy import inner
from numpy.linalg import norm
from strsimpy.metric_lcs import MetricLCS
from text_unidecode import unidecode

from s2and.consts import FASTTEXT_PATH, NUMPY_NAN
from s2and.file_cache import cached_path

if TYPE_CHECKING:
    from s2and.data import NameCounts

logger = logging.getLogger("s2and")

# Lazily-loaded fastText model to avoid heavy import-time cost
_FASTTEXT_MODEL = None
_FASTTEXT_MODEL_INITIALIZED = False
_FASTTEXT_LOADING_ENABLED = True
_FASTTEXT_LOAD_FAILED = False
_FASTTEXT_MODEL_LOCK = threading.Lock()


def set_fasttext_loading_enabled(enabled: bool) -> None:
    """Configure whether language detection may load the fastText model."""

    global _FASTTEXT_LOADING_ENABLED
    global _FASTTEXT_MODEL
    global _FASTTEXT_MODEL_INITIALIZED
    global _FASTTEXT_LOAD_FAILED
    with _FASTTEXT_MODEL_LOCK:
        resolved_enabled = bool(enabled)
        if _FASTTEXT_LOADING_ENABLED == resolved_enabled:
            if not resolved_enabled:
                _FASTTEXT_MODEL = None
                _FASTTEXT_MODEL_INITIALIZED = True
                _FASTTEXT_LOAD_FAILED = False
            elif (
                _FASTTEXT_MODEL is None
                and _FASTTEXT_MODEL_INITIALIZED
                and not _FASTTEXT_LOAD_FAILED
                and os.environ.get("S2AND_SKIP_FASTTEXT", "").lower() not in {"1", "true", "yes"}
            ):
                _FASTTEXT_MODEL_INITIALIZED = False
            return
        _FASTTEXT_LOADING_ENABLED = resolved_enabled
        if resolved_enabled:
            if _FASTTEXT_MODEL is not None or not _FASTTEXT_LOAD_FAILED:
                _FASTTEXT_MODEL_INITIALIZED = False
            return
        _FASTTEXT_MODEL = None
        _FASTTEXT_MODEL_INITIALIZED = True
        _FASTTEXT_LOAD_FAILED = False


def fasttext_loading_enabled() -> bool:
    """Return whether language detection may load the fastText model."""

    with _FASTTEXT_MODEL_LOCK:
        return bool(_FASTTEXT_LOADING_ENABLED)


def _get_fasttext_model():
    """Return a cached fastText model instance, loading on first use."""

    global _FASTTEXT_MODEL
    global _FASTTEXT_MODEL_INITIALIZED
    global _FASTTEXT_LOAD_FAILED
    if os.environ.get("S2AND_SKIP_FASTTEXT", "").lower() in {"1", "true", "yes"}:
        with _FASTTEXT_MODEL_LOCK:
            _FASTTEXT_MODEL = None
            _FASTTEXT_MODEL_INITIALIZED = True
            _FASTTEXT_LOAD_FAILED = False
        return None
    with _FASTTEXT_MODEL_LOCK:
        if not _FASTTEXT_LOADING_ENABLED:
            _FASTTEXT_MODEL = None
            _FASTTEXT_MODEL_INITIALIZED = True
            return None
        if _FASTTEXT_MODEL_INITIALIZED:
            return _FASTTEXT_MODEL
        try:
            _FASTTEXT_MODEL = fasttext.load_model(cached_path(FASTTEXT_PATH))
            _FASTTEXT_LOAD_FAILED = False
        except (OSError, RuntimeError, ValueError) as err:
            # fastText is mandatory in production: a genuine load failure must
            # surface rather than silently degrading to a cld2-only path. Leave
            # _FASTTEXT_MODEL_INITIALIZED unset so a later call re-attempts (and
            # re-raises). To run without fastText (tests only) set
            # S2AND_SKIP_FASTTEXT=1 or use set_fasttext_loading_enabled(False).
            _FASTTEXT_LOAD_FAILED = True
            raise RuntimeError(
                f"fastText language model is required but failed to load from {FASTTEXT_PATH!r}. "
                "Set S2AND_SKIP_FASTTEXT=1 to disable language detection during testing."
            ) from err
        _FASTTEXT_MODEL_INITIALIZED = True
        return _FASTTEXT_MODEL


RE_NORMALIZE_WHOLE_NAME = re.compile(r"[^a-zA-Z\s]+")

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

NAME_PREFIXES = {"dr", "prof", "professor", "mr", "miss", "mrs", "ms", "mx", "sir", "phd", "md", "doctor"}


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


def reconcile_detected_languages(predicted_language_ft: str, predicted_language_2: str) -> tuple[str, bool]:
    """Reconcile the fastText and cld2 language predictions into a final call.

    A prediction is trusted as reliable only when BOTH detectors return a
    concrete language (neither the ``"un_ft"``/``"un_2"`` unknown sentinel) AND
    they agree. Any non-agreement -- an outright disagreement, or only one
    detector responding -- collapses to ``("un", False)``. This preserves the
    invariant ``is_reliable <=> predicted_language != "un"``.

    Args:
        predicted_language_ft: fastText's language code, or ``"un_ft"`` when
            fastText produced no usable prediction (e.g. disabled during tests).
        predicted_language_2: cld2's language code, or ``"un_2"`` when cld2
            failed or returned unknown.

    Returns:
        A ``(predicted_language, is_reliable)`` tuple.
    """

    ft_known = predicted_language_ft != "un_ft"
    cld2_known = predicted_language_2 != "un_2"
    if ft_known and cld2_known and predicted_language_ft == predicted_language_2:
        return predicted_language_2, True
    return "un", False


def detect_language(text: str):
    if len(text.split()) <= 1:
        return (False, False, "un")

    # fasttext (optional if available)
    isuppers = [c.isupper() for c in text if c.isalpha()]
    if len(isuppers) == 0:
        return (False, False, "un")
    ft_model = _get_fasttext_model()
    if ft_model is not None:
        if sum(isuppers) / len(isuppers) > 0.9:
            fasttext_pred = ft_model.predict(text.lower().replace("\n", " "))
            predicted_language_ft = fasttext_pred[0][0].split("__")[-1]
        else:
            fasttext_pred = ft_model.predict(text.replace("\n", " "))
            predicted_language_ft = fasttext_pred[0][0].split("__")[-1]
    else:
        predicted_language_ft = "un_ft"

    # cld2
    try:
        cld2_pred = cld2.detect(text)
        predicted_language_2 = cld2_pred[2][0][1]
        if predicted_language_2 == "un":
            predicted_language_2 = "un_2"
    except (UnicodeError, cld2.error):
        logger.exception("cld2 language detection failed; using unknown language marker")
        predicted_language_2 = "un_2"

    predicted_language, is_reliable = reconcile_detected_languages(predicted_language_ft, predicted_language_2)

    # is_english can now be obtained
    is_english = predicted_language == "en"

    return is_reliable, is_english, predicted_language


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


def split_first_middle_hyphen_aware(first_raw: str | None, middle_raw: str | None) -> tuple[str, str]:
    """Normalize and split first/middle with hyphen awareness for canonical fields.

    Rules:
    - Apostrophes in first are removed (no spaces introduced).
    - If a hyphen exists in the raw first name, keep all first tokens together (no spill into middle).
    - Otherwise, first token stays in first; remaining first tokens spill into middle.
    - A single leading prefix from NAME_PREFIXES is dropped if present.

    Returns (first_without_apostrophe, middle_without_apostrophe), both already normalized.
    """
    first_raw = first_raw or ""
    middle_raw = middle_raw or ""

    has_dash_in_first = has_name_dash(first_raw)
    first_noapos = normalize_text(first_raw, special_case_apostrophes=True)
    middle_norm = normalize_text(middle_raw)

    f_parts = first_noapos.split()
    m_parts = middle_norm.split()
    if f_parts and f_parts[0] in NAME_PREFIXES:
        f_parts = f_parts[1:]

    if not f_parts:
        return "", " ".join(m_parts)
    if has_dash_in_first:
        return " ".join(f_parts), " ".join(m_parts)
    # Legacy spill behavior
    return f_parts[0], " ".join(f_parts[1:] + m_parts)


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


def email_prefix_suffix(email: str) -> tuple[str, str | None]:
    """Split an email into (prefix, suffix), normalized (dot-stripped, lowercased).

    When the address has no ``@``, the whole string is the prefix and the suffix
    is None (missing) rather than a sentinel like "missing" — so two malformed
    emails do not spuriously match on a shared sentinel suffix. Mirrors the Rust
    ``email_parts`` in s2and_rust/src/features.rs.
    """
    if "@" in email:
        prefix_raw, _, suffix_raw = email.rpartition("@")
        prefix = prefix_raw.replace("@", "").strip(".").lower()
        suffix: str | None = suffix_raw.strip(".").lower()
    else:
        prefix = email.strip(".").lower()
        suffix = None
    return prefix, suffix


def get_text_ngrams(
    text: str | None, use_unigrams: bool = False, use_bigrams: bool = True, stopwords: set[str] | None = STOPWORDS
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

    Returns
    -------
    Counter: the ngrams present in the text
    """
    if text is None or len(text) == 0:
        return Counter()

    # The short-token filter is applied whenever word tokenization happens, and
    # is independent of stopword removal. Callers that pass stopwords=None (e.g.
    # reference-author ngrams) still drop 1-2 char tokens like title/venue do.
    words = [word for word in text.split(" ") if len(word) > 2]
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


def get_text_ngrams_words(text: str | None, stopwords: set[str] = STOPWORDS) -> Counter:
    """
    Get word unigrams, bigrams, and trigrams for a piece of text.

    Parameters
    ----------
    text: string
        the text to get ngrams for
    stopwords: Set
        The set of stopwords to filter out before computing word ngrams

    Returns
    -------
    Counter: the ngrams present in the text
    """
    if text is None or len(text) == 0:
        return Counter()
    text_split = [word for word in text.split() if word not in stopwords and len(word) > 1]
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
    for x, y in zip(ta, tb, strict=False):
        if not (x.startswith(y) or y.startswith(x)):
            return False
    return True


def first_names_name_compatible(first_a: str, first_b: str, name_tuples: Set[tuple[str, str]]) -> bool:
    """Return current legacy-compatible first-name compatibility.

    This keeps the normalization migration shim in one place: legacy
    `name_tuples` were curated over single-token names, while normalized first
    names can be multi-token. Remove the joined/first-token probes only after
    canonical name-tuple artifacts are regenerated.
    """

    if same_prefix_tokens(first_a, first_b):
        return True

    first_a_parts = first_a.split()
    first_b_parts = first_b.split()
    first_a_joined = "".join(first_a_parts)
    first_b_joined = "".join(first_b_parts)
    first_a_token = first_a_parts[0] if first_a_parts else first_a
    first_b_token = first_b_parts[0] if first_b_parts else first_b
    return (
        (first_a, first_b) in name_tuples
        or (first_a_joined, first_b_joined) in name_tuples
        or (first_a_token, first_b_token) in name_tuples
    )


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
        initials_1 = {token[0] for token in name_1.split() if token}
        initials_2 = {token[0] for token in name_2.split() if token}
        return 1 if not initials_1.isdisjoint(initials_2) else 0

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
    # using nanmin so as to catch the min of counts, but regular max to propagate the nan
    with warnings.catch_warnings():
        # np.max of 2 nans causes annoying warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        counts_array = np.array(counts, dtype=float)
        counts_min_max = list(np.nanmin(counts_array, axis=0)) + list(np.max(counts_array[:, :2], axis=0))

    return counts_min_max
