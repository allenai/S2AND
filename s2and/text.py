from typing import List, Union, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from s2and.data import NameCounts

import re
import warnings
import numpy as np
from numpy import inner
from numpy.linalg import norm
from collections import Counter

from text_unidecode import unidecode
import fasttext
import pycld2 as cld2
import jellyfish
from strsimpy.metric_lcs import MetricLCS


from s2and.consts import NUMPY_NAN, FASTTEXT_PATH
from s2and.file_cache import cached_path

# Lazily-loaded fastText model to avoid heavy import-time cost
_FASTTEXT_MODEL = None


def _get_fasttext_model():
    """Return a cached fastText model instance, loading on first use.

    Honors env var `S2AND_SKIP_FASTTEXT` to skip loading (tests can rely on CLD2 only).
    """
    import os

    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL

    if os.environ.get("S2AND_SKIP_FASTTEXT", "").lower() in {"1", "true", "yes"}:
        return None

    # Load once and cache
    try:
        _FASTTEXT_MODEL = fasttext.load_model(cached_path(FASTTEXT_PATH))
    except Exception:
        # If loading fails, degrade gracefully to no fastText
        _FASTTEXT_MODEL = None
    return _FASTTEXT_MODEL


RE_NORMALIZE_WHOLE_NAME = re.compile(r"[^a-zA-Z\s]+")

ORCID_PATTERN = re.compile(r"\d{4}-?\d{4}-?\d{4}-?\d{3}[0-9X]")

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
    except:  # noqa: E722
        predicted_language_2 = "un_2"

    if predicted_language_ft == "un_ft" and predicted_language_2 == "un_2":
        predicted_language = "un"
        is_reliable = False
    elif predicted_language_ft == "un_ft":
        predicted_language = predicted_language_2
        is_reliable = True
    elif predicted_language_2 == "un_2":
        predicted_language = predicted_language_ft
        is_reliable = True
    elif predicted_language_2 != predicted_language_ft:
        predicted_language = "un"
        is_reliable = False
    else:
        predicted_language = predicted_language_2
        is_reliable = True

    # is_english can now be obtained
    is_english = predicted_language == "en"

    return is_reliable, is_english, predicted_language


def normalize_text(text: Optional[str], special_case_apostrophes: bool = False) -> str:
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


def split_first_middle_hyphen_aware(first_raw: Optional[str], middle_raw: Optional[str]) -> Tuple[str, str]:
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

    has_dash_in_first = "-" in first_raw
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
) -> List[float]:
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
    if name_1 is None or name_2 is None or len(name_1) <= 1 or len(name_2) <= 1:
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
    a_norm = norm(a)
    b_norm = norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    else:
        return inner(a, b) / (a_norm * b_norm)


def get_text_ngrams(
    text: Optional[str], use_unigrams: bool = False, use_bigrams: bool = True, stopwords: Optional[Set[str]] = STOPWORDS
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

    if stopwords is not None:
        text = " ".join([word for word in text.split(" ") if word not in stopwords and len(word) > 2])

    unigrams = []  # type: ignore
    if use_unigrams:
        unigrams = filter(lambda x: " " not in x, text)  # type: ignore

    bigrams = []  # type: ignore
    if use_bigrams:
        bigrams = map(  # type: ignore
            lambda x: "".join(x),
            filter(lambda x: " " not in x, zip(text, text[1:])),
        )

    trigrams = map(
        lambda x: "".join(x),
        filter(lambda x: " " not in x, zip(text, text[1:], text[2:])),
    )

    quadgrams = map(
        lambda x: "".join(x),
        filter(lambda x: " " not in x, zip(text, text[1:], text[2:], text[3:])),
    )
    ngrams: Counter = Counter()
    ngrams.update(Counter(unigrams))
    ngrams.update(Counter(bigrams))
    ngrams.update(Counter(trigrams))
    ngrams.update(Counter(quadgrams))
    return ngrams


def get_text_ngrams_words(text: Optional[str], stopwords: Set[str] = STOPWORDS) -> Counter:
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
        zip(text_split, text_split[1:]),
    )
    trigrams = map(
        lambda x: " ".join(x),
        zip(text_split, text_split[1:], text_split[2:]),
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
    for x, y in zip(ta, tb):
        if not (x.startswith(y) or y.startswith(x)):
            return False
    return True


def equal(
    name_1: Optional[str],
    name_2: Optional[str],
    default_val: float = NUMPY_NAN,
) -> Union[int, float]:
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
    if name_1 is None or name_2 is None or len(name_1) == 0 or len(name_2) == 0:
        return default_val

    if name_1 == "-" or name_2 == "-":
        return default_val

    if name_1.lower().strip() == name_2.lower().strip():
        return 1
    else:
        return 0


def equal_middle(
    name_1: Optional[str],
    name_2: Optional[str],
    default_val: float = NUMPY_NAN,
) -> Union[int, float]:
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

    if len(name_1) == 1 or len(name_2) == 1:
        if name_1[0] == name_2[0]:
            return 1

    elif name_1 == name_2:
        return 1

    return 0


def equal_initial(
    name_1: Optional[str],
    name_2: Optional[str],
    default_val: float = NUMPY_NAN,
) -> Union[int, float]:
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
    set_1: Set,
    set_2: Set,
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


def diff(value_1: Optional[float], value_2: Optional[float], default_val: float = NUMPY_NAN) -> float:
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
) -> List[Union[int, float]]:
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
