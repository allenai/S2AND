from typing import List, Union, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from s2and.data import NameCounts

import re
import string
from itertools import zip_longest
import warnings
import numpy as np
import pandas as pd
from numpy import inner
from numpy.linalg import norm
from collections import Counter

from text_unidecode import unidecode
import jellyfish
from strsimpy.metric_lcs import MetricLCS

from s2and.consts import NUMPY_NAN

dashes = [
    "\\u002D",
    "\\u058A",
    "\\u05BE",
    "\\u1400",
    "\\u1806",
    "\\u2010",
    "-",
    "\\u2015",
    "\\u2E17",
    "\\u2E1A",
    "\\u2E3A",
    "\\u2E3B",
    "\\u2E40",
    "\\u301C",
    "\\u3030",
    "\\u30A0",
    "\\uFE31",
    "\\uFE32",
    "\\uFE58",
    "\\uFE63",
    "\\uFF0D",
]

RE_DASHES = re.compile(rf"[{''.join(dashes)}]")

RE_APOSTRAPHE_S = re.compile(r"(\w+)'s")
REMOVE_PUNC = str.maketrans(string.punctuation.replace("&", ""), " " * (len(string.punctuation) - 1))
RE_SPACES = re.compile(r"\s+")

WORD_REPLACEMENTS = {
    "the": "",
    "a": "",
    "of": "",
    "&": "and",
}

RE_NORMALIZE_WHOLE_NAME = re.compile(r"[^a-zA-Z0-9\s]+")

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


# TODO: stop-words list must be updated for citations title/abstract related information
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

NAME_PREFIXES = {
    "dr",
    "prof",
    "professor",
    "mr",
    "miss",
    "mrs",
    "ms",
    "mx",
    "sir",
    "phd",
    "md",
    "doctor",
}

NUMERALS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "PAD",  # for padding
}

NUMERAL_PRECEDING_WORDS = {
    "vol",
    "volume",
    "part",
    "issue",
    "number",
    "no",
    "chapter",
    "chap",
    "iss",
    "version",
    "edition",
    "ed",
}


def numeral_similarity(s1, s2):
    """
    This feature finds a;l instances of a location in both strings
    where the content is a numeral (1 to 10 and i through x).
    If it exists, the feature is whether the two
    numerals match. If it does not exist, the feature is NaN.

    This feature is tough because numerals appear for many different reasons.
    """
    s1_split = s1.split()
    s1_len = len(s1_split)
    s2_split = s2.split()
    s2_len = len(s2_split)
    both_numerals = [
        (i, a, b)
        for i, (a, b) in enumerate(zip_longest(s1_split, s2_split, fillvalue="PAD"))
        if a in NUMERALS and b in NUMERALS and i > 0
    ]
    # check two conditions:
    # 1. both numerals is preceeded by NUMERAL_PRECEDING_WORDS
    # or 2. a numeral is the last word in the string for one of the two strings
    both_numerals_filtered = [
        a == b
        for i, a, b in both_numerals
        if (
            a != "PAD"
            and b != "PAD"
            and s1_split[i - 1] in NUMERAL_PRECEDING_WORDS
            and s2_split[i - 1] in NUMERAL_PRECEDING_WORDS
        )
        or (i == s1_len - 1 or i == s2_len - 1)
    ]
    if len(both_numerals_filtered) == 0:
        return NUMPY_NAN  # no location with a numeral in both strings at the same position
    else:
        return all(both_numerals_filtered)  #  whether all locations where both strings have numerals is a match


SPECIAL_PUBLICATION_WORDS = {
    "comment",
    "response",
    "letter",
    "editorial",
    "republished",
    "reply",
    "re",
}


def special_publication_word_similarity(s1, s2):
    s1_set = set(s1.split())
    s2_set = set(s2.split())
    s1_special_overlap = s1_set.intersection(SPECIAL_PUBLICATION_WORDS)
    s2_special_overlap = s2_set.intersection(SPECIAL_PUBLICATION_WORDS)
    if len(s1_special_overlap) == 0 and len(s2_special_overlap) == 0:
        return NUMPY_NAN
    else:
        return s1_special_overlap == s2_special_overlap


def prefix_dist(string_1: str, string_2: str) -> float:
    if string_1 is None or pd.isnull(string_1) or string_2 is None or pd.isnull(string_2):
        return NUMPY_NAN
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
    norm_text = RE_SPACES.sub(" ", norm_text).strip()

    return norm_text


def normalize_venue_name(s: str) -> str:
    if pd.isnull(s) or len(s) == 0:
        return ""

    s = unidecode(s)  # Remove diacritics
    s = s.lower()
    s = RE_DASHES.sub("-", s)  # Normalize dashes
    s = RE_APOSTRAPHE_S.sub(r"\1s", s)
    s = s.translate(REMOVE_PUNC)  # Remove punctuation
    s = RE_SPACES.sub(" ", s).strip()  # Collapse whitespace
    words = s.split(" ")
    s = " ".join(
        WORD_REPLACEMENTS.get(word, word) for word in words
    )  # Replace terms with equivalents, remove stop-words
    s = RE_SPACES.sub(" ", s).strip()  # Collapse whitespace again

    return s


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
    text: Optional[str],
    use_unigrams: bool = False,
    use_bigrams: bool = True,
    stopwords: Optional[Set[str]] = STOPWORDS,
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
        text = " ".join([word for word in text.split(" ") if word not in stopwords])

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
    ngrams = Counter(unigrams) | Counter(bigrams) | Counter(trigrams) | Counter(quadgrams)
    return ngrams


def get_text_ngrams_words(text: Optional[str], stopwords: Optional[Set[str]] = STOPWORDS) -> Counter:
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
    if stopwords is not None:
        text_split = [word for word in text.split() if word not in stopwords]
    else:
        text_split = text.split()
    unigrams = Counter(text_split)
    bigrams = map(
        lambda x: " ".join(x),
        zip(text_split, text_split[1:]),
    )
    trigrams = map(
        lambda x: " ".join(x),
        zip(text_split, text_split[1:], text_split[2:]),
    )
    ngrams = unigrams | Counter(bigrams) | Counter(trigrams)
    return ngrams


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
    counter_1: Counter,
    counter_2: Counter,
    default_val: float = NUMPY_NAN,
    denominator_max: float = np.inf,
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
        counts_min_max = list(np.nanmin(counts, axis=0)) + list(np.max([counts[0][:2], counts[1][:2]], axis=0))

    return counts_min_max
