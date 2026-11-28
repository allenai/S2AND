"""Text processing and compatibility imports."""

import logging as logging
import re as re
import unicodedata as unicodedata
from collections import Counter as Counter
from collections.abc import Set as Set
from functools import lru_cache as lru_cache
from pathlib import Path as Path
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Any as Any
from typing import NamedTuple as NamedTuple

import jellyfish as jellyfish
import numpy as np
import pycld2 as cld2
from numpy import inner as inner
from numpy.linalg import norm as norm
from strsimpy.metric_lcs import MetricLCS as MetricLCS
from text_unidecode import unidecode as unidecode

from s2and.consts import NUMPY_NAN as NUMPY_NAN

from .canonical_names import _ARMENIAN_CONTEXT_SKIPPED_CATEGORIES as _ARMENIAN_CONTEXT_SKIPPED_CATEGORIES
from .canonical_names import _ARMENIAN_LIGATURES as _ARMENIAN_LIGATURES
from .canonical_names import _ARMENIAN_ROMANIZATION as _ARMENIAN_ROMANIZATION
from .canonical_names import _ARMENIAN_SMALL_FROM_CAPITAL as _ARMENIAN_SMALL_FROM_CAPITAL
from .canonical_names import _CANONICAL_NAME_TRANSLATION as _CANONICAL_NAME_TRANSLATION
from .canonical_names import _CJK_NAME_RANGES as _CJK_NAME_RANGES
from .canonical_names import _CJK_SCRIPT_TAGS as _CJK_SCRIPT_TAGS
from .canonical_names import _DOTTED_GIVEN_NAME_ABBREVIATIONS as _DOTTED_GIVEN_NAME_ABBREVIATIONS
from .canonical_names import _GEORGIAN_MTAVRULI_OFFSET as _GEORGIAN_MTAVRULI_OFFSET
from .canonical_names import _GEORGIAN_ROMANIZATION as _GEORGIAN_ROMANIZATION
from .canonical_names import _HAN_NAME_RANGES as _HAN_NAME_RANGES
from .canonical_names import _HANGUL_NAME_RANGES as _HANGUL_NAME_RANGES
from .canonical_names import _JOINED_INITIALS_GIVEN_NAMES as _JOINED_INITIALS_GIVEN_NAMES
from .canonical_names import _KANA_NAME_RANGES as _KANA_NAME_RANGES
from .canonical_names import _MIXED_CASE_INITIALS_VOWELS as _MIXED_CASE_INITIALS_VOWELS
from .canonical_names import _MIXED_CASE_NON_INITIALS as _MIXED_CASE_NON_INITIALS
from .canonical_names import _NAME_INVISIBLE_FORMAT_CHARS as _NAME_INVISIBLE_FORMAT_CHARS
from .canonical_names import _RE_ARMENIAN_LETTER as _RE_ARMENIAN_LETTER
from .canonical_names import _RE_ASCII_LOWERCASE as _RE_ASCII_LOWERCASE
from .canonical_names import _RE_ASCII_UPPERCASE as _RE_ASCII_UPPERCASE
from .canonical_names import _RE_CANONICAL_NON_LETTER as _RE_CANONICAL_NON_LETTER
from .canonical_names import _RE_JOINED_INITIALS as _RE_JOINED_INITIALS
from .canonical_names import _RE_MIXED_CASE_INITIALS as _RE_MIXED_CASE_INITIALS
from .canonical_names import _SLAVIC_SKY_ENDING as _SLAVIC_SKY_ENDING
from .canonical_names import NAME_APOSTROPHE_LIKE_CHARS as NAME_APOSTROPHE_LIKE_CHARS
from .canonical_names import SURNAME_VARIANTS_FILENAME as SURNAME_VARIANTS_FILENAME
from .canonical_names import CanonicalNameParts as CanonicalNameParts
from .canonical_names import _canonical_name_pretranslate as _canonical_name_pretranslate
from .canonical_names import _canonical_name_tokens as _canonical_name_tokens
from .canonical_names import _cjk_script_tag as _cjk_script_tag
from .canonical_names import _is_cjk_name_character as _is_cjk_name_character
from .canonical_names import _joined_initials as _joined_initials
from .canonical_names import _romanize_armenian as _romanize_armenian
from .canonical_names import _widen_kana as _widen_kana
from .canonical_names import canonical_lasts_equivalent as canonical_lasts_equivalent
from .canonical_names import canonical_name_count_keys as canonical_name_count_keys
from .canonical_names import canonical_surname_family as canonical_surname_family
from .canonical_names import canonicalize_name_parts as canonicalize_name_parts
from .canonical_names import canonicalize_name_text as canonicalize_name_text
from .canonical_names import cjk_given_name_for_constraint as cjk_given_name_for_constraint
from .canonical_names import is_cjk_given_name as is_cjk_given_name
from .canonical_names import last_name_allows_joined_initials as last_name_allows_joined_initials
from .canonical_names import surname_variant_map as surname_variant_map
from .name_rules import _MIDDLE_NON_INITIALS as _MIDDLE_NON_INITIALS
from .name_rules import _RE_INITIAL_THEN_NAME as _RE_INITIAL_THEN_NAME
from .name_rules import _RE_SHORT_GIVEN_NAME as _RE_SHORT_GIVEN_NAME
from .name_rules import _ordinary_names_conflict as _ordinary_names_conflict
from .name_rules import _uncovered_first_tail as _uncovered_first_tail
from .name_rules import canonical_name_tuple_pair as canonical_name_tuple_pair
from .name_rules import canonical_names_conflict as canonical_names_conflict
from .name_rules import constraint_middle_tokens as constraint_middle_tokens
from .name_rules import first_names_name_compatible as first_names_name_compatible
from .name_rules import given_name_for_constraint as given_name_for_constraint
from .name_rules import is_joined_reading as is_joined_reading
from .name_rules import joined_initials_for_constraint as joined_initials_for_constraint
from .name_rules import joined_initials_match as joined_initials_match
from .name_rules import middle_names_conflict as middle_names_conflict
from .name_rules import same_prefix_tokens as same_prefix_tokens
from .name_rules import spell_middle_initials as spell_middle_initials
from .name_rules import spelled_middle_initials_for_constraint as spelled_middle_initials_for_constraint
from .name_rules import spelled_reading_forms as spelled_reading_forms
from .normalize import AFFILIATIONS_STOP_WORDS as AFFILIATIONS_STOP_WORDS
from .normalize import DASH_CHARS as DASH_CHARS
from .normalize import DROPPED_AFFIXES as DROPPED_AFFIXES
from .normalize import NAME_DASH_CHARS as NAME_DASH_CHARS
from .normalize import NAME_PREFIXES as NAME_PREFIXES
from .normalize import ORCID_DASH_CLASS as ORCID_DASH_CLASS
from .normalize import ORCID_PATTERN as ORCID_PATTERN
from .normalize import RE_NORMALIZE_TITLE as RE_NORMALIZE_TITLE
from .normalize import RE_NORMALIZE_WHOLE_NAME as RE_NORMALIZE_WHOLE_NAME
from .normalize import STOPWORDS as STOPWORDS
from .normalize import VENUE_STOP_WORDS as VENUE_STOP_WORDS
from .normalize import LanguageDetection as LanguageDetection
from .normalize import _unknown_language_detection as _unknown_language_detection
from .normalize import compute_block as compute_block
from .normalize import detect_language as detect_language
from .normalize import email_prefix_suffix as email_prefix_suffix
from .normalize import get_text_ngrams as get_text_ngrams
from .normalize import get_text_ngrams_words as get_text_ngrams_words
from .normalize import logger as logger
from .normalize import normalize_orcid as normalize_orcid
from .normalize import normalize_orcid_compact as normalize_orcid_compact
from .normalize import normalize_text as normalize_text
from .normalize import normalize_title as normalize_title
from .similarity import TEXT_FUNCTIONS as TEXT_FUNCTIONS
from .similarity import cosine_sim as cosine_sim
from .similarity import counter_jaccard as counter_jaccard
from .similarity import diff as diff
from .similarity import equal as equal
from .similarity import equal_middle as equal_middle
from .similarity import jaccard as jaccard
from .similarity import metric_lcs as metric_lcs
from .similarity import name_counts as name_counts
from .similarity import name_text_features as name_text_features
from .similarity import prefix_dist as prefix_dist

__all__ = [
    "AFFILIATIONS_STOP_WORDS",
    "Any",
    "CanonicalNameParts",
    "Counter",
    "DASH_CHARS",
    "DROPPED_AFFIXES",
    "LanguageDetection",
    "MetricLCS",
    "NAME_APOSTROPHE_LIKE_CHARS",
    "NAME_DASH_CHARS",
    "NAME_PREFIXES",
    "NUMPY_NAN",
    "NamedTuple",
    "ORCID_DASH_CLASS",
    "ORCID_PATTERN",
    "Path",
    "RE_NORMALIZE_TITLE",
    "RE_NORMALIZE_WHOLE_NAME",
    "STOPWORDS",
    "SURNAME_VARIANTS_FILENAME",
    "Set",
    "TEXT_FUNCTIONS",
    "TYPE_CHECKING",
    "VENUE_STOP_WORDS",
    "canonical_lasts_equivalent",
    "canonical_name_count_keys",
    "canonical_name_tuple_pair",
    "canonical_names_conflict",
    "canonical_surname_family",
    "canonicalize_name_parts",
    "canonicalize_name_text",
    "cjk_given_name_for_constraint",
    "cld2",
    "compute_block",
    "constraint_middle_tokens",
    "cosine_sim",
    "counter_jaccard",
    "detect_language",
    "diff",
    "email_prefix_suffix",
    "equal",
    "equal_middle",
    "first_names_name_compatible",
    "get_text_ngrams",
    "get_text_ngrams_words",
    "given_name_for_constraint",
    "inner",
    "is_cjk_given_name",
    "is_joined_reading",
    "jaccard",
    "jellyfish",
    "joined_initials_for_constraint",
    "joined_initials_match",
    "last_name_allows_joined_initials",
    "logger",
    "logging",
    "lru_cache",
    "metric_lcs",
    "middle_names_conflict",
    "name_counts",
    "name_text_features",
    "norm",
    "normalize_orcid",
    "normalize_orcid_compact",
    "normalize_text",
    "normalize_title",
    "np",
    "prefix_dist",
    "re",
    "same_prefix_tokens",
    "spell_middle_initials",
    "spelled_middle_initials_for_constraint",
    "spelled_reading_forms",
    "surname_variant_map",
    "unicodedata",
    "unidecode",
]

_HAN_NAME_RANGES: tuple[tuple[int, int], ...]
_CJK_NAME_RANGES: tuple[tuple[int, int], ...]
_HANGUL_NAME_RANGES: tuple[tuple[int, int], ...]
_KANA_NAME_RANGES: tuple[tuple[int, int], ...]
