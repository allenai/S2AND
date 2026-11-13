"""Ordered feature metadata defining the persisted Python and Rust model columns."""

from typing import NamedTuple


class FeatureSpec(NamedTuple):
    """Name, selection group, and monotonic constraint for one feature column."""

    name: str
    group: str
    monotone_constraint: int


# Column positions are part of the persisted model contract. Keep this order
# stable and regenerate the Rust metadata when changing this specification.
FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec("first_names_equal", "name_similarity", 1),
    FeatureSpec("middle_initials_overlap", "name_similarity", 1),
    FeatureSpec("middle_names_equal", "name_similarity", 1),
    FeatureSpec("middle_one_missing", "name_similarity", 0),
    FeatureSpec("single_char_first", "name_similarity", 0),
    FeatureSpec("single_char_middle", "name_similarity", 0),
    FeatureSpec("affiliation_overlap", "affiliation_similarity", 0),
    FeatureSpec("email_prefix_equal", "email_similarity", 1),
    FeatureSpec("email_suffix_equal", "email_similarity", 1),
    FeatureSpec("coauthor_overlap", "coauthor_similarity", 1),
    FeatureSpec("coauthor_similarity", "coauthor_similarity", 0),
    FeatureSpec("coauthor_match", "coauthor_similarity", 1),
    FeatureSpec("venue_overlap", "venue_similarity", 0),
    FeatureSpec("year_diff", "year_diff", -1),
    FeatureSpec("title_overlap_words", "title_similarity", 1),
    FeatureSpec("title_overlap_chars", "title_similarity", 1),
    FeatureSpec("position_diff", "misc_features", 0),
    FeatureSpec("abstract_count", "misc_features", 0),
    FeatureSpec("english_count", "misc_features", 0),
    FeatureSpec("same_language", "misc_features", 0),
    FeatureSpec("language_reliability_min", "misc_features", 0),
    FeatureSpec("first_name_count_min", "name_counts", 0),
    FeatureSpec("last_first_name_count_min", "name_counts", -1),
    FeatureSpec("last_name_count_min", "name_counts", -1),
    FeatureSpec("last_first_initial_count_min", "name_counts", -1),
    FeatureSpec("first_name_count_max", "name_counts", 0),
    FeatureSpec("last_first_name_count_max", "name_counts", -1),
    FeatureSpec("specter_cosine_sim", "embedding_similarity", 0),
    FeatureSpec("journal_overlap", "journal_similarity", 0),
    FeatureSpec("levenshtein", "advanced_name_similarity", 0),
    FeatureSpec("prefix", "advanced_name_similarity", 0),
    FeatureSpec("lcs", "advanced_name_similarity", 0),
    FeatureSpec("jaro", "advanced_name_similarity", 0),
)

NAME_DEPENDENT_FEATURE_GROUPS: frozenset[str] = frozenset(
    {"name_similarity", "advanced_name_similarity", "name_counts"}
)
DEFAULT_FEATURE_GROUPS: tuple[str, ...] = tuple(dict.fromkeys(feature.group for feature in FEATURE_SCHEMA))
DEFAULT_NAMELESS_FEATURE_GROUPS: tuple[str, ...] = tuple(
    group for group in DEFAULT_FEATURE_GROUPS if group not in NAME_DEPENDENT_FEATURE_GROUPS
)
