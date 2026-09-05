"""Freeze the feature-column contract used by existing fitted models."""

from s2and.feature_schema import FEATURE_SCHEMA
from s2and.featurizer import (
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    NAME_DEPENDENT_FEATURE_GROUPS,
    FeaturizationInfo,
)
from s2and.text import TEXT_FUNCTIONS

# Captured from FeaturizationInfo before introducing the shared schema. Positions
# and constraints are model semantics, so this fixture intentionally stays literal.
GOLDEN_FEATURES = (
    (0, "first_names_equal", "name_similarity", 1),
    (1, "middle_initials_overlap", "name_similarity", 1),
    (2, "middle_names_equal", "name_similarity", 1),
    (3, "middle_one_missing", "name_similarity", 0),
    (4, "single_char_first", "name_similarity", 0),
    (5, "single_char_middle", "name_similarity", 0),
    (6, "affiliation_overlap", "affiliation_similarity", 0),
    (7, "email_prefix_equal", "email_similarity", 1),
    (8, "email_suffix_equal", "email_similarity", 1),
    (9, "coauthor_overlap", "coauthor_similarity", 1),
    (10, "coauthor_similarity", "coauthor_similarity", 0),
    (11, "coauthor_match", "coauthor_similarity", 1),
    (12, "venue_overlap", "venue_similarity", 0),
    (13, "year_diff", "year_diff", -1),
    (14, "title_overlap_words", "title_similarity", 1),
    (15, "title_overlap_chars", "title_similarity", 1),
    (16, "position_diff", "misc_features", 0),
    (17, "abstract_count", "misc_features", 0),
    (18, "english_count", "misc_features", 0),
    (19, "same_language", "misc_features", 0),
    (20, "language_reliability_min", "misc_features", 0),
    (21, "first_name_count_min", "name_counts", 0),
    (22, "last_first_name_count_min", "name_counts", -1),
    (23, "last_name_count_min", "name_counts", -1),
    (24, "last_first_initial_count_min", "name_counts", -1),
    (25, "first_name_count_max", "name_counts", 0),
    (26, "last_first_name_count_max", "name_counts", -1),
    (27, "specter_cosine_sim", "embedding_similarity", 0),
    (28, "journal_overlap", "journal_similarity", 0),
    (29, "levenshtein", "advanced_name_similarity", 0),
    (30, "prefix", "advanced_name_similarity", 0),
    (31, "lcs", "advanced_name_similarity", 0),
    (32, "jaro", "advanced_name_similarity", 0),
)
GOLDEN_GROUPS = tuple(dict.fromkeys(row[2] for row in GOLDEN_FEATURES))
GOLDEN_NAME_GROUPS = frozenset({"name_similarity", "name_counts", "advanced_name_similarity"})


def test_schema_preserves_frozen_columns():
    assert (
        tuple(
            (index, feature.name, feature.group, feature.monotone_constraint)
            for index, feature in enumerate(FEATURE_SCHEMA)
        )
        == GOLDEN_FEATURES
    )


def _assert_selected_contract(groups: list[str]) -> None:
    info = FeaturizationInfo(groups)
    selected = [row for row in GOLDEN_FEATURES if row[2] in groups]
    assert info.features_to_use == groups
    assert info.number_of_features == 33
    assert info.selected_feature_indices() == [row[0] for row in selected]
    assert info.get_feature_names() == [row[1] for row in selected]
    assert info.lightgbm_monotone_constraints == ",".join(str(row[3]) for row in selected)
    assert info.nameless_lightgbm_monotone_constraints == ",".join(
        str(row[3]) for row in selected if row[2] not in GOLDEN_NAME_GROUPS
    )


def test_default_metadata_preserves_fitted_model_contract():
    info = FeaturizationInfo()
    assert tuple(info.features_to_use) == GOLDEN_GROUPS
    assert DEFAULT_FEATURE_GROUPS == GOLDEN_GROUPS
    assert NAME_DEPENDENT_FEATURE_GROUPS == GOLDEN_NAME_GROUPS
    assert DEFAULT_NAMELESS_FEATURE_GROUPS == tuple(group for group in GOLDEN_GROUPS if group not in GOLDEN_NAME_GROUPS)
    assert info.feature_group_to_index == {
        group: [row[0] for row in GOLDEN_FEATURES if row[2] == group] for group in GOLDEN_GROUPS
    }
    _assert_selected_contract(list(GOLDEN_GROUPS))


def test_selected_groups_preserve_canonical_metadata():
    # Selection is independent per group; cover each membership plus ordering,
    # duplication, empty selection, and the production nameless combination.
    for groups in ([], *([group] for group in GOLDEN_GROUPS), list(DEFAULT_NAMELESS_FEATURE_GROUPS)):
        _assert_selected_contract(groups)
    _assert_selected_contract(list(reversed(GOLDEN_GROUPS)) + list(GOLDEN_GROUPS))


def test_advanced_feature_calculation_order_matches_fitted_model_contract():
    assert tuple(name for _function, name in TEXT_FUNCTIONS) == tuple(
        row[1] for row in GOLDEN_FEATURES if row[2] == "advanced_name_similarity"
    )
