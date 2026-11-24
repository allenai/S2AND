"""Augmented identities must be confined to their source training population."""

import pytest

from scripts.pairwise.augmented_leakage import (
    require_declared_sources,
    require_training_members,
    source_members,
)


def test_augmented_source_ids_preserve_separators_and_dataset_scope() -> None:
    members = source_members([("qian___id___suffix", "kisti___id___suffix")])
    require_training_members(members, "qian", {"id___suffix"})
    require_declared_sources(members, ("qian", "kisti", "augmented"), context="Test")
    with pytest.raises(ValueError, match="'kisti'"):
        require_training_members(members, "kisti", set())
    with pytest.raises(ValueError, match="undeclared source datasets.*kisti"):
        require_declared_sources(members, ("qian", "augmented"), context="Test")


def test_augmented_source_ids_require_the_canonical_prefix() -> None:
    with pytest.raises(ValueError, match="Invalid augmented source signature ID"):
        source_members([("unprefixed", "qian___valid")])
