"""Shared contract for canonical ORCID first-name prefix-count artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from s2and.text import same_prefix_tokens

ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION = 1
ORCID_PREFIX_MANIFEST_FILENAME = "first_k_letter_counts_from_orcid.manifest.json"
ORCID_PREFIX_METADATA_FILENAME = "first_k_letter_counts_from_orcid.meta.json"
ORCID_PREFIX_DATA_FILENAME = "first_k_letter_counts_from_orcid.json"
ORCID_PREFIX_PAIR_KEY_SEMANTICS = "unordered_lexicographic"
ORCID_PREFIX_GENERATION_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{12}"


def _is_canonical_prefix_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 2 <= len(value) <= 5
        and value.isascii()
        and value.isprintable()
        and value == value.lower()
    )


def validate_orcid_prefix_counts(
    counts: Mapping[str, Mapping[str, int]],
    *,
    context: str,
) -> tuple[int, int]:
    """Validate canonical pair keys and return outer/pair cardinalities."""

    if not isinstance(counts, dict):
        raise TypeError(f"{context} counts must be a plain dict")
    pair_count = 0
    for left, nested in counts.items():
        if not _is_canonical_prefix_token(left):
            raise ValueError(f"{context} outer keys must be lowercase printable ASCII prefixes of length 2 through 5")
        if not isinstance(nested, dict):
            raise TypeError(f"{context} nested values must be plain dictionaries")
        for right, count in nested.items():
            if not _is_canonical_prefix_token(right) or left >= right:
                raise ValueError(f"{context} pairs must be unequal and lexicographically ordered")
            if left[0] != right[0] or same_prefix_tokens(left, right):
                raise ValueError(f"{context} keys violate the generated prefix-pair semantics")
            if type(count) is not int or count <= 0:
                raise ValueError(f"{context} values must be positive integers")
            pair_count += 1
    return len(counts), pair_count
