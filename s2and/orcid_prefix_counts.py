"""Canonical ORCID first-name prefix counts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and._sha256 import is_lowercase_sha256
from s2and.text import same_prefix_tokens

ORCID_PREFIX_MANIFEST_FILENAME = "first_k_letter_counts_from_orcid.manifest.json"
ORCID_PREFIX_DATA_FILENAME = "first_k_letter_counts_from_orcid.json"


@dataclass(frozen=True, slots=True)
class LoadedOrcidPrefixCounts:
    """Immutable canonical prefix counts and release identities."""

    counts: Mapping[str, Mapping[str, int]]
    data_sha256: str
    name_tuples_sha256: str


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


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Missing canonical ORCID prefix-count {label}: {path}") from error
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Canonical ORCID prefix-count {label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Canonical ORCID prefix-count {label} must be a JSON object: {path}")
    return payload, value


def load_canonical_orcid_prefix_counts(data_dir: str | Path) -> LoadedOrcidPrefixCounts:
    """Load canonical counts from trusted release files."""

    root = Path(data_dir)
    data_payload, raw_counts = _read_json_object(root / ORCID_PREFIX_DATA_FILENAME, label="data")
    validate_orcid_prefix_counts(raw_counts, context="Canonical ORCID prefix-count")
    _, manifest = _read_json_object(root / ORCID_PREFIX_MANIFEST_FILENAME, label="manifest")
    if set(manifest) != {"name_tuples_sha256"} or not is_lowercase_sha256(manifest["name_tuples_sha256"]):
        raise ValueError("Canonical ORCID prefix-count manifest requires one lowercase name_tuples_sha256")
    immutable_counts = MappingProxyType(
        {prefix: MappingProxyType(dict(nested_counts)) for prefix, nested_counts in raw_counts.items()}
    )
    return LoadedOrcidPrefixCounts(
        counts=immutable_counts,
        data_sha256=hashlib.sha256(data_payload).hexdigest(),
        name_tuples_sha256=manifest["name_tuples_sha256"],
    )
