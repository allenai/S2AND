"""Canonical ORCID first-name prefix counts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and._sha256 import is_lowercase_sha256
from s2and.consts import _PACKAGE_DATA_DIR
from s2and.text import same_prefix_tokens

ORCID_PREFIX_MANIFEST_FILENAME = "first_k_letter_counts_from_orcid.manifest.json"
ORCID_PREFIX_DATA_FILENAME = "first_k_letter_counts_from_orcid.json"
ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD = "orcid_prefix_counts_data_sha256"

DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256 = "b2f811ba01e4fea25eb22d3686a4f085d8fae4c0e410b2fcd9921c26940e43ad"
_PACKAGED_ORCID_PREFIX_COUNTS_DIRS = {
    "fcb4cb7bdf30f07366b946b9fdb1bb0a807bdb00f7dda521575daea6b2b9d8ce": Path(_PACKAGE_DATA_DIR),
    DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256: Path(_PACKAGE_DATA_DIR) / "orcid_prefix_counts_v1_4",
}


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


@cache
def load_packaged_orcid_prefix_counts(data_sha256: str) -> LoadedOrcidPrefixCounts:
    """Load the installed artifact for a model hash.

    Args:
        data_sha256: The SHA-256 of the artifact's count data.

    Returns:
        The validated immutable artifact.

    Raises:
        ValueError: The hash is unknown or the packaged bytes differ.
    """

    data_dir = _PACKAGED_ORCID_PREFIX_COUNTS_DIRS.get(data_sha256)
    if data_dir is None:
        raise ValueError(
            f"Unknown {ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD} {data_sha256!r}. "
            f"Packaged hashes: {', '.join(_PACKAGED_ORCID_PREFIX_COUNTS_DIRS)}"
        )
    artifact = load_canonical_orcid_prefix_counts(data_dir)
    if artifact.data_sha256 != data_sha256:
        raise ValueError(
            f"Packaged ORCID prefix-count data at {data_dir} has SHA-256 {artifact.data_sha256}, expected {data_sha256}"
        )
    return artifact
