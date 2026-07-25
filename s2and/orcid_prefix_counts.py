"""Canonical ORCID first-name prefix-count artifact contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and.consts import NORMALIZATION_VERSION
from s2and.text import same_prefix_tokens

ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION = "orcid_prefix_counts_v2"
ORCID_PREFIX_MANIFEST_FILENAME = "first_k_letter_counts_from_orcid.manifest.json"
ORCID_PREFIX_DATA_FILENAME = "first_k_letter_counts_from_orcid.json"
ORCID_PREFIX_PAIR_KEY_SEMANTICS = "unordered_lexicographic"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "normalization_version",
        "pair_key_semantics",
        "generated_at",
        "source_kind",
        "source_snapshot_id",
        "source_query_sha256",
        "selected_rows_sha256",
        "name_tuples_sha256",
        "data_sha256",
        "generator_parameters",
        "metrics",
    }
)


@dataclass(frozen=True, slots=True)
class LoadedOrcidPrefixCounts:
    """Immutable facts from one completely validated ORCID count artifact."""

    counts: Mapping[str, Mapping[str, int]]
    data_sha256: str
    manifest_sha256: str
    source_kind: str
    source_snapshot_id: str
    name_tuples_sha256: str


def _is_canonical_prefix_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 2 <= len(value) <= 5
        and value.isascii()
        and value.isprintable()
        and value == value.lower()
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"ORCID prefix-count manifest {field} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"ORCID prefix-count manifest {field} must be a nonempty string")
    return value


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


def load_canonical_orcid_prefix_counts(
    data_dir: str | Path,
) -> LoadedOrcidPrefixCounts:
    """Load and validate the single manifest authority and its data file."""

    root = Path(data_dir)
    manifest_payload, manifest = _read_json_object(root / ORCID_PREFIX_MANIFEST_FILENAME, label="manifest")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("Canonical ORCID prefix-count manifest fields do not match the artifact contract")
    expected = {
        "schema_version": ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "pair_key_semantics": ORCID_PREFIX_PAIR_KEY_SEMANTICS,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Canonical ORCID prefix-count manifest {field} must equal {value!r}")
    source_kind = _require_nonempty_string(manifest.get("source_kind"), field="source_kind")
    source_snapshot_id = _require_nonempty_string(manifest.get("source_snapshot_id"), field="source_snapshot_id")
    _require_nonempty_string(manifest.get("generated_at"), field="generated_at")
    _require_sha256(manifest.get("source_query_sha256"), field="source_query_sha256")
    _require_sha256(manifest.get("selected_rows_sha256"), field="selected_rows_sha256")
    name_tuples_sha256 = _require_sha256(manifest.get("name_tuples_sha256"), field="name_tuples_sha256")
    data_sha256 = _require_sha256(manifest.get("data_sha256"), field="data_sha256")
    if not isinstance(manifest.get("generator_parameters"), dict):
        raise ValueError("Canonical ORCID prefix-count manifest generator_parameters must be an object")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Canonical ORCID prefix-count manifest metrics must be an object")

    data_payload, raw_counts = _read_json_object(root / ORCID_PREFIX_DATA_FILENAME, label="data")
    if hashlib.sha256(data_payload).hexdigest() != data_sha256:
        raise ValueError("Canonical ORCID prefix-count data SHA-256 does not match its manifest")
    outer_count, pair_count = validate_orcid_prefix_counts(raw_counts, context="Canonical ORCID prefix-count")
    for field, actual in (("output_outer_keys", outer_count), ("output_pair_keys", pair_count)):
        if metrics.get(field) != actual:
            raise ValueError(f"Canonical ORCID prefix-count manifest metrics {field} does not match the data")
    immutable_counts = MappingProxyType(
        {prefix: MappingProxyType(dict(nested_counts)) for prefix, nested_counts in raw_counts.items()}
    )
    return LoadedOrcidPrefixCounts(
        counts=immutable_counts,
        data_sha256=data_sha256,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        source_kind=source_kind,
        source_snapshot_id=source_snapshot_id,
        name_tuples_sha256=name_tuples_sha256,
    )
