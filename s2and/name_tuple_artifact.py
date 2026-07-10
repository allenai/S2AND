"""Strict contract and loader for canonical first-name alias artifacts."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from s2and.consts import _PACKAGE_DATA_DIR, NORMALIZATION_VERSION
from s2and.text import canonical_name_tuple_pair, canonicalize_name_text, same_prefix_tokens

NAME_TUPLE_ARTIFACT_SCHEMA_VERSION = "s2and_name_tuples_v1"
NAME_TUPLE_ARTIFACT_VERSION = 1
NAME_TUPLE_ARTIFACT_SEMANTICS: dict[str, Any] = {
    "encoding": "utf-8",
    "line_format": "name_a,name_b",
    "row_order": "lexicographic_by_fields_unique",
    "directionality": "symmetric_directed_rows",
    "runtime_pair_semantics": "unordered",
    "canonicalizer": "canonicalize_name_text",
    "drop_identity": True,
    "drop_prefix_compatible": True,
}


@dataclass(frozen=True)
class NameTupleArtifactIdentity:
    """Immutable identity fields safe to retain with a loaded artifact."""

    schema_version: str
    artifact_version: int
    normalization_version: str
    data_filename: str
    data_sha256: str
    data_size_bytes: int
    directed_pair_count: int
    unordered_pair_count: int
    source_filename: str
    source_sha256: str
    source_size_bytes: int


@dataclass(frozen=True)
class NameTupleArtifact:
    """Validated immutable alias pairs and their content identity."""

    pairs: frozenset[tuple[str, str]]
    identity: NameTupleArtifactIdentity


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_object(value: Any, *, field: str, metadata_path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Name-tuple metadata {metadata_path} requires object field {field!r}")
    return value


def _require_string(value: Any, *, field: str, metadata_path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Name-tuple metadata {metadata_path} requires nonempty string field {field!r}")
    return value


def _require_nonnegative_int(value: Any, *, field: str, metadata_path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Name-tuple metadata {metadata_path} requires nonnegative integer field {field!r}")
    return value


def _require_sha256(value: Any, *, field: str, metadata_path: Path) -> str:
    digest = _require_string(value, field=field, metadata_path=metadata_path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Name-tuple metadata {metadata_path} requires lowercase SHA-256 field {field!r}")
    return digest


def build_name_tuple_artifact_metadata(
    *,
    source_filename: str,
    source_bytes: bytes,
    data_filename: str,
    data_bytes: bytes,
    directed_pair_count: int,
    unordered_pair_count: int,
    generated_at: str,
    input_pair_count: int,
    dropped_identity: int,
    dropped_prefix_compatible: int,
    dropped_empty: int,
) -> dict[str, Any]:
    """Build canonical metadata for a generated tuple artifact."""

    return {
        "schema_version": NAME_TUPLE_ARTIFACT_SCHEMA_VERSION,
        "artifact_version": NAME_TUPLE_ARTIFACT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "generated_at": generated_at,
        "source": {
            "filename": source_filename,
            "sha256": _sha256_bytes(source_bytes),
            "size_bytes": len(source_bytes),
        },
        "data": {
            "filename": data_filename,
            "sha256": _sha256_bytes(data_bytes),
            "size_bytes": len(data_bytes),
            "directed_pair_count": directed_pair_count,
            "unordered_pair_count": unordered_pair_count,
        },
        "semantics": dict(NAME_TUPLE_ARTIFACT_SEMANTICS),
        "generation_counts": {
            "input_pair_count": input_pair_count,
            "dropped_identity": dropped_identity,
            "dropped_prefix_compatible": dropped_prefix_compatible,
            "dropped_empty": dropped_empty,
        },
    }


def _parse_and_validate_pairs(
    data_bytes: bytes,
    *,
    data_path: Path,
    expected_directed_count: int,
    expected_unordered_count: int,
) -> frozenset[tuple[str, str]]:
    unordered_pairs: set[tuple[str, str]] = set()
    previous: tuple[str, str] | None = None
    directed_pair_count = 0
    for line_number, raw_line in enumerate(io.BytesIO(data_bytes), start=1):
        raw_line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Name-tuple artifact {data_path} is not valid UTF-8 at line {line_number}") from exc
        fields = line.split(",")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"Invalid name tuple at {data_path}:{line_number}: expected two nonempty fields")
        pair = (fields[0], fields[1])
        if previous is not None and pair <= previous:
            raise ValueError(
                f"Invalid name tuple ordering at {data_path}:{line_number}: rows must be unique and sorted by fields"
            )
        previous = pair
        first_a, first_b = pair
        if canonicalize_name_text(first_a) != first_a or canonicalize_name_text(first_b) != first_b:
            raise ValueError(f"Invalid noncanonical name tuple at {data_path}:{line_number}")
        if first_a == first_b:
            raise ValueError(f"Invalid identity name tuple at {data_path}:{line_number}")
        if same_prefix_tokens(first_a, first_b):
            raise ValueError(f"Invalid prefix-compatible name tuple at {data_path}:{line_number}")
        unordered_pairs.add(canonical_name_tuple_pair(first_a, first_b))
        directed_pair_count += 1

    if directed_pair_count != expected_directed_count:
        raise ValueError(
            f"Name-tuple artifact {data_path} directed_pair_count mismatch: "
            f"metadata={expected_directed_count} actual={directed_pair_count}"
        )
    # Sorted uniqueness plus non-identity means an unordered pair has at most
    # two directed rows. Equality proves both directions exist without keeping
    # a second full directed-row representation in memory.
    if directed_pair_count != 2 * len(unordered_pairs):
        raise ValueError(f"Name-tuple artifact {data_path} is missing reverse rows for one or more pairs")
    if len(unordered_pairs) != expected_unordered_count:
        raise ValueError(
            f"Name-tuple artifact {data_path} unordered_pair_count mismatch: "
            f"metadata={expected_unordered_count} actual={len(unordered_pairs)}"
        )
    return frozenset(unordered_pairs)


def load_name_tuple_artifact(path: str | Path) -> NameTupleArtifact:
    """Load a tuple artifact only after validating its adjacent strict sidecar.

    Explicit custom paths use the same contract as the packaged default: the
    sidecar must be named ``<data-path>.meta.json`` and must bind the exact data
    filename, bytes, cardinalities, normalization, and pair semantics.
    """

    data_path = Path(path)
    metadata_path = data_path.with_name(data_path.name + ".meta.json")
    if not data_path.is_file():
        raise FileNotFoundError(f"Name-tuple artifact does not exist: {data_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Name-tuple metadata does not exist: {metadata_path}")

    metadata_bytes = metadata_path.read_bytes()
    data_bytes = data_path.read_bytes()
    if metadata_path.read_bytes() != metadata_bytes:
        raise RuntimeError(f"Name-tuple metadata changed while loading: {metadata_path}")
    try:
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid name-tuple metadata JSON at {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid name-tuple metadata {metadata_path}: expected a JSON object")

    schema_version = _require_string(
        metadata.get("schema_version"), field="schema_version", metadata_path=metadata_path
    )
    if schema_version != NAME_TUPLE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Name-tuple metadata {metadata_path} has unsupported schema_version={schema_version!r}; "
            f"expected {NAME_TUPLE_ARTIFACT_SCHEMA_VERSION!r}"
        )
    artifact_version = _require_nonnegative_int(
        metadata.get("artifact_version"), field="artifact_version", metadata_path=metadata_path
    )
    if artifact_version != NAME_TUPLE_ARTIFACT_VERSION:
        raise ValueError(
            f"Name-tuple metadata {metadata_path} has unsupported artifact_version={artifact_version}; "
            f"expected {NAME_TUPLE_ARTIFACT_VERSION}"
        )
    normalization_version = _require_string(
        metadata.get("normalization_version"), field="normalization_version", metadata_path=metadata_path
    )
    if normalization_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Name-tuple metadata {metadata_path} normalization_version={normalization_version!r}; "
            f"expected {NORMALIZATION_VERSION!r}"
        )
    _require_string(metadata.get("generated_at"), field="generated_at", metadata_path=metadata_path)

    source = _require_object(metadata.get("source"), field="source", metadata_path=metadata_path)
    source_filename = _require_string(source.get("filename"), field="source.filename", metadata_path=metadata_path)
    source_sha256 = _require_sha256(source.get("sha256"), field="source.sha256", metadata_path=metadata_path)
    source_size_bytes = _require_nonnegative_int(
        source.get("size_bytes"), field="source.size_bytes", metadata_path=metadata_path
    )

    data = _require_object(metadata.get("data"), field="data", metadata_path=metadata_path)
    data_filename = _require_string(data.get("filename"), field="data.filename", metadata_path=metadata_path)
    if data_filename != data_path.name:
        raise ValueError(
            f"Name-tuple metadata {metadata_path} binds data.filename={data_filename!r}, expected {data_path.name!r}"
        )
    data_sha256 = _require_sha256(data.get("sha256"), field="data.sha256", metadata_path=metadata_path)
    data_size_bytes = _require_nonnegative_int(
        data.get("size_bytes"), field="data.size_bytes", metadata_path=metadata_path
    )
    if data_size_bytes != len(data_bytes):
        raise ValueError(
            f"Name-tuple artifact {data_path} size mismatch: metadata={data_size_bytes} actual={len(data_bytes)}"
        )
    actual_sha256 = _sha256_bytes(data_bytes)
    if data_sha256 != actual_sha256:
        raise ValueError(
            f"Name-tuple artifact {data_path} SHA-256 mismatch: metadata={data_sha256} actual={actual_sha256}"
        )
    directed_pair_count = _require_nonnegative_int(
        data.get("directed_pair_count"), field="data.directed_pair_count", metadata_path=metadata_path
    )
    unordered_pair_count = _require_nonnegative_int(
        data.get("unordered_pair_count"), field="data.unordered_pair_count", metadata_path=metadata_path
    )

    semantics = _require_object(metadata.get("semantics"), field="semantics", metadata_path=metadata_path)
    if semantics != NAME_TUPLE_ARTIFACT_SEMANTICS:
        raise ValueError(
            f"Name-tuple metadata {metadata_path} has unsupported semantics; "
            f"expected {NAME_TUPLE_ARTIFACT_SEMANTICS!r}"
        )
    generation_counts = _require_object(
        metadata.get("generation_counts"), field="generation_counts", metadata_path=metadata_path
    )
    for field in ("input_pair_count", "dropped_identity", "dropped_prefix_compatible", "dropped_empty"):
        _require_nonnegative_int(
            generation_counts.get(field), field=f"generation_counts.{field}", metadata_path=metadata_path
        )

    pairs = _parse_and_validate_pairs(
        data_bytes,
        data_path=data_path,
        expected_directed_count=directed_pair_count,
        expected_unordered_count=unordered_pair_count,
    )
    identity = NameTupleArtifactIdentity(
        schema_version=schema_version,
        artifact_version=artifact_version,
        normalization_version=normalization_version,
        data_filename=data_filename,
        data_sha256=data_sha256,
        data_size_bytes=data_size_bytes,
        directed_pair_count=directed_pair_count,
        unordered_pair_count=unordered_pair_count,
        source_filename=source_filename,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
    )
    return NameTupleArtifact(pairs=pairs, identity=identity)


@lru_cache(maxsize=1)
def load_packaged_name_tuple_artifact() -> NameTupleArtifact:
    """Validate and retain the immutable packaged canonical artifact once.

    The cached value contains only frozen pairs and a frozen identity. This is
    for installed package data, which is immutable for the process lifetime;
    custom paths deliberately use the uncached loader so mutations are always
    revalidated.
    """

    return load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt")
