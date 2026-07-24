"""Immutable Python views over native-validated name-count manifests."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and.consts import NORMALIZATION_VERSION

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v2"
NAME_COUNTS_PROVENANCE_SCHEMA_VERSION = "name_counts_provenance_v3"
NAME_COUNTS_MANIFEST_SHA256_FIELD = "manifest_sha256"
_NAME_COUNTS_PROVENANCE_FIELDS = frozenset(
    {
        "cardinalities",
        "generated_at",
        "generation_id",
        NAME_COUNTS_MANIFEST_SHA256_FIELD,
        "normalization_version",
        "rejected_row_count",
        "schema_version",
        "selected_rows_sha256",
        "source_kind",
        "source_query_sha256",
        "source_row_count",
        "source_snapshot_id",
    }
)
_MAX_U64 = (1 << 64) - 1


def _require_nonempty_string(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires nonempty string {field}")
    return value


def _require_lowercase_sha256(value: Any, *, field: str, context: str) -> str:
    digest = _require_nonempty_string(value, field=field, context=context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} requires lowercase SHA-256 {field}")
    return digest


def _require_u64(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_U64:
        raise ValueError(f"{context} requires unsigned 64-bit integer {field}")
    return value


def _readonly_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_readonly_value(item) for item in value)
    return value


def readonly_name_counts_provenance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze one already-validated provenance payload."""

    frozen = _readonly_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - helper invariant
        raise TypeError("name-count provenance must remain a mapping")
    return frozen


def validated_name_counts_provenance(value: Any, *, context: str) -> dict[str, Any]:
    """Return one validated v3 source-provenance payload."""

    if not isinstance(value, Mapping) or value.get("schema_version") != NAME_COUNTS_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"{context} requires schema_version={NAME_COUNTS_PROVENANCE_SCHEMA_VERSION!r} provenance")
    extra_fields = set(value) - _NAME_COUNTS_PROVENANCE_FIELDS
    if extra_fields:
        raise ValueError(f"{context} provenance has unsupported fields: {sorted(extra_fields)}")
    if value.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError(
            f"{context} normalization_version={value.get('normalization_version')!r}; "
            f"expected {NORMALIZATION_VERSION!r}"
        )
    for field in ("generation_id", "source_snapshot_id", "source_kind"):
        _require_nonempty_string(value.get(field), field=field, context=f"{context} provenance")
    for field in ("source_query_sha256", "selected_rows_sha256"):
        _require_lowercase_sha256(value.get(field), field=field, context=f"{context} provenance")
    if NAME_COUNTS_MANIFEST_SHA256_FIELD in value:
        _require_lowercase_sha256(
            value[NAME_COUNTS_MANIFEST_SHA256_FIELD],
            field=NAME_COUNTS_MANIFEST_SHA256_FIELD,
            context=f"{context} provenance",
        )
    _require_u64(
        value.get("source_row_count"),
        field="source_row_count",
        context=f"{context} provenance",
    )
    return dict(value)


@dataclass(frozen=True, slots=True)
class ValidatedNameCountsFile:
    """One verified material file declared by a name-count manifest."""

    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedNameCountsManifest:
    """Immutable facts retained after one complete name-count validation."""

    index_dir: Path
    manifest_path: Path
    manifest_sha256: str
    normalization_version: str
    source_provenance: Mapping[str, Any]
    files: Mapping[str, ValidatedNameCountsFile]

    @classmethod
    def load(
        cls,
        index_dir: str | os.PathLike[str],
        *,
        context: str,
    ) -> ValidatedNameCountsManifest:
        """Open one native-validated manifest generation."""

        from s2and.name_counts_index import NameCountsIndex

        _index, manifest = NameCountsIndex._open_with_manifest(index_dir, context=context)
        return manifest

    @classmethod
    def _from_native(
        cls,
        native: Any,
        *,
        index_dir: str | os.PathLike[str],
    ) -> ValidatedNameCountsManifest:
        """Freeze facts already validated and resolved by the native opener."""

        _native_root, source_provenance_json, raw_files = native._validated_manifest_facts()
        root = Path(index_dir)
        manifest_path = root / "manifest.json"
        manifest_sha256 = native.name_counts_manifest_sha256
        provenance = json.loads(source_provenance_json)
        provenance[NAME_COUNTS_MANIFEST_SHA256_FIELD] = manifest_sha256
        files = {
            file_key: ValidatedNameCountsFile(
                path=Path(path),
                byte_count=byte_count,
                sha256=sha256,
            )
            for file_key, path, byte_count, sha256 in raw_files
        }

        return cls(
            index_dir=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            normalization_version=native.normalization_version,
            source_provenance=readonly_name_counts_provenance(provenance),
            files=MappingProxyType(files),
        )
