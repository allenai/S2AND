"""Authoritative validation for manifest-backed name-count indexes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and.consts import NORMALIZATION_VERSION

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"
NAME_COUNTS_PROVENANCE_SCHEMA_VERSION = "name_counts_provenance_v1"
NAME_COUNTS_INDEX_FILE_KEYS = ("first", "last", "first_last", "last_first_initial")
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
    """Return one validated v1 source-provenance payload."""

    if not isinstance(value, Mapping) or value.get("schema_version") != NAME_COUNTS_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(f"{context} requires schema_version={NAME_COUNTS_PROVENANCE_SCHEMA_VERSION!r} provenance")
    if value.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError(
            f"{context} normalization_version={value.get('normalization_version')!r}; "
            f"expected {NORMALIZATION_VERSION!r}"
        )
    for field in ("generation_id", "source_snapshot_id", "source_kind"):
        _require_nonempty_string(value.get(field), field=field, context=f"{context} provenance")
    # pickle_sha256 remains the v1 source-lineage identity until the next model
    # feature-contract schema. Runtime lookup never opens or unpickles that file.
    for field in ("pickle_sha256", "source_query_sha256", "selected_rows_sha256"):
        _require_lowercase_sha256(value.get(field), field=field, context=f"{context} provenance")
    selected_row_count = _require_u64(
        value.get("selected_row_count"),
        field="selected_row_count",
        context=f"{context} provenance",
    )
    source_row_count = _require_u64(
        value.get("source_row_count"),
        field="source_row_count",
        context=f"{context} provenance",
    )
    if source_row_count != selected_row_count:
        raise ValueError(f"{context} provenance selected_row_count/source_row_count mismatch")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    payload: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        index_dir: str | os.PathLike[str],
        *,
        context: str,
    ) -> ValidatedNameCountsManifest:
        """Read and verify one immutable name-count manifest generation."""

        root = Path(index_dir).resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} (missing manifest.json)")
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{context} has invalid manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{context} manifest must contain a JSON object: {manifest_path}")
        if manifest.get("schema_version") != NAME_COUNTS_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"{context} has unsupported schema_version {manifest.get('schema_version')!r}; "
                f"expected {NAME_COUNTS_INDEX_SCHEMA_VERSION!r}: {manifest_path}"
            )
        normalization_version = manifest.get("normalization_version")
        if normalization_version != NORMALIZATION_VERSION:
            raise ValueError(
                f"{context} has invalid normalization_version {normalization_version!r}; "
                f"expected {NORMALIZATION_VERSION!r}: {manifest_path}"
            )
        provenance = validated_name_counts_provenance(
            manifest.get("source_provenance"),
            context=f"{context} source_provenance",
        )
        if provenance["normalization_version"] != normalization_version:
            raise ValueError(f"{context} source_provenance normalization_version mismatch: {manifest_path}")

        raw_files = manifest.get("files")
        if not isinstance(raw_files, Mapping):
            raise ValueError(f"{context} manifest requires files mapping: {manifest_path}")
        files: dict[str, ValidatedNameCountsFile] = {}
        for file_key in NAME_COUNTS_INDEX_FILE_KEYS:
            entry = raw_files.get(file_key)
            if not isinstance(entry, Mapping):
                raise ValueError(f"{context} manifest requires files.{file_key}: {manifest_path}")
            path_value = _require_nonempty_string(
                entry.get("path"),
                field=f"files.{file_key}.path",
                context=f"{context} manifest",
            )
            if not path_value.strip():
                raise ValueError(f"{context} manifest requires nonempty string files.{file_key}.path")
            declared_path = Path(path_value)
            resolved_path = (declared_path if declared_path.is_absolute() else root / declared_path).resolve()
            try:
                resolved_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"{context} manifest files.{file_key}.path escapes the name_counts_index directory: "
                    f"{resolved_path}"
                ) from exc
            if not resolved_path.is_file():
                raise ValueError(f"{context} manifest files.{file_key}.path target is not a file: {resolved_path}")
            marker_path = resolved_path.parent / ".published"
            if not marker_path.is_file():
                raise ValueError(f"{context} manifest files.{file_key} requires published marker: {marker_path}")
            byte_count = _require_u64(
                entry.get("byte_count"),
                field=f"files.{file_key}.byte_count",
                context=f"{context} manifest",
            )
            expected_sha256 = _require_lowercase_sha256(
                entry.get("sha256"),
                field=f"files.{file_key}.sha256",
                context=f"{context} manifest",
            )
            if resolved_path.stat().st_size != byte_count:
                raise ValueError(f"{context} manifest files.{file_key}.byte_count mismatch: {resolved_path}")
            if _sha256_file(resolved_path) != expected_sha256:
                raise ValueError(f"{context} manifest files.{file_key} SHA-256 mismatch: {resolved_path}")
            files[file_key] = ValidatedNameCountsFile(
                path=resolved_path,
                byte_count=byte_count,
                sha256=expected_sha256,
            )

        return cls(
            index_dir=root,
            manifest_path=manifest_path,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            normalization_version=normalization_version,
            source_provenance=readonly_name_counts_provenance(provenance),
            files=MappingProxyType(files),
            payload=_readonly_value(manifest),
        )
