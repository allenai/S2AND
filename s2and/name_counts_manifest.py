"""Immutable Python views over native-validated name-count manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v3"


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
    files: Mapping[str, ValidatedNameCountsFile]

    @classmethod
    def load(
        cls,
        index_dir: str | os.PathLike[str],
    ) -> ValidatedNameCountsManifest:
        """Open one native-validated manifest."""

        from s2and.name_counts_index import NameCountsIndex

        _index, manifest = NameCountsIndex._open_with_manifest(index_dir)
        return manifest

    @classmethod
    def _from_native(
        cls,
        native: Any,
        *,
        index_dir: str | os.PathLike[str],
    ) -> ValidatedNameCountsManifest:
        """Freeze facts already validated and resolved by the native opener."""

        raw_files = native._validated_manifest_files()
        root = Path(index_dir)
        manifest_path = root / "manifest.json"
        manifest_sha256 = native.name_counts_manifest_sha256
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
            files=MappingProxyType(files),
        )
