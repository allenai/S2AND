"""Validated, shared access to the manifest-backed name-count index."""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import (
    NAME_COUNTS_MANIFEST_SHA256_FIELD,
    ValidatedNameCountsManifest,
    readonly_name_counts_provenance,
    validated_name_counts_provenance,
)

_OPEN_CACHE_MAX_PATHS = 4
_OPEN_CACHE_LOCK = threading.Lock()
_OPEN_CACHE: OrderedDict[
    str,
    tuple[bytes, tuple[NameCountsIndex, ValidatedNameCountsManifest]],
] = OrderedDict()


def _lookup_many_deduplicated(
    native: Any,
    first_keys: Sequence[str | None],
    last_keys: Sequence[str | None],
    first_last_keys: Sequence[str | None],
    last_first_initial_keys: Sequence[str | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross the native boundary once per unique key in a bounded batch."""

    columns = (first_keys, last_keys, first_last_keys, last_first_initial_keys)
    lengths = tuple(len(column) for column in columns)
    row_count = lengths[0]
    if any(length != row_count for length in lengths):
        raise ValueError(
            "name-count lookup columns must have equal length: "
            f"first={lengths[0]} last={lengths[1]} first_last={lengths[2]} "
            f"last_first_initial={lengths[3]}"
        )

    unique_columns: list[list[str | None]] = []
    unique_lengths: list[int] = []
    inverse_columns: list[np.ndarray] = []
    for column in columns:
        unique_keys: list[str | None] = []
        key_to_index: dict[str, int] = {}
        inverse = np.full(row_count, -1, dtype=np.int32)
        for row_index, key in enumerate(column):
            if key is None:
                continue
            try:
                unique_index = key_to_index[key]
            except KeyError:
                unique_index = len(unique_keys)
                key_to_index[key] = unique_index
                unique_keys.append(key)
            inverse[row_index] = unique_index
        unique_columns.append(unique_keys)
        unique_lengths.append(len(unique_keys))
        inverse_columns.append(inverse)

    max_unique = max(unique_lengths, default=0)
    if max_unique == 0:
        return (
            np.full(row_count, np.nan, dtype=np.float64),
            np.full(row_count, np.nan, dtype=np.float64),
            np.full(row_count, np.nan, dtype=np.float64),
            np.full(row_count, np.nan, dtype=np.float64),
        )
    for unique_keys in unique_columns:
        unique_keys.extend([None] * (max_unique - len(unique_keys)))
    unique_results = native._lookup_many_unique(*unique_columns)

    outputs: list[np.ndarray] = []
    for unique_result, unique_length, inverse in zip(unique_results, unique_lengths, inverse_columns, strict=True):
        output = np.full(row_count, np.nan, dtype=np.float64)
        present = inverse >= 0
        output[present] = np.asarray(unique_result, dtype=np.float64)[:unique_length][inverse[present]]
        outputs.append(output)
    return outputs[0], outputs[1], outputs[2], outputs[3]


class NameCountsIndex:
    """One immutable native handle over four verified memory-mapped indexes."""

    __slots__ = (
        "__weakref__",
        "_native",
        "manifest_sha256",
        "normalization_version",
        "path",
        "source_provenance",
    )

    def __init__(
        self,
        *,
        native: Any,
        path: str,
        manifest_sha256: str,
        normalization_version: str,
        source_provenance: Mapping[str, Any],
    ) -> None:
        if normalization_version != NORMALIZATION_VERSION:
            raise ValueError(
                f"NameCountsIndex normalization_version={normalization_version!r}; expected {NORMALIZATION_VERSION!r}"
            )
        provenance = validated_name_counts_provenance(
            source_provenance,
            context="NameCountsIndex source_provenance",
        )
        if provenance.get(NAME_COUNTS_MANIFEST_SHA256_FIELD) != manifest_sha256:
            raise ValueError("NameCountsIndex provenance is not bound to its manifest")
        self._native = native
        self.path = path
        self.manifest_sha256 = manifest_sha256
        self.normalization_version = normalization_version
        self.source_provenance = readonly_name_counts_provenance(provenance)

    @classmethod
    def _open_manifest_snapshot(
        cls,
        resolved_path: str,
        manifest_bytes: bytes,
    ) -> tuple[NameCountsIndex, ValidatedNameCountsManifest]:
        """Open one exact native generation and retain its manifest facts."""

        with _OPEN_CACHE_LOCK:
            cached = _OPEN_CACHE.get(resolved_path)
            if cached is not None and cached[0] == manifest_bytes:
                _OPEN_CACHE.move_to_end(resolved_path)
                return cached[1]

        from s2and.runtime import load_s2and_rust_extension

        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        native = load_s2and_rust_extension().NameCountsIndex.open(resolved_path)
        native_manifest_sha256 = native.name_counts_manifest_sha256
        if native_manifest_sha256 != manifest_sha256:
            raise RuntimeError(
                "name-count manifest changed during open; retry the operation: "
                f"expected={manifest_sha256} observed={native_manifest_sha256}"
            )
        manifest = ValidatedNameCountsManifest._from_manifest_bytes(
            resolved_path,
            manifest_bytes,
            context="native-validated name-count index",
            verify_file_digests=False,
        )
        opened = (
            cls(
                native=native,
                path=resolved_path,
                manifest_sha256=manifest_sha256,
                normalization_version=manifest.normalization_version,
                source_provenance=manifest.source_provenance,
            ),
            manifest,
        )
        with _OPEN_CACHE_LOCK:
            cached = _OPEN_CACHE.get(resolved_path)
            if cached is not None and cached[0] == manifest_bytes:
                _OPEN_CACHE.move_to_end(resolved_path)
                return cached[1]
            _OPEN_CACHE[resolved_path] = (manifest_bytes, opened)
            _OPEN_CACHE.move_to_end(resolved_path)
            if len(_OPEN_CACHE) > _OPEN_CACHE_MAX_PATHS:
                _OPEN_CACHE.popitem(last=False)
        return opened

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> NameCountsIndex:
        """Verify and share one complete manifest generation at ``path``."""

        opened, _manifest = cls._open_generation(path)
        return opened

    @classmethod
    def _open_with_manifest(
        cls,
        path: str | os.PathLike[str],
        *,
        context: str,
    ) -> tuple[NameCountsIndex, ValidatedNameCountsManifest]:
        """Open native material and retain facts from the exact same manifest bytes."""

        del context
        opened, manifest = cls._open_generation(path)
        return opened, manifest

    @classmethod
    def _open_generation(
        cls,
        path: str | os.PathLike[str],
    ) -> tuple[NameCountsIndex, ValidatedNameCountsManifest]:
        """Open the exact manifest snapshot currently published at ``path``."""

        resolved_path = str(Path(os.fspath(path)).resolve())
        manifest_path = Path(resolved_path) / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} (missing manifest.json)")
        return cls._open_manifest_snapshot(resolved_path, manifest_path.read_bytes())

    def lookup_many(
        self,
        first_keys: Sequence[str | None],
        last_keys: Sequence[str | None],
        first_last_keys: Sequence[str | None],
        last_first_initial_keys: Sequence[str | None],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resolve four equal-length key columns with current feature semantics."""

        return _lookup_many_deduplicated(
            self._native,
            first_keys,
            last_keys,
            first_last_keys,
            last_first_initial_keys,
        )
