"""Validated, shared access to the manifest-backed name-count index."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import require_name_counts_index_artifact
from s2and.consts import NORMALIZATION_VERSION
from s2and.name_count_binding import NameCountsBinding
from s2and.name_counts_manifest import (
    readonly_name_counts_provenance,
    validated_name_counts_provenance,
)

_INDEX_CACHE: weakref.WeakValueDictionary[tuple[str, str], NameCountsIndex] = weakref.WeakValueDictionary()
_INDEX_OPENINGS: dict[tuple[str, str], Future[NameCountsIndex]] = {}
_INDEX_CACHE_LOCK = threading.Lock()


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
                f"NameCountsIndex normalization_version={normalization_version!r}; "
                f"expected {NORMALIZATION_VERSION!r}"
            )
        provenance = validated_name_counts_provenance(
            source_provenance,
            context="NameCountsIndex source_provenance",
        )
        expected_binding = NameCountsBinding.from_provenance(
            provenance,
            context="NameCountsIndex source_provenance",
        )
        native_normalization = getattr(native, "normalization_version", None)
        if native_normalization != normalization_version:
            raise ValueError(
                "NameCountsIndex native normalization mismatch: "
                f"native={native_normalization!r} expected={normalization_version!r}"
            )
        native_binding = getattr(native, "name_counts_provenance_binding", None)
        expected_binding_tuple = (
            expected_binding.generation_id,
            expected_binding.pickle_sha256,
            expected_binding.source_snapshot_id,
            expected_binding.selected_rows_sha256,
        )
        if native_binding != expected_binding_tuple:
            raise ValueError(
                "NameCountsIndex native provenance mismatch: "
                f"native={native_binding!r} expected={expected_binding_tuple!r}"
            )
        self._native = native
        self.path = path
        self.manifest_sha256 = manifest_sha256
        self.normalization_version = normalization_version
        self.source_provenance = readonly_name_counts_provenance(provenance)

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> NameCountsIndex:
        """Verify and share the exact manifest generation at ``path``."""

        path_text = os.fspath(path)
        resolved_path = str(Path(path_text).resolve())
        manifest_path = Path(resolved_path) / "manifest.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
        except OSError:
            require_name_counts_index_artifact(
                path_text,
                context="Python name-count index",
                producer_hint="publish a manifest-backed name_counts_index directory",
            )
            raise
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        cache_key = (resolved_path, manifest_sha256)

        with _INDEX_CACHE_LOCK:
            cached = _INDEX_CACHE.get(cache_key)
            if cached is not None:
                return cached
            opening = _INDEX_OPENINGS.get(cache_key)
            if opening is None:
                opening = Future()
                _INDEX_OPENINGS[cache_key] = opening
                open_generation = True
            else:
                open_generation = False

        if not open_generation:
            return opening.result()

        try:
            from s2and.runtime import load_s2and_rust_extension

            native = load_s2and_rust_extension().NameCountsIndex.open(resolved_path)
            if manifest_path.read_bytes() != manifest_bytes:
                raise RuntimeError(f"name-count index manifest changed while opening: {manifest_path}")
            # Native open is the material-validation authority. Parse the same
            # validated manifest generation only to retain full provenance.
            manifest = json.loads(manifest_bytes)
            opened = cls(
                native=native,
                path=resolved_path,
                manifest_sha256=manifest_sha256,
                normalization_version=manifest["normalization_version"],
                source_provenance=manifest["source_provenance"],
            )
        except BaseException as exc:
            with _INDEX_CACHE_LOCK:
                _INDEX_OPENINGS.pop(cache_key, None)
            opening.set_exception(exc)
            raise

        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE[cache_key] = opened
            _INDEX_OPENINGS.pop(cache_key, None)
        opening.set_result(opened)
        return opened

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
