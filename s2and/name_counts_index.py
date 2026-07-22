"""Validated, shared access to the manifest-backed name-count index."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import CancelledError, Future
from pathlib import Path
from typing import Any

import numpy as np

from s2and.arrow_inputs import require_name_counts_index_artifact
from s2and.consts import NORMALIZATION_VERSION
from s2and.name_count_binding import NameCountsBinding
from s2and.name_counts_manifest import (
    ValidatedNameCountsManifest,
    readonly_name_counts_provenance,
    validated_name_counts_provenance,
)

_INDEX_CACHE: weakref.WeakValueDictionary[tuple[str, str], NameCountsIndex] = weakref.WeakValueDictionary()
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_INFLIGHT: dict[tuple[str, str], Future[NameCountsIndex]] = {}
_LATEST_INDEX_CACHE: OrderedDict[str, NameCountsIndex] = OrderedDict()
_LATEST_INDEX_CACHE_MAX_PATHS = 4
_MANIFEST_OPEN_ATTEMPTS = 3
logger = logging.getLogger(__name__)


def clear_name_counts_index_cache() -> None:
    """Release every completed cached native index handle."""

    with _INDEX_CACHE_LOCK:
        if _INDEX_INFLIGHT:
            raise RuntimeError("cannot clear the name-count index cache while opens are in flight")
        _INDEX_CACHE.clear()
        _LATEST_INDEX_CACHE.clear()


def evict_name_counts_index(path: str | os.PathLike[str]) -> bool:
    """Release completed cached generations for one resolved index path."""

    resolved_path = str(Path(os.fspath(path)).resolve())
    with _INDEX_CACHE_LOCK:
        if any(cache_path == resolved_path for cache_path, _manifest_sha256 in _INDEX_INFLIGHT):
            raise RuntimeError(f"cannot evict name-count index while an open is in flight: {resolved_path}")
        removed = _LATEST_INDEX_CACHE.pop(resolved_path, None) is not None
        generation_keys = [cache_key for cache_key in _INDEX_CACHE if cache_key[0] == resolved_path]
        for cache_key in generation_keys:
            removed = _INDEX_CACHE.pop(cache_key, None) is not None or removed
        return removed


def _cached_index_locked(cache_key: tuple[str, str]) -> NameCountsIndex | None:
    """Return and retain an exact cached generation while holding the cache lock."""

    resolved_path, manifest_sha256 = cache_key
    latest = _LATEST_INDEX_CACHE.get(resolved_path)
    if latest is not None:
        if latest.manifest_sha256 == manifest_sha256:
            _LATEST_INDEX_CACHE.move_to_end(resolved_path)
            return latest
        del _LATEST_INDEX_CACHE[resolved_path]
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        _retain_latest_index_locked(cached)
    return cached


def _retain_latest_index_locked(index: NameCountsIndex) -> None:
    """Strongly retain the newest generation for a bounded set of paths."""

    _LATEST_INDEX_CACHE[index.path] = index
    _LATEST_INDEX_CACHE.move_to_end(index.path)
    while len(_LATEST_INDEX_CACHE) > _LATEST_INDEX_CACHE_MAX_PATHS:
        _LATEST_INDEX_CACHE.popitem(last=False)


def _discard_inflight(cache_key: tuple[str, str], in_flight: Future[NameCountsIndex]) -> None:
    """Remove one exact in-flight open without disturbing a replacement owner."""

    with _INDEX_CACHE_LOCK:
        if _INDEX_INFLIGHT.get(cache_key) is in_flight:
            del _INDEX_INFLIGHT[cache_key]


class _ManifestGenerationChanged(RuntimeError):
    """Signal that publication changed the root manifest during native open."""

    def __init__(self, *, expected_sha256: str, observed_sha256: Any) -> None:
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256
        super().__init__(
            "name-count manifest changed during native open: "
            f"expected_sha256={expected_sha256} observed_sha256={observed_sha256!r}"
        )


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
        native_manifest_sha256 = getattr(native, "manifest_sha256", None)
        if native_manifest_sha256 != manifest_sha256:
            raise ValueError(
                "NameCountsIndex native manifest mismatch: "
                f"native={native_manifest_sha256!r} expected={manifest_sha256!r}"
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
    def _open_manifest_snapshot(
        cls,
        *,
        resolved_path: str,
        manifest_bytes: bytes,
        manifest_sha256: str,
    ) -> NameCountsIndex:
        """Open one exact manifest generation with per-generation single-flight."""

        cache_key = (resolved_path, manifest_sha256)
        while True:
            with _INDEX_CACHE_LOCK:
                cached = _cached_index_locked(cache_key)
                if cached is not None:
                    return cached
                in_flight = _INDEX_INFLIGHT.get(cache_key)
                if in_flight is None:
                    in_flight = Future()
                    _INDEX_INFLIGHT[cache_key] = in_flight
                    owns_open = True
                else:
                    owns_open = False

            if owns_open:
                break
            try:
                return in_flight.result()
            except CancelledError:
                # The owner was interrupted by a BaseException. The canceled
                # future carries no unobserved exception. Remove it here as
                # well as in the owner cleanup so an interrupted owner cannot
                # leave every later caller retrying the same canceled future.
                _discard_inflight(cache_key, in_flight)
                continue

        try:
            from s2and.runtime import load_s2and_rust_extension

            native = load_s2and_rust_extension().NameCountsIndex.open(resolved_path)
            native_manifest_sha256 = getattr(native, "manifest_sha256", None)
            if native_manifest_sha256 != manifest_sha256:
                raise _ManifestGenerationChanged(
                    expected_sha256=manifest_sha256,
                    observed_sha256=native_manifest_sha256,
                )
            # Native open is the material-validation authority. Parse the
            # exact matched manifest bytes only to retain full provenance.
            manifest = json.loads(manifest_bytes)
            opened = cls(
                native=native,
                path=resolved_path,
                manifest_sha256=manifest_sha256,
                normalization_version=manifest["normalization_version"],
                source_provenance=manifest["source_provenance"],
            )

            with _INDEX_CACHE_LOCK:
                cached = _cached_index_locked(cache_key)
                result = opened if cached is None else cached
                if cached is None:
                    _INDEX_CACHE[cache_key] = opened
                    _retain_latest_index_locked(opened)
            in_flight.set_result(result)
        except BaseException as error:
            if not in_flight.done():
                if isinstance(error, Exception):
                    in_flight.set_exception(error)
                else:
                    in_flight.cancel()
            _discard_inflight(cache_key, in_flight)
            raise
        _discard_inflight(cache_key, in_flight)
        return result

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> NameCountsIndex:
        """Verify and share one complete manifest generation at ``path``."""

        opened, _manifest = cls._open_generation(path, manifest_context=None)
        return opened

    @classmethod
    def _open_with_manifest(
        cls,
        path: str | os.PathLike[str],
        *,
        context: str,
    ) -> tuple[NameCountsIndex, ValidatedNameCountsManifest]:
        """Open native material and retain facts from the exact same manifest bytes."""

        opened, manifest = cls._open_generation(path, manifest_context=context)
        if manifest is None:  # pragma: no cover - private helper invariant
            raise RuntimeError("name-count manifest facts were not retained")
        return opened, manifest

    @classmethod
    def _open_generation(
        cls,
        path: str | os.PathLike[str],
        *,
        manifest_context: str | None,
    ) -> tuple[NameCountsIndex, ValidatedNameCountsManifest | None]:
        """Open one publication-stable generation, retrying root replacement races."""

        path_text = os.fspath(path)
        resolved_path = str(Path(path_text).resolve())
        manifest_path = Path(resolved_path) / "manifest.json"
        last_change: _ManifestGenerationChanged | None = None
        for attempt in range(1, _MANIFEST_OPEN_ATTEMPTS + 1):
            try:
                manifest_bytes = manifest_path.read_bytes()
            except OSError as error:
                if manifest_context is not None and isinstance(error, FileNotFoundError):
                    raise FileNotFoundError(f"{manifest_path} (missing manifest.json)") from None
                require_name_counts_index_artifact(
                    path_text,
                    context="Python name-count index",
                    producer_hint="publish a manifest-backed name_counts_index directory",
                )
                raise
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            try:
                manifest = (
                    None
                    if manifest_context is None
                    else ValidatedNameCountsManifest._from_manifest_bytes(
                        resolved_path,
                        manifest_bytes,
                        context=manifest_context,
                        verify_file_digests=False,
                    )
                )
                opened = cls._open_manifest_snapshot(
                    resolved_path=resolved_path,
                    manifest_bytes=manifest_bytes,
                    manifest_sha256=manifest_sha256,
                )
            except _ManifestGenerationChanged as error:
                last_change = error
            except Exception as error:
                # A publisher may replace the root manifest and clean generation
                # A after native parsed A but before all four files are mapped.
                # Retry only when the root identity actually changed; otherwise
                # preserve the native validation failure verbatim.
                try:
                    current_manifest_bytes = manifest_path.read_bytes()
                except OSError:
                    current_manifest_sha256 = None
                else:
                    current_manifest_sha256 = hashlib.sha256(current_manifest_bytes).hexdigest()
                if current_manifest_sha256 == manifest_sha256:
                    raise
                last_change = _ManifestGenerationChanged(
                    expected_sha256=manifest_sha256,
                    observed_sha256=current_manifest_sha256,
                )
                last_change.__cause__ = error
            else:
                return opened, manifest

            if attempt < _MANIFEST_OPEN_ATTEMPTS:
                logger.warning(
                    "name-count manifest changed during open; retrying attempt=%d max_attempts=%d "
                    "path=%s expected_sha256=%s observed_sha256=%r",
                    attempt,
                    _MANIFEST_OPEN_ATTEMPTS,
                    resolved_path,
                    last_change.expected_sha256,
                    last_change.observed_sha256,
                )
                continue
            logger.error(
                "name-count manifest kept changing during open; final_failure attempt=%d "
                "max_attempts=%d path=%s expected_sha256=%s observed_sha256=%r",
                attempt,
                _MANIFEST_OPEN_ATTEMPTS,
                resolved_path,
                last_change.expected_sha256,
                last_change.observed_sha256,
            )
        raise RuntimeError(
            f"name-count manifest changed during all {_MANIFEST_OPEN_ATTEMPTS} open attempts for {resolved_path}"
        ) from last_change

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
