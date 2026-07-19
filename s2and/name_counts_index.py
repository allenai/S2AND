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
from types import MappingProxyType
from typing import Any

import numpy as np

from s2and.arrow_inputs import require_name_counts_index_artifact
from s2and.consts import NORMALIZATION_VERSION
from s2and.name_count_binding import NameCountsBinding

_INDEX_CACHE: weakref.WeakValueDictionary[tuple[str, str], NameCountsIndex] = weakref.WeakValueDictionary()
_INDEX_OPENINGS: dict[tuple[str, str], Future[NameCountsIndex]] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def _require_lowercase_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} requires a lowercase hexadecimal SHA-256")
    return value


def validated_name_counts_provenance(value: Any, *, context: str) -> dict[str, Any]:
    """Return one validated v1 source-provenance payload."""

    if not isinstance(value, Mapping) or value.get("schema_version") != "name_counts_provenance_v1":
        raise ValueError(f"{context} requires name_counts_provenance_v1 provenance")
    if value.get("normalization_version") != NORMALIZATION_VERSION:
        raise ValueError(
            f"{context} normalization_version={value.get('normalization_version')!r}; "
            f"expected {NORMALIZATION_VERSION!r}"
        )
    for field in ("generation_id", "source_snapshot_id", "source_kind"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{context} provenance requires {field}")
    # pickle_sha256 remains the v1 source-lineage identity until the next model
    # feature-contract schema. Runtime lookup never opens or unpickles that file.
    for field in ("pickle_sha256", "source_query_sha256", "selected_rows_sha256"):
        _require_lowercase_sha256(value.get(field), context=f"{context} provenance {field}")
    selected_row_count = value.get("selected_row_count")
    if not isinstance(selected_row_count, int) or selected_row_count < 0:
        raise ValueError(f"{context} provenance requires a nonnegative selected_row_count")
    if value.get("source_row_count") != selected_row_count:
        raise ValueError(f"{context} provenance selected_row_count/source_row_count mismatch")
    return dict(value)


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
            require_name_counts_index_artifact(
                resolved_path,
                context="Python name-count index",
                producer_hint="publish a manifest-backed name_counts_index directory",
            )
            if manifest_path.read_bytes() != manifest_bytes:
                raise RuntimeError(f"name-count index manifest changed during validation: {manifest_path}")

            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, Mapping):
                raise TypeError(f"name-count index manifest must contain an object: {manifest_path}")
            normalization_version = manifest.get("normalization_version")
            if normalization_version != NORMALIZATION_VERSION:
                raise ValueError(
                    f"name-count index normalization_version={normalization_version!r}; "
                    f"expected {NORMALIZATION_VERSION!r}: {manifest_path}"
                )
            provenance = validated_name_counts_provenance(
                manifest.get("source_provenance"),
                context=f"{manifest_path} source_provenance",
            )
            from s2and.runtime import load_s2and_rust_extension

            native = load_s2and_rust_extension().NameCountsIndex.open(resolved_path)
            if manifest_path.read_bytes() != manifest_bytes:
                raise RuntimeError(f"name-count index manifest changed while opening: {manifest_path}")
            opened = cls(
                native=native,
                path=resolved_path,
                manifest_sha256=manifest_sha256,
                normalization_version=str(normalization_version),
                source_provenance=provenance,
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
