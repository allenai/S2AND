"""Bounded caches owned by an offline EPS sweep or linker replay."""

from __future__ import annotations

import hashlib
import logging
import pickle
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from sys import getsizeof
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view

if TYPE_CHECKING:
    from s2and.incremental_linking.retrieval import RawArrowPlanBundle
    from s2and.incremental_linking.runtime import FirstPassBatch
    from s2and.model.bulk import _AttachmentSubblocks
    from s2and.model.incremental import _AttachmentBlockState

logger = logging.getLogger("s2and")
T = TypeVar("T")


class _BoundedCache(Generic[T]):
    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = limit
        self.size = 0
        self.entries: dict[Hashable, T] = {}
        self.hits = 0

    def get(self, key: Hashable) -> T | None:
        value = self.entries.get(key)
        self.hits += value is not None
        return value

    def fits(self, size: int) -> bool:
        if self.size + size <= self.limit:
            return True
        logger.info(
            "Attachment reuse %s bypassed: retained=%d requested=%d limit=%d", self.name, self.size, size, self.limit
        )
        return False

    def store(self, key: Hashable, value: T, size: int) -> None:
        self.entries[key] = value
        self.size += size


def attachment_dataset_key(dataset: ArrowDataset) -> tuple[ArrowDataset, str, str | None]:
    """Bind cached data to an open Arrow generation and its name-count index."""
    index = dataset.name_counts_index
    return dataset, dataset.generation_id, None if index is None else index.manifest_sha256


def _estimator_cache_fingerprint(estimator: Any) -> Any:
    if estimator is None:
        return None
    inner = getattr(estimator, "classifier", None)
    if inner is not None and inner is not estimator:
        return (
            "wrapped",
            type(estimator).__module__,
            type(estimator).__qualname__,
            _estimator_cache_fingerprint(inner),
        )
    cache_fingerprint = getattr(estimator, "cache_fingerprint", None)
    if callable(cache_fingerprint):
        return (
            type(estimator).__module__,
            type(estimator).__qualname__,
            tuple(cache_fingerprint()),
        )
    booster = getattr(estimator, "booster_", None)
    model_to_string = getattr(booster, "model_to_string", None)
    if callable(model_to_string):
        model_string = str(model_to_string())
        return (
            type(estimator).__module__,
            type(estimator).__qualname__,
            hashlib.blake2b(model_string.encode("utf-8"), digest_size=16).hexdigest(),
        )
    estimator_type = (type(estimator).__module__, type(estimator).__qualname__)
    try:
        serialized = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, pickle.PickleError, RecursionError, TypeError):
        # An estimator without a complete stable representation cannot safely
        # reuse results produced by an earlier state.
        return (*estimator_type, object())
    return (*estimator_type, hashlib.blake2b(serialized, digest_size=16).hexdigest())


def retained_bytes(value: Any) -> int:
    """Count Python and NumPy payloads, including backing arrays of retained views."""
    seen: set[int] = set()

    def size(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        result = getsizeof(item)
        if isinstance(item, np.ndarray):
            if isinstance(item.base, np.ndarray):
                return result + size(item.base)
            if not item.flags.owndata:
                # Arrow planner arrays can borrow a native buffer whose Python wrapper excludes its bytes.
                result += max(int(item.nbytes), int(getattr(item.base, "nbytes", 0)))
            return result + (sum(size(entry) for entry in item.flat) if item.dtype.hasobject else 0)
        if isinstance(item, Mapping):
            if not isinstance(item, dict):
                # A mapping proxy's shallow size excludes its backing dictionary.
                result += getsizeof(dict(item))
            return result + sum(size(key) + size(entry) for key, entry in item.items())
        if isinstance(item, tuple | list | frozenset):
            return result + sum(size(entry) for entry in item)
        if is_dataclass(item):
            return (
                result + size(vars(item))
                if hasattr(item, "__dict__")
                else result + sum(size(getattr(item, field.name)) for field in fields(item))
            )
        return result

    return size(value)


@dataclass(frozen=True)
class ResidualModelDistances:
    """Model scores in the condensed order of the full attachment query group."""

    positions: dict[str, int]
    values: np.ndarray

    def fill_subset(self, signatures: list[str], output: np.ndarray) -> bool:
        """Fill an ordered subset and refuse pairs with changed orientation."""
        positions = np.fromiter(
            (self.positions.get(sid, -1) for sid in signatures), dtype=np.intp, count=len(signatures)
        )
        if positions[0] < 0 or np.any(positions[1:] <= positions[:-1]):
            return False
        offset = 0
        size = len(self.positions)
        for row, left in enumerate(positions[:-1]):
            right = positions[row + 1 :]
            end = offset + len(right)
            start = left * (2 * size - left - 1) // 2 - left - 1
            output[offset:end] = self.values[start + right]
            offset = end
        return True


class AttachReuse:
    """Own one offline sweep's setup, with separate signature and name-token limits.

    Limits count retained signature memberships, including native block rows.
    Use a context manager to release native buffers before closing the dataset.
    """

    def __init__(
        self,
        *,
        setup_signature_limit: int = 400_000,
        residual_byte_limit: int = 1 << 30,
        snapshot_eps: float | None = None,
        snapshot_byte_limit: int = 512 << 20,
    ) -> None:
        self.subblocks: _BoundedCache[_AttachmentSubblocks] = _BoundedCache(
            "subblock signatures", setup_signature_limit
        )
        self.blocks: _BoundedCache[_AttachmentBlockState] = _BoundedCache(
            "featurizer signatures", setup_signature_limit
        )
        self.residuals = _BoundedCache[ResidualModelDistances]("residual bytes", residual_byte_limit)
        self.snapshot_eps = snapshot_eps
        plan_limit = snapshot_byte_limit // 2 if snapshot_eps is not None else 0
        batch_limit = snapshot_byte_limit - plan_limit if snapshot_eps is not None else 0
        self.snapshot_plans: _BoundedCache[RawArrowPlanBundle] = _BoundedCache("first-pass plan bytes", plan_limit)
        self.snapshots: _BoundedCache[FirstPassBatch] = _BoundedCache("first-pass scoring bytes", batch_limit)
        self._bucket = lru_cache(maxsize=32_768)(first_name_bucket_from_token_view)

    def first_name_buckets(self, tokens: list[str], views: list[str]) -> list[str]:
        """Classify repeated native query tokens without repeating normalization."""
        return [self._bucket(token, view) for token, view in zip(tokens, views, strict=True)]

    def __enter__(self) -> AttachReuse:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        """Log retained sizes and release the caches."""
        for block in self.blocks.entries.values():
            block.release()
        for cache in (self.blocks, self.subblocks, self.residuals, self.snapshot_plans, self.snapshots):
            logger.info(
                "Attachment reuse %s: retained=%d limit=%d entries=%d hits=%d",
                cache.name,
                cache.size,
                cache.limit,
                len(cache.entries),
                cache.hits,
            )
            cache.entries.clear()
            cache.size = 0
        logger.info("Attachment reuse first-name buckets: %s", self._bucket.cache_info())
        self._bucket.cache_clear()
