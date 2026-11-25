"""Bounded caches owned by an offline EPS sweep or linker replay."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from functools import lru_cache
from types import TracebackType
from typing import TYPE_CHECKING, Generic, TypeVar

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.gate_buckets import first_name_bucket_from_token_view

if TYPE_CHECKING:
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


class AttachReuse:
    """Own one offline sweep's setup, with separate signature and name-token limits.

    Limits count retained signature memberships, including native block rows.
    Use a context manager to release native buffers before closing the dataset.
    """

    def __init__(self, *, setup_signature_limit: int = 400_000) -> None:
        self.subblocks: _BoundedCache[_AttachmentSubblocks] = _BoundedCache(
            "subblock signatures", setup_signature_limit
        )
        self.blocks: _BoundedCache[_AttachmentBlockState] = _BoundedCache(
            "featurizer signatures", setup_signature_limit
        )
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
        for cache in (self.blocks, self.subblocks):
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
