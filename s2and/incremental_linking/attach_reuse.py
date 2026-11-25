"""Bounded caches owned by an offline EPS sweep or linker replay."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from s2and.arrow_inputs import ArrowDataset

if TYPE_CHECKING:
    from s2and.model.bulk import _AttachmentSubblocks
    from s2and.model.incremental import _AttachmentBlockSetup

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


@dataclass(frozen=True)
class ResidualModelDistances:
    """Model scores in the condensed order of the full attachment query group."""

    positions: dict[str, int]
    values: np.ndarray

    def subset_offsets(self, signatures: list[str]) -> np.ndarray | None:
        """Map ordered subset pairs to the cache and refuse changed orientation."""
        positions = np.fromiter(
            (self.positions.get(sid, -1) for sid in signatures), dtype=np.intp, count=len(signatures)
        )
        if positions[0] < 0 or np.any(positions[1:] <= positions[:-1]):
            return None
        offsets = np.empty(len(positions) * (len(positions) - 1) // 2, dtype=np.intp)
        offset = 0
        size = len(self.positions)
        for row, left in enumerate(positions[:-1]):
            right = positions[row + 1 :]
            end = offset + len(right)
            start = left * (2 * size - left - 1) // 2 - left - 1
            offsets[offset:end] = start + right
            offset = end
        return offsets


class AttachReuse:
    """Own one offline sweep's setup and residual model scores.

    Limits count retained signature memberships, including native block rows.
    Run requests sequentially and close the context before closing the dataset.
    """

    def __init__(self, *, setup_signature_limit: int = 400_000, residual_byte_limit: int = 1 << 30) -> None:
        self.subblocks: _BoundedCache[_AttachmentSubblocks] = _BoundedCache(
            "subblock signatures", setup_signature_limit
        )
        self.blocks: _BoundedCache[_AttachmentBlockSetup] = _BoundedCache(
            "featurizer signatures", setup_signature_limit
        )
        self.residuals = _BoundedCache[ResidualModelDistances]("residual bytes", residual_byte_limit)
        # Distance workers can use the same cache within a request.
        self.residual_lock = Lock()

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
        with self.residual_lock:
            for cache in (self.blocks, self.subblocks, self.residuals):
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
