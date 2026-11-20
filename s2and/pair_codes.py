"""One-byte pair codes: the evidence that the post-linkage corrections read.

Linkage may overwrite a block's distance matrix, so the corrections keep a one-byte-per-pair
copy made before linkage. A code is the model distance times ``SCALE``, rounded down, so
[0, 1] maps to 0..254, and ``HARD_CODE`` (255) marks an impossible pair. The Rust kernels use
the same scale. Name-conflict repair and the year-gap split read codes only through this module,
so the condensed-index arithmetic lives in one place.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from s2and.consts import LARGE_DISTANCE
from s2and.runtime import load_s2and_rust_extension

SCALE = 254.999
HARD_CODE = 255
# The code of the farthest possible pair.
FARTHEST_CODE = 254
# Cells per tile when reading the codes between two groups of mentions; bounds the memory.
TILE_CELLS = 1 << 22


def certain_close_code(close_pair_distance: float) -> int:
    """Return the largest code whose whole range lies under ``close_pair_distance``.

    Code ``c`` holds distances in ``[c, c + 1) / SCALE``, so the result is -1 when no code
    lies wholly under the distance.
    """
    return math.floor(close_pair_distance * SCALE - 1)


def quantize_distances(
    condensed: np.ndarray, *, num_threads: int | None = None, require_hard: bool = True
) -> np.ndarray | None:
    """Return a one-byte-per-pair copy of condensed distances.

    Args:
        condensed: Condensed distances in [0, 1], with exactly ``LARGE_DISTANCE``
            for impossible pairs. Another float dtype or a strided view is copied to
            contiguous float64 first, as linkage accepts those too.
        num_threads: Rust pool size; None uses the default pool.
        require_hard: Return ``None`` for a matrix without an impossible pair,
            where no repair can apply.

    Returns:
        ``uint8`` codes, or ``None`` when the matrix is not a condensed numeric matrix,
        holds another value above one, or, with ``require_hard``, has no impossible pair.
        A value above one is a caller's soft "keep apart" distance, which linkage honors
        and a one-byte code cannot carry.
    """
    numeric = np.issubdtype(condensed.dtype, np.floating) or np.issubdtype(condensed.dtype, np.integer)
    if condensed.ndim != 1 or not numeric:
        return None
    condensed = np.ascontiguousarray(condensed, dtype=np.float64)
    return load_s2and_rust_extension().quantize_block_distances(
        condensed, float(LARGE_DISTANCE), num_threads, require_hard
    )


def condensed_index(width: int, left: np.ndarray | int, right: np.ndarray | int) -> np.ndarray:
    """Return the condensed positions of pairs ``(left, right)``, left != right, in either order."""
    low, high = np.minimum(left, right), np.maximum(left, right)
    return width * low - low * (low + 1) // 2 + high - low - 1


def codes_against(codes: np.ndarray, width: int, mention: int, others: np.ndarray) -> np.ndarray:
    """Return the codes between one mention and other mentions, none equal to it."""
    return codes[condensed_index(width, mention, others)]


def code_tiles(
    codes: np.ndarray, width: int, rows: np.ndarray, columns: np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(first row, codes)`` tiles of the codes between two disjoint groups of mentions."""
    step = max(1, TILE_CELLS // max(1, len(columns)))
    for start in range(0, len(rows), step):
        yield start, codes[condensed_index(width, rows[start : start + step, None], columns[None, :])]
