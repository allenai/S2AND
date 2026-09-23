"""Request-boundary policy for undirected caller distance overrides."""

import math
from collections.abc import Iterable, Mapping


def normalize_partial_supervision(
    partial_supervision: Mapping[tuple[str, str], int | float] | None,
    *,
    stringify_ids: bool = False,
) -> dict[tuple[str, str], int | float]:
    """Copy finite caller distances into one entry per unordered pair.

    Equal reverse entries are redundant; unequal ones are an invalid request.
    Values retain their numeric type and range, including finite soft distances.

    Args:
        partial_supervision: Caller distances, before any internal label offset.
        stringify_ids: Apply the existing Arrow incremental string-ID policy,
            checking for conflicting distances after string conversion.

    Returns:
        An owned mapping with lexically ordered string-ID pairs.

    Raises:
        ValueError: A distance is nonfinite or an unordered pair has conflicting
            distances, including pairs that coincide after string conversion.
    """
    normalized: dict[tuple[str, str], int | float] = {}
    for (left, right), distance in (partial_supervision or {}).items():
        if not math.isfinite(distance):
            raise ValueError(
                f"partial_supervision requires finite distances: pair={(left, right)!r}, value={distance!r}"
            )
        if stringify_ids:
            left, right = str(left), str(right)
        pair = (left, right) if left <= right else (right, left)
        if pair in normalized and normalized[pair] != distance:
            raise ValueError(
                f"partial_supervision has conflicting distances for pair={pair!r}: "
                f"{normalized[pair]!r} and {distance!r}"
            )
        normalized[pair] = distance
    return normalized


def effective_dataset_disallows(
    disallow_pairs: Iterable[tuple[str, str]],
    partial_supervision: Mapping[tuple[str, str], int | float],
    *,
    use_default_constraints: bool,
) -> set[tuple[str, str]]:
    """Resolve dataset negatives before candidate selection or scoring.

    Caller distances override dataset pairs, and remain active independently
    of the default-constraint switch. Required profile assignments are separate
    structural inputs and are not removed by this policy.
    """
    if not use_default_constraints:
        return set()
    return {
        (left, right)
        for left, right in disallow_pairs
        if (left, right) not in partial_supervision and (right, left) not in partial_supervision
    }
