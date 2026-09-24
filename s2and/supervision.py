"""Request-boundary policy for undirected caller distance overrides."""

import math
from collections.abc import Iterable, Mapping
from itertools import chain

from s2and.consts import LARGE_DISTANCE


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


def validate_incremental_seed_disallows(
    required_assignments: Mapping[str, int | str],
    effective_disallows: Iterable[tuple[str, str]],
    partial_supervision: Mapping[tuple[str, str], int | float],
) -> None:
    """Reject explicit negatives that contradict original seed restoration.

    Args:
        required_assignments: Normalized original signature-to-group assignments,
            before temporary splitting, including seeds outside the query list.
        effective_disallows: Dataset negatives after caller overrides and the
            default-constraint switch have been resolved.
        partial_supervision: Normalized caller distances. Only the explicit
            ``LARGE_DISTANCE`` sentinel is a hard disallow.

    Raises:
        ValueError: An active hard disallow connects members of one original
            group. Report the lexical first group/pair, independent of input order.
    """
    conflict: tuple[str, str, str] | None = None
    hard_negative_pairs = chain(
        effective_disallows,
        (pair for pair, distance in partial_supervision.items() if distance == LARGE_DISTANCE),
    )
    for left, right in hard_negative_pairs:
        group = required_assignments.get(left)
        if group is None or group != required_assignments.get(right):
            continue
        left, right = (left, right) if left <= right else (right, left)
        candidate = (str(group), left, right)
        if conflict is None or candidate < conflict:
            conflict = candidate
    if conflict is not None:
        group_id, left, right = conflict
        raise ValueError(
            f"incremental_seed_disallow_conflict: seed_group={group_id!r}, pair={(left, right)!r}; "
            "original seed members must remain together"
        )
