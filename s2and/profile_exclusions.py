"""Profile exclusions: keep excluded mentions out of a claimed profile's cluster.

A profile exclusion ``(anchor, excluded)`` says that ``excluded`` may not end in
the cluster that holds ``anchor``. The anchor is any mention the user placed on
the profile; its require group, when it has one, is the profile.

A plain cluster-seed disallow vetoes a merge in average linkage. When the
excluded mention already sits inside the bulk of the person's work, that veto
keeps the whole profile away from the bulk. Profile exclusions instead evict
the excluded mention. The block first clusters on model distances. Each
excluded mention found in its anchor's cluster then gets hard negatives against
every other member of that cluster, and the block clusters again. The excluded
mention can no longer join the profile at any height, so it cannot block the
profile's own merges, and it stays free to join any other cluster.

A share cutoff that kept the veto for mostly-excluded clusters was tested on
simulated claims over ORCID-labeled blocks and lost to plain eviction every
time it fired, so it is not part of this module.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from s2and.consts import LARGE_DISTANCE

logger = logging.getLogger("s2and")

# A later round is needed only when the rebuilt tree brings a different excluded
# mention into the profile. An evicted mention holds a hard negative against its
# anchor and cannot return, so the rounds end; simulated claims needed at most two.
# Past the bound, the final pass moves what is left to singletons.
MAX_EVICTION_ROUNDS = 8

ProfileExclusions = dict[str, frozenset[str]]


def normalize_profile_exclusions(pairs: Iterable[tuple[Any, Any]] | None) -> ProfileExclusions:
    """Return ``{anchor: excluded mentions}`` with string signature ids.

    Args:
        pairs: ``(anchor_signature_id, excluded_signature_id)`` pairs.

    Returns:
        The excluded mentions per anchor.

    Raises:
        ValueError: A pair excludes its own anchor or has a null id.
    """
    if pairs is None:
        return {}
    grouped: dict[str, set[str]] = defaultdict(set)
    for anchor, excluded in pairs:
        if anchor is None or excluded is None:
            raise ValueError("profile exclusions cannot contain null signature ids")
        anchor_id, excluded_id = str(anchor), str(excluded)
        if anchor_id == excluded_id:
            raise ValueError(f"profile exclusion excludes its own anchor: {anchor_id!r}")
        grouped[anchor_id].add(excluded_id)
    return {anchor: frozenset(excluded) for anchor, excluded in grouped.items()}


def validate_profile_exclusions(
    profile_exclusions: Mapping[str, frozenset[str]],
    cluster_seeds_require: Mapping[str, Any],
) -> None:
    """Reject exclusions that contradict a placement.

    Args:
        profile_exclusions: Excluded mentions per anchor.
        cluster_seeds_require: Required seed group per signature id.

    Raises:
        ValueError: An excluded mention is in its anchor's require group.
    """
    for anchor, excluded in profile_exclusions.items():
        anchor_group = cluster_seeds_require.get(anchor)
        if anchor_group is None:
            continue
        for signature_id in excluded:
            if cluster_seeds_require.get(signature_id) == anchor_group:
                raise ValueError(
                    f"profile exclusion contradicts a placement: {signature_id!r} is required "
                    f"together with its anchor {anchor!r}"
                )


def _violations(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    profile_exclusions: Mapping[str, frozenset[str]],
) -> dict[int, set[int]]:
    """Return ``{anchor label: indexes of excluded mentions with that label}``."""
    index = {signature_id: i for i, signature_id in enumerate(block_signatures)}
    found: dict[int, set[int]] = defaultdict(set)
    for anchor, excluded in profile_exclusions.items():
        anchor_index = index.get(anchor)
        if anchor_index is None:
            continue
        anchor_label = labels[anchor_index]
        for signature_id in excluded:
            excluded_index = index.get(signature_id)
            if excluded_index is not None and labels[excluded_index] == anchor_label:
                found[int(anchor_label)].add(excluded_index)
    return dict(found)


def _set_condensed_pairs(condensed: np.ndarray, n: int, rows: np.ndarray, cols: np.ndarray, value: float) -> None:
    """Set condensed-matrix entries for every (row, col) pair with row != col."""
    i = np.minimum(rows, cols).astype(np.int64)
    j = np.maximum(rows, cols).astype(np.int64)
    keep = i != j
    i, j = i[keep], j[keep]
    condensed[n * i - i * (i + 1) // 2 + j - i - 1] = value


def enforce_profile_exclusions_in_block(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    condensed_distances: np.ndarray,
    profile_exclusions: Mapping[str, frozenset[str]],
    relabel: Callable[[np.ndarray], np.ndarray | Sequence[int]],
    *,
    block_key: str = "",
) -> list[int]:
    """Evict excluded mentions from their anchor's cluster and relink the block.

    Args:
        block_signatures: Signature ids in matrix order.
        labels: Labels from the first clustering of this block.
        condensed_distances: Unmodified condensed distances of the first
            clustering. Profile exclusions are not in it as hard negatives.
        profile_exclusions: Excluded mentions per anchor.
        relabel: Clusters a condensed matrix with the block's normal model and
            seed handling, and returns labels.
        block_key: Block name for logs.

    Returns:
        Labels after eviction. An excluded mention can still share its anchor's
        cluster only when ``MAX_EVICTION_ROUNDS`` rounds did not settle the
        block; ``evict_profile_exclusion_violations`` then removes it.
    """
    labels = [int(label) for label in labels]
    violations = _violations(block_signatures, labels, profile_exclusions)
    if not violations:
        return labels
    n = len(block_signatures)
    distances = np.array(condensed_distances, dtype=np.float64, copy=True)
    for round_number in range(1, MAX_EVICTION_ROUNDS + 1):
        label_array = np.asarray(labels)
        for anchor_label, evicted in violations.items():
            evicted_indexes = np.fromiter(sorted(evicted), dtype=np.int64)
            members = np.flatnonzero(label_array == anchor_label)
            kept = np.setdiff1d(members, evicted_indexes, assume_unique=True)
            # Hard negatives against the whole remaining profile, not just the
            # anchor: the evicted mention may still join anything else.
            rows = np.repeat(evicted_indexes, len(kept))
            cols = np.tile(kept, len(evicted_indexes))
            _set_condensed_pairs(distances, n, rows, cols, LARGE_DISTANCE)
            logger.info(
                "Profile exclusions: block %s round %d evicts %d mention(s) from a %d-mention cluster",
                block_key,
                round_number,
                len(evicted_indexes),
                len(members),
            )
        labels = [int(label) for label in relabel(distances.copy())]
        violations = _violations(block_signatures, labels, profile_exclusions)
        if not violations:
            return labels
    # Do not fail the request: the caller's final pass still enforces the rule.
    logger.warning(
        "Profile exclusions: block %s still has %d violation(s) after %d rounds; the final pass moves them out",
        block_key,
        sum(len(evicted) for evicted in violations.values()),
        MAX_EVICTION_ROUNDS,
    )
    return labels


def evict_profile_exclusion_violations(
    clusters: Mapping[str, list[str]],
    profile_exclusions: Mapping[str, frozenset[str]],
) -> tuple[dict[str, list[str]], int]:
    """Move excluded mentions left in their anchor's final cluster to singletons.

    Steps after block clustering, such as initial-only attachment and the
    reunite step, can move mentions without seeing profile exclusions. This
    final pass keeps the exclusion rule true on every output.

    Args:
        clusters: Final clusters keyed by cluster id.
        profile_exclusions: Excluded mentions per anchor.

    Returns:
        The repaired clusters and the number of mentions moved.
    """
    cluster_of = {signature_id: key for key, members in clusters.items() for signature_id in members}
    to_move: dict[str, set[str]] = defaultdict(set)
    for anchor, excluded in profile_exclusions.items():
        anchor_cluster = cluster_of.get(anchor)
        if anchor_cluster is None:
            continue
        for signature_id in excluded:
            if cluster_of.get(signature_id) == anchor_cluster:
                to_move[anchor_cluster].add(signature_id)
    if not to_move:
        return dict(clusters), 0
    repaired: dict[str, list[str]] = {}
    moved = 0
    for key, members in clusters.items():
        evicted = to_move.get(key, set())
        repaired[key] = [signature_id for signature_id in members if signature_id not in evicted]
        for signature_id in sorted(evicted):
            repaired[f"{key}_excluded_{signature_id}"] = [signature_id]
            moved += 1
    logger.warning("Profile exclusions: moved %d mention(s) out of final clusters to singletons", moved)
    return repaired, moved
