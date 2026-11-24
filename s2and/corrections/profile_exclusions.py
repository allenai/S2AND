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
    found: dict[int, set[int]] = defaultdict(set)
    for anchor_index, excluded_index in _violating_pairs(block_signatures, labels, profile_exclusions):
        found[int(labels[anchor_index])].add(excluded_index)
    return dict(found)


def _violating_pairs(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    profile_exclusions: Mapping[str, frozenset[str]],
) -> list[tuple[int, int]]:
    """Return ``(anchor index, excluded index)`` for each exclusion whose two mentions share a label."""
    index = {signature_id: i for i, signature_id in enumerate(block_signatures)}
    found: list[tuple[int, int]] = []
    for anchor, excluded in profile_exclusions.items():
        anchor_index = index.get(anchor)
        if anchor_index is None:
            continue
        anchor_label = labels[anchor_index]
        for signature_id in excluded:
            excluded_index = index.get(signature_id)
            if excluded_index is not None and labels[excluded_index] == anchor_label:
                found.append((anchor_index, excluded_index))
    return found


def _set_condensed_pairs(condensed: np.ndarray, n: int, rows: np.ndarray, cols: np.ndarray, value: float) -> None:
    """Set condensed-matrix entries for every (row, col) pair with row != col."""
    i = np.minimum(rows, cols).astype(np.int64)
    j = np.maximum(rows, cols).astype(np.int64)
    keep = i != j
    i, j = i[keep], j[keep]
    condensed[n * i - i * (i + 1) // 2 + j - i - 1] = value


def _required_groups(
    block_signatures: Sequence[str], cluster_seeds_require: Mapping[str, Any] | None
) -> dict[int, Any]:
    """Return ``{matrix index: required group}`` for the block's seeded signatures."""
    if not cluster_seeds_require:
        return {}
    return {
        i: cluster_seeds_require[signature_id]
        for i, signature_id in enumerate(block_signatures)
        if signature_id in cluster_seeds_require
    }


def enforce_profile_exclusions_in_block(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    condensed_distances: np.ndarray,
    profile_exclusions: Mapping[str, frozenset[str]],
    relabel: Callable[[np.ndarray], np.ndarray | Sequence[int]],
    *,
    cluster_seeds_require: Mapping[str, Any] | None = None,
    scratch: np.ndarray | None = None,
    block_key: str = "",
) -> list[int]:
    """Evict excluded mentions from their anchor's cluster and relink the block.

    Args:
        block_signatures: Signature ids in matrix order.
        labels: Labels from the first clustering of this block.
        condensed_distances: Unmodified condensed distances of the first
            clustering. Profile exclusions are not in it as hard negatives.
            This function takes ownership and writes hard negatives into it.
        profile_exclusions: Excluded mentions per anchor.
        relabel: Clusters a condensed matrix with the block's normal model and
            seed handling, and returns labels.
        cluster_seeds_require: Required seed group per signature id, as the
            ``relabel`` callable enforces it. A required group leaves as one:
            an excluded mention takes its group members in the cluster with it.
        scratch: A writable array of the same shape and dtype that the caller
            no longer needs. Each relink clusters a copy of the distances;
            with ``scratch`` that copy reuses this buffer and allocates nothing.
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
    distances = condensed_distances
    groups = _required_groups(block_signatures, cluster_seeds_require)
    for round_number in range(1, MAX_EVICTION_ROUNDS + 1):
        label_array = np.asarray(labels)
        pairs = _violating_pairs(block_signatures, labels, profile_exclusions)
        # The pair itself is always separated. Two mentions that exclude each
        # other are both anchors; without this they would stay together.
        _set_condensed_pairs(
            distances,
            n,
            np.fromiter((a for a, _ in pairs), dtype=np.int64),
            np.fromiter((e for _, e in pairs), dtype=np.int64),
            LARGE_DISTANCE,
        )
        violating_anchors = {a for a, _ in pairs}
        for anchor_label, evicted in violations.items():
            # A violating anchor keeps its claim on the cluster; the direct hard
            # negative above lets linkage divide the cluster between the two.
            evicted = evicted - violating_anchors
            if not evicted:
                continue
            members = np.flatnonzero(label_array == anchor_label)
            # A hard negative inside a required group would contradict the group,
            # and the seed merge would undo the eviction. Move the group as one.
            evicted_groups = {groups[i] for i in evicted if i in groups}
            evicted_indexes = np.fromiter(
                sorted(set(evicted) | {int(i) for i in members if groups.get(int(i)) in evicted_groups}),
                dtype=np.int64,
            )
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
        if scratch is not None and scratch.shape == distances.shape and scratch.dtype == distances.dtype:
            np.copyto(scratch, distances)
            working = scratch
        else:
            working = distances.copy()
        labels = [int(label) for label in relabel(working)]
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
    cluster_seeds_require: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[str]], int]:
    """Move excluded mentions left in their anchor's final cluster to new clusters.

    Steps after block clustering, such as initial-only attachment and the
    reunite step, can move mentions without seeing profile exclusions. This
    final pass keeps the exclusion rule true on every output.

    Args:
        clusters: Final clusters keyed by cluster id.
        profile_exclusions: Excluded mentions per anchor.
        cluster_seeds_require: Required seed group per signature id. An excluded
            mention takes its group members in the cluster with it, into one
            new cluster. A mention without a group becomes a singleton.

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
    groups = cluster_seeds_require or {}
    repaired: dict[str, list[str]] = {}
    new_clusters: list[tuple[str, list[str]]] = []
    moved = 0
    for key, members in clusters.items():
        evicted = to_move.get(key, set())
        evicted_groups = {groups[signature_id] for signature_id in evicted if signature_id in groups}
        leaving = [s for s in members if s in evicted or (s in groups and groups[s] in evicted_groups)]
        leaving_set = set(leaving)
        staying = [signature_id for signature_id in members if signature_id not in leaving_set]
        if staying:
            repaired[key] = staying
        by_group: dict[Any, list[str]] = defaultdict(list)
        for signature_id in leaving:
            # A mention without a group gets a group of its own.
            group = ("group", groups[signature_id]) if signature_id in groups else ("single", signature_id)
            by_group[group].append(signature_id)
        for group_members in by_group.values():
            new_clusters.append((f"{key}_excluded_{sorted(group_members)[0]}", group_members))
            moved += len(group_members)
    # A caller's own key can equal a generated one. Never overwrite a cluster.
    used = set(repaired)
    for base_key, group_members in new_clusters:
        new_key, suffix = base_key, 1
        while new_key in used:
            suffix += 1
            new_key = f"{base_key}_{suffix}"
        used.add(new_key)
        repaired[new_key] = group_members
    logger.warning("Profile exclusions: moved %d mention(s) out of final clusters", moved)
    return repaired, moved
