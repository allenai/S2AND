"""Sparse cannot-link-aware restoration of required seed groups after clustering."""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from s2and.consts import LARGE_DISTANCE


def seed_disallow_adjacency(
    disallow_pairs: Iterable[tuple[str, str]],
    partial_supervision: Mapping[tuple[str, str], int | float],
) -> dict[str, set[str]]:
    """Index explicit hard negatives, honoring caller supervision overrides."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in disallow_pairs:
        if (left, right) in partial_supervision or (right, left) in partial_supervision:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    for (left, right), distance in partial_supervision.items():
        if distance == LARGE_DISTANCE:
            adjacency[left].add(right)
            adjacency[right].add(left)
    return dict(adjacency)


def merge_seed_labels(
    signature_ids: Sequence[str],
    labels: np.ndarray,
    seed_groups: Mapping[str, int | str],
    disallow_adjacency: Mapping[str, set[str]],
) -> list[int]:
    """Join compatible seed labels without crossing explicit cannot-link edges.

    Existing clustering labels are indivisible. Forbidden edges move with their
    union-find roots, so later seed unions cannot bypass an earlier prohibition.
    """
    groups: dict[int | str, set[int]] = defaultdict(set)
    for signature_id, label in zip(signature_ids, labels, strict=True):
        if signature_id in seed_groups:
            groups[seed_groups[signature_id]].add(int(label))
    joins = [sorted(group) for group in groups.values() if len(group) > 1]
    if not joins:
        return labels.tolist()

    parent = {int(label): int(label) for label in labels}
    sizes = dict.fromkeys(parent, 1)
    canonical = parent.copy()
    forbidden: dict[int, set[int]] = defaultdict(set)
    if disallow_adjacency:
        signature_labels = dict(zip(signature_ids, map(int, labels), strict=True))
        for signature_id, neighbors in disallow_adjacency.items():
            left = signature_labels.get(signature_id)
            if left is None:
                continue
            for other_id in neighbors:
                right = signature_labels.get(other_id)
                if right is not None and left != right:
                    forbidden[left].add(right)

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    for join in joins:
        # Keep all accepted components as candidates: a blocked merge must not
        # prevent the remaining compatible members of this seed from joining.
        components: list[int] = []
        for label in join:
            root = find(label)
            remaining: list[int] = []
            for candidate in components:
                other = find(candidate)
                if root == other:
                    continue
                if other in forbidden.get(root, ()):
                    remaining.append(other)
                    continue
                if sizes[root] < sizes[other]:
                    root, other = other, root
                parent[other] = root
                sizes[root] += sizes[other]
                canonical[root] = min(canonical[root], canonical[other])
                for neighbor in forbidden.pop(other, ()):
                    forbidden[neighbor].remove(other)
                    forbidden[neighbor].add(root)
                    forbidden[root].add(neighbor)
            components = remaining + [root]
    return [canonical[find(int(label))] for label in labels]


def restore_seed_membership(
    predicted_clusters: Mapping[str, Sequence[str]],
    seed_groups: Mapping[str, int | str],
    disallow_adjacency: Mapping[str, set[str]],
) -> dict[str, int | str]:
    """Carry compatible original seeds into a subblocked incremental pass."""
    if not seed_groups:
        return {
            signature_id: cluster_id
            for cluster_id, signatures in predicted_clusters.items()
            for signature_id in signatures
        }
    cluster_ids = list(predicted_clusters)
    signature_labels = {
        signature_id: label
        for label, signatures in enumerate(predicted_clusters.values())
        for signature_id in signatures
    }
    next_label = len(cluster_ids)
    for signature_id in seed_groups:
        if signature_id not in signature_labels:
            signature_labels[signature_id] = next_label
            next_label += 1
    signature_ids = list(signature_labels)
    merged = merge_seed_labels(
        signature_ids, np.asarray(list(signature_labels.values())), seed_groups, disallow_adjacency
    )
    emitted_ids = set(cluster_ids)
    synthetic_ids: dict[int, str] = {}
    result: dict[str, int | str] = {}
    for signature_id, label in zip(signature_ids, merged, strict=True):
        if label < len(cluster_ids):
            result[signature_id] = cluster_ids[label]
        else:
            if label not in synthetic_ids:
                candidate = str(seed_groups[signature_id])
                while candidate in emitted_ids:
                    candidate = f"seed={candidate}"
                emitted_ids.add(candidate)
                synthetic_ids[label] = candidate
            result[signature_id] = synthetic_ids[label]
    return result
