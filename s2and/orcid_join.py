"""Guarded joins for clusters that carry the same ORCID."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence, Set
from itertools import combinations
from typing import TypeVar

import numpy as np

from s2and.name_conflict_repair import leading_orcids
from s2and.pair_codes import HARD_CODE, SCALE, certain_close_code, code_tiles
from s2and.text import CanonicalNameParts, canonical_names_conflict

logger = logging.getLogger(__name__)
_Label = TypeVar("_Label", int, str)
_MAX_NAME_CONFLICTS = 16
_CLOSE_CODE = certain_close_code(0.20)


def _supported_orcid(observed: Counter[str], size: int, *, across_subblocks: bool) -> str | None:
    leaders = leading_orcids(observed)
    if len(leaders) != 1:
        return None
    orcid = next(iter(leaders))
    count = observed[orcid]
    if 5 * count < 4 * observed.total():
        return None
    if across_subblocks:
        return orcid if count >= 2 and 20 * count >= size else None
    return orcid if count >= 2 or size <= 2 else None


def _can_join(
    groups: Sequence[_Label],
    members: Mapping[_Label, Sequence[str]],
    forbidden: Mapping[str, Set[str]],
    seed_groups: Mapping[str, object] | None,
) -> bool:
    if seed_groups and len({seed_groups[s] for group in groups for s in members[group] if s in seed_groups}) > 1:
        return False
    if not forbidden:
        return True
    labels = {s: group for group in groups for s in members[group]}
    return not any(
        other in labels and labels[other] != label for s, label in labels.items() for other in forbidden.get(s, ())
    )


def _distances_support_join(
    rows: np.ndarray,
    columns: np.ndarray,
    codes: np.ndarray,
    signatures: Sequence[str],
    anchors: set[int],
    max_average: float,
    name_conflicts: Callable[[Sequence[tuple[str, str]]], list[bool]] | None,
) -> bool:
    """Check an upper bound on mean distance and independent close pairs."""
    total = len(rows) * len(columns)
    left_anchors = np.array([i in anchors for i in rows])
    right_anchors = np.array([i in anchors for i in columns])
    unanchored_pairs = total - int(left_anchors.sum()) * int(right_anchors.sum())
    code_sum = close = 0
    hard_pairs: list[tuple[str, str]] = []
    hard_budget = min(_MAX_NAME_CONFLICTS, max(1, total // 10))
    for start, tile in code_tiles(codes, len(signatures), rows, columns):
        hard = tile == HARD_CODE
        hard_count = int(np.count_nonzero(hard))
        if hard_count:
            if name_conflicts is None or len(hard_pairs) + hard_count > hard_budget:
                return False
            hard_pairs.extend(
                (signatures[rows[start + i]], signatures[columns[j]]) for i, j in zip(*np.nonzero(hard), strict=True)
            )
        # The code plus one is an upper bound. Charge name conflicts the farthest possible distance.
        code_sum += int(tile.sum(dtype=np.uint64)) + tile.size - hard_count
        if code_sum > max_average * SCALE * total:
            return False
        anchored = left_anchors[start : start + len(tile), None] & right_anchors[None, :]
        close += int(np.count_nonzero((tile <= _CLOSE_CODE) & ~anchored))
    if 10 * close < unanchored_pairs:
        return False
    return not hard_pairs or (name_conflicts is not None and all(name_conflicts(hard_pairs)))


def join_orcids_in_block(
    signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    orcids: Mapping[str, str],
    *,
    cut_height: float,
    forbidden: Mapping[str, Set[str]],
    seed_groups: Mapping[str, object] | None,
    name_conflicts: Callable[[Sequence[tuple[str, str]]], list[bool]] | None,
    block_key: str,
) -> np.ndarray | Sequence[int]:
    """Join compatible clusters within one distance matrix before the year-gap split."""
    observed: dict[int, Counter[str]] = defaultdict(Counter)
    anchors: dict[str, set[int]] = defaultdict(set)
    for index, (sid, label) in enumerate(zip(signatures, labels, strict=True)):
        if (orcid := orcids.get(sid)) is not None:
            observed[int(label)][orcid] += 1
            anchors[orcid].add(index)
    if len(observed) < 2 or not any(len(indices) >= 2 for indices in anchors.values()):
        return labels

    positions: dict[int, list[int]] = {label: [] for label in observed}
    for index, label in enumerate(labels):
        if (group := positions.get(int(label))) is not None:
            group.append(index)
    candidates: dict[str, list[int]] = defaultdict(list)
    for label, counts in observed.items():
        orcid = _supported_orcid(counts, len(positions[label]), across_subblocks=False)
        if orcid is not None:
            candidates[orcid].append(label)

    merged = None
    max_average = min(0.60, cut_height + 0.10)
    for orcid, groups in candidates.items():
        if len(groups) < 2:
            continue
        member_ids = {label: [signatures[i] for i in positions[label]] for label in groups}
        if not _can_join(groups, member_ids, forbidden, seed_groups):
            continue
        indices = [np.asarray(positions[label], dtype=np.intp) for label in groups]
        if not all(
            _distances_support_join(left, right, codes, signatures, anchors[orcid], max_average, name_conflicts)
            for left, right in combinations(indices, 2)
        ):
            continue
        if merged is None:
            merged = np.array(labels, copy=True)
        for index in indices[1:]:
            merged[index] = groups[0]
        logger.info("ORCID join in block %s merged %d clusters", block_key, len(groups))
    return labels if merged is None else merged


def _names_conflict_across(
    groups: Sequence[str],
    clusters: Mapping[str, Sequence[str]],
    names: Mapping[str, tuple[str, str]],
    name_tuples: Set[tuple[str, str]],
) -> bool:
    """Return whether two clusters of ``groups`` hold name forms that the hard name rule keeps apart.

    Compares each distinct (first, middle) form once; the clusters share their original block.
    """
    forms = [{names[s] for s in clusters[label]} for label in groups]
    for index, left_forms in enumerate(forms):
        for right_forms in forms[index + 1 :]:
            for first_a, middle_a in left_forms:
                for first_b, middle_b in right_forms:
                    if canonical_names_conflict(
                        CanonicalNameParts(first_a, middle_a, ""),
                        CanonicalNameParts(first_b, middle_b, ""),
                        name_tuples,
                    ):
                        return True
    return False


def join_orcids_across_subblocks(
    clusters: dict[str, list[str]],
    orcids: Mapping[str, str],
    origins: Mapping[str, str],
    subblocks: Mapping[str, str],
    load_name_parts: Callable[[Sequence[str]], Sequence[tuple[str, str]]],
    *,
    forbidden: Mapping[str, Set[str]],
    seed_groups: Mapping[str, object] | None,
    name_tuples: Set[tuple[str, str]] = frozenset(),
) -> dict[str, list[str]]:
    """Join final clusters supported by matching ORCIDs in separate subblocks.

    Read names only for candidates. Group by original block, ORCID and dominant full
    first name. Check the whole group at once to avoid pair scans and transitive joins.
    A group whose name forms break the hard name rule across clusters, such as different
    full middle names or different compound first names, is not joined: linkage would
    never put those mentions in one cluster.
    """
    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    anchor_subblocks: dict[str, set[str]] = {}
    for label, members in clusters.items():
        observed = Counter(orcids[s] for s in members if s in orcids)
        orcid = _supported_orcid(observed, len(members), across_subblocks=True)
        if orcid is None:
            continue
        anchors = [s for s in members if orcids.get(s) == orcid]
        anchor_subblocks[label] = {subblocks[s] for s in anchors}
        candidates[origins[anchors[0]], orcid].append(label)

    candidates = {key: labels for key, labels in candidates.items() if len(labels) >= 2}
    signature_ids = [s for labels in candidates.values() for label in labels for s in clusters[label]]
    names = dict(zip(signature_ids, load_name_parts(signature_ids), strict=True))
    parent: dict[str, str] = {}
    for (origin, _orcid), labels in candidates.items():
        matching: dict[str, list[str]] = defaultdict(list)
        for label in labels:
            members = clusters[label]
            first_names = Counter(first.split(maxsplit=1)[0] for s in members if (first := names[s][0]))
            if first_names:
                first, count = first_names.most_common(1)[0]
                if len(first) > 1 and 5 * count >= 4 * len(members):
                    matching[first].append(label)
        for groups in matching.values():
            if len(groups) < 2:
                continue
            sources = [source for label in groups for source in anchor_subblocks[label]]
            if (
                len(set(sources)) == len(sources)
                and _can_join(groups, clusters, forbidden, seed_groups)
                and not _names_conflict_across(groups, clusters, names, name_tuples)
            ):
                parent.update((label, groups[0]) for label in groups[1:])
                logger.info("ORCID join across subblocks of %s merged %d clusters", origin, len(groups))

    if not parent:
        return clusters
    joined: dict[str, list[str]] = defaultdict(list)
    for label, members in clusters.items():
        joined[parent.get(label, label)].extend(members)
    return dict(joined)
