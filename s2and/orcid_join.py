"""Guarded join of final clusters that carry the same ORCID in separate subblocks.

Within one distance matrix no join is needed: equal ORCIDs get distance 0, so linkage merges
their mentions first.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence, Set

from s2and.name_conflict_repair import leading_orcids
from s2and.text import CanonicalNameParts, canonical_names_conflict

logger = logging.getLogger(__name__)


def _supported_orcid(observed: Counter[str], size: int) -> str | None:
    """Return the cluster's one leading ORCID when at least two mentions and 5% of the cluster carry it."""
    leaders = leading_orcids(observed)
    if len(leaders) != 1:
        return None
    orcid = next(iter(leaders))
    count = observed[orcid]
    if 5 * count < 4 * observed.total():
        return None
    return orcid if count >= 2 and 20 * count >= size else None


def _can_join(
    groups: Sequence[str],
    members: Mapping[str, Sequence[str]],
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
        orcid = _supported_orcid(observed, len(members))
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
