"""Guarded join of final clusters that carry the same ORCID in separate subblocks.

Within one distance matrix no join is needed: equal ORCIDs get distance 0, so linkage merges
their mentions first.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence, Set

from s2and.corrections.name_conflict_repair import leading_orcids
from s2and.text import CanonicalNameParts, canonical_names_conflict

logger = logging.getLogger("s2and.orcid_join")


def _supported_orcid(observed: Counter[str], size: int) -> str | None:
    """Return the cluster's one leading ORCID when enough of its mentions carry it.

    The ORCID must hold 80% of the cluster's ORCID mentions, at least two mentions, and 5% of
    the cluster.
    """
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


# A name form that holds at least this share of a cluster takes part in the name-rule check;
# rarer forms are strays from attachment or repair. It mirrors the 80% dominance rule.
SUPPORTED_FORM_SHARE = 0.20
# An initial-only cluster has no first name to compare, so its ORCID must cover this share of it.
INITIAL_ONLY_MIN_COVERAGE = 0.20


def _supported_forms(members: Sequence[str], names: Mapping[str, tuple[str, str]]) -> list[tuple[str, str]]:
    counts = Counter(names[s] for s in members)
    return [form for form, count in counts.items() if count >= SUPPORTED_FORM_SHARE * len(members)]


def _names_conflict_across(
    groups: Sequence[str],
    clusters: Mapping[str, Sequence[str]],
    names: Mapping[str, tuple[str, str]],
    name_tuples: Set[tuple[str, str]],
) -> bool:
    """Return whether two clusters of ``groups`` hold supported name forms that the hard name rule keeps apart.

    Compares each cluster's supported (first, middle) forms once; the clusters share their original block.
    """
    forms = [_supported_forms(clusters[label], names) for label in groups]
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


def _first_word(first: str) -> str:
    words = first.split(maxsplit=1)
    return words[0] if words else ""


def _dominant_first_word(members: Sequence[str], names: Mapping[str, tuple[str, str]]) -> str | None:
    """Return the first word of full first names held by at least 80% of the cluster's full-name mentions.

    Returns "" for a cluster with no full first name (initials only) and None when no word dominates.
    """
    words = Counter(word for s in members if len(word := _first_word(names[s][0])) > 1)
    if not words:
        return ""
    word, count = words.most_common(1)[0]
    return word if 5 * count >= 4 * words.total() else None


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
    single_letter_subblocks: Set[str] = frozenset(),
) -> dict[str, list[str]]:
    """Join final clusters supported by matching ORCIDs in separate subblocks.

    Read names only for candidates. Group by original block, ORCID and the dominant full first
    name among each cluster's full-name mentions. A cluster of initials only joins when its ORCID
    covers ``INITIAL_ONLY_MIN_COVERAGE`` of it and one full first name holds the ORCID's other
    clusters. Anchors in the single-letter subblock, whose mentions attach one at a time, do not
    count as a shared subblock. A group whose supported name forms break the hard name rule
    across clusters, such as different full middle names or different compound first names, is
    not joined. Check the whole group at once to avoid pair scans and transitive joins.
    """
    candidates: dict[tuple[str, str], list[str]] = defaultdict(list)
    anchor_subblocks: dict[str, set[str]] = {}
    anchor_counts: dict[str, int] = {}
    for label, members in clusters.items():
        observed = Counter(orcids[s] for s in members if s in orcids)
        orcid = _supported_orcid(observed, len(members))
        if orcid is None:
            continue
        anchors = [s for s in members if orcids.get(s) == orcid]
        anchor_subblocks[label] = {subblocks[s] for s in anchors} - set(single_letter_subblocks)
        anchor_counts[label] = len(anchors)
        candidates[origins[anchors[0]], orcid].append(label)

    candidates = {key: labels for key, labels in candidates.items() if len(labels) >= 2}
    signature_ids = [s for labels in candidates.values() for label in labels for s in clusters[label]]
    names = dict(zip(signature_ids, load_name_parts(signature_ids), strict=True))
    parent: dict[str, str] = {}
    for (origin, _orcid), labels in candidates.items():
        matching: dict[str, list[str]] = defaultdict(list)
        initials_only: list[str] = []
        for label in labels:
            word = _dominant_first_word(clusters[label], names)
            if word:
                matching[word].append(label)
            elif word == "" and anchor_counts[label] >= INITIAL_ONLY_MIN_COVERAGE * len(clusters[label]):
                initials_only.append(label)
        if len(matching) == 1:
            # With one full first name, the initial-only clusters can only belong to it.
            matching[next(iter(matching))].extend(initials_only)
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
                logger.info(
                    "ORCID join across subblocks of %s merged %d clusters of sizes %s",
                    origin,
                    len(groups),
                    [len(clusters[label]) for label in groups],
                )

    if not parent:
        return clusters
    joined: dict[str, list[str]] = defaultdict(list)
    for label, members in clusters.items():
        joined[parent.get(label, label)].extend(members)
    return dict(joined)
