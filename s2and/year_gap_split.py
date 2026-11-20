"""Year-gap split: separate the parts of a cluster that an empty stretch of years divides.

The pair model sees the years between two papers capped at ``YEAR_DIFF_CAP``, so it cannot
tell 50 years apart from 80. Topic overlap can then pull a 1928 paper into the profile of an
author who started publishing in 2000. A real career leaves papers all along, so a cluster
whose dated mentions leave an empty stretch longer than the cap is cut there.

Two parts across a cut stay together when the model is certain of some pair between them
(a code under the close-pair distance), or they share an ORCID or a required seed group.
A reprint or a re-dated record usually has such a pair: the same title and coauthors. When
the surname block has under 11 mentions (``SMALL_SURNAME_BLOCK``), two
parts that share the first word of a full first name also stay together: there the surname is
rare, so a shared first name usually means one person. Judges found 9 of 28 decided such
cuts right in small blocks, against 37 of 38 in large blocks. The caller decides which
signatures sit in a small surname block (``SMALL_SURNAME_BLOCK``); a claimed profile or a
subblock of a large block does not.

The step only splits, so no other cluster changes. Undated mentions that share an ORCID, a
seed group or a bridging first name move as one group. A group follows the part that shares
one with it, else the part with the lowest average code. A group that shares one with two
parts keeps them together.

The split runs after name-conflict repair (``s2and.block_corrections``) on the same pair codes,
ORCIDs and seed groups, so the repair cannot undo it.

``PLACEHOLDER_YEARS`` count as undated. The corpus stamps 1601 and 1800 on modern papers
whose year it does not know: 1601 has 1,002 papers against 18 in 1600 and a handful after,
and 1800 twice the papers of 1801. Judges found 10 of 50 decided gaps through them right.
Many 1800 papers are real, so a loose group with such a stamp first follows the one part
whose dated years span the stamp.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from s2and.consts import YEAR_DIFF_CAP
from s2and.pair_codes import code_tiles

logger = logging.getLogger("s2and")

# A surname block under this many mentions is small: there a shared first name bridges a gap.
SMALL_SURNAME_BLOCK = 11

# Years the corpus gives papers whose year it does not know; the split treats them as undated.
PLACEHOLDER_YEARS = frozenset({1601, 1800})


def split_years(years: Mapping[str, int]) -> dict[str, int]:
    """Return the years that date a mention for the split: above zero and not a placeholder."""
    return {signature_id: year for signature_id, year in years.items() if year > 0 and year not in PLACEHOLDER_YEARS}


def _has_close_pair(codes: np.ndarray, block_width: int, a: np.ndarray, b: np.ndarray, close_code: int) -> bool:
    """Return whether some pair between two groups has a code of at most ``close_code``."""
    if len(a) > len(b):
        a, b = b, a
    return any(bool((tile <= close_code).any()) for _, tile in code_tiles(codes, block_width, a, b))


def _mean_cross_code(codes: np.ndarray, block_width: int, rows: np.ndarray, cols: np.ndarray) -> float:
    """Return the mean code between two disjoint groups."""
    total = sum(int(tile.sum(dtype=np.int64)) for _, tile in code_tiles(codes, block_width, rows, cols))
    return total / (len(rows) * len(cols))


def _shared(values: Mapping[str, object], a: Sequence[str], b: Sequence[str]) -> bool:
    """Return whether two groups of signatures share a value of ``values``."""
    if not values:
        return False
    seen = {values[s] for s in a if s in values}
    return any(values[s] in seen for s in b if s in values)


def _parts(
    members: Sequence[int],
    block_signatures: Sequence[str],
    years: Mapping[str, int],
    codes: np.ndarray,
    close_code: int,
    orcids: Mapping[str, str],
    seed_groups: Mapping[str, object],
    first_names: Mapping[str, str],
    stamps: Mapping[str, int],
) -> list[list[int]]:
    """Return the parts of one cluster, as lists of block positions; one part when it stays whole.

    ``years`` holds dating years only; ``stamps`` holds the placeholder years of undated mentions.
    """
    dated = sorted((years[block_signatures[m]], m) for m in members if block_signatures[m] in years)
    segments: list[list[int]] = [[dated[0][1]]] if dated else []
    for (previous, _), (year, mention) in zip(dated, dated[1:], strict=False):
        if year - previous > YEAR_DIFF_CAP:
            segments.append([])
        segments[-1].append(mention)
    if len(segments) < 2:
        return [list(members)]
    width = len(block_signatures)
    ids = [[block_signatures[m] for m in segment] for segment in segments]
    arrays = [np.asarray(segment, dtype=np.int64) for segment in segments]
    parent = list(range(len(segments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if find(i) == find(j):
                continue
            if (
                _shared(orcids, ids[i], ids[j])
                or _shared(seed_groups, ids[i], ids[j])
                or _shared(first_names, ids[i], ids[j])
                or _has_close_pair(codes, width, arrays[i], arrays[j], close_code)
            ):
                parent[find(i)] = find(j)
    grouped: dict[int, list[int]] = {}
    for i, segment in enumerate(segments):
        grouped.setdefault(find(i), []).extend(segment)
    if len(grouped) == 1:
        return [list(members)]
    return _place_undated(
        list(grouped.values()), members, block_signatures, years, codes, orcids, seed_groups, first_names, stamps
    )


def _place_undated(
    parts: list[list[int]],
    members: Sequence[int],
    block_signatures: Sequence[str],
    years: Mapping[str, int],
    codes: np.ndarray,
    orcids: Mapping[str, str],
    seed_groups: Mapping[str, object],
    first_names: Mapping[str, str],
    stamps: Mapping[str, int],
) -> list[list[int]]:
    """Add a cluster's undated mentions to its dated parts; one part when they tie all parts together.

    Undated mentions that share an ORCID, a seed group or a bridging first name move as one
    group, so none of these is split. A group that shares one with two parts keeps those parts
    together. A group that shares none and carries placeholder stamps goes to the one part whose
    dated years span every stamp. Any other group goes to the part with the lowest average code.
    """
    undated = [m for m in members if block_signatures[m] not in years]
    # Nodes: parts first, then undated mentions. Two nodes that share a value are joined.
    parent = list(range(len(parts) + len(undated)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_node: dict[tuple[str, object], int] = {}
    node_mentions = [*parts, *([m] for m in undated)]
    for node, mentions in enumerate(node_mentions):
        for m in mentions:
            signature_id = block_signatures[m]
            keys = []
            if signature_id in orcids:
                keys.append(("orcid", orcids[signature_id]))
            if signature_id in seed_groups:
                keys.append(("seed", seed_groups[signature_id]))
            if signature_id in first_names:
                keys.append(("first", first_names[signature_id]))
            for key in keys:
                other = first_node.setdefault(key, node)
                if find(other) != find(node):
                    parent[find(node)] = find(other)
    # Parts sharing no value stay apart: the segment step already joined any that did.
    merged: dict[int, list[int]] = {}
    loose: dict[int, list[int]] = {}
    for node, mentions in enumerate(node_mentions):
        target = merged if node < len(parts) or find(node) in merged else loose
        target.setdefault(find(node), []).extend(mentions)
    if len(merged) == 1:
        return [list(members)]
    new_parts = list(merged.values())
    width = len(block_signatures)
    part_arrays = [np.asarray(part, dtype=np.int64) for part in new_parts]
    spans = []
    for part in new_parts:
        part_years = [years[block_signatures[m]] for m in part if block_signatures[m] in years]
        spans.append((min(part_years), max(part_years)))
    for group in loose.values():
        group_stamps = [stamps[block_signatures[m]] for m in group if block_signatures[m] in stamps]
        # A real 1800 paper inside a 1786-1817 career belongs there; a placeholder spans no part.
        holders = [
            index
            for index, (low, high) in enumerate(spans)
            if group_stamps and all(low <= stamp <= high for stamp in group_stamps)
        ]
        if len(holders) == 1:
            new_parts[holders[0]].extend(group)
            continue
        # Impossible pairs count as code 255, so a name conflict pushes the group away.
        rows = np.asarray(group, dtype=np.int64)
        means = [_mean_cross_code(codes, width, rows, part) for part in part_arrays]
        new_parts[int(np.argmin(means))].extend(group)
    return new_parts


def split_year_gaps(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    years: Mapping[str, int],
    close_code: int,
    *,
    orcids: Mapping[str, str] | None = None,
    seed_groups: Mapping[str, object] | None = None,
    first_names: Mapping[str, str] | None = None,
    block_key: str = "",
) -> np.ndarray | Sequence[int]:
    """Split each cluster at empty stretches of more than ``YEAR_DIFF_CAP`` years.

    Args:
        block_signatures: Signature ids in matrix order.
        labels: Cluster label per signature.
        codes: One-byte condensed pair codes from ``s2and.pair_codes.quantize_distances``.
        years: Paper year per signature id. Years at or below zero and ``PLACEHOLDER_YEARS``
            count as undated; a placeholder still guides where its mention goes.
        close_code: Largest code that is certainly under the close-pair distance; a pair at
            or under it keeps two parts together. Negative keeps no parts together by distance.
        orcids: ORCID per signature id; two parts that share one stay together.
        seed_groups: Required seed group per signature id; two parts that share one stay together.
        first_names: First word of the full first name per signature id; two parts that share
            one stay together. Give it only for signatures whose surname block has under
            ``SMALL_SURNAME_BLOCK`` mentions.
        block_key: Block name for the log.

    Returns:
        ``labels`` unchanged when nothing splits, else new labels. The largest part of a split
        cluster keeps its label; the other parts get new labels.
    """
    clusters: dict[int, list[int]] = {}
    for position, label in enumerate(np.asarray(labels).tolist()):
        clusters.setdefault(int(label), []).append(position)
    new_labels: np.ndarray | None = None
    next_label = max(clusters, default=-1) + 1
    # Read only this block's years: a multi-block request holds every block's years.
    years = {s: years[s] for s in block_signatures if s in years}
    stamps = {signature_id: year for signature_id, year in years.items() if year in PLACEHOLDER_YEARS}
    years = split_years(years)
    for members in clusters.values():
        if sum(block_signatures[m] in years for m in members) < 2:
            continue
        parts = _parts(
            members,
            block_signatures,
            years,
            codes,
            close_code,
            orcids or {},
            seed_groups or {},
            first_names or {},
            stamps,
        )
        if len(parts) == 1:
            continue
        if new_labels is None:
            new_labels = np.array(labels, dtype=np.int64, copy=True)
        parts.sort(key=len, reverse=True)
        for part in parts[1:]:
            new_labels[part] = next_label
            next_label += 1
        spans = []
        for part in parts:
            part_years = [years[block_signatures[m]] for m in part if block_signatures[m] in years]
            spans.append((min(part_years), max(part_years)) if part_years else None)
        logger.info(
            "Year-gap split in block %s: cluster of %d split into parts of %s mentions (years %s)",
            block_key,
            len(members),
            [len(part) for part in parts],
            spans,
        )
    return labels if new_labels is None else new_labels
