"""Year-gap split: separate the parts of a cluster that an empty stretch of years divides.

The pair model sees the years between two papers capped at ``YEAR_DIFF_CAP``, so it cannot
tell 50 years apart from 80. Topic overlap can then pull a 1928 paper into the profile of an
author who started publishing in 2000. A real career leaves papers all along, so a cluster
whose dated mentions leave an empty stretch longer than the cap is cut there.

Two parts across a cut stay together when the model is certain of some pair between them
(a code under the close-pair distance), or they share an ORCID or a required seed group.
A reprint or a re-dated record usually has such a pair: the same title and coauthors. The
step only splits, so no other cluster changes. Undated mentions follow a part that shares
their ORCID or seed group, else the part with the lowest average code.

The split runs after name-conflict repair on the repair's codes, with the repair's
ORCIDs and seed groups, so the repair cannot undo it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from s2and.consts import YEAR_DIFF_CAP

logger = logging.getLogger("s2and")

# Rows per batch when reading the codes between two parts.
_BATCH_CELLS = 1 << 22


def _cross_codes(codes: np.ndarray, block_width: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return the codes between two disjoint groups of mentions, one row per mention of ``rows``."""
    low = np.minimum(rows[:, None], cols[None, :])
    high = np.maximum(rows[:, None], cols[None, :])
    return codes[block_width * low - low * (low + 1) // 2 + high - low - 1]


def _has_close_pair(codes: np.ndarray, block_width: int, a: np.ndarray, b: np.ndarray, close_code: int) -> bool:
    """Return whether some pair between two groups has a code of at most ``close_code``."""
    if len(a) > len(b):
        a, b = b, a
    step = max(1, _BATCH_CELLS // max(1, len(b)))
    return any(
        bool((_cross_codes(codes, block_width, a[start : start + step], b) <= close_code).any())
        for start in range(0, len(a), step)
    )


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
) -> list[list[int]]:
    """Return the parts of one cluster, as lists of block positions; one part when it stays whole."""
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
                or _has_close_pair(codes, width, arrays[i], arrays[j], close_code)
            ):
                parent[find(i)] = find(j)
    grouped: dict[int, list[int]] = {}
    for i, segment in enumerate(segments):
        grouped.setdefault(find(i), []).extend(segment)
    parts = list(grouped.values())
    if len(parts) == 1:
        return [list(members)]
    part_ids = [[block_signatures[m] for m in part] for part in parts]
    part_arrays = [np.asarray(part, dtype=np.int64) for part in parts]
    for mention in members:
        signature_id = block_signatures[mention]
        if signature_id in years:
            continue
        home = next(
            (
                k
                for k, ids_k in enumerate(part_ids)
                if _shared(orcids, [signature_id], ids_k) or _shared(seed_groups, [signature_id], ids_k)
            ),
            None,
        )
        if home is None:
            # Impossible pairs count as code 255, so a name conflict pushes the mention away.
            row = np.asarray([mention], dtype=np.int64)
            means = [float(_cross_codes(codes, width, row, part).mean()) for part in part_arrays]
            home = int(np.argmin(means))
        parts[home].append(mention)
    return parts


def split_year_gaps(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    years: Mapping[str, int],
    close_code: int,
    *,
    orcids: Mapping[str, str] | None = None,
    seed_groups: Mapping[str, object] | None = None,
    block_key: str = "",
) -> np.ndarray | Sequence[int]:
    """Split each cluster at empty stretches of more than ``YEAR_DIFF_CAP`` years.

    Args:
        block_signatures: Signature ids in matrix order.
        labels: Cluster label per signature.
        codes: One-byte condensed pair codes from ``quantize_distances``.
        years: Paper year per signature id, for signatures with a year above zero.
        close_code: Largest code that is certainly under the close-pair distance; a pair at
            or under it keeps two parts together. Negative keeps no parts together by distance.
        orcids: ORCID per signature id; two parts that share one stay together.
        seed_groups: Required seed group per signature id; two parts that share one stay together.
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
    for members in clusters.values():
        if sum(block_signatures[m] in years for m in members) < 2:
            continue
        parts = _parts(members, block_signatures, years, codes, close_code, orcids or {}, seed_groups or {})
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
