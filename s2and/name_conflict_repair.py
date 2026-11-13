"""Name-conflict repair: merge two clusters that one or two odd mentions keep apart.

The name rules mark a pair of mentions as impossible when their names conflict,
and average linkage never merges two clusters across such a pair. One mention
with a wrong or misparsed name inside a person's cluster therefore keeps that
cluster away from the rest of the person's work.

After a block is clustered, this module merges two clusters when all of these hold:

- at most ``MAX_ODD_MENTIONS`` mentions take part in every impossible pair between
  the two clusters; the smallest such set is the odd mentions, and a larger set is
  not tried when the smallest fails a later test;
- without them, each cluster keeps at least ``MIN_SIDE`` mentions;
- without them, each cluster has a mention with a full first name;
- the average model distance between the clusters, without them, is certainly
  below ``max_average_distance``;
- the two clusters do not carry different majority ORCIDs, odd mentions included;
- the two clusters do not hold mentions of two different required seed groups;
- no pair across the two clusters is a seed disallow or a profile exclusion;
- every impossible pair between the two clusters is one the name rules produce
  on their own, with no seed constraint on the pair.

The kernel cannot tell why a pair is impossible. Two claimed profiles, a seed disallow
and a caller's own supervision are kept apart by the same impossible pairs as two
conflicting names. The last test asks the object that built the distances, so a pair
whose names do not conflict is never read as a name conflict. The seed-group
and forbidden-pair tests also cover seeds that only the caller knows.
A precomputed matrix can hold a caller's impossible pair on names that also conflict,
which nothing can tell apart; ``predict(..., dists=...)`` therefore skips the repair.

Codes round distances down, so the average is known to within one code step (0.004).
The repair merges only when the upper end of that range is under the limit.

The odd mentions stay in the merged cluster. Blind judging on seven large blocks
called all 28 sampled merges with one odd mention and all 13 with two correct, and
found most odd mentions belonged to the merged person.

After the pair merges, a repaired pair can absorb one untouched original cluster.
Every original pair must pass these checks. One shared set of at most two odd
mentions must explain their conflicts, and every clean pair must remain below
the distance limit after removing that same set. Each original cluster must keep
five mentions and a full first name. An extension never grows beyond three
original clusters or changes another pair merge.

``Clusterer.name_conflict_repair_max_distance`` sets the distance for every route
that clusters blocks; ``None`` turns the repair off.

Linkage may overwrite the distance matrix, so the caller keeps a one-byte-per-pair
copy made by ``quantize_distances`` before the first linkage.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from s2and.consts import LARGE_DISTANCE
from s2and.runtime import load_s2and_rust_extension

logger = logging.getLogger("s2and")

# Blind judging found 28 of 28 merges below this average distance correct.
DEFAULT_MAX_AVERAGE_DISTANCE = 0.2
MIN_SIDE = 5
MAX_ODD_MENTIONS = 2
# A block needs two sides and an odd mention.
MIN_BLOCK_SIZE = 2 * MIN_SIDE + 1
# A code is the distance times the scale, rounded down: [0, 1] maps to 0..254,
# and 255 marks an impossible pair. The Rust kernels use the same scale.
_SCALE = 254.999


@dataclass(frozen=True)
class _RepairCandidate:
    """Keep the original pair evidence for a possible three-cluster repair."""

    average: float
    a: int
    b: int
    odd: tuple[int, ...]
    code_sum: int


def _common_odd_average_bound(
    candidate: _RepairCandidate,
    odd: tuple[int, ...],
    members: Mapping[int, np.ndarray],
    codes: np.ndarray,
    block_width: int,
) -> float:
    """Recompute a pair's bound after excluding the shared scoring exceptions."""
    side_a, side_b = members[candidate.a], members[candidate.b]
    original_a = side_a[~np.isin(side_a, candidate.odd)]
    original_b = side_b[~np.isin(side_b, candidate.odd)]
    extra_a = original_a[np.isin(original_a, odd)]
    extra_b = original_b[np.isin(original_b, odd)]
    code_sum = candidate.code_sum

    def row_sum(mention: int, others: np.ndarray) -> int:
        low, high = np.minimum(mention, others), np.maximum(mention, others)
        indices = block_width * low - low * (low + 1) // 2 + high - low - 1
        return int(codes[indices].sum(dtype=np.uint64))

    # Read only the extra rows; the kernel already summed the original clean pair.
    for mention in extra_a:
        code_sum -= row_sum(int(mention), original_b)
    for mention in extra_b:
        code_sum -= row_sum(int(mention), original_a)
    for mention in extra_a:
        code_sum += row_sum(int(mention), extra_b)
    count = (len(original_a) - len(extra_a)) * (len(original_b) - len(extra_b))
    return (code_sum / count + 1) / _SCALE


@dataclass(frozen=True)
class NameConflictRepair:
    """Settings and per-signature facts for name-conflict repair.

    Attributes:
        max_average_distance: Merge only below this average model distance.
        full_first_names: Signature ids whose first name is longer than one letter.
        orcids: ORCID per signature id, for signatures that have one.
        name_rules_alone_forbid: For pairs of signature ids, whether the name rules make
            each pair impossible, no seed disallow holds it, and its signatures are not in
            two different required groups. The repair merges two clusters only when this
            holds for every impossible pair between them.
            ``None`` skips the test; every route that clusters real blocks supplies it.
    """

    max_average_distance: float
    full_first_names: Set[str] = field(default_factory=frozenset)
    orcids: Mapping[str, str] = field(default_factory=dict)
    name_rules_alone_forbid: Callable[[Sequence[tuple[str, str]]], Sequence[bool]] | None = field(
        default=None, compare=False
    )


def has_full_first_name(first: str | None) -> bool:
    """Return whether a normalized first name holds a word longer than one letter.

    "j p", from "J.-P.", is two initials and not a full first name.
    """
    return any(len(word) > 1 for word in (first or "").split())


def quantize_distances(condensed: np.ndarray, *, num_threads: int | None = None) -> np.ndarray | None:
    """Return a one-byte-per-pair copy of condensed distances.

    Args:
        condensed: Condensed float64 distances in [0, 1], with exactly
            ``LARGE_DISTANCE`` for impossible pairs.
        num_threads: Rust pool size; None uses the default pool.

    Returns:
        ``uint8`` codes: 255 for an impossible pair, else the distance times
        254.999, rounded down. ``None`` when no repair can apply: the matrix has
        no impossible pair, is not a condensed float64 matrix, or holds another
        value above one. Such a value is a caller's soft "keep apart" distance,
        which linkage honors and a one-byte code cannot carry.
    """
    if condensed.ndim != 1 or condensed.dtype != np.float64 or not condensed.flags.c_contiguous:
        return None
    return load_s2and_rust_extension().quantize_block_distances(condensed, float(LARGE_DISTANCE), num_threads)


def _majority_orcid(signature_ids: Sequence[str], orcids: Mapping[str, str]) -> str | None:
    counts = Counter(orcids[s] for s in signature_ids if s in orcids)
    return counts.most_common(1)[0][0] if counts else None


def _impossible_pairs(
    codes: np.ndarray, block_width: int, side_a: np.ndarray, side_b: np.ndarray, odd: Sequence[int]
) -> list[tuple[int, int]]:
    """Return the impossible pairs between two clusters as matrix positions.

    The kernel chose ``odd`` so that every impossible pair between the clusters holds an
    odd mention; only those rows are read.
    """
    in_a = set(side_a.tolist())
    pairs: set[tuple[int, int]] = set()
    for mention in odd:
        others = side_b if mention in in_a else side_a
        low, high = np.minimum(mention, others), np.maximum(mention, others)
        condensed = block_width * low - low * (low + 1) // 2 + high - low - 1
        for other in others[codes[condensed] == 255].tolist():
            pairs.add((min(mention, other), max(mention, other)))
    return sorted(pairs)


def repair_name_conflicts_in_block(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    repair: NameConflictRepair,
    *,
    forbidden: Mapping[str, Set[str]] | None = None,
    seed_groups: Mapping[str, object] | None = None,
    block_key: str = "",
) -> np.ndarray | Sequence[int]:
    """Merge cluster pairs that one or two odd mentions keep apart.

    Args:
        block_signatures: Signature ids in matrix order.
        labels: Cluster label per signature.
        codes: ``quantize_distances`` output for the block's model distances.
        repair: Settings and per-signature facts.
        forbidden: Signature ids that may not share a cluster with each key,
            from seed disallows and profile exclusions.
        seed_groups: Required seed group per signature id, when different groups
            must stay in different clusters.
        block_key: Block name for logs.

    Returns:
        ``labels`` unchanged when nothing merges, else a new label array.
    """
    label_array = np.asarray(labels)
    unique, inverse, sizes = np.unique(label_array, return_inverse=True, return_counts=True)
    big = np.flatnonzero(sizes >= MIN_SIDE)
    if len(big) < 2:
        return labels
    # Compact cluster index for clusters of at least MIN_SIDE mentions, else -1.
    compact_of_cluster = np.full(len(unique), -1, dtype=np.int64)
    compact_of_cluster[big] = np.arange(len(big))
    cluster = np.ascontiguousarray(compact_of_cluster[inverse])
    # The kernel applies the size and distance tests; the bound it gets is
    # one code step loose, and the strict test follows below.
    candidates = load_s2and_rust_extension().name_conflict_candidates(
        codes, cluster, len(big), MIN_SIDE, repair.max_average_distance * _SCALE, MAX_ODD_MENTIONS
    )
    if not candidates:
        return labels

    members = {index: np.flatnonzero(cluster == index) for index in {c for a, b, _, _ in candidates for c in (a, b)}}
    full_names = repair.full_first_names
    accepted: list[_RepairCandidate] = []
    for a, b, odd, code_sum in candidates:
        odd_ids = {block_signatures[m] for m in odd}
        all_a = [block_signatures[m] for m in members[a]]
        all_b = [block_signatures[m] for m in members[b]]
        ids_a = [s for s in all_a if s not in odd_ids]
        ids_b = [s for s in all_b if s not in odd_ids]
        # Codes round down, so the true average is at most one step above the coded one.
        # Merge only when it is certainly under the limit.
        coded_average = code_sum / (len(ids_a) * len(ids_b))
        if (coded_average + 1) / _SCALE > repair.max_average_distance:
            continue
        # The middle of the range orders the merges and goes to the log.
        average = (coded_average + 0.5) / _SCALE
        if not any(s in full_names for s in ids_a) or not any(s in full_names for s in ids_b):
            continue
        if repair.orcids:
            # The odd mentions stay in the merged cluster, so their ORCIDs count.
            orcid_a = _majority_orcid(all_a, repair.orcids)
            orcid_b = _majority_orcid(all_b, repair.orcids)
            if orcid_a is not None and orcid_b is not None and orcid_a != orcid_b:
                continue
        if seed_groups:
            # Impossible pairs between two required seed groups are not name conflicts.
            groups = {seed_groups[s] for s in (*all_a, *all_b) if s in seed_groups}
            if len(groups) > 1:
                continue
        if forbidden:
            # The odd mentions stay in the merged cluster, so their forbidden pairs count too.
            set_a, set_b = set(all_a), set(all_b)
            if any(not set_b.isdisjoint(forbidden.get(s, ())) for s in all_a):
                continue
            if any(not set_a.isdisjoint(forbidden.get(s, ())) for s in all_b):
                continue
        if repair.name_rules_alone_forbid is not None:
            # An impossible pair that the name rules do not explain comes from a seed or from
            # the caller's supervision, and the repair may not merge across it.
            impossible = _impossible_pairs(codes, len(block_signatures), members[a], members[b], odd)
            verdicts = repair.name_rules_alone_forbid(
                [(block_signatures[left], block_signatures[right]) for left, right in impossible]
            )
            if not all(verdicts):
                continue
        accepted.append(_RepairCandidate(average, a, b, tuple(odd), code_sum))
    if not accepted:
        return labels

    # Closest pair first. A cluster takes part in one merge per block, so no
    # merge rests on a cluster that an earlier merge changed.
    merged = label_array.copy()
    used: set[int] = set()
    selected: list[_RepairCandidate] = []
    for candidate in sorted(accepted, key=lambda item: (item.average, item.a, item.b, item.odd)):
        average, a, b, odd = candidate.average, candidate.a, candidate.b, candidate.odd
        if a in used or b in used:
            continue
        used.update((a, b))
        selected.append(candidate)
        merged[members[b]] = unique[big[a]]
        logger.info(
            "Name-conflict repair: block %s merges clusters of %d and %d mentions, "
            "average distance %.3f, odd mentions %s",
            block_key,
            int(sizes[big[a]]),
            int(sizes[big[b]]),
            average,
            ", ".join(block_signatures[m] for m in odd),
        )

    # Keep every pair decision. Extend only when all three original pairs agree.
    by_pair: dict[tuple[int, int], list[_RepairCandidate]] = {}
    neighbors: dict[int, set[int]] = {}
    for candidate in accepted:
        by_pair.setdefault((candidate.a, candidate.b), []).append(candidate)
        neighbors.setdefault(candidate.a, set()).add(candidate.b)
        neighbors.setdefault(candidate.b, set()).add(candidate.a)
    extensions: list[tuple[float, int, int, int, tuple[int, ...]]] = []
    for pair in selected:
        a, b = pair.a, pair.b
        for c in sorted((neighbors[a] & neighbors[b]) - used):
            for ab, ac, bc in product(by_pair[a, b], by_pair[min(a, c), max(a, c)], by_pair[min(b, c), max(b, c)]):
                odd = tuple(sorted(set(ab.odd) | set(ac.odd) | set(bc.odd)))
                if len(odd) > MAX_ODD_MENTIONS:
                    continue
                # A shared exception can remove another side's only full name.
                kept = [members[index][~np.isin(members[index], odd)] for index in (a, b, c)]
                if any(len(side) < MIN_SIDE for side in kept):
                    continue
                if any(not any(block_signatures[m] in full_names for m in side) for side in kept):
                    continue
                bound = max(
                    _common_odd_average_bound(item, odd, members, codes, len(block_signatures)) for item in (ab, ac, bc)
                )
                if bound <= repair.max_average_distance:
                    extensions.append((bound, a, b, c, odd))
    extended: set[tuple[int, int]] = set()
    for bound, a, b, c, odd in sorted(extensions):
        if c in used or (a, b) in extended:
            continue
        used.add(c)
        extended.add((a, b))
        merged[members[c]] = unique[big[a]]
        logger.info(
            "Name-conflict repair: block %s extends clusters of %d, %d and %d mentions, "
            "maximum distance bound %.3f, odd mentions %s",
            block_key,
            int(sizes[big[a]]),
            int(sizes[big[b]]),
            int(sizes[big[c]]),
            bound,
            ", ".join(block_signatures[m] for m in odd),
        )
    return merged
