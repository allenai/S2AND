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
- without them, each cluster has a mention with a full first name, unless
  ``require_full_first_names`` is off or the side keeps at least
  ``initials_side_min_mentions`` mentions;
- the height between the clusters, without them, is certainly below
  ``max_average_distance``. The height is the average model distance, less
  ``close_pair_bonus`` times the share of pairs under ``close_pair_distance``:
  the merge height that close-pair linkage cuts. A zero bonus makes it the plain
  average;
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
The repair merges only when the upper end of that range is under the limit. A code
counts as a close pair only when its whole range lies under ``close_pair_distance``,
so the share of close pairs is a lower bound and the height an upper bound.

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
import math
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
    close_count: int = 0


def _close_code(repair: NameConflictRepair) -> int:
    """Return the largest code whose whole range lies under the close distance, or -1.

    Code ``c`` holds distances in ``[c, c + 1) / _SCALE``.
    """
    if repair.close_pair_bonus <= 0:
        return -1
    return math.floor(repair.close_pair_distance * _SCALE - 1)


def _height_bound(
    code_sum: float, close_count: float, count: int, repair: NameConflictRepair, smaller_side: int | None = None
) -> float:
    """Upper bound of the close-pair height from a code sum and a lower bound of close pairs.

    With a bonus bound set, a pair whose plain average is over the limit may use the bonus
    only when enough of its pairs are certainly close, its plain average is under the cap and
    its smaller side keeps enough mentions. A few very close pairs then cannot carry two
    clusters that are otherwise far apart. Such a pair gets an infinite bound.
    """
    adjusted = ((code_sum - repair.close_pair_bonus * _SCALE * close_count) / count + 1) / _SCALE
    if repair.close_pair_bonus <= 0 or (
        repair.bonus_min_close_share is None
        and repair.bonus_max_plain_average is None
        and repair.bonus_min_side_mentions is None
    ):
        return adjusted
    plain = (code_sum / count + 1) / _SCALE
    if plain <= repair.max_average_distance:
        return adjusted
    if repair.bonus_min_close_share is not None and close_count < repair.bonus_min_close_share * count:
        return math.inf
    if repair.bonus_max_plain_average is not None and plain > repair.bonus_max_plain_average:
        return math.inf
    if (
        repair.bonus_min_side_mentions is not None
        and smaller_side is not None
        and smaller_side < repair.bonus_min_side_mentions
    ):
        return math.inf
    return adjusted


def _common_odd_average_bound(
    candidate: _RepairCandidate,
    odd: tuple[int, ...],
    members: Mapping[int, np.ndarray],
    codes: np.ndarray,
    block_width: int,
    repair: NameConflictRepair,
) -> float:
    """Recompute a pair's bound after excluding the shared scoring exceptions."""
    side_a, side_b = members[candidate.a], members[candidate.b]
    original_a = side_a[~np.isin(side_a, candidate.odd)]
    original_b = side_b[~np.isin(side_b, candidate.odd)]
    extra_a = original_a[np.isin(original_a, odd)]
    extra_b = original_b[np.isin(original_b, odd)]
    code_sum, close_count = candidate.code_sum, candidate.close_count
    close_code = _close_code(repair)

    def row_sum(mention: int, others: np.ndarray) -> tuple[int, int]:
        low, high = np.minimum(mention, others), np.maximum(mention, others)
        indices = block_width * low - low * (low + 1) // 2 + high - low - 1
        row = codes[indices]
        return int(row.sum(dtype=np.uint64)), int(np.count_nonzero((row != 255) & (row.astype(np.int64) <= close_code)))

    # Read only the extra rows; the kernel already summed the original clean pair.
    for mention in extra_a:
        removed, removed_close = row_sum(int(mention), original_b)
        code_sum, close_count = code_sum - removed, close_count - removed_close
    for mention in extra_b:
        removed, removed_close = row_sum(int(mention), original_a)
        code_sum, close_count = code_sum - removed, close_count - removed_close
    for mention in extra_a:
        added, added_close = row_sum(int(mention), extra_b)
        code_sum, close_count = code_sum + added, close_count + added_close
    side_a_kept, side_b_kept = len(original_a) - len(extra_a), len(original_b) - len(extra_b)
    count = side_a_kept * side_b_kept
    return _height_bound(code_sum, close_count, count, repair, min(side_a_kept, side_b_kept))


@dataclass(frozen=True)
class NameConflictRepair:
    """Settings and per-signature facts for name-conflict repair.

    Attributes:
        max_average_distance: Merge only below this height. With no close-pair bonus,
            the height is the average model distance.
        close_pair_bonus: Bonus that close-pair linkage gives a close pair. Zero
            compares the plain average distance.
        close_pair_distance: Distance under which a pair is close.
        require_full_first_names: Whether each side must keep a mention with a full
            first name once the odd mentions are left out.
        rounds: Most repair rounds per block; each merges a cluster at most once.
        initials_side_min_mentions: A side with no full first name passes the full-name
            test when it keeps at least this many mentions. ``None`` exempts no side.
        bonus_min_close_share: With a close-pair bonus, a pair whose plain average is
            over ``max_average_distance`` needs at least this share of certainly close
            pairs. ``None`` sets no floor.
        bonus_max_plain_average: With a close-pair bonus, the plain average above which
            the bonus cannot carry a pair. ``None`` sets no cap.
        bonus_min_side_mentions: With a close-pair bonus, the fewest mentions each side of
            a pair over ``max_average_distance`` must keep for the bonus to carry it.
            ``None`` sets no minimum beyond the repair's own.
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
    close_pair_bonus: float = 0.0
    close_pair_distance: float = 0.0
    require_full_first_names: bool = True
    rounds: int = 1
    initials_side_min_mentions: int | None = None
    bonus_min_close_share: float | None = None
    bonus_max_plain_average: float | None = None
    bonus_min_side_mentions: int | None = None

    def lacks_full_name(self, side: Sequence[str]) -> bool:
        """Whether a side fails the full-first-name test."""
        if not self.require_full_first_names or any(s in self.full_first_names for s in side):
            return False
        return self.initials_side_min_mentions is None or len(side) < self.initials_side_min_mentions

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if not (math.isfinite(self.close_pair_bonus) and self.close_pair_bonus >= 0):
            raise ValueError("close_pair_bonus must be finite and non-negative")
        if self.close_pair_bonus > 0 and not 0 < self.close_pair_distance <= 1:
            raise ValueError("close_pair_distance must lie in (0, 1] when close_pair_bonus is positive")
        if self.bonus_min_close_share is not None and not 0 <= self.bonus_min_close_share <= 1:
            raise ValueError("bonus_min_close_share must lie in [0, 1]")
        if self.bonus_max_plain_average is not None and not 0 < self.bonus_max_plain_average <= 1:
            raise ValueError("bonus_max_plain_average must lie in (0, 1]")
        if self.bonus_min_side_mentions is not None and self.bonus_min_side_mentions < 1:
            raise ValueError("bonus_min_side_mentions must be at least 1")


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
    """Run up to ``repair.rounds`` repair rounds, stopping at the first that merges nothing.

    A round merges each cluster at most once. A later round sees the merged clusters, so a
    person split three ways by different odd mentions can rejoin over two rounds.
    """
    current = labels
    for _ in range(repair.rounds):
        repaired = _repair_round(
            block_signatures, current, codes, repair, forbidden=forbidden, seed_groups=seed_groups, block_key=block_key
        )
        if repaired is current:
            break
        current = repaired
    return current


def _repair_round(
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
        codes,
        cluster,
        len(big),
        MIN_SIDE,
        repair.max_average_distance * _SCALE,
        MAX_ODD_MENTIONS,
        _close_code(repair),
        repair.close_pair_bonus * _SCALE,
    )
    if not candidates:
        return labels

    members = {index: np.flatnonzero(cluster == index) for index in {c for a, b, *_ in candidates for c in (a, b)}}
    accepted: list[_RepairCandidate] = []
    for a, b, odd, code_sum, close_count in candidates:
        odd_ids = {block_signatures[m] for m in odd}
        all_a = [block_signatures[m] for m in members[a]]
        all_b = [block_signatures[m] for m in members[b]]
        ids_a = [s for s in all_a if s not in odd_ids]
        ids_b = [s for s in all_b if s not in odd_ids]
        # Codes round down, so the true average is at most one step above the coded one.
        # Merge only when the height is certainly under the limit.
        count = len(ids_a) * len(ids_b)
        bound = _height_bound(code_sum, close_count, count, repair, min(len(ids_a), len(ids_b)))
        if bound > repair.max_average_distance:
            continue
        # The middle of the range orders the merges and goes to the log.
        average = bound - 0.5 / _SCALE
        if repair.lacks_full_name(ids_a) or repair.lacks_full_name(ids_b):
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
        accepted.append(_RepairCandidate(average, a, b, tuple(odd), code_sum, close_count))
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
            "Name-conflict repair: block %s merges clusters of %d and %d mentions, height %.3f, odd mentions %s",
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
                if any(repair.lacks_full_name([block_signatures[m] for m in side]) for side in kept):
                    continue
                bound = max(
                    _common_odd_average_bound(item, odd, members, codes, len(block_signatures), repair)
                    for item in (ab, ac, bc)
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
