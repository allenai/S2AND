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

Each round first runs a reading step. A short given name may be joined initials
("Ec" for E. C.), and the name rules veto it against a bare "Edward". Once linkage puts
"Ec" beside "Edward C." mentions, that veto keeps the piece away from the profile that
holds the bare "Edward". Incremental linking accepts a reading when any member of the
profile spells it, so the step does the same: it sets aside each impossible pair between
a reading mention and a mention without a reading, when either of the two clusters
holds a mention that spells the reading; the reading mention's own cluster counts. When
only the own cluster spells it, a pair is set aside only when some own-cluster speller has
no impossible pair with the other mention; otherwise the reading is not the only veto. A
set-aside pair has no model distance, so it counts as the farthest possible pair. Two
clusters with no other impossible pair merge when their close-pair height is under the
linkage cut, as linkage would have merged them even with every set-aside pair far apart.
When other impossible pairs remain, at most ``MAX_ODD_MENTIONS`` mentions must explain
them, and the odd-mention tests above apply. Pair verdicts do not change.

``Clusterer.name_conflict_repair_max_distance`` sets the distance for every route
that clusters blocks; ``None`` turns the repair off.

Linkage may overwrite the distance matrix, so the caller keeps a one-byte-per-pair
copy made by ``quantize_distances`` before the first linkage.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
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
# Clusterer defaults for the close-pair repair. On 127 dev blocks, 7 ORCID-labelled giant
# blocks and 40 fresh blocks, these bounds made no ORCID-confirmed wrong merge. Five rounds
# rejoin a profile split into several parts. A smaller side under 10 mentions was where the
# judged-wrong merges concentrated.
DEFAULT_REPAIR_ROUNDS = 5
DEFAULT_BONUS_MIN_CLOSE_SHARE = 0.10
DEFAULT_BONUS_MAX_PLAIN_AVERAGE = 0.40
DEFAULT_BONUS_MIN_SIDE_MENTIONS = 10
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


def certain_close_code(close_pair_distance: float) -> int:
    """Return the largest code whose whole range lies under ``close_pair_distance``.

    Code ``c`` holds distances in ``[c, c + 1) / _SCALE``.
    """
    return math.floor(close_pair_distance * _SCALE - 1)


def repair_close_code(repair: NameConflictRepair) -> int:
    """Return the largest code certainly under the repair's close distance, or -1 without a bonus."""
    if repair.close_pair_bonus <= 0:
        return -1
    return certain_close_code(repair.close_pair_distance)


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
    if repair.bonus_min_close_share is not None and close_count / count < repair.bonus_min_close_share:
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
    close_code = repair_close_code(repair)

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
        joined_readings: Joined-initials reading (``given_name_for_constraint`` tag J or W)
            per signature id, for signatures that have one.
        reading_spellers: Per reading, the signature ids without a reading whose name spells
            it (``spelled_reading_forms``). The reading step needs both mappings.
        years: Paper year per signature id from ``s2and.year_gap_split.split_years``. The
            year-gap split reads it after the repair.
        first_names: First word of the normalized first name per signature id, for words of
            two or more characters. The year-gap split reads it in blocks under ``MIN_BLOCK_SIZE``.
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
    joined_readings: Mapping[str, str] = field(default_factory=dict)
    reading_spellers: Mapping[str, Set[str]] = field(default_factory=dict)
    years: Mapping[str, int] = field(default_factory=dict)
    first_names: Mapping[str, str] = field(default_factory=dict)

    def lacks_full_name(self, side: Sequence[str]) -> bool:
        """Whether a side fails the full-first-name test."""
        if not self.require_full_first_names or any(s in self.full_first_names for s in side):
            return False
        return self.initials_side_min_mentions is None or len(side) < self.initials_side_min_mentions

    def __post_init__(self) -> None:
        # bool is an int subclass; True as a count is a caller mistake.
        if not isinstance(self.rounds, int) or isinstance(self.rounds, bool) or self.rounds < 1:
            raise ValueError("rounds must be an integer of at least 1")
        if self.initials_side_min_mentions is not None and (
            not isinstance(self.initials_side_min_mentions, int)
            or isinstance(self.initials_side_min_mentions, bool)
            or self.initials_side_min_mentions < 1
        ):
            raise ValueError("initials_side_min_mentions must be an integer of at least 1")
        if not (math.isfinite(self.close_pair_bonus) and self.close_pair_bonus >= 0):
            raise ValueError("close_pair_bonus must be finite and non-negative")
        if self.close_pair_bonus > 0 and not 0 < self.close_pair_distance <= 1:
            raise ValueError("close_pair_distance must lie in (0, 1] when close_pair_bonus is positive")
        if self.bonus_min_close_share is not None and not 0 <= self.bonus_min_close_share <= 1:
            raise ValueError("bonus_min_close_share must lie in [0, 1]")
        if self.bonus_max_plain_average is not None and not 0 < self.bonus_max_plain_average <= 1:
            raise ValueError("bonus_max_plain_average must lie in (0, 1]")
        if self.bonus_min_side_mentions is not None and (
            not isinstance(self.bonus_min_side_mentions, int)
            or isinstance(self.bonus_min_side_mentions, bool)
            or self.bonus_min_side_mentions < 1
        ):
            raise ValueError("bonus_min_side_mentions must be an integer of at least 1")


def has_full_first_name(first: str | None) -> bool:
    """Return whether a normalized first name holds a word longer than one letter.

    "j p", from "J.-P.", is two initials and not a full first name.
    """
    return any(len(word) > 1 for word in (first or "").split())


def quantize_distances(
    condensed: np.ndarray, *, num_threads: int | None = None, require_hard: bool = True
) -> np.ndarray | None:
    """Return a one-byte-per-pair copy of condensed distances.

    Args:
        condensed: Condensed distances in [0, 1], with exactly ``LARGE_DISTANCE``
            for impossible pairs. Another float dtype or a strided view is copied to
            contiguous float64 first, as linkage accepts those too.
        num_threads: Rust pool size; None uses the default pool.
        require_hard: Return ``None`` for a matrix without an impossible pair,
            where no repair can apply. The year-gap split passes False.

    Returns:
        ``uint8`` codes: 255 for an impossible pair, else the distance times
        254.999, rounded down. ``None`` when the matrix is not a condensed float
        matrix, holds another value above one, or, with ``require_hard``, has no
        impossible pair. A value above one is a caller's soft "keep apart"
        distance, which linkage honors and a one-byte code cannot carry.
    """
    if condensed.ndim != 1 or not np.issubdtype(condensed.dtype, np.floating):
        return None
    condensed = np.ascontiguousarray(condensed, dtype=np.float64)
    return load_s2and_rust_extension().quantize_block_distances(
        condensed, float(LARGE_DISTANCE), num_threads, require_hard
    )


def _top_orcids(signature_ids: Sequence[str], orcids: Mapping[str, str]) -> frozenset[str]:
    """Return every ORCID tied for the most mentions in a cluster.

    Keeping all tied ORCIDs makes the repair independent of mention order.
    """
    counts = Counter(orcids[s] for s in signature_ids if s in orcids)
    if not counts:
        return frozenset()
    most = max(counts.values())
    return frozenset(orcid for orcid, count in counts.items() if count == most)


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


def _passes_merge_guards(
    all_a: Sequence[str],
    all_b: Sequence[str],
    impossible_pairs: Callable[[], Sequence[tuple[int, int]]],
    block_signatures: Sequence[str],
    repair: NameConflictRepair,
    forbidden: Mapping[str, Set[str]] | None,
    seed_groups: Mapping[str, object] | None,
) -> bool:
    """Apply the ORCID, seed, forbidden-pair and name-rule tests to a proposed merge."""
    if repair.orcids:
        # The odd mentions stay in the merged cluster, so their ORCIDs count. Two sides
        # with ORCIDs merge only when they share a leading ORCID.
        orcids_a = _top_orcids(all_a, repair.orcids)
        orcids_b = _top_orcids(all_b, repair.orcids)
        if orcids_a and orcids_b and orcids_a.isdisjoint(orcids_b):
            return False
    if seed_groups:
        # Impossible pairs between two required seed groups are not name conflicts.
        groups = {seed_groups[s] for s in (*all_a, *all_b) if s in seed_groups}
        if len(groups) > 1:
            return False
    if forbidden:
        # The odd mentions stay in the merged cluster, so their forbidden pairs count too.
        set_a, set_b = set(all_a), set(all_b)
        if any(not set_b.isdisjoint(forbidden.get(s, ())) for s in all_a):
            return False
        if any(not set_a.isdisjoint(forbidden.get(s, ())) for s in all_b):
            return False
    if repair.name_rules_alone_forbid is not None:
        # An impossible pair that the name rules do not explain comes from a seed or from
        # the caller's supervision, and the repair may not merge across it.
        verdicts = repair.name_rules_alone_forbid(
            [(block_signatures[left], block_signatures[right]) for left, right in impossible_pairs()]
        )
        if not all(verdicts):
            return False
    return True


def _codes_against(codes: np.ndarray, block_width: int, mention: int, others: np.ndarray) -> np.ndarray:
    """Return the codes between one mention and other mentions, none equal to it."""
    low, high = np.minimum(mention, others), np.maximum(mention, others)
    return codes[block_width * low - low * (low + 1) // 2 + high - low - 1]


# The code of the farthest possible pair, which a set-aside pair counts as.
_FARTHEST_CODE = 254
# Cells per tile when the reading step reads the codes between two clusters.
_TILE_CELLS = 1 << 22


def _tiles(
    codes: np.ndarray, block_width: int, side_a: np.ndarray, side_b: np.ndarray
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (first row, codes) tiles of the codes between two disjoint groups of mentions."""
    step = max(1, _TILE_CELLS // max(1, len(side_b)))
    for start in range(0, len(side_a), step):
        rows = side_a[start : start + step, None]
        low, high = np.minimum(rows, side_b[None, :]), np.maximum(rows, side_b[None, :])
        yield start, codes[block_width * low - low * (low + 1) // 2 + high - low - 1]


def _smallest_covers(cells: Sequence[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    """Return every smallest set of at most ``MAX_ODD_MENTIONS`` rows or columns holding all cells.

    Each entry is (0, row) or (1, column); the covers come sorted, and none fit returns [].
    Every cover holds an end of the first cell, so branching on that choice finds them.
    """

    def search(rest: Sequence[tuple[int, int]], depth: int) -> list[tuple[tuple[int, int], ...]]:
        if not rest:
            return [()]
        if depth == 0:
            return []
        row, column = rest[0]
        found: list[tuple[tuple[int, int], ...]] = []
        for axis, index in ((0, row), (1, column)):
            left = [cell for cell in rest if cell[axis] != index]
            found.extend(((axis, index), *tail) for tail in search(left, depth - 1))
        return found

    for size in range(MAX_ODD_MENTIONS + 1):
        covers = search(cells, size)
        if covers:
            return sorted({tuple(sorted(cover)) for cover in covers})
    return []


@dataclass
class _AsideRule:
    """Which impossible pairs between two clusters a joined-initials reading explains.

    A pair is set aside when one mention is a spelled reading and the other has no reading.
    A reading that only its own cluster spells explains a pair only when one of those
    spellers has no impossible pair with the other mention. Otherwise the conflict is not
    the reading's alone ("Ec" beside "Edward C." against "Edward D.").
    """

    aside: tuple[np.ndarray, np.ndarray]
    plain: tuple[np.ndarray, np.ndarray]
    # Per mention, an index into the agreement masks over the other side; row 0 is all True.
    groups: tuple[np.ndarray, np.ndarray]
    agree: tuple[np.ndarray, np.ndarray]

    def explained(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Return the rows x columns cells the readings explain if the pair is impossible."""
        aside_rows, aside_columns = self.aside
        plain_rows, plain_columns = self.plain
        row_groups, column_groups = self.groups
        row_agree, column_agree = self.agree
        rows_ok = row_agree[row_groups[rows]][:, columns]
        columns_ok = column_agree[:, rows][column_groups[columns]].T
        return (aside_rows[rows, None] & plain_columns[None, columns] & rows_ok) | (
            plain_rows[rows, None] & aside_columns[None, columns] & columns_ok
        )


@dataclass
class _PairEvidence:
    """Codes between two clusters, summed per row and per column over possible pairs."""

    row_sum: np.ndarray
    row_count: np.ndarray
    row_close: np.ndarray
    column_sum: np.ndarray
    column_count: np.ndarray
    column_close: np.ndarray
    row_aside: np.ndarray
    column_aside: np.ndarray
    residual: list[tuple[int, int]]
    rule: _AsideRule

    @property
    def set_aside(self) -> int:
        """Return the number of set-aside cells."""
        return int(self.row_aside.sum())


def _pair_evidence(
    codes: np.ndarray,
    block_width: int,
    sides: tuple[np.ndarray, np.ndarray],
    rule: _AsideRule,
    close_code: int,
) -> _PairEvidence | None:
    """Read two clusters tile by tile; None when no small cover can explain the other conflicts.

    ``rule`` says which impossible cells a reading explains; those are set aside. A cover of
    at most two rows or columns holds at most two full lines, so more remaining cells than
    that ends the read early and bounds the memory.
    """
    side_a, side_b = sides
    all_columns = np.arange(len(side_b))
    limit = MAX_ODD_MENTIONS * max(len(side_a), len(side_b))
    evidence = _PairEvidence(
        *(np.zeros(len(side_a), dtype=np.int64) for _ in range(3)),
        *(np.zeros(len(side_b), dtype=np.int64) for _ in range(3)),
        row_aside=np.zeros(len(side_a), dtype=np.int64),
        column_aside=np.zeros(len(side_b), dtype=np.int64),
        residual=[],
        rule=rule,
    )
    for start, tile in _tiles(codes, block_width, side_a, side_b):
        stop = start + len(tile)
        impossible = tile == 255
        possible = ~impossible
        values = np.where(possible, tile, 0).astype(np.int64)
        close = possible & (tile <= close_code) if close_code >= 0 else np.zeros_like(possible)
        evidence.row_sum[start:stop] = values.sum(axis=1)
        evidence.row_count[start:stop] = possible.sum(axis=1)
        evidence.row_close[start:stop] = close.sum(axis=1)
        evidence.column_sum += values.sum(axis=0)
        evidence.column_count += possible.sum(axis=0)
        evidence.column_close += close.sum(axis=0)
        set_aside = impossible & rule.explained(np.arange(start, stop), all_columns)
        evidence.row_aside[start:stop] = set_aside.sum(axis=1)
        evidence.column_aside += set_aside.sum(axis=0)
        rows, columns = np.nonzero(impossible & ~set_aside)
        if len(evidence.residual) + len(rows) > limit:
            return None
        evidence.residual.extend(zip((rows + start).tolist(), columns.tolist(), strict=True))
    return evidence


def _reading_merge_bound(
    evidence: _PairEvidence,
    odd: tuple[list[int], list[int]],
    sides: tuple[np.ndarray, np.ndarray],
    codes: np.ndarray,
    block_signatures: Sequence[str],
    repair: NameConflictRepair,
    cut_height: float,
    close_code: int,
) -> float | None:
    """Return the merge height of a reading-step pair once the odd rows and columns are left out."""
    side_a, side_b = sides
    odd_rows, odd_columns = odd
    code_sum = int(evidence.row_sum.sum() - evidence.row_sum[odd_rows].sum() - evidence.column_sum[odd_columns].sum())
    count = int(
        evidence.row_count.sum() - evidence.row_count[odd_rows].sum() - evidence.column_count[odd_columns].sum()
    )
    close_count = int(
        evidence.row_close.sum() - evidence.row_close[odd_rows].sum() - evidence.column_close[odd_columns].sum()
    )
    aside = int(
        evidence.row_aside.sum() - evidence.row_aside[odd_rows].sum() - evidence.column_aside[odd_columns].sum()
    )
    if odd_rows and odd_columns:
        # Cells in an odd row and an odd column were taken away twice.
        for row in odd_rows:
            cross = _codes_against(codes, len(block_signatures), int(side_a[row]), side_b[odd_columns])
            possible = cross != 255
            code_sum += int(cross[possible].astype(np.int64).sum())
            count += int(np.count_nonzero(possible))
            close_count += int(np.count_nonzero(possible & (cross <= close_code))) if close_code >= 0 else 0
            columns = np.asarray(odd_columns)
            aside += int(np.count_nonzero(~possible & evidence.rule.explained(np.asarray([row]), columns)[0]))
    # A set-aside pair has no model distance. It counts as the farthest possible pair, so the
    # merge holds even if every such pair is far apart, and a few spellers cannot carry two
    # clusters whose pairs are mostly set aside.
    code_sum += _FARTHEST_CODE * aside
    count += aside
    if count == 0:
        return None
    if odd_rows or odd_columns:
        # Other impossible pairs remain, so the odd-mention tests apply.
        rows, columns = set(odd_rows), set(odd_columns)
        kept_a = [block_signatures[m] for i, m in enumerate(side_a.tolist()) if i not in rows]
        kept_b = [block_signatures[m] for j, m in enumerate(side_b.tolist()) if j not in columns]
        smaller = min(len(kept_a), len(kept_b))
        if smaller < MIN_SIDE or repair.lacks_full_name(kept_a) or repair.lacks_full_name(kept_b):
            return None
        bound = _height_bound(code_sum, close_count, count, repair, smaller)
        return bound if bound <= repair.max_average_distance else None
    # Only set-aside pairs kept the clusters apart: merge where linkage would have.
    bound = ((code_sum - repair.close_pair_bonus * _SCALE * close_count) / count + 1) / _SCALE
    return bound if bound <= cut_height else None


def _name_rules_explain_every_pair(
    codes: np.ndarray, block_signatures: Sequence[str], sides: tuple[np.ndarray, np.ndarray], repair: NameConflictRepair
) -> bool:
    """Ask the name-rule test about every impossible pair between two clusters, one tile at a time."""
    if repair.name_rules_alone_forbid is None:
        return True
    side_a, side_b = sides
    for start, tile in _tiles(codes, len(block_signatures), side_a, side_b):
        rows, columns = np.nonzero(tile == 255)
        left, right = side_a[rows + start].tolist(), side_b[columns].tolist()
        batch = [(block_signatures[x], block_signatures[y]) for x, y in zip(left, right, strict=True)]
        if batch and not all(repair.name_rules_alone_forbid(batch)):
            return False
    return True


def _reading_round(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    repair: NameConflictRepair,
    cut_height: float | None,
    *,
    forbidden: Mapping[str, Set[str]] | None = None,
    seed_groups: Mapping[str, object] | None = None,
    block_key: str = "",
) -> np.ndarray | Sequence[int]:
    """Merge cluster pairs that joined-initials readings keep apart (see the module docstring).

    Returns:
        ``labels`` unchanged when nothing merges, else a new label array.
    """
    readings, spellers = repair.joined_readings, repair.reading_spellers
    if cut_height is None or not readings or not spellers:
        return labels
    width = len(block_signatures)
    is_reading = np.fromiter((s in readings for s in block_signatures), dtype=bool, count=width)
    if not is_reading.any():
        return labels
    label_array = np.asarray(labels)
    unique, inverse = np.unique(label_array, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(unique) + 1))

    def members(cluster: int) -> np.ndarray:
        return order[bounds[cluster] : bounds[cluster + 1]]

    position = {signature: index for index, signature in enumerate(block_signatures)}
    # Reading mentions by reading, and the clusters that hold a mention spelling each reading.
    by_value: dict[str, list[int]] = {}
    for mention in np.flatnonzero(is_reading).tolist():
        by_value.setdefault(readings[block_signatures[mention]], []).append(mention)
    spelling_clusters = {
        value: {int(inverse[position[s]]) for s in spellers.get(value, ()) if s in position} for value in by_value
    }
    # Candidates: a reading mention's cluster and each other cluster that spells its reading and
    # holds a mention it has an impossible pair with. When the reading mention's own cluster
    # spells the reading, also each cluster holding such a mention that an own-cluster speller
    # agrees with: the reading is then that mention's only veto ("Ec" beside "Edward C.", against
    # a bare "Edward"). The speller test keeps the block scan to clusters that can merge.
    pairs: set[tuple[int, int]] = set()
    everyone = np.arange(width)
    for value, mentions in by_value.items():
        if not spelling_clusters[value]:
            continue
        targets = np.concatenate([members(c) for c in sorted(spelling_clusters[value])])
        speller_positions = np.asarray(
            sorted(position[s] for s in spellers.get(value, ()) if s in position), dtype=np.int64
        )
        # Whether some own-cluster speller agrees with a mention, read once per mention and cluster.
        # Reading the spellers per reading mention costs readings x spellers x block width.
        # Mentions go cluster by cluster, so only one cluster's masks are held at a time.
        checked = agreed = np.zeros(0, dtype=bool)
        current = -1
        for mention in sorted(mentions, key=lambda m: (int(inverse[m]), m)):
            own = int(inverse[mention])
            others = targets[inverse[targets] != own]
            if len(others):
                partners = others[_codes_against(codes, width, mention, others) == 255]
                pairs.update((min(own, c), max(own, c)) for c in np.unique(inverse[partners]).tolist())
            if own not in spelling_clusters[value]:
                continue
            if own != current:
                current = own
                checked = np.zeros(width, dtype=bool)
                agreed = np.zeros(width, dtype=bool)
            outside = everyone[inverse != own]
            vetoed = outside[_codes_against(codes, width, mention, outside) == 255]
            unread = vetoed[~checked[vetoed]]
            checked[unread] = True
            # Each speller reads only the mentions no earlier speller agreed with.
            for speller in speller_positions[inverse[speller_positions] == own].tolist():
                if not len(unread):
                    break
                agrees = _codes_against(codes, width, speller, unread) != 255
                agreed[unread[agrees]] = True
                unread = unread[~agrees]
            joinable = vetoed[agreed[vetoed]]
            pairs.update((min(own, c), max(own, c)) for c in np.unique(inverse[joinable]).tolist())
    if not pairs:
        return labels

    def spelled_by(side: np.ndarray, pair: tuple[int, int]) -> np.ndarray:
        # Reading mentions of this side whose reading one of the two clusters spells.
        return np.fromiter(
            (
                bool(is_reading[m]) and not spelling_clusters[readings[block_signatures[m]]].isdisjoint(pair)
                for m in side.tolist()
            ),
            dtype=bool,
            count=len(side),
        )

    def own_agreement(side: np.ndarray, other: np.ndarray, own: int, partner: int) -> tuple[np.ndarray, np.ndarray]:
        # For reading mentions only their own cluster spells: which other-side mentions an
        # own-cluster speller of that reading has no impossible pair with.
        groups = np.zeros(len(side), dtype=np.int64)
        masks = [np.ones(len(other), dtype=bool)]
        group_of: dict[str, int] = {}
        for i, m in enumerate(side.tolist()):
            if not is_reading[m]:
                continue
            value = readings[block_signatures[m]]
            if partner in spelling_clusters[value] or own not in spelling_clusters[value]:
                continue
            if value not in group_of:
                own_spellers = np.asarray(
                    [position[s] for s in spellers[value] if s in position and inverse[position[s]] == own],
                    dtype=np.int64,
                )
                agrees = np.zeros(len(other), dtype=bool)
                for _, tile in _tiles(codes, width, own_spellers, other):
                    agrees |= (tile != 255).any(axis=0)
                group_of[value] = len(masks)
                masks.append(agrees)
            groups[i] = group_of[value]
        return groups, np.stack(masks)

    close_code = repair_close_code(repair)
    accepted: list[tuple[float, int, int, tuple[int, ...]]] = []
    for a, b in sorted(pairs):
        sides = (members(a), members(b))
        all_a = [block_signatures[m] for m in sides[0].tolist()]
        all_b = [block_signatures[m] for m in sides[1].tolist()]
        # The cheap guards first; the name-rule test reads every impossible pair, so it runs last.
        if not _passes_merge_guards(all_a, all_b, tuple, block_signatures, repair, forbidden, seed_groups):
            continue
        row_groups, row_agree = own_agreement(sides[0], sides[1], a, b)
        column_groups, column_agree = own_agreement(sides[1], sides[0], b, a)
        rule = _AsideRule(
            aside=(spelled_by(sides[0], (a, b)), spelled_by(sides[1], (a, b))),
            plain=(~is_reading[sides[0]], ~is_reading[sides[1]]),
            groups=(row_groups, column_groups),
            agree=(row_agree, column_agree),
        )
        evidence = _pair_evidence(codes, width, sides, rule, close_code)
        if evidence is None or evidence.set_aside == 0:
            continue
        found = None
        # Take the lowest height over the smallest covers of the remaining conflicts, as the
        # pair repair does, so which cluster comes first cannot change the merge order.
        # An empty cover means no conflict remains.
        for cover in _smallest_covers(evidence.residual):
            odd = ([i for axis, i in cover if axis == 0], [j for axis, j in cover if axis == 1])
            bound = _reading_merge_bound(evidence, odd, sides, codes, block_signatures, repair, cut_height, close_code)
            if bound is not None:
                odd_mentions = tuple(sorted([int(sides[0][i]) for i in odd[0]] + [int(sides[1][j]) for j in odd[1]]))
                if found is None or (bound, odd_mentions) < found:
                    found = (bound, odd_mentions)
        if found is not None and _name_rules_explain_every_pair(codes, block_signatures, sides, repair):
            accepted.append((found[0], a, b, found[1]))
    if not accepted:
        return labels
    merged = label_array.copy()
    used: set[int] = set()
    for bound, a, b, odd in sorted(accepted):
        if a in used or b in used:
            continue
        used.update((a, b))
        merged[members(b)] = unique[a]
        logger.info(
            "Name-conflict repair: block %s merges clusters of %d and %d mentions over joined-initials "
            "readings, height %.3f, odd mentions %s",
            block_key,
            len(members(a)),
            len(members(b)),
            bound,
            ", ".join(block_signatures[m] for m in odd),
        )
    return merged


def repair_name_conflicts_in_block(
    block_signatures: Sequence[str],
    labels: np.ndarray | Sequence[int],
    codes: np.ndarray,
    repair: NameConflictRepair,
    *,
    forbidden: Mapping[str, Set[str]] | None = None,
    seed_groups: Mapping[str, object] | None = None,
    block_key: str = "",
    cut_height: float | None = None,
) -> np.ndarray | Sequence[int]:
    """Run up to ``repair.rounds`` repair rounds, stopping at the first that merges nothing.

    A round runs the reading step, then the odd-mention step, and each step merges a cluster
    at most once. A later round sees the merged clusters, so a person split three ways by
    different odd mentions can rejoin over two rounds. ``cut_height`` is the linkage cut
    that the reading step merges under; ``None`` skips the reading step.
    """
    current = labels
    for _ in range(repair.rounds):
        read = _reading_round(
            block_signatures,
            current,
            codes,
            repair,
            cut_height,
            forbidden=forbidden,
            seed_groups=seed_groups,
            block_key=block_key,
        )
        repaired = _repair_round(
            block_signatures, read, codes, repair, forbidden=forbidden, seed_groups=seed_groups, block_key=block_key
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
        repair_close_code(repair),
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
        if not _passes_merge_guards(
            all_a,
            all_b,
            lambda a=a, b=b, odd=odd: _impossible_pairs(codes, len(block_signatures), members[a], members[b], odd),
            block_signatures,
            repair,
            forbidden,
            seed_groups,
        ):
            continue
        accepted.append(_RepairCandidate(average, a, b, tuple(odd), code_sum, close_count))
    if not accepted:
        return labels

    # Closest pair first. A cluster takes part in one merge per block, so no
    # merge rests on a cluster that an earlier merge changed. The close-pair height
    # can fall below zero, where linkage would clamp it; the order keeps the
    # unclamped value, so the pair with more close evidence goes first.
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
