"""Post-linkage corrections: the one stage between a block's linkage cut and its labels.

Every route that clusters a block from a distance matrix runs the same sequence:

1. linkage cut at EPS, with the close-pair bonus (``s2and.model_pairwise``);
2. the contracts, which are exact and never weighed: required seed groups restored, then
   profile exclusions evicted (``Clusterer._cluster_one_block``);
3. the corrections in this module, which are heuristics shipped on judged evidence:
   name-conflict repair, then the year-gap split. The merge comes first, so it cannot undo a
   split. Subblocked Arrow prediction then joins final clusters that share an ORCID across
   subblocks (``s2and.orcid_join``).

The corrections read the same evidence, a one-byte copy of the block's model distances
(``s2and.pair_codes``) taken before linkage overwrites the matrix, and the same facts per
signature (``BlockCorrections``), gathered once per request from a Rust featurizer or an
``ANDData``. ``Clusterer.name_conflict_repair_max_distance`` turns the repair on and sets its
limit. ``Clusterer.orcid_join`` turns on the join across subblocks.
``Clusterer.year_gap_split`` turns the split on.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from s2and.arrow_inputs import arrow_string_array
from s2and.consts import LARGE_DISTANCE, YEAR_DIFF_CAP
from s2and.corrections.name_conflict_repair import (
    BONUS_MAX_PLAIN_AVERAGE,
    BONUS_MIN_CLOSE_SHARE,
    BONUS_MIN_SIDE_MENTIONS,
    MIN_BLOCK_SIZE,
    REPAIR_ROUNDS,
    NameConflictRepair,
    has_full_first_name,
    repair_name_conflicts_in_block,
)
from s2and.corrections.year_gap_split import SMALL_SURNAME_BLOCK, split_year_gaps, split_years
from s2and.feature_port import _get_rust_featurizer
from s2and.model_pairwise import FastCluster
from s2and.pair_codes import HARD_CODE, certain_close_code, quantize_distances
from s2and.rust_calls import get_constraints_matrix_indexed_rust
from s2and.text import (
    canonicalize_name_parts,
    given_name_for_constraint,
    is_joined_reading,
    normalize_orcid,
    spelled_reading_forms,
)

# A name-rule conflict gets this value in the repair's own constraint calls. An explicit seed
# disallow keeps its own value and takes precedence, so this value marks a conflict of the name
# rules on a pair that no disallow holds.
_NAME_RULE_CONFLICT = 2 * LARGE_DISTANCE


@dataclass(frozen=True)
class BlockCodes:
    """A block's one-byte pair codes and which corrections read them."""

    codes: np.ndarray
    repair: bool
    split: bool


BlockCodes.__module__ = "s2and.block_corrections"


@dataclass(frozen=True)
class BlockCorrections:
    """Settings and per-signature facts of the post-linkage corrections of one request.

    Attributes:
        repair: Name-conflict repair settings and facts; None when the repair is off.
        split_close_code: Largest pair code certainly under the close-pair distance, which
            bridges a year gap; None when the year-gap split is off.
        orcids: ORCID per signature id; empty when the clusterer suppresses ORCIDs.
        years: Paper year per signature id, for years above zero; empty with the split off.
        first_names: First word of a full first name per signature id, only for signatures of
            small surname blocks whose name count does not mark a common surname. The split
            bridges parts that share one.
    """

    repair: NameConflictRepair | None
    split_close_code: int | None
    orcids: Mapping[str, str] = field(default_factory=dict)
    years: Mapping[str, int] = field(default_factory=dict)
    first_names: Mapping[str, str] = field(default_factory=dict)

    def codes_for(self, block_signatures: Sequence[str], distances: np.ndarray, num_threads: int) -> BlockCodes | None:
        """Copy a block's distances to codes before linkage, when a correction will read them.

        Args:
            block_signatures: Signature ids in matrix order.
            distances: The block's condensed model distances.
            num_threads: Quantization threads. One keeps the copy off a pool that may be busy
                scoring the next block.

        Returns:
            None when no correction applies: the repair needs ``MIN_BLOCK_SIZE`` mentions and an
            impossible pair, the split a span of dated years over ``YEAR_DIFF_CAP``, and both a
            plain condensed matrix of model distances.
        """
        repair = self.repair is not None and len(block_signatures) >= MIN_BLOCK_SIZE
        split = self.split_close_code is not None and _spans_a_year_gap(block_signatures, self.years)
        if not (repair or split):
            return None
        require_hard = not split
        codes = quantize_distances(distances, num_threads=num_threads, require_hard=require_hard)
        if codes is None:
            return None
        # Codes quantized with require_hard hold an impossible pair. Without one, the repair skips the block.
        if not require_hard:
            repair = repair and bool(len(codes)) and int(codes.max()) == HARD_CODE
        return BlockCodes(codes, repair, split) if repair or split else None

    def apply(
        self,
        block_signatures: Sequence[str],
        labels: np.ndarray | Sequence[int],
        block_codes: BlockCodes | None,
        *,
        cut_height: float | None,
        forbidden: Mapping[str, Set[str]],
        repair_seed_groups: Mapping[str, Any] | None,
        split_seed_groups: Mapping[str, Any] | None,
        block_key: str = "",
    ) -> np.ndarray | Sequence[int]:
        """Apply the name-conflict repair, then split at year gaps.

        Args:
            block_signatures: Signature ids in matrix order.
            labels: Labels after the linkage cut and the contracts.
            block_codes: ``codes_for`` output; None returns ``labels``.
            cut_height: EPS of an average-linkage cut, under which the repair's reading step
                merges; None skips the reading step.
            forbidden: Signature ids that may not share a cluster with each key: seed
                disallows and profile exclusions, both ways.
            repair_seed_groups: Required seed groups the repair may not join, or None.
            split_seed_groups: Required seed groups the split never separates, or None.
            block_key: Block name for logs.
        """
        if block_codes is None:
            return labels
        if block_codes.repair and self.repair is not None:
            labels = repair_name_conflicts_in_block(
                block_signatures,
                labels,
                block_codes.codes,
                self.repair,
                forbidden=forbidden,
                seed_groups=repair_seed_groups,
                block_key=block_key,
                cut_height=cut_height,
            )
        if block_codes.split and self.split_close_code is not None:
            labels = split_year_gaps(
                block_signatures,
                labels,
                block_codes.codes,
                self.years,
                self.split_close_code,
                orcids=self.orcids,
                seed_groups=split_seed_groups,
                first_names=self.first_names,
                block_key=block_key,
            )
        return labels


BlockCorrections.__module__ = "s2and.block_corrections"


def orcid_join_enabled(clusterer: Any) -> bool:
    """Return whether the clusterer permits guarded ORCID joins."""
    return bool(getattr(clusterer, "orcid_join", True)) and not getattr(clusterer, "suppress_orcid", False)


def _spans_a_year_gap(block_signatures: Sequence[str], years: Mapping[str, int]) -> bool:
    """Return whether a block's dated mentions span more than ``YEAR_DIFF_CAP`` years.

    A shorter span has no empty stretch to cut, so the block skips the split and its code copy.
    """
    if not years:
        return False
    # Read only this block's years: a multi-block request holds every block's years.
    dated = list(split_years({s: years[s] for s in block_signatures if s in years}).values())
    return len(dated) >= 2 and max(dated) - min(dated) > YEAR_DIFF_CAP


def average_linkage_cut(cluster_model: Any, params: Mapping[str, Any] | None) -> float | None:
    """Return the EPS cut of an average-linkage FastCluster, else None.

    The repair's reading step merges two clusters under this cut, as linkage would have.
    """
    if not isinstance(cluster_model, FastCluster) or cluster_model.input_as_observation_matrix:
        return None
    settings = {**cluster_model.get_params(), **(params or {})}
    if settings.get("linkage") != "average" or settings.get("eps") is None:
        return None
    return float(settings["eps"])


def _close_pair_settings(clusterer: Any) -> tuple[float, float]:
    """Return the cluster model's close-pair bonus and distance; a zero bonus means none."""
    cluster_model = getattr(clusterer, "cluster_model", None)
    bonus = float(getattr(cluster_model, "close_pair_bonus", 0.0) or 0.0)
    return bonus, float(getattr(cluster_model, "close_pair_distance", 0.0)) if bonus > 0 else 0.0


def _split_close_code(clusterer: Any) -> int | None:
    """Return the split's close code, or None when the split is off.

    The split bridges a gap on a certain close pair, so it needs a cluster model with a
    close-pair bonus. A close-pair distance under every code gives -1: the split runs and no
    pair bridges a gap.
    """
    bonus, distance = _close_pair_settings(clusterer)
    if not getattr(clusterer, "year_gap_split", True) or bonus <= 0:
        return None
    return certain_close_code(distance)


def year_gap_split_enabled(clusterer: Any) -> bool:
    """Return whether prediction with ``clusterer`` runs the year-gap split."""
    return _split_close_code(clusterer) is not None


# Clusterer attributes that tuned the repair before 1.0.0b14. The repair now uses fixed settings,
# so a clusterer that still sets one would silently get another rule; it is refused instead.
_REMOVED_REPAIR_SETTINGS = (
    "name_conflict_repair_uses_close_pair_bonus",
    "name_conflict_repair_requires_full_first_names",
    "name_conflict_repair_rounds",
    "name_conflict_repair_initials_side_min_mentions",
    "name_conflict_repair_bonus_min_close_share",
    "name_conflict_repair_bonus_max_plain_average",
    "name_conflict_repair_bonus_min_side_mentions",
)


def _repair_settings(clusterer: Any) -> dict[str, Any] | None:
    """Return the repair's keyword settings, or None when the repair is off.

    Raises:
        ValueError: The clusterer sets a repair setting that 1.0.0b14 removed.
    """
    removed = [name for name in _REMOVED_REPAIR_SETTINGS if hasattr(clusterer, name)]
    if removed:
        raise ValueError(
            f"{', '.join(removed)} no longer exist: the repair uses fixed settings since 1.0.0b14 "
            "(see s2and.name_conflict_repair). Delete the attribute; set name_conflict_repair_max_distance "
            "to None to turn the repair off."
        )
    max_average_distance = getattr(clusterer, "name_conflict_repair_max_distance", None)
    if max_average_distance is None:
        return None
    settings: dict[str, Any] = {"max_average_distance": float(max_average_distance), "rounds": REPAIR_ROUNDS}
    bonus, distance = _close_pair_settings(clusterer)
    if bonus > 0:
        settings.update(
            close_pair_bonus=bonus,
            close_pair_distance=distance,
            bonus_min_close_share=BONUS_MIN_CLOSE_SHARE,
            bonus_max_plain_average=BONUS_MAX_PLAIN_AVERAGE,
            bonus_min_side_mentions=BONUS_MIN_SIDE_MENTIONS,
        )
    return settings


def _in_two_required_groups(require: Mapping[str, Any], left: str, right: str) -> bool:
    return left in require and right in require and require[left] != require[right]


def _rust_name_rules_alone_forbid(rust_featurizer: object) -> Callable[[Sequence[tuple[str, str]]], list[bool]]:
    """Return the repair's name-rule test, answered by the featurizer and the seeds it holds.

    A pair passes when its names conflict, no disallow the featurizer holds covers it, and its
    signatures are not in two different required groups. Two signatures of one required group
    pass: prediction may ignore the groups, and one group never keeps its own signatures apart.
    """
    featurizer = cast(Any, rust_featurizer)
    index_by_id: dict[str, int] = {}
    require: dict[str, Any] = {}

    def forbid(pairs: Sequence[tuple[str, str]]) -> list[bool]:
        if not pairs:
            return []
        if not index_by_id:
            index_by_id.update({str(s): i for i, s in enumerate(featurizer.signature_ids())})
            require.update({str(s): str(c) for s, c in featurizer.cluster_seeds_require()})
        values = get_constraints_matrix_indexed_rust(
            [(index_by_id[left], index_by_id[right]) for left, right in pairs],
            featurizer=rust_featurizer,
            high_value=_NAME_RULE_CONFLICT,
            dont_merge_cluster_seeds=True,
            # Required groups are left out of the call, which still applies explicit disallows.
            incremental_dont_use_cluster_seeds=True,
            # One thread: its pool is apart from the scoring pool unless scoring also runs on one thread.
            num_threads=1,
            # The question is about names; an equal ORCID must not hide a conflict.
            suppress_orcid=True,
        )
        return [
            value == _NAME_RULE_CONFLICT and not _in_two_required_groups(require, left, right)
            for value, (left, right) in zip(values, pairs, strict=True)
        ]

    return forbid


def _dataset_name_rules_alone_forbid(dataset: Any) -> Callable[[Sequence[tuple[str, str]]], list[bool]]:
    """Return the same test, answered by ``ANDData`` and its cluster seeds."""

    def forbid(pairs: Sequence[tuple[str, str]]) -> list[bool]:
        return [
            dataset.get_constraint(
                left,
                right,
                high_value=_NAME_RULE_CONFLICT,
                dont_merge_cluster_seeds=True,
                incremental_dont_use_cluster_seeds=True,
                suppress_orcid=True,
            )
            == _NAME_RULE_CONFLICT
            and not _in_two_required_groups(dataset.cluster_seeds_require, left, right)
            for left, right in pairs
        ]

    return forbid


def _common_surname(count: float | None) -> bool:
    """Return whether a last-name-and-first-initial name count marks the name as not rare.

    The counts come from the name-count index, not the request, so they catch a request that
    holds only part of a large block, such as Scholar's reclustering of excluded mentions. A
    count of ``SMALL_SURNAME_BLOCK`` or more rules out the first-name bridge; a few true small
    blocks have such counts and lose it. A missing count rules out nothing.
    """
    return count is not None and not math.isnan(count) and count >= SMALL_SURNAME_BLOCK


def _request_small_block_signatures(block_dict: Mapping[str, Sequence[str]]) -> frozenset[str]:
    """Return the signatures of request blocks under ``SMALL_SURNAME_BLOCK``, for callers that pass whole blocks."""
    return frozenset(str(s) for members in block_dict.values() if len(members) < SMALL_SURNAME_BLOCK for s in members)


def _count_small_blocks(
    candidates: Mapping[str, str | None], block_sizes: Mapping[str, int], request_small: Set[str]
) -> frozenset[str]:
    """Return the candidates whose surname block is small; one with no block key keeps the request rule."""
    return frozenset(
        signature_id
        for signature_id, block in candidates.items()
        if (block_sizes.get(block, 0) < SMALL_SURNAME_BLOCK if block is not None else signature_id in request_small)
    )


def dataset_small_block_signatures(block_dict: Mapping[str, Sequence[str]], dataset: Any) -> frozenset[str] | None:
    """Return the request signatures whose ``author_info_block`` has under ``SMALL_SURNAME_BLOCK`` dataset mentions.

    A claimed profile, an incremental residual or a subblock is smaller than its surname block,
    so the block size comes from the dataset. None when the dataset holds no signatures to count.
    """
    signatures = getattr(dataset, "signatures", None)
    if not signatures:
        return None
    request_small = _request_small_block_signatures(block_dict)
    if not request_small:
        return frozenset()
    candidates = {
        s: getattr(signatures[s], "author_info_block", None) if s in signatures else None for s in request_small
    }
    needed = {block for block in candidates.values() if block is not None}
    sizes: dict[str, int] = defaultdict(int)
    if needed:
        for signature in signatures.values():
            block = signature.author_info_block
            if block in needed:
                sizes[block] += 1
    return _count_small_blocks(candidates, sizes, request_small)


def arrow_small_block_signatures(arrow_lease: Any, block_dict: Mapping[str, Sequence[str]]) -> frozenset[str] | None:
    """Return the request signatures whose ``author_block`` has under ``SMALL_SURNAME_BLOCK`` rows in the dataset.

    Reads two signature columns, and only when a request block is that small. None when the
    signatures table has no ``author_block`` column.
    """
    request_small = _request_small_block_signatures(block_dict)
    if not request_small:
        return frozenset()
    import pyarrow as pa

    pc: Any = __import__("pyarrow.compute").compute
    candidates: dict[str, str | None] = {}
    sizes: dict[str, int] = defaultdict(int)
    with arrow_lease.open_file("signatures") as source_file, pa.PythonFile(source_file, mode="r") as source:
        reader = pa.ipc.open_file(source)
        if "author_block" not in reader.schema.names:
            return None
        id_index = reader.schema.get_field_index("signature_id")
        block_index = reader.schema.get_field_index("author_block")
        value_set = arrow_string_array(sorted(request_small))
        block_columns = []
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            block_column = batch.column(block_index)
            block_columns.append(block_column)
            mask = pc.is_in(batch.column(id_index), value_set=value_set)
            if bool(pc.any(mask).as_py()):
                for signature_id, block in zip(
                    pc.filter(batch.column(id_index), mask).to_pylist(),
                    pc.filter(block_column, mask).to_pylist(),
                    strict=True,
                ):
                    candidates[str(signature_id)] = block
        needed = sorted({block for block in candidates.values() if block is not None})
        if needed:
            needed_set = arrow_string_array(needed)
            for block_column in block_columns:
                kept = pc.filter(block_column, pc.is_in(block_column, value_set=needed_set))
                for entry in pc.value_counts(kept).to_pylist():
                    sizes[entry["values"]] += int(entry["counts"])
    for signature_id in request_small - candidates.keys():
        candidates[signature_id] = None
    return _count_small_blocks(candidates, sizes, request_small)


def _wanted(clusterer: Any, block_dict: Mapping[str, Sequence[str]]) -> tuple[dict[str, Any] | None, int | None] | None:
    """Return the repair settings and the split's close code, or None when no block can be corrected."""
    repair = _repair_settings(clusterer)
    split = _split_close_code(clusterer)
    largest = max(map(len, block_dict.values()), default=0)
    if largest < MIN_BLOCK_SIZE:
        repair = None
    if largest < 2 or (repair is None and split is None):
        return None
    return repair, split


def corrections_from_rust_featurizer(
    clusterer: Any,
    block_dict: Mapping[str, Sequence[str]],
    rust_featurizer: object,
    small_block_signatures: Set[str] | None = None,
) -> BlockCorrections | None:
    """Gather the corrections' facts from a Rust featurizer.

    Args:
        clusterer: Supplies the switches, the repair limit and the cluster model's close-pair bonus.
        block_dict: The blocks this call clusters.
        rust_featurizer: Featurizer over every signature of ``block_dict``.
        small_block_signatures: Signatures of small surname blocks, which may bridge a year gap
            on a shared first name; None treats each block of ``block_dict`` as a whole surname block.

    Returns:
        None when no correction is on or no block is large enough.
    """
    wanted = _wanted(clusterer, block_dict)
    if wanted is None:
        return None
    repair_settings, split_close_code = wanted
    featurizer = cast(Any, rust_featurizer)
    full_first_names, orcid_pairs = featurizer.name_conflict_facts()
    # A clusterer that hides ORCIDs from its constraints hides them from the corrections too.
    orcids = {} if getattr(clusterer, "suppress_orcid", False) else dict(orcid_pairs)
    name_conflicts = _rust_name_rules_alone_forbid(rust_featurizer) if repair_settings is not None else None
    repair = None
    if repair_settings is not None:
        readings, spellers = featurizer.joined_reading_facts()
        repair = NameConflictRepair(
            full_first_names=frozenset(full_first_names),
            orcids=orcids,
            name_rules_alone_forbid=name_conflicts,
            **_reading_fields(readings, spellers),
            **repair_settings,
        )
    years: dict[str, int] = {}
    first_names: dict[str, str] = {}
    if split_close_code is not None:
        years = dict(featurizer.signature_years())
        small = (
            _request_small_block_signatures(block_dict) if small_block_signatures is None else small_block_signatures
        )
        if small:
            first_names = {
                s: name
                for s, name, count in featurizer.signature_first_names()
                if s in small and not _common_surname(count)
            }
    return BlockCorrections(repair, split_close_code, orcids, years, first_names)


def corrections_from_dataset(
    clusterer: Any,
    block_dict: Mapping[str, Sequence[str]],
    dataset: Any,
    *,
    small_block_signatures: frozenset[str] | None = None,
) -> BlockCorrections | None:
    """Gather the same facts from a dataset's Rust featurizer, or from classic ``ANDData`` signatures.

    A stand-in dataset carries its featurizer. An Arrow-backed ``ANDData`` reads its cached
    featurizer, with the dataset's seeds, so it gets the facts production gets. Only an
    ``ANDData`` built without Arrow, which has no Rust featurizer, reads its signatures here.
    ``small_block_signatures`` overrides the small-surname set, for callers that pass subblocks
    of a larger block; otherwise the dataset or the request blocks decide it.
    """
    wanted = _wanted(clusterer, block_dict)
    if wanted is None:
        return None
    rust_featurizer = getattr(dataset, "rust_featurizer", None)
    if rust_featurizer is None and getattr(dataset, "arrow_dataset", None) is not None:
        # Prediction that built its own distances has already built and cached this featurizer.
        rust_featurizer = _get_rust_featurizer(dataset)
    repair_settings, split_close_code = wanted
    if small_block_signatures is None and split_close_code is not None:
        small_block_signatures = dataset_small_block_signatures(block_dict, dataset)
    if rust_featurizer is not None:
        return corrections_from_rust_featurizer(clusterer, block_dict, rust_featurizer, small_block_signatures)
    small = small_block_signatures
    if small is None and split_close_code is not None:
        small = _request_small_block_signatures(block_dict)
    use_orcid = not getattr(clusterer, "suppress_orcid", False)
    full_first_names: set[str] = set()
    orcids: dict[str, str] = {}
    values: dict[str, str | None] = {}
    years: dict[str, int] = {}
    first_names: dict[str, str] = {}
    for members in block_dict.values():
        for signature_id in members:
            signature = dataset.signatures[signature_id]
            if repair_settings is not None:
                values[signature_id] = given_name_for_constraint(
                    signature.author_info_first, signature.author_info_middle
                )
            first = signature.author_info_first_normalized
            if first is None:
                # A dataset built without preprocessing can lack normalized names; parse as the name
                # rules do, so "J. D." is the initial "j" here as it is in production.
                first = canonicalize_name_parts(
                    signature.author_info_first, signature.author_info_middle, signature.author_info_last
                ).first
            if repair_settings is not None and has_full_first_name(first):
                full_first_names.add(signature_id)
            orcid = normalize_orcid(signature.author_info_orcid) if use_orcid and signature.author_info_orcid else None
            if orcid is not None:
                orcids[signature_id] = orcid
            if split_close_code is None:
                continue
            first_word = (first or "").split(maxsplit=1)[:1]
            counts = getattr(signature, "author_info_name_counts", None)
            common = _common_surname(counts.last_first_initial if counts is not None else None)
            if small and signature_id in small and not common and first_word and len(first_word[0]) > 1:
                first_names[signature_id] = first_word[0]
            paper = dataset.papers.get(str(signature.paper_id))
            # A NaN year fails the comparison, so it stays undated.
            if paper is not None and paper.year is not None and paper.year > 0:
                years[signature_id] = int(paper.year)
    name_conflicts = _dataset_name_rules_alone_forbid(dataset) if repair_settings is not None else None
    repair = None
    if repair_settings is not None:
        # The same facts as the Rust featurizer's joined_reading_facts, for the block's signatures.
        readings = [(s, value) for s, value in values.items() if value is not None and is_joined_reading(value)]
        reading_values = {value for _, value in readings}
        spellers: list[tuple[str, str]] = []
        if reading_values:
            for signature_id, value in values.items():
                if is_joined_reading(value):
                    continue
                signature = dataset.signatures[signature_id]
                parts = canonicalize_name_parts(
                    signature.author_info_first, signature.author_info_middle, signature.author_info_last
                )
                spellers.extend(
                    (signature_id, form) for form in spelled_reading_forms(parts, value) if form in reading_values
                )
        repair = NameConflictRepair(
            full_first_names=frozenset(full_first_names),
            orcids=orcids,
            name_rules_alone_forbid=name_conflicts,
            **_reading_fields(readings, spellers),
            **repair_settings,
        )
    return BlockCorrections(repair, split_close_code, orcids, years, first_names)


def _reading_fields(readings: Sequence[tuple[str, str]], spellers: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Return the reading-step fields of ``NameConflictRepair`` from reading and speller pairs."""
    by_reading: dict[str, set[str]] = {}
    for signature_id, reading in spellers:
        by_reading.setdefault(reading, set()).add(signature_id)
    return {"joined_readings": dict(readings), "reading_spellers": by_reading}


def corrections_copy_codes(clusterer: Any) -> bool:
    """Return whether a block clustered by ``clusterer`` may keep a one-byte copy of its distances."""
    return _repair_settings(clusterer) is not None or _split_close_code(clusterer) is not None


def correction_fingerprint(clusterer: Any) -> tuple[Any, ...]:
    """Return the settings that change what the corrections do, for caches keyed on a clusterer."""
    return (_repair_settings(clusterer), _split_close_code(clusterer), orcid_join_enabled(clusterer))
