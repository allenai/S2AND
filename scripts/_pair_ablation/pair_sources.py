"""Canonical pair-source loading and deterministic sampling for ablations.

The ablation runner has several kinds of supervision (gold clusters, fixed
pair-only datasets, historical augmentation, and linker-derived pairs).  This
module gives all of them one deliberately small schema and keeps the fold unit
(``source_domain``) explicit on every row.

The within-block sampler treats the concatenated pair universes as a virtual
sequence.  It samples integer ranks and un-ranks only the selected pairs, so it
uses memory proportional to the number of signatures plus the requested sample
instead of the number of possible pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import gcd
from numbers import Integral, Real
from pathlib import Path
from typing import Literal, cast

import pandas as pd

PAIR_COLUMNS = (
    "source_domain",
    "source_family",
    "pair1",
    "pair2",
    "label",
    "label_rule",
    "origin",
    "group_id",
)
"""Column order for every pair catalog used by the ablation study."""

_PAIR_KEY_COLUMNS = ("source_domain", "pair1", "pair2")
_DATASET_PREFIX_SEPARATOR = "___"


def _nonempty_text(value: object, field: str) -> str:
    if value is None:
        raise ValueError(f"{field} must be non-empty")
    if not isinstance(value, str):
        try:
            is_missing = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_missing = False
        if is_missing:
            raise ValueError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _binary_label(value: object) -> int:
    if isinstance(value, str):
        token = value.strip().upper()
        labels = {"YES": 1, "NO": 0, "1": 1, "0": 0}
        if token in labels:
            return labels[token]
    elif isinstance(value, Integral):
        label = int(value)
        if label in (0, 1):
            return label
    elif isinstance(value, Real):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return int(numeric)
    raise ValueError(f"label must be one of YES, NO, 0, or 1; got {value!r}")


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One canonical labeled pair with its fold and labeling provenance."""

    source_domain: str
    source_family: str
    pair1: str
    pair2: str
    label: int
    label_rule: str
    origin: str
    group_id: str

    def __post_init__(self) -> None:
        source_domain = _nonempty_text(self.source_domain, "source_domain")
        if _DATASET_PREFIX_SEPARATOR in source_domain:
            raise ValueError(f"source_domain must not contain {_DATASET_PREFIX_SEPARATOR!r}: {source_domain!r}")
        source_family = _nonempty_text(self.source_family, "source_family")
        pair1 = _nonempty_text(self.pair1, "pair1")
        pair2 = _nonempty_text(self.pair2, "pair2")
        if pair1 == pair2:
            raise ValueError(f"self-pairs are not valid training examples: {pair1!r}")
        pair1, pair2 = sorted((pair1, pair2))

        object.__setattr__(self, "source_domain", source_domain)
        object.__setattr__(self, "source_family", source_family)
        object.__setattr__(self, "pair1", pair1)
        object.__setattr__(self, "pair2", pair2)
        object.__setattr__(self, "label", _binary_label(self.label))
        object.__setattr__(self, "label_rule", _nonempty_text(self.label_rule, "label_rule"))
        object.__setattr__(self, "origin", _nonempty_text(self.origin, "origin"))
        object.__setattr__(self, "group_id", _nonempty_text(self.group_id, "group_id"))

    def as_dict(self) -> dict[str, str | int]:
        """Return this record in canonical frame-column order."""

        return {column: getattr(self, column) for column in PAIR_COLUMNS}


def empty_pair_frame() -> pd.DataFrame:
    """Return an empty frame with the canonical columns and stable dtypes."""

    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in PAIR_COLUMNS})
    frame["label"] = frame["label"].astype("int8")
    return frame


def _pair_frame(records: Sequence[PairRecord]) -> pd.DataFrame:
    if not records:
        return empty_pair_frame()
    frame = pd.DataFrame.from_records([record.as_dict() for record in records], columns=PAIR_COLUMNS)
    frame["label"] = frame["label"].astype("int8")
    return frame


def canonicalize_pairs(pairs: pd.DataFrame | Iterable[PairRecord | Mapping[str, object]]) -> pd.DataFrame:
    """Validate, orient, de-duplicate, and conflict-check labeled pairs.

    Pair identity is the unordered pair within ``source_domain``.  Repeated
    rows with the same label retain their first provenance row; a repeated pair
    with both labels is rejected rather than allowing input order to decide the
    target.

    Args:
        pairs: Canonical-schema rows or a frame with exactly ``PAIR_COLUMNS``.

    Returns:
        A fresh canonical frame in input order after de-duplication.

    Raises:
        ValueError: If the schema is wrong or any pair has conflicting labels.
    """

    if isinstance(pairs, pd.DataFrame):
        missing = sorted(set(PAIR_COLUMNS) - set(pairs.columns))
        extra = sorted(set(pairs.columns) - set(PAIR_COLUMNS))
        if missing or extra:
            raise ValueError(f"pair frame schema mismatch: missing={missing}, extra={extra}")
        if pairs.empty:
            return empty_pair_frame()
        frame = pairs.loc[:, PAIR_COLUMNS].copy()
        text_columns = tuple(column for column in PAIR_COLUMNS if column != "label")
        for column in text_columns:
            if bool(frame[column].isna().any()):
                raise ValueError(f"{column} must be non-empty")
            frame[column] = frame[column].astype(str).str.strip()
            if bool(frame[column].eq("").any()):
                raise ValueError(f"{column} must be non-empty")
        if bool(frame["source_domain"].str.contains(_DATASET_PREFIX_SEPARATOR, regex=False).any()):
            invalid = str(
                frame.loc[
                    frame["source_domain"].str.contains(_DATASET_PREFIX_SEPARATOR, regex=False),
                    "source_domain",
                ].iloc[0]
            )
            raise ValueError(f"source_domain must not contain {_DATASET_PREFIX_SEPARATOR!r}: {invalid!r}")
        frame["label"] = frame["label"].map(_binary_label).astype("int8")

        left = frame["pair1"].copy()
        right = frame["pair2"].copy()
        if bool(left.eq(right).any()):
            example = str(left.loc[left.eq(right)].iloc[0])
            raise ValueError(f"self-pairs are not valid training examples: {example!r}")
        swap = left.gt(right)
        frame.loc[swap, "pair1"] = right.loc[swap]
        frame.loc[swap, "pair2"] = left.loc[swap]

        duplicate_rows = frame.loc[frame.duplicated(list(_PAIR_KEY_COLUMNS), keep=False)]
        if not duplicate_rows.empty:
            conflicting = duplicate_rows.groupby(list(_PAIR_KEY_COLUMNS), sort=False)["label"].nunique().gt(1)
            if bool(conflicting.any()):
                source_domain, pair1, pair2 = conflicting.index[conflicting][0]
                raise ValueError(
                    "Conflicting labels for " f"source_domain={source_domain!r}, pair=({pair1!r}, {pair2!r})"
                )
        return frame.drop_duplicates(list(_PAIR_KEY_COLUMNS), keep="first").reset_index(drop=True)

    raw_rows: Iterable[PairRecord | Mapping[str, object]] = pairs

    output: list[PairRecord] = []
    seen: dict[tuple[str, str, str], PairRecord] = {}
    for row_number, raw_row in enumerate(raw_rows, start=1):
        if isinstance(raw_row, PairRecord):
            record = raw_row
        elif isinstance(raw_row, Mapping):
            missing = sorted(set(PAIR_COLUMNS) - set(raw_row))
            extra = sorted(set(raw_row) - set(PAIR_COLUMNS))
            if missing or extra:
                raise ValueError(f"pair record {row_number} schema mismatch: missing={missing}, extra={extra}")
            record = PairRecord(**{column: raw_row[column] for column in PAIR_COLUMNS})
        else:
            raise TypeError(f"pair record {row_number} must be PairRecord or Mapping, got {type(raw_row)!r}")

        key = tuple(getattr(record, column) for column in _PAIR_KEY_COLUMNS)
        previous = seen.get(key)
        if previous is None:
            seen[key] = record
            output.append(record)
        elif previous.label != record.label:
            raise ValueError(
                "Conflicting labels for "
                f"source_domain={record.source_domain!r}, pair=({record.pair1!r}, {record.pair2!r}): "
                f"{previous.label} from {previous.origin!r} vs {record.label} from {record.origin!r}"
            )

    return _pair_frame(output)


def concat_pair_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate catalogs and apply cross-catalog de-duplication checks."""

    materialized = list(frames)
    if not materialized:
        return empty_pair_frame()
    return canonicalize_pairs(pd.concat(materialized, ignore_index=True))


def exclude_source_domains(pair_frame: pd.DataFrame, held_out_domains: str | Iterable[str]) -> pd.DataFrame:
    """Remove every row belonging to one or more held-out fold domains."""

    canonical = canonicalize_pairs(pair_frame)
    if isinstance(held_out_domains, str):
        excluded = {_nonempty_text(held_out_domains, "held_out_domain")}
    else:
        excluded = {_nonempty_text(domain, "held_out_domain") for domain in held_out_domains}
    result = canonical.loc[~canonical["source_domain"].isin(excluded), PAIR_COLUMNS]
    return result.reset_index(drop=True)


def _stable_seed(random_seed: int, *parts: str) -> int:
    payload = "\0".join((str(int(random_seed)), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def cap_pairs_per_domain(
    pair_frame: pd.DataFrame,
    cap: int | Mapping[str, int],
    *,
    random_seed: int,
    sampling: Literal["pair_uniform", "query_uniform"] = "pair_uniform",
) -> pd.DataFrame:
    """Apply a deterministic independent cap to every source domain.

    ``pair_uniform`` samples directly from the domain's rows.  The
    ``query_uniform`` policy treats ``group_id`` as the query identifier and
    takes shuffled round-robin passes through groups, preventing large queries
    from consuming the cap before smaller queries contribute.
    """

    if sampling not in {"pair_uniform", "query_uniform"}:
        raise ValueError(f"Unknown sampling policy: {sampling!r}")
    canonical = canonicalize_pairs(pair_frame)
    domains = sorted(str(domain) for domain in canonical["source_domain"].unique())

    def domain_cap(domain: str) -> int:
        if isinstance(cap, Mapping):
            cap_by_domain = cast(Mapping[str, int], cap)
            if domain not in cap_by_domain:
                raise ValueError(f"Missing pair cap for source_domain={domain!r}")
            value = int(cap_by_domain[domain])
        else:
            value = int(cap)
        if value < 0:
            raise ValueError(f"Pair cap must be non-negative for source_domain={domain!r}; got {value}")
        return value

    selected_frames: list[pd.DataFrame] = []
    sort_columns = ["pair1", "pair2", "label", "source_family", "label_rule", "origin", "group_id"]
    for domain in domains:
        domain_frame = canonical.loc[canonical["source_domain"] == domain, PAIR_COLUMNS]
        domain_frame = domain_frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        limit = min(domain_cap(domain), len(domain_frame))
        if limit == 0:
            continue
        rng = random.Random(_stable_seed(random_seed, domain, sampling))

        if sampling == "pair_uniform":
            selected_indices = rng.sample(range(len(domain_frame)), limit)
        else:
            grouped_indices: dict[str, list[int]] = defaultdict(list)
            for index, group_id in enumerate(domain_frame["group_id"]):
                grouped_indices[str(group_id)].append(index)
            group_ids = sorted(grouped_indices)
            rng.shuffle(group_ids)
            for group_id in group_ids:
                rng.shuffle(grouped_indices[group_id])

            selected_indices = []
            depth = 0
            while len(selected_indices) < limit:
                added = False
                for group_id in group_ids:
                    group_indices = grouped_indices[group_id]
                    if depth < len(group_indices):
                        selected_indices.append(group_indices[depth])
                        added = True
                        if len(selected_indices) == limit:
                            break
                if not added:
                    raise AssertionError("query-uniform sampling exhausted rows before reaching its cap")
                depth += 1

        selected_frames.append(domain_frame.iloc[selected_indices].loc[:, PAIR_COLUMNS])

    if not selected_frames:
        return empty_pair_frame()
    return canonicalize_pairs(pd.concat(selected_frames, ignore_index=True))


def _parse_prefixed_identifier(
    value: object,
    *,
    default_domain: str | None,
    require_prefix: bool,
) -> tuple[str, str]:
    raw = _nonempty_text(value, "pair identifier")
    prefix, separator, identifier = raw.partition(_DATASET_PREFIX_SEPARATOR)
    if separator:
        domain = _nonempty_text(prefix, "pair identifier dataset prefix")
        identifier = _nonempty_text(identifier, "stripped pair identifier")
        if default_domain is not None and domain != default_domain:
            raise ValueError(
                f"pair identifier {raw!r} belongs to {domain!r}, expected source_domain={default_domain!r}"
            )
        return domain, identifier
    if require_prefix:
        raise ValueError(f"pair identifier is missing a dataset___ prefix: {raw!r}")
    if default_domain is None:
        raise ValueError(f"Cannot infer source_domain from unprefixed pair identifier: {raw!r}")
    return default_domain, raw


def _load_fixed_pair_csv(
    path: Path,
    *,
    split: str,
    source_domain: str | None,
    source_family: str,
    label_rule: str,
    require_prefixed_domains: bool,
) -> list[PairRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "pairs1" in frame.columns:
        if "pair1" in frame.columns:
            raise ValueError(f"{path} contains both pair1 and the historical pairs1 typo")
        frame = frame.rename(columns={"pairs1": "pair1"})
    required = {"pair1", "pair2", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing fixed-pair columns: {missing}")

    expected_domain = None if source_domain is None else _nonempty_text(source_domain, "source_domain")
    records: list[PairRecord] = []
    for position, row in enumerate(frame.loc[:, ["pair1", "pair2", "label"]].itertuples(index=False), start=2):
        try:
            domain1, pair1 = _parse_prefixed_identifier(
                row.pair1,
                default_domain=expected_domain,
                require_prefix=require_prefixed_domains,
            )
            domain2, pair2 = _parse_prefixed_identifier(
                row.pair2,
                default_domain=expected_domain,
                require_prefix=require_prefixed_domains,
            )
            if domain1 != domain2:
                raise ValueError(f"pair endpoints belong to different domains: {domain1!r} vs {domain2!r}")
            records.append(
                PairRecord(
                    source_domain=domain1,
                    source_family=source_family,
                    pair1=pair1,
                    pair2=pair2,
                    label=_binary_label(row.label),
                    label_rule=label_rule,
                    origin=f"{path.as_posix()}:{position}",
                    group_id=f"{domain1}:{split}:{position - 2}",
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{position}: {exc}") from exc
    return records


def load_fixed_pair_csvs(
    csv_paths: Mapping[str, str | Path],
    *,
    source_domain: str | None,
    source_family: str,
    label_rule: str,
    require_prefixed_domains: bool = False,
) -> pd.DataFrame:
    """Load split-name-to-CSV mappings into the canonical pair schema."""

    if not csv_paths:
        raise ValueError("At least one fixed-pair CSV is required")
    records: list[PairRecord] = []
    for split, raw_path in csv_paths.items():
        split_name = _nonempty_text(split, "split")
        records.extend(
            _load_fixed_pair_csv(
                Path(raw_path),
                split=split_name,
                source_domain=source_domain,
                source_family=_nonempty_text(source_family, "source_family"),
                label_rule=_nonempty_text(label_rule, "label_rule"),
                require_prefixed_domains=require_prefixed_domains,
            )
        )
    return canonicalize_pairs(records)


def load_medline_pairs(
    dataset_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
) -> pd.DataFrame:
    """Load Medline's fixed pair-only supervision."""

    directory = Path(dataset_dir)
    paths = {split: directory / f"{split}_pairs.csv" for split in splits}
    return load_fixed_pair_csvs(
        paths,
        source_domain="medline",
        source_family="pairwise_only",
        label_rule="fixed_pair_label",
    )


def load_historical_augmented_pairs(
    dataset_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
) -> pd.DataFrame:
    """Load historical augmented CSVs and recover each row's source domain.

    The historical test file misspelled its first header as ``pairs1``.  Both
    endpoints must carry the same ``dataset___`` prefix; the prefix is removed
    from the canonical pair identifiers and retained as ``source_domain``.
    """

    directory = Path(dataset_dir)
    paths = {split: directory / f"{split}_pairs.csv" for split in splits}
    return load_fixed_pair_csvs(
        paths,
        source_domain=None,
        source_family="historical_augmented",
        label_rule="historical_augmentation_label",
        require_prefixed_domains=True,
    )


def build_gold_cluster_lookup(clusters: Mapping[object, object]) -> dict[str, str]:
    """Build ``signature_id -> gold_cluster_id`` from supported cluster shapes.

    Supported inputs are the S2AND JSON shape (cluster records containing
    ``signature_ids``), a mapping of cluster ids to signature-id sequences, or
    an already-inverted scalar mapping of signature ids to cluster ids.
    """

    if not isinstance(clusters, Mapping):
        raise TypeError(f"clusters must be a mapping, got {type(clusters)!r}")
    values = list(clusters.values())
    if not values:
        return {}

    records_shape = all(isinstance(value, Mapping) and "signature_ids" in value for value in values)
    sequences_shape = all(
        isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray) for value in values
    )
    scalar_shape = all(
        not isinstance(value, Mapping)
        and (not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray))
        for value in values
    )
    if sum((records_shape, sequences_shape, scalar_shape)) != 1:
        raise ValueError("clusters has a mixed or unsupported shape")

    lookup: dict[str, str] = {}
    raw_signature_ids: dict[str, object] = {}

    def add(signature_id: object, cluster_id: object) -> None:
        signature = _nonempty_text(signature_id, "signature_id")
        cluster = _nonempty_text(cluster_id, "cluster_id")
        if signature in raw_signature_ids and raw_signature_ids[signature] != signature_id:
            raise ValueError(
                "signature ids collide after string normalization: "
                f"{raw_signature_ids[signature]!r}, {signature_id!r}"
            )
        if signature in lookup:
            raise ValueError(
                f"signature_id={signature!r} appears in multiple gold clusters: {lookup[signature]!r}, {cluster!r}"
            )
        raw_signature_ids[signature] = signature_id
        lookup[signature] = cluster

    if scalar_shape:
        for signature_id, cluster_id in clusters.items():
            add(signature_id, cluster_id)
        return lookup

    for outer_cluster_id, raw_cluster in clusters.items():
        cluster_id = _nonempty_text(outer_cluster_id, "cluster_id")
        if records_shape:
            assert isinstance(raw_cluster, Mapping)
            cluster_record = cast(Mapping[object, object], raw_cluster)
            if "cluster_id" in cluster_record:
                embedded_cluster_id = _nonempty_text(cluster_record["cluster_id"], "cluster_id")
                if embedded_cluster_id != cluster_id:
                    raise ValueError(
                        f"cluster key {cluster_id!r} disagrees with embedded cluster_id={embedded_cluster_id!r}"
                    )
            signature_ids = cluster_record["signature_ids"]
            if not isinstance(signature_ids, Sequence) or isinstance(signature_ids, str | bytes | bytearray):
                raise ValueError(f"cluster_id={cluster_id!r} signature_ids must be a sequence")
        else:
            assert isinstance(raw_cluster, Sequence)
            signature_ids = raw_cluster
        for signature_id in signature_ids:
            add(signature_id, cluster_id)
    return lookup


def load_gold_cluster_lookup(path_or_clusters: str | Path | Mapping[object, object]) -> dict[str, str]:
    """Read a cluster JSON file, or invert an in-memory cluster mapping."""

    if isinstance(path_or_clusters, Mapping):
        return build_gold_cluster_lookup(cast(Mapping[object, object], path_or_clusters))
    path = Path(path_or_clusters)
    with path.open("r", encoding="utf-8") as stream:
        clusters: object = json.load(stream)
    if not isinstance(clusters, Mapping):
        raise ValueError(f"{path} must contain a JSON object of gold clusters")
    return build_gold_cluster_lookup(cast(Mapping[object, object], clusters))


def sample_pair_ranks(total_pairs: int, sample_size: int, random_seed: int) -> list[int]:
    """Sample virtual-universe ranks with ``random.sample`` semantics."""

    if total_pairs < 0:
        raise ValueError(f"total_pairs must be non-negative; got {total_pairs}")
    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative; got {sample_size}")
    return random.Random(int(random_seed)).sample(range(total_pairs), min(total_pairs, sample_size))


def unrank_pair(block_size: int, rank: int) -> tuple[int, int]:
    """Map a rank to the lexicographically enumerated ``(i, j)`` with ``i < j``."""

    if block_size < 2:
        raise ValueError(f"block_size must be at least 2; got {block_size}")
    total = block_size * (block_size - 1) // 2
    if rank < 0 or rank >= total:
        raise IndexError(f"pair rank {rank} outside [0, {total}) for block_size={block_size}")

    def pairs_before(row: int) -> int:
        return row * (2 * block_size - row - 1) // 2

    low = 0
    high = block_size - 2
    while low < high:
        middle = (low + high + 1) // 2
        if pairs_before(middle) <= rank:
            low = middle
        else:
            high = middle - 1
    first = low
    second = first + 1 + (rank - pairs_before(first))
    return first, second


def _validated_blocks(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster: Mapping[str, str],
) -> list[tuple[str, tuple[str, ...]]]:
    cluster_lookup = {str(signature): str(cluster) for signature, cluster in signature_to_cluster.items()}
    validated: list[tuple[str, tuple[str, ...]]] = []
    seen_signatures: set[str] = set()
    for raw_block_id, raw_signatures in blocks.items():
        block_id = _nonempty_text(raw_block_id, "block_id")
        if isinstance(raw_signatures, str | bytes | bytearray):
            raise ValueError(f"block_id={block_id!r} signatures must be a sequence, not a string")
        signatures = tuple(_nonempty_text(signature, "signature_id") for signature in raw_signatures)
        for signature in signatures:
            if signature in seen_signatures:
                raise ValueError(f"signature_id={signature!r} appears more than once across blocks")
            if signature not in cluster_lookup:
                raise ValueError(f"signature_id={signature!r} is missing a gold cluster label")
            seen_signatures.add(signature)
        validated.append((block_id, signatures))
    return validated


def sample_within_blocks_uniform(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster: Mapping[str, str],
    sample_size: int,
    *,
    random_seed: int,
    source_domain: str,
    source_family: str = "gold_cluster_uniform",
) -> pd.DataFrame:
    """Uniformly sample labeled within-block pairs from a virtual universe.

    Mapping insertion order and each block's signature order define the virtual
    universe, matching the historical nested-loop enumeration order.
    """

    domain = _nonempty_text(source_domain, "source_domain")
    validated_blocks = _validated_blocks(blocks, signature_to_cluster)
    cluster_lookup = {str(signature): str(cluster) for signature, cluster in signature_to_cluster.items()}

    cumulative_pair_counts: list[int] = []
    total_pairs = 0
    for _, signatures in validated_blocks:
        total_pairs += len(signatures) * (len(signatures) - 1) // 2
        cumulative_pair_counts.append(total_pairs)

    records: list[PairRecord] = []
    for global_rank in sample_pair_ranks(total_pairs, sample_size, random_seed):
        block_index = bisect_right(cumulative_pair_counts, global_rank)
        prior_pairs = 0 if block_index == 0 else cumulative_pair_counts[block_index - 1]
        block_id, signatures = validated_blocks[block_index]
        first_index, second_index = unrank_pair(len(signatures), global_rank - prior_pairs)
        pair1 = signatures[first_index]
        pair2 = signatures[second_index]
        label = int(cluster_lookup[pair1] == cluster_lookup[pair2])
        records.append(
            PairRecord(
                source_domain=domain,
                source_family=source_family,
                pair1=pair1,
                pair2=pair2,
                label=label,
                label_rule="same_gold_cluster" if label else "different_gold_cluster",
                origin=f"gold_clusters:{domain}",
                group_id=block_id,
            )
        )
    return canonicalize_pairs(records)


@dataclass(frozen=True, slots=True)
class _AnchorState:
    block_id: str
    signatures: tuple[str, ...]
    anchor_index: int
    partner_start: int
    partner_step: int


def _coprime_step(modulus: int, rng: random.Random) -> int:
    """Choose a bounded-search stride that permutes ``range(modulus)``."""

    if modulus <= 0:
        raise ValueError(f"modulus must be positive; got {modulus}")
    if modulus == 1:
        return 1
    candidate = rng.randrange(1, modulus)
    for _ in range(modulus - 1):
        if gcd(candidate, modulus) == 1:
            return candidate
        candidate = candidate % (modulus - 1) + 1
    raise AssertionError(f"No coprime stride found for modulus={modulus}")


def sample_within_blocks_anchor_uniform(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster: Mapping[str, str],
    sample_size: int,
    *,
    random_seed: int,
    source_domain: str,
    source_family: str = "gold_cluster_anchor_uniform",
) -> pd.DataFrame:
    """Sample gold pairs with approximately uniform signature exposure.

    Eligible signatures are deterministically shuffled, then visited once per
    round.  Each anchor walks a seed-specific permutation of every other
    signature in its block.  Reciprocal proposals are de-duplicated, so every
    unordered within-block pair can be reached without constructing the pair
    universe in memory.  The sampler returns exactly
    ``min(sample_size, total_within_block_pairs)`` rows and makes at most two
    proposals per possible pair before exhausting the universe.

    Mapping insertion order and each block's signature order are part of the
    deterministic input.  Unlike pair-uniform sampling, a large block does not
    receive ``C(n, 2)`` weight at the start of the sample: active anchors get
    one proposal per round, with the shuffled order resolving partial rounds.
    """

    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative; got {sample_size}")
    domain = _nonempty_text(source_domain, "source_domain")
    family = _nonempty_text(source_family, "source_family")
    validated_blocks = _validated_blocks(blocks, signature_to_cluster)
    cluster_lookup = {str(signature): str(cluster) for signature, cluster in signature_to_cluster.items()}

    total_pairs = sum(len(signatures) * (len(signatures) - 1) // 2 for _, signatures in validated_blocks)
    target_size = min(sample_size, total_pairs)
    if target_size == 0:
        return empty_pair_frame()

    anchors: list[_AnchorState] = []
    max_partner_count = 0
    for block_id, signatures in validated_blocks:
        partner_count = len(signatures) - 1
        if partner_count <= 0:
            continue
        max_partner_count = max(max_partner_count, partner_count)
        for anchor_index, anchor in enumerate(signatures):
            rng = random.Random(_stable_seed(random_seed, domain, "anchor_partner", block_id, anchor))
            anchors.append(
                _AnchorState(
                    block_id=block_id,
                    signatures=signatures,
                    anchor_index=anchor_index,
                    partner_start=rng.randrange(partner_count),
                    partner_step=_coprime_step(partner_count, rng),
                )
            )

    random.Random(_stable_seed(random_seed, domain, "anchor_order")).shuffle(anchors)
    records: list[PairRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for round_index in range(max_partner_count):
        for state in anchors:
            partner_count = len(state.signatures) - 1
            if round_index >= partner_count:
                continue
            partner_slot = (state.partner_start + round_index * state.partner_step) % partner_count
            partner_index = partner_slot if partner_slot < state.anchor_index else partner_slot + 1
            anchor = state.signatures[state.anchor_index]
            partner = state.signatures[partner_index]
            pair_key = (min(anchor, partner), max(anchor, partner))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            label = int(cluster_lookup[anchor] == cluster_lookup[partner])
            records.append(
                PairRecord(
                    source_domain=domain,
                    source_family=family,
                    pair1=anchor,
                    pair2=partner,
                    label=label,
                    label_rule="same_gold_cluster" if label else "different_gold_cluster",
                    origin=f"gold_clusters_anchor_uniform:{domain}",
                    group_id=state.block_id,
                )
            )
            if len(records) == target_size:
                return canonicalize_pairs(records)

    raise AssertionError(
        f"anchor-uniform sampling exhausted {len(records)} unique pairs before target_size={target_size}"
    )


@dataclass(frozen=True, slots=True)
class _NegativeBlockIndex:
    block_id: str
    signatures: tuple[str, ...]
    cluster_ids: tuple[str, ...]
    positions_by_cluster: Mapping[str, tuple[int, ...]]
    cumulative_row_counts: tuple[int, ...]


def _unrank_negative_pair(index: _NegativeBlockIndex, rank: int) -> tuple[str, str]:
    """Unrank a different-cluster pair from one block's compact index."""

    total = index.cumulative_row_counts[-1]
    if rank < 0 or rank >= total:
        raise IndexError(f"negative pair rank {rank} outside [0, {total}) for block_id={index.block_id!r}")
    first_index = bisect_right(index.cumulative_row_counts, rank)
    prior_count = 0 if first_index == 0 else index.cumulative_row_counts[first_index - 1]
    partner_rank = rank - prior_count

    signatures = index.signatures
    same_cluster_positions = index.positions_by_cluster[index.cluster_ids[first_index]]
    same_cluster_after_anchor = bisect_right(same_cluster_positions, first_index)
    low = first_index + 1
    high = len(signatures) - 1
    while low < high:
        middle = (low + high) // 2
        excluded_count = bisect_right(same_cluster_positions, middle) - same_cluster_after_anchor
        eligible_count = middle - first_index - excluded_count
        if eligible_count > partner_rank:
            high = middle
        else:
            low = middle + 1
    return signatures[first_index], signatures[low]


def sample_within_blocks_balanced(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster: Mapping[str, str],
    *,
    positive_size: int,
    negative_size: int,
    random_seed: int,
    source_domain: str,
    source_family: str = "gold_cluster_balanced",
) -> pd.DataFrame:
    """Uniformly sample separate positive and negative gold-pair quotas.

    Each stratum is sampled uniformly without replacement from its own virtual
    universe.  The returned count for a stratum is its requested size when
    enough pairs exist, otherwise every available pair in that stratum is
    returned.  Positive pairs are indexed by gold cluster; negative pairs use
    a linear-size per-block row index, avoiding enumeration of cross-cluster
    pairs even when every signature belongs to a different cluster.
    """

    if positive_size < 0 or negative_size < 0:
        raise ValueError("positive_size and negative_size must be non-negative")
    domain = _nonempty_text(source_domain, "source_domain")
    family = _nonempty_text(source_family, "source_family")
    validated_blocks = _validated_blocks(blocks, signature_to_cluster)
    cluster_lookup = {str(signature): str(cluster) for signature, cluster in signature_to_cluster.items()}

    positive_segments: list[tuple[str, tuple[str, ...]]] = []
    cumulative_positive_counts: list[int] = []
    total_positive = 0
    negative_blocks: list[_NegativeBlockIndex] = []
    cumulative_negative_counts: list[int] = []
    total_negative = 0

    for block_id, signatures in validated_blocks:
        signatures_by_cluster: dict[str, list[str]] = defaultdict(list)
        positions_by_cluster_lists: dict[str, list[int]] = defaultdict(list)
        for position, signature in enumerate(signatures):
            cluster = cluster_lookup[signature]
            signatures_by_cluster[cluster].append(signature)
            positions_by_cluster_lists[cluster].append(position)

        for members in signatures_by_cluster.values():
            segment_size = len(members) * (len(members) - 1) // 2
            if segment_size == 0:
                continue
            total_positive += segment_size
            positive_segments.append((block_id, tuple(members)))
            cumulative_positive_counts.append(total_positive)

        positions_by_cluster = {cluster: tuple(positions) for cluster, positions in positions_by_cluster_lists.items()}
        cumulative_row_counts: list[int] = []
        block_negative = 0
        for position, signature in enumerate(signatures):
            same_cluster_positions = positions_by_cluster[cluster_lookup[signature]]
            same_cluster_after = len(same_cluster_positions) - bisect_right(same_cluster_positions, position)
            block_negative += len(signatures) - position - 1 - same_cluster_after
            cumulative_row_counts.append(block_negative)
        if block_negative:
            total_negative += block_negative
            negative_blocks.append(
                _NegativeBlockIndex(
                    block_id=block_id,
                    signatures=signatures,
                    cluster_ids=tuple(cluster_lookup[signature] for signature in signatures),
                    positions_by_cluster=positions_by_cluster,
                    cumulative_row_counts=tuple(cumulative_row_counts),
                )
            )
            cumulative_negative_counts.append(total_negative)

    records: list[PairRecord] = []
    positive_seed = _stable_seed(random_seed, domain, "balanced_positive")
    for global_rank in sample_pair_ranks(total_positive, positive_size, positive_seed):
        segment_index = bisect_right(cumulative_positive_counts, global_rank)
        prior_count = 0 if segment_index == 0 else cumulative_positive_counts[segment_index - 1]
        block_id, members = positive_segments[segment_index]
        first_index, second_index = unrank_pair(len(members), global_rank - prior_count)
        records.append(
            PairRecord(
                source_domain=domain,
                source_family=family,
                pair1=members[first_index],
                pair2=members[second_index],
                label=1,
                label_rule="same_gold_cluster",
                origin=f"gold_clusters_balanced:{domain}",
                group_id=block_id,
            )
        )

    negative_seed = _stable_seed(random_seed, domain, "balanced_negative")
    for global_rank in sample_pair_ranks(total_negative, negative_size, negative_seed):
        block_index = bisect_right(cumulative_negative_counts, global_rank)
        prior_count = 0 if block_index == 0 else cumulative_negative_counts[block_index - 1]
        block = negative_blocks[block_index]
        pair1, pair2 = _unrank_negative_pair(block, global_rank - prior_count)
        records.append(
            PairRecord(
                source_domain=domain,
                source_family=family,
                pair1=pair1,
                pair2=pair2,
                label=0,
                label_rule="different_gold_cluster",
                origin=f"gold_clusters_balanced:{domain}",
                group_id=block.block_id,
            )
        )

    return canonicalize_pairs(records)


def _normalized_name(value: object) -> str:
    return " ".join(_nonempty_text(value, "signature name").casefold().split())


def _uniform_member_outside_key(
    groups: Mapping[str, Sequence[str]],
    excluded_key: str,
    rng: random.Random,
) -> str | None:
    eligible_count = sum(len(members) for key, members in groups.items() if key != excluded_key)
    if eligible_count == 0:
        return None
    target = rng.randrange(eligible_count)
    for key, members in groups.items():
        if key == excluded_key:
            continue
        if target < len(members):
            return members[target]
        target -= len(members)
    raise AssertionError("eligible partner rank was not resolved")


def sample_name_challenge_pairs(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster: Mapping[str, str],
    signature_to_name: Mapping[str, str],
    *,
    positive_size: int,
    negative_size: int,
    random_seed: int,
    source_domain: str,
    source_family: str = "gold_name_challenge",
) -> pd.DataFrame:
    """Anchor-uniformly sample synonym positives and homonym negatives.

    Each anchor contributes at most one pair to each stratum.  Consequently the
    function can return fewer than the requested counts when the gold data does
    not contain enough distinct eligible anchors; it does not backfill with
    easier pairs.
    """

    if positive_size < 0 or negative_size < 0:
        raise ValueError("positive_size and negative_size must be non-negative")
    domain = _nonempty_text(source_domain, "source_domain")
    validated_blocks = _validated_blocks(blocks, signature_to_cluster)
    cluster_lookup = {str(signature): str(cluster) for signature, cluster in signature_to_cluster.items()}
    name_lookup = {str(signature): _normalized_name(name) for signature, name in signature_to_name.items()}

    anchors: list[tuple[str, str]] = []
    positive_groups: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    negative_groups: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for block_id, signatures in validated_blocks:
        for signature in signatures:
            if signature not in name_lookup:
                raise ValueError(f"signature_id={signature!r} is missing a name for challenge sampling")
            cluster = cluster_lookup[signature]
            name = name_lookup[signature]
            anchors.append((block_id, signature))
            positive_groups[block_id][cluster][name].append(signature)
            negative_groups[block_id][name][cluster].append(signature)

    def sample_stratum(
        *,
        target_size: int,
        label: int,
        stratum: Literal["positive", "negative"],
    ) -> list[PairRecord]:
        rng = random.Random(_stable_seed(random_seed, domain, stratum))
        shuffled_anchors = anchors.copy()
        rng.shuffle(shuffled_anchors)
        records: list[PairRecord] = []
        seen_pairs: set[tuple[str, str]] = set()
        for block_id, anchor in shuffled_anchors:
            if len(records) == target_size:
                break
            cluster = cluster_lookup[anchor]
            name = name_lookup[anchor]
            if stratum == "positive":
                partner = _uniform_member_outside_key(positive_groups[block_id][cluster], name, rng)
                label_rule = "different_name_same_gold_cluster"
            else:
                partner = _uniform_member_outside_key(negative_groups[block_id][name], cluster, rng)
                label_rule = "same_name_different_gold_cluster"
            if partner is None:
                continue
            pair_key = (min(anchor, partner), max(anchor, partner))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            records.append(
                PairRecord(
                    source_domain=domain,
                    source_family=source_family,
                    pair1=anchor,
                    pair2=partner,
                    label=label,
                    label_rule=label_rule,
                    origin=f"name_challenge:{domain}",
                    group_id=block_id,
                )
            )
        return records

    positives = sample_stratum(target_size=positive_size, label=1, stratum="positive")
    negatives = sample_stratum(target_size=negative_size, label=0, stratum="negative")
    return canonicalize_pairs([*positives, *negatives])


__all__ = [
    "PAIR_COLUMNS",
    "PairRecord",
    "build_gold_cluster_lookup",
    "canonicalize_pairs",
    "cap_pairs_per_domain",
    "concat_pair_frames",
    "empty_pair_frame",
    "exclude_source_domains",
    "load_fixed_pair_csvs",
    "load_gold_cluster_lookup",
    "load_historical_augmented_pairs",
    "load_medline_pairs",
    "sample_name_challenge_pairs",
    "sample_pair_ranks",
    "sample_within_blocks_anchor_uniform",
    "sample_within_blocks_balanced",
    "sample_within_blocks_uniform",
    "unrank_pair",
]
