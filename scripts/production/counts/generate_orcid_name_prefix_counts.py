"""Generate canonical unordered ORCID first-name prefix counts.

Warehouse access is intentionally unavailable at import time. Use a bounded
local JSON fixture for development, or pass ``--run-full`` explicitly on
internal infrastructure. The output is one data file and its metadata sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import orjson

from s2and.consts import NORMALIZATION_VERSION
from s2and.data import _load_name_tuples_from_file
from s2and.orcid_prefix_counts import (
    ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
    ORCID_PREFIX_DATA_FILENAME,
    ORCID_PREFIX_METADATA_FILENAME,
    ORCID_PREFIX_PAIR_KEY_SEMANTICS,
    validate_orcid_prefix_counts,
)
from s2and.text import canonicalize_name_parts, normalize_orcid, same_prefix_tokens

K_VALUES = (2, 3, 4, 5)
_CANONICAL_SOURCE_ORCID_PATTERN = re.compile(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9Xx]")
_CANONICAL_SOURCE_ORCID_SQL_PATTERN = (
    r"(?<![0-9x])[0-9]{4}[-‐‑‒–—−﹘﹣－]?[0-9]{4}[-‐‑‒–—−﹘﹣－]?" r"[0-9]{4}[-‐‑‒–—−﹘﹣－]?[0-9]{3}[0-9x](?![0-9x])"
)
_ORCID_DASH_SQL_PATTERN = "[-‐‑‒–—−﹘﹣－]"
QUERY = f"""
with source_rows as (
select p.year, p.inserted paper_inserted,
      pae.corpus_paper_id, pae.source, pae.orcid raw_orcid, pae.position,
      pae.first_name, pa.middle, pae.last_name,
      pa.corpus_author_id, au.ai2_id, pa.inserted pa_inserted,
      pa.updated pa_updated, pa.cluster_block_key, pa.model_version,
      pa.clusterer,
      upper(
        regexp_replace(
          regexp_substr(
            coalesce(pae.orcid, ''),
            '{_CANONICAL_SOURCE_ORCID_SQL_PATTERN}',
            1,
            1,
            'ip'
          ),
          '{_ORCID_DASH_SQL_PATTERN}'
        )
      ) canonical_orcid_compact
from content_ext.paper_authors_orcids pae
join content_ext.papers p
  on pae.corpus_paper_id = p.corpus_paper_id
join content_ext.paper_authors pa
  on pae.corpus_paper_id = pa.corpus_paper_id
 and pae.position = pa.position + 1
 and lower(pae.last_name) = lower(pa.last)
join content_ext.authors au
  on pa.corpus_author_id = au.corpus_author_id
where pae.source in ('Crossref')
  and nullif(trim(coalesce(pae.first_name, '')), '') is not null
)
select year, paper_inserted, corpus_paper_id, source, raw_orcid,
      case
        when canonical_orcid_compact = '' then null
        else substring(canonical_orcid_compact, 1, 4)
          || '-' || substring(canonical_orcid_compact, 5, 4)
          || '-' || substring(canonical_orcid_compact, 9, 4)
          || '-' || substring(canonical_orcid_compact, 13, 4)
      end orcid,
      position, first_name, middle, last_name,
      corpus_author_id, ai2_id, pa_inserted, pa_updated,
      cluster_block_key, model_version, clusterer
from source_rows
order by orcid nulls last,
         first_name, middle, corpus_paper_id, position
"""


def _canonical_source_orcid(value: Any) -> str | None:
    """Normalize source ORCIDs with a cheap path for the warehouse's canonical shape."""

    if value is None:
        return None
    text = str(value).strip()
    if _CANONICAL_SOURCE_ORCID_PATTERN.fullmatch(text) is not None:
        return text if text[18] != "x" else f"{text[:18]}X"
    return normalize_orcid(text)


def canonical_prefix_pair(first: str, second: str) -> tuple[str, str]:
    """Return an order-independent prefix-pair key."""

    return (first, second) if first <= second else (second, first)


def prefix_pairs_for_names(
    first_name: str,
    second_name: str,
    *,
    k_values: Sequence[int] = K_VALUES,
) -> set[tuple[str, str]]:
    """Return canonical unequal prefix pairs for two nonempty names."""

    if not first_name or not second_name or first_name[0] != second_name[0]:
        return set()
    first_prefixes = {first_name[:k] for k in k_values}
    second_prefixes = {second_name[:k] for k in k_values}
    pairs: set[tuple[str, str]] = set()
    for first_prefix in first_prefixes:
        for second_prefix in second_prefixes:
            left, right = canonical_prefix_pair(first_prefix, second_prefix)
            if left != right:
                pairs.add((left, right))
    return pairs


def _merge_prefix_counts(
    orcid_counts: Counter[tuple[str, str]],
    name_tuples: Iterable[tuple[str, str]],
    *,
    min_orcid_count: int,
    min_alias_count: int,
) -> tuple[dict[str, dict[str, int]], dict[str, int], str]:
    """Threshold and merge ORCID/alias counts without a duplicate merged mapping."""

    canonical_name_tuples: set[tuple[str, str]] = set()
    for pair in name_tuples:
        if (
            not isinstance(pair, tuple | list)
            or len(pair) != 2
            or not all(isinstance(name, str) and name for name in pair)
        ):
            raise ValueError("name_tuples must contain pairs of nonempty canonical strings")
        canonical_name_tuples.add(canonical_prefix_pair(*pair))
    sorted_name_tuples = sorted(canonical_name_tuples)
    del canonical_name_tuples
    selected_name_tuple_count = len(sorted_name_tuples)
    name_tuples_digest = hashlib.sha256(b"s2and-orcid-prefix-count-name-tuples-v1\0")
    name_tuples_digest.update(selected_name_tuple_count.to_bytes(8, "little"))
    alias_counts: Counter[tuple[str, str]] = Counter()
    for first_name, second_name in sorted_name_tuples:
        for value in (first_name, second_name):
            encoded = value.encode("utf-8")
            name_tuples_digest.update(len(encoded).to_bytes(8, "little"))
            name_tuples_digest.update(encoded)
        pairs = prefix_pairs_for_names(first_name, second_name)
        alias_counts.update(pair for pair in pairs if not same_prefix_tokens(*pair))
    del sorted_name_tuples

    nested: dict[str, dict[str, int]] = {}
    for (left, right), count in sorted(orcid_counts.items()):
        if count >= min_orcid_count:
            nested.setdefault(left, {})[right] = int(count)
    for (left, right), count in sorted(alias_counts.items()):
        if count >= min_alias_count:
            nested.setdefault(left, {}).setdefault(right, int(count))
    output_pair_keys = sum(len(counts) for counts in nested.values())
    metrics = {
        "orcid_pair_keys_before_threshold": len(orcid_counts),
        "alias_pair_keys_before_threshold": len(alias_counts),
        "selected_name_tuple_pairs": selected_name_tuple_count,
        "output_pair_keys": output_pair_keys,
        "output_outer_keys": len(nested),
    }
    return nested, metrics, name_tuples_digest.hexdigest()


def build_prefix_counts_from_sorted_rows(
    rows: Iterable[Mapping[str, Any]],
    name_tuples: Iterable[tuple[str, str]],
    *,
    min_orcid_count: int = 10,
    min_alias_count: int = 2,
    max_names_per_orcid: int = 100,
) -> tuple[dict[str, dict[str, int]], dict[str, int], str]:
    """Stream sorted source rows one ORCID at a time and hash the selected content."""

    if max_names_per_orcid < 2:
        raise ValueError("max_names_per_orcid must be at least 2")
    source_digest = hashlib.sha256(b"s2and-orcid-prefix-count-selected-input-v1\0")
    metrics = Counter[str]()
    orcid_counts: Counter[tuple[str, str]] = Counter()
    current_orcid: str | None = None
    current_names: set[str] = set()
    previous_orcid: str | None = None
    previous_source_orcid: str | None = None
    previous_normalized_orcid: str | None = None
    canonical_first_cache: dict[tuple[str | None, str | None], str] = {}

    def flush_group() -> None:
        if current_orcid is None:
            return
        unique_name_count = len(current_names)
        metrics["orcid_groups"] += 1
        metrics["unique_orcid_names"] += unique_name_count
        metrics["max_unique_names_per_orcid"] = max(
            metrics["max_unique_names_per_orcid"],
            unique_name_count,
        )
        if unique_name_count > max_names_per_orcid:
            raise ValueError(
                f"ORCID {current_orcid!r} has {unique_name_count} unique names, exceeding "
                f"max_names_per_orcid={max_names_per_orcid}; raise the explicit bound only after "
                "reviewing the source group"
            )
        sorted_names = sorted(current_names)
        digest_group = bytearray()
        orcid_bytes = current_orcid.encode("utf-8")
        digest_group.extend(len(orcid_bytes).to_bytes(8, "little"))
        digest_group.extend(orcid_bytes)
        digest_group.extend(unique_name_count.to_bytes(8, "little"))
        for name in sorted_names:
            name_bytes = name.encode("utf-8")
            digest_group.extend(len(name_bytes).to_bytes(8, "little"))
            digest_group.extend(name_bytes)
        source_digest.update(digest_group)
        metrics["selected_canonical_rows"] += unique_name_count
        for first_name, second_name in combinations(sorted_names, 2):
            pairs = prefix_pairs_for_names(first_name, second_name)
            orcid_counts.update(pair for pair in pairs if not same_prefix_tokens(*pair))

    for row in rows:
        metrics["source_rows"] += 1
        raw_orcid = row.get("raw_orcid", row.get("orcid"))
        source_orcid = row.get("orcid")
        raw_first_value = row.get("first_name")
        raw_middle_value = row.get("middle")
        raw_first = None if raw_first_value is None else str(raw_first_value)
        raw_middle = None if raw_middle_value is None else str(raw_middle_value)
        source_orcid_text = None if source_orcid is None else str(source_orcid)
        if source_orcid_text == previous_source_orcid:
            orcid = previous_normalized_orcid
        else:
            orcid = _canonical_source_orcid(source_orcid_text)
            previous_source_orcid = source_orcid_text
            previous_normalized_orcid = orcid
        if orcid is None:
            metric = "rejected_missing_orcid" if not str(raw_orcid or "").strip() else "rejected_invalid_orcid"
            metrics[metric] += 1
            continue
        if previous_orcid is not None and orcid < previous_orcid:
            raise ValueError("ORCID source rows must be sorted by canonical orcid")
        previous_orcid = orcid
        raw_name_key = (raw_first, raw_middle)
        normalized_first = canonical_first_cache.get(raw_name_key)
        if normalized_first is None:
            normalized_first = canonicalize_name_parts(raw_first, raw_middle, None).first
            if len(canonical_first_cache) >= 100_000:
                canonical_first_cache.clear()
            canonical_first_cache[raw_name_key] = normalized_first
        if not normalized_first:
            metrics["rejected_empty_canonical_first"] += 1
            continue
        if current_orcid is None:
            current_orcid = orcid
        elif orcid != current_orcid:
            flush_group()
            current_names.clear()
            current_orcid = orcid
        if normalized_first not in current_names and len(current_names) >= max_names_per_orcid:
            raise ValueError(
                f"ORCID {orcid!r} has more than max_names_per_orcid={max_names_per_orcid} unique names; "
                "raise the explicit bound only after reviewing the source group"
            )
        current_names.add(normalized_first)
        metrics["accepted_rows"] += 1
    flush_group()
    canonical_first_cache.clear()

    counts, count_metrics, name_tuples_digest = _merge_prefix_counts(
        orcid_counts,
        name_tuples,
        min_orcid_count=min_orcid_count,
        min_alias_count=min_alias_count,
    )
    metrics["max_names_per_orcid_limit"] = max_names_per_orcid
    metrics["rejected_empty_group_names"] = 0
    source_digest.update(b"\0canonical-name-tuples-sha256\0")
    source_digest.update(bytes.fromhex(name_tuples_digest))
    return counts, {**dict(metrics), **count_metrics}, source_digest.hexdigest()


def _write_compact_json(path: Path, payload: Mapping[str, object]) -> str:
    """Write deterministic compact JSON without materializing an intermediate string."""

    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_artifact(
    counts: Mapping[str, Mapping[str, int]],
    *,
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, str]:
    """Write the canonical data file and adjacent metadata sidecar."""

    validate_orcid_prefix_counts(counts, context="counts")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / ORCID_PREFIX_DATA_FILENAME
    metadata_path = output_dir / ORCID_PREFIX_METADATA_FILENAME
    existing_paths = [path for path in (data_path, metadata_path) if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "ORCID prefix-count artifact already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing_paths)
        )

    data_sha256 = _write_compact_json(data_path, counts)
    metadata = {
        "schema_version": ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "pair_key_semantics": ORCID_PREFIX_PAIR_KEY_SEMANTICS,
        "data_sha256": data_sha256,
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return data_path, metadata_path, data_sha256


def _load_fixture_rows(path: Path, limit: int | None) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_bytes())
    if not isinstance(rows, list):
        raise ValueError("--input-json must contain a JSON list of row objects")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("--input-json rows must be JSON objects")
    rows.sort(
        key=lambda row: (
            _canonical_source_orcid(row.get("orcid")) or "",
            "" if row.get("first_name") is None else str(row.get("first_name")),
            "" if row.get("middle") is None else str(row.get("middle")),
        )
    )
    return rows[:limit]


def _warehouse_query(limit: int | None) -> str:
    return QUERY if limit is None else f"{QUERY}\nlimit {int(limit)}"


def _load_warehouse_rows(limit: int | None) -> Iterable[Mapping[str, Any]]:
    try:
        from pys2.pys2 import _evaluate_redshift_query  # type: ignore
    except ImportError as error:
        raise RuntimeError("Warehouse generation requires internal package pys2") from error
    query = _warehouse_query(limit)
    print(json.dumps({"warehouse_query": query, "limit": limit}, indent=2))
    dataframe = _evaluate_redshift_query(query)
    columns = {str(column): index for index, column in enumerate(dataframe.columns)}
    required_columns = ("raw_orcid", "orcid", "first_name", "middle")
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(f"Warehouse result is missing required columns: {missing_columns}")

    def iter_rows() -> Iterator[Mapping[str, Any]]:
        for values in dataframe.itertuples(index=False, name=None):
            yield {column: values[columns[column]] for column in required_columns}

    return iter_rows()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--input-json", type=Path, help="Bounded local row fixture")
    source.add_argument("--run-full", action="store_true", help="Authorize the internal warehouse query")
    parser.add_argument("--limit", type=int, help="Maximum source rows")
    parser.add_argument("--dry-run", action="store_true", help="Print the intended source/query without executing")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument(
        "--max-names-per-orcid",
        type=int,
        default=100,
        help="Hard bound applied before quadratic within-ORCID name pairing",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded generator CLI."""

    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_names_per_orcid < 2:
        raise ValueError("--max-names-per-orcid must be at least 2")
    if args.input_json is None and not args.run_full:
        raise ValueError("Choose --input-json for a fixture or explicitly authorize warehouse access with --run-full")
    source_context = {
        "input_json": None if args.input_json is None else str(args.input_json),
        "run_full": bool(args.run_full),
        "limit": args.limit,
        "output_dir": str(args.output_dir),
        "source_snapshot_id": args.source_snapshot_id,
        "max_names_per_orcid": args.max_names_per_orcid,
    }
    print(json.dumps(source_context, indent=2))
    if args.dry_run:
        if args.run_full:
            print(_warehouse_query(args.limit))
        return 0

    if args.input_json is not None:
        rows = _load_fixture_rows(args.input_json, args.limit)
    else:
        rows = _load_warehouse_rows(args.limit)
    name_tuples = _load_name_tuples_from_file("s2and_name_tuples_canonical.txt")
    counts, metrics, source_digest = build_prefix_counts_from_sorted_rows(
        rows,
        name_tuples,
        max_names_per_orcid=args.max_names_per_orcid,
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.source_snapshot_id) is None:
        raise ValueError("source_snapshot_id must contain only letters, digits, '.', '_', and '-'")
    data_path, metadata_path, data_sha256 = write_artifact(
        counts,
        output_dir=args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "data": str(data_path),
                "metadata": str(metadata_path),
                "data_sha256": data_sha256,
                "source_snapshot_id": args.source_snapshot_id,
                "source_digest": source_digest,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
