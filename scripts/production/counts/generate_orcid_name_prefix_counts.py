"""Generate canonical unordered ORCID first-name prefix counts.

Warehouse access is intentionally unavailable at import time. Use a bounded
local JSON fixture for development, or pass ``--run-full`` explicitly on
internal infrastructure. Outputs are written as an immutable generation and a
pointer manifest is published last.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Any

import orjson

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.consts import NORMALIZATION_VERSION
from s2and.data import _load_name_tuples_from_file
from s2and.text import canonicalize_name_parts, normalize_orcid, same_prefix_tokens

PAIR_KEY_SEMANTICS = "unordered_lexicographic"
K_VALUES = (2, 3, 4, 5)
_CANONICAL_SOURCE_ORCID_PATTERN = re.compile(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9Xx]")
QUERY = """
select p.year, p.inserted paper_inserted,
      pae.corpus_paper_id, pae.source, pae.orcid, pae.position,
      pae.first_name, pa.middle, pae.last_name,
      pa.corpus_author_id, au.ai2_id, pa.inserted pa_inserted,
      pa.updated pa_updated, pa.cluster_block_key, pa.model_version,
      pa.clusterer
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
order by regexp_replace(upper(pae.orcid), '[^0-9X]', ''),
         pae.first_name, pa.middle, pae.corpus_paper_id, pae.position
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


def canonical_orcid_name_groups(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Canonicalize source rows and group nonempty first names by ORCID."""

    groups: dict[str, set[str]] = defaultdict(set)
    cache: dict[tuple[Any, Any], str] = {}
    metrics = Counter[str]()
    for row in rows:
        metrics["source_rows"] += 1
        raw_orcid = row.get("orcid")
        orcid = _canonical_source_orcid(raw_orcid)
        if orcid is None:
            metric = "rejected_missing_orcid" if not str(raw_orcid or "").strip() else "rejected_invalid_orcid"
            metrics[metric] += 1
            continue
        raw_key = (row.get("first_name"), row.get("middle"))
        canonical_first = cache.get(raw_key)
        if canonical_first is None:
            parts = canonicalize_name_parts(raw_key[0], raw_key[1], None)
            canonical_first = parts.first
            cache[raw_key] = canonical_first
        if not canonical_first:
            metrics["rejected_empty_canonical_first"] += 1
            continue
        groups[orcid].add(canonical_first)
        metrics["accepted_rows"] += 1
    metrics["orcid_groups"] = len(groups)
    metrics["unique_orcid_names"] = sum(len(names) for names in groups.values())
    return dict(groups), dict(metrics)


def build_prefix_counts(
    orcid_name_groups: Mapping[str, Iterable[str]],
    name_tuples: Iterable[tuple[str, str]],
    *,
    min_orcid_count: int = 10,
    min_alias_count: int = 2,
    max_names_per_orcid: int = 100,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Build deterministic nested counts using canonical unordered pair keys."""

    if max_names_per_orcid < 2:
        raise ValueError("max_names_per_orcid must be at least 2")
    orcid_counts: Counter[tuple[str, str]] = Counter()
    rejected_empty_names = 0
    max_names_observed = 0
    for orcid, names in orcid_name_groups.items():
        valid_name_set: set[str] = set()
        for name in names:
            if not isinstance(name, str) or not name:
                rejected_empty_names += 1
                continue
            valid_name_set.add(name)
            if len(valid_name_set) > max_names_per_orcid:
                raise ValueError(
                    f"ORCID {orcid!r} has more than max_names_per_orcid={max_names_per_orcid} unique names; "
                    "raise the explicit bound only after reviewing the source group"
                )
        max_names_observed = max(max_names_observed, len(valid_name_set))
        valid_names = sorted(valid_name_set)
        for first_name, second_name in combinations(valid_names, 2):
            pairs = prefix_pairs_for_names(first_name, second_name)
            orcid_counts.update(pair for pair in pairs if not same_prefix_tokens(*pair))

    nested, merge_metrics, _name_tuples_digest = _merge_prefix_counts(
        orcid_counts,
        name_tuples,
        min_orcid_count=min_orcid_count,
        min_alias_count=min_alias_count,
    )
    return nested, {
        **merge_metrics,
        "rejected_empty_group_names": rejected_empty_names,
        "max_unique_names_per_orcid": max_names_observed,
        "max_names_per_orcid_limit": max_names_per_orcid,
    }


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
    previous_raw_orcid: str | None = None
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
        raw_orcid = row.get("orcid")
        raw_first_value = row.get("first_name")
        raw_middle_value = row.get("middle")
        raw_first = None if raw_first_value is None else str(raw_first_value)
        raw_middle = None if raw_middle_value is None else str(raw_middle_value)
        raw_orcid_text = None if raw_orcid is None else str(raw_orcid)
        if raw_orcid_text == previous_raw_orcid:
            orcid = previous_normalized_orcid
        else:
            orcid = _canonical_source_orcid(raw_orcid_text)
            previous_raw_orcid = raw_orcid_text
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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_canonical_prefix_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 2 <= len(value) <= 5
        and value.isascii()
        and value.isprintable()
        and value == value.lower()
    )


def _write_compact_json(path: Path, payload: Mapping[str, object]) -> tuple[str, int]:
    """Write deterministic compact JSON without materializing an intermediate string."""

    encoded = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    with path.open("wb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    return _sha256_bytes(encoded), len(encoded)


def _pointer_generation_id(path: Path) -> str | None:
    """Return the referenced generation, or ``None`` only when no pointer exists."""

    try:
        pointer_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise OSError(f"Unable to read published ORCID prefix-count pointer {path}: {error}") from error
    try:
        payload = json.loads(pointer_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Published ORCID prefix-count pointer is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Published ORCID prefix-count pointer must contain a JSON object: {path}")
    generation_id = payload.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError(f"Published ORCID prefix-count pointer has no valid generation_id: {path}")
    return generation_id


def _validate_counts_for_publication(counts: Mapping[str, Mapping[str, int]]) -> tuple[int, int]:
    """Validate the runtime pair-key contract before writing any generation files."""

    if not isinstance(counts, dict):
        raise TypeError("counts must be a plain dict so publication does not make an unbounded copy")
    pair_count = 0
    for left, nested in counts.items():
        if not _is_canonical_prefix_token(left):
            raise ValueError("counts outer keys must be lowercase printable ASCII prefixes of length 2 through 5")
        if not isinstance(nested, dict):
            raise TypeError("counts nested values must be plain dictionaries")
        for right, count in nested.items():
            if not _is_canonical_prefix_token(right) or left >= right:
                raise ValueError("counts pairs must be unequal and lexicographically ordered")
            if (
                not 2 <= len(left) <= 5
                or not 2 <= len(right) <= 5
                or left[0] != right[0]
                or same_prefix_tokens(left, right)
            ):
                raise ValueError("counts keys violate the generated prefix-pair semantics")
            if type(count) is not int or count <= 0:
                raise ValueError("counts values must be positive integers")
            pair_count += 1
    return len(counts), pair_count


@contextmanager
def _publish_lock(output_dir: Path) -> Iterator[None]:
    """Serialize the short manifest check-and-replace boundary across processes."""

    with exclusive_file_lock(output_dir / ".orcid-prefix-counts.publish.lock"):
        yield


def publish_generation(
    counts: Mapping[str, Mapping[str, int]],
    *,
    output_dir: Path,
    source_snapshot_id: str,
    source_digest: str,
    metrics: Mapping[str, int],
    overwrite: bool,
) -> Path:
    """Publish an immutable data/metadata generation and pointer manifest."""

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_snapshot_id) is None:
        raise ValueError("source_snapshot_id must contain only letters, digits, '.', '_', and '-'")
    if re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("source_digest must be a lowercase SHA-256 digest of the selected canonical inputs")
    outer_key_cardinality, pair_key_cardinality = _validate_counts_for_publication(counts)
    output_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = output_dir / "first_k_letter_counts_from_orcid.manifest.json"
    if pointer_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists; pass --overwrite to replace it: {pointer_path}")
    generation_id = f"{source_snapshot_id}-{uuid.uuid4().hex[:12]}"
    final_dir = output_dir / f"orcid-prefix-counts-{generation_id}"
    if final_dir.exists():
        raise FileExistsError(f"Generation already exists: {final_dir}")
    staging_dir: Path | None = Path(tempfile.mkdtemp(prefix=".orcid-prefix-counts-", dir=output_dir))
    final_dir_published = False
    pointer_tmp: Path | None = None
    try:
        data_path = staging_dir / "first_k_letter_counts_from_orcid.json"
        data_sha256, data_byte_count = _write_compact_json(data_path, counts)
        metadata = {
            "schema_version": 1,
            "normalization_version": NORMALIZATION_VERSION,
            "pair_key_semantics": PAIR_KEY_SEMANTICS,
            "generation_id": generation_id,
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_snapshot_id": source_snapshot_id,
            "source_digest": source_digest,
            "data_sha256": data_sha256,
            "data_byte_count": data_byte_count,
            "outer_key_cardinality": outer_key_cardinality,
            "pair_key_cardinality": pair_key_cardinality,
            "metrics": dict(metrics),
        }
        metadata_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
        (staging_dir / "first_k_letter_counts_from_orcid.meta.json").write_bytes(metadata_bytes)
        for path in staging_dir.iterdir():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
        fsync_directory(staging_dir)
        os.replace(staging_dir, final_dir)
        staging_dir = None
        final_dir_published = True
        fsync_directory(output_dir)
        pointer = {
            "schema_version": 1,
            "generation_id": generation_id,
            "generation_dir": final_dir.name,
            "metadata_sha256": _sha256_bytes(metadata_bytes),
        }
        with _publish_lock(output_dir):
            if pointer_path.exists() and not overwrite:
                raise FileExistsError(f"Manifest already exists; pass --overwrite to replace it: {pointer_path}")
            pointer_tmp = output_dir / f".{pointer_path.name}.{uuid.uuid4().hex}.tmp"
            pointer_tmp.write_text(json.dumps(pointer, sort_keys=True, indent=2), encoding="utf-8")
            with pointer_tmp.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(pointer_tmp, pointer_path)
            pointer_tmp = None
            fsync_directory(output_dir)
        return pointer_path
    finally:
        publication_error = sys.exception()
        if pointer_tmp is not None:
            pointer_tmp.unlink(missing_ok=True)
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)
        if final_dir_published and final_dir.exists():
            with _publish_lock(output_dir):
                try:
                    referenced_generation_id = _pointer_generation_id(pointer_path)
                except (OSError, ValueError) as cleanup_error:
                    if publication_error is None:
                        raise
                    publication_error.add_note(
                        f"Retained generation {final_dir} because the published pointer "
                        f"could not be inspected during cleanup: {cleanup_error}"
                    )
                else:
                    if referenced_generation_id != generation_id:
                        shutil.rmtree(final_dir)
                        fsync_directory(output_dir)


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
    required_columns = ("orcid", "first_name", "middle")
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
    pointer_path = publish_generation(
        counts,
        output_dir=args.output_dir,
        source_snapshot_id=args.source_snapshot_id,
        source_digest=source_digest,
        metrics=metrics,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"manifest": str(pointer_path), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
