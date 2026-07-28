"""Generate one immutable canonical ORCID first-name prefix-count artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import orjson

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.name_tuple_artifact import NameTupleArtifact, load_name_tuple_artifact
from s2and.orcid_prefix_counts import (
    ORCID_PREFIX_DATA_FILENAME,
    ORCID_PREFIX_MANIFEST_FILENAME,
    LoadedOrcidPrefixCounts,
    load_canonical_orcid_prefix_counts,
    validate_orcid_prefix_counts,
)
from s2and.text import canonicalize_name_parts, normalize_orcid, same_prefix_tokens

from ._run_support import (
    load_guardrails,
    require_positive,
    validate_fixture_path,
    validate_output_container,
)

K_VALUES = (2, 3, 4, 5)
MIN_ORCID_COUNT = 10
MIN_ALIAS_COUNT = 2
FIXTURE_MAX_NAMES_PER_ORCID = 100
PROGRESS_EVERY = 100_000
GUARDRAIL_FIELDS = frozenset(
    {
        "max_source_rows",
        "min_source_rows",
        "max_names_per_orcid",
        "max_pair_keys",
        "min_orcid_pair_keys",
    }
)
_CANONICAL_SOURCE_ORCID_PATTERN = re.compile(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9Xx]")


def _canonical_source_orcid(value: Any) -> str | None:
    """Normalize source ORCIDs with a cheap path for the warehouse shape."""

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
    pairs: set[tuple[str, str]] = set()
    for first_prefix in {first_name[:k] for k in k_values}:
        for second_prefix in {second_name[:k] for k in k_values}:
            left, right = canonical_prefix_pair(first_prefix, second_prefix)
            if left != right and not same_prefix_tokens(left, right):
                pairs.add((left, right))
    return pairs


def _merge_prefix_counts(
    orcid_counts: Counter[tuple[str, str]],
    name_tuples: Iterable[tuple[str, str]],
    *,
    min_orcid_count: int,
    min_alias_count: int,
    max_pair_keys: int | None,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Threshold and merge ORCID/alias counts under one live pair bound."""

    canonical_name_tuples: set[tuple[str, str]] = set()
    for pair in name_tuples:
        if (
            not isinstance(pair, tuple | list)
            or len(pair) != 2
            or not all(isinstance(name, str) and name for name in pair)
        ):
            raise ValueError("name_tuples must contain pairs of nonempty canonical strings")
        canonical_name_tuples.add(canonical_prefix_pair(*pair))

    alias_counts: Counter[tuple[str, str]] = Counter()
    for first_name, second_name in sorted(canonical_name_tuples):
        alias_counts.update(prefix_pairs_for_names(first_name, second_name))
        if max_pair_keys is not None and len(alias_counts) > max_pair_keys:
            raise ValueError(f"alias prefix pairs exceeded guardrail max_pair_keys={max_pair_keys}")

    nested: dict[str, dict[str, int]] = {}
    orcid_pair_keys_after_threshold = 0
    for (left, right), count in sorted(orcid_counts.items()):
        if count >= min_orcid_count:
            nested.setdefault(left, {})[right] = int(count)
            orcid_pair_keys_after_threshold += 1
    for (left, right), count in sorted(alias_counts.items()):
        if count >= min_alias_count:
            nested.setdefault(left, {}).setdefault(right, int(count))
    output_pair_keys = sum(len(counts) for counts in nested.values())
    if max_pair_keys is not None and output_pair_keys > max_pair_keys:
        raise ValueError(f"published prefix pairs exceeded guardrail max_pair_keys={max_pair_keys}")
    return nested, {
        "orcid_pair_keys_before_threshold": len(orcid_counts),
        "orcid_pair_keys_after_threshold": orcid_pair_keys_after_threshold,
        "alias_pair_keys_before_threshold": len(alias_counts),
        "selected_name_tuple_pairs": len(canonical_name_tuples),
        "output_pair_keys": output_pair_keys,
        "output_outer_keys": len(nested),
    }


def build_prefix_counts_from_sorted_rows(
    rows: Iterable[Mapping[str, Any]],
    name_tuples: Iterable[tuple[str, str]],
    *,
    min_orcid_count: int = MIN_ORCID_COUNT,
    min_alias_count: int = MIN_ALIAS_COUNT,
    max_names_per_orcid: int = FIXTURE_MAX_NAMES_PER_ORCID,
    max_source_rows: int | None = None,
    max_pair_keys: int | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Stream sorted rows one ORCID at a time under explicit expansion bounds."""

    if max_names_per_orcid < 2:
        raise ValueError("max_names_per_orcid must be at least 2")
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
        metrics["max_unique_names_per_orcid"] = max(metrics["max_unique_names_per_orcid"], unique_name_count)
        sorted_names = sorted(current_names)
        metrics["selected_canonical_rows"] += unique_name_count
        for first_name, second_name in combinations(sorted_names, 2):
            orcid_counts.update(prefix_pairs_for_names(first_name, second_name))
        if max_pair_keys is not None and len(orcid_counts) > max_pair_keys:
            raise ValueError(f"ORCID prefix pairs exceeded guardrail max_pair_keys={max_pair_keys}")

    for row in rows:
        metrics["source_rows"] += 1
        if max_source_rows is not None and metrics["source_rows"] > max_source_rows:
            raise ValueError(f"source rows exceeded guardrail max_source_rows={max_source_rows}")
        if progress_callback is not None and metrics["source_rows"] % PROGRESS_EVERY == 0:
            progress_callback(
                {
                    "source_rows": metrics["source_rows"],
                    "accepted_rows": metrics["accepted_rows"],
                    "orcid_groups_completed": metrics["orcid_groups"],
                    "orcid_pair_keys": len(orcid_counts),
                }
            )

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

        name_key = (raw_first, raw_middle)
        normalized_first = canonical_first_cache.get(name_key)
        if normalized_first is None:
            normalized_first = canonicalize_name_parts(raw_first, raw_middle, None).first
            if len(canonical_first_cache) >= 100_000:
                canonical_first_cache.clear()
            canonical_first_cache[name_key] = normalized_first
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
            raise ValueError(f"ORCID {orcid!r} has more than max_names_per_orcid={max_names_per_orcid} unique names")
        current_names.add(normalized_first)
        metrics["accepted_rows"] += 1
    flush_group()

    counts, count_metrics = _merge_prefix_counts(
        orcid_counts,
        name_tuples,
        min_orcid_count=min_orcid_count,
        min_alias_count=min_alias_count,
        max_pair_keys=max_pair_keys,
    )
    metrics["max_names_per_orcid_limit"] = max_names_per_orcid
    return counts, {**dict(metrics), **count_metrics}


def _publication_payloads(
    counts: Mapping[str, Mapping[str, int]],
    *,
    name_tuples_sha256: str,
) -> dict[str, bytes]:
    """Serialize the trusted runtime data and its tuple dependency."""

    validate_orcid_prefix_counts(counts, context="counts")
    data_payload = orjson.dumps(counts, option=orjson.OPT_SORT_KEYS)
    manifest = {"name_tuples_sha256": name_tuples_sha256}
    return {
        ORCID_PREFIX_DATA_FILENAME: data_payload,
        ORCID_PREFIX_MANIFEST_FILENAME: (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    }


def _publish(
    payloads: Mapping[str, bytes],
    *,
    output_dir: Path,
) -> LoadedOrcidPrefixCounts:
    """Validate once in a sibling directory, then atomically publish it."""

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_parent))
    lock_path = output_parent / f".{output_dir.name}.publish.lock"
    try:
        for filename, payload in payloads.items():
            with (staging_dir / filename).open("wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        fsync_directory(staging_dir)
        loaded = load_canonical_orcid_prefix_counts(staging_dir)
        with exclusive_file_lock(lock_path):
            if output_dir.exists():
                raise FileExistsError(f"publication target already exists: {output_dir}")
            staging_dir.rename(output_dir)
        fsync_directory(output_parent)
        return loaded
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def write_publication(
    counts: Mapping[str, Mapping[str, int]],
    *,
    output_dir: Path,
    name_tuples: NameTupleArtifact,
) -> LoadedOrcidPrefixCounts:
    """Serialize, validate, and atomically publish one fresh artifact."""

    return _publish(
        _publication_payloads(
            counts,
            name_tuples_sha256=name_tuples.data_sha256,
        ),
        output_dir=output_dir,
    )


def _load_fixture_rows(path: Path, limit: int | None) -> list[Mapping[str, Any]]:
    rows = json.loads(path.read_bytes())
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("--input-json must contain a JSON list of row objects")
    rows.sort(
        key=lambda row: (
            _canonical_source_orcid(row.get("orcid")) or "",
            "" if row.get("first_name") is None else str(row.get("first_name")),
            "" if row.get("middle") is None else str(row.get("middle")),
        )
    )
    return rows[:limit]


def _load_reviewed_csv_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream one reviewed, query-ordered warehouse export."""

    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = reader.fieldnames or []
        expected = ["raw_orcid", "orcid", "first_name", "middle"]
        if fields != expected:
            raise ValueError(f"Reviewed ORCID export must have exact header: {expected}")
        for row in reader:
            if None in row:
                raise ValueError("Reviewed ORCID export row has more values than columns")
            yield row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path, help="Bounded local row fixture")
    source.add_argument("--input-csv", type=Path, help="Reviewed full warehouse export")
    parser.add_argument("--limit", type=int, help="Fixture-only row limit")
    parser.add_argument("--guardrails-json", type=Path, help="Reviewed full-run bounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-tuples-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the guarded producer."""

    args = _parser().parse_args(argv)
    limit = require_positive(args.limit, option="--limit")
    full_input = args.input_csv.resolve() if args.input_csv is not None else None
    if full_input is not None and limit is not None:
        raise ValueError("--limit is fixture-only; it does not bound warehouse scan cost")
    output_dir = validate_output_container(args.output_dir, publication_path=args.output_dir)
    fixture = validate_fixture_path(args.input_json) if args.input_json is not None else None
    guardrails = load_guardrails(args.guardrails_json, fields=GUARDRAIL_FIELDS) if full_input is not None else None
    if guardrails is not None:
        if guardrails["min_source_rows"] > guardrails["max_source_rows"]:
            raise ValueError("guardrail min_source_rows must not exceed max_source_rows")
        if guardrails["min_orcid_pair_keys"] > guardrails["max_pair_keys"]:
            raise ValueError("guardrail min_orcid_pair_keys must not exceed max_pair_keys")
        if guardrails["max_names_per_orcid"] < 2:
            raise ValueError("guardrail max_names_per_orcid must be at least 2")

    name_tuples = load_name_tuple_artifact(args.name_tuples_path.resolve())

    if full_input is not None:
        max_names_per_orcid = guardrails["max_names_per_orcid"]
    else:
        assert fixture is not None
        max_names_per_orcid = FIXTURE_MAX_NAMES_PER_ORCID
    plan = {
        "mode": "full" if full_input is not None else "fixture",
        "source": str(full_input or fixture),
        "output_dir": str(output_dir),
        "guardrails": guardrails,
        "limit": limit,
        "name_tuples_sha256": name_tuples.data_sha256,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    if full_input is not None:
        rows = _load_reviewed_csv_rows(full_input)
    else:
        assert fixture is not None
        rows = _load_fixture_rows(fixture, limit)

    def report_progress(metrics: dict[str, int]) -> None:
        print(json.dumps({"event": "orcid_prefix_progress", **metrics}, sort_keys=True))

    counts, metrics = build_prefix_counts_from_sorted_rows(
        rows,
        name_tuples.pairs,
        max_names_per_orcid=max_names_per_orcid,
        max_source_rows=None if guardrails is None else guardrails["max_source_rows"],
        max_pair_keys=None if guardrails is None else guardrails["max_pair_keys"],
        progress_callback=report_progress,
    )
    if int(metrics.get("source_rows", 0)) == 0:
        raise RuntimeError("ORCID prefix-count generation selected zero source rows")
    if int(metrics.get("accepted_rows", 0)) == 0 or int(metrics.get("orcid_groups", 0)) == 0:
        raise RuntimeError("ORCID prefix-count generation selected no usable ORCID/name groups")
    if int(metrics.get("output_pair_keys", 0)) == 0:
        raise RuntimeError("ORCID prefix-count generation produced zero output pair keys")
    if guardrails is not None:
        source_rows = int(metrics["source_rows"])
        orcid_pair_keys = int(metrics["orcid_pair_keys_after_threshold"])
        if source_rows < guardrails["min_source_rows"]:
            raise RuntimeError(
                f"source rows {source_rows} are below guardrail min_source_rows={guardrails['min_source_rows']}"
            )
        if orcid_pair_keys < guardrails["min_orcid_pair_keys"]:
            raise RuntimeError(
                "ORCID-derived pair keys "
                f"{orcid_pair_keys} are below guardrail min_orcid_pair_keys={guardrails['min_orcid_pair_keys']}"
            )

    write_publication(
        counts,
        output_dir=output_dir,
        name_tuples=name_tuples,
    )
    print(
        json.dumps(
            {
                "data": str(output_dir / ORCID_PREFIX_DATA_FILENAME),
                "manifest": str(output_dir / ORCID_PREFIX_MANIFEST_FILENAME),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
