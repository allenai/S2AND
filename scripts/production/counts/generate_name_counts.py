"""Generate canonical name-count artifacts with explicit warehouse guardrails."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import NAME_COUNTS_PROVENANCE_SCHEMA_VERSION
from s2and.text import canonical_name_count_keys, canonicalize_name_parts

QUERY = """
select nvl(first_name, '') as first_name,
       nvl(last_name, '') as last_name,
       count(*) as count
from content.authors
group by nvl(first_name, ''), nvl(last_name, '')
""".strip()

NameCountRow = tuple[str, str, int]
NameCountMappings = tuple[
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
]


def _query_text(limit: int | None) -> str:
    """Return the exact deterministic warehouse query for this run."""

    ordered = f"{QUERY}\norder by first_name, last_name"
    return ordered if limit is None else f"{ordered}\nlimit {int(limit)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-full",
        action="store_true",
        help="authorize access to the internal warehouse",
    )
    source.add_argument(
        "--fixture-input",
        type=Path,
        help="local JSON rows with first_name, last_name, and count",
    )
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument("--limit", type=int, help="Maximum source rows")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace the published manifest pointer")
    return parser


def _validated_limit(value: int | None) -> int | None:
    if value is not None and value < 1:
        raise ValueError("--limit must be positive")
    return value


def _query_rows(limit: int | None) -> Iterator[NameCountRow]:
    try:
        from pys2 import _evaluate_redshift_query  # type: ignore
    except ImportError as exc:
        raise RuntimeError("warehouse generation requires the internal pys2 package") from exc
    frame = _evaluate_redshift_query(_query_text(limit))
    for first, last, count in zip(frame["first_name"], frame["last_name"], frame["count"], strict=True):
        yield str(first), str(last), int(count)


def _fixture_rows(path: Path, limit: int | None) -> Iterator[NameCountRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("fixture input must contain a JSON list")
    for row_index, row in enumerate(payload):
        if limit is not None and row_index >= limit:
            return
        if not isinstance(row, dict):
            raise ValueError(f"fixture row {row_index} must be an object")
        first = row.get("first_name", "")
        last = row.get("last_name", "")
        count = row.get("count")
        if not isinstance(first, str) or not isinstance(last, str):
            raise ValueError(f"fixture row {row_index} names must be strings")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"fixture row {row_index} count must be a positive integer")
        yield first, last, count


def build_name_count_dicts(
    rows: Iterable[NameCountRow],
) -> tuple[NameCountMappings, dict[str, int | str]]:
    """Canonicalize source rows and return the four filtered lookup mappings."""

    counters: list[Counter[str]] = [Counter(), Counter(), Counter(), Counter()]
    key_names = ("first", "last", "first_last", "last_first_initial")
    source_rows = 0
    rejected_rows = 0
    selected_rows_digest = hashlib.sha256()
    for raw_first, raw_last, count in rows:
        source_rows += 1
        for raw_name in (raw_first, raw_last):
            raw_bytes = raw_name.encode("utf-8")
            selected_rows_digest.update(len(raw_bytes).to_bytes(8, "little", signed=False))
            selected_rows_digest.update(raw_bytes)
        selected_rows_digest.update(int(count).to_bytes(8, "little", signed=True))
        keys = canonical_name_count_keys(canonicalize_name_parts(raw_first, None, raw_last))
        accepted = False
        for counter, key_name in zip(counters, key_names, strict=True):
            value = keys[key_name]
            if value is not None:
                counter[value] += count
                accepted = True
        if not accepted:
            rejected_rows += 1
    # Keep only the four Counters themselves resident.  Constructing four full
    # filtered dictionaries temporarily doubled this already-large artifact.
    for counter in counters:
        rejected_keys = [key for key, value in counter.items() if value <= 1]
        for key in rejected_keys:
            del counter[key]
    mappings = (counters[0], counters[1], counters[2], counters[3])
    return mappings, {
        "source_row_count": source_rows,
        "selected_rows_sha256": selected_rows_digest.hexdigest(),
        "rejected_row_count": rejected_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limit = _validated_limit(args.limit)
    source_kind = "redshift:content.authors" if args.run_full else f"fixture:{args.fixture_input.resolve()}"
    query_text = _query_text(limit)
    plan = {
        "source_kind": source_kind,
        "source_snapshot_id": args.source_snapshot_id,
        "limit": limit,
        "output_dir": str(args.output_dir.resolve()),
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "query": query_text,
        "cost_context": "internal warehouse scan; inspect the snapshot and limit before authorizing --run-full",
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    manifest_path = args.output_dir / "name_counts_index" / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"published manifest already exists: {manifest_path}; pass --overwrite")
    rows = _query_rows(limit) if args.run_full else _fixture_rows(args.fixture_input, limit)
    mappings, row_metrics = build_name_count_dicts(rows)
    source_snapshot_id = args.source_snapshot_id.strip()
    if not source_snapshot_id:
        raise ValueError("--source-snapshot-id must be nonempty")
    provenance = {
        "schema_version": NAME_COUNTS_PROVENANCE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "generation_id": f"{source_snapshot_id}-{uuid.uuid4().hex}",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_kind": source_kind,
        "source_snapshot_id": source_snapshot_id,
        "source_query_sha256": plan["query_sha256"],
        "cardinalities": dict(
            zip(
                ("first", "last", "first_last", "last_first_initial"),
                (len(mapping) for mapping in mappings),
                strict=True,
            )
        ),
        **row_metrics,
    }
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

    index_path, index_metrics = write_name_counts_index(
        args.output_dir,
        mappings,
        provenance,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {"result": provenance, "name_counts_index": index_path, "name_counts_index_metrics": index_metrics},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
