"""Generate one immutable canonical name-count index."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from s2and.text import canonical_name_count_keys, canonicalize_name_parts

from ._run_support import (
    load_guardrails,
    require_positive,
    validate_fixture_path,
    validate_output_container,
)

GUARDRAIL_FIELDS = frozenset(
    {
        "max_source_rows",
        "min_source_rows",
        "max_keys_per_mapping",
        "min_keys_per_mapping",
    }
)
PROGRESS_EVERY = 100_000

NameCountRow = tuple[str, str, int]
NameCountMappings = tuple[
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path, help="Reviewed full warehouse export")
    source.add_argument("--fixture-input", type=Path, help="Local JSON row fixture")
    parser.add_argument("--limit", type=int, help="Fixture-only row limit")
    parser.add_argument("--guardrails-json", type=Path, help="Reviewed full-run bounds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _reviewed_csv_rows(path: Path) -> Iterator[NameCountRow]:
    """Stream one reviewed name-count export."""

    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames != ["first_name", "last_name", "count"]:
            raise ValueError("reviewed export columns must be exactly first_name,last_name,count")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"reviewed export row {row_number} does not match its header")
            count = int(row["count"])
            if count < 1:
                raise ValueError(f"reviewed export row {row_number} count must be a positive integer")
            yield row["first_name"], row["last_name"], count


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
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"fixture row {row_index} count must be a positive integer")
        yield first, last, count


def build_name_count_dicts(
    rows: Iterable[NameCountRow],
    *,
    max_source_rows: int | None = None,
    max_keys_per_mapping: int | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> tuple[NameCountMappings, dict[str, int]]:
    """Canonicalize source rows while enforcing the two live size bounds."""

    counters: list[Counter[str]] = [Counter(), Counter(), Counter(), Counter()]
    key_names = ("first", "last", "first_last", "last_first_initial")
    source_rows = 0
    rejected_rows = 0
    for raw_first, raw_last, count in rows:
        source_rows += 1
        if max_source_rows is not None and source_rows > max_source_rows:
            raise ValueError(f"source rows exceeded guardrail max_source_rows={max_source_rows}")

        keys = canonical_name_count_keys(canonicalize_name_parts(raw_first, None, raw_last))
        accepted = False
        for counter, key_name in zip(counters, key_names, strict=True):
            value = keys[key_name]
            if value is not None:
                counter[value] += count
                accepted = True
        if not accepted:
            rejected_rows += 1
        cardinalities = tuple(len(counter) for counter in counters)
        if max_keys_per_mapping is not None and max(cardinalities, default=0) > max_keys_per_mapping:
            raise ValueError(f"a canonical mapping exceeded guardrail max_keys_per_mapping={max_keys_per_mapping}")
        if progress_callback is not None and source_rows % PROGRESS_EVERY == 0:
            progress_callback(
                {
                    "source_rows": source_rows,
                    "rejected_rows": rejected_rows,
                    **dict(zip((f"{name}_keys" for name in key_names), cardinalities, strict=True)),
                }
            )

    for counter in counters:
        for key in [key for key, value in counter.items() if value <= 1]:
            del counter[key]
    return (counters[0], counters[1], counters[2], counters[3]), {
        "source_row_count": source_rows,
        "rejected_row_count": rejected_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limit = require_positive(args.limit, option="--limit")
    full_input = args.input_csv.resolve() if args.input_csv is not None else None
    if full_input is not None and limit is not None:
        raise ValueError("--limit is fixture-only; it does not bound a reviewed export")

    publication = args.output_dir / "name_counts_index"
    output_dir = validate_output_container(args.output_dir, publication_path=publication)
    fixture = validate_fixture_path(args.fixture_input) if args.fixture_input is not None else None
    guardrails = load_guardrails(args.guardrails_json, fields=GUARDRAIL_FIELDS) if full_input is not None else None
    if guardrails is not None:
        if guardrails["min_source_rows"] > guardrails["max_source_rows"]:
            raise ValueError("guardrail min_source_rows must not exceed max_source_rows")
        if guardrails["min_keys_per_mapping"] > guardrails["max_keys_per_mapping"]:
            raise ValueError("guardrail min_keys_per_mapping must not exceed max_keys_per_mapping")

    plan = {
        "mode": "full" if full_input is not None else "fixture",
        "source": str(full_input or fixture),
        "output_dir": str(output_dir),
        "guardrails": guardrails,
        "limit": limit,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    if full_input is not None:
        rows = _reviewed_csv_rows(full_input)
    else:
        assert fixture is not None
        rows = _fixture_rows(fixture, limit)

    def report_progress(metrics: dict[str, int]) -> None:
        print(json.dumps({"event": "name_counts_progress", **metrics}, sort_keys=True))

    mappings, row_metrics = build_name_count_dicts(
        rows,
        max_source_rows=None if guardrails is None else guardrails["max_source_rows"],
        max_keys_per_mapping=None if guardrails is None else guardrails["max_keys_per_mapping"],
        progress_callback=report_progress,
    )
    cardinalities = dict(
        zip(
            ("first", "last", "first_last", "last_first_initial"),
            (len(mapping) for mapping in mappings),
            strict=True,
        )
    )
    source_row_count = int(row_metrics["source_row_count"])
    if source_row_count == 0:
        raise RuntimeError("name-count generation selected zero source rows")
    if sum(cardinalities.values()) == 0:
        raise RuntimeError("name-count generation produced zero retained canonical keys")
    if guardrails is not None:
        if source_row_count < guardrails["min_source_rows"]:
            raise RuntimeError(
                f"source rows {source_row_count} are below guardrail min_source_rows={guardrails['min_source_rows']}"
            )
        below_floor = {key: count for key, count in cardinalities.items() if count < guardrails["min_keys_per_mapping"]}
        if below_floor:
            raise RuntimeError(
                "mapping cardinalities are below guardrail "
                f"min_keys_per_mapping={guardrails['min_keys_per_mapping']}: {below_floor}"
            )

    result = {
        "cardinalities": cardinalities,
        **row_metrics,
    }
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

    index_path, index_metrics = write_name_counts_index(output_dir, mappings)
    print(
        json.dumps(
            {"result": result, "name_counts_index": index_path, "name_counts_index_metrics": index_metrics},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
