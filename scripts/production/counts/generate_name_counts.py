"""Generate one immutable canonical name-count index."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from s2and.text import canonical_name_count_keys, canonicalize_name_parts

from ._run_support import (
    emit_jsonl,
    load_guardrails,
    validate_input_file,
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
    parser.add_argument("--input-csv", type=Path, required=True, help="Reviewed warehouse export")
    parser.add_argument("--guardrails-json", type=Path, required=True, help="Reviewed run bounds")
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
    source = validate_input_file(args.input_csv, option="--input-csv")
    publication = args.output_dir / "name_counts_index"
    output_dir = validate_output_container(args.output_dir, publication_path=publication)
    guardrails = load_guardrails(args.guardrails_json, fields=GUARDRAIL_FIELDS)
    if guardrails["min_source_rows"] > guardrails["max_source_rows"]:
        raise ValueError("guardrail min_source_rows must not exceed max_source_rows")
    if guardrails["min_keys_per_mapping"] > guardrails["max_keys_per_mapping"]:
        raise ValueError("guardrail min_keys_per_mapping must not exceed max_keys_per_mapping")

    plan = {
        "source": str(source),
        "output_dir": str(output_dir),
        "guardrails": guardrails,
    }
    emit_jsonl({"event": "name_counts_plan", "plan": plan})
    rows = _reviewed_csv_rows(source)

    def report_progress(metrics: dict[str, int]) -> None:
        emit_jsonl({"event": "name_counts_progress", **metrics})

    mappings, row_metrics = build_name_count_dicts(
        rows,
        max_source_rows=guardrails["max_source_rows"],
        max_keys_per_mapping=guardrails["max_keys_per_mapping"],
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
    emit_jsonl(
        {
            "event": "name_counts_result",
            "result": result,
            "name_counts_index": index_path,
            "name_counts_index_metrics": index_metrics,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
