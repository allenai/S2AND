"""Reference name counts for the Parquet and Athena producer tests.

The producers aggregate in DuckDB. This plain Python version gives the expected
counts with the same canonical keys and the same drop of keys with a total of one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from s2and.text import canonical_name_count_keys, canonicalize_name_parts

KEY_NAMES = ("first", "last", "first_last", "last_first_initial")


def build_name_count_dicts(
    rows: Iterable[tuple[str, str, int]],
) -> tuple[tuple[Counter[str], Counter[str], Counter[str], Counter[str]], dict[str, int]]:
    """Return the four canonical count maps and the source and rejected row counts."""
    counters: tuple[Counter[str], Counter[str], Counter[str], Counter[str]] = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    source_rows = 0
    rejected_rows = 0
    for raw_first, raw_last, count in rows:
        source_rows += 1
        keys = canonical_name_count_keys(canonicalize_name_parts(raw_first, None, raw_last))
        accepted = False
        for counter, key_name in zip(counters, KEY_NAMES, strict=True):
            value = keys[key_name]
            if value is not None:
                counter[value] += count
                accepted = True
        if not accepted:
            rejected_rows += 1
    for counter in counters:
        for key in [key for key, value in counter.items() if value <= 1]:
            del counter[key]
    return counters, {"source_row_count": source_rows, "rejected_row_count": rejected_rows}
