"""Regenerate the name-tuple alias artifact under canonical_v2 normalization.

Migration step 3 of docs/normalization_migration_blocked.md: name tuples are
regenerated deterministically by re-normalizing the curated raw pairs in
``s2and/data/s2and_unnormalized_filtered_name_tuples.txt`` through the canonical
normalizer, instead of re-running the archived hmni/LLM pipeline.

Policy:
- Each side is canonicalized with ``s2and.text.canonicalize_name_text`` (whole-string
  canonical tokens: apostrophe-like marks deleted, dash-like characters uniform
  separators, transliterated). Aliases are complete first-name strings, so the
  first/middle split does not apply.
- Pairs are emitted symmetrically (both directions), deduplicated, and sorted.
  ``insert_name_tuple_alias`` in s2and_rust/src/ingest_dataset.rs is directional
  and relies on both directions being present.
- Pairs that collapse to identity or become prefix-compatible under
  ``same_prefix_tokens`` are dropped: the runtime checks prefix compatibility
  before consulting tuples, so such entries are dead weight.

Usage:
    uv run python scripts/production/generate_canonical_name_tuples.py [--output PATH]

Default output is ``s2and/data/s2and_name_tuples_canonical.txt`` with a JSON
provenance sidecar. Ship it in the same release unit as the other canonical_v2
artifacts; the loader keeps the "name1,name2" line format.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.text import canonicalize_name_text, same_prefix_tokens

SOURCE_FILENAME = "s2and_unnormalized_filtered_name_tuples.txt"
DEFAULT_OUTPUT_FILENAME = "s2and_name_tuples_canonical.txt"


def regenerate(source_path: str, output_path: str) -> dict:
    raw_pairs: list[tuple[str, str]] = []
    with open(source_path, encoding="utf-8") as source_file:
        for line in source_file:
            fields = line.strip().split(",")
            if len(fields) >= 2 and fields[0] and fields[1]:
                raw_pairs.append((fields[0], fields[1]))

    canonical_pairs: set[tuple[str, str]] = set()
    dropped_identity = 0
    dropped_prefix_compatible = 0
    dropped_empty = 0
    for raw_a, raw_b in raw_pairs:
        name_a = canonicalize_name_text(raw_a)
        name_b = canonicalize_name_text(raw_b)
        if not name_a or not name_b:
            dropped_empty += 1
            continue
        if name_a == name_b:
            dropped_identity += 1
            continue
        if same_prefix_tokens(name_a, name_b):
            dropped_prefix_compatible += 1
            continue
        canonical_pairs.add((name_a, name_b))
        canonical_pairs.add((name_b, name_a))

    with open(output_path, "w", encoding="utf-8", newline="\n") as output_file:
        for name_a, name_b in sorted(canonical_pairs):
            output_file.write(f"{name_a},{name_b}\n")

    metadata = {
        "normalization_version": "canonical_v2",
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": os.path.basename(source_path),
        "input_lines": len(raw_pairs),
        "output_pairs_directed": len(canonical_pairs),
        "dropped_identity": dropped_identity,
        "dropped_prefix_compatible": dropped_prefix_compatible,
        "dropped_empty": dropped_empty,
    }
    with open(output_path + ".meta.json", "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=os.path.join(_PACKAGE_DATA_DIR, SOURCE_FILENAME))
    parser.add_argument("--output", default=os.path.join(_PACKAGE_DATA_DIR, DEFAULT_OUTPUT_FILENAME))
    args = parser.parse_args()
    metadata = regenerate(args.source, args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
