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
- Each unordered pair is emitted once with its fields in lexicographic order.
  Consumers construct any symmetric runtime lookup when loading the artifact.
- Pairs that collapse to identity or become prefix-compatible under
  ``same_prefix_tokens`` are dropped: the runtime checks prefix compatibility
  before consulting tuples, so such entries are dead weight.

Usage:
    uv run python scripts/production/generate_canonical_name_tuples.py [--output PATH]

Default output is ``s2and/data/s2and_name_tuples_canonical.txt`` with a JSON
provenance sidecar under the strict ``s2and_name_tuples_v3`` contract. Data is
replaced first and the fsynced sidecar last as the generation commit marker.
Ship both in the same release unit as the other canonical_v2 artifacts; the
loader keeps the "name1,name2" line format. Generate into an offline staging
location and validate it before package promotion: two same-directory files
cannot be replaced atomically, so interruption between replacements is
fail-closed (checksum mismatch) and requires rerunning generation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_tuple_artifact import build_name_tuple_artifact_metadata
from s2and.text import canonical_name_tuple_pair, canonicalize_name_text, same_prefix_tokens

SOURCE_FILENAME = "s2and_unnormalized_filtered_name_tuples.txt"
DEFAULT_OUTPUT_FILENAME = "s2and_name_tuples_canonical.txt"


def _write_fsynced_temp(destination: Path, payload: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def regenerate(source_path: str, output_path: str) -> dict:
    source = Path(source_path)
    output = Path(output_path)
    source_bytes = source.read_bytes()
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Name-tuple source is not valid UTF-8: {source}") from exc

    raw_pairs: list[tuple[str, str]] = []
    for line_number, line in enumerate(source_text.splitlines(), start=1):
        fields = line.split(",")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"Invalid source tuple at {source}:{line_number}: expected two nonempty fields")
        raw_pairs.append((fields[0], fields[1]))

    canonical_pairs: set[tuple[str, str]] = set()
    dropped_identity = 0
    dropped_prefix_compatible = 0
    dropped_empty = 0
    dropped_duplicate_canonical = 0
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
        canonical_pair = canonical_name_tuple_pair(name_a, name_b)
        if canonical_pair in canonical_pairs:
            dropped_duplicate_canonical += 1
        else:
            canonical_pairs.add(canonical_pair)

    ordered_pairs = sorted(canonical_pairs)
    data_bytes = "".join(f"{name_a},{name_b}\n" for name_a, name_b in ordered_pairs).encode("utf-8")
    metadata = build_name_tuple_artifact_metadata(
        source_filename=source.name,
        source_bytes=source_bytes,
        data_filename=output.name,
        data_bytes=data_bytes,
        pair_count=len(ordered_pairs),
        generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
        input_pair_count=len(raw_pairs),
        dropped_identity=dropped_identity,
        dropped_prefix_compatible=dropped_prefix_compatible,
        dropped_empty=dropped_empty,
        dropped_duplicate_canonical=dropped_duplicate_canonical,
    )
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")

    metadata_path = output.with_name(output.name + ".meta.json")
    with exclusive_file_lock(output.with_name(f".{output.name}.publish.lock")):
        data_temp = _write_fsynced_temp(output, data_bytes)
        metadata_temp = _write_fsynced_temp(metadata_path, metadata_bytes)
        try:
            # The sidecar is the commit marker. The lock serializes writers;
            # readers either see one complete generation or fail its checksum.
            os.replace(data_temp, output)
            fsync_directory(output.parent)
            os.replace(metadata_temp, metadata_path)
            fsync_directory(output.parent)
        finally:
            data_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
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
