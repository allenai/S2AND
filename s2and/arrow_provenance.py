"""Conversion provenance binding an Arrow dataset to the source files it was converted from.

An Arrow manifest proves that the Arrow tables are intact, but not which JSON
inputs produced them. ``scripts/convert_to_arrow.py`` writes a provenance record
next to each converted dataset. Release commands that featurize frozen JSON
inputs through Rust verify that record first, so stale Arrow data with the same
signature IDs but different names, papers or embeddings is rejected.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from s2and._sha256 import is_lowercase_sha256, sha256_file

PROVENANCE_FILENAME = "conversion_provenance.json"
PROVENANCE_KIND = "s2and_arrow_conversion_provenance"
PROVENANCE_FORMAT_VERSION = 1
_PROVENANCE_FIELDS = frozenset({"kind", "format_version", "sources", "arrow_files", "converter"})
# The shared name-count index is verified against the release artifact authority
# separately, so rebinding it (as the public-root assembler does) keeps provenance valid.
_UNBOUND_ARROW_FILE_KEYS = frozenset({"name_counts_index"})


class ArrowProvenanceError(ValueError):
    """An Arrow dataset's conversion provenance does not bind it to the expected sources."""


def _arrow_table_files(dataset_dir: Path) -> dict[str, str]:
    """Return the manifest's content hashes for every artifact provenance binds."""

    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        str(key): str(entry["sha256"])
        for key, entry in sorted(manifest["files"].items())
        if key not in _UNBOUND_ARROW_FILE_KEYS
    }


def write_conversion_provenance(
    dataset_dir: Path,
    *,
    sources: Mapping[str, Path],
    converter: Mapping[str, Any],
) -> Path:
    """Write the provenance record for one freshly converted Arrow dataset.

    Args:
        dataset_dir: Converted dataset directory containing its final ``manifest.json``.
        sources: Source files the converter read, keyed by release plan role
            (``signatures``, ``papers``, ``specter_embeddings`` and optionally ``clusters``).
        converter: Converter revision and options that shaped the output.

    Returns:
        Path of the written record.
    """

    payload = {
        "kind": PROVENANCE_KIND,
        "format_version": PROVENANCE_FORMAT_VERSION,
        "sources": {
            role: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for role, path in sorted(sources.items())
        },
        "arrow_files": _arrow_table_files(dataset_dir),
        "converter": dict(converter),
    }
    record_path = dataset_dir / PROVENANCE_FILENAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=dataset_dir,
            prefix=f".{PROVENANCE_FILENAME}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path.replace(record_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return record_path


def verify_conversion_provenance(
    dataset_dir: Path,
    *,
    expected_source_sha256: Mapping[str, str],
) -> str:
    """Prove that ``dataset_dir`` was converted from the expected source files.

    Args:
        dataset_dir: Arrow dataset directory to verify.
        expected_source_sha256: Frozen source digests, keyed by release plan role.
            Every role listed must appear in the record with the same digest.

    Returns:
        SHA-256 of the provenance record, for reproducibility metadata.

    Raises:
        ArrowProvenanceError: The record is missing or malformed, names different
            sources, or describes different Arrow tables than the manifest.
    """

    record_path = dataset_dir / PROVENANCE_FILENAME
    if not record_path.is_file():
        raise ArrowProvenanceError(f"Arrow dataset {dataset_dir} has no conversion provenance record ({record_path})")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or set(record) != _PROVENANCE_FIELDS:
        raise ArrowProvenanceError(f"Arrow conversion provenance {record_path} has unexpected fields")
    if record["kind"] != PROVENANCE_KIND or record["format_version"] != PROVENANCE_FORMAT_VERSION:
        raise ArrowProvenanceError(f"Arrow conversion provenance {record_path} has an unsupported kind or version")
    sources = record["sources"]
    for role, expected in sorted(expected_source_sha256.items()):
        if not is_lowercase_sha256(expected):
            raise ValueError(f"expected {role} digest must be a lowercase SHA-256, got {expected!r}")
        entry = sources.get(role) if isinstance(sources, dict) else None
        recorded = entry.get("sha256") if isinstance(entry, dict) else None
        if recorded != expected:
            raise ArrowProvenanceError(
                f"Arrow dataset {dataset_dir} was converted from a different {role} source: "
                f"recorded_sha256={recorded!r} expected_sha256={expected!r}"
            )
    if record["arrow_files"] != _arrow_table_files(dataset_dir):
        raise ArrowProvenanceError(
            f"Arrow dataset {dataset_dir} tables differ from the conversion recorded in {record_path}"
        )
    return sha256_file(record_path)


__all__ = [
    "PROVENANCE_FILENAME",
    "ArrowProvenanceError",
    "verify_conversion_provenance",
    "write_conversion_provenance",
]
