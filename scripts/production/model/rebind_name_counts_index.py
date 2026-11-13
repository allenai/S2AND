"""Bind a copy of a complete model to another name-count index without another fit.

Use this after a rebuild of the index that leaves the model's training features in place,
such as a change to the romanization of names outside the training data. The copy keeps the
boosters, the linker, and every clustering setting. Only the recorded index hash, the
linker's digest of the pairwise contract, the release version, and their checksums change.
Score the copy with ``release_pairwise.py evaluate-heldout`` before publishing it.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s2and._sha256 import sha256_file
from s2and.incremental_linking.policy import NAME_COUNTS_MANIFEST_SHA256_FIELD
from s2and.name_counts_index import NameCountsIndex
from s2and.production_bundle import _staged_production_bundle
from s2and.production_model import (
    _featurization_info_from_payload,
    _pairwise_binding_from_validated_parts,
    canonical_json_digest,
    load_production_model,
)

# The only files a rebind may change.
_REBOUND_FILES = frozenset({"clusterer.json", "incremental_linker/metadata.json"})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def rebind_name_counts_index(
    *,
    source_model: Path,
    name_counts_index: Path,
    output_model: Path,
    release_version: str,
) -> dict[str, Any]:
    """Write a copy of ``source_model`` bound to ``name_counts_index``.

    Args:
        source_model: Complete model bundle to copy.
        name_counts_index: Index directory the copy must name.
        output_model: New bundle directory; it must not exist.
        release_version: Release version of the copy.

    Returns:
        A report of the old and new index hashes, versions, and manifest hashes.

    Raises:
        FileExistsError: If ``output_model`` exists.
        ValueError: If the index is the one the source already names.
        RuntimeError: If a file outside the bound metadata changes.
    """
    source, output = source_model.resolve(), output_model.resolve()
    if output.exists():
        raise FileExistsError(f"Output model already exists: {output}")
    load_production_model(source)
    new_hash = NameCountsIndex.open(name_counts_index, verify=True).manifest_sha256
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((source / "clusterer.json").read_text(encoding="utf-8"))
    old_hash = config["feature_contract"][NAME_COUNTS_MANIFEST_SHA256_FIELD]
    if old_hash == new_hash:
        raise ValueError(f"The source model already names this index: {new_hash}")
    old_version = manifest["release_version"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with _staged_production_bundle(output) as staging:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        config["feature_contract"][NAME_COUNTS_MANIFEST_SHA256_FIELD] = new_hash
        _write_json(staging / "clusterer.json", config)
        # The linker records a digest of the pairwise contract, which holds the index hash.
        binding = _pairwise_binding_from_validated_parts(
            manifest,
            config,
            _featurization_info_from_payload(config["featurizer_info"]),
            _featurization_info_from_payload(config["nameless_featurizer_info"]),
        )
        linker_path = staging / "incremental_linker" / "metadata.json"
        linker = json.loads(linker_path.read_text(encoding="utf-8"))
        linker["pairwise_bundle_binding_digest"] = canonical_json_digest(binding)
        _write_json(linker_path, linker)
        manifest["release_version"] = release_version
        for relative in _REBOUND_FILES:
            manifest["sha256"][relative] = sha256_file(staging / relative)
        _write_json(staging / "manifest.json", manifest)
        for relative, expected in manifest["sha256"].items():
            if sha256_file(staging / relative) != expected:
                raise RuntimeError(f"Rebinding changes a trained artifact: {relative}")
        load_production_model(staging)
    return {
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "output_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_name_counts_manifest_sha256": old_hash,
        "output_name_counts_manifest_sha256": new_hash,
        "source_release_version": old_version,
        "release_version": release_version,
        "changed_files": sorted(_REBOUND_FILES | {"manifest.json"}),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--name-counts-index", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--release-version", required=True)
    args = parser.parse_args(argv)
    report = rebind_name_counts_index(
        source_model=args.source_model,
        name_counts_index=args.name_counts_index,
        output_model=args.output_model,
        release_version=args.release_version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
