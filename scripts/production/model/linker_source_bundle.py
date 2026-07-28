"""Assemble the reviewed linker source bundle and public data root."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.incremental_linking_training.classic import load_bundle  # noqa: E402
from scripts.convert_to_arrow import _write_root_manifest  # noqa: E402
from scripts.production.model.train_linker_and_finalize import (  # noqa: E402
    _preflight_source_rows,
    _validate_source_bundle_support_files,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _dataset_manifests(root: Path) -> list[Mapping[str, Any]]:
    entries = _read_json(root / "manifest.json").get("dataset_manifests")
    if not isinstance(entries, list) or not entries or not all(isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"{root / 'manifest.json'} must contain dataset_manifests")
    return entries


def assemble_source_bundle(
    *,
    source_root: Path,
    benchmark_arrow_root: Path,
    replay_arrow_root: Path,
    output_source_bundle: Path,
    output_data_root: Path,
) -> dict[str, Any]:
    """Copy reviewed inputs into fresh source and public-data roots."""

    source_root, benchmark_arrow_root, replay_arrow_root = (
        path.resolve() for path in (source_root, benchmark_arrow_root, replay_arrow_root)
    )
    output_source_bundle, output_data_root = output_source_bundle.resolve(), output_data_root.resolve()
    if output_source_bundle.exists() or output_data_root.exists():
        raise FileExistsError("assembly output directories must not exist")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    shutil.copytree(replay_arrow_root, output_source_bundle)
    shutil.copytree(source_root, output_source_bundle, dirs_exist_ok=True)

    shutil.copytree(benchmark_arrow_root, output_data_root)
    shutil.copytree(replay_arrow_root, output_data_root / "linker_replay")
    _write_root_manifest(
        output_data_root,
        dataset_manifests=_dataset_manifests(benchmark_arrow_root),
        replay_bundles=[{"bundle": "linker_replay", "manifest_path": "linker_replay/manifest.json"}],
    )

    bundle = load_bundle(output_source_bundle)
    _validate_source_bundle_support_files(bundle)
    with ExitStack() as arrow_stack:
        selected_source_rows, _arrow_datasets = _preflight_source_rows(
            bundle,
            name_counts_index_root=None,
            arrow_stack=arrow_stack,
        )
    return {
        "data_root": str(output_data_root),
        "selected_source_rows": selected_source_rows,
        "source_bundle": str(output_source_bundle),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "source-root",
        "benchmark-arrow-root",
        "replay-arrow-root",
        "output-source-bundle",
        "output-data-root",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    print(json.dumps(assemble_source_bundle(**vars(parser.parse_args(argv))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
