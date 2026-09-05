"""Shared helpers for reusable EPS sweep scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from s2and.arrow_inputs import read_arrow_collection_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GOLD_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_gold"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_sweeps_arrow"


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha1_text(value: str) -> str:
    """Return a SHA1 digest for stable cache keys."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def arrow_dataset_dir(arrow_root: Path, dataset: str) -> Path:
    """Return a dataset directory declared by a validated Arrow root."""

    dataset_manifests, _replay_bundles, _release_version = read_arrow_collection_root(arrow_root / "manifest.json")
    manifest_path = dataset_manifests.get(dataset)
    if manifest_path is None:
        raise ValueError(f"Arrow root does not declare dataset={dataset!r}")
    return manifest_path.parent
