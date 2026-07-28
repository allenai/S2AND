"""Shared helpers for reusable EPS sweep scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINKER_BUNDLE_ROOT = PROJECT_ROOT / "s2and" / "data" / "s2and_and_big_blocks_linker_dataset_20260513"
DEFAULT_ARROW_ROOT = PROJECT_ROOT / "s2and" / "data" / "s2and_and_big_blocks_linker_dataset_20260525"
DEFAULT_GOLD_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_gold"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "linking_eps_sweeps_arrow"


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with a trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha1_text(value: str) -> str:
    """Return a SHA1 digest for stable cache keys."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def json_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable digest for a JSON-serializable mapping."""

    return sha1_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def arrow_dataset_dir(arrow_root: Path, dataset: str) -> Path:
    """Return the dataset directory for either supported Arrow root layout."""

    candidates = [
        arrow_root / "datasets" / dataset,
        arrow_root / dataset,
    ]
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate.resolve()
    formatted = ", ".join(str(path / "manifest.json") for path in candidates)
    raise FileNotFoundError(f"No Arrow manifest found for dataset={dataset!r}; checked {formatted}")
