"""Shared helpers for reusable EPS sweep scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from s2and.arrow_inputs import MissingArrowArtifactError, ValidatedArrowInputs, validate_arrow_prediction_artifacts

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


def resolve_existing_path(raw_path: str | Path, *, manifest_dir: Path) -> Path:
    """Resolve a manifest path against common roots and require it to exist."""

    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [manifest_dir / path, PROJECT_ROOT / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Arrow manifest path does not exist: {raw_path!s}")


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


def load_arrow_paths(arrow_root: Path, dataset: str) -> ValidatedArrowInputs:
    """Load and validate an Arrow dataset's manifest path mapping."""

    dataset_dir = arrow_dataset_dir(arrow_root.resolve(), dataset)
    manifest_path = dataset_dir / "manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise TypeError(f"Arrow manifest must contain an object: {manifest_path}")
    raw_paths = manifest.get("paths")
    if not isinstance(raw_paths, Mapping):
        raise ValueError(f"Arrow manifest is missing paths: {manifest_path}")

    paths = {
        str(key): str(resolve_existing_path(str(value), manifest_dir=dataset_dir)) for key, value in raw_paths.items()
    }
    if "specter" not in paths and "specter2" in paths:
        paths["specter"] = paths["specter2"]
    if "specter_batch_index" not in paths and "specter2_batch_index" in paths:
        paths["specter_batch_index"] = paths["specter2_batch_index"]
    paths["manifest"] = str(manifest_path.resolve())
    try:
        return validate_arrow_prediction_artifacts(
            paths,
            require_specter=True,
            require_name_counts_index=True,
            context=f"EPS sweep Arrow dataset {dataset}",
            producer_hint=(
                "convert the linker replay dataset with scripts/convert_to_arrow.py so the manifest "
                "declares name_counts_index and raw-planner batch indexes"
            ),
        )
    except MissingArrowArtifactError as exc:
        raise FileNotFoundError(str(exc)) from exc
