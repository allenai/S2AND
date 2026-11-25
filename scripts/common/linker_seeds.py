"""Read frozen pre-attachment seed membership."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from s2and._sha256 import sha256_file

GATE_SEED_DIGESTS = "gate_seed_sha256"
PRODUCTION_MAX_EXEMPLARS = 4


def read_seed_components(path: Path) -> dict[str, str]:
    """Read one pre-attachment component assignment per seed signature."""
    rows = pq.read_table(path, columns=["signature_id", "component_key"]).to_pylist()
    components = {row["signature_id"]: row["component_key"] for row in rows}
    if len(components) != len(rows) or any(
        not isinstance(signature, str) or not signature or not isinstance(component, str) or not component
        for signature, component in components.items()
    ):
        raise ValueError(f"Invalid or duplicate seed assignments in {path}")
    return components


def read_frozen_seeds(spec: Mapping[str, Any]) -> dict[str, str]:
    """Check the frozen seed receipt before reading the assignments."""
    path = Path(spec["seeds"])
    if sha256_file(path) != spec["seeds_sha256"]:
        raise ValueError(f"Frozen linker seed assignments changed: {path}")
    return read_seed_components(path)


def members_by_component(seeds: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
    """Group signature IDs by component, preserving frozen seed order."""
    members: dict[str, list[str]] = defaultdict(list)
    for signature, component in seeds.items():
        members[component].append(signature)
    return {component: tuple(signatures) for component, signatures in members.items()}
