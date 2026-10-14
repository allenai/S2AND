"""Read fallback-label exclusions without a change to model-plan file roles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LabeledPair = tuple[str, str, int | float]


@dataclass(frozen=True)
class UnknownSignatures:
    """Bind unknown IDs to the exact export sidecar bytes."""

    ids: frozenset[str]
    path: Path | None = None
    sha256: str | None = None

    def binding(self) -> dict[str, str | int | None]:
        """Return the source identity for the training configuration."""
        return {
            "path": None if self.path is None else str(self.path),
            "sha256": self.sha256,
            "signature_count": len(self.ids),
        }


def load_unknown_signatures(files: Mapping[str, Path]) -> UnknownSignatures:
    """Read the uncertainty list beside the source labels.

    Legacy datasets without a sidecar have no declared unknown IDs.
    Multiple sidecars require an explicit source-directory correction.
    """
    label_path = files["clusters"] if "clusters" in files else files["signatures"]
    candidates = sorted(label_path.parent.glob("restored_unknown_signature_ids_*.json"))
    if not candidates:
        return UnknownSignatures(frozenset())
    if len(candidates) != 1:
        raise ValueError(f"Multiple unknown-signature sidecars beside {label_path}")
    path = candidates[0].resolve()
    contents = path.read_bytes()
    values = json.loads(contents)
    if not isinstance(values, list):
        raise ValueError(f"Unknown-signature sidecar must contain a list of nonempty string IDs: {path}")
    ids: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"Unknown-signature sidecar must contain a list of nonempty string IDs: {path}")
        if value in ids:
            raise ValueError(f"Unknown-signature sidecar contains duplicate IDs: {path}")
        ids.add(value)
    return UnknownSignatures(frozenset(ids), path, hashlib.sha256(contents).hexdigest())


def load_declared_unknown_signatures(
    files: Mapping[str, Path],
    declared: Path | None,
    *,
    expected_sha256: str | None = None,
) -> UnknownSignatures:
    """Read score exclusions only from the sidecar that the evaluation authority declares.

    Args:
        files: Label files; the sidecar must sit beside ``clusters`` or ``signatures``.
        declared: The declared sidecar, or ``None`` for a declared absence.
        expected_sha256: The frozen digest of the declared sidecar.

    Returns:
        The declared exclusions, or no exclusions for a declared absence.

    Raises:
        ValueError: If a sidecar is added, removed, moved, or changed after the declaration.
    """
    source = load_unknown_signatures(files)
    if declared is None:
        if source.path is not None:
            raise ValueError(f"Undeclared unknown-signature sidecar beside the labels: {source.path}")
        return source
    if source.path is None:
        raise ValueError(f"Declared unknown-signature sidecar is missing beside the labels: {declared}")
    if source.path != Path(declared).resolve():
        raise ValueError(f"Declared unknown-signature sidecar is not the sidecar beside the labels: {declared}")
    if expected_sha256 is not None and source.sha256 != expected_sha256:
        raise ValueError(f"Unknown-signature sidecar changed after its declaration: {declared}")
    return source


def same_unknown_content(binding: Any, source: UnknownSignatures) -> bool:
    """Compare a recorded sidecar binding with a loaded sidecar, ignoring its location.

    Args:
        binding: One ``UnknownSignatures.binding()`` value from a training record.
        source: The sidecar loaded beside the current labels.

    Returns:
        Whether the recorded and current sidecars have the same bytes and ID count.
    """
    if not isinstance(binding, Mapping) or {"sha256", "signature_count"} - set(binding):
        return False
    current = source.binding()
    return binding["sha256"] == current["sha256"] and binding["signature_count"] == current["signature_count"]


def augmented_unknown_ids(unknown_ids: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """Exclude augmented copies of each unknown benchmark signature."""
    return unknown_ids.get("augmented", frozenset()) | frozenset(
        f"{name}___{signature_id}" for name, ids in unknown_ids.items() if name != "augmented" for signature_id in ids
    )


def exclude_unknown_pairs(pairs: Iterable[LabeledPair], unknown_ids: frozenset[str]) -> list[LabeledPair]:
    """Preserve selected pair order and labels except unknown endpoints."""
    return [
        (str(left), str(right), label)
        for left, right, label in pairs
        if str(left) not in unknown_ids and str(right) not in unknown_ids
    ]
