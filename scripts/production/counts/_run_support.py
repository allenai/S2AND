"""Small boundary checks shared by the count producers."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path


def validate_input_file(path: Path, *, option: str) -> Path:
    """Resolve one required input file before announcing a run plan."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{option} does not exist or is not a file: {resolved}")
    return resolved


def emit_jsonl(payload: object) -> None:
    """Write and flush one machine-readable progress record."""

    print(json.dumps(payload, sort_keys=True), flush=True)


def validate_output_container(output_dir: Path, *, publication_path: Path) -> Path:
    """Resolve an output container and require a fresh publication path."""

    resolved = output_dir.resolve()
    publication = publication_path.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {resolved}")
    if publication.exists():
        raise FileExistsError(f"publication target already exists: {publication}")
    ancestor = resolved
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {ancestor}")
    return resolved


def load_guardrails(path: Path | None, *, fields: Collection[str]) -> dict[str, int]:
    """Load one strict positive-integer guardrail object."""

    if path is None:
        raise ValueError("--guardrails-json is required")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"guardrail file does not exist or is not a file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"guardrail file is invalid JSON: {resolved}") from error
    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"guardrail fields must be exactly {sorted(expected)}")
    for field, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"guardrail {field!r} must be a positive integer")
    return value
