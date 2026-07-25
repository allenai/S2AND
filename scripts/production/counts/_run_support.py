"""Small boundary checks shared by the count producers."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

MAX_FIXTURE_BYTES = 64 * 1024**2


def require_positive(value: int | None, *, option: str) -> int | None:
    """Validate one optional positive integer."""

    if value is not None and value < 1:
        raise ValueError(f"{option} must be positive")
    return value


def validate_fixture_path(path: Path | None) -> Path:
    """Resolve one existing, reasonably sized fixture."""

    if path is None:
        raise ValueError("fixture input is required")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"fixture input does not exist or is not a file: {resolved}")
    size = resolved.stat().st_size
    if size > MAX_FIXTURE_BYTES:
        raise ValueError(f"fixture input is {size} bytes; limit is {MAX_FIXTURE_BYTES}")
    return resolved


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
        raise ValueError("--guardrails-json is required for --run-full")
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
