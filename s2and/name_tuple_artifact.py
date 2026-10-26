"""Loader for canonical first-name alias pairs."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.text import canonicalize_name_text, same_prefix_tokens


@dataclass(frozen=True)
class NameTupleArtifact:
    """Validated immutable alias pairs and their content identity."""

    pairs: frozenset[tuple[str, str]]
    data_sha256: str


def _parse_and_validate_pairs(data_bytes: bytes, *, data_path: Path) -> frozenset[tuple[str, str]]:
    """Parse canonical, sorted alias pairs from one trusted release file."""

    pairs: set[tuple[str, str]] = set()
    previous: tuple[str, str] | None = None
    for line_number, raw_line in enumerate(io.BytesIO(data_bytes), start=1):
        raw_line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Name-tuple artifact {data_path} is not valid UTF-8 at line {line_number}") from exc
        fields = line.split(",")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise ValueError(f"Invalid name tuple at {data_path}:{line_number}: expected two nonempty fields")
        pair = (fields[0], fields[1])
        if previous is not None and pair <= previous:
            raise ValueError(
                f"Invalid name tuple ordering at {data_path}:{line_number}: rows must be unique and sorted by fields"
            )
        previous = pair
        first_a, first_b = pair
        if canonicalize_name_text(first_a) != first_a or canonicalize_name_text(first_b) != first_b:
            raise ValueError(f"Invalid noncanonical name tuple at {data_path}:{line_number}")
        if first_a == first_b:
            raise ValueError(f"Invalid identity name tuple at {data_path}:{line_number}")
        if first_a > first_b:
            raise ValueError(
                f"Invalid name tuple field order at {data_path}:{line_number}: "
                "name_a must be lexicographically less than name_b"
            )
        if same_prefix_tokens(first_a, first_b):
            raise ValueError(f"Invalid prefix-compatible name tuple at {data_path}:{line_number}")
        pairs.add(pair)
    return frozenset(pairs)


def load_name_tuple_artifact(path: str | Path) -> NameTupleArtifact:
    """Load canonical alias pairs from one trusted release file."""

    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Name-tuple artifact does not exist: {data_path}")
    data_bytes = data_path.read_bytes()
    return NameTupleArtifact(
        pairs=_parse_and_validate_pairs(data_bytes, data_path=data_path),
        data_sha256=hashlib.sha256(data_bytes).hexdigest(),
    )


@lru_cache(maxsize=1)
def load_packaged_name_tuple_artifact() -> NameTupleArtifact:
    """Validate and retain the immutable packaged canonical artifact once."""

    return load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt")
