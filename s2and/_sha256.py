"""Shared lexical validation for lowercase SHA-256 digests."""

from __future__ import annotations

import hashlib
import os
from typing import TypeGuard

_LOWERCASE_HEX = frozenset("0123456789abcdef")


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of one file."""

    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def is_lowercase_sha256(value: object) -> TypeGuard[str]:
    """Return whether ``value`` is exactly one lowercase SHA-256 digest."""

    return isinstance(value, str) and len(value) == 64 and all(character in _LOWERCASE_HEX for character in value)
