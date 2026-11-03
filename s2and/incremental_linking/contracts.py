"""Stable digests shared by incremental-linker training and runtime."""

from __future__ import annotations

import hashlib
import json
from typing import Any

DEFAULT_RETRIEVAL_TOP_K = 25
INCREMENTAL_LINKER_FORMAT_VERSION = 1
# Feature and retrieval semantics require a new version even when column names stay equal.
INCREMENTAL_LINKER_FEATURE_CONTRACT_VERSION = 1


def canonical_json_digest(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
