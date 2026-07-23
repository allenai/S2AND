"""Stable digests shared by incremental-linker training and runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from s2and.incremental_linking.features import promoted_linker_feature_columns

CONTRACT_SCHEMA_VERSION = "incremental_linking_contract_v1"
DEFAULT_RETRIEVAL_TOP_K = 25


def canonical_json_digest(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def promoted_linker_feature_schema_payload(feature_columns: Sequence[str] | None = None) -> dict[str, Any]:
    """Return the feature schema used to identify materialized training data."""

    columns = tuple(promoted_linker_feature_columns() if feature_columns is None else feature_columns)
    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_schema": "promoted_53_round6_backward_shap_corr_weighted_pairwise_agg",
        "feature_count": len(columns),
        "feature_columns": list(columns),
    }


def promoted_linker_feature_schema_digest(feature_columns: Sequence[str] | None = None) -> str:
    """Return the stable feature-schema digest."""

    return canonical_json_digest(promoted_linker_feature_schema_payload(feature_columns))
