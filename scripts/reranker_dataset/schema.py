"""Feature schema contracts for reranker training and bundle validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

FEATURE_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FeatureSchema:
    """Stable feature-column contract for ranker and calibrator artifacts."""

    feature_columns: tuple[str, ...]
    preset: str | None = None
    version: int = FEATURE_SCHEMA_VERSION

    @classmethod
    def from_columns(cls, feature_columns: Sequence[str], *, preset: str | None = None) -> FeatureSchema:
        """Build a schema from an ordered feature-column sequence."""

        columns = tuple(str(column) for column in feature_columns)
        if not columns:
            raise ValueError("FeatureSchema requires at least one feature column")
        if len(set(columns)) != len(columns):
            raise ValueError("FeatureSchema feature columns must be unique")
        return cls(feature_columns=columns, preset=preset)

    @property
    def digest(self) -> str:
        """Return a stable digest that changes iff version or columns change."""

        payload = {
            "version": int(self.version),
            "feature_columns": list(self.feature_columns),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def to_json_dict(self) -> dict[str, Any]:
        """Return the persisted schema payload."""

        return {
            "version": int(self.version),
            "preset": self.preset,
            "feature_columns": list(self.feature_columns),
            "digest": self.digest,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> FeatureSchema:
        """Load and verify a persisted schema payload."""

        schema = cls.from_columns(
            [str(column) for column in payload["feature_columns"]],
            preset=(str(payload["preset"]) if payload.get("preset") is not None else None),
        )
        version = int(payload.get("version", FEATURE_SCHEMA_VERSION))
        if version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported feature schema version {version}")
        expected_digest = str(payload.get("digest", ""))
        if expected_digest and expected_digest != schema.digest:
            raise ValueError(f"Feature schema digest mismatch: expected {expected_digest}, computed {schema.digest}")
        return schema

    def assert_matches(self, feature_columns: Sequence[str]) -> None:
        """Raise if another feature-column sequence does not match this schema."""

        other = FeatureSchema.from_columns(feature_columns, preset=self.preset)
        if other.digest != self.digest:
            raise ValueError(f"Feature schema digest mismatch: expected {self.digest}, computed {other.digest}")
