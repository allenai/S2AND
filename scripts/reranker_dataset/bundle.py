"""Bridge-mode bundle metadata contracts for the unified row-engine migration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import FeatureSchema

BUNDLE_CONTRACT_VERSION = 1
CLASSIC_GATE_ONLY_CALIBRATION_SURFACE = "classic_gate_only"


@dataclass(frozen=True)
class RerankerBundleContract:
    """Additive metadata contract shared by row engine, trainer, and validator."""

    feature_schema: FeatureSchema
    calibration_surface: str = CLASSIC_GATE_ONLY_CALIBRATION_SURFACE
    migration_manifest: Mapping[str, str] = field(default_factory=dict)
    version: int = BUNDLE_CONTRACT_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        """Return the persisted contract payload."""

        return {
            "version": int(self.version),
            "feature_schema": self.feature_schema.to_json_dict(),
            "calibration_surface": str(self.calibration_surface),
            "migration_manifest": {str(key): str(value) for key, value in sorted(self.migration_manifest.items())},
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> RerankerBundleContract:
        """Load a persisted contract payload."""

        version = int(payload.get("version", BUNDLE_CONTRACT_VERSION))
        if version != BUNDLE_CONTRACT_VERSION:
            raise ValueError(f"Unsupported reranker bundle contract version {version}")
        return cls(
            feature_schema=FeatureSchema.from_json_dict(dict(payload["feature_schema"])),
            calibration_surface=str(payload.get("calibration_surface", CLASSIC_GATE_ONLY_CALIBRATION_SURFACE)),
            migration_manifest={
                str(key): str(value) for key, value in dict(payload.get("migration_manifest", {})).items()
            },
        )

    def write_json(self, path: Path) -> None:
        """Write a stable contract JSON artifact."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> RerankerBundleContract:
        """Read a contract JSON artifact."""

        return cls.from_json_dict(json.loads(path.read_text(encoding="utf-8")))
