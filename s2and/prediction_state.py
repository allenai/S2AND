"""Request-owned seed overrides and diagnostics for prediction orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionState:
    """Own the mutable inputs and diagnostics of one prediction operation.

    Feature data and fitted models remain outside this state and can be reused
    across operations. Callers transfer owned seed collections to this object.
    """

    cluster_seeds_require: dict[str, int | str] = field(default_factory=dict)
    cluster_seeds_disallow: set[tuple[str, str]] = field(default_factory=set)
    altered_cluster_signatures: list[str] = field(default_factory=list)
    telemetry: dict[str, dict[str, Any]] = field(default_factory=dict)
