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
    # Excluded mentions per anchor; see s2and.profile_exclusions.
    profile_exclusions: dict[str, frozenset[str]] = field(default_factory=dict)
    # Signatures whose surname block in the request's dataset has under MIN_BLOCK_SIZE mentions.
    # The year-gap split's first-name bridge reads it; None treats each request block as whole.
    small_block_signatures: frozenset[str] | None = None
    telemetry: dict[str, dict[str, Any]] = field(default_factory=dict)
