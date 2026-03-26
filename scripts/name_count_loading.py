"""Helpers for deciding whether name-count tables should be materialized."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

LoadNameCountsMode = bool | Literal["auto"]


def _feature_groups_use_name_counts(feature_info: Any | None) -> bool | None:
    """Return whether a feature selection explicitly uses name-count features."""

    features_to_use = getattr(feature_info, "features_to_use", None)
    if not isinstance(features_to_use, Collection) or isinstance(features_to_use, str):
        return None
    return "name_counts" in features_to_use


def resolve_load_name_counts(
    *,
    load_name_counts: LoadNameCountsMode,
    clusterer: Any | None = None,
    featurizer_info: Any | None = None,
    nameless_featurizer_info: Any | None = None,
) -> bool:
    """Resolve whether name-count tables should be loaded for a model-backed path."""

    if isinstance(load_name_counts, bool):
        return bool(load_name_counts)
    if str(load_name_counts) != "auto":
        raise ValueError(f"Unknown load_name_counts mode: {load_name_counts!r}")

    decisions: list[bool] = []
    for feature_info in (
        getattr(clusterer, "featurizer_info", None),
        getattr(clusterer, "nameless_featurizer_info", None),
        featurizer_info,
        nameless_featurizer_info,
    ):
        decision = _feature_groups_use_name_counts(feature_info)
        if decision is not None:
            decisions.append(bool(decision))

    if any(decisions):
        return True
    if decisions:
        return False
    return True
