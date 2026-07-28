"""Small shared policy helpers for incremental linking orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from s2and._sha256 import is_lowercase_sha256
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs

PROMOTED_LINKER_MODEL_SUPPRESS_ORCID = True
NAME_COUNTS_MANIFEST_SHA256_FIELD = "name_counts_manifest_sha256"


def promoted_linker_orcid_force_link_enabled(*, suppress_orcid: bool) -> bool:
    """Return whether ORCID may force a runtime linker decision.

    ORCID is deliberately excluded from learned pair-constraint features. When
    enabled, it acts only through the explicit runtime force-link branch.
    """

    return not bool(suppress_orcid)


def clusterer_uses_name_count_features(clusterer: Any) -> bool:
    """Return whether the clusterer requires global name-count features."""

    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = getattr(featurizer_info, "features_to_use", ())
        if "name_counts" in features_to_use:
            return True
    return False


def clusterer_uses_embedding_features(clusterer: Any) -> bool:
    """Return whether the clusterer requires SPECTER embedding features."""

    for attr_name in ("featurizer_info", "nameless_featurizer_info"):
        featurizer_info = getattr(clusterer, attr_name, None)
        features_to_use = getattr(featurizer_info, "features_to_use", ())
        if "embedding_similarity" in features_to_use:
            return True
    return False


def require_name_counts_manifest_sha256(value: Any, *, context: str) -> str:
    """Return one validated name-count manifest identity."""

    if not is_lowercase_sha256(value):
        raise ValueError(f"{context} requires {NAME_COUNTS_MANIFEST_SHA256_FIELD} as a lowercase SHA-256")
    return value


def _require_name_counts_binding(
    clusterer: Any,
    observed_value: Any,
    *,
    context: str,
    source: str,
) -> None:
    feature_contract = getattr(clusterer, "feature_contract", None)
    if not isinstance(feature_contract, Mapping):
        raise ValueError(f"{context} model requires a feature_contract mapping with a name-count identity")
    expected = require_name_counts_manifest_sha256(
        feature_contract.get(NAME_COUNTS_MANIFEST_SHA256_FIELD),
        context=f"{context} model feature_contract",
    )
    observed = require_name_counts_manifest_sha256(
        observed_value,
        context=f"{context} {source}",
    )
    if observed != expected:
        raise ValueError(
            f"{context} name-count binding mismatch for {source}: expected={expected!r} observed={observed!r}"
        )


def require_arrow_name_counts_index_for_clusterer(
    clusterer: Any,
    arrow_dataset: ArrowDataset,
    *,
    context: str,
) -> None:
    """Require the exact Arrow name-count generation selected by the model."""

    if not clusterer_uses_name_count_features(clusterer):
        return
    if not arrow_dataset.has("name_counts_index"):
        raise ValueError(
            f"{context} with selected name_counts features requires name_counts_index. "
            "Open an Arrow release containing the S2AND name-count index."
        )
    index = arrow_dataset.name_counts_index
    if index is None:  # pragma: no cover - validated-input invariant
        raise RuntimeError("validated Arrow inputs lost the retained name-count index")
    _require_name_counts_binding(
        clusterer,
        index.manifest_sha256,
        context=context,
        source="ArrowDataset.name_counts_index",
    )


def require_dataset_name_counts_binding_for_clusterer(
    clusterer: Any,
    dataset: Any,
    *,
    context: str,
) -> None:
    """Require the exact in-memory name-count generation selected by the model."""

    if not clusterer_uses_name_count_features(clusterer):
        return
    _require_name_counts_binding(
        clusterer,
        getattr(dataset, "name_counts_manifest_sha256", None),
        context=context,
        source="ANDData.name_counts_manifest_sha256",
    )


def require_rust_featurizer_name_counts_binding_for_clusterer(
    clusterer: Any,
    rust_featurizer: Any,
    *,
    context: str,
) -> None:
    """Require the exact name-count generation retained by a Rust featurizer."""

    if not clusterer_uses_name_count_features(clusterer):
        return
    manifest_sha256 = getattr(rust_featurizer, "name_counts_manifest_sha256", None)
    if manifest_sha256 is None:
        raise ValueError(f"{context} requires a Rust featurizer with a verified name-count manifest")
    _require_name_counts_binding(
        clusterer,
        manifest_sha256,
        context=context,
        source="RustFeaturizer.name_counts_manifest_sha256",
    )


def resolve_load_name_counts_policy(
    clusterer: Any,
    load_name_counts: bool | None,
    *,
    context: str,
) -> bool:
    """Return the effective name-count load policy for raw scoring."""

    clusterer_requires_name_counts = clusterer_uses_name_count_features(clusterer)
    if load_name_counts is False and clusterer_requires_name_counts:
        raise ValueError(
            f"{context} cannot run with load_name_counts=False when the clusterer selects name_counts features"
        )
    if load_name_counts is None:
        return clusterer_requires_name_counts
    return bool(load_name_counts)


def dataset_cluster_seed_disallows(dataset: Any) -> set[tuple[str, str]]:
    """Return normalized disallow constraints stored on a request dataset."""

    return set(normalize_cluster_seed_disallow_pairs(getattr(dataset, "cluster_seeds_disallow", set()) or set()))


def request_cluster_seed_disallow_parts(
    dataset: Any,
    arrow_disallows: Iterable[tuple[Any, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    """Return request, dataset, and Arrow disallow sets with one normalization policy."""

    dataset_disallows = dataset_cluster_seed_disallows(dataset)
    arrow_disallow_set = set(normalize_cluster_seed_disallow_pairs(arrow_disallows))
    request_disallows = set(arrow_disallow_set)
    request_disallows.update(dataset_disallows)
    return request_disallows, dataset_disallows, arrow_disallow_set
