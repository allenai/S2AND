"""Small shared policy helpers for incremental linking orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from s2and.arrow_inputs import ValidatedArrowInputs
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs
from s2and.name_count_binding import NameCountsBinding


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


def require_arrow_name_counts_index_for_clusterer(
    clusterer: Any,
    arrow_paths: ValidatedArrowInputs,
    *,
    context: str,
) -> None:
    """Require the exact Arrow name-count generation selected by the model."""

    if not clusterer_uses_name_count_features(clusterer):
        return
    if arrow_paths.get("name_counts_index") is None:
        raise ValueError(
            f"{context} with selected name_counts features requires name_counts_index. "
            "Pass the S2AND name-count index directory in arrow_paths['name_counts_index']."
        )
    expected = NameCountsBinding.from_feature_contract(
        getattr(clusterer, "feature_contract", None),
        context=f"{context} model feature_contract",
    )
    manifest = arrow_paths.name_counts_manifest
    if manifest is None:  # pragma: no cover - validated-input invariant
        raise RuntimeError("validated Arrow inputs lost the retained name-count manifest")
    observed = NameCountsBinding.from_provenance(
        manifest.source_provenance,
        context=f"{context} Arrow name_counts_index source_provenance",
    )
    expected.require_matches(
        observed,
        context=context,
        source="arrow_paths['name_counts_index']",
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
    expected = NameCountsBinding.from_feature_contract(
        getattr(clusterer, "feature_contract", None),
        context=f"{context} model feature_contract",
    )
    observed = NameCountsBinding.from_provenance(
        getattr(dataset, "name_counts_provenance", None),
        context=f"{context} ANDData.name_counts_provenance",
    )
    expected.require_matches(
        observed,
        context=context,
        source="ANDData.name_counts_provenance",
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
    expected = NameCountsBinding.from_feature_contract(
        getattr(clusterer, "feature_contract", None),
        context=f"{context} model feature_contract",
    )
    observed = NameCountsBinding.from_rust_featurizer(
        rust_featurizer,
        context=f"{context} Rust featurizer",
    )
    expected.require_matches(
        observed,
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
