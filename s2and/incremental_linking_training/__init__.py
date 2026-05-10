"""Replay/training helpers for promoted incremental-linker artifacts.

This package is intentionally separate from ``s2and.incremental_linking``.
The runtime package must stay free of training-only imports such as
``s2and.model`` and production-model pickle loading.
"""

from s2and.incremental_linking_training.data_loading import load_clusterer, load_giant_block_dataset
from s2and.incremental_linking_training.name_counts import LoadNameCountsMode, resolve_load_name_counts
from s2and.incremental_linking_training.query_support import (
    DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY,
    FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME,
    ClusterProfile,
    FrozenRustHybridCentroidPolicy,
    RustHybridCentroidScoringConfig,
    build_cluster_profile,
    build_labeled_retrieval_subblock_index,
    build_rust_hybrid_centroid_retriever,
    counter_query_overlap,
    load_labeled_dataset,
    middle_initial_compatibility,
    name_count_rarity_features,
    rank_top_summaries_rust_hybrid_centroid,
    specter_exemplar_similarity,
    title_overlap,
    year_compatibility,
)

__all__ = [
    "DEFAULT_CHOOSER_CACHE_MAX_TOP_K",
    "FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY",
    "FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME",
    "ClusterProfile",
    "FrozenRustHybridCentroidPolicy",
    "LoadNameCountsMode",
    "RustHybridCentroidScoringConfig",
    "build_cluster_profile",
    "build_labeled_retrieval_subblock_index",
    "build_rust_hybrid_centroid_retriever",
    "counter_query_overlap",
    "load_clusterer",
    "load_giant_block_dataset",
    "load_labeled_dataset",
    "middle_initial_compatibility",
    "name_count_rarity_features",
    "rank_top_summaries_rust_hybrid_centroid",
    "resolve_load_name_counts",
    "specter_exemplar_similarity",
    "title_overlap",
    "year_compatibility",
]
