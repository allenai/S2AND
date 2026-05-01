"""Shared helpers for the single-letter reranker pipeline.

This module keeps the new reranker path small and explicit:

1. build real retrieval windows from masked single-letter query views
2. aggregate production pairwise distances inside that retrieved window
3. persist candidate-level rows to ``scratch/``
4. train and evaluate grouped rerankers from those rows

The retrieval view masking is applied at candidate generation time, matching the
Task-1 operating point. Pairwise aggregation reuses the production pairwise
model on the held-out signature against the retrieved candidates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import s2and.data as s2and_data_module
import s2and.subblocking as s2and_subblocking_module
from s2and.consts import DEFAULT_CHUNK_SIZE, LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import many_pairs_featurize
from s2and.model import _predict_and_combine
from s2and.subblocking import make_subblocks_with_telemetry
from s2and.text import name_counts as pairwise_name_counts
from s2and.text import normalize_text, same_prefix_tokens

try:
    from scripts.name_count_loading import LoadNameCountsMode, resolve_load_name_counts
except ImportError:  # pragma: no cover - direct script execution path
    from name_count_loading import LoadNameCountsMode, resolve_load_name_counts  # type: ignore

try:
    import scripts.eval_cluster_retrieval as retrieval
    from scripts.joint_safe_link_dataset_contract import apply_hard_disallow_component_filter
    from scripts.single_letter_retrieval_utils import (
        FrozenRustHybridCentroidPolicy,
        RustHybridCentroidRetrieverHandle,
        build_rust_name_compatible_subblock_selector,
        compute_chooser_summary_features_rust_hybrid_centroid,
        rank_top_summaries,
        rank_top_summaries_rust_hybrid_centroid,
    )
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore
    from joint_safe_link_dataset_contract import apply_hard_disallow_component_filter  # type: ignore
    from single_letter_retrieval_utils import (  # type: ignore
        FrozenRustHybridCentroidPolicy,
        RustHybridCentroidRetrieverHandle,
        build_rust_name_compatible_subblock_selector,
        compute_chooser_summary_features_rust_hybrid_centroid,
        rank_top_summaries,
        rank_top_summaries_rust_hybrid_centroid,
    )

DEFAULT_LABELED_DATASETS = (
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
)
DEFAULT_QUERY_VIEWS = (
    "full",
    "initial_only",
)
DEFAULT_RETRIEVAL_APPROACH = "all__hybrid_centroid"
SUPPORTED_RETRIEVAL_METHODS = frozenset({"hybrid_centroid", "hybrid_exemplar_4"})
RETRIEVAL_ENGINE_CHOICES = frozenset({"auto", "python", "rust"})
RETRIEVAL_AMBIGUITY_SCORE_GAP = 0.02
RETRIEVAL_AMBIGUITY_SAME_FAMILY_GAP = 0.05
DEFAULT_RETRIEVAL_WINDOW_SIZE = 25
DEFAULT_CANDIDATE_WINDOW_SENSITIVITY = (5, 25)
DEFAULT_H_WANG_WINDOW_SENSITIVITY = (5, 25)
DEFAULT_CHOOSER_CACHE_MAX_TOP_K = 25
RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY = "_rust_name_compatible_subblock_selector"
RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_COUNT_KEY = "_rust_name_compatible_subblock_selector_fallback_count"
RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_REASON_KEY = "_rust_name_compatible_subblock_selector_fallback_reason"
STRICT_RUST_NAME_COMPAT_ENV = "S2AND_STRICT_RUST_NAME_COMPAT"
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
COUNT_NORMALIZED_CONFIDENCE_TOP_K = 5
COUNT_NORMALIZED_CONFIDENCE_SUPPORT_GAMMA = 0.5
CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE = 192.0
GENERIC_HEURISTIC_OVERRIDE_MARGIN = 0.01
GENERIC_CROSS_FAMILY_EXTRA_MARGIN = 0.04
GENERIC_FAMILY_MIN_COUNT = 3
GENERIC_FAMILY_MIN_RATIO = 0.6
EXACT_TITLE_ANCHOR_THRESHOLD = 0.95
ANCHOR_SUPPORT_OVERLAP_THRESHOLD = 0.05
ANCHOR_YEAR_COMPATIBILITY_THRESHOLD = 0.85
NEAR_TIED_RETRIEVAL_SCORE_GAP = 0.02
PLAUSIBLE_CONFLICT_RETRIEVAL_GAP = 0.05
SEVERE_CONTRADICTION_THRESHOLD = 0.75
CALIBRATION_BIN_COUNT = 10
BASELINE_FEATURE_COLUMNS = (
    "retrieval_rank",
    "retrieval_score",
    "cluster_size",
    "candidate_count",
    "min_distance",
    "mean_distance",
    "top3_mean_distance",
    "top5_mean_distance",
    "top10_mean_distance",
    "top20_mean_distance",
    "pair_count",
    "count_normalized_confidence",
    "retrieval_score_gap_vs_best_competitor",
    "retrieval_rank_gap_vs_best_competitor",
    "top3_mean_delta_vs_best_competitor",
    "top5_mean_delta_vs_best_competitor",
    "cluster_size_ratio_vs_best_competitor",
    "same_family_vs_best_competitor",
    "same_family_as_top1",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
    "family_instability_flag",
    "fragment_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
)
RAW_METADATA_SIMILARITY_FEATURE_COLUMNS = (
    "raw_max_affiliation_jaccard",
    "raw_max_coauthor_jaccard",
    "raw_max_title_jaccard",
    "raw_max_text_jaccard",
)
PAIRWISE_NAME_COUNT_FEATURE_NAMES = (
    "first_name_count_min",
    "last_first_name_count_min",
    "last_name_count_min",
    "last_first_initial_count_min",
    "first_name_count_max",
    "last_first_name_count_max",
)
NAME_COUNT_RARITY_FEATURE_COLUMNS = (
    "first_name_count_min_rarity",
    "last_first_name_count_min_rarity",
    "last_name_count_min_rarity",
    "last_first_initial_count_min_rarity",
    "first_name_count_max_rarity",
    "last_first_name_count_max_rarity",
    "first_prefix_x_last_first_name_count_min_rarity",
    "candidate_first_name_count_min_rarity",
    "candidate_last_first_name_count_min_rarity",
    "candidate_last_name_count_min_rarity",
    "candidate_last_first_initial_count_min_rarity",
)
PAIRWISE_TRANSFER_FEATURE_COLUMNS = (
    "min_distance",
    "mean_distance",
    "top3_mean_distance",
    "top5_mean_distance",
    "min_distance_best_gap",
    "mean_distance_best_gap",
    "top3_distance_best_gap",
    "top5_distance_best_gap",
    "min_distance_rank_fraction",
    "mean_distance_rank_fraction",
    "top3_distance_rank_fraction",
    "top5_distance_rank_fraction",
    "count_normalized_confidence",
    "distance_spread_top5_minus_min",
    "distance_spread_mean_minus_top5",
    "same_family_as_best_top5",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
    "affiliation_overlap_rank_fraction",
    "coauthor_overlap_rank_fraction",
    "venue_overlap_rank_fraction",
    "year_compatibility_rank_fraction",
    "title_overlap_rank_fraction",
    "specter_centroid_rank_fraction",
    "specter_exemplar_rank_fraction",
    "family_instability_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
)
HEURISTIC_DECISION_FEATURE_COLUMNS = (
    "is_retrieval_top1",
    "is_best_top3",
    "is_best_top5",
    "is_heuristic_choice",
    "same_family_as_heuristic_choice",
    "top3_gap_to_retrieval_top1",
    "top5_gap_to_retrieval_top1",
    "top3_gap_to_heuristic_choice",
    "top5_gap_to_heuristic_choice",
    "heuristic_top1_vs_best_top5_margin",
    "heuristic_margin_threshold",
    "heuristic_margin_slack",
    "heuristic_prefers_top1",
    "heuristic_cross_family_top1_vs_best_top5",
)
HEURISTIC_CORE_FEATURE_COLUMNS = (
    "is_retrieval_top1",
    "is_best_top3",
    "is_best_top5",
    "top3_gap_to_retrieval_top1",
    "top5_gap_to_retrieval_top1",
    "heuristic_top1_vs_best_top5_margin",
    "heuristic_margin_slack",
    "heuristic_prefers_top1",
    "heuristic_cross_family_top1_vs_best_top5",
)
HEURISTIC_STACKING_FEATURE_COLUMNS = (
    *HEURISTIC_CORE_FEATURE_COLUMNS,
    "is_heuristic_choice",
    "same_family_as_heuristic_choice",
)
GENERALIZED_V1_FEATURE_COLUMNS = (
    "retrieval_rank_fraction",
    *PAIRWISE_TRANSFER_FEATURE_COLUMNS,
)
GENERALIZED_V2_FEATURE_COLUMNS = (
    "retrieval_rank_fraction",
    "retrieval_score_rank_fraction",
    "retrieval_score_best_gap",
    *PAIRWISE_TRANSFER_FEATURE_COLUMNS,
)
GENERALIZED_V3_FEATURE_COLUMNS = PAIRWISE_TRANSFER_FEATURE_COLUMNS
GENERALIZED_V4_FEATURE_COLUMNS = tuple(
    column
    for column in PAIRWISE_TRANSFER_FEATURE_COLUMNS
    if column
    not in {
        "venue_overlap",
        "year_compatibility",
        "venue_overlap_rank_fraction",
        "year_compatibility_rank_fraction",
    }
)
GENERALIZED_V5_FEATURE_COLUMNS = (
    *GENERALIZED_V3_FEATURE_COLUMNS,
    *HEURISTIC_DECISION_FEATURE_COLUMNS,
)
GENERALIZED_V6_FEATURE_COLUMNS = (
    *GENERALIZED_V3_FEATURE_COLUMNS,
    *HEURISTIC_CORE_FEATURE_COLUMNS,
)
GENERALIZED_V7_FEATURE_COLUMNS = (
    *GENERALIZED_V3_FEATURE_COLUMNS,
    *HEURISTIC_STACKING_FEATURE_COLUMNS,
)
GENERALIZED_V8_FEATURE_COLUMNS = (
    *GENERALIZED_V7_FEATURE_COLUMNS,
    "cross_family_with_top1",
    "override_slack_vs_top1",
    "beats_top1_after_penalty",
)
GENERALIZED_V9_DROPPED_COLUMNS = frozenset(
    {
        "beats_top1_after_penalty",
        "family_instability_flag",
        "heuristic_cross_family_top1_vs_best_top5",
        "heuristic_margin_slack",
        "heuristic_prefers_top1",
        "heuristic_top1_vs_best_top5_margin",
        "middle_initial_compatibility",
        "query_has_affiliations",
        "query_has_coauthors",
        "query_has_full_first",
        "query_has_middle",
        "query_has_specter",
    }
)
GENERALIZED_V9_FEATURE_COLUMNS = tuple(
    column for column in GENERALIZED_V8_FEATURE_COLUMNS if column not in GENERALIZED_V9_DROPPED_COLUMNS
)
SMALL_CORE_3_FEATURE_COLUMNS = (
    "min_distance_rank_fraction",
    "top3_distance_rank_fraction",
    "is_heuristic_choice",
)
SMALL_CORE_6_FEATURE_COLUMNS = (
    *SMALL_CORE_3_FEATURE_COLUMNS,
    "is_retrieval_top1",
    "affiliation_overlap",
    "count_normalized_confidence",
)
TITLE_FAST_V1_FEATURE_COLUMNS = (
    "min_distance_rank_fraction",
    "top3_distance_rank_fraction",
    "is_heuristic_choice",
    "is_retrieval_top1",
    "affiliation_overlap",
    "count_normalized_confidence",
    "venue_overlap_rank_fraction",
    "same_family_as_top1",
    "top5_gap_to_retrieval_top1",
    "heuristic_margin_slack",
    "heuristic_prefers_top1",
    "affiliation_overlap_rank_fraction",
    "specter_exemplar_rank_fraction",
    "year_compatibility",
    "title_overlap_rank_fraction",
    "coauthor_overlap",
    "venue_overlap",
    "middle_initial_compatibility",
    "title_overlap",
    "coauthor_overlap_rank_fraction",
)
CLASSIC_SHORTLIST_INVARIANT_V1_FEATURE_COLUMNS = (
    "affiliation_overlap",
    "affiliation_contradiction_severity",
    "venue_overlap",
    "coauthor_overlap",
    "middle_initial_compatibility",
    "title_overlap",
    "exact_anchor_evidence_flag",
    "year_compatibility",
    "year_mismatch_severity",
    "specter_exemplar_similarity",
    "specter_centroid_similarity",
    "cluster_size_log_capped",
    "top5_mean_distance",
    "min_distance",
    "distance_spread_top5_minus_min",
    "query_view__full",
    "query_view__initial_only",
)
CLASSIC_BEST21_FEATURE_COLUMNS = (
    "min_distance",
    "top3_distance_best_gap",
    "retrieval_rank",
    "top20_mean_distance",
    "top3_gap_to_heuristic_choice",
    "top1_strongest_contradiction",
    "retrieval_score",
    "title_overlap",
    "retrieval_top1_score",
    "specter_exemplar_similarity",
    "initial_only_x_venue_overlap",
    "count_normalized_confidence",
    "distance_spread_mean_minus_top5",
    "affiliation_contradiction_severity",
    "cluster_size_log_capped",
    "coauthor_gap_to_best_same_coarse_family",
    "near_tied_alternative_count",
    "pair_count",
    "retrieval_score_rank_fraction",
    "top3_distance_rank_fraction",
    "middle_initial_compatibility",
)
FEATURE_PRESETS = {
    "baseline": BASELINE_FEATURE_COLUMNS,
    "generalized_v1": GENERALIZED_V1_FEATURE_COLUMNS,
    "generalized_v2": GENERALIZED_V2_FEATURE_COLUMNS,
    "generalized_v3": GENERALIZED_V3_FEATURE_COLUMNS,
    "generalized_v4": GENERALIZED_V4_FEATURE_COLUMNS,
    "generalized_v5": GENERALIZED_V5_FEATURE_COLUMNS,
    "generalized_v6": GENERALIZED_V6_FEATURE_COLUMNS,
    "generalized_v7": GENERALIZED_V7_FEATURE_COLUMNS,
    "generalized_v8": GENERALIZED_V8_FEATURE_COLUMNS,
    "generalized_v9": GENERALIZED_V9_FEATURE_COLUMNS,
    "small_core_3": SMALL_CORE_3_FEATURE_COLUMNS,
    "small_core_6": SMALL_CORE_6_FEATURE_COLUMNS,
    "title_fast_v1": TITLE_FAST_V1_FEATURE_COLUMNS,
    "classic_shortlist_invariant_v1": CLASSIC_SHORTLIST_INVARIANT_V1_FEATURE_COLUMNS,
    "classic_best21": CLASSIC_BEST21_FEATURE_COLUMNS,
}
DEFAULT_FEATURE_PRESET = "classic_best21"
FEATURE_COLUMNS = FEATURE_PRESETS[DEFAULT_FEATURE_PRESET]
AUDIT_ORCID_METADATA_COLUMNS = (
    "_audit_normalized_orcid",
    "_audit_orcid_group_size",
    "_audit_orcid_group_size_bucket",
)
ENRICHMENT_PROFILES = (
    "none",
    "heuristic_error_regions_v1",
    "heuristic_override_regions_v2",
    "s2and_hard_regions_v1",
)
ROW_COLUMNS = (
    "source",
    "dataset",
    "query_source",
    "query_view",
    "natural_query_view",
    "query_group_id",
    "query_id",
    "query_signature_id",
    "query_first_token",
    "query_first_initial",
    "query_year",
    *AUDIT_ORCID_METADATA_COLUMNS,
    "split",
    "block_key",
    "block_size",
    "component_size",
    "sampling_info_bucket",
    "supervision_type",
    "positive_candidate_count",
    "positive_candidate_keys",
    "group_has_positive",
    "best_positive_retrieval_rank",
    "support_type",
    "query_in_seed_before_holdout",
    "candidate_component_key",
    "candidate_cluster_id",
    "best_competitor_component_key",
    "family_id",
    "dominant_first_name",
    "candidate_year_min",
    "candidate_year_max",
    "label",
    "candidate_count",
    "candidate_signatures",
    "scored_candidate_components",
    "scored_candidate_signatures",
    "orcid_filter_applied",
    "middle_initial_filter_applied",
    "year_range_filter_applied",
    "retrieval_rank",
    "retrieval_score",
    "cluster_size",
    "pair_count",
    "min_distance",
    "mean_distance",
    "top3_mean_distance",
    "top5_mean_distance",
    "top10_mean_distance",
    "top20_mean_distance",
    "count_normalized_confidence",
    "retrieval_score_gap_vs_best_competitor",
    "retrieval_rank_gap_vs_best_competitor",
    "top3_mean_delta_vs_best_competitor",
    "top5_mean_delta_vs_best_competitor",
    "cluster_size_ratio_vs_best_competitor",
    "same_family_vs_best_competitor",
    "same_family_as_top1",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    *RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    *NAME_COUNT_RARITY_FEATURE_COLUMNS,
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
    "family_instability_flag",
    "fragment_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
)
INT_COLUMNS = {
    "query_year",
    "_audit_orcid_group_size",
    "block_size",
    "component_size",
    "positive_candidate_count",
    "group_has_positive",
    "best_positive_retrieval_rank",
    "query_in_seed_before_holdout",
    "label",
    "candidate_count",
    "candidate_signatures",
    "candidate_year_min",
    "candidate_year_max",
    "scored_candidate_components",
    "scored_candidate_signatures",
    "orcid_filter_applied",
    "middle_initial_filter_applied",
    "year_range_filter_applied",
    "retrieval_rank",
    "cluster_size",
    "pair_count",
    "same_family_as_top1",
    "same_family_vs_best_competitor",
    "family_instability_flag",
    "fragment_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
}
FLOAT_COLUMNS = {
    "retrieval_score",
    "min_distance",
    "mean_distance",
    "top3_mean_distance",
    "top5_mean_distance",
    "top10_mean_distance",
    "top20_mean_distance",
    "count_normalized_confidence",
    "retrieval_score_gap_vs_best_competitor",
    "retrieval_rank_gap_vs_best_competitor",
    "top3_mean_delta_vs_best_competitor",
    "top5_mean_delta_vs_best_competitor",
    "cluster_size_ratio_vs_best_competitor",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    *RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
    *NAME_COUNT_RARITY_FEATURE_COLUMNS,
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
}
QUERY_GROUP_METADATA_COLUMNS = (
    "source",
    "dataset",
    "query_source",
    "query_view",
    "natural_query_view",
    "query_group_id",
    "query_id",
    "query_signature_id",
    *AUDIT_ORCID_METADATA_COLUMNS,
    "split",
    "block_key",
    "block_size",
    "block_component_count",
    "component_size",
    "sampling_info_bucket",
    "supervision_type",
    "support_type",
    "query_in_seed_before_holdout",
    "group_has_positive",
    "positive_candidate_count",
    "positive_candidate_keys",
    "best_positive_retrieval_rank",
    "best_positive_rank_bucket",
    "candidate_count",
    "candidate_signatures",
    "scored_candidate_components",
    "scored_candidate_signatures",
    "retrieval_top1_component_key",
    "retrieval_top1_is_positive",
    "recoverable_non_top1",
    "cross_family_top1_vs_positive",
)
QUERY_GROUP_METADATA_INT_COLUMNS = {
    "_audit_orcid_group_size",
    "block_size",
    "block_component_count",
    "component_size",
    "query_in_seed_before_holdout",
    "group_has_positive",
    "positive_candidate_count",
    "best_positive_retrieval_rank",
    "candidate_count",
    "candidate_signatures",
    "scored_candidate_components",
    "scored_candidate_signatures",
    "retrieval_top1_is_positive",
    "recoverable_non_top1",
    "cross_family_top1_vs_positive",
}
DERIVED_FEATURE_COLUMNS = {
    "retrieval_rank_fraction",
    "retrieval_score_rank_fraction",
    "retrieval_score_best_gap",
    "min_distance_best_gap",
    "mean_distance_best_gap",
    "top3_distance_best_gap",
    "top5_distance_best_gap",
    "min_distance_rank_fraction",
    "mean_distance_rank_fraction",
    "top3_distance_rank_fraction",
    "top5_distance_rank_fraction",
    "distance_spread_top5_minus_min",
    "distance_spread_mean_minus_top5",
    "distance_spread_top20_minus_top5",
    "cluster_size_log_capped",
    "same_family_as_best_top5",
    "same_family_as_heuristic_choice",
    "coarse_family_pair_count_top50",
    "candidate_pair_share_within_coarse_family",
    "coarse_family_top5_best_gap",
    "coauthor_gap_to_best_same_coarse_family",
    "affiliation_overlap_rank_fraction",
    "coauthor_overlap_rank_fraction",
    "venue_overlap_rank_fraction",
    "year_compatibility_rank_fraction",
    "title_overlap_rank_fraction",
    "specter_centroid_rank_fraction",
    "specter_exemplar_rank_fraction",
    "is_retrieval_top1",
    "is_best_top3",
    "is_best_top5",
    "is_heuristic_choice",
    "top3_gap_to_retrieval_top1",
    "top5_gap_to_retrieval_top1",
    "top3_gap_to_heuristic_choice",
    "top5_gap_to_heuristic_choice",
    "heuristic_top1_vs_best_top5_margin",
    "heuristic_margin_threshold",
    "heuristic_margin_slack",
    "heuristic_prefers_top1",
    "heuristic_cross_family_top1_vs_best_top5",
    "cross_family_with_top1",
    "override_slack_vs_top1",
    "beats_top1_after_penalty",
    "retrieval_top1_score",
    "retrieval_top1_margin",
    "near_tied_alternative_count",
    "exact_anchor_evidence_flag",
    "top1_exact_anchor_evidence_flag",
    "top1_minus_runnerup_retrieval_score",
    "top1_minus_runnerup_title_overlap",
    "top1_minus_runnerup_coauthor_overlap",
    "top1_minus_runnerup_venue_overlap",
    "top1_minus_runnerup_year_compatibility",
    "top1_minus_runnerup_retrieval_rank",
    "top1_minus_runnerup_count_normalized_confidence",
    "top1_minus_runnerup_cluster_size",
    "year_mismatch_severity",
    "affiliation_contradiction_severity",
    "initial_only_x_title_overlap",
    "initial_only_x_coauthor_overlap",
    "initial_only_x_venue_overlap",
    "candidate_contradiction_count",
    "candidate_contradiction_score",
    "exact_title_identity_conflict_flag",
    "top1_contradiction_count",
    "top1_strongest_contradiction",
    "top1_exact_title_identity_conflict_flag",
    "plausible_conflicting_candidate_count",
    "anchor_evidence_count",
    "strong_positive_anchor_score",
    "weak_residual_anchor_score",
    "sparse_relative_winner_score",
    "query_view__full",
    "query_view__initial_only",
}
NUMERIC_FEATURE_COLUMNS = (
    set(BASELINE_FEATURE_COLUMNS)
    | set(RAW_METADATA_SIMILARITY_FEATURE_COLUMNS)
    | set(NAME_COUNT_RARITY_FEATURE_COLUMNS)
    | DERIVED_FEATURE_COLUMNS
)
MATERIALIZED_DERIVED_ROW_COLUMNS = (*ROW_COLUMNS, *sorted(DERIVED_FEATURE_COLUMNS))

_ORIGINAL_COMPUTE_BLOCK = s2and_data_module.compute_block


@dataclass(frozen=True)
class RerankerQueryCase:
    """One held-out reranker query group."""

    source: str
    dataset: str
    query_id: str
    query_signature_id: str
    block_key: str
    positive_component_keys: frozenset[str]
    support_type: str
    block_size: int
    component_size: int
    sampling_info_bucket: str
    query_source: str = "labeled"
    normalized_orcid: str | None = None
    orcid_group_size: int | None = None
    orcid_group_size_bucket: str | None = None
    split: str = "all"
    supervision_type: str = "labeled"
    query_in_seed_before_holdout: bool = False
    natural_query_view: str | None = None


@dataclass(frozen=True)
class ClusterProfile:
    """Family metadata derived from a candidate cluster summary."""

    cluster_id: str
    family_id: str
    dominant_first_name: str | None
    family_dominance_ratio: float
    family_named_count: int


@dataclass(frozen=True)
class RetrievalApproachSpec:
    """Structured retrieval-window configuration parsed from the CLI string."""

    mode: str
    methods: tuple[str, ...]


def _subblock_tokens(subblock_key: str) -> list[str]:
    """Extract normalized first-name tokens from one subblock key."""

    values: set[str] = set()
    for raw_token in str(subblock_key).split(","):
        token = str(raw_token).strip().split("|", 1)[0].strip()
        if len(token) > 1:
            values.add(token)
    return sorted(values)


def load_retrieval_subblock_index(step2_dir: Path) -> dict[str, Any]:
    """Load the minimal subblock index needed for frozen full-query candidate gating."""

    manifest = json.loads((step2_dir / "subblock_manifest.json").read_text(encoding="utf-8"))
    predicted_clusters = json.loads((step2_dir / "predicted_clusters.json").read_text(encoding="utf-8"))
    signature_to_subblock: dict[str, str] = {}
    for subblock_key, signature_ids in dict(manifest["subblocks"]).items():
        for signature_id in signature_ids:
            signature_to_subblock[str(signature_id)] = str(subblock_key)
    subblock_to_components: dict[str, set[str]] = defaultdict(set)
    subblock_tokens_by_subblock: dict[str, list[str]] = {}
    prefix_to_subblocks: dict[int, dict[str, set[str]]] = {
        2: defaultdict(set),
        3: defaultdict(set),
        4: defaultdict(set),
    }
    for subblock_key, clusters in dict(predicted_clusters).items():
        subblock = str(subblock_key)
        for component_key in dict(clusters).keys():
            subblock_to_components[subblock].add(str(component_key))
        tokens = _subblock_tokens(subblock)
        subblock_tokens_by_subblock[subblock] = tokens
        for token in tokens:
            for prefix_len in (2, 3, 4):
                prefix = token[: min(len(token), prefix_len)]
                if len(prefix) >= 2:
                    prefix_to_subblocks[prefix_len][prefix].add(subblock)
    return {
        "signature_to_subblock": signature_to_subblock,
        "subblock_to_components": {key: sorted(value) for key, value in subblock_to_components.items()},
        "subblock_tokens_by_subblock": subblock_tokens_by_subblock,
        "prefix_to_subblocks": {
            prefix_len: {key: sorted(value) for key, value in mapping.items()}
            for prefix_len, mapping in prefix_to_subblocks.items()
        },
    }


def build_labeled_retrieval_subblock_index(
    *,
    dataset: ANDData,
    block_to_component_keys: dict[str, list[str]],
    component_signatures: dict[str, list[str]],
    maximum_size: int = 15_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen full-query candidate-gate index for labeled datasets."""

    signature_to_subblock: dict[str, str] = {}
    subblock_to_components: dict[str, set[str]] = defaultdict(set)
    subblock_tokens_by_subblock: dict[str, list[str]] = {}
    prefix_to_subblocks: dict[int, dict[str, set[str]]] = {
        2: defaultdict(set),
        3: defaultdict(set),
        4: defaultdict(set),
    }
    telemetry_rows: list[dict[str, Any]] = []

    for block_key, component_keys in block_to_component_keys.items():
        block_signature_ids = sorted(
            {
                str(signature_id)
                for component_key in component_keys
                for signature_id in component_signatures[str(component_key)]
            }
        )
        subblocks, telemetry = make_subblocks_with_telemetry(
            block_signature_ids,
            dataset,
            maximum_size=int(maximum_size),
        )
        local_signature_to_subblock: dict[str, str] = {}
        for local_subblock_key, signature_ids in dict(subblocks).items():
            global_subblock_key = f"{block_key}::{local_subblock_key}"
            for signature_id in signature_ids:
                local_signature_to_subblock[str(signature_id)] = global_subblock_key
                signature_to_subblock[str(signature_id)] = global_subblock_key
            tokens = _subblock_tokens(str(local_subblock_key))
            subblock_tokens_by_subblock[global_subblock_key] = tokens
            for token in tokens:
                for prefix_len in (2, 3, 4):
                    prefix = token[: min(len(token), prefix_len)]
                    if len(prefix) >= 2:
                        prefix_to_subblocks[prefix_len][prefix].add(global_subblock_key)
        for component_key in component_keys:
            for signature_id in component_signatures[str(component_key)]:
                subblock_key = local_signature_to_subblock.get(str(signature_id))
                if subblock_key is not None:
                    subblock_to_components[subblock_key].add(str(component_key))
        telemetry_rows.append(
            {
                "block_key": str(block_key),
                "input_signature_count": int(telemetry["input_signature_count"]),
                "final_subblock_count": int(telemetry["final_subblock_count"]),
                "final_specter_labeled_subblock_count": int(telemetry["final_specter_labeled_subblock_count"]),
                "specter_invocation_count": int(telemetry["specter_invocation_count"]),
            }
        )

    diagnostics = {
        "blocks": int(len(block_to_component_keys)),
        "subblocks": int(len(subblock_to_components)),
        "mean_final_subblock_count_per_block": round(
            float(statistics.mean(int(row["final_subblock_count"]) for row in telemetry_rows)),
            6,
        )
        if telemetry_rows
        else 0.0,
        "blocks_with_specter_subblocks": int(
            sum(1 for row in telemetry_rows if int(row["final_specter_labeled_subblock_count"]) > 0)
        ),
        "blocks_with_specter_invocations": int(
            sum(1 for row in telemetry_rows if int(row["specter_invocation_count"]) > 0)
        ),
    }
    index = {
        "signature_to_subblock": signature_to_subblock,
        "subblock_to_components": {key: sorted(value) for key, value in subblock_to_components.items()},
        "subblock_tokens_by_subblock": subblock_tokens_by_subblock,
        "prefix_to_subblocks": {
            prefix_len: {key: sorted(value) for key, value in mapping.items()}
            for prefix_len, mapping in prefix_to_subblocks.items()
        },
    }
    return index, diagnostics


def _ordered_component_subset(component_keys: list[str], allowed: set[str]) -> list[str]:
    """Preserve original component ordering while applying one allowed set."""

    return [component_key for component_key in component_keys if component_key in allowed]


def _split_global_backfill_candidate_strategy(strategy: str) -> tuple[str, int]:
    """Split ``familyN_plus_global_backfillM`` into its prefix strategy and global backfill count."""

    marker = "_plus_global_backfill"
    if marker not in str(strategy):
        return str(strategy), 0
    base_strategy, backfill_count_text = str(strategy).rsplit(marker, 1)
    if not backfill_count_text.isdigit():
        raise ValueError(f"Invalid frozen full_candidate_strategy global backfill count: {strategy!r}")
    if base_strategy.startswith("family") and not base_strategy.endswith("_only"):
        base_strategy = f"{base_strategy}_only"
    return base_strategy, int(backfill_count_text)


def _append_global_backfill(
    *,
    component_keys: list[str],
    selected_component_keys: list[str],
    global_backfill_count: int,
) -> list[str]:
    """Append a bounded global fallback without removing the prefix-selected candidates."""

    if global_backfill_count <= 0:
        return selected_component_keys
    selected = set(selected_component_keys)
    backfill = [component_key for component_key in component_keys if component_key not in selected][
        : int(global_backfill_count)
    ]
    return [*selected_component_keys, *backfill]


@lru_cache(maxsize=1)
def _filtered_name_tuples() -> frozenset[tuple[str, str]]:
    """Load the filtered S2AND first-name alias tuples for Python fallback selection."""

    path = Path(__file__).resolve().parents[1] / "data" / "s2and_name_tuples_filtered.txt"
    pairs: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip().lower() for part in line.split(",")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                pairs.add((parts[0], parts[1]))
    return frozenset(pairs)


def _first_name_forms(value: str) -> tuple[str, str, str]:
    first = str(value or "").strip().lower()
    parts = first.split()
    joined = "".join(parts)
    token = parts[0] if parts else first
    return first, joined, token


def _first_names_name_compatible(first_1: str, first_2: str) -> bool:
    """Mirror the first-name part of ANDData.get_constraint."""

    first_1 = str(first_1 or "").strip().lower()
    first_2 = str(first_2 or "").strip().lower()
    if not first_1 or not first_2:
        return True
    if first_1[0] != first_2[0]:
        return False
    if same_prefix_tokens(first_1, first_2):
        return True
    first_1_forms = _first_name_forms(first_1)
    first_2_forms = _first_name_forms(first_2)
    name_tuples = _filtered_name_tuples()
    return any((left, right) in name_tuples for left, right in zip(first_1_forms, first_2_forms, strict=True))


def _python_select_name_compatible_component_keys(
    *,
    query_first: str,
    query_subblock: str,
    component_keys: list[str],
    retrieval_subblock_index: dict[str, Any],
    global_backfill_count: int,
) -> list[str]:
    """Select strict name-compatible components without the Rust extension."""

    subblock_to_components = dict(retrieval_subblock_index["subblock_to_components"])
    subblock_tokens_by_subblock = dict(retrieval_subblock_index.get("subblock_tokens_by_subblock", {}))
    same_subblock = set(str(value) for value in subblock_to_components.get(str(query_subblock), []))
    allowed: set[str] = set(same_subblock)
    for subblock, raw_tokens in subblock_tokens_by_subblock.items():
        tokens = [str(token) for token in raw_tokens]
        if any(_first_names_name_compatible(str(query_first), token) for token in tokens):
            allowed.update(str(value) for value in subblock_to_components.get(str(subblock), []))
    selected = _ordered_component_subset(component_keys, allowed)
    return (
        _append_global_backfill(
            component_keys=component_keys,
            selected_component_keys=selected,
            global_backfill_count=global_backfill_count,
        )
        if selected
        else component_keys
    )


def _rust_name_compatible_subblock_selector(
    retrieval_subblock_index: dict[str, Any],
    *,
    strict: bool = False,
) -> Any | None:
    """Return the cached Rust selector for name-compatible subblock gating when available."""

    if RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY not in retrieval_subblock_index:
        try:
            retrieval_subblock_index[RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY] = (
                build_rust_name_compatible_subblock_selector(retrieval_subblock_index)
            )
        except RuntimeError as exc:
            if strict or os.environ.get(STRICT_RUST_NAME_COMPAT_ENV, "").strip().lower() in TRUE_ENV_VALUES:
                raise
            retrieval_subblock_index[RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_COUNT_KEY] = (
                int(retrieval_subblock_index.get(RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_COUNT_KEY, 0) or 0) + 1
            )
            retrieval_subblock_index[RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_REASON_KEY] = repr(exc)
            retrieval_subblock_index[RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY] = None
    return retrieval_subblock_index[RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_KEY]


def _select_name_compatible_component_keys(
    *,
    query: retrieval.QueryFeatures,
    query_signature_id: str,
    query_subblock: str,
    component_keys: list[str],
    retrieval_subblock_index: dict[str, Any],
    global_backfill_count: int,
    strict_name_compat: bool = False,
) -> list[str]:
    """Select components from name-compatible subblocks, preserving same-subblock and backfill."""

    query_first = str(getattr(query, "first", "") or "")
    selector = _rust_name_compatible_subblock_selector(
        retrieval_subblock_index,
        strict=bool(strict_name_compat),
    )
    if selector is not None:
        selected = selector.select(
            str(query_signature_id),
            query_first,
            component_keys,
            global_backfill_count=int(global_backfill_count),
        )
        return list(selected) if selected else component_keys
    return _python_select_name_compatible_component_keys(
        query_first=query_first,
        query_subblock=query_subblock,
        component_keys=component_keys,
        retrieval_subblock_index=retrieval_subblock_index,
        global_backfill_count=global_backfill_count,
    )


def _select_component_keys_for_candidate_strategy(
    *,
    query: retrieval.QueryFeatures,
    query_signature_id: str | None,
    component_keys: list[str],
    strategy: str,
    retrieval_subblock_index: dict[str, Any] | None,
    max_ranked_clusters: int,
    strict_name_compat: bool = False,
) -> list[str]:
    """Select candidate component keys for one frozen full-query candidate strategy."""

    strategy, global_backfill_count = _split_global_backfill_candidate_strategy(str(strategy))
    if not bool(query.has_full_first) or str(strategy) == "global":
        return component_keys
    if query_signature_id is None or retrieval_subblock_index is None:
        return component_keys
    query_subblock = dict(retrieval_subblock_index["signature_to_subblock"]).get(str(query_signature_id))
    if query_subblock is None:
        return component_keys
    subblock_to_components = dict(retrieval_subblock_index["subblock_to_components"])
    same_subblock = set(str(value) for value in subblock_to_components.get(str(query_subblock), []))
    same_keys = _ordered_component_subset(component_keys, same_subblock)
    if str(strategy) == "same_subblock_only":
        return (
            _append_global_backfill(
                component_keys=component_keys,
                selected_component_keys=same_keys,
                global_backfill_count=global_backfill_count,
            )
            if same_keys
            else component_keys
        )
    if str(strategy) == "same_if_small_else_family3":
        if 0 < len(same_keys) <= int(max_ranked_clusters):
            return same_keys
        strategy = "family3_only"
    if str(strategy) in {"name_compat", "name_compat_only"}:
        return _select_name_compatible_component_keys(
            query=query,
            query_signature_id=str(query_signature_id),
            query_subblock=str(query_subblock),
            component_keys=component_keys,
            retrieval_subblock_index=retrieval_subblock_index,
            global_backfill_count=global_backfill_count,
            strict_name_compat=bool(strict_name_compat),
        )
    if str(strategy).startswith("family") and str(strategy).endswith("_only"):
        prefix_len = int(str(strategy)[len("family")])
        query_first = str(getattr(query, "first", "") or "")
        prefix = query_first[: min(len(query_first), prefix_len)]
        if len(prefix) < 2:
            return same_keys if same_keys else component_keys
        allowed: set[str] = set()
        for subblock in dict(retrieval_subblock_index["prefix_to_subblocks"]).get(prefix_len, {}).get(prefix, []):
            allowed.update(str(value) for value in subblock_to_components.get(str(subblock), []))
        selected = _ordered_component_subset(component_keys, allowed)
        return (
            _append_global_backfill(
                component_keys=component_keys,
                selected_component_keys=selected,
                global_backfill_count=global_backfill_count,
            )
            if selected
            else component_keys
        )
    raise ValueError(f"Unknown frozen full_candidate_strategy: {strategy!r}")


@dataclass
class ClusterPairwiseStats:
    """Pairwise query-to-cluster aggregate statistics."""

    cluster_id: str
    retrieval_rank: int
    retrieval_score: float
    cluster_size: int
    family_id: str = ""
    dominant_first_name: str | None = None
    family_dominance_ratio: float = 0.0
    family_named_count: int = 0
    disallow_pair_count: int = 0
    require_pair_count: int = 0
    count: int = 0
    sum_distance: float = 0.0
    min_distance: float = field(default_factory=lambda: float("inf"))
    top_smallest_neg_heap: list[float] = field(default_factory=list)

    def update(self, distance: float, *, max_top_k: int) -> None:
        """Fold one query-to-signature distance into the cluster aggregate."""

        self.count += 1
        self.sum_distance += float(distance)
        self.min_distance = min(self.min_distance, float(distance))
        if max_top_k <= 0:
            return
        if len(self.top_smallest_neg_heap) < max_top_k:
            self.top_smallest_neg_heap.append(-float(distance))
            self.top_smallest_neg_heap.sort()
            return
        current_worst = -self.top_smallest_neg_heap[0]
        if float(distance) < float(current_worst):
            self.top_smallest_neg_heap[0] = -float(distance)
            self.top_smallest_neg_heap.sort()

    @property
    def mean_distance(self) -> float:
        """Return the mean distance; ``inf`` when the cluster saw no pairs."""

        if self.count <= 0:
            return float("inf")
        return float(self.sum_distance / float(self.count))

    def topk_mean_distance(self, top_k: int) -> float:
        """Return the mean of the smallest observed distances up to ``top_k``."""

        if top_k <= 0 or not self.top_smallest_neg_heap:
            return float("inf")
        smallest = sorted(-value for value in self.top_smallest_neg_heap)
        usable = smallest[: min(int(top_k), len(smallest))]
        if not usable:
            return float("inf")
        return float(sum(usable) / len(usable))


@dataclass(frozen=True)
class QueryClusterStatsRequest:
    """One query-window scoring request for batched pairwise aggregation."""

    query_signature_id: str
    shortlist_component_keys: tuple[str, ...]
    candidate_signature_ids_by_component: dict[str, list[str]]
    retrieval_ranks: dict[str, int]
    retrieval_scores: dict[str, float]
    summary_by_component: dict[str, retrieval.ClusterSummary]
    incremental_dont_use_cluster_seeds_component_keys: frozenset[str] = frozenset()
    ignore_disallow_constraints_component_keys: frozenset[str] = frozenset()


def _dataset_has_cluster_seed_constraints(dataset: Any) -> bool:
    return bool(getattr(dataset, "cluster_seeds_require", None)) or bool(
        getattr(dataset, "cluster_seeds_disallow", None)
    )


def _query_case_allows_seed_constraint_bypass(query_case: RerankerQueryCase) -> bool:
    return (
        bool(query_case.query_in_seed_before_holdout)
        or "loo" in str(query_case.source).lower()
        or "loo" in str(query_case.split).lower()
        or "loo" in str(query_case.support_type).lower()
        or "self" in str(query_case.support_type).lower()
    )


def _has_query_seed_connection(
    dataset: Any,
    *,
    query_signature_id: str,
    candidate_signature_ids: Sequence[str],
) -> bool:
    query_signature_id = str(query_signature_id)
    require = getattr(dataset, "cluster_seeds_require", {}) or {}
    disallow = getattr(dataset, "cluster_seeds_disallow", set()) or set()
    query_required_cluster = require.get(query_signature_id)
    for candidate_signature_id in candidate_signature_ids:
        candidate_signature_id = str(candidate_signature_id)
        if (query_signature_id, candidate_signature_id) in disallow or (
            candidate_signature_id,
            query_signature_id,
        ) in disallow:
            return True
        if query_required_cluster is not None and require.get(candidate_signature_id) == query_required_cluster:
            return True
    return False


def seed_constraint_bypass_component_keys(
    *,
    dataset: Any,
    query_case: RerankerQueryCase,
    candidate_signature_ids_by_component: dict[str, list[str]],
) -> frozenset[str]:
    """Return candidate components eligible for held-out seed constraint bypass."""

    if not _dataset_has_cluster_seed_constraints(dataset):
        return frozenset()
    if not _query_case_allows_seed_constraint_bypass(query_case):
        return frozenset()
    return frozenset(
        str(component_key)
        for component_key, signature_ids in candidate_signature_ids_by_component.items()
        if _has_query_seed_connection(
            dataset,
            query_signature_id=str(query_case.query_signature_id),
            candidate_signature_ids=signature_ids,
        )
    )


@dataclass(frozen=True)
class TrainingMatrix:
    """Prepared grouped training data for the reranker."""

    ordered_rows: list[dict[str, Any]]
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    group_ids: list[str]
    kept_group_sizes: dict[str, int]
    dropped_all_negative_group_ids: list[str]
    enrichment_profile: str
    enrichment_rounds: int
    extra_group_copies: int
    groups_with_extra_copies: int
    group_repeat_counts: dict[str, int]


def _safe_compute_block(name: str) -> str:
    """Normalize empty names before delegating to the repo block function."""

    normalized_name = normalize_text(name or "")
    if not normalized_name:
        return ""
    return _ORIGINAL_COMPUTE_BLOCK(normalized_name)


def install_safe_compute_block_patch() -> None:
    """Install the blank-name-safe block computation patch used in retrieval eval."""

    s2and_data_module.__dict__["compute_block"] = _safe_compute_block
    s2and_subblocking_module.__dict__["compute_block"] = _safe_compute_block


def configure_runtime_environment(*, n_jobs: int, backend: str = "rust") -> None:
    """Set the runtime environment for reproducible inference."""

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = backend
    if str(backend).strip().lower() == "rust":
        os.environ[STRICT_RUST_NAME_COMPAT_ENV] = "1"
    thread_count = str(max(1, int(n_jobs)))
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["RAYON_NUM_THREADS"] = thread_count


def _resolve_dataset_file(data_root: Path, dataset_name: str, *candidates: str) -> str:
    """Resolve the first present dataset file path under ``data_root``."""

    for candidate in candidates:
        path = data_root / dataset_name / candidate
        if path.exists():
            return str(path)
    attempted_paths = [str(data_root / dataset_name / value) for value in candidates]
    raise FileNotFoundError(f"Missing dataset file for {dataset_name!r}; tried {attempted_paths}")


def _resolve_specter_file(data_root: Path, dataset_name: str) -> str | None:
    """Resolve the optional specter file for a labeled dataset."""

    for candidate in (
        f"{dataset_name}_specter.pickle",
        "specter.pickle",
        f"{dataset_name}_specter2.pkl",
        "specter2.pkl",
    ):
        path = data_root / dataset_name / candidate
        if path.exists():
            return str(path)
    return None


def _resolve_load_name_counts(
    *,
    load_name_counts: LoadNameCountsMode,
    clusterer: Any | None = None,
) -> bool:
    """Resolve whether labeled reranker datasets should materialize name counts."""

    return resolve_load_name_counts(load_name_counts=load_name_counts, clusterer=clusterer)


def load_labeled_dataset(
    data_root: Path,
    dataset_name: str,
    *,
    n_jobs: int,
    clusterer: Any | None = None,
    load_name_counts: LoadNameCountsMode = "auto",
) -> ANDData:
    """Load one labeled dataset for reranker generation or evaluation."""

    install_safe_compute_block_patch()
    configure_runtime_environment(n_jobs=n_jobs, backend="rust")
    resolved_load_name_counts = _resolve_load_name_counts(
        load_name_counts=load_name_counts,
        clusterer=clusterer,
    )
    return ANDData(
        signatures=_resolve_dataset_file(
            data_root,
            dataset_name,
            f"{dataset_name}_signatures.json",
            "signatures.json",
        ),
        papers=_resolve_dataset_file(
            data_root,
            dataset_name,
            f"{dataset_name}_papers.json",
            "papers.json",
        ),
        name=dataset_name,
        mode="inference",
        specter_embeddings=_resolve_specter_file(data_root, dataset_name),
        clusters=_resolve_dataset_file(
            data_root,
            dataset_name,
            f"{dataset_name}_clusters.json",
            "clusters.json",
        ),
        block_type="s2",
        n_jobs=int(n_jobs),
        load_name_counts=resolved_load_name_counts,
        preprocess=True,
        random_seed=13,
        name_tuples="filtered",
        use_orcid_id=False,
        use_sinonym_overwrite=False,
    )


def block_size_bucket(size: int) -> str:
    if size < 10:
        return "2_9"
    if size < 50:
        return "10_49"
    if size < 200:
        return "50_199"
    return "200_plus"


def component_size_bucket(size: int) -> str:
    if size <= 2:
        return "2"
    if size <= 5:
        return "3_5"
    if size <= 10:
        return "6_10"
    if size <= 20:
        return "11_20"
    return "21_plus"


def _initial_info_bucket(features: retrieval.QueryFeatures) -> str:
    if features.has_specter and (features.has_coauthors or features.has_affiliations):
        return "rich"
    if features.has_specter:
        return "specter_only"
    if features.has_coauthors or features.has_affiliations:
        return "metadata_only"
    return "sparse"


def _choose_heldout_signature(signature_ids: list[str], seed: int) -> str:
    rng = random.Random(seed)
    return str(rng.choice(signature_ids))


def _stable_component_seed(component_key: str, base_seed: int) -> int:
    digest = hashlib.sha256(component_key.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) + int(base_seed)


def _round_robin_sample(cases: list[RerankerQueryCase], *, limit: int | None, seed: int) -> list[RerankerQueryCase]:
    if limit is None or int(limit) <= 0 or len(cases) <= int(limit):
        return list(cases)
    grouped: dict[tuple[str, str, str], list[RerankerQueryCase]] = defaultdict(list)
    for case in cases:
        key = (
            block_size_bucket(int(case.block_size)),
            component_size_bucket(int(case.component_size)),
            str(case.sampling_info_bucket),
        )
        grouped[key].append(case)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    ordered_keys = sorted(grouped)
    selected: list[RerankerQueryCase] = []
    while len(selected) < int(limit):
        progressed = False
        for key in ordered_keys:
            if not grouped[key]:
                continue
            selected.append(grouped[key].pop())
            progressed = True
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def build_component_index(
    dataset: ANDData,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int], int]:
    """Build block-local component membership from a labeled dataset."""

    if dataset.clusters is None:
        raise RuntimeError(f"Dataset {dataset.name!r} has no clusters")

    components: dict[str, list[str]] = {}
    missing_cluster_signature_ids = 0
    for cluster_id, cluster_info in dataset.clusters.items():
        signatures_by_block: dict[str, list[str]] = defaultdict(list)
        for signature_id in cluster_info["signature_ids"]:
            signature_key = str(signature_id)
            block_key = dataset.signature_to_block.get(signature_key)
            if block_key is None:
                missing_cluster_signature_ids += 1
                continue
            signatures_by_block[str(block_key)].append(signature_key)
        for block_key, signature_ids in signatures_by_block.items():
            components[f"{block_key}::{cluster_id}"] = sorted(signature_ids)

    block_to_component_keys: dict[str, list[str]] = defaultdict(list)
    block_sizes: dict[str, int] = defaultdict(int)
    for component_key, signature_ids in components.items():
        block_key, _cluster_id = component_key.split("::", 1)
        block_to_component_keys[block_key].append(component_key)
        block_sizes[block_key] += len(signature_ids)
    for block_key in block_to_component_keys:
        block_to_component_keys[block_key].sort()
    return components, dict(block_to_component_keys), dict(block_sizes), int(missing_cluster_signature_ids)


def build_labeled_query_cases(
    dataset_name: str,
    dataset: ANDData,
    *,
    seed: int,
    sampling_query_view: str,
    limit_queries: int | None = None,
) -> tuple[list[RerankerQueryCase], dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    """Build deterministic held-out query groups from a labeled dataset."""

    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    components, block_to_component_keys, block_sizes, missing_cluster_signature_ids = build_component_index(dataset)
    census = {
        "blocks": len(block_to_component_keys),
        "components": len(components),
        "eligible_components": 0,
        "signatures_total": len(dataset.signatures),
        "missing_cluster_signature_ids": int(missing_cluster_signature_ids),
        "block_size_buckets": Counter(block_size_bucket(size) for size in block_sizes.values()),
        "component_size_buckets": Counter(),
    }
    cases: list[RerankerQueryCase] = []
    for component_key, signature_ids in components.items():
        component_size = len(signature_ids)
        census["component_size_buckets"][component_size_bucket(component_size)] += 1
        if component_size < 2:
            continue
        block_key, _cluster_id = component_key.split("::", 1)
        heldout_signature_id = _choose_heldout_signature(
            signature_ids,
            seed=_stable_component_seed(component_key, seed),
        )
        base_query = retrieval.extract_query_features(
            dataset,
            heldout_signature_id,
            feature_cache=feature_cache,
            orcid_enabled=False,
        )
        sampling_features = retrieval.mask_query_features(
            base_query,
            sampling_query_view,
            orcid_enabled=False,
        )
        census["eligible_components"] += 1
        cases.append(
            RerankerQueryCase(
                source="labeled",
                dataset=dataset_name,
                query_id=str(heldout_signature_id),
                query_signature_id=str(heldout_signature_id),
                block_key=str(block_key),
                positive_component_keys=frozenset({str(component_key)}),
                support_type="labeled",
                block_size=int(block_sizes[block_key]),
                component_size=int(component_size),
                sampling_info_bucket=_initial_info_bucket(sampling_features),
            )
        )
    sampled = _round_robin_sample(cases, limit=limit_queries, seed=seed)
    census["block_size_buckets"] = {str(key): int(value) for key, value in dict(census["block_size_buckets"]).items()}
    census["component_size_buckets"] = {
        str(key): int(value) for key, value in dict(census["component_size_buckets"]).items()
    }
    return sampled, census, block_to_component_keys, components


def build_component_summaries(
    dataset: ANDData,
    component_signatures: dict[str, list[str]],
    *,
    max_exemplars: int,
) -> tuple[dict[str, retrieval.ClusterSummary], dict[str, retrieval.QueryFeatures], float]:
    """Build retrieval summaries for all labeled components once."""

    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    summaries: dict[str, retrieval.ClusterSummary] = {}
    start = time.perf_counter()
    for component_key, signature_ids in component_signatures.items():
        block_key, cluster_id = component_key.split("::", 1)
        summaries[component_key] = retrieval.build_cluster_summary(
            dataset=dataset,
            block_key=block_key,
            cluster_id=cluster_id,
            component_key=component_key,
            signature_ids=list(signature_ids),
            max_exemplars=max_exemplars,
            feature_cache=feature_cache,
            orcid_enabled=False,
        )
    build_ms = (time.perf_counter() - start) * 1000.0
    return summaries, feature_cache, float(build_ms)


def build_cluster_profile(summary: retrieval.ClusterSummary) -> ClusterProfile:
    """Build generic family metadata from a retrieval summary."""

    family_named_count = int(sum(summary.first_name_counts.values()))
    dominant_first_name = None
    family_dominance_ratio = 0.0
    family_id = str(summary.component_key)
    if summary.first_name_counts and family_named_count > 0:
        dominant_first_name, dominant_count = max(
            summary.first_name_counts.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
        family_dominance_ratio = float(dominant_count / family_named_count)
        if int(family_named_count) >= int(GENERIC_FAMILY_MIN_COUNT) and float(family_dominance_ratio) >= float(
            GENERIC_FAMILY_MIN_RATIO
        ):
            family_id = str(dominant_first_name)
    return ClusterProfile(
        cluster_id=str(summary.component_key),
        family_id=str(family_id),
        dominant_first_name=str(dominant_first_name) if dominant_first_name is not None else None,
        family_dominance_ratio=float(family_dominance_ratio),
        family_named_count=int(family_named_count),
    )


def _parse_retrieval_approach(retrieval_approach: str) -> RetrievalApproachSpec:
    """Parse the retrieval approach string used by the cache builder."""

    parts = [str(value) for value in str(retrieval_approach).split("__") if str(value)]
    if len(parts) == 2 and parts[0] == "all":
        methods = (parts[1],)
        if methods[0] not in SUPPORTED_RETRIEVAL_METHODS:
            raise ValueError(f"Unsupported retrieval method {methods[0]!r} in {retrieval_approach!r}")
        return RetrievalApproachSpec(mode="all", methods=methods)
    if len(parts) == 3 and parts[0] in {"all_union", "ambiguous_union"}:
        methods = (parts[1], parts[2])
        invalid = [method for method in methods if method not in SUPPORTED_RETRIEVAL_METHODS]
        if invalid:
            raise ValueError(f"Unsupported retrieval method(s) {invalid!r} in {retrieval_approach!r}")
        if methods[0] == methods[1]:
            raise ValueError(f"Union retrieval approach requires distinct methods, got {retrieval_approach!r}")
        return RetrievalApproachSpec(mode=parts[0], methods=methods)
    raise ValueError(f"Unsupported retrieval_approach {retrieval_approach!r}")


def _rank_method_window(
    *,
    method: str,
    query: retrieval.QueryFeatures,
    candidate_summaries: list[retrieval.ClusterSummary],
    max_block_component_size: int,
    max_ranked_clusters: int,
    rust_hybrid_centroid_retriever: RustHybridCentroidRetrieverHandle | None = None,
    rust_num_threads: int | None = None,
    frozen_rust_hybrid_centroid_policy: FrozenRustHybridCentroidPolicy | None = None,
    retrieval_engine: str = "auto",
) -> tuple[list[tuple[float, retrieval.ClusterSummary]], str, int]:
    """Rank one retrieval method over the filtered candidate summaries."""

    engine = str(retrieval_engine)
    if engine not in RETRIEVAL_ENGINE_CHOICES:
        raise ValueError(f"Unsupported retrieval_engine {retrieval_engine!r}")
    if engine == "rust" and method != "hybrid_centroid":
        raise ValueError(f"retrieval_engine='rust' does not support method {method!r}")
    if engine == "python" and frozen_rust_hybrid_centroid_policy is not None:
        raise ValueError("Frozen Rust retrieval policy requires retrieval_engine='auto' or 'rust'")

    if method == "hybrid_centroid" and rust_hybrid_centroid_retriever is not None:
        override_summary: retrieval.ClusterSummary | None = None
        component_keys: list[str] = []
        fallback_reason = ""
        for summary in candidate_summaries:
            component_key = str(summary.component_key)
            base_summary = rust_hybrid_centroid_retriever.summary_by_component.get(component_key)
            if base_summary is None:
                fallback_reason = f"component {component_key!r} is missing from Rust retriever"
                break
            component_keys.append(component_key)
            if summary is not base_summary:
                if override_summary is not None:
                    fallback_reason = "more than one candidate summary differs from the Rust retriever snapshot"
                    break
                override_summary = summary
        else:
            if engine != "python":
                return (
                    rank_top_summaries_rust_hybrid_centroid(
                        query=query,
                        max_ranked_clusters=max_ranked_clusters,
                        retriever=rust_hybrid_centroid_retriever,
                        component_keys=component_keys,
                        max_block_component_size=max_block_component_size,
                        override_summary=override_summary,
                        num_threads=rust_num_threads,
                        weights=(
                            frozen_rust_hybrid_centroid_policy.weights_for_query(query)
                            if frozen_rust_hybrid_centroid_policy is not None
                            else None
                        ),
                        scoring_config=(
                            frozen_rust_hybrid_centroid_policy.scoring_config_for_query(query)
                            if frozen_rust_hybrid_centroid_policy is not None
                            else None
                        ),
                    ),
                    "rust",
                    0,
                )
        if engine == "rust":
            raise ValueError(f"Strict Rust retrieval cannot rank hybrid_centroid: {fallback_reason}")
        fallback_count = 1 if engine == "auto" and fallback_reason else 0
    elif method == "hybrid_centroid" and engine == "rust":
        raise ValueError("Strict Rust retrieval requires a Rust hybrid-centroid retriever handle")
    else:
        fallback_count = 0
    if method == "hybrid_centroid" and frozen_rust_hybrid_centroid_policy is not None:
        raise ValueError("Frozen Rust retrieval policy requires a Rust hybrid-centroid retriever handle")

    return (
        rank_top_summaries(
            method=method,
            query=query,
            candidate_summaries=candidate_summaries,
            max_block_component_size=max_block_component_size,
            max_ranked_clusters=max_ranked_clusters,
        ),
        "python",
        int(fallback_count),
    )


def _combine_union_method_windows(
    *,
    ranked_by_method: dict[str, list[tuple[float, retrieval.ClusterSummary]]],
    max_ranked_clusters: int,
) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """Combine multiple ranked method windows into one deterministic union window."""

    if max_ranked_clusters <= 0:
        raise ValueError("max_ranked_clusters must be positive")
    if not ranked_by_method:
        return [], {}, {}
    fallback_rank = int(max_ranked_clusters) + 1
    rank_maps = {
        method: {summary.component_key: rank for rank, (_score, summary) in enumerate(ranked, start=1)}
        for method, ranked in ranked_by_method.items()
    }
    score_maps = {
        method: {summary.component_key: float(score) for score, summary in ranked}
        for method, ranked in ranked_by_method.items()
    }
    union_component_keys = {summary.component_key for ranked in ranked_by_method.values() for _score, summary in ranked}
    ordered_component_keys = sorted(
        union_component_keys,
        key=lambda component_key: (
            min(rank_map.get(component_key, fallback_rank) for rank_map in rank_maps.values()),
            statistics.mean(rank_map.get(component_key, fallback_rank) for rank_map in rank_maps.values()),
            -max(score_map.get(component_key, float("-inf")) for score_map in score_maps.values()),
            str(component_key),
        ),
    )[: int(max_ranked_clusters)]
    retrieval_scores = {
        component_key: max(score_map.get(component_key, float("-inf")) for score_map in score_maps.values())
        for component_key in ordered_component_keys
    }
    retrieval_ranks = {component_key: rank for rank, component_key in enumerate(ordered_component_keys, start=1)}
    return ordered_component_keys, retrieval_scores, retrieval_ranks


def _should_expand_ambiguously(
    ranked: list[tuple[float, retrieval.ClusterSummary]],
    *,
    profiles_by_component: dict[str, ClusterProfile],
) -> bool:
    """Return whether the primary lane is ambiguous enough to justify a union expansion."""

    if len(ranked) < 2:
        return False
    top_score, top_summary = ranked[0]
    second_score, second_summary = ranked[1]
    top_profile = profiles_by_component[str(top_summary.component_key)]
    second_profile = profiles_by_component[str(second_summary.component_key)]
    score_gap = float(top_score - second_score)
    if score_gap <= float(RETRIEVAL_AMBIGUITY_SCORE_GAP):
        return True
    same_family = bool(top_profile.family_id) and str(top_profile.family_id) == str(second_profile.family_id)
    if same_family and score_gap <= float(RETRIEVAL_AMBIGUITY_SAME_FAMILY_GAP):
        return True
    if not same_family or len(ranked) < 3:
        return False
    third_score, third_summary = ranked[2]
    third_profile = profiles_by_component[str(third_summary.component_key)]
    return str(third_profile.family_id) == str(top_profile.family_id) and float(top_score - third_score) <= float(
        RETRIEVAL_AMBIGUITY_SAME_FAMILY_GAP
    )


def build_retrieval_window(
    *,
    query: retrieval.QueryFeatures,
    raw_candidate_summaries: list[retrieval.ClusterSummary],
    max_block_component_size: int,
    retrieval_approach: str,
    max_ranked_clusters: int,
    rust_hybrid_centroid_retriever: RustHybridCentroidRetrieverHandle | None = None,
    rust_num_threads: int | None = None,
    frozen_rust_hybrid_centroid_policy: FrozenRustHybridCentroidPolicy | None = None,
    query_signature_id: str | None = None,
    retrieval_subblock_index: dict[str, Any] | None = None,
    retrieval_engine: str = "auto",
) -> tuple[list[str], dict[str, float], dict[str, int], dict[str, int]]:
    """Rank candidate summaries under the fixed retrieval operating point."""

    engine = str(retrieval_engine)
    if engine not in RETRIEVAL_ENGINE_CHOICES:
        raise ValueError(f"Unsupported retrieval_engine {retrieval_engine!r}")
    approach_spec = _parse_retrieval_approach(retrieval_approach)
    candidate_summaries_pre_filter = list(raw_candidate_summaries)
    if frozen_rust_hybrid_centroid_policy is not None:
        selected_component_keys = _select_component_keys_for_candidate_strategy(
            query=query,
            query_signature_id=query_signature_id,
            component_keys=[str(summary.component_key) for summary in raw_candidate_summaries],
            strategy=str(frozen_rust_hybrid_centroid_policy.full_candidate_strategy),
            retrieval_subblock_index=retrieval_subblock_index,
            max_ranked_clusters=max_ranked_clusters,
            strict_name_compat=engine == "rust",
        )
        selected_set = set(selected_component_keys)
        candidate_summaries_pre_filter = [
            summary for summary in raw_candidate_summaries if str(summary.component_key) in selected_set
        ]
    candidate_summaries, filter_state = retrieval.apply_hard_filters(query, candidate_summaries_pre_filter)
    profiles_by_component: dict[str, ClusterProfile] = {}
    if approach_spec.mode == "ambiguous_union":
        profiles_by_component = {
            str(summary.component_key): build_cluster_profile(summary) for summary in candidate_summaries
        }
    ranked_by_method: dict[str, list[tuple[float, retrieval.ClusterSummary]]] = {}
    retrieval_engine_rust_method_count = 0
    retrieval_engine_python_method_count = 0
    retrieval_engine_fallback_count = 0
    for method in approach_spec.methods:
        ranked, method_engine, method_fallback_count = _rank_method_window(
            method=method,
            query=query,
            candidate_summaries=candidate_summaries,
            max_block_component_size=max_block_component_size,
            max_ranked_clusters=max_ranked_clusters,
            rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
            rust_num_threads=rust_num_threads,
            frozen_rust_hybrid_centroid_policy=frozen_rust_hybrid_centroid_policy,
            retrieval_engine=engine,
        )
        ranked_by_method[method] = ranked
        retrieval_engine_rust_method_count += int(method_engine == "rust")
        retrieval_engine_python_method_count += int(method_engine == "python")
        retrieval_engine_fallback_count += int(method_fallback_count)
    if approach_spec.mode == "all":
        primary_method = approach_spec.methods[0]
        ranked = ranked_by_method[primary_method]
        ranked_component_keys = [summary.component_key for _score, summary in ranked]
        retrieval_scores = {summary.component_key: float(score) for score, summary in ranked}
        retrieval_ranks = {component_key: rank for rank, component_key in enumerate(ranked_component_keys, start=1)}
        ambiguity_expanded = 0
    elif approach_spec.mode == "all_union":
        ranked_component_keys, retrieval_scores, retrieval_ranks = _combine_union_method_windows(
            ranked_by_method=ranked_by_method,
            max_ranked_clusters=max_ranked_clusters,
        )
        ambiguity_expanded = 0
    elif approach_spec.mode == "ambiguous_union":
        primary_method = approach_spec.methods[0]
        if _should_expand_ambiguously(
            ranked_by_method[primary_method],
            profiles_by_component=profiles_by_component,
        ):
            ranked_component_keys, retrieval_scores, retrieval_ranks = _combine_union_method_windows(
                ranked_by_method=ranked_by_method,
                max_ranked_clusters=max_ranked_clusters,
            )
            ambiguity_expanded = 1
        else:
            ranked = ranked_by_method[primary_method]
            ranked_component_keys = [summary.component_key for _score, summary in ranked]
            retrieval_scores = {summary.component_key: float(score) for score, summary in ranked}
            retrieval_ranks = {component_key: rank for rank, component_key in enumerate(ranked_component_keys, start=1)}
            ambiguity_expanded = 0
    else:  # pragma: no cover - parser guards this already
        raise ValueError(f"Unsupported retrieval mode {approach_spec.mode!r}")
    return (
        ranked_component_keys,
        retrieval_scores,
        retrieval_ranks,
        {
            "candidate_components": int(len(raw_candidate_summaries)),
            "candidate_signatures": int(sum(summary.size for summary in raw_candidate_summaries)),
            "preselected_candidate_components": int(len(candidate_summaries_pre_filter)),
            "preselected_candidate_signatures": int(sum(summary.size for summary in candidate_summaries_pre_filter)),
            "scored_candidate_components": int(filter_state["scored_candidate_components"]),
            "scored_candidate_signatures": int(filter_state["scored_candidate_signatures"]),
            "orcid_filter_applied": int(filter_state["orcid_filter_applied"]),
            "middle_initial_filter_applied": int(filter_state["middle_initial_filter_applied"]),
            "year_range_filter_applied": int(filter_state["year_range_filter_applied"]),
            "ambiguity_expanded": int(ambiguity_expanded),
            "retrieval_engine_rust_method_count": int(retrieval_engine_rust_method_count),
            "retrieval_engine_python_method_count": int(retrieval_engine_python_method_count),
            "retrieval_engine_fallback_count": int(retrieval_engine_fallback_count),
            "name_compat_selector_fallback_count": int(
                retrieval_subblock_index.get(RUST_NAME_COMPATIBLE_SUBBLOCK_SELECTOR_FALLBACK_COUNT_KEY, 0) or 0
            )
            if retrieval_subblock_index is not None
            else 0,
        },
    )


def _update_cluster_stats_from_batch(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    batch_pairs: list[tuple[str, str]],
    batch_cluster_ids: list[str],
    stats_by_cluster_id: dict[str, ClusterPairwiseStats],
    max_top_k: int,
) -> tuple[int, float, float]:
    """Score one pair batch and fold it into per-cluster stats."""

    if not batch_pairs:
        return 0, 0.0, 0.0
    labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
        dataset,
        batch_pairs,
        partial_supervision={},
        runtime_context=runtime_context,
        incremental_dont_use_cluster_seeds=False,
        constraint_backend=constraint_backend,
    )
    featurize_start = time.perf_counter()
    all_pairs = [(left, right, label) for (left, right), label in zip(batch_pairs, labels, strict=True)]
    batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
        all_pairs,
        dataset,
        clusterer.featurizer_info,
        clusterer.n_jobs,
        use_cache=clusterer.use_cache,
        chunk_size=DEFAULT_CHUNK_SIZE,
        nameless_featurizer_info=clusterer.nameless_featurizer_info,
    )
    featurize_seconds = float(time.perf_counter() - featurize_start)
    batch_predictions, model_predict_seconds = _predict_and_combine(
        clusterer.classifier,
        clusterer.nameless_classifier,
        batch_features,
        batch_labels,
        batch_nameless_features,
        "single_letter_reranker",
        num_threads=clusterer.n_jobs,
    )
    for cluster_id, distance in zip(batch_cluster_ids, batch_predictions, strict=True):
        stats_by_cluster_id[cluster_id].update(float(distance), max_top_k=max_top_k)
    return len(batch_pairs), float(featurize_seconds), float(model_predict_seconds)


def _initialize_query_cluster_stats(
    request: QueryClusterStatsRequest,
) -> dict[str, ClusterPairwiseStats]:
    """Create empty per-cluster accumulators for one query request."""

    stats_by_cluster_id: dict[str, ClusterPairwiseStats] = {}
    for component_key in request.shortlist_component_keys:
        summary = request.summary_by_component[component_key]
        profile = build_cluster_profile(summary)
        stats_by_cluster_id[component_key] = ClusterPairwiseStats(
            cluster_id=str(component_key),
            retrieval_rank=int(request.retrieval_ranks[component_key]),
            retrieval_score=float(request.retrieval_scores[component_key]),
            cluster_size=int(summary.size),
            family_id=str(profile.family_id),
            dominant_first_name=profile.dominant_first_name,
            family_dominance_ratio=float(profile.family_dominance_ratio),
            family_named_count=int(profile.family_named_count),
        )
    return stats_by_cluster_id


def _constraint_label_is_disallow(label: float) -> bool:
    """Return whether one offset constraint label encodes a hard disallow."""

    if math.isnan(float(label)):
        return False
    return float(label) + float(LARGE_INTEGER) >= float(LARGE_DISTANCE)


def _constraint_label_is_require(label: float) -> bool:
    """Return whether one offset constraint label encodes a hard require."""

    if math.isnan(float(label)):
        return False
    return abs(float(label) + float(LARGE_INTEGER)) <= 1e-9


def _update_query_cluster_stats_from_multi_query_batch(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    batch_pairs: list[tuple[str, str]],
    batch_request_indices: list[int],
    batch_cluster_ids: list[str],
    batch_incremental_dont_use_cluster_seeds: list[bool],
    batch_ignore_disallow_constraints: list[bool],
    stats_by_request: list[dict[str, ClusterPairwiseStats]],
    diagnostics_by_request: list[dict[str, float]],
    max_top_k: int,
) -> None:
    """Score one mixed-query batch and fold results back into per-query stats."""

    if not batch_pairs:
        return
    labels, _telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
        dataset,
        batch_pairs,
        partial_supervision={},
        runtime_context=runtime_context,
        incremental_dont_use_cluster_seeds=False,
        constraint_backend=constraint_backend,
    )
    override_indices = [
        index for index, should_override in enumerate(batch_incremental_dont_use_cluster_seeds) if bool(should_override)
    ]
    if override_indices:
        override_pairs = [batch_pairs[index] for index in override_indices]
        override_labels, _override_telemetry = clusterer._resolve_constraint_batch(  # noqa: SLF001
            dataset,
            override_pairs,
            partial_supervision={},
            runtime_context=runtime_context,
            incremental_dont_use_cluster_seeds=True,
            constraint_backend=constraint_backend,
        )
        for index, override_label in zip(override_indices, override_labels, strict=True):
            labels[index] = float(override_label)
    for index, label in enumerate(labels):
        request_index = int(batch_request_indices[index])
        cluster_id = str(batch_cluster_ids[index])
        if _constraint_label_is_require(float(label)):
            stats_by_request[request_index][cluster_id].require_pair_count += 1
        if _constraint_label_is_disallow(float(label)) and not bool(batch_ignore_disallow_constraints[index]):
            stats_by_request[request_index][cluster_id].disallow_pair_count += 1
    for index, should_ignore_disallow in enumerate(batch_ignore_disallow_constraints):
        if bool(should_ignore_disallow) and _constraint_label_is_disallow(float(labels[index])):
            labels[index] = float(np.nan)
    featurize_start = time.perf_counter()
    all_pairs = [(left, right, label) for (left, right), label in zip(batch_pairs, labels, strict=True)]
    batch_features, batch_labels, batch_nameless_features = many_pairs_featurize(
        all_pairs,
        dataset,
        clusterer.featurizer_info,
        clusterer.n_jobs,
        use_cache=clusterer.use_cache,
        chunk_size=DEFAULT_CHUNK_SIZE,
        nameless_featurizer_info=clusterer.nameless_featurizer_info,
    )
    featurize_seconds = float(time.perf_counter() - featurize_start)
    batch_predictions, model_predict_seconds = _predict_and_combine(
        clusterer.classifier,
        clusterer.nameless_classifier,
        batch_features,
        batch_labels,
        batch_nameless_features,
        "single_letter_reranker",
        num_threads=clusterer.n_jobs,
    )
    request_pair_counts: Counter[int] = Counter()
    for request_index, cluster_id, distance in zip(
        batch_request_indices,
        batch_cluster_ids,
        batch_predictions,
        strict=True,
    ):
        stats_by_request[request_index][cluster_id].update(float(distance), max_top_k=max_top_k)
        request_pair_counts[int(request_index)] += 1
    batch_pair_count = max(1, len(batch_pairs))
    for request_index, pair_count in request_pair_counts.items():
        pair_share = float(pair_count) / float(batch_pair_count)
        diagnostics = diagnostics_by_request[int(request_index)]
        diagnostics["pair_count"] += float(pair_count)
        diagnostics["featurize_seconds"] += float(featurize_seconds) * float(pair_share)
        diagnostics["model_predict_seconds"] += float(model_predict_seconds) * float(pair_share)


def compute_query_cluster_stats_batched(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    requests: Sequence[QueryClusterStatsRequest],
    pair_batch_size: int,
    max_top_k: int = DEFAULT_CHOOSER_CACHE_MAX_TOP_K,
) -> list[tuple[dict[str, ClusterPairwiseStats], dict[str, Any]]]:
    """Aggregate query-to-cluster stats for many query windows at once."""

    if int(pair_batch_size) <= 0:
        raise ValueError(f"pair_batch_size must be positive, got {pair_batch_size}")

    stats_by_request = [_initialize_query_cluster_stats(request) for request in requests]
    diagnostics_by_request = [
        {
            "pair_count": 0.0,
            "featurize_seconds": 0.0,
            "model_predict_seconds": 0.0,
        }
        for _request in requests
    ]
    batch_pairs: list[tuple[str, str]] = []
    batch_request_indices: list[int] = []
    batch_cluster_ids: list[str] = []
    batch_incremental_dont_use_cluster_seeds: list[bool] = []
    batch_ignore_disallow_constraints: list[bool] = []

    def flush_batch() -> None:
        nonlocal batch_pairs
        nonlocal batch_request_indices
        nonlocal batch_cluster_ids
        nonlocal batch_incremental_dont_use_cluster_seeds
        nonlocal batch_ignore_disallow_constraints
        _update_query_cluster_stats_from_multi_query_batch(
            clusterer=clusterer,
            dataset=dataset,
            runtime_context=runtime_context,
            constraint_backend=constraint_backend,
            batch_pairs=batch_pairs,
            batch_request_indices=batch_request_indices,
            batch_cluster_ids=batch_cluster_ids,
            batch_incremental_dont_use_cluster_seeds=batch_incremental_dont_use_cluster_seeds,
            batch_ignore_disallow_constraints=batch_ignore_disallow_constraints,
            stats_by_request=stats_by_request,
            diagnostics_by_request=diagnostics_by_request,
            max_top_k=max_top_k,
        )
        batch_pairs = []
        batch_request_indices = []
        batch_cluster_ids = []
        batch_incremental_dont_use_cluster_seeds = []
        batch_ignore_disallow_constraints = []

    for request_index, request in enumerate(requests):
        for component_key in request.shortlist_component_keys:
            signature_ids = request.candidate_signature_ids_by_component[component_key]
            should_bypass_cluster_seeds = (
                str(component_key) in request.incremental_dont_use_cluster_seeds_component_keys
            )
            should_ignore_disallow_constraints = (
                str(component_key) in request.ignore_disallow_constraints_component_keys
            )
            for signature_id in signature_ids:
                batch_pairs.append((str(request.query_signature_id), str(signature_id)))
                batch_request_indices.append(int(request_index))
                batch_cluster_ids.append(str(component_key))
                batch_incremental_dont_use_cluster_seeds.append(bool(should_bypass_cluster_seeds))
                batch_ignore_disallow_constraints.append(bool(should_ignore_disallow_constraints))
                if len(batch_pairs) >= int(pair_batch_size):
                    flush_batch()
    if batch_pairs:
        flush_batch()

    return [
        (
            stats_by_request[request_index],
            {
                "pair_count": int(round(float(diagnostics_by_request[request_index]["pair_count"]))),
                "featurize_seconds": round(float(diagnostics_by_request[request_index]["featurize_seconds"]), 6),
                "model_predict_seconds": round(
                    float(diagnostics_by_request[request_index]["model_predict_seconds"]),
                    6,
                ),
            },
        )
        for request_index in range(len(requests))
    ]


def _counter_query_overlap(query_values: frozenset[str], counter: Counter[str], size: int) -> float:
    if size <= 0 or not query_values or not counter:
        return 0.0
    overlap = sum(float(counter[value]) / float(size) for value in query_values if value in counter)
    return float(overlap / float(len(query_values)))


def _middle_initial_compatibility(query: retrieval.QueryFeatures, summary: retrieval.ClusterSummary) -> float:
    if not query.middle_initials or not summary.middle_initial_counts or summary.size <= 0:
        return 0.0
    overlap = query.middle_initials.intersection(summary.middle_initial_counts.keys())
    if overlap:
        return float(
            sum(float(summary.middle_initial_counts[value]) / float(summary.size) for value in overlap)
            / float(len(query.middle_initials))
        )
    return retrieval.RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE


def _year_compatibility(query_year: int | None, summary: retrieval.ClusterSummary) -> float:
    if query_year is None or summary.year_mean is None:
        return 0.0
    distance = abs(float(query_year) - float(summary.year_mean))
    score = max(0.0, 1.0 - (distance / retrieval.RETRIEVAL_YEAR_SCORE_DECAY_YEARS))
    if summary.year_min is not None and summary.year_max is not None:
        if (
            query_year < int(summary.year_min) - retrieval.RETRIEVAL_YEAR_SCORE_RANGE_GAP
            or query_year > int(summary.year_max) + retrieval.RETRIEVAL_YEAR_SCORE_RANGE_GAP
        ):
            score -= retrieval.RETRIEVAL_YEAR_SCORE_RANGE_PENALTY
    return float(score)


def _title_overlap(query: retrieval.QueryFeatures, summary: retrieval.ClusterSummary) -> float:
    return float(_counter_query_overlap(query.title_terms, summary.title_counts, summary.size))


def _specter_centroid_similarity(query: retrieval.QueryFeatures, summary: retrieval.ClusterSummary) -> float:
    query_vector = getattr(query, "specter", None)
    summary_vector = getattr(summary, "specter_centroid", None)
    if query_vector is None or summary_vector is None:
        return 0.0
    denom = float(np.linalg.norm(query_vector) * np.linalg.norm(summary_vector))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(query_vector, summary_vector) / denom)


def _specter_exemplar_similarity(query: retrieval.QueryFeatures, summary: retrieval.ClusterSummary) -> float:
    query_vector = getattr(query, "specter", None)
    exemplar_vectors = list(getattr(summary, "exemplar_vectors", []) or [])
    if query_vector is None or not exemplar_vectors:
        return 0.0
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 0.0:
        return 0.0
    best = 0.0
    for exemplar in exemplar_vectors:
        exemplar_norm = float(np.linalg.norm(exemplar))
        if exemplar_norm <= 0.0:
            continue
        best = max(best, float(np.dot(query_vector, exemplar) / float(query_norm * exemplar_norm)))
    return float(best)


def _name_count_rarity(value: Any) -> float:
    """Convert an S2 corpus name count into a finite rarity score."""

    if value is None:
        return 0.0
    try:
        count = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(count) or count <= 0.0:
        return 0.0
    return float(1.0 / math.sqrt(count))


def _name_count_attr(value: Any, field_name: str) -> float | None:
    """Return one numeric field from a NameCounts-like object."""

    raw_value = getattr(value, field_name, None)
    if raw_value is None:
        return None
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _candidate_name_count_rarity_features(summary: retrieval.ClusterSummary) -> dict[str, float]:
    """Return visible candidate-component rarity from candidate signature counts."""

    minima: dict[str, float] = {}
    for candidate_name_counts in tuple(getattr(summary, "name_counts_values", ()) or ()):
        for field_name in ("first", "first_last", "last", "last_first_initial"):
            value = _name_count_attr(candidate_name_counts, field_name)
            if value is None:
                continue
            minima[field_name] = min(value, minima.get(field_name, value))

    return {
        "candidate_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first")), 6),
        "candidate_last_first_name_count_min_rarity": round(_name_count_rarity(minima.get("first_last")), 6),
        "candidate_last_name_count_min_rarity": round(_name_count_rarity(minima.get("last")), 6),
        "candidate_last_first_initial_count_min_rarity": round(
            _name_count_rarity(minima.get("last_first_initial")),
            6,
        ),
    }


def _name_count_rarity_features(
    query: retrieval.QueryFeatures,
    summary: retrieval.ClusterSummary,
) -> dict[str, float]:
    """Return component-level rarity features from the pairwise name-count block."""

    candidate_features = _candidate_name_count_rarity_features(summary)
    query_name_counts = getattr(query, "name_counts", None)
    candidate_name_counts_values = tuple(getattr(summary, "name_counts_values", ()) or ())
    if query_name_counts is None or not candidate_name_counts_values:
        return {
            **{column: 0.0 for column in NAME_COUNT_RARITY_FEATURE_COLUMNS if column not in candidate_features},
            **candidate_features,
        }

    observed_minima: dict[str, float] = {}
    for candidate_name_counts in candidate_name_counts_values:
        if candidate_name_counts is None:
            continue
        values = pairwise_name_counts(query_name_counts, candidate_name_counts)
        for feature_name, raw_value in zip(PAIRWISE_NAME_COUNT_FEATURE_NAMES, values, strict=True):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            observed_minima[feature_name] = min(value, observed_minima.get(feature_name, value))

    features = {
        f"{feature_name}_rarity": round(_name_count_rarity(observed_minima.get(feature_name)), 6)
        for feature_name in PAIRWISE_NAME_COUNT_FEATURE_NAMES
    }
    if not bool(query.has_full_first):
        for column in (
            "first_name_count_min_rarity",
            "last_first_name_count_min_rarity",
            "last_first_initial_count_min_rarity",
            "first_name_count_max_rarity",
            "last_first_name_count_max_rarity",
        ):
            features[column] = 0.0
    first_prefix_match = 0.0
    query_first = str(query.first or "")
    if len(query_first) > 1 and int(summary.size) > 0:
        for candidate_first, count in summary.first_name_counts.items():
            if len(candidate_first) > 1 and same_prefix_tokens(query_first, candidate_first):
                first_prefix_match = max(first_prefix_match, float(count) / float(summary.size))
    features["first_prefix_x_last_first_name_count_min_rarity"] = round(
        float(first_prefix_match) * float(features["last_first_name_count_min_rarity"]),
        6,
    )
    return {**features, **candidate_features}


def _rust_summary_feature_overrides(
    *,
    query_features: retrieval.QueryFeatures,
    shortlist_component_keys: list[str],
    summary_by_component: dict[str, retrieval.ClusterSummary],
    rust_hybrid_centroid_retriever: RustHybridCentroidRetrieverHandle | None,
) -> dict[str, dict[str, float]]:
    """Resolve Rust-backed chooser summary features when the retriever supports them."""

    if rust_hybrid_centroid_retriever is None:
        return {}
    return compute_chooser_summary_features_rust_hybrid_centroid(
        query=query_features,
        component_keys=[str(component_key) for component_key in shortlist_component_keys],
        summary_by_component=summary_by_component,
        retriever=rust_hybrid_centroid_retriever,
    )


def count_normalized_confidence(stats: ClusterPairwiseStats, *, max_pair_count_in_group: int) -> float:
    """Return a simple support-aware confidence signal from the pairwise stats."""

    if stats.count <= 0 or max_pair_count_in_group <= 0:
        return 0.0
    topk_distance = stats.topk_mean_distance(int(COUNT_NORMALIZED_CONFIDENCE_TOP_K))
    if not math.isfinite(topk_distance):
        return 0.0
    support = math.log1p(float(stats.count)) / math.log1p(float(max_pair_count_in_group))
    support = float(support) ** float(COUNT_NORMALIZED_CONFIDENCE_SUPPORT_GAMMA)
    quality = max(0.0, 1.0 - float(topk_distance))
    return float(support * quality)


def cluster_size_log_capped(cluster_size: Any) -> float:
    """Return a capped log-size prior anchored to the train p95 component size."""

    size = max(0.0, float(cluster_size or 0.0))
    if size <= 0.0:
        return 0.0
    reference = float(CLUSTER_SIZE_LOG_CAPPED_REFERENCE_SIZE)
    return float(min(1.0, math.log1p(size) / math.log1p(reference)))


def _best_competitor_component_key(
    sorted_component_keys: list[str],
    *,
    current_component_key: str,
) -> str | None:
    for component_key in sorted_component_keys:
        if component_key != current_component_key:
            return str(component_key)
    return None


def _validate_raw_similarity_features_for_components(
    *,
    kept_component_keys: Sequence[str],
    raw_similarity_features_by_component: dict[str, dict[str, float]] | None,
) -> None:
    """Require explicit raw metadata similarity features for emitted rows."""

    if not kept_component_keys:
        return
    required_features = set(RAW_METADATA_SIMILARITY_FEATURE_COLUMNS)
    missing_components: list[str] = []
    missing_features_by_component: dict[str, list[str]] = {}
    for component_key in kept_component_keys:
        component_key = str(component_key)
        component_features = (
            raw_similarity_features_by_component.get(component_key)
            if raw_similarity_features_by_component is not None
            else None
        )
        if component_features is None:
            missing_components.append(component_key)
            continue
        missing_features = sorted(required_features - set(component_features))
        if missing_features:
            missing_features_by_component[component_key] = missing_features
    if missing_components or missing_features_by_component:
        raise ValueError(
            "Missing raw metadata similarity features for candidate row generation: "
            f"missing_components={missing_components} missing_features={missing_features_by_component}"
        )


def make_candidate_rows(
    *,
    query_case: RerankerQueryCase,
    query_view: str,
    query_features: retrieval.QueryFeatures,
    shortlist_component_keys: list[str],
    retrieval_scores: dict[str, float],
    retrieval_ranks: dict[str, int],
    retrieval_window_state: dict[str, int],
    summary_by_component: dict[str, retrieval.ClusterSummary],
    stats_by_component: dict[str, ClusterPairwiseStats],
    rust_hybrid_centroid_retriever: RustHybridCentroidRetrieverHandle | None = None,
    raw_similarity_features_by_component: dict[str, dict[str, float]] | None = None,
    strict_raw_similarity_features: bool = False,
) -> list[dict[str, Any]]:
    """Convert one retrieved candidate window into persisted reranker rows."""

    if not shortlist_component_keys:
        return []
    sorted_component_keys = sorted(
        shortlist_component_keys,
        key=lambda component_key: (int(retrieval_ranks[component_key]), str(component_key)),
    )
    hard_disallow_filter = apply_hard_disallow_component_filter(
        sorted_component_keys,
        disallow_pair_count_by_component={
            str(component_key): int(stats_by_component[component_key].disallow_pair_count)
            for component_key in sorted_component_keys
        },
        preserve_component_keys=query_case.positive_component_keys,
    )
    kept_component_keys = list(hard_disallow_filter.kept_component_keys)
    if not kept_component_keys:
        return []
    if strict_raw_similarity_features:
        _validate_raw_similarity_features_for_components(
            kept_component_keys=kept_component_keys,
            raw_similarity_features_by_component=raw_similarity_features_by_component,
        )
    top1_component_key = kept_component_keys[0]
    top1_stats = stats_by_component[top1_component_key]
    # Reranker supervision must come from labeled positives only; pairwise
    # constraints are retrieval/scoring context, not a source of target labels.
    positive_component_keys = frozenset(
        component_key for component_key in kept_component_keys if component_key in query_case.positive_component_keys
    )
    best_positive_retrieval_rank = (
        min(int(retrieval_ranks[component_key]) for component_key in positive_component_keys)
        if positive_component_keys
        else None
    )
    max_pair_count_in_group = max((int(stats_by_component[key].count) for key in kept_component_keys), default=0)
    rust_feature_overrides = _rust_summary_feature_overrides(
        query_features=query_features,
        shortlist_component_keys=kept_component_keys,
        summary_by_component=summary_by_component,
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
    )
    rows: list[dict[str, Any]] = []
    raw_similarity_features_by_component = raw_similarity_features_by_component or {}
    for component_key in kept_component_keys:
        summary = summary_by_component[component_key]
        stats = stats_by_component[component_key]
        rust_features = rust_feature_overrides.get(str(component_key), {})
        raw_similarity_features = raw_similarity_features_by_component.get(str(component_key), {})
        name_count_rarity_features = _name_count_rarity_features(query_features, summary)
        best_competitor_component_key = _best_competitor_component_key(
            kept_component_keys,
            current_component_key=component_key,
        )
        if best_competitor_component_key is None:
            competitor_stats = stats
            retrieval_score_gap_vs_best_competitor = 0.0
            retrieval_rank_gap_vs_best_competitor = 0.0
        else:
            competitor_stats = stats_by_component[best_competitor_component_key]
            retrieval_score_gap_vs_best_competitor = float(
                retrieval_scores[component_key] - retrieval_scores[best_competitor_component_key]
            )
            retrieval_rank_gap_vs_best_competitor = float(
                int(retrieval_ranks[component_key]) - int(retrieval_ranks[best_competitor_component_key])
            )
        same_family_vs_best_competitor = int(
            bool(stats.family_id)
            and bool(competitor_stats.family_id)
            and str(stats.family_id) == str(competitor_stats.family_id)
        )
        same_family_as_top1 = int(
            bool(stats.family_id) and bool(top1_stats.family_id) and str(stats.family_id) == str(top1_stats.family_id)
        )
        family_instability_flag = int(
            int(stats.family_named_count) >= int(GENERIC_FAMILY_MIN_COUNT)
            and float(stats.family_dominance_ratio) < float(GENERIC_FAMILY_MIN_RATIO)
        )
        fragment_flag = int(
            int(summary.size) <= 2 and int(same_family_as_top1) == 1 and str(component_key) != str(top1_component_key)
        )
        row = {
            "source": str(query_case.source),
            "dataset": str(query_case.dataset),
            "query_source": str(query_case.query_source),
            "query_view": str(query_view),
            "natural_query_view": str(query_case.natural_query_view or query_view),
            "query_group_id": f"{query_case.dataset}:{query_case.query_id}:{query_view}",
            "query_id": str(query_case.query_id),
            "query_signature_id": str(query_case.query_signature_id),
            "query_first_token": str(query_features.first) if query_features.first else None,
            "query_first_initial": str(query_features.first_initial) if query_features.first_initial else None,
            "query_year": (int(query_features.year) if query_features.year is not None else None),
            "_audit_normalized_orcid": (
                str(query_case.normalized_orcid) if query_case.normalized_orcid is not None else None
            ),
            "_audit_orcid_group_size": (
                int(query_case.orcid_group_size) if query_case.orcid_group_size is not None else None
            ),
            "_audit_orcid_group_size_bucket": (
                str(query_case.orcid_group_size_bucket) if query_case.orcid_group_size_bucket is not None else None
            ),
            "split": str(query_case.split),
            "block_key": str(query_case.block_key),
            "block_size": int(query_case.block_size),
            "component_size": int(query_case.component_size),
            "sampling_info_bucket": str(query_case.sampling_info_bucket),
            "supervision_type": str(query_case.supervision_type),
            "positive_candidate_count": int(len(positive_component_keys)),
            "positive_candidate_keys": "|".join(sorted(positive_component_keys)),
            "group_has_positive": int(bool(positive_component_keys)),
            "best_positive_retrieval_rank": (
                int(best_positive_retrieval_rank) if best_positive_retrieval_rank is not None else None
            ),
            "support_type": str(query_case.support_type),
            "query_in_seed_before_holdout": int(bool(query_case.query_in_seed_before_holdout)),
            "candidate_component_key": str(component_key),
            "candidate_cluster_id": str(summary.cluster_id),
            "best_competitor_component_key": (
                str(best_competitor_component_key) if best_competitor_component_key is not None else None
            ),
            "family_id": str(stats.family_id),
            "dominant_first_name": (str(stats.dominant_first_name) if stats.dominant_first_name is not None else None),
            "candidate_year_min": (int(summary.year_min) if summary.year_min is not None else None),
            "candidate_year_max": (int(summary.year_max) if summary.year_max is not None else None),
            "label": int(component_key in positive_component_keys),
            "candidate_count": int(len(kept_component_keys)),
            "candidate_signatures": int(sum(summary_by_component[key].size for key in kept_component_keys)),
            "scored_candidate_components": int(retrieval_window_state["scored_candidate_components"]),
            "scored_candidate_signatures": int(retrieval_window_state["scored_candidate_signatures"]),
            "orcid_filter_applied": int(retrieval_window_state["orcid_filter_applied"]),
            "middle_initial_filter_applied": int(retrieval_window_state["middle_initial_filter_applied"]),
            "year_range_filter_applied": int(retrieval_window_state["year_range_filter_applied"]),
            "retrieval_rank": int(retrieval_ranks[component_key]),
            "retrieval_score": round(float(retrieval_scores[component_key]), 6),
            "cluster_size": int(summary.size),
            "pair_count": int(stats.count),
            "min_distance": round(float(stats.min_distance), 6),
            "mean_distance": round(float(stats.mean_distance), 6),
            "top3_mean_distance": round(float(stats.topk_mean_distance(3)), 6),
            "top5_mean_distance": round(float(stats.topk_mean_distance(5)), 6),
            "top10_mean_distance": round(float(stats.topk_mean_distance(10)), 6),
            "top20_mean_distance": round(float(stats.topk_mean_distance(20)), 6),
            "count_normalized_confidence": round(
                float(count_normalized_confidence(stats, max_pair_count_in_group=max_pair_count_in_group)),
                6,
            ),
            "retrieval_score_gap_vs_best_competitor": round(float(retrieval_score_gap_vs_best_competitor), 6),
            "retrieval_rank_gap_vs_best_competitor": round(float(retrieval_rank_gap_vs_best_competitor), 6),
            "top3_mean_delta_vs_best_competitor": round(
                float(competitor_stats.topk_mean_distance(3) - stats.topk_mean_distance(3)),
                6,
            ),
            "top5_mean_delta_vs_best_competitor": round(
                float(competitor_stats.topk_mean_distance(5) - stats.topk_mean_distance(5)),
                6,
            ),
            "cluster_size_ratio_vs_best_competitor": round(
                float(summary.size / max(1, summary_by_component[best_competitor_component_key].size))
                if best_competitor_component_key is not None
                else 1.0,
                6,
            ),
            "same_family_vs_best_competitor": int(same_family_vs_best_competitor),
            "same_family_as_top1": int(same_family_as_top1),
            "middle_initial_compatibility": round(
                float(
                    rust_features.get(
                        "middle_initial_compatibility", _middle_initial_compatibility(query_features, summary)
                    )
                ),
                6,
            ),
            "affiliation_overlap": round(
                float(
                    rust_features.get(
                        "affiliation_overlap",
                        _counter_query_overlap(
                            query_features.affiliation_terms,
                            summary.affiliation_counts,
                            summary.size,
                        ),
                    )
                ),
                6,
            ),
            "coauthor_overlap": round(
                float(
                    rust_features.get(
                        "coauthor_overlap",
                        _counter_query_overlap(query_features.coauthor_blocks, summary.coauthor_counts, summary.size),
                    )
                ),
                6,
            ),
            "venue_overlap": round(
                float(
                    rust_features.get(
                        "venue_overlap",
                        _counter_query_overlap(query_features.venue_terms, summary.venue_counts, summary.size),
                    )
                ),
                6,
            ),
            "year_compatibility": round(
                float(rust_features.get("year_compatibility", _year_compatibility(query_features.year, summary))),
                6,
            ),
            "title_overlap": round(
                float(rust_features.get("title_overlap", _title_overlap(query_features, summary))),
                6,
            ),
            **{
                feature_name: round(float(raw_similarity_features.get(feature_name, 0.0) or 0.0), 6)
                for feature_name in RAW_METADATA_SIMILARITY_FEATURE_COLUMNS
            },
            **name_count_rarity_features,
            "specter_centroid_similarity": round(
                float(
                    rust_features.get(
                        "specter_centroid_similarity",
                        _specter_centroid_similarity(query_features, summary),
                    )
                ),
                6,
            ),
            "specter_exemplar_similarity": round(
                float(
                    rust_features.get(
                        "specter_exemplar_similarity",
                        _specter_exemplar_similarity(query_features, summary),
                    )
                ),
                6,
            ),
            "family_instability_flag": int(family_instability_flag),
            "fragment_flag": int(fragment_flag),
            "query_has_specter": int(query_features.has_specter),
            "query_has_coauthors": int(query_features.has_coauthors),
            "query_has_affiliations": int(query_features.has_affiliations),
            "query_has_middle": int(query_features.has_middle),
            "query_has_full_first": int(query_features.has_full_first),
        }
        rows.append(row)
    return rows


def write_dict_rows_csv(path: Path, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    """Write dictionaries to CSV with an explicit stable field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[str(value) for value in fieldnames])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def append_dict_rows_csv(path: Path, rows: Sequence[dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    """Append dictionaries to CSV with an explicit stable field order."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[str(value) for value in fieldnames])
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _read_typed_rows_csv(
    path: Path,
    *,
    int_columns: set[str] | None = None,
    float_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read CSV rows using explicit integer and float column sets."""

    resolved_int_columns = set(int_columns or set())
    resolved_float_columns = set(float_columns or set())
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                if value == "":
                    row[key] = None
                elif key in resolved_int_columns:
                    row[key] = int(value)
                elif key in resolved_float_columns:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write candidate rows to a CSV file with a stable column order."""

    write_dict_rows_csv(path, rows, fieldnames=ROW_COLUMNS)


def append_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append candidate rows to a CSV file with a stable column order."""

    append_dict_rows_csv(path, rows, fieldnames=ROW_COLUMNS)


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    """Read persisted reranker rows from ``path``."""

    return _read_typed_rows_csv(path, int_columns=INT_COLUMNS, float_columns=FLOAT_COLUMNS)


def write_query_group_metadata_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one row per query group for cached sampler selection."""

    write_dict_rows_csv(path, rows, fieldnames=QUERY_GROUP_METADATA_COLUMNS)


def append_query_group_metadata_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Append one row per query group for cached sampler selection."""

    append_dict_rows_csv(path, rows, fieldnames=QUERY_GROUP_METADATA_COLUMNS)


def read_query_group_metadata_csv(path: Path) -> list[dict[str, Any]]:
    """Read persisted query-group metadata rows."""

    return _read_typed_rows_csv(
        path,
        int_columns=QUERY_GROUP_METADATA_INT_COLUMNS,
    )


def write_materialized_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write base rows plus all derived numeric features."""

    write_dict_rows_csv(path, rows, fieldnames=MATERIALIZED_DERIVED_ROW_COLUMNS)


def read_materialized_rows_csv(path: Path) -> list[dict[str, Any]]:
    """Read rows that already contain all derived numeric feature columns."""

    return _read_typed_rows_csv(
        path,
        int_columns=INT_COLUMNS,
        float_columns=FLOAT_COLUMNS | DERIVED_FEATURE_COLUMNS,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group candidate rows by their persisted query-group ID."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_group_id"])].append(dict(row))
    for group_id in grouped:
        grouped[group_id].sort(
            key=lambda row: (
                int(row["retrieval_rank"]),
                str(row["candidate_component_key"]),
            )
        )
    return dict(grouped)


def positive_rank_bucket(rank: int | None) -> str:
    """Bucket a positive retrieval rank for summaries and cached sampling."""

    if rank is None:
        return "missing"
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2_3"
    if rank <= 10:
        return "4_10"
    if rank <= 25:
        return "11_25"
    if rank <= 50:
        return "26_50"
    return "51_plus"


def select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    datasets: Sequence[str] | None = None,
    query_view: str | None = None,
    query_views: Sequence[str] | None = None,
    window_size: int | None = None,
    selected_query_group_ids: set[str] | None = None,
    query_sources: Sequence[str] | None = None,
    supervision_types: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter rows by dataset, query view, and candidate-window size."""

    dataset_filter = {str(value) for value in datasets} if datasets is not None else None
    query_group_filter = (
        {str(value) for value in selected_query_group_ids} if selected_query_group_ids is not None else None
    )
    query_view_filter = {str(value) for value in query_views} if query_views is not None else None
    if query_view is not None:
        if query_view_filter is None:
            query_view_filter = {str(query_view)}
        else:
            query_view_filter.add(str(query_view))
    query_source_filter = {str(value) for value in query_sources} if query_sources is not None else None
    supervision_type_filter = {str(value) for value in supervision_types} if supervision_types is not None else None
    split_filter = {str(value) for value in splits} if splits is not None else None
    selected: list[dict[str, Any]] = []
    for row in rows:
        if dataset_filter is not None and str(row["dataset"]) not in dataset_filter:
            continue
        if query_group_filter is not None and str(row["query_group_id"]) not in query_group_filter:
            continue
        if query_view_filter is not None and str(row["query_view"]) not in query_view_filter:
            continue
        if query_source_filter is not None and str(row.get("query_source", row["source"])) not in query_source_filter:
            continue
        if (
            supervision_type_filter is not None
            and str(row.get("supervision_type", "labeled")) not in supervision_type_filter
        ):
            continue
        if split_filter is not None and str(row.get("split", "all")) not in split_filter:
            continue
        if window_size is not None and int(row["retrieval_rank"]) > int(window_size):
            continue
        selected.append(dict(row))
    return selected


def summarize_query_group_rows(
    rows: Sequence[dict[str, Any]],
    *,
    block_component_count: int,
) -> dict[str, Any]:
    """Summarize one persisted query group for cached sampler selection."""

    if not rows:
        raise ValueError("Expected at least one row to summarize a query group")
    ordered_rows = sorted(
        (dict(row) for row in rows),
        key=lambda row: (int(row["retrieval_rank"]), str(row["candidate_component_key"])),
    )
    first_row = ordered_rows[0]
    retrieval_top1_row = ordered_rows[0]
    positive_rows = [row for row in ordered_rows if int(row["label"]) == 1]
    positive_candidate_count = int(len(positive_rows))
    best_positive_retrieval_rank = min(int(row["retrieval_rank"]) for row in positive_rows) if positive_rows else None
    retrieval_top1_is_positive = (
        int(best_positive_retrieval_rank == 1) if best_positive_retrieval_rank is not None else 0
    )
    cross_family_top1_vs_positive = 0
    if positive_rows and not retrieval_top1_is_positive and _has_confident_family_assignment(retrieval_top1_row):
        positive_family_ids = {
            str(row["family_id"]) for row in positive_rows if _has_confident_family_assignment(row) and row["family_id"]
        }
        if positive_family_ids:
            cross_family_top1_vs_positive = int(str(retrieval_top1_row["family_id"]) not in positive_family_ids)
    return {
        "source": str(first_row["source"]),
        "dataset": str(first_row["dataset"]),
        "query_source": str(first_row.get("query_source", first_row["source"])),
        "query_view": str(first_row["query_view"]),
        "natural_query_view": str(first_row.get("natural_query_view", first_row["query_view"])),
        "query_group_id": str(first_row["query_group_id"]),
        "query_id": str(first_row["query_id"]),
        "query_signature_id": str(first_row["query_signature_id"]),
        "_audit_normalized_orcid": (
            str(first_row["_audit_normalized_orcid"]) if first_row.get("_audit_normalized_orcid") is not None else None
        ),
        "_audit_orcid_group_size": (
            int(first_row["_audit_orcid_group_size"]) if first_row.get("_audit_orcid_group_size") is not None else None
        ),
        "_audit_orcid_group_size_bucket": (
            str(first_row["_audit_orcid_group_size_bucket"])
            if first_row.get("_audit_orcid_group_size_bucket") is not None
            else None
        ),
        "split": str(first_row.get("split", "all")),
        "block_key": str(first_row["block_key"]),
        "block_size": int(first_row["block_size"]),
        "block_component_count": int(block_component_count),
        "component_size": int(first_row["component_size"]),
        "sampling_info_bucket": str(first_row["sampling_info_bucket"]),
        "supervision_type": str(first_row.get("supervision_type", "labeled")),
        "support_type": str(first_row["support_type"]),
        "query_in_seed_before_holdout": int(first_row.get("query_in_seed_before_holdout", 0)),
        "group_has_positive": int(positive_candidate_count > 0),
        "positive_candidate_count": int(positive_candidate_count),
        "positive_candidate_keys": "|".join(sorted(str(row["candidate_component_key"]) for row in positive_rows)),
        "best_positive_retrieval_rank": (
            int(best_positive_retrieval_rank) if best_positive_retrieval_rank is not None else None
        ),
        "best_positive_rank_bucket": positive_rank_bucket(best_positive_retrieval_rank),
        "candidate_count": int(first_row["candidate_count"]),
        "candidate_signatures": int(first_row["candidate_signatures"]),
        "scored_candidate_components": int(first_row["scored_candidate_components"]),
        "scored_candidate_signatures": int(first_row["scored_candidate_signatures"]),
        "retrieval_top1_component_key": str(retrieval_top1_row["candidate_component_key"]),
        "retrieval_top1_is_positive": int(retrieval_top1_is_positive),
        "recoverable_non_top1": int(best_positive_retrieval_rank is not None and int(best_positive_retrieval_rank) > 1),
        "cross_family_top1_vs_positive": int(cross_family_top1_vs_positive),
    }


def resolve_feature_columns(
    *,
    feature_preset: str | None = None,
    feature_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Resolve the concrete numeric columns used to train or score the reranker."""

    if feature_preset is not None and feature_columns is not None:
        raise ValueError("Pass either feature_preset or feature_columns, not both")
    if feature_columns is not None:
        resolved = tuple(str(value) for value in feature_columns)
    else:
        preset_name = str(feature_preset or DEFAULT_FEATURE_PRESET)
        if preset_name not in FEATURE_PRESETS:
            raise ValueError(f"Unknown feature preset: {preset_name}. Expected one of {sorted(FEATURE_PRESETS)}")
        resolved = tuple(FEATURE_PRESETS[preset_name])
    unknown_columns = [column for column in resolved if column not in NUMERIC_FEATURE_COLUMNS]
    if unknown_columns:
        raise ValueError(f"Unknown numeric feature columns requested: {unknown_columns}")
    return resolved


def _row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["query_group_id"]), str(row["candidate_component_key"]))


def _coarse_family_key(row: dict[str, Any]) -> str:
    """Return a loose family key from the normalized dominant first name."""

    raw_values = (
        row.get("dominant_first_name"),
        row.get("family_id"),
        row.get("candidate_component_key"),
    )
    for raw_value in raw_values:
        normalized = normalize_text(str(raw_value or ""))
        alpha_only = "".join(character for character in normalized if character.isalpha())
        if alpha_only:
            return str(alpha_only[:3])
    return ""


def _has_confident_family_assignment(row: dict[str, Any]) -> bool:
    family_id = str(row.get("family_id", "") or "")
    component_key = str(row.get("candidate_component_key", "") or "")
    return bool(family_id) and family_id != component_key


def _normalized_alpha(value: Any) -> str:
    """Collapse an arbitrary value down to lowercase alphabetic characters."""

    normalized = normalize_text(str(value or ""))
    return "".join(character for character in normalized if character.isalpha())


def _query_first_initial_from_row(row: dict[str, Any]) -> str:
    """Return the query first-initial signal available in a persisted row."""

    explicit_initial = _normalized_alpha(row.get("query_first_initial"))
    if explicit_initial:
        return explicit_initial[0]
    explicit_token = _normalized_alpha(row.get("query_first_token"))
    if explicit_token:
        return explicit_token[0]
    block_tokens = [_normalized_alpha(token) for token in str(row.get("block_key", "")).split()]
    for token in block_tokens:
        if token:
            return token[0]
    return ""


def _first_name_candidate_compatibility(row: dict[str, Any]) -> float:
    """Score whether a candidate family name is compatible with the query first-name signal."""

    candidate_first = _normalized_alpha(row.get("dominant_first_name"))
    if not candidate_first:
        return 0.0
    query_first = _normalized_alpha(row.get("query_first_token"))
    if query_first:
        if candidate_first == query_first:
            return 1.0
        if candidate_first.startswith(query_first) or query_first.startswith(candidate_first):
            return 1.0
    query_initial = _query_first_initial_from_row(row)
    if query_initial and candidate_first.startswith(query_initial):
        return 1.0
    return 0.0


def _year_mismatch_severity(row: dict[str, Any]) -> float:
    """Return a larger value for stronger year-range contradictions."""

    query_year = row.get("query_year")
    candidate_year_min = row.get("candidate_year_min")
    candidate_year_max = row.get("candidate_year_max")
    if query_year in (None, "") or candidate_year_min in (None, "") or candidate_year_max in (None, ""):
        return 0.0
    query_year_int = int(query_year)
    year_min_int = int(candidate_year_min)
    year_max_int = int(candidate_year_max)
    if query_year_int < year_min_int:
        return round(float(min(1.0, (year_min_int - query_year_int) / 10.0)), 6)
    if query_year_int > year_max_int:
        return round(float(min(1.0, (query_year_int - year_max_int) / 10.0)), 6)
    return 0.0


def _year_missing_flags(row: dict[str, Any]) -> dict[str, int]:
    """Return row-level missing-year indicators without changing feature columns."""

    query_year_missing = int(row.get("query_year") in (None, ""))
    candidate_year_range_missing = int(
        row.get("candidate_year_min") in (None, "") or row.get("candidate_year_max") in (None, "")
    )
    return {
        "query_year_missing": query_year_missing,
        "candidate_year_range_missing": candidate_year_range_missing,
        "any_year_missing": int(query_year_missing or candidate_year_range_missing),
    }


def _affiliation_contradiction_severity(row: dict[str, Any]) -> float:
    """Return a larger value for stronger affiliation contradictions."""

    if int(row.get("query_has_affiliations", 0) or 0) == 0:
        return 0.0
    return round(float(max(0.0, 1.0 - float(row.get("affiliation_overlap", 0.0) or 0.0))), 6)


def _exact_anchor_evidence_flag(row: dict[str, Any]) -> int:
    """Flag exact-title anchor evidence that should strongly support linking."""

    title_overlap = float(row.get("title_overlap", 0.0) or 0.0)
    coauthor_overlap = float(row.get("coauthor_overlap", 0.0) or 0.0)
    affiliation_overlap = float(row.get("affiliation_overlap", 0.0) or 0.0)
    year_compatibility = float(row.get("year_compatibility", 0.0) or 0.0)
    return int(
        title_overlap >= float(EXACT_TITLE_ANCHOR_THRESHOLD)
        and (
            coauthor_overlap >= float(ANCHOR_SUPPORT_OVERLAP_THRESHOLD)
            or affiliation_overlap >= float(ANCHOR_SUPPORT_OVERLAP_THRESHOLD)
            or year_compatibility >= float(ANCHOR_YEAR_COMPATIBILITY_THRESHOLD)
        )
    )


def _candidate_contradiction_signals(row: dict[str, Any]) -> tuple[int, float, int]:
    """Return count, strength, and exact-title-conflict flag for one candidate row."""

    title_overlap = float(row.get("title_overlap", 0.0) or 0.0)
    year_mismatch_severity = _year_mismatch_severity(row)
    affiliation_contradiction_severity = _affiliation_contradiction_severity(row)
    first_name_compatibility = _first_name_candidate_compatibility(row)
    first_name_contradiction = 0.0 if first_name_compatibility > 0.0 else 1.0
    coauthor_contradiction = 0.0
    if int(row.get("query_has_coauthors", 0) or 0) == 1:
        coauthor_contradiction = max(0.0, 1.0 - float(row.get("coauthor_overlap", 0.0) or 0.0))
    contradiction_count = int(year_mismatch_severity >= float(SEVERE_CONTRADICTION_THRESHOLD))
    contradiction_count += int(affiliation_contradiction_severity >= float(SEVERE_CONTRADICTION_THRESHOLD))
    contradiction_count += int(
        coauthor_contradiction >= float(SEVERE_CONTRADICTION_THRESHOLD)
        and title_overlap >= float(EXACT_TITLE_ANCHOR_THRESHOLD)
    )
    contradiction_count += int(first_name_contradiction >= 1.0 and title_overlap >= float(EXACT_TITLE_ANCHOR_THRESHOLD))
    exact_title_identity_conflict_flag = int(
        title_overlap >= float(EXACT_TITLE_ANCHOR_THRESHOLD)
        and (
            year_mismatch_severity >= float(SEVERE_CONTRADICTION_THRESHOLD)
            or affiliation_contradiction_severity >= float(SEVERE_CONTRADICTION_THRESHOLD)
            or first_name_contradiction >= 1.0
        )
    )
    title_anchor = title_overlap >= float(EXACT_TITLE_ANCHOR_THRESHOLD)
    contradiction_score = round(
        float(
            max(
                year_mismatch_severity,
                affiliation_contradiction_severity,
                coauthor_contradiction if title_anchor else 0.0,
                first_name_contradiction if title_anchor else 0.0,
            )
        ),
        6,
    )
    return contradiction_count, contradiction_score, exact_title_identity_conflict_flag


def _float_feature(row: dict[str, Any], column: str, *, default: float = 0.0) -> float:
    """Return one numeric row feature with a stable fallback for derived formulas."""

    value = row.get(column)
    if value is None or value == "":
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return float(numeric)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _anchor_evidence_features(
    row: dict[str, Any],
    *,
    candidate_pair_share_within_coarse_family: float,
    candidate_contradiction_score: float,
) -> dict[str, float]:
    """Return anchor-support features that separate sparse true components from false winners."""

    min_distance = _float_feature(row, "min_distance", default=10000.0)
    specter = _float_feature(row, "specter_exemplar_similarity")
    title = _float_feature(row, "title_overlap")
    coauthor = _float_feature(row, "coauthor_overlap")
    affiliation = _float_feature(row, "affiliation_overlap")
    venue = _float_feature(row, "venue_overlap")
    year = _float_feature(row, "year_compatibility")
    retrieval_gap = _float_feature(row, "retrieval_score_gap_vs_best_competitor")
    same_top1 = _clip01(_float_feature(row, "same_family_as_top1"))
    cluster_size = _float_feature(row, "cluster_size")
    named_signature_count = _float_feature(row, "named_signature_count")
    retrieval_rank = _float_feature(row, "retrieval_rank", default=99.0)
    contradiction = _clip01(float(candidate_contradiction_score))
    candidate_pair_share = _clip01(float(candidate_pair_share_within_coarse_family))

    anchor_count = int(min_distance <= 0.15)
    anchor_count += int(specter >= 0.70)
    anchor_count += int(title >= 0.20)
    anchor_count += int(coauthor >= 0.25)
    anchor_count += int(affiliation >= 0.25)
    anchor_count += int(venue >= 0.20)
    anchor_count += int(year >= 0.90)
    anchor_count += int(retrieval_gap >= 0.02)

    support_strength = (
        0.20 * (1.0 - _clip01(min_distance))
        + 0.20 * _clip01(specter)
        + 0.18 * _clip01(title)
        + 0.18 * _clip01(coauthor)
        + 0.12 * _clip01(affiliation)
        + 0.06 * _clip01(venue)
        + 0.06 * _clip01(year)
    )
    strong_positive_anchor_score = (
        _clip01(support_strength) * (0.5 + 0.5 * same_top1) * (0.35 + 0.65 * _clip01(1.0 - contradiction))
    )

    retrieval_gap_scaled = _clip01((min(0.3, max(-0.2, retrieval_gap)) + 0.2) / 0.5)
    residual_support = (
        0.28 * (1.0 - _clip01(min_distance))
        + 0.20 * _clip01(specter)
        + 0.20 * _clip01(coauthor)
        + 0.14 * _clip01(title)
        + 0.10 * _clip01(year)
        + 0.08 * retrieval_gap_scaled
    )
    tiny_candidate = float(cluster_size <= 2.0 or named_signature_count <= 2.0)
    weak_residual_anchor_score = tiny_candidate * same_top1 * _clip01(residual_support)
    sparse_relative_winner_score = (
        float(retrieval_rank <= 1.0)
        * same_top1
        * _clip01(min(0.3, max(0.0, retrieval_gap)) / 0.3)
        * (1.0 - candidate_pair_share)
        * _clip01(residual_support)
    )
    return {
        "anchor_evidence_count": float(anchor_count),
        "strong_positive_anchor_score": round(float(strong_positive_anchor_score), 6),
        "weak_residual_anchor_score": round(float(weak_residual_anchor_score), 6),
        "sparse_relative_winner_score": round(float(sparse_relative_winner_score), 6),
    }


def _rank_fraction_map(
    rows: Sequence[dict[str, Any]],
    *,
    column: str,
    higher_is_better: bool,
) -> dict[tuple[str, str], float]:
    if not rows:
        return {}
    normalized_values = {_row_identity(row): round(float(row.get(column, 0.0) or 0.0), 12) for row in rows}
    if len(set(normalized_values.values())) == 1:
        return {_row_identity(row): 0.5 for row in rows}
    ordered = sorted(
        rows,
        key=lambda row: (
            -normalized_values[_row_identity(row)] if higher_is_better else normalized_values[_row_identity(row)],
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    denominator = max(1, len(ordered) - 1)
    fractions: dict[tuple[str, str], float] = {}
    start = 0
    while start < len(ordered):
        start_value = normalized_values[_row_identity(ordered[start])]
        end = start + 1
        while end < len(ordered) and normalized_values[_row_identity(ordered[end])] == start_value:
            end += 1
        tied_fraction = round(float(((start + end - 1) / 2) / denominator), 6)
        for index in range(start, end):
            fractions[_row_identity(ordered[index])] = tied_fraction
        start = end
    return fractions


def _build_group_feature_cache(
    rows: Sequence[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
) -> dict[tuple[str, str], dict[str, float]]:
    ordered_by_retrieval = sorted(
        rows,
        key=lambda row: (
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    retrieval_top1_row = ordered_by_retrieval[0]
    retrieval_runner_up_row = ordered_by_retrieval[1] if len(ordered_by_retrieval) > 1 else ordered_by_retrieval[0]
    best_top3_row = min(
        rows,
        key=lambda row: (
            float(row["top3_mean_distance"]),
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    best_top5_row = min(
        rows,
        key=lambda row: (
            float(row["top5_mean_distance"]),
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    heuristic_window_size = max(int(row["retrieval_rank"]) for row in rows)
    heuristic_choice_key, _heuristic_score = choose_generic_heuristic(rows, window_size=heuristic_window_size)
    if heuristic_choice_key is None:
        heuristic_choice_row = retrieval_top1_row
    else:
        heuristic_choice_row = next(
            row for row in rows if str(row["candidate_component_key"]) == str(heuristic_choice_key)
        )
    heuristic_cross_family = int(_rows_are_cross_family(retrieval_top1_row, best_top5_row))
    heuristic_margin_threshold = float(GENERIC_HEURISTIC_OVERRIDE_MARGIN) + (
        float(GENERIC_CROSS_FAMILY_EXTRA_MARGIN) if heuristic_cross_family == 1 else 0.0
    )
    heuristic_top1_vs_best_top5_margin = float(retrieval_top1_row["top5_mean_distance"]) - float(
        best_top5_row["top5_mean_distance"]
    )
    heuristic_margin_slack = float(heuristic_top1_vs_best_top5_margin - heuristic_margin_threshold)
    best_top5_family_id = str(best_top5_row["family_id"])
    heuristic_choice_family_id = str(heuristic_choice_row["family_id"])
    best_values = {
        "retrieval_score": max(float(row["retrieval_score"]) for row in rows),
        "min_distance": min(float(row["min_distance"]) for row in rows),
        "mean_distance": min(float(row["mean_distance"]) for row in rows),
        "top3_mean_distance": min(float(row["top3_mean_distance"]) for row in rows),
        "top5_mean_distance": min(float(row["top5_mean_distance"]) for row in rows),
    }
    top50_rows = [row for row in rows if int(row["retrieval_rank"]) <= 50]
    if not top50_rows:
        top50_rows = list(rows)
    coarse_family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top50_rows:
        coarse_key = _coarse_family_key(row)
        if coarse_key:
            coarse_family_rows[coarse_key].append(row)
    coarse_family_pair_count_top50 = {
        coarse_key: int(sum(int(candidate_row["pair_count"]) for candidate_row in candidate_rows))
        for coarse_key, candidate_rows in coarse_family_rows.items()
    }
    coarse_family_best_top5_row = {
        coarse_key: min(
            candidate_rows,
            key=lambda row: (
                float(row["top5_mean_distance"]),
                int(row["retrieval_rank"]),
                str(row["candidate_component_key"]),
            ),
        )
        for coarse_key, candidate_rows in coarse_family_rows.items()
    }
    rank_maps = {
        "retrieval_score_rank_fraction": _rank_fraction_map(rows, column="retrieval_score", higher_is_better=True),
        "min_distance_rank_fraction": _rank_fraction_map(rows, column="min_distance", higher_is_better=False),
        "mean_distance_rank_fraction": _rank_fraction_map(rows, column="mean_distance", higher_is_better=False),
        "top3_distance_rank_fraction": _rank_fraction_map(rows, column="top3_mean_distance", higher_is_better=False),
        "top5_distance_rank_fraction": _rank_fraction_map(rows, column="top5_mean_distance", higher_is_better=False),
        "affiliation_overlap_rank_fraction": _rank_fraction_map(
            rows,
            column="affiliation_overlap",
            higher_is_better=True,
        ),
        "coauthor_overlap_rank_fraction": _rank_fraction_map(rows, column="coauthor_overlap", higher_is_better=True),
        "venue_overlap_rank_fraction": _rank_fraction_map(rows, column="venue_overlap", higher_is_better=True),
        "year_compatibility_rank_fraction": _rank_fraction_map(
            rows,
            column="year_compatibility",
            higher_is_better=True,
        ),
        "title_overlap_rank_fraction": _rank_fraction_map(rows, column="title_overlap", higher_is_better=True),
        "specter_centroid_rank_fraction": _rank_fraction_map(
            rows,
            column="specter_centroid_similarity",
            higher_is_better=True,
        ),
        "specter_exemplar_rank_fraction": _rank_fraction_map(
            rows,
            column="specter_exemplar_similarity",
            higher_is_better=True,
        ),
    }
    top1_retrieval_score = float(retrieval_top1_row["retrieval_score"])
    runner_up_retrieval_score = float(retrieval_runner_up_row["retrieval_score"])
    near_tied_alternative_count = int(
        sum(
            1
            for candidate_row in ordered_by_retrieval[1:]
            if float(top1_retrieval_score - float(candidate_row["retrieval_score"]))
            <= float(NEAR_TIED_RETRIEVAL_SCORE_GAP)
        )
    )
    top1_contradiction_count, top1_contradiction_score, top1_exact_title_identity_conflict_flag = (
        _candidate_contradiction_signals(retrieval_top1_row)
    )
    top1_exact_anchor_evidence_flag = _exact_anchor_evidence_flag(retrieval_top1_row)

    cache: dict[tuple[str, str], dict[str, float]] = {}
    candidate_contradiction_payload: dict[tuple[str, str], tuple[int, float, int]] = {}
    exact_anchor_flag_by_row: dict[tuple[str, str], int] = {}
    for row in rows:
        row_id = _row_identity(row)
        candidate_contradiction_payload[row_id] = _candidate_contradiction_signals(row)
        exact_anchor_flag_by_row[row_id] = _exact_anchor_evidence_flag(row)
    plausible_conflicting_candidate_count = int(
        sum(
            1
            for row in rows
            if _row_identity(row) != _row_identity(retrieval_top1_row)
            and (
                float(row["retrieval_score"]) >= float(top1_retrieval_score - PLAUSIBLE_CONFLICT_RETRIEVAL_GAP)
                or float(row["title_overlap"]) >= float(EXACT_TITLE_ANCHOR_THRESHOLD)
                or float(row["count_normalized_confidence"]) >= 0.5
            )
            and candidate_contradiction_payload[_row_identity(row)][1] >= 0.5
        )
    )
    for row in rows:
        row_id = _row_identity(row)
        candidate_count = max(1, int(row["candidate_count"]))
        cross_family_with_top1 = int(_rows_are_cross_family(retrieval_top1_row, row))
        coarse_family_key = _coarse_family_key(row)
        coarse_family_pair_count = int(coarse_family_pair_count_top50.get(coarse_family_key, int(row["pair_count"])))
        candidate_pair_share = float(float(row["pair_count"]) / max(1, coarse_family_pair_count))
        best_same_coarse_family_row = coarse_family_best_top5_row.get(coarse_family_key, row)
        top20_mean_distance = (
            float(row["top20_mean_distance"])
            if row.get("top20_mean_distance") is not None
            else float(row["mean_distance"])
        )
        override_slack_vs_top1 = (
            float(retrieval_top1_row["top5_mean_distance"])
            - float(row["top5_mean_distance"])
            - (
                float(GENERIC_HEURISTIC_OVERRIDE_MARGIN)
                + (float(GENERIC_CROSS_FAMILY_EXTRA_MARGIN) if cross_family_with_top1 == 1 else 0.0)
            )
        )
        candidate_contradiction_count, candidate_contradiction_score, exact_title_identity_conflict_flag = (
            candidate_contradiction_payload[row_id]
        )
        year_mismatch_severity = _year_mismatch_severity(row)
        affiliation_contradiction_severity = _affiliation_contradiction_severity(row)
        anchor_evidence_features = _anchor_evidence_features(
            row,
            candidate_pair_share_within_coarse_family=candidate_pair_share,
            candidate_contradiction_score=candidate_contradiction_score,
        )
        cache[row_id] = {
            "retrieval_rank_fraction": round(float((int(row["retrieval_rank"]) - 1) / max(1, candidate_count - 1)), 6),
            "retrieval_score_rank_fraction": rank_maps["retrieval_score_rank_fraction"][row_id],
            "retrieval_score_best_gap": round(float(best_values["retrieval_score"] - float(row["retrieval_score"])), 6),
            "min_distance_best_gap": round(float(float(row["min_distance"]) - best_values["min_distance"]), 6),
            "mean_distance_best_gap": round(float(float(row["mean_distance"]) - best_values["mean_distance"]), 6),
            "top3_distance_best_gap": round(
                float(float(row["top3_mean_distance"]) - best_values["top3_mean_distance"]),
                6,
            ),
            "top5_distance_best_gap": round(
                float(float(row["top5_mean_distance"]) - best_values["top5_mean_distance"]),
                6,
            ),
            "min_distance_rank_fraction": rank_maps["min_distance_rank_fraction"][row_id],
            "mean_distance_rank_fraction": rank_maps["mean_distance_rank_fraction"][row_id],
            "top3_distance_rank_fraction": rank_maps["top3_distance_rank_fraction"][row_id],
            "top5_distance_rank_fraction": rank_maps["top5_distance_rank_fraction"][row_id],
            "distance_spread_top5_minus_min": round(
                float(float(row["top5_mean_distance"]) - float(row["min_distance"])),
                6,
            ),
            "distance_spread_mean_minus_top5": round(
                float(float(row["mean_distance"]) - float(row["top5_mean_distance"])),
                6,
            ),
            "distance_spread_top20_minus_top5": round(
                float(top20_mean_distance - float(row["top5_mean_distance"])),
                6,
            ),
            "cluster_size_log_capped": round(float(cluster_size_log_capped(row.get("cluster_size"))), 6),
            "is_retrieval_top1": int(_row_identity(row) == _row_identity(retrieval_top1_row)),
            "is_best_top3": int(_row_identity(row) == _row_identity(best_top3_row)),
            "is_best_top5": int(_row_identity(row) == _row_identity(best_top5_row)),
            "is_heuristic_choice": int(_row_identity(row) == _row_identity(heuristic_choice_row)),
            "same_family_as_best_top5": int(
                bool(best_top5_family_id) and bool(row["family_id"]) and str(row["family_id"]) == best_top5_family_id
            ),
            "same_family_as_heuristic_choice": int(
                bool(heuristic_choice_family_id)
                and bool(row["family_id"])
                and str(row["family_id"]) == heuristic_choice_family_id
            ),
            "coarse_family_pair_count_top50": float(coarse_family_pair_count),
            "candidate_pair_share_within_coarse_family": round(
                float(candidate_pair_share),
                6,
            ),
            "coarse_family_top5_best_gap": round(
                float(float(row["top5_mean_distance"]) - float(best_same_coarse_family_row["top5_mean_distance"])),
                6,
            ),
            "coauthor_gap_to_best_same_coarse_family": round(
                float(float(row["coauthor_overlap"]) - float(best_same_coarse_family_row["coauthor_overlap"])),
                6,
            ),
            "top3_gap_to_retrieval_top1": round(
                float(float(row["top3_mean_distance"]) - float(retrieval_top1_row["top3_mean_distance"])),
                6,
            ),
            "top5_gap_to_retrieval_top1": round(
                float(float(row["top5_mean_distance"]) - float(retrieval_top1_row["top5_mean_distance"])),
                6,
            ),
            "top3_gap_to_heuristic_choice": round(
                float(float(row["top3_mean_distance"]) - float(heuristic_choice_row["top3_mean_distance"])),
                6,
            ),
            "top5_gap_to_heuristic_choice": round(
                float(float(row["top5_mean_distance"]) - float(heuristic_choice_row["top5_mean_distance"])),
                6,
            ),
            "heuristic_top1_vs_best_top5_margin": round(float(heuristic_top1_vs_best_top5_margin), 6),
            "heuristic_margin_threshold": round(float(heuristic_margin_threshold), 6),
            "heuristic_margin_slack": round(float(heuristic_margin_slack), 6),
            "heuristic_prefers_top1": int(heuristic_margin_slack <= 0.0),
            "heuristic_cross_family_top1_vs_best_top5": int(heuristic_cross_family),
            "cross_family_with_top1": int(cross_family_with_top1),
            "override_slack_vs_top1": round(float(override_slack_vs_top1), 6),
            "beats_top1_after_penalty": int(override_slack_vs_top1 > 0.0),
            "retrieval_top1_score": round(float(top1_retrieval_score), 6),
            "retrieval_top1_margin": round(float(top1_retrieval_score - runner_up_retrieval_score), 6),
            "near_tied_alternative_count": float(near_tied_alternative_count),
            "exact_anchor_evidence_flag": float(exact_anchor_flag_by_row[row_id]),
            "top1_exact_anchor_evidence_flag": float(top1_exact_anchor_evidence_flag),
            "top1_minus_runnerup_retrieval_score": round(
                float(top1_retrieval_score - runner_up_retrieval_score),
                6,
            ),
            "top1_minus_runnerup_title_overlap": round(
                float(float(retrieval_top1_row["title_overlap"]) - float(retrieval_runner_up_row["title_overlap"])),
                6,
            ),
            "top1_minus_runnerup_coauthor_overlap": round(
                float(
                    float(retrieval_top1_row["coauthor_overlap"]) - float(retrieval_runner_up_row["coauthor_overlap"])
                ),
                6,
            ),
            "top1_minus_runnerup_venue_overlap": round(
                float(float(retrieval_top1_row["venue_overlap"]) - float(retrieval_runner_up_row["venue_overlap"])),
                6,
            ),
            "top1_minus_runnerup_year_compatibility": round(
                float(
                    float(retrieval_top1_row["year_compatibility"])
                    - float(retrieval_runner_up_row["year_compatibility"])
                ),
                6,
            ),
            "top1_minus_runnerup_retrieval_rank": round(
                float(int(retrieval_top1_row["retrieval_rank"]) - int(retrieval_runner_up_row["retrieval_rank"])),
                6,
            ),
            "top1_minus_runnerup_count_normalized_confidence": round(
                float(
                    float(retrieval_top1_row["count_normalized_confidence"])
                    - float(retrieval_runner_up_row["count_normalized_confidence"])
                ),
                6,
            ),
            "top1_minus_runnerup_cluster_size": round(
                float(float(retrieval_top1_row["cluster_size"]) - float(retrieval_runner_up_row["cluster_size"])),
                6,
            ),
            "year_mismatch_severity": round(float(year_mismatch_severity), 6),
            "affiliation_contradiction_severity": round(float(affiliation_contradiction_severity), 6),
            "initial_only_x_title_overlap": round(
                float(float(row["title_overlap"]) if str(row.get("query_view", "")) == "initial_only" else 0.0),
                6,
            ),
            "initial_only_x_coauthor_overlap": round(
                float(float(row["coauthor_overlap"]) if str(row.get("query_view", "")) == "initial_only" else 0.0),
                6,
            ),
            "initial_only_x_venue_overlap": round(
                float(float(row["venue_overlap"]) if str(row.get("query_view", "")) == "initial_only" else 0.0),
                6,
            ),
            "candidate_contradiction_count": float(candidate_contradiction_count),
            "candidate_contradiction_score": round(float(candidate_contradiction_score), 6),
            "exact_title_identity_conflict_flag": float(exact_title_identity_conflict_flag),
            "top1_contradiction_count": float(top1_contradiction_count),
            "top1_strongest_contradiction": round(float(top1_contradiction_score), 6),
            "top1_exact_title_identity_conflict_flag": float(top1_exact_title_identity_conflict_flag),
            "plausible_conflicting_candidate_count": float(plausible_conflicting_candidate_count),
            **anchor_evidence_features,
            "query_view__full": float(str(row.get("query_view", "")) == "full"),
            "query_view__initial_only": float(str(row.get("query_view", "")) == "initial_only"),
            "affiliation_overlap_rank_fraction": rank_maps["affiliation_overlap_rank_fraction"][row_id],
            "coauthor_overlap_rank_fraction": rank_maps["coauthor_overlap_rank_fraction"][row_id],
            "venue_overlap_rank_fraction": rank_maps["venue_overlap_rank_fraction"][row_id],
            "year_compatibility_rank_fraction": rank_maps["year_compatibility_rank_fraction"][row_id],
            "title_overlap_rank_fraction": rank_maps["title_overlap_rank_fraction"][row_id],
            "specter_centroid_rank_fraction": rank_maps["specter_centroid_rank_fraction"][row_id],
            "specter_exemplar_rank_fraction": rank_maps["specter_exemplar_rank_fraction"][row_id],
        }
    return cache


def build_feature_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    feature_preset: str | None = None,
    feature_columns: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert persisted rows into the numeric feature matrix expected by LightGBM."""

    resolved_columns = resolve_feature_columns(
        feature_preset=feature_preset,
        feature_columns=feature_columns,
    )
    if not rows:
        return np.zeros((0, len(resolved_columns)), dtype=np.float32)
    grouped = group_rows(rows)
    derived_by_row: dict[tuple[str, str], dict[str, float]] = {}
    derived_columns_needed = [column for column in resolved_columns if column in DERIVED_FEATURE_COLUMNS]
    needs_derived = any(
        any(column not in row or row[column] is None for row in rows) for column in derived_columns_needed
    )
    if needs_derived:
        for group_rows_for_id in grouped.values():
            derived_by_row.update(
                _build_group_feature_cache(
                    group_rows_for_id,
                    feature_columns=resolved_columns,
                )
            )
    matrix_rows: list[list[float]] = []
    for row in rows:
        row_id = _row_identity(row)
        derived = derived_by_row.get(row_id, {})
        row_values: list[float] = []
        for column in resolved_columns:
            if column in DERIVED_FEATURE_COLUMNS and row.get(column) is not None:
                raw_value = row.get(column)
            elif column in DERIVED_FEATURE_COLUMNS:
                raw_value = derived.get(column, row.get(column))
            else:
                raw_value = row[column]
            row_values.append(0.0 if raw_value is None else float(raw_value))
        matrix_rows.append(row_values)
    return np.asarray(matrix_rows, dtype=np.float32)


def materialize_derived_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Persist all derived numeric feature columns alongside the base row schema."""

    grouped = group_rows(rows)
    derived_by_row: dict[tuple[str, str], dict[str, float]] = {}
    for group_rows_for_id in grouped.values():
        derived_by_row.update(_build_group_feature_cache(group_rows_for_id, feature_columns=tuple()))
    materialized_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = _row_identity(row)
        derived = derived_by_row.get(row_id, {})
        materialized_row = dict(row)
        for column in DERIVED_FEATURE_COLUMNS:
            materialized_row[column] = 0.0 if derived.get(column) is None else float(derived[column])
        materialized_rows.append(materialized_row)
    return materialized_rows


def _positive_row_enrichment_score(
    row: dict[str, Any],
    *,
    derived_row: dict[str, float],
    enrichment_profile: str,
) -> float:
    """Return the rejection-sampling acceptance score for one positive row."""

    if enrichment_profile == "none":
        return 0.0
    if enrichment_profile not in ENRICHMENT_PROFILES:
        raise ValueError(
            f"Unknown enrichment profile: {enrichment_profile}. Expected one of {list(ENRICHMENT_PROFILES)}"
        )
    if int(row["label"]) != 1:
        return 0.0
    slack = float(derived_row["heuristic_margin_slack"])
    cross_family = int(derived_row["heuristic_cross_family_top1_vs_best_top5"])
    retrieval_rank = int(row["retrieval_rank"])
    top5_gap_to_top1 = float(derived_row["top5_gap_to_retrieval_top1"])
    score = 0.0
    if enrichment_profile == "heuristic_override_regions_v2":
        if int(derived_row["is_best_top5"]) == 1 and int(derived_row["is_retrieval_top1"]) == 0 and slack >= 0.04:
            if retrieval_rank >= 10:
                score = max(score, 0.55 if cross_family == 1 else 0.45)
            elif slack >= 0.08:
                score = max(score, 0.4 if cross_family == 1 else 0.3)
            else:
                score = max(score, 0.25 if cross_family == 1 else 0.2)
        if (
            int(derived_row["is_heuristic_choice"]) == 1
            and int(derived_row["is_retrieval_top1"]) == 0
            and retrieval_rank >= 25
            and slack >= 0.04
        ):
            score = max(score, 0.5 if cross_family == 1 else 0.4)
        return float(min(1.0, score))
    if enrichment_profile == "s2and_hard_regions_v1":
        # The held-out S2AND ranker regressions are concentrated in non-top1
        # positives, especially clear override regions and deeper retrieval ranks.
        if retrieval_rank >= 26:
            score = max(score, 0.75)
        elif retrieval_rank >= 11:
            score = max(score, 0.5)
        elif retrieval_rank >= 4:
            score = max(score, 0.3)
        elif retrieval_rank >= 2:
            score = max(score, 0.18)
        if int(derived_row["heuristic_prefers_top1"]) == 0:
            if top5_gap_to_top1 <= -0.01:
                score = max(score, 0.8 if cross_family == 1 else 0.7)
            elif slack >= 0.04:
                score = max(score, 0.6 if cross_family == 1 else 0.5)
            else:
                score = max(score, 0.4 if cross_family == 1 else 0.3)
        if retrieval_rank >= 2 and top5_gap_to_top1 <= -0.01:
            score = max(score, 0.75 if cross_family == 1 else 0.65)
        if not _has_confident_family_assignment(row) and retrieval_rank >= 2:
            score = max(score, 0.25)
        return float(min(1.0, score))
    # Region 1: clear heuristic overrides are underrepresented in the labeled data
    # relative to the remaining h_wang regressions, so enrich those positives.
    if int(derived_row["is_best_top5"]) == 1 and int(derived_row["is_retrieval_top1"]) == 0:
        if slack >= 0.08:
            score = max(score, 0.45 if cross_family == 1 else 0.35)
        elif slack >= 0.04:
            score = max(score, 0.35 if cross_family == 1 else 0.25)
    if (
        int(derived_row["is_heuristic_choice"]) == 1
        and int(derived_row["is_retrieval_top1"]) == 0
        and retrieval_rank >= 10
        and slack >= 0.04
    ):
        score = max(score, 0.4 if cross_family == 1 else 0.3)
    # Region 2: extremely tight keep-top1 cases are rare in the labeled data.
    if int(derived_row["is_retrieval_top1"]) == 1 and -0.005 <= slack <= 0.0:
        score = max(score, 0.15 if cross_family == 1 else 0.1)
    return float(min(1.0, score))


def build_training_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    limit_groups: int | None = None,
    seed: int = 13,
    feature_preset: str | None = None,
    feature_columns: Sequence[str] | None = None,
    enrichment_profile: str = "none",
    enrichment_rounds: int = 0,
) -> TrainingMatrix:
    """Prepare grouped training data and drop all-negative groups."""

    grouped = group_rows(rows)
    group_ids = sorted(grouped)
    rng = random.Random(seed)
    if limit_groups is not None and int(limit_groups) > 0 and len(group_ids) > int(limit_groups):
        group_ids = sorted(rng.sample(group_ids, int(limit_groups)))
    ordered_rows: list[dict[str, Any]] = []
    kept_group_sizes: dict[str, int] = {}
    dropped_all_negative_group_ids: list[str] = []
    group_repeat_counts: dict[str, int] = {}
    extra_group_copies = 0
    groups_with_extra_copies = 0
    for group_id in group_ids:
        group_rows_for_id = grouped[group_id]
        if not any(int(row["label"]) == 1 for row in group_rows_for_id):
            dropped_all_negative_group_ids.append(str(group_id))
            continue
        kept_group_sizes[str(group_id)] = int(len(group_rows_for_id))
        repeat_count = 1
        if str(enrichment_profile) != "none" and int(enrichment_rounds) > 0:
            derived_cache = _build_group_feature_cache(group_rows_for_id, feature_columns=tuple())
            enrichment_score = max(
                _positive_row_enrichment_score(
                    row,
                    derived_row=derived_cache[_row_identity(row)],
                    enrichment_profile=str(enrichment_profile),
                )
                for row in group_rows_for_id
                if int(row["label"]) == 1
            )
            accepted_copies = sum(1 for _ in range(int(enrichment_rounds)) if rng.random() < float(enrichment_score))
            repeat_count += int(accepted_copies)
            extra_group_copies += int(accepted_copies)
            if int(accepted_copies) > 0:
                groups_with_extra_copies += 1
        group_repeat_counts[str(group_id)] = int(repeat_count)
        for _ in range(int(repeat_count)):
            ordered_rows.extend(group_rows_for_id)
    features = build_feature_matrix(
        ordered_rows,
        feature_preset=feature_preset,
        feature_columns=feature_columns,
    )
    labels = np.asarray([int(row["label"]) for row in ordered_rows], dtype=np.int32)
    sample_weights = np.asarray(
        [1.0 / float(kept_group_sizes[str(row["query_group_id"])]) for row in ordered_rows],
        dtype=np.float32,
    )
    return TrainingMatrix(
        ordered_rows=list(ordered_rows),
        features=features,
        labels=labels,
        sample_weights=sample_weights,
        group_ids=sorted(kept_group_sizes),
        kept_group_sizes=dict(kept_group_sizes),
        dropped_all_negative_group_ids=sorted(dropped_all_negative_group_ids),
        enrichment_profile=str(enrichment_profile),
        enrichment_rounds=int(enrichment_rounds),
        extra_group_copies=int(extra_group_copies),
        groups_with_extra_copies=int(groups_with_extra_copies),
        group_repeat_counts=dict(group_repeat_counts),
    )


def choose_retrieval_top1(rows: Sequence[dict[str, Any]], *, window_size: int) -> str | None:
    """Return the retrieval-top1 candidate inside ``window_size``."""

    eligible = [row for row in rows if int(row["retrieval_rank"]) <= int(window_size)]
    if not eligible:
        return None
    return str(
        min(
            eligible,
            key=lambda row: (int(row["retrieval_rank"]), str(row["candidate_component_key"])),
        )["candidate_component_key"]
    )


def _rows_are_cross_family(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        _has_confident_family_assignment(left)
        and _has_confident_family_assignment(right)
        and str(left["family_id"]) != str(right["family_id"])
    )


def choose_generic_heuristic(rows: Sequence[dict[str, Any]], *, window_size: int) -> tuple[str | None, float | None]:
    """Apply the generic sticky top-k pairwise heuristic within ``window_size``."""

    eligible = [row for row in rows if int(row["retrieval_rank"]) <= int(window_size)]
    if not eligible:
        return None, None
    retrieval_top1_row = min(
        eligible,
        key=lambda row: (int(row["retrieval_rank"]), str(row["candidate_component_key"])),
    )
    best_row = min(
        eligible,
        key=lambda row: (
            float(row["top5_mean_distance"]),
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    if str(best_row["candidate_component_key"]) != str(retrieval_top1_row["candidate_component_key"]):
        effective_margin = float(GENERIC_HEURISTIC_OVERRIDE_MARGIN)
        if _rows_are_cross_family(retrieval_top1_row, best_row):
            effective_margin += float(GENERIC_CROSS_FAMILY_EXTRA_MARGIN)
        if float(best_row["top5_mean_distance"]) + float(effective_margin) >= float(
            retrieval_top1_row["top5_mean_distance"]
        ):
            best_row = retrieval_top1_row
    return str(best_row["candidate_component_key"]), float(best_row["top5_mean_distance"])


def summarize_dataset_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build the generator spot-check summary requested in the TODO."""

    grouped = group_rows(rows)
    all_negative_groups = [
        group_id for group_id, values in grouped.items() if not any(int(row["label"]) for row in values)
    ]
    summary = {
        "query_groups": int(len(grouped)),
        "row_count": int(len(rows)),
        "positive_rows": int(sum(int(row["label"]) for row in rows)),
        "positive_rate": round(float(sum(int(row["label"]) for row in rows) / max(1, len(rows))), 6),
        "dropped_all_negative_group_count": int(len(all_negative_groups)),
        "candidate_window_coverage": round(float((len(grouped) - len(all_negative_groups)) / max(1, len(grouped))), 6),
        "candidate_count_mean": round(
            float(statistics.mean(len(values) for values in grouped.values())),
            6,
        )
        if grouped
        else 0.0,
        "candidate_count_median": round(
            float(statistics.median(len(values) for values in grouped.values())),
            6,
        )
        if grouped
        else 0.0,
        "best_positive_retrieval_rank_mean": round(
            float(
                statistics.mean(
                    int(values[0]["best_positive_retrieval_rank"])
                    for values in grouped.values()
                    if values[0]["best_positive_retrieval_rank"] is not None
                )
            ),
            6,
        )
        if any(values[0]["best_positive_retrieval_rank"] is not None for values in grouped.values())
        else None,
        "per_supervision_type": {},
        "per_split": {},
        "per_view": {},
    }
    for query_view in sorted({str(row["query_view"]) for row in rows}):
        view_rows = [row for row in rows if str(row["query_view"]) == query_view]
        view_grouped = group_rows(view_rows)
        view_all_negative = [
            group_id for group_id, values in view_grouped.items() if not any(int(row["label"]) for row in values)
        ]
        summary["per_view"][query_view] = {
            "query_groups": int(len(view_grouped)),
            "row_count": int(len(view_rows)),
            "positive_rows": int(sum(int(row["label"]) for row in view_rows)),
            "positive_rate": round(float(sum(int(row["label"]) for row in view_rows) / max(1, len(view_rows))), 6),
            "dropped_all_negative_group_count": int(len(view_all_negative)),
            "candidate_window_coverage": round(
                float((len(view_grouped) - len(view_all_negative)) / max(1, len(view_grouped))),
                6,
            ),
        }
    for supervision_type in sorted({str(row.get("supervision_type", "labeled")) for row in rows}):
        type_rows = [row for row in rows if str(row.get("supervision_type", "labeled")) == supervision_type]
        type_grouped = group_rows(type_rows)
        summary["per_supervision_type"][supervision_type] = {
            "query_groups": int(len(type_grouped)),
            "row_count": int(len(type_rows)),
            "positive_rows": int(sum(int(row["label"]) for row in type_rows)),
        }
    for split in sorted({str(row.get("split", "all")) for row in rows}):
        split_rows = [row for row in rows if str(row.get("split", "all")) == split]
        split_grouped = group_rows(split_rows)
        summary["per_split"][split] = {
            "query_groups": int(len(split_grouped)),
            "row_count": int(len(split_rows)),
            "positive_rows": int(sum(int(row["label"]) for row in split_rows)),
        }
    return summary


def probability_calibration_bins(correctness: Sequence[int], probabilities: Sequence[float]) -> list[dict[str, Any]]:
    """Build compact probability-calibration bins for chosen-candidate predictions."""

    bins: list[list[tuple[float, int]]] = [[] for _ in range(CALIBRATION_BIN_COUNT)]
    for probability, correct in zip(probabilities, correctness, strict=True):
        clipped = min(max(float(probability), 0.0), 1.0)
        index = min(int(clipped * CALIBRATION_BIN_COUNT), CALIBRATION_BIN_COUNT - 1)
        bins[index].append((clipped, int(correct)))
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(bins):
        if not values:
            rows.append(
                {
                    "bin_index": int(index),
                    "lower": round(index / CALIBRATION_BIN_COUNT, 6),
                    "upper": round((index + 1) / CALIBRATION_BIN_COUNT, 6),
                    "count": 0,
                    "mean_probability": None,
                    "accuracy": None,
                }
            )
            continue
        rows.append(
            {
                "bin_index": int(index),
                "lower": round(index / CALIBRATION_BIN_COUNT, 6),
                "upper": round((index + 1) / CALIBRATION_BIN_COUNT, 6),
                "count": int(len(values)),
                "mean_probability": round(float(statistics.mean(probability for probability, _ in values)), 6),
                "accuracy": round(float(statistics.mean(correct for _, correct in values)), 6),
            }
        )
    return rows


def expected_calibration_error(calibration_rows: Sequence[dict[str, Any]]) -> float:
    """Compute the expected calibration error for the chosen-candidate bins."""

    total_count = sum(int(row["count"]) for row in calibration_rows)
    if total_count <= 0:
        return 0.0
    error = 0.0
    for row in calibration_rows:
        if not row["count"]:
            continue
        error += abs(float(row["mean_probability"]) - float(row["accuracy"])) * float(row["count"]) / float(total_count)
    return float(error)
