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
from pathlib import Path
from typing import Any

import numpy as np

import s2and.data as s2and_data_module
import s2and.subblocking as s2and_subblocking_module
from s2and.consts import DEFAULT_CHUNK_SIZE
from s2and.data import ANDData
from s2and.featurizer import many_pairs_featurize
from s2and.model import _predict_and_combine
from s2and.text import normalize_text

try:
    from scripts.name_count_loading import LoadNameCountsMode, resolve_load_name_counts
except ImportError:  # pragma: no cover - direct script execution path
    from name_count_loading import LoadNameCountsMode, resolve_load_name_counts  # type: ignore

try:
    import scripts.eval_cluster_retrieval as retrieval
    from scripts.single_letter_retrieval_utils import rank_top_summaries
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore
    from single_letter_retrieval_utils import rank_top_summaries  # type: ignore

DEFAULT_LABELED_DATASETS = (
    "arnetminer",
    "inspire",
    "kisti",
    "pubmed",
    "qian",
    "zbmath",
)
DEFAULT_QUERY_VIEWS = (
    "initial_only",
    "initial_only_no_specter",
    "initial_only_sparse_metadata",
)
DEFAULT_RETRIEVAL_APPROACH = "all__hybrid_centroid"
SUPPORTED_RETRIEVAL_METHODS = frozenset({"hybrid_centroid", "hybrid_exemplar_4"})
RETRIEVAL_AMBIGUITY_SCORE_GAP = 0.02
RETRIEVAL_AMBIGUITY_SAME_FAMILY_GAP = 0.05
DEFAULT_RETRIEVAL_WINDOW_SIZE = 100
DEFAULT_CANDIDATE_WINDOW_SENSITIVITY = (10, 25, 50, 100)
DEFAULT_H_WANG_WINDOW_SENSITIVITY = (25, 50, 100)
GENERIC_HEURISTIC_OVERRIDE_MARGIN = 0.01
GENERIC_CROSS_FAMILY_EXTRA_MARGIN = 0.03
GENERIC_FAMILY_MIN_COUNT = 3
GENERIC_FAMILY_MIN_RATIO = 0.6
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
    "dominant_name_ratio",
    "named_signature_count",
    "confident_family_flag",
    "same_family_as_top1",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "family_instability_flag",
    "fragment_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
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
    "dominant_name_ratio",
    "confident_family_flag",
    "same_family_as_best_top5",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "affiliation_overlap_rank_fraction",
    "coauthor_overlap_rank_fraction",
    "venue_overlap_rank_fraction",
    "year_compatibility_rank_fraction",
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
        "confident_family_flag",
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
}
DEFAULT_FEATURE_PRESET = "generalized_v3"
FEATURE_COLUMNS = FEATURE_PRESETS[DEFAULT_FEATURE_PRESET]
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
    "normalized_orcid",
    "orcid_group_size",
    "orcid_group_size_bucket",
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
    "dominant_name_ratio",
    "named_signature_count",
    "confident_family_flag",
    "same_family_as_top1",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "family_instability_flag",
    "fragment_flag",
    "query_has_specter",
    "query_has_coauthors",
    "query_has_affiliations",
    "query_has_middle",
    "query_has_full_first",
)
INT_COLUMNS = {
    "orcid_group_size",
    "block_size",
    "component_size",
    "positive_candidate_count",
    "group_has_positive",
    "best_positive_retrieval_rank",
    "query_in_seed_before_holdout",
    "label",
    "candidate_count",
    "candidate_signatures",
    "scored_candidate_components",
    "scored_candidate_signatures",
    "orcid_filter_applied",
    "middle_initial_filter_applied",
    "year_range_filter_applied",
    "retrieval_rank",
    "cluster_size",
    "pair_count",
    "named_signature_count",
    "confident_family_flag",
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
    "dominant_name_ratio",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
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
    "normalized_orcid",
    "orcid_group_size",
    "orcid_group_size_bucket",
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
    "orcid_group_size",
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
}
NUMERIC_FEATURE_COLUMNS = set(BASELINE_FEATURE_COLUMNS) | DERIVED_FEATURE_COLUMNS
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
    dominant_name_ratio: float
    named_signature_count: int


@dataclass(frozen=True)
class RetrievalApproachSpec:
    """Structured retrieval-window configuration parsed from the CLI string."""

    mode: str
    methods: tuple[str, ...]


@dataclass
class ClusterPairwiseStats:
    """Pairwise query-to-cluster aggregate statistics."""

    cluster_id: str
    retrieval_rank: int
    retrieval_score: float
    cluster_size: int
    family_id: str = ""
    dominant_first_name: str | None = None
    dominant_name_ratio: float = 0.0
    named_signature_count: int = 0
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

    named_signature_count = int(sum(summary.first_name_counts.values()))
    dominant_first_name = None
    dominant_name_ratio = 0.0
    family_id = str(summary.component_key)
    if summary.first_name_counts and named_signature_count > 0:
        dominant_first_name, dominant_count = max(
            summary.first_name_counts.items(),
            key=lambda item: (int(item[1]), str(item[0])),
        )
        dominant_name_ratio = float(dominant_count / named_signature_count)
        if int(named_signature_count) >= int(GENERIC_FAMILY_MIN_COUNT) and float(dominant_name_ratio) >= float(
            GENERIC_FAMILY_MIN_RATIO
        ):
            family_id = str(dominant_first_name)
    return ClusterProfile(
        cluster_id=str(summary.component_key),
        family_id=str(family_id),
        dominant_first_name=str(dominant_first_name) if dominant_first_name is not None else None,
        dominant_name_ratio=float(dominant_name_ratio),
        named_signature_count=int(named_signature_count),
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
) -> list[tuple[float, retrieval.ClusterSummary]]:
    """Rank one retrieval method over the filtered candidate summaries."""

    return rank_top_summaries(
        method=method,
        query=query,
        candidate_summaries=candidate_summaries,
        max_block_component_size=max_block_component_size,
        max_ranked_clusters=max_ranked_clusters,
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
) -> tuple[list[str], dict[str, float], dict[str, int], dict[str, int]]:
    """Rank candidate summaries under the fixed retrieval operating point."""

    approach_spec = _parse_retrieval_approach(retrieval_approach)
    candidate_summaries, filter_state = retrieval.apply_hard_filters(query, raw_candidate_summaries)
    profiles_by_component = {
        str(summary.component_key): build_cluster_profile(summary) for summary in candidate_summaries
    }
    ranked_by_method = {
        method: _rank_method_window(
            method=method,
            query=query,
            candidate_summaries=candidate_summaries,
            max_block_component_size=max_block_component_size,
            max_ranked_clusters=max_ranked_clusters,
        )
        for method in approach_spec.methods
    }
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
            "scored_candidate_components": int(filter_state["scored_candidate_components"]),
            "scored_candidate_signatures": int(filter_state["scored_candidate_signatures"]),
            "orcid_filter_applied": int(filter_state["orcid_filter_applied"]),
            "middle_initial_filter_applied": int(filter_state["middle_initial_filter_applied"]),
            "year_range_filter_applied": int(filter_state["year_range_filter_applied"]),
            "ambiguity_expanded": int(ambiguity_expanded),
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
            dominant_name_ratio=float(profile.dominant_name_ratio),
            named_signature_count=int(profile.named_signature_count),
        )
    return stats_by_cluster_id


def _update_query_cluster_stats_from_multi_query_batch(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    batch_pairs: list[tuple[str, str]],
    batch_request_indices: list[int],
    batch_cluster_ids: list[str],
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
    max_top_k: int = 5,
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

    def flush_batch() -> None:
        nonlocal batch_pairs
        nonlocal batch_request_indices
        nonlocal batch_cluster_ids
        _update_query_cluster_stats_from_multi_query_batch(
            clusterer=clusterer,
            dataset=dataset,
            runtime_context=runtime_context,
            constraint_backend=constraint_backend,
            batch_pairs=batch_pairs,
            batch_request_indices=batch_request_indices,
            batch_cluster_ids=batch_cluster_ids,
            stats_by_request=stats_by_request,
            diagnostics_by_request=diagnostics_by_request,
            max_top_k=max_top_k,
        )
        batch_pairs = []
        batch_request_indices = []
        batch_cluster_ids = []

    for request_index, request in enumerate(requests):
        for component_key in request.shortlist_component_keys:
            signature_ids = request.candidate_signature_ids_by_component[component_key]
            for signature_id in signature_ids:
                batch_pairs.append((str(request.query_signature_id), str(signature_id)))
                batch_request_indices.append(int(request_index))
                batch_cluster_ids.append(str(component_key))
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


def compute_query_cluster_stats(
    *,
    clusterer: Any,
    dataset: Any,
    runtime_context: Any,
    constraint_backend: Any,
    query_signature_id: str,
    shortlist_component_keys: list[str],
    candidate_signature_ids_by_component: dict[str, list[str]],
    retrieval_ranks: dict[str, int],
    retrieval_scores: dict[str, float],
    summary_by_component: dict[str, retrieval.ClusterSummary],
    pair_batch_size: int,
    max_top_k: int = 5,
) -> tuple[dict[str, ClusterPairwiseStats], dict[str, Any]]:
    """Aggregate pairwise query-to-cluster stats for the retrieved window."""
    request = QueryClusterStatsRequest(
        query_signature_id=str(query_signature_id),
        shortlist_component_keys=tuple(str(component_key) for component_key in shortlist_component_keys),
        candidate_signature_ids_by_component={
            str(component_key): [
                str(signature_id) for signature_id in candidate_signature_ids_by_component[component_key]
            ]
            for component_key in shortlist_component_keys
        },
        retrieval_ranks={
            str(component_key): int(retrieval_ranks[component_key]) for component_key in shortlist_component_keys
        },
        retrieval_scores={
            str(component_key): float(retrieval_scores[component_key]) for component_key in shortlist_component_keys
        },
        summary_by_component={
            str(component_key): summary_by_component[component_key] for component_key in shortlist_component_keys
        },
    )
    batch_results = compute_query_cluster_stats_batched(
        clusterer=clusterer,
        dataset=dataset,
        runtime_context=runtime_context,
        constraint_backend=constraint_backend,
        requests=[request],
        pair_batch_size=int(pair_batch_size),
        max_top_k=int(max_top_k),
    )
    if len(batch_results) != 1:
        raise RuntimeError(f"Expected one batched query result, got {len(batch_results)}")
    return batch_results[0]


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
    return -0.25


def _year_compatibility(query_year: int | None, summary: retrieval.ClusterSummary) -> float:
    if query_year is None or summary.year_mean is None:
        return 0.0
    distance = abs(float(query_year) - float(summary.year_mean))
    score = max(0.0, 1.0 - (distance / 15.0))
    if summary.year_min is not None and summary.year_max is not None:
        if query_year < int(summary.year_min) - 10 or query_year > int(summary.year_max) + 10:
            score -= 0.15
    return float(score)


def count_normalized_confidence(stats: ClusterPairwiseStats, *, max_pair_count_in_group: int) -> float:
    """Return a simple support-aware confidence signal from the pairwise stats."""

    if stats.count <= 0 or max_pair_count_in_group <= 0:
        return 0.0
    top3_distance = stats.topk_mean_distance(3)
    if not math.isfinite(top3_distance):
        return 0.0
    support = math.log1p(float(stats.count)) / math.log1p(float(max_pair_count_in_group))
    quality = max(0.0, 1.0 - float(top3_distance))
    return float(support * quality)


def _best_competitor_component_key(
    sorted_component_keys: list[str],
    *,
    current_component_key: str,
) -> str | None:
    for component_key in sorted_component_keys:
        if component_key != current_component_key:
            return str(component_key)
    return None


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
) -> list[dict[str, Any]]:
    """Convert one retrieved candidate window into persisted reranker rows."""

    if not shortlist_component_keys:
        return []
    sorted_component_keys = sorted(
        shortlist_component_keys,
        key=lambda component_key: (int(retrieval_ranks[component_key]), str(component_key)),
    )
    top1_component_key = sorted_component_keys[0]
    top1_stats = stats_by_component[top1_component_key]
    positive_component_keys = frozenset(
        component_key for component_key in sorted_component_keys if component_key in query_case.positive_component_keys
    )
    best_positive_retrieval_rank = (
        min(int(retrieval_ranks[component_key]) for component_key in positive_component_keys)
        if positive_component_keys
        else None
    )
    max_pair_count_in_group = max((int(stats.count) for stats in stats_by_component.values()), default=0)
    rows: list[dict[str, Any]] = []
    for component_key in sorted_component_keys:
        summary = summary_by_component[component_key]
        stats = stats_by_component[component_key]
        best_competitor_component_key = _best_competitor_component_key(
            sorted_component_keys,
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
            int(stats.named_signature_count) >= int(GENERIC_FAMILY_MIN_COUNT)
            and float(stats.dominant_name_ratio) < float(GENERIC_FAMILY_MIN_RATIO)
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
            "normalized_orcid": (str(query_case.normalized_orcid) if query_case.normalized_orcid is not None else None),
            "orcid_group_size": (int(query_case.orcid_group_size) if query_case.orcid_group_size is not None else None),
            "orcid_group_size_bucket": (
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
            "label": int(component_key in query_case.positive_component_keys),
            "candidate_count": int(len(shortlist_component_keys)),
            "candidate_signatures": int(sum(summary_by_component[key].size for key in shortlist_component_keys)),
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
            "dominant_name_ratio": round(float(stats.dominant_name_ratio), 6),
            "named_signature_count": int(stats.named_signature_count),
            "confident_family_flag": int(str(stats.family_id) != str(component_key)),
            "same_family_as_top1": int(same_family_as_top1),
            "middle_initial_compatibility": round(float(_middle_initial_compatibility(query_features, summary)), 6),
            "affiliation_overlap": round(
                float(
                    _counter_query_overlap(
                        query_features.affiliation_terms,
                        summary.affiliation_counts,
                        summary.size,
                    )
                ),
                6,
            ),
            "coauthor_overlap": round(
                float(_counter_query_overlap(query_features.coauthor_blocks, summary.coauthor_counts, summary.size)),
                6,
            ),
            "venue_overlap": round(
                float(_counter_query_overlap(query_features.venue_terms, summary.venue_counts, summary.size)),
                6,
            ),
            "year_compatibility": round(float(_year_compatibility(query_features.year, summary)), 6),
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


def read_rows_csv(path: Path) -> list[dict[str, Any]]:
    """Read persisted reranker rows from ``path``."""

    return _read_typed_rows_csv(path, int_columns=INT_COLUMNS, float_columns=FLOAT_COLUMNS)


def write_query_group_metadata_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write one row per query group for cached sampler selection."""

    write_dict_rows_csv(path, rows, fieldnames=QUERY_GROUP_METADATA_COLUMNS)


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
    if positive_rows and not retrieval_top1_is_positive and int(retrieval_top1_row["confident_family_flag"]) == 1:
        positive_family_ids = {
            str(row["family_id"])
            for row in positive_rows
            if int(row["confident_family_flag"]) == 1 and row["family_id"]
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
        "normalized_orcid": (
            str(first_row["normalized_orcid"]) if first_row.get("normalized_orcid") is not None else None
        ),
        "orcid_group_size": (
            int(first_row["orcid_group_size"]) if first_row.get("orcid_group_size") is not None else None
        ),
        "orcid_group_size_bucket": (
            str(first_row["orcid_group_size_bucket"]) if first_row.get("orcid_group_size_bucket") is not None else None
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


def _rank_fraction_map(
    rows: Sequence[dict[str, Any]],
    *,
    column: str,
    higher_is_better: bool,
) -> dict[tuple[str, str], float]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[column]) if higher_is_better else float(row[column]),
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
    denominator = max(1, len(ordered) - 1)
    return {_row_identity(row): round(float(index / denominator), 6) for index, row in enumerate(ordered)}


def _build_group_feature_cache(
    rows: Sequence[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
) -> dict[tuple[str, str], dict[str, float]]:
    retrieval_top1_row = min(
        rows,
        key=lambda row: (
            int(row["retrieval_rank"]),
            str(row["candidate_component_key"]),
        ),
    )
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
    }
    cache: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        row_id = _row_identity(row)
        candidate_count = max(1, int(row["candidate_count"]))
        cross_family_with_top1 = int(_rows_are_cross_family(retrieval_top1_row, row))
        coarse_family_key = _coarse_family_key(row)
        coarse_family_pair_count = int(coarse_family_pair_count_top50.get(coarse_family_key, int(row["pair_count"])))
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
                float(float(row["pair_count"]) / max(1, coarse_family_pair_count)),
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
            "affiliation_overlap_rank_fraction": rank_maps["affiliation_overlap_rank_fraction"][row_id],
            "coauthor_overlap_rank_fraction": rank_maps["coauthor_overlap_rank_fraction"][row_id],
            "venue_overlap_rank_fraction": rank_maps["venue_overlap_rank_fraction"][row_id],
            "year_compatibility_rank_fraction": rank_maps["year_compatibility_rank_fraction"][row_id],
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
        if int(row["confident_family_flag"]) == 0 and retrieval_rank >= 2:
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
        int(left["confident_family_flag"]) == 1
        and int(right["confident_family_flag"]) == 1
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
