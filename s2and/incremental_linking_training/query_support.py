"""Training/replay query, retrieval, and row-signal helpers for the promoted linker."""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

import s2and.data as s2and_data_module
import s2and.subblocking as s2and_subblocking_module
from s2and.data import ANDData
from s2and.incremental_linking.query_adapter import (
    ClusterSummary,
    QueryFeatures,
    RustHybridCentroidRetrieverHandle,
    build_cluster_summary,
    build_rust_hybrid_centroid_retriever,
)
from s2and.incremental_linking.query_adapter import (
    _name_count_rarity_features as name_count_rarity_features,
)
from s2and.incremental_linking.row_features import GENERIC_FAMILY_MIN_COUNT, GENERIC_FAMILY_MIN_RATIO
from s2and.incremental_linking_training import retrieval_policy
from s2and.incremental_linking_training.name_counts import LoadNameCountsMode, resolve_load_name_counts
from s2and.subblocking import make_subblocks_with_telemetry
from s2and.text import normalize_text

DEFAULT_CHOOSER_CACHE_MAX_TOP_K = 25
FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME = "h_wang_any_input_v2"


@dataclass(frozen=True)
class RustHybridCentroidScoringConfig:
    """Optional experimental scoring controls for the Rust retriever."""

    first_name_mode: str = "prefix"
    specter_mode: str = "centroid"
    coauthor_use_idf: bool = False
    coauthor_per_term_cap: float | None = None
    coauthor_total_cap: float | None = None
    drop_candidate_mega_coauthors: bool = False
    mega_coauthor_rescue_query_coverage: float | None = None
    mega_coauthor_rescue_min_shared_blocks: int = 3
    affiliation_use_idf: bool = False
    affiliation_per_term_cap: float | None = None
    affiliation_total_cap: float | None = None
    affiliation_min_token_count: int = 1
    affiliation_unigram_weight: float = 1.0
    affiliation_multi_token_weight: float = 1.0

    def to_kwargs(self) -> dict[str, Any]:
        """Return keyword args expected by the Rust experimental scorer."""

        return {
            "first_name_mode": str(self.first_name_mode),
            "specter_mode": str(self.specter_mode),
            "coauthor_use_idf": bool(self.coauthor_use_idf),
            "coauthor_per_term_cap": self.coauthor_per_term_cap,
            "coauthor_total_cap": self.coauthor_total_cap,
            "drop_candidate_mega_coauthors": bool(self.drop_candidate_mega_coauthors),
            "mega_coauthor_rescue_query_coverage": self.mega_coauthor_rescue_query_coverage,
            "mega_coauthor_rescue_min_shared_blocks": int(self.mega_coauthor_rescue_min_shared_blocks),
            "affiliation_use_idf": bool(self.affiliation_use_idf),
            "affiliation_per_term_cap": self.affiliation_per_term_cap,
            "affiliation_total_cap": self.affiliation_total_cap,
            "affiliation_min_token_count": int(self.affiliation_min_token_count),
            "affiliation_unigram_weight": float(self.affiliation_unigram_weight),
            "affiliation_multi_token_weight": float(self.affiliation_multi_token_weight),
        }


@dataclass(frozen=True)
class FrozenRustHybridCentroidPolicy:
    """Frozen tuned Rust retriever policy used for promoted row generation."""

    full_weights: tuple[float, ...]
    initial_only_weights: tuple[float, ...]
    full_scoring_config: RustHybridCentroidScoringConfig | None = None
    initial_only_scoring_config: RustHybridCentroidScoringConfig | None = None
    full_candidate_strategy: str = "global"

    def weights_for_query(self, query: QueryFeatures) -> tuple[float, ...]:
        """Return the weights for the current masked query view."""

        return self.full_weights if bool(query.has_full_first) else self.initial_only_weights

    def scoring_config_for_query(self, query: QueryFeatures) -> RustHybridCentroidScoringConfig | None:
        """Return the scoring config for the current masked query view."""

        return self.full_scoring_config if bool(query.has_full_first) else self.initial_only_scoring_config

    def uses_exemplar_scoring(self) -> bool:
        """Return whether either query-view scorer needs exemplar vectors."""

        configs = (self.full_scoring_config, self.initial_only_scoring_config)
        return any(config is not None and str(config.specter_mode) != "centroid" for config in configs)

    def to_summary_payload(self, *, policy_name: str) -> dict[str, Any]:
        """Serialize the fixed policy into a reproducible summary payload."""

        return {
            "policy_name": str(policy_name),
            "feature_order": list(retrieval_policy.HYBRID_FEATURE_ORDER),
            "full_candidate_strategy": str(self.full_candidate_strategy),
            "full_weights": {
                name: round(float(value), 6)
                for name, value in zip(retrieval_policy.HYBRID_FEATURE_ORDER, self.full_weights, strict=True)
            },
            "initial_only_weights": {
                name: round(float(value), 6)
                for name, value in zip(retrieval_policy.HYBRID_FEATURE_ORDER, self.initial_only_weights, strict=True)
            },
            "full_scoring_config": (
                self.full_scoring_config.to_kwargs() if self.full_scoring_config is not None else None
            ),
            "initial_only_scoring_config": (
                self.initial_only_scoring_config.to_kwargs() if self.initial_only_scoring_config is not None else None
            ),
        }


FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY = FrozenRustHybridCentroidPolicy(
    full_weights=(0.527232, 0.223412, 0.146909, 0.009439, 0.093007),
    initial_only_weights=(0.520012, 0.220264, 0.109278, 0.150447, 0.0),
    full_scoring_config=RustHybridCentroidScoringConfig(
        first_name_mode="exact_then_prefix_half",
        specter_mode="max_centroid_exemplar",
        coauthor_use_idf=True,
        coauthor_per_term_cap=0.35,
        drop_candidate_mega_coauthors=True,
        mega_coauthor_rescue_query_coverage=0.995,
        mega_coauthor_rescue_min_shared_blocks=3,
        affiliation_use_idf=True,
    ),
    initial_only_scoring_config=RustHybridCentroidScoringConfig(
        first_name_mode="prefix",
        specter_mode="max_centroid_exemplar",
        coauthor_use_idf=True,
        coauthor_per_term_cap=0.35,
        drop_candidate_mega_coauthors=True,
        mega_coauthor_rescue_query_coverage=0.995,
        mega_coauthor_rescue_min_shared_blocks=3,
        affiliation_use_idf=True,
    ),
    full_candidate_strategy="name_compat_plus_global_backfill5",
)


@dataclass(frozen=True)
class ClusterProfile:
    """Family metadata derived from a candidate cluster summary."""

    cluster_id: str
    family_id: str
    dominant_first_name: str | None
    family_dominance_ratio: float
    family_named_count: int


_ORIGINAL_COMPUTE_BLOCK = s2and_data_module.compute_block


def _safe_compute_block(name: str) -> str:
    normalized_name = normalize_text(name or "")
    if not normalized_name:
        return ""
    return _ORIGINAL_COMPUTE_BLOCK(normalized_name)


def install_safe_compute_block_patch() -> None:
    """Treat blank coauthor names as missing during replay dataset ingestion."""

    cast(Any, s2and_data_module).compute_block = _safe_compute_block
    cast(Any, s2and_subblocking_module).compute_block = _safe_compute_block


def configure_runtime_environment(*, n_jobs: int, backend: str = "rust") -> None:
    """Set the runtime environment for reproducible inference."""

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ["S2AND_BACKEND"] = backend
    thread_count = str(max(1, int(n_jobs)))
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["RAYON_NUM_THREADS"] = thread_count


def _resolve_dataset_file(data_root: Path, dataset_name: str, *candidates: str) -> str:
    for candidate in candidates:
        path = data_root / dataset_name / candidate
        if path.exists():
            return str(path)
    attempted_paths = [str(data_root / dataset_name / value) for value in candidates]
    raise FileNotFoundError(f"Missing dataset file for {dataset_name!r}; tried {attempted_paths}")


def _resolve_specter_file(data_root: Path, dataset_name: str) -> str | None:
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


def load_labeled_dataset(
    data_root: Path,
    dataset_name: str,
    *,
    n_jobs: int,
    clusterer: Any | None = None,
    load_name_counts: LoadNameCountsMode = "auto",
) -> ANDData:
    """Load one labeled dataset for promoted replay generation or evaluation."""

    install_safe_compute_block_patch()
    configure_runtime_environment(n_jobs=n_jobs, backend="rust")
    return ANDData(
        signatures=_resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_signatures.json", "signatures.json"),
        papers=_resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_papers.json", "papers.json"),
        name=dataset_name,
        mode="inference",
        specter_embeddings=_resolve_specter_file(data_root, dataset_name),
        clusters=_resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"),
        block_type="s2",
        n_jobs=int(n_jobs),
        load_name_counts=resolve_load_name_counts(load_name_counts=load_name_counts, clusterer=clusterer),
        preprocess=True,
        random_seed=13,
        name_tuples="filtered",
        use_orcid_id=False,
        use_sinonym_overwrite=False,
    )


def _subblock_tokens(subblock_key: str) -> list[str]:
    values: set[str] = set()
    for raw_token in str(subblock_key).split(","):
        token = str(raw_token).strip().split("|", 1)[0].strip()
        if len(token) > 1:
            values.add(token)
    return sorted(values)


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


def build_cluster_profile(summary: ClusterSummary) -> ClusterProfile:
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


def counter_query_overlap(query_values: frozenset[str], counter: Counter[str], size: int) -> float:
    """Return average per-query-token coverage in one cluster counter."""

    if size <= 0 or not query_values or not counter:
        return 0.0
    overlap = sum(float(counter[value]) / float(size) for value in query_values if value in counter)
    return float(overlap / float(len(query_values)))


def middle_initial_compatibility(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return the promoted middle-initial compatibility signal."""

    if not query.middle_initials or not summary.middle_initial_counts or summary.size <= 0:
        return 0.0
    overlap = query.middle_initials.intersection(summary.middle_initial_counts.keys())
    if overlap:
        return float(
            sum(float(summary.middle_initial_counts[value]) / float(summary.size) for value in overlap)
            / float(len(query.middle_initials))
        )
    return retrieval_policy.RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE


def year_compatibility(query_year: int | None, summary: ClusterSummary) -> float:
    """Return the promoted publication-year compatibility signal."""

    if query_year is None or summary.year_mean is None:
        return 0.0
    distance = abs(float(query_year) - float(summary.year_mean))
    score = max(0.0, 1.0 - (distance / retrieval_policy.RETRIEVAL_YEAR_SCORE_DECAY_YEARS))
    if summary.year_min is not None and summary.year_max is not None:
        if (
            query_year < int(summary.year_min) - retrieval_policy.RETRIEVAL_YEAR_SCORE_RANGE_GAP
            or query_year > int(summary.year_max) + retrieval_policy.RETRIEVAL_YEAR_SCORE_RANGE_GAP
        ):
            score -= retrieval_policy.RETRIEVAL_YEAR_SCORE_RANGE_PENALTY
    return float(score)


def title_overlap(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return query-title overlap against the cluster title counter."""

    return float(counter_query_overlap(query.title_terms, summary.title_counts, summary.size))


def specter_exemplar_similarity(query: QueryFeatures, summary: ClusterSummary) -> float:
    """Return max SPECTER similarity to cluster exemplars."""

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


def _resolve_rust_num_threads(num_threads: int | None) -> int | None:
    if num_threads is not None:
        return max(1, int(num_threads))
    for env_var in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        raw_value = os.environ.get(env_var)
        if raw_value is None or not str(raw_value).strip():
            continue
        try:
            parsed = int(raw_value)
        except ValueError:
            continue
        if parsed > 0:
            return int(parsed)
    return None


def rank_top_summaries_rust_hybrid_centroid(
    *,
    query: QueryFeatures,
    max_ranked_clusters: int,
    retriever: RustHybridCentroidRetrieverHandle,
    component_keys: list[str] | None = None,
    max_block_component_size: int | None = None,
    override_summary: ClusterSummary | None = None,
    num_threads: int | None = None,
    weights: tuple[float, ...] | list[float] | None = None,
    scoring_config: RustHybridCentroidScoringConfig | dict[str, Any] | None = None,
) -> list[tuple[float, ClusterSummary]]:
    """Score and rank candidate summaries with the Rust hybrid-centroid path."""

    if max_ranked_clusters <= 0:
        raise ValueError("max_ranked_clusters must be positive")
    resolved_num_threads = _resolve_rust_num_threads(num_threads)
    weights_payload = None if weights is None else [float(value) for value in weights]
    scoring_kwargs = None
    if scoring_config is not None:
        if isinstance(scoring_config, RustHybridCentroidScoringConfig):
            scoring_kwargs = scoring_config.to_kwargs()
        else:
            scoring_kwargs = dict(scoring_config)
    if component_keys is None:
        if scoring_kwargs is not None:
            raise ValueError("scoring_config requires component_keys so the experimental subset scorer can be used")
        if weights_payload is None:
            ranked_component_keys, scores = retriever.retriever.top_k_hybrid_centroid(
                query,
                top_k=int(max_ranked_clusters),
                num_threads=resolved_num_threads,
            )
        else:
            ranked_component_keys, scores = retriever.retriever.top_k_weighted_hybrid_centroid(
                query,
                top_k=int(max_ranked_clusters),
                weights=weights_payload,
                num_threads=resolved_num_threads,
            )
    else:
        if max_block_component_size is None:
            raise ValueError("max_block_component_size is required when component_keys are provided")
        if scoring_kwargs is not None:
            if weights_payload is None:
                raise ValueError("scoring_config requires explicit weights")
            ranked_component_keys, scores = retriever.retriever.top_k_experimental_weighted_hybrid_centroid_subset(
                query,
                component_keys,
                top_k=int(max_ranked_clusters),
                max_block_component_size=int(max_block_component_size),
                weights=weights_payload,
                num_threads=resolved_num_threads,
                override_summary=override_summary,
                **scoring_kwargs,
            )
        elif weights_payload is None:
            ranked_component_keys, scores = retriever.retriever.top_k_hybrid_centroid_subset(
                query,
                component_keys,
                top_k=int(max_ranked_clusters),
                max_block_component_size=int(max_block_component_size),
                num_threads=resolved_num_threads,
                override_summary=override_summary,
            )
        else:
            ranked_component_keys, scores = retriever.retriever.top_k_weighted_hybrid_centroid_subset(
                query,
                component_keys,
                top_k=int(max_ranked_clusters),
                max_block_component_size=int(max_block_component_size),
                weights=weights_payload,
                num_threads=resolved_num_threads,
                override_summary=override_summary,
            )
    return [
        (
            float(score),
            (
                override_summary
                if override_summary is not None and str(component_key) == str(override_summary.component_key)
                else retriever.summary_by_component[str(component_key)]
            ),
        )
        for component_key, score in zip(ranked_component_keys, scores, strict=True)
    ]


__all__ = [
    "DEFAULT_CHOOSER_CACHE_MAX_TOP_K",
    "FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY",
    "FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME",
    "ClusterProfile",
    "FrozenRustHybridCentroidPolicy",
    "RustHybridCentroidScoringConfig",
    "build_cluster_profile",
    "build_cluster_summary",
    "build_labeled_retrieval_subblock_index",
    "build_rust_hybrid_centroid_retriever",
    "counter_query_overlap",
    "load_labeled_dataset",
    "middle_initial_compatibility",
    "name_count_rarity_features",
    "rank_top_summaries_rust_hybrid_centroid",
    "specter_exemplar_similarity",
    "title_overlap",
    "year_compatibility",
]
