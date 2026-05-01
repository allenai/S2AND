"""Shared helpers for single-letter retrieval candidate generation.

These helpers keep deterministic query selection, seed-summary construction,
and top-k retrieval ranking out of the larger reranker pipeline module.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scripts.eval_cluster_retrieval as retrieval
except ImportError:  # pragma: no cover - direct script execution path
    import eval_cluster_retrieval as retrieval  # type: ignore

try:
    import s2and_rust
except ImportError:  # pragma: no cover - Rust extension optional
    s2and_rust = None  # type: ignore[assignment]

RAW_SIGNATURE_TO_CLUSTER_ID_FILENAME = "signature_to_cluster_id.json"
RECONCILED_SIGNATURE_TO_CLUSTER_ID_FILENAME = "reconciled_signature_to_cluster_id.json"
RECONCILED_SIGNATURE_TO_CLUSTER_ID_SUMMARY_FILENAME = "reconciled_signature_to_cluster_id.summary.json"


def _rust_default_tuple(name: str, fallback: tuple[Any, ...]) -> tuple[Any, ...]:
    if s2and_rust is None or not hasattr(s2and_rust, name):
        return fallback
    return tuple(getattr(s2and_rust, name))


def _read_json(path: Path) -> Any:
    """Load one JSON payload from disk."""

    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def load_preferred_signature_to_cluster_id(step2_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Load the reconciled cluster assignment when present, else the raw step-2 mapping."""

    candidates = (
        ("reconciled", step2_dir / RECONCILED_SIGNATURE_TO_CLUSTER_ID_FILENAME),
        ("raw", step2_dir / RAW_SIGNATURE_TO_CLUSTER_ID_FILENAME),
    )
    for source_kind, path in candidates:
        if not path.exists():
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid signature_to_cluster_id payload at {path}: expected object")
        normalized = {str(signature_id): str(cluster_id) for signature_id, cluster_id in payload.items()}
        info = {
            "source": str(source_kind),
            "path": str(path),
            "assignment_count": int(len(normalized)),
        }
        summary_path = step2_dir / RECONCILED_SIGNATURE_TO_CLUSTER_ID_SUMMARY_FILENAME
        if source_kind == "reconciled" and summary_path.exists():
            info["summary_path"] = str(summary_path)
        return normalized, info
    raise FileNotFoundError(f"Missing {RAW_SIGNATURE_TO_CLUSTER_ID_FILENAME} under {step2_dir}")


@dataclass(frozen=True)
class RustHybridCentroidRetrieverHandle:
    """Cached Python + Rust state for exact `hybrid_centroid` retrieval."""

    retriever: Any
    summary_by_component: dict[str, retrieval.ClusterSummary]


@dataclass(frozen=True)
class RustHybridCentroidScoringConfig:
    """Optional experimental scoring controls for the Rust retriever."""

    first_name_mode: str = "prefix"
    specter_mode: str = "centroid"
    coauthor_use_idf: bool = False
    coauthor_per_term_cap: float | None = None
    coauthor_total_cap: float | None = None
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
            "affiliation_use_idf": bool(self.affiliation_use_idf),
            "affiliation_per_term_cap": self.affiliation_per_term_cap,
            "affiliation_total_cap": self.affiliation_total_cap,
            "affiliation_min_token_count": int(self.affiliation_min_token_count),
            "affiliation_unigram_weight": float(self.affiliation_unigram_weight),
            "affiliation_multi_token_weight": float(self.affiliation_multi_token_weight),
        }


@dataclass(frozen=True)
class FrozenRustHybridCentroidPolicy:
    """Frozen tuned Rust retriever policy used for chooser-stage row generation."""

    full_weights: tuple[float, ...]
    initial_only_weights: tuple[float, ...]
    full_scoring_config: RustHybridCentroidScoringConfig | None = None
    initial_only_scoring_config: RustHybridCentroidScoringConfig | None = None
    full_candidate_strategy: str = "global"

    def weights_for_query(self, query: retrieval.QueryFeatures) -> tuple[float, ...]:
        """Return the weights for the current masked query view."""

        return self.full_weights if bool(query.has_full_first) else self.initial_only_weights

    def scoring_config_for_query(
        self,
        query: retrieval.QueryFeatures,
    ) -> RustHybridCentroidScoringConfig | None:
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
            "feature_order": list(RUST_HYBRID_CENTROID_FEATURE_ORDER),
            "full_candidate_strategy": str(self.full_candidate_strategy),
            "full_weights": {
                name: round(float(value), 6)
                for name, value in zip(RUST_HYBRID_CENTROID_FEATURE_ORDER, self.full_weights, strict=True)
            },
            "initial_only_weights": {
                name: round(float(value), 6)
                for name, value in zip(RUST_HYBRID_CENTROID_FEATURE_ORDER, self.initial_only_weights, strict=True)
            },
            "full_scoring_config": (
                self.full_scoring_config.to_kwargs() if self.full_scoring_config is not None else None
            ),
            "initial_only_scoring_config": (
                self.initial_only_scoring_config.to_kwargs() if self.initial_only_scoring_config is not None else None
            ),
        }


RUST_HYBRID_CENTROID_FEATURE_ORDER = tuple(
    str(value) for value in _rust_default_tuple("RETRIEVAL_FEATURE_ORDER", retrieval.HYBRID_FEATURE_ORDER)
)
DEFAULT_RUST_HYBRID_CENTROID_WEIGHTS = tuple(
    float(value)
    for value in _rust_default_tuple(
        "DEFAULT_HYBRID_CENTROID_WEIGHTS",
        retrieval.DEFAULT_HYBRID_CENTROID_WEIGHTS,
    )
)
FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY_NAME = "h_wang_any_input_v2"
FROZEN_BEST_RUST_HYBRID_CENTROID_POLICY = FrozenRustHybridCentroidPolicy(
    full_weights=(0.527232, 0.223412, 0.146909, 0.009439, 0.093007),
    initial_only_weights=(0.520012, 0.220264, 0.109278, 0.150447, 0.0),
    full_scoring_config=RustHybridCentroidScoringConfig(
        first_name_mode="exact_then_prefix_half",
        specter_mode="max_centroid_exemplar",
        coauthor_use_idf=True,
        coauthor_per_term_cap=0.35,
        affiliation_use_idf=True,
    ),
    initial_only_scoring_config=RustHybridCentroidScoringConfig(
        first_name_mode="prefix",
        specter_mode="max_centroid_exemplar",
        coauthor_use_idf=True,
        coauthor_per_term_cap=0.35,
        affiliation_use_idf=True,
    ),
    full_candidate_strategy="name_compat_plus_global_backfill5",
)


def select_query_ids(
    query_ids: list[str],
    *,
    limit_queries: int | None,
    seed: int,
) -> list[str]:
    """Select a deterministic query subset for pilot runs."""

    if limit_queries is None or limit_queries >= len(query_ids):
        return list(query_ids)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(np.asarray(query_ids, dtype=object), size=int(limit_queries), replace=False)
    return sorted(str(query_id) for query_id in selected.tolist())


def invert_signature_to_cluster_id(signature_to_cluster_id: dict[str, str]) -> dict[str, list[str]]:
    """Invert a signature -> cluster map into cluster -> sorted signatures."""

    clusters: dict[str, list[str]] = {}
    for signature_id, cluster_id in signature_to_cluster_id.items():
        clusters.setdefault(str(cluster_id), []).append(str(signature_id))
    for cluster_id in clusters:
        clusters[cluster_id] = sorted(clusters[cluster_id])
    return clusters


def build_seed_summaries(
    *,
    dataset: Any,
    seed_clusters: dict[str, list[str]],
    block_key: str,
    max_exemplars: int,
) -> tuple[list[retrieval.ClusterSummary], dict[str, int], float]:
    """Build retrieval summaries for persisted seed clusters."""

    feature_cache: dict[str, retrieval.QueryFeatures] = {}
    summaries: list[retrieval.ClusterSummary] = []
    cluster_sizes: dict[str, int] = {}
    start = time.perf_counter()
    for cluster_id, signature_ids in seed_clusters.items():
        cluster_sizes[str(cluster_id)] = len(signature_ids)
        summaries.append(
            retrieval.build_cluster_summary(
                dataset=dataset,
                block_key=block_key,
                cluster_id=str(cluster_id),
                component_key=str(cluster_id),
                signature_ids=[str(signature_id) for signature_id in signature_ids],
                max_exemplars=max_exemplars,
                feature_cache=feature_cache,
                orcid_enabled=False,
            )
        )
    build_ms = (time.perf_counter() - start) * 1000.0
    return summaries, cluster_sizes, build_ms


def rank_top_summaries(
    *,
    method: str,
    query: retrieval.QueryFeatures,
    candidate_summaries: list[retrieval.ClusterSummary],
    max_block_component_size: int,
    max_ranked_clusters: int,
) -> list[tuple[float, retrieval.ClusterSummary]]:
    """Score and rank candidate summaries for one retrieval method."""

    if not candidate_summaries:
        return []
    if max_ranked_clusters <= 0:
        raise ValueError("max_ranked_clusters must be positive")
    scores = np.fromiter(
        (
            retrieval.score_summary(method, query, summary, max_block_component_size=max_block_component_size)
            for summary in candidate_summaries
        ),
        dtype=np.float32,
        count=len(candidate_summaries),
    )
    top_n = min(int(max_ranked_clusters), len(candidate_summaries))
    top_indices = np.argpartition(-scores, top_n - 1)[:top_n]
    top_indices = sorted(
        top_indices.tolist(),
        key=lambda idx: (-float(scores[idx]), candidate_summaries[idx].component_key),
    )
    return [(float(scores[idx]), candidate_summaries[idx]) for idx in top_indices]


def build_rust_hybrid_centroid_retriever(
    candidate_summaries: list[retrieval.ClusterSummary],
    *,
    include_exemplars: bool = False,
) -> RustHybridCentroidRetrieverHandle:
    """Build the optional Rust-backed exact retriever for `hybrid_centroid`."""

    if s2and_rust is None or not hasattr(s2and_rust, "RustHybridCentroidRetriever"):
        raise RuntimeError("RustHybridCentroidRetriever is unavailable; build/install s2and_rust first")
    return RustHybridCentroidRetrieverHandle(
        retriever=s2and_rust.RustHybridCentroidRetriever(
            candidate_summaries,
            include_exemplars=bool(include_exemplars),
        ),
        summary_by_component={str(summary.component_key): summary for summary in candidate_summaries},
    )


def build_rust_name_compatible_subblock_selector(retrieval_subblock_index: dict[str, Any]) -> Any:
    """Build the Rust selector for strict name-compatible subblock candidate gating."""

    if s2and_rust is None or not hasattr(s2and_rust, "RustNameCompatibleSubblockSelector"):
        raise RuntimeError("RustNameCompatibleSubblockSelector is unavailable; build/install s2and_rust first")
    return s2and_rust.RustNameCompatibleSubblockSelector(retrieval_subblock_index)


def _resolve_rust_num_threads(num_threads: int | None) -> int | None:
    """Resolve the Rust retriever thread count from the explicit arg or runtime env."""

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
    query: retrieval.QueryFeatures,
    max_ranked_clusters: int,
    retriever: RustHybridCentroidRetrieverHandle,
    component_keys: list[str] | None = None,
    max_block_component_size: int | None = None,
    override_summary: retrieval.ClusterSummary | None = None,
    num_threads: int | None = None,
    weights: tuple[float, ...] | list[float] | None = None,
    scoring_config: RustHybridCentroidScoringConfig | dict[str, Any] | None = None,
) -> list[tuple[float, retrieval.ClusterSummary]]:
    """Score and rank candidate summaries with the optional Rust `hybrid_centroid` path."""

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


def compute_chooser_summary_features_rust_hybrid_centroid(
    *,
    query: retrieval.QueryFeatures,
    component_keys: list[str],
    summary_by_component: dict[str, retrieval.ClusterSummary],
    retriever: RustHybridCentroidRetrieverHandle,
    num_threads: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute chooser summary features for a candidate subset with the Rust retriever."""

    if not hasattr(retriever.retriever, "chooser_feature_rows_subset"):
        raise RuntimeError("chooser_feature_rows_subset is unavailable on RustHybridCentroidRetriever")
    override_summary = None
    override_component_keys: list[str] = []
    for component_key in component_keys:
        current_summary = summary_by_component[str(component_key)]
        base_summary = retriever.summary_by_component.get(str(component_key))
        if base_summary is None:
            raise KeyError(f"Unknown component_key for RustHybridCentroidRetriever: {component_key}")
        if current_summary is not base_summary:
            override_component_keys.append(str(component_key))
            override_summary = current_summary
    if len(override_component_keys) > 1:
        raise ValueError(
            "Rust chooser feature extraction supports at most one override summary per subset; "
            f"got {override_component_keys}"
        )
    resolved_num_threads = _resolve_rust_num_threads(num_threads)
    payload = retriever.retriever.chooser_feature_rows_subset(
        query,
        component_keys,
        num_threads=resolved_num_threads,
        override_summary=override_summary,
    )
    return {
        str(component_key): {str(name): float(value) for name, value in dict(feature_values).items()}
        for component_key, feature_values in dict(payload).items()
    }
