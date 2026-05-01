"""Bridge-mode row generation surface for the reranker dataset migration."""

from __future__ import annotations

from typing import Any

try:
    from scripts.single_letter_reranker_utils import make_candidate_rows
except ImportError:  # pragma: no cover - direct script execution path
    from single_letter_reranker_utils import make_candidate_rows  # type: ignore


def generate_candidate_rows(
    *,
    query_case: Any,
    query_view: str,
    query_features: Any,
    shortlist_component_keys: list[str],
    retrieval_scores: dict[str, float],
    retrieval_ranks: dict[str, int],
    retrieval_window_state: dict[str, int],
    summary_by_component: dict[str, Any],
    stats_by_component: dict[str, Any],
    rust_hybrid_centroid_retriever: Any | None = None,
    raw_similarity_features_by_component: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Generate candidate rows through the Phase 4 bridge surface.

    This function intentionally delegates to the existing row materializer while
    Phase 4 is still in bridge mode. Byte-parity tests pin this contract so the
    implementation can move behind this API without changing persisted rows.
    """

    return make_candidate_rows(
        query_case=query_case,
        query_view=query_view,
        query_features=query_features,
        shortlist_component_keys=shortlist_component_keys,
        retrieval_scores=retrieval_scores,
        retrieval_ranks=retrieval_ranks,
        retrieval_window_state=retrieval_window_state,
        summary_by_component=summary_by_component,
        stats_by_component=stats_by_component,
        rust_hybrid_centroid_retriever=rust_hybrid_centroid_retriever,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )
