"""Bridge-mode row generation surface for the reranker dataset migration."""

from __future__ import annotations

from typing import Any

try:
    from scripts.single_letter_reranker_utils import (
        RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
        apply_hard_disallow_component_filter,
        make_candidate_rows,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from single_letter_reranker_utils import (  # type: ignore
        RAW_METADATA_SIMILARITY_FEATURE_COLUMNS,
        apply_hard_disallow_component_filter,
        make_candidate_rows,
    )


def _validate_raw_similarity_features(
    *,
    query_case: Any,
    shortlist_component_keys: list[str],
    retrieval_ranks: dict[str, int],
    stats_by_component: dict[str, Any],
    raw_similarity_features_by_component: dict[str, dict[str, float]] | None,
) -> None:
    """Require explicit raw metadata features for every row the bridge can emit."""

    if not shortlist_component_keys:
        return
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
    kept_component_keys = [str(component_key) for component_key in hard_disallow_filter.kept_component_keys]
    if not kept_component_keys:
        return
    if raw_similarity_features_by_component is None:
        raise ValueError(
            "Raw metadata similarity features are required for bridge row generation; "
            f"missing components={kept_component_keys}"
        )
    required_features = set(RAW_METADATA_SIMILARITY_FEATURE_COLUMNS)
    missing_components: list[str] = []
    missing_features_by_component: dict[str, list[str]] = {}
    for component_key in kept_component_keys:
        component_features = raw_similarity_features_by_component.get(str(component_key))
        if component_features is None:
            missing_components.append(str(component_key))
            continue
        missing_features = sorted(required_features - set(component_features))
        if missing_features:
            missing_features_by_component[str(component_key)] = missing_features
    if missing_components or missing_features_by_component:
        raise ValueError(
            "Missing raw metadata similarity features for bridge row generation: "
            f"missing_components={missing_components} missing_features={missing_features_by_component}"
        )


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

    _validate_raw_similarity_features(
        query_case=query_case,
        shortlist_component_keys=shortlist_component_keys,
        retrieval_ranks=retrieval_ranks,
        stats_by_component=stats_by_component,
        raw_similarity_features_by_component=raw_similarity_features_by_component,
    )
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
