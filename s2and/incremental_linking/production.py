"""Production orchestration for the promoted incremental linker."""

from __future__ import annotations

import hashlib
import heapq
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import s2and.incremental_linking.artifact as artifact_module
import s2and.incremental_linking.query_adapter as query_adapter_module
import s2and.incremental_linking.runtime as runtime_module
from s2and import feature_port, memory_budget
from s2and.arrow_inputs import require_feature_contract_normalization_version, validate_arrow_prediction_artifacts
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import (
    cluster_seed_disallows_from_arrow_paths,
    feature_block_signature_order_from_raw_candidate_plan,
    read_cluster_seeds_arrow,
    temporary_arrow_paths_with_cluster_seeds,
)
from s2and.incremental_linking.policy import (
    clusterer_uses_embedding_features,
    clusterer_uses_name_count_features,
    request_cluster_seed_disallow_parts,
    require_arrow_name_counts_index_for_clusterer,
    require_dataset_name_counts_binding_for_clusterer,
)
from s2and.runtime import RuntimeContext

logger = logging.getLogger("s2and")

_RAW_ARROW_PLAN_WINDOW_MULTIPLIER = 4

_PROMOTED_INCREMENTAL_SUM_TELEMETRY_FIELDS = frozenset(
    {
        "abstain_count",
        "candidate_row_count",
        "cluster_seed_disallow_excluded_query_count",
        "cluster_seed_disallow_excluded_row_count",
        "cluster_seed_disallow_same_batch_conflict_count",
        "cluster_seed_disallow_same_batch_demoted_abstain_count",
        "cluster_seed_disallow_same_batch_reassigned_link_count",
        "constraint_elapsed_seconds",
        "constraint_partial_supervision_hits",
        "constraint_rust_batch_call_count",
        "constraint_total_pairs",
        "constraint_unresolved_pairs",
        "decision_count",
        "link_count",
        "native_scorer_chunk_count",
        "no_candidate_query_count",
        "pair_count",
        "pairwise_candidate_row_count",
        "pairwise_chunk_count",
        "pairwise_feature_seconds",
        "pairwise_hard_disallow_distance_pair_count",
        "pairwise_pair_count",
        "pairwise_predict_seconds",
        "pairwise_predicted_index_remap_bytes",
        "pairwise_total_seconds",
        "partial_supervision_disallow_between_residual_queries",
        "partial_supervision_disallow_outside_retrieval_window",
        "partial_supervision_ignored_outside_window",
        "partial_supervision_pair_count",
        "partial_supervision_require_outside_retrieval_window",
        "query_count",
        "raw_arrow_featurizer_reused",
        "raw_arrow_featurizer_seconds",
        "raw_arrow_retrieval_seconds",
        "raw_arrow_signal_seconds",
        "retrieved_component_count",
        "row_feature_generated_family_id_count",
        "row_feature_generic_family_override_count",
        *(
            f"raw_arrow_plan_{field}"
            for field in (
                "component_members_payload_secs",
                "component_members_secs",
                "excluded_query_seed_count",
                "feature_secs",
                "metadata_reads_parallel_secs",
                "pair_signature_ids_secs",
                "paper_author_batches_read",
                "paper_author_paper_count",
                "paper_author_rows_scanned",
                "paper_batches_read",
                "paper_count",
                "paper_rows_scanned",
                "payload_seed_signature_count",
                "payload_secs",
                "planner_seed_state_built",
                "planner_seed_state_reused",
                "query_secs",
                "query_signature_count",
                "read_cluster_seeds_secs",
                "read_name_counts_secs",
                "read_paper_authors_secs",
                "read_papers_secs",
                "read_signatures_secs",
                "read_specter_secs",
                "retrieval_secs",
                "signature_batches_read",
                "signature_count",
                "signature_rows_scanned",
                "specter_batches_read",
                "specter_count",
                "specter_rows_scanned",
                "summary_secs",
                "text_context_secs",
                "total_secs",
                "unidecode_char_count",
                "wall_secs",
            )
        ),
    }
)

_PROMOTED_INCREMENTAL_TELEMETRY_MERGE_POLICY = {
    **{key: "sum_numeric" for key in _PROMOTED_INCREMENTAL_SUM_TELEMETRY_FIELDS},
    "retrieval_top_k": "constant",
    "seed_signature_count": "constant",
    "seed_component_count": "constant",
    "raw_arrow_seed_signature_count": "constant",
    "raw_arrow_seed_component_count": "constant",
    "raw_arrow_plan_seed_signature_count": "constant",
    "raw_arrow_plan_cluster_count": "constant",
    "raw_arrow_plan_cluster_seed_disallow_pair_count": "constant",
    "raw_arrow_plan_indexed_arrow_candidate_plan": "constant",
    "raw_arrow_plan_matrix_feature_count": "constant",
    "pairwise_aggregate_feature_count": "constant",
    "pairwise_index_remap_bytes_per_pair": "constant",
    "pairwise_matrix_feature_count": "constant",
    "memory_total_ram_bytes": "first",
    "memory_available_bytes": "first",
    "memory_stage_budget_bytes": "first",
    "memory_predicted_peak_delta_bytes": "first",
    "memory_predicted_peak_rss_bytes": "first",
    "memory_rss_before_bytes": "first",
    "memory_rss_peak_bytes": "first",
    "memory_rss_after_bytes": "first",
    "memory_observed_peak_delta_bytes": "first",
    "memory_observed_end_delta_bytes": "first",
    "memory_prediction_error_ratio": "first",
    "memory_underpredicted": "first",
    "native_scorer_chunk_rows": "min_numeric",
    "native_scorer_stage_budget_bytes": "min_numeric",
    "native_scorer_predicted_peak_delta_bytes": "max_numeric",
    "native_scorer_predicted_peak_rss_bytes": "max_numeric",
}


@dataclass(frozen=True)
class _ScoredQueryDecision:
    """One initial or conflict-rescored query decision with priority metadata."""

    signature_id: str
    decision: runtime_module.LinkOrAbstainDecision
    require_forced: bool


@dataclass(frozen=True)
class _PromotedIncrementalMemoryLayout:
    """Exact persistent and transient feature widths for RAM planning."""

    final_matrix_feature_count: int
    pairwise_matrix_feature_count: int
    aggregate_feature_count: int


def _promoted_incremental_memory_layout(
    clusterer: Any,
    artifact: artifact_module.IncrementalLinkingArtifact,
) -> _PromotedIncrementalMemoryLayout:
    pairwise_matrix_feature_count, aggregate_feature_count = runtime_module.promoted_pairwise_memory_feature_counts(
        clusterer
    )
    return _PromotedIncrementalMemoryLayout(
        final_matrix_feature_count=len(artifact.metadata.feature_columns),
        pairwise_matrix_feature_count=pairwise_matrix_feature_count,
        aggregate_feature_count=aggregate_feature_count,
    )


def _query_decision_priority_key(item: _ScoredQueryDecision) -> tuple[int, float, str]:
    return (
        0 if item.require_forced else 1,
        -float(item.decision.score) if item.decision.score is not None else math.inf,
        item.signature_id,
    )


def _resolve_query_disallows_globally(
    initial_decisions: Mapping[str, _ScoredQueryDecision],
    disallow_partners: Mapping[str, set[str]],
    *,
    rescore: Callable[[str, set[str]], _ScoredQueryDecision],
) -> tuple[dict[str, str], dict[str, int]]:
    """Resolve query-query disallows with one request-global deterministic priority."""

    order = sorted(initial_decisions.values(), key=_query_decision_priority_key)
    finalized: dict[str, _ScoredQueryDecision] = {}
    telemetry = {
        "global_query_disallow_endpoint_count": len(initial_decisions),
        "global_query_disallow_conflict_count": 0,
        "global_query_disallow_rescore_count": 0,
        "global_query_disallow_reassigned_link_count": 0,
        "global_query_disallow_demoted_abstain_count": 0,
    }
    for initial in order:
        signature_id = initial.signature_id
        excluded_components = {
            str(partner_decision.decision.component_key)
            for partner_id in disallow_partners.get(signature_id, ())
            if (partner_decision := finalized.get(str(partner_id))) is not None
            and partner_decision.decision.action == "link"
            and partner_decision.decision.component_key is not None
        }
        decision = initial
        if (
            decision.decision.action == "link"
            and decision.decision.component_key is not None
            and str(decision.decision.component_key) in excluded_components
        ):
            telemetry["global_query_disallow_conflict_count"] += 1
            if decision.require_forced:
                raise ValueError(
                    "cluster_seed_disallow_conflicts_with_require_constraint: "
                    f"query_signature_id={signature_id!r} component_key={decision.decision.component_key!r}"
                )
            decision = rescore(signature_id, excluded_components)
            if decision.signature_id != signature_id:
                raise ValueError(
                    "Global query-disallow rescore returned the wrong signature id: "
                    f"expected={signature_id!r} observed={decision.signature_id!r}"
                )
            telemetry["global_query_disallow_rescore_count"] += 1
            if decision.decision.action == "link":
                component_key = decision.decision.component_key
                if component_key is None or str(component_key) in excluded_components:
                    raise ValueError(
                        "Global query-disallow rescore returned an excluded or missing component: "
                        f"query_signature_id={signature_id!r} component_key={component_key!r} "
                        f"excluded_components={sorted(excluded_components)!r}"
                    )
                telemetry["global_query_disallow_reassigned_link_count"] += 1
            else:
                telemetry["global_query_disallow_demoted_abstain_count"] += 1
        finalized[signature_id] = decision

    linked = {
        signature_id: str(scored.decision.component_key)
        for signature_id, scored in finalized.items()
        if scored.decision.action == "link" and scored.decision.component_key is not None
    }
    return linked, telemetry


def _scored_query_decisions_from_result(
    result: runtime_module.LinkOrAbstainProductionResult,
    *,
    signature_ids_by_index: Sequence[str],
    expected_query_signature_ids: Sequence[str],
) -> dict[str, _ScoredQueryDecision]:
    """Extract compact query decisions without retaining batch feature matrices."""

    expected_ids = {str(signature_id) for signature_id in expected_query_signature_ids}
    require_counts = result.pairwise_model_result.row_signals.get("constraint_require_count")
    scored: dict[str, _ScoredQueryDecision] = {}
    for decision in result.compact_result.decisions:
        query_index = int(decision.query_signature_index)
        if query_index < 0 or query_index >= len(signature_ids_by_index):
            raise ValueError(
                "Promoted query decision index is outside the Rust featurizer signature order: "
                f"query_index={query_index} signature_count={len(signature_ids_by_index)}"
            )
        signature_id = str(signature_ids_by_index[query_index])
        if signature_id not in expected_ids:
            raise ValueError(
                "Promoted query decision references a signature outside the scored batch: "
                f"signature_id={signature_id!r}"
            )
        if signature_id in scored:
            raise ValueError(f"Promoted query decision is duplicated: signature_id={signature_id!r}")
        require_forced = False
        if decision.row_index is not None and require_counts is not None:
            row_index = int(decision.row_index)
            if row_index < 0 or row_index >= len(require_counts):
                raise ValueError(
                    "Promoted query decision row is outside constraint_require_count: "
                    f"signature_id={signature_id!r} row_index={row_index} row_count={len(require_counts)}"
                )
            require_forced = float(require_counts[row_index]) > 0.0
        scored[signature_id] = _ScoredQueryDecision(
            signature_id=signature_id,
            decision=decision,
            require_forced=require_forced,
        )
    missing = expected_ids - set(scored)
    if missing:
        raise ValueError(f"Promoted query result is missing decisions: signature_ids={sorted(missing)!r}")
    return scored


def _raw_arrow_plan_window_size(
    *,
    query_count: int,
    query_batch_size: int,
    plan_window_multiplier: int,
) -> int:
    """Return a positive window step for raw Arrow planning loops."""

    resolved_query_count = max(0, int(query_count))
    resolved_query_batch_size = max(1, int(query_batch_size))
    return max(1, min(resolved_query_count, resolved_query_batch_size * max(1, int(plan_window_multiplier))))


def _raw_arrow_plan_windows(
    query_signature_ids: Sequence[str],
    *,
    window_size: int,
    seed_signature_ids: set[str],
) -> list[list[str]]:
    """Build raw-planner windows without mixing seed-overlap queries."""

    resolved_window_size = max(1, int(window_size))
    windows: list[list[str]] = []
    current: list[str] = []
    for signature_id in query_signature_ids:
        if signature_id in seed_signature_ids:
            if current:
                windows.append(current)
                current = []
            windows.append([signature_id])
            continue
        current.append(signature_id)
        if len(current) >= resolved_window_size:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows


def _disallow_aware_query_batches(
    query_signature_ids: Sequence[str],
    *,
    query_batch_size: int,
    disallow_components_by_id: Mapping[str, frozenset[str]],
) -> list[list[str]]:
    """Pack complete query-disallow components together whenever the budget permits."""

    resolved_batch_size = max(1, int(query_batch_size))
    query_ids = [str(signature_id) for signature_id in query_signature_ids]
    query_id_set = set(query_ids)
    if len(query_id_set) != len(query_ids):
        raise ValueError("Promoted incremental query signature ids must be unique")
    if not disallow_components_by_id:
        return [
            query_ids[start : start + resolved_batch_size] for start in range(0, len(query_ids), resolved_batch_size)
        ]
    components = {
        frozenset(component & query_id_set)
        for signature_id in query_id_set
        if (component := disallow_components_by_id.get(signature_id)) is not None
    }
    component_query_ids = set().union(*components) if components else set()
    units = [sorted(component) for component in components if component]
    free_query_ids = sorted(query_id_set - component_query_ids)
    split_units = [
        unit[start : start + resolved_batch_size]
        for unit in units
        for start in range(0, len(unit), resolved_batch_size)
    ]
    split_units.sort(key=lambda unit: (-len(unit), tuple(unit)))

    # Preallocate the exact fixed-slicing batch count so component-aware packing
    # can never reduce throughput by creating extra scoring calls. Whole units
    # use deterministic best fit; a unit that cannot fit is split across the
    # remaining capacities and therefore routed through global resolution.
    batch_count = (len(query_ids) + resolved_batch_size - 1) // resolved_batch_size
    batches: list[list[str]] = [[] for _ in range(batch_count)]
    remaining_by_batch = [resolved_batch_size] * batch_count
    batches_by_remaining: list[list[int]] = [[] for _ in range(resolved_batch_size + 1)]
    batches_by_remaining[resolved_batch_size] = list(range(batch_count))
    heapq.heapify(batches_by_remaining[resolved_batch_size])

    def _take_batch_with_capacity(minimum_capacity: int) -> int | None:
        for remaining_capacity in range(minimum_capacity, resolved_batch_size + 1):
            heap = batches_by_remaining[remaining_capacity]
            while heap and remaining_by_batch[heap[0]] != remaining_capacity:
                heapq.heappop(heap)
            if heap:
                return heapq.heappop(heap)
        return None

    def _take_batch_with_largest_capacity() -> int | None:
        for remaining_capacity in range(resolved_batch_size, 0, -1):
            heap = batches_by_remaining[remaining_capacity]
            while heap and remaining_by_batch[heap[0]] != remaining_capacity:
                heapq.heappop(heap)
            if heap:
                return heapq.heappop(heap)
        return None

    def _place(batch_index: int, values: Sequence[str]) -> None:
        batches[batch_index].extend(values)
        remaining_by_batch[batch_index] -= len(values)
        heapq.heappush(batches_by_remaining[remaining_by_batch[batch_index]], batch_index)

    for unit in split_units:
        target = _take_batch_with_capacity(len(unit))
        if target is not None:
            _place(target, unit)
            continue
        remaining_unit = list(unit)
        while remaining_unit:
            target = _take_batch_with_largest_capacity()
            if target is None:
                break
            remaining_capacity = remaining_by_batch[target]
            if remaining_capacity == 0:
                continue
            _place(target, remaining_unit[:remaining_capacity])
            del remaining_unit[:remaining_capacity]
        if remaining_unit:
            raise RuntimeError("Promoted incremental deterministic batch packing exhausted total capacity")

    free_offset = 0
    for batch_index, remaining_capacity in enumerate(remaining_by_batch):
        if remaining_capacity <= 0:
            continue
        values = free_query_ids[free_offset : free_offset + remaining_capacity]
        batches[batch_index].extend(values)
        free_offset += len(values)
    if free_offset != len(free_query_ids):
        raise RuntimeError("Promoted incremental deterministic batch packing did not consume free queries")
    if any(not batch for batch in batches):
        raise RuntimeError("Promoted incremental deterministic batch packing produced an empty batch")
    return batches


def _query_batch_plan_windows(
    query_batches: Sequence[Sequence[str]],
    *,
    plan_window_multiplier: int,
) -> list[list[list[str]]]:
    """Group already-budgeted query batches into reusable planner/featurizer windows."""

    resolved_multiplier = max(1, int(plan_window_multiplier))
    return [
        [list(batch) for batch in query_batches[start : start + resolved_multiplier]]
        for start in range(0, len(query_batches), resolved_multiplier)
    ]


def _resize_query_batch_plan_windows(
    query_batches: Sequence[Sequence[str]],
    *,
    query_batch_size: int,
    plan_window_multiplier: int,
) -> list[list[list[str]]]:
    """Split queued batches to a refreshed limit and regroup bounded plan windows."""

    resolved_batch_size = max(1, int(query_batch_size))
    resized_batches = [
        list(batch[start : start + resolved_batch_size])
        for batch in query_batches
        for start in range(0, len(batch), resolved_batch_size)
    ]
    return _query_batch_plan_windows(
        resized_batches,
        plan_window_multiplier=plan_window_multiplier,
    )


def _raw_window_plan_telemetry_fields(raw_candidate_plan: Mapping[str, Any]) -> dict[str, int | float | str]:
    """Return raw Arrow planner telemetry under the window-plan prefix."""

    telemetry = raw_candidate_plan.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    fields: dict[str, int | float | str] = {}
    for key, value in telemetry.items():
        if key == "timings":
            continue
        if isinstance(value, bool):
            fields[f"raw_arrow_window_plan_{key}"] = int(value)
        elif isinstance(value, int | float | str):
            fields[f"raw_arrow_window_plan_{key}"] = value
    timings = telemetry.get("timings")
    if isinstance(timings, Mapping):
        for key, value in timings.items():
            if isinstance(value, int | float):
                fields[f"raw_arrow_window_plan_{key}"] = float(value)
    return fields


def _merge_raw_window_plan_telemetry(
    merged: dict[str, int | float | str],
    fields: Mapping[str, int | float | str],
) -> None:
    """Merge telemetry from one raw Arrow window into an aggregate payload."""

    for key, value in fields.items():
        if isinstance(value, int | float) and not isinstance(value, bool):
            merged[key] = float(merged.get(key, 0.0)) + float(value)
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = value
        elif existing != value:
            merged[key] = "__mixed__"


def _request_cluster_seed_disallows(
    dataset: ANDData,
    arrow_paths: Mapping[str, Any],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    arrow_disallows = cluster_seed_disallows_from_arrow_paths(arrow_paths)
    return request_cluster_seed_disallow_parts(dataset, arrow_disallows)


def _query_disallow_partner_ids(
    unassigned_signature_ids: Sequence[str],
    request_disallows: set[tuple[str, str]],
    partial_supervision: Mapping[tuple[str, str], int | float],
) -> dict[str, set[str]]:
    """Map each unassigned query to its mutually-disallowed unassigned queries.

    Query-vs-seed disallow pairs are enforced by the raw planner's candidate
    exclusion at plan time. A pair between two queries has no component to
    exclude until one endpoint links, so those pairs are scored without an
    artificial batch winner and finalized request-globally in require/score/id
    priority. A conflicting lower-priority query is rescored with finalized
    partner components excluded.
    """

    query_id_set = {str(signature_id) for signature_id in unassigned_signature_ids}
    partners: dict[str, set[str]] = {}

    def _add_pair(left_id: str, right_id: str) -> None:
        if left_id == right_id:
            return
        partners.setdefault(left_id, set()).add(right_id)
        partners.setdefault(right_id, set()).add(left_id)

    for left, right in request_disallows:
        left_id = str(left)
        right_id = str(right)
        if left_id in query_id_set and right_id in query_id_set:
            _add_pair(left_id, right_id)
    for (left, right), value in partial_supervision.items():
        left_id = str(left)
        right_id = str(right)
        if (
            left_id in query_id_set
            and right_id in query_id_set
            and runtime_module._partial_supervision_kind(value) == "disallow"  # noqa: SLF001
        ):
            _add_pair(left_id, right_id)
    return partners


def _query_disallow_components_by_id(
    disallow_partners: Mapping[str, set[str]],
) -> dict[str, frozenset[str]]:
    """Return the undirected query-disallow component containing each endpoint."""

    components_by_id: dict[str, frozenset[str]] = {}
    unseen = {str(signature_id) for signature_id in disallow_partners}
    while unseen:
        start = min(unseen)
        stack = [start]
        component: set[str] = set()
        while stack:
            signature_id = stack.pop()
            if signature_id in component:
                continue
            component.add(signature_id)
            stack.extend(
                str(partner_id)
                for partner_id in disallow_partners.get(signature_id, ())
                if str(partner_id) not in component
            )
        frozen_component = frozenset(component)
        for signature_id in frozen_component:
            components_by_id[signature_id] = frozen_component
        unseen -= component
    return components_by_id


def _cluster_seed_map_fingerprint(cluster_seeds_require: Mapping[Any, Any]) -> tuple[int, str]:
    digest = hashlib.blake2b(digest_size=16)
    items = sorted(
        (str(signature_id), str(component_id)) for signature_id, component_id in cluster_seeds_require.items()
    )
    for signature_id, component_id in items:
        for value in (signature_id, component_id):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
    return len(items), digest.hexdigest()


def _cluster_seeds_arrow_matches(path_value: Any, cluster_seeds_require: Mapping[Any, Any]) -> bool:
    if path_value is None:
        return False
    path = Path(str(path_value))
    if not path.exists():
        return False
    if path.stat().st_size == 0:
        return False
    arrow_cluster_seeds = read_cluster_seeds_arrow(path)
    return _cluster_seed_map_fingerprint(arrow_cluster_seeds) == _cluster_seed_map_fingerprint(cluster_seeds_require)


def _unpack_incremental_seed_setup(
    seed_setup: Sequence[Any],
) -> tuple[
    Mapping[str, int | str],
    Mapping[str, int | str],
    Mapping[str, Sequence[str]],
    Mapping[str, Sequence[str]] | None,
]:
    if len(seed_setup) == 3:
        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse = seed_setup
        return cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, None
    if len(seed_setup) == 4:
        cluster_seeds_require, recluster_map, cluster_seeds_require_inverse, split_cluster_seeds_require_inverse = (
            seed_setup
        )
        return (
            cluster_seeds_require,
            recluster_map,
            cluster_seeds_require_inverse,
            split_cluster_seeds_require_inverse,
        )
    raise ValueError(f"incremental seed setup must have 3 or 4 entries, got {len(seed_setup)}")


def _finish_incremental_with_optional_split_inverse(
    clusterer: Any,
    unassigned_signature_ids: list[str],
    dataset: ANDData,
    linked_signature_clusters: Mapping[str, int | str],
    recluster_map: Mapping[str, int | str],
    cluster_seeds_require_inverse: Mapping[str, Sequence[str]],
    prevent_new_incompatibilities: bool,
    partial_supervision: Mapping[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    *,
    total_ram_bytes: int | None,
    arrow_paths: Mapping[str, Any] | None = None,
    split_cluster_seeds_require_inverse: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    method = clusterer._finish_incremental_with_seed_links
    kwargs: dict[str, Any] = {"total_ram_bytes": total_ram_bytes}
    if arrow_paths is not None:
        kwargs["arrow_paths"] = arrow_paths
    if split_cluster_seeds_require_inverse is not None:
        kwargs["split_cluster_seeds_require_inverse"] = split_cluster_seeds_require_inverse
    return method(
        unassigned_signature_ids,
        dataset,
        linked_signature_clusters,
        recluster_map,
        cluster_seeds_require_inverse,
        prevent_new_incompatibilities,
        partial_supervision,
        runtime_context,
        **kwargs,
    )


def promoted_incremental_component_sizes(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, int]:
    component_sizes: dict[str, int] = {}
    for cluster_id in cluster_seeds_require.values():
        component_key = str(cluster_id)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
    return component_sizes


def _signature_orcid(dataset: ANDData, signature_id: str) -> str | None:
    signatures = getattr(dataset, "signatures", None)
    if not isinstance(signatures, Mapping):
        return None
    signature = signatures.get(str(signature_id))
    if signature is None:
        return None
    value = getattr(signature, "author_info_orcid", None)
    return query_adapter_module.normalize_orcid(value)


def promoted_incremental_orcid_fanout_by_query(
    dataset: ANDData,
    query_signature_ids: Sequence[str],
    cluster_seeds_require: Mapping[str, int | str],
    *,
    orcid_enabled: bool,
) -> dict[str, tuple[int, int]]:
    """Return known ORCID return-all row/pair floors by query signature id."""

    if not orcid_enabled or not query_signature_ids or not cluster_seeds_require:
        return {}

    query_orcid_by_signature_id = {
        str(query_signature_id): query_orcid
        for query_signature_id in query_signature_ids
        if (query_orcid := _signature_orcid(dataset, str(query_signature_id))) is not None
    }
    if not query_orcid_by_signature_id:
        return {}

    query_orcids = set(query_orcid_by_signature_id.values())
    component_orcids: dict[str, set[str]] = {}
    component_sizes: dict[str, int] = {}
    for seed_signature_id, component in cluster_seeds_require.items():
        component_key = str(component)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
        seed_orcid = _signature_orcid(dataset, str(seed_signature_id))
        if seed_orcid in query_orcids:
            component_orcids.setdefault(component_key, set()).add(str(seed_orcid))

    if not component_orcids:
        return {}

    fanout_by_query: dict[str, tuple[int, int]] = {}
    for query_signature_id, query_orcid in query_orcid_by_signature_id.items():
        matching_components = [
            component_key for component_key, orcids in component_orcids.items() if query_orcid in orcids
        ]
        if matching_components:
            fanout_by_query[str(query_signature_id)] = (
                len(matching_components),
                sum(int(component_sizes[component_key]) for component_key in matching_components),
            )
    return fanout_by_query


def _top_k_candidate_floors(component_sizes: Mapping[str, int], retrieval_top_k: int) -> tuple[int, int]:
    candidate_rows = min(max(0, int(retrieval_top_k)), len(component_sizes))
    pairs = int(sum(sorted((max(0, int(size)) for size in component_sizes.values()), reverse=True)[:candidate_rows]))
    return candidate_rows, pairs


def _orcid_fanout_floor_estimates(
    fanout_by_query: Mapping[str, tuple[int, int]],
    query_signature_ids: Sequence[str],
) -> tuple[int | None, int | None]:
    rows = 0
    pairs = 0
    for signature_id in query_signature_ids:
        query_rows, query_pairs = fanout_by_query.get(str(signature_id), (0, 0))
        rows = max(rows, int(query_rows))
        pairs = max(pairs, int(query_pairs))
    return (rows if rows > 0 else None, pairs if pairs > 0 else None)


def _orcid_fanout_floor_totals(
    fanout_by_query: Mapping[str, tuple[int, int]],
    query_signature_ids: Sequence[str],
    *,
    base_candidate_rows_per_query: int,
    base_pairs_per_query: int,
) -> tuple[int | None, int | None]:
    if not query_signature_ids:
        return None, None
    row_total = 0
    pair_total = 0
    base_rows = max(0, int(base_candidate_rows_per_query))
    base_pairs = max(0, int(base_pairs_per_query))
    for signature_id in query_signature_ids:
        query_rows, query_pairs = fanout_by_query.get(str(signature_id), (0, 0))
        row_total += max(base_rows, int(query_rows))
        pair_total += max(base_pairs, int(query_pairs))
    base_row_total = base_rows * len(query_signature_ids)
    base_pair_total = base_pairs * len(query_signature_ids)
    return (
        row_total if row_total > base_row_total else None,
        pair_total if pair_total > base_pair_total else None,
    )


def compute_promoted_incremental_limits(
    *,
    query_count: int,
    component_sizes: Mapping[str, int],
    retrieval_top_k: int,
    memory_layout: _PromotedIncrementalMemoryLayout,
    total_ram_bytes: int | None,
    max_query_batch_size: int | None,
    observed_query_count: int = 0,
    observed_candidate_rows_per_query: int | None = None,
    observed_pairs_per_query: int | None = None,
    candidate_rows_per_query_floor: int | None = None,
    pairs_per_query_floor: int | None = None,
    candidate_rows_total_floor: int | None = None,
    pairs_total_floor: int | None = None,
) -> memory_budget.PromotedPhaseALimits:
    return memory_budget.compute_promoted_phase_a_limits(
        query_count=query_count,
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        final_matrix_feature_count=memory_layout.final_matrix_feature_count,
        pairwise_matrix_feature_count=memory_layout.pairwise_matrix_feature_count,
        aggregate_feature_count=memory_layout.aggregate_feature_count,
        total_ram_bytes=total_ram_bytes,
        max_query_batch_size=max_query_batch_size,
        observed_query_count=observed_query_count,
        observed_candidate_rows_per_query=observed_candidate_rows_per_query,
        observed_pairs_per_query=observed_pairs_per_query,
        candidate_rows_per_query_floor=candidate_rows_per_query_floor,
        pairs_per_query_floor=pairs_per_query_floor,
        candidate_rows_total_floor=candidate_rows_total_floor,
        pairs_total_floor=pairs_total_floor,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )


def raise_if_promoted_incremental_batch_over_budget(limits: memory_budget.PromotedPhaseALimits) -> None:
    if not bool(limits.single_query_exceeds_budget):
        return
    raise MemoryError(
        "Promoted incremental linker cannot fit a single query under the memory budget: "
        f"single_query_predicted_persistent_bytes={int(limits.single_query_predicted_persistent_bytes)} "
        f"stage_budget_bytes={int(limits.stage_budget_bytes)} "
        f"total_ram_bytes={int(limits.total_ram_bytes)} "
        f"current_rss_bytes={int(limits.current_rss_bytes)} "
        f"safety_margin_bytes={int(limits.safety_margin_bytes)}"
    )


def _memory_safe_promoted_query_batch(
    proposed_query_signature_ids: Sequence[str],
    *,
    orcid_fanout_by_query: Mapping[str, tuple[int, int]],
    component_sizes: Mapping[str, int],
    retrieval_top_k: int,
    memory_layout: _PromotedIncrementalMemoryLayout,
    total_ram_bytes: int,
    base_candidate_rows_per_query: int,
    base_pairs_per_query: int,
) -> tuple[list[str], memory_budget.PromotedPhaseALimits]:
    """Shrink a proposed query batch until it fits the latest measured RSS budget."""

    query_batch = [str(signature_id) for signature_id in proposed_query_signature_ids]
    if not query_batch:
        raise ValueError("Promoted incremental query batch cannot be empty")
    while True:
        batch_row_floor, batch_pair_floor = _orcid_fanout_floor_estimates(orcid_fanout_by_query, query_batch)
        batch_row_total_floor, batch_pair_total_floor = _orcid_fanout_floor_totals(
            orcid_fanout_by_query,
            query_batch,
            base_candidate_rows_per_query=base_candidate_rows_per_query,
            base_pairs_per_query=base_pairs_per_query,
        )
        limits = compute_promoted_incremental_limits(
            query_count=len(query_batch),
            component_sizes=component_sizes,
            retrieval_top_k=retrieval_top_k,
            memory_layout=memory_layout,
            total_ram_bytes=total_ram_bytes,
            max_query_batch_size=len(query_batch),
            candidate_rows_per_query_floor=batch_row_floor,
            pairs_per_query_floor=batch_pair_floor,
            candidate_rows_total_floor=batch_row_total_floor,
            pairs_total_floor=batch_pair_total_floor,
        )
        raise_if_promoted_incremental_batch_over_budget(limits)
        safe_size = max(1, min(len(query_batch), int(limits.query_batch_size or 1)))
        if safe_size == len(query_batch):
            return query_batch, limits
        query_batch = query_batch[:safe_size]


def promoted_incremental_observed_probe(
    telemetry: Mapping[str, int | float | str],
    fallback_query_count: int,
) -> tuple[int, int, int] | None:
    try:
        query_count = int(telemetry.get("query_count", fallback_query_count))
        candidate_row_count = int(telemetry.get("candidate_row_count", 0))
        pair_count = int(telemetry.get("pair_count", 0))
    except (TypeError, ValueError):
        return None
    if query_count <= 0 or (candidate_row_count <= 0 and pair_count <= 0):
        return None
    rows_per_query = int(math.ceil(float(candidate_row_count) / float(query_count)))
    pairs_per_query = int(math.ceil(float(pair_count) / float(query_count)))
    return query_count, rows_per_query, pairs_per_query


def merge_promoted_incremental_batch_telemetry(
    batch_telemetries: list[Mapping[str, int | float | str]],
    *,
    batch_sizes: list[int],
    configured_batch_size: int | None,
    initial_limits: memory_budget.PromotedPhaseALimits | None = None,
    final_limits: memory_budget.PromotedPhaseALimits | None = None,
    final_limits_history: Sequence[memory_budget.PromotedPhaseALimits] | None = None,
    calibration_applied: bool = False,
) -> dict[str, int | float | str]:
    merged: dict[str, int | float | str] = {}
    conflict_counts: dict[str, int] = {}
    for telemetry in batch_telemetries:
        for key, value in telemetry.items():
            merge_policy = _PROMOTED_INCREMENTAL_TELEMETRY_MERGE_POLICY.get(key, "constant")
            if merge_policy == "first":
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if merge_policy == "constant":
                if key not in merged:
                    merged[key] = value
                elif merged[key] != value:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if merge_policy == "sum_numeric" and isinstance(value, int | float) and not isinstance(value, bool):
                previous = merged.get(key, 0)
                if isinstance(previous, int | float) and not isinstance(previous, bool):
                    merged[key] = previous + value
                else:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if (
                merge_policy in {"min_numeric", "max_numeric"}
                and isinstance(value, int | float)
                and not isinstance(
                    value,
                    bool,
                )
            ):
                previous = merged.get(key)
                if previous is None:
                    merged[key] = value
                elif isinstance(previous, int | float) and not isinstance(previous, bool):
                    merged[key] = min(previous, value) if merge_policy == "min_numeric" else max(previous, value)
                else:
                    conflict_counts[key] = conflict_counts.get(key, 0) + 1
                continue
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                conflict_counts[key] = conflict_counts.get(key, 0) + 1

    merged["query_batch_count"] = len(batch_sizes)
    merged["query_batch_size_configured"] = int(configured_batch_size or 0)
    merged["query_batch_size_max"] = max(batch_sizes, default=0)
    merged["query_batch_size_min"] = min(batch_sizes, default=0)
    merged.setdefault("query_count", sum(batch_sizes))
    if initial_limits is not None:
        merged["memory_initial_query_batch_size"] = int(initial_limits.query_batch_size)
        merged["memory_initial_predicted_peak_delta_bytes"] = int(initial_limits.predicted_peak_delta_bytes)
        merged["memory_initial_predicted_peak_rss_bytes"] = int(initial_limits.predicted_peak_rss_bytes)
        merged["memory_initial_operational_estimate_source"] = str(initial_limits.operational_estimate_source)
    resolved_final_limits = list(final_limits_history or ())
    if not resolved_final_limits and final_limits is not None:
        resolved_final_limits.append(final_limits)
    if resolved_final_limits:
        merged["memory_final_query_batch_size"] = min(int(item.query_batch_size) for item in resolved_final_limits)
        merged["memory_final_predicted_peak_delta_bytes"] = max(
            int(item.predicted_peak_delta_bytes) for item in resolved_final_limits
        )
        merged["memory_final_predicted_peak_rss_bytes"] = max(
            int(item.predicted_peak_rss_bytes) for item in resolved_final_limits
        )
        estimate_sources = {str(item.operational_estimate_source) for item in resolved_final_limits}
        merged["memory_final_operational_estimate_source"] = (
            next(iter(estimate_sources)) if len(estimate_sources) == 1 else "__mixed__"
        )
    merged["memory_observed_calibration_applied"] = int(bool(calibration_applied))
    for key, count in conflict_counts.items():
        merged[f"{key}_batch_conflict_count"] = int(count)
    return merged


def _summarize_query_views(query_views: tuple[str, ...]) -> str:
    if not query_views:
        return "none"
    unique_views = set(query_views)
    if len(unique_views) == 1:
        return query_views[0]
    return "mixed"


def predict_incremental_promoted_linker_from_arrow_paths(
    clusterer: Any,
    block_signatures: list[str],
    dataset: ANDData,
    *,
    arrow_paths: Mapping[str, Any],
    artifact_dir: Path,
    artifact: artifact_module.IncrementalLinkingArtifact | None = None,
    prevent_new_incompatibilities: bool,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    batching_threshold: int | None,
) -> dict[str, Any]:
    """Run the promoted linker from Arrow artifacts, then finish residuals through the normal path."""

    if artifact is None:
        artifact = artifact_module.load_incremental_linking_artifact(artifact_dir)
    elif artifact.artifact_dir.resolve() != Path(artifact_dir).resolve():
        raise ValueError(
            "Loaded incremental linker artifact path does not match artifact_dir: "
            f"loaded={artifact.artifact_dir} requested={artifact_dir}"
        )
    resolved_total_ram_bytes, _ = memory_budget.resolve_total_ram_bytes(total_ram_bytes)
    clusterer_requires_specter = clusterer_uses_embedding_features(clusterer)
    clusterer_requires_name_counts = clusterer_uses_name_count_features(clusterer)
    base_arrow_path_payload = validate_arrow_prediction_artifacts(
        arrow_paths,
        require_specter=clusterer_requires_specter,
        require_name_counts_index=clusterer_requires_name_counts,
        require_cluster_seeds=False,
        context="Promoted incremental Arrow input",
        producer_hint=(
            "include valid base Arrow tables and any declared seed sidecars before promoted incremental " "seed setup"
        ),
    )
    require_arrow_name_counts_index_for_clusterer(clusterer, base_arrow_path_payload, context="Raw Arrow scoring")
    if hasattr(dataset, "name_counts_provenance"):
        require_dataset_name_counts_binding_for_clusterer(
            clusterer,
            dataset,
            context="Promoted incremental prediction",
        )
    (
        cluster_seeds_require,
        recluster_map,
        cluster_seeds_require_inverse,
        split_cluster_seeds_require_inverse,
    ) = _unpack_incremental_seed_setup(
        clusterer._build_incremental_seed_setup(
            dataset,
            partial_supervision,
            runtime_context,
            total_ram_bytes=resolved_total_ram_bytes,
            arrow_paths=base_arrow_path_payload,
        )
    )
    seed_setup_telemetry = dict(getattr(clusterer, "_last_incremental_seed_setup_telemetry", {}) or {})
    if len(cluster_seeds_require) == 0:
        raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

    unassigned_signature_ids = [
        str(signature_id) for signature_id in block_signatures if str(signature_id) not in cluster_seeds_require
    ]
    component_sizes = promoted_incremental_component_sizes(cluster_seeds_require)
    retrieval_top_k = int(artifact.metadata.retrieval_top_k)
    memory_layout = _promoted_incremental_memory_layout(clusterer, artifact)
    orcid_enabled = not bool(getattr(clusterer, "suppress_orcid", False))
    orcid_fanout_by_query = (
        promoted_incremental_orcid_fanout_by_query(
            dataset,
            unassigned_signature_ids,
            cluster_seeds_require,
            orcid_enabled=orcid_enabled,
        )
        if hasattr(dataset, "signatures")
        else {}
    )
    base_candidate_rows_per_query, base_pairs_per_query = _top_k_candidate_floors(component_sizes, retrieval_top_k)
    initial_row_floor, initial_pair_floor = _orcid_fanout_floor_estimates(
        orcid_fanout_by_query,
        unassigned_signature_ids,
    )
    initial_row_total_floor, initial_pair_total_floor = _orcid_fanout_floor_totals(
        orcid_fanout_by_query,
        unassigned_signature_ids,
        base_candidate_rows_per_query=base_candidate_rows_per_query,
        base_pairs_per_query=base_pairs_per_query,
    )
    initial_limits = compute_promoted_incremental_limits(
        query_count=len(unassigned_signature_ids),
        component_sizes=component_sizes,
        retrieval_top_k=retrieval_top_k,
        memory_layout=memory_layout,
        total_ram_bytes=resolved_total_ram_bytes,
        max_query_batch_size=batching_threshold,
        candidate_rows_per_query_floor=initial_row_floor,
        pairs_per_query_floor=initial_pair_floor,
        candidate_rows_total_floor=initial_row_total_floor,
        pairs_total_floor=initial_pair_total_floor,
    )
    resolved_total_ram_bytes = int(initial_limits.total_ram_bytes)
    raise_if_promoted_incremental_batch_over_budget(initial_limits)
    linked_signature_clusters: dict[str, int | str] = {}
    batch_telemetries: list[Mapping[str, int | float | str]] = []
    batch_sizes: list[int] = []
    query_batch_size = max(1, int(initial_limits.query_batch_size or 1))
    final_limits = initial_limits
    name_tuples = getattr(dataset, "name_tuples", "filtered")
    expected_normalization_version: str = (
        require_feature_contract_normalization_version(
            clusterer,
            context="promoted incremental raw Arrow scoring",
        )
        if unassigned_signature_ids
        else ""
    )
    request_disallows, dataset_disallows, arrow_disallows = _request_cluster_seed_disallows(
        dataset,
        base_arrow_path_payload,
    )
    query_disallow_partners = _query_disallow_partner_ids(
        unassigned_signature_ids,
        request_disallows,
        partial_supervision,
    )
    query_disallow_components_by_id = _query_disallow_components_by_id(query_disallow_partners)
    same_batch_finalized_disallow_query_ids: set[str] = set()
    same_batch_finalized_disallow_components: set[frozenset[str]] = set()
    initial_query_disallow_decisions: dict[str, _ScoredQueryDecision] = {}
    seed_arrow_start = time.perf_counter()
    seed_arrow_matches_cluster_seeds_require = _cluster_seeds_arrow_matches(
        base_arrow_path_payload.get("cluster_seeds"),
        cluster_seeds_require,
    )
    seed_arrow_reused_source = (
        str(seed_setup_telemetry.get("seed_setup_cluster_seeds_source", "")) == "arrow"
        and len(recluster_map) == 0
        and seed_arrow_matches_cluster_seeds_require
        and request_disallows == arrow_disallows
    )
    arrow_path_context: AbstractContextManager[Mapping[str, Any]]
    if seed_arrow_reused_source:
        arrow_path_context = nullcontext(dict(base_arrow_path_payload))
    else:
        arrow_path_context = temporary_arrow_paths_with_cluster_seeds(
            base_arrow_path_payload,
            cluster_seeds_require,
            prefix="s2and_arrow_incremental_cluster_seeds_",
            cluster_seeds_disallow=request_disallows,
        )
    with arrow_path_context as arrow_path_payload:
        if unassigned_signature_ids:
            arrow_path_payload = validate_arrow_prediction_artifacts(
                arrow_path_payload,
                require_specter=clusterer_requires_specter,
                require_name_counts_index=clusterer_requires_name_counts,
                require_cluster_seeds=True,
                context="Promoted incremental Arrow scoring",
                producer_hint=(
                    "include raw Arrow tables, raw-planner batch indexes, and the request-local "
                    "cluster_seeds sidecar before promoted incremental retrieval"
                ),
            )
        seed_arrow_assignment_seconds = time.perf_counter() - seed_arrow_start
        raw_window_plan_count = 0
        raw_window_plan_seconds = 0.0
        raw_window_plan_query_count = 0
        raw_window_featurizer_count = 0
        raw_window_featurizer_seconds = 0.0
        raw_window_featurizer_signature_count = 0
        raw_window_featurizer_reused_batch_count = 0
        raw_window_subset_seconds = 0.0
        raw_window_memory_replan_count = 0
        raw_window_plan_telemetry: dict[str, int | float | str] = {}
        final_limits_history: list[memory_budget.PromotedPhaseALimits] = []

        raw_window_planner_count = 0
        raw_window_planner_batch_plan_count = 0
        raw_window_planner_plan_call_count = 0
        raw_window_planner_plan_seconds = 0.0
        raw_request_planner: Any | None = None
        if unassigned_signature_ids:
            raw_window_start = time.perf_counter()
            raw_request_planner = feature_port._require_rust_runtime().RawBlockQueryCandidatePlanner.from_auto_queries(  # noqa: SLF001
                arrow_path_payload,
                top_k=retrieval_top_k,
                orcid_enabled=bool(orcid_enabled),
                num_threads=clusterer.n_jobs,
                max_exemplars=4,
            )
            raw_window_plan_seconds += time.perf_counter() - raw_window_start
            raw_window_plan_count += 1
            raw_window_plan_query_count += len(unassigned_signature_ids)
            raw_window_planner_count += 1
            _merge_raw_window_plan_telemetry(
                raw_window_plan_telemetry,
                _raw_window_plan_telemetry_fields({"telemetry": raw_request_planner.build_telemetry()}),
            )
            post_planner_batch, final_limits = _memory_safe_promoted_query_batch(
                unassigned_signature_ids[:query_batch_size],
                orcid_fanout_by_query=orcid_fanout_by_query,
                component_sizes=component_sizes,
                retrieval_top_k=retrieval_top_k,
                memory_layout=memory_layout,
                total_ram_bytes=resolved_total_ram_bytes,
                base_candidate_rows_per_query=base_candidate_rows_per_query,
                base_pairs_per_query=base_pairs_per_query,
            )
            query_batch_size = len(post_planner_batch)
            final_limits_history.append(final_limits)
        planned_query_batches = _disallow_aware_query_batches(
            [str(signature_id) for signature_id in unassigned_signature_ids],
            query_batch_size=query_batch_size,
            disallow_components_by_id=query_disallow_components_by_id,
        )
        featurizer_window_multiplier = _RAW_ARROW_PLAN_WINDOW_MULTIPLIER
        query_batch_plan_windows = _query_batch_plan_windows(
            planned_query_batches,
            plan_window_multiplier=featurizer_window_multiplier,
        )
        query_batch_plan_window_queue = deque(query_batch_plan_windows)
        featurizer_window_size = max(
            (
                sum(len(planned_batch) for planned_batch in query_batch_plan_window)
                for query_batch_plan_window in query_batch_plan_windows
            ),
            default=0,
        )

        def _refreshed_window_batch_size(query_batches: Sequence[Sequence[str]]) -> int:
            nonlocal final_limits
            limiting_batch_sizes: list[int] = []
            largest_batch_size = 0
            for planned_batch in query_batches:
                largest_batch_size = max(largest_batch_size, len(planned_batch))
                safe_batch, final_limits = _memory_safe_promoted_query_batch(
                    planned_batch,
                    orcid_fanout_by_query=orcid_fanout_by_query,
                    component_sizes=component_sizes,
                    retrieval_top_k=retrieval_top_k,
                    memory_layout=memory_layout,
                    total_ram_bytes=resolved_total_ram_bytes,
                    base_candidate_rows_per_query=base_candidate_rows_per_query,
                    base_pairs_per_query=base_pairs_per_query,
                )
                final_limits_history.append(final_limits)
                if len(safe_batch) < len(planned_batch):
                    limiting_batch_sizes.append(len(safe_batch))
            return min(limiting_batch_sizes, default=largest_batch_size)

        def _prepend_resized_windows(
            query_batches: Sequence[Sequence[str]],
            *,
            safe_batch_size: int,
        ) -> None:
            resized_windows = _resize_query_batch_plan_windows(
                query_batches,
                query_batch_size=safe_batch_size,
                plan_window_multiplier=featurizer_window_multiplier,
            )
            for resized_window in reversed(resized_windows):
                query_batch_plan_window_queue.appendleft(resized_window)

        while query_batch_plan_window_queue:
            query_batch_plan_window = query_batch_plan_window_queue.popleft()
            refreshed_batch_size = _refreshed_window_batch_size(query_batch_plan_window)
            if any(len(planned_batch) > refreshed_batch_size for planned_batch in query_batch_plan_window):
                _prepend_resized_windows(
                    query_batch_plan_window,
                    safe_batch_size=refreshed_batch_size,
                )
                raw_window_memory_replan_count += 1
                continue
            query_plan_window = [
                signature_id for planned_batch in query_batch_plan_window for signature_id in planned_batch
            ]
            if not query_plan_window:
                continue
            if raw_request_planner is None:
                raise RuntimeError("reusable raw Arrow planner was not initialized")
            raw_window_planner_plan_start = time.perf_counter()
            raw_candidate_plan = raw_request_planner.plan(list(query_plan_window))
            raw_window_planner_plan_call_count += 1
            raw_window_planner_plan_seconds += time.perf_counter() - raw_window_planner_plan_start
            refreshed_batch_size = _refreshed_window_batch_size(query_batch_plan_window)
            if any(len(planned_batch) > refreshed_batch_size for planned_batch in query_batch_plan_window):
                del raw_candidate_plan
                _prepend_resized_windows(
                    query_batch_plan_window,
                    safe_batch_size=refreshed_batch_size,
                )
                raw_window_memory_replan_count += 1
                continue
            raw_window_featurizer_start = time.perf_counter()
            signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
            signature_ids = signature_order.signature_ids
            raw_window_featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
                arrow_path_payload,
                expected_normalization_version=expected_normalization_version,
                signature_ids=signature_ids,
                name_tuples=name_tuples,
                load_name_counts=clusterer_uses_name_count_features(clusterer),
                preprocess=True,
                num_threads=clusterer.n_jobs,
            )
            raw_window_featurizer_seconds += time.perf_counter() - raw_window_featurizer_start
            raw_window_featurizer_count += 1
            raw_window_featurizer_signature_count += len(signature_ids)
            refreshed_batch_size = _refreshed_window_batch_size(query_batch_plan_window)
            if any(len(planned_batch) > refreshed_batch_size for planned_batch in query_batch_plan_window):
                del raw_window_featurizer
                del raw_candidate_plan
                _prepend_resized_windows(
                    query_batch_plan_window,
                    safe_batch_size=refreshed_batch_size,
                )
                raw_window_memory_replan_count += 1
                continue

            query_batch_queue = deque(list(planned_batch) for planned_batch in query_batch_plan_window)
            window_replanned = False
            while query_batch_queue:
                proposed_query_batch = query_batch_queue.popleft()
                query_batch, final_limits = _memory_safe_promoted_query_batch(
                    proposed_query_batch,
                    orcid_fanout_by_query=orcid_fanout_by_query,
                    component_sizes=component_sizes,
                    retrieval_top_k=retrieval_top_k,
                    memory_layout=memory_layout,
                    total_ram_bytes=resolved_total_ram_bytes,
                    base_candidate_rows_per_query=base_candidate_rows_per_query,
                    base_pairs_per_query=base_pairs_per_query,
                )
                final_limits_history.append(final_limits)
                if len(query_batch) < len(proposed_query_batch):
                    remaining_query_batches = [query_batch, proposed_query_batch[len(query_batch) :]]
                    remaining_query_batches.extend(query_batch_queue)
                    query_batch_queue.clear()
                    del raw_window_featurizer
                    del raw_candidate_plan
                    _prepend_resized_windows(
                        remaining_query_batches,
                        safe_batch_size=len(query_batch),
                    )
                    raw_window_memory_replan_count += 1
                    window_replanned = True
                    break
                batch_query_id_set = {str(signature_id) for signature_id in query_batch}
                complete_disallow_query_ids = {
                    signature_id
                    for signature_id in batch_query_id_set
                    if signature_id in query_disallow_components_by_id
                    and query_disallow_components_by_id[signature_id] <= batch_query_id_set
                }
                complete_disallow_partner_ids = {
                    signature_id: {
                        partner_id
                        for partner_id in query_disallow_partners[signature_id]
                        if partner_id in complete_disallow_query_ids
                    }
                    for signature_id in complete_disallow_query_ids
                } or None
                raw_window_subset_start = time.perf_counter()
                if len(query_batch) == len(query_plan_window):
                    batch_raw_candidate_plan = raw_candidate_plan
                else:
                    batch_raw_candidate_plan = runtime_module.subset_raw_candidate_plan_for_query_ids(
                        raw_candidate_plan,
                        query_batch,
                        zero_plan_timings=True,
                    )
                raw_window_planner_batch_plan_count += 1
                raw_window_subset_seconds += time.perf_counter() - raw_window_subset_start
                raw_window_featurizer_reused_batch_count += int(raw_window_featurizer is not None)
                result = runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
                    clusterer,
                    artifact,
                    arrow_paths=arrow_path_payload,
                    query_signature_ids=query_batch,
                    top_k=retrieval_top_k,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    n_jobs=clusterer.n_jobs,
                    total_ram_bytes=resolved_total_ram_bytes,
                    raw_candidate_plan=batch_raw_candidate_plan,
                    rust_featurizer=raw_window_featurizer,
                    raw_arrow_featurizer_source="window",
                    partial_supervision_seed_signature_to_component=cluster_seeds_require,
                    cluster_seed_disallow_partner_ids=complete_disallow_partner_ids,
                    cluster_seed_disallow_excluded_components=None,
                )
                if query_disallow_partners:
                    scored_batch = _scored_query_decisions_from_result(
                        result,
                        signature_ids_by_index=raw_window_featurizer.signature_ids(),
                        expected_query_signature_ids=query_batch,
                    )
                    for signature_id, scored in scored_batch.items():
                        if signature_id in complete_disallow_query_ids:
                            same_batch_finalized_disallow_query_ids.add(signature_id)
                            same_batch_finalized_disallow_components.add(query_disallow_components_by_id[signature_id])
                            if scored.decision.action == "link" and scored.decision.component_key is not None:
                                linked_signature_clusters[signature_id] = str(scored.decision.component_key)
                        elif signature_id in query_disallow_partners:
                            initial_query_disallow_decisions[signature_id] = scored
                        elif scored.decision.action == "link" and scored.decision.component_key is not None:
                            linked_signature_clusters[signature_id] = str(scored.decision.component_key)
                else:
                    linked_signature_clusters.update(dict(result.linked_signature_clusters))
                batch_telemetries.append(dict(result.telemetry))
                batch_sizes.append(len(query_batch))
                del result
                del batch_raw_candidate_plan
            if not window_replanned:
                del raw_window_featurizer
                del raw_candidate_plan

        global_disallow_telemetry: dict[str, int | float | str] = {}
        if query_disallow_partners:
            expected_disallow_query_ids = set(query_disallow_partners) - same_batch_finalized_disallow_query_ids
            observed_disallow_query_ids = set(initial_query_disallow_decisions)
            if observed_disallow_query_ids != expected_disallow_query_ids:
                raise ValueError(
                    "Promoted global query-disallow scoring did not produce exactly one initial decision per endpoint: "
                    f"missing={sorted(expected_disallow_query_ids - observed_disallow_query_ids)!r} "
                    f"extra={sorted(observed_disallow_query_ids - expected_disallow_query_ids)!r}"
                )
            global_rescore_candidate_rows = 0
            global_rescore_pairs = 0
            global_rescore_seconds = 0.0

            def _rescore_query_disallow_endpoint(
                signature_id: str,
                excluded_components: set[str],
            ) -> _ScoredQueryDecision:
                nonlocal final_limits
                nonlocal global_rescore_candidate_rows
                nonlocal global_rescore_pairs
                nonlocal global_rescore_seconds
                nonlocal raw_window_featurizer_count
                nonlocal raw_window_featurizer_seconds
                nonlocal raw_window_featurizer_signature_count
                nonlocal raw_window_planner_batch_plan_count
                nonlocal raw_window_planner_plan_call_count
                nonlocal raw_window_planner_plan_seconds

                rescore_started = time.perf_counter()
                if raw_request_planner is None:
                    raise RuntimeError("reusable raw Arrow planner was not initialized")
                plan_started = time.perf_counter()
                raw_rescore_plan = raw_request_planner.plan([signature_id])
                raw_window_planner_plan_call_count += 1
                raw_window_planner_batch_plan_count += 1
                raw_window_planner_plan_seconds += time.perf_counter() - plan_started
                signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_rescore_plan)
                featurizer_started = time.perf_counter()
                rescore_featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
                    arrow_path_payload,
                    expected_normalization_version=expected_normalization_version,
                    signature_ids=signature_order.signature_ids,
                    name_tuples=name_tuples,
                    load_name_counts=clusterer_uses_name_count_features(clusterer),
                    preprocess=True,
                    num_threads=clusterer.n_jobs,
                )
                raw_window_featurizer_seconds += time.perf_counter() - featurizer_started
                raw_window_featurizer_count += 1
                raw_window_featurizer_signature_count += len(signature_order.signature_ids)
                _, final_limits = _memory_safe_promoted_query_batch(
                    [signature_id],
                    orcid_fanout_by_query=orcid_fanout_by_query,
                    component_sizes=component_sizes,
                    retrieval_top_k=retrieval_top_k,
                    memory_layout=memory_layout,
                    total_ram_bytes=resolved_total_ram_bytes,
                    base_candidate_rows_per_query=base_candidate_rows_per_query,
                    base_pairs_per_query=base_pairs_per_query,
                )
                final_limits_history.append(final_limits)
                rescore_result = runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
                    clusterer,
                    artifact,
                    arrow_paths=arrow_path_payload,
                    query_signature_ids=[signature_id],
                    top_k=retrieval_top_k,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    n_jobs=clusterer.n_jobs,
                    total_ram_bytes=resolved_total_ram_bytes,
                    raw_candidate_plan=raw_rescore_plan,
                    rust_featurizer=rescore_featurizer,
                    raw_arrow_featurizer_source="window",
                    partial_supervision_seed_signature_to_component=cluster_seeds_require,
                    cluster_seed_disallow_partner_ids=None,
                    cluster_seed_disallow_excluded_components={signature_id: excluded_components},
                )
                global_rescore_candidate_rows += int(rescore_result.telemetry.get("candidate_row_count", 0))
                global_rescore_pairs += int(rescore_result.telemetry.get("pair_count", 0))
                global_rescore_seconds += time.perf_counter() - rescore_started
                return _scored_query_decisions_from_result(
                    rescore_result,
                    signature_ids_by_index=rescore_featurizer.signature_ids(),
                    expected_query_signature_ids=[signature_id],
                )[signature_id]

            globally_linked, global_disallow_counts = _resolve_query_disallows_globally(
                initial_query_disallow_decisions,
                query_disallow_partners,
                rescore=_rescore_query_disallow_endpoint,
            )
            linked_signature_clusters.update(globally_linked)
            global_disallow_telemetry = {
                **global_disallow_counts,
                "global_query_disallow_rescore_candidate_row_count": int(global_rescore_candidate_rows),
                "global_query_disallow_rescore_pair_count": int(global_rescore_pairs),
                "global_query_disallow_rescore_seconds": float(global_rescore_seconds),
                "global_query_disallow_same_batch_component_count": int(len(same_batch_finalized_disallow_components)),
            }

        merged_telemetry = merge_promoted_incremental_batch_telemetry(
            batch_telemetries,
            batch_sizes=batch_sizes,
            configured_batch_size=batching_threshold,
            initial_limits=initial_limits,
            final_limits=final_limits,
            final_limits_history=final_limits_history,
        )
        merged_telemetry.update(global_disallow_telemetry)
        merged_telemetry["query_disallow_endpoint_count"] = int(len(query_disallow_partners))
        merged_telemetry["query_disallow_conflicted_query_count"] = int(
            merged_telemetry.get("cluster_seed_disallow_same_batch_conflict_count", 0)
        ) + int(merged_telemetry.get("global_query_disallow_conflict_count", 0))
        merged_telemetry["query_disallow_reassigned_link_count"] = int(
            merged_telemetry.get("cluster_seed_disallow_same_batch_reassigned_link_count", 0)
        ) + int(merged_telemetry.get("global_query_disallow_reassigned_link_count", 0))
        merged_telemetry["query_disallow_demoted_abstain_count"] = int(
            merged_telemetry.get("cluster_seed_disallow_same_batch_demoted_abstain_count", 0)
        ) + int(merged_telemetry.get("global_query_disallow_demoted_abstain_count", 0))
        merged_telemetry["seed_signature_count"] = int(len(cluster_seeds_require))
        merged_telemetry["seed_component_count"] = int(len(cluster_seeds_require_inverse))
        merged_telemetry["raw_arrow_seed_signature_count"] = int(len(cluster_seeds_require))
        merged_telemetry["raw_arrow_seed_component_count"] = int(len(cluster_seeds_require_inverse))
        finish_start = time.perf_counter()
        predicted_clusters = _finish_incremental_with_optional_split_inverse(
            clusterer,
            unassigned_signature_ids,
            dataset,
            linked_signature_clusters,
            recluster_map,
            cluster_seeds_require_inverse,
            prevent_new_incompatibilities,
            partial_supervision,
            runtime_context,
            total_ram_bytes=resolved_total_ram_bytes,
            arrow_paths=arrow_path_payload,
            split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
        )
        finish_seconds = time.perf_counter() - finish_start
        residual_phase_b_telemetry = dict(getattr(clusterer, "_last_incremental_residual_phase_b_telemetry", {}) or {})
        residual_count = sum(
            1 for signature_id in unassigned_signature_ids if signature_id not in linked_signature_clusters
        )
        phase_b_required_bytes = residual_count * (residual_count - 1) // 2 * 8
        payload = {
            "clusters": predicted_clusters,
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": phase_b_required_bytes,
            "phase_b_required_bytes": phase_b_required_bytes,
            "phase_b_residual_count": residual_count,
        }
        payload["incremental_linker_artifact_path"] = str(artifact_dir)
        payload["incremental_linker_query_view"] = "raw_arrow"
        payload["incremental_linker_telemetry"] = {
            **seed_setup_telemetry,
            **merged_telemetry,
            **residual_phase_b_telemetry,
            **raw_window_plan_telemetry,
            "incremental_finish_seconds": float(finish_seconds),
            "seed_arrow_assignment_seconds": float(seed_arrow_assignment_seconds),
            "seed_arrow_reused_source": int(bool(seed_arrow_reused_source)),
            "seed_arrow_dataset_disallow_count": int(len(dataset_disallows)),
            "seed_arrow_disallow_count": int(len(request_disallows)),
            "arrow_promoted_incremental": 1,
            "arrow_path_count": len(arrow_path_payload),
            "raw_arrow_window_plan_count": int(raw_window_plan_count),
            "raw_arrow_window_plan_query_count": int(raw_window_plan_query_count),
            "raw_arrow_window_plan_enabled": int(raw_request_planner is not None),
            "raw_arrow_window_plan_size": int(featurizer_window_size),
            "raw_arrow_window_plan_multiplier": int(featurizer_window_multiplier),
            "raw_arrow_window_plan_seconds": float(raw_window_plan_seconds),
            "raw_arrow_window_featurizer_query_window_size": int(featurizer_window_size),
            "raw_arrow_window_featurizer_window_multiplier": int(featurizer_window_multiplier),
            "raw_arrow_window_featurizer_count": int(raw_window_featurizer_count),
            "raw_arrow_window_featurizer_signature_count": int(raw_window_featurizer_signature_count),
            "raw_arrow_window_featurizer_reused_batch_count": int(raw_window_featurizer_reused_batch_count),
            "raw_arrow_window_memory_replan_count": int(raw_window_memory_replan_count),
            "raw_arrow_window_featurizer_seconds": float(raw_window_featurizer_seconds),
            "raw_arrow_window_subset_seconds": float(raw_window_subset_seconds),
            "raw_arrow_window_planner_count": int(raw_window_planner_count),
            "raw_arrow_window_planner_batch_plan_count": int(raw_window_planner_batch_plan_count),
            "raw_arrow_window_planner_plan_call_count": int(raw_window_planner_plan_call_count),
            "raw_arrow_window_planner_plan_seconds": float(raw_window_planner_plan_seconds),
            "raw_arrow_reusable_planner_enabled": int(raw_request_planner is not None),
        }
        return payload
