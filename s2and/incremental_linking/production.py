"""Production orchestration for the promoted incremental linker."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import s2and.incremental_linking.artifact as artifact_module
import s2and.incremental_linking.query_adapter as query_adapter_module
import s2and.incremental_linking.runtime as runtime_module
from s2and import feature_port, memory_budget
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.incremental_linking.completion import ResidualClusterer, complete_incremental_prediction
from s2and.incremental_linking.completion_metadata import (
    SignatureFirstNames,
    SignatureOrcids,
    log_rejected_links,
    name_tuples_for_incremental_rules,
)
from s2and.incremental_linking.feature_block import (
    normalize_cluster_seed_disallow_pairs,
    temporary_cluster_seed_sidecars,
)
from s2and.incremental_linking.policy import (
    promoted_linker_orcid_force_link_enabled,
    request_cluster_seed_disallow_parts,
)
from s2and.incremental_linking.retrieval import RawArrowPlanBundle
from s2and.incremental_linking.seed_assignment import apply_seed_links
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext

logger = logging.getLogger("s2and")

_RAW_ARROW_REUSE_BATCHES = 4
_QUERY_DISALLOW_FEATURIZER_CACHE_MAX_ENTRIES = 2

_PROMOTED_INCREMENTAL_SUM_TELEMETRY_FIELDS = frozenset(
    {
        "abstain_count",
        "candidate_row_count",
        "cluster_seed_disallow_excluded_query_count",
        "cluster_seed_disallow_excluded_row_count",
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


@dataclass(frozen=True)
class _DirectArrowIncrementalDataset:
    """Canonical request state for direct Arrow incremental prediction."""

    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]]
    cluster_seeds_require: Mapping[str, int | str]
    cluster_seeds_disallow: Iterable[tuple[str, str]]
    altered_cluster_signatures: Sequence[str] | None
    max_seed_cluster_id: int
    signatures: Mapping[str, Any]


@dataclass(frozen=True)
class _QueryDisallowRescoreContext:
    """Stable inputs needed to rescore one cross-batch disallow endpoint."""

    clusterer: Any
    artifact: artifact_module.IncrementalLinkingArtifact
    planner: Any
    arrow_dataset: ArrowDataset
    cluster_seeds_path: Path
    cluster_seed_disallows_path: Path | None
    name_tuples: Any
    retrieval_top_k: int
    partial_supervision: dict[tuple[str, str], int | float]
    runtime_context: RuntimeContext
    total_ram_bytes: int
    cluster_seeds_require: Mapping[str, int | str]
    cluster_seed_representative_by_component: Mapping[str, str]
    orcid_fanout_by_query: Mapping[str, tuple[int, int]]
    component_sizes: memory_budget.PromotedComponentSizeSummary
    memory_layout: _PromotedIncrementalMemoryLayout
    base_candidate_rows_per_query: int
    base_pairs_per_query: int
    featurizer_cache: _QueryDisallowFeaturizerCache


@dataclass
class _QueryDisallowFeaturizerCache:
    """Bounded LRU of request-local native featurizers for disallow rescoring."""

    max_entries: int = _QUERY_DISALLOW_FEATURIZER_CACHE_MAX_ENTRIES
    entries: list[tuple[frozenset[str], Any]] = field(default_factory=list)

    def make_room(self) -> None:
        """Evict least-recent entries before constructing one replacement."""

        while len(self.entries) >= max(1, int(self.max_entries)):
            self.entries.pop(0)

    def retain(self, featurizer: Any, signature_ids: Sequence[str]) -> None:
        """Retain a featurizer without exceeding the fixed request-local bound."""

        self.entries = [(covered, item) for covered, item in self.entries if item is not featurizer]
        self.make_room()
        self.entries.append((frozenset(str(signature_id) for signature_id in signature_ids), featurizer))

    def covering(self, signature_ids: Sequence[str]) -> Any | None:
        """Return the retained featurizer when it covers every requested signature."""

        required = {str(signature_id) for signature_id in signature_ids}
        for index in range(len(self.entries) - 1, -1, -1):
            covered, featurizer = self.entries[index]
            if required.issubset(covered):
                self.entries.append(self.entries.pop(index))
                return featurizer
        return None

    def clear(self) -> None:
        """Release every retained native featurizer before residual Phase B."""

        self.entries.clear()


@dataclass(frozen=True)
class _QueryDisallowRescoreOutcome:
    """Decision, memory limit, and telemetry from one endpoint rescore."""

    decision: _ScoredQueryDecision
    limits: memory_budget.PromotedPhaseALimits
    telemetry: Mapping[str, int | float]


def _promoted_incremental_memory_layout(
    clusterer: Any,
    artifact: artifact_module.IncrementalLinkingArtifact,
) -> _PromotedIncrementalMemoryLayout:
    pairwise_matrix_feature_count, aggregate_feature_count = runtime_module.promoted_pairwise_memory_feature_counts(
        clusterer
    )
    return _PromotedIncrementalMemoryLayout(
        final_matrix_feature_count=len(artifact.feature_columns),
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
    sibling_components: Mapping[str, tuple[str, ...]] | None = None,
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
            sibling
            for partner_id in disallow_partners.get(signature_id, ())
            if (partner_decision := finalized.get(str(partner_id))) is not None
            and partner_decision.decision.action == "link"
            and partner_decision.decision.component_key is not None
            for sibling in (sibling_components or {}).get(
                str(partner_decision.decision.component_key), (str(partner_decision.decision.component_key),)
            )
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
    require_counts = result.decision_row_signals.get("constraint_require_count")
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


def _cluster_seed_representatives(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, str]:
    """Return one deterministic seed from each definitive post-split component."""

    representatives: dict[str, str] = {}
    for seed_id, component_key in cluster_seeds_require.items():
        normalized_seed_id = str(seed_id)
        normalized_component_key = str(component_key)
        current = representatives.get(normalized_component_key)
        if current is None or normalized_seed_id < current:
            representatives[normalized_component_key] = normalized_seed_id
    return representatives


def _restored_profile_siblings(recluster_map: Mapping[str, int | str]) -> dict[str, tuple[str, ...]]:
    """Index natural components that restore to the same claimed profile."""

    components_by_profile: dict[str, list[str]] = {}
    for component, profile in recluster_map.items():
        components_by_profile.setdefault(str(profile), []).append(str(component))
    siblings: dict[str, tuple[str, ...]] = {}
    for components in components_by_profile.values():
        group = tuple(sorted(components))
        siblings.update((component, group) for component in group)
    return siblings


def _expand_restored_profile_disallows(
    disallows: set[tuple[str, str]],
    *,
    query_signature_ids: Sequence[str],
    cluster_seeds_require: Mapping[str, int | str],
    sibling_components: Mapping[str, tuple[str, ...]],
    seed_representatives: Mapping[str, str],
) -> set[tuple[str, str]]:
    """Exclude every sibling split before retrieval can select a restored profile."""

    if not sibling_components or not disallows:
        return disallows
    query_ids = set(query_signature_ids)
    expanded = set(disallows)
    for left, right in disallows:
        if left in query_ids and right in cluster_seeds_require:
            query, seed = left, right
        elif right in query_ids and left in cluster_seeds_require:
            query, seed = right, left
        else:
            continue
        component = str(cluster_seeds_require[seed])
        for sibling in sibling_components.get(component, ()):
            representative = seed_representatives[sibling]
            expanded.add((query, representative) if query < representative else (representative, query))
    return expanded


def _query_seed_disallows_for_rescore(
    context: _QueryDisallowRescoreContext,
    signature_id: str,
    excluded_components: set[str],
) -> set[tuple[str, str]]:
    """Return singleton-query disallows that the planner can apply before top-k."""

    query_id = str(signature_id)
    seed_representative_by_component = context.cluster_seed_representative_by_component
    normalized_excluded_components = {str(component_key) for component_key in excluded_components}
    missing_components = {
        component_key
        for component_key in normalized_excluded_components
        if component_key not in seed_representative_by_component
    }
    if missing_components:
        raise ValueError(
            "Cross-batch query disallow references unknown seed components: "
            f"signature_id={query_id!r} component_keys={sorted(missing_components)!r}"
        )

    return {
        (query_id, seed_representative_by_component[component_key]) for component_key in normalized_excluded_components
    }


def _plan_query_with_excluded_components(
    context: _QueryDisallowRescoreContext,
    signature_id: str,
    excluded_components: set[str],
) -> RawArrowPlanBundle:
    """Build a singleton plan whose exclusions are applied before retrieval top-k."""

    additional_disallows = sorted(_query_seed_disallows_for_rescore(context, signature_id, excluded_components))
    return RawArrowPlanBundle.from_native_mapping(context.planner.plan([str(signature_id)], additional_disallows))


def _rescore_query_disallow_endpoint(
    context: _QueryDisallowRescoreContext,
    signature_id: str,
    excluded_components: set[str],
) -> _QueryDisallowRescoreOutcome:
    """Plan, featurize, score, and extract one cross-batch disallow endpoint."""

    rescore_started = time.perf_counter()
    plan_started = time.perf_counter()
    raw_plan_bundle = _plan_query_with_excluded_components(context, signature_id, excluded_components)
    planner_seconds = time.perf_counter() - plan_started

    required_signature_ids = raw_plan_bundle.signature_order.signature_ids
    featurizer = context.featurizer_cache.covering(required_signature_ids)
    featurizer_seconds = 0.0
    featurizer_build_count = 0
    featurizer_built_signature_count = 0
    if featurizer is None:
        context.featurizer_cache.make_room()
        featurizer_started = time.perf_counter()
        featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
            context.arrow_dataset,
            signature_ids=required_signature_ids,
            name_tuples=context.name_tuples,
            preprocess=True,
            num_threads=context.clusterer.n_jobs,
            cluster_seeds_path=context.cluster_seeds_path,
            cluster_seed_disallows_path=context.cluster_seed_disallows_path,
        )
        featurizer_seconds = time.perf_counter() - featurizer_started
        featurizer_build_count = 1
        featurizer_built_signature_count = len(required_signature_ids)
        context.featurizer_cache.retain(featurizer, required_signature_ids)
    _, limits = _memory_safe_promoted_query_batch(
        [signature_id],
        orcid_fanout_by_query=context.orcid_fanout_by_query,
        component_sizes=context.component_sizes,
        retrieval_top_k=context.retrieval_top_k,
        memory_layout=context.memory_layout,
        total_ram_bytes=context.total_ram_bytes,
        base_candidate_rows_per_query=context.base_candidate_rows_per_query,
        base_pairs_per_query=context.base_pairs_per_query,
        retrieval_payload_resident=True,
    )
    result = runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
        context.clusterer,
        context.artifact,
        arrow_dataset=context.arrow_dataset,
        query_signature_ids=[signature_id],
        top_k=context.retrieval_top_k,
        partial_supervision=context.partial_supervision,
        runtime_context=context.runtime_context,
        n_jobs=context.clusterer.n_jobs,
        total_ram_bytes=context.total_ram_bytes,
        raw_plan_bundle=raw_plan_bundle,
        rust_featurizer=featurizer,
        partial_supervision_seed_signature_to_component=context.cluster_seeds_require,
        cluster_seed_disallow_excluded_components={signature_id: excluded_components},
    )
    decision = _scored_query_decisions_from_result(
        result,
        signature_ids_by_index=featurizer.signature_ids(),
        expected_query_signature_ids=[signature_id],
    )[signature_id]
    return _QueryDisallowRescoreOutcome(
        decision=decision,
        limits=limits,
        telemetry={
            "candidate_rows": int(result.telemetry.get("candidate_row_count", 0)),
            "pairs": int(result.telemetry.get("pair_count", 0)),
            "seconds": time.perf_counter() - rescore_started,
            "planner_seconds": planner_seconds,
            "featurizer_seconds": featurizer_seconds,
            "featurizer_build_count": featurizer_build_count,
            "featurizer_built_signatures": featurizer_built_signature_count,
        },
    )


def _request_cluster_seed_disallows(
    dataset: _DirectArrowIncrementalDataset,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    request_disallows, dataset_disallows, _ = request_cluster_seed_disallow_parts(dataset, ())
    return request_disallows, dataset_disallows


def _plan_time_cluster_seed_disallows(
    request_disallows: set[tuple[str, str]],
    unassigned_signature_ids: Sequence[str],
) -> set[tuple[str, str]]:
    """Remove query-query pairs that cannot exclude a component before either query links."""

    query_ids = {str(signature_id) for signature_id in unassigned_signature_ids}
    return {
        (str(left), str(right))
        for left, right in request_disallows
        if not (str(left) in query_ids and str(right) in query_ids)
    }


def _partial_supervision_plan_disallows(
    partial_supervision: Mapping[tuple[str, str], int | float],
    *,
    query_signature_ids: Sequence[str],
    seed_signature_ids: Iterable[str],
) -> set[tuple[str, str]]:
    """Return explicit query-to-active-seed disallows for candidate planning."""

    query_ids = {str(signature_id) for signature_id in query_signature_ids}
    seed_ids = {str(signature_id) for signature_id in seed_signature_ids}
    pairs = []
    for (left, right), value in partial_supervision.items():
        if value != LARGE_DISTANCE:
            continue
        left_id = str(left)
        right_id = str(right)
        if (left_id in query_ids and right_id in seed_ids) or (right_id in query_ids and left_id in seed_ids):
            pairs.append((left_id, right_id))
    return set(normalize_cluster_seed_disallow_pairs(pairs))


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


def promoted_incremental_component_sizes(cluster_seeds_require: Mapping[str, int | str]) -> dict[str, int]:
    component_sizes: dict[str, int] = {}
    for cluster_id in cluster_seeds_require.values():
        component_key = str(cluster_id)
        component_sizes[component_key] = component_sizes.get(component_key, 0) + 1
    return component_sizes


def _signature_orcid(dataset: _DirectArrowIncrementalDataset, signature_id: str) -> str | None:
    signature = dataset.signatures.get(str(signature_id))
    if signature is None:
        return None
    value = getattr(signature, "author_info_orcid", None)
    return query_adapter_module.normalize_orcid(value)


def promoted_incremental_orcid_fanout_by_query(
    dataset: _DirectArrowIncrementalDataset,
    query_signature_ids: Sequence[str],
    cluster_seeds_require: Mapping[str, int | str],
    *,
    orcid_enabled: bool,
    component_sizes: Mapping[str, int] | None = None,
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
    resolved_component_sizes = component_sizes
    if resolved_component_sizes is None:
        resolved_component_sizes = promoted_incremental_component_sizes(cluster_seeds_require)
    for seed_signature_id, component in cluster_seeds_require.items():
        component_key = str(component)
        seed_orcid = _signature_orcid(dataset, str(seed_signature_id))
        if seed_orcid in query_orcids:
            component_orcids.setdefault(component_key, set()).add(str(seed_orcid))

    if not component_orcids:
        return {}

    fanout_by_orcid: dict[str, tuple[int, int]] = {}
    for component_key, orcids in component_orcids.items():
        component_size = int(resolved_component_sizes[component_key])
        for orcid in orcids:
            component_count, pair_count = fanout_by_orcid.get(orcid, (0, 0))
            fanout_by_orcid[orcid] = (component_count + 1, pair_count + component_size)

    return {
        query_signature_id: fanout_by_orcid[query_orcid]
        for query_signature_id, query_orcid in query_orcid_by_signature_id.items()
        if query_orcid in fanout_by_orcid
    }


def _top_k_candidate_floors(
    component_sizes: Mapping[str, int] | memory_budget.PromotedComponentSizeSummary,
    retrieval_top_k: int,
) -> tuple[int, int]:
    summary = (
        component_sizes
        if isinstance(component_sizes, memory_budget.PromotedComponentSizeSummary)
        else memory_budget.summarize_promoted_component_sizes(component_sizes)
    )
    return summary.top_k_totals(retrieval_top_k)


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
    component_sizes: Mapping[str, int] | memory_budget.PromotedComponentSizeSummary,
    retrieval_top_k: int,
    memory_layout: _PromotedIncrementalMemoryLayout,
    total_ram_bytes: int | None,
    max_query_batch_size: int | None,
    candidate_rows_per_query_floor: int | None = None,
    pairs_per_query_floor: int | None = None,
    candidate_rows_total_floor: int | None = None,
    pairs_total_floor: int | None = None,
    retrieval_payload_resident: bool = False,
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
        candidate_rows_per_query_floor=candidate_rows_per_query_floor,
        pairs_per_query_floor=pairs_per_query_floor,
        candidate_rows_total_floor=candidate_rows_total_floor,
        pairs_total_floor=pairs_total_floor,
        retrieval_payload_resident=retrieval_payload_resident,
        detect_cgroup_fn=memory_budget.detect_cgroup_total_ram_bytes_best_effort,
        detect_total_fn=memory_budget.detect_total_ram_bytes_best_effort,
        current_rss_fn=memory_budget.current_rss_bytes_best_effort,
    )


def _memory_safe_promoted_query_batch(
    proposed_query_signature_ids: Sequence[str],
    *,
    orcid_fanout_by_query: Mapping[str, tuple[int, int]],
    component_sizes: Mapping[str, int] | memory_budget.PromotedComponentSizeSummary,
    retrieval_top_k: int,
    memory_layout: _PromotedIncrementalMemoryLayout,
    total_ram_bytes: int,
    base_candidate_rows_per_query: int,
    base_pairs_per_query: int,
    retrieval_payload_resident: bool = False,
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
            retrieval_payload_resident=retrieval_payload_resident,
        )
        safe_size = max(1, min(len(query_batch), int(limits.query_batch_size or 1)))
        if safe_size == len(query_batch):
            return query_batch, limits
        query_batch = query_batch[:safe_size]


def merge_promoted_incremental_batch_telemetry(
    batch_telemetries: list[Mapping[str, int | float | str]],
    *,
    batch_sizes: list[int],
    configured_batch_size: int | None,
    initial_limits: memory_budget.PromotedPhaseALimits | None = None,
    final_limits: memory_budget.PromotedPhaseALimits | None = None,
    final_limits_history: Sequence[memory_budget.PromotedPhaseALimits] | None = None,
) -> dict[str, int | float | str]:
    merged: dict[str, int | float | str] = {}
    conflict_counts: dict[str, int] = {}
    for telemetry in batch_telemetries:
        for key, value in telemetry.items():
            merge_policy = _PROMOTED_INCREMENTAL_TELEMETRY_MERGE_POLICY.get(key)
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
    for key, count in conflict_counts.items():
        merged[f"{key}_batch_conflict_count"] = int(count)
    return merged


def predict_incremental_promoted_linker_from_arrow(
    clusterer: Any,
    block_signatures: list[str],
    dataset: _DirectArrowIncrementalDataset,
    *,
    cluster_residuals: ResidualClusterer,
    arrow_dataset: ArrowDataset,
    artifact: artifact_module.IncrementalLinkingArtifact,
    prevent_new_incompatibilities: bool,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    batching_threshold: int | None,
) -> dict[str, Any]:
    """Run the promoted linker from one open Arrow dataset."""

    resolved_total_ram_bytes, _ = memory_budget.resolve_total_ram_bytes(total_ram_bytes)
    prediction_state = PredictionState(
        cluster_seeds_require=dict(dataset.cluster_seeds_require),
        cluster_seeds_disallow=set(dataset.cluster_seeds_disallow),
        altered_cluster_signatures=list(dataset.altered_cluster_signatures or ()),
    )
    request_disallows, dataset_disallows = _request_cluster_seed_disallows(dataset)
    (
        cluster_seeds_require,
        recluster_map,
        cluster_seeds_require_inverse,
        split_cluster_seeds_require_inverse,
    ) = clusterer._build_incremental_seed_setup(
        dataset,
        partial_supervision,
        runtime_context,
        total_ram_bytes=resolved_total_ram_bytes,
        arrow_dataset=arrow_dataset,
        cluster_seed_disallows=request_disallows,
        prediction_state=prediction_state,
    )
    seed_setup_telemetry = dict(prediction_state.telemetry.get("incremental_seed_setup", {}))
    if len(cluster_seeds_require) == 0:
        raise ValueError("Promoted incremental linker mode requires at least one seed cluster")

    unassigned_signature_ids = [
        str(signature_id) for signature_id in block_signatures if str(signature_id) not in cluster_seeds_require
    ]
    component_members_for_sizes = (
        cluster_seeds_require_inverse
        if split_cluster_seeds_require_inverse is None
        else split_cluster_seeds_require_inverse
    )
    component_sizes = {
        str(component_id): len(signature_ids) for component_id, signature_ids in component_members_for_sizes.items()
    }
    component_size_summary = memory_budget.summarize_promoted_component_sizes(component_sizes)
    retrieval_top_k = int(artifact.retrieval_top_k)
    memory_layout = _promoted_incremental_memory_layout(clusterer, artifact)
    orcid_enabled = promoted_linker_orcid_force_link_enabled(
        suppress_orcid=bool(getattr(clusterer, "suppress_orcid", False))
    )
    orcid_fanout_by_query = promoted_incremental_orcid_fanout_by_query(
        dataset,
        unassigned_signature_ids,
        cluster_seeds_require,
        orcid_enabled=orcid_enabled,
        component_sizes=component_sizes,
    )
    base_candidate_rows_per_query, base_pairs_per_query = _top_k_candidate_floors(
        component_size_summary,
        retrieval_top_k,
    )
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
        component_sizes=component_size_summary,
        retrieval_top_k=retrieval_top_k,
        memory_layout=memory_layout,
        total_ram_bytes=resolved_total_ram_bytes,
        max_query_batch_size=batching_threshold,
        candidate_rows_per_query_floor=initial_row_floor,
        pairs_per_query_floor=initial_pair_floor,
        candidate_rows_total_floor=initial_row_total_floor,
        pairs_total_floor=initial_pair_total_floor,
    )
    linked_signature_clusters: dict[str, int | str] = {}
    batch_telemetries: list[Mapping[str, int | float | str]] = []
    batch_sizes: list[int] = []
    query_batch_size = max(1, int(initial_limits.query_batch_size or 1))
    final_limits = initial_limits
    name_tuples = dataset.name_tuples
    query_disallow_partners = _query_disallow_partner_ids(
        unassigned_signature_ids,
        request_disallows,
        partial_supervision,
    )
    planner_disallows = _plan_time_cluster_seed_disallows(request_disallows, unassigned_signature_ids)
    planner_disallows.update(
        _partial_supervision_plan_disallows(
            partial_supervision,
            query_signature_ids=unassigned_signature_ids,
            seed_signature_ids=cluster_seeds_require,
        )
    )
    sibling_components = _restored_profile_siblings(recluster_map)
    seed_representatives = {}
    if query_disallow_partners:
        seed_representatives = _cluster_seed_representatives(cluster_seeds_require)
    elif sibling_components and planner_disallows:
        seed_representatives = {
            component: min(str(member) for member in component_members_for_sizes[component])
            for component in sibling_components
        }
    planner_disallows = _expand_restored_profile_disallows(
        planner_disallows,
        query_signature_ids=unassigned_signature_ids,
        cluster_seeds_require=cluster_seeds_require,
        sibling_components=sibling_components,
        seed_representatives=seed_representatives,
    )
    initial_query_disallow_decisions: dict[str, _ScoredQueryDecision] = {}
    seed_arrow_start = time.perf_counter()
    with temporary_cluster_seed_sidecars(
        cluster_seeds_require,
        prefix="s2and_arrow_incremental_cluster_seeds_",
        cluster_seeds_disallow=planner_disallows,
    ) as request_sidecars:
        cluster_seeds_path = Path(request_sidecars["cluster_seeds"])
        cluster_seed_disallows_path = (
            Path(request_sidecars["cluster_seed_disallows"]) if "cluster_seed_disallows" in request_sidecars else None
        )
        seed_arrow_assignment_seconds = time.perf_counter() - seed_arrow_start
        raw_planner_build_seconds = 0.0
        raw_batch_plan_count = 0
        raw_batch_plan_query_count = 0
        raw_batch_plan_seconds = 0.0
        raw_batch_featurizer_count = 0
        raw_batch_featurizer_seconds = 0.0
        raw_batch_featurizer_signature_count = 0
        raw_batch_memory_replan_count = 0
        final_limits_history: list[memory_budget.PromotedPhaseALimits] = []

        raw_request_planner: Any | None = None
        if unassigned_signature_ids:
            planner_build_start = time.perf_counter()
            raw_request_planner = feature_port._require_rust_runtime().RawBlockQueryCandidatePlanner.from_auto_queries(  # noqa: SLF001
                arrow_dataset.native,
                str(cluster_seeds_path),
                top_k=retrieval_top_k,
                cluster_seed_disallows_path=(
                    None if cluster_seed_disallows_path is None else str(cluster_seed_disallows_path)
                ),
                orcid_enabled=bool(orcid_enabled),
                num_threads=clusterer.n_jobs,
                max_exemplars=4,
            )
            raw_planner_build_seconds = time.perf_counter() - planner_build_start
        scoring_batch_size = query_batch_size
        rescore_featurizer_cache = _QueryDisallowFeaturizerCache()

        def _refresh_batch(
            proposed_batch: Sequence[str],
            *,
            retrieval_payload_resident: bool,
        ) -> list[str]:
            nonlocal final_limits
            safe_batch, final_limits = _memory_safe_promoted_query_batch(
                proposed_batch,
                orcid_fanout_by_query=orcid_fanout_by_query,
                component_sizes=component_size_summary,
                retrieval_top_k=retrieval_top_k,
                memory_layout=memory_layout,
                total_ram_bytes=resolved_total_ram_bytes,
                base_candidate_rows_per_query=base_candidate_rows_per_query,
                base_pairs_per_query=base_pairs_per_query,
                retrieval_payload_resident=retrieval_payload_resident,
            )
            final_limits_history.append(final_limits)
            return safe_batch

        query_offset = 0
        while query_offset < len(unassigned_signature_ids):
            proposed_window = unassigned_signature_ids[
                query_offset : query_offset + _RAW_ARROW_REUSE_BATCHES * scoring_batch_size
            ]
            query_window = _refresh_batch(proposed_window, retrieval_payload_resident=False)
            if len(query_window) < len(proposed_window):
                raw_batch_memory_replan_count += 1
            if len(query_window) < scoring_batch_size:
                scoring_batch_size = len(query_window)
            if raw_request_planner is None:
                raise RuntimeError("reusable raw Arrow planner was not initialized")
            planner_start = time.perf_counter()
            raw_candidate_plan = raw_request_planner.plan(query_window)
            raw_batch_plan_seconds += time.perf_counter() - planner_start
            raw_batch_plan_count += 1
            raw_batch_plan_query_count += len(query_window)
            first_batch = query_window[:scoring_batch_size]
            refreshed_batch = _refresh_batch(first_batch, retrieval_payload_resident=True)
            if len(refreshed_batch) < len(first_batch):
                scoring_batch_size = len(refreshed_batch)
                raw_batch_memory_replan_count += 1
            raw_plan_bundle = RawArrowPlanBundle.from_native_mapping(raw_candidate_plan)
            del raw_candidate_plan
            featurizer_start = time.perf_counter()
            signature_ids = raw_plan_bundle.signature_order.signature_ids
            raw_batch_featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
                arrow_dataset,
                signature_ids=signature_ids,
                name_tuples=name_tuples,
                preprocess=True,
                num_threads=clusterer.n_jobs,
                cluster_seeds_path=cluster_seeds_path,
                cluster_seed_disallows_path=cluster_seed_disallows_path,
            )
            raw_batch_featurizer_seconds += time.perf_counter() - featurizer_start
            raw_batch_featurizer_count += 1
            raw_batch_featurizer_signature_count += len(signature_ids)
            window_offset = 0
            while window_offset < len(query_window):
                proposed_query_batch = query_window[window_offset : window_offset + scoring_batch_size]
                query_batch = _refresh_batch(proposed_query_batch, retrieval_payload_resident=True)
                if len(query_batch) < len(proposed_query_batch):
                    scoring_batch_size = len(query_batch)
                    raw_batch_memory_replan_count += 1
                    continue
                batch_plan_bundle = (
                    raw_plan_bundle
                    if len(query_batch) == len(query_window)
                    else raw_plan_bundle.contiguous_query_slice(
                        window_offset,
                        window_offset + len(query_batch),
                    )
                )
                result = runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
                    clusterer,
                    artifact,
                    arrow_dataset=arrow_dataset,
                    query_signature_ids=query_batch,
                    top_k=retrieval_top_k,
                    partial_supervision=partial_supervision,
                    runtime_context=runtime_context,
                    n_jobs=clusterer.n_jobs,
                    total_ram_bytes=resolved_total_ram_bytes,
                    raw_plan_bundle=batch_plan_bundle,
                    rust_featurizer=raw_batch_featurizer,
                    partial_supervision_seed_signature_to_component=cluster_seeds_require,
                    cluster_seed_disallow_excluded_components=None,
                )
                if query_disallow_partners:
                    scored_batch = _scored_query_decisions_from_result(
                        result,
                        signature_ids_by_index=raw_batch_featurizer.signature_ids(),
                        expected_query_signature_ids=query_batch,
                    )
                    for signature_id, scored in scored_batch.items():
                        if signature_id in query_disallow_partners:
                            initial_query_disallow_decisions[signature_id] = scored
                        elif scored.decision.action == "link" and scored.decision.component_key is not None:
                            linked_signature_clusters[signature_id] = str(scored.decision.component_key)
                else:
                    linked_signature_clusters.update(dict(result.linked_signature_clusters))
                batch_telemetries.append(dict(result.telemetry))
                batch_sizes.append(len(query_batch))
                window_offset += len(query_batch)
                del result
                del batch_plan_bundle
            if any(signature_id in query_disallow_partners for signature_id in query_window):
                rescore_featurizer_cache.retain(raw_batch_featurizer, signature_ids)
            query_offset += len(query_window)
            del raw_batch_featurizer
            del raw_plan_bundle

        global_disallow_telemetry: dict[str, int | float | str] = {}
        if query_disallow_partners:
            expected_disallow_query_ids = set(query_disallow_partners)
            observed_disallow_query_ids = set(initial_query_disallow_decisions)
            if observed_disallow_query_ids != expected_disallow_query_ids:
                raise ValueError(
                    "Promoted global query-disallow scoring did not produce exactly one initial decision per endpoint: "
                    f"missing={sorted(expected_disallow_query_ids - observed_disallow_query_ids)!r} "
                    f"extra={sorted(observed_disallow_query_ids - expected_disallow_query_ids)!r}"
                )
            if raw_request_planner is None:
                raise RuntimeError("reusable raw Arrow planner was not initialized")
            rescore_context = _QueryDisallowRescoreContext(
                clusterer=clusterer,
                artifact=artifact,
                planner=raw_request_planner,
                arrow_dataset=arrow_dataset,
                cluster_seeds_path=cluster_seeds_path,
                cluster_seed_disallows_path=cluster_seed_disallows_path,
                name_tuples=name_tuples,
                retrieval_top_k=retrieval_top_k,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=resolved_total_ram_bytes,
                cluster_seeds_require=cluster_seeds_require,
                cluster_seed_representative_by_component=seed_representatives,
                orcid_fanout_by_query=orcid_fanout_by_query,
                component_sizes=component_size_summary,
                memory_layout=memory_layout,
                base_candidate_rows_per_query=base_candidate_rows_per_query,
                base_pairs_per_query=base_pairs_per_query,
                featurizer_cache=rescore_featurizer_cache,
            )
            rescore_outcomes: list[_QueryDisallowRescoreOutcome] = []

            def _record_rescore(
                signature_id: str,
                excluded_components: set[str],
            ) -> _ScoredQueryDecision:
                outcome = _rescore_query_disallow_endpoint(
                    rescore_context,
                    signature_id,
                    excluded_components,
                )
                rescore_outcomes.append(outcome)
                return outcome.decision

            globally_linked, global_disallow_counts = _resolve_query_disallows_globally(
                initial_query_disallow_decisions,
                query_disallow_partners,
                rescore=_record_rescore,
                sibling_components=sibling_components,
            )
            linked_signature_clusters.update(globally_linked)
            final_limits_history.extend(outcome.limits for outcome in rescore_outcomes)
            rescore_totals = (
                {
                    key: sum(outcome.telemetry[key] for outcome in rescore_outcomes)
                    for key in rescore_outcomes[0].telemetry
                }
                if rescore_outcomes
                else {}
            )
            raw_batch_featurizer_count += int(rescore_totals.get("featurizer_build_count", 0))
            raw_batch_featurizer_seconds += float(rescore_totals.get("featurizer_seconds", 0))
            raw_batch_featurizer_signature_count += int(rescore_totals.get("featurizer_built_signatures", 0))
            raw_batch_plan_count += len(rescore_outcomes)
            raw_batch_plan_query_count += len(rescore_outcomes)
            raw_batch_plan_seconds += float(rescore_totals.get("planner_seconds", 0))
            global_disallow_telemetry = {
                **global_disallow_counts,
                "global_query_disallow_rescore_candidate_row_count": int(rescore_totals.get("candidate_rows", 0)),
                "global_query_disallow_rescore_pair_count": int(rescore_totals.get("pairs", 0)),
                "global_query_disallow_rescore_seconds": float(rescore_totals.get("seconds", 0)),
            }
            del _record_rescore
            del rescore_context
            del rescore_outcomes

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
            merged_telemetry.get("global_query_disallow_conflict_count", 0)
        )
        merged_telemetry["query_disallow_reassigned_link_count"] = int(
            merged_telemetry.get("global_query_disallow_reassigned_link_count", 0)
        )
        merged_telemetry["query_disallow_demoted_abstain_count"] = int(
            merged_telemetry.get("global_query_disallow_demoted_abstain_count", 0)
        )
        merged_telemetry["seed_signature_count"] = int(len(cluster_seeds_require))
        merged_telemetry["seed_component_count"] = int(len(cluster_seeds_require_inverse))
        merged_telemetry["raw_arrow_seed_signature_count"] = int(len(cluster_seeds_require))
        merged_telemetry["raw_arrow_seed_component_count"] = int(len(cluster_seeds_require_inverse))
        raw_arrow_reusable_planner_enabled = int(raw_request_planner is not None)
        rescore_featurizer_cache.clear()
        del rescore_featurizer_cache
        del raw_request_planner
        finish_start = time.perf_counter()
        logger.info("Assigning unassigned signatures for incremental clustering")
        check_names = prevent_new_incompatibilities and bool(recluster_map)
        assignment = apply_seed_links(
            unassigned_signature_ids=unassigned_signature_ids,
            linked_signature_to_cluster=linked_signature_clusters,
            recluster_map=recluster_map,
            cluster_seeds_require_inverse=cluster_seeds_require_inverse,
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            first_names=SignatureFirstNames(dataset.signatures) if check_names else {},
            name_tuples=name_tuples_for_incremental_rules(dataset.name_tuples) if check_names else frozenset(),
            split_cluster_seeds_require_inverse=split_cluster_seeds_require_inverse,
        )
        log_rejected_links(assignment.rejected_signature_ids, dataset.signatures)
        predicted_clusters = complete_incremental_prediction(
            assignment,
            first_names=SignatureFirstNames(dataset.signatures),
            orcids=SignatureOrcids(dataset.signatures),
            partial_supervision=partial_supervision,
            use_default_constraints_as_supervision=bool(
                getattr(clusterer, "use_default_constraints_as_supervision", True)
            ),
            suppress_orcid=bool(getattr(clusterer, "suppress_orcid", False)),
            start_cluster_id=int(dataset.max_seed_cluster_id or 0),
            prediction_state=prediction_state,
            cluster_residuals=cluster_residuals,
        )
        finish_seconds = time.perf_counter() - finish_start
        residual_phase_b_telemetry = dict(prediction_state.telemetry.get("incremental_residual_phase_b", {}))
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
        payload["incremental_linker_artifact_path"] = str(artifact.artifact_dir)
        payload["incremental_linker_query_view"] = "raw_arrow"
        payload["incremental_linker_telemetry"] = {
            **seed_setup_telemetry,
            **merged_telemetry,
            **residual_phase_b_telemetry,
            "incremental_finish_seconds": float(finish_seconds),
            "seed_arrow_assignment_seconds": float(seed_arrow_assignment_seconds),
            "seed_arrow_dataset_disallow_count": int(len(dataset_disallows)),
            "seed_arrow_disallow_count": int(len(request_disallows)),
            "arrow_promoted_incremental": 1,
            "raw_arrow_planner_build_seconds": float(raw_planner_build_seconds),
            "raw_arrow_batch_plan_count": int(raw_batch_plan_count),
            "raw_arrow_batch_plan_query_count": int(raw_batch_plan_query_count),
            "raw_arrow_batch_plan_seconds": float(raw_batch_plan_seconds),
            "raw_arrow_batch_featurizer_count": int(raw_batch_featurizer_count),
            "raw_arrow_batch_featurizer_signature_count": int(raw_batch_featurizer_signature_count),
            "raw_arrow_batch_featurizer_seconds": float(raw_batch_featurizer_seconds),
            "raw_arrow_batch_memory_replan_count": int(raw_batch_memory_replan_count),
            "raw_arrow_reusable_planner_enabled": raw_arrow_reusable_planner_enabled,
        }
        return payload
