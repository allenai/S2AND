"""Complete incremental assignments using an explicit residual clustering operation."""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from s2and.consts import LARGE_DISTANCE
from s2and.incremental_linking.seed_assignment import SeedLinkAssignment
from s2and.prediction_state import PredictionState

ResidualClusterer: TypeAlias = Callable[[list[str]], dict[str, list[str]]]
logger = logging.getLogger("s2and")


def first_initials(first: str) -> frozenset[str]:
    """Return initials of all space- or hyphen-separated first-name tokens."""
    tokens = [token for token in first.replace("-", " ").split() if token]
    if not tokens and first:
        stripped = first.strip()
        if stripped:
            tokens = [stripped]
    return frozenset(token[0] for token in tokens if token)


def residual_first_initial_groups(
    signature_ids: Sequence[str],
    *,
    first_names: Mapping[str, str],
    orcids: Mapping[str, str | None],
    partial_supervision: Mapping[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    suppress_orcid: bool,
) -> list[list[str]]:
    """Split residuals only where hard first-initial constraints make it safe.

    Names and ORCIDs are prepared rule metadata. Shared initials, unsuppressed
    shared ORCIDs, and supervision below LARGE_DISTANCE keep signatures together.
    Missing or empty names disable splitting. Input and group order are preserved.
    """
    residual_signature_ids = [str(signature_id) for signature_id in signature_ids]
    if len(residual_signature_ids) <= 1 or not use_default_constraints_as_supervision:
        return [residual_signature_ids]

    initials: dict[str, frozenset[str]] = {}
    for signature_id in residual_signature_ids:
        first = first_names.get(signature_id)
        if not first:
            return [residual_signature_ids]
        signature_initials = first_initials(first)
        if not signature_initials:
            return [residual_signature_ids]
        initials[signature_id] = signature_initials
    if len(set().union(*initials.values())) <= 1:
        return [residual_signature_ids]

    parent = {signature_id: signature_id for signature_id in residual_signature_ids}

    def find(signature_id: str) -> str:
        root = signature_id
        while parent[root] != root:
            root = parent[root]
        while parent[signature_id] != signature_id:
            next_signature_id = parent[signature_id]
            parent[signature_id] = root
            signature_id = next_signature_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    initial_representatives: dict[str, str] = {}
    for signature_id in residual_signature_ids:
        for initial in initials[signature_id]:
            representative = initial_representatives.setdefault(initial, signature_id)
            union(representative, signature_id)

    if not suppress_orcid:
        orcid_representatives: dict[str, str] = {}
        for signature_id in residual_signature_ids:
            orcid = orcids.get(signature_id)
            if orcid is None:
                continue
            representative = orcid_representatives.setdefault(orcid, signature_id)
            union(representative, signature_id)

    residual_signature_id_set = set(residual_signature_ids)
    for (left, right), value in partial_supervision.items():
        left_id = str(left)
        right_id = str(right)
        if left_id not in residual_signature_id_set or right_id not in residual_signature_id_set:
            continue
        if float(value) < LARGE_DISTANCE:
            union(left_id, right_id)

    groups_by_root: dict[str, list[str]] = defaultdict(list)
    for signature_id in residual_signature_ids:
        groups_by_root[find(signature_id)].append(signature_id)
    groups = list(groups_by_root.values())
    if len(groups) <= 1:
        return [residual_signature_ids]
    return groups


def next_unused_cluster_id(pred_clusters: Mapping[str, object], start: int) -> int:
    """Find the first unused numeric output cluster ID at or above start."""
    cluster_id = int(start)
    while str(cluster_id) in pred_clusters:
        cluster_id += 1
    return cluster_id


def complete_incremental_prediction(
    assignment: SeedLinkAssignment,
    *,
    first_names: Mapping[str, str],
    orcids: Mapping[str, str | None],
    partial_supervision: Mapping[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    suppress_orcid: bool,
    start_cluster_id: int,
    prediction_state: PredictionState,
    cluster_residuals: ResidualClusterer,
) -> dict[str, list[str]]:
    """Cluster residual groups and allocate IDs without accessing a model or dataset.

    Args:
        assignment: Seed assignments and residual signatures to complete.
        first_names: Prepared first names for safe residual grouping.
        orcids: Normalized ORCIDs for preserving required cross-initial pairs.
        partial_supervision: Pair distances that can prevent residual splitting.
        use_default_constraints_as_supervision: Whether first-initial rules apply.
        suppress_orcid: Whether to ignore shared ORCIDs when grouping.
        start_cluster_id: First candidate ID for new residual clusters.
        prediction_state: Request-owned destination for completion telemetry.
        cluster_residuals: Backend operation for a group of two or more signatures.

    Returns:
        Owned seed and residual cluster lists. Callback failures propagate unchanged.
    """
    pred_clusters = {cluster_id: list(members) for cluster_id, members in assignment.clusters.items()}
    residual_signature_ids = assignment.residual_signature_ids
    if residual_signature_ids:
        logger.info("Clustering together the still unassigned signatures")
    residual_groups = (
        residual_first_initial_groups(
            residual_signature_ids,
            first_names=first_names,
            orcids=orcids,
            partial_supervision=partial_supervision,
            use_default_constraints_as_supervision=use_default_constraints_as_supervision,
            suppress_orcid=suppress_orcid,
        )
        if residual_signature_ids
        else []
    )
    pair_count_before = len(residual_signature_ids) * (len(residual_signature_ids) - 1) // 2
    pair_count_after = sum(len(group) * (len(group) - 1) // 2 for group in residual_groups)
    prediction_state.telemetry["incremental_residual_phase_b"] = {
        "residual_phase_b_signature_count": len(residual_signature_ids),
        "residual_phase_b_group_count": len(residual_groups),
        "residual_phase_b_pair_count_before": pair_count_before,
        "residual_phase_b_pair_count_after": pair_count_after,
        "residual_phase_b_pair_count_saved": pair_count_before - pair_count_after,
    }
    if residual_signature_ids:
        logger.info(
            "Telemetry stage: stage=incremental_residual_phase_b residual_signatures=%d groups=%d "
            "pairs_before=%d pairs_after=%d",
            len(residual_signature_ids),
            len(residual_groups),
            pair_count_before,
            pair_count_after,
        )
    new_cluster_id = next_unused_cluster_id(pred_clusters, start_cluster_id)
    for residual_group in residual_groups:
        reclustered_output = (
            {"singleton": [residual_group[0]]} if len(residual_group) == 1 else cluster_residuals(residual_group)
        )
        for new_cluster in reclustered_output.values():
            new_cluster_id = next_unused_cluster_id(pred_clusters, new_cluster_id)
            pred_clusters[str(new_cluster_id)] = list(new_cluster)
            new_cluster_id += 1
    logger.info("Done. Returning incrementally predicted clusters")
    return pred_clusters
