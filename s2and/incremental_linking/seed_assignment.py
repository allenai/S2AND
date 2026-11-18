"""Apply incremental seed-link decisions without model or dataset dependencies."""

from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence, Set
from dataclasses import dataclass

from s2and.text import first_names_name_compatible


@dataclass(frozen=True)
class LinkPriority:
    """Order of one accepted link when new links compete for a cluster."""

    forced: bool
    score: float


def reject_conflicting_new_links(
    linked_signature_to_cluster: Mapping[str, int | str],
    *,
    priorities: Mapping[str, LinkPriority],
    recluster_map: Mapping[int | str, int | str],
    conflict: Callable[[str, str], bool],
    conflict_key: Callable[[str], Hashable] | None = None,
) -> tuple[dict[str, int | str], list[str]]:
    """Reject new links that conflict with a link accepted into the same cluster.

    Links that require or ORCID evidence forced come first and are always kept.
    The other links follow by score descending, then by signature ID. A link is
    rejected when ``conflict`` is true for it and a link already accepted into
    the same restored cluster.

    Args:
        linked_signature_to_cluster: Linker decisions, before restoring profiles.
        priorities: Forced flag and score of each linked signature.
        recluster_map: Split-cluster IDs mapped to their original profile IDs.
        conflict: Whether two linked signatures cannot share a cluster.
        conflict_key: Optional key with equal ``conflict`` results for equal
            keys and no conflict inside one key. Each cluster then keeps one
            accepted signature per key, so the check scales with distinct keys.

    Returns:
        The accepted links and the rejected signature IDs in priority order.
    """
    missing = sorted(set(linked_signature_to_cluster) - set(priorities))
    if missing:
        raise ValueError(f"Linked signatures have no link priority: {missing[:10]!r}")

    def priority(signature_id: str) -> tuple[bool, float, str]:
        item = priorities[signature_id]
        return not item.forced, -item.score, signature_id

    key_of = conflict_key if conflict_key is not None else (lambda signature_id: signature_id)
    accepted: dict[str, int | str] = {}
    # One accepted representative per key and cluster.
    accepted_by_cluster: defaultdict[str, dict[Hashable, str]] = defaultdict(dict)
    rejected: list[str] = []
    for signature_id in sorted(linked_signature_to_cluster, key=priority):
        cluster_id = linked_signature_to_cluster[signature_id]
        representatives = accepted_by_cluster[str(recluster_map.get(cluster_id, cluster_id))]
        key = key_of(signature_id)
        if not priorities[signature_id].forced and any(
            conflict(signature_id, member) for member_key, member in representatives.items() if member_key != key
        ):
            rejected.append(signature_id)
            continue
        representatives.setdefault(key, signature_id)
        accepted[signature_id] = cluster_id
    return accepted, rejected


def _seed_name_compatible(
    seed_first: str,
    seed_cjk: str | None,
    query_first: str,
    query_cjk: str | None,
    name_tuples: Set[tuple[str, str]],
) -> bool:
    # Complete CJK names with one script tag compare whole, as in canonical_names_conflict.
    if seed_cjk and query_cjk and seed_cjk[0] == query_cjk[0]:
        return seed_cjk == query_cjk
    return first_names_name_compatible(seed_first, query_first, name_tuples)


@dataclass
class SeedLinkAssignment:
    """Owned clusters, residual signatures, and rejected links for diagnostics."""

    clusters: dict[str, list[str]]
    residual_signature_ids: list[str]
    rejected_signature_ids: list[str]


def apply_seed_links(
    *,
    unassigned_signature_ids: Sequence[str],
    linked_signature_to_cluster: Mapping[str, int | str],
    recluster_map: Mapping[int | str, int | str],
    cluster_seeds_require_inverse: Mapping[int | str, Sequence[str]],
    prevent_new_incompatibilities: bool,
    first_names: Mapping[str, str],
    name_tuples: Set[tuple[str, str]],
    split_cluster_seeds_require_inverse: Mapping[int | str, Sequence[str]] | None = None,
    forced_signature_ids: Set[str] = frozenset(),
    cjk_given_names: Mapping[str, str | None] | None = None,
) -> SeedLinkAssignment:
    """Apply links, restoring altered profiles and checking name compatibility.

    Args:
        unassigned_signature_ids: Query signatures in assignment order.
        linked_signature_to_cluster: Accepted linker decisions, before restoring profiles.
        recluster_map: Split-cluster IDs mapped to their original profile IDs.
        cluster_seeds_require_inverse: Original seed members in output order.
        prevent_new_incompatibilities: Whether restored profiles require compatible names.
        first_names: Prepared first names, accessed only for compatibility checks.
        name_tuples: Resolved canonical name-alias pairs.
        split_cluster_seeds_require_inverse: Members of individual altered-profile splits.
        forced_signature_ids: Queries whose link a require constraint forced. The name
            check skips them, because the constraint already decided the link.
        cjk_given_names: Whole CJK given names (``cjk_given_name_for_constraint``). Two
            complete names in one script are compatible only when equal, as in the hard
            name rule; other pairs use the first-name rule.

    Returns:
        New cluster lists, queries needing residual clustering, and queries whose
        links were rejected by name compatibility. Input collections are unchanged.
    """
    compatibility_members = (
        split_cluster_seeds_require_inverse
        if split_cluster_seeds_require_inverse is not None
        else cluster_seeds_require_inverse
    )
    clusters: defaultdict[str, list[str]] = defaultdict(list)
    residual_signature_ids: list[str] = []
    rejected_signature_ids: list[str] = []
    for cluster_id, members in cluster_seeds_require_inverse.items():
        for signature_id in members:
            clusters[str(cluster_id)].append(signature_id)

    for signature_id in unassigned_signature_ids:
        if signature_id not in linked_signature_to_cluster:
            residual_signature_ids.append(signature_id)
            continue
        linked_cluster_id = linked_signature_to_cluster[signature_id]
        restored_cluster_id = recluster_map.get(linked_cluster_id, linked_cluster_id)
        if (
            prevent_new_incompatibilities
            and linked_cluster_id in recluster_map
            and signature_id not in forced_signature_ids
        ):
            members = compatibility_members.get(
                linked_cluster_id, cluster_seeds_require_inverse.get(restored_cluster_id, [])
            )
            cjk_names: Mapping[str, str | None] = {} if cjk_given_names is None else cjk_given_names
            seed_names = {(first_names[member], cjk_names.get(member)) for member in members}
            seed_names = {(first, cjk) for first, cjk in seed_names if len(first) > 1}
            if seed_names:
                query_first, query_cjk = first_names[signature_id], cjk_names.get(signature_id)
                if not any(
                    _seed_name_compatible(first, cjk, query_first, query_cjk, name_tuples) for first, cjk in seed_names
                ):
                    residual_signature_ids.append(signature_id)
                    rejected_signature_ids.append(signature_id)
                    continue
        clusters[str(restored_cluster_id)].append(signature_id)

    return SeedLinkAssignment(dict(clusters), residual_signature_ids, rejected_signature_ids)
