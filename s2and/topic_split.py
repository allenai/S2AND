"""Post-clustering many-way topic split corroborated by shared coauthors.

Same-name authors who work in different fields (a father/son pair, or several
unrelated researchers who share a name) are frequently over-merged into one
cluster: their papers' SPECTER title/abstract embeddings sit close in *absolute*
cosine (most academic titles are cosine ~0.7-0.9 apart), so the pairwise model
treats them as compatible and average-linkage fuses them. A single cluster can
contain many such distinct identities (e.g. a "N. Harel" block with a dozen
unrelated people).

The signal that *does* separate them is relative/structural: each person's
papers form a tight embedding sub-cloud with a distinct centroid. This pass:

1. Over-segments each predicted cluster into fine topic sub-groups with
   agglomerative clustering on the (cosine) embedding space -- no fixed number
   of groups, so it adapts to however many identities are present.
2. Re-merges sub-groups that share at least ``remerge_min_shared_coauthors``
   coauthors. A genuine single author carries recurring coauthors across their
   own topics, so their sub-groups re-merge; distinct same-name people have
   disjoint coauthor networks and stay separate.

Full coauthor names are used for the re-merge test (not initial+surname blocks),
because common-name blocks like "j lee" collide across distinct people and would
spuriously glue them back together. Affiliation is deliberately not used --
shared geography tokens (e.g. "seattle", "washington") cause false overlap
between distinct people in the same city. The pass only ever splits within an
existing cluster, never merges across clusters, so it cannot reduce recall on
already-separated clusters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from s2and.data import ANDData

logger = logging.getLogger("s2and")


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _coauthors(dataset: "ANDData", signature_id: str) -> set[str]:
    names = dataset.signatures[signature_id].author_info_coauthors
    return names if names else set()


def _embeddings_for_signatures(
    signature_ids: list[str], dataset: "ANDData"
) -> tuple[list[str], np.ndarray | None]:
    """Return (signatures-with-embeddings, row-normalized embedding matrix)."""
    if dataset.specter_embeddings is None:
        return [], None
    have: list[str] = []
    vectors: list[np.ndarray] = []
    for signature_id in signature_ids:
        paper_id = str(dataset.signatures[signature_id].paper_id)
        vector = dataset.specter_embeddings.get(paper_id)
        if vector is None:
            continue
        array = np.asarray(vector, dtype=float)
        if array.size == 0 or np.all(array == 0):
            continue
        have.append(signature_id)
        vectors.append(array)
    if not vectors:
        return [], None
    return have, _normalize_rows(np.vstack(vectors))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def _split_one_cluster(
    signature_ids: list[str],
    dataset: "ANDData",
    *,
    min_cluster: int,
    min_subgroup: int,
    embed_distance_threshold: float,
    remerge_min_shared_coauthors: int,
) -> list[list[str]]:
    """Over-segment one cluster's embeddings, then re-merge by shared coauthors.

    Returns a list of subclusters; a single-element list means "no split".
    """
    from sklearn.cluster import AgglomerativeClustering

    embedded, matrix = _embeddings_for_signatures(signature_ids, dataset)
    if matrix is None or len(embedded) < min_cluster:
        return [signature_ids]

    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=embed_distance_threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(matrix)
    group_keys = sorted(set(int(label) for label in labels))
    if len(group_keys) < 2:
        return [signature_ids]

    # Assign every signature (including embedding-less ones) to a sub-group.
    centroids = {k: _normalize_rows(matrix[labels == k].mean(axis=0)[None, :])[0] for k in group_keys}
    position = {sid: i for i, sid in enumerate(embedded)}
    groups: dict[int, list[str]] = {k: [] for k in group_keys}
    for signature_id in signature_ids:
        if signature_id in position:
            groups[int(labels[position[signature_id]])].append(signature_id)
            continue
        paper_id = str(dataset.signatures[signature_id].paper_id)
        vector = dataset.specter_embeddings.get(paper_id) if dataset.specter_embeddings else None
        if vector is None:
            groups[group_keys[0]].append(signature_id)
        else:
            unit = _normalize_rows(np.asarray(vector, dtype=float)[None, :])[0]
            groups[max(group_keys, key=lambda k: float(np.dot(unit, centroids[k])))].append(signature_id)

    # Re-merge sub-groups that share enough coauthors (full names).
    coauthors = {
        k: (set().union(*[_coauthors(dataset, s) for s in groups[k]]) if groups[k] else set()) for k in group_keys
    }
    index = {k: i for i, k in enumerate(group_keys)}
    union_find = _UnionFind(len(group_keys))
    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            shared = len(coauthors[group_keys[i]] & coauthors[group_keys[j]])
            if shared >= remerge_min_shared_coauthors:
                union_find.union(i, j)

    components: dict[int, list[str]] = {}
    for k in group_keys:
        root = union_find.find(index[k])
        components.setdefault(root, []).extend(groups[k])
    parts = [members for members in components.values() if members]

    # Fold components below min_subgroup into the largest, so the split only
    # produces clusters of meaningful size.
    largest = max(parts, key=len)
    kept = [members for members in parts if len(members) >= min_subgroup or members is largest]
    leftover = [s for members in parts if members is not largest and len(members) < min_subgroup for s in members]
    if leftover:
        largest.extend(leftover)

    return kept if len(kept) > 1 else [signature_ids]


def coauthor_corroborated_split(
    pred_clusters: dict[str, list[str]],
    dataset: "ANDData",
    *,
    min_cluster: int = 8,
    min_subgroup: int = 2,
    embed_distance_threshold: float = 0.17,
    remerge_min_shared_coauthors: int = 2,
) -> dict[str, list[str]]:
    """Split over-merged same-name clusters using embedding topics + shared coauthors.

    Parameters
    ----------
    pred_clusters: dict
        cluster id -> list of signature ids (output of ``Clusterer.predict``)
    dataset: ANDData
        the dataset (provides specter embeddings and coauthor names)
    min_cluster: int
        minimum number of signatures with usable embeddings for a cluster to be
        considered for splitting
    min_subgroup: int
        minimum size of a resulting subcluster (smaller ones fold into the largest)
    embed_distance_threshold: float
        cosine distance threshold for the agglomerative over-segmentation; smaller
        produces finer topic sub-groups
    remerge_min_shared_coauthors: int
        minimum number of shared (full-name) coauthors for two sub-groups to be
        merged back together as the same person

    Returns
    -------
    dict: refined cluster id -> list of signature ids
    """
    if dataset.specter_embeddings is None:
        return pred_clusters

    refined: dict[str, list[str]] = {}
    splits_made = 0
    for cluster_id, signature_ids in pred_clusters.items():
        parts = _split_one_cluster(
            list(signature_ids),
            dataset,
            min_cluster=min_cluster,
            min_subgroup=min_subgroup,
            embed_distance_threshold=embed_distance_threshold,
            remerge_min_shared_coauthors=remerge_min_shared_coauthors,
        )
        if len(parts) > 1:
            splits_made += len(parts) - 1
        for offset, fragment in enumerate(parts):
            key = cluster_id if offset == 0 else f"{cluster_id}__topicsplit{offset}"
            refined[key] = fragment

    if splits_made:
        logger.info(
            "Telemetry: coauthor_corroborated_split made %d split(s), %d -> %d clusters",
            splits_made,
            len(pred_clusters),
            len(refined),
        )
    return refined
