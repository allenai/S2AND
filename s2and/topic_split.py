"""Post-clustering topic split corroborated by coauthor disjointness.

Same-name authors who work in different fields (e.g. a father/son pair, or
several unrelated researchers who share a name) are frequently over-merged into
one cluster: their papers' SPECTER title/abstract embeddings sit close in
*absolute* cosine (most academic titles are cosine ~0.7-0.9 apart), so the
pairwise model treats them as compatible and average-linkage fuses them.

The topic signal that *does* separate them is relative/structural: each person's
papers form a tight embedding sub-cloud, and the two sub-clouds have distinct
centroids. A 2-means split on the embeddings recovers that structure with high
accuracy. But topic alone is not enough to act on -- real, single authors also
span multiple topics, so splitting on topic bimodality alone over-fragments them
and costs recall.

We therefore only split a cluster when the topic split is *corroborated* by the
coauthor graph: the two embedding sub-clusters must have (near-)disjoint
coauthors. A genuine single author carries recurring coauthors across their own
topics; distinct same-name people do not. Affiliation is deliberately not used
as a gate -- geography tokens (e.g. "seattle", "washington") cause spurious
overlap between distinct people in the same city.

This runs as a post-processing pass over predicted clusters and never merges --
it can only split -- so it cannot reduce recall on already-separated clusters.
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


def _coauthor_blocks(dataset: "ANDData", signature_id: str) -> set[str]:
    blocks = dataset.signatures[signature_id].author_info_coauthor_blocks
    return blocks if blocks else set()


def _coauthor_overlap_fraction(group_a: list[str], group_b: list[str], dataset: "ANDData") -> float:
    """Shared coauthors / min(|A|, |B|). Returns 0 when either side has no coauthors.

    Zero is the conservative value for the *split* decision: with no coauthor
    evidence we cannot corroborate that the two sub-clusters are the same person,
    so we let the topic split stand (gated additionally by the silhouette test).
    """
    coauthors_a: set[str] = set().union(*[_coauthor_blocks(dataset, s) for s in group_a]) if group_a else set()
    coauthors_b: set[str] = set().union(*[_coauthor_blocks(dataset, s) for s in group_b]) if group_b else set()
    if not coauthors_a or not coauthors_b:
        return 0.0
    shared = len(coauthors_a & coauthors_b)
    return shared / min(len(coauthors_a), len(coauthors_b))


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


def _try_split_once(
    signature_ids: list[str],
    dataset: "ANDData",
    *,
    min_embedded: int,
    min_subcluster: int,
    silhouette_threshold: float,
    max_coauthor_overlap: float,
) -> list[list[str]] | None:
    """Attempt a single 2-way split. Returns two subclusters, or None if it should not split."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    embedded, matrix = _embeddings_for_signatures(signature_ids, dataset)
    if matrix is None or len(embedded) < min_embedded:
        return None

    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(matrix)
    if int(min(np.bincount(labels))) < min_subcluster:
        return None
    if float(silhouette_score(matrix, labels)) < silhouette_threshold:
        return None

    # Assign every signature (including those without embeddings) to the nearest
    # subcluster centroid; embedding-less signatures fall back to the first group.
    centroids = np.vstack([matrix[labels == k].mean(axis=0) for k in (0, 1)])
    embedded_to_vector = dict(zip(embedded, matrix, strict=True))
    groups: dict[int, list[str]] = {0: [], 1: []}
    for signature_id in signature_ids:
        vector = embedded_to_vector.get(signature_id)
        if vector is None:
            groups[0].append(signature_id)
            continue
        nearest = int(np.argmin([np.linalg.norm(vector - centroids[k]) for k in (0, 1)]))
        groups[nearest].append(signature_id)

    if not groups[0] or not groups[1]:
        return None

    # Corroborate the topic split with coauthor disjointness.
    if _coauthor_overlap_fraction(groups[0], groups[1], dataset) > max_coauthor_overlap:
        return None

    return [groups[0], groups[1]]


def coauthor_corroborated_split(
    pred_clusters: dict[str, list[str]],
    dataset: "ANDData",
    *,
    min_embedded: int = 10,
    min_subcluster: int = 3,
    silhouette_threshold: float = 0.15,
    max_coauthor_overlap: float = 0.05,
) -> dict[str, list[str]]:
    """Split over-merged same-name clusters using embedding topics + coauthor disjointness.

    Recursively splits each predicted cluster while a corroborated 2-way topic
    split exists, so blocks with more than two distinct same-name identities can
    be separated in multiple passes.

    Parameters
    ----------
    pred_clusters: dict
        cluster id -> list of signature ids (output of ``Clusterer.predict``)
    dataset: ANDData
        the dataset (provides specter embeddings and coauthor blocks)
    min_embedded: int
        minimum number of signatures with usable embeddings for a cluster to be
        considered for splitting
    min_subcluster: int
        minimum size of each side of a split
    silhouette_threshold: float
        minimum 2-means silhouette for the topic split to be considered bimodal
    max_coauthor_overlap: float
        maximum allowed coauthor overlap (shared / min) between the two sides for
        the split to be accepted as a different-person separation

    Returns
    -------
    dict: refined cluster id -> list of signature ids
    """
    if dataset.specter_embeddings is None:
        return pred_clusters

    refined: dict[str, list[str]] = {}
    splits_made = 0
    for cluster_id, signature_ids in pred_clusters.items():
        # Work list of fragments still eligible for further splitting.
        pending = [list(signature_ids)]
        finalized: list[list[str]] = []
        while pending:
            fragment = pending.pop()
            parts = _try_split_once(
                fragment,
                dataset,
                min_embedded=min_embedded,
                min_subcluster=min_subcluster,
                silhouette_threshold=silhouette_threshold,
                max_coauthor_overlap=max_coauthor_overlap,
            )
            if parts is None:
                finalized.append(fragment)
            else:
                splits_made += 1
                pending.extend(parts)
        # Keep the original id for the first fragment; suffix the rest so ids stay unique.
        for index, fragment in enumerate(finalized):
            key = cluster_id if index == 0 else f"{cluster_id}__topicsplit{index}"
            refined[key] = fragment

    if splits_made:
        logger.info(
            "Telemetry: coauthor_corroborated_split made %d split(s), %d -> %d clusters",
            splits_made,
            len(pred_clusters),
            len(refined),
        )
    return refined
