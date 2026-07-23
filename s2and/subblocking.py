import hashlib
import json
import logging
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import genieclust
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

from s2and.arrow_inputs import require_arrow_artifacts
from s2and.arrow_schema import validate_arrow_schema
from s2and.consts import _PACKAGE_DATA_DIR, NORMALIZATION_VERSION, SPECTER_DIM
from s2and.incremental_linking.feature_block_arrow import read_arrow_batch_lookup_index_batch_indices_for_request
from s2and.orcid_prefix_counts import (
    ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
    ORCID_PREFIX_DATA_FILENAME,
    ORCID_PREFIX_METADATA_FILENAME,
    ORCID_PREFIX_PAIR_KEY_SEMANTICS,
    validate_orcid_prefix_counts,
)
from s2and.text import (
    AFFILIATIONS_STOP_WORDS,
    canonicalize_name_parts,
    compute_block,
    get_text_ngrams_words,
    normalize_orcid,
    normalize_text,
    same_prefix_tokens,
)

logger = logging.getLogger("s2and")


def _canonical_orcid_prefix_pair(first: str, second: str) -> tuple[str, str]:
    """Return the order-independent key for an ORCID prefix pair."""

    return (first, second) if first <= second else (second, first)


def _orcid_prefix_pair_count(counts: Mapping[str, Mapping[str, int]], first: str, second: str) -> int | None:
    """Look up a count in the canonical lexicographically ordered mapping."""

    left, right = _canonical_orcid_prefix_pair(first, second)
    nested = counts.get(left)
    return None if nested is None else nested.get(right)


def _read_orcid_json(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing canonical ORCID prefix-count {label}: {path}. Regenerate and publish the "
            "canonical ORCID prefix-count data and metadata before using default subblocking priors."
        ) from error
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Canonical ORCID prefix-count {label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Canonical ORCID prefix-count {label} must be a JSON object: {path}")
    return payload, value


def _require_orcid_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("Canonical ORCID prefix-count metadata data_sha256 must be a lowercase SHA-256 digest")
    return value


def _load_canonical_orcid_prefix_count_artifact(
    data_dir: str | Path = _PACKAGE_DATA_DIR,
) -> tuple[Mapping[str, Mapping[str, int]], str]:
    """Load direct count data and its content hash after one-pass validation."""

    root = Path(data_dir)
    metadata_path = root / ORCID_PREFIX_METADATA_FILENAME
    _, metadata = _read_orcid_json(metadata_path, label="metadata")
    expected_metadata = {
        "schema_version": ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "pair_key_semantics": ORCID_PREFIX_PAIR_KEY_SEMANTICS,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Canonical ORCID prefix-count metadata {key} must equal {expected!r}")
    if set(metadata) != {*expected_metadata, "data_sha256"}:
        raise ValueError("Canonical ORCID prefix-count metadata fields do not match the direct artifact contract")
    expected_data_sha256 = _require_orcid_sha256(metadata.get("data_sha256"))

    data_path = root / ORCID_PREFIX_DATA_FILENAME
    data_payload, raw_counts = _read_orcid_json(data_path, label="data")
    if hashlib.sha256(data_payload).hexdigest() != expected_data_sha256:
        raise ValueError("Canonical ORCID prefix-count data SHA-256 does not match its metadata")
    validate_orcid_prefix_counts(raw_counts, context="Canonical ORCID prefix-count")
    immutable_counts = MappingProxyType(
        {prefix: MappingProxyType(dict(nested_counts)) for prefix, nested_counts in raw_counts.items()}
    )
    return immutable_counts, expected_data_sha256


class _LazyCanonicalOrcidPrefixCounts(Mapping[str, Mapping[str, int]]):
    """Defer canonical artifact I/O until subblocking actually needs the priors."""

    def __init__(self, data_dir: str | Path) -> None:
        self._data_dir = Path(data_dir)
        self._loaded: Mapping[str, Mapping[str, int]] | None = None
        self._data_sha256: str | None = None

    def load(self) -> Mapping[str, Mapping[str, int]]:
        loaded = self._loaded
        if loaded is None:
            loaded, data_sha256 = _load_canonical_orcid_prefix_count_artifact(self._data_dir)
            self._data_sha256 = data_sha256
            self._loaded = loaded
        return loaded

    def data_sha256(self) -> str:
        """Return the verified count-data content hash."""

        self.load()
        data_sha256 = self._data_sha256
        if data_sha256 is None:
            raise RuntimeError("Canonical ORCID prefix-count data hash was not retained after loading")
        return data_sha256

    def __getitem__(self, key: str) -> Mapping[str, int]:
        return self.load()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.load())

    def __len__(self) -> int:
        return len(self.load())


def _resolved_orcid_prefix_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return counts.load() if isinstance(counts, _LazyCanonicalOrcidPrefixCounts) else counts


FIRST_K_LETTER_COUNTS = _LazyCanonicalOrcidPrefixCounts(_PACKAGE_DATA_DIR)


def canonical_orcid_prefix_counts_data_sha256() -> str:
    """Return the content hash of the verified packaged ORCID priors."""

    return FIRST_K_LETTER_COUNTS.data_sha256()


def normalize_orcid_for_subblocking(value: Any) -> str | None:
    """Return the canonical ORCID key used by Rust Arrow subblocking inputs.

    This mirrors Rust's `normalize_orcid_owned(...)`: find one ORCID-shaped
    token, uppercase the check digit, and format the result with hyphens.
    """

    return normalize_orcid(value)


@dataclass(frozen=True)
class GraphSubblockingConfig:
    """Configuration for the graph fallback used by subblocking."""

    neighbor_mode: str = "projection"
    neighbors: int = 16
    min_edge_score: float = 0.30
    specter_weight: float = 1.0
    coauthor_weight: float = 0.35
    affiliation_weight: float = 0.20
    max_exact_knn_group_size: int = 25_000
    projection_count: int = 12
    projection_window: int = 12
    max_candidate_edges: int = 5_000_000
    pack_components: bool = True
    component_pack_strategy: str = "edge-greedy"
    sparse_evidence_edges: bool = True
    sparse_evidence_max_posting_size: int = 8
    sparse_evidence_neighbors: int = 1
    sparse_evidence_min_weight: float = 0.40
    sparse_evidence_include_coauthors: bool = True
    sparse_evidence_include_affiliations: bool = False
    component_pack_top_k: int = 8
    local_move_passes: int = 0
    adaptive_projection: bool = False
    adaptive_projection_max_group_size: int = 5_000
    adaptive_projection_count: int = 24
    adaptive_projection_window: int = 24


def _resolve_graph_subblocking_config(raw_config: object) -> GraphSubblockingConfig:
    """Resolve one strict graph-subblocking configuration value."""

    if raw_config is None:
        return GraphSubblockingConfig()
    if isinstance(raw_config, GraphSubblockingConfig):
        return raw_config
    if isinstance(raw_config, Mapping):
        return GraphSubblockingConfig(**dict(raw_config))
    raise ValueError(
        f"subblocking_graph_config must be a GraphSubblockingConfig, mapping, or None; got {type(raw_config).__name__}"
    )


class _UnionFind:
    """Capacity-constrained union find for graph components."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union_if_capacity(self, left: int, right: int, maximum_size: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        merged_size = self.component_size[left_root] + self.component_size[right_root]
        if merged_size > maximum_size:
            return False
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] = merged_size
        return True


@dataclass(frozen=True)
class _SignatureEvidence:
    """Metadata evidence used while scoring candidate graph edges."""

    coauthor_blocks: frozenset[str]
    affiliation_keys: frozenset[str]


def signature_affiliation_feature_keys(signature) -> list[str]:
    if signature.author_info_affiliations_n_grams is not None:
        return list(signature.author_info_affiliations_n_grams.keys())
    return list(_affiliation_ngrams_from_raw_affiliations(signature.author_info_affiliations).keys())


def _affiliation_ngrams_from_raw_affiliations(affiliations: Iterable[Any] | None) -> Counter:
    values = [normalize_text(str(value)) for value in affiliations or () if value is not None]
    text = " ".join(value for value in values if value)
    return get_text_ngrams_words(text, stopwords=AFFILIATIONS_STOP_WORDS) if text else Counter()


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _signature_evidence(
    signature, anddata=None, compute_block_fn: Callable[[str], str] = compute_block
) -> _SignatureEvidence:
    if anddata is None:
        coauthor_blocks = signature.author_info_coauthor_blocks or ()
    else:
        coauthor_blocks = _signature_coauthor_blocks_for_specter(signature, anddata, compute_block_fn)
    return _SignatureEvidence(
        coauthor_blocks=frozenset(str(value) for value in coauthor_blocks if value),
        affiliation_keys=frozenset(signature_affiliation_feature_keys(signature)),
    )


def _embedding_matrix(signature_ids: Sequence[str], anddata, *, dimension: int = SPECTER_DIM) -> np.ndarray:
    first_embedding = next(iter(getattr(anddata, "specter_embeddings", {}).values()), None)
    if first_embedding is not None:
        dimension = int(np.asarray(first_embedding).shape[0])
    matrix = np.zeros((len(signature_ids), int(dimension)), dtype=np.float32)
    for row_index, signature_id in enumerate(signature_ids):
        signature = anddata.signatures[str(signature_id)]
        embedding = getattr(anddata, "specter_embeddings", {}).get(str(signature.paper_id))
        if embedding is not None:
            matrix[row_index, :] = np.asarray(embedding, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] /= norms[nonzero, None]
    return matrix


def _weighted_edge_score(
    cosine_similarity: float,
    left_evidence: _SignatureEvidence,
    right_evidence: _SignatureEvidence,
    config: GraphSubblockingConfig,
) -> float:
    return (
        config.specter_weight * cosine_similarity
        + config.coauthor_weight * _jaccard(left_evidence.coauthor_blocks, right_evidence.coauthor_blocks)
        + config.affiliation_weight * _jaccard(left_evidence.affiliation_keys, right_evidence.affiliation_keys)
    )


def _prune_edge_scores(edge_scores: dict[tuple[int, int], float], max_candidate_edges: int) -> None:
    if max_candidate_edges <= 0 or len(edge_scores) <= max_candidate_edges:
        return
    strongest = sorted(edge_scores.items(), key=lambda item: (-item[1], item[0]))[:max_candidate_edges]
    edge_scores.clear()
    edge_scores.update(strongest)


def _score_candidate_edge(
    edge_scores: dict[tuple[int, int], float],
    *,
    left_index: int,
    right_index: int,
    matrix: np.ndarray,
    evidences: Sequence[_SignatureEvidence],
    config: GraphSubblockingConfig,
) -> None:
    if left_index == right_index:
        return
    left = min(left_index, right_index)
    right = max(left_index, right_index)
    cosine_similarity = max(0.0, float(np.dot(matrix[left], matrix[right])))
    score = _weighted_edge_score(cosine_similarity, evidences[left], evidences[right], config)
    if score < config.min_edge_score:
        return
    current = edge_scores.get((left, right))
    if current is None or score > current:
        edge_scores[(left, right)] = score


def _score_candidate_edge_from_cosine(
    edge_scores: dict[tuple[int, int], float],
    *,
    left_index: int,
    right_index: int,
    cosine_similarity: float,
    evidences: Sequence[_SignatureEvidence],
    config: GraphSubblockingConfig,
) -> None:
    score = _weighted_edge_score(max(0.0, cosine_similarity), evidences[left_index], evidences[right_index], config)
    if score < config.min_edge_score:
        return
    current = edge_scores.get((left_index, right_index))
    if current is None or score > current:
        edge_scores[(left_index, right_index)] = score


def _exact_neighbor_edge_scores(
    matrix: np.ndarray,
    evidences: Sequence[_SignatureEvidence],
    config: GraphSubblockingConfig,
) -> dict[tuple[int, int], float]:
    neighbor_count = min(matrix.shape[0], max(2, int(config.neighbors) + 1))
    nearest = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute")
    distances, indices = nearest.fit(matrix).kneighbors(matrix)
    edge_scores: dict[tuple[int, int], float] = {}
    for left_index in range(matrix.shape[0]):
        for distance, right_index_raw in zip(distances[left_index], indices[left_index], strict=True):
            right_index = int(right_index_raw)
            if right_index == left_index:
                continue
            left = min(left_index, right_index)
            right = max(left_index, right_index)
            cosine_similarity = max(0.0, 1.0 - float(distance))
            score = _weighted_edge_score(cosine_similarity, evidences[left], evidences[right], config)
            if score < config.min_edge_score:
                continue
            current = edge_scores.get((left, right))
            if current is None or score > current:
                edge_scores[(left, right)] = score
    return edge_scores


def _projection_neighbor_edge_scores(
    matrix: np.ndarray,
    evidences: Sequence[_SignatureEvidence],
    config: GraphSubblockingConfig,
    *,
    seed: int,
) -> dict[tuple[int, int], float]:
    if config.projection_count <= 0:
        raise ValueError("Graph subblocking projection_count must be positive")
    if config.projection_window <= 0:
        raise ValueError("Graph subblocking projection_window must be positive")
    rng = np.random.default_rng(seed)
    projection_vectors = rng.standard_normal(
        (matrix.shape[1], int(config.projection_count)),
        dtype=np.float32,
    )
    projection_norms = np.linalg.norm(projection_vectors, axis=0)
    nonzero = projection_norms > 0
    projection_vectors[:, nonzero] /= projection_norms[nonzero]
    projection_scores = matrix @ projection_vectors
    edge_scores: dict[tuple[int, int], float] = {}
    window = int(config.projection_window)
    score_chunk_size = 8192
    pending_left = np.empty(score_chunk_size, dtype=np.int64)
    pending_right = np.empty(score_chunk_size, dtype=np.int64)

    def flush_pending(pending_size: int) -> int:
        if pending_size == 0:
            return 0
        left_chunk = pending_left[:pending_size]
        right_chunk = pending_right[:pending_size]
        cosine_similarities = np.einsum("ij,ij->i", matrix[left_chunk], matrix[right_chunk])
        for row_index, cosine_similarity in enumerate(cosine_similarities):
            _score_candidate_edge_from_cosine(
                edge_scores,
                left_index=int(left_chunk[row_index]),
                right_index=int(right_chunk[row_index]),
                cosine_similarity=float(cosine_similarity),
                evidences=evidences,
                config=config,
            )
        return 0

    for projection_index in range(projection_scores.shape[1]):
        order = np.argsort(projection_scores[:, projection_index], kind="mergesort")
        pending_size = 0
        for position, left_index_raw in enumerate(order):
            left_index = int(left_index_raw)
            stop = min(len(order), position + window + 1)
            for right_index_raw in order[position + 1 : stop]:
                right_index = int(right_index_raw)
                left = min(left_index, right_index)
                right = max(left_index, right_index)
                if (left, right) in edge_scores:
                    continue
                pending_left[pending_size] = left
                pending_right[pending_size] = right
                pending_size += 1
                if pending_size == score_chunk_size:
                    pending_size = flush_pending(pending_size)
        flush_pending(pending_size)
        _prune_edge_scores(edge_scores, int(config.max_candidate_edges))
    return edge_scores


def _add_sparse_evidence_edge_scores(
    edge_scores: dict[tuple[int, int], float],
    matrix: np.ndarray,
    evidences: Sequence[_SignatureEvidence],
    config: GraphSubblockingConfig,
) -> dict[str, int]:
    """Add bounded edges from rare shared coauthor/affiliation evidence."""

    max_posting_size = int(config.sparse_evidence_max_posting_size)
    if max_posting_size <= 1:
        raise ValueError("Graph subblocking sparse_evidence_max_posting_size must be greater than 1")
    neighbor_limit = int(config.sparse_evidence_neighbors)
    if neighbor_limit < 0:
        raise ValueError("Graph subblocking sparse_evidence_neighbors must be non-negative")
    postings: dict[str, list[int]] = defaultdict(list)
    for index, evidence in enumerate(evidences):
        if config.sparse_evidence_include_coauthors:
            for value in evidence.coauthor_blocks:
                if value:
                    postings[f"coauthor:{value}"].append(index)
        if config.sparse_evidence_include_affiliations:
            for value in evidence.affiliation_keys:
                if value:
                    postings[f"affiliation:{value}"].append(index)

    features_considered = 0
    features_skipped = 0
    edge_count_before = len(edge_scores)
    score_threshold = float(config.sparse_evidence_min_weight)
    for indices in postings.values():
        posting_size = len(indices)
        if posting_size <= 1:
            continue
        if posting_size > max_posting_size:
            features_skipped += 1
            continue
        features_considered += 1
        sorted_indices = sorted(indices)
        for left_offset, left_index in enumerate(sorted_indices):
            right_indices = (
                sorted_indices[left_offset + 1 :]
                if neighbor_limit == 0
                else sorted_indices[left_offset + 1 : left_offset + neighbor_limit + 1]
            )
            for right_index in right_indices:
                left_evidence = evidences[left_index]
                right_evidence = evidences[right_index]
                cosine_similarity = max(0.0, float(np.dot(matrix[left_index], matrix[right_index])))
                score = _weighted_edge_score(cosine_similarity, left_evidence, right_evidence, config)
                if score < score_threshold:
                    continue
                left = min(left_index, right_index)
                right = max(left_index, right_index)
                current = edge_scores.get((left, right))
                if current is None or score > current:
                    edge_scores[(left, right)] = score
        if int(config.max_candidate_edges) > 0 and len(edge_scores) > int(config.max_candidate_edges) * 2:
            _prune_edge_scores(edge_scores, int(config.max_candidate_edges))
    _prune_edge_scores(edge_scores, int(config.max_candidate_edges))
    return {
        "sparse_evidence_feature_count": int(features_considered),
        "sparse_evidence_skipped_feature_count": int(features_skipped),
        "sparse_evidence_added_edge_count": max(0, int(len(edge_scores) - edge_count_before)),
        "sparse_evidence_neighbors": int(config.sparse_evidence_neighbors),
    }


def _ordered_components_from_union(
    signature_ids: Sequence[str],
    uf: _UnionFind,
) -> tuple[list[list[str]], list[int], dict[int, int]]:
    root_by_index = [uf.find(index) for index in range(len(signature_ids))]
    components_by_root: dict[int, list[str]] = defaultdict(list)
    for index, signature_id in enumerate(signature_ids):
        components_by_root[root_by_index[index]].append(str(signature_id))
    roots = sorted(
        components_by_root,
        key=lambda root: (-len(components_by_root[root]), sorted(components_by_root[root])[0]),
    )
    component_id_by_root = {root: component_id for component_id, root in enumerate(roots)}
    components = [sorted(components_by_root[root]) for root in roots]
    return components, root_by_index, component_id_by_root


def _pack_components_by_size(components: Sequence[Sequence[str]], target_subblock_size: int) -> list[list[str]]:
    bins: list[list[str]] = []
    bin_sizes: list[int] = []
    for component in components:
        component_size = len(component)
        if component_size > target_subblock_size:
            raise ValueError(
                f"Graph component size {component_size} exceeds target_subblock_size={target_subblock_size}"
            )
        best_bin = None
        best_remaining = None
        for bin_index, bin_size in enumerate(bin_sizes):
            remaining = target_subblock_size - bin_size
            if component_size <= remaining and (best_remaining is None or remaining < best_remaining):
                best_bin = bin_index
                best_remaining = remaining
        if best_bin is None:
            bins.append(list(component))
            bin_sizes.append(component_size)
        else:
            bins[best_bin].extend(component)
            bin_sizes[best_bin] += component_size
    return [sorted(values) for values in bins]


def _component_adjacency(
    edge_scores: Mapping[tuple[int, int], float],
    root_by_index: Sequence[int],
    component_id_by_root: Mapping[int, int],
    *,
    aggregate: bool = False,
) -> dict[int, dict[int, float]]:
    adjacency: dict[int, dict[int, float]] = defaultdict(dict)
    for (left_index, right_index), score in edge_scores.items():
        left_component = component_id_by_root[root_by_index[left_index]]
        right_component = component_id_by_root[root_by_index[right_index]]
        if left_component == right_component:
            continue
        current = adjacency[left_component].get(right_component)
        if aggregate:
            adjacency[left_component][right_component] = (current or 0.0) + score
            adjacency[right_component][left_component] = adjacency[right_component].get(left_component, 0.0) + score
        elif current is None or score > current:
            adjacency[left_component][right_component] = score
            adjacency[right_component][left_component] = score
    return adjacency


def _component_affinities_to_bins(
    component_id: int,
    component_to_bin: Mapping[int, int],
    adjacency: Mapping[int, Mapping[int, float]],
    *,
    top_k: int,
) -> dict[int, float]:
    scores_by_bin: dict[int, list[float]] = defaultdict(list)
    for neighbor_component_id, score in adjacency.get(component_id, {}).items():
        bin_index = component_to_bin.get(neighbor_component_id)
        if bin_index is not None:
            scores_by_bin[bin_index].append(float(score))
    out: dict[int, float] = {}
    for bin_index, scores in scores_by_bin.items():
        if top_k > 0 and len(scores) > top_k:
            scores = sorted(scores, reverse=True)[:top_k]
        out[bin_index] = float(sum(scores))
    return out


def _pack_component_ids_greedy(
    components: Sequence[Sequence[str]],
    edge_scores: Mapping[tuple[int, int], float],
    root_by_index: Sequence[int],
    component_id_by_root: Mapping[int, int],
    target_subblock_size: int,
    config: GraphSubblockingConfig,
) -> list[list[int]]:
    use_aggregate = config.component_pack_strategy == "aggregate-greedy"
    adjacency = _component_adjacency(edge_scores, root_by_index, component_id_by_root, aggregate=use_aggregate)
    component_order = sorted(range(len(components)), key=lambda index: (-len(components[index]), components[index][0]))
    bins: list[list[int]] = []
    bin_sizes: list[int] = []
    component_to_bin: dict[int, int] = {}
    for component_id in component_order:
        component = components[component_id]
        component_size = len(component)
        if component_size > target_subblock_size:
            raise ValueError(
                f"Graph component size {component_size} exceeds target_subblock_size={target_subblock_size}"
            )

        candidate_bins: dict[int, float] = {}
        if use_aggregate:
            affinities = _component_affinities_to_bins(
                component_id,
                component_to_bin,
                adjacency,
                top_k=int(config.component_pack_top_k),
            )
            for bin_index, affinity in affinities.items():
                if bin_index >= len(bins) or bin_sizes[bin_index] + component_size > target_subblock_size:
                    continue
                if affinity > 0.0:
                    candidate_bins[bin_index] = affinity
        else:
            for neighbor_component, score in adjacency.get(component_id, {}).items():
                bin_index = component_to_bin.get(neighbor_component)
                if bin_index is None or bin_sizes[bin_index] + component_size > target_subblock_size:
                    continue
                current = candidate_bins.get(bin_index)
                if current is None or score > current:
                    candidate_bins[bin_index] = score

        if candidate_bins:
            selected_bin = min(
                candidate_bins,
                key=lambda bin_index: (
                    -candidate_bins[bin_index],
                    target_subblock_size - (bin_sizes[bin_index] + component_size),
                    bin_index,
                ),
            )
        else:
            selected_bin = None
            selected_remaining = None
            for bin_index, bin_size in enumerate(bin_sizes):
                remaining = target_subblock_size - bin_size
                if component_size <= remaining and (selected_remaining is None or remaining < selected_remaining):
                    selected_bin = bin_index
                    selected_remaining = remaining

        if selected_bin is None:
            selected_bin = len(bins)
            bins.append([])
            bin_sizes.append(0)
        bins[selected_bin].append(component_id)
        bin_sizes[selected_bin] += component_size
        component_to_bin[component_id] = selected_bin

    if int(config.local_move_passes) > 0:
        bins = _local_move_component_bins(
            components,
            bins,
            adjacency,
            target_subblock_size,
            passes=int(config.local_move_passes),
            top_k=int(config.component_pack_top_k),
        )
    return bins


def _component_ids_to_subblocks(
    components: Sequence[Sequence[str]],
    bins: Sequence[Sequence[int]],
) -> list[list[str]]:
    packed: list[list[str]] = []
    for component_ids in bins:
        values: list[str] = []
        for component_id in sorted(component_ids, key=lambda index: components[index][0]):
            values.extend(components[component_id])
        packed.append(sorted(values))
    return packed


def _local_move_component_bins(
    components: Sequence[Sequence[str]],
    bins: Sequence[Sequence[int]],
    adjacency: Mapping[int, Mapping[int, float]],
    target_subblock_size: int,
    *,
    passes: int,
    top_k: int,
) -> list[list[int]]:
    """Move raw components between bins when aggregate graph affinity improves."""

    working_bins = [list(component_ids) for component_ids in bins]
    bin_sizes = [sum(len(components[component_id]) for component_id in component_ids) for component_ids in working_bins]
    for _pass_index in range(max(0, passes)):
        moved = False
        component_to_bin = {
            component_id: bin_index
            for bin_index, component_ids in enumerate(working_bins)
            for component_id in component_ids
        }
        component_order = sorted(
            component_to_bin,
            key=lambda component_id: (len(components[component_id]), components[component_id][0]),
        )
        for component_id in component_order:
            source_bin = component_to_bin.get(component_id)
            if source_bin is None:
                continue
            component_size = len(components[component_id])
            affinities = _component_affinities_to_bins(
                component_id,
                component_to_bin,
                adjacency,
                top_k=top_k,
            )
            current_affinity = affinities.get(source_bin, 0.0)
            best_bin = None
            best_gain = 0.0
            for target_bin, candidate_affinity in affinities.items():
                if target_bin >= len(working_bins):
                    continue
                if target_bin == source_bin:
                    continue
                if bin_sizes[target_bin] + component_size > target_subblock_size:
                    continue
                gain = candidate_affinity - current_affinity
                if gain > best_gain:
                    best_gain = gain
                    best_bin = target_bin
            if best_bin is None:
                continue
            working_bins[source_bin].remove(component_id)
            working_bins[best_bin].append(component_id)
            bin_sizes[source_bin] -= component_size
            bin_sizes[best_bin] += component_size
            component_to_bin[component_id] = best_bin
            moved = True
        if not moved:
            break
        nonempty_bins = []
        nonempty_sizes = []
        for component_ids, bin_size in zip(working_bins, bin_sizes, strict=True):
            if component_ids:
                nonempty_bins.append(component_ids)
                nonempty_sizes.append(bin_size)
        working_bins = nonempty_bins
        bin_sizes = nonempty_sizes
    return working_bins


def _pack_graph_components(
    components: Sequence[Sequence[str]],
    edge_scores: Mapping[tuple[int, int], float],
    root_by_index: Sequence[int],
    component_id_by_root: Mapping[int, int],
    target_subblock_size: int,
    config: GraphSubblockingConfig,
) -> list[list[str]]:
    if not config.pack_components:
        return [list(component) for component in components]
    if config.component_pack_strategy == "size":
        return _pack_components_by_size(components, target_subblock_size)
    if config.component_pack_strategy in {"edge-greedy", "aggregate-greedy"}:
        return _component_ids_to_subblocks(
            components,
            _pack_component_ids_greedy(
                components,
                edge_scores,
                root_by_index,
                component_id_by_root,
                target_subblock_size,
                config,
            ),
        )
    raise ValueError(f"Unsupported component_pack_strategy={config.component_pack_strategy!r}")


def _effective_graph_config(config: GraphSubblockingConfig, group_size: int) -> GraphSubblockingConfig:
    if (
        not config.adaptive_projection
        or config.neighbor_mode != "projection"
        or group_size > int(config.adaptive_projection_max_group_size)
    ):
        return config
    return replace(
        config,
        projection_count=max(int(config.projection_count), int(config.adaptive_projection_count)),
        projection_window=max(int(config.projection_window), int(config.adaptive_projection_window)),
    )


def cluster_with_graph_fallback(
    signature_ids: Iterable[str],
    anddata,
    target_subblock_size: int = 10000,
    compute_block_fn: Callable[[str], str] = compute_block,
    *,
    config: GraphSubblockingConfig | None = None,
    stats: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Cluster an oversized fallback group with a capacity-constrained graph."""

    config = config or GraphSubblockingConfig()
    if config.neighbor_mode not in {"projection", "exact"}:
        raise ValueError("Graph subblocking neighbor_mode must be 'projection' or 'exact'")
    if config.component_pack_strategy not in {"edge-greedy", "aggregate-greedy", "size"}:
        raise ValueError(
            "Graph subblocking component_pack_strategy must be 'edge-greedy', 'aggregate-greedy', or 'size'"
        )
    fallback_start = time.perf_counter()
    signature_ids = [str(signature_id) for signature_id in signature_ids]
    if len(signature_ids) == 0:
        return {}
    if len(signature_ids) <= target_subblock_size:
        return {"0": signature_ids}
    config = _effective_graph_config(config, len(signature_ids))
    if config.neighbor_mode == "exact" and len(signature_ids) > config.max_exact_knn_group_size:
        raise ValueError(
            "Exact graph subblocking fallback group exceeds max_exact_knn_group_size: "
            f"group_size={len(signature_ids)} max_exact_knn_group_size={config.max_exact_knn_group_size}"
        )

    missing_signature_ids = [signature_id for signature_id in signature_ids if signature_id not in anddata.signatures]
    if missing_signature_ids:
        raise ValueError(f"Graph subblocking evidence is missing signatures: {missing_signature_ids[:10]}")

    matrix = _embedding_matrix(signature_ids, anddata)
    evidences = [
        _signature_evidence(anddata.signatures[signature_id], anddata, compute_block_fn)
        for signature_id in signature_ids
    ]
    edge_start = time.perf_counter()
    if config.neighbor_mode == "exact":
        edge_scores = _exact_neighbor_edge_scores(matrix, evidences, config)
    else:
        group_digest = hashlib.blake2b("\0".join(sorted(signature_ids)).encode("utf-8"), digest_size=8).digest()
        seed = (int(getattr(anddata, "random_seed", 0) or 0) + int.from_bytes(group_digest, "little")) % (2**32 - 1)
        edge_scores = _projection_neighbor_edge_scores(matrix, evidences, config, seed=seed)
    sparse_evidence_stats: dict[str, int] = {
        "sparse_evidence_feature_count": 0,
        "sparse_evidence_skipped_feature_count": 0,
        "sparse_evidence_added_edge_count": 0,
        "sparse_evidence_neighbors": int(config.sparse_evidence_neighbors),
    }
    if config.sparse_evidence_edges:
        sparse_evidence_stats = _add_sparse_evidence_edge_scores(edge_scores, matrix, evidences, config)
    edge_seconds = time.perf_counter() - edge_start

    uf = _UnionFind(len(signature_ids))
    sorted_edges = sorted(
        ((score, left, right) for (left, right), score in edge_scores.items()),
        key=lambda item: (-item[0], signature_ids[item[1]], signature_ids[item[2]]),
    )
    for _score, left, right in sorted_edges:
        uf.union_if_capacity(left, right, int(target_subblock_size))

    raw_components, root_by_index, component_id_by_root = _ordered_components_from_union(signature_ids, uf)
    ordered_components = sorted(
        _pack_graph_components(
            raw_components,
            edge_scores,
            root_by_index,
            component_id_by_root,
            int(target_subblock_size),
            config,
        ),
        key=lambda values: (-len(values), values[0]),
    )
    if stats is not None:
        raw_sizes = [len(values) for values in raw_components]
        sizes = [len(values) for values in ordered_components]
        stats.append(
            {
                "input_signature_count": int(len(signature_ids)),
                "neighbor_mode": config.neighbor_mode,
                "projection_count": int(config.projection_count),
                "projection_window": int(config.projection_window),
                "candidate_edge_count": int(len(edge_scores)),
                **sparse_evidence_stats,
                "raw_component_count": int(len(raw_components)),
                "raw_max_component_size": int(max(raw_sizes)) if raw_sizes else 0,
                "raw_median_component_size": float(np.median(raw_sizes)) if raw_sizes else 0.0,
                "packed_component_count": int(len(ordered_components)),
                "max_component_size": int(max(sizes)) if sizes else 0,
                "median_component_size": float(np.median(sizes)) if sizes else 0.0,
                "pack_components": bool(config.pack_components),
                "component_pack_strategy": config.component_pack_strategy,
                "component_pack_top_k": int(config.component_pack_top_k),
                "local_move_passes": int(config.local_move_passes),
                "edge_build_seconds": float(edge_seconds),
                "total_seconds": float(time.perf_counter() - fallback_start),
            }
        )
    return {str(index): values for index, values in enumerate(ordered_components)}


def _read_arrow_rows_by_values(
    path: Any,
    index_path: Any,
    key_column: str,
    values: Sequence[str],
    *,
    required_columns: set[str],
    optional_columns: Sequence[str] = (),
    table_name: str,
    load_metrics: dict[str, int] | None = None,
) -> list[Mapping[str, Any]]:
    pa = __import__("pyarrow")
    keep_values = {str(value) for value in values}
    if not keep_values:
        return []
    batch_indices = sorted(
        read_arrow_batch_lookup_index_batch_indices_for_request(
            path,
            index_path,
            key_column=key_column,
            values=keep_values,
        )
    )
    rows: list[Mapping[str, Any]] = []
    record_batches_scanned = 0
    rows_scanned = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        required_columns = set(required_columns)
        required_columns.add(key_column)
        selected_columns = required_columns.union(set(optional_columns).intersection(reader.schema.names))
        validate_arrow_schema(reader.schema, table_name=table_name, columns=selected_columns)
        key_column_index = reader.schema.get_field_index(key_column)
        if key_column_index < 0:
            raise ValueError(f"{table_name} Arrow is missing key column for graph subblocking: {key_column!r}")
        selected_column_names = [name for name in reader.schema.names if name in selected_columns]
        selected_column_indices = [reader.schema.get_field_index(name) for name in selected_column_names]
        for batch_index in batch_indices:
            batch = reader.get_batch(batch_index)
            keys = batch.column(key_column_index).to_pylist()
            record_batches_scanned += 1
            rows_scanned += len(keys)
            selected_indices = [
                row_index for row_index, key in enumerate(keys) if key is not None and str(key) in keep_values
            ]
            if not selected_indices:
                continue
            row_batch = pa.record_batch(
                [batch.column(index) for index in selected_column_indices],
                names=selected_column_names,
            )
            selected = (
                row_batch
                if len(selected_indices) == len(keys)
                else row_batch.take(pa.array(selected_indices, type=pa.int64()))
            )
            rows.extend(selected.to_pylist())
    if load_metrics is not None:
        _add_load_metric(load_metrics, f"{table_name}_record_batches_scanned", record_batches_scanned)
        _add_load_metric(load_metrics, f"{table_name}_rows_scanned", rows_scanned)
        _add_load_metric(load_metrics, f"{table_name}_rows_loaded", len(rows))
    return rows


def _add_load_metric(load_metrics: dict[str, int], key: str, value: int) -> None:
    load_metrics[key] = int(load_metrics.get(key, 0)) + int(value)


def _unique_rows_by_key(
    rows_in: Iterable[Mapping[str, Any]],
    *,
    table_name: str,
    key_column: str,
) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in rows_in:
        key_value = row.get(key_column)
        if key_value is None:
            raise ValueError(f"{table_name} Arrow cannot contain null {key_column} values")
        key = str(key_value)
        if not key:
            raise ValueError(f"{table_name} Arrow cannot contain empty {key_column} values")
        if key in rows:
            raise ValueError(f"{table_name} Arrow contains duplicate {key_column}: {key!r}")
        rows[key] = row
    return rows


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise ValueError("Arrow list field unexpectedly decoded as a scalar string")
    items: list[str] = []
    for item in value:
        if item is None:
            raise ValueError("Arrow list field cannot contain null values")
        items.append(str(item))
    return tuple(items)


def _coauthor_blocks_by_paper_from_arrow(
    paths: Mapping[str, Any],
    paper_ids: Sequence[str],
    *,
    load_metrics: dict[str, int],
) -> dict[str, list[tuple[int, str]]]:
    pa = __import__("pyarrow")
    paper_authors_path = paths.get("paper_authors")
    if paper_authors_path is None:
        return {}
    paper_authors_index_path = paths["paper_authors_batch_index"]
    keep_values = {str(value) for value in paper_ids}
    batch_indices = sorted(
        read_arrow_batch_lookup_index_batch_indices_for_request(
            paper_authors_path,
            paper_authors_index_path,
            key_column="paper_id",
            values=keep_values,
        )
    )
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    record_batches_scanned = 0
    rows_scanned = 0
    rows_loaded = 0
    with pa.memory_map(str(paper_authors_path), "r") as source:
        reader = pa.ipc.open_file(source)
        required_columns = {"paper_id", "position", "author_name"}
        validate_arrow_schema(reader.schema, table_name="paper_authors", columns=required_columns)
        paper_id_index = reader.schema.get_field_index("paper_id")
        position_index = reader.schema.get_field_index("position")
        author_name_index = reader.schema.get_field_index("author_name")
        seen_positions_by_paper: dict[str, set[int]] = defaultdict(set)
        for batch_index in batch_indices:
            batch = reader.get_batch(batch_index)
            paper_values = batch.column(paper_id_index).to_pylist()
            record_batches_scanned += 1
            rows_scanned += len(paper_values)
            selected_indices = [
                row_index
                for row_index, paper_id in enumerate(paper_values)
                if paper_id is not None and str(paper_id) in keep_values
            ]
            if not selected_indices:
                continue
            if len(selected_indices) == len(paper_values):
                selected_paper_values = paper_values
                selected_positions = batch.column(position_index).to_pylist()
                selected_author_names = batch.column(author_name_index).to_pylist()
            else:
                take_indices = pa.array(selected_indices, type=pa.int64())
                selected_paper_values = batch.column(paper_id_index).take(take_indices).to_pylist()
                selected_positions = batch.column(position_index).take(take_indices).to_pylist()
                selected_author_names = batch.column(author_name_index).take(take_indices).to_pylist()
            rows_loaded += len(selected_paper_values)
            for paper_id_value, position, author_name_value in zip(
                selected_paper_values,
                selected_positions,
                selected_author_names,
                strict=True,
            ):
                if position is None:
                    raise ValueError("paper_authors Arrow cannot contain null position values")
                paper_id = str(paper_id_value)
                if author_name_value is None:
                    raise ValueError("paper_authors Arrow cannot contain null author_name values")
                author_name = str(author_name_value).strip()
                position_key = int(position)
                if position_key in seen_positions_by_paper[paper_id]:
                    raise ValueError(
                        f"paper_authors Arrow contains duplicate (paper_id, position): ({paper_id!r}, {position_key})"
                    )
                seen_positions_by_paper[paper_id].add(position_key)
                if not author_name:
                    continue
                block = _coauthor_block_from_arrow_author_name(author_name)
                if block:
                    out[paper_id].append((position_key, block))
    _add_load_metric(load_metrics, "paper_authors_record_batches_scanned", record_batches_scanned)
    _add_load_metric(load_metrics, "paper_authors_rows_scanned", rows_scanned)
    _add_load_metric(load_metrics, "paper_authors_rows_loaded", rows_loaded)
    return out


def _specter_embeddings_from_arrow(
    paths: Mapping[str, Any],
    paper_ids: Sequence[str],
    *,
    load_metrics: dict[str, int],
) -> dict[str, np.ndarray]:
    pa = __import__("pyarrow")
    if "specter" not in paths:
        raise ValueError("Graph subblocking requires a 'specter' Arrow path")
    specter_path = paths["specter"]
    specter_index_path = paths["specter_batch_index"]
    keep_values = {str(value) for value in paper_ids}
    batch_indices = sorted(
        read_arrow_batch_lookup_index_batch_indices_for_request(
            specter_path,
            specter_index_path,
            key_column="paper_id",
            values=keep_values,
        )
    )
    embeddings: dict[str, np.ndarray] = {}
    record_batches_scanned = 0
    rows_scanned = 0
    rows_loaded = 0
    with pa.memory_map(str(specter_path), "r") as source:
        reader = pa.ipc.open_file(source)
        required_columns = {"paper_id", "embedding"}
        validate_arrow_schema(reader.schema, table_name="specter", columns=required_columns)
        embedding_type = reader.schema.field("embedding").type
        if int(embedding_type.list_size) <= 0:
            raise ValueError("specter Arrow embedding column must have positive dimension for graph subblocking")
        paper_id_index = reader.schema.get_field_index("paper_id")
        embedding_index = reader.schema.get_field_index("embedding")
        for batch_index in batch_indices:
            batch = reader.get_batch(batch_index)
            paper_values = batch.column(paper_id_index).to_pylist()
            record_batches_scanned += 1
            rows_scanned += len(paper_values)
            selected_indices = [
                row_index
                for row_index, paper_id in enumerate(paper_values)
                if paper_id is not None and str(paper_id) in keep_values
            ]
            if not selected_indices:
                continue
            if len(selected_indices) == len(paper_values):
                selected_paper_values = paper_values
                selected_embeddings = batch.column(embedding_index).to_pylist()
            else:
                take_indices = pa.array(selected_indices, type=pa.int64())
                selected_paper_values = batch.column(paper_id_index).take(take_indices).to_pylist()
                selected_embeddings = batch.column(embedding_index).take(take_indices).to_pylist()
            rows_loaded += len(selected_paper_values)
            for paper_id_value, embedding in zip(selected_paper_values, selected_embeddings, strict=True):
                paper_id = str(paper_id_value)
                if not paper_id:
                    raise ValueError("specter Arrow cannot contain empty paper_id values")
                if paper_id in embeddings:
                    raise ValueError(f"specter Arrow contains duplicate paper_id: {paper_id!r}")
                if embedding is None:
                    raise ValueError("specter Arrow cannot contain null embedding values")
                embeddings[paper_id] = np.asarray(embedding, dtype=np.float32)
    _add_load_metric(load_metrics, "specter_record_batches_scanned", record_batches_scanned)
    _add_load_metric(load_metrics, "specter_rows_scanned", rows_scanned)
    _add_load_metric(load_metrics, "specter_rows_loaded", rows_loaded)
    return embeddings


def _require_arrow_graph_subblocking_artifacts(paths: Mapping[str, Any]) -> dict[str, str]:
    return require_arrow_artifacts(
        paths,
        required_keys=(
            "signatures",
            "signatures_batch_index",
            "paper_authors",
            "paper_authors_batch_index",
            "specter",
            "specter_batch_index",
        ),
        context="Arrow graph subblocking",
        producer_hint=(
            "include signatures, paper_authors, specter, and matching raw-planner batch indexes; "
            "Arrow graph subblocking refuses filtered full scans"
        ),
    )


def _load_arrow_graph_subblocking_dataset(
    paths: Mapping[str, Any],
    signature_ids: Sequence[str],
    *,
    random_seed: int,
    load_metrics: dict[str, int],
) -> SimpleNamespace:
    paths = _require_arrow_graph_subblocking_artifacts(paths)
    signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
    signature_rows = _read_arrow_rows_by_values(
        paths["signatures"],
        paths["signatures_batch_index"],
        "signature_id",
        signature_ids,
        required_columns={
            "signature_id",
            "paper_id",
            "author_first",
            "author_middle",
            "author_affiliations",
            "author_position",
        },
        optional_columns=("author_orcid",),
        table_name="signatures",
        load_metrics=load_metrics,
    )
    signatures_by_id = _unique_rows_by_key(signature_rows, table_name="signatures", key_column="signature_id")
    missing_signature_ids = [signature_id for signature_id in signature_ids if signature_id not in signatures_by_id]
    if missing_signature_ids:
        raise ValueError(f"signatures Arrow is missing graph-subblocking signature ids: {missing_signature_ids[:10]}")

    paper_id_by_signature: dict[str, str] = {}
    for signature_id in signature_ids:
        raw_paper_id = signatures_by_id[signature_id].get("paper_id")
        if raw_paper_id is None or not str(raw_paper_id).strip():
            raise ValueError(
                "signatures Arrow cannot contain null/empty paper_id values for graph subblocking "
                f"(signature_id={signature_id})"
            )
        paper_id_by_signature[signature_id] = str(raw_paper_id)
    paper_ids = tuple(dict.fromkeys(paper_id_by_signature.values()))
    coauthors_by_paper = _coauthor_blocks_by_paper_from_arrow(paths, paper_ids, load_metrics=load_metrics)
    specter_embeddings = _specter_embeddings_from_arrow(paths, paper_ids, load_metrics=load_metrics)

    signature_objects: dict[str, SimpleNamespace] = {}
    for signature_id in signature_ids:
        row = signatures_by_id[signature_id]
        paper_id = paper_id_by_signature[signature_id]
        if row.get("author_position") is None:
            raise ValueError(
                "signatures Arrow cannot contain null author_position values for graph subblocking "
                f"(signature_id={signature_id})"
            )
        position = int(row["author_position"])
        affiliations = _string_tuple(row.get("author_affiliations"))
        coauthor_blocks = tuple(
            block for author_position, block in coauthors_by_paper.get(paper_id, ()) if int(author_position) != position
        )
        signature_objects[signature_id] = SimpleNamespace(
            signature_id=signature_id,
            paper_id=paper_id,
            author_info_first=_optional_str(row.get("author_first")),
            author_info_middle=_optional_str(row.get("author_middle")),
            author_info_first_normalized_without_apostrophe=None,
            author_info_middle_normalized_without_apostrophe=None,
            author_info_affiliations=affiliations,
            author_info_affiliations_n_grams=_affiliation_ngrams_from_raw_affiliations(affiliations),
            author_info_coauthor_blocks=coauthor_blocks,
            author_info_coauthors=None,
            author_info_orcid=_optional_str(row.get("author_orcid")),
            author_info_position=position,
        )

    return SimpleNamespace(
        signatures=signature_objects,
        papers={},
        specter_embeddings=specter_embeddings,
        random_seed=int(random_seed),
    )


def _coauthor_block_from_arrow_author_name(author_name_value: Any) -> str:
    normalized_name = normalize_text(str(author_name_value or "").strip())
    if not normalized_name:
        return ""
    return compute_block(normalized_name)


class ArrowGraphSubblockingFallback:
    """Lazy Arrow-backed callable for graph subblocking fallback."""

    def __init__(
        self,
        paths: Mapping[str, Any],
        signature_ids: Sequence[str],
        *,
        config: GraphSubblockingConfig | None = None,
        random_seed: int = 0,
    ) -> None:
        self.paths = dict(paths)
        self.signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
        self.config = config or GraphSubblockingConfig()
        self.random_seed = int(random_seed)
        self.stats: list[dict[str, Any]] = []
        self.load_metrics: dict[str, int] = {}
        self.load_seconds = 0.0
        self._dataset: SimpleNamespace | None = None

    def _load_signature_ids(self, signature_ids: Sequence[str]) -> None:
        signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
        start = time.perf_counter()
        self._dataset = _load_arrow_graph_subblocking_dataset(
            self.paths,
            signature_ids,
            random_seed=self.random_seed,
            load_metrics=self.load_metrics,
        )
        self.load_seconds = float(time.perf_counter() - start)

    def prepare(self, signature_groups: Iterable[Iterable[str]]) -> None:
        """Preload the memory-resident Arrow evidence store once."""

        groups = [tuple(dict.fromkeys(str(signature_id) for signature_id in group)) for group in signature_groups]
        signature_ids = tuple(dict.fromkeys(signature_id for group in groups for signature_id in group))
        self.load_metrics["prepared_group_count"] = len(groups)
        self.load_metrics["prepared_signature_count"] = len(signature_ids)
        if self._dataset is None:
            self._load_signature_ids(signature_ids)

    def _dataset_or_load(self, signature_ids: Sequence[str]) -> SimpleNamespace:
        if self._dataset is None:
            self._load_signature_ids(signature_ids)
        dataset = self._dataset
        if dataset is None:
            raise RuntimeError("Arrow graph subblocking dataset did not load")
        missing_signature_ids = [
            str(signature_id) for signature_id in signature_ids if str(signature_id) not in dataset.signatures
        ]
        if missing_signature_ids:
            raise ValueError(
                f"Arrow graph subblocking evidence is missing required signatures: {missing_signature_ids[:10]}"
            )
        return dataset

    def __call__(
        self,
        signature_ids: Iterable[str],
        anddata,
        target_subblock_size: int = 10000,
        compute_block_fn: Callable[[str], str] = compute_block,
    ) -> dict[str, list[str]]:
        del anddata
        signature_id_tuple = tuple(str(signature_id) for signature_id in signature_ids)
        return cluster_with_graph_fallback(
            signature_id_tuple,
            self._dataset_or_load(signature_id_tuple),
            target_subblock_size=target_subblock_size,
            compute_block_fn=compute_block_fn,
            config=self.config,
            stats=self.stats,
        )


def make_arrow_graph_subblocking_cluster_fn(
    paths: Mapping[str, Any],
    signature_ids: Sequence[str],
    *,
    config: GraphSubblockingConfig | None = None,
    random_seed: int = 0,
) -> ArrowGraphSubblockingFallback:
    """Return a lazy Arrow-backed graph fallback callable for subblocking."""

    return ArrowGraphSubblockingFallback(
        paths,
        signature_ids,
        config=config,
        random_seed=random_seed,
    )


class DatasetGraphSubblockingFallback:
    """ANDData-backed callable for graph subblocking fallback."""

    load_seconds = 0.0

    def __init__(self, *, config: GraphSubblockingConfig | None = None) -> None:
        self.config = config or GraphSubblockingConfig()
        self.stats: list[dict[str, Any]] = []

    def __call__(
        self,
        signature_ids: Iterable[str],
        anddata,
        target_subblock_size: int = 10000,
        compute_block_fn: Callable[[str], str] = compute_block,
    ) -> dict[str, list[str]]:
        return cluster_with_graph_fallback(
            signature_ids,
            anddata,
            target_subblock_size=target_subblock_size,
            compute_block_fn=compute_block_fn,
            config=self.config,
            stats=self.stats,
        )


def make_dataset_graph_subblocking_cluster_fn(
    *,
    config: GraphSubblockingConfig | None = None,
) -> DatasetGraphSubblockingFallback:
    """Return an ANDData-backed graph fallback callable for subblocking."""

    return DatasetGraphSubblockingFallback(config=config)


def signature_name_parts_for_subblocking(signature) -> tuple[str, str]:
    """Canonical (canonical_v2) first/middle for subblocking keys.

    Dash handling is uniform (D4): every dash-like character binds given-name
    compounds identically, replacing the retired ASCII-kept/non-ASCII-spilled
    legacy repair.
    """

    first = signature.author_info_first_normalized_without_apostrophe
    middle = signature.author_info_middle_normalized_without_apostrophe
    if first is not None and middle is not None:
        return first, middle
    # Rust preprocessing can defer normalized name fields; reconstruct with Python-equivalent logic.
    parts = canonicalize_name_parts(signature.author_info_first, signature.author_info_middle, None)
    return parts.first, parts.middle


def _signature_coauthor_blocks_for_specter(signature, anddata, compute_block_fn=compute_block) -> list[str]:
    coauthor_blocks = signature.author_info_coauthor_blocks
    if coauthor_blocks is not None:
        return list(coauthor_blocks)

    coauthors = signature.author_info_coauthors
    if coauthors is None:
        if signature.author_info_position is None:
            return []
        paper = anddata.papers.get(str(signature.paper_id))
        if paper is None:
            paper = anddata.papers.get(signature.paper_id)
        if paper is None:
            return []
        coauthors = [
            author.author_name for author in paper.authors if author.position != signature.author_info_position
        ]
    return [block for author in coauthors if (block := compute_block_fn(str(author or "")))]


def cluster_with_specter(signature_ids, anddata, target_subblock_size=10000, compute_block_fn=compute_block):
    """Helper function to cluster signature ids into subblocks using specter embeddings.
    Also tries to add simple embeddings of co-author blocks and affiliation n-grams.

    Args:
        signature_ids (list[str/int]): signature_ids
        anddata (s2and.data.ANDData): the anddata dataset
        target_subblock_size (int, optional): The desired maximum subblock size.
            If any of the resulting clusters are bigger than this, we chop them up randomly.
            Defaults to 10000.

    Returns:
        clusters: dict with keys as cluster_ids and values as list of signature_ids.
    """
    if len(signature_ids) == 0:
        return {}
    elif len(signature_ids) <= target_subblock_size:
        return {"0": signature_ids}

    # extract all the specter stuff in order of the signatures
    X_specter = np.array(
        [
            anddata.specter_embeddings.get(str(anddata.signatures[i].paper_id), np.zeros(SPECTER_DIM))
            for i in signature_ids
        ]
    )

    coauthor_values = [
        _signature_coauthor_blocks_for_specter(anddata.signatures[i], anddata, compute_block_fn) for i in signature_ids
    ]
    affiliation_values = [signature_affiliation_feature_keys(anddata.signatures[i]) for i in signature_ids]
    if not any(coauthor_values) or not any(affiliation_values):
        logger.warning(
            "SPECTER subblocking auxiliary features are unavailable; using embeddings only "
            "(signature_count=%d has_coauthor_evidence=%s has_affiliation_evidence=%s)",
            len(signature_ids),
            any(coauthor_values),
            any(affiliation_values),
        )
        X = X_specter
    else:
        coauthor_matrix = MultiLabelBinarizer(sparse_output=True).fit_transform(coauthor_values)
        affiliation_matrix = TfidfVectorizer(preprocessor=None, analyzer=lambda x: x).fit_transform(affiliation_values)
        if min(coauthor_matrix.shape) < SPECTER_DIM or min(affiliation_matrix.shape) < SPECTER_DIM:
            logger.warning(
                "SPECTER subblocking auxiliary matrices are too small; using embeddings only "
                "(signature_count=%d coauthor_shape=%s affiliation_shape=%s required_components=%d)",
                len(signature_ids),
                coauthor_matrix.shape,
                affiliation_matrix.shape,
                SPECTER_DIM,
            )
            X = X_specter
        else:
            coauthor_svd = TruncatedSVD(n_components=SPECTER_DIM).fit_transform(coauthor_matrix)
            affiliation_svd = TruncatedSVD(n_components=SPECTER_DIM).fit_transform(affiliation_matrix)
            X = X_specter + np.mean([coauthor_svd, affiliation_svd], axis=0)

    # how many subblocks do we want given this data and target subblock size?
    # should be at least 2 if we end up here otherwise there is no point
    num_desired_subblocks = min(
        int(np.ceil(len(signature_ids) / target_subblock_size)),
        len(signature_ids) - 1,
    )

    if np.any(X):
        g = genieclust.Genie(n_clusters=num_desired_subblocks, gini_threshold=0.01)
        labels = g.fit_predict(X)
    else:
        logger.warning(
            "SPECTER subblocking has no nonzero evidence; using deterministic capacity splitting "
            "(signature_count=%d target_subblock_size=%d)",
            len(signature_ids),
            target_subblock_size,
        )
        labels = np.zeros(len(signature_ids), dtype=int)

    subblocks = defaultdict(list)
    for sig_id, label in zip(signature_ids, labels, strict=True):
        subblocks[label].append(sig_id)
    # if any subblock is above the target size, just chop it up randomly into pieces that are below the target size
    seed_base = int(getattr(anddata, "random_seed", 0) or 0)
    for label, subblock in list(subblocks.items()):
        if len(subblock) > target_subblock_size:
            # Keep oversize split order deterministic for reproducible subblocking behavior.
            label_seed = seed_base + sum(ord(ch) for ch in str(label))
            random.Random(label_seed).shuffle(subblock)
            num_new_subblocks = int(np.ceil(len(subblock) / target_subblock_size))
            for i in range(num_new_subblocks):
                subblocks[f"{label}.{i}"] = subblock[i * target_subblock_size : (i + 1) * target_subblock_size]
            del subblocks[label]

    # assert that the subblocks has a complete clustering of the input signature_ids
    assert sum(len(subblock) for subblock in subblocks.values()) == len(signature_ids)

    return dict(subblocks)


def subdivide_helper(names, signature_ids, maximum_size, starting_k=2):
    """Helper function to subdivide a list of names into subblocks of maximum_size.
    Uses the first k letters of the names to subdivide. If the subblocks are still too big,
    then it will subdivide further by increasing k. It keeps going until each prefix bucket fits
    ``maximum_size`` or every available name character has been consumed. Remaining identical
    full-name groups are returned separately for the next fallback stage.

    Args:
        names (list of strings): the names to subdivide
        signature_ids (list[str/int]): the signature_ids corresponding to the names
        maximum_size (int): the maximum size of each subblock allowed
        starting_k (int, optional): The starting k to use for the first subdivision.
            Defaults to 2.

    Returns:
        output: dict with keys as subblock names and values as list of signature_ids
        output_cant_subdivide: dict with keys as subblock names and values as list of signature_ids
            that cant be subdivided further
    """
    # start with 2 letters only, then subdivide further to 3 letters, etc until the maximum_size is reached
    n_signature_ids = len(signature_ids)
    if n_signature_ids == 0:
        return {}, {}
    output = {}
    output_cant_subdivide = {}
    max_len = max([len(name) for name in names])
    for k in range(starting_k, max_len + 1):
        # note: any time we take something like XYZ and make it into XYZA, XYZB, ...
        # we will have some leftover ones that are just XYZ. those will end up in their own subblock
        names_up_to_k = np.array([name[0:k] for name in names])
        # use Series.value_counts to avoid the deprecated pd.value_counts API
        counts_up_to_k = pd.Series(names_up_to_k).value_counts()
        # find the ones that are a good size, and then take the rest and subdivide further
        good_size_flag = counts_up_to_k <= maximum_size
        counts_up_to_k_good_size = counts_up_to_k[good_size_flag]
        # store each subblock in output
        for name in counts_up_to_k_good_size.index:
            flag = names_up_to_k == name
            output[name] = signature_ids[flag]
        # take the rest and subdivide further
        bad_names = set(counts_up_to_k[counts_up_to_k > maximum_size].index)
        bad_size_flag = np.array([i[0:k] in bad_names for i in names], dtype=bool)
        names = names[bad_size_flag]
        signature_ids = signature_ids[bad_size_flag]
        if len(names) == 0:
            break
    # last ditch clean-up in case things didn't work out
    if len(names) > 0:
        for full_name in pd.Series(names).value_counts().index:
            flag = names == full_name
            output_cant_subdivide[f"full={full_name}"] = signature_ids[flag]
    # assert that the combo of the output and output_cant_subdivide is a complete clustering of the input signature_ids
    assert (
        sum(len(subblock) for subblock in output.values())
        + sum(len(subblock) for subblock in output_cant_subdivide.values())
        == n_signature_ids
    )
    return output, output_cant_subdivide


def _specter_labeled_subblock_stats(subblocks: dict[str, list[str]]) -> tuple[int, int]:
    """Count final subblocks whose lineage includes SPECTER fallback.

    Args:
        subblocks: Mapping from subblock key to signature IDs.

    Returns:
        Tuple of `(subblock_count, signature_count)` for keys containing `|specter=`.
    """
    specter_keys = [key for key in subblocks if "|specter=" in key]
    specter_signature_count = sum(len(subblocks[key]) for key in specter_keys)
    return len(specter_keys), specter_signature_count


def _subblock_merge_candidate_metadata(key: str, size: int) -> tuple[int, str, str | None, str | None, str | None]:
    key_parts = key.split("|")
    first_name = key_parts[0]
    middle_name = key_parts[1].partition("=")[2] if len(key_parts) > 1 and "=" in key_parts[1] else None
    if len(first_name) > 1:
        name_for_splits = first_name
    elif len(first_name) == 1 and middle_name is not None:
        name_for_splits = middle_name
    else:
        name_for_splits = None
    # canonical_v2: prefix-count lookups use the full canonical name string; the
    # legacy first-token reduction for multi-token names is retired with the
    # regenerated ORCID prefix-count artifact.
    lookup = name_for_splits
    return size, first_name, middle_name, name_for_splits, lookup


def _sorted_subblock_merge_candidates(
    output: dict[str, list[str]],
    maximum_size: int,
    first_k_letter_counts_sorted: Mapping[str, Mapping[str, int]],
) -> list[tuple[tuple[str, str], float]]:
    """Return legacy subblock merge candidates with key metadata parsed once."""

    small_enough_keys = [key for key, value in output.items() if len(value) < maximum_size]
    metadata: dict[str, tuple[int, str, str | None, str, str | None]] = {}
    mergeable_keys: list[str] = []
    for key in small_enough_keys:
        size, first_name, middle_name, name_for_splits, lookup = _subblock_merge_candidate_metadata(
            key,
            len(output[key]),
        )
        if name_for_splits is None:
            continue
        metadata[key] = (size, first_name, middle_name, name_for_splits, lookup)
        mergeable_keys.append(key)
    candidates: list[tuple[tuple[str, str], float]] = []
    for pair in combinations(mergeable_keys, 2):
        size_1, first_name_1, middle_name_1, name_for_splits_1, lookup_1 = metadata[pair[0]]
        size_2, first_name_2, middle_name_2, name_for_splits_2, lookup_2 = metadata[pair[1]]
        if size_1 + size_2 > maximum_size:
            continue
        both_multi_letter = len(first_name_1) > 1 and len(first_name_2) > 1
        both_single_letter_with_middle = (
            len(first_name_1) == 1
            and len(first_name_2) == 1
            and middle_name_1 is not None
            and middle_name_2 is not None
        )
        if not both_multi_letter and not both_single_letter_with_middle:
            continue

        if name_for_splits_1 == name_for_splits_2:
            if middle_name_1 is not None and middle_name_2 is not None:
                score = 0
                for i in range(1, min(len(middle_name_1), len(middle_name_2)) + 1):
                    if middle_name_1[:i] == middle_name_2[:i]:
                        score = i
            else:
                score = 0
            candidates.append((pair, 1e10 + score))
        elif same_prefix_tokens(name_for_splits_1, name_for_splits_2):
            score = min(len(name_for_splits_1), len(name_for_splits_2))
            candidates.append((pair, 1e5 + score))
        elif lookup_1 is not None and lookup_2 is not None:
            pair_count = _orcid_prefix_pair_count(first_k_letter_counts_sorted, lookup_1, lookup_2)
            if pair_count is not None:
                candidates.append((pair, pair_count))
    return sorted(candidates, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)


def _normalize_unique_subblocking_signature_ids(signature_ids: Iterable[Any]) -> list[str]:
    """Normalize subblocking IDs and reject duplicate identities."""

    normalized_signature_ids = [str(signature_id) for signature_id in signature_ids]
    seen_signature_ids: set[str] = set()
    for signature_id in normalized_signature_ids:
        if signature_id in seen_signature_ids:
            raise ValueError(f"Subblocking signature_ids must be unique after string coercion: {signature_id!r}")
        seen_signature_ids.add(signature_id)
    return normalized_signature_ids


def _make_subblocks_with_telemetry_arrow_rust(
    arrow_paths: Mapping[str, Any],
    signature_ids,
    maximum_size=15000,
    first_k_letter_counts_sorted=FIRST_K_LETTER_COUNTS,
    graph_subblocking_config: GraphSubblockingConfig | None = None,
    graph_subblocking_random_seed: int = 0,
    use_orcid_subblocking: bool = True,
):
    """Run native Rust graph subblocking with signature rows loaded from Arrow."""

    from s2and.runtime import load_s2and_rust_extension

    first_k_letter_counts_sorted = _resolved_orcid_prefix_counts(first_k_letter_counts_sorted)
    rust_prefix_counts = {prefix: dict(nested_counts) for prefix, nested_counts in first_k_letter_counts_sorted.items()}
    rust_make_subblocks = load_s2and_rust_extension().make_subblocks_with_telemetry_arrow_native_graph
    subblocks, telemetry = rust_make_subblocks(
        dict(arrow_paths),
        _normalize_unique_subblocking_signature_ids(signature_ids),
        int(maximum_size),
        rust_prefix_counts,
        graph_subblocking_config,
        int(graph_subblocking_random_seed),
        bool(use_orcid_subblocking),
    )
    return {str(key): list(values) for key, values in dict(subblocks).items()}, dict(telemetry)


def make_subblocks_with_telemetry(
    signature_ids,
    anddata,
    maximum_size=15000,
    first_k_letter_counts_sorted=FIRST_K_LETTER_COUNTS,
    compute_block_fn=compute_block,
    specter_cluster_fn=None,
    use_orcid_subblocking: bool = True,
):
    """Split signature IDs into subblocks and report how the partition was built.

    This function takes a list of signature IDs and splits them into subblocks of maximum_size.
    It first splits by first name initial letter. Then it recursively splits any subblocks larger than
    maximum_size using middle names and the SPECTER clustering algorithm. Finally, it merges any subblocks
    smaller than maximum_size that share name attributes.

    There is an optional ORCID repair pass: when `use_orcid_subblocking` is true, whole subblocks
    that contain the same normalized ORCID are merged only when the combined subblocks fit within
    `maximum_size`.

    Args:
        signature_ids (list[str/int]): List of signature IDs.
        anddata (s2and.data.ANDData): Contains name attribute data for the signatures.
        maximum_size (int): Maximum size of any subblock. Default is 15000.
        first_k_letter_counts_sorted (dict): Dictionary of name letter counts, used for merging subblocks.
            Already included in the package. Default is FIRST_K_LETTER_COUNTS, which is imported
            in this file.
        use_orcid_subblocking (bool): Whether to run the final same-ORCID co-location pass.

    Returns:
        tuple[dict, dict]: `(subblocks, telemetry)` where `subblocks` is the final partition and
        `telemetry` reports first-name dead-ends, SPECTER fallback usage, and final SPECTER-labeled
        subblock counts/signatures.
    """
    logger.info("Beginning subblocking...")
    first_k_letter_counts_sorted = _resolved_orcid_prefix_counts(first_k_letter_counts_sorted)
    signature_ids = np.asarray(_normalize_unique_subblocking_signature_ids(signature_ids))
    first_middle_names = [signature_name_parts_for_subblocking(anddata.signatures[i]) for i in signature_ids]
    first_names = np.array([name_parts[0] for name_parts in first_middle_names])
    middle_names = np.array([name_parts[1] for name_parts in first_middle_names])

    # set aside those that are only 1 letter long for a different treatment
    single_letter_first_names_flag = np.array([len(first_name) <= 1 for first_name in first_names])
    telemetry = {
        "maximum_size": int(maximum_size),
        "input_signature_count": int(len(signature_ids)),
        "single_letter_first_name_signature_count": int(np.sum(single_letter_first_names_flag)),
        "multi_letter_first_name_signature_count": int(np.sum(~single_letter_first_names_flag)),
        "first_name_dead_end_block_count": 0,
        "first_name_dead_end_signature_count": 0,
        "specter_fallback_candidate_block_count": 0,
        "specter_fallback_candidate_signature_count": 0,
        "specter_non_invoked_candidate_block_count": 0,
        "specter_non_invoked_candidate_signature_count": 0,
        "specter_invocation_count": 0,
        "specter_input_signature_count": 0,
        "pre_merge_subblock_count": 0,
        "pre_merge_specter_labeled_subblock_count": 0,
        "pre_merge_specter_labeled_signature_count": 0,
        "orcid_subblocking_enabled": bool(use_orcid_subblocking),
        "orcid_merge_skipped_due_to_capacity_count": 0,
        "orcid_merge_skipped_due_to_capacity_signature_count": 0,
        "final_subblock_count": 0,
        "final_specter_labeled_subblock_count": 0,
        "final_specter_labeled_signature_count": 0,
    }

    # first letter is
    first_letter = "?"  # could happen if all the first names are missing
    for name in first_names:
        if len(name) > 0:
            first_letter = name[0]
            break

    # first pass through the more-than-one-letter first names
    logger.info("First pass through the more-than-one-letter first names")
    output, output_cant_subdivide = subdivide_helper(
        first_names[~single_letter_first_names_flag], signature_ids[~single_letter_first_names_flag], maximum_size
    )
    telemetry["first_name_dead_end_block_count"] = int(len(output_cant_subdivide))
    telemetry["first_name_dead_end_signature_count"] = int(sum(len(v) for v in output_cant_subdivide.values()))

    # for each block in output_cant_subdivide, we need to subdivide it further using middle names
    if len(output_cant_subdivide) > 0:
        logger.info(
            "Subdividing the more-than-one-letter first names that could not be subdivided further using middle names"
        )
    output_for_specter = {}
    for key, sig_ids_loop in output_cant_subdivide.items():
        middle_names_loop = np.array(
            [signature_name_parts_for_subblocking(anddata.signatures[i])[1] for i in sig_ids_loop]
        )
        output_loop, output_cant_subdivide_loop = subdivide_helper(
            middle_names_loop, sig_ids_loop, maximum_size, starting_k=1
        )
        # the key in output loop should be pre-pended by the loop key
        for key_loop in list(output_loop.keys()):
            output_loop[key + "|middle=" + str(key_loop)] = output_loop.pop(key_loop)
        for key_loop in list(output_cant_subdivide_loop.keys()):
            output_cant_subdivide_loop[key + "|middle=" + str(key_loop)] = output_cant_subdivide_loop.pop(key_loop)
        # now update the output
        output.update(output_loop)
        output_for_specter.update(output_cant_subdivide_loop)

    # deal with the single (or zero) letter first names
    if len(first_names[single_letter_first_names_flag]) <= maximum_size:
        if np.mean(single_letter_first_names_flag) > 0:
            output[first_letter] = signature_ids[single_letter_first_names_flag]
    else:
        logger.info("Subdividing the too-big single letter subblock using middle names")
        output_single_letter_first_name, output_cant_subdivide_single_letter_first_name = subdivide_helper(
            middle_names[single_letter_first_names_flag],
            signature_ids[single_letter_first_names_flag],
            maximum_size,
            starting_k=1,
        )
        # modify the key to indicate what this is
        for key in list(output_single_letter_first_name.keys()):
            output_single_letter_first_name[f"{first_letter}|middle=" + str(key)] = output_single_letter_first_name.pop(
                key
            )
        for key in list(output_cant_subdivide_single_letter_first_name.keys()):
            output_cant_subdivide_single_letter_first_name[f"{first_letter}|middle=" + str(key)] = (
                output_cant_subdivide_single_letter_first_name.pop(key)
            )
        output.update(output_single_letter_first_name)
        output_for_specter.update(
            output_cant_subdivide_single_letter_first_name
        )  # since it already went through the middle name step

    telemetry["specter_fallback_candidate_block_count"] = int(len(output_for_specter))
    telemetry["specter_fallback_candidate_signature_count"] = int(sum(len(v) for v in output_for_specter.values()))

    # for each subblock that STILL can't be subdivided, we must use SPECTER
    # which also does totally random sub-blocking in case things went awry
    if len(output_for_specter) > 0:
        logger.info(
            "Subdividing the subblocks that could not be subdivided via middle names using SPECTER "
            "(and random subblocking)"
        )
    fallback_cluster_fn = specter_cluster_fn or cluster_with_specter
    fallback_signature_groups = [
        tuple(str(signature_id) for signature_id in sig_ids_loop)
        for sig_ids_loop in output_for_specter.values()
        if len(sig_ids_loop) > maximum_size
    ]
    prepare_fallback = getattr(fallback_cluster_fn, "prepare", None)
    if callable(prepare_fallback) and fallback_signature_groups:
        prepare_fallback(fallback_signature_groups)
    for key, sig_ids_loop in output_for_specter.items():
        output_loop = {}
        if len(sig_ids_loop) <= maximum_size:
            # edge case where the subblock is already fine
            telemetry["specter_non_invoked_candidate_block_count"] += 1
            telemetry["specter_non_invoked_candidate_signature_count"] += int(len(sig_ids_loop))
            output_loop[key] = sig_ids_loop
        else:
            telemetry["specter_invocation_count"] += 1
            telemetry["specter_input_signature_count"] += int(len(sig_ids_loop))
            specter_clustering = fallback_cluster_fn(
                sig_ids_loop,
                anddata,
                target_subblock_size=maximum_size,
                compute_block_fn=compute_block_fn,
            )
            # prepend the key to the specter_clustering keys
            for key_loop in list(specter_clustering.keys()):
                output_loop[key + "|specter=" + str(key_loop)] = specter_clustering.pop(key_loop)
        output.update(output_loop)

    pre_merge_specter_subblock_count, pre_merge_specter_signature_count = _specter_labeled_subblock_stats(output)
    telemetry["pre_merge_subblock_count"] = int(len(output))
    telemetry["pre_merge_specter_labeled_subblock_count"] = int(pre_merge_specter_subblock_count)
    telemetry["pre_merge_specter_labeled_signature_count"] = int(pre_merge_specter_signature_count)

    """
    Merging too small subblocks back up to maximum_size
    If we found that the subblock Jame* was too big, afterwards some of the subblocks
    like James*, Jamen*, Jamek* etc may be too small and could be joined again while
    still being below the maximum size.
    
    This is done by looking at all the subblocks that are small enough, and then
    checking (a) they are plausible to be merged (b) their join is small enough.
    
    First step is to find candidates for merging.
    """
    logger.info("Starting to merge subblocks. First step is to find candidates for merging.")
    small_enough_pairs_sorted = _sorted_subblock_merge_candidates(
        output,
        maximum_size,
        first_k_letter_counts_sorted,
    )
    # now we go down the list and merge until we reach merged subblocks not above maximum size
    # it's possible that when we merge subblock A and B, the resulting subblock is still below thresh
    # and then it's legal to merge A/B with C, so we have to keep track of all that
    merging_log = {}  # what we will actually merge after we're done. cluster id -> set of keys
    inverse_merging_log = {}  # need this to see if things are in the same subblock already
    cluster_id = 0
    # we'll use this to see how many tuples a key appears in
    # and if a proposed merge appears in more than one
    # then we have a problem and it shouldn't occur
    logger.info(f"Number of small enough pairs to consider for subblock merging: {len(small_enough_pairs_sorted)}")
    logger.info("Merging subblocks...")
    for pair, _ in small_enough_pairs_sorted:
        # see where both parts of the pair are in the merging log
        pair_1_cluster_id = inverse_merging_log.get(pair[0], None)
        pair_2_cluster_id = inverse_merging_log.get(pair[1], None)
        # if neither are in the log, then we can just add them to it
        if pair_1_cluster_id is None and pair_2_cluster_id is None:
            merging_log[cluster_id] = set(pair)
            inverse_merging_log[pair[0]] = cluster_id
            inverse_merging_log[pair[1]] = cluster_id
            cluster_id += 1
        # if both are in the merging log but they have the SAME cluster id, then we don't need to do anything
        elif pair_1_cluster_id is not None and pair_2_cluster_id is not None and pair_1_cluster_id == pair_2_cluster_id:
            continue
        else:
            # if both are in the merging log but they have DIFFERENT cluster ids
            # then we should check if their clusters can be joined legally
            if (
                pair_1_cluster_id is not None
                and pair_2_cluster_id is not None
                and pair_1_cluster_id != pair_2_cluster_id
            ):
                proposed_cluster = merging_log[pair_1_cluster_id].union(merging_log[pair_2_cluster_id])
            # if only one is in the merging log, then we should check if the other can be added to it legally
            elif pair_1_cluster_id is not None and pair_2_cluster_id is None:
                proposed_cluster = merging_log[pair_1_cluster_id].union(set(pair))
            # and vice versa
            elif pair_1_cluster_id is None and pair_2_cluster_id is not None:
                proposed_cluster = merging_log[pair_2_cluster_id].union(set(pair))
            else:
                raise ValueError("This should never happen")
            size_of_proposed = sum([len(output[k]) for k in proposed_cluster])
            if size_of_proposed <= maximum_size:
                if pair_1_cluster_id is not None:
                    merging_log[pair_1_cluster_id] = proposed_cluster
                    if pair_2_cluster_id is not None:
                        del merging_log[pair_2_cluster_id]
                    for k in proposed_cluster:
                        inverse_merging_log[k] = pair_1_cluster_id
                else:
                    merging_log[pair_2_cluster_id] = proposed_cluster
                    if pair_1_cluster_id is not None:
                        del merging_log[pair_1_cluster_id]
                    for k in proposed_cluster:
                        inverse_merging_log[k] = pair_2_cluster_id

    # double check that nothing weird happened: each key should only appear in one subblock
    counter_of_keys = defaultdict(int)
    for keys_to_merge in merging_log.values():
        for k in keys_to_merge:
            counter_of_keys[k] += 1

    if not all(value == 1 for value in counter_of_keys.values()):
        raise RuntimeError("Subblocking merge invariant failed: a key belongs to multiple merge components")

    # now perform the actual merges
    for merge_cluster_id in sorted(merging_log):
        # Keep merged member ordering deterministic across processes.
        keys_to_merge = sorted(merging_log[merge_cluster_id])
        key_of_keys = ", ".join(keys_to_merge)
        signature_ids_stacked = np.hstack([output[k] for k in keys_to_merge])
        output[key_of_keys] = signature_ids_stacked
        # delete what was merged
        for k in keys_to_merge:
            del output[k]

    # values in output should be lists
    for k in list(output.keys()):
        output[k] = list(output[k])

    sig_id_to_subblock_id = {}
    for subblock_id, sig_ids in output.items():
        for sig_id in sig_ids:
            sig_id_to_subblock_id[sig_id] = subblock_id

    if use_orcid_subblocking:
        # Final ORCID repair pass. It only coarsens existing subblocks: extracting only
        # same-ORCID signatures from a source subblock can fragment otherwise good name
        # buckets. If the whole set of subblocks containing an ORCID cannot fit under
        # the capacity cap, leave the split in place and report it.
        orcid_to_sig_ids = defaultdict(list)
        for sig_id in sig_id_to_subblock_id:
            orcid = normalize_orcid_for_subblocking(getattr(anddata.signatures[sig_id], "author_info_orcid", None))
            if orcid is not None:
                orcid_to_sig_ids[orcid].append(sig_id)
        subblock_ids = list(output)
        subblock_index = {subblock_id: index for index, subblock_id in enumerate(subblock_ids)}
        parent = list(range(len(subblock_ids)))
        subblock_component_size = [1] * len(subblock_ids)

        def find_subblock_root(index: int) -> int:
            root = index
            while parent[root] != root:
                root = parent[root]
            while parent[index] != index:
                parent_index = parent[index]
                parent[index] = root
                index = parent_index
            return root

        def union_subblocks(left: str, right: str) -> None:
            left_root = find_subblock_root(subblock_index[left])
            right_root = find_subblock_root(subblock_index[right])
            if left_root != right_root:
                if subblock_component_size[left_root] < subblock_component_size[right_root]:
                    left_root, right_root = right_root, left_root
                parent[right_root] = left_root
                subblock_component_size[left_root] += subblock_component_size[right_root]

        orcid_to_subblock_ids: dict[str, list[str]] = {}
        for orcid, orcid_sig_ids in orcid_to_sig_ids.items():
            seen_subblocks: set[str] = set()
            unique_subblock_ids: list[str] = []
            for sig_id in orcid_sig_ids:
                subblock_id = sig_id_to_subblock_id[sig_id]
                if subblock_id in seen_subblocks:
                    continue
                seen_subblocks.add(subblock_id)
                unique_subblock_ids.append(subblock_id)
            if len(unique_subblock_ids) <= 1:
                continue
            orcid_to_subblock_ids[orcid] = unique_subblock_ids
            first_subblock_id = unique_subblock_ids[0]
            for subblock_id in unique_subblock_ids[1:]:
                union_subblocks(first_subblock_id, subblock_id)

        components_by_root: dict[int, list[str]] = defaultdict(list)
        component_roots: list[int] = []
        seen_roots: set[int] = set()
        for subblock_id in subblock_ids:
            root = find_subblock_root(subblock_index[subblock_id])
            components_by_root[root].append(subblock_id)
            if root not in seen_roots:
                seen_roots.add(root)
                component_roots.append(root)

        skipped_orcid_counts_by_root: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for orcid, unique_subblock_ids in orcid_to_subblock_ids.items():
            root = find_subblock_root(subblock_index[unique_subblock_ids[0]])
            skipped_orcid_counts_by_root[root].append((orcid, len(orcid_to_sig_ids[orcid])))

        merge_actions: list[tuple[str, list[str], list[str]]] = []
        for root in component_roots:
            unique_subblock_ids = components_by_root[root]
            if len(unique_subblock_ids) <= 1:
                continue
            unique_subblock_ids = sorted(
                unique_subblock_ids,
                key=lambda x: (x.count("specter") * 10 + x.count("|"), x),
            )
            total_subblock_sig_count = sum(len(output[subblock_id]) for subblock_id in unique_subblock_ids)
            if total_subblock_sig_count > maximum_size:
                for orcid, total_orcid_sig_count in skipped_orcid_counts_by_root[root]:
                    telemetry["orcid_merge_skipped_due_to_capacity_count"] += 1
                    telemetry["orcid_merge_skipped_due_to_capacity_signature_count"] += int(total_orcid_sig_count)
                    logger.warning(
                        "Skipping ORCID merge for %s across %d subblocks; whole-subblock merge exceeds maximum_size=%d",
                        orcid,
                        len(unique_subblock_ids),
                        maximum_size,
                    )
                continue
            key_of_keys = ", ".join(unique_subblock_ids)
            signature_ids_stacked = []
            for subblock_id in unique_subblock_ids:
                signature_ids_stacked.extend(output[subblock_id])
            merge_actions.append((key_of_keys, signature_ids_stacked, unique_subblock_ids))

        for key_of_keys, signature_ids_stacked, unique_subblock_ids in merge_actions:
            for subblock_id in unique_subblock_ids:
                del output[subblock_id]
            output[key_of_keys] = signature_ids_stacked

    # let's assert that we have done a complete partition. Multiset equality, not
    # set equality: a signature_id duplicated across subblocks would pass a set
    # check but bind nondeterministically in the downstream ORCID merge (Python
    # dict insertion order vs Rust last-subblock-wins), silently diverging the
    # two implementations.
    output_signature_ids = sorted(str(signature_id) for k in output for signature_id in output[k])
    expected_signature_ids = sorted(str(signature_id) for signature_id in signature_ids)
    if output_signature_ids != expected_signature_ids:
        raise RuntimeError("Subblocking output is not a complete one-to-one partition of signature_ids")

    # before the end, makes sure everything is a standard list
    for k in list(output.keys()):
        output[k] = list(output[k])

    average_subblock_length = np.mean([len(output[k]) for k in output])
    logger.info(
        f"Done subblocking. There are {len(output)} subblocks with an average of "
        f"{average_subblock_length} signatures each."
    )
    final_specter_subblock_count, final_specter_signature_count = _specter_labeled_subblock_stats(output)
    telemetry["final_subblock_count"] = int(len(output))
    telemetry["final_specter_labeled_subblock_count"] = int(final_specter_subblock_count)
    telemetry["final_specter_labeled_signature_count"] = int(final_specter_signature_count)
    return output, telemetry


def make_subblocks(
    signature_ids,
    anddata,
    maximum_size=15000,
    first_k_letter_counts_sorted=FIRST_K_LETTER_COUNTS,
    compute_block_fn=compute_block,
    specter_cluster_fn=None,
    use_orcid_subblocking: bool = True,
):
    """Split signature IDs into subblocks based on name attributes.

    This is the existing production-facing wrapper around
    `make_subblocks_with_telemetry(...)` and preserves the original return type.

    Args:
        signature_ids (list[str/int]): List of signature IDs.
        anddata (s2and.data.ANDData): Contains name attribute data for the signatures.
        maximum_size (int): Maximum size of any subblock. Default is 15000.
        first_k_letter_counts_sorted (dict): Prefix-count priors used when merging small subblocks.
        use_orcid_subblocking (bool): Whether to run the final same-ORCID co-location pass.

    Returns:
        dict: Dictionary of subblock keys mapped to lists of signature IDs.
    """
    output, _ = make_subblocks_with_telemetry(
        signature_ids,
        anddata,
        maximum_size=maximum_size,
        first_k_letter_counts_sorted=first_k_letter_counts_sorted,
        compute_block_fn=compute_block_fn,
        specter_cluster_fn=specter_cluster_fn,
        use_orcid_subblocking=use_orcid_subblocking,
    )
    return output
