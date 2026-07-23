"""FeatureBlock in-memory contract types and validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

FEATURE_BLOCK_SCHEMA_VERSION = "feature_block_v2"
FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION = "feature_block_arrow_v2"


def normalize_cluster_seed_disallow_pairs(
    pairs: Iterable[tuple[Any, Any]],
    *,
    valid_signature_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return canonical undirected disallow pairs after schema validation."""

    valid_signature_id_set = None if valid_signature_ids is None else {str(value) for value in valid_signature_ids}
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in pairs:
        left_id = str(left)
        right_id = str(right)
        if not left_id or not right_id:
            raise ValueError("cluster seed disallow pairs cannot contain empty signature ids")
        if left_id == right_id:
            raise ValueError(f"cluster seed disallow pair cannot be a self-pair: {left_id!r}")
        if valid_signature_id_set is not None:
            missing = sorted({left_id, right_id}.difference(valid_signature_id_set))
            if missing:
                raise ValueError(f"cluster seed disallow pair contains signatures missing from FeatureBlock: {missing}")
        pair = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def filter_cluster_seed_disallows_for_signature_subset(
    pairs: Iterable[tuple[Any, Any]],
    signature_ids: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    """Keep only disallow pairs whose endpoints are both inside a sliced signature set."""

    signature_id_set = {str(signature_id) for signature_id in signature_ids}
    filtered: list[tuple[str, str]] = []
    for left, right in pairs:
        left_id = str(left)
        right_id = str(right)
        if left_id in signature_id_set and right_id in signature_id_set:
            filtered.append((left_id, right_id))
    return normalize_cluster_seed_disallow_pairs(filtered)


def _strict_string_tuple(value: Any, *, field_name: str, skip_none: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise ValueError(f"{field_name} must be a sequence, not a scalar string")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    items: list[str] = []
    for item in value:
        if item is None:
            if skip_none:
                continue
            raise ValueError(f"{field_name} cannot contain null values")
        items.append(str(item))
    return tuple(items)


@dataclass(frozen=True)
class FeatureBlockSignature:
    """One signature row in the inference contract."""

    signature_id: str
    paper_id: str
    author_first: str | None
    author_middle: str | None
    author_last: str | None
    author_suffix: str | None
    author_affiliations: tuple[str, ...]
    author_orcid: str | None
    author_position: int | None
    author_block: str | None = None
    author_email: str | None = None
    source_author_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "signature_id", str(self.signature_id))
        object.__setattr__(self, "paper_id", str(self.paper_id))
        object.__setattr__(
            self,
            "author_affiliations",
            _strict_string_tuple(self.author_affiliations, field_name="FeatureBlockSignature.author_affiliations"),
        )
        object.__setattr__(
            self,
            "source_author_ids",
            _strict_string_tuple(self.source_author_ids, field_name="FeatureBlockSignature.source_author_ids"),
        )
        if not self.signature_id:
            raise ValueError("FeatureBlockSignature.signature_id must be non-empty")
        if not self.paper_id:
            raise ValueError(f"FeatureBlockSignature.paper_id must be non-empty for {self.signature_id!r}")
        if self.author_position is not None:
            object.__setattr__(self, "author_position", int(self.author_position))


@dataclass(frozen=True)
class FeatureBlockPaper:
    """One paper row in the inference contract."""

    paper_id: str
    title: str | None
    abstract: str | None
    venue: str | None
    journal_name: str | None
    year: int | None
    predicted_language: str | None = None
    is_reliable: bool | None = None
    language_reliability: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", str(self.paper_id))
        if not self.paper_id:
            raise ValueError("FeatureBlockPaper.paper_id must be non-empty")
        if self.year is not None:
            object.__setattr__(self, "year", int(self.year))
        predicted_language = None if self.predicted_language is None else str(self.predicted_language)
        is_reliable = _optional_bool(self.is_reliable, field_name="FeatureBlockPaper.is_reliable")
        language_reliability = _optional_float(
            self.language_reliability,
            field_name="FeatureBlockPaper.language_reliability",
        )
        object.__setattr__(self, "predicted_language", predicted_language)
        object.__setattr__(
            self,
            "is_reliable",
            is_reliable,
        )
        object.__setattr__(
            self,
            "language_reliability",
            language_reliability,
        )
        if language_reliability is not None:
            if not np.isfinite(language_reliability):
                raise ValueError("FeatureBlockPaper.language_reliability must be finite")
            if not 0.0 <= language_reliability <= 1.0:
                raise ValueError("FeatureBlockPaper.language_reliability must be in [0.0, 1.0]")
            if is_reliable is False and language_reliability != 0.0:
                raise ValueError(
                    "FeatureBlockPaper.language_reliability must be 0.0 when FeatureBlockPaper.is_reliable is false"
                )
        if predicted_language is None:
            if is_reliable is not None or language_reliability is not None:
                raise ValueError(
                    "FeatureBlockPaper.is_reliable and FeatureBlockPaper.language_reliability "
                    "require FeatureBlockPaper.predicted_language"
                )
        else:
            if not predicted_language.strip():
                raise ValueError("FeatureBlockPaper.predicted_language must be non-empty")
            if is_reliable is None:
                raise ValueError("FeatureBlockPaper.predicted_language requires FeatureBlockPaper.is_reliable")
            if language_reliability is None:
                raise ValueError("FeatureBlockPaper.predicted_language requires FeatureBlockPaper.language_reliability")


@dataclass(frozen=True)
class FeatureBlockPaperAuthor:
    """One paper-author child row in the inference contract."""

    paper_id: str
    position: int
    author_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "paper_id", str(self.paper_id))
        object.__setattr__(self, "position", int(self.position))
        object.__setattr__(self, "author_name", str(self.author_name))
        if not self.paper_id:
            raise ValueError("FeatureBlockPaperAuthor.paper_id must be non-empty")


@dataclass(frozen=True)
class FeatureBlockSignatureOrder:
    """Deterministic mini-block signature order for numeric linker arrays."""

    signature_ids: tuple[str, ...]
    query_signature_ids: tuple[str, ...] = ()
    schema_version: str = FEATURE_BLOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        signature_ids = tuple(str(value) for value in self.signature_ids)
        query_signature_ids = tuple(str(value) for value in self.query_signature_ids)
        if len(set(signature_ids)) != len(signature_ids):
            raise ValueError("FeatureBlockSignatureOrder.signature_ids must be unique")
        missing_queries = sorted(set(query_signature_ids) - set(signature_ids))
        if missing_queries:
            raise ValueError(f"query_signature_ids are missing from signature_ids: {missing_queries}")
        object.__setattr__(self, "signature_ids", signature_ids)
        object.__setattr__(self, "query_signature_ids", query_signature_ids)

    @property
    def signature_id_to_index(self) -> dict[str, int]:
        """Return this order as the numeric linker index map."""

        return {signature_id: index for index, signature_id in enumerate(self.signature_ids)}


@dataclass(frozen=True)
class FeatureBlock:
    """Typed inference inputs, smaller than full `ANDData`."""

    signatures: tuple[FeatureBlockSignature, ...]
    papers: tuple[FeatureBlockPaper, ...] = ()
    paper_authors: tuple[FeatureBlockPaperAuthor, ...] = ()
    cluster_seeds_require: tuple[tuple[str, str], ...] = ()
    cluster_seeds_disallow: tuple[tuple[str, str], ...] = ()
    query_signature_ids: tuple[str, ...] = ()
    specter_paper_ids: tuple[str, ...] = ()
    specter_embeddings: np.ndarray | None = None
    schema_version: str = FEATURE_BLOCK_SCHEMA_VERSION
    _signature_ids_cache: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _signature_order_cache: FeatureBlockSignatureOrder = field(init=False, repr=False, compare=False)
    _signature_id_to_index_cache: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        signature_ids = tuple(signature.signature_id for signature in self.signatures)
        object.__setattr__(self, "_signature_ids_cache", signature_ids)
        if len(set(signature_ids)) != len(signature_ids):
            raise ValueError("FeatureBlock signatures must have unique signature_id values")
        paper_ids = tuple(paper.paper_id for paper in self.papers)
        if len(set(paper_ids)) != len(paper_ids):
            raise ValueError("FeatureBlock papers must have unique paper_id values")
        seen_paper_author_positions: set[tuple[str, int]] = set()
        for author in self.paper_authors:
            key = (str(author.paper_id), int(author.position))
            if key in seen_paper_author_positions:
                raise ValueError(f"FeatureBlock paper_authors contains duplicate (paper_id, position): {key!r}")
            seen_paper_author_positions.add(key)

        signature_id_set = set(signature_ids)
        query_signature_ids = tuple(str(value) for value in self.query_signature_ids)
        missing_queries = sorted(set(query_signature_ids) - signature_id_set)
        if missing_queries:
            raise ValueError(f"query_signature_ids are missing from FeatureBlock signatures: {missing_queries}")
        object.__setattr__(self, "query_signature_ids", query_signature_ids)
        signature_order = FeatureBlockSignatureOrder(
            signature_ids=signature_ids,
            query_signature_ids=query_signature_ids,
        )
        object.__setattr__(self, "_signature_order_cache", signature_order)
        object.__setattr__(
            self,
            "_signature_id_to_index_cache",
            MappingProxyType(signature_order.signature_id_to_index),
        )

        require_pair_list: list[tuple[str, str]] = []
        seen_require_signature_ids: set[str] = set()
        for signature_id, component_id in self.cluster_seeds_require:
            signature_key = str(signature_id)
            component_key = str(component_id)
            if not signature_key:
                raise ValueError("cluster_seeds_require cannot contain empty signature_id values")
            if not component_key:
                raise ValueError(f"cluster_seeds_require cannot contain empty component_id values: {signature_key!r}")
            if signature_key in seen_require_signature_ids:
                raise ValueError(f"cluster_seeds_require contains duplicate signature_id: {signature_key!r}")
            seen_require_signature_ids.add(signature_key)
            require_pair_list.append((signature_key, component_key))
        require_pairs = tuple(require_pair_list)
        missing_require = sorted(
            signature_id for signature_id, _component_id in require_pairs if signature_id not in signature_id_set
        )
        if missing_require:
            raise ValueError(f"cluster_seeds_require contains signatures missing from FeatureBlock: {missing_require}")
        object.__setattr__(self, "cluster_seeds_require", require_pairs)

        disallow_pairs = normalize_cluster_seed_disallow_pairs(
            self.cluster_seeds_disallow,
            valid_signature_ids=signature_id_set,
        )
        object.__setattr__(self, "cluster_seeds_disallow", disallow_pairs)

        specter_paper_ids = tuple(str(value) for value in self.specter_paper_ids)
        if any(not paper_id for paper_id in specter_paper_ids):
            raise ValueError("specter_paper_ids cannot contain empty paper_id values")
        if len(set(specter_paper_ids)) != len(specter_paper_ids):
            raise ValueError("specter_paper_ids must contain unique paper_id values")
        if self.specter_embeddings is None:
            if specter_paper_ids:
                raise ValueError("specter_paper_ids requires specter_embeddings")
        else:
            embeddings = np.asarray(self.specter_embeddings, dtype=np.float32)
            if embeddings.ndim != 2:
                raise ValueError(f"specter_embeddings must be 2D, got shape={embeddings.shape}")
            if embeddings.shape[0] != len(specter_paper_ids):
                raise ValueError(
                    "specter_embeddings row count must match specter_paper_ids: "
                    f"{embeddings.shape[0]} != {len(specter_paper_ids)}"
                )
            object.__setattr__(self, "specter_embeddings", np.ascontiguousarray(embeddings, dtype=np.float32))
        object.__setattr__(self, "specter_paper_ids", specter_paper_ids)

    @property
    def signature_ids(self) -> tuple[str, ...]:
        """Return signature ids in this block's deterministic order."""

        return self._signature_ids_cache

    @property
    def signature_order(self) -> FeatureBlockSignatureOrder:
        """Return the signature order used by numeric linker arrays."""

        return self._signature_order_cache

    @property
    def signature_id_to_index(self) -> Mapping[str, int]:
        """Return a signature-id to row-index map."""

        return self._signature_id_to_index_cache

    @property
    def seed_component_members(self) -> dict[str, tuple[str, ...]]:
        """Return require-seed members grouped by component id."""

        members: dict[str, list[str]] = {}
        for signature_id, component_id in self.cluster_seeds_require:
            members.setdefault(component_id, []).append(signature_id)
        return {component_id: tuple(signature_ids) for component_id, signature_ids in members.items()}

    def to_arrow_tables(self) -> dict[str, Any]:
        """Return Arrow tables for the current Rust raw-candidate schema."""

        import pyarrow as pa

        tables: dict[str, Any] = {
            "signatures": pa.table(
                {
                    "signature_id": pa.array([row.signature_id for row in self.signatures], type=pa.string()),
                    "paper_id": pa.array([row.paper_id for row in self.signatures], type=pa.string()),
                    "author_first": pa.array([row.author_first for row in self.signatures], type=pa.string()),
                    "author_middle": pa.array([row.author_middle for row in self.signatures], type=pa.string()),
                    "author_last": pa.array([row.author_last for row in self.signatures], type=pa.string()),
                    "author_suffix": pa.array([row.author_suffix for row in self.signatures], type=pa.string()),
                    "author_affiliations": pa.array(
                        [list(row.author_affiliations) for row in self.signatures],
                        type=pa.list_(pa.string()),
                    ),
                    "author_orcid": pa.array([row.author_orcid for row in self.signatures], type=pa.string()),
                    "author_position": pa.array([row.author_position for row in self.signatures], type=pa.int64()),
                    "author_block": pa.array([row.author_block for row in self.signatures], type=pa.string()),
                    "author_email": pa.array([row.author_email for row in self.signatures], type=pa.string()),
                    "source_author_ids": pa.array(
                        [list(row.source_author_ids) for row in self.signatures],
                        type=pa.list_(pa.string()),
                    ),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": pa.array([row.paper_id for row in self.papers], type=pa.string()),
                    "title": pa.array([row.title for row in self.papers], type=pa.string()),
                    "abstract": pa.array([row.abstract for row in self.papers], type=pa.string()),
                    "venue": pa.array([row.venue for row in self.papers], type=pa.string()),
                    "journal_name": pa.array([row.journal_name for row in self.papers], type=pa.string()),
                    "year": pa.array([row.year for row in self.papers], type=pa.int64()),
                    "predicted_language": pa.array(
                        [row.predicted_language for row in self.papers],
                        type=pa.string(),
                    ),
                    "is_reliable": pa.array([row.is_reliable for row in self.papers], type=pa.bool_()),
                    "language_reliability": pa.array(
                        [row.language_reliability for row in self.papers],
                        type=pa.float64(),
                    ),
                }
            ),
            "paper_authors": pa.table(
                {
                    "paper_id": pa.array([row.paper_id for row in self.paper_authors], type=pa.string()),
                    "position": pa.array([row.position for row in self.paper_authors], type=pa.int64()),
                    "author_name": pa.array([row.author_name for row in self.paper_authors], type=pa.string()),
                }
            ),
            "cluster_seeds": pa.table(
                {
                    "signature_id": pa.array(
                        [signature_id for signature_id, _component_id in self.cluster_seeds_require],
                        type=pa.string(),
                    ),
                    "cluster_id": pa.array(
                        [component_id for _signature_id, component_id in self.cluster_seeds_require],
                        type=pa.string(),
                    ),
                }
            ),
            "cluster_seed_disallows": pa.table(
                {
                    "signature_id_1": pa.array(
                        [left for left, _right in self.cluster_seeds_disallow],
                        type=pa.string(),
                    ),
                    "signature_id_2": pa.array(
                        [right for _left, right in self.cluster_seeds_disallow],
                        type=pa.string(),
                    ),
                }
            ),
        }
        if self.specter_embeddings is not None:
            flat = pa.array(np.ravel(self.specter_embeddings), type=pa.float32())
            tables["specter"] = pa.table(
                {
                    "paper_id": pa.array(list(self.specter_paper_ids), type=pa.string()),
                    "embedding": pa.FixedSizeListArray.from_arrays(flat, int(self.specter_embeddings.shape[1])),
                }
            )
        return tables


def feature_block_signature_order_from_raw_candidate_plan(plan: Mapping[str, Any]) -> FeatureBlockSignatureOrder:
    """Build a deterministic mini-block signature order from a raw candidate plan."""

    query_signature_ids = tuple(str(value) for value in _required_plan_sequence(plan, "query_signature_ids"))
    pair_signature_ids = (
        *_required_plan_sequence(plan, "left_signature_ids"),
        *_required_plan_sequence(plan, "right_signature_ids"),
    )
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for value in (
        *query_signature_ids,
        *pair_signature_ids,
    ):
        signature_id = str(value)
        if signature_id in seen:
            continue
        seen.add(signature_id)
        ordered_ids.append(signature_id)
    if not ordered_ids:
        raise ValueError("raw candidate plan must select at least one signature")
    return FeatureBlockSignatureOrder(signature_ids=tuple(ordered_ids), query_signature_ids=query_signature_ids)


def feature_block_for_signature_order(
    feature_block: FeatureBlock,
    signature_order: FeatureBlockSignatureOrder,
) -> FeatureBlock:
    """Return a `FeatureBlock` subset ordered for numeric raw-plan scoring."""

    signatures_by_id = {row.signature_id: row for row in feature_block.signatures}
    missing = [signature_id for signature_id in signature_order.signature_ids if signature_id not in signatures_by_id]
    if missing:
        raise ValueError(f"FeatureBlock is missing raw-plan signatures: {missing}")

    signatures = tuple(signatures_by_id[signature_id] for signature_id in signature_order.signature_ids)
    paper_ids = set(row.paper_id for row in signatures)
    available_paper_ids = {row.paper_id for row in feature_block.papers}
    missing_papers = paper_ids - available_paper_ids
    if missing_papers:
        raise ValueError(f"FeatureBlock is missing papers referenced by selected signatures: {sorted(missing_papers)}")
    papers = tuple(row for row in feature_block.papers if row.paper_id in paper_ids)
    paper_id_set = set(row.paper_id for row in papers)
    paper_authors = tuple(row for row in feature_block.paper_authors if row.paper_id in paper_id_set)
    signature_id_set = set(signature_order.signature_ids)
    require_pairs = tuple(
        (signature_id, component_id)
        for signature_id, component_id in feature_block.cluster_seeds_require
        if signature_id in signature_id_set
    )
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        feature_block.cluster_seeds_disallow,
        signature_id_set,
    )
    specter_paper_ids: list[str] = []
    specter_rows: list[np.ndarray] = []
    if feature_block.specter_embeddings is not None:
        specter_by_paper_id = {
            paper_id: np.asarray(feature_block.specter_embeddings[index], dtype=np.float32)
            for index, paper_id in enumerate(feature_block.specter_paper_ids)
        }
        for paper in papers:
            paper_id = paper.paper_id
            if paper_id in specter_by_paper_id:
                specter_paper_ids.append(paper_id)
                specter_rows.append(specter_by_paper_id[paper_id])
    return FeatureBlock(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=tuple(specter_paper_ids),
        specter_embeddings=None if not specter_rows else np.vstack(specter_rows).astype(np.float32),
    )


def _required_plan_sequence(plan: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    if key not in plan:
        raise KeyError(f"raw candidate plan is missing required key: {key}")
    value = plan[key]
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(value)
    raise TypeError(f"raw candidate plan key {key!r} must be a sequence")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any, *, field_name: str = "value") -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or null, got {value!r}") from exc


def _optional_bool(value: Any, *, field_name: str = "value") -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false"}:
            return False
        if normalized in {"1", "true"}:
            return True
    raise ValueError(f"{field_name} must be a boolean, 0/1, true/false, or null, got {value!r}")


def _optional_float(value: Any, *, field_name: str = "value") -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float or null, got {value!r}") from exc


def _feature_block_specter_from_mapping(
    paper_ids: Sequence[str],
    specter_embeddings: Mapping[Any, Any] | tuple[Any, Any] | None,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not specter_embeddings:
        return (), None
    if isinstance(specter_embeddings, tuple):
        if len(specter_embeddings) != 2:
            raise ValueError("SPECTER tuple payload must be (matrix, paper_ids)")
        matrix, keys = specter_embeddings
        matrix_array = np.asarray(matrix, dtype=np.float32)
        if matrix_array.ndim != 2:
            raise ValueError(f"SPECTER tuple matrix must be 2D, got shape={matrix_array.shape}")
        key_list = list(keys)
        if len(key_list) != int(matrix_array.shape[0]):
            raise ValueError(
                f"SPECTER tuple key count must match matrix rows: keys={len(key_list)}, rows={matrix_array.shape[0]}"
            )
        specter_mapping: Mapping[Any, Any] = {
            str(key): np.ascontiguousarray(matrix_array[index], dtype=np.float32) for index, key in enumerate(key_list)
        }
    else:
        specter_mapping = specter_embeddings
    selected_paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper_id in paper_ids:
        vector = specter_mapping.get(str(paper_id))
        if vector is None:
            vector = specter_mapping.get(paper_id)
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors in a FeatureBlock must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper_id!r}"
            )
        selected_paper_ids.append(str(paper_id))
        vectors.append(array)
    if not vectors:
        return (), None
    return tuple(selected_paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
