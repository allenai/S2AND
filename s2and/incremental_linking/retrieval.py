"""Rust retrieval-to-candidate-batch bridge for incremental linking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from s2and.incremental_linking.array_validation import as_retrieval_rank_uint16_1d, as_uint32_1d
from s2and.incremental_linking.feature_block import FeatureBlockSignatureOrder
from s2and.incremental_linking.gate_buckets import (
    QueryView,
    first_name_bucket_array,
    normalize_query_views,
    validate_query_view,
)
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch

RAW_CANDIDATE_PLAN_SCHEMA_VERSION = "raw_arrow_candidate_plan_v2"
RAW_CANDIDATE_PLAN_BATCH_ROW_KEYS: tuple[str, ...] = (
    "row_query_signature_indices",
    "row_component_keys",
    "retrieval_scores",
    "retrieval_ranks",
)
RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("row_component_sizes", "cluster_size", np.float32),
    ("row_named_signature_counts", "named_signature_count", np.float32),
    ("row_dominant_first_names", "dominant_first_name", object),
    ("row_candidate_year_min", "candidate_year_min", np.int32),
    ("row_candidate_year_max", "candidate_year_max", np.int32),
    ("row_candidate_year_range_missing", "candidate_year_range_missing", np.uint8),
    ("row_query_first_tokens", "query_first_token", object),
    ("row_query_years", "query_year", np.int32),
    ("row_query_year_missing", "query_year_missing", np.uint8),
    ("row_query_has_affiliations", "query_has_affiliations", np.uint8),
    ("row_query_has_coauthors", "query_has_coauthors", np.uint8),
    ("row_orcid_match", "orcid_match", np.uint8),
    ("middle_initial_compatibility", "middle_initial_compatibility", np.float32),
    ("affiliation_overlap", "affiliation_overlap", np.float32),
    ("coauthor_overlap", "coauthor_overlap", np.float32),
    ("venue_overlap", "venue_overlap", np.float32),
    ("year_compatibility", "year_compatibility", np.float32),
    ("title_overlap", "title_overlap", np.float32),
    ("specter_centroid_similarity", "specter_centroid_similarity", np.float32),
    ("specter_exemplar_similarity", "specter_exemplar_similarity", np.float32),
    ("row_last_name_count_min_rarity", "last_name_count_min_rarity", np.float32),
    ("row_candidate_last_name_count_min_rarity", "candidate_last_name_count_min_rarity", np.float32),
    (
        "row_candidate_last_first_name_count_min_rarity",
        "candidate_last_first_name_count_min_rarity",
        np.float32,
    ),
    ("row_last_first_name_count_min_rarity", "last_first_name_count_min_rarity", np.float32),
    (
        "row_first_prefix_x_last_first_name_count_min_rarity",
        "first_prefix_x_last_first_name_count_min_rarity",
        np.float32,
    ),
    ("row_candidate_cluster_max_paper_author_count", "candidate_cluster_max_paper_author_count", np.float32),
    ("row_paper_author_list_max_jaccard", "paper_author_list_max_jaccard", np.float32),
    ("row_paper_author_list_max_containment", "paper_author_list_max_containment", np.float32),
    ("row_paper_author_list_max_overlap_count", "paper_author_list_max_overlap_count", np.float32),
    ("row_local_author_window10_jaccard_max", "local_author_window10_jaccard_max", np.float32),
    (
        "row_local_author_window10_overlap_count_max",
        "local_author_window10_overlap_count_max",
        np.float32,
    ),
    ("row_best_author_count_log_absdiff", "best_author_count_log_absdiff", np.float32),
)
RAW_CANDIDATE_PLAN_ROW_KEYS: tuple[str, ...] = (
    *RAW_CANDIDATE_PLAN_BATCH_ROW_KEYS,
    *(raw_key for raw_key, _signal_key, _dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS),
)
RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS: tuple[str, ...] = ("pair_row_indices",)
RAW_CANDIDATE_PLAN_PAIR_ID_KEYS: tuple[str, ...] = (
    "left_signature_ids",
    "right_signature_ids",
)
RAW_CANDIDATE_PLAN_PAIR_KEYS: tuple[str, ...] = (
    *RAW_CANDIDATE_PLAN_PAIR_INDEX_KEYS,
    *RAW_CANDIDATE_PLAN_PAIR_ID_KEYS,
)
RUST_PAIR_PLAN_PAIR_KEYS: tuple[str, ...] = (
    "left_signature_indices",
    "right_signature_indices",
    "pair_row_indices",
)
RUST_PAIR_PLAN_ROW_SIGNAL_KEYS: tuple[str, ...] = (
    "row_query_signature_indices",
    "row_component_keys",
    "retrieval_scores",
    "retrieval_ranks",
    "row_query_first_tokens",
    "row_component_sizes",
    "row_named_signature_counts",
    "row_dominant_first_names",
    "row_candidate_year_min",
    "row_candidate_year_max",
    "row_candidate_year_range_missing",
    "row_query_years",
    "row_query_year_missing",
    "row_query_has_affiliations",
    "row_query_has_coauthors",
    "row_orcid_match",
    "middle_initial_compatibility",
    "affiliation_overlap",
    "coauthor_overlap",
    "venue_overlap",
    "year_compatibility",
    "title_overlap",
    "specter_centroid_similarity",
    "specter_exemplar_similarity",
)
REQUIRED_RUST_PAIR_PLAN_KEYS: tuple[str, ...] = (
    "row_count",
    *RUST_PAIR_PLAN_PAIR_KEYS,
    *RUST_PAIR_PLAN_ROW_SIGNAL_KEYS,
)


def _query_author_for_retrieval_row_signal(query: Any) -> str:
    """Resolve a display author string for a retrieval query.

    Mirrors the gate-side helper in s2and.incremental_linking.runtime so that the
    Rust retrieval path can populate `query_author` row signals directly, matching
    the raw-Arrow retrieval path's row-signal contract.
    """
    value = getattr(query, "query_author", None)
    if value is not None and str(value).strip():
        return str(value)

    def first_present(*names: str) -> Any:
        for name in names:
            attr_value = getattr(query, name, None)
            if attr_value is not None and str(attr_value).strip():
                return attr_value
        return None

    parts = [
        first_present("first", "author_info_first"),
        first_present("middle", "author_info_middle"),
        first_present("last", "author_info_last"),
        first_present("suffix", "author_info_suffix"),
    ]
    return " ".join(str(part).strip() for part in parts if part is not None and str(part).strip())


@dataclass(frozen=True)
class LinkerRetrievalBatch:
    """Retrieved candidate rows, flat pair plan, and compact row-level signals."""

    candidate_batch: LinkerCandidateBatch
    row_signals: dict[str, Any]


@dataclass(frozen=True)
class RawArrowPlanBundle:
    """Validated raw plan with an owned, normalized linker-bridge payload."""

    plan: Mapping[str, Any]
    signature_order: FeatureBlockSignatureOrder
    query_signature_ids: tuple[str, ...]
    query_views: tuple[QueryView, ...]
    row_count: int
    pair_count: int
    row_query_offsets: np.ndarray
    left_signature_ids: tuple[str, ...]
    right_signature_ids: tuple[str, ...]
    pair_row_indices: np.ndarray
    row_component_keys: tuple[str, ...]
    retrieval_scores: np.ndarray
    retrieval_ranks: np.ndarray
    row_signals: Mapping[str, np.ndarray]

    def to_mutable_mapping(self) -> dict[str, Any]:
        """Return a detached mutable plan for the public mapping API."""

        return {key: _mutable_plan_value(value) for key, value in self.plan.items()}

    @classmethod
    def from_mapping(cls, plan: Mapping[str, Any]) -> RawArrowPlanBundle:
        """Validate once and own every value consumed by the linker bridge."""

        schema_version = _required_raw_plan_value(plan, "schema_version")
        if schema_version != RAW_CANDIDATE_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "raw candidate plan schema_version must be "
                f"{RAW_CANDIDATE_PLAN_SCHEMA_VERSION!r}, got {schema_version!r}"
            )
        row_count = _raw_plan_nonnegative_count(plan, "row_count")
        pair_count = _raw_plan_nonnegative_count(plan, "pair_count")
        missing = sorted(
            key
            for key in (
                "query_signature_ids",
                "query_views",
                "query_authors",
                "component_members",
                *RAW_CANDIDATE_PLAN_ROW_KEYS,
                *RAW_CANDIDATE_PLAN_PAIR_KEYS,
            )
            if key not in plan
        )
        if missing:
            raise KeyError(f"raw candidate plan is missing required keys: {missing}")
        component_members = _required_raw_plan_value(plan, "component_members")
        if not isinstance(component_members, Mapping):
            raise ValueError("raw candidate plan component_members must be a mapping")
        legacy_pair_index_keys = sorted(
            key for key in ("left_signature_indices", "right_signature_indices") if key in plan
        )
        if legacy_pair_index_keys:
            raise ValueError(
                "raw candidate plan must not include legacy numeric pair indices; "
                f"unexpected keys={legacy_pair_index_keys}"
            )

        query_count = _raw_plan_sequence_length(plan, "query_signature_ids")
        if query_count == 0:
            raise ValueError("raw candidate plan query_signature_ids must be non-empty")
        query_signature_ids = tuple(str(value) for value in _required_raw_plan_value(plan, "query_signature_ids"))
        if len(set(query_signature_ids)) != len(query_signature_ids):
            raise ValueError("raw candidate plan query_signature_ids must be unique")
        for key in ("query_views", "query_authors"):
            length = _raw_plan_sequence_length(plan, key)
            if length != query_count:
                raise ValueError(
                    f"raw candidate plan {key} length must match query_signature_ids: {length} != {query_count}"
                )
        query_views = tuple(validate_query_view(value) for value in _required_raw_plan_value(plan, "query_views"))
        query_authors = tuple(str(value or "") for value in _required_raw_plan_value(plan, "query_authors"))

        row_query_offsets = _raw_plan_array(plan, "row_query_signature_indices", np.uint32, row_count)
        _validate_uint32_indices_below(
            row_query_offsets,
            key="row_query_signature_indices",
            upper_bound=query_count,
            bound_name="query_signature_ids length",
        )
        row_component_keys_array = _raw_plan_array(plan, "row_component_keys", object, row_count)
        row_component_keys = tuple(str(value) for value in row_component_keys_array)
        retrieval_scores = _raw_plan_array(plan, "retrieval_scores", np.float32, row_count)
        retrieval_ranks = _raw_plan_retrieval_ranks(plan, row_count)
        pair_row_indices = _raw_plan_array(plan, "pair_row_indices", np.uint32, pair_count)
        _validate_uint32_indices_below(
            pair_row_indices,
            key="pair_row_indices",
            upper_bound=row_count,
            bound_name="row_count",
        )
        left_signature_ids = _raw_plan_id_tuple(plan, "left_signature_ids", pair_count)
        right_signature_ids = _raw_plan_id_tuple(plan, "right_signature_ids", pair_count)
        for pair_index, (left_signature_id, row_index) in enumerate(
            zip(left_signature_ids, pair_row_indices, strict=True)
        ):
            query_offset = int(row_query_offsets[int(row_index)])
            expected_left_signature_id = query_signature_ids[query_offset]
            if left_signature_id != expected_left_signature_id:
                raise ValueError(
                    "raw candidate plan left_signature_ids must match each pair row query_signature_ids: "
                    f"pair {pair_index} has {left_signature_id!r}, expected {expected_left_signature_id!r}"
                )

        normalized_row_signals = {
            signal_key: _raw_plan_array(plan, raw_key, dtype, row_count)
            for raw_key, signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS
        }
        row_query_views = _owned_readonly_array(
            [query_views[int(offset)] for offset in row_query_offsets],
            dtype=object,
        )
        row_query_authors = _owned_readonly_array(
            [query_authors[int(offset)] for offset in row_query_offsets],
            dtype=object,
        )
        row_component_key_array = _owned_readonly_array(row_component_keys, dtype=object)
        row_signals = {
            "retrieval_score": retrieval_scores,
            "retrieval_rank": retrieval_ranks,
            "candidate_component_key": row_component_key_array,
            "query_view": row_query_views,
            "query_author": row_query_authors,
            "first_name_bucket": _owned_readonly_array(
                first_name_bucket_array(normalized_row_signals["query_first_token"], row_query_views),
                dtype=object,
            ),
        }
        row_signals.update(normalized_row_signals)

        normalized_plan_values: dict[str, Any] = {
            "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
            "query_signature_ids": query_signature_ids,
            "query_views": query_views,
            "query_authors": query_authors,
            "row_count": row_count,
            "pair_count": pair_count,
            "row_query_signature_indices": row_query_offsets,
            "row_component_keys": row_component_keys,
            "retrieval_scores": retrieval_scores,
            "retrieval_ranks": retrieval_ranks,
            "pair_row_indices": pair_row_indices,
            "left_signature_ids": left_signature_ids,
            "right_signature_ids": right_signature_ids,
        }
        for raw_key, signal_key, _dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
            normalized_plan_values[raw_key] = normalized_row_signals[signal_key]
        owned_plan = _owned_plan_mapping(plan, normalized_values=normalized_plan_values)

        ordered_ids: list[str] = []
        seen: set[str] = set()
        for value in (
            *query_signature_ids,
            *left_signature_ids,
            *right_signature_ids,
        ):
            if value in seen:
                continue
            seen.add(value)
            ordered_ids.append(value)

        return cls(
            plan=owned_plan,
            signature_order=FeatureBlockSignatureOrder(
                signature_ids=tuple(ordered_ids),
                query_signature_ids=query_signature_ids,
            ),
            query_signature_ids=query_signature_ids,
            query_views=query_views,
            row_count=row_count,
            pair_count=pair_count,
            row_query_offsets=row_query_offsets,
            left_signature_ids=left_signature_ids,
            right_signature_ids=right_signature_ids,
            pair_row_indices=pair_row_indices,
            row_component_keys=row_component_keys,
            retrieval_scores=retrieval_scores,
            retrieval_ranks=retrieval_ranks,
            row_signals=MappingProxyType(row_signals),
        )


def _rust_retriever_object(retriever: Any) -> Any:
    return getattr(retriever, "retriever", retriever)


def _as_uint32_mapping(component_member_indices_by_key: Mapping[str, Sequence[int] | np.ndarray]) -> dict[str, Any]:
    return {
        str(component_key): as_uint32_1d(f"component_member_indices_by_key[{component_key!r}]", member_indices)
        for component_key, member_indices in component_member_indices_by_key.items()
    }


def _validate_rust_pair_plan_schema(plan: Mapping[str, Any]) -> None:
    missing = sorted(key for key in REQUIRED_RUST_PAIR_PLAN_KEYS if key not in plan)
    if missing:
        raise RuntimeError(
            "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan violated its result contract: "
            f"missing keys={missing}"
        )
    row_count = int(plan["row_count"])
    if row_count < 0:
        raise RuntimeError(
            "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan violated its result contract: "
            f"row_count must be non-negative, got {row_count}"
        )
    pair_count = _rust_plan_sequence_len(plan, "pair_row_indices")
    for key in ("left_signature_indices", "right_signature_indices"):
        length = _rust_plan_sequence_len(plan, key)
        if length != pair_count:
            raise RuntimeError(
                "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan violated its pair-array contract: "
                f"{key} length={length} pair_row_indices length={pair_count}"
            )
    for key in RUST_PAIR_PLAN_ROW_SIGNAL_KEYS:
        length = _rust_plan_sequence_len(plan, key)
        if length != row_count:
            raise RuntimeError(
                "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan violated its row-array contract: "
                f"{key} length={length} row_count={row_count}"
            )


def _rust_plan_sequence_len(plan: Mapping[str, Any], key: str) -> int:
    try:
        return len(plan[key])
    except TypeError as exc:
        raise RuntimeError(
            "RustHybridCentroidRetriever.top_k_hybrid_centroid_pair_plan violated its result contract: "
            f"key {key!r} is not a sized sequence"
        ) from exc


def _required_raw_plan_value(plan: Mapping[str, Any], key: str) -> Any:
    if key not in plan:
        raise KeyError(f"raw candidate plan is missing required key: {key}")
    return plan[key]


def _uint8_flag_array(key: str, raw_values: Any, expected_length: int | None = None) -> np.ndarray:
    values = np.asarray(raw_values)
    if values.ndim != 1:
        raise ValueError(f"raw candidate plan key {key!r} must be 1D, got {values.shape}")
    if expected_length is not None and len(values) != int(expected_length):
        raise ValueError(f"raw candidate plan key {key!r} must be 1D with length {expected_length}, got {values.shape}")
    if values.dtype == np.bool_:
        return values.astype(np.uint8)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"raw candidate plan key {key!r} must contain 0/1 integer flags")
    invalid = (values < 0) | (values > 1)
    if bool(np.any(invalid)):
        invalid_value = values[invalid][0]
        raise ValueError(f"raw candidate plan key {key!r} contains non-0/1 flag value {invalid_value!r}")
    return values.astype(np.uint8)


def _owned_readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Copy a normalized 1D array into bundle-owned read-only storage."""

    owned = np.array(values, dtype=dtype, copy=True, order="C")
    if owned.dtype.hasobject:
        for index, item in np.ndenumerate(owned):
            owned[index] = _owned_plan_value(item)
    owned.setflags(write=False)
    return owned


def _owned_plan_value(value: Any) -> Any:
    """Recursively detach and freeze a non-canonical raw-plan value."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            owned = np.empty(value.shape, dtype=object)
            for index, item in np.ndenumerate(value):
                owned[index] = _owned_plan_value(item)
            owned.setflags(write=False)
            return owned
        return _owned_readonly_array(value, dtype=value.dtype)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _owned_plan_value(item) for key, item in value.items()})
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_owned_plan_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_owned_plan_value(item) for item in value)
    return value


def _owned_plan_mapping(
    plan: Mapping[str, Any],
    *,
    normalized_values: Mapping[str, Any],
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: normalized_values[key] if key in normalized_values else _owned_plan_value(value)
            for key, value in plan.items()
        }
    )


def _mutable_plan_value(value: Any) -> Any:
    """Recursively copy a bundle plan value back to mutable containers."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            mutable = np.empty(value.shape, dtype=object)
            for index, item in np.ndenumerate(value):
                mutable[index] = _mutable_plan_value(item)
            return mutable
        return np.array(value, dtype=value.dtype, copy=True, order="C")
    if isinstance(value, Mapping):
        return {key: _mutable_plan_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_plan_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_mutable_plan_value(item) for item in value}
    return value


def _raw_plan_array(plan: Mapping[str, Any], key: str, dtype: Any, expected_length: int) -> np.ndarray:
    if dtype == np.uint32:
        values = as_uint32_1d(key, _required_raw_plan_value(plan, key))
    elif dtype == np.uint8:
        values = _uint8_flag_array(key, _required_raw_plan_value(plan, key), expected_length)
    else:
        values = np.asarray(_required_raw_plan_value(plan, key), dtype=dtype)
    if values.ndim != 1 or len(values) != int(expected_length):
        raise ValueError(f"raw candidate plan key {key!r} must be 1D with length {expected_length}, got {values.shape}")
    return _owned_readonly_array(values, dtype=values.dtype)


def _raw_plan_retrieval_ranks(plan: Mapping[str, Any], expected_length: int) -> np.ndarray:
    ranks = as_retrieval_rank_uint16_1d(
        "retrieval_ranks",
        _required_raw_plan_value(plan, "retrieval_ranks"),
    )
    if len(ranks) != int(expected_length):
        raise ValueError(
            "raw candidate plan key 'retrieval_ranks' must be 1D with length " f"{expected_length}, got {ranks.shape}"
        )
    return _owned_readonly_array(ranks, dtype=np.uint16)


def _raw_plan_id_tuple(plan: Mapping[str, Any], key: str, expected_length: int) -> tuple[str, ...]:
    length = _raw_plan_sequence_length(plan, key)
    if length != int(expected_length):
        raise ValueError(f"raw candidate plan {key} length must match pair_count: {length} != {expected_length}")
    return tuple(str(value) for value in _required_raw_plan_value(plan, key))


def _validate_uint32_indices_below(values: np.ndarray, *, key: str, upper_bound: int, bound_name: str) -> None:
    if len(values) == 0:
        return
    invalid = values >= int(upper_bound)
    if bool(np.any(invalid)):
        invalid_value = int(values[invalid][0])
        raise ValueError(
            f"raw candidate plan {key} contains index {invalid_value} outside {bound_name}={int(upper_bound)}"
        )


def _raw_plan_nonnegative_count(plan: Mapping[str, Any], key: str) -> int:
    value = int(_required_raw_plan_value(plan, key))
    if value < 0:
        raise ValueError(f"raw candidate plan {key} must be non-negative")
    return value


def _raw_plan_sequence_length(plan: Mapping[str, Any], key: str) -> int:
    try:
        return len(_required_raw_plan_value(plan, key))
    except TypeError as exc:
        raise ValueError(f"raw candidate plan key {key!r} must be a sized 1D sequence") from exc


def validate_raw_candidate_plan_schema(plan: Mapping[str, Any]) -> None:
    """Validate the raw Arrow candidate-plan payload before slicing or remapping it."""

    RawArrowPlanBundle.from_mapping(plan)


def _signature_indices_from_ids(
    signature_ids: Sequence[Any],
    signature_id_to_index: Mapping[str, int],
    *,
    field_name: str,
) -> np.ndarray:
    try:
        return as_uint32_1d(
            field_name,
            [int(signature_id_to_index[str(signature_id)]) for signature_id in signature_ids],
        )
    except KeyError as exc:
        raise KeyError(
            f"{field_name} contains signature_id not present in signature_id_to_index: {str(exc.args[0])!r}"
        ) from exc


def _signature_id_to_index_from_order(
    signature_order: FeatureBlockSignatureOrder | Sequence[Any],
) -> dict[str, int]:
    if isinstance(signature_order, FeatureBlockSignatureOrder):
        return signature_order.signature_id_to_index
    return {str(signature_id): index for index, signature_id in enumerate(signature_order)}


def build_linker_retrieval_batch_from_raw_plan_bundle(
    bundle: RawArrowPlanBundle,
    *,
    signature_id_to_index: Mapping[str, int] | None = None,
    feature_block_signature_order: FeatureBlockSignatureOrder | Sequence[Any] | None = None,
) -> LinkerRetrievalBatch:
    """Convert a validated raw plan into the numeric linker batch contract.

    The raw Arrow API uses request-local query offsets and signature ids. The
    downstream linker runtime expects numeric indices in the current
    dataset/featurizer signature order. This bridge performs only that mapping;
    it does not rerun retrieval.
    """

    if signature_id_to_index is None:
        if feature_block_signature_order is None:
            signature_id_to_index = bundle.signature_order.signature_id_to_index
        else:
            signature_id_to_index = _signature_id_to_index_from_order(feature_block_signature_order)
    elif feature_block_signature_order is not None:
        raise ValueError("Pass only one of signature_id_to_index or feature_block_signature_order")

    row_count = bundle.row_count
    query_indices_by_offset = _signature_indices_from_ids(
        bundle.query_signature_ids,
        signature_id_to_index,
        field_name="query_signature_ids",
    )
    row_query_signature_indices = query_indices_by_offset[bundle.row_query_offsets]
    left_signature_indices = _signature_indices_from_ids(
        bundle.left_signature_ids,
        signature_id_to_index,
        field_name="left_signature_ids",
    )
    right_signature_indices = _signature_indices_from_ids(
        bundle.right_signature_ids,
        signature_id_to_index,
        field_name="right_signature_ids",
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left_signature_indices,
        right_signature_indices=right_signature_indices,
        pair_row_indices=bundle.pair_row_indices,
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=bundle.row_component_keys,
        retrieval_scores=bundle.retrieval_scores,
        retrieval_ranks=bundle.retrieval_ranks,
    )
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals=dict(bundle.row_signals))


def build_linker_retrieval_batch_from_raw_candidate_plan(
    plan: Mapping[str, Any],
    *,
    signature_id_to_index: Mapping[str, int] | None = None,
    feature_block_signature_order: FeatureBlockSignatureOrder | Sequence[Any] | None = None,
) -> LinkerRetrievalBatch:
    """Validate and convert a raw id-based candidate plan at the public boundary."""

    return build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_mapping(plan),
        signature_id_to_index=signature_id_to_index,
        feature_block_signature_order=feature_block_signature_order,
    )


def build_linker_retrieval_batch_rust(
    *,
    retriever: Any,
    queries: Sequence[Any],
    query_signature_indices: Sequence[int] | np.ndarray,
    query_signature_ids: Sequence[str] | None = None,
    component_member_indices_by_key: Mapping[str, Sequence[int] | np.ndarray],
    top_k: int,
    query_view: str | Sequence[str],
    n_jobs: int | None = None,
    retrieval_subblock_index: Mapping[str, Any] | None = None,
    query_candidate_component_keys_by_signature_id: Mapping[str, Sequence[str]] | None = None,
    full_first_global_backfill_count: int = 5,
) -> LinkerRetrievalBatch:
    """Retrieve candidates in Rust and return the shared numeric candidate-batch contract."""

    normalized_query_views = normalize_query_views(query_view, len(queries))
    query_signature_indices_array = as_uint32_1d("query_signature_indices", query_signature_indices)
    if not isinstance(normalized_query_views, str):
        query_index_values = [int(value) for value in query_signature_indices_array]
        if len(set(query_index_values)) != len(query_index_values):
            raise ValueError("query_signature_indices must be unique when per-query query_view values are provided")
    rust_retriever = _rust_retriever_object(retriever)
    if retrieval_subblock_index is not None or query_candidate_component_keys_by_signature_id is not None:
        if query_signature_ids is None:
            raise ValueError(
                "query_signature_ids are required when retrieval_subblock_index or query candidate keys are provided"
            )
        if len(query_signature_ids) != len(queries):
            raise ValueError(
                "queries and query_signature_ids must have equal length: "
                f"{len(queries)} != {len(query_signature_ids)}"
            )
        plan = rust_retriever.top_k_hybrid_centroid_pair_plan(
            list(queries),
            query_signature_indices_array,
            _as_uint32_mapping(component_member_indices_by_key),
            int(top_k),
            None if n_jobs is None else int(n_jobs),
            [str(value) for value in query_signature_ids],
            None if retrieval_subblock_index is None else dict(retrieval_subblock_index),
            (
                None
                if query_candidate_component_keys_by_signature_id is None
                else {
                    str(query_signature_id): [str(component_key) for component_key in component_keys]
                    for query_signature_id, component_keys in query_candidate_component_keys_by_signature_id.items()
                }
            ),
            int(full_first_global_backfill_count),
        )
    else:
        plan = rust_retriever.top_k_hybrid_centroid_pair_plan(
            list(queries),
            query_signature_indices_array,
            _as_uint32_mapping(component_member_indices_by_key),
            int(top_k),
            None if n_jobs is None else int(n_jobs),
        )
    _validate_rust_pair_plan_schema(plan)
    row_count = int(plan["row_count"])
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=as_uint32_1d("left_signature_indices", plan["left_signature_indices"]),
        right_signature_indices=as_uint32_1d("right_signature_indices", plan["right_signature_indices"]),
        pair_row_indices=as_uint32_1d("pair_row_indices", plan["pair_row_indices"]),
        row_query_signature_indices=as_uint32_1d(
            "row_query_signature_indices",
            plan["row_query_signature_indices"],
        ),
        row_component_keys=tuple(str(value) for value in plan["row_component_keys"]),
        retrieval_scores=np.asarray(plan["retrieval_scores"], dtype=np.float32),
        retrieval_ranks=as_retrieval_rank_uint16_1d("retrieval_ranks", plan["retrieval_ranks"]),
    )
    if isinstance(normalized_query_views, str):
        query_views: Any = np.full(row_count, normalized_query_views, dtype=object)
    else:
        row_query_signature_indices = candidate_batch.row_query_signature_indices
        if row_query_signature_indices is None:
            raise RuntimeError("Rust retrieval plan did not provide row_query_signature_indices")
        query_view_by_query_index = {
            int(query_index): str(current_query_view)
            for query_index, current_query_view in zip(
                query_signature_indices_array,
                normalized_query_views,
                strict=True,
            )
        }
        query_views = np.asarray(
            [query_view_by_query_index[int(query_index)] for query_index in row_query_signature_indices],
            dtype=object,
        )
    query_first_tokens = np.asarray(plan["row_query_first_tokens"], dtype=object)
    # Mirror the raw-Arrow path: broadcast per-query author strings onto each row so
    # downstream gate features (top_meta_query_author_len) read query_author directly
    # from row_signals instead of relying on a separate runtime patch-in.
    row_query_signature_indices_arr = candidate_batch.row_query_signature_indices
    if row_query_signature_indices_arr is None:
        raise RuntimeError("Rust retrieval plan did not provide row_query_signature_indices")
    per_query_authors = np.asarray(
        [_query_author_for_retrieval_row_signal(query) for query in queries],
        dtype=object,
    )
    query_index_to_offset = {
        int(query_index): offset for offset, query_index in enumerate(query_signature_indices_array)
    }
    query_authors_per_row = np.asarray(
        [per_query_authors[query_index_to_offset[int(query_index)]] for query_index in row_query_signature_indices_arr],
        dtype=object,
    )
    row_signals: dict[str, Any] = {
        "retrieval_score": candidate_batch.retrieval_scores,
        "retrieval_rank": candidate_batch.retrieval_ranks,
        "candidate_component_key": np.asarray(candidate_batch.row_component_keys, dtype=object),
        "query_view": query_views,
        "query_author": query_authors_per_row,
        "cluster_size": np.asarray(plan["row_component_sizes"], dtype=np.float32),
        "named_signature_count": np.asarray(plan["row_named_signature_counts"], dtype=np.float32),
        "dominant_first_name": np.asarray(plan["row_dominant_first_names"], dtype=object),
        "candidate_year_min": np.asarray(plan["row_candidate_year_min"], dtype=np.int32),
        "candidate_year_max": np.asarray(plan["row_candidate_year_max"], dtype=np.int32),
        "candidate_year_range_missing": _uint8_flag_array(
            "row_candidate_year_range_missing",
            plan["row_candidate_year_range_missing"],
            row_count,
        ),
        "query_first_token": query_first_tokens,
        "first_name_bucket": first_name_bucket_array(query_first_tokens, query_views),
        "query_year": np.asarray(plan["row_query_years"], dtype=np.int32),
        "query_year_missing": _uint8_flag_array("row_query_year_missing", plan["row_query_year_missing"], row_count),
        "query_has_affiliations": _uint8_flag_array(
            "row_query_has_affiliations",
            plan["row_query_has_affiliations"],
            row_count,
        ),
        "query_has_coauthors": _uint8_flag_array(
            "row_query_has_coauthors",
            plan["row_query_has_coauthors"],
            row_count,
        ),
        "orcid_match": _uint8_flag_array("row_orcid_match", plan["row_orcid_match"], row_count),
        "middle_initial_compatibility": np.asarray(plan["middle_initial_compatibility"], dtype=np.float32),
        "affiliation_overlap": np.asarray(plan["affiliation_overlap"], dtype=np.float32),
        "coauthor_overlap": np.asarray(plan["coauthor_overlap"], dtype=np.float32),
        "venue_overlap": np.asarray(plan["venue_overlap"], dtype=np.float32),
        "year_compatibility": np.asarray(plan["year_compatibility"], dtype=np.float32),
        "title_overlap": np.asarray(plan["title_overlap"], dtype=np.float32),
        "specter_centroid_similarity": np.asarray(plan["specter_centroid_similarity"], dtype=np.float32),
        "specter_exemplar_similarity": np.asarray(plan["specter_exemplar_similarity"], dtype=np.float32),
    }
    return LinkerRetrievalBatch(candidate_batch=candidate_batch, row_signals=row_signals)
