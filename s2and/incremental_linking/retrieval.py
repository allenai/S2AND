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
    validate_query_view,
)
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch

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


@dataclass(frozen=True)
class LinkerRetrievalBatch:
    """Retrieved candidate rows, flat pair plan, and compact row-level signals."""

    candidate_batch: LinkerCandidateBatch
    row_signals: dict[str, Any]


@dataclass(frozen=True)
class RawArrowPlanBundle:
    """Validated, immutable raw-plan values used by the linker runtime."""

    signature_order: FeatureBlockSignatureOrder
    query_signature_ids: tuple[str, ...]
    query_views: tuple[QueryView, ...]
    query_authors: tuple[str, ...]
    seed_signature_ids: tuple[str, ...] | None
    component_members: Mapping[str, tuple[str, ...]]
    telemetry: Mapping[str, Any] | None
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

    def contiguous_query_slice(self, start: int, stop: int) -> RawArrowPlanBundle:
        """Return the row and pair views for a contiguous query slice."""

        query_signature_ids = self.query_signature_ids[start:stop]
        row_start = int(np.searchsorted(self.row_query_offsets, start, side="left"))
        row_stop = int(np.searchsorted(self.row_query_offsets, stop, side="left"))
        pair_start = int(np.searchsorted(self.pair_row_indices, row_start, side="left"))
        pair_stop = int(np.searchsorted(self.pair_row_indices, row_stop, side="left"))
        row_query_offsets = (self.row_query_offsets[row_start:row_stop] - start).astype(np.uint32, copy=False)
        pair_row_indices = (self.pair_row_indices[pair_start:pair_stop] - row_start).astype(np.uint32, copy=False)
        row_query_offsets.setflags(write=False)
        pair_row_indices.setflags(write=False)
        return RawArrowPlanBundle(
            signature_order=FeatureBlockSignatureOrder(
                signature_ids=self.signature_order.signature_ids,
                query_signature_ids=query_signature_ids,
            ),
            query_signature_ids=query_signature_ids,
            query_views=self.query_views[start:stop],
            query_authors=self.query_authors[start:stop],
            seed_signature_ids=self.seed_signature_ids,
            component_members=self.component_members,
            telemetry=self.telemetry if start == 0 else None,
            row_count=row_stop - row_start,
            pair_count=pair_stop - pair_start,
            row_query_offsets=row_query_offsets,
            left_signature_ids=self.left_signature_ids[pair_start:pair_stop],
            right_signature_ids=self.right_signature_ids[pair_start:pair_stop],
            pair_row_indices=pair_row_indices,
            row_component_keys=self.row_component_keys[row_start:row_stop],
            retrieval_scores=self.retrieval_scores[row_start:row_stop],
            retrieval_ranks=self.retrieval_ranks[row_start:row_stop],
            row_signals=MappingProxyType({key: values[row_start:row_stop] for key, values in self.row_signals.items()}),
        )

    @classmethod
    def _from_normalized_values(
        cls,
        *,
        query_signature_ids: Sequence[Any],
        query_views: Sequence[QueryView],
        query_authors: Sequence[Any],
        seed_signature_ids: Sequence[Any] | None,
        component_members: Mapping[Any, Sequence[Any]],
        telemetry: Mapping[Any, Any] | None,
        row_query_offsets: Any,
        left_signature_ids: Sequence[Any],
        right_signature_ids: Sequence[Any],
        pair_row_indices: Any,
        row_component_keys: Sequence[Any],
        retrieval_scores: Any,
        retrieval_ranks: Any,
        normalized_row_signals: Mapping[str, Any],
    ) -> RawArrowPlanBundle:
        """Own normalized values and derive the runtime-only views once."""

        owned_query_signature_ids = tuple(str(value) for value in query_signature_ids)
        owned_query_views = tuple(query_views)
        owned_query_authors = tuple(str(value or "") for value in query_authors)
        owned_row_query_offsets = _readonly_array(row_query_offsets, dtype=np.uint32)
        owned_left_signature_ids = tuple(str(value) for value in left_signature_ids)
        owned_right_signature_ids = tuple(str(value) for value in right_signature_ids)
        owned_pair_row_indices = _readonly_array(pair_row_indices, dtype=np.uint32)
        owned_row_component_keys = tuple(str(value) for value in row_component_keys)
        owned_retrieval_scores = _readonly_array(retrieval_scores, dtype=np.float32)
        owned_retrieval_ranks = _readonly_array(retrieval_ranks, dtype=np.uint16)
        owned_normalized_row_signals = {
            signal_key: _readonly_array(values) for signal_key, values in normalized_row_signals.items()
        }
        row_query_views = _readonly_array(
            [owned_query_views[int(offset)] for offset in owned_row_query_offsets],
            dtype=object,
        )
        row_query_authors = _readonly_array(
            [owned_query_authors[int(offset)] for offset in owned_row_query_offsets],
            dtype=object,
        )
        row_signals = {
            "retrieval_score": owned_retrieval_scores,
            "retrieval_rank": owned_retrieval_ranks,
            "candidate_component_key": _readonly_array(owned_row_component_keys, dtype=object),
            "query_view": row_query_views,
            "query_author": row_query_authors,
            "first_name_bucket": _readonly_array(
                first_name_bucket_array(owned_normalized_row_signals["query_first_token"], row_query_views),
                dtype=object,
            ),
            **owned_normalized_row_signals,
        }

        ordered_ids: list[str] = []
        seen: set[str] = set()
        for value in (
            *owned_query_signature_ids,
            *owned_left_signature_ids,
            *owned_right_signature_ids,
        ):
            if value in seen:
                continue
            seen.add(value)
            ordered_ids.append(value)

        return cls(
            signature_order=FeatureBlockSignatureOrder(
                signature_ids=tuple(ordered_ids),
                query_signature_ids=owned_query_signature_ids,
            ),
            query_signature_ids=owned_query_signature_ids,
            query_views=owned_query_views,
            query_authors=owned_query_authors,
            seed_signature_ids=(
                None if seed_signature_ids is None else tuple(str(value) for value in seed_signature_ids)
            ),
            component_members=MappingProxyType(
                {
                    str(component_key): tuple(str(member) for member in members)
                    for component_key, members in component_members.items()
                }
            ),
            telemetry=(
                None
                if telemetry is None
                else MappingProxyType({str(key): _owned_plan_value(value) for key, value in telemetry.items()})
            ),
            row_count=len(owned_row_query_offsets),
            pair_count=len(owned_pair_row_indices),
            row_query_offsets=owned_row_query_offsets,
            left_signature_ids=owned_left_signature_ids,
            right_signature_ids=owned_right_signature_ids,
            pair_row_indices=owned_pair_row_indices,
            row_component_keys=owned_row_component_keys,
            retrieval_scores=owned_retrieval_scores,
            retrieval_ranks=owned_retrieval_ranks,
            row_signals=MappingProxyType(row_signals),
        )

    @classmethod
    def from_native_mapping(cls, plan: Mapping[str, Any]) -> RawArrowPlanBundle:
        """Validate and adopt arrays freshly returned by the native planner.

        Native planner results transfer exclusive array ownership to Python.
        Adopting those arrays avoids a second full payload copy; the adopted
        arrays are made read-only before the caller can observe the bundle.
        """

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
        telemetry = plan.get("telemetry")
        if telemetry is not None and not isinstance(telemetry, Mapping):
            raise ValueError("raw candidate plan telemetry must be a mapping")
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
        return cls._from_normalized_values(
            query_signature_ids=query_signature_ids,
            query_views=query_views,
            query_authors=query_authors,
            seed_signature_ids=plan.get("seed_signature_ids"),
            component_members=component_members,
            telemetry=telemetry,
            row_query_offsets=row_query_offsets,
            left_signature_ids=left_signature_ids,
            right_signature_ids=right_signature_ids,
            pair_row_indices=pair_row_indices,
            row_component_keys=row_component_keys,
            retrieval_scores=retrieval_scores,
            retrieval_ranks=retrieval_ranks,
            normalized_row_signals=normalized_row_signals,
        )


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
    return values.astype(np.uint8, copy=False)


def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Return normalized read-only storage from a native planner value."""

    owned = np.asarray(values, dtype=dtype, order="C")
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
        owned = np.array(value, dtype=value.dtype, copy=True, order="C")
        owned.setflags(write=False)
        return owned
    if isinstance(value, Mapping):
        return MappingProxyType({key: _owned_plan_value(item) for key, item in value.items()})
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_owned_plan_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_owned_plan_value(item) for item in value)
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
    return values


def _raw_plan_retrieval_ranks(plan: Mapping[str, Any], expected_length: int) -> np.ndarray:
    ranks = as_retrieval_rank_uint16_1d(
        "retrieval_ranks",
        _required_raw_plan_value(plan, "retrieval_ranks"),
    )
    if len(ranks) != int(expected_length):
        raise ValueError(
            f"raw candidate plan key 'retrieval_ranks' must be 1D with length {expected_length}, got {ranks.shape}"
        )
    return ranks


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
