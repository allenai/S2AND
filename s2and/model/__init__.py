"""Clustering entry points and compatibility imports."""

from __future__ import annotations

import copy as copy
import gc as gc
import hashlib as hashlib
import logging as logging
import math as math
import pickle as pickle
import sys as sys
import threading as threading
import time as time
from collections import OrderedDict as OrderedDict
from collections import defaultdict as defaultdict
from collections.abc import Callable as Callable
from collections.abc import Collection as Collection
from collections.abc import Iterable as Iterable
from collections.abc import Mapping as Mapping
from collections.abc import Sequence as Sequence
from collections.abc import Set as Set
from concurrent.futures import Future as Future
from concurrent.futures import ThreadPoolExecutor as ThreadPoolExecutor
from contextvars import ContextVar as ContextVar
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from functools import partial as partial
from functools import wraps as wraps
from pathlib import Path as Path
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import Any as Any
from typing import ParamSpec as ParamSpec
from typing import TypeVar as TypeVar
from typing import cast as cast

import numpy as np  # noqa: F401
from tqdm import tqdm as tqdm

from s2and import memory_budget as memory_budget
from s2and.arrow_inputs import ArrowDataset as ArrowDataset
from s2and.arrow_inputs import arrow_string_array as arrow_string_array
from s2and.block_corrections import BlockCorrections as BlockCorrections
from s2and.block_corrections import arrow_small_block_signatures as arrow_small_block_signatures
from s2and.block_corrections import average_linkage_cut as average_linkage_cut
from s2and.block_corrections import correction_fingerprint as correction_fingerprint
from s2and.block_corrections import corrections_copy_codes as corrections_copy_codes
from s2and.block_corrections import corrections_from_dataset as corrections_from_dataset
from s2and.block_corrections import corrections_from_rust_featurizer as corrections_from_rust_featurizer
from s2and.block_corrections import orcid_join_enabled as orcid_join_enabled
from s2and.block_corrections import year_gap_split_enabled as year_gap_split_enabled
from s2and.consts import DEFAULT_CHUNK_SIZE as DEFAULT_CHUNK_SIZE
from s2and.consts import LARGE_DISTANCE as LARGE_DISTANCE
from s2and.consts import LARGE_INTEGER as LARGE_INTEGER
from s2and.data import ANDData as ANDData
from s2and.data import _resolve_signature_splits as _resolve_signature_splits
from s2and.feature_port import RESIDUAL_FEATURIZER as RESIDUAL_FEATURIZER
from s2and.feature_port import _get_rust_feature_data as _get_rust_feature_data
from s2and.feature_port import _get_rust_featurizer as _get_rust_featurizer
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset as build_rust_featurizer_from_arrow_dataset
from s2and.feature_port import release_rust_handle as release_rust_handle
from s2and.featurizer import FeaturizationInfo as FeaturizationInfo
from s2and.featurizer import many_pairs_featurize as many_pairs_featurize
from s2and.incremental_linking.attach_reuse import _estimator_cache_fingerprint as _estimator_cache_fingerprint
from s2and.incremental_linking.completion import ResidualClusterer as ResidualClusterer
from s2and.incremental_linking.completion_metadata import (
    name_tuples_for_incremental_rules as _name_tuples_for_incremental_rules,  # noqa: F401
)
from s2and.incremental_linking.feature_block import (
    normalize_cluster_seed_disallow_pairs as normalize_cluster_seed_disallow_pairs,
)
from s2and.incremental_linking.feature_block import temporary_cluster_seed_sidecars as temporary_cluster_seed_sidecars
from s2and.incremental_linking.policy import NAME_COUNTS_MANIFEST_SHA256_FIELD as NAME_COUNTS_MANIFEST_SHA256_FIELD
from s2and.incremental_linking.policy import clusterer_uses_embedding_features as clusterer_uses_embedding_features
from s2and.incremental_linking.policy import clusterer_uses_name_count_features as clusterer_uses_name_count_features
from s2and.incremental_linking.policy import request_cluster_seed_disallow_parts as request_cluster_seed_disallow_parts
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer as _require_arrow_name_counts_index_for_clusterer,  # noqa: F401
)
from s2and.incremental_linking.policy import (
    require_dataset_name_counts_binding_for_clusterer as require_dataset_name_counts_binding_for_clusterer,
)
from s2and.incremental_linking.policy import require_name_counts_manifest_sha256 as require_name_counts_manifest_sha256
from s2and.incremental_linking.policy import (
    require_rust_featurizer_name_counts_binding_for_clusterer,  # noqa: F401
)
from s2and.incremental_linking.policy import resolve_load_name_counts_policy as resolve_load_name_counts_policy
from s2and.incremental_linking.production import _DirectArrowIncrementalDataset as _DirectArrowIncrementalDataset
from s2and.incremental_linking.production import (
    predict_incremental_promoted_linker_from_arrow as predict_incremental_promoted_linker_from_arrow,
)
from s2and.incremental_linking.production import start_background_seed_planner as start_background_seed_planner
from s2and.metrics import b3_precision_recall_fscore as b3_precision_recall_fscore
from s2and.model_pairwise import FastCluster as FastCluster
from s2and.model_pairwise import PairwiseModeler as PairwiseModeler
from s2and.model_pairwise import clone_cluster_model as clone_cluster_model
from s2and.model_pairwise import intify as intify
from s2and.model_pairwise import native_pairwise_scorers as native_pairwise_scorers
from s2and.model_pairwise import predict_pairwise_class0 as predict_pairwise_class0
from s2and.name_conflict_repair import DEFAULT_MAX_AVERAGE_DISTANCE as DEFAULT_MAX_AVERAGE_DISTANCE
from s2and.orcid_join import join_orcids_across_subblocks as join_orcids_across_subblocks
from s2and.orcid_prefix_counts import DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256 as DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256
from s2and.orcid_prefix_counts import ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD as ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD
from s2and.orcid_prefix_counts import load_packaged_orcid_prefix_counts as load_packaged_orcid_prefix_counts
from s2and.prediction_state import PredictionState as PredictionState
from s2and.profile_exclusions import enforce_profile_exclusions_in_block as enforce_profile_exclusions_in_block
from s2and.profile_exclusions import evict_profile_exclusion_violations as evict_profile_exclusion_violations
from s2and.profile_exclusions import normalize_profile_exclusions as normalize_profile_exclusions
from s2and.profile_exclusions import validate_profile_exclusions as validate_profile_exclusions
from s2and.runtime import RuntimeContext as RuntimeContext
from s2and.runtime import build_runtime_context as build_runtime_context
from s2and.runtime import dataset_stage_uses_rust as dataset_stage_uses_rust
from s2and.runtime import stage_uses_rust as stage_uses_rust
from s2and.rust_calls import (
    _get_constraint_labels_block_upper_triangle_indexed_rust,  # noqa: F401
)
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust as build_block_upper_triangle_feature_matrix_indexed_rust,
)
from s2and.rust_calls import (
    get_constraints_block_upper_triangle_indexed_rust as get_constraints_block_upper_triangle_indexed_rust,
)
from s2and.rust_calls import get_constraints_matrix_indexed_rust as get_constraints_matrix_indexed_rust
from s2and.rust_calls import take_feature_columns_rust as take_feature_columns_rust
from s2and.seed_merging import filter_generated_seed_negatives as filter_generated_seed_negatives
from s2and.seed_merging import merge_seed_labels as merge_seed_labels
from s2and.seed_merging import pack_seed_components as pack_seed_components
from s2and.seed_merging import restore_seed_membership as restore_seed_membership
from s2and.seed_merging import seed_disallow_adjacency as seed_disallow_adjacency
from s2and.subblocking import GraphSubblockingConfig as GraphSubblockingConfig
from s2and.subblocking import _make_subblocks_with_telemetry_arrow_rust as _make_subblocks_with_telemetry_arrow_rust
from s2and.subblocking import _resolve_graph_subblocking_config as _resolve_graph_subblocking_config
from s2and.subblocking import subblocking_name_parts as subblocking_name_parts
from s2and.supervision import effective_dataset_disallows as effective_dataset_disallows
from s2and.supervision import normalize_partial_supervision as normalize_partial_supervision
from s2and.supervision import validate_incremental_seed_disallows as validate_incremental_seed_disallows
from s2and.supervision import validate_seed_group_separations as validate_seed_group_separations
from s2and.text import canonicalize_name_parts as canonicalize_name_parts
from s2and.text import normalize_orcid as normalize_orcid
from s2and.text import normalize_orcid_compact as normalize_orcid_compact
from s2and.thread_config import resolve_n_jobs as resolve_n_jobs

from ._shared import _ARROW_ATTACH_BLOCK_STATE as _ARROW_ATTACH_BLOCK_STATE
from ._shared import _ARROW_ATTACH_NAME_CHECK_EXEMPT as _ARROW_ATTACH_NAME_CHECK_EXEMPT
from ._shared import _ARROW_BULK_DISTANCE_OVERRIDES as _ARROW_BULK_DISTANCE_OVERRIDES
from ._shared import _P as _P
from ._shared import _R as _R
from ._shared import _clusterer_with_prediction_params as _clusterer_with_prediction_params
from ._shared import _gc_paused as _gc_paused
from ._shared import _prediction_cluster_model_params as _prediction_cluster_model_params
from ._shared import logger as logger
from .bulk import _allocate_subblock_key as _allocate_subblock_key
from .bulk import _combine_subblocked_predictions as _combine_subblocked_predictions
from .bulk import _group_subblocking_firsts as _group_subblocking_firsts
from .bulk import _join_arrow_subblock_clusters as _join_arrow_subblock_clusters
from .bulk import _load_arrow_canonical_name_parts as _load_arrow_canonical_name_parts
from .bulk import _subblocked_prediction_scopes as _subblocked_prediction_scopes
from .bulk import _subblocking_firsts as _subblocking_firsts
from .classic import _prediction_state_from_dataset as _prediction_state_from_dataset
from .classic import _validate_python_seed_separations as _validate_python_seed_separations
from .clusterer import Clusterer as Clusterer
from .clusterer import _propagate_n_jobs as _propagate_n_jobs
from .constraints import DEFAULT_ORCID_NAME_WAIVER as DEFAULT_ORCID_NAME_WAIVER
from .constraints import _accumulate_constraint_telemetry as _accumulate_constraint_telemetry
from .constraints import _build_incremental_constraint_backend as _build_incremental_constraint_backend
from .constraints import _build_signature_index_by_id as _build_signature_index_by_id
from .constraints import _cluster_seed_disallows_for_request as _cluster_seed_disallows_for_request
from .constraints import _cluster_seeds_require_from_rust_featurizer as _cluster_seeds_require_from_rust_featurizer
from .constraints import _ConstraintBatchTelemetry as _ConstraintBatchTelemetry
from .constraints import _ConstraintPolicy as _ConstraintPolicy
from .constraints import _ConstraintTelemetryAccumulator as _ConstraintTelemetryAccumulator
from .constraints import _IncrementalConstraintBackend as _IncrementalConstraintBackend
from .constraints import _initialize_incremental_constraint_backend as _initialize_incremental_constraint_backend
from .constraints import _normalize_cluster_seeds_require as _normalize_cluster_seeds_require
from .constraints import _orcid_name_waiver_setting as _orcid_name_waiver_setting
from .constraints import (
    _partial_supervision_with_cluster_seed_disallows as _partial_supervision_with_cluster_seed_disallows,
)
from .constraints import (
    _partial_supervision_with_cluster_seed_overrides as _partial_supervision_with_cluster_seed_overrides,
)
from .constraints import _profile_exclusion_kwargs as _profile_exclusion_kwargs
from .constraints import _resolve_constraint_api_mode as _resolve_constraint_api_mode
from .constraints import _resolve_constraint_labels_batch as _resolve_constraint_labels_batch
from .constraints import _RustFeaturizerPredictDataset as _RustFeaturizerPredictDataset
from .constraints import (
    _set_partial_supervision_if_absent_bidirectional as _set_partial_supervision_if_absent_bidirectional,
)
from .constraints import _use_rust_constraints as _use_rust_constraints
from .distances import _CONCURRENT_DISTANCE_BLOCK_MAX_SIZE as _CONCURRENT_DISTANCE_BLOCK_MAX_SIZE
from .distances import _CONCURRENT_DISTANCE_BLOCKS as _CONCURRENT_DISTANCE_BLOCKS
from .distances import _FUSED_BLOCK_SCORING as _FUSED_BLOCK_SCORING
from .distances import _ONE_PASS_BLOCK_FILL as _ONE_PASS_BLOCK_FILL
from .distances import _PREDICT_FEATURE_COPY_MAX_BYTES as _PREDICT_FEATURE_COPY_MAX_BYTES
from .distances import _build_block_feature_matrices_indexed_rust as _build_block_feature_matrices_indexed_rust
from .distances import (
    _build_partial_supervision_offset_maps_for_block as _build_partial_supervision_offset_maps_for_block,
)
from .distances import _compute_predict_batch_chunk_plan as _compute_predict_batch_chunk_plan
from .distances import _condensed_pair_index as _condensed_pair_index
from .distances import _count_selected_features as _count_selected_features
from .distances import _DistanceMatrixChunk as _DistanceMatrixChunk
from .distances import _ensure_lightgbm_fitted as _ensure_lightgbm_fitted
from .distances import _guard_predict_block_matrix_allocation as _guard_predict_block_matrix_allocation
from .distances import _native_block_scorers as _native_block_scorers
from .distances import _one_pass_tile_overrides as _one_pass_tile_overrides
from .distances import _partial_supervision_label_overrides_for_block as _partial_supervision_label_overrides_for_block
from .distances import _predict_and_combine as _predict_and_combine
from .distances import _predict_class0_with_runtime as _predict_class0_with_runtime
from .distances import _predict_distance_matrix_bytes as _predict_distance_matrix_bytes
from .distances import _PredictedDistanceMatrixBatch as _PredictedDistanceMatrixBatch
from .distances import _PredictedDistanceMatrixChunk as _PredictedDistanceMatrixChunk
from .distances import _selected_feature_indices as _selected_feature_indices
from .distances import _upper_triangle_indices_for_range as _upper_triangle_indices_for_range
from .incremental import _ALTERED_PRESPLIT_CACHE_INIT_LOCK as _ALTERED_PRESPLIT_CACHE_INIT_LOCK
from .incremental import _ALTERED_PRESPLIT_CACHE_MAX_ENTRIES as _ALTERED_PRESPLIT_CACHE_MAX_ENTRIES
from .incremental import _NO_ORCID_SIGNATURE_INFO as _NO_ORCID_SIGNATURE_INFO
from .incremental import _altered_cluster_signatures_from_values as _altered_cluster_signatures_from_values
from .incremental import _altered_presplit_cache_key as _altered_presplit_cache_key
from .incremental import _altered_presplit_cache_state as _altered_presplit_cache_state
from .incremental import _AlteredPresplitJob as _AlteredPresplitJob
from .incremental import _ArrowIncrementalSignatureInfo as _ArrowIncrementalSignatureInfo
from .incremental import _AttachmentBlockState as _AttachmentBlockState
from .incremental import _cacheable_value as _cacheable_value
from .incremental import _can_skip_orcid_homogeneous_altered_presplit as _can_skip_orcid_homogeneous_altered_presplit
from .incremental import _cluster_seeds_require_inverse as _cluster_seeds_require_inverse
from .incremental import _dataset_altered_cluster_signatures as _dataset_altered_cluster_signatures
from .incremental import _get_altered_presplit_cache_entry as _get_altered_presplit_cache_entry
from .incremental import _has_in_profile_pair as _has_in_profile_pair
from .incremental import _IncrementalSeedSetup as _IncrementalSeedSetup
from .incremental import _load_arrow_incremental_orcid_signature_info as _load_arrow_incremental_orcid_signature_info
from .incremental import _load_arrow_incremental_signature_info as _load_arrow_incremental_signature_info
from .incremental import _model_presplit_cache_fingerprint as _model_presplit_cache_fingerprint
from .incremental import _normalized_orcid_for_presplit_skip as _normalized_orcid_for_presplit_skip
from .incremental import _optional_arrow_string as _optional_arrow_string
from .incremental import _partial_supervision_cache_fingerprint as _partial_supervision_cache_fingerprint
from .incremental import _put_altered_presplit_cache_entry as _put_altered_presplit_cache_entry
from .incremental import _required_incremental_linker_artifact as _required_incremental_linker_artifact
from .incremental import _seed_cluster_count_from_seed_map as _seed_cluster_count_from_seed_map
from .incremental import _signature_batch_indices_for_request as _signature_batch_indices_for_request

_ARROW_BULK_DISTANCE_OVERRIDES: ContextVar[bool]
_ARROW_ATTACH_NAME_CHECK_EXEMPT: ContextVar[frozenset[str] | None]
_ARROW_ATTACH_BLOCK_STATE: ContextVar[_AttachmentBlockState | None]
DEFAULT_ORCID_NAME_WAIVER: tuple[int, float]
_one_pass_tile_overrides: dict[str, int]

__all__ = [
    "ANDData",
    "Any",
    "ArrowDataset",
    "BlockCorrections",
    "Callable",
    "Clusterer",
    "Collection",
    "ContextVar",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_AVERAGE_DISTANCE",
    "DEFAULT_ORCID_NAME_WAIVER",
    "DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256",
    "FastCluster",
    "FeaturizationInfo",
    "Future",
    "GraphSubblockingConfig",
    "Iterable",
    "LARGE_DISTANCE",
    "LARGE_INTEGER",
    "Mapping",
    "NAME_COUNTS_MANIFEST_SHA256_FIELD",
    "ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD",
    "OrderedDict",
    "PairwiseModeler",
    "ParamSpec",
    "Path",
    "PredictionState",
    "RESIDUAL_FEATURIZER",
    "ResidualClusterer",
    "RuntimeContext",
    "Sequence",
    "Set",
    "TYPE_CHECKING",
    "ThreadPoolExecutor",
    "TypeVar",
    "annotations",
    "arrow_small_block_signatures",
    "arrow_string_array",
    "average_linkage_cut",
    "b3_precision_recall_fscore",
    "build_block_upper_triangle_feature_matrix_indexed_rust",
    "build_runtime_context",
    "build_rust_featurizer_from_arrow_dataset",
    "canonicalize_name_parts",
    "cast",
    "clone_cluster_model",
    "clusterer_uses_embedding_features",
    "clusterer_uses_name_count_features",
    "copy",
    "correction_fingerprint",
    "corrections_copy_codes",
    "corrections_from_dataset",
    "corrections_from_rust_featurizer",
    "dataclass",
    "dataset_stage_uses_rust",
    "defaultdict",
    "effective_dataset_disallows",
    "enforce_profile_exclusions_in_block",
    "evict_profile_exclusion_violations",
    "field",
    "filter_generated_seed_negatives",
    "gc",
    "get_constraints_block_upper_triangle_indexed_rust",
    "get_constraints_matrix_indexed_rust",
    "hashlib",
    "intify",
    "join_orcids_across_subblocks",
    "load_packaged_orcid_prefix_counts",
    "logger",
    "logging",
    "many_pairs_featurize",
    "math",
    "memory_budget",
    "merge_seed_labels",
    "native_pairwise_scorers",
    "normalize_cluster_seed_disallow_pairs",
    "normalize_orcid",
    "normalize_orcid_compact",
    "normalize_partial_supervision",
    "normalize_profile_exclusions",
    "np",
    "orcid_join_enabled",
    "pack_seed_components",
    "partial",
    "pickle",
    "predict_incremental_promoted_linker_from_arrow",
    "predict_pairwise_class0",
    "release_rust_handle",
    "request_cluster_seed_disallow_parts",
    "require_dataset_name_counts_binding_for_clusterer",
    "require_name_counts_manifest_sha256",
    "require_rust_featurizer_name_counts_binding_for_clusterer",
    "resolve_load_name_counts_policy",
    "resolve_n_jobs",
    "restore_seed_membership",
    "seed_disallow_adjacency",
    "stage_uses_rust",
    "start_background_seed_planner",
    "subblocking_name_parts",
    "sys",
    "take_feature_columns_rust",
    "temporary_cluster_seed_sidecars",
    "threading",
    "time",
    "tqdm",
    "validate_incremental_seed_disallows",
    "validate_profile_exclusions",
    "validate_seed_group_separations",
    "wraps",
    "year_gap_split_enabled",
]
