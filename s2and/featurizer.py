import contextlib
import functools
import gc
import json
import logging
import os
import platform
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from s2and import feature_port, memory_budget
from s2and.consts import (
    CACHE_ROOT,
    DEFAULT_CHUNK_SIZE,
    FEATURIZER_VERSION,
    LARGE_INTEGER,
    NUMPY_NAN,
)
from s2and.data import ANDData
from s2and.mp import UniversalPool
from s2and.runtime import RuntimeContext, build_runtime_context, dataset_stage_uses_rust, stage_uses_rust
from s2and.text import (
    TEXT_FUNCTIONS,
    cosine_sim,
    counter_jaccard,
    diff,
    email_prefix_suffix,
    equal,
    equal_middle,
    jaccard,
    name_counts,
    name_text_features,
)

logger = logging.getLogger("s2and")

TupleOfArrays = tuple[np.ndarray, np.ndarray, np.ndarray | None]

# Environment configuration caches (read once per process for consistency)
CACHED_FEATURES: dict[str, dict[str, Any]] = {}
_CACHED_FEATURES_LOCK = threading.Lock()
global_dataset: ANDData | None = None
global_runtime_context: RuntimeContext | None = None
_RUST_BATCH_CALIBRATION_LOCK = threading.Lock()
_RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES: int | None = None
_RUST_BATCH_CALIBRATION_ATTEMPTED = False


def _use_rust_featurizer(
    runtime_context: RuntimeContext | None = None,
    dataset: ANDData | None = None,
) -> bool:
    """Return whether Rust pair featurization applies.

    With a dataset, this is the dataset-shape decision: Rust featurizers are
    built exclusively from Arrow artifacts, so datasets without
    ``arrow_paths`` use the Python featurizer (or raise when
    the Rust backend was requested explicitly). Without a dataset, this is the
    backend-level decision only.
    """

    if runtime_context is None:
        runtime_context = (
            dataset.runtime_context
            if dataset is not None
            else build_runtime_context(
                "pair_featurization",
                backend="python",
            )
        )
    if dataset is None:
        return stage_uses_rust(runtime_context)
    return dataset_stage_uses_rust(runtime_context, dataset)


def _has_missing_signature_ngrams_for_pairs(
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
) -> tuple[bool, int]:
    signatures = getattr(dataset, "signatures", {})
    if not signatures:
        return False, 0

    inspected = 0
    inspected_signature_ids = set()
    for sig_id_1, sig_id_2, _ in signature_pairs:
        for signature_id in (sig_id_1, sig_id_2):
            if signature_id in inspected_signature_ids:
                continue
            inspected_signature_ids.add(signature_id)
            signature = signatures.get(signature_id)
            if signature is None:
                continue
            inspected += 1
            if signature.author_info_affiliations_n_grams is None or signature.author_info_coauthor_n_grams is None:
                return True, inspected
    return False, inspected


def _ensure_python_pair_signature_ngrams(
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
    runtime_context: RuntimeContext,
) -> None:
    if _use_rust_featurizer(runtime_context, dataset):
        return
    rust_lifecycle_policy = getattr(dataset, "rust_lifecycle_policy", None)
    if bool(getattr(rust_lifecycle_policy, "defer_signature_fields_to_rust", False)) and signature_pairs:
        raise RuntimeError(
            "Python featurization cannot run on a dataset built with normalized signature fields deferred to Rust. "
            "Rebuild the dataset with S2AND_BACKEND=python/backend='python', or keep Rust featurization active with "
            "validated Arrow featurizer paths."
        )
    if getattr(dataset, "_s2and_python_pair_ngrams_ready", False):
        return

    materialize_fn = getattr(dataset, "materialize_signature_ngrams_python", None)
    if materialize_fn is None:
        return

    has_missing_ngrams, inspected_signature_count = _has_missing_signature_ngrams_for_pairs(dataset, signature_pairs)
    if not has_missing_ngrams:
        if inspected_signature_count == len(getattr(dataset, "signatures", {})):
            dataset._s2and_python_pair_ngrams_ready = True
        return

    materialize_start = time.perf_counter()
    materialize_fn()
    dataset._s2and_python_pair_ngrams_ready = True
    logger.info(
        "Telemetry stage: stage=python_pair_signature_ngrams_materialize seconds=%.3f "
        "inspected_signatures=%d total_signatures=%d backend=%s run_id=%s",
        time.perf_counter() - materialize_start,
        inspected_signature_count,
        len(getattr(dataset, "signatures", {})),
        runtime_context.backend,
        runtime_context.run_id,
    )


def _rust_batch_probe_row_counts(total_pairs: int, *, probe_count: int, min_total_pairs: int) -> list[int]:
    bounded_total_pairs = max(0, int(total_pairs))
    if probe_count <= 0:
        return []
    if bounded_total_pairs < max(int(min_total_pairs), int(probe_count)):
        return []

    canonical = [10_000, 50_000, 100_000]
    canonical_fits = [value for value in canonical if value <= bounded_total_pairs]
    if len(canonical_fits) >= probe_count:
        return canonical_fits[:probe_count]

    quantile_rows: list[int] = []
    for probe_index in range(probe_count):
        quantile = float(probe_index + 1) / float(probe_count)
        row_count = int(round(float(bounded_total_pairs) * quantile))
        row_count = max(1, min(bounded_total_pairs, row_count))
        quantile_rows.append(row_count)

    deduped = sorted(set(quantile_rows))
    if len(deduped) < probe_count:
        return []
    return deduped[-probe_count:]


def _prefault_scratch_array_pages_inplace(array: np.ndarray) -> None:
    """Fault virtual pages into RSS by writing zeros into a scratch buffer.

    This intentionally mutates the provided array and must only be used with
    scratch buffers that are immediately overwritten.
    """
    if array.size <= 0:
        return
    byte_view = array.view(np.uint8).ravel()
    byte_view[0] = 0
    byte_view[-1] = 0
    byte_view[::4096] = 0


def _maybe_calibrate_rust_batch_fixed_overhead_bytes(
    *,
    rust_featurizer: Any,
    pieces_of_work: list[tuple[tuple[str, str], int]],
    signature_id_to_index: dict[Any, int],
    rust_selected_indices: list[int] | None,
    selected_feature_count: int,
    nameless_feature_count: int,
    row_overhead_bytes: int,
    persistent_row_overhead_bytes: int,
    configured_fixed_overhead_bytes: int,
    num_threads: int,
    total_ram_for_stage: int | None,
    run_id: str,
) -> int | None:
    global _RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES
    global _RUST_BATCH_CALIBRATION_ATTEMPTED

    probe_count = 3
    min_total_pairs = 30_000

    with _RUST_BATCH_CALIBRATION_LOCK:
        if _RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES is not None:
            return int(_RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES)
        if _RUST_BATCH_CALIBRATION_ATTEMPTED:
            return None
        _RUST_BATCH_CALIBRATION_ATTEMPTED = True

    probe_rows = _rust_batch_probe_row_counts(
        len(pieces_of_work),
        probe_count=probe_count,
        min_total_pairs=min_total_pairs,
    )
    if len(probe_rows) < probe_count:
        return None
    if total_ram_for_stage is None:
        return None

    fixed_samples: list[int] = []
    chunk_feature_count = max(1, int(selected_feature_count) + int(nameless_feature_count))
    row_overhead_bounded = max(0, int(row_overhead_bytes))
    persistent_row_overhead_bounded = max(0, int(persistent_row_overhead_bytes))

    try:
        for row_count in probe_rows:
            probe_work = pieces_of_work[: int(row_count)]
            if len(probe_work) < int(row_count):
                continue
            probe_pairs = [pair for pair, _ in probe_work]
            rss_before_bytes, _ = memory_budget.current_rss_bytes_best_effort(total_ram_for_stage)
            rss_peak_bytes = rss_before_bytes

            probe_features = np.empty((int(row_count), int(selected_feature_count)), dtype=np.float64)
            probe_nameless_features: np.ndarray | None = None
            if int(nameless_feature_count) > 0:
                probe_nameless_features = np.empty((int(row_count), int(nameless_feature_count)), dtype=np.float64)
            probe_labels = np.empty(int(row_count), dtype=np.float64)
            _prefault_scratch_array_pages_inplace(probe_features)
            if probe_nameless_features is not None:
                _prefault_scratch_array_pages_inplace(probe_nameless_features)
            _prefault_scratch_array_pages_inplace(probe_labels)

            rss_now_bytes, _ = memory_budget.current_rss_bytes_best_effort(total_ram_for_stage)
            rss_peak_bytes = max(rss_peak_bytes, int(rss_now_bytes))

            probe_pairs_indexed = [
                (
                    _signature_id_to_index_or_raise(signature_id_to_index, pair[0]),
                    _signature_id_to_index_or_raise(signature_id_to_index, pair[1]),
                )
                for pair in probe_pairs
            ]
            probe_chunk = np.asarray(
                rust_featurizer.featurize_pairs_matrix_indexed(
                    probe_pairs_indexed,
                    rust_selected_indices,
                    int(num_threads),
                    np.nan,
                ),
                dtype=np.float64,
            )

            rss_after_call_bytes, _ = memory_budget.current_rss_bytes_best_effort(total_ram_for_stage)
            rss_peak_bytes = max(rss_peak_bytes, int(rss_after_call_bytes))
            observed_peak_delta_bytes = int(rss_peak_bytes) - int(rss_before_bytes)

            modeled_features_bytes = int(row_count) * int(selected_feature_count + nameless_feature_count) * 8
            modeled_labels_bytes = int(row_count) * 8
            modeled_chunk_bytes = int(row_count) * int(chunk_feature_count * 8 + row_overhead_bounded)
            modeled_persistent_bytes = int(row_count) * int(persistent_row_overhead_bounded)
            estimated_fixed_bytes = int(
                observed_peak_delta_bytes
                - modeled_features_bytes
                - modeled_labels_bytes
                - modeled_chunk_bytes
                - modeled_persistent_bytes
            )
            fixed_samples.append(estimated_fixed_bytes)

            del probe_features
            del probe_nameless_features
            del probe_labels
            del probe_chunk
            gc.collect()
    except Exception as exc:
        with _RUST_BATCH_CALIBRATION_LOCK:
            _RUST_BATCH_CALIBRATION_ATTEMPTED = True
        logger.warning(
            "Rust batch startup calibration failed; using configured fixed overhead " "(run_id=%s error=%s)",
            run_id,
            exc,
        )
        return None

    if not fixed_samples:
        return None

    observed_fixed_bytes_estimate = max(0, int(max(fixed_samples)))
    observed_fixed_bytes_estimate = min(observed_fixed_bytes_estimate, 512 * (1 << 20))
    configured_fixed_bytes = max(0, int(configured_fixed_overhead_bytes))
    if observed_fixed_bytes_estimate <= int(float(configured_fixed_bytes) * 1.2):
        with _RUST_BATCH_CALIBRATION_LOCK:
            _RUST_BATCH_CALIBRATION_ATTEMPTED = True
        return None

    calibrated_fixed_bytes = max(configured_fixed_bytes, observed_fixed_bytes_estimate)

    with _RUST_BATCH_CALIBRATION_LOCK:
        _RUST_BATCH_CALIBRATED_FIXED_OVERHEAD_BYTES = int(calibrated_fixed_bytes)
        _RUST_BATCH_CALIBRATION_ATTEMPTED = True

    logger.info(
        "Telemetry: rust_batch_startup_calibration probes=%d probe_rows=%s "
        "fixed_overhead_bytes_calibrated=%d row_overhead_bytes=%d persistent_row_overhead_bytes=%d run_id=%s",
        len(probe_rows),
        ",".join(str(v) for v in probe_rows),
        int(calibrated_fixed_bytes),
        int(row_overhead_bounded),
        int(persistent_row_overhead_bounded),
        run_id,
    )
    return int(calibrated_fixed_bytes)


def _log_featurization_backend_decision(
    runtime_context: RuntimeContext,
    pieces_of_work_count: int,
    n_jobs: int,
    use_rust_featurizer: bool,
    rust_module_available: bool,
) -> None:
    if pieces_of_work_count <= 0:
        logger.info("Featurization backend decision: skipped compute (all pairs were cached or pre-labeled)")
        return

    if use_rust_featurizer and rust_module_available:
        backend = "rust_batch"
    else:
        backend = "python_parallel" if n_jobs > 1 else "python_serial"

    logger.info(
        "Featurization backend decision: backend=%s pieces=%d n_jobs=%d "
        "use_rust_featurizer=%s rust_module_available=%s "
        "runtime_backend=%s run_id=%s",
        backend,
        pieces_of_work_count,
        n_jobs,
        use_rust_featurizer,
        rust_module_available,
        runtime_context.backend,
        runtime_context.run_id,
    )

    notes = []
    if not use_rust_featurizer:
        notes.append("pair_featurization stage set to Python by runtime context")
    if use_rust_featurizer and not rust_module_available:
        notes.append("s2and_rust extension unavailable")
    if notes:
        logger.info("Featurization backend notes: %s", "; ".join(notes))


def _contiguous_index_slice(indices: list[int]) -> slice | None:
    if not indices:
        return None
    start = int(indices[0])
    for offset, index in enumerate(indices):
        if int(index) != start + offset:
            return None
    return slice(start, start + len(indices))


@dataclass(frozen=True)
class ScatterContext:
    features: np.ndarray
    nameless_features: np.ndarray | None
    coauthor_similarity_values: np.ndarray | None
    identity_selected_indices: bool
    indices_to_use: list[int]
    nameless_indices_to_use: list[int]
    selected_positions: list[int]
    nameless_positions: list[int]
    coauthor_similarity_index: int | None
    coauthor_position: int | None


@dataclass(frozen=True)
class RustBatchExecutionResult:
    rust_batch_plan: memory_budget.RustBatchChunkPlan
    new_features_count: int
    rust_batch_total_ram_for_stage: int | None
    rust_batch_rss_before_bytes: int
    rust_batch_rss_peak_bytes: int
    rust_batch_rss_source: str
    rust_batch_adaptive_halvings: int


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Resolved internal cache decisions for a public ``use_cache`` value."""

    pair_feature_cache_enabled: bool

    @property
    def requires_full_feature_rows(self) -> bool:
        return self.pair_feature_cache_enabled


def resolve_cache_policy(use_cache: bool) -> CachePolicy:
    """Resolve the public cache flag into the internal cache policy."""
    enabled = bool(use_cache)
    return CachePolicy(pair_feature_cache_enabled=enabled)


def _scatter_feature_row_from_source(
    *,
    feature_output: np.ndarray,
    output_index: int,
    scatter_context: ScatterContext,
    rust_chunk_is_full: bool,
) -> None:
    features = scatter_context.features
    nameless_features = scatter_context.nameless_features
    coauthor_similarity_values = scatter_context.coauthor_similarity_values
    if rust_chunk_is_full:
        if scatter_context.identity_selected_indices:
            features[output_index, :] = feature_output
        else:
            features[output_index, :] = feature_output[scatter_context.indices_to_use]
        if nameless_features is not None:
            nameless_features[output_index, :] = feature_output[scatter_context.nameless_indices_to_use]
        if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
            coauthor_similarity_values[output_index] = feature_output[scatter_context.coauthor_similarity_index]
        return

    features[output_index, :] = feature_output[scatter_context.selected_positions]
    if nameless_features is not None:
        nameless_features[output_index, :] = feature_output[scatter_context.nameless_positions]
    if (
        coauthor_similarity_values is not None
        and scatter_context.coauthor_similarity_index is not None
        and scatter_context.coauthor_position is not None
    ):
        coauthor_similarity_values[output_index] = feature_output[scatter_context.coauthor_position]


def _scatter_chunk_to_output(
    *,
    rust_features_chunk: np.ndarray,
    chunk_indices: list[int],
    scatter_context: ScatterContext,
    rust_chunk_is_full: bool,
) -> None:
    features = scatter_context.features
    nameless_features = scatter_context.nameless_features
    coauthor_similarity_values = scatter_context.coauthor_similarity_values
    chunk_slice = _contiguous_index_slice(chunk_indices)
    if chunk_slice is not None:
        if rust_chunk_is_full:
            if scatter_context.identity_selected_indices:
                features[chunk_slice, :] = rust_features_chunk
            else:
                np.take(
                    rust_features_chunk,
                    scatter_context.indices_to_use,
                    axis=1,
                    out=features[chunk_slice, :],
                )
            if nameless_features is not None:
                np.take(
                    rust_features_chunk,
                    scatter_context.nameless_indices_to_use,
                    axis=1,
                    out=nameless_features[chunk_slice, :],
                )
            if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
                coauthor_similarity_values[chunk_slice] = rust_features_chunk[
                    :,
                    scatter_context.coauthor_similarity_index,
                ]
            return

        np.take(
            rust_features_chunk,
            scatter_context.selected_positions,
            axis=1,
            out=features[chunk_slice, :],
        )
        if nameless_features is not None:
            np.take(
                rust_features_chunk,
                scatter_context.nameless_positions,
                axis=1,
                out=nameless_features[chunk_slice, :],
            )
        if (
            coauthor_similarity_values is not None
            and scatter_context.coauthor_similarity_index is not None
            and scatter_context.coauthor_position is not None
        ):
            coauthor_similarity_values[chunk_slice] = rust_features_chunk[:, scatter_context.coauthor_position]
        return

    if rust_chunk_is_full:
        if scatter_context.identity_selected_indices:
            features[chunk_indices, :] = rust_features_chunk
        else:
            features[chunk_indices, :] = rust_features_chunk[:, scatter_context.indices_to_use]
        if nameless_features is not None:
            nameless_features[chunk_indices, :] = rust_features_chunk[:, scatter_context.nameless_indices_to_use]
        if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
            coauthor_similarity_values[chunk_indices] = rust_features_chunk[
                :,
                scatter_context.coauthor_similarity_index,
            ]
        return

    features[chunk_indices, :] = rust_features_chunk[:, scatter_context.selected_positions]
    if nameless_features is not None:
        nameless_features[chunk_indices, :] = rust_features_chunk[:, scatter_context.nameless_positions]
    if (
        coauthor_similarity_values is not None
        and scatter_context.coauthor_similarity_index is not None
        and scatter_context.coauthor_position is not None
    ):
        coauthor_similarity_values[chunk_indices] = rust_features_chunk[:, scatter_context.coauthor_position]


def _cache_feature_output(
    *,
    cached_features: dict[str, Any],
    cache_key: str,
    feature_output: np.ndarray | list[int | float],
) -> None:
    # Preserve cache value immutability and avoid aliasing when worker buffers are reused.
    feature_output_cached = np.asarray(feature_output, dtype=np.float64).copy()
    cached_features["features"][cache_key] = feature_output_cached
    cached_features["__new_features__"][cache_key] = feature_output_cached


def _write_feature_row(
    *,
    feature_output: np.ndarray | list[int | float],
    output_index: int,
    signature_pairs: list[tuple[str, str, int | float]],
    featurizer_info: "FeaturizationInfo",
    cached_features: dict[str, Any],
    use_cache: bool,
    scatter_context: ScatterContext,
    source_is_full: bool,
) -> int:
    if use_cache:
        cache_key = featurizer_info.feature_cache_key(signature_pairs[output_index])
        _cache_feature_output(
            cached_features=cached_features,
            cache_key=cache_key,
            feature_output=feature_output,
        )
    _scatter_feature_row_from_source(
        feature_output=np.asarray(feature_output, dtype=np.float64),
        output_index=int(output_index),
        scatter_context=scatter_context,
        rust_chunk_is_full=source_is_full,
    )
    return 1 if use_cache else 0


# ── constants for cache writes ───────────────────────────
INCREMENTAL_WRITE_THRESHOLD = 1000
PAIR_FEATURE_CACHE_DB_FILENAME = "pair_features.sqlite3"
PAIR_FEATURE_CACHE_SCHEMA_VERSION = 1
PAIR_FEATURE_CACHE_WRITE_MAX_RETRIES = 3
PAIR_FEATURE_CACHE_BUSY_TIMEOUT_SECONDS = 30.0
PAIR_FEATURE_CACHE_WRITE_BACKOFF_SECONDS = 0.1
_FEATURE_CACHE_METADATA_KEY_SCHEMA_VERSION = "schema_version"
_FEATURE_CACHE_METADATA_KEY_FEATURE_COUNT = "feature_count"
_FEATURE_CACHE_METADATA_KEY_FEATURES_TO_USE_JSON = "features_to_use_json"


def _signature_id_to_index_or_raise(signature_id_to_index: dict[Any, int], signature_id: Any) -> int:
    if signature_id in signature_id_to_index:
        return int(signature_id_to_index[signature_id])
    signature_id_str = str(signature_id)
    if signature_id_str in signature_id_to_index:
        return int(signature_id_to_index[signature_id_str])
    raise ValueError(
        "Rust indexed pair featurization received signature_id not present in Rust featurizer signature_ids: "
        f"{signature_id!r}"
    )


class FeaturizationInfo:
    """
    Class to store information about how to generate and cache features

    Inputs:
        features_to_use: List[str]
            list of feature types to use
        featurizer_version: int
            What version of the featurizer we are on. This should be
            incremented when changing how features are computed so that a new cache
            is created
    """

    def __init__(
        self,
        features_to_use: list[str] | None = None,
        featurizer_version: int = FEATURIZER_VERSION,
    ):
        if features_to_use is None:
            features_to_use = [
                "name_similarity",
                "affiliation_similarity",
                "email_similarity",
                "coauthor_similarity",
                "venue_similarity",
                "year_diff",
                "title_similarity",
                "misc_features",
                "name_counts",
                "embedding_similarity",
                "journal_similarity",
                "advanced_name_similarity",
            ]
        self.features_to_use = list(features_to_use)

        self.feature_group_to_index = {
            "name_similarity": [0, 1, 2, 3, 4, 5],
            "affiliation_similarity": [6],
            "email_similarity": [7, 8],
            "coauthor_similarity": [9, 10, 11],
            "venue_similarity": [12],
            "year_diff": [13],
            "title_similarity": [14, 15],
            "misc_features": [16, 17, 18, 19, 20],
            "name_counts": [21, 22, 23, 24, 25, 26],
            "embedding_similarity": [27],
            "journal_similarity": [28],
            "advanced_name_similarity": [29, 30, 31, 32],
        }
        unknown_feature_groups = sorted(set(self.features_to_use) - set(self.feature_group_to_index))
        if unknown_feature_groups:
            known_feature_groups = ", ".join(sorted(self.feature_group_to_index))
            raise ValueError(
                f"Unknown feature group(s): {unknown_feature_groups}. Known feature groups: {known_feature_groups}"
            )

        max_feature_index = max(
            (feature_index for group in self.feature_group_to_index.values() for feature_index in group),
            default=-1,
        )
        self.number_of_features = max_feature_index + 1

        lightgbm_monotone_constraints = {
            "name_similarity": ["1", "1", "1", "0", "0", "0"],
            "affiliation_similarity": ["0"],
            "email_similarity": ["1", "1"],
            "coauthor_similarity": ["1", "0", "1"],
            "venue_similarity": ["0"],
            "year_diff": ["-1"],
            "title_similarity": ["1", "1"],
            "misc_features": ["0", "0", "0", "0", "0"],
            "name_counts": ["0", "-1", "-1", "-1", "0", "-1"],
            "embedding_similarity": ["0"],
            "journal_similarity": ["0"],
            "advanced_name_similarity": ["0", "0", "0", "0"],
        }

        self.lightgbm_monotone_constraints = ",".join(
            [
                ",".join(constraints)
                for feature_category, constraints in lightgbm_monotone_constraints.items()
                if feature_category in features_to_use
            ]
        )
        self.nameless_lightgbm_monotone_constraints = ",".join(
            [
                ",".join(constraints)
                for feature_category, constraints in lightgbm_monotone_constraints.items()
                if feature_category in features_to_use
                and feature_category not in {"advanced_name_similarity", "name_similarity", "name_counts"}
            ]
        )

        # NOTE: Increment this anytime a change is made to the featurization logic
        self.featurizer_version = featurizer_version

    def __setstate__(self, state: dict) -> None:
        # Derived state (feature_group_to_index, number_of_features, monotone
        # constraints) must reflect the current feature layout, not the layout
        # at pickle time. Old pickles carry index maps for removed feature
        # groups (e.g. reference_features), so rebuild everything from
        # features_to_use. featurizer_version stays artifact-pinned, matching
        # how native bundles reconstruct FeaturizationInfo from metadata.
        self.__init__(
            features_to_use=list(state["features_to_use"]),
            featurizer_version=int(state.get("featurizer_version", FEATURIZER_VERSION)),
        )

    def get_feature_names(self) -> list[str]:
        """
        Gets all of the feature names

        Returns
        -------
        List[string]: List of all the features names
        """
        feature_names = []

        # name features
        if "name_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "first_names_equal",
                    "middle_initials_overlap",
                    "middle_names_equal",
                    "middle_one_missing",
                    "single_char_first",
                    "single_char_middle",
                ]
            )

        # affiliation features
        if "affiliation_similarity" in self.features_to_use:
            feature_names.append("affiliation_overlap")

        # email features
        if "email_similarity" in self.features_to_use:
            feature_names.extend(["email_prefix_equal", "email_suffix_equal"])

        # co author features
        if "coauthor_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "coauthor_overlap",
                    "coauthor_similarity",
                    "coauthor_match",
                ]
            )

        # venue features
        if "venue_similarity" in self.features_to_use:
            feature_names.append("venue_overlap")

        # year features
        if "year_diff" in self.features_to_use:
            feature_names.append("year_diff")

        # title features
        if "title_similarity" in self.features_to_use:
            feature_names.extend(["title_overlap_words", "title_overlap_chars"])

        # position features
        if "misc_features" in self.features_to_use:
            feature_names.extend(
                ["position_diff", "abstract_count", "english_count", "same_language", "language_reliability_min"]
            )

        # name count features
        if "name_counts" in self.features_to_use:
            feature_names.extend(
                [
                    "first_name_count_min",
                    "last_first_name_count_min",
                    "last_name_count_min",
                    "last_first_initial_count_min",
                    "first_name_count_max",
                    "last_first_name_count_max",
                ]
            )

        # specter features
        if "embedding_similarity" in self.features_to_use:
            feature_names.append("specter_cosine_sim")

        if "journal_similarity" in self.features_to_use:
            feature_names.append("journal_overlap")

        if "advanced_name_similarity" in self.features_to_use:
            similarity_names = [func[1] for func in TEXT_FUNCTIONS]
            feature_names.extend(similarity_names)

        return feature_names

    @staticmethod
    def feature_cache_key(signature_pair: tuple[Any, ...]) -> str:
        """
        returns the key in the feature cache dictionary for a signature pair

        Parameters
        ----------
        signature_pair: Tuple[string]
            pair of signature ids

        Returns
        -------
        string: the cache key
        """
        return f"{signature_pair[0]}___{signature_pair[1]}"

    @staticmethod
    def feature_cache_lookup_keys(signature_pair: tuple[Any, ...]) -> tuple[str, ...]:
        """Return cache lookup keys in forward-first order, including reverse when distinct."""
        forward = FeaturizationInfo.feature_cache_key(signature_pair)
        reverse = FeaturizationInfo.feature_cache_key((signature_pair[1], signature_pair[0]))
        if reverse == forward:
            return (forward,)
        return (forward, reverse)

    def cache_directory(self, dataset_name: str) -> str:
        """
        returns the cache directory for this dataset and featurizer version

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the cache directory
        """
        return os.path.join(CACHE_ROOT, dataset_name, str(self.featurizer_version))

    def cache_db_path(self, dataset_name: str) -> str:
        """
        returns the SQLite database path for the features cache

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the full file path for the SQLite cache database
        """
        return os.path.join(
            self.cache_directory(dataset_name),
            PAIR_FEATURE_CACHE_DB_FILENAME,
        )

    def cache_storage_key(self, dataset_name: str) -> str:
        """Return the canonical in-memory cache key for a dataset cache."""
        return self.cache_db_path(dataset_name)

    def _fresh_cache_payload(self) -> dict[str, Any]:
        return {
            "features": {},
            "features_to_use": list(self.features_to_use),
            "__new_features__": {},
        }

    @staticmethod
    def _sqlite_feature_blob(feature_output: np.ndarray | list[int | float]) -> sqlite3.Binary:
        feature_array = np.asarray(feature_output, dtype=np.float64)
        return sqlite3.Binary(feature_array.tobytes(order="C"))

    @staticmethod
    def _decode_sqlite_feature_blob(feature_blob: bytes | bytearray | memoryview) -> np.ndarray:
        feature_view = np.frombuffer(feature_blob, dtype=np.float64)
        if int(feature_view.shape[0]) != int(NUM_FEATURES):
            raise ValueError(
                "Pair-feature cache entry has unexpected width "
                f"expected={NUM_FEATURES} got={int(feature_view.shape[0])}"
            )
        return feature_view.copy()

    @staticmethod
    def _configure_pair_feature_cache_connection(connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA busy_timeout = {int(PAIR_FEATURE_CACHE_BUSY_TIMEOUT_SECONDS * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")

    @classmethod
    def _initialize_pair_feature_cache_schema(cls, connection: sqlite3.Connection) -> None:
        cls._configure_pair_feature_cache_connection(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pair_features (
                cache_key TEXT PRIMARY KEY,
                feature_blob BLOB NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    def _load_sqlite_cache(self, db_path: str) -> dict[str, Any]:
        cached_features = self._fresh_cache_payload()
        with sqlite3.connect(db_path, timeout=PAIR_FEATURE_CACHE_BUSY_TIMEOUT_SECONDS) as connection:
            self._initialize_pair_feature_cache_schema(connection)
            metadata_rows = dict(connection.execute("SELECT key, value FROM cache_metadata"))
            schema_version_raw = metadata_rows.get(_FEATURE_CACHE_METADATA_KEY_SCHEMA_VERSION)
            if schema_version_raw is not None and int(schema_version_raw) != int(PAIR_FEATURE_CACHE_SCHEMA_VERSION):
                raise RuntimeError(
                    "Unsupported pair-feature cache schema version "
                    f"path={db_path} expected={PAIR_FEATURE_CACHE_SCHEMA_VERSION} got={schema_version_raw}"
                )
            stored_features_to_use_json = metadata_rows.get(_FEATURE_CACHE_METADATA_KEY_FEATURES_TO_USE_JSON)
            if stored_features_to_use_json is not None:
                try:
                    stored_features_to_use = json.loads(stored_features_to_use_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid pair-feature cache features_to_use metadata at {db_path}: {exc.msg}"
                    ) from exc
                if isinstance(stored_features_to_use, list) and all(
                    isinstance(value, str) for value in stored_features_to_use
                ):
                    cached_features["features_to_use"] = list(stored_features_to_use)
            for cache_key, feature_blob in connection.execute("SELECT cache_key, feature_blob FROM pair_features"):
                cached_features["features"][str(cache_key)] = self._decode_sqlite_feature_blob(feature_blob)
        cached_features["__cache_backend__"] = "sqlite"
        cached_features["__cache_db_path__"] = db_path
        return cached_features

    def load_cache(self, dataset_name: str) -> dict[str, Any]:
        """Load the persisted pair-feature cache for a dataset."""
        db_path = self.cache_db_path(dataset_name)
        if os.path.exists(db_path):
            return self._load_sqlite_cache(db_path)
        cached_features = self._fresh_cache_payload()
        cached_features["__cache_backend__"] = "empty"
        return cached_features

    def write_cache(self, cached_features: dict, dataset_name: str, incremental: bool = False):
        """
        Writes the cached features to the SQLite-backed features cache

        Parameters
        ----------
        cached_features: Dict
            the features, keyed by signature pair
        dataset_name: str
            the name of the dataset
        incremental: bool
            if True, only write new features from __new_features__ key

        Returns
        -------
        nothing, writes the cache database
        """
        cache_dir = self.cache_directory(dataset_name)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        db_path = self.cache_db_path(dataset_name)
        full_features = cached_features.get("features", {})
        if not isinstance(full_features, dict):
            raise ValueError("cached_features['features'] must be a dictionary")
        new_features = cached_features.get("__new_features__", {})
        if not isinstance(new_features, dict):
            raise ValueError("cached_features['__new_features__'] must be a dictionary")

        replace_existing_rows = not incremental
        features_to_persist = new_features if incremental else full_features
        if len(features_to_persist) <= 0:
            return

        features_to_use_json = json.dumps(cached_features.get("features_to_use", []), separators=(",", ":"))
        for attempt in range(PAIR_FEATURE_CACHE_WRITE_MAX_RETRIES):
            try:
                with sqlite3.connect(db_path, timeout=PAIR_FEATURE_CACHE_BUSY_TIMEOUT_SECONDS) as connection:
                    self._initialize_pair_feature_cache_schema(connection)
                    with connection:
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO cache_metadata(key, value)
                            VALUES (?, ?)
                            """,
                            (_FEATURE_CACHE_METADATA_KEY_SCHEMA_VERSION, str(PAIR_FEATURE_CACHE_SCHEMA_VERSION)),
                        )
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO cache_metadata(key, value)
                            VALUES (?, ?)
                            """,
                            (_FEATURE_CACHE_METADATA_KEY_FEATURE_COUNT, str(NUM_FEATURES)),
                        )
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO cache_metadata(key, value)
                            VALUES (?, ?)
                            """,
                            (_FEATURE_CACHE_METADATA_KEY_FEATURES_TO_USE_JSON, features_to_use_json),
                        )
                        if replace_existing_rows:
                            connection.execute("DELETE FROM pair_features")
                        connection.executemany(
                            """
                            INSERT INTO pair_features(cache_key, feature_blob)
                            VALUES(?, ?)
                            ON CONFLICT(cache_key) DO UPDATE SET feature_blob=excluded.feature_blob
                            """,
                            (
                                (
                                    str(cache_key),
                                    self._sqlite_feature_blob(feature_output),
                                )
                                for cache_key, feature_output in features_to_persist.items()
                            ),
                        )
                cached_features["__new_features__"] = {}
                cached_features["__cache_backend__"] = "sqlite"
                cached_features["__cache_db_path__"] = db_path
                return
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() and attempt < PAIR_FEATURE_CACHE_WRITE_MAX_RETRIES - 1:
                    logger.warning(
                        "Pair-feature cache write attempt %d failed due to SQLite lock; retrying path=%s",
                        attempt + 1,
                        db_path,
                    )
                    time.sleep(PAIR_FEATURE_CACHE_WRITE_BACKOFF_SECONDS * (attempt + 1))
                    continue
                logger.error(
                    "Pair-feature cache write failed after %d attempts path=%s error=%s",
                    PAIR_FEATURE_CACHE_WRITE_MAX_RETRIES,
                    db_path,
                    exc,
                )
                raise


NUM_FEATURES = FeaturizationInfo().number_of_features


def _single_pair_featurize(
    work_input: tuple[str, str],
    index: int = -1,
    *,
    dataset: ANDData | None = None,
    runtime_context: RuntimeContext | None = None,
) -> tuple[list[int | float], int]:
    """
    Creates the features array for a single signature pair
    The default worker path reads dataset/runtime context from globals so process pools do not pickle the dataset
    into every submitted chunk. Direct tests may pass them explicitly.

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of signature ids
    index: int
        the index of the pair in the list of all pairs,
        used to keep track of cached features

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    global global_dataset
    global global_runtime_context

    if dataset is None:
        dataset = global_dataset
    if dataset is None:
        raise RuntimeError("global_dataset is not initialized; call many_pairs_featurize first")

    features = []

    signature_1 = dataset.signatures[work_input[0]]
    signature_2 = dataset.signatures[work_input[1]]

    paper_id_1 = signature_1.paper_id
    paper_id_2 = signature_2.paper_id

    paper_1 = dataset.papers[str(paper_id_1)]
    paper_2 = dataset.papers[str(paper_id_2)]

    features.extend(
        [
            equal(
                signature_1.author_info_first_normalized_without_apostrophe,
                signature_2.author_info_first_normalized_without_apostrophe,
            ),
            counter_jaccard(
                Counter(
                    [
                        p[0]
                        for p in signature_1.author_info_middle_normalized_without_apostrophe.split(" ")
                        if len(p) > 0
                    ]
                ),
                Counter(
                    [
                        p[0]
                        for p in signature_2.author_info_middle_normalized_without_apostrophe.split(" ")
                        if len(p) > 0
                    ]
                ),
            ),
            equal_middle(
                signature_1.author_info_middle_normalized_without_apostrophe,
                signature_2.author_info_middle_normalized_without_apostrophe,
            ),
            (
                len(signature_1.author_info_middle_normalized_without_apostrophe) == 0
                and len(signature_2.author_info_middle_normalized_without_apostrophe) != 0
            )
            or (
                len(signature_2.author_info_middle_normalized_without_apostrophe) == 0
                and len(signature_1.author_info_middle_normalized_without_apostrophe) != 0
            ),
            len(signature_1.author_info_first_normalized_without_apostrophe) == 1
            or len(signature_2.author_info_first_normalized_without_apostrophe) == 1,
            any(len(middle) == 1 for middle in signature_1.author_info_middle_normalized_without_apostrophe.split(" "))
            or any(
                len(middle) == 1 for middle in signature_2.author_info_middle_normalized_without_apostrophe.split(" ")
            ),
        ]
    )

    features.append(
        counter_jaccard(
            signature_1.author_info_affiliations_n_grams,
            signature_2.author_info_affiliations_n_grams,
        )
    )

    email_prefix_1: str | None = None
    email_prefix_2: str | None = None
    email_suffix_1: str | None = None
    email_suffix_2: str | None = None
    if (
        signature_1.author_info_email is not None
        and len(signature_1.author_info_email) > 0
        and signature_2.author_info_email is not None
        and len(signature_2.author_info_email) > 0
    ):
        email_prefix_1, email_suffix_1 = email_prefix_suffix(signature_1.author_info_email)
        email_prefix_2, email_suffix_2 = email_prefix_suffix(signature_2.author_info_email)

    features.extend(
        [
            (
                email_prefix_1 == email_prefix_2
                if email_prefix_1 is not None and email_prefix_2 is not None
                else NUMPY_NAN
            ),
            (
                email_suffix_1 == email_suffix_2
                if email_suffix_1 is not None and email_suffix_2 is not None
                else NUMPY_NAN
            ),
        ]
    )

    features.extend(
        [
            jaccard(signature_1.author_info_coauthor_blocks, signature_2.author_info_coauthor_blocks),
            counter_jaccard(
                signature_1.author_info_coauthor_n_grams,
                signature_2.author_info_coauthor_n_grams,
                denominator_max=5000,
            ),
            jaccard(signature_1.author_info_coauthors, signature_2.author_info_coauthors),
        ]
    )

    features.append(counter_jaccard(paper_1.venue_ngrams, paper_2.venue_ngrams))

    features.append(
        np.minimum(
            diff(
                paper_1.year if paper_1.year is not None and paper_1.year > 0 else None,
                paper_2.year if paper_2.year is not None and paper_2.year > 0 else None,
            ),
            50,
        )
    )  # magic number!

    features.extend(
        [
            counter_jaccard(paper_1.title_ngrams_words, paper_2.title_ngrams_words),
            counter_jaccard(paper_1.title_ngrams_chars, paper_2.title_ngrams_chars),
        ]
    )

    english_or_unknown_count = int(paper_1.predicted_language in {"en", "un"}) + int(
        paper_2.predicted_language in {"en", "un"}
    )

    features.extend(
        [
            np.minimum(
                diff(
                    signature_1.author_info_position,
                    signature_2.author_info_position,
                ),
                50,
            ),
            int(paper_1.has_abstract) + int(paper_2.has_abstract),
            english_or_unknown_count,
            paper_1.predicted_language == paper_2.predicted_language,
            min(float(paper_1.language_reliability or 0.0), float(paper_2.language_reliability or 0.0)),
        ]
    )

    features.extend(
        name_counts(
            signature_1.author_info_name_counts,
            signature_2.author_info_name_counts,
        )
    )

    specter_1 = None
    specter_2 = None
    if english_or_unknown_count == 2 and dataset.specter_embeddings is not None:
        if str(paper_id_1) in dataset.specter_embeddings:
            specter_1 = dataset.specter_embeddings[str(paper_id_1)]
            if np.all(specter_1 == 0):
                specter_1 = None
        if str(paper_id_2) in dataset.specter_embeddings:
            specter_2 = dataset.specter_embeddings[str(paper_id_2)]
            if np.all(specter_2 == 0):
                specter_2 = None

    if specter_1 is not None and specter_2 is not None:
        specter_sim = cosine_sim(specter_1, specter_2) + 1
    else:
        specter_sim = NUMPY_NAN

    features.append(specter_sim)  # , abstract_count, english_count])

    features.append(counter_jaccard(paper_1.journal_ngrams, paper_2.journal_ngrams))

    features.extend(
        name_text_features(
            signature_1.author_info_first_normalized_without_apostrophe,
            signature_2.author_info_first_normalized_without_apostrophe,
        )
    )

    # unifying feature type in features array
    features = [float(val) if isinstance(val, np.floating | float) else int(val) for val in features]

    return features, index


def parallel_helper(piece_of_work: tuple, worker_func: Callable):
    """
    Helper function to explode tuple arguments

    Parameters
    ----------
    piece_of_work: Tuple
        the input for the worker func, in tuple form
    worker_func: Callable
        the function that will do the work

    Returns
    -------
    returns the result of calling the worker function
    """
    result = worker_func(*piece_of_work)
    return result


def _execute_python_featurization_phase(
    *,
    pieces_of_work: list[tuple[tuple[str, str], int]],
    n_jobs: int,
    chunk_size: int,
    use_cache: bool,
    signature_pairs: list[tuple[str, str, int | float]],
    featurizer_info: FeaturizationInfo,
    scatter_context: ScatterContext,
    cached_features: dict[str, Any],
) -> tuple[str, int]:
    new_features_count = 0
    if n_jobs > 1:
        backend_used = "python_parallel"
        if use_cache:
            logger.info("Cache changed, making %d feature vectors in parallel", len(pieces_of_work))
        else:
            logger.info("Making %d feature vectors in parallel", len(pieces_of_work))

        pool_size = n_jobs if len(pieces_of_work) > 1000 else 1
        # Explicit platform policy to avoid implicit UniversalPool defaults at call sites.
        use_threads = platform.system() in ("Windows", "Darwin")
        with UniversalPool(processes=pool_size, use_threads=use_threads) as p:
            work_count = len(pieces_of_work)
            with tqdm(total=work_count, desc="Doing work", disable=work_count <= 10000) as pbar:
                for feature_output, index in p.imap(
                    functools.partial(parallel_helper, worker_func=_single_pair_featurize),
                    pieces_of_work,
                    min(chunk_size, max(1, int((work_count / n_jobs) / 2))),
                ):
                    new_features_count += _write_feature_row(
                        feature_output=feature_output,
                        output_index=int(index),
                        signature_pairs=signature_pairs,
                        featurizer_info=featurizer_info,
                        cached_features=cached_features,
                        use_cache=use_cache,
                        scatter_context=scatter_context,
                        source_is_full=True,
                    )
                    pbar.update()
        return backend_used, new_features_count

    backend_used = "python_serial"
    if use_cache:
        logger.info("Cache changed, making %d feature vectors in serial", len(pieces_of_work))
    else:
        logger.info("Making %d feature vectors in serial", len(pieces_of_work))
    partial_func = functools.partial(parallel_helper, worker_func=_single_pair_featurize)
    for piece in tqdm(pieces_of_work, total=len(pieces_of_work), desc="Doing work"):
        result = partial_func(piece)
        new_features_count += _write_feature_row(
            feature_output=result[0],
            output_index=int(result[1]),
            signature_pairs=signature_pairs,
            featurizer_info=featurizer_info,
            cached_features=cached_features,
            use_cache=use_cache,
            scatter_context=scatter_context,
            source_is_full=True,
        )
    return backend_used, new_features_count


def _execute_rust_batch_featurization_phase(
    *,
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
    pieces_of_work: list[tuple[tuple[str, str], int]],
    featurizer_info: FeaturizationInfo,
    runtime_context: RuntimeContext,
    n_jobs: int,
    cache_policy: CachePolicy,
    total_ram_bytes: int | None,
    rust_batch_total_ram_for_stage: int | None,
    rust_batch_rss_before_bytes: int,
    rust_batch_rss_peak_bytes: int,
    rust_batch_rss_source: str,
    rust_batch_rss_baseline_locked: bool,
    indices_to_use: list[int],
    nameless_indices_to_use: list[int],
    indices_needed_for_compute: list[int],
    identity_selected_indices: bool,
    coauthor_similarity_index: int | None,
    features: np.ndarray,
    nameless_features: np.ndarray | None,
    coauthor_similarity_values: np.ndarray | None,
    cached_features: dict[str, Any],
) -> RustBatchExecutionResult:
    if len(pieces_of_work) <= 0:
        raise ValueError("Rust batch execution requires non-empty pieces_of_work")

    def _sample_rss_peak() -> None:
        nonlocal rust_batch_rss_peak_bytes
        if rust_batch_total_ram_for_stage is None:
            return
        rss_now, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
        if rss_now > rust_batch_rss_peak_bytes:
            rust_batch_rss_peak_bytes = rss_now

    class _RustBatchRssSampler:
        def __init__(self, interval_seconds: float):
            self.interval_seconds = interval_seconds
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None

        def _run(self) -> None:
            while not self._stop.is_set():
                _sample_rss_peak()
                self._stop.wait(self.interval_seconds)

        def __enter__(self):
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
            return False

    def _rust_batch_sampler_context():
        return contextlib.nullcontext()

    rust_featurizer = feature_port._get_rust_featurizer(
        dataset,
        runtime_context=runtime_context,
    )
    if not hasattr(rust_featurizer, "featurize_pairs_matrix_indexed") or not hasattr(rust_featurizer, "signature_ids"):
        raise RuntimeError(
            "Rust batch pair featurization requires featurize_pairs_matrix_indexed and signature_ids; "
            "rebuild/install a supported s2and-rust extension."
        )
    rust_selected_indices: list[int] | None = None
    if not cache_policy.requires_full_feature_rows and len(indices_needed_for_compute) > 0:
        rust_selected_indices = indices_needed_for_compute
    signature_id_to_index: dict[Any, int] = {}
    rust_signature_ids = rust_featurizer.signature_ids()
    for idx, sig_id in enumerate(rust_signature_ids):
        signature_id_to_index[sig_id] = int(idx)
        signature_id_to_index[str(sig_id)] = int(idx)
    logger.info("Rust indexed pair API enabled (signature_count=%d)", len(signature_id_to_index))
    rust_feature_count = NUM_FEATURES if rust_selected_indices is None else len(rust_selected_indices)
    rust_prediction_params = memory_budget.resolve_rust_batch_prediction_params()
    configured_fixed_overhead_bytes = int(rust_prediction_params["fixed_overhead_bytes"])
    calibrated_fixed_overhead_bytes: int | None = None
    if rust_batch_total_ram_for_stage is not None:
        calibrated_fixed_overhead_bytes = _maybe_calibrate_rust_batch_fixed_overhead_bytes(
            rust_featurizer=rust_featurizer,
            pieces_of_work=pieces_of_work,
            signature_id_to_index=signature_id_to_index,
            rust_selected_indices=rust_selected_indices,
            selected_feature_count=len(indices_to_use),
            nameless_feature_count=len(nameless_indices_to_use),
            row_overhead_bytes=int(rust_prediction_params["row_overhead_bytes"]),
            persistent_row_overhead_bytes=int(rust_prediction_params["persistent_row_overhead_bytes"]),
            configured_fixed_overhead_bytes=configured_fixed_overhead_bytes,
            num_threads=max(1, int(n_jobs)),
            total_ram_for_stage=rust_batch_total_ram_for_stage,
            run_id=runtime_context.run_id,
        )
    fixed_overhead_bytes_for_plan = configured_fixed_overhead_bytes
    if calibrated_fixed_overhead_bytes is not None:
        fixed_overhead_bytes_for_plan = max(
            configured_fixed_overhead_bytes,
            int(calibrated_fixed_overhead_bytes),
        )

    rust_batch_plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=rust_feature_count,
        total_pairs=len(pieces_of_work),
        total_rows=len(signature_pairs),
        selected_feature_count=len(indices_to_use),
        nameless_feature_count=len(nameless_indices_to_use),
        total_ram_bytes=(
            rust_batch_total_ram_for_stage if rust_batch_total_ram_for_stage is not None else total_ram_bytes
        ),
        base_chunk_pairs=int(rust_prediction_params["base_chunk_pairs"]),
        row_overhead_bytes=int(rust_prediction_params["row_overhead_bytes"]),
        persistent_row_overhead_bytes=int(rust_prediction_params["persistent_row_overhead_bytes"]),
        fixed_overhead_bytes=int(fixed_overhead_bytes_for_plan),
    )
    target_chunk_size = int(rust_batch_plan.chunk_pairs)
    total_ram_for_stage = int(rust_batch_plan.total_ram_bytes)
    predicted_stage_peak_delta_bytes = int(rust_batch_plan.predicted_stage_peak_delta_bytes)
    predicted_stage_peak_rss_bytes = int(rust_batch_plan.predicted_stage_peak_rss_bytes)
    if rust_batch_total_ram_for_stage != total_ram_for_stage:
        rust_batch_total_ram_for_stage = total_ram_for_stage
        if not rust_batch_rss_baseline_locked:
            rust_batch_rss_before_bytes, rust_batch_rss_source = memory_budget.current_rss_bytes_best_effort(
                total_ram_for_stage
            )
            rust_batch_rss_peak_bytes = rust_batch_rss_before_bytes
    _sample_rss_peak()
    logger.info(
        "Making %d feature vectors in Rust batch mode (target_chunk_size=%d "
        "base_chunk_pairs=%d bytes_per_pair_row=%d predicted_chunk_bytes=%d "
        "predicted_stage_peak_delta_bytes=%d predicted_stage_peak_rss_bytes=%d stage_budget_bytes=%d "
        "total_ram=%d total_ram_source=%s available=%d)",
        len(pieces_of_work),
        target_chunk_size,
        int(rust_batch_plan.base_chunk_pairs),
        int(rust_batch_plan.bytes_per_pair_row),
        int(rust_batch_plan.predicted_chunk_bytes),
        predicted_stage_peak_delta_bytes,
        predicted_stage_peak_rss_bytes,
        int(rust_batch_plan.stage_budget_bytes),
        int(rust_batch_plan.total_ram_bytes),
        str(rust_batch_plan.total_ram_source),
        int(rust_batch_plan.available_bytes),
    )

    selected_positions: list[int] = indices_to_use
    nameless_positions: list[int] = nameless_indices_to_use
    coauthor_position: int | None = coauthor_similarity_index
    if rust_selected_indices is not None:
        pos_by_feature_idx = {int(feature_idx): int(pos) for pos, feature_idx in enumerate(rust_selected_indices)}
        selected_positions = [pos_by_feature_idx[idx] for idx in indices_to_use]
        nameless_positions = [pos_by_feature_idx[idx] for idx in nameless_indices_to_use]
        if coauthor_similarity_index is not None:
            coauthor_position = pos_by_feature_idx[coauthor_similarity_index]

    rust_scatter_context = ScatterContext(
        features=features,
        nameless_features=nameless_features,
        coauthor_similarity_values=coauthor_similarity_values,
        identity_selected_indices=identity_selected_indices,
        indices_to_use=indices_to_use,
        nameless_indices_to_use=nameless_indices_to_use,
        selected_positions=selected_positions,
        nameless_positions=nameless_positions,
        coauthor_similarity_index=coauthor_similarity_index,
        coauthor_position=coauthor_position,
    )

    num_threads = max(1, int(n_jobs))
    rust_batch_adaptive_halvings = 0
    new_features_count = 0
    with _rust_batch_sampler_context():
        with tqdm(
            total=len(pieces_of_work),
            desc="Rust batch featurization",
            disable=len(pieces_of_work) <= 10000,
        ) as pbar:
            start_index = 0
            while start_index < len(pieces_of_work):
                chunk_work = pieces_of_work[start_index : start_index + target_chunk_size]
                rust_pairs_chunk = [pair for pair, _ in chunk_work]
                rust_pairs_chunk_indexed = [
                    (
                        _signature_id_to_index_or_raise(signature_id_to_index, pair[0]),
                        _signature_id_to_index_or_raise(signature_id_to_index, pair[1]),
                    )
                    for pair in rust_pairs_chunk
                ]
                rust_features_chunk = np.asarray(
                    rust_featurizer.featurize_pairs_matrix_indexed(
                        rust_pairs_chunk_indexed,
                        rust_selected_indices,
                        num_threads,
                        np.nan,
                    ),
                    dtype=np.float64,
                )

                if rust_features_chunk.shape[0] != len(chunk_work):
                    raise RuntimeError(
                        "Rust batch featurizer returned mismatched feature count: "
                        f"expected={len(chunk_work)} got={rust_features_chunk.shape[0]}"
                    )
                rust_chunk_columns = int(rust_features_chunk.shape[1])
                selected_column_count = (
                    len(rust_selected_indices) if rust_selected_indices is not None else NUM_FEATURES
                )
                if rust_selected_indices is None and rust_chunk_columns != NUM_FEATURES:
                    raise RuntimeError(
                        "Rust batch featurizer returned unexpected feature width: "
                        f"expected={NUM_FEATURES} got={rust_chunk_columns}"
                    )
                if rust_selected_indices is not None and rust_chunk_columns not in {
                    NUM_FEATURES,
                    selected_column_count,
                }:
                    raise RuntimeError(
                        "Rust batch featurizer returned unexpected feature width: "
                        f"expected={selected_column_count} (selected) or {NUM_FEATURES} (full) "
                        f"got={rust_chunk_columns}"
                    )
                rust_chunk_is_full = rust_chunk_columns == NUM_FEATURES
                chunk_indices = [index for _, index in chunk_work]

                if cache_policy.pair_feature_cache_enabled:
                    for row_offset, index in enumerate(chunk_indices):
                        new_features_count += _write_feature_row(
                            feature_output=rust_features_chunk[row_offset],
                            output_index=int(index),
                            signature_pairs=signature_pairs,
                            featurizer_info=featurizer_info,
                            cached_features=cached_features,
                            use_cache=cache_policy.pair_feature_cache_enabled,
                            scatter_context=rust_scatter_context,
                            source_is_full=rust_chunk_is_full,
                        )
                else:
                    _scatter_chunk_to_output(
                        rust_features_chunk=rust_features_chunk,
                        chunk_indices=chunk_indices,
                        scatter_context=rust_scatter_context,
                        rust_chunk_is_full=rust_chunk_is_full,
                    )
                _sample_rss_peak()
                if (
                    rust_batch_total_ram_for_stage is not None
                    and rust_batch_adaptive_halvings < 3
                    and predicted_stage_peak_delta_bytes > 0
                ):
                    observed_delta = max(0, rust_batch_rss_peak_bytes - rust_batch_rss_before_bytes)
                    if observed_delta > predicted_stage_peak_delta_bytes * 1.2:
                        target_chunk_size = max(1, target_chunk_size // 2)
                        rust_batch_adaptive_halvings += 1
                        logger.warning(
                            "Rust batch adaptive chunking: observed_delta=%d > predicted_delta=%d * 1.2; "
                            "halving target_chunk_size to %d (halving %d/3) run_id=%s",
                            observed_delta,
                            predicted_stage_peak_delta_bytes,
                            target_chunk_size,
                            rust_batch_adaptive_halvings,
                            runtime_context.run_id,
                        )
                pbar.update(len(chunk_work))
                start_index += len(chunk_work)

    _sample_rss_peak()
    return RustBatchExecutionResult(
        rust_batch_plan=rust_batch_plan,
        new_features_count=int(new_features_count),
        rust_batch_total_ram_for_stage=rust_batch_total_ram_for_stage,
        rust_batch_rss_before_bytes=int(rust_batch_rss_before_bytes),
        rust_batch_rss_peak_bytes=int(rust_batch_rss_peak_bytes),
        rust_batch_rss_source=str(rust_batch_rss_source),
        rust_batch_adaptive_halvings=int(rust_batch_adaptive_halvings),
    )


def many_pairs_featurize(
    signature_pairs: list[tuple[str, str, int | float]],
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int,
    use_cache: bool,
    chunk_size: int,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Featurizes many pairs

    Parameters
    ----------
    signature_pairs: List[pairs]
        the pairs to featurize
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training rows
    total_ram_bytes: Optional[int]
        Optional explicit RAM input used for stage-wise memory budgeting of Rust batch featurization.

    Returns
    -------
    np.ndarray: the main features for all the pairs
    np.ndarray: the labels for all the pairs
    np.ndarray: the nameless features for all the pairs
    """
    featurize_start = time.perf_counter()
    backend_used = "cached_only"
    if runtime_context is None:
        runtime_context = dataset.runtime_context
    cache_policy = resolve_cache_policy(use_cache)
    signature_pairs = [(str(pair[0]), str(pair[1]), pair[2]) for pair in signature_pairs]
    _ensure_python_pair_signature_ngrams(dataset, signature_pairs, runtime_context)

    global global_dataset
    global global_runtime_context
    global_dataset = dataset
    global_runtime_context = runtime_context

    cached_features: dict[str, Any] = {"features": {}}
    cache_changed = False
    new_features_count = 0
    did_rust_batch = False
    rust_batch_plan: memory_budget.RustBatchChunkPlan | None = None
    rust_batch_total_ram_for_stage: int | None = None
    rust_batch_rss_before_bytes = 0
    rust_batch_rss_peak_bytes = 0
    rust_batch_rss_source = "unavailable"
    rust_batch_rss_baseline_locked = False
    rust_batch_adaptive_halvings = 0

    rust_module_available = False
    if _use_rust_featurizer(runtime_context, dataset):
        try:
            # Prewarm so the Arrow featurizer build doesn't land inside the RSS measurement window.
            feature_port._get_rust_featurizer(
                dataset,
                runtime_context=runtime_context,
            )
            rust_module_available = True
        except Exception as exc:
            raise RuntimeError("Rust featurizer init failed " f"(run_id={runtime_context.run_id} error={exc})") from exc
        try:
            rust_batch_total_ram_for_stage, _ = memory_budget.resolve_total_ram_bytes(total_ram_bytes)
            rust_batch_rss_before_bytes, rust_batch_rss_source = memory_budget.current_rss_bytes_best_effort(
                rust_batch_total_ram_for_stage
            )
            rust_batch_rss_peak_bytes = rust_batch_rss_before_bytes
            rust_batch_rss_baseline_locked = True
        except RuntimeError:
            # Preserve behavior for all-cached paths when RAM autodetection is unavailable.
            rust_batch_total_ram_for_stage = None

    def _sample_rust_batch_rss_peak() -> None:
        nonlocal rust_batch_rss_peak_bytes
        if rust_batch_total_ram_for_stage is None:
            return
        rss_now, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
        if rss_now > rust_batch_rss_peak_bytes:
            rust_batch_rss_peak_bytes = rss_now

    if cache_policy.pair_feature_cache_enabled:
        logger.info("Loading cache...")
        if not os.path.exists(featurizer_info.cache_directory(dataset.name)):
            os.makedirs(featurizer_info.cache_directory(dataset.name))
        cache_storage_key = featurizer_info.cache_storage_key(dataset.name)
        if os.path.exists(featurizer_info.cache_db_path(dataset.name)):
            with _CACHED_FEATURES_LOCK:
                in_memory = CACHED_FEATURES.get(cache_storage_key)
            if in_memory is not None:
                cached_features = in_memory
            else:
                cached_features = featurizer_info.load_cache(dataset.name)
                logger.info("Cache loaded with %d keys", len(cached_features["features"]))
        else:
            logger.info("Cache initiated.")
            cached_features = featurizer_info._fresh_cache_payload()

        # Initialize buffer for new features if not already present
        if "__new_features__" not in cached_features:
            cached_features["__new_features__"] = {}

    indices_to_use_set: set[int] = set()
    for feature_name in featurizer_info.features_to_use:
        indices_to_use_set.update(featurizer_info.feature_group_to_index[feature_name])
    indices_to_use: list[int] = sorted(indices_to_use_set)

    nameless_indices_to_use: list[int] = []
    if nameless_featurizer_info is not None:
        nameless_indices_to_use_set: set[int] = set()
        for feature_name in nameless_featurizer_info.features_to_use:
            nameless_indices_to_use_set.update(nameless_featurizer_info.feature_group_to_index[feature_name])
        nameless_indices_to_use = sorted(nameless_indices_to_use_set)

    identity_selected_indices = indices_to_use == list(range(NUM_FEATURES))
    coauthor_similarity_index: int | None = None
    coauthor_similarity_values: np.ndarray | None = None
    if delete_training_data:
        coauthor_similarity_index = featurizer_info.feature_group_to_index["coauthor_similarity"][1]
        coauthor_similarity_values = np.full(len(signature_pairs), -float(LARGE_INTEGER), dtype=np.float64)

    indices_needed_for_compute: list[int] = sorted(
        set(indices_to_use)
        | set(nameless_indices_to_use)
        | ({coauthor_similarity_index} if coauthor_similarity_index is not None else set())
    )

    features = np.full(
        (len(signature_pairs), len(indices_to_use)),
        -float(LARGE_INTEGER),
        dtype=np.float64,
    )
    labels = np.zeros(len(signature_pairs))
    nameless_features: np.ndarray | None = None
    if nameless_featurizer_info is not None:
        nameless_features = np.full(
            (len(signature_pairs), len(nameless_indices_to_use)),
            -float(LARGE_INTEGER),
            dtype=np.float64,
        )
    default_scatter_context = ScatterContext(
        features=features,
        nameless_features=nameless_features,
        coauthor_similarity_values=coauthor_similarity_values,
        identity_selected_indices=identity_selected_indices,
        indices_to_use=indices_to_use,
        nameless_indices_to_use=nameless_indices_to_use,
        selected_positions=indices_to_use,
        nameless_positions=nameless_indices_to_use,
        coauthor_similarity_index=coauthor_similarity_index,
        coauthor_position=coauthor_similarity_index,
    )
    _sample_rust_batch_rss_peak()
    pieces_of_work = []
    logger.info("Creating %d pieces of work", len(signature_pairs))
    for i, pair in tqdm(enumerate(signature_pairs), desc="Creating work", disable=len(signature_pairs) <= 100000):
        labels[i] = pair[2]

        # negative labels are an indication of partial supervision
        if pair[2] < 0:
            continue

        if cache_policy.pair_feature_cache_enabled:
            cached_vector = None
            for cache_key in featurizer_info.feature_cache_lookup_keys((pair[0], pair[1])):
                cached_vector = cached_features["features"].get(cache_key)
                if cached_vector is not None:
                    break
            if cached_vector is not None:
                _scatter_feature_row_from_source(
                    feature_output=np.asarray(cached_vector, dtype=np.float64),
                    output_index=i,
                    scatter_context=default_scatter_context,
                    rust_chunk_is_full=True,
                )
                continue

        cache_changed = True
        pieces_of_work.append(((pair[0], pair[1]), i))

    logger.info("Created pieces of work")

    if cache_changed:
        use_rust = _use_rust_featurizer(runtime_context, dataset)
        if use_rust and not rust_module_available:
            raise RuntimeError(
                "Rust backend requested for pair_featurization but s2and_rust extension is unavailable "
                f"(run_id={runtime_context.run_id})"
            )

        _log_featurization_backend_decision(
            runtime_context=runtime_context,
            pieces_of_work_count=len(pieces_of_work),
            n_jobs=n_jobs,
            use_rust_featurizer=use_rust,
            rust_module_available=rust_module_available,
        )

        if use_rust and rust_module_available and len(pieces_of_work) > 0:
            try:
                rust_batch_result = _execute_rust_batch_featurization_phase(
                    dataset=dataset,
                    signature_pairs=signature_pairs,
                    pieces_of_work=pieces_of_work,
                    featurizer_info=featurizer_info,
                    runtime_context=runtime_context,
                    n_jobs=n_jobs,
                    cache_policy=cache_policy,
                    total_ram_bytes=total_ram_bytes,
                    rust_batch_total_ram_for_stage=rust_batch_total_ram_for_stage,
                    rust_batch_rss_before_bytes=rust_batch_rss_before_bytes,
                    rust_batch_rss_peak_bytes=rust_batch_rss_peak_bytes,
                    rust_batch_rss_source=rust_batch_rss_source,
                    rust_batch_rss_baseline_locked=rust_batch_rss_baseline_locked,
                    indices_to_use=indices_to_use,
                    nameless_indices_to_use=nameless_indices_to_use,
                    indices_needed_for_compute=indices_needed_for_compute,
                    identity_selected_indices=identity_selected_indices,
                    coauthor_similarity_index=coauthor_similarity_index,
                    features=features,
                    nameless_features=nameless_features,
                    coauthor_similarity_values=coauthor_similarity_values,
                    cached_features=cached_features,
                )
                rust_batch_plan = rust_batch_result.rust_batch_plan
                rust_batch_total_ram_for_stage = rust_batch_result.rust_batch_total_ram_for_stage
                rust_batch_rss_before_bytes = rust_batch_result.rust_batch_rss_before_bytes
                rust_batch_rss_peak_bytes = rust_batch_result.rust_batch_rss_peak_bytes
                rust_batch_rss_source = rust_batch_result.rust_batch_rss_source
                rust_batch_adaptive_halvings = rust_batch_result.rust_batch_adaptive_halvings
                did_rust_batch = True
                backend_used = "rust_batch"
                new_features_count += int(rust_batch_result.new_features_count)
            except Exception as exc:
                raise RuntimeError(
                    "Rust batch featurization failed in strict rust backend "
                    f"(pairs={len(pieces_of_work)} run_id={runtime_context.run_id} "
                    f"failure_reason={exc})"
                ) from exc

        if use_rust and not did_rust_batch and len(pieces_of_work) > 0:
            raise RuntimeError(
                "Rust pair_featurization stage was selected but Rust batch execution did not complete "
                f"(run_id={runtime_context.run_id})"
            )

        if not did_rust_batch:
            backend_used, python_new_features = _execute_python_featurization_phase(
                pieces_of_work=pieces_of_work,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
                use_cache=cache_policy.pair_feature_cache_enabled,
                signature_pairs=signature_pairs,
                featurizer_info=featurizer_info,
                scatter_context=default_scatter_context,
                cached_features=cached_features,
            )
            new_features_count += int(python_new_features)
        _sample_rust_batch_rss_peak()
        logger.info("Work completed")
    else:
        logger.info("Featurization backend decision: skipped compute (all pairs were cached or pre-labeled)")

    if cache_policy.pair_feature_cache_enabled and cache_changed:
        # Only do incremental writes if we have enough new features to justify the overhead
        new_features_in_buffer = len(cached_features.get("__new_features__", {}))

        if new_features_in_buffer >= INCREMENTAL_WRITE_THRESHOLD:
            logger.info(
                "Collected %d new features in this run; writing incrementally during final cache flush",
                new_features_in_buffer,
            )
        else:
            logger.info("Only %d new features - will write at end", new_features_in_buffer)
    _sample_rust_batch_rss_peak()

    if (
        cache_policy.pair_feature_cache_enabled
        and cache_changed
        and len(cached_features.get("__new_features__", {})) > 0
    ):
        # Always write any remaining new features at the end.
        # Have to do this before subselecting features.
        new_features_in_buffer = len(cached_features["__new_features__"])
        logger.info("Writing final %d new features to cache", new_features_in_buffer)
        featurizer_info.write_cache(cached_features, dataset.name, incremental=True)
        logger.info("Cache written with %d total keys.", len(cached_features["features"]))
    _sample_rust_batch_rss_peak()

    if cache_policy.pair_feature_cache_enabled:
        logger.info("Writing to in memory cache")
        # use the variable from above, to be sure we are using the same path
        cache_path = featurizer_info.cache_storage_key(dataset.name)
        with _CACHED_FEATURES_LOCK:
            CACHED_FEATURES[cache_path] = cached_features
        logger.info("In memory cache written")
    _sample_rust_batch_rss_peak()

    if delete_training_data:
        logger.info("Deleting some training rows")
        negative_label_indices = labels == 0
        if coauthor_similarity_values is None:
            raise RuntimeError("delete_training_data requires coauthor_similarity_values to be computed")
        high_coauthor_sim_indices = coauthor_similarity_values > 0.95
        indices_to_remove = negative_label_indices & high_coauthor_sim_indices
        logger.info("Intending to remove %d rows", int(sum(indices_to_remove)))
        original_size = len(labels)
        features = features[~indices_to_remove, :]
        if nameless_features is not None:
            nameless_features = nameless_features[~indices_to_remove, :]
        labels = labels[~indices_to_remove]
        logger.info(
            "Removed %d rows and %d labels",
            int(original_size - features.shape[0]),
            int(original_size - len(labels)),
        )
    _sample_rust_batch_rss_peak()

    logger.info("Making numpy arrays for features and labels")
    if nameless_features is not None:
        nameless_features[np.isnan(nameless_features)] = nan_value
        _sample_rust_batch_rss_peak()
    features[np.isnan(features)] = nan_value
    _sample_rust_batch_rss_peak()

    if did_rust_batch and rust_batch_plan is not None:
        _sample_rust_batch_rss_peak()
        rss_after_bytes = rust_batch_rss_before_bytes
        if rust_batch_total_ram_for_stage is not None:
            rss_after_bytes, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
            _sample_rust_batch_rss_peak()
        rust_batch_prediction = memory_budget.summarize_prediction_accuracy(
            stage_name="pair_featurization_rust_batch",
            predicted_peak_delta_bytes=int(rust_batch_plan.predicted_stage_peak_delta_bytes),
            rss_before_bytes=rust_batch_rss_before_bytes,
            rss_peak_bytes=rust_batch_rss_peak_bytes,
            rss_after_bytes=rss_after_bytes,
        )
        logger.info(
            "Telemetry: pair_featurization_memory stage=%s prediction_contract_version=%s "
            "predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d predicted_bytes=%d "
            "total_rows=%d selected_feature_count=%d nameless_feature_count=%d "
            "predicted_features_matrix_bytes=%d predicted_labels_bytes=%d predicted_chunk_bytes=%d "
            "predicted_persistent_row_overhead_bytes=%d predicted_fixed_overhead_bytes=%d "
            "rss_before_bytes=%d rss_peak_bytes=%d rss_after_bytes=%d observed_peak_delta_bytes=%d "
            "prediction_error_ratio=%.3f underpredicted=%s adaptive_halvings=%d rss_source=%s",
            rust_batch_prediction.stage_name,
            str(rust_batch_prediction.prediction_contract_version),
            int(rust_batch_prediction.predicted_peak_delta_bytes),
            int(rust_batch_prediction.predicted_peak_rss_bytes),
            int(rust_batch_prediction.predicted_bytes),
            int(rust_batch_plan.total_rows),
            int(rust_batch_plan.selected_feature_count),
            int(rust_batch_plan.nameless_feature_count),
            int(rust_batch_plan.predicted_features_matrix_bytes),
            int(rust_batch_plan.predicted_labels_bytes),
            int(rust_batch_plan.predicted_chunk_bytes),
            int(rust_batch_plan.predicted_persistent_row_overhead_bytes),
            int(rust_batch_plan.predicted_fixed_overhead_bytes),
            int(rust_batch_prediction.rss_before_bytes),
            int(rust_batch_prediction.rss_peak_bytes),
            int(rust_batch_prediction.rss_after_bytes),
            int(rust_batch_prediction.observed_peak_delta_bytes),
            float(rust_batch_prediction.prediction_error_ratio),
            bool(rust_batch_prediction.underpredicted),
            rust_batch_adaptive_halvings,
            rust_batch_rss_source,
        )
        memory_budget.emit_memory_telemetry(
            {
                "stage": rust_batch_prediction.stage_name,
                "prediction_contract_version": rust_batch_prediction.prediction_contract_version,
                "predicted_peak_delta_bytes": rust_batch_prediction.predicted_peak_delta_bytes,
                "predicted_peak_rss_bytes": rust_batch_prediction.predicted_peak_rss_bytes,
                "predicted_bytes": rust_batch_prediction.predicted_bytes,
                "total_rows": rust_batch_plan.total_rows,
                "selected_feature_count": rust_batch_plan.selected_feature_count,
                "nameless_feature_count": rust_batch_plan.nameless_feature_count,
                "predicted_features_matrix_bytes": rust_batch_plan.predicted_features_matrix_bytes,
                "predicted_labels_bytes": rust_batch_plan.predicted_labels_bytes,
                "predicted_chunk_bytes": rust_batch_plan.predicted_chunk_bytes,
                "predicted_persistent_row_overhead_bytes": rust_batch_plan.predicted_persistent_row_overhead_bytes,
                "predicted_fixed_overhead_bytes": rust_batch_plan.predicted_fixed_overhead_bytes,
                "rss_before_bytes": rust_batch_prediction.rss_before_bytes,
                "rss_peak_bytes": rust_batch_prediction.rss_peak_bytes,
                "rss_after_bytes": rust_batch_prediction.rss_after_bytes,
                "observed_peak_delta_bytes": rust_batch_prediction.observed_peak_delta_bytes,
                "prediction_error_ratio": rust_batch_prediction.prediction_error_ratio,
                "underpredicted": rust_batch_prediction.underpredicted,
                "adaptive_halvings": rust_batch_adaptive_halvings,
                "rss_source": rust_batch_rss_source,
            }
        )

    logger.info(
        "Telemetry stage: stage=pair_featurization seconds=%.3f total_pairs=%d uncached_pairs=%d backend=%s",
        time.perf_counter() - featurize_start,
        len(signature_pairs),
        len(pieces_of_work),
        backend_used,
    )
    logger.info("Numpy arrays made")
    return features, labels, nameless_features


def featurize(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int = 1,
    use_cache: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
    total_ram_bytes: int | None = None,
) -> tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays] | TupleOfArrays:
    """
    Featurizes the input dataset

    Parameters
    ----------
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training examples
    total_ram_bytes: Optional[int]
        Optional explicit RAM input used for stage-wise memory budgeting.

    Returns
    -------
    train/val/test features and labels if mode is 'train',
    features and labels for all pairs if mode is 'inference'
    """
    if dataset.mode == "inference":
        logger.info("featurizing all pairs")
        all_pairs = dataset.all_pairs()
        all_features = many_pairs_featurize(
            all_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("featurized all pairs")
        return all_features
    else:
        if dataset.train_pairs is None:
            if dataset.train_blocks is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures_fixed()
            elif dataset.train_signatures is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_data_signatures_fixed()
            else:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures()

            train_pairs, val_pairs, test_pairs = dataset.split_pairs(train_signatures, val_signatures, test_signatures)

        else:
            train_pairs, val_pairs, test_pairs = dataset.fixed_pairs()

        logger.info("featurizing train")
        train_features = many_pairs_featurize(
            train_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            delete_training_data,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("featurized train, featurizing val")
        val_features = many_pairs_featurize(
            val_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("featurized val, featurizing test")
        test_features = many_pairs_featurize(
            test_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("featurized test")
        return train_features, val_features, test_features
