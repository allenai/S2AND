import logging
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from s2and.arrow_inputs import (
    ValidatedArrowInputs,
    require_normalization_version,
)
from s2and.consts import CLUSTER_SEEDS_LOOKUP
from s2and.data import ANDData
from s2and.name_count_binding import NameCountsBinding
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.runtime import load_s2and_rust_extension
from s2and.rust_calls import (
    build_block_upper_triangle_feature_matrix_indexed_rust,
    build_linker_pair_aggregate_stats_arrays_rust,
    build_linker_pair_distance_accumulators_rust,
    build_linker_pair_features_and_aggregate_stats_arrays_rust,
    get_constraint_labels_index_arrays_rust,
    get_constraints_block_upper_triangle_indexed_rust,
    get_constraints_matrix_indexed_rust,
)
from s2and.thread_config import resolve_n_jobs

# Treat extension as Any for typing; it is optional and loaded on first use.
s2and_rust: Any | None = None

logger = logging.getLogger("s2and")
_S2AND_RUST_LOAD_LOCK = threading.Lock()
_SourcePathFingerprint = tuple[Any, ...]


class _CacheEntry:
    """The one cached Rust featurizer for a live dataset."""

    __slots__ = ("cache_key", "featurizer", "build_count")

    def __init__(
        self,
        cache_key: "_RustFeaturizerCacheKey",
        featurizer: Any,
        build_count: int,
    ):
        self.cache_key = cache_key
        self.featurizer = featurizer
        self.build_count = build_count


class _RustFeaturizerCacheState:
    """Lock and optional cached value owned by one live dataset."""

    __slots__ = ("lock", "entry")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.entry: _CacheEntry | None = None


@dataclass(frozen=True)
class _RustFeaturizerBuildKey:
    """Inputs consumed when constructing an Arrow-backed Rust featurizer."""

    preprocess: bool
    use_orcid_id: bool
    n_jobs: int
    source_paths: tuple[_SourcePathFingerprint, ...]
    name_tuples: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class _RustFeaturizerCacheKey:
    build: _RustFeaturizerBuildKey
    cluster_seeds_version: int


# Cache state has one authority per dataset: one lock and at most one entry.
_RUST_FEATURIZER_STATES: "weakref.WeakKeyDictionary[ANDData, _RustFeaturizerCacheState]" = weakref.WeakKeyDictionary()
_RUST_FEATURIZER_STATES_LOCK = threading.Lock()


def _require_rust_runtime() -> Any:
    return _ensure_s2and_rust_loaded()


def _ensure_s2and_rust_loaded() -> Any:
    global s2and_rust
    if s2and_rust is not None:
        return s2and_rust
    with _S2AND_RUST_LOAD_LOCK:
        if s2and_rust is None:
            s2and_rust = load_s2and_rust_extension()
    return s2and_rust


def _dataset_name_for_logs(dataset: Any) -> str:
    name = getattr(dataset, "name", None)
    return str(name) if name is not None else f"<unnamed:{id(dataset)}>"


def _dataset_mode_for_logs(dataset: Any) -> str:
    mode = str(getattr(dataset, "mode", "")).strip()
    return mode if mode else "unknown"


def _runtime_callsite_for_logs(dataset: Any, runtime_context: Any | None = None) -> tuple[str, str]:
    context = runtime_context if runtime_context is not None else getattr(dataset, "runtime_context", None)
    operation = str(getattr(context, "operation", "unknown"))
    run_id_raw = getattr(context, "run_id", None)
    run_id = str(run_id_raw) if run_id_raw is not None else f"dataset-{id(dataset)}"
    return operation, run_id


def _rust_featurizer_source_paths(dataset: Any) -> tuple[_SourcePathFingerprint, ...]:
    arrow_paths = getattr(dataset, "arrow_paths", None)
    if not isinstance(arrow_paths, Mapping) or not arrow_paths:
        raise RuntimeError("Rust training featurization requires dataset.arrow_paths")
    generation = getattr(dataset, "arrow_artifact_generation", None)
    if not isinstance(generation, str) or not generation:
        raise ValueError("dataset.arrow_paths requires a retained arrow_artifact_generation")
    exact_paths = tuple(sorted((str(key), str(value)) for key, value in arrow_paths.items()))
    return (("validated_arrow_generation", generation, exact_paths),)


def _rust_featurizer_build_key(dataset: Any) -> _RustFeaturizerBuildKey:
    name_tuples = getattr(dataset, "name_tuples", None)
    return _RustFeaturizerBuildKey(
        preprocess=bool(getattr(dataset, "preprocess", False)),
        use_orcid_id=bool(getattr(dataset, "use_orcid_id", True)),
        n_jobs=resolve_n_jobs(getattr(dataset, "n_jobs", 1)),
        source_paths=_rust_featurizer_source_paths(dataset),
        name_tuples=frozenset(name_tuples or ()),
    )


def _rust_featurizer_cache_key(
    dataset: Any,
    *,
    cluster_seeds_version: int | None = None,
) -> _RustFeaturizerCacheKey:
    resolved_seed_version = (
        _cluster_seeds_version_for_cache(dataset) if cluster_seeds_version is None else int(cluster_seeds_version)
    )
    return _RustFeaturizerCacheKey(
        build=_rust_featurizer_build_key(dataset),
        cluster_seeds_version=resolved_seed_version,
    )


def _rust_featurizer_state(dataset: ANDData) -> _RustFeaturizerCacheState:
    with _RUST_FEATURIZER_STATES_LOCK:
        state = _RUST_FEATURIZER_STATES.get(dataset)
        if state is None:
            state = _RustFeaturizerCacheState()
            _RUST_FEATURIZER_STATES[dataset] = state
        return state


def _cluster_seeds_version_for_cache(dataset: Any) -> int:
    return int(getattr(dataset, "_cluster_seeds_version", 0))


def _get_cached_rust_featurizer_for_cluster_seed_update(
    dataset: ANDData,
    *,
    runtime_context: Any | None = None,
) -> Any:
    """Return a cache-family featurizer suitable for an in-place seed update."""

    operation, run_id = _runtime_callsite_for_logs(dataset, runtime_context)
    dataset_mode = _dataset_mode_for_logs(dataset)
    current_seed_version = _cluster_seeds_version_for_cache(dataset)
    current_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=current_seed_version)
    entry = _rust_featurizer_state(dataset).entry
    if entry is not None and entry.cache_key.build == current_cache_key.build:
        if entry.cache_key.cluster_seeds_version != current_seed_version:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=seed_update_reuse dataset=%s mode=%s op=%s run=%s "
                "stale_seed_version=%d current_seed_version=%d",
                _dataset_name_for_logs(dataset),
                dataset_mode,
                operation,
                run_id,
                entry.cache_key.cluster_seeds_version,
                current_seed_version,
            )
        return entry.featurizer
    return _get_rust_featurizer(dataset, runtime_context=runtime_context)


def update_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: Any | None = None,
    *,
    bump_version: bool = False,
) -> None:
    """Sync current Python cluster seeds into the cached Rust featurizer."""

    state = _rust_featurizer_state(dataset)
    with state.lock:
        featurizer = _get_cached_rust_featurizer_for_cluster_seed_update(
            dataset,
            runtime_context=runtime_context,
        )
        current_seed_version = _cluster_seeds_version_for_cache(dataset)
        target_seed_version = current_seed_version + 1 if bump_version else current_seed_version
        target_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=target_seed_version)
        entry = state.entry
        if entry is None or entry.featurizer is not featurizer:
            raise RuntimeError(
                "Rust featurizer cache changed while its per-dataset lock was held "
                f"(dataset={_dataset_name_for_logs(dataset)})"
            )
        featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
        entry.cache_key = target_cache_key
        if bump_version:
            dataset._cluster_seeds_version = target_seed_version


def build_rust_featurizer_from_arrow_paths(
    paths: ValidatedArrowInputs,
    *,
    expected_normalization_version: str,
    signature_ids: Sequence[Any] | None = None,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    load_name_counts: bool = False,
    preprocess: bool = True,
    cluster_seed_require_value: float = 0.0,
    cluster_seed_disallow_value: float = 10000.0,
    num_threads: int | None = None,
    name_counts_index: Any | None = None,
    use_orcid_id: bool = True,
) -> Any:
    """Build a Rust featurizer directly from Arrow IPC FeatureBlock paths.

    ``name_counts_index`` is an already validated native snapshot. When it is
    omitted, the handle retained by Arrow validation is reused automatically.
    ``use_orcid_id=False`` suppresses ORCID while native signature records are
    constructed, without rewriting the immutable Arrow generation.
    """

    method = _require_rust_runtime().RustFeaturizer.from_arrow_paths
    expected_version = require_normalization_version(
        expected_normalization_version,
        context="RustFeaturizer.from_arrow_paths",
    )
    if not isinstance(paths, ValidatedArrowInputs):
        raise TypeError("build_rust_featurizer_from_arrow_paths requires ValidatedArrowInputs")
    if paths.normalization_version != expected_version:
        raise ValueError(
            "RustFeaturizer.from_arrow_paths normalization_version mismatch: "
            f"artifact={paths.normalization_version!r} expected={expected_version!r}"
        )
    normalized_paths = dict(paths)
    normalized_paths.pop("query_signatures", None)
    normalized_paths.pop("manifest", None)
    if not load_name_counts:
        if name_counts_index is not None:
            raise ValueError("name_counts_index requires load_name_counts=True")
        normalized_paths.pop("name_counts_index", None)
    elif name_counts_index is None:
        name_counts_index = paths._retained_native_name_counts_index()  # noqa: SLF001
    if load_name_counts and name_counts_index is not None:
        manifest = paths.name_counts_manifest
        if manifest is None:
            raise RuntimeError("validated Arrow inputs lost the retained name-count manifest")
        observed_manifest_sha256 = getattr(name_counts_index, "manifest_sha256", None)
        if observed_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError(
                "shared name-count handle manifest mismatch: "
                f"handle={observed_manifest_sha256!r} manifest={manifest.manifest_sha256!r}"
            )
        observed_normalization_version = getattr(name_counts_index, "normalization_version", None)
        if observed_normalization_version != manifest.normalization_version:
            raise ValueError(
                "shared name-count handle normalization mismatch: "
                f"handle={observed_normalization_version!r} manifest={manifest.normalization_version!r}"
            )
        expected_binding = NameCountsBinding.from_provenance(
            manifest.source_provenance,
            context="Arrow name-count manifest source_provenance",
        )
        observed_binding = NameCountsBinding.from_rust_featurizer(
            name_counts_index,
            context="shared name-count handle",
        )
        expected_binding.require_matches(
            observed_binding,
            context="RustFeaturizer.from_arrow_paths",
            source="shared NameCountsIndex handle",
        )
    resolved_name_tuples = name_tuples
    if name_tuples is None:
        # Package data is immutable for the process lifetime. Reuse only the
        # frozen validated artifact so repeated Arrow requests do not rehash it.
        resolved_name_tuples = load_packaged_name_tuple_artifact().pairs
    args = (
        normalized_paths,
        None if signature_ids is None else [str(value) for value in signature_ids],
        resolved_name_tuples,
        bool(preprocess),
        float(cluster_seed_require_value),
        float(cluster_seed_disallow_value),
        None if num_threads is None else resolve_n_jobs(num_threads),
    )
    if name_counts_index is None and use_orcid_id:
        return method(*args)
    if use_orcid_id:
        return method(*args, name_counts_index)
    return method(*args, name_counts_index, False)


def build_rust_featurizer(dataset: ANDData) -> tuple[Any, dict[str, float]]:
    """Build a Rust featurizer for a dataset.

    Rust training datasets carry one immutable ``dataset.arrow_paths`` mapping
    attached by
    ``s2and.arrow_training``), which builds through ``from_arrow_paths`` —
    the same fast Arrow door production inference uses — loading every
    signature in the bundle (sorted ids) with Rust-side name counts.
    """
    pre_build_start = time.perf_counter()
    _require_rust_runtime()
    num_threads = resolve_n_jobs(getattr(dataset, "n_jobs", 1))
    arrow_paths = getattr(dataset, "arrow_paths", None)
    if not arrow_paths:
        raise RuntimeError(
            "Rust featurizer construction requires Arrow IPC artifacts "
            "(dataset.arrow_paths). Build the dataset through "
            "s2and.arrow_training.build_training_anddata_from_arrow."
        )
    name_counts_provenance = getattr(dataset, "name_counts_provenance", None)
    if not isinstance(name_counts_provenance, Mapping):
        raise ValueError("Arrow-backed training requires name_counts_provenance")
    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_start = time.perf_counter()
    featurizer = build_rust_featurizer_from_arrow_paths(
        arrow_paths,
        expected_normalization_version=require_normalization_version(
            name_counts_provenance.get("normalization_version"),
            context="Arrow-backed training name-count provenance",
        ),
        signature_ids=None,
        name_tuples=getattr(dataset, "name_tuples", None),
        load_name_counts=True,
        preprocess=bool(getattr(dataset, "preprocess", True)),
        cluster_seed_require_value=float(CLUSTER_SEEDS_LOOKUP["require"]),
        cluster_seed_disallow_value=float(CLUSTER_SEEDS_LOOKUP["disallow"]),
        num_threads=num_threads,
        use_orcid_id=bool(getattr(dataset, "use_orcid_id", True)),
    )
    ffi_seconds = time.perf_counter() - ffi_start
    return (
        featurizer,
        {
            "pre_build_seconds": pre_build_seconds,
            "ffi_seconds": ffi_seconds,
            "post_build_seconds": 0.0,
        },
    )


def _build_rust_featurizer_strict(dataset: ANDData) -> tuple[Any, dict[str, float], float]:
    build_start = time.perf_counter()
    featurizer, build_timings = build_rust_featurizer(dataset)
    return featurizer, build_timings, time.perf_counter() - build_start


def _get_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> Any:
    _require_rust_runtime()
    operation, run_id = _runtime_callsite_for_logs(dataset, runtime_context)
    dataset_mode = _dataset_mode_for_logs(dataset)
    ds_log = _dataset_name_for_logs(dataset)

    state = _rust_featurizer_state(dataset)
    with state.lock:
        cache_key = _rust_featurizer_cache_key(dataset)
        previous_entry = state.entry
        if previous_entry is not None and previous_entry.cache_key == cache_key:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=hit dataset=%s mode=%s op=%s run=%s builds=%d",
                ds_log,
                dataset_mode,
                operation,
                run_id,
                previous_entry.build_count,
            )
            return previous_entry.featurizer

        logger.info(
            "Telemetry: rust_featurizer_cache cache=miss dataset=%s mode=%s op=%s run=%s builds=%d",
            ds_log,
            dataset_mode,
            operation,
            run_id,
            0 if previous_entry is None else previous_entry.build_count,
        )
        featurizer, build_timings, build_seconds = _build_rust_featurizer_strict(dataset)
        if (
            getattr(dataset, "_cluster_seeds_source", "python") != "arrow"
            and getattr(dataset, "_rust_cluster_seeds_synced_version", None) == cache_key.cluster_seeds_version
        ):
            seed_overlay_start = time.perf_counter()
            featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
            seed_overlay_seconds = time.perf_counter() - seed_overlay_start
            build_seconds += seed_overlay_seconds
            build_timings = {
                **build_timings,
                "post_build_seconds": build_timings.get("post_build_seconds", 0.0) + seed_overlay_seconds,
            }
        current_cache_key = _rust_featurizer_cache_key(dataset)
        if current_cache_key != cache_key:
            raise RuntimeError(
                "Rust featurizer inputs changed while it was being built "
                f"(dataset={ds_log}, mode={dataset_mode}, run={run_id})"
            )
        build_count = 1 if previous_entry is None else previous_entry.build_count + 1
        logger.info(
            "Telemetry: rust_core_build seconds=%.3f dataset=%s path=%s count=%d pre=%.3f ffi=%.3f post=%.3f",
            build_seconds,
            ds_log,
            "from_arrow_paths",
            build_count,
            build_timings.get("pre_build_seconds", 0.0),
            build_timings.get("ffi_seconds", 0.0),
            build_timings.get("post_build_seconds", 0.0),
        )
        state.entry = _CacheEntry(
            cache_key=cache_key,
            featurizer=featurizer,
            build_count=build_count,
        )
        logger.info(
            "Telemetry: rust_featurizer_cache_fill source=build dataset=%s path=%s count=%d",
            ds_log,
            "from_arrow_paths",
            build_count,
        )
        return featurizer


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    state = _rust_featurizer_state(dataset)
    with state.lock:
        removed = state.entry is not None
        state.entry = None
        return removed


def clear_rust_featurizer_cache() -> int:
    """Clear all in-memory Rust featurizer cache entries."""
    with _RUST_FEATURIZER_STATES_LOCK:
        states = list(_RUST_FEATURIZER_STATES.values())
    count = 0
    for state in states:
        with state.lock:
            count += state.entry is not None
            state.entry = None
    return count


def warm_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> None:
    """Preload the Rust featurizer into memory for low-latency inference."""
    _get_rust_featurizer(
        dataset,
        runtime_context=runtime_context,
    )


__all__ = [
    "build_rust_featurizer",
    "build_block_upper_triangle_feature_matrix_indexed_rust",
    "build_linker_pair_aggregate_stats_arrays_rust",
    "build_linker_pair_distance_accumulators_rust",
    "build_linker_pair_features_and_aggregate_stats_arrays_rust",
    "build_rust_featurizer_from_arrow_paths",
    "clear_rust_featurizer_cache",
    "evict_rust_featurizer",
    "get_constraint_labels_index_arrays_rust",
    "get_constraints_block_upper_triangle_indexed_rust",
    "get_constraints_matrix_indexed_rust",
    "s2and_rust",
    "update_rust_cluster_seeds",
    "warm_rust_featurizer",
]
