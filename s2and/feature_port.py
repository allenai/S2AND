import hashlib
import logging
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

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
    """Composite cache entry for one Rust featurizer build option."""

    __slots__ = ("featurizer", "build_count")

    def __init__(
        self,
        featurizer: Any,
        build_count: int = 0,
    ):
        self.featurizer = featurizer
        self.build_count = build_count


class _InFlightFeaturizerBuild:
    """Tracks a single in-flight Rust featurizer build for a dataset."""

    __slots__ = ("event", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: Exception | None = None


@dataclass(frozen=True)
class _CollectionFingerprint:
    """Cheap shallow fingerprint for dataset objects that key cached Rust featurizers."""

    object_id: int
    length: int
    digest: int = 0


@dataclass(frozen=True)
class _RustFeaturizerNonSeedFingerprint:
    preprocess: bool
    use_orcid_id: bool
    n_jobs: int
    source_paths: tuple[_SourcePathFingerprint, ...]
    signatures: _CollectionFingerprint
    papers: _CollectionFingerprint
    specter_embeddings: _CollectionFingerprint
    name_tuples: _CollectionFingerprint


@dataclass(frozen=True)
class _RustFeaturizerSeedFingerprint:
    cluster_seeds_version: int
    cluster_seeds_require: _CollectionFingerprint
    cluster_seeds_disallow: _CollectionFingerprint


@dataclass(frozen=True)
class _RustFeaturizerCacheKey:
    non_seed: _RustFeaturizerNonSeedFingerprint
    seed: _RustFeaturizerSeedFingerprint


_RustFeaturizerBuildCountKey = str

# Single WeakKeyDictionary eliminates desync risk between separate weak dicts.
_RUST_FEATURIZER_CACHE: "weakref.WeakKeyDictionary[ANDData, dict[_RustFeaturizerCacheKey, _CacheEntry]]" = (
    weakref.WeakKeyDictionary()
)
_RUST_FEATURIZER_CACHE_LOCK = threading.RLock()
_RUST_FEATURIZER_INFLIGHT_BUILDS: weakref.WeakKeyDictionary[
    ANDData, dict[_RustFeaturizerCacheKey, _InFlightFeaturizerBuild]
] = weakref.WeakKeyDictionary()
_RUST_FEATURIZER_BUILD_COUNTS: "weakref.WeakKeyDictionary[ANDData, dict[_RustFeaturizerBuildCountKey, int]]" = (
    weakref.WeakKeyDictionary()
)
_RUST_FEATURIZER_CACHE_EPOCHS: "weakref.WeakKeyDictionary[ANDData, int]" = weakref.WeakKeyDictionary()
_RUST_CLUSTER_SEED_UPDATE_LOCKS: "weakref.WeakKeyDictionary[ANDData, Any]" = weakref.WeakKeyDictionary()
RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES = 3
RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS = 0.01


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


def _rust_featurizer_empty_wait_max_retries() -> int:
    return max(1, int(RUST_FEATURIZER_EMPTY_WAIT_MAX_RETRIES))


def _rust_featurizer_empty_wait_backoff_seconds() -> float:
    return max(0.0, float(RUST_FEATURIZER_EMPTY_WAIT_BACKOFF_SECONDS))


def _bounded_repr(value: Any) -> str:
    text = repr(value)
    if len(text) > 256:
        return f"{text[:256]}..."
    return text


def _fingerprint_token_digest(tokens: list[tuple[str, int]]) -> int:
    digest = 14695981039346656037
    for token_text, token_id in sorted(tokens):
        token_hash = hash((token_text, token_id)) & 0xFFFFFFFFFFFFFFFF
        digest ^= token_hash
        digest = (digest * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest


def _collection_value_token(value: Any) -> int:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(tuple(array.shape)).encode("utf-8"))
        hasher.update(array.view(np.uint8).tobytes())
        return int.from_bytes(hasher.digest(), "little", signed=False)
    try:
        return hash(value) & 0xFFFFFFFFFFFFFFFF
    except TypeError:
        return id(value)


def _collection_fingerprint(value: Any) -> _CollectionFingerprint:
    if value is None:
        return _CollectionFingerprint(object_id=0, length=0)
    try:
        length = len(value)
    except TypeError:
        return _CollectionFingerprint(object_id=id(value), length=-1)

    tokens: list[tuple[str, int]] = []
    if isinstance(value, Mapping):
        tokens = [
            (_bounded_repr(sample_key), _collection_value_token(sample_value))
            for sample_key, sample_value in value.items()
        ]
    elif isinstance(value, frozenset):
        # Frozenset caches its own hash after the first call. Arrow attachment
        # freezes name aliases, making repeat cache validation O(1) while
        # preventing undetectable in-place mutation.
        return _CollectionFingerprint(object_id=id(value), length=int(length), digest=hash(value))
    elif isinstance(value, set):
        tokens = [(_bounded_repr(sample_value), 0) for sample_value in value]
    return _CollectionFingerprint(object_id=id(value), length=int(length), digest=_fingerprint_token_digest(tokens))


def _rust_featurizer_source_paths(dataset: Any) -> tuple[_SourcePathFingerprint, ...]:
    arrow_paths = getattr(dataset, "arrow_paths", None)
    if not isinstance(arrow_paths, Mapping) or not arrow_paths:
        raise RuntimeError("Rust training featurization requires dataset.arrow_paths")
    generation = getattr(dataset, "arrow_artifact_generation", None)
    if not isinstance(generation, str) or not generation:
        raise ValueError("dataset.arrow_paths requires a retained arrow_artifact_generation")
    exact_paths = tuple(sorted((str(key), str(value)) for key, value in arrow_paths.items()))
    return (("validated_arrow_generation", generation, exact_paths),)


def _rust_featurizer_non_seed_fingerprint(dataset: Any) -> _RustFeaturizerNonSeedFingerprint:
    empty = _CollectionFingerprint(object_id=0, length=0)
    return _RustFeaturizerNonSeedFingerprint(
        preprocess=bool(getattr(dataset, "preprocess", False)),
        use_orcid_id=bool(getattr(dataset, "use_orcid_id", True)),
        n_jobs=resolve_n_jobs(getattr(dataset, "n_jobs", 1)),
        source_paths=_rust_featurizer_source_paths(dataset),
        # from_arrow_paths does not consume the Python object graph. Excluding
        # it avoids rescanning duplicate training dictionaries on every cache
        # lookup and prevents irrelevant Python mutations from rebuilding the
        # native featurizer.
        signatures=empty,
        papers=empty,
        specter_embeddings=empty,
        name_tuples=_collection_fingerprint(getattr(dataset, "name_tuples", None)),
    )


def _cached_rust_featurizer_non_seed_fingerprint(dataset: Any) -> _RustFeaturizerNonSeedFingerprint:
    # Material state must be observed on every lookup. Arrow-backed datasets
    # use immutable paths and frozen name aliases, so this remains a small,
    # constant-memory boundary check rather than a scan of their Python rows.
    return _rust_featurizer_non_seed_fingerprint(dataset)


def _rust_featurizer_cache_key(
    dataset: Any,
    *,
    cluster_seeds_version: int | None = None,
) -> _RustFeaturizerCacheKey:
    resolved_seed_version = (
        _cluster_seeds_version_for_cache(dataset) if cluster_seeds_version is None else int(cluster_seeds_version)
    )
    return _RustFeaturizerCacheKey(
        non_seed=_cached_rust_featurizer_non_seed_fingerprint(dataset),
        seed=_RustFeaturizerSeedFingerprint(
            cluster_seeds_version=resolved_seed_version,
            cluster_seeds_require=_collection_fingerprint(getattr(dataset, "cluster_seeds_require", None)),
            cluster_seeds_disallow=_collection_fingerprint(getattr(dataset, "cluster_seeds_disallow", None)),
        ),
    )


def _rust_featurizer_cache_key_for_logs(cache_key: _RustFeaturizerCacheKey) -> str:
    return (
        f"seed={cache_key.seed.cluster_seeds_version} "
        f"non_seed={hash(cache_key.non_seed)} "
        f"require_len={cache_key.seed.cluster_seeds_require.length} "
        f"disallow_len={cache_key.seed.cluster_seeds_disallow.length}"
    )


def _rust_cluster_seed_update_lock(dataset: ANDData) -> Any:
    with _RUST_FEATURIZER_CACHE_LOCK:
        update_lock = _RUST_CLUSTER_SEED_UPDATE_LOCKS.get(dataset)
        if update_lock is None:
            update_lock = threading.RLock()
            _RUST_CLUSTER_SEED_UPDATE_LOCKS[dataset] = update_lock
        return update_lock


def _rust_featurizer_cache_epoch_locked(dataset: ANDData) -> int:
    return int(_RUST_FEATURIZER_CACHE_EPOCHS.get(dataset, 0))


def _bump_rust_featurizer_cache_epoch_locked(dataset: ANDData) -> int:
    next_epoch = _rust_featurizer_cache_epoch_locked(dataset) + 1
    _RUST_FEATURIZER_CACHE_EPOCHS[dataset] = next_epoch
    return next_epoch


def _cluster_seeds_version_for_cache(dataset: Any) -> int:
    missing = object()
    raw_version = getattr(dataset, "_cluster_seeds_version", missing)
    if raw_version is missing:
        return 0
    return int(cast(Any, raw_version))


def _rust_featurizer_cache_entry_count_locked() -> int:
    return sum(len(entries) for entries in _RUST_FEATURIZER_CACHE.values())


def _prune_stale_cache_entries_locked(dataset: ANDData, current_cache_key: _RustFeaturizerCacheKey) -> int:
    entries = _RUST_FEATURIZER_CACHE.get(dataset)
    if not entries:
        return 0
    stale_keys = [cache_key for cache_key in entries if cache_key != current_cache_key]
    for cache_key in stale_keys:
        del entries[cache_key]
    return len(stale_keys)


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
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        if entries:
            current_entry = entries.get(current_cache_key)
            if current_entry is not None:
                return current_entry.featurizer
            stale_matches = [
                (cache_key, entry)
                for cache_key, entry in entries.items()
                if cache_key.non_seed == current_cache_key.non_seed
            ]
            if stale_matches:
                stale_key, stale_entry = max(stale_matches, key=lambda item: item[0].seed.cluster_seeds_version)
                logger.info(
                    "Telemetry: rust_featurizer_cache cache=seed_update_reuse dataset=%s mode=%s op=%s run=%s "
                    "stale_seed_version=%d current_seed_version=%d",
                    _dataset_name_for_logs(dataset),
                    dataset_mode,
                    operation,
                    run_id,
                    stale_key.seed.cluster_seeds_version,
                    current_seed_version,
                )
                return stale_entry.featurizer
    return _get_rust_featurizer(dataset, runtime_context=runtime_context)


def _promote_cached_rust_featurizer_cluster_seed_version(
    dataset: ANDData,
    featurizer: Any,
    *,
    target_seed_version: int,
) -> bool:
    """Move an updated cached featurizer to the dataset's current seed version."""

    promoted = False
    new_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=target_seed_version)
    with _RUST_FEATURIZER_CACHE_LOCK:
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        if not entries:
            return False
        for cache_key, entry in list(entries.items()):
            if entry.featurizer is featurizer:
                entries[new_cache_key] = entry
                if new_cache_key != cache_key:
                    del entries[cache_key]
                promoted = True
        if promoted:
            for cache_key in list(entries):
                if cache_key != new_cache_key:
                    del entries[cache_key]
    return promoted


def update_rust_cluster_seeds(
    dataset: ANDData,
    runtime_context: Any | None = None,
    *,
    bump_version: bool = False,
) -> None:
    """Sync current Python cluster seeds into the cached Rust featurizer."""

    with _rust_cluster_seed_update_lock(dataset):
        featurizer = _get_cached_rust_featurizer_for_cluster_seed_update(
            dataset,
            runtime_context=runtime_context,
        )
        current_seed_version = _cluster_seeds_version_for_cache(dataset)
        target_seed_version = current_seed_version + 1 if bump_version else current_seed_version
        previous_require = dict(featurizer.cluster_seeds_require())
        previous_disallow = {tuple(pair) for pair in featurizer.cluster_seeds_disallow()}
        featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
        try:
            _promote_cached_rust_featurizer_cluster_seed_version(
                dataset,
                featurizer,
                target_seed_version=target_seed_version,
            )
        except Exception:
            featurizer.update_cluster_seeds(previous_require, previous_disallow)
            raise
        if bump_version:
            dataset._cluster_seeds_version = target_seed_version


def _rust_featurizer_build_count_key(
    cache_key: _RustFeaturizerCacheKey | None,
) -> _RustFeaturizerBuildCountKey:
    del cache_key
    return "from_arrow_paths"


def _increment_rust_featurizer_build_count_locked(
    dataset: ANDData,
    cache_key: _RustFeaturizerCacheKey | None = None,
) -> int:
    build_count_key = _rust_featurizer_build_count_key(cache_key)
    counts = _RUST_FEATURIZER_BUILD_COUNTS.get(dataset)
    if counts is None:
        counts = {}
        _RUST_FEATURIZER_BUILD_COUNTS[dataset] = counts
    count = int(counts.get(build_count_key, 0)) + 1
    counts[build_count_key] = count
    return count


def _rust_featurizer_build_count(
    dataset: ANDData,
    cache_key: _RustFeaturizerCacheKey | None = None,
) -> int:
    build_count_key = _rust_featurizer_build_count_key(cache_key)
    with _RUST_FEATURIZER_CACHE_LOCK:
        counts = _RUST_FEATURIZER_BUILD_COUNTS.get(dataset)
        return 0 if counts is None else int(counts.get(build_count_key, 0))


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


@dataclass(frozen=True)
class _RustFeaturizerBuildContext:
    operation: str
    run_id: str
    dataset_mode: str
    dataset_name_for_logs: str
    cluster_seeds_version: int
    cache_epoch: int
    cache_key: _RustFeaturizerCacheKey


def _rust_featurizer_build_context(
    *,
    operation: str,
    run_id: str,
    dataset_mode: str,
    dataset_name_for_logs: str,
    cluster_seeds_version: int,
    cache_epoch: int,
    cache_key: _RustFeaturizerCacheKey,
) -> _RustFeaturizerBuildContext:
    return _RustFeaturizerBuildContext(
        operation=operation,
        run_id=run_id,
        dataset_mode=dataset_mode,
        dataset_name_for_logs=dataset_name_for_logs,
        cluster_seeds_version=cluster_seeds_version,
        cache_epoch=cache_epoch,
        cache_key=cache_key,
    )


def _get_or_wait_for_cached(
    dataset: ANDData,
    *,
    build_context: _RustFeaturizerBuildContext,
) -> tuple[Any | None, _InFlightFeaturizerBuild | None]:
    inflight_build: _InFlightFeaturizerBuild | None = None
    cache_key = build_context.cache_key
    with _RUST_FEATURIZER_CACHE_LOCK:
        if _rust_featurizer_cache_epoch_locked(dataset) != build_context.cache_epoch:
            return None, None
        current_seed_version = _cluster_seeds_version_for_cache(dataset)
        current_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=current_seed_version)
        if current_cache_key != build_context.cache_key:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=retry_material_fingerprint dataset=%s mode=%s op=%s run=%s "
                "snapshotted_seed_version=%d current_seed_version=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                build_context.cluster_seeds_version,
                current_seed_version,
            )
            return None, None
        pruned_count = _prune_stale_cache_entries_locked(dataset, current_cache_key)
        if pruned_count:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=prune_stale_material dataset=%s mode=%s op=%s run=%s "
                "current_seed_version=%d pruned=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                current_seed_version,
                pruned_count,
            )
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(cache_key)
        if entry is not None:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=hit dataset=%s mode=%s op=%s run=%s builds=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                entry.build_count,
            )
            return entry.featurizer, None
        if entries:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=option_miss dataset=%s mode=%s op=%s run=%s cached_keys=%s",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                [_rust_featurizer_cache_key_for_logs(entry_key) for entry_key in entries],
            )

        inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
        inflight_build = None if inflight_entries is None else inflight_entries.get(cache_key)
        if inflight_build is None:
            inflight_build = _InFlightFeaturizerBuild()
            if inflight_entries is None:
                inflight_entries = {}
                _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset] = inflight_entries
            inflight_entries[cache_key] = inflight_build
            logger.info(
                "Telemetry: rust_featurizer_cache cache=miss dataset=%s mode=%s op=%s run=%s builds=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                0,
            )
            return None, inflight_build

        logger.info(
            "Telemetry: rust_featurizer_cache cache=wait dataset=%s mode=%s op=%s run=%s builds=%d",
            build_context.dataset_name_for_logs,
            build_context.dataset_mode,
            build_context.operation,
            build_context.run_id,
            0,
        )

    inflight_build.event.wait()
    with _RUST_FEATURIZER_CACHE_LOCK:
        if _rust_featurizer_cache_epoch_locked(dataset) != build_context.cache_epoch:
            return None, None
        current_seed_version = _cluster_seeds_version_for_cache(dataset)
        current_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=current_seed_version)
        if current_cache_key != build_context.cache_key:
            logger.info(
                "Telemetry: rust_featurizer_cache cache=retry_material_fingerprint_after_wait "
                "dataset=%s mode=%s op=%s run=%s "
                "snapshotted_seed_version=%d current_seed_version=%d",
                build_context.dataset_name_for_logs,
                build_context.dataset_mode,
                build_context.operation,
                build_context.run_id,
                build_context.cluster_seeds_version,
                current_seed_version,
            )
            return None, None
        entries = _RUST_FEATURIZER_CACHE.get(dataset)
        entry = None if entries is None else entries.get(cache_key)
        if entry is not None:
            return entry.featurizer, None
        build_error = inflight_build.error
    if build_error is not None:
        raise RuntimeError(
            "Rust featurizer build failed for dataset="
            f"{build_context.dataset_name_for_logs} while waiting for concurrent builder"
        ) from build_error
    return None, None


def _build_and_cache_rust_featurizer(
    dataset: ANDData,
    *,
    inflight_build: _InFlightFeaturizerBuild,
    build_context: _RustFeaturizerBuildContext,
) -> Any:
    cache_key = build_context.cache_key
    try:
        featurizer, build_timings, build_seconds = _build_rust_featurizer_strict(dataset)
        logger.info(
            "Telemetry: rust_core_build seconds=%.3f dataset=%s path=%s count=%d pre=%.3f ffi=%.3f post=%.3f",
            build_seconds,
            build_context.dataset_name_for_logs,
            "from_arrow_paths",
            _rust_featurizer_build_count(dataset, cache_key),
            build_timings.get("pre_build_seconds", 0.0),
            build_timings.get("ffi_seconds", 0.0),
            build_timings.get("post_build_seconds", 0.0),
        )

        with _RUST_FEATURIZER_CACHE_LOCK:
            if _rust_featurizer_cache_epoch_locked(dataset) != build_context.cache_epoch:
                inflight_build.error = None
                inflight_build.event.set()
                inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
                if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                    del inflight_entries[cache_key]
                    if not inflight_entries:
                        del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
                return None
            current_seed_version = _cluster_seeds_version_for_cache(dataset)
            current_cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=current_seed_version)
            if current_cache_key != build_context.cache_key:
                logger.info(
                    "Telemetry: rust_featurizer_cache cache=discard_stale_build dataset=%s mode=%s op=%s run=%s "
                    "snapshotted_seed_version=%d current_seed_version=%d",
                    build_context.dataset_name_for_logs,
                    build_context.dataset_mode,
                    build_context.operation,
                    build_context.run_id,
                    build_context.cluster_seeds_version,
                    current_seed_version,
                )
                _prune_stale_cache_entries_locked(dataset, current_cache_key)
                inflight_build.error = None
                inflight_build.event.set()
                inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
                if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                    del inflight_entries[cache_key]
                    if not inflight_entries:
                        del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
                return None
            _prune_stale_cache_entries_locked(dataset, build_context.cache_key)
            build_count = _increment_rust_featurizer_build_count_locked(dataset, cache_key)
            entries = _RUST_FEATURIZER_CACHE.get(dataset)
            if entries is None:
                entries = {}
                _RUST_FEATURIZER_CACHE[dataset] = entries
            entries[cache_key] = _CacheEntry(
                featurizer=featurizer,
                build_count=build_count,
            )
            logger.info(
                "Telemetry: rust_featurizer_cache_fill source=build dataset=%s path=%s count=%d",
                build_context.dataset_name_for_logs,
                "from_arrow_paths",
                build_count,
            )
            inflight_build.error = None
            inflight_build.event.set()
            inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
            if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                del inflight_entries[cache_key]
                if not inflight_entries:
                    del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
    except Exception as build_error:
        with _RUST_FEATURIZER_CACHE_LOCK:
            inflight_build.error = build_error
            inflight_build.event.set()
            inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.get(dataset)
            if inflight_entries is not None and inflight_entries.get(cache_key) is inflight_build:
                del inflight_entries[cache_key]
                if not inflight_entries:
                    del _RUST_FEATURIZER_INFLIGHT_BUILDS[dataset]
        raise

    return featurizer


def _get_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> Any:
    _require_rust_runtime()
    operation, run_id = _runtime_callsite_for_logs(dataset, runtime_context)
    dataset_mode = _dataset_mode_for_logs(dataset)
    ds_log = _dataset_name_for_logs(dataset)
    max_empty_wait_retries = _rust_featurizer_empty_wait_max_retries()
    empty_wait_backoff_seconds = _rust_featurizer_empty_wait_backoff_seconds()
    empty_wait_attempt = 0
    stale_build_attempt = 0

    with _rust_cluster_seed_update_lock(dataset):
        while True:
            with _RUST_FEATURIZER_CACHE_LOCK:
                cluster_seeds_version = _cluster_seeds_version_for_cache(dataset)
                cache_epoch = _rust_featurizer_cache_epoch_locked(dataset)
            cache_key = _rust_featurizer_cache_key(dataset, cluster_seeds_version=cluster_seeds_version)
            build_context = _rust_featurizer_build_context(
                operation=operation,
                run_id=run_id,
                dataset_mode=dataset_mode,
                dataset_name_for_logs=ds_log,
                cluster_seeds_version=cluster_seeds_version,
                cache_epoch=cache_epoch,
                cache_key=cache_key,
            )
            featurizer, inflight_build = _get_or_wait_for_cached(dataset, build_context=build_context)
            if featurizer is not None:
                return featurizer
            if inflight_build is None:
                stale_build_attempt = 0
                empty_wait_attempt += 1
                if empty_wait_attempt > max_empty_wait_retries:
                    raise RuntimeError(
                        "Rust featurizer cache resolution exhausted retries for empty wait state "
                        f"(dataset={ds_log}, mode={dataset_mode}, run={run_id}, "
                        f"attempts={empty_wait_attempt}, max_retries={max_empty_wait_retries})"
                    )
                backoff_seconds = empty_wait_backoff_seconds * float(empty_wait_attempt)
                logger.warning(
                    "Telemetry: rust_featurizer_cache cache=retry_empty dataset=%s mode=%s op=%s run=%s "
                    "attempt=%d/%d backoff_seconds=%.3f",
                    ds_log,
                    dataset_mode,
                    operation,
                    run_id,
                    empty_wait_attempt,
                    max_empty_wait_retries,
                    backoff_seconds,
                )
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds)
                continue
            featurizer = _build_and_cache_rust_featurizer(
                dataset,
                inflight_build=inflight_build,
                build_context=build_context,
            )
            if featurizer is not None:
                return featurizer
            stale_build_attempt += 1
            if stale_build_attempt > max_empty_wait_retries:
                raise RuntimeError(
                    "Rust featurizer cache resolution exhausted retries for stale build state "
                    f"(dataset={ds_log}, mode={dataset_mode}, run={run_id}, "
                    f"attempts={stale_build_attempt}, max_retries={max_empty_wait_retries})"
                )
            logger.warning(
                "Telemetry: rust_featurizer_cache cache=retry_stale_build dataset=%s mode=%s op=%s run=%s "
                "attempt=%d/%d",
                ds_log,
                dataset_mode,
                operation,
                run_id,
                stale_build_attempt,
                max_empty_wait_retries,
            )
            empty_wait_attempt = 0
            continue


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        _bump_rust_featurizer_cache_epoch_locked(dataset)
        removed = False
        if dataset in _RUST_FEATURIZER_CACHE:
            del _RUST_FEATURIZER_CACHE[dataset]
            removed = True
        inflight_entries = _RUST_FEATURIZER_INFLIGHT_BUILDS.pop(dataset, None)
        if inflight_entries:
            for inflight_build in inflight_entries.values():
                inflight_build.error = None
                inflight_build.event.set()
        if dataset in _RUST_FEATURIZER_BUILD_COUNTS:
            del _RUST_FEATURIZER_BUILD_COUNTS[dataset]
        return removed


def clear_rust_featurizer_cache() -> int:
    """Clear all in-memory Rust featurizer cache entries."""
    with _RUST_FEATURIZER_CACHE_LOCK:
        count = _rust_featurizer_cache_entry_count_locked()
        datasets = set(_RUST_FEATURIZER_CACHE.keys())
        datasets.update(_RUST_FEATURIZER_INFLIGHT_BUILDS.keys())
        datasets.update(_RUST_FEATURIZER_BUILD_COUNTS.keys())
        for dataset in datasets:
            _bump_rust_featurizer_cache_epoch_locked(dataset)
        for inflight_entries in _RUST_FEATURIZER_INFLIGHT_BUILDS.values():
            for inflight_build in inflight_entries.values():
                inflight_build.error = None
                inflight_build.event.set()
        _RUST_FEATURIZER_CACHE.clear()
        _RUST_FEATURIZER_INFLIGHT_BUILDS.clear()
        _RUST_FEATURIZER_BUILD_COUNTS.clear()
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
