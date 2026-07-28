import logging
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import CLUSTER_SEEDS_LOOKUP
from s2and.data import ANDData
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
_ClusterSeedStamp = tuple[object, int, object, int]
_MISSING = object()


class _MutationTrackedDict(dict[Any, Any]):
    """A compact dict that records built-in content mutations."""

    __slots__ = ("_identity_token", "_mutation_version")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if hasattr(self, "_mutation_version"):
            self._touch()
        else:
            self._identity_token = object()
            self._mutation_version = 0
        dict.__init__(self, *args, **kwargs)

    @property
    def identity_token(self) -> object:
        return self._identity_token

    @property
    def mutation_version(self) -> int:
        return self._mutation_version

    def _touch(self) -> None:
        if not hasattr(self, "_mutation_version"):
            self._identity_token = object()
            self._mutation_version = 1
        else:
            self._mutation_version += 1

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self and dict.__getitem__(self, key) == value:
            return
        self._touch()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, key)
        self._touch()

    def __ior__(self, other: Any) -> "_MutationTrackedDict":
        if not other:
            return self
        self._touch()
        dict.__ior__(self, other)
        return self

    def clear(self) -> None:
        if not self:
            return
        self._touch()
        dict.clear(self)

    def pop(self, key: Any, default: Any = _MISSING) -> Any:
        if key not in self:
            if default is _MISSING:
                return dict.pop(self, key)
            return default
        self._touch()
        return dict.pop(self, key)

    def popitem(self) -> tuple[Any, Any]:
        self._touch()
        return dict.popitem(self)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key in self:
            return dict.__getitem__(self, key)
        self._touch()
        return dict.setdefault(self, key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            return
        if len(args) == 1 and not kwargs and isinstance(args[0], Mapping) and not args[0]:
            return
        self._touch()
        dict.update(self, *args, **kwargs)


class _MutationTrackedSet(set[tuple[Any, Any]]):
    """A compact set that records built-in content mutations."""

    __slots__ = ("_identity_token", "_mutation_version")

    def __init__(self, *args: Any) -> None:
        if hasattr(self, "_mutation_version"):
            self._touch()
        else:
            self._identity_token = object()
            self._mutation_version = 0
        set.__init__(self, *args)

    @property
    def identity_token(self) -> object:
        return self._identity_token

    @property
    def mutation_version(self) -> int:
        return self._mutation_version

    def _touch(self) -> None:
        if not hasattr(self, "_mutation_version"):
            self._identity_token = object()
            self._mutation_version = 1
        else:
            self._mutation_version += 1

    def __iand__(self, other: Any) -> "_MutationTrackedSet":
        if not self or other is self:
            return self
        self._touch()
        set.__iand__(self, other)
        return self

    def __ior__(self, other: Any) -> "_MutationTrackedSet":
        if not other or other is self:
            return self
        self._touch()
        set.__ior__(self, other)
        return self

    def __isub__(self, other: Any) -> "_MutationTrackedSet":
        if not self or not other:
            return self
        self._touch()
        set.__isub__(self, other)
        return self

    def __ixor__(self, other: Any) -> "_MutationTrackedSet":
        if not other:
            return self
        self._touch()
        set.__ixor__(self, other)
        return self

    def add(self, element: tuple[Any, Any]) -> None:
        if element in self:
            return
        self._touch()
        set.add(self, element)

    def clear(self) -> None:
        if not self:
            return
        self._touch()
        set.clear(self)

    def difference_update(self, *others: Any) -> None:
        if not self or not others or all(not other for other in others):
            return
        self._touch()
        set.difference_update(self, *others)

    def discard(self, element: object) -> None:
        if element not in self:
            return
        self._touch()
        set.discard(self, element)

    def intersection_update(self, *others: Any) -> None:
        if not self or not others:
            return
        self._touch()
        set.intersection_update(self, *others)

    def pop(self) -> Any:
        value = set.pop(self)
        self._touch()
        return value

    def remove(self, element: tuple[Any, Any]) -> None:
        set.remove(self, element)
        self._touch()

    def symmetric_difference_update(self, other: Any) -> None:
        if not other:
            return
        self._touch()
        set.symmetric_difference_update(self, other)

    def update(self, *others: Any) -> None:
        if not others or all(not other for other in others):
            return
        self._touch()
        set.update(self, *others)


class _RustFeaturizerState:
    """One dataset's cached native featurizer and synchronized inputs."""

    __slots__ = ("lock", "featurizer", "build_inputs", "synced_seed_stamp", "build_count")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.featurizer: Any | None = None
        self.build_inputs: tuple[Any, ...] | None = None
        self.synced_seed_stamp: _ClusterSeedStamp | None = None
        self.build_count = 0


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


def _rust_featurizer_build_inputs(dataset: ANDData) -> tuple[Any, ...]:
    arrow_dataset = dataset.arrow_dataset
    if arrow_dataset is None:
        raise RuntimeError("Rust training featurization requires dataset.arrow_dataset")
    return (
        bool(dataset.preprocess),
        bool(dataset.use_orcid_id),
        resolve_n_jobs(dataset.n_jobs),
        arrow_dataset.native,
        frozenset(dataset.name_tuples or ()),
    )


def _rust_featurizer_state(dataset: ANDData) -> _RustFeaturizerState:
    state = getattr(dataset, "_s2and_rust_featurizer_state", None)
    if state is None:
        state = vars(dataset).setdefault("_s2and_rust_featurizer_state", _RustFeaturizerState())
    return state


def _rust_featurizer_build_count(dataset: ANDData) -> int:
    """Return the cached Rust featurizer's build count for one dataset."""

    state = _rust_featurizer_state(dataset)
    with state.lock:
        return state.build_count


def _cluster_seed_stamp(dataset: ANDData) -> _ClusterSeedStamp | None:
    """Return constant-size identity/version state for Python-owned seeds."""

    require = getattr(dataset, "cluster_seeds_require", None)
    if require is None:
        require = {}
    disallow = getattr(dataset, "cluster_seeds_disallow", None)
    if disallow is None:
        disallow = set()

    if not isinstance(require, _MutationTrackedDict):
        require = _MutationTrackedDict(require)
        dataset.cluster_seeds_require = require
    if not isinstance(disallow, _MutationTrackedSet):
        disallow = _MutationTrackedSet(disallow)
        dataset.cluster_seeds_disallow = disallow
    return (
        require.identity_token,
        require.mutation_version,
        disallow.identity_token,
        disallow.mutation_version,
    )


def build_rust_featurizer_from_arrow_dataset(
    arrow_dataset: ArrowDataset,
    *,
    signature_ids: Sequence[Any] | None = None,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    preprocess: bool = True,
    cluster_seed_require_value: float = 0.0,
    cluster_seed_disallow_value: float = 10000.0,
    num_threads: int | None = None,
    use_orcid_id: bool = True,
    cluster_seeds_path: Path | None = None,
    cluster_seed_disallows_path: Path | None = None,
) -> Any:
    """Build a Rust featurizer from one retained Arrow dataset.

    ``use_orcid_id=False`` suppresses ORCID while native signature records are
    constructed. Request-local cluster-seed sidecars remain explicit.
    """

    if not isinstance(arrow_dataset, ArrowDataset):
        raise TypeError("build_rust_featurizer_from_arrow_dataset requires ArrowDataset")
    resolved_name_tuples = load_packaged_name_tuple_artifact().pairs if name_tuples is None else name_tuples
    return _require_rust_runtime().RustFeaturizer.from_arrow_dataset(
        arrow_dataset.native,
        None if signature_ids is None else [str(value) for value in signature_ids],
        resolved_name_tuples,
        cluster_seeds_path=None if cluster_seeds_path is None else str(cluster_seeds_path),
        cluster_seed_disallows_path=(None if cluster_seed_disallows_path is None else str(cluster_seed_disallows_path)),
        preprocess=bool(preprocess),
        cluster_seed_require_value=float(cluster_seed_require_value),
        cluster_seed_disallow_value=float(cluster_seed_disallow_value),
        num_threads=None if num_threads is None else resolve_n_jobs(num_threads),
        use_orcid_id=bool(use_orcid_id),
    )


def build_rust_featurizer(dataset: ANDData) -> tuple[Any, dict[str, float]]:
    """Build a Rust featurizer for a dataset.

    Rust training datasets retain one open ``dataset.arrow_dataset`` resource,
    which owns the exact files and native name-count snapshot used here.
    """
    pre_build_start = time.perf_counter()
    _require_rust_runtime()
    num_threads = resolve_n_jobs(dataset.n_jobs)
    arrow_dataset = dataset.arrow_dataset
    if arrow_dataset is None:
        raise RuntimeError(
            "Rust featurizer construction requires dataset.arrow_dataset. "
            "Build the dataset through "
            "s2and.arrow_training.build_training_anddata_from_arrow."
        )
    pre_build_seconds = time.perf_counter() - pre_build_start
    ffi_start = time.perf_counter()
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset,
        signature_ids=None,
        name_tuples=dataset.name_tuples,
        preprocess=bool(dataset.preprocess),
        cluster_seed_require_value=float(CLUSTER_SEEDS_LOOKUP["require"]),
        cluster_seed_disallow_value=float(CLUSTER_SEEDS_LOOKUP["disallow"]),
        num_threads=num_threads,
        use_orcid_id=bool(dataset.use_orcid_id),
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


def _get_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> Any:
    _require_rust_runtime()
    ds_log = _dataset_name_for_logs(dataset)

    state = _rust_featurizer_state(dataset)
    with state.lock:
        build_inputs = _rust_featurizer_build_inputs(dataset)
        if state.featurizer is not None and state.build_inputs == build_inputs:
            seed_stamp = _cluster_seed_stamp(dataset)
            if seed_stamp is not None and seed_stamp != state.synced_seed_stamp:
                state.featurizer.update_cluster_seeds(
                    dataset.cluster_seeds_require,
                    dataset.cluster_seeds_disallow,
                )
                state.synced_seed_stamp = seed_stamp
            return state.featurizer

        build_start = time.perf_counter()
        featurizer, build_timings = build_rust_featurizer(dataset)
        build_seconds = time.perf_counter() - build_start
        if _rust_featurizer_build_inputs(dataset) != build_inputs:
            raise RuntimeError(f"Rust featurizer inputs changed while it was being built (dataset={ds_log})")

        seed_stamp = _cluster_seed_stamp(dataset)
        if seed_stamp is not None:
            seed_overlay_start = time.perf_counter()
            featurizer.update_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow)
            seed_overlay_seconds = time.perf_counter() - seed_overlay_start
            build_seconds += seed_overlay_seconds
            build_timings = {
                **build_timings,
                "post_build_seconds": build_timings.get("post_build_seconds", 0.0) + seed_overlay_seconds,
            }
        build_count = state.build_count + 1
        logger.info(
            "Telemetry: rust_core_build seconds=%.3f dataset=%s source=%s count=%d pre=%.3f ffi=%.3f post=%.3f",
            build_seconds,
            ds_log,
            "from_arrow_dataset",
            build_count,
            build_timings.get("pre_build_seconds", 0.0),
            build_timings.get("ffi_seconds", 0.0),
            build_timings.get("post_build_seconds", 0.0),
        )
        state.featurizer = featurizer
        state.build_inputs = build_inputs
        state.synced_seed_stamp = seed_stamp
        state.build_count = build_count
        return featurizer


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    state = _rust_featurizer_state(dataset)
    with state.lock:
        removed = state.featurizer is not None
        state.featurizer = None
        state.build_inputs = None
        state.synced_seed_stamp = None
        state.build_count = 0
        return removed


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
    "build_rust_featurizer_from_arrow_dataset",
    "evict_rust_featurizer",
    "get_constraint_labels_index_arrays_rust",
    "get_constraints_block_upper_triangle_indexed_rust",
    "get_constraints_matrix_indexed_rust",
    "s2and_rust",
    "warm_rust_featurizer",
]
