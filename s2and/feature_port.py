from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from s2and import memory_budget
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import CLUSTER_SEEDS_LOOKUP
from s2and.data import ANDData
from s2and.feature_schema import FEATURE_SCHEMA
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
    score_linker_pair_features_and_aggregate_stats_rust,
)
from s2and.thread_config import resolve_n_jobs

if TYPE_CHECKING:
    from s2and.featurizer import FeaturizationInfo
    from s2and.prediction_state import PredictionState

# Treat extension as Any for typing; it is optional and loaded on first use.
s2and_rust: Any | None = None

logger = logging.getLogger("s2and")
_S2AND_RUST_LOAD_LOCK = threading.Lock()


class _RustFeaturizerState:
    """One dataset's cached native feature backing and build inputs."""

    __slots__ = ("lock", "featurizer", "build_inputs", "build_count")

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.featurizer: Any | None = None
        self.build_inputs: tuple[Any, ...] | None = None
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


def build_rust_featurizer_from_planner(
    planner: Any,
    *,
    signature_ids: Sequence[Any],
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    preprocess: bool = True,
    num_threads: int | None = None,
    cluster_seeds_path: Path | None = None,
    cluster_seed_disallows_path: Path | None = None,
) -> Any | None:
    """Build a Rust featurizer from rows a raw candidate planner already holds.

    The planner holds its seeds' rows and its latest plan's query rows, which
    cover the plan's signatures. The result equals
    ``build_rust_featurizer_from_arrow_dataset`` over the same arguments, without
    reading Arrow again. Returns None when the planner lacks some row; callers
    then build from the Arrow dataset.
    """

    resolved_name_tuples = load_packaged_name_tuple_artifact().pairs if name_tuples is None else name_tuples
    return planner.build_featurizer(
        [str(value) for value in signature_ids],
        resolved_name_tuples,
        cluster_seeds_path=None if cluster_seeds_path is None else str(cluster_seeds_path),
        cluster_seed_disallows_path=(None if cluster_seed_disallows_path is None else str(cluster_seed_disallows_path)),
        preprocess=bool(preprocess),
        num_threads=None if num_threads is None else resolve_n_jobs(num_threads),
    )


def _arrow_signature_ids(arrow_dataset: ArrowDataset) -> set[str]:
    """Read the Arrow signatures table's IDs without building native signature records."""

    import pyarrow.ipc as ipc

    with arrow_dataset.use() as lease, lease.open_file("signatures") as source:
        reader = ipc.open_file(source)
        return {
            str(value)
            for batch_index in range(reader.num_record_batches)
            for value in reader.get_batch(batch_index).column("signature_id").to_pylist()
        }


def build_verified_rust_featurizer(
    dataset: ANDData,
    arrow_dataset: ArrowDataset,
    *,
    signature_ids: Iterable[str],
    name_counts_manifest_sha256: str,
    num_threads: int,
) -> Any:
    """Build a Rust featurizer for selected signatures of ``dataset`` from its converted Arrow data.

    ``dataset`` stays the authority for splits, pairs and cluster truth. It is
    either JSON-loaded or built from ``arrow_dataset`` itself; it need not be preprocessed or carry SPECTER or name
    counts, since every feature comes from Arrow. The caller must already have
    verified the Arrow conversion provenance against ``dataset``'s source files.
    This checks that the Arrow data uses the expected name counts and exactly the
    dataset's signature IDs, and uses the dataset's name tuples and ORCID policy.

    Only ``signature_ids`` and their papers are built, so build time and memory
    follow the scored signatures rather than the dataset. Each pair's features
    depend only on its two signatures and their papers.

    Args:
        dataset: JSON-loaded dataset whose pairs will be featurized.
        arrow_dataset: Open Arrow conversion of the same source files.
        signature_ids: Members of the pairs or blocks to featurize.
        name_counts_manifest_sha256: Release name-count index identity.
        num_threads: Native featurizer threads.

    Returns:
        A seedless Rust featurizer covering exactly ``signature_ids``.
    """

    arrow_counts = arrow_dataset.name_counts_index
    observed_counts = None if arrow_counts is None else arrow_counts.manifest_sha256
    if observed_counts != name_counts_manifest_sha256:
        raise ValueError(
            f"Arrow data for {dataset.name!r} uses the wrong name-count index: "
            f"expected={name_counts_manifest_sha256!r} arrow={observed_counts!r}"
        )
    json_signature_ids = {str(signature_id) for signature_id in dataset.signatures}
    arrow_signature_ids = _arrow_signature_ids(arrow_dataset)
    # A dataset built from this Arrow dataset selects one population from it, such as the
    # development blocks. A JSON-loaded dataset must name exactly the converted signatures.
    if getattr(dataset, "arrow_dataset", None) is arrow_dataset:
        mismatch = not json_signature_ids <= arrow_signature_ids
    else:
        mismatch = arrow_signature_ids != json_signature_ids
    if mismatch:
        raise ValueError(
            f"Arrow signatures for {dataset.name!r} differ from its JSON signatures: "
            f"only_arrow={sorted(arrow_signature_ids - json_signature_ids)[:5]} "
            f"only_json={sorted(json_signature_ids - arrow_signature_ids)[:5]}"
        )
    requested = sorted({str(signature_id) for signature_id in signature_ids})
    unknown = [signature_id for signature_id in requested if signature_id not in json_signature_ids]
    if unknown:
        raise ValueError(f"Requested signatures are not in {dataset.name!r}: {unknown[:5]}")
    featurizer = build_rust_featurizer_from_arrow_dataset(
        arrow_dataset,
        signature_ids=requested,
        name_tuples=dataset.name_tuples,
        preprocess=True,
        cluster_seed_require_value=float(CLUSTER_SEEDS_LOOKUP["require"]),
        cluster_seed_disallow_value=float(CLUSTER_SEEDS_LOOKUP["disallow"]),
        num_threads=num_threads,
        use_orcid_id=bool(dataset.use_orcid_id),
    )
    if len(featurizer.signature_ids()) != len(requested):
        raise RuntimeError(f"Rust featurizer for {dataset.name!r} does not cover the requested signatures")
    return featurizer


def featurize_pairs_with_rust(
    rust_featurizer: Any,
    signature_pairs: Sequence[tuple[str, str, int | float]],
    featurizer_info: FeaturizationInfo,
    *,
    nameless_featurizer_info: FeaturizationInfo | None,
    n_jobs: int,
    nan_value: float,
    total_ram_bytes: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Featurize explicit signature pairs with a Rust featurizer.

    Returns the same ``(features, labels, nameless_features)`` layout as
    :func:`s2and.featurizer.many_pairs_featurize`: selected columns in canonical
    order, labels in input order, and NaN replaced by ``nan_value``. Chunks are
    sized by the Rust batch memory planner from ``total_ram_bytes``.

    Args:
        rust_featurizer: Featurizer whose signatures include every pair member.
        signature_pairs: ``(signature_id_1, signature_id_2, label)`` rows; labels must be non-negative.
        featurizer_info: Main model feature selection.
        nameless_featurizer_info: Nameless model feature selection, or ``None``.
        n_jobs: Native featurization threads.
        nan_value: Replacement for missing feature values.
        total_ram_bytes: Explicit RAM budget, or ``None`` to autodetect.

    Returns:
        Tuple of (features, labels, nameless features or None).
    """

    # Fixed-pair selections arrive as numpy records; normalize like many_pairs_featurize.
    signature_pairs = [(str(pair[0]), str(pair[1]), pair[2]) for pair in signature_pairs]
    labels = np.asarray([float(pair[2]) for pair in signature_pairs], dtype=np.float64)
    if np.any(labels < 0):
        raise ValueError("featurize_pairs_with_rust does not accept partial-supervision (negative) labels")
    main_indices = featurizer_info.selected_feature_indices()
    nameless_indices = [] if nameless_featurizer_info is None else nameless_featurizer_info.selected_feature_indices()
    union_indices = sorted(set(main_indices) | set(nameless_indices))
    row_count = len(signature_pairs)
    features = np.empty((row_count, len(main_indices)), dtype=np.float64)
    nameless_features = (
        None if nameless_featurizer_info is None else np.empty((row_count, len(nameless_indices)), dtype=np.float64)
    )
    if row_count > 0:
        index_by_signature_id = {
            str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
        }
        missing = sorted({member for pair in signature_pairs for member in pair[:2]}.difference(index_by_signature_id))
        if missing:
            raise ValueError(f"Rust featurizer is missing pair signatures: {missing[:10]}")
        plan = memory_budget.compute_rust_batch_chunk_plan(
            num_features=len(union_indices),
            total_pairs=row_count,
            total_rows=row_count,
            selected_feature_count=len(main_indices),
            nameless_feature_count=len(nameless_indices),
            total_ram_bytes=total_ram_bytes,
        )
        chunk_pairs = int(plan.chunk_pairs)
        full_width = len(FEATURE_SCHEMA)
        for start in range(0, row_count, chunk_pairs):
            chunk = signature_pairs[start : start + chunk_pairs]
            computed = np.asarray(
                rust_featurizer.featurize_pairs_matrix_indexed(
                    [(index_by_signature_id[str(left)], index_by_signature_id[str(right)]) for left, right, _ in chunk],
                    union_indices,
                    resolve_n_jobs(n_jobs),
                    np.nan,
                ),
                dtype=np.float64,
            )
            if computed.shape[1] == len(union_indices):
                column_of = {feature: position for position, feature in enumerate(union_indices)}
            elif computed.shape[1] == full_width:
                column_of = {feature: feature for feature in union_indices}
            else:
                raise RuntimeError(
                    f"Rust featurizer returned {computed.shape[1]} columns; "
                    f"expected {len(union_indices)} (selected) or {full_width} (full)"
                )
            if computed.shape[0] != len(chunk):
                raise RuntimeError(f"Rust featurizer returned {computed.shape[0]} rows for {len(chunk)} pairs")
            stop = start + len(chunk)
            features[start:stop] = computed[:, [column_of[feature] for feature in main_indices]]
            if nameless_features is not None:
                nameless_features[start:stop] = computed[:, [column_of[feature] for feature in nameless_indices]]
    if not np.isnan(nan_value):
        features[np.isnan(features)] = nan_value
        if nameless_features is not None:
            nameless_features[np.isnan(nameless_features)] = nan_value
    return features, labels, nameless_features


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
    *,
    prediction_state: PredictionState | None = None,
) -> Any:
    """Return a request-owned seed overlay sharing cached native features."""
    base = _get_rust_feature_data(dataset)
    seeds = dataset if prediction_state is None else prediction_state
    return base.with_cluster_seeds(seeds.cluster_seeds_require, seeds.cluster_seeds_disallow)


def _get_rust_feature_data(dataset: ANDData) -> Any:
    """Return cached seedless native features without copying request seeds."""
    _require_rust_runtime()
    ds_log = _dataset_name_for_logs(dataset)

    state = _rust_featurizer_state(dataset)
    with state.lock:
        build_inputs = _rust_featurizer_build_inputs(dataset)
        if state.featurizer is not None and state.build_inputs == build_inputs:
            return state.featurizer

        build_start = time.perf_counter()
        featurizer, build_timings = build_rust_featurizer(dataset)
        build_seconds = time.perf_counter() - build_start
        if _rust_featurizer_build_inputs(dataset) != build_inputs:
            raise RuntimeError(f"Rust featurizer inputs changed while it was being built (dataset={ds_log})")

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
        state.build_count = build_count
        return featurizer


def release_rust_handle(handle: Any) -> None:
    """Release native state, freeing paper buffers before background cleanup.

    Featurizers free paper buffers before residual scoring can allocate its matrix.
    Remaining metadata can finish in the background. Handles without `release`
    are left to the garbage collector.
    """

    release = getattr(handle, "release", None)
    if callable(release):
        release()


def evict_rust_featurizer(dataset: ANDData) -> bool:
    """Evict a single dataset's Rust featurizer from the in-memory cache."""
    state = _rust_featurizer_state(dataset)
    with state.lock:
        removed = state.featurizer is not None
        state.featurizer = None
        state.build_inputs = None
        state.build_count = 0
        return removed


def warm_rust_featurizer(
    dataset: ANDData,
    runtime_context: Any | None = None,
) -> None:
    """Preload the Rust featurizer into memory for low-latency inference."""
    _get_rust_feature_data(dataset)


__all__ = [
    "build_rust_featurizer",
    "build_block_upper_triangle_feature_matrix_indexed_rust",
    "build_linker_pair_aggregate_stats_arrays_rust",
    "build_linker_pair_distance_accumulators_rust",
    "build_linker_pair_features_and_aggregate_stats_arrays_rust",
    "build_rust_featurizer_from_arrow_dataset",
    "build_rust_featurizer_from_planner",
    "build_verified_rust_featurizer",
    "evict_rust_featurizer",
    "featurize_pairs_with_rust",
    "get_constraint_labels_index_arrays_rust",
    "get_constraints_block_upper_triangle_indexed_rust",
    "get_constraints_matrix_indexed_rust",
    "release_rust_handle",
    "score_linker_pair_features_and_aggregate_stats_rust",
    "s2and_rust",
    "warm_rust_featurizer",
]
