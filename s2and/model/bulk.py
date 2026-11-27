"""Plan bulk clustering, cluster blocks, and reunite subblock predictions."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence, Set
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from s2and import memory_budget
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.corrections.block_corrections import (
    BlockCorrections,
    arrow_small_block_signatures,
    average_linkage_cut,
    corrections_from_rust_featurizer,
    orcid_join_enabled,
    year_gap_split_enabled,
)
from s2and.corrections.orcid_join import join_orcids_across_subblocks
from s2and.corrections.profile_exclusions import enforce_profile_exclusions_in_block
from s2and.corrections.seed_merging import (
    filter_generated_seed_negatives,
    merge_seed_labels,
    pack_seed_components,
    restore_seed_membership,
    seed_disallow_adjacency,
)
from s2and.data import ANDData
from s2and.feature_port import RESIDUAL_FEATURIZER, build_rust_featurizer_from_arrow_dataset
from s2and.incremental_linking.attach_reuse import AttachReuse, attachment_dataset_key
from s2and.incremental_linking.completion_metadata import (
    name_tuples_for_incremental_rules as _name_tuples_for_incremental_rules,
)
from s2and.incremental_linking.feature_block import (
    normalize_cluster_seed_disallow_pairs,
    temporary_cluster_seed_sidecars,
)
from s2and.incremental_linking.policy import (
    clusterer_uses_embedding_features,
    require_rust_featurizer_name_counts_binding_for_clusterer,
    resolve_load_name_counts_policy,
)
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer as _require_arrow_name_counts_index_for_clusterer,
)
from s2and.model_pairwise import FastCluster, clone_cluster_model, intify
from s2and.orcid_prefix_counts import (
    DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256,
    ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD,
    load_packaged_orcid_prefix_counts,
)
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context, load_s2and_rust_extension, stage_uses_rust
from s2and.subblocking import (
    GraphSubblockingConfig,
    _resolve_graph_subblocking_config,
    _resolved_orcid_prefix_counts,
    subblocking_name_parts,
)
from s2and.supervision import effective_dataset_disallows
from s2and.text import canonicalize_name_parts, normalize_orcid
from s2and.thread_config import resolve_n_jobs

from ._shared import (
    _ARROW_ATTACH_BLOCK_STATE,
    _ARROW_ATTACH_NAME_CHECK_EXEMPT,
    _ARROW_BULK_DISTANCE_OVERRIDES,
    _clusterer_with_prediction_params,
    _prediction_cluster_model_params,
    logger,
)
from .constraints import (
    _build_signature_index_by_id,
    _normalize_cluster_seeds_require,
    _partial_supervision_with_cluster_seed_disallows,
    _partial_supervision_with_cluster_seed_overrides,
    _profile_exclusion_kwargs,
    _RustFeaturizerPredictDataset,
)
from .distances import _CONCURRENT_DISTANCE_BLOCK_MAX_SIZE, _CONCURRENT_DISTANCE_BLOCKS
from .incremental import _AttachmentBlockSetup, _AttachmentBlockState, _signature_batch_indices_for_request

if TYPE_CHECKING:
    from .clusterer import Clusterer


def _load_arrow_canonical_name_parts(arrow_dataset: Any, signature_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Return canonical (first, middle) for each signature, in the order of ``signature_ids``."""

    if not signature_ids:
        return []
    import pyarrow as pa

    # The last name takes part because canonicalization reads it (D9 joined capitals).
    columns = ["signature_id", "author_first", "author_middle", "author_last"]
    requested = set(signature_ids)
    # Whole oversized blocks come here. A Python set filter avoids one Arrow hash build per batch,
    # and many records share raw name strings, so each distinct triple is canonicalized once.
    canonical: dict[tuple[str | None, str | None, str | None], tuple[str, str]] = {}
    names: dict[str, tuple[str, str]] = {}
    with arrow_dataset.open_file("signatures") as source_file, pa.PythonFile(source_file, mode="r") as source:
        reader = pa.ipc.open_file(source)
        missing_columns = sorted(set(columns).difference(reader.schema.names))
        if missing_columns:
            raise ValueError(f"signatures Arrow is missing required columns for subblock names: {missing_columns}")
        column_indexes = [reader.schema.get_field_index(column) for column in columns]
        for batch_index in _signature_batch_indices_for_request(arrow_dataset, signature_ids):
            batch = reader.get_batch(batch_index)
            for signature_id, first, middle, last in zip(
                *(batch.column(index).to_pylist() for index in column_indexes), strict=True
            ):
                if signature_id not in requested:
                    continue
                if signature_id in names:
                    raise ValueError(f"signatures Arrow contains duplicate signature_id: {signature_id!r}")
                parts = canonical.get((first, middle, last))
                if parts is None:
                    canonical_parts = canonicalize_name_parts(first, middle, last)
                    parts = canonical[(first, middle, last)] = (canonical_parts.first, canonical_parts.middle)
                names[signature_id] = parts
    missing_signature_ids = [signature_id for signature_id in signature_ids if signature_id not in names]
    if missing_signature_ids:
        raise ValueError(f"Arrow signatures are missing subblocking signature ids: {missing_signature_ids[:10]}")
    return [names[signature_id] for signature_id in signature_ids]


def _subblocking_firsts(
    signature_ids: Sequence[Any],
    name_parts: Sequence[tuple[str, str]],
    maximum_size: int,
) -> dict[Any, str]:
    """Map each signature of one subblocked block to the first name of its subblock key."""

    key_names = subblocking_name_parts(name_parts, maximum_size)
    return {signature_id: first for signature_id, (first, _middle) in zip(signature_ids, key_names, strict=True)}


def _group_subblocking_firsts(
    groups: Mapping[str, Sequence[Any]],
    original_blocks: Mapping[str, Sequence[Any]],
    maximum_size: int,
    load_name_parts: Callable[[Sequence[Any]], Sequence[tuple[str, str]]],
    known_firsts: Mapping[str, str] | None = None,
) -> dict[Any, str]:
    """Return the subblock-key first name of the first member of each nonempty group.

    The compound key rule counts names over the whole original block, so a subblocked block loads
    the names of all its members. A block at or below ``maximum_size`` is not subblocked and keeps
    its canonical first names.
    """

    representatives = {members[0] for members in groups.values() if members}
    firsts: dict[Any, str] = {}
    for block_signatures in original_blocks.values():
        local = [sid for sid in block_signatures if sid in representatives]
        if not local:
            continue
        if known_firsts is not None and all(sid in known_firsts for sid in local):
            firsts.update((sid, known_firsts[sid]) for sid in local)
            continue
        if len(block_signatures) > maximum_size:
            block_firsts = _subblocking_firsts(block_signatures, load_name_parts(block_signatures), maximum_size)
            firsts.update((sid, block_firsts[sid]) for sid in local)
        else:
            firsts.update((sid, first) for sid, (first, _middle) in zip(local, load_name_parts(local), strict=True))
    return firsts


def _join_arrow_subblock_clusters(
    clusters: dict[str, list[str]],
    oversized_blocks: Mapping[str, Sequence[str]],
    subblocked: Mapping[str, Sequence[str]],
    arrow_dataset: ArrowDataset,
    arrow_lease: Any,
    *,
    num_threads: int,
    forbidden: dict[str, set[str]],
    profile_exclusions: Mapping[str, Set[str]],
    seed_groups: Mapping[str, object] | None,
    name_tuples: Set[tuple[str, str]] = frozenset(),
    single_letter_subblocks: Set[str] = frozenset(),
) -> dict[str, list[str]]:
    """Load ORCID anchors from oversized blocks and join with original-block scope."""
    rows = arrow_dataset.native.signature_orcids(
        [signature for members in oversized_blocks.values() for signature in members], num_threads
    )
    if rows is None:
        return clusters
    signature_ids, positions, values = rows
    orcids = {
        signature_ids[position]: orcid
        for position, value in zip(positions, values, strict=True)
        if (orcid := normalize_orcid(value)) is not None
    }
    if len(orcids) < 4:
        return clusters
    origins = {s: key for key, members in oversized_blocks.items() for s in members if s in orcids}
    subblocks = {s: key for key, members in subblocked.items() for s in members if s in orcids}
    for anchor, excluded in profile_exclusions.items():
        forbidden.setdefault(anchor, set()).update(excluded)
    return join_orcids_across_subblocks(
        clusters,
        orcids,
        origins,
        subblocks,
        partial(_load_arrow_canonical_name_parts, arrow_lease),
        forbidden=forbidden,
        seed_groups=seed_groups,
        name_tuples=name_tuples,
        single_letter_subblocks=single_letter_subblocks,
    )


def _allocate_subblock_key(
    base_key: str,
    *,
    original_block_keys: Collection[str],
    emitted_block_keys: Collection[str],
) -> str:
    """Return a deterministic generated key that cannot overwrite a block.

    Args:
        base_key: Preferred generated subblock key.
        original_block_keys: Keys supplied by the caller.
        emitted_block_keys: Generated or copied keys already emitted.

    Returns:
        The preferred key, or its first available collision-suffixed variant.
    """

    allocated_key = base_key
    collision_index = 1
    while allocated_key in original_block_keys or allocated_key in emitted_block_keys:
        allocated_key = f"{base_key}|collision={collision_index:04d}"
        collision_index += 1
    return allocated_key


def _subblocked_prediction_scopes(
    predicted_clusters: dict[str, list[str]],
    original_blocks: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """Keep each sequential attachment pass inside its original input block."""
    origins = {sid: block for block, members in original_blocks.items() for sid in members}
    scoped: dict[str, dict[str, list[str]]] = {block: {} for block in original_blocks}
    for cluster_id, members in predicted_clusters.items():
        if members:
            origin = origins[members[0]]
            if any(origins[sid] != origin for sid in members):
                raise RuntimeError("Subblock prediction crossed an original input-block boundary")
            scoped[origin][cluster_id] = members
    return origins, scoped


def _combine_subblocked_predictions(scopes: Mapping[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    """Combine independent predictions without overwriting reused cluster IDs."""
    result: dict[str, list[str]] = {}
    reserved = {key for clusters in scopes.values() for key in clusters}
    next_suffix: dict[str, int] = {}
    for clusters in scopes.values():
        for key, members in clusters.items():
            if key in result:
                base_key = key
                index = next_suffix.get(base_key, 1)
                key = f"{base_key}|collision={index:04d}"
                while key in reserved or key in result:
                    index += 1
                    key = f"{base_key}|collision={index:04d}"
                next_suffix[base_key] = index + 1
            result[key] = members
    return result


def _predict_from_prepared_rust_featurizer(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    rust_featurizer: object,
    *,
    proxy_dataset: _RustFeaturizerPredictDataset,
    dists: dict[str, np.ndarray] | None,
    cluster_model_params: dict[str, Any] | None,
    partial_supervision: dict[tuple[str, str], int | float],
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    explicit_cluster_seeds_require: dict[str, int | str] | None,
    resolved_cluster_seeds_disallow: set[tuple[str, str]],
    prediction_state: PredictionState | None = None,
    residual_reuse: _AttachmentBlockState | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """Predict with request metadata already exported from a Rust featurizer."""

    if any(len(signatures) > 1 for signatures in block_dict.values()):
        self = _clusterer_with_prediction_params(self, cluster_model_params)
    cluster_model_params = _prediction_cluster_model_params(self, cluster_model_params)
    if prediction_state is None:
        prediction_state = PredictionState()

    built_dists = dists is None
    if built_dists:
        require_rust_featurizer_name_counts_binding_for_clusterer(
            self,
            rust_featurizer,
            context="Clusterer.predict_from_rust_featurizer",
        )
    if built_dists:
        pred_clusters: defaultdict[str, list[str]] = defaultdict(list)
        disallow_adjacency = seed_disallow_adjacency(proxy_dataset.cluster_seeds_disallow, partial_supervision)
        corrections = corrections_from_rust_featurizer(
            self,
            block_dict,
            rust_featurizer,
            prediction_state.small_block_signatures,
        )
        effective_cluster_model_params = cluster_model_params
        if isinstance(self.cluster_model, FastCluster):
            fastcluster_params: dict[str, Any] = dict(cluster_model_params or {})
            fastcluster_params.setdefault("preserve_input", False)
            effective_cluster_model_params = fastcluster_params
        make_dists_seconds = 0.0
        cluster_seconds = 0.0
        make_dists_telemetry: dict[str, int | float | str] = {}
        signature_index_by_id = (
            _build_signature_index_by_id(rust_featurizer)
            if any(len(signatures) > 1 for signatures in block_dict.values())
            else None
        )
        # Linkage releases the GIL, so one block clusters on a worker while the
        # next block's distances build. At most two matrices are alive, and two
        # only while the clustering one is under a tenth of the RAM budget.
        # Larger blocks go first, so the longest linkages overlap the most
        # work and the last, exposed linkage is the smallest. Blocks cluster
        # independently, and results are collected in block order.
        overlap_limit_bytes = memory_budget.resolve_total_ram_bytes(total_ram_bytes)[0] // 10
        block_results: dict[str, Future[tuple[np.ndarray | list[int], float]] | None] = {}
        blocks_largest_first = sorted(block_dict.items(), key=lambda item: len(item[1]), reverse=True)
        pending: Future[tuple[np.ndarray | list[int], float]] | None = None
        pending_bytes = 0

        def timed_cluster(
            block_key: str, signatures: list[str], dist_matrix: np.ndarray | None
        ) -> tuple[np.ndarray | list[int], float]:
            start = time.perf_counter()
            labels = self._cluster_one_block_with_logging(
                signatures,
                dist_matrix,
                effective_cluster_model_params,
                cast(Any, proxy_dataset),
                disallow_adjacency,
                block_key=block_key,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                corrections=corrections,
                **_profile_exclusion_kwargs(prediction_state),
            )
            return labels, time.perf_counter() - start

        def block_distances(
            block_key: str, signatures: list[str], block_state: PredictionState
        ) -> tuple[dict[str, np.ndarray], float]:
            assert signature_index_by_id is not None
            start = time.perf_counter()
            block_partial_supervision = _partial_supervision_with_cluster_seed_overrides(
                signatures,
                partial_supervision,
                cluster_seeds_require=explicit_cluster_seeds_require,
                cluster_seeds_disallow=resolved_cluster_seeds_disallow,
                dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            )
            block_dists = self._make_distance_matrices_from_verified_rust_featurizer(
                {block_key: signatures},
                rust_featurizer,
                partial_supervision=block_partial_supervision,
                incremental_dont_use_cluster_seeds=(
                    incremental_dont_use_cluster_seeds or explicit_cluster_seeds_require is not None
                ),
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                signature_index_by_id=signature_index_by_id,
                prediction_state=block_state,
                residual_reuse=residual_reuse,
            )
            return block_dists, time.perf_counter() - start

        # A small block fills a few tiles, too few for the pool. Fill those
        # blocks together: each call joins the same native pool, so the
        # request keeps its thread count, and each block gets its own
        # telemetry, merged below in block order. Only a few blocks ahead
        # of the loop are in flight, so their matrices add little memory.
        small_block_states: dict[str, PredictionState] = {}
        small_block_futures: dict[str, Future[tuple[dict[str, np.ndarray], float]]] = {}
        small_blocks = [
            (block_key, signatures)
            for block_key, signatures in blocks_largest_first
            if 1 < len(signatures) <= _CONCURRENT_DISTANCE_BLOCK_MAX_SIZE
        ]
        small_block_positions = {block_key: position for position, (block_key, _) in enumerate(small_blocks)}
        distance_workers = min(len(small_blocks), _CONCURRENT_DISTANCE_BLOCKS)
        submitted_small_blocks = 0
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2and-cluster") as cluster_pool,
            ThreadPoolExecutor(
                max_workers=max(1, distance_workers), thread_name_prefix="s2and-block-distances"
            ) as distance_pool,
        ):
            for block_key, signatures in blocks_largest_first:
                if len(signatures) <= 1:
                    block_results[block_key] = None
                    continue
                if pending is not None and pending_bytes > overlap_limit_bytes:
                    pending.result()
                small_position = small_block_positions.get(block_key)
                if distance_workers > 1 and small_position is not None:
                    end = min(len(small_blocks), small_position + 2 * distance_workers)
                    while submitted_small_blocks < end:
                        small_key, small_signatures = small_blocks[submitted_small_blocks]
                        small_block_states[small_key] = PredictionState()
                        small_block_futures[small_key] = distance_pool.submit(
                            block_distances, small_key, small_signatures, small_block_states[small_key]
                        )
                        submitted_small_blocks += 1
                    block_dists, block_seconds = small_block_futures.pop(block_key).result()
                    block_state = small_block_states.pop(block_key)
                else:
                    block_dists, block_seconds = block_distances(block_key, signatures, prediction_state)
                    block_state = prediction_state
                make_dists_seconds += block_seconds
                block_make_dists_telemetry = dict(block_state.telemetry.get("rust_featurizer_make_dists", {}))
                if block_state is not prediction_state:
                    prediction_state.telemetry["rust_featurizer_make_dists"] = block_make_dists_telemetry
                for key, value in block_make_dists_telemetry.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        make_dists_telemetry[key] = float(make_dists_telemetry.get(key, 0.0)) + float(value)
                    elif key not in make_dists_telemetry:
                        make_dists_telemetry[key] = value
                dist_matrix = block_dists[block_key]
                # Drop this frame's references once the worker holds the matrix, so
                # it is freed when clustering ends.
                del block_dists
                if pending is not None:
                    pending.result()
                pending = cluster_pool.submit(timed_cluster, block_key, signatures, dist_matrix)
                pending_bytes = 0 if dist_matrix is None else int(dist_matrix.nbytes)
                block_results[block_key] = pending
                del dist_matrix
        for block_key, signatures in block_dict.items():
            outcome = block_results[block_key]
            if outcome is None:
                if signatures:
                    pred_clusters[block_key + "_0"].append(signatures[0])
                continue
            labels, block_cluster_seconds = outcome.result()
            cluster_seconds += block_cluster_seconds
            for signature, label in zip(signatures, labels, strict=True):
                pred_clusters[block_key + "_" + str(label)].append(signature)
        make_dists_telemetry["total_seconds"] = float(make_dists_seconds)
        make_dists_telemetry["block_count"] = int(len(block_dict))
        make_dists_telemetry["pair_count"] = int(
            sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())
        )
        out_dists = None
    else:
        make_dists_seconds = 0.0
        cluster_start = time.perf_counter()
        pred_clusters, out_dists = self.predict_helper(
            block_dict,
            cast(Any, proxy_dataset),
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            use_s2_clusters=False,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
        )
        cluster_seconds = time.perf_counter() - cluster_start
        make_dists_telemetry = {}
    telemetry = {
        "make_dists_seconds": float(make_dists_seconds),
        "cluster_from_dists_seconds": float(cluster_seconds),
        "block_count": int(len(block_dict)),
        "pair_count": int(sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())),
        **{f"make_dists_{key}": value for key, value in make_dists_telemetry.items()},
    }
    prediction_state.telemetry["rust_featurizer_predict"] = telemetry
    logger.info(
        "Telemetry stage: stage=rust_featurizer_predict make_dists_seconds=%.3f "
        "cluster_from_dists_seconds=%.3f blocks=%d pairs=%d",
        telemetry["make_dists_seconds"],
        telemetry["cluster_from_dists_seconds"],
        telemetry["block_count"],
        telemetry["pair_count"],
    )
    return dict(pred_clusters), out_dists


def _make_subblocks_with_telemetry_arrow_rust(
    arrow_dataset: Any,
    signature_ids: list[str],
    *,
    maximum_size: int,
    first_k_letter_counts_sorted: Mapping[str, Mapping[str, int]],
    graph_subblocking_config: GraphSubblockingConfig,
    graph_subblocking_random_seed: int,
    use_orcid_subblocking: bool,
    num_threads: int,
    firsts_for_seed_ids: set[str],
) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, list[str]]]:
    """Run subblocking on the request's pool and return first names with the plan."""
    counts = _resolved_orcid_prefix_counts(first_k_letter_counts_sorted)
    native = load_s2and_rust_extension()
    subblocks, telemetry, dead_ends = native.make_subblocks_with_telemetry_arrow_native_graph(
        arrow_dataset.native,
        signature_ids,
        maximum_size,
        {prefix: dict(nested) for prefix, nested in counts.items()},
        graph_subblocking_config,
        graph_subblocking_random_seed,
        use_orcid_subblocking,
        num_threads=num_threads,
        firsts_for_seed_ids=firsts_for_seed_ids,
    )
    return subblocks, telemetry, dict(dead_ends)


def _build_arrow_subblocked_block_dict(
    self: Clusterer,
    block_dict: Mapping[str, Sequence[str]],
    arrow_dataset: Any,
    *,
    batching_threshold: int,
    cluster_seeds_require: Mapping[str, int | str],
    prediction_state: PredictionState | None = None,
    firsts_out: dict[str, str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Partition oversized blocks without splitting required seed components.

    Returns:
        The subblocks, and the key first name of the dead end of each record that the
        native planner reports in a no-middle dead-end bucket.
    """

    if prediction_state is None:
        prediction_state = PredictionState()

    subblocked: dict[str, list[str]] = {}
    no_middle_dead_ends: dict[str, str] = {}
    telemetry_by_block: dict[str, dict[str, Any]] = {}
    normalized_blocks = {
        block_key: [str(signature_id) for signature_id in block_dict[block_key]] for block_key in sorted(block_dict)
    }
    normalized_seed_map = {
        str(signature_id): str(component_id) for signature_id, component_id in cluster_seeds_require.items()
    }
    seed_components_by_block: dict[str, dict[str, list[str]]] = {}
    for block_key, block_signatures in normalized_blocks.items():
        signature_counts: defaultdict[str, int] = defaultdict(int)
        for signature_id in block_signatures:
            signature_counts[signature_id] += 1
        duplicate_input_ids = sorted(signature_id for signature_id, count in signature_counts.items() if count > 1)
        if duplicate_input_ids:
            raise ValueError(
                "Arrow subblocking requires unique signature IDs within each input block "
                f"(block={block_key} duplicates={duplicate_input_ids})"
            )

        components: defaultdict[str, list[str]] = defaultdict(list)
        for signature_id in block_signatures:
            component_id = normalized_seed_map.get(signature_id)
            if component_id is not None:
                components[component_id].append(signature_id)
        for component_id, component_signatures in components.items():
            if len(component_signatures) > batching_threshold:
                raise ValueError(
                    "cluster_seeds_require component exceeds batching_threshold before Arrow subblocking "
                    f"(block={block_key} component={component_id} "
                    f"size={len(component_signatures)} threshold={batching_threshold})"
                )
        seed_components_by_block[block_key] = dict(components)

    for block_key in sorted(block_dict):
        block_signatures = normalized_blocks[block_key]
        if len(block_signatures) <= batching_threshold:
            subblocked[block_key] = block_signatures
            continue

        seed_components = seed_components_by_block[block_key]
        seed_ids = {sid for members in seed_components.values() for sid in members}
        native_subblocks, telemetry, no_middle_buckets = _make_subblocks_with_telemetry_arrow_rust(
            arrow_dataset,
            block_signatures,
            maximum_size=batching_threshold,
            first_k_letter_counts_sorted=self._subblocking_prefix_counts(),
            graph_subblocking_config=self._subblocking_graph_config(),
            graph_subblocking_random_seed=int(getattr(self, "random_state", 0) or 0),
            use_orcid_subblocking=True,
            num_threads=resolve_n_jobs(self.n_jobs),
            firsts_for_seed_ids=seed_ids,
        )
        # Seed packing can promote a different member to the head of a subblock.
        firsts = telemetry.pop("subblock_firsts", None)
        for first, bucket in no_middle_buckets.items():
            no_middle_dead_ends.update(dict.fromkeys(bucket, first))
        normalized_native_subblocks: dict[str, list[str]] = {}
        output_locations: defaultdict[str, list[str]] = defaultdict(list)
        for subblock_key in sorted(native_subblocks):
            subblock_signatures = [str(signature_id) for signature_id in native_subblocks[subblock_key]]
            if len(subblock_signatures) > batching_threshold:
                raise RuntimeError(
                    "Rust Arrow subblocking exceeded batching_threshold "
                    f"(block={block_key} subblock={subblock_key} "
                    f"size={len(subblock_signatures)} threshold={batching_threshold})"
                )
            normalized_native_subblocks[str(subblock_key)] = subblock_signatures
            for signature_id in subblock_signatures:
                output_locations[signature_id].append(str(subblock_key))

        expected_signature_ids = set(block_signatures)
        missing_signature_ids = sorted(expected_signature_ids - set(output_locations))
        unexpected_signature_ids = sorted(set(output_locations) - expected_signature_ids)
        duplicate_signature_ids = sorted(
            signature_id for signature_id, locations in output_locations.items() if len(locations) > 1
        )
        if missing_signature_ids or unexpected_signature_ids or duplicate_signature_ids:
            raise RuntimeError(
                "Rust Arrow subblocking must return every input signature exactly once "
                f"(block={block_key} missing={missing_signature_ids} "
                f"unexpected={unexpected_signature_ids} duplicates={duplicate_signature_ids})"
            )

        repacked_seed_component_count = sum(
            len({output_locations[signature_id][0] for signature_id in component_signatures}) > 1
            for component_signatures in seed_components.values()
        )
        full_name_subblocks: set[str] = set()
        if repacked_seed_component_count:
            if firsts is None:
                firsts = _subblocking_firsts(
                    block_signatures,
                    _load_arrow_canonical_name_parts(arrow_dataset, block_signatures),
                    batching_threshold,
                )
            full_name_subblocks = {
                key for key, members in normalized_native_subblocks.items() if members and len(firsts[members[0]]) > 1
            }
        repaired_subblock_count = 0
        repaired = pack_seed_components(
            normalized_native_subblocks,
            seed_components,
            batching_threshold,
            full_name_subblocks=full_name_subblocks,
        )
        for subblock_key, repaired_parts in repaired.items():
            for part_index, repaired_part in enumerate(repaired_parts):
                repaired_key_base = f"{block_key}|subblock={subblock_key}"
                if len(repaired_parts) > 1:
                    repaired_key_base += f"|repair_part={part_index:04d}"
                repaired_key = _allocate_subblock_key(
                    repaired_key_base,
                    original_block_keys=normalized_blocks.keys(),
                    emitted_block_keys=subblocked.keys(),
                )
                subblocked[repaired_key] = repaired_part
                if firsts_out is not None and firsts is not None:
                    firsts_out[repaired_part[0]] = firsts[repaired_part[0]]
                repaired_subblock_count += 1

        block_telemetry = dict(telemetry)
        block_telemetry.update(
            {
                "seed_component_count": int(len(seed_components)),
                "seed_signature_count": len(seed_ids),
                "seed_components_repacked": int(repacked_seed_component_count),
                "repaired_final_subblock_count": int(repaired_subblock_count),
            }
        )
        telemetry_by_block[block_key] = block_telemetry

    prediction_state.telemetry["rust_arrow_subblocking"] = {
        "enabled": 1,
        "maximum_size": int(batching_threshold),
        "input_block_count": int(len(block_dict)),
        "oversized_block_count": int(len(telemetry_by_block)),
        "blocks": telemetry_by_block,
    }
    return subblocked, no_middle_dead_ends


@dataclass
class _AttachmentSubblocks:
    subblocked: dict[str, list[str]]
    multiple_letter: dict[str, list[str]]
    single_letter: dict[str, list[str]]
    alert: bool
    no_middle_dead_ends: dict[str, str]
    featurizer_signature_ids: list[str]
    telemetry: dict[str, Any]


class _AttachmentBlockCache:
    """Hold at most one original block's attachment state across passes."""

    def __init__(
        self,
        arrow_dataset: ArrowDataset,
        original_blocks: Mapping[str, Sequence[str]],
        name_tuples: frozenset[tuple[str, str]],
        num_threads: int,
        attach_reuse: AttachReuse | None,
    ) -> None:
        self.arrow_dataset = arrow_dataset
        self.original_blocks = original_blocks
        self.name_tuples = name_tuples
        self.num_threads = num_threads
        self.attach_reuse = attach_reuse
        self.current: tuple[str, _AttachmentBlockState] | None = None

    def get(self, origin: str) -> _AttachmentBlockState:
        if self.current is not None and self.current[0] == origin:
            return self.current[1]
        self.close(background=False)
        members = [str(signature_id) for signature_id in self.original_blocks[origin]]
        state = _AttachmentBlockState(self.arrow_dataset, members, self.name_tuples, self.num_threads)
        reuse = self.attach_reuse
        if reuse is not None:
            key = (attachment_dataset_key(self.arrow_dataset), tuple(members), self.name_tuples)
            setup = reuse.blocks.get(key)
            if setup is None and reuse.blocks.fits(len(members)):
                setup = _AttachmentBlockSetup()
                reuse.blocks.store(key, setup, len(members))
            if setup is not None:
                state.setup = setup
                state.attach_reuse = reuse
        self.current = (origin, state)
        return state

    def close(self, *, background: bool = True) -> None:
        if self.current is not None:
            self.current[1].release(background=background)
            self.current = None


def _predict_subblocked_single_letter_groups_from_arrow(
    self: Clusterer,
    block_dict_single_letter: Mapping[str, Sequence[str]],
    *,
    pred_clusters: dict[str, list[str]],
    arrow_dataset: ArrowDataset,
    batching_threshold: int,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    name_tuples: frozenset[tuple[str, str]],
    cluster_seeds_disallow: set[tuple[str, str]],
    cluster_seeds_require: Mapping[str, int | str],
    original_blocks: Mapping[str, Sequence[str]],
    attach_reuse: AttachReuse | None = None,
    cache_residuals: bool = False,
    attachment_cache: _AttachmentBlockCache | None = None,
) -> dict[str, list[str]]:
    """Attach initial-only groups using already normalized caller supervision."""

    if not block_dict_single_letter:
        return pred_clusters

    origins, scoped_clusters = _subblocked_prediction_scopes(pred_clusters, original_blocks)
    scoped_seeds: dict[str, dict[str, int | str]] = defaultdict(dict)
    for sid, component in cluster_seeds_require.items():
        if sid in origins:
            scoped_seeds[origins[sid]][sid] = component
    disallow_adjacency = seed_disallow_adjacency(cluster_seeds_disallow, partial_supervision)
    with ExitStack() as stack:
        if attachment_cache is None:
            attachment_cache = _AttachmentBlockCache(
                arrow_dataset, original_blocks, name_tuples, self.n_jobs, attach_reuse
            )
            stack.callback(attachment_cache.close)
        for block_key in sorted(block_dict_single_letter):
            group_signature_ids = [str(signature_id) for signature_id in block_dict_single_letter[block_key]]
            origin = origins[group_signature_ids[0]]
            state = attachment_cache.get(origin)
            if cache_residuals:
                state.residual_group = tuple(group_signature_ids)
            synthetic_seed_map = restore_seed_membership(
                scoped_clusters[origin], scoped_seeds[origin], disallow_adjacency
            )
            # Future initial-only groups are not known to this request's planner.
            # Retain the full disallows above and expose only currently addressable
            # endpoints here; later passes will see those queries as seeds.
            # Known ids are the seeds and this group. Test membership in both
            # rather than copy every seed into a set on each pass.
            group_ids = set(group_signature_ids)
            current_disallows = {
                (left, right)
                for left, right in cluster_seeds_disallow
                if (left in synthetic_seed_map or left in group_ids)
                and (right in synthetic_seed_map or right in group_ids)
            }
            current_supervision = {
                (left, right): distance
                for (left, right), distance in partial_supervision.items()
                if (left in synthetic_seed_map or left in group_ids)
                and (right in synthetic_seed_map or right in group_ids)
            }
            current_disallows, current_supervision = filter_generated_seed_negatives(
                synthetic_seed_map, current_disallows, current_supervision
            )
            bulk_context_token = _ARROW_BULK_DISTANCE_OVERRIDES.set(True)
            name_check_token = _ARROW_ATTACH_NAME_CHECK_EXEMPT.set(frozenset(scoped_seeds[origin]))
            block_state_token = _ARROW_ATTACH_BLOCK_STATE.set(state)
            try:
                incremental_result = self.predict_incremental_from_arrow(
                    group_signature_ids,
                    arrow_dataset,
                    prevent_new_incompatibilities=True,
                    batching_threshold=batching_threshold,
                    partial_supervision=current_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                    name_tuples=name_tuples,
                    cluster_seeds_require=synthetic_seed_map,
                    cluster_seeds_disallow=current_disallows,
                    altered_cluster_signatures=[],
                )
            finally:
                _ARROW_ATTACH_BLOCK_STATE.reset(block_state_token)
                _ARROW_ATTACH_NAME_CHECK_EXEMPT.reset(name_check_token)
                _ARROW_BULK_DISTANCE_OVERRIDES.reset(bulk_context_token)
            clusters_payload = incremental_result.get("clusters")
            if not isinstance(clusters_payload, dict):
                raise RuntimeError(
                    "predict_incremental_from_arrow returned invalid clusters payload; expected dict "
                    f"got {type(clusters_payload).__name__}"
                )
            pred_clusters_intermediate: dict[str, list[str]] = {}
            for cluster_id, signatures in clusters_payload.items():
                if not isinstance(signatures, list):
                    raise RuntimeError(
                        "predict_incremental_from_arrow returned invalid cluster member payload; expected list "
                        f"for cluster_id={cluster_id!r}, got {type(signatures).__name__}"
                    )
                cluster_key = str(cluster_id)
                normalized_signatures = [str(signature_id) for signature_id in signatures]
                pred_clusters_intermediate[cluster_key] = normalized_signatures
            scoped_clusters[origin] = pred_clusters_intermediate
    return _combine_subblocked_predictions(scoped_clusters)


def _predict_from_arrow(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    arrow_dataset: ArrowDataset,
    arrow_lease: Any,
    cluster_seed_sidecars: Mapping[str, str],
    *,
    dists: dict[str, np.ndarray] | None,
    cluster_model_params: dict[str, Any] | None,
    effective_partial_supervision: dict[tuple[str, str], int | float],
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    batching_threshold: int | None,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None,
    cluster_seed_disallows: set[tuple[str, str]],
    signature_ids: list[str],
    needs_subblocking: bool,
    prediction_cluster_seeds_require: dict[str, int | str] | None,
    prediction_state: PredictionState | None = None,
    bulk_clusters: Callable[[Mapping[str, list[str]]], dict[str, list[str]]] | None = None,
    attach_reuse: AttachReuse | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """Run native subblocking and prediction from one request-bound Arrow view.

    ``bulk_clusters`` replaces only the bulk clustering of full-name subblocks; see
    ``_predict_from_arrow_request``.
    """

    if prediction_state is None:
        prediction_state = PredictionState()
    if bulk_clusters is not None and not needs_subblocking:
        raise ValueError("bulk_clusters applies only to subblocked Arrow prediction")
    # Before subblocking: a subblock, a claimed profile or a residual is not its surname block.
    # Only the year-gap split reads them, and it runs only with a close-pair bonus.
    if year_gap_split_enabled(self):
        prediction_state.small_block_signatures = arrow_small_block_signatures(arrow_lease, block_dict)

    arrow_subblocking_seconds = 0.0
    block_dict_multiple_letter: dict[str, list[str]] | None = None
    block_dict_single_letter: dict[str, list[str]] | None = None
    subblocking_alert_flag: bool | None = None
    featurizer_signature_ids = signature_ids
    prediction_name_tuples = name_tuples
    effective_cluster_seeds_require: dict[str, int | str] = {}
    active_cluster_seeds_require: dict[str, int | str] = {}
    no_middle_dead_ends: dict[str, str] = {}
    if needs_subblocking:
        assert batching_threshold is not None
        block_dict = {key: [str(signature_id) for signature_id in members] for key, members in block_dict.items()}
        if self.use_default_constraints_as_supervision and not incremental_dont_use_cluster_seeds:
            if prediction_cluster_seeds_require is not None:
                effective_cluster_seeds_require.update(prediction_cluster_seeds_require)
        active_signature_id_set = set(signature_ids)
        active_cluster_seeds_require = {
            signature_id: component_id
            for signature_id, component_id in effective_cluster_seeds_require.items()
            if signature_id in active_signature_id_set
        }
        atomic_seed_signature_ids: set[str] = set()
        for block_signatures in block_dict.values():
            component_signatures: defaultdict[str, list[str]] = defaultdict(list)
            for signature_id in block_signatures:
                normalized_signature_id = str(signature_id)
                component_id = active_cluster_seeds_require.get(normalized_signature_id)
                if component_id is not None:
                    component_signatures[str(component_id)].append(normalized_signature_id)
            for signatures in component_signatures.values():
                if len(signatures) > 1:
                    atomic_seed_signature_ids.update(signatures)
        atomic_cluster_seeds_require = {
            signature_id: component_id
            for signature_id, component_id in active_cluster_seeds_require.items()
            if signature_id in atomic_seed_signature_ids
        }
        subblocking_start = time.perf_counter()
        subblocks = None
        if attach_reuse is not None:
            feature_contract = getattr(self, "feature_contract", None)
            prefix_digest = (
                feature_contract.get(ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD)
                if isinstance(feature_contract, Mapping)
                else None
            )
            subblock_key = (
                attachment_dataset_key(arrow_dataset),
                tuple((key, tuple(value)) for key, value in sorted(block_dict.items())),
                batching_threshold,
                tuple(sorted(atomic_cluster_seeds_require.items())),
                prefix_digest or DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256,
                self._subblocking_graph_config(),
                int(self.random_state or 0),
            )
            subblocks = attach_reuse.subblocks.get(subblock_key)
        if subblocks is None:
            native_firsts: dict[str, str] = {}
            block_dict_subblocked, no_middle_dead_ends = self._build_arrow_subblocked_block_dict(
                block_dict,
                arrow_lease,
                batching_threshold=batching_threshold,
                cluster_seeds_require=atomic_cluster_seeds_require,
                prediction_state=prediction_state,
                firsts_out=native_firsts,
            )
            subblocking_firsts = _group_subblocking_firsts(
                block_dict_subblocked,
                block_dict,
                batching_threshold,
                lambda signature_ids: _load_arrow_canonical_name_parts(arrow_lease, signature_ids),
                native_firsts,
            )
            (
                block_dict_multiple_letter,
                block_dict_single_letter,
                subblocking_alert_flag,
            ) = self._partition_subblocked_first_name_groups(
                block_dict_subblocked,
                subblocking_firsts,
                original_blocks=block_dict,
            )
            featurizer_signature_ids = list(
                dict.fromkeys(
                    str(signature_id)
                    for block_key in sorted(block_dict_multiple_letter)
                    for signature_id in block_dict_multiple_letter[block_key]
                )
            )
            if attach_reuse is not None and attach_reuse.subblocks.fits(len(signature_ids)):
                attach_reuse.subblocks.store(
                    subblock_key,
                    _AttachmentSubblocks(
                        block_dict_subblocked,
                        block_dict_multiple_letter,
                        block_dict_single_letter,
                        subblocking_alert_flag,
                        no_middle_dead_ends,
                        featurizer_signature_ids,
                        deepcopy(prediction_state.telemetry["rust_arrow_subblocking"]),
                    ),
                    len(signature_ids),
                )
        else:
            block_dict_subblocked = subblocks.subblocked
            block_dict_multiple_letter = subblocks.multiple_letter
            block_dict_single_letter = subblocks.single_letter
            subblocking_alert_flag = subblocks.alert
            no_middle_dead_ends = subblocks.no_middle_dead_ends
            featurizer_signature_ids = subblocks.featurizer_signature_ids
            prediction_state.telemetry["rust_arrow_subblocking"] = deepcopy(subblocks.telemetry)
        arrow_subblocking_seconds = time.perf_counter() - subblocking_start
        # The sequential initial-only passes re-enter the public incremental
        # API. Freeze canonical aliases once so each pass can reuse them.
        prediction_name_tuples = frozenset(_name_tuples_for_incremental_rules(name_tuples))
    else:
        prediction_state.telemetry["rust_arrow_subblocking"] = {
            "enabled": 0,
            "maximum_size": batching_threshold,
            "oversized_block_count": 0,
        }
    featurizer_start = time.perf_counter()
    cluster_seeds_path = (
        Path(cluster_seed_sidecars["cluster_seeds"]) if "cluster_seeds" in cluster_seed_sidecars else None
    )
    cluster_seed_disallows_path = (
        Path(cluster_seed_sidecars["cluster_seed_disallows"])
        if "cluster_seed_disallows" in cluster_seed_sidecars
        else None
    )
    attachment_state = _ARROW_ATTACH_BLOCK_STATE.get()
    if bulk_clusters is not None:
        # The caller supplies the bulk clusters, so no bulk distances are computed.
        rust_featurizer = None
    elif (
        attachment_state is not None
        and attachment_state.arrow_dataset is arrow_dataset
        and attachment_state.name_tuples == prediction_name_tuples
        and attachment_state.signature_ids.issuperset(featurizer_signature_ids)
    ):
        # Residual clustering inside an attachment pass covers a few hundred
        # signatures, but a fresh build rescans the block's Arrow tables to find
        # them. A view of the block featurizer equals that build exactly.
        rust_featurizer = attachment_state.featurizer().view_with_cluster_seed_sidecars(
            [str(signature_id) for signature_id in featurizer_signature_ids],
            None if cluster_seeds_path is None else str(cluster_seeds_path),
            None if cluster_seed_disallows_path is None else str(cluster_seed_disallows_path),
            num_threads=resolve_n_jobs(self.n_jobs),
        )
    elif (
        (held := RESIDUAL_FEATURIZER.get()) is not None
        and held.arrow_dataset is arrow_dataset
        and held.name_tuples == prediction_name_tuples
        and held.signature_ids.issuperset(featurizer_signature_ids)
    ):
        # Residual clustering of an incremental request: the view equals an
        # Arrow build over these signatures with these sidecars.
        rust_featurizer = held.featurizer.view_with_cluster_seed_sidecars(
            [str(signature_id) for signature_id in featurizer_signature_ids],
            None if cluster_seeds_path is None else str(cluster_seeds_path),
            None if cluster_seed_disallows_path is None else str(cluster_seed_disallows_path),
            num_threads=resolve_n_jobs(self.n_jobs),
        )
    else:
        rust_featurizer = build_rust_featurizer_from_arrow_dataset(
            arrow_dataset,
            signature_ids=featurizer_signature_ids,
            name_tuples=prediction_name_tuples,
            preprocess=True,
            num_threads=self.n_jobs,
            cluster_seeds_path=cluster_seeds_path,
            cluster_seed_disallows_path=cluster_seed_disallows_path,
        )
    if rust_featurizer is not None and self.featurizer_info.legacy_paper_years:
        rust_featurizer = cast(Any, rust_featurizer).with_legacy_paper_years()
    arrow_featurizer_seconds = time.perf_counter() - featurizer_start
    predict_start = time.perf_counter()
    if needs_subblocking:
        assert batching_threshold is not None
        assert block_dict_multiple_letter is not None
        assert block_dict_single_letter is not None
        assert subblocking_alert_flag is not None
        assert isinstance(prediction_name_tuples, frozenset)
        if bulk_clusters is None:
            proxy_dataset = _RustFeaturizerPredictDataset(
                cluster_seeds_require=effective_cluster_seeds_require,
                cluster_seeds_disallow=cluster_seed_disallows,
                signatures={},
                repair_cluster_seeds_require=dict(prediction_cluster_seeds_require or {}),
            )
            pred_clusters = self._predict_subblocked_multiple_letter_groups(
                block_dict_multiple_letter,
                alert_flag=subblocking_alert_flag,
                dataset=cast(ANDData, proxy_dataset),
                cluster_model_params=cluster_model_params,
                partial_supervision=effective_partial_supervision,
                use_s2_clusters=False,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                rust_featurizer=rust_featurizer,
                cluster_seeds_disallow=cluster_seed_disallows,
                explicit_cluster_seeds_require=None,
                prediction_state=prediction_state,
            )
            del proxy_dataset
        else:
            pred_clusters = bulk_clusters(block_dict_multiple_letter)
        # Release the bulk featurizer before building any attachment state.
        del rust_featurizer
        attachment_cache = _AttachmentBlockCache(
            arrow_dataset, block_dict, prediction_name_tuples, self.n_jobs, attach_reuse
        )
        with ExitStack() as stack:
            stack.callback(attachment_cache.close)
            attach_from_arrow = partial(
                self._predict_subblocked_single_letter_groups_from_arrow,
                arrow_dataset=arrow_dataset,
                batching_threshold=batching_threshold,
                partial_supervision=effective_partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                name_tuples=prediction_name_tuples,
                cluster_seeds_disallow=cluster_seed_disallows,
                original_blocks=block_dict,
                attachment_cache=attachment_cache,
            )
            pred_clusters = self._reunite_no_middle_dead_ends(
                pred_clusters,
                no_middle_dead_ends=no_middle_dead_ends,
                required_seed_ids=active_cluster_seeds_require.keys(),
                bulk_groups=block_dict_multiple_letter,
                original_blocks=block_dict,
                # No required seed member is marked, so none is visible or queried here. The
                # empty map still turns on the attach name checks.
                attach=lambda queries, seeds: attach_from_arrow(queries, pred_clusters=seeds, cluster_seeds_require={}),
                prediction_state=prediction_state,
            )
            pred_clusters = attach_from_arrow(
                block_dict_single_letter,
                pred_clusters=pred_clusters,
                cluster_seeds_require=active_cluster_seeds_require,
                cache_residuals=attach_reuse is not None,
            )
        if orcid_join_enabled(self) and average_linkage_cut(self.cluster_model, cluster_model_params) is not None:
            oversized = {key: members for key, members in block_dict.items() if len(members) > batching_threshold}
            pred_clusters = _join_arrow_subblock_clusters(
                pred_clusters,
                oversized,
                block_dict_subblocked,
                arrow_dataset,
                arrow_lease,
                num_threads=resolve_n_jobs(self.n_jobs),
                forbidden=seed_disallow_adjacency(
                    prediction_state.orcid_join_disallows | cluster_seed_disallows,
                    {pair: LARGE_DISTANCE for pair, distance in effective_partial_supervision.items() if distance > 1},
                ),
                profile_exclusions=prediction_state.profile_exclusions,
                seed_groups=prediction_cluster_seeds_require if self.dont_merge_cluster_seeds else None,
                name_tuples=prediction_name_tuples,
                single_letter_subblocks=frozenset(block_dict_single_letter),
            )
        result = dict(pred_clusters), None
    else:
        result = self._predict_from_rust_featurizer(
            block_dict,
            rust_featurizer,
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=effective_partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            cluster_seeds_require=None,
            cluster_seeds_disallow=cluster_seed_disallows,
            prediction_state=prediction_state,
            residual_reuse=(
                attachment_state
                if attachment_state is not None and attachment_state.residual_group is not None
                else None
            ),
        )
    predict_seconds = time.perf_counter() - predict_start
    nested_telemetry = dict(prediction_state.telemetry.get("rust_featurizer_predict", {}))
    telemetry = {
        "arrow_subblocking_seconds": float(arrow_subblocking_seconds),
        "arrow_featurizer_seconds": float(arrow_featurizer_seconds),
        "rust_featurizer_predict_seconds": float(predict_seconds),
        "total_seconds": float(arrow_subblocking_seconds + arrow_featurizer_seconds + predict_seconds),
        "signature_count": int(len(signature_ids)),
        "featurizer_signature_count": int(len(featurizer_signature_ids)),
        "block_count": int(len(block_dict)),
        "pair_count": int(sum(len(signatures) * (len(signatures) - 1) // 2 for signatures in block_dict.values())),
        **{f"rust_{key}": value for key, value in nested_telemetry.items()},
    }
    prediction_state.telemetry["arrow_predict"] = telemetry
    logger.info(
        "Telemetry stage: stage=arrow_predict_total total_seconds=%.3f "
        "arrow_subblocking_seconds=%.3f arrow_featurizer_seconds=%.3f "
        "rust_featurizer_predict_seconds=%.3f "
        "signatures=%d blocks=%d pairs=%d",
        telemetry["total_seconds"],
        telemetry["arrow_subblocking_seconds"],
        telemetry["arrow_featurizer_seconds"],
        telemetry["rust_featurizer_predict_seconds"],
        telemetry["signature_count"],
        telemetry["block_count"],
        telemetry["pair_count"],
    )
    return result


def _predict_from_arrow_request(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    arrow_dataset: ArrowDataset,
    dists: dict[str, np.ndarray] | None = None,
    cluster_model_params: dict[str, Any] | None = None,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
    batching_threshold: int | None = None,
    load_name_counts: bool | None = None,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    prediction_state: PredictionState | None = None,
    bulk_clusters: Callable[[Mapping[str, list[str]]], dict[str, list[str]]] | None = None,
    attach_reuse: AttachReuse | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """Predict blocks through one open Arrow dataset with optional request-local prediction state.

    ``bulk_clusters`` lets an experiment supply the bulk clusters of the full-name subblocks,
    keyed like ``_predict_subblocked_multiple_letter_groups`` output. Subblocking, the
    no-middle reunite step, and initial-only attachment still run unchanged. The EPS sweep
    uses it to cut cached linkage trees instead of recomputing distances for each EPS.
    """

    if any(len(signatures) > 1 for signatures in block_dict.values()):
        self = _clusterer_with_prediction_params(self, cluster_model_params)
    if prediction_state is None:
        prediction_state = PredictionState()

    if runtime_context is None:
        runtime_context = build_runtime_context("cluster_predict_from_arrow", backend="rust")
    elif not stage_uses_rust(runtime_context):
        raise ValueError("Clusterer.predict_from_arrow requires a Rust runtime context")
    if batching_threshold is not None and batching_threshold <= 0:
        raise ValueError("batching_threshold must be positive")
    if batching_threshold is not None and dists is not None:
        raise ValueError("batching_threshold cannot be used with precomputed dists")
    needs_subblocking = batching_threshold is not None and any(
        len(signatures) > batching_threshold for signatures in block_dict.values()
    )
    resolved_load_name_counts = resolve_load_name_counts_policy(
        self,
        load_name_counts,
        context="Clusterer.predict_from_arrow",
    )
    if not isinstance(arrow_dataset, ArrowDataset):
        raise TypeError("Clusterer.predict_from_arrow requires an open ArrowDataset")
    with arrow_dataset.use(
        require_specter=clusterer_uses_embedding_features(self) or needs_subblocking,
        require_name_counts_index=resolved_load_name_counts,
    ) as arrow_lease:
        _require_arrow_name_counts_index_for_clusterer(self, arrow_dataset, context="Arrow prediction")
        signature_ids = list(
            dict.fromkeys(str(signature_id) for signatures in block_dict.values() for signature_id in signatures)
        )
        explicit_cluster_seeds_require = (
            None if cluster_seeds_require is None else _normalize_cluster_seeds_require(cluster_seeds_require)
        )
        cluster_seed_disallows = (
            set()
            if cluster_seeds_disallow is None
            else set(normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow))
        )
        if dists is not None and cluster_seed_disallows:
            raise ValueError(
                "cluster_seeds_disallow cannot be used with precomputed dists because disallow pairs "
                "would not be injected into the distance matrix"
            )
        if cluster_seed_disallows and orcid_join_enabled(self):
            if prediction_state is None:
                prediction_state = PredictionState()
            prediction_state.orcid_join_disallows = cluster_seed_disallows
        cluster_seed_disallows = effective_dataset_disallows(
            cluster_seed_disallows,
            partial_supervision or {},
            use_default_constraints=self.use_default_constraints_as_supervision,
        )
        effective_partial_supervision = _partial_supervision_with_cluster_seed_disallows(
            signature_ids,
            self,
            partial_supervision or {},
            cluster_seed_disallows=cluster_seed_disallows,
        )
        prediction_cluster_seeds_require = explicit_cluster_seeds_require
        if prediction_cluster_seeds_require is None and not cluster_seed_disallows:
            return self._predict_from_arrow(
                block_dict,
                arrow_dataset,
                arrow_lease,
                {},
                dists=dists,
                cluster_model_params=cluster_model_params,
                effective_partial_supervision=effective_partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                batching_threshold=batching_threshold,
                name_tuples=name_tuples,
                cluster_seed_disallows=cluster_seed_disallows,
                signature_ids=signature_ids,
                needs_subblocking=needs_subblocking,
                prediction_cluster_seeds_require=prediction_cluster_seeds_require,
                prediction_state=prediction_state,
                bulk_clusters=bulk_clusters,
                attach_reuse=attach_reuse,
            )
        with temporary_cluster_seed_sidecars(
            prediction_cluster_seeds_require or {},
            prefix="s2and_arrow_bulk_cluster_seeds_",
            cluster_seeds_disallow=cluster_seed_disallows,
        ) as sidecars:
            return self._predict_from_arrow(
                block_dict,
                arrow_dataset,
                arrow_lease,
                sidecars,
                dists=dists,
                cluster_model_params=cluster_model_params,
                effective_partial_supervision=effective_partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                batching_threshold=batching_threshold,
                name_tuples=name_tuples,
                cluster_seed_disallows=cluster_seed_disallows,
                signature_ids=signature_ids,
                needs_subblocking=needs_subblocking,
                prediction_cluster_seeds_require=prediction_cluster_seeds_require,
                prediction_state=prediction_state,
                bulk_clusters=bulk_clusters,
                attach_reuse=attach_reuse,
            )


def _subblocking_graph_config(self: Clusterer) -> GraphSubblockingConfig:
    return _resolve_graph_subblocking_config(getattr(self, "subblocking_graph_config", None))


def _subblocking_prefix_counts(self: Clusterer) -> Mapping[str, Mapping[str, int]]:
    """Get the ORCID prefix counts named by this model's feature contract.

    The contract is the hash a bundle export records, so prediction before export and after
    reload use the same counts. A model without a contract hash uses the package default.
    """

    contract = getattr(self, "feature_contract", None)
    data_sha256 = contract.get(ORCID_PREFIX_COUNTS_DATA_SHA256_FIELD) if isinstance(contract, Mapping) else None
    return load_packaged_orcid_prefix_counts(data_sha256 or DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256).counts


def _partition_subblocked_first_name_groups(
    self: Clusterer,
    block_dict_subblocked: dict[str, list[str]],
    subblocking_firsts: Mapping[str, str],
    *,
    original_blocks: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], bool]:
    """Split groups into bulk and initial-only attachment by the key first of their first member.

    ``subblocking_firsts`` maps the first member of each nonempty group to the first name of
    its subblock key (see ``_group_subblocking_firsts``). A compound such as ``"j d"`` keys as
    ``"j"``, so its group attaches like other initials.
    """

    nonempty_subblocks = {
        block_key: block_signatures
        for block_key, block_signatures in block_dict_subblocked.items()
        if len(block_signatures) > 0
    }
    single_letter = {
        block_key: block_signatures
        for block_key, block_signatures in nonempty_subblocks.items()
        if len(subblocking_firsts[block_signatures[0]]) <= 1
    }
    multiple_letter = {
        block_key: block_signatures
        for block_key, block_signatures in nonempty_subblocks.items()
        if block_key not in single_letter
    }
    if len(multiple_letter) == 0:
        return single_letter, {}, True
    origins = {sid: block for block, members in original_blocks.items() for sid in members}
    full_name_origins = {origins[members[0]] for members in multiple_letter.values()}
    # An all-initial original block has no local candidate seeds. Run it
    # through bulk prediction even when another block has full names.
    for key, members in list(single_letter.items()):
        if origins[members[0]] not in full_name_origins:
            multiple_letter[key] = single_letter.pop(key)
    return multiple_letter, single_letter, False


def _reunite_no_middle_dead_ends(
    self: Clusterer,
    pred_clusters: dict[str, list[str]],
    *,
    no_middle_dead_ends: Mapping[str, str],
    required_seed_ids: Collection[str],
    bulk_groups: Mapping[str, Sequence[str]],
    original_blocks: Mapping[str, Sequence[str]],
    attach: Callable[[dict[str, list[str]], dict[str, list[str]]], dict[str, list[str]]],
    prediction_state: PredictionState,
) -> dict[str, list[str]]:
    """Reattach no-middle dead-end records that the size cap split across subblocks.

    A record is marked when the planner reports it in the no-middle bucket of a first-name
    dead end and it is not a member of a caller-required seed group. Bulk clustering runs per
    subblock (piece), so marked records of one person in different pieces never meet. The
    step works on each bucket of each original block alone. It detaches the bulk clusters
    that hold only marked records of the bucket when they occupy two or more pieces. The
    clusters of the piece with the most of these records (the first in key order on ties)
    stay as seeds, because an attachment pass needs at least one seed. The records of each
    other piece then attach in key order, as one ``attach`` group per piece. A pass sees only
    the marked-only clusters of its bucket: the seeds and the clusters of earlier passes.
    Clusters that hold an unmarked record stay hidden, because attachments to them were
    mostly wrong. The caller runs the initial-only groups after this step, with every
    cluster visible.

    Args:
        pred_clusters: Bulk clusters of the multiple-letter groups.
        no_middle_dead_ends: The key first name of the dead end of each record in a
            no-middle dead-end bucket.
        required_seed_ids: Members of active caller-required seed groups. They are never
            marked, so their groups do not split or merge differently.
        bulk_groups: Multiple-letter groups that bulk clustering ran on, keyed by piece.
        original_blocks: Caller blocks, which scope each pass.
        attach: Runs the route's sequential attachment on ``(query groups, seed clusters)``
            and returns the resulting clusters.
        prediction_state: Receives the step telemetry.

    Returns:
        ``pred_clusters`` itself when no bucket has marked-only clusters in two or more
        pieces. Else the kept clusters followed by the reunited clusters of each bucket.
    """

    required = set(required_seed_ids)
    marked = {
        signature_id: first for signature_id, first in no_middle_dead_ends.items() if signature_id not in required
    }
    piece_by_signature = {
        signature_id: piece
        for piece, members in bulk_groups.items()
        for signature_id in members
        if signature_id in marked
    }
    origin_by_signature = {
        signature_id: origin
        for origin, members in original_blocks.items()
        for signature_id in members
        if signature_id in marked
    }
    # bucket (original block, dead-end first name) -> piece -> marked-only cluster IDs
    cluster_ids_by_bucket: defaultdict[tuple[str, str], defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cluster_id, members in pred_clusters.items():
        if not members or any(signature_id not in marked for signature_id in members):
            continue
        first = marked[members[0]]
        # Bulk can join records of two dead ends in one merged piece. Such a cluster has
        # no single bucket, so it stays as it is.
        if all(marked[signature_id] == first for signature_id in members):
            bucket = (origin_by_signature[members[0]], first)
            cluster_ids_by_bucket[bucket][piece_by_signature[members[0]]].append(cluster_id)
    passes: list[tuple[dict[str, list[str]], dict[str, list[str]]]] = []
    detached: set[str] = set()
    for bucket in sorted(cluster_ids_by_bucket):
        by_piece = cluster_ids_by_bucket[bucket]
        if len(by_piece) < 2:
            continue
        pieces = sorted(by_piece)
        # The largest piece keeps the most bulk clusters. A merged piece that holds only the
        # leftover of the cut can sort first and would leave few seeds.
        seed_piece = max(pieces, key=lambda piece: sum(len(pred_clusters[c]) for c in by_piece[piece]))
        seeds = {cluster_id: pred_clusters[cluster_id] for cluster_id in by_piece[seed_piece]}
        queries = {
            piece: [signature_id for cluster_id in by_piece[piece] for signature_id in pred_clusters[cluster_id]]
            for piece in pieces
            if piece != seed_piece
        }
        passes.append((queries, seeds))
        detached.update(cluster_id for cluster_ids in by_piece.values() for cluster_id in cluster_ids)
    telemetry: dict[str, Any] = {
        "marked_signature_count": len(marked),
        "bucket_count": len(passes),
        "detached_cluster_count": len(detached),
        "seed_cluster_count": sum(len(seeds) for _queries, seeds in passes),
        "query_group_count": sum(len(queries) for queries, _seeds in passes),
        "query_signature_count": sum(len(members) for queries, _seeds in passes for members in queries.values()),
        "seconds": 0.0,
    }
    prediction_state.telemetry["subblocked_reunite"] = telemetry
    if not passes:
        return pred_clusters
    logger.info(
        "Reuniting %d no-middle dead-end signatures of %d buckets in %d pieces against %d seed clusters",
        telemetry["query_signature_count"],
        telemetry["bucket_count"],
        telemetry["query_group_count"],
        telemetry["seed_cluster_count"],
    )
    start = time.perf_counter()
    # One call per bucket keeps the clusters of other buckets out of view.
    reunited = {f"bucket={index}": attach(queries, seeds) for index, (queries, seeds) in enumerate(passes)}
    telemetry["seconds"] = time.perf_counter() - start
    kept = {cluster_id: members for cluster_id, members in pred_clusters.items() if cluster_id not in detached}
    return _combine_subblocked_predictions({"kept": kept, **reunited})


def _cluster_one_block(
    self: Clusterer,
    block_signatures: list[str],
    dist_matrix: np.ndarray | None,
    cluster_model_params: dict[str, Any] | None,
    dataset: ANDData,
    disallow_adjacency: Mapping[str, set[str]],
    *,
    incremental_dont_use_cluster_seeds: bool = False,
    prediction_state: PredictionState | None = None,
    profile_exclusions: Mapping[str, frozenset[str]] | None = None,
    corrections: BlockCorrections | None = None,
    block_key: str = "",
) -> np.ndarray | list[int]:
    """Cluster one block from a distance matrix and return labels.

    Profile exclusions, when the block holds one of their anchors, evict
    excluded mentions from the anchor's cluster and relink the block. The
    post-linkage corrections (``s2and.block_corrections``) then run.
    """
    if len(block_signatures) == 0:
        return []
    if len(block_signatures) == 1:
        return [0]

    if dist_matrix is None:
        raise ValueError("Distance matrix is required for blocks with more than one signature.")

    block_signature_set = set(block_signatures) if profile_exclusions else set()
    block_exclusions = {
        anchor: excluded for anchor, excluded in (profile_exclusions or {}).items() if anchor in block_signature_set
    }
    if block_exclusions and not isinstance(self.cluster_model, FastCluster):
        raise ValueError("profile exclusions require the FastCluster cluster model")
    # FastCluster may overwrite its input; keep the model distances for relinking.
    original_distances = np.array(dist_matrix, copy=True) if block_exclusions else None
    # When clustering consumes its input, the caller's matrix is spent after
    # the first fit. Relinks reuse it, so a block peaks at two matrices.
    preserve_input = (cluster_model_params or {}).get(
        "preserve_input", getattr(self.cluster_model, "preserve_input", True)
    )
    scratch = dist_matrix if block_exclusions and not preserve_input and dist_matrix.flags.writeable else None

    def relabel(matrix: np.ndarray) -> np.ndarray | list[int]:
        cluster_model = clone_cluster_model(self.cluster_model)
        params = {k: intify(v) for k, v in (cluster_model_params or {}).items()}
        cluster_model.set_params(**params)
        cluster_model.fit(matrix)
        return self._finish_block_labels(
            block_signatures,
            cluster_model.labels_,
            dataset,
            disallow_adjacency,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            prediction_state=prediction_state,
        )

    # Linkage may overwrite the matrix, so the corrections keep a one-byte-per-pair copy.
    # One thread: the copy then runs on this thread. A pool job would wait behind the
    # next block's scoring, which measured 270 times slower on a giant block.
    block_codes = None if corrections is None else corrections.codes_for(block_signatures, dist_matrix, 1)

    labels = relabel(dist_matrix)
    if original_distances is not None:
        labels = enforce_profile_exclusions_in_block(
            block_signatures,
            labels,
            original_distances,
            block_exclusions,
            relabel,
            cluster_seeds_require=self._enforced_cluster_seeds_require(
                dataset, prediction_state, incremental_dont_use_cluster_seeds
            ),
            scratch=scratch,
            block_key=block_key,
        )
    if corrections is None or block_codes is None:
        return labels
    forbidden: Mapping[str, Set[str]] = disallow_adjacency
    if block_exclusions and block_codes.repair:
        # A correction may not bring an excluded mention back to its anchor.
        both_ways: dict[str, set[str]] = {key: set(value) for key, value in disallow_adjacency.items()}
        for anchor, excluded in block_exclusions.items():
            both_ways.setdefault(anchor, set()).update(excluded)
            for signature_id in excluded:
                both_ways.setdefault(signature_id, set()).add(anchor)
        forbidden = both_ways
    return cast(
        "np.ndarray | list[int]",
        corrections.apply(
            block_signatures,
            labels,
            block_codes,
            cut_height=average_linkage_cut(self.cluster_model, cluster_model_params),
            forbidden=forbidden,
            # Each view copies the request's seeds, so it is built only for a step that runs.
            repair_seed_groups=(self._repair_seed_groups(dataset, prediction_state) if block_codes.repair else None),
            # A split never separates a required seed group, so it reads every seed in sight,
            # unless the caller turned seeds off, as the altered-profile pre-split does.
            split_seed_groups=(
                self._seeds_in_sight(dataset, prediction_state)
                if block_codes.split and not incremental_dont_use_cluster_seeds
                else None
            ),
            block_key=block_key,
        ),
    )


def _finish_block_labels(
    self: Clusterer,
    block_signatures: list[str],
    labels: np.ndarray,
    dataset: ANDData,
    disallow_adjacency: Mapping[str, set[str]],
    *,
    incremental_dont_use_cluster_seeds: bool = False,
    prediction_state: PredictionState | None = None,
) -> np.ndarray | list[int]:
    """Relabel outliers and join seed groups after a block's cluster-model fit."""
    max_label = labels.max()
    # In HDBSCAN, label -1 denotes outliers.
    # Give each outlier its own unique label starting at max_label + 1.
    negative_one_label_locations = np.where(labels == -1)[0]
    for i, loc in enumerate(negative_one_label_locations):
        labels[loc] = max_label + 1 + i
    cluster_seeds_require = self._enforced_cluster_seeds_require(
        dataset, prediction_state, incremental_dont_use_cluster_seeds
    )
    if cluster_seeds_require is not None:
        return merge_seed_labels(block_signatures, labels, cluster_seeds_require, disallow_adjacency)
    return labels


def _enforced_cluster_seeds_require(
    self: Clusterer,
    dataset: ANDData,
    prediction_state: PredictionState | None,
    incremental_dont_use_cluster_seeds: bool,
) -> Mapping[str, Any] | None:
    """Return the required seed groups that block labels must keep, or None when seeds are off."""
    if not self.use_default_constraints_as_supervision or incremental_dont_use_cluster_seeds:
        return None
    return dataset.cluster_seeds_require if prediction_state is None else prediction_state.cluster_seeds_require


def _repair_seed_groups(
    self: Clusterer,
    dataset: ANDData,
    prediction_state: PredictionState | None,
) -> Mapping[str, Any] | None:
    """Return the required seed groups that a block correction may not join, or None.

    Reads every seed in sight, whether or not block labels enforce it. Explicit seeds stay
    in the distances when default constraints are off, and a needless veto only skips a repair.
    """
    if not self.dont_merge_cluster_seeds:
        return None
    return self._seeds_in_sight(dataset, prediction_state)


def _cluster_one_block_with_logging(
    self: Clusterer,
    block_signatures: list[str],
    dist_matrix: np.ndarray | None,
    cluster_model_params: dict[str, Any] | None,
    dataset: ANDData,
    disallow_adjacency: Mapping[str, set[str]],
    *,
    block_key: str,
    incremental_dont_use_cluster_seeds: bool,
    prediction_state: PredictionState | None = None,
    profile_exclusions: Mapping[str, frozenset[str]] | None = None,
    corrections: BlockCorrections | None = None,
) -> np.ndarray | list[int]:
    """Cluster one block and emit explicit entry/exit logs around the cluster-model fit."""
    cluster_model_name = type(self.cluster_model).__name__
    logger.info(
        "Starting cluster_model.fit for block %s using %s (signatures=%d)",
        block_key,
        cluster_model_name,
        len(block_signatures),
    )
    cluster_start = time.perf_counter()
    labels = self._cluster_one_block(
        block_signatures,
        dist_matrix,
        cluster_model_params,
        dataset,
        disallow_adjacency,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        prediction_state=prediction_state,
        profile_exclusions=profile_exclusions,
        corrections=corrections,
        block_key=block_key,
    )
    logger.info(
        "Finished cluster_model.fit for block %s using %s in %.3fs (clusters=%d)",
        block_key,
        cluster_model_name,
        time.perf_counter() - cluster_start,
        len(set(labels)),
    )
    return labels
