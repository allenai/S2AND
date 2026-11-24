"""Resolve pair constraints and request-level supervision."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import _get_rust_featurizer
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs
from s2and.incremental_linking.policy import request_cluster_seed_disallow_parts
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context, dataset_stage_uses_rust, stage_uses_rust
from s2and.rust_calls import get_constraints_matrix_indexed_rust

if TYPE_CHECKING:
    from .clusterer import Clusterer


def _normalize_cluster_seeds_require(cluster_seeds_require: Mapping[Any, Any]) -> dict[str, int | str]:
    """Return the seed map with the same string policy used by Arrow sidecars."""

    # Attachment passes hand over half a million string ids; when every key and
    # value already is a str, a copy equals the conversion.
    if set(map(type, cluster_seeds_require)) <= {str} and set(map(type, cluster_seeds_require.values())) <= {str}:
        return dict(cluster_seeds_require)
    return {str(signature_id): str(cluster_id) for signature_id, cluster_id in cluster_seeds_require.items()}


def _profile_exclusion_kwargs(prediction_state: PredictionState | None) -> dict[str, Any]:
    """Return the clustering keyword for profile exclusions, empty when there are none."""
    if prediction_state is None or not prediction_state.profile_exclusions:
        return {}
    return {"profile_exclusions": prediction_state.profile_exclusions}


def _cluster_seed_disallows_for_request(
    dataset: Any,
    explicit_disallows: Iterable[tuple[Any, Any]] = (),
) -> set[tuple[str, str]]:
    request_disallows, _dataset_disallows, _arrow_disallows = request_cluster_seed_disallow_parts(
        dataset,
        normalize_cluster_seed_disallow_pairs(explicit_disallows),
    )
    return request_disallows


def _partial_supervision_with_cluster_seed_disallows(
    signatures: Sequence[str],
    dataset: Any,
    partial_supervision: Mapping[tuple[str, str], int | float],
    cluster_seed_disallows: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], int | float]:
    merged: dict[tuple[str, str], int | float] = dict(partial_supervision)
    signature_set = {str(signature_id) for signature_id in signatures}
    disallow_pairs = (
        _cluster_seed_disallows_for_request(dataset) if cluster_seed_disallows is None else cluster_seed_disallows
    )
    for left, right in disallow_pairs:
        left_id = str(left)
        right_id = str(right)
        if (
            left_id in signature_set
            and right_id in signature_set
            and (left_id, right_id) not in merged
            and (right_id, left_id) not in merged
        ):
            merged[(left_id, right_id)] = LARGE_DISTANCE
    return merged


def _set_partial_supervision_if_absent_bidirectional(
    merged: dict[tuple[str, str], int | float],
    left_id: str,
    right_id: str,
    value: int | float,
) -> None:
    if left_id == right_id:
        return
    if (left_id, right_id) in merged or (right_id, left_id) in merged:
        return
    merged[(left_id, right_id)] = value


def _partial_supervision_with_cluster_seed_overrides(
    signatures: Sequence[str],
    partial_supervision: Mapping[tuple[str, str], int | float],
    *,
    cluster_seeds_require: Mapping[str, int | str] | None = None,
    cluster_seeds_disallow: Iterable[tuple[str, str]] = (),
    dont_merge_cluster_seeds: bool = True,
    incremental_dont_use_cluster_seeds: bool = False,
) -> dict[tuple[str, str], int | float]:
    """Merge request-scoped seed overrides into block-local partial supervision."""

    merged: dict[tuple[str, str], int | float] = dict(partial_supervision)
    signature_set = {str(signature_id) for signature_id in signatures}
    for left, right in cluster_seeds_disallow:
        left_id = str(left)
        right_id = str(right)
        if left_id in signature_set and right_id in signature_set:
            _set_partial_supervision_if_absent_bidirectional(merged, left_id, right_id, LARGE_DISTANCE)
    if cluster_seeds_require and not incremental_dont_use_cluster_seeds:
        seed_component_by_signature: dict[str, str] = {}
        for signature_id, component_id in cluster_seeds_require.items():
            signature_key = str(signature_id)
            if signature_key in signature_set:
                seed_component_by_signature[signature_key] = str(component_id)
        if dont_merge_cluster_seeds:
            seeded_signatures = list(seed_component_by_signature)
            for left_index, left_id in enumerate(seeded_signatures):
                left_component = seed_component_by_signature[left_id]
                for right_id in seeded_signatures[left_index + 1 :]:
                    if left_component != seed_component_by_signature[right_id]:
                        _set_partial_supervision_if_absent_bidirectional(
                            merged,
                            left_id,
                            right_id,
                            LARGE_DISTANCE,
                        )
        signatures_by_component: dict[str, list[str]] = defaultdict(list)
        for signature_id, component_id in seed_component_by_signature.items():
            signatures_by_component[component_id].append(signature_id)
        for component_signatures in signatures_by_component.values():
            for left_index, left_id in enumerate(component_signatures):
                for right_id in component_signatures[left_index + 1 :]:
                    _set_partial_supervision_if_absent_bidirectional(merged, left_id, right_id, 0)
    return merged


# The fewest full-first-name mentions an ORCID needs, and the largest share of them a
# name may conflict with. See Clusterer.orcid_name_waiver.
DEFAULT_ORCID_NAME_WAIVER: tuple[int, float] = (5, 0.1)


def _orcid_name_waiver_setting(clusterer: Any) -> tuple[int, float] | None:
    """Return a clusterer's ORCID name waiver as the tuple the Rust calls take."""
    setting = getattr(clusterer, "orcid_name_waiver", None)
    if setting is None:
        return None
    # bool is an int, and a string is a sequence; neither is a setting.
    valid = (
        isinstance(setting, tuple | list)
        and len(setting) == 2
        and isinstance(setting[0], int)
        and isinstance(setting[1], int | float)
        and not any(isinstance(value, bool) for value in setting)
        and setting[0] >= 1
        and 0.0 <= setting[1] <= 1.0
    )
    if not valid:
        raise ValueError(f"orcid_name_waiver must be None or (mentions >= 1, share in [0, 1]), got {setting!r}")
    return (setting[0], float(setting[1]))


def _use_rust_constraints(
    runtime_context: RuntimeContext | None = None,
    dataset: ANDData | None = None,
) -> bool:
    if runtime_context is None:
        runtime_context = (
            dataset.runtime_context
            if dataset is not None
            else build_runtime_context(
                "constraints",
                backend="python",
            )
        )
    if dataset is None:
        return stage_uses_rust(runtime_context)
    # Rust constraints run on the dataset's Rust featurizer, which is built
    # exclusively from Arrow artifacts; datasets without them use Python constraints.
    return dataset_stage_uses_rust(runtime_context, dataset)


@dataclass(frozen=True)
class _ConstraintPolicy:
    """Resolved hard-constraint behavior for one constraint evaluation boundary."""

    dont_merge_cluster_seeds: bool = True
    incremental_dont_use_cluster_seeds: bool = False
    suppress_orcid: bool = False


def _initialize_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
    prediction_state: PredictionState | None = None,
) -> tuple[object | None, bool | None]:
    if not use_default_constraints_as_supervision:
        return None, None

    use_rust_constraints = _use_rust_constraints(runtime_context, dataset)
    if not use_rust_constraints:
        return None, False

    return _get_rust_featurizer(dataset, prediction_state=prediction_state), True


def _resolve_constraint_api_mode(
    rust_featurizer: object | None,
    use_rust_constraints: bool | None,
) -> str:
    if not use_rust_constraints or rust_featurizer is None:
        return "python"
    return "indexed"


def _build_signature_index_by_id(rust_featurizer: object) -> dict[str, int]:
    signature_ids = cast(Any, rust_featurizer).signature_ids()
    return {str(sig_id): idx for idx, sig_id in enumerate(signature_ids)}


def _build_incremental_constraint_backend(
    dataset: ANDData,
    *,
    use_default_constraints_as_supervision: bool,
    runtime_context: RuntimeContext,
    suppress_orcid: bool = False,
    prediction_state: PredictionState | None = None,
) -> _IncrementalConstraintBackend:
    """Build Phase A constraint backend state once for reuse across subblocks."""
    rust_featurizer, use_rust_constraints = _initialize_incremental_constraint_backend(
        dataset,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        runtime_context=runtime_context,
        prediction_state=prediction_state,
    )
    constraint_api_mode = _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    signature_index_by_id: dict[str, int] | None = None
    if constraint_api_mode == "indexed" and rust_featurizer is not None:
        signature_index_by_id = _build_signature_index_by_id(rust_featurizer)
    return _IncrementalConstraintBackend(
        rust_featurizer=rust_featurizer,
        use_rust_constraints=use_rust_constraints,
        constraint_api_mode=constraint_api_mode,
        signature_index_by_id=signature_index_by_id,
        suppress_orcid=suppress_orcid,
        prediction_state=prediction_state,
    )


def _resolve_constraint_labels_batch(
    dataset: ANDData,
    pair_ids: list[tuple[str, str]],
    *,
    constraint_backend: _IncrementalConstraintBackend | None = None,
    partial_supervision: dict[tuple[str, str], int | float],
    use_default_constraints_as_supervision: bool,
    constraint_policy: _ConstraintPolicy,
    rust_featurizer: object | None = None,
    use_rust_constraints: bool | None = None,
    runtime_context: RuntimeContext,
    num_threads: int | None = None,
    constraint_api_mode: str | None = None,
    signature_index_by_id: dict[str, int] | None = None,
    prediction_state: PredictionState | None = None,
) -> tuple[list[float], _ConstraintBatchTelemetry]:
    if constraint_backend is not None:
        rust_featurizer = constraint_backend.rust_featurizer
        prediction_state = constraint_backend.prediction_state
        use_rust_constraints = constraint_backend.use_rust_constraints
        constraint_policy = _ConstraintPolicy(
            dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
            suppress_orcid=constraint_backend.suppress_orcid,
        )
        if constraint_api_mode is None:
            constraint_api_mode = constraint_backend.constraint_api_mode
        if signature_index_by_id is None:
            signature_index_by_id = constraint_backend.signature_index_by_id

    labels: list[float] = [float(np.nan)] * len(pair_ids)
    unresolved_pairs: list[tuple[str, str]] = []
    unresolved_indices: list[int] = []
    partial_hits = 0
    for idx, (sig_id_1, sig_id_2) in enumerate(pair_ids):
        if (sig_id_1, sig_id_2) in partial_supervision:
            # Subtract LARGE_INTEGER so downstream featurization knows not to recompute these constraints.
            labels[idx] = float(partial_supervision[(sig_id_1, sig_id_2)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        if (sig_id_2, sig_id_1) in partial_supervision:
            # Subtract LARGE_INTEGER so downstream featurization knows not to recompute these constraints.
            labels[idx] = float(partial_supervision[(sig_id_2, sig_id_1)] - LARGE_INTEGER)
            partial_hits += 1
            continue
        unresolved_pairs.append((sig_id_1, sig_id_2))
        unresolved_indices.append(idx)

    mode = constraint_api_mode or _resolve_constraint_api_mode(rust_featurizer, use_rust_constraints)
    telemetry = _ConstraintBatchTelemetry(
        total_pairs=int(len(pair_ids)),
        partial_supervision_hits=int(partial_hits),
        unresolved_pairs=int(len(unresolved_pairs)),
        rust_batch_call_count=0,
        api_mode=mode,
        elapsed_seconds=0.0,
    )
    if not unresolved_pairs or not use_default_constraints_as_supervision:
        if not use_default_constraints_as_supervision:
            telemetry.api_mode = "partial_only"
        return labels, telemetry

    start = time.perf_counter()
    values: list[float | None]

    def _resolve_values_python() -> list[float | None]:
        if prediction_state is not None:
            return [
                dataset._get_constraint(
                    s1,
                    s2,
                    cluster_seeds_require=prediction_state.cluster_seeds_require,
                    cluster_seeds_disallow=prediction_state.cluster_seeds_disallow,
                    dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
                    incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
                    suppress_orcid=constraint_policy.suppress_orcid,
                )
                for s1, s2 in unresolved_pairs
            ]
        return [
            dataset.get_constraint(
                s1,
                s2,
                dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
                suppress_orcid=constraint_policy.suppress_orcid,
            )
            for s1, s2 in unresolved_pairs
        ]

    if use_rust_constraints and rust_featurizer is not None and mode == "indexed":

        def _resolve_values_rust() -> list[float | None]:
            if signature_index_by_id is None:
                raise RuntimeError("Indexed constraint API requested without signature index lookup")
            indexed_pairs = [(signature_index_by_id[s1], signature_index_by_id[s2]) for s1, s2 in unresolved_pairs]
            return get_constraints_matrix_indexed_rust(
                indexed_pairs,
                dont_merge_cluster_seeds=constraint_policy.dont_merge_cluster_seeds,
                incremental_dont_use_cluster_seeds=constraint_policy.incremental_dont_use_cluster_seeds,
                num_threads=num_threads,
                featurizer=rust_featurizer,
                suppress_orcid=constraint_policy.suppress_orcid,
            )

        try:
            values = _resolve_values_rust()
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Rust batch constraint evaluation failed in strict rust backend "
                f"(pairs={len(unresolved_pairs)} run_id={runtime_context.run_id} error={exc})"
            ) from exc
        telemetry.rust_batch_call_count = 1
    else:
        values = _resolve_values_python()
        telemetry.api_mode = "python"

    telemetry.elapsed_seconds = float(time.perf_counter() - start)
    for idx, value in zip(unresolved_indices, values, strict=True):
        if value is None:
            labels[idx] = float(np.nan)
        else:
            # Keep partial/constraint labels in the LARGE_INTEGER-offset convention.
            labels[idx] = float(value - LARGE_INTEGER)
    return labels, telemetry


@dataclass
class _ConstraintTelemetryAccumulator:
    total_pairs: int = 0
    partial_supervision_hits: int = 0
    unresolved_pairs: int = 0
    rust_batch_call_count: int = 0
    elapsed_seconds: float = 0.0
    api_modes: set[str] = field(default_factory=set)

    @property
    def api_mode_summary(self) -> str:
        return ",".join(sorted(self.api_modes)) if self.api_modes else "none"


@dataclass
class _ConstraintBatchTelemetry:
    total_pairs: int
    partial_supervision_hits: int
    unresolved_pairs: int
    rust_batch_call_count: int
    api_mode: str
    elapsed_seconds: float


def _accumulate_constraint_telemetry(
    accumulator: _ConstraintTelemetryAccumulator,
    batch_telemetry: _ConstraintBatchTelemetry,
) -> None:
    accumulator.total_pairs += int(batch_telemetry.total_pairs)
    accumulator.partial_supervision_hits += int(batch_telemetry.partial_supervision_hits)
    accumulator.unresolved_pairs += int(batch_telemetry.unresolved_pairs)
    accumulator.rust_batch_call_count += int(batch_telemetry.rust_batch_call_count)
    accumulator.elapsed_seconds += float(batch_telemetry.elapsed_seconds)
    accumulator.api_modes.add(str(batch_telemetry.api_mode))


@dataclass(frozen=True)
class _IncrementalConstraintBackend:
    """Pre-computed Phase A constraint backend state, invariant across subblocks."""

    rust_featurizer: object | None
    use_rust_constraints: bool | None
    constraint_api_mode: str
    signature_index_by_id: dict[str, int] | None
    suppress_orcid: bool = False
    prediction_state: PredictionState | None = None


@dataclass
class _RustFeaturizerPredictDataset:
    """Minimal dataset shape needed after Arrow inputs have entered Rust."""

    cluster_seeds_require: dict[str, int | str]
    cluster_seeds_disallow: set[tuple[str, str]]
    signatures: Mapping[str, Any] = field(default_factory=dict)
    name: str = "rust_featurizer_predict"
    # The stand-in has no signatures; name-conflict repair reads its facts from the featurizer.
    rust_featurizer: Any = None
    # Every required seed group in the request, even when linkage ignores them, so the repair
    # keeps them apart as it does on the Python route.
    repair_cluster_seeds_require: Mapping[str, Any] = field(default_factory=dict)


def _cluster_seeds_require_from_rust_featurizer(rust_featurizer: Any) -> dict[str, int | str]:
    return {
        str(signature_id): str(component_id) for signature_id, component_id in rust_featurizer.cluster_seeds_require()
    }


def _resolve_constraint_batch(
    self: Clusterer,
    dataset: ANDData,
    pair_ids: list[tuple[str, str]],
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    *,
    incremental_dont_use_cluster_seeds: bool,
    constraint_backend: _IncrementalConstraintBackend,
    prediction_state: PredictionState | None = None,
) -> tuple[list[float], _ConstraintBatchTelemetry]:
    return _resolve_constraint_labels_batch(
        dataset,
        pair_ids,
        constraint_backend=constraint_backend,
        partial_supervision=partial_supervision,
        use_default_constraints_as_supervision=self.use_default_constraints_as_supervision,
        constraint_policy=_ConstraintPolicy(
            dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            suppress_orcid=constraint_backend.suppress_orcid,
        ),
        runtime_context=runtime_context,
        num_threads=self.n_jobs,
        prediction_state=prediction_state,
    )
