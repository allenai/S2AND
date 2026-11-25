"""Set up seed profiles and predict incremental Arrow requests."""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from s2and import memory_budget
from s2and.arrow_inputs import ArrowDataset, arrow_string_array
from s2and.corrections.block_corrections import correction_fingerprint
from s2and.data import ANDData
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset, release_rust_handle
from s2and.incremental_linking.attach_reuse import AttachReuse
from s2and.incremental_linking.completion import ResidualClusterer
from s2and.incremental_linking.completion_metadata import (
    name_tuples_for_incremental_rules as _name_tuples_for_incremental_rules,
)
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs
from s2and.incremental_linking.policy import clusterer_uses_embedding_features, clusterer_uses_name_count_features
from s2and.incremental_linking.policy import (
    require_arrow_name_counts_index_for_clusterer as _require_arrow_name_counts_index_for_clusterer,
)
from s2and.incremental_linking.production import (
    _DirectArrowIncrementalDataset,
    predict_incremental_promoted_linker_from_arrow,
    start_background_seed_planner,
)
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context, stage_uses_rust
from s2and.supervision import (
    effective_dataset_disallows,
    normalize_partial_supervision,
    validate_incremental_seed_disallows,
)
from s2and.text import CanonicalNameParts, canonicalize_name_parts, normalize_orcid_compact
from s2and.thread_config import resolve_n_jobs

from ._shared import (
    _ARROW_ATTACH_BLOCK_STATE,
    _ARROW_ATTACH_NAME_CHECK_EXEMPT,
    _ARROW_BULK_DISTANCE_OVERRIDES,
    _gc_paused,
    logger,
)
from .constraints import (
    _cluster_seed_disallows_for_request,
    _normalize_cluster_seeds_require,
    _orcid_name_waiver_setting,
    _partial_supervision_with_cluster_seed_disallows,
)

if TYPE_CHECKING:
    from .clusterer import Clusterer

_ALTERED_PRESPLIT_CACHE_MAX_ENTRIES = 128


_ALTERED_PRESPLIT_CACHE_INIT_LOCK = threading.Lock()


class _AttachmentBlockState:
    """Native state that the attachment passes of one original block share.

    The featurizer is built when a pass first needs features, which skips the cost
    when the incremental step never featurizes, for example a caller override of
    ``predict_incremental_from_arrow``. The incremental step stores its planner in
    ``planner`` (with the build arguments in ``planner_key``) for the next pass.
    """

    def __init__(
        self,
        arrow_dataset: ArrowDataset,
        signature_ids: list[str],
        name_tuples: Any,
        num_threads: int,
    ) -> None:
        self.signature_ids = frozenset(signature_ids)
        self.name_tuples = name_tuples
        self._build_args = (arrow_dataset, list(signature_ids), num_threads)
        self._featurizer: Any | None = None
        self._orcid_info: dict[str, _ArrowIncrementalSignatureInfo] | None = None
        self.shared_setup: _AttachmentBlockState | None = None
        self.attach_reuse: AttachReuse | None = None
        self.planner: Any | None = None
        self.planner_key: tuple[Any, ...] | None = None

    @property
    def name_parts(self) -> dict[str, tuple[CanonicalNameParts, str | None]]:
        if not hasattr(self, "_name_parts"):
            self._name_parts: dict[str, tuple[CanonicalNameParts, str | None]] = {}
        return self._name_parts

    @property
    def arrow_dataset(self) -> ArrowDataset:
        return self._build_args[0]

    def orcid_signature_info(
        self, arrow_lease: Any, signature_ids: Sequence[str], *, num_threads: int | None = None
    ) -> dict[str, _ArrowIncrementalSignatureInfo]:
        """Return `_load_arrow_incremental_orcid_signature_info` for ``signature_ids``.

        Every pass asks for most of the block's seeds, so the block's ORCIDs are
        read once, through the calling pass's lease, instead of once per pass.
        Entries keep the loader's Arrow row order.
        """
        if not self.signature_ids.issuperset(signature_ids):
            # Rows outside the block can sit anywhere in the table, so read all
            # requested ids together to keep the loader's order.
            return _load_arrow_incremental_orcid_signature_info(arrow_lease, signature_ids, num_threads=num_threads)
        if self._orcid_info is None:
            self._orcid_info = _load_arrow_incremental_orcid_signature_info(
                arrow_lease, self._build_args[1], num_threads=num_threads
            )
        requested = set(signature_ids)
        return {signature_id: info for signature_id, info in self._orcid_info.items() if signature_id in requested}

    def release(self) -> None:
        """Release paper buffers, then drop remaining native state in the background."""

        for handle in (self._featurizer, self.planner):
            if handle is not None:
                release_rust_handle(handle)
        self._featurizer = None
        self.planner = None

    def signature_info(
        self, arrow_lease: Any, signature_ids: Sequence[str]
    ) -> dict[str, _ArrowIncrementalSignatureInfo]:
        """Read each query's metadata once across the offline block's passes."""
        if not hasattr(self, "_signature_info"):
            self._signature_info: dict[str, _ArrowIncrementalSignatureInfo] = {}
        missing = [sid for sid in signature_ids if sid not in self._signature_info]
        self._signature_info.update(_load_arrow_incremental_signature_info(arrow_lease, missing))
        return {sid: self._signature_info[sid] for sid in signature_ids}

    def featurizer(self) -> Any:
        if self.shared_setup is not None:
            return self.shared_setup.featurizer()
        if self._featurizer is None:
            arrow_dataset, signature_ids, num_threads = self._build_args
            self._featurizer = build_rust_featurizer_from_arrow_dataset(
                arrow_dataset,
                signature_ids=signature_ids,
                name_tuples=self.name_tuples,
                preprocess=True,
                num_threads=num_threads,
            )
        return self._featurizer


@dataclass(frozen=True)
class _ArrowIncrementalSignatureInfo:
    """Signature fields needed by Arrow-only incremental completion."""

    paper_id: str | None
    author_info_first: str | None
    author_info_last: str | None
    author_info_first_normalized: str | None
    author_info_orcid: str | None
    author_info_middle: str | None = None


def _cacheable_value(value: Any) -> Any:
    if isinstance(value, int | float | str | type(None)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _cacheable_value(item)) for key, item in value.items()))
    if isinstance(value, set | frozenset):
        return tuple(sorted((_cacheable_value(item) for item in value), key=repr))
    if isinstance(value, list | tuple):
        return tuple(_cacheable_value(item) for item in value)
    return repr(value)


def _partial_supervision_cache_fingerprint(
    partial_supervision: Mapping[tuple[Any, Any], int | float],
) -> tuple[tuple[str, str, float], ...]:
    return tuple(sorted((str(left), str(right), float(value)) for (left, right), value in partial_supervision.items()))


@dataclass(frozen=True)
class _AlteredPresplitJob:
    block_key: str
    altered_cluster_num: int | str
    signature_ids: list[str]
    partial_supervision: dict[tuple[str, str], int | float]
    cache_key: tuple[Any, ...] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class _IncrementalSeedSetup:
    """Request-local seed maps produced once by incremental seed setup.

    Frozen fields cannot be reassigned, but the mappings are the setup's owned
    objects and are not copied or frozen; consumers treat them as read-only.

    Attributes:
        required_assignments: Signature to its current scoring component. An
            altered profile's signatures map to split components, not the profile.
        recluster_map: Split scoring component to its original profile ID.
        original_members: Original profile ID to its members, in output order.
        split_members: Scoring component to its members. The same object as
            ``original_members`` when no profile was split.
    """

    required_assignments: dict[str, int | str]
    recluster_map: dict[int | str, int | str]
    original_members: dict[int | str, list[str]]
    split_members: dict[int | str, list[str]]


def _model_presplit_cache_fingerprint(clusterer: Any) -> tuple[Any, ...]:
    cluster_model = getattr(clusterer, "cluster_model", None)
    get_params = getattr(cluster_model, "get_params", None)
    cluster_model_params = get_params(deep=False) if callable(get_params) else {}
    return (
        _estimator_cache_fingerprint(getattr(clusterer, "classifier", None)),
        _estimator_cache_fingerprint(getattr(clusterer, "nameless_classifier", None)),
        _cacheable_value(cluster_model_params),
        _cacheable_value(getattr(getattr(clusterer, "featurizer_info", None), "features_to_use", ())),
        _cacheable_value(getattr(getattr(clusterer, "nameless_featurizer_info", None), "features_to_use", ())),
        bool(getattr(clusterer, "use_default_constraints_as_supervision", True)),
        bool(getattr(clusterer, "dont_merge_cluster_seeds", True)),
        bool(getattr(clusterer, "suppress_orcid", False)),
        # The corrections' settings change the pre-split partition too.
        _cacheable_value(correction_fingerprint(clusterer)),
        _orcid_name_waiver_setting(clusterer),
    )


def _estimator_cache_fingerprint(estimator: Any) -> Any:
    if estimator is None:
        return None
    inner = getattr(estimator, "classifier", None)
    if inner is not None and inner is not estimator:
        return (
            "wrapped",
            type(estimator).__module__,
            type(estimator).__qualname__,
            _estimator_cache_fingerprint(inner),
        )
    cache_fingerprint = getattr(estimator, "cache_fingerprint", None)
    if callable(cache_fingerprint):
        return (
            type(estimator).__module__,
            type(estimator).__qualname__,
            tuple(cache_fingerprint()),
        )
    booster = getattr(estimator, "booster_", None)
    model_to_string = getattr(booster, "model_to_string", None)
    if callable(model_to_string):
        model_string = str(model_to_string())
        return (
            type(estimator).__module__,
            type(estimator).__qualname__,
            hashlib.blake2b(model_string.encode("utf-8"), digest_size=16).hexdigest(),
        )
    estimator_type = (type(estimator).__module__, type(estimator).__qualname__)
    try:
        serialized = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    except (AttributeError, pickle.PickleError, RecursionError, TypeError):
        # An estimator without a complete stable representation cannot safely
        # reuse results produced by an earlier state.
        return (*estimator_type, object())
    return (*estimator_type, hashlib.blake2b(serialized, digest_size=16).hexdigest())


def _altered_presplit_cache_state(
    clusterer: Any,
) -> tuple[OrderedDict[tuple[Any, ...], tuple[tuple[str, ...], ...]], Any]:
    clusterer = getattr(clusterer, "_s2and_prediction_cache_owner", clusterer)
    with _ALTERED_PRESPLIT_CACHE_INIT_LOCK:
        cache = getattr(clusterer, "_s2and_altered_presplit_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            clusterer._s2and_altered_presplit_cache = cache
    return cache, _ALTERED_PRESPLIT_CACHE_INIT_LOCK


def _get_altered_presplit_cache_entry(
    clusterer: Any,
    key: tuple[Any, ...],
) -> tuple[tuple[str, ...], ...] | None:
    cache, lock = _altered_presplit_cache_state(clusterer)
    with lock:
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value


def _put_altered_presplit_cache_entry(
    clusterer: Any,
    key: tuple[Any, ...],
    clusters: Sequence[Sequence[str]],
) -> None:
    value = tuple(tuple(str(signature_id) for signature_id in cluster) for cluster in clusters)
    cache, lock = _altered_presplit_cache_state(clusterer)
    with lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _ALTERED_PRESPLIT_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)


def _altered_presplit_cache_key(
    *,
    mode: str,
    altered_cluster_num: int | str,
    signature_ids: Sequence[str],
    effective_partial_supervision: Mapping[tuple[Any, Any], int | float],
    arrow_generation_id: str | None,
    name_tuples_key: Any,
    model_fingerprint: tuple[Any, ...],
) -> tuple[Any, ...]:
    return (
        "altered_presplit_v1",
        str(mode),
        str(altered_cluster_num),
        tuple(str(signature_id) for signature_id in signature_ids),
        _partial_supervision_cache_fingerprint(effective_partial_supervision),
        arrow_generation_id,
        name_tuples_key,
        model_fingerprint,
    )


def _normalized_orcid_for_presplit_skip(dataset: Any, signature_id: str) -> str | None:
    signatures = getattr(dataset, "signatures", None)
    if not isinstance(signatures, Mapping):
        return None
    signature = signatures.get(str(signature_id))
    if signature is None:
        return None
    raw_orcid = getattr(signature, "author_info_orcid", None)
    if raw_orcid is None:
        return None
    return normalize_orcid_compact(raw_orcid)


def _has_in_profile_pair(
    pairs: Mapping[tuple[Any, Any], Any] | set[tuple[Any, Any]],
    signature_ids: Sequence[str],
) -> bool:
    signature_id_set = {str(signature_id) for signature_id in signature_ids}
    for left, right in pairs:
        if str(left) in signature_id_set and str(right) in signature_id_set:
            return True
    return False


def _can_skip_orcid_homogeneous_altered_presplit(
    clusterer: Any,
    dataset: Any,
    signature_ids: Sequence[str],
    partial_supervision: Mapping[tuple[Any, Any], int | float],
    cluster_seed_disallows: set[tuple[str, str]] | None = None,
) -> bool:
    if not bool(getattr(clusterer, "use_default_constraints_as_supervision", True)):
        return False
    if bool(getattr(clusterer, "suppress_orcid", False)):
        return False
    if _has_in_profile_pair(partial_supervision, signature_ids):
        return False
    disallow_pairs = (
        _cluster_seed_disallows_for_request(dataset) if cluster_seed_disallows is None else cluster_seed_disallows
    )
    if _has_in_profile_pair(disallow_pairs, signature_ids):
        return False
    orcid_values = [_normalized_orcid_for_presplit_skip(dataset, str(signature_id)) for signature_id in signature_ids]
    if any(orcid is None for orcid in orcid_values):
        return False
    return len(set(orcid_values)) == 1


def _seed_cluster_count_from_seed_map(cluster_seeds_require: Mapping[Any, Any]) -> int:
    """Return the number of unique seed components in the supplied seed map."""

    return len({str(cluster_id) for cluster_id in cluster_seeds_require.values()})


def _optional_arrow_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _signature_batch_indices_for_request(
    arrow_dataset: Any,
    requested_signature_ids: Sequence[str],
) -> tuple[int, ...]:
    return tuple(arrow_dataset.native.batch_indices("signatures", list(requested_signature_ids)))


def _load_arrow_incremental_signature_info(
    arrow_dataset: Any,
    signature_ids: Iterable[Any],
) -> dict[str, _ArrowIncrementalSignatureInfo]:
    """Load request-local signature metadata from `signatures.arrow`."""

    requested_signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
    if not requested_signature_ids:
        return {}
    requested_signature_id_set = set(requested_signature_ids)
    required_columns = {
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
    }
    import pyarrow as pa

    pc: Any = __import__("pyarrow.compute").compute

    value_set = arrow_string_array(requested_signature_ids)
    signatures: dict[str, _ArrowIncrementalSignatureInfo] = {}
    with arrow_dataset.open_file("signatures") as source_file, pa.PythonFile(source_file, mode="r") as source:
        reader = pa.ipc.open_file(source)
        missing_columns = sorted(required_columns.difference(reader.schema.names))
        if missing_columns:
            raise ValueError(
                f"signatures Arrow is missing required columns for incremental metadata: {missing_columns}"
            )
        selected_columns = required_columns.union({"author_orcid"}.intersection(reader.schema.names))
        required_column_names = sorted(selected_columns)
        signature_id_column_index = reader.schema.get_field_index("signature_id")
        for batch_index in _signature_batch_indices_for_request(arrow_dataset, requested_signature_ids):
            batch = reader.get_batch(batch_index)
            signature_id_column = batch.column(signature_id_column_index)
            mask = pc.is_in(signature_id_column, value_set=value_set)
            if not bool(pc.any(mask).as_py()):
                continue
            filtered = pa.Table.from_batches([batch], schema=reader.schema).select(required_column_names).filter(mask)
            for row in filtered.to_pylist():
                signature_id_value = row.get("signature_id")
                if signature_id_value is None:
                    raise ValueError("signatures Arrow cannot contain null signature_id values")
                signature_id = str(signature_id_value)
                if signature_id not in requested_signature_id_set:
                    continue
                if signature_id in signatures:
                    raise ValueError(f"signatures Arrow contains duplicate signature_id: {signature_id!r}")
                first = _optional_arrow_string(row.get("author_first"))
                middle = _optional_arrow_string(row.get("author_middle"))
                last = _optional_arrow_string(row.get("author_last"))
                first_normalized = canonicalize_name_parts(first, middle, last).first
                signatures[signature_id] = _ArrowIncrementalSignatureInfo(
                    paper_id=_optional_arrow_string(row.get("paper_id")),
                    author_info_first=first,
                    author_info_last=last,
                    author_info_first_normalized=first_normalized or None,
                    author_info_orcid=_optional_arrow_string(row.get("author_orcid")),
                    author_info_middle=middle,
                )
    missing_signature_ids = [signature_id for signature_id in requested_signature_ids if signature_id not in signatures]
    if missing_signature_ids:
        raise ValueError(f"Arrow signatures are missing incremental signature ids: {missing_signature_ids[:10]}")
    return signatures


_NO_ORCID_SIGNATURE_INFO = _ArrowIncrementalSignatureInfo(
    paper_id=None,
    author_info_first=None,
    author_info_last=None,
    author_info_first_normalized=None,
    author_info_orcid=None,
)


def _load_arrow_incremental_orcid_signature_info(
    arrow_dataset: Any,
    signature_ids: Iterable[Any],
    *,
    num_threads: int | None = None,
) -> dict[str, _ArrowIncrementalSignatureInfo]:
    """Load ORCID-only seed metadata for promoted incremental budget sizing.

    Missing seed signatures are skipped to match ANDData's missing-signature
    ORCID behavior. Entries follow Arrow row order.
    """

    requested_signature_ids = tuple(dict.fromkeys(str(signature_id) for signature_id in signature_ids))
    if not requested_signature_ids:
        return {}
    # Rust decodes only the id and ORCID columns of the indexed batches; a
    # pyarrow loop over them took 2 s per large incremental request.
    rows = arrow_dataset.native.signature_orcids(
        list(requested_signature_ids),
        None if num_threads is None else resolve_n_jobs(num_threads),
    )
    if rows is None:
        return {}
    found_signature_ids, orcid_positions, orcids = rows
    # Most seeds carry no ORCID; they share one record, which is never mutated.
    # Replacing a value keeps its key's place, so entries stay in row order.
    info = dict.fromkeys(found_signature_ids, _NO_ORCID_SIGNATURE_INFO)
    for position, orcid in zip(orcid_positions, orcids, strict=True):
        info[found_signature_ids[position]] = _ArrowIncrementalSignatureInfo(
            paper_id=None,
            author_info_first=None,
            author_info_last=None,
            author_info_first_normalized=None,
            author_info_orcid=orcid,
        )
    return info


def _cluster_seeds_require_inverse(
    cluster_seeds_require: Mapping[Any, Any],
) -> dict[int | str, list[str]]:
    inverse: dict[int | str, list[str]] = defaultdict(list)
    for signature_id, cluster_num in cluster_seeds_require.items():
        inverse[str(cluster_num)].append(str(signature_id))
    return inverse


def _dataset_altered_cluster_signatures(
    dataset: Any,
) -> list[str]:
    return _altered_cluster_signatures_from_values(getattr(dataset, "altered_cluster_signatures", None))


def _altered_cluster_signatures_from_values(
    values: Sequence[Any] | None,
) -> list[str]:
    return [] if values is None else [str(value) for value in values]


def _required_incremental_linker_artifact(clusterer: Any) -> Any:
    """Return the linker artifact attached by production-bundle loading."""

    artifact = getattr(clusterer, "incremental_linker_artifact", None)
    if artifact is None:
        raise FileNotFoundError(
            "Promoted incremental prediction requires an attached incremental linker artifact. "
            "Load a complete production bundle before calling the direct Arrow API."
        )
    return artifact


def _build_incremental_seed_setup(
    self: Clusterer,
    dataset: ANDData,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None = None,
    arrow_dataset: ArrowDataset | None = None,
    cluster_seed_disallows: set[tuple[str, str]] | None = None,
    prediction_state: PredictionState | None = None,
) -> _IncrementalSeedSetup:
    seed_setup_start = time.perf_counter()
    altered_presplit_predict_seconds = 0.0
    altered_presplit_block_count = 0
    altered_presplit_signature_count = 0
    altered_presplit_cache_hit_count = 0
    altered_presplit_cache_miss_count = 0
    altered_presplit_orcid_skip_count = 0
    altered_cluster_count = 0
    recluster_map: dict[int | str, int | str] = {}
    # Values are normalized into the owned mapping below; the source is
    # read-only here, so cloning it first only doubles request memory.
    source_cluster_seeds_require = (
        (getattr(dataset, "cluster_seeds_require", {}) or {})
        if prediction_state is None
        else prediction_state.cluster_seeds_require
    )
    altered_cluster_signatures = (
        _dataset_altered_cluster_signatures(dataset)
        if prediction_state is None
        else prediction_state.altered_cluster_signatures
    )
    cluster_seeds_require: dict[str, int | str]
    if (
        isinstance(dataset, _DirectArrowIncrementalDataset)
        and not altered_cluster_signatures
        and isinstance(source_cluster_seeds_require, dict)
    ):
        # The private direct-Arrow boundary already owns and normalized
        # this dict. With no altered-profile rewrite, seed setup is
        # read-only and can reuse it without another full request copy.
        cluster_seeds_require = cast(dict[str, int | str], source_cluster_seeds_require)
    else:
        cluster_seeds_require = {
            str(signature_id): str(cluster_num) for signature_id, cluster_num in source_cluster_seeds_require.items()
        }
    cluster_seeds_require_inverse = _cluster_seeds_require_inverse(cluster_seeds_require)

    request_cluster_seed_disallows = (
        (
            _cluster_seed_disallows_for_request(dataset)
            if prediction_state is None
            else prediction_state.cluster_seeds_disallow
        )
        if cluster_seed_disallows is None
        else set(normalize_cluster_seed_disallow_pairs(cluster_seed_disallows))
    )
    request_cluster_seed_disallows = effective_dataset_disallows(
        request_cluster_seed_disallows,
        partial_supervision,
        use_default_constraints=self.use_default_constraints_as_supervision,
    )
    # Both incremental routes restore every original seed group. Validate
    # that contract before scoring or temporary splitting can hide a conflict.
    validate_incremental_seed_disallows(cluster_seeds_require, request_cluster_seed_disallows, partial_supervision)
    # Split altered claimed profiles once so incremental assignment can map back to original cluster IDs.
    # Claimed profiles from production can be "unnatural" with respect to S2AND constraints;
    # this pre-split step aligns them to natural-looking clusters before adding new signatures.
    if len(altered_cluster_signatures) > 0:
        missing_altered_seeds = sorted(
            str(signature_id)
            for signature_id in altered_cluster_signatures
            if str(signature_id) not in cluster_seeds_require
        )
        if missing_altered_seeds:
            raise ValueError(
                "altered_cluster_signatures must all be present in cluster_seeds_require; "
                f"missing={missing_altered_seeds[:10]}"
            )
        logger.info("Dealing with altered cluster signatures")
        altered_cluster_nums: set[int | str] = set()
        for altered_signature_id in altered_cluster_signatures:
            altered_signature_key = str(altered_signature_id)
            altered_cluster_num: int | str = cluster_seeds_require[altered_signature_key]
            altered_cluster_nums.add(altered_cluster_num)
        sorted_altered_cluster_nums = sorted(altered_cluster_nums, key=str)
        altered_cluster_count = len(sorted_altered_cluster_nums)
        model_cache_fingerprint = _model_presplit_cache_fingerprint(self)
        name_tuples = getattr(dataset, "name_tuples", None)
        name_tuples_key: Any | None = None
        reclustered_by_cluster_num: dict[int | str, list[list[str]]] = defaultdict(list)
        presplit_jobs: list[_AlteredPresplitJob] = []
        for altered_index, altered_cluster_num in enumerate(sorted_altered_cluster_nums):
            signature_ids_for_cluster_num = cluster_seeds_require_inverse.get(str(altered_cluster_num), [])
            if len(signature_ids_for_cluster_num) <= 1:
                continue
            altered_presplit_block_count += 1
            altered_presplit_signature_count += len(signature_ids_for_cluster_num)
            if _can_skip_orcid_homogeneous_altered_presplit(
                self,
                dataset,
                signature_ids_for_cluster_num,
                partial_supervision,
                cluster_seed_disallows=request_cluster_seed_disallows,
            ):
                altered_presplit_orcid_skip_count += 1
                continue

            # During this pre-split, do not apply incoming cluster seeds as constraints.
            # At this stage we are splitting claimed profiles to match S2AND predictions,
            # so claimed-profile seeds should not bias the split.
            effective_partial_supervision = _partial_supervision_with_cluster_seed_disallows(
                signature_ids_for_cluster_num,
                dataset,
                partial_supervision,
                cluster_seed_disallows=request_cluster_seed_disallows,
            )
            if arrow_dataset is not None:
                if name_tuples_key is None:
                    name_tuples_key = _cacheable_value(name_tuples)
                cache_key = _altered_presplit_cache_key(
                    mode="arrow",
                    altered_cluster_num=altered_cluster_num,
                    signature_ids=signature_ids_for_cluster_num,
                    effective_partial_supervision=effective_partial_supervision,
                    arrow_generation_id=arrow_dataset.generation_id,
                    name_tuples_key=name_tuples_key,
                    model_fingerprint=model_cache_fingerprint,
                )
            else:
                cache_key = None
            if cache_key is not None:
                cached_clusters = _get_altered_presplit_cache_entry(self, cache_key)
                if cached_clusters is not None:
                    altered_presplit_cache_hit_count += 1
                    reclustered_by_cluster_num[altered_cluster_num].extend(
                        [list(cluster) for cluster in cached_clusters]
                    )
                    continue
                altered_presplit_cache_miss_count += 1
            presplit_jobs.append(
                _AlteredPresplitJob(
                    block_key=f"altered_profile_{altered_index}",
                    altered_cluster_num=altered_cluster_num,
                    signature_ids=list(signature_ids_for_cluster_num),
                    partial_supervision=effective_partial_supervision,
                    cache_key=cache_key,
                )
            )

        if presplit_jobs and arrow_dataset is not None:
            presplit_block_dict = {job.block_key: job.signature_ids for job in presplit_jobs}
            signature_to_cluster_num = {
                str(signature_id): job.altered_cluster_num
                for job in presplit_jobs
                for signature_id in job.signature_ids
            }
            presplit_partial_supervision: dict[tuple[str, str], int | float] = {}
            for job in presplit_jobs:
                presplit_partial_supervision.update(job.partial_supervision)
            presplit_start = time.perf_counter()
            reclustered_output, _ = self.predict_from_arrow(
                presplit_block_dict,
                arrow_dataset,
                incremental_dont_use_cluster_seeds=True,
                partial_supervision=presplit_partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                load_name_counts=clusterer_uses_name_count_features(self),
                name_tuples=name_tuples,
                cluster_seeds_disallow=request_cluster_seed_disallows,
                altered_cluster_signatures=[],
            )
            altered_presplit_predict_seconds += time.perf_counter() - presplit_start
            for new_cluster_of_signatures in reclustered_output.values():
                if len(new_cluster_of_signatures) == 0:
                    continue
                output_cluster_nums = {
                    signature_to_cluster_num[str(signature_id)] for signature_id in new_cluster_of_signatures
                }
                if len(output_cluster_nums) != 1:
                    raise ValueError(
                        "Altered profile pre-split produced a cluster spanning claimed profiles: "
                        f"{sorted(str(cluster_num) for cluster_num in output_cluster_nums)}"
                    )
                altered_cluster_num = next(iter(output_cluster_nums))
                reclustered_by_cluster_num[altered_cluster_num].append(list(new_cluster_of_signatures))
            for job in presplit_jobs:
                if job.cache_key is not None:
                    _put_altered_presplit_cache_entry(
                        self,
                        job.cache_key,
                        reclustered_by_cluster_num.get(job.altered_cluster_num, []),
                    )
        elif presplit_jobs:
            for job in presplit_jobs:
                presplit_start = time.perf_counter()
                reclustered_output, _ = self.predict_helper(
                    {"block": job.signature_ids},
                    dataset,
                    incremental_dont_use_cluster_seeds=True,
                    partial_supervision=job.partial_supervision,
                    runtime_context=runtime_context,
                    total_ram_bytes=total_ram_bytes,
                    prediction_state=prediction_state,
                )
                altered_presplit_predict_seconds += time.perf_counter() - presplit_start
                reclustered_by_cluster_num[job.altered_cluster_num].extend(
                    [list(new_cluster) for new_cluster in reclustered_output.values()]
                )

        # Split IDs must not reuse an existing profile ID such as "p_0".
        used_component_ids = set(cluster_seeds_require_inverse)
        for altered_cluster_num in sorted_altered_cluster_nums:
            new_clusters = reclustered_by_cluster_num.get(altered_cluster_num, [])
            if len(new_clusters) <= 1:
                continue
            for i, new_cluster_of_signatures in enumerate(new_clusters):
                new_cluster_num = f"{altered_cluster_num}_{i}"
                collision_index = 1
                while new_cluster_num in used_component_ids:
                    new_cluster_num = f"{altered_cluster_num}_{i}|collision={collision_index:04d}"
                    collision_index += 1
                used_component_ids.add(new_cluster_num)
                recluster_map[new_cluster_num] = altered_cluster_num
                for reclustered_signature_id in new_cluster_of_signatures:
                    cluster_seeds_require[reclustered_signature_id] = new_cluster_num

    split_cluster_seeds_require_inverse = (
        _cluster_seeds_require_inverse(cluster_seeds_require) if recluster_map else cluster_seeds_require_inverse
    )
    if prediction_state is None:
        prediction_state = PredictionState()
    prediction_state.telemetry["incremental_seed_setup"] = {
        "seed_setup_seconds": float(time.perf_counter() - seed_setup_start),
        "seed_setup_seed_signature_count": int(len(cluster_seeds_require)),
        "seed_setup_component_count": int(len(split_cluster_seeds_require_inverse)),
        "seed_setup_altered_signature_count": int(len(altered_cluster_signatures)),
        "seed_setup_altered_presplit_block_count": int(altered_presplit_block_count),
        "seed_setup_altered_presplit_signature_count": int(altered_presplit_signature_count),
        "seed_setup_altered_presplit_predict_seconds": float(altered_presplit_predict_seconds),
        "seed_setup_altered_presplit_cache_hit_count": int(altered_presplit_cache_hit_count),
        "seed_setup_altered_presplit_cache_miss_count": int(altered_presplit_cache_miss_count),
        "seed_setup_altered_presplit_orcid_skip_count": int(altered_presplit_orcid_skip_count),
        "seed_setup_recluster_map_entry_count": int(len(recluster_map)),
        "seed_setup_altered_cluster_count": int(altered_cluster_count),
    }
    return _IncrementalSeedSetup(
        required_assignments=cluster_seeds_require,
        recluster_map=recluster_map,
        original_members=cluster_seeds_require_inverse,
        split_members=split_cluster_seeds_require_inverse,
    )


def _make_residual_clusterer(
    self: Clusterer,
    dataset: ANDData,
    *,
    partial_supervision: dict[tuple[str, str], int | float],
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    arrow_dataset: ArrowDataset | None = None,
    cluster_seed_disallows: set[tuple[str, str]] | None = None,
    prediction_state: PredictionState | None = None,
) -> ResidualClusterer:
    """Bind backend execution and request constraints for the completion phase."""
    request_disallows = (
        (
            _cluster_seed_disallows_for_request(dataset)
            if prediction_state is None
            else prediction_state.cluster_seeds_disallow
        )
        if cluster_seed_disallows is None
        else set(normalize_cluster_seed_disallow_pairs(cluster_seed_disallows))
    )

    def cluster_residuals(signature_ids: list[str]) -> dict[str, list[str]]:
        if arrow_dataset is None:
            clusters, _ = self.predict_helper(
                {"block": signature_ids},
                dataset,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                prediction_state=prediction_state,
            )
        else:
            residual_supervision = _partial_supervision_with_cluster_seed_disallows(
                signature_ids, dataset, partial_supervision, cluster_seed_disallows=request_disallows
            )
            logger.info(
                "Running incremental residual Phase B through Arrow/Rust: residual_signatures=%d",
                len(signature_ids),
            )
            clusters, _ = self.predict_from_arrow(
                {"block": signature_ids},
                arrow_dataset,
                partial_supervision=residual_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                load_name_counts=clusterer_uses_name_count_features(self),
                name_tuples=getattr(dataset, "name_tuples", None),
                cluster_seeds_disallow=request_disallows,
            )
        return clusters

    return cluster_residuals


@_gc_paused
def predict_incremental_from_arrow(
    self: Clusterer,
    block_signatures: list[str],
    arrow_dataset: ArrowDataset,
    *,
    prevent_new_incompatibilities: bool = True,
    batching_threshold: int | None = None,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    altered_cluster_signatures: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Predict incremental clustering through one open Arrow dataset.

    Seed assignments and request-local disallows are explicit arguments;
    immutable base artifacts come only from ``arrow_dataset``.
    """

    partial_supervision_is_distance_override = _ARROW_BULK_DISTANCE_OVERRIDES.get()
    name_check_exempt_signature_ids = _ARROW_ATTACH_NAME_CHECK_EXEMPT.get()
    partial_supervision = normalize_partial_supervision(partial_supervision, stringify_ids=True)
    if runtime_context is None:
        runtime_context = build_runtime_context(
            "cluster_predict_incremental_from_arrow",
            backend="rust",
        )
    elif not stage_uses_rust(runtime_context):
        raise ValueError("Clusterer.predict_incremental_from_arrow requires a Rust runtime context")
    if not isinstance(arrow_dataset, ArrowDataset):
        raise TypeError("Clusterer.predict_incremental_from_arrow requires an open ArrowDataset")
    seed_signature_to_cluster = _normalize_cluster_seeds_require(cluster_seeds_require or {})
    if not seed_signature_to_cluster:
        raise ValueError("Clusterer.predict_incremental_from_arrow requires nonempty cluster_seeds_require")
    require_name_counts_index = clusterer_uses_name_count_features(self)
    with arrow_dataset.use(
        require_specter=clusterer_uses_embedding_features(self),
        require_name_counts_index=require_name_counts_index,
    ) as arrow_lease:
        _require_arrow_name_counts_index_for_clusterer(
            self,
            arrow_dataset,
            context="Arrow incremental prediction",
        )
        resolved_name_tuples = _name_tuples_for_incremental_rules(name_tuples)
        resolved_cluster_seed_disallows = (
            normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow)
            if cluster_seeds_disallow is not None
            else set()
        )
        resolved_cluster_seed_disallows = effective_dataset_disallows(
            resolved_cluster_seed_disallows,
            partial_supervision,
            use_default_constraints=self.use_default_constraints_as_supervision,
        )
        resolved_altered_cluster_signatures = (
            [str(signature_id) for signature_id in altered_cluster_signatures]
            if altered_cluster_signatures is not None
            else None
        )
        block_signature_ids = [str(signature_id) for signature_id in block_signatures]
        unassigned_signature_ids = [
            signature_id for signature_id in block_signature_ids if signature_id not in seed_signature_to_cluster
        ]
        # The planner over the original seeds builds while the rest of the
        # setup runs; the call below owns it, and a failure before then frees it.
        seed_planner = start_background_seed_planner(
            self,
            block_signature_ids,
            arrow_dataset=arrow_dataset,
            artifact=_required_incremental_linker_artifact(self),
            cluster_seeds_require=seed_signature_to_cluster,
            altered_cluster_signatures=resolved_altered_cluster_signatures,
            query_signature_ids=unassigned_signature_ids,
            attachment_state=_ARROW_ATTACH_BLOCK_STATE.get(),
        )
        try:
            altered_signature_ids = _altered_cluster_signatures_from_values(altered_cluster_signatures)
            altered_cluster_ids = {
                str(seed_signature_to_cluster[signature_id])
                for signature_id in altered_signature_ids
                if signature_id in seed_signature_to_cluster
            }
            altered_seed_signature_ids = (
                [
                    signature_id
                    for signature_id, cluster_id in seed_signature_to_cluster.items()
                    if str(cluster_id) in altered_cluster_ids
                ]
                if altered_cluster_ids
                else []
            )
            metadata_signature_ids = (*unassigned_signature_ids, *altered_seed_signature_ids)
            attachment_state = _ARROW_ATTACH_BLOCK_STATE.get()
            shared_setup = None if attachment_state is None else attachment_state.shared_setup
            signatures = (
                _load_arrow_incremental_signature_info(arrow_lease, metadata_signature_ids)
                if shared_setup is None
                else shared_setup.signature_info(arrow_lease, metadata_signature_ids)
            )
            if not bool(getattr(self, "suppress_orcid", False)) and any(
                (signature_info := signatures.get(signature_id)) is not None
                and signature_info.author_info_orcid is not None
                and signature_info.author_info_orcid.strip()
                for signature_id in unassigned_signature_ids
            ):
                seed_orcid_signature_ids = [
                    signature_id for signature_id in seed_signature_to_cluster if signature_id not in signatures
                ]
                attachment_state = _ARROW_ATTACH_BLOCK_STATE.get()
                signatures.update(
                    attachment_state.orcid_signature_info(
                        arrow_lease, seed_orcid_signature_ids, num_threads=self.n_jobs
                    )
                    if attachment_state is not None
                    else _load_arrow_incremental_orcid_signature_info(
                        arrow_lease,
                        seed_orcid_signature_ids,
                        num_threads=self.n_jobs,
                    )
                )
            request_dataset = _DirectArrowIncrementalDataset(
                name_tuples=resolved_name_tuples,
                cluster_seeds_require=seed_signature_to_cluster,
                cluster_seeds_disallow=resolved_cluster_seed_disallows,
                altered_cluster_signatures=resolved_altered_cluster_signatures,
                max_seed_cluster_id=_seed_cluster_count_from_seed_map(seed_signature_to_cluster),
                signatures=signatures,
                partial_supervision_is_distance_override=partial_supervision_is_distance_override,
                name_check_exempt_signature_ids=name_check_exempt_signature_ids,
            )
            logger.info("Running direct Arrow promoted incremental prediction")
            return predict_incremental_promoted_linker_from_arrow(
                self,
                block_signature_ids,
                request_dataset,
                arrow_dataset=arrow_dataset,
                artifact=_required_incremental_linker_artifact(self),
                cluster_residuals=self._make_residual_clusterer(
                    cast(ANDData, request_dataset),
                    partial_supervision=partial_supervision or {},
                    runtime_context=runtime_context,
                    total_ram_bytes=memory_budget.resolve_total_ram_bytes(total_ram_bytes)[0],
                    arrow_dataset=arrow_dataset,
                ),
                prevent_new_incompatibilities=prevent_new_incompatibilities,
                partial_supervision=partial_supervision,
                runtime_context=runtime_context,
                total_ram_bytes=total_ram_bytes,
                batching_threshold=batching_threshold,
                attachment_state=_ARROW_ATTACH_BLOCK_STATE.get(),
                query_signature_ids=unassigned_signature_ids,
                seed_planner=seed_planner,
            )
        finally:
            if seed_planner is not None:
                seed_planner.release()
